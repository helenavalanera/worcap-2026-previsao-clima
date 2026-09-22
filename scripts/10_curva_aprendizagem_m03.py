"""Gate 10 - curva de aprendizagem do M03: mesma formulacao por residuo, variando so o volume
de treino (300k / 600k / 1,2M), com amostras aninhadas (300k esta contida em 600k, que esta
contida em 1,2M) para isolar o efeito do volume do efeito do sorteio.
"""

import gc
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from lightgbm import LGBMRegressor

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

VARS_ATMOSFERICAS = [
    "t2", "cloud_cover", "shum_850", "surface_pressure",
    "u_850", "v_850", "temperature_850", "rel_hum_850", "geopotential_850",
]
FEATURES = VARS_ATMOSFERICAS + ["lat", "lon", "month_sin", "month_cos", "tp_climatologia"]

SEED = 42
MESES_AMOSTRADOS = 150
PONTOS_POR_MES_MAX = 8000
TAMANHOS_PONTOS_POR_MES = [2000, 4000, 8000]  # 300k, 600k, 1.2M (150 meses x pontos)

B00_RMSE = {"A": 1.8938, "B": 1.9003}


def memoria_disponivel_mb():
    saida = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    pagina = 16384
    valores = {}
    for linha in saida.splitlines():
        if ":" in linha:
            chave, val = linha.split(":")
            val = val.strip().rstrip(".")
            if val.isdigit():
                valores[chave.strip()] = int(val)
    livre = valores.get("Pages free", 0) + valores.get("Pages inactive", 0) + valores.get("Pages speculative", 0)
    return livre * pagina / 1e6


def abre_variaveis():
    tp = xr.open_dataset(RAW_DIR / "treino_tp.nc")["tp"]
    tp_alvo = xr.open_dataset(RAW_DIR / "treino_tp_alvo.nc")["tp_alvo"]
    atmosfericas = {v: xr.open_dataset(RAW_DIR / f"treino_{v}.nc")[v] for v in VARS_ATMOSFERICAS}
    lat = tp_alvo["lat"].values
    lon = tp_alvo["lon"].values
    tempos = tp_alvo["time"].values
    return tp, tp_alvo, atmosfericas, lat, lon, tempos


def climatologia_ate(tp, cutoff):
    historico = tp.sel(time=slice(None, cutoff))
    return historico.groupby("time.month").mean("time").values


def mes_alvo(t_idx, tempos):
    return (pd.Timestamp(tempos[t_idx]).month % 12) + 1


def constroi_cache(cutoff, tp_alvo, atmosfericas, clim, lat, lon, tempos, seed=SEED):
    """Le cada mes uma unica vez, no maior tamanho de pool (8000 pontos/mes), e guarda em
    listas por variavel. Tamanhos menores sao obtidos por prefixo (fatia [:N]) desse pool,
    garantindo o aninhamento 300k subset 600k subset 1,2M."""
    elegiveis = np.where(tempos < np.datetime64(cutoff))[0]
    rng = np.random.default_rng(seed)
    n_meses = min(MESES_AMOSTRADOS, len(elegiveis))
    meses_idx = np.sort(rng.choice(elegiveis, size=n_meses, replace=False))

    cache = {v: [] for v in FEATURES}
    cache["y_residuo"] = []

    for t_idx in meses_idx:
        lat_idx = rng.integers(0, len(lat), size=PONTOS_POR_MES_MAX)
        lon_idx = rng.integers(0, len(lon), size=PONTOS_POR_MES_MAX)

        mes_origem = pd.Timestamp(tempos[t_idx]).month
        mes_target = mes_alvo(t_idx, tempos)

        y_slice = tp_alvo.isel(time=t_idx).values
        y_alvo = y_slice[lat_idx, lon_idx].astype(np.float32)
        # nos meses elegiveis (< cutoff) tp_alvo nunca e NaN; sem filtro extra necessario aqui

        clim_slice = clim[mes_target - 1]
        clim_pontos = clim_slice[lat_idx, lon_idx].astype(np.float32)

        cache["lat"].append(lat[lat_idx].astype(np.float32))
        cache["lon"].append(lon[lon_idx].astype(np.float32))
        cache["month_sin"].append(np.full(PONTOS_POR_MES_MAX, np.sin(2 * np.pi * mes_origem / 12), dtype=np.float32))
        cache["month_cos"].append(np.full(PONTOS_POR_MES_MAX, np.cos(2 * np.pi * mes_origem / 12), dtype=np.float32))
        cache["tp_climatologia"].append(clim_pontos)
        cache["y_residuo"].append((y_alvo - clim_pontos).astype(np.float32))

        for v in VARS_ATMOSFERICAS:
            slice_v = atmosfericas[v].isel(time=t_idx).values
            cache[v].append(slice_v[lat_idx, lon_idx].astype(np.float32))

    return cache, n_meses, len(elegiveis)


def monta_dataset_do_cache(cache, pontos_por_mes):
    X = pd.DataFrame({v: np.concatenate([bloco[:pontos_por_mes] for bloco in cache[v]]) for v in FEATURES})
    y_res = np.concatenate([bloco[:pontos_por_mes] for bloco in cache["y_residuo"]])
    return X, y_res


def valida_backtest(modelo, origem, ultimo_alvo_disponivel, tp_alvo, atmosfericas, clim, lat, lon, tempos):
    origem = pd.Timestamp(origem)
    origens_validacao = pd.date_range(origem, periods=24, freq="MS")
    origens_validacao = [o for o in origens_validacao
                          if (o + pd.DateOffset(months=1)) <= pd.Timestamp(ultimo_alvo_disponivel)]

    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    lat_flat = lat_grid.ravel().astype(np.float32)
    lon_flat = lon_grid.ravel().astype(np.float32)

    linhas = []
    for origem_val in origens_validacao:
        t_idx = int(np.where(tempos == np.datetime64(origem_val))[0][0])
        horizonte = (origem_val.year - origem.year) * 12 + (origem_val.month - origem.month) + 1
        mes_target = mes_alvo(t_idx, tempos)

        month_sin = np.full(lat_flat.shape, np.sin(2 * np.pi * origem_val.month / 12), dtype=np.float32)
        month_cos = np.full(lat_flat.shape, np.cos(2 * np.pi * origem_val.month / 12), dtype=np.float32)
        clim_flat = clim[mes_target - 1].ravel().astype(np.float32)

        dados = {"lat": lat_flat, "lon": lon_flat, "month_sin": month_sin, "month_cos": month_cos,
                 "tp_climatologia": clim_flat}
        for v in VARS_ATMOSFERICAS:
            dados[v] = atmosfericas[v].isel(time=t_idx).values.ravel().astype(np.float32)
        X_mes = pd.DataFrame(dados)[FEATURES]

        y_real = tp_alvo.isel(time=t_idx).values.ravel()
        tp_predito = clim_flat + modelo.predict(X_mes)

        erro2 = (tp_predito - y_real) ** 2
        linhas.append({"horizonte": horizonte, "rmse": float(np.sqrt(np.mean(erro2))), "n": len(y_real)})

        del X_mes, tp_predito, erro2, y_real
        gc.collect()

    df = pd.DataFrame(linhas)
    rmse_global = float(np.sqrt((df["rmse"] ** 2 * df["n"]).sum() / df["n"].sum()))
    return rmse_global


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    tp, tp_alvo, atmosfericas, lat, lon, tempos = abre_variaveis()

    backtests = {"A": ("2020-12-01", "2022-12-01"), "B": ("2021-12-01", "2022-12-01")}
    resultados = []

    for nome_bt, (origem, ultimo_alvo) in backtests.items():
        print(f"\n########## Backtest {nome_bt} (origem {origem}) ##########")
        clim = climatologia_ate(tp, origem)

        print("construindo cache (leitura unica por mes, pool de 8000 pontos/mes)...")
        cache, n_meses, n_elegiveis = constroi_cache(origem, tp_alvo, atmosfericas, clim, lat, lon, tempos)
        print(f"meses usados: {n_meses}/{n_elegiveis} elegiveis ({100 * n_meses / n_elegiveis:.1f}%)")

        for pontos_por_mes in TAMANHOS_PONTOS_POR_MES:
            n_linhas = n_meses * pontos_por_mes
            mem_disp = memoria_disponivel_mb()
            mem_X_estimada = n_linhas * len(FEATURES) * 4 / 1e6
            mem_y_estimada = n_linhas * 4 / 1e6

            print(f"\n--- tamanho {n_linhas:,} linhas ({pontos_por_mes} pontos/mes) ---")
            print(f"X estimado: {mem_X_estimada:.1f} MB | y estimado: {mem_y_estimada:.1f} MB | "
                  f"memoria disponivel (livre+inativa): {mem_disp:.0f} MB")
            print("estrategia: cache lido uma vez por mes do NetCDF; tamanhos menores sao fatias do mesmo pool")

            X, y_res = monta_dataset_do_cache(cache, pontos_por_mes)

            modelo = LGBMRegressor(
                n_estimators=300,
                num_leaves=31,
                learning_rate=0.05,
                random_state=SEED,
                verbosity=-1,
            )
            modelo.fit(X, y_res)
            del X, y_res
            gc.collect()

            rmse = valida_backtest(modelo, origem, ultimo_alvo, tp_alvo, atmosfericas, clim, lat, lon, tempos)
            delta_b00 = rmse - B00_RMSE[nome_bt]
            print(f"RMSE (n={n_linhas:,}): {rmse:.4f} | delta vs B00: {delta_b00:+.4f}")

            resultados.append({"backtest": nome_bt, "amostras": n_linhas, "rmse": rmse,
                                "delta_b00": delta_b00})

            del modelo
            gc.collect()

        del cache, clim
        gc.collect()

    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv(RESULTS_DIR / "m03_curva_aprendizagem.csv", index=False)

    print("\n--- resumo curva de aprendizagem ---")
    print(df_resultados.to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    for bt in ["A", "B"]:
        sub = df_resultados[df_resultados.backtest == bt].sort_values("amostras")
        plt.plot(sub.amostras, sub.rmse, marker="o", label=f"M03 - Backtest {bt}")
        plt.axhline(B00_RMSE[bt], linestyle="--", alpha=0.6, label=f"B00 - Backtest {bt}")
    plt.xlabel("amostras de treino")
    plt.ylabel("RMSE (mm/day)")
    plt.title("M03 - curva de aprendizagem")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "m03_curva_aprendizagem.png", dpi=120)
    print("\ngrafico salvo em results/m03_curva_aprendizagem.png")
