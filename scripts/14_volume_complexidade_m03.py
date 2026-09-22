"""Gate 14 - unica celula faltante do experimento 2x2 (volume x complexidade): 1,2M + num_leaves=15.

As outras tres celulas (600k/31, 1.2M/31, 600k/15) ja foram treinadas nos Gates 10 e 13, com
amostragem, features, seed e protocolo identicos - reaproveitadas sem retreinar.
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
PONTOS_POR_MES = 8000  # 150 x 8000 = 1.200.000

B00_RMSE = {"A": 1.8938, "B": 1.9003}

# celulas ja conhecidas dos Gates 10 e 13 (mesma amostragem/seed/protocolo)
CELULAS_CONHECIDAS = {
    (600_000, 31): {"A": 1.9190, "B": 1.8941},
    (1_200_000, 31): {"A": 1.9169, "B": 1.8960},
    (600_000, 15): {"A": 1.9048, "B": 1.8874},
}


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


def monta_treino(cutoff, tp_alvo, atmosfericas, clim, lat, lon, tempos, seed=SEED):
    elegiveis = np.where(tempos < np.datetime64(cutoff))[0]
    rng = np.random.default_rng(seed)
    n_meses = min(MESES_AMOSTRADOS, len(elegiveis))
    meses_idx = np.sort(rng.choice(elegiveis, size=n_meses, replace=False))

    blocos = {v: [] for v in FEATURES}
    blocos["y_residuo"] = []

    for t_idx in meses_idx:
        lat_idx = rng.integers(0, len(lat), size=PONTOS_POR_MES)
        lon_idx = rng.integers(0, len(lon), size=PONTOS_POR_MES)

        mes_origem = pd.Timestamp(tempos[t_idx]).month
        mes_target = mes_alvo(t_idx, tempos)

        y_alvo = tp_alvo.isel(time=t_idx).values[lat_idx, lon_idx].astype(np.float32)
        clim_pontos = clim[mes_target - 1][lat_idx, lon_idx].astype(np.float32)

        blocos["lat"].append(lat[lat_idx].astype(np.float32))
        blocos["lon"].append(lon[lon_idx].astype(np.float32))
        blocos["month_sin"].append(np.full(PONTOS_POR_MES, np.sin(2 * np.pi * mes_origem / 12), dtype=np.float32))
        blocos["month_cos"].append(np.full(PONTOS_POR_MES, np.cos(2 * np.pi * mes_origem / 12), dtype=np.float32))
        blocos["tp_climatologia"].append(clim_pontos)
        blocos["y_residuo"].append((y_alvo - clim_pontos).astype(np.float32))

        for v in VARS_ATMOSFERICAS:
            slice_v = atmosfericas[v].isel(time=t_idx).values
            blocos[v].append(slice_v[lat_idx, lon_idx].astype(np.float32))

    X = pd.DataFrame({v: np.concatenate(blocos[v]) for v in FEATURES})
    y_res = np.concatenate(blocos["y_residuo"])
    return X, y_res, n_meses, len(elegiveis)


def valida_backtest(modelo, origem, ultimo_alvo_disponivel, tp_alvo, atmosfericas, clim, lat, lon, tempos):
    origem = pd.Timestamp(origem)
    origens_validacao = pd.date_range(origem, periods=24, freq="MS")
    origens_validacao = [o for o in origens_validacao
                          if (o + pd.DateOffset(months=1)) <= pd.Timestamp(ultimo_alvo_disponivel)]

    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    lat_flat = lat_grid.ravel().astype(np.float32)
    lon_flat = lon_grid.ravel().astype(np.float32)

    linhas = []
    stats_pred = {"n": 0, "n_nan": 0, "n_neg": 0, "soma": 0.0, "min": np.inf, "max": -np.inf}

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

        stats_pred["n"] += len(tp_predito)
        stats_pred["n_nan"] += int(np.isnan(tp_predito).sum())
        stats_pred["n_neg"] += int((tp_predito < 0).sum())
        stats_pred["soma"] += float(np.nansum(tp_predito))
        stats_pred["min"] = min(stats_pred["min"], float(np.nanmin(tp_predito)))
        stats_pred["max"] = max(stats_pred["max"], float(np.nanmax(tp_predito)))

        erro2 = (tp_predito - y_real) ** 2
        linhas.append({"horizonte": horizonte, "rmse": float(np.sqrt(np.mean(erro2))), "n": len(y_real)})

        del X_mes, tp_predito, erro2, y_real
        gc.collect()

    df = pd.DataFrame(linhas)
    rmse_global = float(np.sqrt((df["rmse"] ** 2 * df["n"]).sum() / df["n"].sum()))
    stats_pred["media"] = stats_pred["soma"] / stats_pred["n"]
    return df, rmse_global, stats_pred


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    tp, tp_alvo, atmosfericas, lat, lon, tempos = abre_variaveis()

    print(f"features: {len(FEATURES)} | amostras: 1,200,000 (150 meses x {PONTOS_POR_MES} pontos/mes)")
    print(f"X estimado: {1_200_000 * len(FEATURES) * 4 / 1e6:.1f} MB | "
          f"y estimado: {1_200_000 * 4 / 1e6:.1f} MB")
    print("estrategia: cada mes lido uma unica vez do NetCDF; um backtest por vez, memoria liberada entre eles")

    backtests = {"A": ("2020-12-01", "2022-12-01"), "B": ("2021-12-01", "2022-12-01")}
    resultados_horizonte = []
    resultados_globais = {}

    for nome_bt, (origem, ultimo_alvo) in backtests.items():
        print(f"\n=== Backtest {nome_bt} (origem {origem}) ===")
        mem_disp = memoria_disponivel_mb()
        print(f"memoria disponivel antes do backtest: {mem_disp:.0f} MB")

        clim = climatologia_ate(tp, origem)
        X, y_res, n_meses, n_elegiveis = monta_treino(origem, tp_alvo, atmosfericas, clim, lat, lon, tempos)
        print(f"treino: {len(X):,} linhas ({n_meses}/{n_elegiveis} meses elegiveis)")

        modelo = LGBMRegressor(
            n_estimators=300,
            num_leaves=15,
            learning_rate=0.05,
            random_state=SEED,
            verbosity=-1,
        )
        modelo.fit(X, y_res)
        del X, y_res
        gc.collect()

        df_horizontes, rmse_global, stats_pred = valida_backtest(
            modelo, origem, ultimo_alvo, tp_alvo, atmosfericas, clim, lat, lon, tempos
        )
        df_horizontes["backtest"] = nome_bt
        resultados_horizonte.append(df_horizontes)
        resultados_globais[nome_bt] = rmse_global

        print(f"RMSE global (1.2M, num_leaves=15): {rmse_global:.4f}")
        print("sanity check:", {k: round(v, 4) if isinstance(v, float) else v for k, v in stats_pred.items()},
              "pct_negativo:", round(100 * stats_pred["n_neg"] / stats_pred["n"], 3))

        del modelo, clim
        gc.collect()

    todos_horizontes = pd.concat(resultados_horizonte, ignore_index=True)
    todos_horizontes.to_csv(RESULTS_DIR / "m03_1_2M_leaves15_por_horizonte.csv", index=False)

    def faixas(df):
        return {
            "h1_6": df[df.horizonte.between(1, 6)]["rmse"].mean(),
            "h7_12": df[df.horizonte.between(7, 12)]["rmse"].mean(),
            "h13_18": df[df.horizonte.between(13, 18)]["rmse"].mean(),
            "h19_24": df[df.horizonte.between(19, 24)]["rmse"].mean(),
        }

    # monta a tabela 2x2 completa (3 celulas conhecidas + a nova)
    linhas_2x2 = []
    for (volume, leaves), rmses in CELULAS_CONHECIDAS.items():
        linhas_2x2.append({"volume": volume, "num_leaves": leaves, "rmse_A": rmses["A"], "rmse_B": rmses["B"]})
    linhas_2x2.append({"volume": 1_200_000, "num_leaves": 15,
                        "rmse_A": resultados_globais["A"], "rmse_B": resultados_globais["B"]})
    df_2x2 = pd.DataFrame(linhas_2x2)
    df_2x2["media"] = (df_2x2["rmse_A"] + df_2x2["rmse_B"]) / 2
    df_2x2["pior_caso"] = df_2x2[["rmse_A", "rmse_B"]].max(axis=1)
    df_2x2.to_csv(RESULTS_DIR / "m03_volume_complexidade.csv", index=False)

    print("\n--- tabela 2x2 completa ---")
    print(df_2x2.to_string(index=False))

    df_a_nova = todos_horizontes[todos_horizontes.backtest == "A"]
    print("\n--- 1.2M/15 por faixa de horizonte (Backtest A) ---")
    print(faixas(df_a_nova))
    print("\n--- 1.2M/15 por horizonte individual (Backtest A) ---")
    print(df_a_nova[["horizonte", "rmse"]].to_string(index=False))
