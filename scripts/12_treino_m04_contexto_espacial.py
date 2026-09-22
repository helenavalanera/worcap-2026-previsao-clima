"""Gate 12 - M04: mesmas features do M03 + media 3x3 das 9 variaveis atmosfericas.

Vizinhanca: para cada celula, a media inclui a propria celula e ate 8 vizinhos imediatos
(N, S, L, O e diagonais). Nas bordas do dominio, a media usa somente os vizinhos que
existem de fato (sem zeros artificiais e sem "espelhar" a borda) - por isso o padding
usado para calcular a janela e NaN, e a media e feita ignorando NaN.

Amostra: reproduz exatamente a mesma sequencia de sorteio (seed, ordem, meses, pontos) do
Gate 10, para que os 600k pontos de M04 sejam identicos aos do M03-600k - controle pareado
por construcao, sem precisar retreinar o M03.
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
VARS_3X3 = [f"{v}_media_3x3" for v in VARS_ATMOSFERICAS]
FEATURES = VARS_ATMOSFERICAS + VARS_3X3 + ["lat", "lon", "month_sin", "month_cos", "tp_climatologia"]

SEED = 42
MESES_AMOSTRADOS = 150
PONTOS_POR_MES_MAX = 8000
PONTOS_POR_MES_600K = 4000  # 150 * 4000 = 600.000

B00_RMSE = {"A": 1.8938, "B": 1.9003}
M03_600K_RMSE = {"A": 1.9190, "B": 1.8941}  # resultado do Gate 10, mesma amostra


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


def media_3x3(grid):
    """Media da celula + ate 8 vizinhos, ignorando vizinhos fora do dominio (sem zero artificial)."""
    padded = np.pad(grid, 1, mode="constant", constant_values=np.nan)
    janelas = np.stack([
        padded[dy:dy + grid.shape[0], dx:dx + grid.shape[1]]
        for dy in range(3) for dx in range(3)
    ], axis=0)
    with np.errstate(invalid="ignore"):
        media = np.nanmean(janelas, axis=0)
    return media.astype(np.float32)


def audita_grade(lat, lon):
    print("=== auditoria da grade ===")
    res_lat = np.diff(lat)
    res_lon = np.diff(lon)
    print(f"resolucao lat: min={res_lat.min():.4f} max={res_lat.max():.4f} (constante? {np.allclose(res_lat, res_lat[0])})")
    print(f"resolucao lon: min={res_lon.min():.4f} max={res_lon.max():.4f} (constante? {np.allclose(res_lon, res_lon[0])})")
    print(f"lat: {lat.min()} -> {lat.max()} ({'crescente' if lat[-1] > lat[0] else 'decrescente'})")
    print(f"lon: {lon.min()} -> {lon.max()} ({'crescente' if lon[-1] > lon[0] else 'decrescente'})")

    n_lat, n_lon = len(lat), len(lon)
    cantos = 4
    bordas = 2 * (n_lat - 2) + 2 * (n_lon - 2)
    interior = n_lat * n_lon - cantos - bordas
    print(f"celulas: {cantos} cantos (4 vizinhos), {bordas} de borda (6 vizinhos), "
          f"{interior} interiores (9 vizinhos)")
    print("tratamento de borda: media so dos vizinhos existentes (padding NaN + nanmean), sem zeros artificiais")


def demonstra_equivalencia(atmosfericas, lat, lon, tempos):
    """Mostra, para uma celula real, os valores da janela 3x3 e a media resultante."""
    var = "t2"
    t_idx = 0  # 1940-01, mes de treino
    grid = atmosfericas[var].isel(time=t_idx).values
    i, j = 150, 130  # ponto interior, longe da borda
    janela = grid[i - 1:i + 2, j - 1:j + 2]
    media_calc = media_3x3(grid)[i, j]

    print(f"\n=== equivalencia treino/teste: {var}, celula lat={lat[i]:.2f} lon={lon[j]:.2f} ===")
    print("janela 3x3 (valores reais):")
    print(np.round(janela, 3))
    print(f"valor central: {grid[i, j]:.4f}")
    print(f"media 3x3 calculada pela funcao: {media_calc:.4f}")
    print(f"media manual da janela: {janela.mean():.4f}")
    print("a mesma funcao media_3x3() e aplicada identicamente a qualquer mes, seja do historico "
          "de treino ou da grade de validacao/teste - nao ha nenhuma diferenca de calculo entre eles.")


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
            cache[f"{v}_media_3x3"].append(media_3x3(slice_v)[lat_idx, lon_idx])

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
            slice_v = atmosfericas[v].isel(time=t_idx).values
            dados[v] = slice_v.ravel().astype(np.float32)
            dados[f"{v}_media_3x3"] = media_3x3(slice_v).ravel().astype(np.float32)
        X_mes = pd.DataFrame(dados)[FEATURES]

        y_real = tp_alvo.isel(time=t_idx).values.ravel()
        tp_predito = clim_flat + modelo.predict(X_mes)

        erro2 = (tp_predito - y_real) ** 2
        linhas.append({"horizonte": horizonte, "rmse": float(np.sqrt(np.mean(erro2))), "n": len(y_real)})

        del X_mes, tp_predito, erro2, y_real
        gc.collect()

    df = pd.DataFrame(linhas)
    rmse_global = float(np.sqrt((df["rmse"] ** 2 * df["n"]).sum() / df["n"].sum()))
    return df, rmse_global


def roda_backtest(nome, origem, ultimo_alvo, tp, tp_alvo, atmosfericas, lat, lon, tempos):
    print(f"\n=== Backtest {nome} (origem {origem}) ===")
    clim = climatologia_ate(tp, origem)

    cache, n_meses, n_elegiveis = constroi_cache(origem, tp_alvo, atmosfericas, clim, lat, lon, tempos)

    n_linhas = n_meses * PONTOS_POR_MES_600K
    mem_disp = memoria_disponivel_mb()
    mem_X_estimada = n_linhas * len(FEATURES) * 4 / 1e6
    print(f"features: {len(FEATURES)} ({len(VARS_ATMOSFERICAS)} atmosfericas + {len(VARS_3X3)} 3x3 + "
          f"lat/lon/month_sin/month_cos/tp_climatologia)")
    print(f"amostras: {n_linhas:,} | X estimado: {mem_X_estimada:.1f} MB | "
          f"memoria disponivel: {mem_disp:.0f} MB")

    X, y_res = monta_dataset_do_cache(cache, PONTOS_POR_MES_600K)
    del cache
    gc.collect()

    modelo = LGBMRegressor(
        n_estimators=300,
        num_leaves=31,
        learning_rate=0.05,
        random_state=SEED,
        verbosity=-1,
    )
    modelo.fit(X, y_res)
    importancias = pd.Series(modelo.feature_importances_, index=FEATURES).sort_values(ascending=False)

    del X, y_res
    gc.collect()

    df_horizontes, rmse_global = valida_backtest(
        modelo, origem, ultimo_alvo, tp_alvo, atmosfericas, clim, lat, lon, tempos
    )
    print(f"RMSE global M04: {rmse_global:.4f} | M03 (mesma amostra, Gate 10): {M03_600K_RMSE[nome]:.4f} | "
          f"B00: {B00_RMSE[nome]:.4f}")

    del modelo, clim
    gc.collect()

    return df_horizontes, rmse_global, importancias


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    tp, tp_alvo, atmosfericas, lat, lon, tempos = abre_variaveis()

    audita_grade(lat, lon)
    demonstra_equivalencia(atmosfericas, lat, lon, tempos)

    df_a, rmse_a, imp_a = roda_backtest("A", "2020-12-01", "2022-12-01", tp, tp_alvo, atmosfericas, lat, lon, tempos)
    gc.collect()
    df_b, rmse_b, imp_b = roda_backtest("B", "2021-12-01", "2022-12-01", tp, tp_alvo, atmosfericas, lat, lon, tempos)
    gc.collect()

    df_a["backtest"] = "A"
    df_b["backtest"] = "B"
    todos = pd.concat([df_a, df_b], ignore_index=True)
    todos.to_csv(RESULTS_DIR / "m04_rmse_por_horizonte.csv", index=False)

    imp = pd.DataFrame({"backtest_A": imp_a, "backtest_B": imp_b})
    imp.to_csv(RESULTS_DIR / "m04_importancia_features.csv")

    def faixas(df):
        return {
            "h1_6": df[df.horizonte.between(1, 6)]["rmse"].mean(),
            "h7_12": df[df.horizonte.between(7, 12)]["rmse"].mean(),
            "h13_18": df[df.horizonte.between(13, 18)]["rmse"].mean(),
            "h19_24": df[df.horizonte.between(19, 24)]["rmse"].mean(),
        }

    print("\n--- resumo M04 ---")
    print("Backtest A:", "global", round(rmse_a, 4), faixas(df_a))
    print("Backtest B:", "global", round(rmse_b, 4), faixas(df_b))

    print("\n--- importancia de features (Backtest A, top 10) ---")
    print(imp_a.head(10).to_string())
    print("\n--- importancia de features (Backtest B, top 10) ---")
    print(imp_b.head(10).to_string())
