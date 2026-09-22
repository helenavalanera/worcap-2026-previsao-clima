"""Gate 07 - primeiro modelo supervisionado (M01): gradient boosting sobre variaveis atmosfericas.

Amostragem: o treino nao materializa os ~78 milhoes de pontos do historico. Para cada
backtest, sorteamos um subconjunto de meses e, dentro de cada mes, um subconjunto de
pontos da grade. Cada mes e lido isoladamente do NetCDF (uma fatia 2D por vez), entao a
memoria usada em qualquer momento e pequena, mesmo que o historico completo seja grande.

Validacao: sempre usa a grade cheia (nenhuma amostragem), mas tambem processada um mes
de cada vez, acumulando soma de erro quadratico e contagem, para nao manter todas as
previsoes na memoria ao mesmo tempo.
"""

import gc
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
FEATURES = VARS_ATMOSFERICAS + ["lat", "lon", "month_sin", "month_cos"]

SEED = 42
MESES_AMOSTRADOS = 150
PONTOS_POR_MES = 2000


def abre_variaveis():
    tp_alvo = xr.open_dataset(RAW_DIR / "treino_tp_alvo.nc")["tp_alvo"]
    atmosfericas = {v: xr.open_dataset(RAW_DIR / f"treino_{v}.nc")[v] for v in VARS_ATMOSFERICAS}
    lat = tp_alvo["lat"].values
    lon = tp_alvo["lon"].values
    tempos = tp_alvo["time"].values
    return tp_alvo, atmosfericas, lat, lon, tempos


def monta_linha(t_idx, lat_idx, lon_idx, tp_alvo, atmosfericas, lat, lon, tempos):
    """Le uma unica fatia de tempo (2D lat x lon) por variavel e extrai os pontos sorteados."""
    mes = pd.Timestamp(tempos[t_idx]).month
    month_sin = np.sin(2 * np.pi * mes / 12)
    month_cos = np.cos(2 * np.pi * mes / 12)

    y_slice = tp_alvo.isel(time=t_idx).values
    y = y_slice[lat_idx, lon_idx].astype(np.float32)
    validos = ~np.isnan(y)

    if validos.sum() == 0:
        return None, None

    dados = {
        "lat": lat[lat_idx[validos]].astype(np.float32),
        "lon": lon[lon_idx[validos]].astype(np.float32),
        "month_sin": np.full(validos.sum(), month_sin, dtype=np.float32),
        "month_cos": np.full(validos.sum(), month_cos, dtype=np.float32),
    }
    for v in VARS_ATMOSFERICAS:
        slice_v = atmosfericas[v].isel(time=t_idx).values
        dados[v] = slice_v[lat_idx[validos], lon_idx[validos]].astype(np.float32)

    return pd.DataFrame(dados)[FEATURES], y[validos]


def amostra_treino(cutoff, tp_alvo, atmosfericas, lat, lon, tempos, seed=SEED):
    """Amostra pontos de treino usando apenas meses anteriores ao cutoff (sem tocar no mes de origem)."""
    elegiveis = np.where(tempos < np.datetime64(cutoff))[0]
    rng = np.random.default_rng(seed)
    n_meses = min(MESES_AMOSTRADOS, len(elegiveis))
    meses_idx = np.sort(rng.choice(elegiveis, size=n_meses, replace=False))

    blocos_X, blocos_y = [], []
    for t_idx in meses_idx:
        lat_idx = rng.integers(0, len(lat), size=PONTOS_POR_MES)
        lon_idx = rng.integers(0, len(lon), size=PONTOS_POR_MES)
        X_mes, y_mes = monta_linha(t_idx, lat_idx, lon_idx, tp_alvo, atmosfericas, lat, lon, tempos)
        if X_mes is not None:
            blocos_X.append(X_mes)
            blocos_y.append(y_mes)

    X = pd.concat(blocos_X, ignore_index=True)
    y = np.concatenate(blocos_y)
    return X, y, n_meses, len(elegiveis)


def valida_backtest(modelo, origem, ultimo_alvo_disponivel, tp_alvo, atmosfericas, lat, lon, tempos):
    """Roda a validacao completa (grade cheia), um mes de cada vez, acumulando erro."""
    origem = pd.Timestamp(origem)
    origens_validacao = pd.date_range(origem, periods=24, freq="MS")
    origens_validacao = [o for o in origens_validacao
                          if (o + pd.DateOffset(months=1)) <= pd.Timestamp(ultimo_alvo_disponivel)]

    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    lat_flat = lat_grid.ravel().astype(np.float32)
    lon_flat = lon_grid.ravel().astype(np.float32)

    linhas_horizonte = []
    for origem_val in origens_validacao:
        t_idx = int(np.where(tempos == np.datetime64(origem_val))[0][0])
        horizonte = (origem_val.year - origem.year) * 12 + (origem_val.month - origem.month) + 1

        mes = origem_val.month
        month_sin = np.full(lat_flat.shape, np.sin(2 * np.pi * mes / 12), dtype=np.float32)
        month_cos = np.full(lat_flat.shape, np.cos(2 * np.pi * mes / 12), dtype=np.float32)

        dados = {"lat": lat_flat, "lon": lon_flat, "month_sin": month_sin, "month_cos": month_cos}
        for v in VARS_ATMOSFERICAS:
            dados[v] = atmosfericas[v].isel(time=t_idx).values.ravel().astype(np.float32)
        X_mes = pd.DataFrame(dados)[FEATURES]

        y_real = tp_alvo.isel(time=t_idx).values.ravel()
        pred = modelo.predict(X_mes)

        erro2 = (pred - y_real) ** 2
        rmse_horizonte = float(np.sqrt(np.mean(erro2)))
        linhas_horizonte.append({"horizonte": horizonte, "rmse": rmse_horizonte, "n": len(y_real)})

        del X_mes, pred, erro2, y_real
        gc.collect()

    df_horizontes = pd.DataFrame(linhas_horizonte)
    rmse_global = float(np.sqrt((df_horizontes["rmse"] ** 2 * df_horizontes["n"]).sum() / df_horizontes["n"].sum()))
    return df_horizontes, rmse_global


def roda_backtest(nome, origem, ultimo_alvo_disponivel, tp_alvo, atmosfericas, lat, lon, tempos):
    print(f"\n=== Backtest {nome} (origem {origem}) ===")

    X_train, y_train, n_meses, n_elegiveis = amostra_treino(origem, tp_alvo, atmosfericas, lat, lon, tempos)
    mem_X_mb = X_train.memory_usage(deep=True).sum() / 1e6
    mem_y_mb = y_train.nbytes / 1e6
    print(f"amostras de treino: {len(X_train):,} ({n_meses}/{n_elegiveis} meses elegiveis, "
          f"{100 * n_meses / n_elegiveis:.1f}% do periodo, {PONTOS_POR_MES} pontos/mes)")
    print(f"memoria estimada: X_train={mem_X_mb:.1f} MB | y_train={mem_y_mb:.1f} MB")

    modelo = LGBMRegressor(
        n_estimators=300,
        num_leaves=31,
        learning_rate=0.05,
        random_state=SEED,
        verbosity=-1,
    )
    modelo.fit(X_train, y_train)

    del X_train, y_train
    gc.collect()

    df_horizontes, rmse_global = valida_backtest(
        modelo, origem, ultimo_alvo_disponivel, tp_alvo, atmosfericas, lat, lon, tempos
    )
    print(f"RMSE global: {rmse_global:.4f}")

    del modelo
    gc.collect()

    return df_horizontes, rmse_global


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    tp_alvo, atmosfericas, lat, lon, tempos = abre_variaveis()

    df_a, rmse_a = roda_backtest("A", "2020-12-01", "2022-12-01", tp_alvo, atmosfericas, lat, lon, tempos)
    gc.collect()
    df_b, rmse_b = roda_backtest("B", "2021-12-01", "2022-12-01", tp_alvo, atmosfericas, lat, lon, tempos)
    gc.collect()

    df_a["backtest"] = "A"
    df_b["backtest"] = "B"
    todos = pd.concat([df_a, df_b], ignore_index=True)
    todos.to_csv(RESULTS_DIR / "m01_rmse_por_horizonte.csv", index=False)

    def faixas(df):
        return {
            "h1_6": df[df.horizonte.between(1, 6)]["rmse"].mean(),
            "h7_12": df[df.horizonte.between(7, 12)]["rmse"].mean(),
            "h13_18": df[df.horizonte.between(13, 18)]["rmse"].mean(),
            "h19_24": df[df.horizonte.between(19, 24)]["rmse"].mean(),
        }

    print("\n--- resumo M01 ---")
    print("Backtest A:", "global", round(rmse_a, 4), faixas(df_a))
    print("Backtest B:", "global", round(rmse_b, 4), faixas(df_b))
