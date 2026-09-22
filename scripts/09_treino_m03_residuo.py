"""Gate 09 - M03: mesmo setup do M02, mas o modelo aprende o residuo em relacao a climatologia.

y_residuo = tp_alvo - tp_climatologia
tp_predito = tp_climatologia + modelo.predict(X)

O RMSE final e sempre calculado entre tp_predito e tp_alvo (nunca so do residuo).
Climatologia segue a mesma regra do M02: calculada apenas ate a origem de cada backtest,
buscada pelo mes-alvo (M+1).
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
FEATURES = VARS_ATMOSFERICAS + ["lat", "lon", "month_sin", "month_cos", "tp_climatologia"]

SEED = 42
MESES_AMOSTRADOS = 150
PONTOS_POR_MES = 2000


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
    return historico.groupby("time.month").mean("time").values  # (12, lat, lon)


def mes_alvo(t_idx, tempos):
    return (pd.Timestamp(tempos[t_idx]).month % 12) + 1


def monta_linha(t_idx, lat_idx, lon_idx, tp_alvo, atmosfericas, clim, lat, lon, tempos):
    mes_origem = pd.Timestamp(tempos[t_idx]).month
    mes_target = mes_alvo(t_idx, tempos)
    month_sin = np.sin(2 * np.pi * mes_origem / 12)
    month_cos = np.cos(2 * np.pi * mes_origem / 12)

    y_slice = tp_alvo.isel(time=t_idx).values
    y_alvo = y_slice[lat_idx, lon_idx].astype(np.float32)
    validos = ~np.isnan(y_alvo)
    if validos.sum() == 0:
        return None, None, None

    clim_slice = clim[mes_target - 1]
    clim_pontos = clim_slice[lat_idx[validos], lon_idx[validos]].astype(np.float32)

    dados = {
        "lat": lat[lat_idx[validos]].astype(np.float32),
        "lon": lon[lon_idx[validos]].astype(np.float32),
        "month_sin": np.full(validos.sum(), month_sin, dtype=np.float32),
        "month_cos": np.full(validos.sum(), month_cos, dtype=np.float32),
        "tp_climatologia": clim_pontos,
    }
    for v in VARS_ATMOSFERICAS:
        slice_v = atmosfericas[v].isel(time=t_idx).values
        dados[v] = slice_v[lat_idx[validos], lon_idx[validos]].astype(np.float32)

    y_alvo_validos = y_alvo[validos]
    y_residuo = (y_alvo_validos - clim_pontos).astype(np.float32)

    return pd.DataFrame(dados)[FEATURES], y_residuo, y_alvo_validos


def amostra_treino(cutoff, tp_alvo, atmosfericas, clim, lat, lon, tempos, seed=SEED):
    elegiveis = np.where(tempos < np.datetime64(cutoff))[0]
    rng = np.random.default_rng(seed)
    n_meses = min(MESES_AMOSTRADOS, len(elegiveis))
    meses_idx = np.sort(rng.choice(elegiveis, size=n_meses, replace=False))

    blocos_X, blocos_yres = [], []
    for t_idx in meses_idx:
        lat_idx = rng.integers(0, len(lat), size=PONTOS_POR_MES)
        lon_idx = rng.integers(0, len(lon), size=PONTOS_POR_MES)
        X_mes, y_res_mes, _ = monta_linha(t_idx, lat_idx, lon_idx, tp_alvo, atmosfericas, clim, lat, lon, tempos)
        if X_mes is not None:
            blocos_X.append(X_mes)
            blocos_yres.append(y_res_mes)

    X = pd.concat(blocos_X, ignore_index=True)
    y_res = np.concatenate(blocos_yres)
    return X, y_res, n_meses, len(elegiveis)


def valida_backtest(modelo, origem, ultimo_alvo_disponivel, tp_alvo, atmosfericas, clim, lat, lon, tempos):
    origem = pd.Timestamp(origem)
    origens_validacao = pd.date_range(origem, periods=24, freq="MS")
    origens_validacao = [o for o in origens_validacao
                          if (o + pd.DateOffset(months=1)) <= pd.Timestamp(ultimo_alvo_disponivel)]

    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    lat_flat = lat_grid.ravel().astype(np.float32)
    lon_flat = lon_grid.ravel().astype(np.float32)

    linhas_horizonte = []
    linhas_horizonte_climzero = []
    stats_pred = {"n": 0, "n_neg": 0, "soma": 0.0, "min": np.inf, "max": -np.inf}

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
        residuo_predito = modelo.predict(X_mes)
        tp_predito = clim_flat + residuo_predito

        stats_pred["n"] += len(tp_predito)
        stats_pred["n_neg"] += int((tp_predito < 0).sum())
        stats_pred["soma"] += float(np.sum(tp_predito))
        stats_pred["min"] = min(stats_pred["min"], float(np.min(tp_predito)))
        stats_pred["max"] = max(stats_pred["max"], float(np.max(tp_predito)))

        erro2 = (tp_predito - y_real) ** 2
        rmse_horizonte = float(np.sqrt(np.mean(erro2)))
        linhas_horizonte.append({"horizonte": horizonte, "rmse": rmse_horizonte, "n": len(y_real)})

        # regra "residuo = 0" equivale a usar a propria climatologia como previsao (= B00)
        erro2_climzero = (clim_flat - y_real) ** 2
        linhas_horizonte_climzero.append({"horizonte": horizonte, "rmse": float(np.sqrt(np.mean(erro2_climzero))),
                                           "n": len(y_real)})

        del X_mes, residuo_predito, tp_predito, erro2, y_real, erro2_climzero
        gc.collect()

    df_horizontes = pd.DataFrame(linhas_horizonte)
    df_climzero = pd.DataFrame(linhas_horizonte_climzero)
    rmse_global = float(np.sqrt((df_horizontes["rmse"] ** 2 * df_horizontes["n"]).sum() / df_horizontes["n"].sum()))
    rmse_global_climzero = float(np.sqrt((df_climzero["rmse"] ** 2 * df_climzero["n"]).sum() / df_climzero["n"].sum()))
    stats_pred["media"] = stats_pred["soma"] / stats_pred["n"]
    return df_horizontes, rmse_global, rmse_global_climzero, stats_pred


def roda_backtest(nome, origem, ultimo_alvo_disponivel, tp, tp_alvo, atmosfericas, lat, lon, tempos):
    print(f"\n=== Backtest {nome} (origem {origem}) ===")
    print(f"periodo usado para climatologia: inicio do historico -> {origem}")
    print(f"periodo previsto: {(pd.Timestamp(origem) + pd.DateOffset(months=1)).date()} -> {ultimo_alvo_disponivel}")

    clim = climatologia_ate(tp, origem)

    X_train, y_res_train, n_meses, n_elegiveis = amostra_treino(origem, tp_alvo, atmosfericas, clim, lat, lon, tempos)
    print(f"amostras de treino: {len(X_train):,} ({n_meses}/{n_elegiveis} meses elegiveis, "
          f"{100 * n_meses / n_elegiveis:.1f}% do periodo)")

    print("sanity check y_residuo:", {
        "media": round(float(np.mean(y_res_train)), 4),
        "mediana": round(float(np.median(y_res_train)), 4),
        "desvio_padrao": round(float(np.std(y_res_train)), 4),
        "min": round(float(np.min(y_res_train)), 4),
        "max": round(float(np.max(y_res_train)), 4),
        "pct_positivo": round(100 * float((y_res_train > 0).mean()), 2),
        "pct_negativo": round(100 * float((y_res_train < 0).mean()), 2),
    })

    modelo = LGBMRegressor(
        n_estimators=300,
        num_leaves=31,
        learning_rate=0.05,
        random_state=SEED,
        verbosity=-1,
    )
    modelo.fit(X_train, y_res_train)

    del X_train, y_res_train
    gc.collect()

    df_horizontes, rmse_global, rmse_climzero, stats_pred = valida_backtest(
        modelo, origem, ultimo_alvo_disponivel, tp_alvo, atmosfericas, clim, lat, lon, tempos
    )
    print(f"RMSE global (M03, tp_predito vs tp_alvo): {rmse_global:.4f}")
    print(f"RMSE global (residuo=0, equivale ao B00): {rmse_climzero:.4f}")
    print("sanity check tp_predito:", {k: round(v, 4) if isinstance(v, float) else v for k, v in stats_pred.items()},
          "pct_negativo:", round(100 * stats_pred["n_neg"] / stats_pred["n"], 3))

    del modelo, clim
    gc.collect()

    return df_horizontes, rmse_global, rmse_climzero


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    tp, tp_alvo, atmosfericas, lat, lon, tempos = abre_variaveis()

    df_a, rmse_a, rmse_a_climzero = roda_backtest(
        "A", "2020-12-01", "2022-12-01", tp, tp_alvo, atmosfericas, lat, lon, tempos)
    gc.collect()
    df_b, rmse_b, rmse_b_climzero = roda_backtest(
        "B", "2021-12-01", "2022-12-01", tp, tp_alvo, atmosfericas, lat, lon, tempos)
    gc.collect()

    df_a["backtest"] = "A"
    df_b["backtest"] = "B"
    todos = pd.concat([df_a, df_b], ignore_index=True)
    todos.to_csv(RESULTS_DIR / "m03_rmse_por_horizonte.csv", index=False)

    def faixas(df):
        return {
            "h1_6": df[df.horizonte.between(1, 6)]["rmse"].mean(),
            "h7_12": df[df.horizonte.between(7, 12)]["rmse"].mean(),
            "h13_18": df[df.horizonte.between(13, 18)]["rmse"].mean(),
            "h19_24": df[df.horizonte.between(19, 24)]["rmse"].mean(),
        }

    print("\n--- resumo M03 ---")
    print("Backtest A:", "global", round(rmse_a, 4), "residuo=0 (B00):", round(rmse_a_climzero, 4), faixas(df_a))
    print("Backtest B:", "global", round(rmse_b, 4), "residuo=0 (B00):", round(rmse_b_climzero, 4), faixas(df_b))
