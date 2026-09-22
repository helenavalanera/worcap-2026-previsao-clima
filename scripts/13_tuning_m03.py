"""Gate 13 - tuning controlado do LightGBM sobre o M03 (formulacao por residuo, 600k amostras).

Os dados (cache + X/y) sao construidos uma unica vez por backtest, fora do loop de
configuracoes - os dados nao mudam entre configuracoes, so os hiperparametros do modelo.
Um modelo por vez e treinado, avaliado e liberado antes do proximo.
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
PONTOS_POR_MES_MAX = 8000
PONTOS_POR_MES_600K = 4000

B00_RMSE = {"A": 1.8938, "B": 1.9003}

CONFIGS = [
    {"id": "T00", "mudanca": "controle (config atual do M03)", "hipotese": "referencia",
     "params": {}},
    {"id": "T01", "mudanca": "num_leaves 31 -> 15", "hipotese": "arvore mais simples generaliza melhor (residuo e ruidoso)",
     "params": {"num_leaves": 15}},
    {"id": "T02", "mudanca": "num_leaves 31 -> 63", "hipotese": "mais capacidade capta padroes finos, risco de overfit",
     "params": {"num_leaves": 63}},
    {"id": "T03", "mudanca": "min_child_samples 20 -> 50", "hipotese": "regularizacao mais forte reduz overfit a pontos isolados",
     "params": {"min_child_samples": 50}},
    {"id": "T04", "mudanca": "min_child_samples 20 -> 10", "hipotese": "menos regularizacao, ajusta mais ao treino (risco maior de overfit)",
     "params": {"min_child_samples": 10}},
    {"id": "T05", "mudanca": "learning_rate 0.05 -> 0.02, n_estimators 300 -> 800", "hipotese": "aprendizado mais gradual generaliza melhor",
     "params": {"learning_rate": 0.02, "n_estimators": 800}},
    {"id": "T06", "mudanca": "colsample_bytree=0.8, subsample=0.8, subsample_freq=1", "hipotese": "subamostragem de linhas/colunas reduz variancia entre arvores",
     "params": {"colsample_bytree": 0.8, "subsample": 0.8, "subsample_freq": 1}},
]


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

        y_alvo = tp_alvo.isel(time=t_idx).values[lat_idx, lon_idx].astype(np.float32)
        clim_pontos = clim[mes_target - 1][lat_idx, lon_idx].astype(np.float32)

        cache["lat"].append(lat[lat_idx].astype(np.float32))
        cache["lon"].append(lon[lon_idx].astype(np.float32))
        cache["month_sin"].append(np.full(PONTOS_POR_MES_MAX, np.sin(2 * np.pi * mes_origem / 12), dtype=np.float32))
        cache["month_cos"].append(np.full(PONTOS_POR_MES_MAX, np.cos(2 * np.pi * mes_origem / 12), dtype=np.float32))
        cache["tp_climatologia"].append(clim_pontos)
        cache["y_residuo"].append((y_alvo - clim_pontos).astype(np.float32))

        for v in VARS_ATMOSFERICAS:
            slice_v = atmosfericas[v].isel(time=t_idx).values
            cache[v].append(slice_v[lat_idx, lon_idx].astype(np.float32))

    return cache


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

    soma_erro2, n_total = 0.0, 0
    for origem_val in origens_validacao:
        t_idx = int(np.where(tempos == np.datetime64(origem_val))[0][0])
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

        soma_erro2 += float(np.sum((tp_predito - y_real) ** 2))
        n_total += len(y_real)

        del X_mes, tp_predito, y_real
        gc.collect()

    return float(np.sqrt(soma_erro2 / n_total))


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    tp, tp_alvo, atmosfericas, lat, lon, tempos = abre_variaveis()

    print("=== configuracoes testadas ===")
    for cfg in CONFIGS:
        print(f"{cfg['id']}: {cfg['mudanca']} | hipotese: {cfg['hipotese']}")

    datasets = {}
    for nome_bt, origem in [("A", "2020-12-01"), ("B", "2021-12-01")]:
        clim = climatologia_ate(tp, origem)
        cache = constroi_cache(origem, tp_alvo, atmosfericas, clim, lat, lon, tempos)
        X, y_res = monta_dataset_do_cache(cache, PONTOS_POR_MES_600K)
        datasets[nome_bt] = {"X": X, "y": y_res, "clim": clim, "origem": origem}
        del cache
        gc.collect()
        print(f"\nBacktest {nome_bt}: dataset pronto ({len(X):,} linhas)")

    resultados = []
    ultimo_alvo = "2022-12-01"

    for cfg in CONFIGS:
        params = {"n_estimators": 300, "num_leaves": 31, "learning_rate": 0.05, "random_state": SEED,
                   "verbosity": -1}
        params.update(cfg["params"])

        rmses = {}
        for nome_bt in ["A", "B"]:
            ds = datasets[nome_bt]
            modelo = LGBMRegressor(**params)
            modelo.fit(ds["X"], ds["y"])
            rmse = valida_backtest(modelo, ds["origem"], ultimo_alvo, tp_alvo, atmosfericas, ds["clim"],
                                    lat, lon, tempos)
            rmses[nome_bt] = rmse
            del modelo
            gc.collect()

        media = (rmses["A"] + rmses["B"]) / 2
        pior = max(rmses["A"], rmses["B"])
        resultados.append({
            "config": cfg["id"], "mudanca": cfg["mudanca"], "hipotese": cfg["hipotese"],
            "rmse_A": rmses["A"], "rmse_B": rmses["B"], "media": media, "pior_caso": pior,
        })
        print(f"\n{cfg['id']}: A={rmses['A']:.4f} B={rmses['B']:.4f} media={media:.4f} pior={pior:.4f}")

    df_resultados = pd.DataFrame(resultados)
    rmse_t00_a = df_resultados.loc[df_resultados.config == "T00", "rmse_A"].values[0]
    rmse_t00_b = df_resultados.loc[df_resultados.config == "T00", "rmse_B"].values[0]
    df_resultados["delta_T00_A"] = df_resultados["rmse_A"] - rmse_t00_a
    df_resultados["delta_T00_B"] = df_resultados["rmse_B"] - rmse_t00_b
    df_resultados["delta_B00_A"] = df_resultados["rmse_A"] - B00_RMSE["A"]
    df_resultados["delta_B00_B"] = df_resultados["rmse_B"] - B00_RMSE["B"]

    df_resultados.to_csv(RESULTS_DIR / "m03_tuning.csv", index=False)
    print("\n--- resultado completo ---")
    print(df_resultados.to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    x = np.arange(len(df_resultados))
    plt.plot(x, df_resultados["rmse_A"], marker="o", label="Backtest A")
    plt.plot(x, df_resultados["rmse_B"], marker="o", label="Backtest B")
    plt.axhline(B00_RMSE["A"], linestyle="--", alpha=0.5, color="tab:blue", label="B00 - A")
    plt.axhline(B00_RMSE["B"], linestyle="--", alpha=0.5, color="tab:orange", label="B00 - B")
    plt.xticks(x, df_resultados["config"])
    plt.ylabel("RMSE (mm/day)")
    plt.title("Gate 13 - tuning M03: RMSE por configuracao")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "m03_tuning.png", dpi=120)
    print("\ngrafico salvo em results/m03_tuning.png")
