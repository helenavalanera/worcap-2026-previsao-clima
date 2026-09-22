"""Gate 18 - M05: M03 + anomalias das 9 variaveis atmosfericas em relacao a climatologia propria.

Alinhamento temporal (preserva a semantica do M03):
- feature absoluta e anomalia de uma variavel atmosferica: mes de origem M (mesmo mes da
  propria feature).
- tp_climatologia: mes-alvo M+1.
- climatologias (da precipitacao e das 9 variaveis atmosfericas) usam somente historico
  com time < cutoff do backtest - mesma regra ja validada no B00/M03.
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
VARS_ANOMALIA = [f"{v}_anomalia" for v in VARS_ATMOSFERICAS]
FEATURES = VARS_ATMOSFERICAS + VARS_ANOMALIA + ["lat", "lon", "month_sin", "month_cos", "tp_climatologia"]

SEED = 42
MESES_AMOSTRADOS = 150
PONTOS_POR_MES = 4000  # 150 x 4000 = 600.000

B00_RMSE = {"A": 1.8938, "B": 1.9003}
M03_RMSE = {"A": 1.9048, "B": 1.8874}


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


def climatologia_ate(da, cutoff):
    historico = da.sel(time=slice(None, cutoff))
    return historico.groupby("time.month").mean("time").values.astype(np.float32)  # (12, lat, lon)


def mes_alvo(t_idx, tempos):
    return (pd.Timestamp(tempos[t_idx]).month % 12) + 1


def auditoria_anomalias_e_leakage(origem, tp, atmosfericas, clim_tp, clim_atm, lat, lon, tempos):
    print(f"=== auditoria de leakage: climatologias do Backtest (origem {origem}) ===")
    cutoff_ts = pd.Timestamp(origem)
    print(f"climatologia de tp usa historico ate {origem} (inclusive) - "
          f"ultimo mes do historico usado: {str(tp.sel(time=slice(None, origem)).time.max().values)[:10]}")
    for v in VARS_ATMOSFERICAS[:1]:
        ultimo = str(atmosfericas[v].sel(time=slice(None, origem)).time.max().values)[:10]
        print(f"climatologia de {v} usa historico ate {origem} - ultimo mes usado: {ultimo} "
              f"(<= origem? {pd.Timestamp(ultimo) <= cutoff_ts})")
    print()


def sanity_check_espacial(atmosfericas, clim_atm, lat, lon, tempos):
    print("=== sanity check espacial e temporal (observado - climatologia = anomalia) ===")
    exemplos = [
        ("tropical", -5.0, -60.0),
        ("subtropical", -25.0, -50.0),
        ("mais ao sul", -55.0, -65.0),
    ]
    t_idx = int(np.where(tempos == np.datetime64("2020-06-01"))[0][0])
    mes = pd.Timestamp(tempos[t_idx]).month
    var = "t2"
    grid_obs = atmosfericas[var].isel(time=t_idx).values
    for nome, lat_alvo, lon_alvo in exemplos:
        i = int(np.argmin(np.abs(lat - lat_alvo)))
        j = int(np.argmin(np.abs(lon - lon_alvo)))
        obs = float(grid_obs[i, j])
        clim = float(clim_atm[var][mes - 1, i, j])
        anom = obs - clim
        print(f"{nome} (lat={lat[i]:.2f}, lon={lon[j]:.2f}, mes={mes}, var={var}): "
              f"observado={obs:.3f} | climatologia={clim:.3f} | anomalia={anom:.3f} | "
              f"conferencia manual ok={abs((obs - clim) - anom) < 1e-6}")
    print()


def monta_treino(cutoff, tp_alvo, atmosfericas, clim_tp, clim_atm, lat, lon, tempos, seed=SEED):
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
        clim_pontos = clim_tp[mes_target - 1][lat_idx, lon_idx]

        blocos["lat"].append(lat[lat_idx].astype(np.float32))
        blocos["lon"].append(lon[lon_idx].astype(np.float32))
        blocos["month_sin"].append(np.full(PONTOS_POR_MES, np.sin(2 * np.pi * mes_origem / 12), dtype=np.float32))
        blocos["month_cos"].append(np.full(PONTOS_POR_MES, np.cos(2 * np.pi * mes_origem / 12), dtype=np.float32))
        blocos["tp_climatologia"].append(clim_pontos)
        blocos["y_residuo"].append(y_alvo - clim_pontos)

        for v in VARS_ATMOSFERICAS:
            slice_v = atmosfericas[v].isel(time=t_idx).values
            valor = slice_v[lat_idx, lon_idx].astype(np.float32)
            clim_v = clim_atm[v][mes_origem - 1][lat_idx, lon_idx]
            blocos[v].append(valor)
            blocos[f"{v}_anomalia"].append(valor - clim_v)

    X = pd.DataFrame({v: np.concatenate(blocos[v]) for v in FEATURES})
    y_res = np.concatenate(blocos["y_residuo"])
    return X, y_res


def relatorio_distribuicao_anomalias(X):
    print("=== distribuicao das anomalias (no conjunto de treino amostrado) ===")
    linhas = []
    for v in VARS_ANOMALIA:
        s = X[v]
        linhas.append({
            "feature": v, "media": s.mean(), "mediana": s.median(), "desvio_padrao": s.std(),
            "min": s.min(), "max": s.max(), "q01": s.quantile(0.01), "q99": s.quantile(0.99),
            "pct_nan": round(100 * s.isnull().mean(), 4),
        })
    df = pd.DataFrame(linhas)
    print(df.to_string(index=False))
    print()
    return df


def valida_backtest(modelo, origem, ultimo_alvo_disponivel, tp_alvo, atmosfericas, clim_tp, clim_atm,
                     lat, lon, tempos):
    origem = pd.Timestamp(origem)
    origens_validacao = pd.date_range(origem, periods=24, freq="MS")
    origens_validacao = [o for o in origens_validacao
                          if (o + pd.DateOffset(months=1)) <= pd.Timestamp(ultimo_alvo_disponivel)]

    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    lat_flat = lat_grid.ravel().astype(np.float32)
    lon_flat = lon_grid.ravel().astype(np.float32)

    linhas = []
    stats_pred = {"n": 0, "n_nan": 0, "n_inf": 0, "n_neg": 0, "soma": 0.0, "min": np.inf, "max": -np.inf}

    for origem_val in origens_validacao:
        t_idx = int(np.where(tempos == np.datetime64(origem_val))[0][0])
        horizonte = (origem_val.year - origem.year) * 12 + (origem_val.month - origem.month) + 1
        mes_origem = origem_val.month
        mes_target = mes_alvo(t_idx, tempos)

        month_sin = np.full(lat_flat.shape, np.sin(2 * np.pi * mes_origem / 12), dtype=np.float32)
        month_cos = np.full(lat_flat.shape, np.cos(2 * np.pi * mes_origem / 12), dtype=np.float32)
        clim_flat = clim_tp[mes_target - 1].ravel()

        dados = {"lat": lat_flat, "lon": lon_flat, "month_sin": month_sin, "month_cos": month_cos,
                 "tp_climatologia": clim_flat}
        for v in VARS_ATMOSFERICAS:
            slice_v = atmosfericas[v].isel(time=t_idx).values.astype(np.float32)
            clim_v_flat = clim_atm[v][mes_origem - 1].ravel()
            dados[v] = slice_v.ravel()
            dados[f"{v}_anomalia"] = slice_v.ravel() - clim_v_flat
        X_mes = pd.DataFrame(dados)[FEATURES]

        y_real = tp_alvo.isel(time=t_idx).values.ravel()
        tp_predito = clim_flat + modelo.predict(X_mes)

        stats_pred["n"] += len(tp_predito)
        stats_pred["n_nan"] += int(np.isnan(tp_predito).sum())
        stats_pred["n_inf"] += int(np.isinf(tp_predito).sum())
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


def roda_backtest(nome, origem, ultimo_alvo, tp, tp_alvo, atmosfericas, lat, lon, tempos):
    print(f"\n########## Backtest {nome} (origem {origem}) ##########")
    clim_tp = climatologia_ate(tp, origem)
    clim_atm = {v: climatologia_ate(atmosfericas[v], origem) for v in VARS_ATMOSFERICAS}

    auditoria_anomalias_e_leakage(origem, tp, atmosfericas, clim_tp, clim_atm, lat, lon, tempos)
    if nome == "A":
        sanity_check_espacial(atmosfericas, clim_atm, lat, lon, tempos)

    mem_disp = memoria_disponivel_mb()
    n_linhas = MESES_AMOSTRADOS * PONTOS_POR_MES
    print(f"linhas: {n_linhas:,} | features: {len(FEATURES)} (esperado 23) | "
          f"X estimado: {n_linhas * len(FEATURES) * 4 / 1e6:.1f} MB | "
          f"y estimado: {n_linhas * 4 / 1e6:.1f} MB | memoria disponivel: {mem_disp:.0f} MB")
    assert len(FEATURES) == 23, "numero de features nao bate com o esperado"

    X, y_res = monta_treino(origem, tp_alvo, atmosfericas, clim_tp, clim_atm, lat, lon, tempos)

    if nome == "A":
        relatorio_distribuicao_anomalias(X)

    modelo = LGBMRegressor(
        n_estimators=300,
        num_leaves=15,
        learning_rate=0.05,
        random_state=SEED,
        verbosity=-1,
    )
    modelo.fit(X, y_res)
    importancias = pd.Series(modelo.feature_importances_, index=FEATURES).sort_values(ascending=False)

    del X, y_res
    gc.collect()

    df_horizontes, rmse_global, stats_pred = valida_backtest(
        modelo, origem, ultimo_alvo, tp_alvo, atmosfericas, clim_tp, clim_atm, lat, lon, tempos
    )
    print(f"RMSE global M05: {rmse_global:.4f} | M03: {M03_RMSE[nome]:.4f} | B00: {B00_RMSE[nome]:.4f}")
    print("sanity check previsoes:", {k: round(v, 4) if isinstance(v, float) else v for k, v in stats_pred.items()},
          "pct_negativo:", round(100 * stats_pred["n_neg"] / stats_pred["n"], 3))

    del modelo, clim_tp, clim_atm
    gc.collect()

    return df_horizontes, rmse_global, importancias


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    tp, tp_alvo, atmosfericas, lat, lon, tempos = abre_variaveis()

    df_a, rmse_a, imp_a = roda_backtest("A", "2020-12-01", "2022-12-01", tp, tp_alvo, atmosfericas, lat, lon, tempos)
    gc.collect()
    df_b, rmse_b, imp_b = roda_backtest("B", "2021-12-01", "2022-12-01", tp, tp_alvo, atmosfericas, lat, lon, tempos)
    gc.collect()

    df_a["backtest"] = "A"
    df_b["backtest"] = "B"
    todos = pd.concat([df_a, df_b], ignore_index=True)
    todos.to_csv(RESULTS_DIR / "m05_por_horizonte.csv", index=False)

    imp = pd.DataFrame({"backtest_A": imp_a, "backtest_B": imp_b}).fillna(0)
    imp.to_csv(RESULTS_DIR / "m05_feature_importance.csv")

    grupos = {
        "absolutas": VARS_ATMOSFERICAS, "anomalias": VARS_ANOMALIA,
        "geografico_temporal": ["lat", "lon", "month_sin", "month_cos"], "tp_climatologia": ["tp_climatologia"],
    }
    print("\n--- importancia por grupo ---")
    for nome_grupo, feats in grupos.items():
        soma_a = imp_a[feats].sum()
        soma_b = imp_b[feats].sum()
        print(f"{nome_grupo}: A={soma_a} ({100*soma_a/imp_a.sum():.1f}%) | "
              f"B={soma_b} ({100*soma_b/imp_b.sum():.1f}%)")

    print("\n--- pares absoluta x anomalia (Backtest A) ---")
    for v in VARS_ATMOSFERICAS:
        print(f"{v}: {imp_a[v]} | {v}_anomalia: {imp_a[f'{v}_anomalia']}")

    def faixas(df):
        return {
            "h1_6": df[df.horizonte.between(1, 6)]["rmse"].mean(),
            "h7_12": df[df.horizonte.between(7, 12)]["rmse"].mean(),
            "h13_18": df[df.horizonte.between(13, 18)]["rmse"].mean(),
            "h19_24": df[df.horizonte.between(19, 24)]["rmse"].mean(),
        }

    resumo = pd.DataFrame([
        {"modelo": "B00", "rmse_A": B00_RMSE["A"], "rmse_B": B00_RMSE["B"]},
        {"modelo": "M03", "rmse_A": M03_RMSE["A"], "rmse_B": M03_RMSE["B"]},
        {"modelo": "M05", "rmse_A": rmse_a, "rmse_B": rmse_b},
    ])
    resumo["media"] = (resumo["rmse_A"] + resumo["rmse_B"]) / 2
    resumo["pior_caso"] = resumo[["rmse_A", "rmse_B"]].max(axis=1)
    resumo.to_csv(RESULTS_DIR / "m05_anomalias_atmosfericas.csv", index=False)

    print("\n--- resumo final M05 ---")
    print(resumo.to_string(index=False))
    print("\nM05 - B00: A=%.4f (%.2f%%) | B=%.4f (%.2f%%)" % (
        rmse_a - B00_RMSE["A"], 100 * (rmse_a - B00_RMSE["A"]) / B00_RMSE["A"],
        rmse_b - B00_RMSE["B"], 100 * (rmse_b - B00_RMSE["B"]) / B00_RMSE["B"]))
    print("M05 - M03: A=%.4f (%.2f%%) | B=%.4f (%.2f%%)" % (
        rmse_a - M03_RMSE["A"], 100 * (rmse_a - M03_RMSE["A"]) / M03_RMSE["A"],
        rmse_b - M03_RMSE["B"], 100 * (rmse_b - M03_RMSE["B"]) / M03_RMSE["B"]))

    print("\n--- M05 por faixa de horizonte ---")
    print("Backtest A:", faixas(df_a))
    print("Backtest B:", faixas(df_b))
