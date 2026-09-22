"""Gate 21 - M05 + ONI: identico ao Gate 19 (M05, 3 folds rolling-origin), acrescentando
uma unica feature (oni) ao dataset: o indice ONI do mes de origem de cada linha.

Fonte do ONI: serie publicada pela NOAA CPC (data/external/oni_raw.txt), 1950-presente.
LIMITACAO DOCUMENTADA: esta e a serie oficial atual, nao um arquivo historico "como era
conhecido em tempo real" em cada cutoff (nao ha acesso a vintages arquivados). Para
1940-1949 (fora da cobertura do ONI) o valor fica NaN - nao e inventado; o LightGBM lida
com NaN nativamente.

Criterio de sucesso fixado ANTES da execucao (nao alterado depois do resultado):
1) RMSE agregado (M05+ONI) < RMSE agregado (M05 puro) com melhora >= 0.3% relativo.
2) sinal do delta (M05+ONI - M05) igual em pelo menos 2 dos 3 folds.
"""

import gc
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from lightgbm import LGBMRegressor

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
EXTERNAL_DIR = Path(__file__).resolve().parent.parent / "data" / "external"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

VARS_ATMOSFERICAS = [
    "t2", "cloud_cover", "shum_850", "surface_pressure",
    "u_850", "v_850", "temperature_850", "rel_hum_850", "geopotential_850",
]
VARS_ANOMALIA = [f"{v}_anomalia" for v in VARS_ATMOSFERICAS]
FEATURES = VARS_ATMOSFERICAS + VARS_ANOMALIA + ["lat", "lon", "month_sin", "month_cos", "tp_climatologia", "oni"]

SEED = 42
MESES_AMOSTRADOS = 150
PONTOS_POR_MES = 4000

FOLDS = [
    {"nome": "F1", "cutoff": "2016-12-01", "eval_start": "2017-01-01", "eval_end": "2018-12-01"},
    {"nome": "F2", "cutoff": "2018-12-01", "eval_start": "2019-01-01", "eval_end": "2020-12-01"},
    {"nome": "F3", "cutoff": "2020-12-01", "eval_start": "2021-01-01", "eval_end": "2022-12-01"},
]

# resultados do Gate 19 (M05 puro), reaproveitados como referencia fixa - nao retreinados aqui
M05_PURO_FOLD = {"F1": 1.881280, "F2": 1.836916, "F3": 1.890257}
M05_PURO_AGREGADO = 1.869630

LAT_EDGES = [-60, -45, -30, -15, 0, 15]
LAT_LABELS = ["-60_-45", "-45_-30", "-30_-15", "-15_0", "0_15"]
INT_EDGES = [1, 3, 5, 10, 20]
INT_LABELS = ["0-1", "1-3", "3-5", "5-10", "10-20", ">20"]

SEAS_MES_CENTRAL = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
                     "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12}


def carrega_oni_mensal():
    df = pd.read_csv(EXTERNAL_DIR / "oni_raw.txt", sep=r"\s+")
    df["mes"] = df["SEAS"].map(SEAS_MES_CENTRAL)
    df["data"] = pd.to_datetime(dict(year=df["YR"], month=df["mes"], day=1))
    serie = df.set_index("data")["ANOM"]
    print(f"ONI carregado: {serie.index.min().date()} -> {serie.index.max().date()} "
          f"({len(serie)} meses); fonte: NOAA CPC (serie atual, nao vintage operacional historico)")
    return serie


def climatologia_ate(da, cutoff):
    historico = da.sel(time=slice(None, cutoff))
    return historico.groupby("time.month").mean("time").values.astype(np.float32)


def mes_alvo(t_idx, tempos):
    return (pd.Timestamp(tempos[t_idx]).month % 12) + 1


def oni_do_mes(serie_oni, data):
    data = pd.Timestamp(data.year, data.month, 1)
    return float(serie_oni.get(data, np.nan))


def monta_treino(cutoff, tp_alvo, atmosfericas, clim_tp, clim_atm, serie_oni, lat, lon, tempos, seed=SEED):
    elegiveis = np.where(tempos < np.datetime64(cutoff))[0]
    rng = np.random.default_rng(seed)
    n_meses = min(MESES_AMOSTRADOS, len(elegiveis))
    meses_idx = np.sort(rng.choice(elegiveis, size=n_meses, replace=False))

    max_target = pd.Timestamp(tempos[meses_idx[-1]]) + pd.DateOffset(months=1)
    assert max_target <= pd.Timestamp(cutoff), "target de treino ultrapassou o cutoff do fold"

    blocos = {v: [] for v in FEATURES}
    blocos["y_residuo"] = []

    for t_idx in meses_idx:
        lat_idx = rng.integers(0, len(lat), size=PONTOS_POR_MES)
        lon_idx = rng.integers(0, len(lon), size=PONTOS_POR_MES)

        data_origem = pd.Timestamp(tempos[t_idx])
        mes_origem = data_origem.month
        mes_target = mes_alvo(t_idx, tempos)

        y_alvo = tp_alvo.isel(time=t_idx).values[lat_idx, lon_idx].astype(np.float32)
        clim_pontos = clim_tp[mes_target - 1][lat_idx, lon_idx]

        blocos["lat"].append(lat[lat_idx].astype(np.float32))
        blocos["lon"].append(lon[lon_idx].astype(np.float32))
        blocos["month_sin"].append(np.full(PONTOS_POR_MES, np.sin(2 * np.pi * mes_origem / 12), dtype=np.float32))
        blocos["month_cos"].append(np.full(PONTOS_POR_MES, np.cos(2 * np.pi * mes_origem / 12), dtype=np.float32))
        blocos["tp_climatologia"].append(clim_pontos)
        blocos["oni"].append(np.full(PONTOS_POR_MES, oni_do_mes(serie_oni, data_origem), dtype=np.float32))
        blocos["y_residuo"].append(y_alvo - clim_pontos)

        for v in VARS_ATMOSFERICAS:
            slice_v = atmosfericas[v].isel(time=t_idx).values
            valor = slice_v[lat_idx, lon_idx].astype(np.float32)
            clim_v = clim_atm[v][mes_origem - 1][lat_idx, lon_idx]
            blocos[v].append(valor)
            blocos[f"{v}_anomalia"].append(valor - clim_v)

    X = pd.DataFrame({v: np.concatenate(blocos[v]) for v in FEATURES})
    y_res = np.concatenate(blocos["y_residuo"])
    pct_nan_oni = round(100 * X["oni"].isnull().mean(), 3)
    print(f"  ONI no treino: {pct_nan_oni}% NaN (meses fora da cobertura 1950+)")
    return X, y_res, n_meses, len(elegiveis)


class Acumulador:
    def __init__(self):
        self.fold = {}
        self.lat_bin = {lb: [0.0, 0.0, 0] for lb in LAT_LABELS}
        self.intensidade = {ib: [0.0, 0.0, 0] for ib in INT_LABELS}
        self.sse_m05oni_vetor = []
        self.n_total = 0

    def registra_mes(self, nome_fold, lat_flat, y_true, pred_b00, pred_m05oni):
        err_m05oni = y_true - pred_m05oni
        sq_m05oni = (err_m05oni ** 2).astype(np.float64)

        if nome_fold not in self.fold:
            self.fold[nome_fold] = [0.0, 0]
        self.fold[nome_fold][0] += float(sq_m05oni.sum())
        self.fold[nome_fold][1] += len(sq_m05oni)

        idx_lat = np.digitize(lat_flat, LAT_EDGES[1:-1])
        for i, lb in enumerate(LAT_LABELS):
            mask = idx_lat == i
            if mask.any():
                self.lat_bin[lb][1] += float(sq_m05oni[mask].sum())
                self.lat_bin[lb][2] += int(mask.sum())

        idx_int = np.digitize(y_true, INT_EDGES)
        for i, ib in enumerate(INT_LABELS):
            mask = idx_int == i
            if mask.any():
                self.intensidade[ib][1] += float(sq_m05oni[mask].sum())
                self.intensidade[ib][2] += int(mask.sum())

        self.sse_m05oni_vetor.append(sq_m05oni.astype(np.float32))
        self.n_total += len(y_true)


def concentracao_sse(vetor_sse):
    v = np.sort(vetor_sse)[::-1]
    total = v.sum()
    n = len(v)
    return {f"top_{int(p*100)}pct": round(100 * float(v[:max(1,int(n*p))].sum() / total), 3) for p in [0.01, 0.05, 0.10]}


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    inicio_parteB = time.time()

    tp = xr.open_dataset(RAW_DIR / "treino_tp.nc")["tp"]
    tp_alvo = xr.open_dataset(RAW_DIR / "treino_tp_alvo.nc")["tp_alvo"]
    atmosfericas = {v: xr.open_dataset(RAW_DIR / f"treino_{v}.nc")[v] for v in VARS_ATMOSFERICAS}
    lat = tp_alvo["lat"].values
    lon = tp_alvo["lon"].values
    tempos = tp_alvo["time"].values
    serie_oni = carrega_oni_mensal()

    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    lat_flat_global = lat_grid.ravel().astype(np.float32)
    lon_flat_global = lon_grid.ravel().astype(np.float32)

    acc = Acumulador()
    tempos_fold = {}

    for fold in FOLDS:
        t0 = time.time()
        nome, cutoff, eval_start, eval_end = fold["nome"], fold["cutoff"], fold["eval_start"], fold["eval_end"]
        print(f"\n########## {nome} (cutoff={cutoff}) ##########")

        clim_tp = climatologia_ate(tp, cutoff)
        clim_atm = {v: climatologia_ate(atmosfericas[v], cutoff) for v in VARS_ATMOSFERICAS}

        X, y_res, n_meses, n_elegiveis = monta_treino(cutoff, tp_alvo, atmosfericas, clim_tp, clim_atm,
                                                        serie_oni, lat, lon, tempos)
        print(f"  treino: {len(X):,} linhas | features: {len(FEATURES)} (esperado 24)")
        assert len(FEATURES) == 24

        modelo = LGBMRegressor(n_estimators=300, num_leaves=15, learning_rate=0.05,
                                random_state=SEED, verbosity=-1)
        modelo.fit(X, y_res)
        importancias = pd.Series(modelo.feature_importances_, index=FEATURES).sort_values(ascending=False)
        print(f"  importancia do oni: {importancias.get('oni', 0)} "
              f"(posicao {list(importancias.index).index('oni')+1} de {len(FEATURES)})")
        del X, y_res
        gc.collect()

        origens_avaliacao = pd.date_range(eval_start, eval_end, freq="MS")
        for target in origens_avaliacao:
            origem_val = target - pd.DateOffset(months=1)
            t_idx = int(np.where(tempos == np.datetime64(origem_val))[0][0])
            mes_origem = origem_val.month
            mes_target = target.month

            clim_flat = clim_tp[mes_target - 1].ravel()
            month_sin = np.full(lat_flat_global.shape, np.sin(2 * np.pi * mes_origem / 12), dtype=np.float32)
            month_cos = np.full(lat_flat_global.shape, np.cos(2 * np.pi * mes_origem / 12), dtype=np.float32)
            oni_flat = np.full(lat_flat_global.shape, oni_do_mes(serie_oni, origem_val), dtype=np.float32)

            dados = {"lat": lat_flat_global, "lon": lon_flat_global, "month_sin": month_sin, "month_cos": month_cos,
                     "tp_climatologia": clim_flat, "oni": oni_flat}
            for v in VARS_ATMOSFERICAS:
                slice_v = atmosfericas[v].isel(time=t_idx).values.astype(np.float32)
                clim_v_flat = clim_atm[v][mes_origem - 1].ravel()
                dados[v] = slice_v.ravel()
                dados[f"{v}_anomalia"] = slice_v.ravel() - clim_v_flat
            X_mes = pd.DataFrame(dados)[FEATURES]

            y_true = tp_alvo.isel(time=t_idx).values.ravel()
            pred_b00 = clim_flat
            pred_m05oni = clim_flat + modelo.predict(X_mes)

            acc.registra_mes(nome, lat_flat_global, y_true, pred_b00, pred_m05oni)

            del X_mes, pred_m05oni, y_true
            gc.collect()

        del modelo, clim_tp, clim_atm
        gc.collect()
        tempos_fold[nome] = time.time() - t0
        rmse_fold = np.sqrt(acc.fold[nome][0] / acc.fold[nome][1])
        print(f"  RMSE M05+ONI ({nome}): {rmse_fold:.4f} | M05 puro (Gate 19): {M05_PURO_FOLD[nome]:.4f} | "
              f"delta: {rmse_fold - M05_PURO_FOLD[nome]:+.4f} | tempo: {tempos_fold[nome]:.1f}s")

    # ---------- consolidacao ----------
    print("\n\n=== CONSOLIDACAO M05+ONI ===")
    linhas_fold = []
    sse_total, n_total = 0.0, 0
    deltas_sinal = []
    for fold in FOLDS:
        nome = fold["nome"]
        sse, n = acc.fold[nome]
        rmse = np.sqrt(sse / n)
        delta = rmse - M05_PURO_FOLD[nome]
        deltas_sinal.append(delta)
        linhas_fold.append({"fold": nome, "cutoff": fold["cutoff"], "modelo": "M05+ONI",
                             "SSE": sse, "N": n, "RMSE": rmse,
                             "RMSE_M05_puro": M05_PURO_FOLD[nome], "delta": delta})
        sse_total += sse
        n_total += n

    rmse_agregado = np.sqrt(sse_total / n_total)
    delta_agregado = rmse_agregado - M05_PURO_AGREGADO
    delta_pct = 100 * delta_agregado / M05_PURO_AGREGADO
    linhas_fold.append({"fold": "AGREGADO", "cutoff": "", "modelo": "M05+ONI", "SSE": sse_total, "N": n_total,
                         "RMSE": rmse_agregado, "RMSE_M05_puro": M05_PURO_AGREGADO, "delta": delta_agregado})
    df_folds = pd.DataFrame(linhas_fold)
    df_folds.to_csv(RESULTS_DIR / "rolling_oni_folds.csv", index=False)
    print(df_folds.to_string(index=False))

    print(f"\nRMSE agregado M05+ONI: {rmse_agregado:.4f} | M05 puro: {M05_PURO_AGREGADO:.4f} | "
          f"delta: {delta_agregado:+.4f} ({delta_pct:+.3f}%)")

    sinais_iguais = sum(1 for d in deltas_sinal if np.sign(d) == np.sign(delta_agregado))
    print(f"sinal do delta por fold: {[round(d,4) for d in deltas_sinal]} | "
          f"sinal igual ao agregado em {sinais_iguais}/3 folds")

    criterio_1 = delta_pct <= -0.3
    criterio_2 = sinais_iguais >= 2
    print(f"\ncriterio 1 (melhora >=0.3% relativo): {criterio_1}")
    print(f"criterio 2 (mesmo sinal em >=2/3 folds): {criterio_2}")
    print(f"CRITERIO ATENDIDO (ambos): {criterio_1 and criterio_2}")

    linhas_lat = []
    for lb in LAT_LABELS:
        _, sse, n = acc.lat_bin[lb]
        linhas_lat.append({"faixa_lat": lb, "N": n, "M05_ONI_RMSE": np.sqrt(sse / n) if n else np.nan})
    df_lat = pd.DataFrame(linhas_lat)
    df_lat.to_csv(RESULTS_DIR / "rolling_oni_latitude.csv", index=False)

    linhas_int = []
    for ib in INT_LABELS:
        _, sse, n = acc.intensidade[ib]
        linhas_int.append({"faixa": ib, "N": n, "M05_ONI_RMSE": np.sqrt(sse / n) if n else np.nan,
                            "pct_SSE": round(100 * sse / sse_total, 3)})
    df_int = pd.DataFrame(linhas_int)
    df_int.to_csv(RESULTS_DIR / "rolling_oni_intensity.csv", index=False)

    print("\n--- diagnostico secundario: latitude ---")
    print(df_lat.to_string(index=False))
    print("\n--- diagnostico secundario: intensidade ---")
    print(df_int.to_string(index=False))

    sse_m05oni_vetor = np.concatenate(acc.sse_m05oni_vetor)
    conc = concentracao_sse(sse_m05oni_vetor)
    print(f"\n--- concentracao do SSE (M05+ONI) ---\n{conc}")

    tempo_parteB = time.time() - inicio_parteB
    pico_mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    print(f"\ntempo Parte B: {tempo_parteB:.1f}s ({tempo_parteB/60:.1f} min)")
    print(f"pico aproximado de memoria (RSS): {pico_mem_mb:.0f} MB")
