"""Gate 19 - validacao rolling-origin (3 folds historicos) comparando B00 x M05, com
diagnostico de erro (fold, ano da janela, mes-calendario, intensidade, latitude,
concentracao do SSE) calculado em passe unico (streaming), sem materializar a tabela
completa de ~5,6 milhoes de observacoes.

Cada fold recalcula do zero: climatologia de tp (para B00 e para tp_climatologia),
climatologias das 9 variaveis atmosfericas (para as anomalias) e o treino do M05 -
tudo usando somente historico ate o cutoff do fold. As climatologias ficam congeladas
durante toda a janela de avaliacao de 24 meses.
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

FOLDS = [
    {"nome": "F1", "cutoff": "2016-12-01", "eval_start": "2017-01-01", "eval_end": "2018-12-01"},
    {"nome": "F2", "cutoff": "2018-12-01", "eval_start": "2019-01-01", "eval_end": "2020-12-01"},
    {"nome": "F3", "cutoff": "2020-12-01", "eval_start": "2021-01-01", "eval_end": "2022-12-01"},
]

LAT_EDGES = [-60, -45, -30, -15, 0, 15]
LAT_LABELS = ["-60_-45", "-45_-30", "-30_-15", "-15_0", "0_15"]
INT_EDGES = [1, 3, 5, 10, 20]
INT_LABELS = ["0-1", "1-3", "3-5", "5-10", "10-20", ">20"]


def climatologia_ate(da, cutoff):
    historico = da.sel(time=slice(None, cutoff))
    return historico.groupby("time.month").mean("time").values.astype(np.float32)


def mes_alvo(t_idx, tempos):
    return (pd.Timestamp(tempos[t_idx]).month % 12) + 1


def monta_treino(cutoff, tp_alvo, atmosfericas, clim_tp, clim_atm, lat, lon, tempos, seed=SEED):
    elegiveis = np.where(tempos < np.datetime64(cutoff))[0]
    rng = np.random.default_rng(seed)
    n_meses = min(MESES_AMOSTRADOS, len(elegiveis))
    meses_idx = np.sort(rng.choice(elegiveis, size=n_meses, replace=False))

    max_target = pd.Timestamp(tempos[meses_idx[-1]]) + pd.DateOffset(months=1)
    print(f"  primeiro feature_time treino: {pd.Timestamp(tempos[meses_idx[0]]).date()}")
    print(f"  ultimo feature_time treino:   {pd.Timestamp(tempos[meses_idx[-1]]).date()}")
    print(f"  primeiro target_date treino:  {(pd.Timestamp(tempos[meses_idx[0]]) + pd.DateOffset(months=1)).date()}")
    print(f"  ultimo target_date treino:    {max_target.date()}")
    print(f"  cutoff:                       {cutoff}")
    assert max_target <= pd.Timestamp(cutoff), "target de treino ultrapassou o cutoff do fold"

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
    return X, y_res, n_meses, len(elegiveis)


class Acumulador:
    """Acumula SSE/N em passe unico, por varios recortes, sem guardar a tabela completa."""

    def __init__(self):
        self.fold = {}          # nome_fold -> {"b00":[sse,n], "m05":[sse,n]}
        self.ano = {1: [0.0, 0.0, 0], 2: [0.0, 0.0, 0]}          # ano -> [sse_b00, sse_m05, n]
        self.mes = {m: [0.0, 0.0, 0] for m in range(1, 13)}       # mes -> [sse_b00, sse_m05, n]
        self.lat_bin = {lb: [0.0, 0.0, 0] for lb in LAT_LABELS}
        self.intensidade = {ib: [0.0, 0.0, 0] for ib in INT_LABELS}
        self.sse_b00_vetor = []
        self.sse_m05_vetor = []
        self.sse_m05_clip_total = 0.0
        self.n_clip_alterado = 0
        self.n_total = 0
        self.min_y_true = np.inf

    def registra_mes(self, nome_fold, ano_janela, mes_cal, lat_flat, y_true, pred_b00, pred_m05):
        err_b00 = y_true - pred_b00
        err_m05 = y_true - pred_m05
        sq_b00 = (err_b00 ** 2).astype(np.float64)
        sq_m05 = (err_m05 ** 2).astype(np.float64)

        if nome_fold not in self.fold:
            self.fold[nome_fold] = {"b00": [0.0, 0], "m05": [0.0, 0]}
        self.fold[nome_fold]["b00"][0] += float(sq_b00.sum())
        self.fold[nome_fold]["b00"][1] += len(sq_b00)
        self.fold[nome_fold]["m05"][0] += float(sq_m05.sum())
        self.fold[nome_fold]["m05"][1] += len(sq_m05)

        self.ano[ano_janela][0] += float(sq_b00.sum())
        self.ano[ano_janela][1] += float(sq_m05.sum())
        self.ano[ano_janela][2] += len(sq_b00)

        self.mes[mes_cal][0] += float(sq_b00.sum())
        self.mes[mes_cal][1] += float(sq_m05.sum())
        self.mes[mes_cal][2] += len(sq_b00)

        idx_lat = np.digitize(lat_flat, LAT_EDGES[1:-1])  # 0..4
        for i, lb in enumerate(LAT_LABELS):
            mask = idx_lat == i
            if mask.any():
                self.lat_bin[lb][0] += float(sq_b00[mask].sum())
                self.lat_bin[lb][1] += float(sq_m05[mask].sum())
                self.lat_bin[lb][2] += int(mask.sum())

        idx_int = np.digitize(y_true, INT_EDGES)  # 0..5
        for i, ib in enumerate(INT_LABELS):
            mask = idx_int == i
            if mask.any():
                self.intensidade[ib][0] += float(sq_b00[mask].sum())
                self.intensidade[ib][1] += float(sq_m05[mask].sum())
                self.intensidade[ib][2] += int(mask.sum())

        self.sse_b00_vetor.append(sq_b00.astype(np.float32))
        self.sse_m05_vetor.append(sq_m05.astype(np.float32))

        pred_m05_clip = np.maximum(0.0, pred_m05)
        alterado = pred_m05 < 0
        self.n_clip_alterado += int(alterado.sum())
        self.sse_m05_clip_total += float(((y_true - pred_m05_clip) ** 2).sum())

        self.n_total += len(y_true)
        self.min_y_true = min(self.min_y_true, float(y_true.min()))


def concentracao_sse(vetor_sse):
    v = np.sort(vetor_sse)[::-1]
    total = v.sum()
    n = len(v)
    resultado = {}
    for pct in [0.01, 0.05, 0.10]:
        k = max(1, int(n * pct))
        resultado[f"top_{int(pct*100)}pct"] = round(100 * float(v[:k].sum() / total), 3)
    return resultado


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    inicio_geral = time.time()
    limite_minutos = 90
    print(f"=== TIME-BOX: limite definido em {limite_minutos} minutos a partir de agora ===\n")

    tp = xr.open_dataset(RAW_DIR / "treino_tp.nc")["tp"]
    tp_alvo = xr.open_dataset(RAW_DIR / "treino_tp_alvo.nc")["tp_alvo"]
    atmosfericas = {v: xr.open_dataset(RAW_DIR / f"treino_{v}.nc")[v] for v in VARS_ATMOSFERICAS}
    lat = tp_alvo["lat"].values
    lon = tp_alvo["lon"].values
    tempos = tp_alvo["time"].values

    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    lat_flat_global = lat_grid.ravel().astype(np.float32)

    acc = Acumulador()
    tempos_fold = {}

    for fold in FOLDS:
        t0 = time.time()
        nome, cutoff, eval_start, eval_end = fold["nome"], fold["cutoff"], fold["eval_start"], fold["eval_end"]
        print(f"\n########## {nome} (cutoff={cutoff}, avaliacao {eval_start} -> {eval_end}) ##########")

        clim_tp = climatologia_ate(tp, cutoff)
        clim_atm = {v: climatologia_ate(atmosfericas[v], cutoff) for v in VARS_ATMOSFERICAS}

        max_tp_clim_date = str(tp.sel(time=slice(None, cutoff)).time.max().values)[:10]
        max_atm_clim_date = str(atmosfericas["t2"].sel(time=slice(None, cutoff)).time.max().values)[:10]
        print("--- auditoria de leakage ---")
        print(f"cutoff={cutoff} | max_tp_climatology_date={max_tp_clim_date} | "
              f"max_atmos_climatology_date={max_atm_clim_date}")
        assert pd.Timestamp(max_tp_clim_date) <= pd.Timestamp(cutoff)
        assert pd.Timestamp(max_atm_clim_date) <= pd.Timestamp(cutoff)
        assert pd.Timestamp(eval_start) > pd.Timestamp(cutoff)

        print("--- amostragem de treino ---")
        X, y_res, n_meses, n_elegiveis = monta_treino(cutoff, tp_alvo, atmosfericas, clim_tp, clim_atm,
                                                        lat, lon, tempos)
        print(f"  meses elegiveis: {n_elegiveis} | meses sorteados: {n_meses} | "
              f"pontos/mes: {PONTOS_POR_MES} | total treino: {len(X):,}")

        modelo = LGBMRegressor(n_estimators=300, num_leaves=15, learning_rate=0.05,
                                random_state=SEED, verbosity=-1)
        modelo.fit(X, y_res)
        del X, y_res
        gc.collect()

        print("--- avaliacao (passe unico, streaming) ---")
        origens_avaliacao = pd.date_range(eval_start, eval_end, freq="MS")
        for i, target in enumerate(origens_avaliacao):
            origem_val = target - pd.DateOffset(months=1)
            assert origem_val == target - pd.DateOffset(months=1)  # feature/origin = target - 1 mes

            t_idx = int(np.where(tempos == np.datetime64(origem_val))[0][0])
            mes_origem = origem_val.month
            mes_target = target.month
            ano_janela = 1 if i < 12 else 2

            clim_flat = clim_tp[mes_target - 1].ravel()
            month_sin = np.full(lat_flat_global.shape, np.sin(2 * np.pi * mes_origem / 12), dtype=np.float32)
            month_cos = np.full(lat_flat_global.shape, np.cos(2 * np.pi * mes_origem / 12), dtype=np.float32)
            lon_flat = lon_grid.ravel().astype(np.float32)

            dados = {"lat": lat_flat_global, "lon": lon_flat, "month_sin": month_sin, "month_cos": month_cos,
                     "tp_climatologia": clim_flat}
            for v in VARS_ATMOSFERICAS:
                slice_v = atmosfericas[v].isel(time=t_idx).values.astype(np.float32)
                clim_v_flat = clim_atm[v][mes_origem - 1].ravel()
                dados[v] = slice_v.ravel()
                dados[f"{v}_anomalia"] = slice_v.ravel() - clim_v_flat
            X_mes = pd.DataFrame(dados)[FEATURES]

            y_true = tp_alvo.isel(time=t_idx).values.ravel()
            pred_b00 = clim_flat
            pred_m05 = clim_flat + modelo.predict(X_mes)

            acc.registra_mes(nome, ano_janela, mes_target, lat_flat_global, y_true, pred_b00, pred_m05)

            del X_mes, pred_m05, y_true
            gc.collect()

        del modelo, clim_tp, clim_atm
        gc.collect()

        tempos_fold[nome] = time.time() - t0
        print(f"tempo do fold: {tempos_fold[nome]:.1f}s")

        rmse_b00_fold = np.sqrt(acc.fold[nome]["b00"][0] / acc.fold[nome]["b00"][1])
        rmse_m05_fold = np.sqrt(acc.fold[nome]["m05"][0] / acc.fold[nome]["m05"][1])
        print(f"checkpoint {nome}: RMSE B00={rmse_b00_fold:.4f} | RMSE M05={rmse_m05_fold:.4f} | "
              f"NaN/inf inesperados=nao verificado alem dos asserts acima")

    # ---------- consolidacao ----------
    print("\n\n=== CONSOLIDACAO FINAL ===")

    linhas_fold = []
    sse_b00_total, sse_m05_total, n_total = 0.0, 0.0, 0
    for fold in FOLDS:
        nome = fold["nome"]
        sse_b00, n_b00 = acc.fold[nome]["b00"]
        sse_m05, n_m05 = acc.fold[nome]["m05"]
        linhas_fold.append({"fold": nome, "cutoff": fold["cutoff"], "inicio_teste": fold["eval_start"],
                             "fim_teste": fold["eval_end"], "modelo": "B00", "SSE": sse_b00, "N": n_b00,
                             "RMSE": np.sqrt(sse_b00 / n_b00)})
        linhas_fold.append({"fold": nome, "cutoff": fold["cutoff"], "inicio_teste": fold["eval_start"],
                             "fim_teste": fold["eval_end"], "modelo": "M05", "SSE": sse_m05, "N": n_m05,
                             "RMSE": np.sqrt(sse_m05 / n_m05)})
        sse_b00_total += sse_b00
        sse_m05_total += sse_m05
        n_total += n_b00

    rmse_b00_agregado = np.sqrt(sse_b00_total / n_total)
    rmse_m05_agregado = np.sqrt(sse_m05_total / n_total)
    linhas_fold.append({"fold": "AGREGADO", "cutoff": "", "inicio_teste": "2017-01", "fim_teste": "2022-12",
                         "modelo": "B00", "SSE": sse_b00_total, "N": n_total, "RMSE": rmse_b00_agregado})
    linhas_fold.append({"fold": "AGREGADO", "cutoff": "", "inicio_teste": "2017-01", "fim_teste": "2022-12",
                         "modelo": "M05", "SSE": sse_m05_total, "N": n_total, "RMSE": rmse_m05_agregado})
    df_folds = pd.DataFrame(linhas_fold)
    df_folds.to_csv(RESULTS_DIR / "rolling_folds.csv", index=False)
    print("\n--- rolling_folds.csv ---")
    print(df_folds.to_string(index=False))

    linhas_mes = []
    for m in range(1, 13):
        sse_b00, sse_m05, n = acc.mes[m]
        linhas_mes.append({"mes": m, "N": n, "B00_RMSE": np.sqrt(sse_b00 / n), "M05_RMSE": np.sqrt(sse_m05 / n),
                            "delta": np.sqrt(sse_m05 / n) - np.sqrt(sse_b00 / n)})
    df_mes = pd.DataFrame(linhas_mes)
    df_mes.to_csv(RESULTS_DIR / "rolling_month.csv", index=False)
    print("\n--- rolling_month.csv ---")
    print(df_mes.to_string(index=False))

    linhas_int = []
    for ib in INT_LABELS:
        sse_b00, sse_m05, n = acc.intensidade[ib]
        linhas_int.append({
            "faixa": ib, "N": n, "pct_observacoes": round(100 * n / n_total, 3),
            "B00_SSE": sse_b00, "M05_SSE": sse_m05,
            "B00_RMSE": np.sqrt(sse_b00 / n) if n else np.nan, "M05_RMSE": np.sqrt(sse_m05 / n) if n else np.nan,
            "pct_SSE_B00": round(100 * sse_b00 / sse_b00_total, 3), "pct_SSE_M05": round(100 * sse_m05 / sse_m05_total, 3),
        })
    df_int = pd.DataFrame(linhas_int)
    df_int.to_csv(RESULTS_DIR / "rolling_intensity.csv", index=False)
    print("\n--- rolling_intensity.csv ---")
    print(df_int.to_string(index=False))

    linhas_lat = []
    for lb in LAT_LABELS:
        sse_b00, sse_m05, n = acc.lat_bin[lb]
        linhas_lat.append({"faixa_lat": lb, "N": n, "B00_RMSE": np.sqrt(sse_b00 / n), "M05_RMSE": np.sqrt(sse_m05 / n),
                            "delta": np.sqrt(sse_m05 / n) - np.sqrt(sse_b00 / n)})
    df_lat = pd.DataFrame(linhas_lat)
    df_lat.to_csv(RESULTS_DIR / "rolling_latitude.csv", index=False)
    print("\n--- rolling_latitude.csv ---")
    print(df_lat.to_string(index=False))

    print("\n--- ano 1 x ano 2 da janela (NAO e horizonte) ---")
    for ano in [1, 2]:
        sse_b00, sse_m05, n = acc.ano[ano]
        print(f"ano {ano}: N={n} | B00_RMSE={np.sqrt(sse_b00/n):.4f} | M05_RMSE={np.sqrt(sse_m05/n):.4f}")

    sse_b00_vetor = np.concatenate(acc.sse_b00_vetor)
    sse_m05_vetor = np.concatenate(acc.sse_m05_vetor)
    conc_b00 = concentracao_sse(sse_b00_vetor)
    conc_m05 = concentracao_sse(sse_m05_vetor)
    df_conc = pd.DataFrame([{"modelo": "B00", **conc_b00}, {"modelo": "M05", **conc_m05}])
    df_conc.to_csv(RESULTS_DIR / "rolling_sse_concentration.csv", index=False)
    print("\n--- rolling_sse_concentration.csv (pooled nos 3 folds) ---")
    print(df_conc.to_string(index=False))

    print("\n--- clipping (pos-processamento, sem retreino) ---")
    print(f"min(y_true) observado em toda a avaliacao: {acc.min_y_true:.6f} "
          f"(todos >=0? {acc.min_y_true >= 0})")
    rmse_m05_clip = np.sqrt(acc.sse_m05_clip_total / acc.n_total)
    print(f"pct previsoes M05 alteradas pelo clip: {100*acc.n_clip_alterado/acc.n_total:.3f}%")
    print(f"RMSE M05 agregado antes do clip: {rmse_m05_agregado:.4f}")
    print(f"RMSE M05 agregado depois do clip: {rmse_m05_clip:.4f}")

    pico_mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # macOS: bytes -> KB -> MB
    tempo_total = time.time() - inicio_geral
    print(f"\n--- performance ---")
    print(f"pico aproximado de memoria (RSS): {pico_mem_mb:.0f} MB")
    print(f"tempo por fold: {tempos_fold}")
    print(f"tempo total: {tempo_total:.1f}s ({tempo_total/60:.1f} min) | "
          f"dentro do time-box de {limite_minutos} min? {tempo_total/60 <= limite_minutos}")
