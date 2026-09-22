"""Gate 21 (corrigido) - M05+ONI com alinhamento temporal correto do ONI, identificado
pela auditoria do Gate 21.1: o valor de ONI usado para o mes de origem M agora e o da
janela trimestral cujo ULTIMO mes e M (disponibilidade conservadora), nao mais o valor
centrado em M (que incluia o mes-alvo M+1 - leakage).

Mesma estrutura do Gate 19/21: 3 folds rolling-origin, climatologias recalculadas e
congeladas por fold, mesma amostragem/seed/config do M03-T01. Nenhuma outra mudanca.

Tambem acumula, no mesmo passe, o diagnostico de complementaridade entre os residuos
de B00 e M05+ONI (Parte G): correlacao de Pearson, media/desvio dos residuos, RMSE de
cada um e percentual de casos em que cada modelo tem menor erro absoluto.
"""

import gc
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

B00_RMSE_FOLD = {"F1": 1.860401, "F2": 1.867259, "F3": 1.893772}
B00_RMSE_AGREGADO = 1.873866

SEAS_CENTRO = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
               "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12}


def carrega_oni_mensal_corrigido():
    """Indexa cada valor de ONI pela data de disponibilidade conservadora (ultimo mes
    da janela trimestral), nao pelo mes central - corrige o leakage do Gate 21.1."""
    df = pd.read_csv(EXTERNAL_DIR / "oni_raw.txt", sep=r"\s+")
    datas_disp = []
    for _, row in df.iterrows():
        centro = SEAS_CENTRO[row["SEAS"]]
        yr = int(row["YR"])
        if centro == 12:
            mes3, ano3 = 1, yr + 1
        else:
            mes3, ano3 = centro + 1, yr
        datas_disp.append(pd.Timestamp(ano3, mes3, 1))
    df["disponibilidade"] = datas_disp
    serie = df.set_index("disponibilidade")["ANOM"]
    print(f"ONI (corrigido) carregado: {serie.index.min().date()} -> {serie.index.max().date()} "
          f"({len(serie)} meses); indexado por disponibilidade conservadora (ultimo mes da janela)")
    return serie


def oni_do_mes(serie_oni, data):
    data = pd.Timestamp(data.year, data.month, 1)
    return float(serie_oni.get(data, np.nan))


def climatologia_ate(da, cutoff):
    historico = da.sel(time=slice(None, cutoff))
    return historico.groupby("time.month").mean("time").values.astype(np.float32)


def mes_alvo(t_idx, tempos):
    return (pd.Timestamp(tempos[t_idx]).month % 12) + 1


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
    print(f"  ONI no treino: {pct_nan_oni}% NaN")
    return X, y_res


class AcumuladorDiagnostico:
    """Acumula, em passe unico: RMSE de B00 e M05+ONI, e estatisticas de complementaridade
    dos residuos (correlacao de Pearson via somas streaming, sem guardar vetores completos)."""

    def __init__(self):
        self.fold = {}
        self.soma_e_b00 = 0.0
        self.soma_e_oni = 0.0
        self.soma_e_b00_sq = 0.0
        self.soma_e_oni_sq = 0.0
        self.soma_e_b00_e_oni = 0.0
        self.n_menor_erro_b00 = 0
        self.n_menor_erro_oni = 0
        self.n = 0

    def registra(self, nome_fold, y_true, pred_b00, pred_oni):
        e_b00 = y_true - pred_b00
        e_oni = y_true - pred_oni

        if nome_fold not in self.fold:
            self.fold[nome_fold] = {"b00": [0.0, 0], "oni": [0.0, 0]}
        self.fold[nome_fold]["b00"][0] += float((e_b00 ** 2).sum())
        self.fold[nome_fold]["b00"][1] += len(e_b00)
        self.fold[nome_fold]["oni"][0] += float((e_oni ** 2).sum())
        self.fold[nome_fold]["oni"][1] += len(e_oni)

        self.soma_e_b00 += float(e_b00.sum())
        self.soma_e_oni += float(e_oni.sum())
        self.soma_e_b00_sq += float((e_b00 ** 2).sum())
        self.soma_e_oni_sq += float((e_oni ** 2).sum())
        self.soma_e_b00_e_oni += float((e_b00 * e_oni).sum())

        self.n_menor_erro_b00 += int((np.abs(e_b00) < np.abs(e_oni)).sum())
        self.n_menor_erro_oni += int((np.abs(e_oni) < np.abs(e_b00)).sum())
        self.n += len(e_b00)

    def pearson(self):
        n = self.n
        media_b00 = self.soma_e_b00 / n
        media_oni = self.soma_e_oni / n
        cov = self.soma_e_b00_e_oni / n - media_b00 * media_oni
        var_b00 = self.soma_e_b00_sq / n - media_b00 ** 2
        var_oni = self.soma_e_oni_sq / n - media_oni ** 2
        return cov / np.sqrt(var_b00 * var_oni)


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    inicio = time.time()

    tp = xr.open_dataset(RAW_DIR / "treino_tp.nc")["tp"]
    tp_alvo = xr.open_dataset(RAW_DIR / "treino_tp_alvo.nc")["tp_alvo"]
    atmosfericas = {v: xr.open_dataset(RAW_DIR / f"treino_{v}.nc")[v] for v in VARS_ATMOSFERICAS}
    lat = tp_alvo["lat"].values
    lon = tp_alvo["lon"].values
    tempos = tp_alvo["time"].values
    serie_oni = carrega_oni_mensal_corrigido()

    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    lat_flat_global = lat_grid.ravel().astype(np.float32)
    lon_flat_global = lon_grid.ravel().astype(np.float32)

    acc = AcumuladorDiagnostico()
    tempos_fold = {}

    for fold in FOLDS:
        t0 = time.time()
        nome, cutoff, eval_start, eval_end = fold["nome"], fold["cutoff"], fold["eval_start"], fold["eval_end"]
        print(f"\n########## {nome} (cutoff={cutoff}) ##########")

        clim_tp = climatologia_ate(tp, cutoff)
        clim_atm = {v: climatologia_ate(atmosfericas[v], cutoff) for v in VARS_ATMOSFERICAS}

        X, y_res = monta_treino(cutoff, tp_alvo, atmosfericas, clim_tp, clim_atm, serie_oni, lat, lon, tempos)
        assert len(FEATURES) == 24

        modelo = LGBMRegressor(n_estimators=300, num_leaves=15, learning_rate=0.05,
                                random_state=SEED, verbosity=-1)
        modelo.fit(X, y_res)
        importancias = pd.Series(modelo.feature_importances_, index=FEATURES).sort_values(ascending=False)
        print(f"  importancia do oni (corrigido): {importancias.get('oni', 0)} "
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
            pred_oni = clim_flat + modelo.predict(X_mes)

            acc.registra(nome, y_true, pred_b00, pred_oni)

            del X_mes, pred_oni, y_true
            gc.collect()

        del modelo, clim_tp, clim_atm
        gc.collect()
        tempos_fold[nome] = time.time() - t0
        sse_oni, n_oni = acc.fold[nome]["oni"]
        rmse_oni_fold = np.sqrt(sse_oni / n_oni)
        print(f"  RMSE M05+ONI corrigido ({nome}): {rmse_oni_fold:.4f} | B00: {B00_RMSE_FOLD[nome]:.4f} | "
              f"tempo: {tempos_fold[nome]:.1f}s")

    print("\n\n=== CONSOLIDACAO M05+ONI (corrigido) ===")
    linhas = []
    sse_b00_total, sse_oni_total, n_total = 0.0, 0.0, 0
    for fold in FOLDS:
        nome = fold["nome"]
        sse_b00, n_b00 = acc.fold[nome]["b00"]
        sse_oni, n_oni = acc.fold[nome]["oni"]
        rmse_b00 = np.sqrt(sse_b00 / n_b00)
        rmse_oni = np.sqrt(sse_oni / n_oni)
        linhas.append({"fold": nome, "cutoff": fold["cutoff"], "B00_RMSE": rmse_b00, "M05_ONI_RMSE": rmse_oni,
                        "delta_ONI_menos_B00": rmse_oni - rmse_b00})
        sse_b00_total += sse_b00
        sse_oni_total += sse_oni
        n_total += n_b00

    rmse_b00_agg = np.sqrt(sse_b00_total / n_total)
    rmse_oni_agg = np.sqrt(sse_oni_total / n_total)
    linhas.append({"fold": "AGREGADO", "cutoff": "", "B00_RMSE": rmse_b00_agg, "M05_ONI_RMSE": rmse_oni_agg,
                    "delta_ONI_menos_B00": rmse_oni_agg - rmse_b00_agg})
    df_final = pd.DataFrame(linhas)
    df_final.to_csv(RESULTS_DIR / "rolling_oni_folds_corrigido.csv", index=False)
    print(df_final.to_string(index=False))

    delta_pct = 100 * (rmse_oni_agg - rmse_b00_agg) / rmse_b00_agg
    print(f"\ndelta agregado ONI-B00: {rmse_oni_agg - rmse_b00_agg:+.4f} ({delta_pct:+.3f}%)")

    # diagnostico de complementaridade (Parte G)
    corr = acc.pearson()
    rmse_b00_total = np.sqrt(acc.soma_e_b00_sq / acc.n)
    rmse_oni_total = np.sqrt(acc.soma_e_oni_sq / acc.n)
    diag = {
        "correlacao_pearson_residuos": corr,
        "media_residuo_b00": acc.soma_e_b00 / acc.n, "media_residuo_oni": acc.soma_e_oni / acc.n,
        "rmse_b00": rmse_b00_total, "rmse_oni": rmse_oni_total,
        "pct_menor_erro_abs_b00": round(100 * acc.n_menor_erro_b00 / acc.n, 3),
        "pct_menor_erro_abs_oni": round(100 * acc.n_menor_erro_oni / acc.n, 3),
    }
    print("\n=== Parte G: diagnostico de complementaridade B00 x M05+ONI ===")
    for k, v in diag.items():
        print(f"{k}: {v}")
    pd.DataFrame([diag]).to_csv(RESULTS_DIR / "ensemble_diagnostic.csv", index=False)

    print(f"\ntempo total: {time.time()-inicio:.1f}s ({(time.time()-inicio)/60:.1f} min)")
