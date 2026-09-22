"""Gate 23 - CENSO: climatologia mensal espacial condicionada ao regime ENSO, comparada
com B00 (climatologia tradicional) e M05+ONI (reaproveitado sem alteracao, para gerar as
mesmas quebras de diagnostico - regime/mes/latitude/intensidade - em um unico passe).

Regime ENSO classificado com regra fixa sobre o ONI (indexacao corrigida do Gate 21.1,
disponibilidade conservadora = ultimo mes da janela trimestral):
  El Nino: ONI >= +0.5 | La Nina: ONI <= -0.5 | Neutral: caso contrario.

A climatologia CENSO de cada mes-calendario M usa, para cada ocorrencia historica de M,
o regime vigente na origem dessa propria ocorrencia (M-1) - a mesma convencao usada na
previsao (secao 7), aplicada tambem na construcao do historico (secao 6).
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
MIN_SAMPLES_REGIME = 5
K_SHRINK = 10

FOLDS = [
    {"nome": "F1", "cutoff": "2016-12-01", "eval_start": "2017-01-01", "eval_end": "2018-12-01"},
    {"nome": "F2", "cutoff": "2018-12-01", "eval_start": "2019-01-01", "eval_end": "2020-12-01"},
    {"nome": "F3", "cutoff": "2020-12-01", "eval_start": "2021-01-01", "eval_end": "2022-12-01"},
]

REGIMES = ["ElNino", "Neutral", "LaNina"]
LAT_EDGES = [-60, -45, -30, -15, 0, 15]
LAT_LABELS = ["-60_-45", "-45_-30", "-30_-15", "-15_0", "0_15"]
INT_EDGES = [1, 3, 5, 10, 20]
INT_LABELS = ["0-1", "1-3", "3-5", "5-10", "10-20", ">20"]

SEAS_CENTRO = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
               "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12}


def carrega_oni_disponibilidade():
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
    return df.set_index("disponibilidade")["ANOM"]


def oni_do_mes(serie_oni, data):
    data = pd.Timestamp(data.year, data.month, 1)
    return serie_oni.get(data, np.nan)


def classifica_regime(valor_oni):
    if pd.isna(valor_oni):
        return None
    if valor_oni >= 0.5:
        return "ElNino"
    if valor_oni <= -0.5:
        return "LaNina"
    return "Neutral"


def mes_alvo(t_idx, tempos):
    return (pd.Timestamp(tempos[t_idx]).month % 12) + 1


def climatologia_ate(da, cutoff):
    historico = da.sel(time=slice(None, cutoff))
    return historico.groupby("time.month").mean("time").values.astype(np.float32)


def constroi_censo(tp, cutoff, serie_oni):
    """Climatologia B00 (12,lat,lon) e CENSO {(mes,regime): grid}, com contagem de anos."""
    hist = tp.sel(time=slice(None, cutoff))
    tempos_hist = hist.time.values
    valores = hist.values.astype(np.float32)  # (n_tempo, lat, lon)

    clim_b00 = hist.groupby("time.month").mean("time").values.astype(np.float32)

    clim_censo = {}
    n_censo = {}
    for mes in range(1, 13):
        for regime in REGIMES:
            idx = []
            for i, t in enumerate(tempos_hist):
                data_t = pd.Timestamp(t)
                if data_t.month != mes:
                    continue
                origem = data_t - pd.DateOffset(months=1)
                r = classifica_regime(oni_do_mes(serie_oni, origem))
                if r == regime:
                    idx.append(i)
            n_censo[(mes, regime)] = len(idx)
            if len(idx) > 0:
                clim_censo[(mes, regime)] = valores[idx].mean(axis=0)
            else:
                clim_censo[(mes, regime)] = clim_b00[mes - 1]  # fallback ja embutido na construcao

    return clim_b00, clim_censo, n_censo


def monta_treino_oni(cutoff, tp_alvo, atmosfericas, clim_tp, clim_atm, serie_oni, lat, lon, tempos, seed=SEED):
    """Identico ao Gate 21.2 - reutilizado sem alteracao para gerar o M05+ONI de comparacao."""
    elegiveis = np.where(tempos < np.datetime64(cutoff))[0]
    rng = np.random.default_rng(seed)
    n_meses = min(MESES_AMOSTRADOS, len(elegiveis))
    meses_idx = np.sort(rng.choice(elegiveis, size=n_meses, replace=False))

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
    return X, y_res


class Acumulador:
    def __init__(self):
        self.fold = {m: {} for m in ["b00", "censo", "censo_shrink", "oni"]}
        self.regime = {r: {m: [0.0, 0] for m in ["b00", "censo", "oni"]} for r in REGIMES}
        self.mes = {mm: {m: [0.0, 0] for m in ["b00", "censo", "oni"]} for mm in range(1, 13)}
        self.lat_bin = {lb: {m: [0.0, 0] for m in ["b00", "censo", "oni"]} for lb in LAT_LABELS}
        self.intensidade = {ib: {m: [0.0, 0] for m in ["b00", "censo", "oni"]} for ib in INT_LABELS}
        self.n_fallback = 0
        self.n_total_censo = 0

    def registra(self, nome_fold, regime, mes_target, lat_flat, y_true, preds, usou_fallback):
        for modelo, pred in preds.items():
            sq = ((y_true - pred) ** 2).astype(np.float64)
            self.fold[modelo].setdefault(nome_fold, [0.0, 0])
            self.fold[modelo][nome_fold][0] += float(sq.sum())
            self.fold[modelo][nome_fold][1] += len(sq)

            if modelo in ("b00", "censo", "oni"):
                if regime is not None:
                    self.regime[regime][modelo][0] += float(sq.sum())
                    self.regime[regime][modelo][1] += len(sq)
                self.mes[mes_target][modelo][0] += float(sq.sum())
                self.mes[mes_target][modelo][1] += len(sq)

                idx_lat = np.digitize(lat_flat, LAT_EDGES[1:-1])
                for i, lb in enumerate(LAT_LABELS):
                    mask = idx_lat == i
                    if mask.any():
                        self.lat_bin[lb][modelo][0] += float(sq[mask].sum())
                        self.lat_bin[lb][modelo][1] += int(mask.sum())

                idx_int = np.digitize(y_true, INT_EDGES)
                for i, ib in enumerate(INT_LABELS):
                    mask = idx_int == i
                    if mask.any():
                        self.intensidade[ib][modelo][0] += float(sq[mask].sum())
                        self.intensidade[ib][modelo][1] += int(mask.sum())

        self.n_total_censo += len(y_true)
        if usou_fallback:
            self.n_fallback += len(y_true)


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    inicio = time.time()

    tp = xr.open_dataset(RAW_DIR / "treino_tp.nc")["tp"]
    tp_alvo = xr.open_dataset(RAW_DIR / "treino_tp_alvo.nc")["tp_alvo"]
    atmosfericas = {v: xr.open_dataset(RAW_DIR / f"treino_{v}.nc")[v] for v in VARS_ATMOSFERICAS}
    lat = tp_alvo["lat"].values
    lon = tp_alvo["lon"].values
    tempos = tp_alvo["time"].values
    serie_oni = carrega_oni_disponibilidade()

    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    lat_flat_g = lat_grid.ravel().astype(np.float32)
    lon_flat_g = lon_grid.ravel().astype(np.float32)

    acc = Acumulador()
    linhas_cobertura = []

    for fold in FOLDS:
        t0 = time.time()
        nome, cutoff, eval_start, eval_end = fold["nome"], fold["cutoff"], fold["eval_start"], fold["eval_end"]
        print(f"\n########## {nome} (cutoff={cutoff}) ##########")

        print("construindo climatologias B00 e CENSO...")
        clim_b00, clim_censo, n_censo = constroi_censo(tp, cutoff, serie_oni)
        for (mes, regime), n in n_censo.items():
            linhas_cobertura.append({"fold": nome, "mes": mes, "regime": regime, "n_anos": n})

        clim_atm = {v: climatologia_ate(atmosfericas[v], cutoff) for v in VARS_ATMOSFERICAS}

        print("treinando M05+ONI (infraestrutura reaproveitada do Gate 21.2, sem alteracao)...")
        X, y_res = monta_treino_oni(cutoff, tp_alvo, atmosfericas, clim_b00, clim_atm, serie_oni, lat, lon, tempos)
        modelo_oni = LGBMRegressor(n_estimators=300, num_leaves=15, learning_rate=0.05,
                                    random_state=SEED, verbosity=-1)
        modelo_oni.fit(X, y_res)
        del X, y_res
        gc.collect()

        origens_avaliacao = pd.date_range(eval_start, eval_end, freq="MS")
        for target in origens_avaliacao:
            origem_val = target - pd.DateOffset(months=1)
            t_idx = int(np.where(tempos == np.datetime64(origem_val))[0][0])
            mes_origem = origem_val.month
            mes_target = target.month

            oni_val = oni_do_mes(serie_oni, origem_val)
            regime = classifica_regime(oni_val)

            clim_flat_b00 = clim_b00[mes_target - 1].ravel()
            n_regime = n_censo.get((mes_target, regime), 0) if regime else 0
            usar_fallback = (regime is None) or (n_regime < MIN_SAMPLES_REGIME)
            if usar_fallback:
                clim_flat_censo = clim_flat_b00
            else:
                clim_flat_censo = clim_censo[(mes_target, regime)].ravel()

            w = n_regime / (n_regime + K_SHRINK)
            clim_flat_shrink = w * clim_censo.get((mes_target, regime), clim_b00[mes_target - 1]).ravel() + \
                (1 - w) * clim_flat_b00

            month_sin = np.full(lat_flat_g.shape, np.sin(2 * np.pi * mes_origem / 12), dtype=np.float32)
            month_cos = np.full(lat_flat_g.shape, np.cos(2 * np.pi * mes_origem / 12), dtype=np.float32)
            oni_flat = np.full(lat_flat_g.shape, oni_val if not pd.isna(oni_val) else np.nan, dtype=np.float32)

            dados = {"lat": lat_flat_g, "lon": lon_flat_g, "month_sin": month_sin, "month_cos": month_cos,
                     "tp_climatologia": clim_flat_b00, "oni": oni_flat}
            for v in VARS_ATMOSFERICAS:
                slice_v = atmosfericas[v].isel(time=t_idx).values.astype(np.float32)
                clim_v_flat = clim_atm[v][mes_origem - 1].ravel()
                dados[v] = slice_v.ravel()
                dados[f"{v}_anomalia"] = slice_v.ravel() - clim_v_flat
            X_mes = pd.DataFrame(dados)[FEATURES]

            y_true = tp_alvo.isel(time=t_idx).values.ravel()
            pred_oni = clim_flat_b00 + modelo_oni.predict(X_mes)

            preds = {"b00": clim_flat_b00, "censo": clim_flat_censo, "censo_shrink": clim_flat_shrink,
                     "oni": pred_oni}
            acc.registra(nome, regime, mes_target, lat_flat_g, y_true, preds, usar_fallback)

            del X_mes, pred_oni, y_true
            gc.collect()

        del modelo_oni, clim_atm
        gc.collect()
        print(f"tempo do fold: {time.time()-t0:.1f}s")

    # ---------- consolidacao ----------
    print("\n\n=== COBERTURA HISTORICA (mes x regime x fold) ===")
    df_cob = pd.DataFrame(linhas_cobertura)
    df_cob.to_csv(RESULTS_DIR / "censo_sample_counts.csv", index=False)
    resumo_cob = df_cob.groupby(["mes", "regime"])["n_anos"].agg(["min", "median", "max"]).reset_index()
    print(resumo_cob.to_string(index=False))

    pct_fallback = 100 * acc.n_fallback / acc.n_total_censo
    print(f"\nfallback (n_regime < {MIN_SAMPLES_REGIME} ou regime indisponivel): "
          f"{acc.n_fallback}/{acc.n_total_censo} = {pct_fallback:.3f}%")

    print("\n=== RESULTADO PRINCIPAL POR FOLD ===")
    linhas_fold = []
    sse_tot = {m: 0.0 for m in ["b00", "censo", "censo_shrink", "oni"]}
    n_tot = {m: 0 for m in ["b00", "censo", "censo_shrink", "oni"]}
    for fold in FOLDS:
        nome = fold["nome"]
        linha = {"fold": nome, "cutoff": fold["cutoff"]}
        for modelo in ["b00", "censo", "censo_shrink", "oni"]:
            sse, n = acc.fold[modelo][nome]
            rmse = np.sqrt(sse / n)
            linha[f"{modelo}_RMSE"] = rmse
            sse_tot[modelo] += sse
            n_tot[modelo] += n
        linha["delta_censo_b00"] = linha["censo_RMSE"] - linha["b00_RMSE"]
        linha["delta_censo_oni"] = linha["censo_RMSE"] - linha["oni_RMSE"]
        linhas_fold.append(linha)

    linha_agg = {"fold": "AGREGADO", "cutoff": ""}
    for modelo in ["b00", "censo", "censo_shrink", "oni"]:
        linha_agg[f"{modelo}_RMSE"] = np.sqrt(sse_tot[modelo] / n_tot[modelo])
    linha_agg["delta_censo_b00"] = linha_agg["censo_RMSE"] - linha_agg["b00_RMSE"]
    linha_agg["delta_censo_oni"] = linha_agg["censo_RMSE"] - linha_agg["oni_RMSE"]
    linhas_fold.append(linha_agg)

    df_folds = pd.DataFrame(linhas_fold)
    df_folds.to_csv(RESULTS_DIR / "censo_folds.csv", index=False)
    print(df_folds.to_string(index=False))

    deltas_por_fold = [linhas_fold[i]["delta_censo_b00"] for i in range(3)]
    print(f"\ndelta CENSO-B00 por fold: {[round(d,4) for d in deltas_por_fold]}")

    print("\n=== RESULTADO POR REGIME ===")
    linhas_regime = []
    for r in REGIMES:
        linha = {"regime": r}
        for modelo in ["b00", "censo", "oni"]:
            sse, n = acc.regime[r][modelo]
            linha[f"{modelo}_RMSE"] = np.sqrt(sse / n) if n else np.nan
            linha["N"] = n
        linha["delta_censo_b00"] = linha["censo_RMSE"] - linha["b00_RMSE"]
        linha["delta_oni_b00"] = linha["oni_RMSE"] - linha["b00_RMSE"]
        linhas_regime.append(linha)
    df_regime = pd.DataFrame(linhas_regime)
    df_regime.to_csv(RESULTS_DIR / "censo_regimes.csv", index=False)
    print(df_regime.to_string(index=False))

    print("\n=== RESULTADO POR MES ===")
    linhas_mes = []
    for mm in range(1, 13):
        linha = {"mes": mm}
        for modelo in ["b00", "censo", "oni"]:
            sse, n = acc.mes[mm][modelo]
            linha[f"{modelo}_RMSE"] = np.sqrt(sse / n)
        linhas_mes.append(linha)
    df_mes = pd.DataFrame(linhas_mes)
    df_mes.to_csv(RESULTS_DIR / "censo_month.csv", index=False)
    print(df_mes.to_string(index=False))

    print("\n=== RESULTADO POR LATITUDE ===")
    linhas_lat = []
    for lb in LAT_LABELS:
        linha = {"faixa_lat": lb}
        for modelo in ["b00", "censo", "oni"]:
            sse, n = acc.lat_bin[lb][modelo]
            linha[f"{modelo}_RMSE"] = np.sqrt(sse / n)
        linhas_lat.append(linha)
    df_lat = pd.DataFrame(linhas_lat)
    df_lat.to_csv(RESULTS_DIR / "censo_latitude.csv", index=False)
    print(df_lat.to_string(index=False))

    print("\n=== RESULTADO POR INTENSIDADE ===")
    linhas_int = []
    for ib in INT_LABELS:
        linha = {"faixa": ib}
        for modelo in ["b00", "censo", "oni"]:
            sse, n = acc.intensidade[ib][modelo]
            linha[f"{modelo}_RMSE"] = np.sqrt(sse / n)
            linha["N"] = n
        linhas_int.append(linha)
    df_int = pd.DataFrame(linhas_int)
    df_int.to_csv(RESULTS_DIR / "censo_intensity.csv", index=False)
    print(df_int.to_string(index=False))

    print(f"\ntempo total: {time.time()-inicio:.1f}s ({(time.time()-inicio)/60:.1f} min)")
