"""Gate 22 - modelo de producao M05+ONI: treina com todo o historico legitimamente
disponivel ate 2022-12 e gera as previsoes oficiais 2023-2024 (S02).

ONI usa a indexacao corrigida (disponibilidade conservadora - ultimo mes da janela
trimestral), validada no Gate 21.1/21.2. Sem clipping durante a validacao interna;
clipping aplicado apenas na submissao final (Parte D1), pois precipitacao real e
sempre >= 0.
"""

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from lightgbm import LGBMRegressor

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
EXTERNAL_DIR = Path(__file__).resolve().parent.parent / "data" / "external"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SUBMISSIONS_DIR = Path(__file__).resolve().parent.parent / "submissions"

VARS_ATMOSFERICAS = [
    "t2", "cloud_cover", "shum_850", "surface_pressure",
    "u_850", "v_850", "temperature_850", "rel_hum_850", "geopotential_850",
]
VARS_ANOMALIA = [f"{v}_anomalia" for v in VARS_ATMOSFERICAS]
FEATURES = VARS_ATMOSFERICAS + VARS_ANOMALIA + ["lat", "lon", "month_sin", "month_cos", "tp_climatologia", "oni"]

SEED = 42
MESES_AMOSTRADOS = 150
PONTOS_POR_MES = 4000
ORIGEM_REAL = "2022-12-01"

SEAS_CENTRO = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
               "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12}


def carrega_oni_mensal_corrigido():
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
    return float(serie_oni.get(data, np.nan))


def constroi_id(ano, mes, lat_val, lon_val):
    lat_r = round(float(lat_val), 2) + 0.0
    lon_r = round(float(lon_val), 2) + 0.0
    return f"{ano}_{mes:02d}_{lat_r:.2f}_{lon_r:.2f}"


def climatologia_real(tp):
    historico = tp.sel(time=slice(None, ORIGEM_REAL))
    return historico.groupby("time.month").mean("time").values.astype(np.float32)


def mes_alvo(t_idx, tempos):
    return (pd.Timestamp(tempos[t_idx]).month % 12) + 1


def monta_treino(tp_alvo, atmosfericas, clim_tp, clim_atm, serie_oni, lat, lon, tempos, seed=SEED):
    elegiveis = np.where(tempos < np.datetime64(ORIGEM_REAL))[0]
    rng = np.random.default_rng(seed)
    n_meses = min(MESES_AMOSTRADOS, len(elegiveis))
    meses_idx = np.sort(rng.choice(elegiveis, size=n_meses, replace=False))

    max_target = pd.Timestamp(tempos[meses_idx[-1]]) + pd.DateOffset(months=1)
    print(f"meses elegiveis: {len(elegiveis)} | sorteados: {n_meses} | "
          f"ultimo target_date treino: {max_target.date()} (cutoff={ORIGEM_REAL})")
    assert max_target <= pd.Timestamp(ORIGEM_REAL)

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
    print(f"ONI no treino: {round(100*X['oni'].isnull().mean(),3)}% NaN")
    return X, y_res


def gera_predicoes(teste, clim_tp, clim_atm, serie_oni, lat, lon, modelo):
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    lat_flat = lat_grid.ravel().astype(np.float32)
    lon_flat = lon_grid.ravel().astype(np.float32)

    linhas = []
    for t_idx in range(teste.sizes["time"]):
        time_val = pd.Timestamp(teste.time.isel(time=t_idx).values)
        origem_val = pd.Timestamp(teste.time_origem.isel(time=t_idx).values)
        mes_target = time_val.month
        mes_origem = origem_val.month

        clim_flat = clim_tp[mes_target - 1].ravel()
        month_sin = np.full(lat_flat.shape, np.sin(2 * np.pi * mes_origem / 12), dtype=np.float32)
        month_cos = np.full(lat_flat.shape, np.cos(2 * np.pi * mes_origem / 12), dtype=np.float32)
        oni_flat = np.full(lat_flat.shape, oni_do_mes(serie_oni, origem_val), dtype=np.float32)

        dados = {"lat": lat_flat, "lon": lon_flat, "month_sin": month_sin, "month_cos": month_cos,
                 "tp_climatologia": clim_flat, "oni": oni_flat}
        for v in VARS_ATMOSFERICAS:
            slice_v = teste[v].isel(time=t_idx).values.astype(np.float32)
            clim_v_flat = clim_atm[v][mes_origem - 1].ravel()
            dados[v] = slice_v.ravel()
            dados[f"{v}_anomalia"] = slice_v.ravel() - clim_v_flat
        X_mes = pd.DataFrame(dados)[FEATURES]

        residuo = modelo.predict(X_mes)
        tp_predito = clim_flat + residuo

        ids = [constroi_id(time_val.year, time_val.month, la, lo) for la, lo in zip(lat_flat, lon_flat)]
        linhas.append(pd.DataFrame({"id": ids, "tp_mm_day": tp_predito.astype(np.float32)}))

        del X_mes, residuo, tp_predito
        gc.collect()

    return pd.concat(linhas, ignore_index=True)


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    SUBMISSIONS_DIR.mkdir(exist_ok=True)

    tp = xr.open_dataset(RAW_DIR / "treino_tp.nc")["tp"]
    tp_alvo = xr.open_dataset(RAW_DIR / "treino_tp_alvo.nc")["tp_alvo"]
    atmosfericas_treino = {v: xr.open_dataset(RAW_DIR / f"treino_{v}.nc")[v] for v in VARS_ATMOSFERICAS}
    teste = xr.open_dataset(RAW_DIR / "teste_features.nc")
    sample = pd.read_csv(RAW_DIR / "sample_submission.csv")
    lat = tp_alvo["lat"].values
    lon = tp_alvo["lon"].values
    tempos = tp_alvo["time"].values
    serie_oni = carrega_oni_mensal_corrigido()

    print("=== climatologias de producao (historico completo ate 2022-12) ===")
    clim_tp = climatologia_real(tp)
    clim_atm = {v: climatologia_real(atmosfericas_treino[v]) for v in VARS_ATMOSFERICAS}

    print("\n=== treino final M05+ONI ===")
    X, y_res = monta_treino(tp_alvo, atmosfericas_treino, clim_tp, clim_atm, serie_oni, lat, lon, tempos)
    print(f"amostras: {len(X):,} | features: {len(FEATURES)} (esperado 24)")
    assert len(FEATURES) == 24

    modelo = LGBMRegressor(n_estimators=300, num_leaves=15, learning_rate=0.05, random_state=SEED, verbosity=-1)
    modelo.fit(X, y_res)
    del X, y_res
    gc.collect()

    print("\n=== inferencia 2023-2024 ===")
    pred = gera_predicoes(teste, clim_tp, clim_atm, serie_oni, lat, lon, modelo)
    del modelo
    gc.collect()

    s02 = sample[["id"]].merge(pred, on="id", how="left")
    n_faltando = int(s02["tp_mm_day"].isnull().sum())
    n_dup = int(pred["id"].duplicated().sum())

    print("\n=== D1. clipping ===")
    n_neg_antes = int((s02["tp_mm_day"] < 0).sum())
    min_antes = float(s02["tp_mm_day"].min())
    s02["tp_mm_day"] = s02["tp_mm_day"].clip(lower=0)
    print(f"previsoes negativas antes do clip: {n_neg_antes} ({100*n_neg_antes/len(s02):.3f}%) | "
          f"minimo antes: {min_antes:.4f} | minimo depois: {s02['tp_mm_day'].min():.4f}")

    print("\n=== validacao estrutural S02 ===")
    ids_iguais = bool((s02["id"].values == sample["id"].values).all())
    n_nan = int(s02["tp_mm_day"].isnull().sum())
    n_inf = int(np.isinf(s02["tp_mm_day"].values).sum())
    print(f"linhas: {len(s02)} (esperado 1885464, ok={len(s02)==1_885_464})")
    print(f"ids identicos e na mesma ordem: {ids_iguais} | duplicados: {n_dup} | faltando: {n_faltando}")
    print(f"NaN: {n_nan} | inf: {n_inf} | dtype: {s02['tp_mm_day'].dtype}")

    for idx, label in [(0, "primeira"), (len(sample)//2, "intermediaria"), (len(sample)-1, "ultima")]:
        print(f"{label}: id={s02['id'].iloc[idx]} valor={s02['tp_mm_day'].iloc[idx]:.4f}")

    print("\n=== distribuicao S02 ===")
    print(s02["tp_mm_day"].describe())

    s02[["id", "tp_mm_day"]].to_csv(SUBMISSIONS_DIR / "s02_m05_oni.csv", index=False)
    tamanho_mb = (SUBMISSIONS_DIR / "s02_m05_oni.csv").stat().st_size / 1e6
    print(f"\ns02_m05_oni.csv salvo: {tamanho_mb:.1f} MB")
    print(f"pronta para upload? {'SIM' if (len(s02)==1_885_464 and ids_iguais and n_dup==0 and n_nan==0 and n_inf==0) else 'NAO'}")
