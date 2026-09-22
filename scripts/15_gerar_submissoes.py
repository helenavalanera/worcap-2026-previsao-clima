"""Gate 15 - pipeline de inferencia oficial (2023-2024) e geracao das submissoes diagnosticas.

S00: climatologia mensal espacial (B00), calculada com todo o historico legitimamente
disponivel ate a origem real da competicao (dez/2022).

S01: M03 consolidado (residuo em relacao a climatologia), treinado com 600k amostras
(150 meses x 4000 pontos/mes, seed=42) sorteadas do mesmo universo elegivel, e aplicado
aos 24 meses oficiais de teste.

O id de cada linha e reconstruido a partir de time/lat/lon e alinhado contra
sample_submission.csv por merge - a ordem final e sempre a do sample, nunca assumida
por flatten().
"""

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from lightgbm import LGBMRegressor

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SUBMISSIONS_DIR = Path(__file__).resolve().parent.parent / "submissions"

VARS_ATMOSFERICAS = [
    "t2", "cloud_cover", "shum_850", "surface_pressure",
    "u_850", "v_850", "temperature_850", "rel_hum_850", "geopotential_850",
]
FEATURES = VARS_ATMOSFERICAS + ["lat", "lon", "month_sin", "month_cos", "tp_climatologia"]

SEED = 42
MESES_AMOSTRADOS = 150
PONTOS_POR_MES = 4000  # 150 x 4000 = 600.000
ORIGEM_REAL = "2022-12-01"


def constroi_id(ano, mes, lat_val, lon_val):
    lat_r = round(float(lat_val), 2) + 0.0
    lon_r = round(float(lon_val), 2) + 0.0
    return f"{ano}_{mes:02d}_{lat_r:.2f}_{lon_r:.2f}"


# ---------- 1. protocolo temporal ----------

def confirma_protocolo(teste, atmosfericas_treino):
    print("=== confirmacao do protocolo temporal (treino vs teste) ===")
    tempos_t = pd.to_datetime(teste.time.values)
    origens_t = pd.to_datetime(teste.time_origem.values)
    diffs = [(t.year - o.year) * 12 + (t.month - o.month) for t, o in zip(tempos_t, origens_t)]
    print(f"time_origem = time - 1 mes, para todas as {len(diffs)} linhas? {set(diffs) == {1}}")

    for var in ["t2", "rel_hum_850"]:
        teste_val = float(teste[var].isel(time=0).mean())
        treino_val = float(atmosfericas_treino[var].sel(time="2022-12-01").mean())
        print(f"{var}: teste(time=2023-01) media espacial = {teste_val:.4f} | "
              f"treino(time_origem=2022-12) media espacial = {treino_val:.4f} | "
              f"identico? {abs(teste_val - treino_val) < 1e-4}")
    print("conclusao: features atmosfericas do teste(T) equivalem a features(M=time_origem) do "
          "treino, e M -> M+1=T e a mesma relacao usada no treino. Nenhum shift adicional aplicado.\n")


# ---------- 2. auditoria do periodo oficial ----------

def audita_periodo(teste, sample):
    print("=== auditoria do periodo oficial de teste ===")
    n_time, n_lat, n_lon = teste.sizes["time"], teste.sizes["lat"], teste.sizes["lon"]
    print(f"meses: {n_time} ({str(teste.time.min().values)[:10]} -> {str(teste.time.max().values)[:10]})")
    print(f"lat: {n_lat} | lon: {n_lon} | celulas/mes: {n_lat * n_lon}")
    total_esperado = n_time * n_lat * n_lon
    print(f"total esperado: {total_esperado:,}")
    print(f"linhas em sample_submission.csv: {len(sample):,}")
    assert total_esperado == len(sample) == 1_885_464, "contagem nao bate com o esperado"
    print("OK: bate com 1.885.464 e com o sample_submission.\n")


# ---------- 3. climatologia (S00 e feature do S01) ----------

def climatologia_real(tp):
    historico = tp.sel(time=slice(None, ORIGEM_REAL))
    print(f"climatologia calculada com historico: inicio -> {ORIGEM_REAL} ({historico.sizes['time']} meses)")
    return historico.groupby("time.month").mean("time").values  # (12, lat, lon)


# ---------- 4. amostragem de treino (mesma estrategia do Gate 13/14) ----------

def mes_alvo(t_idx, tempos):
    return (pd.Timestamp(tempos[t_idx]).month % 12) + 1


def monta_treino(tp_alvo, atmosfericas, clim, lat, lon, tempos, seed=SEED):
    elegiveis = np.where(tempos < np.datetime64(ORIGEM_REAL))[0]
    print(f"meses elegiveis para treino (time < {ORIGEM_REAL}): {len(elegiveis)} "
          f"({str(tempos[elegiveis[0]])[:10]} -> {str(tempos[elegiveis[-1]])[:10]})")

    rng = np.random.default_rng(seed)
    n_meses = min(MESES_AMOSTRADOS, len(elegiveis))
    meses_idx = np.sort(rng.choice(elegiveis, size=n_meses, replace=False))
    print(f"meses sorteados (seed={seed}): {n_meses} de {len(elegiveis)} elegiveis "
          f"({100 * n_meses / len(elegiveis):.1f}%)")
    print(f"pontos por mes sorteados (seed continua da mesma sequencia): {PONTOS_POR_MES} de "
          f"{len(lat) * len(lon):,} celulas da grade")
    print("nenhuma observacao futura entra: todo mes sorteado e < 2022-12, e o alvo de cada "
          "linha (mes+1) nunca ultrapassa 2022-12, a ultima precipitacao real conhecida.\n")

    blocos = {v: [] for v in FEATURES}
    blocos["y_residuo"] = []

    for t_idx in meses_idx:
        lat_idx = rng.integers(0, len(lat), size=PONTOS_POR_MES)
        lon_idx = rng.integers(0, len(lon), size=PONTOS_POR_MES)

        mes_origem = pd.Timestamp(tempos[t_idx]).month
        mes_target = mes_alvo(t_idx, tempos)

        y_alvo = tp_alvo.isel(time=t_idx).values[lat_idx, lon_idx].astype(np.float32)
        clim_pontos = clim[mes_target - 1][lat_idx, lon_idx].astype(np.float32)

        blocos["lat"].append(lat[lat_idx].astype(np.float32))
        blocos["lon"].append(lon[lon_idx].astype(np.float32))
        blocos["month_sin"].append(np.full(PONTOS_POR_MES, np.sin(2 * np.pi * mes_origem / 12), dtype=np.float32))
        blocos["month_cos"].append(np.full(PONTOS_POR_MES, np.cos(2 * np.pi * mes_origem / 12), dtype=np.float32))
        blocos["tp_climatologia"].append(clim_pontos)
        blocos["y_residuo"].append((y_alvo - clim_pontos).astype(np.float32))

        for v in VARS_ATMOSFERICAS:
            slice_v = atmosfericas[v].isel(time=t_idx).values
            blocos[v].append(slice_v[lat_idx, lon_idx].astype(np.float32))

    X = pd.DataFrame({v: np.concatenate(blocos[v]) for v in FEATURES})
    y_res = np.concatenate(blocos["y_residuo"])
    return X, y_res


# ---------- 5. inferencia sobre os 24 meses de teste ----------

def gera_predicoes(teste, clim, lat, lon, modelo=None):
    """modelo=None -> gera S00 (so climatologia). modelo!=None -> gera S01 (climatologia + residuo)."""
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    lat_flat = lat_grid.ravel().astype(np.float32)
    lon_flat = lon_grid.ravel().astype(np.float32)

    linhas = []
    for t_idx in range(teste.sizes["time"]):
        time_val = pd.Timestamp(teste.time.isel(time=t_idx).values)
        origem_val = pd.Timestamp(teste.time_origem.isel(time=t_idx).values)
        mes_target = time_val.month
        mes_origem = origem_val.month

        clim_flat = clim[mes_target - 1].ravel().astype(np.float32)

        if modelo is None:
            tp_predito = clim_flat.copy()
        else:
            month_sin = np.full(lat_flat.shape, np.sin(2 * np.pi * mes_origem / 12), dtype=np.float32)
            month_cos = np.full(lat_flat.shape, np.cos(2 * np.pi * mes_origem / 12), dtype=np.float32)
            dados = {"lat": lat_flat, "lon": lon_flat, "month_sin": month_sin, "month_cos": month_cos,
                     "tp_climatologia": clim_flat}
            for v in VARS_ATMOSFERICAS:
                dados[v] = teste[v].isel(time=t_idx).values.ravel().astype(np.float32)
            X_mes = pd.DataFrame(dados)[FEATURES]
            residuo = modelo.predict(X_mes)
            tp_predito = clim_flat + residuo
            del X_mes, residuo
            gc.collect()

        ids = [constroi_id(time_val.year, time_val.month, la, lo) for la, lo in zip(lat_flat, lon_flat)]
        linhas.append(pd.DataFrame({"id": ids, "tp_mm_day": tp_predito.astype(np.float32)}))

    return pd.concat(linhas, ignore_index=True)


def alinha_com_sample(pred_df, sample):
    alinhado = sample[["id"]].merge(pred_df, on="id", how="left")
    n_faltando = int(alinhado["tp_mm_day"].isnull().sum())
    n_dup = int(pred_df["id"].duplicated().sum())
    return alinhado, n_faltando, n_dup


def estatisticas(serie):
    return {
        "min": float(serie.min()), "max": float(serie.max()), "media": float(serie.mean()),
        "mediana": float(serie.median()), "desvio_padrao": float(serie.std()),
        "pct_negativo": round(100 * float((serie < 0).mean()), 3),
        "q01": float(serie.quantile(0.01)), "q05": float(serie.quantile(0.05)),
        "q25": float(serie.quantile(0.25)), "q75": float(serie.quantile(0.75)),
        "q95": float(serie.quantile(0.95)), "q99": float(serie.quantile(0.99)),
    }


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

    confirma_protocolo(teste, atmosfericas_treino)
    audita_periodo(teste, sample)

    # verificacao manual de 3 ids (inicio, meio, fim)
    print("=== verificacao manual de ids (inicio / meio / fim) ===")
    for idx in [0, len(sample) // 2, len(sample) - 1]:
        id_str = sample["id"].iloc[idx]
        ano, mes, lat_s, lon_s = id_str.split("_")
        print(f"id={id_str} -> ano={ano} mes={mes} lat={lat_s} lon={lon_s} "
              f"(celula existe na grade de teste? "
              f"{float(lat_s) in lat and float(lon_s) in lon})")
    print()

    clim = climatologia_real(tp)

    # ---------- S00 ----------
    print("=== gerando S00 (climatologia) ===")
    pred_s00 = gera_predicoes(teste, clim, lat, lon, modelo=None)
    s00, s00_faltando, s00_dup = alinha_com_sample(pred_s00, sample)
    del pred_s00
    gc.collect()

    # ---------- treino S01 ----------
    print("=== treinando S01 (M03 consolidado) ===")
    X_train, y_train = monta_treino(tp_alvo, atmosfericas_treino, clim, lat, lon, tempos)
    print(f"amostras de treino: {len(X_train):,}")

    modelo = LGBMRegressor(
        n_estimators=300,
        num_leaves=15,
        learning_rate=0.05,
        random_state=SEED,
        verbosity=-1,
    )
    modelo.fit(X_train, y_train)
    del X_train, y_train
    gc.collect()

    print("=== gerando S01 (climatologia + residuo previsto) ===")
    pred_s01 = gera_predicoes(teste, clim, lat, lon, modelo=modelo)
    s01, s01_faltando, s01_dup = alinha_com_sample(pred_s01, sample)
    del pred_s01, modelo
    gc.collect()

    # ---------- validacao estrutural ----------
    print("\n=== validacao estrutural ===")
    for nome, df, faltando, dup in [("S00", s00, s00_faltando, s00_dup), ("S01", s01, s01_faltando, s01_dup)]:
        n_linhas = len(df)
        ids_iguais = bool((df["id"].values == sample["id"].values).all())
        n_nan = int(df["tp_mm_day"].isnull().sum())
        n_inf = int(np.isinf(df["tp_mm_day"].values).sum())
        print(f"{nome}: linhas={n_linhas} (esperado 1885464, ok={n_linhas == 1_885_464}) | "
              f"ids identicos ao sample (mesma ordem)={ids_iguais} | duplicados={dup} | "
              f"faltando_no_merge={faltando} | NaN={n_nan} | inf={n_inf} | "
              f"dtype={df['tp_mm_day'].dtype}")

    # ---------- sanity checks de distribuicao ----------
    print("\n=== distribuicao S00 ===")
    print(estatisticas(s00["tp_mm_day"]))
    print("\n=== distribuicao S01 ===")
    print(estatisticas(s01["tp_mm_day"]))

    # ---------- comparacao S00 x S01 ----------
    diff = s01["tp_mm_day"] - s00["tp_mm_day"]
    print("\n=== comparacao S00 x S01 (nao mede qual e melhor, so quanto o ML altera a climatologia) ===")
    print({
        "diferenca_media": float(diff.mean()),
        "MAE": float(diff.abs().mean()),
        "RMSE_entre_predicoes": float(np.sqrt((diff ** 2).mean())),
        "correlacao": float(np.corrcoef(s00["tp_mm_day"], s01["tp_mm_day"])[0, 1]),
        "pct_S01_maior": round(100 * float((s01["tp_mm_day"] > s00["tp_mm_day"]).mean()), 2),
        "pct_S01_menor": round(100 * float((s01["tp_mm_day"] < s00["tp_mm_day"]).mean()), 2),
    })

    ids_split = sample["id"].str.split("_", expand=True)
    meses_id = ids_split[0] + "_" + ids_split[1]
    resumo_mes = pd.DataFrame({"mes": meses_id, "diff_abs": diff.abs()}).groupby("mes")["diff_abs"].mean()
    resumo_mes.to_csv(RESULTS_DIR / "s00_s01_diferenca_por_mes.csv")
    print("\ndiferenca media absoluta por mes salva em results/s00_s01_diferenca_por_mes.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 4))
    plt.plot(resumo_mes.index, resumo_mes.values, marker="o")
    plt.xticks(rotation=90, fontsize=7)
    plt.ylabel("diferenca media absoluta (mm/day)")
    plt.title("S00 x S01 - magnitude media da diferenca por mes")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "s00_s01_diferenca_por_mes.png", dpi=120)
    print("grafico salvo em results/s00_s01_diferenca_por_mes.png")

    # ---------- salva submissoes ----------
    s00[["id", "tp_mm_day"]].to_csv(SUBMISSIONS_DIR / "s00_climatologia.csv", index=False)
    s01[["id", "tp_mm_day"]].to_csv(SUBMISSIONS_DIR / "s01_m03_residuo.csv", index=False)

    for f in ["s00_climatologia.csv", "s01_m03_residuo.csv"]:
        tamanho_mb = (SUBMISSIONS_DIR / f).stat().st_size / 1e6
        print(f"\n{f}: {tamanho_mb:.1f} MB")
