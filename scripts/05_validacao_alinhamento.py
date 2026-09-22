"""Validacao do alinhamento temporal entre features e alvo, em treino e teste."""

from pathlib import Path

import pandas as pd
import xarray as xr

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def media_espacial(da, data):
    val = da.sel(time=data).mean()
    return float(val) if val.notnull() else float("nan")


if __name__ == "__main__":
    tp = xr.open_dataset(RAW_DIR / "treino_tp.nc")["tp"]
    alvo = xr.open_dataset(RAW_DIR / "treino_tp_alvo.nc")["tp_alvo"]
    t2 = xr.open_dataset(RAW_DIR / "treino_t2.nc")["t2"]
    rh = xr.open_dataset(RAW_DIR / "treino_rel_hum_850.nc")["rel_hum_850"]
    teste = xr.open_dataset(RAW_DIR / "teste_features.nc")

    datas_treino = ["1940-01-01", "2020-12-01", "2021-12-01", "2022-12-01"]
    linhas = []
    for d in datas_treino:
        d = pd.Timestamp(d)
        linhas.append({
            "arquivo": "treino",
            "time": d.date(),
            "tp(M)": round(media_espacial(tp, d), 3),
            "tp_alvo(M)": round(media_espacial(alvo, d), 3) if d < tp.time.max().values else "NaN",
            "t2(M)": round(media_espacial(t2, d), 3),
            "rel_hum_850(M)": round(media_espacial(rh, d), 3),
        })
    print(pd.DataFrame(linhas).to_string(index=False))

    print()
    datas_teste = ["2023-01-01", "2023-02-01", "2024-12-01"]
    linhas_teste = []
    for d in datas_teste:
        linha = teste.sel(time=d)
        linhas_teste.append({
            "time": str(linha.time.values)[:10],
            "time_origem": str(linha.time_origem.values)[:10],
            "lag_meses": int(linha.lag_meses.values),
            "tp_ultima_obs": round(float(linha.tp_ultima_obs.mean()), 3),
            "t2": round(float(linha.t2.mean()), 3),
            "rel_hum_850": round(float(linha.rel_hum_850.mean()), 3),
            "tp_alvo": "NaN" if bool(linha.tp_alvo.isnull().all()) else "presente",
        })
    print(pd.DataFrame(linhas_teste).to_string(index=False))
