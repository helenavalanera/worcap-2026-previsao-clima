"""Gate 06 - monta e valida o dataset de features para o M01. Nao treina nenhum modelo."""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

VARIAVEIS_ATMOSFERICAS = [
    "t2", "cloud_cover", "shum_850", "surface_pressure",
    "u_850", "v_850", "temperature_850", "rel_hum_850", "geopotential_850",
]


def monta_dataset():
    """Combina as variaveis atmosfericas + tp_alvo em um unico xr.Dataset, todas em time=M."""
    ds = xr.open_dataset(RAW_DIR / "treino_tp_alvo.nc")

    for var in VARIAVEIS_ATMOSFERICAS:
        arq = RAW_DIR / f"treino_{var}.nc"
        ds[var] = xr.open_dataset(arq)[var]

    mes = ds["time"].dt.month
    ds["month_sin"] = np.sin(2 * np.pi * mes / 12)
    ds["month_cos"] = np.cos(2 * np.pi * mes / 12)

    return ds


if __name__ == "__main__":
    ds = monta_dataset()

    print("variaveis no dataset:", list(ds.data_vars))
    print("dims:", dict(ds.sizes))

    # validacao: um ponto geografico ao longo de varios meses
    ponto = ds.sel(lat=-23.5, lon=-46.5, method="nearest")
    exemplo = ponto.isel(time=slice(0, 6))
    tabela = pd.DataFrame({
        "time": exemplo.time.values,
        "tp_alvo": exemplo.tp_alvo.values.round(3),
        "t2": exemplo.t2.values.round(2),
        "rel_hum_850": exemplo.rel_hum_850.values.round(2),
        "month_sin": exemplo.month_sin.values.round(3),
        "month_cos": exemplo.month_cos.values.round(3),
    })
    print("\nponto lat=-23.5 lon=-46.5 (regiao de Sao Paulo), primeiros 6 meses:")
    print(tabela.to_string(index=False))

    # checagem de nulos por variavel (exceto tp_alvo, que tem NaN esperado no ultimo mes)
    print("\nNaNs por variavel:")
    for var in VARIAVEIS_ATMOSFERICAS + ["month_sin", "month_cos"]:
        n_nan = int(ds[var].isnull().sum())
        print(f"  {var}: {n_nan}")
    n_nan_alvo = int(ds["tp_alvo"].isnull().sum())
    print(f"  tp_alvo: {n_nan_alvo} (esperado = tamanho de uma grade, ultimo mes sem M+1)")

    n_linhas = ds.sizes["time"] * ds.sizes["lat"] * ds.sizes["lon"]
    n_colunas = len(VARIAVEIS_ATMOSFERICAS) + 2 + 2  # atmosfericas + lat/lon + month_sin/cos
    print(f"\ntamanho do dataset achatado (estimado): {n_linhas:,} linhas x ~{n_colunas} colunas")
