"""Auditoria dos dados brutos: inventario, dimensoes, unidades, periodo, grade e NaNs."""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def inventario():
    print("=" * 70)
    print("INVENTARIO DE ARQUIVOS")
    print("=" * 70)
    rows = []
    for f in sorted(RAW_DIR.iterdir()):
        if f.name.startswith("."):
            continue
        size_mb = f.stat().st_size / (1024 * 1024)
        rows.append({"arquivo": f.name, "extensao": f.suffix, "tamanho_mb": round(size_mb, 1)})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


def audita_netcdf(nome_arquivo):
    ds = xr.open_dataset(RAW_DIR / nome_arquivo)
    var = list(ds.data_vars)[0]
    da = ds[var]

    n_total = da.size
    n_nan = int(da.isnull().sum())

    print(f"\n--- {nome_arquivo} ---")
    print("variavel:", var, "| atributos:", dict(da.attrs))
    print("dims:", dict(ds.sizes))
    print("dtype:", da.dtype)
    print("tempo: ", str(ds.time.min().values)[:10], "->", str(ds.time.max().values)[:10],
          "| n_meses:", ds.sizes.get("time", "-"))
    lat_name = "lat" if "lat" in ds.coords else None
    lon_name = "lon" if "lon" in ds.coords else None
    if lat_name and lon_name:
        print("lat:", float(ds.lat.min()), "->", float(ds.lat.max()), "| n_lat:", ds.sizes["lat"])
        print("lon:", float(ds.lon.min()), "->", float(ds.lon.max()), "| n_lon:", ds.sizes["lon"])
        res_lat = float(np.diff(ds.lat.values).mean())
        res_lon = float(np.diff(ds.lon.values).mean())
        print("resolucao aprox: lat=%.3f lon=%.3f" % (res_lat, res_lon))
    print(f"NaNs: {n_nan}/{n_total} ({100*n_nan/n_total:.2f}%)")

    ds.close()


def audita_temporal(nome_arquivo):
    ds = xr.open_dataset(RAW_DIR / nome_arquivo)
    tempos = pd.to_datetime(ds.time.values)
    esperado = pd.date_range(tempos.min(), tempos.max(), freq="MS")
    faltando = esperado.difference(tempos)
    duplicados = tempos[tempos.duplicated()]
    ordenado = list(tempos) == sorted(tempos)

    print(f"\n--- auditoria temporal: {nome_arquivo} ---")
    print("meses faltando:", len(faltando), list(faltando)[:5])
    print("duplicados:", len(duplicados))
    print("ordenado:", ordenado)
    ds.close()


def audita_sample_submission():
    print("\n" + "=" * 70)
    print("SAMPLE SUBMISSION")
    print("=" * 70)
    df = pd.read_csv(RAW_DIR / "sample_submission.csv")
    print("shape:", df.shape)
    print("colunas:", list(df.columns))
    print(df.head(3).to_string(index=False))
    print(df.tail(3).to_string(index=False))

    id_col = df.columns[0]
    partes = df[id_col].str.split("_", expand=True)
    print("\nexemplo de split do id por '_':")
    print(partes.head(3).to_string(index=False))
    print("numero de partes no id:", partes.shape[1])

    print("\nanos distintos no id:", sorted(partes[0].unique())[:5], "...")
    print("meses distintos no id:", sorted(partes[1].unique()))
    print("valores nulos por coluna:\n", df.isnull().sum())


if __name__ == "__main__":
    inventario()

    print("\n" + "=" * 70)
    print("AUDITORIA NETCDF - ARQUIVOS DE TREINO")
    print("=" * 70)
    arquivos_nc = sorted([f.name for f in RAW_DIR.glob("treino_*.nc")])
    for f in arquivos_nc:
        audita_netcdf(f)

    print("\n" + "=" * 70)
    print("AUDITORIA NETCDF - ARQUIVO DE TESTE")
    print("=" * 70)
    audita_netcdf("teste_features.nc")

    audita_temporal("treino_tp.nc")
    audita_temporal("treino_tp_alvo.nc")
    audita_temporal("teste_features.nc")

    audita_sample_submission()
