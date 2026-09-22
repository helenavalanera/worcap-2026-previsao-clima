"""B00 - climatologia mensal espacial, validada com backtests multi-horizonte (1-24 meses)."""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def climatologia_ate(tp, data_corte):
    """Media mensal por lat/lon usando somente dados ate a data de corte (inclusive)."""
    historico = tp.sel(time=slice(None, data_corte))
    return historico.groupby("time.month").mean("time")


def roda_backtest(tp, origem, ultimo_alvo_disponivel, nome):
    clim = climatologia_ate(tp, origem)

    alvos = pd.date_range(origem, periods=25, freq="MS")[1:]  # 24 meses seguintes
    alvos = [a for a in alvos if a <= pd.Timestamp(ultimo_alvo_disponivel)]

    linhas = []
    for alvo in alvos:
        horizonte = (alvo.year - origem.year) * 12 + (alvo.month - origem.month)
        pred = clim.sel(month=alvo.month)
        real = tp.sel(time=alvo)
        erro2 = (pred - real) ** 2
        rmse_horizonte = float(np.sqrt(erro2.mean()))
        linhas.append({"backtest": nome, "horizonte": horizonte, "rmse": rmse_horizonte})

    return pd.DataFrame(linhas)


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    ds = xr.open_dataset(RAW_DIR / "treino_tp.nc")
    tp = ds["tp"]

    res_a = roda_backtest(tp, pd.Timestamp("2020-12-01"), "2022-12-01", "A")
    res_b = roda_backtest(tp, pd.Timestamp("2021-12-01"), "2022-12-01", "B")

    print(f"Backtest A: {len(res_a)} horizontes validados (esperado 24)")
    print(f"Backtest B: {len(res_b)} horizontes validados (esperado 12, resto sem verdade disponivel)")

    todos = pd.concat([res_a, res_b], ignore_index=True)
    todos.to_csv(RESULTS_DIR / "b00_rmse_por_horizonte.csv", index=False)

    def resumo(df, nome):
        global_rmse = float(np.sqrt((df["rmse"] ** 2).mean()))
        faixas = {
            "h1_6": df[df.horizonte.between(1, 6)]["rmse"].mean(),
            "h7_12": df[df.horizonte.between(7, 12)]["rmse"].mean(),
            "h13_18": df[df.horizonte.between(13, 18)]["rmse"].mean(),
            "h19_24": df[df.horizonte.between(19, 24)]["rmse"].mean(),
        }
        print(f"\n--- {nome} ---")
        print("RMSE global:", round(global_rmse, 4))
        for k, v in faixas.items():
            print(f"RMSE medio {k}:", "sem dados" if pd.isna(v) else round(v, 4))
        return global_rmse, faixas

    resumo_a = resumo(res_a, "Backtest A (origem dez/2020)")
    resumo_b = resumo(res_b, "Backtest B (origem dez/2021, so horizontes 1-12)")

    # grafico simples RMSE x horizonte
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    plt.plot(res_a.horizonte, res_a.rmse, marker="o", label="Backtest A (origem dez/2020)")
    plt.plot(res_b.horizonte, res_b.rmse, marker="o", label="Backtest B (origem dez/2021)")
    plt.xlabel("horizonte (meses a frente)")
    plt.ylabel("RMSE (mm/day)")
    plt.title("B00 - Climatologia: RMSE por horizonte")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "b00_rmse_por_horizonte.png", dpi=120)
    print("\nGrafico salvo em results/b00_rmse_por_horizonte.png")
