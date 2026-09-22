"""B02 - blend entre climatologia (B00) e persistencia (B01): previsao = alpha*clim + (1-alpha)*persist."""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

ALPHAS = np.round(np.arange(0.0, 1.01, 0.1), 1)


def climatologia_ate(tp, data_corte):
    historico = tp.sel(time=slice(None, data_corte))
    return historico.groupby("time.month").mean("time")


def roda_backtest_blend(tp, origem, ultimo_alvo_disponivel, nome, alpha):
    clim = climatologia_ate(tp, origem)
    persist = tp.sel(time=origem)

    alvos = pd.date_range(origem, periods=25, freq="MS")[1:]
    alvos = [a for a in alvos if a <= pd.Timestamp(ultimo_alvo_disponivel)]

    linhas = []
    for alvo in alvos:
        horizonte = (alvo.year - origem.year) * 12 + (alvo.month - origem.month)
        pred = alpha * clim.sel(month=alvo.month) + (1 - alpha) * persist
        real = tp.sel(time=alvo)
        erro2 = (pred - real) ** 2
        rmse_horizonte = float(np.sqrt(erro2.mean()))
        linhas.append({"backtest": nome, "horizonte": horizonte, "alpha": alpha, "rmse": rmse_horizonte})

    return pd.DataFrame(linhas)


if __name__ == "__main__":
    ds = xr.open_dataset(RAW_DIR / "treino_tp.nc")
    tp = ds["tp"]

    todos = []
    for alpha in ALPHAS:
        res_a = roda_backtest_blend(tp, pd.Timestamp("2020-12-01"), "2022-12-01", "A", alpha)
        res_b = roda_backtest_blend(tp, pd.Timestamp("2021-12-01"), "2022-12-01", "B", alpha)
        todos.append(res_a)
        todos.append(res_b)
    todos = pd.concat(todos, ignore_index=True)
    todos.to_csv(RESULTS_DIR / "b02_rmse_por_alpha_horizonte.csv", index=False)

    resumo = []
    for alpha in ALPHAS:
        rmse_a = float(np.sqrt((todos[(todos.alpha == alpha) & (todos.backtest == "A")]["rmse"] ** 2).mean()))
        rmse_b = float(np.sqrt((todos[(todos.alpha == alpha) & (todos.backtest == "B")]["rmse"] ** 2).mean()))
        resumo.append({"alpha": alpha, "rmse_backtestA": rmse_a, "rmse_backtestB": rmse_b,
                        "media_A_B": (rmse_a + rmse_b) / 2})
    resumo = pd.DataFrame(resumo)
    resumo.to_csv(RESULTS_DIR / "b02_resumo_por_alpha.csv", index=False)

    print(resumo.to_string(index=False))

    melhor = resumo.loc[resumo["media_A_B"].idxmin()]
    print("\nmelhor alpha (por media A/B):", melhor["alpha"])
    print(melhor.to_string())
