"""B01 - persistencia: usa a precipitacao da origem como previsao constante para os 24 horizontes."""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def roda_backtest_persistencia(tp, origem, ultimo_alvo_disponivel, nome):
    pred = tp.sel(time=origem)  # precipitacao da origem, constante para todos os horizontes

    alvos = pd.date_range(origem, periods=25, freq="MS")[1:]
    alvos = [a for a in alvos if a <= pd.Timestamp(ultimo_alvo_disponivel)]

    linhas = []
    for alvo in alvos:
        horizonte = (alvo.year - origem.year) * 12 + (alvo.month - origem.month)
        real = tp.sel(time=alvo)
        erro2 = (pred - real) ** 2
        rmse_horizonte = float(np.sqrt(erro2.mean()))
        linhas.append({"backtest": nome, "horizonte": horizonte, "rmse": rmse_horizonte})

    return pd.DataFrame(linhas)


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


if __name__ == "__main__":
    ds = xr.open_dataset(RAW_DIR / "treino_tp.nc")
    tp = ds["tp"]

    res_a = roda_backtest_persistencia(tp, pd.Timestamp("2020-12-01"), "2022-12-01", "A")
    res_b = roda_backtest_persistencia(tp, pd.Timestamp("2021-12-01"), "2022-12-01", "B")

    todos = pd.concat([res_a, res_b], ignore_index=True)
    todos.to_csv(RESULTS_DIR / "b01_rmse_por_horizonte.csv", index=False)

    resumo_a = resumo(res_a, "Backtest A (persistencia, origem dez/2020)")
    resumo_b = resumo(res_b, "Backtest B (persistencia, origem dez/2021, so h1-12)")

    # comparacao direta por horizonte, backtest A (unico com os 24 horizontes completos)
    b00 = pd.read_csv(RESULTS_DIR / "b00_rmse_por_horizonte.csv")
    b00_a = b00[b00.backtest == "A"].set_index("horizonte")["rmse"]
    b01_a = res_a.set_index("horizonte")["rmse"]

    comparacao = pd.DataFrame({"b00": b00_a, "b01": b01_a})
    comparacao["vencedor"] = np.where(comparacao["b01"] < comparacao["b00"], "B01", "B00")
    comparacao.to_csv(RESULTS_DIR / "b00_vs_b01_por_horizonte.csv")
    print("\n--- comparacao por horizonte (Backtest A) ---")
    print(comparacao.to_string())

    n_b01_vence = int((comparacao["vencedor"] == "B01").sum())
    print(f"\nB01 supera B00 em {n_b01_vence}/24 horizontes")

    # grafico: substitui o anterior, agora com as duas curvas do Backtest A
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    plt.plot(comparacao.index, comparacao["b00"], marker="o", label="B00 - climatologia")
    plt.plot(comparacao.index, comparacao["b01"], marker="o", label="B01 - persistencia")
    plt.xlabel("horizonte (meses a frente)")
    plt.ylabel("RMSE (mm/day)")
    plt.title("B00 x B01 - RMSE por horizonte (Backtest A, origem dez/2020)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "b00_rmse_por_horizonte.png", dpi=120)
    print("\nGrafico atualizado em results/b00_rmse_por_horizonte.png")
