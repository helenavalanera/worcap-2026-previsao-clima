"""Gate 11 - auditoria: lag_meses tem correspondencia defensavel em exemplos historicos de treino?

Nao treina nenhum modelo. Verifica, com os proprios dados, se lag_meses corresponde a alguma
diferenca real na informacao disponivel por linha (frescor das features atmosfericas), ou se e
apenas um indice de posicao relativo a uma unica ancora fixa do periodo de teste (dez/2022).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def auditoria_teste():
    teste = xr.open_dataset(RAW_DIR / "teste_features.nc")
    tempos = pd.to_datetime(teste.time.values)
    origens = pd.to_datetime(teste.time_origem.values)
    lag = teste.lag_meses.values

    frescor_meses = [(t.year - o.year) * 12 + (t.month - o.month) for t, o in zip(tempos, origens)]

    tabela = pd.DataFrame({
        "time": tempos.date,
        "time_origem": origens.date,
        "lag_meses": lag,
        "frescor_features_meses": frescor_meses,
    })
    print("=== teste: lag_meses vs frescor real das features (time - time_origem) ===")
    print(tabela.to_string(index=False))
    print(f"\nfrescor das features e sempre {set(frescor_meses)} mes(es), independente do lag_meses "
          f"(que varia de {lag.min()} a {lag.max()})")
    print("conclusao parcial: lag_meses NAO mede o frescor da informacao atmosferica disponivel;")
    print("mede apenas a distancia ate uma unica ancora fixa (dez/2022, a ultima precipitacao real conhecida).")
    return tabela


def auditoria_treino_naive(origem_backtest, meses_amostrados, seed=42):
    """O que aconteceria se calculassemos lag_meses para o treino do mesmo jeito que no teste:
    lag = mes_alvo - ancora, usando a origem do backtest como ancora (unica ancora fixa disponivel,
    equivalente ao dez/2022 do teste real)."""
    tp_alvo = xr.open_dataset(RAW_DIR / "treino_tp_alvo.nc")["tp_alvo"]
    tempos = tp_alvo["time"].values

    cutoff = np.datetime64(origem_backtest)
    elegiveis = np.where(tempos < cutoff)[0]
    rng = np.random.default_rng(seed)
    meses_idx = np.sort(rng.choice(elegiveis, size=min(meses_amostrados, len(elegiveis)), replace=False))

    ancora = pd.Timestamp(origem_backtest)
    lags_naive = []
    for t_idx in meses_idx:
        m = pd.Timestamp(tempos[t_idx])
        alvo = m + pd.DateOffset(months=1)
        lag = (alvo.year - ancora.year) * 12 + (alvo.month - ancora.month)
        lags_naive.append(lag)

    lags_naive = np.array(lags_naive)
    print(f"\n=== treino: lag 'ingenuo' (mes_alvo - origem_do_backtest={origem_backtest}) ===")
    print(f"amostras: {len(lags_naive)} | min={lags_naive.min()} | max={lags_naive.max()} | "
          f"% dentro de 1-24: {100 * np.mean((lags_naive >= 1) & (lags_naive <= 24)):.2f}%")
    print("faixa esperada no teste: 1 a 24")
    return lags_naive


if __name__ == "__main__":
    auditoria_teste()
    auditoria_treino_naive("2020-12-01", meses_amostrados=150)
    auditoria_treino_naive("2021-12-01", meses_amostrados=150)
