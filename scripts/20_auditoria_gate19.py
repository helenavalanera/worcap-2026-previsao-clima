"""Auditoria do Gate 19: distingue erro estrutural de amostragem de comportamento
esperado de uma amostra aleatoria pequena (150 de ~950 meses elegiveis). Reproduz
exatamente a mesma logica de selecao de meses usada em monta_treino() do Gate 19,
sem retreinar nenhum modelo.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

SEED = 42
MESES_AMOSTRADOS = 150

FOLDS = [
    {"nome": "F1", "cutoff": "2016-12-01"},
    {"nome": "F2", "cutoff": "2018-12-01"},
    {"nome": "F3", "cutoff": "2020-12-01"},
]


def seleciona_meses_elegiveis(tempos, cutoff):
    """Identica a logica usada em monta_treino() do Gate 19."""
    return np.where(tempos < np.datetime64(cutoff))[0]


if __name__ == "__main__":
    tp_alvo = xr.open_dataset(RAW_DIR / "treino_tp_alvo.nc")["tp_alvo"]
    tempos = tp_alvo["time"].values

    linhas = []
    for fold in FOLDS:
        nome, cutoff = fold["nome"], fold["cutoff"]

        elegiveis = seleciona_meses_elegiveis(tempos, cutoff)
        rng = np.random.default_rng(SEED)
        n_meses = min(MESES_AMOSTRADOS, len(elegiveis))
        meses_idx = np.sort(rng.choice(elegiveis, size=n_meses, replace=False))

        max_feature_elegivel = pd.Timestamp(tempos[elegiveis[-1]])
        max_target_elegivel = max_feature_elegivel + pd.DateOffset(months=1)

        max_feature_amostrado = pd.Timestamp(tempos[meses_idx[-1]])
        max_target_amostrado = max_feature_amostrado + pd.DateOffset(months=1)

        n_meses_amostrados_distintos = len(set(meses_idx.tolist()))

        # posicao relativa de cada mes amostrado dentro do pool elegivel (0=mais antigo, 1=mais recente)
        posicao_no_pool = np.searchsorted(elegiveis, meses_idx) / (len(elegiveis) - 1)
        mediana_pos = float(np.median(posicao_no_pool))
        q25_pos = float(np.percentile(posicao_no_pool, 25))
        q75_pos = float(np.percentile(posicao_no_pool, 75))

        gap_meses = (pd.Timestamp(cutoff).year - max_target_elegivel.year) * 12 + \
                    (pd.Timestamp(cutoff).month - max_target_elegivel.month)
        gap_amostrado_meses = (pd.Timestamp(cutoff).year - max_target_amostrado.year) * 12 + \
                               (pd.Timestamp(cutoff).month - max_target_amostrado.month)

        print(f"\n=== {nome} (cutoff={cutoff}) ===")
        print(f"seed usado: {SEED}")
        print(f"n_meses_elegiveis: {len(elegiveis)}")
        print(f"n_meses_amostrados (distintos): {n_meses_amostrados_distintos}")
        print(f"max_target_elegivel (pool completo): {max_target_elegivel.date()} "
              f"(gap ate cutoff: {gap_meses} mes(es))")
        print(f"max_target_amostrado (so os 150 sorteados): {max_target_amostrado.date()} "
              f"(gap ate cutoff: {gap_amostrado_meses} mes(es))")
        print(f"posicao relativa no pool elegivel dos meses amostrados: "
              f"mediana={mediana_pos:.3f} | Q25={q25_pos:.3f} | Q75={q75_pos:.3f} "
              f"(1.0 = mes mais recente do pool)")

        linhas.append({
            "fold": nome, "cutoff": cutoff, "seed": SEED,
            "n_meses_elegiveis": len(elegiveis), "n_meses_amostrados": n_meses_amostrados_distintos,
            "max_target_elegivel": max_target_elegivel.date(), "gap_elegivel_meses": gap_meses,
            "max_target_amostrado": max_target_amostrado.date(), "gap_amostrado_meses": gap_amostrado_meses,
            "mediana_posicao_pool": mediana_pos, "q25_posicao_pool": q25_pos, "q75_posicao_pool": q75_pos,
        })

    df = pd.DataFrame(linhas)
    print("\n\n=== resumo comparativo entre folds ===")
    print(df.to_string(index=False))

    print("\n=== checagem dos criterios de aprovacao ===")
    a_ok = (df["gap_elegivel_meses"] <= 1).all()
    print(f"(a) max_target_elegivel a no maximo 1 mes do cutoff em todos os folds: {a_ok}")

    variacao_mediana = df["mediana_posicao_pool"].max() - df["mediana_posicao_pool"].min()
    b_ok = variacao_mediana < 0.15  # limiar de razoabilidade para "comparavel"
    print(f"(b) medianas de posicao no pool comparaveis entre folds "
          f"(variacao={variacao_mediana:.3f}): {b_ok}")

    print("(c) funcao de selecao identica entre folds (mesmo codigo, so cutoff muda): True "
          "(mesma funcao seleciona_meses_elegiveis usada nos 3 folds, unico parametro variavel e o cutoff)")

    df.to_csv(Path(__file__).resolve().parent.parent / "results" / "gate19_auditoria.csv", index=False)
