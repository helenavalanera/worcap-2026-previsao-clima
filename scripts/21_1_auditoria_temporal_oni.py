"""Gate 21.1 - auditoria temporal do ONI usado no Gate 21.

Verifica se o valor de ONI associado a cada mes de origem usa apenas informacao
disponivel ate aquele mes (sem olhar o mes seguinte, que e o mes-alvo da previsao).
"""

from pathlib import Path

import pandas as pd

EXTERNAL_DIR = Path(__file__).resolve().parent.parent / "data" / "external"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# janela de 3 meses de cada codigo SEAS (letra a letra)
SEAS_MESES = {
    "DJF": [12, 1, 2], "JFM": [1, 2, 3], "FMA": [2, 3, 4], "MAM": [3, 4, 5],
    "AMJ": [4, 5, 6], "MJJ": [5, 6, 7], "JJA": [6, 7, 8], "JAS": [7, 8, 9],
    "ASO": [8, 9, 10], "SON": [9, 10, 11], "OND": [10, 11, 12], "NDJ": [11, 12, 1],
}
SEAS_CENTRO = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
               "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12}


def carrega_bruto():
    df = pd.read_csv(EXTERNAL_DIR / "oni_raw.txt", sep=r"\s+")
    return df


def constroi_indice_disponibilidade_conservadora(df):
    """Para cada linha SEAS/YR, calcula os 3 meses reais da janela e a data em que
    essa janela fica totalmente disponivel: o ULTIMO mes contido nela (conservador,
    conforme A3)."""
    linhas = []
    for _, row in df.iterrows():
        seas, yr, anom = row["SEAS"], int(row["YR"]), row["ANOM"]
        centro = SEAS_CENTRO[seas]
        # mes 1 = centro-1 (ano anterior se centro==1), mes 3 = centro+1 (ano seguinte se centro==12)
        if centro == 1:
            mes1, ano1 = 12, yr - 1
        else:
            mes1, ano1 = centro - 1, yr
        mes2, ano2 = centro, yr
        if centro == 12:
            mes3, ano3 = 1, yr + 1
        else:
            mes3, ano3 = centro + 1, yr

        disponibilidade = pd.Timestamp(ano3, mes3, 1)  # ultimo mes da janela
        centro_data = pd.Timestamp(ano2, mes2, 1)

        linhas.append({
            "seas": seas, "yr": yr, "anom": anom,
            "mes1": pd.Timestamp(ano1, mes1, 1), "mes2": centro_data, "mes3": pd.Timestamp(ano3, mes3, 1),
            "disponibilidade_conservadora": disponibilidade,
            "data_indexada_antiga_por_centro": centro_data,  # como o Gate 21 original indexava
        })
    aud = pd.DataFrame(linhas)
    # indice novo (correto): por data de DISPONIBILIDADE (ultimo mes da janela)
    serie_correta = aud.set_index("disponibilidade_conservadora")["anom"]
    # indice antigo (usado no Gate 21, com leakage): por mes CENTRAL da janela
    serie_antiga = aud.set_index("data_indexada_antiga_por_centro")["anom"]
    return aud, serie_correta, serie_antiga


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    df = carrega_bruto()

    print("=== A1. Fonte ===")
    print("arquivo: data/external/oni_raw.txt")
    print("formato original: colunas SEAS (codigo de 3 letras), YR (ano), TOTAL (SST media), ANOM (anomalia)")
    print(f"primeira linha: {df.iloc[0]['SEAS']} {df.iloc[0]['YR']}")
    print(f"ultima linha: {df.iloc[-1]['SEAS']} {df.iloc[-1]['YR']}")
    print(f"valores ausentes: {df.isnull().sum().sum()}")
    print("interpretacao original (Gate 21): SEAS mapeado ao MES CENTRAL da janela de 3 meses "
          "(ex.: DJF -> janeiro)")

    aud, serie_correta, serie_antiga = constroi_indice_disponibilidade_conservadora(df)
    aud.to_csv(RESULTS_DIR / "oni_temporal_audit.csv", index=False)

    print("\n=== A2/A3. Reconstrucao das janelas (exemplo DJF e NDJ) ===")
    print(aud[aud.seas.isin(["DJF", "NDJ"])].head(4).to_string(index=False))

    print("\n=== comparacao direta: indexacao antiga (Gate 21) x nova (conservadora) ===")
    exemplo = aud[(aud.seas == "DJF") & (aud.yr == 2017)].iloc[0]
    print(f"DJF 2017: janela = {exemplo['mes1'].date()}, {exemplo['mes2'].date()}, {exemplo['mes3'].date()}")
    print(f"  Gate 21 (antigo) indexava esse valor em: {exemplo['data_indexada_antiga_por_centro'].date()} "
          f"(mes central)")
    print(f"  disponibilidade real conservadora: {exemplo['disponibilidade_conservadora'].date()} (ultimo mes)")
    print(f"  => se usado para origem=janeiro/2017 (mes central), o valor so fica completo em "
          f"{exemplo['disponibilidade_conservadora'].date()}, um mes DEPOIS da origem. LEAKAGE CONFIRMADO "
          f"na indexacao antiga.")

    print("\n=== A4. Auditoria sobre 12 casos reais (F1/F2/F3, meses variados) ===")
    casos = [
        ("F1", "2017-01-01"), ("F1", "2017-06-01"), ("F1", "2017-12-01"), ("F1", "2018-09-01"),
        ("F2", "2019-02-01"), ("F2", "2019-07-01"), ("F2", "2020-03-01"), ("F2", "2020-11-01"),
        ("F3", "2021-01-01"), ("F3", "2021-08-01"), ("F3", "2022-05-01"), ("F3", "2022-12-01"),
    ]

    linhas_casos = []
    for nome_fold, target_str in casos:
        target = pd.Timestamp(target_str)
        origem = target - pd.DateOffset(months=1)

        # valor usado pela indexacao ANTIGA (com leakage) e pela NOVA (corrigida)
        valor_antigo = serie_antiga.get(pd.Timestamp(origem.year, origem.month, 1), None)
        valor_novo = serie_correta.get(pd.Timestamp(origem.year, origem.month, 1), None)

        linha_aud_antiga = aud[aud["data_indexada_antiga_por_centro"] == pd.Timestamp(origem.year, origem.month, 1)]
        if len(linha_aud_antiga):
            janela = linha_aud_antiga.iloc[0]
            max_oni_mes_antigo = janela["mes3"]
        else:
            max_oni_mes_antigo = None

        linha_aud_nova = aud[aud["disponibilidade_conservadora"] == pd.Timestamp(origem.year, origem.month, 1)]
        if len(linha_aud_nova):
            janela_n = linha_aud_nova.iloc[0]
            max_oni_mes_novo = janela_n["mes3"]
            oni_window_novo = f"{janela_n['mes1'].strftime('%Y-%m')}/{janela_n['mes2'].strftime('%Y-%m')}/{janela_n['mes3'].strftime('%Y-%m')}"
        else:
            max_oni_mes_novo = None
            oni_window_novo = None

        assert_antigo_ok = (max_oni_mes_antigo is not None) and (max_oni_mes_antigo <= origem)
        assert_novo_ok = (max_oni_mes_novo is not None) and (max_oni_mes_novo <= origem)

        linhas_casos.append({
            "fold": nome_fold, "target_date": target.date(), "origin_date": origem.date(),
            "oni_valor_ANTIGO_gate21": valor_antigo, "max_oni_month_ANTIGO": max_oni_mes_antigo,
            "assert_antigo_passa": assert_antigo_ok,
            "oni_valor_NOVO_corrigido": valor_novo, "oni_window_novo": oni_window_novo,
            "max_oni_month_NOVO": max_oni_mes_novo, "assert_novo_passa": assert_novo_ok,
        })

    df_casos = pd.DataFrame(linhas_casos)
    print(df_casos.to_string(index=False))

    n_falhas_antigo = int((~df_casos["assert_antigo_passa"]).sum())
    n_falhas_novo = int((~df_casos["assert_novo_passa"]).sum())
    print(f"\nindexacao ANTIGA (Gate 21 original): falhas de assert = {n_falhas_antigo}/12 "
          f"-> {'LEAKAGE CONFIRMADO' if n_falhas_antigo > 0 else 'sem leakage'}")
    print(f"indexacao NOVA (corrigida): falhas de assert = {n_falhas_novo}/12 "
          f"-> {'LEAKAGE' if n_falhas_novo > 0 else 'SEM LEAKAGE - aprovado'}")
