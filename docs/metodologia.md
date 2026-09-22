# Metodologia — WORCAP 2026

Este documento detalha decisões metodológicas que ficariam longas demais no README: a auditoria de leakage do índice ONI, a decomposição de erro do modelo final e o histórico de experimentos descartados.

## Evolução B00 → M05+ONI

O projeto seguiu uma sequência incremental, cada etapa respondendo a uma pergunta experimental específica:

1. **B00 (climatologia)** — baseline: qual o erro de simplesmente prever a média histórica do mês/célula?
2. **B01/B02 (persistência e blend)** — a precipitação do mês atual ajuda a prever a do mês seguinte? (Não — persistência perdeu para climatologia em todos os horizontes testados.)
3. **M01 (LightGBM direto)** — variáveis atmosféricas conseguem prever precipitação melhor que a climatologia, sem nenhum tratamento especial? (Não — ficou bem pior, provavelmente por amostragem esparsa demais para o modelo reconstruir a climatologia fina que o B00 calcula com o histórico completo.)
4. **M02 (climatologia como feature)** — dar a climatologia ao modelo como informação de entrada ajuda? (Aproximou do baseline, mas não superou.)
5. **M03 (resíduo)** — é mais fácil prever o desvio em relação à climatologia do que o valor absoluto? (Sim, formulação adotada dali em diante.)
6. **M04 (contexto espacial 3×3)** — vizinhança imediata ajuda? (Sem ganho consistente entre os dois backtests testados.)
7. **Gate 13/14 (tuning e volume)** — o modelo está limitado por hiperparâmetros ou por volume de dados? Reduzir a complexidade da árvore (`num_leaves` de 31 para 15) trouxe o maior ganho isolado até esse ponto; aumentar o volume de treino de 600k para 1,2M trouxe ganho pequeno e inconsistente entre folds.
8. **M05 (anomalias atmosféricas)** — anomalias (desvio em relação à climatologia da própria variável) são mais informativas que valores absolutos? (As anomalias passaram a responder por mais importância no modelo que as variáveis absolutas correspondentes, mas a validação rolling de 3 folds não mostrou ganho robusto e consistente sobre B00 nessa etapa.)
9. **Gate 19 (rolling-origin)** — a comparação B00 × M05 em 2 backtests é suficiente para decidir? Não: a validação rolling em 3 janelas históricas independentes revelou resultado inconsistente entre folds (F1 pior, F2 melhor, F3 quase empatado), classificado como evidência inconclusiva.
10. **M05+ONI** — falta alguma fonte de variabilidade de larga escala? Adicionar o índice ONI (proxy ENSO) produziu o primeiro modelo a superar B00 nos 3/3 folds de forma consistente.
11. **CENSO (climatologia condicionada por ENSO)** — o ganho do ONI pode vir só de uma climatologia condicionada ao regime, sem machine learning? Não: CENSO pura ficou pior que B00 no agregado e nos 3 folds; M05+ONI continuou superior em todo recorte de diagnóstico testado (regime, mês, latitude, intensidade).

## Auditoria temporal do ONI

### Hipótese inicial

O índice ONI é uma média móvel trimestral (ex.: DJF = dezembro+janeiro+fevereiro). A primeira implementação indexou cada valor pelo **mês central** da janela (DJF → janeiro), assumindo que esse era o mês de referência natural do valor.

### Identificação do problema

Ao revisar o alinhamento temporal, foi identificado que essa indexação fazia o valor de ONI usado para uma origem em janeiro incluir fevereiro — exatamente o mês-alvo da previsão feita a partir dessa origem. Ou seja: o modelo estaria, sem perceber, recebendo uma fração da informação do próprio mês que deveria prever.

### Teste formal

Uma auditoria reconstruiu explicitamente os três meses de cada janela ONI e testou, em 12 casos reais distribuídos entre os três folds e diferentes meses do ano, o critério `max_oni_month <= origin_date`. **12 dos 12 casos falharam** na indexação original.

### Correção

A indexação foi refeita usando `availability_month = último mês da janela` (ex.: DJF passa a ser associado a fevereiro, não a janeiro) — a data mais cedo em que aquele valor de ONI pode legitimamente ser considerado "conhecido". Reaplicando o mesmo teste de 12 casos, **0 falhas**.

### Resultado após a correção

Os três folds foram reexecutados integralmente com a indexação corrigida, sem alterar amostragem, seed, features ou hiperparâmetros. O resultado corrigido não piorou — ficou **melhor** que a versão originalmente contaminada:

| Versão | RMSE agregado (rolling) | Delta vs B00 |
|---|---:|---:|
| M05+ONI (indexação com leakage, invalidada) | 1,8542 | -1,05% |
| M05+ONI (indexação corrigida) | 1,8514 | -1,20% |

Isso é reportado com transparência como parte do processo, não omitido: a versão contaminada aparece marcada como inválida em `results/experiments.csv`.

### Limitação de vintage documentada

A série usada (`data/external/oni_raw.txt`) é a publicação atual da NOAA CPC. Ela não representa necessariamente o valor que estaria disponível operacionalmente, em tempo real, em cada data histórica passada — séries climáticas às vezes recebem pequenas revisões retroativas de base period. Essa é uma limitação metodológica documentada, não uma equivalência perfeita a um sistema operacional histórico. Meses anteriores a 1950 (fora da cobertura da série) permanecem sem valor de ONI (NaN), sem nenhum valor inventado — o LightGBM lida com valores ausentes nativamente.

## Decomposição de erro

### Por intensidade (rolling, agregado dos 3 folds, B00)

| Faixa | % das observações | % do SSE |
|---|---:|---:|
| 0–1 mm/dia | ≈28,2% | ≈5,7–5,8% |
| 10–20 mm/dia | ≈6,35% | ≈30% |
| >20 mm/dia | ≈0,6% | ≈17–18% |

O erro quadrático é fortemente dominado por uma fração pequena de eventos de intensidade moderada a alta — não pela chuva fraca, que é a maioria das observações.

### Concentração do erro

No B00: os 1% maiores erros quadráticos respondem por ≈30,7% do SSE total; os 5% maiores, por ≈59,6%; os 10% maiores, por ≈74,3%. O M05 apresentou uma distribuição de concentração muito semelhante — evidência de que essa concentração é uma característica estrutural do problema (a distribuição de precipitação tem cauda pesada), não uma peculiaridade de um modelo específico.

### Complementaridade entre modelos (para decisão de ensemble)

A correlação de Pearson entre os resíduos de B00 e M05+ONI, calculada sobre as mesmas 3 janelas de validação, foi de **0,981** — extremamente alta. Por esse motivo, não foi realizada busca extensa de pesos de ensemble: a evidência disponível já indicava baixo potencial de ganho incremental combinando os dois modelos linearmente.

## Experimentos descartados (resumo qualitativo)

Além dos listados no README, vale registrar o que foi tentado e por que não avançou:

- **`lag_meses` como feature**: auditoria mostrou que não existe correspondência defensável entre exemplos históricos de treino e os horizontes de 1–24 meses do teste real (o "atraso" do teste mede distância até uma âncora fixa, não frescor de informação — as features atmosféricas são sempre igualmente "frescas", independente do horizonte). Classificado como hipótese não testável com a estrutura de dados disponível.
- **Tuning extensivo de hiperparâmetros**: uma busca pequena e racional (7 configurações) já mostrou retorno decrescente claro após a redução inicial de `num_leaves`; não foi expandida.
- **Volume de treino acima de 1,2M exemplos**: não testado, por restrição de memória da máquina de desenvolvimento (8 GB).
