# WORCAP 2026 — Previsão Mensal de Precipitação sobre a América do Sul

Competição Kaggle: [previsao-climatica-de-precipitacao-sobre-a-america-do-sul](https://www.kaggle.com/competitions/previsao-climatica-de-precipitacao-sobre-a-america-do-sul)

## Visão geral

Este repositório documenta o desenvolvimento completo de uma solução de machine learning para prever a precipitação média mensal sobre a América do Sul, um mês à frente, usando variáveis atmosféricas derivadas do ERA5. O projeto foi conduzido de forma incremental: cada etapa (baseline, feature engineering, validação, correção de bugs) ficou registrada em um script e em um commit próprios, preservando o histórico experimental completo.

## Desafio

Dado o estado atmosférico observado em um mês `M` (temperatura, umidade, pressão, vento, etc.), prever a precipitação média diária do mês seguinte `M+1`, célula a célula, em uma grade regular de 0,25° cobrindo a América do Sul. A avaliação é feita por RMSE (mm/dia) sobre um conjunto de teste oficial de 24 meses (2023–2024).

## Dados

- **Treino**: histórico mensal de 1940-01 a 2022-12 (996 meses), grade de 301 × 261 pontos (78.561 células/mês), derivado do ERA5.
- **Teste oficial**: 2023-01 a 2024-12 (24 meses), mesma grade — 1.885.464 previsões esperadas.
- **Variáveis atmosféricas**: `t2`, `cloud_cover`, `shum_850`, `surface_pressure`, `u_850`, `v_850`, `temperature_850`, `rel_hum_850`, `geopotential_850`.
- **Dado externo**: índice ONI (Oceanic Niño Index), série pública da NOAA CPC (`data/external/oni_raw.txt`), usada como proxy do regime ENSO.

## Formulação temporal

A relação básica usada em todo o projeto é:

```
X(M) = variáveis atmosféricas do mês M
y(M) = tp_alvo(M) = precipitação observada em M+1
```

No conjunto de teste oficial, a coluna `time` representa o mês-alvo, e as variáveis atmosféricas fornecidas nessa linha correspondem ao mês de origem `time_origem = time - 1 mês` — a mesma relação usada no treino, sem nenhum deslocamento adicional. Essa equivalência foi verificada numericamente (comparação direta de valores entre treino e teste) antes de qualquer modelagem.

## Estrutura do projeto

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── external/oni_raw.txt      # índice ONI (NOAA CPC)
├── scripts/                      # um script por gate experimental (numerados em ordem)
├── results/                      # métricas, tabelas de diagnóstico e gráficos de cada gate
├── submissions/                  # CSVs de submissão gerados (fora do versionamento)
└── docs/
    └── metodologia.md            # detalhamento da auditoria de leakage e decomposição de erro
```

`data/raw/` (arquivos `.nc` baixados da competição) não é versionado — veja [Reprodutibilidade](#reprodutibilidade).

## Metodologia

### Baseline climatológico

`B00` prevê, para cada célula e mês-calendário, a média histórica de precipitação daquele mês naquele ponto, calculada apenas com dados anteriores à origem da previsão (sem nenhum vazamento de informação futura). Esse baseline se mostrou difícil de superar: o primeiro modelo de gradient boosting treinado diretamente sobre os valores absolutos de precipitação (`M01`) ficou bem pior que a climatologia (RMSE ≈ 2,25–2,26 contra ≈ 1,89–1,90 do B00), mesmo usando as mesmas variáveis atmosféricas.

### Modelo residual

Em vez de prever a precipitação diretamente, o modelo passou a prever o **resíduo** em relação à climatologia (`M03`): `y_residuo = tp_alvo - tp_climatologia`, reconstruindo a previsão final como `tp_climatologia + resíduo_previsto`. Essa mudança de formulação aproximou o modelo do baseline e mostrou o primeiro sinal real de que as variáveis atmosféricas continham informação útil além da sazonalidade.

### Anomalias atmosféricas

`M05` acrescentou, para cada uma das 9 variáveis atmosféricas, uma anomalia (`valor observado - climatologia própria da variável naquele mês/célula`) às features absolutas. A hipótese era que desvios em relação ao normal climatológico são mais informativos que valores absolutos — e as anomalias de fato passaram a responder por uma fração maior da importância do modelo do que as variáveis absolutas correspondentes.

### ONI / ENSO

A mudança estrutural mais relevante do projeto foi adicionar o índice ONI (proxy do ciclo El Niño/La Niña) como feature (`M05+ONI`). Esse foi o primeiro modelo a superar a climatologia de forma consistente nos três períodos de validação histórica testados.

## Validação

### Rolling-origin

A validação final usa três janelas históricas não sobrepostas, cada uma com origem fixa (cutoff), climatologias recalculadas exclusivamente com dados anteriores a essa origem, e 24 meses de avaliação:

| Fold | Cutoff | Período avaliado |
|---|---|---|
| F1 | 2016-12 | 2017-01 → 2018-12 |
| F2 | 2018-12 | 2019-01 → 2020-12 |
| F3 | 2020-12 | 2021-01 → 2022-12 |

O RMSE agregado é calculado a partir da soma dos erros quadráticos dos três folds (não da média simples dos RMSEs), o que evita distorção por diferenças de tamanho amostral entre folds.

### Prevenção de leakage

Regras aplicadas em todo o pipeline:

- **Treino supervisionado**: toda amostra de treino satisfaz `target_date <= cutoff`, verificado por `assert` explícito em cada fold.
- **Climatologias** (de precipitação e das 9 variáveis atmosféricas): calculadas apenas com histórico anterior ou igual ao cutoff, e mantidas congeladas durante toda a janela de avaliação.
- **Atmosfera de validação/teste**: sempre a do mês de origem legítimo (`T-1`), nunca do mês-alvo.
- **ONI**: a janela trimestral do índice só é usada depois de completamente disponível (ver seção de auditoria abaixo — esta regra foi corrigida após um erro identificado no processo).
- **Teste oficial**: nenhuma precipitação observada de 2023/2024 entra em qualquer etapa do treino ou das climatologias.

Detalhamento completo da auditoria de leakage do ONI em [docs/metodologia.md](docs/metodologia.md).

## Experimentos

| ID | Abordagem | Resultado | Decisão |
|---|---|---|---|
| B00 | Climatologia mensal espacial | Baseline forte; RMSE 1,8938/1,9003 (backtests iniciais); 1,8739 (rolling agregado) | Mantido como referência em todo o projeto |
| B01 | Persistência (precipitação da origem repetida) | Muito pior que B00 em todos os horizontes testados | Descartado |
| B02 | Blend climatologia + persistência | Melhor peso encontrado foi 100% climatologia (= B00) | Persistência descartada como componente direto |
| M01 | LightGBM direto sobre valores absolutos | RMSE ≈ 2,25–2,26, bem pior que B00 | Descartado; motivou a formulação residual |
| M02 | Climatologia como feature + valores absolutos | RMSE 1,9602/1,9518 — entre M01 e B00, não supera B00 | Motivou M03 |
| M03 | Alvo = resíduo em relação à climatologia | RMSE 1,9131/1,8987 (config inicial); 1,9048/1,8874 após tuning (`num_leaves=15`) | Base para os modelos seguintes |
| M04 | M03 + média espacial 3×3 das variáveis atmosféricas | RMSE 1,9149/1,9009 — sem melhora consistente frente a M03 | Descartado |
| M05 | M03 + anomalias das 9 variáveis atmosféricas | RMSE 1,8903/1,9005; rolling agregado 1,8696 (não superou B00 de forma robusta) | Base para M05+ONI |
| M05+ONI | M05 + índice ONI (indexação corrigida) | Rolling agregado 1,8514; supera B00 nos 3/3 folds (-1,20%) | **Modelo final** |
| CENSO | Climatologia condicionada ao regime ENSO (sem ML) | Rolling agregado 1,8917 — pior que B00 | Descartado |
| CENSO_SHRINK | CENSO com shrinkage em direção a B00 | Rolling agregado 1,8673 — levemente melhor que B00, mas bem pior que M05+ONI | Descartado |

Detalhes numéricos completos de cada experimento em [results/experiments.csv](results/experiments.csv).

## Modelo final

**M05+ONI** — LightGBM sobre o resíduo em relação à climatologia.

- **24 features**: 9 variáveis atmosféricas absolutas + 9 anomalias correspondentes + `lat`, `lon`, `month_sin`, `month_cos`, `tp_climatologia`, `oni`.
- **Target**: `y_residuo = tp_alvo - tp_climatologia`; previsão final = `tp_climatologia + resíduo_previsto`.
- **Modelo**: `LightGBM(n_estimators=300, num_leaves=15, learning_rate=0.05, random_state=42)`.
- **Treino de produção**: histórico completo até 2022-12 (995 meses elegíveis), amostra de 150 meses × 4.000 células/mês (seed=42) ≈ 600.000 exemplos.
- **Clipping**: aplicado apenas na submissão final (`max(0, previsão)`), já que a precipitação real nunca é negativa.

## Resultados

### Validação histórica (rolling-origin, RMSE)

| Modelo | F1 (2017–18) | F2 (2019–20) | F3 (2021–22) | Agregado |
|---|---:|---:|---:|---:|
| B00 | 1,8604 | 1,8673 | 1,8938 | 1,8739 |
| M05 | 1,8813 | 1,8369 | 1,8903 | 1,8696 |
| **M05+ONI** | **1,8483** | **1,8276** | **1,8779** | **1,8514** |
| CENSO | 1,8614 | 1,9091 | 1,9041 | 1,8917 |
| CENSO_SHRINK | 1,8516 | 1,8694 | 1,8808 | 1,8673 |

M05+ONI supera B00 nos três folds de forma consistente (delta agregado -1,20%).

### Submissões públicas (Kaggle)

| Submissão | Modelo | Public RMSE |
|---|---|---:|
| S00 | B00 (climatologia) | 1,85077 |
| S01 | M03 (resíduo, sem ONI) | 1,85663 |
| S02 | M05+ONI (final) | 1,84982 |

O rolling histórico é o critério usado para a seleção metodológica do modelo. O public score do Kaggle (baseado em ~50% do conjunto de teste) serve apenas como confirmação externa — o ganho relativo observado publicamente entre S00 e S02 (-0,05%) é bem menor que o observado no rolling agregado (-1,20%), o que é registrado como uma divergência real entre validação local e teste público, não resolvida no escopo deste projeto. O desempenho no leaderboard privado (a outra metade do teste) é desconhecido.

## Diagnóstico do erro

- **Intensidade**: o erro é dominado por uma fração pequena de observações. Chuva fraca (0–1 mm/dia) é ~28% das observações mas só ~6% do SSE total; eventos de 10–20 mm/dia são ~6% das observações e ~30% do SSE; eventos >20 mm/dia são ~0,6% das observações e ~17–18% do SSE.
- **Concentração do erro**: no B00, os 10% maiores erros quadráticos respondem por ~74% do SSE total — uma característica estrutural do problema (precipitação tem cauda pesada), não específica de um modelo.
- **Espaço**: o erro varia por faixa de latitude; a faixa tropical (-15° a 0°) mostrou o pior desempenho relativo do M05 em relação a B00 nos diagnósticos realizados.
- **Complementaridade B00 × M05+ONI**: a correlação entre os resíduos dos dois modelos é 0,981 (extremamente alta) — por isso não foi feita uma busca extensa de pesos de ensemble; a evidência disponível indicava baixo potencial de ganho incremental.

Detalhamento completo em [docs/metodologia.md](docs/metodologia.md).

## Reprodutibilidade

Todos os scripts fixam `seed=42` e usam apenas dados determinísticos (histórico de treino + série ONI pública). Rota mínima para reproduzir a submissão final:

```bash
# 1. ambiente
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. dados da competição (kagglehub autentica no primeiro uso)
python scripts/download_data.py

# 3. dado externo (ONI) já está em data/external/oni_raw.txt

# 4. gerar o modelo final e a submissão S02
python scripts/22_inferencia_m05_oni.py
```

Não é necessário rodar os demais scripts numerados (`01` a `21`) para reproduzir a submissão final — eles documentam o processo experimental completo (baselines, tuning, validação rolling, auditoria de leakage) que levou à escolha do M05+ONI, mas `scripts/22_inferencia_m05_oni.py` já encapsula a climatologia de produção, o treino final e a geração da submissão de ponta a ponta.

## Estrutura de arquivos

- `scripts/01` a `scripts/17`: auditoria de dados, baselines (B00–B02), primeiros modelos supervisionados (M01–M04), tuning e primeiras submissões diagnósticas (S00/S01).
- `scripts/18` a `scripts/23`: anomalias atmosféricas (M05), validação rolling-origin, integração e correção do ONI, modelo de produção (S02) e experimento de climatologia condicionada por ENSO (CENSO).
- `results/`: uma tabela (e, quando aplicável, um gráfico) por experimento, mais `experiments.csv` (registro consolidado) e `submissions.csv` (histórico de envios ao Kaggle).

## Limitações

- **Vintage do ONI**: a série usada é a publicação atual da NOAA CPC, não um arquivo histórico "como era conhecido em tempo real" em cada cutoff — uma limitação de fonte de dados documentada, não corrigível com os recursos disponíveis.
- **Amostragem de treino**: o modelo final treina com ~600 mil pontos de um universo de dezenas de milhões, por restrição de memória (8 GB) — o Gate 10/14 mostrou retorno pequeno ao aumentar esse volume, mas o efeito de usar a base completa nunca foi testado.
- **Eventos extremos**: os eventos de maior intensidade (>20 mm/dia) concentram parte desproporcional do erro, e nenhuma técnica específica para cauda pesada (ex. loss Tweedie, modelo em duas etapas) foi testada.
- **Validação limitada a três janelas**: o rolling-origin cobre 2017–2022; é uma amostra de períodos históricos, não uma garantia de generalização para 2023–2024.
- **Public leaderboard ≠ avaliação final**: representa ~50% do teste oficial; o desempenho na outra metade (avaliação privada) não é conhecido.

## Trabalhos futuros

Registrados como possibilidades não implementadas neste projeto:

- Outras teleconexões climáticas (IOD, PDO, SAM, MJO).
- Campos completos de SST, ou produtos de previsão sazonal (NMME, SEAS5).
- Features de topografia/elevação (relevante dado o efeito orográfico da Cordilheira dos Andes).
- Gradientes espaciais e medidas de convergência de umidade.
- Modelagem regionalizada (por bioma/regime climático) em vez de um modelo único para todo o domínio.
- Treino com o volume completo de dados disponível (fora do limite de amostragem usado aqui).
- Arquiteturas espaço-temporais (CNN, U-Net, ConvLSTM) para capturar estrutura espacial de forma mais rica que médias locais.
- Ensemble com modelos genuinamente complementares (a tentativa com B00 mostrou baixo potencial pela alta correlação de resíduos).

## Competências demonstradas

Python científico (NumPy, pandas, Xarray), manipulação de dados NetCDF/geoespaciais, engenharia de features para séries temporais e climatologia, gradient boosting (LightGBM), desenho e execução de validação rolling-origin, auditoria e correção de data leakage temporal, integração de dados climáticos externos (ONI/NOAA), decomposição e diagnóstico de erro, e versionamento reprodutível com Git.

## Autoria

Helena de Oliveira Valanera
