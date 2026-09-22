# WORCAP 2026 — Previsão Climática de Precipitação sobre a América do Sul

Competição Kaggle: https://www.kaggle.com/competitions/previsao-climatica-de-precipitacao-sobre-a-america-do-sul

## Estrutura

```
data/
  raw/         # arquivos baixados do Kaggle, sem alteração
  processed/   # dados limpos/transformados prontos para modelagem
notebooks/     # exploração e prototipagem
src/           # código reutilizável (data loading, features, modelos, avaliação)
models/        # modelos treinados salvos (.pkl, .joblib, checkpoints)
submissions/   # arquivos de submissão gerados
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Configure a API do Kaggle (uma vez): baixe `kaggle.json` em https://www.kaggle.com/settings > API,
e coloque em `~/.kaggle/kaggle.json` (permissão `chmod 600`).

## Baixar os dados

Opção 1 — via `kagglehub` (mais simples, autentica no primeiro uso):

```bash
python scripts/download_data.py
```

Opção 2 — via Kaggle CLI (precisa de `kaggle.json` configurado):

```bash
bash scripts/download_data.sh
```

## Fluxo de trabalho

1. `notebooks/01_eda.ipynb` — exploração inicial dos dados.
2. `src/data.py` — carregamento e limpeza.
3. `src/features.py` — engenharia de atributos.
4. `src/train.py` — treino e validação do modelo.
5. `src/predict.py` — geração do arquivo de submissão.
