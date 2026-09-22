"""Baixa os dados da competição via kagglehub e copia para data/raw/."""

import shutil
from pathlib import Path

import kagglehub

DEST = Path(__file__).resolve().parent.parent / "data" / "raw"

if __name__ == "__main__":
    path = kagglehub.competition_download(
        "previsao-climatica-de-precipitacao-sobre-a-america-do-sul"
    )
    print("Path to competition files:", path)

    DEST.mkdir(parents=True, exist_ok=True)
    for item in Path(path).iterdir():
        target = DEST / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)

    print(f"Arquivos copiados para {DEST}")
