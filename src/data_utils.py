"""Общие утилиты: загрузка конфига, JSONL, seeding."""

import json
import random
from pathlib import Path

import numpy as np
import yaml


def load_config(path: str = "config.yaml") -> dict:
    """Загрузить YAML-конфиг."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: str | Path) -> list[dict]:
    """Загрузить JSONL построчно."""
    path = Path(path)
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def save_jsonl(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def set_seed(seed: int) -> None:
    """Воспроизводимость для всех уровней."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dir(path: str | Path) -> Path:
    """Создать директорию если её нет, вернуть Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
