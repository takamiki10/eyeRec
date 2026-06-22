from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, base_dir: Path | None = None) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (base_dir or project_root()) / path


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data or {}


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def load_label_mapping(path: str | Path) -> dict[str, int]:
    data = load_yaml(path)
    labels = data.get("labels")
    if not isinstance(labels, dict) or not labels:
        raise ValueError(f"Expected a non-empty 'labels' mapping in {path}.")
    return {str(label): int(index) for label, index in labels.items()}


def invert_mapping(label_to_index: dict[str, int]) -> dict[int, str]:
    return {index: label for label, index in label_to_index.items()}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_name: str = "auto") -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)
