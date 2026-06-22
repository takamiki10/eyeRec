"""YAML configuration loader."""

from pathlib import Path
from typing import Union

import yaml


def load_yaml(path: Union[str, Path]) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}
