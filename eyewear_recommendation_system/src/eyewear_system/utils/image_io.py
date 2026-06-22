"""Image loading helpers."""

from pathlib import Path
from typing import Union

from PIL import Image


def load_image(image_path: Union[str, Path]):
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")
