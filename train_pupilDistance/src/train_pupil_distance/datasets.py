from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class PupilDistanceDataset(Dataset):
    def __init__(
        self,
        image_dir: str | Path,
        labels_csv: str | Path,
        transform: Callable | None = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.labels_csv = Path(labels_csv)
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self) -> list[tuple[Path, float]]:
        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Image folder not found: {self.image_dir}. "
                "Expected split images under data/processed/train, val, or test."
            )
        if not self.labels_csv.exists():
            raise FileNotFoundError(
                f"Labels CSV not found: {self.labels_csv}. "
                "Expected columns: image_path,pupil_distance."
            )

        samples: list[tuple[Path, float]] = []
        with self.labels_csv.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            required_columns = {"image_path", "pupil_distance"}
            if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
                raise ValueError(f"Expected CSV columns {sorted(required_columns)} in {self.labels_csv}.")

            for row in reader:
                relative_path = row["image_path"].strip()
                value = float(row["pupil_distance"])
                image_path = Path(relative_path)
                if not image_path.is_absolute():
                    image_path = self.image_dir / image_path
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                if not image_path.exists():
                    raise FileNotFoundError(f"Image listed in {self.labels_csv} was not found: {image_path}")
                samples.append((image_path, value))

        if not samples:
            raise FileNotFoundError(
                f"No labeled images found from {self.labels_csv}. "
                "CSV rows should use image_path values relative to the split image folder."
            )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, value = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(value, dtype=torch.float32)


def build_train_transforms(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_eval_transforms(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
