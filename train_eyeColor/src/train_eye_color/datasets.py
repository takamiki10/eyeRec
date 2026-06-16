from __future__ import annotations

from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class EyeColorFolderDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        label_to_index: dict[str, int],
        transform: Callable | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.label_to_index = label_to_index
        self.transform = transform
        self.samples = self._find_samples()

    def _find_samples(self) -> list[tuple[Path, int]]:
        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset folder not found: {self.root_dir}. "
                "Run scripts/prepare_dataset.py after placing the dataset in data/raw/."
            )

        samples: list[tuple[Path, int]] = []
        missing_labels: list[str] = []

        for label, index in self.label_to_index.items():
            label_dir = self.root_dir / label
            if not label_dir.exists():
                missing_labels.append(label)
                continue

            for image_path in sorted(label_dir.rglob("*")):
                if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    samples.append((image_path, index))

        if missing_labels:
            raise FileNotFoundError(
                f"Missing class folders in {self.root_dir}: {', '.join(missing_labels)}. "
                "Expected one folder per label: "
                f"{', '.join(self.label_to_index.keys())}."
            )

        if not samples:
            raise FileNotFoundError(
                f"No images found in {self.root_dir}. "
                f"Supported extensions: {', '.join(sorted(IMAGE_EXTENSIONS))}."
            )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


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
