from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_source_split(raw_dir: Path, source_split: str) -> Path:
    candidates = [source_split]
    if source_split == "valid":
        candidates.append("val")

    for candidate in candidates:
        source_root = raw_dir / candidate
        if source_root.exists():
            return source_root

    raise FileNotFoundError(
        f"Expected split folder not found under {raw_dir}: {' or '.join(candidates)}. "
        "Place images and labels CSV files under data/raw/."
    )


def resolve_labels_csv(raw_dir: Path, source_split: str) -> Path:
    candidates = [raw_dir / f"{source_split}_labels.csv", raw_dir / source_split / "labels.csv"]
    if source_split == "valid":
        candidates.extend([raw_dir / "val_labels.csv", raw_dir / "val" / "labels.csv"])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Expected labels CSV for split '{source_split}'. "
        "Use <split>_labels.csv in data/raw/ or labels.csv inside the split folder."
    )


def copy_split(raw_dir: Path, output_dir: Path, source_split: str, target_split: str) -> None:
    source_root = resolve_source_split(raw_dir, source_split)
    source_labels = resolve_labels_csv(raw_dir, source_split)
    target_root = output_dir / target_split
    target_labels = output_dir / f"{target_split}_labels.csv"

    if target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied_rows: list[dict[str, str]] = []
    with source_labels.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {"image_path", "pupil_distance"}
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            raise ValueError(f"Expected CSV columns {sorted(required_columns)} in {source_labels}.")

        for row in reader:
            source_image = Path(row["image_path"].strip())
            if not source_image.is_absolute():
                source_image = source_root / source_image
            if source_image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if not source_image.exists():
                raise FileNotFoundError(f"Image listed in {source_labels} was not found: {source_image}")

            destination = target_root / source_image.name
            if destination.exists():
                destination = target_root / f"{source_image.stem}_{len(copied_rows)}{source_image.suffix}"
            shutil.copy2(source_image, destination)
            copied_rows.append({"image_path": destination.name, "pupil_distance": row["pupil_distance"]})

    with target_labels.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["image_path", "pupil_distance"])
        writer.writeheader()
        writer.writerows(copied_rows)

    print(f"{source_root.name} -> {target_split}: {len(copied_rows)} labeled images")


def prepare_dataset(raw_dir: str | Path, output_dir: str | Path) -> None:
    project_dir = Path(__file__).resolve().parents[1]
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    if not raw_dir.is_absolute():
        raw_dir = project_dir / raw_dir
    if not output_dir.is_absolute():
        output_dir = project_dir / output_dir

    split_map = {"train": "train", "valid": "val", "test": "test"}
    for source_split, target_split in split_map.items():
        copy_split(raw_dir, output_dir, source_split, target_split)

    print(f"Prepared dataset at: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local pupil distance regression dataset.")
    parser.add_argument("--raw_dir", default="data/raw", help="Folder containing source images and labels.")
    parser.add_argument("--output_dir", default="data/processed", help="Folder to receive train/val/test data.")
    args = parser.parse_args()
    prepare_dataset(args.raw_dir, args.output_dir)


if __name__ == "__main__":
    main()
