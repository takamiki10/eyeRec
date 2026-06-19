from __future__ import annotations

import argparse
import shutil
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_ALIASES: dict[str, str] = {}


def load_labels(project_dir: Path) -> list[str]:
    mapping_path = project_dir / "configs" / "label_mapping.yaml"
    with mapping_path.open("r", encoding="utf-8") as file:
        if yaml is not None:
            data = yaml.safe_load(file)
            return list(data["labels"].keys())

        labels: list[str] = []
        in_labels_block = False
        for line in file:
            stripped = line.strip()
            if stripped == "labels:":
                in_labels_block = True
                continue
            if in_labels_block and stripped and not line.startswith(" "):
                break
            if in_labels_block and ":" in stripped:
                labels.append(stripped.split(":", 1)[0])

    if not labels:
        raise ValueError(f"Could not read labels from {mapping_path}. Install PyYAML or check the file format.")
    return labels


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
        "Place the manually downloaded Roboflow image-classification dataset under data/raw/."
    )


def target_labels_for_source(source_label: str, labels: list[str]) -> list[str]:
    normalized_label = source_label.lower()
    if normalized_label in labels:
        return [normalized_label]

    alias_target = CLASS_ALIASES.get(normalized_label)
    if alias_target in labels:
        return [alias_target]

    return []


def copy_images(source_dir: Path, target_dir: Path) -> int:
    copied_count = 0
    target_dir.mkdir(parents=True, exist_ok=True)

    for image_path in sorted(source_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        destination = target_dir / image_path.name
        if destination.exists():
            destination = target_dir / f"{image_path.stem}_{copied_count}{image_path.suffix}"
        shutil.copy2(image_path, destination)
        copied_count += 1

    return copied_count


def copy_split(raw_dir: Path, output_dir: Path, source_split: str, target_split: str, labels: list[str]) -> None:
    source_root = resolve_source_split(raw_dir, source_split)

    target_root = output_dir / target_split
    if target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    counts = {label: 0 for label in labels}
    source_label_dirs = [path for path in source_root.iterdir() if path.is_dir()]
    ignored_labels: list[str] = []

    for source_label_dir in source_label_dirs:
        target_labels = target_labels_for_source(source_label_dir.name, labels)
        if not target_labels:
            ignored_labels.append(source_label_dir.name)
            continue

        for target_label in target_labels:
            counts[target_label] += copy_images(source_label_dir, target_root / target_label)

    for label in labels:
        (target_root / label).mkdir(parents=True, exist_ok=True)

    empty_labels = [label for label, count in counts.items() if count == 0]
    print(f"{source_root.name} -> {target_split}:")
    for label, count in counts.items():
        print(f"  {label}: {count} images")

    if empty_labels:
        print(
            "  warning: no images found for configured labels: "
            f"{', '.join(empty_labels)}. Empty folders were created so the schema stays stable."
        )
    if ignored_labels:
        print(f"  warning: ignored unconfigured labels: {', '.join(sorted(ignored_labels))}")


def prepare_dataset(raw_dir: str | Path, output_dir: str | Path) -> None:
    project_dir = Path(__file__).resolve().parents[1]
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    if not raw_dir.is_absolute():
        raw_dir = project_dir / raw_dir
    if not output_dir.is_absolute():
        output_dir = project_dir / output_dir

    labels = load_labels(project_dir)
    split_map = {"train": "train", "valid": "val", "test": "test"}

    for source_split, target_split in split_map.items():
        copy_split(raw_dir, output_dir, source_split, target_split, labels)

    print(f"Prepared dataset at: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local eye shape dataset folders.")
    parser.add_argument("--raw_dir", default="data/raw", help="Folder containing manually downloaded dataset.")
    parser.add_argument("--output_dir", default="data/processed", help="Folder to receive train/val/test folders.")
    args = parser.parse_args()
    prepare_dataset(args.raw_dir, args.output_dir)


if __name__ == "__main__":
    main()
