from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2

try:
    import yaml
except ImportError:
    yaml = None

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EYE_CROP_PADDING = 0.85
CLASS_ALIASES: dict[str, str] = {
    "deep-set": "deep_set",
    "protuding": "protruding",
}


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
        "Place the manually downloaded Roboflow dataset under data/raw/."
    )


def discover_dataset_roots(raw_dir: Path) -> list[Path]:
    roots: list[Path] = []
    if (raw_dir / "data.yaml").exists() or (raw_dir / "train").exists():
        roots.append(raw_dir)

    for child in sorted(path for path in raw_dir.iterdir() if path.is_dir()):
        if (child / "data.yaml").exists() or (child / "train").exists():
            roots.append(child)

    return roots or [raw_dir]


def load_yolo_class_names(raw_dir: Path) -> list[str]:
    yaml_path = raw_dir / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"YOLO dataset detected but missing class config: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as file:
        if yaml is not None:
            data = yaml.safe_load(file) or {}
            names = data.get("names")
            if isinstance(names, dict):
                return [str(names[index]) for index in sorted(names)]
            if isinstance(names, list):
                return [str(name) for name in names]

        in_names_block = False
        names: list[str] = []
        for line in file:
            stripped = line.strip()
            if stripped == "names:":
                in_names_block = True
                continue
            if in_names_block and stripped.startswith("-"):
                names.append(stripped[1:].strip().strip("'\""))
            elif in_names_block and stripped:
                break

    if not names:
        raise ValueError(f"Could not read YOLO class names from {yaml_path}.")
    return names


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


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


def crop_annotation(image, annotation: list[str]):
    if len(annotation) < 5:
        return None

    image_height, image_width = image.shape[:2]
    try:
        coordinates = [float(value) for value in annotation[1:]]
    except ValueError:
        return None

    if len(coordinates) == 4:
        center_x, center_y, width, height = coordinates
        left = int((center_x - width / 2) * image_width)
        top = int((center_y - height / 2) * image_height)
        right = int((center_x + width / 2) * image_width)
        bottom = int((center_y + height / 2) * image_height)
    elif len(coordinates) >= 6 and len(coordinates) % 2 == 0:
        points_x = coordinates[0::2]
        points_y = coordinates[1::2]
        center_x = sum(points_x) / len(points_x)
        left = int(min(points_x) * image_width)
        top = int(min(points_y) * image_height)
        right = int(max(points_x) * image_width)
        bottom = int(max(points_y) * image_height)
    else:
        return None

    width = right - left
    height = bottom - top
    pad_x = int(width * EYE_CROP_PADDING)
    pad_y = int(height * EYE_CROP_PADDING)
    left = clamp(left - pad_x, 0, image_width)
    top = clamp(top - pad_y, 0, image_height)
    right = clamp(right + pad_x, 0, image_width)
    bottom = clamp(bottom + pad_y, 0, image_height)
    if right <= left or bottom <= top:
        return None

    crop = image[top:bottom, left:right]
    if center_x < 0.5:
        crop = cv2.flip(crop, 1)
    return crop


def copy_yolo_split(
    raw_dir: Path,
    output_dir: Path,
    source_split: str,
    target_split: str,
    labels: list[str],
    clear_target: bool = True,
) -> None:
    source_root = resolve_source_split(raw_dir, source_split)
    images_root = source_root / "images"
    annotations_root = source_root / "labels"
    if not images_root.exists() or not annotations_root.exists():
        raise FileNotFoundError(
            f"Expected YOLO split folders under {source_root}: images/ and labels/."
        )

    class_names = load_yolo_class_names(raw_dir)
    target_root = output_dir / target_split
    if clear_target and target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    counts = {label: 0 for label in labels}
    ignored_labels: list[str] = []

    for image_path in sorted(images_root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        annotation_path = annotations_root / f"{image_path.stem}.txt"
        if not annotation_path.exists():
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  warning: could not read image: {image_path}")
            continue

        with annotation_path.open("r", encoding="utf-8") as file:
            for annotation_index, line in enumerate(file):
                parts = line.strip().split()
                if not parts:
                    continue

                class_index = int(float(parts[0]))
                if class_index >= len(class_names):
                    continue

                target_labels = target_labels_for_source(class_names[class_index], labels)
                if not target_labels:
                    ignored_labels.append(class_names[class_index])
                    continue

                crop = crop_annotation(image, parts)
                if crop is None:
                    continue

                for target_label in target_labels:
                    destination_dir = target_root / target_label
                    destination_dir.mkdir(parents=True, exist_ok=True)
                    safe_dataset_name = raw_dir.name.replace(" ", "_")
                    destination = destination_dir / (
                        f"{safe_dataset_name}_{image_path.stem}_eye{annotation_index + 1}_"
                        f"{counts[target_label]}{image_path.suffix}"
                    )
                    cv2.imwrite(str(destination), crop)
                    counts[target_label] += 1

    for label in labels:
        (target_root / label).mkdir(parents=True, exist_ok=True)

    empty_labels = [label for label, count in counts.items() if count == 0]
    print(f"{source_root.name} YOLO -> {target_split}:")
    for label, count in counts.items():
        print(f"  {label}: {count} images")
    if empty_labels:
        print(f"  warning: no images found for configured labels: {', '.join(empty_labels)}")
    if ignored_labels:
        print(f"  warning: ignored unconfigured labels: {', '.join(sorted(set(ignored_labels)))}")


def copy_split(
    raw_dir: Path,
    output_dir: Path,
    source_split: str,
    target_split: str,
    labels: list[str],
    clear_target: bool = True,
) -> None:
    source_root = resolve_source_split(raw_dir, source_split)
    if (source_root / "images").exists() and (source_root / "labels").exists():
        copy_yolo_split(raw_dir, output_dir, source_split, target_split, labels, clear_target=clear_target)
        return

    target_root = output_dir / target_split
    if clear_target and target_root.exists():
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
    dataset_roots = discover_dataset_roots(raw_dir)
    print("Preparing eye-shape datasets:")
    for dataset_root in dataset_roots:
        print(f"  - {dataset_root}")

    for source_split, target_split in split_map.items():
        for index, dataset_root in enumerate(dataset_roots):
            copy_split(
                dataset_root,
                output_dir,
                source_split,
                target_split,
                labels,
                clear_target=index == 0,
            )

    print(f"Prepared dataset at: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local eye shape dataset folders.")
    parser.add_argument("--raw_dir", default="data/raw", help="Folder containing manually downloaded dataset.")
    parser.add_argument("--output_dir", default="data/processed", help="Folder to receive train/val/test folders.")
    args = parser.parse_args()
    prepare_dataset(args.raw_dir, args.output_dir)


if __name__ == "__main__":
    main()
