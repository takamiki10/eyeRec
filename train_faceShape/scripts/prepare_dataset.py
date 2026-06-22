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
FACE_CROP_PADDING = 0.35
CLASS_ALIASES: dict[str, str] = {
    "long": "oblong",
}
SPLIT_ALIASES = {
    "train": ["train", "training", "training_set"],
    "valid": ["valid", "validation", "validation_set", "val"],
    "test": ["test", "testing", "testing_set"],
}


def load_face_detector():
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise FileNotFoundError(f"Could not load OpenCV face cascade: {cascade_path}")
    return detector


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
    candidates = SPLIT_ALIASES.get(source_split, [source_split])
    for candidate in candidates:
        direct_root = raw_dir / candidate
        if direct_root.exists():
            return direct_root

    for candidate in candidates:
        matches = [path for path in raw_dir.rglob(candidate) if path.is_dir()]
        if matches:
            return sorted(matches, key=lambda path: len(path.parts))[0]

    raise FileNotFoundError(
        f"Expected split folder not found under {raw_dir}: {' or '.join(candidates)}. "
        "Place the face-shape dataset under data/raw/."
    )


def discover_dataset_roots(raw_dir: Path) -> list[Path]:
    roots: list[Path] = []
    if any((raw_dir / alias).exists() for alias in SPLIT_ALIASES["train"]):
        roots.append(raw_dir)

    for child in sorted(path for path in raw_dir.iterdir() if path.is_dir()):
        if any((child / alias).exists() for alias in SPLIT_ALIASES["train"]):
            roots.append(child)

    return roots or [raw_dir]


def make_validation_split_from_train(output_dir: Path, labels: list[str], every_nth: int = 5) -> None:
    train_root = output_dir / "train"
    val_root = output_dir / "val"
    if not train_root.exists():
        return

    val_root.mkdir(parents=True, exist_ok=True)
    for label in labels:
        train_label_dir = train_root / label
        val_label_dir = val_root / label
        val_label_dir.mkdir(parents=True, exist_ok=True)
        if not train_label_dir.exists():
            continue

        images = sorted(path for path in train_label_dir.iterdir() if path.is_file())
        for index, image_path in enumerate(images):
            if index % every_nth != 0:
                continue
            destination = val_label_dir / image_path.name
            if destination.exists():
                destination = val_label_dir / f"{image_path.stem}_{index}{image_path.suffix}"
            shutil.move(str(image_path), str(destination))

    print("Created validation split from every fifth training image.")


def target_labels_for_source(source_label: str, labels: list[str]) -> list[str]:
    normalized_label = source_label.lower().strip().replace("_", "-").replace(" ", "-")
    if normalized_label in labels:
        return [normalized_label]

    alias_target = CLASS_ALIASES.get(normalized_label)
    if alias_target in labels:
        return [alias_target]

    return []


def label_from_flat_filename(image_path: Path, labels: list[str]) -> list[str]:
    stem = image_path.stem.lower()
    normalized_stem = stem.replace("_", "-").replace(" ", "-")
    prefix = normalized_stem.split("-", 1)[0]
    return target_labels_for_source(prefix, labels)


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


def crop_face(image_path: Path, face_detector):
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(48, 48),
    )
    if len(faces) == 0:
        return image

    x, y, width, height = max(faces, key=lambda face: face[2] * face[3])
    image_height, image_width = image.shape[:2]
    pad_x = int(width * FACE_CROP_PADDING)
    pad_y = int(height * FACE_CROP_PADDING)
    left = clamp(x - pad_x, 0, image_width)
    top = clamp(y - pad_y, 0, image_height)
    right = clamp(x + width + pad_x, 0, image_width)
    bottom = clamp(y + height + pad_y, 0, image_height)
    if right <= left or bottom <= top:
        return image

    return image[top:bottom, left:right]


def save_prepared_image(image_path: Path, destination: Path, face_detector) -> bool:
    crop = crop_face(image_path, face_detector)
    if crop is None:
        print(f"  warning: could not read image: {image_path}")
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(destination), crop))


def copy_images(source_dir: Path, target_dir: Path, face_detector, dataset_prefix: str = "") -> int:
    copied_count = 0
    target_dir.mkdir(parents=True, exist_ok=True)

    for image_path in sorted(source_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        filename = f"{dataset_prefix}_{image_path.name}" if dataset_prefix else image_path.name
        destination = target_dir / filename
        if destination.exists():
            destination = target_dir / f"{image_path.stem}_{copied_count}{image_path.suffix}"
        if save_prepared_image(image_path, destination, face_detector):
            copied_count += 1

    return copied_count


def copy_split(
    raw_dir: Path,
    output_dir: Path,
    source_split: str,
    target_split: str,
    labels: list[str],
    face_detector,
    clear_target: bool = True,
) -> None:
    source_root = resolve_source_split(raw_dir, source_split)

    target_root = output_dir / target_split
    if clear_target and target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    counts = {label: 0 for label in labels}
    source_label_dirs = [path for path in source_root.iterdir() if path.is_dir()]
    ignored_labels: list[str] = []
    safe_dataset_name = raw_dir.name.replace(" ", "_")

    if source_label_dirs:
        for source_label_dir in source_label_dirs:
            target_labels = target_labels_for_source(source_label_dir.name, labels)
            if not target_labels:
                ignored_labels.append(source_label_dir.name)
                continue

            for target_label in target_labels:
                counts[target_label] += copy_images(
                    source_label_dir,
                    target_root / target_label,
                    face_detector,
                    dataset_prefix=safe_dataset_name,
                )
    else:
        for image_path in sorted(source_root.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            target_labels = label_from_flat_filename(image_path, labels)
            if not target_labels:
                ignored_labels.append(image_path.name)
                continue

            for target_label in target_labels:
                destination_dir = target_root / target_label
                destination_dir.mkdir(parents=True, exist_ok=True)
                destination = destination_dir / f"{safe_dataset_name}_{image_path.name}"
                if destination.exists():
                    destination = destination_dir / (
                        f"{safe_dataset_name}_{image_path.stem}_{counts[target_label]}{image_path.suffix}"
                    )
                if save_prepared_image(image_path, destination, face_detector):
                    counts[target_label] += 1

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
    face_detector = load_face_detector()
    split_map = {"train": "train", "valid": "val", "test": "test"}
    dataset_roots = discover_dataset_roots(raw_dir)
    print("Preparing face-shape datasets:")
    for dataset_root in dataset_roots:
        print(f"  - {dataset_root}")

    for source_split, target_split in split_map.items():
        copied_any = False
        for index, dataset_root in enumerate(dataset_roots):
            try:
                copy_split(
                    dataset_root,
                    output_dir,
                    source_split,
                    target_split,
                    labels,
                    face_detector,
                    clear_target=index == 0,
                )
                copied_any = True
            except FileNotFoundError:
                if source_split == "valid":
                    continue
                raise

        if source_split == "valid" and not copied_any:
            make_validation_split_from_train(output_dir, labels)

    print(f"Prepared dataset at: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local face shape dataset folders.")
    parser.add_argument("--raw_dir", default="data/raw", help="Folder containing manually downloaded dataset.")
    parser.add_argument("--output_dir", default="data/processed", help="Folder to receive train/val/test folders.")
    args = parser.parse_args()
    prepare_dataset(args.raw_dir, args.output_dir)


if __name__ == "__main__":
    main()
