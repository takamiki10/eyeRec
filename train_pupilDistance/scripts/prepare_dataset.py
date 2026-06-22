from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PUPIL_DIAMETER_COLUMNS = ("Pupil diameter left", "Pupil diameter right")


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
    if (source_root / "images").exists() and (source_root / "labels").exists():
        copy_yolo_pupil_distance_split(source_root, output_dir, target_split)
        return

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


def read_yolo_centers(annotation_path: Path) -> list[tuple[float, float, float]]:
    centers: list[tuple[float, float, float]] = []
    with annotation_path.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            centers.append((x_center, y_center, width * height))
    return centers


def pupil_distance_from_yolo(annotation_path: Path) -> float | None:
    centers = read_yolo_centers(annotation_path)
    if len(centers) < 2:
        return None

    largest_two = sorted(centers, key=lambda item: item[2], reverse=True)[:2]
    left, right = sorted(largest_two, key=lambda item: item[0])
    return math.sqrt((right[0] - left[0]) ** 2 + (right[1] - left[1]) ** 2)


def copy_yolo_pupil_distance_split(source_root: Path, output_dir: Path, target_split: str) -> None:
    images_root = source_root / "images"
    labels_root = source_root / "labels"
    target_root = output_dir / target_split
    target_labels = output_dir / f"{target_split}_labels.csv"

    if target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    skipped = 0
    for image_path in sorted(images_root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        annotation_path = labels_root / f"{image_path.stem}.txt"
        if not annotation_path.exists():
            skipped += 1
            continue

        pupil_distance = pupil_distance_from_yolo(annotation_path)
        if pupil_distance is None:
            skipped += 1
            continue

        destination = target_root / image_path.name
        if destination.exists():
            destination = target_root / f"{image_path.stem}_{len(rows)}{image_path.suffix}"
        shutil.copy2(image_path, destination)
        rows.append({"image_path": destination.name, "pupil_distance": f"{pupil_distance:.8f}"})

    with target_labels.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["image_path", "pupil_distance"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"{source_root.name} YOLO -> {target_split}: {len(rows)} labeled images")
    if skipped:
        print(f"  warning: skipped {skipped} images without two pupil annotations")


def parse_number(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip().replace(",", ".")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def delimiter_for(path: Path) -> str:
    return "\t" if path.suffix.lower() == ".tsv" else ","


def find_eye_tracking_logs(raw_dir: Path) -> list[Path]:
    candidates = [*raw_dir.rglob("*.tsv"), *raw_dir.rglob("*.csv")]
    return sorted(
        path
        for path in candidates
        if "Questionnaire" not in path.name and "Pupil Diameters" in str(path)
    )


def summarize_pupil_log(path: Path) -> dict[str, str] | None:
    values: list[float] = []
    participant = path.stem
    media_names: set[str] = set()

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter=delimiter_for(path))
        if not reader.fieldnames or not any(column in reader.fieldnames for column in PUPIL_DIAMETER_COLUMNS):
            return None

        for row in reader:
            participant = row.get("Participant name") or participant
            media_name = (row.get("Presented Media name") or "").strip()
            if media_name:
                media_names.add(media_name)

            row_values = [parse_number(row.get(column)) for column in PUPIL_DIAMETER_COLUMNS]
            row_values = [value for value in row_values if value is not None]
            if row_values:
                values.append(sum(row_values) / len(row_values))

    if not values:
        return None

    return {
        "source_file": str(path),
        "participant": participant,
        "sample_count": str(len(values)),
        "mean_pupil_diameter": f"{sum(values) / len(values):.6f}",
        "min_pupil_diameter": f"{min(values):.6f}",
        "max_pupil_diameter": f"{max(values):.6f}",
        "presented_media_count": str(len(media_names)),
    }


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "source_file",
        "participant",
        "sample_count",
        "mean_pupil_diameter",
        "min_pupil_diameter",
        "max_pupil_diameter",
        "presented_media_count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_eye_tracking_summary(raw_dir: Path, output_dir: Path) -> bool:
    log_paths = find_eye_tracking_logs(raw_dir)
    if not log_paths:
        return False

    rows = [row for path in log_paths if (row := summarize_pupil_log(path)) is not None]
    if not rows:
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(output_dir / "pupil_diameter_summary.csv", rows)

    train_end = max(1, int(len(rows) * 0.70))
    val_end = max(train_end + 1, int(len(rows) * 0.85)) if len(rows) > 1 else train_end
    write_rows(output_dir / "train_pupil_diameter_summary.csv", rows[:train_end])
    write_rows(output_dir / "val_pupil_diameter_summary.csv", rows[train_end:val_end])
    write_rows(output_dir / "test_pupil_diameter_summary.csv", rows[val_end:])

    print(f"Prepared pupil diameter summaries at: {output_dir}")
    print(f"  total recordings: {len(rows)}")
    print(f"  train recordings: {len(rows[:train_end])}")
    print(f"  val recordings: {len(rows[train_end:val_end])}")
    print(f"  test recordings: {len(rows[val_end:])}")
    print(
        "  note: this dataset contains eye-tracking pupil diameter logs."
    )
    return True


def prepare_dataset(raw_dir: str | Path, output_dir: str | Path) -> None:
    project_dir = Path(__file__).resolve().parents[1]
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    if not raw_dir.is_absolute():
        raw_dir = project_dir / raw_dir
    if not output_dir.is_absolute():
        output_dir = project_dir / output_dir

    if not (raw_dir / "train").exists() and prepare_eye_tracking_summary(raw_dir, output_dir):
        return

    split_map = {"train": "train", "valid": "val", "test": "test"}
    for source_split, target_split in split_map.items():
        copy_split(raw_dir, output_dir, source_split, target_split)

    print(f"Prepared dataset at: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local pupil distance or pupil diameter dataset.")
    parser.add_argument("--raw_dir", default="data/raw", help="Folder containing source images and labels.")
    parser.add_argument("--output_dir", default="data/processed", help="Folder to receive train/val/test data.")
    args = parser.parse_args()
    prepare_dataset(args.raw_dir, args.output_dir)


if __name__ == "__main__":
    main()
