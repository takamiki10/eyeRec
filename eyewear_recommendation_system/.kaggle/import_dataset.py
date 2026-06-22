from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

DATASETS = {
    "eye-shape": {
        "kind": "roboflow",
        "workspace": "eyeshapes",
        "project": "eye-shapes",
        "version": 2,
        "format": "yolov8",
        "destination": REPO_ROOT / "train_eyeShape" / "data" / "raw",
        "prepare_script": REPO_ROOT / "train_eyeShape" / "scripts" / "prepare_dataset.py",
    },
    "face-shape": {
        "kind": "kaggle",
        "slug": "niten19/face-shape-dataset",
        "destination": REPO_ROOT / "train_faceShape" / "data" / "raw",
        "prepare_script": REPO_ROOT / "train_faceShape" / "scripts" / "prepare_dataset.py",
        "labels": ["oval", "round", "square", "heart", "diamond", "oblong"],
    },
    "pupil-distance": {
        "kind": "kaggle",
        "slug": "vijuls/pupildiameterdatasets",
        "destination": REPO_ROOT / "train_pupilDistance" / "data" / "raw",
        "prepare_script": REPO_ROOT / "train_pupilDistance" / "scripts" / "prepare_dataset.py",
    },
}


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, check=True, cwd=REPO_ROOT, env=env)


def kaggle_env() -> dict[str, str]:
    env = os.environ.copy()
    local_config = Path(__file__).resolve().parent / "kaggle.json"
    if local_config.exists():
        env["KAGGLE_CONFIG_DIR"] = str(local_config.parent)
    return env


def download_kaggle(config: dict[str, object]) -> None:
    destination = Path(config["destination"])
    destination.mkdir(parents=True, exist_ok=True)
    run(
        [
            str(REPO_ROOT / ".venv" / "bin" / "kaggle"),
            "datasets",
            "download",
            "-d",
            str(config["slug"]),
            "-p",
            str(destination),
            "--unzip",
        ],
        env=kaggle_env(),
    )


def class_label(path: Path, labels: list[str]) -> str | None:
    normalized = path.name.lower().replace(" ", "_").replace("-", "_")
    for label in labels:
        if normalized == label:
            return label
    return None


def split_name(path: Path) -> str | None:
    normalized = path.name.lower().replace(" ", "_").replace("-", "_")
    if normalized in {"train", "training", "training_set"}:
        return "train"
    if normalized in {"valid", "validation", "validation_set", "val"}:
        return "valid"
    if normalized in {"test", "testing", "testing_set"}:
        return "test"
    return None


def copy_class_tree(source_split: Path, target_split: Path, labels: list[str]) -> int:
    copied = 0
    for class_dir in sorted(path for path in source_split.iterdir() if path.is_dir()):
        label = class_label(class_dir, labels)
        if label is None:
            continue

        destination_dir = target_split / label
        destination_dir.mkdir(parents=True, exist_ok=True)
        for image_path in sorted(class_dir.rglob("*")):
            if not image_path.is_file():
                continue
            destination = destination_dir / image_path.name
            if destination.exists():
                destination = destination_dir / f"{image_path.stem}_{copied}{image_path.suffix}"
            shutil.copy2(image_path, destination)
            copied += 1
    return copied


def create_validation_split(train_dir: Path, valid_dir: Path, labels: list[str], every_nth: int = 5) -> None:
    valid_dir.mkdir(parents=True, exist_ok=True)
    for label in labels:
        source_dir = train_dir / label
        target_dir = valid_dir / label
        target_dir.mkdir(parents=True, exist_ok=True)
        if not source_dir.exists():
            continue

        images = sorted(path for path in source_dir.iterdir() if path.is_file())
        for index, image_path in enumerate(images):
            if index % every_nth != 0:
                continue
            destination = target_dir / image_path.name
            if destination.exists():
                destination = target_dir / f"{image_path.stem}_{index}{image_path.suffix}"
            shutil.move(str(image_path), str(destination))


def normalize_face_shape_download(config: dict[str, object]) -> None:
    destination = Path(config["destination"])
    labels = list(config["labels"])
    if (destination / "train").exists() and ((destination / "valid").exists() or (destination / "val").exists()):
        return

    split_dirs: dict[str, Path] = {}
    for path in destination.rglob("*"):
        if not path.is_dir():
            continue
        name = split_name(path)
        if name and name not in split_dirs:
            split_dirs[name] = path

    if "train" not in split_dirs:
        print("warning: could not find a face-shape training split to normalize.")
        return

    normalized_root = destination / "_normalized"
    if normalized_root.exists():
        shutil.rmtree(normalized_root)
    normalized_root.mkdir(parents=True)

    for source_name, target_name in {"train": "train", "valid": "valid", "test": "test"}.items():
        source_split = split_dirs.get(source_name)
        if source_split is None:
            continue
        copied = copy_class_tree(source_split, normalized_root / target_name, labels)
        print(f"normalized {source_split} -> {target_name}: {copied} files")

    if not (normalized_root / "valid").exists():
        create_validation_split(normalized_root / "train", normalized_root / "valid", labels)
        print("created validation split from every fifth training image")

    for split in ["train", "valid", "test"]:
        source = normalized_root / split
        if not source.exists():
            continue
        target = destination / split
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(source), str(target))
    shutil.rmtree(normalized_root, ignore_errors=True)


def download_roboflow(config: dict[str, object]) -> None:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Set ROBOFLOW_API_KEY before downloading from Roboflow.")

    try:
        from roboflow import Roboflow
    except ImportError as error:
        raise SystemExit("Install the Roboflow SDK first: pip install roboflow") from error

    destination = Path(config["destination"])
    destination.mkdir(parents=True, exist_ok=True)
    download_root = destination / "_roboflow_download"
    if download_root.exists():
        shutil.rmtree(download_root)
    download_root.mkdir(parents=True)

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(str(config["workspace"])).project(str(config["project"]))
    dataset = project.version(int(config["version"])).download(str(config["format"]), location=str(download_root))

    dataset_path = Path(dataset.location)
    for item in dataset_path.iterdir():
        target = destination / item.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(item), str(target))
    shutil.rmtree(download_root, ignore_errors=True)


def prepare_dataset(config: dict[str, object]) -> None:
    prepare_script = Path(config["prepare_script"])
    run([sys.executable, str(prepare_script), "--raw_dir", "data/raw", "--output_dir", "data/processed"])


def import_dataset(name: str, prepare: bool) -> None:
    config = DATASETS[name]
    if config["kind"] == "kaggle":
        download_kaggle(config)
        if name == "face-shape":
            normalize_face_shape_download(config)
    elif config["kind"] == "roboflow":
        download_roboflow(config)
    else:
        raise ValueError(f"Unsupported dataset kind: {config['kind']}")

    if prepare:
        prepare_dataset(config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download EyeRec training datasets into their feature folders.")
    parser.add_argument("dataset", choices=[*DATASETS.keys(), "all"])
    parser.add_argument("--prepare", action="store_true", help="Run the feature folder prepare_dataset.py after download.")
    args = parser.parse_args()

    names = DATASETS.keys() if args.dataset == "all" else [args.dataset]
    for name in names:
        import_dataset(name, args.prepare)


if __name__ == "__main__":
    main()
