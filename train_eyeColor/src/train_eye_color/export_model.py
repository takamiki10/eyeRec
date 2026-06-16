from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def export_to_app(
    source_dir: str | Path = "artifacts/exported",
    destination_dir: str | Path = "../eyewear_recommendation_system/models/feature_extractors/eye_color",
) -> tuple[Path, Path]:
    project_dir = Path(__file__).resolve().parents[2]
    source_dir = Path(source_dir)
    destination_dir = Path(destination_dir)

    if not source_dir.is_absolute():
        source_dir = project_dir / source_dir
    if not destination_dir.is_absolute():
        destination_dir = project_dir / destination_dir

    model_source = source_dir / "eye_color_model.pt"
    mapping_source = source_dir / "label_mapping.json"
    if not model_source.exists():
        raise FileNotFoundError(f"Missing model artifact: {model_source}. Run training first.")
    if not mapping_source.exists():
        raise FileNotFoundError(f"Missing label mapping artifact: {mapping_source}. Run training first.")

    destination_dir.mkdir(parents=True, exist_ok=True)
    model_destination = destination_dir / "eye_color_model.pt"
    mapping_destination = destination_dir / "label_mapping.json"
    shutil.copy2(model_source, model_destination)
    shutil.copy2(mapping_source, mapping_destination)

    print(f"Copied model to: {model_destination}")
    print(f"Copied label mapping to: {mapping_destination}")
    return model_destination, mapping_destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Export eye color model artifacts into the main EyeRec app.")
    parser.add_argument("--source_dir", default="artifacts/exported", help="Folder containing exported artifacts.")
    parser.add_argument(
        "--destination_dir",
        default="../eyewear_recommendation_system/models/feature_extractors/eye_color",
        help="Destination folder inside the main application.",
    )
    args = parser.parse_args()
    export_to_app(args.source_dir, args.destination_dir)


if __name__ == "__main__":
    main()
