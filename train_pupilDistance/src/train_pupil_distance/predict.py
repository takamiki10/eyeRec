from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from train_pupil_distance.datasets import build_eval_transforms
from train_pupil_distance.model import load_model_for_inference
from train_pupil_distance.utils import get_device, load_yaml, resolve_path


def predict(image_path: str | Path, config_path: str | Path = "configs/train_config.yaml") -> dict:
    config_path = Path(config_path)
    config = load_yaml(config_path)
    base_dir = config_path.resolve().parents[1]
    device = get_device(str(config["device"]))

    model_path = resolve_path(config["export_dir"], base_dir) / "pupil_distance_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run training first.")

    model = load_model_for_inference(
        str(model_path),
        model_name=str(config["model_name"]),
        device=device,
    )

    image_path = resolve_path(image_path, base_dir)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    transform = build_eval_transforms(int(config["image_size"]))
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        predicted_value = float(model(image_tensor).squeeze().cpu().item())

    result = {
        "pupil_distance": predicted_value,
        "unit": "normalized_face_width",
    }

    print(f"pupil_distance: {result['pupil_distance']:.4f}")
    print(f"unit: {result['unit']}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict pupil distance for one image.")
    parser.add_argument("--image", required=True, help="Path to an image.")
    parser.add_argument("--config", default="configs/train_config.yaml", help="Path to train_config.yaml.")
    args = parser.parse_args()
    predict(args.image, args.config)


if __name__ == "__main__":
    main()
