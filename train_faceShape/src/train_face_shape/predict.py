from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from train_face_shape.datasets import build_eval_transforms
from train_face_shape.model import load_model_for_inference
from train_face_shape.utils import get_device, invert_mapping, load_label_mapping, load_yaml, resolve_path


def predict(image_path: str | Path, config_path: str | Path = "configs/train_config.yaml") -> dict:
    config_path = Path(config_path)
    config = load_yaml(config_path)
    base_dir = config_path.resolve().parents[1]
    label_to_index = load_label_mapping(base_dir / "configs" / "label_mapping.yaml")
    index_to_label = invert_mapping(label_to_index)
    device = get_device(str(config["device"]))

    model_path = resolve_path(config["export_dir"], base_dir) / "face_shape_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run training first.")

    model = load_model_for_inference(
        str(model_path),
        num_classes=len(label_to_index),
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
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()

    predicted_index = int(probabilities.argmax().item())
    predicted_label = index_to_label[predicted_index]
    class_probabilities = {
        index_to_label[index]: float(probabilities[index].item()) for index in range(len(index_to_label))
    }

    result = {
        "predicted_label": predicted_label,
        "confidence": class_probabilities[predicted_label],
        "class_probabilities": class_probabilities,
    }

    print(f"predicted_label: {result['predicted_label']}")
    print(f"confidence: {result['confidence']:.4f}")
    print("class_probabilities:")
    for label, probability in class_probabilities.items():
        print(f"  {label}: {probability:.4f}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict face shape for one image.")
    parser.add_argument("--image", required=True, help="Path to an image.")
    parser.add_argument("--config", default="configs/train_config.yaml", help="Path to train_config.yaml.")
    args = parser.parse_args()
    predict(args.image, args.config)


if __name__ == "__main__":
    main()
