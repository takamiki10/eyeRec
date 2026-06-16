from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_eye_color.datasets import EyeColorFolderDataset, build_eval_transforms
from train_eye_color.model import load_model_for_inference
from train_eye_color.utils import get_device, invert_mapping, load_label_mapping, load_yaml, resolve_path, save_json


def evaluate(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    config = load_yaml(config_path)
    base_dir = config_path.resolve().parents[1]
    label_to_index = load_label_mapping(base_dir / "configs" / "label_mapping.yaml")
    index_to_label = invert_mapping(label_to_index)
    device = get_device(str(config["device"]))

    test_dataset = EyeColorFolderDataset(
        resolve_path(config["test_dir"], base_dir),
        label_to_index,
        transform=build_eval_transforms(int(config["image_size"])),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
    )

    model_path = resolve_path(config["export_dir"], base_dir) / "eye_color_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run training first.")

    model = load_model_for_inference(
        str(model_path),
        num_classes=len(label_to_index),
        model_name=str(config["model_name"]),
        device=device,
    )

    class_correct = {label: 0 for label in label_to_index}
    class_total = {label: 0 for label in label_to_index}
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            for prediction, label_index in zip(predictions.cpu().tolist(), labels.cpu().tolist()):
                label_name = index_to_label[label_index]
                class_total[label_name] += 1
                if prediction == label_index:
                    class_correct[label_name] += 1

    accuracy = correct / total if total else 0.0
    per_class_accuracy = {
        label: (class_correct[label] / class_total[label] if class_total[label] else 0.0)
        for label in label_to_index
    }

    results = {
        "accuracy": accuracy,
        "per_class_accuracy": per_class_accuracy,
        "class_total": class_total,
    }

    print(f"accuracy: {accuracy:.4f}")
    for label, value in per_class_accuracy.items():
        print(f"{label}: {value:.4f} ({class_correct[label]}/{class_total[label]})")

    metrics_dir = resolve_path(config["metrics_dir"], base_dir)
    save_json(results, metrics_dir / "evaluation_metrics.json")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the exported eye color classifier.")
    parser.add_argument("--config", default="configs/train_config.yaml", help="Path to train_config.yaml.")
    args = parser.parse_args()
    evaluate(args.config)


if __name__ == "__main__":
    main()
