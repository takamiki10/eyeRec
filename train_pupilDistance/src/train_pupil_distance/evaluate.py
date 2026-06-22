from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_pupil_distance.datasets import PupilDistanceDataset, build_eval_transforms
from train_pupil_distance.model import load_model_for_inference
from train_pupil_distance.utils import get_device, load_yaml, resolve_path, save_json


def evaluate(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    config = load_yaml(config_path)
    base_dir = config_path.resolve().parents[1]
    device = get_device(str(config["device"]))

    test_dataset = PupilDistanceDataset(
        resolve_path(config["test_dir"], base_dir),
        resolve_path(config["test_labels"], base_dir),
        transform=build_eval_transforms(int(config["image_size"])),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
    )

    model_path = resolve_path(config["export_dir"], base_dir) / "pupil_distance_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run training first.")

    model = load_model_for_inference(
        str(model_path),
        model_name=str(config["model_name"]),
        device=device,
    )

    absolute_error_sum = 0.0
    squared_error_sum = 0.0
    tolerance_hits = {
        "within_0_02": 0,
        "within_0_03": 0,
        "within_0_05": 0,
    }
    total = 0

    with torch.no_grad():
        for images, targets in tqdm(test_loader, leave=False):
            images = images.to(device)
            targets = targets.to(device)
            predictions = model(images).squeeze(1)
            errors = predictions - targets
            absolute_errors = torch.abs(errors)

            absolute_error_sum += absolute_errors.sum().item()
            squared_error_sum += torch.square(errors).sum().item()
            tolerance_hits["within_0_02"] += (absolute_errors <= 0.02).sum().item()
            tolerance_hits["within_0_03"] += (absolute_errors <= 0.03).sum().item()
            tolerance_hits["within_0_05"] += (absolute_errors <= 0.05).sum().item()
            total += targets.size(0)

    mae = absolute_error_sum / total if total else 0.0
    rmse = math.sqrt(squared_error_sum / total) if total else 0.0
    tolerance_accuracy = {
        name: hits / total if total else 0.0
        for name, hits in tolerance_hits.items()
    }
    results = {
        "mae": mae,
        "rmse": rmse,
        **tolerance_accuracy,
        "total": total,
    }

    print(f"mae: {mae:.4f}")
    print(f"rmse: {rmse:.4f}")
    for name, value in tolerance_accuracy.items():
        print(f"{name}: {value:.4f} ({tolerance_hits[name]}/{total})")

    metrics_dir = resolve_path(config["metrics_dir"], base_dir)
    save_json(results, metrics_dir / "evaluation_metrics.json")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the exported pupil distance regressor.")
    parser.add_argument("--config", default="configs/train_config.yaml", help="Path to train_config.yaml.")
    args = parser.parse_args()
    evaluate(args.config)


if __name__ == "__main__":
    main()
