from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_eye_color.datasets import EyeColorFolderDataset, build_eval_transforms, build_train_transforms
from train_eye_color.model import build_model
from train_eye_color.utils import (
    get_device,
    invert_mapping,
    load_label_mapping,
    load_yaml,
    resolve_path,
    save_json,
    set_seed,
)


def run_epoch(model, dataloader, criterion, device, optimizer=None) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total


def save_best_export(checkpoint: dict, export_dir: Path, epoch: int, val_accuracy: float) -> None:
    versioned_path = export_dir / f"eye_color_model_epoch_{epoch:03d}_val_{val_accuracy:.4f}.pt"
    torch.save(checkpoint, versioned_path)

    final_path = export_dir / "eye_color_model.pt"
    temp_path = export_dir / f".{final_path.name}.tmp_{epoch:03d}_{int(time.time())}"
    try:
        torch.save(checkpoint, temp_path)
        temp_path.replace(final_path)
    except (OSError, RuntimeError) as error:
        if temp_path.exists():
            temp_path.unlink()
        print(
            "warning: saved best model version but could not update "
            f"{final_path}: {error}. Close any process syncing or reading that file, "
            f"then copy {versioned_path.name} to {final_path.name}."
        )


def train(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    config = load_yaml(config_path)
    base_dir = config_path.resolve().parents[1]
    label_to_index = load_label_mapping(base_dir / "configs" / "label_mapping.yaml")
    index_to_label = invert_mapping(label_to_index)

    set_seed(int(config["seed"]))
    device = get_device(str(config["device"]))

    train_dataset = EyeColorFolderDataset(
        resolve_path(config["train_dir"], base_dir),
        label_to_index,
        transform=build_train_transforms(int(config["image_size"])),
    )
    val_dataset = EyeColorFolderDataset(
        resolve_path(config["val_dir"], base_dir),
        label_to_index,
        transform=build_eval_transforms(int(config["image_size"])),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
    )

    model = build_model(
        num_classes=len(label_to_index),
        model_name=str(config["model_name"]),
        use_pretrained=bool(config["use_pretrained"]),
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))

    checkpoint_dir = resolve_path(config["checkpoint_dir"], base_dir)
    export_dir = resolve_path(config["export_dir"], base_dir)
    metrics_dir = resolve_path(config["metrics_dir"], base_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    best_val_accuracy = 0.0
    history = []
    started_at = time.time()

    for epoch in range(1, int(config["epochs"]) + 1):
        print(f"Epoch {epoch}/{config['epochs']}")
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
        history.append(epoch_metrics)
        print(
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "label_to_index": label_to_index,
            "index_to_label": index_to_label,
            "config": config,
            "epoch": epoch,
            "val_accuracy": val_accuracy,
        }
        torch.save(checkpoint, checkpoint_dir / f"epoch_{epoch:03d}.pt")

        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_best_export(checkpoint, export_dir, epoch, val_accuracy)

    training_metrics = {
        "best_val_accuracy": best_val_accuracy,
        "duration_seconds": round(time.time() - started_at, 2),
        "history": history,
    }
    save_json(training_metrics, metrics_dir / "training_metrics.json")
    save_json({"labels": label_to_index}, export_dir / "label_mapping.json")
    return training_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the local EyeRec eye color classifier.")
    parser.add_argument("--config", default="configs/train_config.yaml", help="Path to train_config.yaml.")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
