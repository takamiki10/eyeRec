from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from train_eye_shape.datasets import EyeShapeFolderDataset, build_eval_transforms, build_train_transforms
from train_eye_shape.model import build_model
from train_eye_shape.utils import (
    get_device,
    invert_mapping,
    load_label_mapping,
    load_yaml,
    resolve_path,
    save_json,
    set_seed,
)


def class_counts_from_samples(samples: list[tuple[Path, int]], num_classes: int) -> list[int]:
    counts = [0 for _ in range(num_classes)]
    for _, label_index in samples:
        counts[label_index] += 1
    return counts


def build_balanced_sampler(samples: list[tuple[Path, int]], num_classes: int) -> WeightedRandomSampler:
    counts = class_counts_from_samples(samples, num_classes)
    sample_weights = [1.0 / max(counts[label_index], 1) for _, label_index in samples]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def build_class_weight_tensor(samples: list[tuple[Path, int]], num_classes: int, device: torch.device) -> torch.Tensor:
    counts = class_counts_from_samples(samples, num_classes)
    total = sum(counts)
    weights = [total / (num_classes * max(count, 1)) for count in counts]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def set_feature_training(model: nn.Module, trainable: bool) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = trainable
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def unfreeze_last_feature_blocks(model: nn.Module, block_count: int) -> None:
    set_feature_training(model, trainable=False)
    if block_count <= 0:
        return
    for block in model.features[-block_count:]:
        for parameter in block.parameters():
            parameter.requires_grad = True


def build_optimizer(model: nn.Module, learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return torch.optim.AdamW(trainable_parameters, lr=learning_rate, weight_decay=weight_decay)


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


def train(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    config = load_yaml(config_path)
    base_dir = config_path.resolve().parents[1]
    label_to_index = load_label_mapping(base_dir / "configs" / "label_mapping.yaml")
    index_to_label = invert_mapping(label_to_index)

    set_seed(int(config["seed"]))
    device = get_device(str(config["device"]))
    print(f"Using device: {device}")

    train_dataset = EyeShapeFolderDataset(
        resolve_path(config["train_dir"], base_dir),
        label_to_index,
        transform=build_train_transforms(int(config["image_size"])),
    )
    val_dataset = EyeShapeFolderDataset(
        resolve_path(config["val_dir"], base_dir),
        label_to_index,
        transform=build_eval_transforms(int(config["image_size"])),
    )

    balanced_sampler = build_balanced_sampler(train_dataset.samples, len(label_to_index))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        sampler=balanced_sampler,
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
    frozen_epochs = int(config.get("frozen_epochs", 0))
    unfreeze_last_n_blocks = int(config.get("unfreeze_last_n_blocks", 0))
    if frozen_epochs > 0:
        set_feature_training(model, trainable=False)

    class_weights = build_class_weight_tensor(train_dataset.samples, len(label_to_index), device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=float(config.get("label_smoothing", 0.0)),
    )
    optimizer = build_optimizer(
        model,
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(config["epochs"]), 1))

    checkpoint_dir = resolve_path(config["checkpoint_dir"], base_dir)
    export_dir = resolve_path(config["export_dir"], base_dir)
    metrics_dir = resolve_path(config["metrics_dir"], base_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    best_val_accuracy = 0.0
    epochs_without_improvement = 0
    early_stopping_patience = int(config.get("early_stopping_patience", 0))
    save_every_n_epochs = int(config.get("save_every_n_epochs", 0))
    history = []
    started_at = time.time()

    for epoch in range(1, int(config["epochs"]) + 1):
        if epoch == frozen_epochs + 1 and frozen_epochs > 0:
            unfreeze_last_feature_blocks(model, unfreeze_last_n_blocks)
            optimizer = build_optimizer(
                model,
                learning_rate=float(config.get("fine_tune_learning_rate", config["learning_rate"])),
                weight_decay=float(config.get("weight_decay", 0.0)),
            )
            remaining_epochs = max(int(config["epochs"]) - epoch + 1, 1)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_epochs)
            print(f"Unfroze last {unfreeze_last_n_blocks} feature blocks for fine-tuning.")

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
        scheduler.step()

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "label_to_index": label_to_index,
            "index_to_label": index_to_label,
            "config": config,
            "epoch": epoch,
            "val_accuracy": val_accuracy,
        }
        if save_every_n_epochs > 0 and epoch % save_every_n_epochs == 0:
            torch.save(checkpoint, checkpoint_dir / f"epoch_{epoch:03d}.pt")

        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
            torch.save(checkpoint, export_dir / "eye_shape_model.pt")
        else:
            epochs_without_improvement += 1

        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping after {epoch} epochs. "
                f"Best val_acc={best_val_accuracy:.4f}."
            )
            break

    training_metrics = {
        "best_val_accuracy": best_val_accuracy,
        "duration_seconds": round(time.time() - started_at, 2),
        "history": history,
    }
    save_json(training_metrics, metrics_dir / "training_metrics.json")
    save_json({"labels": label_to_index}, export_dir / "label_mapping.json")
    return training_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the local EyeRec eye shape classifier.")
    parser.add_argument("--config", default="configs/train_config.yaml", help="Path to train_config.yaml.")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
