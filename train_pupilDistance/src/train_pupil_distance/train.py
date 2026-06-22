from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_pupil_distance.datasets import PupilDistanceDataset, build_eval_transforms, build_train_transforms
from train_pupil_distance.model import build_model
from train_pupil_distance.utils import (
    get_device,
    load_yaml,
    resolve_path,
    save_json,
    set_seed,
)


def target_mean(samples: list[tuple[Path, float]]) -> float:
    return sum(value for _, value in samples) / max(len(samples), 1)


def initialize_regression_head(model: nn.Module, mean_target: float) -> None:
    head = model.classifier[-1]
    if not isinstance(head, nn.Sequential) or not isinstance(head[0], nn.Linear):
        return
    scaled_target = min(max((mean_target - 0.15) / 0.20, 1e-4), 1.0 - 1e-4)
    bias = torch.logit(torch.tensor(scaled_target)).item()
    nn.init.zeros_(head[0].weight)
    nn.init.constant_(head[0].bias, bias)


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
    total_absolute_error = 0.0
    total = 0

    for images, targets in tqdm(dataloader, leave=False):
        images = images.to(device)
        targets = targets.to(device)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, targets)
            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_absolute_error += torch.abs(outputs.detach() - targets).sum().item()
        total += batch_size

    return total_loss / total, total_absolute_error / total


def train(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    config = load_yaml(config_path)
    base_dir = config_path.resolve().parents[1]

    set_seed(int(config["seed"]))
    device = get_device(str(config["device"]))
    print(f"Using device: {device}")

    train_dataset = PupilDistanceDataset(
        resolve_path(config["train_dir"], base_dir),
        resolve_path(config["train_labels"], base_dir),
        transform=build_train_transforms(int(config["image_size"])),
    )
    val_dataset = PupilDistanceDataset(
        resolve_path(config["val_dir"], base_dir),
        resolve_path(config["val_labels"], base_dir),
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

    model = build_model(model_name=str(config["model_name"]), use_pretrained=bool(config["use_pretrained"])).to(device)
    initialize_regression_head(model, target_mean(train_dataset.samples))
    frozen_epochs = int(config.get("frozen_epochs", 0))
    unfreeze_last_n_blocks = int(config.get("unfreeze_last_n_blocks", 0))
    if frozen_epochs > 0:
        set_feature_training(model, trainable=False)

    criterion = nn.SmoothL1Loss(beta=float(config.get("smooth_l1_beta", 0.02)))
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

    best_val_mae = float("inf")
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
        train_loss, train_mae = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_mae = run_epoch(model, val_loader, criterion, device)

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mae": train_mae,
            "val_loss": val_loss,
            "val_mae": val_mae,
        }
        history.append(epoch_metrics)
        print(
            f"train_loss={train_loss:.4f} train_mae={train_mae:.4f} "
            f"val_loss={val_loss:.4f} val_mae={val_mae:.4f}"
        )
        scheduler.step()

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "val_mae": val_mae,
        }
        if save_every_n_epochs > 0 and epoch % save_every_n_epochs == 0:
            torch.save(checkpoint, checkpoint_dir / f"epoch_{epoch:03d}.pt")

        if val_mae <= best_val_mae:
            best_val_mae = val_mae
            epochs_without_improvement = 0
            torch.save(checkpoint, export_dir / "pupil_distance_model.pt")
        else:
            epochs_without_improvement += 1

        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping after {epoch} epochs. "
                f"Best val_mae={best_val_mae:.4f}."
            )
            break

    training_metrics = {
        "best_val_mae": best_val_mae,
        "duration_seconds": round(time.time() - started_at, 2),
        "history": history,
    }
    save_json(training_metrics, metrics_dir / "training_metrics.json")
    return training_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the local EyeRec pupil distance regressor.")
    parser.add_argument("--config", default="configs/train_config.yaml", help="Path to train_config.yaml.")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
