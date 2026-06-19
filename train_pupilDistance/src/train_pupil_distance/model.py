from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import torch
from torch import nn
from torchvision import models


def _ensure_pretrained_weights_are_cached(weights) -> None:
    filename = Path(urlparse(weights.url).path).name
    checkpoint_path = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "use_pretrained=True requires torchvision weights to already exist in the local "
            f"torch cache: {checkpoint_path}. This project does not download models automatically. "
            "Set use_pretrained: false or manually place the weights in the torch cache."
        )


def build_model(model_name: str = "mobilenet_v3_small", use_pretrained: bool = False) -> nn.Module:
    if model_name != "mobilenet_v3_small":
        raise ValueError("Only model_name='mobilenet_v3_small' is supported in this scaffold.")

    weights = models.MobileNet_V3_Small_Weights.DEFAULT if use_pretrained else None
    if weights is not None:
        _ensure_pretrained_weights_are_cached(weights)

    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 1)
    return model


def load_model_for_inference(
    model_path: str,
    model_name: str = "mobilenet_v3_small",
    device: torch.device | str = "cpu",
) -> nn.Module:
    model = build_model(model_name=model_name, use_pretrained=False)
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
