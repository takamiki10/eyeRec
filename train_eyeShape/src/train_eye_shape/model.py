from __future__ import annotations

import torch
from torch import nn
from torchvision import models


def build_model(
    num_classes: int,
    model_name: str = "mobilenet_v3_small",
    use_pretrained: bool = False,
) -> nn.Module:
    if model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if use_pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
    elif model_name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if use_pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if use_pretrained else None
        model = models.efficientnet_b0(weights=weights)
    else:
        raise ValueError(
            "Supported model_name values: mobilenet_v3_small, mobilenet_v3_large, efficientnet_b0."
        )

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def load_model_for_inference(
    model_path: str,
    num_classes: int,
    model_name: str = "mobilenet_v3_small",
    device: torch.device | str = "cpu",
) -> nn.Module:
    model = build_model(num_classes=num_classes, model_name=model_name, use_pretrained=False)
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
