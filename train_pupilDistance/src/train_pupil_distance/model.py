from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class ScaledSigmoid(nn.Module):
    def __init__(self, output_min: float = 0.15, output_max: float = 0.35) -> None:
        super().__init__()
        self.output_min = output_min
        self.output_scale = output_max - output_min

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.output_min + self.output_scale * values


def build_model(model_name: str = "mobilenet_v3_small", use_pretrained: bool = False) -> nn.Module:
    if model_name != "mobilenet_v3_small":
        raise ValueError("Only model_name='mobilenet_v3_small' is supported in this scaffold.")

    weights = models.MobileNet_V3_Small_Weights.DEFAULT if use_pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Sequential(
        nn.Linear(in_features, 1),
        nn.Sigmoid(),
        ScaledSigmoid(),
    )
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
