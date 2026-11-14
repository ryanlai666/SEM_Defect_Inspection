"""Fine-tuned CNN classifier (ResNet-18 / ConvNeXt-T, ImageNet init) for the supervised and k-shot
fine-tuning baselines."""

from __future__ import annotations

import timm
import torch.nn as nn

from .backbones import CNNS


def build_classifier(
    name: str = "resnet18", n_classes: int = 6, pretrained: bool = True, drop_rate: float = 0.1
) -> nn.Module:
    return timm.create_model(
        CNNS.get(name, name), pretrained=pretrained, num_classes=n_classes, drop_rate=drop_rate
    )
