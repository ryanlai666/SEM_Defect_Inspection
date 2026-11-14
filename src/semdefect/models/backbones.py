"""Pretrained backbones via timm (weights from the Hugging Face hub, cached under ~/.cache).

DINOv2 (Oquab et al., arXiv:2304.07193) ViT-S/14 and ViT-B/14 are used *frozen*; their positional
embeddings (trained at 518 px) are resampled by timm to the requested input size.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

DINOV2 = {
    "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
    "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
    "dinov2_vits14_reg": "vit_small_patch14_reg4_dinov2.lvd142m",
    "dinov2_vitb14_reg": "vit_base_patch14_reg4_dinov2.lvd142m",
}
CNNS = {
    "resnet18": "resnet18.tv_in1k",
    "wide_resnet50_2": "wide_resnet50_2.tv_in1k",
    "convnext_tiny": "convnext_tiny.fb_in1k",
}


def count_params(m: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad or not trainable_only)


class DinoV2(nn.Module):
    """Frozen DINOv2 wrapper exposing CLS / patch tokens and the last-n intermediate layers."""

    def __init__(self, name: str = "dinov2_vits14", img_size: int = 224, pretrained: bool = True):
        super().__init__()
        timm_name = DINOV2.get(name, name)
        self.model = timm.create_model(timm_name, pretrained=pretrained, img_size=img_size, num_classes=0)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.embed_dim = self.model.embed_dim
        self.patch = self.model.patch_embed.patch_size[0]
        self.img_size = img_size
        self.grid = img_size // self.patch
        self.n_prefix = getattr(self.model, "num_prefix_tokens", 1)

    def train(self, mode: bool = True):  # always frozen / eval
        super().train(False)
        return self

    @torch.no_grad()
    def tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (cls (B,D), patch tokens (B,N,D)) after the final norm."""
        feats = self.model.forward_features(x)
        return feats[:, 0], feats[:, self.n_prefix :]

    @torch.no_grad()
    def global_features(self, x: torch.Tensor) -> torch.Tensor:
        """DINOv2 linear-eval descriptor: concat(CLS, mean(patch tokens)) -> (B, 2D)."""
        cls, patch = self.tokens(x)
        return torch.cat([cls, patch.mean(1)], dim=1)

    @torch.no_grad()
    def intermediate(self, x: torch.Tensor, n_last: int = 4, norm: bool = True) -> list[torch.Tensor]:
        """Last `n_last` block outputs as (B,D,h,w) maps."""
        return self.model.forward_intermediates(
            x, indices=n_last, norm=norm, output_fmt="NCHW", intermediates_only=True
        )


class CNNFeatures(nn.Module):
    """Frozen CNN multi-scale feature extractor (e.g. WideResNet-50 layer2/3 for PatchCore)."""

    def __init__(self, name: str = "wide_resnet50_2", out_indices=(2, 3), pretrained: bool = True):
        super().__init__()
        self.model = timm.create_model(
            CNNS.get(name, name), pretrained=pretrained, features_only=True, out_indices=out_indices
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.channels = self.model.feature_info.channels()
        self.reductions = self.model.feature_info.reduction()

    def train(self, mode: bool = True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.model(x)


def upsample_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
