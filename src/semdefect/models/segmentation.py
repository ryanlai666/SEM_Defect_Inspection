"""Segmentation models.

* `UNet`          - classic U-Net (Ronneberger et al., arXiv:1505.04597) trained from scratch
                    (`encoder=None`) or with an ImageNet ResNet-18 encoder (`encoder="resnet18"`).
* `DinoSegHead`   - frozen DINOv2 encoder + light decoder: last-4-layer patch tokens are fused by a
                    1x1 conv, progressively upsampled, and refined with a shallow full-resolution image
                    stem so thin defect boundaries are not limited by the 14 px patch grid.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import CNNS, DinoV2


def conv_bn_relu(cin: int, cout: int, k: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, padding=k // 2, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True)
    )


class DoubleConv(nn.Sequential):
    def __init__(self, cin: int, cout: int):
        super().__init__(conv_bn_relu(cin, cout), conv_bn_relu(cout, cout))


class Up(nn.Module):
    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        self.conv = DoubleConv(cin + cskip, cout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        size = skip.shape[-2:] if skip is not None else tuple(s * 2 for s in x.shape[-2:])
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], 1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, base: int = 32, encoder: str | None = None, pretrained: bool = True, in_ch: int = 3):
        super().__init__()
        self.encoder_name = encoder
        if encoder is None:
            ch = [base, base * 2, base * 4, base * 8, base * 16]
            self.enc = nn.ModuleList(
                [DoubleConv(in_ch, ch[0])] + [DoubleConv(ch[i], ch[i + 1]) for i in range(4)]
            )
            self.pool = nn.MaxPool2d(2)
            skips = ch[:4]
            bottom = ch[4]
        else:
            self.backbone = timm.create_model(
                CNNS.get(encoder, encoder),
                pretrained=pretrained,
                features_only=True,
                out_indices=(0, 1, 2, 3, 4),
            )
            ch = self.backbone.feature_info.channels()  # resnet18: 64,64,128,256,512 @ /2../32
            skips = ch[:4]
            bottom = ch[4]
        dec = [base * 8, base * 4, base * 2, base]
        self.up = nn.ModuleList(
            [
                Up(bottom, skips[3], dec[0]),
                Up(dec[0], skips[2], dec[1]),
                Up(dec[1], skips[1], dec[2]),
                Up(dec[2], skips[0], dec[3]),
            ]
        )
        self.head = nn.Conv2d(dec[3], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        if self.encoder_name is None:
            feats = []
            h = x
            for i, block in enumerate(self.enc):
                h = block(h if i == 0 else self.pool(h))
                feats.append(h)
        else:
            feats = self.backbone(x)
        h = feats[4]
        for up, skip in zip(self.up, reversed(feats[:4])):
            h = up(h, skip)
        return F.interpolate(self.head(h), size=size, mode="bilinear", align_corners=False)


class DinoSegHead(nn.Module):
    def __init__(
        self,
        backbone: str = "dinov2_vits14",
        img_size: int = 448,
        n_layers: int = 4,
        dec_ch: int = 128,
        pretrained: bool = True,
        stem: bool = True,
    ):
        super().__init__()
        self.encoder = DinoV2(backbone, img_size=img_size, pretrained=pretrained)
        self.n_layers = n_layers
        d = self.encoder.embed_dim
        self.fuse = nn.Sequential(
            nn.BatchNorm2d(d * n_layers), nn.Conv2d(d * n_layers, dec_ch * 2, 1), nn.ReLU(True)
        )
        self.up1 = conv_bn_relu(dec_ch * 2, dec_ch)
        self.up2 = conv_bn_relu(dec_ch, dec_ch // 2)
        self.use_stem = stem
        if stem:  # cheap full-res path: 2 convs at 1/2 res
            self.stem = nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(True),
                conv_bn_relu(16, 16),
            )
        self.refine = conv_bn_relu(dec_ch // 2 + (16 if stem else 0), 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        with torch.no_grad():
            feats = self.encoder.intermediate(x, n_last=self.n_layers)
        h = self.fuse(torch.cat(feats, 1))  # (B, 2C, g, g)
        h = self.up1(F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False))
        h = self.up2(F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False))
        half = (size[0] // 2, size[1] // 2)
        h = F.interpolate(h, size=half, mode="bilinear", align_corners=False)
        if self.use_stem:
            h = torch.cat([h, self.stem(x)], 1)
        h = self.head(self.refine(h))
        return F.interpolate(h, size=size, mode="bilinear", align_corners=False)


def dice_bce_loss(
    logits: torch.Tensor, target: torch.Tensor, bce_w: float = 1.0, dice_w: float = 1.0
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).flatten(1).sum(1)
    denom = p.flatten(1).sum(1) + target.flatten(1).sum(1)
    dice = 1 - (2 * inter + 1.0) / (denom + 1.0)
    return bce_w * bce + dice_w * dice.mean()


def build_segmenter(cfg: dict, pretrained: bool = True) -> nn.Module:
    m = cfg["model"]
    if m["name"] == "unet":
        return UNet(base=m.get("base", 32), encoder=m.get("encoder"), pretrained=pretrained)
    if m["name"] == "dino_seg":
        return DinoSegHead(
            backbone=m.get("backbone", "dinov2_vits14"),
            img_size=cfg["data"]["img_size"],
            n_layers=m.get("n_layers", 4),
            dec_ch=m.get("dec_ch", 128),
            pretrained=pretrained,
            stem=m.get("stem", True),
        )
    raise ValueError(m["name"])
