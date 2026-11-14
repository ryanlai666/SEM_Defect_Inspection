"""EfficientAD-S (Batzner et al., arXiv:2303.14535), own implementation.

* Teacher: small patch-description network (PDN) distilled from WideResNet-101 on ImageNet.  We load
  the publicly released distilled teacher from the anomalib project (Apache-2.0; layer names match
  this module) - see DATA.md.  Without it the teacher is random (CI / unit tests only).
* Student: PDN with 2x384 outputs (teacher-mimic + autoencoder-mimic); autoencoder: 6-conv encoder,
  8-conv upsampling decoder.
* Losses (Alg. 1): hard-feature student-teacher loss (0.999-quantile mining) [+ optional ImageNet
  penalty, disabled by default since ImageNet is not downloaded - documented deviation],
  autoencoder loss, student-autoencoder loss.
* Inference: map_ST and map_AE normalised by quantiles (q=0.9, 0.995) of validation-normal maps,
  combined 0.5/0.5; image score = max of the combined map.

Inputs are ImageNet-normalised 3-channel images (the dataset already does this), so no internal
normalisation layer is needed; 256x256 input -> 56x56 feature maps (padding=False variant).
"""

from __future__ import annotations

import math
import zipfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import resolve_path

TEACHER_URL = (
    "https://github.com/open-edge-platform/anomalib/releases/download/"
    "efficientad_pretrained_weights/efficientad_pretrained_weights.zip"
)
TEACHER_DIR = "data/cache/pretrained/efficientad"


class PDNSmall(nn.Module):
    def __init__(self, out_channels: int = 384, padding: bool = False):
        super().__init__()
        p = 1 if padding else 0
        self.conv1 = nn.Conv2d(3, 128, 4, padding=3 * p)
        self.conv2 = nn.Conv2d(128, 256, 4, padding=3 * p)
        self.conv3 = nn.Conv2d(256, 256, 3, padding=1 * p)
        self.conv4 = nn.Conv2d(256, out_channels, 4, padding=0)
        self.avgpool1 = nn.AvgPool2d(2, 2, padding=1 * p)
        self.avgpool2 = nn.AvgPool2d(2, 2, padding=1 * p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool1(F.relu(self.conv1(x)))
        x = self.avgpool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        return self.conv4(x)


class AutoEncoder(nn.Module):
    def __init__(self, out_channels: int = 384, padding: bool = False):
        super().__init__()
        self.padding = padding
        enc = [(3, 32), (32, 32), (32, 64), (64, 64), (64, 64)]
        self.enc = nn.ModuleList([nn.Conv2d(a, b, 4, stride=2, padding=1) for a, b in enc])
        self.enc_last = nn.Conv2d(64, 64, 8)
        self.dec = nn.ModuleList([nn.Conv2d(64, 64, 4, padding=2) for _ in range(6)])
        self.dec7 = nn.Conv2d(64, 64, 3, padding=1)
        self.dec8 = nn.Conv2d(64, out_channels, 3, padding=1)
        self.drop = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        for c in self.enc:
            x = F.relu(c(x))
        x = self.enc_last(x)
        sizes = [
            (h // 64 - 1, w // 64 - 1),
            (h // 32, w // 32),
            (h // 16 - 1, w // 16 - 1),
            (h // 8, w // 8),
            (h // 4 - 1, w // 4 - 1),
            (h // 2 - 1, w // 2 - 1),
        ]
        for c, s in zip(self.dec, sizes):
            x = self.drop(F.relu(c(F.interpolate(x, size=s, mode="bilinear"))))
        last = (math.ceil(h / 4) - (0 if self.padding else 8), math.ceil(w / 4) - (0 if self.padding else 8))
        x = F.relu(self.dec7(F.interpolate(x, size=last, mode="bilinear")))
        return self.dec8(x)


def fetch_teacher(dest: str | Path = TEACHER_DIR) -> Path:
    """Download (40 MB) and unpack the distilled PDN teachers; returns the small-teacher path."""
    dest = resolve_path(dest)
    hits = list(dest.rglob("*teacher_small*.pth"))
    if hits:
        return hits[0]
    dest.mkdir(parents=True, exist_ok=True)
    zpath = dest / "efficientad_pretrained_weights.zip"
    if not zpath.exists():
        torch.hub.download_url_to_file(TEACHER_URL, str(zpath), progress=False)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(dest)
    hits = list(dest.rglob("*teacher_small*.pth"))
    if not hits:
        raise FileNotFoundError(f"teacher_small weights not found inside {zpath}")
    return hits[0]


class EfficientAD(nn.Module):
    def __init__(self, out_channels: int = 384, pretrained_teacher: bool = True, padding: bool = False):
        super().__init__()
        self.c = out_channels
        self.teacher = PDNSmall(out_channels, padding)
        self.student = PDNSmall(2 * out_channels, padding)
        self.ae = AutoEncoder(out_channels, padding)
        if pretrained_teacher:
            sd = torch.load(fetch_teacher(), map_location="cpu", weights_only=True)
            self.teacher.load_state_dict(sd)
        self.teacher.eval().requires_grad_(False)
        self.register_buffer("t_mean", torch.zeros(1, out_channels, 1, 1))
        self.register_buffer("t_std", torch.ones(1, out_channels, 1, 1))
        self.register_buffer("q", torch.tensor([0.0, 1.0, 0.0, 1.0]))  # qa_st, qb_st, qa_ae, qb_ae

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    @torch.no_grad()
    def teacher_norm(self, x: torch.Tensor) -> torch.Tensor:
        return (self.teacher(x) - self.t_mean) / self.t_std

    @torch.no_grad()
    def fit_teacher_stats(self, loader, device, max_batches: int | None = None) -> None:
        s1 = s2 = None
        n = 0
        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            y = self.teacher(batch["image"].to(device))
            cnt = y.shape[0] * y.shape[2] * y.shape[3]
            a = y.sum((0, 2, 3), keepdim=True)
            b = y.square().sum((0, 2, 3), keepdim=True)
            s1 = a if s1 is None else s1 + a
            s2 = b if s2 is None else s2 + b
            n += cnt
        mean = s1 / n
        self.t_mean.copy_(mean)
        self.t_std.copy_((s2 / n - mean.square()).clamp_min(1e-12).sqrt())

    def losses(self, x: torch.Tensor, x_aug: torch.Tensor, x_penalty: torch.Tensor | None = None) -> dict:
        t = self.teacher_norm(x)
        s = self.student(x)
        d = (t - s[:, : self.c]).square()
        flat = d.flatten()
        if flat.numel() > 2**24:  # torch.quantile limit
            flat = flat[torch.randperm(flat.numel(), device=flat.device)[: 2**24]]
        d_hard = torch.quantile(flat, 0.999)
        loss_hard = d[d >= d_hard].mean()
        loss_pen = (
            self.student(x_penalty)[:, : self.c].square().mean()
            if x_penalty is not None
            else torch.zeros((), device=x.device)
        )
        ae = self.ae(x_aug)
        t_aug = self.teacher_norm(x_aug)
        s_aug = self.student(x_aug)[:, self.c :]
        loss_ae = (t_aug - ae).square().mean()
        loss_stae = (ae - s_aug).square().mean()
        total = loss_hard + loss_pen + loss_ae + loss_stae
        return {
            "loss": total,
            "loss_st": loss_hard,
            "loss_penalty": loss_pen,
            "loss_ae": loss_ae,
            "loss_stae": loss_stae,
        }

    @torch.no_grad()
    def raw_maps(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.teacher_norm(x)
        s = self.student(x)
        ae = self.ae(x)
        map_st = (t - s[:, : self.c]).square().mean(1, keepdim=True)
        map_ae = (ae - s[:, self.c :]).square().mean(1, keepdim=True)
        return map_st, map_ae

    @torch.no_grad()
    def fit_quantiles(self, loader, device, max_batches: int | None = None) -> None:
        sts, aes = [], []
        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            a, b = self.raw_maps(batch["image"].to(device))
            sts.append(a.flatten())
            aes.append(b.flatten())
        st, ae = torch.cat(sts), torch.cat(aes)
        q = lambda v, p: torch.quantile(v[torch.randperm(v.numel(), device=v.device)[: 2**24]], p)  # noqa: E731
        self.q.copy_(torch.stack([q(st, 0.9), q(st, 0.995), q(ae, 0.9), q(ae, 0.995)]))

    @torch.no_grad()
    def predict(self, x: torch.Tensor, out_size: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        map_st, map_ae = self.raw_maps(x)
        qa_st, qb_st, qa_ae, qb_ae = self.q
        map_st = 0.1 * (map_st - qa_st) / (qb_st - qa_st + 1e-12)
        map_ae = 0.1 * (map_ae - qa_ae) / (qb_ae - qa_ae + 1e-12)
        m = 0.5 * map_st + 0.5 * map_ae
        m = F.pad(m, (4, 4, 4, 4))  # padding=False variant: re-align the 56x56 map to the image grid
        m = F.interpolate(m, size=(out_size or x.shape[-1],) * 2, mode="bilinear", align_corners=False)[:, 0]
        return m.flatten(1).amax(1), m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ONNX/TensorRT export path: normalised anomaly map (B,1,56+8,56+8) before resizing."""
        map_st, map_ae = self.raw_maps(x)
        qa_st, qb_st, qa_ae, qb_ae = self.q
        m = 0.05 * (map_st - qa_st) / (qb_st - qa_st + 1e-12) + 0.05 * (map_ae - qa_ae) / (
            qb_ae - qa_ae + 1e-12
        )
        return F.pad(m, (4, 4, 4, 4))


def ae_augment(x01_3ch: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """EfficientAD AE augmentation on *unnormalised* [0,1] images: brightness / contrast / saturation
    with coefficient U(0.8,1.2).  For grey SEM images saturation is a no-op, so it falls back to
    brightness."""
    coef = float(torch.empty(1).uniform_(0.8, 1.2, generator=generator))
    which = int(torch.randint(0, 3, (1,), generator=generator))
    if which == 1:
        mean = x01_3ch.mean(dim=(-3, -2, -1), keepdim=True)
        return ((x01_3ch - mean) * coef + mean).clamp(0, 1)
    return (x01_3ch * coef).clamp(0, 1)
