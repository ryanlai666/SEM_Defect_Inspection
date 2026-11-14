"""Memory-bank anomaly detectors: PatchCore and AnomalyDINO-style patch kNN.

PatchCore (Roth et al., arXiv:2106.08265): ImageNet WideResNet-50 layer2+layer3 features, 3x3 local
neighbourhood averaging, channel pooling to `embed_dim`, greedy k-center coreset of the normal patch
memory bank; patch score = distance to the nearest memory patch, image score = max patch score.

AnomalyDINO (Damm et al., arXiv:2405.14529): frozen DINOv2 ViT-S/14 patch tokens of a few normal
reference images (optionally rotation-augmented); patch score = cosine distance to the nearest
reference patch; image score = mean of the top-1% patch scores.  No background masking is used
(SEM on unpatterned wafers is texture-like).

Both detectors share the kNN backends in `semdefect.kernels.knn` so the custom CUDA kernel can be
swapped in for `torch.cdist + topk` with one config switch.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kernels.knn import knn
from ..robustness import gaussian_kernel1d
from .backbones import CNNFeatures, DinoV2


def smooth_maps(maps: torch.Tensor, size: int, sigma: float = 4.0) -> torch.Tensor:
    """(B,h,w) patch scores -> (B,size,size) bilinear upsample + Gaussian blur."""
    m = F.interpolate(maps.unsqueeze(1), size=(size, size), mode="bilinear", align_corners=False)
    if sigma > 0:
        k = gaussian_kernel1d(sigma, m.device, m.dtype)
        r = (k.numel() - 1) // 2
        m = F.conv2d(F.pad(m, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
        m = F.conv2d(F.pad(m, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
    return m.squeeze(1)


def offmask_patches(emb: torch.Tensor, mask: torch.Tensor, margin_px: int) -> torch.Tensor:
    """Select patch embeddings far from any defect pixel (normal-pool augmentation).

    emb: (B,N,D) patch embeddings on a square grid; mask: (B,1,S,S) binary defect masks.
    A patch is kept when no (dilated by `margin_px`) mask pixel falls in its cell."""
    return emb[offmask_keep(mask, int(math.isqrt(emb.shape[1])), margin_px)]


def offmask_keep(mask: torch.Tensor, side: int, margin_px: int) -> torch.Tensor:
    """(B,1,S,S) binary masks -> (B, side*side) bool: True where no mask pixel dilated by `margin_px`
    falls in the patch cell."""
    m = mask.float()
    if margin_px > 0:
        m = F.max_pool2d(m, 2 * margin_px + 1, stride=1, padding=margin_px)
    return ~(F.adaptive_max_pool2d(m, (side, side)).reshape(m.shape[0], side * side) > 0)


def greedy_coreset(feats: torch.Tensor, n_select: int, proj_dim: int = 128, seed: int = 0) -> torch.Tensor:
    """Greedy k-center selection (approximate, on a Gaussian random projection).  Returns indices."""
    n = feats.shape[0]
    if n_select >= n:
        return torch.arange(n, device=feats.device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    proj = torch.randn(feats.shape[1], proj_dim, generator=g).to(feats.device, torch.float32)
    # project in chunks: a full FP32 copy of a 400k x 1024 bank alone would be 1.7 GB
    z = torch.cat([feats[i : i + 65536].float() @ proj for i in range(0, n, 65536)]) / math.sqrt(proj_dim)
    sel = torch.empty(n_select, dtype=torch.long, device=feats.device)
    cur = int(torch.randint(n, (1,), generator=g))
    min_d = torch.full((n,), float("inf"), device=feats.device)
    for i in range(n_select):
        sel[i] = cur
        d = (z - z[cur]).square().sum(1)
        min_d = torch.minimum(min_d, d)
        cur = int(torch.argmax(min_d))
    return sel


class PatchCoreEmbedder(nn.Module):
    """Backbone + aggregation, exported to ONNX/TensorRT as one graph: (B,3,H,W) -> (B, h*w, D)."""

    def __init__(
        self, backbone: str = "wide_resnet50_2", layers=(2, 3), embed_dim: int = 1024, pretrained: bool = True
    ):
        super().__init__()
        self.features = CNNFeatures(backbone, out_indices=tuple(layers), pretrained=pretrained)
        self.embed_dim = embed_dim
        self.n_layers = len(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        ref = feats[0]
        h, w = ref.shape[-2:]
        per = self.embed_dim // self.n_layers
        outs = []
        for f in feats:
            f = F.avg_pool2d(f, 3, stride=1, padding=1, count_include_pad=False)
            if f.shape[-2:] != (h, w):
                f = F.interpolate(f, size=(h, w), mode="bilinear", align_corners=False)
            b, c = f.shape[:2]
            f = f.permute(0, 2, 3, 1)
            if c % per == 0:  # identical to adaptive_avg_pool1d for divisible sizes; ONNX/TensorRT friendly
                f = f.reshape(b, h * w, per, c // per).mean(-1)
            else:
                f = F.adaptive_avg_pool1d(f.reshape(b * h * w, 1, c), per).reshape(b, h * w, per)
            outs.append(f)
        return torch.cat(outs, dim=2)


class PatchCore(nn.Module):
    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers=(2, 3),
        embed_dim: int = 1024,
        coreset_ratio: float = 0.1,
        proj_dim: int = 128,
        blur_sigma: float = 4.0,
        knn_backend: str = "torch",
        pretrained: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        self.embedder = PatchCoreEmbedder(backbone, layers, embed_dim, pretrained)
        self.coreset_ratio = coreset_ratio
        self.proj_dim = proj_dim
        self.blur_sigma = blur_sigma
        self.knn_backend = knn_backend
        self.seed = seed
        self.register_buffer("memory", torch.empty(0, embed_dim), persistent=True)
        self.grid: tuple[int, int] | None = None

    @torch.no_grad()
    def fit(
        self,
        loader,
        device: torch.device,
        max_images: int | None = None,
        offmask_loader=None,
        margin_px: int = 16,
    ) -> dict:
        self.to(device).eval()
        bank, seen = [], 0
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            bank.append(self.embedder(x).reshape(-1, self.embedder.embed_dim).half())
            seen += x.shape[0]
            if max_images and seen >= max_images:
                break
        n_off = 0
        for batch in offmask_loader or []:
            emb = self.embedder(batch["image"].to(device, non_blocking=True))
            sel = offmask_patches(emb, batch["mask"].to(device), margin_px).half()
            n_off += sel.shape[0]
            bank.append(sel)
        feats = torch.cat(bank)
        del bank  # free the per-batch chunks before any FP32 conversion
        n_sel = max(1, int(round(self.coreset_ratio * feats.shape[0])))
        idx = greedy_coreset(feats, n_sel, self.proj_dim, self.seed)
        self.memory = feats[idx].float().contiguous()
        return {
            "n_patches": int(feats.shape[0]),
            "n_offmask_patches": n_off,
            "memory_size": int(self.memory.shape[0]),
        }

    @torch.no_grad()
    def predict(self, x: torch.Tensor, out_size: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.embedder(x)  # (B, N, D)
        b, n, d = emb.shape
        side = int(math.isqrt(n))
        dist, _ = knn(emb.reshape(-1, d).float(), self.memory, k=1, backend=self.knn_backend)
        patch = dist[:, 0].clamp_min(0).sqrt().reshape(b, side, side)
        maps = smooth_maps(patch, out_size or x.shape[-1], self.blur_sigma)
        return patch.flatten(1).amax(1), maps


class AnomalyDINO(nn.Module):
    def __init__(
        self,
        backbone: str = "dinov2_vits14",
        img_size: int = 448,
        top_frac: float = 0.01,
        blur_sigma: float = 4.0,
        rotations: bool = True,
        knn_backend: str = "torch",
        pretrained: bool = True,
        max_bank: int | None = None,
        seed: int = 0,
    ):
        super().__init__()
        self.encoder = DinoV2(backbone, img_size=img_size, pretrained=pretrained)
        self.top_frac = top_frac
        self.blur_sigma = blur_sigma
        self.rotations = rotations
        self.knn_backend = knn_backend
        self.max_bank = max_bank
        self.seed = seed
        self.register_buffer("memory", torch.empty(0, self.encoder.embed_dim), persistent=True)

    def _patches(self, x: torch.Tensor) -> torch.Tensor:
        _, p = self.encoder.tokens(x)
        return F.normalize(p.float(), dim=-1)

    @torch.no_grad()
    def fit(
        self,
        loader,
        device: torch.device,
        max_images: int | None = None,
        offmask_loader=None,
        margin_px: int = 16,
    ) -> dict:
        self.to(device).eval()
        d = self.encoder.embed_dim
        npatch = self.encoder.grid**2
        n_views = 4 if self.rotations else 1

        def _count(ld) -> int:  # DataLoader or any re-iterable of batches (unit tests)
            return len(ld.dataset) if hasattr(ld, "dataset") else sum(b["image"].shape[0] for b in ld)

        n_img = _count(loader) if max_images is None else min(max_images, _count(loader))
        n_off_img = _count(offmask_loader) if offmask_loader is not None else 0
        upper = n_img * n_views * npatch + n_off_img * npatch
        if self.max_bank:  # coreset path needs all features at once (FP16 staging)
            buf = torch.empty(upper, d, dtype=torch.float16, device=device)
        else:  # fill the FP32 bank in place: no FP16 staging copy (peak = bank + one batch)
            buf = torch.empty(upper, d, dtype=torch.float32, device=device)
        o, seen = 0, 0
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            views = [x] + ([torch.rot90(x, k, (-2, -1)) for k in (1, 2, 3)] if self.rotations else [])
            for v in views:
                f = self._patches(v).reshape(-1, d)
                buf[o : o + f.shape[0]] = f
                o += f.shape[0]
            seen += x.shape[0]
            if max_images and seen >= max_images:
                break
        n_off = 0
        for batch in offmask_loader or []:
            emb = self._patches(batch["image"].to(device, non_blocking=True))
            sel = offmask_patches(emb, batch["mask"].to(device), margin_px)
            buf[o : o + sel.shape[0]] = sel
            o += sel.shape[0]
            n_off += sel.shape[0]
        n_total = o
        feats = buf[:n_total]
        if self.max_bank and n_total > self.max_bank:
            idx = greedy_coreset(feats, self.max_bank, seed=self.seed)
            self.memory = feats[idx].float().contiguous()
            del buf
        else:
            self.memory = feats.float() if feats.dtype != torch.float32 else feats  # view into buf, no copy
        return {
            "n_patches": int(n_total),
            "n_offmask_patches": n_off,
            "memory_size": int(self.memory.shape[0]),
        }

    @torch.no_grad()
    def predict(self, x: torch.Tensor, out_size: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        p = self._patches(x)
        b, n, d = p.shape
        side = int(math.isqrt(n))
        dist2, _ = knn(p.reshape(-1, d), self.memory, k=1, backend=self.knn_backend)
        cos_dist = (dist2[:, 0] / 2).clamp(0, 2).reshape(b, n)  # ||a-b||^2 = 2 - 2cos for unit vectors
        n_top = max(1, int(round(self.top_frac * n)))
        score = cos_dist.topk(n_top, dim=1).values.mean(1)
        maps = smooth_maps(cos_dist.reshape(b, side, side), out_size or x.shape[-1], self.blur_sigma)
        return score, maps
