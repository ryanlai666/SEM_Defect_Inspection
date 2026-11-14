"""Dinomaly-lite (Guo et al., arXiv:2405.14325), own compact implementation.

Frozen DINOv2(-reg) encoder; the mean of the middle encoder layers passes a *noisy bottleneck*
(MLP with dropout) into a decoder of `n_dec` transformer blocks with *linear attention*; decoder
outputs are grouped and compared to grouped encoder layers ("loose reconstruction") with a
hard-mining global cosine loss ("loose loss").  Anomaly map = mean over groups of per-patch cosine
distance; image score = mean of the top-1% map values.

Differences from the paper (documented in README): ViT-S/14 default (8 GB GPU), fixed 2 groups,
AdamW + cosine schedule instead of StableAdamW.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .anomaly import smooth_maps
from .backbones import DinoV2


class LinearAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 6):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        # The sums over n tokens of positive (ELU+1) features overflow FP16 (max 65504) for a few hundred
        # tokens -> NaN under fp16 autocast.  Do the reductions in FP32 regardless of autocast.
        with torch.autocast(device_type=x.device.type, enabled=False):
            q, k, v = F.elu(q.float()) + 1, F.elu(k.float()) + 1, v.float()
            kv = torch.einsum("bhnd,bhne->bhde", k, v)
            z = 1.0 / (torch.einsum("bhnd,bhd->bhn", q, k.sum(2)) + 1e-6)
            out = torch.einsum("bhnd,bhde,bhn->bhne", q, kv, z)
        return self.proj(out.transpose(1, 2).reshape(b, n, d).to(x.dtype))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = LinearAttention(dim, heads)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class Dinomaly(nn.Module):
    def __init__(
        self,
        backbone: str = "dinov2_vits14_reg",
        img_size: int = 280,
        enc_layers: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9),
        n_groups: int = 2,
        dropout: float = 0.2,
        top_frac: float = 0.01,
        blur_sigma: float = 4.0,
        pretrained: bool = True,
    ):
        super().__init__()
        self.encoder = DinoV2(backbone, img_size=img_size, pretrained=pretrained)
        d = self.encoder.embed_dim
        heads = max(1, d // 64)
        self.enc_layers = list(enc_layers)
        self.n_groups = n_groups
        self.bottleneck = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d), nn.Dropout(dropout)
        )
        self.decoder = nn.ModuleList([Block(d, heads) for _ in range(len(enc_layers))])
        self.top_frac = top_frac
        self.blur_sigma = blur_sigma

    @torch.no_grad()
    def _encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        m = self.encoder.model
        feats = m.forward_intermediates(
            x, indices=self.enc_layers, norm=False, output_fmt="NLC", intermediates_only=True
        )
        return [f.float() for f in feats]  # each (B, N, D) patch tokens (prefix dropped by timm)

    def _groups(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        size = len(feats) // self.n_groups
        return [torch.stack(feats[i * size : (i + 1) * size]).mean(0) for i in range(self.n_groups)]

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        enc = self._encode(x)
        h = self.bottleneck(torch.stack(enc).mean(0))
        dec = []
        for blk in self.decoder:
            h = blk(h)
            dec.append(h)
        return self._groups(enc), self._groups(dec[::-1])

    def loss(
        self,
        x: torch.Tensor,
        hard_frac: float = 0.9,
        easy_scale: float = 0.1,
        keep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Global hard-mining cosine loss: gradients of the easiest (1-hard_frac) points are scaled.

        keep: optional (B,N) bool, False for patches near a defect (off-mask protocol).  Dropped patches
        are zeroed in both encoder and decoder tokens, so they add nothing to the global cosine and are
        never learned as "normal"; the hard-mining quantile is taken over kept patches only."""
        e_groups, d_groups = self(x)
        total = 0.0
        for e, d in zip(e_groups, d_groups):
            if keep is not None:
                w = keep.to(e.dtype).unsqueeze(-1)
                e, d = e * w, d * w
            point_dist = 1 - F.cosine_similarity(e.detach(), d, dim=-1)  # (B,N)
            pool = point_dist.detach()[keep] if keep is not None else point_dist.detach().flatten()
            if pool.numel() == 0:
                pool = point_dist.detach().flatten()
            thr = torch.quantile(pool.float(), 1 - hard_frac)
            scale = torch.where(point_dist.detach() < thr, easy_scale, 1.0)
            d_scaled = d * scale.unsqueeze(-1) + d.detach() * (1 - scale.unsqueeze(-1))
            total = (
                total + (1 - F.cosine_similarity(e.detach().flatten(1), d_scaled.flatten(1), dim=1)).mean()
            )
        return total / len(e_groups)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, out_size: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        e_groups, d_groups = self(x)
        dist = torch.stack([1 - F.cosine_similarity(e, d, dim=-1) for e, d in zip(e_groups, d_groups)]).mean(
            0
        )
        b, n = dist.shape
        side = int(math.isqrt(n))
        maps = smooth_maps(dist.reshape(b, side, side), out_size or x.shape[-1], self.blur_sigma)
        flat = maps.flatten(1)
        k = max(1, int(self.top_frac * flat.shape[1]))
        return flat.topk(k, dim=1).values.mean(1), maps
