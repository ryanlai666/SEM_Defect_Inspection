"""Training-free sanity baselines.

`HighPassEnergy` (anomaly detection): remove low-frequency shading/charging with a Gaussian high-pass
(sigma_hp px), take the local RMS of the residual (Gaussian window sigma_map px) as the anomaly map and its
maximum as the image score.  No training data, no learned features; parameters fixed a priori
(sigma_hp = sigma_map = 4 px at the model resolution), never tuned on test data.

It exists to measure how *hard* a protocol is: if it matches a deep model, the protocol mostly tests
"is there any texture at all", not learned normality (see REPORT.md, section 4.1).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.carinthia import IMAGENET_MEAN, IMAGENET_STD
from ..robustness import blur


class HighPassEnergy(nn.Module):
    def __init__(self, sigma_hp: float = 4.0, sigma_map: float = 4.0, top_frac: float | None = None):
        super().__init__()
        self.sigma_hp = sigma_hp
        self.sigma_map = sigma_map
        self.top_frac = top_frac  # None -> max of the map (like PatchCore); else mean of top fraction

    def fit(self, *args, **kwargs) -> dict:  # training-free
        return {"n_patches": 0, "memory_size": 0}

    @torch.no_grad()
    def predict(self, x: torch.Tensor, out_size: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        grey = (
            x[:, :1].float() * IMAGENET_STD[0] + IMAGENET_MEAN[0]
        )  # undo normalisation (channels identical)
        hp = grey - blur(grey, self.sigma_hp)
        m = blur(hp.square(), self.sigma_map).clamp_min(0).sqrt()
        size = out_size or x.shape[-1]
        if m.shape[-1] != size:
            m = F.interpolate(m, size=(size, size), mode="bilinear", align_corners=False)
        m = m[:, 0]
        flat = m.flatten(1)
        if self.top_frac:
            k = max(1, int(self.top_frac * flat.shape[1]))
            return flat.topk(k, dim=1).values.mean(1), m
        return flat.amax(1), m
