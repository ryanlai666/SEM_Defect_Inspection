"""SEM-like image degradations for robustness evaluation and noise augmentation.

All functions take grey images in [0,1] shaped (B,1,H,W) or (1,H,W) and an optional
`torch.Generator` (seeded per batch => deterministic sweeps).

* shot noise   - SEM secondary-electron counts are Poisson: I' = Poisson(dose * I) / dose, where
                 `dose` is the expected electron count at full-scale intensity (lower = faster scan,
                 noisier image).
* blur         - Gaussian beam-spot / defocus blur, sigma in pixels at model resolution.
* charging     - low-frequency brightness drift from surface charging on insulating films: a smooth
                 random field (bicubic-upsampled coarse grid) plus a ramp along the slow-scan axis.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F

SEVERITIES: dict[str, list[float]] = {
    # level 0 = clean is implicit; 5 increasing severities each
    "shot_noise": [200.0, 100.0, 50.0, 20.0, 10.0],  # dose (electrons @ full scale)
    "blur": [0.5, 1.0, 1.5, 2.0, 3.0],  # sigma px
    "charging": [0.05, 0.1, 0.15, 0.2, 0.3],  # drift amplitude (fraction of full scale)
}


def _as4d(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
    return (x.unsqueeze(0), True) if x.dim() == 3 else (x, False)


def shot_noise(x: torch.Tensor, dose: float, generator: torch.Generator | None = None) -> torch.Tensor:
    lam = (x.clamp(0, 1) * dose).float()
    if generator is not None and generator.device.type != lam.device.type:
        noisy = torch.poisson(lam.cpu(), generator=generator).to(lam.device)
    else:
        noisy = torch.poisson(lam, generator=generator)
    return (noisy / dose).to(x.dtype).clamp(0, 1)


def gaussian_kernel1d(sigma: float, device=None, dtype=torch.float32) -> torch.Tensor:
    radius = max(1, int(math.ceil(3 * sigma)))
    t = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    return k / k.sum()


def blur(x: torch.Tensor, sigma: float, generator: torch.Generator | None = None) -> torch.Tensor:
    if sigma <= 0:
        return x
    x4, sq = _as4d(x)
    c = x4.shape[1]
    k = gaussian_kernel1d(sigma, x4.device, x4.dtype)
    r = (k.numel() - 1) // 2
    y = F.pad(x4, (r, r, 0, 0), mode="reflect")
    y = F.conv2d(y, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    y = F.pad(y, (0, 0, r, r), mode="reflect")
    y = F.conv2d(y, k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)
    return y.squeeze(0) if sq else y


def charging_drift(
    x: torch.Tensor, amplitude: float, generator: torch.Generator | None = None, grid: int = 4
) -> torch.Tensor:
    x4, sq = _as4d(x)
    b, _, h, w = x4.shape
    coarse = torch.randn(b, 1, grid, grid, generator=generator, device="cpu").to(x4.device, x4.dtype)
    field = F.interpolate(coarse, size=(h, w), mode="bicubic", align_corners=True)
    field = field / field.flatten(1).std(dim=1).clamp_min(1e-6).view(b, 1, 1, 1)
    ramp_dir = torch.rand(b, generator=generator, device="cpu").to(x4.device, x4.dtype) * 2 - 1
    ramp = torch.linspace(-1, 1, h, device=x4.device, dtype=x4.dtype).view(1, 1, h, 1)
    drift = amplitude * (0.5 * field + 0.5 * ramp_dir.view(b, 1, 1, 1) * ramp)
    y = (x4 + drift).clamp(0, 1)
    return y.squeeze(0) if sq else y


PERTURBATIONS: dict[str, Callable] = {"shot_noise": shot_noise, "blur": blur, "charging": charging_drift}


def make_perturbation(
    name: str, level: int
) -> Callable[[torch.Tensor, torch.Generator | None], torch.Tensor]:
    """level in 1..5 (0 => identity)."""
    if level == 0 or name == "clean":
        return lambda x, g=None: x
    fn = PERTURBATIONS[name]
    param = SEVERITIES[name][level - 1]
    return lambda x, g=None: fn(x, param, generator=g)


class RandomSEMNoise:
    """Train-time augmentation: with prob p apply one random perturbation at a random level<=max_level."""

    def __init__(self, p: float = 0.5, max_level: int = 3, generator: torch.Generator | None = None):
        self.p, self.max_level, self.g = p, max_level, generator

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if float(torch.rand(1, generator=self.g)) >= self.p:
            return x
        names = list(PERTURBATIONS)
        name = names[int(torch.randint(len(names), (1,), generator=self.g))]
        level = int(torch.randint(1, self.max_level + 1, (1,), generator=self.g))
        return make_perturbation(name, level)(x, self.g)
