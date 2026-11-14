"""Joint image/mask augmentation on tensors (grey [0,1] image, binary mask).

SEM defect appearance on unpatterned wafers is (to first order) rotation- and flip-invariant, so the
dihedral group D4 is used, plus mild intensity jitter (contrast/brightness/gamma) emulating
detector-gain changes.  Geometric warps that could push the centred defect out of frame are avoided.
"""

from __future__ import annotations

import torch


class JointAugment:
    def __init__(
        self,
        flips: bool = True,
        rot90: bool = True,
        brightness: float = 0.1,
        contrast: float = 0.15,
        gamma: float = 0.15,
        generator: torch.Generator | None = None,
    ):
        self.flips = flips
        self.rot90 = rot90
        self.brightness = brightness
        self.contrast = contrast
        self.gamma = gamma
        self.g = generator

    def _u(self, lo: float, hi: float) -> float:
        return float(torch.empty(1).uniform_(lo, hi, generator=self.g))

    def __call__(self, x: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.flips and self._u(0, 1) < 0.5:
            x, m = x.flip(-1), m.flip(-1)
        if self.flips and self._u(0, 1) < 0.5:
            x, m = x.flip(-2), m.flip(-2)
        if self.rot90:
            k = int(self._u(0, 4)) % 4
            if k:
                x, m = torch.rot90(x, k, (-2, -1)), torch.rot90(m, k, (-2, -1))
        if self.contrast:
            mean = x.mean()
            x = (x - mean) * self._u(1 - self.contrast, 1 + self.contrast) + mean
        if self.brightness:
            x = x + self._u(-self.brightness, self.brightness)
        x = x.clamp(0, 1)
        if self.gamma:
            x = x.clamp_min(1e-6) ** self._u(1 - self.gamma, 1 + self.gamma)
        return x.contiguous(), m.contiguous()
