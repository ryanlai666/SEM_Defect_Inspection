"""Fixtures: a tiny on-disk dataset in the exact Carinthia-S layout (CSV ; images/ masks/), so the
real loader / split / engine code paths run on CPU in CI without downloading anything."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from semdefect.config import load_config

REPO = Path(__file__).resolve().parents[1]


def _make_fake_carinthia(root: Path, per_class: dict[int, int], size: int = 96, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    rows = ["image_path;mask_path;filename;label"]
    yy, xx = np.mgrid[0:size, 0:size]
    k = 0
    for label, n in per_class.items():
        for _ in range(n):
            name = f"{k:032x}"
            k += 1
            img = rng.normal(100, 6, (size, size))
            m = np.zeros((size, size), bool)
            if label != 6:
                cy, cx = rng.integers(size // 3, 2 * size // 3, 2)
                r = 3 + label * 2
                m = (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2
                img[m] += 60 if label % 2 else -60
            img[:, -3:] = 5  # dark right border like the real data
            Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(
                root / "images" / f"{name}.jpg", quality=95
            )
            Image.fromarray((m * 255).astype(np.uint8)).save(root / "masks" / f"{name}.png")
            rows.append(f"images/{name}.jpg;masks/{name}.png;{name};{label}")
    (root / "carinthia-s.csv").write_text("\n".join(rows) + "\n")
    return root


@pytest.fixture(scope="session")
def fake_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("carinthia") / "data"
    return _make_fake_carinthia(root, {1: 6, 2: 4, 3: 12, 4: 6, 5: 3, 6: 12})


@pytest.fixture()
def cfg_factory(fake_root, tmp_path):
    def make(config: str, *overrides: str) -> dict:
        base = [
            f"data.root={fake_root.as_posix()}",
            f"data.split_dir={(tmp_path / 'splits').as_posix()}",
            f"output_dir={(tmp_path / 'out').as_posix()}",
            "tracking.enabled=false",
            "data.num_workers=0",
            "device=cpu",
            "amp=false",
            "smoke.enabled=true",
            "smoke.n_images=24",
            "smoke.max_steps=2",
            "model.pretrained=false",
            "smoke.keep_checkpoints=true",
            f"data.cache_dir={(tmp_path / 'cache').as_posix()}",  # never write into the repo's data/cache
        ]
        return load_config(REPO / config, base + list(overrides))

    return make


@pytest.fixture(autouse=True)
def _threads():
    torch.set_num_threads(min(4, torch.get_num_threads()))
