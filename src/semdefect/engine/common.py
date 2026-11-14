"""Shared plumbing for the training / evaluation entry points."""

from __future__ import annotations

import shutil
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from ..data import CarinthiaS, SyntheticSEM, load_index, load_or_make_split
from ..data.carinthia import to_model_input
from ..robustness import make_perturbation
from ..utils import (
    REPO_ROOT,
    apply_gpu_guard_cap,
    cuda_peak_memory,
    get_device,
    save_json,
    seed_everything,
)


def setup(cfg: dict) -> torch.device:
    seed_everything(int(cfg.get("seed", 0)), bool(cfg.get("deterministic", False)))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # autotuning pays off for training; inference-only runs with a ragged last batch re-tune per shape
        torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", True))
        # Shared GPU: stay inside the gpu_guard reservation (GPU_GUARD_GB); smoke tests may cap tighter.
        smoke_cap = float(cfg.get("smoke", {}).get("vram_cap_gb", 0) or 0) if is_smoke(cfg) else 0.0
        apply_gpu_guard_cap(smoke_cap)
    return get_device(cfg.get("device", "auto"))


def output_dir(cfg: dict) -> Path:
    base = REPO_ROOT / cfg.get("output_dir", "outputs")  # absolute output_dir overrides REPO_ROOT
    d = base / cfg.get("name", "run")
    if cfg.get("smoke", {}).get("enabled"):
        d = base / "smoke" / cfg.get("name", "run")
    d.mkdir(parents=True, exist_ok=True)
    return d


def is_smoke(cfg: dict) -> bool:
    return bool(cfg.get("smoke", {}).get("enabled", False))


def load_data(cfg: dict):
    """Returns (df, split) for Carinthia-S, or (None, None) when `data.synthetic` is set (tests/CI)."""
    dcfg = cfg["data"]
    if dcfg.get("synthetic"):
        return None, None
    df = load_index(dcfg.get("root", "data/raw/carinthia_s/data"))
    split = load_or_make_split(
        df, seed=int(dcfg.get("split_seed", 0)), split_dir=dcfg.get("split_dir", "data/splits")
    )
    return df, split


def subsample(
    indices: Sequence[int], n: int | None, seed: int = 0, labels: Sequence[int] | None = None
) -> list[int]:
    """Deterministic subsample; when labels are given, round-robin over classes (keeps rare classes)."""
    indices = list(indices)
    if not n or n >= len(indices):
        return indices
    rng = np.random.default_rng(seed)
    if labels is None:
        return sorted(rng.choice(indices, n, replace=False).tolist())
    by_c: dict[int, list[int]] = {}
    for i, c in zip(indices, labels):
        by_c.setdefault(int(c), []).append(int(i))
    for c in by_c:
        by_c[c] = rng.permutation(by_c[c]).tolist()
    out: list[int] = []
    while len(out) < n:
        for c in sorted(by_c):
            if by_c[c] and len(out) < n:
                out.append(by_c[c].pop())
    return sorted(out)


def make_dataset(
    cfg: dict, df, indices, train: bool = False, img_size: int | None = None, mask_size: int | None = None
) -> Dataset:
    dcfg = cfg["data"]
    size = img_size or int(dcfg["img_size"])
    if dcfg.get("synthetic"):
        ds = SyntheticSEM(n=int(dcfg.get("synthetic_n", 24)), img_size=size, n_classes=3, seed=0)
        return Subset(ds, list(indices)) if indices is not None else ds
    augment = None
    if train and cfg.get("augment", {}).get("enabled", True):
        from ..data import JointAugment

        a = cfg.get("augment", {})
        augment = JointAugment(
            flips=a.get("flips", True),
            rot90=a.get("rot90", True),
            brightness=a.get("brightness", 0.1),
            contrast=a.get("contrast", 0.15),
            gamma=a.get("gamma", 0.15),
        )
    perturb = None
    if train and cfg.get("augment", {}).get("sem_noise", False):
        from ..robustness import RandomSEMNoise

        perturb = RandomSEMNoise(p=0.5, max_level=3)
    return CarinthiaS(
        df,
        indices,
        img_size=size,
        classes=dcfg.get("classes", [1, 2, 3, 4, 5, 6]),
        augment=augment,
        perturb=perturb,
        resize_mode=dcfg.get("resize_mode", "area"),
        cache=dcfg.get("cache", False),
        load_masks=bool(dcfg.get("load_masks", True)),
        mask_size=mask_size,
        cache_dir=dcfg.get("cache_dir", "data/cache/arrays"),
    )


def make_loader(
    ds: Dataset, batch_size: int, shuffle: bool, cfg: dict, drop_last: bool = False
) -> DataLoader:
    nw = int(cfg["data"].get("num_workers", 2))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=nw,
        # pinning host memory creates a CUDA context: only when the run actually uses the GPU
        pin_memory=torch.cuda.is_available() and str(cfg.get("device", "auto")) != "cpu",
        drop_last=drop_last,
        persistent_workers=nw > 0,
    )


def infinite(loader: DataLoader) -> Iterator[dict]:
    while True:
        yield from loader


def perturbed_batches(
    loader: DataLoader, device: torch.device, perturbation: str = "clean", level: int = 0, seed: int = 0
):
    """Yields batches with `image` recomputed from a deterministically perturbed `image01`."""
    fn = make_perturbation(perturbation, level)
    for bi, batch in enumerate(loader):
        x01 = batch["image01"].to(device, non_blocking=True)
        if level > 0:
            g = torch.Generator(device="cpu").manual_seed(seed * 100_003 + bi)
            x01 = fn(x01, g).clamp(0, 1)
        batch = dict(batch)
        batch["image01"] = x01
        batch["image"] = to_model_input(x01)
        yield batch


class SmokeRecorder:
    """Measures throughput and peak VRAM of a smoke run and writes outputs/smoke/<name>/smoke.json."""

    def __init__(self, cfg: dict, device: torch.device):
        self.cfg, self.device = cfg, device
        self.t0 = time.perf_counter()
        self.records: dict = {
            "name": cfg.get("name"),
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        }
        self._mem = cuda_peak_memory(device)
        self.mem_stats = self._mem.__enter__()

    def add(self, **kw) -> None:
        self.records.update(kw)

    def finish(self, out_dir: Path) -> dict:
        self._mem.__exit__(None, None, None)
        self.records["peak_vram_mib"] = round(self.mem_stats.get("peak_mib", 0.0), 1)
        self.records["peak_reserved_mib"] = round(self.mem_stats.get("peak_reserved_mib", 0.0), 1)
        self.records["wall_s"] = round(time.perf_counter() - self.t0, 2)
        save_json(self.records, out_dir / "smoke.json")
        # smoke checkpoints only exercise the save path; drop them (optimizer states are large)
        if not self.cfg.get("smoke", {}).get("keep_checkpoints", False):
            shutil.rmtree(out_dir / "checkpoints", ignore_errors=True)
        return self.records
