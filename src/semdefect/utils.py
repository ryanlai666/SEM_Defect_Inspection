"""Small shared helpers: seeding, device selection, timing, VRAM accounting, JSON IO."""

from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


CUDA_CONTEXT_GB = 0.4  # approximate CUDA context + cuDNN/cuBLAS handles outside the caching allocator


def gpu_guard_gb() -> float:
    """VRAM reservation (GB) granted by the shared-GPU guard (env GPU_GUARD_GB), 0 if not guarded."""
    try:
        return float(os.environ.get("GPU_GUARD_GB", "0") or 0)
    except ValueError:
        return 0.0


def apply_gpu_guard_cap(extra_cap_gb: float = 0.0) -> float | None:
    """Cap this process to its VRAM reservation on a shared GPU (env GPU_GUARD_GB, see scripts/guarded.py).

    PyTorch's caching allocator is limited to (reservation - CUDA context) so that the whole process
    (allocator + context) stays inside the reservation.  CuPy's default pool gets the same limit.  An optional
    `extra_cap_gb` (smoke tests) tightens it further.  Returns the applied allocator cap in GB."""
    gb = gpu_guard_gb()
    caps = [c for c in (gb - CUDA_CONTEXT_GB if gb > 0 else 0.0, extra_cap_gb) if c > 0]
    if not caps or not torch.cuda.is_available():
        return None
    cap = max(0.25, min(caps))
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap * 2**30 / total))
    try:  # CuPy is optional
        import cupy as cp

        cp.get_default_memory_pool().set_limit(size=int(cap * 2**30))
    except Exception:
        pass
    return cap


def get_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def resolve_path(p: str | Path) -> Path:
    """Resolve config paths relative to the repo root (works from any cwd)."""
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p)


@contextmanager
def cuda_peak_memory(device: torch.device | None = None):
    """Context manager recording peak allocated CUDA memory (MiB) into the yielded dict."""
    stats: dict[str, float] = {"peak_mib": 0.0}
    use = torch.cuda.is_available() and (device is None or device.type == "cuda")
    if use:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    try:
        yield stats
    finally:
        if use:
            torch.cuda.synchronize()
            stats["peak_mib"] = torch.cuda.max_memory_allocated() / 2**20
            stats["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20


class Throughput:
    """Accumulates items/s while excluding warm-up iterations."""

    def __init__(self, warmup: int = 2):
        self.warmup = warmup
        self.n_iter = 0
        self.items = 0
        self.elapsed = 0.0
        self.all_items = 0
        self.all_elapsed = 0.0
        self._t0 = 0.0

    def start(self) -> None:
        if torch.cuda.is_available() and torch.cuda.is_initialized():  # never create a context in CPU runs
            torch.cuda.synchronize()
        self._t0 = time.perf_counter()

    def stop(self, n_items: int) -> None:
        if torch.cuda.is_available() and torch.cuda.is_initialized():  # never create a context in CPU runs
            torch.cuda.synchronize()
        dt = time.perf_counter() - self._t0
        self.n_iter += 1
        self.all_items += n_items
        self.all_elapsed += dt
        if self.n_iter > self.warmup:
            self.items += n_items
            self.elapsed += dt

    @property
    def items_per_s(self) -> float:
        """Post-warm-up rate; falls back to the all-iterations rate for very short runs."""
        if self.elapsed > 0:
            return self.items / self.elapsed
        return self.all_items / self.all_elapsed if self.all_elapsed > 0 else float("nan")

    @property
    def iters_per_s(self) -> float:
        n = self.n_iter - self.warmup
        if self.elapsed > 0 and n > 0:
            return n / self.elapsed
        return self.n_iter / self.all_elapsed if self.all_elapsed > 0 else float("nan")


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer | np.floating):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")
