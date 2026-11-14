"""Checkpoint / resume support shared by every training script.

* Time-based saves (default every 15 min, Colab rule: <= 20 min) plus end-of-epoch saves.
* Atomic writes (tmp file + os.replace) so a Colab disconnect never leaves a corrupt `last.ckpt`.
* Stores model/optimizer/scheduler/scaler state, epoch/step counters, RNG states and the
  MLflow run id so a resumed job continues the same MLflow run.
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


class CheckpointManager:
    def __init__(self, ckpt_dir: str | Path, every_minutes: float = 15.0, keep_best: bool = True):
        self.dir = Path(ckpt_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.every_s = every_minutes * 60.0
        self.keep_best = keep_best
        self._last_save = time.monotonic()
        self.best_metric: float | None = None

    @property
    def last_path(self) -> Path:
        return self.dir / "last.ckpt"

    @property
    def best_path(self) -> Path:
        return self.dir / "best.ckpt"

    def due(self) -> bool:
        return (time.monotonic() - self._last_save) >= self.every_s

    def save(self, state: dict[str, Any], path: Path | None = None) -> Path:
        path = path or self.last_path
        state = dict(state)
        state["rng"] = _rng_state()
        state["best_metric"] = self.best_metric
        tmp = path.with_suffix(".tmp")
        torch.save(state, tmp)
        os.replace(tmp, path)
        self._last_save = time.monotonic()
        return path

    def maybe_save(self, state_fn) -> bool:
        """Save `state_fn()` if the time budget elapsed (state built lazily)."""
        if self.due():
            self.save(state_fn())
            return True
        return False

    def update_best(self, metric: float, state: dict[str, Any], mode: str = "max") -> bool:
        better = (
            self.best_metric is None
            or (mode == "max" and metric > self.best_metric)
            or (mode == "min" and metric < self.best_metric)
        )
        if better:
            self.best_metric = metric
            if self.keep_best:
                self.save(state, self.best_path)
        return better

    def finalize(self, keep_last: bool = False) -> None:
        """Called after a run completed (test metrics written): `last.ckpt` (with optimizer state) is
        only needed to resume an unfinished run, so it is removed unless `keep_last`."""
        if not keep_last:
            self.last_path.unlink(missing_ok=True)

    def load_last(self, map_location: str | torch.device = "cpu") -> dict[str, Any] | None:
        if not self.last_path.exists():
            return None
        state = torch.load(self.last_path, map_location=map_location, weights_only=False)
        self.best_metric = state.get("best_metric")
        if "rng" in state:
            _set_rng_state(state["rng"])
        return state


def _rng_state() -> dict[str, Any]:
    st = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st: dict[str, Any]) -> None:
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(st["cuda"])
        except RuntimeError:  # different #GPUs after resume
            pass
