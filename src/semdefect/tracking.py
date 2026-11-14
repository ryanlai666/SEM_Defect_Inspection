"""MLflow wrapper.  Local file store at ./mlruns by default; degrades to a no-op if MLflow
is missing or `tracking.enabled: false` (unit tests)."""

from __future__ import annotations

import math
import os
from typing import Any

from .config import flatten
from .utils import REPO_ROOT


class Tracker:
    def __init__(self, cfg: dict, run_name: str | None = None, resume_run_id: str | None = None):
        tcfg = cfg.get("tracking", {})
        self.enabled = bool(tcfg.get("enabled", True))
        self.run_id: str | None = None
        self._mlflow = None
        if not self.enabled:
            return
        # MLflow >= 3.x keeps the ./mlruns file store in maintenance mode behind an opt-in.  We keep it:
        # it needs no server and, unlike SQLite, it is safe on a Colab Google-Drive mount.
        # Set tracking.uri to e.g. sqlite:///mlruns/mlflow.db to use a database backend instead.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        try:
            import mlflow
        except ImportError:  # pragma: no cover
            self.enabled = False
            return
        self._mlflow = mlflow
        uri = tcfg.get("uri") or (REPO_ROOT / "mlruns").as_uri()
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(tcfg.get("experiment", cfg.get("task", "semdefect")))
        if resume_run_id:
            run = mlflow.start_run(run_id=resume_run_id)
        else:
            run = mlflow.start_run(run_name=run_name or cfg.get("name"))
            params = {k: _short(v) for k, v in flatten(cfg).items() if not k.startswith("_meta")}
            # MLflow caps a single log_params batch at 100 entries.
            items = list(params.items())
            for i in range(0, len(items), 90):
                mlflow.log_params(dict(items[i : i + 90]))
        self.run_id = run.info.run_id

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if not self.enabled:
            return
        clean = {k: float(v) for k, v in metrics.items() if _is_number(v)}
        if clean:
            self._mlflow.log_metrics(clean, step=step)

    def set_tags(self, tags: dict[str, Any]) -> None:
        if self.enabled:
            self._mlflow.set_tags({k: str(v) for k, v in tags.items()})

    def log_artifact(self, path: str) -> None:
        if self.enabled:
            self._mlflow.log_artifact(path)

    def end(self, status: str = "FINISHED") -> None:
        if self.enabled and self._mlflow.active_run() is not None:
            self._mlflow.end_run(status=status)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        self.end("FAILED" if exc_type else "FINISHED")


def _is_number(v: Any) -> bool:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return not math.isnan(f)


def _short(v: Any) -> str:
    s = str(v)
    return s if len(s) <= 500 else s[:497] + "..."
