"""Import before `ultralytics`: keep its settings file inside the project and switch off its network sync
(anonymous usage analytics) and auto-logging integrations (it would otherwise auto-log to MLflow etc.)."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))


def quiet_settings() -> None:
    from ultralytics import settings

    settings.update(
        {
            k: False
            for k in ("sync", "mlflow", "clearml", "comet", "dvc", "raytune", "tensorboard", "wandb")
            if k in settings
        }
    )
