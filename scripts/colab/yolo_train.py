"""Train YOLO26n-seg (Ultralytics, AGPL-3.0) on the Carinthia-S defect masks (Colab L4).

COCO-pretrained yolo26n-seg.pt, imgsz 480 (images are 477x477 after the border crop), 5 defect classes,
class-6 images as background.  Best checkpoint = Ultralytics' fitness on the val split (as the U-Net's
best-on-val Dice).  Resumes from last.pt (written every epoch) if the VM restarts.

    python scripts/colab/yolo_train.py [--epochs 35]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _yolo_env import quiet_settings  # sets YOLO_CONFIG_DIR before ultralytics is imported

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "outputs" / "yolo26n_seg_colab"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/yolo_data/data.yaml")
    ap.add_argument("--epochs", type=int, default=35)  # fixed budget shared with the detector benchmark
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default=None, help="e.g. cpu (local smoke test); default: Ultralytics auto")
    ap.add_argument("--run-dir", default=str(RUN))
    a = ap.parse_args()
    run = Path(a.run_dir)

    import ultralytics
    from ultralytics import YOLO

    quiet_settings()

    last = run / "train" / "weights" / "last.pt"
    if last.exists():
        model = YOLO(str(last))
        model.train(resume=True)
    else:
        model = YOLO("yolo26n-seg.pt")
        model.train(
            data=a.data,
            epochs=a.epochs,
            imgsz=480,
            batch=a.batch,
            workers=a.workers,
            seed=0,
            deterministic=True,
            patience=0,  # fixed budget: no early stopping
            project=str(run),
            device=a.device,
            name="train",
            exist_ok=True,
            flipud=0.5,  # SEM top views have no preferred orientation
            fliplr=0.5,
            plots=True,
            amp=True,
        )
    (run / "train_env.json").write_text(
        json.dumps({"ultralytics": ultralytics.__version__, "epochs": a.epochs})
    )


if __name__ == "__main__":
    main()
