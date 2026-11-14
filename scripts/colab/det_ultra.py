"""Detector benchmark (Colab L4): Ultralytics detectors trained and evaluated identically.

  yolo26n    YOLO26n (COCO yolo26n.pt)
  sme_yolo   "SME-YOLO" exactly as implemented in the author's PCB defect-detection project
             (PCB_wafer_defect_detection/models/sme_yolo.py): a stock YOLOv11n
             (yolo11n.pt); that project notes no public SME-YOLO code and leaves NWD / EUCB / MSFA as TODOs
  rtdetr_l   RT-DETR-L (COCO rtdetr-l.pt), as models/rt_detr.py of that project

Fixed budget: EPOCHS epochs at imgsz 480, seed 0, early stopping off; the final (EMA) checkpoint is
evaluated (no model selection on val), with pycocotools on the test split (det_common.coco_metrics).

    python scripts/colab/det_ultra.py --name yolo26n
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from _yolo_env import quiet_settings  # sets YOLO_CONFIG_DIR before ultralytics is imported
from det_common import EPOCHS, IMGSZ, coco_metrics, env, gflops, n_params, save, timed

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = {"yolo26n": "yolo26n.pt", "sme_yolo": "yolo11n.pt", "rtdetr_l": "rtdetr-l.pt"}
BATCH = {"yolo26n": 32, "sme_yolo": 32, "rtdetr_l": 16}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", choices=list(WEIGHTS), required=True)
    ap.add_argument("--data", default="/content/det_data")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--skip-latency", action="store_true")
    a = ap.parse_args()
    data = Path(a.data)
    run = Path(a.run_dir) if a.run_dir else ROOT / "outputs" / f"det_{a.name}_colab"

    import ultralytics
    from ultralytics import RTDETR, YOLO

    quiet_settings()
    cls = RTDETR if a.name == "rtdetr_l" else YOLO
    last = run / "train" / "weights" / "last.pt"
    done = run / "train" / "results.csv"
    finished = done.exists() and len(done.read_text().strip().splitlines()) - 1 >= a.epochs
    if last.exists() and not finished:
        cls(str(last)).train(resume=True)
    elif not finished:
        cls(WEIGHTS[a.name]).train(
            data=str(data / "data.yaml"),
            epochs=a.epochs,
            imgsz=IMGSZ,
            batch=BATCH[a.name],
            workers=a.workers,
            device=a.device,
            seed=0,
            deterministic=True,
            patience=0,  # fixed budget: no early stopping
            project=str(run),
            name="train",
            exist_ok=True,
            flipud=0.5,
            fliplr=0.5,
            amp=True,
            plots=False,
        )

    # ------------------------------------------------------------ test predictions -> COCO results
    def predict(weights: Path) -> list[dict]:
        out = []
        for r in cls(str(weights)).predict(
            str(data / "images" / "test"),
            imgsz=IMGSZ,
            conf=0.001,
            max_det=100,
            stream=True,
            verbose=False,
            device=a.device,
        ):
            img_id = int(Path(r.path).stem)
            xyxy = r.boxes.xyxy.cpu().numpy()
            for (x0, y0, x1, y1), c, sc in zip(xyxy, r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                out.append(
                    {
                        "image_id": img_id,
                        "category_id": int(c) + 1,
                        "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                        "score": float(sc),
                    }
                )
        return out

    gt = data / "annotations" / "instances_test.json"
    results = predict(last)
    best = run / "train" / "weights" / "best.pt"
    res = {
        "model": a.name,
        "weights": WEIGHTS[a.name],
        "epochs": a.epochs,
        "imgsz": IMGSZ,
        "checkpoint": "last (EMA), no val selection",
        "test": coco_metrics(gt, results),
        # secondary: Ultralytics' own val-fitness checkpoint (the primary row uses no val selection)
        "test_best_on_val": coco_metrics(gt, predict(best)) if best.exists() else None,
        "env": {**env(), "ultralytics": ultralytics.__version__},
    }
    net = cls(str(last)).model.float().eval()
    res["params"] = n_params(net)
    dev = torch.device("cuda" if torch.cuda.device_count() and a.device != "cpu" else "cpu")
    net = net.to(dev)
    x = torch.rand(1, 3, IMGSZ, IMGSZ, device=dev)
    res["gflops_480"] = gflops(net, x)

    if not a.skip_latency and dev.type == "cuda":
        fused = net.fuse() if hasattr(net, "fuse") else net
        f16, x16 = fused.half(), x.half()
        with torch.no_grad():
            res["latency_model_fp16_bs1"] = timed(lambda: f16(x16))
        img = np.repeat(np.random.default_rng(0).integers(0, 256, (477, 477, 1), dtype=np.uint8), 3, axis=2)
        pm = cls(str(last))  # fresh predictor configured for FP16
        res["latency_e2e_fp16_bs1"] = timed(
            lambda: pm.predict(img, imgsz=IMGSZ, conf=0.25, max_det=100, quantize=16, verbose=False)[0].boxes
        )
        res["latency_note"] = (
            "model = forward on a GPU tensor (YOLOv11n's NMS is outside the forward; YOLO26 / RT-DETR are NMS-free); "
            "e2e = Ultralytics predict on a host uint8 image incl. letterbox, NMS / post-processing"
        )
    save(run, res)


if __name__ == "__main__":
    sys.exit(main())
