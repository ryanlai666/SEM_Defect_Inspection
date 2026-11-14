"""Shared pieces of the detector benchmark (run on a Colab L4): one evaluator, one FLOP counter, one timer.

* COCO evaluation with pycocotools on the test split (boxes = mask connected components, 5 classes):
  mAP50-95, mAP50, AP_small, AR_small@100 (IoU .50:.95) and recall at IoU 0.50 on small defects
  (area < 32^2 px), plus class-agnostic mAP50 / mAP50-95 (all classes merged: localisation only).
* GFLOPs = torch.utils.flop_counter.FlopCounterMode on a 1x3x480x480 input (2 FLOPs per multiply-add),
  the same counter for every architecture.
* Latency: batch 1, FP16, CUDA-synchronised wall time; 50 warm-up calls excluded, 300 timed calls.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import time
from pathlib import Path

import numpy as np
import torch

EPOCHS = 35  # fixed budget for every detector (= the local U-Net's 6,000 x 16 samples / 2,753 train images)
IMGSZ = 480
WARM, N = 50, 300


def coco_metrics(gt_json: str | Path, results: list[dict]) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(gt_json))

    def run(gt_coco, res, agnostic: bool) -> COCOeval:
        if agnostic:
            gt_coco = copy.deepcopy(gt_coco)
            for a in gt_coco.dataset["annotations"]:
                a["category_id"] = 1
            gt_coco.dataset["categories"] = [{"id": 1, "name": "defect"}]
            gt_coco.createIndex()
            res = [{**r, "category_id": 1} for r in res]
        with contextlib.redirect_stdout(io.StringIO()):
            dt = gt_coco.loadRes(res) if res else COCO()
            ev = COCOeval(gt_coco, dt, "bbox")
            ev.params.maxDets = [1, 10, 100]
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        return ev

    if not results:
        return {"mAP50_95": 0.0, "mAP50": 0.0, "n_detections": 0}
    ev = run(gt, results, False)
    rec = ev.eval["recall"]  # [T, K, A, M]; area index 1 = small, maxDets index 2 = 100
    r50_small = rec[0, :, 1, 2]
    ag = run(gt, results, True)
    n_small = sum(1 for a in gt.dataset["annotations"] if a["area"] < 32**2)
    return {
        "mAP50_95": float(ev.stats[0]),
        "mAP50": float(ev.stats[1]),
        "AP_small": float(ev.stats[3]),
        "AR_small_100": float(ev.stats[9]),
        "recall50_small": float(r50_small[r50_small > -1].mean()) if (r50_small > -1).any() else float("nan"),
        "agnostic_mAP50_95": float(ag.stats[0]),
        "agnostic_mAP50": float(ag.stats[1]),
        "agnostic_recall50_small": float(ag.eval["recall"][0, 0, 1, 2]),
        "n_gt_boxes": len(gt.dataset["annotations"]),
        "n_gt_small": n_small,
        "n_detections": len(results),
    }


def gflops(model: torch.nn.Module, x: torch.Tensor) -> float:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.utils.flop_counter import FlopCounterMode

    # math SDPA backend so attention FLOPs are counted the same way for every model (as DEIMv2's profiler)
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH), FlopCounterMode(display=False) as fc:
        model(x)
    return fc.get_total_flops() / 1e9


def n_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def timed(fn) -> dict:
    for _ in range(WARM):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(N):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    a = np.asarray(ts)
    return {
        "p50_ms": float(np.percentile(a, 50)),
        "p99_ms": float(np.percentile(a, 99)),
        "mean_ms": float(a.mean()),
    }


def env() -> dict:
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.device_count() else "cpu",
    }


def save(out_dir: Path, res: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "det_metrics.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "env"}, indent=1))
