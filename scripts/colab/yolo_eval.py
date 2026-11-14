"""Evaluate the trained YOLO26n-seg with the project's own metric code and protocols (Colab L4).

Per test image, from Ultralytics predictions (conf >= 0.001, retina masks at 477x477):
  * segmentation mask   = union of instance masks with conf >= 0.25 (Ultralytics' default, fixed a priori)
  * anomaly score map   = max over instances of conf_i * mask_i (a ranking for pixel AUROC / AP / AUPRO)
  * image anomaly score = max instance conf (0 without detections)
  * image class         = class of the top instance if conf >= 0.25, else class 6 ("no visible defect")
Metrics, identical definitions / resolutions as the local runs:
  * segmentation (U-Net table): SegmentationMeter on the 919 test images at 256 px
  * off-mask localisation task (lead AD table): pixel metrics on the 874 test defect images at 224 px
  * class-6 protocol: image AUROC + pixel metrics on the AD test set (45 normal + 874 defect) at 224 px
  * classification: 6-class / per-class F1 on the 919 test images

    python scripts/colab/yolo_eval.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from _yolo_env import quiet_settings  # sets YOLO_CONFIG_DIR before ultralytics is imported

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from semdefect.data import anomaly_split, load_index, load_or_make_split, read_mask  # noqa: E402
from semdefect.data.carinthia import crop_border, read_gray, resize01  # noqa: E402
from semdefect.metrics import SegmentationMeter, anomaly_metrics, classification_metrics  # noqa: E402

RUN = ROOT / "outputs" / "yolo26n_seg_colab"
SEG_SIZE, AD_SIZE, CONF = 256, 224, 0.25


def gt_mask(path: str, size: int) -> np.ndarray:
    m = torch.from_numpy(crop_border(read_mask(path)).astype(np.float32))[None]
    return (resize01(m, size, "area") >= 0.5).float()[0].numpy()


def down(a: np.ndarray, size: int) -> np.ndarray:
    return resize01(torch.from_numpy(a.astype(np.float32))[None], size, "area")[0].numpy()


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(RUN))
    ap.add_argument(
        "--limit", type=int, default=0, help="about N evenly spaced test images (smoke test only)"
    )
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    run = Path(a.run_dir)
    from ultralytics import YOLO

    quiet_settings()

    df0 = load_index(ROOT / "data/raw/carinthia_s/data")
    split = load_or_make_split(df0, "carinthia_s", seed=0)
    df = df0.set_index("idx")
    lab = df["label"]
    test = list(split["test"])
    if a.limit:
        test = test[:: max(1, len(test) // a.limit)]
    model = YOLO(str(run / "train" / "weights" / "best.pt"))

    pred: dict[int, dict] = {}
    t0 = time.perf_counter()
    for i in test:
        img = crop_border(read_gray(df.loc[i, "image_path"]))
        r = model.predict(
            np.repeat(img[:, :, None], 3, axis=2),
            imgsz=480,
            conf=0.001,
            retina_masks=True,
            verbose=False,
            device=a.device,
        )[0]
        h, w = img.shape
        smap = np.zeros((h, w), np.float32)
        seg = np.zeros((h, w), bool)
        score, cls = 0.0, 6
        if r.masks is not None and len(r.boxes):
            confs = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int) + 1
            masks = r.masks.data.cpu().numpy() > 0.5
            for c, m in zip(confs, masks):
                smap = np.maximum(smap, c * m)
                if c >= CONF:
                    seg |= m
            top = int(np.argmax(confs))
            score = float(confs[top])
            if confs[top] >= CONF:
                cls = int(classes[top])
        pred[i] = {"map": smap, "seg": seg, "score": score, "cls": cls}
    infer_s = time.perf_counter() - t0

    # segmentation (same as evaluate_segmenter: 256 px, threshold 0.5 on the area-resized prediction)
    meter = SegmentationMeter(0.5)
    for i in test:
        meter.update(down(pred[i]["seg"], SEG_SIZE)[None], gt_mask(df.loc[i, "mask_path"], SEG_SIZE)[None])
    seg_met = meter.compute()
    np.savez_compressed(run / "test_per_image.npz", dice=np.asarray(meter.dice), iou=np.asarray(meter.iou))

    # anomaly protocols (224 px)
    asplit = anomaly_split(df0, split, normal_label=6)
    ad_test = [i for i in asplit["test"] if i in set(test)]

    def ad(ids):
        y = [int(lab[i] != 6) for i in ids]
        s = [pred[i]["score"] for i in ids]
        # same dtypes as evaluate_anomaly: uint8 masks, maps stored as float16
        masks = np.stack([gt_mask(df.loc[i, "mask_path"], AD_SIZE) for i in ids]).astype(np.uint8)
        maps = np.stack([down(pred[i]["map"], AD_SIZE) for i in ids]).astype(np.float16).astype(np.float32)
        return anomaly_metrics(y, s, masks, maps), y, s

    class6, y6, s6 = ad(ad_test)
    offmask, _, _ = ad([i for i in ad_test if lab[i] != 6])
    np.savez_compressed(
        run / "test_scores.npz",
        idx=np.asarray(ad_test),
        scores=np.asarray(s6),
        labels=np.asarray(y6),
        classes=np.asarray([int(lab[i]) for i in ad_test]),
    )
    cls_met = classification_metrics(
        [int(lab[i]) for i in test], [pred[i]["cls"] for i in test], labels=[1, 2, 3, 4, 5, 6]
    )
    out = {
        "protocol": "Colab L4; YOLO26n-seg best.pt (val fitness); conf>=0.25 for masks/classes, conf>=0.001 for "
        "score maps; segmentation at 256 px on the 919 test images; anomaly metrics at 224 px",
        "segmentation": seg_met,
        "ad_offmask_task": offmask,
        "ad_class6_protocol": class6,
        "classification": cls_met,
        "n_test": len(test),
        "eval_seconds_incl_io": infer_s,
        "env": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.device_count() else "cpu",
        },
    }
    (run / "test_metrics.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("segmentation", "ad_offmask_task")}, indent=1))
    print(
        "classification macro-F1",
        cls_met["macro_f1"],
        "image AUROC (class-6 protocol)",
        class6["image_auroc"],
    )


if __name__ == "__main__":
    main()
