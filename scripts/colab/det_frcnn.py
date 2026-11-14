"""Detector benchmark (Colab L4): Faster R-CNN ResNet-50-FPN v2, torchvision COCO weights.

Model as in the author's PCB defect-detection project (models/faster_rcnn.py: fasterrcnn_resnet50_fpn_v2(weights='DEFAULT') with
a new FastRCNNPredictor head), here with 5 defect classes + background and the benchmark's common settings:
input resized to 480 (min_size = max_size = 480), random h/v flips, EPOCHS epochs, seed 0, batch 8,
SGD lr 0.01 / momentum 0.9 / wd 1e-4, 500-iteration linear warm-up then cosine, AMP; final checkpoint evaluated
(no model selection on val).  Resumable from the per-epoch checkpoint.

    python scripts/colab/det_frcnn.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from det_common import EPOCHS, IMGSZ, coco_metrics, env, gflops, n_params, save, timed
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]


class CocoDet(Dataset):
    def __init__(self, root: Path, part: str, train: bool):
        d = json.loads((root / "annotations" / f"instances_{part}.json").read_text())
        self.dir = root / "images" / part
        self.images = d["images"]
        self.anns: dict[int, list] = {im["id"]: [] for im in self.images}
        for a in d["annotations"]:
            self.anns[a["image_id"]].append(a)
        self.train = train

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, k: int):
        im = self.images[k]
        g = cv2.imread(str(self.dir / im["file_name"]), cv2.IMREAD_GRAYSCALE)
        x = torch.from_numpy(g).float().div_(255.0)[None].repeat(3, 1, 1)
        anns = self.anns[im["id"]]
        b = torch.tensor(
            [
                [a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                for a in anns
            ],
            dtype=torch.float32,
        ).reshape(-1, 4)
        lab = torch.tensor([a["category_id"] for a in anns], dtype=torch.int64)
        if self.train:
            h, w = g.shape
            if random.random() < 0.5:
                x = x.flip(-1)
                b[:, [0, 2]] = w - b[:, [2, 0]]
            if random.random() < 0.5:
                x = x.flip(-2)
                b[:, [1, 3]] = h - b[:, [3, 1]]
        return x, {"boxes": b, "labels": lab, "image_id": im["id"]}


def collate(batch):
    return tuple(zip(*batch))


def build(num_classes: int = 6, pretrained: bool = True):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    m = fasterrcnn_resnet50_fpn_v2(
        weights="DEFAULT" if pretrained else None,
        weights_backbone=None,
        min_size=IMGSZ,
        max_size=IMGSZ,
        box_score_thresh=0.001,
        box_detections_per_img=100,
    )
    m.roi_heads.box_predictor = FastRCNNPredictor(
        m.roi_heads.box_predictor.cls_score.in_features, num_classes
    )
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/det_data")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--run-dir", default=str(ROOT / "outputs" / "det_faster_rcnn_colab"))
    ap.add_argument("--max-iters", type=int, default=0, help="smoke test only")
    ap.add_argument("--skip-latency", action="store_true")
    ap.add_argument(
        "--no-pretrained", action="store_true", help="smoke test without downloading COCO weights"
    )
    a = ap.parse_args()
    data, run = Path(a.data), Path(a.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    model = build(pretrained=not a.no_pretrained).to(dev)
    ds = CocoDet(data, "train", True)
    g = torch.Generator().manual_seed(0)
    dl = DataLoader(
        ds,
        a.batch,
        shuffle=True,
        num_workers=a.workers,
        collate_fn=collate,
        drop_last=True,
        generator=g,
        persistent_workers=a.workers > 0,
    )
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=1e-4)
    total = a.epochs * len(dl)
    warm = min(500, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (
            (0.001 + 0.999 * s / warm)
            if s < warm
            else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))
        ),
    )
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    ckpt = run / "last.pt"
    epoch0 = 0
    if ckpt.exists():
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        epoch0 = st["epoch"]
    t_train = time.perf_counter()
    it = 0
    for ep in range(epoch0, a.epochs):
        model.train()
        t0, losses = time.perf_counter(), []
        for xs, ts in dl:
            xs = [x.to(dev, non_blocking=True) for x in xs]
            ts = [{"boxes": t["boxes"].to(dev), "labels": t["labels"].to(dev)} for t in ts]
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                loss = sum(model(xs, ts).values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
            it += 1
            if a.max_iters and it >= a.max_iters:
                break
        with open(run / "train_log.csv", "a") as log:
            log.write(f"{ep + 1},{np.mean(losses):.4f},{time.perf_counter() - t0:.1f}\n")
        torch.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": ep + 1,
            },
            ckpt,
        )
        if a.max_iters and it >= a.max_iters:
            break
    train_s = time.perf_counter() - t_train

    # ------------------------------------------------------------ test predictions -> COCO results
    model.eval()
    test = CocoDet(data, "test", False)
    tl = DataLoader(test, 8, shuffle=False, num_workers=a.workers, collate_fn=collate)
    results = []
    with torch.no_grad():
        for xs, ts in tl:
            outs = model([x.to(dev) for x in xs])
            for o, t in zip(outs, ts):
                for (x0, y0, x1, y1), c, s in zip(
                    o["boxes"].cpu().numpy(), o["labels"].cpu().numpy(), o["scores"].cpu().numpy()
                ):
                    results.append(
                        {
                            "image_id": int(t["image_id"]),
                            "category_id": int(c),
                            "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                            "score": float(s),
                        }
                    )
    res = {
        "model": "faster_rcnn_r50_fpn_v2",
        "weights": "torchvision FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1",
        "epochs": a.epochs,
        "imgsz": IMGSZ,
        "checkpoint": "last, no val selection",
        "train_seconds_this_session": train_s,
        "test": coco_metrics(data / "annotations" / "instances_test.json", results),
        "params": n_params(model),
        "env": env(),
    }
    x480 = torch.rand(3, IMGSZ, IMGSZ, device=dev)
    res["gflops_480"] = gflops(lambda inp: model([inp[0]]), x480[None])
    if not a.skip_latency and dev.type == "cuda":
        x477 = torch.rand(3, 477, 477, device=dev)

        def fwd():
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                return model([x477])

        res["latency_model_fp16_bs1"] = timed(fwd)
        res["latency_note"] = (
            "full forward incl. resize/normalise transform, RPN, RoI heads and NMS (AMP FP16)"
        )
    save(run, res)


if __name__ == "__main__":
    main()
