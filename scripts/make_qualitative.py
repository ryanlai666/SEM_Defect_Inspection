"""Qualitative figure (CPU only, from saved checkpoints): test images with ground truth, segmentation
predictions (U-Net-R18, DINOv2-S + decoder) and anomaly maps (PatchCore, Dinomaly).

    python scripts/make_qualitative.py            # -> docs/figures/qualitative.png

Images: the first test image of classes 1, 3 (a small and a large defect), 4 and 6 (normal).
Anomaly maps use a single-hue sequential ramp (reference palette blue 100 -> 700), clipped at the
99th percentile of the test-normal maps, so "bright" means "more anomalous than normal surfaces".
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from semdefect.data import CarinthiaS, load_index, load_or_make_split  # noqa: E402
from semdefect.models import Dinomaly, PatchCore, build_segmenter  # noqa: E402

OUT = ROOT / "outputs"
BLUES = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
GT_COLOR, PRED_COLOR = "#eb6834", "#1baf7a"


def pick_images(df, split) -> list[int]:
    lab = df.set_index("idx")["label"]
    test = split["test"]
    picks = []
    for c in (1, 4, 6):
        picks.append(next(i for i in test if lab[i] == c))
    # class 3: one small and one large defect (by mask area)
    c3 = [i for i in test if lab[i] == 3][:200]
    areas = []
    for i in c3:
        from semdefect.data import read_mask

        areas.append(read_mask(df.at[i, "mask_path"]).mean())
    order = np.argsort(areas)
    picks.insert(1, c3[order[len(order) // 10]])
    picks.insert(2, c3[order[-len(order) // 10]])
    return picks


def seg_predict(run: str, x: torch.Tensor) -> np.ndarray | None:
    ck = OUT / run / "checkpoints" / "best.ckpt"
    if not ck.exists():
        return None
    st = torch.load(ck, map_location="cpu", weights_only=False)
    cfg = st["cfg"]
    m = build_segmenter(cfg, pretrained=False)
    m.load_state_dict(st["model"])
    m.eval()
    size = int(cfg["data"]["img_size"])
    with torch.no_grad():
        xx = torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        p = torch.sigmoid(m(xx))
        p = torch.nn.functional.interpolate(p, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return (p[:, 0] >= 0.5).numpy()


def patchcore_maps(x: torch.Tensor) -> np.ndarray | None:
    ck = OUT / "patchcore_all_s0" / "checkpoints" / "memory_bank.ckpt"
    if not ck.exists():
        return None
    st = torch.load(ck, map_location="cpu", weights_only=False)
    m = dict(st["cfg"]["model"])
    m.pop("name")
    m.pop("pretrained", None)
    m["knn_backend"] = "torch"
    pc = PatchCore(pretrained=True, **m)
    pc.memory = st["memory"].float()
    pc.eval()
    xx = torch.nn.functional.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
    with torch.no_grad():
        _, maps = pc.predict(xx, out_size=x.shape[-1])
    return maps.numpy()


def dinomaly_maps(x: torch.Tensor) -> np.ndarray | None:
    ck = OUT / "dinomaly_s0" / "checkpoints" / "final.ckpt"
    if not ck.exists():
        return None
    st = torch.load(ck, map_location="cpu", weights_only=False)
    cfg = st["cfg"]
    m = dict(cfg["model"])
    m.pop("name")
    m.pop("pretrained", None)
    dm = Dinomaly(img_size=int(cfg["data"]["img_size"]), pretrained=False, **m)
    dm.load_state_dict(st["model"])
    dm.eval()
    size = int(cfg["data"]["img_size"])
    xx = torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    with torch.no_grad():
        _, maps = dm.predict(xx, out_size=x.shape[-1])
    return maps.numpy()


def main() -> None:
    torch.set_num_threads(6)
    df = load_index()
    split = load_or_make_split(df, seed=0)
    idx = pick_images(df, split)
    ds = CarinthiaS(df, idx, img_size=448, cache=False)
    items = [ds[i] for i in range(len(ds))]
    x = torch.stack([it["image"] for it in items])
    gray = [it["image01"][0].numpy() for it in items]
    gt = [it["mask"][0].numpy() > 0.5 for it in items]
    labels = [it["label"] for it in items]

    cols = [("image + ground truth", None)]
    preds = {}
    for run, title in (("unet_r18_s0", "U-Net-R18 mask"), ("dino_seg_s0", "DINOv2-S + decoder mask")):
        p = seg_predict(run, x)
        if p is not None:
            preds[title] = p
            cols.append((title, "seg"))
    for fn, title in ((patchcore_maps, "PatchCore map"), (dinomaly_maps, "Dinomaly map")):
        mp = fn(x)
        if mp is not None:
            preds[title] = mp
            cols.append((title, "map"))
    n, k = len(items), len(cols)
    fig, axes = plt.subplots(n, k, figsize=(2.2 * k, 2.2 * n), squeeze=False)
    for r in range(n):
        for c, (title, kind) in enumerate(cols):
            ax = axes[r, c]
            ax.imshow(gray[r], cmap="gray", vmin=0, vmax=1)
            if kind is None and gt[r].any():
                ax.contour(gt[r], levels=[0.5], colors=GT_COLOR, linewidths=1.2)
            elif kind == "seg":
                if gt[r].any():
                    ax.contour(gt[r], levels=[0.5], colors=GT_COLOR, linewidths=1.0)
                if preds[title][r].any():
                    ax.contour(preds[title][r], levels=[0.5], colors=PRED_COLOR, linewidths=1.2)
            elif kind == "map":
                normal_rows = [i for i in range(n) if labels[i] == 6]
                ref = preds[title][normal_rows].ravel() if normal_rows else preds[title].ravel()
                lo, hi = np.percentile(ref, 50), np.percentile(preds[title], 99.5)
                ax.imshow(
                    np.clip((preds[title][r] - lo) / max(hi - lo, 1e-6), 0, 1),
                    cmap=BLUES,
                    alpha=0.6,
                    vmin=0,
                    vmax=1,
                )
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=8)
            if c == 0:
                ax.set_ylabel(f"class {labels[r]}" + (" (normal)" if labels[r] == 6 else ""), fontsize=8)
    fig.suptitle(
        "Test-split examples — ground truth: orange contour · predicted mask: aqua contour · "
        "anomaly maps: blue (darker = more anomalous)",
        fontsize=9,
    )
    fig.tight_layout()
    out = ROOT / "docs" / "figures" / "qualitative.png"
    fig.savefig(out, dpi=130)
    print("wrote", out.relative_to(ROOT), "columns:", [c[0] for c in cols])


if __name__ == "__main__":
    main()
