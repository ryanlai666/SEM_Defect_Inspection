"""Figures for REPORT.md, built only from run outputs (outputs/*/…json, outputs/bench/*.json).

    python scripts/make_report_figures.py        # writes docs/figures/*.png (skips figures without data)

Style: reference data-viz palette (first three categorical slots, validated all-pairs: blue, orange,
aqua; aqua < 3:1 contrast -> every figure is direct-labelled and every number is also in a REPORT
table), line style as secondary encoding, recessive axes, thin marks.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = ROOT / "docs" / "figures"
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "violet": "#7b5cd6"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "legend.frameon": False,
        "lines.linewidth": 2,
        "lines.markersize": 6,
    }
)


def jload(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def fig_fewshot(_unused=None) -> Path | None:
    """Two panels from outputs/results_summary.json (run scripts/collect_results.py first):
    6-class macro-F1 and macro-F1 over the 4 classes with >= 55 images."""
    summ = jload(OUT / "results_summary.json") or {}
    if not summ.get("fewshot"):
        return None
    ks = ["1", "5", "10", "20", "full"]
    xs = np.arange(len(ks))
    series = [
        ("DINOv2-S linear", "DINOv2-S linear probe (C chosen on val)", C["blue"], "-"),
        ("DINOv2-S knn1", "DINOv2-S cosine 1-NN", C["blue"], ":"),
        ("DINOv2-B linear", "DINOv2-B linear probe (C chosen on val)", C["orange"], "-"),
        ("ResNet-18 fine-tuned", "ResNet-18 fine-tuned", C["aqua"], "-"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    for ax, key, title in (
        (axes[0], "fewshot", "6 classes (macro-F1)"),
        (axes[1], "fewshot_4class", "classes 1, 3, 4, 6 (macro-F1)"),
    ):
        tab = summ.get(key, {})
        for name, label, col, ls in series:
            d = tab.get(name)
            if not d:
                continue
            m = np.array([d[k]["mean"] if d.get(k) else np.nan for k in ks])
            s = np.array([d[k]["std"] if d.get(k) else np.nan for k in ks])
            ok = np.isfinite(m)
            ax.plot(xs[ok], m[ok], ls, color=col, marker="o", label=label)
            ax.fill_between(xs[ok], (m - s)[ok], (m + s)[ok], color=col, alpha=0.10, linewidth=0)
        ax.set_xticks(xs, [f"k={k}" if k != "full" else "full\ntrain" for k in ks])
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.set_xlabel("labelled images per class")
    axes[0].set_ylabel("macro-F1 on the test split (mean ± std over seeds)")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Few-shot SEM defect classification on Carinthia-S", fontsize=10, fontweight="bold")
    fig.tight_layout()
    p = FIG / "fewshot_macro_f1.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_ad_tasks(_unused=None) -> Path | None:
    """Two panels from outputs/results_summary.json: the off-mask pixel task (lead) and the class-6 protocol,
    each with the training-free high-pass baseline (neutral grey) next to the learned detectors."""
    summ = (jload(OUT / "results_summary.json") or {}).get("anomaly", {})
    if not summ:
        return None
    grey = "#9a9890"
    panels = [
        (
            "Anomaly localisation on defect images — off-mask normals (lead task)",
            "Off-mask pixel task (874 test defect images)",
            [
                ("High-pass energy (training-free baseline)", "high-pass baseline", grey),
                ("PatchCore WRN-50 + off-mask normals", "PatchCore", C["blue"]),
                ("AnomalyDINO ViT-S/14 + off-mask normals", "AnomalyDINO", C["orange"]),
                (
                    "Dinomaly ViT-S/14-reg, loss masked to off-mask patches (Colab L4)",
                    "Dinomaly (Colab L4)",
                    C["aqua"],
                ),
                (
                    "YOLO26n-seg score map — SUPERVISED reference, trained on the masks (Colab L4)",
                    "YOLO26n-seg (supervised reference)",
                    C["violet"],
                ),
            ],
            [("pixel_ap", "pixel AP"), ("aupro_05", "AUPRO@5%")],
        ),
        (
            "Class-6 protocol (normal = 'no visible defect' images) — effectively texture detection",
            "Class-6 protocol (45 normal vs 874 defect images)",
            [
                ("High-pass energy (training-free baseline)", "high-pass baseline", grey),
                ("PatchCore WRN-50, all 136 normals", "PatchCore", C["blue"]),
                ("AnomalyDINO ViT-S/14, all 136 normals", "AnomalyDINO", C["orange"]),
                (
                    "YOLO26n-seg max confidence — SUPERVISED reference (Colab L4)",
                    "YOLO26n-seg (supervised reference)",
                    C["violet"],
                ),
            ],
            [("image_auroc", "image AUROC"), ("pixel_ap", "pixel AP"), ("aupro_05", "AUPRO@5%")],
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, (prefix, title, methods, metrics) in zip(axes, panels):
        methods = [m for m in methods if f"{prefix} :: {m[0]}" in summ]  # Colab rows only once they exist
        xs = np.arange(len(metrics))
        w = 0.8 / len(methods)
        for j, (key, lab, col) in enumerate(methods):
            vals, errs = [], []
            for mk, _ in metrics:
                st = (summ.get(f"{prefix} :: {key}") or {}).get(mk)
                vals.append(st["mean"] if st else np.nan)
                errs.append(st["std"] if st else 0.0)
            pos = xs - 0.4 + w / 2 + j * w
            ax.bar(pos, vals, width=w - 0.03, color=col, yerr=errs, capsize=2, label=lab)
            for x_, v in zip(pos, vals):
                if np.isfinite(v):
                    ax.annotate(
                        f"{v:.2f}",
                        (x_, v),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        fontsize=7,
                        color=INK2,
                    )
        ax.set_xticks(xs, [m[1] for m in metrics])
        ax.set_ylim(0, 1.08)
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("score on the test split")
    seen: dict = {}
    for ax in axes:  # union of both panels' entries, colour fixed per method
        for h, lab in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(lab, h)
    handles, labels = list(seen.values()), list(seen)
    fig.legend(
        handles, labels, fontsize=8, loc="lower center", ncol=len(labels)
    )  # below the panels, off the bars
    fig.suptitle(
        "Anomaly detection on Carinthia-S vs a training-free high-pass baseline",
        fontsize=10,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    p = FIG / "ad_tasks.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def robustness_series() -> list[tuple[str, str, dict]]:
    """[(panel title, metric name, {perturbation: [(level, value)]})]"""
    panels = []
    r = jload(OUT / "highpass_split0" / "test_metrics.json")  # AD robustness: training-free baseline only
    if r and r.get("robustness"):
        for metric, mname in (("image_auroc", "image AUROC"), ("pixel_ap", "pixel AP")):
            d: dict = {}
            for key, v in r["robustness"].items():
                pert, lvl = key.rsplit("_L", 1)
                d.setdefault(pert, []).append((int(lvl), v.get(metric)))
            for pert in d:
                d[pert] = [(0, r["test"].get(metric))] + sorted(d[pert])
            panels.append((f"High-pass AD baseline — {mname}", mname, d))
    for run, title, metric, mname in (
        ("resnet18_full_s0", "ResNet-18 classifier", "macro_f1", "macro-F1"),
        ("unet_r18_s0", "U-Net-R18 segmentation", "dice_mean", "Dice"),
        ("dino_seg_s0", "DINOv2-S segmentation", "dice_mean", "Dice"),
    ):
        r = jload(OUT / run / "robustness.json")
        if r:
            d = {}
            clean = r.get("clean_L0", {}).get(metric)
            for key, v in r.items():
                pert, lvl = key.rsplit("_L", 1)
                if pert != "clean":
                    d.setdefault(pert, []).append((int(lvl), v.get(metric)))
            for pert in d:
                d[pert] = [(0, clean)] + sorted(d[pert])
            panels.append((f"{title} — {mname}", mname, d))
    return panels


def fig_robustness(panels) -> Path | None:
    if not panels:
        return None
    n = len(panels)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 2.9 * rows), squeeze=False)
    names = {
        "shot_noise": ("shot noise", C["blue"], "-"),
        "blur": ("beam blur", C["orange"], "--"),
        "charging": ("charging drift", C["aqua"], ":"),
    }
    for ax, (title, mname, d) in zip(axes.flat, panels):
        for pert, (lab, col, ls) in names.items():
            if pert not in d:
                continue
            lv = np.array([a for a, _ in d[pert]], float)
            val = np.array([np.nan if b is None else b for _, b in d[pert]], float)
            ax.plot(lv, val, ls, color=col, marker="o", label=lab)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("severity level (0 = clean)")
        ax.set_ylabel(mname)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Robustness to SEM-like degradations (test split)", fontsize=10, fontweight="bold")
    fig.tight_layout()
    p = FIG / "robustness.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# ----------------------------------------------------------------------------------- kernels + deployment
def fig_knn_speedup() -> Path | None:
    """Each custom kernel vs its precision-matched PyTorch baseline (no FP16-vs-FP32 speed-ups)."""
    k = jload(OUT / "bench" / "kernels_v2.json") or jload(OUT / "bench" / "kernels.json")
    if not k:
        return None
    nice = {
        "patchcore_1img": "PatchCore, 1 image",
        "patchcore_8img": "PatchCore, 8 images",
        "anomalydino_1img": "AnomalyDINO, 1 image",
    }
    cases: dict = {}
    for r in k["knn"]:
        key = f"{nice.get(r['case'], r['case'])}\nbank N={r['N'] // 1000}k, D={r['D']}"
        cases.setdefault(key, {})[r["backend"].split("(")[0]] = r["p50_ms"]
    pairs = [
        ("nvcc_ext_fused", "torch_cdist_topk", "FP32 tiled kernel vs PyTorch FP32 (cdist + topk)", C["blue"])
    ]
    if any("torch_fp16_mm_topk" in d for d in cases.values()):
        pairs.append(
            (
                "nvcc_wmma_fp16_top1",
                "torch_fp16_mm_topk",
                "tensor-core FP16 kernel vs PyTorch FP16 (cuBLAS + topk)",
                C["orange"],
            )
        )
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ys = np.arange(len(cases))
    h = 0.8 / len(pairs)
    for j, (ours, base, lab, col) in enumerate(pairs):
        sp = [d.get(base, np.nan) / d.get(ours, np.nan) for d in cases.values()]
        pos = ys - 0.4 + h / 2 + j * h
        ax.barh(pos, sp, height=h - 0.04, color=col, label=lab)
        for y, v in zip(pos, sp):
            if np.isfinite(v):
                ax.annotate(
                    f"{v:.2f}×",
                    (v, y),
                    xytext=(4, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=8,
                    color=INK2,
                )
    ax.axvline(1.0, color=INK2, linewidth=1)
    ax.set_yticks(ys, list(cases))
    ax.invert_yaxis()
    ax.set_xlim(0, ax.get_xlim()[1] * 1.15)
    ax.set_xlabel("speed-up vs the precision-matched PyTorch baseline (p50; > 1 = faster)")
    ax.set_title("kNN search over a memory bank — custom CUDA kernels (RTX 3070 Laptop)", loc="left")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    fig.tight_layout()
    p = FIG / "knn_speedup.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_pipeline_latency() -> Path | None:
    g = jload(OUT / "bench" / "engine_full.json") or jload(OUT / "bench" / "engine.json")
    if not g:
        return None
    e2e = g["latency_end_to_end"]
    order = [
        ("baseline_torch_fp32_bs8", "PyTorch FP32 (torch preprocessing + cdist kNN)", C["blue"]),
        ("baseline_torch_amp_fp16_bs8", "PyTorch AMP FP16 (+ cuBLAS FP16 kNN)", C["blue"]),
        ("kernels+trt_fp16_bs8", "fused preprocess + TensorRT FP16 + FP32 kNN kernel", C["orange"]),
        ("kernels(wmma_knn)+trt_fp16_bs8", "fused preprocess + TensorRT FP16 + tensor-core kNN", C["orange"]),
    ]
    rows = [
        (lab, e2e[k]["p50_ms"], e2e[k].get("p50_std_ms", 0.0), e2e[k]["img_per_s"], col)
        for k, lab, col in order
        if k in e2e
    ]
    runner = jload(OUT / "bench" / "runner_full_fp16_b8_s1_knnfp16.json") or jload(
        OUT / "bench" / "runner_fp16_b8_s1_knnfp16.json"
    )
    if runner:
        rows.append(
            (
                "C++ runner (incl. pinned H2D/D2H), FP16 + tensor-core kNN",
                runner["latency_ms_p50"],
                runner.get("latency_ms_p50_std", 0.0),
                runner["throughput_img_per_s"],
                C["orange"],
            )
        )
    fig, ax = plt.subplots(figsize=(9.5, 3.2))
    ys = np.arange(len(rows))
    ax.barh(
        ys, [r[1] for r in rows], xerr=[r[2] for r in rows], height=0.6, color=[r[4] for r in rows], capsize=2
    )
    for y, (_, v, _sd, ips, _c) in zip(ys, rows):
        ax.annotate(
            f"{v:.1f} ms  ({ips:.0f} img/s)",
            (v, y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK2,
        )
    ax.set_yticks(ys, [r[0] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, max(r[1] for r in rows) * 1.45)
    ax.set_xlabel("p50 latency per batch of 8 raw 480×480 images (ms, lower is better)")
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color=rows[0][4], label="PyTorch"),
            Patch(color=rows[-1][4], label="ours (CUDA kernels / TensorRT)"),
        ],
        fontsize=8,
        loc="lower right",
    )
    ax.set_title("PatchCore: raw SEM image → anomaly score, RTX 3070 Laptop", loc="left")
    fig.tight_layout()
    p = FIG / "pipeline_latency.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_detectors() -> Path | None:
    """Detector benchmark (Colab L4): mAP50-95 vs batch-1 FP16 latency, one labelled point per model."""
    models = [
        ("det_yolo26n_colab", "YOLO26n", C["blue"]),
        ("det_sme_yolo_colab", '"SME-YOLO" (YOLOv11n)', C["orange"]),
        ("det_rtdetr_l_colab", "RT-DETR-L", C["aqua"]),
        ("det_faster_rcnn_colab", "Faster R-CNN R50-FPN v2", C["violet"]),
        ("det_deimv2_l_colab", "DEIMv2-L", "#c9407a"),
    ]
    pts = []
    for d, lab, col in models:
        r = jload(OUT / d / "det_metrics.json")
        lat = (r or {}).get("latency_model_fp16_bs1", {}).get("p50_ms")
        if r and lat:
            pts.append((lab, col, lat, r["test"]["mAP50_95"], r["test"]["mAP50"], r["params"] / 1e6))
    if not pts:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for lab, col, x, y, y50, prm in pts:
        ax.scatter([x], [y], s=60, color=col, edgecolor=SURFACE, linewidth=2, zorder=3, label=lab)
        ax.annotate(
            f"{lab}\nmAP50 {y50:.3f} · {prm:.1f} M params",
            (x, y),
            xytext=(8, -4),
            textcoords="offset points",
            fontsize=7.5,
            color=INK2,
            va="top",
        )
    ax.set_xscale("log")
    ax.set_xlabel("L4 latency, batch 1, FP16, model forward (ms, log scale)")
    ax.set_ylabel("test mAP50-95 (COCO, 5 defect classes)")
    ax.set_title("Detector benchmark on Carinthia-S (35 epochs, 480 px, seed 0)", loc="left")
    xs = [p[2] for p in pts]
    ax.set_xlim(min(xs) / 1.6, max(xs) * 4)
    ys = [p[3] for p in pts]
    ax.set_ylim(min(ys) - 0.06, max(ys) + 0.03)  # room for the labels below the points
    fig.tight_layout()
    p = FIG / "detectors.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_hero(rows: tuple[int, ...] = (0, 2, 4)) -> Path | None:
    """README cover figure: the header and three panel rows (class 1, the large class-3 defect, class 6
    normal) cropped from docs/figures/qualitative.png (written by scripts/make_qualitative.py)."""
    from PIL import Image

    src = FIG / "qualitative.png"
    if not src.exists():
        return None
    img = np.asarray(Image.open(src).convert("RGB"))
    # panel borders: 1-px dark horizontal lines across the first panel, one (top, bottom) pair per row
    lines = np.flatnonzero((img[:, 50:290].mean(2) < 60).mean(1) > 0.9)
    tops, bottoms = lines[0::2], lines[1::2]
    if len(tops) != len(bottoms) or max(rows) >= len(tops):
        return None
    pad_top, pad_bottom = 8, 10
    parts = [img[: tops[0] - pad_top]]  # figure title + column titles
    parts += [img[tops[r] - pad_top : min(bottoms[r] + pad_bottom, img.shape[0])] for r in rows]
    p = FIG / "hero.png"
    Image.fromarray(np.concatenate(parts, axis=0)).save(p, optimize=True)
    return p


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows GBK consoles
    FIG.mkdir(parents=True, exist_ok=True)
    made = [
        fig_fewshot(),
        fig_ad_tasks(),
        fig_robustness(robustness_series()),
        fig_knn_speedup(),
        fig_pipeline_latency(),
        fig_detectors(),
        fig_hero(),
    ]
    for p in made:
        if p:
            print("wrote", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
