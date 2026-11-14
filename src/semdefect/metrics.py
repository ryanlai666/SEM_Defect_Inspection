"""Metrics: classification (acc, macro-F1), segmentation (IoU, Dice), anomaly detection
(image AUROC, pixel AUROC, AUPRO@30%FPR).

Pixel-level metrics avoid materialising sklearn ROC curves over tens of millions of pixels:
pixel AUROC uses a 2^16-bin histogram (abs. error < 1e-4 vs exact, see tests) and AUPRO evaluates
per-region overlap at thresholds placed at quantiles of the normal-pixel score distribution, so the
FPR axis in [0, 0.3] is sampled uniformly.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import ndimage
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

# ----------------------------------------------------------------------------- classification


def classification_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[int] | None = None
) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = sorted(set(y_true.tolist())) if labels is None else list(labels)
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(np.mean(per_class)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class_f1": {str(c): float(f) for c, f in zip(labels, per_class)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": [int(c) for c in labels],
    }


# ----------------------------------------------------------------------------- segmentation


class SegmentationMeter:
    """Accumulates per-image and dataset-level (micro) IoU / Dice for a binary foreground.

    Images with an empty ground-truth mask (class 6) are excluded from the per-image mean IoU/Dice
    and instead counted for an image-level false-positive rate (any predicted pixels)."""

    def __init__(self, threshold: float = 0.5, min_fp_pixels: int = 1):
        self.t = threshold
        self.min_fp = min_fp_pixels
        self.tp = self.fp = self.fn = 0
        self.iou: list[float] = []
        self.dice: list[float] = []
        self.empty_total = 0
        self.empty_fp = 0

    def update(self, prob: np.ndarray, gt: np.ndarray) -> None:
        """prob, gt: (B,H,W) or (B,1,H,W) arrays."""
        prob = np.asarray(prob).reshape(len(prob), -1)
        gt = np.asarray(gt).reshape(len(gt), -1) > 0.5
        pred = prob >= self.t
        for p, g in zip(pred, gt):
            tp = int((p & g).sum())
            fp = int((p & ~g).sum())
            fn = int((~p & g).sum())
            self.tp += tp
            self.fp += fp
            self.fn += fn
            if g.sum() == 0:
                self.empty_total += 1
                self.empty_fp += int(p.sum() >= self.min_fp)
            else:
                self.iou.append(tp / max(tp + fp + fn, 1))
                self.dice.append(2 * tp / max(2 * tp + fp + fn, 1))

    def compute(self) -> dict:
        return {
            "iou_mean": float(np.mean(self.iou)) if self.iou else float("nan"),
            "dice_mean": float(np.mean(self.dice)) if self.dice else float("nan"),
            "iou_micro": self.tp / max(self.tp + self.fp + self.fn, 1),
            "dice_micro": 2 * self.tp / max(2 * self.tp + self.fp + self.fn, 1),
            "empty_image_fp_rate": self.empty_fp / self.empty_total if self.empty_total else float("nan"),
            "n_defect_images": len(self.iou),
            "n_empty_images": self.empty_total,
        }


# ----------------------------------------------------------------------------- anomaly detection


def image_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    labels = np.asarray(labels)
    if labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, np.asarray(scores)))


def binned_auroc(labels: np.ndarray, scores: np.ndarray, n_bins: int = 65536) -> float:
    """AUROC via score histograms (memory O(n_bins)); ties inside a bin count as 0.5."""
    labels = np.asarray(labels).ravel().astype(bool)
    scores = np.asarray(scores, dtype=np.float64).ravel()
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    lo, hi = scores.min(), scores.max()
    if hi <= lo:
        return 0.5
    idx = np.clip(((scores - lo) / (hi - lo) * (n_bins - 1)).astype(np.int64), 0, n_bins - 1)
    pos = np.bincount(idx[labels], minlength=n_bins).astype(np.float64)
    neg = np.bincount(idx[~labels], minlength=n_bins).astype(np.float64)
    neg_below = np.cumsum(neg) - neg
    auc = (pos * (neg_below + 0.5 * neg)).sum() / (n_pos * n_neg)
    return float(auc)


def binned_average_precision(labels: np.ndarray, scores: np.ndarray, n_bins: int = 65536) -> float:
    """Pixel average precision (area under precision-recall, step interpolation) via score histograms.
    Unlike pixel AUROC it is not inflated by the huge number of easy background pixels."""
    labels = np.asarray(labels).ravel().astype(bool)
    scores = np.asarray(scores, dtype=np.float64).ravel()
    n_pos = int(labels.sum())
    if n_pos == 0 or n_pos == labels.size:
        return float("nan")
    lo, hi = scores.min(), scores.max()
    if hi <= lo:
        return n_pos / labels.size
    idx = np.clip(((scores - lo) / (hi - lo) * (n_bins - 1)).astype(np.int64), 0, n_bins - 1)
    pos = np.bincount(idx[labels], minlength=n_bins)[::-1].astype(np.float64)  # descending thresholds
    neg = np.bincount(idx[~labels], minlength=n_bins)[::-1].astype(np.float64)
    tp, fp = np.cumsum(pos), np.cumsum(neg)
    keep = pos > 0
    precision = tp[keep] / (tp[keep] + fp[keep])
    return float(np.sum(precision * pos[keep]) / n_pos)


def aupro(masks: np.ndarray, maps: np.ndarray, max_fpr: float = 0.3, n_thresholds: int = 300) -> float:
    """Area under the per-region-overlap curve up to `max_fpr`, normalised to [0,1].

    masks: (N,H,W) binary ground truth; maps: (N,H,W) anomaly scores (higher = more anomalous).
    Follows the MVTec AD definition (Bergmann et al.): connected components of the GT are regions;
    PRO(t) = mean over regions of the fraction of region pixels with score > t; FPR(t) over all
    GT-negative pixels.
    """
    masks = np.asarray(masks).astype(bool)
    maps = np.asarray(maps, dtype=np.float32)
    region_scores: list[np.ndarray] = []
    for m, a in zip(masks, maps):
        if not m.any():
            continue
        lab, n = ndimage.label(m, structure=np.ones((3, 3)))
        for r in range(1, n + 1):
            region_scores.append(a[lab == r])
    if not region_scores:
        return float("nan")
    normal = np.sort(maps[~masks].ravel())
    if normal.size == 0:
        return float("nan")
    # thresholds such that FPR runs uniformly over [0, max_fpr]
    fprs_target = np.linspace(0.0, max_fpr, n_thresholds)
    q = np.clip(1.0 - fprs_target, 0.0, 1.0)
    thr = np.quantile(normal, q)
    fpr = 1.0 - np.searchsorted(normal, thr, side="right") / normal.size
    pro = np.zeros_like(thr)
    for rs in region_scores:
        s = np.sort(rs)
        pro += 1.0 - np.searchsorted(s, thr, side="right") / s.size
    pro /= len(region_scores)
    order = np.argsort(fpr)
    fpr, pro = fpr[order], pro[order]
    keep = fpr <= max_fpr
    fpr, pro = fpr[keep], pro[keep]
    if fpr[-1] < max_fpr:  # extend to max_fpr by interpolation on the last segment
        fpr = np.append(fpr, max_fpr)
        pro = np.append(pro, pro[-1])
    return float(np.trapezoid(pro, fpr) / max_fpr)


def anomaly_metrics(
    image_labels: Sequence[int],
    image_scores: Sequence[float],
    masks: np.ndarray | None = None,
    maps: np.ndarray | None = None,
) -> dict:
    out = {"image_auroc": image_auroc(image_labels, image_scores)}
    if masks is not None and maps is not None:
        out["pixel_auroc"] = binned_auroc(masks, maps)
        out["aupro_30"] = aupro(masks, maps, 0.3)
        out["aupro_05"] = aupro(masks, maps, 0.05)  # strict: low false-positive regime
        out["pixel_ap"] = binned_average_precision(masks, maps)
    return out
