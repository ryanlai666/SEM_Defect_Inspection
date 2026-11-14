import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from semdefect.metrics import (
    SegmentationMeter,
    aupro,
    binned_auroc,
    binned_average_precision,
    classification_metrics,
)
from semdefect.robustness import SEVERITIES, blur, charging_drift, make_perturbation, shot_noise


def test_binned_auroc_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.random(200_000) < 0.05
    s = rng.normal(0, 1, y.size) + 1.2 * y
    assert abs(binned_auroc(y, s) - roc_auc_score(y, s)) < 1e-4


def test_aupro_perfect_and_random():
    rng = np.random.default_rng(1)
    masks = np.zeros((6, 64, 64), bool)
    for i in range(6):
        masks[i, 10 + i : 20 + i, 30:40] = True
    perfect = masks.astype(np.float32) + 0.01 * rng.random(masks.shape)
    assert aupro(masks, perfect) > 0.99
    rand = rng.random(masks.shape).astype(np.float32)
    assert 0.05 < aupro(masks, rand) < 0.25  # random scores -> PRO ~ FPR -> AUPRO ~ 0.15


def test_segmentation_meter_handles_empty_masks():
    m = SegmentationMeter()
    gt = np.zeros((2, 8, 8))
    gt[0, 2:4, 2:4] = 1
    pred = gt.copy()
    pred[1, 0, 0] = 1  # one false-positive pixel on the empty image
    m.update(pred, gt)
    r = m.compute()
    assert r["iou_mean"] == 1.0 and r["empty_image_fp_rate"] == 1.0 and r["n_empty_images"] == 1


def test_classification_metrics_macro_f1():
    r = classification_metrics([1, 1, 2, 3], [1, 2, 2, 3], labels=[1, 2, 3])
    assert abs(r["accuracy"] - 0.75) < 1e-9
    assert abs(r["macro_f1"] - np.mean([2 / 3, 2 / 3, 1.0])) < 1e-9


def test_shot_noise_unbiased_and_dose_dependent():
    x = torch.full((1, 1, 128, 128), 0.4)
    g = torch.Generator().manual_seed(0)
    lo = shot_noise(x, 10.0, g)
    hi = shot_noise(x, 1000.0, g)
    assert abs(lo.mean().item() - 0.4) < 0.02
    assert lo.std() > 3 * hi.std()


def test_blur_and_charging_properties():
    g = torch.Generator().manual_seed(0)
    x = torch.rand(2, 1, 64, 64)
    b = blur(x, 2.0)
    assert b.shape == x.shape and b.std() < x.std() and abs(b.mean() - x.mean()) < 0.01
    c1 = charging_drift(x, 0.2, torch.Generator().manual_seed(5))
    c2 = charging_drift(x, 0.2, torch.Generator().manual_seed(5))
    assert torch.equal(c1, c2) and c1.min() >= 0 and c1.max() <= 1
    d = (c1 - x).abs().mean()
    assert 0.0 < d < 0.3
    assert make_perturbation("blur", 0)(x, g) is x
    for name, levels in SEVERITIES.items():
        assert len(levels) == 5, name


def test_binned_average_precision_matches_sklearn():
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(3)
    y = rng.random(100_000) < 0.02
    s = rng.normal(0, 1, y.size) + 2.0 * y
    assert abs(binned_average_precision(y, s) - average_precision_score(y, s)) < 2e-3
