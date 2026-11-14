"""Robustness sweep for trained classifiers / segmenters (anomaly models: `robustness.enabled` in
their own config).  Loads `outputs/<name>/checkpoints/best.ckpt` and evaluates the test split under
shot noise, blur and charging drift at increasing severity."""

from __future__ import annotations

import torch

from ..models import build_classifier, build_segmenter
from ..robustness import SEVERITIES
from ..tracking import Tracker
from ..utils import save_json
from .classify import evaluate_classifier
from .common import load_data, make_dataset, make_loader, output_dir, setup
from .segment import evaluate_segmenter


def run_robustness(cfg: dict) -> dict:
    device = setup(cfg)
    out = output_dir(cfg)
    df, split = load_data(cfg)
    task = cfg["task"]
    ckpt = torch.load(out / "checkpoints" / "best.ckpt", map_location=device, weights_only=False)
    classes = list(cfg["data"].get("classes", [1, 2, 3, 4, 5, 6]))
    if task == "classify":
        model = build_classifier(cfg["model"]["name"], len(classes), pretrained=False)
    elif task == "segment":
        model = build_segmenter(cfg, pretrained=False)
    else:
        raise ValueError("robustness sweep supports task in {classify, segment}")
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    # same evaluation batch size as the run's own test pass (train.eval_batch_size; top-level key as a fallback)
    ebs = int(cfg.get("train", {}).get("eval_batch_size", cfg.get("eval_batch_size", 32)))
    loader = make_loader(make_dataset(cfg, df, split["test"]), ebs, False, cfg)
    rcfg = cfg.get("robustness", {})
    results = {}
    tracker = Tracker(cfg, run_name=f"{cfg.get('name')}_robustness")
    for pert in ["clean"] + list(rcfg.get("perturbations", list(SEVERITIES))):
        for level in [0] if pert == "clean" else rcfg.get("levels", [1, 2, 3, 4, 5]):
            if task == "classify":
                met = evaluate_classifier(model, loader, device, classes, perturbation=pert, level=level)
                key = "macro_f1"
            else:
                met = evaluate_segmenter(model, loader, device, perturbation=pert, level=level)
                met.pop("_per_image", None)
                key = "dice_mean"
            results[f"{pert}_L{level}"] = met
            tracker.log_metrics({f"{pert}_L{level}_{key}": met[key]})
    save_json(results, out / "robustness.json")
    tracker.end()
    return results
