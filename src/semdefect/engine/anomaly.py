"""Anomaly detection runner: PatchCore, AnomalyDINO, EfficientAD, Dinomaly.

Protocol (see docs/dataset_analysis.md): "normal" = class 6 ("no visible defect") images;
train on train-split normals only (optionally k of them: few-shot AD), evaluate on the test split:
image AUROC (normal vs classes 1-5), per-defect-class image AUROC, pixel AUROC and AUPRO@0.3 at a
common `eval_size`.  Optional: augment the normal pool of memory-bank models with patches from
train-split *defect* images that lie > margin px from the expert mask ("off-mask" proxy normals).
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ..checkpoint import CheckpointManager
from ..data import anomaly_split
from ..data.carinthia import to_model_input
from ..metrics import anomaly_metrics, image_auroc
from ..models import AnomalyDINO, Dinomaly, EfficientAD, PatchCore, count_params
from ..models.anomaly import offmask_keep
from ..models.baselines import HighPassEnergy
from ..models.efficientad import ae_augment
from ..robustness import SEVERITIES
from ..tracking import Tracker
from ..utils import Throughput, save_json
from .common import (
    SmokeRecorder,
    infinite,
    is_smoke,
    load_data,
    make_dataset,
    make_loader,
    output_dir,
    perturbed_batches,
    setup,
    subsample,
)

MEMORY_BANK = ("patchcore", "anomalydino", "highpass")  # fit-without-gradient models


def build_anomaly_model(cfg: dict) -> torch.nn.Module:
    m = dict(cfg["model"])
    name = m.pop("name")
    pre = m.pop("pretrained", True)
    if name == "patchcore":
        return PatchCore(pretrained=pre, **m)
    if name == "anomalydino":
        return AnomalyDINO(img_size=int(cfg["data"]["img_size"]), pretrained=pre, **m)
    if name == "efficientad":
        return EfficientAD(pretrained_teacher=pre, **m)
    if name == "dinomaly":
        return Dinomaly(img_size=int(cfg["data"]["img_size"]), pretrained=pre, **m)
    if name == "highpass":
        return HighPassEnergy(**m)
    raise ValueError(name)


@torch.no_grad()
def evaluate_anomaly(
    model,
    loader,
    device,
    eval_size: int,
    amp: bool = True,
    perturbation: str = "clean",
    level: int = 0,
    keep_maps: bool = True,
    tput: Throughput | None = None,
) -> dict:
    model.eval()
    scores, labels, classes, maps, masks = [], [], [], [], []
    for batch in perturbed_batches(loader, device, perturbation, level):
        if tput:
            tput.start()
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            s, m = model.predict(batch["image"], out_size=eval_size)
        if tput:
            tput.stop(batch["image"].shape[0])
        scores.append(s.float().cpu())
        labels.append(batch["is_anomalous"])
        classes.append(batch["label"])
        if keep_maps:
            maps.append(m.float().cpu().half())
            masks.append(batch["mask"][:, 0].to(torch.uint8))
    s = torch.cat(scores).numpy()
    y = torch.cat(labels).numpy()
    c = torch.cat(classes).numpy()
    met = anomaly_metrics(
        y,
        s,
        torch.cat(masks).numpy() if keep_maps else None,
        torch.cat(maps).float().numpy() if keep_maps else None,
    )
    for cls in sorted(set(c.tolist()) - {6}):
        sel = (c == cls) | (y == 0)
        if (c == cls).sum() > 0 and (y[sel] == 0).sum() > 0:
            met[f"image_auroc_class{cls}"] = image_auroc(y[sel], s[sel])
    met["n_normal"] = int((y == 0).sum())
    met["n_anomalous"] = int((y == 1).sum())
    met["_arrays"] = {"scores": s, "labels": y, "classes": c}  # popped by the caller (per-image, for CIs)
    return met


def _train_efficientad(
    model: EfficientAD, cfg, train_loader, val_normal_loader, device, out, tracker, smoke, ckpt
):
    tcfg = cfg["train"]
    params = list(model.student.parameters()) + list(model.ae.parameters())
    opt = torch.optim.Adam(
        params, lr=float(tcfg.get("lr", 1e-4)), weight_decay=float(tcfg.get("weight_decay", 1e-5))
    )
    total = int(cfg["smoke"]["max_steps"]) if smoke else int(tcfg.get("steps", 70000))
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, int(0.95 * total)), gamma=0.1)
    step = 0
    state = ckpt.load_last() if cfg.get("resume", True) and not smoke else None
    if state:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        step = state["step"]
    else:
        model.fit_teacher_stats(train_loader, device)
    tput = Throughput(warmup=2)
    it = infinite(train_loader)
    model.train()

    def snapshot():
        return {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "step": step,
            "mlflow_run_id": tracker.run_id,
            "cfg": cfg,
        }

    while step < total:
        batch = next(it)
        tput.start()
        x = batch["image"].to(device, non_blocking=True)
        x01 = batch["image01"].to(device, non_blocking=True).repeat(1, 3, 1, 1)
        x_aug = (
            ae_augment(x01) - torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        ) / torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        losses = model.losses(x, x_aug)
        if not torch.isfinite(losses["loss"]):
            raise FloatingPointError(f"non-finite EfficientAD loss at step {step}")
        opt.zero_grad(set_to_none=True)
        losses["loss"].backward()
        opt.step()
        sched.step()
        tput.stop(x.shape[0])
        step += 1
        if step % 50 == 0:
            tracker.log_metrics({k: v.item() for k, v in losses.items()}, step)
        ckpt.maybe_save(snapshot)
    ckpt.save(snapshot())
    model.eval()
    model.fit_quantiles(val_normal_loader, device)
    return tput


def _train_dinomaly(model: Dinomaly, cfg, train_loader, device, out, tracker, smoke, ckpt):
    tcfg = cfg["train"]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        params,
        lr=float(tcfg.get("lr", 2e-3)),
        weight_decay=float(tcfg.get("weight_decay", 1e-4)),
        betas=(0.9, 0.999),
    )
    total = int(cfg["smoke"]["max_steps"]) if smoke else int(tcfg.get("steps", 10000))
    warm = min(int(tcfg.get("warmup_steps", 100)), max(1, total // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (
            min(1.0, (s + 1) / warm) * (0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * min(1.0, s / max(1, total)))))
        ),
    )
    amp = device.type == "cuda" and cfg.get("amp", True)
    offmask = int(tcfg.get("offmask_images", 0)) > 0
    margin = int(tcfg.get("margin_px", 16))
    side = model.encoder.grid
    step = 0
    state = ckpt.load_last() if cfg.get("resume", True) and not smoke else None
    if state:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        step = state["step"]
    tput = Throughput(warmup=2)
    it = infinite(train_loader)
    model.train()

    def snapshot():
        return {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "step": step,
            "mlflow_run_id": tracker.run_id,
            "cfg": cfg,
        }

    while step < total:
        batch = next(it)
        tput.start()
        x = batch["image"].to(device, non_blocking=True)
        keep = offmask_keep(batch["mask"].to(device), side, margin) if offmask else None
        with torch.autocast(device.type, dtype=torch.bfloat16 if amp else torch.float32, enabled=amp):
            loss = model.loss(x, keep=keep)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite Dinomaly loss at step {step}; lower train.lr or disable amp"
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 0.1)
        opt.step()
        sched.step()
        tput.stop(x.shape[0])
        step += 1
        if step % 50 == 0:
            tracker.log_metrics({"loss": loss.item(), "lr": sched.get_last_lr()[0]}, step)
        ckpt.maybe_save(snapshot)
    ckpt.save(snapshot())
    return tput


def run_anomaly(cfg: dict) -> dict:
    device = setup(cfg)
    out = output_dir(cfg)
    df, split = load_data(cfg)
    lab = df.set_index("idx")["label"]
    smoke = is_smoke(cfg)
    rec = SmokeRecorder(cfg, device) if smoke else None
    name = cfg["model"]["name"]
    eval_size = int(cfg.get("eval", {}).get("eval_size", 224))
    tcfg = cfg.get("train", {})
    asplit = anomaly_split(df, split, normal_label=6)
    train_normal = asplit["train_normal"]
    k_normal = tcfg.get("k_normal")
    if k_normal:
        train_normal = subsample(train_normal, int(k_normal), seed=int(tcfg.get("k_seed", 0)))
    test_idx = asplit["test"]
    if cfg.get("eval", {}).get("defect_images_only", False):
        # pixel-level localisation task: only defect images; normal pixels = off-mask pixels of defect images
        test_idx = [i for i in test_idx if lab[i] != 6]
    val_normal = [i for i in asplit["val"] if lab[i] == 6]
    if smoke:
        n = int(cfg["smoke"].get("n_images", 50))
        train_normal = train_normal[: max(4, n // 3)]
        test_idx = subsample(test_idx, n - len(train_normal), labels=[lab[i] for i in test_idx])
        val_normal = val_normal[:8]

    bs = int(tcfg.get("batch_size", 8))
    model = build_anomaly_model(cfg).to(device)
    ckpt = CheckpointManager(out / "checkpoints", every_minutes=float(cfg.get("checkpoint_minutes", 15)))
    run_id = None
    if cfg.get("resume", True) and not smoke and ckpt.last_path.exists():
        run_id = torch.load(ckpt.last_path, map_location="cpu", weights_only=False).get("mlflow_run_id")
    tracker = Tracker(cfg, run_name=cfg.get("name"), resume_run_id=run_id)
    t0 = time.perf_counter()
    fit_info: dict = {}
    train_tput = None
    if name in MEMORY_BANK:
        train_loader = make_loader(make_dataset(cfg, df, train_normal), bs, False, cfg)
        off_loader = None
        mcfg = cfg.get("memory", {})
        if int(mcfg.get("offmask_images", 0)) > 0:
            defect_train = [i for i in split["train"] if lab[i] != 6]
            n_off = int(mcfg["offmask_images"]) if not smoke else 8
            off_idx = subsample(defect_train, n_off, seed=1, labels=[lab[i] for i in defect_train])
            off_loader = make_loader(make_dataset(cfg, df, off_idx), bs, False, cfg)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            fit_info = model.fit(
                train_loader, device, offmask_loader=off_loader, margin_px=int(mcfg.get("margin_px", 16))
            )
        if mcfg.get("save_memory_bank", True) and not smoke and hasattr(model, "memory"):
            # bank only (FP16), the frozen backbone is re-created from its pretrained weights at load time
            torch.save(
                {"memory": model.memory.half().cpu(), "cfg": cfg}, out / "checkpoints" / "memory_bank.ckpt"
            )
    else:
        train_idx = list(train_normal)
        if int(tcfg.get("offmask_images", 0)) > 0:
            # off-mask protocol for trained detectors: add train defect images (same subsample as the
            # memory-bank models); the loss ignores patches within margin_px of their masks
            defect_train = [i for i in split["train"] if lab[i] != 6]
            n_off = int(tcfg["offmask_images"]) if not smoke else 8
            off_idx = subsample(defect_train, n_off, seed=1, labels=[lab[i] for i in defect_train])
            train_idx += list(off_idx)
            fit_info["n_offmask_images"] = len(off_idx)
        train_loader = make_loader(
            make_dataset(cfg, df, train_idx, train=bool(tcfg.get("augment", False))),
            min(bs, len(train_idx)),
            True,
            cfg,
            drop_last=len(train_idx) > bs,
        )
        if name == "efficientad":
            val_loader = make_loader(make_dataset(cfg, df, val_normal), bs, False, cfg)
            train_tput = _train_efficientad(
                model, cfg, train_loader, val_loader, device, out, tracker, smoke, ckpt
            )
        else:
            train_tput = _train_dinomaly(model, cfg, train_loader, device, out, tracker, smoke, ckpt)
        torch.save({"model": model.state_dict(), "cfg": cfg}, out / "checkpoints" / "final.ckpt")
    fit_seconds = time.perf_counter() - t0

    test_ds = make_dataset(cfg, df, test_idx, mask_size=eval_size)
    test_loader = make_loader(test_ds, int(cfg.get("eval", {}).get("batch_size", 8)), False, cfg)
    tput = Throughput(warmup=1)
    met = evaluate_anomaly(model, test_loader, device, eval_size, tput=tput)
    arrays = met.pop("_arrays")
    np.savez_compressed(out / "test_scores.npz", idx=np.asarray(test_idx), **arrays)
    met.update(
        {
            "fit_seconds": fit_seconds,
            "infer_img_per_s": tput.items_per_s,
            "n_train_normal": len(train_normal),
            **fit_info,
        }
    )
    tracker.log_metrics({f"test_{k}": v for k, v in met.items()})

    rob = {}
    rcfg = cfg.get("robustness", {})
    if rcfg.get("enabled", False):
        for pert in rcfg.get("perturbations", list(SEVERITIES)):
            for level in rcfg.get("levels", [1, 3, 5]):
                r = evaluate_anomaly(model, test_loader, device, eval_size, perturbation=pert, level=level)
                r.pop("_arrays", None)
                rob[f"{pert}_L{level}"] = r
                tracker.log_metrics(
                    {
                        f"rob_{pert}_L{level}_image_auroc": r["image_auroc"],
                        f"rob_{pert}_L{level}_pixel_auroc": r.get("pixel_auroc", float("nan")),
                    }
                )
    save_json({"test": met, "robustness": rob}, out / "test_metrics.json")
    if not smoke:
        ckpt.finalize(bool(cfg.get("keep_last_checkpoint", False)))
    tracker.end()
    if rec:
        rec.add(
            fit_seconds=round(fit_seconds, 2),
            infer_img_per_s=round(tput.items_per_s, 1),
            n_train=len(train_normal),
            n_test=len(test_idx),
            params=count_params(model),
            params_trainable=count_params(model, True),
            smoke_image_auroc=met["image_auroc"],
            **(
                {
                    "train_img_per_s": round(train_tput.items_per_s, 1),
                    "train_it_per_s": round(train_tput.iters_per_s, 2),
                    "steps": int(cfg["smoke"]["max_steps"]),
                }
                if train_tput
                else {}
            ),
            **{k: v for k, v in fit_info.items()},
        )
        return rec.finish(out)
    return met


__all__ = ["build_anomaly_model", "evaluate_anomaly", "run_anomaly", "to_model_input"]
