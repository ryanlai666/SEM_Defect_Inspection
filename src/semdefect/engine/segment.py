"""Binary defect segmentation: U-Net baseline vs frozen DINOv2 + light decoder."""

from __future__ import annotations

import numpy as np
import torch

from ..checkpoint import CheckpointManager
from ..metrics import SegmentationMeter
from ..models import build_segmenter, count_params, dice_bce_loss
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


@torch.no_grad()
def evaluate_segmenter(model, loader, device, amp=True, threshold=0.5, perturbation="clean", level=0) -> dict:
    model.eval()
    meter = SegmentationMeter(threshold)
    for batch in perturbed_batches(loader, device, perturbation, level):
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            prob = torch.sigmoid(model(batch["image"]).float())
        meter.update(prob.cpu().numpy(), batch["mask"].numpy())
    met = meter.compute()
    met["_per_image"] = {"dice": np.asarray(meter.dice), "iou": np.asarray(meter.iou)}  # popped by callers
    return met


def train_segmentation(cfg: dict) -> dict:
    device = setup(cfg)
    out = output_dir(cfg)
    df, split = load_data(cfg)
    tcfg = cfg["train"]
    smoke = is_smoke(cfg)
    rec = SmokeRecorder(cfg, device) if smoke else None
    lab = df.set_index("idx")["label"]
    train_idx, val_idx, test_idx = list(split["train"]), list(split["val"]), list(split["test"])
    if not cfg["data"].get("include_empty_masks", True):  # optionally drop class-6 (empty-mask) images
        train_idx = [i for i in train_idx if lab[i] != 6]
    if smoke:
        n = int(cfg["smoke"].get("n_images", 50))
        train_idx = subsample(train_idx, n, labels=[lab[i] for i in train_idx])
        val_idx = test_idx = subsample(val_idx, n // 2, labels=[lab[i] for i in val_idx])

    train_ds = make_dataset(cfg, df, train_idx, train=True)
    bs = min(int(tcfg.get("batch_size", 16)), len(train_ds))
    train_loader = make_loader(train_ds, bs, True, cfg, drop_last=len(train_ds) > bs)
    val_loader = make_loader(make_dataset(cfg, df, val_idx), int(tcfg.get("eval_batch_size", 32)), False, cfg)

    model = build_segmenter(cfg, pretrained=cfg["model"].get("pretrained", True)).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        params, lr=float(tcfg.get("lr", 1e-3)), weight_decay=float(tcfg.get("weight_decay", 1e-4))
    )
    total = int(cfg["smoke"]["max_steps"]) if smoke else int(tcfg.get("steps", 6000))
    warm = min(int(tcfg.get("warmup_steps", 200)), max(1, total // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / max(1, total))))
    )
    amp = device.type == "cuda" and cfg.get("amp", True)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ckpt = CheckpointManager(out / "checkpoints", every_minutes=float(cfg.get("checkpoint_minutes", 15)))

    step, run_id = 0, None
    state = ckpt.load_last() if cfg.get("resume", True) and not smoke else None
    if state:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        scaler.load_state_dict(state["scaler"])
        step, run_id = state["step"], state.get("mlflow_run_id")
        print(f"resumed from step {step}")
    tracker = Tracker(cfg, run_name=cfg.get("name"), resume_run_id=run_id)

    def snapshot():
        return {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "mlflow_run_id": tracker.run_id,
            "cfg": cfg,
        }

    eval_every = int(tcfg.get("eval_every", 500)) if not smoke else total
    tput = Throughput(warmup=2)
    it = infinite(train_loader)
    model.train()
    while step < total:
        batch = next(it)
        tput.start()
        x = batch["image"].to(device, non_blocking=True)
        m = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            logits = model(x)
        loss = dice_bce_loss(
            logits.float(), m, float(tcfg.get("bce_weight", 1.0)), float(tcfg.get("dice_weight", 1.0))
        )
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        tput.stop(x.shape[0])
        step += 1
        if step % 20 == 0:
            tracker.log_metrics({"train_loss": loss.item(), "lr": sched.get_last_lr()[0]}, step)
        if step % eval_every == 0 or step == total:
            met = evaluate_segmenter(model, val_loader, device, amp)
            met.pop("_per_image")
            tracker.log_metrics({f"val_{k}": v for k, v in met.items()}, step)
            ckpt.update_best(met["dice_mean"], {"model": model.state_dict(), "step": step, "cfg": cfg})
            model.train()
        ckpt.maybe_save(snapshot)
    ckpt.save(snapshot())

    if ckpt.best_path.exists():
        model.load_state_dict(torch.load(ckpt.best_path, map_location=device, weights_only=False)["model"])
    test_loader = make_loader(
        make_dataset(cfg, df, test_idx), int(tcfg.get("eval_batch_size", 32)), False, cfg
    )
    test = evaluate_segmenter(model, test_loader, device, amp)
    np.savez_compressed(out / "test_per_image.npz", **test.pop("_per_image"))
    tracker.log_metrics({f"test_{k}": v for k, v in test.items()})
    save_json(test, out / "test_metrics.json")
    if not smoke:
        ckpt.finalize(bool(cfg.get("keep_last_checkpoint", False)))
    tracker.end()
    if rec:
        rec.add(
            train_img_per_s=round(tput.items_per_s, 1),
            train_it_per_s=round(tput.iters_per_s, 2),
            batch_size=bs,
            steps=total,
            params_total=count_params(model),
            params_trainable=count_params(model, True),
        )
        return rec.finish(out)
    return test
