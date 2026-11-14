"""Classification: frozen-DINOv2 few-shot probes and fine-tuned CNN baselines."""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from ..checkpoint import CheckpointManager
from ..data import fewshot_supports
from ..metrics import classification_metrics
from ..models import DinoV2, LinearProbe, PrototypeClassifier, build_classifier, count_params
from ..models.probes import NearestNeighbor
from ..tracking import Tracker
from ..utils import REPO_ROOT, Throughput, save_json
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

# ============================================================================ frozen features


@torch.no_grad()
def extract_features(cfg: dict, df, indices, device, tput: Throughput | None = None) -> np.ndarray:
    m = cfg["model"]
    enc = DinoV2(
        m["backbone"], img_size=int(cfg["data"]["img_size"]), pretrained=m.get("pretrained", True)
    ).to(device)
    ds = make_dataset(cfg, df, indices, train=False)
    loader = make_loader(ds, int(cfg.get("batch_size", 64)), False, cfg)
    feats = []
    amp = device.type == "cuda" and cfg.get("amp", True)
    for batch in loader:
        if tput:
            tput.start()
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            f = enc.global_features(batch["image"].to(device, non_blocking=True))
        feats.append(f.float().cpu())
        if tput:
            tput.stop(batch["image"].shape[0])
    return torch.cat(feats).numpy()


def cached_features(cfg: dict, df, device) -> tuple[np.ndarray, np.ndarray]:
    """All-image features, cached to data/cache/features/<backbone>_<size>_<precision>.npz.

    Stored in float32 so every run that reads the cache sees the same features as the run that extracted
    them.  The precision tag records how they were computed."""
    m = cfg["model"]
    prec = "amp16" if (device.type == "cuda" and cfg.get("amp", True)) else "fp32"
    path = REPO_ROOT / "data/cache/features" / f"{m['backbone']}_{cfg['data']['img_size']}_{prec}.npz"
    if path.exists():
        z = np.load(path)
        return z["idx"], z["feats"].astype(np.float32)
    idx = df["idx"].to_numpy()
    feats = extract_features(cfg, df, idx, device).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, idx=idx, feats=feats)
    return idx, feats


def run_fewshot(cfg: dict) -> dict:
    """Linear probe and prototype classifiers on frozen DINOv2 features for every (k, seed)."""
    device = setup(cfg)
    out = output_dir(cfg)
    df, split = load_data(cfg)
    fs = cfg.get("fewshot", {})
    ks = list(fs.get("ks", [1, 5, 10, 20]))
    seeds = list(fs.get("seeds", [0, 1, 2, 3, 4]))
    methods = list(fs.get("methods", ["linear", "proto"]))
    classes = list(cfg["data"].get("classes", [1, 2, 3, 4, 5, 6]))
    smoke = is_smoke(cfg)
    rec = SmokeRecorder(cfg, device) if smoke else None
    lab = df.set_index("idx")["label"]

    test_idx = list(split["test"])
    if smoke:  # ~50 images total: tiny support pool + tiny query set
        n = int(cfg["smoke"].get("n_images", 50))
        pool = subsample(split["train"], n // 2, labels=[lab[i] for i in split["train"]])
        test_idx = subsample(test_idx, n - len(pool), labels=[lab[i] for i in test_idx])
        tput = Throughput(warmup=1)
        val_s = subsample(split["val"], max(6, n // 4), labels=[lab[i] for i in split["val"]])
        all_idx = sorted(set(pool) | set(test_idx) | set(val_s))
        feats_all = extract_features(cfg, df, all_idx, device, tput)
        feat_of = dict(zip(all_idx, feats_all))
        train_pool = pool
        rec.add(
            feature_img_per_s=round(tput.items_per_s, 1),
            n_images=len(all_idx),
            params=count_params(
                DinoV2(cfg["model"]["backbone"], int(cfg["data"]["img_size"]), pretrained=False)
            ),
        )
    else:
        idx, feats = cached_features(cfg, df, device)
        feat_of = dict(zip(idx.tolist(), feats.astype(np.float32)))
        train_pool = split["train"]

    sup = fewshot_supports(df, train_pool, ks=ks, seeds=seeds, classes=classes)
    x_te = np.stack([feat_of[i] for i in test_idx])
    y_te = np.array([lab[i] for i in test_idx])
    val_idx = [i for i in split["val"] if i in feat_of]
    x_va = np.stack([feat_of[i] for i in val_idx])
    y_va = np.array([lab[i] for i in val_idx])
    c_grid = [float(c) for c in fs.get("C_grid", [0.01, 0.1, 1.0, 10.0, 100.0])]
    results: dict = {
        "k_eff": sup["k_eff"],
        "runs": [],
        "selected_C": {},
        "feature_precision": "fp32" if not (device.type == "cuda" and cfg.get("amp", True)) else "amp16",
        "test_support": {str(c): int((y_te == c).sum()) for c in classes},
        "protocol": "linear: C selected on the labelled val split (mean val macro-F1 over support seeds, "
        "ties -> smaller C); linear_C1: fixed C=1 (untuned); proto: CL2N prototypes; knn1: cosine 1-NN",
    }
    tracker = Tracker(cfg, run_name=cfg.get("name"))
    t0 = time.perf_counter()
    for k in ks + (["full"] if fs.get("include_full", True) else []):
        seeds_k = seeds if k != "full" else [0]
        sets = {}
        for seed in seeds_k:
            tr = (
                list(train_pool)
                if k == "full"
                else [i for c in classes for i in sup["supports"][str(seed)][str(k)][str(c)]]
            )
            sets[seed] = (np.stack([feat_of[i] for i in tr]), np.array([lab[i] for i in tr]))
        best_c = float(fs.get("C", 1.0))
        if "linear" in methods:  # hyper-parameter selection on val only, never on test
            val_f1 = {}
            for c in c_grid:
                val_f1[c] = float(
                    np.mean(
                        [
                            classification_metrics(
                                y_va, LinearProbe(C=c).fit(*sets[sd]).predict(x_va), labels=classes
                            )["macro_f1"]
                            for sd in seeds_k
                        ]
                    )
                )
            best_c = max(c_grid, key=lambda c: (val_f1[c], -c))
            results["selected_C"][str(k)] = {
                "C": best_c,
                "val_macro_f1_by_C": {str(c): v for c, v in val_f1.items()},
            }
        for seed in seeds_k:
            x_tr, y_tr = sets[seed]
            for method in methods:
                if method == "linear":
                    clf = LinearProbe(C=best_c)
                elif method == "linear_C1":
                    clf = LinearProbe(C=1.0)
                elif method == "proto":
                    clf = PrototypeClassifier(center=bool(fs.get("center", True)))
                elif method == "knn1":
                    clf = NearestNeighbor()
                else:
                    raise ValueError(method)
                clf.fit(x_tr, y_tr)
                met = classification_metrics(y_te, clf.predict(x_te), labels=classes)
                results["runs"].append({"k": k, "seed": seed, "method": method, **met})
    results["fit_eval_seconds"] = time.perf_counter() - t0
    summary = summarize_fewshot(results["runs"])
    results["summary"] = summary
    for key, v in summary.items():
        tracker.log_metrics(
            {
                f"{key}_macro_f1_mean": v["macro_f1_mean"],
                f"{key}_macro_f1_std": v["macro_f1_std"],
                f"{key}_acc_mean": v["accuracy_mean"],
            }
        )
    save_json(results, out / "fewshot_results.json")
    tracker.log_artifact(str(out / "fewshot_results.json"))
    tracker.end()
    if rec:
        rec.add(fit_eval_seconds=round(results["fit_eval_seconds"], 2), n_configs=len(results["runs"]))
        return rec.finish(out)
    return summary


def summarize_fewshot(runs: list[dict]) -> dict:
    groups: dict = {}
    for r in runs:
        groups.setdefault(f"{r['method']}_k{r['k']}", []).append(r)
    out = {}
    for key, rs in groups.items():
        f1 = np.array([r["macro_f1"] for r in rs])
        acc = np.array([r["accuracy"] for r in rs])
        out[key] = {
            "macro_f1_mean": float(f1.mean()),
            "macro_f1_std": float(f1.std(ddof=1)) if f1.size > 1 else 0.0,  # sample std
            "accuracy_mean": float(acc.mean()),
            "n_seeds": len(rs),
        }
    return out


# ============================================================================ CNN fine-tuning


def _class_weights(labels: np.ndarray, classes: list[int], power: float) -> torch.Tensor:
    counts = np.array([max(1, (labels == c).sum()) for c in classes], dtype=np.float64)
    w = (counts.sum() / counts) ** power
    return torch.tensor(w / w.mean(), dtype=torch.float32)


@torch.no_grad()
def evaluate_classifier(model, loader, device, classes, amp=True, perturbation="clean", level=0) -> dict:
    model.eval()
    ys, ps = [], []
    for batch in perturbed_batches(loader, device, perturbation, level):
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = model(batch["image"])
        ps.append(logits.argmax(1).cpu())
        ys.append(batch["label_idx"])
    y = torch.cat(ys).numpy()
    p = torch.cat(ps).numpy()
    met = classification_metrics([classes[i] for i in y], [classes[i] for i in p], labels=classes)
    return met


def train_cnn(cfg: dict) -> dict:
    device = setup(cfg)
    out = output_dir(cfg)
    df, split = load_data(cfg)
    classes = list(cfg["data"].get("classes", [1, 2, 3, 4, 5, 6]))
    tcfg = cfg["train"]
    lab = df.set_index("idx")["label"]
    smoke = is_smoke(cfg)
    rec = SmokeRecorder(cfg, device) if smoke else None

    fs = cfg.get("fewshot", {})
    if fs.get("k"):
        sup = fewshot_supports(
            df, split["train"], ks=[int(fs["k"])], seeds=[int(fs.get("seed", 0))], classes=classes
        )
        train_idx = [
            i for c in classes for i in sup["supports"][str(fs.get("seed", 0))][str(fs["k"])][str(c)]
        ]
    else:
        train_idx = list(split["train"])
    val_idx, test_idx = list(split["val"]), list(split["test"])
    if smoke:
        n = int(cfg["smoke"].get("n_images", 50))
        train_idx = subsample(train_idx, n, labels=[lab[i] for i in train_idx])
        val_idx = subsample(val_idx, n // 2, labels=[lab[i] for i in val_idx])
        test_idx = val_idx

    train_ds = make_dataset(cfg, df, train_idx, train=True)
    bs = min(int(tcfg.get("batch_size", 64)), len(train_ds))
    train_loader = make_loader(train_ds, bs, True, cfg, drop_last=len(train_ds) > bs)
    val_loader = make_loader(
        make_dataset(cfg, df, val_idx), int(tcfg.get("eval_batch_size", 128)), False, cfg
    )

    model = build_classifier(
        cfg["model"]["name"], len(classes), pretrained=cfg["model"].get("pretrained", True)
    ).to(device)
    y_tr = np.array([lab[i] for i in train_idx])
    weights = _class_weights(y_tr, classes, float(tcfg.get("class_weight_power", 0.5))).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(tcfg.get("lr", 3e-4)), weight_decay=float(tcfg.get("weight_decay", 0.05))
    )
    total = int(cfg["smoke"]["max_steps"]) if smoke else int(tcfg.get("steps", 3000))
    warm = min(int(tcfg.get("warmup_steps", 100)), max(1, total // 10))
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

    eval_every = int(tcfg.get("eval_every", 250)) if not smoke else total
    select_best = bool(tcfg.get("select_best", True))
    tput = Throughput(warmup=2)
    it = infinite(train_loader)
    model.train()
    while step < total:
        batch = next(it)
        tput.start()
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label_idx"].to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            loss = F.cross_entropy(
                model(x), y, weight=weights, label_smoothing=float(tcfg.get("label_smoothing", 0.0))
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
        if select_best and (step % eval_every == 0 or step == total):
            met = evaluate_classifier(model, val_loader, device, classes, amp)
            tracker.log_metrics({"val_macro_f1": met["macro_f1"], "val_acc": met["accuracy"]}, step)
            ckpt.update_best(met["macro_f1"], {"model": model.state_dict(), "step": step, "cfg": cfg})
            model.train()
        ckpt.maybe_save(snapshot)
    ckpt.save(snapshot())

    if select_best and ckpt.best_path.exists():
        model.load_state_dict(torch.load(ckpt.best_path, map_location=device, weights_only=False)["model"])
    elif not select_best:  # k-shot honest: final weights, no selection on the 919-image labelled val split
        ckpt.save({"model": model.state_dict(), "step": step, "cfg": cfg}, ckpt.best_path)
    test_loader = make_loader(
        make_dataset(cfg, df, test_idx), int(tcfg.get("eval_batch_size", 128)), False, cfg
    )
    test = evaluate_classifier(model, test_loader, device, classes, amp)
    test["checkpoint"] = "best-on-val" if select_best else "last"
    test["n_train_images"] = len(train_idx)
    tracker.log_metrics({"test_macro_f1": test["macro_f1"], "test_acc": test["accuracy"]})
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
            params=count_params(model),
        )
        return rec.finish(out)
    return test
