"""Command line interface.

    semdefect prepare                                   # splits + few-shot supports -> data/splits/
    semdefect run configs/classify/fewshot_dinov2_vits14.yaml [key=value ...]
    semdefect run configs/anomaly/patchcore.yaml --smoke
    semdefect robustness configs/segment/unet.yaml

`run` dispatches on the config's `task` field (fewshot | classify | segment | anomaly).
`--smoke` runs the TRAINING-GATE smoke test: ~50 images, a few steps, records it/s and peak VRAM to
outputs/smoke/<name>/smoke.json.
"""

from __future__ import annotations

import argparse
import json
import time

from .config import load_config


def _run(cfg: dict) -> dict:
    task = cfg["task"]
    if task == "fewshot":
        from .engine.classify import run_fewshot

        return run_fewshot(cfg)
    if task == "classify":
        from .engine.classify import train_cnn

        return train_cnn(cfg)
    if task == "segment":
        from .engine.segment import train_segmentation

        return train_segmentation(cfg)
    if task == "anomaly":
        from .engine.anomaly import run_anomaly

        return run_anomaly(cfg)
    raise ValueError(f"unknown task {task}")


def prepare(seeds: list[int], ks: list[int]) -> None:
    from .data import fewshot_supports, load_index, load_or_make_split
    from .utils import resolve_path, save_json

    df = load_index()
    for seed in seeds:
        split = load_or_make_split(df, seed=seed)
        sup = fewshot_supports(df, split["train"], ks=ks, seeds=[0, 1, 2, 3, 4])
        save_json(sup, resolve_path(f"data/splits/carinthia_s_fewshot_split{seed}.json"))
        counts = {
            name: df[df["idx"].isin(idx)]["label"].value_counts().sort_index().to_dict()
            for name, idx in split.items()
        }
        print(
            f"split seed {seed}: "
            + json.dumps({k: {int(a): int(b) for a, b in v.items()} for k, v in counts.items()})
        )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="semdefect")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="create deterministic splits and few-shot supports")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10, 20])
    for name in ("run", "robustness"):
        r = sub.add_parser(name)
        r.add_argument("config")
        r.add_argument("overrides", nargs="*", help="dotted key=value overrides")
        r.add_argument("--smoke", action="store_true", help="tiny smoke test (~50 images, a few steps)")
    args, extra = ap.parse_known_args(argv)
    bad = [e for e in extra if "=" not in e]
    if bad:
        ap.error(f"unrecognized arguments: {' '.join(bad)}")
    if args.cmd == "prepare":
        prepare(args.seeds, args.ks)
        return
    overrides = list(args.overrides) + extra  # key=value overrides may appear before or after flags
    if args.smoke:
        overrides.append("smoke.enabled=true")
    cfg = load_config(args.config, overrides)
    t0 = time.time()
    if args.cmd == "run":
        res = _run(cfg)
    else:
        from .engine.robustness import run_robustness

        res = run_robustness(cfg)
    _write_resources(cfg, args.cmd, time.time() - t0)
    print(json.dumps(res, indent=2, default=str)[:4000])


def _write_resources(cfg: dict, cmd: str, wall_s: float) -> None:
    """Per-run compute record (wall time, peak VRAM, device, VRAM reservation) for the campaign log."""
    import os

    import torch

    from .engine.common import output_dir
    from .utils import save_json

    rec = {
        "cmd": cmd,
        "wall_s": round(wall_s, 1),
        "device": str(cfg.get("device", "auto")),
        "gpu_guard_gb": os.environ.get("GPU_GUARD_GB"),
    }
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        rec["peak_alloc_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        rec["peak_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
        rec["gpu"] = torch.cuda.get_device_name(0)
    save_json(rec, output_dir(cfg) / f"resources_{cmd}.json")


if __name__ == "__main__":
    main()
