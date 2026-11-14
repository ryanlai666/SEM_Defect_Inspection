"""Smoke tests on the real data: every model on ~50 images for a few steps.

Every run goes through scripts/guarded.py (the optional shared-GPU guard, env GPU_GUARD) with an honest
peak-VRAM reservation; the process caps itself to that reservation.
On CUDA OOM the reservation is raised by 1 GB (up to 4 GB), then the batch size is halved.
Each config runs in a fresh process (clean VRAM accounting).  Results: outputs/smoke/summary.json.

    python scripts/build_cache.py      # CPU, once (no GPU memory reserved while decoding JPEGs)
    python scripts/smoke_all.py        # all
    python scripts/smoke_all.py --only patchcore anomalydino
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
sys.path.insert(0, str(ROOT / "scripts"))
from guarded import guarded_cmd  # noqa: E402

# (config, batch-size key, initial reservation GB incl. ~0.4 GB CUDA context, extra overrides)
RUNS = [
    ("configs/classify/fewshot_dinov2_vits14.yaml", "batch_size", 1.0, []),
    ("configs/classify/fewshot_dinov2_vitb14.yaml", "batch_size", 1.5, []),
    ("configs/classify/resnet18_full.yaml", "train.batch_size", 2.5, ["smoke.n_images=64"]),
    ("configs/classify/convnext_tiny_full.yaml", "train.batch_size", 3.0, ["smoke.n_images=64"]),
    ("configs/segment/unet.yaml", "train.batch_size", 3.0, []),
    ("configs/segment/unet_r18.yaml", "train.batch_size", 2.0, []),
    ("configs/segment/dino_seg_vits14.yaml", "train.batch_size", 2.0, []),
    ("configs/anomaly/patchcore.yaml", "train.batch_size", 1.0, []),
    ("configs/anomaly/anomalydino.yaml", "train.batch_size", 1.5, []),
    ("configs/anomaly/efficientad.yaml", "train.batch_size", 1.5, ["smoke.max_steps=60"]),
    ("configs/anomaly/dinomaly.yaml", "train.batch_size", 2.0, []),
]
MAX_GB = 4.0


def batch_of(cfg_path: str, key: str) -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from semdefect.config import get_by_dotted, load_config

    return int(get_by_dotted(load_config(ROOT / cfg_path), key, 8))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    summary_path = ROOT / "outputs/smoke/summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    for cfg_path, bkey, gb, extra in RUNS:
        name = Path(cfg_path).stem
        if args.only and not any(o in name for o in args.only):
            continue
        bs = batch_of(cfg_path, bkey)
        while bs >= 1:
            inner = [
                PY,
                "-m",
                "semdefect.cli",
                "run",
                cfg_path,
                "--smoke",
                f"{bkey}={bs}",
                "tracking.experiment=smoke",
                *extra,
            ]
            cmd = guarded_cmd(inner, gb, f"semdefect smoke {name} bs{bs}")
            t0 = time.time()
            p = subprocess.run(
                cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace"
            )
            dt = time.time() - t0
            if p.returncode == 0:
                break
            err = p.stderr[-3000:] + p.stdout[-1500:]
            if "OutOfMemory" in err or "out of memory" in err:
                if gb + 1.0 <= MAX_GB:
                    print(
                        f"[{name}] OOM inside {gb} GB reservation at batch {bs} -> reserve {gb + 1.0} GB",
                        flush=True,
                    )
                    gb += 1.0
                else:
                    print(f"[{name}] OOM at {gb} GB, batch {bs} -> halving batch", flush=True)
                    bs //= 2
                continue
            print(f"[{name}] FAILED rc={p.returncode} ({dt:.0f}s):\n{err}", flush=True)
            break
        out = p.stdout
        cfg_name = json.loads(out[out.find("{\n") :])["name"] if p.returncode == 0 else name
        smoke_json = ROOT / "outputs/smoke" / cfg_name / "smoke.json"
        if p.returncode == 0 and smoke_json.exists():
            rec = json.loads(smoke_json.read_text())
            rec.update(
                config=cfg_path,
                batch_size_run=bs,
                batch_size_planned=batch_of(cfg_path, bkey),
                reservation_gb=gb,
                process_wall_s=round(dt, 1),
            )
            summary[name] = rec
            print(
                f"[{name}] ok ({dt:.0f}s incl. guard wait): "
                + json.dumps({k: v for k, v in rec.items() if k != "config"}),
                flush=True,
            )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
