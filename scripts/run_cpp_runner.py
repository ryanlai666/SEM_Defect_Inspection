"""Drive cpp/build/trt_runner(.exe) over engines x batch x streams, each run through the GPU guard,
and check its image scores against the PyTorch FP32 reference written by bench_engine.py.

    python scripts/run_cpp_runner.py            # after bench_engine.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from guarded import guarded_cmd  # noqa: E402

from semdefect.metrics import image_auroc  # noqa: E402

EXP = ROOT / "outputs" / "export"
BENCH = ROOT / "outputs" / "bench"
EXE = ROOT / "cpp" / "build" / ("trt_runner.exe" if os.name == "nt" else "trt_runner")
TRT_DIR = ROOT / ".venv" / "Lib" / "site-packages" / "tensorrt_libs"


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="use outputs/export_<tag> and write runner_<tag>_*.json")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--smoke", action="store_true", help="short code check (guard --smoke, <= 1.5 GB)")
    a = ap.parse_args()
    global EXP
    if a.tag:
        EXP = ROOT / "outputs" / f"export_{a.tag}"
    prefix = f"runner_{a.tag}_" if a.tag else "runner_"
    y = np.load(EXP / "test_labels.npy")
    fp = EXP / "test_scores_torch_fp32_floatpath.npy"  # float-path reference if available
    ref = np.load(fp if fp.exists() else EXP / "test_scores_torch_fp32.npy")
    configs = [  # (engine precision, batch, streams, knn kernel)
        ("fp32", 8, 1, "fp32"),
        ("fp16", 1, 1, "fp32"),
        ("fp16", 8, 1, "fp32"),
        ("fp16", 8, 2, "fp32"),
        ("fp16", 1, 1, "fp16"),
        ("fp16", 8, 1, "fp16"),
        ("fp16", 8, 2, "fp16"),
    ]
    for prec, bs, streams, knn in configs:
        eng = EXP / f"patchcore_wrn50_256_{prec}.engine"
        if not eng.exists():
            continue
        tag = f"{prec}_b{bs}_s{streams}_knn{knn}"
        scores_path = BENCH / f"{prefix}scores_{tag}.bin"
        out_json = BENCH / f"{prefix}{tag}.json"
        reps = []
        for rep in range(a.repeats):  # independent process launches (no single-run timings)
            rep_json = BENCH / f"{prefix}{tag}.rep{rep}.json"
            cmd = [
                str(EXE),
                "--engine",
                str(eng),
                "--bank",
                str(EXP / "patchcore_bank.bin"),
                "--images",
                str(EXP / "test_raw_u8.bin"),
                "--trt-dir",
                str(TRT_DIR),
                "--batch",
                str(bs),
                "--streams",
                str(streams),
                "--knn",
                knn,
                "--iters",
                "200",
                "--scores-out",
                str(scores_path),
                "--json",
                str(rep_json),
            ]
            pr = subprocess.run(
                guarded_cmd(
                    cmd, 1.0 if a.smoke else 2.0, f"semdefect trt_runner {tag} rep{rep}", smoke=a.smoke
                ),
                capture_output=True,
                text=True,
            )
            if pr.returncode != 0:
                print(f"{tag} rep{rep}: FAILED", pr.stdout[-2000:], pr.stderr[-2000:], sep="\n")
                break
            reps.append(json.loads(rep_json.read_text()))
            rep_json.unlink()
        if not reps:
            continue
        r = dict(reps[0])
        for key in ("latency_ms_p50", "latency_ms_p99", "throughput_img_per_s", "vram_used_by_runner_mib"):
            vals = np.array([x[key] for x in reps], dtype=float)
            r[key] = float(vals.mean())
            r[key + "_std"] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
            r[key + "_runs"] = vals.tolist()
        r["repeats"] = len(reps)
        s = np.fromfile(scores_path, dtype=np.float32)[: len(ref)]
        auc, auc_ref = image_auroc(y, s), image_auroc(y, ref)
        rel = float(np.max(np.abs(s - ref) / np.abs(ref)))
        r["parity"] = f"r={np.corrcoef(s, ref)[0, 1]:.7f}, max rel {rel:.1e}, dAUROC={auc - auc_ref:+.4f}"
        r["image_auroc_subset"] = auc
        out_json.write_text(json.dumps(r, indent=2))
        print(
            tag,
            json.dumps(
                {k: r[k] for k in ("latency_ms_p50", "latency_ms_p50_std", "throughput_img_per_s", "parity")}
            ),
        )


if __name__ == "__main__":
    main()
