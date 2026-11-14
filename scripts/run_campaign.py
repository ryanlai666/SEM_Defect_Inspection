"""Resumable driver for the full local experiment plan (`plan()` below, 61 runs).

* Runs every planned run in priority order; GPU runs go through scripts/guarded.py (the optional
  shared-GPU guard, env GPU_GUARD; direct execution otherwise), CPU runs start immediately.
* Skips runs that already finished (their result json exists) -> safe to re-launch after a failure.
* Training scripts checkpoint every 15 min and resume from last.ckpt, so an interrupted run continues.
* Appends one line per event to outputs/campaign.log: start / end time, exit code, wall time, peak VRAM
  (from outputs/<name>/resources_*.json written by the CLI) and the VRAM reservation.

    python scripts/run_campaign.py                 # all tiers
    python scripts/run_campaign.py --tier cpu      # only CPU runs
    python scripts/run_campaign.py --list          # show plan + status
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe") if os.name == "nt" else sys.executable
LOG = ROOT / "outputs" / "campaign.log"
sys.path.insert(0, str(ROOT / "scripts"))
from guarded import guarded_cmd  # noqa: E402

# exact FP32 fused kNN kernel (identical results to cdist+topk, no M x N distance tile in memory)
KNN = "model.knn_backend=ext"
CPU_ENV = {"OMP_NUM_THREADS": "6", "MKL_NUM_THREADS": "6"}  # be considerate: shared 14-core machine


@dataclass
class Run:
    name: str
    tier: str  # cpu | t1 | t2 | t3
    cmd: list[str]
    gb: float | None  # None = CPU-only, no guard
    done_files: list[str] = field(default_factory=list)  # relative to outputs/<name>/ unless absolute-ish
    note: str = ""
    self_guarded: bool = False  # GPU work, but the command wraps each GPU sub-step itself


MAX_WORKERS = 2  # DataLoader workers per run (memory-safe on a shared 40 GB host)
MIN_FREE_RAM_GB = 8.0  # do not start a run while the shared host has less free RAM than this


def cli(config: str, *overrides: str, sub: str = "run") -> list[str]:
    return [PY, "-m", "semdefect.cli", sub, config, f"data.num_workers={MAX_WORKERS}", *overrides]


def free_ram_gb() -> float:
    """Free physical RAM (GB), stdlib only (Windows: GlobalMemoryStatusEx; Linux: /proc/meminfo)."""
    if os.name == "nt":
        import ctypes

        class MEMSTAT(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        st = MEMSTAT()
        st.dwLength = ctypes.sizeof(MEMSTAT)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return st.ullAvailPhys / 2**30
    with open("/proc/meminfo") as f:
        info = {ln.split(":")[0]: float(ln.split()[1]) for ln in f}
    return info.get("MemAvailable", 0.0) / 2**20


def wait_for_ram(name: str) -> None:
    logged = False
    while (free := free_ram_gb()) < MIN_FREE_RAM_GB:
        if not logged:
            log(f"WAIT-RAM {name}: free RAM {free:.1f} GB < {MIN_FREE_RAM_GB} GB, waiting")
            logged = True
        time.sleep(60)
    if logged:
        log(f"RAM-OK {name}: free RAM {free:.1f} GB")


def plan() -> list[Run]:
    """The local experiment plan.  Not included (compute budget): EfficientAD accuracy runs, INT8 calibration,
    AD image-AUROC robustness sweeps, extra PatchCore coreset seeds; anomaly k-shot, ConvNeXt and the scratch
    U-Net run with 1 seed.  Included: FP16 PyTorch baselines (kernel + end-to-end), off-mask PatchCore /
    AnomalyDINO, 3x timing repeats.  Dinomaly ran separately on a Colab L4 (scripts/colab/run_all.sh)."""
    R: list[Run] = []
    # ------------------------------------------------------------------ CPU: few-shot probes (FP32 features,
    # C tuned on val, + untuned C=1, prototypes, 1-NN) and the training-free high-pass AD baselines
    for bb in ("vits14", "vitb14"):
        for sp in (0, 1, 2):
            n = f"fewshot_{bb}_split{sp}"
            R.append(
                Run(
                    n,
                    "cpu",
                    cli(
                        f"configs/classify/fewshot_dinov2_{bb}.yaml",
                        f"data.split_seed={sp}",
                        f"name={n}",
                        "device=cpu",
                        "amp=false",
                        "batch_size=32",
                    ),
                    None,
                    ["fewshot_results.json"],
                    "FP32 features (CPU); C selected on val",
                )
            )
    for sp in (0, 1, 2):
        rob = ["robustness.enabled=true", "robustness.levels=[1,3,5]"] if sp == 0 else []
        R.append(
            Run(
                f"highpass_split{sp}",
                "cpu",
                cli(
                    "configs/anomaly/highpass.yaml",
                    f"data.split_seed={sp}",
                    f"name=highpass_split{sp}",
                    "device=cpu",
                    "amp=false",
                    *rob,
                ),
                None,
                ["test_metrics.json"],
                "training-free baseline, class-6 protocol",
            )
        )
        R.append(
            Run(
                f"highpass_offmask_split{sp}",
                "cpu",
                cli(
                    "configs/anomaly/highpass_offmask.yaml",
                    f"data.split_seed={sp}",
                    f"name=highpass_offmask_split{sp}",
                    "device=cpu",
                    "amp=false",
                ),
                None,
                ["test_metrics.json"],
                "training-free baseline, defect-only pixel task",
            )
        )
    # ------------------------------------------------------------------ tier 1: anomaly detection + deployment
    R.append(
        Run(
            "patchcore_all_s0",
            "t1",
            cli("configs/anomaly/patchcore.yaml", "name=patchcore_all_s0", "model.seed=0", KNN),
            1.5,
            ["test_metrics.json"],
            "class-6 protocol (texture detection; see high-pass baseline)",
        )
    )
    R.append(
        Run(
            "anomalydino_all",
            "t1",
            cli("configs/anomaly/anomalydino.yaml", "name=anomalydino_all", KNN),
            2.5,
            ["test_metrics.json"],
            "class-6 protocol",
        )
    )
    R.append(
        Run(
            "patchcore_offmask",
            "t1",
            cli("configs/anomaly/patchcore_offmask.yaml", "name=patchcore_offmask", KNN),
            2.5,
            ["test_metrics.json"],
            "harder task: off-mask normals, pixel metrics on defect images",
        )
    )
    R.append(
        Run(
            "anomalydino_offmask",
            "t1",
            cli("configs/anomaly/anomalydino_offmask.yaml", "name=anomalydino_offmask", KNN),
            3.5,  # ~827k x 384 FP32 bank (1.3 GB) + FP16 copy during fit
            ["test_metrics.json"],
            "harder task",
        )
    )
    for meth, gb in (("patchcore", 1.5), ("anomalydino", 2.5)):
        for k in (1, 5, 10):
            n = f"{meth}_k{k}_s0"
            R.append(
                Run(
                    n,
                    "t1",
                    cli(
                        f"configs/anomaly/{meth}.yaml",
                        f"name={n}",
                        f"train.k_normal={k}",
                        "train.k_seed=0",
                        KNN,
                    ),
                    gb,
                    ["test_metrics.json"],
                    "few-shot AD, 1 seed",
                )
            )
    R.append(
        Run(
            "bench_kernels_v2",
            "t1",
            [PY, "scripts/bench_kernels.py", "--out", "outputs/bench/kernels_v2.json"],
            2.5,
            ["../bench/kernels_v2.json"],
            "kNN/preprocess micro-benchmarks incl. FP16 PyTorch baseline, 3 rounds",
        )
    )
    R.append(
        Run(
            "deploy_patchcore_full",
            "t1",
            [
                PY,
                "scripts/bench_engine.py",
                "--n-normal",
                "136",
                "--n-test",
                "919",
                "--out-tag",
                "full",
                "--skip-int8",
            ],
            3.0,
            ["../bench/engine_full.json"],
            "TRT FP32/FP16 vs PyTorch FP32/AMP-FP16, full test split, 3 rounds",
        )
    )
    R.append(
        Run(
            "deploy_runner_full",
            "t1",
            [PY, "scripts/run_cpp_runner.py", "--tag", "full", "--repeats", "3"],
            None,
            ["../bench/runner_full_fp16_b8_s1_knnfp16.json"],
            "C++ runner, 3 repeats (wraps each call itself)",
            self_guarded=True,
        )
    )
    R.append(
        Run(
            "deploy_maps_full",
            "t1",
            [PY, "scripts/check_runner_maps.py", "--tag", "full", "--n", "200", "--n-test", "919"],
            1.5,
            ["../bench/runner_maps_parity_full.json"],
            "C++ runner anomaly maps vs PyTorch, first 200 test images",
        )
    )
    R.append(
        Run(
            "bench_runner_graph_full",
            "t1",
            [PY, "scripts/bench_runner_graph.py", "--tag", "full"],
            1.0,
            ["../bench/runner_graph_summary_full.json"],
            "C++ runner with vs without CUDA-graph replay, 3 repeats",
        )
    )
    # ------------------------------------------------------------------ tier 2: supervised baselines
    for s in (0, 1, 2):
        n = f"resnet18_full_s{s}"
        R.append(
            Run(
                n,
                "t2",
                cli("configs/classify/resnet18_full.yaml", f"name={n}", f"seed={s}"),
                2.5,
                ["test_metrics.json"],
            )
        )
    for k in (1, 5, 10, 20):
        for s in range(5):
            n = f"resnet18_k{k}_s{s}"
            R.append(
                Run(
                    n,
                    "t2",
                    cli(
                        "configs/classify/resnet18_fewshot.yaml",
                        f"name={n}",
                        f"seed={s}",
                        f"fewshot.k={k}",
                        f"fewshot.seed={s}",
                    ),
                    2.5,
                    ["test_metrics.json"],
                    "last checkpoint (no val selection)",
                )
            )
    for s in (0, 1, 2):
        n = f"unet_r18_s{s}"
        R.append(
            Run(
                n,
                "t2",
                cli("configs/segment/unet_r18.yaml", f"name={n}", f"seed={s}"),
                2.5,
                ["test_metrics.json"],
            )
        )
    for s in (0, 1, 2):
        n = f"dino_seg_s{s}"
        R.append(
            Run(
                n,
                "t2",
                cli("configs/segment/dino_seg_vits14.yaml", f"name={n}", f"seed={s}"),
                2.5,
                ["test_metrics.json"],
            )
        )
    R.append(
        Run(
            "resnet18_full_s0_robustness",
            "t2",
            cli("configs/classify/resnet18_full.yaml", "name=resnet18_full_s0", sub="robustness"),
            2.0,
            ["../resnet18_full_s0/robustness.json"],
        )
    )
    R.append(
        Run(
            "unet_r18_s0_robustness",
            "t2",
            cli("configs/segment/unet_r18.yaml", "name=unet_r18_s0", sub="robustness"),
            2.0,
            ["../unet_r18_s0/robustness.json"],
        )
    )
    R.append(
        Run(
            "dino_seg_s0_robustness",
            "t2",
            cli("configs/segment/dino_seg_vits14.yaml", "name=dino_seg_s0", sub="robustness"),
            2.0,
            ["../dino_seg_s0/robustness.json"],
        )
    )
    # ------------------------------------------------------------------ tier 3: single-seed longer baselines
    R.append(
        Run(
            "convnext_tiny_full_s0",
            "t3",
            cli("configs/classify/convnext_tiny_full.yaml", "name=convnext_tiny_full_s0", "seed=0"),
            3.5,
            ["test_metrics.json"],
            "1 seed",
        )
    )
    R.append(
        Run(
            "unet_scratch_s0",
            "t3",
            cli("configs/segment/unet.yaml", "name=unet_scratch_s0", "seed=0"),
            3.5,
            ["test_metrics.json"],
            "1 seed",
        )
    )
    return R


def is_done(r: Run) -> bool:
    base = ROOT / "outputs" / r.name
    return bool(r.done_files) and all((base / f).resolve().exists() for f in r.done_files)


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def resources(r: Run) -> dict:
    sub = "robustness" if r.name.endswith("_robustness") else "run"
    name = r.name.replace("_robustness", "")
    p = ROOT / "outputs" / name / f"resources_{sub}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", nargs="*", default=["cpu", "t1", "t2", "t3"])
    ap.add_argument("--only", nargs="*", default=None, help="substring filter on run names")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    runs = [r for r in plan() if r.tier in a.tier and (not a.only or any(o in r.name for o in a.only))]
    if a.list:
        for r in runs:
            print(
                f"{'DONE ' if is_done(r) else 'todo '} {r.tier:3s} {r.gb if r.gb else 'cpu':>4} GB  {r.name}"
            )
        print(f"{sum(is_done(r) for r in runs)}/{len(runs)} done")
        return
    log(f"CAMPAIGN start: {len(runs)} runs ({sum(is_done(r) for r in runs)} already done), tiers {a.tier}")
    for r in runs:
        if is_done(r):
            continue
        cpu_only = r.gb is None and not r.self_guarded
        env = dict(
            os.environ,
            PYTHONIOENCODING="utf-8",
            MLFLOW_DISABLE_AGENT_HINT="1",
            **(CPU_ENV if cpu_only else {}),
        )
        cmd = r.cmd if r.gb is None else guarded_cmd(r.cmd, r.gb, f"semdefect campaign {r.name}")
        reserve = r.gb or ("self-guarded" if r.self_guarded else "cpu")
        wait_for_ram(r.name)
        log(f"START {r.name} tier={r.tier} reserve_gb={reserve} free_ram={free_ram_gb():.1f}GB {r.note}")
        t0 = time.time()
        out_log = ROOT / "outputs" / "campaign_logs" / f"{r.name}.log"
        out_log.parent.mkdir(parents=True, exist_ok=True)
        with open(out_log, "a", encoding="utf-8") as f:
            f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
            f.flush()
            rc = subprocess.call(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        wall = time.time() - t0
        res = resources(r)
        status = "OK" if rc == 0 and is_done(r) else f"FAILED(rc={rc})"
        log(
            f"END {r.name} {status} wall={wall / 60:.1f}min run_wall={res.get('wall_s', '?')}s "
            f"peak_alloc={res.get('peak_alloc_mib', '-')}MiB peak_reserved={res.get('peak_reserved_mib', '-')}MiB "
            f"reserved=({res.get('gpu_guard_gb', '-')} GB) log={out_log.name}"
        )
    done = sum(is_done(r) for r in runs)
    log(f"CAMPAIGN end: {done}/{len(runs)} done")


if __name__ == "__main__":
    main()
