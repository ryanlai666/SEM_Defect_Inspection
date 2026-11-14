"""Kernel micro-benchmarks (not training): custom CUDA kernels vs PyTorch on the RTX 3070.

kNN top-1 over a memory bank (PatchCore D=1024, AnomalyDINO D=384) and fused SEM preprocessing.
Timing: CUDA events, 3 warm-ups, median/p99 of `--repeats` runs.  Memory: peak *extra* allocation
above the inputs (torch.cuda.max_memory_allocated delta).  Run through the GPU guard:

    python scripts/guarded.py --gb 2.5 -- .venv/Scripts/python.exe scripts/bench_kernels.py
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from semdefect.data.carinthia import to_model_input  # noqa: E402
from semdefect.kernels import ext  # noqa: E402
from semdefect.kernels.cupy_kernels import knn_topk_cupy, preprocess_cupy, row_sqnorm_cupy  # noqa: E402
from semdefect.kernels.knn import knn_torch, knn_torch_fp16  # noqa: E402
from semdefect.utils import apply_gpu_guard_cap, save_json  # noqa: E402

ROUNDS = 3  # independent timing rounds per measurement (single runs on a shared GPU are noisy)


def time_cuda(fn, repeats: int, warmup: int = 3) -> dict:
    p50s, p99s = [], []
    for _ in range(ROUNDS):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(repeats):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            fn()
            b.record()
            b.synchronize()
            ts.append(a.elapsed_time(b))
        ts = np.array(ts)
        p50s.append(float(np.median(ts)))
        p99s.append(float(np.percentile(ts, 99)))
    return {
        "p50_ms": float(np.mean(p50s)),
        "p50_std_ms": float(np.std(p50s, ddof=1)),
        "p99_ms": float(np.mean(p99s)),
        "p50_rounds_ms": p50s,
        "rounds": ROUNDS,
    }


def peak_extra_mib(fn) -> float:
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20


def torch_chunk(n: int, budget_mb: int = 256) -> int:
    """Rows per cdist chunk so that one (chunk x N) FP32 distance tile stays within `budget_mb`."""
    return max(64, int(budget_mb * 2**20 // (4 * n)))


def bench_knn(repeats: int) -> list[dict]:
    cases = [  # (name, M queries, N bank, D)
        ("patchcore_1img", 1024, 10_000, 1024),
        ("patchcore_1img", 1024, 100_000, 1024),
        ("patchcore_8img", 8192, 10_000, 1024),
        ("patchcore_8img", 8192, 100_000, 1024),
        ("anomalydino_1img", 1024, 131_072, 384),
    ]
    rows = []
    for name, m, n, d in cases:
        g = torch.Generator(device="cuda").manual_seed(0)
        q = torch.randn(m, d, device="cuda", generator=g)
        bank = torch.randn(n, d, device="cuda", generator=g)
        bnorm = row_sqnorm_cupy(bank)
        chunk = torch_chunk(n)
        fns = {
            f"torch_cdist_topk(chunk={chunk})": lambda q=q, bank=bank, chunk=chunk: knn_torch(
                q, bank, 1, chunk
            ),
            "cupy_nvrtc_fused": lambda q=q, bank=bank, bnorm=bnorm: knn_topk_cupy(
                q, bank, 1, bank_norm=bnorm
            ),
        }
        b16_t = bank.half()
        b16_tn = b16_t.float().square().sum(1)
        fns["torch_fp16_mm_topk"] = lambda q=q, b16=b16_t, bn=b16_tn: knn_torch_fp16(
            q, None, 1, bank16=b16, bank_norm=bn
        )
        if ext.available():
            fns["nvcc_ext_fused"] = lambda q=q, bank=bank, bnorm=bnorm: ext.knn_topk_ext(
                q, bank, 1, bank_norm=bnorm
            )
            if d % 32 == 0:  # tensor cores: bank converted to FP16 once (as a deployed index would be)
                b16 = bank.half()
                b16n = ext.row_sqnorm_f16_ext(b16)
                fns["nvcc_wmma_fp16_top1"] = lambda q=q, bank=bank, b16=b16, b16n=b16n: ext.knn_top1_fp16_ext(
                    q, bank, 1, bank16=b16, bank_norm=b16n
                )
        ref_d, ref_i = knn_torch(q, bank, 1, chunk)
        for be, fn in fns.items():
            got_d, got_i = fn()
            agree = float((got_i == ref_i).float().mean())
            t = time_cuda(fn, repeats)
            mem = peak_extra_mib(fn)
            flop = 2.0 * m * n * d
            row = dict(
                case=name,
                M=m,
                N=n,
                D=d,
                k=1,
                backend=be,
                **t,
                peak_extra_mib=round(mem, 1),
                queries_per_s=m / (t["p50_ms"] / 1e3),
                tflops=flop / (t["p50_ms"] / 1e3) / 1e12,
                index_agreement=agree,
                max_rel_err=float(((got_d - ref_d).abs() / ref_d.clamp_min(1)).max()),
            )
            rows.append(row)
            print(
                f"{name:18s} M={m:5d} N={n:6d} D={d:4d} {be:28s} p50 {t['p50_ms']:8.2f} ms  p99 {t['p99_ms']:8.2f} ms  "
                f"{row['tflops']:5.2f} TFLOP/s  +{mem:7.1f} MiB  agree {agree:.4f}",
                flush=True,
            )
        del q, bank, bnorm
        torch.cuda.empty_cache()
    return rows


def torch_preprocess(raw: torch.Tensor, size: int) -> torch.Tensor:
    x = raw[:, 1:478, 0:477].float().unsqueeze(1) / 255.0
    return to_model_input(F.adaptive_avg_pool2d(x, size))


def bench_preprocess(repeats: int) -> list[dict]:
    rows = []
    for size in (256, 448):
        for b in (1, 8, 32):
            raw = torch.randint(0, 256, (b, 480, 480), dtype=torch.uint8, device="cuda")
            fns = {
                "torch_ops(crop,float,pool,repeat,norm)": lambda raw=raw, size=size: torch_preprocess(
                    raw, size
                ),
                "cupy_nvrtc_fused": lambda raw=raw, size=size: preprocess_cupy(raw, size, (1, 0, 477, 477)),
            }
            if ext.available():
                fns["nvcc_ext_fused"] = lambda raw=raw, size=size: ext.preprocess_ext(
                    raw, size, (1, 0, 477, 477)
                )
            ref = torch_preprocess(raw, size)
            for be, fn in fns.items():
                err = float((fn() - ref).abs().max())
                t = time_cuda(fn, repeats)
                mem = peak_extra_mib(fn)
                rows.append(
                    dict(
                        out_size=size,
                        batch=b,
                        backend=be,
                        **t,
                        peak_extra_mib=round(mem, 2),
                        img_per_s=b / (t["p50_ms"] / 1e3),
                        max_abs_err=err,
                    )
                )
                print(
                    f"preprocess {size} B={b:2d} {be:40s} p50 {t['p50_ms']:7.3f} ms p99 {t['p99_ms']:7.3f} ms "
                    f"+{mem:6.1f} MiB err {err:.1e}",
                    flush=True,
                )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--out", default="outputs/bench/kernels.json")
    a = ap.parse_args()
    cap = apply_gpu_guard_cap()
    torch.backends.cuda.matmul.allow_tf32 = False  # FP32 GEMM for an apples-to-apples FP32 kernel comparison
    res = {
        "env": dict(
            gpu=torch.cuda.get_device_name(0),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            platform=platform.platform(),
            allocator_cap_gb=cap,
            tf32=False,
            timing="CUDA events, 3 warm-ups, p50/p99 over repeats; inputs resident on GPU",
        ),
        "knn": bench_knn(a.repeats),
        "preprocess": bench_preprocess(a.repeats),
    }
    save_json(res, ROOT / a.out)


if __name__ == "__main__":
    main()
