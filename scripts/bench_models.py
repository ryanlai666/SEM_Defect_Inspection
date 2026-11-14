"""Steady-state forward throughput of the frozen backbones / inference graphs (no training, no data
loading): basis for feature-extraction and memory-bank fitting time estimates.

    python scripts/guarded.py --gb 2.5 -- .venv/Scripts/python.exe scripts/bench_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from semdefect.models import DinoV2, PatchCoreEmbedder, count_params  # noqa: E402
from semdefect.utils import apply_gpu_guard_cap, save_json  # noqa: E402


def bench(fn, x, iters=20, warmup=5) -> float:
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(warmup):
            fn(x)
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            fn(x)
        b.record()
        b.synchronize()
    return x.shape[0] * iters / (a.elapsed_time(b) / 1e3)


def main() -> None:
    cap = apply_gpu_guard_cap()
    torch.backends.cudnn.benchmark = True
    rows = []
    cases = [
        ("dinov2_vits14", 224, 64),
        ("dinov2_vitb14", 224, 32),
        ("dinov2_vits14", 448, 16),
        ("dinov2_vits14_reg", 280, 32),
        ("wrn50_patchcore_embed", 256, 16),
    ]
    for name, size, bs in cases:
        if name.startswith("wrn50"):
            m = PatchCoreEmbedder("wide_resnet50_2", (2, 3), 1024, pretrained=False).cuda().eval()
            fn = m
        else:
            m = DinoV2(name, img_size=size, pretrained=False).cuda().eval()
            fn = m.global_features if size == 224 else (lambda x, m=m: m.tokens(x))
        x = torch.randn(bs, 3, size, size, device="cuda")
        torch.cuda.reset_peak_memory_stats()
        ips = bench(fn, x)
        rows.append(
            dict(
                model=name,
                img_size=size,
                batch=bs,
                fp16_autocast=True,
                img_per_s=round(ips, 1),
                params=count_params(m),
                peak_alloc_mib=round(torch.cuda.max_memory_allocated() / 2**20, 1),
            )
        )
        print(rows[-1], flush=True)
        del m, x
        torch.cuda.empty_cache()
    save_json(
        {"allocator_cap_gb": cap, "gpu": torch.cuda.get_device_name(0), "rows": rows},
        ROOT / "outputs/bench/models.json",
    )


if __name__ == "__main__":
    main()
