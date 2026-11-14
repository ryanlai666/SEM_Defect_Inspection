"""Render docs/benchmarks.md from outputs/bench/{kernels,engine,models,runner_*}.json.

python scripts/report_bench.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / "outputs" / "bench"


def load(name):
    p = B / name
    return json.loads(p.read_text()) if p.exists() else None


def _pm(t: dict) -> str:
    sd = t.get("p50_std_ms")
    return f"{t['p50_ms']:.2f}" + (f" ± {sd:.2f}" if sd is not None else "")


def decomposition(k: dict | None, g: dict | None) -> list[str]:
    """Split every speed-up into a precision gain (PyTorch FP32 -> PyTorch FP16) and an engineering gain
    (PyTorch FP16 -> custom kernels / TensorRT)."""
    L = [
        "## Speed-up decomposition: precision gain vs engineering gain",
        "",
        "precision gain = PyTorch FP32 / PyTorch FP16 (same algorithm, half precision via AMP / cuBLAS FP16); "
        "engineering gain = PyTorch FP16 / ours (custom kernels, TensorRT). p50 latency in ms "
        "(mean ± sd over 3 timing rounds when available).",
        "",
    ]
    rows = []
    if k:
        by = {}
        for r in k["knn"]:
            by.setdefault((r["case"], r["N"], r["D"]), {})[r["backend"].split("(")[0]] = r
        for (case, n, d), b in by.items():
            a, f16, ours = (
                b.get("torch_cdist_topk"),
                b.get("torch_fp16_mm_topk"),
                b.get("nvcc_wmma_fp16_top1"),
            )
            if a and f16 and ours:
                rows.append((f"kNN {case}, N={n // 1000}k, D={d}", a, f16, ours))
    if g:
        e = g["latency_end_to_end"]
        for bs in (1, 8):
            a, f16 = e.get(f"baseline_torch_fp32_bs{bs}"), e.get(f"baseline_torch_amp_fp16_bs{bs}")
            ours = e.get(f"kernels(wmma_knn)+trt_fp16_bs{bs}")
            if a and f16 and ours:
                rows.append((f"PatchCore end-to-end, batch {bs}", a, f16, ours))
    if not rows:
        return []
    L += [
        "| measurement | PyTorch FP32 | PyTorch FP16 | ours (FP16) | precision gain | engineering gain | total |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, a, f16, ours in rows:
        L.append(
            f"| {name} | {_pm(a)} | {_pm(f16)} | {_pm(ours)} | {a['p50_ms'] / f16['p50_ms']:.2f}× | "
            f"{f16['p50_ms'] / ours['p50_ms']:.2f}× | {a['p50_ms'] / ours['p50_ms']:.2f}× |"
        )
    return L + [""]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows GBK consoles
    L = ["# Benchmarks (measured on this machine; regenerate with `python scripts/report_bench.py`)", ""]
    k = load("kernels_v2.json") or load("kernels.json")  # v2: FP16 baseline + 3 rounds
    if k:
        e = k["env"]
        L += [
            f"GPU: {e['gpu']} · torch {e['torch']} (CUDA {e['cuda']}) · TF32 off · {e['timing']}",
            "",
            "The GPU was shared with other jobs (all launched through the same VRAM guard), so p99",
            "tails can include contention; p50 is the robust statistic (mean ± sd over 3 timing rounds where shown).",
            "",
            "## kNN top-1 over a memory bank (FP32 backends + FP16 baseline and tensor-core kernel)",
            "",
            "| case | M queries | N bank | D | backend | p50 ms | p99 ms | TFLOP/s | extra VRAM MiB | idx agree |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in k["knn"]:
            L.append(
                f"| {r['case']} | {r['M']} | {r['N']} | {r['D']} | {r['backend']} | {r['p50_ms']:.2f}"
                + (f" ± {r['p50_std_ms']:.2f}" if "p50_std_ms" in r else "")
                + f" | {r['p99_ms']:.2f} | "
                f"{r['tflops']:.2f} | {r['peak_extra_mib']:.0f} | {r['index_agreement']:.4f} |"
            )
        L += [
            "",
            "## Fused SEM preprocessing (480x480 uint8 -> crop 477 -> area-resize -> normalise -> 3ch)",
            "",
            "| out size | batch | backend | p50 ms | p99 ms | img/s | extra VRAM MiB | max abs err |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in k["preprocess"]:
            L.append(
                f"| {r['out_size']} | {r['batch']} | {r['backend']} | {r['p50_ms']:.3f}"
                + (f" ± {r['p50_std_ms']:.3f}" if "p50_std_ms" in r else "")
                + f" | {r['p99_ms']:.3f} | {r['img_per_s']:.0f} | "
                f"{r['peak_extra_mib']:.1f} | {r['max_abs_err']:.1e} |"
            )
        L.append("")
    m = load("models.json")
    if m:
        L += [
            "## Steady-state forward throughput (FP16 autocast, synthetic input, no data loading)",
            "",
            "| model | input | batch | img/s | params | peak alloc MiB |",
            "|---|---|---|---|---|---|",
        ]
        L += [
            f"| {r['model']} | {r['img_size']} | {r['batch']} | {r['img_per_s']:.0f} | {r['params'] / 1e6:.1f} M | {r['peak_alloc_mib']:.0f} |"
            for r in m["rows"]
        ] + [""]
    g = load("engine_full.json") or load("engine.json")  # full: 136-normal bank, 919 images
    if g:
        d = g["data"]
        L += [
            "## PatchCore deployment: PyTorch vs TensorRT (+ custom kernels)",
            "",
            f"TensorRT {g['env']['tensorrt']}. Memory bank from {d['n_bank_images']} train normals "
            f"({g['fit']['memory_size']} patches after 10 % coreset); accuracy on "
            + (
                f"the full {d['n_test']}-image test split ({d['n_test_normal']} normal)"
                if d["n_test"] >= 919
                else f"a stratified {d['n_test']}-image test subset ({d['n_test_normal']} normal)"
            )
            + ". All variants share the FP32 bank.",
            "",
            "| embedder | image AUROC | ΔAUROC vs torch FP32 | score Pearson r | max rel. score diff |",
            "|---|---|---|---|---|",
        ]
        for n, a in g["accuracy_subset"].items():
            L.append(
                f"| {n} | {a['image_auroc']:.4f} | {a['auroc_delta_vs_torch_fp32']:+.4f} | {a['score_pearson']:.7f} | {a['score_max_rel_diff']:.2e} |"
            )
        L += ["", "| embedder only | batch | p50 ms | p99 ms | img/s |", "|---|---|---|---|---|"]
        for n, t in g["latency_embedder"].items():
            name, bs = n.rsplit("_bs", 1)
            L.append(f"| {name} | {bs} | {t['p50_ms']:.2f} | {t['p99_ms']:.2f} | {t['img_per_s']:.0f} |")
        L += [
            "",
            "| end-to-end from raw uint8 on GPU | batch | p50 ms | p99 ms | img/s |",
            "|---|---|---|---|---|",
        ]
        for n, t in g["latency_end_to_end"].items():
            name, bs = n.rsplit("_bs", 1)
            L.append(f"| {name} | {bs} | {t['p50_ms']:.2f} | {t['p99_ms']:.2f} | {t['img_per_s']:.0f} |")
        L += [
            "",
            f"Engine sizes (MB): {g.get('engine_mb')}; build seconds: {g.get('build_seconds')}; ONNX {g.get('onnx_mb', 0):.0f} MB.",
            "",
        ]
        if "efficientad_trt_fp16_untrained_latency" in g:
            L += [
                "EfficientAD-S TensorRT FP16 latency (untrained weights, latency only): "
                + ", ".join(
                    f"{k}: p50 {v['p50_ms']:.2f} ms ({v['img_per_s']:.0f} img/s)"
                    for k, v in g["efficientad_trt_fp16_untrained_latency"].items()
                ),
                "",
            ]
    # full test split, 3 repeats; the earlier 120-image files only if the full set is missing
    runs = sorted(B.glob("runner_full_*.json"))
    full = bool(runs)
    if not runs:
        runs = sorted(
            p
            for p in B.glob("runner_*.json")
            if not p.name.startswith(("runner_maps", "runner_graph", "runner_full"))
        )

    def ms(r, k, f=".2f"):
        sd = r.get(k + "_std")
        return f"{r[k]:{f}} ± {sd:{f}}" if sd is not None else f"{r[k]:{f}}"

    if runs:
        L += [
            "## C++ runner (trt_runner.exe): pinned H2D -> fused preprocess -> TensorRT -> fused kNN -> max -> D2H",
            "",
            (
                "All 919 test images, PatchCore bank of the 136 class-6 train images; mean ± sd over 3 repeated runs "
                "(each: p50 / p99 over the runner's iterations)."
                if full
                else "120-image test subset, single runs."
            ),
            "",
            "| engine | kNN kernel | batch | streams | p50 ms / batch | p99 ms / batch | img/s | runner VRAM MiB | parity vs PyTorch FP32 |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for p in runs:
            r = json.loads(p.read_text())
            L.append(
                f"| {Path(r['engine']).stem} | {r.get('knn', 'fp32')} | {r['batch']} | {r['streams']} | {ms(r, 'latency_ms_p50')} | "
                f"{ms(r, 'latency_ms_p99')} | {ms(r, 'throughput_img_per_s', '.0f')} | {r['vram_used_by_runner_mib']:.0f} | "
                f"{r.get('parity', '—')} |"
            )
        L.append("")
    graph = load("runner_graph_summary_full.json") or load("runner_graph_summary.json")
    if isinstance(graph, list) and graph:
        L += [
            "## CUDA-graph replay in the runner (FP16 engine + tensor-core kNN)",
            "",
            "| batch | CUDA graph | p50 ms / batch | p99 ms / batch | img/s | max |score diff| graph vs direct |",
            "|---|---|---|---|---|---|",
        ]
        for r in graph:
            L.append(
                f"| {r['batch']} | {'yes' if r.get('cuda_graph') else 'no'} | {ms(r, 'latency_ms_p50', '.3f')} | "
                f"{ms(r, 'latency_ms_p99', '.3f')} | {ms(r, 'throughput_img_per_s', '.0f')} | "
                f"{r.get('graph_vs_direct_max_abs_score_diff', '—')} |"
            )
        L.append("")
    # the full-split run supersedes the earlier 120-image one
    for tag in ("_full",) if load("runner_maps_parity_full.json") else ("",):
        mp = load(f"runner_maps_parity{tag}.json")
        if not mp:
            continue
        L += [
            f"## C++ runner anomaly maps vs PyTorch ({mp['n_images']} test images, {mp['map_size']}² maps)",
            "",
            "Maps = sqrt -> bilinear upsample -> Gaussian blur (sigma 4, reflect), computed by CUDA kernels in the runner.",
            "",
            "| maps from | pixel AUROC | AUPRO@0.3 | AUPRO@0.05 | pixel AP | max abs diff vs PyTorch (relative) | mean abs diff |",
            "|---|---|---|---|---|---|---|",
        ]
        for k, v in mp.items():
            if isinstance(v, dict):
                diff = (
                    f"{v['max_abs_diff']:.4f} ({v['rel_max_abs_diff']:.2%})"
                    if "max_abs_diff" in v
                    else "reference"
                )
                mean = f"{v['mean_abs_diff']:.5f}" if "mean_abs_diff" in v else "—"
                extra = " | ".join(f"{v[m]:.5f}" if m in v else "—" for m in ("aupro_05", "pixel_ap"))
                L.append(
                    f"| {k} | {v['pixel_auroc']:.5f} | {v['aupro_30']:.5f} | {extra} | {diff} | {mean} |"
                )
        L.append("")
    L += decomposition(
        load("kernels_v2.json") or load("kernels.json"), load("engine_full.json") or load("engine.json")
    )
    (ROOT / "docs" / "benchmarks.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
