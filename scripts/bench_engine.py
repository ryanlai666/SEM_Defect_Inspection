"""PatchCore deployment benchmark: PyTorch FP32/FP16 vs TensorRT FP32/FP16/INT8 (+ custom kernels).

Memory bank from `--n-normal` train normals, evaluation on a stratified `--n-test` subset of the test
split (the reported run uses all 136 normals and all 919 test images).  Every precision variant uses the *same* FP32 memory
bank, so AUROC / score deltas isolate the embedder precision.  Also writes the artefacts consumed by
the C++ runner (engine, bank, raw images).

    python scripts/guarded.py --gb 3 -- .venv/Scripts/python.exe scripts/bench_engine.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from semdefect.data import CarinthiaS, anomaly_split, load_index, load_or_make_split, read_gray  # noqa: E402
from semdefect.data.carinthia import to_model_input  # noqa: E402
from semdefect.engine.common import subsample  # noqa: E402
from semdefect.export.onnx_export import export_efficientad, export_patchcore_embedder  # noqa: E402
from semdefect.export.trt import TRTModule, build_engine, onnx_size_mb  # noqa: E402
from semdefect.kernels import ext  # noqa: E402
from semdefect.kernels.knn import knn, knn_torch_fp16  # noqa: E402
from semdefect.metrics import image_auroc  # noqa: E402
from semdefect.models import PatchCore  # noqa: E402
from semdefect.utils import apply_gpu_guard_cap, gpu_guard_gb, save_json  # noqa: E402

OUT = ROOT / "outputs" / "export"
S = 256


ROUNDS = 3  # independent timing rounds (single measurements on a shared GPU are noisy)


def time_cuda(fn, repeats=30, warmup=5) -> dict:
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
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-normal", type=int, default=32)
    ap.add_argument("--n-test", type=int, default=120)
    ap.add_argument("--skip-int8", action="store_true")
    ap.add_argument(
        "--out-tag", default="", help="write outputs/export_<tag>/ and outputs/bench/engine_<tag>.json"
    )
    a = ap.parse_args()
    global OUT
    if a.out_tag:
        OUT = ROOT / "outputs" / f"export_{a.out_tag}"
    OUT.mkdir(parents=True, exist_ok=True)
    out_json = ROOT / "outputs" / "bench" / (f"engine_{a.out_tag}.json" if a.out_tag else "engine.json")
    g = gpu_guard_gb()
    torch_cap = apply_gpu_guard_cap(
        (g - 0.4) / 2 if g else 0
    )  # other half of the reservation: TRT workspace/engines
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda")
    res: dict = {
        "env": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "reservation_gb": g,
            "torch_allocator_cap_gb": torch_cap,
        }
    }
    import tensorrt as trt

    res["env"]["tensorrt"] = trt.__version__

    # ---------------------------------------------------------------- data (bank + test subset)
    df = load_index()
    split = load_or_make_split(df, seed=0)
    asp = anomaly_split(df, split)
    lab = df.set_index("idx")["label"]
    normals = asp["train_normal"][: a.n_normal]
    test_idx = subsample(asp["test"], a.n_test, seed=0, labels=[lab[i] for i in asp["test"]])
    cache = "disk"
    ds_train = CarinthiaS(df, normals, img_size=S, cache=cache, load_masks=False)
    # float path (no uint8 cache) for test images: bit-compatible with the C++ runner preprocessing
    ds_test = CarinthiaS(df, test_idx, img_size=S, cache=False, load_masks=False)
    x_train = torch.stack([ds_train[i]["image"] for i in range(len(ds_train))])
    x_test = torch.stack([ds_test[i]["image"] for i in range(len(ds_test))])
    y_test = np.array([int(lab[i] != 6) for i in test_idx])
    res["data"] = {
        "n_bank_images": len(normals),
        "n_test": len(test_idx),
        "n_test_normal": int((y_test == 0).sum()),
    }

    # ---------------------------------------------------------------- reference PyTorch FP32 PatchCore
    pc = PatchCore(coreset_ratio=0.1, knn_backend="ext" if ext.available() else "torch").to(dev).eval()
    t0 = time.time()
    fit = pc.fit([{"image": x_train[i : i + 8]} for i in range(0, len(x_train), 8)], dev)
    res["fit"] = {**fit, "seconds": time.time() - t0}
    bank = pc.memory

    def scores_from_embedding(emb: torch.Tensor) -> torch.Tensor:
        b, n, d = emb.shape
        dist, _ = knn(emb.reshape(-1, d).float().contiguous(), bank, 1, pc.knn_backend)
        return dist[:, 0].clamp_min(0).sqrt().reshape(b, n).amax(1)

    @torch.no_grad()
    def run_all(embed_fn, bs=8) -> np.ndarray:
        out = []
        for i in range(0, len(x_test), bs):
            out.append(scores_from_embedding(embed_fn(x_test[i : i + bs].to(dev))).cpu())
        return torch.cat(out).numpy()

    emb_t = pc.embedder

    def torch_fp32(x):
        return emb_t(x)

    def torch_fp16(x):
        with torch.autocast("cuda", dtype=torch.float16):
            return emb_t(x).float()

    variants = {"torch_fp32": torch_fp32, "torch_amp_fp16": torch_fp16}

    # ---------------------------------------------------------------- ONNX + TensorRT engines
    onnx_path = export_patchcore_embedder(OUT / "patchcore_wrn50_256.onnx", S)
    res["onnx_mb"] = onnx_size_mb(onnx_path)
    calib = [x_train[i : i + 8] for i in range(0, len(x_train), 8)]
    precisions = ["fp32", "fp16"] + ([] if a.skip_int8 else ["int8"])
    engines = {}
    for p in precisions:
        t0 = time.time()
        eng = build_engine(
            onnx_path,
            OUT / f"patchcore_wrn50_256_{p}.engine",
            S,
            p,
            max_batch=8,
            opt_batch=8,
            calib_batches=calib if p == "int8" else None,
        )
        engines[p] = TRTModule(eng)
        res.setdefault("build_seconds", {})[p] = round(time.time() - t0, 1)
        res.setdefault("engine_mb", {})[p] = round(eng.stat().st_size / 2**20, 1)
        variants[f"trt_{p}"] = engines[p]

    # ---------------------------------------------------------------- accuracy parity on the subset
    ref = run_all(torch_fp32)
    acc = {}
    for name, fn in variants.items():
        s = ref if name == "torch_fp32" else run_all(fn)
        acc[name] = {
            "image_auroc": image_auroc(y_test, s),
            "auroc_delta_vs_torch_fp32": image_auroc(y_test, s) - image_auroc(y_test, ref),
            "score_pearson": float(np.corrcoef(s, ref)[0, 1]),
            "score_max_rel_diff": float(np.max(np.abs(s - ref) / np.maximum(np.abs(ref), 1e-6))),
        }
        print(name, acc[name], flush=True)
    if ext.available() and "fp16" in engines:  # TRT FP16 embedder + tensor-core FP16 kNN (full FP16 path)
        b16 = bank.half().contiguous()
        b16n = ext.row_sqnorm_f16_ext(b16)
        out = []
        with torch.no_grad():
            for i in range(0, len(x_test), 8):
                e = engines["fp16"](x_test[i : i + 8].to(dev))
                b_, n_, d_ = e.shape
                dist, _ = ext.knn_top1_fp16_ext(e.reshape(-1, d_), bank, 1, bank16=b16, bank_norm=b16n)
                out.append(dist[:, 0].clamp_min(0).sqrt().reshape(b_, n_).amax(1).cpu())
        s = torch.cat(out).numpy()
        acc["trt_fp16+wmma_fp16_knn"] = {
            "image_auroc": image_auroc(y_test, s),
            "auroc_delta_vs_torch_fp32": image_auroc(y_test, s) - image_auroc(y_test, ref),
            "score_pearson": float(np.corrcoef(s, ref)[0, 1]),
            "score_max_rel_diff": float(np.max(np.abs(s - ref) / np.maximum(np.abs(ref), 1e-6))),
        }
        print("trt_fp16+wmma_fp16_knn", acc["trt_fp16+wmma_fp16_knn"], flush=True)
    res["accuracy_subset"] = acc

    # ---------------------------------------------------------------- latency: embedder only
    lat = {}
    for bs in (1, 8):
        x = x_test[:bs].to(dev)
        for name, fn in variants.items():
            torch.cuda.reset_peak_memory_stats()
            t = time_cuda(lambda fn=fn, x=x: fn(x))
            lat[f"{name}_bs{bs}"] = {
                **t,
                "img_per_s": bs / (t["p50_ms"] / 1e3),
                "torch_peak_alloc_mib": torch.cuda.max_memory_allocated() / 2**20,
            }
            print(
                f"embedder {name:15s} bs={bs}: p50 {t['p50_ms']:.2f} ms p99 {t['p99_ms']:.2f} ms", flush=True
            )
    res["latency_embedder"] = lat

    # ---------------------------------------------------------------- latency: end-to-end from raw uint8
    raw = torch.from_numpy(np.stack([read_gray(df.at[i, "image_path"]) for i in test_idx[:8]])).to(dev)

    def baseline(raw_b):  # torch ops preprocessing + torch FP32 embedder + cdist/topk
        x = raw_b[:, 1:478, 0:477].float().unsqueeze(1) / 255.0
        x = to_model_input(torch.nn.functional.adaptive_avg_pool2d(x, S))
        e = emb_t(x)
        b, n, d = e.shape
        dist, _ = knn(e.reshape(-1, d), bank, 1, "torch")
        return dist[:, 0].sqrt().reshape(b, n).amax(1)

    bank16_t = bank.half().contiguous()
    bank16_tn = bank16_t.float().square().sum(1)

    def baseline_fp16(raw_b):  # precision-matched PyTorch baseline: AMP FP16 embedder + cuBLAS FP16 kNN
        x = raw_b[:, 1:478, 0:477].float().unsqueeze(1) / 255.0
        x = to_model_input(torch.nn.functional.adaptive_avg_pool2d(x, S))
        with torch.autocast("cuda", dtype=torch.float16):
            e = emb_t(x)
        b, n, d = e.shape
        dist, _ = knn_torch_fp16(e.reshape(-1, d), None, 1, bank16=bank16_t, bank_norm=bank16_tn)
        return dist[:, 0].sqrt().reshape(b, n).amax(1)

    e2e = {}
    if ext.available():
        bank16 = bank.half().contiguous()
        bank16_norm = ext.row_sqnorm_f16_ext(bank16)
    for bs in (1, 8):
        rb = raw[:bs].contiguous()
        e2e[f"baseline_torch_fp32_bs{bs}"] = time_cuda(lambda rb=rb: baseline(rb))
        e2e[f"baseline_torch_amp_fp16_bs{bs}"] = time_cuda(lambda rb=rb: baseline_fp16(rb))
        if ext.available():
            for p in ("fp16", "int8"):
                if p not in engines:
                    continue

                def opt(rb=rb, p=p):  # fused preprocess kernel + TRT + fused kNN kernel
                    x = ext.preprocess_ext(rb, S, (1, 0, 477, 477))
                    return scores_from_embedding(engines[p](x))

                e2e[f"kernels+trt_{p}_bs{bs}"] = time_cuda(opt)

                def opt_tc(rb=rb, p=p):  # ... with the tensor-core FP16 top-1 kNN instead
                    x = ext.preprocess_ext(rb, S, (1, 0, 477, 477))
                    e = engines[p](x)
                    b_, n_, d_ = e.shape
                    dist, _ = ext.knn_top1_fp16_ext(
                        e.reshape(-1, d_), bank, 1, bank16=bank16, bank_norm=bank16_norm
                    )
                    return dist[:, 0].clamp_min(0).sqrt().reshape(b_, n_).amax(1)

                e2e[f"kernels(wmma_knn)+trt_{p}_bs{bs}"] = time_cuda(opt_tc)
        for k, v in e2e.items():
            v["img_per_s"] = int(k.split("_bs")[-1]) / (v["p50_ms"] / 1e3)
    res["latency_end_to_end"] = e2e
    print(json.dumps(e2e, indent=1), flush=True)

    # ---------------------------------------------------------------- artefacts for the C++ runner
    bank_cpu = bank.float().cpu().numpy()
    with open(OUT / "patchcore_bank.bin", "wb") as f:
        np.array(bank_cpu.shape, dtype=np.int32).tofile(f)
        bank_cpu.tofile(f)
    raw_all = np.stack([read_gray(df.at[i, "image_path"]) for i in test_idx])
    with open(OUT / "test_raw_u8.bin", "wb") as f:
        np.array(raw_all.shape, dtype=np.int32).tofile(f)
        raw_all.tofile(f)
    np.save(OUT / "test_labels.npy", y_test)
    np.save(OUT / "test_scores_torch_fp32.npy", ref)
    np.save(OUT / "test_scores_torch_fp32_floatpath.npy", ref)  # test images use the float path

    # ---------------------------------------------------------------- EfficientAD: export + FP16 latency
    try:
        ea_onnx = export_efficientad(OUT / "efficientad_s_256.onnx", img_size=S)
        ea = TRTModule(build_engine(ea_onnx, OUT / "efficientad_s_256_fp16.engine", S, "fp16"))
        for bs in (1, 8):
            x = x_test[:bs].to(dev)
            t = time_cuda(lambda x=x: ea(x))
            res.setdefault("efficientad_trt_fp16_untrained_latency", {})[f"bs{bs}"] = {
                **t,
                "img_per_s": bs / (t["p50_ms"] / 1e3),
            }
    except Exception as e:  # pragma: no cover
        res["efficientad_export_error"] = repr(e)
    save_json(res, out_json)
    print(f"saved {out_json}")


if __name__ == "__main__":
    main()
