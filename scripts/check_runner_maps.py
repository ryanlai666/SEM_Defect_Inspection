"""Parity of the C++ runner's anomaly maps vs the PyTorch PatchCore reference (pixel level).

Runs trt_runner (FP32 engine + FP32 kNN, and FP16 engine + tensor-core kNN) with --maps-out on the
images exported by bench_engine.py, recomputes the maps in PyTorch FP32 from the same memory bank
(float preprocessing path), and compares max |diff|, pixel AUROC and AUPRO.  Run through the guard:

    python scripts/guarded.py --gb 1.5 --smoke -- .venv/Scripts/python.exe scripts/check_runner_maps.py [--tag full]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from semdefect.data import CarinthiaS, anomaly_split, load_index, load_or_make_split  # noqa: E402
from semdefect.engine.common import subsample  # noqa: E402
from semdefect.kernels.knn import knn  # noqa: E402
from semdefect.metrics import aupro, binned_auroc, binned_average_precision  # noqa: E402
from semdefect.models import PatchCoreEmbedder  # noqa: E402
from semdefect.models.anomaly import smooth_maps  # noqa: E402
from semdefect.utils import apply_gpu_guard_cap, save_json  # noqa: E402

EXE = ROOT / "cpp" / "build" / ("trt_runner.exe" if os.name == "nt" else "trt_runner")
TRT_DIR = ROOT / ".venv" / "Lib" / "site-packages" / "tensorrt_libs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--n", type=int, default=120, help="number of images with maps")
    ap.add_argument("--n-test", type=int, default=120, help="must match the bench_engine.py subset")
    a = ap.parse_args()
    apply_gpu_guard_cap()
    exp = ROOT / "outputs" / (f"export_{a.tag}" if a.tag else "export")
    S, MS = 256, 224
    df = load_index()
    split = load_or_make_split(df, seed=0)
    asp = anomaly_split(df, split)
    lab = df.set_index("idx")["label"]
    test_idx = subsample(asp["test"], a.n_test, seed=0, labels=[lab[i] for i in asp["test"]])[: a.n]
    ds = CarinthiaS(df, test_idx, img_size=S, cache=False, mask_size=MS)
    masks = np.stack([ds[i]["mask"][0].numpy() for i in range(len(ds))]).astype(bool)
    x = torch.stack([ds[i]["image"] for i in range(len(ds))]).cuda()

    with open(exp / "patchcore_bank.bin", "rb") as f:
        n, d = np.fromfile(f, np.int32, 2)
        bank = torch.from_numpy(np.fromfile(f, np.float32).reshape(n, d)).cuda()
    emb = PatchCoreEmbedder().cuda().eval()
    ref = []
    with torch.no_grad():
        for i in range(0, len(x), 8):
            e = emb(x[i : i + 8])
            b, p, dd = e.shape
            dist, _ = knn(e.reshape(-1, dd), bank, 1, "torch")
            ref.append(smooth_maps(dist[:, 0].clamp_min(0).sqrt().reshape(b, 32, 32), MS, 4.0).cpu())
    ref = torch.cat(ref).numpy()
    del emb, x
    torch.cuda.empty_cache()

    res = {
        "n_images": len(test_idx),
        "map_size": MS,
        "torch_fp32": {
            "pixel_auroc": binned_auroc(masks, ref),
            "aupro_30": aupro(masks, ref),
            "aupro_05": aupro(masks, ref, 0.05),
            "pixel_ap": binned_average_precision(masks, ref),
        },
    }
    for prec, knn_kind in (("fp32", "fp32"), ("fp16", "fp16")):
        out = (
            ROOT / "outputs" / "bench" / f"runner_maps_{prec}_knn{knn_kind}{'_' + a.tag if a.tag else ''}.bin"
        )
        cmd = [
            str(EXE),
            "--engine",
            str(exp / f"patchcore_wrn50_256_{prec}.engine"),
            "--bank",
            str(exp / "patchcore_bank.bin"),
            "--images",
            str(exp / "test_raw_u8.bin"),
            "--trt-dir",
            str(TRT_DIR),
            "--batch",
            "8",
            "--streams",
            "1",
            "--iters",
            "20",
            "--knn",
            knn_kind,
            "--maps-out",
            str(out),
            "--maps-n",
            str(len(test_idx)),
            "--map-size",
            str(MS),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        got = np.fromfile(out, np.float32).reshape(-1, MS, MS)[: len(test_idx)]
        res[f"cpp_{prec}_knn{knn_kind}"] = {
            "max_abs_diff": float(np.abs(got - ref).max()),
            "mean_abs_diff": float(np.abs(got - ref).mean()),
            "rel_max_abs_diff": float(np.abs(got - ref).max() / np.abs(ref).max()),
            "pixel_auroc": binned_auroc(masks, got),
            "aupro_30": aupro(masks, got),
            "aupro_05": aupro(masks, got, 0.05),
            "pixel_ap": binned_average_precision(masks, got),
        }
    print(json.dumps(res, indent=1))
    save_json(res, ROOT / "outputs" / "bench" / f"runner_maps_parity{'_' + a.tag if a.tag else ''}.json")


if __name__ == "__main__":
    main()
