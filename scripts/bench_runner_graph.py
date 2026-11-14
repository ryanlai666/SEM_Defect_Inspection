"""C++ runner with vs without CUDA-graph replay (FP16 engine + tensor-core kNN), batch 1 and 8.

python scripts/guarded.py --gb 1.0 --smoke -- .venv/Scripts/python.exe scripts/bench_runner_graph.py [--tag full]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "cpp" / "build" / ("trt_runner.exe" if os.name == "nt" else "trt_runner")
TRT_DIR = ROOT / ".venv" / "Lib" / "site-packages" / "tensorrt_libs"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--iters", type=int, default=400)
    a = ap.parse_args()
    exp = ROOT / "outputs" / (f"export_{a.tag}" if a.tag else "export")
    rows = []
    for bs in (1, 8):
        for graph in (False, True):
            tag = f"fp16_b{bs}_s1_knnfp16{'_graph' if graph else ''}"
            out = ROOT / "outputs" / "bench" / f"runner_graphtest{'_' + a.tag if a.tag else ''}_{tag}.json"
            cmd = [
                str(EXE),
                "--engine",
                str(exp / "patchcore_wrn50_256_fp16.engine"),
                "--bank",
                str(exp / "patchcore_bank.bin"),
                "--images",
                str(exp / "test_raw_u8.bin"),
                "--trt-dir",
                str(TRT_DIR),
                "--batch",
                str(bs),
                "--streams",
                "1",
                "--iters",
                str(a.iters),
                "--knn",
                "fp16",
                "--json",
                str(out),
            ] + (["--cuda-graph"] if graph else [])
            reps = []
            for _ in range(3):  # 3 independent process launches
                subprocess.run(cmd, check=True, capture_output=True)
                reps.append(json.loads(out.read_text()))
            r = dict(reps[0])
            for key in ("latency_ms_p50", "latency_ms_p99", "throughput_img_per_s"):
                v = [x[key] for x in reps]
                r[key] = sum(v) / len(v)
                r[key + "_std"] = (sum((t - r[key]) ** 2 for t in v) / (len(v) - 1)) ** 0.5
                r[key + "_runs"] = v
            r["graph_vs_direct_max_abs_score_diff"] = max(
                x.get("graph_vs_direct_max_abs_score_diff") or 0 for x in reps
            )
            out.write_text(json.dumps(r, indent=1))
            rows.append(r)
            print(
                f"{tag:32s} p50 {r['latency_ms_p50']:.3f} ± {r['latency_ms_p50_std']:.3f} ms  "
                f"p99 {r['latency_ms_p99']:.3f} ms  {r['throughput_img_per_s']:.0f} img/s  "
                f"graph-vs-direct |dscore| {r['graph_vs_direct_max_abs_score_diff']}"
            )
    (ROOT / "outputs" / "bench" / f"runner_graph_summary{'_' + a.tag if a.tag else ''}.json").write_text(
        json.dumps(rows, indent=1)
    )


if __name__ == "__main__":
    main()
