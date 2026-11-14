"""Run a GPU command through an optional VRAM-reservation wrapper ("GPU guard") on a shared GPU,
or directly when none is configured (the default: Colab, CI, a dedicated workstation).

    python scripts/guarded.py --gb 2 --note "resnet18 smoke" -- python -m semdefect.cli run cfg.yaml --smoke

Configuration (environment variables):
* ``GPU_GUARD``      path to the wrapper script; unset or missing file -> run the command directly.
  The wrapper is called as ``python $GPU_GUARD run [$GPU_GUARD_ARGS] --gb <GB> --note <text> -- <cmd>``
  and is expected to wait until <GB> of VRAM is free, export ``GPU_GUARD_GB=<GB>`` and run the command.
* ``GPU_GUARD_ARGS`` extra arguments for the wrapper (shell-split), e.g. a job owner name.

Either way the wrapped process caps its own allocator to the reservation (``GPU_GUARD_GB``, set here
when running unwrapped; see ``semdefect.utils.apply_gpu_guard_cap``).  ``--gb`` is the honest peak
VRAM incl. ~0.4 GB of CUDA context.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def guard_path() -> Path | None:
    p = os.environ.get("GPU_GUARD", "").strip()
    return Path(p) if p and Path(p).is_file() else None


def guarded_cmd(
    cmd: list[str], gb: float, note: str = "", max_wait_min: float | None = None, smoke: bool = False
) -> list[str]:
    """Wrap `cmd` for the guard if one is configured, else return it unchanged.
    `smoke=True` marks a short test (<= 1.5 GB, < 10 min); never use it for real training runs."""
    guard = guard_path()
    if guard is None:
        return cmd
    g = [sys.executable, str(guard), "run", *shlex.split(os.environ.get("GPU_GUARD_ARGS", ""))]
    g += ["--gb", f"{gb:g}", "--note", note[:80]]
    if smoke:
        g.append("--smoke")
    if max_wait_min is not None:
        g += ["--max-wait-min", f"{max_wait_min:g}"]
    return g + ["--", *cmd]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gb", type=float, required=True, help="peak VRAM incl. CUDA context (GB)")
    ap.add_argument("--note", default="")
    ap.add_argument("--max-wait-min", type=float, default=None)
    ap.add_argument("--smoke", action="store_true", help="short test <= 1.5 GB / < 10 min")
    ap.add_argument("command", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not cmd:
        ap.error("missing command after --")
    if guard_path() is None:
        os.environ.setdefault("GPU_GUARD_GB", f"{a.gb:g}")  # still self-cap when unguarded
    return subprocess.call(guarded_cmd(cmd, a.gb, a.note or " ".join(cmd)[:80], a.max_wait_min, a.smoke))


if __name__ == "__main__":
    sys.exit(main())
