"""Build the uint8 memmap caches (CPU only) for every (img_size, mask_size) used by the configs.

Run once before GPU jobs so no GPU memory is reserved while JPEGs are being decoded.

    python scripts/build_cache.py                 # all configs
    python scripts/build_cache.py configs/anomaly/patchcore.yaml
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semdefect.config import load_config  # noqa: E402
from semdefect.data import load_index  # noqa: E402
from semdefect.data.carinthia import DEFAULT_CROP, build_disk_cache  # noqa: E402


def combos(cfg: dict) -> set[tuple[int, int, str]]:
    d = cfg["data"]
    size, mode = int(d["img_size"]), d.get("resize_mode", "area")
    out = {(size, size, mode)}
    if cfg["task"] == "anomaly":  # test images carry masks at the common eval size
        out.add((size, int(cfg.get("eval", {}).get("eval_size", 224)), mode))
    return out


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]] or sorted(
        p for p in (ROOT / "configs").rglob("*.yaml") if not p.name.startswith("_") and p.name != "base.yaml"
    )
    todo: set[tuple[int, int, str]] = set()
    for p in paths:
        todo |= combos(load_config(p))
    df = load_index()
    for img, mask, mode in sorted(todo):
        t0 = time.time()
        ip, mp = build_disk_cache(df, img, mask, mode, DEFAULT_CROP)
        print(f"img {img} mask {mask} {mode}: {ip.name}, {mp.name} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
