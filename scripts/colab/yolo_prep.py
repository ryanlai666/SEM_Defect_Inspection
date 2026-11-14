"""Convert Carinthia-S + the project's split (seed 0) into an Ultralytics YOLO-seg dataset.

* images: same 3-px border crop as everywhere else (rows 1-477, cols 0-476 -> 477x477), grayscale PNG
* labels: one polygon per external contour of the binary mask (>127), class = defect class 1..5 -> 0..4;
  class-6 images ("no visible defect", empty masks) are background images with an empty label file
* splits: train / val / test exactly as data/splits/carinthia_s_split_seed0.json

    python scripts/colab/yolo_prep.py --out /content/yolo_data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from semdefect.data import load_index, load_or_make_split, read_mask  # noqa: E402
from semdefect.data.carinthia import crop_border, read_gray  # noqa: E402

NAMES = ["class1", "class2", "class3", "class4", "class5"]  # Carinthia-S numeric classes 1..5


def polygons(mask: np.ndarray) -> list[np.ndarray]:
    """Binary mask (H,W) -> list of (K,2) float polygons in normalised xy (K >= 3)."""
    h, w = mask.shape
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cs:
        c = c.reshape(-1, 2).astype(np.float64)
        if len(c) < 3:  # 1-2 px blobs: use their bounding square so the instance is not lost
            x0, y0 = c.min(0)
            x1, y1 = c.max(0) + 1
            c = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
        c[:, 0] /= w
        c[:, 1] /= h
        out.append(np.clip(c, 0, 1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/content/yolo_data")
    ap.add_argument("--root", default=str(ROOT / "data/raw/carinthia_s/data"))
    ap.add_argument(
        "--limit", type=int, default=0, help="about N evenly spaced images per split (tests only)"
    )
    a = ap.parse_args()
    out = Path(a.out)
    df0 = load_index(a.root)
    split = load_or_make_split(df0, "carinthia_s", seed=0)
    df = df0.set_index("idx")
    stats = {}
    for part in ("train", "val", "test"):
        (out / "images" / part).mkdir(parents=True, exist_ok=True)
        (out / "labels" / part).mkdir(parents=True, exist_ok=True)
        n_inst = n_bg = 0
        ids = split[part][:: max(1, len(split[part]) // a.limit)] if a.limit else split[part]
        for i in ids:
            row = df.loc[i]
            img = read_gray(row["image_path"])
            img = crop_border(img)
            m = crop_border(read_mask(row["mask_path"])) > 0
            stem = f"{int(i):05d}"
            cv2.imwrite(str(out / "images" / part / f"{stem}.png"), img)
            lines = []
            if int(row["label"]) != 6:
                for p in polygons(m):
                    lines.append(f"{int(row['label']) - 1} " + " ".join(f"{v:.6f}" for v in p.reshape(-1)))
            n_inst += len(lines)
            n_bg += not lines
            (out / "labels" / part / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        stats[part] = {"images": len(ids), "instances": n_inst, "background_images": n_bg}
    (out / "data.yaml").write_text(
        f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names: {dict(enumerate(NAMES))}\n"
    )
    print(stats)


if __name__ == "__main__":
    main()
