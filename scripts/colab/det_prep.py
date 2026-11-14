"""Detection data for the detector benchmark (Colab L4).

Boxes = bounding boxes of the 8-connected components of each binary defect mask (after the 3-px border crop,
477x477), class = image defect class 1..5; class-6 images are background images.  Same split (seed 0) and the
same PNGs as the YOLO-seg dataset (hard-linked from --src).  Writes:
  <out>/images/{train,val,test}/*.png, <out>/labels/{...}/*.txt  (YOLO boxes, class 0..4)
  <out>/data.yaml
  <out>/annotations/instances_{train,val,test}.json  (COCO; category_id 1..5, area = component pixel count)

    python scripts/colab/det_prep.py --src /content/yolo_data --out /content/det_data
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from semdefect.data import load_index, load_or_make_split, read_mask  # noqa: E402
from semdefect.data.carinthia import crop_border  # noqa: E402

NAMES = ["class1", "class2", "class3", "class4", "class5"]
SMALL = 32**2  # COCO "small": area < 32^2 px (original 477x477 resolution)


def components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """-> [(x, y, w, h, area_px)] of the 8-connected components."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [tuple(int(v) for v in stats[k, :5]) for k in range(1, n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/content/yolo_data", help="YOLO-seg dataset (images are reused)")
    ap.add_argument("--out", default="/content/det_data")
    ap.add_argument("--root", default=str(ROOT / "data/raw/carinthia_s/data"))
    ap.add_argument(
        "--limit", type=int, default=0, help="about N evenly spaced images per split (tests only)"
    )
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    df0 = load_index(a.root)
    split = load_or_make_split(df0, "carinthia_s", seed=0)
    df = df0.set_index("idx")
    (out / "annotations").mkdir(parents=True, exist_ok=True)
    stats = {}
    for part in ("train", "val", "test"):
        (out / "images" / part).mkdir(parents=True, exist_ok=True)
        (out / "labels" / part).mkdir(parents=True, exist_ok=True)
        ids = split[part][:: max(1, len(split[part]) // a.limit)] if a.limit else split[part]
        coco = {
            "images": [],
            "annotations": [],
            "categories": [{"id": c + 1, "name": n} for c, n in enumerate(NAMES)],
        }
        n_small = 0
        for i in ids:
            stem = f"{int(i):05d}"
            s_img, d_img = src / "images" / part / f"{stem}.png", out / "images" / part / f"{stem}.png"
            if not d_img.exists():
                try:
                    os.link(s_img, d_img)
                except OSError:
                    shutil.copy2(s_img, d_img)
            m = crop_border(read_mask(df.loc[i, "mask_path"])) > 0
            h, w = m.shape
            coco["images"].append({"id": int(i), "file_name": f"{stem}.png", "width": w, "height": h})
            lines = []
            if int(df.loc[i, "label"]) != 6:
                c = int(df.loc[i, "label"])
                for x, y, bw, bh, area in components(m):
                    lines.append(
                        f"{c - 1} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}"
                    )
                    coco["annotations"].append(
                        {
                            "id": len(coco["annotations"]) + 1,
                            "image_id": int(i),
                            "category_id": c,
                            "bbox": [x, y, bw, bh],
                            "area": area,
                            "iscrowd": 0,
                        }
                    )
                    n_small += area < SMALL
            (out / "labels" / part / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        (out / "annotations" / f"instances_{part}.json").write_text(json.dumps(coco))
        zero = {  # DEIMv2 with remap_mscoco_category=False uses category_id as the label index
            **coco,
            "categories": [{**c, "id": c["id"] - 1} for c in coco["categories"]],
            "annotations": [{**a, "category_id": a["category_id"] - 1} for a in coco["annotations"]],
        }
        (out / "annotations" / f"instances_{part}_0based.json").write_text(json.dumps(zero))
        stats[part] = {
            "images": len(ids),
            "boxes": len(coco["annotations"]),
            "small_boxes": n_small,
            "background_images": len(ids) - len({a["image_id"] for a in coco["annotations"]}),
        }
    (out / "data.yaml").write_text(
        f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names: {dict(enumerate(NAMES))}\n"
    )
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(stats)


if __name__ == "__main__":
    main()
