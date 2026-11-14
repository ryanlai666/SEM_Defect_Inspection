"""Inspect Carinthia-S and write docs/dataset_analysis.md (+ figures in docs/figures/).

Checks: class counts, image sizes / modes / bit depth, mask value sets and binarisation, mask area per
class, empty masks, right-edge border, exact + near duplicates (perceptual dHash), intensity statistics
per class (shortcut check for the anomaly protocol), defect centroid positions, split summary.

    python scripts/analyze_dataset.py
"""

from __future__ import annotations

import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from semdefect.data import load_index, load_or_make_split  # noqa: E402
from semdefect.utils import save_json  # noqa: E402


def dhash(a: np.ndarray, size: int = 8) -> int:
    im = Image.fromarray(a).resize((size + 1, size), Image.BILINEAR)
    d = np.asarray(im, dtype=np.int16)
    bits = (d[:, 1:] > d[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def main() -> None:
    df = load_index()
    figs = ROOT / "docs" / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    n = len(df)
    counts = df["label"].value_counts().sort_index().to_dict()

    sizes, modes, msizes, mmodes = Counter(), Counter(), Counter(), Counter()
    nonbinary, empty_by_cls = [], Counter()
    area = defaultdict(list)
    n_comp = defaultdict(list)
    cent = defaultdict(list)
    mean_i = defaultdict(list)
    std_i = defaultdict(list)
    md5s: dict[str, list[int]] = defaultdict(list)
    dh: dict[int, list[int]] = defaultdict(list)
    border_cols = []
    for r in df.itertuples():
        raw = Path(r.image_path).read_bytes()
        im = Image.open(r.image_path)
        sizes[im.size] += 1
        modes[im.mode] += 1
        a = np.asarray(im.convert("L"))
        md5s[hashlib.md5(a.tobytes()).hexdigest()].append(r.idx)
        dh[dhash(a)].append(r.idx)
        border_cols.append(a[:, -5:].mean(0))
        core = a[1:-2, :-3].astype(np.float32)
        mean_i[r.label].append(core.mean())
        std_i[r.label].append(core.std())
        mi = Image.open(r.mask_path)
        msizes[mi.size] += 1
        mmodes[mi.mode] += 1
        m = np.asarray(mi.convert("L"))
        vals = np.unique(m)
        if not set(vals.tolist()) <= {0, 255}:
            nonbinary.append((r.idx, int(r.label), len(vals), float((m > 0).mean()), float((m > 127).mean())))
        mb = m > 127
        if not mb.any():
            empty_by_cls[r.label] += 1
        else:
            area[r.label].append(mb.mean())
            lab, k = ndimage.label(mb)
            n_comp[r.label].append(k)
            cy, cx = ndimage.center_of_mass(mb)
            cent[r.label].append((cy / m.shape[0], cx / m.shape[1]))
        del raw

    exact_dups = [v for v in md5s.values() if len(v) > 1]
    near_dups = [v for v in dh.values() if len(v) > 1]
    border = np.mean(border_cols, axis=0)

    # --------------------------------------------------------------- figures
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh([str(c) for c in counts], list(counts.values()), color="#4C78A8")
    ax.set_xscale("log")
    ax.set_xlabel("# images (log)")
    ax.set_ylabel("class")
    for i, v in enumerate(counts.values()):
        ax.text(v, i, f" {v}", va="center", fontsize=8)
    ax.set_title("Carinthia-S class distribution")
    fig.tight_layout()
    fig.savefig(figs / "class_distribution.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(2, 6, figsize=(13, 4.6))
    rng = np.random.default_rng(0)
    for j, c in enumerate(sorted(counts)):
        sub = df[df["label"] == c]
        r = sub.iloc[int(rng.integers(len(sub)))]
        a = np.asarray(Image.open(r["image_path"]).convert("L"))
        m = np.asarray(Image.open(r["mask_path"]).convert("L")) > 127
        axes[0, j].imshow(a, cmap="gray")
        axes[0, j].set_title(f"class {c} (n={counts[c]})", fontsize=9)
        axes[1, j].imshow(a, cmap="gray")
        if m.any():
            axes[1, j].contour(m, levels=[0.5], colors="r", linewidths=0.8)
        axes[1, j].set_title(f"mask area {m.mean() * 100:.2f}%", fontsize=8)
        for ax in axes[:, j]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(figs / "examples.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3))
    for c in sorted(area):
        ax.hist(np.log10(np.array(area[c]) * 100), bins=30, alpha=0.6, label=f"class {c}", density=True)
    ax.set_xlabel("log10(mask area %)")
    ax.legend(fontsize=7)
    ax.set_title("Defect area per class")
    fig.tight_layout()
    fig.savefig(figs / "mask_area.png", dpi=130)
    plt.close(fig)

    split = load_or_make_split(df, seed=0)
    split_counts = {
        k: df[df["idx"].isin(v)]["label"].value_counts().sort_index().to_dict() for k, v in split.items()
    }

    stats = {
        "n_images": n,
        "class_counts": {int(k): int(v) for k, v in counts.items()},
        "image_sizes": {str(k): v for k, v in sizes.items()},
        "image_modes": dict(modes),
        "mask_sizes": {str(k): v for k, v in msizes.items()},
        "mask_modes": dict(mmodes),
        "non_binary_masks": nonbinary,
        "empty_masks_by_class": {int(k): int(v) for k, v in empty_by_cls.items()},
        "mask_area_pct": {
            int(c): {
                "median": float(np.median(v) * 100),
                "p5": float(np.percentile(v, 5) * 100),
                "p95": float(np.percentile(v, 95) * 100),
            }
            for c, v in area.items()
        },
        "components_per_mask": {
            int(c): {
                "mean": float(np.mean(v)),
                "max": int(np.max(v)),
                "frac_multi": float(np.mean(np.array(v) > 1)),
            }
            for c, v in n_comp.items()
        },
        "centroid_mean_yx": {int(c): np.mean(v, axis=0).round(3).tolist() for c, v in cent.items()},
        "centroid_std_yx": {int(c): np.std(v, axis=0).round(3).tolist() for c, v in cent.items()},
        "intensity_mean_by_class": {int(c): [float(np.mean(v)), float(np.std(v))] for c, v in mean_i.items()},
        "intensity_std_by_class": {int(c): [float(np.mean(v)), float(np.std(v))] for c, v in std_i.items()},
        "right_border_last5_cols_mean": border.round(1).tolist(),
        "exact_duplicate_groups": exact_dups,
        "near_duplicate_groups_dhash64": near_dups,
        "split_seed0_counts": {k: {int(a): int(b) for a, b in v.items()} for k, v in split_counts.items()},
    }
    save_json(stats, ROOT / "docs" / "dataset_stats.json")
    print({k: v for k, v in stats.items() if k not in ("non_binary_masks", "near_duplicate_groups_dhash64")})
    print("non-binary masks:", len(nonbinary), nonbinary[:10])
    print("near-dup groups:", len(near_dups), [len(g) for g in near_dups][:20])


if __name__ == "__main__":
    main()
