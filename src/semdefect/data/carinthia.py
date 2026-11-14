"""Carinthia-S (Zenodo 10.5281/zenodo.16895427, CC BY 4.0) index + torch datasets.

Facts verified by `scripts/analyze_dataset.py` (see docs/dataset_analysis.md):
4,591 grayscale 8-bit JPEGs of 480x480 px, one binary mask PNG per image, 6 numeric classes
(1:55, 2:8, 3:4008, 4:289, 5:4, 6:227).  Class 6 = "no visible defect" (SEM mis-alignment),
its masks are empty.  A ~3 px dark border runs along the right edge, so images are cropped
to a 477x477 square before resizing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from ..utils import resolve_path

DEFAULT_ROOT = "data/raw/carinthia_s/data"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# (top, bottom, left, right) pixels removed -> 477 x 477 square without the dark right border.
DEFAULT_CROP = (1, 2, 0, 3)
NORMAL_LABEL = 6
# Label/mask inconsistencies found by scripts/analyze_dataset.py (keyed by filename, not row order).
# Excluded from the anomaly-detection protocol only; kept for classification / segmentation.
AD_EXCLUDE = {
    "8b782473c99742e394df74b430412232": "class 6 ('no visible defect') but its mask marks a 50-px region",
}


def load_index(root: str | Path = DEFAULT_ROOT) -> pd.DataFrame:
    root = resolve_path(root)
    csv = root / "carinthia-s.csv"
    if not csv.exists():
        raise FileNotFoundError(f"{csv} not found - run `python scripts/download_data.py` first")
    df = pd.read_csv(csv, sep=";")
    df["label"] = df["label"].astype(int)
    df["image_path"] = [str(root / p) for p in df["image_path"]]
    df["mask_path"] = [str(root / p) for p in df["mask_path"]]
    df = df.reset_index(drop=True)
    df["idx"] = np.arange(len(df))
    return df


def read_gray(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)


def read_mask(path: str | Path, threshold: int = 127) -> np.ndarray:
    """Masks are nominally {0,255}; a handful contain JPEG-like grey levels, so threshold."""
    with Image.open(path) as im:
        return (np.asarray(im.convert("L")) > threshold).astype(np.uint8)


def crop_border(a: np.ndarray, crop: Sequence[int] = DEFAULT_CROP) -> np.ndarray:
    t, b, left, r = crop
    h, w = a.shape[:2]
    return a[t : h - b, left : w - r]


def to_model_input(
    x01: torch.Tensor, mean: Sequence[float] = IMAGENET_MEAN, std: Sequence[float] = IMAGENET_STD
) -> torch.Tensor:
    """(…,1,H,W) grey in [0,1] -> (…,3,H,W) ImageNet-normalised (grey replicated)."""
    m = torch.tensor(mean, dtype=x01.dtype, device=x01.device).view(3, 1, 1)
    s = torch.tensor(std, dtype=x01.dtype, device=x01.device).view(3, 1, 1)
    rep = [1] * (x01.dim() - 3) + [3, 1, 1]
    return (x01.repeat(*rep) - m) / s


def resize01(x: torch.Tensor, size: int, mode: str = "area") -> torch.Tensor:
    """Resize (C,H,W) or (B,C,H,W) float tensor.  `area` == adaptive average pooling, which is
    anti-aliased for downscaling and is mirrored exactly by the CUDA preprocessing kernel."""
    squeeze = x.dim() == 3
    if squeeze:
        x = x.unsqueeze(0)
    if mode == "bilinear":
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=False)
    elif mode == "area":
        x = F.adaptive_avg_pool2d(x, (size, size))
    elif mode == "nearest":
        x = F.interpolate(x, size=(size, size), mode="nearest")
    else:
        raise ValueError(mode)
    return x.squeeze(0) if squeeze else x


class CarinthiaS(Dataset):
    """Returns dict(image=(3,S,S) normalised, image01=(1,S,S), mask=(1,S,S) {0,1}, label, label_idx,
    is_anomalous, idx).

    Args:
        df: index from `load_index`.
        indices: row indices (dataset-level `idx`) forming this subset.
        img_size: square output size (multiple of 14 for DINOv2, e.g. 224/448).
        classes: label ids in the order defining `label_idx`.
        augment: callable(img01, mask) -> (img01, mask) applied on tensors (train only).
        perturb: callable(img01) -> img01 (robustness evaluation, applied after augment).
        cache: False | "ram" | "disk".  "ram": per-process dict of resized uint8 arrays;
            "disk": memory-mapped uint8 .npy arrays built once by `build_disk_cache` (shared by all
            DataLoader workers through the OS page cache - removes the JPEG-decode bottleneck).
            Cached images are quantised to uint8 after resizing (<= 0.5/255 error).
        mask_size: mask resolution if different from img_size (anomaly maps are scored at a common size).
        Note: geometric augmentation requires mask_size == img_size.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        indices: Sequence[int],
        img_size: int = 224,
        classes: Sequence[int] = (1, 2, 3, 4, 5, 6),
        augment: Callable | None = None,
        perturb: Callable | None = None,
        resize_mode: str = "area",
        crop: Sequence[int] = DEFAULT_CROP,
        cache: bool | str = False,
        load_masks: bool = True,
        mask_size: int | None = None,
        cache_dir: str | Path = "data/cache/arrays",
    ):
        self.df = df
        self.indices = list(map(int, indices))
        self.img_size = img_size
        self.classes = list(classes)
        self.cls_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.augment = augment
        self.perturb = perturb
        self.resize_mode = resize_mode
        self.crop = crop
        self.load_masks = load_masks
        self.mask_size = mask_size or img_size
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = {} if cache in (True, "ram") else None
        self._disk: tuple[Path, Path] | None = None
        self._mm: tuple[np.ndarray, np.ndarray] | None = None  # opened lazily (per worker process)
        if cache == "disk":
            self._disk = build_disk_cache(df, img_size, self.mask_size, resize_mode, crop, cache_dir)

    def __len__(self) -> int:
        return len(self.indices)

    def labels(self) -> np.ndarray:
        return self.df.loc[self.indices, "label"].to_numpy()

    def _load(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._disk is not None:
            if self._mm is None:
                self._mm = (np.load(self._disk[0], mmap_mode="r"), np.load(self._disk[1], mmap_mode="r"))
            img = torch.from_numpy(np.array(self._mm[0][i], dtype=np.float32) / 255.0)[None]
            m = torch.from_numpy(np.array(self._mm[1][i], dtype=np.float32))[None]
            return img, m
        if self._cache is not None and i in self._cache:
            img_u8, mask_u8 = self._cache[i]
        else:
            row = self.df.loc[i]
            img = crop_border(read_gray(row["image_path"]), self.crop)
            x = torch.from_numpy(img.astype(np.float32) / 255.0)[None]
            x = resize01(x, self.img_size, self.resize_mode)
            if self.load_masks:
                m = crop_border(read_mask(row["mask_path"]), self.crop)
                mt = torch.from_numpy(m.astype(np.float32))[None]
                # area-resize then threshold at 0.5 keeps thin defects roughly area-preserving
                mt = (resize01(mt, self.mask_size, "area") >= 0.5).float()
            else:
                mt = torch.zeros(1, self.mask_size, self.mask_size)
            if self._cache is not None:  # RAM cache stores uint8 (<=0.5/255 rounding error)
                self._cache[i] = ((x * 255.0).round().clamp(0, 255).to(torch.uint8), mt.to(torch.uint8))
                img_u8, mask_u8 = self._cache[i]
            else:  # exact float path: bit-compatible with the CUDA preprocess kernel
                return x, mt
        return img_u8.float() / 255.0, mask_u8.float()

    def __getstate__(self):  # never pickle an open memmap into DataLoader workers
        st = self.__dict__.copy()
        st["_mm"] = None
        return st

    def __getitem__(self, j: int) -> dict:
        i = self.indices[j]
        x01, mask = self._load(i)
        if self.augment is not None:
            x01, mask = self.augment(x01, mask)
        if self.perturb is not None:
            x01 = self.perturb(x01).clamp(0.0, 1.0)
        label = int(self.df.at[i, "label"])
        return {
            "image": to_model_input(x01),
            "image01": x01,
            "mask": mask,
            "label": label,
            "label_idx": self.cls_to_idx.get(label, -1),
            "is_anomalous": int(label != NORMAL_LABEL),
            "idx": i,
        }


def build_disk_cache(
    df: pd.DataFrame,
    img_size: int,
    mask_size: int,
    resize_mode: str = "area",
    crop: Sequence[int] = DEFAULT_CROP,
    cache_dir: str | Path = "data/cache/arrays",
) -> tuple[Path, Path]:
    """Decode + crop + resize every image once into uint8 (N,S,S) .npy files indexed by `idx`.

    Disk: N x S^2 bytes, e.g. 4591 images -> 230 MB @224, 301 MB @256, 922 MB @448."""
    cdir = resolve_path(cache_dir)
    tag = f"n{len(df)}_c{'-'.join(map(str, crop))}"
    ipath = cdir / f"img_{img_size}_{resize_mode}_{tag}.npy"
    mpath = cdir / f"mask_{mask_size}_{tag}.npy"
    need_img, need_mask = not ipath.exists(), not mpath.exists()
    if not (need_img or need_mask):
        return ipath, mpath
    cdir.mkdir(parents=True, exist_ok=True)
    n = int(df["idx"].max()) + 1
    imgs = np.zeros((n, img_size, img_size), np.uint8) if need_img else None
    masks = np.zeros((n, mask_size, mask_size), np.uint8) if need_mask else None
    for row in df.itertuples():
        if need_img:
            x = torch.from_numpy(crop_border(read_gray(row.image_path), crop).astype(np.float32) / 255.0)[None]
            imgs[row.idx] = (resize01(x, img_size, resize_mode)[0] * 255.0).round().clamp(0, 255).byte().numpy()
        if need_mask:
            mt = torch.from_numpy(crop_border(read_mask(row.mask_path), crop).astype(np.float32))[None]
            masks[row.idx] = (resize01(mt, mask_size, "area")[0] >= 0.5).byte().numpy()
    # Only missing files are written: replacing a file that another dataset has memory-mapped fails on Windows.
    for path, arr in ((ipath, imgs), (mpath, masks)):
        if arr is not None:
            tmp = path.with_suffix(".tmp.npy")
            np.save(tmp, arr)
            tmp.replace(path)
    return ipath, mpath


class SyntheticSEM(Dataset):
    """Tiny procedurally generated SEM-like dataset for CPU unit tests / CI (no download).

    Background: smooth grain texture + Poisson-like noise; defects: bright/dark blobs or scratches
    whose class is encoded by shape, mask given exactly."""

    def __init__(self, n: int = 24, img_size: int = 64, n_classes: int = 3, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.items = []
        yy, xx = np.mgrid[0:img_size, 0:img_size]
        for i in range(n):
            label = i % n_classes
            bg = rng.normal(0.45, 0.05, (img_size, img_size))
            k = np.ones(5) / 5
            bg = np.apply_along_axis(lambda r, k=k: np.convolve(r, k, mode="same"), 1, bg)
            cy, cx = rng.integers(img_size // 4, 3 * img_size // 4, size=2)
            r = rng.integers(3, img_size // 6)
            if label == 0:  # normal (no defect)
                m = np.zeros_like(bg, dtype=bool)
            elif label == 1:  # particle
                m = (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2
            else:  # scratch
                m = (np.abs((yy - cy) - (xx - cx)) <= 1) & (np.abs(xx - cx) <= 2 * r)
            img = bg + 0.4 * m
            img = np.clip(img + rng.normal(0, 0.02, bg.shape), 0, 1).astype(np.float32)
            self.items.append((img, m.astype(np.float32), label))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        img, m, label = self.items[i]
        x01 = torch.from_numpy(img)[None]
        return {
            "image": to_model_input(x01),
            "image01": x01,
            "mask": torch.from_numpy(m)[None],
            "label": label,
            "label_idx": label,
            "is_anomalous": int(label != 0),
            "idx": i,
        }
