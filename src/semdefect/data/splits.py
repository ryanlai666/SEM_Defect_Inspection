"""Deterministic, group-aware stratified splits and nested few-shot support sets.

Design choices (documented in docs/dataset_analysis.md):
* Stratified per class, fractions train/val/test = 0.6/0.2/0.2 by default.
* Any class with >= 3 samples gets at least one val and one test sample (class 5 has only 4).
* Exact-duplicate images (same pixel hash) share a `group` and are never split across subsets.
* Few-shot supports are drawn ONLY from the train split and are *nested*
  (support_k=1 ⊂ support_k=5 ⊂ ...) per seed, which lowers variance of k-curves.
  If a class has fewer train images than k, all of them are used (k_eff is recorded).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import load_json, resolve_path, save_json


def _alloc(n: int, frac_val: float, frac_test: float) -> tuple[int, int, int]:
    if n < 3:
        return n, 0, 0
    n_test = max(1, int(round(frac_test * n)))
    n_val = max(1, int(round(frac_val * n)))
    n_train = n - n_val - n_test
    if n_train < 1:  # tiny classes: keep at least one training example
        n_train, n_val, n_test = 1, max(1, n - 2), 1
        n_val = n - n_train - n_test
    return n_train, n_val, n_test


def stratified_group_split(
    df: pd.DataFrame,
    seed: int = 0,
    frac_val: float = 0.2,
    frac_test: float = 0.2,
    label_col: str = "label",
    group_col: str | None = "group",
) -> dict[str, list[int]]:
    """Returns {'train': [...idx], 'val': [...], 'test': [...]} with dataset-level `idx` values."""
    rng = np.random.default_rng(seed)
    out: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    if group_col is None or group_col not in df.columns:
        df = df.assign(_g=df["idx"].to_numpy())
        group_col = "_g"
    for label in sorted(df[label_col].unique()):
        sub = df[df[label_col] == label]
        groups = sorted(sub[group_col].unique().tolist())
        rng.shuffle(groups)
        n_tr, n_va, _ = _alloc(len(groups), frac_val, frac_test)
        parts = {
            "train": groups[:n_tr],
            "val": groups[n_tr : n_tr + n_va],
            "test": groups[n_tr + n_va :],
        }
        for name, gs in parts.items():
            out[name].extend(sub[sub[group_col].isin(gs)]["idx"].tolist())
    for k in out:
        out[k] = sorted(int(i) for i in out[k])
    _check_disjoint(out)
    return out


def _check_disjoint(split: dict[str, list[int]]) -> None:
    names = list(split)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            inter = set(split[names[a]]) & set(split[names[b]])
            if inter:
                raise AssertionError(f"split overlap {names[a]}/{names[b]}: {sorted(inter)[:5]}")


def fewshot_supports(
    df: pd.DataFrame,
    train_idx: Sequence[int],
    ks: Sequence[int] = (1, 5, 10, 20),
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    classes: Sequence[int] | None = None,
    label_col: str = "label",
) -> dict:
    """Nested few-shot support sets.  Returns
    {'ks': [...], 'seeds': [...], 'supports': {seed: {k: {label: [idx...]}}}, 'k_eff': {...}}"""
    tr = df[df["idx"].isin(set(map(int, train_idx)))]
    classes = sorted(tr[label_col].unique().tolist()) if classes is None else list(classes)
    supports: dict = {}
    k_eff: dict = {}
    for seed in seeds:
        rng = np.random.default_rng(10_000 + seed)
        supports[str(seed)] = {}
        perms = {}
        for c in classes:
            pool = sorted(tr[tr[label_col] == c]["idx"].tolist())
            perms[c] = [int(x) for x in rng.permutation(pool)]
        for k in ks:
            supports[str(seed)][str(k)] = {str(c): perms[c][:k] for c in classes}
            k_eff[str(k)] = {str(c): min(k, len(perms[c])) for c in classes}
    return {"ks": list(ks), "seeds": list(seeds), "classes": classes, "supports": supports, "k_eff": k_eff}


def split_path(name: str, seed: int, split_dir: str | Path = "data/splits") -> Path:
    return resolve_path(Path(split_dir) / f"{name}_split_seed{seed}.json")


def load_or_make_split(
    df: pd.DataFrame, name: str = "carinthia_s", seed: int = 0, split_dir: str | Path = "data/splits", **kw
) -> dict:
    p = split_path(name, seed, split_dir)
    if p.exists():
        split = load_json(p)
        expected = set(df["idx"].tolist())
        got = set(split["train"]) | set(split["val"]) | set(split["test"])
        if got != expected:
            raise ValueError(f"{p} does not match the dataset index - delete it to regenerate")
        return split
    split = stratified_group_split(df, seed=seed, **kw)
    save_json(split, p)
    return split


def anomaly_split(df: pd.DataFrame, split: dict[str, list[int]], normal_label: int = 6) -> dict:
    """Anomaly-detection protocol on top of the classification split.

    train  = normal (class-6, 'no visible defect') images of the train split only;
    val    = val split (normal + defective) for threshold / hyper-parameter selection;
    test   = test split, normal vs defective (classes 1-5 have pixel masks for pixel AUROC / AUPRO).
    """
    from .carinthia import AD_EXCLUDE

    lab = df.set_index("idx")["label"]
    excl = set(df[df["filename"].isin(AD_EXCLUDE)]["idx"].tolist()) if "filename" in df else set()
    keep = lambda ids: [i for i in ids if i not in excl]  # noqa: E731
    return {
        "train_normal": [i for i in keep(split["train"]) if lab[i] == normal_label],
        "val": keep(split["val"]),
        "test": keep(split["test"]),
        "excluded": sorted(excl),
    }
