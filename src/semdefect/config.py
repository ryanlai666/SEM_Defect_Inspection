"""Minimal YAML config system: `_base_` inheritance + dotted CLI overrides.

Kept dependency-free on purpose (no Hydra) so configs work identically on Windows,
Linux CI and Colab.  Example::

    cfg = load_config("configs/classify/fewshot_dinov2.yaml", ["model.backbone=dinov2_vitb14", "seeds=[0,1]"])
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    bases = data.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]
    merged: dict = {}
    for b in bases:
        merged = _deep_merge(merged, _load_yaml((path.parent / b).resolve()))
    return _deep_merge(merged, data)


def set_by_dotted(cfg: dict, key: str, value: Any) -> None:
    node = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def get_by_dotted(cfg: dict, key: str, default: Any = None) -> Any:
    node: Any = cfg
    for p in key.split("."):
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node


def parse_overrides(overrides: list[str] | None) -> list[tuple[str, Any]]:
    out = []
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key=value, got {item!r}")
        k, v = item.split("=", 1)
        out.append((k.strip(), yaml.safe_load(v)))
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    cfg = _load_yaml(Path(path).resolve())
    for k, v in parse_overrides(overrides):
        set_by_dotted(cfg, k, v)
    cfg.setdefault("_meta", {})["config_path"] = str(path)
    return cfg


def flatten(cfg: dict, prefix: str = "") -> dict[str, Any]:
    """Flatten nested config for MLflow params."""
    flat: dict[str, Any] = {}
    for k, v in cfg.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(flatten(v, key + "."))
        else:
            flat[key] = v
    return flat
