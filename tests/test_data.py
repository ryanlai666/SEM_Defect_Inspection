import numpy as np
import pytest
import torch

from semdefect.config import load_config
from semdefect.data import CarinthiaS, fewshot_supports, load_index, stratified_group_split
from semdefect.data.carinthia import crop_border, resize01
from semdefect.data.splits import anomaly_split


def test_load_index_and_dataset(fake_root):
    df = load_index(fake_root)
    assert len(df) == 43 and set(df.label) == {1, 2, 3, 4, 5, 6}
    ds = CarinthiaS(df, df.idx[:4], img_size=32)
    item = ds[0]
    assert item["image"].shape == (3, 32, 32)
    assert item["image01"].min() >= 0 and item["image01"].max() <= 1
    assert set(torch.unique(item["mask"]).tolist()) <= {0.0, 1.0}


def test_border_crop_removes_dark_columns(fake_root):
    df = load_index(fake_root)
    from semdefect.data import read_gray

    a = crop_border(read_gray(df.image_path[0]))
    assert a.shape == (93, 93)
    assert a[:, -1].mean() > 50  # the 3 dark columns are gone


def test_area_resize_equals_adaptive_pool():
    x = torch.rand(1, 1, 477, 477)
    y = resize01(x, 224, "area")
    ref = torch.nn.functional.adaptive_avg_pool2d(x, 224)
    assert torch.allclose(y, ref)


def test_split_is_deterministic_disjoint_and_stratified(fake_root):
    df = load_index(fake_root)
    a = stratified_group_split(df, seed=3)
    b = stratified_group_split(df, seed=3)
    assert a == b
    all_ids = set(a["train"]) | set(a["val"]) | set(a["test"])
    assert all_ids == set(df.idx) and sum(len(v) for v in a.values()) == len(df)
    lab = df.set_index("idx")["label"]
    for c in df.label.unique():  # every class (>=3 samples) appears in every subset
        for part in a.values():
            assert any(lab[i] == c for i in part)


def test_fewshot_supports_nested_and_capped(fake_root):
    df = load_index(fake_root)
    split = stratified_group_split(df, seed=0)
    sup = fewshot_supports(df, split["train"], ks=[1, 2, 20], seeds=[0, 1])
    s0 = sup["supports"]["0"]
    for c in sup["classes"]:
        assert set(s0["1"][str(c)]) <= set(s0["2"][str(c)]) <= set(s0["20"][str(c)])
        assert set(s0["20"][str(c)]) <= set(split["train"])
    # class 5 has 3 images -> 1 train image -> k_eff capped at 1
    assert sup["k_eff"]["20"]["5"] == 1


def test_anomaly_split_uses_only_train_normals(fake_root):
    df = load_index(fake_root)
    split = stratified_group_split(df, seed=0)
    a = anomaly_split(df, split)
    lab = df.set_index("idx")["label"]
    assert a["train_normal"] and all(lab[i] == 6 for i in a["train_normal"])
    assert not set(a["train_normal"]) & set(a["test"])


def test_config_inheritance_and_overrides(tmp_path):
    (tmp_path / "base.yaml").write_text("a: 1\nnested: {x: 1, y: 2}\n")
    (tmp_path / "child.yaml").write_text("_base_: base.yaml\nnested: {y: 3}\n")
    cfg = load_config(tmp_path / "child.yaml", ["nested.z=[1,2]", "a=0.5"])
    assert cfg["nested"] == {"x": 1, "y": 3, "z": [1, 2]} and cfg["a"] == 0.5


def test_all_repo_configs_parse():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "configs"
    files = [p for p in root.rglob("*.yaml") if not p.name.startswith("_") and p.name != "base.yaml"]
    assert len(files) >= 10
    for p in files:
        cfg = load_config(p)
        assert cfg["task"] in {"fewshot", "classify", "segment", "anomaly"}, p
        assert "name" in cfg


@pytest.mark.parametrize("n", [0, 5])
def test_subsample_keeps_rare_classes(n):
    from semdefect.engine.common import subsample

    idx = list(range(100))
    labels = [0] * 90 + [1] * 8 + [2] * 2
    out = subsample(idx, n or None, labels=labels)
    if n:
        assert len(out) == 5 and {labels[i] for i in out} == {0, 1, 2}
    else:
        assert out == idx
    assert np.all(np.diff(out) > 0)
