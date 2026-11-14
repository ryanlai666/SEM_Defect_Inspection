"""The run-phase plan must be well-formed: unique names, existing configs/scripts, VRAM reservations within budget."""

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _plan():
    spec = importlib.util.spec_from_file_location("run_campaign", REPO / "scripts" / "run_campaign.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_campaign"] = mod  # dataclasses resolve the defining module via sys.modules
    spec.loader.exec_module(mod)
    return mod.plan()


def test_plan_is_well_formed():
    runs = _plan()
    names = [r.name for r in runs]
    assert len(names) == len(set(names)), "duplicate run names"
    for r in runs:
        assert r.tier in {"cpu", "t1", "t2", "t3"}
        assert r.gb is None or 0 < r.gb <= 5.5, r.name
        assert r.done_files, r.name
        for tok in r.cmd:
            if tok.endswith((".yaml", ".py")) and ("configs/" in tok or "scripts/" in tok):
                assert (REPO / tok).exists(), f"{r.name}: missing {tok}"
    # robustness sweeps must come after the training run they evaluate
    for i, r in enumerate(runs):
        if r.name.endswith("_robustness"):
            base = r.name.removesuffix("_robustness")
            assert base in names[:i], r.name
