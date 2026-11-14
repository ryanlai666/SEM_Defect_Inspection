"""End-to-end smoke runs of every entry point on the fake on-disk dataset (CPU, random weights)."""

import json

import torch

from semdefect.cli import _run
from semdefect.engine.common import output_dir


def test_fewshot_engine(cfg_factory):
    cfg = cfg_factory(
        "configs/classify/fewshot_dinov2_vits14.yaml", "data.img_size=56", "fewshot.seeds=[0,1]"
    )
    rec = _run(cfg)
    assert rec["n_configs"] == 4 * 2 * 4 + 4  # 4 methods x 2 seeds x 4 ks + full (1 seed)
    assert (output_dir(cfg) / "fewshot_results.json").exists()


def test_cnn_engine_and_resume(cfg_factory):
    cfg = cfg_factory("configs/classify/resnet18_full.yaml", "data.img_size=64", "train.batch_size=4")
    rec = _run(cfg)
    assert rec["steps"] == 2 and rec["train_img_per_s"] > 0
    ck = torch.load(output_dir(cfg) / "checkpoints" / "last.ckpt", weights_only=False)
    assert ck["step"] == 2 and "rng" in ck


def test_segmentation_engine(cfg_factory):
    cfg = cfg_factory("configs/segment/unet.yaml", "data.img_size=64", "model.base=8", "train.batch_size=4")
    rec = _run(cfg)
    metrics = json.loads((output_dir(cfg) / "test_metrics.json").read_text())
    assert "dice_mean" in metrics and rec["params_trainable"] > 0


def test_anomaly_engine_patchcore(cfg_factory):
    cfg = cfg_factory(
        "configs/anomaly/patchcore.yaml",
        "data.img_size=64",
        "model.backbone=resnet18",
        "model.embed_dim=64",
        "eval.eval_size=32",
        "memory.offmask_images=4",
        "robustness.enabled=true",
        "robustness.levels=[1]",
    )
    _run(cfg)
    res = json.loads((output_dir(cfg) / "test_metrics.json").read_text())
    assert 0.0 <= res["test"]["image_auroc"] <= 1.0
    assert "pixel_auroc" in res["test"] and "aupro_30" in res["test"]
    assert set(res["robustness"]) == {"shot_noise_L1", "blur_L1", "charging_L1"}


def test_anomaly_engine_efficientad(cfg_factory):
    cfg = cfg_factory(
        "configs/anomaly/efficientad.yaml", "data.img_size=256", "eval.eval_size=32", "train.batch_size=2"
    )
    rec = _run(cfg)
    assert rec["train_it_per_s"] > 0


def test_training_resumes_from_last_checkpoint(cfg_factory, tmp_path, capsys):
    common = [
        "smoke.enabled=false",
        "data.img_size=64",
        "train.batch_size=8",
        "train.eval_every=2",
        "keep_last_checkpoint=true",
        "tracking.enabled=true",
        f"tracking.uri={(tmp_path / 'mlruns').as_uri()}",
    ]
    cfg = cfg_factory("configs/classify/resnet18_full.yaml", *common, "train.steps=2")
    _run(cfg)
    ck_path = output_dir(cfg) / "checkpoints" / "last.ckpt"
    first = torch.load(ck_path, weights_only=False)
    assert first["step"] == 2
    cfg2 = cfg_factory("configs/classify/resnet18_full.yaml", *common, "train.steps=4")
    _run(cfg2)
    assert "resumed from step 2" in capsys.readouterr().out
    second = torch.load(ck_path, weights_only=False)
    assert second["step"] == 4
    assert second["mlflow_run_id"] == first["mlflow_run_id"]  # same MLflow run continues


def test_last_checkpoint_removed_after_completion(cfg_factory):
    cfg = cfg_factory(
        "configs/classify/resnet18_full.yaml",
        "smoke.enabled=false",
        "data.img_size=64",
        "train.batch_size=8",
        "train.steps=2",
        "train.eval_every=2",
    )
    _run(cfg)
    ckdir = output_dir(cfg) / "checkpoints"
    assert not (ckdir / "last.ckpt").exists() and (ckdir / "best.ckpt").exists()


def test_robustness_sweep_after_training(cfg_factory):
    from semdefect.engine.robustness import run_robustness

    common = [
        "smoke.enabled=false",
        "data.img_size=64",
        "train.batch_size=8",
        "train.steps=2",
        "train.eval_every=2",
    ]
    cfg = cfg_factory("configs/classify/resnet18_full.yaml", *common, "robustness.levels=[1]")
    _run(cfg)
    res = run_robustness(cfg)
    assert set(res) == {"clean_L0", "shot_noise_L1", "blur_L1", "charging_L1"}
    assert all(0.0 <= r["macro_f1"] <= 1.0 for r in res.values())


def test_cli_writes_resources(cfg_factory, tmp_path, monkeypatch):
    import json as _json

    from semdefect import cli

    cfg = cfg_factory("configs/segment/unet.yaml", "data.img_size=64", "model.base=8", "train.batch_size=4")
    ovr = [
        f"{k}={v}"
        for k, v in (
            ("data.root", cfg["data"]["root"]),
            ("data.split_dir", cfg["data"]["split_dir"]),
            ("data.cache_dir", cfg["data"]["cache_dir"]),
            ("output_dir", cfg["output_dir"]),
        )
    ]
    cli.main(
        [
            "run",
            "configs/segment/unet.yaml",
            "--smoke",
            "tracking.enabled=false",
            "data.num_workers=0",
            "device=cpu",
            "amp=false",
            "smoke.n_images=24",
            "smoke.max_steps=2",
            "model.pretrained=false",
            "data.img_size=64",
            "model.base=8",
            "train.batch_size=4",
            *ovr,
        ]
    )
    rec = _json.loads((output_dir(cfg) / "resources_run.json").read_text())
    assert rec["cmd"] == "run" and rec["wall_s"] > 0
