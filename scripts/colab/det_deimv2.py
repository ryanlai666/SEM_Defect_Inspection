"""Detector benchmark (Colab L4): DEIMv2-L (DINOv3-S STAs backbone), official repo, fine-tuned from COCO.

Code:    https://github.com/Intellindust-AI-Lab/DEIMv2 at a pinned commit, DEIMv2 License (NON-COMMERCIAL, not
         Apache-2.0). Local patches: RandomVerticalFlip registered as a transform; torchvision >= 0.21
         transform-API shim.
Weights: the official COCO checkpoint on Hugging Face (Intellindust/DEIMv2_DINOv3_L_COCO, linked from the README,
         not gated), converted from safetensors to the {'model': state_dict} file that `train.py -t` expects.
         The separate DINOv3 backbone file (gated Meta download) is NOT used: the COCO checkpoint holds all weights.
Recipe:  official training loop and its checkpoint logic (best-on-val stage-1 reload at the no-aug switch, best
         stage-2 kept) - val-based selection is built into this recipe, unlike the other four detectors.
Config:  scripts/colab/deimv2/deimv2_dinov3_l_sem.yml (35 epochs, 480 px, batch 16).

    python scripts/colab/det_deimv2.py --stage all      # setup + train + eval
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from det_common import EPOCHS, IMGSZ, coco_metrics, env, gflops, n_params, save, timed

ROOT = Path(__file__).resolve().parents[2]
REPO = Path("/content/DEIMv2")
COMMIT = "f005f50ce565e63f6c298feccb51901a98c3d035"
HF_REPO = "Intellindust/DEIMv2_DINOv3_L_COCO"
CFG = "configs/deimv2/deimv2_dinov3_l_sem.yml"
RUN = ROOT / "outputs" / "det_deimv2_l_colab"
INIT = Path("/content/deimv2_l_coco_init.pth")
PATCH = """
# [SEM benchmark patch 1] vertical flip, for parity with the other detectors
RandomVerticalFlip = register()(T.RandomVerticalFlip)

# [SEM benchmark patch 2] torchvision >= 0.21 renamed Transform._transform / _get_params to transform /
# make_params; route the new hooks to this module's old-style overrides (no behaviour change)
for _c in list(globals().values()):
    if isinstance(_c, type) and issubclass(_c, T.Transform) and _c.__module__ == __name__:
        if "_transform" in _c.__dict__ and "transform" not in _c.__dict__:
            _c.transform = _c._transform
        if "_get_params" in _c.__dict__ and "make_params" not in _c.__dict__:
            _c.make_params = _c._get_params
"""


def sh(*cmd: str, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=cwd)


def setup() -> dict:
    if not (REPO / "train.py").exists():
        sh("git", "clone", "https://github.com/Intellindust-AI-Lab/DEIMv2.git", str(REPO))
    sh("git", "checkout", "-q", COMMIT, cwd=REPO)
    tf = REPO / "engine/data/transforms/_transforms.py"
    if "[SEM benchmark patch" not in tf.read_text():
        with open(tf, "a") as f:
            f.write(PATCH)
    here = Path(__file__).resolve().parent / "deimv2"
    shutil.copy(here / "sem_detection.yml", REPO / "configs/dataset/sem_detection.yml")
    shutil.copy(here / "deimv2_dinov3_l_sem.yml", REPO / CFG)
    if not INIT.exists():
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        sd = load_file(hf_hub_download(HF_REPO, "model.safetensors"))
        sd = {k: v for k, v in sd.items() if not k.startswith("postprocessor.")}
        torch.save({"model": sd}, INIT)
        print(f"converted {len(sd)} tensors from {HF_REPO} -> {INIT}")
    return {"repo_commit": COMMIT, "init": f"hf://{HF_REPO}/model.safetensors"}


def train() -> None:
    out = RUN / "train"
    last = out / "last.pth"
    log = out / "log.txt"
    n_done = len(log.read_text().strip().splitlines()) if log.exists() else 0
    if n_done >= EPOCHS:
        return
    args = ["python", "train.py", "-c", CFG, "--use-amp", "--seed=0"]
    args += ["-r", str(last)] if last.exists() else ["-t", str(INIT)]
    sh(*args, cwd=REPO)


def final_checkpoint() -> Path:
    out = RUN / "train"
    for name in ("best_stg2.pth", "best_stg1.pth", "last.pth"):
        if (out / name).exists():
            return out / name
    raise FileNotFoundError(out)


@torch.no_grad()
def evaluate(skip_latency: bool) -> None:
    sys.path.insert(0, str(REPO))
    import json

    import torchvision.transforms as T
    from engine.core import YAMLConfig
    from PIL import Image

    ck = final_checkpoint()
    cfg = YAMLConfig(str(REPO / CFG), resume=str(ck))
    state = torch.load(ck, map_location="cpu", weights_only=False)
    cfg.model.load_state_dict(state["ema"]["module"] if "ema" in state else state["model"])
    dev = torch.device("cuda")
    model = cfg.model.deploy().to(dev).eval()
    post = cfg.postprocessor.deploy().to(dev).eval()
    tfm = T.Compose(
        [T.Resize((IMGSZ, IMGSZ)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    gt = json.loads(Path("/content/det_data/annotations/instances_test.json").read_text())
    results = []
    for im in gt["images"]:
        pil = Image.open(Path("/content/det_data/images/test") / im["file_name"]).convert("RGB")
        x = tfm(pil)[None].to(dev)
        size = torch.tensor([[im["width"], im["height"]]], device=dev)
        labels, boxes, scores = post(model(x), size)
        top = scores[0].argsort(descending=True)[:100]  # 100 detections per image, as for the other detectors
        for k in top.tolist():
            x0, y0, x1, y1 = boxes[0, k].tolist()
            results.append(
                {
                    "image_id": im["id"],
                    "category_id": int(labels[0, k]) + 1,  # 0-based training labels -> COCO ids 1..5
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "score": float(scores[0, k]),
                }
            )
    res = {
        "model": "deimv2_l_dinov3",
        "weights": f"{HF_REPO} (COCO), fine-tuned",
        "epochs": EPOCHS,
        "imgsz": IMGSZ,
        "checkpoint": f"{ck.name} (official recipe: val-based stage checkpoints, EMA)",
        "license": "DEIMv2 License (non-commercial); DINOv3-derived weights under the DINOv3 license",
        "test": coco_metrics("/content/det_data/annotations/instances_test.json", results),
        "env": env(),
    }

    class Full(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m, self.p = model, post

        def forward(self, x):
            return self.p(self.m(x), torch.tensor([[477, 477]], device=x.device))

    full = Full().eval()
    res["params"] = n_params(model)
    x480 = torch.rand(1, 3, IMGSZ, IMGSZ, device=dev)
    res["gflops_480"] = gflops(full, x480)
    if not skip_latency:

        def fwd():
            with torch.autocast("cuda", dtype=torch.float16):
                return full(x480)

        res["latency_model_fp16_bs1"] = timed(fwd)
        res["latency_note"] = "deploy-mode model + post-processor (top-k, box rescale; NMS-free), AMP FP16"
    save(RUN, res)
    np.save(RUN / "n_results.npy", np.asarray([len(results)]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["setup", "train", "eval", "all"], default="all")
    ap.add_argument("--skip-latency", action="store_true")
    a = ap.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    if a.stage in ("setup", "all"):
        info = setup()
        (RUN / "setup.json").write_text(str(info))
    if a.stage in ("train", "all"):
        train()
    if a.stage in ("eval", "all"):
        evaluate(a.skip_latency)


if __name__ == "__main__":
    main()
