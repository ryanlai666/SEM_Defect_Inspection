"""Batch-1 latency of YOLO26n-seg vs the local U-Net-R18 and PatchCore on the same Colab L4.

Two views per model and precision (FP32 / FP16):
  * model: GPU-resident input tensor -> raw model output (CUDA-synchronised)
  * e2e:   host uint8 477x477 image -> resize/normalise -> model -> mask / anomaly map back on the host
           (YOLO: Ultralytics `predict` incl. letterbox + mask post-processing)
U-Net-R18 (256 px) and PatchCore (WRN-50 at 256 px, kNN over a 38,740 x 1024 bank = the off-mask bank size,
PyTorch backends; the custom CUDA/TensorRT path is benchmarked locally) use random weights: latency does not
depend on the values.  p50 / p99 over 300 timed calls after 30 warm-up calls.

    python scripts/colab/latency_l4.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from _yolo_env import quiet_settings  # sets YOLO_CONFIG_DIR before ultralytics is imported

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from semdefect.data.carinthia import to_model_input  # noqa: E402
from semdefect.kernels.knn import knn_torch, knn_torch_fp16  # noqa: E402
from semdefect.models.anomaly import PatchCoreEmbedder, smooth_maps  # noqa: E402
from semdefect.models.segmentation import UNet  # noqa: E402

RUN = ROOT / "outputs" / "yolo26n_seg_colab"
N, WARM = 300, 30


def timed(fn) -> dict:
    for _ in range(WARM):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(N):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    a = np.asarray(ts)
    return {
        "p50_ms": float(np.percentile(a, 50)),
        "p99_ms": float(np.percentile(a, 99)),
        "mean_ms": float(a.mean()),
    }


def prep(img_u8: np.ndarray, size: int, dev) -> torch.Tensor:
    x = torch.from_numpy(img_u8).to(dev, non_blocking=True).float().div_(255.0)[None, None]
    return to_model_input(F.adaptive_avg_pool2d(x, (size, size)))


@torch.no_grad()
def main() -> None:
    from ultralytics import YOLO

    quiet_settings()

    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (477, 477), dtype=np.uint8)
    img3 = np.repeat(img[:, :, None], 3, axis=2)
    res: dict = {
        "env": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "protocol": __doc__.strip().splitlines()[0],
    }

    # ---------------------------------------------------------------- YOLO26n-seg
    import ultralytics

    best = str(RUN / "train" / "weights" / "best.pt")
    res["env"]["ultralytics"] = ultralytics.__version__
    for half in (False, True):
        # fresh instance per precision: Ultralytics configures FP16 only when its predictor is first built
        yolo = YOLO(best)
        tag = "fp16" if half else "fp32"
        res[f"yolo26n_seg_e2e_{tag}"] = timed(
            lambda y=yolo, h=half: (
                y.predict(img3, imgsz=480, quantize=16 if h else 32, conf=0.25, verbose=False)[0].masks
            )
        )
        del yolo
    net = YOLO(best).model.float().to(dev).eval()
    if hasattr(net, "fuse"):
        net = net.fuse()
    res["yolo26n_seg_params"] = int(sum(p.numel() for p in net.parameters()))
    x480 = torch.rand(1, 3, 480, 480, device=dev)
    res["yolo26n_seg_model_fp32"] = timed(lambda: net(x480))
    net16 = net.half()
    x480h = x480.half()
    res["yolo26n_seg_model_fp16"] = timed(lambda: net16(x480h))

    # ---------------------------------------------------------------- U-Net-R18 (256 px)
    unet = UNet(base=32, encoder="resnet18", pretrained=False).to(dev).eval()
    res["unet_r18_params"] = int(sum(p.numel() for p in unet.parameters()))
    x256 = torch.randn(1, 3, 256, 256, device=dev)
    for amp in (False, True):
        tag = "fp16" if amp else "fp32"

        def unet_model(a=amp):
            with torch.autocast("cuda", dtype=torch.float16, enabled=a):
                return unet(x256)

        def unet_e2e(a=amp):
            with torch.autocast("cuda", dtype=torch.float16, enabled=a):
                return (torch.sigmoid(unet(prep(img, 256, dev)).float()) >= 0.5).cpu()

        res[f"unet_r18_model_{tag}"] = timed(unet_model)
        res[f"unet_r18_e2e_{tag}"] = timed(unet_e2e)

    # ---------------------------------------------------------------- PatchCore (WRN-50, 256 px)
    emb = PatchCoreEmbedder("wide_resnet50_2", (2, 3), 1024, pretrained=False).to(dev).eval()
    res["patchcore_embedder_params"] = int(sum(p.numel() for p in emb.parameters()))
    bank = torch.randn(38740, 1024, device=dev)
    bank16 = bank.half()
    bank16_n = bank16.float().square().sum(1)
    for amp in (False, True):
        tag = "fp16" if amp else "fp32"

        def pc(x, a=amp):
            with torch.autocast("cuda", dtype=torch.float16, enabled=a):
                e = emb(x)
            q = e.reshape(-1, 1024).float()
            if a:
                d, _ = knn_torch_fp16(q, None, 1, bank16=bank16, bank_norm=bank16_n)
            else:
                d, _ = knn_torch(q, bank, 1)
            patch = d[:, 0].clamp_min(0).sqrt().reshape(1, 32, 32)
            return patch.amax(), smooth_maps(patch, 224, 4.0)

        res[f"patchcore_model_{tag}"] = timed(lambda f=pc: f(x256))
        res[f"patchcore_e2e_{tag}"] = timed(lambda f=pc: [t.cpu() for t in f(prep(img, 256, dev))])

    (RUN / "latency_l4.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
