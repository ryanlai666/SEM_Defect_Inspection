"""ONNX export of the deployable anomaly-detection graphs.

* PatchCore embedder: WideResNet-50 layer2/3 + local aggregation + channel pooling,
  (B,3,S,S) float32 -> (B, (S/8)^2, D).  The kNN against the memory bank runs in the custom CUDA
  kernel (not in the graph), exactly like the C++ runner.
* EfficientAD-S: teacher + student + autoencoder + quantile normalisation -> (B,1,64,64) map.

Batch is a dynamic axis; opset 17.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..models import EfficientAD, PatchCoreEmbedder


def export_onnx(
    model: torch.nn.Module,
    path: str | Path,
    img_size: int,
    input_name="image",
    output_name="output",
    opset: int = 17,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval().cpu().float()
    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=[input_name],
        output_names=[output_name],
        opset_version=opset,
        dynamic_axes={input_name: {0: "batch"}, output_name: {0: "batch"}},
        do_constant_folding=True,
        dynamo=False,
    )
    import onnx

    onnx.checker.check_model(onnx.load(str(path)))
    return path


def export_patchcore_embedder(
    path: str | Path,
    img_size: int = 256,
    backbone: str = "wide_resnet50_2",
    embed_dim: int = 1024,
    pretrained: bool = True,
) -> Path:
    emb = PatchCoreEmbedder(backbone, (2, 3), embed_dim, pretrained=pretrained)
    return export_onnx(emb, path, img_size, output_name="embedding")


def export_efficientad(
    path: str | Path, state_dict: dict | None = None, img_size: int = 256, pretrained: bool = True
) -> Path:
    m = EfficientAD(pretrained_teacher=pretrained and state_dict is None)
    if state_dict is not None:
        m.load_state_dict(state_dict)
    return export_onnx(m, path, img_size, output_name="anomaly_map")
