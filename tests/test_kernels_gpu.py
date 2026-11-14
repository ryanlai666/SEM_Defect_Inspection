"""Parity tests: custom CUDA kernels (NVRTC/CuPy prototype and nvcc pybind11 module) vs PyTorch.

Skipped automatically without a GPU (CI).  Locally run through the shared-GPU guard:
    python scripts/guarded.py --gb 1.5 -- .venv/Scripts/python.exe -m pytest -m gpu tests/test_kernels_gpu.py
"""

import pytest
import torch
import torch.nn.functional as F

from semdefect.data.carinthia import to_model_input
from semdefect.kernels.knn import knn_torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _backends():
    out = []
    try:
        import cupy  # noqa: F401

        from semdefect.kernels.cupy_kernels import knn_topk_cupy, preprocess_cupy

        out.append(pytest.param((knn_topk_cupy, preprocess_cupy), id="cupy-nvrtc"))
    except ImportError:
        pass
    from semdefect.kernels import ext

    if ext.available():
        out.append(pytest.param((ext.knn_topk_ext, ext.preprocess_ext), id="nvcc-ext", marks=pytest.mark.ext))
    return out or [pytest.param(None, marks=pytest.mark.skip(reason="no kernel backend available"))]


@pytest.fixture(autouse=True)
def _cap():
    from semdefect.utils import apply_gpu_guard_cap

    apply_gpu_guard_cap()


@cuda
@pytest.mark.gpu
@pytest.mark.parametrize("fns", _backends())
@pytest.mark.parametrize(
    "m,n,d,k",
    [(1, 1, 1, 1), (37, 129, 7, 1), (1024, 20_000, 1024, 1), (300, 5000, 384, 9), (64, 4096, 384, 32)],
)
def test_knn_matches_torch(fns, m, n, d, k):
    knn_fn = fns[0]
    g = torch.Generator(device="cuda").manual_seed(m * 7 + n)
    q = torch.randn(m, d, device="cuda", generator=g)
    b = torch.randn(n, d, device="cuda", generator=g)
    d1, i1 = knn_fn(q, b, k)
    d0, i0 = knn_torch(q, b, k)
    torch.cuda.synchronize()
    assert d1.shape == (m, k) and i1.dtype == torch.int64
    torch.testing.assert_close(d1, d0, rtol=1e-4, atol=1e-3)
    assert (i1 == i0).float().mean() > 0.995  # only near-ties may differ


@cuda
@pytest.mark.gpu
@pytest.mark.parametrize("fns", _backends())
@pytest.mark.parametrize("mode", ["area", "bilinear"])
@pytest.mark.parametrize("size", [224, 256, 448])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.int16])
def test_preprocess_matches_torch(fns, mode, size, dtype):
    pre_fn = fns[1]
    hi = 255 if dtype == torch.uint8 else 32767
    raw = torch.randint(0, hi + 1, (3, 480, 480), device="cuda").to(dtype)
    out = pre_fn(raw, size, (1, 0, 477, 477), mode)
    x = raw[:, 1:478, 0:477].float().unsqueeze(1) / (255.0 if dtype == torch.uint8 else 65535.0)
    if mode == "area":
        ref = F.adaptive_avg_pool2d(x, size)
    else:
        ref = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=False)
    torch.testing.assert_close(out, to_model_input(ref), rtol=0, atol=5e-5)


@cuda
@pytest.mark.gpu
def test_patchcore_backends_agree():
    from semdefect.kernels.knn import knn

    q = torch.randn(2048, 1024, device="cuda")
    bank = torch.randn(10_000, 1024, device="cuda")
    ref, _ = knn(q, bank, 1, "torch")
    for be in ("cupy", "ext"):
        try:
            got, _ = knn(q, bank, 1, be)
        except ImportError:
            continue
        torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-3)


@cuda
@pytest.mark.gpu
@pytest.mark.parametrize("m,n,d", [(1, 1, 32), (37, 129, 32), (1024, 20_000, 1024), (2048, 131_072, 384)])
def test_tensor_core_top1_matches_fp32(m, n, d):
    from semdefect.kernels import ext

    if not ext.available():
        pytest.skip("extension not built")
    g = torch.Generator(device="cuda").manual_seed(m + n)
    q = torch.randn(m, d, device="cuda", generator=g)
    b = torch.randn(n, d, device="cuda", generator=g)
    d1, i1 = ext.knn_top1_fp16_ext(q, b)
    d0, i0 = knn_torch(q.half().float(), b.half().float(), 1)  # same FP16-rounded inputs
    torch.testing.assert_close(d1, d0, rtol=2e-3, atol=2e-2)
    assert (i1 == i0).float().mean() > 0.99


@cuda
@pytest.mark.gpu
def test_torch_fp16_baseline_matches_fp32_on_rounded_inputs():
    from semdefect.kernels.knn import knn_torch_fp16

    g = torch.Generator(device="cuda").manual_seed(1)
    q = torch.randn(512, 256, device="cuda", generator=g)
    b = torch.randn(4000, 256, device="cuda", generator=g)
    d1, i1 = knn_torch_fp16(q, b, 1)
    d0, i0 = knn_torch(q.half().float(), b.half().float(), 1)
    torch.testing.assert_close(d1, d0, rtol=2e-3, atol=2e-2)
    assert (i1 == i0).float().mean() > 0.99
