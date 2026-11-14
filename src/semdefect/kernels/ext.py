"""Loader + torch-facing wrappers for the nvcc-compiled pybind11 module `semdefect_cuda` (cpp/).

Build it with `cpp\\build.bat` (Windows, MSVC + nvcc from the project-local conda env) or
`cmake -S cpp -B cpp/build && cmake --build cpp/build` on Linux/Colab; the module is copied next to
this file.  All calls pass raw device pointers of torch tensors and PyTorch's current CUDA stream.
"""

from __future__ import annotations

import importlib
from functools import lru_cache

import torch

from .cupy_kernels import num_splits, round_k


@lru_cache(maxsize=1)
def _ext():
    try:
        return importlib.import_module("semdefect.kernels.semdefect_cuda")
    except ImportError as e:  # pragma: no cover - depends on the local build
        raise ImportError(
            "semdefect_cuda extension not built - run cpp\\build.bat (see README, Part B)"
        ) from e


def available() -> bool:
    try:
        _ext()
        return True
    except ImportError:
        return False


def _stream() -> int:
    return torch.cuda.current_stream().cuda_stream


def row_sqnorm_ext(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous().float()
    out = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
    _ext().row_sqnorm(x.data_ptr(), x.shape[0], x.shape[1], out.data_ptr(), _stream())
    return out


def knn_topk_ext(
    q: torch.Tensor,
    bank: torch.Tensor,
    k: int = 1,
    bank_norm: torch.Tensor | None = None,
    splits: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = q.contiguous().float()
    bank = bank.contiguous().float()
    m, d = q.shape
    n = bank.shape[0]
    kk = round_k(k)
    bank_norm = row_sqnorm_ext(bank) if bank_norm is None else bank_norm
    q_norm = row_sqnorm_ext(q)
    sm = torch.cuda.get_device_properties(q.device).multi_processor_count
    s = splits or num_splits(m, n, sm)
    nps = ((n + s - 1) // s + 63) // 64 * 64
    s = (n + nps - 1) // nps
    part_d = torch.empty(m * s * kk, device=q.device, dtype=torch.float32)
    part_i = torch.empty(m * s * kk, device=q.device, dtype=torch.int32)
    out_d = torch.empty(m, kk, device=q.device, dtype=torch.float32)
    out_i = torch.empty(m, kk, device=q.device, dtype=torch.int64)
    _ext().knn_topk(
        q.data_ptr(),
        bank.data_ptr(),
        bank_norm.data_ptr(),
        q_norm.data_ptr(),
        m,
        n,
        d,
        kk,
        s,
        nps,
        part_d.data_ptr(),
        part_i.data_ptr(),
        out_d.data_ptr(),
        out_i.data_ptr(),
        _stream(),
    )
    return out_d[:, :k], out_i[:, :k]


def preprocess_ext(
    raw: torch.Tensor,
    out_size: int,
    crop: tuple[int, int, int, int],
    mode: str = "area",
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    max_value: float | None = None,
) -> torch.Tensor:
    b, h, w = raw.shape
    y0, x0, ch, cw = crop
    is16 = raw.dtype in (torch.int16, torch.uint16)
    scale = 1.0 / (max_value or (65535.0 if is16 else 255.0))
    raw = raw.contiguous()
    out = torch.empty(b, 3, out_size, out_size, device=raw.device, dtype=torch.float32)
    _ext().preprocess(
        raw.data_ptr(),
        is16,
        out.data_ptr(),
        b,
        h,
        w,
        y0,
        x0,
        ch,
        cw,
        out_size,
        out_size,
        0 if mode == "area" else 1,
        scale,
        list(mean),
        [1.0 / s for s in std],
        _stream(),
    )
    return out


def knn_top1_fp16_ext(
    q: torch.Tensor,
    bank: torch.Tensor,
    k: int = 1,
    bank16: torch.Tensor | None = None,
    bank_norm: torch.Tensor | None = None,
    splits: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tensor-core nearest neighbour (k must be 1): FP16 inputs, FP32 accumulation, D % 32 == 0.
    Returns squared L2 (M,1) float32 and indices (M,1) int64.  Pass a pre-converted `bank16` and its
    `bank_norm` (from `row_sqnorm_f16_ext`) to avoid re-converting the memory bank on every call."""
    if k != 1:
        raise ValueError("the tensor-core path implements k = 1 only")
    m, d = q.shape
    if d % 32:
        raise ValueError("tensor-core path needs D % 32 == 0")
    q16 = q.contiguous().half()
    bank16 = bank.contiguous().half() if bank16 is None else bank16
    n = bank16.shape[0]
    bank_norm = row_sqnorm_f16_ext(bank16) if bank_norm is None else bank_norm
    q_norm = row_sqnorm_f16_ext(q16)
    sm = torch.cuda.get_device_properties(q.device).multi_processor_count
    s = splits or num_splits(m, n, sm)
    nps = ((n + s - 1) // s + 63) // 64 * 64
    s = (n + nps - 1) // nps
    part_d = torch.empty(m * s, device=q.device, dtype=torch.float32)
    part_i = torch.empty(m * s, device=q.device, dtype=torch.int32)
    out_d = torch.empty(m, 1, device=q.device, dtype=torch.float32)
    out_i = torch.empty(m, 1, device=q.device, dtype=torch.int64)
    _ext().knn_top1_f16(
        q16.data_ptr(),
        bank16.data_ptr(),
        bank_norm.data_ptr(),
        q_norm.data_ptr(),
        m,
        n,
        d,
        s,
        nps,
        part_d.data_ptr(),
        part_i.data_ptr(),
        out_d.data_ptr(),
        out_i.data_ptr(),
        _stream(),
    )
    return out_d, out_i


def row_sqnorm_f16_ext(x16: torch.Tensor) -> torch.Tensor:
    x16 = x16.contiguous().half()
    out = torch.empty(x16.shape[0], device=x16.device, dtype=torch.float32)
    _ext().row_sqnorm_f16(x16.data_ptr(), x16.shape[0], x16.shape[1], out.data_ptr(), _stream())
    return out
