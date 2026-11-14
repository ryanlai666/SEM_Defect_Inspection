"""NVRTC prototypes (CuPy RawModule) of the fused preprocessing and kNN top-k kernels.

The kernel source (`cuda/*.cuh`) is the *same file* that nvcc compiles in `cpp/`; this module
JIT-compiles it at first use.  Tensors are shared zero-copy with PyTorch via raw device pointers
on PyTorch's current stream.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

CUDA_DIR = Path(__file__).with_name("cuda")
SUPPORTED_K = (1, 2, 4, 8, 16, 32)


@lru_cache(maxsize=1)
def _module():
    import cupy as cp

    src = (CUDA_DIR / "preprocess.cuh").read_text() + "\n" + (CUDA_DIR / "knn_topk.cuh").read_text()
    names = ["sem_preprocess_u8", "sem_preprocess_u16", "row_sqnorm"]
    names += [f"knn_tile_k{k}" for k in SUPPORTED_K] + [f"knn_merge_k{k}" for k in SUPPORTED_K]
    mod = cp.RawModule(code=src, options=("-std=c++17", "--use_fast_math"), name_expressions=None)
    return {n: mod.get_function(n) for n in names}


def _stream():
    import cupy as cp

    return cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)


def _p(t: torch.Tensor):
    import cupy as cp

    return cp.uint64(t.data_ptr())


def round_k(k: int) -> int:
    for s in SUPPORTED_K:
        if k <= s:
            return s
    raise ValueError(f"k={k} > {SUPPORTED_K[-1]} not supported")


def num_splits(m: int, n: int, sm_count: int, bn: int = 64, bm: int = 64) -> int:
    """Split the bank across blocks until the grid has >= 4 blocks per SM (or tiles run out)."""
    m_blocks = (m + bm - 1) // bm
    n_tiles = (n + bn - 1) // bn
    want = max(1, (4 * sm_count + m_blocks - 1) // m_blocks)
    return int(max(1, min(want, n_tiles, 64)))


def row_sqnorm_cupy(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous().float()
    out = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
    warps_per_block = 8
    grid = ((x.shape[0] + warps_per_block - 1) // warps_per_block,)
    import cupy as cp

    with _stream():
        _module()["row_sqnorm"](
            grid, (32 * warps_per_block,), (_p(x), cp.int32(x.shape[0]), cp.int32(x.shape[1]), _p(out))
        )
    return out


def knn_topk_cupy(
    q: torch.Tensor,
    bank: torch.Tensor,
    k: int = 1,
    bank_norm: torch.Tensor | None = None,
    splits: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Squared-L2 k-NN of each query row in `bank`. Returns (dist (M,k) ascending, idx (M,k) int64)."""
    import cupy as cp

    q = q.contiguous().float()
    bank = bank.contiguous().float()
    m, d = q.shape
    n = bank.shape[0]
    kk = round_k(k)
    if bank_norm is None:
        bank_norm = row_sqnorm_cupy(bank)
    q_norm = row_sqnorm_cupy(q)
    sm = torch.cuda.get_device_properties(q.device).multi_processor_count
    s = splits or num_splits(m, n, sm)
    nps = ((n + s - 1) // s + 63) // 64 * 64
    s = (n + nps - 1) // nps
    part_d = torch.empty(m * s * kk, device=q.device, dtype=torch.float32)
    part_i = torch.empty(m * s * kk, device=q.device, dtype=torch.int32)
    out_d = torch.empty(m, kk, device=q.device, dtype=torch.float32)
    out_i = torch.empty(m, kk, device=q.device, dtype=torch.int64)
    f = _module()
    with _stream():
        f[f"knn_tile_k{kk}"](
            ((m + 63) // 64, s),
            (256,),
            (
                _p(q),
                _p(bank),
                _p(bank_norm),
                cp.int32(m),
                cp.int32(n),
                cp.int32(d),
                cp.int32(nps),
                _p(part_d),
                _p(part_i),
            ),
        )
        f[f"knn_merge_k{kk}"](
            ((m + 127) // 128,),
            (128,),
            (_p(part_d), _p(part_i), _p(q_norm), cp.int32(m), cp.int32(s), _p(out_d), _p(out_i)),
        )
    return out_d[:, :k], out_i[:, :k]


def preprocess_cupy(
    raw: torch.Tensor,
    out_size: int,
    crop: tuple[int, int, int, int],
    mode: str = "area",
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    max_value: float | None = None,
) -> torch.Tensor:
    """raw: (B,H,W) uint8 or (B,H,W) int16/uint16-as-int32 tensor on GPU; crop = (y0, x0, ch, cw)."""
    import cupy as cp

    b, h, w = raw.shape
    y0, x0, ch, cw = crop
    if raw.dtype == torch.uint8:
        fn, scale = "sem_preprocess_u8", 1.0 / (max_value or 255.0)
    elif raw.dtype in (torch.int16, torch.uint16):
        fn, scale = "sem_preprocess_u16", 1.0 / (max_value or 65535.0)
    else:
        raise TypeError(raw.dtype)
    raw = raw.contiguous()
    out = torch.empty(b, 3, out_size, out_size, device=raw.device, dtype=torch.float32)
    total = b * out_size * out_size
    grid = (min((total + 255) // 256, 65535),)
    args = [_p(raw), _p(out)] + [
        cp.int32(v) for v in (b, h, w, y0, x0, ch, cw, out_size, out_size, 0 if mode == "area" else 1)
    ]
    args += [cp.float32(scale)] + [cp.float32(v) for v in mean] + [cp.float32(1.0 / s) for s in std]
    with _stream():
        _module()[fn](grid, (256,), tuple(args))
    return out
