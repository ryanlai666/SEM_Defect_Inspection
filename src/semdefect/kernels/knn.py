"""k-nearest-neighbour search over a PatchCore / AnomalyDINO memory bank.

Backends (all return *squared* L2 distances, ascending, shape (M,k), plus int64 indices):

* ``torch`` - reference: chunked ``torch.cdist`` (cuBLAS GEMM path) + ``torch.topk``.
              Materialises a (chunk x N) distance tile per chunk.
* ``cupy``  - fused tiled-GEMM + register top-k kernel compiled at runtime with NVRTC
              (prototype, `semdefect/kernels/knn_topk.cu`).
* ``ext``   - the same kernel compiled ahead-of-time by nvcc into the pybind11 module
              ``semdefect_cuda`` (cpp/), called zero-copy on torch tensors.
* ``torch_fp16`` - FP16 reference: cuBLAS half GEMM + topk (fair baseline for ``ext_fp16``).
* ``ext_fp16`` - k = 1 only: tensor-core (WMMA) variant, FP16 inputs / FP32 accumulation (nvcc only).

FAISS-GPU has no Windows pip wheel, hence the custom kernel.
"""

from __future__ import annotations

import torch

_BACKENDS = ("torch", "torch_fp16", "cupy", "ext", "ext_fp16")


def knn_torch(
    q: torch.Tensor, bank: torch.Tensor, k: int = 1, chunk: int | None = None, tile_mb: int = 256
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference: chunked cdist + topk.  The query chunk is sized so that one (chunk x N) FP32 distance
    tile stays within `tile_mb` (a fixed chunk would need 4096 x 557k x 4 B = 9 GB for a full
    AnomalyDINO bank)."""
    if chunk is None:
        chunk = max(64, int(tile_mb * 2**20 // (4 * max(1, bank.shape[0]))))
    dists, idxs = [], []
    for s in range(0, q.shape[0], chunk):
        d = torch.cdist(q[s : s + chunk].float(), bank.float(), compute_mode="use_mm_for_euclid_dist")
        v, i = torch.topk(d, k, dim=1, largest=False, sorted=True)
        del d
        dists.append(v.square())
        idxs.append(i)
    return torch.cat(dists), torch.cat(idxs)


def knn_torch_fp16(
    q: torch.Tensor,
    bank: torch.Tensor,
    k: int = 1,
    bank16: torch.Tensor | None = None,
    bank_norm: torch.Tensor | None = None,
    tile_mb: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP16 PyTorch reference (the fair baseline for the tensor-core kernel): cuBLAS half GEMM for q.x,
    norms of the FP16-rounded vectors in FP32, distances in FP32, chunked top-k."""
    bank16 = bank.half() if bank16 is None else bank16
    bn = bank16.float().square().sum(1) if bank_norm is None else bank_norm
    q16 = q.half()
    qn = q16.float().square().sum(1)
    chunk = max(64, int(tile_mb * 2**20 // (4 * max(1, bank16.shape[0]))))
    dists, idxs = [], []
    for s in range(0, q16.shape[0], chunk):
        dot = q16[s : s + chunk] @ bank16.T  # FP16 tile
        d = dot.float()  # one FP32 tile; epilogue in place (peak 1.5 tiles, not 4)
        del dot
        d.mul_(-2.0).add_(bn[None, :]).add_(qn[s : s + chunk, None]).clamp_min_(0)
        v, i = torch.topk(d, k, dim=1, largest=False, sorted=True)
        del d
        dists.append(v)
        idxs.append(i)
    return torch.cat(dists), torch.cat(idxs)


def knn(
    q: torch.Tensor, bank: torch.Tensor, k: int = 1, backend: str = "torch", chunk: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {_BACKENDS}")
    if backend == "torch" or q.device.type != "cuda":
        return knn_torch(q, bank, k, chunk)
    if backend == "torch_fp16":
        return knn_torch_fp16(q, bank, k)
    if backend == "cupy":
        from .cupy_kernels import knn_topk_cupy

        return knn_topk_cupy(q, bank, k)
    if backend == "ext_fp16" and k == 1 and q.shape[1] % 32 == 0:
        from .ext import knn_top1_fp16_ext

        return knn_top1_fp16_ext(q, bank, 1)
    from .ext import knn_topk_ext

    return knn_topk_ext(q, bank, k)
