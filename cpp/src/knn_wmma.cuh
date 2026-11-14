// Tensor-core (WMMA) nearest-neighbour (k = 1) search: FP16 inputs, FP32 accumulation.
//
// Motivation (docs/benchmarks.md): after the TensorRT FP16 embedder, the FP32 SIMT kNN kernel is ~70 %
// of PatchCore's end-to-end latency at batch 8.  PatchCore and AnomalyDINO only need the nearest
// neighbour, so this kernel specialises k = 1 and moves the q.x products onto tensor cores.
//
// Block = 4 warps, 64 queries x 64 bank points per tile; each warp owns a 32x32 sub-tile (2x2 WMMA
// 16x16x16 fragments).  Q/X tiles (64 x 32 halves) are staged in shared memory with 128-bit loads,
// the FP32 tile of dot products is written back to shared memory and folded into a per-thread running
// minimum of ||x||^2 - 2 q.x (2 threads per query row, merged with a warp shuffle).  Same split-bank
// grid + merge kernel as the FP32 path.  Requires D % 32 == 0 and sm_70+.
#pragma once

#include <cuda_fp16.h>
#include <mma.h>

#define KW_BM 64
#define KW_BN 64
#define KW_BK 32
#define KW_PAD 8
#define KW_THREADS 128

__global__ void __launch_bounds__(KW_THREADS)
    knn_top1_wmma_kernel(const __half* __restrict__ Q, const __half* __restrict__ X, const float* __restrict__ xnorm,
                         int M, int N, int D, int nps, float* __restrict__ part_d, int* __restrict__ part_i) {
    using namespace nvcuda;
    __shared__ __align__(32) __half As[KW_BM][KW_BK + KW_PAD];
    __shared__ __align__(32) __half Bs[KW_BN][KW_BK + KW_PAD];
    __shared__ __align__(32) float Cs[KW_BM][KW_BN + 4];

    const int tid = threadIdx.x, warp = tid >> 5;
    const int wr = warp >> 1, wc = warp & 1;  // warp sub-tile: rows wr*32.., cols wc*32..
    const int m0 = blockIdx.x * KW_BM;
    const int S = gridDim.y, s = blockIdx.y;
    const int n_begin = s * nps, n_end = min(N, n_begin + nps);
    const int row = tid >> 1, half_ = tid & 1;  // top-1 owner: query row, 32-column half

    float best = 3.402823466e+38f;
    int best_i = -1;
    const uint4 zero = make_uint4(0u, 0u, 0u, 0u);

    for (int n0 = n_begin; n0 < n_end; n0 += KW_BN) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.f);

        for (int k0 = 0; k0 < D; k0 += KW_BK) {
#pragma unroll
            for (int l = 0; l < 2; ++l) {  // 64 rows x 32 halves = 256 x uint4 per operand, 2 per thread
                const int e = tid + l * KW_THREADS;
                const int r = e >> 2, c8 = (e & 3) * 8;
                const int gm = m0 + r, gn = n0 + r;
                *reinterpret_cast<uint4*>(&As[r][c8]) =
                    gm < M ? *reinterpret_cast<const uint4*>(Q + (long long)gm * D + k0 + c8) : zero;
                *reinterpret_cast<uint4*>(&Bs[r][c8]) =
                    gn < n_end ? *reinterpret_cast<const uint4*>(X + (long long)gn * D + k0 + c8) : zero;
            }
            __syncthreads();
#pragma unroll
            for (int kk = 0; kk < KW_BK; kk += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a[2];
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b[2];
#pragma unroll
                for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(a[i], &As[wr * 32 + i * 16][kk], KW_BK + KW_PAD);
#pragma unroll
                for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(b[j], &Bs[wc * 32 + j * 16][kk], KW_BK + KW_PAD);
#pragma unroll
                for (int i = 0; i < 2; ++i)
#pragma unroll
                    for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
            }
            __syncthreads();
        }
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int j = 0; j < 2; ++j)
                wmma::store_matrix_sync(&Cs[wr * 32 + i * 16][wc * 32 + j * 16], acc[i][j], KW_BN + 4,
                                        wmma::mem_row_major);
        __syncthreads();
#pragma unroll 8
        for (int c = 0; c < 32; ++c) {
            const int col = half_ * 32 + c, gn = n0 + col;
            if (gn < n_end) {
                const float v = fmaf(-2.f, Cs[row][col], xnorm[gn]);
                if (v < best) {
                    best = v;
                    best_i = gn;
                }
            }
        }
        __syncthreads();
    }
    // the two halves of a row are adjacent lanes of the same warp
    const float ob = __shfl_xor_sync(0xffffffffu, best, 1);
    const int oi = __shfl_xor_sync(0xffffffffu, best_i, 1);
    if (ob < best) {
        best = ob;
        best_i = oi;
    }
    const int gm = m0 + row;
    if (half_ == 0 && gm < M) {
        part_d[(long long)gm * S + s] = best;
        part_i[(long long)gm * S + s] = best_i;
    }
}

__global__ void f32_to_f16_kernel(const float* __restrict__ in, __half* __restrict__ out, long long n) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        out[i] = __float2half_rn(in[i]);
}

// Squared norms of FP16 rows accumulated in FP32 (consistent with the tensor-core dot products).
__global__ void row_sqnorm_f16_kernel(const __half* __restrict__ X, int N, int D, float* __restrict__ out) {
    const int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (w >= N) return;
    const __half* r = X + (long long)w * D;
    float acc = 0.f;
    for (int k = lane; k < D; k += 32) {
        const float v = __half2float(r[k]);
        acc = fmaf(v, v, acc);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) out[w] = acc;
}
