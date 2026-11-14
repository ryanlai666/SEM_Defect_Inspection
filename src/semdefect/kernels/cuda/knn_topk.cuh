// Fused tiled squared-L2 distance + top-k selection for PatchCore / AnomalyDINO memory banks.
// (single source for NVRTC and nvcc builds; no #includes)
//
//   d(q, x) = ||q||^2 + ||x||^2 - 2 q.x        q: (M, D) queries,  x: (N, D) memory bank
//
// Why fused: torch.cdist + topk materialises an M x N distance matrix (for a batch of 8 images
// x 1024 patches vs a 100k-patch bank that is 3.3 GB in FP32) and reads it back for the top-k.
// Here each block streams bank tiles through shared memory, accumulates q.x in registers
// (64x64 block tile, 4x4 per thread), and folds every 64x64 distance tile straight into per-query
// top-k lists held in registers; the distance matrix never touches global memory.  Workspace is
// O(M * splits * K).
//
// Parallelism: grid = (ceil(M/64), S).  For small M (a single image) the bank is split S ways so
// that all SMs are busy; a second kernel merges the S partial top-k lists and adds ||q||^2.
#pragma once

#define KNN_BM 64
#define KNN_BN 64
#define KNN_BK 16
#define KNN_THREADS 256
#define KNN_INF 3.402823466e+38f

template <int K>
__device__ __forceinline__ void knn_insert(float (&d)[K], int (&ix)[K], float v, int id) {
    if (v < d[K - 1]) {
        d[K - 1] = v;
        ix[K - 1] = id;
#pragma unroll
        for (int j = K - 1; j > 0; --j) {
            if (d[j] < d[j - 1]) {
                const float t = d[j];
                d[j] = d[j - 1];
                d[j - 1] = t;
                const int u = ix[j];
                ix[j] = ix[j - 1];
                ix[j - 1] = u;
            }
        }
    }
}

// part_d / part_i: (M, S, K) partial lists of (||x||^2 - 2 q.x, index), ascending.
template <int K>
__device__ __forceinline__ void knn_tile_impl(const float* __restrict__ Q, const float* __restrict__ X,
                                              const float* __restrict__ xnorm, int M, int N, int D,
                                              int n_per_split, float* __restrict__ part_d,
                                              int* __restrict__ part_i) {
    __shared__ float As[KNN_BK][KNN_BM + 1];
    __shared__ float Bs[KNN_BK][KNN_BN + 1];
    __shared__ float Ds[KNN_BM][KNN_BN + 1];

    const int tid = threadIdx.x;
    const int ty = tid / 16, tx = tid % 16;      // GEMM micro-tile owner: rows ty*4.., cols tx*4..
    const int tr = tid >> 2, tseg = (tid & 3) * 16;  // top-k owner: query row tr, 16-column segment
    const int m0 = blockIdx.x * KNN_BM;
    const int S = gridDim.y, s = blockIdx.y;
    const int n_begin = s * n_per_split;
    const int n_end = min(N, n_begin + n_per_split);

    float bd[K];
    int bi[K];
#pragma unroll
    for (int j = 0; j < K; ++j) {
        bd[j] = KNN_INF;
        bi[j] = -1;
    }

    for (int n0 = n_begin; n0 < n_end; n0 += KNN_BN) {
        float acc[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;

        for (int k0 = 0; k0 < D; k0 += KNN_BK) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {  // 64x16 tiles of Q and X, 4 elements per thread
                const int e = tid + l * KNN_THREADS;
                const int r = e / KNN_BK, c = e % KNN_BK;
                const int gk = k0 + c;
                const int gm = m0 + r, gn = n0 + r;
                As[c][r] = (gm < M && gk < D) ? Q[(long long)gm * D + gk] : 0.f;
                Bs[c][r] = (gn < n_end && gk < D) ? X[(long long)gn * D + gk] : 0.f;
            }
            __syncthreads();
#pragma unroll
            for (int kk = 0; kk < KNN_BK; ++kk) {
                float a[4], b[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    a[i] = As[kk][ty * 4 + i];
                    b[i] = Bs[kk][tx * 4 + i];
                }
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
            }
            __syncthreads();
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int col = tx * 4 + j, gn = n0 + col;
                Ds[ty * 4 + i][col] = (gn < n_end) ? fmaf(-2.f, acc[i][j], xnorm[gn]) : KNN_INF;
            }
        __syncthreads();
#pragma unroll 4
        for (int c = 0; c < 16; ++c) knn_insert<K>(bd, bi, Ds[tr][tseg + c], n0 + tseg + c);
        __syncthreads();
    }

    // Merge the 4 per-thread lists of each query (lanes 1..3 -> lane 0) through shared memory.
    float* md = &Ds[0][0];                              // 64*K floats
    int* mi = reinterpret_cast<int*>(md + KNN_BM * K);  // 64*K ints   (fits for K <= 32)
    for (int lane = 1; lane < 4; ++lane) {
        if ((tid & 3) == lane) {
#pragma unroll
            for (int j = 0; j < K; ++j) {
                md[tr * K + j] = bd[j];
                mi[tr * K + j] = bi[j];
            }
        }
        __syncthreads();
        if ((tid & 3) == 0) {
            for (int j = 0; j < K; ++j) {
                if (md[tr * K + j] >= bd[K - 1]) break;  // incoming list is sorted
                knn_insert<K>(bd, bi, md[tr * K + j], mi[tr * K + j]);
            }
        }
        __syncthreads();
    }
    const int gm = m0 + tr;
    if ((tid & 3) == 0 && gm < M) {
        float* od = part_d + ((long long)gm * S + s) * K;
        int* oi = part_i + ((long long)gm * S + s) * K;
#pragma unroll
        for (int j = 0; j < K; ++j) {
            od[j] = bd[j];
            oi[j] = bi[j];
        }
    }
}

// One thread per query: merge S partial lists, add ||q||^2, clamp >= 0 (fp cancellation).
template <int K>
__device__ __forceinline__ void knn_merge_impl(const float* __restrict__ part_d, const int* __restrict__ part_i,
                                               const float* __restrict__ qnorm, int M, int S,
                                               float* __restrict__ out_d, long long* __restrict__ out_i) {
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;
    float bd[K];
    int bi[K];
#pragma unroll
    for (int j = 0; j < K; ++j) {
        bd[j] = KNN_INF;
        bi[j] = -1;
    }
    const float* pd = part_d + (long long)m * S * K;
    const int* pi = part_i + (long long)m * S * K;
    for (int t = 0; t < S * K; ++t)
        if (pi[t] >= 0) knn_insert<K>(bd, bi, pd[t], pi[t]);
    const float qn = qnorm[m];
#pragma unroll
    for (int j = 0; j < K; ++j) {
        const float v = bd[j] + qn;
        out_d[(long long)m * K + j] = v > 0.f ? v : 0.f;
        out_i[(long long)m * K + j] = bi[j];
    }
}

// Squared L2 norm of each row: one warp per row.
extern "C" __global__ void row_sqnorm(const float* __restrict__ X, int N, int D, float* __restrict__ out) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= N) return;
    const float* row = X + (long long)warp * D;
    float acc = 0.f;
    for (int k = lane; k < D; k += 32) acc = fmaf(row[k], row[k], acc);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) out[warp] = acc;
}

#define SEMDEFECT_KNN_INSTANTIATE(K)                                                                          \
    extern "C" __global__ void __launch_bounds__(KNN_THREADS)                                                 \
        knn_tile_k##K(const float* __restrict__ Q, const float* __restrict__ X, const float* __restrict__ xn, \
                      int M, int N, int D, int nps, float* __restrict__ pd, int* __restrict__ pi) {          \
        knn_tile_impl<K>(Q, X, xn, M, N, D, nps, pd, pi);                                                    \
    }                                                                                                         \
    extern "C" __global__ void knn_merge_k##K(const float* __restrict__ pd, const int* __restrict__ pi,       \
                                              const float* __restrict__ qn, int M, int S,                     \
                                              float* __restrict__ od, long long* __restrict__ oi) {           \
        knn_merge_impl<K>(pd, pi, qn, M, S, od, oi);                                                          \
    }

SEMDEFECT_KNN_INSTANTIATE(1)
SEMDEFECT_KNN_INSTANTIATE(2)
SEMDEFECT_KNN_INSTANTIATE(4)
SEMDEFECT_KNN_INSTANTIATE(8)
SEMDEFECT_KNN_INSTANTIATE(16)
SEMDEFECT_KNN_INSTANTIATE(32)
