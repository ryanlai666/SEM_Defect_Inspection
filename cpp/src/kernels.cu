// nvcc build of the shared kernel sources + host-side launchers.
#include "semdefect/kernels.h"

// Single source of truth shared with the NVRTC/CuPy prototype (src/semdefect/kernels/cuda/).
#include "knn_topk.cuh"
#include "preprocess.cuh"
#include "knn_wmma.cuh"

#include <algorithm>

namespace semdefect {

namespace {
int pp_grid(long long total) {
    return static_cast<int>(std::min<long long>((total + SEMDEFECT_PP_THREADS - 1) / SEMDEFECT_PP_THREADS, 65535));
}
}  // namespace

cudaError_t preprocess_u8(const uint8_t* src, float* dst, int B, int H, int W, const PreprocessParams& p,
                          cudaStream_t stream) {
    const long long total = static_cast<long long>(B) * p.oh * p.ow;
    sem_preprocess_u8<<<pp_grid(total), SEMDEFECT_PP_THREADS, 0, stream>>>(
        src, dst, B, H, W, p.y0, p.x0, p.ch, p.cw, p.oh, p.ow, static_cast<int>(p.mode), p.in_scale, p.mean[0],
        p.mean[1], p.mean[2], p.inv_std[0], p.inv_std[1], p.inv_std[2]);
    return cudaGetLastError();
}

cudaError_t preprocess_u16(const uint16_t* src, float* dst, int B, int H, int W, const PreprocessParams& p,
                           cudaStream_t stream) {
    const long long total = static_cast<long long>(B) * p.oh * p.ow;
    sem_preprocess_u16<<<pp_grid(total), SEMDEFECT_PP_THREADS, 0, stream>>>(
        src, dst, B, H, W, p.y0, p.x0, p.ch, p.cw, p.oh, p.ow, static_cast<int>(p.mode), p.in_scale, p.mean[0],
        p.mean[1], p.mean[2], p.inv_std[0], p.inv_std[1], p.inv_std[2]);
    return cudaGetLastError();
}

cudaError_t row_sqnorm(const float* x, int N, int D, float* out, cudaStream_t stream) {
    const int warps = 8;
    ::row_sqnorm<<<(N + warps - 1) / warps, 32 * warps, 0, stream>>>(x, N, D, out);
    return cudaGetLastError();
}

namespace {
__global__ void image_max_sqrt_kernel(const float* __restrict__ d, int P, float* __restrict__ out) {
    __shared__ float red[256];
    const float* row = d + static_cast<long long>(blockIdx.x) * P;
    float m = 0.f;
    for (int p = threadIdx.x; p < P; p += blockDim.x) m = fmaxf(m, row[p]);
    red[threadIdx.x] = m;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = sqrtf(red[0]);
}
}  // namespace

cudaError_t f32_to_f16(const float* in, void* out_half, long long n, cudaStream_t stream) {
    const int grid = static_cast<int>(std::min<long long>((n + 255) / 256, 65535));
    f32_to_f16_kernel<<<grid, 256, 0, stream>>>(in, static_cast<__half*>(out_half), n);
    return cudaGetLastError();
}

cudaError_t row_sqnorm_f16(const void* x_half, int N, int D, float* out, cudaStream_t stream) {
    row_sqnorm_f16_kernel<<<(N + 7) / 8, 256, 0, stream>>>(static_cast<const __half*>(x_half), N, D, out);
    return cudaGetLastError();
}

cudaError_t knn_top1_f16(const void* q_half, const void* x_half, const float* xnorm, const float* qnorm, int M, int N,
                         int D, int S, int nps, float* part_d, int* part_i, float* out_d, int64_t* out_i,
                         cudaStream_t stream) {
    if (M <= 0 || N <= 0 || D <= 0 || D % KW_BK != 0) return cudaErrorInvalidValue;
    const dim3 grid((M + KW_BM - 1) / KW_BM, S);
    knn_top1_wmma_kernel<<<grid, KW_THREADS, 0, stream>>>(static_cast<const __half*>(q_half),
                                                           static_cast<const __half*>(x_half), xnorm, M, N, D, nps,
                                                           part_d, part_i);
    knn_merge_k1<<<(M + 127) / 128, 128, 0, stream>>>(part_d, part_i, qnorm, M, S, out_d,
                                                       reinterpret_cast<long long*>(out_i));
    return cudaGetLastError();
}

cudaError_t image_max_sqrt(const float* d, int B, int P, float* out, cudaStream_t stream) {
    image_max_sqrt_kernel<<<B, 256, 0, stream>>>(d, P, out);
    return cudaGetLastError();
}

int knn_round_k(int k) {
    for (int s : {1, 2, 4, 8, 16, 32})
        if (k <= s) return s;
    throw std::invalid_argument("k > 32 not supported");
}

int knn_num_splits(int M, int N, int sm_count) {
    const int m_blocks = (M + KNN_BM - 1) / KNN_BM;
    const int n_tiles = (N + KNN_BN - 1) / KNN_BN;
    const int want = std::max(1, (4 * sm_count + m_blocks - 1) / m_blocks);
    return std::max(1, std::min({want, n_tiles, 64}));
}

int knn_n_per_split(int N, int splits) {
    const int per = (N + splits - 1) / splits;
    return (per + KNN_BN - 1) / KNN_BN * KNN_BN;
}

#define SEMDEFECT_KNN_CASE(KV)                                                                  \
    case KV:                                                                                    \
        knn_tile_k##KV<<<grid, KNN_THREADS, 0, stream>>>(q, x, xnorm, M, N, D, nps, part_d, part_i); \
        knn_merge_k##KV<<<(M + 127) / 128, 128, 0, stream>>>(part_d, part_i, qnorm, M, S, out_d,     \
                                                            reinterpret_cast<long long*>(out_i));    \
        break;

cudaError_t knn_topk(const float* q, const float* x, const float* xnorm, const float* qnorm, int M, int N, int D,
                     int K, int S, int nps, float* part_d, int* part_i, float* out_d, int64_t* out_i,
                     cudaStream_t stream) {
    if (M <= 0 || N <= 0 || D <= 0) return cudaErrorInvalidValue;
    const dim3 grid((M + KNN_BM - 1) / KNN_BM, S);
    switch (K) {
        SEMDEFECT_KNN_CASE(1)
        SEMDEFECT_KNN_CASE(2)
        SEMDEFECT_KNN_CASE(4)
        SEMDEFECT_KNN_CASE(8)
        SEMDEFECT_KNN_CASE(16)
        SEMDEFECT_KNN_CASE(32)
        default:
            return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

// ------------------------------------------------------------------------------------ KnnIndex
KnnIndex::KnnIndex(const float* host_bank, int N, int D) : N_(N), D_(D) {
    int dev = 0;
    SEMDEFECT_CUDA_CHECK(cudaGetDevice(&dev));
    SEMDEFECT_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count_, cudaDevAttrMultiProcessorCount, dev));
    SEMDEFECT_CUDA_CHECK(cudaMalloc(&bank_, sizeof(float) * N * D));
    SEMDEFECT_CUDA_CHECK(cudaMalloc(&bank_norm_, sizeof(float) * N));
    SEMDEFECT_CUDA_CHECK(cudaMemcpy(bank_, host_bank, sizeof(float) * N * D, cudaMemcpyHostToDevice));
    SEMDEFECT_CUDA_CHECK(row_sqnorm(bank_, N, D, bank_norm_, nullptr));
    SEMDEFECT_CUDA_CHECK(cudaDeviceSynchronize());
}

KnnIndex::~KnnIndex() {
    cudaFree(bank_);
    cudaFree(bank_norm_);
    cudaFree(q_norm_);
    cudaFree(part_d_);
    cudaFree(part_i_);
}

void KnnIndex::reserve(size_t m_s_k, int M) {
    if (m_s_k > ws_elems_) {
        cudaFree(part_d_);
        cudaFree(part_i_);
        SEMDEFECT_CUDA_CHECK(cudaMalloc(&part_d_, sizeof(float) * m_s_k));
        SEMDEFECT_CUDA_CHECK(cudaMalloc(&part_i_, sizeof(int) * m_s_k));
        ws_elems_ = m_s_k;
    }
    if (M > q_cap_) {
        cudaFree(q_norm_);
        SEMDEFECT_CUDA_CHECK(cudaMalloc(&q_norm_, sizeof(float) * M));
        q_cap_ = M;
    }
}

void KnnIndex::search(const float* q, int M, int K, float* out_d, int64_t* out_i, cudaStream_t stream) {
    if (knn_round_k(K) != K) throw std::invalid_argument("KnnIndex::search: K must be 1,2,4,8,16 or 32");
    const int S0 = knn_num_splits(M, N_, sm_count_);
    const int nps = knn_n_per_split(N_, S0);
    const int S = knn_effective_splits(N_, nps);
    reserve(static_cast<size_t>(M) * S * K, M);
    SEMDEFECT_CUDA_CHECK(row_sqnorm(q, M, D_, q_norm_, stream));
    SEMDEFECT_CUDA_CHECK(knn_topk(q, bank_, bank_norm_, q_norm_, M, N_, D_, K, S, nps, part_d_, part_i_, out_d, out_i,
                                  stream));
}

}  // namespace semdefect
