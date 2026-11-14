// Host API of the semdefect CUDA kernels (nvcc build of src/semdefect/kernels/cuda/*.cuh).
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace semdefect {

#define SEMDEFECT_CUDA_CHECK(expr)                                                                     \
    do {                                                                                               \
        cudaError_t _e = (expr);                                                                       \
        if (_e != cudaSuccess)                                                                         \
            throw std::runtime_error(std::string(#expr) + " failed: " + cudaGetErrorString(_e) + " @ " + \
                                     __FILE__ + ":" + std::to_string(__LINE__));                       \
    } while (0)

enum class ResizeMode : int { Area = 0, Bilinear = 1 };

struct PreprocessParams {
    int y0 = 0, x0 = 0, ch = 0, cw = 0;  // crop window inside the raw image
    int oh = 224, ow = 224;              // network input size
    ResizeMode mode = ResizeMode::Area;
    float in_scale = 1.f / 255.f;        // 1/255 (8-bit) or 1/65535 (16-bit detector)
    float mean[3] = {0.485f, 0.456f, 0.406f};
    float inv_std[3] = {1.f / 0.229f, 1.f / 0.224f, 1.f / 0.225f};
};

// (B,H,W) raw grey -> (B,3,oh,ow) float32 normalised network input.
cudaError_t preprocess_u8(const uint8_t* src, float* dst, int B, int H, int W, const PreprocessParams& p,
                          cudaStream_t stream);
cudaError_t preprocess_u16(const uint16_t* src, float* dst, int B, int H, int W, const PreprocessParams& p,
                           cudaStream_t stream);

// Squared L2 norm of each of the N rows of a (N,D) row-major matrix.
cudaError_t row_sqnorm(const float* x, int N, int D, float* out, cudaStream_t stream);

// k-NN helpers.  K must be one of 1,2,4,8,16,32 (see knn_round_k).
int knn_round_k(int k);
int knn_num_splits(int M, int N, int sm_count);
int knn_n_per_split(int N, int splits);  // multiple of the 64-point bank tile
inline int knn_effective_splits(int N, int nps) { return (N + nps - 1) / nps; }

// part_d/part_i: workspace of M*S*K floats / ints.  out_d: (M,K) squared L2, out_i: (M,K) int64.
cudaError_t knn_topk(const float* q, const float* x, const float* xnorm, const float* qnorm, int M, int N, int D,
                     int K, int S, int nps, float* part_d, int* part_i, float* out_d, int64_t* out_i,
                     cudaStream_t stream);

// ---- Tensor-core k = 1 path (FP16 inputs, FP32 accumulation; D % 32 == 0, sm_70+). Buffers typed void* =
// device arrays of __half.  part_d/part_i: M*S workspace; out_d (M) squared L2, out_i (M) int64.
cudaError_t f32_to_f16(const float* in, void* out_half, long long n, cudaStream_t stream);
cudaError_t row_sqnorm_f16(const void* x_half, int N, int D, float* out, cudaStream_t stream);
cudaError_t knn_top1_f16(const void* q_half, const void* x_half, const float* xnorm, const float* qnorm, int M, int N,
                         int D, int S, int nps, float* part_d, int* part_i, float* out_d, int64_t* out_i,
                         cudaStream_t stream);

// PatchCore image score: out[b] = sqrt(max_p d[b, p]) for (B, P) squared distances.
cudaError_t image_max_sqrt(const float* d, int B, int P, float* out, cudaStream_t stream);

// Anomaly maps (mirrors semdefect.models.anomaly.smooth_maps): (B,h,w) squared NN distances -> sqrt ->
// bilinear upsample to (S,S) (align_corners=False) -> separable Gaussian blur, reflect padding.
// taps: device array of 2*radius+1 weights (see gaussian_taps); tmp/out: device buffers of B*S*S floats.
std::vector<float> gaussian_taps(float sigma);
cudaError_t anomaly_maps(const float* d_sq, int B, int h, int w, int S, const float* taps, int radius, float* tmp,
                         float* out, cudaStream_t stream);

// RAII convenience: device-resident memory bank with cached norms and a reusable workspace.
class KnnIndex {
   public:
    KnnIndex(const float* host_bank, int N, int D);
    ~KnnIndex();
    KnnIndex(const KnnIndex&) = delete;
    KnnIndex& operator=(const KnnIndex&) = delete;
    // q: device (M,D) queries.  out_d/out_i: device buffers of M*K.
    void search(const float* q, int M, int K, float* out_d, int64_t* out_i, cudaStream_t stream);
    int size() const { return N_; }
    int dim() const { return D_; }

   private:
    void reserve(size_t m_s_k, int M);
    float* bank_ = nullptr;
    float* bank_norm_ = nullptr;
    float* q_norm_ = nullptr;
    float* part_d_ = nullptr;
    int* part_i_ = nullptr;
    size_t ws_elems_ = 0;
    int q_cap_ = 0;
    int N_, D_, sm_count_ = 1;
};

}  // namespace semdefect
