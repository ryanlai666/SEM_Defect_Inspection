// PatchCore anomaly-map post-processing on the GPU (mirrors semdefect.models.anomaly.smooth_maps):
//   (B, h, w) squared NN distances -> sqrt -> bilinear upsample to (S, S) (align_corners=False)
//   -> separable Gaussian blur (sigma, radius ceil(3 sigma), reflect padding) -> (B, S, S)
#include <cmath>
#include <vector>

#include "semdefect/kernels.h"

namespace semdefect {
namespace {

__device__ __forceinline__ int reflect_idx(int i, int n) {  // PyTorch 'reflect' (no edge repeat)
    if (n == 1) return 0;
    while (i < 0 || i >= n) i = i < 0 ? -i : 2 * (n - 1) - i;
    return i;
}

__global__ void upsample_sqrt_kernel(const float* __restrict__ d, int B, int h, int w, int S, float* __restrict__ out) {
    const long long total = static_cast<long long>(B) * S * S;
    const float sy = static_cast<float>(h) / S, sx = static_cast<float>(w) / S;
    for (long long t = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; t < total;
         t += static_cast<long long>(gridDim.x) * blockDim.x) {
        const int ox = static_cast<int>(t % S), oy = static_cast<int>((t / S) % S), b = static_cast<int>(t / (S * S));
        float fy = fmaxf(0.f, (oy + 0.5f) * sy - 0.5f), fx = fmaxf(0.f, (ox + 0.5f) * sx - 0.5f);
        const int iy = static_cast<int>(fy), ix = static_cast<int>(fx);
        const int py = iy < h - 1 ? 1 : 0, px = ix < w - 1 ? 1 : 0;
        const float ly = fy - iy, lx = fx - ix;
        const float* p = d + static_cast<long long>(b) * h * w;
        auto v = [&](int y, int x) { return sqrtf(fmaxf(p[y * w + x], 0.f)); };
        out[t] = (1.f - ly) * ((1.f - lx) * v(iy, ix) + lx * v(iy, ix + px)) +
                 ly * ((1.f - lx) * v(iy + py, ix) + lx * v(iy + py, ix + px));
    }
}

template <bool kHorizontal>
__global__ void blur_kernel(const float* __restrict__ in, int B, int S, const float* __restrict__ g, int r,
                            float* __restrict__ out) {
    const long long total = static_cast<long long>(B) * S * S;
    for (long long t = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; t < total;
         t += static_cast<long long>(gridDim.x) * blockDim.x) {
        const int x = static_cast<int>(t % S), y = static_cast<int>((t / S) % S);
        const float* img = in + (t / (static_cast<long long>(S) * S)) * S * S;
        float acc = 0.f;
        for (int k = -r; k <= r; ++k) {
            const float val = kHorizontal ? img[y * S + reflect_idx(x + k, S)] : img[reflect_idx(y + k, S) * S + x];
            acc = fmaf(g[k + r], val, acc);
        }
        out[t] = acc;
    }
}

int grid_for(long long n) { return static_cast<int>(std::min<long long>((n + 255) / 256, 65535)); }

}  // namespace

std::vector<float> gaussian_taps(float sigma) {
    const int r = std::max(1, static_cast<int>(std::ceil(3.f * sigma)));
    std::vector<float> g(2 * r + 1);
    float s = 0.f;
    for (int i = -r; i <= r; ++i) s += (g[i + r] = std::exp(-0.5f * (i / sigma) * (i / sigma)));
    for (auto& v : g) v /= s;
    return g;
}

cudaError_t anomaly_maps(const float* d_sq, int B, int h, int w, int S, const float* taps, int radius, float* tmp,
                         float* out, cudaStream_t stream) {
    const long long n = static_cast<long long>(B) * S * S;
    upsample_sqrt_kernel<<<grid_for(n), 256, 0, stream>>>(d_sq, B, h, w, S, out);
    if (radius > 0) {
        blur_kernel<true><<<grid_for(n), 256, 0, stream>>>(out, B, S, taps, radius, tmp);
        blur_kernel<false><<<grid_for(n), 256, 0, stream>>>(tmp, B, S, taps, radius, out);
    }
    return cudaGetLastError();
}

}  // namespace semdefect
