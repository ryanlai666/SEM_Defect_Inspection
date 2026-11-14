#include "semdefect/cpu_reference.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>

namespace semdefect::ref {

template <typename T>
std::vector<float> preprocess(const std::vector<T>& src, int B, int H, int W, const PreprocessParams& p) {
    std::vector<float> out(static_cast<size_t>(B) * 3 * p.oh * p.ow);
    const size_t plane = static_cast<size_t>(p.oh) * p.ow;
    auto at = [&](int b, int y, int x) {
        return static_cast<double>(src[(static_cast<size_t>(b) * H + p.y0 + y) * W + p.x0 + x]);
    };
    for (int b = 0; b < B; ++b)
        for (int oy = 0; oy < p.oh; ++oy)
            for (int ox = 0; ox < p.ow; ++ox) {
                double v = 0.0;
                if (p.mode == ResizeMode::Area) {
                    const int sy = oy * p.ch / p.oh, ey = ((oy + 1) * p.ch + p.oh - 1) / p.oh;
                    const int sx = ox * p.cw / p.ow, ex = ((ox + 1) * p.cw + p.ow - 1) / p.ow;
                    for (int y = sy; y < ey; ++y)
                        for (int x = sx; x < ex; ++x) v += at(b, y, x);
                    v /= static_cast<double>((ey - sy) * (ex - sx));
                } else {
                    // Source coordinates in float32 exactly like PyTorch's area_pixel_compute_source_index
                    // (scale = (float)in/out); only the interpolation itself is done in double.
                    const float scy = static_cast<float>(p.ch) / static_cast<float>(p.oh);
                    const float scx = static_cast<float>(p.cw) / static_cast<float>(p.ow);
                    // fmaf: nvcc (and PyTorch's CUDA build) contract a*b-c into one FMA by default.
                    const float fy = std::max(0.f, std::fmaf(static_cast<float>(oy) + 0.5f, scy, -0.5f));
                    const float fx = std::max(0.f, std::fmaf(static_cast<float>(ox) + 0.5f, scx, -0.5f));
                    const int iy = static_cast<int>(fy), ix = static_cast<int>(fx);
                    const int py = iy < p.ch - 1 ? 1 : 0, px = ix < p.cw - 1 ? 1 : 0;
                    const double ly = static_cast<double>(fy - static_cast<float>(iy));
                    const double lx = static_cast<double>(fx - static_cast<float>(ix));
                    v = (1 - ly) * ((1 - lx) * at(b, iy, ix) + lx * at(b, iy, ix + px)) +
                        ly * ((1 - lx) * at(b, iy + py, ix) + lx * at(b, iy + py, ix + px));
                }
                v *= p.in_scale;
                for (int c = 0; c < 3; ++c)
                    out[(static_cast<size_t>(b) * 3 + c) * plane + static_cast<size_t>(oy) * p.ow + ox] =
                        static_cast<float>((v - p.mean[c]) * p.inv_std[c]);
            }
    return out;
}

template std::vector<float> preprocess<uint8_t>(const std::vector<uint8_t>&, int, int, int, const PreprocessParams&);
template std::vector<float> preprocess<uint16_t>(const std::vector<uint16_t>&, int, int, int,
                                                 const PreprocessParams&);

void knn(const std::vector<float>& q, const std::vector<float>& x, int M, int N, int D, int K,
         std::vector<float>& out_d, std::vector<int64_t>& out_i) {
    out_d.assign(static_cast<size_t>(M) * K, 0.f);
    out_i.assign(static_cast<size_t>(M) * K, -1);
    std::vector<std::pair<double, int>> d(N);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double s = 0.0;
            for (int k = 0; k < D; ++k) {
                const double diff = static_cast<double>(q[static_cast<size_t>(m) * D + k]) - x[static_cast<size_t>(n) * D + k];
                s += diff * diff;
            }
            d[n] = {s, n};
        }
        const int kk = std::min(K, N);
        std::partial_sort(d.begin(), d.begin() + kk, d.end());
        for (int j = 0; j < kk; ++j) {
            out_d[static_cast<size_t>(m) * K + j] = static_cast<float>(d[j].first);
            out_i[static_cast<size_t>(m) * K + j] = d[j].second;
        }
    }
}

std::vector<float> anomaly_maps(const std::vector<float>& d_sq, int B, int h, int w, int S, float sigma) {
    std::vector<double> up(static_cast<size_t>(B) * S * S), tmp(up.size());
    const float sy = static_cast<float>(h) / S, sx = static_cast<float>(w) / S;
    for (int b = 0; b < B; ++b)
        for (int oy = 0; oy < S; ++oy)
            for (int ox = 0; ox < S; ++ox) {
                const float fy = std::max(0.f, std::fmaf(oy + 0.5f, sy, -0.5f));
                const float fx = std::max(0.f, std::fmaf(ox + 0.5f, sx, -0.5f));
                const int iy = static_cast<int>(fy), ix = static_cast<int>(fx);
                const int py = iy < h - 1 ? 1 : 0, px = ix < w - 1 ? 1 : 0;
                const double ly = fy - iy, lx = fx - ix;
                auto v = [&](int y, int x) { return std::sqrt(std::max(0.0, double(d_sq[(size_t(b) * h + y) * w + x]))); };
                up[(size_t(b) * S + oy) * S + ox] = (1 - ly) * ((1 - lx) * v(iy, ix) + lx * v(iy, ix + px)) +
                                                    ly * ((1 - lx) * v(iy + py, ix) + lx * v(iy + py, ix + px));
            }
    const int r = std::max(1, static_cast<int>(std::ceil(3.0 * sigma)));
    std::vector<double> g(2 * r + 1);
    double gs = 0;
    for (int i = -r; i <= r; ++i) gs += (g[i + r] = std::exp(-0.5 * (i / double(sigma)) * (i / double(sigma))));
    for (auto& x : g) x /= gs;
    auto refl = [](int i, int n) {
        while (i < 0 || i >= n) i = i < 0 ? -i : 2 * (n - 1) - i;
        return i;
    };
    for (int b = 0; b < B; ++b)
        for (int y = 0; y < S; ++y)
            for (int x = 0; x < S; ++x) {
                double a = 0;
                for (int k = -r; k <= r; ++k) a += g[k + r] * up[(size_t(b) * S + y) * S + refl(x + k, S)];
                tmp[(size_t(b) * S + y) * S + x] = a;
            }
    std::vector<float> out(up.size());
    for (int b = 0; b < B; ++b)
        for (int y = 0; y < S; ++y)
            for (int x = 0; x < S; ++x) {
                double a = 0;
                for (int k = -r; k <= r; ++k) a += g[k + r] * tmp[(size_t(b) * S + refl(y + k, S)) * S + x];
                out[(size_t(b) * S + y) * S + x] = static_cast<float>(a);
            }
    return out;
}

}  // namespace semdefect::ref
