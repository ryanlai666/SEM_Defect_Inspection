// Straightforward CPU reference implementations used by the GoogleTest suite.
#pragma once

#include <cstdint>
#include <vector>

#include "semdefect/kernels.h"

namespace semdefect::ref {

// Same semantics as the CUDA kernel: crop -> area (adaptive avg pool) | bilinear -> scale -> normalise.
template <typename T>
std::vector<float> preprocess(const std::vector<T>& src, int B, int H, int W, const PreprocessParams& p);

// Brute-force exact k-NN (squared L2) in double precision; ascending.
void knn(const std::vector<float>& q, const std::vector<float>& x, int M, int N, int D, int K,
         std::vector<float>& out_d, std::vector<int64_t>& out_i);

// sqrt -> bilinear upsample (align_corners=False) -> separable Gaussian blur (reflect), double precision.
std::vector<float> anomaly_maps(const std::vector<float>& d_sq, int B, int h, int w, int S, float sigma);

}  // namespace semdefect::ref
