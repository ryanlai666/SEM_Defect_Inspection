// pybind11 module `semdefect_cuda`: thin, zero-copy bindings taking raw device pointers (torch
// tensor.data_ptr()) and a CUDA stream handle (torch.cuda.current_stream().cuda_stream).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstdint>

#include "semdefect/kernels.h"

namespace py = pybind11;
using namespace semdefect;

namespace {
cudaStream_t as_stream(std::uintptr_t s) { return reinterpret_cast<cudaStream_t>(s); }
void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
}  // namespace

PYBIND11_MODULE(semdefect_cuda, m) {
    m.doc() = "SEM defect inspection CUDA kernels (nvcc build): fused preprocessing and kNN top-k";

    m.def(
        "row_sqnorm",
        [](std::uintptr_t x, int N, int D, std::uintptr_t out, std::uintptr_t stream) {
            check(row_sqnorm(reinterpret_cast<const float*>(x), N, D, reinterpret_cast<float*>(out),
                             as_stream(stream)));
        },
        py::arg("x"), py::arg("N"), py::arg("D"), py::arg("out"), py::arg("stream") = 0);

    m.def(
        "knn_topk",
        [](std::uintptr_t q, std::uintptr_t x, std::uintptr_t xnorm, std::uintptr_t qnorm, int M, int N, int D, int K,
           int S, int nps, std::uintptr_t part_d, std::uintptr_t part_i, std::uintptr_t out_d, std::uintptr_t out_i,
           std::uintptr_t stream) {
            py::gil_scoped_release release;
            check(knn_topk(reinterpret_cast<const float*>(q), reinterpret_cast<const float*>(x),
                           reinterpret_cast<const float*>(xnorm), reinterpret_cast<const float*>(qnorm), M, N, D, K,
                           S, nps, reinterpret_cast<float*>(part_d), reinterpret_cast<int*>(part_i),
                           reinterpret_cast<float*>(out_d), reinterpret_cast<int64_t*>(out_i), as_stream(stream)));
        },
        "Fused squared-L2 + top-k (K in 1,2,4,8,16,32)");

    m.def(
        "row_sqnorm_f16",
        [](std::uintptr_t x, int N, int D, std::uintptr_t out, std::uintptr_t stream) {
            check(row_sqnorm_f16(reinterpret_cast<const void*>(x), N, D, reinterpret_cast<float*>(out),
                                 as_stream(stream)));
        },
        py::arg("x"), py::arg("N"), py::arg("D"), py::arg("out"), py::arg("stream") = 0);

    m.def(
        "knn_top1_f16",
        [](std::uintptr_t q, std::uintptr_t x, std::uintptr_t xnorm, std::uintptr_t qnorm, int M, int N, int D, int S,
           int nps, std::uintptr_t part_d, std::uintptr_t part_i, std::uintptr_t out_d, std::uintptr_t out_i,
           std::uintptr_t stream) {
            py::gil_scoped_release release;
            check(knn_top1_f16(reinterpret_cast<const void*>(q), reinterpret_cast<const void*>(x),
                               reinterpret_cast<const float*>(xnorm), reinterpret_cast<const float*>(qnorm), M, N, D, S,
                               nps, reinterpret_cast<float*>(part_d), reinterpret_cast<int*>(part_i),
                               reinterpret_cast<float*>(out_d), reinterpret_cast<int64_t*>(out_i), as_stream(stream)));
        },
        "Tensor-core (WMMA) nearest neighbour: FP16 inputs, FP32 accumulation, D % 32 == 0");

    m.def(
        "preprocess",
        [](std::uintptr_t src, bool is16, std::uintptr_t dst, int B, int H, int W, int y0, int x0, int ch, int cw,
           int oh, int ow, int mode, float in_scale, std::array<float, 3> mean, std::array<float, 3> inv_std,
           std::uintptr_t stream) {
            PreprocessParams p;
            p.y0 = y0;
            p.x0 = x0;
            p.ch = ch;
            p.cw = cw;
            p.oh = oh;
            p.ow = ow;
            p.mode = static_cast<ResizeMode>(mode);
            p.in_scale = in_scale;
            for (int c = 0; c < 3; ++c) {
                p.mean[c] = mean[c];
                p.inv_std[c] = inv_std[c];
            }
            if (is16)
                check(preprocess_u16(reinterpret_cast<const uint16_t*>(src), reinterpret_cast<float*>(dst), B, H, W, p,
                                     as_stream(stream)));
            else
                check(preprocess_u8(reinterpret_cast<const uint8_t*>(src), reinterpret_cast<float*>(dst), B, H, W, p,
                                    as_stream(stream)));
        },
        "Fused crop/resize/scale/normalise of raw grey SEM images to (B,3,oh,ow) float32");
}
