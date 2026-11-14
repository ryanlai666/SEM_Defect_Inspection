// GoogleTest: CUDA kernels vs CPU reference implementations.
#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <tuple>
#include <vector>

#include "semdefect/cpu_reference.h"
#include "semdefect/kernels.h"

#include <cuda_fp16.h>

using namespace semdefect;

namespace {

template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr;
    size_t n = 0;
    explicit DeviceBuffer(size_t count) : n(count) { SEMDEFECT_CUDA_CHECK(cudaMalloc(&ptr, sizeof(T) * count)); }
    explicit DeviceBuffer(const std::vector<T>& h) : DeviceBuffer(h.size()) {
        SEMDEFECT_CUDA_CHECK(cudaMemcpy(ptr, h.data(), sizeof(T) * n, cudaMemcpyHostToDevice));
    }
    ~DeviceBuffer() { cudaFree(ptr); }
    std::vector<T> host() const {
        std::vector<T> h(n);
        SEMDEFECT_CUDA_CHECK(cudaMemcpy(h.data(), ptr, sizeof(T) * n, cudaMemcpyDeviceToHost));
        return h;
    }
};

bool has_gpu() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

}  // namespace

// ------------------------------------------------------------------------------- preprocessing
class PreprocessTest : public ::testing::TestWithParam<std::tuple<ResizeMode, int, bool>> {};

TEST_P(PreprocessTest, MatchesCpuReference) {
    if (!has_gpu()) GTEST_SKIP() << "no CUDA device";
    const auto [mode, out, sixteen_bit] = GetParam();
    const int B = 2, H = 480, W = 480;
    PreprocessParams p;
    p.y0 = 1;
    p.x0 = 0;
    p.ch = 477;
    p.cw = 477;  // Carinthia-S border crop
    p.oh = p.ow = out;
    p.mode = mode;
    std::mt19937 rng(42);
    std::vector<float> got, want;
    if (sixteen_bit) {
        std::uniform_int_distribution<int> dist(0, 65535);
        std::vector<uint16_t> img(static_cast<size_t>(B) * H * W);
        for (auto& v : img) v = static_cast<uint16_t>(dist(rng));
        p.in_scale = 1.f / 65535.f;
        DeviceBuffer<uint16_t> d_in(img);
        DeviceBuffer<float> d_out(static_cast<size_t>(B) * 3 * out * out);
        SEMDEFECT_CUDA_CHECK(preprocess_u16(d_in.ptr, d_out.ptr, B, H, W, p, nullptr));
        got = d_out.host();
        want = ref::preprocess(img, B, H, W, p);
    } else {
        std::uniform_int_distribution<int> dist(0, 255);
        std::vector<uint8_t> img(static_cast<size_t>(B) * H * W);
        for (auto& v : img) v = static_cast<uint8_t>(dist(rng));
        DeviceBuffer<uint8_t> d_in(img);
        DeviceBuffer<float> d_out(static_cast<size_t>(B) * 3 * out * out);
        SEMDEFECT_CUDA_CHECK(preprocess_u8(d_in.ptr, d_out.ptr, B, H, W, p, nullptr));
        got = d_out.host();
        want = ref::preprocess(img, B, H, W, p);
    }
    ASSERT_EQ(got.size(), want.size());
    double max_err = 0;
    for (size_t i = 0; i < got.size(); ++i) max_err = std::max(max_err, std::abs(double(got[i]) - want[i]));
    EXPECT_LT(max_err, 2e-5) << "max abs error " << max_err;
}

INSTANTIATE_TEST_SUITE_P(Modes, PreprocessTest,
                         ::testing::Combine(::testing::Values(ResizeMode::Area, ResizeMode::Bilinear),
                                            ::testing::Values(224, 256, 448), ::testing::Bool()));

// ------------------------------------------------------------------------------------- k-NN
struct KnnCase {
    int M, N, D, K, S;
};

class KnnTest : public ::testing::TestWithParam<KnnCase> {};

TEST_P(KnnTest, MatchesBruteForce) {
    if (!has_gpu()) GTEST_SKIP() << "no CUDA device";
    const auto c = GetParam();
    std::mt19937 rng(7 + c.M + c.N);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> q(static_cast<size_t>(c.M) * c.D), x(static_cast<size_t>(c.N) * c.D);
    for (auto& v : q) v = nd(rng);
    for (auto& v : x) v = nd(rng);
    DeviceBuffer<float> dq(q), dx(x), xn(c.N), qn(c.M);
    SEMDEFECT_CUDA_CHECK(row_sqnorm(dx.ptr, c.N, c.D, xn.ptr, nullptr));
    SEMDEFECT_CUDA_CHECK(row_sqnorm(dq.ptr, c.M, c.D, qn.ptr, nullptr));
    const int nps = knn_n_per_split(c.N, c.S);
    const int S = knn_effective_splits(c.N, nps);
    DeviceBuffer<float> pd(static_cast<size_t>(c.M) * S * c.K), od(static_cast<size_t>(c.M) * c.K);
    DeviceBuffer<int> pi(static_cast<size_t>(c.M) * S * c.K);
    DeviceBuffer<int64_t> oi(static_cast<size_t>(c.M) * c.K);
    SEMDEFECT_CUDA_CHECK(
        knn_topk(dq.ptr, dx.ptr, xn.ptr, qn.ptr, c.M, c.N, c.D, c.K, S, nps, pd.ptr, pi.ptr, od.ptr, oi.ptr, nullptr));
    SEMDEFECT_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> rd;
    std::vector<int64_t> ri;
    ref::knn(q, x, c.M, c.N, c.D, c.K, rd, ri);
    const auto gd = od.host();
    const auto gi = oi.host();
    int index_mismatch = 0;
    for (size_t i = 0; i < gd.size(); ++i) {
        EXPECT_NEAR(gd[i], rd[i], 1e-3f * std::max(1.f, rd[i])) << "entry " << i;
        if (gi[i] != ri[i]) ++index_mismatch;  // only legitimate for near-ties
    }
    EXPECT_LE(index_mismatch, static_cast<int>(gd.size() / 200 + 1));
}

INSTANTIATE_TEST_SUITE_P(Shapes, KnnTest,
                         ::testing::Values(KnnCase{1, 1, 1, 1, 1},        // degenerate
                                           KnnCase{37, 129, 7, 1, 1},     // nothing tile-aligned
                                           KnnCase{64, 64, 16, 4, 1},     // exactly one tile
                                           KnnCase{100, 1000, 64, 8, 3},  // split bank
                                           KnnCase{256, 3000, 128, 16, 7},
                                           KnnCase{130, 500, 384, 32, 2},  // K = 32 (max)
                                           KnnCase{1024, 2048, 1024, 1, 16}));  // PatchCore-like

TEST(KnnIndex, SearchMatchesExplicitCall) {
    if (!has_gpu()) GTEST_SKIP() << "no CUDA device";
    const int N = 777, D = 33, M = 99, K = 4;
    std::mt19937 rng(3);
    std::normal_distribution<float> nd;
    std::vector<float> x(static_cast<size_t>(N) * D), q(static_cast<size_t>(M) * D);
    for (auto& v : x) v = nd(rng);
    for (auto& v : q) v = nd(rng);
    KnnIndex index(x.data(), N, D);
    DeviceBuffer<float> dq(q), od(static_cast<size_t>(M) * K);
    DeviceBuffer<int64_t> oi(static_cast<size_t>(M) * K);
    index.search(dq.ptr, M, K, od.ptr, oi.ptr, nullptr);
    SEMDEFECT_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> rd;
    std::vector<int64_t> ri;
    ref::knn(q, x, M, N, D, K, rd, ri);
    const auto gd = od.host();
    for (size_t i = 0; i < gd.size(); ++i) EXPECT_NEAR(gd[i], rd[i], 1e-3f * std::max(1.f, rd[i]));
}

TEST(KnnIndex, RejectsUnsupportedK) {
    EXPECT_THROW(knn_round_k(33), std::invalid_argument);
    EXPECT_EQ(knn_round_k(3), 4);
    EXPECT_EQ(knn_round_k(17), 32);
}

// ------------------------------------------------------------------- tensor-core k = 1 (FP16 inputs)
class KnnWmmaTest : public ::testing::TestWithParam<KnnCase> {};

TEST_P(KnnWmmaTest, MatchesBruteForceOnFp16RoundedInputs) {
    if (!has_gpu()) GTEST_SKIP() << "no CUDA device";
    const auto c = GetParam();
    std::mt19937 rng(11 + c.M + c.N);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> q(static_cast<size_t>(c.M) * c.D), x(static_cast<size_t>(c.N) * c.D);
    std::vector<__half> qh(q.size()), xh(x.size());
    for (size_t i = 0; i < q.size(); ++i) {  // reference sees exactly the FP16-rounded values
        qh[i] = __float2half(nd(rng));
        q[i] = __half2float(qh[i]);
    }
    for (size_t i = 0; i < x.size(); ++i) {
        xh[i] = __float2half(nd(rng));
        x[i] = __half2float(xh[i]);
    }
    DeviceBuffer<__half> dq(qh), dx(xh);
    DeviceBuffer<float> xn(c.N), qn(c.M);
    SEMDEFECT_CUDA_CHECK(row_sqnorm_f16(dx.ptr, c.N, c.D, xn.ptr, nullptr));
    SEMDEFECT_CUDA_CHECK(row_sqnorm_f16(dq.ptr, c.M, c.D, qn.ptr, nullptr));
    const int nps = knn_n_per_split(c.N, c.S);
    const int S = knn_effective_splits(c.N, nps);
    DeviceBuffer<float> pd(static_cast<size_t>(c.M) * S), od(c.M);
    DeviceBuffer<int> pi(static_cast<size_t>(c.M) * S);
    DeviceBuffer<int64_t> oi(c.M);
    SEMDEFECT_CUDA_CHECK(
        knn_top1_f16(dq.ptr, dx.ptr, xn.ptr, qn.ptr, c.M, c.N, c.D, S, nps, pd.ptr, pi.ptr, od.ptr, oi.ptr, nullptr));
    SEMDEFECT_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> rd;
    std::vector<int64_t> ri;
    ref::knn(q, x, c.M, c.N, c.D, 1, rd, ri);
    const auto gd = od.host();
    const auto gi = oi.host();
    int mismatch = 0;
    for (int m = 0; m < c.M; ++m) {
        EXPECT_NEAR(gd[m], rd[m], 2e-3f * std::max(1.f, rd[m])) << "query " << m;
        if (gi[m] != ri[m]) ++mismatch;
    }
    EXPECT_LE(mismatch, c.M / 100 + 1);
}

INSTANTIATE_TEST_SUITE_P(Shapes, KnnWmmaTest,
                         ::testing::Values(KnnCase{1, 1, 32, 1, 1}, KnnCase{37, 129, 32, 1, 1},
                                           KnnCase{100, 1000, 384, 1, 3}, KnnCase{1024, 2048, 1024, 1, 16},
                                           KnnCase{65, 4097, 96, 1, 5}));

TEST(KnnWmma, RejectsDimensionNotMultipleOf32) {
    EXPECT_EQ(knn_top1_f16(nullptr, nullptr, nullptr, nullptr, 4, 4, 40, 1, 64, nullptr, nullptr, nullptr, nullptr,
                           nullptr),
              cudaErrorInvalidValue);
}

// ------------------------------------------------------------------- anomaly-map post-processing
TEST(AnomalyMaps, MatchesCpuReference) {
    if (!has_gpu()) GTEST_SKIP() << "no CUDA device";
    const int B = 3, h = 32, w = 32, S = 224;
    const float sigma = 4.f;
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> ud(0.f, 9.f);
    std::vector<float> d(static_cast<size_t>(B) * h * w);
    for (auto& v : d) v = ud(rng);
    const auto taps = gaussian_taps(sigma);
    const int radius = static_cast<int>(taps.size() / 2);
    DeviceBuffer<float> dd(d), dt(taps), tmp(static_cast<size_t>(B) * S * S), out(static_cast<size_t>(B) * S * S);
    SEMDEFECT_CUDA_CHECK(anomaly_maps(dd.ptr, B, h, w, S, dt.ptr, radius, tmp.ptr, out.ptr, nullptr));
    SEMDEFECT_CUDA_CHECK(cudaDeviceSynchronize());
    const auto got = out.host();
    const auto want = ref::anomaly_maps(d, B, h, w, S, sigma);
    double max_err = 0;
    for (size_t i = 0; i < got.size(); ++i) max_err = std::max(max_err, std::abs(double(got[i]) - want[i]));
    EXPECT_LT(max_err, 1e-5) << "max abs error " << max_err;
}
