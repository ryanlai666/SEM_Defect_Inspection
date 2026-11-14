// trt_runner: end-to-end PatchCore inference in C++ on raw 8-bit SEM images.
//
//   pinned host uint8 batch --H2D--> fused preprocess kernel (crop/area-resize/normalise, 3ch)
//   --> TensorRT engine (WRN-50 embedder, FP32/FP16/INT8) --> fused tiled L2 + top-1 kNN kernel
//   --> per-image max --> D2H scores
//
// Several CUDA streams (each with its own IExecutionContext and buffers) keep the GPU busy while
// the host stages the next batch.  TensorRT is loaded at run time from the pip wheel's
// nvinfer_10.dll / libnvinfer.so.10: only `createInferRuntime_INTERNAL` is resolved, everything else
// is virtual dispatch through the public headers, so no import library / developer login is needed.
//
// Usage:
//   trt_runner --engine E --bank bank.bin --images raw_u8.bin [--batch 8] [--streams 2] [--iters 200]
//              [--trt-dir DIR] [--size 256] [--knn fp32|fp16] [--scores-out scores.bin] [--json out.json]
//              [--maps-out maps.bin --maps-n 16 --map-size 224 --sigma 4]   (localisation maps, float32)
//              [--cuda-graph]   (replay the per-batch GPU sequence as one CUDA graph)
#include <NvInferRuntime.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "semdefect/kernels.h"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

using namespace semdefect;

namespace {

struct Logger final : nvinfer1::ILogger {
    void log(Severity s, nvinfer1::AsciiChar const* msg) noexcept override {
        if (s <= Severity::kWARNING) std::cerr << "[TensorRT] " << msg << "\n";
    }
};

using CreateRuntimeFn = void* (*)(void*, int32_t);

CreateRuntimeFn load_tensorrt(const std::string& dir) {
#ifdef _WIN32
    std::string path = dir.empty() ? "nvinfer_10.dll" : dir + "\\nvinfer_10.dll";
    if (!dir.empty()) SetDllDirectoryA(dir.c_str());  // dependent DLLs live next to nvinfer
    HMODULE h = LoadLibraryA(path.c_str());
    if (!h) throw std::runtime_error("cannot load " + path + " (error " + std::to_string(GetLastError()) + ")");
    auto fn = reinterpret_cast<CreateRuntimeFn>(GetProcAddress(h, "createInferRuntime_INTERNAL"));
#else
    std::string path = dir.empty() ? "libnvinfer.so.10" : dir + "/libnvinfer.so.10";
    void* h = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!h) throw std::runtime_error(std::string("cannot load ") + path + ": " + dlerror());
    auto fn = reinterpret_cast<CreateRuntimeFn>(dlsym(h, "createInferRuntime_INTERNAL"));
#endif
    if (!fn) throw std::runtime_error("createInferRuntime_INTERNAL not found");
    return fn;
}

template <typename T>
std::vector<T> read_blob(const std::string& path, std::vector<int32_t>& shape, int ndim) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    shape.resize(ndim);
    f.read(reinterpret_cast<char*>(shape.data()), sizeof(int32_t) * ndim);
    size_t n = 1;
    for (int d : shape) n *= static_cast<size_t>(d);
    std::vector<T> v(n);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(sizeof(T) * n));
    if (!f) throw std::runtime_error("short read " + path);
    return v;
}

struct Args {
    std::string engine, bank, images, trt_dir, scores_out, json, knn = "fp32", maps_out;
    int batch = 8, streams = 2, iters = 200, size = 256, warmup = 10, maps_n = 16, map_size = 224;
    float sigma = 4.f;
    bool cuda_graph = false;
};

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + k);
            return argv[++i];
        };
        if (k == "--engine") a.engine = next();
        else if (k == "--bank") a.bank = next();
        else if (k == "--images") a.images = next();
        else if (k == "--trt-dir") a.trt_dir = next();
        else if (k == "--scores-out") a.scores_out = next();
        else if (k == "--json") a.json = next();
        else if (k == "--batch") a.batch = std::stoi(next());
        else if (k == "--streams") a.streams = std::stoi(next());
        else if (k == "--iters") a.iters = std::stoi(next());
        else if (k == "--size") a.size = std::stoi(next());
        else if (k == "--warmup") a.warmup = std::stoi(next());
        else if (k == "--knn") a.knn = next();
        else if (k == "--maps-out") a.maps_out = next();  // float32 (maps_n, map_size, map_size) anomaly maps
        else if (k == "--maps-n") a.maps_n = std::stoi(next());
        else if (k == "--map-size") a.map_size = std::stoi(next());
        else if (k == "--sigma") a.sigma = std::stof(next());
        else if (k == "--cuda-graph") a.cuda_graph = true;  // replay the per-batch sequence as one CUDA graph  // fp32 (SIMT tiled) | fp16 (tensor-core WMMA, k = 1)
        else throw std::runtime_error("unknown arg " + k);
    }
    if (a.engine.empty() || a.bank.empty() || a.images.empty()) throw std::runtime_error("--engine, --bank, --images required");
    return a;
}

double pct(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    const double idx = p / 100.0 * (v.size() - 1);
    const size_t lo = static_cast<size_t>(idx), hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (v[hi] - v[lo]) * (idx - lo);
}

struct Slot {  // per-stream resources
    cudaStream_t stream{};
    cudaEvent_t start{}, done{};
    nvinfer1::IExecutionContext* ctx = nullptr;
    uint8_t *h_raw = nullptr, *d_raw = nullptr;
    void* d_emb16 = nullptr;
    float *d_in = nullptr, *d_emb = nullptr, *d_qn = nullptr, *d_dist = nullptr, *d_part = nullptr, *d_score = nullptr,
          *h_score = nullptr;
    int* d_parti = nullptr;
    int64_t* d_idx = nullptr;
    int first_image = -1, n = 0;
    cudaGraphExec_t graph = nullptr;  // optional CUDA graph of the whole per-batch GPU sequence
};

}  // namespace

int main(int argc, char** argv) try {
    const Args a = parse(argc, argv);
    // ------------------------------------------------------------------ inputs
    std::vector<int32_t> bshape, ishape;
    auto bank = read_blob<float>(a.bank, bshape, 2);  // (N, D)
    auto imgs = read_blob<uint8_t>(a.images, ishape, 3);  // (n_img, H, W)
    const int N = bshape[0], D = bshape[1], n_img = ishape[0], H = ishape[1], W = ishape[2];
    const int B = a.batch, P = (a.size / 8) * (a.size / 8);
    size_t free0 = 0, total = 0;
    SEMDEFECT_CUDA_CHECK(cudaMemGetInfo(&free0, &total));

    // ------------------------------------------------------------------ TensorRT
    Logger logger;
    auto create = load_tensorrt(a.trt_dir);
    std::unique_ptr<nvinfer1::IRuntime> runtime(static_cast<nvinfer1::IRuntime*>(create(&logger, NV_TENSORRT_VERSION)));
    if (!runtime) throw std::runtime_error("createInferRuntime failed (TensorRT version mismatch?)");
    std::ifstream ef(a.engine, std::ios::binary);
    std::vector<char> blob((std::istreambuf_iterator<char>(ef)), std::istreambuf_iterator<char>());
    std::unique_ptr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(blob.data(), blob.size()));
    if (!engine) throw std::runtime_error("deserializeCudaEngine failed");
    std::string in_name, out_name;
    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
        const char* nm = engine->getIOTensorName(i);
        (engine->getTensorIOMode(nm) == nvinfer1::TensorIOMode::kINPUT ? in_name : out_name) = nm;
    }

    // ------------------------------------------------------------------ memory bank (shared, read-only)
    const bool fp16_knn = a.knn == "fp16";
    if (fp16_knn && D % 32 != 0) throw std::runtime_error("--knn fp16 needs D % 32 == 0");
    float *d_bank = nullptr, *d_bnorm = nullptr;
    void* d_bank16 = nullptr;
    SEMDEFECT_CUDA_CHECK(cudaMalloc(&d_bank, sizeof(float) * bank.size()));
    SEMDEFECT_CUDA_CHECK(cudaMalloc(&d_bnorm, sizeof(float) * N));
    SEMDEFECT_CUDA_CHECK(cudaMemcpy(d_bank, bank.data(), sizeof(float) * bank.size(), cudaMemcpyHostToDevice));
    if (fp16_knn) {  // FP16 bank (half the bytes) + norms consistent with the FP16 values
        SEMDEFECT_CUDA_CHECK(cudaMalloc(&d_bank16, 2 * bank.size()));
        SEMDEFECT_CUDA_CHECK(f32_to_f16(d_bank, d_bank16, static_cast<long long>(bank.size()), nullptr));
        SEMDEFECT_CUDA_CHECK(row_sqnorm_f16(d_bank16, N, D, d_bnorm, nullptr));
        SEMDEFECT_CUDA_CHECK(cudaFree(d_bank));
        d_bank = nullptr;
    } else {
        SEMDEFECT_CUDA_CHECK(row_sqnorm(d_bank, N, D, d_bnorm, nullptr));
    }
    int dev = 0, sms = 1;
    SEMDEFECT_CUDA_CHECK(cudaGetDevice(&dev));
    SEMDEFECT_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    const int M = B * P;
    const int S = knn_effective_splits(N, knn_n_per_split(N, knn_num_splits(M, N, sms)));
    const int nps = knn_n_per_split(N, knn_num_splits(M, N, sms));

    PreprocessParams pp;
    pp.y0 = 1; pp.x0 = 0; pp.ch = 477; pp.cw = 477; pp.oh = pp.ow = a.size;  // Carinthia-S border crop

    std::vector<Slot> slots(a.streams);
    for (auto& s : slots) {
        SEMDEFECT_CUDA_CHECK(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking));
        SEMDEFECT_CUDA_CHECK(cudaEventCreate(&s.start));
        SEMDEFECT_CUDA_CHECK(cudaEventCreate(&s.done));
        SEMDEFECT_CUDA_CHECK(cudaHostAlloc(&s.h_raw, static_cast<size_t>(B) * H * W, cudaHostAllocDefault));
        SEMDEFECT_CUDA_CHECK(cudaHostAlloc(&s.h_score, sizeof(float) * B, cudaHostAllocDefault));
        SEMDEFECT_CUDA_CHECK(cudaMalloc(&s.d_raw, static_cast<size_t>(B) * H * W));
        SEMDEFECT_CUDA_CHECK(cudaMalloc(&s.d_in, sizeof(float) * B * 3 * a.size * a.size));
        SEMDEFECT_CUDA_CHECK(cudaMalloc(&s.d_emb, sizeof(float) * static_cast<size_t>(M) * D));
        if (fp16_knn) SEMDEFECT_CUDA_CHECK(cudaMalloc(&s.d_emb16, 2 * static_cast<size_t>(M) * D));
        SEMDEFECT_CUDA_CHECK(cudaMalloc(&s.d_qn, sizeof(float) * M));
        SEMDEFECT_CUDA_CHECK(cudaMalloc(&s.d_dist, sizeof(float) * M));
        SEMDEFECT_CUDA_CHECK(cudaMalloc(&s.d_idx, sizeof(int64_t) * M));
        SEMDEFECT_CUDA_CHECK(cudaMalloc(&s.d_part, sizeof(float) * static_cast<size_t>(M) * S));
        SEMDEFECT_CUDA_CHECK(cudaMalloc(&s.d_parti, sizeof(int) * static_cast<size_t>(M) * S));
        SEMDEFECT_CUDA_CHECK(cudaMalloc(&s.d_score, sizeof(float) * B));
        s.ctx = engine->createExecutionContext();
        if (!s.ctx) throw std::runtime_error("createExecutionContext failed");
        s.ctx->setInputShape(in_name.c_str(), nvinfer1::Dims4{B, 3, a.size, a.size});
        s.ctx->setTensorAddress(in_name.c_str(), s.d_in);
        s.ctx->setTensorAddress(out_name.c_str(), s.d_emb);
    }

    // GPU work of one batch (no host staging, no events): issued directly or captured into a CUDA graph.
    auto enqueue_gpu = [&](Slot& s) {
        SEMDEFECT_CUDA_CHECK(cudaMemcpyAsync(s.d_raw, s.h_raw, static_cast<size_t>(B) * H * W, cudaMemcpyHostToDevice, s.stream));
        SEMDEFECT_CUDA_CHECK(preprocess_u8(s.d_raw, s.d_in, B, H, W, pp, s.stream));
        if (!s.ctx->enqueueV3(s.stream)) throw std::runtime_error("enqueueV3 failed");
        if (fp16_knn) {
            SEMDEFECT_CUDA_CHECK(f32_to_f16(s.d_emb, s.d_emb16, static_cast<long long>(M) * D, s.stream));
            SEMDEFECT_CUDA_CHECK(row_sqnorm_f16(s.d_emb16, M, D, s.d_qn, s.stream));
            SEMDEFECT_CUDA_CHECK(knn_top1_f16(s.d_emb16, d_bank16, d_bnorm, s.d_qn, M, N, D, S, nps, s.d_part,
                                              s.d_parti, s.d_dist, s.d_idx, s.stream));
        } else {
            SEMDEFECT_CUDA_CHECK(row_sqnorm(s.d_emb, M, D, s.d_qn, s.stream));
            SEMDEFECT_CUDA_CHECK(knn_topk(s.d_emb, d_bank, d_bnorm, s.d_qn, M, N, D, 1, S, nps, s.d_part, s.d_parti,
                                          s.d_dist, s.d_idx, s.stream));
        }
        SEMDEFECT_CUDA_CHECK(image_max_sqrt(s.d_dist, B, P, s.d_score, s.stream));
        SEMDEFECT_CUDA_CHECK(cudaMemcpyAsync(s.h_score, s.d_score, sizeof(float) * B, cudaMemcpyDeviceToHost, s.stream));
    };
    auto submit = [&](Slot& s, int first) {
        s.first_image = first;
        s.n = B;
        for (int b = 0; b < B; ++b)  // stage the batch in pinned memory (wraps around the image set)
            std::memcpy(s.h_raw + static_cast<size_t>(b) * H * W,
                        imgs.data() + static_cast<size_t>((first + b) % n_img) * H * W, static_cast<size_t>(H) * W);
        SEMDEFECT_CUDA_CHECK(cudaEventRecord(s.start, s.stream));
        if (s.graph)
            SEMDEFECT_CUDA_CHECK(cudaGraphLaunch(s.graph, s.stream));
        else
            enqueue_gpu(s);
        SEMDEFECT_CUDA_CHECK(cudaEventRecord(s.done, s.stream));
    };

    // ------------------------------------------------------------------ pass 1: scores (+ maps) for every image
    const int grid_side = a.size / 8;
    const bool want_maps = !a.maps_out.empty() && a.maps_n > 0;
    const size_t map_px = static_cast<size_t>(a.map_size) * a.map_size;
    float *d_taps = nullptr, *d_map_tmp = nullptr, *d_maps = nullptr;
    std::vector<float> maps;
    int radius = 0;
    if (want_maps) {
        const auto taps = gaussian_taps(a.sigma);
        radius = static_cast<int>(taps.size() / 2);
        SEMDEFECT_CUDA_CHECK(cudaMalloc(&d_taps, sizeof(float) * taps.size()));
        SEMDEFECT_CUDA_CHECK(cudaMemcpy(d_taps, taps.data(), sizeof(float) * taps.size(), cudaMemcpyHostToDevice));
        SEMDEFECT_CUDA_CHECK(cudaMalloc(&d_map_tmp, sizeof(float) * B * map_px));
        SEMDEFECT_CUDA_CHECK(cudaMalloc(&d_maps, sizeof(float) * B * map_px));
        maps.resize(static_cast<size_t>(std::min(a.maps_n, n_img)) * map_px);
    }
    std::vector<float> scores(n_img, 0.f);
    for (int first = 0; first < n_img; first += B) {
        submit(slots[0], first);
        SEMDEFECT_CUDA_CHECK(cudaEventSynchronize(slots[0].done));
        for (int b = 0; b < B && first + b < n_img; ++b) scores[first + b] = slots[0].h_score[b];
        if (want_maps && first < a.maps_n) {  // localisation maps for the first maps_n images
            SEMDEFECT_CUDA_CHECK(anomaly_maps(slots[0].d_dist, B, grid_side, grid_side, a.map_size, d_taps, radius,
                                              d_map_tmp, d_maps, slots[0].stream));
            const int nb = std::min({B, a.maps_n - first, n_img - first});
            SEMDEFECT_CUDA_CHECK(cudaMemcpyAsync(maps.data() + static_cast<size_t>(first) * map_px, d_maps,
                                                 sizeof(float) * nb * map_px, cudaMemcpyDeviceToHost, slots[0].stream));
            SEMDEFECT_CUDA_CHECK(cudaStreamSynchronize(slots[0].stream));
        }
    }
    if (want_maps) {
        std::ofstream mo(a.maps_out, std::ios::binary);
        mo.write(reinterpret_cast<const char*>(maps.data()), static_cast<std::streamsize>(sizeof(float) * maps.size()));
        cudaFree(d_taps);
        cudaFree(d_map_tmp);
        cudaFree(d_maps);
    }
    if (!a.scores_out.empty()) {
        std::ofstream so(a.scores_out, std::ios::binary);
        so.write(reinterpret_cast<const char*>(scores.data()), static_cast<std::streamsize>(sizeof(float) * n_img));
    }
    size_t free1 = 0;
    SEMDEFECT_CUDA_CHECK(cudaMemGetInfo(&free1, &total));

    // ------------------------------------------------------------------ optional CUDA graphs (after pass 1, which
    // also served as TensorRT's required pre-capture enqueue); buffers are fixed per slot so a graph is reusable.
    if (a.cuda_graph) {
        for (auto& s : slots) {
            if (!s.ctx->enqueueV3(s.stream)) throw std::runtime_error("pre-capture enqueueV3 failed");
            SEMDEFECT_CUDA_CHECK(cudaStreamSynchronize(s.stream));
            cudaGraph_t g = nullptr;
            SEMDEFECT_CUDA_CHECK(cudaStreamBeginCapture(s.stream, cudaStreamCaptureModeThreadLocal));
            enqueue_gpu(s);
            SEMDEFECT_CUDA_CHECK(cudaStreamEndCapture(s.stream, &g));
            SEMDEFECT_CUDA_CHECK(cudaGraphInstantiate(&s.graph, g, 0));
            SEMDEFECT_CUDA_CHECK(cudaGraphDestroy(g));
        }
    }
    double graph_parity = 0.0;  // graph replay must reproduce the direct-launch scores exactly
    if (a.cuda_graph) {
        submit(slots[0], 0);
        SEMDEFECT_CUDA_CHECK(cudaEventSynchronize(slots[0].done));
        for (int b = 0; b < B && b < n_img; ++b)
            graph_parity = std::max(graph_parity, static_cast<double>(std::abs(slots[0].h_score[b] - scores[b])));
    }

    // ------------------------------------------------------------------ pass 2: latency / throughput
    std::vector<double> lat;
    int submitted = 0;
    for (int i = 0; i < a.warmup; ++i) submit(slots[i % a.streams], (i * B) % n_img);
    SEMDEFECT_CUDA_CHECK(cudaDeviceSynchronize());
    const auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < a.iters; ++it) {
        Slot& s = slots[it % a.streams];
        if (it >= a.streams) {  // reuse this slot: wait for its previous batch, record its latency
            SEMDEFECT_CUDA_CHECK(cudaEventSynchronize(s.done));
            float ms = 0.f;
            SEMDEFECT_CUDA_CHECK(cudaEventElapsedTime(&ms, s.start, s.done));
            lat.push_back(ms);
        }
        submit(s, (it * B) % n_img);
        ++submitted;
    }
    for (int k = 0; k < a.streams && k < a.iters; ++k) {
        Slot& s = slots[(a.iters - 1 - k) % a.streams];
        SEMDEFECT_CUDA_CHECK(cudaEventSynchronize(s.done));
        float ms = 0.f;
        SEMDEFECT_CUDA_CHECK(cudaEventElapsedTime(&ms, s.start, s.done));
        lat.push_back(ms);
    }
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double ips = static_cast<double>(submitted) * B / wall;

    std::string engine_json = a.engine;  // JSON-safe: Windows backslashes -> forward slashes
    std::replace(engine_json.begin(), engine_json.end(), '\\', '/');
    std::ostringstream js;
    js << "{\n  \"engine\": \"" << engine_json << "\", \"batch\": " << B << ", \"streams\": " << a.streams
       << ", \"iters\": " << a.iters << ", \"knn\": \"" << a.knn << "\", \"cuda_graph\": " << (a.cuda_graph ? "true" : "false") << ", \"graph_vs_direct_max_abs_score_diff\": " << graph_parity << ",\n  \"bank_N\": " << N << ", \"D\": " << D << ", \"patches_per_image\": " << P
       << ",\n  \"latency_ms_p50\": " << pct(lat, 50) << ", \"latency_ms_p99\": " << pct(lat, 99)
       << ",\n  \"throughput_img_per_s\": " << ips << ",\n  \"vram_used_by_runner_mib\": "
       << (static_cast<double>(free0) - static_cast<double>(free1)) / (1 << 20)
       << ",\n  \"engine_device_memory_mib\": " << static_cast<double>(engine->getDeviceMemorySizeV2()) / (1 << 20)
       << "\n}\n";
    std::cout << js.str();
    if (!a.json.empty()) std::ofstream(a.json) << js.str();

    for (auto& s : slots) {
        if (s.graph) cudaGraphExecDestroy(s.graph);
        delete s.ctx;
        cudaFree(s.d_emb16);
        cudaFree(s.d_raw); cudaFree(s.d_in); cudaFree(s.d_emb); cudaFree(s.d_qn); cudaFree(s.d_dist);
        cudaFree(s.d_idx); cudaFree(s.d_part); cudaFree(s.d_parti); cudaFree(s.d_score);
        cudaFreeHost(s.h_raw); cudaFreeHost(s.h_score);
        cudaEventDestroy(s.start); cudaEventDestroy(s.done); cudaStreamDestroy(s.stream);
    }
    cudaFree(d_bank);
    cudaFree(d_bank16);
    cudaFree(d_bnorm);
    return 0;
} catch (const std::exception& e) {
    std::cerr << "trt_runner error: " << e.what() << "\n";
    return 1;
}
