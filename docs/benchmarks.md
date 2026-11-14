# Benchmarks (measured on this machine; regenerate with `python scripts/report_bench.py`)

GPU: NVIDIA GeForce RTX 3070 Laptop GPU · torch 2.11.0+cu128 (CUDA 12.8) · TF32 off · CUDA events, 3 warm-ups, p50/p99 over repeats; inputs resident on GPU

The GPU was shared with other jobs (all launched through the same VRAM guard), so p99
tails can include contention; p50 is the robust statistic (mean ± sd over 3 timing rounds where shown).

## kNN top-1 over a memory bank (FP32 backends + FP16 baseline and tensor-core kernel)

| case | M queries | N bank | D | backend | p50 ms | p99 ms | TFLOP/s | extra VRAM MiB | idx agree |
|---|---|---|---|---|---|---|---|---|---|
| patchcore_1img | 1024 | 10000 | 1024 | torch_cdist_topk(chunk=6710) | 4.69 ± 0.03 | 5.68 | 4.47 | 84 | 1.0000 |
| patchcore_1img | 1024 | 10000 | 1024 | cupy_nvrtc_fused | 5.00 ± 0.12 | 5.76 | 4.19 | 0 | 1.0000 |
| patchcore_1img | 1024 | 10000 | 1024 | torch_fp16_mm_topk | 3.06 ± 0.01 | 4.05 | 6.85 | 62 | 0.9980 |
| patchcore_1img | 1024 | 10000 | 1024 | nvcc_ext_fused | 5.22 ± 0.06 | 6.19 | 4.02 | 0 | 1.0000 |
| patchcore_1img | 1024 | 10000 | 1024 | nvcc_wmma_fp16_top1 | 1.37 ± 0.08 | 1.98 | 15.29 | 2 | 0.9990 |
| patchcore_1img | 1024 | 100000 | 1024 | torch_cdist_topk(chunk=671) | 51.05 ± 0.43 | 53.06 | 4.11 | 651 | 1.0000 |
| patchcore_1img | 1024 | 100000 | 1024 | cupy_nvrtc_fused | 46.76 ± 0.20 | 48.19 | 4.48 | 0 | 1.0000 |
| patchcore_1img | 1024 | 100000 | 1024 | torch_fp16_mm_topk | 25.33 ± 0.08 | 27.59 | 8.28 | 386 | 0.9941 |
| patchcore_1img | 1024 | 100000 | 1024 | nvcc_ext_fused | 49.99 ± 0.21 | 50.84 | 4.20 | 0 | 1.0000 |
| patchcore_1img | 1024 | 100000 | 1024 | nvcc_wmma_fp16_top1 | 10.81 ± 0.07 | 11.82 | 19.40 | 2 | 0.9980 |
| patchcore_8img | 8192 | 10000 | 1024 | torch_cdist_topk(chunk=6710) | 35.04 ± 0.08 | 36.53 | 4.79 | 321 | 1.0000 |
| patchcore_8img | 8192 | 10000 | 1024 | cupy_nvrtc_fused | 43.53 ± 0.54 | 44.98 | 3.85 | 0 | 1.0000 |
| patchcore_8img | 8192 | 10000 | 1024 | torch_fp16_mm_topk | 19.89 ± 0.09 | 20.97 | 8.43 | 400 | 0.9983 |
| patchcore_8img | 8192 | 10000 | 1024 | nvcc_ext_fused | 45.31 ± 0.06 | 46.93 | 3.70 | 0 | 1.0000 |
| patchcore_8img | 8192 | 10000 | 1024 | nvcc_wmma_fp16_top1 | 11.59 ± 0.10 | 12.38 | 14.48 | 16 | 0.9993 |
| patchcore_8img | 8192 | 100000 | 1024 | torch_cdist_topk(chunk=671) | 391.96 ± 0.95 | 434.06 | 4.28 | 651 | 1.0000 |
| patchcore_8img | 8192 | 100000 | 1024 | cupy_nvrtc_fused | 445.59 ± 1.95 | 468.68 | 3.77 | 0 | 1.0000 |
| patchcore_8img | 8192 | 100000 | 1024 | torch_fp16_mm_topk | 199.07 ± 0.38 | 202.77 | 8.43 | 400 | 0.9972 |
| patchcore_8img | 8192 | 100000 | 1024 | nvcc_ext_fused | 477.54 ± 3.15 | 485.97 | 3.51 | 0 | 1.0000 |
| patchcore_8img | 8192 | 100000 | 1024 | nvcc_wmma_fp16_top1 | 133.33 ± 1.61 | 143.47 | 12.58 | 16 | 0.9991 |
| anomalydino_1img | 1024 | 131072 | 384 | torch_cdist_topk(chunk=512) | 34.53 ± 0.28 | 37.07 | 2.99 | 450 | 1.0000 |
| anomalydino_1img | 1024 | 131072 | 384 | cupy_nvrtc_fused | 25.84 ± 0.10 | 27.27 | 3.99 | 0 | 1.0000 |
| anomalydino_1img | 1024 | 131072 | 384 | torch_fp16_mm_topk | 26.27 ± 0.53 | 28.59 | 3.92 | 385 | 0.9990 |
| anomalydino_1img | 1024 | 131072 | 384 | nvcc_ext_fused | 26.21 ± 0.05 | 28.30 | 3.93 | 0 | 1.0000 |
| anomalydino_1img | 1024 | 131072 | 384 | nvcc_wmma_fp16_top1 | 6.89 ± 0.22 | 9.70 | 14.96 | 1 | 1.0000 |

## Fused SEM preprocessing (480x480 uint8 -> crop 477 -> area-resize -> normalise -> 3ch)

| out size | batch | backend | p50 ms | p99 ms | img/s | extra VRAM MiB | max abs err |
|---|---|---|---|---|---|---|---|---|
| 256 | 1 | torch_ops(crop,float,pool,repeat,norm) | 1.185 ± 0.136 | 2.423 | 844 | 2.6 | 0.0e+00 |
| 256 | 1 | cupy_nvrtc_fused | 0.199 ± 0.024 | 2.338 | 5014 | 0.8 | 8.3e-07 |
| 256 | 1 | nvcc_ext_fused | 0.151 ± 0.034 | 0.475 | 6632 | 0.8 | 8.3e-07 |
| 256 | 8 | torch_ops(crop,float,pool,repeat,norm) | 1.278 ± 0.121 | 1.866 | 6262 | 20.9 | 0.0e+00 |
| 256 | 8 | cupy_nvrtc_fused | 0.315 ± 0.059 | 0.900 | 25388 | 6.0 | 8.3e-07 |
| 256 | 8 | nvcc_ext_fused | 0.220 ± 0.021 | 0.925 | 36315 | 6.0 | 8.3e-07 |
| 256 | 32 | torch_ops(crop,float,pool,repeat,norm) | 1.428 ± 0.041 | 3.939 | 22413 | 83.8 | 0.0e+00 |
| 256 | 32 | cupy_nvrtc_fused | 0.355 ± 0.010 | 0.801 | 90124 | 24.0 | 8.3e-07 |
| 256 | 32 | nvcc_ext_fused | 0.276 ± 0.023 | 0.598 | 115828 | 24.0 | 8.3e-07 |
| 448 | 1 | torch_ops(crop,float,pool,repeat,norm) | 1.287 ± 0.102 | 1.883 | 777 | 6.2 | 0.0e+00 |
| 448 | 1 | cupy_nvrtc_fused | 0.279 ± 0.095 | 0.591 | 3590 | 2.3 | 7.2e-07 |
| 448 | 1 | nvcc_ext_fused | 0.183 ± 0.015 | 0.852 | 5453 | 2.3 | 7.2e-07 |
| 448 | 8 | torch_ops(crop,float,pool,repeat,norm) | 1.216 ± 0.076 | 2.192 | 6576 | 49.8 | 0.0e+00 |
| 448 | 8 | cupy_nvrtc_fused | 0.324 ± 0.048 | 0.675 | 24715 | 18.4 | 8.3e-07 |
| 448 | 8 | nvcc_ext_fused | 0.217 ± 0.035 | 0.515 | 36797 | 18.4 | 8.3e-07 |
| 448 | 32 | torch_ops(crop,float,pool,repeat,norm) | 2.094 ± 0.013 | 3.620 | 15283 | 199.3 | 0.0e+00 |
| 448 | 32 | cupy_nvrtc_fused | 0.598 ± 0.016 | 0.962 | 53495 | 73.5 | 8.3e-07 |
| 448 | 32 | nvcc_ext_fused | 0.463 ± 0.027 | 0.628 | 69064 | 73.5 | 8.3e-07 |

## Steady-state forward throughput (FP16 autocast, synthetic input, no data loading)

| model | input | batch | img/s | params | peak alloc MiB |
|---|---|---|---|---|---|
| dinov2_vits14 | 224 | 64 | 1270 | 21.6 M | 321 |
| dinov2_vitb14 | 224 | 32 | 428 | 85.7 M | 547 |
| dinov2_vits14 | 448 | 16 | 264 | 21.9 M | 326 |
| dinov2_vits14_reg | 280 | 32 | 765 | 21.7 M | 273 |
| wrn50_patchcore_embed | 256 | 16 | 464 | 24.9 M | 334 |

## PatchCore deployment: PyTorch vs TensorRT (+ custom kernels)

TensorRT 10.8.0.43. Memory bank from 136 train normals (13926 patches after 10 % coreset); accuracy on the full 919-image test split (45 normal). All variants share the FP32 bank.

| embedder | image AUROC | ΔAUROC vs torch FP32 | score Pearson r | max rel. score diff |
|---|---|---|---|---|
| torch_fp32 | 1.0000 | +0.0000 | 1.0000000 | 0.00e+00 |
| torch_amp_fp16 | 1.0000 | +0.0000 | 0.9999992 | 8.75e-03 |
| trt_fp32 | 1.0000 | +0.0000 | 0.9999995 | 5.65e-03 |
| trt_fp16 | 1.0000 | +0.0000 | 0.9999964 | 1.24e-02 |
| trt_fp16+wmma_fp16_knn | 1.0000 | +0.0000 | 0.9999964 | 1.27e-02 |

| embedder only | batch | p50 ms | p99 ms | img/s |
|---|---|---|---|---|
| torch_fp32 | 1 | 4.77 | 10.09 | 210 |
| torch_amp_fp16 | 1 | 6.90 | 14.25 | 145 |
| trt_fp32 | 1 | 3.29 | 3.74 | 304 |
| trt_fp16 | 1 | 2.35 | 2.73 | 426 |
| torch_fp32 | 8 | 26.74 | 28.85 | 299 |
| torch_amp_fp16 | 8 | 17.11 | 18.37 | 468 |
| trt_fp32 | 8 | 15.92 | 17.34 | 502 |
| trt_fp16 | 8 | 5.79 | 6.52 | 1381 |

| end-to-end from raw uint8 on GPU | batch | p50 ms | p99 ms | img/s |
|---|---|---|---|---|
| baseline_torch_fp32 | 1 | 11.91 | 16.30 | 84 |
| baseline_torch_amp_fp16 | 1 | 9.57 | 14.74 | 105 |
| kernels+trt_fp16 | 1 | 9.00 | 9.82 | 111 |
| kernels(wmma_knn)+trt_fp16 | 1 | 3.79 | 4.42 | 264 |
| baseline_torch_fp32 | 8 | 72.42 | 75.59 | 110 |
| baseline_torch_amp_fp16 | 8 | 44.02 | 46.24 | 182 |
| kernels+trt_fp16 | 8 | 60.86 | 63.20 | 131 |
| kernels(wmma_knn)+trt_fp16 | 8 | 21.19 | 22.37 | 378 |

Engine sizes (MB): {'fp32': 139.7, 'fp16': 49.8}; build seconds: {'fp32': 36.1, 'fp16': 40.5}; ONNX 95 MB.

EfficientAD-S TensorRT FP16 latency (untrained weights, latency only): bs1: p50 3.12 ms (320 img/s), bs8: p50 14.98 ms (534 img/s)

## C++ runner (trt_runner.exe): pinned H2D -> fused preprocess -> TensorRT -> fused kNN -> max -> D2H

All 919 test images, PatchCore bank of the 136 class-6 train images; mean ± sd over 3 repeated runs (each: p50 / p99 over the runner's iterations).

| engine | kNN kernel | batch | streams | p50 ms / batch | p99 ms / batch | img/s | runner VRAM MiB | parity vs PyTorch FP32 |
|---|---|---|---|---|---|---|---|---|
| patchcore_wrn50_256_fp16 | fp16 | 1 | 1 | 3.90 ± 0.01 | 4.21 ± 0.18 | 250 ± 1 | 132 | r=0.9999963, max rel 1.3e-02, dAUROC=+0.0000 |
| patchcore_wrn50_256_fp16 | fp32 | 1 | 1 | 8.85 ± 0.06 | 9.23 ± 0.19 | 112 ± 1 | 158 | r=0.9999963, max rel 1.3e-02, dAUROC=+0.0000 |
| patchcore_wrn50_256_fp16 | fp16 | 8 | 1 | 21.68 ± 0.06 | 22.54 ± 0.12 | 363 ± 1 | 182 | r=0.9999963, max rel 1.3e-02, dAUROC=+0.0000 |
| patchcore_wrn50_256_fp16 | fp32 | 8 | 1 | 61.19 ± 0.10 | 68.64 ± 11.00 | 129 ± 1 | 194 | r=0.9999963, max rel 1.3e-02, dAUROC=+0.0000 |
| patchcore_wrn50_256_fp16 | fp16 | 8 | 2 | 21.80 ± 0.09 | 22.57 ± 0.15 | 366 ± 2 | 284 | r=0.9999963, max rel 1.3e-02, dAUROC=+0.0000 |
| patchcore_wrn50_256_fp16 | fp32 | 8 | 2 | 62.70 ± 1.09 | 71.03 ± 7.23 | 127 ± 3 | 280 | r=0.9999963, max rel 1.3e-02, dAUROC=+0.0000 |
| patchcore_wrn50_256_fp32 | fp32 | 8 | 1 | 71.65 ± 0.52 | 72.84 ± 0.65 | 111 ± 1 | 338 | r=0.9999995, max rel 5.6e-03, dAUROC=+0.0000 |

## CUDA-graph replay in the runner (FP16 engine + tensor-core kNN)

| batch | CUDA graph | p50 ms / batch | p99 ms / batch | img/s | max |score diff| graph vs direct |
|---|---|---|---|---|---|
| 1 | no | 3.785 ± 0.015 | 4.100 ± 0.038 | 258 ± 1 | 0 |
| 1 | yes | 3.658 ± 0.011 | 3.870 ± 0.070 | 267 ± 1 | 0 |
| 8 | no | 21.577 ± 0.110 | 22.776 ± 0.217 | 363 ± 2 | 0 |
| 8 | yes | 21.792 ± 0.194 | 22.923 ± 0.405 | 361 ± 2 | 0 |

## C++ runner anomaly maps vs PyTorch (200 test images, 224² maps)

Maps = sqrt -> bilinear upsample -> Gaussian blur (sigma 4, reflect), computed by CUDA kernels in the runner.

| maps from | pixel AUROC | AUPRO@0.3 | AUPRO@0.05 | pixel AP | max abs diff vs PyTorch (relative) | mean abs diff |
|---|---|---|---|---|---|---|
| torch_fp32 | 0.98665 | 0.95941 | 0.82307 | 0.79054 | reference | — |
| cpp_fp32_knnfp32 | 0.98665 | 0.95940 | 0.82303 | 0.79049 | 0.0048 (0.16%) | 0.00037 |
| cpp_fp16_knnfp16 | 0.98664 | 0.95938 | 0.82300 | 0.79035 | 0.0122 (0.41%) | 0.00116 |

## Speed-up decomposition: precision gain vs engineering gain

precision gain = PyTorch FP32 / PyTorch FP16 (same algorithm, half precision via AMP / cuBLAS FP16); engineering gain = PyTorch FP16 / ours (custom kernels, TensorRT). p50 latency in ms (mean ± sd over 3 timing rounds when available).

| measurement | PyTorch FP32 | PyTorch FP16 | ours (FP16) | precision gain | engineering gain | total |
|---|---|---|---|---|---|---|
| kNN patchcore_1img, N=10k, D=1024 | 4.69 ± 0.03 | 3.06 ± 0.01 | 1.37 ± 0.08 | 1.53× | 2.23× | 3.42× |
| kNN patchcore_1img, N=100k, D=1024 | 51.05 ± 0.43 | 25.33 ± 0.08 | 10.81 ± 0.07 | 2.02× | 2.34× | 4.72× |
| kNN patchcore_8img, N=10k, D=1024 | 35.04 ± 0.08 | 19.89 ± 0.09 | 11.59 ± 0.10 | 1.76× | 1.72× | 3.02× |
| kNN patchcore_8img, N=100k, D=1024 | 391.96 ± 0.95 | 199.07 ± 0.38 | 133.33 ± 1.61 | 1.97× | 1.49× | 2.94× |
| kNN anomalydino_1img, N=131k, D=384 | 34.53 ± 0.28 | 26.27 ± 0.53 | 6.89 ± 0.22 | 1.31× | 3.81× | 5.01× |
| PatchCore end-to-end, batch 1 | 11.91 ± 1.30 | 9.57 ± 0.56 | 3.79 ± 0.04 | 1.24× | 2.53× | 3.14× |
| PatchCore end-to-end, batch 8 | 72.42 ± 1.28 | 44.02 ± 0.39 | 21.19 ± 0.11 | 1.65× | 2.08× | 3.42× |

