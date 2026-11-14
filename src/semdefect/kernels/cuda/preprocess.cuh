// Fused SEM image preprocessing kernel (single source for NVRTC and nvcc builds).
//
//   (B, H, W) uint8 | uint16 grey  --crop-->  (B, ch, cw)  --resize-->  (B, oh, ow)
//   --> * in_scale (1/255 or 1/65535) --> (v - mean_c) * inv_std_c  --> (B, 3, oh, ow) float32
//
// One read of the raw detector image, one write of the network input: replaces 5 separate
// PyTorch ops (to-float, crop, resize, scale, normalise+channel-replicate) and their temporaries.
//
// Resize modes reproduce PyTorch exactly:
//   mode 0 = "area"     == F.adaptive_avg_pool2d   (anti-aliased downscale; used for training)
//   mode 1 = "bilinear" == F.interpolate(mode="bilinear", align_corners=False, antialias=False)
//
// Self-contained on purpose: no #includes, so NVRTC compiles it without a CUDA toolkit on disk.
#pragma once

#ifndef SEMDEFECT_PP_THREADS
#define SEMDEFECT_PP_THREADS 256
#endif

template <typename Tin>
__device__ __forceinline__ float sd_load(const Tin* __restrict__ img, long long pitch, int y, int x) {
    return (float)img[(long long)y * pitch + x];
}

template <typename Tin>
__device__ __forceinline__ void sem_preprocess_impl(
    const Tin* __restrict__ src, float* __restrict__ dst, int B, int H, int W, int y0, int x0, int ch, int cw,
    int oh, int ow, int mode, float in_scale, float m0, float m1, float m2, float is0, float is1, float is2) {
    const long long total = (long long)B * oh * ow;
    const long long plane = (long long)oh * ow;
    for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; t < total;
         t += (long long)gridDim.x * blockDim.x) {
        const int ox = (int)(t % ow);
        const int oy = (int)((t / ow) % oh);
        const int b = (int)(t / plane);
        const Tin* img = src + (long long)b * H * W + (long long)y0 * W + x0;
        float v;
        if (mode == 0) {  // adaptive average pooling bins: [floor(i*in/out), ceil((i+1)*in/out))
            const int sy = (oy * ch) / oh, ey = ((oy + 1) * ch + oh - 1) / oh;
            const int sx = (ox * cw) / ow, ex = ((ox + 1) * cw + ow - 1) / ow;
            float acc = 0.f;
            for (int y = sy; y < ey; ++y)
                for (int x = sx; x < ex; ++x) acc += sd_load(img, W, y, x);
            v = acc / (float)((ey - sy) * (ex - sx));
        } else {  // bilinear, half-pixel centres, clamp at 0 (PyTorch area_pixel_compute_source_index)
            const float scy = (float)ch / (float)oh, scx = (float)cw / (float)ow;
            float fy = ((float)oy + 0.5f) * scy - 0.5f;
            float fx = ((float)ox + 0.5f) * scx - 0.5f;
            fy = fy < 0.f ? 0.f : fy;
            fx = fx < 0.f ? 0.f : fx;
            const int iy = (int)fy, ix = (int)fx;
            const int py = iy < ch - 1 ? 1 : 0, px = ix < cw - 1 ? 1 : 0;
            const float ly = fy - (float)iy, lx = fx - (float)ix;
            const float hy = 1.f - ly, hx = 1.f - lx;
            v = hy * (hx * sd_load(img, W, iy, ix) + lx * sd_load(img, W, iy, ix + px)) +
                ly * (hx * sd_load(img, W, iy + py, ix) + lx * sd_load(img, W, iy + py, ix + px));
        }
        v *= in_scale;
        float* o = dst + (long long)b * 3 * plane + (long long)oy * ow + ox;
        o[0] = (v - m0) * is0;
        o[plane] = (v - m1) * is1;
        o[2 * plane] = (v - m2) * is2;
    }
}

extern "C" __global__ void __launch_bounds__(SEMDEFECT_PP_THREADS) sem_preprocess_u8(
    const unsigned char* __restrict__ src, float* __restrict__ dst, int B, int H, int W, int y0, int x0, int ch,
    int cw, int oh, int ow, int mode, float in_scale, float m0, float m1, float m2, float is0, float is1, float is2) {
    sem_preprocess_impl<unsigned char>(src, dst, B, H, W, y0, x0, ch, cw, oh, ow, mode, in_scale, m0, m1, m2, is0, is1,
                                       is2);
}

extern "C" __global__ void __launch_bounds__(SEMDEFECT_PP_THREADS) sem_preprocess_u16(
    const unsigned short* __restrict__ src, float* __restrict__ dst, int B, int H, int W, int y0, int x0, int ch,
    int cw, int oh, int ow, int mode, float in_scale, float m0, float m1, float m2, float is0, float is1, float is2) {
    sem_preprocess_impl<unsigned short>(src, dst, B, H, W, y0, x0, ch, cw, oh, ow, mode, in_scale, m0, m1, m2, is0,
                                        is1, is2);
}
