/*
 * quant_act_extracted.cuh — Phase-0 extraction of the h3.c NVFP4 activation
 * quantize path. Verbatim logic, namespaced `h3p0` to avoid clashes.
 *
 * Sources (h3.c repo, file h3_gpu_cuda.cu):
 *   - h3_e2m1_round                      : lines 4338-4351
 *   - h3_e2m1_nibble                     : lines 4353-4366
 *   - h3_f32_to_ue4m3                    : lines 4368-4383
 *   - h3_ue4m3_to_f32                    : lines 4385-4390
 *   - h3_cuda_quantize_act_nvfp4_kernel  : lines 4403-4483
 *   - host wrapper launch config         : lines 4489-4517
 *
 * The SF swizzle layout (measured on sm_120 cuBLASLt, see the comment block at
 * h3_gpu_cuda.cu:4326-4337):
 *   logical (row r, micro-block c)  ->
 *     tile = (r/128)*ncb + c/4        (ncb = (K/16)/4, K%64==0)
 *     off  = tile*512 + (r%128)*4 + (c%4)
 * plus K-tail coverage out to ceil(ncb/32)*32 tile-columns (2048-K-elem
 * granularity) and sf=0 on padding rows, because UE4M3 bytes >= 0xFE are NaN.
 *
 * Host wrapper contract: caller allocates everything; this file never mallocs.
 */
#pragma once

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stddef.h>

namespace h3p0 {

/* SF buffer size in bytes for an [m, k] activation (K%64==0).
 * ceil(m/128) row-tiles * round_up(k/64, 32) tile-cols * 512 bytes. */
static inline size_t sf_bytes(size_t m, size_t k) {
    size_t nc = k / 16;              /* micro-blocks per row          */
    size_t ncb = nc / 4;             /* tile-cols actually used (k/64) */
    size_t ncb_rd = (ncb + 31u) & ~(size_t)31u; /* padded to 2048-K elems */
    return ((m + 127u) / 128u) * ncb_rd * 512u;
}

/* packed FP4 data size in bytes for an [m, k] activation */
static inline size_t fp4_bytes(size_t m, size_t k) { return m * (k / 2); }

/* E2M1 magnitudes {0,.5,1,1.5,2,3,4,6}; round-to-nearest on the FP4 grid.
 * (h3_gpu_cuda.cu:4338-4351) */
__device__ __forceinline__ float e2m1_round(float q) {
    float a = fabsf(q);
    float r;
    if (a < 0.25f) r = 0.0f;
    else if (a < 0.75f) r = 0.5f;
    else if (a < 1.25f) r = 1.0f;
    else if (a < 1.75f) r = 1.5f;
    else if (a < 2.5f) r = 2.0f;
    else if (a < 3.5f) r = 3.0f;
    else if (a < 5.0f) r = 4.0f;
    else r = 6.0f;
    return copysignf(r, q);
}

/* (h3_gpu_cuda.cu:4353-4366) */
__device__ __forceinline__ uint8_t e2m1_nibble(float q) {
    float v = e2m1_round(q);
    float a = fabsf(v);
    uint8_t mag;
    if (a < 0.25f) mag = 0;
    else if (a < 0.75f) mag = 1;
    else if (a < 1.25f) mag = 2;
    else if (a < 1.75f) mag = 3;
    else if (a < 2.5f) mag = 4;
    else if (a < 3.5f) mag = 5;
    else if (a < 5.0f) mag = 6;
    else mag = 7;   /* E2M1 max = 6.0 (bits 0111); no NaN encoding exists */
    return (uint8_t)((v < 0.0f ? 8 : 0) | mag);
}

/* (h3_gpu_cuda.cu:4368-4383) */
__device__ __forceinline__ uint8_t f32_to_ue4m3(float v) {
    int e, m, E;
    float f;
    if (!(v > 0.0f)) return 0;          /* 0 / negative / NaN -> 0 scale */
    f = frexpf(v, &e);                  /* v = f*2^e, f in [0.5,1) */
    E = e - 1 + 7;
    if (E <= 0) {                        /* subnormal */
        m = (int)lroundf(ldexpf(v, 6) * 8.0f);
        if (m > 7) m = 7;
        return (uint8_t)m;
    }
    if (E >= 15) return 0x7E;            /* clamp to max normal (448) */
    m = (int)lroundf((f * 2.0f - 1.0f) * 8.0f);
    if (m == 8) { E += 1; m = 0; if (E >= 15) return 0x7E; }
    return (uint8_t)((E << 3) | m);
}

/* (h3_gpu_cuda.cu:4385-4390) */
__device__ __forceinline__ float ue4m3_to_f32(uint8_t b) {
    int e = (b >> 3) & 0xF, m = b & 0x7;
    if (e == 0) return ldexpf((float)m / 8.0f, -6);
    if (e == 0xF && m == 0x7) return NAN;
    return ldexpf(1.0f + (float)m / 8.0f, e - 7);
}

/* Host-side decode of the same formats (for dequant / spot-checks). */
static inline float e2m1_to_f32_host(uint8_t nib) {
    static const float lut[16] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                  -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f };
    return lut[nib & 0xF];
}
static inline float ue4m3_to_f32_host(uint8_t b) {
    int e = (b >> 3) & 0xF, m = b & 0x7;
    if (e == 0) return ldexpf((float)m / 8.0f, -6);
    if (e == 0xF && m == 0x7) return NAN;
    return ldexpf(1.0f + (float)m / 8.0f, e - 7);
}

/* x[m,k] f32 -> x_fp4[m,k/2] packed (hi nibble = even K) + sf swizzled UE4M3.
 * Grid is sized over m_pad = round_up(m,128); rows [m, m_pad) run a zero-fill
 * path (nibble 0 is memset by the caller; sf 0 written here) because cuBLASLt
 * reads the full 128-row x 2048-K-elem SF tile grid and a stale byte >= 0xFE
 * is UE4M3 NaN. (h3_gpu_cuda.cu:4403-4483) */
__global__ void quantize_act_nvfp4_kernel(const float *__restrict__ x,
                                          uint8_t *__restrict__ x_fp4,
                                          uint8_t *__restrict__ sf,
                                          size_t m, size_t k) {
    const size_t row = (size_t)blockIdx.x + (size_t)blockIdx.y * gridDim.x;
    const int tid = (int)threadIdx.x;
    const int mb = tid >> 4;             /* micro-block index within block */
    const int lane_in_mb = tid & 15;
    const size_t nc = k >> 4;            /* k/16 */
    const int mb_per_block = 16;         /* 256 threads / 16 lanes */
    const size_t ncb = nc >> 2;          /* k/64; assumes k%64==0 */
    /* cuBLASLt reads ceil(ncb/32)*32 tile-columns per 128-row tile (2048 K
     * elems each); cover that padded extent so no stale byte is ever read. */
    const size_t ncb_rd = (ncb + 31u) & ~(size_t)31u;
    const size_t tile_base = (row >> 7) * ncb_rd;
    const size_t r128 = row & 127u;      /* row % 128 : major axis in tile */
    if (row >= m) {
        /* Padding row: zero-fill ALL 4 byte-columns of its SF across the full
         * padded extent (writing only cc=0 leaves cc=1..3 as NaN garbage). */
        if (lane_in_mb == 0) {
            size_t tc;
            int cc;
            for (tc = (size_t)(tid >> 4); tc < ncb_rd; tc += 16u)
                for (cc = 0; cc < 4; cc++)
                    sf[(tile_base + tc) * 512 + r128 * 4 + (size_t)cc] = 0;
        }
        return;
    }
    {
        const float *xr = x + row * k;
        uint8_t *xr_fp4 = x_fp4 + row * (k >> 1);
        int nblk = (int)(nc / mb_per_block);
        int blk;
        for (blk = 0; blk < nblk; blk++) {
            int g = blk * mb_per_block + mb;   /* global micro-block in row */
            float v = xr[(size_t)g * 16 + lane_in_mb];
            float amax = fabsf(v);
            float scale_f, inv;
            uint8_t sf_b;
            uint8_t nib;
            /* 16-wide butterfly amax within the micro-block (lanes 0-15 of
             * each half-warp). */
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 8));
            scale_f = (amax > 0.0f) ? (amax / 6.0f) : 1.0f;
            sf_b = f32_to_ue4m3(scale_f);
            scale_f = ue4m3_to_f32(sf_b);   /* use exact representable */
            inv = 1.0f / scale_f;
            nib = e2m1_nibble(v * inv);
            /* lane0 writes SF in the measured sm_120 layout. */
            if (lane_in_mb == 0) {
                size_t cc = (size_t)g & 3u;
                size_t off_sf =
                    (tile_base + ((size_t)g >> 2)) * 512 + r128 * 4 + cc;
                sf[off_sf] = sf_b;
            }
            /* gather lo nibble from odd lane to even lane */
            {
                uint8_t nib_lo = (uint8_t)__shfl_down_sync(0xffffffffu, nib, 1);
                if ((lane_in_mb & 1) == 0) {
                    size_t col = (size_t)g * 16 + lane_in_mb; /* even */
                    xr_fp4[col >> 1] = (uint8_t)((nib << 4) | (nib_lo & 0xF));
                }
            }
        }
        /* K-tail zero-fill: tile-cols [ncb, ncb_rd) for K%2048!=0. */
        if (ncb_rd > ncb && lane_in_mb == 0) {
            size_t tc;
            int cc;
            for (tc = ncb + (size_t)(tid >> 4); tc < ncb_rd; tc += 16u)
                for (cc = 0; cc < 4; cc++)
                    sf[(tile_base + tc) * 512 + r128 * 4 + (size_t)cc] = 0;
        }
    }
}

/* Host wrapper (h3_gpu_cuda.cu:4489-4517). Caller provides all buffers:
 *   d_x     [m*k]       FP32 input (device)
 *   d_x_fp4 [fp4_bytes(m_pad,k)]  pre-memset(0) by caller (padding nibble=0;
 *                                   kernel launches over m_pad rows, so the
 *                                   buffer must cover padding rows too)
 *   d_sf    [sf_bytes(m,k)]       pre-memset(0) by caller (belt-and-braces)
 * Returns cudaGetLastError() after launch. */
static inline cudaError_t quantize_act(const float *d_x, uint8_t *d_x_fp4,
                                       uint8_t *d_sf, size_t m, size_t k,
                                       cudaStream_t stream = 0) {
    unsigned threads = 256;
    size_t m_pad = (m + 127u) & ~(size_t)127u;
    unsigned gx = m_pad > 65535u ? 65535u : (unsigned)m_pad;
    unsigned gy = (unsigned)((m_pad + gx - 1u) / gx);
    dim3 grid(gx, gy, 1);
    quantize_act_nvfp4_kernel<<<grid, threads, 0, stream>>>(d_x, d_x_fp4, d_sf,
                                                            m, k);
    return cudaGetLastError();
}

} // namespace h3p0
