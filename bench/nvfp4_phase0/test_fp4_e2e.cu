// NVFP4 e2e: argv[1]=32x4|linear  argv[2]=stride_ncb_rd|stride_ncb
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define CKC(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "cuda err %s @%d\n", cudaGetErrorString(e_), __LINE__); \
    return 1; } } while (0)
#define CKL(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cublasLt err %d @%d\n", (int)s_, __LINE__); \
    return 1; } } while (0)

static int g_use_32x4 = 1;
static int g_stride_ncb_rd = 1;

__device__ __forceinline__ float h3_e2m1_round(float q) {
    float a = fabsf(q); float r;
    if (a < 0.25f) r = 0.0f; else if (a < 0.75f) r = 0.5f;
    else if (a < 1.25f) r = 1.0f; else if (a < 1.75f) r = 1.5f;
    else if (a < 2.5f) r = 2.0f; else if (a < 3.5f) r = 3.0f;
    else if (a < 5.0f) r = 4.0f; else r = 6.0f;
    return copysignf(r, q);
}
__device__ __forceinline__ uint8_t h3_e2m1_nibble(float q) {
    float v = h3_e2m1_round(q); float a = fabsf(v); uint8_t mag;
    if (a < 0.25f) mag = 0; else if (a < 0.75f) mag = 1;
    else if (a < 1.25f) mag = 2; else if (a < 1.75f) mag = 3;
    else if (a < 2.5f) mag = 4; else if (a < 3.5f) mag = 5;
    else if (a < 5.0f) mag = 6; else mag = 7;
    return (uint8_t)((v < 0.0f ? 8 : 0) | mag);
}
__device__ __forceinline__ uint8_t h3_f32_to_ue4m3(float v) {
    int e, m, E; float f;
    if (!(v > 0.0f)) return 0;
    f = frexpf(v, &e); E = e - 1 + 7;
    if (E <= 0) { m = (int)lroundf(ldexpf(v, 6) * 8.0f); if (m > 7) m = 7; return (uint8_t)m; }
    if (E >= 15) return 0x7E;
    m = (int)lroundf((f * 2.0f - 1.0f) * 8.0f);
    if (m == 8) { E += 1; m = 0; if (E >= 15) return 0x7E; }
    return (uint8_t)((E << 3) | m);
}
__device__ __forceinline__ float h3_ue4m3_to_f32(uint8_t b) {
    int e = (b >> 3) & 0xF, m = b & 0x7;
    if (e == 0) return ldexpf((float)m / 8.0f, -6);
    return ldexpf(1.0f + (float)m / 8.0f, e - 7);
}

__device__ int d_use_32x4;
__device__ int d_stride_ncb_rd;

__device__ __forceinline__ size_t sf_off(size_t row, size_t g, size_t ncb, size_t ncb_rd) {
    size_t stride = d_stride_ncb_rd ? ncb_rd : ncb;
    size_t tile_base = (row >> 7) * stride;
    size_t r128 = row & 127u;
    size_t cc = g & 3u;
    size_t tc = g >> 2;
    size_t in;
    if (d_use_32x4)
        in = 16u * (r128 & 31u) + 4u * (r128 >> 5) + cc;
    else
        in = r128 * 4u + cc;
    return (tile_base + tc) * 512u + in;
}

__global__ void quant_nvfp4_kernel(const float *__restrict__ x,
                                   uint8_t *__restrict__ x_fp4,
                                   uint8_t *__restrict__ sf, size_t m, size_t k) {
    const size_t row = (size_t)blockIdx.x + (size_t)blockIdx.y * gridDim.x;
    const int tid = (int)threadIdx.x;
    const int mb = tid >> 4;
    const int lane_in_mb = tid & 15;
    const size_t nc = k >> 4;
    const size_t ncb = nc >> 2;
    const size_t ncb_rd = (ncb + 31u) & ~(size_t)31u;
    if (row >= m) return;
    const float *xr = x + row * k;
    uint8_t *xr_fp4 = x_fp4 + row * (k >> 1);
    int nblk = (int)(nc / 16);
    for (int blk = 0; blk < nblk; blk++) {
        int g = blk * 16 + mb;
        float v = xr[(size_t)g * 16 + lane_in_mb];
        float amax = fabsf(v);
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 8));
        float scale_f = (amax > 0.0f) ? (amax / 6.0f) : 1.0f;
        uint8_t sf_b = h3_f32_to_ue4m3(scale_f);
        scale_f = h3_ue4m3_to_f32(sf_b);
        float inv = 1.0f / scale_f;
        uint8_t nib = h3_e2m1_nibble(v * inv);
        if (lane_in_mb == 0)
            sf[sf_off(row, (size_t)g, ncb, ncb_rd)] = sf_b;
        uint8_t nib_lo = (uint8_t)__shfl_down_sync(0xffffffffu, nib, 1);
        if ((lane_in_mb & 1) == 0) {
            size_t col = (size_t)g * 16 + lane_in_mb;
            xr_fp4[col >> 1] = (uint8_t)((nib << 4) | (nib_lo & 0xF));
        }
    }
}

static size_t sf_bytes_for(size_t rows_pad, size_t k, int stride_rd) {
    size_t nc = k / 16, ncb = nc / 4;
    size_t ncb_rd = (ncb + 31) & ~31u;
    size_t rb = rows_pad / 128;
    size_t stride = stride_rd ? ncb_rd : ncb;
    /* allocate enough for either; use max */
    size_t cols = stride_rd ? ncb_rd : ncb;
    (void)stride;
    return rb * cols * 512;
}

static float e2m1_from_nib(uint8_t nib) {
    static const float mag[8] = {0,0.5f,1,1.5f,2,3,4,6};
    float v = mag[nib & 7];
    return (nib & 8) ? -v : v;
}
static float ue4m3_to_f32_host(uint8_t b) {
    int e = (b >> 3) & 0xF, m = b & 0x7;
    if (e == 0) return ldexpf((float)m / 8.0f, -6);
    return ldexpf(1.0f + (float)m / 8.0f, e - 7);
}
static uint8_t f32_to_ue4m3_host(float v) {
    int e, m, E; float f;
    if (!(v > 0.0f)) return 0;
    f = frexpf(v, &e); E = e - 1 + 7;
    if (E <= 0) { m = (int)lroundf(ldexpf(v, 6) * 8.0f); if (m > 7) m = 7; return (uint8_t)m; }
    if (E >= 15) return 0x7E;
    m = (int)lroundf((f * 2.0f - 1.0f) * 8.0f);
    if (m == 8) { E += 1; m = 0; if (E >= 15) return 0x7E; }
    return (uint8_t)((E << 3) | m);
}
static uint8_t e2m1_nibble_host(float q) {
    float a = fabsf(q), r, v; uint8_t mag;
    if (a < 0.25f) r = 0; else if (a < 0.75f) r = 0.5f;
    else if (a < 1.25f) r = 1; else if (a < 1.75f) r = 1.5f;
    else if (a < 2.5f) r = 2; else if (a < 3.5f) r = 3;
    else if (a < 5.0f) r = 4; else r = 6;
    v = copysignf(r, q); a = fabsf(v);
    if (a < 0.25f) mag = 0; else if (a < 0.75f) mag = 1;
    else if (a < 1.25f) mag = 2; else if (a < 1.75f) mag = 3;
    else if (a < 2.5f) mag = 4; else if (a < 3.5f) mag = 5;
    else if (a < 5.0f) mag = 6; else mag = 7;
    return (uint8_t)((v < 0 ? 8 : 0) | mag);
}
static void quant_ref_row(const float *x, size_t k, uint8_t *fp4, float *sf16) {
    size_t nc = k / 16;
    for (size_t g = 0; g < nc; g++) {
        float amax = 0;
        for (int j = 0; j < 16; j++) amax = fmaxf(amax, fabsf(x[g * 16 + j]));
        float scale = (amax > 0) ? (amax / 6.0f) : 1.0f;
        uint8_t sfb = f32_to_ue4m3_host(scale);
        scale = ue4m3_to_f32_host(sfb);
        sf16[g] = scale;
        float inv = 1.0f / scale;
        for (int j = 0; j < 16; j += 2) {
            uint8_t ne = e2m1_nibble_host(x[g * 16 + j] * inv);
            uint8_t no = e2m1_nibble_host(x[g * 16 + j + 1] * inv);
            fp4[(g * 16 + j) / 2] = (uint8_t)((ne << 4) | (no & 0xF));
        }
    }
}

int main(int argc, char **argv) {
    g_use_32x4 = 1; g_stride_ncb_rd = 1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "linear")) g_use_32x4 = 0;
        if (!strcmp(argv[i], "32x4")) g_use_32x4 = 1;
        if (!strcmp(argv[i], "stride_ncb")) g_stride_ncb_rd = 0;
        if (!strcmp(argv[i], "stride_ncb_rd")) g_stride_ncb_rd = 1;
    }
    CKC(cudaMemcpyToSymbol(d_use_32x4, &g_use_32x4, sizeof(int)));
    CKC(cudaMemcpyToSymbol(d_stride_ncb_rd, &g_stride_ncb_rd, sizeof(int)));
    printf("mode: %s %s\n", g_use_32x4 ? "32x4" : "linear",
           g_stride_ncb_rd ? "stride_ncb_rd" : "stride_ncb");

    const size_t m = 256, n = 256, k = 512;
    const size_t m_pad = m, n_pad = n;
    float *hx = (float *)malloc(m * k * 4), *hw = (float *)malloc(n * k * 4);
    srand(7);
    for (size_t i = 0; i < m * k; i++) hx[i] = ((rand() % 2000) - 1000) / 250.f;
    for (size_t i = 0; i < n * k; i++) hw[i] = ((rand() % 2000) - 1000) / 250.f;

    uint8_t *hxf = (uint8_t *)malloc(m * k / 2), *hwf = (uint8_t *)malloc(n * k / 2);
    float *sxf = (float *)malloc(m * (k / 16) * 4), *swf = (float *)malloc(n * (k / 16) * 4);
    for (size_t r = 0; r < m; r++) quant_ref_row(hx + r * k, k, hxf + r * (k / 2), sxf + r * (k / 16));
    for (size_t r = 0; r < n; r++) quant_ref_row(hw + r * k, k, hwf + r * (k / 2), swf + r * (k / 16));

    float *href = (float *)malloc(m * n * 4);
    size_t nc = k / 16;
    for (size_t r = 0; r < m; r++)
        for (size_t c = 0; c < n; c++) {
            double acc = 0;
            for (size_t g = 0; g < nc; g++) {
                float sx = sxf[r * nc + g], sw = swf[c * nc + g];
                for (int j = 0; j < 16; j += 2) {
                    uint8_t bx = hxf[(r * k + g * 16 + j) / 2];
                    uint8_t bw = hwf[(c * k + g * 16 + j) / 2];
                    float xe = e2m1_from_nib((uint8_t)(bx >> 4));
                    float xo = e2m1_from_nib((uint8_t)(bx & 0xF));
                    float we = e2m1_from_nib((uint8_t)(bw >> 4));
                    float wo = e2m1_from_nib((uint8_t)(bw & 0xF));
                    acc += (double)(xe * sx) * (we * sw);
                    acc += (double)(xo * sx) * (wo * sw);
                }
            }
            href[r * n + c] = (float)acc;
        }

    float *dx, *dw; uint8_t *dx4, *dw4, *dsfx, *dsfw; __nv_bfloat16 *dy;
    size_t sfx_bytes = sf_bytes_for(m_pad, k, g_stride_ncb_rd);
    size_t sfw_bytes = sf_bytes_for(n_pad, k, g_stride_ncb_rd);
    /* always alloc max pad so device never OOB if symbol wrong */
    size_t sfx_alloc = sf_bytes_for(m_pad, k, 1), sfw_alloc = sf_bytes_for(n_pad, k, 1);
    CKC(cudaMalloc(&dx, m * k * 4)); CKC(cudaMalloc(&dw, n * k * 4));
    CKC(cudaMalloc(&dx4, m_pad * k / 2)); CKC(cudaMalloc(&dw4, n_pad * k / 2));
    CKC(cudaMalloc(&dsfx, sfx_alloc)); CKC(cudaMalloc(&dsfw, sfw_alloc));
    CKC(cudaMalloc(&dy, m_pad * n * 2));
    CKC(cudaMemset(dx4, 0, m_pad * k / 2)); CKC(cudaMemset(dw4, 0, n_pad * k / 2));
    CKC(cudaMemset(dsfx, 0, sfx_alloc)); CKC(cudaMemset(dsfw, 0, sfw_alloc));
    CKC(cudaMemcpy(dx, hx, m * k * 4, cudaMemcpyHostToDevice));
    CKC(cudaMemcpy(dw, hw, n * k * 4, cudaMemcpyHostToDevice));

    unsigned gx = (unsigned)m_pad, gy = 1;
    quant_nvfp4_kernel<<<dim3(gx, gy, 1), 256>>>(dx, dx4, dsfx, m, k);
    quant_nvfp4_kernel<<<dim3((unsigned)n_pad, 1, 1), 256>>>(dw, dw4, dsfw, n, k);
    CKC(cudaDeviceSynchronize());

    /* SF occupancy */
    uint8_t *hsf = (uint8_t *)malloc(sfx_alloc);
    CKC(cudaMemcpy(hsf, dsfx, sfx_alloc, cudaMemcpyDeviceToHost));
    size_t nz0=0,nz1=0;
    for (size_t i=0;i<sfx_alloc/2;i++) if(hsf[i]) nz0++;
    for (size_t i=sfx_alloc/2;i<sfx_alloc;i++) if(hsf[i]) nz1++;
    printf("SF nonzero bytes lo_half=%zu hi_half=%zu (alloc=%zu)\n", nz0, nz1, sfx_alloc);

    cublasLtHandle_t lt; CKL(cublasLtCreate(&lt));
    cublasLtMatmulDesc_t desc;
    CKL(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    CKL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
    CKL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
    cublasLtMatmulMatrixScale_t v16 = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    cublasLtMatmulMatrixScale_t s32 = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    CKL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &v16, sizeof(v16)));
    CKL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &v16, sizeof(v16)));
    CKL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_C_SCALE_MODE, &s32, sizeof(s32)));
    CKL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_D_SCALE_MODE, &s32, sizeof(s32)));
    CKL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dsfw, sizeof(dsfw)));
    CKL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dsfx, sizeof(dsfx)));
    cublasLtPointerMode_t pm = CUBLASLT_POINTER_MODE_DEVICE;
    CKL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pm, sizeof(pm)));
    cublasLtMatrixLayout_t la, lb, lc;
    CKL(cublasLtMatrixLayoutCreate(&la, CUDA_R_4F_E2M1, k, n_pad, k));
    CKL(cublasLtMatrixLayoutCreate(&lb, CUDA_R_4F_E2M1, k, m_pad, k));
    CKL(cublasLtMatrixLayoutCreate(&lc, CUDA_R_16BF, n, m_pad, n));
    cublasLtMatmulPreference_t pref; CKL(cublasLtMatmulPreferenceCreate(&pref));
    size_t ws = 64 << 20; void *dws; CKC(cudaMalloc(&dws, ws));
    CKL(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
    cublasLtMatmulHeuristicResult_t heur[4]; int ret = 0;
    CKL(cublasLtMatmulAlgoGetHeuristic(lt, desc, la, lb, lc, lc, pref, 4, heur, &ret));
    if (ret < 1) { fprintf(stderr, "no fp4 algo\n"); return 1; }
    float *dab; CKC(cudaMalloc(&dab, 8));
    float hab[2] = {1.f, 0.f};
    CKC(cudaMemcpy(dab, hab, 8, cudaMemcpyHostToDevice));
    CKL(cublasLtMatmul(lt, desc, dab, dw4, la, dx4, lb, dab + 1, dy, lc, dy, lc,
                       &heur[0].algo, dws, ws, 0));
    CKC(cudaDeviceSynchronize());

    __nv_bfloat16 *hy = (__nv_bfloat16 *)malloc(m_pad * n * 2);
    CKC(cudaMemcpy(hy, dy, m_pad * n * 2, cudaMemcpyDeviceToHost));
    double max_abs = 0, sref = 0, serr = 0; size_t nan_n = 0;
    for (size_t i = 0; i < m * n; i++) {
        float yv = __bfloat162float(hy[i]);
        if (!isfinite(yv)) { nan_n++; continue; }
        double e = fabs((double)yv - href[i]);
        if (e > max_abs) max_abs = e;
        sref += (double)href[i] * href[i]; serr += e * e;
    }
    double rel = sqrt(serr / (sref + 1e-9));
    printf("rel_l2=%.6f max_abs=%.4f nan=%zu\n", rel, max_abs, nan_n);
    size_t rows_chk[] = {0,1,32,127,128,200};
    for (size_t ri = 0; ri < 6; ri++) {
        size_t r = rows_chk[ri];
        double sr=0,se=0;
        for (size_t c = 0; c < n; c++) {
            size_t i = r * n + c;
            float yv = __bfloat162float(hy[i]);
            double e = fabs((double)yv - href[i]);
            sr += (double)href[i]*href[i]; se += e*e;
        }
        printf("  row %zu: rel=%.6f Y0=%.3f ref0=%.3f\n",
               r, sqrt(se/(sr+1e-9)), __bfloat162float(hy[r*n]), href[r*n]);
    }
    int pass = (rel < 0.05 && nan_n == 0);
    printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
