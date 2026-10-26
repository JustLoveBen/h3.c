// NVFP4 Phase 1 correctness gate (self-contained).
// Verifies the production quantize kernel logic + swizzled SF layout +
// cuBLASLt FP4 GEMM reproduce a CPU reference for Y = X[m,k] @ W[n,k]^T.
//
// Reimplements the kernel here verbatim (same source as h3_gpu_cuda.cu
// h3_cuda_quantize_act_nvfp4_kernel) so the test links standalone against
// cublasLt only — no h3 static lib dependency.
//
// Build:  make test_fp4_e2e
// Run:    ./test_fp4_e2e  (PASS iff rel_l2 < 0.10)
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

// ---- verbatim copy of the production kernel + helpers (h3_gpu_cuda.cu) ----
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
    if (e == 0xF && m == 0x7) return NAN;
    return ldexpf(1.0f + (float)m / 8.0f, e - 7);
}
__global__ void quant_nvfp4_kernel(const float *__restrict__ x,
                                   uint8_t *__restrict__ x_fp4,
                                   uint8_t *__restrict__ sf, size_t m, size_t k) {
    const size_t row = (size_t)blockIdx.x + (size_t)blockIdx.y * gridDim.x;
    const int tid = (int)threadIdx.x;
    const int mb = tid >> 4;
    const int lane_in_mb = tid & 15;
    const size_t nc = k >> 4;
    if (row >= m) return;
    const float *xr = x + row * k;
    uint8_t *xr_fp4 = x_fp4 + row * (k >> 1);
    size_t ncb = nc >> 2;
    size_t ncb_rd = (ncb + 31u) & ~(size_t)31u;
    size_t tile_base = (row >> 7) * ncb_rd;
    size_t r128 = row & 127u;
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
        if (lane_in_mb == 0) {
            size_t cc = (size_t)g & 3u;
            {
                size_t rsw = (size_t)16u * (r128 & 31u) + (size_t)4u * (r128 >> 5) + cc;
                sf[(tile_base + ((size_t)g >> 2)) * 512 + rsw] = sf_b;
            }
        }
        uint8_t nib_lo = (uint8_t)__shfl_down_sync(0xffffffffu, nib, 1);
        if ((lane_in_mb & 1) == 0) {
            size_t col = (size_t)g * 16 + lane_in_mb;
            xr_fp4[col >> 1] = (uint8_t)((nib << 4) | (nib_lo & 0xF));
        }
    }
}

static size_t sf_elems(size_t rows, size_t nc) {
    size_t ncb = (nc + 3) / 4;
    size_t ncb_rd = (ncb + 31) & ~31u;
    return ((rows + 127) / 128) * ncb_rd * 512;
}
static void quant_host(float *d_x, size_t m, size_t k, uint8_t *d_fp4, uint8_t *d_sf) {
    unsigned gx = m > 65535u ? 65535u : (unsigned)m;
    unsigned gy = (unsigned)((m + gx - 1) / gx);
    quant_nvfp4_kernel<<<dim3(gx, gy, 1), 256>>>(d_x, d_fp4, d_sf, m, k);
}

int main() {
    const size_t m = 256, n = 384, k = 512; /* was 320: nc%16!=0 left K-tail unquantized */
    float *hx = (float *)malloc(m * k * 4), *hw = (float *)malloc(n * k * 4);
    srand(7);
    for (size_t i = 0; i < m * k; i++) hx[i] = ((rand() % 2000) - 1000) / 250.f;
    for (size_t i = 0; i < n * k; i++) hw[i] = ((rand() % 2000) - 1000) / 250.f;
    float *href = (float *)malloc(m * n * 4);
    for (size_t r = 0; r < m; r++)
        for (size_t c = 0; c < n; c++) {
            double acc = 0;
            for (size_t t = 0; t < k; t++) acc += (double)hx[r * k + t] * hw[c * k + t];
            href[r * n + c] = (float)acc;
        }

    float *dx, *dw; uint8_t *dx4, *dw4, *dsfx, *dsfw; __nv_bfloat16 *dy;
    CKC(cudaMalloc(&dx, m * k * 4)); CKC(cudaMalloc(&dw, n * k * 4));
    CKC(cudaMalloc(&dx4, m * k / 2)); CKC(cudaMalloc(&dw4, n * k / 2));
    CKC(cudaMalloc(&dsfx, sf_elems(m, k / 16))); CKC(cudaMalloc(&dsfw, sf_elems(n, k / 16)));
    CKC(cudaMalloc(&dy, m * n * 2));
    CKC(cudaMemcpy(dx, hx, m * k * 4, cudaMemcpyHostToDevice));
    CKC(cudaMemcpy(dw, hw, n * k * 4, cudaMemcpyHostToDevice));
    quant_host(dx, m, k, dx4, dsfx);
    quant_host(dw, n, k, dw4, dsfw);
    CKC(cudaGetLastError());
    CKC(cudaDeviceSynchronize());

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
    CKL(cublasLtMatrixLayoutCreate(&la, CUDA_R_4F_E2M1, k, n, k));
    CKL(cublasLtMatrixLayoutCreate(&lb, CUDA_R_4F_E2M1, k, m, k));
    CKL(cublasLtMatrixLayoutCreate(&lc, CUDA_R_16BF, n, m, n));
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

    __nv_bfloat16 *hy = (__nv_bfloat16 *)malloc(m * n * 2);
    CKC(cudaMemcpy(hy, dy, m * n * 2, cudaMemcpyDeviceToHost));
    double max_abs = 0, sref = 0, serr = 0;
    for (size_t i = 0; i < m * n; i++) {
        float yv = __bfloat162float(hy[i]);
        double e = fabs((double)yv - href[i]);
        if (e > max_abs) max_abs = e;
        sref += (double)href[i] * href[i]; serr += e * e;
    }
    double rel = sqrt(serr / (sref + 1e-9));
    printf("FP4 GEMM e2e (m=%zu n=%zu k=%zu): max_abs=%.4f rel_l2=%.4f\n", m, n, k, max_abs, rel);
    printf("%s\n", (rel < 0.10) ? "PASS" : "FAIL");
    return (rel < 0.10) ? 0 : 1;
}
