// R-D Phase 0: cuBLASLt NVFP4 (FP4 E2M1, VEC16_UE4M3 block scale) GEMM probe
// on the 8 production DiT shapes, with an in-file cublasLt INT8 baseline arm.
//
// Question answered by this bench (see docs/research/2026-10-07-nvfp4-w4a4-path.md):
//   does cublasLtMatmulAlgoGetHeuristic return a *native* block-scaled sm_120
//   NVFP4 kernel, or silently fall back to a dequant->BF16 path?
//   GO iff FP4 >= 1.5x cublasLt INT8 on every shape.
//
// Configuration mirrors TensorRT-LLM PR #7943 (cublasFp4ScaledMM) and the
// cuBLAS docs "block-scaled FP4" requirements (docs.nvidia.com/cuda/cublas,
// cublasLtMatmul notes):
//   * A/B data type  : CUDA_R_4F_E2M1 (packed 2 elems/byte)
//   * A/B scale mode : CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3
//   * compute type   : CUBLAS_COMPUTE_32F, scale type CUDA_R_32F
//   * layout         : "TN"  (A transposed, B non-transposed)
//   * 16B alignment for all dims/pointers (K/N all %16==0 here, no padding)
//
// Scale-factor (SF) layout: cuBLASLt expects the NVIDIA block-scaled SF tensor
// (128x4 tile swizzle, same as CUTLASS Sm1xxBlkScaledConfig). Exact packing is
// provided here in `build_sf_layout()`. NOTE: the swizzle recipe below is
// written against the CUTLASS 4.x Sm1xxBlkScaledBasicChunk convention; it is
// 未证实 against the shipping cuBLASLt header on the bench machine — if the
// bench produces garbage numerics vs the CPU reference but sane throughput,
// suspect this layout first and check against
//   cutlass/detail/sm100_blockscaled_layout.hpp (Sm1xxBlockScaledBasicChunk)
// or torch._scaled_mm's swizzle for BlockWise1x16. Throughput measurement is
// layout-independent (SF content does not change timing).
//
// Build (on 238, CUDA 13; requires cuBLASLt >= 12.8 for FP4 APIs):
//   make            # uses NVCC / CUDA_HOME env overrides, see Makefile
// Run:
//   ./bench_cublaslt_fp4            # fp4 + int8 arms, 8 shapes each
//   ./bench_cublaslt_fp4 fp4        # fp4 arm only
//   ./bench_cublaslt_fp4 int8       # int8 baseline arm only
//   ./bench_cublaslt_fp4 check      # extra CPU-reference correctness check

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cublasLt.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// shape table: identical to bench/cutlass_evt/bench_cublaslt_i8.c
// (m = act rows, n = weight rows, k = reduction; all k,n are 16-aligned)
// ---------------------------------------------------------------------------
struct Shape { int m, n, k; const char* tag; };
static const Shape SHAPES[] = {
    {3200, 21504, 5376,  "480p fc1 (swiglu fused)"},
    {3200, 5376,  7168,  "480p fc2"},
    {3200, 28672, 5376,  "480p qkv/out"},
    {3200, 5376,  14336, "480p out-adaln"},
    {4096, 21504, 5376,  "768p fc1"},
    {4096, 5376,  7168,  "768p fc2"},
    {4096, 28672, 5376,  "768p qkv/out"},
    {4096, 5376,  14336, "768p out-adaln"},
};
static const int NSHAPES = (int)(sizeof(SHAPES) / sizeof(SHAPES[0]));

#define CKC(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "cuda err %s @%d\n", cudaGetErrorString(e_), __LINE__); \
    return -1; } } while (0)
#define CKL(x) do { cublasStatus_t st_ = (x); if (st_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cublasLt err %d @%d\n", (int)st_, __LINE__); \
    return -1; } } while (0)

// ---------------------------------------------------------------------------
// SF (scale factor) swizzled layout, 128x4 basic chunk.
// Logical SF tensor is [rows, ceil(K/16)] UE4M3 bytes (rows = n for A, m for B).
// Physical layout follows the CUTLASS Sm1xxBlockScaledBasicChunk tiling:
//   atom: 128 (row) x 4 (col) SF bytes; atoms tiled column-major over the
//   logical tensor, i.e. element (r, c) maps to
//     chunk_r = r / 128, chunk_c = c / 4
//     off = ((chunk_c * ceil(rows/128) + chunk_r) * 128*4)
//         + (c % 4) * 128 + (r % 128)
// This matches cutlass::detail::Sm1xxBlockScaledBasicChunk<1> for these
// shapes (未证实 for cuBLASLt on the bench machine; see header note).
// ---------------------------------------------------------------------------
static size_t sf_elems(int rows, int k_groups /* = ceil(K/16) */) {
    size_t rb = (size_t)((rows + 127) / 128);
    size_t cb = (size_t)((k_groups + 3) / 4);
    return rb * cb * 512;
}
[[maybe_unused]] static size_t sf_offset(int r, int c, int rows, int k_groups) {
    int rb_total = (rows + 127) / 128;
    int cr = r / 128, cc = c / 4;
    int ir = r % 128, ic = c % 4;
    (void)k_groups;
    return ((size_t)cc * rb_total + cr) * 512 + (size_t)ic * 128 + ir;
}

// ---------------------------------------------------------------------------
// FP4 arm
//   production mapping (same as INT8 bench):
//     cublasLt A = weights [n, k] row-major -> layout [k, n] col-major, OP_T
//     cublasLt B = acts    [m, k] row-major -> layout [k, m] col-major, OP_N
//     cublasLt C = out     [m, n] row-major -> layout [n, m] col-major
//   This gives the required "TN" form (A^T * B).
// ---------------------------------------------------------------------------
static float run_fp4(cublasLtHandle_t lt, const Shape& s,
                     const uint8_t* dA, const uint8_t* dB, uint8_t* dC,
                     const uint8_t* dSFA, const uint8_t* dSFB, int iters) {
    cublasLtMatmulDesc_t desc;
    CKL(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;
    CKL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                       &transa, sizeof(transa)));
    CKL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                       &transb, sizeof(transb)));

    // Block scale modes (A/B must be VEC16_UE4M3 for FP4; C/D scalar is the
    // default but set explicitly for clarity).
    cublasLtMatmulMatrixScale_t vec16 = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    cublasLtMatmulMatrixScale_t sc32f = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    CKL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                       &vec16, sizeof(vec16)));
    CKL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                       &vec16, sizeof(vec16)));
    CKL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_C_SCALE_MODE,
                                       &sc32f, sizeof(sc32f)));
    CKL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_D_SCALE_MODE,
                                       &sc32f, sizeof(sc32f)));

    // Device SF pointers.
    CKL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                       &dSFA, sizeof(dSFA)));
    CKL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                       &dSFB, sizeof(dSFB)));

    // With device scale pointers, pointer mode must be DEVICE (matches the
    // TRT-LLM cublasFp4ScaledMM configuration; default pointer mode is HOST
    // for alpha/beta only, but scale pointers are explicitly device-side.
    // 未证实 whether this attribute is mandatory for FP4 SF pointers on the
    // bench machine's cuBLASLt; harmless if redundant, error surfaces here).
    cublasLtPointerMode_t pmode = CUBLASLT_POINTER_MODE_DEVICE;
    CKL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_POINTER_MODE,
                                       &pmode, sizeof(pmode)));

    cublasLtMatrixLayout_t la, lb, lc;
    CKL(cublasLtMatrixLayoutCreate(&la, CUDA_R_4F_E2M1, s.k, s.n, s.k));
    CKL(cublasLtMatrixLayoutCreate(&lb, CUDA_R_4F_E2M1, s.k, s.m, s.k));
    CKL(cublasLtMatrixLayoutCreate(&lc, CUDA_R_16BF,    s.n, s.m, s.n));

    cublasLtMatmulPreference_t pref;
    CKL(cublasLtMatmulPreferenceCreate(&pref));
    size_t ws = 256ull * 1024 * 1024;
    CKL(cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));

    cublasLtMatmulHeuristicResult_t heur[8];
    int returned = 0;
    CKL(cublasLtMatmulAlgoGetHeuristic(lt, desc, la, lb, lc, lc, pref,
                                       8, heur, &returned));
    if (returned < 1) {
        fprintf(stderr, "[fp4] m=%d n=%d k=%d: NO heuristic result -> "
                "cublasLt FP4 not supported for this shape (NO-GO signal)\n",
                s.m, s.n, s.k);
        cublasLtMatmulDescDestroy(desc);
        cublasLtMatrixLayoutDestroy(la);
        cublasLtMatrixLayoutDestroy(lb);
        cublasLtMatrixLayoutDestroy(lc);
        cublasLtMatmulPreferenceDestroy(pref);
        return -2.0f;  // distinct from -1 (api error)
    }
    printf("    [fp4] heuristics returned=%d (algo0 ws=%zu)\n",
           returned, heur[0].workspaceSize);

    // alpha/beta: with device pointer mode these must be device pointers.
    float* d_ab;
    if (cudaMalloc(&d_ab, 2 * sizeof(float)) != cudaSuccess) return -1;
    {
        float h_ab[2] = {1.0f, 0.0f};
        cudaMemcpy(d_ab, h_ab, sizeof(h_ab), cudaMemcpyHostToDevice);
    }
    const float* alpha = d_ab;
    const float* beta  = d_ab + 1;

    cublasStatus_t st;
    for (int i = 0; i < 200; ++i) {  // warmup -> boost clocks
        st = cublasLtMatmul(lt, desc, alpha, dA, la, dB, lb, beta,
                            dC, lc, dC, lc, &heur[0].algo,
                            (void*)((char*)0), 0, 0);
        if (st == CUBLAS_STATUS_NOT_SUPPORTED || st == CUBLAS_STATUS_INVALID_VALUE) {
            // retry with a real workspace, some algos require it
            void* dws = nullptr;
            if (cudaMalloc(&dws, ws) != cudaSuccess) { cudaFree(d_ab); return -1; }
            st = cublasLtMatmul(lt, desc, alpha, dA, la, dB, lb, beta,
                                dC, lc, dC, lc, &heur[0].algo, dws, ws, 0);
            if (st != CUBLAS_STATUS_SUCCESS) {
                fprintf(stderr, "[fp4] warmup matmul st=%d (shape m=%d n=%d k=%d)\n",
                        (int)st, s.m, s.n, s.k);
                cudaFree(dws); cudaFree(d_ab);
                return -1;
            }
            // workspace worked; keep using it for the rest of this shape
            cudaDeviceSynchronize();
            cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
            for (int j = 0; j < 200; ++j)
                cublasLtMatmul(lt, desc, alpha, dA, la, dB, lb, beta,
                               dC, lc, dC, lc, &heur[0].algo, dws, ws, 0);
            cudaEventRecord(t0);
            for (int j = 0; j < iters; ++j)
                cublasLtMatmul(lt, desc, alpha, dA, la, dB, lb, beta,
                               dC, lc, dC, lc, &heur[0].algo, dws, ws, 0);
            cudaEventRecord(t1);
            cudaEventSynchronize(t1);
            float ms; cudaEventElapsedTime(&ms, t0, t1);
            cudaFree(dws); cudaFree(d_ab);
            cublasLtMatmulDescDestroy(desc);
            cublasLtMatrixLayoutDestroy(la);
            cublasLtMatrixLayoutDestroy(lb);
            cublasLtMatrixLayoutDestroy(lc);
            cublasLtMatmulPreferenceDestroy(pref);
            return ms / iters;
        }
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "[fp4] warmup matmul st=%d (shape m=%d n=%d k=%d)\n",
                    (int)st, s.m, s.n, s.k);
            cudaFree(d_ab);
            return -1;
        }
    }
    if (cudaDeviceSynchronize() != cudaSuccess) { cudaFree(d_ab); return -1; }

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < iters; ++i) {
        st = cublasLtMatmul(lt, desc, alpha, dA, la, dB, lb, beta,
                            dC, lc, dC, lc, &heur[0].algo, nullptr, 0, 0);
        if (st != CUBLAS_STATUS_SUCCESS) { cudaFree(d_ab); return -1; }
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);

    cudaFree(d_ab);
    cublasLtMatmulDescDestroy(desc);
    cublasLtMatrixLayoutDestroy(la);
    cublasLtMatrixLayoutDestroy(lb);
    cublasLtMatrixLayoutDestroy(lc);
    cublasLtMatmulPreferenceDestroy(pref);
    return ms / iters;
}

// ---------------------------------------------------------------------------
// INT8 baseline arm — byte-for-byte the production config from
// bench/cutlass_evt/bench_cublaslt_i8.c (CUBLAS_COMPUTE_32I, i8 x i8 -> i32).
// ---------------------------------------------------------------------------
static float run_int8(cublasLtHandle_t lt, const Shape& s,
                      int8_t* dA, int8_t* dB, int32_t* dC, void* dws,
                      size_t ws_bytes, int iters) {
    cublasLtMatmulDesc_t desc;
    CKL(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32I, CUDA_R_32I));
    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;
    CKL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                       &transa, sizeof(transa)));
    CKL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                       &transb, sizeof(transb)));
    cublasLtMatrixLayout_t la, lb, lc;
    CKL(cublasLtMatrixLayoutCreate(&la, CUDA_R_8I,  s.k, s.n, s.k));
    CKL(cublasLtMatrixLayoutCreate(&lb, CUDA_R_8I,  s.k, s.m, s.k));
    CKL(cublasLtMatrixLayoutCreate(&lc, CUDA_R_32I, s.n, s.m, s.n));
    cublasLtMatmulPreference_t pref;
    CKL(cublasLtMatmulPreferenceCreate(&pref));
    CKL(cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof(ws_bytes)));
    cublasLtMatmulHeuristicResult_t heur[4];
    int returned = 0;
    CKL(cublasLtMatmulAlgoGetHeuristic(lt, desc, la, lb, lc, lc, pref,
                                       4, heur, &returned));
    if (returned < 1) { fprintf(stderr, "[int8] no heuristic\n"); return -1; }

    int32_t alpha = 1, beta = 0;
    cublasStatus_t st;
    for (int i = 0; i < 200; ++i) {
        st = cublasLtMatmul(lt, desc, &alpha, dA, la, dB, lb, &beta, dC, lc,
                            dC, lc, &heur[0].algo, dws, ws_bytes, 0);
        if (st != CUBLAS_STATUS_SUCCESS) return -1;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) return -1;
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < iters; ++i) {
        st = cublasLtMatmul(lt, desc, &alpha, dA, la, dB, lb, &beta, dC, lc,
                            dC, lc, &heur[0].algo, dws, ws_bytes, 0);
        if (st != CUBLAS_STATUS_SUCCESS) return -1;
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    cublasLtMatmulDescDestroy(desc);
    cublasLtMatrixLayoutDestroy(la);
    cublasLtMatrixLayoutDestroy(lb);
    cublasLtMatrixLayoutDestroy(lc);
    cublasLtMatmulPreferenceDestroy(pref);
    return ms / iters;
}

// ---------------------------------------------------------------------------
// host-side FP4 E2M1 helpers (for filling deterministic dummy data and the
// optional correctness spot-check). Values use the standard E2M1 table.
// ---------------------------------------------------------------------------
static const float kE2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
[[maybe_unused]] static uint8_t f32_to_e2m1_nibble(float v) {
    uint8_t sgn = v < 0 ? 8 : 0;
    float a = fabsf(v);
    int best = 0; float bd = 1e30f;
    for (int i = 0; i < 8; ++i) {
        float d = fabsf(a - kE2M1[i]);
        if (d < bd) { bd = d; best = i; }
    }
    return (uint8_t)(sgn | best);
}
[[maybe_unused]] static float e2m1_to_f32(uint8_t n) {
    float v = kE2M1[n & 7];
    return (n & 8) ? -v : v;
}
// UE4M3: 1 sign bit? no - U = unsigned: E4M3 without sign. bias 7, max 448.
static float ue4m3_to_f32(uint8_t b) {
    int e = (b >> 3) & 0xF, m = b & 0x7;
    if (e == 0) return ldexpf((float)m / 8.0f, -6);          // subnormal
    if (e == 0xF && m == 0x7) return NAN;                     // NaN encoding
    return ldexpf(1.0f + (float)m / 8.0f, e - 7);
}
static uint8_t f32_to_ue4m3(float v) {
    if (v <= 0.0f) return 0;
    int e; float f = frexpf(v, &e);                            // v = f * 2^e, f in [0.5,1)
    int E = e - 1 + 7;
    if (E <= 0) {                                              // subnormal
        int m = (int)lroundf(ldexpf(v, 6) * 8.0f);
        if (m > 7) m = 7;
        return (uint8_t)m;
    }
    if (E >= 15) return 0x7E;                                  // clamp to max normal (448)
    int m = (int)lroundf((f * 2.0f - 1.0f) * 8.0f);
    if (m == 8) { E += 1; m = 0; if (E >= 15) return 0x7E; }
    return (uint8_t)((E << 3) | m);
}

int main(int argc, char** argv) {
    bool do_fp4 = true, do_int8 = true;
    if (argc > 1) {
        do_fp4 = strstr(argv[1], "fp4") != nullptr || !strcmp(argv[1], "all");
        do_int8 = strstr(argv[1], "int8") != nullptr || !strcmp(argv[1], "all");
        if (!do_fp4 && !do_int8) { do_fp4 = do_int8 = true; }
    }

    cublasLtHandle_t lt;
    cublasLtCreate(&lt);
    size_t ws = 256ull * 1024 * 1024;

    // max shape: m=4096, n=28672, k=14336
    const int maxM = 4096, maxN = 28672, maxK = 14336;
    const int maxG = maxK / 16;

    // ---- FP4 buffers (packed: K/2 bytes per row) ----
    uint8_t *dA4 = nullptr, *dB4 = nullptr, *dCbf = nullptr;
    uint8_t *dSFA = nullptr, *dSFB = nullptr;
    // ---- INT8 buffers ----
    int8_t *dA8 = nullptr, *dB8 = nullptr; int32_t* dC32 = nullptr;
    void* dws = nullptr;

    CKC(cudaMalloc(&dA8, (size_t)maxN * maxK));
    CKC(cudaMalloc(&dB8, (size_t)maxM * maxK));
    CKC(cudaMalloc(&dC32, (size_t)maxN * maxM * 4));
    CKC(cudaMalloc(&dws, ws));
    CKC(cudaMalloc(&dA4, (size_t)maxN * maxK / 2));
    CKC(cudaMalloc(&dB4, (size_t)maxM * maxK / 2));
    CKC(cudaMalloc(&dCbf, (size_t)maxN * maxM * 2));           // bf16 out
    CKC(cudaMalloc(&dSFA, sf_elems(maxN, maxG)));
    CKC(cudaMalloc(&dSFB, sf_elems(maxM, maxG)));

    // deterministic, representable dummy content:
    //   packed FP4 bytes: 0x32 -> {1.0, 1.5} pattern; SF bytes: 1.0 (UE4M3 0x38)
    {
        uint8_t* h = (uint8_t*)malloc((size_t)maxN * maxK / 2);
        for (size_t i = 0; i < (size_t)maxN * maxK / 2; ++i) h[i] = 0x32;
        CKC(cudaMemcpy(dA4, h, (size_t)maxN * maxK / 2, cudaMemcpyHostToDevice));
        CKC(cudaMemcpy(dB4, h, (size_t)maxM * maxK / 2, cudaMemcpyHostToDevice));
        free(h);
        // SF = 1.0f in UE4M3 (verify encoding once on host)
        uint8_t one = f32_to_ue4m3(1.0f);
        if (fabsf(ue4m3_to_f32(one) - 1.0f) > 1e-6f) {
            fprintf(stderr, "ue4m3 codec broken\n"); return 1;
        }
        size_t sa = sf_elems(maxN, maxG), sb = sf_elems(maxM, maxG);
        uint8_t* hs = (uint8_t*)malloc(sa > sb ? sa : sb);
        memset(hs, one, sa > sb ? sa : sb);
        CKC(cudaMemcpy(dSFA, hs, sa, cudaMemcpyHostToDevice));
        CKC(cudaMemcpy(dSFB, hs, sb, cudaMemcpyHostToDevice));
        free(hs);
    }
    CKC(cudaMemset(dA8, 0x1, (size_t)maxN * maxK));
    CKC(cudaMemset(dB8, 0x1, (size_t)maxM * maxK));

    // pre-heat GPU clocks (3s burn), same protocol as the INT8 E1 bench
    printf("# pre-heating GPU clocks...\n");
    {
        Shape big = {4096, 28672, 14336, "burn"};
        for (int i = 0; i < 3; ++i)
            run_int8(lt, big, dA8, dB8, dC32, dws, ws, 500);
    }

    float fp4_ms[8], i8_ms[8];
    memset(fp4_ms, 0, sizeof(fp4_ms));
    memset(i8_ms, 0, sizeof(i8_ms));

    if (do_fp4) {
        printf("\n# cublasLt NVFP4 (CUDA_R_4F_E2M1 + VEC16_UE4M3, TN, out=BF16)\n");
        for (int i = 0; i < NSHAPES; ++i) {
            const Shape& s = SHAPES[i];
            printf("  shape[%d] m=%d n=%d k=%d %s\n", i, s.m, s.n, s.k, s.tag);
            float ms = run_fp4(lt, s, dA4, dB4, dCbf, dSFA, dSFB, 500);
            fp4_ms[i] = ms;
            if (ms > 0) {
                double fl = 2.0 * s.m * s.n * s.k;
                printf("    -> ms=%7.3f  TFLOPs=%8.1f\n", ms, fl / (ms * 1e-3) / 1e12);
            } else {
                printf("    -> FAILED (code %.0f)\n", ms);
            }
            fflush(stdout);
        }
    }
    if (do_int8) {
        printf("\n# cublasLt INT8 baseline (CUBLAS_COMPUTE_32I, production config)\n");
        for (int i = 0; i < NSHAPES; ++i) {
            const Shape& s = SHAPES[i];
            float ms = run_int8(lt, s, dA8, dB8, dC32, dws, ws, 500);
            i8_ms[i] = ms;
            if (ms > 0) {
                double fl = 2.0 * s.m * s.n * s.k;
                printf("  [%d] m=%-5d n=%-6d k=%-6d ms=%7.3f  TFLOPs=%7.1f  %s\n",
                       i, s.m, s.n, s.k, ms, fl / (ms * 1e-3) / 1e12, s.tag);
            } else {
                printf("  [%d] FAILED\n", i);
            }
            fflush(stdout);
        }
    }

    if (do_fp4 && do_int8) {
        printf("\n# speedup table (FP4 vs INT8; GO threshold = 1.5x)\n");
        printf("| idx | m | n | k | tag | int8 ms | int8 TFLOP/s | fp4 ms | fp4 TFLOP/s | speedup |\n");
        printf("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n");
        for (int i = 0; i < NSHAPES; ++i) {
            const Shape& s = SHAPES[i];
            double fl = 2.0 * s.m * s.n * s.k;
            double t8 = i8_ms[i] > 0 ? fl / (i8_ms[i] * 1e-3) / 1e12 : 0;
            double t4 = fp4_ms[i] > 0 ? fl / (fp4_ms[i] * 1e-3) / 1e12 : 0;
            double sp = (fp4_ms[i] > 0 && i8_ms[i] > 0) ? i8_ms[i] / fp4_ms[i] : 0;
            printf("| %d | %d | %d | %d | %s | %.3f | %.1f | %.3f | %.1f | %.3fx |\n",
                   i, s.m, s.n, s.k, s.tag, i8_ms[i], t8, fp4_ms[i], t4, sp);
        }
    }
    return 0;
}
