// R-D baseline: cublasLt INT8 GEMM on the 8 production DiT shapes.
// Mirrors h3_cuda_lt_gemm_i8_i32() from h3_gpu_cuda.cu: CUBLAS_COMPUTE_32I,
// int8 x int8 -> int32, default algo (heuristic pick=0), 256MiB workspace.
// This is the number CUTLASS must match (>=95%) for E1 to proceed.
//
// Build (on 238):
//   nvcc -O3 -x cu -std=c++14 -arch=sm_120a bench_cublaslt_i8.c \
//        -lcublasLt -o bench_cublaslt_i8
// Run: ./bench_cublaslt_i8

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cublasLt.h>
#include <cuda_runtime.h>

struct Shape { int m, n, k; const char* tag; };
static const Shape SHAPES[] = {
    {3200, 21504, 5376,  "480p fc1"},
    {3200, 5376,  7168,  "480p fc2"},
    {3200, 28672, 5376,  "480p qkv/out"},
    {3200, 5376,  14336, "480p out-adaln"},
    {4096, 21504, 5376,  "768p fc1"},
    {4096, 5376,  7168,  "768p fc2"},
    {4096, 28672, 5376,  "768p qkv/out"},
    {4096, 5376,  14336, "768p out-adaln"},
};

#define CK(x) do { cublasStatus_t st_ = (x); if (st_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cublasLt err %d @%d\n", (int)st_, __LINE__); return -1; } } while(0)

static float run_shape(cublasLtHandle_t lt, const Shape& s,
                       int8_t* dA, int8_t* dB, int32_t* dC, void* dws,
                       size_t ws_bytes, int iters = 300) {
    // cublasLt layout mirrors h3_cuda_lt_gemm_i8_i32:
    //   desc: CUBLAS_COMPUTE_32I, scaleType CUDA_R_32I
    //   A layout: [n, k] row-major  (weights)
    //   B layout: [m, k] row-major  (acts, viewed transposed)
    //   C layout: [n, m] row-major  (acc int32)
    // Mirror h3_gpu_cuda.cu:2441-2479 exactly:
    //   TRANSA=T, TRANSB=N
    //   A layout: [k, n] ld=k  (col-major view of row-major [n,k])
    //   B layout: [k, m] ld=k  (col-major view of row-major [m,k])
    //   C layout: [n, m] ld=n  (col-major [m,n])
    cublasLtMatmulDesc_t desc;
    CK(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32I, CUDA_R_32I));
    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;
    CK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                      &transa, sizeof(transa)));
    CK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                      &transb, sizeof(transb)));

    cublasLtMatrixLayout_t la, lb, lc;
    CK(cublasLtMatrixLayoutCreate(&la, CUDA_R_8I,  s.k, s.n, s.k));
    CK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_8I,  s.k, s.m, s.k));
    CK(cublasLtMatrixLayoutCreate(&lc, CUDA_R_32I, s.n, s.m, s.n));

    cublasLtMatmulPreference_t pref;
    CK(cublasLtMatmulPreferenceCreate(&pref));
    CK(cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof(ws_bytes)));

    cublasLtMatmulHeuristicResult_t heur[4];
    int returned = 0;
    CK(cublasLtMatmulAlgoGetHeuristic(lt, desc, la, lb, lc, lc, pref,
                                      4, heur, &returned));
    if (returned < 1) { fprintf(stderr, "no heuristic\n"); return -1; }

    int32_t alpha = 1, beta = 0;
    cublasStatus_t st;
    // aggressive warmup to push GPU to boost clocks
    for (int i = 0; i < 200; ++i) {
        st = cublasLtMatmul(lt, desc, &alpha, dA, la, dB, lb, &beta, dC, lc,
                            dC, lc, &heur[0].algo, dws, ws_bytes, 0);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "warmup matmul st=%d @%d\n", (int)st, __LINE__);
            return -1;
        }
    }
    cudaError_t ce = cudaDeviceSynchronize();
    if (ce != cudaSuccess) { fprintf(stderr, "warmup sync: %s\n", cudaGetErrorString(ce)); return -1; }

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < iters; ++i) {
        st = cublasLtMatmul(lt, desc, &alpha, dA, la, dB, lb, &beta, dC, lc,
                            dC, lc, &heur[0].algo, dws, ws_bytes, 0);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "loop matmul st=%d @%d\n", (int)st, __LINE__);
            return -1;
        }
    }
    cudaEventRecord(t1);
    ce = cudaEventSynchronize(t1);
    if (ce != cudaSuccess) { fprintf(stderr, "event sync: %s\n", cudaGetErrorString(ce)); return -1; }
    float ms; cudaEventElapsedTime(&ms, t0, t1);

    cublasLtMatmulDescDestroy(desc);
    cublasLtMatrixLayoutDestroy(la);
    cublasLtMatrixLayoutDestroy(lb);
    cublasLtMatrixLayoutDestroy(lc);
    cublasLtMatmulPreferenceDestroy(pref);
    return ms / iters;
}

int main() {
    cublasLtHandle_t lt;
    cublasLtCreate(&lt);
    size_t ws = 256ull * 1024 * 1024;

    int8_t* dA; int8_t* dB; int32_t* dC; void* dws;
    cudaMalloc(&dA, (size_t)28672 * 14336);
    cudaMalloc(&dB, (size_t)4096 * 14336);
    cudaMalloc(&dC, (size_t)28672 * 4096 * 4);
    cudaMalloc(&dws, ws);
    cudaMemset(dA, 0x1, (size_t)28672 * 14336);
    cudaMemset(dB, 0x1, (size_t)4096 * 14336);

    // push GPU to boost clocks with a 3s GEMM burn
    printf("# pre-heating GPU clocks...\n");
    {
        Shape big = {4096, 28672, 14336, "burn"};
        for (int i = 0; i < 3; ++i) run_shape(lt, big, dA, dB, dC, dws, ws, 500);
    }
    printf("# cublasLt INT8 (CUBLAS_COMPUTE_32I) baseline — production config\n\n");
    for (int i = 0; i < 8; ++i) {
        float ms = run_shape(lt, SHAPES[i], dA, dB, dC, dws, ws, 500);
        double fl = 2.0 * SHAPES[i].m * SHAPES[i].n * SHAPES[i].k;
        double tf = fl / (ms * 1e-3) / 1e12;
        printf("[%d] m=%-5d n=%-6d k=%-6d  ms=%7.3f  TFLOPs=%7.1f  %s\n",
               i, SHAPES[i].m, SHAPES[i].n, SHAPES[i].k, ms, tf, SHAPES[i].tag);
        fflush(stdout);
    }
    return 0;
}
