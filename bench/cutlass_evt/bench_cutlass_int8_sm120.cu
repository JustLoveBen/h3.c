// R-D: CUTLASS sm_120 INT8 GEMM bench for 8 production DiT shapes.
// Compares CUTLASS sm80-style TensorOp INT8 collective (what cublasLt picks on
// sm_120: cutlass_80_i16832) emitted directly from CUTLASS headers, against
// the cublasLt production baseline (measured separately by bench_cublaslt_i8.c).
//
// For each shape we try a small tile-config sweep and report the best.
// Decision rule (from audit): pass if CUTLASS >= 95% of cublasLt on every shape.
//
// Build (on 238):
//   nvcc -O3 -std=c++17 -arch=sm_120a \
//     -I/mnt/nvme1n1/workspace/cutlass/include \
//     -I/mnt/nvme1n1/workspace/cutlass/tools/util/include \
//     bench_cutlass_int8_sm120.cu -o bench_cutlass_int8_sm120
//
// Run: ./bench_cutlass_int8_sm120

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/util/host_tensor.h"

using namespace cutlass;

// ---- shape table: (m, n, k) production DiT INT8 GEMMs -------------------
struct GemmShapeCfg { int m, n, k; const char* tag; };
static const GemmShapeCfg SHAPES[] = {
    {3200, 21504, 5376,  "480p fc1 (swiglu fused n=ffn*2)"},
    {3200, 5376,  7168,  "480p fc2"},
    {3200, 28672, 5376,  "480p qkv/out"},
    {3200, 5376,  14336, "480p out-adaln"},
    {4096, 21504, 5376,  "768p fc1"},
    {4096, 5376,  7168,  "768p fc2"},
    {4096, 28672, 5376,  "768p qkv/out"},
    {4096, 5376,  14336, "768p out-adaln"},
};

// ---- GEMM typedef factory --------------------------------------------------
// CUTLASS INT8 TensorOp requires A=RowMajor, B=ColumnMajor (Turing-style
// mma.sync.m16n8k32: A row-major K-contig, B col-major K-contig).
//
// Mapping from production (cublasLt view):
//   cublasLt: A=[k,n] ld=k col-major  ==  CUTLASS: A=[n,k] row-major
//   cublasLt: B=[k,m] ld=k col-major  ==  CUTLASS: B=[m,k] row-major
//   cublasLt: C=[n,m] ld=n col-major  ==  CUTLASS: C=[m,n] row-major
//
// CUTLASS convention: A [M,K] row-major, B [N,K] column-major (i.e. K-major),
// C [M,N] row-major. Mapping:
//   CUTLASS M = production n  (weight rows)
//   CUTLASS N = production m  (act rows)
//   CUTLASS K = production k
//   A_cutlass[M,K] = W[n,k] row-major
//   B_cutlass[N,K] = X[m,k] row-major  (CUTLASS stores as col-major [N,K] = row-major [K,N]... see below)
//   C_cutlass[M,N] = acc[n,m] row-major
//
// For CUTLASS INT8: B must be ColumnMajor. ColumnMajor [N,K] means stride (1, N)
// i.e. element (n,k) at offset n + k*N. Production d_x is X[m,k] row-major
// (stride (k,1), element (m,k) at m*k+k). These are the same memory when we
// identify CUTLASS N=production m, K=production k: element (N=n_c, K=k_c) at
// offset n_c + k_c*N = m + k*m = row-major [m,k]. Matches.
template <int CtaM, int CtaN, int CtaK, int WarpM, int WarpN, int WarpK,
          int Stages, int AlignA, int AlignB>
struct GemmCfg {
    using Gemm = cutlass::gemm::device::GemmUniversal<
        int8_t, cutlass::layout::RowMajor,    // A: [n,k] row-major
        int8_t, cutlass::layout::ColumnMajor, // B: [m,k] viewed as col-major [N,K] where N=m, K=k
        int32_t, cutlass::layout::RowMajor,   // C: [n,m] row-major
        int32_t,                              // ElementAccumulator
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,                  // sm_120 uses Sm80 IMMA kernels
        cutlass::gemm::GemmShape<CtaM, CtaN, CtaK>,
        cutlass::gemm::GemmShape<WarpM, WarpN, WarpK>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<
            int32_t, 128 / cutlass::sizeof_bits<int32_t>::value, int32_t, int32_t>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
        Stages,
        AlignA, AlignB,
        cutlass::arch::OpMultiplyAddSaturate,
        cutlass::ComplexTransform::kNone,
        cutlass::ComplexTransform::kNone,
        false, false, false
    >;
};

// Generic bench driver --------------------------------------------------------
template <typename Gemm>
float bench_one(const GemmShapeCfg& s, int8_t* dA, int8_t* dB, int32_t* dC,
                int iters = 100) {
    // CUTLASS: M = production n, N = production m, K = production k
    // A [M=n, K=k] row-major: leading-dim stride = k
    // B [N=m, K=k] column-major (K-major): leading-dim stride = N = m
    // C [M=n, N=m] row-major: leading-dim stride = N = m
    typename Gemm::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {s.n, s.m, s.k},                       // problem_size (M, N, K)
        /*batch_count=*/ 1,
        {(int32_t)1, (int32_t)0},              // epilogue alpha/beta
        dA, dB, dC, dC,                        // ptr_A/B/C/D
        /*batch_stride_A=*/ (int64_t)s.n * s.k,
        /*batch_stride_B=*/ (int64_t)s.m * s.k,
        /*batch_stride_C=*/ (int64_t)s.n * s.m,
        /*batch_stride_D=*/ (int64_t)s.n * s.m,
        /*stride_a=*/ (int64_t)s.k,
        /*stride_b=*/ (int64_t)s.m,
        /*stride_c=*/ (int64_t)s.m,
        /*stride_d=*/ (int64_t)s.m);
    size_t ws = Gemm::get_workspace_size(args);
    void* dws = nullptr;
    if (ws) cudaMalloc(&dws, ws);
    Gemm gemm_op;
    cutlass::Status st = gemm_op.can_implement(args);
    if (st != cutlass::Status::kSuccess) {
        if (dws) cudaFree(dws);
        return -1.f;
    }
    st = gemm_op.initialize(args, dws);
    if (st != cutlass::Status::kSuccess) {
        if (dws) cudaFree(dws);
        return -1.f;
    }
    // warmup
    for (int i = 0; i < 5; ++i) gemm_op();
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < iters; ++i) gemm_op();
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0.f; cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    if (dws) cudaFree(dws);
    return ms / iters;
}

#define TRY_CFG(CtaM,CtaN,CtaK,Wm,Wn,Wk,St,Al, tagname)                        \
    {                                                                          \
        using G = typename GemmCfg<CtaM,CtaN,CtaK,Wm,Wn,Wk,St,Al,Al>::Gemm;    \
        float ms = bench_one<G>(s, dA, dB, dC, 300);                                \
        if (ms > 0) {                                                          \
            double fl = 2.0 * (double)s.m * s.n * s.k;                         \
            double tf = fl / (ms * 1e-3) / 1e12;                               \
            printf("  cfg %-14s ms=%7.3f  TFLOPs=%7.1f\n", tagname, ms, tf);   \
            if (tf > best_tf) { best_tf = tf; best_ms = ms; best_tag=tagname; }\
        }                                                                      \
    }

int main() {
    printf("# CUTLASS sm_120 INT8 GEMM bench — 8 production shapes\n");
    printf("# A=[n,k] s8 row-major, B=[m,k] s8 row-major, C=[n,m] s32 row-major\n\n");

    // biggest allocation needed
    int8_t* dA;  int8_t* dB;  int32_t* dC;
    cudaMalloc(&dA, (size_t)28672 * 14336);
    cudaMalloc(&dB, (size_t)4096 * 14336);
    cudaMalloc(&dC, (size_t)28672 * 4096 * 4);
    cudaMemset(dA, 0x1, (size_t)28672 * 14336);
    cudaMemset(dB, 0x1, (size_t)4096 * 14336);
    cudaMemset(dC, 0, (size_t)28672 * 4096 * 4);

    // push GPU to boost clocks with a 3s GEMM burn (matches cublasLt baseline)
    printf("# pre-heating GPU clocks...\n");
    {
        using BurnGemm = typename GemmCfg<256, 128, 64, 64, 64, 64, 4, 16, 16>::Gemm;
        GemmShapeCfg big = {4096, 28672, 14336, "burn"};
        for (int i = 0; i < 3; ++i) {
            float ms = bench_one<BurnGemm>(big, dA, dB, dC, 500);
            printf("  burn iter %d ms=%.3f\n", i, ms);
        }
    }

    printf("%-4s %-6s %-6s %-6s  %-30s\n", "idx", "m", "n", "k", "tag");
    for (int i = 0; i < 8; ++i) {
        const GemmShapeCfg& s = SHAPES[i];
        printf("[%d] m=%-5d n=%-6d k=%-6d  %s\n", i, s.m, s.n, s.k, s.tag);
        double best_tf = 0; float best_ms = -1; const char* best_tag = "-";
        // Sweep: 4 configs (same set cublasLt heuristics effectively considers)
        TRY_CFG(256, 128, 64,  64, 64, 64, 4, 16, "256x128x64s4");
        TRY_CFG(128, 256, 64,  64, 64, 64, 4, 16, "128x256x64s4");
        TRY_CFG(128, 128, 64,  64, 64, 64, 4, 16, "128x128x64s4");
        TRY_CFG(128, 128, 128, 64, 64, 64, 3, 16, "128x128x128s3");
        printf("  BEST %-14s ms=%7.3f  TFLOPs=%7.1f\n\n", best_tag, best_ms, best_tf);
        fflush(stdout);
    }

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
