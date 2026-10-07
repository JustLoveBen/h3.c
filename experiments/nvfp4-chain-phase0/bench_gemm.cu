/***************************************************************************************************
 * bench_gemm.cu — P4.2 Phase-0 GEMM benchmark (design doc §9 step 2)
 *
 * Compares, on the production FC1/FC2 shapes:
 *   (a) CUTLASS example-79b-style sm_120 NVFP4-in -> NVFP4-out block-scaled
 *       GEMM with SFD generation (LinCombBlockScaleFactor epilogue,
 *       Sm1xxBlockScaledOutputConfig::tile_atom_to_shape_SFD layout).
 *       Collective/builder configuration copied faithfully from
 *       CUTLASS v4.8.0 examples/79_blackwell_geforce_gemm/
 *       79b_blackwell_geforce_nvfp4_nvfp4_gemm.cu (lines 94-180, 404-533).
 *   (b) the extracted cublasLt FP4 block-scaled GEMM (input-side only,
 *       BF16 output) exactly as configured in the h3.c repo
 *       (h3_gpu_cuda.cu:5325-5561): TN form, CUDA_R_4F_E2M1 inputs,
 *       CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3 on A and B,
 *       SCALAR_32F on C/D, BF16 output, heuristic-picked algo.
 *
 * Shapes (design doc §9): FC1  M=16000 (15973 rounded to 128), N=28672, K=5376
 *                         FC2  M=16000,                        N=5376,  K=14336
 * Output: CSV rows  impl,shape,M,N,K,ms_avg  (+ a rel_l2 correctness spot-check
 * between one CUTLASS run and one cublasLt run on identical inputs).
 *
 * Mapping for BOTH paths (same as h3.c production):
 *   A = weights [n,k] row-major FP4   (cublasLt: [k,n] col-major, OP_T)
 *   B = acts    [m,k] row-major FP4   (cublasLt: [k,m] col-major, OP_N)
 *   D = out     [m,n] row-major       (CUTLASS: FP4+SFD; cublasLt: BF16)
 *
 * Requires sm_120a (RTX 5090) and CUDA >= 12.8. Nothing here runs on a Mac.
 **************************************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublasLt.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"

#include "quant_act_extracted.cuh"

#define CK(x)                                                                  \
  do {                                                                         \
    cudaError_t e_ = (x);                                                      \
    if (e_ != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_),      \
              __FILE__, __LINE__);                                             \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CKL(x)                                                                 \
  do {                                                                         \
    cublasStatus_t s_ = (x);                                                   \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                         \
      fprintf(stderr, "cuBLASLt error %d at %s:%d\n", (int)s_, __FILE__,       \
              __LINE__);                                                       \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CUTLASS_CHECK(x)                                                       \
  do {                                                                         \
    cutlass::Status s_ = (x);                                                  \
    if (s_ != cutlass::Status::kSuccess) {                                     \
      fprintf(stderr, "CUTLASS error %d at %s:%d\n", (int)s_, __FILE__,        \
              __LINE__);                                                       \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// CUTLASS 79b-style NVFP4->NVFP4 GEMM (SFD-generating epilogue)
/// Types copied 1:1 from cutlass/examples/79_blackwell_geforce_gemm/
/// 79b_blackwell_geforce_nvfp4_nvfp4_gemm.cu (v4.8.0), lines 94-180.
/// A=RowMajor is the activation, B=ColumnMajor is the weight: exactly the
/// roles the example uses for its A/B, so no collective change is needed.
/////////////////////////////////////////////////////////////////////////////////////////////////

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>; // activation [m,k]
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;

using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>; // weight [n,k]
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;

using ElementD = cutlass::float_e2m1_t;    // chained FP4 output
using ElementSFD = cutlass::float_ue4m3_t; // NVFP4 output scale factor (vec16)
using ElementC = cutlass::bfloat16_t;      // C source (beta=0 here)
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
using LayoutSFDTag = LayoutDTag;

constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using ElementAccumulator = float;
using ElementCompute = float;
using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

using ThreadBlockShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>; // sm_120 GeForce: no TMA multicast

constexpr int InputSFVectorSize = 16;
constexpr int OutputSFVectorSize = InputSFVectorSize;

using FusionOperation = cutlass::epilogue::fusion::LinCombBlockScaleFactor<
    OutputSFVectorSize, ElementD, ElementCompute, ElementSFD, LayoutSFDTag,
    ElementC>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
    ElementAccumulator, ElementC, LayoutCTag, AlignmentC, ElementD, LayoutDTag,
    AlignmentD, cutlass::epilogue::collective::EpilogueScheduleAuto,
    FusionOperation>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB,
    LayoutBTag, AlignmentB, ElementAccumulator, ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using SfdOutputCfg = cutlass::detail::Sm1xxBlockScaledOutputConfig;
using LayoutSFD = typename SfdOutputCfg::LayoutSF;
using FusionOp = typename Gemm::EpilogueOutputOp;
constexpr bool IsBlockScaleSupported = FusionOp::IsBlockScaleSupported;
static_assert(IsBlockScaleSupported, "79b epilogue must support SFD gen");

#endif // CUTLASS_ARCH_MMA_SM120_SUPPORTED

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Common helpers
/////////////////////////////////////////////////////////////////////////////////////////////////

static void fill_uniform(float *h, size_t n, uint32_t *state, float lo,
                         float hi) {
  for (size_t i = 0; i < n; i++) {
    // xorshift32
    uint32_t x = *state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *state = x;
    h[i] = lo + (hi - lo) * ((float)(x & 0xFFFFFF) / (float)0x1000000);
  }
}

struct DeviceBuf {
  void *p = nullptr;
  size_t bytes = 0;
  DeviceBuf() = default;
  explicit DeviceBuf(size_t b) : bytes(b) { CK(cudaMalloc(&p, b)); }
  ~DeviceBuf() { if (p) cudaFree(p); }
  DeviceBuf(const DeviceBuf &) = delete;
  DeviceBuf &operator=(const DeviceBuf &) = delete;
};

static float bench_ms(std::function<void()> fn, int warmup, int iters) {
  for (int i = 0; i < warmup; i++) fn();
  CK(cudaDeviceSynchronize());
  cudaEvent_t t0, t1;
  CK(cudaEventCreate(&t0));
  CK(cudaEventCreate(&t1));
  CK(cudaEventRecord(t0));
  for (int i = 0; i < iters; i++) fn();
  CK(cudaEventRecord(t1));
  CK(cudaEventSynchronize(t1));
  float ms = 0.f;
  CK(cudaEventElapsedTime(&ms, t0, t1));
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  return ms / (float)iters;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/// cublasLt FP4 block-scaled GEMM, extracted from h3_gpu_cuda.cu:5325-5561.
/// Identical descriptor sequence: compute FP32, TN, VEC16_UE4M3 A/B scale
/// modes, SCALAR_32F C/D, SF pointers seeded before the heuristic query,
/// BF16 output.
/////////////////////////////////////////////////////////////////////////////////////////////////
struct LtFp4Gemm {
  cublasLtHandle_t lt = nullptr;
  cublasLtMatmulDesc_t desc = nullptr;
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;
  cublasLtMatmulHeuristicResult_t heur{};
  DeviceBuf workspace;
  DeviceBuf ab; // persistent device [alpha,beta], POINTER_MODE_DEVICE
  size_t m = 0, n = 0, k = 0;

  void init(size_t m_, size_t n_, size_t k_, const void *d_sf_w,
            const void *d_sf_x) {
    m = m_; n = n_; k = k_;
    workspace = DeviceBuf(32u << 20); // same order as production lt workspace
    ab = DeviceBuf(2 * sizeof(float));
    float h_ab[2] = {1.0f, 0.0f};
    CK(cudaMemcpy(ab.p, h_ab, sizeof(h_ab), cudaMemcpyHostToDevice));

    CKL(cublasLtCreate(&lt));
    cublasOperation_t transa = CUBLAS_OP_T; // weights [n,k] row -> [k,n] col
    cublasOperation_t transb = CUBLAS_OP_N; // acts    [m,k] row -> [k,m] col
    cublasLtMatmulMatrixScale_t vec16 =
        CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    cublasLtMatmulMatrixScale_t sc32f =
        CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    cublasLtPointerMode_t pmode = CUBLASLT_POINTER_MODE_DEVICE;

    CKL(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CKL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                       &transa, sizeof(transa)));
    CKL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                       &transb, sizeof(transb)));
    CKL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                       &vec16, sizeof(vec16)));
    CKL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                       &vec16, sizeof(vec16)));
    CKL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_C_SCALE_MODE,
                                       &sc32f, sizeof(sc32f)));
    CKL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_D_SCALE_MODE,
                                       &sc32f, sizeof(sc32f)));
    CKL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_POINTER_MODE,
                                       &pmode, sizeof(pmode)));
    // SF pointers must be present BEFORE the heuristic query (h3.c comment at
    // h3_gpu_cuda.cu:5436-5440: without them, returned=0 / st=7).
    CKL(cublasLtMatmulDescSetAttribute(desc,
                                       CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                       &d_sf_w, sizeof(d_sf_w)));
    CKL(cublasLtMatmulDescSetAttribute(desc,
                                       CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                       &d_sf_x, sizeof(d_sf_x)));

    CKL(cublasLtMatrixLayoutCreate(&a, CUDA_R_4F_E2M1, (uint64_t)k,
                                   (uint64_t)n, (int64_t)k));
    CKL(cublasLtMatrixLayoutCreate(&b, CUDA_R_4F_E2M1, (uint64_t)k,
                                   (uint64_t)m, (int64_t)k));
    CKL(cublasLtMatrixLayoutCreate(&c, CUDA_R_16BF, (uint64_t)n, (uint64_t)m,
                                   (int64_t)n));

    CKL(cublasLtMatmulPreferenceCreate(&pref));
    size_t ws = workspace.bytes;
    CKL(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));

    cublasLtMatmulHeuristicResult_t heurs[8];
    int returned = 0;
    CKL(cublasLtMatmulAlgoGetHeuristic(lt, desc, a, b, c, c, pref, 8, heurs,
                                       &returned));
    if (returned <= 0) {
      fprintf(stderr, "[fp4] heuristic NO-GO m=%zu n=%zu k=%zu\n", m, n, k);
      exit(1);
    }
    int pick = -1;
    for (int i = 0; i < returned; i++)
      if (heurs[i].workspaceSize <= workspace.bytes) { pick = i; break; }
    if (pick < 0) {
      fprintf(stderr, "[fp4] algo needs more workspace\n");
      exit(1);
    }
    heur = heurs[pick];
    fprintf(stderr, "[cublasLt] m=%zu n=%zu k=%zu heuristic algos=%d picked=%d\n",
            m, n, k, returned, pick);
  }

  void run(const void *d_w_fp4, const void *d_x_fp4, const void *d_sf_w,
           const void *d_sf_x, void *d_y_bf16) {
    CKL(cublasLtMatmulDescSetAttribute(desc,
                                       CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                       &d_sf_w, sizeof(d_sf_w)));
    CKL(cublasLtMatmulDescSetAttribute(desc,
                                       CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                       &d_sf_x, sizeof(d_sf_x)));
    const float *alpha = (const float *)ab.p;
    const float *beta = (const float *)ab.p + 1;
    CKL(cublasLtMatmul(lt, desc, alpha, d_w_fp4, a, d_x_fp4, b, beta,
                       d_y_bf16, c, d_y_bf16, c, &heur.algo, workspace.p,
                       heur.workspaceSize, 0));
  }

  ~LtFp4Gemm() {
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (a) cublasLtMatrixLayoutDestroy(a);
    if (b) cublasLtMatrixLayoutDestroy(b);
    if (c) cublasLtMatrixLayoutDestroy(c);
    if (desc) cublasLtMatmulDescDestroy(desc);
    if (lt) cublasLtDestroy(lt);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Dequant helpers for the correctness spot-check
/////////////////////////////////////////////////////////////////////////////////////////////////

// Device-side dequant for an [m, n] FP4+SF tensor in the h3 layout (§1.3).
__global__ void dequant_h3(const uint8_t *__restrict__ fp4,
                           const uint8_t *__restrict__ sf,
                           float *__restrict__ out, size_t m, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m * n) return;
  size_t r = i / n, c = i % n;
  uint8_t byte = fp4[r * (n / 2) + c / 2];
  uint8_t nib = (c & 1) ? (byte & 0xF) : (byte >> 4);
  float mag;
  switch (nib & 7) {
    case 0: mag = 0.f; break;
    case 1: mag = .5f; break;
    case 2: mag = 1.f; break;
    case 3: mag = 1.5f; break;
    case 4: mag = 2.f; break;
    case 5: mag = 3.f; break;
    case 6: mag = 4.f; break;
    default: mag = 6.f; break;
  }
  float dq = (nib & 8) ? -mag : mag;
  size_t ncb = (n / 16) / 4;
  size_t ncb_rd = (ncb + 31u) & ~(size_t)31u;
  size_t tile = (r / 128) * ncb_rd + (c / 16) / 4;
  uint8_t sf_b = sf[tile * 512 + (r % 128) * 4 + ((c / 16) % 4)];
  out[i] = dq * h3p0::ue4m3_to_f32(sf_b);
}

__global__ void bf16_to_f32(const __nv_bfloat16 *__restrict__ in,
                            float *__restrict__ out, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __bfloat162float(in[i]);
}

static double rel_l2(const std::vector<float> &a, const std::vector<float> &b) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < a.size(); i++) {
    double d = (double)a[i] - (double)b[i];
    num += d * d;
    den += (double)b[i] * (double)b[i];
  }
  return den > 0.0 ? sqrt(num / den) : (num > 0.0 ? INFINITY : 0.0);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
#if !defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) && !defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  fprintf(stderr, "Compiled without sm_120/121 support (CUTLASS_ARCH_MMA_SM120_SUPPORTED undefined).\n");
  return 1;
#else
  int dev = 0;
  CK(cudaGetDevice(&dev));
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, dev));
  fprintf(stderr, "GPU: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
  if (!(prop.major == 12 && (prop.minor == 0 || prop.minor == 1))) {
    fprintf(stderr, "This bench requires Blackwell GeForce (sm_120/121).\n");
    return 1;
  }

  int iters = 50;
  if (argc > 1) iters = atoi(argv[1]);

  struct Shape { const char *name; size_t m, n, k; };
  // Design doc §9: M=15973 rounded up to a 128 multiple; K is already %64==0.
  std::vector<Shape> shapes = {
      {"FC1", 16000, 28672, 5376},
      {"FC2", 16000, 5376, 14336},
  };

  printf("impl,shape,M,N,K,ms_avg\n");

  for (const auto &s : shapes) {
    size_t m = s.m, n = s.n, k = s.k;
    fprintf(stderr, "== %s M=%zu N=%zu K=%zu ==\n", s.name, m, n, k);

    // --- host init: random FP32 act + weight, quantize both with h3p0 ---
    std::vector<float> h_x(m * k), h_w(n * k);
    uint32_t st = 0x9e3779b9u;
    fill_uniform(h_x.data(), m * k, &st, -1.f, 1.f);
    fill_uniform(h_w.data(), n * k, &st, -1.f, 1.f);

    DeviceBuf d_x(m * k * 4), d_w(n * k * 4);
    CK(cudaMemcpy(d_x.p, h_x.data(), m * k * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_w.p, h_w.data(), n * k * 4, cudaMemcpyHostToDevice));

    // Note: the quantize kernel launches over m_pad = round_up(m,128) rows, so
    // the FP4 buffers must be allocated with the padded row count.
    size_t m_pad = (m + 127u) & ~(size_t)127u;
    DeviceBuf d_x_fp4(h3p0::fp4_bytes(m_pad, k)), d_w_fp4(h3p0::fp4_bytes(n, k));
    DeviceBuf d_sf_x(h3p0::sf_bytes(m, k)), d_sf_w(h3p0::sf_bytes(n, k));
    CK(cudaMemset(d_x_fp4.p, 0, h3p0::fp4_bytes(m_pad, k)));
    CK(cudaMemset(d_w_fp4.p, 0, h3p0::fp4_bytes(n, k)));
    CK(cudaMemset(d_sf_x.p, 0, h3p0::sf_bytes(m, k)));
    CK(cudaMemset(d_sf_w.p, 0, h3p0::sf_bytes(n, k)));
    CK(h3p0::quantize_act((const float *)d_x.p, (uint8_t *)d_x_fp4.p,
                          (uint8_t *)d_sf_x.p, m, k));
    CK(h3p0::quantize_act((const float *)d_w.p, (uint8_t *)d_w_fp4.p,
                          (uint8_t *)d_sf_w.p, n, k));
    CK(cudaDeviceSynchronize());

    // ---------------- CUTLASS 79b path ----------------
    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {(int)m, (int)k, 1});
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {(int)n, (int)k, 1});
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {(int)m, (int)n, 1});
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {(int)m, (int)n, 1});

    using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape((int)m, (int)n, (int)k, 1));
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape((int)m, (int)n, (int)k, 1));
    auto layout_SFD = SfdOutputCfg::tile_atom_to_shape_SFD(cute::make_shape((int)m, (int)n, (int)k, 1));
    size_t sfd_bytes = (size_t)cute::cosize(layout_SFD); // sf elems; UE4M3 = 1B
    fprintf(stderr, "[cutlass] layout_SFD cosize=%zu h3 sf_bytes(m,n)=%zu\n",
            sfd_bytes, h3p0::sf_bytes(m, n));

    DeviceBuf d_d_fp4(m * (n / 2)), d_sfd(sfd_bytes);
    DeviceBuf d_norm(4);
    // norm_constant=1: output must be exactly alpha*acc+beta*C in value so the
    // spot-check against cublasLt compares the same math (79b uses 2 as a demo;
    // that version dequantizes to 2x, which would pollute rel_l2).
    float h_norm = 1.0f;
    CK(cudaMemcpy(d_norm.p, &h_norm, 4, cudaMemcpyHostToDevice));

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {(int)m, (int)n, (int)k, 1},
        {cute::recast_ptr<typename ElementA::DataType>(d_x_fp4.p), stride_A,
         cute::recast_ptr<typename ElementB::DataType>(d_w_fp4.p), stride_B,
         cute::recast_ptr<typename ElementA::ScaleFactorType>(d_sf_x.p), layout_SFA,
         cute::recast_ptr<typename ElementB::ScaleFactorType>(d_sf_w.p), layout_SFB},
        {{1.0f, 0.0f},
         nullptr, stride_C,
         cute::recast_ptr<ElementD>(d_d_fp4.p), stride_D}};
    arguments.epilogue.thread.block_scale_factor_ptr =
        cute::recast_ptr<ElementSFD>(d_sfd.p);
    arguments.epilogue.thread.norm_constant_ptr = (const float *)d_norm.p;

    Gemm gemm;
    size_t ws_size = Gemm::get_workspace_size(arguments);
    DeviceBuf workspace(ws_size ? ws_size : 16);
    CUTLASS_CHECK(gemm.can_implement(arguments));
    CUTLASS_CHECK(gemm.initialize(arguments, workspace.p));

    double cutlass_ms = bench_ms([&] { CUTLASS_CHECK(gemm.run()); }, 3, iters);
    printf("cutlass79b,%s,%zu,%zu,%zu,%.4f\n", s.name, m, n, k, cutlass_ms);
    fflush(stdout);

    // ---------------- cublasLt path ----------------
    DeviceBuf d_y_bf16(m * n * sizeof(__nv_bfloat16));
    LtFp4Gemm lt;
    lt.init(m, n, k, d_sf_w.p, d_sf_x.p);
    double lt_ms = bench_ms(
        [&] {
          lt.run(d_w_fp4.p, d_x_fp4.p, d_sf_w.p, d_sf_x.p, d_y_bf16.p);
        },
        3, iters);
    printf("cublaslt_fp4,%s,%zu,%zu,%zu,%.4f\n", s.name, m, n, k, lt_ms);
    fflush(stdout);

    // ---------------- correctness spot-check ----------------
    // Same inputs, same math modulo epilogue: dequant both outputs to FP32.
    DeviceBuf d_out_cut(m * n * 4), d_out_lt(m * n * 4);
    {
      size_t tot = m * n;
      dequant_h3<<<(unsigned)((tot + 255) / 256), 256>>>(
          (const uint8_t *)d_d_fp4.p, (const uint8_t *)d_sfd.p,
          (float *)d_out_cut.p, m, n);
      bf16_to_f32<<<(unsigned)((tot + 255) / 256), 256>>>(
          (const __nv_bfloat16 *)d_y_bf16.p, (float *)d_out_lt.p, tot);
      CK(cudaGetLastError());
      CK(cudaDeviceSynchronize());
    }
    std::vector<float> h_cut(m * n), h_lt(m * n);
    CK(cudaMemcpy(h_cut.data(), d_out_cut.p, m * n * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_lt.data(), d_out_lt.p, m * n * 4, cudaMemcpyDeviceToHost));
    double rl2 = rel_l2(h_cut, h_lt);
    printf("spotcheck,%s,%zu,%zu,%zu,rel_l2=%.6f\n", s.name, m, n, k, rl2);
    fprintf(stderr, "[spotcheck] %s rel_l2(cutlass_dequant*norm, lt_bf16) = %.6f\n",
            s.name, rl2);
  }
  return 0;
#endif
}
