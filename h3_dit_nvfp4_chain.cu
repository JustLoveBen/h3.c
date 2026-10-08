/***************************************************************************************************
 * h3_dit_nvfp4_chain.cu — P4.2 Phase-1 FC1→FC2 chained direct-output (sm_120a)
 *
 * Design: docs/research/2026-10-08-nvfp4-chained-direct-output.md (§9 Phase-1
 * "方案 B"). Goal: eliminate the activation-quantization transit between the
 * DiT MLP FC1 and FC2 GEMMs.
 *
 *   Standard (P4.1):  FC1 GEMM -> BF16 -> SwiGLU -> FP32 -> quantize -> FP4 -> FC2 GEMM
 *   Chained (Phase-1): FC1 GEMM -> FP32 acc -> [swiglu+requant] -> FP4+SF -> FC2 GEMM
 *
 * This unit carries the parts that are self-contained (do NOT touch the static
 * helpers inside h3_gpu_cuda.cu):
 *
 *   1. h3_cutlass_nvfp4_fc1_f32_run  — CUTLASS 79b-style sm_120 block-scaled
 *      NVFP4 GEMM, FP32 accumulator DIRECT output (ElementD = float, no BF16
 *      rounding, no SFD generation — Phase-1 keeps the acc in FP32; Phase-2
 *      flips ElementD to FP4+SFD per §9 Phase-2 方案 A). Writes acc[m, 2*ffn].
 *
 *   2. h3_nvfp4_swiglu_requant_dev    — fused SwiGLU + NVFP4 requantize kernel.
 *      Reads FP32 acc[m, 2*ffn], computes silu(g)*u -> [m, ffn], and emits
 *      packed E2M1 + 128x4-swizzled UE4M3 SF in EXACTLY the layout that
 *      h3_cuda_quantize_act_nvfp4_kernel produces (§1.3), so the downstream
 *      FC2 GEMM (cublasLt or CUTLASS) consumes it with zero reformatting.
 *
 * The FC2 pre-quantized GEMM entry (h3_cuda_gemm_fp4_pre_q_dev) lives in
 * h3_gpu_cuda.cu because it reuses the static BF16 gate epilogue + cublasLt
 * path there; this unit only PRODUCES the FP4+SF activation for it.
 *
 * Requires sm_120a (RTX 5090) and CUTLASS >= 4.x headers. Compiled as a
 * separate library (H3_ENABLE_NVFP4_CHAIN) so the main translation units stay
 * free of the slow CUTLASS headers and the sm_120a arch requirement.
 **************************************************************************************************/

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>

#include "h3_gpu_cuda.h"

#if defined(H3_ENABLE_NVFP4_CHAIN)

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
#include "cutlass/util/packed_stride.hpp"

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
#define H3_CHAIN_SM120 1
#else
#define H3_CHAIN_SM120 0
#endif

/* ------------------------------------------------------------------------- */
/* Error plumbing (chain-local; mirrors the h3_cuda_set_error convention)     */
/* ------------------------------------------------------------------------- */
static void chain_set_error(char *error, size_t error_size, const char *msg) {
  if (error && error_size) {
    snprintf(error, error_size, "%s", msg ? msg : "nvfp4 chain error");
  }
}
static void chain_set_error_cuda(char *error, size_t error_size,
                                 const char *what, cudaError_t st) {
  if (error && error_size) {
    snprintf(error, error_size, "%s: %s", what, cudaGetErrorString(st));
  }
}

/* ------------------------------------------------------------------------- */
/* FP4 (E2M1) / UE4M3 conversion — device copies of the h3_gpu_cuda.cu statics. */
/* These MUST match h3_gpu_cuda.cu:4338-4400 bit-for-bit so the SFD/FP4 we emit */
/* is consumable by the existing FC2 cublasLt path.                            */
/* ------------------------------------------------------------------------- */
__device__ __forceinline__ float chain_e2m1_round(float q) {
  /* round-to-nearest-even on the E2M1 grid {0,.5,1,1.5,2,3,4,6} */
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
__device__ __forceinline__ uint8_t chain_e2m1_nibble(float q) {
  float v = chain_e2m1_round(q);
  float a = fabsf(v);
  uint8_t mag;
  if (a < 0.25f) mag = 0;
  else if (a < 0.75f) mag = 1;
  else if (a < 1.25f) mag = 2;
  else if (a < 1.75f) mag = 3;
  else if (a < 2.5f) mag = 4;
  else if (a < 3.5f) mag = 5;
  else if (a < 5.0f) mag = 6;
  else mag = 7;
  return (uint8_t)((v < 0.0f ? 8 : 0) | mag);
}
__device__ __forceinline__ uint8_t chain_f32_to_ue4m3(float v) {
  int e, m, E;
  float f;
  if (!(v > 0.0f)) return 0;
  f = frexpf(v, &e);
  E = e - 1 + 7;
  if (E <= 0) {
    m = (int)lroundf(ldexpf(v, 6) * 8.0f);
    if (m > 7) m = 7;
    return (uint8_t)m;
  }
  if (E >= 15) return 0x7E;
  m = (int)lroundf((f * 2.0f - 1.0f) * 8.0f);
  if (m == 8) { E += 1; m = 0; if (E >= 15) return 0x7E; }
  return (uint8_t)((E << 3) | m);
}
__device__ __forceinline__ float chain_ue4m3_to_f32(uint8_t b) {
  int e = (b >> 3) & 0xF, m = b & 0x7;
  if (e == 0) return ldexpf((float)m / 8.0f, -6);
  if (e == 0xF && m == 0x7) return NAN;
  return ldexpf(1.0f + (float)m / 8.0f, e - 7);
}

/* ------------------------------------------------------------------------- */
/* CUTLASS 79b-style block-scaled NVFP4 GEMM, FP32 direct-output.             */
/* A = activation [m,k] row-major FP4; B = weight [n,k] col-major FP4.        */
/* ElementD = float (Phase-1: acc stays FP32; Phase-2 flips to FP4+SFD).      */
/* ------------------------------------------------------------------------- */
#if H3_CHAIN_SM120
using ChainElemA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ChainLayoutA = cutlass::layout::RowMajor;
using ChainElemB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ChainLayoutB = cutlass::layout::ColumnMajor;
constexpr int ChainAlignAB = 32;

using ChainElemD = float;                    /* FP32 acc direct output */
using ChainLayoutD = cutlass::layout::RowMajor;
constexpr int ChainAlignD = 128 / cutlass::sizeof_bits<ChainElemD>::value;

using ChainElemC = float;                    /* unused (beta=0) */
using ChainLayoutC = cutlass::layout::RowMajor;
constexpr int ChainAlignC = 128 / cutlass::sizeof_bits<ChainElemC>::value;

using ChainAccum = float;
using ChainCompute = float;
using ChainArch = cutlass::arch::Sm120;
using ChainOpClass = cutlass::arch::OpClassBlockScaledTensorOp;

using ChainTB = cute::Shape<cute::_128, cute::_128, cute::_128>;
using ChainCluster = cute::Shape<cute::_1, cute::_1, cute::_1>;

/* Plain LinearCombination epilogue (alpha*acc + beta*C -> FP32 D), no SFD. */
using ChainEpiFusion = cutlass::epilogue::fusion::LinearCombination<
    ChainElemD, ChainCompute, ChainElemC, ChainCompute>;

using ChainEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ChainArch, ChainOpClass, ChainTB, ChainCluster,
    cutlass::epilogue::collective::EpilogueTileAuto, ChainAccum, ChainCompute,
    ChainElemC, ChainLayoutC, ChainAlignC, ChainElemD, ChainLayoutD, ChainAlignD,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    ChainEpiFusion>::CollectiveOp;

using ChainMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ChainArch, ChainOpClass, ChainElemA, ChainLayoutA, ChainAlignAB,
    ChainElemB, ChainLayoutB, ChainAlignAB, ChainAccum, ChainTB, ChainCluster,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename ChainEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;

using ChainKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>, ChainMainloop, ChainEpilogue, void>;
using ChainGemm = cutlass::gemm::device::GemmUniversalAdapter<ChainKernel>;

using ChainStrideA = typename ChainGemm::GemmKernel::StrideA;
using ChainStrideB = typename ChainGemm::GemmKernel::StrideB;
using ChainStrideC = typename ChainGemm::GemmKernel::StrideC;
using ChainStrideD = typename ChainGemm::GemmKernel::StrideD;
using ChainLayoutSFA = typename ChainGemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using ChainLayoutSFB = typename ChainGemm::GemmKernel::CollectiveMainloop::LayoutSFB;
#endif /* H3_CHAIN_SM120 */

/* ------------------------------------------------------------------------- */
/* Fused SwiGLU + NVFP4 requantize.                                           */
/*   acc[m, 2*ffn] (FP32, FC1 output, gate=[:, :ffn], up=[:, ffn:])           */
/*   -> act[m, ffn] = silu(gate)*up  (optionally +lora on gate/up)            */
/*   -> quantize each 16-wide micro-block: E2M1 + UE4M3 SF                    */
/* Output layout matches h3_cuda_quantize_act_nvfp4_kernel (§1.3) exactly.    */
/* Grid: one warp per (row, 16-micro-block group); mirrors quantize_act.      */
/* ------------------------------------------------------------------------- */
__global__ void h3_chain_swiglu_requant_kernel(
    const float *__restrict__ acc,   /* [m, 2*ffn] FP32 */
    float acc_scale,                 /* weight->scale2 * act_scale for FC1 */
    const float *__restrict__ lora,  /* [m, 2*ffn] or NULL */
    uint8_t *__restrict__ act_fp4,   /* [m, ffn/2] packed */
    uint8_t *__restrict__ sf,        /* swizzled UE4M3 [., ffn/16] */
    size_t m, size_t ffn) {
  const size_t row = (size_t)blockIdx.x + (size_t)blockIdx.y * gridDim.x;
  const int tid = (int)threadIdx.x;
  const int mb = tid >> 4;             /* micro-block index within block */
  const int lane_in_mb = tid & 15;
  const size_t nc = ffn >> 4;          /* ffn/16 */
  const int mb_per_block = 16;         /* 256 threads / 16 lanes */
  const size_t ncb = nc >> 2;          /* ffn/64; assumes ffn%64==0 */
  const size_t ncb_rd = (ncb + 31u) & ~(size_t)31u;
  const size_t tile_base = (row >> 7) * ncb_rd;
  const size_t r128 = row & 127u;
  if (row >= m) {
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
    const float *g_row = acc + row * (2u * ffn);        /* gate half */
    const float *u_row = g_row + ffn;                   /* up half */
    uint8_t *ar_fp4 = act_fp4 + row * (ffn >> 1);
    int nblk = (int)(nc / mb_per_block);
    int blk;
    for (blk = 0; blk < nblk; blk++) {
      int g = blk * mb_per_block + mb;         /* global micro-block in row */
      size_t col = (size_t)g * 16 + lane_in_mb;
      float gate = acc_scale * g_row[col];
      float up = acc_scale * u_row[col];
      if (lora) {
        gate += lora[row * (2u * ffn) + col];
        up += lora[row * (2u * ffn) + ffn + col];
      }
      float v = (gate / (1.0f + expf(-gate))) * up;   /* silu(g)*u */
      float amax = fabsf(v);
      float scale_f, inv;
      uint8_t sf_b, nib;
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 8));
      scale_f = (amax > 0.0f) ? (amax / 6.0f) : 1.0f;
      sf_b = chain_f32_to_ue4m3(scale_f);
      scale_f = chain_ue4m3_to_f32(sf_b);
      inv = 1.0f / scale_f;
      nib = chain_e2m1_nibble(v * inv);
      if (lane_in_mb == 0) {
        size_t cc = (size_t)g & 3u;
        size_t off_sf = (tile_base + ((size_t)g >> 2)) * 512 + r128 * 4 + cc;
        sf[off_sf] = sf_b;
      }
      {
        uint8_t nib_lo = (uint8_t)__shfl_down_sync(0xffffffffu, nib, 1);
        if ((lane_in_mb & 1) == 0) {
          ar_fp4[col >> 1] = (uint8_t)((nib << 4) | (nib_lo & 0xF));
        }
      }
    }
    /* K-tail zero-fill: tile-cols [ncb, ncb_rd) for ffn%2048!=0. */
    if (lane_in_mb == 0 && ncb_rd > ncb) {
      size_t tc;
      int cc;
      for (tc = ncb + (size_t)(tid >> 4); tc < ncb_rd; tc += 16u)
        for (cc = 0; cc < 4; cc++)
          sf[(tile_base + tc) * 512 + r128 * 4 + (size_t)cc] = 0;
    }
  }
}

/* Host wrapper: SwiGLU + requant. ffn%64==0 required (14336 OK). */
int h3_nvfp4_swiglu_requant_dev(h3_cuda *cuda, const float *d_acc_f32,
                                float acc_scale, const float *d_lora,
                                size_t m, size_t ffn, uint8_t *d_act_fp4,
                                uint8_t *d_sf_act, char *error,
                                size_t error_size) {
  cudaStream_t stream;
  size_t m_pad, sf_ncb, sf_rb, sf_bytes;
  dim3 grid, threads;
  if (!cuda || !d_acc_f32 || !d_act_fp4 || !d_sf_act || m == 0 ||
      (ffn & 63u) != 0) {
    chain_set_error(error, error_size,
                    "swiglu_requant: invalid args (ffn%64!=0)");
    return 0;
  }
  stream = (cudaStream_t)h3_cuda_op_stream(cuda);
  m_pad = (m + 127u) & ~(size_t)127u;
  sf_ncb = (((ffn / 16u) + 3u) / 4u + 31u) & ~(size_t)31u;
  sf_rb = m_pad / 128u;
  sf_bytes = sf_rb * sf_ncb * 512u;
  /* Zero the whole SF buffer first (padding rows + K-tail read as 0, not
   * cudaMalloc garbage >= 0xFE which is UE4M3 NaN — same rationale as the
   * quantize path in h3_gpu_cuda.cu). */
  cudaMemsetAsync(d_sf_act, 0, sf_bytes, stream);
  threads = dim3(256, 1, 1);
  /* grid.x*grid.y covers m_pad rows (cap grid.y at 65535). */
  grid = dim3((unsigned)(m_pad > 65535u ? 65535u : m_pad),
              (unsigned)((m_pad + 65534u) / 65535u), 1);
  h3_chain_swiglu_requant_kernel<<<grid, threads, 0, stream>>>(
      d_acc_f32, acc_scale, d_lora, d_act_fp4, d_sf_act, m, ffn);
  {
    cudaError_t st = cudaGetLastError();
    if (st != cudaSuccess) {
      chain_set_error_cuda(error, error_size, "swiglu_requant kernel", st);
      return 0;
    }
  }
  return 1;
}

/* ------------------------------------------------------------------------- */
/* CUTLASS FC1 GEMM host wrapper: FP32 acc direct output.                     */
/* ------------------------------------------------------------------------- */
int h3_cutlass_nvfp4_fc1_f32_run(h3_cuda *cuda,
                                 const uint8_t *d_w_fp4,
                                 const uint8_t *d_sf_w,
                                 const uint8_t *d_x_fp4,
                                 const uint8_t *d_sf_x, size_t m, size_t n,
                                 size_t k, float *d_acc_f32, double *gemm_ms,
                                 char *error, size_t error_size) {
#if !H3_CHAIN_SM120
  (void)cuda; (void)d_w_fp4; (void)d_sf_w; (void)d_x_fp4; (void)d_sf_x;
  (void)m; (void)n; (void)k; (void)d_acc_f32; (void)gemm_ms;
  chain_set_error(error, error_size,
                  "nvfp4 chain built without sm_120a support");
  return 0;
#else
  cudaStream_t stream;
  cudaEvent_t ev0 = NULL, ev1 = NULL;
  int want_ms;
  if (!cuda || !d_w_fp4 || !d_sf_w || !d_x_fp4 || !d_sf_x || !d_acc_f32 ||
      m == 0 || n == 0 || k == 0 || (k & 63u) != 0) {
    chain_set_error(error, error_size, "cutlass fc1: invalid args (K%64!=0)");
    return 0;
  }
  if (gemm_ms) *gemm_ms = 0.0;
  stream = (cudaStream_t)h3_cuda_op_stream(cuda);

  ChainStrideA stride_A =
      cutlass::make_cute_packed_stride(ChainStrideA{}, {(int)m, (int)k, 1});
  ChainStrideB stride_B =
      cutlass::make_cute_packed_stride(ChainStrideB{}, {(int)n, (int)k, 1});
  ChainStrideC stride_C =
      cutlass::make_cute_packed_stride(ChainStrideC{}, {(int)m, (int)n, 1});
  ChainStrideD stride_D =
      cutlass::make_cute_packed_stride(ChainStrideD{}, {(int)m, (int)n, 1});

  using BlkCfg = typename ChainGemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  auto layout_SFA =
      BlkCfg::tile_atom_to_shape_SFA(cute::make_shape((int)m, (int)n, (int)k, 1));
  auto layout_SFB =
      BlkCfg::tile_atom_to_shape_SFB(cute::make_shape((int)m, (int)n, (int)k, 1));

  typename ChainGemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {(int)m, (int)n, (int)k, 1},
      {reinterpret_cast<cutlass::float_e2m1_t const *>(d_x_fp4), stride_A,
       reinterpret_cast<cutlass::float_e2m1_t const *>(d_w_fp4), stride_B,
       reinterpret_cast<cutlass::float_ue4m3_t const *>(d_sf_x), layout_SFA,
       reinterpret_cast<cutlass::float_ue4m3_t const *>(d_sf_w), layout_SFB},
      {{1.0f, 0.0f},
       nullptr, stride_C,
       reinterpret_cast<ChainElemD *>(d_acc_f32), stride_D}};

  ChainGemm gemm;
  size_t ws_size = ChainGemm::get_workspace_size(arguments);
  void *workspace = NULL;
  if (ws_size) {
    if (cudaMalloc(&workspace, ws_size) != cudaSuccess) {
      chain_set_error(error, error_size, "cutlass fc1: workspace OOM");
      return 0;
    }
  }
  cutlass::Status cs = gemm.can_implement(arguments);
  if (cs != cutlass::Status::kSuccess) {
    if (workspace) cudaFree(workspace);
    chain_set_error(error, error_size, "cutlass fc1: can_implement failed");
    return 0;
  }
  cs = gemm.initialize(arguments, workspace);
  if (cs != cutlass::Status::kSuccess) {
    if (workspace) cudaFree(workspace);
    chain_set_error(error, error_size, "cutlass fc1: initialize failed");
    return 0;
  }

  want_ms = (gemm_ms != NULL);
  if (want_ms) {
    if (cudaEventCreate(&ev0) != cudaSuccess ||
        cudaEventCreate(&ev1) != cudaSuccess) {
      if (ev0) cudaEventDestroy(ev0);
      if (ev1) cudaEventDestroy(ev1);
      if (workspace) cudaFree(workspace);
      chain_set_error(error, error_size, "cutlass fc1: event create");
      return 0;
    }
    cudaEventRecord(ev0, stream);
  }

  cs = gemm.run();
  cudaError_t lst = cudaGetLastError();

  if (want_ms) {
    float el = 0.0f;
    cudaEventRecord(ev1, stream);
    cudaEventSynchronize(ev1);
    cudaEventElapsedTime(&el, ev0, ev1);
    *gemm_ms = (double)el;
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
  }
  if (workspace) {
    cudaStreamSynchronize(stream);
    cudaFree(workspace);
  }

  if (cs != cutlass::Status::kSuccess) {
    chain_set_error(error, error_size, "cutlass fc1: run failed");
    return 0;
  }
  if (lst != cudaSuccess) {
    chain_set_error_cuda(error, error_size, "cutlass fc1 kernel", lst);
    return 0;
  }
  return 1;
#endif /* H3_CHAIN_SM120 */
}

#else /* !H3_ENABLE_NVFP4_CHAIN */

/* Stub translations so the library links when the chain is disabled. */
int h3_nvfp4_swiglu_requant_dev(h3_cuda *cuda, const float *d_acc_f32,
                                float acc_scale, const float *d_lora,
                                size_t m, size_t ffn, uint8_t *d_act_fp4,
                                uint8_t *d_sf_act, char *error,
                                size_t error_size) {
  (void)cuda; (void)d_acc_f32; (void)acc_scale; (void)d_lora; (void)m;
  (void)ffn; (void)d_act_fp4; (void)d_sf_act;
  if (error && error_size)
    snprintf(error, error_size, "nvfp4 chain disabled at build time");
  return 0;
}
int h3_cutlass_nvfp4_fc1_f32_run(h3_cuda *cuda, const uint8_t *d_w_fp4,
                                 const uint8_t *d_sf_w, const uint8_t *d_x_fp4,
                                 const uint8_t *d_sf_x, size_t m, size_t n,
                                 size_t k, float *d_acc_f32, double *gemm_ms,
                                 char *error, size_t error_size) {
  (void)cuda; (void)d_w_fp4; (void)d_sf_w; (void)d_x_fp4; (void)d_sf_x;
  (void)m; (void)n; (void)k; (void)d_acc_f32; (void)gemm_ms;
  if (error && error_size)
    snprintf(error, error_size, "nvfp4 chain disabled at build time");
  return 0;
}

#endif /* H3_ENABLE_NVFP4_CHAIN */
