/***************************************************************************************************
 * sfd_compare.cu — P4.2 Phase-0 SFD layout gate (design doc §9 step 3)
 *
 * Byte-exact comparison of the output scale-factor (SFD) buffer produced by
 *   (a) the CUTLASS 79b-style sm_120 NVFP4->NVFP4 epilogue
 *       (LinCombBlockScaleFactor + Sm1xxBlockScaledOutputConfig::
 *        tile_atom_to_shape_SFD layout), vs
 *   (b) the h3.c production quantize kernel h3p0::quantize_act
 *       (extracted verbatim into quant_act_extracted.cuh), which is the exact
 *       layout the cublasLt FP4 input path consumes downstream.
 *
 * To make the two SFD producers compute on the same data, A and B of the
 * GEMM are set to a per-K-column identity: exactly one micro-block per K
 * column pair carries value 1.0 (SF=0x38, nibble 2) and the rest are zero
 * (SF=0x00). Then acc[m,n] == x[m,n] exactly in FP32, and the 79b epilogue
 * quantizes acc with norm_constant=1 -> D/SFD must equal quantize_act(x).
 *
 *   A = act [m,k] row-major, B = weight [n,k] col-major (== [k,n] row-major)
 *   A[m,kk]  = 1 iff kk == m % k          (m > k rows re-test columns)
 *   B[kk,n]  = 1 iff kk == n % k          (n > k cols re-test, e.g. FC1)
 *
 * Both operands are quantized through h3p0::quantize_act itself, so SFA/SFB
 * are guaranteed to be in a layout the CUTLASS mainloop accepts.
 *
 * Shapes tested (design doc §1.3/§9 — K must be a multiple of 64):
 *   FC1 out:      M=16000 (128-multiple), N=28672, K=5376
 *   FC2 out:      M=16000,                N=5376,  K=14336
 *   FC1 ragged M: M=15973 (padding rows must carry SF=0), N=28672, K=5376
 *   K-tail:       M=384,  N=4160 (ncb=65 -> padded extent 96 tile-cols), K=5376
 *
 * Requires 100% byte match over the full padded SF extent -> PASS/FAIL.
 * Mismatches are decoded into (tile_row, tile_col, r128, cc) per the §1.3
 * layout formula  off = (tile_row*ncb_rd + tile_col)*512 + r128*4 + cc.
 **************************************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#include <cuda_runtime.h>

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
#include "cutlass/util/host_tensor.h"

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

// Same type configuration as bench_gemm.cu / CUTLASS example 79b (v4.8.0).
using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;
using ElementD = cutlass::float_e2m1_t;
using ElementSFD = cutlass::float_ue4m3_t;
using ElementC = cutlass::bfloat16_t;
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
using ClusterShape = Shape<_1, _1, _1>;
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
using SfdOutputCfg = cutlass::detail::Sm1xxBlockScaledOutputConfig<OutputSFVectorSize>;
using FusionOp = typename Gemm::EpilogueOutputOp;
constexpr bool IsBlockScaleSupported = FusionOp::IsBlockScaleSupported;
static_assert(IsBlockScaleSupported, "79b epilogue must support SFD gen");

struct DeviceBuf {
  void *p = nullptr;
  size_t bytes = 0;
  DeviceBuf() = default;
  explicit DeviceBuf(size_t b) : bytes(b) { CK(cudaMalloc(&p, b)); }
  ~DeviceBuf() { if (p) cudaFree(p); }
  DeviceBuf(const DeviceBuf &) = delete;
  DeviceBuf &operator=(const DeviceBuf &) = delete;
  DeviceBuf(DeviceBuf &&o) noexcept : p(o.p), bytes(o.bytes) { o.p = nullptr; o.bytes = 0; }
  DeviceBuf &operator=(DeviceBuf &&o) noexcept {
    if (this != &o) { if (p) cudaFree(p); p = o.p; bytes = o.bytes; o.p = nullptr; o.bytes = 0; }
    return *this;
  }
};

static void fill_uniform(float *h, size_t n, uint32_t *state, float lo,
                         float hi) {
  for (size_t i = 0; i < n; i++) {
    uint32_t x = *state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *state = x;
    h[i] = lo + (hi - lo) * ((float)(x & 0xFFFFFF) / (float)0x1000000);
  }
}

// Quantize a host FP32 [rows, k] tensor to device FP4+SF via h3p0.
// The kernel launches over round_up(rows,128), so pad the FP4 buffer.
static void quantize_to_device(const std::vector<float> &h, size_t rows,
                               size_t k, DeviceBuf &d_fp4, DeviceBuf &d_sf) {
  DeviceBuf d_x(rows * k * 4);
  CK(cudaMemcpy(d_x.p, h.data(), rows * k * 4, cudaMemcpyHostToDevice));
  size_t rows_pad = (rows + 127u) & ~(size_t)127u;
  d_fp4 = DeviceBuf(h3p0::fp4_bytes(rows_pad, k));
  d_sf = DeviceBuf(h3p0::sf_bytes(rows, k));
  CK(cudaMemset(d_fp4.p, 0, h3p0::fp4_bytes(rows_pad, k)));
  CK(cudaMemset(d_sf.p, 0, h3p0::sf_bytes(rows, k)));
  CK(h3p0::quantize_act((const float *)d_x.p, (uint8_t *)d_fp4.p,
                        (uint8_t *)d_sf.p, rows, k));
  CK(cudaDeviceSynchronize());
}

struct ShapeCfg {
  const char *name;
  size_t m, n, k;
};

static bool run_shape(const ShapeCfg &s) {
  size_t m = s.m, n = s.n, k = s.k;
  fprintf(stderr, "== %s M=%zu N=%zu K=%zu ==\n", s.name, m, n, k);

  // Activation: random.
  std::vector<float> h_x(m * k);
  uint32_t st = 0x1234abcd ^ (uint32_t)(m * 1315423911u) ^ (uint32_t)n;
  fill_uniform(h_x.data(), m * k, &st, -2.f, 2.f);

  // Weight: per-K-column identity, B[kk, jj] = 1 iff kk == jj % k.
  std::vector<float> h_w(n * k, 0.0f);
  for (size_t jj = 0; jj < n; jj++) h_w[jj * k + (jj % k)] = 1.0f;
  // Activation rows: A[i, kk] = 1 iff kk == i % k would destroy randomness;
  // instead the identity lives entirely in B, so acc[i,jj] = x[i, jj%k] and
  // for jj >= k the epilogue quantizes a repeat of an earlier value: both SFD
  // producers still see identical data per (row, micro-block), so the byte
  // comparison remains valid for every output micro-block.

  DeviceBuf d_x_fp4, d_sf_x, d_w_fp4, d_sf_w;
  quantize_to_device(h_x, m, k, d_x_fp4, d_sf_x);
  quantize_to_device(h_w, n, k, d_w_fp4, d_sf_w);

  // Reference SFD: quantize act over the OUTPUT extent [m, n] built by
  // expanding acc[i,jj] = x[i, jj%k] — only needed when n > k (FC1). When
  // n <= k the acc is just x[:, :n] which is NOT what quantize_act(x[m,n])
  // sees... so we always materialize acc explicitly for the reference.
  std::vector<float> h_acc(m * n);
  for (size_t i = 0; i < m; i++)
    for (size_t jj = 0; jj < n; jj++) h_acc[i * n + jj] = h_x[i * k + (jj % k)];
  DeviceBuf d_ref_fp4, d_ref_sf;
  quantize_to_device(h_acc, m, n, d_ref_fp4, d_ref_sf);

  // CUTLASS 79b GEMM: A=act [m,k] row, B=weight [n,k] col, D=[m,n] FP4+SFD.
  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {(int)m, (int)k, 1});
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {(int)n, (int)k, 1});
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {(int)m, (int)n, 1});
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {(int)m, (int)n, 1});

  using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape((int)m, (int)n, (int)k, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape((int)m, (int)n, (int)k, 1));
  auto layout_SFD = SfdOutputCfg::tile_atom_to_shape_SFD(cute::make_shape((int)m, (int)n, (int)k, 1));
  size_t sfd_bytes = (size_t)cute::cosize(layout_SFD);
  size_t ref_bytes = h3p0::sf_bytes(m, n);
  fprintf(stderr, "  SFD extent: cutlass cosize=%zu h3 sf_bytes=%zu %s\n",
          sfd_bytes, ref_bytes, sfd_bytes == ref_bytes ? "(match)" : "(MISMATCH!)");

  // Inputs are already h3p0-quantized in DeviceBuf (d_x_fp4/d_w_fp4/d_sf_*),
  // whose layout matches the CUTLASS SF swizzle (design doc §1.3). We pass
  // them directly via reinterpret_cast to the CUTLASS element pointer types —
  // exactly what HostTensor::device_data() does internally.
  size_t m_pad = (m + 127u) & ~(size_t)127u;
  (void)m_pad;
  DeviceBuf d_d_fp4(m * (n / 2)), d_sfd(sfd_bytes), d_norm(sizeof(float));
  CK(cudaMemset(d_d_fp4.p, 0xEE, m * (n / 2))); // poison: detect unwritten bytes
  CK(cudaMemset(d_sfd.p, 0xEE, sfd_bytes));
  float h_norm = 1.0f;
  CK(cudaMemcpy(d_norm.p, &h_norm, sizeof(float), cudaMemcpyHostToDevice));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {(int)m, (int)n, (int)k, 1},
      {reinterpret_cast<cutlass::float_e2m1_t const *>(d_x_fp4.p), stride_A,
       reinterpret_cast<cutlass::float_e2m1_t const *>(d_w_fp4.p), stride_B,
       reinterpret_cast<cutlass::float_ue4m3_t const *>(d_sf_x.p), layout_SFA,
       reinterpret_cast<cutlass::float_ue4m3_t const *>(d_sf_w.p), layout_SFB},
      {{1.0f, 0.0f},
       nullptr, stride_C,
       reinterpret_cast<cutlass::float_e2m1_t *>(d_d_fp4.p), stride_D}};
  if constexpr (IsBlockScaleSupported) {
    arguments.epilogue.thread.block_scale_factor_ptr =
        reinterpret_cast<cutlass::float_ue4m3_t *>(d_sfd.p);
    arguments.epilogue.thread.norm_constant_ptr =
        reinterpret_cast<float const *>(d_norm.p);
  }

  Gemm gemm;
  size_t ws_size = Gemm::get_workspace_size(arguments);
  DeviceBuf workspace(ws_size ? ws_size : 16);
  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.p));
  CUTLASS_CHECK(gemm.run());
  CK(cudaDeviceSynchronize());

  // Byte-compare SFD over the common extent (should be identical sizes).
  size_t cmp = sfd_bytes < ref_bytes ? sfd_bytes : ref_bytes;
  std::vector<uint8_t> h_sfd(cmp), h_ref(cmp);
  CK(cudaMemcpy(h_sfd.data(), d_sfd.p, cmp, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(h_ref.data(), d_ref_sf.p, cmp, cudaMemcpyDeviceToHost));

  size_t ncb = (n / 16) / 4;
  size_t ncb_rd = (ncb + 31u) & ~(size_t)31u;
  size_t mismatches = 0;
  for (size_t off = 0; off < cmp; off++) {
    if (h_sfd[off] != h_ref[off]) {
      if (mismatches < 16) {
        // Decode per §1.3: off = (tile_row*ncb_rd + tile_col)*512 + r128*4 + cc
        size_t tile = off / 512, rem = off % 512;
        size_t tile_row = tile / ncb_rd, tile_col = tile % ncb_rd;
        size_t r128 = rem / 4, cc = rem % 4;
        printf("  MISMATCH off=%zu tile_row=%zu tile_col=%zu r128=%zu cc=%zu "
               "cutlass=0x%02x h3=0x%02x\n",
               off, tile_row, tile_col, r128, cc, h_sfd[off], h_ref[off]);
      }
      mismatches++;
    }
  }
  bool size_ok = (sfd_bytes == ref_bytes);
  bool pass = (mismatches == 0) && size_ok;
  printf("sfd_compare,%s,M=%zu,N=%zu,K=%zu,bytes=%zu,mismatches=%zu,%s\n",
         s.name, m, n, k, cmp, mismatches, pass ? "PASS" : "FAIL");
  return pass;
}

#endif // CUTLASS_ARCH_MMA_SM120_SUPPORTED

int main() {
#if !defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) && !defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  fprintf(stderr, "Compiled without sm_120/121 support.\n");
  return 1;
#else
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  fprintf(stderr, "GPU: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
  if (!(prop.major == 12 && (prop.minor == 0 || prop.minor == 1))) {
    fprintf(stderr, "Requires Blackwell GeForce (sm_120/121).\n");
    return 1;
  }

  bool all_pass = true;
  // FC1 / FC2 production shapes (K=5376 and K=14336 are both %64==0; neither
  // is %2048==0, so both already exercise the K-tail padded extent).
  all_pass &= run_shape({"FC1", 16000, 28672, 5376});
  all_pass &= run_shape({"FC2", 16000, 5376, 14336});
  // Ragged M (15973 = 124*128 + 101): padding rows must carry SF=0 exactly.
  all_pass &= run_shape({"FC1_M15973", 15973, 28672, 5376});
  // Explicit K-tail stress with a small M: ncb=65 -> padded extent 96 cols.
  all_pass &= run_shape({"KTAIL", 384, 4160, 5376});

  printf("sfd_compare,ALL,-,-,-,-,-,%s\n", all_pass ? "PASS" : "FAIL");
  return all_pass ? 0 : 2;
#endif
}
