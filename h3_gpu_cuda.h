#ifndef H3_GPU_CUDA_H
#define H3_GPU_CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Independent CUDA probe surface. Does not implement or replace h3_gpu.h. */

typedef struct {
    int device_index;
    char name[256];
    int compute_major;
    int compute_minor;
    size_t total_memory_bytes;
    size_t free_memory_bytes;
    int multiprocessor_count;
    int driver_version;
    int runtime_version;
} h3_cuda_device_info;

typedef struct h3_cuda h3_cuda;

/* device_index < 0 selects CUDA device 0. */
h3_cuda *h3_cuda_create(int device_index, char *error, size_t error_size);
void h3_cuda_free(h3_cuda *cuda);
/* Free DiT-sized scratch (SDPA / GEMM acc / GQA / act) but keep the CUDA
 * context, cuBLAS, and cuDNN handle so VAE decode does not rebuild JIT
 * graphs from a fresh ctx. */
void h3_cuda_release_heavy_workspace(h3_cuda *cuda);

/* Pre-size INT8 GEMM acc / act ConvRot / SDPA scratch for this seq so the
 * first DiT layer does not cudaMalloc into a full weight window. */
int h3_cuda_pregrow_dit_scratch(h3_cuda *cuda, size_t seq, size_t hidden,
                                size_t heads, size_t head_dim, size_t ffn,
                                char *error, size_t error_size);

/* Upload ConvRot Hadamard on this handle (pageable H2D). Call from one
 * thread per device before U>1 workers enqueue GEMM — two threads sharing
 * the runtime bounce buffer IMA rank 1. U=1 hits this lazily in GEMM. */
int h3_cuda_pregrow_hadamard(h3_cuda *cuda, int gs, char *error,
                             size_t error_size);

/* 1 if GEMM acc + SDPA scratch are already allocated (WS-first pregrow).
 * Refit must not subtract another 8GiB from remaining free — that is what
 * clamped cafe_p2 to window=19 after pregrow left 15GiB free. */
int h3_cuda_dit_scratch_pregrown(const h3_cuda *cuda);

/* Ulysses only: bind this handle's device, then drop scratch that landed
 * on another GPU (quiet GEMM skips SetDevice and would IMA rank 1). */
int h3_cuda_bind_device(h3_cuda *cuda, char *error, size_t error_size);
int h3_cuda_dit_scratch_rehome(h3_cuda *cuda, char *error, size_t error_size);
int h3_cuda_dit_scratch_check(h3_cuda *cuda, char *error, size_t error_size);
int h3_cuda_recreate_blas(h3_cuda *cuda, char *error, size_t error_size);

/* JIT the fused cuDNN SDPA graph for this logical seq (pads internally).
 * Safe to call while TE mmap is opening: keeps a separate plan slot so TE's
 * causal seq=576 graph is not destroyed. Lossless warmup of DiT step1. */
int h3_cuda_sdpa_prebuild(h3_cuda *cuda, size_t seq, size_t heads,
                          size_t head_dim, float scale, int causal,
                          char *error, size_t error_size);

/* cuDNN frontend JIT needs NVRTC (libnvrtc.so.13 on CUDA 13, .so.12 on 12.x).
 * Preloaded at create(); NULL if missing. */
const char *h3_cuda_nvrtc_path(void);
int h3_cuda_nvrtc_ok(void);

int h3_cuda_device_count(char *error, size_t error_size);
int h3_cuda_probe(int device_index, h3_cuda_device_info *info,
                  char *error, size_t error_size);
int h3_cuda_get_info(const h3_cuda *cuda, h3_cuda_device_info *info);

/* Refresh free/total VRAM via cudaMemGetInfo (also updates cached info). */
int h3_cuda_mem_info(h3_cuda *cuda, size_t *free_bytes, size_t *total_bytes,
                     char *error, size_t error_size);

/* Host FP32 GEMM via cuBLAS: y[M,N] = x[M,K] @ W[N,K]^T (row-major).
 * Uploads x/W, runs cublasSgemm, downloads y. Optional timing in *gemm_ms. */
int h3_cuda_gemm_f32(h3_cuda *cuda, const float *x, const float *weight,
                     size_t m, size_t n, size_t k, float *y, double *gemm_ms,
                     char *error, size_t error_size);

/* Host NHWC Conv3d + bias. Weight is PyTorch [Cout,Cin,Kd,Kh,Kw].
 * Input is already padded (causal time / reflect spatial). Output is
 * [Od,Oh,Ow,Cout] with Od=(D-Kd)/st+1. Used by host VAE encode. */
int h3_cuda_conv3d_nhwc_bias_f32(
    h3_cuda *cuda, const float *input, uint32_t depth, uint32_t height,
    uint32_t width, const float *weight, const float *bias, uint32_t cin,
    uint32_t cout, uint32_t kd, uint32_t kh, uint32_t kw, uint32_t stride_t,
    uint32_t stride_h, uint32_t stride_w, float *output, char *error,
    size_t error_size);

/* Like h3_cuda_gemm_f32 but W already on device (no weight H2D). */
int h3_cuda_gemm_f32_dw(h3_cuda *cuda, const float *x, const float *d_weight,
                        size_t m, size_t n, size_t k, float *y, double *gemm_ms,
                        char *error, size_t error_size);

/* Fully device-resident FP32 GEMM: d_x / d_weight / d_y on GPU. */
int h3_cuda_gemm_f32_dd(h3_cuda *cuda, const float *d_x, const float *d_weight,
                        size_t m, size_t n, size_t k, float *d_y,
                        double *gemm_ms, char *error, size_t error_size);

/* y = alpha * (x @ W^T) + beta * y. LoRA second GEMM uses alpha=scale, beta=1. */
int h3_cuda_gemm_f32_dd_ex(h3_cuda *cuda, const float *d_x,
                           const float *d_weight, size_t m, size_t n, size_t k,
                           float *d_y, float alpha, float beta, double *gemm_ms,
                           char *error, size_t error_size);

/* Device act ConvRot: d_y[m,k] = d_x @ H per group (cuBLAS strided).
 * d_x and d_y must not alias. */
int h3_cuda_convrot_act_f32_dd(h3_cuda *cuda, const float *d_x, size_t m,
                               size_t k, int convrot_groupsize, float *d_y,
                               char *error, size_t error_size);

/* Device FWHT act ConvRot (Sylvester Hadamard; Comfy VAE INT8). */
int h3_cuda_fwht_act_f32_dd(h3_cuda *cuda, const float *d_x, size_t m, size_t k,
                            int convrot_groupsize, float *d_y, char *error,
                            size_t error_size);

/* Device FP16 Tensor-Core GEMM (inputs/weights/out are __half*).
 * y = x @ W^T  (row-major), same layout as gemm_f32_dd. */
int h3_cuda_gemm_f16_dd(h3_cuda *cuda, const void *d_x_half,
                        const void *d_weight_half, size_t m, size_t n, size_t k,
                        void *d_y_half, double *gemm_ms, char *error,
                        size_t error_size);

/* Cast helpers for VAE FP16 path. */
int h3_cuda_f32_to_f16_dev(h3_cuda *cuda, const float *d_in, void *d_out_half,
                           size_t n, char *error, size_t error_size);
int h3_cuda_f16_to_f32_dev(h3_cuda *cuda, const void *d_in_half, float *d_out,
                           size_t n, char *error, size_t error_size);

/* Device I8×per-row scale → F32. d_i8 is [n,k] row-major, d_scale is [n],
 * d_out is [n,k]. stream may be NULL (default stream). cuda handle unused. */
int h3_cuda_dequant_i8_rowscale_f32(const int8_t *d_i8, const float *d_scale,
                                    size_t n, size_t k, float *d_out,
                                    void *stream, char *error,
                                    size_t error_size);

/* Toggle graph-capture quiet mode (skip event sync / forbid realloc). */
void h3_cuda_set_graph_capture(h3_cuda *cuda, int enabled);
int h3_cuda_graph_capture(const h3_cuda *cuda);
/* Quiet: skip per-op EventSynchronize but allow buffer growth. */
void h3_cuda_set_quiet(h3_cuda *cuda, int enabled);
int h3_cuda_quiet(const h3_cuda *cuda);

/* Borrowed compute stream for device-resident GEMM/SDPA. NULL restores
 * the legacy default stream (U=1). Also binds cuBLAS / cuDNN. */
int h3_cuda_set_op_stream(h3_cuda *cuda, void *stream, char *error,
                          size_t error_size);
void *h3_cuda_op_stream(const h3_cuda *cuda);

/* Side streams (NonBlocking). index 0..3; NULL if unavailable.
 * Used to overlap weight prefetch / aux H2D with compute. */
void *h3_cuda_stream(h3_cuda *cuda, int index);
int h3_cuda_stream_count(void);

/* Record an event on the default stream (for cross-stream waits). */
int h3_cuda_event_record_default(h3_cuda *cuda, void **out_event, char *error,
                                 size_t error_size);
int h3_cuda_stream_wait_event(void *stream, void *event, char *error,
                              size_t error_size);
void h3_cuda_event_destroy(void *event);

/* Small device helpers used by VAE GPU-resident decode. */
int h3_cuda_rms_norm_f32_dev(h3_cuda *cuda, const float *d_x, const float *d_w,
                             size_t rows, size_t width, float eps, float *d_out,
                             char *error, size_t error_size);
int h3_cuda_layer_norm_bias_f32_dev(h3_cuda *cuda, const float *d_x,
                                    const float *d_w, const float *d_b,
                                    size_t rows, size_t width, float eps,
                                    float *d_out, char *error,
                                    size_t error_size);
int h3_cuda_silu_mul_f32_dev(h3_cuda *cuda, const float *d_fused, size_t rows,
                             size_t width, float *d_out, char *error,
                             size_t error_size);
int h3_cuda_residual_scale_f32_dev(h3_cuda *cuda, float *d_x,
                                   const float *d_branch, const float *d_scale,
                                   size_t rows, size_t width, char *error,
                                   size_t error_size);
/* qkv_thirds [seq, 3*inner] → Q/K/V [seq,heads,d] with inner=heads*d. */
int h3_cuda_qkv_thirds_to_heads_f32_dev(h3_cuda *cuda, const float *d_qkv,
                                        size_t seq, size_t heads,
                                        size_t head_dim, float *d_q, float *d_k,
                                        float *d_v, char *error,
                                        size_t error_size);
int h3_cuda_bias_add_f32_dev(h3_cuda *cuda, float *d_y, const float *d_bias,
                             size_t rows, size_t width, char *error,
                             size_t error_size);

/* Host INT8×INT8→INT32 GEMM + scale dequant → FP32 (same semantics as
 * h3_w4a8_gemm_i8_dequant_f32). Prefers cuBLASLt CUBLAS_COMPUTE_32I;
 * falls back to cublasGemmEx then a device INT8 GEMM kernel.
 * y[m,n] = act_scale[row] * s_channel[col] * Σ_k x_i8[m,k] * w_i8[n,k]
 * act_scales length 1 (per_tensor_scale≠0) or M. *gemm_ms covers device
 * GEMM+dequant only (excludes H2D/D2H). backend_out (optional) receives
 * "cublasLt" / "cublasGemmEx" / "cuda_kernel". */
int h3_cuda_gemm_i8_dequant_f32(h3_cuda *cuda, const int8_t *x_i8,
                                const int8_t *w_i8, const float *act_scales,
                                int per_tensor_scale, const float *s_channel,
                                size_t m, size_t n, size_t k, float *y_f32,
                                double *gemm_ms, char *backend_out,
                                size_t backend_out_size, char *error,
                                size_t error_size);

/* --- P4 step-5: device W4A8 unpack → resident INT8 weight --- */

typedef struct h3_cuda_w4a8_weight h3_cuda_w4a8_weight;

/* Upload packed W4A8 (+ codebook / s_rel / s_channel) and unpack on device
 * to INT8[N,K] (ConvRot-rotated basis). INT8 + s_channel stay resident.
 * *unpack_ms = device unpack kernel only; *h2d_ms = packed+scales H2D. */
int h3_cuda_w4a8_weight_create(h3_cuda *cuda, const int8_t *packed_host,
                               size_t n, size_t k_half,
                               const float *codebook16,
                               const uint8_t *s_rel_f8,
                               const float *s_channel, int group_size,
                               h3_cuda_w4a8_weight **out, double *unpack_ms,
                               double *h2d_ms, char *error, size_t error_size);

/* Upload already-unpacked INT8 ConvRot weights (Comfy pruned_int8_convrot).
 * w_i8 is [N,K] row-major; s_channel is [N] (weight_scale). Same resident
 * layout as W4A8 after device unpack — reuses INT8 GEMM act path.
 * borrow_host=0: on success takes ownership of w_i8 + s_channel (freed after
 * side-stream H2D via ensure_ready / weight_free). borrow_host=1: caller keeps
 * host buffers (for cache reuse across device evict/reload). On failure
 * caller always retains host pointers. */
int h3_cuda_w4a8_weight_create_from_i8(h3_cuda *cuda, int8_t *w_i8, size_t n,
                                       size_t k, float *s_channel,
                                       h3_cuda_w4a8_weight **out,
                                       double *h2d_ms, int borrow_host,
                                       char *error, size_t error_size);

/* Prefer cudaHostAlloc; fall back to malloc. *pinned_out=1 if cuda-pinned. */
void *h3_cuda_host_alloc(size_t bytes, int *pinned_out);
void h3_cuda_host_free(void *p, int pinned);

/* Wait until weight H2D is visible to the default stream; releases owned
 * host staging. Safe to call multiple times. */
int h3_cuda_w4a8_weight_ensure_ready(h3_cuda_w4a8_weight *weight, char *error,
                                     size_t error_size);

/* 1 if d_w_i8 / d_s_channel live on want_dev. Ulysses rank checks. */
int h3_cuda_w4a8_weight_on_device(const h3_cuda_w4a8_weight *weight,
                                  int want_dev, char *error, size_t error_size);

/* Synchronize the H2D/unpack side stream only (does not sync default). */
int h3_cuda_copy_stream_synchronize(h3_cuda *cuda, char *error,
                                    size_t error_size);

void h3_cuda_w4a8_weight_free(h3_cuda_w4a8_weight *weight);

size_t h3_cuda_w4a8_weight_n(const h3_cuda_w4a8_weight *weight);
size_t h3_cuda_w4a8_weight_k(const h3_cuda_w4a8_weight *weight);

/* Download resident INT8[N,K] for host comparison (caller allocates n*k). */
int h3_cuda_w4a8_weight_download_int8(const h3_cuda_w4a8_weight *weight,
                                      int8_t *host_out, char *error,
                                      size_t error_size);

/* Same as h3_cuda_gemm_i8_dequant_f32 but W + s_channel already on device
 * (no weight H2D). Uploads only x_i8 + act_scales; downloads y. */
int h3_cuda_gemm_i8_dequant_f32_w(h3_cuda *cuda,
                                  const h3_cuda_w4a8_weight *weight,
                                  const int8_t *x_i8, const float *act_scales,
                                  int per_tensor_scale, size_t m, float *y_f32,
                                  double *gemm_ms, char *backend_out,
                                  size_t backend_out_size, char *error,
                                  size_t error_size);

/* Host F32 X[m,K] → device ConvRot(group) + per-row INT8 quant + resident
 * INT8 GEMM + dequant → host Y[m,N]. Keeps act path on GPU (one H2D/D2H).
 * *prep_ms = ConvRot+quant; *gemm_ms = GEMM+dequant (device events). */
int h3_cuda_gemm_i8_dequant_f32_w_act(
    h3_cuda *cuda, const h3_cuda_w4a8_weight *weight, const float *x_f32,
    int convrot_groupsize, size_t m, float *y_f32, double *prep_ms,
    double *gemm_ms, char *backend_out, size_t backend_out_size, char *error,
    size_t error_size);

/* Device-resident act path: d_x_f32 / d_y_f32 already on GPU (no H2D/D2H).
 * ConvRot uses cublasSgemmStridedBatched over all groups when available. */
int h3_cuda_gemm_i8_dequant_f32_w_act_dev(
    h3_cuda *cuda, const h3_cuda_w4a8_weight *weight, const float *d_x_f32,
    int convrot_groupsize, size_t m, float *d_y_f32, double *prep_ms,
    double *gemm_ms, char *backend_out, size_t backend_out_size, char *error,
    size_t error_size);

/* FlashAttention-style tiled SDPA (FP32, online softmax).
 * Q/K/V/out layout: [seq, heads, head_dim] (no GQA; Q heads == KV heads).
 * head_dim must be <= 128. *sdpa_ms = device kernel only (excl. H2D/D2H). */
int h3_cuda_sdpa_f32(h3_cuda *cuda, const float *query, const float *key,
                     const float *value, size_t seq, size_t heads,
                     size_t head_dim, float scale, float *out, double *sdpa_ms,
                     char *error, size_t error_size);

/* FP16 Flash / TC SDPA (host F32 in/out). Prefer for long seq.
 * *sdpa_ms = device convert+kernel (excl. H2D/D2H of host F32). */
int h3_cuda_sdpa_f16(h3_cuda *cuda, const float *query, const float *key,
                     const float *value, size_t seq, size_t heads,
                     size_t head_dim, float scale, float *out, double *sdpa_ms,
                     char *error, size_t error_size);

/* Device-resident FP16 Flash SDPA: Q/K/V/out already on GPU (F32). */
int h3_cuda_sdpa_f16_dev(h3_cuda *cuda, const float *d_query,
                         const float *d_key, const float *d_value, size_t seq,
                         size_t heads, size_t head_dim, float scale,
                         float *d_out, double *sdpa_ms, char *error,
                         size_t error_size);

/* YCNodes Distance Attention (opt-in). Spatiotemporal Gaussian penalty on
 * video-video scores only; text/audio/cond spans stay unpenalized (Sol lesson).
 * Env: H3_DIT_DIST_ATTN=1, H3_DIT_DIST_SCALE (default 15), H3_DIT_DIST_TEMPORAL
 * (default 3; YCNodes node default 2, max 10). CLI --dist-attn /
 * --dist-scale / --dist-temporal. Default OFF. Video span only. */
int h3_cuda_dist_attn_configure(h3_cuda *cuda, size_t seq, size_t video_start,
                                size_t video_seq, int grid_t, int grid_h,
                                int grid_w, float scale, float temporal_weight,
                                int enabled, char *error, size_t error_size);
void h3_cuda_dist_attn_clear(void);
int h3_cuda_dist_attn_active(void);

/* Extended device SDPA. attn_window==0 → global; >0 → token window ±W
 * (quality risk vs trained global attention — opt-in only).
 * backend_out receives: cuda_flash_f16 / cuda_flash_f16_tc / cuda_flash_f16_win /
 * cuda_cudnn_sdpa / cuda_sage2 (if built with cuDNN; sage2 is opt-in). */
int h3_cuda_sdpa_f16_dev_ex(h3_cuda *cuda, const float *d_query,
                            const float *d_key, const float *d_value,
                            size_t seq, size_t heads, size_t head_dim,
                            float scale, int attn_window, float *d_out,
                            double *sdpa_ms, char *backend_out,
                            size_t backend_out_size, char *error,
                            size_t error_size);

/* Host-wrapper of *_dev_ex (uploads Q/K/V, downloads out). */
int h3_cuda_sdpa_f16_ex(h3_cuda *cuda, const float *query, const float *key,
                        const float *value, size_t seq, size_t heads,
                        size_t head_dim, float scale, int attn_window,
                        float *out, double *sdpa_ms, char *backend_out,
                        size_t backend_out_size, char *error,
                        size_t error_size);

/* Causal GQA SDPA for TE: Q[T,Hq,D] K/V[T,Hkv,D] → O[T,Hq,D].
 * Prefers cuDNN fused causal Flash (Hq via KV-head repeat); naive GPU
 * fallback. Host F32 in/out. */
int h3_cuda_sdpa_gqa_causal_f16(h3_cuda *cuda, const float *query,
                                const float *key, const float *value,
                                size_t seq, size_t q_heads, size_t kv_heads,
                                size_t head_dim, float scale, float *out,
                                double *sdpa_ms, char *backend_out,
                                size_t backend_out_size, char *error,
                                size_t error_size);

/* Device-resident GQA (Q/K/V/O on GPU). Opt-in for TE act residency;
 * default host wrapper remains the theme-path contract. */
int h3_cuda_sdpa_gqa_causal_f16_dev(h3_cuda *cuda, const float *d_query,
                                    const float *d_key, const float *d_value,
                                    size_t seq, size_t q_heads, size_t kv_heads,
                                    size_t head_dim, float scale, float *d_out,
                                    double *sdpa_ms, char *backend_out,
                                    size_t backend_out_size, char *error,
                                    size_t error_size);

/* SDPA backend preference via env H3_CUDA_SDPA_BACKEND=
 *   auto|cudnn|sage3|sol_cute|flash|tc|sage2|sol
 *   Tier-0: auto→cudnn (never auto-picks lossy backends).
 *   Tier-1 product accel: sage3, sol_cute (first-class; see h3_sdpa.h).
 *   Forbidden as product: sage2 (ref), sol (nested-loop fake). */
const char *h3_cuda_sdpa_backend_name(void);
/* Log resolved backend + device caps once (generate / create). */
void h3_cuda_sdpa_log_selection(int device_index);
/* Last Sol-Attn exact-tile fraction (query×block), or -1 if unused. */
double h3_cuda_sol_last_exact_frac(void);
/* Official SM89 CuTe Sol-Attn opt-in: reset layer clock at each denoise step. */
void h3_cuda_sol_cute_set_step(int step);
/* Print cute_sm89 hits / dense_skip / fallback counts (stderr). */
void h3_cuda_sol_cute_dump_stats(void);
/* Official SageAttention3 (Blackwell FP4) plugin stats. */
void h3_cuda_sage3_dump_stats(void);
/* Opt-in DiT breakdown: H3_SOL_PROFILE=1. Extra EventSynchronize; not a
 * speedup path. Per-step line + totals on dump. */
int h3_cuda_sol_profile_enabled(void);
void h3_cuda_sol_profile_add_unpack(double ms);
void h3_cuda_sol_profile_finish_step(int step, double step_ms);

/* Device-side DiT elementwise (host F32 in/out). Used by W4A8 bypass.
 * modulation is [mod_rows, slots, width]; row_map[i] selects modality row. */
int h3_cuda_adaln_f32(h3_cuda *cuda, const float *x, const float *norm_w,
                      const float *modulation, const uint32_t *row_map,
                      size_t rows, size_t width, size_t slots, size_t mod_rows,
                      size_t shift_slot, size_t scale_slot, float epsilon,
                      float *out, double *op_ms, char *error,
                      size_t error_size);
int h3_cuda_gate_f32(h3_cuda *cuda, const float *residual, const float *branch,
                     const float *modulation, const uint32_t *row_map,
                     size_t rows, size_t width, size_t slots, size_t mod_rows,
                     size_t gate_slot, float *out, double *op_ms, char *error,
                     size_t error_size);
int h3_cuda_swiglu_f32(h3_cuda *cuda, const float *fused, size_t rows,
                       size_t width, float *out, double *op_ms, char *error,
                       size_t error_size);
int h3_cuda_qkv_rope_f32(h3_cuda *cuda, const float *qkv, const float *q_norm,
                         const float *k_norm, const float *rope_cos,
                         const float *rope_sin, size_t seq, size_t heads,
                         size_t head_dim, size_t rope_half, float epsilon,
                         float *query, float *key, float *value, double *op_ms,
                         char *error, size_t error_size);

/* Device-pointer DiT ops (no H2D/D2H). NULL d_row_map → identity row index. */
int h3_cuda_adaln_f32_dev(h3_cuda *cuda, const float *d_x, const float *d_norm_w,
                          const float *d_modulation, const uint32_t *d_row_map,
                          size_t rows, size_t width, size_t slots,
                          size_t shift_slot, size_t scale_slot, float epsilon,
                          float *d_out, double *op_ms, char *error,
                          size_t error_size);
int h3_cuda_gate_f32_dev(h3_cuda *cuda, const float *d_residual,
                         const float *d_branch, const float *d_modulation,
                         const uint32_t *d_row_map, size_t rows, size_t width,
                         size_t slots, size_t gate_slot, float *d_out,
                         double *op_ms, char *error, size_t error_size);
int h3_cuda_swiglu_f32_dev(h3_cuda *cuda, const float *d_fused, size_t rows,
                           size_t width, float *d_out, double *op_ms,
                           char *error, size_t error_size);

/* SwiGLU(fused) → INT8 ConvRot+GEMM(fc2) on the same stream, no host sync
 * between kernels. Same math as swiglu_dev + gemm_i8_w_act_dev. */
int h3_cuda_ffn_swiglu_i8_dev(
    h3_cuda *cuda, const h3_cuda_w4a8_weight *w_fc2, const float *d_fused,
    float *d_act, int convrot_groupsize, size_t m, size_t ffn, float *d_y,
    double *swiglu_ms, double *prep_ms, double *gemm_ms, char *backend_out,
    size_t backend_out_size, char *error, size_t error_size);
int h3_cuda_qkv_rope_f32_dev(h3_cuda *cuda, const float *d_qkv,
                             const float *d_q_norm, const float *d_k_norm,
                             const float *d_rope_cos, const float *d_rope_sin,
                             size_t seq, size_t heads, size_t head_dim,
                             size_t rope_half, float epsilon, float *d_query,
                             float *d_key, float *d_value, double *op_ms,
                             char *error, size_t error_size);

/* Comfy DiT apply_rope_split_half: pair dim d with d+rope_half. */
int h3_cuda_qkv_rope_split_half_f32_dev(
    h3_cuda *cuda, const float *d_qkv, const float *d_q_norm,
    const float *d_k_norm, const float *d_rope_cos, const float *d_rope_sin,
    size_t seq, size_t heads, size_t head_dim, size_t rope_half, float epsilon,
    float *d_query, float *d_key, float *d_value, double *op_ms, char *error,
    size_t error_size);

/* RoPE writes FP16 Q/K/V into SDPA scratch then runs SDPA (no f32 QKV store). */
int h3_cuda_qkv_rope_sdpa_f16_dev(
    h3_cuda *cuda, const float *d_qkv, const float *d_q_norm,
    const float *d_k_norm, const float *d_rope_cos, const float *d_rope_sin,
    size_t seq, size_t heads, size_t head_dim, size_t rope_half, float epsilon,
    float scale, int attn_window, float *d_attn_out, double *rope_ms,
    double *sdpa_ms, char *backend_out, size_t backend_out_size, char *error,
    size_t error_size);
/* Same, with explicit QKV layout (grouped=Metal interleaved; split_half=Comfy). */
int h3_cuda_qkv_rope_sdpa_f16_dev_layout(
    h3_cuda *cuda, const float *d_qkv, const float *d_q_norm,
    const float *d_k_norm, const float *d_rope_cos, const float *d_rope_sin,
    size_t seq, size_t heads, size_t head_dim, size_t rope_half, float epsilon,
    float scale, int attn_window, int grouped, int split_half,
    float *d_attn_out, double *rope_ms, double *sdpa_ms, char *backend_out,
    size_t backend_out_size, char *error, size_t error_size);

/* Metal-compatible token-reduction pool/expand (host F32 activations).
 * pairs: [reduced_rows*2] {first,second}; baseline_idx: UINT32_MAX=none.
 * Pool keeps original/baseline on device (*d_original_out / *d_baseline_out;
 * caller cudaFree). Expand consumes those device buffers + host reduced. */
int h3_cuda_token_pool_f32(h3_cuda *cuda, const float *input, size_t full_rows,
                           const uint32_t *pairs, const uint32_t *baseline_idx,
                           size_t reduced_rows, size_t baseline_rows,
                           size_t width, float *output, float **d_original_out,
                           float **d_baseline_out, double *op_ms, char *error,
                           size_t error_size);
int h3_cuda_token_expand_f32(h3_cuda *cuda, const float *d_original,
                             const float *reduced, const float *d_baseline,
                             const uint32_t *baseline_idx,
                             const uint32_t *parents, size_t full_rows,
                             size_t reduced_rows, size_t baseline_rows,
                             size_t width, uint32_t video_start,
                             uint32_t video_rows, float update_scale,
                             float *output, double *op_ms, char *error,
                             size_t error_size);

/* Device-resident pool/expand (no activation PCIe). Caller owns all device
 * buffers; pairs/baseline_idx/parents are still host (tiny index tables). */
int h3_cuda_token_pool_f32_dev(h3_cuda *cuda, const float *d_input,
                               size_t full_rows, const uint32_t *pairs,
                               const uint32_t *baseline_idx, size_t reduced_rows,
                               size_t baseline_rows, size_t width,
                               float *d_output, float *d_original,
                               float *d_baseline, double *op_ms, char *error,
                               size_t error_size);
int h3_cuda_token_expand_f32_dev(
    h3_cuda *cuda, const float *d_original, const float *d_reduced,
    const float *d_baseline, const uint32_t *baseline_idx,
    const uint32_t *parents, size_t full_rows, size_t reduced_rows,
    size_t baseline_rows, size_t width, uint32_t video_start,
    uint32_t video_rows, float update_scale, float *d_output, double *op_ms,
    char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
