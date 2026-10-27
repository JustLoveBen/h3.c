#ifndef H3_DIT_W4A8_H
#define H3_DIT_W4A8_H

#include "h3_gpu_cuda.h"
#include "h3_lora.h"
#include "h3_nvfp4.h"
#include "h3_w4a8.h"
#include "h3_w4a8_cache.h"
#include "h3_w4a8_dequant.h"

#include <stddef.h>
#include <stdio.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* P4/E2: CUDA DiT W4A8 bypass (block → multi-block → patch/final → sample).
 *
 * Reserved for a future h3_dit CUDA path. Does not replace Metal h3_dit /
 * BF16 load_block. Callers: tools/h3_dit_w4a8_*_probe.c /
 * tools/h3_w4a8_generate_probe.c.
 *
 * Quantized linears: qkv / out / fc1 / fc2 via CUDA ConvRot+INT8 act +
 * resident INT8 GEMM (env H3_DIT_W4A8_FORCE_CPU_ACT=1 → host ConvRot/quant).
 * Non-quant: AdaLN / QK-RMS / RoPE / gate / SwiGLU on CUDA by default
 * (H3_DIT_W4A8_FORCE_CPU_OPS=1 → host F32). SDPA default = FP16 flash
 * (H3_DIT_W4A8_FORCE_SDPA_F32=1 → FP32 flash; FORCE_CPU_SDPA=1 → host).
 * Default block path keeps activations GPU-resident (one H2D/D2H per block);
 * H3_DIT_W4A8_FORCE_HOST_BLOCK=1 restores the per-op H2D/D2H host staging path.
 * End-to-end forward keeps patch→layers→final activations on device by default
 * (H3_DIT_W4A8_HOST_FWD=1 restores host hidden staging between stages).
 * SDPA backend: H3_CUDA_SDPA_BACKEND=auto|cudnn|sage3|sol_cute|…
 *   (auto→cudnn Tier-0; sage3/sol_cute = Tier-1 product accel; sage2/sol
 *    forbidden as product). See h3_sdpa.h.
 * Optional token window: H3_DIT_W4A8_ATTN_WINDOW=W (0=global; quality risk).
 * Distance attention (YCNodes Gaussian RF, video span only):
 *   H3_DIT_DIST_ATTN=1 (default off). H3_DIT_DIST_SCALE=15 H3_DIT_DIST_TEMPORAL=3
 *   H3_DIT_DIST_START_SIGMA=2.5 H3_DIT_DIST_END_SIGMA=0. Does not touch
 *   text/audio tokens. Not a speedup.
 *
 * QKV layout: default contiguous [Q|K|V] (Comfy split). Set
 * H3_DIT_QKV_GROUPED=1 for Metal interleaved-per-head exports.
 * RoPE: default Comfy first/second-half split. Set H3_DIT_ROPE_HALF_OFFSET=1
 * for Metal d↔d+48 pairing. AdaLN: no SiLU on curve table (H3_DIT_ADALN_SILU=1
 * opt-in).
 * CFG: H3 is CFG-distilled (vLLM-Omni/SGLang/Metal) — keep scale=1.
 *
 * Text condition: packed as leading tokens (modality tag=1); video tag=0.
 * Token path mirrors Metal: condition_proj → token_refiner.blocks.{0,1}
 * → final_norm (BF16 weights loaded as F32). H3_DIT_NO_REFINER=1 skips
 * refiner (condition_proj only; debug/ablation).
 * Metal changes: none. */

enum {
    H3_DIT_W4A8_HIDDEN = 5376,
    H3_DIT_W4A8_TEXT_DIM = 5120,
    H3_DIT_W4A8_HEADS = 56,
    H3_DIT_W4A8_HEAD_DIM = 128,
    H3_DIT_W4A8_INNER = H3_DIT_W4A8_HEADS * H3_DIT_W4A8_HEAD_DIM, /* 7168 */
    H3_DIT_W4A8_FFN = 14336,
    H3_DIT_W4A8_SLOTS = 6,
    H3_DIT_W4A8_MODALITIES = 3,
    H3_DIT_W4A8_MOD_VIDEO = 0,
    H3_DIT_W4A8_MOD_TEXT = 1,
    H3_DIT_W4A8_MOD_AUDIO = 2,
    H3_DIT_W4A8_ROPE_FREQS = 16,
    H3_DIT_W4A8_ROPE_HALF = 48, /* 3 axes * 16 */
    /* Kijai W4A8 mixed: adaln_t_table[1025,8] + adaln_proj [M*S*H, 8] */
    H3_DIT_W4A8_TIME_FEAT = 8,
    H3_DIT_W4A8_VIDEO_PATCH = 96,
    H3_DIT_W4A8_AUDIO_CH = 32,
    H3_DIT_W4A8_FINAL_SLOTS = 2,
    H3_DIT_W4A8_N_BLOCKS = 50
};

#define H3_DIT_W4A8_ADALN_OUT                                              \
    ((size_t)H3_DIT_W4A8_MODALITIES * H3_DIT_W4A8_SLOTS * H3_DIT_W4A8_HIDDEN)

typedef struct h3_dit_w4a8 h3_dit_w4a8;

typedef struct {
    double convrot_ms;
    double quant_ms;
    double gemm_ms;
    char backend[32];
    size_t m;
    size_t n;
    size_t k;
} h3_dit_w4a8_linear_stats;

/* Host F32 copies of non-quantized block tensors (from W4A8 file). */
typedef struct {
    float *norm1;      /* [HIDDEN] BF16 */
    float *norm2;      /* [HIDDEN] BF16 */
    float *q_norm;     /* [HEAD_DIM] BF16 */
    float *k_norm;     /* [HEAD_DIM] BF16 */
    float *adaln_w;    /* [ADALN_OUT, TIME_FEAT] F16 row-major */
    float *adaln_b;    /* [ADALN_OUT] F16 */
    size_t adaln_out;  /* expected H3_DIT_W4A8_ADALN_OUT */
    size_t adaln_in;   /* expected H3_DIT_W4A8_TIME_FEAT */
    float *inv_freq;   /* [ROPE_FREQS] F32 */
    float *t_table;    /* [t_rows, TIME_FEAT] F32; may be NULL */
    size_t t_table_rows;
} h3_dit_w4a8_block_aux;

/* Optional timing for full-block forward. */
typedef struct {
    double adaln_attn_ms;
    double qkv_ms;
    double rope_ms;
    double sdpa_ms;      /* true SDPA only (verbose); quiet uses block_gpu_ms */
    double block_gpu_ms; /* quiet: whole-block GPU wall (not SDPA share) */
    double out_ms;
    double gate_attn_ms;
    double adaln_mlp_ms;
    double fc1_ms;
    double swiglu_ms;
    double fc2_ms;
    double gate_mlp_ms;
    double act_prep_ms; /* sum ConvRot+quant over 4 linears (CUDA or CPU) */
    char qkv_backend[32];
    char out_backend[32];
    char fc1_backend[32];
    char fc2_backend[32];
    char sdpa_backend[32]; /* cuda_flash_f16 / cuda_cudnn_sdpa / cuda_sage2 / … */
} h3_dit_w4a8_block_stats;

/* BF16 token_refiner block (Metal refine_text); weights dequantized to F32. */
typedef struct {
    float *norm1;  /* [HIDDEN] */
    float *norm2;  /* [HIDDEN] */
    float *q_norm; /* [HEAD_DIM] */
    float *k_norm; /* [HEAD_DIM] */
    float *qkv_w;  /* [INNER*3, HIDDEN] */
    float *out_w;  /* [HIDDEN, INNER] */
    float *fc1_w;  /* [FFN*2, HIDDEN] */
    float *fc2_w;  /* [HIDDEN, FFN] */
} h3_dit_w4a8_refiner_block;

/* Host F32 patch / final / shared t_table / condition_proj (from W4A8 file). */
typedef struct {
    float *video_patch_w; /* [HIDDEN, VIDEO_PATCH] */
    float *video_patch_b; /* [HIDDEN] */
    float *audio_patch_w; /* [HIDDEN, AUDIO_CH] optional (Au3) */
    float *audio_patch_b; /* [HIDDEN] optional */
    float *final_norm;    /* [HIDDEN] */
    float *final_video_w; /* [VIDEO_PATCH, HIDDEN] */
    float *final_video_b; /* [VIDEO_PATCH] */
    float *final_audio_w; /* [AUDIO_CH, HIDDEN] optional (Au4) */
    float *final_audio_b; /* [AUDIO_CH] optional */
    float *final_adaln_w; /* [HIDDEN*FINAL_SLOTS, TIME_FEAT] */
    float *final_adaln_b; /* [HIDDEN*FINAL_SLOTS] */
    float *t_table;       /* [t_rows, TIME_FEAT] */
    size_t t_table_rows;
    float *inv_freq; /* [ROPE_FREQS] */
    float *condition_w; /* [HIDDEN, TEXT_DIM] optional */
    float *condition_b; /* [HIDDEN] optional */
    h3_dit_w4a8_refiner_block refiner[2];
    float *refiner_final_norm; /* [HIDDEN] optional */
    int has_refiner;
} h3_dit_w4a8_core;

typedef struct {
    unsigned first_block;
    unsigned n_blocks;
    double total_ms;
    double prefetch_ms;
    double blocks_ms;        /* warm-prefetch后的 compute 墙钟（含层间） */
    double h2d_ms;           /* x/rope/aux(/map) H2D */
    double compute_ms;       /* GPU block 链（CUDA event；quiet 也可开） */
    double d2h_ms;           /* y D2H */
    double prefetch_wait_ms; /* 侧流 unpack 空等粗估（0=未测） */
    double gemm_ms_sum;
    double lora_ms; /* GPU LoRA GEMM wall (already-scaled delta), 0=unmeasured */
    int all_finite;
} h3_dit_w4a8_layers_stats;

typedef struct {
    double patch_ms;
    double layers_ms;
    double layers_prefetch_ms;
    double layers_h2d_ms;
    double layers_compute_ms;
    double layers_d2h_ms;
    double layers_prefetch_wait_ms;
    double final_ms;
    double total_ms;
    unsigned first_block;
    unsigned n_blocks;
    size_t t_row;
    int all_finite;
    /* AdaLN visual-condition (Metal time axis). 0 when cond_seq=0. */
    int visual_cond_t_active;
    uint32_t visual_cond_mod_row; /* 3 = VIDEO@0.999; else MOD_VIDEO */
    size_t visual_cond_t_row;
} h3_dit_w4a8_forward_stats;

void h3_dit_w4a8_config_init(h3_w4a8_cache_config *cfg);

/* Borrow cuda + store. Opens an INT8 weight cache; does not prefetch. */
h3_dit_w4a8 *h3_dit_w4a8_open(h3_cuda *cuda, h3_w4a8_store *store,
                              const h3_w4a8_cache_config *cfg, char *error,
                              size_t error_size);
/* Comfy INT8 ConvRot DiT (pruned_int8_convrot); reuses W4A8 forward stack. */
h3_dit_w4a8 *h3_dit_w4a8_open_int8(h3_cuda *cuda, h3_dit_int8_store *store,
                                   const h3_w4a8_cache_config *cfg, char *error,
                                   size_t error_size);
/* Community NVFP4 DiT (dequant→INT8 GEMM). Not gold. ConvRot off. */
h3_dit_w4a8 *h3_dit_w4a8_open_nvfp4(h3_cuda *cuda, h3_nvfp4_store *store,
                                    const h3_w4a8_cache_config *cfg,
                                    char *error, size_t error_size);
int h3_dit_w4a8_load_core_nvfp4(const h3_nvfp4_store *store,
                                h3_dit_w4a8_core *core, char *error,
                                size_t error_size);
void h3_dit_w4a8_free(h3_dit_w4a8 *ctx);

h3_w4a8_cache *h3_dit_w4a8_cache(h3_dit_w4a8 *ctx);
void h3_dit_w4a8_release_block_workspace(h3_dit_w4a8 *ctx);
/* SelfLift hires: allocate fwd-act + all-aux block WS for `seq` BEFORE
 * refitting INT8. Matches layers_forward resident need (ext act). */
int h3_dit_w4a8_pregrow_layers_ws(h3_dit_w4a8 *ctx, size_t seq,
                                  unsigned n_blocks, char *error,
                                  size_t error_size);
h3_cuda *h3_dit_w4a8_cuda(h3_dit_w4a8 *ctx);
const h3_w4a8_store *h3_dit_w4a8_store(const h3_dit_w4a8 *ctx);
int h3_dit_w4a8_convrot_groupsize(const h3_dit_w4a8 *ctx);
int h3_dit_w4a8_group_size(const h3_dit_w4a8 *ctx);
/* DiT INT8 act ConvRot in F16. Default ON after AdaLN-range prescale
 * (5090 KPI-B finite). Rollback = H3_DIT_ACT_F16=0. */
int h3_dit_w4a8_act_f16(void);
/* C015/C042: store d_qkv/d_wide as FP16 in block WS (≠ ACT_F16 ConvRot).
 * Default OFF. Stub until construction lands — getter never arms packing. */
int h3_dit_w4a8_act_fp16_stage(void);
/* Opt-in / VRAM_LEDGER companion: print block-WS component GiB + C015 save. */
void h3_dit_w4a8_ws_component_ledger_fprint(FILE *out, const h3_dit_w4a8 *ctx,
                                           size_t seq, unsigned n_blocks,
                                           int use_ext_act);
/* Fuse post-GEMM gate/AdaLN/SwiGLU. Default ON. Rollback = H3_DIT_FUSED_EPILOGUE=0. */
int h3_dit_w4a8_fused_epilogue(void);
/* LoRA A/B as F16 + mixed GEMM. Opt-in H3_DIT_LORA_F16=1 (cuBLAS mixed n=64 不支持). */
int h3_dit_w4a8_lora_f16(void);
/* LoRA F32 GEMM TF32. Opt-in H3_DIT_LORA_TF32=1 (KPI-B clip1 NaN, 不进产品). */
int h3_dit_w4a8_lora_tf32(void);
/* LoRA GEMM on side stream overlapping INT8. Opt-in H3_DIT_LORA_OVERLAP=1 (hot waffle). */
int h3_dit_w4a8_lora_overlap(void);

/* Metal --token-reduction: pair horizontal video tokens in middle blocks.
 * Default off. Env: H3_TOKEN_REDUCTION_BLOCKS=BEGIN:END,
 * H3_TOKEN_REDUCTION_EARLY=STEPS:END|0, H3_TOKEN_REDUCTION_SCALE. */
void h3_dit_w4a8_set_token_reduction(h3_dit_w4a8 *ctx, int enabled);
int h3_dit_w4a8_token_reduction_enabled(const h3_dit_w4a8 *ctx);
void h3_dit_w4a8_set_denoise_step(h3_dit_w4a8 *ctx, int step);

/* TaoMate layer-wise clean AV KV: Q stays current packed seq; K/V grow only
 * at SDPA. max_tokens is the resident hist cap (sink+recent), not full clip.
 * mode: 0=off, 1=noisy (read hist), 2=commit (read + stage selected rows). */
int h3_dit_w4a8_clean_kv_enable(h3_dit_w4a8 *ctx, size_t max_tokens, char *error,
                                size_t error_size);
void h3_dit_w4a8_clean_kv_free(h3_dit_w4a8 *ctx);
int h3_dit_w4a8_clean_kv_enabled(const h3_dit_w4a8 *ctx);
size_t h3_dit_w4a8_clean_kv_max_tokens(const h3_dit_w4a8 *ctx);
size_t h3_dit_w4a8_clean_kv_hist_tokens(const h3_dit_w4a8 *ctx);
int h3_dit_w4a8_clean_kv_begin(h3_dit_w4a8 *ctx, int mode,
                               const uint32_t *save_idx, size_t save_n,
                               size_t save_dst, char *error, size_t error_size);
void h3_dit_w4a8_clean_kv_finish(h3_dit_w4a8 *ctx, size_t hist_n);
/* Borrowed Metal DiT text tags (0=VIDEO, 1=TEXT, 2=AUDIO). NULL → all TEXT.
 * Vision span rows in the packed TE sequence must be tag=0 (h3_multimodal.c).
 * Does not copy; caller keeps the buffer alive across forward_avc. */
void h3_dit_w4a8_set_text_mod_tags(h3_dit_w4a8 *ctx, const uint8_t *tags,
                                   size_t n);

/* Borrowed LoRA set (NULL = gold path, zero extra kernels). Uploads A/B to GPU. */
int h3_dit_w4a8_set_lora(h3_dit_w4a8 *ctx, const h3_lora_set *lora, char *error,
                         size_t error_size);

/* Ulysses SP comm vtable. Implemented by the caller on top of
 * h3_ulysses_comm so this module stays free of NCCL symbols (links fine
 * against the stub). cfg is the caller's h3_ulysses_cfg with an inited comm.
 * a2a_* operate on the buffers previously bound per rank (F32 payload). */
typedef struct h3_dit_w4a8_sp_comm {
    void *cfg;
    int (*bind)(void *cfg, int rank, const void *d_in, void *d_out, char *err,
                size_t err_size);
    int (*set_compute)(void *cfg, int rank, void *stream, char *err,
                       size_t err_size);
    int (*a2a_qkv)(void *cfg, size_t seq_local, size_t heads, size_t head_dim,
                   char *err, size_t err_size);
    int (*a2a_attn_out)(void *cfg, size_t seq_local, size_t heads,
                        size_t head_dim, char *err, size_t err_size);
} h3_dit_w4a8_sp_comm;

/* Attach Ulysses rank contexts. rank_ctx[0] must be ctx itself; each rank
 * holds a full INT8 ctx on its own CUDA device (devices[]). u<=1 detaches
 * (today's single-GPU path, zero extra A2A). While attached,
 * h3_dit_w4a8_layers_forward_f32 routes to the SP loop. Device-resident
 * I/O is allowed (scatter/gather from io_d_act); H3_DIT_W4A8_HOST_FWD=1
 * restores host hidden staging.
 * Insertion point per block: QKV GEMM + LoRA residual → A2A → RoPE+SDPA on
 * the first S_real rows only with H/U heads (zero-pad K rows must never
 * enter softmax) → pad attn rows zeroed → reverse A2A → out-proj/MLP. */
int h3_dit_w4a8_set_ulysses(h3_dit_w4a8 *ctx, h3_dit_w4a8 *const *rank_ctx,
                            const int *devices, int u,
                            const h3_dit_w4a8_sp_comm *comm, char *error,
                            size_t error_size);

int h3_dit_w4a8_prefetch(h3_dit_w4a8 *ctx, unsigned first_block,
                         unsigned n_blocks, double *elapsed_ms, char *error,
                         size_t error_size);
int h3_dit_w4a8_stage_host(h3_dit_w4a8 *ctx, unsigned first_block,
                           unsigned n_blocks, double *elapsed_ms, char *error,
                           size_t error_size);
int h3_dit_w4a8_get_cache_stats(h3_dit_w4a8 *ctx, h3_w4a8_cache_stats *stats,
                                char *error, size_t error_size);

/* Ensure the linear is resident and return N (out) / K (in). */
int h3_dit_w4a8_linear_shape(h3_dit_w4a8 *ctx, unsigned block,
                             h3_w4a8_linear_kind kind, size_t *n_out,
                             size_t *k_out, char *error, size_t error_size);

/* Whole-step CUDA Graph diagnostics (H3_DIT_W4A8_CUDA_GRAPH=1).
 * mode: 0=off, 1=eager-fallback, 2=graph replay. reason: short tag. */
void h3_dit_w4a8_cuda_graph_last(int *mode, const char **reason);

/* Quantized linear for DiT CUDA:
 *   X_rot = ConvRot(X) along K (group = convrot_groupsize)
 *   X_i8, scales = per-row INT8 quant
 *   Y = resident INT8 GEMM (h3_cuda_gemm_i8_dequant_f32_w)
 * x_f32 is [m, K], y_f32 is [m, N]. stats may be NULL. */
int h3_dit_w4a8_linear_f32(h3_dit_w4a8 *ctx, unsigned block,
                           h3_w4a8_linear_kind kind, const float *x_f32,
                           size_t m, float *y_f32,
                           h3_dit_w4a8_linear_stats *stats, char *error,
                           size_t error_size);

/* Load non-quantized block tensors (norm / qk_norm / adaln / rope / t_table).
 * Caller frees with h3_dit_w4a8_block_aux_free. */
int h3_dit_w4a8_load_block_aux(const h3_w4a8_store *store, unsigned block,
                               h3_dit_w4a8_block_aux *aux, char *error,
                               size_t error_size);
int h3_dit_w4a8_load_block_aux_hdr(const h3_st_header *hdr, unsigned block,
                                   h3_dit_w4a8_block_aux *aux, char *error,
                                   size_t error_size);
void h3_dit_w4a8_block_aux_free(h3_dit_w4a8_block_aux *aux);

/* Lazily cached aux on ctx (no free by caller; lives until dit_free). */
const h3_dit_w4a8_block_aux *h3_dit_w4a8_cached_block_aux(
    h3_dit_w4a8 *ctx, unsigned block, char *error, size_t error_size);

/* time_feat[TIME_FEAT] → modulation[ADALN_OUT] = W @ t + b (row-major W). */
int h3_dit_w4a8_adaln_project_f32(const h3_dit_w4a8_block_aux *aux,
                                  const float *time_feat, float *modulation,
                                  char *error, size_t error_size);

/* Lookup adaln_t_table[row] → time_feat[TIME_FEAT]. */
int h3_dit_w4a8_time_feat_from_table(const h3_dit_w4a8_block_aux *aux,
                                     size_t table_row, float *time_feat,
                                     char *error, size_t error_size);

/* Build RoPE cos/sin [seq, ROPE_HALF] from inv_freq and Metal float (t,h,w).
 * Matches h3_dit.c prepare_rope (spatial axes already scaled; spatial_scale
 * multiplies h/w like Metal). */
int h3_dit_w4a8_rope_tables_f32(const float *inv_freq, size_t seq,
                                const float *pos_t, const float *pos_h,
                                const float *pos_w, float spatial_scale,
                                float *cos_out, float *sin_out, char *error,
                                size_t error_size);

/* Phase-A profile: cumulative RoPE table build time and call count.
 * Enable with H3_ROPE_PROFILE=1. */
void h3_dit_w4a8_rope_profile_get(double *ms, unsigned long *calls);
void h3_dit_w4a8_rope_profile_get_ex(double *ms, unsigned long *calls,
                                     unsigned long *hits,
                                     unsigned long *misses);
void h3_dit_w4a8_rope_profile_reset(void);

/* Metal h3_layout video/audio RoPE coords (float). video_seq=T*(H/2)*(W/2).
 * text_len is the temporal cursor start (same as layout builder). */
int h3_dit_w4a8_fill_metal_video_pos(size_t text_len, int latent_t,
                                     int latent_h, int latent_w, float *pos_t,
                                     float *pos_h, float *pos_w, char *error,
                                     size_t error_size);
/* audio_seq = 2*audio_t; fills stream0 then stream1 (Metal pack order). */
int h3_dit_w4a8_fill_metal_audio_pos(size_t text_len, int audio_t, int latent_h,
                                     int latent_w, float *pos_t, float *pos_h,
                                     float *pos_w, char *error,
                                     size_t error_size);

/* Metal video 时间轴累加：sum_t rescale * per_token[t%5]。
 * fill_metal_video_pos 用同一公式步进。尾帧 condition_time =
 * text_len + span_sum(latent_t) - (5/3)（h3_host.c last keyframe）。 */
float h3_dit_w4a8_video_span_sum(int latent_t);

/* 单帧 COND 网格：空间与 fill_metal_video_pos 同一 canvas，
 * 全部 pos_t = condition_time（首帧=text_len；尾帧见 Task F）。
 * 不加 frame_rescale 时间步进（那是目标视频轴）。 */
int h3_dit_w4a8_fill_metal_cond_pos(float condition_time, int latent_h,
                                    int latent_w, float *pos_t, float *pos_h,
                                    float *pos_w, char *error,
                                    size_t error_size);

/* Metal h3_host.h: canvas multiple / VAE spatial. CUDA copies; do not
 * include Accelerate h3_host.h. */
#ifndef H3_DIT_W4A8_CANVAS_MULTIPLE
#define H3_DIT_W4A8_CANVAS_MULTIPLE 32
#endif
#ifndef H3_DIT_W4A8_VAE_SPATIAL
#define H3_DIT_W4A8_VAE_SPATIAL 16
#endif

/* Metal h3_reference_image_canvas: down-only, aspect-preserving.
 * max_short_edge=0 matches output pixel area; >0 is a short-edge cap.
 * Gold numbers: tests/test_h3.c (1920x1080 vs 512x512 → 672x384). */
int h3_dit_w4a8_reference_image_canvas(int width, int height,
                                       int target_width, int target_height,
                                       int max_short_edge, int *adapted_w,
                                       int *adapted_h);

/* Ref2VA H3_SEG_REF_IMAGE RoPE: same spatial grid as COND, but pos_t is
 * the layout cursor (text_len, then +1 per ordered ref) — not keyframe
 * 0 / last. After N image refs, target audio/video start at text_len+N. */
int h3_dit_w4a8_fill_metal_ref_image_pos(float cursor, int latent_h,
                                         int latent_w, float *pos_t,
                                         float *pos_h, float *pos_w,
                                         char *error, size_t error_size);

/* Metal h3_reference_video_canvas: adapt then allow upscale when source
 * area is smaller than target. Gold: tests/test_h3.c. */
int h3_dit_w4a8_reference_video_canvas(int width, int height, int *adapted_w,
                                       int *adapted_h);

/* Ref2VA H3_SEG_REF_IMAGE for video: same spatial/time step as target
 * video RoPE, but starts at layout cursor (not text_len alone). */
int h3_dit_w4a8_fill_metal_ref_video_pos(float cursor, int latent_t,
                                         int latent_h, int latent_w,
                                         float *pos_t, float *pos_h,
                                         float *pos_w, char *error,
                                         size_t error_size);

/* Ref2VA H3_SEG_REF_AUDIO RoPE: same spatial grid as target audio, but
 * pos_t starts at the layout cursor after ordered visual refs. */
int h3_dit_w4a8_fill_metal_ref_audio_pos(float cursor, int audio_t, int latent_h,
                                         int latent_w, float *pos_t,
                                         float *pos_h, float *pos_w,
                                         char *error, size_t error_size);

/* CPU F32 ops matching Metal h3_*_f32 formulas. */
void h3_dit_w4a8_rms_norm_f32(const float *x, const float *weight, size_t rows,
                              size_t width, float epsilon, float *out);

/* AdaLN: RMSNorm*w then y = n*(1+scale)+shift. modulation [mod_rows, slots, W]
 * indexed by row_map[row] (NULL → identity row). */
void h3_dit_w4a8_adaln_f32(const float *x, const float *norm_w,
                           const float *modulation, const uint32_t *row_map,
                           size_t rows, size_t width, size_t slots,
                           size_t shift_slot, size_t scale_slot, float epsilon,
                           float *out);

void h3_dit_w4a8_gate_f32(const float *residual, const float *branch,
                          const float *modulation, const uint32_t *row_map,
                          size_t rows, size_t width, size_t slots,
                          size_t gate_slot, float *out);

/* Grouped QKV layout (per-head Q|K|V) + Q/K RMSNorm + RoPE.
 * qkv [seq, 3*heads*dim]; q/k/v [seq, heads*dim]; rope [seq, rope_half]. */
int h3_dit_w4a8_qkv_rope_f32(const float *qkv, const float *q_norm,
                             const float *k_norm, const float *rope_cos,
                             const float *rope_sin, size_t seq, size_t heads,
                             size_t head_dim, size_t rope_half, float epsilon,
                             float *query, float *key, float *value,
                             char *error, size_t error_size);

/* P1 FP verification paths (NVIDIA Sol-Engine Pillar-1 MAE gates).
 * Default-OFF switches H3_DIT_QKV_MERGE / H3_DIT_QKNORM_ROPE_FUSE gate the
 * experiments in probes; these reference helpers are pure math, no env reads.
 *
 * h3_dit_w4a8_fp_qkv_split3_f32: treat wqkv [3*n, k] as three stacked
 * [n, k] projections (q rows [0,n), k rows [n,2n), v rows [2n,3n)) and run
 * three narrow GEMMs y[m,3n] = x[m,k] @ W^T.  Because the split happens along
 * the output (N) axis, the per-row dot products are identical to one wide
 * merged GEMM (each output column is an independent dot), which is exactly
 * what the resident INT8 path already executes.  The probe uses this to
 * compute MAE(merged, split3) and prove it is 0 (FP32) — the FP math
 * equivalence gate required before INT8 evaluation.
 *
 * h3_dit_w4a8_fp_qknorm_rope_unfused_f32: two-pass reference that first
 * RMS-normalizes Q/K into a materialized intermediate buffer (the unfused
 * baseline the fused h3_dit_w4a8_qkv_rope_f32 replaces), then applies RoPE.
 * Used to compute MAE(fused, unfused) — expected ~1e-7 (single vs double
 * rounding on the normalized value). */
int h3_dit_w4a8_fp_qkv_split3_f32(const float *x, const float *wqkv,
                                  size_t m, size_t n, size_t k, float *y,
                                  char *error, size_t error_size);
int h3_dit_w4a8_fp_qknorm_rope_unfused_f32(
    const float *qkv, const float *q_norm, const float *k_norm,
    const float *rope_cos, const float *rope_sin, size_t seq, size_t heads,
    size_t head_dim, size_t rope_half, float epsilon, float *query, float *key,
    float *value, char *error, size_t error_size);
/* Opt-in wiring: 1 when H3_DIT_QKV_MERGE=1 / H3_DIT_QKNORM_ROPE_FUSE=1. */
int h3_dit_w4a8_fp_qkv_merge_enabled(void);
int h3_dit_w4a8_fp_qknorm_rope_fuse_enabled(void);

/* Naive SDPA: out[seq, heads*dim], scale typically 1/sqrt(head_dim). */
int h3_dit_w4a8_sdpa_f32(const float *query, const float *key,
                         const float *value, size_t seq, size_t heads,
                         size_t head_dim, float scale, float *out, char *error,
                         size_t error_size);

/* CPU SwiGLU: fused[row, 2*width] → out[row, width]
 * out = silu(gate) * up, gate=fused[...,0:width], up=fused[...,width:]. */
void h3_dit_w4a8_swiglu_f32(const float *fused, size_t rows, size_t width,
                            float *out);

/* Attention stub (legacy): qkv[m, 3*inner] Q|K|V contiguous thirds → V. */
int h3_dit_w4a8_qkv_take_v(const float *qkv, size_t m, size_t inner,
                           float *v_out, char *error, size_t error_size);

/* Full-ish single-block forward (CUDA linears + CPU non-quant ops).
 * x/y: [seq, HIDDEN]. modulation: [mod_rows, SLOTS, HIDDEN] flattened;
 * row_map length seq (NULL → all zeros). rope_cos/sin: [seq, ROPE_HALF].
 * use_cuda_linears=0 runs host INT8 GEMM via caller-supplied path? —
 * always uses CUDA resident GEMM when ctx is non-NULL.
 * stats may be NULL. */
int h3_dit_w4a8_block_forward_f32(
    h3_dit_w4a8 *ctx, unsigned block, const h3_dit_w4a8_block_aux *aux,
    const float *x, size_t seq, const float *modulation,
    const uint32_t *row_map, const float *rope_cos, const float *rope_sin,
    float *y, h3_dit_w4a8_block_stats *stats, char *error, size_t error_size);

/* Load patch / final / shared t_table + inv_freq. Caller frees with
 * h3_dit_w4a8_core_free. */
int h3_dit_w4a8_load_core(const h3_w4a8_store *store, h3_dit_w4a8_core *core,
                          char *error, size_t error_size);
int h3_dit_w4a8_load_core_hdr(const h3_st_header *hdr, h3_dit_w4a8_core *core,
                              char *error, size_t error_size);
int h3_dit_w4a8_load_core_int8(const h3_dit_int8_store *store,
                               h3_dit_w4a8_core *core, char *error,
                               size_t error_size);
void h3_dit_w4a8_core_free(h3_dit_w4a8_core *core);

/* Map sigma in [0,1] → t_table row (0 .. rows-1). */
size_t h3_dit_w4a8_t_row_from_sigma(float sigma, size_t t_table_rows);

/* Metal AdaLN visual_condition vs CUDA flow-σ.
 * Metal compares time = 1 - sigmas.video (clean end); fallback when
 * time >= 0.999. Operationally: active iff cond_seq>0 && sigma > 0.001.
 * Do not compare flow-σ against 0.999 (F2VA step0 σ=1 would drop cond). */
int h3_dit_w4a8_visual_cond_adaln_active(size_t cond_seq, float sigma);
/* 3 when active (4th modulation row = VIDEO slice @ time 0.999);
 * otherwise H3_DIT_W4A8_MOD_VIDEO. */
uint32_t h3_dit_w4a8_visual_cond_mod_row(size_t cond_seq, float sigma);
/* t_table row for Metal times[visual_condition]=0.999 (time axis). */
size_t h3_dit_w4a8_visual_cond_t_row(size_t t_table_rows);
/* Metal audio_condition: active when target audio still denoising. */
int h3_dit_w4a8_audio_cond_adaln_active(size_t audio_cond_seq,
                                        float audio_sigma);
uint32_t h3_dit_w4a8_audio_cond_mod_row(size_t audio_cond_seq, float audio_sigma,
                                        int visual_cond_active);
size_t h3_dit_w4a8_audio_cond_t_row(size_t t_table_rows);

/* Dense F32 linear: y[m,N] = x[m,K] @ W[N,K]^T + b[N] (b may be NULL). */
int h3_dit_w4a8_dense_f32(const float *x, const float *weight,
                          const float *bias, size_t m, size_t n, size_t k,
                          float *y, char *error, size_t error_size);

/* video latent [seq, VIDEO_PATCH] → hidden [seq, HIDDEN]. */
int h3_dit_w4a8_patch_video_f32(const h3_dit_w4a8_core *core, const float *x,
                                size_t seq, float *y, char *error,
                                size_t error_size);
/* cuBLAS patch (resident W); falls back to host if cuda unavailable. */
int h3_dit_w4a8_patch_video_cuda(h3_dit_w4a8 *ctx, const h3_dit_w4a8_core *core,
                                 const float *x, size_t seq, float *y,
                                 char *error, size_t error_size);
int h3_dit_w4a8_patch_audio_f32(const h3_dit_w4a8_core *core, const float *x,
                                size_t seq, float *y, char *error,
                                size_t error_size);
int h3_dit_w4a8_patch_audio_cuda(h3_dit_w4a8 *ctx, const h3_dit_w4a8_core *core,
                                 const float *x, size_t seq, float *y,
                                 char *error, size_t error_size);

/* final_layer AdaLN project: time_feat → modulation[FINAL_SLOTS*HIDDEN]. */
int h3_dit_w4a8_final_adaln_project_f32(const h3_dit_w4a8_core *core,
                                        const float *time_feat,
                                        float *modulation, char *error,
                                        size_t error_size);

/* hidden [seq, HIDDEN] → velocity [seq, VIDEO_PATCH] via final AdaLN+linear. */
int h3_dit_w4a8_final_video_f32(const h3_dit_w4a8_core *core, const float *x,
                                size_t seq, const float *time_feat, float *y,
                                char *error, size_t error_size);
int h3_dit_w4a8_final_video_cuda(h3_dit_w4a8 *ctx, const h3_dit_w4a8_core *core,
                                 const float *x, size_t seq,
                                 const float *time_feat, float *y, char *error,
                                 size_t error_size);
int h3_dit_w4a8_final_audio_f32(const h3_dit_w4a8_core *core, const float *x,
                                size_t seq, const float *time_feat, float *y,
                                char *error, size_t error_size);
int h3_dit_w4a8_final_audio_cuda(h3_dit_w4a8 *ctx, const h3_dit_w4a8_core *core,
                                 const float *x, size_t seq,
                                 const float *time_feat, float *y, char *error,
                                 size_t error_size);

/* Chain blocks [first, first+n). Loads per-block aux; uses shared t_table row.
 * x/y: [seq, HIDDEN]. Prefetch is caller's responsibility (or pass
 * auto_prefetch≠0). stats may be NULL. */
int h3_dit_w4a8_layers_forward_f32(
    h3_dit_w4a8 *ctx, unsigned first_block, unsigned n_blocks, size_t t_row,
    const float *x, size_t seq, const uint32_t *row_map, const float *rope_cos,
    const float *rope_sin, float *y, int auto_prefetch,
    h3_dit_w4a8_layers_stats *stats, char *error, size_t error_size);

/* Project TE / stub embeddings [tokens, TEXT_DIM] → [tokens, HIDDEN]
 * via condition_proj only (no token_refiner). */
int h3_dit_w4a8_project_text_f32(const h3_dit_w4a8_core *core,
                                 const float *text_emb, size_t tokens,
                                 float *text_hidden, char *error,
                                 size_t error_size);

/* Metal refine_text: condition_proj → 2× refiner blocks → final_norm.
 * Uses CUDA GEMM/SDPA when ctx!=NULL. H3_DIT_NO_REFINER=1 → project only. */
int h3_dit_w4a8_refine_text_f32(h3_dit_w4a8 *ctx, const h3_dit_w4a8_core *core,
                                const float *text_emb, size_t tokens,
                                float *text_hidden, char *error,
                                size_t error_size);

/* Video forward with optional text condition packed as leading tokens.
 * text_hidden may be NULL / text_seq=0 for unconditional.
 * latent/velocity: [video_seq, 96]. pos_* length video_seq (Metal float). */
int h3_dit_w4a8_forward_video_f32(
    h3_dit_w4a8 *ctx, const h3_dit_w4a8_core *core, unsigned first_block,
    unsigned n_blocks, float sigma, const float *latent, size_t video_seq,
    const float *text_hidden, size_t text_seq, const float *pos_t,
    const float *pos_h, const float *pos_w, float spatial_scale, float *velocity,
    int auto_prefetch, h3_dit_w4a8_forward_stats *stats, char *error,
    size_t error_size);

/* Pack [text | audio | video] (Metal segment order), emit video+audio velocity.
 * audio_latent / audio_velocity: packed rows [audio_seq, AUDIO_CH]
 *   (Metal h3_dit_pack_audio layout; audio_seq = 2 * audio_t).
 * a_pos_* length audio_seq; v_pos_* length video_seq.
 * sigma = video flow (shift 12); audio_sigma = audio flow (shift 3).
 * Matches Metal schedule_row_map + vLLM-Omni/SGLang dual schedulers. */
int h3_dit_w4a8_forward_av_f32(
    h3_dit_w4a8 *ctx, const h3_dit_w4a8_core *core, unsigned first_block,
    unsigned n_blocks, float sigma, float audio_sigma, const float *video_latent,
    size_t video_seq, const float *audio_latent, size_t audio_seq,
    const float *text_hidden, size_t text_seq, const float *v_pos_t,
    const float *v_pos_h, const float *v_pos_w, const float *a_pos_t,
    const float *a_pos_h, const float *a_pos_w, float spatial_scale,
    float *video_velocity, float *audio_velocity, int auto_prefetch,
    h3_dit_w4a8_forward_stats *stats, char *error, size_t error_size);

/* Pack [text | cond | audio_cond | audio | video] — Metal Ref2VA / keyframe
 * order. audio_cond_seq=0 && cond_seq=0 → 与 forward_av 逐 bit 相同。
 * cond / audio_cond 行参与 attention；velocity 只写 target video×96 与
 * audio×32（条件段不进 Euler）。 */
int h3_dit_w4a8_forward_avc_f32(
    h3_dit_w4a8 *ctx, const h3_dit_w4a8_core *core, unsigned first_block,
    unsigned n_blocks, float sigma, float audio_sigma,
    const float *video_latent, size_t video_seq,
    const float *audio_latent, size_t audio_seq,
    const float *cond_latent, size_t cond_seq,
    const float *audio_cond_latent, size_t audio_cond_seq,
    const float *text_hidden, size_t text_seq,
    const float *v_pos_t, const float *v_pos_h, const float *v_pos_w,
    const float *a_pos_t, const float *a_pos_h, const float *a_pos_w,
    const float *c_pos_t, const float *c_pos_h, const float *c_pos_w,
    const float *ac_pos_t, const float *ac_pos_h, const float *ac_pos_w,
    float spatial_scale,
    float *video_velocity, float *audio_velocity,
    int auto_prefetch, h3_dit_w4a8_forward_stats *stats,
    char *error, size_t error_size);

/* Pack/unpack [32,2,T] ↔ row-major [2*T, 32] (Metal audio DiT layout). */
int h3_dit_w4a8_pack_audio_latent(const float *latent_c2t, int audio_t,
                                  float *rows_seq32, char *error,
                                  size_t error_size);
int h3_dit_w4a8_unpack_audio_latent(const float *rows_seq32, int audio_t,
                                    float *latent_c2t, char *error,
                                    size_t error_size);

/* Euler: sample += (sigma - sigma_next) * velocity. */
int h3_dit_w4a8_euler_step_f32(float *sample, const float *velocity,
                               size_t count, float sigma, float sigma_next,
                               char *error, size_t error_size);

/* Official Metal-compatible Euler reuse schedule (see h3_dit_reuse_schedule).
 * Marks steps that must run a full DiT; skipped steps extrapolate velocity.
 * Returns selected count, or -1 on bad args. */
int h3_dit_w4a8_reuse_schedule(int steps, int reuse_interval, uint8_t *selected,
                               size_t selected_count);

/* Linear velocity extrapolation used by Metal Euler reuse (host f32). */
void h3_dit_w4a8_extrapolate_velocity_f32(float *out, const float *last,
                                          const float *previous, size_t count,
                                          float current_sigma, float last_sigma,
                                          float previous_sigma,
                                          int have_previous);

/* Mean absolute relative L1: sum|a-b| / (sum|b|+eps). Used by MagCache-lite. */
float h3_dit_w4a8_velocity_rel_l1_f32(const float *a, const float *b,
                                      size_t count);

/* Mean abs magnitude: sum|x|/count. MagCache-style magnitude probe. */
float h3_dit_w4a8_velocity_mean_abs_f32(const float *x, size_t count);

/* FNV-1a over float bit patterns (reproducible checksum). */
uint64_t h3_dit_w4a8_checksum_f32(const float *x, size_t n);

#ifdef __cplusplus
}
#endif

#endif
