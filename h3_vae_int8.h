#ifndef H3_VAE_INT8_H
#define H3_VAE_INT8_H

#include "h3_safetensors.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* E3: Video VAE INT8+scale minimal decode (host F32).
 * Checkpoint: I8 weight + F32 weight_scale (+ optional comfy_quant).
 * Weights are ConvRot-rotated (group=256); linears ConvRot acts (X@H) before
 * GEMM. H3_VAE_NO_CONVROT=1 disables. Full tiling still TODO. */

enum {
    H3_VAE_INT8_LATENT_CH = 24,
    H3_VAE_INT8_HIDDEN = 2048,
    H3_VAE_INT8_FFN = 8192,
    H3_VAE_INT8_INNER = 2048, /* 32*64 */
    H3_VAE_INT8_HEADS = 32,
    H3_VAE_INT8_HEAD_DIM = 64,
    H3_VAE_INT8_ROPE_HALF = 24, /* Metal video VAE: 3 axes × 8 freqs */
    H3_VAE_INT8_REGISTERS = 4,
    H3_VAE_INT8_OUTPUT_PATCH = 3072, /* 3*4*16*16 */
    H3_VAE_INT8_SPATIAL_RATIO = 16,
    H3_VAE_INT8_N_BLOCKS = 36
};

typedef struct {
    char path[512];
    h3_st_header header;
    h3_st_quant_info quant;
    /* Optional read-only mmap of the safetensors file (Linux/macOS). */
    int map_fd;
    const uint8_t *map;
    size_t map_len;
} h3_vae_int8_store;

typedef struct {
    char base_name[256];
    const h3_st_tensor *weight; /* I8 [N, K] */
    const h3_st_tensor *scale;  /* F32 [N] or [N,1] */
    const h3_st_tensor *bias;   /* optional F32 [N] */
    size_t n;
    size_t k;
} h3_vae_int8_linear;

typedef struct {
    float *rgb; /* frame-major interleaved RGB F32 in [0,1] */
    int frames;
    int height;
    int width;
    /* Allocation discipline for rgb: 0=malloc, 1=cudaHostAlloc(pinned).
     * Only set by the GPU unpatch D2H path; frames_free honors it. */
    int rgb_pinned;
} h3_vae_int8_frames;

typedef struct {
    double load_ms;
    double canvas_h2d_ms;  /* patch/canvas upload (0 if host-only) */
    double blocks_ms;      /* transformer blocks (+ temporal loop) */
    double unpatch_d2h_ms; /* canvas/result download + unpatch/proj host */
    double total_ms;
    unsigned n_blocks;
    int all_finite;
    char attn_backend[32]; /* effective: cudnn_batched|cudnn_loop|… */
} h3_vae_int8_decode_stats;

int h3_vae_int8_store_open(const char *path, h3_vae_int8_store *store,
                           char *error, size_t error_size);
void h3_vae_int8_store_free(h3_vae_int8_store *store);

/* Load residency: default ON (community VAE stays). Only
 * H3_VAE_WEIGHT_CACHE=0 disables. Tile batch default 4 (Comfy row). */
int h3_vae_int8_weight_cache_wanted(void);
int h3_vae_int8_tile_batch_want(void);
/* --repeat clip-gap VAE W: keep=0 (default, hot vae_load≈0),
 * device=1 (drop device W, keep host pack), full=2 (drop pack too).
 * H3_VAE_REPEAT_RELEASE=keep|device|full */
enum { H3_VAE_REPEAT_KEEP = 0, H3_VAE_REPEAT_DEVICE = 1, H3_VAE_REPEAT_FULL = 2 };
int h3_vae_int8_repeat_release_mode(void);
/* M4: F16 GEMM without F32↔F16 HBM casts. H3_VAE_GEMM_F16_DIRECT=0 rolls back. */
int h3_vae_int8_gemm_f16_direct(void);
/* M4: fuse bias+SiLU / bias+residual. H3_VAE_FUSED_EPILOGUE=0 rolls back. */
int h3_vae_int8_fused_epilogue(void);
/* M4: keep in-layer activations F16 (RMS/bias/SiLU F16-acc). Residual
 * stays F32. H3_VAE_ACT_F16=0 rolls back to per-GEMM HBM cast. */
int h3_vae_int8_act_f16(void);
/* M4 R10: unset H3_VAE_CUDA_GRAPH = 0 (do not follow FLASH_SDPA). */
int h3_vae_int8_cuda_graph_wanted(void);
int h3_vae_int8_store_copy(const h3_vae_int8_store *store, size_t file_offset,
                           void *dst, size_t nbytes, char *error,
                           size_t error_size);

/* Resolve absolute base (without ".weight"), e.g.
 * "decoder.transformer_blocks.0.attn.to_out". */
int h3_vae_int8_find_linear(const h3_vae_int8_store *store, const char *base,
                            h3_vae_int8_linear *out, char *error,
                            size_t error_size);

/* Dequant I8 * per-row scale → host F32[N*K] (W_rot). Caller frees. */
float *h3_vae_int8_dequant_f32(const h3_vae_int8_store *store,
                               const h3_vae_int8_linear *lin, char *error,
                               size_t error_size);

/* ConvRot group size for K, or 0 if disabled / not divisible. */
int h3_vae_int8_convrot_groupsize(const h3_vae_int8_store *store, size_t k);

/* y[m,N] = x[m,K] @ W[N,K]^T (+ optional bias). */
int h3_vae_int8_gemm_f32(const float *x, const float *w, size_t m, size_t n,
                         size_t k, float *y, char *error, size_t error_size);
int h3_vae_int8_linear_f32(const h3_vae_int8_store *store,
                           const h3_vae_int8_linear *lin, const float *x,
                           size_t m, float *y, char *error, size_t error_size);

/* Unpack DiT packed latent [tokens,96] (2x2×24) → CHW [24,T,H,W].
 * Pack order matches Metal h3_dit_patchify_video: c-major then (dy,dx).
 * tokens must equal T*(H/2)*(W/2); H,W even. Caller frees. */
float *h3_vae_int8_unpack_dit_latent_f32(const float *packed, int latent_t,
                                         int latent_h, int latent_w,
                                         char *error, size_t error_size);
/* Inverse of unpack: CHW [24,T,H,W] → packed [tokens,96]. out must hold
 * T*(H/2)*(W/2)*96 floats. */
int h3_vae_int8_pack_dit_latent_f32(const float *chw, int latent_t,
                                    int latent_h, int latent_w, float *out,
                                    char *error, size_t error_size);
/* In-place CHW denorm: x = x*std+mean per channel (once). */
void h3_vae_int8_denorm_latent_chw(float *chw, int latent_t, int latent_h,
                                   int latent_w,
                                   const float mean[H3_VAE_INT8_LATENT_CH],
                                   const float std_[H3_VAE_INT8_LATENT_CH]);
void h3_vae_int8_denorm_vec_f32(float *x,
                                const float mean[H3_VAE_INT8_LATENT_CH],
                                const float std_[H3_VAE_INT8_LATENT_CH],
                                size_t n);

/* Load checkpoint latents_mean/std (DiT latents are normalized; decode
 * denorms with x*std+mean before post_quant, matching Metal prepare_input). */
int h3_vae_int8_load_latent_stats(const h3_vae_int8_store *store,
                                  float mean[H3_VAE_INT8_LATENT_CH],
                                  float std_[H3_VAE_INT8_LATENT_CH],
                                  char *error, size_t error_size);

/* Metal h3_video_vae prepare_rope: patch tokens get (t,h,w) axes; remaining
 * sequence rows (registers/zero) get cos=1,sin=0.
 * cos/sin length = seq * H3_VAE_INT8_ROPE_HALF.
 * prepare_rope is tile-local (denom = latent_h/w). prepare_rope_ex places
 * the tile on a full canvas so spatial tiles share one [-1,1] map.
 * Comfy MiniMaxH3VideoVAE.create_token_ids uses the CURRENT tensor H/W
 * (tile-local). Canvas denom is OOD and prints the 16px ViT patch grid. */
int h3_vae_int8_prepare_rope(int t_len, int latent_h, int latent_w,
                             size_t n_patches, size_t seq, float *cos,
                             float *sin, char *error, size_t error_size);
int h3_vae_int8_prepare_rope_ex(int t_len, int latent_h, int latent_w,
                                int origin_h, int origin_w, int full_h,
                                int full_w, size_t n_patches, size_t seq,
                                float *cos, float *sin, char *error,
                                size_t error_size);

/* Phase-A profile: cumulative RoPE table build time and call count.
 * Enable with H3_ROPE_PROFILE=1. */
void h3_vae_int8_rope_profile_get(double *ms, unsigned long *calls);
void h3_vae_int8_rope_profile_reset(void);

/* Minimal decode: post_quant → embed → N INT8 blocks → norm_out → proj_out
 * → RGB frames. latent_chw is [24,T,H,W] contiguous C-major.
 * n_blocks=0 skips transformers (heads only). stats may be NULL.
 * Caller frees frames with h3_vae_int8_frames_free. */
int h3_vae_int8_decode_f32(const h3_vae_int8_store *store,
                           const float *latent_chw, int latent_t, int latent_h,
                           int latent_w, unsigned n_blocks,
                           h3_vae_int8_frames *frames,
                           h3_vae_int8_decode_stats *stats, char *error,
                           size_t error_size);

void h3_vae_int8_frames_free(h3_vae_int8_frames *frames);
/* Register the free fn used for rgb_pinned=1 (cudaFreeHost). Set by the
 * CUDA decode side; default NULL → plain free (host path). */
void h3_vae_int8_set_rgb_host_free(void (*fn)(void *));

/* Soften 16×16 VAE patch seams on packed RGB [frames,H,W,3] in [0,1].
 * Strength from H3_VAE_DEBLOCK (0 disables; default 0.28). */
void h3_vae_int8_deblock16_rgb(float *rgb, int frames, int height, int width);

/* Write first frame (or frame_index) as binary P6 PPM. */
int h3_vae_int8_write_ppm(const h3_vae_int8_frames *frames, int frame_index,
                          const char *path, char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
