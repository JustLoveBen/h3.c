#ifndef H3_NVFP4_H
#define H3_NVFP4_H

#include "h3_multimodal_cuda.h"
#include "h3_safetensors.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* E1/G1: NVFP4-AWQ-like TE helpers.
 * Layout (Comfy Qwen3VL nvfp4):
 *   linear weight U8[N, K/2] + weight_scale F8_E4M3[N, groups] +
 *   weight_scale_2 F32 scalar
 *   optional AWQ smooth: pre_quant_scale BF16[K] (o_proj + down_proj only;
 *   Comfy ops: input = input * pre_quant_scale before matmul)
 *   embed_tokens I8[V, D] + weight_scale F32[V,1]
 *   language layers: model.layers.{i}.* (50 layers in released file)
 * Configurable-layer transformer encode is implemented; full 50L is slow on
 * host (chunked NVFP4 dequant+GEMM) but CUDA cuBLAS path makes ≤8L@16tok
 * product-viable. CUDA token ids via h3_token_ids (file/canary); HF
 * tokenizer.json BPE / visual tower still Metal-only. */

enum {
    H3_NVFP4_TEXT_DIM = 5120,
    H3_NVFP4_INTERMEDIATE = 25600,
    H3_NVFP4_QUERY_HEADS = 64,
    H3_NVFP4_KV_HEADS = 8,
    H3_NVFP4_HEAD_DIM = 128,
    H3_NVFP4_QUERY_DIM = H3_NVFP4_QUERY_HEADS * H3_NVFP4_HEAD_DIM, /* 8192 */
    H3_NVFP4_KV_DIM = H3_NVFP4_KV_HEADS * H3_NVFP4_HEAD_DIM,       /* 1024 */
    H3_NVFP4_N_LAYERS = 50
};

typedef struct {
    char path[512];
    h3_st_header header;
    h3_st_quant_info quant;
    /* Optional read-only mmap of the safetensors file (Linux/macOS). */
    int map_fd;
    const uint8_t *map;
    size_t map_len;
} h3_nvfp4_store;

typedef struct {
    char base_name[256];
    const h3_st_tensor *weight;  /* U8 [N, K_half] */
    const h3_st_tensor *scale;   /* F8_E4M3 [N, groups] */
    const h3_st_tensor *scale_2; /* F32 [] or [1] */
    /* Optional ModelOpt AWQ smooth scale BF16[K]; NULL if absent. */
    const h3_st_tensor *pre_quant_scale;
    size_t n;
    size_t k;      /* logical = 2 * K_half */
    size_t groups; /* scale.shape[1] */
    size_t group_size;
} h3_nvfp4_linear;

typedef struct {
    const h3_st_tensor *weight; /* I8 [V, D] */
    const h3_st_tensor *scale;  /* F32 [V] or [V,1] */
    size_t vocab;
    size_t dim;
} h3_nvfp4_embed;

int h3_nvfp4_store_open(const char *path, h3_nvfp4_store *store, char *error,
                        size_t error_size);
void h3_nvfp4_store_free(h3_nvfp4_store *store);

int h3_nvfp4_find_linear(const h3_nvfp4_store *store, const char *base,
                         h3_nvfp4_linear *out, char *error, size_t error_size);

/* Resolve model.embed_tokens (I8 + F32 scale). */
int h3_nvfp4_find_embed(const h3_nvfp4_store *store, h3_nvfp4_embed *out,
                        char *error, size_t error_size);

/* FP4 E2M1 nibble → F32 (OCP-style table). */
float h3_fp4_e2m1_to_f32(uint8_t nibble);

/* Unswizzle on-disk NVFP4 weight_scale (cuBLAS tiled) → logical [N, groups]. */
int h3_nvfp4_unswizzle_scales_u8(const uint8_t *blocked, size_t num_rows,
                                 size_t num_cols, uint8_t *out, char *error,
                                 size_t error_size);

/* Dequant first `max_rows` rows (0 = all) to F32[rows*K]. Caller frees. */
float *h3_nvfp4_dequant_rows_f32(const h3_nvfp4_store *store,
                                 const h3_nvfp4_linear *lin, size_t max_rows,
                                 size_t *out_rows, char *error,
                                 size_t error_size);

/* Lookup embed rows for token_ids → F32[tokens, dim]. Caller frees. */
float *h3_nvfp4_embed_lookup_f32(const h3_nvfp4_store *store,
                                 const h3_nvfp4_embed *emb,
                                 const int32_t *token_ids, size_t tokens,
                                 char *error, size_t error_size);

/* Synthetic fixed-length text condition [tokens, TEXT_DIM] (no TE weights).
 * Deterministic from seed. Caller frees. */
float *h3_nvfp4_synth_condition_f32(size_t tokens, uint64_t seed, char *error,
                                    size_t error_size);

/* Host GEMM: y[m,N] = x[m,K] @ W[N,K]^T. */
int h3_nvfp4_gemm_f32(const float *x, const float *w, size_t m, size_t n,
                      size_t k, float *y, char *error, size_t error_size);

/* Minimal TE path smoke: embed lookup (or synth) → optional one NVFP4 linear
 * applied as Y = X @ W[:out_dim,:]^T truncated to TEXT_DIM.
 * If apply_linear_base is NULL, returns embed/synth only.
 * out_tokens receives token count; caller frees returned [tokens, TEXT_DIM]. */
float *h3_nvfp4_make_condition_f32(const h3_nvfp4_store *store,
                                   const int32_t *token_ids, size_t tokens,
                                   uint64_t synth_seed,
                                   const char *apply_linear_base,
                                   size_t *out_tokens, char *error,
                                   size_t error_size);

typedef struct {
    unsigned n_layers;
    double embed_ms;
    double layers_ms;
    double total_ms;
    int all_finite;
    int used_cuda;     /* 1 if linear backend was CUDA for this encode */
    int embed_cache_hit;
} h3_nvfp4_encode_stats;

/* Copy `nbytes` from store file at absolute file_offset (mmap or pread). */
int h3_nvfp4_store_copy(const h3_nvfp4_store *store, size_t file_offset,
                        void *dst, size_t nbytes, char *error,
                        size_t error_size);

/* Optional accelerated linear backend (cuBLAS). NULL restores host GEMM. */
typedef int (*h3_nvfp4_linear_fn)(void *ctx, const h3_nvfp4_store *store,
                                  const h3_nvfp4_linear *lin, const float *x,
                                  size_t m, float *y, char *error,
                                  size_t error_size);
void h3_nvfp4_set_linear_backend(void *ctx, h3_nvfp4_linear_fn fn);

/* Chunked NVFP4 linear: y[m,N] = x[m,K] @ W[N,K]^T (streams dequant rows).
 * Uses CUDA backend when set via h3_nvfp4_set_linear_backend. */
int h3_nvfp4_linear_f32(const h3_nvfp4_store *store, const h3_nvfp4_linear *lin,
                        const float *x, size_t m, float *y, char *error,
                        size_t error_size);

/* Drop process-local embed lookup cache (path+ids keyed). */
void h3_nvfp4_embed_cache_clear(void);
void h3_nvfp4_cuda_cache_clear(void);

/* G1: real Qwen language transformer path (not embed-only stub).
 * Runs layers [0, n_layers). n_layers=0 → embed only; >50 clamped to 50.
 * Env H3_NVFP4_TE_LAYERS overrides n_layers when set (>0).
 * Returns [tokens, TEXT_DIM]; caller frees. stats may be NULL. */
float *h3_nvfp4_encode_layers_f32(const h3_nvfp4_store *store,
                                  const int32_t *token_ids, size_t tokens,
                                  unsigned n_layers,
                                  h3_nvfp4_encode_stats *stats, char *error,
                                  size_t error_size);

/* No span → identical to h3_nvfp4_encode_layers_f32 (G1 path unchanged).
 * With spans: replace inner IMAGE_PAD rows with merged, then add deepstack
 * after language layers 0/1/2 (Metal h3_text_encode_multimodal_bf16). */
float *h3_nvfp4_encode_layers_multimodal_f32(
    const h3_nvfp4_store *store,
    const int32_t *token_ids, size_t tokens,
    const h3_cuda_vision_span *spans, size_t span_count,
    const float *merged,
    const float *deepstack[3],
    unsigned n_layers,
    h3_nvfp4_encode_stats *stats, char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
