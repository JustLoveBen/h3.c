#ifndef H3_DIT_INT8_H
#define H3_DIT_INT8_H

#include "h3_safetensors.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Comfy-Org / Kijai INT8 ConvRot DiT (次选主题跟随路径).
 * Checkpoint: pruned_int8_convrot — I8 weight + F32 weight_scale + U8 comfy_quant
 * (same layout family as video VAE int8_scale_convrot_like; NOT W4A8 codebook).
 * Host open / find / dequant first; CUDA GEMM path is follow-on. */

enum {
    H3_DIT_INT8_HIDDEN = 5376,
    H3_DIT_INT8_N_BLOCKS = 50,
    H3_DIT_INT8_DEFAULT_CONVROT_GS = 256
};

typedef struct h3_dit_int8_store h3_dit_int8_store;

typedef struct {
    char base_name[192];
    const h3_st_tensor *weight; /* I8 [N, K] full K (not nibble-packed) */
    const h3_st_tensor *scale;  /* F32 [N] or [N,1] */
    const h3_st_tensor *comfy_quant; /* optional U8 blob */
    size_t n;
    size_t k;
} h3_dit_int8_linear;

#define H3_DIT_INT8_SUFFIX_QKV "attn.qkv_proj"
#define H3_DIT_INT8_SUFFIX_OUT "attn.out_proj"
#define H3_DIT_INT8_SUFFIX_FC1 "mlp.fc1"
#define H3_DIT_INT8_SUFFIX_FC2 "mlp.fc2"

h3_dit_int8_store *h3_dit_int8_store_open(const char *path, char *error,
                                          size_t error_size);
void h3_dit_int8_store_free(h3_dit_int8_store *store);

const h3_st_header *h3_dit_int8_store_header(const h3_dit_int8_store *store);
const h3_st_quant_info *h3_dit_int8_store_quant(const h3_dit_int8_store *store);

/* Reject W4A8 codebook files; require I8 + weight_scale linears. */
int h3_dit_int8_store_is_int8_convrot(const h3_dit_int8_store *store);

int h3_dit_int8_find_linear(const h3_dit_int8_store *store, const char *base,
                            h3_dit_int8_linear *out, char *error,
                            size_t error_size);

int h3_dit_int8_find_block_linear(const h3_dit_int8_store *store,
                                  unsigned block_index, const char *suffix,
                                  h3_dit_int8_linear *out, char *error,
                                  size_t error_size);

const h3_st_tensor *h3_dit_int8_find(const h3_dit_int8_store *store,
                                     const char *name);

int h3_dit_int8_read_tensor(const h3_dit_int8_store *store,
                            const h3_st_tensor *tensor, void *dst, size_t bytes,
                            char *error, size_t error_size);

/* Dequant I8 * per-row scale → host F32[N*K]. Caller frees. */
float *h3_dit_int8_dequant_f32(const h3_dit_int8_store *store,
                               const h3_dit_int8_linear *lin, char *error,
                               size_t error_size);

/* ConvRot group size for K, or 0 if disabled / not divisible. */
int h3_dit_int8_convrot_groupsize(const h3_dit_int8_store *store, size_t k);

#ifdef __cplusplus
}
#endif

#endif
