#ifndef H3_W4A8_H
#define H3_W4A8_H

#include "h3_safetensors.h"

#include <stddef.h>
#include <stdint.h>

/* P3: single-file W4A8 DiT probe / load_block-direction accessors.
 * Does not implement ConvRot, INT8 GEMM, or end-to-end generate (P4). */

typedef struct h3_w4a8_store h3_w4a8_store;

/* One asym_w4a8_int8 linear: packed weight + codebook + channel/rel scales. */
typedef struct {
    char base_name[192];
    const h3_st_tensor *weight;           /* I8 packed */
    const h3_st_tensor *weight_codebook;  /* F32[16] */
    const h3_st_tensor *weight_s_channel; /* F32 */
    const h3_st_tensor *weight_s_rel;     /* F8_E4M3 */
} h3_w4a8_quad;

/* Suffixes matching h3_dit.c load_block linearized weights. */
#define H3_W4A8_SUFFIX_QKV "attn.qkv_proj"
#define H3_W4A8_SUFFIX_OUT "attn.out_proj"
#define H3_W4A8_SUFFIX_FC1 "mlp.fc1"
#define H3_W4A8_SUFFIX_FC2 "mlp.fc2"

h3_w4a8_store *h3_w4a8_store_open(const char *path, char *error,
                                  size_t error_size);
void h3_w4a8_store_free(h3_w4a8_store *store);

const h3_st_header *h3_w4a8_store_header(const h3_w4a8_store *store);
const h3_st_quant_info *h3_w4a8_store_quant(const h3_w4a8_store *store);

/* Resolve blocks.{i}.{suffix} quadruplet (suffix without ".weight"). */
int h3_w4a8_find_block_linear(const h3_w4a8_store *store, unsigned block_index,
                              const char *suffix, h3_w4a8_quad *out,
                              char *error, size_t error_size);

/* Resolve an absolute base name such as "blocks.0.attn.qkv_proj". */
int h3_w4a8_find_quad(const h3_w4a8_store *store, const char *base_name,
                      h3_w4a8_quad *out, char *error, size_t error_size);

/* Copy one tensor payload into a host buffer (exact nbytes). */
int h3_w4a8_read_tensor(const h3_w4a8_store *store, const h3_st_tensor *tensor,
                        void *dst, size_t bytes, char *error,
                        size_t error_size);

/* Convenience: allocate + read; caller frees with free(). */
void *h3_w4a8_dup_tensor(const h3_w4a8_store *store, const h3_st_tensor *tensor,
                         size_t *out_bytes, char *error, size_t error_size);

/* Lookup by absolute tensor name (e.g. "blocks.0.norm1.weight"). */
const h3_st_tensor *h3_w4a8_find(const h3_w4a8_store *store, const char *name);

/* Load BF16 / F16 / F32 tensor as host F32; caller frees with free().
 * out_elements receives element count; out_dtype may be NULL. */
float *h3_w4a8_load_f32(const h3_w4a8_store *store, const char *name,
                        size_t *out_elements, h3_dtype *out_dtype, char *error,
                        size_t error_size);

#endif
