#ifndef H3_DIT_NVFP4_H
#define H3_DIT_NVFP4_H

#include "h3_nvfp4.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Community pruned_nvfp4 DiT (vonkaiser/lilcheaty Comfy). Not gold.
 * Layout: U8 nibble weight + F8_E4M3 scale + F32 scale_2, same as TE.
 * Probe failure must hard-exit — never silent-fallback to INT8 ConvRot. */

#define H3_DIT_NVFP4_SUFFIX_QKV "attn.qkv_proj"
#define H3_DIT_NVFP4_SUFFIX_OUT "attn.out_proj"
#define H3_DIT_NVFP4_SUFFIX_FC1 "mlp.fc1"
#define H3_DIT_NVFP4_SUFFIX_FC2 "mlp.fc2"

/* 1 if path/header is NVFP4 DiT (qkv U8+E4M3+scale_2). 0 + err otherwise. */
int h3_dit_nvfp4_probe(const h3_nvfp4_store *store, char *error,
                       size_t error_size);

int h3_dit_nvfp4_path_hints(const char *path);

int h3_dit_nvfp4_find_block_linear(const h3_nvfp4_store *store,
                                   unsigned block_index, const char *suffix,
                                   h3_nvfp4_linear *out, char *error,
                                   size_t error_size);

/* Dequant NVFP4 → F32 → per-row INT8 (amax/127). Caller frees *out_i8 and
 * *out_scale (n floats). Logical K = 2 * packed K. */
int h3_dit_nvfp4_dequant_to_i8(const h3_nvfp4_store *store,
                               const h3_nvfp4_linear *lin, int8_t **out_i8,
                               float **out_scale, size_t *out_n, size_t *out_k,
                               char *error, size_t error_size);

/* INT8 sidecar next to the NVFP4 file (not gold). Dirty key = miss, recompute.
 * Never silent-fallback to INT8 ConvRot gold weights. Disable: H3_NVFP4_INT8_SIDECAR=0 */
int h3_dit_nvfp4_sidecar_dir(const char *nvfp4_path, char *out_dir, size_t n);
int h3_dit_nvfp4_sidecar_try_load(const char *nvfp4_path, unsigned block,
                                  const char *kind_name, size_t n, size_t k,
                                  int8_t *i8, float *scales);
int h3_dit_nvfp4_sidecar_save(const char *nvfp4_path, unsigned block,
                              const char *kind_name, size_t n, size_t k,
                              const int8_t *i8, const float *scales,
                              char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
