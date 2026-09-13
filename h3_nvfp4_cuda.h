#ifndef H3_NVFP4_CUDA_H
#define H3_NVFP4_CUDA_H

#include "h3_gpu_cuda.h"
#include "h3_nvfp4.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Bind TE NVFP4 linears to cuBLAS (default on for CUDA generate/probes).
 * CUDA path keeps x/y/W in grow-only device workspaces. Text encode bulk-
 * uploads packed NVFP4+scales to a GPU arena (A1; H3_TE_PACKED_RESIDENT=0
 * reverts to per-linear H2D). Host packed cache remains the fallback.
 * Env H3_NVFP4_HOST_GEMM=1 forces host even after enable.
 * Env H3_NVFP4_DISABLE_DEV_CACHE=1 skips the host scale/packed cache.
 * Env H3_NVFP4_HOST_ACT=1 forces host activation path (no device TE layers).
 * Env H3_NVFP4_DEV_ACT=0 disables device-resident text TE (default is on).
 * Env H3_TE_PACKED_RESIDENT=0 disables packed GPU arena (default on).
 * Env H3_TE_MMAP_POPULATE=0 skips MAP_POPULATE at TE open (default on).
 * Env H3_TE_PREFAULT=0 skips sync page-touch before packed H2D (default on). */
int h3_nvfp4_cuda_enable(h3_cuda *cuda, char *error, size_t error_size);
void h3_nvfp4_cuda_disable(void);
int h3_nvfp4_cuda_enabled(void);
void h3_nvfp4_cuda_cache_clear(void);

/* One-shot: run after TE packed GPU upload, before TE layers (disk free). */
void h3_nvfp4_cuda_set_after_packed_hook(void (*fn)(void *), void *user);

/* Device-resident text TE layers (no vision / mRoPE). Default on;
 * H3_NVFP4_HOST_ACT=1 reverts. hidden is host [tokens, TEXT_DIM] in/out.
 * Returns 0 when disabled / unavailable (caller uses host act path). */
int h3_nvfp4_cuda_encode_text_layers(const h3_nvfp4_store *store, float *hidden,
                                     size_t tokens, unsigned n_layers,
                                     char *error, size_t error_size);

/* Causal GQA via bound CUDA handle (cuDNN fused + naive GPU fallback).
 * Returns 0 if CUDA TE is disabled (caller should use host GQA). */
int h3_nvfp4_cuda_gqa_causal(const float *q, const float *k, const float *v,
                             size_t tokens, size_t q_heads, size_t kv_heads,
                             size_t head_dim, float *out, char *error,
                             size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
