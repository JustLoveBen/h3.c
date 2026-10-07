#ifndef H3_VAE_INT8_CUDA_H
#define H3_VAE_INT8_CUDA_H

#include "h3_gpu_cuda.h"
#include "h3_vae_int8.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* CUDA-accelerated VAE INT8 decode (main path):
 * - denorm latents_mean/std (Metal prepare_input)
 * - linears via h3_cuda_gemm_f32 after host dequant (or resident weights)
 * - attention via interleaved video QKV RoPE + h3_cuda_sdpa_f16
 * - transformer blocks run on temporal chunks (H3_VAE_TEMPORAL_CHUNK)
 * Does not replace host h3_vae_int8_decode_f32. Metal untouched. */
int h3_vae_int8_decode_cuda(h3_cuda *cuda, const h3_vae_int8_store *store,
                            const float *latent_chw, int latent_t, int latent_h,
                            int latent_w, unsigned n_blocks,
                            h3_vae_int8_frames *frames,
                            h3_vae_int8_decode_stats *stats, char *error,
                            size_t error_size);

/* Dequant-once weight pack for all blocks' linears + norms.
 * Default CUDA path: mmap I8 → device row-scale dequant → resident F32 W
 * (no 7GB host F32). H3_VAE_HOST_DEQUANT=1 forces the old host loop.
 * H3_VAE_WEIGHT_CACHE default ON (set 0 to drop the pack after decode).
 * H3_VAE_CUDA_W_RESIDENT=0 keeps host W and re-uploads per GEMM. */
/* Drop mmap cudaHostRegister. Call before h3_vae_int8_store_free. */
void h3_vae_int8_cuda_hostreg_release(void);

/* Pin store mmap for DMA H2D. No device W alloc. Safe to call from a
 * worker thread after cudaSetDevice (R11 prewarm). 1 if already pinned
 * or just registered. H3_VAE_HOSTREG=0 → 0. */
int h3_vae_int8_cuda_hostreg_prepare(const h3_vae_int8_store *store,
                                     char *error, size_t error_size);

typedef struct h3_vae_int8_cuda_weights h3_vae_int8_cuda_weights;

int h3_vae_int8_cuda_weights_load(const h3_vae_int8_store *store,
                                  unsigned n_blocks,
                                  h3_vae_int8_cuda_weights **out,
                                  double *load_ms, char *error,
                                  size_t error_size);
void h3_vae_int8_cuda_weights_free(h3_vae_int8_cuda_weights *w);

/* Drop cached GPU VAE W + grow-only scratch so the next DiT clip can
 * reload INT8. Keeps the host pack + mmap pin (hostreg). Safe no-op if
 * nothing is cached. Alias of release_device. */
void h3_vae_int8_cuda_release_cached(void);
void h3_vae_int8_cuda_release_device(void);
/* Drop host pack + device W + scratch. mmap pin stays. */
void h3_vae_int8_cuda_release_full(void);
/* 1 if cached device W is live (hot load can be ≈0). */
int h3_vae_int8_cuda_device_resident(void);
int h3_vae_int8_cuda_hostreg_ok(void);

int h3_vae_int8_decode_cuda_ex(h3_cuda *cuda, const h3_vae_int8_store *store,
                               h3_vae_int8_cuda_weights *weights,
                               const float *latent_chw, int latent_t,
                               int latent_h, int latent_w, unsigned n_blocks,
                               h3_vae_int8_frames *frames,
                               h3_vae_int8_decode_stats *stats, char *error,
                               size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
