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
 * H3_VAE_WEIGHT_CACHE=1 keeps the pack across decode() calls (opt-in VRAM).
 * H3_VAE_CUDA_W_RESIDENT=0 keeps host W and re-uploads per GEMM. */
typedef struct h3_vae_int8_cuda_weights h3_vae_int8_cuda_weights;

int h3_vae_int8_cuda_weights_load(const h3_vae_int8_store *store,
                                  unsigned n_blocks,
                                  h3_vae_int8_cuda_weights **out,
                                  double *load_ms, char *error,
                                  size_t error_size);
void h3_vae_int8_cuda_weights_free(h3_vae_int8_cuda_weights *w);

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
