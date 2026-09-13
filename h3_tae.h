#ifndef H3_TAE_H
#define H3_TAE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Tiny autoencoder decode — turbo opt-in only. Does not replace official
 * fp16 VAE tile+deblock (h3_vae_int8_cuda). P1: stub unless weights land.
 */
int h3_tae_open(const char *path, void **ctx, char *err, size_t err_size);
void h3_tae_close(void *ctx);
int h3_tae_decode_f32(void *ctx, const float *latent, int t, int h, int w,
                      uint8_t *rgb_out, char *err, size_t err_size);

#ifdef __cplusplus
}
#endif

#endif
