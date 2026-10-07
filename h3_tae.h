#ifndef H3_TAE_H
#define H3_TAE_H

#include "h3_vae_int8.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Tiny autoencoder decode — turbo/nvfp4 opt-in only. Does not replace official
 * fp16 VAE tile+deblock (h3_vae_int8_cuda). madebyollin TAEHV / taeh3.
 * Quality is lossy (texture smear / color shift). Gold path must stay official.
 *
 * Latent: official unpack [24,T,H,W], same denorm (x*std+mean) as official VAE.
 * Spatial 16× (864×480 from 54×30). Temporal 4× then frames_to_trim=3, take
 * want_frames (mux N). Mismatched C/scale/ratio → hard fail, never official.
 */
int h3_tae_allow_opt_in(int non_gold, const char *vae_decode,
                        const char *tae_path, char *err, size_t err_size);

int h3_tae_open(const char *path, void **ctx, char *err, size_t err_size);
void h3_tae_close(void *ctx);
/* Optional: bind generate's h3_cuda* so decode uses GPU. No-op if cuda=NULL. */
int h3_tae_bind_cuda(void *ctx, void *cuda, char *err, size_t err_size);
int h3_tae_decode_f32(void *ctx, const float *latent, int t, int h, int w,
                      uint8_t *rgb_out, char *err, size_t err_size);
/* latent_chw: official unpack [24,T,H,W]. want_frames = mux N (e.g. 120). */
int h3_tae_decode_frames(void *ctx, const float *latent_chw, int t, int h,
                         int w, int want_frames, h3_vae_int8_frames *frames,
                         char *err, size_t err_size);

#ifdef __cplusplus
}
#endif

#endif
