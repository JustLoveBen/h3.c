#ifndef H3_LATENT_UPSCALER_H
#define H3_LATENT_UPSCALER_H

#include "h3_gpu_cuda.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* LBH-123-AI LatentResizer3D (not paper SelfLift-rich). Gold never calls this.
 * z_low / z_high: CHW [24,T,h,w] / [24,T,H,W] in the SAME space Comfy
 * MiniMaxH3Video.process_out/in expose (identity, scale=1) — DiT-normalized.
 * Forward still applies learned_latent_lift's (x-mean)/std and *std+mean. */

typedef struct h3_latent_upscaler h3_latent_upscaler;

int h3_latent_upscaler_open(const char *path, h3_latent_upscaler **out,
                            char *err, size_t err_size);
void h3_latent_upscaler_close(h3_latent_upscaler *u);

/* Drop device feature maps / slots. Weights stay in host RAM for the next
 * forward. Call before TAE when DiT remains resident. */
void h3_latent_upscaler_release_device(h3_latent_upscaler *u);

/* Probe only: 1 if conv_in is 24→512 3x3x3. Hard-fail generate if 0. */
int h3_latent_upscaler_probe(const h3_latent_upscaler *u, int *in_ch,
                             int *hid, char *err, size_t err_size);

int h3_latent_upscaler_forward(h3_latent_upscaler *u, h3_cuda *cuda,
                               const float *z_low, int t, int h, int w,
                               int out_h, int out_w, const float *mean24,
                               const float *std24, float *z_high, char *err,
                               size_t err_size);

/* Layer-by-layer F32 oracle (P0). `z_low` is VAE-native CHW [24,T,h,w].
 * stage: 0=(x-mean)/std  1=conv_in  2=first in_block  3=all in_blocks
 *        4=trilinear  5=full lift (incl. *std+mean, 24ch).
 * Stages 1–4 write 512-ch feature maps. */
int h3_latent_upscaler_oracle(h3_latent_upscaler *u, h3_cuda *cuda, int stage,
                              const float *z_low, int t, int h, int w,
                              int dst_h, int dst_w, const float *mean24,
                              const float *std24, float *out, int *got_c,
                              int *got_t, int *got_h, int *got_w, char *err,
                              size_t err_size);

/* First 8 conv_in.weight values after F32 unfold (layout check). */
int h3_latent_upscaler_conv_in_head(const h3_latent_upscaler *u, float *out8);

#ifdef __cplusplus
}
#endif

#endif
