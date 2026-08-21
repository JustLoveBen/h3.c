#ifndef H3_VAE_ENCODE_F32_H
#define H3_VAE_ENCODE_F32_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int time;    /* h3_video_encoder_latent_t(frames) = ceil(frames/4) */
    int height;  /* pixels/16 */
    int width;
    float *values; /* channel-major [24,time,height,width] */
} h3_vae_latent_f32;

int h3_vae_encode_f32(
    const char *vae_path,
    const float *rgb01_chtw, int frames, int height, int width,
    h3_vae_latent_f32 *out,
    char *error, size_t error_size);
void h3_vae_latent_f32_free(h3_vae_latent_f32 *latent);

/* 与 Metal h3_dit_patchify_video 同一循环顺序：t, h+=2, w+=2, c, dh, dw */
int h3_vae_patchify_video_96(const float *latent_24thw, int time,
                             int height, int width, float *rows_seq96,
                             size_t row_elements);

/* generate 路径调用；不要写进 encoder */
void h3_vae_augment_0p999(float *rows, size_t count, uint64_t seed);

#ifdef __cplusplus
}
#endif

#endif
