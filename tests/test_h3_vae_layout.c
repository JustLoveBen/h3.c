/* CPU: unpack↔pack identity; denorm x*std+mean is applied once. */
#include "h3_vae_int8.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
static void expect(int c, const char *m) {
    if (!c) {
        fprintf(stderr, "FAIL %s\n", m);
        fails++;
    }
}

int main(void) {
    const int t = 2, h = 4, w = 6;
    const size_t tokens =
        (size_t)t * (size_t)(h / 2) * (size_t)(w / 2);
    const size_t packed_n = tokens * 96;
    const size_t chw_n =
        (size_t)H3_VAE_INT8_LATENT_CH * (size_t)t * (size_t)h * (size_t)w;
    float *packed = (float *)malloc(packed_n * sizeof(float));
    float *packed2 = (float *)malloc(packed_n * sizeof(float));
    float *chw = NULL;
    float *den1 = (float *)malloc(chw_n * sizeof(float));
    float *den2 = (float *)malloc(chw_n * sizeof(float));
    float mean[H3_VAE_INT8_LATENT_CH];
    float std_[H3_VAE_INT8_LATENT_CH];
    char err[128];
    size_t i;
    int c;

    if (!packed || !packed2 || !den1 || !den2) return 1;
    for (i = 0; i < packed_n; i++) packed[i] = (float)((int)i % 17) * 0.25f;
    chw = h3_vae_int8_unpack_dit_latent_f32(packed, t, h, w, err, sizeof(err));
    expect(chw != NULL, "unpack");
    expect(h3_vae_int8_pack_dit_latent_f32(chw, t, h, w, packed2, err,
                                           sizeof(err)) == 1,
           "pack");
    expect(memcmp(packed, packed2, packed_n * sizeof(float)) == 0,
           "unpack/pack identity");

    for (c = 0; c < H3_VAE_INT8_LATENT_CH; c++) {
        mean[c] = 0.1f * (float)c;
        std_[c] = 1.0f + 0.05f * (float)c;
    }
    memcpy(den1, chw, chw_n * sizeof(float));
    memcpy(den2, chw, chw_n * sizeof(float));
    h3_vae_int8_denorm_latent_chw(den1, t, h, w, mean, std_);
    h3_vae_int8_denorm_latent_chw(den2, t, h, w, mean, std_);
    h3_vae_int8_denorm_latent_chw(den2, t, h, w, mean, std_);
    {
        size_t src =
            ((size_t)3 * (size_t)t + 1) * (size_t)h * (size_t)w + 2 * (size_t)w +
            1;
        float raw = chw[src];
        float once = raw * std_[3] + mean[3];
        expect(fabsf(den1[src] - once) < 1e-6f, "denorm once math");
        expect(fabsf(den1[src] - den2[src]) > 1e-5f, "twice != once");
    }

    free(packed);
    free(packed2);
    free(chw);
    free(den1);
    free(den2);
    return fails ? 1 : (printf("test_h3_vae_layout: PASS\n"), 0);
}
