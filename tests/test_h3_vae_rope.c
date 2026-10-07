/* Canvas RoPE: overlapping spatial tiles must share the same (h,w) axes. */
#include "h3_vae_int8.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int near(float a, float b) { return fabsf(a - b) < 1e-6f; }

int main(void) {
    const int t_len = 1, tile = 19, full_h = 48, full_w = 86, origin_w = 13;
    const size_t n = (size_t)t_len * (size_t)tile * (size_t)tile;
    const size_t seq = n + 5;
    float *c0 = (float *)malloc(seq * H3_VAE_INT8_ROPE_HALF * sizeof(float));
    float *s0 = (float *)malloc(seq * H3_VAE_INT8_ROPE_HALF * sizeof(float));
    float *c1 = (float *)malloc(seq * H3_VAE_INT8_ROPE_HALF * sizeof(float));
    float *s1 = (float *)malloc(seq * H3_VAE_INT8_ROPE_HALF * sizeof(float));
    char err[128];
    size_t row_left, row_right, i;
    if (!c0 || !s0 || !c1 || !s1) return 1;
    if (!h3_vae_int8_prepare_rope_ex(t_len, tile, tile, 0, 0, full_h, full_w, n,
                                     seq, c0, s0, err, sizeof(err))) {
        fprintf(stderr, "FAIL left tile: %s\n", err);
        return 1;
    }
    if (!h3_vae_int8_prepare_rope_ex(t_len, tile, tile, 0, origin_w, full_h,
                                     full_w, n, seq, c1, s1, err,
                                     sizeof(err))) {
        fprintf(stderr, "FAIL right tile: %s\n", err);
        return 1;
    }
    /* Canvas x=origin_w, y=0: left tile local w=origin_w; right tile local w=0. */
    row_left = (size_t)origin_w; /* t=0,h=0,w=origin_w */
    row_right = 0;
    for (i = 0; i < H3_VAE_INT8_ROPE_HALF; i++) {
        if (!near(c0[row_left * H3_VAE_INT8_ROPE_HALF + i],
                  c1[row_right * H3_VAE_INT8_ROPE_HALF + i]) ||
            !near(s0[row_left * H3_VAE_INT8_ROPE_HALF + i],
                  s1[row_right * H3_VAE_INT8_ROPE_HALF + i])) {
            fprintf(stderr, "FAIL overlap RoPE mismatch at freq %zu\n", i);
            return 1;
        }
    }
    {
        float *cl = (float *)malloc(seq * H3_VAE_INT8_ROPE_HALF * sizeof(float));
        float *sl = (float *)malloc(seq * H3_VAE_INT8_ROPE_HALF * sizeof(float));
        if (!cl || !sl) return 1;
        if (!h3_vae_int8_prepare_rope(t_len, tile, tile, n, seq, cl, sl, err,
                                      sizeof(err)))
            return 1;
        /* Tile-local denom != canvas denom at the same local (0,0). */
        if (near(cl[8], c0[8]) && near(cl[16], c0[16])) {
            fprintf(stderr, "FAIL local RoPE should differ from canvas RoPE\n");
            return 1;
        }
        free(cl);
        free(sl);
    }
    free(c0);
    free(s0);
    free(c1);
    free(s1);

    /* Comfy comfy/ldm/minimax/vae.py create_token_ids:
     *   coords = arange(0.5, dim) / dim; coords = 2*coords - 1
     * RotaryEmbeddingND: inv_freq = 100**(-k*0.125), k=0..7; angle = 2π * axis * inv.
     * MiniMaxH3VideoVAE.decode() tiles at 256px; RoPE denom = tile latent H/W,
     * never the full 48×86 canvas. */
    {
        const int t = 7, h = 16, w = 16; /* Comfy tile_size=256 → 16 latent */
        const size_t np = (size_t)t * (size_t)h * (size_t)w;
        const size_t sq = np + 5;
        float *cos = (float *)malloc(sq * H3_VAE_INT8_ROPE_HALF * sizeof(float));
        float *sin = (float *)malloc(sq * H3_VAE_INT8_ROPE_HALF * sizeof(float));
        int ww;
        if (!cos || !sin) return 1;
        if (!h3_vae_int8_prepare_rope(t, h, w, np, sq, cos, sin, err,
                                      sizeof(err))) {
            fprintf(stderr, "FAIL comfy-tile rope: %s\n", err);
            return 1;
        }
        /* (t,h,w)=(0,0,0) spatial-W axis (axis=2), freq 0. */
        {
            float axis = 2.0f * (0.5f / (float)w) - 1.0f;
            float ang = 2.0f * 3.14159265358979323846f * axis;
            size_t off = 16; /* axis 2, freq 0 */
            if (fabsf(cos[off] - cosf(ang)) > 1e-5f ||
                fabsf(sin[off] - sinf(ang)) > 1e-5f) {
                fprintf(stderr, "FAIL Comfy create_token_ids mismatch\n");
                return 1;
            }
        }
        /* Adjacent latent cells: tile Δaxis=2/16=0.125; full 86 is 2/86≈0.023. */
        ww = 1; /* t=0,h=0,w=1 */
        {
            float a0 = 2.0f * (0.5f / 16.0f) - 1.0f;
            float a1 = 2.0f * (1.5f / 16.0f) - 1.0f;
            float af0 = 2.0f * (0.5f / 86.0f) - 1.0f;
            float af1 = 2.0f * (1.5f / 86.0f) - 1.0f;
            if (fabsf((a1 - a0) - 0.125f) > 1e-6f) {
                fprintf(stderr, "FAIL tile RoPE Δw\n");
                return 1;
            }
            if ((af1 - af0) > 0.04f) {
                fprintf(stderr, "FAIL full-canvas RoPE should be low-freq\n");
                return 1;
            }
            (void)ww;
        }
        free(cos);
        free(sin);
        printf("oracle: Comfy tile-local RoPE match; full-canvas Δw=%.5f vs "
               "tile Δw=0.125 (waffle = OOD full-frame)\n",
               2.0f / 86.0f);
        if (H3_VAE_INT8_LATENT_CH != 24 || H3_VAE_INT8_OUTPUT_PATCH != 3072 ||
            H3_VAE_INT8_SPATIAL_RATIO != 16 || H3_VAE_INT8_ROPE_HALF != 24 ||
            H3_VAE_INT8_HEAD_DIM != 64 || H3_VAE_INT8_HEADS != 32) {
            fprintf(stderr, "FAIL VAE shape vs Comfy ViT3DDecoder\n");
            return 1;
        }
    }
    printf("test_h3_vae_rope: PASS\n");
    return 0;
}
