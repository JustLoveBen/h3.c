#include "h3_selflift.h"

#include <math.h>
#include <stdlib.h>

int h3_selflift_lowres_latent(int latent_h, int latent_w, float scale,
                              int *out_h, int *out_w) {
    int h, w;
    if (!out_h || !out_w || latent_h < 2 || latent_w < 2)
        return 0;
    if (!(scale > 0.0f) || scale > 1.0f)
        scale = 0.5f;
    h = (int)lround((double)latent_h * (double)scale / 2.0) * 2;
    w = (int)lround((double)latent_w * (double)scale / 2.0) * 2;
    if (h < 2)
        h = 2;
    if (w < 2)
        w = 2;
    *out_h = h;
    *out_w = w;
    return 1;
}

int h3_selflift_lowres_pixels(int pix_h, int pix_w, float scale, int *out_h,
                              int *out_w) {
    int lh, lw, lo_h, lo_w;
    if (!out_h || !out_w || pix_h < 16 || pix_w < 16)
        return 0;
    /* Official VAE spatial 16×. */
    lh = pix_h / 16;
    lw = pix_w / 16;
    if (!h3_selflift_lowres_latent(lh, lw, scale, &lo_h, &lo_w))
        return 0;
    *out_h = lo_h * 16;
    *out_w = lo_w * 16;
    return 1;
}

int h3_selflift_taomate_low_nfe(int steps) {
    const char *z0;
    /* Comfy minimax-h3-r2v-taomate-2stage.json: BasicScheduler steps=5,
     * SplitSigmas step=3 → 3 NFE @ stage1 + 2 NFE @ hires. */
    if (steps == 5)
        return 3;
    /* Distilled 3-interval {0,16,33,49}: default 2 low + LBH + 1 high.
     * Z0_ONLY: all 3 Euler at low-res, LBH the clean x0, skip high DiT. */
    if (steps != 3)
        return 0;
    z0 = getenv("H3_LIFT_Z0_ONLY");
    if (z0 && z0[0] == '1')
        return 3;
    return 2;
}
