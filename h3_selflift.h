#ifndef H3_SELFLIFT_H
#define H3_SELFLIFT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Geometry helpers for SelfLift port. Gold never calls this.
 * Plugin formula on latent: h = max(2, round(H * scale / 2) * 2).
 * 1344×768 → latent 84×48 → scale 0.5 → 42×24 → pixels 672×384. */

int h3_selflift_lowres_latent(int latent_h, int latent_w, float scale,
                              int *out_h, int *out_w);
int h3_selflift_lowres_pixels(int pix_h, int pix_w, float scale, int *out_h,
                              int *out_w);

/* TaoMate distilled 3-step {0,16,33,49}: default 2 low + LBH + 1 high.
 * Comfy 2-stage JSON (BasicScheduler steps=5, SplitSigmas step=3): 3 low + 2 high.
 * H3_LIFT_Z0_ONLY=1 on distilled 3 → 3 low, 0 high. 0 if steps is neither. */
int h3_selflift_taomate_low_nfe(int steps);

#ifdef __cplusplus
}
#endif

#endif
