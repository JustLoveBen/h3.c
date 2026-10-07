/* tests/test_h3_selflift.c — geometry only. Not a SelfLift port. */
#include "h3_selflift.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int h, w;
    if (!h3_selflift_lowres_latent(48, 84, 0.5f, &h, &w) || h != 24 ||
        w != 42) {
        fprintf(stderr, "FAIL latent 84x48 → 42x24 got %dx%d\n", w, h);
        return 1;
    }
    if (!h3_selflift_lowres_pixels(768, 1344, 0.5f, &h, &w) || h != 384 ||
        w != 672) {
        fprintf(stderr, "FAIL pixels 1344x768 → 672x384 got %dx%d\n", w, h);
        return 1;
    }
    if (h3_selflift_taomate_low_nfe(3) != 2) {
        fprintf(stderr, "FAIL taomate low nfe\n");
        return 1;
    }
    if (setenv("H3_LIFT_Z0_ONLY", "1", 1) != 0 ||
        h3_selflift_taomate_low_nfe(3) != 3) {
        fprintf(stderr, "FAIL taomate Z0_ONLY low nfe\n");
        return 1;
    }
    unsetenv("H3_LIFT_Z0_ONLY");
    if (h3_selflift_taomate_low_nfe(3) != 2) {
        fprintf(stderr, "FAIL taomate low nfe after unset Z0\n");
        return 1;
    }
    if (h3_selflift_taomate_low_nfe(4) != 0) {
        fprintf(stderr, "FAIL 4-step must not use taomate lift split\n");
        return 1;
    }
    if (h3_selflift_taomate_low_nfe(5) != 3) {
        fprintf(stderr, "FAIL Comfy simple-5 SplitSigmas step=3 → 3 low\n");
        return 1;
    }
    printf("test_h3_selflift: PASS (geometry only; no LatentResizer3D)\n");
    return 0;
}
