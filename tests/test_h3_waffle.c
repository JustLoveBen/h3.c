#include "h3_waffle.h"

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
    const int w = 64, h = 64;
    uint8_t *smooth = (uint8_t *)malloc((size_t)w * h * 3);
    uint8_t *grid = (uint8_t *)malloc((size_t)w * h * 3);
    uint8_t *flat = (uint8_t *)calloc((size_t)w * h * 3, 1);
    double b_s, b_g, b_f, r_f = -1.0;
    int low_f = 0;
    int y, x;
    if (!smooth || !grid || !flat) return 1;
    for (y = 0; y < h; y++) {
        for (x = 0; x < w; x++) {
            size_t i = ((size_t)y * w + x) * 3;
            uint8_t v = (uint8_t)(x * 2 + y);
            smooth[i] = smooth[i + 1] = smooth[i + 2] = v;
            v = ((x % 16) == 0 || (y % 16) == 0) ? 255 : 16;
            grid[i] = grid[i + 1] = grid[i + 2] = v;
        }
    }
    b_s = h3_waffle_b16_rgb24(smooth, w, h);
    b_g = h3_waffle_b16_rgb24(grid, w, h);
    b_f = h3_waffle_b16_gated_rgb24(flat, w, h, &r_f, &low_f);
    expect(b_s > 0.0 && b_s < 1.5, "smooth b16 low");
    expect(b_g > 2.0, "grid b16 high");
    expect(b_f == 0.0 && low_f == 0, "flat frame: zero inner, gate untouched");
    /* gate semantics: grid frame (pure 16px block-mean jumps, flat
     * interiors — the DMAD-noise fingerprint) must be flagged; a smooth
     * gradient ramp has within-block variance and must not. */
    {
        double r_s = 0.0, r_g = 0.0;
        int low_s = -1, low_g = -1;
        h3_waffle_b16_gated_rgb24(smooth, w, h, &r_s, &low_s);
        h3_waffle_b16_gated_rgb24(grid, w, h, &r_g, &low_g);
        expect(low_g == 1 && r_g < H3_WAFFLE_B16_MIN_BLOCK_RATIO,
               "grid (patch-noise) flagged INVALID");
        expect(r_s > r_g, "smooth ratio above grid ratio");
        (void)r_f;
    }
    free(smooth);
    free(grid);
    free(flat);
    return fails ? 1 : (printf("test_h3_waffle: PASS\n"), 0);
}
