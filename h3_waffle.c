#include "h3_waffle.h"

#include <stdio.h>
#include <stdlib.h>

static int luma(const uint8_t *p) {
    return (int)p[0] * 77 + (int)p[1] * 150 + (int)p[2] * 29;
}

double h3_waffle_b16_rgb24(const uint8_t *rgb, int width, int height) {
    return h3_waffle_b16_gated_rgb24(rgb, width, height, NULL, NULL);
}

double h3_waffle_b16_gated_rgb24(const uint8_t *rgb, int width, int height,
                                 double *block_ratio, int *low_energy) {
    long long edge = 0, inner = 0;
    long n_edge = 0, n_inner = 0;
    int y, x;
    if (block_ratio) *block_ratio = 0.0;
    if (low_energy) *low_energy = 0;
    if (!rgb || width < 32 || height < 32) return 0.0;
    for (y = 1; y < height; y++) {
        const uint8_t *row = rgb + (size_t)y * (size_t)width * 3;
        const uint8_t *prev = rgb + (size_t)(y - 1) * (size_t)width * 3;
        int y_edge = ((y % 16) == 0);
        for (x = 1; x < width; x++) {
            int x_edge = ((x % 16) == 0);
            int dx = luma(row + x * 3) - luma(row + (x - 1) * 3);
            int dy = luma(row + x * 3) - luma(prev + x * 3);
            long e = (long)dx * (long)dx + (long)dy * (long)dy;
            if (x_edge || y_edge) {
                edge += e;
                n_edge++;
            } else {
                inner += e;
                n_inner++;
            }
        }
    }
    if (n_edge < 1 || n_inner < 1 || inner < 1) return 0.0;
    if (block_ratio || low_energy) {
        /* 16px between-block / within-block luma variance ratio over the
         * cropped complete-block region (same luma scale as the gradient
         * sums above; the scale cancels in the ratio). */
        int nbx = width / 16, nby = height / 16;
        double ratio = 0.0;
        if (nbx >= 1 && nby >= 1) {
            double gsum = 0.0, gsq = 0.0, wsum = 0.0;
            long nblocks = (long)nbx * (long)nby;
            int bx, by, px, py;
            for (by = 0; by < nby; by++) {
                for (bx = 0; bx < nbx; bx++) {
                    double bsum = 0.0, bsq = 0.0, bmean;
                    for (py = 0; py < 16; py++) {
                        const uint8_t *r =
                            rgb + (size_t)(by * 16 + py) * (size_t)width * 3;
                        for (px = 0; px < 16; px++) {
                            double l = (double)luma(r + (bx * 16 + px) * 3);
                            bsum += l;
                            bsq += l * l;
                        }
                    }
                    bmean = bsum / 256.0;
                    gsum += bmean;
                    gsq += bmean * bmean;
                    wsum += bsq / 256.0 - bmean * bmean;
                }
            }
            {
                double gmean = gsum / (double)nblocks;
                double between = gsq / (double)nblocks - gmean * gmean;
                double within = wsum / (double)nblocks;
                if (within > 0.0 && between > 0.0)
                    ratio = between / within;
            }
        }
        if (block_ratio) *block_ratio = ratio;
        if (low_energy && ratio < H3_WAFFLE_B16_MIN_BLOCK_RATIO)
            *low_energy = 1;
    }
    return ((double)edge / (double)n_edge) / ((double)inner / (double)n_inner);
}

void h3_waffle_log_rgb24(const uint8_t *rgb, int nframes, int width, int height) {
    static const int idx[] = {0, 24, 48, 90, 119};
    size_t fb;
    int i;
    if (!rgb || nframes < 1 || width < 1 || height < 1) return;
    fb = (size_t)width * (size_t)height * 3;
    for (i = 0; i < 5; i++) {
        int f = idx[i];
        double b16, ratio = 0.0;
        int low = 0;
        if (f >= nframes) f = nframes - 1;
        b16 = h3_waffle_b16_gated_rgb24(rgb + (size_t)f * fb, width, height,
                                        &ratio, &low);
        if (low)
            printf("waffle: frame=%d b16=%.3f INVALID_LOWENERGY "
                   "block_ratio=%.1f gate=%.1f\n",
                   f, b16, ratio, H3_WAFFLE_B16_MIN_BLOCK_RATIO);
        else
            printf("waffle: frame=%d b16=%.3f\n", f, b16);
    }
    fflush(stdout);
}
