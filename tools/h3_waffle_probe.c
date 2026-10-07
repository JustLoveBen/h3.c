#include "h3_waffle.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_ppm(const char *path, uint8_t **rgb, int *w, int *h) {
    FILE *fp;
    char magic[8];
    int maxv = 0;
    size_t n;
    fp = fopen(path, "rb");
    if (!fp) return 0;
    if (fscanf(fp, "%7s %d %d %d", magic, w, h, &maxv) != 4 ||
        strcmp(magic, "P6") != 0 || *w < 1 || *h < 1 || maxv != 255) {
        fclose(fp);
        return 0;
    }
    fgetc(fp);
    n = (size_t)(*w) * (size_t)(*h) * 3;
    *rgb = (uint8_t *)malloc(n);
    if (!*rgb) {
        fclose(fp);
        return 0;
    }
    if (fread(*rgb, 1, n, fp) != n) {
        free(*rgb);
        *rgb = NULL;
        fclose(fp);
        return 0;
    }
    fclose(fp);
    return 1;
}

int main(int argc, char **argv) {
    uint8_t *rgb = NULL;
    int w = 0, h = 0, i;
    if (argc < 2) {
        fprintf(stderr, "usage: %s frame.ppm [frame.ppm ...]\n", argv[0]);
        return 2;
    }
    for (i = 1; i < argc; i++) {
        double b16, ratio = 0.0;
        int low = 0;
        if (!read_ppm(argv[i], &rgb, &w, &h)) {
            fprintf(stderr, "waffle_probe: cannot read %s\n", argv[i]);
            return 1;
        }
        b16 = h3_waffle_b16_gated_rgb24(rgb, w, h, &ratio, &low);
        if (low)
            printf("waffle: frame=%s b16=%.3f %dx%d INVALID_LOWENERGY "
                   "block_ratio=%.1f gate=%.1f\n",
                   argv[i], b16, w, h, ratio, H3_WAFFLE_B16_MIN_BLOCK_RATIO);
        else
            printf("waffle: frame=%s b16=%.3f %dx%d\n", argv[i], b16, w, h);
        free(rgb);
        rgb = NULL;
    }
    return 0;
}
