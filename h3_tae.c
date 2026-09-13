#include "h3_tae.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int h3_tae_open(const char *path, void **ctx, char *err, size_t err_size) {
    if (ctx)
        *ctx = NULL;
    if (!path || !path[0]) {
        if (err && err_size)
            snprintf(err, err_size,
                     "TAE 无 --tae 权（仅 turbo；不是金线成片；不静默 official）");
        return 0;
    }
    if (err && err_size)
        snprintf(err, err_size,
                 "TAE decode deferred（未实现完整权；不是金线成片）: %s", path);
    return 0;
}

void h3_tae_close(void *ctx) { (void)ctx; }

int h3_tae_decode_f32(void *ctx, const float *latent, int t, int h, int w,
                      uint8_t *rgb_out, char *err, size_t err_size) {
    (void)ctx;
    (void)latent;
    (void)t;
    (void)h;
    (void)w;
    (void)rgb_out;
    if (err && err_size)
        snprintf(err, err_size, "TAE decode deferred（不是金线成片）");
    return 0;
}
