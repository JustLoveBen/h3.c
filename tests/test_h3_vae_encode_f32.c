/* tests/test_h3_vae_encode_f32.c
 * Stage C: shared visual-condition payload (encode / patchify / 0.999 augment).
 * Does not claim a first-frame product path. */
#include "h3_vae_encode_f32.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* h3_gpu_cuda.a pulls ConvRot helpers we do not exercise. */
float *h3_w4a8_hadamard_normalized(int size, char *error, size_t error_size) {
    (void)size;
    if (error && error_size) snprintf(error, error_size, "hadamard stub");
    return NULL;
}
float *h3_w4a8_hadamard_sylvester_normalized(int size, char *error,
                                             size_t error_size) {
    (void)size;
    if (error && error_size) snprintf(error, error_size, "hadamard stub");
    return NULL;
}

#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif

static double test_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

#define LATENT_CH 24
#define PATCH 96

static int g_fail;

static void fail_msg(const char *msg) {
    fprintf(stderr, "FAIL tests/test_h3_vae_encode_f32.c: %s\n", msg);
    g_fail = 1;
}

static void test_patchify_order(void) {
    /* 1×2×2 latent, C=24 unique values; rows[0] from c=0,t=0,h=0,w=0 */
    const int time = 1, height = 2, width = 2;
    const size_t latent_n = (size_t)LATENT_CH * time * height * width;
    const size_t row_n = latent_n; /* 24*1*2*2 == 96 */
    float latent[24 * 1 * 2 * 2];
    float rows[96];
    int c, h, w;
    size_t i;
    if (row_n != 96) {
        fail_msg("patchify fixture size");
        return;
    }
    for (c = 0; c < LATENT_CH; c++)
        for (h = 0; h < height; h++)
            for (w = 0; w < width; w++) {
                size_t idx = (((size_t)c * time + 0) * height + h) * width + w;
                latent[idx] = (float)(c * 1000 + h * 10 + w);
            }
    if (!h3_vae_patchify_video_96(latent, time, height, width, rows, row_n)) {
        fail_msg("patchify returned 0");
        return;
    }
    if (rows[0] != 0.0f) {
        char buf[128];
        snprintf(buf, sizeof(buf), "rows[0]=%g want 0 (c=0,t=0,h=0,w=0)",
                 (double)rows[0]);
        fail_msg(buf);
        return;
    }
    /* inner nest after t,h,w origin: c, dh, dw → next is dh=0,dw=1 → w=1 */
    if (rows[1] != 1.0f) {
        char buf[128];
        snprintf(buf, sizeof(buf), "rows[1]=%g want 1 (c=0,h=0,w=1)",
                 (double)rows[1]);
        fail_msg(buf);
        return;
    }
    /* after dw loop: dh=1,dw=0 → h=1,w=0 */
    if (rows[2] != 10.0f) {
        char buf[128];
        snprintf(buf, sizeof(buf), "rows[2]=%g want 10 (c=0,h=1,w=0)",
                 (double)rows[2]);
        fail_msg(buf);
        return;
    }
    for (i = 0; i < row_n; i++) {
        if (!isfinite(rows[i])) {
            fail_msg("patchify produced non-finite");
            return;
        }
    }
    puts("ok: patchify order t,h+=2,w+=2,c,dh,dw");
}

static void test_augment_formula(void) {
    float x = 1.0f;
    /* Hand-eval of h3_host PCG Box-Muller: seed=1 first N → y=0.999+0.001*N */
    const float want = 0.998016536f;
    h3_vae_augment_0p999(&x, 1, 1);
    if (!isfinite(x) || fabsf(x - want) > 1e-6f) {
        char buf[128];
        snprintf(buf, sizeof(buf), "augment(1, seed=1)=%.9g want %.9g",
                 (double)x, (double)want);
        fail_msg(buf);
        return;
    }
    puts("ok: augment 0.999 * x + 0.001 * N(seed)");
}

static void fill_tile_rgb(float *rgb, int frames, int height, int width) {
    int c, t, y, x;
    for (c = 0; c < 3; c++)
        for (t = 0; t < frames; t++)
            for (y = 0; y < height; y++)
                for (x = 0; x < width; x++) {
                    size_t idx = (((size_t)c * frames + t) * height + y) *
                                     width + x;
                    float u = (float)((x * 13 + y * 7 + t * 3 + c * 17) % 251) /
                              250.0f;
                    rgb[idx] = u;
                }
}

static int load_raw_f32(const char *path, float *dst, size_t n) {
    FILE *file = fopen(path, "rb");
    size_t got;
    if (!file) return 0;
    got = fread(dst, sizeof(float), n, file);
    fclose(file);
    return got == n;
}

static int save_raw_f32(const char *path, const float *src, size_t n) {
    FILE *file = fopen(path, "wb");
    size_t put;
    if (!file) return 0;
    put = fwrite(src, sizeof(float), n, file);
    fclose(file);
    return put == n;
}

static int load_jpeg_rgb01(const char *jpeg, int height, int width,
                           float *rgb, char *err, size_t es) {
    char cmd[1024];
    char tmp[128];
    FILE *file;
    unsigned char *raw;
    size_t n = (size_t)height * (size_t)width * 3u;
    size_t got;
    int y, x, c;
    snprintf(tmp, sizeof(tmp), "/tmp/h3_vae_enc_%d.rgb", (int)getpid());
    snprintf(cmd, sizeof(cmd),
             "ffmpeg -v error -y -i \"%s\" -vf scale=%d:%d:flags=bicubic "
             "-pix_fmt rgb24 \"%s\"",
             jpeg, width, height, tmp);
    if (system(cmd) != 0) {
        snprintf(err, es, "ffmpeg failed converting jpeg to %dx%d rgb", width,
                 height);
        unlink(tmp);
        return 0;
    }
    raw = malloc(n);
    file = fopen(tmp, "rb");
    got = file ? fread(raw, 1, n, file) : 0;
    if (file) fclose(file);
    unlink(tmp);
    if (!raw || got != n) {
        free(raw);
        snprintf(err, es, "cannot read converted jpeg rgb");
        return 0;
    }
    for (c = 0; c < 3; c++)
        for (y = 0; y < height; y++)
            for (x = 0; x < width; x++) {
                size_t src = ((size_t)y * width + x) * 3u + (size_t)c;
                size_t dst = (((size_t)c * height + y) * width) + x;
                rgb[dst] = (float)raw[src] / 255.0f;
            }
    free(raw);
    return 1;
}

static void test_encode_tile(const char *vae_path, const char *jpeg_path,
                             const char *ref_path, const char *dump_latent,
                             const char *dump_rows, int height, int width) {
    const int frames = 1;
    if (height < 32 || width < 32 || (height % 16) || (width % 16)) {
        fail_msg("encode height/width must be multiples of 16 and >=32");
        return;
    }
    const int want_t = (frames + 3) / 4;
    const int want_h = height / 16;
    const int want_w = width / 16;
    const size_t rgb_n = (size_t)3 * frames * height * width;
    const size_t lat_n = (size_t)LATENT_CH * want_t * want_h * want_w;
    const size_t row_n = lat_n;
    float *rgb = malloc(rgb_n * sizeof(*rgb));
    float *rows = malloc(row_n * sizeof(*rows));
    float *ref = NULL;
    h3_vae_latent_f32 lat = {0};
    char error[512];
    size_t i;
    double max_abs = 0.0, mean = 0.0, sq = 0.0, t0;
    if (!rgb || !rows) {
        fail_msg("OOM tile rgb/rows");
        free(rgb);
        free(rows);
        return;
    }
    error[0] = '\0';
    if (jpeg_path && *jpeg_path) {
        if (!load_jpeg_rgb01(jpeg_path, height, width, rgb, error,
                             sizeof(error))) {
            fail_msg(error);
            free(rgb);
            free(rows);
            return;
        }
        printf("encode input: jpeg resized %dx%d\n", width, height);
    } else {
        fill_tile_rgb(rgb, frames, height, width);
        printf("encode input: synthetic %dx%d\n", width, height);
    }
    t0 = test_now_ms();
    if (!h3_vae_encode_f32(vae_path, rgb, frames, height, width, &lat, error,
                           sizeof(error))) {
        char buf[640];
        snprintf(buf, sizeof(buf), "encode failed: %s", error);
        fail_msg(buf);
        free(rgb);
        free(rows);
        return;
    }
    if (lat.time != want_t || lat.height != want_h || lat.width != want_w ||
        !lat.values) {
        char buf[160];
        snprintf(buf, sizeof(buf),
                 "encode shape %d x %d x %d want %d x %d x %d",
                 lat.time, lat.height, lat.width, want_t, want_h, want_w);
        fail_msg(buf);
        h3_vae_latent_f32_free(&lat);
        free(rgb);
        free(rows);
        return;
    }
    for (i = 0; i < lat_n; i++) {
        if (!isfinite(lat.values[i])) {
            fail_msg("encode produced non-finite latent");
            h3_vae_latent_f32_free(&lat);
            free(rgb);
            free(rows);
            return;
        }
        mean += (double)lat.values[i];
        sq += (double)lat.values[i] * (double)lat.values[i];
    }
    mean /= (double)lat_n;
    sq = sqrt(sq / (double)lat_n - mean * mean);
    {
        uint64_t cs = 1469598103934665603ull;
        for (i = 0; i < lat_n; i++) {
            uint32_t bits;
            memcpy(&bits, &lat.values[i], sizeof(bits));
            cs ^= (uint64_t)bits;
            cs *= 1099511628211ull;
        }
        printf("encode tile: shape [24,%d,%d,%d] mean=%.5g std=%.5g "
               "wall_ms=%.1f host=%s checksum=0x%016llx\n",
               lat.time, lat.height, lat.width, mean, sq, test_now_ms() - t0,
               (getenv("H3_VAE_ENCODE_HOST") &&
                getenv("H3_VAE_ENCODE_HOST")[0] == '1')
                   ? "1"
                   : "0",
               (unsigned long long)cs);
    }
    if (!h3_vae_patchify_video_96(lat.values, lat.time, lat.height, lat.width,
                                  rows, row_n)) {
        fail_msg("patchify of encoded latent failed");
        h3_vae_latent_f32_free(&lat);
        free(rgb);
        free(rows);
        return;
    }
    printf("condition rows: %zu x %d (patchify 2x2x24)\n",
           row_n / PATCH, PATCH);
    if (ref_path && *ref_path) {
        ref = malloc(lat_n * sizeof(*ref));
        if (!ref || !load_raw_f32(ref_path, ref, lat_n)) {
            fail_msg("cannot read --ref-latent (channel-major F32 24x1x16x16)");
            free(ref);
            h3_vae_latent_f32_free(&lat);
            free(rgb);
            free(rows);
            return;
        }
        for (i = 0; i < lat_n; i++) {
            double d = fabs((double)lat.values[i] - (double)ref[i]);
            if (d > max_abs) max_abs = d;
        }
        /* Single 256 tile, 1 frame vs Darwin Metal h3_video_vae_encode
         * / encode_tile dump (not torch). Plan nails 5e-2.
         * Measured 2026-08-19 4090 fill_tile_rgb: max_abs=6.50e-6. Do not loosen. */
        printf("encode vs ref-latent: max_abs=%.7g (threshold 5e-2)\n", max_abs);
        if (max_abs >= 5e-2) {
            char buf[128];
            snprintf(buf, sizeof(buf), "max_abs %.7g >= 5e-2", max_abs);
            fail_msg(buf);
        }
        free(ref);
    }
    if (dump_latent && *dump_latent &&
        !save_raw_f32(dump_latent, lat.values, lat_n))
        fail_msg("cannot write --dump-latent");
    if (dump_rows && *dump_rows && !save_raw_f32(dump_rows, rows, row_n))
        fail_msg("cannot write --dump-rows");
    h3_vae_latent_f32_free(&lat);
    free(rgb);
    free(rows);
    if (!g_fail)
        printf("ok: encode %dx%d + patchify 96-d rows\n", width, height);
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s [--vae FILE] [--jpeg FILE] [--ref-latent FILE] "
            "[--dump-latent FILE] [--dump-rows FILE] [--width N] [--height N]\n",
            argv0);
}

int main(int argc, char **argv) {
    const char *vae = NULL, *jpeg = NULL, *ref = NULL, *dump_lat = NULL,
               *dump_rows = NULL;
    int i, height = 256, width = 256;
    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--vae") && i + 1 < argc)
            vae = argv[++i];
        else if (!strcmp(argv[i], "--jpeg") && i + 1 < argc)
            jpeg = argv[++i];
        else if (!strcmp(argv[i], "--ref-latent") && i + 1 < argc)
            ref = argv[++i];
        else if (!strcmp(argv[i], "--dump-latent") && i + 1 < argc)
            dump_lat = argv[++i];
        else if (!strcmp(argv[i], "--dump-rows") && i + 1 < argc)
            dump_rows = argv[++i];
        else if (!strcmp(argv[i], "--width") && i + 1 < argc)
            width = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--height") && i + 1 < argc)
            height = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--help")) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    test_patchify_order();
    test_augment_formula();
    if (vae)
        test_encode_tile(vae, jpeg, ref, dump_lat, dump_rows, height, width);
    else
        puts("skip: encode tile (pass --vae to run)");
    return g_fail ? 1 : 0;
}
