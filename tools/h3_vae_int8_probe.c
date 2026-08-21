/* E3: Video VAE INT8 dequant + optional minimal decode→PPM.
 *   cmake --build build-cuda -j --target h3_vae_int8_probe
 *   ./build-cuda/h3_vae_int8_probe FILE.safetensors
 *   ./build-cuda/h3_vae_int8_probe --decode --blocks 1 --out frame.ppm FILE
 */
#include "h3_vae_int8.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int all_finite(const float *x, size_t n) {
    size_t i;
    for (i = 0; i < n; i++)
        if (!isfinite(x[i])) return 0;
    return 1;
}

static float absmax(const float *x, size_t n) {
    float m = 0.0f;
    size_t i;
    for (i = 0; i < n; i++) {
        float a = fabsf(x[i]);
        if (a > m) m = a;
    }
    return m;
}

int main(int argc, char **argv) {
    const char *path = NULL;
    const char *base = "decoder.transformer_blocks.0.attn.to_out";
    const char *out_ppm = NULL;
    size_t batch = 4;
    unsigned blocks = 1;
    int do_decode = 0;
    int lh = 2, lw = 2;
    const int lt = 1;
    char error[512];
    h3_vae_int8_store store;
    h3_vae_int8_linear lin;
    h3_vae_int8_frames frames;
    h3_vae_int8_decode_stats stats;
    float *w = NULL, *x = NULL, *y = NULL, *chw = NULL;
    size_t i, n_chw;
    int rc = 1;
    int iarg;

    memset(&store, 0, sizeof(store));
    memset(&frames, 0, sizeof(frames));
    for (iarg = 1; iarg < argc; iarg++) {
        if (!strcmp(argv[iarg], "--base") && iarg + 1 < argc) {
            base = argv[++iarg];
        } else if (!strcmp(argv[iarg], "--batch") && iarg + 1 < argc) {
            batch = (size_t)strtoul(argv[++iarg], NULL, 10);
        } else if (!strcmp(argv[iarg], "--decode")) {
            do_decode = 1;
        } else if (!strcmp(argv[iarg], "--blocks") && iarg + 1 < argc) {
            blocks = (unsigned)strtoul(argv[++iarg], NULL, 10);
        } else if (!strcmp(argv[iarg], "--out") && iarg + 1 < argc) {
            out_ppm = argv[++iarg];
        } else if (!strcmp(argv[iarg], "--latent-h") && iarg + 1 < argc) {
            lh = (int)strtol(argv[++iarg], NULL, 10);
        } else if (!strcmp(argv[iarg], "--latent-w") && iarg + 1 < argc) {
            lw = (int)strtol(argv[++iarg], NULL, 10);
        } else if (argv[iarg][0] == '-') {
            fprintf(stderr, "unknown option: %s\n", argv[iarg]);
            return 2;
        } else {
            path = argv[iarg];
        }
    }
    if (!path) {
        fprintf(stderr, "usage: %s [--decode] [--blocks N] [--out ppm] "
                        "FILE.safetensors\n",
                argv[0]);
        return 2;
    }

    if (!h3_vae_int8_store_open(path, &store, error, sizeof(error))) {
        fprintf(stderr, "open failed: %s\n", error);
        return 1;
    }
    printf("path: %s\n", path);
    printf("tensors: %zu  format=%s\n", store.header.tensor_count,
           store.quant.format[0] ? store.quant.format : "(none)");
    printf("note: ConvRot act NOT applied; decode is minimal host path\n");

    if (!h3_vae_int8_find_linear(&store, base, &lin, error, sizeof(error))) {
        fprintf(stderr, "find_linear failed: %s\n", error);
        goto cleanup;
    }
    printf("linear: %s  I8[%zu,%zu]\n", lin.base_name, lin.n, lin.k);
    w = h3_vae_int8_dequant_f32(&store, &lin, error, sizeof(error));
    if (!w) {
        fprintf(stderr, "dequant failed: %s\n", error);
        goto cleanup;
    }
    x = (float *)malloc(batch * lin.k * sizeof(float));
    y = (float *)malloc(batch * lin.n * sizeof(float));
    if (!x || !y) {
        fprintf(stderr, "OOM\n");
        goto cleanup;
    }
    for (i = 0; i < batch * lin.k; i++)
        x[i] = ((float)((i * 17) % 100) - 50.0f) / 50.0f;
    if (!h3_vae_int8_gemm_f32(x, w, batch, lin.n, lin.k, y, error,
                              sizeof(error))) {
        fprintf(stderr, "gemm failed: %s\n", error);
        goto cleanup;
    }
    printf("gemm: y_absmax=%.6g finite=%d\n", absmax(y, batch * lin.n),
           all_finite(y, batch * lin.n));

    if (do_decode) {
        n_chw = (size_t)H3_VAE_INT8_LATENT_CH * (size_t)lt * (size_t)lh *
                (size_t)lw;
        chw = (float *)malloc(n_chw * sizeof(float));
        if (!chw) {
            fprintf(stderr, "OOM chw\n");
            goto cleanup;
        }
        for (i = 0; i < n_chw; i++)
            chw[i] = ((float)((i * 13) % 100) - 50.0f) / 200.0f;
        memset(&stats, 0, sizeof(stats));
        if (!h3_vae_int8_decode_f32(&store, chw, lt, lh, lw, blocks, &frames,
                                    &stats, error, sizeof(error))) {
            fprintf(stderr, "decode failed: %s\n", error);
            goto cleanup;
        }
        printf("decode: blocks=%u frames=%d %dx%d finite=%d total_ms=%.1f\n",
               stats.n_blocks, frames.frames, frames.width, frames.height,
               stats.all_finite, stats.total_ms);
        if (out_ppm &&
            !h3_vae_int8_write_ppm(&frames, 0, out_ppm, error, sizeof(error))) {
            fprintf(stderr, "ppm failed: %s\n", error);
            goto cleanup;
        }
        if (out_ppm) printf("wrote: %s\n", out_ppm);
        if (!stats.all_finite) goto cleanup;
    }

    if (all_finite(w, lin.n * lin.k) && all_finite(y, batch * lin.n)) {
        printf("summary: vae_int8 %s → PASS\n",
               do_decode ? "dequant+decode" : "dequant+gemm");
        rc = 0;
    } else {
        printf("summary: FAIL\n");
    }

cleanup:
    free(w);
    free(x);
    free(y);
    free(chw);
    h3_vae_int8_frames_free(&frames);
    h3_vae_int8_store_free(&store);
    return rc;
}
