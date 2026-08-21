/* P4 M2: patch_embed → N layers → final_layer (video-only W4A8 DiT).
 * Build:
 *   cmake --build build-cuda -j --target h3_dit_w4a8_forward_probe
 *   ./build-cuda/h3_dit_w4a8_forward_probe --layers 4 --batch 4 FILE.safetensors
 */
#include "h3_dit_w4a8.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

static float hash_to_unit(uint64_t seed, size_t index) {
    uint64_t h = mix64(seed ^ (uint64_t)(index + 1) * 0x9e3779b97f4a7c15ULL);
    return (float)(int32_t)(h >> 32) * (1.0f / 2147483648.0f);
}

static const char *format_bytes(size_t bytes, char *buf, size_t buf_size) {
    const double gb = (double)bytes / (1024.0 * 1024.0 * 1024.0);
    const double mb = (double)bytes / (1024.0 * 1024.0);
    if (gb >= 1.0)
        snprintf(buf, buf_size, "%.3f GiB (%zu B)", gb, bytes);
    else
        snprintf(buf, buf_size, "%.1f MiB (%zu B)", mb, bytes);
    return buf;
}

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
    unsigned layers = 4;
    size_t batch = 4;
    float sigma = 0.5f;
    int device = 0;
    const char *path = NULL;
    char error[512];
    char b1[64], b2[64];
    h3_w4a8_store *store = NULL;
    h3_cuda *cuda = NULL;
    h3_dit_w4a8 *ctx = NULL;
    h3_dit_w4a8_core core;
    h3_w4a8_cache_config cfg;
    h3_w4a8_cache_stats cst;
    h3_cuda_device_info info;
    h3_dit_w4a8_forward_stats fst;
    float *latent = NULL, *velocity = NULL;
    int *pt = NULL, *ph = NULL, *pw = NULL;
    size_t free0 = 0, total0 = 0;
    size_t i, n_lat;
    int rc = 1;
    int iarg;

    memset(&core, 0, sizeof(core));

    for (iarg = 1; iarg < argc; iarg++) {
        if (!strcmp(argv[iarg], "--layers") && iarg + 1 < argc) {
            layers = (unsigned)strtoul(argv[++iarg], NULL, 10);
        } else if (!strcmp(argv[iarg], "--batch") && iarg + 1 < argc) {
            batch = (size_t)strtoul(argv[++iarg], NULL, 10);
        } else if (!strcmp(argv[iarg], "--sigma") && iarg + 1 < argc) {
            sigma = strtof(argv[++iarg], NULL);
        } else if (!strcmp(argv[iarg], "--device") && iarg + 1 < argc) {
            device = (int)strtol(argv[++iarg], NULL, 10);
        } else if (!strcmp(argv[iarg], "-h") || !strcmp(argv[iarg], "--help")) {
            fprintf(stderr,
                    "usage: %s [--layers N] [--batch M] [--sigma S] "
                    "FILE.safetensors\n",
                    argv[0]);
            return 0;
        } else if (argv[iarg][0] == '-') {
            fprintf(stderr, "unknown option: %s\n", argv[iarg]);
            return 2;
        } else {
            path = argv[iarg];
        }
    }
    if (!path || layers == 0 || batch == 0) {
        fprintf(stderr, "usage: %s FILE.safetensors\n", argv[0]);
        return 2;
    }

    store = h3_w4a8_store_open(path, error, sizeof(error));
    if (!store) {
        fprintf(stderr, "open failed: %s\n", error);
        return 1;
    }
    if (!h3_dit_w4a8_load_core(store, &core, error, sizeof(error))) {
        fprintf(stderr, "load_core failed: %s\n", error);
        goto cleanup;
    }
    printf("core: video_patch [%d,%d]  final_video [%d,%d]  "
           "final_adaln [%d,%d]  t_table [%zu,%d]\n",
           H3_DIT_W4A8_HIDDEN, H3_DIT_W4A8_VIDEO_PATCH, H3_DIT_W4A8_VIDEO_PATCH,
           H3_DIT_W4A8_HIDDEN, H3_DIT_W4A8_FINAL_SLOTS * H3_DIT_W4A8_HIDDEN,
           H3_DIT_W4A8_TIME_FEAT, core.t_table_rows, H3_DIT_W4A8_TIME_FEAT);

    cuda = h3_cuda_create(device, error, sizeof(error));
    if (!cuda) {
        fprintf(stderr, "cuda create failed: %s\n", error);
        goto cleanup;
    }
    h3_cuda_get_info(cuda, &info);
    h3_cuda_mem_info(cuda, &free0, &total0, NULL, 0);
    h3_dit_w4a8_config_init(&cfg);
    cfg.policy = H3_W4A8_CACHE_RESIDENT_ALL;
    ctx = h3_dit_w4a8_open(cuda, store, &cfg, error, sizeof(error));
    if (!ctx) {
        fprintf(stderr, "dit open failed: %s\n", error);
        goto cleanup;
    }

    n_lat = batch * H3_DIT_W4A8_VIDEO_PATCH;
    latent = (float *)malloc(n_lat * sizeof(float));
    velocity = (float *)malloc(n_lat * sizeof(float));
    pt = (int *)malloc(batch * sizeof(int));
    ph = (int *)malloc(batch * sizeof(int));
    pw = (int *)malloc(batch * sizeof(int));
    if (!latent || !velocity || !pt || !ph || !pw) {
        fprintf(stderr, "OOM\n");
        goto cleanup;
    }
    for (i = 0; i < n_lat; i++) latent[i] = hash_to_unit(0x5041544348ULL, i);
    for (i = 0; i < batch; i++) {
        pt[i] = (int)i;
        ph[i] = (int)(i % 2);
        pw[i] = (int)(i % 4);
    }

    printf("path: %s\n", path);
    printf("cuda_device: [%d] %s\n", info.device_index, info.name);
    printf("vram_before: free=%s  total=%s\n",
           format_bytes(free0, b1, sizeof(b1)),
           format_bytes(total0, b2, sizeof(b2)));
    printf("forward: noise latent → patch → layers 0..%u → final  "
           "sigma=%.4f seq=%zu\n",
           layers - 1, sigma, batch);

    memset(&fst, 0, sizeof(fst));
    if (!h3_dit_w4a8_forward_video_f32(ctx, &core, 0, layers, sigma, latent,
                                       batch, NULL, 0, pt, ph, pw, 1.0f,
                                       velocity, 1, &fst, error,
                                       sizeof(error))) {
        fprintf(stderr, "forward_video failed: %s\n", error);
        goto cleanup;
    }
    h3_dit_w4a8_get_cache_stats(ctx, &cst, error, sizeof(error));

    printf("t_row=%zu  patch_ms=%.2f  layers_ms=%.2f  final_ms=%.2f  "
           "total_ms=%.2f\n",
           fst.t_row, fst.patch_ms, fst.layers_ms, fst.final_ms, fst.total_ms);
    printf("velocity: shape=[%zu,%d] absmax=%.6g finite=%d checksum=0x%016llx\n",
           batch, H3_DIT_W4A8_VIDEO_PATCH, absmax(velocity, n_lat),
           all_finite(velocity, n_lat),
           (unsigned long long)h3_dit_w4a8_checksum_f32(velocity, n_lat));
    printf("resident=%s  vram_free=%s\n",
           format_bytes(cst.bytes_resident, b1, sizeof(b1)),
           format_bytes(cst.vram_free_bytes, b2, sizeof(b2)));

    if (fst.all_finite && all_finite(velocity, n_lat) &&
        velocity != NULL /* same shape as latent */) {
        printf("summary: patch+layers+%u+final → PASS\n", layers);
        rc = 0;
    } else {
        printf("summary: FAIL\n");
        rc = 1;
    }

cleanup:
    free(latent);
    free(velocity);
    free(pt);
    free(ph);
    free(pw);
    h3_dit_w4a8_core_free(&core);
    h3_dit_w4a8_free(ctx);
    h3_cuda_free(cuda);
    h3_w4a8_store_free(store);
    return rc;
}
