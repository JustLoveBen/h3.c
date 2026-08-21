/* P4 M3 / E2: Euler sampling on W4A8 DiT with optional text context.
 * Build:
 *   cmake --build build-cuda -j --target h3_dit_w4a8_sample_probe
 *   ./build-cuda/h3_dit_w4a8_sample_probe --steps 2 --layers 4 FILE.safetensors
 *   ./build-cuda/h3_dit_w4a8_sample_probe --context 8 --steps 2 --layers 4 FILE
 */
#include "h3_dit_w4a8.h"
#include "h3_nvfp4.h"

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

static int build_linear_sigmas(int steps, float *sigmas) {
    int i;
    if (!sigmas || steps < 1) return 0;
    for (i = 0; i <= steps; i++)
        sigmas[i] = 1.0f - (float)i / (float)steps;
    sigmas[steps] = 0.0f;
    return 1;
}

static int run_sample(h3_dit_w4a8 *ctx, const h3_dit_w4a8_core *core,
                      unsigned layers, int steps, const float *sigmas,
                      float *latent, float *velocity, size_t batch,
                      size_t n_lat, const float *text_h, size_t text_seq,
                      const int *pt, const int *ph, const int *pw,
                      double *total_ms, char *error, size_t error_size) {
    int step;
    *total_ms = 0.0;
    for (step = 0; step < steps; step++) {
        h3_dit_w4a8_forward_stats fst;
        memset(&fst, 0, sizeof(fst));
        if (!h3_dit_w4a8_forward_video_f32(
                ctx, core, 0, layers, sigmas[step], latent, batch, text_h,
                text_seq, pt, ph, pw, 1.0f, velocity, 0, &fst, error,
                error_size))
            return 0;
        if (!all_finite(velocity, n_lat)) {
            snprintf(error, error_size, "step %d velocity non-finite", step);
            return 0;
        }
        if (!h3_dit_w4a8_euler_step_f32(latent, velocity, n_lat, sigmas[step],
                                        sigmas[step + 1], error, error_size))
            return 0;
        if (!all_finite(latent, n_lat)) {
            snprintf(error, error_size, "step %d latent non-finite", step);
            return 0;
        }
        *total_ms += fst.total_ms;
        printf("  step %d: sigma=%.4f→%.4f t_row=%zu vel_absmax=%.6g "
               "lat_absmax=%.6g forward_ms=%.1f\n",
               step, sigmas[step], sigmas[step + 1], fst.t_row,
               absmax(velocity, n_lat), absmax(latent, n_lat), fst.total_ms);
    }
    return 1;
}

int main(int argc, char **argv) {
    unsigned layers = 4;
    int steps = 2;
    size_t batch = 4;
    size_t context = 0;
    uint64_t seed = 0x53414D504C45ULL;
    uint64_t ctx_seed = 0x435458540000ULL;
    int device = 0;
    int compare_uncond = 0;
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
    float *latent = NULL, *velocity = NULL, *sigmas = NULL;
    float *text_emb = NULL, *text_h = NULL;
    int *pt = NULL, *ph = NULL, *pw = NULL;
    size_t free0 = 0, total0 = 0;
    size_t i, n_lat;
    double total_ms = 0.0;
    int rc = 1;
    int iarg;
    uint64_t csum0 = 0, csum1 = 0, csum_uncond = 0;

    memset(&core, 0, sizeof(core));

    for (iarg = 1; iarg < argc; iarg++) {
        if (!strcmp(argv[iarg], "--layers") && iarg + 1 < argc) {
            layers = (unsigned)strtoul(argv[++iarg], NULL, 10);
        } else if (!strcmp(argv[iarg], "--steps") && iarg + 1 < argc) {
            steps = (int)strtol(argv[++iarg], NULL, 10);
        } else if (!strcmp(argv[iarg], "--batch") && iarg + 1 < argc) {
            batch = (size_t)strtoul(argv[++iarg], NULL, 10);
        } else if (!strcmp(argv[iarg], "--context") && iarg + 1 < argc) {
            context = (size_t)strtoul(argv[++iarg], NULL, 10);
        } else if (!strcmp(argv[iarg], "--context-seed") && iarg + 1 < argc) {
            ctx_seed = strtoull(argv[++iarg], NULL, 0);
        } else if (!strcmp(argv[iarg], "--compare-uncond")) {
            compare_uncond = 1;
        } else if (!strcmp(argv[iarg], "--seed") && iarg + 1 < argc) {
            seed = strtoull(argv[++iarg], NULL, 0);
        } else if (!strcmp(argv[iarg], "--device") && iarg + 1 < argc) {
            device = (int)strtol(argv[++iarg], NULL, 10);
        } else if (!strcmp(argv[iarg], "-h") || !strcmp(argv[iarg], "--help")) {
            fprintf(stderr,
                    "usage: %s [--steps K] [--layers N] [--batch M] "
                    "[--context T] [--compare-uncond] [--seed S] "
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
    if (!path || layers == 0 || batch == 0 || steps < 1) {
        fprintf(stderr, "usage: %s FILE.safetensors\n", argv[0]);
        return 2;
    }
    if (context && compare_uncond == 0) compare_uncond = 1;

    store = h3_w4a8_store_open(path, error, sizeof(error));
    if (!store) {
        fprintf(stderr, "open failed: %s\n", error);
        return 1;
    }
    if (!h3_dit_w4a8_load_core(store, &core, error, sizeof(error))) {
        fprintf(stderr, "load_core failed: %s\n", error);
        goto cleanup;
    }
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
    sigmas = (float *)malloc((size_t)(steps + 1) * sizeof(float));
    pt = (int *)malloc(batch * sizeof(int));
    ph = (int *)malloc(batch * sizeof(int));
    pw = (int *)malloc(batch * sizeof(int));
    if (!latent || !velocity || !sigmas || !pt || !ph || !pw) {
        fprintf(stderr, "OOM\n");
        goto cleanup;
    }
    for (i = 0; i < n_lat; i++) latent[i] = hash_to_unit(seed, i);
    for (i = 0; i < batch; i++) {
        pt[i] = (int)i;
        ph[i] = (int)(i % 2);
        pw[i] = (int)(i % 4);
    }
    if (!build_linear_sigmas(steps, sigmas)) {
        fprintf(stderr, "cannot build sigma schedule\n");
        goto cleanup;
    }
    csum0 = h3_dit_w4a8_checksum_f32(latent, n_lat);

    if (context) {
        if (!core.condition_w) {
            fprintf(stderr, "condition_proj missing in checkpoint\n");
            goto cleanup;
        }
        text_emb =
            h3_nvfp4_synth_condition_f32(context, ctx_seed, error, sizeof(error));
        text_h = (float *)malloc(context * H3_DIT_W4A8_HIDDEN * sizeof(float));
        if (!text_emb || !text_h) {
            fprintf(stderr, "OOM text condition\n");
            goto cleanup;
        }
        if (!h3_dit_w4a8_project_text_f32(&core, text_emb, context, text_h,
                                          error, sizeof(error))) {
            fprintf(stderr, "project_text failed: %s\n", error);
            goto cleanup;
        }
        printf("context: tokens=%zu seed=0x%llx projected_absmax=%.6g\n",
               context, (unsigned long long)ctx_seed,
               absmax(text_h, context * H3_DIT_W4A8_HIDDEN));
    }

    printf("path: %s\n", path);
    printf("cuda_device: [%d] %s\n", info.device_index, info.name);
    printf("vram_before: free=%s  total=%s\n",
           format_bytes(free0, b1, sizeof(b1)),
           format_bytes(total0, b2, sizeof(b2)));
    printf("sample: steps=%d layers=0..%u video_seq=%zu text_seq=%zu "
           "seed=0x%llx\n",
           steps, layers - 1, batch, context, (unsigned long long)seed);
    printf("sigmas:");
    {
        int step;
        for (step = 0; step <= steps; step++) printf(" %.4f", sigmas[step]);
    }
    printf("\n");
    printf("latent0: absmax=%.6g checksum=0x%016llx\n", absmax(latent, n_lat),
           (unsigned long long)csum0);

    {
        double pms = 0.0;
        if (!h3_dit_w4a8_prefetch(ctx, 0, layers, &pms, error, sizeof(error))) {
            fprintf(stderr, "prefetch failed: %s\n", error);
            goto cleanup;
        }
        printf("prefetch_ms=%.1f\n", pms);
    }

    if (!run_sample(ctx, &core, layers, steps, sigmas, latent, velocity, batch,
                    n_lat, text_h, context, pt, ph, pw, &total_ms, error,
                    sizeof(error))) {
        fprintf(stderr, "sample failed: %s\n", error);
        goto cleanup;
    }

    csum1 = h3_dit_w4a8_checksum_f32(latent, n_lat);
    h3_dit_w4a8_get_cache_stats(ctx, &cst, error, sizeof(error));
    printf("latent_final: absmax=%.6g checksum=0x%016llx  total_forward_ms=%.1f\n",
           absmax(latent, n_lat), (unsigned long long)csum1, total_ms);
    printf("resident=%s  vram_free=%s\n",
           format_bytes(cst.bytes_resident, b1, sizeof(b1)),
           format_bytes(cst.vram_free_bytes, b2, sizeof(b2)));

    if (context && compare_uncond) {
        float *lat_u = (float *)malloc(n_lat * sizeof(float));
        double ms_u = 0.0;
        if (!lat_u) {
            fprintf(stderr, "OOM uncond\n");
            goto cleanup;
        }
        for (i = 0; i < n_lat; i++) lat_u[i] = hash_to_unit(seed, i);
        printf("uncond_compare:\n");
        if (!run_sample(ctx, &core, layers, steps, sigmas, lat_u, velocity,
                        batch, n_lat, NULL, 0, pt, ph, pw, &ms_u, error,
                        sizeof(error))) {
            fprintf(stderr, "uncond sample failed: %s\n", error);
            free(lat_u);
            goto cleanup;
        }
        csum_uncond = h3_dit_w4a8_checksum_f32(lat_u, n_lat);
        free(lat_u);
        printf("context_vs_uncond: csum_ctx=0x%016llx csum_uncond=0x%016llx "
               "%s\n",
               (unsigned long long)csum1, (unsigned long long)csum_uncond,
               csum1 != csum_uncond ? "DISTINCT (condition effective)"
                                    : "SAME (FAIL — condition ignored?)");
        if (csum1 == csum_uncond) {
            rc = 1;
            goto cleanup;
        }
    }

    /* Reproducibility */
    {
        float *latent2 = (float *)malloc(n_lat * sizeof(float));
        uint64_t csum2;
        double ms2 = 0.0;
        if (!latent2) {
            fprintf(stderr, "OOM (repro)\n");
            goto cleanup;
        }
        for (i = 0; i < n_lat; i++) latent2[i] = hash_to_unit(seed, i);
        if (!run_sample(ctx, &core, layers, steps, sigmas, latent2, velocity,
                        batch, n_lat, text_h, context, pt, ph, pw, &ms2, error,
                        sizeof(error))) {
            free(latent2);
            fprintf(stderr, "repro failed: %s\n", error);
            goto cleanup;
        }
        csum2 = h3_dit_w4a8_checksum_f32(latent2, n_lat);
        free(latent2);
        if (csum2 != csum1) {
            printf("repro: FAIL csum1=0x%016llx csum2=0x%016llx\n",
                   (unsigned long long)csum1, (unsigned long long)csum2);
            rc = 1;
        } else {
            printf("repro: checksum match PASS\n");
            printf("summary: steps=%d layers=%u text_seq=%zu → PASS\n", steps,
                   layers, context);
            rc = 0;
        }
    }

cleanup:
    free(latent);
    free(velocity);
    free(sigmas);
    free(pt);
    free(ph);
    free(pw);
    free(text_emb);
    free(text_h);
    h3_dit_w4a8_core_free(&core);
    h3_dit_w4a8_free(ctx);
    h3_cuda_free(cuda);
    h3_w4a8_store_free(store);
    return rc;
}
