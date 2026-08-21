/* E4: end-to-end W4A8 generate smoke (pure C/CUDA).
 *   condition → DiT Euler few-step → latent (+ optional VAE PPM)
 *
 *   cmake --build build-cuda -j --target h3_w4a8_generate_probe
 *   ./build-cuda/h3_w4a8_generate_probe \
 *     --dit .../minimax_h3_fl2va_pruned_w4a8_mixed.safetensors \
 *     --vae .../minimax_h3_video_vae_int8_convrot.safetensors \
 *     --te  .../qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
 *     --steps 2 --layers 4 --context 8 --latent-h 2 --latent-w 2 \
 *     --out /home/haiyang/h3_c/out
 */
#include "h3_dit_w4a8.h"
#include "h3_nvfp4.h"
#include "h3_vae_int8.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

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

static int ensure_dir(const char *path) {
    struct stat st;
    if (!path || !path[0]) return 0;
    if (stat(path, &st) == 0) return S_ISDIR(st.st_mode);
    return mkdir(path, 0755) == 0;
}

static int write_latent_raw(const char *path, const float *x, size_t n,
                            char *error, size_t error_size) {
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        snprintf(error, error_size, "cannot write %s", path);
        return 0;
    }
    if (fwrite(x, sizeof(float), n, fp) != n) {
        fclose(fp);
        snprintf(error, error_size, "short write %s", path);
        return 0;
    }
    fclose(fp);
    return 1;
}

int main(int argc, char **argv) {
    const char *dit_path = NULL;
    const char *vae_path = NULL;
    const char *te_path = NULL;
    const char *out_dir = NULL;
    unsigned layers = 4;
    unsigned vae_blocks = 1;
    int steps = 2;
    int latent_t = 1, latent_h = 2, latent_w = 2;
    size_t context = 8;
    uint64_t seed = 0x47454E45524154ULL;
    uint64_t ctx_seed = 0x43545854ULL;
    int device = 0;
    int use_te_embed = 0;
    char error[512];
    char b1[64], b2[64];
    char pathbuf[768];

    h3_w4a8_store *dit_store = NULL;
    h3_cuda *cuda = NULL;
    h3_dit_w4a8 *dit = NULL;
    h3_dit_w4a8_core core;
    h3_w4a8_cache_config cfg;
    h3_w4a8_cache_stats cst;
    h3_cuda_device_info info;
    h3_nvfp4_store te_store;
    h3_vae_int8_store vae_store;
    h3_vae_int8_frames frames;
    h3_vae_int8_decode_stats vst;

    float *latent = NULL, *velocity = NULL, *sigmas = NULL;
    float *text_emb = NULL, *text_h = NULL, *latent_chw = NULL;
    int32_t *token_ids = NULL;
    int *pt = NULL, *ph = NULL, *pw = NULL;
    size_t video_seq, n_lat, i;
    size_t free0 = 0, total0 = 0, free1 = 0;
    double total_ms = 0.0;
    uint64_t csum_lat = 0, csum_uncond = 0;
    int step, iarg, rc = 1;

    memset(&core, 0, sizeof(core));
    memset(&te_store, 0, sizeof(te_store));
    memset(&vae_store, 0, sizeof(vae_store));
    memset(&frames, 0, sizeof(frames));
    memset(&vst, 0, sizeof(vst));

    for (iarg = 1; iarg < argc; iarg++) {
        if (!strcmp(argv[iarg], "--dit") && iarg + 1 < argc) {
            dit_path = argv[++iarg];
        } else if (!strcmp(argv[iarg], "--vae") && iarg + 1 < argc) {
            vae_path = argv[++iarg];
        } else if (!strcmp(argv[iarg], "--te") && iarg + 1 < argc) {
            te_path = argv[++iarg];
            use_te_embed = 1;
        } else if (!strcmp(argv[iarg], "--out") && iarg + 1 < argc) {
            out_dir = argv[++iarg];
        } else if (!strcmp(argv[iarg], "--layers") && iarg + 1 < argc) {
            layers = (unsigned)strtoul(argv[++iarg], NULL, 10);
        } else if (!strcmp(argv[iarg], "--vae-blocks") && iarg + 1 < argc) {
            vae_blocks = (unsigned)strtoul(argv[++iarg], NULL, 10);
        } else if (!strcmp(argv[iarg], "--steps") && iarg + 1 < argc) {
            steps = (int)strtol(argv[++iarg], NULL, 10);
        } else if (!strcmp(argv[iarg], "--context") && iarg + 1 < argc) {
            context = (size_t)strtoul(argv[++iarg], NULL, 10);
        } else if (!strcmp(argv[iarg], "--latent-t") && iarg + 1 < argc) {
            latent_t = (int)strtol(argv[++iarg], NULL, 10);
        } else if (!strcmp(argv[iarg], "--latent-h") && iarg + 1 < argc) {
            latent_h = (int)strtol(argv[++iarg], NULL, 10);
        } else if (!strcmp(argv[iarg], "--latent-w") && iarg + 1 < argc) {
            latent_w = (int)strtol(argv[++iarg], NULL, 10);
        } else if (!strcmp(argv[iarg], "--seed") && iarg + 1 < argc) {
            seed = strtoull(argv[++iarg], NULL, 0);
        } else if (!strcmp(argv[iarg], "--device") && iarg + 1 < argc) {
            device = (int)strtol(argv[++iarg], NULL, 10);
        } else if (!strcmp(argv[iarg], "-h") || !strcmp(argv[iarg], "--help")) {
            fprintf(stderr,
                    "usage: %s --dit DIT.safetensors [--vae VAE] [--te TE] "
                    "[--out DIR] [--steps K] [--layers N] [--context T] "
                    "[--latent-h H --latent-w W] [--vae-blocks B]\n",
                    argv[0]);
            return 0;
        } else if (argv[iarg][0] == '-') {
            fprintf(stderr, "unknown option: %s\n", argv[iarg]);
            return 2;
        } else if (!dit_path) {
            dit_path = argv[iarg];
        }
    }

    if (!dit_path || layers == 0 || steps < 1 || context == 0 || latent_t < 1 ||
        latent_h < 2 || latent_w < 2 || (latent_h % 2) || (latent_w % 2)) {
        fprintf(stderr,
                "need --dit, --context>0, even latent-h/w >=2, steps/layers\n");
        return 2;
    }

    video_seq = (size_t)latent_t * (size_t)(latent_h / 2) * (size_t)(latent_w / 2);
    n_lat = video_seq * H3_DIT_W4A8_VIDEO_PATCH;

    dit_store = h3_w4a8_store_open(dit_path, error, sizeof(error));
    if (!dit_store) {
        fprintf(stderr, "dit open failed: %s\n", error);
        return 1;
    }
    if (!h3_dit_w4a8_load_core(dit_store, &core, error, sizeof(error))) {
        fprintf(stderr, "load_core failed: %s\n", error);
        goto cleanup;
    }
    if (!core.condition_w) {
        fprintf(stderr, "condition_proj missing — cannot inject text\n");
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
    dit = h3_dit_w4a8_open(cuda, dit_store, &cfg, error, sizeof(error));
    if (!dit) {
        fprintf(stderr, "dit_w4a8 open failed: %s\n", error);
        goto cleanup;
    }

    /* E1: TE embed lookup or synth stub → DiT condition_proj. */
    if (use_te_embed && te_path) {
        if (!h3_nvfp4_store_open(te_path, &te_store, error, sizeof(error))) {
            fprintf(stderr, "te open failed: %s\n", error);
            goto cleanup;
        }
        token_ids = (int32_t *)malloc(context * sizeof(int32_t));
        if (!token_ids) {
            fprintf(stderr, "OOM tokens\n");
            goto cleanup;
        }
        for (i = 0; i < context; i++)
            token_ids[i] = (int32_t)(1 + (mix64(ctx_seed + i) % 1000));
        text_emb = h3_nvfp4_make_condition_f32(&te_store, token_ids, context,
                                               ctx_seed, NULL, NULL, error,
                                               sizeof(error));
        printf("te: embed_lookup tokens=%zu path=%s\n", context, te_path);
    } else {
        text_emb =
            h3_nvfp4_synth_condition_f32(context, ctx_seed, error, sizeof(error));
        printf("te: synth_stub tokens=%zu (full TE forward TODO)\n", context);
    }
    if (!text_emb) {
        fprintf(stderr, "make condition failed: %s\n", error);
        goto cleanup;
    }
    text_h = (float *)malloc(context * H3_DIT_W4A8_HIDDEN * sizeof(float));
    if (!text_h) {
        fprintf(stderr, "OOM text_h\n");
        goto cleanup;
    }
    if (!h3_dit_w4a8_project_text_f32(&core, text_emb, context, text_h, error,
                                      sizeof(error))) {
        fprintf(stderr, "project_text failed: %s\n", error);
        goto cleanup;
    }

    latent = (float *)malloc(n_lat * sizeof(float));
    velocity = (float *)malloc(n_lat * sizeof(float));
    sigmas = (float *)malloc((size_t)(steps + 1) * sizeof(float));
    pt = (int *)malloc(video_seq * sizeof(int));
    ph = (int *)malloc(video_seq * sizeof(int));
    pw = (int *)malloc(video_seq * sizeof(int));
    if (!latent || !velocity || !sigmas || !pt || !ph || !pw) {
        fprintf(stderr, "OOM buffers\n");
        goto cleanup;
    }
    for (i = 0; i < n_lat; i++) latent[i] = hash_to_unit(seed, i);
    {
        size_t idx = 0;
        int t, yh, xw;
        for (t = 0; t < latent_t; t++)
            for (yh = 0; yh < latent_h / 2; yh++)
                for (xw = 0; xw < latent_w / 2; xw++, idx++) {
                    pt[idx] = t;
                    ph[idx] = yh;
                    pw[idx] = xw;
                }
    }
    if (!build_linear_sigmas(steps, sigmas)) {
        fprintf(stderr, "sigma schedule failed\n");
        goto cleanup;
    }

    printf("=== h3_w4a8_generate_probe ===\n");
    printf("cuda_device: [%d] %s\n", info.device_index, info.name);
    printf("vram_before: free=%s total=%s\n",
           format_bytes(free0, b1, sizeof(b1)),
           format_bytes(total0, b2, sizeof(b2)));
    printf("dit: %s\n", dit_path);
    printf("shape: latent_t=%d h=%d w=%d → video_seq=%zu text_seq=%zu "
           "steps=%d layers=%u\n",
           latent_t, latent_h, latent_w, video_seq, context, steps, layers);
    printf("text_hidden absmax=%.6g checksum=0x%016llx\n",
           absmax(text_h, context * H3_DIT_W4A8_HIDDEN),
           (unsigned long long)h3_dit_w4a8_checksum_f32(
               text_h, context * H3_DIT_W4A8_HIDDEN));

    {
        double pms = 0.0;
        if (!h3_dit_w4a8_prefetch(dit, 0, layers, &pms, error, sizeof(error))) {
            fprintf(stderr, "prefetch failed: %s\n", error);
            goto cleanup;
        }
        printf("prefetch_ms=%.1f\n", pms);
    }

    for (step = 0; step < steps; step++) {
        h3_dit_w4a8_forward_stats fst;
        memset(&fst, 0, sizeof(fst));
        if (!h3_dit_w4a8_forward_video_f32(
                dit, &core, 0, layers, sigmas[step], latent, video_seq, text_h,
                context, pt, ph, pw, 1.0f, velocity, 0, &fst, error,
                sizeof(error))) {
            fprintf(stderr, "forward step %d failed: %s\n", step, error);
            goto cleanup;
        }
        if (!h3_dit_w4a8_euler_step_f32(latent, velocity, n_lat, sigmas[step],
                                        sigmas[step + 1], error,
                                        sizeof(error))) {
            fprintf(stderr, "euler step %d failed: %s\n", step, error);
            goto cleanup;
        }
        if (!all_finite(latent, n_lat)) {
            fprintf(stderr, "latent non-finite at step %d\n", step);
            goto cleanup;
        }
        total_ms += fst.total_ms;
        printf("  step %d: sigma=%.4f→%.4f t_row=%zu lat_absmax=%.6g "
               "forward_ms=%.1f\n",
               step, sigmas[step], sigmas[step + 1], fst.t_row,
               absmax(latent, n_lat), fst.total_ms);
    }
    csum_lat = h3_dit_w4a8_checksum_f32(latent, n_lat);
    h3_dit_w4a8_get_cache_stats(dit, &cst, error, sizeof(error));
    h3_cuda_mem_info(cuda, &free1, NULL, NULL, 0);
    printf("latent_final: absmax=%.6g checksum=0x%016llx total_forward_ms=%.1f\n",
           absmax(latent, n_lat), (unsigned long long)csum_lat, total_ms);
    printf("resident=%s vram_free=%s\n",
           format_bytes(cst.bytes_resident, b1, sizeof(b1)),
           format_bytes(free1, b2, sizeof(b2)));

    /* Prove condition enters the network. */
    {
        float *lat_u = (float *)malloc(n_lat * sizeof(float));
        double ms_u = 0.0;
        if (!lat_u) {
            fprintf(stderr, "OOM uncond\n");
            goto cleanup;
        }
        for (i = 0; i < n_lat; i++) lat_u[i] = hash_to_unit(seed, i);
        for (step = 0; step < steps; step++) {
            h3_dit_w4a8_forward_stats fst;
            memset(&fst, 0, sizeof(fst));
            if (!h3_dit_w4a8_forward_video_f32(
                    dit, &core, 0, layers, sigmas[step], lat_u, video_seq, NULL,
                    0, pt, ph, pw, 1.0f, velocity, 0, &fst, error,
                    sizeof(error)) ||
                !h3_dit_w4a8_euler_step_f32(lat_u, velocity, n_lat,
                                            sigmas[step], sigmas[step + 1],
                                            error, sizeof(error))) {
                free(lat_u);
                fprintf(stderr, "uncond failed: %s\n", error);
                goto cleanup;
            }
            ms_u += fst.total_ms;
        }
        csum_uncond = h3_dit_w4a8_checksum_f32(lat_u, n_lat);
        free(lat_u);
        printf("condition_check: csum_ctx=0x%016llx csum_uncond=0x%016llx %s\n",
               (unsigned long long)csum_lat, (unsigned long long)csum_uncond,
               csum_lat != csum_uncond ? "DISTINCT PASS" : "SAME FAIL");
        if (csum_lat == csum_uncond) goto cleanup;
        (void)ms_u;
    }

    if (out_dir) {
        if (!ensure_dir(out_dir)) {
            fprintf(stderr, "cannot create out dir %s\n", out_dir);
            goto cleanup;
        }
        snprintf(pathbuf, sizeof(pathbuf), "%s/latent_f32.raw", out_dir);
        if (!write_latent_raw(pathbuf, latent, n_lat, error, sizeof(error))) {
            fprintf(stderr, "%s\n", error);
            goto cleanup;
        }
        printf("wrote: %s (%zu floats)\n", pathbuf, n_lat);
    }

    /* E3: optional VAE decode → PPM */
    if (vae_path) {
        if (!h3_vae_int8_store_open(vae_path, &vae_store, error,
                                    sizeof(error))) {
            fprintf(stderr, "vae open failed: %s\n", error);
            goto cleanup;
        }
        latent_chw = h3_vae_int8_unpack_dit_latent_f32(
            latent, latent_t, latent_h, latent_w, error, sizeof(error));
        if (!latent_chw) {
            fprintf(stderr, "unpack latent failed: %s\n", error);
            goto cleanup;
        }
        if (!h3_vae_int8_decode_f32(&vae_store, latent_chw, latent_t, latent_h,
                                    latent_w, vae_blocks, &frames, &vst, error,
                                    sizeof(error))) {
            fprintf(stderr, "vae decode failed: %s\n", error);
            goto cleanup;
        }
        printf("vae_decode: blocks=%u frames=%d %dx%d finite=%d "
               "load_ms=%.1f blocks_ms=%.1f total_ms=%.1f\n",
               vst.n_blocks, frames.frames, frames.width, frames.height,
               vst.all_finite, vst.load_ms, vst.blocks_ms, vst.total_ms);
        if (!vst.all_finite) {
            fprintf(stderr, "vae pixels non-finite\n");
            goto cleanup;
        }
        if (out_dir) {
            snprintf(pathbuf, sizeof(pathbuf), "%s/frame0.ppm", out_dir);
            if (!h3_vae_int8_write_ppm(&frames, 0, pathbuf, error,
                                       sizeof(error))) {
                fprintf(stderr, "ppm write failed: %s\n", error);
                goto cleanup;
            }
            printf("wrote: %s\n", pathbuf);
        }
    } else {
        printf("vae: skipped (pass --vae to decode)\n");
    }

    printf("summary: generate_probe → PASS "
           "(conditioned DiT + latent%s)\n",
           vae_path ? " + VAE ppm" : "");
    rc = 0;

cleanup:
    free(latent);
    free(velocity);
    free(sigmas);
    free(text_emb);
    free(text_h);
    free(token_ids);
    free(pt);
    free(ph);
    free(pw);
    free(latent_chw);
    h3_vae_int8_frames_free(&frames);
    h3_vae_int8_store_free(&vae_store);
    h3_nvfp4_store_free(&te_store);
    h3_dit_w4a8_core_free(&core);
    h3_dit_w4a8_free(dit);
    h3_cuda_free(cuda);
    h3_w4a8_store_free(dit_store);
    return rc;
}
