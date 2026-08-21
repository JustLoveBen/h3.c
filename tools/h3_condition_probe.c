/* Diagnose text-condition effectiveness across prompts.
 * Compares TE embeds, condition_proj/refiner outputs, and 1-step DiT velocity.
 */
#include "h3_dit_w4a8.h"
#include "h3_nvfp4.h"
#include "h3_nvfp4_cuda.h"
#include "h3_token_ids.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float cosine(const float *a, const float *b, size_t n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    size_t i;
    for (i = 0; i < n; i++) {
        double x = a[i], y = b[i];
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if (na < 1e-30 || nb < 1e-30) return 0.0f;
    return (float)(dot / (sqrt(na) * sqrt(nb)));
}

static float mean_abs(const float *a, size_t n) {
    double s = 0.0;
    size_t i;
    for (i = 0; i < n; i++) s += fabs((double)a[i]);
    return (float)(s / (double)(n ? n : 1));
}

static float hash_to_unit(uint64_t seed, size_t index) {
    uint64_t x = seed ^ (uint64_t)(index + 1) * 0x9e3779b97f4a7c15ULL;
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return (float)(int32_t)(x >> 32) * (1.0f / 2147483648.0f);
}

static void fill_normal(uint64_t seed, float *dst, size_t count) {
    size_t i;
    for (i = 0; i + 1 < count; i += 2) {
        float u1 = 0.5f * (hash_to_unit(seed, i) + 1.0f);
        float u2 = 0.5f * (hash_to_unit(seed, i + 1) + 1.0f);
        if (u1 < 1e-7f) u1 = 1e-7f;
        if (u1 > 1.0f - 1e-7f) u1 = 1.0f - 1e-7f;
        {
            float r = sqrtf(-2.0f * logf(u1));
            float th = 6.28318530718f * u2;
            dst[i] = r * cosf(th);
            dst[i + 1] = r * sinf(th);
        }
    }
    if (i < count) dst[i] = hash_to_unit(seed, i);
}

int main(int argc, char **argv) {
    const char *dit_path = NULL, *te_path = NULL;
    const char *files[8];
    const char *names[8];
    int n_files = 0;
    unsigned te_layers = 4;
    unsigned dit_layers = 4;
    size_t context = 32;
    size_t video_seq = 256; /* small probe seq */
    uint64_t seed = 0x483343554441ULL;
    char error[512];
    h3_nvfp4_store te;
    h3_w4a8_store *dit_store = NULL;
    h3_cuda *cuda = NULL;
    h3_dit_w4a8 *dit = NULL;
    h3_dit_w4a8_core core;
    h3_w4a8_cache_config cfg;
    float *emb[8] = {0}, *ref[8] = {0}, *vel[8] = {0};
    float *latent = NULL, *pt = NULL, *ph = NULL, *pw = NULL;
    size_t i, j, n_lat;
    int iarg, rc = 1;

    memset(&te, 0, sizeof(te));
    memset(&core, 0, sizeof(core));

    for (iarg = 1; iarg < argc; iarg++) {
        if (!strcmp(argv[iarg], "--dit") && iarg + 1 < argc)
            dit_path = argv[++iarg];
        else if (!strcmp(argv[iarg], "--te") && iarg + 1 < argc)
            te_path = argv[++iarg];
        else if (!strcmp(argv[iarg], "--te-layers") && iarg + 1 < argc)
            te_layers = (unsigned)strtoul(argv[++iarg], NULL, 10);
        else if (!strcmp(argv[iarg], "--dit-layers") && iarg + 1 < argc)
            dit_layers = (unsigned)strtoul(argv[++iarg], NULL, 10);
        else if (!strcmp(argv[iarg], "--context") && iarg + 1 < argc)
            context = (size_t)strtoul(argv[++iarg], NULL, 10);
        else if (!strcmp(argv[iarg], "--video-seq") && iarg + 1 < argc)
            video_seq = (size_t)strtoul(argv[++iarg], NULL, 10);
        else if (!strcmp(argv[iarg], "--seed") && iarg + 1 < argc)
            seed = strtoull(argv[++iarg], NULL, 0);
        else if (!strcmp(argv[iarg], "--token-ids-file") && iarg + 1 < argc) {
            if (n_files >= 8) {
                fprintf(stderr, "too many token files\n");
                return 2;
            }
            files[n_files] = argv[++iarg];
            names[n_files] = files[n_files];
            n_files++;
        } else if (!strcmp(argv[iarg], "--name") && iarg + 1 < argc &&
                   n_files > 0) {
            names[n_files - 1] = argv[++iarg];
        } else {
            fprintf(stderr,
                    "usage: %s --dit DIT --te TE [--te-layers N] "
                    "[--dit-layers N] [--context T] [--video-seq V] [--seed S] "
                    "(--token-ids-file F [--name N])+ \n",
                    argv[0]);
            return 2;
        }
    }
    if (!dit_path || !te_path || n_files < 2) {
        fprintf(stderr, "need --dit --te and >=2 --token-ids-file\n");
        return 2;
    }

    if (!h3_nvfp4_store_open(te_path, &te, error, sizeof(error))) {
        fprintf(stderr, "te open: %s\n", error);
        return 1;
    }
    dit_store = h3_w4a8_store_open(dit_path, error, sizeof(error));
    if (!dit_store) {
        fprintf(stderr, "dit open: %s\n", error);
        goto cleanup;
    }
    if (!h3_dit_w4a8_load_core(dit_store, &core, error, sizeof(error))) {
        fprintf(stderr, "load_core: %s\n", error);
        goto cleanup;
    }
    cuda = h3_cuda_create(0, error, sizeof(error));
    if (!cuda) {
        fprintf(stderr, "cuda: %s\n", error);
        goto cleanup;
    }
    if (!h3_nvfp4_cuda_enable(cuda, error, sizeof(error))) {
        fprintf(stderr, "te cuda bind: %s\n", error);
        goto cleanup;
    }
    printf("te_backend: %s\n",
           h3_nvfp4_cuda_enabled() ? "cublas" : "host_gemm");
    h3_dit_w4a8_config_init(&cfg);
    cfg.policy = H3_W4A8_CACHE_RESIDENT_ALL;
    dit = h3_dit_w4a8_open(cuda, dit_store, &cfg, error, sizeof(error));
    if (!dit) {
        fprintf(stderr, "dit_w4a8_open: %s\n", error);
        goto cleanup;
    }

    n_lat = video_seq * (size_t)H3_DIT_W4A8_VIDEO_PATCH;
    latent = (float *)malloc(n_lat * sizeof(float));
    pt = (float *)malloc(video_seq * sizeof(float));
    ph = (float *)malloc(video_seq * sizeof(float));
    pw = (float *)malloc(video_seq * sizeof(float));
    if (!latent || !pt || !ph || !pw) {
        fprintf(stderr, "OOM\n");
        goto cleanup;
    }
    fill_normal(seed, latent, n_lat);
    for (i = 0; i < video_seq; i++) {
        pt[i] = (float)(i / 16);
        ph[i] = (float)((i / 4) % 4);
        pw[i] = (float)(i % 4);
    }

    printf("probe: te_layers=%u dit_layers=%u context=%zu video_seq=%zu "
           "seed=0x%llx n_files=%d has_condition=%d has_refiner=%d\n",
           te_layers, dit_layers, context, video_seq,
           (unsigned long long)seed, n_files, core.condition_w != NULL,
           core.has_refiner);

    /* Dump AdaLN modality scales/gates — text scale≈-1 would kill conditioning. */
    {
        h3_dit_w4a8_block_aux aux0;
        float time_feat[H3_DIT_W4A8_TIME_FEAT];
        float *mod = NULL;
        float sigmas_try[4] = {1.0f, 0.75f, 0.5f, 0.25f};
        int si;
        memset(&aux0, 0, sizeof(aux0));
        if (h3_dit_w4a8_load_block_aux(dit_store, 0, &aux0, error,
                                       sizeof(error))) {
            mod = (float *)malloc(H3_DIT_W4A8_ADALN_OUT * sizeof(float));
            if (mod) {
                printf("adaln_modality_dump (block0):\n");
                for (si = 0; si < 4; si++) {
                    size_t trow = h3_dit_w4a8_t_row_from_sigma(
                        sigmas_try[si], core.t_table_rows);
                    size_t m, near;
                    memcpy(time_feat,
                           core.t_table + trow * H3_DIT_W4A8_TIME_FEAT,
                           sizeof(time_feat));
                    if (!h3_dit_w4a8_adaln_project_f32(&aux0, time_feat, mod,
                                                       error, sizeof(error)))
                        break;
                    printf("  sigma=%.2f t_row=%zu\n", sigmas_try[si], trow);
                    for (m = 0; m < 3; m++) {
                        const float *base =
                            mod + m * (size_t)H3_DIT_W4A8_SLOTS *
                                      H3_DIT_W4A8_HIDDEN;
                        const float *scale_msa =
                            base + 1 * H3_DIT_W4A8_HIDDEN;
                        const float *gate_msa =
                            base + 2 * H3_DIT_W4A8_HIDDEN;
                        near = 0;
                        for (i = 0; i < (size_t)H3_DIT_W4A8_HIDDEN; i++)
                            if (fabsf(scale_msa[i] + 1.0f) < 0.05f) near++;
                        printf("    %s scale_msa_absmean=%.4g "
                               "gate_msa_absmean=%.4g near-1=%.3f\n",
                               m == 0   ? "VIDEO"
                               : m == 1 ? "TEXT"
                                        : "AUDIO",
                               mean_abs(scale_msa, H3_DIT_W4A8_HIDDEN),
                               mean_abs(gate_msa, H3_DIT_W4A8_HIDDEN),
                               (float)near / (float)H3_DIT_W4A8_HIDDEN);
                    }
                }
            }
            free(mod);
            h3_dit_w4a8_block_aux_free(&aux0);
        } else {
            printf("adaln dump skipped: %s\n", error);
        }
    }

    for (i = 0; i < (size_t)n_files; i++) {
        int32_t *ids = NULL;
        size_t n_ids = 0;
        h3_nvfp4_encode_stats est;
        h3_dit_w4a8_forward_stats fst;
        memset(&est, 0, sizeof(est));
        memset(&fst, 0, sizeof(fst));
        if (!h3_token_ids_load_file(files[i], &ids, &n_ids, error,
                                    sizeof(error))) {
            fprintf(stderr, "token %s: %s\n", names[i], error);
            goto cleanup;
        }
        printf("tokens[%s]: raw_n=%zu head=", names[i], n_ids);
        for (j = 0; j < n_ids && j < 8; j++) printf("%d ", ids[j]);
        printf("\n");
        if (!h3_token_ids_fit(&ids, &n_ids, context, error, sizeof(error))) {
            fprintf(stderr, "fit %s: %s\n", names[i], error);
            free(ids);
            goto cleanup;
        }
        emb[i] = h3_nvfp4_encode_layers_f32(&te, ids, context, te_layers, &est,
                                            error, sizeof(error));
        free(ids);
        if (!emb[i]) {
            fprintf(stderr, "TE %s: %s\n", names[i], error);
            goto cleanup;
        }
        printf("  te[%s]: absmean=%.6g finite=%d embed_ms=%.1f layers_ms=%.1f "
               "cuda=%d\n",
               names[i], mean_abs(emb[i], context * H3_NVFP4_TEXT_DIM),
               est.all_finite, est.embed_ms, est.layers_ms, est.used_cuda);

        ref[i] = (float *)malloc(context * (size_t)H3_DIT_W4A8_HIDDEN *
                                 sizeof(float));
        if (!ref[i]) goto cleanup;
        if (!h3_dit_w4a8_refine_text_f32(dit, &core, emb[i], context, ref[i],
                                         error, sizeof(error))) {
            fprintf(stderr, "refine %s: %s\n", names[i], error);
            goto cleanup;
        }
        {
            const char *tg = getenv("H3_DIT_TEXT_GAIN");
            float gain = tg ? (float)strtod(tg, NULL) : 1.0f;
            if (isfinite(gain) && fabsf(gain - 1.0f) > 1e-6f && gain > 0.0f) {
                size_t n = context * (size_t)H3_DIT_W4A8_HIDDEN;
                size_t k;
                for (k = 0; k < n; k++) ref[i][k] *= gain;
                if (i == 0)
                    printf("text_gain: H3_DIT_TEXT_GAIN=%.3g applied to refined\n",
                           (double)gain);
            }
        }
        printf("  refined[%s]: absmean=%.6g\n", names[i],
               mean_abs(ref[i], context * H3_DIT_W4A8_HIDDEN));

        vel[i] = (float *)malloc(n_lat * sizeof(float));
        if (!vel[i]) goto cleanup;
        /* restore identical noise for each condition */
        fill_normal(seed, latent, n_lat);
        if (!h3_dit_w4a8_forward_video_f32(
                dit, &core, 0, dit_layers, 1.0f, latent, video_seq, ref[i],
                context, pt, ph, pw, 1.0f, vel[i], 1, &fst, error,
                sizeof(error))) {
            fprintf(stderr, "forward %s: %s\n", names[i], error);
            goto cleanup;
        }
        printf("  vel[%s]: absmean=%.6g csum=0x%016llx forward_ms=%.1f\n",
               names[i], mean_abs(vel[i], n_lat),
               (unsigned long long)h3_dit_w4a8_checksum_f32(vel[i], n_lat),
               fst.total_ms);
    }

    printf("\n=== pairwise cosine ===\n");
    for (i = 0; i < (size_t)n_files; i++) {
        for (j = i + 1; j < (size_t)n_files; j++) {
            float ce = cosine(emb[i], emb[j],
                              context * (size_t)H3_NVFP4_TEXT_DIM);
            float cr = cosine(ref[i], ref[j],
                              context * (size_t)H3_DIT_W4A8_HIDDEN);
            float cv = cosine(vel[i], vel[j], n_lat);
            int same_csum =
                h3_dit_w4a8_checksum_f32(vel[i], n_lat) ==
                h3_dit_w4a8_checksum_f32(vel[j], n_lat);
            printf("%s vs %s: te_cos=%.6f refined_cos=%.6f vel_cos=%.6f "
                   "vel_csum_%s\n",
                   names[i], names[j], ce, cr, cv,
                   same_csum ? "SAME" : "DIFF");
        }
    }

    /* seed sensitivity with first prompt */
    {
        float *vel_b = (float *)malloc(n_lat * sizeof(float));
        h3_dit_w4a8_forward_stats fst;
        memset(&fst, 0, sizeof(fst));
        if (!vel_b) goto cleanup;
        fill_normal(seed ^ 0xDEADBEEFULL, latent, n_lat);
        if (!h3_dit_w4a8_forward_video_f32(
                dit, &core, 0, dit_layers, 1.0f, latent, video_seq, ref[0],
                context, pt, ph, pw, 1.0f, vel_b, 1, &fst, error,
                sizeof(error))) {
            fprintf(stderr, "seed forward: %s\n", error);
            free(vel_b);
            goto cleanup;
        }
        printf("\nseed_sensitivity[%s]: same_prompt different_seed "
               "vel_cos=%.6f csum_%s\n",
               names[0], cosine(vel[0], vel_b, n_lat),
               h3_dit_w4a8_checksum_f32(vel[0], n_lat) ==
                       h3_dit_w4a8_checksum_f32(vel_b, n_lat)
                   ? "SAME"
                   : "DIFF");
        free(vel_b);
    }

    /* uncond (zeros text) vs first */
    {
        float *z = (float *)calloc(context * (size_t)H3_DIT_W4A8_HIDDEN,
                                   sizeof(float));
        float *vel_u = (float *)malloc(n_lat * sizeof(float));
        float *vel_n = (float *)malloc(n_lat * sizeof(float));
        h3_dit_w4a8_forward_stats fst;
        memset(&fst, 0, sizeof(fst));
        if (!z || !vel_u || !vel_n) {
            free(z);
            free(vel_u);
            free(vel_n);
            goto cleanup;
        }
        fill_normal(seed, latent, n_lat);
        if (!h3_dit_w4a8_forward_video_f32(
                dit, &core, 0, dit_layers, 0.5f, latent, video_seq, z, context,
                pt, ph, pw, 1.0f, vel_u, 1, &fst, error, sizeof(error))) {
            fprintf(stderr, "uncond forward: %s\n", error);
            free(z);
            free(vel_u);
            free(vel_n);
            goto cleanup;
        }
        fill_normal(seed, latent, n_lat);
        if (!h3_dit_w4a8_forward_video_f32(
                dit, &core, 0, dit_layers, 0.5f, latent, video_seq, ref[0],
                context, pt, ph, pw, 1.0f, vel_n, 1, &fst, error,
                sizeof(error))) {
            fprintf(stderr, "cond@0.5 forward: %s\n", error);
            free(z);
            free(vel_u);
            free(vel_n);
            goto cleanup;
        }
        {
            double num = 0.0, den = 0.0;
            size_t k;
            for (k = 0; k < n_lat; k++) {
                double d = (double)vel_n[k] - (double)vel_u[k];
                num += d * d;
                den += (double)vel_n[k] * (double)vel_n[k];
            }
            printf("cond_vs_zero_text@sigma0.5[%s]: vel_cos=%.6f rel_l2=%.6f "
                   "csum_%s\n",
                   names[0], cosine(vel_n, vel_u, n_lat),
                   (float)sqrt(num / (den + 1e-30)),
                   h3_dit_w4a8_checksum_f32(vel_n, n_lat) ==
                           h3_dit_w4a8_checksum_f32(vel_u, n_lat)
                       ? "SAME"
                       : "DIFF");
        }
        /* CFG-amplified: v' = vu + s*(vc-vu); compare two prompts at cfg=7 */
        if (n_files >= 2) {
            float *vel_a = (float *)malloc(n_lat * sizeof(float));
            float *vel_b = (float *)malloc(n_lat * sizeof(float));
            float *cfg_a = (float *)malloc(n_lat * sizeof(float));
            float *cfg_b = (float *)malloc(n_lat * sizeof(float));
            const float cfg = 7.0f;
            size_t k;
            memset(&fst, 0, sizeof(fst));
            if (vel_a && vel_b && cfg_a && cfg_b) {
                fill_normal(seed, latent, n_lat);
                h3_dit_w4a8_forward_video_f32(
                    dit, &core, 0, dit_layers, 0.5f, latent, video_seq, ref[0],
                    context, pt, ph, pw, 1.0f, vel_a, 1, &fst, error,
                    sizeof(error));
                fill_normal(seed, latent, n_lat);
                h3_dit_w4a8_forward_video_f32(
                    dit, &core, 0, dit_layers, 0.5f, latent, video_seq, ref[1],
                    context, pt, ph, pw, 1.0f, vel_b, 1, &fst, error,
                    sizeof(error));
                for (k = 0; k < n_lat; k++) {
                    cfg_a[k] = vel_u[k] + cfg * (vel_a[k] - vel_u[k]);
                    cfg_b[k] = vel_u[k] + cfg * (vel_b[k] - vel_u[k]);
                }
                printf("cfg7_prompt_delta[%s vs %s]: raw_cos=%.6f cfg_cos=%.6f\n",
                       names[0], names[1], cosine(vel_a, vel_b, n_lat),
                       cosine(cfg_a, cfg_b, n_lat));
            }
            free(vel_a);
            free(vel_b);
            free(cfg_a);
            free(cfg_b);
        }
        /* no-text-seq (text_seq=0) */
        fill_normal(seed, latent, n_lat);
        if (!h3_dit_w4a8_forward_video_f32(
                dit, &core, 0, dit_layers, 0.5f, latent, video_seq, NULL, 0, pt,
                ph, pw, 1.0f, vel_u, 1, &fst, error, sizeof(error))) {
            fprintf(stderr, "empty-text forward: %s\n", error);
        } else {
            printf("cond_vs_empty_seq[%s]: vel_cos=%.6f csum_%s\n", names[0],
                   cosine(vel_n, vel_u, n_lat),
                   h3_dit_w4a8_checksum_f32(vel_n, n_lat) ==
                           h3_dit_w4a8_checksum_f32(vel_u, n_lat)
                       ? "SAME"
                       : "DIFF");
        }
        free(z);
        free(vel_u);
        free(vel_n);
    }

    rc = 0;
cleanup:
    for (i = 0; i < 8; i++) {
        free(emb[i]);
        free(ref[i]);
        free(vel[i]);
    }
    free(latent);
    free(pt);
    free(ph);
    free(pw);
    h3_dit_w4a8_core_free(&core);
    h3_dit_w4a8_free(dit);
    h3_cuda_free(cuda);
    h3_w4a8_store_free(dit_store);
    h3_nvfp4_store_free(&te);
    return rc;
}
