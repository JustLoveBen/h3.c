/* Au3/Au4: DiT W4A8 audio patch + AV forward + Euler smoke. */
#include "h3_dit_w4a8.h"

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

static uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

static float hash_u(uint64_t seed, size_t i) {
    uint64_t h = mix64(seed ^ (uint64_t)(i + 1) * 0x9e3779b97f4a7c15ULL);
    return (float)(int32_t)(h >> 32) * (1.0f / 2147483648.0f);
}

static int build_sigmas(int steps, float *sigmas) {
    int i;
    if (!sigmas || steps < 1) return 0;
    for (i = 0; i <= steps; i++)
        sigmas[i] = 1.0f - (float)i / (float)steps;
    sigmas[steps] = 0.0f;
    return 1;
}

int main(int argc, char **argv) {
    const char *dit_path =
        "/home/haiyang/h3_c/models/kijai/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors";
    unsigned layers = 2;
    int steps = 2;
    int audio_t = 4;
    int vh = 2, vw = 2, vt = 2; /* tiny video grid */
    int device = 0, iarg, step;
    char error[512];
    h3_cuda *cuda = NULL;
    h3_w4a8_store *store = NULL;
    h3_dit_w4a8 *dit = NULL;
    h3_dit_w4a8_core core;
    h3_w4a8_cache_config cfg;
    float *video_lat = NULL, *audio_c2t = NULL, *audio_rows = NULL;
    float *v_vel = NULL, *a_vel = NULL, *sigmas = NULL;
    int *vpt = NULL, *vph = NULL, *vpw = NULL, *apt = NULL;
    size_t video_seq, audio_seq, n_v, n_a, i;
    double total_ms = 0.0;

    memset(&core, 0, sizeof(core));
    for (iarg = 1; iarg < argc; iarg++) {
        if (!strcmp(argv[iarg], "--dit") && iarg + 1 < argc)
            dit_path = argv[++iarg];
        else if (!strcmp(argv[iarg], "--layers") && iarg + 1 < argc)
            layers = (unsigned)atoi(argv[++iarg]);
        else if (!strcmp(argv[iarg], "--steps") && iarg + 1 < argc)
            steps = atoi(argv[++iarg]);
        else if (!strcmp(argv[iarg], "--audio-t") && iarg + 1 < argc)
            audio_t = atoi(argv[++iarg]);
        else if (!strcmp(argv[iarg], "--device") && iarg + 1 < argc)
            device = atoi(argv[++iarg]);
        else {
            fprintf(stderr,
                    "usage: %s [--dit PATH] [--layers N] [--steps N] "
                    "[--audio-t T]\n",
                    argv[0]);
            return 2;
        }
    }

    video_seq = (size_t)vt * (size_t)vh * (size_t)vw;
    audio_seq = (size_t)2 * (size_t)audio_t;
    n_v = video_seq * H3_DIT_W4A8_VIDEO_PATCH;
    n_a = audio_seq * H3_DIT_W4A8_AUDIO_CH;

    video_lat = calloc(n_v, sizeof(float));
    audio_c2t = calloc((size_t)H3_DIT_W4A8_AUDIO_CH * 2 * (size_t)audio_t,
                       sizeof(float));
    audio_rows = calloc(n_a, sizeof(float));
    v_vel = calloc(n_v, sizeof(float));
    a_vel = calloc(n_a, sizeof(float));
    sigmas = calloc((size_t)steps + 1, sizeof(float));
    vpt = calloc(video_seq, sizeof(int));
    vph = calloc(video_seq, sizeof(int));
    vpw = calloc(video_seq, sizeof(int));
    apt = calloc(audio_seq, sizeof(int));
    if (!video_lat || !audio_c2t || !audio_rows || !v_vel || !a_vel || !sigmas ||
        !vpt || !vph || !vpw || !apt) {
        fprintf(stderr, "OOM\n");
        return 1;
    }

    for (i = 0; i < n_v; i++) video_lat[i] = hash_u(0xA11D10ULL, i) * 0.1f;
    for (i = 0; i < (size_t)H3_DIT_W4A8_AUDIO_CH * 2 * (size_t)audio_t; i++)
        audio_c2t[i] = hash_u(0xA11D20ULL, i) * 0.1f;
    if (!h3_dit_w4a8_pack_audio_latent(audio_c2t, audio_t, audio_rows, error,
                                       sizeof(error))) {
        fprintf(stderr, "pack failed: %s\n", error);
        return 1;
    }
    {
        size_t idx = 0;
        int t, h, w;
        for (t = 0; t < vt; t++)
            for (h = 0; h < vh; h++)
                for (w = 0; w < vw; w++) {
                    vpt[idx] = t;
                    vph[idx] = h;
                    vpw[idx] = w;
                    idx++;
                }
        for (i = 0; i < audio_seq; i++)
            apt[i] = (int)(i % (size_t)audio_t);
    }
    if (!build_sigmas(steps, sigmas)) return 1;

    cuda = h3_cuda_create(device, error, sizeof(error));
    if (!cuda) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    store = h3_w4a8_store_open(dit_path, error, sizeof(error));
    if (!store) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    h3_dit_w4a8_config_init(&cfg);
    dit = h3_dit_w4a8_open(cuda, store, &cfg, error, sizeof(error));
    if (!dit || !h3_dit_w4a8_load_core(store, &core, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    if (!core.audio_patch_w || !core.final_audio_w) {
        fprintf(stderr, "FAIL: audio weights missing in DiT store\n");
        return 1;
    }
    if (!h3_dit_w4a8_prefetch(dit, 0, layers, NULL, error, sizeof(error))) {
        fprintf(stderr, "prefetch: %s\n", error);
        return 1;
    }

    printf("Au3/Au4 DiT AV smoke layers=%u steps=%d video_seq=%zu audio_seq=%zu\n",
           layers, steps, video_seq, audio_seq);

    for (step = 0; step < steps; step++) {
        h3_dit_w4a8_forward_stats fst;
        memset(&fst, 0, sizeof(fst));
        if (!h3_dit_w4a8_forward_av_f32(
                dit, &core, 0, layers, sigmas[step], sigmas[step], video_lat,
                video_seq, audio_rows, audio_seq, NULL, 0, (const float *)vpt,
                (const float *)vph, (const float *)vpw, (const float *)apt,
                (const float *)apt, (const float *)apt, 1.0f, v_vel, a_vel, 0,
                &fst, error, sizeof(error))) {
            fprintf(stderr, "forward_av step %d: %s\n", step, error);
            return 1;
        }
        if (!all_finite(v_vel, n_v) || !all_finite(a_vel, n_a)) {
            fprintf(stderr, "non-finite velocity at step %d\n", step);
            return 1;
        }
        if (!h3_dit_w4a8_euler_step_f32(video_lat, v_vel, n_v, sigmas[step],
                                        sigmas[step + 1], error,
                                        sizeof(error)) ||
            !h3_dit_w4a8_euler_step_f32(audio_rows, a_vel, n_a, sigmas[step],
                                        sigmas[step + 1], error,
                                        sizeof(error))) {
            fprintf(stderr, "euler: %s\n", error);
            return 1;
        }
        total_ms += fst.total_ms;
        printf("  step %d: v_absmax=%.5g a_absmax=%.5g forward_ms=%.1f "
               "finite=%d\n",
               step, absmax(v_vel, n_v), absmax(a_vel, n_a), fst.total_ms,
               fst.all_finite);
    }

    if (!h3_dit_w4a8_unpack_audio_latent(audio_rows, audio_t, audio_c2t, error,
                                         sizeof(error))) {
        fprintf(stderr, "unpack: %s\n", error);
        return 1;
    }
    printf("PASS Au3/Au4 AV sample total_ms=%.1f audio_lat_absmax=%.5g\n",
           total_ms, absmax(audio_c2t, (size_t)H3_DIT_W4A8_AUDIO_CH * 2 *
                                           (size_t)audio_t));

    h3_dit_w4a8_core_free(&core);
    h3_dit_w4a8_free(dit);
    h3_w4a8_store_free(store);
    h3_cuda_free(cuda);
    free(video_lat);
    free(audio_c2t);
    free(audio_rows);
    free(v_vel);
    free(a_vel);
    free(sigmas);
    free(vpt);
    free(vph);
    free(vpw);
    free(apt);
    return 0;
}
