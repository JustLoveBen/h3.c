/* G4: Formal CUDA bypass generate CLI (T2V subset).
 *
 * Delivery note (A7): this is the CUDA *bypass* product entry — it does NOT
 * replace Darwin Metal `h3_generate` / `h3_dit.c`. Same weights + shapes;
 * pure C/CUDA.
 *
 * Default: 5s 480P (864x480), Euler steps, configurable DiT/TE/VAE depth.
 * Writes PPM frames (and optional latent.raw) under --out.
 *
 *   cmake --build build-cuda -j --target h3_cuda_generate
 *   ./build-cuda/h3_cuda_generate --dit DIT --vae VAE --te TE --out DIR
 */
#include "h3_dit_w4a8.h"
#include "h3_dit_int8.h"
#include "h3_ffmpeg.h"
#include "h3_multimodal_cuda.h"
#include "h3_nvfp4.h"
#include "h3_nvfp4_cuda.h"
#include "h3_token_ids.h"
#include "h3_vae_encode_f32.h"
#include "h3_vae_int8.h"
#include "h3_vae_int8_cuda.h"
#include "h3_vision_cuda.h"
#include "h3_audio_vae_cuda.h"
#include "h3_audio_vae_layout.h"
#include "h3_cuda_device_caps.h"
#include "h3_generate_preset.h"
#include "h3_lora.h"
#include "h3_sdpa.h"
#include "h3_step_cache.h"
#include "h3_tae.h"
#include "h3_te_cache.h"
#include "h3_ulysses.h"
#include "h3_ulysses_comm.h"

#include <cuda_runtime.h>
#include <fcntl.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

/* Text TE length policy (CUDA bypass). Official MiniMax API: ≤7000 chars.
 * Comfy Qwen3-VL TE: max_length≈∞, pad_to_max_length=False. Metal: exact len.
 * 2048 tokens ≈ headroom for the API char cap under Qwen BPE. */
#ifndef H3_CUDA_TEXT_CONTEXT_DEFAULT
#define H3_CUDA_TEXT_CONTEXT_DEFAULT 256u
#endif
#ifndef H3_CUDA_TEXT_CONTEXT_HARD_MAX
/* F2VA 1×Picture + 1626 prompt ≈ 2039. FL2VA 2×Picture + same prompt ≈ 2452. */
#define H3_CUDA_TEXT_CONTEXT_HARD_MAX 4096u
#endif

#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static int h3_path_is_file(const char *path) {
    struct stat st;
    return path && *path && stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

static void h3_dirname_copy(const char *path, char *out, size_t n) {
    const char *slash;
    if (!out || n < 2) return;
    if (!path || !*path) {
        snprintf(out, n, ".");
        return;
    }
    slash = strrchr(path, '/');
    if (!slash) {
        snprintf(out, n, ".");
        return;
    }
    if (slash == path) {
        snprintf(out, n, "/");
        return;
    }
    {
        size_t len = (size_t)(slash - path);
        if (len >= n) len = n - 1;
        memcpy(out, path, len);
        out[len] = '\0';
    }
}

/* T2VA default: decode + mux audio (Metal h3_generate always does).
 * Look next to --vae, H3_AUDIO_VAE, then the Comfy-Org filename. */
static int h3_resolve_default_audio_vae(const char *vae_path,
                                        const char *dit_path, char *out,
                                        size_t out_size) {
    static const char *k_name = "minimax_h3_audio_vae_fp32.safetensors";
    static const char *k_fallback[] = {
        "/home/haiyang/h3_c/models/comfy-org/vae/"
        "minimax_h3_audio_vae_fp32.safetensors",
        "models/comfy-org/vae/minimax_h3_audio_vae_fp32.safetensors",
        NULL
    };
    const char *env = getenv("H3_AUDIO_VAE");
    char dir[768], trial[1024];
    size_t i;
    if (env && h3_path_is_file(env)) {
        snprintf(out, out_size, "%s", env);
        return 1;
    }
    if (vae_path && *vae_path) {
        h3_dirname_copy(vae_path, dir, sizeof(dir));
        snprintf(trial, sizeof(trial), "%s/%s", dir, k_name);
        if (h3_path_is_file(trial)) {
            snprintf(out, out_size, "%s", trial);
            return 1;
        }
    }
    if (dit_path && *dit_path) {
        h3_dirname_copy(dit_path, dir, sizeof(dir));
        snprintf(trial, sizeof(trial), "%s/../vae/%s", dir, k_name);
        if (h3_path_is_file(trial)) {
            snprintf(out, out_size, "%s", trial);
            return 1;
        }
    }
    for (i = 0; k_fallback[i]; i++) {
        if (h3_path_is_file(k_fallback[i])) {
            snprintf(out, out_size, "%s", k_fallback[i]);
            return 1;
        }
    }
    return 0;
}

typedef struct {
    h3_cuda *cuda;
    size_t seq;
    size_t heads;
    size_t dim;
    float scale;
    int causal;
    char err[256];
    int ok;
} sdpa_prebuild_job;

static void *sdpa_prebuild_th(void *arg) {
    sdpa_prebuild_job *j = (sdpa_prebuild_job *)arg;
    h3_cuda_device_info info;
    if (!j) return NULL;
    memset(&info, 0, sizeof(info));
    if (h3_cuda_get_info(j->cuda, &info))
        (void)cudaSetDevice(info.device_index);
    j->ok = h3_cuda_sdpa_prebuild(j->cuda, j->seq, j->heads, j->dim, j->scale,
                                  j->causal, j->err, sizeof(j->err));
    return NULL;
}

/* TE-overlap: disk → pinned host only (no GPU). Step1 then H2D-only. */
typedef struct {
    h3_dit_w4a8 *dit;
    unsigned first;
    unsigned n;
    double ms;
    char err[256];
    int ok;
} dit_host_stage_job;

static void *dit_host_stage_th(void *arg) {
    dit_host_stage_job *j = (dit_host_stage_job *)arg;
    if (!j || !j->dit) return NULL;
    j->ok = h3_dit_w4a8_stage_host(j->dit, j->first, j->n, &j->ms, j->err,
                                   sizeof(j->err));
    return NULL;
}

typedef struct {
    dit_host_stage_job *job;
    pthread_t *th;
    int *ok;
    double *t0;
} host_stage_kick;

static void kick_dit_host_stage(void *arg) {
    host_stage_kick *k = (host_stage_kick *)arg;
    if (!k || !k->job || !k->job->dit || !k->th || !k->ok || !k->t0) return;
    if (*k->ok) return;
    *k->t0 = now_ms();
    if (pthread_create(k->th, NULL, dit_host_stage_th, k->job) == 0) {
        *k->ok = 1;
        printf("dit_host_stage: after_te_packed disk→host blocks=%u "
               "(no GPU; H3_DIT_HOST_STAGE_TE=0 disables)\n",
               k->job->n);
    } else {
        printf("dit_host_stage: pthread_create failed; step1 will disk-load\n");
    }
    fflush(stdout);
}

static void *vae_willneed_th(void *arg) {
    const char *path = (const char *)arg;
    int fd;
    if (!path || !path[0]) return NULL;
    fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
#ifdef POSIX_FADV_WILLNEED
    (void)posix_fadvise(fd, 0, 0, POSIX_FADV_WILLNEED);
#endif
    close(fd);
    return NULL;
}

typedef struct {
    h3_cuda *cuda;
    const char *audio_vae_path;
    const float *audio_c2t;
    int audio_t;
    h3_audio_vae_cuda_waveform *awave;
    h3_audio_vae_cuda_stats *ast;
    char err[512];
    int ok;
} audio_decode_job;

static void *audio_decode_th(void *arg) {
    audio_decode_job *j = (audio_decode_job *)arg;
    h3_cuda_device_info info;
    if (!j) return NULL;
    memset(&info, 0, sizeof(info));
    if (h3_cuda_get_info(j->cuda, &info))
        (void)cudaSetDevice(info.device_index);
    j->ok = h3_audio_vae_cuda_decode(j->cuda, j->audio_vae_path, j->audio_c2t,
                                     j->audio_t, NULL, NULL, j->awave, j->ast,
                                     j->err, sizeof(j->err));
    return NULL;
}

static uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

static int load_f32_raw(const char *path, float *dst, size_t n, char *error,
                        size_t error_size) {
    FILE *fp;
    size_t got;
    if (!path || !dst || n == 0) {
        snprintf(error, error_size, "invalid load_f32_raw args");
        return 0;
    }
    fp = fopen(path, "rb");
    if (!fp) {
        snprintf(error, error_size, "cannot open %s", path);
        return 0;
    }
    got = fread(dst, sizeof(float), n, fp);
    fclose(fp);
    if (got != n) {
        snprintf(error, error_size, "expected %zu f32 from %s, got %zu", n, path,
                 got);
        return 0;
    }
    return 1;
}

static uint8_t *frames_to_rgb24(const h3_vae_int8_frames *frames, size_t *nbytes,
                                char *error, size_t error_size) {
    size_t n, i;
    uint8_t *out;
    if (!frames || !frames->rgb || frames->frames < 1) {
        snprintf(error, error_size, "no frames for mux");
        return NULL;
    }
    n = (size_t)frames->frames * (size_t)frames->width *
        (size_t)frames->height * 3;
    out = (uint8_t *)malloc(n);
    if (!out) {
        snprintf(error, error_size, "OOM rgb24 mux buffer");
        return NULL;
    }
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (i = 0; i < n / 3; i++) {
        float r = frames->rgb[i * 3 + 0];
        float g = frames->rgb[i * 3 + 1];
        float b = frames->rgb[i * 3 + 2];
        if (r < 0.0f) r = 0.0f;
        if (g < 0.0f) g = 0.0f;
        if (b < 0.0f) b = 0.0f;
        if (r > 1.0f) r = 1.0f;
        if (g > 1.0f) g = 1.0f;
        if (b > 1.0f) b = 1.0f;
        out[i * 3 + 0] = (uint8_t)(r * 255.0f + 0.5f);
        out[i * 3 + 1] = (uint8_t)(g * 255.0f + 0.5f);
        out[i * 3 + 2] = (uint8_t)(b * 255.0f + 0.5f);
    }
    if (nbytes) *nbytes = n;
    return out;
}

static int write_ppm_rgb24(const char *path, const uint8_t *rgb, int width,
                           int height, char *error, size_t error_size) {
    FILE *fp;
    size_t n;
    if (!path || !rgb || width < 1 || height < 1) {
        snprintf(error, error_size, "invalid ppm rgb24 args");
        return 0;
    }
    fp = fopen(path, "wb");
    if (!fp) {
        snprintf(error, error_size, "cannot open ppm for write");
        return 0;
    }
    fprintf(fp, "P6\n%d %d\n255\n", width, height);
    n = (size_t)width * (size_t)height * 3;
    if (fwrite(rgb, 1, n, fp) != n) {
        fclose(fp);
        snprintf(error, error_size, "ppm fwrite short");
        return 0;
    }
    fclose(fp);
    return 1;
}

/* Channel-major [3,H,W] RGB in [0,1] → P6, for f0-vs-jpeg quality gate. */
static int write_chw01_ppm(const char *path, const float *chw, int width,
                           int height, char *error, size_t error_size) {
    FILE *fp;
    int y, x;
    if (!path || !chw || width < 1 || height < 1) {
        snprintf(error, error_size, "invalid ppm args");
        return 0;
    }
    fp = fopen(path, "wb");
    if (!fp) {
        snprintf(error, error_size, "cannot write %s", path);
        return 0;
    }
    fprintf(fp, "P6\n%d %d\n255\n", width, height);
    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            float r = chw[(size_t)0 * (size_t)height * (size_t)width +
                          (size_t)y * (size_t)width + (size_t)x];
            float g = chw[(size_t)1 * (size_t)height * (size_t)width +
                          (size_t)y * (size_t)width + (size_t)x];
            float b = chw[(size_t)2 * (size_t)height * (size_t)width +
                          (size_t)y * (size_t)width + (size_t)x];
            uint8_t px[3];
            if (r < 0.0f) r = 0.0f;
            if (g < 0.0f) g = 0.0f;
            if (b < 0.0f) b = 0.0f;
            if (r > 1.0f) r = 1.0f;
            if (g > 1.0f) g = 1.0f;
            if (b > 1.0f) b = 1.0f;
            px[0] = (uint8_t)(r * 255.0f + 0.5f);
            px[1] = (uint8_t)(g * 255.0f + 0.5f);
            px[2] = (uint8_t)(b * 255.0f + 0.5f);
            fwrite(px, 1, 3, fp);
        }
    }
    fclose(fp);
    return 1;
}

static const char *official_keyframe_flag(const char *first, const char *last) {
    if (first && last) return "--first-frame/--last-frame";
    if (last) return "--last-frame";
    return "--first-frame";
}

/* Last-frame COND RoPE time. Metal uses aligned last pixel
 * (text_len + span_sum(T) - 5/3 == text_len + (aligned-1)*(5/3)).
 * CUDA muxes frames_req of VAE output (5s: 120 of 124). Binding the
 * padded last pixel (123) parks the lock in dropped frames 120–123
 * (latent t=T-1); mux f119 is the previous 1-frame latent. Product
 * last-frame must use mux last so 5.000s/120 actually shows it. */
static float fl2va_last_condition_time(size_t text_len, int last_pixel) {
    if (last_pixel < 0)
        last_pixel = 0;
    return (float)text_len + (5.0f / 3.0f) * (float)last_pixel;
}

#ifndef H3_CUDA_MAX_REF_IMAGES
/* Picture 1/2 prefixes exist in h3_multimodal_cuda (Metal max is 9). */
#define H3_CUDA_MAX_REF_IMAGES 2
#endif
#ifndef H3_CUDA_MAX_REF_AUDIOS
#define H3_CUDA_MAX_REF_AUDIOS 3
#endif
#ifndef H3_CUDA_MAX_REF_VIDEO_BLOCKS
#define H3_CUDA_MAX_REF_VIDEO_BLOCKS 8
#endif

static int path_contains_ci(const char *path, const char *needle) {
    size_t n, m, i, j;
    if (!path || !needle) return 0;
    n = strlen(path);
    m = strlen(needle);
    if (!m || m > n) return 0;
    for (i = 0; i + m <= n; i++) {
        for (j = 0; j < m; j++) {
            unsigned char a = (unsigned char)path[i + j];
            unsigned char b = (unsigned char)needle[j];
            if (a >= 'A' && a <= 'Z') a = (unsigned char)(a + 32);
            if (b >= 'A' && b <= 'Z') b = (unsigned char)(b + 32);
            if (a != b) break;
        }
        if (j == m) return 1;
    }
    return 0;
}

/* 1-frame still → patchified COND rows. Caller owns `rows` (row_cap floats). */
static int official_encode_cond_rows(const char *enc_path, const float *rgb,
                                     int height, int width, size_t row_cap,
                                     float *rows, size_t *row_elems_out,
                                     int expect_h, int expect_w, char *error,
                                     size_t error_size) {
    h3_vae_latent_f32 enc;
    size_t cond_seq, row_elems;
    memset(&enc, 0, sizeof(enc));
    if (!h3_vae_encode_f32(enc_path, rgb, 1, height, width, &enc, error,
                           error_size))
        return 0;
    if (enc.time != 1 || enc.height != expect_h || enc.width != expect_w) {
        snprintf(error, error_size,
                 "official COND latent %dx%dx%d, expected 1x%dx%d", enc.time,
                 enc.height, enc.width, expect_h, expect_w);
        h3_vae_latent_f32_free(&enc);
        return 0;
    }
    cond_seq = (size_t)enc.time * (size_t)(enc.height / 2) *
               (size_t)(enc.width / 2);
    row_elems = cond_seq * (size_t)H3_DIT_W4A8_VIDEO_PATCH;
    if (row_elems > row_cap) {
        snprintf(error, error_size, "official COND rows %zu exceed cap %zu",
                 row_elems, row_cap);
        h3_vae_latent_f32_free(&enc);
        return 0;
    }
    if (!h3_vae_patchify_video_96(enc.values, enc.time, enc.height, enc.width,
                                  rows, row_elems)) {
        snprintf(error, error_size, "cond patchify failed");
        h3_vae_latent_f32_free(&enc);
        return 0;
    }
    h3_vae_latent_f32_free(&enc);
    *row_elems_out = row_elems;
    return 1;
}

/* Multi-frame Ref2VA video → H3_SEG_REF_IMAGE rows (causal T_lat). */
static int official_encode_ref_video_rows(const char *enc_path,
                                          const float *rgb_chtw, int frames,
                                          int height, int width, float **rows_out,
                                          size_t *row_elems_out, int *latent_t_out,
                                          int *latent_h_out, int *latent_w_out,
                                          char *error, size_t error_size) {
    h3_vae_latent_f32 enc;
    size_t cond_seq, row_elems;
    float *rows;
    memset(&enc, 0, sizeof(enc));
    if (rows_out) *rows_out = NULL;
    if (row_elems_out) *row_elems_out = 0;
    if (!h3_vae_encode_f32(enc_path, rgb_chtw, frames, height, width, &enc,
                           error, error_size))
        return 0;
    if (enc.time < 1 || enc.height < 2 || enc.width < 2 || (enc.height % 2) ||
        (enc.width % 2)) {
        snprintf(error, error_size, "ref-video latent shape %dx%dx%d invalid",
                 enc.time, enc.height, enc.width);
        h3_vae_latent_f32_free(&enc);
        return 0;
    }
    cond_seq = (size_t)enc.time * (size_t)(enc.height / 2) *
               (size_t)(enc.width / 2);
    row_elems = cond_seq * (size_t)H3_DIT_W4A8_VIDEO_PATCH;
    rows = (float *)malloc(row_elems * sizeof(float));
    if (!rows) {
        snprintf(error, error_size, "OOM ref-video cond rows");
        h3_vae_latent_f32_free(&enc);
        return 0;
    }
    if (!h3_vae_patchify_video_96(enc.values, enc.time, enc.height, enc.width,
                                  rows, row_elems)) {
        snprintf(error, error_size, "ref-video patchify failed");
        free(rows);
        h3_vae_latent_f32_free(&enc);
        return 0;
    }
    if (latent_t_out) *latent_t_out = enc.time;
    if (latent_h_out) *latent_h_out = enc.height;
    if (latent_w_out) *latent_w_out = enc.width;
    h3_vae_latent_f32_free(&enc);
    *rows_out = rows;
    *row_elems_out = row_elems;
    return 1;
}

static float *extract_vision_pair_chtw(const float *pixels, int frames,
                                       int height, int width, int first,
                                       int second) {
    size_t area;
    float *pair;
    int times[2];
    int time, channel;
    if (!pixels || frames < 1 || height < 1 || width < 1 || first < 0 ||
        second < 0 || first >= frames || second >= frames)
        return NULL;
    area = (size_t)height * (size_t)width;
    pair = (float *)malloc(6 * area * sizeof(float));
    if (!pair) return NULL;
    times[0] = first;
    times[1] = second;
    for (time = 0; time < 2; time++)
        for (channel = 0; channel < 3; channel++) {
            size_t source =
                ((size_t)channel * (size_t)frames + (size_t)times[time]) * area;
            size_t destination =
                ((size_t)time * 3 + (size_t)channel) * area;
            memcpy(pair + destination, pixels + source, area * sizeof(float));
        }
    return pair;
}

static float hash_to_unit(uint64_t seed, size_t index) {
    uint64_t h = mix64(seed ^ (uint64_t)(index + 1) * 0x9e3779b97f4a7c15ULL);
    return (float)(int32_t)(h >> 32) * (1.0f / 2147483648.0f);
}

/* Metal-compatible N(0,1) via Box-Muller on hash uniforms (no Accelerate). */
static void fill_normal_f32(uint64_t seed, float *dst, size_t count) {
    size_t i;
    for (i = 0; i + 1 < count; i += 2) {
        float u1 = 0.5f * (hash_to_unit(seed, i) + 1.0f);
        float u2 = 0.5f * (hash_to_unit(seed, i + 1) + 1.0f);
        float radius, angle;
        if (!(u1 > 1e-12f) || !isfinite(u1)) u1 = 1e-12f;
        if (!(u2 >= 0.0f) || !isfinite(u2)) u2 = 0.0f;
        if (u1 > 1.0f) u1 = 1.0f;
        if (u2 > 1.0f) u2 = 1.0f;
        radius = sqrtf(-2.0f * logf(u1));
        angle = 6.283185307179586f * u2;
        dst[i] = radius * cosf(angle);
        dst[i + 1] = radius * sinf(angle);
    }
    if (i < count) dst[i] = hash_to_unit(seed, i);
}

static int ensure_dir(const char *path) {
    struct stat st;
    if (!path || !path[0]) return 0;
    if (stat(path, &st) == 0) return S_ISDIR(st.st_mode);
    return mkdir(path, 0755) == 0;
}

/* Numerical oracle dumps (H3_DUMP_DIR). Used to align TE/velocity vs Comfy. */
static int dump_f32_raw(const char *dir, const char *name, const float *data,
                        size_t count) {
    char path[1024];
    FILE *fp;
    if (!dir || !dir[0] || !name || !data || count == 0) return 0;
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "dump: cannot write %s\n", path);
        return 0;
    }
    if (fwrite(data, sizeof(float), count, fp) != count) {
        fprintf(stderr, "dump: short write %s\n", path);
        fclose(fp);
        return 0;
    }
    fclose(fp);
    printf("dump: %s count=%zu\n", path, count);
    return 1;
}

static int dump_meta_txt(const char *dir, size_t context, int latent_t,
                         int latent_h, int latent_w, int audio_t, int steps,
                         uint64_t seed, float sigma0, float audio_sigma0,
                         size_t video_seq, size_t audio_seq, size_t n_lat) {
    char path[1024];
    FILE *fp;
    if (!dir || !dir[0]) return 0;
    snprintf(path, sizeof(path), "%s/meta.txt", dir);
    fp = fopen(path, "w");
    if (!fp) return 0;
    fprintf(fp,
            "context=%zu\nlatent_t=%d\nlatent_h=%d\nlatent_w=%d\naudio_t=%d\n"
            "steps=%d\nseed=0x%llx\nsigma0=%.8g\naudio_sigma0=%.8g\n"
            "video_seq=%zu\naudio_seq=%zu\nn_lat=%zu\n"
            "te_raw_shape=%zu,%d\nte_refined_shape=%zu,%d\n"
            "latent_shape=%zu,%d\n"
            "note=te_raw=[tokens,5120] te_refined=[tokens,5376] "
            "latent=[video_seq,96] patch-major T,H/2,W/2\n",
            context, latent_t, latent_h, latent_w, audio_t, steps,
            (unsigned long long)seed, (double)sigma0, (double)audio_sigma0,
            video_seq, audio_seq, n_lat, context, H3_DIT_W4A8_TEXT_DIM, context,
            H3_DIT_W4A8_HIDDEN, video_seq, H3_DIT_W4A8_VIDEO_PATCH);
    fclose(fp);
    return 1;
}

/* Mirror h3_serving_schedule_build / H3_VIDEO_SIGMA_SHIFT without h3_host.c. */
#ifndef H3_CUDA_VIDEO_SIGMA_SHIFT
#define H3_CUDA_VIDEO_SIGMA_SHIFT 12.0f
#endif
#ifndef H3_CUDA_AUDIO_SIGMA_SHIFT
#define H3_CUDA_AUDIO_SIGMA_SHIFT 3.0f
#endif

static float shifted_sigma_from_base(float base, float shift) {
    if (!(base > 0.0f)) return 0.0f;
    return shift * base / (1.0f + (shift - 1.0f) * base);
}

static int build_serving_sigmas(int steps, float shift, float *sigmas) {
    int i;
    if (!sigmas || steps < 1) return 0;
    for (i = 0; i <= steps; i++) {
        float base = 1.0f - (float)i / (float)steps;
        sigmas[i] = shifted_sigma_from_base(base, shift);
    }
    sigmas[steps] = 0.0f;
    return 1;
}

/* F2VA t=0 img2img (CUDA lite; not Metal condition-rows).
 *
 * Mix on the *unshifted* serving grid u=1-i/steps. Shifted σ (shift=12)
 * stays ~1 until the last jump, so x=(1-σ)*img+σ*noise would snap identity
 * only at the end and wash the jpeg for most of Euler.
 *
 * img_w: 0.25 at i=0 → 1.0 at i=steps. Not 100% lock (standing pose) and
 * not a one-shot 0.999/0.001 inject. After the last step t=0 is x_img, so
 * decoded f0 is the VAE round-trip of the first frame. */
static float apply_t0_img2img(float *latent, const float *x_img, size_t n,
                              int step_after, int steps) {
    float u, img_w;
    size_t i;
    if (!latent || !x_img || n == 0) return 0.0f;
    if (steps < 1) steps = 1;
    if (step_after < 0) step_after = 0;
    if (step_after > steps) step_after = steps;
    u = 1.0f - (float)step_after / (float)steps;
    img_w = 1.0f - 0.75f * u;
    if (step_after >= steps) img_w = 1.0f;
    if (img_w < 0.0f) img_w = 0.0f;
    if (img_w > 1.0f) img_w = 1.0f;
    for (i = 0; i < n; i++)
        latent[i] = img_w * x_img[i] + (1.0f - img_w) * latent[i];
    return img_w;
}

/* Official H3 temporal: align frames then video_latent_t (not frames/4). */
static int align_frame_count(int requested) {
    int value = requested < 5 ? 5 : requested;
    int remainder = (value - 5) % 17;
    if (remainder != 0) value += 17 - remainder;
    return value;
}

static int video_latent_t_from_frames(int frame_count) {
    if (frame_count <= 5) return 2;
    return ((frame_count - 5) / 17) * 5 + 2;
}

static int write_wav_f32_stereo(const char *path, const float *pcm_ch_major,
                                int channels, int samples, int sample_rate,
                                char *error, size_t error_size) {
    /* pcm_ch_major: [channels, samples] → interleaved int16 WAV. */
    FILE *fp = NULL;
    int32_t data_bytes, riff_size;
    int16_t *inter = NULL;
    int s, c;
    if (!path || !pcm_ch_major || channels != 2 || samples < 1) {
        snprintf(error, error_size, "invalid wav args");
        return 0;
    }
    inter = (int16_t *)malloc((size_t)channels * (size_t)samples * sizeof(int16_t));
    if (!inter) {
        snprintf(error, error_size, "OOM wav");
        return 0;
    }
    for (s = 0; s < samples; s++) {
        for (c = 0; c < channels; c++) {
            float v = pcm_ch_major[(size_t)c * (size_t)samples + (size_t)s];
            if (v > 1.0f) v = 1.0f;
            if (v < -1.0f) v = -1.0f;
            inter[(size_t)s * (size_t)channels + (size_t)c] =
                (int16_t)lrintf(v * 32767.0f);
        }
    }
    data_bytes = (int32_t)((size_t)channels * (size_t)samples * 2u);
    riff_size = 36 + data_bytes;
    fp = fopen(path, "wb");
    if (!fp) {
        snprintf(error, error_size, "cannot open %s", path);
        free(inter);
        return 0;
    }
    fwrite("RIFF", 1, 4, fp);
    fwrite(&riff_size, 4, 1, fp);
    fwrite("WAVEfmt ", 1, 8, fp);
    {
        int32_t chunk = 16, byte_rate;
        int16_t audio_fmt = 1, ch = (int16_t)channels, bps = 16;
        int16_t block_align = (int16_t)(channels * 2);
        int32_t sr = sample_rate;
        byte_rate = sample_rate * channels * 2;
        fwrite(&chunk, 4, 1, fp);
        fwrite(&audio_fmt, 2, 1, fp);
        fwrite(&ch, 2, 1, fp);
        fwrite(&sr, 4, 1, fp);
        fwrite(&byte_rate, 4, 1, fp);
        fwrite(&block_align, 2, 1, fp);
        fwrite(&bps, 2, 1, fp);
    }
    fwrite("data", 1, 4, fp);
    fwrite(&data_bytes, 4, 1, fp);
    fwrite(inter, 2, (size_t)channels * (size_t)samples, fp);
    fclose(fp);
    free(inter);
    return 1;
}

static int gen_uly_bind(void *cfg, int rank, const void *d_in, void *d_out,
                        char *err, size_t err_size) {
    return h3_ulysses_comm_bind((h3_ulysses_cfg *)cfg, rank, d_in, d_out, err,
                                err_size);
}

static int gen_uly_set_compute(void *cfg, int rank, void *stream, char *err,
                               size_t err_size) {
    return h3_ulysses_comm_set_compute((h3_ulysses_cfg *)cfg, rank, stream, err,
                                       err_size);
}

static int gen_uly_a2a_qkv(void *cfg, size_t seq_local, size_t heads,
                           size_t head_dim, char *err, size_t err_size) {
    return h3_ulysses_a2a_qkv((h3_ulysses_cfg *)cfg, NULL, NULL, seq_local,
                              heads, head_dim, 4, err, err_size);
}

static int gen_uly_a2a_attn(void *cfg, size_t seq_local, size_t heads,
                            size_t head_dim, char *err, size_t err_size) {
    return h3_ulysses_a2a_attn_out((h3_ulysses_cfg *)cfg, NULL, NULL, seq_local,
                                   heads, head_dim, 4, err, err_size);
}

static void gen_uly_free_ranks(h3_dit_w4a8 *dit0, h3_dit_w4a8 **uly_dit,
                               h3_cuda **uly_cuda) {
    int ui;
    char ignore[64];
    if (dit0)
        (void)h3_dit_w4a8_set_ulysses(dit0, NULL, NULL, 1, NULL, ignore,
                                      sizeof(ignore));
    for (ui = 1; ui < 8; ui++) {
        if (uly_dit && uly_dit[ui]) {
            h3_dit_w4a8_free(uly_dit[ui]);
            uly_dit[ui] = NULL;
        }
        if (uly_cuda && uly_cuda[ui]) {
            h3_cuda_free(uly_cuda[ui]);
            uly_cuda[ui] = NULL;
        }
    }
}

int main(int argc, char **argv) {
    const char *dit_path = NULL, *vae_path = NULL, *te_path = NULL;
    const char *audio_vae_path = NULL;
    char audio_vae_resolved[1024];
    char mux_resolved[1024];
    int audio_vae_explicit = 0;
    const char *out_dir = "out_cuda_t2v";
    const char *sdpa = "cudnn";
    const char *preset_name = NULL;
    const char *lora_path = NULL;
    const char *vae_decode = "official";
    const char *tae_path = NULL;
    const char *te_cache_dir = NULL;
    float lora_scale = 1.0f;
    int steps_explicit = 0, sdpa_explicit = 0;
    int ulysses = 1;
    h3_generate_preset_out preset_out;
    h3_lora_set lora_set;
    int lora_loaded = 0;
    h3_ulysses_cfg ulysses_cfg;
    const char *token_ids_file = NULL;
    const char *token_ids_list = NULL;
    const char *token_canary = NULL;
    const char *init_latent_path = NULL;
    const char *first_frame_path = NULL;
    const char *last_frame_path = NULL;
    const char *ref_image_paths[H3_CUDA_MAX_REF_IMAGES];
    int n_ref_images = 0;
    const char *ref_audio_paths[H3_CUDA_MAX_REF_AUDIOS];
    int n_ref_audios = 0;
    const char *ref_video_path = NULL;
    int ref_silent_video = 0;
    int last_keyframe = 0;
    int official_kf = 0; /* 1=F2VA, 2=L2VA, 3=FL2VA */
    const char *cond_rows_path = NULL;
    const char *vae_encode_path = NULL;
    const char *vision_path = NULL;
    const char *mux_path = NULL;
    unsigned dit_layers = 50;
    unsigned vae_blocks = 36;
    unsigned te_layers = 50;
    int vae_repeats = 1;
    int steps = 2;
    int reuse = 0; /* 0=default full-eval baseline; --reuse N opt-in */
    float magcache_thresh = 0.0f; /* 0=off; lite Mag; ≠ independent Tea */
    int magcache_calibrated = 0; /* Tea-style poly under mag flag (legacy) */
    h3_step_cache step_cache;    /* independent Tea / FB / Easy stub */
    int token_reduction = 0; /* Metal --token-reduction */
    int lock_first_frame = 0; /* debug G3-lite: keep t=0 latent slice fixed */
    int init_first_frame = 0; /* debug G3-lite: t=0 img2img from --init-latent */
    int width = 864, height = 480, seconds = 5;
    int frames_override = -1; /* --frames N overrides seconds*24 */
    int audio_t = -1; /* -1 → derive from seconds when T2AV */
    /* Text context budget (NOT a silent truncate length).
     * Evidence:
     * - MiniMax API prompt limit ≤ 7000 characters (~2k Qwen tokens EN)
     * - Comfy Qwen3VLSDTokenizer: max_length=99999999, pad_to_max_length=False
     * - Metal T2VA: exact token count (no fixed pad)
     * - vLLM-Omni / SGLang: variable text_len in packed sequence
     * Default 256 is a VRAM-friendly budget for 24GB; ids longer than budget
     * auto-raise up to H3_CUDA_TEXT_CONTEXT_HARD_MAX (never silent truncate). */
    size_t context = H3_CUDA_TEXT_CONTEXT_DEFAULT;
    uint64_t seed = 0x483343554441ULL;
    /* T2VA theme default cfg=5 (underwater turtle / orange cat). Official
     * --first-frame F2VA / --last-frame L2VA / both FL2VA / --ref-image
     * Ref2VA override to 1 unless --cfg is passed (distilled + image
     * condition; cfg=5 on 8-step INT8 blows later frames into hotspots). */
    float cfg_scale = 5.0f;
    int cfg_explicit = 0;
    int device = 0, rc = 1, iarg, step, skip_te = 0, skip_vae = 0;
    const char *dump_dir = getenv("H3_DUMP_DIR");
    int dump_only = getenv("H3_DUMP_ONLY") && getenv("H3_DUMP_ONLY")[0] == '1';
    int skip_audio = 0, skip_dit = 0, probe_vae_load = 0;
    /* Comfy fl2va always packs [text|audio|video] even for “video-only”
     * prompts (empty audio latent). Missing audio tokens → dialogue prior.
     * pack_audio: DiT forward includes audio rows; decode_audio: default
     * ON (auto --audio-vae / H3_AUDIO_VAE). --skip-audio is video-only. */
    int pack_audio = 0, decode_audio = 0;
    float *frame0_lock = NULL;
    float *frame0_img = NULL; /* debug G3-lite: clean encoded t=0 */
    size_t frame0_elems = 0;
    float *ff_rgb = NULL;
    float *lf_rgb = NULL;
    float *ref_rgb[H3_CUDA_MAX_REF_IMAGES];
    int ref_pw[H3_CUDA_MAX_REF_IMAGES], ref_ph[H3_CUDA_MAX_REF_IMAGES];
    int ref_lh[H3_CUDA_MAX_REF_IMAGES], ref_lw[H3_CUDA_MAX_REF_IMAGES];
    float *ref_video_rgb = NULL;
    int ref_video_frames = 0;
    int ref_vh = 0, ref_vw = 0, ref_vlh = 0, ref_vlw = 0, ref_vlt = 0;
    size_t target_rope_t0 = 0;
    float *cond_rows = NULL, *cpt = NULL, *cph = NULL, *cpw = NULL;
    size_t cond_seq = 0;
    float *audio_cond_rows = NULL;
    float *acpt = NULL, *acph = NULL, *acpw = NULL;
    size_t audio_cond_seq = 0;
    int ref_audio_t_sum = 0;
    float *vision_merged = NULL;
    float *vision_ds[3];
    h3_cuda_vision_span vision_spans[H3_CUDA_MAX_REF_VIDEO_BLOCKS];
    int vision_span_count = 0;
    int have_vision_span = 0;
    int no_vision_inject = 0; /* diagnostic A/B: COND without Qwen vision TE */
    /* CUDA INT8 official F2VA default: Picture+vision stay TEXT AdaLN.
     * Metal tags those rows VIDEO@σ; that 8-step INT8 path blows f7 into a
     * hotspot. --dit-text-tags restores Metal-like tagging for debug only. */
    int dit_text_tags = 0;
    uint8_t *text_mod_tags = NULL;
    uint8_t *mux_rgb = NULL;
    int latent_t, latent_h, latent_w;
    int frames_aligned = 0;
    int frames_req = 0;
    int have_window = 0;
    int dist_attn = 0;
    int reuse_explicit = 0;
    int effective_reuse = 1;
    char reuse_label[32];
    size_t video_seq, n_lat, audio_seq = 0, n_audio = 0, i;
    char error[512];
    char pathbuf[768];
    uint8_t *reuse_sel = NULL;
    float *vel_prev = NULL, *vel_last = NULL;
    float *avel_prev = NULL, *avel_last = NULL;
    int last_eval = -1, prev_eval = -1;
    int dit_evals = 0, dit_skips = 0;
    float mag_accum = 0.0f;
    float mag_last_delta = 0.0f;
    int mag_have_probe = 0;
    int mag_consec = 0;
    int mag_max_consec = 2; /* H3 Safe: at most 2 adjacent skips */
    int mag_fresh = 0; /* 1 after a small-delta eval; 0 after force_eval */
    char step_cache_err[256];

    h3_w4a8_store *dit_store = NULL;
    h3_dit_int8_store *dit_int8 = NULL;
    int dit_is_int8 = 0;
    h3_cuda *cuda = NULL;
    h3_dit_w4a8 *dit = NULL;
    h3_cuda *uly_cuda[8];
    h3_dit_w4a8 *uly_dit[8];
    h3_dit_w4a8_core core;
    h3_w4a8_cache_config cfg;
    h3_nvfp4_store te_store;
    h3_nvfp4_encode_stats te_st;
    h3_vae_int8_store vae_store;
    h3_vae_int8_frames frames;
    h3_vae_int8_decode_stats vst;
    h3_audio_vae_cuda_waveform awave;
    h3_audio_vae_cuda_stats ast;

    float *latent = NULL, *velocity = NULL, *sigmas = NULL, *audio_sigmas = NULL;
    float *audio_c2t = NULL, *audio_rows = NULL, *audio_vel = NULL;
    float *text_emb = NULL, *text_h = NULL, *text_h_uncond = NULL;
    float *velocity_uncond = NULL, *audio_vel_uncond = NULL;
    float *latent_chw = NULL;
    int32_t *token_ids = NULL;
    float *pt = NULL, *ph = NULL, *pw = NULL;
    float *apt = NULL, *aph = NULL, *apw = NULL;
    double t_te = 0, t_dit = 0, t_vae = 0, t_audio = 0, t_ppm = 0, t_mux = 0;
    double t0, t_all0;
    void *prefetch_done = NULL;
    int overlap = 1;
    pthread_t sdpa_th = 0;
    pthread_t vae_th = 0;
    pthread_t audio_th = 0;
    pthread_t host_stage_th = 0;
    int sdpa_th_ok = 0;
    int vae_th_ok = 0;
    int audio_th_ok = 0;
    int host_stage_th_ok = 0;
    sdpa_prebuild_job sdpa_job;
    audio_decode_job audio_job;
    dit_host_stage_job host_stage_job;
    host_stage_kick host_stage_kick_job;
    double host_stage_t0 = 0.0;

    memset(&core, 0, sizeof(core));
    memset(&te_store, 0, sizeof(te_store));
    memset(&vae_store, 0, sizeof(vae_store));
    memset(&frames, 0, sizeof(frames));
    memset(&te_st, 0, sizeof(te_st));
    memset(&vst, 0, sizeof(vst));
    memset(&awave, 0, sizeof(awave));
    memset(&ast, 0, sizeof(ast));
    memset(&sdpa_job, 0, sizeof(sdpa_job));
    memset(&audio_job, 0, sizeof(audio_job));
    memset(&host_stage_job, 0, sizeof(host_stage_job));
    memset(&host_stage_kick_job, 0, sizeof(host_stage_kick_job));
    memset(&preset_out, 0, sizeof(preset_out));
    memset(&lora_set, 0, sizeof(lora_set));
    memset(&ulysses_cfg, 0, sizeof(ulysses_cfg));
    memset(uly_cuda, 0, sizeof(uly_cuda));
    memset(uly_dit, 0, sizeof(uly_dit));
    ulysses_cfg.tp_size = 1;
    ulysses_cfg.ulysses = 1;
    ulysses_cfg.ring = 1;
    memset(vision_spans, 0, sizeof(vision_spans));
    memset(ref_rgb, 0, sizeof(ref_rgb));
    memset(ref_image_paths, 0, sizeof(ref_image_paths));
    memset(ref_audio_paths, 0, sizeof(ref_audio_paths));
    vision_ds[0] = vision_ds[1] = vision_ds[2] = NULL;

    h3_step_cache_init(&step_cache);
    step_cache_err[0] = '\0';
    for (iarg = 1; iarg < argc; iarg++) {
        if (!strcmp(argv[iarg], "--dit") && iarg + 1 < argc)
            dit_path = argv[++iarg];
        else if (!strcmp(argv[iarg], "--vae") && iarg + 1 < argc)
            vae_path = argv[++iarg];
        else if (!strcmp(argv[iarg], "--te") && iarg + 1 < argc)
            te_path = argv[++iarg];
        else if (!strcmp(argv[iarg], "--audio-vae") && iarg + 1 < argc) {
            audio_vae_path = argv[++iarg];
            audio_vae_explicit = 1;
        }
        else if (!strcmp(argv[iarg], "--out") && iarg + 1 < argc)
            out_dir = argv[++iarg];
        else if (!strcmp(argv[iarg], "--layers") && iarg + 1 < argc)
            dit_layers = (unsigned)strtoul(argv[++iarg], NULL, 10);
        else if (!strcmp(argv[iarg], "--te-layers") && iarg + 1 < argc)
            te_layers = (unsigned)strtoul(argv[++iarg], NULL, 10);
        else if (!strcmp(argv[iarg], "--vae-blocks") && iarg + 1 < argc)
            vae_blocks = (unsigned)strtoul(argv[++iarg], NULL, 10);
        else if (!strcmp(argv[iarg], "--vae-repeats") && iarg + 1 < argc) {
            vae_repeats = (int)strtol(argv[++iarg], NULL, 10);
            if (vae_repeats < 1) vae_repeats = 1;
        }
        else if (!strcmp(argv[iarg], "--steps") && iarg + 1 < argc) {
            steps = (int)strtol(argv[++iarg], NULL, 10);
            steps_explicit = 1;
        }
        else if (!strcmp(argv[iarg], "--reuse") && iarg + 1 < argc) {
            reuse = (int)strtol(argv[++iarg], NULL, 10);
            if (reuse < 1) reuse = 1;
            if (reuse > 32) reuse = 32;
            reuse_explicit = 1;
        } else if (!strcmp(argv[iarg], "--magcache-thresh") &&
                   iarg + 1 < argc) {
            magcache_thresh = (float)strtod(argv[++iarg], NULL);
            if (!(magcache_thresh > 0.0f) || !isfinite(magcache_thresh))
                magcache_thresh = 0.0f;
        } else if (!strcmp(argv[iarg], "--magcache-calibrated")) {
            magcache_calibrated = 1;
        } else if (!strcmp(argv[iarg], "--teacache-thresh") &&
                   iarg + 1 < argc) {
            if (step_cache.kind != H3_STEP_CACHE_OFF &&
                step_cache.kind != H3_STEP_CACHE_TEA) {
                fprintf(stderr,
                        "h3: --teacache-thresh conflicts with other "
                        "step-cache kind (%s)\n",
                        h3_step_cache_kind_name(step_cache.kind));
                return 1;
            }
            step_cache.kind = H3_STEP_CACHE_TEA;
            step_cache.thresh = (float)strtod(argv[++iarg], NULL);
            if (!(step_cache.thresh > 0.0f) || !isfinite(step_cache.thresh)) {
                step_cache.thresh = 0.0f;
                step_cache.kind = H3_STEP_CACHE_OFF;
            }
        } else if (!strcmp(argv[iarg], "--fbcache-thresh") &&
                   iarg + 1 < argc) {
            if (step_cache.kind != H3_STEP_CACHE_OFF &&
                step_cache.kind != H3_STEP_CACHE_FB) {
                fprintf(stderr,
                        "h3: --fbcache-thresh conflicts with other "
                        "step-cache kind (%s)\n",
                        h3_step_cache_kind_name(step_cache.kind));
                return 1;
            }
            step_cache.kind = H3_STEP_CACHE_FB;
            step_cache.thresh = (float)strtod(argv[++iarg], NULL);
            if (!(step_cache.thresh > 0.0f) || !isfinite(step_cache.thresh)) {
                step_cache.thresh = 0.0f;
                step_cache.kind = H3_STEP_CACHE_OFF;
            }
        } else if (!strcmp(argv[iarg], "--easycache-thresh") &&
                   iarg + 1 < argc) {
            /* Stub: always hard-fail at validate (NOT IMPLEMENTED). */
            (void)strtod(argv[++iarg], NULL);
            step_cache.kind = H3_STEP_CACHE_EASY;
            step_cache.thresh = 1.0f;
        } else if (!strcmp(argv[iarg], "--token-reduction")) {
            token_reduction = 1;
        } else if (!strcmp(argv[iarg], "--token-ids-file") && iarg + 1 < argc) {
            token_ids_file = argv[++iarg];
        } else if (!strcmp(argv[iarg], "--token-ids") && iarg + 1 < argc) {
            token_ids_list = argv[++iarg];
        } else if (!strcmp(argv[iarg], "--token-canary") && iarg + 1 < argc) {
            token_canary = argv[++iarg];
        } else if (!strcmp(argv[iarg], "--init-latent") && iarg + 1 < argc) {
            init_latent_path = argv[++iarg];
        } else if (!strcmp(argv[iarg], "--first-frame") && iarg + 1 < argc) {
            first_frame_path = argv[++iarg];
        } else if (!strcmp(argv[iarg], "--last-frame") && iarg + 1 < argc) {
            last_frame_path = argv[++iarg];
        } else if (!strcmp(argv[iarg], "--ref-image") && iarg + 1 < argc) {
            if (n_ref_images >= H3_CUDA_MAX_REF_IMAGES) {
                fprintf(stderr,
                        "h3: --ref-image supports at most %d images in this "
                        "CUDA bypass (Picture 1/2 prefixes; Metal allows 9)\n",
                        H3_CUDA_MAX_REF_IMAGES);
                return 1;
            }
            ref_image_paths[n_ref_images++] = argv[++iarg];
        } else if (!strcmp(argv[iarg], "--ref-audio") && iarg + 1 < argc) {
            if (n_ref_audios >= H3_CUDA_MAX_REF_AUDIOS) {
                fprintf(stderr,
                        "h3: --ref-audio supports at most %d clips in this "
                        "CUDA bypass (Metal allows 3, total ≤15s)\n",
                        H3_CUDA_MAX_REF_AUDIOS);
                return 1;
            }
            ref_audio_paths[n_ref_audios++] = argv[++iarg];
        } else if (!strcmp(argv[iarg], "--ref-video") && iarg + 1 < argc) {
            if (ref_video_path) {
                fprintf(stderr, "h3: only one --ref-video in this CUDA bypass\n");
                return 1;
            }
            ref_video_path = argv[++iarg];
            ref_silent_video = 0;
        } else if (!strcmp(argv[iarg], "--ref-silent-video") &&
                   iarg + 1 < argc) {
            if (ref_video_path) {
                fprintf(stderr,
                        "h3: only one --ref-video/--ref-silent-video\n");
                return 1;
            }
            ref_video_path = argv[++iarg];
            ref_silent_video = 1;
        } else if (!strcmp(argv[iarg], "--ref-merged")) {
            fprintf(stderr,
                    "%s is Ref2VA stage H2/H3 (not this slice); mutually "
                    "exclusive with --first-frame / --last-frame keyframes\n",
                    argv[iarg]);
            return 1;
        } else if (!strcmp(argv[iarg], "--cond-rows") && iarg + 1 < argc) {
            cond_rows_path = argv[++iarg];
        } else if (!strcmp(argv[iarg], "--vae-encode") && iarg + 1 < argc) {
            vae_encode_path = argv[++iarg];
        } else if (!strcmp(argv[iarg], "--vision") && iarg + 1 < argc) {
            vision_path = argv[++iarg];
        } else if (!strcmp(argv[iarg], "--no-vision-inject")) {
            no_vision_inject = 1;
        } else if (!strcmp(argv[iarg], "--dit-text-tags")) {
            dit_text_tags = 1;
        } else if (!strcmp(argv[iarg], "--no-dit-text-tags")) {
            dit_text_tags = 0; /* explicit CUDA INT8 default; alias */
        } else if (!strcmp(argv[iarg], "--init-first-frame")) {
            init_first_frame = 1;
        } else if (!strcmp(argv[iarg], "--lock-first-frame")) {
            lock_first_frame = 1;
        } else if (!strcmp(argv[iarg], "--mux") && iarg + 1 < argc) {
            mux_path = argv[++iarg];
        }         else if (!strcmp(argv[iarg], "--context") && iarg + 1 < argc)
            context = (size_t)strtoul(argv[++iarg], NULL, 10);
        else if (!strcmp(argv[iarg], "--seed") && iarg + 1 < argc)
            seed = strtoull(argv[++iarg], NULL, 0);
        else if (!strcmp(argv[iarg], "--cfg") && iarg + 1 < argc) {
            cfg_scale = (float)strtod(argv[++iarg], NULL);
            cfg_explicit = 1;
            if (!(cfg_scale > 0.0f) || !isfinite(cfg_scale)) {
                cfg_explicit = 0;
                cfg_scale = 5.0f;
            }
        } else if (!strcmp(argv[iarg], "--width") && iarg + 1 < argc)
            width = (int)strtol(argv[++iarg], NULL, 10);
        else if (!strcmp(argv[iarg], "--height") && iarg + 1 < argc)
            height = (int)strtol(argv[++iarg], NULL, 10);
        else if (!strcmp(argv[iarg], "--seconds") && iarg + 1 < argc)
            seconds = (int)strtol(argv[++iarg], NULL, 10);
        else if (!strcmp(argv[iarg], "--frames") && iarg + 1 < argc)
            frames_override = (int)strtol(argv[++iarg], NULL, 10);
        else if (!strcmp(argv[iarg], "--audio-t") && iarg + 1 < argc)
            audio_t = (int)strtol(argv[++iarg], NULL, 10);
        else if (!strcmp(argv[iarg], "--sdpa-backend") && iarg + 1 < argc) {
            sdpa = argv[++iarg];
            sdpa_explicit = 1;
        } else if (!strcmp(argv[iarg], "--preset") && iarg + 1 < argc)
            preset_name = argv[++iarg];
        else if (!strcmp(argv[iarg], "--lora") && iarg + 1 < argc)
            lora_path = argv[++iarg];
        else if (!strcmp(argv[iarg], "--lora-scale") && iarg + 1 < argc)
            lora_scale = (float)strtod(argv[++iarg], NULL);
        else if (!strcmp(argv[iarg], "--ulysses") && iarg + 1 < argc)
            ulysses = (int)strtol(argv[++iarg], NULL, 10);
        else if (!strcmp(argv[iarg], "--vae-decode") && iarg + 1 < argc)
            vae_decode = argv[++iarg];
        else if (!strcmp(argv[iarg], "--tae") && iarg + 1 < argc)
            tae_path = argv[++iarg];
        else if (!strcmp(argv[iarg], "--te-cache") && iarg + 1 < argc)
            te_cache_dir = argv[++iarg];
        else if (!strcmp(argv[iarg], "--window") && iarg + 1 < argc) {
            char buf[32];
            snprintf(buf, sizeof(buf), "%s", argv[++iarg]);
            setenv("H3_DIT_W4A8_ATTN_WINDOW", buf, 1);
            have_window = 1;
        } else if (!strcmp(argv[iarg], "--dist-attn")) {
            setenv("H3_DIT_DIST_ATTN", "1", 1);
            dist_attn = 1;
        } else if (!strcmp(argv[iarg], "--dist-scale") && iarg + 1 < argc) {
            setenv("H3_DIT_DIST_SCALE", argv[++iarg], 1);
            setenv("H3_DIT_DIST_ATTN", "1", 1);
            dist_attn = 1;
        } else if (!strcmp(argv[iarg], "--dist-temporal") && iarg + 1 < argc) {
            setenv("H3_DIT_DIST_TEMPORAL", argv[++iarg], 1);
            setenv("H3_DIT_DIST_ATTN", "1", 1);
            dist_attn = 1;
        } else if (!strcmp(argv[iarg], "--no-overlap"))
            overlap = 0;
        else if (!strcmp(argv[iarg], "--skip-te"))
            skip_te = 1;
        else if (!strcmp(argv[iarg], "--skip-dit"))
            skip_dit = 1;
        else if (!strcmp(argv[iarg], "--skip-vae"))
            skip_vae = 1;
        else if (!strcmp(argv[iarg], "--skip-audio"))
            skip_audio = 1;
        else if (!strcmp(argv[iarg], "--probe-vae-load"))
            probe_vae_load = 1;
        else if (!strcmp(argv[iarg], "--device") && iarg + 1 < argc)
            device = (int)strtol(argv[++iarg], NULL, 10);
        else if (!strcmp(argv[iarg], "-h") || !strcmp(argv[iarg], "--help")) {
            fprintf(stderr,
                    "h3_cuda_generate — CUDA bypass T2V/T2AV CLI (NOT Metal h3)\n"
                    "usage: %s --dit DIT --vae VAE [--te TE] [--audio-vae A] "
                    "--out DIR\n"
                    "  T2AV: audio VAE is ON by default (Comfy FP32 sibling of "
                    "--vae, or H3_AUDIO_VAE). --skip-audio for video-only. "
                    "Optional --audio-t.\n"
                    "  --reuse N          Metal-compatible Euler reuse "
                    "(default off; pass 1 for explicit full-eval schedule)\n"
                    "  --token-reduction  Metal-compatible horizontal video "
                    "token pooling in middle blocks\n"
                    "  --magcache-thresh T  MagCache lite skip "
                    "(0=off default; H3 Safe A/B try 0.08, max 2 consec). "
                    "Not independent TeaCache; calibrated flag ≈ Tea-style "
                    "poly under the mag CLI.\n"
                    "  --magcache-calibrated  Use poly(rel_l1) under mag "
                    "(legacy; prefer --teacache-thresh for independent Tea)\n"
                    "                      magcache ignored when --reuse > 1\n"
                    "  --teacache-thresh T  Independent TeaCache "
                    "(poly rel_L1 + AV veto; 0=off default). "
                    "Hard-mutex with mag/fb/reuse>1. "
                    "H3_TEACACHE_MAX_CONSEC (default 2). "
                    "8-step: do NOT copy 50-step 0.10/0.17 as lossless.\n"
                    "  --fbcache-thresh T   Conservative FBCache residual gate "
                    "on velocity |ratio-1| (+AV veto; 0=off). "
                    "Hard-mutex with tea/mag/reuse>1. "
                    "H3_FBCACHE_MAX_CONSEC / H3_FBCACHE_WARMUP.\n"
                    "  --easycache-thresh T  NOT IMPLEMENTED (hard error; "
                    "refuses half-wired EasyCache on AV path)\n"
                    "  --context N        TE length *budget* (default %u). "
                    "With --token-ids*: uses exact id count (auto-raises up to "
                    "%u; NEVER silent-truncates). Smoke-only smaller N is OK; "
                    "quality/official prompts must not use 16.\n"
                    "  --token-ids-file F  Load Qwen token ids (exact length; "
                    "no pad/trunc)\n"
                    "  --token-ids LIST   Comma/space ids (exact length)\n"
                    "  --token-canary N   Built-in fox|cinematic|im_start\n"
                    "  --te-layers N      NVFP4 TE depth (default 50)\n"
                    "  --vae-repeats N    Repeat VAE decode in-process "
                    "(default 1; useful for load-cache A/B)\n"
                    "  --probe-vae-load   Only measure VAE weights_load path\n"
                    "  --seconds N        Duration at 24 fps (default 5). "
                    "Frame count = N*24, aligned up to 5+17k. "
                    "Do not pass --frames if you want a real N-second mp4.\n"
                    "  --frames N         Override frame count (NOT duration). "
                    "Mux is always 24 fps, so --frames 17 → ~0.92s even with "
                    "--seconds 5. For 5s/10s product clips omit this.\n"
                    "  --seed N           Noise seed (default 0x483343554441)\n"
                    "  --preset gold|turbo4\n"
                    "      gold（可省略）= 今日金线路径，不改 steps/backend/cache。\n"
                    "      turbo4 = 非金线极速档：steps=4 + sage3 + 要求 --lora；cache 关。\n"
                    "      未蒸馏 --steps 4 而无 LoRA 不是产品。缺省不传 preset。\n"
                    "      --preset fast4 已更名为 turbo4。\n"
                    "  --lora FILE.safetensors   turbo4 必填；金线禁止依赖此文件\n"
                    "  --lora-scale S            默认 1.0\n"
                    "  --ulysses N        Sequence-parallel Ulysses (default 1).\n"
                    "                     Experimental. Legal {1,2,4,8}; U=7 illegal.\n"
                    "                     N=1 does not init NCCL. Does not change sdpa.\n"
                    "                     N>1 drives DiT A2A after QKV+LoRA; SDPA uses\n"
                    "                     S_real (pad rows excluded). Gold default stays 1.\n"
                    "  --te-cache DIR     Opt-in TE prompt embedding disk cache.\n"
                    "                     Or H3_TE_PROMPT_CACHE. Default off (金线不变).\n"
                    "                     Text-only; vision/multimodal never cached.\n"
                    "  --vae-decode official|tae  Default official. tae 仅 turbo4\n"
                    "                     （TAE 不是金线成片）；需 --tae 权，禁止静默 official。\n"
                    "  --tae FILE         TAE weights (turbo + --vae-decode tae)\n"
                    "  --sdpa-backend B   DiT SDPA (h3_sdpa registry):\n"
                    "                     Tier-0 quality: cudnn (default) | auto→cudnn\n"
                    "                     Tier-1 accel:   sage3 | sol_cute  (first-class)\n"
                    "                     Forbidden product: sage2 (ref), sol (nested-loop)\n"
                    "                     Legacy debug: flash | tc\n"
                    "                     Env H3_CUDA_SDPA_BACKEND. Never auto-picks lossy.\n"
                    "                     sage3: H3_SAGE3_LIB + H3_SAGE3_ROOT (5090/FP4).\n"
                    "                     sol_cute: H3_SOL_CUTE_LIB + H3_SOL_ATTN_ROOT "
                    "(4090/sm_89).\n"
                    "  --cfg S            CFG scale. Two defaults: T2VA theme "
                    "path is 5 (underwater turtle / orange cat gates need it). "
                    "Official --first-frame F2VA / --last-frame L2VA / both "
                    "FL2VA / --ref-image Ref2VA is 1 "
                    "(CFG-distilled + image condition; cfg=5 on 8-step INT8 "
                    "later frames → hotspot). "
                    "Pass 1 to disable. S>1 uses TE(pad)×refiner uncond. "
                    "Official image conditions keep Picture+vision "
                    "in uncond and "
                    "pads only the prompt ids (Metal vision is not CFG-amplified).\n"
                    "  --window W         SDPA token window (default 0=global; "
                    "W>0 is ±W local — severs text↔video on T2AV).\n"
                    "  --dist-attn        YCNodes Gaussian RF on video-video "
                    "only (opt-in; default off). H3_DIT_DIST_SCALE=15 "
                    "H3_DIT_DIST_TEMPORAL=3. Does not touch text/audio. "
                    "Not a speedup; cafe temporal-stability knife.\n"
                    "  --dist-scale S     Enable dist-attn with receptive-field "
                    "scale S (smaller = more spatial-local).\n"
                    "  --dist-temporal W  Enable dist-attn; time-axis weight W "
                    "(YCNodes temporal_weight; default 3, node default 2, "
                    "node max 10). Larger W penalizes far frames more. "
                    "Video span only; text/audio unpenalized.\n"
                    "  --first-frame IMG  Official F2VA/FL2VA condition: jpeg → "
                    "VAE encode → DiT COND rows + Qwen vision TE. STRETCH to "
                    "canvas. keyframes=[0]. Default --cfg 1 (not the T2VA "
                    "cfg=5). Not G3-lite. "
                    "Do not mix with --init-latent / --init-first-frame / "
                    "--lock-first-frame / --ref-*. Combine with --last-frame "
                    "for official FL2VA (two COND segments). With --skip-dit: "
                    "VAE roundtrip (encode canvas → decode, no DiT) — f0 PSNR "
                    "ceiling.\n"
                    "  --ref-image IMG    Official Ref2VA image reference "
                    "(CUDA bypass). Repeating the flag appends ordered "
                    "<Picture N>. Uses a DISTINCT Ref2VA DiT (--dit path must "
                    "contain ref2va; FL2VA filenames are refused — no silent "
                    "fallback to FL2VA + init-latent). Canvas is Metal "
                    "h3_reference_image_canvas: down-only, aspect-preserving "
                    "(not STRETCH/COVER to the output). Pack "
                    "[text|H3_SEG_REF_IMAGE|audio|video]; RoPE cursor = "
                    "text_seq then +1 per image (not keyframe 0/last). "
                    "AdaLN still visual_condition. Mutually exclusive with "
                    "--first-frame / --last-frame. Default --cfg 1. Not lite.\n"
                    "  --ref-audio WAV    Official Ref2VA audio (must also pass "
                    "--ref-image or --ref-video). Packs H3_SEG_REF_AUDIO; "
                    "AdaLN audio_condition time=1.0; target audio still "
                    "denoises. 2–15s, ≤3 clips, total ≤15s. Not lite.\n"
                    "  --ref-video MP4    Official Ref2VA video (CUDA bypass). "
                    "Canvas = Metal h3_reference_video_canvas. Vision: pair "
                    "every 12 frames + <Video n> timestamps. Visual pack is "
                    "H3_SEG_REF_IMAGE (Metal); logs H3_SEG_REF_VIDEO. "
                    "Embedded audio → H3_SEG_REF_AUDIO unless "
                    "--ref-silent-video. Ref2VA DiT required. Mutually "
                    "exclusive with keyframes / --ref-image. Default --cfg 1.\n"
                    "  --ref-silent-video MP4  Like --ref-video but ignore "
                    "embedded soundtrack.\n"
                    "  --last-frame IMG   Official L2VA/FL2VA condition: jpeg → "
                    "VAE encode → DiT COND rows + Qwen vision TE. COVER-crop to "
                    "canvas (Metal last-frame fit). keyframes=[mux_last]; "
                    "condition_time = text_seq + (mux_frames-1)*(5/3) so the "
                    "lock sits on the delivered last frame (5s mux 120 of "
                    "VAE 124; Metal aligned-last=123 is dropped). Default "
                    "--cfg 1. Not G3-lite; do not mix with lite flags or "
                    "--ref-*. With --first-frame: FL2VA pack "
                    "[text|cond0|cond1|audio|video], first STRETCH at "
                    "text_seq, last COVER at mux-last RoPE. With --skip-dit: "
                    "VAE roundtrip "
                    "ceiling for the last-frame canvas (use decoded f0 vs "
                    "COVER; do not use temporally-blurred still-plant last "
                    "frame as the ceiling).\n"
                    "  --cond-rows RAW   Reload official cond_rows.raw from a "
                    "prior --first-frame / --last-frame / FL2VA / --ref-image "
                    "run (already 0.999-augmented). Skips "
                    "host VAE encode. Still requires --first-frame and/or "
                    "--last-frame, or --ref-image. Dual keyframes expect 2× "
                    "frame rows. Not lite.\n"
                    "  --vae-encode PATH  Video VAE *encoder* weights "
                    "(fp16 encoder.*). Default: --vae. INT8 decode ckpt often "
                    "cannot encode; pass minimax_h3_video_vae_fp16 if they "
                    "differ. T2VA *quality* decode uses that fp16 file "
                    "(256px tiles + deblock16, F32 GEMM); INT8 VAE is the "
                    "speed decode. Do not set H3_VAE_NO_SPATIAL_TILE=1 on "
                    "the quality knife.\n"
                    "  --vision PATH      Qwen visual tower. Default: --te "
                    "(same NVFP4 file with visual.*).\n"
                    "  --no-vision-inject Diagnostic A/B: keep official VAE "
                    "COND, skip Picture+Qwen vision TE (bare prompt ids). "
                    "Not a quality-gate path.\n"
                    "  --dit-text-tags    Debug: Metal-like AdaLN VIDEO@σ on "
                    "Picture+vision span (tag=0 on START+pads+END). CUDA INT8 "
                    "official F2VA defaults this OFF — those rows are condition "
                    "embeddings, not video latent. Not a Metal bit-match claim.\n"
                    "  --no-dit-text-tags Explicit CUDA INT8 default (alias): "
                    "keep vision TE, all text rows TEXT AdaLN.\n"
                    "  --init-latent RAW  Debug only; non-official G3-lite. "
                    "F32 [T,H/2,W/2,96] start latent. Not official F2VA/L2VA.\n"
                    "  --init-first-frame Debug only; non-official G3-lite. "
                    "t=0 img2img mix from --init-latent (img_w 0.25→1). "
                    "Not official condition.\n"
                    "  --lock-first-frame Debug only; non-official G3-lite. "
                    "Keep t=0 latent tokens fixed each step (standing-pose "
                    "risk). Not official condition.\n"
                    "  --mux PATH.mp4     ffmpeg container (default DIR/out.mp4). "
                    "T2VA muxes decoded audio (dialogue+ambience) unless "
                    "--skip-audio. Not a silent AAC pad.\n",
                    argv[0], (unsigned)H3_CUDA_TEXT_CONTEXT_DEFAULT,
                    (unsigned)H3_CUDA_TEXT_CONTEXT_HARD_MAX);
            return 0;
        } else {
            fprintf(stderr, "unknown option: %s\n", argv[iarg]);
            return 2;
        }
    }

    h3_step_cache_apply_env(&step_cache);
    if (!h3_step_cache_validate(&step_cache, magcache_thresh,
                               reuse_explicit ? reuse : 1, step_cache_err,
                               sizeof(step_cache_err))) {
        fprintf(stderr, "h3: step_cache: %s\n", step_cache_err);
        return 2;
    }

    {
        h3_generate_preset_in pin;
        memset(&pin, 0, sizeof(pin));
        pin.preset_name = preset_name;
        pin.lora_path = lora_path;
        pin.steps_explicit = steps_explicit;
        pin.steps = steps;
        pin.sdpa_explicit = sdpa_explicit;
        pin.sdpa = sdpa;
        pin.cfg_explicit = cfg_explicit;
        pin.cfg_scale = cfg_scale;
        pin.magcache_thresh = magcache_thresh;
        pin.step_cache_kind = (int)step_cache.kind;
        pin.reuse_interval = reuse_explicit ? reuse : 1;
        if (!h3_generate_preset_apply(&pin, &preset_out, error, sizeof(error))) {
            fprintf(stderr, "h3: %s\n", error);
            return 1;
        }
        steps = preset_out.steps;
        sdpa = preset_out.sdpa;
        cfg_scale = preset_out.cfg_scale;
        if (preset_out.lora_path)
            lora_path = preset_out.lora_path;
        if (preset_out.non_gold)
            fprintf(stderr, "h3: preset=turbo4 非金线（MAE 金线不适用）\n");
    }
    if (ulysses < 1)
        ulysses = 1;
    ulysses_cfg.tp_size = 1;
    ulysses_cfg.ulysses = ulysses;
    ulysses_cfg.ring = 1;
    ulysses_cfg.world_size = ulysses;
    if (!h3_ulysses_validate(H3_DIT_W4A8_HEADS, H3_ULYSSES_PACKED_ALIGN, 1,
                             ulysses, 1, error, sizeof(error))) {
        fprintf(stderr, "h3: %s\n", error);
        return 1;
    }
    if (ulysses > 1 && token_reduction) {
        fprintf(stderr, "h3: --ulysses>1 与 --token-reduction 互斥\n");
        return 1;
    }
    if (!te_cache_dir || !te_cache_dir[0]) {
        const char *e = getenv("H3_TE_PROMPT_CACHE");
        if (e && e[0])
            te_cache_dir = e;
    }
    if (strcmp(vae_decode, "official") != 0 && strcmp(vae_decode, "tae") != 0) {
        fprintf(stderr, "h3: --vae-decode must be official|tae\n");
        return 1;
    }
    if (!strcmp(vae_decode, "tae")) {
        if (!preset_out.non_gold) {
            fprintf(stderr, "h3: TAE 不是金线成片（仅 --preset turbo4）\n");
            return 1;
        }
        if (!tae_path || !tae_path[0]) {
            fprintf(stderr, "h3: --vae-decode tae 需要 --tae 权（不静默 official）\n");
            return 1;
        }
        {
            void *tae_ctx = NULL;
            if (!h3_tae_open(tae_path, &tae_ctx, error, sizeof(error))) {
                fprintf(stderr, "h3: %s\n", error);
                return 1;
            }
            h3_tae_close(tae_ctx);
        }
    }

    if (probe_vae_load && !vae_path) {
        fprintf(stderr, "--probe-vae-load requires --vae\n");
        return 2;
    }
    if (!probe_vae_load && !dit_path) {
        fprintf(stderr, "need --dit\n");
        return 2;
    }
    if (skip_dit && !init_latent_path && !first_frame_path &&
        !last_frame_path) {
        fprintf(stderr,
                "--skip-dit requires --init-latent or --first-frame / "
                "--last-frame (VAE encode→decode roundtrip)\n");
        return 2;
    }
    if (skip_dit) skip_te = 1;
    if ((first_frame_path || last_frame_path || n_ref_images ||
         ref_video_path) &&
        !cfg_explicit)
        cfg_scale = 1.0f; /* official F2VA/L2VA/FL2VA/Ref2VA: distilled + image */
    official_kf = (first_frame_path ? 1 : 0) | (last_frame_path ? 2 : 0);
    if ((n_ref_images || ref_video_path) &&
        (first_frame_path || last_frame_path)) {
        fprintf(stderr,
                "h3: Ref2VA --ref-image/--ref-video cannot be combined with "
                "--first-frame / --last-frame keyframes\n");
        return 1;
    }
    if (n_ref_audios && !n_ref_images && !ref_video_path) {
        fprintf(stderr,
                "h3: --ref-audio requires --ref-image or --ref-video "
                "(standalone audio is invalid)\n");
        return 1;
    }
    if (!skip_audio && !audio_vae_path) {
        if (h3_resolve_default_audio_vae(vae_path, dit_path, audio_vae_resolved,
                                         sizeof(audio_vae_resolved))) {
            audio_vae_path = audio_vae_resolved;
        } else {
            fprintf(stderr,
                    "warning: T2VA needs minimax_h3_audio_vae_fp32.safetensors "
                    "(--audio-vae or H3_AUDIO_VAE); mux will be silent\n");
        }
    }
    if (!mux_path && !skip_vae && !probe_vae_load) {
        snprintf(mux_resolved, sizeof(mux_resolved), "%s/out.mp4", out_dir);
        mux_path = mux_resolved;
    }
    if (n_ref_audios && !audio_vae_path) {
        fprintf(stderr, "h3: --ref-audio requires --audio-vae\n");
        return 1;
    }
    if (n_ref_audios && (first_frame_path || last_frame_path)) {
        fprintf(stderr,
                "h3: Ref2VA --ref-audio cannot be combined with "
                "--first-frame / --last-frame keyframes\n");
        return 1;
    }
    if (ref_video_path && n_ref_images) {
        fprintf(stderr,
                "h3: --ref-video and --ref-image are mutually exclusive in "
                "this CUDA bypass slice\n");
        return 1;
    }
    if ((n_ref_images || ref_video_path) &&
        (init_latent_path || lock_first_frame || init_first_frame)) {
        fprintf(stderr,
                "--ref-image/--ref-video is official Ref2VA; do not mix with "
                "G3-lite\n");
        return 1;
    }
    if (n_ref_audios &&
        (init_latent_path || lock_first_frame || init_first_frame)) {
        fprintf(stderr, "--ref-audio is official Ref2VA; do not mix with G3-lite\n");
        return 1;
    }
    if (n_ref_images || ref_video_path) {
        if (path_contains_ci(dit_path, "fl2va")) {
            fprintf(stderr,
                    "--ref-* requires the Ref2VA DiT; refusing FL2VA %s "
                    "(will not fall back to FL2VA + init-latent)\n",
                    dit_path);
            return 1;
        }
        if (!path_contains_ci(dit_path, "ref2va")) {
            fprintf(stderr,
                    "--ref-* requires a Ref2VA DiT (--dit path must "
                    "contain ref2va). Missing/wrong file is a hard fail, not "
                    "an FL2VA fallback.\n");
            return 1;
        }
    }
    if ((first_frame_path || last_frame_path) &&
        (init_latent_path || lock_first_frame || init_first_frame)) {
        fprintf(stderr,
                "%s is official condition; do not mix with G3-lite\n",
                official_keyframe_flag(first_frame_path, last_frame_path));
        return 1;
    }
    if (!skip_vae && !vae_path) {
        fprintf(stderr, "need --vae (or --skip-vae)\n");
        return 2;
    }
    if (!probe_vae_load && !skip_te && !te_path) {
        fprintf(stderr, "need --te (or --skip-te for synth condition)\n");
        return 2;
    }
    if (cond_rows_path && !first_frame_path && !last_frame_path &&
        !n_ref_images && !ref_video_path) {
        fprintf(stderr,
                "--cond-rows requires --first-frame / --last-frame / "
                "--ref-image / --ref-video (official cache)\n");
        return 1;
    }
    if (init_latent_path || lock_first_frame || init_first_frame) {
        fprintf(stderr,
                "warning: non-official G3-lite (debug; not official "
                "--first-frame / --last-frame / --ref-image condition)\n");
    }
    if ((first_frame_path || last_frame_path || n_ref_images ||
         ref_video_path) &&
        skip_te && !skip_dit) {
        fprintf(stderr,
                "%s requires TE (do not pass --skip-te)\n",
                ref_video_path   ? "--ref-video"
                : n_ref_images   ? "--ref-image"
                                 : official_keyframe_flag(first_frame_path,
                                                          last_frame_path));
        return 2;
    }
    if ((first_frame_path || last_frame_path || n_ref_images ||
         ref_video_path) &&
        skip_audio && !skip_dit) {
        fprintf(stderr,
                "%s requires fl2va/ref2va audio pack (do not pass --skip-audio)\n",
                ref_video_path   ? "--ref-video"
                : n_ref_images   ? "--ref-image"
                                 : official_keyframe_flag(first_frame_path,
                                                          last_frame_path));
        return 2;
    }
    if (ref_video_path && !ref_silent_video && !audio_vae_path) {
        fprintf(stderr,
                "h3: --ref-video with soundtrack requires --audio-vae "
                "(or pass --ref-silent-video)\n");
        return 1;
    }
    if (!vae_encode_path) vae_encode_path = vae_path;
    if (!vision_path) vision_path = te_path;

    /* Shape from width/height/seconds (Metal h3_temporal / VAE spatial /16). */
    {
        int vae_out_frames;
        int mux_n;
        frames_req = frames_override > 0 ? frames_override : (seconds * 24);
        frames_aligned = align_frame_count(frames_req < 1 ? 1 : frames_req);
        /* Bind last-frame to the muxed last pixel, not aligned-1 (dropped). */
        last_keyframe = frames_req > 0 ? frames_req - 1 : 0;
        if (frames_aligned > 0 && last_keyframe > frames_aligned - 1)
            last_keyframe = frames_aligned - 1;
        latent_w = width / 16;
        latent_h = height / 16;
        latent_t = video_latent_t_from_frames(frames_aligned);
        if (latent_t < 1) latent_t = 1;
        if ((latent_h % 2) || (latent_w % 2)) {
            fprintf(stderr, "latent H/W must be even (pixel %% 32 == 0)\n");
            return 2;
        }
        /* Metal/Comfy: (T-2)/5*17+5 when T>=7 and (T-2)%5==0; else legacy T*4. */
        if (latent_t >= 7 && (latent_t - 2) % 5 == 0)
            vae_out_frames = ((latent_t - 2) / 5) * 17 + 5;
        else if (latent_t < 7)
            vae_out_frames = 22; /* pad-to-7 metal chunk */
        else
            vae_out_frames = latent_t * 4;
        mux_n = frames_req < 1 ? 1 : frames_req;
        if (mux_n > vae_out_frames) mux_n = vae_out_frames;
        printf("temporal: frames_req=%d aligned=%d → latent_t=%d "
               "(VAE metal temporal emits %d frames; mux %d @24fps = %.4fs)\n",
               frames_req, frames_aligned, latent_t, vae_out_frames, mux_n,
               (double)mux_n / 24.0);
        if (seconds > 0 &&
            fabs((double)mux_n / 24.0 - (double)seconds) > 0.2) {
            fprintf(stderr,
                    "warning: mux wall-clock %.4fs != --seconds %d "
                    "(pass --seconds without --frames for product 5s/10s; "
                    "--frames 17 is a 0.92s dev clip)\n",
                    (double)mux_n / 24.0, seconds);
        }
    }
    /* Quality default: global SDPA. Local --window W (e.g. 18) only attends
     * ±W tokens; with T2AV pack [text|audio|video] audio_seq≈400, text never
     * reaches video → dialogue-prior mode collapse. Product speed may still
     * pass --window 18 explicitly. */
    if (!have_window)
        setenv("H3_DIT_W4A8_ATTN_WINDOW", "0", 1);
    if (getenv("H3_DIT_DIST_ATTN") && getenv("H3_DIT_DIST_ATTN")[0] == '1')
        dist_attn = 1;

    setenv("H3_CUDA_SDPA_BACKEND", sdpa, 1);
    {
        h3_cuda_device_caps caps;
        h3_sdpa_backend_id sid;
        int avail = 0;
        char reason[320];
        char caps_line[384];
        if (h3_cuda_query_caps(device, &caps, error, sizeof(error))) {
            h3_cuda_caps_format(&caps, caps_line, sizeof(caps_line));
            printf("device_caps: %s\n", caps_line);
            h3_sdpa_resolve(sdpa, &caps, &sid, &avail, reason, sizeof(reason));
            printf("sdpa_select: request=%s resolved=%s available=%d (%s)\n",
                   sdpa, h3_sdpa_backend_id_name(sid), avail, reason);
            if (!avail && (sid == H3_SDPA_ID_SAGE3 || sid == H3_SDPA_ID_SOL_CUTE)) {
                fprintf(stderr,
                        "error: Tier-1 SDPA '%s' not available: %s\n"
                        "  (will not silently use nested-loop/sage2 fake kernels)\n",
                        h3_sdpa_backend_id_name(sid), reason);
                return 1;
            }
            if (sid == H3_SDPA_ID_SAGE2 || sid == H3_SDPA_ID_SOL) {
                fprintf(stderr,
                        "warning: '%s' is forbidden as a product backend "
                        "(reference / nested-loop fake)\n",
                        h3_sdpa_backend_id_name(sid));
            }
        } else {
            fprintf(stderr, "warning: device_caps probe failed: %s\n", error);
        }
    }
    effective_reuse = reuse_explicit ? reuse : 1;
    snprintf(reuse_label, sizeof(reuse_label), "%s",
             reuse_explicit ? "" : "off");
    if (reuse_explicit)
        snprintf(reuse_label, sizeof(reuse_label), "%d", effective_reuse);
    if (!ensure_dir(out_dir)) {
        fprintf(stderr, "cannot create out dir %s\n", out_dir);
        return 1;
    }

    printf("=== h3_cuda_generate (CUDA bypass · %s pack) ===\n",
           (n_ref_images || ref_video_path) ? "ref2va" : "fl2va");
    {
        const char *aw = getenv("H3_DIT_W4A8_ATTN_WINDOW");
        printf("shape: %dx%d %ds → latent %dx%dx%d  dit_layers=%u te_layers=%u "
               "vae_blocks=%u vae_repeats=%d steps=%d reuse=%s magcache=%.3g%s "
               "step_cache=%s/%.3g token_red=%d "
               "sdpa=%s attn_window=%s dist_attn=%s overlap=%d cfg=%.3g seed=0x%llx "
               "context_budget=%zu (default %u; hard_max %u) preset=%s ulysses=%d "
               "vae_decode=%s\n",
               width, height, seconds, latent_t, latent_h, latent_w, dit_layers,
               te_layers, vae_blocks, vae_repeats, steps, reuse_label,
               (double)magcache_thresh,
               magcache_calibrated ? "(cal)" : "",
               h3_step_cache_kind_name(step_cache.kind),
               (double)step_cache.thresh, token_reduction, sdpa,
               aw && *aw ? aw : "0", dist_attn ? "on" : "off", overlap,
               (double)cfg_scale,
               (unsigned long long)seed, context,
               (unsigned)H3_CUDA_TEXT_CONTEXT_DEFAULT,
               (unsigned)H3_CUDA_TEXT_CONTEXT_HARD_MAX,
               preset_name && preset_name[0] ? preset_name : "none", ulysses,
               vae_decode);
        if (te_cache_dir && te_cache_dir[0])
            printf("te_prompt_cache: dir=%s (opt-in; gold default off)\n",
                   te_cache_dir);
        if (first_frame_path || last_frame_path || n_ref_images ||
            ref_video_path) {
            printf("cfg_defaults: t2va=5 f2va=1 l2va=1 fl2va=1 ref2va=1 "
                   "this_run=%.3g explicit=%d (official image-condition "
                   "default 1; T2VA theme stays 5)\n",
                   (double)cfg_scale, cfg_explicit);
        }
    }
    if (!skip_dit && dit_layers < H3_DIT_W4A8_N_BLOCKS) {
        fprintf(stderr,
                "warning: --layers %u < %d produces OOD velocity (absmax~100) "
                "and color-block VAE output; use --layers %d for quality\n",
                dit_layers, H3_DIT_W4A8_N_BLOCKS, H3_DIT_W4A8_N_BLOCKS);
    }
    if (effective_reuse > 1 && magcache_thresh > 0.0f)
        printf("note: magcache ignored when reuse>1 (fixed schedule wins)\n");
    fflush(stdout);

    if (probe_vae_load) {
        h3_vae_int8_cuda_weights *probe_w = NULL;
        int repeat;
        if (!h3_vae_int8_store_open(vae_path, &vae_store, error,
                                    sizeof(error))) {
            fprintf(stderr, "vae open: %s\n", error);
            goto cleanup;
        }
        printf("probe_vae_load: path=%s repeats=%d blocks=%u cache=%s resident=%s\n",
               vae_path, vae_repeats, vae_blocks,
               getenv("H3_VAE_WEIGHT_CACHE") ? getenv("H3_VAE_WEIGHT_CACHE") : "0",
               getenv("H3_VAE_CUDA_W_RESIDENT")
                   ? getenv("H3_VAE_CUDA_W_RESIDENT")
                   : "1");
        for (repeat = 0; repeat < vae_repeats; repeat++) {
            double load_ms = 0.0;
            if (!h3_vae_int8_cuda_weights_load(&vae_store, vae_blocks, &probe_w,
                                               &load_ms, error,
                                               sizeof(error))) {
                fprintf(stderr, "vae load probe: %s\n", error);
                goto cleanup;
            }
            printf("probe_vae_load[%d/%d]: load_ms=%.1f\n", repeat + 1,
                   vae_repeats, load_ms);
            h3_vae_int8_cuda_weights_free(probe_w);
            probe_w = NULL;
        }
        rc = 0;
        goto cleanup;
    }

    t_all0 = now_ms();
    /* Prefer Comfy INT8 ConvRot DiT (theme-follow path); else W4A8. */
    dit_int8 = h3_dit_int8_store_open(dit_path, error, sizeof(error));
    if (dit_int8) {
        dit_is_int8 = 1;
        printf("dit_format: INT8 ConvRot (theme-follow path)\n");
        if (n_ref_images || ref_video_path)
            printf("dit_tree: Ref2VA %s (not FL2VA)\n", dit_path);
        if (!h3_dit_w4a8_load_core_int8(dit_int8, &core, error,
                                        sizeof(error))) {
            fprintf(stderr, "load_core_int8: %s\n", error);
            goto cleanup;
        }
    } else {
        dit_store = h3_w4a8_store_open(dit_path, error, sizeof(error));
        if (!dit_store) {
            if (n_ref_images || ref_video_path)
                fprintf(stderr,
                        "Ref2VA DiT load failed (not falling back to FL2VA): "
                        "%s\n",
                        error);
            else
                fprintf(stderr, "dit open (W4A8 after INT8 reject): %s\n",
                        error);
            return 1;
        }
        printf("dit_format: W4A8 mixed\n");
        if (n_ref_images || ref_video_path)
            printf("dit_tree: Ref2VA %s (not FL2VA)\n", dit_path);
        if (!h3_dit_w4a8_load_core(dit_store, &core, error, sizeof(error))) {
            fprintf(stderr, "load_core: %s\n", error);
            goto cleanup;
        }
    }
    /* Comfy EmptyMiniMaxH3LatentAV always includes audio rows in the pack.
     * fl2va DiT has audio_patch/final. Decode+mux is the T2VA product default;
     * --skip-audio forces legacy text|video (silent mp4). */
    pack_audio = (!skip_audio && core.audio_patch_w && core.final_audio_w) ? 1
                                                                          : 0;
    decode_audio = (audio_vae_path && pack_audio && !skip_dit) ? 1 : 0;
    if (audio_vae_path && !skip_audio && !pack_audio) {
        fprintf(stderr,
                "audio-vae requested but DiT missing audio_patch/final weights\n");
        goto cleanup;
    }
    if (pack_audio) {
        if (audio_t < 1) {
            /* Comfy temporal_shape: round(aligned_frames/24 * 40). */
            audio_t = (int)lround((double)frames_aligned * 40.0 / 24.0);
            if (audio_t < 1) audio_t = 1;
        }
        printf("audio_pack: T=%d seq=%d (Comfy text|audio|video)%s\n", audio_t,
               audio_t * 2, decode_audio ? " + decode" : " decode=off");
        if (audio_vae_path)
            printf("audio_vae: %s (%s)\n", audio_vae_path,
                   audio_vae_explicit ? "explicit" : "auto");
    } else {
        printf("audio_pack: off (legacy text|video; theme may collapse)\n");
    }
    if (ulysses > 1)
        device = 0; /* N>1 ignores --device; devices 0..N-1 */
    cuda = h3_cuda_create(device, error, sizeof(error));
    if (!cuda) {
        fprintf(stderr, "cuda: %s\n", error);
        goto cleanup;
    }
    if (ulysses > 1) {
        int ids[8];
        int ui;
        for (ui = 0; ui < ulysses && ui < 8; ui++)
            ids[ui] = ui;
        if (!h3_ulysses_comm_init(&ulysses_cfg, ids, ulysses, error,
                                  sizeof(error))) {
            fprintf(stderr, "h3: ulysses nccl: %s\n", error);
            goto cleanup;
        }
        printf("ulysses: N=%d NCCL inited (experimental; default still U=1)\n",
               ulysses);
    }
    h3_vae_encode_f32_bind_cuda(cuda);
    h3_vision_encode_f32_bind_cuda(cuda);
    printf("nvrtc: %s\n",
           h3_cuda_nvrtc_ok() ? h3_cuda_nvrtc_path() : "MISSING libnvrtc.so.12");
    /* TE NVFP4 linears → cuBLAS (H3_NVFP4_HOST_GEMM=1 to force host). */
    if (!skip_dit && !skip_te) {
        if (!h3_nvfp4_cuda_enable(cuda, error, sizeof(error))) {
            fprintf(stderr, "te cuda bind: %s\n", error);
            goto cleanup;
        }
        printf("te_backend: %s\n",
               h3_nvfp4_cuda_enabled() ? "cublas" : "host_gemm");
    }
    h3_dit_w4a8_config_init(&cfg);
    if (!skip_dit) {
        if (dit_is_int8)
            dit = h3_dit_w4a8_open_int8(cuda, dit_int8, &cfg, error,
                                        sizeof(error));
        else
            dit = h3_dit_w4a8_open(cuda, dit_store, &cfg, error, sizeof(error));
        if (!dit) {
            fprintf(stderr, "dit open: %s\n", error);
            goto cleanup;
        }
        if (token_reduction) {
            h3_dit_w4a8_set_token_reduction(dit, 1);
            printf("token_reduction: on (Metal pair semantics; begin/end via "
                   "env)\n");
        }
        if (lora_path && lora_path[0]) {
            if (!(lora_scale > 0.0f) || !isfinite(lora_scale))
                lora_scale = 1.0f;
            if (!h3_lora_load(lora_path, lora_scale, &lora_set, error,
                              sizeof(error))) {
                fprintf(stderr, "h3: lora 非金线: %s\n", error);
                goto cleanup;
            }
            lora_loaded = 1;
            if (!h3_dit_w4a8_set_lora(dit, &lora_set, error, sizeof(error))) {
                fprintf(stderr, "h3: lora attach: %s\n", error);
                goto cleanup;
            }
            printf("lora: path=%s scale=%.3g pairs=%d ignored_vsa=%d (非金线)\n",
                   lora_set.path, (double)lora_set.scale, lora_set.n_pairs,
                   lora_set.ignored_vsa_tensors);
        }
        if (ulysses > 1 && dit) {
            h3_dit_w4a8_sp_comm comm;
            int ids[8];
            int ui;
            /* Device-resident patch↔layers↔final when unset. Set
             * H3_DIT_W4A8_HOST_FWD=1 to restore host hidden staging. */
            memset(&comm, 0, sizeof(comm));
            comm.cfg = &ulysses_cfg;
            comm.bind = gen_uly_bind;
            comm.set_compute = gen_uly_set_compute;
            comm.a2a_qkv = gen_uly_a2a_qkv;
            comm.a2a_attn_out = gen_uly_a2a_attn;
            uly_cuda[0] = cuda;
            uly_dit[0] = dit;
            for (ui = 0; ui < ulysses && ui < 8; ui++)
                ids[ui] = ui;
            for (ui = 1; ui < ulysses && ui < 8; ui++) {
                uly_cuda[ui] = h3_cuda_create(ui, error, sizeof(error));
                if (!uly_cuda[ui]) {
                    fprintf(stderr, "h3: ulysses rank %d cuda: %s\n", ui, error);
                    goto cleanup;
                }
                if (dit_is_int8)
                    uly_dit[ui] = h3_dit_w4a8_open_int8(uly_cuda[ui], dit_int8,
                                                        &cfg, error,
                                                        sizeof(error));
                else
                    uly_dit[ui] = h3_dit_w4a8_open(uly_cuda[ui], dit_store, &cfg,
                                                   error, sizeof(error));
                if (!uly_dit[ui]) {
                    fprintf(stderr, "h3: ulysses rank %d dit: %s\n", ui, error);
                    goto cleanup;
                }
                if (lora_loaded &&
                    !h3_dit_w4a8_set_lora(uly_dit[ui], &lora_set, error,
                                          sizeof(error))) {
                    fprintf(stderr, "h3: ulysses rank %d lora: %s\n", ui, error);
                    goto cleanup;
                }
            }
            if (!h3_dit_w4a8_set_ulysses(dit, uly_dit, ids, ulysses, &comm,
                                         error, sizeof(error))) {
                fprintf(stderr, "h3: ulysses attach: %s\n", error);
                goto cleanup;
            }
            printf("ulysses: N=%d DiT SP attached (A2A after QKV+LoRA; "
                   "SDPA seq=S_real heads=H/U; device-resident unless "
                   "HOST_FWD=1)\n",
                   ulysses);
            /* Rank open leaves current device on U-1; TE/primary DiT are 0. */
            if (cudaSetDevice(0) != cudaSuccess) {
                fprintf(stderr, "h3: ulysses restore primary device failed\n");
                goto cleanup;
            }
        }
    } else if (first_frame_path && last_frame_path) {
        printf("skip_dit: VAE roundtrip encode→decode from FL2VA "
               "(first@t0 last@tLast lerp middle; not G3-lite; not last-frame "
               "PSNR ceiling)\n");
    } else if (last_frame_path) {
        printf("skip_dit: VAE roundtrip encode→decode from --last-frame "
               "(no DiT; not G3-lite)\n");
    } else if (first_frame_path) {
        printf("skip_dit: VAE roundtrip encode→decode from --first-frame "
               "(no DiT; not G3-lite)\n");
    } else {
        printf("skip_dit: decode-only from --init-latent\n");
    }

    /* Stretch (F2VA) and/or cover (L2VA) canvases are VAE-roundtrip refs. */
    if (first_frame_path) {
        if (!h3_ffmpeg_read_image_f32(first_frame_path, width, height,
                                      H3_IMAGE_FIT_STRETCH, &ff_rgb, error,
                                      sizeof(error))) {
            fprintf(stderr, "first-frame read: %s\n", error);
            goto cleanup;
        }
        snprintf(pathbuf, sizeof(pathbuf), "%s/first_frame_in.ppm", out_dir);
        if (!write_chw01_ppm(pathbuf, ff_rgb, width, height, error,
                             sizeof(error))) {
            fprintf(stderr, "first-frame ppm: %s\n", error);
            goto cleanup;
        }
        printf("first_frame: stretch %dx%d from %s (wrote %s)\n", width, height,
               first_frame_path, pathbuf);
        fflush(stdout);
    }
    if (last_frame_path) {
        if (!h3_ffmpeg_read_image_f32(last_frame_path, width, height,
                                      H3_IMAGE_FIT_COVER, &lf_rgb, error,
                                      sizeof(error))) {
            fprintf(stderr, "last-frame read: %s\n", error);
            goto cleanup;
        }
        snprintf(pathbuf, sizeof(pathbuf), "%s/last_frame_in.ppm", out_dir);
        if (!write_chw01_ppm(pathbuf, lf_rgb, width, height, error,
                             sizeof(error))) {
            fprintf(stderr, "last-frame ppm: %s\n", error);
            goto cleanup;
        }
        printf("last_frame: cover %dx%d from %s (wrote %s)\n", width, height,
               last_frame_path, pathbuf);
        fflush(stdout);
    }
    if (n_ref_images) {
        int ri;
        for (ri = 0; ri < n_ref_images; ri++) {
            int src_w = 0, src_h = 0, canvas_w = 0, canvas_h = 0;
            if (!h3_ffprobe_visual_size(ref_image_paths[ri], &src_w, &src_h,
                                        error, sizeof(error))) {
                fprintf(stderr, "ref-image probe %d: %s\n", ri + 1, error);
                goto cleanup;
            }
            if (!h3_dit_w4a8_reference_image_canvas(src_w, src_h, width, height,
                                                    0, &canvas_w, &canvas_h)) {
                fprintf(stderr,
                        "ref-image %d: cannot resolve canvas from %dx%d "
                        "(target %dx%d)\n",
                        ri + 1, src_w, src_h, width, height);
                goto cleanup;
            }
            if ((canvas_w % H3_DIT_W4A8_CANVAS_MULTIPLE) ||
                (canvas_h % H3_DIT_W4A8_CANVAS_MULTIPLE) ||
                (canvas_w / H3_DIT_W4A8_VAE_SPATIAL) % 2 ||
                (canvas_h / H3_DIT_W4A8_VAE_SPATIAL) % 2) {
                fprintf(stderr,
                        "ref-image %d canvas %dx%d not VAE-aligned\n", ri + 1,
                        canvas_w, canvas_h);
                goto cleanup;
            }
            if (!h3_ffmpeg_read_image_f32(ref_image_paths[ri], canvas_w,
                                          canvas_h, H3_IMAGE_FIT_STRETCH,
                                          &ref_rgb[ri], error, sizeof(error))) {
                fprintf(stderr, "ref-image read %d: %s\n", ri + 1, error);
                goto cleanup;
            }
            ref_pw[ri] = canvas_w;
            ref_ph[ri] = canvas_h;
            ref_lw[ri] = canvas_w / H3_DIT_W4A8_VAE_SPATIAL;
            ref_lh[ri] = canvas_h / H3_DIT_W4A8_VAE_SPATIAL;
            snprintf(pathbuf, sizeof(pathbuf), "%s/ref_image_%d_in.ppm",
                     out_dir, ri + 1);
            if (!write_chw01_ppm(pathbuf, ref_rgb[ri], canvas_w, canvas_h,
                                 error, sizeof(error))) {
                fprintf(stderr, "ref-image ppm %d: %s\n", ri + 1, error);
                goto cleanup;
            }
            printf("ref_image: n=%d src=%dx%d canvas=%dx%d latent=%dx%d "
                   "fit=down-only (not STRETCH/COVER to output) from %s "
                   "(wrote %s)\n",
                   ri + 1, src_w, src_h, canvas_w, canvas_h, ref_lh[ri],
                   ref_lw[ri], ref_image_paths[ri], pathbuf);
            fflush(stdout);
        }
    }
    if (ref_video_path) {
        int src_w = 0, src_h = 0, canvas_w = 0, canvas_h = 0;
        int max_frames = frames_override > 0 ? frames_override : seconds * 24;
        if (!h3_ffprobe_visual_size(ref_video_path, &src_w, &src_h, error,
                                    sizeof(error))) {
            fprintf(stderr, "ref-video probe: %s\n", error);
            goto cleanup;
        }
        if (!h3_dit_w4a8_reference_video_canvas(src_w, src_h, &canvas_w,
                                                &canvas_h)) {
            fprintf(stderr, "ref-video: cannot resolve canvas from %dx%d\n",
                    src_w, src_h);
            goto cleanup;
        }
        if ((canvas_w % H3_DIT_W4A8_CANVAS_MULTIPLE) ||
            (canvas_h % H3_DIT_W4A8_CANVAS_MULTIPLE)) {
            fprintf(stderr, "ref-video canvas %dx%d not VAE-aligned\n", canvas_w,
                    canvas_h);
            goto cleanup;
        }
        if (!h3_ffmpeg_read_video_f32(ref_video_path, canvas_w, canvas_h,
                                      frames_aligned > 0 ? frames_aligned
                                                         : max_frames,
                                      &ref_video_rgb, &ref_video_frames, error,
                                      sizeof(error))) {
            fprintf(stderr, "ref-video read: %s\n", error);
            goto cleanup;
        }
        ref_vw = canvas_w;
        ref_vh = canvas_h;
        ref_vlw = canvas_w / H3_DIT_W4A8_VAE_SPATIAL;
        ref_vlh = canvas_h / H3_DIT_W4A8_VAE_SPATIAL;
        printf("ref_video: src=%dx%d canvas=%dx%d frames=%d silent=%d from %s "
               "(H3_LAYOUT_REF_VIDEO → pack H3_SEG_REF_IMAGE visual; "
               "optional H3_SEG_REF_AUDIO)\n",
               src_w, src_h, canvas_w, canvas_h, ref_video_frames,
               ref_silent_video, ref_video_path);
        fflush(stdout);
    }

    /* Resolve real text length before RoPE / buffer sizing (Metal = exact). */
    if (!skip_dit && !skip_te && te_path) {
        size_t n_ids = 0;
        size_t max_ctx = context;
        int have_ids = 0;
        if (token_ids_file) {
            if (!h3_token_ids_load_file(token_ids_file, &token_ids, &n_ids,
                                        error, sizeof(error))) {
                fprintf(stderr, "token-ids-file: %s\n", error);
                goto cleanup;
            }
            have_ids = 1;
        } else if (token_ids_list) {
            if (!h3_token_ids_parse_list(token_ids_list, &token_ids, &n_ids,
                                         error, sizeof(error))) {
                fprintf(stderr, "token-ids: %s\n", error);
                goto cleanup;
            }
            have_ids = 1;
        } else if (token_canary) {
            if (!h3_token_ids_canary(token_canary, &token_ids, &n_ids, error,
                                     sizeof(error))) {
                fprintf(stderr, "token-canary: %s\n", error);
                goto cleanup;
            }
            have_ids = 1;
        }
        if (have_ids) {
            /* Never silent-truncate. Historical --context 16 chopped official
             * T2VA prompts and collapsed to the dialogue prior. Auto-raise to
             * exact id count up to HARD_MAX (≈ MiniMax ≤7000-char API). */
            if (n_ids > H3_CUDA_TEXT_CONTEXT_HARD_MAX) {
                fprintf(stderr,
                        "token-ids: count=%zu exceeds hard max %u "
                        "(official API ≤7000 chars ≈≤%u Qwen tokens; "
                        "shorten prompt)\n",
                        n_ids, (unsigned)H3_CUDA_TEXT_CONTEXT_HARD_MAX,
                        (unsigned)H3_CUDA_TEXT_CONTEXT_HARD_MAX);
                goto cleanup;
            }
            if (n_ids > max_ctx) {
                printf("te_tokens: raising --context budget %zu → %zu to fit "
                       "ids (NO truncate; Metal/Comfy exact-length)\n",
                       max_ctx, n_ids);
            } else if (n_ids < max_ctx) {
                printf("te_tokens: shrinking budget %zu → exact %zu "
                       "(no pad; Metal/Comfy exact-length)\n",
                       max_ctx, n_ids);
            }
            context = n_ids;
            printf("te_tokens: source=%s count=%zu context_budget_was=%zu "
                   "context_actual=%zu truncate=0\n",
                   token_ids_file   ? "file"
                   : token_ids_list ? "list"
                                    : "canary",
                   n_ids, max_ctx, context);
            if (n_ids >= 128) {
                if (h3_nvfp4_cuda_enabled()) {
                    fprintf(stderr,
                            "te_tokens: %zu — GQA uses GPU causal SDPA "
                            "(cuDNN fused, naive GPU fallback). "
                            "H3_NVFP4_HOST_GQA=1 forces host O(T^2).\n",
                            n_ids);
                } else {
                    fprintf(stderr,
                            "warning: TE tokens=%zu; host GQA is O(T^2). "
                            "Theme path is ~20 tokens / ~15s TE; long F2VA prompts "
                            "can look hung (CPU 99%%, GPU 0%%) without layer logs.\n",
                            n_ids);
                }
            }
            fflush(stdout);
            fflush(stderr);
        }
        if ((official_kf || n_ref_images || ref_video_path) && !token_ids) {
            fprintf(stderr,
                    "%s requires --token-ids-file / --token-ids / "
                    "--token-canary (official prompt ids)\n",
                    ref_video_path ? "--ref-video"
                    : n_ref_images ? "--ref-image"
                                   : official_keyframe_flag(first_frame_path,
                                                            last_frame_path));
            goto cleanup;
        }
        if (official_kf) {
            const char *vis = vision_path ? vision_path : te_path;
            const char *flag =
                official_keyframe_flag(first_frame_path, last_frame_path);
            const float *pic_rgb[2];
            size_t vis_tokens[2], vis_one, vis_elems, prompt_n, mm_n = 0;
            int grid_h[2], grid_w[2];
            size_t n_pic = 0, p;
            uint32_t *mm_ids = NULL;
            if (first_frame_path) pic_rgb[n_pic++] = ff_rgb;
            if (last_frame_path) pic_rgb[n_pic++] = lf_rgb;
            if (!no_vision_inject && !vis) {
                fprintf(stderr, "%s needs --vision or --te\n", flag);
                goto cleanup;
            }
            for (p = 0; p < n_pic; p++) {
                if (!pic_rgb[p]) {
                    fprintf(stderr, "%s rgb missing (picture %zu)\n", flag,
                            p + 1);
                    goto cleanup;
                }
            }
            if (no_vision_inject) {
                printf("vision_inject=0 (A/B: official COND kept; skip "
                       "Picture+Qwen vision TE; bare prompt ids=%zu)\n",
                       context);
                fflush(stdout);
            } else {
            vis_one = (size_t)(height / 32) * (size_t)(width / 32);
            vis_elems = n_pic * vis_one * (size_t)H3_CUDA_VISION_WIDTH;
            vision_merged = (float *)malloc(vis_elems * sizeof(float));
            vision_ds[0] = (float *)malloc(vis_elems * sizeof(float));
            vision_ds[1] = (float *)malloc(vis_elems * sizeof(float));
            vision_ds[2] = (float *)malloc(vis_elems * sizeof(float));
            if (!vision_merged || !vision_ds[0] || !vision_ds[1] ||
                !vision_ds[2]) {
                fprintf(stderr, "OOM vision buffers\n");
                goto cleanup;
            }
            printf("vision_encode: %s pictures=%zu "
                   "(cublas+cudnn SDPA if bound; H3_VISION_HOST=1 / "
                   "H3_VISION_SDPA=host rollback)\n",
                   vis, n_pic);
            fflush(stdout);
            for (p = 0; p < n_pic; p++) {
                size_t off = p * vis_one;
                float *merged_p =
                    vision_merged + off * (size_t)H3_CUDA_VISION_WIDTH;
                float *ds_p[3];
                size_t got = vis_one;
                double t_vis;
                ds_p[0] = vision_ds[0] + off * (size_t)H3_CUDA_VISION_WIDTH;
                ds_p[1] = vision_ds[1] + off * (size_t)H3_CUDA_VISION_WIDTH;
                ds_p[2] = vision_ds[2] + off * (size_t)H3_CUDA_VISION_WIDTH;
                t_vis = now_ms();
                if (!h3_vision_encode_f32(vis, pic_rgb[p], 1, height, width,
                                          merged_p, ds_p, &grid_h[p],
                                          &grid_w[p], &got, error,
                                          sizeof(error))) {
                    fprintf(stderr, "vision encode picture %zu: %s\n", p + 1,
                            error);
                    goto cleanup;
                }
                if (got > vis_one) {
                    fprintf(stderr,
                            "vision tokens=%zu overflowed per-image cap %zu\n",
                            got, vis_one);
                    goto cleanup;
                }
                vis_tokens[p] = got;
                printf("vision_encode: picture=%zu tokens=%zu patch_grid=%dx%d "
                       "merged_hw=%dx%d ms=%.1f (pass vision.tokens + patch "
                       "grid; not merged 1x1)\n",
                       p + 1, vis_tokens[p], grid_h[p], grid_w[p],
                       grid_h[p] / 2, grid_w[p] / 2, now_ms() - t_vis);
                fflush(stdout);
            }
            prompt_n = context;
            memset(vision_spans, 0, sizeof(vision_spans));
            if (!h3_multimodal_cuda_build_fl2va_ids(
                    n_pic, (const uint32_t *)token_ids, prompt_n, vis_tokens,
                    grid_h, grid_w, &mm_ids, &mm_n, vision_spans, error,
                    sizeof(error))) {
                fprintf(stderr, "build_fl2va_ids: %s\n", error);
                goto cleanup;
            }
            for (p = 0; p < n_pic; p++) {
                if (vision_spans[p].tokens < 2 ||
                    vision_spans[p].tokens - 2 != vis_tokens[p]) {
                    fprintf(stderr,
                            "span[%zu].tokens=%zu want START+pads+END "
                            "(pads=vision.tokens %zu)\n",
                            p, vision_spans[p].tokens, vis_tokens[p]);
                    free(mm_ids);
                    goto cleanup;
                }
            }
            if (mm_n > H3_CUDA_TEXT_CONTEXT_HARD_MAX) {
                fprintf(stderr,
                        "multimodal ids=%zu exceeds hard max %u "
                        "(prompt=%zu pictures=%zu)\n",
                        mm_n, (unsigned)H3_CUDA_TEXT_CONTEXT_HARD_MAX, prompt_n,
                        n_pic);
                free(mm_ids);
                goto cleanup;
            }
            free(token_ids);
            token_ids = (int32_t *)mm_ids;
            context = mm_n;
            vision_span_count = (int)n_pic;
            have_vision_span = 1;
            if (official_kf == 3) {
                printf("official_condition=1 keyframes=0,last last_keyframe=%d "
                       "aligned_last=%d pictures=2 vision.tokens=%zu+%zu "
                       "span.tokens=%zu+%zu text_seq=%zu vision_inject=1 "
                       "(multimodal; pack text|cond|cond|audio|video; "
                       "last-frame RoPE on mux last not VAE pad)\n",
                       last_keyframe,
                       frames_aligned > 0 ? frames_aligned - 1 : 0,
                       vis_tokens[0], vis_tokens[1],
                       vision_spans[0].tokens, vision_spans[1].tokens, context);
            } else {
                printf("official_condition=1 keyframes=%d vision.tokens=%zu "
                       "span.tokens=%zu text_seq=%zu vision_inject=1 "
                       "(multimodal; not bare prompt)\n",
                       last_frame_path ? last_keyframe : 0, vis_tokens[0],
                       vision_spans[0].tokens, context);
            }
            fflush(stdout);
            }
        }
        if (n_ref_images) {
            const char *vis = vision_path ? vision_path : te_path;
            size_t vis_tokens[H3_CUDA_MAX_REF_IMAGES];
            int grid_h[H3_CUDA_MAX_REF_IMAGES], grid_w[H3_CUDA_MAX_REF_IMAGES];
            size_t vis_elems = 0, prompt_n, mm_n = 0, p, cap, off;
            uint32_t *mm_ids = NULL;
            if (!no_vision_inject && !vis) {
                fprintf(stderr, "--ref-image needs --vision or --te\n");
                goto cleanup;
            }
            for (p = 0; p < (size_t)n_ref_images; p++) {
                if (!ref_rgb[p]) {
                    fprintf(stderr, "--ref-image rgb missing (picture %zu)\n",
                            p + 1);
                    goto cleanup;
                }
            }
            if (no_vision_inject) {
                printf("vision_inject=0 (A/B: official REF kept; skip "
                       "Picture+Qwen vision TE; bare prompt ids=%zu)\n",
                       context);
                fflush(stdout);
            } else {
                for (p = 0; p < (size_t)n_ref_images; p++) {
                    cap = (size_t)(ref_ph[p] / 32) * (size_t)(ref_pw[p] / 32);
                    vis_elems += cap;
                }
                vis_elems *= (size_t)H3_CUDA_VISION_WIDTH;
                vision_merged = (float *)malloc(vis_elems * sizeof(float));
                vision_ds[0] = (float *)malloc(vis_elems * sizeof(float));
                vision_ds[1] = (float *)malloc(vis_elems * sizeof(float));
                vision_ds[2] = (float *)malloc(vis_elems * sizeof(float));
                if (!vision_merged || !vision_ds[0] || !vision_ds[1] ||
                    !vision_ds[2]) {
                    fprintf(stderr, "OOM vision buffers\n");
                    goto cleanup;
                }
                printf("vision_encode: %s pictures=%d (Ref2VA host F32)\n", vis,
                       n_ref_images);
                fflush(stdout);
                off = 0;
                for (p = 0; p < (size_t)n_ref_images; p++) {
                    float *merged_p;
                    float *ds_p[3];
                    size_t got;
                    double t_vis;
                    cap = (size_t)(ref_ph[p] / 32) * (size_t)(ref_pw[p] / 32);
                    merged_p =
                        vision_merged + off * (size_t)H3_CUDA_VISION_WIDTH;
                    got = cap;
                    ds_p[0] = vision_ds[0] + off * (size_t)H3_CUDA_VISION_WIDTH;
                    ds_p[1] = vision_ds[1] + off * (size_t)H3_CUDA_VISION_WIDTH;
                    ds_p[2] = vision_ds[2] + off * (size_t)H3_CUDA_VISION_WIDTH;
                    t_vis = now_ms();
                    if (!h3_vision_encode_f32(vis, ref_rgb[p], 1, ref_ph[p],
                                              ref_pw[p], merged_p, ds_p,
                                              &grid_h[p], &grid_w[p], &got,
                                              error, sizeof(error))) {
                        fprintf(stderr, "vision encode ref picture %zu: %s\n",
                                p + 1, error);
                        goto cleanup;
                    }
                    if (got > cap) {
                        fprintf(stderr,
                                "vision tokens=%zu overflowed ref cap %zu\n",
                                got, cap);
                        goto cleanup;
                    }
                    vis_tokens[p] = got;
                    printf("vision_encode: picture=%zu tokens=%zu "
                           "patch_grid=%dx%d canvas=%dx%d ms=%.1f "
                           "(ordered <Picture %zu>)\n",
                           p + 1, vis_tokens[p], grid_h[p], grid_w[p],
                           ref_pw[p], ref_ph[p], now_ms() - t_vis, p + 1);
                    fflush(stdout);
                    off += cap;
                }
                prompt_n = context;
                memset(vision_spans, 0, sizeof(vision_spans));
                if (!h3_multimodal_cuda_build_fl2va_ids(
                        (size_t)n_ref_images, (const uint32_t *)token_ids,
                        prompt_n, vis_tokens, grid_h, grid_w, &mm_ids, &mm_n,
                        vision_spans, error, sizeof(error))) {
                    fprintf(stderr, "build_fl2va_ids (Ref2VA Picture): %s\n",
                            error);
                    goto cleanup;
                }
                for (p = 0; p < (size_t)n_ref_images; p++) {
                    if (vision_spans[p].tokens < 2 ||
                        vision_spans[p].tokens - 2 != vis_tokens[p]) {
                        fprintf(stderr,
                                "span[%zu].tokens=%zu want START+pads+END "
                                "(pads=vision.tokens %zu)\n",
                                p, vision_spans[p].tokens, vis_tokens[p]);
                        free(mm_ids);
                        goto cleanup;
                    }
                }
                if (mm_n > H3_CUDA_TEXT_CONTEXT_HARD_MAX) {
                    fprintf(stderr,
                            "multimodal ids=%zu exceeds hard max %u "
                            "(prompt=%zu pictures=%d)\n",
                            mm_n, (unsigned)H3_CUDA_TEXT_CONTEXT_HARD_MAX,
                            prompt_n, n_ref_images);
                    free(mm_ids);
                    goto cleanup;
                }
                free(token_ids);
                token_ids = (int32_t *)mm_ids;
                context = mm_n;
                vision_span_count = n_ref_images;
                have_vision_span = 1;
                printf("official_reference=1 H3_SEG_REF_IMAGE n=%d "
                       "vision.tokens=%zu span.tokens=%zu text_seq=%zu "
                       "vision_inject=1 (multimodal; pack "
                       "text|ref_img|audio|video; not keyframes)\n",
                       n_ref_images, vis_tokens[0], vision_spans[0].tokens,
                       context);
                fflush(stdout);
            }
        }
        if (ref_video_path && ref_video_rgb && ref_video_frames > 0) {
            const char *vis = vision_path ? vision_path : te_path;
            size_t samples =
                ((size_t)ref_video_frames + 11u) / 12u;
            size_t blocks = (samples + 1u) / 2u;
            size_t vis_tokens[H3_CUDA_MAX_REF_VIDEO_BLOCKS];
            int grid_h[H3_CUDA_MAX_REF_VIDEO_BLOCKS];
            int grid_w[H3_CUDA_MAX_REF_VIDEO_BLOCKS];
            double timestamps[H3_CUDA_MAX_REF_VIDEO_BLOCKS];
            size_t vis_elems, prompt_n, mm_n = 0, b, cap, off;
            uint32_t *mm_ids = NULL;
            if (blocks < 1 || blocks > H3_CUDA_MAX_REF_VIDEO_BLOCKS) {
                fprintf(stderr,
                        "ref-video: vision blocks=%zu unsupported (max %d)\n",
                        blocks, H3_CUDA_MAX_REF_VIDEO_BLOCKS);
                goto cleanup;
            }
            if (!no_vision_inject && !vis) {
                fprintf(stderr, "--ref-video needs --vision or --te\n");
                goto cleanup;
            }
            if (no_vision_inject) {
                printf("vision_inject=0 (A/B: ref-video encode kept; bare "
                       "prompt)\n");
                fflush(stdout);
            } else {
                cap = (size_t)(ref_vh / 32) * (size_t)(ref_vw / 32);
                vis_elems = blocks * cap * (size_t)H3_CUDA_VISION_WIDTH;
                vision_merged = (float *)malloc(vis_elems * sizeof(float));
                vision_ds[0] = (float *)malloc(vis_elems * sizeof(float));
                vision_ds[1] = (float *)malloc(vis_elems * sizeof(float));
                vision_ds[2] = (float *)malloc(vis_elems * sizeof(float));
                if (!vision_merged || !vision_ds[0] || !vision_ds[1] ||
                    !vision_ds[2]) {
                    fprintf(stderr, "OOM vision buffers (ref-video)\n");
                    goto cleanup;
                }
                printf("vision_encode: %s video blocks=%zu samples=%zu "
                       "frames=%d (Ref2VA host F32, pair every 12)\n",
                       vis, blocks, samples, ref_video_frames);
                fflush(stdout);
                off = 0;
                for (b = 0; b < blocks; b++) {
                    size_t first_sample = 2 * b;
                    size_t second_sample =
                        first_sample + 1 < samples ? first_sample + 1
                                                   : first_sample;
                    int first = (int)(first_sample * 12);
                    int second = (int)(second_sample * 12);
                    float *pair;
                    float *merged_p;
                    float *ds_p[3];
                    size_t got;
                    double t_vis;
                    if (first >= ref_video_frames)
                        first = ref_video_frames - 1;
                    if (second >= ref_video_frames)
                        second = ref_video_frames - 1;
                    timestamps[b] =
                        ((double)first_sample + (double)second_sample) / 4.0;
                    pair = extract_vision_pair_chtw(ref_video_rgb,
                                                    ref_video_frames, ref_vh,
                                                    ref_vw, first, second);
                    if (!pair) {
                        fprintf(stderr, "ref-video vision pair OOM/bad idx\n");
                        goto cleanup;
                    }
                    merged_p =
                        vision_merged + off * (size_t)H3_CUDA_VISION_WIDTH;
                    ds_p[0] = vision_ds[0] + off * (size_t)H3_CUDA_VISION_WIDTH;
                    ds_p[1] = vision_ds[1] + off * (size_t)H3_CUDA_VISION_WIDTH;
                    ds_p[2] = vision_ds[2] + off * (size_t)H3_CUDA_VISION_WIDTH;
                    got = cap;
                    t_vis = now_ms();
                    if (!h3_vision_encode_f32(vis, pair, 2, ref_vh, ref_vw,
                                              merged_p, ds_p, &grid_h[b],
                                              &grid_w[b], &got, error,
                                              sizeof(error))) {
                        fprintf(stderr, "vision encode ref-video block %zu: %s\n",
                                b, error);
                        free(pair);
                        goto cleanup;
                    }
                    free(pair);
                    if (got > cap) {
                        fprintf(stderr,
                                "vision tokens=%zu overflowed video cap %zu\n",
                                got, cap);
                        goto cleanup;
                    }
                    vis_tokens[b] = got;
                    printf("vision_encode: video_block=%zu tokens=%zu "
                           "patch_grid=%dx%d ts=%.1f frames=%d+%d ms=%.1f\n",
                           b, vis_tokens[b], grid_h[b], grid_w[b],
                           timestamps[b], first, second, now_ms() - t_vis);
                    fflush(stdout);
                    off += cap;
                }
                prompt_n = context;
                memset(vision_spans, 0, sizeof(vision_spans));
                if (!h3_multimodal_cuda_build_ref_video_ids(
                        1, blocks, timestamps, (const uint32_t *)token_ids,
                        prompt_n, vis_tokens, grid_h, grid_w, &mm_ids, &mm_n,
                        vision_spans, error, sizeof(error))) {
                    fprintf(stderr, "build_ref_video_ids: %s\n", error);
                    goto cleanup;
                }
                if (mm_n > H3_CUDA_TEXT_CONTEXT_HARD_MAX) {
                    fprintf(stderr,
                            "multimodal ids=%zu exceeds hard max %u "
                            "(ref-video blocks=%zu)\n",
                            mm_n, (unsigned)H3_CUDA_TEXT_CONTEXT_HARD_MAX,
                            blocks);
                    free(mm_ids);
                    goto cleanup;
                }
                free(token_ids);
                token_ids = (int32_t *)mm_ids;
                context = mm_n;
                vision_span_count = (int)blocks;
                have_vision_span = 1;
                printf("official_reference=1 H3_SEG_REF_VIDEO visual→"
                       "H3_SEG_REF_IMAGE blocks=%zu text_seq=%zu "
                       "vision_inject=1 (not keyframes)\n",
                       blocks, context);
                fflush(stdout);
            }
        }
    }

    video_seq = (size_t)latent_t * (size_t)(latent_h / 2) * (size_t)(latent_w / 2);
    n_lat = video_seq * H3_DIT_W4A8_VIDEO_PATCH;
    latent = (float *)malloc(n_lat * sizeof(float));
    velocity = (float *)malloc(n_lat * sizeof(float));
    sigmas = (float *)malloc((size_t)(steps + 1) * sizeof(float));
    text_h = (float *)malloc(context * H3_DIT_W4A8_HIDDEN * sizeof(float));
    pt = (float *)malloc(video_seq * sizeof(float));
    ph = (float *)malloc(video_seq * sizeof(float));
    pw = (float *)malloc(video_seq * sizeof(float));
    if (!latent || !velocity || !sigmas || !text_h || !pt || !ph || !pw) {
        fprintf(stderr, "OOM\n");
        goto cleanup;
    }
    if (cfg_scale > 1.0001f) {
        text_h_uncond =
            (float *)calloc(context * H3_DIT_W4A8_HIDDEN, sizeof(float));
        velocity_uncond = (float *)malloc(n_lat * sizeof(float));
        if (!text_h_uncond || !velocity_uncond) {
            fprintf(stderr, "OOM cfg buffers\n");
            goto cleanup;
        }
        /* Placeholder until TE pad-uncond is filled below. */
        printf("cfg: scale=%.3g uncond=te_pad_refiner (2× DiT evals)\n",
               (double)cfg_scale);
    } else {
        printf("cfg: scale=%.3g (disabled; %s)\n", (double)cfg_scale,
               official_kf == 3    ? "FL2VA default 1"
               : last_frame_path   ? "L2VA default 1"
               : first_frame_path  ? "F2VA default 1"
               : (n_ref_images || ref_video_path) ? "Ref2VA default 1"
                                   : "explicit");
    }
    /* FL2VA keyframes do not advance the layout cursor. Ref2VA images do
     * (+1 each), so target audio/video RoPE starts at text_seq + n_refs. */
    target_rope_t0 = context + (size_t)n_ref_images;
    if (pack_audio) {
        audio_seq = (size_t)2 * (size_t)audio_t;
        n_audio = audio_seq * H3_DIT_W4A8_AUDIO_CH;
        audio_c2t = (float *)calloc((size_t)H3_DIT_W4A8_AUDIO_CH * 2 *
                                        (size_t)audio_t,
                                    sizeof(float));
        audio_rows = (float *)calloc(n_audio, sizeof(float));
        audio_vel = (float *)calloc(n_audio, sizeof(float));
        apt = (float *)calloc(audio_seq, sizeof(float));
        aph = (float *)calloc(audio_seq, sizeof(float));
        apw = (float *)calloc(audio_seq, sizeof(float));
        if (!audio_c2t || !audio_rows || !audio_vel || !apt || !aph || !apw) {
            fprintf(stderr, "OOM audio\n");
            goto cleanup;
        }
        fill_normal_f32(seed + 0xA11DULL, audio_c2t,
                        (size_t)H3_DIT_W4A8_AUDIO_CH * 2 * (size_t)audio_t);
        if (!h3_dit_w4a8_pack_audio_latent(audio_c2t, audio_t, audio_rows,
                                           error, sizeof(error))) {
            fprintf(stderr, "pack audio: %s\n", error);
            goto cleanup;
        }
        /* target audio RoPE filled after ref-image/ref-audio segments. */
    }
    fill_normal_f32(seed, latent, n_lat);
    frame0_elems = (size_t)(latent_h / 2) * (size_t)(latent_w / 2) *
                   (size_t)H3_DIT_W4A8_VIDEO_PATCH;
    if (init_first_frame && !init_latent_path) {
        fprintf(stderr, "--init-first-frame requires --init-latent\n");
        goto cleanup;
    }
    if (init_latent_path && init_first_frame) {
        float img_w0;
        /* Prefix-read: full [T,...] dumps are OK; only t=0 is consumed. */
        frame0_img = (float *)malloc(frame0_elems * sizeof(float));
        if (!frame0_img) {
            fprintf(stderr, "OOM init-first-frame\n");
            goto cleanup;
        }
        if (!load_f32_raw(init_latent_path, frame0_img, frame0_elems, error,
                          sizeof(error))) {
            fprintf(stderr, "init-first-frame: %s\n", error);
            goto cleanup;
        }
        /* t>0 stays Gaussian. t=0 starts as 0.25*x_img + 0.75*noise (not
         * a one-shot 0.999/0.001 inject). Per-step mix runs after Euler. */
        img_w0 = apply_t0_img2img(latent, frame0_img, frame0_elems, 0, steps);
        printf("f2va: t=0 img2img (%zu elems); t>0 Gaussian; "
               "img_w 0.25→1 on unshifted u=1-i/steps (init img_w=%.3g); "
               "lock_first_frame=%d (ignored when img2img is on)\n",
               frame0_elems, (double)img_w0, lock_first_frame);
        if (lock_first_frame) {
            fprintf(stderr,
                    "warning: --lock-first-frame ignored under "
                    "--init-first-frame (100%% pin = standing pose). "
                    "Using per-step t=0 img2img instead.\n");
            lock_first_frame = 0;
        }
    } else if (init_latent_path) {
        if (!load_f32_raw(init_latent_path, latent, n_lat, error,
                          sizeof(error))) {
            fprintf(stderr, "init-latent: %s\n", error);
            goto cleanup;
        }
        printf("init_latent: loaded %zu f32 from %s (full T; standing-pose "
               "risk if the dump is a still encode)\n",
               n_lat, init_latent_path);
    }
    if (lock_first_frame) {
        frame0_lock = (float *)malloc(frame0_elems * sizeof(float));
        if (!frame0_lock) {
            fprintf(stderr, "OOM frame0 lock\n");
            goto cleanup;
        }
        memcpy(frame0_lock, latent, frame0_elems * sizeof(float));
        printf("g3_lite: lock_first_frame on (%zu elems)\n", frame0_elems);
    }
    if (official_kf) {
        const char *enc_path = vae_encode_path ? vae_encode_path : vae_path;
        const char *flag =
            official_keyframe_flag(first_frame_path, last_frame_path);
        const float *pic_rgb[2];
        size_t n_kf = 0, k, frame_rows, frame_n, row_elems, got;
        double t_enc;
        float last_condition_time;
        if (first_frame_path) pic_rgb[n_kf++] = ff_rgb;
        if (last_frame_path) pic_rgb[n_kf++] = lf_rgb;
        for (k = 0; k < n_kf; k++) {
            if (!pic_rgb[k]) {
                fprintf(stderr, "%s rgb missing (keyframe %zu)\n", flag, k);
                goto cleanup;
            }
        }
        frame_rows = (size_t)(latent_h / 2) * (size_t)(latent_w / 2);
        frame_n = frame_rows * (size_t)H3_DIT_W4A8_VIDEO_PATCH;
        cond_seq = n_kf * frame_rows;
        row_elems = cond_seq * (size_t)H3_DIT_W4A8_VIDEO_PATCH;
        cond_rows = (float *)malloc(row_elems * sizeof(float));
        cpt = (float *)malloc(cond_seq * sizeof(float));
        cph = (float *)malloc(cond_seq * sizeof(float));
        cpw = (float *)malloc(cond_seq * sizeof(float));
        if (!cond_rows || !cpt || !cph || !cpw) {
            fprintf(stderr, "OOM cond rows\n");
            goto cleanup;
        }
        t_enc = now_ms();
        if (cond_rows_path) {
            if (!load_f32_raw(cond_rows_path, cond_rows, row_elems, error,
                              sizeof(error))) {
                fprintf(stderr, "cond-rows: %s\n", error);
                goto cleanup;
            }
            printf("cond_rows: cache %s n=%zu keyframes=%zu checksum=0x%016llx "
                   "(skip encode+augment; official payload)\n",
                   cond_rows_path, row_elems, n_kf,
                   (unsigned long long)h3_dit_w4a8_checksum_f32(cond_rows,
                                                                row_elems));
        } else {
            if (!enc_path) {
                fprintf(stderr, "%s needs --vae-encode or --vae\n", flag);
                goto cleanup;
            }
            printf("vae_encode: %s keyframes=%zu (official COND; not "
                   "--init-latent)\n",
                   enc_path, n_kf);
            fflush(stdout);
            for (k = 0; k < n_kf; k++) {
                if (!official_encode_cond_rows(
                        enc_path, pic_rgb[k], height, width, frame_n,
                        cond_rows + k * frame_n, &got, latent_h, latent_w,
                        error, sizeof(error))) {
                    fprintf(stderr,
                            "vae encode keyframe %zu: %s\n"
                            "(INT8 decode ckpt often cannot encode; pass "
                            "--vae-encode fp16 encoder weights)\n",
                            k, error);
                    goto cleanup;
                }
                if (got != frame_n) {
                    fprintf(stderr,
                            "vae encode keyframe %zu rows %zu want %zu\n", k,
                            got, frame_n);
                    goto cleanup;
                }
            }
            /* Metal augments each keyframe span from the same seed. */
            if (!skip_dit) {
                for (k = 0; k < n_kf; k++)
                    h3_vae_augment_0p999(cond_rows + k * frame_n, frame_n,
                                         seed);
            }
            {
                FILE *cfp;
                snprintf(pathbuf, sizeof(pathbuf), "%s/cond_rows.raw", out_dir);
                cfp = fopen(pathbuf, "wb");
                if (cfp) {
                    fwrite(cond_rows, sizeof(float), row_elems, cfp);
                    fclose(cfp);
                }
                if (skip_dit) {
                    printf("cond_rows: encode no-augment checksum=0x%016llx "
                           "n=%zu keyframes=%zu (VAE roundtrip; wrote %s)\n",
                           (unsigned long long)h3_dit_w4a8_checksum_f32(
                               cond_rows, row_elems),
                           row_elems, n_kf, pathbuf);
                } else {
                    printf("cond_augment: 0.999*encode+0.001*N(seed=%llu) per "
                           "keyframe checksum=0x%016llx n=%zu (wrote %s)\n",
                           (unsigned long long)seed,
                           (unsigned long long)h3_dit_w4a8_checksum_f32(
                               cond_rows, row_elems),
                           row_elems, pathbuf);
                }
            }
        }
        k = 0;
        if (first_frame_path) {
            if (!h3_dit_w4a8_fill_metal_cond_pos((float)context, latent_h,
                                                 latent_w, cpt + k * frame_rows,
                                                 cph + k * frame_rows,
                                                 cpw + k * frame_rows, error,
                                                 sizeof(error))) {
                fprintf(stderr, "cond pos first: %s\n", error);
                goto cleanup;
            }
            k++;
        }
        last_condition_time =
            fl2va_last_condition_time(context, last_keyframe);
        if (last_frame_path) {
            if (!h3_dit_w4a8_fill_metal_cond_pos(
                    last_condition_time, latent_h, latent_w,
                    cpt + k * frame_rows, cph + k * frame_rows,
                    cpw + k * frame_rows, error, sizeof(error))) {
                fprintf(stderr, "cond pos last: %s\n", error);
                goto cleanup;
            }
        }
        if (official_kf == 3) {
            printf("cond_seq=%zu keyframes=0,last last_keyframe=%d "
                   "aligned_last=%d official_condition=1 "
                   "condition_time0=text_seq=%zu "
                   "condition_time1=text_seq+(mux_last)*(5/3)=%.6g "
                   "span_sum=%.6g metal_aligned_last_t=%.6g encode_ms=%.1f "
                   "pack=text|cond|cond|audio|video\n",
                   cond_seq, last_keyframe,
                   frames_aligned > 0 ? frames_aligned - 1 : 0, context,
                   (double)last_condition_time,
                   (double)h3_dit_w4a8_video_span_sum(latent_t),
                   (double)((float)context +
                            h3_dit_w4a8_video_span_sum(latent_t) -
                            (5.0f / 3.0f)),
                   now_ms() - t_enc);
        } else if (last_frame_path) {
            printf("cond_seq=%zu keyframes=%d official_condition=1 "
                   "condition_time=text_seq+(mux_last)*(5/3)=%.6g "
                   "text_seq=%zu span_sum=%.6g encode_ms=%.1f\n",
                   cond_seq, last_keyframe, (double)last_condition_time,
                   context, (double)h3_dit_w4a8_video_span_sum(latent_t),
                   now_ms() - t_enc);
        } else {
            printf("cond_seq=%zu keyframes=0 official_condition=1 "
                   "condition_time=text_seq=%zu encode_ms=%.1f\n",
                   cond_seq, context, now_ms() - t_enc);
        }
        fflush(stdout);
        if (skip_dit && cond_rows && cond_seq > 0) {
            int t;
            if (frame_n * (size_t)latent_t != n_lat) {
                fprintf(stderr,
                        "vae_roundtrip: cond frame %zu * T=%d != n_lat %zu\n",
                        frame_n, latent_t, n_lat);
                goto cleanup;
            }
            if (n_kf == 1) {
                for (t = 0; t < latent_t; t++)
                    memcpy(latent + (size_t)t * frame_n, cond_rows,
                           frame_n * sizeof(float));
                printf("vae_roundtrip: planted still cond_seq=%zu across "
                       "latent_t=%d (encode→decode skip DiT; not G3-lite)\n",
                       frame_rows, latent_t);
            } else {
                const float *first_rows = cond_rows;
                const float *last_rows = cond_rows + frame_n;
                for (t = 0; t < latent_t; t++) {
                    float *dst = latent + (size_t)t * frame_n;
                    if (t == 0) {
                        memcpy(dst, first_rows, frame_n * sizeof(float));
                    } else if (t == latent_t - 1) {
                        memcpy(dst, last_rows, frame_n * sizeof(float));
                    } else {
                        float u = (float)t / (float)(latent_t - 1);
                        size_t e;
                        for (e = 0; e < frame_n; e++)
                            dst[e] = first_rows[e] +
                                     (last_rows[e] - first_rows[e]) * u;
                    }
                }
                printf("vae_roundtrip: dual plant first@t0 last@t%d lerp "
                       "middle (NOT last-frame PSNR ceiling; use single "
                       "--last-frame skip-dit f0 vs COVER)\n",
                       latent_t - 1);
            }
            fflush(stdout);
        }
    }
    if (n_ref_images) {
        const char *enc_path = vae_encode_path ? vae_encode_path : vae_path;
        size_t ri, off_rows = 0, row_elems = 0, got;
        double t_enc;
        float cursor;
        for (ri = 0; ri < (size_t)n_ref_images; ri++) {
            size_t rows = (size_t)(ref_lh[ri] / 2) * (size_t)(ref_lw[ri] / 2);
            cond_seq += rows;
            row_elems += rows * (size_t)H3_DIT_W4A8_VIDEO_PATCH;
        }
        cond_rows = (float *)malloc(row_elems * sizeof(float));
        cpt = (float *)malloc(cond_seq * sizeof(float));
        cph = (float *)malloc(cond_seq * sizeof(float));
        cpw = (float *)malloc(cond_seq * sizeof(float));
        if (!cond_rows || !cpt || !cph || !cpw) {
            fprintf(stderr, "OOM ref cond rows\n");
            goto cleanup;
        }
        t_enc = now_ms();
        if (cond_rows_path) {
            if (!load_f32_raw(cond_rows_path, cond_rows, row_elems, error,
                              sizeof(error))) {
                fprintf(stderr, "cond-rows: %s\n", error);
                goto cleanup;
            }
            printf("cond_rows: cache %s n=%zu H3_SEG_REF_IMAGE n=%d "
                   "checksum=0x%016llx (skip encode+augment; official "
                   "Ref2VA payload)\n",
                   cond_rows_path, row_elems, n_ref_images,
                   (unsigned long long)h3_dit_w4a8_checksum_f32(cond_rows,
                                                                row_elems));
        } else {
            if (!enc_path) {
                fprintf(stderr, "--ref-image needs --vae-encode or --vae\n");
                goto cleanup;
            }
            printf("vae_encode: %s H3_SEG_REF_IMAGE n=%d (official REF; not "
                   "keyframe COND; not --init-latent)\n",
                   enc_path, n_ref_images);
            fflush(stdout);
            off_rows = 0;
            for (ri = 0; ri < (size_t)n_ref_images; ri++) {
                size_t rows =
                    (size_t)(ref_lh[ri] / 2) * (size_t)(ref_lw[ri] / 2);
                size_t n = rows * (size_t)H3_DIT_W4A8_VIDEO_PATCH;
                if (!official_encode_cond_rows(
                        enc_path, ref_rgb[ri], ref_ph[ri], ref_pw[ri], n,
                        cond_rows + off_rows * (size_t)H3_DIT_W4A8_VIDEO_PATCH,
                        &got, ref_lh[ri], ref_lw[ri], error, sizeof(error))) {
                    fprintf(stderr,
                            "vae encode ref-image %zu: %s\n"
                            "(INT8 decode ckpt often cannot encode; pass "
                            "--vae-encode fp16 encoder weights)\n",
                            ri + 1, error);
                    goto cleanup;
                }
                if (got != n) {
                    fprintf(stderr, "vae encode ref-image %zu rows %zu want %zu\n",
                            ri + 1, got, n);
                    goto cleanup;
                }
                off_rows += rows;
            }
            if (!skip_dit) {
                off_rows = 0;
                for (ri = 0; ri < (size_t)n_ref_images; ri++) {
                    size_t rows =
                        (size_t)(ref_lh[ri] / 2) * (size_t)(ref_lw[ri] / 2);
                    size_t n = rows * (size_t)H3_DIT_W4A8_VIDEO_PATCH;
                    h3_vae_augment_0p999(
                        cond_rows + off_rows * (size_t)H3_DIT_W4A8_VIDEO_PATCH,
                        n, seed);
                    off_rows += rows;
                }
            }
            {
                FILE *cfp;
                snprintf(pathbuf, sizeof(pathbuf), "%s/cond_rows.raw", out_dir);
                cfp = fopen(pathbuf, "wb");
                if (cfp) {
                    fwrite(cond_rows, sizeof(float), row_elems, cfp);
                    fclose(cfp);
                }
                printf("cond_augment: 0.999*encode+0.001*N(seed=%llu) per "
                       "H3_SEG_REF_IMAGE checksum=0x%016llx n=%zu (wrote %s)\n",
                       (unsigned long long)seed,
                       (unsigned long long)h3_dit_w4a8_checksum_f32(cond_rows,
                                                                    row_elems),
                       row_elems, pathbuf);
            }
        }
        off_rows = 0;
        cursor = (float)context;
        for (ri = 0; ri < (size_t)n_ref_images; ri++) {
            size_t rows = (size_t)(ref_lh[ri] / 2) * (size_t)(ref_lw[ri] / 2);
            if (!h3_dit_w4a8_fill_metal_ref_image_pos(
                    cursor, ref_lh[ri], ref_lw[ri], cpt + off_rows,
                    cph + off_rows, cpw + off_rows, error, sizeof(error))) {
                fprintf(stderr, "ref pos image %zu: %s\n", ri + 1, error);
                goto cleanup;
            }
            off_rows += rows;
            cursor += 1.0f;
        }
        printf("cond_seq=%zu official_reference=1 H3_SEG_REF_IMAGE n=%d "
               "cursor0=text_seq=%zu cursor_end=%.6g "
               "target_rope_t0=text_seq+n=%zu pack=text|ref_img|audio|video "
               "adaln=visual_condition encode_ms=%.1f\n",
               cond_seq, n_ref_images, context, (double)cursor,
               target_rope_t0, now_ms() - t_enc);
        fflush(stdout);
    }
    if (ref_video_path && ref_video_rgb) {
        const char *enc_path = vae_encode_path ? vae_encode_path : vae_path;
        size_t row_elems = 0;
        float *rows = NULL;
        float cursor;
        double t_enc = now_ms();
        float span;
        if (!enc_path) {
            fprintf(stderr, "--ref-video needs --vae-encode or --vae\n");
            goto cleanup;
        }
        printf("vae_encode: %s H3_SEG_REF_VIDEO→REF_IMAGE frames=%d "
               "(official REF; causal T_lat; not keyframe)\n",
               enc_path, ref_video_frames);
        fflush(stdout);
        if (cond_rows_path) {
            /* Caller must supply already-augmented rows matching this video. */
            FILE *fp = fopen(cond_rows_path, "rb");
            long sz;
            if (!fp) {
                fprintf(stderr, "cond-rows open: %s\n", cond_rows_path);
                goto cleanup;
            }
            fseek(fp, 0, SEEK_END);
            sz = ftell(fp);
            fseek(fp, 0, SEEK_SET);
            if (sz < 0 || (sz % (long)sizeof(float))) {
                fclose(fp);
                fprintf(stderr, "cond-rows bad size\n");
                goto cleanup;
            }
            row_elems = (size_t)sz / sizeof(float);
            rows = (float *)malloc(row_elems * sizeof(float));
            if (!rows || fread(rows, sizeof(float), row_elems, fp) != row_elems) {
                fclose(fp);
                free(rows);
                fprintf(stderr, "cond-rows read failed\n");
                goto cleanup;
            }
            fclose(fp);
            ref_vlt = (int)(row_elems / ((size_t)(ref_vlh / 2) *
                                         (size_t)(ref_vlw / 2) *
                                         (size_t)H3_DIT_W4A8_VIDEO_PATCH));
            if (ref_vlt < 1) {
                free(rows);
                fprintf(stderr, "cond-rows geometry mismatch for ref-video\n");
                goto cleanup;
            }
            printf("cond_rows: cache %s n=%zu H3_SEG_REF_VIDEO T≈%d "
                   "checksum=0x%016llx\n",
                   cond_rows_path, row_elems, ref_vlt,
                   (unsigned long long)h3_dit_w4a8_checksum_f32(rows, row_elems));
        } else if (!official_encode_ref_video_rows(
                       enc_path, ref_video_rgb, ref_video_frames, ref_vh, ref_vw,
                       &rows, &row_elems, &ref_vlt, &ref_vlh, &ref_vlw, error,
                       sizeof(error))) {
            fprintf(stderr, "vae encode ref-video: %s\n", error);
            goto cleanup;
        }
        cond_seq = row_elems / (size_t)H3_DIT_W4A8_VIDEO_PATCH;
        cond_rows = rows;
        cpt = (float *)malloc(cond_seq * sizeof(float));
        cph = (float *)malloc(cond_seq * sizeof(float));
        cpw = (float *)malloc(cond_seq * sizeof(float));
        if (!cpt || !cph || !cpw) {
            fprintf(stderr, "OOM ref-video cond pos\n");
            goto cleanup;
        }
        if (!skip_dit && !cond_rows_path)
            h3_vae_augment_0p999(cond_rows, row_elems, seed);
        {
            FILE *cfp;
            snprintf(pathbuf, sizeof(pathbuf), "%s/cond_rows.raw", out_dir);
            cfp = fopen(pathbuf, "wb");
            if (cfp) {
                fwrite(cond_rows, sizeof(float), row_elems, cfp);
                fclose(cfp);
            }
            printf("cond_augment: H3_SEG_REF_IMAGE (video) checksum=0x%016llx "
                   "n=%zu T=%d HxW=%dx%d (wrote %s)\n",
                   (unsigned long long)h3_dit_w4a8_checksum_f32(cond_rows,
                                                                row_elems),
                   row_elems, ref_vlt, ref_vlh, ref_vlw, pathbuf);
        }
        cursor = (float)context;
        if (!h3_dit_w4a8_fill_metal_ref_video_pos(cursor, ref_vlt, ref_vlh,
                                                  ref_vlw, cpt, cph, cpw, error,
                                                  sizeof(error))) {
            fprintf(stderr, "ref-video pos: %s\n", error);
            goto cleanup;
        }
        span = h3_dit_w4a8_video_span_sum(ref_vlt);
        cursor += span;
        printf("cond_seq=%zu official_reference=1 H3_SEG_REF_VIDEO→"
               "H3_SEG_REF_IMAGE T=%d span=%.6g cursor_end=%.6g "
               "encode_ms=%.1f\n",
               cond_seq, ref_vlt, (double)span, (double)cursor,
               now_ms() - t_enc);
        fflush(stdout);
        /* Advance target rope by video span; audio_cond may extend further. */
        target_rope_t0 = context + (size_t)llround((double)span);
        if (!ref_silent_video && n_ref_audios == 0 && audio_vae_path) {
            /* Embedded soundtrack → same H2 REF_AUDIO path. */
            ref_audio_paths[0] = ref_video_path;
            n_ref_audios = 1;
            printf("ref_video: including embedded audio → H3_SEG_REF_AUDIO "
                   "(use --ref-silent-video to skip)\n");
            fflush(stdout);
        }
    }
    if (n_ref_audios) {
        size_t ai, off_rows = 0, row_elems = 0;
        float cursor;
        size_t total_pcm = 0;
        int ref_audio_ts[H3_CUDA_MAX_REF_AUDIOS];
        double t_aenc = now_ms();
        for (ai = 0; ai < (size_t)n_ref_audios; ai++) {
            float *pcm = NULL;
            int samples = 0;
            h3_audio_vae_cuda_latent alat = {0};
            h3_audio_vae_cuda_stats aenc_st = {0};
            size_t rows;
            if (!h3_ffmpeg_read_audio_f32(ref_audio_paths[ai], 32000 * 15,
                                          ref_video_path &&
                                              !strcmp(ref_audio_paths[ai],
                                                      ref_video_path),
                                          &pcm, &samples, error,
                                          sizeof(error))) {
                fprintf(stderr, "ref-audio read %zu: %s\n", ai + 1, error);
                goto cleanup;
            }
            if (!(ref_video_path &&
                  !strcmp(ref_audio_paths[ai], ref_video_path)) &&
                samples < 32000 * 2) {
                fprintf(stderr,
                        "ref-audio %zu: clip must be at least 2s (got %d "
                        "samples)\n",
                        ai + 1, samples);
                free(pcm);
                goto cleanup;
            }
            if (samples < 1) {
                fprintf(stderr, "ref-audio %zu: empty pcm\n", ai + 1);
                free(pcm);
                goto cleanup;
            }
            if ((size_t)samples > (size_t)32000 * 15 - total_pcm) {
                fprintf(stderr,
                        "ref-audio: ordered clips exceed 15s total limit\n");
                free(pcm);
                goto cleanup;
            }
            total_pcm += (size_t)samples;
            if (!h3_audio_vae_cuda_encode(cuda, audio_vae_path, pcm, samples,
                                          NULL, NULL, &alat, &aenc_st, error,
                                          sizeof(error))) {
                fprintf(stderr, "ref-audio encode %zu: %s\n", ai + 1, error);
                free(pcm);
                h3_audio_vae_cuda_latent_free(&alat);
                goto cleanup;
            }
            free(pcm);
            if (alat.channels != 32 || alat.stereo != 2 || alat.length < 1) {
                fprintf(stderr, "ref-audio %zu: invalid latent geometry\n",
                        ai + 1);
                h3_audio_vae_cuda_latent_free(&alat);
                goto cleanup;
            }
            ref_audio_ts[ai] = alat.length;
            rows = (size_t)2 * (size_t)alat.length;
            {
                size_t prev_elems =
                    off_rows * (size_t)H3_DIT_W4A8_AUDIO_CH;
                size_t clip_elems =
                    rows * (size_t)H3_DIT_W4A8_AUDIO_CH;
                float *grown;
                audio_cond_seq += rows;
                row_elems = prev_elems + clip_elems;
                grown = (float *)realloc(audio_cond_rows,
                                         row_elems * sizeof(float));
                if (!grown) {
                    h3_audio_vae_cuda_latent_free(&alat);
                    fprintf(stderr, "OOM ref audio cond rows\n");
                    goto cleanup;
                }
                audio_cond_rows = grown;
                if (!h3_dit_w4a8_pack_audio_latent(
                        alat.values, alat.length,
                        audio_cond_rows + prev_elems, error, sizeof(error))) {
                    fprintf(stderr, "ref-audio pack %zu: %s\n", ai + 1, error);
                    h3_audio_vae_cuda_latent_free(&alat);
                    goto cleanup;
                }
            }
            off_rows += rows;
            ref_audio_t_sum += alat.length;
            h3_audio_vae_cuda_latent_free(&alat);
        }
        acpt = (float *)malloc(audio_cond_seq * sizeof(float));
        acph = (float *)malloc(audio_cond_seq * sizeof(float));
        acpw = (float *)malloc(audio_cond_seq * sizeof(float));
        if (!acpt || !acph || !acpw) {
            fprintf(stderr, "OOM ref audio cond pos\n");
            goto cleanup;
        }
        cursor = (float)context + (float)n_ref_images;
        off_rows = 0;
        for (ai = 0; ai < (size_t)n_ref_audios; ai++) {
            size_t rows = (size_t)2 * (size_t)ref_audio_ts[ai];
            if (!h3_dit_w4a8_fill_metal_ref_audio_pos(
                    cursor, ref_audio_ts[ai], latent_h, latent_w,
                    acpt + off_rows, acph + off_rows, acpw + off_rows, error,
                    sizeof(error))) {
                fprintf(stderr, "ref-audio pos %zu: %s\n", ai + 1, error);
                goto cleanup;
            }
            off_rows += rows;
            cursor += (float)ref_audio_ts[ai];
        }
        printf("audio_cond_seq=%zu official_reference=1 H3_SEG_REF_AUDIO "
               "n=%d latent_t_sum=%d pack=text|ref_img|ref_audio|audio|video "
               "encode_ms=%.1f checksum=0x%016llx\n",
               audio_cond_seq, n_ref_audios, ref_audio_t_sum,
               now_ms() - t_aenc,
               (unsigned long long)h3_dit_w4a8_checksum_f32(
                   audio_cond_rows, row_elems));
        fflush(stdout);
    }
    target_rope_t0 = context + (size_t)n_ref_images + (size_t)ref_audio_t_sum;
    if (ref_video_path && ref_vlt > 0) {
        size_t video_cursor =
            context + (size_t)llround((double)h3_dit_w4a8_video_span_sum(ref_vlt));
        size_t audio_cursor =
            context + (size_t)n_ref_images + (size_t)ref_audio_t_sum;
        target_rope_t0 =
            video_cursor > audio_cursor ? video_cursor : audio_cursor;
    }
    if (pack_audio) {
        if (!h3_dit_w4a8_fill_metal_audio_pos(target_rope_t0, audio_t, latent_h,
                                              latent_w, apt, aph, apw, error,
                                              sizeof(error))) {
            fprintf(stderr, "audio pos: %s\n", error);
            goto cleanup;
        }
    }
    build_serving_sigmas(steps, H3_CUDA_VIDEO_SIGMA_SHIFT, sigmas);
    audio_sigmas = (float *)malloc((size_t)(steps + 1) * sizeof(float));
    if (!audio_sigmas) {
        fprintf(stderr, "OOM audio sigmas\n");
        goto cleanup;
    }
    build_serving_sigmas(steps, H3_CUDA_AUDIO_SIGMA_SHIFT, audio_sigmas);
    printf("sigma[0..%d] video:", steps);
    for (step = 0; step <= steps; step++)
        printf(" %.4g", (double)sigmas[step]);
    printf("  audio:");
    for (step = 0; step <= steps; step++)
        printf(" %.4g", (double)audio_sigmas[step]);
    printf("\n");
    reuse_sel = (uint8_t *)calloc((size_t)steps, 1);
    vel_last = (float *)malloc(n_lat * sizeof(float));
    vel_prev = (float *)malloc(n_lat * sizeof(float));
    if (!reuse_sel || !vel_last || !vel_prev) {
        fprintf(stderr, "OOM reuse buffers\n");
        goto cleanup;
    }
    if (pack_audio) {
        avel_last = (float *)calloc(n_audio, sizeof(float));
        avel_prev = (float *)calloc(n_audio, sizeof(float));
        if (!avel_last || !avel_prev) {
            fprintf(stderr, "OOM audio reuse buffers\n");
            goto cleanup;
        }
        if (cfg_scale > 1.0001f) {
            audio_vel_uncond = (float *)calloc(n_audio, sizeof(float));
            if (!audio_vel_uncond) {
                fprintf(stderr, "OOM audio cfg buffers\n");
                goto cleanup;
            }
        }
    }
    {
        int nsel = h3_dit_w4a8_reuse_schedule(steps, effective_reuse, reuse_sel,
                                              (size_t)steps);
        if (nsel < 0) {
            fprintf(stderr, "bad reuse schedule\n");
            goto cleanup;
        }
        if (reuse_explicit || magcache_thresh > 0.0f ||
            h3_step_cache_enabled(&step_cache)) {
            const char *mc = getenv("H3_MAGCACHE_MAX_CONSEC");
            if (mc && mc[0]) {
                long v = strtol(mc, NULL, 10);
                if (v >= 1 && v <= 8) mag_max_consec = (int)v;
            }
            printf("reuse_schedule: interval=%d selected=%d/%d magcache_thresh=%.3g "
                   "max_consec=%d step_cache=%s thresh=%.3g max_consec=%d "
                   "av_split=%d (A/B only; default remains off)\n",
                   effective_reuse, nsel, steps, (double)magcache_thresh,
                   mag_max_consec, h3_step_cache_kind_name(step_cache.kind),
                   (double)step_cache.thresh, step_cache.max_consec,
                   pack_audio ? 1 : 0);
        } else {
            printf("reuse_schedule: disabled default_full_eval=1 magcache_thresh=%.3g "
                   "step_cache=off\n",
                   (double)magcache_thresh);
        }
    }
    if (!h3_dit_w4a8_fill_metal_video_pos(target_rope_t0, latent_t, latent_h,
                                          latent_w, pt, ph, pw, error,
                                          sizeof(error))) {
        fprintf(stderr, "video pos: %s\n", error);
        goto cleanup;
    }
    printf("rope_pos: metal_layout text_len=%zu target_rope_t0=%zu "
           "cond_seq=%zu audio_cond_seq=%zu video_seq=%zu audio_seq=%zu%s\n",
           context, target_rope_t0, cond_seq, audio_cond_seq, video_seq,
           pack_audio ? audio_seq : 0,
           (n_ref_images || ref_video_path) ? " (Ref2VA cursor)" : "");
    if (pack_audio)
        printf("adaln_dual_sigma: video_shift=%.3g audio_shift=%.3g "
               "(Metal/vLLM schedule_row_map)\n",
               (double)H3_CUDA_VIDEO_SIGMA_SHIFT,
               (double)H3_CUDA_AUDIO_SIGMA_SHIFT);

    /* cafe 5s: do not unpack DiT weights or JIT DiT SDPA during TE.
     * Cross-thread cuDNN graphs (p5a) made step1 slower; GPU prefetch
     * during TE stole VRAM/SM from encode. Multi-slot plans still keep
     * TE 576 and DiT 16k after encode. H3_DIT_TE_OVERLAP_BLOCKS opts in. */
    t0 = now_ms();
    if (!skip_dit && !skip_te && te_path) {
        if (latent_t >= 32)
            printf("dit_prefetch: skip TE-overlap GPU window (latent_t=%d; "
                   "leave VRAM for 24fps activations)\n",
                   latent_t);
        printf("te_open: %s\n", te_path);
        fflush(stdout);
        if (!h3_nvfp4_store_open(te_path, &te_store, error, sizeof(error))) {
            fprintf(stderr, "te open: %s\n", error);
            goto cleanup;
        }
        /* After TE packed H2D, disk is free: stage INT8 host while TE
         * layers run. Starting at te_open contended the 12.8GiB TE upload. */
        if (dit_is_int8 && dit) {
            const char *hs = getenv("H3_DIT_HOST_STAGE_TE");
            int want_hs = !(hs && hs[0] == '0');
            if (want_hs) {
                host_stage_job.dit = dit;
                host_stage_job.first = 0;
                host_stage_job.n = dit_layers;
                host_stage_job.ms = 0.0;
                host_stage_job.ok = 0;
                host_stage_job.err[0] = '\0';
                host_stage_kick_job.job = &host_stage_job;
                host_stage_kick_job.th = &host_stage_th;
                host_stage_kick_job.ok = &host_stage_th_ok;
                host_stage_kick_job.t0 = &host_stage_t0;
                h3_nvfp4_cuda_set_after_packed_hook(kick_dit_host_stage,
                                                    &host_stage_kick_job);
            }
        }
        if (!skip_dit && overlap && dit && latent_t < 32 && !host_stage_th_ok) {
            double ignore = 0.0;
            unsigned win = h3_w4a8_cache_window_size(h3_dit_w4a8_cache(dit));
            unsigned n_pref = dit_layers;
            const char *ov = getenv("H3_DIT_TE_OVERLAP_BLOCKS");
            if (ov && ov[0]) {
                long v = strtol(ov, NULL, 10);
                if (v >= 0 && v <= 50) n_pref = (unsigned)v;
            }
            if (win > 0 && win < n_pref) n_pref = win;
            if (n_pref >= 1) {
                (void)h3_dit_w4a8_prefetch(dit, 0, n_pref, &ignore, error,
                                           sizeof(error));
                (void)h3_cuda_event_record_default(cuda, &prefetch_done, error,
                                                   sizeof(error));
                printf("dit_prefetch: TE-overlap GPU window blocks=%u "
                       "latent_t=%d\n",
                       n_pref, latent_t);
                fflush(stdout);
            }
        }
        if (!token_ids) {
            token_ids = (int32_t *)malloc(context * sizeof(int32_t));
            if (!token_ids) goto cleanup;
            for (i = 0; i < context; i++)
                token_ids[i] = (int32_t)(1 + (mix64(seed + i) % 1000));
            printf("te_tokens: source=synth_random (pass --token-ids-file "
                   "/ --token-canary for real Qwen ids)\n");
        }
        printf("te_encode_begin: tokens=%zu layers=%u "
               "(H3_NVFP4_VERBOSE=1 for per-layer)\n",
               context, te_layers);
        fflush(stdout);
        {
            int te_pc_hit = 0;
            char te_pc_hex[32];
            te_pc_hex[0] = '\0';
            if (te_cache_dir && te_cache_dir[0] && !have_vision_span &&
                token_ids &&
                h3_te_cache_key(token_ids, context, te_layers, te_path,
                                te_pc_hex, sizeof(te_pc_hex), error,
                                sizeof(error))) {
                if (h3_te_cache_load(te_cache_dir, te_pc_hex, context, te_layers,
                                     &text_emb, error, sizeof(error))) {
                    te_pc_hit = 1;
                    te_st.embed_cache_hit = 1;
                    te_st.n_layers = te_layers;
                    te_st.all_finite = 1;
                    printf("te_prompt_cache: HIT dir=%s key=%s tokens=%zu\n",
                           te_cache_dir, te_pc_hex, context);
                    fflush(stdout);
                } else if (error[0] && strstr(error, "header mismatch")) {
                    fprintf(stderr, "te_prompt_cache: %s (recompute)\n", error);
                    error[0] = '\0';
                }
            }
            if (!te_pc_hit) {
                if (have_vision_span) {
                    const float *ds[3];
                    ds[0] = vision_ds[0];
                    ds[1] = vision_ds[1];
                    ds[2] = vision_ds[2];
                    text_emb = h3_nvfp4_encode_layers_multimodal_f32(
                        &te_store, token_ids, context, vision_spans,
                        (size_t)vision_span_count, vision_merged, ds, te_layers,
                        &te_st, error, sizeof(error));
                } else {
                    text_emb = h3_nvfp4_encode_layers_f32(
                        &te_store, token_ids, context, te_layers, &te_st, error,
                        sizeof(error));
                }
                if (text_emb && te_cache_dir && te_cache_dir[0] &&
                    !have_vision_span && token_ids && te_pc_hex[0]) {
                    if (!h3_te_cache_save(te_cache_dir, te_pc_hex, text_emb,
                                          context, te_layers, error,
                                          sizeof(error))) {
                        fprintf(stderr, "te_prompt_cache: save failed: %s\n",
                                error);
                        error[0] = '\0';
                    } else {
                        printf("te_prompt_cache: MISS wrote dir=%s key=%s\n",
                               te_cache_dir, te_pc_hex);
                    }
                }
            }
        }
        if (!text_emb) {
            fprintf(stderr, "TE encode: %s\n", error);
            goto cleanup;
        }
        printf("te_encode: layers=%u tokens=%zu embed_ms=%.1f layers_ms=%.1f "
               "total_ms=%.1f finite=%d cuda=%d embed_cache=%d\n",
               te_st.n_layers, context, te_st.embed_ms, te_st.layers_ms,
               te_st.total_ms, te_st.all_finite, te_st.used_cuda,
               te_st.embed_cache_hit);
        fflush(stdout);
        if (dump_dir && dump_dir[0]) {
            if (!ensure_dir(dump_dir)) {
                fprintf(stderr, "H3_DUMP_DIR mkdir failed: %s\n", dump_dir);
                goto cleanup;
            }
            dump_f32_raw(dump_dir, "c_te_raw.f32", text_emb,
                         context * (size_t)H3_DIT_W4A8_TEXT_DIM);
        }
    } else if (!skip_dit) {
        text_emb = h3_nvfp4_synth_condition_f32(context, seed, error,
                                                sizeof(error));
        printf("te: synth_stub (skip-te or no --te)\n");
    }
    t_te = now_ms() - t0;
    if (!skip_dit) {
    if (!text_emb) {
        fprintf(stderr, "condition failed: %s\n", error);
        goto cleanup;
    }
    if (!h3_dit_w4a8_refine_text_f32(dit, &core, text_emb, context, text_h,
                                     error, sizeof(error))) {
        fprintf(stderr, "token_refiner/condition_proj: %s\n", error);
        goto cleanup;
    }
    if (dump_dir && dump_dir[0]) {
        dump_f32_raw(dump_dir, "c_te_refined.f32", text_h,
                     context * (size_t)H3_DIT_W4A8_HIDDEN);
        dump_f32_raw(dump_dir, "c_latent0.f32", latent, n_lat);
        if (pack_audio && audio_c2t && audio_rows) {
            dump_f32_raw(dump_dir, "c_audio0_c2t.f32", audio_c2t,
                         (size_t)H3_DIT_W4A8_AUDIO_CH * 2 * (size_t)audio_t);
            dump_f32_raw(dump_dir, "c_audio0_rows.f32", audio_rows, n_audio);
        }
        dump_meta_txt(dump_dir, context, latent_t, latent_h, latent_w, audio_t,
                      steps, seed, sigmas[0],
                      audio_sigmas ? audio_sigmas[0] : 0.0f, video_seq,
                      pack_audio ? audio_seq : 0, n_lat);
    }
    /* Debug/ablation: scale refined text rows (default 1). Weak TE→DiT
     * coupling can leave the sampler in the dialogue prior; gain>1 amplifies
     * prompt deltas before AdaLN/attn. */
    {
        const char *tg = getenv("H3_DIT_TEXT_GAIN");
        float gain = tg ? (float)strtod(tg, NULL) : 1.0f;
        if (isfinite(gain) && fabsf(gain - 1.0f) > 1e-6f && gain > 0.0f) {
            size_t n = context * (size_t)H3_DIT_W4A8_HIDDEN;
            for (i = 0; i < n; i++) text_h[i] *= gain;
            printf("text_condition: H3_DIT_TEXT_GAIN=%.3g applied to refined "
                   "text_h\n",
                   (double)gain);
        }
    }
    printf("text_condition: refiner=%d tokens=%zu text_mod_row=%s "
           "(refiner QKV follows H3_DIT_QKV_GROUPED; default contiguous) "
           "text_h checksum=0x%016llx n=%zu\n",
           core.has_refiner && !getenv("H3_DIT_NO_REFINER") ? 1 : 0, context,
           getenv("H3_DIT_TEXT_MOD_ROW") ? getenv("H3_DIT_TEXT_MOD_ROW")
                                        : "1(default_TEXT)",
           (unsigned long long)h3_dit_w4a8_checksum_f32(
               text_h, context * (size_t)H3_DIT_W4A8_HIDDEN),
           context * (size_t)H3_DIT_W4A8_HIDDEN);
    {
        const char *gq = getenv("H3_DIT_QKV_GROUPED");
        const char *half_off = getenv("H3_DIT_ROPE_HALF_OFFSET");
        const char *rs = getenv("H3_DIT_ROPE_SPLIT_HALF");
        int grouped = gq && gq[0] == '1';
        int split_half = 1;
        if (half_off && half_off[0] == '1')
            split_half = 0;
        else if (rs && rs[0] == '0')
            split_half = 0;
        else if (rs && rs[0] == '1')
            split_half = 1;
        printf("qkv_rope: layout=%s rope=%s adaln_silu=%s\n",
               grouped ? "interleaved_per_head" : "contiguous_QKV",
               split_half ? "split_half(Comfy/Kijai)" : "half_offset(Metal)",
               (getenv("H3_DIT_ADALN_SILU") &&
                getenv("H3_DIT_ADALN_SILU")[0] == '1')
                   ? "on"
                   : "off(curve)");
    }

    /* CFG uncond: Comfy empty prompt is pad-token TE → condition_proj/refiner.
     * Raw zero post-refiner hidden is OOD and can amplify the dialogue prior.
     * Official F2VA: keep Picture+vision inject; pad only the prompt ids so
     * CFG does not 5×-amplify the vision tower (Metal has no CFG). */
    if (cfg_scale > 1.0001f && text_h_uncond && !skip_te && te_path) {
        int32_t *pad_ids = (int32_t *)malloc(context * sizeof(int32_t));
        float *pad_emb = NULL;
        size_t pi, prompt_start = 0;
        int keep_vision = have_vision_span && vision_merged &&
                          vision_ds[0] && vision_ds[1] && vision_ds[2];
        if (!pad_ids) {
            fprintf(stderr, "OOM cfg pad ids\n");
            goto cleanup;
        }
        if (keep_vision) {
            const h3_cuda_vision_span *last_span =
                &vision_spans[vision_span_count - 1];
            prompt_start = last_span->start + last_span->tokens;
            if (prompt_start > context) {
                free(pad_ids);
                fprintf(stderr, "cfg uncond prompt_start overflow\n");
                goto cleanup;
            }
            memcpy(pad_ids, token_ids, context * sizeof(int32_t));
            for (pi = prompt_start; pi < context; pi++)
                pad_ids[pi] = 151643; /* Qwen <|endoftext|> / pad */
            {
                const float *ds[3];
                ds[0] = vision_ds[0];
                ds[1] = vision_ds[1];
                ds[2] = vision_ds[2];
                pad_emb = h3_nvfp4_encode_layers_multimodal_f32(
                    &te_store, pad_ids, context, vision_spans,
                    (size_t)vision_span_count, vision_merged, ds, te_layers,
                    &te_st, error, sizeof(error));
            }
        } else {
            for (pi = 0; pi < context; pi++)
                pad_ids[pi] = 151643;
            pad_emb = h3_nvfp4_encode_layers_f32(&te_store, pad_ids, context,
                                                 te_layers, &te_st, error,
                                                 sizeof(error));
        }
        free(pad_ids);
        if (!pad_emb) {
            fprintf(stderr, "TE uncond(pad) encode: %s\n", error);
            goto cleanup;
        }
        if (!h3_dit_w4a8_refine_text_f32(dit, &core, pad_emb, context,
                                         text_h_uncond, error,
                                         sizeof(error))) {
            free(pad_emb);
            fprintf(stderr, "token_refiner uncond: %s\n", error);
            goto cleanup;
        }
        free(pad_emb);
        printf("cfg_uncond: te_pad_tokens=%zu encode_ms=%.1f cuda=%d "
               "embed_cache=%d vision_inject=%d prompt_start=%zu "
               "(pad prompt only; DiT cond_seq=%zu stays in both forwards)\n",
               context, te_st.total_ms, te_st.used_cuda, te_st.embed_cache_hit,
               keep_vision ? 1 : 0, prompt_start, cond_seq);
    }
    free(vision_merged);
    vision_merged = NULL;
    free(vision_ds[0]);
    free(vision_ds[1]);
    free(vision_ds[2]);
    vision_ds[0] = vision_ds[1] = vision_ds[2] = NULL;
    if (have_vision_span && dit && !dit_text_tags) {
        printf("dit_text_tags: off (CUDA INT8 official default: "
               "Picture+vision stay TEXT AdaLN, not VIDEO@σ; not Metal "
               "bit-match. --dit-text-tags restores Metal-like tagging)\n");
    }
    if (have_vision_span && dit && dit_text_tags) {
        text_mod_tags = (uint8_t *)malloc(context);
        if (!text_mod_tags) {
            fprintf(stderr, "OOM DiT text tags\n");
            goto cleanup;
        }
        if (!h3_multimodal_cuda_dit_text_tags(context, vision_spans,
                                              (size_t)vision_span_count,
                                              text_mod_tags, error,
                                              sizeof(error))) {
            fprintf(stderr, "dit text tags: %s\n", error);
            goto cleanup;
        }
        h3_dit_w4a8_set_text_mod_tags(dit, text_mod_tags, context);
        {
            size_t video_n = 0;
            for (i = 0; i < context; i++)
                if (text_mod_tags[i] == 0) video_n++;
            printf("dit_text_tags: video_tagged=%zu/%zu (debug Metal-like "
                   "tag=0 on START+pads+END; AdaLN VIDEO@σ. CUDA INT8 8-step "
                   "quality path defaults this OFF)\n",
                   video_n, context);
        }
    }

    /* Official / longer prompts need larger --context; free TE VRAM before
     * CFG×2 DiT so 24GB can keep the weight window. text_h stays on host. */
    free(text_emb);
    text_emb = NULL;
    if (!skip_te && te_path) {
        h3_nvfp4_store_free(&te_store);
        h3_nvfp4_cuda_disable();
        printf("released TE after encode (before DiT; context=%zu)\n",
               context);
    }
    if (host_stage_th_ok) {
        pthread_join(host_stage_th, NULL);
        host_stage_th_ok = 0;
        printf("dit_host_stage: done ok=%d wall_ms=%.1f stage_ms=%.1f "
               "blocks=%u%s%s\n",
               host_stage_job.ok, now_ms() - host_stage_t0, host_stage_job.ms,
               host_stage_job.n, host_stage_job.ok ? "" : " err=",
               host_stage_job.ok
                   ? ""
                   : (host_stage_job.err[0] ? host_stage_job.err : "unknown"));
        fflush(stdout);
    }

    if (prefetch_done) {
        void *st0 = h3_cuda_stream(cuda, 0);
        if (st0)
            (void)h3_cuda_stream_wait_event(st0, prefetch_done, error,
                                            sizeof(error));
        /* event ownership: destroy via cudaEventDestroy through cast */
        cudaEventDestroy((cudaEvent_t)prefetch_done);
        prefetch_done = NULL;
    }

    /* After TE: JIT the DiT cuDNN plan on the owner handle so step1's
     * first SDPA is a hit. Worker-thread JIT (p5) bound the graph to the
     * wrong handle and slowed step1 — do not bring that back.
     * Unpack stays inside step1 (full window after TE OOMs the later WS).
     * H3_DIT_PLAN_REUSE=0 disables. */
    if (!skip_dit && dit && cuda) {
        const char *pr = getenv("H3_DIT_PLAN_REUSE");
        int want_reuse = !(pr && pr[0] == '0');
        if (want_reuse) {
            size_t packed_seq = context + cond_seq + audio_cond_seq +
                                video_seq + (pack_audio ? audio_seq : 0);
            float attn_scale = 1.0f / sqrtf((float)H3_DIT_W4A8_HEAD_DIM);
            double t_prep = now_ms();
            char pb_err[256];
            int pb_ok;
            pb_err[0] = '\0';
            {
                size_t pb_heads = (size_t)H3_DIT_W4A8_HEADS;
                int pbi;
                if (ulysses > 1)
                    pb_heads /= (size_t)ulysses;
                pb_ok = h3_cuda_sdpa_prebuild(
                    cuda, packed_seq, pb_heads,
                    (size_t)H3_DIT_W4A8_HEAD_DIM, attn_scale, 0, pb_err,
                    sizeof(pb_err));
                printf("dit_plan_reuse: after_te same_handle prebuild=%d "
                       "pb_ms=%.1f seq=%zu heads=%zu d=%d causal=0 ulysses=%d "
                       "(H3_DIT_PLAN_REUSE=0 disables)\n",
                       pb_ok, now_ms() - t_prep, packed_seq, pb_heads,
                       H3_DIT_W4A8_HEAD_DIM, ulysses);
                if (!pb_ok)
                    printf("dit_plan_reuse: prebuild_err=%s\n",
                           pb_err[0] ? pb_err : "unknown");
                for (pbi = 1; pbi < ulysses && pbi < 8; pbi++) {
                    if (!uly_cuda[pbi])
                        continue;
                    pb_err[0] = '\0';
                    if (!h3_cuda_sdpa_prebuild(
                            uly_cuda[pbi], packed_seq, pb_heads,
                            (size_t)H3_DIT_W4A8_HEAD_DIM, attn_scale, 0,
                            pb_err, sizeof(pb_err)))
                        printf("dit_plan_reuse: rank%d prebuild_err=%s\n", pbi,
                               pb_err[0] ? pb_err : "unknown");
                }
            }
            fflush(stdout);
        }
    }

    t0 = now_ms();
    for (step = 0; step < steps; step++) {
        h3_dit_w4a8_forward_stats fst;
        int evaluate = reuse_sel[step] ? 1 : 0;
        memset(&fst, 0, sizeof(fst));

        /* MagCache (reuse==1 only): skip middle while accum < thresh AND
         * consecutive skips < mag_max_consec (H3 Safe default 2). First/last
         * step always eval. Audio |ratio-1| can veto a video-only hit. */
        if (effective_reuse == 1 && magcache_thresh > 0.0f && step > 0 &&
            step < steps - 1 &&             mag_have_probe && mag_accum < magcache_thresh &&
            mag_consec < mag_max_consec && mag_fresh)
            evaluate = 0;
        /* Independent Tea / FB (h3_step_cache): mutex vs mag already enforced. */
        if (h3_step_cache_enabled(&step_cache) &&
            !h3_step_cache_should_eval(&step_cache, step, steps))
            evaluate = 0;
        if (step == 0 || step == steps - 1) evaluate = 1;

        h3_dit_w4a8_set_denoise_step(dit, step);

        if (evaluate) {
            if (pack_audio) {
                if (!h3_dit_w4a8_forward_avc_f32(
                        dit, &core, 0, dit_layers, sigmas[step],
                        audio_sigmas[step], latent, video_seq, audio_rows,
                        audio_seq, cond_seq ? cond_rows : NULL, cond_seq,
                        audio_cond_seq ? audio_cond_rows : NULL, audio_cond_seq,
                        text_h, context, pt, ph, pw, apt, aph, apw,
                        cond_seq ? cpt : NULL, cond_seq ? cph : NULL,
                        cond_seq ? cpw : NULL,
                        audio_cond_seq ? acpt : NULL,
                        audio_cond_seq ? acph : NULL,
                        audio_cond_seq ? acpw : NULL, 1.0f, velocity,
                        audio_vel, 1, &fst, error, sizeof(error))) {
                    fprintf(stderr, "dit AV forward step %d: %s\n", step,
                            error);
                    goto cleanup;
                }
            } else if (!h3_dit_w4a8_forward_video_f32(
                           dit, &core, 0, dit_layers, sigmas[step], latent,
                           video_seq, text_h, context, pt, ph, pw, 1.0f,
                           velocity, 1, &fst, error, sizeof(error))) {
                fprintf(stderr, "dit forward step %d: %s\n", step, error);
                goto cleanup;
            }
            if (step == 0)
                printf("adaln_cond: step0 sigma=%.4g active=%d mod_row=%u "
                       "t_row=%zu cond_seq=%zu (want active=1 mod_row=3 for "
                       "F2VA σ=1)\n",
                       (double)sigmas[step], fst.visual_cond_t_active,
                       fst.visual_cond_mod_row, fst.visual_cond_t_row,
                       cond_seq);
            if (cfg_scale > 1.0001f && text_h_uncond && velocity_uncond) {
                h3_dit_w4a8_forward_stats fst_u;
                double ignore = 0.0;
                memset(&fst_u, 0, sizeof(fst_u));
                if (pack_audio) {
                    if (!h3_dit_w4a8_forward_avc_f32(
                            dit, &core, 0, dit_layers, sigmas[step],
                            audio_sigmas[step], latent, video_seq, audio_rows,
                            audio_seq, cond_seq ? cond_rows : NULL, cond_seq,
                            audio_cond_seq ? audio_cond_rows : NULL,
                            audio_cond_seq, text_h_uncond, context, pt, ph, pw,
                            apt, aph, apw, cond_seq ? cpt : NULL,
                            cond_seq ? cph : NULL, cond_seq ? cpw : NULL,
                            audio_cond_seq ? acpt : NULL,
                            audio_cond_seq ? acph : NULL,
                            audio_cond_seq ? acpw : NULL, 1.0f, velocity_uncond,
                            audio_vel_uncond, 1, &fst_u, error,
                            sizeof(error))) {
                        fprintf(stderr, "dit AV uncond step %d: %s\n", step,
                                error);
                        goto cleanup;
                    }
                } else if (!h3_dit_w4a8_forward_video_f32(
                               dit, &core, 0, dit_layers, sigmas[step], latent,
                               video_seq, text_h_uncond, context, pt, ph, pw,
                               1.0f, velocity_uncond, 1, &fst_u, error,
                               sizeof(error))) {
                    fprintf(stderr, "dit uncond step %d: %s\n", step, error);
                    goto cleanup;
                }
                for (i = 0; i < n_lat; i++)
                    velocity[i] =
                        velocity_uncond[i] +
                        cfg_scale * (velocity[i] - velocity_uncond[i]);
                if (pack_audio && audio_vel_uncond) {
                    for (i = 0; i < n_audio; i++)
                        audio_vel[i] =
                            audio_vel_uncond[i] +
                            cfg_scale * (audio_vel[i] - audio_vel_uncond[i]);
                }
                ignore = fst_u.total_ms;
                fst.total_ms += ignore;
                printf("  cfg: scale=%.3g uncond_ms=%.1f\n", (double)cfg_scale,
                       ignore);
            }
            if (step == 0 && dump_dir && dump_dir[0]) {
                dump_f32_raw(dump_dir, "c_velocity0.f32", velocity, n_lat);
                if (pack_audio && audio_vel)
                    dump_f32_raw(dump_dir, "c_audio_velocity0.f32", audio_vel,
                                 n_audio);
                printf("dump: first-step velocity written (sigma=%.6g)\n",
                       (double)sigmas[step]);
                if (dump_only) {
                    printf("H3_DUMP_ONLY=1: stop after step0 forward "
                           "(before euler)\n");
                    t_dit = now_ms() - t0;
                    printf("dit_schedule: dump_only evals=1 dit_ms=%.1f\n",
                           t_dit);
                    rc = 0;
                    goto cleanup;
                }
            }
            if (last_eval >= 0 && effective_reuse == 1 &&
                magcache_thresh > 0.0f) {
                float delta;
                if (magcache_calibrated) {
                    /* Calibrated MagCache: x = |mean_abs ratio - 1|,
                     * poly ≈ 0.1x^3+0.2x^2+0.7x (milder than raw x→more skips). */
                    float ma =
                        h3_dit_w4a8_velocity_mean_abs_f32(velocity, n_lat);
                    float mp =
                        h3_dit_w4a8_velocity_mean_abs_f32(vel_last, n_lat);
                    float ratio = mp > 1e-12f ? ma / mp : 1.0f;
                    float x = fabsf(ratio - 1.0f);
                    if (pack_audio && audio_vel && avel_last) {
                        float aa = h3_dit_w4a8_velocity_mean_abs_f32(audio_vel,
                                                                     n_audio);
                        float ap = h3_dit_w4a8_velocity_mean_abs_f32(avel_last,
                                                                     n_audio);
                        float ar = ap > 1e-12f ? aa / ap : 1.0f;
                        float xa = fabsf(ar - 1.0f);
                        if (xa > x) x = xa;
                    }
                    delta = ((0.10f * x + 0.20f) * x + 0.70f) * x;
                    printf("  magcache_cal: |ratio-1|=%.4g poly=%.4g ",
                           (double)x, (double)delta);
                } else {
                    float ma =
                        h3_dit_w4a8_velocity_mean_abs_f32(velocity, n_lat);
                    float mp =
                        h3_dit_w4a8_velocity_mean_abs_f32(vel_last, n_lat);
                    float ratio = mp > 1e-12f ? ma / mp : 1.0f;
                    delta = fabsf(ratio - 1.0f);
                    if (pack_audio && audio_vel && avel_last) {
                        float aa = h3_dit_w4a8_velocity_mean_abs_f32(audio_vel,
                                                                     n_audio);
                        float ap = h3_dit_w4a8_velocity_mean_abs_f32(avel_last,
                                                                     n_audio);
                        float ar = ap > 1e-12f ? aa / ap : 1.0f;
                        float da = fabsf(ar - 1.0f);
                        if (da > delta) delta = da;
                    }
                    printf("  magcache: |ratio-1|=%.4g ", (double)delta);
                }
                mag_accum += delta;
                mag_last_delta = delta;
                mag_have_probe = 1;
                mag_fresh = mag_accum < magcache_thresh ? 1 : 0;
                printf("accum=%.4g thresh=%.4g → %s\n", (double)mag_accum,
                       (double)magcache_thresh,
                       mag_fresh ? "allow_skip" : "force_eval_reset");
                if (!mag_fresh) mag_accum = 0.0f;
            }
            if (last_eval >= 0 && h3_step_cache_enabled(&step_cache)) {
                float ma =
                    h3_dit_w4a8_velocity_mean_abs_f32(velocity, n_lat);
                float mp =
                    h3_dit_w4a8_velocity_mean_abs_f32(vel_last, n_lat);
                float aa = 0.0f, ap = 0.0f;
                int have_a = 0;
                float delta;
                if (pack_audio && audio_vel && avel_last) {
                    aa = h3_dit_w4a8_velocity_mean_abs_f32(audio_vel, n_audio);
                    ap = h3_dit_w4a8_velocity_mean_abs_f32(avel_last, n_audio);
                    have_a = 1;
                }
                delta = h3_step_cache_delta_from_velocities(
                    step_cache.kind, ma, mp, have_a, aa, ap);
                h3_step_cache_on_eval(&step_cache, delta);
                printf("  %s: delta=%.4g accum=%.4g last=%.4g thresh=%.4g "
                       "consec=%d → %s\n",
                       h3_step_cache_kind_name(step_cache.kind), (double)delta,
                       (double)step_cache.accum, (double)step_cache.last_delta,
                       (double)step_cache.thresh, step_cache.consec,
                       step_cache.fresh ? "allow_skip" : "force_eval");
            }
            if (last_eval >= 0) {
                memcpy(vel_prev, vel_last, n_lat * sizeof(float));
                if (pack_audio && avel_prev && avel_last)
                    memcpy(avel_prev, avel_last, n_audio * sizeof(float));
                prev_eval = last_eval;
            }
            memcpy(vel_last, velocity, n_lat * sizeof(float));
            if (pack_audio && avel_last)
                memcpy(avel_last, audio_vel, n_audio * sizeof(float));
            last_eval = step;
            mag_consec = 0;
            dit_evals++;
            printf("dit_step %d/%d: EVAL patch=%.1f layers=%.1f final=%.1f "
                   "total=%.1f\n",
                   step + 1, steps, fst.patch_ms, fst.layers_ms, fst.final_ms,
                   fst.total_ms);
            h3_cuda_sol_profile_finish_step(step, fst.total_ms);
        } else {
            if (effective_reuse == 1 && magcache_thresh > 0.0f &&
                mag_have_probe) {
                mag_accum += mag_last_delta;
                mag_consec++;
                printf("  magcache: skip grow accum=%.4g (+%.4g) consec=%d/%d\n",
                       (double)mag_accum, (double)mag_last_delta, mag_consec,
                       mag_max_consec);
            }
            if (h3_step_cache_enabled(&step_cache)) {
                h3_step_cache_on_skip(&step_cache);
                printf("  %s: skip grow accum=%.4g (+%.4g) consec=%d/%d\n",
                       h3_step_cache_kind_name(step_cache.kind),
                       (double)step_cache.accum, (double)step_cache.last_delta,
                       step_cache.consec, step_cache.max_consec);
            }
            if (last_eval < 0) {
                fprintf(stderr, "reuse skip before first eval\n");
                goto cleanup;
            }
            h3_dit_w4a8_extrapolate_velocity_f32(
                velocity, vel_last, prev_eval >= 0 ? vel_prev : NULL, n_lat,
                sigmas[step], sigmas[last_eval],
                prev_eval >= 0 ? sigmas[prev_eval] : 0.0f, prev_eval >= 0);
            if (pack_audio) {
                h3_dit_w4a8_extrapolate_velocity_f32(
                    audio_vel, avel_last, prev_eval >= 0 ? avel_prev : NULL,
                    n_audio, audio_sigmas[step], audio_sigmas[last_eval],
                    prev_eval >= 0 ? audio_sigmas[prev_eval] : 0.0f,
                    prev_eval >= 0);
            }
            dit_skips++;
            printf("dit_step %d/%d: SKIP (extrapolate from %d)\n", step + 1,
                   steps, last_eval);
        }

        /* Sampler contract (Comfy FLOW+CONST ≡ Metal euler):
         * DiT emits raw video_out=v. Comfy wraps model_output=-v; CONST makes
         * denoised=X+σ·v; ODE dX/dσ=-v ⇒ X += (σ-σ')·v. A prior bug treated
         * -v as x0 directly (black/olive). Always use velocity euler here;
         * dual audio_sigmas already encode the audio shift (no extra slope). */
        {
            if (pack_audio) {
                if (!h3_dit_w4a8_euler_step_f32(
                        audio_rows, audio_vel, n_audio, audio_sigmas[step],
                        audio_sigmas[step + 1], error, sizeof(error))) {
                    fprintf(stderr, "audio euler: %s\n", error);
                    goto cleanup;
                }
            }
            if (!h3_dit_w4a8_euler_step_f32(
                    latent, velocity, n_lat, sigmas[step],
                    sigmas[step + 1], error, sizeof(error))) {
                fprintf(stderr, "euler: %s\n", error);
                goto cleanup;
            }
            if (step == 0)
                printf("sampler: velocity_euler (X += (σ-σ')·v; "
                       "Comfy CONST-equivalent)\n");
        }
        if (frame0_img && frame0_elems) {
            float img_w = apply_t0_img2img(latent, frame0_img, frame0_elems,
                                           step + 1, steps);
            printf("f2va_t0_img2img: after_step=%d/%d img_w=%.3g "
                   "(u=%.3g unshifted; σ=%.4g→%.4g)\n",
                   step + 1, steps, (double)img_w,
                   (double)(1.0f - (float)(step + 1) / (float)steps),
                   (double)sigmas[step], (double)sigmas[step + 1]);
        } else if (frame0_lock)
            memcpy(latent, frame0_lock, frame0_elems * sizeof(float));
        fflush(stdout);
    }
    t_dit = now_ms() - t0;
    printf("dit_schedule: evals=%d skips=%d dit_ms=%.1f\n", dit_evals,
           dit_skips, t_dit);
    h3_cuda_sol_cute_dump_stats();
    h3_cuda_sage3_dump_stats();
    } else {
        printf("dit_schedule: skipped (decode-only)\n");
    }
    {
        double s = 0.0, s2 = 0.0;
        float mn = latent[0], mx = latent[0];
        for (i = 0; i < n_lat; i++) {
            float v = latent[i];
            s += v;
            s2 += (double)v * (double)v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        {
            double mean = s / (double)n_lat;
            double var = s2 / (double)n_lat - mean * mean;
            printf("latent_stats: mean=%.4g std=%.4g min=%.4g max=%.4g "
                   "n=%zu checksum=0x%016llx\n",
                   mean, var > 0.0 ? sqrt(var) : 0.0, (double)mn, (double)mx,
                   n_lat,
                   (unsigned long long)h3_dit_w4a8_checksum_f32(latent, n_lat));
        }
    }

    snprintf(pathbuf, sizeof(pathbuf), "%s/latent.raw", out_dir);
    {
        FILE *fp = fopen(pathbuf, "wb");
        if (fp) {
            fwrite(latent, sizeof(float), n_lat, fp);
            fclose(fp);
            printf("wrote %s (%zu floats)\n", pathbuf, n_lat);
        }
    }

    /* Drop DiT/TE resident weights before VAE so 50L + decode fits in 24GB. */
    h3_dit_w4a8_core_free(&core);
    gen_uly_free_ranks(dit, uly_dit, uly_cuda);
    h3_dit_w4a8_free(dit);
    dit = NULL;
    if (lora_loaded) {
        h3_lora_free(&lora_set);
        lora_loaded = 0;
    }
    h3_nvfp4_store_free(&te_store);
    h3_nvfp4_cuda_disable();
    h3_w4a8_store_free(dit_store);
    dit_store = NULL;
    h3_dit_int8_store_free(dit_int8);
    dit_int8 = NULL;
    /* Keep the CUDA ctx (cuDNN handle / NVRTC / plans). Trim DiT-sized
     * scratch so 50L decode fits; do not rebuild the device. */
    h3_cuda_release_heavy_workspace(cuda);
    printf("released DiT/TE before VAE (kept CUDA ctx; trimmed DiT scratch)\n");
    fflush(stdout);
    /* Full-frame decode is faster but leaves a 16px VAE-patch waffle
     * (b16≈2.6 vs ≈1.04 with 256px tiles + deblock16). Default back to
     * Metal-style tiles now that weights are loaded once. Opt in to the
     * speed path: H3_VAE_NO_SPATIAL_TILE=1. */
    if (!getenv("H3_VAE_NO_SPATIAL_TILE") && !getenv("H3_VAE_TILE_PIXELS")) {
        printf("vae: spatial tiles (default 256px + deblock16; "
               "H3_VAE_NO_SPATIAL_TILE=1 forces full-frame)\n");
        fflush(stdout);
    }

    if (vae_th_ok) {
        pthread_join(vae_th, NULL);
        vae_th_ok = 0;
    }
    if (!skip_vae) {
        if (!h3_vae_int8_store_open(vae_path, &vae_store, error,
                                    sizeof(error))) {
            fprintf(stderr, "vae open: %s\n", error);
            goto cleanup;
        }
        {
            const h3_st_tensor *vw = h3_st_find(
                &vae_store.header,
                "decoder.transformer_blocks.0.attn.to_qkv.weight");
            const char *kind = "unknown";
            if (vw) {
                if (vw->dtype == H3_DTYPE_I8) kind = "INT8+scale";
                else if (vw->dtype == H3_DTYPE_F16) kind = "F16 native";
                else if (vw->dtype == H3_DTYPE_BF16) kind = "BF16 native";
                else if (vw->dtype == H3_DTYPE_F32) kind = "F32 native";
            }
            printf("vae_ckpt: %s convrot=%d (%s)\n", kind,
                   vae_store.quant.convrot,
                   vw && vw->dtype == H3_DTYPE_F16
                       ? "quality decode: fp16 W → F32 GEMM + tile/deblock"
                       : "speed decode: INT8 VAE; quality knife uses fp16");
            fflush(stdout);
        }
        latent_chw = h3_vae_int8_unpack_dit_latent_f32(
            latent, latent_t, latent_h, latent_w, error, sizeof(error));
        if (!latent_chw) {
            fprintf(stderr, "unpack: %s\n", error);
            goto cleanup;
        }
        t0 = now_ms();
        for (i = 0; i < (size_t)vae_repeats; i++) {
            h3_vae_int8_frames run_frames;
            h3_vae_int8_decode_stats run_vst;
            double repeat_ms;
            memset(&run_frames, 0, sizeof(run_frames));
            memset(&run_vst, 0, sizeof(run_vst));
            if (!h3_vae_int8_decode_cuda(cuda, &vae_store, latent_chw, latent_t,
                                         latent_h, latent_w, vae_blocks,
                                         &run_frames, &run_vst, error,
                                         sizeof(error))) {
                fprintf(stderr, "vae decode: %s\n", error);
                h3_vae_int8_frames_free(&run_frames);
                goto cleanup;
            }
            repeat_ms = now_ms() - t0;
            if (i > 0) h3_vae_int8_frames_free(&frames);
            frames = run_frames;
            vst = run_vst;
            fprintf(stderr,
                    "vae_repeat %zu/%d: load=%.1f blocks=%.1f total=%.1f "
                    "wall=%.1f\n",
                    i + 1, vae_repeats, vst.load_ms, vst.blocks_ms, vst.total_ms,
                    repeat_ms);
            t_vae = repeat_ms;
            t0 = now_ms();
        }
        printf("vae: load=%.1f canvas_h2d=%.1f blocks=%.1f unpatch_d2h=%.1f "
               "total=%.1f frames=%d %dx%d\n",
               vst.load_ms, vst.canvas_h2d_ms, vst.blocks_ms, vst.unpatch_d2h_ms,
               vst.total_ms, frames.frames, frames.width, frames.height);

        /* A3: audio VAE decode overlaps PPM dump (GPU idle). Default on;
         * H3_AUDIO_VAE_OVERLAP=0 forces sequential. */
        if (decode_audio) {
            const char *ov = getenv("H3_AUDIO_VAE_OVERLAP");
            int want_ov = !(ov && ov[0] == '0');
            t0 = now_ms();
            if (!h3_dit_w4a8_unpack_audio_latent(audio_rows, audio_t, audio_c2t,
                                                 error, sizeof(error))) {
                fprintf(stderr, "unpack audio: %s\n", error);
                goto cleanup;
            }
            audio_job.cuda = cuda;
            audio_job.audio_vae_path = audio_vae_path;
            audio_job.audio_c2t = audio_c2t;
            audio_job.audio_t = audio_t;
            audio_job.awave = &awave;
            audio_job.ast = &ast;
            audio_job.ok = 0;
            if (want_ov &&
                pthread_create(&audio_th, NULL, audio_decode_th, &audio_job) ==
                    0) {
                audio_th_ok = 1;
                printf("audio_vae: overlap=ppm (H3_AUDIO_VAE_OVERLAP=0 "
                       "disables)\n");
                fflush(stdout);
            } else {
                if (!h3_audio_vae_cuda_decode(cuda, audio_vae_path, audio_c2t,
                                              audio_t, NULL, NULL, &awave, &ast,
                                              error, sizeof(error))) {
                    fprintf(stderr, "audio vae decode: %s\n", error);
                    goto cleanup;
                }
                t_audio = now_ms() - t0;
            }
        }
        {
            const char *skip_ppm_e = getenv("H3_SKIP_PPM");
            int skip_ppm = skip_ppm_e && skip_ppm_e[0] == '1';
            int f;
            int nwrite = frames.frames < 120 ? frames.frames : 120;
            size_t nbytes = 0;
            double t_io0 = now_ms();
            mux_rgb = frames_to_rgb24(&frames, &nbytes, error, sizeof(error));
            if (!mux_rgb) {
                fprintf(stderr, "rgb24: %s\n", error);
                goto cleanup;
            }
            if (!skip_ppm) {
                size_t frame_bytes =
                    (size_t)frames.width * (size_t)frames.height * 3;
                for (f = 0; f < nwrite; f++) {
                    snprintf(pathbuf, sizeof(pathbuf), "%s/frame_%04d.ppm",
                             out_dir, f);
                    if (!write_ppm_rgb24(pathbuf, mux_rgb + (size_t)f * frame_bytes,
                                         frames.width, frames.height, error,
                                         sizeof(error))) {
                        fprintf(stderr, "ppm %d: %s\n", f, error);
                        goto cleanup;
                    }
                }
                printf("wrote %d PPM frames under %s/\n", nwrite, out_dir);
            } else {
                printf("ppm: skipped (H3_SKIP_PPM=1)\n");
            }
            t_ppm = now_ms() - t_io0;
        }
    }

    if (audio_th_ok) {
        pthread_join(audio_th, NULL);
        audio_th_ok = 0;
        if (!audio_job.ok) {
            fprintf(stderr, "audio vae decode: %s\n", audio_job.err);
            goto cleanup;
        }
        t_audio = ast.total_ms;
        printf("audio_vae: load=%.1f decode=%.1f total=%.1f samples=%d "
               "audio_ms=%.1f overlap=ppm\n",
               ast.load_ms, ast.decode_ms, ast.total_ms, awave.samples,
               t_audio);
    } else if (decode_audio && awave.pcm) {
        printf("audio_vae: load=%.1f decode=%.1f total=%.1f samples=%d "
               "audio_ms=%.1f\n",
               ast.load_ms, ast.decode_ms, ast.total_ms, awave.samples,
               t_audio);
    }

    if (decode_audio && awave.pcm) {
        snprintf(pathbuf, sizeof(pathbuf), "%s/audio.wav", out_dir);
        if (!write_wav_f32_stereo(pathbuf, awave.pcm, awave.channels,
                                  awave.samples, awave.sample_rate, error,
                                  sizeof(error))) {
            fprintf(stderr, "wav: %s\n", error);
            goto cleanup;
        }
        printf("wrote %s\n", pathbuf);
        {
            int i, n = awave.samples * (awave.channels > 0 ? awave.channels : 1);
            float peak = 0.0f;
            for (i = 0; i < n; i++) {
                float a = fabsf(awave.pcm[i]);
                if (a > peak) peak = a;
            }
            printf("audio_pcm: samples=%d sr=%d ch=%d peak=%.4f "
                   "duration=%.3fs\n",
                   awave.samples, awave.sample_rate, awave.channels,
                   (double)peak,
                   awave.sample_rate > 0
                       ? (double)awave.samples / (double)awave.sample_rate
                       : 0.0);
            if (!(peak > 1.0e-4f))
                fprintf(stderr,
                        "warning: decoded audio near-silent (peak=%.6f); "
                        "not muxing a fake empty AAC track\n",
                        (double)peak);
        }
    }

    if (mux_path) {
        if (skip_vae || !frames.rgb) {
            fprintf(stderr, "mux requires VAE decode frames\n");
            goto cleanup;
        }
        {
            size_t nbytes = 0;
            int mux_frames = frames_req;
            double t_mux0 = now_ms();
            if (mux_frames < 1) mux_frames = 1;
            if (mux_frames > frames.frames) mux_frames = frames.frames;
            if (!mux_rgb) {
                mux_rgb = frames_to_rgb24(&frames, &nbytes, error,
                                          sizeof(error));
                if (!mux_rgb) {
                    fprintf(stderr, "mux rgb: %s\n", error);
                    goto cleanup;
                }
            }
            if (decode_audio && awave.pcm) {
                int i, n = awave.samples *
                           (awave.channels > 0 ? awave.channels : 1);
                float peak = 0.0f;
                for (i = 0; i < n; i++) {
                    float a = fabsf(awave.pcm[i]);
                    if (a > peak) peak = a;
                }
                if (!(peak > 1.0e-4f)) {
                    fprintf(stderr,
                            "mux: refusing silent audio (peak=%.6f); "
                            "fix Audio VAE decode instead of padding AAC\n",
                            (double)peak);
                    goto cleanup;
                }
                if (!h3_ffmpeg_write_av_rgb24_f32(
                        mux_path, mux_rgb, mux_frames, frames.width,
                        frames.height, 24, awave.pcm, awave.samples,
                        awave.channels, awave.sample_rate, error,
                        sizeof(error))) {
                    fprintf(stderr, "mux: %s\n", error);
                    goto cleanup;
                }
                printf("wrote mux %s (%d/%d frames + %d samples)\n", mux_path,
                       mux_frames, frames.frames, awave.samples);
            } else {
                if (!h3_ffmpeg_write_rgb24(mux_path, mux_rgb, mux_frames,
                                           frames.width, frames.height, 24,
                                           error, sizeof(error))) {
                    fprintf(stderr, "mux video-only: %s\n", error);
                    goto cleanup;
                }
                printf("wrote mux %s (%d/%d frames @24fps = %.4fs wall, "
                       "video-only%s)\n",
                       mux_path, mux_frames, frames.frames,
                       (double)mux_frames / 24.0,
                       skip_audio ? "; --skip-audio" : "; no audio VAE");
            }
            {
                double mux_sec = (double)mux_frames / 24.0;
                printf("mux_duration: %.3fs (%d frames @24fps, requested "
                       "--seconds %d)\n",
                       mux_sec, mux_frames, seconds);
                if (mux_frames < seconds * 24) {
                    fprintf(stderr,
                            "mux WARNING: %d frames @24fps = %.3fs, but "
                            "--seconds %d wants %d frames (%.1fs). "
                            "Omit --frames (or pass --frames >= %d) for "
                            "full duration.\n",
                            mux_frames, mux_sec, seconds, seconds * 24,
                            (double)seconds, seconds * 24);
                }
            }
            t_mux = now_ms() - t_mux0;
        }
    }

    printf("phase_ms: te=%.1f dit=%.1f vae=%.1f audio=%.1f ppm=%.1f mux=%.1f "
           "total=%.1f dit_evals=%d dit_skips=%d\n",
           t_te, t_dit, t_vae, t_audio, t_ppm, t_mux, now_ms() - t_all0,
           dit_evals, dit_skips);
    if (pack_audio)
        printf("summary: %s pack PASS (audio_pack=on decode=%s)\n",
               (n_ref_images || ref_video_path) ? "ref2va" : "fl2va",
               decode_audio ? "on" : "off");
    else
        printf("summary: text|video PASS (audio_pack=off; theme may fail)\n");
    rc = 0;

cleanup:
    if (audio_th_ok) pthread_join(audio_th, NULL);
    if (sdpa_th_ok) pthread_join(sdpa_th, NULL);
    if (vae_th_ok) pthread_join(vae_th, NULL);
    if (host_stage_th_ok) pthread_join(host_stage_th, NULL);
    if (prefetch_done) cudaEventDestroy((cudaEvent_t)prefetch_done);
    free(latent);
    free(velocity);
    free(sigmas);
    free(audio_sigmas);
    free(reuse_sel);
    free(vel_last);
    free(vel_prev);
    free(avel_last);
    free(avel_prev);
    free(audio_c2t);
    free(audio_rows);
    free(audio_vel);
    free(text_emb);
    free(text_h);
    free(text_h_uncond);
    free(velocity_uncond);
    free(audio_vel_uncond);
    free(latent_chw);
    free(token_ids);
    free(pt);
    free(ph);
    free(pw);
    free(apt);
    free(aph);
    free(apw);
    free(frame0_lock);
    free(frame0_img);
    free(ff_rgb);
    free(lf_rgb);
    free(ref_video_rgb);
    {
        int ri;
        for (ri = 0; ri < H3_CUDA_MAX_REF_IMAGES; ri++)
            free(ref_rgb[ri]);
    }
    free(cond_rows);
    free(cpt);
    free(cph);
    free(cpw);
    free(audio_cond_rows);
    free(acpt);
    free(acph);
    free(acpw);
    free(vision_merged);
    free(vision_ds[0]);
    free(vision_ds[1]);
    free(vision_ds[2]);
    free(text_mod_tags);
    free(mux_rgb);
    h3_audio_vae_cuda_waveform_free(&awave);
    h3_vae_int8_frames_free(&frames);
    h3_vae_int8_store_free(&vae_store);
    h3_nvfp4_store_free(&te_store);
    h3_dit_w4a8_core_free(&core);
    gen_uly_free_ranks(dit, uly_dit, uly_cuda);
    h3_dit_w4a8_free(dit);
    dit = NULL;
    if (lora_loaded) {
        h3_lora_free(&lora_set);
        lora_loaded = 0;
    }
    h3_ulysses_comm_destroy(&ulysses_cfg);
    h3_nvfp4_cuda_disable();
    {
        int ui;
        for (ui = 1; ui < 8; ui++) {
            if (uly_cuda[ui]) {
                h3_cuda_free(uly_cuda[ui]);
                uly_cuda[ui] = NULL;
            }
        }
    }
    h3_cuda_free(cuda);
    h3_w4a8_store_free(dit_store);
    h3_dit_int8_store_free(dit_int8);
    return rc;
}
