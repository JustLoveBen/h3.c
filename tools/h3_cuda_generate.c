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
#include "h3_dit_nvfp4.h"
#include "h3_av_sync.h"
#include "h3_ffmpeg.h"
#include "h3_run_report.h"
#include "h3_waffle.h"
#include "h3_multimodal_cuda.h"
#include "h3_nvfp4.h"
#include "h3_nvfp4_cuda.h"
#include "h3_token_ids.h"
#include "h3_vae_encode_f32.h"
#include "h3_vae_int8.h"
#include "h3_vae_int8_cuda.h"
#include "h3_vae_prewarm.h"
#include "h3_vision_cuda.h"
#include "h3_audio_vae_cuda.h"
#include "h3_audio_vae_layout.h"
#include "h3_cuda_device_caps.h"
#include "h3_generate_preset.h"
#include "h3_latent_upscaler.h"
#include "h3_lora.h"
#include "h3_fewstep_contract.h"
#include "h3_env_contract.h"
#include "h3_sampler.h"
#include "h3_kpi_guard.h"
#include "h3_sdpa.h"
#include "h3_selflift.h"
#include "h3_step_cache.h"
#include "h3_tae.h"
#include "h3_taomate.h"
#include "h3_te_cache.h"
#include "h3_lifecycle_whitelist.h"
#include "h3_ulysses.h"
#include "h3_ulysses_comm.h"
#include "h3_w4a8_window.h"

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <fcntl.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <pthread.h>
#include <stdint.h>
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

/* C074: structured failure tags for log grepping (print before exit). */
static void h3_fail_tag(const char *tag) {
    if (!tag || !tag[0])
        return;
    fprintf(stderr, "tag=%s\n", tag);
    fflush(stderr);
}

/* C040: opt-in VRAM ledger (H3_VRAM_LEDGER=1). Host builds without CUDA
 * skip via stub; this TU always has cuda_runtime.h.
 * Points: after_cuda → after_te → before_dit → peak_dit (DiT resident) →
 *         after_dit_release → before_vae → after_vae (last --repeat clip).
 * Each line also emits vram_ledger_delta vs the previous armed point (分桶). */
static int h3_vram_ledger_on(void) {
    const char *e = getenv("H3_VRAM_LEDGER");
    return (e && e[0] == '1') ? 1 : 0;
}

static void h3_vram_ledger_point(const char *point) {
    static double prev_used_mib = -1.0;
    static const char *prev_point = NULL;
    size_t free_b = 0, total_b = 0;
    double used_mib;
    if (!h3_vram_ledger_on() || !point)
        return;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        (void)cudaGetLastError();
        printf("vram_ledger: point=%s status=memgetinfo_fail\n", point);
        fflush(stdout);
        return;
    }
    used_mib = (double)(total_b - free_b) / (1024.0 * 1024.0);
    printf("vram_ledger: point=%s free_mib=%.1f used_mib=%.1f total_mib=%.1f\n",
           point, (double)free_b / (1024.0 * 1024.0), used_mib,
           (double)total_b / (1024.0 * 1024.0));
    if (prev_used_mib >= 0.0 && prev_point)
        printf("vram_ledger_delta: from=%s to=%s d_used_mib=%+.1f\n",
               prev_point, point, used_mib - prev_used_mib);
    prev_used_mib = used_mib;
    prev_point = point;
    fflush(stdout);
}


/* Opt-in nsys capture of the first hot-clip high-res NFE. Default off. */
static int h3_nsys_hires_begin(int clip_i, int first_hi_nfe) {
    const char *e = getenv("H3_NSYS_HIRES");
    if (!(e && e[0] == '1' && e[1] == '\0' && clip_i > 0 && first_hi_nfe))
        return 0;
    printf("nsys_hires: profiler_start clip=%d\n", clip_i + 1);
    fflush(stdout);
    cudaProfilerStart();
    return 1;
}

static void h3_nsys_hires_end(int on, double dit_ms) {
    if (!on)
        return;
    cudaProfilerStop();
    printf("nsys_hires: profiler_stop dit_ms=%.1f\n", dit_ms);
    fflush(stdout);
}

static void gen_mem_trace(h3_cuda *cuda, const char *phase) {
    const char *e = getenv("H3_GEN_MEM_TRACE");
    size_t free_b = 0, total_b = 0;
    char err[64];
    if (!e || e[0] != '1' || e[1] != '\0' || !phase)
        return;
    err[0] = '\0';
    if (cuda)
        (void)h3_cuda_mem_info(cuda, &free_b, &total_b, err, sizeof(err));
    printf("mem: phase=%s free_gib=%.2f total_gib=%.2f\n", phase,
           (double)free_b / (1024.0 * 1024.0 * 1024.0),
           (double)total_b / (1024.0 * 1024.0 * 1024.0));
    fflush(stdout);
}

/* Spatial adjacent-pixel energy on one HxW plane. Checkerboard → ratio ~1. */
static float chw_adj_ratio(const float *z, int t, int h, int w) {
    double s = 0.0, d = 0.0;
    int hh, ww, n = 0;
    (void)t;
    if (!z || h < 1 || w < 2)
        return 0.0f;
    for (hh = 0; hh < h; hh++)
        for (ww = 0; ww < w; ww++) {
            float v = z[(size_t)hh * (size_t)w + (size_t)ww];
            s += fabsf(v);
            if (ww + 1 < w) {
                d += fabsf(v - z[(size_t)hh * (size_t)w + (size_t)(ww + 1)]);
                n++;
            }
        }
    if (s < 1e-8 || n < 1)
        return 0.0f;
    return (float)((d / (double)n) / (s / (double)(h * w)));
}

static void dump_f32_bin(const char *path, const float *z, size_t n) {
    FILE *f;
    if (!path || !z || !n)
        return;
    f = fopen(path, "wb");
    if (!f)
        return;
    fwrite(z, sizeof(float), n, f);
    fclose(f);
}

/* Per-channel CTHW stats. Flower-screen often hides in ch≠0. */
static void dump_chw_channel_stats(const char *tag, const float *z, int t,
                                   int h, int w) {
    int c, worst_c = 0;
    float worst = 0.0f;
    size_t plane = (size_t)h * (size_t)w;
    size_t ct = (size_t)t * plane;
    if (!z || t < 1 || h < 1 || w < 1)
        return;
    printf("selflift: %s CTHW=24x%dx%dx%d\n", tag, t, h, w);
    for (c = 0; c < H3_VAE_INT8_LATENT_CH; c++) {
        const float *base = z + (size_t)c * ct;
        double mn = 1e30, mx = -1e30, sm = 0.0;
        size_t i, n = ct;
        float adj = chw_adj_ratio(base, 1, h, w);
        int nan = 0;
        for (i = 0; i < n; i++) {
            double v = (double)base[i];
            if (!isfinite(v))
                nan++;
            if (v < mn)
                mn = v;
            if (v > mx)
                mx = v;
            sm += v;
        }
        if (adj > worst) {
            worst = adj;
            worst_c = c;
        }
        if (c < 3 || c == 8 || c == 16 || c == 23 || adj > 0.25f || nan) {
            printf("selflift: %s ch%02d min=%.4g max=%.4g mean=%.4g "
                   "adj_t0=%.4g nan=%d\n",
                   tag, c, mn, mx, sm / (double)n, (double)adj, nan);
        }
    }
    printf("selflift: %s worst_adj ch%02d=%.4g (~1=checkerboard)\n", tag,
           worst_c, (double)worst);
    fflush(stdout);
}

static void dump_chw_pgm(const char *path, const float *z, int h, int w) {
    FILE *f;
    int i, g;
    float mn = 1e30f, mx = -1e30f;
    if (!path || !z || h < 1 || w < 1)
        return;
    for (i = 0; i < h * w; i++) {
        if (z[i] < mn)
            mn = z[i];
        if (z[i] > mx)
            mx = z[i];
    }
    f = fopen(path, "wb");
    if (!f)
        return;
    fprintf(f, "P5\n%d %d\n255\n", w, h);
    for (i = 0; i < h * w; i++) {
        g = 0;
        if (mx > mn)
            g = (int)((z[i] - mn) / (mx - mn) * 255.0f + 0.5f);
        if (g < 0)
            g = 0;
        if (g > 255)
            g = 255;
        fputc(g, f);
    }
    fclose(f);
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
    volatile int done;
} dit_host_stage_job;

static void *dit_host_stage_th(void *arg) {
    dit_host_stage_job *j = (dit_host_stage_job *)arg;
    if (!j || !j->dit) return NULL;
    j->ok = h3_dit_w4a8_stage_host(j->dit, j->first, j->n, &j->ms, j->err,
                                   sizeof(j->err));
    j->done = 1;
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
        printf("dit_host_stage: kick disk→host blocks=%u "
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

/* R11: last-2 NFE kick. Thread body lives in h3_vae_prewarm.c. */
static int kick_vae_prewarm(pthread_t *th, h3_vae_prewarm_job *job,
                            const char *path, h3_vae_int8_store *store,
                            int *opened, int device, int at_step, int want_dev,
                            unsigned n_blocks) {
    if (!th || !job || !path || !path[0] || !store) return 0;
    if (!h3_vae_prewarm_wanted()) return 0;
    memset(job, 0, sizeof(*job));
    job->path = path;
    job->store = store;
    job->opened = opened;
    job->device = device;
    job->want_hostreg = 1;
    job->want_dev = want_dev;
    job->n_blocks = n_blocks;
    job->at_step = at_step;
    if (pthread_create(th, NULL, h3_vae_prewarm_th, job) != 0) return 0;
    printf("vae_th: created at=dit_step%d want_dev=%d\n", at_step, want_dev);
    h3_lifecycle_trace(want_dev ? "VAE prewarm device-W" : "VAE prewarm host",
                       "dit_step", "kick", want_dev ? "want_dev=1" : "host_only");
    fflush(stdout);
    return 1;
}

/* T4-c: prewarm 线程预填 device W 消除热 clip vae_load 余尾（~0.3s）。
 * H3_VAE_PREWARM_DEV=auto（默认）：仅当当前空闲显存 ≥ 3×VAE device W
 * （~1.8GiB 权 + ~3.6GiB 安全垫给解码激活/scratch + sol_cute 激活）
 * 才预填，768P/480P hot clip 32GB 无 headroom 时自动退回 want_dev=0
 * （现状，零风险）；=1 强制预填；=0 关闭。 */
static int vae_prewarm_want_dev(int device) {
    const char *e = getenv("H3_VAE_PREWARM_DEV");
    if (e && e[0] == '0' && e[1] == '\0') return 0;
    if (e && e[0] == '1' && e[1] == '\0') return 1;
    {
        const size_t need = (size_t)6 * 1024 * 1024 * 1024ull;
        size_t free_b = 0, total_b = 0;
        if (device < 0 || cudaSetDevice(device) != cudaSuccess) {
            (void)cudaGetLastError();
            return 0;
        }
        if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
            (void)cudaGetLastError();
            return 0;
        }
        if (free_b < need) {
            printf("vae_prewarm_dev: free=%.2fGiB < %.2fGiB gate → host-only "
                   "prewarm (H3_VAE_PREWARM_DEV=1 forces)\n",
                   (double)free_b / (1024.0 * 1024.0 * 1024.0),
                   (double)need / (1024.0 * 1024.0 * 1024.0));
            fflush(stdout);
            return 0;
        }
        printf("vae_prewarm_dev: free=%.2fGiB ≥ gate → device W prefill\n",
               (double)free_b / (1024.0 * 1024.0 * 1024.0));
        fflush(stdout);
        return 1;
    }
}

/* R20: decode|ppm|0. Default decode (overlap video VAE). */
static int audio_vae_overlap_mode(void) {
    const char *ov = getenv("H3_AUDIO_VAE_OVERLAP");
    if (!ov || !ov[0]) return 2;
    if (ov[0] == '0' && ov[1] == '\0') return 0;
    if (strcmp(ov, "ppm") == 0) return 1;
    return 2;
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

/* P0 A/B verification only (H3_DUMP_LATENT / H3_REPLAY_LATENT): dump the
 * final packed DiT latent (f32, layout [T,H/2,W/2,96], same as latent.raw)
 * before VAE consumes it, and replay a dump straight into VAE decode so the
 * async D2H arm can be compared bit-exact on a fixed input. Not a product
 * path; both switches default off. */
static int dump_latent_file(const char *path, const float *latent, size_t n,
                            int t, int h, int w, char *error,
                            size_t error_size) {
    char hdr[64];
    FILE *fp;
    if (!path || !path[0] || !latent) return 1;
    snprintf(hdr, sizeof(hdr), "H3LAT1 %d %d %d %zu\n", t, h, w, n);
    fp = fopen(path, "wb");
    if (!fp) {
        snprintf(error, error_size, "cannot open %s for write", path);
        return 0;
    }
    if (fwrite(hdr, 1, sizeof(hdr), fp) != sizeof(hdr) ||
        fwrite(latent, sizeof(float), n, fp) != n) {
        fclose(fp);
        snprintf(error, error_size, "short write %s", path);
        return 0;
    }
    fclose(fp);
    return 1;
}

static int load_latent_file(const char *path, float *latent, size_t n_expect,
                            int *t_out, int *h_out, int *w_out, char *error,
                            size_t error_size) {
    char hdr[64], magic[7];
    FILE *fp;
    int t = 0, h = 0, w = 0;
    size_t n = 0;
    fp = fopen(path, "rb");
    if (!fp) {
        snprintf(error, error_size, "cannot open latent dump %s", path);
        return 0;
    }
    if (fread(hdr, 1, sizeof(hdr), fp) != sizeof(hdr) ||
        sscanf(hdr, "%6s %d %d %d %zu", magic, &t, &h, &w, &n) != 5 ||
        strcmp(magic, "H3LAT1") != 0) {
        fclose(fp);
        snprintf(error, error_size, "bad latent dump header in %s", path);
        return 0;
    }
    if (t < 1 || h < 2 || w < 2 || (h % 2) || (w % 2) ||
        n != (size_t)t * (size_t)(h / 2) * (size_t)(w / 2) * 96) {
        fclose(fp);
        snprintf(error, error_size,
                 "latent dump %s shape %dx%dx%d inconsistent with n=%zu", path,
                 t, h, w, n);
        return 0;
    }
    if (latent) {
        if (n != n_expect) {
            fclose(fp);
            snprintf(error, error_size,
                     "latent dump n=%zu, run buffer %zu (geometry mismatch)",
                     n, n_expect);
            return 0;
        }
        if (fread(latent, sizeof(float), n, fp) != n) {
            fclose(fp);
            snprintf(error, error_size, "short read %s", path);
            return 0;
        }
    }
    fclose(fp);
    if (t_out) *t_out = t;
    if (h_out) *h_out = h;
    if (w_out) *w_out = w;
    return 1;
}

static uint64_t fnv1a64_bytes(const uint8_t *p, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    size_t i;
    for (i = 0; i < n; i++) {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ULL;
    }
    return h;
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
    if (i < count)     dst[i] = hash_to_unit(seed, i);
}

/* --repeat N: unique mux per in-process clip. clip_i is 0-based. */
static void mux_path_for_clip(char *dst, size_t dst_sz, const char *base,
                              int clip_i, int n) {
    const char *dot;
    const char *slash;
    if (!dst || dst_sz < 8) return;
    if (!base || !base[0]) {
        dst[0] = '\0';
        return;
    }
    if (n <= 1) {
        snprintf(dst, dst_sz, "%s", base);
        return;
    }
    slash = strrchr(base, '/');
    dot = strrchr(base, '.');
    if (dot && (!slash || dot > slash))
        snprintf(dst, dst_sz, "%.*s_r%d%s", (int)(dot - base), base, clip_i + 1,
                 dot);
    else
        snprintf(dst, dst_sz, "%s_r%d", base, clip_i + 1);
}

static int grow_f32(float **p, size_t n, char *err, size_t err_size) {
    float *q;
    if (n == 0) {
        snprintf(err, err_size, "grow_f32 empty");
        return 0;
    }
    q = (float *)realloc(*p, n * sizeof(float));
    if (!q) {
        snprintf(err, err_size, "OOM realloc %zu f32", n);
        return 0;
    }
    *p = q;
    return 1;
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

/* Dispatch one denoise step: velocity_euler or lcm_renoise (DMAD). */
static int apply_sampler_step_f32(h3_sampler_id sid, float *sample,
                                  const float *velocity, size_t count,
                                  float sigma, float sigma_next,
                                  float *renoise_scratch, uint64_t noise_seed,
                                  char *error, size_t error_size) {
    if (sid == H3_SAMPLER_LCM_RENOISE) {
        const float *noise = NULL;
        if (sigma_next > 1e-8f) {
            if (!renoise_scratch) {
                snprintf(error, error_size, "lcm_renoise: no scratch");
                return 0;
            }
            fill_normal_f32(noise_seed, renoise_scratch, count);
            noise = renoise_scratch;
        }
        return h3_sampler_lcm_renoise_step_f32(sample, velocity, count, sigma,
                                               sigma_next, noise, error,
                                               error_size);
    }
    return h3_dit_w4a8_euler_step_f32(sample, velocity, count, sigma,
                                      sigma_next, error, error_size);
}

#ifndef H3_CUDA_AUDIO_SIGMA_SHIFT
#define H3_CUDA_AUDIO_SIGMA_SHIFT 3.0f
#endif

/* DMAD 复验（docs/research/2026-10-07-dmad-noise-diagnosis.md）：DMAD 官方
 * audio sigma shift=2.0（基座管线默认 3.0）。H3_AUDIO_SIGMA_SHIFT 环境变量
 * opt-in 覆盖；缺省 3.0 不改任何既有行为/金线。 */
static float audio_sigma_shift_effective(void) {
    const char *e = getenv("H3_AUDIO_SIGMA_SHIFT");
    if (e && e[0]) {
        float v = strtof(e, NULL);
        if (isfinite(v) && v >= 1.0f && v <= 20.0f)
            return v;
    }
    return H3_CUDA_AUDIO_SIGMA_SHIFT;
}

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

static void taomate_copy_video_t(const float *src, float *dst, int t0, int t1,
                                 size_t t_floats) {
    size_t n = (size_t)(t1 - t0) * t_floats;
    memcpy(dst, src + (size_t)t0 * t_floats, n * sizeof(float));
}

static void taomate_copy_audio_t(const float *src, float *dst, int audio_t,
                                 int a0, int a1) {
    int n = a1 - a0;
    size_t ch = (size_t)H3_DIT_W4A8_AUDIO_CH;
    memcpy(dst, src + (size_t)a0 * ch, (size_t)n * ch * sizeof(float));
    memcpy(dst + (size_t)n * ch, src + ((size_t)audio_t + (size_t)a0) * ch,
           (size_t)n * ch * sizeof(float));
}

static void taomate_write_audio_t(float *dst, const float *src, int audio_t,
                                  int a0, int a1) {
    int n = a1 - a0;
    size_t ch = (size_t)H3_DIT_W4A8_AUDIO_CH;
    memcpy(dst + (size_t)a0 * ch, src, (size_t)n * ch * sizeof(float));
    memcpy(dst + ((size_t)audio_t + (size_t)a0) * ch, src + (size_t)n * ch,
           (size_t)n * ch * sizeof(float));
}

static void taomate_copy_audio_pos(const float *src, float *dst, int audio_t,
                                   int a0, int a1) {
    int n = a1 - a0;
    memcpy(dst, src + a0, (size_t)n * sizeof(float));
    memcpy(dst + n, src + audio_t + a0, (size_t)n * sizeof(float));
}

static int taomate_clamp_ov_t(long v) {
    if (v < 0)
        return 0;
    if (v > 12)
        return 12;
    return (int)v;
}

static unsigned lift_hires_window_want(void) {
    unsigned want = 50;
    const char *hw = getenv("H3_LIFT_HIRES_WINDOW");
    if (hw && hw[0]) {
        long v = strtol(hw, NULL, 10);
        if (v >= 8 && v <= 50)
            want = (unsigned)v;
    }
    return want;
}

static void lift_fit_hires_window(h3_w4a8_cache *cw, h3_cuda *cuda,
                                  size_t packed_seq, unsigned want) {
    h3_w4a8_cache_stats st;
    size_t free_b = 0, total_b = 0;
    unsigned fit = 8;
    char mem_err[128];
    memset(&st, 0, sizeof(st));
    (void)h3_w4a8_cache_get_stats(cw, &st, NULL, 0);
    mem_err[0] = '\0';
    if (!h3_cuda_mem_info(cuda, &free_b, &total_b, mem_err, sizeof(mem_err))) {
        h3_w4a8_cache_set_window_size(cw, want < 24 ? want : 24);
        printf("selflift: hires mem_info failed (%s); window=%u\n", mem_err,
               h3_w4a8_cache_window_size(cw));
        fflush(stdout);
        return;
    }
    if (st.bytes_per_block_est > 0) {
        size_t reserve = (size_t)1536ULL << 20;
        size_t budget = free_b > reserve ? free_b - reserve : 0;
        fit = (unsigned)(budget / st.bytes_per_block_est);
    }
    if (fit < 8)
        fit = 8;
    if (fit > want)
        fit = want;
    h3_w4a8_cache_set_window_size(cw, fit);
    printf("selflift: hires seq=%zu window=%u free_giB=%.2f (INT8 after WS)\n",
           packed_seq, fit, (double)free_b / (1024.0 * 1024.0 * 1024.0));
    fflush(stdout);
}

static int lift_prepare_hires_vram(h3_dit_w4a8 *dit, h3_cuda *cuda,
                                   size_t packed_seq, double *switch_ms,
                                   char *error, size_t error_size) {
    h3_w4a8_cache *cw;
    char scratch_err[256];
    unsigned want;
    int ws_ok;
    double t_all, t_drop, t_pre, t_refit;
    int skip_sage_prebuild = 0;
    unsigned window_after = 0;
    if (switch_ms)
        *switch_ms = 0.0;
    if (!dit || !cuda)
        return 1;
    cw = h3_dit_w4a8_cache(dit);
    want = lift_hires_window_want();
    t_all = now_ms();
    /* WS-first: drop INT8, grow hires act/block WS, then refit INT8 into
     * leftover VRAM. Keeping 50 then allocating 8.8GiB WS made make_room
     * drop 50→32 and the next w_i8 malloc OOM (hires 2 NFE ~31s thrash). */
    if (cw) {
        h3_w4a8_cache_unfreeze_window(cw);
        h3_w4a8_cache_reset_refit(cw);
        h3_w4a8_cache_drop_device_weights(cw);
        h3_w4a8_cache_log_window(cw, "hires_before");
    }
    h3_dit_w4a8_release_block_workspace(dit);
    h3_cuda_release_heavy_workspace(cuda);
    /* R4: leftover low-res sage pool is the wrong seq and pins window=1.
     * 768P sage3 is a fake hit (R3) — drop unless hits==want on this run. */
    {
        unsigned long long sw = 0, sh = 0, sf = 0;
        size_t free_b = 0, total_b = 0;
        size_t sage_now = h3_cuda_sage3_pool_bytes();
        size_t sage_hi = 4ull * packed_seq * (size_t)H3_DIT_W4A8_HEADS *
                         (size_t)H3_DIT_W4A8_HEAD_DIM * 2ull;
        size_t reserve = (size_t)1536ULL << 20;
        size_t bpb = cw ? h3_w4a8_cache_bytes_per_block(cw) : 0;
        unsigned n_blocks = 50;
        unsigned fit_with, fit_without;
        int sage_true = 0;
        char mem_err[128];
        h3_cuda_sage3_get_stats(&sw, &sh, &sf);
        sage_true = (sw > 0ull && sh == sw);
        mem_err[0] = '\0';
        (void)h3_cuda_mem_info(cuda, &free_b, &total_b, mem_err,
                               sizeof(mem_err));
        if (cw && h3_w4a8_cache_max_blocks_fit(cw) > 0)
            n_blocks = 50;
        fit_with = h3_w4a8_cache_max_blocks_fit_calc(
            free_b, reserve, sage_now + sage_hi, bpb, n_blocks);
        fit_without = h3_w4a8_cache_max_blocks_fit_calc(free_b, reserve, 0, bpb,
                                                        n_blocks);
        skip_sage_prebuild = h3_w4a8_cache_should_drop_sage_pool(
            fit_with, fit_without, 8, sage_true);
        if (h3_sdpa_cached_id() != H3_SDPA_ID_SAGE3)
            skip_sage_prebuild = 1;
        if (skip_sage_prebuild) {
            h3_cuda_sage3_drop_pool();
            printf("dit_int8_cache: tag=hires_drop_sage fit_with=%u "
                   "fit_without=%u sage_true=%d sage_now_mib=%.1f "
                   "sage_hi_mib=%.1f\n",
                   fit_with, fit_without, sage_true,
                   (double)sage_now / (1024.0 * 1024.0),
                   (double)sage_hi / (1024.0 * 1024.0));
            fflush(stdout);
        }
        /* Low-res sol QKVO is the wrong seq and sits in the torch cache. */
        h3_cuda_sol_cute_drop_pool();
        h3_cuda_sol_cute_trim_cache();
    }
    t_drop = now_ms() - t_all;
    t_all = now_ms();
    scratch_err[0] = '\0';
    if (!h3_cuda_pregrow_dit_scratch(cuda, packed_seq, H3_DIT_W4A8_HIDDEN,
                                     (size_t)H3_DIT_W4A8_HEADS,
                                     (size_t)H3_DIT_W4A8_HEAD_DIM,
                                     (size_t)H3_DIT_W4A8_FFN, scratch_err,
                                     sizeof(scratch_err))) {
        snprintf(error, error_size, "%s", scratch_err);
        return 0;
    }
    scratch_err[0] = '\0';
    ws_ok = h3_dit_w4a8_pregrow_layers_ws(dit, packed_seq,
                                          (unsigned)H3_DIT_W4A8_N_BLOCKS,
                                          scratch_err, sizeof(scratch_err));
    if (!ws_ok) {
        printf("selflift: hires WS pregrow failed (%s); continue, fit after SDPA\n",
               scratch_err[0] ? scratch_err : "unknown");
        fflush(stdout);
    }
    /* WS-first order for 32GB: compile hires sol while INT8 is down, drop
     * QKVO, then refit window=32, then prefetch INT8 before first SDPA
     * reallocates QKVO. Skip hires cuDNN plan when sol will own DiT attn. */
    if (h3_sdpa_cached_id() == H3_SDPA_ID_SOL_CUTE &&
        h3_sol_cute_seq_allowed(packed_seq)) {
        printf("selflift: hires sol warmup seq=%zu (compile then drop QKVO)\n",
               packed_seq);
        fflush(stdout);
        h3_cuda_sol_cute_warmup_shape(packed_seq, (size_t)H3_DIT_W4A8_HEADS,
                                      (size_t)H3_DIT_W4A8_HEAD_DIM);
        h3_cuda_sol_cute_trim_cache();
        printf("selflift: skip hires cudnn prebuild seq=%zu (sol owns attn)\n",
               packed_seq);
        fflush(stdout);
    } else if (h3_sdpa_cached_id() == H3_SDPA_ID_SOL_CUTE) {
        printf("selflift: mixed arm seq=%zu max_seq=%zu (hires cudnn)\n",
               packed_seq, h3_sol_cute_max_seq());
        fflush(stdout);
        {
            float attn_scale = 1.0f / sqrtf((float)H3_DIT_W4A8_HEAD_DIM);
            char pb_err[256];
            pb_err[0] = '\0';
            if (!h3_cuda_sdpa_prebuild(cuda, packed_seq,
                                       (size_t)H3_DIT_W4A8_HEADS,
                                       (size_t)H3_DIT_W4A8_HEAD_DIM, attn_scale,
                                       0, pb_err, sizeof(pb_err))) {
                printf("selflift: hires cudnn prebuild seq=%zu failed (%s)\n",
                       packed_seq, pb_err[0] ? pb_err : "unknown");
            }
            fflush(stdout);
        }
    } else if (!skip_sage_prebuild) {
        float attn_scale = 1.0f / sqrtf((float)H3_DIT_W4A8_HEAD_DIM);
        char pb_err[256];
        pb_err[0] = '\0';
        if (!h3_cuda_sdpa_prebuild(cuda, packed_seq, (size_t)H3_DIT_W4A8_HEADS,
                                   (size_t)H3_DIT_W4A8_HEAD_DIM, attn_scale, 0,
                                   pb_err, sizeof(pb_err))) {
            printf("selflift: hires sdpa_prebuild seq=%zu failed (%s)\n",
                   packed_seq, pb_err[0] ? pb_err : "unknown");
        } else {
            printf("selflift: hires sdpa_prebuild seq=%zu "
                   "(after WS, before INT8)\n",
                   packed_seq);
        }
        fflush(stdout);
    } else if (h3_sdpa_cached_id() != H3_SDPA_ID_SAGE3) {
        float attn_scale = 1.0f / sqrtf((float)H3_DIT_W4A8_HEAD_DIM);
        char pb_err[256];
        pb_err[0] = '\0';
        if (!h3_cuda_sdpa_prebuild(cuda, packed_seq, (size_t)H3_DIT_W4A8_HEADS,
                                   (size_t)H3_DIT_W4A8_HEAD_DIM, attn_scale, 0,
                                   pb_err, sizeof(pb_err))) {
            printf("selflift: hires cudnn prebuild seq=%zu failed (%s)\n",
                   packed_seq, pb_err[0] ? pb_err : "unknown");
        }
        fflush(stdout);
    } else {
        printf("selflift: skip sage3 prebuild seq=%zu (R4 window)\n",
               packed_seq);
        fflush(stdout);
    }
    t_pre = now_ms() - t_all;
    t_all = now_ms();
    if (cw) {
        char refit_err[256];
        h3_w4a8_cache_set_window_size(cw, want);
        refit_err[0] = '\0';
        if (!h3_w4a8_cache_refit_after_workspace(cw, refit_err,
                                                 sizeof(refit_err))) {
            printf("selflift: hires INT8 refit failed (%s); fit from free\n",
                   refit_err[0] ? refit_err : "unknown");
            fflush(stdout);
            lift_fit_hires_window(cw, cuda, packed_seq, want);
        } else {
            printf("selflift: hires WS-first seq=%zu INT8 window=%u "
                   "(no make_room drop)\n",
                   packed_seq, h3_w4a8_cache_window_size(cw));
            fflush(stdout);
        }
        h3_w4a8_cache_freeze_window(cw);
        h3_w4a8_cache_log_window(cw, "hires_after");
        window_after = h3_w4a8_cache_window_size(cw);
    }
    t_refit = now_ms() - t_all;
    if (switch_ms)
        *switch_ms = t_drop + t_pre + t_refit;
    printf("lift_switch: drop_ms=%.1f prebuild_ms=%.1f refit_ms=%.1f "
           "window_after=%u overlap=%d\n",
           t_drop, t_pre, t_refit, window_after, 0);
    fflush(stdout);
    gen_mem_trace(cuda, "lift_switch");
    return 1;
}

static void taomate_append_video_kv(float *kv_v, float *kv_pt, float *kv_ph,
                                    float *kv_pw, size_t *kv_vseq,
                                    const float *latent, const float *pt,
                                    const float *ph, const float *pw, int t0,
                                    int t1, size_t spatial, size_t t_floats) {
    size_t hvs, hvn;
    if (!kv_v || !kv_vseq || t1 <= t0)
        return;
    hvs = (size_t)(t1 - t0) * spatial;
    hvn = (size_t)(t1 - t0) * t_floats;
    memcpy(kv_v + (*kv_vseq) * (size_t)H3_DIT_W4A8_VIDEO_PATCH,
           latent + (size_t)t0 * t_floats, hvn * sizeof(float));
    memcpy(kv_pt + *kv_vseq, pt + (size_t)t0 * spatial, hvs * sizeof(float));
    memcpy(kv_ph + *kv_vseq, ph + (size_t)t0 * spatial, hvs * sizeof(float));
    memcpy(kv_pw + *kv_vseq, pw + (size_t)t0 * spatial, hvs * sizeof(float));
    *kv_vseq += hvs;
}

static void taomate_append_audio_kv(float *kv_a, float *kv_apt, float *kv_aph,
                                    float *kv_apw, size_t *kv_aseq,
                                    const float *audio_rows, const float *apt,
                                    const float *aph, const float *apw,
                                    int audio_t, int a0, int a1) {
    size_t has;
    if (!kv_a || !kv_aseq || a1 <= a0)
        return;
    has = (size_t)(a1 - a0);
    taomate_copy_audio_t(audio_rows,
                         kv_a + (*kv_aseq) * (size_t)H3_DIT_W4A8_AUDIO_CH,
                         audio_t, a0, a1);
    memcpy(kv_apt + *kv_aseq, apt + (size_t)a0, has * sizeof(float));
    memcpy(kv_aph + *kv_aseq, aph + (size_t)a0, has * sizeof(float));
    memcpy(kv_apw + *kv_aseq, apw + (size_t)a0, has * sizeof(float));
    memcpy(kv_apt + *kv_aseq + has, apt + (size_t)audio_t + (size_t)a0,
           has * sizeof(float));
    memcpy(kv_aph + *kv_aseq + has, aph + (size_t)audio_t + (size_t)a0,
           has * sizeof(float));
    memcpy(kv_apw + *kv_aseq + has, apw + (size_t)audio_t + (size_t)a0,
           has * sizeof(float));
    *kv_aseq += has * 2u;
}

/* Bounded overlap cond (default). Full-chunk CLEAN_KV is opt-in and can OOM. */
static void taomate_pack_history(const h3_taomate_plan *plan, int pi,
                                 int clean_kv, int hist_t, int sink_t,
                                 const float *latent, const float *pt,
                                 const float *ph, const float *pw,
                                 const float *audio_rows, const float *apt,
                                 const float *aph, const float *apw,
                                 int pack_audio, int audio_t, size_t spatial,
                                 size_t t_floats, float *kv_v, float *kv_pt,
                                 float *kv_ph, float *kv_pw, float *kv_a,
                                 float *kv_apt, float *kv_aph, float *kv_apw,
                                 size_t *kv_vseq, size_t *kv_aseq) {
    *kv_vseq = 0;
    *kv_aseq = 0;
    if (!plan || pi <= 0 || !kv_v)
        return;
    if (clean_kv) {
        int hist[4];
        int nhist = 0;
        int hi;
        hist[nhist++] = 0;
        if (pi - 1 != 0)
            hist[nhist++] = pi - 1;
        if (pi >= 2 && (pi - 2) != 0 && (pi - 2) != (pi - 1))
            hist[nhist++] = pi - 2;
        for (hi = 0; hi < nhist; hi++) {
            const h3_taomate_phase *hp = &plan->phases[hist[hi]];
            taomate_append_video_kv(kv_v, kv_pt, kv_ph, kv_pw, kv_vseq, latent,
                                    pt, ph, pw, hp->video_t0, hp->video_t1,
                                    spatial, t_floats);
            if (pack_audio && kv_a && hp->audio_t1 > hp->audio_t0)
                taomate_append_audio_kv(kv_a, kv_apt, kv_aph, kv_apw, kv_aseq,
                                        audio_rows, apt, aph, apw, audio_t,
                                        hp->audio_t0, hp->audio_t1);
        }
        return;
    }
    if (hist_t <= 0 && sink_t <= 0)
        return;
    {
        h3_taomate_ov_span sp[H3_TAOMATE_MAX_OV_SPANS];
        int ns = h3_taomate_overlap_spans(plan, pi, hist_t, sink_t, sp);
        int si;
        for (si = 0; si < ns; si++) {
            taomate_append_video_kv(kv_v, kv_pt, kv_ph, kv_pw, kv_vseq, latent,
                                    pt, ph, pw, sp[si].video_t0, sp[si].video_t1,
                                    spatial, t_floats);
            if (pack_audio && kv_a && sp[si].audio_t1 > sp[si].audio_t0)
                taomate_append_audio_kv(kv_a, kv_apt, kv_aph, kv_apw, kv_aseq,
                                        audio_rows, apt, aph, apw, audio_t,
                                        sp[si].audio_t0, sp[si].audio_t1);
        }
    }
}

static size_t taomate_phase_packed_seq(const h3_taomate_plan *plan, int pi,
                                       size_t context, size_t spatial,
                                       int pack_audio, int clean_kv, int hist_t,
                                       int sink_t) {
    const h3_taomate_phase *ph;
    size_t vs, as, extra_v = 0, extra_a = 0;
    int t0w, a0w;
    int denoise_ov = clean_kv ? 0 : hist_t;
    int cond_hist = clean_kv ? hist_t : 0;
    if (!plan || pi < 0 || pi >= plan->n_phases)
        return context;
    ph = &plan->phases[pi];
    t0w = h3_taomate_denoise_t0(plan, pi, denoise_ov);
    a0w = h3_taomate_denoise_a0(plan, pi, denoise_ov);
    vs = (size_t)(ph->video_t1 - t0w) * spatial;
    as = pack_audio ? (size_t)(ph->audio_t1 - a0w) * 2u : 0;
    /* Cond tokens: sink (identity) and/or full-chunk CLEAN_KV. Not full-clip Q. */
    if (pi > 0 && (cond_hist > 0 || sink_t > 0)) {
        h3_taomate_ov_span sp[H3_TAOMATE_MAX_OV_SPANS];
        int ns = h3_taomate_overlap_spans(plan, pi, cond_hist, sink_t, sp);
        int si;
        for (si = 0; si < ns; si++) {
            extra_v += (size_t)(sp[si].video_t1 - sp[si].video_t0) * spatial;
            extra_a += pack_audio
                           ? (size_t)(sp[si].audio_t1 - sp[si].audio_t0) * 2u
                           : 0;
        }
    }
    return context + vs + as + extra_v + extra_a;
}

static void taomate_blend_video_overlap(float *latent, const float *v_chunk,
                                        int t0_ov, int t0, int t1,
                                        size_t t_floats) {
    int t, ov = t0 - t0_ov;
    size_t i;
    if (!latent || !v_chunk || t1 <= t0)
        return;
    for (t = 0; t < ov; t++) {
        float w = (float)(t + 1) / (float)(ov + 1);
        const float *ns = v_chunk + (size_t)t * t_floats;
        float *ds = latent + (size_t)(t0_ov + t) * t_floats;
        for (i = 0; i < t_floats; i++)
            ds[i] = (1.0f - w) * ds[i] + w * ns[i];
    }
    memcpy(latent + (size_t)t0 * t_floats,
           v_chunk + (size_t)ov * t_floats,
           (size_t)(t1 - t0) * t_floats * sizeof(float));
}

static void taomate_blend_audio_overlap(float *dst, const float *src_win,
                                        int audio_t, int a0_ov, int a0,
                                        int a1) {
    int n_win = a1 - a0_ov;
    int ov = a0 - a0_ov;
    int n_new = a1 - a0;
    size_t ch = (size_t)H3_DIT_W4A8_AUDIO_CH;
    int t, s, i;
    if (!dst || !src_win || n_new < 0)
        return;
    for (s = 0; s < 2; s++) {
        float *dstream = dst + (size_t)(s ? audio_t : 0) * ch;
        const float *sstream = src_win + (size_t)(s ? n_win : 0) * ch;
        for (t = 0; t < ov; t++) {
            float w = (float)(t + 1) / (float)(ov + 1);
            float *d = dstream + (size_t)(a0_ov + t) * ch;
            const float *n = sstream + (size_t)t * ch;
            for (i = 0; i < (int)ch; i++)
                d[i] = (1.0f - w) * d[i] + w * n[i];
        }
        if (n_new > 0)
            memcpy(dstream + (size_t)a0 * ch, sstream + (size_t)ov * ch,
                   (size_t)n_new * ch * sizeof(float));
    }
}

/* Dual-sample on TaoMate 3-step: x0 from noisy x+σv, LBH, re-noise, euler.
 * already_x0=1: latent is already clean x0 (all Euler at low-res); LBH then
 * decode, no re-noise / high DiT. Return 1=continue high-res, 2=Z0 stop, 0=fail. */
static int generate_lbh_transition(
    float **latent, float **vel, float **pt, float **ph, float **pw,
    float **apt, float **aph, float **apw, size_t *n_lat, size_t *video_seq,
    int *latent_h, int *latent_w, int latent_t, int hi_lh, int hi_lw,
    int target_rope_t0, float sigma_k, float sigma_next, int pack_audio,
    int audio_t, size_t n_audio, float *audio_rows, float *audio_vel,
    float audio_sigma_k, float audio_sigma_next, unsigned long long seed,
    const float *lift_mean, const float *lift_std, h3_latent_upscaler *lat_up,
    h3_cuda *cuda, h3_dit_w4a8 *dit, size_t context, size_t audio_seq,
    int already_x0, double *out_ms, char *error, size_t error_size) {
    float *x0_pack = NULL, *z_low = NULL, *z_high = NULL, *z0_pack = NULL;
    float *noise_hi = NULL;
    int lo_h = *latent_h, lo_w = *latent_w;
    size_t n_hi, video_seq_hi, n_tok_hi, ii;
    double t_lift;
    int z0_only = 0;
    const char *z0env;

    if (out_ms)
        *out_ms = 0.0;
    if (!latent || !*latent || !lat_up || !lift_mean || !lift_std)
        return 0;
    if (!already_x0 && (!vel || !*vel))
        return 0;
    if (!already_x0 && !(sigma_k > 1e-8f)) {
        snprintf(error, error_size, "lbh: bad sigma");
        return 0;
    }
    t_lift = now_ms();
    if (!already_x0 && pack_audio && audio_rows && audio_vel && n_audio) {
        if (!h3_dit_w4a8_euler_step_f32(audio_rows, audio_vel, n_audio,
                                        audio_sigma_k, audio_sigma_next, error,
                                        error_size))
            return 0;
    }
    x0_pack = (float *)malloc(*n_lat * sizeof(float));
    if (!x0_pack) {
        snprintf(error, error_size, "OOM lift x0");
        return 0;
    }
    if (already_x0) {
        memcpy(x0_pack, *latent, *n_lat * sizeof(float));
    } else {
        for (ii = 0; ii < *n_lat; ii++)
            x0_pack[ii] = (*latent)[ii] + sigma_k * (*vel)[ii];
    }
    z_low = h3_vae_int8_unpack_dit_latent_f32(x0_pack, latent_t, lo_h, lo_w,
                                              error, error_size);
    free(x0_pack);
    x0_pack = NULL;
    if (!z_low)
        return 0;
    dump_chw_channel_stats("z_low", z_low, latent_t, lo_h, lo_w);
    z_high = (float *)malloc((size_t)H3_VAE_INT8_LATENT_CH * (size_t)latent_t *
                             (size_t)hi_lh * (size_t)hi_lw * sizeof(float));
    if (!z_high) {
        free(z_low);
        snprintf(error, error_size, "OOM lift z_high");
        return 0;
    }
    if (!h3_latent_upscaler_forward(lat_up, cuda, z_low, latent_t, lo_h, lo_w,
                                    hi_lh, hi_lw, lift_mean, lift_std, z_high,
                                    error, error_size)) {
        free(z_low);
        free(z_high);
        return 0;
    }
    {
        float r_lo = chw_adj_ratio(z_low, latent_t, lo_h, lo_w);
        float r_hi = chw_adj_ratio(z_high, latent_t, hi_lh, hi_lw);
        printf("selflift: chw_adj ch0/t0 low=%.4g high=%.4g "
               "(~1=checkerboard, ~0.1=smooth)\n",
               (double)r_lo, (double)r_hi);
        dump_chw_channel_stats("z_high", z_high, latent_t, hi_lh, hi_lw);
        fflush(stdout);
    }
    free(z_low);
    z_low = NULL;
    video_seq_hi = (size_t)latent_t * (size_t)(hi_lh / 2) * (size_t)(hi_lw / 2);
    n_tok_hi = video_seq_hi * (size_t)H3_DIT_W4A8_VIDEO_PATCH;
    z0_pack = (float *)malloc(n_tok_hi * sizeof(float));
    if (!z0_pack ||
        !h3_vae_patchify_video_96(z_high, latent_t, hi_lh, hi_lw, z0_pack,
                                  n_tok_hi)) {
        snprintf(error, error_size, "lift pack failed");
        free(z_high);
        free(z0_pack);
        return 0;
    }
    free(z_high);
    z_high = NULL;
    n_hi = n_tok_hi;
    if (!grow_f32(latent, n_hi, error, error_size) ||
        !grow_f32(vel, n_hi, error, error_size) ||
        !grow_f32(pt, video_seq_hi, error, error_size) ||
        !grow_f32(ph, video_seq_hi, error, error_size) ||
        !grow_f32(pw, video_seq_hi, error, error_size)) {
        free(z0_pack);
        return 0;
    }
    z0env = getenv("H3_LIFT_Z0_ONLY");
    z0_only = already_x0 || (z0env && z0env[0] == '1') ? 1 : 0;
    if (z0_only) {
        memcpy(*latent, z0_pack, n_hi * sizeof(float));
        *latent_h = hi_lh;
        *latent_w = hi_lw;
        *video_seq = video_seq_hi;
        *n_lat = n_hi;
        free(z0_pack);
        printf("selflift: Z0_ONLY decode lifted endpoint (no remaining DiT)\n");
        if (out_ms)
            *out_ms = now_ms() - t_lift;
        return 2;
    }
    noise_hi = (float *)malloc(n_hi * sizeof(float));
    if (!noise_hi) {
        free(z0_pack);
        snprintf(error, error_size, "OOM lift noise");
        return 0;
    }
    fill_normal_f32(seed + 1ULL, noise_hi, n_hi);
    for (ii = 0; ii < n_hi; ii++) {
        float z0 = z0_pack[ii];
        (*latent)[ii] = sigma_k * noise_hi[ii] + (1.0f - sigma_k) * z0;
        (*vel)[ii] = (z0 - (*latent)[ii]) / sigma_k;
    }
    free(noise_hi);
    free(z0_pack);
    if (!h3_dit_w4a8_euler_step_f32(*latent, *vel, n_hi, sigma_k, sigma_next,
                                    error, error_size))
        return 0;
    *latent_h = hi_lh;
    *latent_w = hi_lw;
    *video_seq = video_seq_hi;
    *n_lat = n_hi;
    if (!h3_dit_w4a8_fill_metal_video_pos((size_t)target_rope_t0, latent_t,
                                          *latent_h, *latent_w, *pt, *ph, *pw,
                                          error, error_size))
        return 0;
    if (pack_audio && apt && aph && apw && *apt && *aph && *apw) {
        if (!h3_dit_w4a8_fill_metal_audio_pos((size_t)target_rope_t0, audio_t,
                                              *latent_h, *latent_w, *apt, *aph,
                                              *apw, error, error_size))
            return 0;
    }
    (void)context;
    (void)audio_seq;
    if (dit && cuda) {
        h3_dit_w4a8_release_block_workspace(dit);
        h3_cuda_release_heavy_workspace(cuda);
    }
    printf("selflift: transition σ=%.4g→%.4g latent %dx%d→%dx%d tokens %zu "
           "lift_ms=%.1f (TaoMate 3-step; no extra DiT NFE)\n",
           (double)sigma_k, (double)sigma_next, lo_h, lo_w, hi_lh, hi_lw, *n_lat,
           now_ms() - t_lift);
    fflush(stdout);
    if (out_ms)
        *out_ms = now_ms() - t_lift;
    (void)audio_t;
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

static int gen_uly_a2a_dtype(void) {
    const char *e = getenv("H3_ULYSSES_A2A_F32");
    if (e && e[0] == '1' && e[1] == '\0')
        return 4;
    return 2; /* Phase 2: BF16 A2A default. Gold U=1 never calls this. */
}

static int gen_uly_a2a_qkv(void *cfg, size_t seq_local, size_t heads,
                           size_t head_dim, char *err, size_t err_size) {
    return h3_ulysses_a2a_qkv((h3_ulysses_cfg *)cfg, NULL, NULL, seq_local,
                              heads, head_dim, gen_uly_a2a_dtype(), err,
                              err_size);
}

static int gen_uly_a2a_attn(void *cfg, size_t seq_local, size_t heads,
                            size_t head_dim, char *err, size_t err_size) {
    return h3_ulysses_a2a_attn_out((h3_ulysses_cfg *)cfg, NULL, NULL, seq_local,
                                   heads, head_dim, gen_uly_a2a_dtype(), err,
                                   err_size);
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
    /* True-FP32 VAE SDPA: TF32 must be off before the CUDA context exists. */
    {
        const char *vae_f32 = getenv("H3_VAE_SDPA_F32");
        if (vae_f32 && vae_f32[0] == '1' && vae_f32[1] == '\0')
            setenv("NVIDIA_TF32_OVERRIDE", "0", 1);
    }
    const char *dit_path = NULL, *vae_path = NULL, *te_path = NULL;
    const char *audio_vae_path = NULL;
    char audio_vae_resolved[1024];
    char mux_resolved[1024];
    int audio_vae_explicit = 0;
    const char *out_dir = "out_cuda_t2v";
    const char *sdpa = "cudnn";
    char sdpa_effective[64] = "";
    const char *preset_name = NULL;
    const char *lora_path = NULL;
    const char *fewstep_contract_cli = NULL;
    const char *fewstep_preset_cli = NULL;
    h3_fewstep_id fewstep_id = H3_FEWSTEP_NONE;
    const char *sampler_cli = NULL;
    int sampler_explicit = 0;
    h3_sampler_id sampler_id = H3_SAMPLER_VELOCITY_EULER;
    float *renoise_scratch = NULL;
    size_t renoise_scratch_n = 0;
    const char *vae_decode = "official";
    const char *tae_path = NULL;
    const char *te_cache_dir = NULL;
    const char *latent_upscaler_path = NULL;
    int lift_transition_cli = 0;
    int lift_on = 0;
    int lift_k = 0;
    int hi_lh = 0, hi_lw = 0;
    h3_latent_upscaler *lat_up = NULL;
    /* Cold-clip waffle observability gate (direction A): clip1 max_b16 uses
     * an independent, looser threshold than the hot golden line. WARN-only —
     * never FAILs, never alters generation. H3_WAFFLE_COLD_GUARD=1 replaces
     * the free()'d clip1 mux frames with a bit-identical re-decode before the
     * clip1 mux write (an H3_WAFFLE_GUARD_RETRY-style retry for the cold
     * clip; it hides the flaky decode from the muxed mp4 but the WARN is
     * still logged with the pre-retry stats). */
    int waffle_cold_guard = 0;
    double waffle_cold_max_b16 = 0.0;
    float lift_mean[H3_VAE_INT8_LATENT_CH];
    float lift_std[H3_VAE_INT8_LATENT_CH];
    int lift_stats_ok = 0;
    float lora_scale = 1.0f;
    int steps_explicit = 0, sdpa_explicit = 0;
    int ulysses = 1;
    h3_generate_preset_out preset_out;
    h3_lora_set lora_set;
    int lora_loaded = 0;
    /* Opt-in 主体+收尾 LoRA 混合调度（taomate refiner A/B；默认关）。
     * H3_REFINER_LORA=FILE + H3_REFINER_LORA_STEPS=N + H3_REFINER_LORA_SCALE=S:
     * 主循环末 N 步（步号 >= steps-N）前临时切到该 LoRA，其余步保持 --lora 主体。 */
    h3_lora_set refiner_lora_set;
    int refiner_lora_loaded = 0;
    int refiner_lora_attached = 0;
    int refiner_lora_steps = 0;
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
    const char *report_path = NULL;
    unsigned dit_layers = 50;
    unsigned vae_blocks = 36;
    unsigned te_layers = 50;
    int vae_repeats = 1;
    int clip_repeat = 1;
    int clip_i = 0;
    int vae_opened = 0;
    int orig_lh = 0, orig_lw = 0;
    size_t orig_n_lat = 0, orig_vseq = 0;
    char mux_clip_path[1024];
    const char *mux_write = NULL;
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
    const char *dump_latent_path = getenv("H3_DUMP_LATENT");
    const char *replay_latent_path = getenv("H3_REPLAY_LATENT");
    int replay_latent = replay_latent_path && replay_latent_path[0];
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
    h3_taomate_plan tm_plan;
    int tm_plan_ok = 0;
    /* Packed-Q CLEAN_KV concatenates history into Q (OOM at 768P).
     * Default overlap is denoise-window blend. H3_TAOMATE_LAYER_KV=1 is the
     * real TaoMate hook: Q stays the chunk, K/V carry sink+recent. */
    int tm_clean_kv = 0;
    int tm_layer_kv = 0;
    int tm_hist_t = 2;
    int tm_sink_t = 0;
    size_t tm_kv_sink_n = 0;
    char step_cache_err[256];

    h3_w4a8_store *dit_store = NULL;
    h3_dit_int8_store *dit_int8 = NULL;
    h3_nvfp4_store dit_nvfp4;
    int dit_is_int8 = 0;
    int dit_is_nvfp4 = 0;
    int dit_nvfp4_opened = 0;
    void *tae_keep = NULL;
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
    double t_lift_switch = 0;
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
    int audio_overlap = 0;
    sdpa_prebuild_job sdpa_job;
    audio_decode_job audio_job;
    h3_vae_prewarm_job vae_prewarm_job;
    double vae_prewarm_t0 = 0.0;
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
    memset(&vae_prewarm_job, 0, sizeof(vae_prewarm_job));
    memset(&host_stage_job, 0, sizeof(host_stage_job));
    memset(&host_stage_kick_job, 0, sizeof(host_stage_kick_job));
    memset(&preset_out, 0, sizeof(preset_out));
    memset(&dit_nvfp4, 0, sizeof(dit_nvfp4));
    dit_nvfp4.map_fd = -1;
    memset(&lora_set, 0, sizeof(lora_set));
    memset(&refiner_lora_set, 0, sizeof(refiner_lora_set));
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
        }         else if (!strcmp(argv[iarg], "--repeat") && iarg + 1 < argc) {
            clip_repeat = (int)strtol(argv[++iarg], NULL, 10);
            if (clip_repeat < 1) clip_repeat = 1;
            /* R54 wants --repeat 30. Silent 8-cap made 30 a lie. */
            if (clip_repeat > 32) clip_repeat = 32;
            {
                /* Registered in tools/h3_plugin.h; read via getenv here (the
                 * registry helper is static to its header and has no linkable
                 * TU). Semantics match h3_plugin_enabled. */
                const char *wcg = getenv("H3_WAFFLE_COLD_GUARD");
                waffle_cold_guard =
                    wcg && wcg[0] && strcmp(wcg, "0") != 0 &&
                    strcmp(wcg, "false") != 0 && strcmp(wcg, "off") != 0 &&
                    strcmp(wcg, "no") != 0;
            }
            waffle_cold_max_b16 = h3_kpi_guard_cold_max_b16_threshold();
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
        } else if (!strcmp(argv[iarg], "--report") && iarg + 1 < argc) {
            report_path = argv[++iarg];
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
        else if (!strcmp(argv[iarg], "--fewstep-contract") &&
                   iarg + 1 < argc)
            fewstep_contract_cli = argv[++iarg];
        else if (!strcmp(argv[iarg], "--fewstep-preset") &&
                   iarg + 1 < argc)
            fewstep_preset_cli = argv[++iarg];
        else if (!strcmp(argv[iarg], "--sampler") && iarg + 1 < argc) {
            sampler_cli = argv[++iarg];
            sampler_explicit = 1;
        } else if (!strcmp(argv[iarg], "--ulysses") && iarg + 1 < argc)
            ulysses = (int)strtol(argv[++iarg], NULL, 10);
        else if (!strcmp(argv[iarg], "--vae-decode") && iarg + 1 < argc)
            vae_decode = argv[++iarg];
        else if (!strcmp(argv[iarg], "--tae") && iarg + 1 < argc)
            tae_path = argv[++iarg];
        else if (!strcmp(argv[iarg], "--te-cache") && iarg + 1 < argc)
            te_cache_dir = argv[++iarg];
        else if (!strcmp(argv[iarg], "--latent-upscaler") && iarg + 1 < argc)
            latent_upscaler_path = argv[++iarg];
        else if (!strcmp(argv[iarg], "--lift-transition") && iarg + 1 < argc)
            lift_transition_cli = (int)strtol(argv[++iarg], NULL, 10);
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
                    "  --repeat N         In-process clip loop after weights "
                    "load (default 1, max 32). Reuses LoRA / INT8 pin / sage3 / DiT; "
                    "does not re-pin. Hot clip i>1 is the serve-style e2e. "
                    "Mux gets _rK suffix when N>1. Gold default stays 1.\n"
                    "  --probe-vae-load   Only measure VAE weights_load path\n"
                    "  --report FILE.json  Write KPI JSON (clip N → FILE.clipN.json)\n"
                    "  --seconds N        Duration at 24 fps (default 5). "
                    "Frame count = N*24, aligned up to 5+17k. "
                    "Do not pass --frames if you want a real N-second mp4.\n"
                    "  --frames N         Override frame count (NOT duration). "
                    "Mux is always 24 fps, so --frames 17 → ~0.92s even with "
                    "--seconds 5. For 5s/10s product clips omit this.\n"
                    "  --seed N           Noise seed (default 0x483343554441)\n"
                    "  --preset gold|turbo4|nvfp4|lift|taomate|taomate-lift\n"
                    "      gold（可省略）= 今日金线路径，不改 steps/backend/cache。\n"
                    "      turbo4 = 非金线极速档：steps=4 + sage3 + 要求 --lora；cache 关。\n"
                    "      taomate = 非金线：TaoMate 3-step 锚点 {0,16,33,49} +\n"
                    "              HF TaoLiveAIGC/TaoMate-H3 LoRA。\n"
                    "              duration≤5s = Comfy 整段一次采（1 phase）；\n"
                    "              10s+ 仍 codec chunk。H3_TAOMATE_FORCE_CHUNKS=1\n"
                    "              恢复 5s 切块。须 --lora 或 H3_TAOMATE_LORA。\n"
                    "              禁止 4-step。--steps 6 = 50 格中点（非 turbo4）。\n"
                    "      taomate-lift = 同上 3-step LoRA，但 2 NFE 走 672×384 +\n"
                    "              LBH + 1 NFE 高分收尾。须 --latent-upscaler。\n"
                    "              5s 同样整段一次，不切块。\n"
                    "              H3_LIFT_Z0_ONLY=1 → 3 NFE 全低分 + LBH + 0 高分 DiT。\n"
                    "              不是 turbo4 lift。金线不走此档。\n"
                    "      nvfp4 = 非金线加速档：steps=8 + cudnn，须 --dit pruned_nvfp4；\n"
                    "              例 minimax_h3_fl2va_pruned_nvfp4.safetensors。\n"
                    "              探测失败硬退出，不回退 INT8。可能有损，不是金线。\n"
                    "      lift / self-lift = 非金线加速档：LBH 3D LatentResizer\n"
                    "              接 Euler mid-σ（默认 8step、transition=6）。\n"
                    "              必须 --latent-upscaler FILE；禁止 nearest-only。\n"
                    "              金线 768P 仍走原生 8step，不传此 preset。\n"
                    "      未蒸馏 --steps 4 而无 LoRA 不是产品。缺省不传 preset。\n"
                    "      --preset fast4 已更名为 turbo4。\n"
                    "  --lora FILE.safetensors   turbo4 必填；金线禁止依赖此文件\n"
                    "  --lora-scale S            默认 1.0\n"
                    "  --fewstep-contract pdmd4|pdmd2|dmad4\n"
                    "      少步合同自检（或 H3_FEWSTEP_CONTRACT）。未设=off。\n"
                    "      错 steps/shift/preset(turbo4)/sampler 硬 FAIL。\n"
                    "      dmad4 需要 --sampler lcm_renoise（未显式设时自动选）。\n"
                    "      H3_AUDIO_SIGMA_SHIFT=2；H3_FEWSTEP_STRICT=1；\n"
                    "      H3_FEWSTEP_ALLOW_NVFP4=1。\n"
                    "  --fewstep-preset pdmd4|pdmd2|dmad4\n"
                    "      一键绑合同：steps/cfg/shift/sampler/--fewstep-contract。\n"
                    "      或 H3_FEWSTEP_PRESET=。勿与 --preset turbo4/taomate 叠。\n"
                    "      pdmd4: 4-step Euler shift 12/3 cfg1；pdmd2: 2-step；\n"
                    "      dmad4: 4-step lcm_renoise shift 12/2 cfg1。\n"
                    "      Rank (beyond-DMAD): prefer pdmd4 over dmad4 at same\n"
                    "      NFE when quality matters (paper auto-score + OR).\n"
                    "      仍须 --lora 原件；显式 --steps/--cfg/--sampler 优先。\n"
                    "  --sampler velocity_euler|lcm_renoise\n"
                    "      或 H3_SAMPLER=。默认 velocity_euler。\n"
                    "      lcm_renoise = Comfy lcm+simple（DMAD 合同）。\n"
                    "      别名：euler / lcm / lcm+simple。\n"
                    "  --latent-upscaler FILE    lift / taomate-lift 必填；LBH 3D\n"
                    "  --lift-transition K       低分 DiT 次数（8step 默认 6；4step=3；\n"
                    "                            taomate 3-step 默认 2；Z0_ONLY 时=3）\n"
                    "  --ulysses N        Sequence-parallel Ulysses (default 1).\n"
                    "                     Experimental. Legal {1,2,4,8}; U=7 illegal.\n"
                    "                     N=1 does not init NCCL. Does not change sdpa.\n"
                    "                     N>1 drives DiT A2A after QKV+LoRA; SDPA uses\n"
                    "                     S_real (pad rows excluded). Gold default stays 1.\n"
                    "  --te-cache DIR     Opt-in TE prompt embedding disk cache.\n"
                    "                     H3_TE_CACHE_AUTO=1 + --repeat>=2\n"
                    "  H3_VRAM_LEDGER=1   C040 cudaMemGetInfo buckets (after_cuda..after_vae)\n"
                    "                     → auto dir out/te_cache_auto (C062).\n"
                    "                     H3_CUDA_L2_PIN=1 → pin Hadamard L2 (C017).\n                     H3_ROPE_PROFILE=1 → dit/vae rope ms + memo hits.\n                     H3_ROPE_MEMO=0 → disable RoPE host memo (KPI).\n"
                    "                     Or H3_TE_PROMPT_CACHE. Default off (金线不变).\n"
                    "                     H3_TE_CACHE_REQUIRE_HIT=1 → MISS is FAIL_ENV.\n"
                    "                     Text-only; vision/multimodal never cached.\n"
                    "  --vae-decode official|tae  Default official（金线必须 official）。\n"
                    "                     tae 仅 turbo4/nvfp4 快档 opt-in：非金线、画质有损\n"
                    "                     （纹理糊/色偏）；需 --tae 权，禁止静默 official。\n"
                    "  --tae FILE         TAEHV taeh3 权（快档；写了即走 --vae-decode tae）。\n"
                    "                     非无损。金线开 tae 硬失败 RC=1。\n"
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
        if (preset_out.id == H3_PRESET_LIFT || preset_out.lift)
            lift_on = 1;
        if (preset_out.id == H3_PRESET_TAOMATE && latent_upscaler_path &&
            latent_upscaler_path[0])
            lift_on = 1;
        if (preset_out.lift && !lift_on) {
            fprintf(stderr,
                    "h3: preset=taomate-lift 必须 --latent-upscaler FILE "
                    "（或 H3_LATENT_UPSCALER）\n");
            return 1;
        }
        if (preset_out.non_gold)
            fprintf(stderr,
                    "h3: preset=%s 非金线（MAE 金线不适用）\n",
                    preset_name && preset_name[0] ? preset_name : "fast");
    }
    /* One-shot fewstep preset: expand steps/cfg/audio shift/contract/sampler. */
    {
        char perr[384];
        h3_fewstep_preset_in pin;
        h3_fewstep_preset_out pout;
        h3_fewstep_id contract_pre = H3_FEWSTEP_NONE;
        int contract_pre_set = 0;
        perr[0] = '\0';
        if (fewstep_contract_cli && fewstep_contract_cli[0]) {
            contract_pre = h3_fewstep_contract_parse(fewstep_contract_cli, perr,
                                                     sizeof(perr));
            if (perr[0]) {
                fprintf(stderr, "h3: %s\n", perr);
                h3_fail_tag("FAIL_CONTRACT");
                return 1;
            }
            contract_pre_set = 1;
        } else if (getenv("H3_FEWSTEP_CONTRACT") &&
                   getenv("H3_FEWSTEP_CONTRACT")[0]) {
            contract_pre = h3_fewstep_contract_resolve(NULL, perr, sizeof(perr));
            if (perr[0]) {
                fprintf(stderr, "h3: %s\n", perr);
                h3_fail_tag("FAIL_CONTRACT");
                return 1;
            }
            contract_pre_set = 1;
        }
        memset(&pin, 0, sizeof(pin));
        pin.name = fewstep_preset_cli;
        pin.steps_explicit = steps_explicit;
        pin.steps = steps;
        pin.cfg_explicit = cfg_explicit;
        pin.cfg_scale = cfg_scale;
        pin.sampler_explicit = sampler_explicit;
        pin.contract_already_set = contract_pre_set;
        pin.contract_already = contract_pre;
        if (!h3_fewstep_preset_apply(&pin, &pout, perr, sizeof(perr))) {
            fprintf(stderr, "h3: %s\n", perr);
            h3_fail_tag("FAIL_CONTRACT");
            return 1;
        }
        if (pout.applied) {
            if (preset_out.id == H3_PRESET_TURBO4 ||
                preset_out.id == H3_PRESET_TAOMATE) {
                fprintf(stderr,
                        "h3: --fewstep-preset %s 与 --preset %s 互斥"
                        "（会改写 sigma；请无 speed preset + serving shift）\n",
                        h3_fewstep_contract_name(pout.id),
                        preset_name && preset_name[0] ? preset_name : "?");
                h3_fail_tag("FAIL_CONTRACT");
                return 1;
            }
            steps = pout.steps;
            cfg_scale = pout.cfg_scale;
            if (!cfg_explicit)
                cfg_explicit = 1; /* treat as bound so later F2VA default skips */
            if (pout.set_audio_env && pout.audio_env_value) {
                /* setenv(...,0) keeps a pre-set env; refuse wrong shift early. */
                if (!h3_fewstep_preset_audio_env_compatible(pout.audio_shift,
                                                            perr,
                                                            sizeof(perr))) {
                    fprintf(stderr, "h3: %s\n", perr);
                    h3_fail_tag("FAIL_ENV");
                    return 1;
                }
                setenv("H3_AUDIO_SIGMA_SHIFT", pout.audio_env_value, 0);
            }
            if (!fewstep_contract_cli || !fewstep_contract_cli[0])
                fewstep_contract_cli = h3_fewstep_contract_name(pout.id);
            if (!sampler_explicit && pout.want_lcm_renoise) {
                sampler_cli = "lcm_renoise";
                /* still not "explicit" so dmad auto path also works */
            }
            h3_fewstep_preset_print(&pout);
        }
    }
    {
        char serr[256];
        serr[0] = '\0';
        if (!sampler_explicit && getenv("H3_SAMPLER") && getenv("H3_SAMPLER")[0])
            sampler_explicit = 1;
        sampler_id = h3_sampler_resolve(sampler_cli, serr, sizeof(serr));
        if (serr[0]) {
            fprintf(stderr, "h3: %s\n", serr);
            h3_fail_tag("FAIL_CONTRACT");
            return 1;
        }
    }
    {
        char ferr[384];
        h3_fewstep_contract_in fin;
        int want_nvfp4;
        ferr[0] = '\0';
        fewstep_id = h3_fewstep_contract_resolve(fewstep_contract_cli, ferr,
                                                 sizeof(ferr));
        if (ferr[0]) {
            fprintf(stderr, "h3: %s\n", ferr);
            h3_fail_tag("FAIL_CONTRACT");
            return 1;
        }
        /* dmad4 defaults to lcm_renoise when user did not pick a sampler. */
        if (fewstep_id == H3_FEWSTEP_DMAD4 && !sampler_explicit)
            sampler_id = H3_SAMPLER_LCM_RENOISE;
        if (sampler_id == H3_SAMPLER_LCM_RENOISE &&
            (lift_on || preset_out.id == H3_PRESET_TAOMATE)) {
            fprintf(stderr,
                    "h3: --sampler lcm_renoise 与 lift/taomate 不兼容"
                    "（主路径 fewstep 专用）\n");
            h3_fail_tag("FAIL_CONTRACT");
            return 1;
        }
        memset(&fin, 0, sizeof(fin));
        fin.id = fewstep_id;
        fin.steps = steps;
        fin.preset_name = preset_name;
        fin.preset_rewrites_sigma =
            (preset_out.id == H3_PRESET_TURBO4 ||
             preset_out.id == H3_PRESET_TAOMATE) ? 1 : 0;
        fin.lora_path = lora_path;
        fin.video_shift = H3_CUDA_VIDEO_SIGMA_SHIFT;
        fin.audio_shift = audio_sigma_shift_effective();
        fin.cfg_scale = cfg_scale;
        want_nvfp4 = (preset_out.id == H3_PRESET_NVFP4) ||
                     (getenv("H3_DIT_NVFP4") &&
                      getenv("H3_DIT_NVFP4")[0] == '1');
        fin.nvfp4_enabled = want_nvfp4 ? 1 : 0;
        fin.sampler_is_velocity_euler =
            (sampler_id == H3_SAMPLER_VELOCITY_EULER) ? 1 : 0;
        fin.has_lcm_renoise =
            (sampler_id == H3_SAMPLER_LCM_RENOISE) ? 1 : 0;
        if (!h3_fewstep_contract_check_early(&fin, ferr, sizeof(ferr))) {
            fprintf(stderr, "h3: %s\n", ferr);
            h3_fail_tag("FAIL_CONTRACT");
            return 1;
        }
        printf("sampler: selected=%s (explicit=%d fewstep=%s)\n",
               h3_sampler_name(sampler_id), sampler_explicit,
               h3_fewstep_contract_name(fewstep_id));
        if (fewstep_id != H3_FEWSTEP_NONE) {
            printf("fewstep_bound: contract=%s steps=%d cfg=%.3g "
                   "video_shift=%.3g audio_shift=%.3g sampler=%s "
                   "speed_preset=%s\n",
                   h3_fewstep_contract_name(fewstep_id), steps,
                   (double)cfg_scale, (double)H3_CUDA_VIDEO_SIGMA_SHIFT,
                   (double)audio_sigma_shift_effective(),
                   h3_sampler_name(sampler_id),
                   preset_name && preset_name[0] ? preset_name : "none");
        }
        fflush(stdout);
    }
    /* C061: inline env contract after CLI/fewstep resolve, before heavy work. */
    {
        h3_env_contract_cli ecl;
        int want_nvfp4_arm =
            (preset_out.id == H3_PRESET_NVFP4) ||
            (getenv("H3_DIT_NVFP4") && getenv("H3_DIT_NVFP4")[0] == '1');
        memset(&ecl, 0, sizeof(ecl));
        ecl.seed = seed;
        ecl.skip_audio = skip_audio;
        ecl.fewstep_preset =
            (fewstep_preset_cli && fewstep_preset_cli[0]) ? fewstep_preset_cli
            : (getenv("H3_FEWSTEP_PRESET") && getenv("H3_FEWSTEP_PRESET")[0])
                  ? getenv("H3_FEWSTEP_PRESET")
                  : "none";
        ecl.fewstep_contract = h3_fewstep_contract_name(fewstep_id);
        ecl.sampler = h3_sampler_name(sampler_id);
        ecl.sdpa = sdpa;
        ecl.arm = want_nvfp4_arm ? "nvfp4" : "int8";
        h3_env_contract_fprint(stdout, &ecl);
        if (h3_lifecycle_trace_wanted()) {
            printf("lifecycle_trace: enabled (H3_LIFECYCLE_TRACE=1; "
                   "debug only, no overlap change)\n");
            h3_lifecycle_whitelist_fprint(stdout);
        }
        if (h3_env_contract_expect_p0_visual()) {
            char eerr[512];
            if (!h3_env_contract_check_p0(ecl.arm, eerr, sizeof(eerr))) {
                fprintf(stderr, "h3: P0 visual env contract FAIL: %s\n",
                        eerr[0] ? eerr : "unknown");
                h3_fail_tag("FAIL_ENV");
                return 1;
            }
        }
    }
    if (ulysses < 1)
        ulysses = 1;
    if (lift_on) {
        if (!latent_upscaler_path || !latent_upscaler_path[0]) {
            const char *e = getenv("H3_LATENT_UPSCALER");
            if (e && e[0])
                latent_upscaler_path = e;
        }
        if (!latent_upscaler_path || !latent_upscaler_path[0]) {
            fprintf(stderr,
                    "h3: lift/taomate-lift 必须 --latent-upscaler FILE "
                    "（或 H3_LATENT_UPSCALER）；禁止 nearest-only\n");
            return 1;
        }
        if (ulysses > 1) {
            fprintf(stderr, "h3: preset=lift 尚未接 Ulysses>1\n");
            return 1;
        }
        if (first_frame_path || last_frame_path || n_ref_images ||
            ref_video_path) {
            fprintf(stderr,
                    "h3: preset=lift 尚未接视觉条件缩格子（FL2VA/ref）；"
                    "本轮只接 T2VA\n");
            return 1;
        }
        if (init_latent_path || lock_first_frame || init_first_frame) {
            fprintf(stderr,
                    "h3: preset=lift 与 --init-latent / lock-first-frame 互斥\n");
            return 1;
        }
    }
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
    /* C062: opt-in auto TE cache for in-process --repeat>=2 (bit-identical
     * emb reuse). Still off unless H3_TE_CACHE_AUTO=1; never gold default. */
    if ((!te_cache_dir || !te_cache_dir[0]) && clip_repeat >= 2) {
        const char *ae = getenv("H3_TE_CACHE_AUTO");
        if (ae && ae[0] == '1') {
            static char te_cache_auto_buf[1100];
            snprintf(te_cache_auto_buf, sizeof(te_cache_auto_buf),
                     "%s/te_cache_auto", out_dir);
            te_cache_dir = te_cache_auto_buf;
            printf("te_prompt_cache: AUTO dir=%s (H3_TE_CACHE_AUTO=1 "
                   "repeat=%d; emb reuse across clips)\n",
                   te_cache_dir, clip_repeat);
            fflush(stdout);
        }
    }
    if (tae_path && tae_path[0] && !strcmp(vae_decode, "official"))
        vae_decode = "tae";
    if (!h3_tae_allow_opt_in(preset_out.non_gold, vae_decode, tae_path, error,
                             sizeof(error))) {
        fprintf(stderr, "h3: %s\n", error);
        return 1;
    }
    if (!strcmp(vae_decode, "tae")) {
        if (!h3_tae_open(tae_path, &tae_keep, error, sizeof(error))) {
            fprintf(stderr, "h3: %s\n", error);
            return 1;
        }
    }

    if (probe_vae_load && !vae_path) {
        fprintf(stderr, "--probe-vae-load requires --vae\n");
        return 2;
    }
    if (!probe_vae_load && !dit_path) {
        if (preset_out.id == H3_PRESET_NVFP4)
            fprintf(stderr,
                    "preset=nvfp4 DISARMED — need INT8 --dit + H3_DIT_NVFP4=1 "
                    "+ H3_DIT_NVFP4_FILE=w4a4\n");
        else
            fprintf(stderr, "need --dit\n");
        return 2;
    }
    if (replay_latent) {
        /* P0 A/B verification: replay a dumped final latent straight into
         * VAE decode (skip TE+DiT). One clip, fixed input; async D2H off/on
         * must produce bit-identical rgb on this path. */
        int lt = 0, lh = 0, lw = 0;
        skip_dit = 1;
        clip_repeat = 1;
        frames_override = -1;
        if (!load_latent_file(replay_latent_path, NULL, 0, &lt, &lh, &lw,
                              error, sizeof(error)))
            return 2;
        frames_req = ((lt - 2) / 5) * 17 + 5;
        if (frames_req < 5) frames_req = 5;
        frames_aligned = align_frame_count(frames_req);
        if (video_latent_t_from_frames(frames_aligned) != lt) {
            fprintf(stderr,
                    "replay_latent: t=%d not on the 5+17k aligned grid\n", lt);
            return 2;
        }
        seconds = frames_req / 24;
        if (seconds < 1) seconds = 1;
        width = lw * 16;
        height = lh * 16;
        printf("replay_latent: %s → %dx%dx%d (W=%d H=%d frames=%d; "
               "VAE-only A/B, not generation)\n",
               replay_latent_path, lt, lh, lw, width, height, frames_req);
        printf("dump_once_score_many: mode=replay_geometry path=%s "
               "latent=%dx%dx%d do_not_retune_gold=1\n",
               replay_latent_path, lt, lh, lw);
    }
    if (skip_dit && !init_latent_path && !first_frame_path &&
        !last_frame_path && !replay_latent) {
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
        hi_lh = latent_h;
        hi_lw = latent_w;
        if (lift_on) {
            int lo_h = 0, lo_w = 0;
            if (lift_transition_cli > 0)
                lift_k = lift_transition_cli;
            else if (preset_out.id == H3_PRESET_TAOMATE) {
                lift_k = h3_selflift_taomate_low_nfe(steps);
                if (lift_k < 1)
                    lift_k = steps > 1 ? steps - 1 : 1;
            } else if (steps == 8)
                lift_k = 6;
            else if (steps == 4)
                lift_k = 3;
            else
                lift_k = steps > 1 ? steps - 1 : 1;
            {
                int all_low = (lift_k == steps);
                if (lift_k < 1 || lift_k > steps ||
                    (all_low && preset_out.id != H3_PRESET_TAOMATE)) {
                    fprintf(stderr,
                            "h3: lift transition_step=%d illegal for steps=%d\n",
                            lift_k, steps);
                    return 1;
                }
            }
            if (!h3_selflift_lowres_latent(hi_lh, hi_lw, 0.5f, &lo_h, &lo_w)) {
                fprintf(stderr, "h3: selflift lowres geometry failed\n");
                return 1;
            }
            if (preset_out.id == H3_PRESET_TAOMATE) {
                const char *sw = getenv("H3_TAOMATE_STAGE1_W");
                const char *sh = getenv("H3_TAOMATE_STAGE1_H");
                int stage1_w = 0, stage1_h = 0;
                if (sw && sw[0])
                    stage1_w = (int)strtol(sw, NULL, 10);
                if (sh && sh[0])
                    stage1_h = (int)strtol(sh, NULL, 10);
                /* Comfy 2-stage: first_stage 864×480, not 0.5×1344. */
                if (stage1_w >= 32 && stage1_h >= 32 && (stage1_w % 32) == 0 &&
                    (stage1_h % 32) == 0) {
                    lo_w = stage1_w / 16;
                    lo_h = stage1_h / 16;
                    if ((lo_w & 1) || (lo_h & 1) || lo_w < 2 || lo_h < 2) {
                        fprintf(stderr,
                                "h3: H3_TAOMATE_STAGE1 %dx%d not VAE/patch aligned\n",
                                stage1_w, stage1_h);
                        return 1;
                    }
                    printf("selflift: Comfy stage1 %dx%d → latent %dx%d "
                           "(not 0.5×%dx%d)\n",
                           stage1_w, stage1_h, lo_w, lo_h, width, height);
                }
            }
            if (lo_h == hi_lh && lo_w == hi_lw) {
                fprintf(stderr,
                        "h3: lift lowres == hires (%dx%d); refusing identity/"
                        "nearest-only\n",
                        lo_h, lo_w);
                return 1;
            }
            latent_h = lo_h;
            latent_w = lo_w;
            printf("selflift: target_pix=%dx%d latent %dx%d → low %dx%d "
                   "transition_step=%d (NFE low=%d high=%d) upscaler=%s\n",
                   width, height, hi_lh, hi_lw, latent_h, latent_w, lift_k,
                   lift_k, steps - lift_k, latent_upscaler_path);
            if (preset_out.id == H3_PRESET_TAOMATE && lift_k == steps)
                printf("selflift: taomate Z0_ONLY (3 low NFE + LBH, 0 high DiT)\n");
            fflush(stdout);
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
        if (preset_out.id == H3_PRESET_TAOMATE) {
            const char *ck = getenv("H3_TAOMATE_CLEAN_KV");
            const char *hov = getenv("H3_TAOMATE_CHUNK_OVERLAP");
            const char *hsn = getenv("H3_TAOMATE_SINK_T");
            const char *lkv = getenv("H3_TAOMATE_LAYER_KV");
            if (vae_decode && !strcmp(vae_decode, "official") &&
                !getenv("H3_VAE_WEIGHT_CACHE"))
                setenv("H3_VAE_WEIGHT_CACHE", "1", 1);
            memset(&tm_plan, 0, sizeof(tm_plan));
            if (lkv && lkv[0] == '1')
                tm_layer_kv = 1;
            if (ck && ck[0] == '1')
                tm_clean_kv = 1;
            else if (ck && ck[0] == '0')
                tm_clean_kv = 0;
            if (hov && hov[0])
                tm_hist_t = taomate_clamp_ov_t(strtol(hov, NULL, 10));
            if (hsn && hsn[0])
                tm_sink_t = taomate_clamp_ov_t(strtol(hsn, NULL, 10));
            if (tm_layer_kv) {
                tm_clean_kv = 0;
                if (tm_sink_t <= 0)
                    tm_sink_t = 2;
                fprintf(stderr,
                        "taomate: layer-wise KV on (Q=chunk, K/V hist; "
                        "not packed-Q CLEAN_KV). hist_t=%d sink_t=%d\n",
                        tm_hist_t, tm_sink_t);
            } else if (tm_clean_kv) {
                fprintf(stderr,
                        "warning: H3_TAOMATE_CLEAN_KV=1 concatenates whole prior "
                        "chunks into Q (768P OOM risk). Prefer "
                        "H3_TAOMATE_LAYER_KV=1 or bounded overlap.\n");
            }
            if (!h3_taomate_plan_build(frames_aligned, &tm_plan, error,
                                       sizeof(error))) {
                fprintf(stderr, "h3: taomate plan: %s\n", error);
                return 1;
            }
            {
                const char *force_chunks = getenv("H3_TAOMATE_FORCE_CHUNKS");
                int want_oneshot = (frames_aligned == 124);
                if (force_chunks && force_chunks[0] == '1')
                    want_oneshot = 0;
                if (want_oneshot) {
                    if (!h3_taomate_plan_build_oneshot(frames_aligned, &tm_plan,
                                                       error, sizeof(error))) {
                        fprintf(stderr, "h3: taomate oneshot: %s\n", error);
                        return 1;
                    }
                    tm_layer_kv = 0;
                    tm_clean_kv = 0;
                    tm_hist_t = 0;
                    tm_sink_t = 0;
                    /* Full-clip Q ≈37k. sage3 packed 14344 gemm-acc OOM'd;
                     * gold 5s dense uses cuDNN. Keep explicit --sdpa. */
                    if (!sdpa_explicit) {
                        sdpa = "cudnn";
                        printf("taomate_oneshot: SDPA cudnn (full 5s Q; not "
                               "sage3 pad, not 4 codec chunks)\n");
                    }
                }
            }
            tm_plan_ok = 1;
            printf("taomate_plan: frames=%d video_t=%d audio_t=%d phases=%d "
                   "clean_kv=%d layer_kv=%d hist_t=%d sink_t=%d canvas=%dx%d "
                   "(Comfy 2-stage target 1376x768; CUDA generate allows it)\n",
                   tm_plan.native_frames, tm_plan.video_t, tm_plan.audio_t,
                   tm_plan.n_phases, tm_clean_kv, tm_layer_kv, tm_hist_t,
                   tm_sink_t, width, height);
            {
                int pi;
                for (pi = 0; pi < tm_plan.n_phases; pi++) {
                    const h3_taomate_phase *ph = &tm_plan.phases[pi];
                    printf("  chunk%d groups=%d frames=%d..%d video_t=%d..%d "
                           "audio_t=%d..%d\n",
                           pi, ph->group_count, ph->frame_start, ph->frame_stop,
                           ph->video_t0, ph->video_t1, ph->audio_t0,
                           ph->audio_t1);
                }
            }
            if (tm_plan.video_t != latent_t) {
                fprintf(stderr,
                        "h3: taomate video_t %d != latent_t %d\n",
                        tm_plan.video_t, latent_t);
                return 1;
            }
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
        if (te_cache_dir && te_cache_dir[0]) {
            const char *req = getenv("H3_TE_CACHE_REQUIRE_HIT");
            printf("te_prompt_cache: dir=%s require_hit=%d "
                   "(opt-in; gold default off)\n",
                   te_cache_dir, (req && req[0] == '1') ? 1 : 0);
        }
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
               h3_vae_int8_weight_cache_wanted() ? "1" : "0",
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
    /* NVFP4 accel SKU: probe first, hard-fail, never silent INT8. */
    {
        /* DISARM default: path "*nvfp4*" or preset nvfp4 must NOT open the
         * host dequant→INT8 track. Opt-in only via H3_DIT_NVFP4_HOST_DEQUANT=1
         * (legacy Sep-13 bypass; not a product goal). Native FP4 = H3_DIT_NVFP4. */
        int want_nvfp4_host_dequant =
            (getenv("H3_DIT_NVFP4_HOST_DEQUANT") &&
             getenv("H3_DIT_NVFP4_HOST_DEQUANT")[0] == '1');
        int want_nvfp4 = want_nvfp4_host_dequant &&
                         ((preset_out.id == H3_PRESET_NVFP4) ||
                          h3_dit_nvfp4_path_hints(dit_path));
        if (want_nvfp4) {
            if (!h3_nvfp4_store_open(dit_path, &dit_nvfp4, error,
                                     sizeof(error))) {
                fprintf(stderr,
                        "h3: NVFP4 DiT 探测失败（不是金线；不回退 INT8）: %s\n",
                        error);
                return 1;
            }
            dit_nvfp4_opened = 1;
            if (!h3_dit_nvfp4_probe(&dit_nvfp4, error, sizeof(error))) {
                fprintf(stderr, "h3: %s\n", error);
                h3_nvfp4_store_free(&dit_nvfp4);
                dit_nvfp4_opened = 0;
                return 1;
            }
            dit_is_nvfp4 = 1;
            printf("dit_format: NVFP4 (DISARM host-dequant track; H3_DIT_NVFP4_HOST_DEQUANT=1)\n");
            if (n_ref_images || ref_video_path)
                printf("dit_tree: Ref2VA %s (not FL2VA)\n", dit_path);
            if (!h3_dit_w4a8_load_core_nvfp4(&dit_nvfp4, &core, error,
                                             sizeof(error))) {
                fprintf(stderr, "load_core_nvfp4: %s\n", error);
                goto cleanup;
            }
        }
    }
    /* Prefer Comfy INT8 ConvRot DiT (theme-follow / gold); else W4A8. */
    if (!dit_is_nvfp4)
        dit_int8 = h3_dit_int8_store_open(dit_path, error, sizeof(error));
    if (dit_int8) {
        dit_is_int8 = 1;
        /* H3_DIT_NVFP4=1 keeps INT8 ConvRot weights but routes GEMM via FP4
         * sidecar track — protocol arm=nvfp4 keys off dit_format: NVFP4. */
        if (getenv("H3_DIT_NVFP4") && getenv("H3_DIT_NVFP4")[0] == '1')
            printf("dit_format: NVFP4 (H3_DIT_NVFP4=1 GEMM track; "
                   "weight_store=INT8 ConvRot)\n");
        else
            printf("dit_format: INT8 ConvRot (theme-follow path)\n");
        if (n_ref_images || ref_video_path)
            printf("dit_tree: Ref2VA %s (not FL2VA)\n", dit_path);
        if (!h3_dit_w4a8_load_core_int8(dit_int8, &core, error,
                                        sizeof(error))) {
            fprintf(stderr, "load_core_int8: %s\n", error);
            goto cleanup;
        }
    } else if (!dit_is_nvfp4) {
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
    h3_vram_ledger_point("after_cuda");
    if (lift_on) {
        int in_ch = 0, hid = 0;
        h3_vae_int8_store vs;
        memset(&vs, 0, sizeof(vs));
        memset(lift_mean, 0, sizeof(lift_mean));
        memset(lift_std, 0, sizeof(lift_std));
        if (!h3_latent_upscaler_open(latent_upscaler_path, &lat_up, error,
                                     sizeof(error))) {
            fprintf(stderr, "h3: latent-upscaler open: %s\n", error);
            goto cleanup;
        }
        if (!h3_latent_upscaler_probe(lat_up, &in_ch, &hid, error,
                                      sizeof(error))) {
            fprintf(stderr, "h3: latent-upscaler probe: %s\n", error);
            goto cleanup;
        }
        if (!vae_path) {
            fprintf(stderr, "h3: preset=lift 需要 --vae 以加载 latents_mean/std\n");
            goto cleanup;
        }
        if (!h3_vae_int8_store_open(vae_path, &vs, error, sizeof(error)) ||
            !h3_vae_int8_load_latent_stats(&vs, lift_mean, lift_std, error,
                                           sizeof(error))) {
            fprintf(stderr, "h3: lift latent stats: %s\n", error);
            h3_vae_int8_store_free(&vs);
            goto cleanup;
        }
        h3_vae_int8_store_free(&vs);
        lift_stats_ok = 1;
        printf("selflift: LBH 24→%d loaded (%s); VAE mean/std ok\n", hid,
               latent_upscaler_path);
        fflush(stdout);
    }
    if (tae_keep && !h3_tae_bind_cuda(tae_keep, cuda, error, sizeof(error))) {
        fprintf(stderr, "h3: %s\n", error);
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
        if (dit_is_nvfp4)
            dit = h3_dit_w4a8_open_nvfp4(cuda, &dit_nvfp4, &cfg, error,
                                         sizeof(error));
        else if (dit_is_int8)
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
            {
                unsigned lora_flags = 0;
                if (preset_out.id == H3_PRESET_TAOMATE)
                    lora_flags = H3_LORA_REQUIRE_REFINER;
                if (!h3_lora_load_flags(lora_path, lora_scale, lora_flags,
                                        &lora_set, error, sizeof(error))) {
                    fprintf(stderr, "h3: lora 非金线: %s\n", error);
                    goto cleanup;
                }
            }
            lora_loaded = 1;
            if (!h3_dit_w4a8_set_lora(dit, &lora_set, error, sizeof(error))) {
                fprintf(stderr, "h3: lora attach: %s\n", error);
                goto cleanup;
            }
            printf("lora: path=%s scale=%.3g pairs=%d refiner_pairs=%d "
                   "ignored_vsa=%d (非金线)\n",
                   lora_set.path, (double)lora_set.scale, lora_set.n_pairs,
                   lora_set.n_refiner_pairs, lora_set.ignored_vsa_tensors);
        }
        {
            const char *rl = getenv("H3_REFINER_LORA");
            const char *rn = getenv("H3_REFINER_LORA_STEPS");
            const char *rs = getenv("H3_REFINER_LORA_SCALE");
            if (rl && *rl && lora_loaded) {
                float rscale = 1.0f;
                int rsteps = 1;
                if (rs && *rs) {
                    rscale = (float)strtod(rs, NULL);
                    if (!(rscale > 0.0f) || !isfinite(rscale))
                        rscale = 1.0f;
                }
                if (rn && *rn) {
                    rsteps = (int)strtol(rn, NULL, 10);
                    if (rsteps < 1)
                        rsteps = 1;
                }
                if (!h3_lora_load_flags(rl, rscale, 0, &refiner_lora_set, error,
                                        sizeof(error))) {
                    fprintf(stderr, "h3: refiner lora: %s\n", error);
                    goto cleanup;
                }
                refiner_lora_loaded = 1;
                refiner_lora_steps = rsteps;
                printf("refiner_lora: path=%s scale=%.3g tail_steps=%d "
                       "pairs=%d (opt-in; 主体 %s 保持)\n",
                       rl, (double)rscale, rsteps, refiner_lora_set.n_pairs,
                       lora_path);
            }
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
                if (dit_is_nvfp4)
                    uly_dit[ui] = h3_dit_w4a8_open_nvfp4(uly_cuda[ui],
                                                         &dit_nvfp4, &cfg,
                                                         error, sizeof(error));
                else if (dit_is_int8)
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
            h3_fail_tag("FAIL_OOM");
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
            h3_fail_tag("FAIL_OOM");
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
            h3_fail_tag("FAIL_OOM");
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
    orig_lh = latent_h;
    orig_lw = latent_w;
    orig_n_lat = n_lat;
    orig_vseq = video_seq;
    latent = (float *)malloc(n_lat * sizeof(float));
    velocity = (float *)malloc(n_lat * sizeof(float));
    sigmas = (float *)malloc((size_t)(steps + 1) * sizeof(float));
    text_h = (float *)malloc(context * H3_DIT_W4A8_HIDDEN * sizeof(float));
    pt = (float *)malloc(video_seq * sizeof(float));
    ph = (float *)malloc(video_seq * sizeof(float));
    pw = (float *)malloc(video_seq * sizeof(float));
    if (!latent || !velocity || !sigmas || !text_h || !pt || !ph || !pw) {
        fprintf(stderr, "OOM\n");
            h3_fail_tag("FAIL_OOM_DIT");
        goto cleanup;
    }
    if (cfg_scale > 1.0001f) {
        text_h_uncond =
            (float *)calloc(context * H3_DIT_W4A8_HIDDEN, sizeof(float));
        velocity_uncond = (float *)malloc(n_lat * sizeof(float));
        if (!text_h_uncond || !velocity_uncond) {
            fprintf(stderr, "OOM cfg buffers\n");
            h3_fail_tag("FAIL_OOM_DIT");
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
            h3_fail_tag("FAIL_OOM_DIT");
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
            h3_fail_tag("FAIL_OOM");
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
    if (replay_latent) {
        if (!load_latent_file(replay_latent_path, latent, n_lat, NULL, NULL,
                              NULL, error, sizeof(error))) {
            fprintf(stderr, "replay_latent: %s\n", error);
            goto cleanup;
        }
        printf("replay_latent: loaded %zu f32 checksum=0x%016llx "
               "(bit-fixed VAE input; async off/on must match)\n",
               n_lat,
               (unsigned long long)h3_dit_w4a8_checksum_f32(latent, n_lat));
        printf("# H3_SCORE_MANY mode=replay path=%s checksum=0x%016llx "
               "n=%zu\n",
               replay_latent_path,
               (unsigned long long)h3_dit_w4a8_checksum_f32(latent, n_lat),
               n_lat);
        printf("dump_once_score_many: mode=replay path=%s "
               "do_not_retune_gold=1\n",
               replay_latent_path);
    }
    if (lock_first_frame) {
        frame0_lock = (float *)malloc(frame0_elems * sizeof(float));
        if (!frame0_lock) {
            fprintf(stderr, "OOM frame0 lock\n");
            h3_fail_tag("FAIL_OOM");
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
            h3_fail_tag("FAIL_OOM");
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
            h3_fail_tag("FAIL_OOM");
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
            h3_fail_tag("FAIL_OOM");
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
            h3_fail_tag("FAIL_OOM");
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
            h3_fail_tag("FAIL_OOM");
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
    if (preset_out.id == H3_PRESET_TAOMATE) {
        if (steps == 5) {
            /* Comfy BasicScheduler simple + MiniMax shift. Not distilled 3. */
            if (!build_serving_sigmas(steps, H3_CUDA_VIDEO_SIGMA_SHIFT, sigmas)) {
                fprintf(stderr, "taomate Comfy simple-5 video schedule failed\n");
                goto cleanup;
            }
        } else if (!h3_taomate_nfe_sigmas(steps, H3_CUDA_VIDEO_SIGMA_SHIFT,
                                          sigmas)) {
            fprintf(stderr, "taomate video schedule failed nfe=%d\n", steps);
            goto cleanup;
        }
    } else {
        build_serving_sigmas(steps, H3_CUDA_VIDEO_SIGMA_SHIFT, sigmas);
    }
    audio_sigmas = (float *)malloc((size_t)(steps + 1) * sizeof(float));
    if (!audio_sigmas) {
        fprintf(stderr, "OOM audio sigmas\n");
            h3_fail_tag("FAIL_OOM_DIT");
        goto cleanup;
    }
    if (preset_out.id == H3_PRESET_TAOMATE) {
        if (steps == 5) {
            if (!build_serving_sigmas(steps, H3_CUDA_AUDIO_SIGMA_SHIFT,
                                      audio_sigmas)) {
                fprintf(stderr, "taomate Comfy simple-5 audio schedule failed\n");
                goto cleanup;
            }
            printf("taomate_schedule: nfe=5 simple+shift "
                   "(Comfy BasicScheduler/SplitSigmas 3+2; 非 distilled-3)\n");
            printf("taomate_sampler: Euler CONST (Comfy KSamplerSelect=er_sde "
                   "VP ER-SDE-3; needs x0+stage history, not ported)\n");
        } else if (!h3_taomate_nfe_sigmas(steps, H3_CUDA_AUDIO_SIGMA_SHIFT,
                                         audio_sigmas)) {
            fprintf(stderr, "taomate audio schedule failed nfe=%d\n", steps);
            goto cleanup;
        } else {
            printf("taomate_schedule: nfe=%d %s (非 serving-3)\n", steps,
                   steps == 6 ? "anchors=0,8,16,24,33,41,49"
                              : "anchors=0,16,33,49");
        }
    } else {
        build_serving_sigmas(steps, audio_sigma_shift_effective(), audio_sigmas);
    }
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
            h3_fail_tag("FAIL_OOM_DIT");
        goto cleanup;
    }
    if (pack_audio) {
        avel_last = (float *)calloc(n_audio, sizeof(float));
        avel_prev = (float *)calloc(n_audio, sizeof(float));
        if (!avel_last || !avel_prev) {
            fprintf(stderr, "OOM audio reuse buffers\n");
            h3_fail_tag("FAIL_OOM_DIT");
            goto cleanup;
        }
        if (cfg_scale > 1.0001f) {
            audio_vel_uncond = (float *)calloc(n_audio, sizeof(float));
            if (!audio_vel_uncond) {
                fprintf(stderr, "OOM audio cfg buffers\n");
            h3_fail_tag("FAIL_OOM_DIT");
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
               (double)audio_sigma_shift_effective());
    if (fewstep_id != H3_FEWSTEP_NONE) {
        char ferr[512];
        h3_fewstep_contract_in fin;
        memset(&fin, 0, sizeof(fin));
        fin.id = fewstep_id;
        fin.steps = steps;
        fin.audio_shift = audio_sigma_shift_effective();
        fin.video_sigmas = sigmas;
        fin.audio_sigmas = audio_sigmas;
        ferr[0] = '\0';
        if (!h3_fewstep_contract_check_sigmas(&fin, ferr, sizeof(ferr))) {
            fprintf(stderr, "h3: %s\n", ferr);
            h3_fail_tag("FAIL_CONTRACT");
            goto cleanup;
        }
    }

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
        if ((dit_is_int8 || dit_is_nvfp4) && dit) {
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
                    printf("te_prompt_cache: HIT dir=%s key=%s tokens=%zu "
                           "layers=%u dim=%d\n",
                           te_cache_dir, te_pc_hex, context, te_layers,
                           (int)H3_TE_CACHE_TEXT_DIM);
                    /* Packed hook never fires on HIT — kick host_stage here. */
                    if ((dit_is_int8 || dit_is_nvfp4) && dit &&
                        !host_stage_th_ok) {
                        const char *hs = getenv("H3_DIT_HOST_STAGE_TE");
                        if (!(hs && hs[0] == '0')) {
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
                            kick_dit_host_stage(&host_stage_kick_job);
                        }
                    }
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
                        printf("te_prompt_cache: MISS wrote dir=%s key=%s "
                               "tokens=%zu layers=%u dim=%d\n",
                               te_cache_dir, te_pc_hex, context, te_layers,
                               (int)H3_TE_CACHE_TEXT_DIM);
                    }
                }
            }
            if (te_cache_dir && te_cache_dir[0] && !te_pc_hit &&
                !have_vision_span) {
                /* Structured MISS even when save skipped (no hex / save fail). */
                if (!te_pc_hex[0])
                    printf("te_prompt_cache: MISS dir=%s key=none tokens=%zu "
                           "layers=%u dim=%d reason=no_key\n",
                           te_cache_dir, context, te_layers,
                           (int)H3_TE_CACHE_TEXT_DIM);
                {
                    const char *req = getenv("H3_TE_CACHE_REQUIRE_HIT");
                    if (req && req[0] == '1') {
                        fprintf(stderr,
                                "te_prompt_cache: REQUIRE_HIT failed "
                                "(H3_TE_CACHE_REQUIRE_HIT=1; warm cache first)\n");
                        h3_fail_tag("FAIL_ENV");
                        goto cleanup;
                    }
                }
            }
        }
        if (te_cache_dir && te_cache_dir[0] && have_vision_span) {
            printf("te_prompt_cache: SKIP dir=%s reason=vision_span\n",
                   te_cache_dir);
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
            h3_fail_tag("FAIL_OOM_DIT");
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
            h3_fail_tag("FAIL_OOM_DIT");
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
        h3_lifecycle_trace("TE text_emb+store", "post_refine", "release",
                           "before_DiT");
    }
    if (host_stage_th_ok && dit_is_nvfp4) {
        /* NVFP4 host dequant reallocs the same host slots the DiT upload
         * reads. Overlap UAF'd (SIGSEGV) at first qkv GEMM. Join first.
         * Gold INT8 aliases mmap and may overlap (want=2 below). */
        pthread_join(host_stage_th, NULL);
        host_stage_th_ok = 0;
        printf("dit_host_stage: NVFP4 joined before DiT (no overlap；非金线)\n");
        h3_lifecycle_trace("DiT host_stage pin buf (NVFP4)", "pre_DiT",
                           "join", "no_overlap");
        fflush(stdout);
    } else if (host_stage_th_ok) {
        /* Do not join all 50 before DiT. Wait for the first 2 blocks so
         * step1 can start; remaining disk→host overlaps compute. */
        unsigned want = 2;
        double tw0 = now_ms();
        const char *we = getenv("H3_DIT_HOST_STAGE_WAIT_BLOCKS");
        h3_w4a8_cache *hc = h3_dit_w4a8_cache(dit);
        if (we && we[0]) {
            long v = strtol(we, NULL, 10);
            if (v >= 1 && v <= 50) want = (unsigned)v;
        }
        while (hc) {
            unsigned ready = h3_w4a8_cache_host_ready_blocks(hc);
            if (ready >= want || host_stage_job.done) break;
            if (now_ms() - tw0 > 120000.0) break;
            usleep(2000);
        }
        printf("dit_host_stage: wait_first want=%u ready=%u wait_ms=%.1f "
               "(thread still running; join after DiT)\n",
               want,
               hc ? h3_w4a8_cache_host_ready_blocks(hc) : 0,
               now_ms() - tw0);
        h3_lifecycle_trace("DiT host_stage pin buf (INT8)", "pre_DiT",
                           "wait_first", "overlap_tail_ok");
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

    if (sdpa_th_ok) {
        pthread_join(sdpa_th, NULL);
        sdpa_th_ok = 0;
        printf("sage3_warmup: join ok=%d seq=%zu err=%s\n", sdpa_job.ok,
               sdpa_job.seq, sdpa_job.err[0] ? sdpa_job.err : "-");
        fflush(stdout);
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
            if (preset_out.id == H3_PRESET_TAOMATE && tm_plan_ok &&
                latent_t > 0) {
                size_t spatial = video_seq / (size_t)latent_t;
                int pi;
                size_t max_c = context;
                for (pi = 0; pi < tm_plan.n_phases; pi++) {
                    size_t s = taomate_phase_packed_seq(
                        &tm_plan, pi, context, spatial, pack_audio,
                        tm_layer_kv ? 0 : tm_clean_kv,
                        tm_layer_kv ? 0 : tm_hist_t,
                        tm_layer_kv ? 0 : tm_sink_t);
                    if (s > max_c)
                        max_c = s;
                }
                packed_seq = max_c;
                printf("taomate_pack: prebuild_seq=%zu (max chunk Q%s "
                       "hist_t=%d sink_t=%d, not full video_seq=%zu)\n",
                       packed_seq, tm_layer_kv ? ", layer-KV K/V extra" : "+overlap",
                       tm_hist_t, tm_sink_t, video_seq);
            }
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
                /* Extra chunk seqs (10/5 T) so later phases do not JIT. */
                if (preset_out.id == H3_PRESET_TAOMATE && tm_plan_ok &&
                    sdpa && !strcmp(sdpa, "sage3")) {
                    size_t spatial = video_seq / (size_t)latent_t;
                    int pj;
                    for (pj = 0; pj < tm_plan.n_phases; pj++) {
                        size_t s = taomate_phase_packed_seq(
                            &tm_plan, pj, context, spatial, pack_audio,
                            tm_layer_kv ? 0 : tm_clean_kv,
                            tm_layer_kv ? 0 : tm_hist_t,
                            tm_layer_kv ? 0 : tm_sink_t);
                        if (s == packed_seq)
                            continue;
                        pb_err[0] = '\0';
                        (void)h3_cuda_sage3_warmup(cuda, s, pb_heads,
                                                   (size_t)H3_DIT_W4A8_HEAD_DIM,
                                                   attn_scale, pb_err,
                                                   sizeof(pb_err));
                    }
                }
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
        if (dit) {
            h3_w4a8_cache *cw0 = h3_dit_w4a8_cache(dit);
            if (cw0)
                h3_w4a8_cache_log_window(cw0, "after_te");
        }
        gen_mem_trace(cuda, "after_te");
        h3_vram_ledger_point("after_te");
    }

clip_denoise:
    h3_vram_ledger_point("before_dit");
    gen_mem_trace(cuda, clip_i > 0 ? "clip_start_hot" : "clip_start");
    mux_write = mux_path;
    if (mux_path && clip_repeat > 1) {
        mux_path_for_clip(mux_clip_path, sizeof(mux_clip_path), mux_path, clip_i,
                          clip_repeat);
        mux_write = mux_clip_path;
    }
    if (clip_i > 0) {
        /* Restore pre-LBH geometry: Z0 grow_f32 mutates latent_h/n_lat. */
        latent_h = orig_lh;
        latent_w = orig_lw;
        n_lat = orig_n_lat;
        video_seq = orig_vseq;
        fill_normal_f32(seed + (uint64_t)clip_i, latent, n_lat);
        if (pack_audio && audio_c2t) {
            fill_normal_f32(seed + 0xA11DULL + (uint64_t)clip_i, audio_c2t,
                            (size_t)H3_DIT_W4A8_AUDIO_CH * 2 * (size_t)audio_t);
            if (!h3_dit_w4a8_pack_audio_latent(audio_c2t, audio_t, audio_rows,
                                               error, sizeof(error))) {
                fprintf(stderr, "pack audio (repeat): %s\n", error);
                goto cleanup;
            }
        }
        /* LBH rewrote pt/ph/pw to hires RoPE. Clip 2 must denoise at
         * stage-1 geometry with matching positions or VAE sees weave. */
        if (!h3_dit_w4a8_fill_metal_video_pos(target_rope_t0, latent_t,
                                              latent_h, latent_w, pt, ph, pw,
                                              error, sizeof(error))) {
            fprintf(stderr, "video pos (repeat): %s\n", error);
            goto cleanup;
        }
        if (pack_audio && apt && aph && apw) {
            if (!h3_dit_w4a8_fill_metal_audio_pos(
                    target_rope_t0, audio_t, latent_h, latent_w, apt, aph, apw,
                    error, sizeof(error))) {
                fprintf(stderr, "audio pos (repeat): %s\n", error);
                goto cleanup;
            }
        }
        printf("repeat_clip: restored rope latent %dx%d video_seq=%zu\n",
               latent_w, latent_h, video_seq);
        t_all0 = now_ms();
        t_te = 0;
        t_dit = t_vae = t_audio = t_ppm = t_mux = 0;
        t_lift_switch = 0;
        audio_overlap = 0;
        memset(&vae_prewarm_job, 0, sizeof(vae_prewarm_job));
        dit_evals = dit_skips = 0;
        last_eval = prev_eval = -1;
        mag_accum = 0.0f;
        mag_have_probe = 0;
        mag_consec = 0;
        mag_fresh = 0;
        free(mux_rgb);
        mux_rgb = NULL;
        free(latent_chw);
        latent_chw = NULL;
        h3_vae_int8_frames_free(&frames);
        h3_audio_vae_cuda_waveform_free(&awave);
        printf("repeat_clip: i=%d/%d hot=1 (reuse LoRA/INT8 pin/sage3/DiT; "
               "no re-pin) mux=%s\n",
               clip_i + 1, clip_repeat, mux_write ? mux_write : "-");
        fflush(stdout);
    } else if (clip_repeat > 1) {
        printf("repeat_clip: i=1/%d hot=0 mux=%s\n", clip_repeat,
               mux_write ? mux_write : "-");
        fflush(stdout);
    }

    t0 = now_ms();
    if (preset_out.id == H3_PRESET_TAOMATE && tm_plan_ok) {
        size_t spatial = video_seq / (size_t)latent_t;
        size_t t_floats = n_lat / (size_t)latent_t;
        int pi, st;
        int max_vt = 0, max_at = 0;
        float *v_chunk = NULL, *v_vel = NULL, *a_chunk = NULL, *a_vel = NULL;
        float *kv_v = NULL, *kv_a = NULL;
        float *kv_pt = NULL, *kv_ph = NULL, *kv_pw = NULL;
        float *kv_apt = NULL, *kv_aph = NULL, *kv_apw = NULL;
        float *a_ptc = NULL, *a_phc = NULL, *a_pwc = NULL;
        float *tm_vel = NULL, *tm_a_vel = NULL;
        float *lat_noise = NULL, *au_noise = NULL;
        double tm_low_ms = 0.0, tm_high_ms = 0.0, tm_lbh_ms = 0.0;
        int denoise_ov = tm_layer_kv ? 0 : (tm_clean_kv ? 0 : tm_hist_t);
        int cond_hist = tm_layer_kv ? 0 : (tm_clean_kv ? tm_hist_t : 0);
        int pack_sink = tm_layer_kv ? 0 : tm_sink_t;
        int use_kv = !tm_layer_kv && (tm_clean_kv || (tm_sink_t > 0));
        int layer_kv_on = 0;
        uint32_t tm_sidx[16384];
        int st_hi;
        int lbh_rc = 1;
        if (cond_seq || audio_cond_seq) {
            fprintf(stderr, "h3: taomate 拒绝与 F2VA/ref cond 叠用\n");
            goto cleanup;
        }
        for (pi = 0; pi < tm_plan.n_phases; pi++) {
            int t0w = h3_taomate_denoise_t0(&tm_plan, pi, denoise_ov);
            int a0w = h3_taomate_denoise_a0(&tm_plan, pi, denoise_ov);
            int vt = tm_plan.phases[pi].video_t1 - t0w;
            int at = tm_plan.phases[pi].audio_t1 - a0w;
            if (vt > max_vt) max_vt = vt;
            if (at > max_at) max_at = at;
        }
        lat_noise = (float *)malloc(n_lat * sizeof(float));
        if (lat_noise)
            memcpy(lat_noise, latent, n_lat * sizeof(float));
        if (pack_audio && audio_rows && n_audio) {
            au_noise = (float *)malloc(n_audio * sizeof(float));
            if (au_noise)
                memcpy(au_noise, audio_rows, n_audio * sizeof(float));
        }
        v_chunk = (float *)malloc((size_t)max_vt * t_floats * sizeof(float));
        v_vel = (float *)malloc((size_t)max_vt * t_floats * sizeof(float));
        if (pack_audio && max_at > 0) {
            a_chunk = (float *)calloc((size_t)max_at * 2u *
                                          (size_t)H3_DIT_W4A8_AUDIO_CH,
                                      sizeof(float));
            a_vel = (float *)calloc((size_t)max_at * 2u *
                                        (size_t)H3_DIT_W4A8_AUDIO_CH,
                                    sizeof(float));
            a_ptc = (float *)malloc((size_t)max_at * 2u * sizeof(float));
            a_phc = (float *)malloc((size_t)max_at * 2u * sizeof(float));
            a_pwc = (float *)malloc((size_t)max_at * 2u * sizeof(float));
        }
        if (use_kv) {
            kv_v = (float *)malloc(n_lat * sizeof(float));
            kv_pt = (float *)malloc(video_seq * sizeof(float));
            kv_ph = (float *)malloc(video_seq * sizeof(float));
            kv_pw = (float *)malloc(video_seq * sizeof(float));
            if (pack_audio) {
                kv_a = (float *)malloc(n_audio * sizeof(float));
                kv_apt = (float *)malloc(audio_seq * sizeof(float));
                kv_aph = (float *)malloc(audio_seq * sizeof(float));
                kv_apw = (float *)malloc(audio_seq * sizeof(float));
            }
        }
        if (lift_on) {
            tm_vel = (float *)malloc(n_lat * sizeof(float));
            if (pack_audio && n_audio)
                tm_a_vel = (float *)calloc(n_audio, sizeof(float));
        }
        if (!v_chunk || !v_vel || !lat_noise ||
            (pack_audio && max_at > 0 && (!a_chunk || !a_vel)) ||
            (pack_audio && n_audio && !au_noise) ||
            (use_kv && (!kv_v || !kv_pt || !kv_ph || !kv_pw)) ||
            (lift_on && !tm_vel) || (lift_on && pack_audio && n_audio && !tm_a_vel)) {
            fprintf(stderr, "OOM taomate chunk buffers\n");
            h3_fail_tag("FAIL_OOM_DIT");
            free(v_chunk);
            free(v_vel);
            free(a_chunk);
            free(a_vel);
            free(kv_v);
            free(kv_a);
            free(kv_pt);
            free(kv_ph);
            free(kv_pw);
            free(kv_apt);
            free(kv_aph);
            free(kv_apw);
            free(tm_vel);
            free(tm_a_vel);
            free(lat_noise);
            free(au_noise);
            goto cleanup;
        }
        st_hi = (lift_on && lift_k > 0 && lift_k < steps) ? lift_k : steps;
        printf("taomate_stream: phases=%d max_vt=%d spatial=%zu clean_kv=%d "
               "layer_kv=%d denoise_ov=%d sink_t=%d lift=%d low_nfe=%d high_nfe=%d "
               "pix_latent=%dx%d%s\n",
               tm_plan.n_phases, max_vt, spatial, tm_clean_kv, tm_layer_kv,
               denoise_ov, tm_sink_t, lift_on, lift_on ? st_hi : steps,
               lift_on ? (steps - st_hi) : 0, latent_h, latent_w,
               tm_plan.n_phases == 1 ? " (Comfy oneshot 5s; full-clip Q)"
                                     : " (codec chunks)");
        if (tm_layer_kv && dit && !lift_on && ulysses <= 1) {
            int try_hist = tm_hist_t, try_sink = tm_sink_t;
            char kv_err[256];
            while (try_hist > 0 || try_sink > 0) {
                size_t budget = h3_taomate_layer_kv_budget(
                    spatial, try_hist, try_sink, pack_audio);
                h3_w4a8_cache *cw = h3_dit_w4a8_cache(dit);
                size_t kv_b =
                    budget * (size_t)H3_DIT_W4A8_HEADS *
                    (size_t)H3_DIT_W4A8_HEAD_DIM * sizeof(uint16_t) *
                    (size_t)H3_DIT_W4A8_N_BLOCKS * 2u;
                kv_err[0] = '\0';
                if (cw)
                    (void)h3_w4a8_cache_make_room(
                        cw, kv_b + ((size_t)3ULL << 30), kv_err,
                        sizeof(kv_err));
                kv_err[0] = '\0';
                if (h3_dit_w4a8_clean_kv_enable(dit, budget, kv_err,
                                                sizeof(kv_err))) {
                    tm_hist_t = try_hist;
                    tm_sink_t = try_sink;
                    layer_kv_on = 1;
                    printf("taomate_layer_kv: enabled max_tokens=%zu hist_t=%d "
                           "sink_t=%d (Q=chunk; hist via rect flash, not sage3 pad)\n",
                           budget, try_hist, try_sink);
                    break;
                }
                printf("taomate_layer_kv: OOM max_tokens=%zu hist_t=%d sink_t=%d: %s\n",
                       budget, try_hist, try_sink,
                       kv_err[0] ? kv_err : "cudaMalloc");
                fflush(stdout);
                if (try_sink > 0) {
                    try_sink = 0;
                    continue;
                }
                if (try_hist <= 1)
                    break;
                try_hist--;
            }
            if (!layer_kv_on) {
                fprintf(stderr, "taomate_layer_kv: gave up, running without KV\n");
                tm_layer_kv = 0;
            }
        }
        for (pi = 0; pi < tm_plan.n_phases; pi++) {
            const h3_taomate_phase *tch = &tm_plan.phases[pi];
            int t0_ov = h3_taomate_denoise_t0(&tm_plan, pi, denoise_ov);
            int a0_ov = h3_taomate_denoise_a0(&tm_plan, pi, denoise_ov);
            int t0 = tch->video_t0, t1 = tch->video_t1;
            int a0 = tch->audio_t0, a1 = tch->audio_t1;
            int vt_ex = t1 - t0;
            int at_ex = a1 - a0;
            int vt = t1 - t0_ov;
            int at = a1 - a0_ov;
            size_t vseq = (size_t)vt * spatial;
            size_t aseq = pack_audio ? (size_t)at * 2u : 0;
            size_t vn = (size_t)vt * t_floats;
            size_t an = aseq * (size_t)H3_DIT_W4A8_AUDIO_CH;
            size_t kv_vseq = 0, kv_aseq = 0;
            const float *vpos_t = pt + (size_t)t0_ov * spatial;
            const float *vpos_h = ph + (size_t)t0_ov * spatial;
            const float *vpos_w = pw + (size_t)t0_ov * spatial;
            const float *apos_t = NULL, *apos_h = NULL, *apos_w = NULL;
            if (layer_kv_on && dit) {
                h3_w4a8_cache *cw = h3_dit_w4a8_cache(dit);
                if (cw)
                    h3_w4a8_cache_unfreeze_window(cw);
            }
            if (pack_audio && apt && aph && apw && a_ptc && at > 0) {
                taomate_copy_audio_pos(apt, a_ptc, audio_t, a0_ov, a1);
                taomate_copy_audio_pos(aph, a_phc, audio_t, a0_ov, a1);
                taomate_copy_audio_pos(apw, a_pwc, audio_t, a0_ov, a1);
                apos_t = a_ptc;
                apos_h = a_phc;
                apos_w = a_pwc;
            }
            taomate_copy_video_t(lat_noise, v_chunk, t0_ov, t1, t_floats);
            if (pack_audio && a_chunk && at > 0)
                taomate_copy_audio_t(au_noise ? au_noise : audio_rows, a_chunk,
                                     audio_t, a0_ov, a1);
            if (use_kv)
                taomate_pack_history(
                    &tm_plan, pi, tm_clean_kv, cond_hist, pack_sink, latent, pt,
                    ph, pw, audio_rows, apt, aph, apw, pack_audio, audio_t,
                    spatial, t_floats, kv_v, kv_pt, kv_ph, kv_pw, kv_a, kv_apt,
                    kv_aph, kv_apw, &kv_vseq, &kv_aseq);
            printf("taomate_chunk: i=%d video_t=%d..%d win=%d..%d seq=%zu "
                   "kv_vseq=%zu audio_t=%d..%d a_win=%d\n",
                   pi, t0, t1, t0_ov, t1, vseq, kv_vseq, a0, a1, a0_ov);
            fflush(stdout);
            for (st = 0; st < st_hi; st++) {
                h3_dit_w4a8_forward_stats tmfst;
                int is_lift_step = lift_on && lift_k < steps &&
                                   (st == lift_k - 1);
                memset(&tmfst, 0, sizeof(tmfst));
                h3_dit_w4a8_set_denoise_step(dit, st);
                if (layer_kv_on &&
                    !h3_dit_w4a8_clean_kv_begin(dit, 1, NULL, 0, 0, error,
                                                sizeof(error))) {
                    fprintf(stderr, "dit taomate layer-kv noisy: %s\n", error);
                    free(v_chunk); free(v_vel); free(a_chunk); free(a_vel);
                    free(kv_v); free(kv_a); free(kv_pt); free(kv_ph);
                    free(kv_pw); free(kv_apt); free(kv_aph); free(kv_apw);
                    goto cleanup;
                }
                if (pack_audio) {
                    if (!h3_dit_w4a8_forward_avc_f32(
                            dit, &core, 0, dit_layers, sigmas[st],
                            audio_sigmas[st], v_chunk, vseq, a_chunk, aseq,
                            kv_vseq ? kv_v : NULL, kv_vseq,
                            kv_aseq ? kv_a : NULL, kv_aseq, text_h, context,
                            vpos_t, vpos_h, vpos_w, apos_t, apos_h, apos_w,
                            kv_vseq ? kv_pt : NULL, kv_vseq ? kv_ph : NULL,
                            kv_vseq ? kv_pw : NULL,
                            kv_aseq ? kv_apt : NULL, kv_aseq ? kv_aph : NULL,
                            kv_aseq ? kv_apw : NULL, 1.0f, v_vel, a_vel, 1,
                            &tmfst, error, sizeof(error))) {
                        if (layer_kv_on)
                            h3_dit_w4a8_clean_kv_finish(dit, h3_dit_w4a8_clean_kv_hist_tokens(dit));
                        fprintf(stderr, "dit taomate chunk %d step %d: %s\n",
                                pi, st, error);
                        free(v_chunk); free(v_vel); free(a_chunk); free(a_vel);
                        free(kv_v); free(kv_a); free(kv_pt); free(kv_ph);
                        free(kv_pw); free(kv_apt); free(kv_aph); free(kv_apw);
                        goto cleanup;
                    }
                } else if (!h3_dit_w4a8_forward_video_f32(
                               dit, &core, 0, dit_layers, sigmas[st], v_chunk,
                               vseq, text_h, context, vpos_t, vpos_h, vpos_w,
                               1.0f, v_vel, 1, &tmfst, error, sizeof(error))) {
                    if (layer_kv_on)
                        h3_dit_w4a8_clean_kv_finish(dit, h3_dit_w4a8_clean_kv_hist_tokens(dit));
                    fprintf(stderr, "dit taomate chunk %d step %d: %s\n", pi, st,
                            error);
                    free(v_chunk); free(v_vel); free(a_chunk); free(a_vel);
                    free(kv_v); free(kv_a); free(kv_pt); free(kv_ph);
                    free(kv_pw); free(kv_apt); free(kv_aph); free(kv_apw);
                    goto cleanup;
                }
                if (layer_kv_on)
                    h3_dit_w4a8_clean_kv_finish(
                        dit, h3_dit_w4a8_clean_kv_hist_tokens(dit));
                dit_evals++;
                if (is_lift_step && tm_vel) {
                    memcpy(tm_vel + (size_t)t0 * t_floats,
                           v_vel + (size_t)(t0 - t0_ov) * t_floats,
                           (size_t)vt_ex * t_floats * sizeof(float));
                    if (pack_audio && a_vel && tm_a_vel && at_ex > 0 &&
                        a0_ov == a0)
                        taomate_write_audio_t(tm_a_vel, a_vel, audio_t, a0, a1);
                    tm_low_ms += tmfst.total_ms;
                    printf("taomate_step: chunk=%d step=%d/%d sigma=%.4g "
                           "dit_ms=%.1f packed=%zu (hold for LBH x0)\n",
                           pi, st + 1, steps, (double)sigmas[st],
                           tmfst.total_ms,
                           context + vseq + aseq + kv_vseq + kv_aseq);
                    fflush(stdout);
                    continue;
                }
                if (pack_audio && a_chunk && an) {
                    if (!h3_dit_w4a8_euler_step_f32(
                            a_chunk, a_vel, an, audio_sigmas[st],
                            audio_sigmas[st + 1], error, sizeof(error))) {
                        fprintf(stderr, "taomate audio euler: %s\n", error);
                        free(v_chunk); free(v_vel); free(a_chunk); free(a_vel);
                        free(kv_v); free(kv_a); free(kv_pt); free(kv_ph);
                        free(kv_pw); free(kv_apt); free(kv_aph); free(kv_apw);
                        goto cleanup;
                    }
                }
                if (!h3_dit_w4a8_euler_step_f32(v_chunk, v_vel, vn, sigmas[st],
                                                sigmas[st + 1], error,
                                                sizeof(error))) {
                    fprintf(stderr, "taomate euler chunk %d step %d: %s\n", pi,
                            st, error);
                    free(v_chunk); free(v_vel); free(a_chunk); free(a_vel);
                    free(kv_v); free(kv_a); free(kv_pt); free(kv_ph);
                    free(kv_pw); free(kv_apt); free(kv_aph); free(kv_apw);
                    goto cleanup;
                }
                printf("taomate_step: chunk=%d step=%d/%d sigma=%.4g->%.4g "
                       "dit_ms=%.1f packed=%zu%s\n",
                       pi, st + 1, steps, (double)sigmas[st],
                       (double)sigmas[st + 1], tmfst.total_ms,
                       context + vseq + aseq + kv_vseq + kv_aseq,
                       lift_on ? " low" : "");
                fflush(stdout);
                if (lift_on)
                    tm_low_ms += tmfst.total_ms;
            }
            taomate_blend_video_overlap(latent, v_chunk, t0_ov, t0, t1,
                                        t_floats);
            if (pack_audio && a_chunk && at > 0)
                taomate_blend_audio_overlap(audio_rows, a_chunk, audio_t, a0_ov,
                                            a0, a1);
            if (layer_kv_on && pi + 1 < tm_plan.n_phases) {
                size_t n_sink_part = 0, n_save, save_dst;
                h3_dit_w4a8_forward_stats cfst;
                size_t audio_off = context;
                size_t video_off = context + aseq;
                taomate_copy_video_t(latent, v_chunk, t0, t1, t_floats);
                if (pack_audio && a_chunk && at_ex > 0)
                    taomate_copy_audio_t(audio_rows, a_chunk, audio_t, a0, a1);
                n_save = h3_taomate_layer_kv_save_idx(
                    &tm_plan, pi, tm_hist_t, tm_sink_t, audio_off, video_off,
                    spatial, pack_audio, at_ex, tm_sidx,
                    sizeof(tm_sidx) / sizeof(tm_sidx[0]), &n_sink_part);
                if (n_save == 0) {
                    fprintf(stderr, "taomate_layer_kv: commit idx empty chunk %d\n",
                            pi);
                    free(v_chunk); free(v_vel); free(a_chunk); free(a_vel);
                    free(kv_v); free(kv_a); free(kv_pt); free(kv_ph);
                    free(kv_pw); free(kv_apt); free(kv_aph); free(kv_apw);
                    goto cleanup;
                }
                save_dst = (pi == 0) ? 0 : tm_kv_sink_n;
                if (pi == 0)
                    tm_kv_sink_n = n_sink_part;
                memset(&cfst, 0, sizeof(cfst));
                if (!h3_dit_w4a8_clean_kv_begin(dit, 2, tm_sidx, n_save, save_dst,
                                                error, sizeof(error))) {
                    fprintf(stderr, "taomate_layer_kv commit begin: %s\n", error);
                    free(v_chunk); free(v_vel); free(a_chunk); free(a_vel);
                    goto cleanup;
                }
                h3_dit_w4a8_set_denoise_step(dit, st_hi);
                if (pack_audio) {
                    if (!h3_dit_w4a8_forward_avc_f32(
                            dit, &core, 0, dit_layers, 0.0f, 0.0f, v_chunk,
                            vseq, a_chunk, aseq, NULL, 0, NULL, 0, text_h,
                            context, vpos_t, vpos_h, vpos_w, apos_t, apos_h,
                            apos_w, NULL, NULL, NULL, NULL, NULL, NULL, 1.0f,
                            v_vel, a_vel, 1, &cfst, error, sizeof(error))) {
                        h3_dit_w4a8_clean_kv_finish(dit, save_dst);
                        fprintf(stderr, "taomate_layer_kv clean fwd chunk %d: %s\n",
                                pi, error);
                        free(v_chunk); free(v_vel); free(a_chunk); free(a_vel);
                        goto cleanup;
                    }
                } else if (!h3_dit_w4a8_forward_video_f32(
                               dit, &core, 0, dit_layers, 0.0f, v_chunk, vseq,
                               text_h, context, vpos_t, vpos_h, vpos_w, 1.0f,
                               v_vel, 1, &cfst, error, sizeof(error))) {
                    h3_dit_w4a8_clean_kv_finish(dit, save_dst);
                    fprintf(stderr, "taomate_layer_kv clean fwd chunk %d: %s\n",
                            pi, error);
                    free(v_chunk); free(v_vel); free(a_chunk); free(a_vel);
                    goto cleanup;
                }
                h3_dit_w4a8_clean_kv_finish(dit, save_dst + n_save);
                dit_evals++;
                printf("taomate_layer_kv: commit chunk=%d save=%zu sink=%zu "
                       "hist_now=%zu dit_ms=%.1f\n",
                       pi, n_save, tm_kv_sink_n,
                       h3_dit_w4a8_clean_kv_hist_tokens(dit), cfst.total_ms);
                fflush(stdout);
            }
        }
        if (lift_on && tm_vel && lat_up) {
            int hi_st;
            {
                int already_x0 = (lift_k >= steps);
                float sk = already_x0 ? 0.0f : sigmas[lift_k - 1];
                float sn = already_x0 ? 0.0f : sigmas[lift_k];
                float ask = already_x0 ? 0.0f : audio_sigmas[lift_k - 1];
                float asn = already_x0 ? 0.0f : audio_sigmas[lift_k];
                lbh_rc = generate_lbh_transition(
                    &latent, &tm_vel, &pt, &ph, &pw, &apt, &aph, &apw, &n_lat,
                    &video_seq, &latent_h, &latent_w, latent_t, hi_lh, hi_lw,
                    (int)target_rope_t0, sk, sn, pack_audio, audio_t, n_audio,
                    audio_rows, tm_a_vel, ask, asn, seed, lift_mean, lift_std,
                    lat_up, cuda, dit, context, audio_seq, already_x0,
                    &tm_lbh_ms, error, sizeof(error));
            }
            if (!lbh_rc) {
                fprintf(stderr, "h3: taomate LBH: %s\n", error);
                free(v_chunk); free(v_vel); free(a_chunk); free(a_vel);
                free(kv_v); free(kv_a); free(kv_pt); free(kv_ph); free(kv_pw);
                free(kv_apt); free(kv_aph); free(kv_apw);
                free(a_ptc); free(a_phc); free(a_pwc);
                free(tm_vel); free(tm_a_vel);
                goto cleanup;
            }
            /* Comfy MinimaxH3LatentUpscaler3D force_unload=true before stage2. */
            if (lat_up)
                h3_latent_upscaler_release_device(lat_up);
            spatial = video_seq / (size_t)latent_t;
            t_floats = n_lat / (size_t)latent_t;
            if (lbh_rc == 1) {
            free(lat_noise);
            lat_noise = (float *)malloc(n_lat * sizeof(float));
            if (lat_noise)
                memcpy(lat_noise, latent, n_lat * sizeof(float));
            if (pack_audio && audio_rows && n_audio) {
                free(au_noise);
                au_noise = (float *)malloc(n_audio * sizeof(float));
                if (au_noise)
                    memcpy(au_noise, audio_rows, n_audio * sizeof(float));
            }
            free(v_chunk);
            free(v_vel);
            v_chunk = (float *)malloc((size_t)max_vt * t_floats * sizeof(float));
            v_vel = (float *)malloc((size_t)max_vt * t_floats * sizeof(float));
            if (!v_chunk || !v_vel || !lat_noise) {
                fprintf(stderr, "OOM taomate hires chunk buffers\n");
            h3_fail_tag("FAIL_OOM_DIT");
                free(v_chunk); free(v_vel); free(a_chunk); free(a_vel);
                free(tm_vel); free(tm_a_vel); free(lat_noise); free(au_noise);
                goto cleanup;
            }
            }
            if (lbh_rc == 1 && cuda) {
                int pii;
                size_t max_c = context;
                for (pii = 0; pii < tm_plan.n_phases; pii++) {
                    size_t s = taomate_phase_packed_seq(
                        &tm_plan, pii, context, spatial, pack_audio,
                        0, denoise_ov, 0);
                    if (s > max_c)
                        max_c = s;
                }
                printf("taomate_lift: hires prebuild_seq=%zu\n", max_c);
                {
                    double sw_ms = 0.0;
                    if (!lift_prepare_hires_vram(dit, cuda, max_c, &sw_ms,
                                                 error, sizeof(error))) {
                        fprintf(stderr, "h3: taomate hires vram: %s\n", error);
                        free(v_chunk); free(v_vel); free(a_chunk); free(a_vel);
                        free(tm_vel); free(tm_a_vel);
                        goto cleanup;
                    }
                    t_lift_switch += sw_ms;
                }
            }
            if (lbh_rc == 1) {
                printf("taomate_lift: high-res tail steps=%d..%d latent %dx%d "
                       "spatial=%zu\n",
                       lift_k + 1, steps, latent_h, latent_w, spatial);
                for (pi = 0; pi < tm_plan.n_phases; pi++) {
                    const h3_taomate_phase *tch = &tm_plan.phases[pi];
                    int t0_ov = h3_taomate_denoise_t0(&tm_plan, pi, denoise_ov);
                    int a0_ov = h3_taomate_denoise_a0(&tm_plan, pi, denoise_ov);
                    int t0 = tch->video_t0, t1 = tch->video_t1;
                    int a0 = tch->audio_t0, a1 = tch->audio_t1;
                    int vt = t1 - t0_ov;
                    int at = a1 - a0_ov;
                    size_t vseq = (size_t)vt * spatial;
                    size_t aseq = pack_audio ? (size_t)at * 2u : 0;
                    size_t vn = (size_t)vt * t_floats;
                    size_t an = aseq * (size_t)H3_DIT_W4A8_AUDIO_CH;
                    const float *vpos_t = pt + (size_t)t0_ov * spatial;
                    const float *vpos_h = ph + (size_t)t0_ov * spatial;
                    const float *vpos_w = pw + (size_t)t0_ov * spatial;
                    const float *apos_t = NULL, *apos_h = NULL, *apos_w = NULL;
                    if (pack_audio && apt && aph && apw && a_ptc && at > 0) {
                        taomate_copy_audio_pos(apt, a_ptc, audio_t, a0_ov, a1);
                        taomate_copy_audio_pos(aph, a_phc, audio_t, a0_ov, a1);
                        taomate_copy_audio_pos(apw, a_pwc, audio_t, a0_ov, a1);
                        apos_t = a_ptc;
                        apos_h = a_phc;
                        apos_w = a_pwc;
                    }
                    taomate_copy_video_t(lat_noise, v_chunk, t0_ov, t1,
                                         t_floats);
                    if (pack_audio && a_chunk && at > 0)
                        taomate_copy_audio_t(au_noise ? au_noise : audio_rows,
                                             a_chunk, audio_t, a0_ov, a1);
                    printf("taomate_chunk: i=%d video_t=%d..%d win=%d..%d "
                           "seq=%zu (hires no sink)\n",
                           pi, t0, t1, t0_ov, t1, vseq);
                    for (hi_st = lift_k; hi_st < steps; hi_st++) {
                        if (!vae_th_ok && !skip_vae && vae_path &&
                            hi_st == steps - 2) {
                            vae_prewarm_t0 = now_ms();
                            if (kick_vae_prewarm(&vae_th, &vae_prewarm_job,
                                                 vae_path, &vae_store,
                                                 &vae_opened, device,
                                                 hi_st + 1,
                                                 vae_prewarm_want_dev(device),
                                                 vae_blocks))
                                vae_th_ok = 1;
                        }
                        h3_dit_w4a8_forward_stats tmfst;
                        int nsys_hi = 0;
                        memset(&tmfst, 0, sizeof(tmfst));
                        nsys_hi = h3_nsys_hires_begin(clip_i, hi_st == lift_k);
                        h3_dit_w4a8_set_denoise_step(dit, hi_st);
                        if (pack_audio) {
                            if (!h3_dit_w4a8_forward_avc_f32(
                                    dit, &core, 0, dit_layers, sigmas[hi_st],
                                    audio_sigmas[hi_st], v_chunk, vseq, a_chunk,
                                    aseq, NULL, 0, NULL, 0, text_h, context,
                                    vpos_t, vpos_h, vpos_w, apos_t, apos_h,
                                    apos_w, NULL, NULL, NULL, NULL, NULL, NULL,
                                    1.0f, v_vel, a_vel, 1, &tmfst, error,
                                    sizeof(error))) {
                                h3_nsys_hires_end(nsys_hi, 0.0);
                                fprintf(stderr,
                                        "dit taomate hires chunk %d step %d: %s\n",
                                        pi, hi_st, error);
                                free(v_chunk); free(v_vel); free(a_chunk);
                                free(a_vel); free(tm_vel); free(tm_a_vel);
                                goto cleanup;
                            }
                        } else if (!h3_dit_w4a8_forward_video_f32(
                                       dit, &core, 0, dit_layers, sigmas[hi_st],
                                       v_chunk, vseq, text_h, context, vpos_t,
                                       vpos_h, vpos_w, 1.0f, v_vel, 1, &tmfst,
                                       error, sizeof(error))) {
                            h3_nsys_hires_end(nsys_hi, 0.0);
                            fprintf(stderr,
                                    "dit taomate hires chunk %d step %d: %s\n",
                                    pi, hi_st, error);
                            free(v_chunk); free(v_vel); free(tm_vel);
                            goto cleanup;
                        }
                        h3_nsys_hires_end(nsys_hi, tmfst.total_ms);
                        dit_evals++;
                        if (pack_audio && a_chunk && an) {
                            if (!h3_dit_w4a8_euler_step_f32(
                                    a_chunk, a_vel, an, audio_sigmas[hi_st],
                                    audio_sigmas[hi_st + 1], error,
                                    sizeof(error))) {
                                fprintf(stderr, "taomate hires audio euler: %s\n",
                                        error);
                                free(v_chunk); free(v_vel); free(tm_vel);
                                goto cleanup;
                            }
                        }
                        if (!h3_dit_w4a8_euler_step_f32(
                                v_chunk, v_vel, vn, sigmas[hi_st],
                                sigmas[hi_st + 1], error, sizeof(error))) {
                            fprintf(stderr, "taomate hires euler: %s\n", error);
                            free(v_chunk); free(v_vel); free(tm_vel);
                            goto cleanup;
                        }
                        tm_high_ms += tmfst.total_ms;
                        printf("taomate_step: chunk=%d step=%d/%d sigma=%.4g->%.4g "
                               "dit_ms=%.1f packed=%zu high\n",
                               pi, hi_st + 1, steps, (double)sigmas[hi_st],
                               (double)sigmas[hi_st + 1], tmfst.total_ms,
                               context + vseq + aseq);
                        fflush(stdout);
                    }
                    taomate_blend_video_overlap(latent, v_chunk, t0_ov, t0, t1,
                                                t_floats);
                    if (pack_audio && a_chunk && at > 0)
                        taomate_blend_audio_overlap(audio_rows, a_chunk, audio_t,
                                                    a0_ov, a0, a1);
                }
            }
            printf("taomate_lift: low_dit_ms=%.1f lbh_ms=%.1f high_dit_ms=%.1f "
                   "nfe_low=%d nfe_high=%d z0_only=%d\n",
                   tm_low_ms, tm_lbh_ms, tm_high_ms, st_hi,
                   lbh_rc == 1 ? (steps - st_hi) : 0, lbh_rc == 2);
            fflush(stdout);
        }
        free(v_chunk);
        free(v_vel);
        free(a_chunk);
        free(a_vel);
        free(kv_v);
        free(kv_a);
        free(kv_pt);
        free(kv_ph);
        free(kv_pw);
        free(kv_apt);
        free(kv_aph);
        free(kv_apw);
        free(a_ptc);
        free(a_phc);
        free(a_pwc);
        free(tm_vel);
        free(tm_a_vel);
        free(lat_noise);
        free(au_noise);
    } else
    for (step = 0; step < steps; step++) {
        h3_dit_w4a8_forward_stats fst;
        int evaluate = reuse_sel[step] ? 1 : 0;
        memset(&fst, 0, sizeof(fst));
        if (!vae_th_ok && !skip_vae && vae_path &&
            step == (steps >= 2 ? steps - 2 : 0)) {
            vae_prewarm_t0 = now_ms();
            if (kick_vae_prewarm(&vae_th, &vae_prewarm_job, vae_path,
                                 &vae_store, &vae_opened, device,
                                 step + 1, vae_prewarm_want_dev(device),
                                 vae_blocks))
                vae_th_ok = 1;
        }

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

        /* Opt-in refiner 切换：末 N 步前把 LoRA 换为 refiner set，其余步换回主体。 */
        if (refiner_lora_loaded) {
            int want_refiner = (step >= steps - refiner_lora_steps);
            if (want_refiner && !refiner_lora_attached) {
                double tsw = now_ms();
                if (!h3_dit_w4a8_set_lora(dit, &refiner_lora_set, error,
                                          sizeof(error))) {
                    fprintf(stderr, "h3: refiner lora attach: %s\n", error);
                    goto cleanup;
                }
                refiner_lora_attached = 1;
                printf("refiner_lora: attach at step %d/%d (switch_ms=%.1f)\n",
                       step + 1, steps, now_ms() - tsw);
            } else if (!want_refiner && refiner_lora_attached) {
                if (!h3_dit_w4a8_set_lora(dit, &lora_set, error,
                                          sizeof(error))) {
                    fprintf(stderr, "h3: main lora re-attach: %s\n", error);
                    goto cleanup;
                }
                refiner_lora_attached = 0;
            }
        }

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
            if (step == 0)
                printf("dit_step0_layers: prefetch=%.1f h2d=%.1f compute=%.1f "
                       "d2h=%.1f wait=%.1f\n",
                       fst.layers_prefetch_ms, fst.layers_h2d_ms,
                       fst.layers_compute_ms, fst.layers_d2h_ms,
                       fst.layers_prefetch_wait_ms);
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
        if (lift_on && step == lift_k - 1) {
            float *x0_pack = NULL, *z_low = NULL, *z_high = NULL, *z0_pack = NULL;
            float *noise_hi = NULL;
            float sigma_k = sigmas[step];
            float sigma_next = sigmas[step + 1];
            int lo_h = latent_h, lo_w = latent_w;
            size_t n_hi, video_seq_hi, n_tok_hi;
            double t_lift = now_ms();
            size_t ii;
            if (pack_audio) {
                if (!h3_dit_w4a8_euler_step_f32(
                        audio_rows, audio_vel, n_audio, audio_sigmas[step],
                        audio_sigmas[step + 1], error, sizeof(error))) {
                    fprintf(stderr, "audio euler (lift aux): %s\n", error);
                    goto cleanup;
                }
            }
            if (!(sigma_k > 1e-8f) || !lift_stats_ok || !lat_up) {
                fprintf(stderr, "h3: lift transition rejected (sigma/stats)\n");
                goto cleanup;
            }
            x0_pack = (float *)malloc(n_lat * sizeof(float));
            if (!x0_pack) {
                fprintf(stderr, "OOM lift x0\n");
            h3_fail_tag("FAIL_OOM_DIT");
                goto cleanup;
            }
            for (ii = 0; ii < n_lat; ii++)
                x0_pack[ii] = latent[ii] + sigma_k * velocity[ii];
            z_low = h3_vae_int8_unpack_dit_latent_f32(x0_pack, latent_t, lo_h,
                                                      lo_w, error, sizeof(error));
            free(x0_pack);
            x0_pack = NULL;
            if (!z_low) {
                fprintf(stderr, "lift unpack: %s\n", error);
                goto cleanup;
            }
            /* Comfy MiniMaxH3Video.process_out is identity (scale=1). LBH
             * learned_latent_lift still does (x-mean)/std internally on that
             * DiT-normalized CHW. Do NOT *std+mean here — that was the P0
             * magenta-grid (network saw VAE-native as if it were DiT-space). */
            z_high = (float *)malloc((size_t)H3_VAE_INT8_LATENT_CH *
                                     (size_t)latent_t * (size_t)hi_lh *
                                     (size_t)hi_lw * sizeof(float));
            if (!z_high) {
                free(z_low);
                fprintf(stderr, "OOM lift z_high\n");
            h3_fail_tag("FAIL_OOM_DIT");
                goto cleanup;
            }
            {
                size_t nlo = (size_t)H3_VAE_INT8_LATENT_CH * (size_t)latent_t *
                             (size_t)lo_h * (size_t)lo_w;
                dump_chw_channel_stats("z_low", z_low, latent_t, lo_h, lo_w);
                dump_f32_bin("/tmp/h3_lift_z_low.f32", z_low, nlo);
                dump_chw_pgm("/tmp/h3_lift_z_low.pgm", z_low, lo_h, lo_w);
                dump_chw_pgm("/tmp/h3_lift_z_low_ch1.pgm",
                             z_low + nlo / 24u, lo_h, lo_w);
                dump_chw_pgm("/tmp/h3_lift_z_low_ch2.pgm",
                             z_low + 2u * nlo / 24u, lo_h, lo_w);
            }
            {
                const char *zlow_only = getenv("H3_LIFT_ZLOW_ONLY");
                if (zlow_only && zlow_only[0] == '1') {
                    size_t n_tok_lo = (size_t)latent_t * (size_t)(lo_h / 2) *
                                      (size_t)(lo_w / 2) *
                                      (size_t)H3_DIT_W4A8_VIDEO_PATCH;
                    float *z0_lo = (float *)malloc(n_tok_lo * sizeof(float));
                    if (!z0_lo ||
                        !h3_vae_patchify_video_96(z_low, latent_t, lo_h, lo_w,
                                                  z0_lo, n_tok_lo)) {
                        fprintf(stderr, "lift zlow-only pack failed\n");
                        free(z_low);
                        free(z_high);
                        free(z0_lo);
                        goto cleanup;
                    }
                    memcpy(latent, z0_lo, n_tok_lo * sizeof(float));
                    n_lat = n_tok_lo;
                    free(z0_lo);
                    free(z_low);
                    free(z_high);
                    printf("selflift: ZLOW_ONLY decode low-res endpoint "
                           "(skip LBH / no remaining DiT) %dx%d\n",
                           lo_h, lo_w);
                    fflush(stdout);
                    break;
                }
            }
            if (!h3_latent_upscaler_forward(lat_up, cuda, z_low, latent_t, lo_h,
                                            lo_w, hi_lh, hi_lw, lift_mean,
                                            lift_std, z_high, error,
                                            sizeof(error))) {
                fprintf(stderr, "h3: LBH forward: %s\n", error);
                free(z_low);
                free(z_high);
                goto cleanup;
            }
            {
                size_t nhi = (size_t)H3_VAE_INT8_LATENT_CH * (size_t)latent_t *
                             (size_t)hi_lh * (size_t)hi_lw;
                float r_lo = chw_adj_ratio(z_low, latent_t, lo_h, lo_w);
                float r_hi = chw_adj_ratio(z_high, latent_t, hi_lh, hi_lw);
                printf("selflift: chw_adj ch0/t0 low=%.4g high=%.4g "
                       "(~1=checkerboard, ~0.1=smooth)\n",
                       (double)r_lo, (double)r_hi);
                dump_chw_channel_stats("z_high", z_high, latent_t, hi_lh, hi_lw);
                dump_f32_bin("/tmp/h3_lift_z_high.f32", z_high, nhi);
                dump_chw_pgm("/tmp/h3_lift_z_high.pgm", z_high, hi_lh, hi_lw);
                dump_chw_pgm("/tmp/h3_lift_z_high_ch1.pgm",
                             z_high + nhi / 24u, hi_lh, hi_lw);
                dump_chw_pgm("/tmp/h3_lift_z_high_ch2.pgm",
                             z_high + 2u * nhi / 24u, hi_lh, hi_lw);
                fflush(stdout);
            }
            free(z_low);
            z_low = NULL;
            /* process_in is also identity for MiniMaxH3Video; LBH already
             * returned DiT-normalized CHW (*std+mean of its internal space). */
            video_seq_hi = (size_t)latent_t * (size_t)(hi_lh / 2) *
                           (size_t)(hi_lw / 2);
            n_tok_hi = video_seq_hi * (size_t)H3_DIT_W4A8_VIDEO_PATCH;
            z0_pack = (float *)malloc(n_tok_hi * sizeof(float));
            if (!z0_pack ||
                !h3_vae_patchify_video_96(z_high, latent_t, hi_lh, hi_lw, z0_pack,
                                          n_tok_hi)) {
                fprintf(stderr, "lift pack failed\n");
                free(z_high);
                free(z0_pack);
                goto cleanup;
            }
            {
                float *rt = h3_vae_int8_unpack_dit_latent_f32(
                    z0_pack, latent_t, hi_lh, hi_lw, error, sizeof(error));
                double mx = 0.0;
                size_t nchw = (size_t)H3_VAE_INT8_LATENT_CH * (size_t)latent_t *
                              (size_t)hi_lh * (size_t)hi_lw;
                size_t k;
                if (!rt) {
                    fprintf(stderr, "lift pack-unpack: %s\n", error);
                    free(z_high);
                    free(z0_pack);
                    goto cleanup;
                }
                for (k = 0; k < nchw; k++) {
                    double d = fabs((double)rt[k] - (double)z_high[k]);
                    if (d > mx)
                        mx = d;
                }
                printf("selflift: pack↔unpack maxabs=%.6g\n", mx);
                free(rt);
            }
            free(z_high);
            z_high = NULL;
            n_hi = n_tok_hi;
            if (!grow_f32(&latent, n_hi, error, sizeof(error)) ||
                !grow_f32(&velocity, n_hi, error, sizeof(error)) ||
                !grow_f32(&vel_last, n_hi, error, sizeof(error)) ||
                !grow_f32(&vel_prev, n_hi, error, sizeof(error)) ||
                !grow_f32(&pt, video_seq_hi, error, sizeof(error)) ||
                !grow_f32(&ph, video_seq_hi, error, sizeof(error)) ||
                !grow_f32(&pw, video_seq_hi, error, sizeof(error))) {
                fprintf(stderr, "h3: lift realloc: %s\n", error);
                free(z0_pack);
                goto cleanup;
            }
            if (velocity_uncond &&
                !grow_f32(&velocity_uncond, n_hi, error, sizeof(error))) {
                fprintf(stderr, "h3: lift realloc uncond: %s\n", error);
                free(z0_pack);
                goto cleanup;
            }
            {
                const char *z0only = getenv("H3_LIFT_Z0_ONLY");
                if (z0only && z0only[0] == '1') {
                    memcpy(latent, z0_pack, n_hi * sizeof(float));
                    latent_h = hi_lh;
                    latent_w = hi_lw;
                    video_seq = video_seq_hi;
                    n_lat = n_hi;
                    free(z0_pack);
                    printf("selflift: Z0_ONLY decode lifted endpoint (no "
                           "re-noise / no remaining DiT)\n");
                    fflush(stdout);
                    break;
                }
            }
            noise_hi = (float *)malloc(n_hi * sizeof(float));
            if (!noise_hi) {
                free(z0_pack);
                fprintf(stderr, "OOM lift noise\n");
            h3_fail_tag("FAIL_OOM_DIT");
                goto cleanup;
            }
            fill_normal_f32(seed + 1ULL, noise_hi, n_hi);
            for (ii = 0; ii < n_hi; ii++) {
                float z0 = z0_pack[ii];
                latent[ii] = sigma_k * noise_hi[ii] + (1.0f - sigma_k) * z0;
                velocity[ii] = (z0 - latent[ii]) / sigma_k;
            }
            free(noise_hi);
            free(z0_pack);
            if (!h3_dit_w4a8_euler_step_f32(latent, velocity, n_hi, sigma_k,
                                            sigma_next, error, sizeof(error))) {
                fprintf(stderr, "lift euler: %s\n", error);
                goto cleanup;
            }
            latent_h = hi_lh;
            latent_w = hi_lw;
            video_seq = video_seq_hi;
            n_lat = n_hi;
            memcpy(vel_last, velocity, n_lat * sizeof(float));
            if (!h3_dit_w4a8_fill_metal_video_pos(target_rope_t0, latent_t,
                                                  latent_h, latent_w, pt, ph, pw,
                                                  error, sizeof(error))) {
                fprintf(stderr, "lift video pos: %s\n", error);
                goto cleanup;
            }
            if (pack_audio) {
                if (!h3_dit_w4a8_fill_metal_audio_pos(
                        target_rope_t0, audio_t, latent_h, latent_w, apt, aph,
                        apw, error, sizeof(error))) {
                    fprintf(stderr, "lift audio pos: %s\n", error);
                    goto cleanup;
                }
            }
            if (dit && cuda) {
                size_t packed_seq = context + cond_seq + audio_cond_seq +
                                    video_seq + (pack_audio ? audio_seq : 0);
                {
                    double sw_ms = 0.0;
                    if (!lift_prepare_hires_vram(dit, cuda, packed_seq, &sw_ms,
                                                 error, sizeof(error))) {
                        fprintf(stderr, "h3: lift hires vram: %s\n", error);
                        goto cleanup;
                    }
                    t_lift_switch += sw_ms;
                }
            }
            printf("selflift: transition step=%d σ=%.4g→%.4g latent %dx%d→%dx%d "
                   "tokens %zu lift_ms=%.1f (no extra DiT NFE)\n",
                   step + 1, (double)sigma_k, (double)sigma_next, lo_h, lo_w,
                   hi_lh, hi_lw, n_lat, now_ms() - t_lift);
            fflush(stdout);
            continue;
        }
        {
            size_t need = n_lat;
            if (pack_audio && n_audio > need)
                need = n_audio;
            if (sampler_id == H3_SAMPLER_LCM_RENOISE && need > renoise_scratch_n) {
                float *nb = (float *)realloc(renoise_scratch,
                                             need * sizeof(float));
                if (!nb) {
                    fprintf(stderr, "OOM lcm renoise scratch\n");
            h3_fail_tag("FAIL_OOM_DIT");
                    goto cleanup;
                }
                renoise_scratch = nb;
                renoise_scratch_n = need;
            }
            if (pack_audio) {
                if (!apply_sampler_step_f32(
                        sampler_id, audio_rows, audio_vel, n_audio,
                        audio_sigmas[step], audio_sigmas[step + 1],
                        renoise_scratch,
                        seed + 0xA11DULL + 0x10000ULL + (uint64_t)step, error,
                        sizeof(error))) {
                    fprintf(stderr, "audio sampler: %s\n", error);
                    goto cleanup;
                }
            }
            if (!apply_sampler_step_f32(
                    sampler_id, latent, velocity, n_lat, sigmas[step],
                    sigmas[step + 1], renoise_scratch,
                    seed + 0x1C40000ULL + (uint64_t)step, error,
                    sizeof(error))) {
                fprintf(stderr, "sampler: %s\n", error);
                goto cleanup;
            }
            if (step == 0) {
                if (sampler_id == H3_SAMPLER_LCM_RENOISE)
                    printf("sampler: lcm_renoise "
                           "(x0=X+σ·v; X=(1-σ')·x0+σ'·ε; Comfy lcm+simple)\n");
                else
                    printf("sampler: velocity_euler (X += (σ-σ')·v; "
                           "Comfy CONST-equivalent)\n");
            }
        }
        if (getenv("H3_STEP_LATENT_STATS")) {
            double ss = 0.0, ss2 = 0.0;
            size_t si;
            for (si = 0; si < n_lat; si++) {
                double v = latent[si];
                ss += v;
                ss2 += v * v;
            }
            double sm = ss / (double)n_lat;
            double sv = ss2 / (double)n_lat - sm * sm;
            printf("step_latent_stats: step=%d/%d mean=%.6g std=%.6g "
                   "checksum=0x%016llx\n",
                   step + 1, steps, sm, sv > 0.0 ? sqrt(sv) : 0.0,
                   (unsigned long long)h3_dit_w4a8_checksum_f32(latent,
                                                                n_lat));
            if (step == 0) {
                const char *s1 = getenv("H3_STEP1_LATENT_DUMP");
                if (s1 && s1[0]) {
                    FILE *sfp = fopen(s1, "wb");
                    if (sfp) {
                        fwrite(latent, sizeof(float), n_lat, sfp);
                        fclose(sfp);
                        printf("step_latent_stats: wrote step1 raw → %s\n",
                               s1);
                    } else {
                        fprintf(stderr,
                                "step_latent_stats: cannot open %s\n", s1);
                    }
                }
            }
            fflush(stdout);
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
    {
        double rope_ms = 0.0;
        unsigned long rope_calls = 0, hits = 0, misses = 0;
        h3_dit_w4a8_rope_profile_get_ex(&rope_ms, &rope_calls, &hits, &misses);
        if (rope_calls > 0)
            printf("dit_rope_profile: calls=%lu total_ms=%.1f avg_ms=%.2f "
                   "memo_hits=%lu memo_misses=%lu\n",
                   rope_calls, rope_ms, rope_ms / (double)rope_calls, hits,
                   misses);
        h3_dit_w4a8_rope_profile_reset();
    }
    if (host_stage_th_ok) {
        pthread_join(host_stage_th, NULL);
        host_stage_th_ok = 0;
        printf("dit_host_stage: done ok=%d wall_ms=%.1f stage_ms=%.1f "
               "blocks=%u ready=%u%s%s\n",
               host_stage_job.ok, now_ms() - host_stage_t0, host_stage_job.ms,
               host_stage_job.n,
               h3_dit_w4a8_cache(dit)
                   ? h3_w4a8_cache_host_ready_blocks(h3_dit_w4a8_cache(dit))
                   : 0,
               host_stage_job.ok ? "" : " err=",
               host_stage_job.ok
                   ? ""
                   : (host_stage_job.err[0] ? host_stage_job.err : "unknown"));
        fflush(stdout);
    }
    h3_cuda_sol_cute_dump_stats();
    h3_cuda_sage3_dump_stats();
    {
        h3_cuda_sdpa_counters ctr;
        unsigned long long w = 0, h = 0, f = 0, dskip = 0;
        char effective[64];
        char why[128];
        int plugin = 0;
        memset(&ctr, 0, sizeof(ctr));
        h3_cuda_sdpa_counters_get(&ctr);
        if (sdpa && (!strcmp(sdpa, "sage3") || !strcmp(sdpa, "sageattn3"))) {
            w = ctr.sage3_want;
            h = ctr.sage3_hits;
            f = ctr.sage3_fallback;
            plugin = 1;
        } else if (sdpa && (strstr(sdpa, "sol") || strstr(sdpa, "cute"))) {
            w = ctr.sol_cute_want;
            h = ctr.sol_cute_hits;
            f = ctr.sol_cute_fallback;
            dskip = ctr.sol_cute_dense_skip;
            plugin = 1;
        } else {
            w = ctr.cudnn_want;
            h = ctr.cudnn_hits;
            f = ctr.cudnn_fallback;
        }
        h3_kpi_guard_sdpa_effective(sdpa, w, h, f, dskip,
                                    h3_cuda_sdpa_last_backend_name(),
                                    effective, sizeof(effective));
        snprintf(sdpa_effective, sizeof(sdpa_effective), "%s", effective);
        printf("sdpa_backend=%s want=%llu hits=%llu fallback=%llu "
               "dense_skip=%llu request=%s\n",
               effective, w, h, f, dskip, sdpa ? sdpa : "auto");
        if (!plugin)
            printf("dit_graph: eager reason=cudnn_uncapturable\n");
        printf("sdpa_actual: request=%s last=%s sage3 want=%llu hits=%llu "
               "fallback=%llu cudnn want=%llu hits=%llu fallback=%llu\n",
               sdpa, h3_cuda_sdpa_last_backend_name(), ctr.sage3_want,
               ctr.sage3_hits, ctr.sage3_fallback, ctr.cudnn_want,
               ctr.cudnn_hits, ctr.cudnn_fallback);
        fflush(stdout);
        if (plugin && !h3_kpi_guard_sdpa_verdict(w, h, dskip,
                                                 h3_sdpa_strict(), why,
                                                 sizeof(why))) {
            fprintf(stderr,
                    "sdpa_strict: FAIL want=%llu hits=%llu fallback=%llu "
                    "dense_skip=%llu backend=%s reason=%s\n",
                    w, h, f, dskip, effective, why);
            goto cleanup;
        }
        if (h3_sdpa_sage3_require_hit(ctr.sage3_want, ctr.sage3_hits)) {
            fprintf(stderr,
                    "error: sage3 want=%llu hits=%llu fallback=%llu "
                    "(假 hit；H3_SAGE3_REQUIRE_HIT=0 才允许 fallback 续跑)\n",
                    ctr.sage3_want, ctr.sage3_hits, ctr.sage3_fallback);
            goto cleanup;
        }
        if (!plugin && w > 0ull && h == w)
            printf("sdpa_strict: PASS hits==want backend=%s\n", effective);
    }
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
    /* C040: true DiT-resident peak (weights + scratch still on device). */
    h3_vram_ledger_point("peak_dit");
    if (dump_latent_path && dump_latent_path[0]) {
        if (!dump_latent_file(dump_latent_path, latent, n_lat, latent_t,
                              latent_h, latent_w, error, sizeof(error))) {
            fprintf(stderr, "dump_latent: %s\n", error);
            goto cleanup;
        }
        printf("dump_latent: clip=%d wrote %zu f32 (%dx%dx%d) checksum=0x%016llx → %s\n",
               clip_i + 1, n_lat, latent_t, latent_h, latent_w,
               (unsigned long long)h3_dit_w4a8_checksum_f32(latent, n_lat),
               dump_latent_path);
        printf("# H3_DUMP_ONCE path=%s tokens_n/a checksum=0x%016llx "
               "shape=%dx%dx%d layout=THW96\n",
               dump_latent_path,
               (unsigned long long)h3_dit_w4a8_checksum_f32(latent, n_lat),
               latent_t, latent_h, latent_w);
        printf("dump_once_score_many: mode=dump path=%s "
               "do_not_retune_gold=1\n",
               dump_latent_path);
        fflush(stdout);
        /* Exit right after this clip's final latent is dumped, before VAE.
         * A pre-existing clip2 crash (silent exit_group(1), no stderr —
         * reproducible on the untouched baseline too) would otherwise kill
         * the run before the latent reaches the replay A/B. The dump is the
         * only product of this mode, so rc=0 here. Replay with
         * H3_REPLAY_LATENT for the VAE async off/on bit-exact A/B. */
        printf("dump_latent: done; exiting before clip %d VAE\n", clip_i + 1);
        fflush(stdout);
        rc = 0;
        goto cleanup;
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

    /* Drop DiT/TE resident weights before VAE so 50L + decode fits in 24GB.
     * In-process --repeat keeps them until the last clip so INT8 pin / LoRA /
     * sage3 plans are not rebuilt. TAE on 5090 32GB fits beside DiT. */
    if (clip_i + 1 >= clip_repeat) {
        h3_dit_w4a8_core_free(&core);
        gen_uly_free_ranks(dit, uly_dit, uly_cuda);
        h3_dit_w4a8_free(dit);
        dit = NULL;
        if (lora_loaded) {
            h3_lora_free(&lora_set);
            lora_loaded = 0;
        }
        if (refiner_lora_loaded) {
            h3_lora_free(&refiner_lora_set);
            refiner_lora_loaded = 0;
            refiner_lora_attached = 0;
        }
        h3_nvfp4_store_free(&te_store);
        h3_nvfp4_cuda_disable();
        h3_w4a8_store_free(dit_store);
        dit_store = NULL;
        h3_dit_int8_store_free(dit_int8);
        dit_int8 = NULL;
        if (dit_nvfp4_opened) {
            h3_nvfp4_store_free(&dit_nvfp4);
            dit_nvfp4_opened = 0;
        }
        /* Keep the CUDA ctx (cuDNN handle / NVRTC / plans). Trim DiT-sized
         * scratch so 50L decode fits; do not rebuild the device. */
        h3_cuda_release_heavy_workspace(cuda);
        h3_cuda_sage3_drop_pool();
        h3_cuda_sol_cute_drop_pool();
        h3_vram_ledger_point("after_dit_release");
        printf("released DiT/TE before VAE (kept CUDA ctx; trimmed DiT scratch; "
               "dropped sage3/sol_cute pools)\n");
        h3_vram_ledger_point("before_vae");
        fflush(stdout);
        if (preset_out.id == H3_PRESET_TAOMATE && vae_decode &&
            !strcmp(vae_decode, "official")) {
            /* Last clip: DiT gone. Resident W helps GEMM; keep 256–320
             * tiles — 512px waffle (16px patch) on clip2. */
            if (!getenv("H3_VAE_CUDA_W_RESIDENT"))
                setenv("H3_VAE_CUDA_W_RESIDENT", "1", 1);
            if (!getenv("H3_VAE_WEIGHT_CACHE"))
                setenv("H3_VAE_WEIGHT_CACHE", "1", 1);
            printf("vae: last-clip official w_resident=%s cache=%s "
                   "(tiles stay 256–320; 512 waffle)\n",
                   getenv("H3_VAE_CUDA_W_RESIDENT"),
                   getenv("H3_VAE_WEIGHT_CACHE"));
            fflush(stdout);
        }
    } else {
        if (lat_up)
            h3_latent_upscaler_release_device(lat_up);
        if (dit)
            h3_dit_w4a8_release_block_workspace(dit);
        h3_cuda_release_heavy_workspace(cuda);
        if (dit) {
            h3_w4a8_cache *cw = h3_dit_w4a8_cache(dit);
            if (cw) {
                /* Official VAE tiles need a few GiB, not 24. 24GiB keep
                 * dropped window to 4 and the next clip streamed wrong
                 * LRU layers into LBH (mosaic). TAE still wants 24. */
                int tae_keep = vae_decode && !strcmp(vae_decode, "tae");
                size_t keep_free = tae_keep ? ((size_t)24ULL << 30)
                                           : ((size_t)10ULL << 30);
                h3_w4a8_cache_unfreeze_window(cw);
                if (!h3_w4a8_cache_make_room(cw, keep_free, error,
                                             sizeof(error)))
                    fprintf(stderr, "repeat_keep make_room: %s\n", error);
                else
                    printf("repeat_keep: INT8 make_room keep=%.1fGiB "
                           "window=%u (%s)\n",
                           (double)keep_free / (1024.0 * 1024.0 * 1024.0),
                           h3_w4a8_cache_window_size(cw),
                           tae_keep ? "TAE" : "official VAE");
            }
        }
        if (dit && vae_decode && !strcmp(vae_decode, "official")) {
            h3_w4a8_cache *cw2 = h3_dit_w4a8_cache(dit);
            if (cw2) {
                h3_w4a8_cache_drop_device_weights(cw2);
                printf("repeat_keep: dropped INT8 device before official VAE "
                       "(host pin kept; clip2 H2D from pin)\n");
            }
            /* Do not cudaFree sage3 QKVO here: Python _STATE still holds
             * those pointers. Clip 2 DiT would IMA. Last-clip branch
             * drops after DiT is gone. */
        }
        printf("repeat_keep: DiT/LoRA/INT8 host-pin live for clip %d/%d\n",
               clip_i + 2, clip_repeat);
        fflush(stdout);
    }
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
        printf("vae_th: joined_ms=%.1f ok=%d hostreg=%d device=%d "
               "opened=%d work_ms=%.1f load_ms=%.1f%s%s\n",
               now_ms() - vae_prewarm_t0, vae_prewarm_job.ok,
               vae_prewarm_job.hostreg_ok, vae_prewarm_job.device_ok,
               vae_opened, vae_prewarm_job.ms, vae_prewarm_job.load_ms,
               vae_prewarm_job.ok || !vae_prewarm_job.err[0] ? "" : " err=",
               vae_prewarm_job.ok || !vae_prewarm_job.err[0]
                   ? ""
                   : vae_prewarm_job.err);
        fflush(stdout);
        if (vae_prewarm_job.ms > 0.0)
            printf("phase_sum: overlap_ms=%.1f overlap(dit_tail,vae_load)\n",
                   vae_prewarm_job.ms);
    }
    if (!skip_vae) {
        if (!vae_opened) {
            if (!h3_vae_int8_store_open(vae_path, &vae_store, error,
                                        sizeof(error))) {
                fprintf(stderr, "vae open: %s\n", error);
                goto cleanup;
            }
            vae_opened = 1;
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
        audio_overlap = audio_vae_overlap_mode();
        if (decode_audio && audio_overlap == 2) {
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
            audio_job.err[0] = '\0';
            if (pthread_create(&audio_th, NULL, audio_decode_th, &audio_job) ==
                0) {
                audio_th_ok = 1;
                printf("audio_vae: overlap=decode\n");
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
        t0 = now_ms();
        for (i = 0; i < (size_t)vae_repeats; i++) {
            h3_vae_int8_frames run_frames;
            h3_vae_int8_decode_stats run_vst;
            double repeat_ms;
            memset(&run_frames, 0, sizeof(run_frames));
            memset(&run_vst, 0, sizeof(run_vst));
            if (!strcmp(vae_decode, "tae")) {
                float mean[H3_VAE_INT8_LATENT_CH], stdv[H3_VAE_INT8_LATENT_CH];
                int c, tt, hh, ww;
                if (!tae_keep) {
                    fprintf(stderr, "vae decode: TAE ctx missing\n");
                    goto cleanup;
                }
                if (h3_vae_int8_load_latent_stats(&vae_store, mean, stdv, error,
                                                  sizeof(error))) {
                    size_t plane = (size_t)latent_t * (size_t)latent_h *
                                   (size_t)latent_w;
                    for (c = 0; c < H3_VAE_INT8_LATENT_CH; c++)
                        for (tt = 0; tt < latent_t; tt++)
                            for (hh = 0; hh < latent_h; hh++)
                                for (ww = 0; ww < latent_w; ww++) {
                                    size_t ix =
                                        (((size_t)c * (size_t)latent_t +
                                          (size_t)tt) *
                                             (size_t)latent_h +
                                         (size_t)hh) *
                                            (size_t)latent_w +
                                        (size_t)ww;
                                    (void)plane;
                                    latent_chw[ix] =
                                        latent_chw[ix] * stdv[c] + mean[c];
                                }
                }
                if (!h3_tae_decode_frames(tae_keep, latent_chw, latent_t,
                                          latent_h, latent_w, frames_req,
                                          &run_frames, error, sizeof(error))) {
                    fprintf(stderr, "tae decode: %s\n", error);
                    h3_vae_int8_frames_free(&run_frames);
                    goto cleanup;
                }
                run_vst.total_ms = now_ms() - t0;
                run_vst.blocks_ms = run_vst.total_ms;
                printf("vae_decode: tae (非金线 TAEHV taeh3；画质有损)\n");
            } else if (!h3_vae_int8_decode_cuda(cuda, &vae_store, latent_chw,
                                                latent_t, latent_h, latent_w,
                                                vae_blocks, &run_frames,
                                                &run_vst, error,
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
        h3_vram_ledger_point("after_vae");
        {
            double rope_ms = 0.0;
            unsigned long rope_calls = 0, hits = 0, misses = 0;
            h3_vae_int8_rope_profile_get_ex(&rope_ms, &rope_calls, &hits,
                                           &misses);
            if (rope_calls > 0)
                printf("vae_rope_profile: calls=%lu total_ms=%.1f avg_ms=%.2f "
                       "memo_hits=%lu memo_misses=%lu\n",
                       rope_calls, rope_ms, rope_ms / (double)rope_calls, hits,
                       misses);
            h3_vae_int8_rope_profile_reset();
        }

        /* R20: overlap=decode already kicked before video VAE. ppm|0 here. */
        if (decode_audio && !audio_th_ok && !awave.pcm) {
            int want_ov = (audio_overlap == 1);
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
            audio_job.err[0] = '\0';
            if (want_ov &&
                pthread_create(&audio_th, NULL, audio_decode_th, &audio_job) ==
                    0) {
                audio_th_ok = 1;
                printf("audio_vae: overlap=ppm "
                       "(H3_AUDIO_VAE_OVERLAP=decode|0)\n");
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
            int skip_ppm = (skip_ppm_e && skip_ppm_e[0] == '1') ? 1 : 0;
            int f;
            if (skip_ppm_e && skip_ppm_e[0] == '0')
                skip_ppm = 0;
            else if (!skip_ppm_e && preset_out.id == H3_PRESET_TAOMATE &&
                     mux_path)
                skip_ppm = 1;
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
                printf("ppm: skipped (mux-only taomate; H3_SKIP_PPM=0 restores)\n");
            }
            t_ppm = now_ms() - t_io0;
            if (mux_rgb && replay_latent) {
                /* Checksum the RAW f32 VAE output (frames.rgb), not the
                 * uint8-quantized mux buffer: quantization can mask a real
                 * f32 difference. This is the bit-exact verdict for the
                 * async off/on A/B. */
                size_t fn = (size_t)frames.frames * (size_t)frames.width *
                            (size_t)frames.height * 3;
                size_t fb = (size_t)frames.width * (size_t)frames.height * 3;
                int nf = frames.frames, fi;
                const char *av = getenv("H3_VAE_UNPATCH_ASYNC");
                printf("replay_rgb: frames=%d %dx%d f32_total=%zu "
                       "checksum=0x%016llx unpatch_async=%s\n",
                       nf, frames.width, frames.height, fn,
                       (unsigned long long)fnv1a64_bytes(
                           (const uint8_t *)frames.rgb, fn * sizeof(float)),
                       (av && av[0] == '1' && av[1] == '\0') ? "on" : "off");
                for (fi = 0; fi < nf; fi++)
                    printf("replay_rgb_frame[%d]: 0x%016llx\n", fi,
                           (unsigned long long)fnv1a64_bytes(
                               (const uint8_t *)(frames.rgb +
                                                 (size_t)fi * fb),
                               fb * sizeof(float)));
                fflush(stdout);
            }
            if (mux_rgb)
                h3_waffle_log_rgb24(mux_rgb, frames.frames, frames.width,
                                    frames.height);
            /* Cold gate (observational, WARN-only): clip1 b16 stats are
             * aggregated separately from the hot golden line with an
             * independent, looser threshold (cold clip1 is naturally
             * noisier). The waffle: lines above stay untouched so the hot
             * golden regex keeps working. */
            if (clip_i == 0 && mux_rgb) {
                static const int wf_idx[] = {0, 24, 48, 90, 119};
                double b16[5];
                double max_b16 = 0.0, mean_b16 = 0.0;
                int wf, nf = frames.frames;
                int n_low = 0, n_valid = 0;
                size_t fb = (size_t)frames.width * (size_t)frames.height * 3;
                for (wf = 0; wf < 5; wf++) {
                    int f = wf_idx[wf];
                    int low = 0;
                    if (f >= nf) f = nf - 1;
                    if (f < 0) f = 0;
                    b16[wf] = h3_waffle_b16_gated_rgb24(mux_rgb + (size_t)f * fb,
                                                        frames.width,
                                                        frames.height,
                                                        NULL, &low);
                    if (low) {
                        n_low++;
                        b16[wf] = 0.0;
                    } else {
                        n_valid++;
                        if (b16[wf] > max_b16) max_b16 = b16[wf];
                        mean_b16 += b16[wf];
                    }
                }
                if (n_valid > 0)
                    mean_b16 /= (double)n_valid;
                printf("waffle_cold_clip1: max_b16=%.3f mean_b16=%.3f "
                       "threshold=%.3f low_energy_frames=%d/5\n",
                       max_b16, mean_b16, waffle_cold_max_b16, n_low);
                if (n_low == 5)
                    printf("waffle_guard: WARN cold clip1 all 5 frames "
                           "INVALID_LOWENERGY (no semantic energy; b16 "
                           "meaningless — likely pure-noise output)\n");
                else if (!h3_kpi_guard_waffle_cold_verdict(
                        max_b16, waffle_cold_max_b16, NULL, 0)) {
                    printf("waffle_guard: WARN cold clip1 max_b16=%.3f > "
                           "threshold=%.3f (observational; H3_WAFFLE_COLD_MAX "
                           "to tune; no FAIL — hot golden line unchanged)\n",
                           max_b16, waffle_cold_max_b16);
                    if (waffle_cold_guard)
                        goto waffle_cold_retry;
                }
                fflush(stdout);
            }
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
               "audio_ms=%.1f overlap=%s\n",
               ast.load_ms, ast.decode_ms, ast.total_ms, awave.samples,
               t_audio, audio_overlap == 2 ? "decode" : "ppm");
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
waffle_cold_retry:
        if (waffle_cold_guard && clip_i == 0 && latent_chw && frames.rgb &&
            waffle_cold_max_b16 > 0.0 &&
            strcmp(vae_decode, "official") == 0) {
            /* Re-decode clip1 (official VAE, same latent): bit-identical
             * when the flaky high-noise decode did not fire. The WARN above
             * already logged the pre-retry stats. */
            h3_vae_int8_frames retry_frames;
            h3_vae_int8_decode_stats retry_vst;
            double rt0 = now_ms();
            memset(&retry_frames, 0, sizeof(retry_frames));
            memset(&retry_vst, 0, sizeof(retry_vst));
            if (!h3_vae_int8_decode_cuda(cuda, &vae_store, latent_chw,
                                         latent_t, latent_h, latent_w,
                                         vae_blocks, &retry_frames,
                                         &retry_vst, error, sizeof(error))) {
                fprintf(stderr, "waffle_guard: cold retry decode: %s\n",
                        error);
                h3_vae_int8_frames_free(&retry_frames);
                goto cleanup;
            }
            printf("waffle_guard: cold clip1 retry decode total=%.1f "
                   "(H3_WAFFLE_COLD_GUARD=1)\n",
                   now_ms() - rt0);
            h3_vae_int8_frames_free(&frames);
            frames = retry_frames;
            vst = retry_vst;
            free(mux_rgb);
            mux_rgb = NULL;
            waffle_cold_guard = 0; /* one retry per cold clip */
        }
        {
            /* Waffle-cold retry rewrites frames; the label target sits in
             * the guarded block above. */
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
                {
                    h3_ffmpeg_av_stats mux_st;
                    const char *mux_out = mux_write ? mux_write : mux_path;
                    memset(&mux_st, 0, sizeof(mux_st));
                    if (!h3_ffmpeg_write_av_rgb24_f32_ex(
                            mux_out, mux_rgb, mux_frames, frames.width,
                            frames.height, 24, awave.pcm, awave.samples,
                            awave.channels, awave.sample_rate, &mux_st, error,
                            sizeof(error))) {
                        fprintf(stderr, "mux: %s\n", error);
                        goto cleanup;
                    }
                    printf("wrote mux %s (%d/%d frames + %d samples)\n",
                           mux_out, mux_frames, frames.frames, awave.samples);
                    printf("mux: pipe_wait_ms=%.1f encode_ms=%.1f "
                           "frames_pushed=%d venc=%s threads=%d\n",
                           mux_st.pipe_wait_ms, mux_st.encode_ms,
                           mux_st.frames_pushed, h3_ffmpeg_venc(),
                           h3_ffmpeg_threads());
                    if (!h3_av_sync_log_file(mux_out, (double)seconds, 0.05,
                                             error, sizeof(error))) {
                        fprintf(stderr, "av_sync_strict: FAIL reason=%s\n",
                                error);
                        rc = 5;
                    }
                }
            } else {
                if (!h3_ffmpeg_write_rgb24(mux_write ? mux_write : mux_path,
                                           mux_rgb, mux_frames, frames.width,
                                           frames.height, 24, error,
                                           sizeof(error))) {
                    fprintf(stderr, "mux video-only: %s\n", error);
                    goto cleanup;
                }
                printf("wrote mux %s (%d/%d frames @24fps = %.4fs wall, "
                       "video-only%s)\n",
                       mux_write ? mux_write : mux_path, mux_frames,
                       frames.frames, (double)mux_frames / 24.0,
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

    printf("phase_ms: te=%.1f dit=%.1f lift_switch=%.1f vae=%.1f vae_load=%.1f "
           "vae_blocks=%.1f "
           "audio=%.1f ppm=%.1f mux=%.1f "
           "total=%.1f dit_evals=%d dit_skips=%d hot=%d clip=%d/%d\n",
           t_te, t_dit, t_lift_switch, t_vae, vst.load_ms, vst.blocks_ms,
           t_audio, t_ppm, t_mux, now_ms() - t_all0, dit_evals, dit_skips,
           clip_i > 0 ? 1 : 0, clip_i + 1, clip_repeat);
    gen_mem_trace(cuda, clip_i > 0 ? "vae_done_hot" : "vae_done");
    if (report_path) {
        h3_run_report_in rin;
        char rpath[1024];
        memset(&rin, 0, sizeof(rin));
        if (clip_repeat > 1)
            snprintf(rpath, sizeof(rpath), "%s.clip%d.json", report_path,
                     clip_i + 1);
        else
            snprintf(rpath, sizeof(rpath), "%s", report_path);
        rin.path = rpath;
        rin.preset = sdpa;
        rin.sdpa_backend = sdpa_effective[0] ? sdpa_effective : sdpa;
        rin.vae_attn_backend =
            vst.attn_backend[0] ? vst.attn_backend : NULL;
        rin.mode = pack_audio ? "t2va" : "t2v";
        rin.quant = "int8_convrot";
        rin.width = width;
        rin.height = height;
        rin.seconds = seconds;
        rin.hot = clip_i > 0 ? 1 : 0;
        rin.clip = clip_i + 1;
        rin.clip_n = clip_repeat;
        rin.te_ms = t_te;
        rin.dit_ms = t_dit;
        rin.vae_load_ms = vst.load_ms;
        rin.vae_blocks_ms = vst.blocks_ms;
        rin.audio_ms = t_audio;
        rin.mux_ms = t_mux;
        rin.total_ms = now_ms() - t_all0;
        if (!h3_run_report_write(&rin, error, sizeof(error)))
            fprintf(stderr, "report: %s\n", error);
        else
            printf("report: wrote %s\n", rpath);
    }
    printf("repeat_core: dit+vae+mux=%.1f vs_kpi=27s (clip=%d hot=%d)\n",
           t_dit + t_vae + t_mux, clip_i + 1, clip_i > 0 ? 1 : 0);
    if (pack_audio)
        printf("summary: %s pack PASS (audio_pack=on decode=%s)\n",
               (n_ref_images || ref_video_path) ? "ref2va" : "fl2va",
               decode_audio ? "on" : "off");
    else
        printf("summary: text|video PASS (audio_pack=off; theme may fail)\n");
    if (clip_i + 1 < clip_repeat) {
        int rel = h3_vae_int8_repeat_release_mode();
        if (rel == H3_VAE_REPEAT_FULL) {
            h3_vae_int8_cuda_release_full();
            printf("repeat_keep: VAE pack+device dropped before clip %d DiT "
                   "(H3_VAE_REPEAT_RELEASE=full)\n",
                   clip_i + 2);
        } else if (rel == H3_VAE_REPEAT_DEVICE) {
            h3_vae_int8_cuda_release_device();
            printf("repeat_keep: dropped VAE GPU W/scratch before clip %d DiT "
                   "(host pack+mmap pin kept)\n",
                   clip_i + 2);
        } else {
            printf("repeat_keep: VAE device W resident for clip %d "
                   "(vae_load≈0; H3_VAE_REPEAT_RELEASE=device|full to drop)\n",
                   clip_i + 2);
        }
        fflush(stdout);
        clip_i++;
        goto clip_denoise;
    }
    if (rc != 5) rc = 0;

cleanup:
    if (audio_th_ok) pthread_join(audio_th, NULL);
    if (sdpa_th_ok) pthread_join(sdpa_th, NULL);
    if (vae_th_ok) pthread_join(vae_th, NULL);
    if (host_stage_th_ok) pthread_join(host_stage_th, NULL);
    if (prefetch_done) cudaEventDestroy((cudaEvent_t)prefetch_done);
    free(renoise_scratch);
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
    h3_latent_upscaler_close(lat_up);
    lat_up = NULL;
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
    h3_vae_int8_cuda_hostreg_release();
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
    if (refiner_lora_loaded) {
        h3_lora_free(&refiner_lora_set);
        refiner_lora_loaded = 0;
        refiner_lora_attached = 0;
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
    if (dit_nvfp4_opened) {
        h3_nvfp4_store_free(&dit_nvfp4);
        dit_nvfp4_opened = 0;
    }
    if (tae_keep) {
        h3_tae_close(tae_keep);
        tae_keep = NULL;
    }
    return rc;
}
