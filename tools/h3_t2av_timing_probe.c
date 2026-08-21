/* T2AV timing probe (CUDA W4A8 bypass): shape math + phased wall times.
 *
 * Does NOT replace Metal `h3` CLI. Measures what the current NVIDIA path can
 * actually run: TE embed stub, DiT video forward (CPU SDPA), optional VAE.
 * Audio DiT tokens / audio VAE decode are reported as gaps when missing.
 *
 * Build:
 *   cmake --build build-cuda -j --target h3_t2av_timing_probe
 * Example:
 *   ./build-cuda/h3_t2av_timing_probe --seconds 5 --width 1280 --height 704 \
 *     --layers 50 --steps 20 --window 2 --dit-seq 1024 --dit-layers 1 \
 *     --dit DIT.safetensors --te TE.safetensors --vae VAE.safetensors
 */
#include "h3_dit_w4a8.h"
#include "h3_nvfp4.h"
#include "h3_vae_int8.h"
#include "h3_vae_int8_cuda.h"

#include <cuda_runtime_api.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Mirror h3_host.h constants / shape helpers (avoid linking Accelerate host). */
#define H3_CANVAS_MULTIPLE 32
#define H3_MAX_PIXELS (768 * 1344)
#define H3_FPS 24
#define H3_AUDIO_LATENT_FPS 40
#define H3_VAE_SPATIAL_RATIO 16
#define H3_DEFAULT_WIDTH 864
#define H3_DEFAULT_HEIGHT 480
#define H3_DEFAULT_STEPS 20
#define H3_DEFAULT_DIT_LAYERS 50

static int align_frame_count(int requested) {
    int value = requested < 5 ? 5 : requested;
    int remainder = (value - 5) % 17;
    if (remainder != 0) value += 17 - remainder;
    return value;
}

static int video_latent_t(int frame_count) {
    if (frame_count <= 5) return 2;
    return ((frame_count - 5) / 17) * 5 + 2;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

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
        snprintf(buf, buf_size, "%.3f GiB", gb);
    else
        snprintf(buf, buf_size, "%.1f MiB", mb);
    return buf;
}

static int canvas_ok(int w, int h) {
    return w >= 32 && h >= 32 && (w % H3_CANVAS_MULTIPLE) == 0 &&
           (h % H3_CANVAS_MULTIPLE) == 0 &&
           (int64_t)w * (int64_t)h <= (int64_t)H3_MAX_PIXELS;
}

int main(int argc, char **argv) {
    const char *dit_path = NULL;
    const char *te_path = NULL;
    const char *vae_path = NULL;
    int width = H3_DEFAULT_WIDTH;
    int height = H3_DEFAULT_HEIGHT;
    double seconds = 5.0;
    int steps = H3_DEFAULT_STEPS;
    unsigned layers = H3_DEFAULT_DIT_LAYERS;
    unsigned dit_layers = 1;
    size_t dit_seq = 0; /* 0 → use full video_seq (may OOM / be very slow) */
    size_t context = 256;
    unsigned window_size = 0; /* 0 → cache auto max_blocks_fit */
    unsigned vae_blocks = 1;
    unsigned reuse = 1; /* Metal core_reuse_interval analogue: compute 1/reuse steps */
    float token_reduce = 1.0f; /* 1.0 = full video_seq; <1 scales run_seq */
    int attn_window = 0; /* 0=global SDPA; >0 token window (quality risk) */
    const char *sdpa_backend_pref = NULL;
    const char *preset = NULL; /* product | quality */
    int device = 0;
    int skip_dit = 0;
    int skip_te = 0;
    int skip_vae = 0;
    int cpu_sdpa = 0;
    int cpu_act = 0;
    int sdpa_f32 = 0;
    int check_sdpa = 0;
    int e2e_loop = 0; /* repeat dit_layers stack across effective steps */
    int force_flash = 0;
    char error[512];
    char b1[64], b2[64], b3[64];

    int frames_req, frames, latent_t, latent_h, latent_w, audio_t;
    size_t video_seq, target_seq, run_seq;
    size_t i;
    int iarg;

    h3_w4a8_store *dit_store = NULL;
    h3_cuda *cuda = NULL;
    h3_dit_w4a8 *dit = NULL;
    h3_dit_w4a8_core core;
    h3_w4a8_cache_config cfg;
    h3_w4a8_cache_stats cst;
    h3_cuda_device_info info;
    h3_nvfp4_store te_store;
    h3_vae_int8_store vae_store;
    h3_vae_int8_frames vae_frames;
    h3_vae_int8_decode_stats vst;

    double t0, t1, wall0, wall1;
    double te_ms = 0.0, prefetch_ms = 0.0, dit_ms = 0.0, vae_ms = 0.0;
    double sdpa_ms = 0.0, gemm_ms = 0.0, adaln_ms = 0.0, rope_ms = 0.0;
    double swiglu_ms = 0.0, other_ms = 0.0, act_prep_ms = 0.0;
    char sdpa_backend[32] = "unset";
    size_t free0 = 0, total0 = 0;
    int rc = 1;

    memset(&core, 0, sizeof(core));
    memset(&te_store, 0, sizeof(te_store));
    memset(&vae_store, 0, sizeof(vae_store));
    memset(&vae_frames, 0, sizeof(vae_frames));

    for (iarg = 1; iarg < argc; iarg++) {
        if (!strcmp(argv[iarg], "--dit") && iarg + 1 < argc) {
            dit_path = argv[++iarg];
        } else if (!strcmp(argv[iarg], "--te") && iarg + 1 < argc) {
            te_path = argv[++iarg];
        } else if (!strcmp(argv[iarg], "--vae") && iarg + 1 < argc) {
            vae_path = argv[++iarg];
        } else if (!strcmp(argv[iarg], "--width") && iarg + 1 < argc) {
            width = (int)strtol(argv[++iarg], NULL, 10);
        } else if (!strcmp(argv[iarg], "--height") && iarg + 1 < argc) {
            height = (int)strtol(argv[++iarg], NULL, 10);
        } else if (!strcmp(argv[iarg], "--seconds") && iarg + 1 < argc) {
            seconds = strtod(argv[++iarg], NULL);
        } else if (!strcmp(argv[iarg], "--steps") && iarg + 1 < argc) {
            steps = (int)strtol(argv[++iarg], NULL, 10);
        } else if (!strcmp(argv[iarg], "--layers") && iarg + 1 < argc) {
            layers = (unsigned)strtoul(argv[++iarg], NULL, 10);
        } else if (!strcmp(argv[iarg], "--dit-layers") && iarg + 1 < argc) {
            dit_layers = (unsigned)strtoul(argv[++iarg], NULL, 10);
        } else if (!strcmp(argv[iarg], "--dit-seq") && iarg + 1 < argc) {
            dit_seq = (size_t)strtoul(argv[++iarg], NULL, 10);
        } else if (!strcmp(argv[iarg], "--context") && iarg + 1 < argc) {
            context = (size_t)strtoul(argv[++iarg], NULL, 10);
        } else if (!strcmp(argv[iarg], "--window") && iarg + 1 < argc) {
            window_size = (unsigned)strtoul(argv[++iarg], NULL, 10);
        } else if (!strcmp(argv[iarg], "--attn-window") && iarg + 1 < argc) {
            attn_window = (int)strtol(argv[++iarg], NULL, 10);
            if (attn_window < 0) attn_window = 0;
        } else if (!strcmp(argv[iarg], "--sdpa-backend") && iarg + 1 < argc) {
            sdpa_backend_pref = argv[++iarg];
        } else if (!strcmp(argv[iarg], "--preset") && iarg + 1 < argc) {
            preset = argv[++iarg];
        } else if (!strcmp(argv[iarg], "--vae-blocks") && iarg + 1 < argc) {
            vae_blocks = (unsigned)strtoul(argv[++iarg], NULL, 10);
        } else if (!strcmp(argv[iarg], "--device") && iarg + 1 < argc) {
            device = (int)strtol(argv[++iarg], NULL, 10);
        } else if (!strcmp(argv[iarg], "--skip-dit")) {
            skip_dit = 1;
        } else if (!strcmp(argv[iarg], "--skip-te")) {
            skip_te = 1;
        } else if (!strcmp(argv[iarg], "--skip-vae")) {
            skip_vae = 1;
        } else if (!strcmp(argv[iarg], "--cpu-sdpa")) {
            cpu_sdpa = 1;
        } else if (!strcmp(argv[iarg], "--cpu-act")) {
            cpu_act = 1;
        } else if (!strcmp(argv[iarg], "--sdpa-f32")) {
            sdpa_f32 = 1;
        } else if (!strcmp(argv[iarg], "--force-flash")) {
            force_flash = 1;
        } else if (!strcmp(argv[iarg], "--check-sdpa")) {
            check_sdpa = 1;
        } else if (!strcmp(argv[iarg], "--e2e")) {
            e2e_loop = 1;
        } else if (!strcmp(argv[iarg], "--reuse") && iarg + 1 < argc) {
            reuse = (unsigned)strtoul(argv[++iarg], NULL, 10);
            if (reuse < 1) reuse = 1;
        } else if (!strcmp(argv[iarg], "--token-reduce") && iarg + 1 < argc) {
            token_reduce = (float)strtod(argv[++iarg], NULL);
            if (token_reduce <= 0.0f) token_reduce = 1.0f;
            if (token_reduce > 1.0f) token_reduce = 1.0f;
        } else if (!strcmp(argv[iarg], "-h") || !strcmp(argv[iarg], "--help")) {
            fprintf(stderr,
                    "usage: %s --dit DIT [--te TE] [--vae VAE] "
                    "[--seconds 5 --width 864 --height 480] "
                    "[--steps 20 --layers 50] [--dit-seq N] [--dit-layers N] "
                    "[--window W] [--attn-window W] [--reuse N] "
                    "[--token-reduce F] [--preset product|quality] [--e2e] "
                    "[--sdpa-backend auto|flash|tc|cudnn] [--force-flash] "
                    "[--cpu-sdpa] [--sdpa-f32] [--cpu-act] [--check-sdpa]\n",
                    argv[0]);
            return 0;
        } else if (argv[iarg][0] == '-') {
            fprintf(stderr, "unknown option: %s\n", argv[iarg]);
            return 2;
        } else if (!dit_path) {
            dit_path = argv[iarg];
        }
    }

    /* Presets: product = fast knobs; quality = push full layers/steps/VAE. */
    if (preset) {
        if (!strcmp(preset, "product")) {
            steps = 4;
            dit_layers = 4;
            layers = 50;
            reuse = 2;
            token_reduce = 0.35f;
            if (attn_window <= 0) attn_window = 256;
            vae_blocks = 8;
            e2e_loop = 1;
            printf("preset=product: steps=4 dit_layers=4 reuse=2 "
                   "token_reduce=0.35 attn_window=%d vae_blocks=%u "
                   "(QUALITY RISK vs full global attn)\n",
                   attn_window, vae_blocks);
        } else if (!strcmp(preset, "quality")) {
            steps = 20;
            dit_layers = 50;
            layers = 50;
            reuse = 1;
            token_reduce = 1.0f;
            attn_window = 0;
            vae_blocks = 36;
            e2e_loop = 1;
            printf("preset=quality: steps=20 dit_layers=50 reuse=1 "
                   "token_reduce=1 attn_window=0 vae_blocks=36 "
                   "(full-ish; may be very slow)\n");
        } else {
            fprintf(stderr, "unknown --preset %s (use product|quality)\n",
                    preset);
            return 2;
        }
    }

    wall0 = now_ms();
    frames_req = (int)llround(seconds * (double)H3_FPS);
    if (frames_req < 1) frames_req = 1;
    frames = align_frame_count(frames_req);
    latent_t = video_latent_t(frames);
    latent_w = width / H3_VAE_SPATIAL_RATIO;
    latent_h = height / H3_VAE_SPATIAL_RATIO;
    audio_t = (int)llround((double)frames * (double)H3_AUDIO_LATENT_FPS /
                           (double)H3_FPS);
    video_seq = (size_t)latent_t * (size_t)(latent_h / 2) * (size_t)(latent_w / 2);
    target_seq = context + video_seq + (size_t)audio_t; /* audio≈1 tok/timestep */
    run_seq = dit_seq ? dit_seq : video_seq;
    if (!dit_seq && token_reduce < 1.0f) {
        run_seq = (size_t)llround((double)video_seq * (double)token_reduce);
        if (run_seq < 1) run_seq = 1;
    }

    printf("=== T2AV shape (official CLI math) ===\n");
    printf("request: --seconds %.3g → frames_req=%d → aligned=%d (%.3fs @ %dfps)\n",
           seconds, frames_req, frames, (double)frames / H3_FPS, H3_FPS);
    printf("canvas: %dx%d valid=%d (H3 max %dx%d product; default CLI %dx%d)\n",
           width, height, canvas_ok(width, height), 1344, 768,
           H3_DEFAULT_WIDTH, H3_DEFAULT_HEIGHT);
    printf("note: 1280x720 INVALID (h%%32!=0); closest 720p-class=%dx%d; "
           "480p landscape default=%dx%d\n",
           1280, 704, H3_DEFAULT_WIDTH, H3_DEFAULT_HEIGHT);
    printf("latent: t=%d h=%d w=%d (VAE spatial /%d, patch 2x2)\n", latent_t,
           latent_h, latent_w, H3_VAE_SPATIAL_RATIO);
    printf("tokens: video_seq=%zu  audio_t≈%d (not wired in W4A8 DiT)  "
           "text_context=%zu  target_seq≈%zu\n",
           video_seq, audio_t, context, target_seq);
    printf("schedule: steps=%d layers=%u reuse=%u token_reduce=%.3g e2e=%d\n",
           steps, layers, reuse, token_reduce, e2e_loop);
    printf("measure: dit_seq=%zu dit_layers=%u cache_window=%u attn_window=%d "
           "sdpa=%s backend_pref=%s\n",
           run_seq, dit_layers, window_size, attn_window,
           cpu_sdpa ? "cpu_naive" : (sdpa_f32 ? "cuda_flash_f32" : "cuda_flash_f16"),
           sdpa_backend_pref ? sdpa_backend_pref
                             : (force_flash ? "flash" : "auto"));
    if (attn_window > 0)
        printf("WARN: attn_window=%d is NOT training-faithful global attention; "
               "for speed A/B only\n",
               attn_window);
    printf("gaps: G2 audio DiT+VAE (BLOCKED on CUDA); tokenizer/visual TE; "
           "Metal generate CLI not replaced (CUDA bypass)\n\n");

    if (!dit_path) {
        fprintf(stderr, "need --dit\n");
        return 2;
    }
    if (!canvas_ok(width, height) || (latent_h % 2) || (latent_w % 2)) {
        fprintf(stderr, "invalid canvas / odd latent\n");
        return 2;
    }
    if (dit_layers == 0 || dit_layers > layers || layers > H3_DIT_W4A8_N_BLOCKS) {
        fprintf(stderr, "invalid layers\n");
        return 2;
    }

    /* ---- TE embed (phase, then free) ---- */
    if (!skip_te && te_path) {
        float *emb = NULL;
        int32_t *ids = NULL;
        size_t out_tok = 0;
        printf("=== phase TE embed stub ===\n");
        t0 = now_ms();
        if (!h3_nvfp4_store_open(te_path, &te_store, error, sizeof(error))) {
            fprintf(stderr, "te open failed: %s\n", error);
            goto cleanup;
        }
        ids = (int32_t *)malloc(context * sizeof(int32_t));
        if (!ids) {
            fprintf(stderr, "TE OOM\n");
            goto cleanup;
        }
        for (i = 0; i < context; i++)
            ids[i] = (int32_t)((i * 131 + 17) % 32000);
        emb = h3_nvfp4_make_condition_f32(&te_store, ids, context, 0x5445ULL,
                                          NULL, &out_tok, error, sizeof(error));
        if (!emb) {
            fprintf(stderr, "te embed failed: %s\n", error);
            free(ids);
            goto cleanup;
        }
        t1 = now_ms();
        te_ms = t1 - t0;
        printf("te_embed: tokens=%zu dim=%d wall_ms=%.1f  (lookup only; "
               "no 32B transformer)\n",
               out_tok, H3_NVFP4_TEXT_DIM, te_ms);
        free(ids);
        free(emb);
        h3_nvfp4_store_free(&te_store);
        memset(&te_store, 0, sizeof(te_store));
    } else {
        printf("=== phase TE: skipped ===\n");
    }

    /* ---- DiT timed forward ---- */
    if (!skip_dit) {
        float *x = NULL, *y = NULL, *cos = NULL, *sin = NULL;
        float *modulation = NULL;
        float time_feat[H3_DIT_W4A8_TIME_FEAT];
        int *pt = NULL, *ph = NULL, *pw = NULL;
        uint32_t *row_map = NULL;
        h3_dit_w4a8_block_aux aux0;
        unsigned b;
        int dit_bufs_pinned = 0;

        memset(&aux0, 0, sizeof(aux0));
        if (cpu_sdpa) setenv("H3_DIT_W4A8_FORCE_CPU_SDPA", "1", 1);
        else
            unsetenv("H3_DIT_W4A8_FORCE_CPU_SDPA");
        if (sdpa_f32) setenv("H3_DIT_W4A8_FORCE_SDPA_F32", "1", 1);
        else
            unsetenv("H3_DIT_W4A8_FORCE_SDPA_F32");
        if (cpu_act) setenv("H3_DIT_W4A8_FORCE_CPU_ACT", "1", 1);
        else
            unsetenv("H3_DIT_W4A8_FORCE_CPU_ACT");
        {
            char aw[32];
            snprintf(aw, sizeof(aw), "%d", attn_window);
            setenv("H3_DIT_W4A8_ATTN_WINDOW", aw, 1);
        }
        if (force_flash)
            setenv("H3_CUDA_SDPA_BACKEND", "flash", 1);
        else if (sdpa_backend_pref)
            setenv("H3_CUDA_SDPA_BACKEND", sdpa_backend_pref, 1);
        else
            setenv("H3_CUDA_SDPA_BACKEND", "auto", 1);
        printf("\n=== phase DiT (W4A8 CUDA; sdpa=%s act=%s attn_window=%d "
               "backend=%s) ===\n",
               cpu_sdpa ? "cpu_naive"
                        : (sdpa_f32 ? "cuda_flash_f32" : "cuda_flash_f16"),
               cpu_act ? "cpu_convrot+quant" : "cuda_convrot+quant",
               attn_window, getenv("H3_CUDA_SDPA_BACKEND"));
        dit_store = h3_w4a8_store_open(dit_path, error, sizeof(error));
        if (!dit_store) {
            fprintf(stderr, "dit open failed: %s\n", error);
            goto cleanup;
        }
        cuda = h3_cuda_create(device, error, sizeof(error));
        if (!cuda) {
            fprintf(stderr, "cuda create failed: %s\n", error);
            goto cleanup;
        }
        h3_cuda_get_info(cuda, &info);
        h3_cuda_mem_info(cuda, &free0, &total0, NULL, 0);
        printf("cuda: [%d] %s  vram_free=%s / %s\n", info.device_index,
               info.name, format_bytes(free0, b1, sizeof(b1)),
               format_bytes(total0, b2, sizeof(b2)));

        h3_dit_w4a8_config_init(&cfg);
        cfg.policy = H3_W4A8_CACHE_WINDOW;
        /* window_size=0 → h3_w4a8_cache_open picks max_blocks_fit (anti-thrash). */
        cfg.window_size = window_size;
        dit = h3_dit_w4a8_open(cuda, dit_store, &cfg, error, sizeof(error));
        if (!dit) {
            fprintf(stderr, "dit open failed: %s\n", error);
            goto cleanup;
        }
        if (h3_dit_w4a8_get_cache_stats(dit, &cst, error, sizeof(error))) {
            printf("cache_open: policy=%s window=%u max_fit=%u reserve_aware=1\n",
                   h3_w4a8_cache_policy_name(cst.effective_policy),
                   cst.window_size, cst.max_blocks_fit);
        }
        if (!h3_dit_w4a8_load_core(dit_store, &core, error, sizeof(error))) {
            fprintf(stderr, "load_core failed: %s\n", error);
            goto cleanup;
        }
        if (!h3_dit_w4a8_load_block_aux(dit_store, 0, &aux0, error,
                                        sizeof(error))) {
            fprintf(stderr, "load aux0 failed: %s\n", error);
            goto cleanup;
        }

        x = NULL;
        y = NULL;
        cos = NULL;
        sin = NULL;
        pt = NULL;
        ph = NULL;
        pw = NULL;
        row_map = NULL;
        modulation = NULL;
        dit_bufs_pinned = 0;
        if (cudaMallocHost((void **)&x,
                           run_seq * H3_DIT_W4A8_HIDDEN * sizeof(float)) ==
                cudaSuccess &&
            cudaMallocHost((void **)&y,
                           run_seq * H3_DIT_W4A8_HIDDEN * sizeof(float)) ==
                cudaSuccess &&
            cudaMallocHost((void **)&cos,
                           run_seq * H3_DIT_W4A8_ROPE_HALF * sizeof(float)) ==
                cudaSuccess &&
            cudaMallocHost((void **)&sin,
                           run_seq * H3_DIT_W4A8_ROPE_HALF * sizeof(float)) ==
                cudaSuccess) {
            dit_bufs_pinned = 1;
            printf("dit_host_buffers: pinned\n");
        } else {
            if (x) cudaFreeHost(x);
            if (y) cudaFreeHost(y);
            if (cos) cudaFreeHost(cos);
            if (sin) cudaFreeHost(sin);
            x = (float *)malloc(run_seq * H3_DIT_W4A8_HIDDEN * sizeof(float));
            y = (float *)malloc(run_seq * H3_DIT_W4A8_HIDDEN * sizeof(float));
            cos = (float *)malloc(run_seq * H3_DIT_W4A8_ROPE_HALF * sizeof(float));
            sin = (float *)malloc(run_seq * H3_DIT_W4A8_ROPE_HALF * sizeof(float));
            printf("dit_host_buffers: pageable_fallback\n");
        }
        pt = (int *)malloc(run_seq * sizeof(int));
        ph = (int *)malloc(run_seq * sizeof(int));
        pw = (int *)malloc(run_seq * sizeof(int));
        row_map = (uint32_t *)calloc(run_seq, sizeof(uint32_t));
        modulation = (float *)malloc(H3_DIT_W4A8_ADALN_OUT * sizeof(float));
        if (!x || !y || !cos || !sin || !pt || !ph || !pw || !row_map ||
            !modulation) {
            fprintf(stderr, "DiT host OOM at seq=%zu\n", run_seq);
            goto dit_fail;
        }
        for (i = 0; i < run_seq * H3_DIT_W4A8_HIDDEN; i++)
            x[i] = hash_to_unit(0x54324156ULL, i);
        for (i = 0; i < run_seq; i++) {
            pt[i] = (int)(i % (size_t)(latent_t > 0 ? latent_t : 1));
            ph[i] = (int)((i / 2) % (size_t)(latent_h > 0 ? latent_h / 2 : 1));
            pw[i] = (int)(i % (size_t)(latent_w > 0 ? latent_w / 2 : 1));
        }
        if (!h3_dit_w4a8_rope_tables_f32(aux0.inv_freq, run_seq, pt, ph, pw,
                                         1.0f, cos, sin, error, sizeof(error))) {
            fprintf(stderr, "rope failed: %s\n", error);
            goto dit_fail;
        }
        if (!h3_dit_w4a8_time_feat_from_table(&aux0, 512, time_feat, error,
                                              sizeof(error)) ||
            !h3_dit_w4a8_adaln_project_f32(&aux0, time_feat, modulation, error,
                                           sizeof(error))) {
            fprintf(stderr, "adaln setup failed: %s\n", error);
            goto dit_fail;
        }

        if (check_sdpa) {
            /* Numerical: CUDA flash vs CPU naive on a short random QKV. */
            const size_t cseq = run_seq > 128 ? 128 : run_seq;
            const size_t heads = H3_DIT_W4A8_HEADS;
            const size_t hd = H3_DIT_W4A8_HEAD_DIM;
            const size_t nq = cseq * heads * hd;
            float *q = (float *)malloc(nq * sizeof(float));
            float *k = (float *)malloc(nq * sizeof(float));
            float *v = (float *)malloc(nq * sizeof(float));
            float *o_cpu = (float *)malloc(nq * sizeof(float));
            float *o_gpu = (float *)malloc(nq * sizeof(float));
            float scale = 1.0f / sqrtf((float)hd);
            float max_abs = 0.0f;
            double rmse = 0.0;
            size_t ii;
            int ok_chk = 0;
            if (q && k && v && o_cpu && o_gpu) {
                for (ii = 0; ii < nq; ii++) {
                    q[ii] = hash_to_unit(0x53445041ULL, ii);
                    k[ii] = hash_to_unit(0x53445042ULL, ii);
                    v[ii] = hash_to_unit(0x53445043ULL, ii);
                }
                {
                    char be[32] = "";
                    int gok =
                        h3_dit_w4a8_sdpa_f32(q, k, v, cseq, heads, hd, scale,
                                             o_cpu, error, sizeof(error)) &&
                        (sdpa_f32
                             ? h3_cuda_sdpa_f32(cuda, q, k, v, cseq, heads, hd,
                                                scale, o_gpu, NULL, error,
                                                sizeof(error))
                             : h3_cuda_sdpa_f16_ex(cuda, q, k, v, cseq, heads,
                                                   hd, scale, 0, o_gpu, NULL, be,
                                                   sizeof(be), error,
                                                   sizeof(error)));
                    if (gok) {
                        int is_sage = be[0] && strstr(be, "sage") != NULL;
                        float atol = sdpa_f32 ? 2e-3f : (is_sage ? 2.0f : 5e-2f);
                        float rtol_rmse =
                            sdpa_f32 ? 5e-4f : (is_sage ? 5e-1f : 1e-2f);
                        h3_w4a8_compare_f32(o_cpu, o_gpu, nq, &max_abs, &rmse);
                        ok_chk = max_abs < atol && (float)rmse < rtol_rmse;
                        printf("check_sdpa(%s/%s): seq=%zu heads=%zu dim=%zu "
                               "max_abs=%.3e rmse=%.3e atol=%.3g  %s\n",
                               sdpa_f32 ? "f32" : "f16",
                               be[0] ? be : "default", cseq, heads, hd, max_abs,
                               rmse, atol, ok_chk ? "PASS" : "FAIL");
                        /* A/B: if current is sage2, also report cudnn baseline. */
                        if (is_sage && !sdpa_f32) {
                            char be2[32] = "";
                            float *o_ref = (float *)malloc(nq * sizeof(float));
                            const char *prev = getenv("H3_CUDA_SDPA_BACKEND");
                            char prev_buf[64] = "cudnn";
                            if (prev && prev[0])
                                snprintf(prev_buf, sizeof(prev_buf), "%s",
                                         prev);
                            if (o_ref) {
                                setenv("H3_CUDA_SDPA_BACKEND", "cudnn", 1);
                                if (h3_cuda_sdpa_f16_ex(
                                        cuda, q, k, v, cseq, heads, hd, scale, 0,
                                        o_ref, NULL, be2, sizeof(be2), error,
                                        sizeof(error))) {
                                    float ab_max = 0.0f;
                                    double ab_rmse = 0.0;
                                    h3_w4a8_compare_f32(o_ref, o_gpu, nq,
                                                        &ab_max, &ab_rmse);
                                    printf("check_sdpa_AB(cudnn vs sage2): "
                                           "max_abs=%.3e rmse=%.3e "
                                           "backends=%s/%s\n",
                                           ab_max, ab_rmse,
                                           be2[0] ? be2 : "cudnn", be);
                                }
                                free(o_ref);
                            }
                            setenv("H3_CUDA_SDPA_BACKEND", prev_buf, 1);
                        }
                    } else {
                        printf("check_sdpa: FAILED (%s)\n", error);
                    }
                }
            } else {
                printf("check_sdpa: OOM\n");
            }
            free(q);
            free(k);
            free(v);
            free(o_cpu);
            free(o_gpu);
        }

        t0 = now_ms();
        {
            const int force_per_block =
                getenv("H3_T2AV_FORCE_PER_BLOCK") &&
                getenv("H3_T2AV_FORCE_PER_BLOCK")[0] == '1';
            /* ≥2L device-chain: skip outer prefetch so layers_forward can
             * cudaMalloc the activation workspace before weight residency. */
            if (!force_per_block && dit_layers >= 2) {
                prefetch_ms = 0.0;
                t1 = now_ms();
                printf("prefetch: deferred_to_layers_forward window=%u\n",
                       window_size);
            } else if (!h3_dit_w4a8_prefetch(dit, 0, dit_layers, &prefetch_ms,
                                             error, sizeof(error))) {
                fprintf(stderr, "prefetch failed: %s\n", error);
                goto dit_fail;
            } else {
                t1 = now_ms();
                printf("prefetch: api_ms=%.1f wall_ms=%.1f window=%u\n",
                       prefetch_ms, t1 - t0, window_size);
            }
        }

        {
            unsigned eff_steps = 1;
            unsigned step_i, computed_steps = 0;
            const int force_per_block =
                getenv("H3_T2AV_FORCE_PER_BLOCK") &&
                getenv("H3_T2AV_FORCE_PER_BLOCK")[0] == '1';
            if (e2e_loop) {
                eff_steps = (unsigned)steps;
                if (reuse > 1)
                    eff_steps = (eff_steps + reuse - 1) / reuse;
                if (eff_steps < 1) eff_steps = 1;
            }
            t0 = now_ms();
            for (step_i = 0; step_i < eff_steps; step_i++) {
                /* 1L: per-block path (same H2D once; clearer quiet wall).
                 * ≥2L: device-chained layers_forward (kill inter-layer PCIe).
                 * H3_T2AV_FORCE_PER_BLOCK=1 forces per-block for any N. */
                if (!force_per_block && dit_layers >= 2) {
                    h3_dit_w4a8_layers_stats lst;
                    memset(&lst, 0, sizeof(lst));
                    if (!h3_dit_w4a8_layers_forward_f32(
                            dit, 0, dit_layers, 512, x, run_seq, row_map, cos,
                            sin, y, 0, &lst, error, sizeof(error))) {
                        fprintf(stderr, "layers_forward failed: %s\n", error);
                        goto dit_fail;
                    }
                    /* Swap so next step sees y as input if e2e. */
                    {
                        float *tmp = x;
                        x = y;
                        y = tmp;
                    }
                    if (lst.blocks_ms > 0.0)
                        /* Keep historical field name in probe aggregate as
                         * "blocks wall" for e2e share plots; not SDPA. */
                        sdpa_ms += lst.blocks_ms;
                    snprintf(sdpa_backend, sizeof(sdpa_backend), "%s",
                             "cuda_layers_chain");
                    if (step_i == 0 || step_i + 1 == eff_steps) {
                        printf("  step %u/%u layers_chain: blocks_ms=%.1f "
                               "h2d=%.1f compute=%.1f d2h=%.1f "
                               "prefetch_wait=%.1f total_ms=%.1f "
                               "finite=%d backend=%s\n"
                               "    note: blocks_ms is layer-chain wall after "
                               "warm prefetch; not true sdpa_ms\n",
                               step_i + 1, eff_steps, lst.blocks_ms, lst.h2d_ms,
                               lst.compute_ms, lst.d2h_ms, lst.prefetch_wait_ms,
                               lst.total_ms, lst.all_finite, sdpa_backend);
                        fflush(stdout);
                    }
                } else {
                    for (b = 0; b < dit_layers; b++) {
                        const h3_dit_w4a8_block_aux *aux;
                        h3_dit_w4a8_block_stats bst;
                        float *cur = (b % 2 == 0) ? x : y;
                        float *nxt = (b % 2 == 0) ? y : x;
                        memset(&bst, 0, sizeof(bst));
                        aux = h3_dit_w4a8_cached_block_aux(dit, b, error,
                                                           sizeof(error));
                        if (!aux) {
                            fprintf(stderr, "load aux %u failed: %s\n", b, error);
                            goto dit_fail;
                        }
                        if (!h3_dit_w4a8_time_feat_from_table(aux, 512, time_feat,
                                                              error,
                                                              sizeof(error)) ||
                            !h3_dit_w4a8_adaln_project_f32(aux, time_feat,
                                                           modulation, error,
                                                           sizeof(error))) {
                            fprintf(stderr, "adaln %u failed: %s\n", b, error);
                            goto dit_fail;
                        }
                        if (!h3_dit_w4a8_block_forward_f32(
                                dit, b, aux, cur, run_seq, modulation, row_map, cos,
                                sin, nxt, &bst, error, sizeof(error))) {
                            fprintf(stderr, "block %u failed: %s\n", b, error);
                            goto dit_fail;
                        }
                        sdpa_ms += bst.block_gpu_ms > 0.0 ? bst.block_gpu_ms
                                                          : bst.sdpa_ms;
                        gemm_ms +=
                            bst.qkv_ms + bst.out_ms + bst.fc1_ms + bst.fc2_ms;
                        act_prep_ms += bst.act_prep_ms;
                        adaln_ms += bst.adaln_attn_ms + bst.adaln_mlp_ms;
                        rope_ms += bst.rope_ms;
                        swiglu_ms += bst.swiglu_ms;
                        other_ms += bst.gate_attn_ms + bst.gate_mlp_ms;
                        if (bst.sdpa_backend[0])
                            snprintf(sdpa_backend, sizeof(sdpa_backend), "%s",
                                     bst.sdpa_backend);
                        if (b == 0 || b + 1 == dit_layers || (b + 1) % 10 == 0) {
                            printf("  step %u/%u block %u/%u: sdpa_ms=%.1f "
                                   "gemm_ms=%.1f act_prep_ms=%.1f backend=%s\n",
                                   step_i + 1, eff_steps, b + 1, dit_layers,
                                   sdpa_ms, gemm_ms, act_prep_ms, sdpa_backend);
                            fflush(stdout);
                        }
                    }
                }
                computed_steps++;
            }
            t1 = now_ms();
            dit_ms = t1 - t0;
            printf("e2e_dit: computed_steps=%u (schedule steps=%d reuse=%u) "
                   "layers_per_step=%u\n",
                   computed_steps, steps, reuse, dit_layers);
        }

        if (!h3_dit_w4a8_get_cache_stats(dit, &cst, error, sizeof(error))) {
            fprintf(stderr, "cache stats failed: %s\n", error);
            goto dit_fail;
        }
        printf("dit_forward: seq=%zu layers=%u wall_ms=%.1f sdpa_backend=%s\n",
               run_seq, dit_layers, dit_ms, sdpa_backend);
        printf("  breakdown_ms: sdpa=%.1f gemm=%.1f act_prep=%.1f adaln=%.1f "
               "rope=%.1f swiglu=%.1f gate=%.1f\n",
               sdpa_ms, gemm_ms, act_prep_ms, adaln_ms, rope_ms, swiglu_ms,
               other_ms);
        printf("  cache: policy=%s resident_blocks=%u resident=%s vram_free=%s\n",
               h3_w4a8_cache_policy_name(cst.effective_policy),
               cst.n_resident_blocks,
               format_bytes(cst.bytes_resident, b1, sizeof(b1)),
               format_bytes(cst.vram_free_bytes, b2, sizeof(b2)));
        {
            double per_layer = dit_ms / (double)dit_layers;
            double per_step_full = per_layer * (double)layers;
            double dit_total = per_step_full * (double)steps;
            double scale = (double)video_seq / (double)run_seq;
            /* MLP~linear, SDPA~quadratic — blend using measured share. */
            double sdpa_share = (dit_ms > 0.0) ? (sdpa_ms / dit_ms) : 0.9;
            double scale_eff =
                sdpa_share * (scale * scale) + (1.0 - sdpa_share) * scale;
            printf("extrapolate_same_seq: 1-step@%u-layers≈%.1fs  "
                   "%d-steps≈%.1fs (%.1f min)\n",
                   layers, per_step_full / 1000.0, steps, dit_total / 1000.0,
                   dit_total / 60000.0);
            printf("extrapolate_to_video_seq=%zu: scale_eff≈%.2f "
                   "(sdpa_share=%.2f) → 1-step≈%.1f min  %d-steps≈%.1f min\n",
                   video_seq, scale_eff, sdpa_share,
                   per_step_full * scale_eff / 60000.0, steps,
                   dit_total * scale_eff / 60000.0);
        }

    dit_fail:
        if (dit_bufs_pinned) {
            if (x) cudaFreeHost(x);
            if (y) cudaFreeHost(y);
            if (cos) cudaFreeHost(cos);
            if (sin) cudaFreeHost(sin);
        } else {
            free(x);
            free(y);
            free(cos);
            free(sin);
        }
        free(pt);
        free(ph);
        free(pw);
        free(row_map);
        free(modulation);
        h3_dit_w4a8_block_aux_free(&aux0);
        h3_dit_w4a8_core_free(&core);
        h3_dit_w4a8_free(dit);
        dit = NULL;
        h3_cuda_free(cuda);
        cuda = NULL;
        h3_w4a8_store_free(dit_store);
        dit_store = NULL;
        /* drop GPU residency before VAE */
    } else {
        printf("\n=== phase DiT: skipped ===\n");
    }

    /* ---- VAE decode (CUDA main path; full 480p latent by default) ---- */
    if (!skip_vae && vae_path) {
        float *chw = NULL;
        size_t n_chw;
        int vh = latent_h;
        int vw = latent_w;
        int vt = latent_t > 0 ? latent_t : 1;
        h3_cuda *vae_cuda = NULL;
        /* Stream/chunk temporal if patches huge: keep T but allow --vae-blocks */
        printf("\n=== phase VAE int8 CUDA decode ===\n");
        t0 = now_ms();
        if (!h3_vae_int8_store_open(vae_path, &vae_store, error, sizeof(error))) {
            fprintf(stderr, "vae open failed: %s\n", error);
            goto cleanup;
        }
        vae_cuda = h3_cuda_create(device, error, sizeof(error));
        if (!vae_cuda) {
            fprintf(stderr, "vae cuda create failed: %s\n", error);
            goto cleanup;
        }
        n_chw = (size_t)H3_VAE_INT8_LATENT_CH * (size_t)vt * (size_t)vh *
                (size_t)vw;
        chw = (float *)malloc(n_chw * sizeof(float));
        if (!chw) {
            fprintf(stderr, "VAE OOM\n");
            h3_cuda_free(vae_cuda);
            goto cleanup;
        }
        for (i = 0; i < n_chw; i++) chw[i] = hash_to_unit(0x564145ULL, i) * 0.1f;
        if (!h3_vae_int8_decode_cuda(vae_cuda, &vae_store, chw, vt, vh, vw,
                                     vae_blocks, &vae_frames, &vst, error,
                                     sizeof(error))) {
            fprintf(stderr, "vae decode failed: %s\n", error);
            free(chw);
            h3_cuda_free(vae_cuda);
            goto cleanup;
        }
        t1 = now_ms();
        vae_ms = t1 - t0;
        printf("vae_decode_cuda: latent=%dx%dx%d blocks=%u → frames=%d %dx%d "
               "wall_ms=%.1f (api_total_ms=%.1f blocks_ms=%.1f all_finite=%d)\n",
               vt, vh, vw, vae_blocks, vae_frames.frames, vae_frames.width,
               vae_frames.height, vae_ms, vst.total_ms, vst.blocks_ms,
               vst.all_finite);
        free(chw);
        h3_vae_int8_frames_free(&vae_frames);
        h3_vae_int8_store_free(&vae_store);
        memset(&vae_store, 0, sizeof(vae_store));
        h3_cuda_free(vae_cuda);
    } else {
        printf("\n=== phase VAE: skipped ===\n");
    }

    wall1 = now_ms();
    printf("\n=== summary ===\n");
    printf("phase_ms: te=%.1f dit=%.1f(prefetch_api=%.1f) vae=%.1f  "
           "wall_total=%.1f (%.3fs)\n",
           te_ms, dit_ms, prefetch_ms, vae_ms, wall1 - wall0,
           (wall1 - wall0) / 1000.0);
    printf("knobs: reuse=%u token_reduce=%.3g steps=%d dit_layers=%u "
           "layers_cfg=%u vae_blocks=%u sdpa=%s\n",
           reuse, token_reduce, steps, dit_layers, layers, vae_blocks,
           cpu_sdpa ? "cpu" : (sdpa_f32 ? "f32" : "f16"));
    {
        double per_block = (dit_layers > 0) ? (dit_ms / (double)dit_layers) : 0.0;
        if (e2e_loop && steps > 0 && reuse > 0) {
            unsigned eff = (unsigned)((steps + (int)reuse - 1) / (int)reuse);
            if (eff < 1) eff = 1;
            per_block = dit_ms / ((double)eff * (double)dit_layers);
        }
        double full_20x50 =
            per_block * 50.0 * 20.0 *
            ((double)video_seq / (double)(run_seq ? run_seq : 1));
        /* SDPA-quadratic blend for full seq if reduced */
        if (run_seq != video_seq && run_seq > 0) {
            double scale = (double)video_seq / (double)run_seq;
            double sdpa_share = (dit_ms > 0.0) ? (sdpa_ms / dit_ms) : 0.9;
            double scale_eff =
                sdpa_share * (scale * scale) + (1.0 - sdpa_share) * scale;
            full_20x50 = per_block * 50.0 * 20.0 * scale_eff;
        }
        printf("extrapolate_full_20x50_video_seq: ≈%.1fs (%.2f min) "
               "(from measured per-block)\n",
               full_20x50 / 1000.0, full_20x50 / 60000.0);
        printf("gate_20s: wall=%.3fs → %s\n", (wall1 - wall0) / 1000.0,
               ((wall1 - wall0) <= 20000.0) ? "PASS" : "FAIL");
    }
    printf("produced: latent/video decode=%s  full T2AV mp4=NO "
           "(CUDA bypass)\n",
           (!skip_vae && vae_path) ? "yes-cuda" : "no");
    printf("audio: G2 BLOCKED — h3_audio_vae is Metal/shader-only; W4A8 DiT "
           "has no audio token path on CUDA bypass\n");
    rc = 0;

cleanup:
    h3_dit_w4a8_core_free(&core);
    h3_dit_w4a8_free(dit);
    h3_cuda_free(cuda);
    h3_w4a8_store_free(dit_store);
    h3_nvfp4_store_free(&te_store);
    h3_vae_int8_store_free(&vae_store);
    h3_vae_int8_frames_free(&vae_frames);
    return rc;
}
