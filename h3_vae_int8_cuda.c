#include "h3_vae_int8_cuda.h"
#include "h3_vae_int8.h"
#include "h3_gpu_cuda_vae.h"
#include "h3_w4a8_dequant.h"
#include <cuda_runtime_api.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static void fail(char *error, size_t error_size, const char *msg) {
    if (!error || !error_size) return;
    snprintf(error, error_size, "%s", msg ? msg : "h3_vae_int8_cuda error");
}

static double elapsed_ms(clock_t t0, clock_t t1) {
    return 1000.0 * (double)(t1 - t0) / (double)CLOCKS_PER_SEC;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static void vae_fill_attn_backend(h3_vae_int8_decode_stats *stats) {
    const char *be;
    if (!stats) return;
    be = h3_cuda_vae_sdpa_last_backend();
    snprintf(stats->attn_backend, sizeof(stats->attn_backend), "%s",
             be ? be : "");
}

typedef struct {
    float *w_qkv, *w_out, *w_w1, *w_w2;
    float *b_qkv, *b_out, *b_w1, *b_w2;
    float *norm1, *norm2, *scale1, *scale2;
    size_t n_qkv, k_qkv, n_out, k_out, n_w1, k_w1, n_w2, k_w2;
    /* Optional device copies (H3_VAE_CUDA_W_RESIDENT=1). */
    float *d_w_qkv, *d_w_out, *d_w_w1, *d_w_w2;
    float *d_b_qkv, *d_b_out, *d_b_w1, *d_b_w2;
    float *d_norm1, *d_norm2, *d_scale1, *d_scale2;
    /* FP16 TC weights (optional; preferred when present). */
    void *d_w_qkv_h, *d_w_out_h, *d_w_w1_h, *d_w_w2_h;
} h3_vae_block_w;

struct h3_vae_int8_cuda_weights {
    unsigned n_blocks;
    h3_vae_block_w *blocks;
    int device_resident;
    int cache_pinned;
    unsigned cache_refs;
    /* Native F16/BF16/F32 checkpoint (not INT8+scale). Quality decode keeps
     * F32 GEMM; skip F16-TC recast that INT8 speed path uses. */
    int native_float_w;
    char cache_path[512];
    /* Head LN+proj (decoder.norm_out / proj_out), uploaded once. */
    float *d_norm_out_w, *d_norm_out_b;
    float *d_proj_out_w, *d_proj_out_b;
    float *d_embed_w, *d_embed_b, *d_registers;
    /* Host heads reused across metal temporal windows × spatial tiles. */
    float *h_post_w, *h_post_b, *h_embed_w, *h_embed_b, *h_registers;
    float *h_norm_w, *h_norm_b, *h_proj_w, *h_proj_b;
};

static int gemm_f16_via(h3_cuda *cuda, const float *d_x, const void *d_w_h,
                        size_t m, size_t n, size_t k, float *d_y,
                        void *d_scratch_h, char *error, size_t error_size);
static int gemm_f16_act(h3_cuda *cuda, const void *d_x_h, const void *d_w_h,
                        size_t m, size_t n, size_t k, void *d_y_h, char *error,
                        size_t error_size);
static int run_block_cuda_w_dev_f16(h3_cuda *cuda, const h3_vae_block_w *bw,
                                    float *d_seq, float *d_ws, size_t seq,
                                    const float *d_rope_cos,
                                    const float *d_rope_sin,
                                    const float *d_qk_ones, int convrot_gs,
                                    char *error, size_t error_size);
static int cast_w_to_f16(h3_cuda *cuda, const float *d_f32, size_t n,
                         void **d_h, char *error, size_t error_size);
static int ensure_vae_head_dev(h3_vae_int8_cuda_weights *w, const float *norm_w,
                               const float *norm_b, const float *proj_w,
                               const float *proj_b);
static int ensure_vae_embed_dev(h3_vae_int8_cuda_weights *w, const float *embed_w,
                                const float *embed_b, const float *registers);

static int ensure_block_f16_weights(h3_cuda *cuda, h3_vae_block_w *bw,
                                    char *error, size_t error_size) {
    if (!bw || !bw->d_w_qkv) return 0;
    if (bw->d_w_qkv_h && bw->d_w_out_h && bw->d_w_w1_h && bw->d_w_w2_h)
        return 1;
    if (!cast_w_to_f16(cuda, bw->d_w_qkv, bw->n_qkv * bw->k_qkv, &bw->d_w_qkv_h,
                       error, error_size) ||
        !cast_w_to_f16(cuda, bw->d_w_out, bw->n_out * bw->k_out, &bw->d_w_out_h,
                       error, error_size) ||
        !cast_w_to_f16(cuda, bw->d_w_w1, bw->n_w1 * bw->k_w1, &bw->d_w_w1_h,
                       error, error_size) ||
        !cast_w_to_f16(cuda, bw->d_w_w2, bw->n_w2 * bw->k_w2, &bw->d_w_w2_h,
                       error, error_size)) {
        static int logged_f16_fail;
        if (!logged_f16_fail) {
            fprintf(stderr,
                    "vae_f16_w: cast failed (F32 GEMM fallback; ~3× slower). "
                    "%s\n",
                    error && error[0] ? error : "cudaMalloc?");
            logged_f16_fail = 1;
        }
        if (bw->d_w_qkv_h) {
            cudaFree(bw->d_w_qkv_h);
            bw->d_w_qkv_h = NULL;
        }
        if (bw->d_w_out_h) {
            cudaFree(bw->d_w_out_h);
            bw->d_w_out_h = NULL;
        }
        if (bw->d_w_w1_h) {
            cudaFree(bw->d_w_w1_h);
            bw->d_w_w1_h = NULL;
        }
        if (bw->d_w_w2_h) {
            cudaFree(bw->d_w_w2_h);
            bw->d_w_w2_h = NULL;
        }
        return 0;
    }
    return 1;
}

static int env_flag_is_one(const char *name) {
    const char *e = getenv(name);
    return e && e[0] == '1' && e[1] == '\0';
}

/* Default-ON variant for registered plugin switches (default "1"). */
static int env_flag_default_on(const char *name) {
    const char *e = getenv(name);
    return !(e && e[0] == '0' && e[1] == '\0');
}

/* P0: VAE GPU unpatch D2H — pinned + async/stream toggles.
 * H3_VAE_UNPATCH_PINNED (default 1): alloc fr->rgb via cudaHostAlloc so the
 *   D2H tail goes through DMA (pageable memcpy pays an extra host staging
 *   copy + sync). Bit-identical, just faster.
 * H3_VAE_UNPATCH_ASYNC (default 1): move the D2H onto a side stream with a
 *   per-slot done-event chain + cross-tile pinned pool, overlapping the DMA
 *   with the next tile's GPU decode. Re-enabled as default on 2026-10-06 after
 *   the fr->rgb ownership fix (malloc-fallback slots were wrongly freed via
 *   cudaFreeHost, latching cudaErrorInvalidValue → cold NaN + clip2 crash);
 *   Release(-O3) 4-run re-verify on 238 is bit-identical async=0/1 (cold
 *   0x2d5c5202…, hot 0x923b0db3…) with unpatch_d2h 3200→26 ms.
 *   unpatch_d2h 3004ms -> 26ms (-99%) when explicitly enabled and healthy.
 *   Sync semantics only, never numerics.
 */
static int vae_unpatch_pinned(void) {
    return env_flag_default_on("H3_VAE_UNPATCH_PINNED");
}
static int vae_unpatch_async(void) {
    return env_flag_default_on("H3_VAE_UNPATCH_ASYNC");
}
/* Registered into the host frames_free so pinned rgb uses cudaFreeHost. */
static void vae_rgb_pinned_free(void *p) { h3_cuda_host_free(p, 1); }

/* Grow-only scratch so 18 spatial × 7 temporal windows do not cudaMalloc
 * ~0.5GiB workspace 126 times (and F16→F32 convert proj_out 25MiB each). */
typedef struct {
    size_t cap_d_slice, cap_d_ws, cap_d_canvas, cap_d_reg, cap_d_qk;
    size_t cap_d_rope, cap_d_rgb, cap_d_pat, cap_h_slice, cap_h_rope,
        cap_h_patches, cap_h_seq, cap_h_proj;
    float *d_slice, *d_ws, *d_canvas, *d_registers;
    /* P2 T4-b: second workspace for 2-stream block split (768P only). */
    float *d_ws2;
    size_t cap_d_ws2;
    float *d_pat;
    float *d_rope_cos, *d_rope_sin, *d_qk_ones, *d_rgb;
    float *h_slice, *h_rope_cos, *h_rope_sin;
    float *h_patches, *h_seq, *h_projected;
    int reuse_hits;
} vae_scratch_pool;
static vae_scratch_pool g_vae_scratch;

/* P0 刀1/刀2 state (defined below, reset by vae_scratch_drop). */
/* At most H3_VAE_MAX_TILE_BATCH(8) tiles are in flight per decode call; the
 * pool only needs to cover one batch's worth of concurrent async D2H. */
#define VAE_D2H_SLOTS 8
typedef struct {
    void *base;
    size_t cap;
    int pinned; /* 1 = cudaHostAlloc, 0 = malloc */
    int in_use[VAE_D2H_SLOTS];
} vae_rgb_arena;
static vae_rgb_arena g_vae_rgb_arena;

typedef struct {
    float *rgb;
    void *event;
} vae_d2h_slot;
static vae_d2h_slot g_vae_d2h_slots[VAE_D2H_SLOTS];
static float *g_vae_d_rgb_lanes[VAE_D2H_SLOTS];
static size_t g_vae_d_rgb_lane_cap = 0;
static void *g_vae_d2h_side = NULL;
static void *g_vae_d2h_side2 = NULL;
static h3_cuda *g_vae_d2h_cuda = NULL;
static int g_vae_d2h_armed = 0;

static int vae_d2h_lane_of(const float *d_rgb) {
    int i;
    for (i = 0; i < VAE_D2H_SLOTS; i++)
        if (g_vae_d_rgb_lanes[i] == d_rgb) return i;
    return -1;
}

static int vae_d2h_wait(const float *d_rgb) {
    int lane;
    if (!g_vae_d2h_armed) return 1;
    lane = vae_d2h_lane_of(d_rgb);
    if (lane < 0) return 1;
    if (g_vae_d2h_slots[lane].event &&
        cudaEventSynchronize((cudaEvent_t)g_vae_d2h_slots[lane].event) !=
            cudaSuccess)
        return 0;
    g_vae_d2h_slots[lane].rgb = NULL;
    g_vae_d2h_slots[lane].event = NULL;
    g_vae_d2h_armed = 0;
    return 1;
}

static void vae_d2h_reset(void) {
    int i;
    for (i = 0; i < VAE_D2H_SLOTS; i++) {
        if (g_vae_d2h_slots[i].event) {
            cudaEventSynchronize((cudaEvent_t)g_vae_d2h_slots[i].event);
            cudaEventDestroy((cudaEvent_t)g_vae_d2h_slots[i].event);
        }
        g_vae_d2h_slots[i].rgb = NULL;
        g_vae_d2h_slots[i].event = NULL;
        if (g_vae_d_rgb_lanes[i]) cudaFree(g_vae_d_rgb_lanes[i]);
        g_vae_d_rgb_lanes[i] = NULL;
    }
    g_vae_d_rgb_lane_cap = 0;
    g_vae_d2h_armed = 0;
    g_vae_d2h_cuda = NULL;
}

static void vae_rgb_arena_reset(void) {
    if (g_vae_rgb_arena.base) {
        if (g_vae_rgb_arena.pinned)
            h3_cuda_host_free(g_vae_rgb_arena.base, 1);
        else
            free(g_vae_rgb_arena.base);
    }
    memset(&g_vae_rgb_arena, 0, sizeof(g_vae_rgb_arena));
}

static float *vae_rgb_arena_acquire(size_t bytes) {
    vae_rgb_arena *a = &g_vae_rgb_arena;
    int i;
    if (a->base && a->cap >= bytes) {
        for (i = 0; i < VAE_D2H_SLOTS; i++) {
            if (!a->in_use[i]) {
                a->in_use[i] = 1;
                return (float *)((char *)a->base + (size_t)i * a->cap);
            }
        }
        return NULL; /* slots exhausted */
    }
    /* Need to (re)allocate the arena. Only safe when NOTHING is in flight:
     * a pending async memcpy into an old slot would write freed memory. */
    if (a->base) {
        for (i = 0; i < VAE_D2H_SLOTS; i++)
            if (a->in_use[i]) return NULL; /* caller falls back to own alloc */
        if (a->pinned)
            h3_cuda_host_free(a->base, 1);
        else
            free(a->base);
        a->base = NULL;
        a->cap = 0;
    }
    a->base = h3_cuda_host_alloc(bytes * (size_t)VAE_D2H_SLOTS, &a->pinned);
    if (!a->base) {
        a->base = malloc(bytes * (size_t)VAE_D2H_SLOTS);
        a->pinned = 0;
    }
    if (!a->base) return NULL;
    a->cap = bytes;
    a->in_use[0] = 1;
    return (float *)a->base;
}

static void vae_rgb_arena_release(const float *p) {
    vae_rgb_arena *a = &g_vae_rgb_arena;
    char *base = (char *)a->base;
    size_t i;
    if (!base || !a->cap) return;
    if ((const char *)p < base ||
        (size_t)((const char *)p - base) >= (size_t)VAE_D2H_SLOTS * a->cap)
        return;
    i = (size_t)((const char *)p - base) / a->cap;
    if ((size_t)i < VAE_D2H_SLOTS) a->in_use[i] = 0;
}

/* Per-rgb done-event lookup; clears the entry so recycled pointers in a
 * later call can never match a stale event. */
static void *vae_d2h_event_take(const float *rgb) {
    int i;
    void *ev = NULL;
    for (i = 0; i < VAE_D2H_SLOTS; i++) {
        if (g_vae_d2h_slots[i].rgb == rgb && g_vae_d2h_slots[i].event) {
            ev = g_vae_d2h_slots[i].event;
            g_vae_d2h_slots[i].rgb = NULL;
            g_vae_d2h_slots[i].event = NULL;
            break;
        }
    }
    return ev;
}

static int vae_d2h_event_put(float *rgb, void *event) {
    int i;
    for (i = 0; i < VAE_D2H_SLOTS; i++) {
        if (!g_vae_d2h_slots[i].event) {
            g_vae_d2h_slots[i].rgb = rgb;
            g_vae_d2h_slots[i].event = event;
            /* The next decode call that reuses the shared d_rgb scratch must
             * drain this D2H first; only relevant for the non-lane path. */
            if (!g_vae_d2h_armed && g_vae_scratch.d_rgb &&
                vae_d2h_lane_of(g_vae_scratch.d_rgb) < 0)
                g_vae_d2h_armed = 1;
            return 1;
        }
    }
    return 0;
}

/* Consume a tile's fr->rgb: wait its D2H done-event (if any), copy out to a
 * fresh malloc buffer, and release the pinned pool slot. Bit-identical: the
 * copy source bytes equal what a synchronous cudaMemcpy would have produced. */
static int vae_d2h_sync_and_copy(h3_vae_int8_frames *fr, float **out,
                                 size_t rgb_bytes) {
    void *ev;
    float *dst;
    if (!fr || !out || !fr->rgb) return 0;
    ev = vae_d2h_event_take(fr->rgb);
    if (ev) {
        if (cudaEventSynchronize((cudaEvent_t)ev) != cudaSuccess)
            return 0;
        cudaEventDestroy((cudaEvent_t)ev);
    }
    dst = (float *)malloc(rgb_bytes);
    if (!dst) return 0;
    memcpy(dst, fr->rgb, rgb_bytes);
    /* Free by exact ownership. rgb_pinned==2 = pinned-pool slot (release, not
     * free); ==1 = true cudaHostAlloc (cudaFreeHost); ==0 = plain malloc
     * (free). The pre-fix else-branch called cudaFreeHost on the ==0 malloc
     * fallback, leaving cudaErrorInvalidValue latched in the CUDA error state
     * (vae_rgb_pinned_free never checks it), which surfaced at the next
     * non-quiet cudaGetLastError as a spurious "bias_add_dev kernel" failure
     * on the hot clip-2 DiT step 0. */
    if (fr->rgb_pinned == 2)
        vae_rgb_arena_release(fr->rgb);
    else if (fr->rgb_pinned == 1)
        vae_rgb_pinned_free(fr->rgb);
    else
        free(fr->rgb);
    fr->rgb = NULL;
    fr->rgb_pinned = 0;
    *out = dst;
    return 1;
}

static int env_int_default(const char *name, int def) {
    const char *e = getenv(name);
    char *end = NULL;
    long v;
    if (!e || !e[0]) return def;
    v = strtol(e, &end, 10);
    if (end == e) return def;
    if (v < 0) return def;
    if (v > 1024) return 1024;
    return (int)v;
}

/* When set, decode_cuda_ex treats latent_t as one Metal/Comfy window:
 * full-window attention + 22-frame unpack with FRAME_OFFSET skip. */
static int g_vae_metal_chunk_unpack = 0;

/* Comfy tiled_decode batches up to 4 spatial tiles as independent sequences
 * (GEMM rows concat, attention per tile). Node 74 `_decode_tile_row`.
 * Raised to 8: a 1376px row is 6 tiles and 8x ~265MiB tile scratch fits in
 * the ~20GiB free on hot last-clip (fit gate still applies; default stays 4,
 * H3_VAE_TILE_BATCH=8 opts in). */
#define H3_VAE_MAX_TILE_BATCH 8
static int g_vae_batch_n = 1;
static const float *g_vae_batch_chw[H3_VAE_MAX_TILE_BATCH];
static h3_vae_int8_frames g_vae_batch_frames[H3_VAE_MAX_TILE_BATCH];
static size_t g_vae_attn_seq = 0;

/* Spatial tiling used to call decode_cuda per tile, re-loading / re-dequanting
 * all VAE blocks from safetensors each time (~6s × N tiles → ~90s on 480p).
 * Borrow a preloaded pack so tiles only pay compute. */
static h3_vae_int8_cuda_weights *g_vae_borrowed_weights = NULL;
static h3_vae_int8_cuda_weights *g_vae_cached_weights = NULL;
static const void *g_vae_hostreg_map = NULL;
static size_t g_vae_hostreg_len = 0;

/* Spatial tiles must share the full-canvas RoPE map. Tile-local [-1,1]
 * makes every 304px block a separate picture (visible panels). 0 full = local. */
static int g_vae_rope_origin_h = 0;
static int g_vae_rope_origin_w = 0;
static int g_vae_rope_full_h = 0;
static int g_vae_rope_full_w = 0;

static cudaGraphExec_t g_vae_layer_graph;
static size_t g_vae_layer_graph_seq;
static unsigned g_vae_layer_graph_n;
static int g_vae_layer_graph_ok;
static int g_vae_layer_graph_fail;
static int g_vae_layer_graph_warm;

static void vae_layer_graph_drop(void) {
    if (g_vae_layer_graph)
        cudaGraphExecDestroy(g_vae_layer_graph);
    g_vae_layer_graph = NULL;
    g_vae_layer_graph_ok = 0;
    g_vae_layer_graph_seq = 0;
    g_vae_layer_graph_n = 0;
    g_vae_layer_graph_fail = 0;
    g_vae_layer_graph_warm = 0;
}

static int vae_host_grow(void **p, size_t *cap, size_t need) {
    if (p && *p && cap && *cap >= need) return 1;
    free(p ? *p : NULL);
    if (p) *p = NULL;
    if (cap) *cap = 0;
    if (!p || need == 0) return 1;
    *p = malloc(need);
    if (!*p) return 0;
    *cap = need;
    return 1;
}

static int vae_cuda_grow(void **p, size_t *cap, size_t need) {
    if (p && *p && cap && *cap >= need) return 1;
    if (p && *p) cudaFree(*p);
    if (p) *p = NULL;
    if (cap) *cap = 0;
    if (!p || need == 0) return 1;
    if (cudaMalloc(p, need) != cudaSuccess) {
        *p = NULL;
        return 0;
    }
    *cap = need;
    return 1;
}

static void vae_scratch_drop(void) {
    vae_scratch_pool *s = &g_vae_scratch;
    if (s->d_slice) cudaFree(s->d_slice);
    if (s->d_ws) cudaFree(s->d_ws);
    if (s->d_ws2) cudaFree(s->d_ws2);
    if (s->d_canvas) cudaFree(s->d_canvas);
    if (s->d_registers) cudaFree(s->d_registers);
    if (s->d_rope_cos) cudaFree(s->d_rope_cos);
    if (s->d_rope_sin) cudaFree(s->d_rope_sin);
    if (s->d_qk_ones) cudaFree(s->d_qk_ones);
    if (s->d_rgb) cudaFree(s->d_rgb);
    if (s->d_pat) cudaFree(s->d_pat);
    vae_d2h_reset();
    vae_rgb_arena_reset();
    vae_layer_graph_drop();
    free(s->h_slice);
    free(s->h_rope_cos);
    free(s->h_rope_sin);
    free(s->h_patches);
    free(s->h_seq);
    free(s->h_projected);
    memset(s, 0, sizeof(*s));
}

static int vae_weight_cache_enabled(void) {
    return h3_vae_int8_weight_cache_wanted();
}

static void vae_weight_cache_drop(void);
static void vae_weights_device_drop(h3_vae_int8_cuda_weights *w);
static int vae_weights_device_restore(h3_vae_int8_cuda_weights *w);

static float *vae_extract_time_chw(const float *chw, int latent_t, int latent_h,
                                   int latent_w, int t0, int t_len,
                                   char *error, size_t error_size) {
    float *out;
    size_t c, t, h, w, n;
    if (!chw || t_len < 1 || t0 < 0 || t0 + t_len > latent_t) {
        fail(error, error_size, "invalid latent time slice");
        return NULL;
    }
    n = (size_t)H3_VAE_INT8_LATENT_CH * (size_t)t_len * (size_t)latent_h *
        (size_t)latent_w;
    out = (float *)malloc(n * sizeof(float));
    if (!out) {
        fail(error, error_size, "OOM latent time slice");
        return NULL;
    }
    for (c = 0; c < (size_t)H3_VAE_INT8_LATENT_CH; c++)
        for (t = 0; t < (size_t)t_len; t++)
            for (h = 0; h < (size_t)latent_h; h++)
                for (w = 0; w < (size_t)latent_w; w++) {
                    size_t src =
                        ((c * (size_t)latent_t + (size_t)t0 + t) *
                             (size_t)latent_h +
                         h) *
                            (size_t)latent_w +
                        w;
                    size_t dst =
                        ((c * (size_t)t_len + t) * (size_t)latent_h + h) *
                            (size_t)latent_w +
                        w;
                    out[dst] = chw[src];
                }
    return out;
}

static int env_is_one(const char *name) {
    const char *e = getenv(name);
    return e && e[0] == '1' && e[1] == '\0';
}

/* P2 T4-b: VAE blocks 2-stream split. EXPERIMENTAL, default 0. */
static int vae_blocks_2stream_wanted(void) {
    return env_is_one("H3_VAE_BLOCKS_2STREAM");
}

/* Cap Comfy-style tile batch by free VRAM. Sage3-resident 5090 has ~6GiB
 * free; batch=4 scratch 1.66GiB then cudaMalloc y OOM. Hot last-clip
 * (DiT/sage3 gone) has ~20GiB and can take 4. */
static int vae_tile_batch_fit(int want) {
    size_t free_b = 0, total_b = 0;
    size_t slack = (size_t)env_int_default("H3_VAE_BATCH_SLACK_GIB", 3) << 30;
    /* Per-tile VRAM ≈265MiB at 320px (slice 8KiB seq ×2048 + ffn 8KiB×8192
     * f32 + f16 act copies + rgb + graphs). 512MiB was a 2x-margin guess
     * that silently capped batch=8 to 6 even with ~20GiB free. */
    size_t per = (size_t)env_int_default("H3_VAE_BATCH_PER_MIB", 384) << 20;
    size_t usable;
    int fit;
    if (want < 1) want = 1;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return want;
    usable = (free_b > slack) ? (free_b - slack) : 0;
    fit = (int)(usable / per);
    if (fit < 1) fit = 1;
    if (fit > want) fit = want;
    fprintf(stderr, "vae_kitchen: free_mib=%.0f cap_batch=%d want=%d\n",
            (double)free_b / (1024.0 * 1024.0), fit, want);
    return fit;
}

static int vae_error_oom(const char *error) {
    return error && (strstr(error, "out of memory") || strstr(error, "OOM"));
}

static float *load_f32(const h3_vae_int8_store *store, const char *name,
                       size_t want, char *error, size_t error_size);

static int load_bias(const h3_vae_int8_store *store, const h3_vae_int8_linear *lin,
                     float **out, char *error, size_t error_size) {
    char bname[288];
    *out = NULL;
    if (!lin->bias) return 1;
    snprintf(bname, sizeof(bname), "%s.bias", lin->base_name);
    *out = load_f32(store, bname, lin->n, error, error_size);
    return *out != NULL;
}


static int linear_cuda_w(h3_cuda *cuda, const float *w, const float *d_w,
                         const float *bias, size_t n, size_t k, const float *x,
                         size_t m, float *y, int convrot_gs, char *error,
                         size_t error_size) {
    size_t i, o;
    int ok;
    float *x_rot = NULL;
    const float *x_use = x;
    if ((!w && !d_w) || !x || !y) {
        fail(error, error_size, "invalid linear_cuda_w");
        return 0;
    }
    if (convrot_gs > 0) {
        x_rot = (float *)malloc(m * k * sizeof(float));
        if (!x_rot) {
            fail(error, error_size, "OOM act ConvRot");
            return 0;
        }
        if (!h3_w4a8_convrot_act_f32(x, m, k, convrot_gs, x_rot, error,
                                     error_size)) {
            free(x_rot);
            return 0;
        }
        x_use = x_rot;
    }
    if (d_w)
        ok = h3_cuda_gemm_f32_dw(cuda, x_use, d_w, m, n, k, y, NULL, error,
                                 error_size);
    else
        ok = h3_cuda_gemm_f32(cuda, x_use, w, m, n, k, y, NULL, error,
                              error_size);
    free(x_rot);
    if (!ok) return 0;
    if (bias) {
        for (i = 0; i < m; i++)
            for (o = 0; o < n; o++) y[i * n + o] += bias[o];
    }
    return 1;
}

static int linear_cuda(h3_cuda *cuda, const h3_vae_int8_store *store,
                       const h3_vae_int8_linear *lin, const float *x, size_t m,
                       float *y, char *error, size_t error_size) {
    float *w = NULL;
    float *bias = NULL;
    int ok = 0;
    int gs;
    w = h3_vae_int8_dequant_f32(store, lin, error, error_size);
    if (!w) return 0;
    if (!load_bias(store, lin, &bias, error, error_size)) goto done;
    gs = h3_vae_int8_convrot_groupsize(store, lin->k);
    ok = linear_cuda_w(cuda, w, NULL, bias, lin->n, lin->k, x, m, y, gs, error,
                       error_size);
done:
    free(w);
    free(bias);
    return ok;
}

static void rms_norm(const float *x, const float *w, size_t rows, size_t width,
                     float eps, float *out) {
    size_t r, c;
    for (r = 0; r < rows; r++) {
        const float *xr = x + r * width;
        float *yr = out + r * width;
        float sum = 0.0f, inv;
        for (c = 0; c < width; c++) sum += xr[c] * xr[c];
        inv = 1.0f / sqrtf(sum / (float)width + eps);
        for (c = 0; c < width; c++) yr[c] = xr[c] * inv * w[c];
    }
}

static void silu_mul(const float *fused, size_t rows, size_t width, float *out) {
    size_t r, c;
    for (r = 0; r < rows; r++) {
        const float *gate = fused + r * width * 2;
        const float *up = gate + width;
        float *yr = out + r * width;
        for (c = 0; c < width; c++) {
            float g = gate[c];
            yr[c] = (g / (1.0f + expf(-g))) * up[c];
        }
    }
}

/* H3_VAE_QKV_LAYOUT=thirds|interleaved|norope
 * thirds (default): Comfy cat(Q,K,V) then pack→RoPE
 * interleaved: treat linear out as Metal video interleaved, RoPE direct
 * norope: old thirds split SDPA without RoPE (ablation)
 */
static int vae_qkv_layout_mode(void) {
    const char *e = getenv("H3_VAE_QKV_LAYOUT");
    if (!e || !*e) return 1; /* interleaved = Metal video default */
    if (e[0] == 'i' || e[0] == 'I') return 1;
    if (e[0] == 'n' || e[0] == 'N') return 2;
    return 0;
}

/* Pack cat(Q,K,V) thirds → per-head interleaved Q|K|V for CUDA rope kernel. */
static void pack_thirds_to_interleaved(const float *thirds, float *inter,
                                       size_t seq, size_t heads,
                                       size_t head_dim) {
    size_t inner = heads * head_dim;
    size_t s, h, d;
    for (s = 0; s < seq; s++) {
        const float *src = thirds + s * inner * 3;
        float *dst = inter + s * inner * 3;
        for (h = 0; h < heads; h++) {
            for (d = 0; d < head_dim; d++) {
                dst[h * head_dim * 3 + d] = src[h * head_dim + d];
                dst[h * head_dim * 3 + head_dim + d] =
                    src[inner + h * head_dim + d];
                dst[h * head_dim * 3 + 2 * head_dim + d] =
                    src[2 * inner + h * head_dim + d];
            }
        }
    }
}

/* Kijai/Comfy to_qkv is thirds; pack→interleaved then RoPE + SDPA. */
static int sdpa_cuda_rope(h3_cuda *cuda, const float *qkv,
                          const float *rope_cos, const float *rope_sin,
                          size_t seq, size_t heads, size_t head_dim,
                          float *out, char *error, size_t error_size) {
    size_t inner = heads * head_dim;
    size_t n = seq * inner;
    float *q = NULL, *k = NULL, *v = NULL, *inter = NULL;
    float ones[H3_VAE_INT8_HEAD_DIM];
    float scale = 1.0f / sqrtf((float)head_dim);
    size_t d;
    int ok = 0;
    for (d = 0; d < H3_VAE_INT8_HEAD_DIM; d++) ones[d] = 1.0f;
    q = (float *)malloc(n * sizeof(float));
    k = (float *)malloc(n * sizeof(float));
    v = (float *)malloc(n * sizeof(float));
    inter = (float *)malloc(seq * inner * 3 * sizeof(float));
    if (!q || !k || !v || !inter) {
        fail(error, error_size, "OOM vae sdpa split");
        goto done;
    }
    {
        int mode = vae_qkv_layout_mode();
        const float *rope_in = qkv;
        if (mode == 2) {
            /* norope: thirds split */
            size_t s, h, d;
            size_t inner2 = heads * head_dim;
            for (s = 0; s < seq; s++) {
                const float *row = qkv + s * inner2 * 3;
                for (h = 0; h < heads; h++)
                    for (d = 0; d < head_dim; d++) {
                        size_t dst = (s * heads + h) * head_dim + d;
                        q[dst] = row[h * head_dim + d];
                        k[dst] = row[inner2 + h * head_dim + d];
                        v[dst] = row[2 * inner2 + h * head_dim + d];
                    }
            }
            ok = h3_cuda_sdpa_f16(cuda, q, k, v, seq, heads, head_dim, scale,
                                  out, NULL, error, error_size);
            goto done;
        }
        if (mode == 0) {
            pack_thirds_to_interleaved(qkv, inter, seq, heads, head_dim);
            rope_in = inter;
        }
        /* Host fallback: Metal half-pair RoPE on CPU, then SDPA. */
        {
            size_t s, h, dd;
            size_t rope_half = H3_VAE_INT8_ROPE_HALF;
            float eps = 1e-5f;
            for (s = 0; s < seq; s++) {
                const float *row_base = rope_in + s * inner * 3;
                const float *c_row = rope_cos + s * rope_half;
                const float *s_row = rope_sin + s * rope_half;
                for (h = 0; h < heads; h++) {
                    const float *q_base = row_base + h * head_dim * 3;
                    const float *k_base = q_base + head_dim;
                    const float *v_base = k_base + head_dim;
                    float *q_out = q + (s * heads + h) * head_dim;
                    float *k_out = k + (s * heads + h) * head_dim;
                    float *v_out = v + (s * heads + h) * head_dim;
                    float q_sum = 0.0f, k_sum = 0.0f, qi, ki;
                    for (dd = 0; dd < head_dim; dd++) {
                        q_sum += q_base[dd] * q_base[dd];
                        k_sum += k_base[dd] * k_base[dd];
                        v_out[dd] = v_base[dd];
                    }
                    qi = 1.0f / sqrtf(q_sum / (float)head_dim + eps);
                    ki = 1.0f / sqrtf(k_sum / (float)head_dim + eps);
                    for (dd = 0; dd < head_dim; dd++) {
                        float q0 = q_base[dd] * qi;
                        float k0 = k_base[dd] * ki;
                        if (dd < rope_half) {
                            size_t pair = dd + rope_half;
                            float q1 = q_base[pair] * qi;
                            float k1 = k_base[pair] * ki;
                            float c = c_row[dd], s_ = s_row[dd];
                            q0 = q0 * c - q1 * s_;
                            k0 = k0 * c - k1 * s_;
                        } else if (dd < rope_half * 2) {
                            size_t pair = dd - rope_half;
                            float q1 = q_base[pair] * qi;
                            float k1 = k_base[pair] * ki;
                            float c = c_row[pair], s_ = s_row[pair];
                            q0 = q0 * c + q1 * s_;
                            k0 = k0 * c + k1 * s_;
                        }
                        q_out[dd] = q0;
                        k_out[dd] = k0;
                    }
                }
            }
        }
    }
    ok = h3_cuda_sdpa_f16(cuda, q, k, v, seq, heads, head_dim, scale, out,
                          NULL, error, error_size);
done:
    free(q);
    free(k);
    free(v);
    free(inter);
    return ok;
}

static float *load_f32(const h3_vae_int8_store *store, const char *name,
                       size_t want, char *error, size_t error_size) {
    /* Reuse host decode path: dequant not needed for BF16/F32 heads.
     * Call public linear find? Use h3_vae store via temporary open pattern —
     * simplest: duplicate via h3_vae_int8_decode's dependency by calling
     * h3_vae_int8_linear only for I8; for named F32 use st API. */
    const h3_st_tensor *t = h3_st_find(&store->header, name);
    float *buf;
    size_t n, i;
    uint16_t *u16 = NULL;
    if (!t) {
        fail(error, error_size, "missing tensor");
        return NULL;
    }
    n = (size_t)h3_st_tensor_elements(t);
    if (want && n != want) {
        fail(error, error_size, "tensor count mismatch");
        return NULL;
    }
    buf = (float *)malloc(n * sizeof(float));
    if (!buf) {
        fail(error, error_size, "OOM");
        return NULL;
    }
    if (t->dtype == H3_DTYPE_F32) {
        if (!h3_vae_int8_store_copy(store, (size_t)t->file_offset, buf,
                                    n * sizeof(float), error, error_size)) {
            free(buf);
            return NULL;
        }
        return buf;
    }
    if (t->dtype != H3_DTYPE_BF16 && t->dtype != H3_DTYPE_F16) {
        free(buf);
        fail(error, error_size, "unsupported dtype");
        return NULL;
    }
    u16 = (uint16_t *)malloc(n * sizeof(uint16_t));
    if (!u16 || !h3_vae_int8_store_copy(store, (size_t)t->file_offset, u16,
                                        n * sizeof(uint16_t), error,
                                        error_size)) {
        free(u16);
        free(buf);
        return NULL;
    }
    for (i = 0; i < n; i++) {
        if (t->dtype == H3_DTYPE_BF16) {
            uint32_t bits = ((uint32_t)u16[i]) << 16;
            memcpy(&buf[i], &bits, sizeof(float));
        } else {
            uint16_t h = u16[i];
            uint32_t sign = (uint32_t)(h >> 15) << 31;
            uint32_t exp = (h >> 10) & 0x1F;
            uint32_t mant = h & 0x3FF;
            uint32_t bits;
            if (exp == 0) {
                if (mant == 0) bits = sign;
                else {
                    exp = 127 - 15 + 1;
                    while ((mant & 0x400) == 0) {
                        mant <<= 1;
                        exp--;
                    }
                    mant &= 0x3FF;
                    bits = sign | (exp << 23) | (mant << 13);
                }
            } else if (exp == 31) {
                bits = sign | 0x7F800000u | (mant << 13);
            } else {
                bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
            }
            memcpy(&buf[i], &bits, sizeof(float));
        }
    }
    free(u16);
    return buf;
}

static int ensure_vae_decode_heads(const h3_vae_int8_store *store,
                                   h3_vae_int8_cuda_weights *w, char *error,
                                   size_t error_size) {
    if (!store || !w) return 0;
    if (w->h_proj_w && w->h_embed_w && w->h_post_w && w->h_registers &&
        w->h_norm_w)
        return 1;
    w->h_post_w = load_f32(store, "post_quant_conv.weight",
                           (size_t)H3_VAE_INT8_LATENT_CH * H3_VAE_INT8_LATENT_CH,
                           error, error_size);
    w->h_post_b = load_f32(store, "post_quant_conv.bias", H3_VAE_INT8_LATENT_CH,
                           error, error_size);
    w->h_embed_w = load_f32(store, "decoder.x_embedder.weight",
                            (size_t)H3_VAE_INT8_HIDDEN * H3_VAE_INT8_LATENT_CH,
                            error, error_size);
    w->h_embed_b = load_f32(store, "decoder.x_embedder.bias", H3_VAE_INT8_HIDDEN,
                            error, error_size);
    w->h_registers = load_f32(store, "decoder.register_tokens",
                              (size_t)H3_VAE_INT8_REGISTERS * H3_VAE_INT8_HIDDEN,
                              error, error_size);
    w->h_norm_w = load_f32(store, "decoder.norm_out.weight", H3_VAE_INT8_HIDDEN,
                           error, error_size);
    w->h_norm_b = load_f32(store, "decoder.norm_out.bias", H3_VAE_INT8_HIDDEN,
                           error, error_size);
    w->h_proj_w = load_f32(store, "decoder.proj_out.weight",
                           (size_t)H3_VAE_INT8_OUTPUT_PATCH * H3_VAE_INT8_HIDDEN,
                           error, error_size);
    w->h_proj_b = load_f32(store, "decoder.proj_out.bias",
                           H3_VAE_INT8_OUTPUT_PATCH, error, error_size);
    if (!w->h_post_w || !w->h_post_b || !w->h_embed_w || !w->h_embed_b ||
        !w->h_registers || !w->h_norm_w || !w->h_norm_b || !w->h_proj_w ||
        !w->h_proj_b)
        return 0;
    return 1;
}

/* Fully GPU-resident block: d_seq already on device.
 * Prefers FP16 Tensor-Core GEMM when half weights are present. */
static int run_block_cuda_w_dev(h3_cuda *cuda, const h3_vae_block_w *bw,
                                float *d_seq, float *d_ws, size_t seq,
                                const float *d_rope_cos,
                                const float *d_rope_sin,
                                const float *d_qk_ones, int convrot_gs,
                                char *error, size_t error_size) {
    const size_t hidden = H3_VAE_INT8_HIDDEN;
    const size_t heads = H3_VAE_INT8_HEADS;
    const size_t head_dim = H3_VAE_INT8_HEAD_DIM;
    const size_t inner = heads * head_dim;
    const size_t ffn = H3_VAE_INT8_FFN;
    float scale = 1.0f / sqrtf((float)head_dim);
    float *d_tmp, *d_qkv, *d_q, *d_k, *d_v, *d_attn, *d_branch, *d_ff, *d_act;
    float *d_rot;
    void *d_hscratch;
    size_t off = 0;
    size_t max_mn, max_mk, hbytes, rot_k;
    char *base = (char *)d_ws;
    int use_f16;
    int saved_quiet;
    const float *d_gemm_x;

    if (!bw->d_w_qkv || !bw->d_w_out || !bw->d_w_w1 || !bw->d_w_w2 ||
        !bw->d_norm1 || !bw->d_norm2 || !bw->d_scale1 || !bw->d_scale2) {
        fail(error, error_size, "vae block missing device weights");
        return 0;
    }
    use_f16 = (bw->d_w_qkv_h && bw->d_w_out_h && bw->d_w_w1_h && bw->d_w_w2_h);
    {
        static int logged_m4;
        if (!logged_m4) {
            fprintf(stderr, "vae_gemm: f16_direct=%d fused_epi=%d act_f16=%d\n",
                    h3_vae_int8_gemm_f16_direct(),
                    h3_vae_int8_fused_epilogue(), h3_vae_int8_act_f16());
            logged_m4 = 1;
        }
    }
    if (use_f16 && h3_vae_int8_act_f16())
        return run_block_cuda_w_dev_f16(cuda, bw, d_seq, d_ws, seq, d_rope_cos,
                                        d_rope_sin, d_qk_ones, convrot_gs,
                                        error, error_size);

    /* Workspace: FP32 temps + optional half scratch for largest gemm I/O. */
    max_mk = seq * (bw->k_qkv > bw->k_w1 ? bw->k_qkv : bw->k_w1);
    if (seq * bw->k_out > max_mk) max_mk = seq * bw->k_out;
    if (seq * bw->k_w2 > max_mk) max_mk = seq * bw->k_w2;
    max_mn = seq * (bw->n_qkv > bw->n_w1 ? bw->n_qkv : bw->n_w1);
    if (seq * bw->n_out > max_mn) max_mn = seq * bw->n_out;
    if (seq * bw->n_w2 > max_mn) max_mn = seq * bw->n_w2;
    hbytes = use_f16 ? (max_mk + max_mn) * 2 : 0;
    rot_k = bw->k_w2 > bw->k_qkv ? bw->k_w2 : bw->k_qkv;
    if (bw->k_out > rot_k) rot_k = bw->k_out;
    if (bw->k_w1 > rot_k) rot_k = bw->k_w1;

#define H3_VAE_TAKE(n)                                                         \
    (off += (n) * sizeof(float),                                               \
     (float *)(base + (off - (n) * sizeof(float))))
    d_tmp = H3_VAE_TAKE(seq * hidden);
    d_qkv = H3_VAE_TAKE(seq * bw->n_qkv);
    d_q = H3_VAE_TAKE(seq * inner);
    d_k = H3_VAE_TAKE(seq * inner);
    d_v = H3_VAE_TAKE(seq * inner);
    d_attn = H3_VAE_TAKE(seq * hidden);
    d_branch = H3_VAE_TAKE(seq * hidden);
    d_ff = H3_VAE_TAKE(seq * bw->n_w1);
    d_act = H3_VAE_TAKE(seq * ffn);
    d_rot = H3_VAE_TAKE(seq * rot_k);
#undef H3_VAE_TAKE
    d_hscratch = use_f16 ? (void *)(base + off) : NULL;
    off += hbytes;

    saved_quiet = h3_cuda_quiet(cuda);
    h3_cuda_set_quiet(cuda, 1);

    if (!h3_cuda_rms_norm_f32_dev(cuda, d_seq, bw->d_norm1, seq, hidden, 1e-5f,
                                  d_tmp, error, error_size)) {
        h3_cuda_set_quiet(cuda, saved_quiet);
        return 0;
    }
    d_gemm_x = d_tmp;
    if (convrot_gs > 0) {
        if (!h3_cuda_convrot_act_f32_dd(cuda, d_tmp, seq, bw->k_qkv,
                                        convrot_gs, d_rot, error,
                                        error_size)) {
            h3_cuda_set_quiet(cuda, saved_quiet);
            return 0;
        }
        d_gemm_x = d_rot;
    }
    if (use_f16) {
        if (!gemm_f16_via(cuda, d_gemm_x, bw->d_w_qkv_h, seq, bw->n_qkv,
                          bw->k_qkv, d_qkv, d_hscratch, error, error_size)) {
            h3_cuda_set_quiet(cuda, saved_quiet);
            return 0;
        }
    } else if (!h3_cuda_gemm_f32_dd(cuda, d_gemm_x, bw->d_w_qkv, seq,
                                    bw->n_qkv, bw->k_qkv, d_qkv, NULL, error,
                                    error_size)) {
        h3_cuda_set_quiet(cuda, saved_quiet);
        return 0;
    }
    if (bw->d_b_qkv &&
        !h3_cuda_bias_add_f32_dev(cuda, d_qkv, bw->d_b_qkv, seq, bw->n_qkv,
                                  error, error_size)) {
        h3_cuda_set_quiet(cuda, saved_quiet);
        return 0;
    }
    if (!d_rope_cos || !d_rope_sin || !d_qk_ones) {
        fail(error, error_size, "vae block missing rope");
        h3_cuda_set_quiet(cuda, saved_quiet);
        return 0;
    }
    {
        int mode = vae_qkv_layout_mode();
        int want_f32 = env_is_one("H3_VAE_SDPA_F32");
        size_t attn_seq = seq;
        size_t n_batch = 1;
        if (g_vae_attn_seq > 0 && seq >= g_vae_attn_seq &&
            (seq % g_vae_attn_seq) == 0) {
            attn_seq = g_vae_attn_seq;
            n_batch = seq / attn_seq;
        }
        if (mode == 2) {
            if (!h3_cuda_qkv_thirds_to_heads_f32_dev(cuda, d_qkv, seq, heads,
                                                     head_dim, d_q, d_k, d_v,
                                                     error, error_size)) {
                h3_cuda_set_quiet(cuda, saved_quiet);
                return 0;
            }
            if (want_f32) {
                if (!h3_cuda_sdpa_f32_dev(cuda, d_q, d_k, d_v, seq, heads,
                                          head_dim, scale, d_attn, NULL, error,
                                          error_size)) {
                    h3_cuda_set_quiet(cuda, saved_quiet);
                    return 0;
                }
            } else if (!h3_cuda_sdpa_f16_dev(cuda, d_q, d_k, d_v, seq, heads,
                                             head_dim, scale, d_attn, NULL,
                                             error, error_size)) {
                h3_cuda_set_quiet(cuda, saved_quiet);
                return 0;
            }
        } else if (want_f32) {
            int grouped = (mode == 1);
            int split_half = 1;
            const char *half_off = getenv("H3_DIT_ROPE_HALF_OFFSET");
            const char *e = getenv("H3_DIT_ROPE_SPLIT_HALF");
            if (half_off && half_off[0] == '1' && half_off[1] == '\0')
                split_half = 0;
            if (e && e[0] == '0' && e[1] == '\0') split_half = 0;
            if (e && e[0] == '1' && e[1] == '\0') split_half = 1;
            if (!h3_cuda_qkv_rope_f32_dev_layout(
                    cuda, d_qkv, d_qk_ones, d_qk_ones, d_rope_cos, d_rope_sin,
                    seq, heads, head_dim, H3_VAE_INT8_ROPE_HALF, 1e-5f,
                    grouped, split_half, d_q, d_k, d_v, NULL, error,
                    error_size)) {
                h3_cuda_set_quiet(cuda, saved_quiet);
                return 0;
            }
            /* Full-frame: F16 cuDNN waffles. F32 cuDNN (TF32 off) via
             * h3_cuda_sdpa_f32_dev; naive flash only if no F32 engine. */
            {
                static int logged_be;
                if (!h3_cuda_sdpa_f32_dev(cuda, d_q, d_k, d_v, seq, heads,
                                          head_dim, scale, d_attn, NULL, error,
                                          error_size)) {
                    h3_cuda_set_quiet(cuda, saved_quiet);
                    return 0;
                }
                if (!logged_be) {
                    fprintf(stderr,
                            "vae_fullframe_sdpa: seq=%zu heads=%zu d=%zu "
                            "backend=f32 (cudnn-or-flash; see sdpa_f32)\n",
                            seq, heads, head_dim);
                    logged_be = 1;
                }
            }
        } else {
            if (mode == 0) {
                float *h_qkv =
                    (float *)malloc(seq * inner * 3 * sizeof(float));
                float *h_inter =
                    (float *)malloc(seq * inner * 3 * sizeof(float));
                if (!h_qkv || !h_inter) {
                    free(h_qkv);
                    free(h_inter);
                    fail(error, error_size, "OOM vae qkv pack");
                    h3_cuda_set_quiet(cuda, saved_quiet);
                    return 0;
                }
                if (cudaMemcpy(h_qkv, d_qkv, seq * inner * 3 * sizeof(float),
                               cudaMemcpyDeviceToHost) != cudaSuccess) {
                    free(h_qkv);
                    free(h_inter);
                    fail(error, error_size, "vae qkv D2H failed");
                    h3_cuda_set_quiet(cuda, saved_quiet);
                    return 0;
                }
                pack_thirds_to_interleaved(h_qkv, h_inter, seq, heads,
                                           head_dim);
                if (cudaMemcpy(d_qkv, h_inter, seq * inner * 3 * sizeof(float),
                               cudaMemcpyHostToDevice) != cudaSuccess) {
                    free(h_qkv);
                    free(h_inter);
                    fail(error, error_size, "vae qkv H2D failed");
                    h3_cuda_set_quiet(cuda, saved_quiet);
                    return 0;
                }
                free(h_qkv);
                free(h_inter);
            }
            /* Metal VAE: interleaved QKV + RoPE fused into FP16 SDPA scratch.
             * n_batch>1: one cuDNN SDPA over B tiles (Comfy _decode_tile_row);
             * RoPE stays tile-local (same table, origin 0). */
            {
                int split_half = 1;
                const char *half_off = getenv("H3_DIT_ROPE_HALF_OFFSET");
                const char *e = getenv("H3_DIT_ROPE_SPLIT_HALF");
                if (half_off && half_off[0] == '1' && half_off[1] == '\0')
                    split_half = 0;
                if (e && e[0] == '0' && e[1] == '\0') split_half = 0;
                if (e && e[0] == '1' && e[1] == '\0') split_half = 1;
                if (!h3_cuda_qkv_rope_sdpa_f16_dev_layout_nb(
                        cuda, d_qkv, d_qk_ones, d_qk_ones, d_rope_cos,
                        d_rope_sin, attn_seq, n_batch, heads, head_dim,
                        H3_VAE_INT8_ROPE_HALF, 1e-5f, scale, 1, split_half,
                        d_attn, error, error_size)) {
                    h3_cuda_set_quiet(cuda, saved_quiet);
                    return 0;
                }
            }
        }
    }
    if (use_f16) {
        d_gemm_x = d_attn;
        if (convrot_gs > 0) {
            if (!h3_cuda_convrot_act_f32_dd(cuda, d_attn, seq, bw->k_out,
                                            convrot_gs, d_rot, error,
                                            error_size)) {
                h3_cuda_set_quiet(cuda, saved_quiet);
                return 0;
            }
            d_gemm_x = d_rot;
        }
        if (!gemm_f16_via(cuda, d_gemm_x, bw->d_w_out_h, seq, bw->n_out,
                          bw->k_out, d_branch, d_hscratch, error, error_size)) {
            h3_cuda_set_quiet(cuda, saved_quiet);
            return 0;
        }
    } else {
        d_gemm_x = d_attn;
        if (convrot_gs > 0) {
            if (!h3_cuda_convrot_act_f32_dd(cuda, d_attn, seq, bw->k_out,
                                            convrot_gs, d_rot, error,
                                            error_size)) {
                h3_cuda_set_quiet(cuda, saved_quiet);
                return 0;
            }
            d_gemm_x = d_rot;
        }
        if (!h3_cuda_gemm_f32_dd(cuda, d_gemm_x, bw->d_w_out, seq, bw->n_out,
                                 bw->k_out, d_branch, NULL, error,
                                 error_size)) {
            h3_cuda_set_quiet(cuda, saved_quiet);
            return 0;
        }
    }
    if (h3_vae_int8_fused_epilogue()) {
        if (!h3_cuda_vae_bias_residual_scale_f32_dev(
                cuda, d_seq, d_branch, bw->d_b_out, bw->d_scale1, seq, hidden,
                error, error_size)) {
            h3_cuda_set_quiet(cuda, saved_quiet);
            return 0;
        }
    } else {
        if (bw->d_b_out &&
            !h3_cuda_bias_add_f32_dev(cuda, d_branch, bw->d_b_out, seq,
                                      bw->n_out, error, error_size)) {
            h3_cuda_set_quiet(cuda, saved_quiet);
            return 0;
        }
        if (!h3_cuda_residual_scale_f32_dev(cuda, d_seq, d_branch, bw->d_scale1,
                                            seq, hidden, error, error_size)) {
            h3_cuda_set_quiet(cuda, saved_quiet);
            return 0;
        }
    }

    if (!h3_cuda_rms_norm_f32_dev(cuda, d_seq, bw->d_norm2, seq, hidden, 1e-5f,
                                  d_tmp, error, error_size)) {
        h3_cuda_set_quiet(cuda, saved_quiet);
        return 0;
    }
    d_gemm_x = d_tmp;
    if (convrot_gs > 0) {
        if (!h3_cuda_convrot_act_f32_dd(cuda, d_tmp, seq, bw->k_w1, convrot_gs,
                                        d_rot, error, error_size)) {
            h3_cuda_set_quiet(cuda, saved_quiet);
            return 0;
        }
        d_gemm_x = d_rot;
    }
    if (use_f16) {
        if (!gemm_f16_via(cuda, d_gemm_x, bw->d_w_w1_h, seq, bw->n_w1,
                          bw->k_w1, d_ff, d_hscratch, error, error_size)) {
            h3_cuda_set_quiet(cuda, saved_quiet);
            return 0;
        }
    } else if (!h3_cuda_gemm_f32_dd(cuda, d_gemm_x, bw->d_w_w1, seq, bw->n_w1,
                                    bw->k_w1, d_ff, NULL, error, error_size)) {
        h3_cuda_set_quiet(cuda, saved_quiet);
        return 0;
    }
    if (h3_vae_int8_fused_epilogue()) {
        if (!h3_cuda_vae_bias_silu_mul_f32_dev(cuda, d_ff, bw->d_b_w1, seq, ffn,
                                               d_act, error, error_size)) {
            h3_cuda_set_quiet(cuda, saved_quiet);
            return 0;
        }
    } else {
        if (bw->d_b_w1 &&
            !h3_cuda_bias_add_f32_dev(cuda, d_ff, bw->d_b_w1, seq, bw->n_w1,
                                      error, error_size)) {
            h3_cuda_set_quiet(cuda, saved_quiet);
            return 0;
        }
        if (!h3_cuda_silu_mul_f32_dev(cuda, d_ff, seq, ffn, d_act, error,
                                      error_size)) {
            h3_cuda_set_quiet(cuda, saved_quiet);
            return 0;
        }
    }
    d_gemm_x = d_act;
    if (convrot_gs > 0) {
        if (!h3_cuda_convrot_act_f32_dd(cuda, d_act, seq, bw->k_w2, convrot_gs,
                                        d_rot, error, error_size)) {
            h3_cuda_set_quiet(cuda, saved_quiet);
            return 0;
        }
        d_gemm_x = d_rot;
    }
    if (use_f16) {
        if (!gemm_f16_via(cuda, d_gemm_x, bw->d_w_w2_h, seq, bw->n_w2,
                          bw->k_w2, d_branch, d_hscratch, error, error_size)) {
            h3_cuda_set_quiet(cuda, saved_quiet);
            return 0;
        }
    } else if (!h3_cuda_gemm_f32_dd(cuda, d_gemm_x, bw->d_w_w2, seq, bw->n_w2,
                                    bw->k_w2, d_branch, NULL, error,
                                    error_size)) {
        h3_cuda_set_quiet(cuda, saved_quiet);
        return 0;
    }
    if (h3_vae_int8_fused_epilogue()) {
        if (!h3_cuda_vae_bias_residual_scale_f32_dev(
                cuda, d_seq, d_branch, bw->d_b_w2, bw->d_scale2, seq, hidden,
                error, error_size)) {
            h3_cuda_set_quiet(cuda, saved_quiet);
            return 0;
        }
    } else {
        if (bw->d_b_w2 &&
            !h3_cuda_bias_add_f32_dev(cuda, d_branch, bw->d_b_w2, seq, bw->n_w2,
                                      error, error_size)) {
            h3_cuda_set_quiet(cuda, saved_quiet);
            return 0;
        }
        if (!h3_cuda_residual_scale_f32_dev(cuda, d_seq, d_branch, bw->d_scale2,
                                            seq, hidden, error, error_size)) {
            h3_cuda_set_quiet(cuda, saved_quiet);
            return 0;
        }
    }
    h3_cuda_set_quiet(cuda, saved_quiet);
    return 1;
}

static int run_block_cuda_w(h3_cuda *cuda, const h3_vae_block_w *bw,
                            float *seq_h, size_t seq, const float *rope_cos,
                            const float *rope_sin, const float *d_rope_cos,
                            const float *d_rope_sin, const float *d_qk_ones,
                            char *error, size_t error_size) {
    /* Prefer fully device-resident path when weights are on GPU. */
    if (bw->d_w_qkv && bw->d_norm1 && d_rope_cos && d_rope_sin && d_qk_ones) {
        size_t hidden = H3_VAE_INT8_HIDDEN;
        size_t inner = (size_t)H3_VAE_INT8_HEADS * H3_VAE_INT8_HEAD_DIM;
        size_t need =
            seq * (hidden * 3 + bw->n_qkv + inner * 3 + bw->n_w1 +
                   H3_VAE_INT8_FFN + 8192) *
            sizeof(float);
        float *d_seq = NULL, *d_ws = NULL;
        int ok = 0;
        if (cudaMalloc((void **)&d_seq, seq * hidden * sizeof(float)) !=
                cudaSuccess ||
            cudaMalloc((void **)&d_ws, need) != cudaSuccess) {
            fail(error, error_size, "OOM vae resident block");
            if (d_seq) cudaFree(d_seq);
            if (d_ws) cudaFree(d_ws);
            /* fall through to host staging path */
        } else if (cudaMemcpy(d_seq, seq_h, seq * hidden * sizeof(float),
                              cudaMemcpyHostToDevice) != cudaSuccess) {
            fail(error, error_size, "vae resident H2D failed");
            cudaFree(d_seq);
            cudaFree(d_ws);
        } else if (!run_block_cuda_w_dev(cuda, bw, d_seq, d_ws, seq, d_rope_cos,
                                         d_rope_sin, d_qk_ones, 256, error,
                                         error_size)) {
            cudaFree(d_seq);
            cudaFree(d_ws);
            return 0;
        } else if (cudaMemcpy(seq_h, d_seq, seq * hidden * sizeof(float),
                              cudaMemcpyDeviceToHost) != cudaSuccess) {
            fail(error, error_size, "vae resident D2H failed");
            cudaFree(d_seq);
            cudaFree(d_ws);
            return 0;
        } else {
            cudaFree(d_seq);
            cudaFree(d_ws);
            return 1;
        }
        (void)ok;
    }

    {
        float *tmp = NULL, *qkv_y = NULL, *attn = NULL, *branch = NULL;
        float *ff = NULL, *act = NULL;
        size_t hidden = H3_VAE_INT8_HIDDEN;
        int ok = 0;
        tmp = (float *)malloc(seq * hidden * sizeof(float));
        qkv_y = (float *)malloc(seq * bw->n_qkv * sizeof(float));
        attn = (float *)malloc(seq * hidden * sizeof(float));
        branch = (float *)malloc(seq * hidden * sizeof(float));
        ff = (float *)malloc(seq * bw->n_w1 * sizeof(float));
        act = (float *)malloc(seq * H3_VAE_INT8_FFN * sizeof(float));
        if (!tmp || !qkv_y || !attn || !branch || !ff || !act) {
            fail(error, error_size, "OOM vae cuda block");
            goto done;
        }

        rms_norm(seq_h, bw->norm1, seq, hidden, 1e-5f, tmp);
        if (!linear_cuda_w(cuda, bw->w_qkv, bw->d_w_qkv, bw->b_qkv, bw->n_qkv,
                           bw->k_qkv, tmp, seq, qkv_y, 256, error,
                           error_size))
            goto done;
        if (!sdpa_cuda_rope(cuda, qkv_y, rope_cos, rope_sin, seq,
                            H3_VAE_INT8_HEADS, H3_VAE_INT8_HEAD_DIM, attn,
                            error, error_size))
            goto done;
        if (!linear_cuda_w(cuda, bw->w_out, bw->d_w_out, bw->b_out, bw->n_out,
                           bw->k_out, attn, seq, branch, 256, error,
                           error_size))
            goto done;
        {
            size_t i, c;
            for (i = 0; i < seq; i++)
                for (c = 0; c < hidden; c++)
                    seq_h[i * hidden + c] +=
                        branch[i * hidden + c] * bw->scale1[c];
        }

        rms_norm(seq_h, bw->norm2, seq, hidden, 1e-5f, tmp);
        if (!linear_cuda_w(cuda, bw->w_w1, bw->d_w_w1, bw->b_w1, bw->n_w1,
                           bw->k_w1, tmp, seq, ff, 256, error, error_size))
            goto done;
        silu_mul(ff, seq, H3_VAE_INT8_FFN, act);
        if (!linear_cuda_w(cuda, bw->w_w2, bw->d_w_w2, bw->b_w2, bw->n_w2,
                           bw->k_w2, act, seq, branch, 256, error,
                           error_size))
            goto done;
        {
            size_t i, c;
            for (i = 0; i < seq; i++)
                for (c = 0; c < hidden; c++)
                    seq_h[i * hidden + c] +=
                        branch[i * hidden + c] * bw->scale2[c];
        }
        ok = 1;
    done:
        free(tmp);
        free(qkv_y);
        free(attn);
        free(branch);
        free(ff);
        free(act);
        return ok;
    }
}

static int run_block_cuda(h3_cuda *cuda, const h3_vae_int8_store *store,
                          unsigned block, float *seq_h, size_t seq,
                          const float *rope_cos, const float *rope_sin,
                          char *error, size_t error_size) {
    char base[128];
    h3_vae_int8_linear qkv, out, w1, w2;
    h3_vae_block_w bw;
    float *norm1 = NULL, *norm2 = NULL, *scale1 = NULL, *scale2 = NULL;
    size_t hidden = H3_VAE_INT8_HIDDEN;
    int ok = 0;
    memset(&bw, 0, sizeof(bw));

    snprintf(base, sizeof(base), "decoder.transformer_blocks.%u.attn.to_qkv",
             block);
    if (!h3_vae_int8_find_linear(store, base, &qkv, error, error_size))
        return 0;
    snprintf(base, sizeof(base), "decoder.transformer_blocks.%u.attn.to_out",
             block);
    if (!h3_vae_int8_find_linear(store, base, &out, error, error_size))
        return 0;
    snprintf(base, sizeof(base), "decoder.transformer_blocks.%u.ff.w1", block);
    if (!h3_vae_int8_find_linear(store, base, &w1, error, error_size)) return 0;
    snprintf(base, sizeof(base), "decoder.transformer_blocks.%u.ff.w2", block);
    if (!h3_vae_int8_find_linear(store, base, &w2, error, error_size)) return 0;

    snprintf(base, sizeof(base), "decoder.transformer_blocks.%u.norm1.weight",
             block);
    norm1 = load_f32(store, base, hidden, error, error_size);
    snprintf(base, sizeof(base), "decoder.transformer_blocks.%u.norm2.weight",
             block);
    norm2 = load_f32(store, base, hidden, error, error_size);
    snprintf(base, sizeof(base), "decoder.transformer_blocks.%u.scale1", block);
    scale1 = load_f32(store, base, hidden, error, error_size);
    snprintf(base, sizeof(base), "decoder.transformer_blocks.%u.scale2", block);
    scale2 = load_f32(store, base, hidden, error, error_size);
    bw.w_qkv = h3_vae_int8_dequant_f32(store, &qkv, error, error_size);
    bw.w_out = h3_vae_int8_dequant_f32(store, &out, error, error_size);
    bw.w_w1 = h3_vae_int8_dequant_f32(store, &w1, error, error_size);
    bw.w_w2 = h3_vae_int8_dequant_f32(store, &w2, error, error_size);
    if (!norm1 || !norm2 || !scale1 || !scale2 || !bw.w_qkv || !bw.w_out ||
        !bw.w_w1 || !bw.w_w2) {
        fail(error, error_size, "OOM/load vae cuda block");
        goto done;
    }
    if (!load_bias(store, &qkv, &bw.b_qkv, error, error_size) ||
        !load_bias(store, &out, &bw.b_out, error, error_size) ||
        !load_bias(store, &w1, &bw.b_w1, error, error_size) ||
        !load_bias(store, &w2, &bw.b_w2, error, error_size))
        goto done;
    bw.norm1 = norm1;
    bw.norm2 = norm2;
    bw.scale1 = scale1;
    bw.scale2 = scale2;
    bw.n_qkv = qkv.n;
    bw.k_qkv = qkv.k;
    bw.n_out = out.n;
    bw.k_out = out.k;
    bw.n_w1 = w1.n;
    bw.k_w1 = w1.k;
    bw.n_w2 = w2.n;
    bw.k_w2 = w2.k;
    ok = run_block_cuda_w(cuda, &bw, seq_h, seq, rope_cos, rope_sin, NULL,
                            NULL, NULL, error, error_size);
done:
    free(bw.w_qkv);
    free(bw.w_out);
    free(bw.w_w1);
    free(bw.w_w2);
    free(bw.b_qkv);
    free(bw.b_out);
    free(bw.b_w1);
    free(bw.b_w2);
    free(norm1);
    free(norm2);
    free(scale1);
    free(scale2);
    return ok;
}

/* ---- Metal-style spatial tiling (TILE_PIXELS=256, overlap≥64) ----
 * Full-frame attention at 480p (seq~10k) collapses to block wallpaper;
 * decode ≤256px tiles then blend, matching h3_video_vae.c. */
enum {
    H3_VAE_CUDA_TILE_PIXELS = 256,
    H3_VAE_CUDA_TILE_OVERLAP_MIN = 64
};

typedef struct {
    int count;
    int length;
    int *starts;
    int *overlaps;
} h3_vae_tile_axis;

static void h3_vae_tile_axis_free(h3_vae_tile_axis *axis) {
    if (!axis) return;
    free(axis->starts);
    free(axis->overlaps);
    memset(axis, 0, sizeof(*axis));
}

static int h3_vae_tile_overlap_min(void) {
    int v = env_int_default("H3_VAE_TILE_OVERLAP", H3_VAE_CUDA_TILE_OVERLAP_MIN);
    if (v < 32) v = 32;
    if (v % H3_VAE_INT8_SPATIAL_RATIO) v = H3_VAE_CUDA_TILE_OVERLAP_MIN;
    return v;
}

static int h3_vae_tile_count(int extent, int tile_pixels, int overlap_min) {
    int count;
    if (extent <= tile_pixels) return 1;
    count = (extent + tile_pixels - 1) / tile_pixels;
    while (tile_pixels * count - overlap_min * (count - 1) < extent)
        count++;
    return count;
}

/* Match Metal h3_video_vae.c: if H3_VAE_TILE_PIXELS is unset, pick 256–320
 * to minimize tile count × area. Fixed 256px on 864×480 is a 5×3 grid whose
 * y=112 seam cuts two-shot faces; 272px is 4×2 and keeps heads in the top
 * tile. Full-frame 480 attention still waffles — do not exceed 320 unless
 * the env override is set. */
static int h3_vae_configured_tile_pixels(int pixel_h, int pixel_w) {
    const char *value = getenv("H3_VAE_TILE_PIXELS");
    int ratio = H3_VAE_INT8_SPATIAL_RATIO;
    int overlap = h3_vae_tile_overlap_min();
    int best = H3_VAE_CUDA_TILE_PIXELS;
    uint64_t best_score = UINT64_MAX;
    int pixels;
    if (value && *value) {
        int v = env_int_default("H3_VAE_TILE_PIXELS", H3_VAE_CUDA_TILE_PIXELS);
        if (v < 128) v = 128;
        if (v > 512) v = 512;
        if (v % ratio) v = H3_VAE_CUDA_TILE_PIXELS;
        return v;
    }
    if (pixel_h < 1 || pixel_w < 1) return best;
    for (pixels = H3_VAE_CUDA_TILE_PIXELS; pixels <= 320; pixels += ratio) {
        int ny = h3_vae_tile_count(pixel_h, pixels, overlap);
        int nx = h3_vae_tile_count(pixel_w, pixels, overlap);
        uint64_t score = (uint64_t)ny * (uint64_t)nx * (uint64_t)pixels *
                         (uint64_t)pixels;
        if (score < best_score) {
            best = pixels;
            best_score = score;
        }
    }
    return best;
}

static int h3_vae_tile_axis_build(int extent, int tile_pixels,
                                  h3_vae_tile_axis *axis, char *error,
                                  size_t error_size) {
    int count, index, remaining, unit;
    int ratio = H3_VAE_INT8_SPATIAL_RATIO;
    int overlap_min = h3_vae_tile_overlap_min();
    memset(axis, 0, sizeof(*axis));
    if (extent < 1 || extent % ratio) {
        fail(error, error_size, "VAE tile extent not multiple of 16");
        return 0;
    }
    if (extent <= tile_pixels) {
        axis->count = 1;
        axis->length = extent;
        axis->starts = (int *)calloc(1, sizeof(int));
        if (!axis->starts) {
            fail(error, error_size, "OOM tile axis");
            return 0;
        }
        return 1;
    }
    if (overlap_min >= tile_pixels) {
        fail(error, error_size, "VAE tile overlap must be < tile size");
        return 0;
    }
    count = h3_vae_tile_count(extent, tile_pixels, overlap_min);
    axis->count = count;
    axis->length = tile_pixels;
    axis->starts = (int *)calloc((size_t)count, sizeof(int));
    axis->overlaps = (int *)calloc((size_t)(count - 1), sizeof(int));
    if (!axis->starts || !axis->overlaps) {
        h3_vae_tile_axis_free(axis);
        fail(error, error_size, "OOM tile axis");
        return 0;
    }
    for (index = 0; index < count - 1; index++)
        axis->overlaps[index] = overlap_min;
    remaining = tile_pixels * count - overlap_min * (count - 1) - extent;
    for (unit = 0; unit < remaining / ratio; unit++)
        axis->overlaps[unit % (count - 1)] += ratio;
    for (index = 1; index < count; index++)
        axis->starts[index] =
            axis->starts[index - 1] + tile_pixels - axis->overlaps[index - 1];
    return 1;
}

static float *h3_vae_extract_spatial_chw(const float *chw, int latent_t,
                                         int latent_h, int latent_w,
                                         int start_h, int start_w, int tile_h,
                                         int tile_w, char *error,
                                         size_t error_size) {
    float *out;
    size_t c, t, h, w, n;
    if (!chw || tile_h < 1 || tile_w < 1 || start_h < 0 || start_w < 0 ||
        start_h + tile_h > latent_h || start_w + tile_w > latent_w) {
        fail(error, error_size, "invalid spatial latent tile");
        return NULL;
    }
    n = (size_t)H3_VAE_INT8_LATENT_CH * (size_t)latent_t * (size_t)tile_h *
        (size_t)tile_w;
    out = (float *)malloc(n * sizeof(float));
    if (!out) {
        fail(error, error_size, "OOM spatial latent tile");
        return NULL;
    }
    for (c = 0; c < (size_t)H3_VAE_INT8_LATENT_CH; c++)
        for (t = 0; t < (size_t)latent_t; t++)
            for (h = 0; h < (size_t)tile_h; h++)
                for (w = 0; w < (size_t)tile_w; w++) {
                    size_t src =
                        ((c * (size_t)latent_t + t) * (size_t)latent_h +
                         (size_t)start_h + h) *
                            (size_t)latent_w +
                        (size_t)start_w + w;
                    size_t dst =
                        ((c * (size_t)latent_t + t) * (size_t)tile_h + h) *
                            (size_t)tile_w +
                        w;
                    out[dst] = chw[src];
                }
    return out;
}

/* Smoothstep 0→1 across the overlap strip. Outer canvas edges stay weight 1. */
static float h3_vae_edge_ramp(int pos, int len, int ov_lo, int ov_hi) {
    float w = 1.0f;
    if (ov_lo > 1 && pos < ov_lo) {
        float t = (float)pos / (float)ov_lo;
        w = t * t * (3.0f - 2.0f * t);
    }
    if (ov_hi > 1 && pos >= len - ov_hi) {
        float t = (float)(len - 1 - pos) / (float)ov_hi;
        float w2;
        if (t < 0.0f) t = 0.0f;
        if (t > 1.0f) t = 1.0f;
        w2 = t * t * (3.0f - 2.0f * t);
        if (w2 < w) w = w2;
    }
    if (w < 1e-4f) w = 1e-4f;
    return w;
}

/* Extra 8px feather on known tile cuts. deblock16 only touches 1px at x%16==0. */
static void h3_vae_feather_tile_seams(float *rgb, int frames, int height,
                                      int width, const h3_vae_tile_axis *y_axis,
                                      const h3_vae_tile_axis *x_axis) {
    const int rad = 8;
    const float strength = 0.40f;
    int f, i, d, y, x, c;
    if (!rgb || frames < 1 || height < 32 || width < 32) return;
    for (f = 0; f < frames; f++) {
        float *fr = rgb + (size_t)f * (size_t)height * (size_t)width * 3;
        for (i = 1; i < x_axis->count; i++) {
            int seam = x_axis->starts[i];
            for (d = -rad; d <= rad; d++) {
                int xs = seam + d;
                float w, t;
                if (xs < 1 || xs >= width - 1) continue;
                t = 1.0f - (float)(d < 0 ? -d : d) / (float)rad;
                if (t < 0.0f) t = 0.0f;
                w = strength * t * t * (3.0f - 2.0f * t);
                for (y = 0; y < height; y++) {
                    for (c = 0; c < 3; c++) {
                        size_t p = ((size_t)y * (size_t)width + (size_t)xs) * 3 +
                                   (size_t)c;
                        float left = fr[p - 3];
                        float right = fr[p + 3];
                        float mid = 0.5f * (left + right);
                        fr[p] = fr[p] * (1.0f - w) + mid * w;
                    }
                }
            }
        }
        for (i = 1; i < y_axis->count; i++) {
            int seam = y_axis->starts[i];
            for (d = -rad; d <= rad; d++) {
                int ys = seam + d;
                float w, t;
                if (ys < 1 || ys >= height - 1) continue;
                t = 1.0f - (float)(d < 0 ? -d : d) / (float)rad;
                if (t < 0.0f) t = 0.0f;
                w = strength * t * t * (3.0f - 2.0f * t);
                for (x = 0; x < width; x++) {
                    for (c = 0; c < 3; c++) {
                        size_t p = ((size_t)ys * (size_t)width + (size_t)x) * 3 +
                                   (size_t)c;
                        float up = fr[p - (size_t)width * 3];
                        float down = fr[p + (size_t)width * 3];
                        float mid = 0.5f * (up + down);
                        fr[p] = fr[p] * (1.0f - w) + mid * w;
                    }
                }
            }
        }
    }
}

/* Comfy MiniMaxH3VideoVAE.tiled_decode: sequential linear blend then crop the
 * incoming tile's right/bottom overlap. Node 74 VAEDecode uses this, not
 * full-frame attention. Hann all-cover is a C-stack extra. */
static void h3_vae_comfy_blend_h(float *tile, const float *above, int frames,
                                 int th, int tw, int blend) {
    int f, y, x, c;
    if (blend < 1) return;
    if (blend > th) blend = th;
    for (f = 0; f < frames; f++)
        for (y = 0; y < blend; y++) {
            float wb = (float)y / (float)blend;
            float wa = 1.0f - wb;
            for (x = 0; x < tw; x++) {
                size_t dst = (((size_t)f * (size_t)th + (size_t)y) * (size_t)tw +
                              (size_t)x) *
                             3;
                size_t src =
                    (((size_t)f * (size_t)blend + (size_t)y) * (size_t)tw +
                     (size_t)x) *
                    3;
                for (c = 0; c < 3; c++)
                    tile[dst + (size_t)c] =
                        above[src + (size_t)c] * wa + tile[dst + (size_t)c] * wb;
            }
        }
}

static void h3_vae_comfy_blend_w(float *tile, const float *left, int frames,
                                 int th, int tw, int blend) {
    int f, y, x, c;
    if (blend < 1) return;
    if (blend > tw) blend = tw;
    for (f = 0; f < frames; f++)
        for (y = 0; y < th; y++)
            for (x = 0; x < blend; x++) {
                float wb = (float)x / (float)blend;
                float wa = 1.0f - wb;
                size_t dst = (((size_t)f * (size_t)th + (size_t)y) * (size_t)tw +
                              (size_t)x) *
                             3;
                size_t src =
                    (((size_t)f * (size_t)th + (size_t)y) * (size_t)blend +
                     (size_t)x) *
                    3;
                for (c = 0; c < 3; c++)
                    tile[dst + (size_t)c] =
                        left[src + (size_t)c] * wa + tile[dst + (size_t)c] * wb;
            }
}

static int h3_vae_stitch_rgb_tiles_comfy(float **tiles,
                                         const h3_vae_tile_axis *y_axis,
                                         const h3_vae_tile_axis *x_axis,
                                         int frame_count,
                                         h3_vae_int8_frames *output,
                                         char *error, size_t error_size) {
    int ny = y_axis->count, nx = x_axis->count;
    int th = y_axis->length, tw = x_axis->length;
    int full_h = y_axis->starts[ny - 1] + th;
    int full_w = x_axis->starts[nx - 1] + tw;
    size_t count = (size_t)frame_count * (size_t)full_h * (size_t)full_w * 3;
    size_t tile_n = (size_t)frame_count * (size_t)th * (size_t)tw * 3;
    float *rgb = (float *)calloc(count, sizeof(float));
    float *work = (float *)malloc(tile_n * sizeof(float));
    float **row_tails = NULL;
    float *left_tail = NULL;
    int i, j, f, y, x, c, out_y = 0;
    int failed = 0;
    if (!rgb || !work) {
        free(rgb);
        free(work);
        fail(error, error_size, "OOM Comfy VAE stitch");
        return 0;
    }
    if (ny > 1) {
        row_tails = (float **)calloc((size_t)nx, sizeof(float *));
        if (!row_tails) failed = 1;
    }
    for (i = 0; i < ny && !failed; i++) {
        int ov_bot = (i + 1 < ny) ? y_axis->overlaps[i] : 0;
        int ov_top = i ? y_axis->overlaps[i - 1] : 0;
        float **new_tails = NULL;
        int out_x = 0;
        int wrote_h = 0;
        if (ov_bot > 0) {
            new_tails = (float **)calloc((size_t)nx, sizeof(float *));
            if (!new_tails) {
                failed = 1;
                break;
            }
        }
        free(left_tail);
        left_tail = NULL;
        for (j = 0; j < nx && !failed; j++) {
            int ov_right = (j + 1 < nx) ? x_axis->overlaps[j] : 0;
            int ov_left = j ? x_axis->overlaps[j - 1] : 0;
            const float *src = tiles[i * nx + j];
            int keep_h = th - ov_bot;
            int keep_w = tw - ov_right;
            memcpy(work, src, tile_n * sizeof(float));
            if (ov_bot > 0) {
                size_t tn = (size_t)frame_count * (size_t)ov_bot * (size_t)tw * 3;
                new_tails[j] = (float *)malloc(tn * sizeof(float));
                if (!new_tails[j]) {
                    failed = 1;
                    break;
                }
                for (f = 0; f < frame_count; f++)
                    memcpy(new_tails[j] +
                               (size_t)f * (size_t)ov_bot * (size_t)tw * 3,
                           work + (((size_t)f * (size_t)th +
                                    (size_t)(th - ov_bot)) *
                                       (size_t)tw) *
                                      3,
                           (size_t)ov_bot * (size_t)tw * 3 * sizeof(float));
            }
            if (ov_right > 0) {
                size_t ln = (size_t)frame_count * (size_t)th * (size_t)ov_right *
                            3;
                float *next_left = (float *)malloc(ln * sizeof(float));
                if (!next_left) {
                    failed = 1;
                    break;
                }
                for (f = 0; f < frame_count; f++)
                    for (y = 0; y < th; y++)
                        memcpy(next_left +
                                   (((size_t)f * (size_t)th + (size_t)y) *
                                        (size_t)ov_right) *
                                       3,
                               work +
                                   (((size_t)f * (size_t)th + (size_t)y) *
                                        (size_t)tw +
                                    (size_t)(tw - ov_right)) *
                                       3,
                               (size_t)ov_right * 3 * sizeof(float));
                if (i > 0 && row_tails && row_tails[j])
                    h3_vae_comfy_blend_h(work, row_tails[j], frame_count, th, tw,
                                         ov_top);
                if (j > 0 && left_tail)
                    h3_vae_comfy_blend_w(work, left_tail, frame_count, th, tw,
                                         ov_left);
                free(left_tail);
                left_tail = next_left;
            } else {
                if (i > 0 && row_tails && row_tails[j])
                    h3_vae_comfy_blend_h(work, row_tails[j], frame_count, th, tw,
                                         ov_top);
                if (j > 0 && left_tail)
                    h3_vae_comfy_blend_w(work, left_tail, frame_count, th, tw,
                                         ov_left);
                free(left_tail);
                left_tail = NULL;
            }
            if (keep_h < 1) keep_h = 1;
            if (keep_w < 1) keep_w = 1;
            if (out_y + keep_h > full_h) keep_h = full_h - out_y;
            if (out_x + keep_w > full_w) keep_w = full_w - out_x;
            for (f = 0; f < frame_count; f++)
                for (y = 0; y < keep_h; y++)
                    for (x = 0; x < keep_w; x++) {
                        size_t dst =
                            ((((size_t)f * (size_t)full_h + (size_t)(out_y + y)) *
                                  (size_t)full_w +
                              (size_t)(out_x + x))) *
                            3;
                        size_t src_i =
                            (((size_t)f * (size_t)th + (size_t)y) * (size_t)tw +
                             (size_t)x) *
                            3;
                        for (c = 0; c < 3; c++)
                            rgb[dst + (size_t)c] = work[src_i + (size_t)c];
                    }
            wrote_h = keep_h;
            out_x += keep_w;
        }
        if (row_tails) {
            for (j = 0; j < nx; j++) free(row_tails[j]);
            free(row_tails);
        }
        row_tails = new_tails;
        out_y += wrote_h;
    }
    if (row_tails) {
        for (j = 0; j < nx; j++) free(row_tails[j]);
        free(row_tails);
    }
    free(left_tail);
    free(work);
    if (failed || out_y != full_h) {
        free(rgb);
        fail(error, error_size, failed ? "OOM Comfy VAE stitch"
                                       : "Comfy VAE stitch height mismatch");
        return 0;
    }
    output->frames = frame_count;
    output->height = full_h;
    output->width = full_w;
    output->rgb = rgb;
    return 1;
}

/* Distance-weighted blend of every covering tile (legacy C-stack Hann). */
static int h3_vae_stitch_rgb_tiles(float **tiles, const h3_vae_tile_axis *y_axis,
                                   const h3_vae_tile_axis *x_axis,
                                   int frame_count, h3_vae_int8_frames *output,
                                   char *error, size_t error_size) {
    int full_h = y_axis->starts[y_axis->count - 1] + y_axis->length;
    int full_w = x_axis->starts[x_axis->count - 1] + x_axis->length;
    size_t count = (size_t)frame_count * (size_t)full_h * (size_t)full_w * 3;
    size_t pix = (size_t)full_h * (size_t)full_w;
    float *rgb;
    int frame;
    int failed = 0;
    rgb = (float *)malloc(count * sizeof(float));
    if (!rgb) {
        fail(error, error_size, "OOM stitching VAE spatial tiles");
        return 0;
    }
    for (frame = 0; frame < frame_count; frame++) {
        float *acc = (float *)calloc(pix * 3u, sizeof(float));
        float *wsum = (float *)calloc(pix, sizeof(float));
        int tile_y, tile_x, y, x, channel;
        size_t p;
        if (!acc || !wsum) {
            free(acc);
            free(wsum);
            failed = 1;
            break;
        }
        for (tile_y = 0; tile_y < y_axis->count; tile_y++)
            for (tile_x = 0; tile_x < x_axis->count; tile_x++) {
                int index = tile_y * x_axis->count + tile_x;
                const float *current = tiles[index];
                int ov_top = tile_y ? y_axis->overlaps[tile_y - 1] : 0;
                int ov_left = tile_x ? x_axis->overlaps[tile_x - 1] : 0;
                int ov_bot = (tile_y + 1 < y_axis->count)
                                 ? y_axis->overlaps[tile_y]
                                 : 0;
                int ov_right = (tile_x + 1 < x_axis->count)
                                   ? x_axis->overlaps[tile_x]
                                   : 0;
                int y0 = y_axis->starts[tile_y];
                int x0 = x_axis->starts[tile_x];
                int th = y_axis->length;
                int tw = x_axis->length;
                for (y = 0; y < th; y++) {
                    int gy = y0 + y;
                    float wy;
                    if (gy < 0 || gy >= full_h) continue;
                    wy = h3_vae_edge_ramp(y, th, ov_top, ov_bot);
                    for (x = 0; x < tw; x++) {
                        int gx = x0 + x;
                        float w;
                        size_t dst, local;
                        if (gx < 0 || gx >= full_w) continue;
                        w = wy * h3_vae_edge_ramp(x, tw, ov_left, ov_right);
                        dst = (size_t)gy * (size_t)full_w + (size_t)gx;
                        local = (((size_t)frame * (size_t)th + (size_t)y) *
                                     (size_t)tw +
                                 (size_t)x) *
                                3;
                        wsum[dst] += w;
                        for (channel = 0; channel < 3; channel++)
                            acc[dst * 3 + (size_t)channel] +=
                                current[local + (size_t)channel] * w;
                    }
                }
            }
        for (p = 0; p < pix; p++) {
            float s = wsum[p];
            size_t dst = ((size_t)frame * pix + p) * 3;
            if (s < 1e-8f) s = 1.0f;
            rgb[dst + 0] = acc[p * 3 + 0] / s;
            rgb[dst + 1] = acc[p * 3 + 1] / s;
            rgb[dst + 2] = acc[p * 3 + 2] / s;
        }
        free(acc);
        free(wsum);
    }
    if (failed) {
        free(rgb);
        fail(error, error_size, "OOM stitching VAE spatial tiles");
        return 0;
    }
    h3_vae_feather_tile_seams(rgb, frame_count, full_h, full_w, y_axis, x_axis);
    output->frames = frame_count;
    output->height = full_h;
    output->width = full_w;
    output->rgb = rgb;
    h3_vae_int8_deblock16_rgb(rgb, frame_count, full_h, full_w);
    return 1;
}

static int h3_vae_decode_cuda_spatial(
    h3_cuda *cuda, const h3_vae_int8_store *store, const float *latent_chw,
    int latent_t, int latent_h, int latent_w, unsigned n_blocks,
    h3_vae_int8_frames *frames, h3_vae_int8_decode_stats *stats, char *error,
    size_t error_size) {
    h3_vae_tile_axis y_axis, x_axis;
    float **tiles = NULL;
    int *tile_frames = NULL;
    int tile_pixels, pixel_h, pixel_w, ratio;
    int ty, n_tiles = 0, ok = 0, frame_count = 0;
    int i;
    const char *prev_no_tile;
    char prev_buf[8];
    int had_prev = 0;
    h3_vae_int8_cuda_weights *shared_w = NULL;
    double shared_load_ms = 0.0;
    h3_vae_int8_decode_stats st_acc;
    memset(&y_axis, 0, sizeof(y_axis));
    memset(&x_axis, 0, sizeof(x_axis));
    memset(&st_acc, 0, sizeof(st_acc));
    if (frames) memset(frames, 0, sizeof(*frames));
    if (stats) memset(stats, 0, sizeof(*stats));
    ratio = H3_VAE_INT8_SPATIAL_RATIO;
    pixel_h = latent_h * ratio;
    pixel_w = latent_w * ratio;
    tile_pixels = h3_vae_configured_tile_pixels(pixel_h, pixel_w);
    if (!h3_vae_tile_axis_build(pixel_h, tile_pixels, &y_axis, error,
                                error_size) ||
        !h3_vae_tile_axis_build(pixel_w, tile_pixels, &x_axis, error,
                                error_size))
        goto done;
    n_tiles = y_axis.count * x_axis.count;
    tiles = (float **)calloc((size_t)n_tiles, sizeof(float *));
    tile_frames = (int *)calloc((size_t)n_tiles, sizeof(int));
    if (!tiles || !tile_frames) {
        fail(error, error_size, "OOM VAE spatial tile table");
        goto done;
    }
    /* Load/dequant once for all tiles (was the 480p VAE ~90s wall). */
    if (n_blocks > 0) {
        if (!h3_vae_int8_cuda_weights_load(store, n_blocks, &shared_w,
                                           &shared_load_ms, error,
                                           error_size))
            goto done;
        g_vae_borrowed_weights = shared_w;
    }
    prev_no_tile = getenv("H3_VAE_NO_SPATIAL_TILE");
    if (prev_no_tile) {
        snprintf(prev_buf, sizeof(prev_buf), "%s", prev_no_tile);
        had_prev = 1;
    }
    setenv("H3_VAE_NO_SPATIAL_TILE", "1", 1);
    fprintf(stderr,
            "vae_spatial_tile: %dx%d → tiles %dx%d (tile=%dpx overlap≥%d) "
            "%s shared_weights=1 load_ms=%.1f%s\n",
            pixel_w, pixel_h, x_axis.count, y_axis.count, tile_pixels,
            h3_vae_tile_overlap_min(),
            env_is_one("H3_VAE_COMFY_BLEND") ? "comfy_linear_blend=1"
                                             : "hann_blend=1", shared_load_ms,
            env_is_one("H3_VAE_VERBOSE") ? "" : " (H3_VAE_VERBOSE=1 for tiles)");
    ok = 1;
    {
        int bmax = h3_vae_int8_tile_batch_want();
        if (bmax < 1) bmax = 1;
        if (bmax > H3_VAE_MAX_TILE_BATCH) bmax = H3_VAE_MAX_TILE_BATCH;
        bmax = vae_tile_batch_fit(bmax);
        if (bmax > 1 || env_is_one("H3_VAE_FLASH_SDPA")) {
            static int logged_k;
            if (!logged_k) {
                fprintf(stderr,
                        "vae_tile: flash=%d batch=%d graph=%d "
                        "(Comfy _decode_tile_row; tile-local RoPE)\n",
                        env_is_one("H3_VAE_FLASH_SDPA"), bmax,
                        h3_vae_int8_cuda_graph_wanted());
                logged_k = 1;
            }
        }
        for (ty = 0; ty < y_axis.count && ok; ty++) {
            int tx0 = 0;
            while (tx0 < x_axis.count && ok) {
                int nb = x_axis.count - tx0;
                int b;
                float *slices[H3_VAE_MAX_TILE_BATCH];
                h3_vae_int8_frames decoded;
                h3_vae_int8_decode_stats st;
                if (nb > bmax) nb = bmax;
                memset(slices, 0, sizeof(slices));
                memset(&decoded, 0, sizeof(decoded));
                memset(&st, 0, sizeof(st));
                for (b = 0; b < nb; b++) {
                    int tx = tx0 + b;
                    int start_h = y_axis.starts[ty] / ratio;
                    int start_w = x_axis.starts[tx] / ratio;
                    int tile_h = y_axis.length / ratio;
                    int tile_w = x_axis.length / ratio;
                    slices[b] = h3_vae_extract_spatial_chw(
                        latent_chw, latent_t, latent_h, latent_w, start_h,
                        start_w, tile_h, tile_w, error, error_size);
                    if (!slices[b]) {
                        ok = 0;
                        break;
                    }
                }
                if (!ok) {
                    for (b = 0; b < nb; b++) free(slices[b]);
                    break;
                }
                g_vae_batch_n = nb;
                for (b = 1; b < nb; b++) {
                    g_vae_batch_chw[b] = slices[b];
                    memset(&g_vae_batch_frames[b], 0, sizeof(g_vae_batch_frames[b]));
                }
                ok = h3_vae_int8_decode_cuda(cuda, store, slices[0], latent_t,
                                             y_axis.length / ratio,
                                             x_axis.length / ratio, n_blocks,
                                             &decoded, &st, error, error_size);
                g_vae_batch_n = 1;
                for (b = 1; b < H3_VAE_MAX_TILE_BATCH; b++)
                    g_vae_batch_chw[b] = NULL;
                for (b = 0; b < nb; b++) free(slices[b]);
                if (!ok) {
                    h3_vae_int8_frames_free(&decoded);
                    for (b = 1; b < nb; b++)
                        h3_vae_int8_frames_free(&g_vae_batch_frames[b]);
                    if (bmax > 1 && vae_error_oom(error)) {
                        int next = (bmax > 2) ? 2 : 1;
                        fprintf(stderr,
                                "vae_kitchen: OOM at batch=%d, retry batch=%d\n",
                                bmax, next);
                        vae_scratch_drop();
                        bmax = next;
                        ok = 1;
                        if (error && error_size) error[0] = '\0';
                        continue;
                    }
                    break;
                }
                if (decoded.height != y_axis.length ||
                    decoded.width != x_axis.length) {
                    fail(error, error_size, "spatial tile decode size mismatch");
                    h3_vae_int8_frames_free(&decoded);
                    for (b = 1; b < nb; b++)
                        h3_vae_int8_frames_free(&g_vae_batch_frames[b]);
                    ok = 0;
                    break;
                }
                if (frame_count == 0)
                    frame_count = decoded.frames;
                else if (decoded.frames != frame_count) {
                    fail(error, error_size, "spatial tile frame count mismatch");
                    h3_vae_int8_frames_free(&decoded);
                    ok = 0;
                    break;
                }
                for (b = 0; b < nb; b++) {
                    int index = ty * x_axis.count + (tx0 + b);
                    h3_vae_int8_frames *fr =
                        (b == 0) ? &decoded : &g_vae_batch_frames[b];
                    if (b > 0 && (fr->frames != frame_count ||
                                  fr->height != y_axis.length ||
                                  fr->width != x_axis.length)) {
                        fail(error, error_size, "spatial batch tile mismatch");
                        ok = 0;
                        break;
                    }
                    /* Consume this tile's fr->rgb: wait its D2H done-event
                     * (async pipeline) and detach into a malloc buffer so the
                     * pinned pool slot is reusable by the next tile. The
                     * stitchers free tiles[] with plain free() downstream. */
                    if (fr->rgb) {
                        size_t tile_bytes =
                            (size_t)fr->frames * (size_t)fr->height *
                            (size_t)fr->width * 3u * sizeof(float);
                        if (!vae_d2h_sync_and_copy(fr, &tiles[index],
                                                   tile_bytes)) {
                            fail(error, error_size,
                                 "vae spatial tile D2H consume failed");
                            ok = 0;
                            break;
                        }
                    }
                    tile_frames[index] = fr->frames;
                    h3_vae_int8_frames_free(fr);
                }
                if (!ok) break;
                st_acc.blocks_ms += st.blocks_ms;
                st_acc.canvas_h2d_ms += st.canvas_h2d_ms;
                st_acc.unpatch_d2h_ms += st.unpatch_d2h_ms;
                st_acc.total_ms += st.total_ms;
                st_acc.n_blocks = st.n_blocks;
                st_acc.all_finite = st.all_finite;
                if (env_is_one("H3_VAE_VERBOSE"))
                    fprintf(stderr,
                            "  tile_row y=%d x0=%d n=%d frames=%d\n", ty, tx0,
                            nb, frame_count);
                tx0 += nb;
            }
        }
    }
    if (had_prev)
        setenv("H3_VAE_NO_SPATIAL_TILE", prev_buf, 1);
    else
        unsetenv("H3_VAE_NO_SPATIAL_TILE");
    g_vae_borrowed_weights = NULL;
    g_vae_rope_origin_h = g_vae_rope_origin_w = 0;
    g_vae_rope_full_h = g_vae_rope_full_w = 0;
    if (ok)
        ok = env_is_one("H3_VAE_COMFY_BLEND")
                 ? h3_vae_stitch_rgb_tiles_comfy(tiles, &y_axis, &x_axis,
                                                 frame_count, frames, error,
                                                 error_size)
                 : h3_vae_stitch_rgb_tiles(tiles, &y_axis, &x_axis, frame_count,
                                           frames, error, error_size);
    if (ok && stats) {
        *stats = st_acc;
        stats->load_ms = shared_load_ms;
        vae_fill_attn_backend(stats);
    }
done:
    g_vae_borrowed_weights = NULL;
    g_vae_rope_origin_h = g_vae_rope_origin_w = 0;
    g_vae_rope_full_h = g_vae_rope_full_w = 0;
    h3_vae_int8_cuda_weights_free(shared_w);
    if (tiles) {
        for (i = 0; i < n_tiles; i++) free(tiles[i]);
        free(tiles);
    }
    free(tile_frames);
    h3_vae_tile_axis_free(&y_axis);
    h3_vae_tile_axis_free(&x_axis);
    if (!ok && frames) h3_vae_int8_frames_free(frames);
    return ok;
}

int h3_vae_int8_decode_cuda(h3_cuda *cuda, const h3_vae_int8_store *store,
                            const float *latent_chw, int latent_t, int latent_h,
                            int latent_w, unsigned n_blocks,
                            h3_vae_int8_frames *frames,
                            h3_vae_int8_decode_stats *stats, char *error,
                            size_t error_size) {
    h3_vae_int8_cuda_weights *w = NULL;
    double wms = 0.0;
    int ok;
    int use_metal;
    float *work_chw = NULL;
    int work_t = latent_t;
    int free_work = 0;
    int pixel_h, pixel_w, tile_pixels;
    int own_weights = 0;
    if (!cuda)
        return h3_vae_int8_decode_f32(store, latent_chw, latent_t, latent_h,
                                      latent_w, n_blocks, frames, stats, error,
                                      error_size);

    pixel_h = latent_h * H3_VAE_INT8_SPATIAL_RATIO;
    pixel_w = latent_w * H3_VAE_INT8_SPATIAL_RATIO;
    tile_pixels = h3_vae_configured_tile_pixels(pixel_h, pixel_w);
    if (!env_is_one("H3_VAE_NO_SPATIAL_TILE") &&
        (pixel_h > tile_pixels || pixel_w > tile_pixels)) {
        return h3_vae_decode_cuda_spatial(cuda, store, latent_chw, latent_t,
                                          latent_h, latent_w, n_blocks, frames,
                                          stats, error, error_size);
    }
    {
        static int logged_full;
        if (!logged_full && env_is_one("H3_VAE_NO_SPATIAL_TILE") &&
            !g_vae_borrowed_weights) {
            fprintf(stderr,
                    "vae_fullframe: %dx%d latent %dx%dx%d sdpa=%s "
                    "(Comfy VAEDecode; no spatial tiles)\n",
                    pixel_w, pixel_h, latent_t, latent_h, latent_w,
                    env_is_one("H3_VAE_SDPA_F32") ? "f32cudnn" : "f16");
            logged_full = 1;
        }
    }

    if (n_blocks > 0) {
        if (g_vae_borrowed_weights) {
            w = g_vae_borrowed_weights;
            wms = 0.0; /* counted once by spatial driver */
            own_weights = 0;
        } else if (!h3_vae_int8_cuda_weights_load(store, n_blocks, &w, &wms,
                                                  error, error_size)) {
            return 0;
        } else {
            own_weights = 1;
        }
    }

    /* Metal/Comfy temporal: pad short clips, then windows of 7 / stride 5. */
    use_metal = !env_is_one("H3_VAE_FLAT_TEMPORAL");
    if (use_metal && latent_t > 0 && latent_t < 7) {
        size_t c, t, h, ww, n;
        n = (size_t)H3_VAE_INT8_LATENT_CH * 7 * (size_t)latent_h *
            (size_t)latent_w;
        work_chw = (float *)malloc(n * sizeof(float));
        if (!work_chw) {
            fail(error, error_size, "OOM pad latent to 7");
            if (own_weights) h3_vae_int8_cuda_weights_free(w);
            return 0;
        }
        for (c = 0; c < (size_t)H3_VAE_INT8_LATENT_CH; c++)
            for (t = 0; t < 7; t++) {
                size_t src_t = t < (size_t)latent_t ? t : (size_t)latent_t - 1;
                for (h = 0; h < (size_t)latent_h; h++)
                    for (ww = 0; ww < (size_t)latent_w; ww++) {
                        size_t src =
                            ((c * (size_t)latent_t + src_t) * (size_t)latent_h +
                             h) *
                                (size_t)latent_w +
                            ww;
                        size_t dst =
                            ((c * 7 + t) * (size_t)latent_h + h) *
                                (size_t)latent_w +
                            ww;
                        work_chw[dst] = latent_chw[src];
                    }
            }
        work_t = 7;
        free_work = 1;
    } else {
        work_chw = (float *)latent_chw;
        work_t = latent_t;
    }

    if (use_metal && work_t >= 7 && (work_t - 2) % 5 == 0) {
        int chunks = (work_t - 2) / 5;
        int out_frames = chunks * 17 + 5;
        int pixel_h = latent_h * H3_VAE_INT8_SPATIAL_RATIO;
        int pixel_w = latent_w * H3_VAE_INT8_SPATIAL_RATIO;
        size_t frame_el = (size_t)pixel_h * (size_t)pixel_w * 3;
        float *final_rgb = NULL;
        float *overlap = NULL;
        float *final_b[H3_VAE_MAX_TILE_BATCH];
        float *overlap_b[H3_VAE_MAX_TILE_BATCH];
        const float *full_b[H3_VAE_MAX_TILE_BATCH];
        int chunk, nb_m, bi;
        h3_vae_int8_decode_stats st_acc;
        memset(frames, 0, sizeof(*frames));
        if (stats) memset(stats, 0, sizeof(*stats));
        memset(&st_acc, 0, sizeof(st_acc));
        memset(final_b, 0, sizeof(final_b));
        memset(overlap_b, 0, sizeof(overlap_b));
        nb_m = g_vae_batch_n;
        if (nb_m < 1) nb_m = 1;
        if (nb_m > H3_VAE_MAX_TILE_BATCH) nb_m = H3_VAE_MAX_TILE_BATCH;
        ok = 1;
        full_b[0] = work_chw;
        for (bi = 1; bi < nb_m; bi++)
            full_b[bi] = g_vae_batch_chw[bi];
        final_rgb =
            (float *)malloc((size_t)out_frames * frame_el * sizeof(float));
        overlap = (float *)malloc(5 * frame_el * sizeof(float));
        final_b[0] = final_rgb;
        overlap_b[0] = overlap;
        for (bi = 1; bi < nb_m; bi++) {
            final_b[bi] =
                (float *)malloc((size_t)out_frames * frame_el * sizeof(float));
            overlap_b[bi] = (float *)malloc(5 * frame_el * sizeof(float));
            if (!final_b[bi] || !overlap_b[bi])
                ok = 0;
        }
        if (!final_rgb || !overlap || !ok) {
            for (bi = 0; bi < nb_m; bi++) {
                free(final_b[bi]);
                free(overlap_b[bi]);
            }
            if (free_work) free(work_chw);
            fail(error, error_size, "OOM metal temporal canvas");
            if (own_weights) h3_vae_int8_cuda_weights_free(w);
            return 0;
        }
        if (env_is_one("H3_VAE_VERBOSE"))
            fprintf(stderr,
                    "vae_metal_temporal: latent_t=%d chunks=%d → frames=%d "
                    "(window=7 stride=5) batch=%d\n",
                    work_t, chunks, out_frames, nb_m);
        ok = 1;
        for (chunk = 0; chunk < chunks && ok; chunk++) {
            float *slice[H3_VAE_MAX_TILE_BATCH];
            h3_vae_int8_frames decoded;
            h3_vae_int8_decode_stats st;
            memset(slice, 0, sizeof(slice));
            memset(&decoded, 0, sizeof(decoded));
            memset(&st, 0, sizeof(st));
            for (bi = 0; bi < nb_m; bi++) {
                const float *src = full_b[bi] ? full_b[bi] : work_chw;
                slice[bi] = vae_extract_time_chw(src, work_t, latent_h,
                                                 latent_w, chunk * 5, 7, error,
                                                 error_size);
                if (!slice[bi]) {
                    ok = 0;
                    break;
                }
            }
            if (!ok) {
                for (bi = 0; bi < nb_m; bi++) free(slice[bi]);
                break;
            }
            for (bi = 1; bi < nb_m; bi++)
                g_vae_batch_chw[bi] = slice[bi];
            g_vae_metal_chunk_unpack = 1;
            ok = h3_vae_int8_decode_cuda_ex(cuda, store, w, slice[0], 7,
                                            latent_h, latent_w, n_blocks,
                                            &decoded, &st, error, error_size);
            g_vae_metal_chunk_unpack = 0;
            for (bi = 0; bi < nb_m; bi++) {
                free(slice[bi]);
                if (bi > 0) g_vae_batch_chw[bi] = full_b[bi];
            }
            if (!ok) break;
            if (decoded.frames != 22) {
                fail(error, error_size, "metal chunk unpack expected 22 frames");
                h3_vae_int8_frames_free(&decoded);
                for (bi = 1; bi < nb_m; bi++)
                    h3_vae_int8_frames_free(&g_vae_batch_frames[bi]);
                ok = 0;
                break;
            }
            for (bi = 0; bi < nb_m; bi++) {
                h3_vae_int8_frames *dec =
                    (bi == 0) ? &decoded : &g_vae_batch_frames[bi];
                float *fin = final_b[bi];
                float *ov = overlap_b[bi];
                if (!dec->rgb || dec->frames != 22) {
                    fail(error, error_size, "metal batch tile missing rgb");
                    ok = 0;
                    break;
                }
                /* Drain this chunk's D2H done-event before blending/copying
                 * dec->rgb; release the pinned pool slot afterwards. */
                if (!vae_d2h_sync_and_copy(dec, &dec->rgb,
                                           (size_t)22 * frame_el *
                                               sizeof(float))) {
                    fail(error, error_size, "metal chunk D2H consume failed");
                    ok = 0;
                    break;
                }
                if (chunk) {
                    int f;
                    for (f = 0; f < 5; f++) {
                        float t = (float)f / 5.0f;
                        float alpha = t * t * (3.0f - 2.0f * t);
                        size_t base = (size_t)f * frame_el;
                        size_t i;
                        for (i = 0; i < frame_el; i++)
                            dec->rgb[base + i] =
                                ov[base + i] * (1.0f - alpha) +
                                dec->rgb[base + i] * alpha;
                    }
                }
                memcpy(fin + (size_t)chunk * 17 * frame_el, dec->rgb,
                       17 * frame_el * sizeof(float));
                memcpy(ov, dec->rgb + 17 * frame_el, 5 * frame_el * sizeof(float));
                if (chunk == chunks - 1)
                    memcpy(fin + (size_t)chunks * 17 * frame_el, ov,
                           5 * frame_el * sizeof(float));
                h3_vae_int8_frames_free(dec);
            }
            st_acc.blocks_ms += st.blocks_ms;
            st_acc.canvas_h2d_ms += st.canvas_h2d_ms;
            st_acc.unpatch_d2h_ms += st.unpatch_d2h_ms;
            st_acc.total_ms += st.total_ms;
            st_acc.n_blocks = st.n_blocks;
            st_acc.all_finite = st.all_finite;
        }
        if (ok) {
            frames->rgb = final_b[0];
            frames->frames = out_frames;
            frames->height = pixel_h;
            frames->width = pixel_w;
            final_b[0] = NULL;
            for (bi = 1; bi < nb_m; bi++) {
                g_vae_batch_frames[bi].rgb = final_b[bi];
                g_vae_batch_frames[bi].frames = out_frames;
                g_vae_batch_frames[bi].height = pixel_h;
                g_vae_batch_frames[bi].width = pixel_w;
                final_b[bi] = NULL;
            }
            if (!g_vae_borrowed_weights)
                h3_vae_int8_deblock16_rgb(frames->rgb, frames->frames,
                                          frames->height, frames->width);
            if (stats) {
                *stats = st_acc;
                stats->load_ms = wms;
                vae_fill_attn_backend(stats);
            }
        } else {
            for (bi = 0; bi < nb_m; bi++)
                free(final_b[bi]);
        }
        for (bi = 0; bi < nb_m; bi++)
            free(overlap_b[bi]);
        if (free_work) free(work_chw);
        if (own_weights) h3_vae_int8_cuda_weights_free(w);
        return ok;
    }

    g_vae_metal_chunk_unpack = (use_metal && work_t == 7) ? 1 : 0;
    ok = h3_vae_int8_decode_cuda_ex(cuda, store, w, work_chw, work_t, latent_h,
                                    latent_w, n_blocks, frames, stats, error,
                                    error_size);
    g_vae_metal_chunk_unpack = 0;
    if (ok && frames && frames->rgb && !g_vae_borrowed_weights)
        h3_vae_int8_deblock16_rgb(frames->rgb, frames->frames, frames->height,
                                  frames->width);
    if (stats && w) stats->load_ms += wms;
    if (stats) vae_fill_attn_backend(stats);
    if (free_work) free(work_chw);
    if (own_weights) h3_vae_int8_cuda_weights_free(w);
    return ok;
}

/* P2 T4-b: run blocks over half the batched token rows on `d_ws2` using the
 * current op stream (already switched by the caller). The batch/tile dim is
 * leading (batch-stride = g_vae_attn_seq rows), so each half is an
 * independent contiguous slab and the sequential per-block chain inside one
 * half is bit-identical to the full-batch run. */
static int vae_run_layers_dev_half(h3_cuda *cuda,
                                   h3_vae_int8_cuda_weights *weights,
                                   const h3_vae_int8_store *store,
                                   float *d_slice, float *d_ws,
                                   size_t seq_half, const float *d_rope_cos,
                                   const float *d_rope_sin,
                                   const float *d_qk_ones, unsigned n_blocks,
                                   char *error, size_t error_size) {
    unsigned b;
    if (!cuda || !weights || !d_slice || !d_ws || seq_half == 0 ||
        n_blocks == 0)
        return 0;
    for (b = 0; b < n_blocks; b++) {
        int gs_run =
            h3_vae_int8_convrot_groupsize(store, weights->blocks[b].k_qkv);
        if (!run_block_cuda_w_dev(cuda, &weights->blocks[b], d_slice, d_ws,
                                  seq_half, d_rope_cos, d_rope_sin, d_qk_ones,
                                  gs_run, error, error_size))
            return 0;
    }
    return 1;
}

/* P2 T4-b: 2-stream split of the batch/tile dim (768P full-frame only).
 * Lane0 runs batch0 on the legacy stream with d_ws; lane1 runs batch1..N-1
 * on side stream 2 with a second workspace d_ws2 (same ws_need). Fork/join
 * is event-based; both halves re-join the legacy stream before returning.
 * Bit-identical: rows are contiguous independent slabs (batch-stride =
 * g_vae_attn_seq), no cross-row op exists, and per-row kernel order is
 * unchanged — only the stream assignment differs. */
static int vae_run_layers_dev_2stream(h3_cuda *cuda,
                                      h3_vae_int8_cuda_weights *weights,
                                      const h3_vae_int8_store *store,
                                      float *d_slice, float *d_ws, float *d_ws2,
                                      size_t seq, const float *d_rope_cos,
                                      const float *d_rope_sin,
                                      const float *d_qk_ones, unsigned n_blocks,
                                      char *error, size_t error_size) {
    void *op0;
    void *s2;
    void *ev_fork = NULL, *ev_l0 = NULL, *ev_l1 = NULL;
    size_t attn = g_vae_attn_seq;
    size_t rows_lane0, rows_lane1;
    int ok0 = 0, ok1 = 0, ok = 0;
    int nb = g_vae_batch_n;
    int n1;

    if (!cuda || !weights || !d_slice || !d_ws || !d_ws2 || seq == 0 ||
        n_blocks == 0 || attn == 0 || nb < 2 || (seq % attn) != 0 ||
        (size_t)nb != seq / attn)
        return 0;
    s2 = h3_cuda_stream(cuda, 2);
    if (!s2) return 0;
    op0 = h3_cuda_op_stream(cuda);
    n1 = nb / 2; /* lane1 takes the first half (contiguous slab) */
    if (n1 < 1 || n1 >= nb) return 0;
    rows_lane1 = (size_t)n1 * attn;
    rows_lane0 = seq - rows_lane1;

    if (cudaEventCreateWithFlags((cudaEvent_t *)&ev_fork,
                                 cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags((cudaEvent_t *)&ev_l0,
                                 cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags((cudaEvent_t *)&ev_l1,
                                 cudaEventDisableTiming) != cudaSuccess) {
        fail(error, error_size, "vae 2stream event create failed");
        goto out;
    }
    /* Fork: side stream waits until the legacy stream produced d_slice. */
    if (!h3_cuda_event_record(op0, ev_fork, error, error_size) ||
        !h3_cuda_stream_wait_event(s2, ev_fork, error, error_size))
        goto out;

    /* Lane1 on the side stream with the second workspace. */
    if (!h3_cuda_set_op_stream(cuda, s2, error, error_size))
        goto out;
    ok1 = vae_run_layers_dev_half(cuda, weights, store, d_slice, d_ws2,
                                  rows_lane1, d_rope_cos, d_rope_sin,
                                  d_qk_ones, n_blocks, error, error_size);
    if (ok1 && !h3_cuda_event_record(s2, ev_l1, error, error_size))
        ok1 = 0;

    /* Lane0 back on the legacy stream with the primary workspace. */
    if (!h3_cuda_set_op_stream(cuda, op0, error, error_size))
        goto out;
    ok0 = vae_run_layers_dev_half(cuda, weights, store,
                                  d_slice + rows_lane1 * H3_VAE_INT8_HIDDEN,
                                  d_ws, rows_lane0, d_rope_cos, d_rope_sin,
                                  d_qk_ones, n_blocks, error, error_size);
    if (ok0 && !h3_cuda_event_record(op0, ev_l0, error, error_size))
        ok0 = 0;

    /* Join: the consumer (proj/unpatch) must see both halves. */
    if (ok0 && ok1 &&
        !h3_cuda_stream_wait_event(op0, ev_l1, error, error_size))
        goto out;
    if (ok0 && ok1) ok = 1;

out:
    /* Always restore the caller's op stream before returning. */
    {
        char rerr[64];
        (void)h3_cuda_set_op_stream(cuda, op0, rerr, sizeof(rerr));
    }
    if (ev_fork) h3_cuda_event_destroy(ev_fork);
    if (ev_l0) h3_cuda_event_destroy(ev_l0);
    if (ev_l1) h3_cuda_event_destroy(ev_l1);
    (void)ev_l0; /* lane0 already runs on the consumer's stream */
    if (!ok && (!ok0 || !ok1) && error && error_size && !error[0])
        fail(error, error_size, "vae 2stream block run failed");
    return ok;
}

static int vae_run_layers_dev(h3_cuda *cuda, h3_vae_int8_cuda_weights *weights,
                              const h3_vae_int8_store *store, float *d_slice,
                              float *d_ws, size_t seq, const float *d_rope_cos,
                              const float *d_rope_sin, const float *d_qk_ones,
                              unsigned n_blocks, char *error,
                              size_t error_size) {
    unsigned b;
    int want_graph;
    if (!cuda || !weights || !d_slice || !d_ws || seq == 0 || n_blocks == 0)
        return 0;
    /* M4 R10: unset = 0. Do not follow H3_VAE_FLASH_SDPA (cuDNN capture aborts). */
    want_graph = h3_vae_int8_cuda_graph_wanted();
    if (want_graph && g_vae_layer_graph_ok && g_vae_layer_graph &&
        g_vae_layer_graph_seq == seq && g_vae_layer_graph_n == n_blocks) {
        if (cudaGraphLaunch(g_vae_layer_graph, 0) != cudaSuccess) {
            fail(error, error_size, "vae layer graph launch failed");
            return 0;
        }
        return 1;
    }
    /* First window stays eager so SDPA scratch / cuDNN plans exist. A failed
     * capture discards the stream work — never treat it as a completed decode. */
    if (want_graph && !g_vae_layer_graph_fail && !g_vae_layer_graph_ok &&
        !g_vae_layer_graph_warm) {
        g_vae_layer_graph_warm = 1;
        for (b = 0; b < n_blocks; b++) {
            int gs_run =
                h3_vae_int8_convrot_groupsize(store, weights->blocks[b].k_qkv);
            if (!run_block_cuda_w_dev(cuda, &weights->blocks[b], d_slice, d_ws,
                                      seq, d_rope_cos, d_rope_sin, d_qk_ones,
                                      gs_run, error, error_size))
                return 0;
        }
        return 1;
    }
    if (want_graph && !g_vae_layer_graph_fail && !g_vae_layer_graph_ok) {
        cudaGraph_t g = NULL;
        cudaError_t gst;
        h3_cuda_set_graph_capture(cuda, 1);
        gst = cudaStreamBeginCapture(0, cudaStreamCaptureModeRelaxed);
        if (gst == cudaSuccess) {
            int cok = 1;
            for (b = 0; b < n_blocks && cok; b++) {
                int gs_run = h3_vae_int8_convrot_groupsize(
                    store, weights->blocks[b].k_qkv);
                cok = run_block_cuda_w_dev(cuda, &weights->blocks[b], d_slice,
                                           d_ws, seq, d_rope_cos, d_rope_sin,
                                           d_qk_ones, gs_run, error,
                                           error_size);
            }
            gst = cudaStreamEndCapture(0, &g);
            h3_cuda_set_graph_capture(cuda, 0);
            if (cok && gst == cudaSuccess && g) {
                cudaGraphExec_t ge2 = NULL;
                if (cudaGraphInstantiate(&ge2, g, 0) == cudaSuccess) {
                    cudaGraphDestroy(g);
                    g_vae_layer_graph = ge2;
                    g_vae_layer_graph_seq = seq;
                    g_vae_layer_graph_n = n_blocks;
                    g_vae_layer_graph_ok = 1;
                    fprintf(stderr,
                            "vae_layer_graph: captured seq=%zu blocks=%u\n",
                            seq, n_blocks);
                    if (cudaGraphLaunch(g_vae_layer_graph, 0) != cudaSuccess) {
                        fail(error, error_size, "vae layer graph launch failed");
                        return 0;
                    }
                    return 1;
                }
                cudaGraphDestroy(g);
            } else if (g) {
                cudaGraphDestroy(g);
            }
            g_vae_layer_graph_fail = 1;
            fprintf(stderr,
                    "vae_graph: capture FAIL → eager cok=%d end=%s\n",
                    cok, cudaGetErrorString(gst));
            (void)cudaGetLastError();
            if (error && error_size) error[0] = '\0';
        } else {
            h3_cuda_set_graph_capture(cuda, 0);
            g_vae_layer_graph_fail = 1;
        }
    }
    for (b = 0; b < n_blocks; b++) {
        int gs_run =
            h3_vae_int8_convrot_groupsize(store, weights->blocks[b].k_qkv);
        if (!run_block_cuda_w_dev(cuda, &weights->blocks[b], d_slice, d_ws, seq,
                                  d_rope_cos, d_rope_sin, d_qk_ones, gs_run,
                                  error, error_size))
            return 0;
    }
    return 1;
}

int h3_vae_int8_decode_cuda_ex(h3_cuda *cuda, const h3_vae_int8_store *store,
                               h3_vae_int8_cuda_weights *weights,
                               const float *latent_chw, int latent_t,
                               int latent_h, int latent_w, unsigned n_blocks,
                               h3_vae_int8_frames *frames,
                               h3_vae_int8_decode_stats *stats, char *error,
                               size_t error_size) {
    /* Delegate skeleton to host decode when cuda NULL. */
    float *post_w = NULL, *post_b = NULL;
    float *embed_w = NULL, *embed_b = NULL;
    float *registers = NULL;
    float *norm_w = NULL, *norm_b = NULL;
    float *proj_w = NULL, *proj_b = NULL;
    float *patches = NULL, *seq_h = NULL, *projected = NULL, *rgb = NULL;
    float latent_mean[H3_VAE_INT8_LATENT_CH];
    float latent_std[H3_VAE_INT8_LATENT_CH];
    size_t n_patches, seq, i, c, t, h, w;
    unsigned b;
    clock_t t0, t1, tall0;
    int ok = 0;
    int pixel_h, pixel_w, out_frames;
    int gpu_proj_done = 0;
    int gpu_unpatch_done = 0;
    int gpu_embed_done = 0;
    int heads_cached = 0;
    int host_pooled = 0;
    int flash_sdpa = 0;

    if (!cuda)
        return h3_vae_int8_decode_f32(store, latent_chw, latent_t, latent_h,
                                      latent_w, n_blocks, frames, stats, error,
                                      error_size);

    if (stats) memset(stats, 0, sizeof(*stats));
    if (!store || !latent_chw || !frames || latent_t < 1 || latent_h < 1 ||
        latent_w < 1) {
        fail(error, error_size, "invalid decode_cuda args");
        return 0;
    }
    if (n_blocks > H3_VAE_INT8_N_BLOCKS) n_blocks = H3_VAE_INT8_N_BLOCKS;
    memset(frames, 0, sizeof(*frames));
    /* Fail-safe: any read of fr->rgb in the call-family below goes through a
     * per-slot done-event wait, but callers that bypass those helpers could
     * still see a half-written buffer. Drain every outstanding D2H here; by
     * this point every tile's DMA has long since overlapped compute, so the
     * waits are no-ops in the steady state. */
    {
        int si;
        for (si = 0; si < VAE_D2H_SLOTS; si++) {
            if (g_vae_d2h_slots[si].event) {
                if (cudaEventSynchronize((cudaEvent_t)
                        g_vae_d2h_slots[si].event) != cudaSuccess) {
                    fail(error, error_size, "vae unpatch D2H drain failed");
                    goto done;
                }
                cudaEventDestroy((cudaEvent_t)g_vae_d2h_slots[si].event);
                g_vae_d2h_slots[si].rgb = NULL;
                g_vae_d2h_slots[si].event = NULL;
            }
        }
    }
    flash_sdpa = env_is_one("H3_VAE_FLASH_SDPA");
    if (flash_sdpa)
        h3_cuda_sdpa_set_force_flash(1);
    n_patches = (size_t)latent_t * (size_t)latent_h * (size_t)latent_w;
    {
        int nb = g_vae_batch_n;
        if (nb < 1) nb = 1;
        if (nb > H3_VAE_MAX_TILE_BATCH) nb = H3_VAE_MAX_TILE_BATCH;
        g_vae_batch_n = nb;
        g_vae_attn_seq = n_patches + (size_t)H3_VAE_INT8_REGISTERS + 1;
        g_vae_batch_chw[0] = latent_chw;
    }
    seq = g_vae_attn_seq * (size_t)g_vae_batch_n;
    pixel_h = latent_h * H3_VAE_INT8_SPATIAL_RATIO;
    pixel_w = latent_w * H3_VAE_INT8_SPATIAL_RATIO;
    /* Metal first-chunk unpack: 7 tokens → 22 frames (skip gap 20..22). */
    out_frames = (g_vae_metal_chunk_unpack && latent_t == 7) ? 22
                                                            : (latent_t * 4);

    tall0 = clock();
    t0 = clock();
    if (weights && ensure_vae_decode_heads(store, weights, error, error_size)) {
        post_w = weights->h_post_w;
        post_b = weights->h_post_b;
        embed_w = weights->h_embed_w;
        embed_b = weights->h_embed_b;
        registers = weights->h_registers;
        norm_w = weights->h_norm_w;
        norm_b = weights->h_norm_b;
        proj_w = weights->h_proj_w;
        proj_b = weights->h_proj_b;
        heads_cached = 1;
    } else {
        post_w = load_f32(store, "post_quant_conv.weight",
                          (size_t)H3_VAE_INT8_LATENT_CH * H3_VAE_INT8_LATENT_CH,
                          error, error_size);
        post_b = load_f32(store, "post_quant_conv.bias", H3_VAE_INT8_LATENT_CH,
                           error, error_size);
        embed_w = load_f32(store, "decoder.x_embedder.weight",
                            (size_t)H3_VAE_INT8_HIDDEN * H3_VAE_INT8_LATENT_CH,
                            error, error_size);
        embed_b = load_f32(store, "decoder.x_embedder.bias", H3_VAE_INT8_HIDDEN,
                            error, error_size);
        registers = load_f32(store, "decoder.register_tokens",
                              (size_t)H3_VAE_INT8_REGISTERS * H3_VAE_INT8_HIDDEN,
                              error, error_size);
        norm_w = load_f32(store, "decoder.norm_out.weight", H3_VAE_INT8_HIDDEN,
                          error, error_size);
        norm_b = load_f32(store, "decoder.norm_out.bias", H3_VAE_INT8_HIDDEN,
                           error, error_size);
        proj_w = load_f32(store, "decoder.proj_out.weight",
                           (size_t)H3_VAE_INT8_OUTPUT_PATCH * H3_VAE_INT8_HIDDEN,
                           error, error_size);
        proj_b = load_f32(store, "decoder.proj_out.bias",
                           H3_VAE_INT8_OUTPUT_PATCH, error, error_size);
    }
    if (vae_host_grow((void **)&g_vae_scratch.h_patches,
                      &g_vae_scratch.cap_h_patches,
                      n_patches * (size_t)g_vae_batch_n *
                          H3_VAE_INT8_LATENT_CH * sizeof(float)) &&
        vae_host_grow((void **)&g_vae_scratch.h_seq, &g_vae_scratch.cap_h_seq,
                      seq * H3_VAE_INT8_HIDDEN * sizeof(float)) &&
        vae_host_grow((void **)&g_vae_scratch.h_projected,
                      &g_vae_scratch.cap_h_proj,
                      n_patches * (size_t)g_vae_batch_n *
                          H3_VAE_INT8_OUTPUT_PATCH * sizeof(float))) {
        patches = g_vae_scratch.h_patches;
        seq_h = g_vae_scratch.h_seq;
        projected = g_vae_scratch.h_projected;
        host_pooled = 1;
    } else {
        patches =
            (float *)malloc(n_patches * (size_t)g_vae_batch_n *
                            H3_VAE_INT8_LATENT_CH * sizeof(float));
        seq_h = (float *)calloc(seq * H3_VAE_INT8_HIDDEN, sizeof(float));
        projected = (float *)malloc(n_patches * (size_t)g_vae_batch_n *
                                    H3_VAE_INT8_OUTPUT_PATCH *
                                    sizeof(float));
    }
    if (!post_w || !post_b || !embed_w || !embed_b || !registers || !norm_w ||
        !norm_b || !proj_w || !proj_b || !patches || !seq_h || !projected) {
        fail(error, error_size, "OOM decode_cuda heads");
        goto done;
    }
    if (!h3_vae_int8_load_latent_stats(store, latent_mean, latent_std, error,
                                       error_size))
        goto done;
    t1 = clock();
    if (stats) stats->load_ms = elapsed_ms(t0, t1);

    i = 0;
    {
        int b;
        for (b = 0; b < g_vae_batch_n; b++) {
            const float *src_chw =
                (b == 0 || !g_vae_batch_chw[b]) ? latent_chw
                                                : g_vae_batch_chw[b];
            for (t = 0; t < (size_t)latent_t; t++)
                for (h = 0; h < (size_t)latent_h; h++)
                    for (w = 0; w < (size_t)latent_w; w++, i++) {
                        float x[H3_VAE_INT8_LATENT_CH];
                        float y[H3_VAE_INT8_LATENT_CH];
                        for (c = 0; c < H3_VAE_INT8_LATENT_CH; c++) {
                            size_t src =
                                ((c * (size_t)latent_t + t) *
                                     (size_t)latent_h +
                                 h) *
                                    (size_t)latent_w +
                                w;
                            x[c] = src_chw[src];
                        }
                        h3_vae_int8_denorm_vec_f32(x, latent_mean, latent_std,
                                                   H3_VAE_INT8_LATENT_CH);
                        for (c = 0; c < H3_VAE_INT8_LATENT_CH; c++) {
                            const float *wr =
                                post_w + c * H3_VAE_INT8_LATENT_CH;
                            float sum = post_b[c];
                            size_t k;
                            for (k = 0; k < H3_VAE_INT8_LATENT_CH; k++)
                                sum += wr[k] * x[k];
                            y[c] = sum;
                        }
                        memcpy(patches + i * H3_VAE_INT8_LATENT_CH, y,
                               H3_VAE_INT8_LATENT_CH * sizeof(float));
                    }
        }
    }

    t0 = clock();
    if (n_blocks > 0) {
        /* Temporal strategy:
         * - Default (auto): chunk=1 for large spat (480p+); cross-frame merge
         *   makes SDPA O((C·spat)²) and was slower on 4090 @ 864×480.
         * - Small spat (≤512): auto may use chunk=2.
         * - H3_VAE_TEMPORAL_CHUNK=N forces merge (A/B / low-res only).
         * Device path keeps the full T·H·W canvas on GPU (one H2D/D2H) and
         * D2D-copies each frame into the working slice. */
        const size_t spat = (size_t)latent_h * (size_t)latent_w;
        const size_t hidden = H3_VAE_INT8_HIDDEN;
        const size_t inner =
            (size_t)H3_VAE_INT8_HEADS * H3_VAE_INT8_HEAD_DIM;
        const size_t reg_tail =
            (size_t)H3_VAE_INT8_REGISTERS + 1; /* registers + zero token */
        int chunk_cfg = env_int_default("H3_VAE_TEMPORAL_CHUNK", 0);
        int chunk = 1;
        int use_dev_slice = 0;
        float *slice = NULL;
        float *d_slice = NULL, *d_ws = NULL, *d_canvas = NULL;
        float *d_registers = NULL;
        float *rope_cos = NULL, *rope_sin = NULL;
        float *d_rope_cos = NULL, *d_rope_sin = NULL, *d_qk_ones = NULL;
        float ones_host[H3_VAE_INT8_HEAD_DIM];
        size_t chunk_seq = 0, ws_need = 0, rope_cap = 0;
        size_t t0_frame;
        int use_dev_canvas = 0;
        int want_2s = 0;
        size_t od;
        for (od = 0; od < H3_VAE_INT8_HEAD_DIM; od++) ones_host[od] = 1.0f;

        if (chunk_cfg > 0) {
            chunk = chunk_cfg;
            if (chunk > latent_t) chunk = latent_t;
            if (chunk < 1) chunk = 1;
        } else if (g_vae_metal_chunk_unpack) {
            /* One Metal/Comfy window: attend across all tokens in the clip. */
            chunk = latent_t;
        } else {
            /* Auto: merge only when spat is small enough that C² attn is cheap. */
            if (spat <= 512 && latent_t >= 2)
                chunk = 2;
            else
                chunk = 1;
        }

        try_alloc_chunk:
        chunk_seq = (size_t)chunk * spat + reg_tail;
        chunk_seq *= (size_t)g_vae_batch_n;
        if (!vae_host_grow((void **)&g_vae_scratch.h_slice,
                           &g_vae_scratch.cap_h_slice,
                           chunk_seq * hidden * sizeof(float))) {
            fail(error, error_size, "OOM vae temporal chunk");
            goto done;
        }
        slice = g_vae_scratch.h_slice;
        use_dev_slice = 0;
        use_dev_canvas = 0;
        d_slice = d_ws = d_canvas = d_registers = NULL;
        if (weights && weights->device_resident && weights->n_blocks > 0 &&
            weights->blocks[0].d_w_qkv && weights->blocks[0].d_norm1) {
            size_t n_qkv = weights->blocks[0].n_qkv;
            size_t n_w1 = weights->blocks[0].n_w1;
            size_t n_out = weights->blocks[0].n_out;
            size_t n_w2 = weights->blocks[0].n_w2;
            size_t k_qkv = weights->blocks[0].k_qkv;
            size_t k_w1 = weights->blocks[0].k_w1;
            size_t max_mk = chunk_seq * (k_qkv > k_w1 ? k_qkv : k_w1);
            size_t max_mn = chunk_seq * (n_qkv > n_w1 ? n_qkv : n_w1);
            unsigned bi;
            int hit;
            if (chunk_seq * n_out > max_mn) max_mn = chunk_seq * n_out;
            if (chunk_seq * n_w2 > max_mn) max_mn = chunk_seq * n_w2;
            /* Load is F16→F32 on GPU (P0). Decode still recasts to F16 TC
             * like cafe p2 (native_float_w=0 host dequant). Skip only if
             * H3_VAE_F32_GEMM=1; otherwise F32 GEMM is ~3× slower on 4090. */
            if (!env_flag_is_one("H3_VAE_F32_GEMM")) {
                for (bi = 0; bi < weights->n_blocks && bi < n_blocks; bi++)
                    (void)ensure_block_f16_weights(cuda, &weights->blocks[bi],
                                                   error, error_size);
            }
            ws_need =
                chunk_seq *
                    (hidden * 3 + n_qkv + inner * 3 + n_w1 + H3_VAE_INT8_FFN +
                     8192) *
                    sizeof(float) +
                (max_mk + max_mn) * 2;
            hit = g_vae_scratch.d_ws && g_vae_scratch.cap_d_ws >= ws_need &&
                  g_vae_scratch.d_slice &&
                  g_vae_scratch.cap_d_slice >=
                      chunk_seq * hidden * sizeof(float);
            if (vae_cuda_grow((void **)&g_vae_scratch.d_slice,
                              &g_vae_scratch.cap_d_slice,
                              chunk_seq * hidden * sizeof(float)) &&
                vae_cuda_grow((void **)&g_vae_scratch.d_ws,
                              &g_vae_scratch.cap_d_ws, ws_need)) {
                d_slice = g_vae_scratch.d_slice;
                d_ws = g_vae_scratch.d_ws;
                use_dev_slice = 1;
                if (hit)
                    g_vae_scratch.reuse_hits++;
                if (g_vae_scratch.reuse_hits == 1 ||
                    (!hit && g_vae_scratch.reuse_hits == 0))
                    fprintf(stderr,
                            "vae_scratch: %s chunk_seq=%zu ws_mib=%.1f\n",
                            hit ? "hit" : "fill", chunk_seq,
                            (double)ws_need / (1024.0 * 1024.0));
                /* P2 T4-b: second workspace so lane1 never aliases lane0.
                 * 768P full-frame only (batched 256px tiles): batch>=2,
                 * whole clip in one chunk, CUDA-graph off (capture forbids
                 * side-stream forks). A failed grow just disables 2-stream;
                 * the lane0 path is untouched. */
                want_2s = vae_blocks_2stream_wanted() &&
                          g_vae_batch_n >= 2 && chunk >= latent_t &&
                          !h3_vae_int8_cuda_graph_wanted();
                if (want_2s) {
                    if (vae_cuda_grow((void **)&g_vae_scratch.d_ws2,
                                      &g_vae_scratch.cap_d_ws2, ws_need)) {
                        static int logged_2s;
                        if (!logged_2s) {
                            fprintf(stderr,
                                    "vae_blocks_2stream: batch=%d ws2_mib=%.1f "
                                    "(768P, H3_VAE_BLOCKS_2STREAM=1)\n",
                                    g_vae_batch_n,
                                    (double)ws_need / (1024.0 * 1024.0));
                            logged_2s = 1;
                        }
                    } else {
                        fprintf(stderr,
                                "vae_blocks_2stream: ws2 alloc failed, "
                                "fallback 1-stream\n");
                        want_2s = 0;
                    }
                }
            } else if (chunk_cfg == 0 && chunk > 1) {
                chunk = 1;
                goto try_alloc_chunk;
            }
            if (use_dev_slice && chunk >= latent_t && weights &&
                ensure_vae_embed_dev(weights, embed_w, embed_b, registers) &&
                vae_cuda_grow((void **)&g_vae_scratch.d_pat,
                              &g_vae_scratch.cap_d_pat,
                              n_patches * (size_t)g_vae_batch_n *
                                  (size_t)H3_VAE_INT8_LATENT_CH *
                                  sizeof(float)) &&
                cudaMemcpy(g_vae_scratch.d_pat, patches,
                           n_patches * (size_t)g_vae_batch_n *
                               (size_t)H3_VAE_INT8_LATENT_CH * sizeof(float),
                           cudaMemcpyHostToDevice) == cudaSuccess) {
                int eb;
                int eok = 1;
                for (eb = 0; eb < g_vae_batch_n && eok; eb++) {
                    float *d_tok = d_slice + (size_t)eb * g_vae_attn_seq * hidden;
                    const float *d_pat_b =
                        g_vae_scratch.d_pat +
                        (size_t)eb * n_patches * (size_t)H3_VAE_INT8_LATENT_CH;
                    eok = h3_cuda_gemm_f32_dd(
                              cuda, d_pat_b, weights->d_embed_w, n_patches,
                              H3_VAE_INT8_HIDDEN, H3_VAE_INT8_LATENT_CH, d_tok,
                              NULL, error, error_size) &&
                          h3_cuda_bias_add_f32_dev(
                              cuda, d_tok, weights->d_embed_b, n_patches,
                              H3_VAE_INT8_HIDDEN, error, error_size) &&
                          cudaMemcpy(d_tok + n_patches * hidden,
                                     weights->d_registers,
                                     (size_t)H3_VAE_INT8_REGISTERS * hidden *
                                         sizeof(float),
                                     cudaMemcpyDeviceToDevice) == cudaSuccess &&
                          cudaMemset(d_tok + (n_patches +
                                              (size_t)H3_VAE_INT8_REGISTERS) *
                                         hidden,
                                     0, hidden * sizeof(float)) == cudaSuccess;
                }
                if (eok) {
                    gpu_embed_done = 1;
                    {
                        static int logged_emb;
                        if (!logged_emb) {
                            fprintf(stderr,
                                    "vae_embed: device n_patches=%zu batch=%d "
                                    "(skip host gemm_f32 malloc)\n",
                                    n_patches, g_vae_batch_n);
                            logged_emb = 1;
                        }
                    }
                }
            }
            /* Full temporal canvas on GPU: one H2D of patches, one D2H. */
            if (use_dev_slice && chunk == 1 && !gpu_embed_done &&
                vae_cuda_grow((void **)&g_vae_scratch.d_canvas,
                              &g_vae_scratch.cap_d_canvas,
                              n_patches * hidden * sizeof(float)) &&
                vae_cuda_grow((void **)&g_vae_scratch.d_registers,
                              &g_vae_scratch.cap_d_reg,
                              (size_t)H3_VAE_INT8_REGISTERS * hidden *
                                  sizeof(float))) {
                clock_t tc0 = clock();
                d_canvas = g_vae_scratch.d_canvas;
                d_registers = g_vae_scratch.d_registers;
                if (cudaMemcpy(d_canvas, seq_h, n_patches * hidden * sizeof(float),
                               cudaMemcpyHostToDevice) == cudaSuccess &&
                    cudaMemcpy(d_registers, registers,
                               (size_t)H3_VAE_INT8_REGISTERS * hidden *
                                   sizeof(float),
                               cudaMemcpyHostToDevice) == cudaSuccess) {
                    use_dev_canvas = 1;
                    if (stats) stats->canvas_h2d_ms = elapsed_ms(tc0, clock());
                } else {
                    d_canvas = NULL;
                    d_registers = NULL;
                }
            }
        }

        if (!gpu_embed_done) {
            if (!h3_cuda_gemm_f32(cuda, patches, embed_w, n_patches,
                                  H3_VAE_INT8_HIDDEN, H3_VAE_INT8_LATENT_CH,
                                  seq_h, NULL, error, error_size))
                goto done;
            for (i = 0; i < n_patches; i++)
                for (c = 0; c < H3_VAE_INT8_HIDDEN; c++)
                    seq_h[i * H3_VAE_INT8_HIDDEN + c] += embed_b[c];
            memcpy(seq_h + n_patches * H3_VAE_INT8_HIDDEN, registers,
                   (size_t)H3_VAE_INT8_REGISTERS * H3_VAE_INT8_HIDDEN *
                       sizeof(float));
        }

        if (env_is_one("H3_VAE_VERBOSE")) {
            fprintf(stderr,
                    "vae_temporal_chunk=%d seq=%zu device=%d canvas=%d\n", chunk,
                    chunk_seq, use_dev_slice, use_dev_canvas);
        }

        rope_cap = chunk_seq;
        if (!vae_host_grow((void **)&g_vae_scratch.h_rope_cos,
                           &g_vae_scratch.cap_h_rope,
                           rope_cap * H3_VAE_INT8_ROPE_HALF * sizeof(float)) ||
            !vae_host_grow((void **)&g_vae_scratch.h_rope_sin,
                           &g_vae_scratch.cap_h_rope,
                           rope_cap * H3_VAE_INT8_ROPE_HALF * sizeof(float))) {
            fail(error, error_size, "OOM vae rope");
            goto done;
        }
        rope_cos = g_vae_scratch.h_rope_cos;
        rope_sin = g_vae_scratch.h_rope_sin;
        if (use_dev_slice || use_dev_canvas) {
            size_t rope_bytes =
                rope_cap * H3_VAE_INT8_ROPE_HALF * sizeof(float);
            if (!vae_cuda_grow((void **)&g_vae_scratch.d_rope_cos,
                               &g_vae_scratch.cap_d_rope, rope_bytes) ||
                !vae_cuda_grow((void **)&g_vae_scratch.d_rope_sin,
                               &g_vae_scratch.cap_d_rope, rope_bytes) ||
                !vae_cuda_grow((void **)&g_vae_scratch.d_qk_ones,
                               &g_vae_scratch.cap_d_qk,
                               H3_VAE_INT8_HEAD_DIM * sizeof(float)) ||
                cudaMemcpy(g_vae_scratch.d_qk_ones, ones_host,
                           H3_VAE_INT8_HEAD_DIM * sizeof(float),
                           cudaMemcpyHostToDevice) != cudaSuccess) {
                fail(error, error_size, "OOM vae rope device");
                goto done;
            }
            d_rope_cos = g_vae_scratch.d_rope_cos;
            d_rope_sin = g_vae_scratch.d_rope_sin;
            d_qk_ones = g_vae_scratch.d_qk_ones;
        }

        g_vae_d2h_cuda = cuda;
        for (t0_frame = 0; t0_frame < (size_t)latent_t;
             t0_frame += (size_t)chunk) {
            /* Cross-tile pipeline overlap point: before enqueueing this
             * tile's GPU compute (rope H2D / blocks), drain the previous
             * tile's D2H on its side stream so the DMA engine is never the
             * host's critical path. This is a host-side wait only; nothing
             * here reorders the op stream. */
            if (g_vae_d2h_armed && t0_frame > 0 &&
                !vae_d2h_wait(g_vae_scratch.d_rgb)) {
                fail(error, error_size, "vae unpatch D2H event wait failed");
                goto done;
            }
            size_t c_here = (size_t)chunk;
            size_t ci;
            size_t seq_here;
            if (t0_frame + c_here > (size_t)latent_t)
                c_here = (size_t)latent_t - t0_frame;
            seq_here = c_here * spat + reg_tail;
            if (!h3_vae_int8_prepare_rope_ex(
                    (int)c_here, latent_h, latent_w, g_vae_rope_origin_h,
                    g_vae_rope_origin_w,
                    g_vae_rope_full_h > 0 ? g_vae_rope_full_h : latent_h,
                    g_vae_rope_full_w > 0 ? g_vae_rope_full_w : latent_w,
                    c_here * spat, seq_here, rope_cos, rope_sin, error,
                    error_size)) {
                goto done;
            }
            if (d_rope_cos &&
                (cudaMemcpy(d_rope_cos, rope_cos,
                            seq_here * H3_VAE_INT8_ROPE_HALF * sizeof(float),
                            cudaMemcpyHostToDevice) != cudaSuccess ||
                 cudaMemcpy(d_rope_sin, rope_sin,
                            seq_here * H3_VAE_INT8_ROPE_HALF * sizeof(float),
                            cudaMemcpyHostToDevice) != cudaSuccess)) {
                fail(error, error_size, "vae rope H2D failed");
                goto done;
            }

            if (gpu_embed_done && use_dev_slice && chunk >= latent_t &&
                c_here == (size_t)latent_t) {
                g_vae_attn_seq = seq_here;
                if (want_2s && g_vae_scratch.d_ws2) {
                    if (!vae_run_layers_dev_2stream(
                            cuda, weights, store, d_slice, d_ws,
                            g_vae_scratch.d_ws2,
                            seq_here * (size_t)g_vae_batch_n, d_rope_cos,
                            d_rope_sin, d_qk_ones, n_blocks, error,
                            error_size))
                        goto done;
                } else if (!vae_run_layers_dev(
                               cuda, weights, store, d_slice, d_ws,
                               seq_here * (size_t)g_vae_batch_n, d_rope_cos,
                               d_rope_sin, d_qk_ones, n_blocks, error,
                               error_size))
                    goto done;
            } else if (use_dev_canvas && chunk == 1 && c_here == 1) {
                /* D2D pack: spat from canvas + registers + zero token. */
                if (cudaMemcpy(d_slice, d_canvas + t0_frame * spat * hidden,
                               spat * hidden * sizeof(float),
                               cudaMemcpyDeviceToDevice) != cudaSuccess ||
                    cudaMemcpy(d_slice + spat * hidden, d_registers,
                               (size_t)H3_VAE_INT8_REGISTERS * hidden *
                                   sizeof(float),
                               cudaMemcpyDeviceToDevice) != cudaSuccess ||
                    cudaMemset(d_slice + (spat + H3_VAE_INT8_REGISTERS) * hidden,
                               0, hidden * sizeof(float)) != cudaSuccess) {
                    fail(error, error_size, "vae canvas pack failed");
                    goto done;
                }
                if (!vae_run_layers_dev(cuda, weights, store, d_slice, d_ws,
                                        seq_here, d_rope_cos, d_rope_sin,
                                        d_qk_ones, n_blocks, error,
                                        error_size))
                    goto done;
                if (cudaMemcpy(d_canvas + t0_frame * spat * hidden, d_slice,
                               spat * hidden * sizeof(float),
                               cudaMemcpyDeviceToDevice) != cudaSuccess) {
                    fail(error, error_size, "vae canvas writeback failed");
                    goto done;
                }
                continue;
            } else {
            for (ci = 0; ci < c_here; ci++) {
                float *src =
                    seq_h + (t0_frame + ci) * spat * hidden;
                memcpy(slice + ci * spat * hidden, src,
                       spat * hidden * sizeof(float));
            }
            memcpy(slice + c_here * spat * hidden, registers,
                   (size_t)H3_VAE_INT8_REGISTERS * hidden * sizeof(float));
            memset(slice + (c_here * spat + H3_VAE_INT8_REGISTERS) * hidden, 0,
                   hidden * sizeof(float));

            if (use_dev_slice) {
                if (cudaMemcpy(d_slice, slice,
                               seq_here * hidden * sizeof(float),
                               cudaMemcpyHostToDevice) != cudaSuccess) {
                    fail(error, error_size, "vae chunk H2D failed");
                    goto done;
                }
                if (!vae_run_layers_dev(cuda, weights, store, d_slice, d_ws,
                                        seq_here, d_rope_cos, d_rope_sin,
                                        d_qk_ones, n_blocks, error,
                                        error_size))
                    goto done;
                if (!(chunk >= latent_t && weights && d_ws) &&
                    cudaMemcpy(slice, d_slice,
                               seq_here * hidden * sizeof(float),
                               cudaMemcpyDeviceToHost) != cudaSuccess) {
                    fail(error, error_size, "vae chunk D2H failed");
                    goto done;
                }
            } else {
                for (b = 0; b < n_blocks; b++) {
                    int bok;
                    if (weights && b < weights->n_blocks)
                        bok = run_block_cuda_w(cuda, &weights->blocks[b], slice,
                                               seq_here, rope_cos, rope_sin,
                                               d_rope_cos, d_rope_sin, d_qk_ones,
                                               error, error_size);
                    else
                        bok = run_block_cuda(cuda, store, b, slice, seq_here,
                                             rope_cos, rope_sin, error,
                                             error_size);
                    if (!bok) {
                        goto done;
                    }
                }
            }
            for (ci = 0; ci < c_here; ci++) {
                float *dst =
                    seq_h + (t0_frame + ci) * spat * hidden;
                memcpy(dst, slice + ci * spat * hidden,
                       spat * hidden * sizeof(float));
            }
            }
        }
    if (!gpu_proj_done && weights && d_ws &&
        ((use_dev_canvas && d_canvas) ||
         (use_dev_slice && d_slice && chunk >= latent_t)) &&
        ensure_vae_head_dev(weights, norm_w, norm_b, proj_w, proj_b)) {
        float *d_src0 = (use_dev_canvas && d_canvas) ? d_canvas : d_slice;
        float *d_ln = d_ws;
        float *d_pj =
            d_ws + n_patches * (size_t)H3_VAE_INT8_HIDDEN;
        clock_t tp0 = clock();
        int pb;
        int pok = 1;
        int metal_22 =
            (g_vae_metal_chunk_unpack && out_frames == 22) ? 1 : 0;
        size_t rgb_n = (size_t)out_frames * (size_t)pixel_h *
                       (size_t)pixel_w * 3u;
        const size_t rgb_bytes = rgb_n * sizeof(float);
        const int want_pin = vae_unpatch_pinned();
        const int want_async = want_pin && vae_unpatch_async();
        int pipe_on = 0;
        if (want_async) {
            int li;
            if (!g_vae_d2h_side)
                g_vae_d2h_side = h3_cuda_stream(cuda, 0);
            if (!g_vae_d2h_side2)
                g_vae_d2h_side2 = h3_cuda_stream(cuda, 1);
            if (g_vae_d2h_side && g_vae_d2h_side2) {
                pipe_on = 1;
                for (li = 0; li < VAE_D2H_SLOTS; li++) {
                    if (!g_vae_d_rgb_lanes[li] &&
                        cudaMalloc((void **)&g_vae_d_rgb_lanes[li],
                                   rgb_bytes) != cudaSuccess) {
                        g_vae_d_rgb_lanes[li] = NULL;
                        break;
                    }
                }
                if ((size_t)li > g_vae_d_rgb_lane_cap)
                    g_vae_d_rgb_lane_cap = (size_t)li;
            }
        }
            for (pb = 0; pb < g_vae_batch_n && pok; pb++) {
                float *d_src =
                    d_src0 + (size_t)pb * g_vae_attn_seq *
                                 (size_t)H3_VAE_INT8_HIDDEN;
                h3_vae_int8_frames *fr =
                    (pb == 0) ? frames : &g_vae_batch_frames[pb];
                float *d_rgb_dst = NULL;
                int lane = -1;
                if (pipe_on && pb < (int)g_vae_d_rgb_lane_cap &&
                    g_vae_d_rgb_lanes[pb])
                    lane = pb;
                pok = h3_cuda_layer_norm_bias_f32_dev(
                          cuda, d_src, weights->d_norm_out_w,
                          weights->d_norm_out_b, n_patches, H3_VAE_INT8_HIDDEN,
                          1e-5f, d_ln, error, error_size) &&
                      h3_cuda_gemm_f32_dd(cuda, d_ln, weights->d_proj_out_w,
                                          n_patches, H3_VAE_INT8_OUTPUT_PATCH,
                                          H3_VAE_INT8_HIDDEN, d_pj, NULL, error,
                                          error_size) &&
                      h3_cuda_bias_add_f32_dev(
                          cuda, d_pj, weights->d_proj_out_b, n_patches,
                          H3_VAE_INT8_OUTPUT_PATCH, error, error_size);
                if (!pok) break;
                if (env_flag_is_one("H3_VAE_HOST_UNPATCH")) {
                    pok = 0;
                    break;
                }
                if (lane >= 0)
                    d_rgb_dst = g_vae_d_rgb_lanes[lane];
                else if (!vae_cuda_grow((void **)&g_vae_scratch.d_rgb,
                                        &g_vae_scratch.cap_d_rgb, rgb_bytes)) {
                    pok = 0;
                    d_rgb_dst = NULL;
                } else
                    d_rgb_dst = g_vae_scratch.d_rgb;
                if (pok &&
                    !h3_cuda_vae_unpatch_rgb(
                        cuda, d_pj, d_rgb_dst, out_frames, pixel_h, pixel_w,
                        latent_t, latent_h, latent_w, metal_22, error,
                        error_size)) {
                    pok = 0;
                }
                if (!pok) break;
                {
                    int pinned = 0;
                    if (lane >= 0) {
                        rgb = vae_rgb_arena_acquire(rgb_bytes);
                        if (rgb) pinned = 2;
                    }
                    if (!rgb) {
                        if (want_pin)
                            rgb = (float *)h3_cuda_host_alloc(rgb_bytes,
                                                              &pinned);
                        else
                            rgb = (float *)malloc(rgb_bytes);
                        pinned = pinned ? 1 : 0;
                    }
                    if (!rgb) {
                        pok = 0;
                        break;
                    }
                    if (lane >= 0 && pinned == 2) {
                        /* True cross-tile pipeline: record the op-stream
                         * tail, launch D2H on a side stream, and return to
                         * the host WITHOUT synchronizing. The consumer waits
                         * on this slot's done-event before reading fr->rgb,
                         * so DMA(tile N) overlaps GPU compute(tile N+1). */
                        /* op stream may be NULL on paths where the VAE runs
                         * kernels on the default stream (metal temporal); the
                         * unpatch kernel itself uses h3_cuda_op_st() which
                         * falls back to stream 0. Record the chain on stream
                         * 0 in that case so the event still orders after the
                         * kernel. */
                        void *op = h3_cuda_op_stream(cuda);
                        void *side =
                            (lane & 1) ? g_vae_d2h_side2 : g_vae_d2h_side;
                        cudaEvent_t done_ev = NULL;
                        if (!op) op = (void *)0; /* default stream */
                        if (cudaEventCreateWithFlags(
                                &done_ev, cudaEventDisableTiming) !=
                            cudaSuccess)
                            done_ev = NULL;
                        if (done_ev && side &&
                            h3_cuda_event_record(op, (void *)done_ev, error,
                                                 error_size) &&
                            h3_cuda_stream_wait_event(side, (void *)done_ev,
                                                      error, error_size) &&
                            cudaMemcpyAsync(rgb, d_rgb_dst, rgb_bytes,
                                            cudaMemcpyDeviceToHost,
                                            (cudaStream_t)side) ==
                                cudaSuccess &&
                            cudaEventRecord(done_ev, (cudaStream_t)side) ==
                                cudaSuccess &&
                            vae_d2h_event_put(rgb, done_ev)) {
                            fr->rgb = rgb;
                            fr->rgb_pinned = 2;
                        } else {
                            if (done_ev) {
                                cudaEventSynchronize(done_ev);
                                cudaEventDestroy(done_ev);
                            }
                            vae_rgb_arena_release(rgb);
                            rgb = NULL;
                            pok = 0;
                            break;
                        }
                    } else if (cudaMemcpy(rgb, d_rgb_dst, rgb_bytes,
                                          cudaMemcpyDeviceToHost) ==
                               cudaSuccess) {
                        fr->rgb = rgb;
                        fr->rgb_pinned = pinned;
                    } else {
                        if (pinned == 2)
                            vae_rgb_arena_release(rgb);
                        else if (pinned)
                            h3_cuda_host_free(rgb, 1);
                        else
                            free(rgb);
                        rgb = NULL;
                        pok = 0;
                        break;
                    }
                }
                fr->frames = out_frames;
                fr->height = pixel_h;
                fr->width = pixel_w;
                rgb = NULL;
            }
            if (pok) {
                gpu_proj_done = 1;
                gpu_unpatch_done = 1;
                if (stats)
                    stats->unpatch_d2h_ms = elapsed_ms(tp0, clock());
                {
                    static int logged;
                    if (!logged) {
                        fprintf(stderr,
                                "vae_unpatch: gpu frames=%d %dx%d batch=%d "
                                "(skip host RGB loop)\n",
                                out_frames, pixel_w, pixel_h, g_vae_batch_n);
                        logged = 1;
                    }
                }
            } else {
                int cb;
                for (cb = 0; cb < g_vae_batch_n; cb++) {
                    h3_vae_int8_frames *fr =
                        (cb == 0) ? frames : &g_vae_batch_frames[cb];
                    h3_vae_int8_frames_free(fr);
                }
                if (error && error_size) error[0] = '\0';
            }
        }
        if (!gpu_proj_done && use_dev_slice && d_slice && chunk >= latent_t) {
            (void)cudaMemcpy(seq_h, d_slice,
                             n_patches * hidden * sizeof(float),
                             cudaMemcpyDeviceToHost);
        }
        if (use_dev_canvas && !gpu_proj_done) {
            clock_t td0 = clock();
            if (cudaMemcpy(seq_h, d_canvas, n_patches * hidden * sizeof(float),
                           cudaMemcpyDeviceToHost) != cudaSuccess) {
                fail(error, error_size, "vae canvas D2H failed");
                goto done;
            }
            if (stats) stats->unpatch_d2h_ms = elapsed_ms(td0, clock());
        }
        /* Grow-only pool: keep d_* / host slice across tiles. */
    }
    t1 = clock();
    if (stats) {
        stats->blocks_ms = elapsed_ms(t0, t1);
        stats->n_blocks = n_blocks;
    }

    if (!gpu_proj_done) {
        float *normed =
            (float *)malloc(n_patches * H3_VAE_INT8_HIDDEN * sizeof(float));
        size_t r, col;
        if (!normed) {
            fail(error, error_size, "OOM norm_out");
            goto done;
        }
        /* LayerNorm only on patch tokens (exclude registers). */
        for (r = 0; r < n_patches; r++) {
            const float *xr = seq_h + r * H3_VAE_INT8_HIDDEN;
            float *yr = normed + r * H3_VAE_INT8_HIDDEN;
            float mean = 0.0f, var = 0.0f, inv;
            for (col = 0; col < H3_VAE_INT8_HIDDEN; col++) mean += xr[col];
            mean /= (float)H3_VAE_INT8_HIDDEN;
            for (col = 0; col < H3_VAE_INT8_HIDDEN; col++) {
                float d = xr[col] - mean;
                var += d * d;
            }
            inv = 1.0f / sqrtf(var / (float)H3_VAE_INT8_HIDDEN + 1e-5f);
            for (col = 0; col < H3_VAE_INT8_HIDDEN; col++)
                yr[col] = (xr[col] - mean) * inv * norm_w[col] + norm_b[col];
        }
        if (!h3_cuda_gemm_f32(cuda, normed, proj_w, n_patches,
                              H3_VAE_INT8_OUTPUT_PATCH, H3_VAE_INT8_HIDDEN,
                              projected, NULL, error, error_size)) {
            free(normed);
            goto done;
        }
        for (i = 0; i < n_patches; i++)
            for (c = 0; c < H3_VAE_INT8_OUTPUT_PATCH; c++)
                projected[i * H3_VAE_INT8_OUTPUT_PATCH + c] += proj_b[c];
        free(normed);
    }

    if (!gpu_unpatch_done) {
        static const float mean[] = {0.485f, 0.456f, 0.406f};
        static const float deviation[] = {0.229f, 0.224f, 0.225f};
        size_t rgb_n =
            (size_t)out_frames * (size_t)pixel_h * (size_t)pixel_w * 3;
        int frame, y, x, ch;
        rgb = (float *)malloc(rgb_n * sizeof(float));
        if (!rgb) {
            fail(error, error_size, "OOM rgb");
            goto done;
        }
        for (frame = 0; frame < out_frames; frame++) {
            /* Metal unpack_frame_range: decoded_t = frame + FRAME_OFFSET(3);
             * for the 22-frame first chunk, frames >= 17 skip the 3-frame gap. */
            int decoded_t = frame + 3;
            int patch_t, within_t;
            if (g_vae_metal_chunk_unpack && out_frames == 22 && frame >= 17)
                decoded_t += 3;
            patch_t = decoded_t / 4;
            within_t = decoded_t % 4;
            if (patch_t >= latent_t) patch_t = latent_t - 1;
            for (y = 0; y < pixel_h; y++) {
                int patch_y = y / 16, within_y = y % 16;
                for (x = 0; x < pixel_w; x++) {
                    int patch_x = x / 16, within_x = x % 16;
                    size_t patch = ((size_t)patch_t * (size_t)latent_h +
                                    (size_t)patch_y) *
                                       (size_t)latent_w +
                                   (size_t)patch_x;
                    for (ch = 0; ch < 3; ch++) {
                        size_t component =
                            ((((size_t)ch * 4 + (size_t)within_t) * 16 +
                              (size_t)within_y) *
                                 16 +
                             (size_t)within_x);
                        float value =
                            projected[patch * H3_VAE_INT8_OUTPUT_PATCH +
                                      component] *
                                deviation[ch] +
                            mean[ch];
                        size_t dst =
                            (((size_t)frame * (size_t)pixel_h + (size_t)y) *
                                 (size_t)pixel_w +
                             (size_t)x) *
                                3 +
                            (size_t)ch;
                        if (value < 0.0f) value = 0.0f;
                        if (value > 1.0f) value = 1.0f;
                        rgb[dst] = value;
                    }
                }
            }
        }
        frames->rgb = rgb;
        frames->frames = out_frames;
        frames->height = pixel_h;
        frames->width = pixel_w;
        rgb = NULL;
    }

    if (stats) {
        stats->total_ms = elapsed_ms(tall0, clock());
        vae_fill_attn_backend(stats);
        stats->all_finite = 1;
        for (i = 0; i < 64 && i < (size_t)frames->frames *
                                       (size_t)frames->height *
                                       (size_t)frames->width * 3;
             i++) {
            if (!isfinite(frames->rgb[i])) {
                stats->all_finite = 0;
                break;
            }
        }
    }
    ok = 1;
done:
    if (flash_sdpa)
        h3_cuda_sdpa_set_force_flash(0);
    if (!heads_cached) {
        free(post_w);
        free(post_b);
        free(embed_w);
        free(embed_b);
        free(registers);
        free(norm_w);
        free(norm_b);
        free(proj_w);
        free(proj_b);
    }
    if (!host_pooled) {
        free(patches);
        free(seq_h);
        free(projected);
    }
    free(rgb);
    if (!ok) h3_vae_int8_frames_free(frames);
    return ok;
}

void h3_vae_int8_cuda_weights_free(h3_vae_int8_cuda_weights *w) {
    unsigned b;
    if (!w) return;
    if (w->cache_pinned) {
        if (w->cache_refs > 0) w->cache_refs--;
        return;
    }
    if (w->blocks) {
        for (b = 0; b < w->n_blocks; b++) {
            h3_vae_block_w *bw = &w->blocks[b];
            free(bw->w_qkv);
            free(bw->w_out);
            free(bw->w_w1);
            free(bw->w_w2);
            free(bw->b_qkv);
            free(bw->b_out);
            free(bw->b_w1);
            free(bw->b_w2);
            free(bw->norm1);
            free(bw->norm2);
            free(bw->scale1);
            free(bw->scale2);
            if (bw->d_w_qkv) cudaFree(bw->d_w_qkv);
            if (bw->d_w_out) cudaFree(bw->d_w_out);
            if (bw->d_w_w1) cudaFree(bw->d_w_w1);
            if (bw->d_w_w2) cudaFree(bw->d_w_w2);
            if (bw->d_b_qkv) cudaFree(bw->d_b_qkv);
            if (bw->d_b_out) cudaFree(bw->d_b_out);
            if (bw->d_b_w1) cudaFree(bw->d_b_w1);
            if (bw->d_b_w2) cudaFree(bw->d_b_w2);
            if (bw->d_norm1) cudaFree(bw->d_norm1);
            if (bw->d_norm2) cudaFree(bw->d_norm2);
            if (bw->d_scale1) cudaFree(bw->d_scale1);
            if (bw->d_scale2) cudaFree(bw->d_scale2);
            if (bw->d_w_qkv_h) cudaFree(bw->d_w_qkv_h);
            if (bw->d_w_out_h) cudaFree(bw->d_w_out_h);
            if (bw->d_w_w1_h) cudaFree(bw->d_w_w1_h);
            if (bw->d_w_w2_h) cudaFree(bw->d_w_w2_h);
        }
        free(w->blocks);
    }
    if (w->d_norm_out_w) cudaFree(w->d_norm_out_w);
    if (w->d_norm_out_b) cudaFree(w->d_norm_out_b);
    if (w->d_proj_out_w) cudaFree(w->d_proj_out_w);
    if (w->d_proj_out_b) cudaFree(w->d_proj_out_b);
    if (w->d_embed_w) cudaFree(w->d_embed_w);
    if (w->d_embed_b) cudaFree(w->d_embed_b);
    if (w->d_registers) cudaFree(w->d_registers);
    free(w->h_post_w);
    free(w->h_post_b);
    free(w->h_embed_w);
    free(w->h_embed_b);
    free(w->h_registers);
    free(w->h_norm_w);
    free(w->h_norm_b);
    free(w->h_proj_w);
    free(w->h_proj_b);
    vae_scratch_drop();
    free(w);
}

void h3_vae_int8_cuda_release_device(void) {
    /* Keep host pack + mmap pin. Only drop device W / scratch so the next
     * DiT clip can reclaim VRAM; clip-2 load restores from pack (no full
     * host dequant) or hits if device W was never dropped. */
    if (g_vae_cached_weights)
        vae_weights_device_drop(g_vae_cached_weights);
    vae_scratch_drop();
    fprintf(stderr,
            "vae_release_device: dropped device W/scratch host_pack=%s "
            "mmap_pin=%d\n",
            g_vae_cached_weights ? "kept" : "none",
            g_vae_hostreg_map ? 1 : 0);
}

void h3_vae_int8_cuda_release_cached(void) {
    h3_vae_int8_cuda_release_device();
}

void h3_vae_int8_cuda_release_full(void) {
    vae_weight_cache_drop();
    vae_scratch_drop();
    fprintf(stderr, "vae_release_full: dropped pack+device W mmap_pin=%d\n",
            g_vae_hostreg_map ? 1 : 0);
}

int h3_vae_int8_cuda_device_resident(void) {
    return g_vae_cached_weights && g_vae_cached_weights->device_resident &&
           g_vae_cached_weights->blocks &&
           g_vae_cached_weights->n_blocks > 0 &&
           g_vae_cached_weights->blocks[0].d_w_qkv != NULL;
}

int h3_vae_int8_cuda_hostreg_ok(void) { return g_vae_hostreg_map ? 1 : 0; }

void h3_vae_int8_cuda_hostreg_release(void) {
    if (!g_vae_hostreg_map) return;
    (void)cudaHostUnregister((void *)g_vae_hostreg_map);
    (void)cudaGetLastError();
    g_vae_hostreg_map = NULL;
    g_vae_hostreg_len = 0;
}

int h3_vae_int8_cuda_hostreg_prepare(const h3_vae_int8_store *store,
                                     char *error, size_t error_size) {
    const char *hr;
    unsigned flags = 0;
    cudaError_t hrst;
    if (error && error_size) error[0] = '\0';
    if (!store || !store->map || store->map_len == 0) {
        fail(error, error_size, "vae hostreg: no mmap");
        return 0;
    }
    hr = getenv("H3_VAE_HOSTREG");
    if (hr && hr[0] == '0' && hr[1] == '\0') {
        fail(error, error_size, "vae hostreg: H3_VAE_HOSTREG=0");
        return 0;
    }
    if (g_vae_hostreg_map == store->map && g_vae_hostreg_len == store->map_len)
        return 1;
    if (g_vae_hostreg_map) h3_vae_int8_cuda_hostreg_release();
#ifdef cudaHostRegisterReadOnly
    flags |= cudaHostRegisterReadOnly;
#endif
    hrst = cudaHostRegister((void *)store->map, store->map_len, flags);
    if (hrst != cudaSuccess) {
        fprintf(stderr, "vae_hostreg: fail %s (sync H2D)\n",
                cudaGetErrorString(hrst));
        (void)cudaGetLastError();
        fail(error, error_size, "vae hostreg: cudaHostRegister failed");
        return 0;
    }
    g_vae_hostreg_map = store->map;
    g_vae_hostreg_len = store->map_len;
    return 1;
}

static void vae_weight_cache_drop(void) {
    h3_vae_int8_cuda_weights *cached = g_vae_cached_weights;
    if (!cached) return;
    g_vae_cached_weights = NULL;
    cached->cache_pinned = 0;
    cached->cache_refs = 0;
    h3_vae_int8_cuda_weights_free(cached);
    vae_scratch_drop();
}

static int upload_w(float *host, size_t n, float **dev) {
    if (!host || n == 0) {
        *dev = NULL;
        return 1;
    }
    if (cudaMalloc((void **)dev, n * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "upload_w: cudaMalloc %zu floats failed\n", n);
        return 0;
    }
    if (cudaMemcpy(*dev, host, n * sizeof(float), cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        fprintf(stderr, "upload_w: cudaMemcpy %zu floats failed\n", n);
        cudaFree(*dev);
        *dev = NULL;
        return 0;
    }
    return 1;
}

static void vae_cuda_free_f(float **p) {
    if (p && *p) {
        cudaFree(*p);
        *p = NULL;
    }
}

static void vae_cuda_free_v(void **p) {
    if (p && *p) {
        cudaFree(*p);
        *p = NULL;
    }
}

static void vae_block_device_drop(h3_vae_block_w *bw) {
    if (!bw) return;
    /* Do not D2H 7GB F32 for host_pack_restore. Hot refill is mmap H2D
     * overlapped with last NFE (prewarm), not a blocking pack snapshot. */
    vae_cuda_free_f(&bw->d_w_qkv);
    vae_cuda_free_f(&bw->d_w_out);
    vae_cuda_free_f(&bw->d_w_w1);
    vae_cuda_free_f(&bw->d_w_w2);
    vae_cuda_free_f(&bw->d_b_qkv);
    vae_cuda_free_f(&bw->d_b_out);
    vae_cuda_free_f(&bw->d_b_w1);
    vae_cuda_free_f(&bw->d_b_w2);
    vae_cuda_free_f(&bw->d_norm1);
    vae_cuda_free_f(&bw->d_norm2);
    vae_cuda_free_f(&bw->d_scale1);
    vae_cuda_free_f(&bw->d_scale2);
    vae_cuda_free_v(&bw->d_w_qkv_h);
    vae_cuda_free_v(&bw->d_w_out_h);
    vae_cuda_free_v(&bw->d_w_w1_h);
    vae_cuda_free_v(&bw->d_w_w2_h);
}

static void vae_weights_device_drop(h3_vae_int8_cuda_weights *w) {
    unsigned b;
    if (!w) return;
    if (w->blocks) {
        for (b = 0; b < w->n_blocks; b++)
            vae_block_device_drop(&w->blocks[b]);
    }
    vae_cuda_free_f(&w->d_norm_out_w);
    vae_cuda_free_f(&w->d_norm_out_b);
    vae_cuda_free_f(&w->d_proj_out_w);
    vae_cuda_free_f(&w->d_proj_out_b);
    vae_cuda_free_f(&w->d_embed_w);
    vae_cuda_free_f(&w->d_embed_b);
    vae_cuda_free_f(&w->d_registers);
    w->device_resident = 0;
}

static int vae_weights_device_restore(h3_vae_int8_cuda_weights *w) {
    unsigned b;
    const size_t hidden = H3_VAE_INT8_HIDDEN;
    if (!w || !w->blocks) {
        fprintf(stderr, "vae_weights_device_restore: w/blocks NULL\n");
        return 0;
    }
    for (b = 0; b < w->n_blocks; b++) {
        h3_vae_block_w *bw = &w->blocks[b];
        if (!bw->norm1 || !bw->norm2 || !bw->scale1 || !bw->scale2) {
            fprintf(stderr,
                    "vae_weights_device_restore: block %u norm/scale NULL\n",
                    b);
            return 0;
        }
        /* block_gpu=1 (GPU direct load) never allocated host w_*, so restore
         * must fallback to GPU path (same as initial load). */
        if (!bw->w_qkv || !bw->w_out || !bw->w_w1 || !bw->w_w2) {
            fprintf(stderr,
                    "vae_weights_device_restore: block %u host w NULL "
                    "(block_gpu path) → fallback GPU direct load\n",
                    b);
            return 0;
        }
        if (!(upload_w(bw->w_qkv, bw->n_qkv * bw->k_qkv, &bw->d_w_qkv) &&
              upload_w(bw->w_out, bw->n_out * bw->k_out, &bw->d_w_out) &&
              upload_w(bw->w_w1, bw->n_w1 * bw->k_w1, &bw->d_w_w1) &&
              upload_w(bw->w_w2, bw->n_w2 * bw->k_w2, &bw->d_w_w2) &&
              upload_w(bw->norm1, hidden, &bw->d_norm1) &&
              upload_w(bw->norm2, hidden, &bw->d_norm2) &&
              upload_w(bw->scale1, hidden, &bw->d_scale1) &&
              upload_w(bw->scale2, hidden, &bw->d_scale2) &&
              (!bw->b_qkv || upload_w(bw->b_qkv, bw->n_qkv, &bw->d_b_qkv)) &&
              (!bw->b_out || upload_w(bw->b_out, bw->n_out, &bw->d_b_out)) &&
              (!bw->b_w1 || upload_w(bw->b_w1, bw->n_w1, &bw->d_b_w1)) &&
              (!bw->b_w2 || upload_w(bw->b_w2, bw->n_w2, &bw->d_b_w2)))) {
            fprintf(stderr,
                    "vae_weights_device_restore: block %u upload failed\n",
                    b);
            vae_block_device_drop(bw);
            return 0;
        }
    }
    w->device_resident = 1;
    return 1;
}

static int ensure_vae_head_dev(h3_vae_int8_cuda_weights *w, const float *norm_w,
                               const float *norm_b, const float *proj_w,
                               const float *proj_b) {
    const size_t hidden = H3_VAE_INT8_HIDDEN;
    const size_t pout = H3_VAE_INT8_OUTPUT_PATCH;
    if (!w) return 0;
    if (w->d_proj_out_w && w->d_proj_out_b && w->d_norm_out_w &&
        w->d_norm_out_b)
        return 1;
    if (w->d_norm_out_w) {
        cudaFree(w->d_norm_out_w);
        w->d_norm_out_w = NULL;
    }
    if (w->d_norm_out_b) {
        cudaFree(w->d_norm_out_b);
        w->d_norm_out_b = NULL;
    }
    if (w->d_proj_out_w) {
        cudaFree(w->d_proj_out_w);
        w->d_proj_out_w = NULL;
    }
    if (w->d_proj_out_b) {
        cudaFree(w->d_proj_out_b);
        w->d_proj_out_b = NULL;
    }
    if (!upload_w((float *)norm_w, hidden, &w->d_norm_out_w) ||
        !upload_w((float *)norm_b, hidden, &w->d_norm_out_b) ||
        !upload_w((float *)proj_w, pout * hidden, &w->d_proj_out_w) ||
        !upload_w((float *)proj_b, pout, &w->d_proj_out_b))
        return 0;
    return 1;
}

static int ensure_vae_embed_dev(h3_vae_int8_cuda_weights *w, const float *embed_w,
                                const float *embed_b, const float *registers) {
    const size_t hidden = H3_VAE_INT8_HIDDEN;
    const size_t k = H3_VAE_INT8_LATENT_CH;
    const size_t nreg = (size_t)H3_VAE_INT8_REGISTERS * hidden;
    if (!w || !embed_w || !embed_b || !registers)
        return 0;
    if (w->d_embed_w && w->d_embed_b && w->d_registers)
        return 1;
    if (w->d_embed_w) {
        cudaFree(w->d_embed_w);
        w->d_embed_w = NULL;
    }
    if (w->d_embed_b) {
        cudaFree(w->d_embed_b);
        w->d_embed_b = NULL;
    }
    if (w->d_registers) {
        cudaFree(w->d_registers);
        w->d_registers = NULL;
    }
    if (!upload_w((float *)embed_w, hidden * k, &w->d_embed_w) ||
        !upload_w((float *)embed_b, hidden, &w->d_embed_b) ||
        !upload_w((float *)registers, nreg, &w->d_registers))
        return 0;
    return 1;
}

static const void *vae_mmap_bytes(const h3_vae_int8_store *store, uint64_t off,
                                  size_t nbytes) {
    if (!store || !store->map || nbytes == 0) return NULL;
    if ((size_t)off > store->map_len ||
        nbytes > store->map_len - (size_t)off)
        return NULL;
    return store->map + (size_t)off;
}

static int vae_h2d_bytes(const h3_vae_int8_store *store, uint64_t off,
                         void *d_dst, size_t nbytes, void *h_stage,
                         size_t stage_bytes, cudaStream_t stream,
                         int mmap_registered, char *error, size_t error_size) {
    const void *src = vae_mmap_bytes(store, off, nbytes);
    cudaError_t st;
    /* Registered mmap can DMA. Unregistered file mmap + cudaMemcpyAsync on a
     * non-default stream is illegal (cafe p0/p2: hostreg=0 → 8s host F16). */
    if (src && mmap_registered) {
        if (stream)
            st = cudaMemcpyAsync(d_dst, src, nbytes, cudaMemcpyHostToDevice,
                                 stream);
        else
            st = cudaMemcpy(d_dst, src, nbytes, cudaMemcpyHostToDevice);
        if (st == cudaSuccess) return 1;
        {
            static int logged;
            if (!logged) {
                fprintf(stderr, "vae_h2d: registered mmap fail %s → staging\n",
                        cudaGetErrorString(st));
                fflush(stderr);
                logged = 1;
            }
        }
    }
    if (h_stage && nbytes <= stage_bytes) {
        if (src)
            memcpy(h_stage, src, nbytes);
        else if (!h3_vae_int8_store_copy(store, (size_t)off, h_stage, nbytes,
                                         error, error_size))
            return 0;
        if (stream)
            st = cudaMemcpyAsync(d_dst, h_stage, nbytes, cudaMemcpyHostToDevice,
                                 stream);
        else
            st = cudaMemcpy(d_dst, h_stage, nbytes, cudaMemcpyHostToDevice);
        if (st != cudaSuccess) {
            snprintf(error, error_size, "vae H2D %s", cudaGetErrorString(st));
            return 0;
        }
        return 1;
    }
    if (src && !stream) {
        st = cudaMemcpy(d_dst, src, nbytes, cudaMemcpyHostToDevice);
        if (st == cudaSuccess) return 1;
        snprintf(error, error_size, "vae H2D %s", cudaGetErrorString(st));
        return 0;
    }
    fail(error, error_size, "vae i8 staging too small");
    return 0;
}

/* I8 mmap/H2D + device row-scale dequant → d_out F32[N*K]. No host F32 W. */
static int dequant_linear_gpu(const h3_vae_int8_store *store,
                              const h3_vae_int8_linear *lin, float **d_out,
                              int8_t *d_i8, size_t i8_bytes, float *d_scale,
                              size_t scale_cap, void *h_i8_stage,
                              size_t stage_bytes, void *h_scale_stage,
                              size_t scale_stage_bytes, cudaStream_t stream,
                              int mmap_registered, char *error,
                              size_t error_size) {
    size_t nk = lin->n * lin->k;
    size_t scale_bytes = lin->n * sizeof(float);
    *d_out = NULL;
    if (nk > i8_bytes || lin->n > scale_cap) {
        fail(error, error_size, "vae gpu dequant scratch too small");
        return 0;
    }
    if (cudaMalloc((void **)d_out, nk * sizeof(float)) != cudaSuccess) {
        fail(error, error_size, "vae gpu dequant cudaMalloc W");
        return 0;
    }
    if (!vae_h2d_bytes(store, lin->weight->file_offset, d_i8, nk, h_i8_stage,
                       stage_bytes, stream, mmap_registered, error,
                       error_size)) {
        cudaFree(*d_out);
        *d_out = NULL;
        return 0;
    }
    if (!vae_h2d_bytes(store, lin->scale->file_offset, d_scale, scale_bytes,
                       h_scale_stage, scale_stage_bytes, stream,
                       mmap_registered, error, error_size)) {
        cudaFree(*d_out);
        *d_out = NULL;
        return 0;
    }
    if (!h3_cuda_dequant_i8_rowscale_f32(d_i8, d_scale, lin->n, lin->k, *d_out,
                                         stream, error, error_size)) {
        cudaFree(*d_out);
        *d_out = NULL;
        return 0;
    }
    return 1;
}

/* Native F16 weight → device F32 (IEEE upcast). Avoids the host F16→F32
 * loop that made fp16 decode load ~10s. Same math as load_f32().
 * Reuses d_f16_scratch / h_stage across linears (no per-weight malloc). */
static int load_linear_f16_gpu(const h3_vae_int8_store *store,
                               const h3_vae_int8_linear *lin, float **d_out,
                               void **d_f16_scratch, size_t *d_f16_cap,
                               void *h_stage, size_t stage_bytes,
                               cudaStream_t stream, int mmap_registered,
                               char *error, size_t error_size) {
    size_t nk, f16_bytes;
    *d_out = NULL;
    if (!lin || !lin->weight || lin->weight->dtype != H3_DTYPE_F16) return 0;
    nk = lin->n * lin->k;
    if (nk == 0) return 0;
    f16_bytes = nk * sizeof(uint16_t);
    if (cudaMalloc((void **)d_out, nk * sizeof(float)) != cudaSuccess) {
        fail(error, error_size, "vae f16 load cudaMalloc W");
        return 0;
    }
    if (!d_f16_scratch || !d_f16_cap) {
        cudaFree(*d_out);
        *d_out = NULL;
        fail(error, error_size, "vae f16 load missing scratch");
        return 0;
    }
    if (!*d_f16_scratch || *d_f16_cap < f16_bytes) {
        if (*d_f16_scratch) cudaFree(*d_f16_scratch);
        *d_f16_scratch = NULL;
        *d_f16_cap = 0;
        if (cudaMalloc(d_f16_scratch, f16_bytes) != cudaSuccess) {
            cudaFree(*d_out);
            *d_out = NULL;
            fail(error, error_size, "vae f16 load cudaMalloc half");
            return 0;
        }
        *d_f16_cap = f16_bytes;
    }
    if (!vae_h2d_bytes(store, lin->weight->file_offset, *d_f16_scratch,
                       f16_bytes, h_stage, stage_bytes, stream, mmap_registered,
                       error, error_size)) {
        cudaFree(*d_out);
        *d_out = NULL;
        return 0;
    }
    if (stream)
        cudaStreamSynchronize(stream);
    else
        cudaDeviceSynchronize();
    if (!h3_cuda_f16_to_f32_dev(NULL, *d_f16_scratch, *d_out, nk, error,
                                error_size)) {
        cudaFree(*d_out);
        *d_out = NULL;
        return 0;
    }
    if (stream)
        cudaStreamSynchronize(stream);
    else
        cudaDeviceSynchronize();
    return 1;
}

/* Allocate device half buffer and cast from device FP32 (2 bytes/elem). */
static int cast_w_to_f16(h3_cuda *cuda, const float *d_f32, size_t n,
                         void **d_h, char *error, size_t error_size) {
    if (!d_f32 || n == 0) {
        *d_h = NULL;
        return 1;
    }
    if (cudaMalloc(d_h, n * 2) != cudaSuccess) {
        *d_h = NULL;
        return 0;
    }
    if (!h3_cuda_f32_to_f16_dev(cuda, d_f32, *d_h, n, error, error_size)) {
        cudaFree(*d_h);
        *d_h = NULL;
        return 0;
    }
    return 1;
}

/* F16×F16→F16, COMPUTE_32F. No HBM cast. */
static int gemm_f16_act(h3_cuda *cuda, const void *d_x_h, const void *d_w_h,
                        size_t m, size_t n, size_t k, void *d_y_h, char *error,
                        size_t error_size) {
    if (!d_w_h) {
        fail(error, error_size, "vae f16 weight missing");
        return 0;
    }
    return h3_cuda_gemm_f16_acc32_dd(cuda, d_x_h, d_w_h, m, n, k, d_y_h, error,
                                     error_size);
}

static int vae_prep_f16(h3_cuda *cuda, const float *d_f32, void *d_xh,
                        void *d_rot, size_t seq, size_t k, int convrot_gs,
                        const void **d_gemm_x, char *error, size_t error_size) {
    if (convrot_gs > 0) {
        if (!h3_cuda_convrot_act_f16_dd(cuda, d_xh, seq, k, convrot_gs, d_rot,
                                        error, error_size))
            return 0;
        *d_gemm_x = d_rot;
        return 1;
    }
    (void)d_f32;
    *d_gemm_x = d_xh;
    return 1;
}

/* Layer-local F16 activations. Residual stays F32. */
static int run_block_cuda_w_dev_f16(h3_cuda *cuda, const h3_vae_block_w *bw,
                                    float *d_seq, float *d_ws, size_t seq,
                                    const float *d_rope_cos,
                                    const float *d_rope_sin,
                                    const float *d_qk_ones, int convrot_gs,
                                    char *error, size_t error_size) {
    const size_t hidden = H3_VAE_INT8_HIDDEN;
    const size_t heads = H3_VAE_INT8_HEADS;
    const size_t head_dim = H3_VAE_INT8_HEAD_DIM;
    const size_t ffn = H3_VAE_INT8_FFN;
    float scale = 1.0f / sqrtf((float)head_dim);
    char *base = (char *)d_ws;
    size_t off = 0;
    size_t rot_k;
    void *d_xh, *d_rot, *d_qkv, *d_attn, *d_branch, *d_ff, *d_act;
    const void *d_gemm_x;
    int saved_quiet;
    int mode;
    size_t attn_seq, n_batch;
    int split_half = 1;

    {
        static int logged_path;
        if (!logged_path) {
            fprintf(stderr, "vae_gemm_path: act_f16\n");
            logged_path = 1;
        }
    }

    rot_k = bw->k_w2 > bw->k_qkv ? bw->k_w2 : bw->k_qkv;
    if (bw->k_out > rot_k) rot_k = bw->k_out;
    if (bw->k_w1 > rot_k) rot_k = bw->k_w1;

#define H3_VAE_TAKE_H(n)                                                       \
    (off = (off + 15u) & ~(size_t)15u, off += (n) * 2u,                        \
     (void *)(base + (off - (n) * 2u)))
    d_xh = H3_VAE_TAKE_H(seq * rot_k);
    d_rot = H3_VAE_TAKE_H(seq * rot_k);
    d_qkv = H3_VAE_TAKE_H(seq * bw->n_qkv);
    d_attn = H3_VAE_TAKE_H(seq * hidden);
    d_branch = H3_VAE_TAKE_H(seq * hidden);
    d_ff = H3_VAE_TAKE_H(seq * bw->n_w1);
    d_act = H3_VAE_TAKE_H(seq * ffn);
#undef H3_VAE_TAKE_H

    saved_quiet = h3_cuda_quiet(cuda);
    h3_cuda_set_quiet(cuda, 1);

    if (!h3_cuda_vae_rms_norm_f32_to_f16_dev(cuda, d_seq, bw->d_norm1, seq,
                                             hidden, 1e-5f, d_xh, error,
                                             error_size)) {
        h3_cuda_set_quiet(cuda, saved_quiet);
        return 0;
    }
    if (!vae_prep_f16(cuda, NULL, d_xh, d_rot, seq, bw->k_qkv, convrot_gs,
                      &d_gemm_x, error, error_size) ||
        !gemm_f16_act(cuda, d_gemm_x, bw->d_w_qkv_h, seq, bw->n_qkv, bw->k_qkv,
                      d_qkv, error, error_size)) {
        h3_cuda_set_quiet(cuda, saved_quiet);
        return 0;
    }
    if (bw->d_b_qkv &&
        !h3_cuda_vae_bias_add_f16_dev(cuda, d_qkv, bw->d_b_qkv, seq, bw->n_qkv,
                                      error, error_size)) {
        h3_cuda_set_quiet(cuda, saved_quiet);
        return 0;
    }
    if (!d_rope_cos || !d_rope_sin || !d_qk_ones) {
        fail(error, error_size, "vae block missing rope");
        h3_cuda_set_quiet(cuda, saved_quiet);
        return 0;
    }
    mode = vae_qkv_layout_mode();
    attn_seq = seq;
    n_batch = 1;
    if (g_vae_attn_seq > 0 && seq >= g_vae_attn_seq &&
        (seq % g_vae_attn_seq) == 0) {
        attn_seq = g_vae_attn_seq;
        n_batch = seq / attn_seq;
    }
    {
        const char *half_off = getenv("H3_DIT_ROPE_HALF_OFFSET");
        const char *e = getenv("H3_DIT_ROPE_SPLIT_HALF");
        if (half_off && half_off[0] == '1' && half_off[1] == '\0')
            split_half = 0;
        if (e && e[0] == '0' && e[1] == '\0') split_half = 0;
        if (e && e[0] == '1' && e[1] == '\0') split_half = 1;
    }
    if (mode == 2) {
        fail(error, error_size, "vae act_f16 norope unsupported");
        h3_cuda_set_quiet(cuda, saved_quiet);
        return 0;
    }
    if (!h3_cuda_vae_qkv_rope_sdpa_f16io_dev_layout_nb(
            cuda, d_qkv, d_qk_ones, d_qk_ones, d_rope_cos, d_rope_sin, attn_seq,
            n_batch, heads, head_dim, H3_VAE_INT8_ROPE_HALF, 1e-5f, scale,
            mode == 1, split_half, d_attn, error, error_size)) {
        h3_cuda_set_quiet(cuda, saved_quiet);
        return 0;
    }

    d_gemm_x = d_attn;
    if (convrot_gs > 0) {
        if (!h3_cuda_convrot_act_f16_dd(cuda, d_attn, seq, bw->k_out, convrot_gs,
                                        d_rot, error, error_size)) {
            h3_cuda_set_quiet(cuda, saved_quiet);
            return 0;
        }
        d_gemm_x = d_rot;
    }
    if (!gemm_f16_act(cuda, d_gemm_x, bw->d_w_out_h, seq, bw->n_out, bw->k_out,
                      d_branch, error, error_size) ||
        !h3_cuda_vae_bias_residual_scale_f16_dev(
            cuda, d_seq, d_branch, bw->d_b_out, bw->d_scale1, seq, hidden,
            error, error_size)) {
        h3_cuda_set_quiet(cuda, saved_quiet);
        return 0;
    }

    if (!h3_cuda_vae_rms_norm_f32_to_f16_dev(cuda, d_seq, bw->d_norm2, seq,
                                             hidden, 1e-5f, d_xh, error,
                                             error_size) ||
        !vae_prep_f16(cuda, NULL, d_xh, d_rot, seq, bw->k_w1, convrot_gs,
                      &d_gemm_x, error, error_size) ||
        !gemm_f16_act(cuda, d_gemm_x, bw->d_w_w1_h, seq, bw->n_w1, bw->k_w1,
                      d_ff, error, error_size) ||
        !h3_cuda_vae_bias_silu_mul_f16_dev(cuda, d_ff, bw->d_b_w1, seq, ffn,
                                           d_act, error, error_size)) {
        h3_cuda_set_quiet(cuda, saved_quiet);
        return 0;
    }
    d_gemm_x = d_act;
    if (convrot_gs > 0) {
        if (!h3_cuda_convrot_act_f16_dd(cuda, d_act, seq, bw->k_w2, convrot_gs,
                                        d_rot, error, error_size)) {
            h3_cuda_set_quiet(cuda, saved_quiet);
            return 0;
        }
        d_gemm_x = d_rot;
    }
    if (!gemm_f16_act(cuda, d_gemm_x, bw->d_w_w2_h, seq, bw->n_w2, bw->k_w2,
                      d_branch, error, error_size) ||
        !h3_cuda_vae_bias_residual_scale_f16_dev(
            cuda, d_seq, d_branch, bw->d_b_w2, bw->d_scale2, seq, hidden, error,
            error_size)) {
        h3_cuda_set_quiet(cuda, saved_quiet);
        return 0;
    }
    h3_cuda_set_quiet(cuda, saved_quiet);
    return 1;
}

/* FP16 TC GEMM with FP32 I/O. Mixed F32×F16 is tried first; 5090 has no
 * algo (NOT_SUPPORTED / lt_heur=0). F32/TF32 W is slower (~17s vs ~11.7s
 * 768P blocks) so product fallback stays HBM cast. */
static int gemm_f16_via(h3_cuda *cuda, const float *d_x, const void *d_w_h,
                        size_t m, size_t n, size_t k, float *d_y,
                        void *d_scratch_h, char *error, size_t error_size) {
    void *d_xh = d_scratch_h;
    if (!d_w_h) {
        fail(error, error_size, "vae f16 weight missing");
        return 0;
    }
    if (h3_vae_int8_gemm_f16_direct() &&
        h3_cuda_gemm_f32in_f16w_f32out_dd(cuda, d_x, d_w_h, m, n, k, d_y, NULL,
                                          error, error_size)) {
        static int logged_mix;
        if (!logged_mix) {
            fprintf(stderr, "vae_gemm_path: mixed_f32in_f16w_f32out\n");
            logged_mix = 1;
        }
        return 1;
    }
    {
        static int logged_fb;
        if (!logged_fb) {
            fprintf(stderr, "vae_gemm_path: fallback_cast\n");
            logged_fb = 1;
        }
    }
    if (!d_scratch_h) {
        fail(error, error_size, "vae f16 scratch missing");
        return 0;
    }
    if (!h3_cuda_f32_to_f16_dev(cuda, d_x, d_xh, m * k, error, error_size))
        return 0;
    if (h3_cuda_gemm_f16_f32out_dd(cuda, d_xh, d_w_h, m, n, k, d_y, error,
                                   error_size))
        return 1;
    {
        void *d_yh = (char *)d_scratch_h + m * k * 2;
        if (!h3_cuda_gemm_f16_dd(cuda, d_xh, d_w_h, m, n, k, d_yh, NULL, error,
                                 error_size))
            return 0;
        return h3_cuda_f16_to_f32_dev(cuda, d_yh, d_y, m * n, error,
                                      error_size);
    }
}

int h3_vae_int8_cuda_weights_load(const h3_vae_int8_store *store,
                                  unsigned n_blocks,
                                  h3_vae_int8_cuda_weights **out,
                                  double *load_ms, char *error,
                                  size_t error_size) {
    /* GPU unpatch D2H pins fr->rgb via cudaHostAlloc; ensure frames_free
     * releases it with cudaFreeHost (registered once, host path keeps free). */
    h3_vae_int8_set_rgb_host_free(vae_rgb_pinned_free);
    h3_vae_int8_cuda_weights *w;
    unsigned b;
    double t0_ms, t_find_ms, t_f32_ms, t_i8_ms, t_bias_ms, t_h2d_ms, t_done_ms;
    size_t hidden = H3_VAE_INT8_HIDDEN;
    int want_dev = 1;
    int gpu_dequant = 0;
    int hostreg = 0;
    int cache_enabled = vae_weight_cache_enabled();
    cudaStream_t load_stream = 0;
    int8_t *d_i8 = NULL;
    float *d_scale = NULL;
    void *h_i8_stage = NULL;
    void *h_scale_stage = NULL;
    int h_i8_pinned = 0, h_scale_pinned = 0;
    int hostreg_owned = 0;
    void *d_f16_scratch = NULL;
    size_t d_f16_cap = 0;
    void *h_f16_stage = NULL;
    int h_f16_pinned = 0;
    size_t h_f16_stage_bytes = 0;
    const size_t i8_stage_bytes = 32u * 1024u * 1024u;
    const size_t scale_cap = 16384;
    const size_t scale_stage_bytes = scale_cap * sizeof(float);
    {
        const char *e = getenv("H3_VAE_CUDA_W_RESIDENT");
        if (e && e[0] == '0' && e[1] == '\0') want_dev = 0;
    }

    if (load_ms) *load_ms = 0.0;
    if (!store || !out) {
        fail(error, error_size, "invalid weights_load");
        return 0;
    }
    if (n_blocks > H3_VAE_INT8_N_BLOCKS) n_blocks = H3_VAE_INT8_N_BLOCKS;
    if (cache_enabled && g_vae_cached_weights &&
        g_vae_cached_weights->n_blocks == n_blocks &&
        !strcmp(g_vae_cached_weights->cache_path, store->path)) {
        int have_dev =
            g_vae_cached_weights->device_resident && g_vae_cached_weights->blocks &&
            n_blocks > 0 && g_vae_cached_weights->blocks[0].d_w_qkv &&
            g_vae_cached_weights->blocks[n_blocks - 1].d_w_qkv;
        if (want_dev && !have_dev) {
            double t_restore0 = now_ms();
            if (!vae_weights_device_restore(g_vae_cached_weights)) {
                fprintf(stderr,
                        "vae_weights_load: cache=host_pack_restore_fail "
                        "blocks=%u path=%s (full reload)\n",
                        n_blocks, store->path);
                vae_weight_cache_drop();
            } else {
                if (load_ms) *load_ms = now_ms() - t_restore0;
                g_vae_cached_weights->cache_refs++;
                *out = g_vae_cached_weights;
                fprintf(stderr,
                        "vae_weights_load: cache=host_pack_restore blocks=%u "
                        "device=%d total_ms=%.1f path=%s\n",
                        n_blocks, want_dev, load_ms ? *load_ms : 0.0,
                        store->path);
                return 1;
            }
        } else if (g_vae_cached_weights) {
            g_vae_cached_weights->cache_refs++;
            *out = g_vae_cached_weights;
            fprintf(stderr,
                    "vae_weights_load: cache=hit blocks=%u device=%d path=%s\n",
                    n_blocks, want_dev, store->path);
            return 1;
        }
    }
    if (cache_enabled && g_vae_cached_weights &&
        (strcmp(g_vae_cached_weights->cache_path, store->path) ||
         g_vae_cached_weights->n_blocks != n_blocks)) {
        vae_weight_cache_drop();
    }
    w = (h3_vae_int8_cuda_weights *)calloc(1, sizeof(*w));
    if (!w) {
        fail(error, error_size, "OOM weights");
        return 0;
    }
    w->n_blocks = n_blocks;
    w->blocks = (h3_vae_block_w *)calloc(n_blocks, sizeof(h3_vae_block_w));
    if (!w->blocks) {
        free(w);
        fail(error, error_size, "OOM weights blocks");
        return 0;
    }
    t0_ms = now_ms();
    gpu_dequant = want_dev && !env_flag_is_one("H3_VAE_HOST_DEQUANT");
    if (gpu_dequant) {
        int native_f16_ckpt = 0;
        const h3_st_tensor *peek = h3_st_find(
            &store->header, "decoder.transformer_blocks.0.attn.to_qkv.weight");
        if (peek && peek->dtype == H3_DTYPE_F16) native_f16_ckpt = 1;
        if (store->map && store->map_len > 0) {
#ifdef MADV_WILLNEED
            (void)madvise((void *)store->map, store->map_len, MADV_WILLNEED);
#endif
            /* Pin VAE mmap for the store lifetime (community resident load).
             * H3_VAE_HOSTREG=0 disables. Do not unregister after one fill. */
            {
                const char *hr = getenv("H3_VAE_HOSTREG");
                int skip_hostreg = hr && hr[0] == '0' && hr[1] == '\0';
                if (!skip_hostreg) {
                    if (g_vae_hostreg_map == store->map &&
                        g_vae_hostreg_len == store->map_len) {
                        hostreg = 1;
                    } else {
                        unsigned flags = 0;
#ifdef cudaHostRegisterReadOnly
                        flags |= cudaHostRegisterReadOnly;
#endif
                        if (g_vae_hostreg_map)
                            h3_vae_int8_cuda_hostreg_release();
                        {
                            cudaError_t hrst = cudaHostRegister(
                                (void *)store->map, store->map_len, flags);
                            if (hrst == cudaSuccess) {
                                hostreg = 1;
                                hostreg_owned = 1;
                                g_vae_hostreg_map = store->map;
                                g_vae_hostreg_len = store->map_len;
                            } else {
                                fprintf(stderr,
                                        "vae_hostreg: fail %s (sync H2D)\n",
                                        cudaGetErrorString(hrst));
                                (void)cudaGetLastError();
                            }
                        }
                    }
                }
            }
        }
        if (hostreg) {
            if (cudaStreamCreateWithFlags(&load_stream,
                                          cudaStreamNonBlocking) != cudaSuccess)
                load_stream = 0;
        } else {
            /* Unregistered mmap + non-blocking stream memcpyAsync fails
             * (cafe p0: hostreg=0 → gpu_dequant=0 → 8s host F16 dequant). */
            load_stream = 0;
        }
        if (!native_f16_ckpt) {
            if (cudaMalloc((void **)&d_i8, i8_stage_bytes) != cudaSuccess ||
                cudaMalloc((void **)&d_scale, scale_stage_bytes) !=
                    cudaSuccess) {
                gpu_dequant = 0;
                if (d_i8) {
                    cudaFree(d_i8);
                    d_i8 = NULL;
                }
                if (d_scale) {
                    cudaFree(d_scale);
                    d_scale = NULL;
                }
            }
        }
        if (gpu_dequant && !store->map) {
            h_i8_stage = h3_cuda_host_alloc(i8_stage_bytes, &h_i8_pinned);
            h_scale_stage =
                h3_cuda_host_alloc(scale_stage_bytes, &h_scale_pinned);
            if (!h_i8_stage || !h_scale_stage) gpu_dequant = 0;
        }
        if (gpu_dequant && native_f16_ckpt) {
            /* qkv F16 ≈ 3*2048*2048*2 = 50MiB; 96MiB covers FFN. hostreg=0
             * copies mmap → pinned then legal H2D (cafe p2: 8s host loop). */
            h_f16_stage_bytes = (size_t)96 << 20;
            h_f16_stage = h3_cuda_host_alloc(h_f16_stage_bytes, &h_f16_pinned);
            if (!h_f16_stage) gpu_dequant = 0;
        }
    }
    t_find_ms = t_f32_ms = t_i8_ms = t_bias_ms = t_h2d_ms = 0.0;
    for (b = 0; b < n_blocks; b++) {
        char base[128];
        h3_vae_int8_linear qkv, out_l, w1, w2;
        h3_vae_block_w *bw = &w->blocks[b];
        int block_gpu = gpu_dequant;
        double tb0 = now_ms();
        snprintf(base, sizeof(base),
                 "decoder.transformer_blocks.%u.attn.to_qkv", b);
        if (!h3_vae_int8_find_linear(store, base, &qkv, error, error_size))
            goto fail;
        snprintf(base, sizeof(base),
                 "decoder.transformer_blocks.%u.attn.to_out", b);
        if (!h3_vae_int8_find_linear(store, base, &out_l, error, error_size))
            goto fail;
        snprintf(base, sizeof(base), "decoder.transformer_blocks.%u.ff.w1", b);
        if (!h3_vae_int8_find_linear(store, base, &w1, error, error_size))
            goto fail;
        snprintf(base, sizeof(base), "decoder.transformer_blocks.%u.ff.w2", b);
        if (!h3_vae_int8_find_linear(store, base, &w2, error, error_size))
            goto fail;
        t_find_ms += now_ms() - tb0;
        tb0 = now_ms();
        snprintf(base, sizeof(base),
                 "decoder.transformer_blocks.%u.norm1.weight", b);
        bw->norm1 = load_f32(store, base, hidden, error, error_size);
        snprintf(base, sizeof(base),
                 "decoder.transformer_blocks.%u.norm2.weight", b);
        bw->norm2 = load_f32(store, base, hidden, error, error_size);
        snprintf(base, sizeof(base), "decoder.transformer_blocks.%u.scale1", b);
        bw->scale1 = load_f32(store, base, hidden, error, error_size);
        snprintf(base, sizeof(base), "decoder.transformer_blocks.%u.scale2", b);
        bw->scale2 = load_f32(store, base, hidden, error, error_size);
        /* Ablation: H3_VAE_UNIT_SCALE=1 replaces tiny learned gates with 1. */
        if (bw->scale1 && bw->scale2 && env_flag_is_one("H3_VAE_UNIT_SCALE")) {
            size_t si;
            for (si = 0; si < hidden; si++) {
                bw->scale1[si] = 1.0f;
                bw->scale2[si] = 1.0f;
            }
        }
        t_f32_ms += now_ms() - tb0;
        bw->n_qkv = qkv.n;
        bw->k_qkv = qkv.k;
        bw->n_out = out_l.n;
        bw->k_out = out_l.k;
        bw->n_w1 = w1.n;
        bw->k_w1 = w1.k;
        bw->n_w2 = w2.n;
        bw->k_w2 = w2.k;
        tb0 = now_ms();
        if (block_gpu &&
            (qkv.weight->dtype == H3_DTYPE_F16 && !qkv.scale)) {
            size_t f16_need = qkv.n * qkv.k;
            if (out_l.n * out_l.k > f16_need) f16_need = out_l.n * out_l.k;
            if (w1.n * w1.k > f16_need) f16_need = w1.n * w1.k;
            if (w2.n * w2.k > f16_need) f16_need = w2.n * w2.k;
            f16_need *= sizeof(uint16_t);
            if (f16_need > h_f16_stage_bytes) {
                h3_cuda_host_free(h_f16_stage, h_f16_pinned);
                h_f16_stage = h3_cuda_host_alloc(f16_need, &h_f16_pinned);
                h_f16_stage_bytes = h_f16_stage ? f16_need : 0;
                h_f16_pinned = h_f16_stage ? h_f16_pinned : 0;
                if (!h_f16_stage) {
                    block_gpu = 0;
                    gpu_dequant = 0;
                }
            }
            /* Native F16: GPU upcast to resident F32 (quality decode). */
            if (block_gpu) {
                if (!load_linear_f16_gpu(store, &qkv, &bw->d_w_qkv,
                                         &d_f16_scratch, &d_f16_cap,
                                         h_f16_stage, h_f16_stage_bytes,
                                         load_stream, hostreg, error,
                                         error_size) ||
                    !load_linear_f16_gpu(store, &out_l, &bw->d_w_out,
                                         &d_f16_scratch, &d_f16_cap,
                                         h_f16_stage, h_f16_stage_bytes,
                                         load_stream, hostreg, error,
                                         error_size) ||
                    !load_linear_f16_gpu(store, &w1, &bw->d_w_w1,
                                         &d_f16_scratch, &d_f16_cap,
                                         h_f16_stage, h_f16_stage_bytes,
                                         load_stream, hostreg, error,
                                         error_size) ||
                    !load_linear_f16_gpu(store, &w2, &bw->d_w_w2,
                                         &d_f16_scratch, &d_f16_cap,
                                         h_f16_stage, h_f16_stage_bytes,
                                         load_stream, hostreg, error,
                                         error_size)) {
                if (load_stream) cudaStreamSynchronize(load_stream);
                block_gpu = 0;
                gpu_dequant = 0;
                {
                    static int logged_f16;
                    if (!logged_f16) {
                        fprintf(stderr,
                                "vae_f16_gpu_load: fail block=%u %s (fallback "
                                "host dequant)\n",
                                b, error && error[0] ? error : "unknown");
                        fflush(stderr);
                        logged_f16 = 1;
                    }
                }
                if (bw->d_w_qkv) {
                    cudaFree(bw->d_w_qkv);
                    bw->d_w_qkv = NULL;
                }
                if (bw->d_w_out) {
                    cudaFree(bw->d_w_out);
                    bw->d_w_out = NULL;
                }
                if (bw->d_w_w1) {
                    cudaFree(bw->d_w_w1);
                    bw->d_w_w1 = NULL;
                }
                if (bw->d_w_w2) {
                    cudaFree(bw->d_w_w2);
                    bw->d_w_w2 = NULL;
                }
            } else {
                w->native_float_w = 1;
            }
            }
        } else if (block_gpu &&
                   (qkv.weight->dtype != H3_DTYPE_I8 || !qkv.scale)) {
            /* BF16/F32 native: host convert (rare). */
            block_gpu = 0;
            gpu_dequant = 0;
        }
        if (block_gpu && !bw->d_w_qkv) {
            if (!dequant_linear_gpu(store, &qkv, &bw->d_w_qkv, d_i8,
                                    i8_stage_bytes, d_scale, scale_cap,
                                    h_i8_stage, i8_stage_bytes, h_scale_stage,
                                    scale_stage_bytes, load_stream, hostreg,
                                    error, error_size) ||
                !dequant_linear_gpu(store, &out_l, &bw->d_w_out, d_i8,
                                    i8_stage_bytes, d_scale, scale_cap,
                                    h_i8_stage, i8_stage_bytes, h_scale_stage,
                                    scale_stage_bytes, load_stream, hostreg,
                                    error, error_size) ||
                !dequant_linear_gpu(store, &w1, &bw->d_w_w1, d_i8,
                                    i8_stage_bytes, d_scale, scale_cap,
                                    h_i8_stage, i8_stage_bytes, h_scale_stage,
                                    scale_stage_bytes, load_stream, hostreg,
                                    error, error_size) ||
                !dequant_linear_gpu(store, &w2, &bw->d_w_w2, d_i8,
                                    i8_stage_bytes, d_scale, scale_cap,
                                    h_i8_stage, i8_stage_bytes, h_scale_stage,
                                    scale_stage_bytes, load_stream, hostreg,
                                    error, error_size)) {
                if (load_stream) cudaStreamSynchronize(load_stream);
                block_gpu = 0;
                gpu_dequant = 0;
                if (bw->d_w_qkv) {
                    cudaFree(bw->d_w_qkv);
                    bw->d_w_qkv = NULL;
                }
                if (bw->d_w_out) {
                    cudaFree(bw->d_w_out);
                    bw->d_w_out = NULL;
                }
                if (bw->d_w_w1) {
                    cudaFree(bw->d_w_w1);
                    bw->d_w_w1 = NULL;
                }
                if (bw->d_w_w2) {
                    cudaFree(bw->d_w_w2);
                    bw->d_w_w2 = NULL;
                }
            }
        }
        if (!block_gpu) {
            bw->w_qkv = h3_vae_int8_dequant_f32(store, &qkv, error, error_size);
            bw->w_out =
                h3_vae_int8_dequant_f32(store, &out_l, error, error_size);
            bw->w_w1 = h3_vae_int8_dequant_f32(store, &w1, error, error_size);
            bw->w_w2 = h3_vae_int8_dequant_f32(store, &w2, error, error_size);
        }
        if (!bw->norm1 || !bw->norm2 || !bw->scale1 || !bw->scale2 ||
            (!block_gpu &&
             (!bw->w_qkv || !bw->w_out || !bw->w_w1 || !bw->w_w2)) ||
            (block_gpu &&
             (!bw->d_w_qkv || !bw->d_w_out || !bw->d_w_w1 || !bw->d_w_w2))) {
            fail(error, error_size, "vae weight dequant failed");
            goto fail;
        }
        t_i8_ms += now_ms() - tb0;
        tb0 = now_ms();
        if (!load_bias(store, &qkv, &bw->b_qkv, error, error_size) ||
            !load_bias(store, &out_l, &bw->b_out, error, error_size) ||
            !load_bias(store, &w1, &bw->b_w1, error, error_size) ||
            !load_bias(store, &w2, &bw->b_w2, error, error_size))
            goto fail;
        t_bias_ms += now_ms() - tb0;
        if (want_dev) {
            tb0 = now_ms();
            if ((block_gpu ||
                 (upload_w(bw->w_qkv, bw->n_qkv * bw->k_qkv, &bw->d_w_qkv) &&
                  upload_w(bw->w_out, bw->n_out * bw->k_out, &bw->d_w_out) &&
                  upload_w(bw->w_w1, bw->n_w1 * bw->k_w1, &bw->d_w_w1) &&
                  upload_w(bw->w_w2, bw->n_w2 * bw->k_w2, &bw->d_w_w2))) &&
                upload_w(bw->norm1, hidden, &bw->d_norm1) &&
                upload_w(bw->norm2, hidden, &bw->d_norm2) &&
                upload_w(bw->scale1, hidden, &bw->d_scale1) &&
                upload_w(bw->scale2, hidden, &bw->d_scale2) &&
                (!bw->b_qkv || upload_w(bw->b_qkv, bw->n_qkv, &bw->d_b_qkv)) &&
                (!bw->b_out || upload_w(bw->b_out, bw->n_out, &bw->d_b_out)) &&
                (!bw->b_w1 || upload_w(bw->b_w1, bw->n_w1, &bw->d_b_w1)) &&
                (!bw->b_w2 || upload_w(bw->b_w2, bw->n_w2, &bw->d_b_w2))) {
                w->device_resident = 1;
                /* Host W no longer needed for decode — free to save RAM. */
                free(bw->w_qkv);
                bw->w_qkv = NULL;
                free(bw->w_out);
                bw->w_out = NULL;
                free(bw->w_w1);
                bw->w_w1 = NULL;
                free(bw->w_w2);
                bw->w_w2 = NULL;
            } else {
                /* Best-effort: keep host weights if VRAM tight. */
                if (bw->d_w_qkv) {
                    cudaFree(bw->d_w_qkv);
                    bw->d_w_qkv = NULL;
                }
                if (bw->d_w_out) {
                    cudaFree(bw->d_w_out);
                    bw->d_w_out = NULL;
                }
                if (bw->d_w_w1) {
                    cudaFree(bw->d_w_w1);
                    bw->d_w_w1 = NULL;
                }
                if (bw->d_w_w2) {
                    cudaFree(bw->d_w_w2);
                    bw->d_w_w2 = NULL;
                }
                if (bw->d_norm1) {
                    cudaFree(bw->d_norm1);
                    bw->d_norm1 = NULL;
                }
                if (bw->d_norm2) {
                    cudaFree(bw->d_norm2);
                    bw->d_norm2 = NULL;
                }
                if (bw->d_scale1) {
                    cudaFree(bw->d_scale1);
                    bw->d_scale1 = NULL;
                }
                if (bw->d_scale2) {
                    cudaFree(bw->d_scale2);
                    bw->d_scale2 = NULL;
                }
                if (bw->d_b_qkv) {
                    cudaFree(bw->d_b_qkv);
                    bw->d_b_qkv = NULL;
                }
                if (bw->d_b_out) {
                    cudaFree(bw->d_b_out);
                    bw->d_b_out = NULL;
                }
                if (bw->d_b_w1) {
                    cudaFree(bw->d_b_w1);
                    bw->d_b_w1 = NULL;
                }
                if (bw->d_b_w2) {
                    cudaFree(bw->d_b_w2);
                    bw->d_b_w2 = NULL;
                }
                want_dev = 0;
                gpu_dequant = 0;
            }
            t_h2d_ms += now_ms() - tb0;
        }
    }
    if (load_stream) cudaStreamSynchronize(load_stream);
    t_done_ms = now_ms();
    if (load_ms) *load_ms = t_done_ms - t0_ms;
    fprintf(stderr,
            "vae_weights_load: cache=%s mmap=%d gpu_dequant=%d hostreg=%d "
            "native_f16=%d blocks=%u device=%d find_ms=%.1f f32_ms=%.1f "
            "dequant_ms=%.1f bias_ms=%.1f h2d_ms=%.1f total_ms=%.1f\n",
            cache_enabled ? "fill" : "off", store->map ? 1 : 0,
            gpu_dequant, hostreg, w->native_float_w, n_blocks,
            w->device_resident ? 1 : 0, t_find_ms, t_f32_ms, t_i8_ms,
            t_bias_ms, t_h2d_ms, t_done_ms - t0_ms);
    if (d_i8) cudaFree(d_i8);
    if (d_scale) cudaFree(d_scale);
    if (d_f16_scratch) cudaFree(d_f16_scratch);
    if (h_i8_stage) h3_cuda_host_free(h_i8_stage, h_i8_pinned);
    if (h_scale_stage) h3_cuda_host_free(h_scale_stage, h_scale_pinned);
    if (h_f16_stage) h3_cuda_host_free(h_f16_stage, h_f16_pinned);
    if (load_stream) cudaStreamDestroy(load_stream);
    /* Keep mmap pin across decode/repeat. Unregister in hostreg_release. */
    (void)ensure_vae_decode_heads(store, w, error, error_size);
    if (cache_enabled) {
        vae_weight_cache_drop();
        w->cache_pinned = 1;
        w->cache_refs = 1;
        snprintf(w->cache_path, sizeof(w->cache_path), "%s", store->path);
        g_vae_cached_weights = w;
    }
    *out = w;
    return 1;
fail:
    if (load_stream) {
        cudaStreamSynchronize(load_stream);
        cudaStreamDestroy(load_stream);
    }
    if (d_i8) cudaFree(d_i8);
    if (d_scale) cudaFree(d_scale);
    if (d_f16_scratch) cudaFree(d_f16_scratch);
    if (h_i8_stage) h3_cuda_host_free(h_i8_stage, h_i8_pinned);
    if (h_scale_stage) h3_cuda_host_free(h_scale_stage, h_scale_pinned);
    if (h_f16_stage) h3_cuda_host_free(h_f16_stage, h_f16_pinned);
    if (hostreg_owned) h3_vae_int8_cuda_hostreg_release();
    h3_vae_int8_cuda_weights_free(w);
    return 0;
}
