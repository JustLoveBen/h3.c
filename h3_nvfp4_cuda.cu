#include "h3_nvfp4_cuda.h"
#include "h3_nvfp4.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Touch TE mmap pages so packed H2D is not page-fault / reclaim thrash
 * (5090 cafe warm saw upload_ms spike to ~50s when DiT/VAE evicted TE). */
static double nvfp4_prefault_mmap(const uint8_t *map, size_t map_len) {
    struct timespec t0, t1;
    size_t stride, i;
    volatile unsigned acc = 0;
    const char *env;
    if (!map || map_len == 0) return 0.0;
    env = getenv("H3_TE_PREFAULT");
    if (env && env[0] == '0') return 0.0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    /* 2MiB stride: kernel readahead fills the gap under MADV_SEQUENTIAL. */
    stride = (size_t)2 << 20;
    for (i = 0; i < map_len; i += stride) acc += map[i];
    if (map_len > 0) acc += map[map_len - 1];
    (void)acc;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    return ((double)t1.tv_sec * 1000.0 + (double)t1.tv_nsec / 1e6) -
           ((double)t0.tv_sec * 1000.0 + (double)t0.tv_nsec / 1e6);
}

static h3_cuda *g_te_cuda = NULL;
static int g_forced_host = 0;
static void (*g_after_packed_hook)(void *) = NULL;
static void *g_after_packed_user = NULL;

/* Growable host cache: unswizzled scales + pre_quant + packed weights
 * (CFG 2nd encode hits). A tiny device d_w LRU cannot cover 50L×7 unique
 * linears; host packed cache skips mmap→staging memcpy on repeat. */
typedef struct {
    int valid;
    size_t weight_off;
    size_t scale_off;
    size_t scale2_off;
    size_t pq_off;
    size_t n;
    size_t k;
    size_t groups;
    float scale2;
    uint8_t *logical_scales;
    uint8_t *packed; /* NVFP4 packed [N, K/2]; NULL only if OOM on cache fill */
    size_t packed_bytes;
    int packed_pinned; /* cudaHostRegister so H2D is not pageable */
    float *pre_quant; /* NULL if this linear has no AWQ scale */
} nvfp4_host_scale_slot;

static nvfp4_host_scale_slot *g_host_scales = NULL;
static int g_host_scales_n = 0;
static int g_host_scales_cap = 0;

/* Persistent device workspaces (grow-only; freed on cache_clear / disable). */
static uint8_t *g_d_packed = NULL;
static uint8_t *g_d_scales = NULL;
static float *g_d_w = NULL;
static float *g_d_x = NULL;
static float *g_d_y = NULL;
static float *g_d_pq = NULL;
static size_t g_d_packed_cap = 0;
static size_t g_d_scales_cap = 0;
static size_t g_d_w_cap = 0;
static size_t g_d_x_cap = 0;
static size_t g_d_y_cap = 0;
static size_t g_d_pq_cap = 0;
static const float *g_x_host = NULL;
static size_t g_x_host_bytes = 0;

static uint8_t *g_h_blocked = NULL;
static uint8_t *g_h_logical = NULL;
static uint16_t *g_h_pq_raw = NULL;
static float *g_h_pq_f32 = NULL;
static uint8_t *g_h_packed_stage = NULL;
static size_t g_h_blocked_cap = 0;
static size_t g_h_logical_cap = 0;
static size_t g_h_pq_raw_cap = 0;
static size_t g_h_pq_f32_cap = 0;
static size_t g_h_packed_stage_cap = 0;
static int g_h_packed_stage_pinned = 0;

/* A1: packed NVFP4 + unswizzled scales (+ AWQ pre_quant) stay on GPU for the
 * TE encode window. generate() already disables TE before DiT, so this does
 * not fight the INT8 weight window. H3_TE_PACKED_RESIDENT=0 disables. */
typedef struct {
    int valid;
    size_t weight_off;
    size_t n, k, groups, group_size;
    size_t packed_bytes, scale_bytes, pq_elems;
    uint8_t *d_packed;
    uint8_t *d_scales;
    float *d_pq;
    float scale2;
} nvfp4_dev_linear;

static nvfp4_dev_linear *g_dev_lin = NULL;
static int g_dev_lin_n = 0;
static uint8_t *g_d_te_w_arena = NULL;
static size_t g_d_te_w_arena_bytes = 0;

static float *g_d_ln_in = NULL;
static float *g_d_ln_post = NULL;
static float *g_d_qn_all = NULL;
static float *g_d_kn_all = NULL;
static size_t g_d_ln_in_cap = 0;
static size_t g_d_ln_post_cap = 0;
static size_t g_d_qn_all_cap = 0;
static size_t g_d_kn_all_cap = 0;
static unsigned g_ln_cached_layers = 0;

static void fail(char *error, size_t error_size, const char *msg) {
    if (!error || !error_size) return;
    snprintf(error, error_size, "%s", msg ? msg : "h3_nvfp4_cuda error");
}

static int nvfp4_cache_disabled(void) {
    const char *env = getenv("H3_NVFP4_DISABLE_DEV_CACHE");
    return env && env[0] == '1';
}

static int nvfp4_want_packed_resident(void) {
    const char *env = getenv("H3_TE_PACKED_RESIDENT");
    if (env && env[0] == '0') return 0;
    return 1;
}

static size_t nvfp4_align256(size_t n) {
    return (n + (size_t)255) & ~(size_t)255;
}

static int g_map_hostreg = 0;
static int g_map_hostreg_tried = 0;
static const uint8_t *g_map_hostreg_ptr = NULL;

static void nvfp4_try_register_map(const h3_nvfp4_store *store) {
    cudaError_t st;
    const char *force;
    static int logged = 0;
    if (!store || !store->map || store->map_len == 0) return;
    if (g_map_hostreg && g_map_hostreg_ptr == store->map) return;
    if (g_map_hostreg_tried && g_map_hostreg_ptr == store->map) return;
    if (g_map_hostreg && g_map_hostreg_ptr && g_map_hostreg_ptr != store->map) {
        cudaHostUnregister((void *)g_map_hostreg_ptr);
        g_map_hostreg = 0;
        g_map_hostreg_ptr = NULL;
        g_map_hostreg_tried = 0;
    }
    /* 15GB TE mmap cannot pin on 4090 hosts; retrying per-linear costs ~2s.
     * H3_TE_MMAP_HOSTREG=1 forces an attempt even for huge maps. */
    force = getenv("H3_TE_MMAP_HOSTREG");
    if (!(force && force[0] == '1') && store->map_len > (256u << 20)) {
        g_map_hostreg_tried = 1;
        g_map_hostreg_ptr = store->map;
        if (!logged) {
            fprintf(stderr, "te_packed: mmap_hostreg=skip bytes=%zu\n",
                    store->map_len);
            fflush(stderr);
            logged = 1;
        }
        return;
    }
    g_map_hostreg_tried = 1;
    g_map_hostreg_ptr = store->map;
    st = cudaHostRegister((void *)store->map, store->map_len,
                          cudaHostRegisterDefault);
    if (st == cudaSuccess) {
        g_map_hostreg = 1;
    } else {
        (void)cudaGetLastError(); /* clear sticky */
    }
    if (!logged) {
        fprintf(stderr, "te_packed: mmap_hostreg=%d bytes=%zu\n", g_map_hostreg,
                store->map_len);
        fflush(stderr);
        logged = 1;
    }
}

static const uint8_t *nvfp4_mmap_packed(const h3_nvfp4_store *store,
                                        const h3_nvfp4_linear *lin,
                                        size_t packed_bytes) {
    size_t off;
    if (!g_map_hostreg || !store || !store->map || !lin || !lin->weight)
        return NULL;
    off = (size_t)lin->weight->file_offset;
    if (off + packed_bytes > store->map_len) return NULL;
    return store->map + off;
}

static int nvfp4_h2d(void *dst, const void *src, size_t bytes, int src_pinned,
                     char *error, size_t error_size, const char *what) {
    cudaError_t st;
    if (!dst || !src || bytes == 0) {
        fail(error, error_size, what);
        return 0;
    }
    if (src_pinned)
        st = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, 0);
    else
        st = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
    if (st != cudaSuccess) {
        fail(error, error_size, what);
        return 0;
    }
    return 1;
}

static void nvfp4_host_scale_slot_reset(nvfp4_host_scale_slot *slot);
static void nvfp4_act_workspace_free(void);
static void nvfp4_dev_lin_free(void);

static void nvfp4_try_pin_packed(nvfp4_host_scale_slot *slot) {
    if (!slot || !slot->packed || slot->packed_pinned || slot->packed_bytes == 0)
        return;
    if (cudaHostRegister(slot->packed, slot->packed_bytes,
                         cudaHostRegisterDefault) == cudaSuccess)
        slot->packed_pinned = 1;
}

static void nvfp4_host_scale_slot_reset(nvfp4_host_scale_slot *slot) {
    if (!slot) return;
    if (slot->packed && slot->packed_pinned)
        cudaHostUnregister(slot->packed);
    free(slot->logical_scales);
    free(slot->packed);
    free(slot->pre_quant);
    memset(slot, 0, sizeof(*slot));
}

static void nvfp4_workspace_free(void) {
    if (g_d_packed) cudaFree(g_d_packed);
    if (g_d_scales) cudaFree(g_d_scales);
    if (g_d_w) cudaFree(g_d_w);
    if (g_d_x) cudaFree(g_d_x);
    if (g_d_y) cudaFree(g_d_y);
    if (g_d_pq) cudaFree(g_d_pq);
    g_d_packed = NULL;
    g_d_scales = NULL;
    g_d_w = NULL;
    g_d_x = NULL;
    g_d_y = NULL;
    g_d_pq = NULL;
    g_d_packed_cap = g_d_scales_cap = g_d_w_cap = 0;
    g_d_x_cap = g_d_y_cap = g_d_pq_cap = 0;
    g_x_host = NULL;
    g_x_host_bytes = 0;
    free(g_h_blocked);
    free(g_h_logical);
    free(g_h_pq_raw);
    free(g_h_pq_f32);
    if (g_h_packed_stage) {
        if (g_h_packed_stage_pinned)
            cudaFreeHost(g_h_packed_stage);
        else
            free(g_h_packed_stage);
    }
    g_h_blocked = NULL;
    g_h_logical = NULL;
    g_h_pq_raw = NULL;
    g_h_pq_f32 = NULL;
    g_h_packed_stage = NULL;
    g_h_blocked_cap = g_h_logical_cap = g_h_pq_raw_cap = 0;
    g_h_pq_f32_cap = 0;
    g_h_packed_stage_cap = 0;
    g_h_packed_stage_pinned = 0;
    nvfp4_act_workspace_free();
    nvfp4_dev_lin_free();
    if (g_map_hostreg && g_map_hostreg_ptr) {
        cudaHostUnregister((void *)g_map_hostreg_ptr);
        g_map_hostreg = 0;
        g_map_hostreg_ptr = NULL;
    }
}

extern "C" void h3_nvfp4_cuda_cache_clear(void) {
    int i;
    for (i = 0; i < g_host_scales_n; i++)
        nvfp4_host_scale_slot_reset(&g_host_scales[i]);
    free(g_host_scales);
    g_host_scales = NULL;
    g_host_scales_n = 0;
    g_host_scales_cap = 0;
    nvfp4_workspace_free();
}

static int ensure_packed_stage(size_t need, char *error, size_t error_size) {
    if (g_h_packed_stage && g_h_packed_stage_cap >= need) return 1;
    if (g_h_packed_stage) {
        if (g_h_packed_stage_pinned)
            cudaFreeHost(g_h_packed_stage);
        else
            free(g_h_packed_stage);
        g_h_packed_stage = NULL;
        g_h_packed_stage_cap = 0;
        g_h_packed_stage_pinned = 0;
    }
    if (cudaMallocHost((void **)&g_h_packed_stage, need) == cudaSuccess) {
        g_h_packed_stage_pinned = 1;
        g_h_packed_stage_cap = need;
        return 1;
    }
    g_h_packed_stage = (uint8_t *)malloc(need);
    if (!g_h_packed_stage) {
        fail(error, error_size, "OOM nvfp4 packed staging");
        return 0;
    }
    g_h_packed_stage_pinned = 0;
    g_h_packed_stage_cap = need;
    return 1;
}

static int ensure_host_bytes(void **ptr, size_t *cap, size_t need, char *error,
                             size_t error_size) {
    void *p;
    if (!ptr || !cap) return 0;
    if (*ptr && *cap >= need) return 1;
    p = realloc(*ptr, need);
    if (!p) {
        fail(error, error_size, "OOM nvfp4 cuda host workspace");
        return 0;
    }
    *ptr = p;
    *cap = need;
    return 1;
}

static int ensure_dev_bytes(void **ptr, size_t *cap, size_t need, char *error,
                            size_t error_size) {
    cudaError_t st;
    if (!ptr || !cap) return 0;
    if (*ptr && *cap >= need) return 1;
    if (*ptr) {
        cudaFree(*ptr);
        *ptr = NULL;
        *cap = 0;
    }
    st = cudaMalloc(ptr, need);
    if (st != cudaSuccess) {
        fail(error, error_size, "cudaMalloc TE workspace");
        *ptr = NULL;
        return 0;
    }
    *cap = need;
    return 1;
}

static nvfp4_host_scale_slot *nvfp4_host_find(const h3_nvfp4_linear *lin) {
    int i;
    if (!lin || !lin->weight || !lin->scale || !lin->scale_2) return NULL;
    for (i = 0; i < g_host_scales_n; i++) {
        nvfp4_host_scale_slot *slot = &g_host_scales[i];
        if (!slot->valid) continue;
        if (slot->weight_off == (size_t)lin->weight->file_offset &&
            slot->scale_off == (size_t)lin->scale->file_offset &&
            slot->scale2_off == (size_t)lin->scale_2->file_offset &&
            slot->n == lin->n && slot->k == lin->k &&
            slot->groups == lin->groups)
            return slot;
    }
    return NULL;
}

static nvfp4_host_scale_slot *nvfp4_host_add(void) {
    nvfp4_host_scale_slot *p;
    if (g_host_scales_n == g_host_scales_cap) {
        int ncap = g_host_scales_cap ? g_host_scales_cap * 2 : 32;
        p = (nvfp4_host_scale_slot *)realloc(
            g_host_scales, (size_t)ncap * sizeof(*g_host_scales));
        if (!p) return NULL;
        memset(p + g_host_scales_cap, 0,
               (size_t)(ncap - g_host_scales_cap) * sizeof(*p));
        g_host_scales = p;
        g_host_scales_cap = ncap;
    }
    p = &g_host_scales[g_host_scales_n++];
    memset(p, 0, sizeof(*p));
    return p;
}

static float nvfp4_bf16_bits_to_f32(uint16_t bits) {
    union {
        uint32_t u;
        float f;
    } v;
    v.u = ((uint32_t)bits) << 16;
    return v.f;
}

/* OCP FP4 E2M1 + FP8 E4M3FN tables (match host helpers). */
__device__ __constant__ float kFp4E2M1[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

__device__ static float f8e4m3_to_f32(uint8_t bits) {
    const int sign = (bits >> 7) & 1;
    const int exp = (bits >> 3) & 0xF;
    const int mant = bits & 0x7;
    float value;
    if (exp == 0xF && mant == 0x7) {
        value = nanf("");
    } else if (exp == 0) {
        value = (mant == 0) ? 0.0f : ldexpf((float)mant, 1 - 7 - 3);
    } else {
        value = ldexpf(1.0f + (float)mant / 8.0f, exp - 7);
    }
    return sign ? -value : value;
}

__global__ void nvfp4_dequant_kernel(const uint8_t *packed,
                                     const uint8_t *scales_f8, float scale2,
                                     size_t n, size_t k, size_t groups,
                                     size_t group_size, float *out) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = n * (k / 2);
    size_t row, j, col0, col1, g0, g1;
    uint8_t byte;
    if (idx >= total) return;
    row = idx / (k / 2);
    j = idx % (k / 2);
    byte = packed[idx];
    col0 = j * 2;
    col1 = col0 + 1;
    g0 = col0 / group_size;
    g1 = col1 / group_size;
    {
        float s0 = f8e4m3_to_f32(scales_f8[row * groups + g0]) * scale2;
        float s1 = f8e4m3_to_f32(scales_f8[row * groups + g1]) * scale2;
        /* Comfy NVFP4 hi_first: high nibble → even K, low → odd. */
        out[row * k + col0] = kFp4E2M1[(byte >> 4) & 0xF] * s0;
        out[row * k + col1] = kFp4E2M1[byte & 0xF] * s1;
    }
}

/* AWQ smooth: x[row, k] *= pre_quant_scale[k] (same as host nvfp4_apply_pre_quant). */
__global__ void nvfp4_pre_quant_kernel(float *x, const float *scale, size_t m,
                                       size_t k) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * k) return;
    x[idx] *= scale[idx % k];
}

/* Device dequant + cuBLAS. Activations live in TE workspaces across q/k/v
 * (same host x pointer skips H2D). Packed H2D from host cache or mmap;
 * scales+packed cached on host after first touch. */
static int linear_cuda(void *ctx, const h3_nvfp4_store *store,
                       const h3_nvfp4_linear *lin, const float *x, size_t m,
                       float *y, char *error, size_t error_size) {
    h3_cuda *cuda = (h3_cuda *)ctx;
    size_t k_half, packed_bytes, scale_bytes, x_bytes, y_bytes, w_bytes;
    nvfp4_host_scale_slot *slot = NULL;
    const uint8_t *logical_scales = NULL;
    const uint8_t *packed_src = NULL;
    float scale2 = 0.0f;
    const float *pq_host = NULL;
    cudaError_t st;
    int threads = 256;
    int blocks;
    size_t total_nibbles;
    int reuse_x;

    if (!cuda || !store || !lin || !x || !y || m == 0) {
        fail(error, error_size, "invalid nvfp4 cuda linear args");
        return 0;
    }
    if (lin->k < 2 || (lin->k % 2) != 0) {
        fail(error, error_size, "nvfp4 k not even");
        return 0;
    }
    k_half = lin->k / 2;
    packed_bytes = lin->n * k_half;
    scale_bytes = lin->n * lin->groups;
    x_bytes = m * lin->k * sizeof(float);
    y_bytes = m * lin->n * sizeof(float);
    w_bytes = lin->n * lin->k * sizeof(float);

    nvfp4_try_register_map(store);

    if (!nvfp4_cache_disabled()) slot = nvfp4_host_find(lin);
    packed_src = nvfp4_mmap_packed(store, lin, packed_bytes);
    if (slot && slot->packed && slot->packed_bytes == packed_bytes) {
        logical_scales = slot->logical_scales;
        scale2 = slot->scale2;
        pq_host = slot->pre_quant;
        nvfp4_try_pin_packed(slot);
        if (!packed_src) packed_src = slot->packed;
    } else {
        if (!ensure_host_bytes((void **)&g_h_blocked, &g_h_blocked_cap,
                               scale_bytes, error, error_size) ||
            !ensure_host_bytes((void **)&g_h_logical, &g_h_logical_cap,
                               scale_bytes, error, error_size))
            return 0;
        if (!h3_nvfp4_store_copy(store, (size_t)lin->scale->file_offset,
                                 g_h_blocked, scale_bytes, error,
                                 error_size) ||
            !h3_nvfp4_store_copy(store, (size_t)lin->scale_2->file_offset,
                                 &scale2, sizeof(float), error, error_size))
            return 0;
        if (!h3_nvfp4_unswizzle_scales_u8(g_h_blocked, lin->n, lin->groups,
                                          g_h_logical, error, error_size))
            return 0;
        logical_scales = g_h_logical;

        if (lin->pre_quant_scale) {
            size_t i, k = lin->k;
            if (!ensure_host_bytes((void **)&g_h_pq_raw, &g_h_pq_raw_cap,
                                   k * sizeof(uint16_t), error, error_size) ||
                !ensure_host_bytes((void **)&g_h_pq_f32, &g_h_pq_f32_cap,
                                   k * sizeof(float), error, error_size))
                return 0;
            if (!h3_nvfp4_store_copy(
                    store, (size_t)lin->pre_quant_scale->file_offset, g_h_pq_raw,
                    k * sizeof(uint16_t), error, error_size))
                return 0;
            for (i = 0; i < k; i++)
                g_h_pq_f32[i] = nvfp4_bf16_bits_to_f32(g_h_pq_raw[i]);
            pq_host = g_h_pq_f32;
        }

        /* mmap_hostreg: H2D packed from the file map (no 12GB host copy).
         * Fallback: host packed cache + pin. */
        if (!packed_src && !nvfp4_cache_disabled()) {
            if (!slot) slot = nvfp4_host_add();
            if (!slot) {
                fail(error, error_size, "OOM nvfp4 host scale cache");
                return 0;
            }
            if (!slot->packed && !g_map_hostreg) {
                slot->packed = (uint8_t *)malloc(packed_bytes);
                if (slot->packed) {
                    if (store->map && store->map_len > 0) {
                        size_t off = (size_t)lin->weight->file_offset;
                        if (off + packed_bytes > store->map_len) {
                            fail(error, error_size, "packed H2D past mmap end");
                            return 0;
                        }
                        memcpy(slot->packed, store->map + off, packed_bytes);
                    } else if (!h3_nvfp4_store_copy(
                                   store, (size_t)lin->weight->file_offset,
                                   slot->packed, packed_bytes, error,
                                   error_size)) {
                        free(slot->packed);
                        slot->packed = NULL;
                        return 0;
                    }
                    slot->packed_bytes = packed_bytes;
                }
            }
            if (slot->packed && slot->packed_bytes == packed_bytes) {
                nvfp4_try_pin_packed(slot);
                packed_src = slot->packed;
            }
        }
        if (!packed_src) {
            if (!ensure_packed_stage(packed_bytes, error, error_size))
                return 0;
            if (store->map && store->map_len > 0) {
                size_t off = (size_t)lin->weight->file_offset;
                if (off + packed_bytes > store->map_len) {
                    fail(error, error_size, "packed H2D past mmap end");
                    return 0;
                }
                memcpy(g_h_packed_stage, store->map + off, packed_bytes);
            } else if (!h3_nvfp4_store_copy(
                           store, (size_t)lin->weight->file_offset,
                           g_h_packed_stage, packed_bytes, error,
                           error_size)) {
                return 0;
            }
            packed_src = g_h_packed_stage;
        }

        if (!nvfp4_cache_disabled()) {
            if (!slot) slot = nvfp4_host_add();
            if (!slot) {
                fail(error, error_size, "OOM nvfp4 host scale cache");
                return 0;
            }
            if (!slot->logical_scales) {
                slot->logical_scales = (uint8_t *)malloc(scale_bytes);
                if (!slot->logical_scales) {
                    fail(error, error_size, "OOM nvfp4 host scale copy");
                    nvfp4_host_scale_slot_reset(slot);
                    if (g_host_scales_n > 0) g_host_scales_n--;
                    return 0;
                }
                memcpy(slot->logical_scales, g_h_logical, scale_bytes);
            }
            if (!slot->packed && !g_map_hostreg) {
                slot->packed = (uint8_t *)malloc(packed_bytes);
                if (slot->packed) {
                    memcpy(slot->packed, packed_src, packed_bytes);
                    slot->packed_bytes = packed_bytes;
                    nvfp4_try_pin_packed(slot);
                    packed_src = slot->packed;
                }
            }
            slot->valid = 1;
            slot->weight_off = (size_t)lin->weight->file_offset;
            slot->scale_off = (size_t)lin->scale->file_offset;
            slot->scale2_off = (size_t)lin->scale_2->file_offset;
            slot->n = lin->n;
            slot->k = lin->k;
            slot->groups = lin->groups;
            slot->scale2 = scale2;
            logical_scales = slot->logical_scales;
            if (lin->pre_quant_scale && !slot->pre_quant) {
                size_t k = lin->k;
                slot->pre_quant = (float *)malloc(k * sizeof(float));
                if (!slot->pre_quant) {
                    fail(error, error_size, "OOM nvfp4 host pre_quant");
                    nvfp4_host_scale_slot_reset(slot);
                    if (g_host_scales_n > 0) g_host_scales_n--;
                    return 0;
                }
                slot->pq_off = (size_t)lin->pre_quant_scale->file_offset;
                memcpy(slot->pre_quant, g_h_pq_f32, k * sizeof(float));
                pq_host = slot->pre_quant;
            } else if (slot->pre_quant) {
                pq_host = slot->pre_quant;
            }
        }
    }

    if (!packed_src) {
        fail(error, error_size, "missing packed weights");
        return 0;
    }

    {
        size_t old_x_cap = g_d_x_cap;
        if (!ensure_dev_bytes((void **)&g_d_packed, &g_d_packed_cap, packed_bytes,
                              error, error_size) ||
            !ensure_dev_bytes((void **)&g_d_scales, &g_d_scales_cap, scale_bytes,
                              error, error_size) ||
            !ensure_dev_bytes((void **)&g_d_w, &g_d_w_cap, w_bytes, error,
                              error_size) ||
            !ensure_dev_bytes((void **)&g_d_x, &g_d_x_cap, x_bytes, error,
                              error_size) ||
            !ensure_dev_bytes((void **)&g_d_y, &g_d_y_cap, y_bytes, error,
                              error_size)) {
            g_x_host = NULL;
            return 0;
        }
        if (g_d_x_cap != old_x_cap) {
            g_x_host = NULL;
            g_x_host_bytes = 0;
        }
    }

    {
        int packed_pinned = 0;
        if (g_map_hostreg && packed_src && store && store->map &&
            packed_src >= store->map &&
            packed_src < store->map + store->map_len)
            packed_pinned = 1;
        else if (slot && slot->packed_pinned && packed_src == slot->packed)
            packed_pinned = 1;
        else if (packed_src == g_h_packed_stage && g_h_packed_stage_pinned)
            packed_pinned = 1;
        if (!nvfp4_h2d(g_d_packed, packed_src, packed_bytes, packed_pinned,
                       error, error_size, "H2D packed") ||
            !nvfp4_h2d(g_d_scales, logical_scales, scale_bytes, 0, error,
                       error_size, "H2D scales"))
            return 0;
    }

    total_nibbles = lin->n * k_half;
    blocks = (int)((total_nibbles + (size_t)threads - 1) / (size_t)threads);
    nvfp4_dequant_kernel<<<blocks, threads>>>(g_d_packed, g_d_scales, scale2,
                                              lin->n, lin->k, lin->groups,
                                              lin->group_size, g_d_w);
    st = cudaGetLastError();
    if (st != cudaSuccess) {
        fail(error, error_size, "nvfp4 dequant kernel");
        return 0;
    }

    reuse_x = (g_x_host == x && g_x_host_bytes == x_bytes);
    if (!reuse_x) {
        st = cudaMemcpy(g_d_x, x, x_bytes, cudaMemcpyHostToDevice);
        if (st != cudaSuccess) {
            fail(error, error_size, "H2D x");
            g_x_host = NULL;
            return 0;
        }
        g_x_host = x;
        g_x_host_bytes = x_bytes;
    }

    /* Device-side AWQ pre_quant (o_proj / down_proj). Mutates d_x so the
     * host-pointer reuse key is invalidated. */
    if (lin->pre_quant_scale) {
        if (!pq_host) {
            fail(error, error_size, "missing pre_quant host scale");
            return 0;
        }
        if (!ensure_dev_bytes((void **)&g_d_pq, &g_d_pq_cap,
                              lin->k * sizeof(float), error, error_size))
            return 0;
        st = cudaMemcpy(g_d_pq, pq_host, lin->k * sizeof(float),
                        cudaMemcpyHostToDevice);
        if (st != cudaSuccess) {
            fail(error, error_size, "H2D pre_quant");
            return 0;
        }
        blocks = (int)((m * lin->k + (size_t)threads - 1) / (size_t)threads);
        nvfp4_pre_quant_kernel<<<blocks, threads>>>(g_d_x, g_d_pq, m, lin->k);
        st = cudaGetLastError();
        if (st != cudaSuccess) {
            fail(error, error_size, "nvfp4 pre_quant kernel");
            g_x_host = NULL;
            return 0;
        }
        g_x_host = NULL;
        g_x_host_bytes = 0;
    }

    if (!h3_cuda_gemm_f32_dd(cuda, g_d_x, g_d_w, m, lin->n, lin->k, g_d_y, NULL,
                             error, error_size)) {
        g_x_host = NULL;
        return 0;
    }
    st = cudaMemcpy(y, g_d_y, y_bytes, cudaMemcpyDeviceToHost);
    if (st != cudaSuccess) {
        fail(error, error_size, "D2H y");
        return 0;
    }
    return 1;
}

extern "C" int h3_nvfp4_cuda_enable(h3_cuda *cuda, char *error,
                                    size_t error_size) {
    const char *env = getenv("H3_NVFP4_HOST_GEMM");
    g_forced_host = (env && env[0] == '1');
    if (!cuda) {
        fail(error, error_size, "h3_nvfp4_cuda_enable needs cuda");
        return 0;
    }
    if (g_forced_host) {
        g_te_cuda = NULL;
        h3_nvfp4_cuda_cache_clear();
        h3_nvfp4_set_linear_backend(NULL, NULL);
        return 1;
    }
    g_te_cuda = cuda;
    h3_nvfp4_set_linear_backend(cuda, linear_cuda);
    return 1;
}

extern "C" void h3_nvfp4_cuda_disable(void) {
    h3_nvfp4_cuda_cache_clear();
    g_te_cuda = NULL;
    g_after_packed_hook = NULL;
    g_after_packed_user = NULL;
    h3_nvfp4_set_linear_backend(NULL, NULL);
}

extern "C" void h3_nvfp4_cuda_set_after_packed_hook(void (*fn)(void *),
                                                    void *user) {
    g_after_packed_hook = fn;
    g_after_packed_user = user;
}

extern "C" int h3_nvfp4_cuda_enabled(void) {
    return g_te_cuda != NULL && !g_forced_host;
}

extern "C" int h3_nvfp4_cuda_gqa_causal(const float *q, const float *k,
                                        const float *v, size_t tokens,
                                        size_t q_heads, size_t kv_heads,
                                        size_t head_dim, float *out,
                                        char *error, size_t error_size) {
    char backend[64];
    const char *force = getenv("H3_NVFP4_HOST_GQA");
    static int logged = 0;
    if (force && force[0] == '1') return 0;
    if (!g_te_cuda || g_forced_host) return 0;
    if (!h3_cuda_sdpa_gqa_causal_f16(g_te_cuda, q, k, v, tokens, q_heads,
                                     kv_heads, head_dim,
                                     1.0f / sqrtf((float)head_dim), out, NULL,
                                     backend, sizeof(backend), error,
                                     error_size))
        return 0;
    if (!logged) {
        fprintf(stderr, "te_gqa: backend=%s tokens=%zu q_heads=%zu kv_heads=%zu\n",
                backend, tokens, q_heads, kv_heads);
        fflush(stderr);
        logged = 1;
    }
    return 1;
}

/* ---- Device-resident text TE encode (activations stay on GPU) ---- */

static float *g_d_hidden = NULL;
static float *g_d_normed = NULL;
static float *g_d_q = NULL;
static float *g_d_k = NULL;
static float *g_d_v = NULL;
static float *g_d_attn = NULL;
static float *g_d_tmp = NULL; /* attn_out / down residual */
static float *g_d_gate = NULL;
static float *g_d_up = NULL;
static float *g_d_ffn = NULL;
static float *g_d_nw = NULL; /* RMS / head-norm weights */
static size_t g_d_hidden_cap = 0;
static size_t g_d_normed_cap = 0;
static size_t g_d_q_cap = 0;
static size_t g_d_k_cap = 0;
static size_t g_d_v_cap = 0;
static size_t g_d_attn_cap = 0;
static size_t g_d_tmp_cap = 0;
static size_t g_d_gate_cap = 0;
static size_t g_d_up_cap = 0;
static size_t g_d_ffn_cap = 0;
static size_t g_d_nw_cap = 0;

static void nvfp4_dev_lin_free(void) {
    if (g_d_te_w_arena) cudaFree(g_d_te_w_arena);
    g_d_te_w_arena = NULL;
    g_d_te_w_arena_bytes = 0;
    free(g_dev_lin);
    g_dev_lin = NULL;
    g_dev_lin_n = 0;
}

static void nvfp4_act_workspace_free(void) {
    if (g_d_hidden) cudaFree(g_d_hidden);
    if (g_d_normed) cudaFree(g_d_normed);
    if (g_d_q) cudaFree(g_d_q);
    if (g_d_k) cudaFree(g_d_k);
    if (g_d_v) cudaFree(g_d_v);
    if (g_d_attn) cudaFree(g_d_attn);
    if (g_d_tmp) cudaFree(g_d_tmp);
    if (g_d_gate) cudaFree(g_d_gate);
    if (g_d_up) cudaFree(g_d_up);
    if (g_d_ffn) cudaFree(g_d_ffn);
    if (g_d_nw) cudaFree(g_d_nw);
    if (g_d_ln_in) cudaFree(g_d_ln_in);
    if (g_d_ln_post) cudaFree(g_d_ln_post);
    if (g_d_qn_all) cudaFree(g_d_qn_all);
    if (g_d_kn_all) cudaFree(g_d_kn_all);
    g_d_hidden = g_d_normed = g_d_q = g_d_k = g_d_v = NULL;
    g_d_attn = g_d_tmp = g_d_gate = g_d_up = g_d_ffn = g_d_nw = NULL;
    g_d_ln_in = g_d_ln_post = g_d_qn_all = g_d_kn_all = NULL;
    g_d_hidden_cap = g_d_normed_cap = g_d_q_cap = g_d_k_cap = g_d_v_cap = 0;
    g_d_attn_cap = g_d_tmp_cap = g_d_gate_cap = g_d_up_cap = g_d_ffn_cap = 0;
    g_d_nw_cap = 0;
    g_d_ln_in_cap = g_d_ln_post_cap = g_d_qn_all_cap = g_d_kn_all_cap = 0;
    g_ln_cached_layers = 0;
}

__global__ void nvfp4_head_rms_kernel(float *x, const float *w, size_t tokens,
                                      size_t heads, size_t head_dim, float eps) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = tokens * heads;
    size_t t, h, d;
    float mean, inv;
    float *xr;
    if (idx >= total) return;
    t = idx / heads;
    h = idx % heads;
    xr = x + (t * heads + h) * head_dim;
    mean = 0.0f;
    for (d = 0; d < head_dim; d++) mean += xr[d] * xr[d];
    inv = rsqrtf(mean / (float)head_dim + eps);
    for (d = 0; d < head_dim; d++) xr[d] = xr[d] * inv * w[d];
}

__global__ void nvfp4_rope_text_kernel(float *q, float *k, size_t tokens,
                                       size_t q_heads, size_t kv_heads,
                                       size_t head_dim, float theta) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t half = head_dim / 2;
    size_t total = tokens * (q_heads + kv_heads);
    size_t t, h, i;
    float *x;
    if (idx >= total) return;
    if (idx < tokens * q_heads) {
        t = idx / q_heads;
        h = idx % q_heads;
        x = q + (t * q_heads + h) * head_dim;
    } else {
        size_t j = idx - tokens * q_heads;
        t = j / kv_heads;
        h = j % kv_heads;
        x = k + (t * kv_heads + h) * head_dim;
    }
    for (i = 0; i < half; i++) {
        float freq = 1.0f / powf(theta, (float)(2 * i) / (float)head_dim);
        float ang = (float)t * freq;
        float c = cosf(ang), s = sinf(ang);
        float a = x[i], b = x[i + half];
        x[i] = a * c - b * s;
        x[i + half] = a * s + b * c;
    }
}

__global__ void nvfp4_silu_gate_up_kernel(const float *gate, const float *up,
                                          float *out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    float g, sig;
    if (i >= n) return;
    g = gate[i];
    sig = 1.0f / (1.0f + expf(-g));
    out[i] = (g * sig) * up[i];
}

__global__ void nvfp4_add_inplace_kernel(float *dst, const float *src,
                                         size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

static float *nvfp4_load_bf16_host(const h3_nvfp4_store *store, const char *name,
                                   size_t expect, char *error,
                                   size_t error_size) {
    const h3_st_tensor *t;
    uint16_t *raw = NULL;
    float *out = NULL;
    size_t i, n;
    t = h3_st_find(&store->header, name);
    if (!t || t->dtype != H3_DTYPE_BF16) {
        fail(error, error_size, "missing BF16 tensor");
        return NULL;
    }
    n = (size_t)h3_st_tensor_elements(t);
    if (expect && n != expect) {
        fail(error, error_size, "BF16 tensor size mismatch");
        return NULL;
    }
    raw = (uint16_t *)malloc(n * sizeof(uint16_t));
    out = (float *)malloc(n * sizeof(float));
    if (!raw || !out) {
        free(raw);
        free(out);
        fail(error, error_size, "OOM BF16 load");
        return NULL;
    }
    if (!h3_nvfp4_store_copy(store, (size_t)t->file_offset, raw,
                             n * sizeof(uint16_t), error, error_size)) {
        free(raw);
        free(out);
        return NULL;
    }
    for (i = 0; i < n; i++) out[i] = nvfp4_bf16_bits_to_f32(raw[i]);
    free(raw);
    return out;
}

static const char *k_te_linear_suffix[] = {
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj",    "mlp.up_proj",
    "mlp.down_proj"};
enum { TE_LINEARS_PER_LAYER = 7 };

static nvfp4_dev_linear *nvfp4_dev_lin_find(const h3_nvfp4_linear *lin) {
    int i;
    if (!lin || !lin->weight) return NULL;
    for (i = 0; i < g_dev_lin_n; i++) {
        if (g_dev_lin[i].valid &&
            g_dev_lin[i].weight_off == (size_t)lin->weight->file_offset &&
            g_dev_lin[i].n == lin->n && g_dev_lin[i].k == lin->k)
            return &g_dev_lin[i];
    }
    return NULL;
}

/* Bulk-upload all text-layer packed NVFP4 (+ scales, AWQ) into one GPU arena.
 * Returns 0 on skip/OOM (caller keeps per-linear H2D). Never fails encode. */
static int nvfp4_te_packed_preload(const h3_nvfp4_store *store,
                                   unsigned n_layers, char *error,
                                   size_t error_size) {
    unsigned layer;
    int n_lin, idx;
    size_t need = 0, cursor = 0, slack;
    size_t free_b = 0, total_b = 0;
    cudaError_t st;
    struct timespec ts0, ts1;
    static int logged = 0;
    char ign[256];

    if (!store || n_layers == 0 || !nvfp4_want_packed_resident()) return 0;
    if (g_d_te_w_arena && g_dev_lin_n > 0) return 1;

    n_lin = (int)n_layers * TE_LINEARS_PER_LAYER;
    g_dev_lin = (nvfp4_dev_linear *)calloc((size_t)n_lin, sizeof(*g_dev_lin));
    if (!g_dev_lin) return 0;
    g_dev_lin_n = n_lin;

    idx = 0;
    for (layer = 0; layer < n_layers; layer++) {
        int j;
        for (j = 0; j < TE_LINEARS_PER_LAYER; j++, idx++) {
            char base[192];
            h3_nvfp4_linear lin;
            size_t packed_bytes, scale_bytes, pq_bytes;
            snprintf(base, sizeof(base), "model.layers.%u.%s", layer,
                     k_te_linear_suffix[j]);
            if (!h3_nvfp4_find_linear(store, base, &lin, ign, sizeof(ign))) {
                nvfp4_dev_lin_free();
                return 0;
            }
            packed_bytes = lin.n * (lin.k / 2);
            scale_bytes = lin.n * lin.groups;
            pq_bytes = lin.pre_quant_scale ? lin.k * sizeof(float) : 0;
            g_dev_lin[idx].weight_off = (size_t)lin.weight->file_offset;
            g_dev_lin[idx].n = lin.n;
            g_dev_lin[idx].k = lin.k;
            g_dev_lin[idx].groups = lin.groups;
            g_dev_lin[idx].group_size = lin.group_size;
            g_dev_lin[idx].packed_bytes = packed_bytes;
            g_dev_lin[idx].scale_bytes = scale_bytes;
            g_dev_lin[idx].pq_elems = lin.pre_quant_scale ? lin.k : 0;
            need += nvfp4_align256(packed_bytes);
            need += nvfp4_align256(scale_bytes);
            if (pq_bytes) need += nvfp4_align256(pq_bytes);
        }
    }

    slack = (size_t)2048 << 20;
    st = cudaMemGetInfo(&free_b, &total_b);
    if (st != cudaSuccess || free_b < need + slack) {
        if (!logged) {
            fprintf(stderr,
                    "te_packed: device_resident=0 reason=vram need=%.2fGiB "
                    "free=%.2fGiB slack=2GiB\n",
                    (double)need / (1024.0 * 1024.0 * 1024.0),
                    st == cudaSuccess
                        ? (double)free_b / (1024.0 * 1024.0 * 1024.0)
                        : -1.0);
            fflush(stderr);
            logged = 1;
        }
        nvfp4_dev_lin_free();
        return 0;
    }
    st = cudaMalloc((void **)&g_d_te_w_arena, need);
    if (st != cudaSuccess) {
        (void)cudaGetLastError();
        if (!logged) {
            fprintf(stderr, "te_packed: device_resident=0 reason=cudaMalloc "
                            "need=%.2fGiB\n",
                    (double)need / (1024.0 * 1024.0 * 1024.0));
            fflush(stderr);
            logged = 1;
        }
        nvfp4_dev_lin_free();
        return 0;
    }
    g_d_te_w_arena_bytes = need;

    {
        double prefault_ms = 0.0;
        if (store->map && store->map_len > 0) {
#ifdef MADV_WILLNEED
            (void)madvise((void *)store->map, store->map_len, MADV_WILLNEED);
#endif
            prefault_ms = nvfp4_prefault_mmap(store->map, store->map_len);
            if (!logged && prefault_ms > 0.0) {
                fprintf(stderr, "te_packed: prefault_ms=%.1f bytes=%.2fGiB\n",
                        prefault_ms,
                        (double)store->map_len / (1024.0 * 1024.0 * 1024.0));
                fflush(stderr);
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &ts0);
    idx = 0;
    for (layer = 0; layer < n_layers; layer++) {
        int j;
        for (j = 0; j < TE_LINEARS_PER_LAYER; j++, idx++) {
            char base[192];
            h3_nvfp4_linear lin;
            nvfp4_dev_linear *dl = &g_dev_lin[idx];
            size_t packed_bytes = dl->packed_bytes;
            size_t scale_bytes = dl->scale_bytes;
            float scale2 = 0.0f;
            snprintf(base, sizeof(base), "model.layers.%u.%s", layer,
                     k_te_linear_suffix[j]);
            if (!h3_nvfp4_find_linear(store, base, &lin, ign, sizeof(ign))) {
                nvfp4_dev_lin_free();
                return 0;
            }
            if (!ensure_host_bytes((void **)&g_h_blocked, &g_h_blocked_cap,
                                   scale_bytes, ign, sizeof(ign)) ||
                !ensure_host_bytes((void **)&g_h_logical, &g_h_logical_cap,
                                   scale_bytes, ign, sizeof(ign))) {
                nvfp4_dev_lin_free();
                return 0;
            }
            {
                const void *packed_src = NULL;
                int packed_pinned = 0;
                size_t off = (size_t)lin.weight->file_offset;
                /* mmap → GPU directly. Staging memcpy doubled 12.8GiB H2D
                 * (~10.5s). cudaHostRegister of the 15GB map is skipped. */
                if (store->map && store->map_len > 0) {
                    if (off + packed_bytes > store->map_len) {
                        nvfp4_dev_lin_free();
                        return 0;
                    }
                    packed_src = store->map + off;
                    packed_pinned = g_map_hostreg ? 1 : 0;
                } else {
                    if (!ensure_packed_stage(packed_bytes, ign, sizeof(ign))) {
                        nvfp4_dev_lin_free();
                        return 0;
                    }
                    if (g_h_packed_stage_pinned)
                        cudaStreamSynchronize(0);
                    if (!h3_nvfp4_store_copy(store, off, g_h_packed_stage,
                                             packed_bytes, ign, sizeof(ign))) {
                        nvfp4_dev_lin_free();
                        return 0;
                    }
                    packed_src = g_h_packed_stage;
                    packed_pinned = g_h_packed_stage_pinned;
                }
                if (!h3_nvfp4_store_copy(store, (size_t)lin.scale->file_offset,
                                         g_h_blocked, scale_bytes, ign,
                                         sizeof(ign)) ||
                    !h3_nvfp4_store_copy(store,
                                         (size_t)lin.scale_2->file_offset,
                                         &scale2, sizeof(float), ign,
                                         sizeof(ign)) ||
                    !h3_nvfp4_unswizzle_scales_u8(g_h_blocked, lin.n,
                                                  lin.groups, g_h_logical, ign,
                                                  sizeof(ign))) {
                    nvfp4_dev_lin_free();
                    return 0;
                }
                dl->d_packed = g_d_te_w_arena + cursor;
                cursor += nvfp4_align256(packed_bytes);
                dl->d_scales = g_d_te_w_arena + cursor;
                cursor += nvfp4_align256(scale_bytes);
                if (!nvfp4_h2d(dl->d_packed, packed_src, packed_bytes,
                               packed_pinned, ign, sizeof(ign),
                               "H2D packed resident") ||
                    !nvfp4_h2d(dl->d_scales, g_h_logical, scale_bytes, 0, ign,
                               sizeof(ign), "H2D scales resident")) {
                    nvfp4_dev_lin_free();
                    return 0;
                }
            }
            dl->scale2 = scale2;
            if (lin.pre_quant_scale) {
                size_t k = lin.k, t;
                dl->d_pq = (float *)(g_d_te_w_arena + cursor);
                cursor += nvfp4_align256(k * sizeof(float));
                if (!ensure_host_bytes((void **)&g_h_pq_raw, &g_h_pq_raw_cap,
                                       k * sizeof(uint16_t), ign,
                                       sizeof(ign)) ||
                    !ensure_host_bytes((void **)&g_h_pq_f32, &g_h_pq_f32_cap,
                                       k * sizeof(float), ign, sizeof(ign)) ||
                    !h3_nvfp4_store_copy(
                        store, (size_t)lin.pre_quant_scale->file_offset,
                        g_h_pq_raw, k * sizeof(uint16_t), ign, sizeof(ign))) {
                    nvfp4_dev_lin_free();
                    return 0;
                }
                for (t = 0; t < k; t++)
                    g_h_pq_f32[t] = nvfp4_bf16_bits_to_f32(g_h_pq_raw[t]);
                if (!nvfp4_h2d(dl->d_pq, g_h_pq_f32, k * sizeof(float), 0, ign,
                               sizeof(ign), "H2D pre_quant resident")) {
                    nvfp4_dev_lin_free();
                    return 0;
                }
            }
            dl->valid = 1;
        }
    }
    st = cudaDeviceSynchronize();
    if (st != cudaSuccess) {
        nvfp4_dev_lin_free();
        return 0;
    }
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    if (!logged) {
        double ms =
            ((double)ts1.tv_sec * 1000.0 + (double)ts1.tv_nsec / 1e6) -
            ((double)ts0.tv_sec * 1000.0 + (double)ts0.tv_nsec / 1e6);
        fprintf(stderr,
                "te_packed: device_resident=1 linears=%d bytes=%.2fGiB "
                "upload_ms=%.1f path=%s (H3_TE_PACKED_RESIDENT=0 reverts)\n",
                n_lin, (double)need / (1024.0 * 1024.0 * 1024.0), ms,
                (store->map && store->map_len > 0)
                    ? (g_map_hostreg ? "mmap_pinned" : "mmap_direct")
                    : "stage");
        fflush(stderr);
        logged = 1;
    }
    (void)error;
    (void)error_size;
    return 1;
}

static int nvfp4_te_ln_preload(const h3_nvfp4_store *store, unsigned n_layers,
                               char *error, size_t error_size) {
    unsigned layer;
    size_t td = (size_t)H3_NVFP4_TEXT_DIM;
    size_t hd = (size_t)H3_NVFP4_HEAD_DIM;
    if (!store || n_layers == 0) return 0;
    if (g_ln_cached_layers >= n_layers && g_d_ln_in && g_d_ln_post &&
        g_d_qn_all && g_d_kn_all)
        return 1;
    if (!ensure_dev_bytes((void **)&g_d_ln_in, &g_d_ln_in_cap,
                          (size_t)n_layers * td * sizeof(float), error,
                          error_size) ||
        !ensure_dev_bytes((void **)&g_d_ln_post, &g_d_ln_post_cap,
                          (size_t)n_layers * td * sizeof(float), error,
                          error_size) ||
        !ensure_dev_bytes((void **)&g_d_qn_all, &g_d_qn_all_cap,
                          (size_t)n_layers * hd * sizeof(float), error,
                          error_size) ||
        !ensure_dev_bytes((void **)&g_d_kn_all, &g_d_kn_all_cap,
                          (size_t)n_layers * hd * sizeof(float), error,
                          error_size))
        return 0;
    for (layer = 0; layer < n_layers; layer++) {
        char base[192];
        float *in_w, *post_w, *qn_w, *kn_w;
        cudaError_t st;
        snprintf(base, sizeof(base), "model.layers.%u.input_layernorm.weight",
                 layer);
        in_w = nvfp4_load_bf16_host(store, base, td, error, error_size);
        snprintf(base, sizeof(base),
                 "model.layers.%u.post_attention_layernorm.weight", layer);
        post_w = nvfp4_load_bf16_host(store, base, td, error, error_size);
        snprintf(base, sizeof(base), "model.layers.%u.self_attn.q_norm.weight",
                 layer);
        qn_w = nvfp4_load_bf16_host(store, base, hd, error, error_size);
        snprintf(base, sizeof(base), "model.layers.%u.self_attn.k_norm.weight",
                 layer);
        kn_w = nvfp4_load_bf16_host(store, base, hd, error, error_size);
        if (!in_w || !post_w || !qn_w || !kn_w) {
            free(in_w);
            free(post_w);
            free(qn_w);
            free(kn_w);
            return 0;
        }
        st = cudaMemcpy(g_d_ln_in + (size_t)layer * td, in_w,
                        td * sizeof(float), cudaMemcpyHostToDevice);
        if (st == cudaSuccess)
            st = cudaMemcpy(g_d_ln_post + (size_t)layer * td, post_w,
                            td * sizeof(float), cudaMemcpyHostToDevice);
        if (st == cudaSuccess)
            st = cudaMemcpy(g_d_qn_all + (size_t)layer * hd, qn_w,
                            hd * sizeof(float), cudaMemcpyHostToDevice);
        if (st == cudaSuccess)
            st = cudaMemcpy(g_d_kn_all + (size_t)layer * hd, kn_w,
                            hd * sizeof(float), cudaMemcpyHostToDevice);
        free(in_w);
        free(post_w);
        free(qn_w);
        free(kn_w);
        if (st != cudaSuccess) {
            fail(error, error_size, "H2D TE layernorm cache");
            return 0;
        }
    }
    g_ln_cached_layers = n_layers;
    return 1;
}

/* Resolve packed+scales into g_d_w (dequant). Shared by host-y linear path
 * helpers and device-act encode. */
static int nvfp4_prepare_w_dev(const h3_nvfp4_store *store,
                               const h3_nvfp4_linear *lin, char *error,
                               size_t error_size) {
    size_t k_half, packed_bytes, scale_bytes, w_bytes, total_nibbles;
    nvfp4_host_scale_slot *slot = NULL;
    nvfp4_dev_linear *dl;
    const uint8_t *logical_scales = NULL;
    const uint8_t *packed_src = NULL;
    float scale2 = 0.0f;
    cudaError_t st;
    int threads = 256;
    int blocks;

    if (!store || !lin) {
        fail(error, error_size, "invalid prepare_w");
        return 0;
    }
    if (lin->k < 2 || (lin->k % 2) != 0) {
        fail(error, error_size, "nvfp4 k not even");
        return 0;
    }
    k_half = lin->k / 2;
    packed_bytes = lin->n * k_half;
    scale_bytes = lin->n * lin->groups;
    w_bytes = lin->n * lin->k * sizeof(float);

    dl = nvfp4_dev_lin_find(lin);
    if (dl && dl->d_packed && dl->d_scales) {
        if (!ensure_dev_bytes((void **)&g_d_w, &g_d_w_cap, w_bytes, error,
                              error_size))
            return 0;
        total_nibbles = lin->n * k_half;
        blocks = (int)((total_nibbles + (size_t)threads - 1) / (size_t)threads);
        nvfp4_dequant_kernel<<<blocks, threads>>>(
            dl->d_packed, dl->d_scales, dl->scale2, lin->n, lin->k, lin->groups,
            lin->group_size, g_d_w);
        st = cudaGetLastError();
        if (st != cudaSuccess) {
            fail(error, error_size, "nvfp4 dequant kernel");
            return 0;
        }
        return 1;
    }

    nvfp4_try_register_map(store);

    if (!nvfp4_cache_disabled()) slot = nvfp4_host_find(lin);
    packed_src = nvfp4_mmap_packed(store, lin, packed_bytes);
    if (slot && slot->logical_scales) {
        logical_scales = slot->logical_scales;
        scale2 = slot->scale2;
        nvfp4_try_pin_packed(slot);
        if (!packed_src && slot->packed && slot->packed_bytes == packed_bytes)
            packed_src = slot->packed;
    }
    if (!logical_scales || !packed_src) {
        if (!ensure_host_bytes((void **)&g_h_blocked, &g_h_blocked_cap,
                               scale_bytes, error, error_size) ||
            !ensure_host_bytes((void **)&g_h_logical, &g_h_logical_cap,
                               scale_bytes, error, error_size))
            return 0;
        if (!h3_nvfp4_store_copy(store, (size_t)lin->scale->file_offset,
                                 g_h_blocked, scale_bytes, error,
                                 error_size) ||
            !h3_nvfp4_store_copy(store, (size_t)lin->scale_2->file_offset,
                                 &scale2, sizeof(float), error, error_size))
            return 0;
        if (!h3_nvfp4_unswizzle_scales_u8(g_h_blocked, lin->n, lin->groups,
                                          g_h_logical, error, error_size))
            return 0;
        logical_scales = g_h_logical;
        if (!packed_src) {
            if (!ensure_packed_stage(packed_bytes, error, error_size)) return 0;
            if (store->map && store->map_len > 0) {
                size_t off = (size_t)lin->weight->file_offset;
                if (off + packed_bytes > store->map_len) {
                    fail(error, error_size, "packed H2D past mmap end");
                    return 0;
                }
                memcpy(g_h_packed_stage, store->map + off, packed_bytes);
            } else if (!h3_nvfp4_store_copy(store, (size_t)lin->weight->file_offset,
                                            g_h_packed_stage, packed_bytes, error,
                                            error_size)) {
                return 0;
            }
            packed_src = g_h_packed_stage;
        }
        if (!nvfp4_cache_disabled()) {
            if (!slot) slot = nvfp4_host_add();
            if (!slot) {
                fail(error, error_size, "OOM nvfp4 host scale cache");
                return 0;
            }
            if (!slot->logical_scales) {
                slot->logical_scales = (uint8_t *)malloc(scale_bytes);
                if (!slot->logical_scales) {
                    fail(error, error_size, "OOM nvfp4 host scale copy");
                    nvfp4_host_scale_slot_reset(slot);
                    if (g_host_scales_n > 0) g_host_scales_n--;
                    return 0;
                }
                memcpy(slot->logical_scales, g_h_logical, scale_bytes);
            }
            if (!slot->packed && !g_map_hostreg) {
                slot->packed = (uint8_t *)malloc(packed_bytes);
                if (slot->packed) {
                    memcpy(slot->packed, packed_src, packed_bytes);
                    slot->packed_bytes = packed_bytes;
                    nvfp4_try_pin_packed(slot);
                    packed_src = slot->packed;
                }
            }
            slot->valid = 1;
            slot->weight_off = (size_t)lin->weight->file_offset;
            slot->scale_off = (size_t)lin->scale->file_offset;
            slot->scale2_off = (size_t)lin->scale_2->file_offset;
            slot->n = lin->n;
            slot->k = lin->k;
            slot->groups = lin->groups;
            slot->scale2 = scale2;
            logical_scales = slot->logical_scales;
            if (lin->pre_quant_scale && !slot->pre_quant) {
                size_t i, k = lin->k;
                if (!ensure_host_bytes((void **)&g_h_pq_raw, &g_h_pq_raw_cap,
                                       k * sizeof(uint16_t), error,
                                       error_size) ||
                    !ensure_host_bytes((void **)&g_h_pq_f32, &g_h_pq_f32_cap,
                                       k * sizeof(float), error, error_size))
                    return 0;
                if (!h3_nvfp4_store_copy(
                        store, (size_t)lin->pre_quant_scale->file_offset,
                        g_h_pq_raw, k * sizeof(uint16_t), error, error_size))
                    return 0;
                for (i = 0; i < k; i++)
                    g_h_pq_f32[i] = nvfp4_bf16_bits_to_f32(g_h_pq_raw[i]);
                slot->pre_quant = (float *)malloc(k * sizeof(float));
                if (!slot->pre_quant) {
                    fail(error, error_size, "OOM nvfp4 host pre_quant");
                    nvfp4_host_scale_slot_reset(slot);
                    if (g_host_scales_n > 0) g_host_scales_n--;
                    return 0;
                }
                slot->pq_off = (size_t)lin->pre_quant_scale->file_offset;
                memcpy(slot->pre_quant, g_h_pq_f32, k * sizeof(float));
            }
        }
    }

    if (!ensure_dev_bytes((void **)&g_d_packed, &g_d_packed_cap, packed_bytes,
                          error, error_size) ||
        !ensure_dev_bytes((void **)&g_d_scales, &g_d_scales_cap, scale_bytes,
                          error, error_size) ||
        !ensure_dev_bytes((void **)&g_d_w, &g_d_w_cap, w_bytes, error,
                          error_size))
        return 0;
    {
        int packed_pinned = 0;
        if (g_map_hostreg && packed_src && store && store->map &&
            packed_src >= store->map &&
            packed_src < store->map + store->map_len)
            packed_pinned = 1;
        else if (slot && slot->packed_pinned && packed_src == slot->packed)
            packed_pinned = 1;
        else if (packed_src == g_h_packed_stage && g_h_packed_stage_pinned)
            packed_pinned = 1;
        if (!nvfp4_h2d(g_d_packed, packed_src, packed_bytes, packed_pinned,
                       error, error_size, "H2D packed") ||
            !nvfp4_h2d(g_d_scales, logical_scales, scale_bytes, 0, error,
                       error_size, "H2D scales"))
            return 0;
    }
    total_nibbles = lin->n * k_half;
    blocks = (int)((total_nibbles + (size_t)threads - 1) / (size_t)threads);
    nvfp4_dequant_kernel<<<blocks, threads>>>(g_d_packed, g_d_scales, scale2,
                                              lin->n, lin->k, lin->groups,
                                              lin->group_size, g_d_w);
    st = cudaGetLastError();
    if (st != cudaSuccess) {
        fail(error, error_size, "nvfp4 dequant kernel");
        return 0;
    }
    return 1;
}

static int nvfp4_linear_dx_dy(const h3_nvfp4_store *store,
                              const h3_nvfp4_linear *lin, float *d_x,
                              size_t m, float *d_y, char *error,
                              size_t error_size) {
    nvfp4_host_scale_slot *slot = NULL;
    const float *pq_host = NULL;
    cudaError_t st;
    int threads = 256;
    int blocks;

    if (!g_te_cuda || !d_x || !d_y || m == 0) {
        fail(error, error_size, "invalid linear_dx_dy");
        return 0;
    }
    if (!nvfp4_prepare_w_dev(store, lin, error, error_size)) return 0;
    if (lin->pre_quant_scale) {
        nvfp4_dev_linear *dl = nvfp4_dev_lin_find(lin);
        const float *d_pq_src = NULL;
        if (dl && dl->d_pq) {
            d_pq_src = dl->d_pq;
        } else {
            if (!nvfp4_cache_disabled()) slot = nvfp4_host_find(lin);
            pq_host = slot ? slot->pre_quant : NULL;
            if (!pq_host) {
                size_t i, k = lin->k;
                if (!ensure_host_bytes((void **)&g_h_pq_raw, &g_h_pq_raw_cap,
                                       k * sizeof(uint16_t), error,
                                       error_size) ||
                    !ensure_host_bytes((void **)&g_h_pq_f32, &g_h_pq_f32_cap,
                                       k * sizeof(float), error, error_size))
                    return 0;
                if (!h3_nvfp4_store_copy(
                        store, (size_t)lin->pre_quant_scale->file_offset,
                        g_h_pq_raw, k * sizeof(uint16_t), error, error_size))
                    return 0;
                for (i = 0; i < k; i++)
                    g_h_pq_f32[i] = nvfp4_bf16_bits_to_f32(g_h_pq_raw[i]);
                pq_host = g_h_pq_f32;
            }
            if (!ensure_dev_bytes((void **)&g_d_pq, &g_d_pq_cap,
                                  lin->k * sizeof(float), error, error_size))
                return 0;
            st = cudaMemcpy(g_d_pq, pq_host, lin->k * sizeof(float),
                            cudaMemcpyHostToDevice);
            if (st != cudaSuccess) {
                fail(error, error_size, "H2D pre_quant");
                return 0;
            }
            d_pq_src = g_d_pq;
        }
        blocks = (int)((m * lin->k + (size_t)threads - 1) / (size_t)threads);
        nvfp4_pre_quant_kernel<<<blocks, threads>>>(d_x, d_pq_src, m, lin->k);
        st = cudaGetLastError();
        if (st != cudaSuccess) {
            fail(error, error_size, "nvfp4 pre_quant kernel");
            return 0;
        }
    }
    if (!h3_cuda_gemm_f32_dd(g_te_cuda, d_x, g_d_w, m, lin->n, lin->k, d_y,
                             NULL, error, error_size))
        return 0;
    return 1;
}

extern "C" int h3_nvfp4_cuda_encode_text_layers(const h3_nvfp4_store *store,
                                                float *hidden, size_t tokens,
                                                unsigned n_layers, char *error,
                                                size_t error_size) {
    const float eps = 1e-6f;
    const float rope_theta = 5000000.0f;
    const char *host_act;
    unsigned layer;
    size_t td = (size_t)H3_NVFP4_TEXT_DIM;
    size_t qd = (size_t)H3_NVFP4_QUERY_DIM;
    size_t kd = (size_t)H3_NVFP4_KV_DIM;
    size_t mid = (size_t)H3_NVFP4_INTERMEDIATE;
    size_t hid_bytes, q_bytes, k_bytes, mid_bytes;
    cudaError_t st;
    static int logged = 0;
    int verbose = 0;
    {
        const char *v = getenv("H3_NVFP4_VERBOSE");
        verbose = v && v[0] == '1';
    }

    host_act = getenv("H3_NVFP4_HOST_ACT");
    if (host_act && host_act[0] == '1') return 0;
    /* Default: device-resident activations. H3_NVFP4_DEV_ACT=0 also opts out. */
    {
        const char *dev_act = getenv("H3_NVFP4_DEV_ACT");
        if (dev_act && dev_act[0] == '0') return 0;
    }
    if (!g_te_cuda || g_forced_host || !store || !hidden || tokens == 0 ||
        n_layers == 0)
        return 0;

    hid_bytes = tokens * td * sizeof(float);
    q_bytes = tokens * qd * sizeof(float);
    k_bytes = tokens * kd * sizeof(float);
    mid_bytes = tokens * mid * sizeof(float);

    if (!ensure_dev_bytes((void **)&g_d_hidden, &g_d_hidden_cap, hid_bytes,
                          error, error_size) ||
        !ensure_dev_bytes((void **)&g_d_normed, &g_d_normed_cap, hid_bytes,
                          error, error_size) ||
        !ensure_dev_bytes((void **)&g_d_q, &g_d_q_cap, q_bytes, error,
                          error_size) ||
        !ensure_dev_bytes((void **)&g_d_k, &g_d_k_cap, k_bytes, error,
                          error_size) ||
        !ensure_dev_bytes((void **)&g_d_v, &g_d_v_cap, k_bytes, error,
                          error_size) ||
        !ensure_dev_bytes((void **)&g_d_attn, &g_d_attn_cap, q_bytes, error,
                          error_size) ||
        !ensure_dev_bytes((void **)&g_d_tmp, &g_d_tmp_cap, hid_bytes, error,
                          error_size) ||
        !ensure_dev_bytes((void **)&g_d_gate, &g_d_gate_cap, mid_bytes, error,
                          error_size) ||
        !ensure_dev_bytes((void **)&g_d_up, &g_d_up_cap, mid_bytes, error,
                          error_size) ||
        !ensure_dev_bytes((void **)&g_d_ffn, &g_d_ffn_cap, mid_bytes, error,
                          error_size) ||
        !ensure_dev_bytes((void **)&g_d_nw, &g_d_nw_cap,
                          td * sizeof(float), error, error_size))
        return 0;

    st = cudaMemcpy(g_d_hidden, hidden, hid_bytes, cudaMemcpyHostToDevice);
    if (st != cudaSuccess) {
        fail(error, error_size, "H2D TE hidden");
        return 0;
    }
    if (!logged) {
        fprintf(stderr,
                "te_act: device_resident=1 tokens=%zu layers=%u "
                "(default; H3_NVFP4_HOST_ACT=1 reverts)\n",
                tokens, n_layers);
        fflush(stderr);
        logged = 1;
    }
    (void)nvfp4_te_packed_preload(store, n_layers, error, error_size);
    if (g_after_packed_hook) {
        void (*fn)(void *) = g_after_packed_hook;
        void *user = g_after_packed_user;
        g_after_packed_hook = NULL;
        g_after_packed_user = NULL;
        fn(user);
    }
    if (!nvfp4_te_ln_preload(store, n_layers, error, error_size)) {
        if (error) error[0] = '\0';
        g_ln_cached_layers = 0;
    }

    for (layer = 0; layer < n_layers; layer++) {
        char base[192];
        h3_nvfp4_linear lin;
        float *in_w = NULL, *post_w = NULL, *qn_w = NULL, *kn_w = NULL;
        const float *d_in_ln, *d_post_ln, *d_qn, *d_kn;
        int use_ln_cache = (g_ln_cached_layers >= n_layers);
        int thr = 256;
        int blocks;
        size_t n_elem;
        struct timespec ts0, ts1;

        clock_gettime(CLOCK_MONOTONIC, &ts0);
        if (verbose) {
            fprintf(stderr, "te_layer_begin %u/%u tokens=%zu\n", layer + 1,
                    n_layers, tokens);
            fflush(stderr);
        }

        if (use_ln_cache) {
            d_in_ln = g_d_ln_in + (size_t)layer * td;
        } else {
            snprintf(base, sizeof(base),
                     "model.layers.%u.input_layernorm.weight", layer);
            in_w = nvfp4_load_bf16_host(store, base, td, error, error_size);
            if (!in_w) return 0;
            st = cudaMemcpy(g_d_nw, in_w, td * sizeof(float),
                            cudaMemcpyHostToDevice);
            free(in_w);
            if (st != cudaSuccess) {
                fail(error, error_size, "H2D input_layernorm");
                return 0;
            }
            d_in_ln = g_d_nw;
        }
        if (!h3_cuda_rms_norm_f32_dev(g_te_cuda, g_d_hidden, d_in_ln, tokens, td,
                                      eps, g_d_normed, error, error_size))
            return 0;

        snprintf(base, sizeof(base), "model.layers.%u.self_attn.q_proj", layer);
        if (!h3_nvfp4_find_linear(store, base, &lin, error, error_size) ||
            !nvfp4_linear_dx_dy(store, &lin, g_d_normed, tokens, g_d_q, error,
                               error_size))
            return 0;
        snprintf(base, sizeof(base), "model.layers.%u.self_attn.k_proj", layer);
        if (!h3_nvfp4_find_linear(store, base, &lin, error, error_size) ||
            !nvfp4_linear_dx_dy(store, &lin, g_d_normed, tokens, g_d_k, error,
                               error_size))
            return 0;
        snprintf(base, sizeof(base), "model.layers.%u.self_attn.v_proj", layer);
        if (!h3_nvfp4_find_linear(store, base, &lin, error, error_size) ||
            !nvfp4_linear_dx_dy(store, &lin, g_d_normed, tokens, g_d_v, error,
                               error_size))
            return 0;

        if (use_ln_cache) {
            d_qn = g_d_qn_all + (size_t)layer * (size_t)H3_NVFP4_HEAD_DIM;
            d_kn = g_d_kn_all + (size_t)layer * (size_t)H3_NVFP4_HEAD_DIM;
        } else {
            snprintf(base, sizeof(base),
                     "model.layers.%u.self_attn.q_norm.weight", layer);
            qn_w = nvfp4_load_bf16_host(store, base, (size_t)H3_NVFP4_HEAD_DIM,
                                        error, error_size);
            snprintf(base, sizeof(base),
                     "model.layers.%u.self_attn.k_norm.weight", layer);
            kn_w = nvfp4_load_bf16_host(store, base, (size_t)H3_NVFP4_HEAD_DIM,
                                        error, error_size);
            if (!qn_w || !kn_w) {
                free(qn_w);
                free(kn_w);
                return 0;
            }
            st = cudaMemcpy(g_d_nw, qn_w,
                            (size_t)H3_NVFP4_HEAD_DIM * sizeof(float),
                            cudaMemcpyHostToDevice);
            free(qn_w);
            if (st != cudaSuccess) {
                free(kn_w);
                fail(error, error_size, "H2D q_norm");
                return 0;
            }
            d_qn = g_d_nw;
            /* k_norm copied after q head-rms so they share g_d_nw. */
        }
        blocks = (int)((tokens * (size_t)H3_NVFP4_QUERY_HEADS +
                        (size_t)thr - 1) /
                       (size_t)thr);
        nvfp4_head_rms_kernel<<<blocks, thr>>>(
            g_d_q, d_qn, tokens, (size_t)H3_NVFP4_QUERY_HEADS,
            (size_t)H3_NVFP4_HEAD_DIM, eps);
        if (!use_ln_cache) {
            st = cudaMemcpy(g_d_nw, kn_w,
                            (size_t)H3_NVFP4_HEAD_DIM * sizeof(float),
                            cudaMemcpyHostToDevice);
            free(kn_w);
            if (st != cudaSuccess) {
                fail(error, error_size, "H2D k_norm");
                return 0;
            }
            d_kn = g_d_nw;
        }
        blocks = (int)((tokens * (size_t)H3_NVFP4_KV_HEADS + (size_t)thr - 1) /
                       (size_t)thr);
        nvfp4_head_rms_kernel<<<blocks, thr>>>(
            g_d_k, d_kn, tokens, (size_t)H3_NVFP4_KV_HEADS,
            (size_t)H3_NVFP4_HEAD_DIM, eps);

        blocks = (int)((tokens * ((size_t)H3_NVFP4_QUERY_HEADS +
                                  (size_t)H3_NVFP4_KV_HEADS) +
                        (size_t)thr - 1) /
                       (size_t)thr);
        nvfp4_rope_text_kernel<<<blocks, thr>>>(
            g_d_q, g_d_k, tokens, (size_t)H3_NVFP4_QUERY_HEADS,
            (size_t)H3_NVFP4_KV_HEADS, (size_t)H3_NVFP4_HEAD_DIM, rope_theta);
        st = cudaGetLastError();
        if (st != cudaSuccess) {
            fail(error, error_size, "TE rope/head_rms kernel");
            return 0;
        }

        if (!h3_cuda_sdpa_gqa_causal_f16_dev(
                g_te_cuda, g_d_q, g_d_k, g_d_v, tokens,
                (size_t)H3_NVFP4_QUERY_HEADS, (size_t)H3_NVFP4_KV_HEADS,
                (size_t)H3_NVFP4_HEAD_DIM,
                1.0f / sqrtf((float)H3_NVFP4_HEAD_DIM), g_d_attn, NULL, NULL, 0,
                error, error_size))
            return 0;

        snprintf(base, sizeof(base), "model.layers.%u.self_attn.o_proj", layer);
        if (!h3_nvfp4_find_linear(store, base, &lin, error, error_size) ||
            !nvfp4_linear_dx_dy(store, &lin, g_d_attn, tokens, g_d_tmp, error,
                               error_size))
            return 0;
        n_elem = tokens * td;
        blocks = (int)((n_elem + (size_t)thr - 1) / (size_t)thr);
        nvfp4_add_inplace_kernel<<<blocks, thr>>>(g_d_hidden, g_d_tmp, n_elem);

        if (use_ln_cache) {
            d_post_ln = g_d_ln_post + (size_t)layer * td;
        } else {
            snprintf(base, sizeof(base),
                     "model.layers.%u.post_attention_layernorm.weight", layer);
            post_w = nvfp4_load_bf16_host(store, base, td, error, error_size);
            if (!post_w) return 0;
            st = cudaMemcpy(g_d_nw, post_w, td * sizeof(float),
                            cudaMemcpyHostToDevice);
            free(post_w);
            if (st != cudaSuccess) {
                fail(error, error_size, "H2D post_layernorm");
                return 0;
            }
            d_post_ln = g_d_nw;
        }
        if (!h3_cuda_rms_norm_f32_dev(g_te_cuda, g_d_hidden, d_post_ln, tokens,
                                      td, eps, g_d_normed, error, error_size))
            return 0;

        snprintf(base, sizeof(base), "model.layers.%u.mlp.gate_proj", layer);
        if (!h3_nvfp4_find_linear(store, base, &lin, error, error_size) ||
            !nvfp4_linear_dx_dy(store, &lin, g_d_normed, tokens, g_d_gate, error,
                               error_size))
            return 0;
        snprintf(base, sizeof(base), "model.layers.%u.mlp.up_proj", layer);
        if (!h3_nvfp4_find_linear(store, base, &lin, error, error_size) ||
            !nvfp4_linear_dx_dy(store, &lin, g_d_normed, tokens, g_d_up, error,
                               error_size))
            return 0;
        n_elem = tokens * mid;
        blocks = (int)((n_elem + (size_t)thr - 1) / (size_t)thr);
        nvfp4_silu_gate_up_kernel<<<blocks, thr>>>(g_d_gate, g_d_up, g_d_ffn,
                                                   n_elem);
        st = cudaGetLastError();
        if (st != cudaSuccess) {
            fail(error, error_size, "TE silu_mul kernel");
            return 0;
        }
        snprintf(base, sizeof(base), "model.layers.%u.mlp.down_proj", layer);
        if (!h3_nvfp4_find_linear(store, base, &lin, error, error_size) ||
            !nvfp4_linear_dx_dy(store, &lin, g_d_ffn, tokens, g_d_tmp, error,
                               error_size))
            return 0;
        n_elem = tokens * td;
        blocks = (int)((n_elem + (size_t)thr - 1) / (size_t)thr);
        nvfp4_add_inplace_kernel<<<blocks, thr>>>(g_d_hidden, g_d_tmp, n_elem);

        clock_gettime(CLOCK_MONOTONIC, &ts1);
        if (verbose) {
            fprintf(stderr, "te_layer_done %u/%u wall_ms=%.1f\n", layer + 1,
                    n_layers,
                    ((double)ts1.tv_sec * 1000.0 + (double)ts1.tv_nsec / 1e6) -
                        ((double)ts0.tv_sec * 1000.0 + (double)ts0.tv_nsec / 1e6));
            fflush(stderr);
        }
    }

    st = cudaMemcpy(hidden, g_d_hidden, hid_bytes, cudaMemcpyDeviceToHost);
    if (st != cudaSuccess) {
        fail(error, error_size, "D2H TE hidden");
        return 0;
    }
    return 1;
}
