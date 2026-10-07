#include "h3_w4a8_cache.h"

#include "h3_dit_nvfp4.h"
#include "h3_w4a8_dequant.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/mman.h>
#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif

typedef struct {
    int8_t *h_w;
    float *h_s;
    size_t n;
    size_t k;
    int pinned_w;
    int pinned_s;
    int from_arena; /* 1: do not free; lives in cache host_arena */
    int ready;      /* 1: bytes valid for H2D */
} h3_w4a8_int8_host_linear;

typedef struct {
    h3_cuda_w4a8_weight *linears[H3_W4A8_LINEAR_COUNT];
    h3_w4a8_int8_host_linear host[H3_W4A8_LINEAR_COUNT]; /* INT8 staging */
    size_t bytes;
    uint64_t last_used;
    int resident;
} h3_w4a8_cache_slot;

struct h3_w4a8_cache {
    h3_cuda *cuda;
    h3_w4a8_store *store;           /* W4A8 codebook path (mutually exclusive) */
    h3_dit_int8_store *int8_store;  /* INT8 ConvRot path */
    h3_nvfp4_store *nvfp4_store;    /* NVFP4 DiT → requant INT8 */
    unsigned n_blocks;
    h3_w4a8_cache_policy requested_policy;
    h3_w4a8_cache_policy effective_policy;
    unsigned window_size;
    unsigned open_window_cap; /* open-time window; refit must not grow past */
    size_t vram_budget_bytes;
    size_t vram_reserve_bytes;
    int group_size;
    int auto_downgraded;
    int refit_done;
    unsigned int8_disk_loads;
    unsigned int8_host_hits;
    unsigned nvfp4_sidecar_hits;
    unsigned nvfp4_sidecar_misses;
    unsigned nvfp4_sidecar_writes;
    h3_w4a8_cache_slot *slots;
    unsigned n_resident;
    size_t bytes_resident;
    size_t bytes_per_block_est;
    size_t bytes_estimated_all;
    unsigned max_blocks_fit;
    uint64_t tick;
    int window_frozen;
    int async_h2d; /* U=1: skip per-block copy-stream drain */
    unsigned host_ready_blocks;
    pthread_mutex_t host_mu;
    int host_mu_ready;
    void *host_arena;
    size_t host_arena_bytes;
    int host_arena_pinned;
    double host_stage_alloc_ms;
    double host_stage_read_ms;
    size_t host_stage_bytes;
    unsigned host_stage_n_linears;
    unsigned host_stage_n_skip;
    int host_stage_threads;
    int host_stage_mmap;
    void *host_mmap;
    size_t host_mmap_len;
    int host_mmap_registered;
    /* Evicted device INT8 buffers, reused by the next load (no cudaMalloc). */
    h3_cuda_w4a8_weight *recycle[H3_W4A8_LINEAR_COUNT];
    unsigned recycle_hits;
    /* H3_DIT_NVFP4=1: 4 linears FP4-served; INT8 copies are lazy-only. */
    int nvfp4_int8_skip;
};

static void free_slot(h3_w4a8_cache *cache, unsigned block);

static void recycle_flush(h3_w4a8_cache *cache) {
    int i;
    if (!cache) return;
    for (i = 0; i < H3_W4A8_LINEAR_COUNT; i++) {
        if (cache->recycle[i]) {
            h3_cuda_w4a8_weight_free(cache->recycle[i]);
            cache->recycle[i] = NULL;
        }
    }
}

static void steal_or_free_slot(h3_w4a8_cache *cache, unsigned block) {
    h3_w4a8_cache_slot *slot;
    int i;
    if (!cache || !cache->slots || block >= cache->n_blocks) return;
    slot = &cache->slots[block];
    if (slot->resident) {
        for (i = 0; i < H3_W4A8_LINEAR_COUNT; i++) {
            if (!slot->linears[i]) continue;
            if (!cache->recycle[i]) {
                cache->recycle[i] = slot->linears[i];
                slot->linears[i] = NULL;
            }
        }
    }
    free_slot(cache, block);
}

static int try_recycle_upload(h3_w4a8_cache *cache, h3_w4a8_linear_kind kind,
                              int8_t *h_w, size_t n, size_t k, float *h_s,
                              h3_cuda_w4a8_weight **out, char *error,
                              size_t error_size) {
    h3_cuda_w4a8_weight *dw;
    char detail[256];
    if (!cache || (unsigned)kind >= (unsigned)H3_W4A8_LINEAR_COUNT || !out)
        return 0;
    dw = cache->recycle[kind];
    if (!dw || h3_cuda_w4a8_weight_n(dw) != n ||
        h3_cuda_w4a8_weight_k(dw) != k)
        return 0;
    detail[0] = '\0';
    if (!h3_cuda_w4a8_weight_upload_i8(dw, h_w, n, k, h_s, 1, detail,
                                       sizeof(detail))) {
        if (error && error_size)
            snprintf(error, error_size, "INT8 recycle upload: %s", detail);
        return 0;
    }
    cache->recycle[kind] = NULL;
    cache->recycle_hits++;
    if (cache->recycle_hits == 1 || (cache->recycle_hits % 200u) == 0)
        printf("dit_int8_recycle: hits=%u (H2D into parked d_w_i8)\n",
               cache->recycle_hits);
    *out = dw;
    return 1;
}

static void fail(char *error, size_t error_size, const char *msg) {
    if (!error || !error_size) return;
    snprintf(error, error_size, "%s", msg ? msg : "W4A8 cache error");
}

static void cache_init_host_lock(h3_w4a8_cache *cache) {
    if (!cache || cache->host_mu_ready) return;
    if (pthread_mutex_init(&cache->host_mu, NULL) == 0)
        cache->host_mu_ready = 1;
}

static double cache_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

void h3_w4a8_cache_config_init(h3_w4a8_cache_config *cfg) {
    if (!cfg) return;
    memset(cfg, 0, sizeof(*cfg));
    cfg->n_blocks = H3_W4A8_CACHE_DEFAULT_BLOCKS;
    cfg->policy = H3_W4A8_CACHE_RESIDENT_ALL;
    cfg->window_size = 0;
    cfg->vram_budget_bytes = 0;
    cfg->vram_reserve_bytes = H3_W4A8_CACHE_DEFAULT_RESERVE_BYTES;
    {
        const char *e = getenv("H3_W4A8_CACHE_RESERVE_MIB");
        if (e && e[0]) {
            long mib = strtol(e, NULL, 10);
            if (mib > 0 && mib < 24000)
                cfg->vram_reserve_bytes = (size_t)mib << 20;
        }
    }
}

unsigned h3_w4a8_cache_window_size(const h3_w4a8_cache *cache) {
    return cache ? cache->window_size : 0;
}

unsigned h3_w4a8_cache_max_blocks_fit(const h3_w4a8_cache *cache) {
    return cache ? cache->max_blocks_fit : 0;
}

unsigned h3_w4a8_cache_n_resident(const h3_w4a8_cache *cache) {
    return cache ? cache->n_resident : 0;
}

unsigned h3_w4a8_cache_int8_host_hits(const h3_w4a8_cache *cache) {
    return cache ? cache->int8_host_hits : 0;
}

size_t h3_w4a8_cache_bytes_per_block(const h3_w4a8_cache *cache) {
    return cache ? cache->bytes_per_block_est : 0;
}

void h3_w4a8_cache_log_window(const h3_w4a8_cache *cache, const char *tag) {
    size_t free_b = 0, total_b = 0;
    size_t sage_b;
    char mem_err[128];
    if (!cache)
        return;
    mem_err[0] = '\0';
    (void)h3_cuda_mem_info(cache->cuda, &free_b, &total_b, mem_err,
                           sizeof(mem_err));
    sage_b = h3_cuda_sage3_pool_bytes();
    printf("dit_int8_cache: tag=%s window=%u max_fit=%u resident=%u "
           "host_hits=%u free_gib=%.2f sage_pool_mib=%.1f\n",
           tag && tag[0] ? tag : "-", cache->window_size, cache->max_blocks_fit,
           cache->n_resident, cache->int8_host_hits,
           (double)free_b / (1024.0 * 1024.0 * 1024.0),
           (double)sage_b / (1024.0 * 1024.0));
    fflush(stdout);
}

const char *h3_w4a8_cache_policy_name(h3_w4a8_cache_policy policy) {
    switch (policy) {
    case H3_W4A8_CACHE_RESIDENT_ALL:
        return "resident_all";
    case H3_W4A8_CACHE_WINDOW:
        return "window";
    case H3_W4A8_CACHE_LRU:
        return "lru";
    default:
        return "unknown";
    }
}

const char *h3_w4a8_linear_kind_name(h3_w4a8_linear_kind kind) {
    switch (kind) {
    case H3_W4A8_LINEAR_QKV:
        return "qkv";
    case H3_W4A8_LINEAR_OUT:
        return "out";
    case H3_W4A8_LINEAR_FC1:
        return "fc1";
    case H3_W4A8_LINEAR_FC2:
        return "fc2";
    default:
        return "?";
    }
}

const char *h3_w4a8_linear_kind_suffix(h3_w4a8_linear_kind kind) {
    switch (kind) {
    case H3_W4A8_LINEAR_QKV:
        return H3_W4A8_SUFFIX_QKV;
    case H3_W4A8_LINEAR_OUT:
        return H3_W4A8_SUFFIX_OUT;
    case H3_W4A8_LINEAR_FC1:
        return H3_W4A8_SUFFIX_FC1;
    case H3_W4A8_LINEAR_FC2:
        return H3_W4A8_SUFFIX_FC2;
    default:
        return NULL;
    }
}

static size_t estimate_quad_resident_bytes(const h3_w4a8_quad *quad) {
    if (!quad || !quad->weight || quad->weight->ndim != 2) return 0;
    size_t n = (size_t)quad->weight->shape[0];
    size_t k_half = (size_t)quad->weight->shape[1];
    size_t k = k_half * 2;
    return n * k + n * sizeof(float);
}

static size_t estimate_int8_linear_bytes(const h3_dit_int8_linear *lin) {
    if (!lin || !lin->weight || lin->n == 0 || lin->k == 0) return 0;
    return lin->n * lin->k + lin->n * sizeof(float);
}

static size_t estimate_block_bytes(const h3_w4a8_store *store,
                                   unsigned block) {
    size_t total = 0;
    for (int k = 0; k < H3_W4A8_LINEAR_COUNT; k++) {
        h3_w4a8_quad quad;
        char err[128];
        if (!h3_w4a8_find_block_linear(store, block,
                                       h3_w4a8_linear_kind_suffix(
                                           (h3_w4a8_linear_kind)k),
                                       &quad, err, sizeof(err))) {
            continue;
        }
        total += estimate_quad_resident_bytes(&quad);
    }
    return total;
}

static size_t estimate_block_bytes_nvfp4(const h3_nvfp4_store *store,
                                         unsigned block) {
    size_t total = 0;
    int k;
    for (k = 0; k < H3_W4A8_LINEAR_COUNT; k++) {
        h3_nvfp4_linear lin;
        char err[128];
        if (!h3_dit_nvfp4_find_block_linear(store, block,
                                            h3_w4a8_linear_kind_suffix(
                                                (h3_w4a8_linear_kind)k),
                                            &lin, err, sizeof(err)))
            continue;
        if (lin.n && lin.k) total += lin.n * lin.k + lin.n * sizeof(float);
    }
    return total;
}

static size_t estimate_block_bytes_int8(const h3_dit_int8_store *store,
                                        unsigned block) {
    size_t total = 0;
    for (int k = 0; k < H3_W4A8_LINEAR_COUNT; k++) {
        h3_dit_int8_linear lin;
        char err[128];
        if (!h3_dit_int8_find_block_linear(store, block,
                                           h3_w4a8_linear_kind_suffix(
                                               (h3_w4a8_linear_kind)k),
                                           &lin, err, sizeof(err))) {
            continue;
        }
        total += estimate_int8_linear_bytes(&lin);
    }
    return total;
}

static void free_int8_host_linear(h3_w4a8_int8_host_linear *h) {
    if (!h) return;
    if (!h->from_arena) {
        h3_cuda_host_free(h->h_w, h->pinned_w);
        h3_cuda_host_free(h->h_s, h->pinned_s);
    }
    memset(h, 0, sizeof(*h));
}

static void free_slot(h3_w4a8_cache *cache, unsigned block) {
    if (!cache || block >= cache->n_blocks) return;
    h3_w4a8_cache_slot *slot = &cache->slots[block];
    if (!slot->resident) return;
    for (int i = 0; i < H3_W4A8_LINEAR_COUNT; i++) {
        if (slot->linears[i]) {
            h3_cuda_w4a8_weight_free(slot->linears[i]);
            slot->linears[i] = NULL;
        }
        /* Keep INT8 host staging across device eviction (reload = H2D only). */
    }
    if (cache->bytes_resident >= slot->bytes)
        cache->bytes_resident -= slot->bytes;
    else
        cache->bytes_resident = 0;
    slot->bytes = 0;
    slot->resident = 0;
    slot->last_used = 0;
    if (cache->n_resident > 0) cache->n_resident--;
}

static int load_one_linear_nvfp4(h3_w4a8_cache *cache, unsigned block,
                                 h3_w4a8_linear_kind kind,
                                 h3_cuda_w4a8_weight **out, size_t *bytes_out,
                                 char *error, size_t error_size) {
    char detail[384];
    h3_nvfp4_linear lin;
    const char *suffix = h3_w4a8_linear_kind_suffix(kind);
    h3_w4a8_int8_host_linear *host;
    h3_cuda_w4a8_weight *dw = NULL;
    int8_t *tmp_i8 = NULL;
    float *tmp_s = NULL;
    size_t n = 0, k = 0, w_bytes, s_bytes;

    if (out) *out = NULL;
    if (bytes_out) *bytes_out = 0;
    if (!suffix || !cache || !cache->nvfp4_store) {
        fail(error, error_size, "invalid NVFP4 linear kind");
        return 0;
    }
    if (!h3_dit_nvfp4_find_block_linear(cache->nvfp4_store, block, suffix, &lin,
                                        detail, sizeof(detail))) {
        snprintf(error, error_size, "NVFP4 blocks.%u.%s: %s", block,
                 h3_w4a8_linear_kind_name(kind), detail);
        return 0;
    }
    host = &cache->slots[block].host[kind];
    if (host->h_w && host->h_s && host->n == lin.n && host->k == lin.k) {
        cache->int8_host_hits++;
        if (!h3_cuda_w4a8_weight_create_from_i8(cache->cuda, host->h_w, lin.n,
                                                lin.k, host->h_s, &dw, NULL, 1,
                                                detail, sizeof(detail))) {
            snprintf(error, error_size, "NVFP4 INT8 upload failed: %s", detail);
            return 0;
        }
        if (bytes_out) *bytes_out = lin.n * lin.k + lin.n * sizeof(float);
        *out = dw;
        return 1;
    }
    n = lin.n;
    k = lin.k;
    w_bytes = n * k;
    s_bytes = n * sizeof(float);
    tmp_i8 = (int8_t *)malloc(w_bytes);
    tmp_s = (float *)malloc(s_bytes);
    if (tmp_i8 && tmp_s &&
        h3_dit_nvfp4_sidecar_try_load(cache->nvfp4_store->path, block,
                                      h3_w4a8_linear_kind_name(kind), n, k,
                                      tmp_i8, tmp_s)) {
        cache->nvfp4_sidecar_hits++;
    } else {
        free(tmp_i8);
        free(tmp_s);
        tmp_i8 = NULL;
        tmp_s = NULL;
        cache->nvfp4_sidecar_misses++;
        if (!h3_dit_nvfp4_dequant_to_i8(cache->nvfp4_store, &lin, &tmp_i8,
                                        &tmp_s, &n, &k, detail,
                                        sizeof(detail))) {
            snprintf(error, error_size, "NVFP4 dequant: %s", detail);
            return 0;
        }
        if (h3_dit_nvfp4_sidecar_save(cache->nvfp4_store->path, block,
                                      h3_w4a8_linear_kind_name(kind), n, k,
                                      tmp_i8, tmp_s, detail, sizeof(detail)))
            cache->nvfp4_sidecar_writes++;
    }
    w_bytes = n * k;
    s_bytes = n * sizeof(float);
    free_int8_host_linear(host);
    host->h_w = (int8_t *)h3_cuda_host_alloc(w_bytes, &host->pinned_w);
    host->h_s = (float *)h3_cuda_host_alloc(s_bytes, &host->pinned_s);
    if (host->h_w && host->h_s) {
        memcpy(host->h_w, tmp_i8, w_bytes);
        memcpy(host->h_s, tmp_s, s_bytes);
        host->n = n;
        host->k = k;
        host->ready = 1;
        free(tmp_i8);
        free(tmp_s);
        tmp_i8 = NULL;
        tmp_s = NULL;
        cache->int8_disk_loads++;
        if (!h3_cuda_w4a8_weight_create_from_i8(cache->cuda, host->h_w, n, k,
                                                host->h_s, &dw, NULL, 1, detail,
                                                sizeof(detail))) {
            snprintf(error, error_size, "NVFP4 INT8 upload failed: %s", detail);
            return 0;
        }
    } else {
        free_int8_host_linear(host);
        if (!h3_cuda_w4a8_weight_create_from_i8(cache->cuda, tmp_i8, n, k, tmp_s,
                                                &dw, NULL, 0, detail,
                                                sizeof(detail))) {
            snprintf(error, error_size, "NVFP4 INT8 upload failed: %s", detail);
            free(tmp_i8);
            free(tmp_s);
            return 0;
        }
        cache->int8_disk_loads++;
    }
    if (bytes_out) *bytes_out = w_bytes + s_bytes;
    *out = dw;
    return 1;
}

static int load_one_linear_int8(h3_w4a8_cache *cache, unsigned block,
                                h3_w4a8_linear_kind kind,
                                h3_cuda_w4a8_weight **out, size_t *bytes_out,
                                char *error, size_t error_size) {
    char detail[384];
    h3_dit_int8_linear lin;
    const char *suffix = h3_w4a8_linear_kind_suffix(kind);
    h3_w4a8_int8_host_linear *host;
    h3_cuda_w4a8_weight *dw = NULL;
    size_t w_bytes = 0, s_bytes = 0;

    if (out) *out = NULL;
    if (bytes_out) *bytes_out = 0;
    if (!suffix) {
        fail(error, error_size, "invalid linear kind");
        return 0;
    }
    if (!h3_dit_int8_find_block_linear(cache->int8_store, block, suffix, &lin,
                                       detail, sizeof(detail))) {
        snprintf(error, error_size, "blocks.%u.%s: %s", block,
                 h3_w4a8_linear_kind_name(kind), detail);
        return 0;
    }
    w_bytes = lin.n * lin.k;
    s_bytes = lin.n * sizeof(float);
    host = &cache->slots[block].host[kind];

    if (cache->host_mu_ready)
        pthread_mutex_lock(&cache->host_mu);
    if (host->h_w && !host->ready) {
        int spins = 0;
        while (host->h_w && !host->ready && spins < 20000) {
            if (cache->host_mu_ready)
                pthread_mutex_unlock(&cache->host_mu);
            usleep(200);
            if (cache->host_mu_ready)
                pthread_mutex_lock(&cache->host_mu);
            spins++;
        }
    }
    if (host->ready && host->h_w && host->h_s && host->n == lin.n &&
        host->k == lin.k) {
        if (cache->host_mu_ready)
            pthread_mutex_unlock(&cache->host_mu);
        cache->int8_host_hits++;
    } else {
        if (cache->host_mu_ready)
            pthread_mutex_unlock(&cache->host_mu);
        free_int8_host_linear(host);
        host->h_w = (int8_t *)h3_cuda_host_alloc(w_bytes, &host->pinned_w);
        host->h_s = (float *)h3_cuda_host_alloc(s_bytes, &host->pinned_s);
        if (!host->h_w || !host->h_s) {
            /* Host staging OOM: ephemeral upload (no cross-evict reuse). */
            int8_t *w_i8;
            float *scales;
            free_int8_host_linear(host);
            w_i8 = (int8_t *)malloc(w_bytes);
            scales = (float *)malloc(s_bytes);
            if (!w_i8 || !scales) {
                free(w_i8);
                free(scales);
                fail(error, error_size, "OOM INT8 DiT linear host buffers");
                return 0;
            }
            if (!h3_dit_int8_read_tensor(cache->int8_store, lin.weight, w_i8,
                                         w_bytes, detail, sizeof(detail)) ||
                !h3_dit_int8_read_tensor(cache->int8_store, lin.scale, scales,
                                         s_bytes, detail, sizeof(detail))) {
                snprintf(error, error_size, "INT8 read failed: %s", detail);
                free(w_i8);
                free(scales);
                return 0;
            }
            if (!h3_cuda_w4a8_weight_create_from_i8(
                    cache->cuda, w_i8, lin.n, lin.k, scales, &dw, NULL,
                    /*borrow_host=*/0, detail, sizeof(detail))) {
                snprintf(error, error_size, "INT8 upload failed: %s", detail);
                free(w_i8);
                free(scales);
                return 0;
            }
            cache->int8_disk_loads++;
            if (bytes_out) *bytes_out = w_bytes + s_bytes;
            *out = dw;
            return 1;
        }
        if (!h3_dit_int8_read_tensor(cache->int8_store, lin.weight, host->h_w,
                                     w_bytes, detail, sizeof(detail)) ||
            !h3_dit_int8_read_tensor(cache->int8_store, lin.scale, host->h_s,
                                     s_bytes, detail, sizeof(detail))) {
            snprintf(error, error_size, "INT8 read failed: %s", detail);
            free_int8_host_linear(host);
            return 0;
        }
        host->n = lin.n;
        host->k = lin.k;
        host->ready = 1;
        cache->int8_disk_loads++;
    }

    /* Always borrow: host staging lives in the cache slot. */
    if (try_recycle_upload(cache, kind, host->h_w, lin.n, lin.k, host->h_s,
                           &dw, error, error_size)) {
        if (bytes_out) *bytes_out = w_bytes + s_bytes;
        *out = dw;
        return 1;
    }
    if (!h3_cuda_w4a8_weight_create_from_i8(cache->cuda, host->h_w, lin.n,
                                            lin.k, host->h_s, &dw, NULL,
                                            /*borrow_host=*/1, detail,
                                            sizeof(detail))) {
        snprintf(error, error_size, "INT8 upload failed: %s", detail);
        return 0;
    }
    if (bytes_out) *bytes_out = w_bytes + s_bytes;
    *out = dw;
    return 1;
}

static int load_one_linear(h3_w4a8_cache *cache, unsigned block,
                           h3_w4a8_linear_kind kind,
                           h3_cuda_w4a8_weight **out, size_t *bytes_out,
                           char *error, size_t error_size) {
    char detail[384];
    h3_w4a8_quad quad;
    const char *suffix = h3_w4a8_linear_kind_suffix(kind);
    int8_t *packed = NULL;
    float *codebook = NULL;
    uint8_t *s_rel = NULL;
    float *s_channel = NULL;
    size_t nbytes = 0;
    h3_cuda_w4a8_weight *dw = NULL;

    if (out) *out = NULL;
    if (bytes_out) *bytes_out = 0;
    if (cache && cache->nvfp4_store)
        return load_one_linear_nvfp4(cache, block, kind, out, bytes_out, error,
                                     error_size);
    if (cache && cache->int8_store)
        return load_one_linear_int8(cache, block, kind, out, bytes_out, error,
                                    error_size);
    if (!suffix) {
        fail(error, error_size, "invalid linear kind");
        return 0;
    }
    if (!h3_w4a8_find_block_linear(cache->store, block, suffix, &quad, detail,
                                   sizeof(detail))) {
        snprintf(error, error_size, "blocks.%u.%s: %s", block,
                 h3_w4a8_linear_kind_name(kind), detail);
        return 0;
    }
    if (!quad.weight || quad.weight->ndim != 2) {
        fail(error, error_size, "packed weight shape invalid");
        return 0;
    }

    packed = (int8_t *)h3_w4a8_dup_tensor(cache->store, quad.weight, &nbytes,
                                          detail, sizeof(detail));
    codebook = (float *)h3_w4a8_dup_tensor(cache->store, quad.weight_codebook,
                                           &nbytes, detail, sizeof(detail));
    s_rel = (uint8_t *)h3_w4a8_dup_tensor(cache->store, quad.weight_s_rel,
                                          &nbytes, detail, sizeof(detail));
    s_channel = (float *)h3_w4a8_dup_tensor(
        cache->store, quad.weight_s_channel, &nbytes, detail, sizeof(detail));
    if (!packed || !codebook || !s_rel || !s_channel) {
        snprintf(error, error_size, "dup tensor failed: %s", detail);
        free(packed);
        free(codebook);
        free(s_rel);
        free(s_channel);
        return 0;
    }

    {
        size_t n = (size_t)quad.weight->shape[0];
        size_t k_half = (size_t)quad.weight->shape[1];
        if (!h3_cuda_w4a8_weight_create(cache->cuda, packed, n, k_half,
                                        codebook, s_rel, s_channel,
                                        cache->group_size, &dw, NULL, NULL,
                                        detail, sizeof(detail))) {
            snprintf(error, error_size, "device unpack failed: %s", detail);
            free(packed);
            free(codebook);
            free(s_rel);
            free(s_channel);
            return 0;
        }
        if (bytes_out) *bytes_out = n * (k_half * 2) + n * sizeof(float);
    }

    free(packed);
    free(codebook);
    free(s_rel);
    free(s_channel);
    *out = dw;
    return 1;
}

static int load_one_into_slot(h3_w4a8_cache *cache, unsigned block,
                              h3_w4a8_linear_kind kind, char *error,
                              size_t error_size) {
    h3_w4a8_cache_slot *slot = &cache->slots[block];
    size_t b = 0;
    if (slot->linears[kind]) return 1;
    if (!load_one_linear(cache, block, kind, &slot->linears[kind], &b, error,
                         error_size)) {
        if (slot->linears[kind]) {
            h3_cuda_w4a8_weight_free(slot->linears[kind]);
            slot->linears[kind] = NULL;
        }
        return 0;
    }
    if (!slot->resident) {
        slot->resident = 1;
        slot->last_used = ++cache->tick;
        cache->n_resident++;
    }
    slot->bytes += b;
    cache->bytes_resident += b;
    return 1;
}

static int load_block(h3_w4a8_cache *cache, unsigned block, char *error,
                      size_t error_size) {
    h3_w4a8_cache_slot *slot;
    if (!cache || block >= cache->n_blocks) {
        fail(error, error_size, "block out of range");
        return 0;
    }
    slot = &cache->slots[block];
    if (slot->resident) return 1;
    /* nvfp4 track: FP4 owns the 4 linears; INT8 is lazy-only via get_linear. */
    if (cache->nvfp4_int8_skip) return 1;

    for (int i = 0; i < H3_W4A8_LINEAR_COUNT; i++) {
        if (!load_one_into_slot(cache, block, (h3_w4a8_linear_kind)i, error,
                                error_size)) {
            /* roll back partial */
            for (int j = 0; j <= i; j++) {
                if (slot->linears[j]) {
                    h3_cuda_w4a8_weight_free(slot->linears[j]);
                    slot->linears[j] = NULL;
                }
            }
            if (slot->resident) {
                slot->resident = 0;
                slot->bytes = 0;
                if (cache->n_resident > 0) cache->n_resident--;
            }
            return 0;
        }
    }

    /* INT8 path queues H2D on the side stream. U=1 async_h2d leaves the
     * ready event for GEMM wait_ready so next-block H2D overlaps compute.
     * Ulysses keeps the per-block drain (default). */
    if (cache->int8_store && cache->cuda && !cache->async_h2d) {
        char detail[256];
        if (!h3_cuda_copy_stream_synchronize(cache->cuda, detail,
                                             sizeof(detail))) {
            snprintf(error, error_size, "INT8 block H2D: %s", detail);
            for (int j = 0; j < H3_W4A8_LINEAR_COUNT; j++) {
                if (slot->linears[j]) {
                    h3_cuda_w4a8_weight_free(slot->linears[j]);
                    slot->linears[j] = NULL;
                }
            }
            return 0;
        }
        for (int j = 0; j < H3_W4A8_LINEAR_COUNT; j++) {
            if (slot->linears[j] &&
                !h3_cuda_w4a8_weight_ensure_ready(slot->linears[j], detail,
                                                  sizeof(detail))) {
                snprintf(error, error_size, "INT8 ensure_ready: %s", detail);
                for (int k = 0; k < H3_W4A8_LINEAR_COUNT; k++) {
                    if (slot->linears[k]) {
                        h3_cuda_w4a8_weight_free(slot->linears[k]);
                        slot->linears[k] = NULL;
                    }
                }
                return 0;
            }
        }
    }

    slot->last_used = ++cache->tick;
    return 1;
}

static unsigned find_lru_victim(const h3_w4a8_cache *cache,
                                unsigned keep_block) {
    unsigned victim = cache->n_blocks;
    uint64_t best = UINT64_MAX;
    for (unsigned b = 0; b < cache->n_blocks; b++) {
        if (b == keep_block) continue;
        if (!cache->slots[b].resident) continue;
        if (cache->slots[b].last_used < best) {
            best = cache->slots[b].last_used;
            victim = b;
        }
    }
    return victim;
}

static int ensure_capacity_for(h3_w4a8_cache *cache, unsigned block,
                               char *error, size_t error_size) {
    if (cache->slots[block].resident) return 1;

    if (cache->effective_policy == H3_W4A8_CACHE_RESIDENT_ALL) {
        /* no eviction; just load (budget checked at open) */
        return 1;
    }

    if (cache->effective_policy == H3_W4A8_CACHE_WINDOW) {
        unsigned w = cache->window_size;
        unsigned start;
        if (w == 0) w = 1;
        if (block + 1 >= w)
            start = block + 1 - w;
        else
            start = 0;
        unsigned end = start + w; /* exclusive */
        if (end > cache->n_blocks) {
            end = cache->n_blocks;
            start = end > w ? end - w : 0;
        }
        for (unsigned b = 0; b < cache->n_blocks; b++) {
            if (!cache->slots[b].resident) continue;
            if (b < start || b >= end) steal_or_free_slot(cache, b);
        }
        return 1;
    }

    /* LRU */
    while (cache->n_resident >= cache->window_size) {
        unsigned victim = find_lru_victim(cache, block);
        if (victim >= cache->n_blocks) break;
        steal_or_free_slot(cache, victim);
    }
    (void)error;
    (void)error_size;
    return 1;
}

h3_w4a8_cache *h3_w4a8_cache_open(h3_cuda *cuda, h3_w4a8_store *store,
                                  const h3_w4a8_cache_config *cfg, char *error,
                                  size_t error_size) {
    h3_w4a8_cache_config local;
    h3_w4a8_cache *cache = NULL;
    size_t free_b = 0, total_b = 0;
    char detail[256];

    if (error && error_size) error[0] = '\0';
    if (!cuda || !store) {
        fail(error, error_size, "cuda and store required");
        return NULL;
    }
    if (cfg)
        local = *cfg;
    else
        h3_w4a8_cache_config_init(&local);
    if (local.n_blocks == 0) local.n_blocks = H3_W4A8_CACHE_DEFAULT_BLOCKS;
    if (local.vram_reserve_bytes == 0)
        local.vram_reserve_bytes = H3_W4A8_CACHE_DEFAULT_RESERVE_BYTES;

    if (!h3_cuda_mem_info(cuda, &free_b, &total_b, detail, sizeof(detail))) {
        snprintf(error, error_size, "mem_info: %s", detail);
        return NULL;
    }

    cache = (h3_w4a8_cache *)calloc(1, sizeof(*cache));
    if (!cache) {
        fail(error, error_size, "out of memory");
        return NULL;
    }
    cache->cuda = cuda;
    cache_init_host_lock(cache);
    cache->store = store;
    cache->int8_store = NULL;
    cache->n_blocks = local.n_blocks;
    cache->requested_policy = local.policy;
    cache->effective_policy = local.policy;
    cache->vram_reserve_bytes = local.vram_reserve_bytes;
    cache->slots =
        (h3_w4a8_cache_slot *)calloc(cache->n_blocks, sizeof(*cache->slots));
    if (!cache->slots) {
        fail(error, error_size, "out of memory for slots");
        free(cache);
        return NULL;
    }

    {
        const h3_st_quant_info *q = h3_w4a8_store_quant(store);
        cache->group_size = (q && q->group_size > 0)
                                ? q->group_size
                                : H3_W4A8_DEFAULT_GROUP_SIZE;
    }

    /* Estimate from block 0 (all blocks share shapes in pruned W4A8). */
    cache->bytes_per_block_est = estimate_block_bytes(store, 0);
    if (cache->bytes_per_block_est == 0) {
        /* try a few blocks */
        for (unsigned b = 1; b < cache->n_blocks && b < 4; b++) {
            cache->bytes_per_block_est = estimate_block_bytes(store, b);
            if (cache->bytes_per_block_est) break;
        }
    }
    cache->bytes_estimated_all =
        cache->bytes_per_block_est * (size_t)cache->n_blocks;

    if (local.vram_budget_bytes)
        cache->vram_budget_bytes = local.vram_budget_bytes;
    else if (free_b > local.vram_reserve_bytes)
        cache->vram_budget_bytes = free_b - local.vram_reserve_bytes;
    else
        cache->vram_budget_bytes = free_b / 2;

    if (cache->bytes_per_block_est > 0) {
        cache->max_blocks_fit =
            (unsigned)(cache->vram_budget_bytes / cache->bytes_per_block_est);
        if (cache->max_blocks_fit > cache->n_blocks)
            cache->max_blocks_fit = cache->n_blocks;
    } else {
        cache->max_blocks_fit = 0;
    }

    if (cache->effective_policy == H3_W4A8_CACHE_RESIDENT_ALL) {
        if (cache->max_blocks_fit >= cache->n_blocks) {
            cache->window_size = cache->n_blocks;
        } else {
            /* Auto-downgrade to WINDOW with as many as fit (at least 1). */
            cache->auto_downgraded = 1;
            cache->effective_policy = H3_W4A8_CACHE_WINDOW;
            cache->window_size =
                cache->max_blocks_fit > 0 ? cache->max_blocks_fit : 1;
        }
    } else {
        if (local.window_size > 0)
            cache->window_size = local.window_size;
        else
            cache->window_size =
                cache->max_blocks_fit > 0 ? cache->max_blocks_fit : 1;
        if (cache->window_size > cache->n_blocks)
            cache->window_size = cache->n_blocks;
        if (cache->window_size > cache->max_blocks_fit &&
            cache->max_blocks_fit > 0)
            cache->window_size = cache->max_blocks_fit;
    }

    return cache;
}

h3_w4a8_cache *h3_w4a8_cache_open_int8(h3_cuda *cuda, h3_dit_int8_store *store,
                                       const h3_w4a8_cache_config *cfg,
                                       char *error, size_t error_size) {
    h3_w4a8_cache_config local;
    h3_w4a8_cache *cache = NULL;
    size_t free_b = 0, total_b = 0;
    char detail[256];

    if (error && error_size) error[0] = '\0';
    if (!cuda || !store) {
        fail(error, error_size, "cuda and INT8 store required");
        return NULL;
    }
    if (cfg)
        local = *cfg;
    else
        h3_w4a8_cache_config_init(&local);
    if (local.n_blocks == 0) local.n_blocks = H3_W4A8_CACHE_DEFAULT_BLOCKS;
    if (local.vram_reserve_bytes == 0)
        local.vram_reserve_bytes = H3_W4A8_CACHE_DEFAULT_RESERVE_BYTES;

    if (!h3_cuda_mem_info(cuda, &free_b, &total_b, detail, sizeof(detail))) {
        snprintf(error, error_size, "mem_info: %s", detail);
        return NULL;
    }

    cache = (h3_w4a8_cache *)calloc(1, sizeof(*cache));
    if (!cache) {
        fail(error, error_size, "out of memory");
        return NULL;
    }
    cache->cuda = cuda;
    cache_init_host_lock(cache);
    cache->store = NULL;
    cache->int8_store = store;
    cache->nvfp4_store = NULL;
    cache->n_blocks = local.n_blocks;
    cache->requested_policy = local.policy;
    cache->effective_policy = local.policy;
    cache->vram_reserve_bytes = local.vram_reserve_bytes;
    cache->group_size = 0; /* unused for INT8 upload path */
    cache->slots =
        (h3_w4a8_cache_slot *)calloc(cache->n_blocks, sizeof(*cache->slots));
    if (!cache->slots) {
        fail(error, error_size, "out of memory for slots");
        free(cache);
        return NULL;
    }

    cache->bytes_per_block_est = estimate_block_bytes_int8(store, 0);
    if (cache->bytes_per_block_est == 0) {
        for (unsigned b = 1; b < cache->n_blocks && b < 4; b++) {
            cache->bytes_per_block_est = estimate_block_bytes_int8(store, b);
            if (cache->bytes_per_block_est) break;
        }
    }
    cache->bytes_estimated_all =
        cache->bytes_per_block_est * (size_t)cache->n_blocks;

    if (local.vram_budget_bytes)
        cache->vram_budget_bytes = local.vram_budget_bytes;
    else if (free_b > local.vram_reserve_bytes)
        cache->vram_budget_bytes = free_b - local.vram_reserve_bytes;
    else
        cache->vram_budget_bytes = free_b / 2;

    if (cache->bytes_per_block_est > 0) {
        cache->max_blocks_fit =
            (unsigned)(cache->vram_budget_bytes / cache->bytes_per_block_est);
        if (cache->max_blocks_fit > cache->n_blocks)
            cache->max_blocks_fit = cache->n_blocks;
    } else {
        cache->max_blocks_fit = 0;
    }

    if (cache->effective_policy == H3_W4A8_CACHE_RESIDENT_ALL) {
        if (cache->max_blocks_fit >= cache->n_blocks) {
            cache->window_size = cache->n_blocks;
        } else {
            cache->auto_downgraded = 1;
            cache->effective_policy = H3_W4A8_CACHE_WINDOW;
            cache->window_size =
                cache->max_blocks_fit > 0 ? cache->max_blocks_fit : 1;
        }
    } else {
        if (local.window_size > 0)
            cache->window_size = local.window_size;
        else
            cache->window_size =
                cache->max_blocks_fit > 0 ? cache->max_blocks_fit : 1;
        if (cache->window_size > cache->n_blocks)
            cache->window_size = cache->n_blocks;
        if (cache->window_size > cache->max_blocks_fit &&
            cache->max_blocks_fit > 0)
            cache->window_size = cache->max_blocks_fit;
    }

    /* INT8 blocks are full I8[N,K] — max_fit is typically smaller than W4A8.
     * Host staging (pinned) kills re-pread on window reload; WS-first refit
     * may grow window to resident_all after activation workspace lands. */
    printf("dit_int8_cache: policy=%s window=%u max_fit=%u bytes/block≈%zu "
           "reserve_giB=%.2f host_stage=pinned%s\n",
           h3_w4a8_cache_policy_name(cache->effective_policy),
           cache->window_size, cache->max_blocks_fit,
           cache->bytes_per_block_est,
           (double)cache->vram_reserve_bytes /
               (1024.0 * 1024.0 * 1024.0),
           cache->auto_downgraded ? " auto_downgraded=1" : "");
    fflush(stdout);
    cache->open_window_cap = cache->window_size;
    h3_w4a8_cache_log_window(cache, "open");

    return cache;
}

h3_w4a8_cache *h3_w4a8_cache_open_nvfp4(h3_cuda *cuda, h3_nvfp4_store *store,
                                        const h3_w4a8_cache_config *cfg,
                                        char *error, size_t error_size) {
    h3_w4a8_cache_config local;
    h3_w4a8_cache *cache = NULL;
    size_t free_b = 0, total_b = 0;
    char detail[256];

    if (error && error_size) error[0] = '\0';
    if (!cuda || !store) {
        fail(error, error_size, "cuda and NVFP4 store required");
        return NULL;
    }
    if (cfg)
        local = *cfg;
    else
        h3_w4a8_cache_config_init(&local);
    if (local.n_blocks == 0) local.n_blocks = H3_W4A8_CACHE_DEFAULT_BLOCKS;
    if (local.vram_reserve_bytes == 0)
        local.vram_reserve_bytes = H3_W4A8_CACHE_DEFAULT_RESERVE_BYTES;

    if (!h3_cuda_mem_info(cuda, &free_b, &total_b, detail, sizeof(detail))) {
        snprintf(error, error_size, "mem_info: %s", detail);
        return NULL;
    }

    cache = (h3_w4a8_cache *)calloc(1, sizeof(*cache));
    if (!cache) {
        fail(error, error_size, "out of memory");
        return NULL;
    }
    cache->cuda = cuda;
    cache_init_host_lock(cache);
    cache->store = NULL;
    cache->int8_store = NULL;
    cache->nvfp4_store = store;
    cache->n_blocks = local.n_blocks;
    cache->requested_policy = local.policy;
    cache->effective_policy = local.policy;
    cache->vram_reserve_bytes = local.vram_reserve_bytes;
    cache->group_size = 0;
    cache->slots =
        (h3_w4a8_cache_slot *)calloc(cache->n_blocks, sizeof(*cache->slots));
    if (!cache->slots) {
        fail(error, error_size, "out of memory for slots");
        free(cache);
        return NULL;
    }

    cache->bytes_per_block_est = estimate_block_bytes_nvfp4(store, 0);
    if (cache->bytes_per_block_est == 0) {
        unsigned b;
        for (b = 1; b < cache->n_blocks && b < 4; b++) {
            cache->bytes_per_block_est = estimate_block_bytes_nvfp4(store, b);
            if (cache->bytes_per_block_est) break;
        }
    }
    cache->bytes_estimated_all =
        cache->bytes_per_block_est * (size_t)cache->n_blocks;

    if (local.vram_budget_bytes)
        cache->vram_budget_bytes = local.vram_budget_bytes;
    else if (free_b > local.vram_reserve_bytes)
        cache->vram_budget_bytes = free_b - local.vram_reserve_bytes;
    else
        cache->vram_budget_bytes = free_b / 2;

    if (cache->bytes_per_block_est > 0) {
        cache->max_blocks_fit =
            (unsigned)(cache->vram_budget_bytes / cache->bytes_per_block_est);
        if (cache->max_blocks_fit > cache->n_blocks)
            cache->max_blocks_fit = cache->n_blocks;
    } else {
        cache->max_blocks_fit = 0;
    }

    if (cache->effective_policy == H3_W4A8_CACHE_RESIDENT_ALL) {
        if (cache->max_blocks_fit >= cache->n_blocks) {
            cache->window_size = cache->n_blocks;
        } else {
            cache->auto_downgraded = 1;
            cache->effective_policy = H3_W4A8_CACHE_WINDOW;
            cache->window_size =
                cache->max_blocks_fit > 0 ? cache->max_blocks_fit : 1;
        }
    } else {
        if (local.window_size > 0)
            cache->window_size = local.window_size;
        else
            cache->window_size =
                cache->max_blocks_fit > 0 ? cache->max_blocks_fit : 1;
        if (cache->window_size > cache->n_blocks)
            cache->window_size = cache->n_blocks;
        if (cache->window_size > cache->max_blocks_fit &&
            cache->max_blocks_fit > 0)
            cache->window_size = cache->max_blocks_fit;
    }

    printf("dit_nvfp4_cache: policy=%s window=%u max_fit=%u bytes/block≈%zu "
           "reserve_giB=%.2f dequant→INT8 GEMM (非金线)%s\n",
           h3_w4a8_cache_policy_name(cache->effective_policy),
           cache->window_size, cache->max_blocks_fit,
           cache->bytes_per_block_est,
           (double)cache->vram_reserve_bytes / (1024.0 * 1024.0 * 1024.0),
           cache->auto_downgraded ? " auto_downgraded=1" : "");
    fflush(stdout);
    cache->open_window_cap = cache->window_size;
    return cache;
}

void h3_w4a8_cache_free(h3_w4a8_cache *cache) {
    if (!cache) return;
    if (cache->slots) {
        for (unsigned b = 0; b < cache->n_blocks; b++) {
            free_slot(cache, b);
            for (int i = 0; i < H3_W4A8_LINEAR_COUNT; i++)
                free_int8_host_linear(&cache->slots[b].host[i]);
        }
        free(cache->slots);
    }
    recycle_flush(cache);
    if (cache->host_arena) {
        h3_cuda_host_free(cache->host_arena, cache->host_arena_pinned);
        cache->host_arena = NULL;
    }
    if (cache->host_mmap) {
        if (cache->host_mmap_registered)
            h3_cuda_host_unregister(cache->host_mmap);
        munmap(cache->host_mmap, cache->host_mmap_len);
        cache->host_mmap = NULL;
        cache->host_mmap_registered = 0;
    }
    if (cache->host_mu_ready) {
        pthread_mutex_destroy(&cache->host_mu);
        cache->host_mu_ready = 0;
    }
    free(cache);
}

int h3_w4a8_cache_refit_after_workspace(h3_w4a8_cache *cache, char *error,
                                        size_t error_size) {
    size_t free_b = 0, total_b = 0;
    size_t reserve;
    size_t budget;
    unsigned fit;
    char detail[256];

    if (error && error_size) error[0] = '\0';
    if (!cache) {
        fail(error, error_size, "cache is NULL");
        return 0;
    }
    /* INT8-only: WS-first frees headroom that open-time 8GiB reserve hid. */
    if ((!cache->int8_store && !cache->nvfp4_store) || cache->refit_done)
        return 1;
    /* nvfp4 track: INT8 blocks are never pre-resident; nothing to refit. */
    if (cache->nvfp4_int8_skip) {
        cache->refit_done = 1;
        return 1;
    }
    cache->refit_done = 1;

    if (!h3_cuda_mem_info(cache->cuda, &free_b, &total_b, detail,
                          sizeof(detail))) {
        snprintf(error, error_size, "refit mem_info: %s", detail);
        return 0;
    }
    /* Open-time default 8192MiB is for the case refit runs BEFORE pregrow
     * (else gemm acc OOM'd when 1536MiB grew window 41→47). After
     * h3_cuda_pregrow_dit_scratch those buffers are already in 'used', so
     * subtracting another 8GiB from remaining free double-counts and
     * clamped cafe_p2 to window=19 (host_hits=292). Keep a 1536MiB
     * fragmentation cushion. Env H3_W4A8_CACHE_REFIT_RESERVE_MIB wins. */
    reserve = (size_t)8192ULL << 20;
    if (h3_cuda_dit_scratch_pregrown(cache->cuda))
        reserve = (size_t)1536ULL << 20;
    if (cache->vram_reserve_bytes &&
        cache->vram_reserve_bytes < H3_W4A8_CACHE_DEFAULT_RESERVE_BYTES)
        reserve = cache->vram_reserve_bytes;
    {
        const char *e = getenv("H3_W4A8_CACHE_REFIT_RESERVE_MIB");
        if (e && *e) {
            long mib = strtol(e, NULL, 10);
            if (mib > 0) reserve = (size_t)mib << 20;
        }
    }
    if (cache->bytes_per_block_est == 0) return 1;
    {
        size_t loaded = cache->bytes_resident;
        size_t usable = free_b + loaded;
        unsigned cap = cache->open_window_cap > 0 ? cache->open_window_cap
                       : (cache->window_size > 0 ? cache->window_size : 1);
        unsigned keep = 1;
        /* Never grow past open-time cap (1536MiB → 47 → gemm OOM).
         * Use open_window_cap, not the possibly-shrunk window_size, so a
         * pre-refit make_room cannot pin us at 19. */
        if (usable <= reserve) {
            fit = keep;
        } else {
            budget = usable - reserve;
            fit = (unsigned)(budget / cache->bytes_per_block_est);
            if (fit < keep) fit = keep;
            if (fit > cap) fit = cap;
            if (fit < 1) fit = 1;
            if (fit > cache->n_blocks) fit = cache->n_blocks;
        }
    }
    if (fit == cache->window_size &&
        (fit < cache->n_blocks ||
         cache->effective_policy != H3_W4A8_CACHE_RESIDENT_ALL)) {
        printf("dit_int8_cache_refit: no_change window=%u fit=%u free_giB=%.2f "
               "reserve_giB=%.2f\n",
               cache->window_size, fit,
               (double)free_b / (1024.0 * 1024.0 * 1024.0),
               (double)reserve / (1024.0 * 1024.0 * 1024.0));
        fflush(stdout);
        return 1;
    }
    cache->window_size = fit;
    cache->max_blocks_fit = fit;
    cache->vram_budget_bytes = (free_b > reserve) ? (free_b - reserve) : 0;
    if (fit >= cache->n_blocks) {
        cache->effective_policy = H3_W4A8_CACHE_RESIDENT_ALL;
        cache->auto_downgraded = 0;
    } else {
        cache->effective_policy = H3_W4A8_CACHE_WINDOW;
        cache->auto_downgraded = 1;
    }
    while (cache->n_resident > cache->window_size) {
        unsigned victim = find_lru_victim(cache, cache->n_blocks);
        if (victim >= cache->n_blocks) break;
        free_slot(cache, victim);
    }
    printf("dit_int8_cache_refit: policy=%s window=%u max_fit=%u "
           "free_giB=%.2f reserve_giB=%.2f pregrown=%d disk_loads=%u "
           "host_hits=%u%s\n",
           h3_w4a8_cache_policy_name(cache->effective_policy),
           cache->window_size, cache->max_blocks_fit,
           (double)free_b / (1024.0 * 1024.0 * 1024.0),
           (double)reserve / (1024.0 * 1024.0 * 1024.0),
           h3_cuda_dit_scratch_pregrown(cache->cuda) ? 1 : 0,
           cache->int8_disk_loads, cache->int8_host_hits,
           fit < cache->n_blocks ? " (clamped)" : "");
    fflush(stdout);
    h3_w4a8_cache_log_window(cache, "refit");
    return 1;
}

void h3_w4a8_cache_freeze_window(h3_w4a8_cache *cache) {
    const char *e;
    if (!cache || cache->window_frozen) return;
    e = getenv("H3_DIT_INT8_ALLOW_SHRINK");
    if (e && e[0] == '1' && e[1] == '\0') return;
    cache->window_frozen = 1;
    printf("dit_int8_cache_freeze: window=%u resident=%u (ALLOW_SHRINK=1 to "
           "unlock)\n",
           cache->window_size, cache->n_resident);
    fflush(stdout);
}

void h3_w4a8_cache_unfreeze_window(h3_w4a8_cache *cache) {
    if (!cache)
        return;
    cache->window_frozen = 0;
}

void h3_w4a8_cache_reset_refit(h3_w4a8_cache *cache) {
    if (!cache)
        return;
    cache->refit_done = 0;
}

void h3_w4a8_cache_drop_device_weights(h3_w4a8_cache *cache) {
    unsigned b;
    if (!cache || !cache->slots)
        return;
    for (b = 0; b < cache->n_blocks; b++)
        free_slot(cache, b);
    recycle_flush(cache);
}

void h3_w4a8_cache_set_window_size(h3_w4a8_cache *cache, unsigned n) {
    if (!cache)
        return;
    if (n < 1)
        n = 1;
    if (n > cache->n_blocks)
        n = cache->n_blocks;
    cache->window_size = n;
    cache->max_blocks_fit = n;
    cache->effective_policy = H3_W4A8_CACHE_WINDOW;
}

int h3_w4a8_cache_make_room(h3_w4a8_cache *cache, size_t keep_free_bytes,
                            char *error, size_t error_size) {
    size_t free_b = 0, total_b = 0;
    char detail[256];
    unsigned dropped = 0;

    if (error && error_size) error[0] = '\0';
    if (!cache) {
        fail(error, error_size, "cache is NULL");
        return 0;
    }
    if (keep_free_bytes == 0) return 1;
    if (cache->window_frozen) return 1;
    if (!h3_cuda_mem_info(cache->cuda, &free_b, &total_b, detail,
                          sizeof(detail))) {
        snprintf(error, error_size, "make_room mem_info: %s", detail);
        return 0;
    }
    while (free_b < keep_free_bytes && cache->n_resident > 0) {
        unsigned victim = find_lru_victim(cache, cache->n_blocks);
        if (victim >= cache->n_blocks) break;
        free_slot(cache, victim);
        dropped++;
        recycle_flush(cache);
        if (!h3_cuda_mem_info(cache->cuda, &free_b, &total_b, detail,
                              sizeof(detail)))
            break;
    }
    if (dropped) {
        unsigned fit = cache->n_resident > 0 ? cache->n_resident : 1;
        if (cache->window_size > fit) cache->window_size = fit;
        cache->max_blocks_fit = fit;
        printf("dit_int8_cache_make_room: dropped=%u resident=%u window=%u "
               "free_giB=%.2f keep_giB=%.2f\n",
               dropped, cache->n_resident, cache->window_size,
               (double)free_b / (1024.0 * 1024.0 * 1024.0),
               (double)keep_free_bytes / (1024.0 * 1024.0 * 1024.0));
        fflush(stdout);
    }
    return 1;
}

void h3_w4a8_cache_log_int8_host(const h3_w4a8_cache *cache) {
    static int logged;
    if (!cache || (!cache->int8_store && !cache->nvfp4_store) || logged)
        return;
    logged = 1;
    printf("dit_int8_host: disk_loads=%u host_hits=%u recycle_hits=%u "
           "(hits=reload without safetensors re-pread; recycle=no cudaMalloc)\n",
           cache->int8_disk_loads, cache->int8_host_hits, cache->recycle_hits);
    fflush(stdout);
}

int h3_w4a8_cache_prefetch(h3_w4a8_cache *cache, unsigned first_block,
                           unsigned n_blocks, double *elapsed_ms, char *error,
                           size_t error_size) {
    clock_t t0, t1;
    unsigned end;
    if (elapsed_ms) *elapsed_ms = 0.0;
    if (!cache) {
        fail(error, error_size, "cache is NULL");
        return 0;
    }
    if (first_block >= cache->n_blocks) {
        fail(error, error_size, "first_block out of range");
        return 0;
    }
    end = first_block + n_blocks;
    if (end > cache->n_blocks) end = cache->n_blocks;

    t0 = clock();
    /* nvfp4 track: FP4 serves the 4 linears; do not preload INT8 blocks. */
    if (cache->nvfp4_int8_skip) {
        t1 = clock();
        if (elapsed_ms)
            *elapsed_ms = 1000.0 * (double)(t1 - t0) / (double)CLOCKS_PER_SEC;
        return 1;
    }

    if (cache->effective_policy == H3_W4A8_CACHE_WINDOW ||
        cache->effective_policy == H3_W4A8_CACHE_LRU) {
        unsigned cap = cache->window_size;
        if (end - first_block > cap) end = first_block + cap;
    }

    t0 = clock();
    for (unsigned b = first_block; b < end; b++) {
        if (cache->effective_policy == H3_W4A8_CACHE_LRU) {
            while (cache->n_resident >= cache->window_size &&
                   !cache->slots[b].resident) {
                unsigned victim = find_lru_victim(cache, b);
                if (victim >= cache->n_blocks) break;
                steal_or_free_slot(cache, victim);
            }
        } else if (cache->effective_policy == H3_W4A8_CACHE_WINDOW) {
            if (!ensure_capacity_for(cache, b, error, error_size)) return 0;
        }
        if (!load_block(cache, b, error, error_size)) return 0;
    }
    t1 = clock();
    if (elapsed_ms)
        *elapsed_ms = 1000.0 * (double)(t1 - t0) / (double)CLOCKS_PER_SEC;
    return 1;
}

static int stage_one_host_nvfp4(h3_w4a8_cache *cache, unsigned block,
                                h3_w4a8_linear_kind kind, char *error,
                                size_t error_size) {
    char detail[384];
    h3_nvfp4_linear lin;
    const char *suffix = h3_w4a8_linear_kind_suffix(kind);
    h3_w4a8_int8_host_linear *host;
    int8_t *tmp_i8 = NULL;
    float *tmp_s = NULL;
    size_t n = 0, k = 0, w_bytes, s_bytes;
    if (!suffix || !cache->nvfp4_store) {
        fail(error, error_size, "invalid NVFP4 host stage");
        return 0;
    }
    if (!h3_dit_nvfp4_find_block_linear(cache->nvfp4_store, block, suffix, &lin,
                                        detail, sizeof(detail))) {
        snprintf(error, error_size, "NVFP4 host blocks.%u.%s: %s", block,
                 h3_w4a8_linear_kind_name(kind), detail);
        return 0;
    }
    host = &cache->slots[block].host[kind];
    if (host->h_w && host->h_s && host->n == lin.n && host->k == lin.k &&
        host->ready)
        return 1;
    n = lin.n;
    k = lin.k;
    w_bytes = n * k;
    s_bytes = n * sizeof(float);
    tmp_i8 = (int8_t *)malloc(w_bytes);
    tmp_s = (float *)malloc(s_bytes);
    if (tmp_i8 && tmp_s &&
        h3_dit_nvfp4_sidecar_try_load(cache->nvfp4_store->path, block,
                                      h3_w4a8_linear_kind_name(kind), n, k,
                                      tmp_i8, tmp_s)) {
        cache->nvfp4_sidecar_hits++;
    } else {
        free(tmp_i8);
        free(tmp_s);
        tmp_i8 = NULL;
        tmp_s = NULL;
        cache->nvfp4_sidecar_misses++;
        if (!h3_dit_nvfp4_dequant_to_i8(cache->nvfp4_store, &lin, &tmp_i8,
                                        &tmp_s, &n, &k, detail,
                                        sizeof(detail))) {
            snprintf(error, error_size, "NVFP4 host dequant: %s", detail);
            return 0;
        }
        if (h3_dit_nvfp4_sidecar_save(cache->nvfp4_store->path, block,
                                      h3_w4a8_linear_kind_name(kind), n, k,
                                      tmp_i8, tmp_s, detail, sizeof(detail)))
            cache->nvfp4_sidecar_writes++;
    }
    w_bytes = n * k;
    s_bytes = n * sizeof(float);
    if (cache->host_mu_ready)
        pthread_mutex_lock(&cache->host_mu);
    free_int8_host_linear(host);
    host->h_w = (int8_t *)h3_cuda_host_alloc(w_bytes, &host->pinned_w);
    host->h_s = (float *)h3_cuda_host_alloc(s_bytes, &host->pinned_s);
    if (!host->h_w || !host->h_s) {
        free_int8_host_linear(host);
        host->h_w = tmp_i8;
        host->h_s = tmp_s;
        host->pinned_w = host->pinned_s = 0;
        host->n = n;
        host->k = k;
        host->ready = 1;
        cache->int8_disk_loads++;
        if (cache->host_mu_ready)
            pthread_mutex_unlock(&cache->host_mu);
        return 1;
    }
    memcpy(host->h_w, tmp_i8, w_bytes);
    memcpy(host->h_s, tmp_s, s_bytes);
    host->n = n;
    host->k = k;
    host->ready = 1;
    if (cache->host_mu_ready)
        pthread_mutex_unlock(&cache->host_mu);
    free(tmp_i8);
    free(tmp_s);
    cache->int8_disk_loads++;
    return 1;
}

static int stage_one_host_int8(h3_w4a8_cache *cache, unsigned block,
                               h3_w4a8_linear_kind kind, char *error,
                               size_t error_size) {
    char detail[384];
    h3_dit_int8_linear lin;
    const char *suffix = h3_w4a8_linear_kind_suffix(kind);
    h3_w4a8_int8_host_linear *host;
    size_t w_bytes, s_bytes;
    if (!suffix) {
        fail(error, error_size, "invalid linear kind");
        return 0;
    }
    if (!h3_dit_int8_find_block_linear(cache->int8_store, block, suffix, &lin,
                                       detail, sizeof(detail))) {
        snprintf(error, error_size, "blocks.%u.%s: %s", block,
                 h3_w4a8_linear_kind_name(kind), detail);
        return 0;
    }
    host = &cache->slots[block].host[kind];
    if (host->h_w && host->h_s && host->n == lin.n && host->k == lin.k)
        return 1;
    w_bytes = lin.n * lin.k;
    s_bytes = lin.n * sizeof(float);
    free_int8_host_linear(host);
    host->h_w = (int8_t *)h3_cuda_host_alloc(w_bytes, &host->pinned_w);
    host->h_s = (float *)h3_cuda_host_alloc(s_bytes, &host->pinned_s);
    if (!host->h_w || !host->h_s) {
        free_int8_host_linear(host);
        host->h_w = (int8_t *)malloc(w_bytes);
        host->h_s = (float *)malloc(s_bytes);
        host->pinned_w = host->pinned_s = 0;
        if (!host->h_w || !host->h_s) {
            free_int8_host_linear(host);
            fail(error, error_size, "OOM INT8 host stage");
            return 0;
        }
    }
    if (!h3_dit_int8_read_tensor(cache->int8_store, lin.weight, host->h_w,
                                 w_bytes, detail, sizeof(detail)) ||
        !h3_dit_int8_read_tensor(cache->int8_store, lin.scale, host->h_s,
                                 s_bytes, detail, sizeof(detail))) {
        snprintf(error, error_size, "INT8 host stage read: %s", detail);
        free_int8_host_linear(host);
        return 0;
    }
    host->n = lin.n;
    host->k = lin.k;
    host->ready = 1;
    cache->int8_disk_loads++;
    return 1;
}

static size_t stage_align(size_t n) {
    return (n + 255u) & ~(size_t)255u;
}

static int stage_pread(int fd, const unsigned char *map, size_t map_len,
                       const h3_st_tensor *tensor, void *dst, size_t bytes,
                       char *error, size_t error_size) {
    uint64_t off;
    if (!tensor || !dst || !bytes) {
        fail(error, error_size, "invalid stage pread");
        return 0;
    }
    off = tensor->file_offset;
    if (map) {
        if (off + bytes > map_len) {
            fail(error, error_size, "mmap stage overflow");
            return 0;
        }
        memcpy(dst, map + off, bytes);
        return 1;
    }
    {
        size_t done = 0;
        while (done < bytes) {
            size_t chunk = bytes - done;
            ssize_t n;
            if (chunk > (size_t)(1u << 30))
                chunk = (size_t)(1u << 30);
            n = pread(fd, (unsigned char *)dst + done, chunk,
                      (off_t)(off + done));
            if (n <= 0) {
                snprintf(error, error_size, "stage pread: %s",
                         n < 0 ? strerror(errno) : "EOF");
                return 0;
            }
            done += (size_t)n;
        }
    }
    return 1;
}

typedef struct {
    h3_w4a8_cache *cache;
    unsigned first;
    unsigned end;
    unsigned *next;
    int fd;
    const unsigned char *map;
    size_t map_len;
    char err[256];
    int ok;
} stage_read_job;

static void *stage_read_th(void *arg) {
    stage_read_job *j = (stage_read_job *)arg;
    unsigned b;
    if (!j || !j->cache || !j->next) return NULL;
    j->ok = 1;
    j->err[0] = '\0';
    for (;;) {
        b = (unsigned)__sync_fetch_and_add((int *)j->next, 1);
        if (b >= j->end) break;
        {
            int k;
            for (k = 0; k < H3_W4A8_LINEAR_COUNT; k++) {
                h3_dit_int8_linear lin;
                h3_w4a8_int8_host_linear *host;
                char detail[256];
                if (!h3_dit_int8_find_block_linear(
                        j->cache->int8_store, b,
                        h3_w4a8_linear_kind_suffix((h3_w4a8_linear_kind)k),
                        &lin, detail, sizeof(detail))) {
                    snprintf(j->err, sizeof(j->err), "%s", detail);
                    j->ok = 0;
                    return NULL;
                }
                host = &j->cache->slots[b].host[k];
                if (host->ready) continue;
                if (!host->h_w || !host->h_s) {
                    snprintf(j->err, sizeof(j->err),
                             "stage missing arena slot b%u k%d", b, k);
                    j->ok = 0;
                    return NULL;
                }
                if (!stage_pread(j->fd, j->map, j->map_len, lin.weight,
                                 host->h_w, lin.n * lin.k, j->err,
                                 sizeof(j->err)) ||
                    !stage_pread(j->fd, j->map, j->map_len, lin.scale,
                                 host->h_s, lin.n * sizeof(float), j->err,
                                 sizeof(j->err))) {
                    j->ok = 0;
                    return NULL;
                }
                if (j->cache->host_mu_ready)
                    pthread_mutex_lock(&j->cache->host_mu);
                host->n = lin.n;
                host->k = lin.k;
                host->ready = 1;
                j->cache->int8_disk_loads++;
                if (j->cache->host_mu_ready)
                    pthread_mutex_unlock(&j->cache->host_mu);
            }
        }
        if (j->cache->host_mu_ready)
            pthread_mutex_lock(&j->cache->host_mu);
        j->cache->host_ready_blocks++;
        if (j->cache->host_mu_ready)
            pthread_mutex_unlock(&j->cache->host_mu);
    }
    return NULL;
}

int h3_w4a8_cache_stage_host_int8(h3_w4a8_cache *cache, unsigned first_block,
                                  unsigned n_blocks, double *elapsed_ms,
                                  char *error, size_t error_size) {
    unsigned end, b;
    int k;
    double t0, t_alloc0, t_read0;
    int verbose;
    if (elapsed_ms) *elapsed_ms = 0.0;
    if (!cache || (!cache->int8_store && !cache->nvfp4_store)) {
        fail(error, error_size, "INT8/NVFP4 cache required for host stage");
        return 0;
    }
    if (first_block >= cache->n_blocks) {
        fail(error, error_size, "first_block out of range");
        return 0;
    }
    end = first_block + n_blocks;
    if (end > cache->n_blocks) end = cache->n_blocks;
    /* nvfp4 track: no INT8 host staging (FP4 sidecar is the weight source). */
    if (cache->nvfp4_int8_skip) {
        cache->host_ready_blocks = 0;
        return 1;
    }
    verbose = 0;
    {
        const char *e = getenv("H3_DIT_HOST_STAGE_VERBOSE");
        verbose = e && e[0] == '1' && e[1] == '\0';
    }
    t0 = cache_now_ms();
    cache->host_ready_blocks = 0;
    cache->host_stage_alloc_ms = 0.0;
    cache->host_stage_read_ms = 0.0;
    cache->host_stage_bytes = 0;
    cache->host_stage_n_linears = 0;
    cache->host_stage_n_skip = 0;
    cache->host_stage_mmap = 0;

    /* NVFP4 host stage stays serial (dequant or sidecar); not gold INT8. */
    if (cache->nvfp4_store) {
        for (b = first_block; b < end; b++) {
            for (k = 0; k < H3_W4A8_LINEAR_COUNT; k++) {
                if (!stage_one_host_nvfp4(cache, b, (h3_w4a8_linear_kind)k,
                                          error, error_size))
                    return 0;
            }
            cache->host_ready_blocks++;
        }
        if (elapsed_ms) *elapsed_ms = cache_now_ms() - t0;
        fprintf(stderr,
                "nvfp4_sidecar: hits=%u misses=%u writes=%u wall_ms=%.1f "
                "(dirty key=recompute; never gold INT8 fallback)\n",
                cache->nvfp4_sidecar_hits, cache->nvfp4_sidecar_misses,
                cache->nvfp4_sidecar_writes, elapsed_ms ? *elapsed_ms : 0.0);
        fflush(stderr);
        return 1;
    }

    {
        const h3_st_header *hdr = h3_dit_int8_store_header(cache->int8_store);
        int fd = -1;
        const unsigned char *map = NULL;
        size_t map_len = 0;
        size_t need = 0;
        unsigned n_skip = 0, n_lin = 0;
        int nthreads = 4;
        const char *te = getenv("H3_DIT_HOST_STAGE_THREADS");

        if (te && te[0]) {
            long v = strtol(te, NULL, 10);
            if (v >= 1 && v <= 16) nthreads = (int)v;
        }
        cache->host_stage_threads = nthreads;

        if (!hdr || !hdr->path) {
            fail(error, error_size, "INT8 store has no path");
            return 0;
        }
        fd = open(hdr->path, O_RDONLY);
        if (fd < 0) {
            snprintf(error, error_size, "stage open: %s", strerror(errno));
            return 0;
        }
#ifdef POSIX_FADV_WILLNEED
        (void)posix_fadvise(fd, 0, 0, POSIX_FADV_WILLNEED);
#endif
        if (hdr->file_size > 0) {
            size_t page = (size_t)sysconf(_SC_PAGESIZE);
            size_t want = (size_t)hdr->file_size;
            if (page > 0)
                want = (want + page - 1) & ~(page - 1);
            void *p = mmap(NULL, want, PROT_READ, MAP_SHARED, fd, 0);
            if (p != MAP_FAILED) {
                map = (const unsigned char *)p;
                map_len = want;
                cache->host_stage_mmap = 1;
#ifdef MADV_WILLNEED
                (void)madvise((void *)map, map_len, MADV_WILLNEED);
#endif
            }
        }

        /* Fast path: pin the file map and alias tensors (no 18GiB extra copy). */
        if (map && !cache->host_mmap_registered) {
            char reg_err[128];
            t_alloc0 = cache_now_ms();
            if (h3_cuda_host_register((void *)map, map_len, reg_err,
                                      sizeof(reg_err))) {
                cache->host_mmap = (void *)map;
                cache->host_mmap_len = map_len;
                cache->host_mmap_registered = 1;
                cache->host_stage_alloc_ms = cache_now_ms() - t_alloc0;
                t_read0 = cache_now_ms();
                for (b = first_block; b < end; b++) {
                    for (k = 0; k < H3_W4A8_LINEAR_COUNT; k++) {
                        h3_dit_int8_linear lin;
                        h3_w4a8_int8_host_linear *host;
                        char detail[256];
                        if (!h3_dit_int8_find_block_linear(
                                cache->int8_store, b,
                                h3_w4a8_linear_kind_suffix(
                                    (h3_w4a8_linear_kind)k),
                                &lin, detail, sizeof(detail))) {
                            snprintf(error, error_size, "%s", detail);
                            close(fd);
                            return 0;
                        }
                        host = &cache->slots[b].host[k];
                        if (host->ready && host->h_w) {
                            n_skip++;
                            continue;
                        }
                        if (lin.weight->file_offset + lin.n * lin.k > map_len ||
                            lin.scale->file_offset + lin.n * sizeof(float) >
                                map_len) {
                            fail(error, error_size, "mmap alias overflow");
                            close(fd);
                            return 0;
                        }
                        host->h_w = (int8_t *)(map + lin.weight->file_offset);
                        host->h_s = (float *)(map + lin.scale->file_offset);
                        host->n = lin.n;
                        host->k = lin.k;
                        host->from_arena = 1;
                        host->pinned_w = 1;
                        host->pinned_s = 1;
                        host->ready = 1;
                        n_lin++;
                        cache->int8_disk_loads++;
                    }
                    cache->host_ready_blocks++;
                }
                cache->host_stage_read_ms = cache_now_ms() - t_read0;
                cache->host_stage_n_linears = n_lin;
                cache->host_stage_n_skip = n_skip;
                cache->host_stage_bytes = map_len;
                cache->host_arena_pinned = 1;
                close(fd);
                if (elapsed_ms) *elapsed_ms = cache_now_ms() - t0;
                printf("dit_host_stage_break: wall_ms=%.1f pin_ms=%.1f "
                       "read_ms=%.1f bytes=%.2fGiB linears=%u skip=%u "
                       "threads=0 mmap=1 ready_blocks=%u pinned=1 alias=1%s\n",
                       elapsed_ms ? *elapsed_ms : cache_now_ms() - t0,
                       cache->host_stage_alloc_ms, cache->host_stage_read_ms,
                       (double)cache->host_stage_bytes /
                           (1024.0 * 1024.0 * 1024.0),
                       cache->host_stage_n_linears, cache->host_stage_n_skip,
                       cache->host_ready_blocks, verbose ? " verbose=1" : "");
                fflush(stdout);
                return 1;
            }
            printf("dit_host_stage: cudaHostRegister fallback (%s) "
                   "map_len=%.2fGiB → arena copy\n",
                   reg_err[0] ? reg_err : "unknown",
                   (double)map_len / (1024.0 * 1024.0 * 1024.0));
            fflush(stdout);
        }

        t_alloc0 = cache_now_ms();
        for (b = first_block; b < end; b++) {
            for (k = 0; k < H3_W4A8_LINEAR_COUNT; k++) {
                h3_dit_int8_linear lin;
                h3_w4a8_int8_host_linear *host;
                char detail[256];
                if (!h3_dit_int8_find_block_linear(
                        cache->int8_store, b,
                        h3_w4a8_linear_kind_suffix((h3_w4a8_linear_kind)k),
                        &lin, detail, sizeof(detail))) {
                    snprintf(error, error_size, "%s", detail);
                    if (map) munmap((void *)map, map_len);
                    close(fd);
                    return 0;
                }
                host = &cache->slots[b].host[k];
                if (host->ready && host->h_w && host->n == lin.n &&
                    host->k == lin.k) {
                    n_skip++;
                    continue;
                }
                need += stage_align(lin.n * lin.k);
                need += stage_align(lin.n * sizeof(float));
                n_lin++;
            }
        }

        if (need > 0 && !cache->host_arena) {
            int pinned = 0;
            cache->host_arena = h3_cuda_host_alloc(need, &pinned);
            cache->host_arena_bytes = need;
            cache->host_arena_pinned = pinned;
            if (!cache->host_arena) {
                fail(error, error_size, "OOM INT8 host stage arena");
                if (map) munmap((void *)map, map_len);
                close(fd);
                return 0;
            }
        }
        {
            unsigned char *cur = (unsigned char *)cache->host_arena;
            size_t used = 0;
            for (b = first_block; b < end; b++) {
                for (k = 0; k < H3_W4A8_LINEAR_COUNT; k++) {
                    h3_dit_int8_linear lin;
                    h3_w4a8_int8_host_linear *host;
                    size_t w_bytes, s_bytes;
                    char detail[256];
                    if (!h3_dit_int8_find_block_linear(
                            cache->int8_store, b,
                            h3_w4a8_linear_kind_suffix(
                                (h3_w4a8_linear_kind)k),
                            &lin, detail, sizeof(detail))) {
                        snprintf(error, error_size, "%s", detail);
                        if (map) munmap((void *)map, map_len);
                        close(fd);
                        return 0;
                    }
                    host = &cache->slots[b].host[k];
                    if (host->ready && host->h_w) continue;
                    w_bytes = stage_align(lin.n * lin.k);
                    s_bytes = stage_align(lin.n * sizeof(float));
                    if (cur && used + w_bytes + s_bytes <=
                                   cache->host_arena_bytes) {
                        host->h_w = (int8_t *)(cur + used);
                        used += w_bytes;
                        host->h_s = (float *)(cur + used);
                        used += s_bytes;
                        host->pinned_w = cache->host_arena_pinned;
                        host->pinned_s = cache->host_arena_pinned;
                        host->from_arena = 1;
                        host->ready = 0;
                    } else if (!host->h_w) {
                        host->h_w = (int8_t *)h3_cuda_host_alloc(
                            lin.n * lin.k, &host->pinned_w);
                        host->h_s = (float *)h3_cuda_host_alloc(
                            lin.n * sizeof(float), &host->pinned_s);
                        if (!host->h_w || !host->h_s) {
                            fail(error, error_size, "OOM INT8 host stage");
                            if (map) munmap((void *)map, map_len);
                            close(fd);
                            return 0;
                        }
                        host->from_arena = 0;
                        host->ready = 0;
                    }
                }
            }
        }
        cache->host_stage_alloc_ms = cache_now_ms() - t_alloc0;
        cache->host_stage_n_linears = n_lin;
        cache->host_stage_n_skip = n_skip;
        cache->host_stage_bytes = need;

        t_read0 = cache_now_ms();
        {
            unsigned next = first_block;
            int nth = nthreads;
            unsigned nblk = end - first_block;
            stage_read_job jobs[16];
            pthread_t th[16];
            int i, spawned = 0;
            if (nth > (int)nblk) nth = (int)nblk;
            if (nth < 1) nth = 1;
            memset(jobs, 0, sizeof(jobs));
            for (i = 0; i < nth; i++) {
                jobs[i].cache = cache;
                jobs[i].first = first_block;
                jobs[i].end = end;
                jobs[i].next = &next;
                jobs[i].fd = fd;
                jobs[i].map = map;
                jobs[i].map_len = map_len;
                jobs[i].ok = 1;
                if (i + 1 < nth) {
                    if (pthread_create(&th[i], NULL, stage_read_th,
                                       &jobs[i]) == 0)
                        spawned++;
                    else {
                        stage_read_th(&jobs[i]);
                    }
                } else {
                    stage_read_th(&jobs[i]);
                }
            }
            for (i = 0; i < spawned; i++)
                pthread_join(th[i], NULL);
            for (i = 0; i < nth; i++) {
                if (!jobs[i].ok) {
                    snprintf(error, error_size, "%s",
                             jobs[i].err[0] ? jobs[i].err : "stage read");
                    if (map) munmap((void *)map, map_len);
                    close(fd);
                    return 0;
                }
            }
        }
        cache->host_stage_read_ms = cache_now_ms() - t_read0;
        if (map) munmap((void *)map, map_len);
        close(fd);
    }

    if (elapsed_ms) *elapsed_ms = cache_now_ms() - t0;
    printf("dit_host_stage_break: wall_ms=%.1f pin_ms=%.1f read_ms=%.1f "
           "bytes=%.2fGiB linears=%u skip=%u threads=%d mmap=%d "
           "ready_blocks=%u pinned=%d%s\n",
           elapsed_ms ? *elapsed_ms : cache_now_ms() - t0,
           cache->host_stage_alloc_ms, cache->host_stage_read_ms,
           (double)cache->host_stage_bytes / (1024.0 * 1024.0 * 1024.0),
           cache->host_stage_n_linears, cache->host_stage_n_skip,
           cache->host_stage_threads, cache->host_stage_mmap,
           cache->host_ready_blocks, cache->host_arena_pinned,
           verbose ? " verbose=1" : "");
    fflush(stdout);
    return 1;
}

unsigned h3_w4a8_cache_host_ready_blocks(const h3_w4a8_cache *cache) {
    return cache ? cache->host_ready_blocks : 0;
}

void h3_w4a8_cache_set_async_h2d(h3_w4a8_cache *cache, int on) {
    if (cache) cache->async_h2d = on ? 1 : 0;
}

const h3_cuda_w4a8_weight *h3_w4a8_cache_get_linear(h3_w4a8_cache *cache,
                                                    unsigned block,
                                                    h3_w4a8_linear_kind kind,
                                                    char *error,
                                                    size_t error_size) {
    if (!cache || block >= cache->n_blocks ||
        (unsigned)kind >= (unsigned)H3_W4A8_LINEAR_COUNT) {
        fail(error, error_size, "invalid get_linear args");
        return NULL;
    }
    if (!ensure_capacity_for(cache, block, error, error_size)) return NULL;
    if (!cache->nvfp4_int8_skip) {
        if (!load_block(cache, block, error, error_size)) return NULL;
    } else if (!load_one_into_slot(cache, block, kind, error, error_size)) {
        return NULL;
    }
    cache->slots[block].last_used = ++cache->tick;
    return cache->slots[block].linears[kind];
}

int h3_w4a8_cache_block_is_resident(h3_w4a8_cache *cache, unsigned block) {
    if (!cache || !cache->slots || block >= cache->n_blocks) return 0;
    if (!cache->slots[block].resident) return 0;
    for (int i = 0; i < H3_W4A8_LINEAR_COUNT; i++) {
        if (!cache->slots[block].linears[i]) return 0;
    }
    return 1;
}

void h3_w4a8_cache_set_nvfp4_int8_skip(h3_w4a8_cache *cache, int on) {
    if (!cache) return;
    cache->nvfp4_int8_skip = on ? 1 : 0;
    if (cache->nvfp4_int8_skip)
        printf("dit_int8_cache: nvfp4_int8_skip=1 (FP4 linears; INT8 "
               "fallback loads lazily on FP4 miss only)\n");
}

int h3_w4a8_cache_get_stats(h3_w4a8_cache *cache, h3_w4a8_cache_stats *stats,
                            char *error, size_t error_size) {
    size_t free_b = 0, total_b = 0;
    unsigned n_linears = 0;
    char detail[128];
    if (!cache || !stats) {
        fail(error, error_size, "invalid get_stats args");
        return 0;
    }
    memset(stats, 0, sizeof(*stats));
    if (!h3_cuda_mem_info(cache->cuda, &free_b, &total_b, detail,
                          sizeof(detail))) {
        snprintf(error, error_size, "mem_info: %s", detail);
        return 0;
    }
    for (unsigned b = 0; b < cache->n_blocks; b++) {
        if (!cache->slots[b].resident) continue;
        for (int i = 0; i < H3_W4A8_LINEAR_COUNT; i++) {
            if (cache->slots[b].linears[i]) n_linears++;
        }
    }
    stats->n_blocks = cache->n_blocks;
    stats->n_resident_blocks = cache->n_resident;
    stats->n_resident_linears = n_linears;
    stats->bytes_resident = cache->bytes_resident;
    stats->bytes_estimated_all = cache->bytes_estimated_all;
    stats->bytes_per_block_est = cache->bytes_per_block_est;
    stats->max_blocks_fit = cache->max_blocks_fit;
    stats->requested_policy = cache->requested_policy;
    stats->effective_policy = cache->effective_policy;
    stats->window_size = cache->window_size;
    stats->auto_downgraded = cache->auto_downgraded;
    stats->vram_free_bytes = free_b;
    stats->vram_total_bytes = total_b;
    stats->vram_budget_bytes = cache->vram_budget_bytes;
    return 1;
}
