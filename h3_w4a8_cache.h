#ifndef H3_W4A8_CACHE_H
#define H3_W4A8_CACHE_H

#include "h3_dit_int8.h"
#include "h3_gpu_cuda.h"
#include "h3_w4a8.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* P4 step-6: 50-block W4A8 INT8 weight residency / swap manager.
 * Also serves Comfy INT8 ConvRot DiT (I8 + weight_scale) via open_int8.
 * C API reserved for future h3_dit CUDA wiring — does not rewrite Metal DiT. */

#define H3_W4A8_CACHE_DEFAULT_BLOCKS 50
/* Leave headroom for DiT activation workspace (full 480P seq can be multi-GB)
 * so max_blocks_fit does not over-promise and OOM when workspace is allocated. */
#define H3_W4A8_CACHE_DEFAULT_RESERVE_BYTES ((size_t)8ULL << 30)

typedef enum {
    H3_W4A8_CACHE_RESIDENT_ALL = 0, /* pack all blocks if VRAM allows */
    H3_W4A8_CACHE_WINDOW = 1,       /* keep a contiguous window of N blocks */
    H3_W4A8_CACHE_LRU = 2           /* keep the N most recently used blocks */
} h3_w4a8_cache_policy;

typedef enum {
    H3_W4A8_LINEAR_QKV = 0,
    H3_W4A8_LINEAR_OUT = 1,
    H3_W4A8_LINEAR_FC1 = 2,
    H3_W4A8_LINEAR_FC2 = 3,
    H3_W4A8_LINEAR_COUNT = 4
} h3_w4a8_linear_kind;

typedef struct {
    unsigned n_blocks; /* default 50 */
    h3_w4a8_cache_policy policy;
    unsigned window_size; /* WINDOW/LRU max resident blocks; 0 → auto max_fit */
    size_t vram_budget_bytes;  /* 0 → free_at_open - reserve */
    size_t vram_reserve_bytes; /* headroom; 0 → DEFAULT_RESERVE */
} h3_w4a8_cache_config;

typedef struct {
    unsigned n_blocks;
    unsigned n_resident_blocks;
    unsigned n_resident_linears;
    size_t bytes_resident;       /* INT8 + s_channel currently on device */
    size_t bytes_estimated_all;  /* estimate for all n_blocks */
    size_t bytes_per_block_est;  /* average estimate from header shapes */
    unsigned max_blocks_fit;     /* how many blocks fit in budget */
    h3_w4a8_cache_policy requested_policy;
    h3_w4a8_cache_policy effective_policy;
    unsigned window_size;
    int auto_downgraded; /* RESIDENT_ALL → WINDOW when VRAM tight */
    size_t vram_free_bytes;
    size_t vram_total_bytes;
    size_t vram_budget_bytes;
} h3_w4a8_cache_stats;

typedef struct h3_w4a8_cache h3_w4a8_cache;

void h3_w4a8_cache_config_init(h3_w4a8_cache_config *cfg);

/* Open manager (does not prefetch). Store is borrowed; cuda is borrowed. */
h3_w4a8_cache *h3_w4a8_cache_open(h3_cuda *cuda, h3_w4a8_store *store,
                                  const h3_w4a8_cache_config *cfg, char *error,
                                  size_t error_size);
/* INT8 ConvRot DiT (I8 weight + F32 weight_scale); same residency policy. */
h3_w4a8_cache *h3_w4a8_cache_open_int8(h3_cuda *cuda, h3_dit_int8_store *store,
                                       const h3_w4a8_cache_config *cfg,
                                       char *error, size_t error_size);
void h3_w4a8_cache_free(h3_w4a8_cache *cache);

/* Unpack + pin [first_block, first_block+n_blocks) under the active policy.
 * For RESIDENT_ALL, pass first=0, n=n_blocks. *elapsed_ms is wall time. */
int h3_w4a8_cache_prefetch(h3_w4a8_cache *cache, unsigned first_block,
                           unsigned n_blocks, double *elapsed_ms, char *error,
                           size_t error_size);

/* Disk → pinned host only (no GPU). Safe during TE: does not steal VRAM.
 * Later prefetch/get_linear become host_hits + H2D. */
int h3_w4a8_cache_stage_host_int8(h3_w4a8_cache *cache, unsigned first_block,
                                  unsigned n_blocks, double *elapsed_ms,
                                  char *error, size_t error_size);

/* Ensure block is resident (evict per policy) and return device weight.
 * Pointer owned by cache until free/evict. */
const h3_cuda_w4a8_weight *h3_w4a8_cache_get_linear(h3_w4a8_cache *cache,
                                                    unsigned block,
                                                    h3_w4a8_linear_kind kind,
                                                    char *error,
                                                    size_t error_size);

int h3_w4a8_cache_get_stats(h3_w4a8_cache *cache, h3_w4a8_cache_stats *stats,
                            char *error, size_t error_size);

/* Effective sliding-window width (after auto-fit / clamp). */
unsigned h3_w4a8_cache_window_size(const h3_w4a8_cache *cache);
unsigned h3_w4a8_cache_max_blocks_fit(const h3_w4a8_cache *cache);

/* After activation workspace is allocated (WS-first), remeasure free VRAM.
 * Holds the open-time window (never grow — that OOMs gemm acc; never shrink
 * just because TE skipped prefetch). No-op for W4A8 / already-refit. */
int h3_w4a8_cache_refit_after_workspace(h3_w4a8_cache *cache, char *error,
                                        size_t error_size);

/* Evict LRU INT8 blocks until cudaMemGetInfo free >= keep_free_bytes.
 * Used before a large activation workspace (true 5s/10s seq) so overlap
 * prefetch of a 41-block window cannot starve DiT rope/act/cuDNN.
 * No-op after freeze_window (cafe p0: make_room after step 1 dropped 29→16).
 * H3_DIT_INT8_ALLOW_SHRINK=1 re-enables eviction. */
int h3_w4a8_cache_make_room(h3_w4a8_cache *cache, size_t keep_free_bytes,
                            char *error, size_t error_size);

/* After the first full DiT step succeeds, stop shrinking the INT8 window.
 * LightX2V extreme path keeps cpu_offload=false for throughput. */
void h3_w4a8_cache_freeze_window(h3_w4a8_cache *cache);

/* One-shot INT8 host-staging counters (disk vs H2D-only reload). */
void h3_w4a8_cache_log_int8_host(const h3_w4a8_cache *cache);

const char *h3_w4a8_cache_policy_name(h3_w4a8_cache_policy policy);
const char *h3_w4a8_linear_kind_name(h3_w4a8_linear_kind kind);
const char *h3_w4a8_linear_kind_suffix(h3_w4a8_linear_kind kind);

#ifdef __cplusplus
}
#endif

#endif
