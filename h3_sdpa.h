#ifndef H3_SDPA_H
#define H3_SDPA_H

#include "h3_cuda_device_caps.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Unified SDPA backend registry (CLI / env / docs share these names).
 *
 * Tier-0 quality default:  cudnn
 * Tier-1 product accel:    sage3, sol_cute   (must be first-class; not "diagnostic only")
 * Forbidden as product:    sage2 (in-repo ref), sol (nested-loop fake)
 * Legacy / debug:          flash, tc
 *
 * auto → always cudnn (never auto-picks a quality-lossy backend).
 */

typedef enum {
    H3_SDPA_ID_CUDNN = 0,
    H3_SDPA_ID_SAGE3,
    H3_SDPA_ID_SOL_CUTE,
    H3_SDPA_ID_FLASH,
    H3_SDPA_ID_TC,
    H3_SDPA_ID_SAGE2, /* in-repo reference — NOT product */
    H3_SDPA_ID_SOL,   /* nested-loop — NOT product / NOT sol_cute */
    H3_SDPA_ID_UNKNOWN
} h3_sdpa_backend_id;

typedef struct {
    h3_sdpa_backend_id id;
    const char *name;       /* stable CLI token */
    int quality_safe;       /* Tier-0 */
    int product_accel;      /* Tier-1 first-class accel */
    int forbidden_product;  /* must not be marketed as real Sage3/Sol */
} h3_sdpa_backend_desc;

const h3_sdpa_backend_desc *h3_sdpa_backend_table(size_t *count);

/* Normalize aliases → stable name (sageattn3→sage3, sol-cute→sol_cute, …). */
const char *h3_sdpa_normalize_name(const char *request);

/* Resolve request (+ optional caps) to a backend id.
 * Missing Tier-1 plugin: still returns the requested id but sets *available=0
 * and fills reason; callers must error clearly (no silent fake kernel).
 * auto / empty → cudnn, available=1 (if has_cudnn_sdpa or caps NULL). */
int h3_sdpa_resolve(const char *request, const h3_cuda_device_caps *caps,
                    h3_sdpa_backend_id *out_id, int *available, char *reason,
                    size_t reason_size);

const char *h3_sdpa_backend_id_name(h3_sdpa_backend_id id);

/* Cached resolve from H3_CUDA_SDPA_BACKEND (+ last query_caps if set). */
h3_sdpa_backend_id h3_sdpa_cached_id(void);
void h3_sdpa_cache_invalidate(void);

/* Log one line: selected backend + reason (+ caps summary if provided). */
void h3_sdpa_log_selection(const h3_cuda_device_caps *caps);

/* Accel sage3: want>0 && hits==0 is a hard fail unless
 * H3_SAGE3_REQUIRE_HIT=0 (legacy: clear error and keep dense). Default hard
 * fail. Gold/cudnn never increments want, so this stays 0. */
static inline int h3_sdpa_sage3_require_hit(unsigned long long want,
                                            unsigned long long hits) {
    const char *e = getenv("H3_SAGE3_REQUIRE_HIT");
    if (e && e[0] == '0' && e[1] == '\0') return 0;
    return want > 0ull && hits == 0ull;
}

/* VAE batched SDPA + DiT accel share this env. Default on; =0 allows
 * named fallback (cudnn_loop). Does not change sage3_require_hit. */
static inline int h3_sdpa_strict(void) {
    const char *e = getenv("H3_SDPA_STRICT");
    if (e && e[0] == '0' && e[1] == '\0') return 0;
    return 1;
}

/* Official MMDiT dense guard: first-N denoise steps / first-N layers run
 * dense by design (arXiv:2607.24027 recipe). Read the same env vars the
 * router in h3_gpu_cuda.cu uses so the KPI verdict can subtract dense_skip
 * from the strict expectation instead of FAILing a guard arm. */
static inline int h3_sol_first_dense_steps(void) {
    const char *e = getenv("H3_SOL_FIRST_DENSE_STEPS");
    if (!e || !e[0]) e = getenv("SOL_ATTN_FIRST_DENSE_STEPS");
    if (!e || !e[0]) return 2;
    return atoi(e);
}
static inline int h3_sol_first_dense_layers(void) {
    const char *e = getenv("H3_SOL_FIRST_DENSE_LAYERS");
    if (!e || !e[0]) e = getenv("SOL_ATTN_FIRST_DENSE_LAYERS");
    if (!e || !e[0]) return 2;
    return atoi(e);
}
static inline int h3_sol_dense_guard_on(void) {
    return h3_sol_first_dense_steps() > 0 || h3_sol_first_dense_layers() > 0;
}

static inline int h3_sdpa_vae_batched_fail_hard(void) {
    return h3_sdpa_strict();
}

/* Official NVlabs sol-engine CuTe backends (interface.py _CUTE_BACKENDS).
 * "triton" is the missing-CuTe fallback (triton_ref) — not a product kernel. */
static inline int h3_sdpa_sol_cute_is_official(const char *backend) {
    if (!backend || !backend[0]) return 0;
    return !strcmp(backend, "cute_sm89") || !strcmp(backend, "cute_sm90") ||
           !strcmp(backend, "cute_sm100") || !strcmp(backend, "cute_sm120");
}

/* Persistent sol_cute QKVO: 4 BF16 tensors [1,T,H,D]. */
static inline size_t h3_sol_cute_qkvo_bytes(size_t seq, size_t heads,
                                            size_t dim) {
    return 4ull * seq * heads * dim * 2ull;
}

/* Native F16 QKV copy into sol (not 4 * padded cuDNN scratch). */
static inline size_t h3_sol_cute_f16_qkv_scratch_bytes(size_t seq, size_t heads,
                                                       size_t dim) {
    return 3ull * seq * heads * dim * 2ull;
}

/* H3_SOL_MAX_SEQ=0 / unset → no cap (try hires sol). Mixed arm: e.g. 16000
 * keeps low-res sol and sends 768P seq=39172 to cuDNN. */
static inline size_t h3_sol_cute_max_seq(void) {
    const char *e = getenv("H3_SOL_MAX_SEQ");
    unsigned long v;
    if (!e || !e[0]) return 0;
    v = strtoul(e, NULL, 10);
    return (size_t)v;
}

static inline int h3_sol_cute_seq_allowed(size_t seq) {
    size_t cap = h3_sol_cute_max_seq();
    if (cap == 0) return 1;
    return seq <= cap;
}

#ifdef __cplusplus
}
#endif

#endif /* H3_SDPA_H */
