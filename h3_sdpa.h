#ifndef H3_SDPA_H
#define H3_SDPA_H

#include "h3_cuda_device_caps.h"

#include <stddef.h>

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

#ifdef __cplusplus
}
#endif

#endif /* H3_SDPA_H */
