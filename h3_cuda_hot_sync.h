#ifndef H3_CUDA_HOT_SYNC_H
#define H3_CUDA_HOT_SYNC_H

/*
 * C070: H3_FORBID_HOT_SYNC — opt-in assert against hot-path device sync.
 *
 * Classes (see docs/research/2026-10-27-cuda-sync-audit.md):
 *   H3_SYNC_LIFECYCLE  — stage boundary / one-shot setup (allowed)
 *   H3_SYNC_KPI        — profile/timer fence only (allowed; not "speedup")
 *   H3_SYNC_DEBUG      — diag / Ulysses phase log (allowed)
 *   H3_SYNC_HOT        — per-step / per-block / fallback that can sit in
 *                        the denoise hot path (FORBIDDEN when env=1)
 *
 * Default: env unset → no-op (same as today). Env=1 → HOT class aborts.
 */

#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    H3_SYNC_LIFECYCLE = 0,
    H3_SYNC_KPI = 1,
    H3_SYNC_DEBUG = 2,
    H3_SYNC_HOT = 3
};

static inline int h3_forbid_hot_sync_on(void) {
    const char *e = getenv("H3_FORBID_HOT_SYNC");
    return e && e[0] == '1' && e[1] == '\0';
}

/* Returns cudaSuccess, or never returns on forbidden HOT sync. */
static inline cudaError_t h3_cuda_device_sync_class(int sync_class,
                                                    const char *site) {
    if (sync_class == H3_SYNC_HOT && h3_forbid_hot_sync_on()) {
        fprintf(stderr,
                "h3: H3_FORBID_HOT_SYNC=1 blocked cudaDeviceSynchronize at %s "
                "(class=HOT). Use stream events or move sync to lifecycle/KPI.\n",
                site ? site : "?");
        fflush(stderr);
        abort();
    }
    return cudaDeviceSynchronize();
}

#ifdef __cplusplus
}
#endif

#endif /* H3_CUDA_HOT_SYNC_H */
