#ifndef H3_VAE_PREWARM_H
#define H3_VAE_PREWARM_H

#include "h3_vae_int8.h"

#include <stddef.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* R11: DiT last-NFE ∥ VAE mmap + hostreg + optional device W refill.
 * H3_GEN_VAE_PREWARM=0 disables. Default on. */
static inline int h3_vae_prewarm_wanted(void) {
    const char *e = getenv("H3_GEN_VAE_PREWARM");
    if (e && e[0] == '0' && e[1] == '\0') return 0;
    return 1;
}

typedef struct {
    const char *path;
    h3_vae_int8_store *store;
    int *opened;
    int device; /* cudaSetDevice; <0 skips hostreg */
    int want_hostreg;
    int want_dev; /* 1 → weights_load device W during last NFE */
    unsigned n_blocks;
    int at_step; /* 1-based dit step for logs */
    int ok;
    int hostreg_ok;
    int device_ok;
    double ms;
    double load_ms;
    char err[512];
} h3_vae_prewarm_job;

void *h3_vae_prewarm_th(void *arg);

#ifdef __cplusplus
}
#endif

#endif
