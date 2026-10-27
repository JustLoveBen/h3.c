#ifndef H3_LIFECYCLE_WHITELIST_H
#define H3_LIFECYCLE_WHITELIST_H

/*
 * C064 — TE / DiT / VAE DeferredRelease whitelist (host-safe, offline).
 *
 * Authority: docs/research/2026-10-26-te-lifecycle-audit.md
 * Productization note: docs/research/2026-10-27-lifecycle-deferred-release.md
 *
 * Purpose: name objects that may outlive their produce stage, and those that
 * must free immediately. This header does NOT enable new overlap; hot-path
 * gates in tools/h3_cuda_generate.c / h3_vae_prewarm.* stay authoritative.
 *
 * safe_overlap values:
 *   0 = must free / join before next stage (no defer)
 *   1 = may overlap next stage only with documented event / wait
 *  -1 = gated / experimental (default off or arm-specific forbid)
 *
 * Debug: H3_LIFECYCLE_TRACE=1 → h3_lifecycle_trace() prints transitions.
 */

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct h3_lifecycle_whitelist_row {
    const char *object;       /* e.g. TE text_emb scratch */
    const char *produce;      /* produce stage */
    const char *release;      /* required release stage */
    const char *event_dep;    /* event / join / wait; "-" if none */
    int safe_overlap;         /* 0 / 1 / -1 — see above */
    const char *notes;
} h3_lifecycle_whitelist_row;

/* Keep in sync with te-lifecycle-audit §1 / §DeepInfer DeferredRelease table. */
static const h3_lifecycle_whitelist_row k_h3_lifecycle_whitelist[] = {
    {"TE text_emb scratch", "te_encode", "after refine (before DiT)", "-", 0,
     "free(text_emb) immediately; do not defer to DiT step0"},
    {"TE nvfp4 store (weights)", "te_open", "after refine store_free", "-", 0,
     "h3_nvfp4_store_free + cuda_disable; frees VRAM for DiT window"},
    {"TE prompt disk cache IO", "te_encode (opt-in)", "N/A (disk)", "-", 0,
     "H3_TE_PROMPT_CACHE; vision spans never cached"},
    {"DiT host_stage pin buf (INT8)", "TE∥host_stage kick",
     "wait_first then join after DiT", "host_ready_blocks / pthread", 1,
     "Gold INT8 may overlap step1 after wait_blocks (default 2)"},
    {"DiT host_stage pin buf (NVFP4)", "TE∥host_stage kick",
     "join BEFORE DiT", "pthread_join", 0,
     "Non-gold: dequant realloc UAF if overlapped; must join"},
    {"DiT TE GPU prefetch", "TE encode (opt)", "stream wait before DiT",
     "cuda event prefetch_done", -1,
     "H3_DIT_TE_OVERLAP_BLOCKS=N>0 + HOST_STAGE_TE=0 opts in; "
     "latent_t>=32 default skip (C064)"},
    {"DiT INT8 resident_all W", "DiT step0 upload", "after DiT / cache policy",
     "-", 0, "Do not prefill VAE device-W over resident peak"},
    {"VAE prewarm host mmap/hostreg", "kick_vae_prewarm (last-NFE default)",
     "VAE decode join", "vae_th join", 1,
     "H3_GEN_VAE_PREWARM; host-only is the safe default under pressure"},
    {"VAE prewarm device-W (~1.8GiB)", "kick_vae_prewarm want_dev",
     "before DiT resident peak / after DiT", "free_giB gate", -1,
     "Gated: INT8 resident=0 → host-only unless H3_VAE_PREWARM_DEV=1"},
    {"VAE rope device tables", "VAE chunk prepare", "scratch drop / origin reset",
     "same-key skip H2D", 1, "C069: reuse when key matches; no cross-clip assume"},
};

enum {
    H3_LIFECYCLE_WHITELIST_N =
        (int)(sizeof(k_h3_lifecycle_whitelist) /
              sizeof(k_h3_lifecycle_whitelist[0]))
};

/* Allowed host sync points (documentation rows; not enforced here). */
static const char *const k_h3_lifecycle_allowed_sync[] = {
    "TE host_stage join (NVFP4 before DiT; INT8 after DiT)",
    "VAE prewarm / decode thread join",
    "explicit nsys / mem-trace boundary",
    "process exit / cleanup",
};

enum {
    H3_LIFECYCLE_ALLOWED_SYNC_N =
        (int)(sizeof(k_h3_lifecycle_allowed_sync) /
              sizeof(k_h3_lifecycle_allowed_sync[0]))
};

/* Hot-path forbid (P3 enforce elsewhere): DiT inter-layer cudaDeviceSynchronize
 * / blocking H2D as the default path. */

static inline int h3_lifecycle_trace_wanted(void) {
    const char *e = getenv("H3_LIFECYCLE_TRACE");
    return (e && e[0] == '1') ? 1 : 0;
}

/* Debug-only transition print. Never changes control flow. */
static inline void h3_lifecycle_trace(const char *object, const char *stage,
                                      const char *action, const char *detail) {
    if (!h3_lifecycle_trace_wanted())
        return;
    fprintf(stdout, "lifecycle_trace: object=%s stage=%s action=%s detail=%s\n",
            object ? object : "?", stage ? stage : "?",
            action ? action : "?", detail ? detail : "-");
    fflush(stdout);
}

static inline void h3_lifecycle_whitelist_fprint(FILE *out) {
    int i;
    if (!out)
        out = stdout;
    fprintf(out, "# H3_LIFECYCLE_WHITELIST  n=%d\n", H3_LIFECYCLE_WHITELIST_N);
    fprintf(out,
            "# columns: object | produce | release | event_dep | "
            "safe_overlap | notes\n");
    for (i = 0; i < H3_LIFECYCLE_WHITELIST_N; i++) {
        const h3_lifecycle_whitelist_row *r = &k_h3_lifecycle_whitelist[i];
        fprintf(out, "lifecycle_wl: object=%s produce=%s release=%s "
                     "event=%s safe_overlap=%d notes=%s\n",
                r->object, r->produce, r->release, r->event_dep,
                r->safe_overlap, r->notes);
    }
    fprintf(out, "# H3_LIFECYCLE_ALLOWED_SYNC  n=%d\n",
            H3_LIFECYCLE_ALLOWED_SYNC_N);
    for (i = 0; i < H3_LIFECYCLE_ALLOWED_SYNC_N; i++)
        fprintf(out, "lifecycle_sync_ok: %s\n", k_h3_lifecycle_allowed_sync[i]);
    fflush(out);
}

#ifdef __cplusplus
}
#endif

#endif /* H3_LIFECYCLE_WHITELIST_H */
