#ifndef H3_STEP_CACHE_H
#define H3_STEP_CACHE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Independent step/block cache decision module (opt-in only).
 *
 * Quality default: all kinds OFF. Generate CLI wires this into the denoise
 * loop; MagCache lite remains in generate for backward CLI compatibility and
 * must not be stacked with Tea/FB/Easy (hard error).
 *
 * Kinds:
 *   TEA  — TeaCache-style poly(rel_L1) accumulate + AV veto (raise delta)
 *   FB   — conservative first-block / residual gate on velocity proxy
 *          (single-step |ratio-1|, no poly accumulate; AV veto)
 *   EASY — NOT IMPLEMENTED (configure fails hard)
 *
 * Not MagCache: do not route --magcache-* through this module.
 */

typedef enum {
    H3_STEP_CACHE_OFF = 0,
    H3_STEP_CACHE_TEA = 1,
    H3_STEP_CACHE_FB = 2,
    H3_STEP_CACHE_EASY = 3
} h3_step_cache_kind;

typedef struct {
    h3_step_cache_kind kind;
    float thresh;       /* 0 → treated as off */
    int max_consec;     /* adjacent skips cap; default 2 */
    int warmup_steps;   /* FB: force-eval first N steps (default 1) */
    /* runtime */
    float accum;
    float last_delta;
    int consec;
    int have_probe;
    int fresh; /* 1 → allow skip while under thresh */
} h3_step_cache;

void h3_step_cache_init(h3_step_cache *c);

/* Parse env defaults: H3_TEACACHE_MAX_CONSEC / H3_FBCACHE_MAX_CONSEC /
 * H3_FBCACHE_WARMUP. Does not enable a kind. */
void h3_step_cache_apply_env(h3_step_cache *c);

/*
 * Mutual exclusion vs Mag / reuse / Easy.
 * Returns 0 on success; on failure fills error and returns 0.
 * Policy (hard error):
 *   - Easy always fails (NOT IMPLEMENTED)
 *   - Tea/FB thresh>0 with magcache_thresh>0 → error
 *   - Tea/FB with reuse_interval>1 → error
 *   - Tea and FB both thresh>0 → error
 */
int h3_step_cache_validate(const h3_step_cache *c, float magcache_thresh,
                           int reuse_interval, char *error, size_t error_size);

int h3_step_cache_enabled(const h3_step_cache *c);

const char *h3_step_cache_kind_name(h3_step_cache_kind kind);

/* First / last always eval. Middle may skip when probe allows. */
int h3_step_cache_should_eval(const h3_step_cache *c, int step, int steps);

/* TeaCache poly on |ratio-1| (official-style mild cubic). */
float h3_step_cache_tea_poly(float x);

/*
 * Build delta from video (+ optional audio) mean-abs ratios.
 * AV veto: audio |ratio-1| raises the video delta (never silences audio).
 * kind selects Tea poly vs FB raw |ratio-1|.
 */
float h3_step_cache_delta_from_velocities(h3_step_cache_kind kind,
                                          float video_mean_abs,
                                          float video_prev_mean_abs,
                                          int have_audio,
                                          float audio_mean_abs,
                                          float audio_prev_mean_abs);

/* After a full DiT eval: update accum / residual probe. */
void h3_step_cache_on_eval(h3_step_cache *c, float delta);

/* After a skip: grow Tea accum / bump consec. */
void h3_step_cache_on_skip(h3_step_cache *c);

#ifdef __cplusplus
}
#endif

#endif /* H3_STEP_CACHE_H */
