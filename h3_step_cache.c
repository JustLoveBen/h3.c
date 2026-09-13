#include "h3_step_cache.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void h3_step_cache_init(h3_step_cache *c) {
    if (!c) return;
    memset(c, 0, sizeof(*c));
    c->max_consec = 2;
    c->warmup_steps = 1;
    c->fresh = 1;
}

static int env_int(const char *name, int fallback) {
    const char *e = getenv(name);
    char *end = NULL;
    long v;
    if (!e || !e[0]) return fallback;
    v = strtol(e, &end, 10);
    if (end == e || *end) return fallback;
    if (v < 0 || v > 32) return fallback;
    return (int)v;
}

void h3_step_cache_apply_env(h3_step_cache *c) {
    if (!c) return;
    if (c->kind == H3_STEP_CACHE_TEA)
        c->max_consec = env_int("H3_TEACACHE_MAX_CONSEC", c->max_consec);
    else if (c->kind == H3_STEP_CACHE_FB) {
        c->max_consec = env_int("H3_FBCACHE_MAX_CONSEC", c->max_consec);
        c->warmup_steps = env_int("H3_FBCACHE_WARMUP", c->warmup_steps);
        if (c->warmup_steps < 1) c->warmup_steps = 1;
    }
}

int h3_step_cache_enabled(const h3_step_cache *c) {
    return c && c->kind != H3_STEP_CACHE_OFF && c->thresh > 0.0f &&
           isfinite(c->thresh);
}

const char *h3_step_cache_kind_name(h3_step_cache_kind kind) {
    switch (kind) {
    case H3_STEP_CACHE_TEA:
        return "teacache";
    case H3_STEP_CACHE_FB:
        return "fbcache";
    case H3_STEP_CACHE_EASY:
        return "easycache";
    case H3_STEP_CACHE_OFF:
    default:
        return "off";
    }
}

int h3_step_cache_validate(const h3_step_cache *c, float magcache_thresh,
                           int reuse_interval, char *error, size_t error_size) {
    int tea_on, fb_on, mag_on, reuse_on;
    if (!c) {
        if (error && error_size)
            snprintf(error, error_size, "step_cache null");
        return 0;
    }
    if (c->kind == H3_STEP_CACHE_EASY) {
        if (error && error_size)
            snprintf(error, error_size,
                     "easycache NOT IMPLEMENTED (AV split required; see "
                     "plan — refuse half-wired path)");
        return 0;
    }
    tea_on = (c->kind == H3_STEP_CACHE_TEA && c->thresh > 0.0f);
    fb_on = (c->kind == H3_STEP_CACHE_FB && c->thresh > 0.0f);
    mag_on = magcache_thresh > 0.0f && isfinite(magcache_thresh);
    reuse_on = reuse_interval > 1;
    if (tea_on && fb_on) {
        if (error && error_size)
            snprintf(error, error_size,
                     "teacache and fbcache are mutually exclusive");
        return 0;
    }
    if ((tea_on || fb_on) && mag_on) {
        if (error && error_size)
            snprintf(error, error_size,
                     "%s mutually exclusive with --magcache-thresh",
                     h3_step_cache_kind_name(c->kind));
        return 0;
    }
    if ((tea_on || fb_on) && reuse_on) {
        if (error && error_size)
            snprintf(error, error_size,
                     "%s mutually exclusive with --reuse > 1 "
                     "(fixed schedule wins only alone)",
                     h3_step_cache_kind_name(c->kind));
        return 0;
    }
    if (c->thresh > 0.0f && c->kind == H3_STEP_CACHE_OFF) {
        if (error && error_size)
            snprintf(error, error_size, "step_cache thresh set but kind=off");
        return 0;
    }
    return 1;
}

int h3_step_cache_should_eval(const h3_step_cache *c, int step, int steps) {
    if (!h3_step_cache_enabled(c)) return 1;
    if (step <= 0 || step >= steps - 1) return 1;
    if (c->kind == H3_STEP_CACHE_FB && step < c->warmup_steps) return 1;
    if (!c->have_probe) return 1;
    if (c->consec >= c->max_consec) return 1;
    if (c->kind == H3_STEP_CACHE_TEA) {
        if (!c->fresh) return 1;
        if (!(c->accum < c->thresh)) return 1;
        return 0; /* skip */
    }
    if (c->kind == H3_STEP_CACHE_FB) {
        /* Single-step residual gate (conservative FB / Cache-DiT thought). */
        if (!(c->last_delta < c->thresh)) return 1;
        return 0;
    }
    return 1;
}

float h3_step_cache_tea_poly(float x) {
    if (!(x >= 0.0f) || !isfinite(x)) x = 0.0f;
    return ((0.10f * x + 0.20f) * x + 0.70f) * x;
}

float h3_step_cache_delta_from_velocities(h3_step_cache_kind kind,
                                          float video_mean_abs,
                                          float video_prev_mean_abs,
                                          int have_audio,
                                          float audio_mean_abs,
                                          float audio_prev_mean_abs) {
    float ratio, x, xa;
    ratio = video_prev_mean_abs > 1e-12f ? video_mean_abs / video_prev_mean_abs
                                         : 1.0f;
    x = fabsf(ratio - 1.0f);
    if (have_audio) {
        ratio = audio_prev_mean_abs > 1e-12f
                    ? audio_mean_abs / audio_prev_mean_abs
                    : 1.0f;
        xa = fabsf(ratio - 1.0f);
        if (xa > x) x = xa; /* AV veto: raise delta */
    }
    if (kind == H3_STEP_CACHE_TEA) return h3_step_cache_tea_poly(x);
    return x; /* FB / raw */
}

void h3_step_cache_on_eval(h3_step_cache *c, float delta) {
    if (!h3_step_cache_enabled(c)) return;
    if (!(delta >= 0.0f) || !isfinite(delta)) delta = 0.0f;
    c->last_delta = delta;
    c->have_probe = 1;
    c->consec = 0;
    if (c->kind == H3_STEP_CACHE_TEA) {
        c->accum += delta;
        c->fresh = c->accum < c->thresh ? 1 : 0;
        if (!c->fresh) c->accum = 0.0f;
    } else if (c->kind == H3_STEP_CACHE_FB) {
        c->accum = delta; /* last residual only */
        c->fresh = delta < c->thresh ? 1 : 0;
    }
}

void h3_step_cache_on_skip(h3_step_cache *c) {
    if (!h3_step_cache_enabled(c) || !c->have_probe) return;
    c->consec++;
    if (c->kind == H3_STEP_CACHE_TEA)
        c->accum += c->last_delta;
}
