#include "h3_kpi_guard.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void h3_kpi_guard_sdpa_effective(const char *request, unsigned long long want,
                                 unsigned long long hits,
                                 unsigned long long fallback,
                                 unsigned long long dense_skip,
                                 const char *last_kernel, char *buf,
                                 size_t n) {
    const char *req = (request && request[0]) ? request : "auto";
    const char *last = (last_kernel && last_kernel[0] &&
                        strcmp(last_kernel, "(none)") != 0)
                           ? last_kernel
                           : NULL;
    unsigned long long guarded;
    (void)fallback;
    if (!buf || n == 0)
        return;
    buf[0] = '\0';
    if (want > 0ull && hits == 0ull) {
        if (dense_skip >= want) {
            /* Every call took the dense guard: expected on a guard arm. */
            snprintf(buf, n, "%s", last ? last : req);
            return;
        }
        snprintf(buf, n, "fallback:%s", req);
        return;
    }
    if (want > 0ull && hits < want) {
        /* Dense-guard-covered shortfall is the design path, not a mixed
         * sparse/dense accident. Report the real backend (last kernel). */
        guarded = want - hits;
        if (dense_skip >= guarded) {
            snprintf(buf, n, "%s", last ? last : req);
            return;
        }
        snprintf(buf, n, "mixed");
        return;
    }
    if (want > 0ull && hits == want) {
        snprintf(buf, n, "%s", last ? last : req);
        return;
    }
    snprintf(buf, n, "%s", last ? last : req);
}

int h3_kpi_guard_sdpa_verdict(unsigned long long want, unsigned long long hits,
                              unsigned long long dense_skip, int strict,
                              char *why, size_t n) {
    unsigned long long expected;
    if (want == 0ull) {
        if (why && n)
            snprintf(why, n, "want=0");
        return 1;
    }
    /* Official MMDiT dense guard (FIRST_DENSE_STEPS/LAYERS): those calls go
     * dense on purpose, so the expected sparse-hit count is want-dense_skip,
     * not want. Anything below THAT is a genuine fallback. */
    expected = (dense_skip < want) ? (want - dense_skip) : 0ull;
    if (hits >= expected) {
        if (why && n)
            snprintf(why, n, "hits=%llu expected=%llu (want=%llu dense_skip=%llu)",
                     hits, expected, want, dense_skip);
        return 1;
    }
    if (hits == 0ull) {
        if (why && n)
            snprintf(why, n, "want>0 hits=0 (dense_skip=%llu)", dense_skip);
        return strict ? 0 : 1;
    }
    if (why && n)
        snprintf(why, n, "mixed hits=%llu expected=%llu (want=%llu dense_skip=%llu)",
                 hits, expected, want, dense_skip);
    return strict ? 0 : 1;
}

double h3_kpi_guard_waffle_max_b16(const double *b16, int n) {
    double mx = 0.0;
    int i;
    if (!b16 || n < 1)
        return 0.0;
    for (i = 0; i < n; i++)
        if (b16[i] > mx)
            mx = b16[i];
    return mx;
}

double h3_kpi_guard_waffle_mean_b16(const double *b16, int n) {
    double s = 0.0;
    int i;
    if (!b16 || n < 1)
        return 0.0;
    for (i = 0; i < n; i++)
        s += b16[i];
    return s / (double)n;
}

double h3_kpi_guard_cold_max_b16_threshold(void) {
    const char *e = getenv("H3_WAFFLE_COLD_MAX");
    char *end = NULL;
    double v;
    if (!e || !e[0])
        return 2.0;
    v = strtod(e, &end);
    if (!end || end == e || *end != '\0' || !(v > 0.0)) {
        fprintf(stderr,
                "waffle_guard: H3_WAFFLE_COLD_MAX='%s' invalid; using 2.0\n",
                e);
        return 2.0;
    }
    return v;
}

int h3_kpi_guard_waffle_cold_verdict(double max_b16, double cold_threshold,
                                     char *why, size_t n) {
    if (why && n)
        snprintf(why, n, "cold max_b16=%.3f threshold=%.3f", max_b16,
                 cold_threshold);
    /* b16<=0 means "no data" (decode skipped / degenerate frame) — never a
     * WARN. threshold<=0 disables the cold gate. */
    if (!(max_b16 > 0.0) || !(cold_threshold > 0.0))
        return 1;
    return max_b16 > cold_threshold ? 0 : 1;
}

int h3_kpi_guard_waffle_hot_verdict(double max_b16, double hot_threshold,
                                    char *why, size_t n) {
    if (why && n)
        snprintf(why, n, "hot max_b16=%.3f threshold=%.3f", max_b16,
                 hot_threshold);
    if (!(max_b16 > 0.0) || !(hot_threshold > 0.0))
        return 1;
    return max_b16 > hot_threshold ? 0 : 1;
}
