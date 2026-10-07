#ifndef H3_KPI_GUARD_H
#define H3_KPI_GUARD_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Effective SDPA name from request + plugin/cudnn counters.
 * hits==want && want>0 → last_kernel if set, else request
 * want>0 && hits==0    → fallback:<request>
 * want>0 && hits<want  → mixed (unless dense_skip covers the gap: guarded)
 * want==0              → last_kernel if set, else request
 * Pure C, no CUDA. */
void h3_kpi_guard_sdpa_effective(const char *request, unsigned long long want,
                                 unsigned long long hits,
                                 unsigned long long fallback,
                                 unsigned long long dense_skip,
                                 const char *last_kernel, char *buf, size_t n);

/* strict && want>0 && hits<expected → 0 (FAIL). why is always filled.
 * expected = want - dense_skip (the official MMDiT dense guard routes those
 * calls to dense by design; they are not a real fallback). When
 * dense_skip==0 the semantics are identical to the old hits==want rule. */
int h3_kpi_guard_sdpa_verdict(unsigned long long want, unsigned long long hits,
                              unsigned long long dense_skip, int strict,
                              char *why, size_t n);

/* Waffle b16 guard (cold clip1 / hot clipN separate thresholds).
 * Cold clip1 is naturally noisier than hot clips, so it must not share the
 * hot threshold. Values <= 0 are treated as NaN (b16==0.0 means "no data"
 * in h3_waffle_b16_rgb24) and never trigger a WARN.
 *
 * verdict: max_b16 > threshold (and threshold > 0) → 0 (over); else 1.
 * cold_max_b16_threshold returns the env-tunable cold threshold:
 *   H3_WAFFLE_COLD_MAX unset/invalid/<=0 → 2.0 (cold gate is observational:
 *   a parse error is loudly reported but never disables the gate silently).
 * Pure C, no CUDA. */
double h3_kpi_guard_cold_max_b16_threshold(void);
int h3_kpi_guard_waffle_cold_verdict(double max_b16, double cold_threshold,
                                     char *why, size_t n);
int h3_kpi_guard_waffle_hot_verdict(double max_b16, double hot_threshold,
                                    char *why, size_t n);
double h3_kpi_guard_waffle_max_b16(const double *b16, int n);
double h3_kpi_guard_waffle_mean_b16(const double *b16, int n);

#ifdef __cplusplus
}
#endif

#endif /* H3_KPI_GUARD_H */
