/* tests/test_h3_kpi_guard.c
 * CPU-only: effective SDPA name + strict verdict. */
#include "h3_kpi_guard.h"

#include <stdio.h>
#include <string.h>

static int fails;

static void expect(int c, const char *m) {
    if (!c) {
        fprintf(stderr, "FAIL %s\n", m);
        fails++;
    }
}

static void expect_str(const char *got, const char *want, const char *m) {
    if (!got || !want || strcmp(got, want) != 0) {
        fprintf(stderr, "FAIL %s got='%s' want='%s'\n", m, got ? got : "(null)",
                want ? want : "(null)");
        fails++;
    }
}

int main(void) {
    char buf[64];
    char why[128];

    h3_kpi_guard_sdpa_effective("sage3", 200, 0, 200, 0, "cuda_cudnn_sdpa",
                                buf, sizeof(buf));
    expect_str(buf, "fallback:sage3", "768P sage3 miss is fallback");

    h3_kpi_guard_sdpa_effective("sage3", 400, 400, 0, 0, "sage3_blackwell_fp4",
                                buf, sizeof(buf));
    expect_str(buf, "sage3_blackwell_fp4", "sage3 hits==want uses last");

    h3_kpi_guard_sdpa_effective("sage3", 10, 4, 6, 0, "cuda_cudnn_sdpa", buf,
                                sizeof(buf));
    expect_str(buf, "mixed", "partial sage3 is mixed");

    h3_kpi_guard_sdpa_effective("cudnn", 250, 250, 0, 0, "cuda_cudnn_sdpa",
                                buf, sizeof(buf));
    expect_str(buf, "cuda_cudnn_sdpa", "cudnn hits==want true name");

    h3_kpi_guard_sdpa_effective("cudnn", 0, 0, 0, 0, "(none)", buf,
                                sizeof(buf));
    expect_str(buf, "cudnn", "want=0 falls back to request");

    h3_kpi_guard_sdpa_effective("sol_cute", 80, 0, 80, 0, NULL, buf,
                                sizeof(buf));
    expect_str(buf, "fallback:sol_cute", "sol miss is fallback");

    /* dense-guard arm: shortfall fully covered by dense_skip → real name. */
    h3_kpi_guard_sdpa_effective("sol_cute", 400, 384, 0, 16, "cute_sm120",
                                buf, sizeof(buf));
    expect_str(buf, "cute_sm120", "V1-a dense guard not mixed");
    h3_kpi_guard_sdpa_effective("sol_cute", 400, 300, 0, 100, "cute_sm120",
                                buf, sizeof(buf));
    expect_str(buf, "cute_sm120", "V1-c dense guard not mixed");
    /* shortfall NOT covered by dense_skip → still mixed. */
    h3_kpi_guard_sdpa_effective("sol_cute", 400, 380, 20, 16, "cute_sm120",
                                buf, sizeof(buf));
    expect_str(buf, "mixed", "uncovered shortfall is mixed");

    expect(h3_kpi_guard_sdpa_verdict(0, 0, 0, 1, why, sizeof(why)) == 1,
           "want=0 PASS");
    expect(h3_kpi_guard_sdpa_verdict(10, 0, 0, 1, why, sizeof(why)) == 0,
           "strict miss FAIL");
    expect(strstr(why, "hits=0") != NULL, "strict why has hits=0");
    expect(h3_kpi_guard_sdpa_verdict(10, 0, 0, 0, why, sizeof(why)) == 1,
           "non-strict miss PASS");
    expect(h3_kpi_guard_sdpa_verdict(10, 4, 0, 1, why, sizeof(why)) == 0,
           "mixed FAIL");
    expect(strstr(why, "mixed") != NULL, "mixed why");
    expect(h3_kpi_guard_sdpa_verdict(10, 10, 0, 1, why, sizeof(why)) == 1,
           "hits==want PASS");

    /* dense guard: hits = want - dense_skip → PASS (design behavior). */
    expect(h3_kpi_guard_sdpa_verdict(400, 384, 16, 1, why, sizeof(why)) == 1,
           "V1-a guard PASS strict");
    expect(h3_kpi_guard_sdpa_verdict(400, 300, 100, 1, why, sizeof(why)) == 1,
           "V1-c guard PASS strict");
    expect(h3_kpi_guard_sdpa_verdict(400, 288, 112, 1, why, sizeof(why)) == 1,
           "V1-ac guard PASS strict");
    /* real fallback BELOW the guard-adjusted expectation → still FAIL. */
    expect(h3_kpi_guard_sdpa_verdict(400, 383, 16, 1, why, sizeof(why)) == 0,
           "one real fallback still FAIL");
    expect(h3_kpi_guard_sdpa_verdict(400, 0, 16, 1, why, sizeof(why)) == 0,
           "zero hits under guard FAIL");
    /* dense_skip > want clamped: hits=0 → PASS (everything was dense). */
    expect(h3_kpi_guard_sdpa_verdict(4, 0, 100, 1, why, sizeof(why)) == 1,
           "dense_skip>want clamped PASS");

    /* Waffle cold/hot gates (direction A: cold clip1 independent threshold).
     * cold uses its own looser threshold; hot semantics unchanged. */
    {
        double b[5] = {1.1, 3.741, 0.9, 1.0, 1.2};
        expect(h3_kpi_guard_waffle_max_b16(b, 5) == 3.741, "waffle max");
        expect(h3_kpi_guard_waffle_mean_b16(b, 5) > 1.5 &&
                   h3_kpi_guard_waffle_mean_b16(b, 5) < 1.7,
               "waffle mean");
        expect(h3_kpi_guard_waffle_max_b16(NULL, 5) == 0.0, "max null=0");
        expect(h3_kpi_guard_waffle_mean_b16(b, 0) == 0.0, "mean n=0 -> 0");

        /* cold: >2.0 → over (WARN in caller), <=2.0 → ok. */
        expect(h3_kpi_guard_waffle_cold_verdict(3.741, 2.0, why,
                                                sizeof(why)) == 0,
               "cold 3.741 > 2.0 over");
        expect(strstr(why, "cold") != NULL, "cold why tagged");
        expect(h3_kpi_guard_waffle_cold_verdict(1.89, 2.0, why,
                                                sizeof(why)) == 1,
               "cold 1.89 under 2.0 ok");
        expect(h3_kpi_guard_waffle_cold_verdict(2.0, 2.0, why,
                                                sizeof(why)) == 1,
               "cold boundary == threshold ok");
        /* b16==0 means no data (NaN/0 frames) — never a WARN. */
        expect(h3_kpi_guard_waffle_cold_verdict(0.0, 2.0, why,
                                                sizeof(why)) == 1,
               "cold b16=0 no-data ok");
        /* threshold<=0 disables the cold gate. */
        expect(h3_kpi_guard_waffle_cold_verdict(9.9, 0.0, why,
                                                sizeof(why)) == 1,
               "cold threshold=0 disabled");

        /* hot: same shape, independent threshold value. */
        expect(h3_kpi_guard_waffle_hot_verdict(1.716, 1.7, why,
                                               sizeof(why)) == 0,
               "hot 1.716 > 1.7 over");
        expect(strstr(why, "hot") != NULL, "hot why tagged");
        expect(h3_kpi_guard_waffle_hot_verdict(1.696, 1.7, why,
                                               sizeof(why)) == 1,
               "hot under threshold ok");

        /* cold must NOT be judged by the hot threshold (1.7): 1.89 fails
         * the hot threshold but passes the cold one. */
        expect(h3_kpi_guard_waffle_hot_verdict(1.89, 1.7, why,
                                               sizeof(why)) == 0,
               "1.89 over hot 1.7");
        expect(h3_kpi_guard_waffle_cold_verdict(1.89, 2.0, why,
                                                sizeof(why)) == 1,
               "1.89 under cold 2.0 (separate thresholds)");
    }

    return fails ? 1 : (printf("test_h3_kpi_guard: PASS\n"), 0);
}
