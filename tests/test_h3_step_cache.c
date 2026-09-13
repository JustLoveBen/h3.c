/* Unit tests for h3_step_cache decision logic (no GPU). */
#include "h3_step_cache.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int g_fail;

static void expect(int cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", msg);
        g_fail++;
    }
}

static void test_tea_poly(void) {
    float p = h3_step_cache_tea_poly(0.1f);
    /* 0.1*(0.1)^3 + 0.2*(0.1)^2 + 0.7*0.1 = 0.0001 + 0.002 + 0.07 = 0.0721 */
    expect(fabsf(p - 0.0721f) < 1e-5f, "tea poly(0.1)");
    expect(h3_step_cache_tea_poly(0.0f) == 0.0f, "tea poly(0)");
}

static void test_av_veto_raises(void) {
    float d_v = h3_step_cache_delta_from_velocities(
        H3_STEP_CACHE_FB, 1.0f, 1.0f, 0, 0.0f, 0.0f);
    float d_a = h3_step_cache_delta_from_velocities(
        H3_STEP_CACHE_FB, 1.0f, 1.0f, 1, 1.5f, 1.0f);
    expect(d_v < 1e-6f, "video-stable delta~0");
    expect(d_a > 0.4f, "audio veto raises delta");
}

static void test_tea_skip_middle(void) {
    h3_step_cache c;
    h3_step_cache_init(&c);
    c.kind = H3_STEP_CACHE_TEA;
    c.thresh = 0.05f;
    c.max_consec = 2;
    /* After small delta eval, middle step may skip */
    h3_step_cache_on_eval(&c, 0.01f);
    expect(h3_step_cache_should_eval(&c, 0, 8) == 1, "tea first always eval");
    expect(h3_step_cache_should_eval(&c, 7, 8) == 1, "tea last always eval");
    expect(h3_step_cache_should_eval(&c, 3, 8) == 0, "tea middle skip");
    h3_step_cache_on_skip(&c);
    h3_step_cache_on_skip(&c);
    expect(h3_step_cache_should_eval(&c, 4, 8) == 1, "tea max_consec forces eval");
}

static void test_fb_residual_gate(void) {
    h3_step_cache c;
    h3_step_cache_init(&c);
    c.kind = H3_STEP_CACHE_FB;
    c.thresh = 0.05f;
    c.warmup_steps = 1;
    h3_step_cache_on_eval(&c, 0.02f);
    expect(h3_step_cache_should_eval(&c, 2, 8) == 0, "fb small residual skip");
    h3_step_cache_on_eval(&c, 0.20f);
    expect(h3_step_cache_should_eval(&c, 3, 8) == 1, "fb large residual eval");
}

static void test_mutex(void) {
    h3_step_cache c;
    char err[256];
    h3_step_cache_init(&c);
    c.kind = H3_STEP_CACHE_TEA;
    c.thresh = 0.05f;
    expect(h3_step_cache_validate(&c, 0.0f, 1, err, sizeof(err)) == 1,
           "tea alone ok");
    expect(h3_step_cache_validate(&c, 0.08f, 1, err, sizeof(err)) == 0,
           "tea+mag hard error");
    expect(h3_step_cache_validate(&c, 0.0f, 2, err, sizeof(err)) == 0,
           "tea+reuse hard error");
    c.kind = H3_STEP_CACHE_EASY;
    c.thresh = 0.1f;
    expect(h3_step_cache_validate(&c, 0.0f, 1, err, sizeof(err)) == 0,
           "easy NOT IMPLEMENTED");
    expect(strstr(err, "NOT IMPLEMENTED") != NULL, "easy error text");
}

int main(void) {
    g_fail = 0;
    test_tea_poly();
    test_av_veto_raises();
    test_tea_skip_middle();
    test_fb_residual_gate();
    test_mutex();
    if (g_fail) {
        fprintf(stderr, "%d assertion(s) failed\n", g_fail);
        return 1;
    }
    printf("test_h3_step_cache: PASS\n");
    return 0;
}
