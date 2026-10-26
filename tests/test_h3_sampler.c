#include "h3_sampler.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int fails;

static void expect(int cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", msg);
        fails++;
    }
}

static void expect_near(float a, float b, float tol, const char *msg) {
    if (!(fabsf(a - b) <= tol)) {
        fprintf(stderr, "FAIL: %s (got %.8g want %.8g)\n", msg, (double)a,
                (double)b);
        fails++;
    }
}

int main(void) {
    char err[256];
    float x[4], v[4], noise[4];
    h3_sampler_id id;

    fails = 0;

    expect(h3_sampler_parse(NULL, err, sizeof err) == H3_SAMPLER_VELOCITY_EULER,
           "null → euler");
    expect(h3_sampler_parse("euler", err, sizeof err) ==
               H3_SAMPLER_VELOCITY_EULER,
           "euler alias");
    expect(h3_sampler_parse("lcm_renoise", err, sizeof err) ==
               H3_SAMPLER_LCM_RENOISE,
           "lcm_renoise");
    expect(h3_sampler_parse("lcm", err, sizeof err) == H3_SAMPLER_LCM_RENOISE,
           "lcm alias");
    expect(h3_sampler_parse("lcm+simple", err, sizeof err) ==
               H3_SAMPLER_LCM_RENOISE,
           "lcm+simple alias");
    err[0] = '\0';
    id = h3_sampler_parse("bogus", err, sizeof err);
    expect(id == H3_SAMPLER_VELOCITY_EULER, "bogus → euler fallback");
    expect(err[0] != '\0', "bogus fills err");
    expect(strcmp(h3_sampler_name(H3_SAMPLER_LCM_RENOISE), "lcm_renoise") == 0,
           "name");

    /* Final step σ′=0: X ← x0 = X + σ·v  (no noise). */
    x[0] = 1.0f;
    x[1] = -2.0f;
    v[0] = 0.5f;
    v[1] = 1.0f;
    err[0] = '\0';
    expect(h3_sampler_lcm_renoise_step_f32(x, v, 2, 0.8f, 0.0f, NULL, err,
                                           sizeof err) == 1,
           "final step ok");
    expect_near(x[0], 1.0f + 0.8f * 0.5f, 1e-6f, "final x0[0]");
    expect_near(x[1], -2.0f + 0.8f * 1.0f, 1e-6f, "final x0[1]");

    /* Mid step: X ← (1−σ′)·x0 + σ′·ε */
    x[0] = 0.0f;
    x[1] = 1.0f;
    x[2] = -1.0f;
    x[3] = 2.0f;
    v[0] = 1.0f;
    v[1] = 0.0f;
    v[2] = 2.0f;
    v[3] = -0.5f;
    noise[0] = 0.25f;
    noise[1] = -0.5f;
    noise[2] = 1.0f;
    noise[3] = 0.0f;
    {
        float sigma = 1.0f, sigma_next = 0.25f;
        float x0_0 = 0.0f + sigma * 1.0f;
        float x0_1 = 1.0f + sigma * 0.0f;
        float x0_2 = -1.0f + sigma * 2.0f;
        float x0_3 = 2.0f + sigma * (-0.5f);
        expect(h3_sampler_lcm_renoise_step_f32(x, v, 4, sigma, sigma_next, noise,
                                               err, sizeof err) == 1,
               "mid step ok");
        expect_near(x[0], (1.0f - sigma_next) * x0_0 + sigma_next * noise[0],
                    1e-6f, "mid[0]");
        expect_near(x[1], (1.0f - sigma_next) * x0_1 + sigma_next * noise[1],
                    1e-6f, "mid[1]");
        expect_near(x[2], (1.0f - sigma_next) * x0_2 + sigma_next * noise[2],
                    1e-6f, "mid[2]");
        expect_near(x[3], (1.0f - sigma_next) * x0_3 + sigma_next * noise[3],
                    1e-6f, "mid[3]");
    }

    /* Mid step without noise → fail */
    expect(h3_sampler_lcm_renoise_step_f32(x, v, 2, 1.0f, 0.5f, NULL, err,
                                           sizeof err) == 0,
           "mid needs noise");
    expect(strstr(err, "noise") != NULL, "noise msg");

    /* Contrast with euler: same X,v,σ,σ′ → different mid-step result. */
    {
        float xe[2] = {0.0f, 1.0f};
        float xv[2] = {1.0f, 0.0f};
        float sigma = 1.0f, sigma_next = 0.5f;
        float n[2] = {0.0f, 0.0f};
        float euler0, lcm0;
        /* euler: X += (σ−σ′)·v */
        euler0 = xe[0] + (sigma - sigma_next) * xv[0];
        expect(h3_sampler_lcm_renoise_step_f32(xe, xv, 2, sigma, sigma_next, n,
                                               err, sizeof err) == 1,
               "contrast lcm");
        lcm0 = xe[0];
        /* x0=0+1*1=1; X=(1-0.5)*1+0.5*0=0.5; euler=0+(1-0.5)*1=0.5 — same
         * when noise=0 and this particular (X,v). Use nonzero noise: */
        xe[0] = 0.0f;
        xe[1] = 1.0f;
        n[0] = 2.0f;
        n[1] = -1.0f;
        expect(h3_sampler_lcm_renoise_step_f32(xe, xv, 2, sigma, sigma_next, n,
                                               err, sizeof err) == 1,
               "contrast noisy");
        /* x0=1; X=0.5*1+0.5*2=1.5; euler would be 0.5 */
        expect_near(xe[0], 1.5f, 1e-6f, "noisy lcm ≠ euler");
        expect(fabsf(xe[0] - euler0) > 0.4f, "diverges from euler");
        (void)lcm0;
    }


    /* Edge cases: empty count, sigma_next≈0 with noise ptr, bad sigma. */
    {
        float xe[1] = {1.0f};
        float xv[1] = {0.0f};
        float n[1] = {9.0f};
        err[0] = '\0';
        expect(h3_sampler_lcm_renoise_step_f32(xe, xv, 0, 1.0f, 0.5f, n, err,
                                               sizeof err) == 0,
               "empty count fail");
        expect(h3_sampler_lcm_renoise_step_f32(NULL, xv, 1, 1.0f, 0.0f, NULL,
                                               err, sizeof err) == 0,
               "null sample fail");
        /* Final step ignores noise pointer. */
        xe[0] = 1.0f;
        xv[0] = 0.5f;
        expect(h3_sampler_lcm_renoise_step_f32(xe, xv, 1, 0.4f, 0.0f, n, err,
                                               sizeof err) == 1,
               "final ignores noise");
        expect_near(xe[0], 1.0f + 0.4f * 0.5f, 1e-6f, "final+noiseptr");
        /* Non-finite sigma → fail */
        xe[0] = 0.0f;
        expect(h3_sampler_lcm_renoise_step_f32(xe, xv, 1, NAN, 0.5f, n,
                                               err, sizeof err) == 0,
               "nan sigma fail");
    }

    if (fails) {
        fprintf(stderr, "%d failure(s)\n", fails);
        return 1;
    }
    printf("ok h3_sampler_test\n");
    return 0;
}
