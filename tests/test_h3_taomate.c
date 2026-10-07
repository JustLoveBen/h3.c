/* tests/test_h3_taomate.c */
#include "h3_taomate.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int fails;
static void expect(int c, const char *m) {
    if (!c) {
        fprintf(stderr, "FAIL %s\n", m);
        fails++;
    }
}

static int near(float a, float b, float eps) {
    return fabsf(a - b) <= eps;
}

int main(void) {
    float v[4], a[4];
    h3_taomate_plan p;
    char err[192];

    expect(H3_TAOMATE_STATE_INDICES[0] == 0, "idx0");
    expect(H3_TAOMATE_STATE_INDICES[1] == 16, "idx16");
    expect(H3_TAOMATE_STATE_INDICES[2] == 33, "idx33");
    expect(H3_TAOMATE_STATE_INDICES[3] == 49, "idx49");

    expect(h3_taomate_distilled_sigmas(12.0f, v) == 1, "video sigmas");
    expect(near(v[0], 1.0f, 1e-5f), "v0=1");
    expect(near(v[1], 0.961165f, 1e-4f), "v16");
    expect(near(v[2], 0.853333f, 1e-4f), "v33");
    expect(near(v[3], 0.0f, 1e-6f), "v49=0");

    expect(h3_taomate_distilled_sigmas(3.0f, a) == 1, "audio sigmas");
    expect(near(a[0], 1.0f, 1e-5f), "a0=1");
    expect(near(a[1], 0.86087f, 1e-4f), "a16");
    expect(near(a[2], 0.59259f, 1e-4f), "a33");
    expect(near(a[3], 0.0f, 1e-6f), "a49=0");

    {
        float q[8];
        expect(H3_TAOMATE_QUALITY_INDICES[1] == 8, "qidx8");
        expect(H3_TAOMATE_QUALITY_INDICES[3] == 24, "qidx24");
        expect(h3_taomate_nfe_sigmas(6, 12.0f, q) == 1, "nfe6 video");
        expect(near(q[0], 1.0f, 1e-5f), "q0=1");
        expect(near(q[2], 0.961165f, 1e-4f), "q nfe6 keeps v16");
        expect(near(q[4], 0.853333f, 1e-4f), "q nfe6 keeps v33");
        expect(near(q[6], 0.0f, 1e-6f), "q nfe6 terminal 0");
        expect(q[1] > q[2] && q[2] > q[3] && q[3] > q[4], "nfe6 decreasing");
        expect(h3_taomate_nfe_sigmas(4, 12.0f, q) == 0, "nfe4 rejected");
        expect(h3_taomate_layer_kv_budget(1008, 6, 2, 1) ==
                   1008 * 8 + 8 * 16 + 32,
               "budget hist6+sink2");
    }

    /* Serving 3-step is NOT TaoMate — first interior sigma must differ. */
    {
        float serving_mid = 12.0f * (2.0f / 3.0f) /
                            (1.0f + 11.0f * (2.0f / 3.0f));
        expect(!near(v[1], serving_mid, 1e-3f), "not serving-3");
    }

    expect(h3_taomate_audio_latent_boundary(0) == 0, "audio 0");
    expect(h3_taomate_audio_latent_boundary(39) == 65, "audio 39");
    expect(h3_taomate_audio_latent_boundary(124) == 207, "audio 124");

    memset(&p, 0, sizeof(p));
    expect(h3_taomate_plan_build(124, &p, err, sizeof(err)) == 1, "plan 5s");
    expect(p.n_phases == 4 && p.video_t == 37 && p.audio_t == 207, "5s totals");
    expect(p.phases[0].video_t1 - p.phases[0].video_t0 == 12, "p0 v=12");
    expect(p.phases[1].video_t1 - p.phases[1].video_t0 == 10, "p1 v=10");
    expect(p.phases[2].video_t1 - p.phases[2].video_t0 == 10, "p2 v=10");
    expect(p.phases[3].video_t1 - p.phases[3].video_t0 == 5, "p3 v=5");
    expect(p.phases[0].group_count == 2 && p.phases[3].group_count == 1,
           "groups 2,2,2,1");

    memset(&p, 0, sizeof(p));
    expect(h3_taomate_plan_build(243, &p, err, sizeof(err)) == 1, "plan 10s");
    expect(p.n_phases == 8 && p.video_t == 72, "10s 8 chunks / 72 T");
    expect(p.phases[4].video_t1 - p.phases[4].video_t0 == 10, "cont p0 v=10");
    expect(p.phases[7].video_t1 - p.phases[7].video_t0 == 5, "cont last v=5");
    expect(p.phases[4].video_t0 == 37, "cont starts at 37");

    expect(h3_taomate_plan_build(17, &p, err, sizeof(err)) == 0, "reject short");

    memset(&p, 0, sizeof(p));
    expect(h3_taomate_plan_build_oneshot(124, &p, err, sizeof(err)) == 1,
           "oneshot 5s");
    expect(p.n_phases == 1 && p.video_t == 37 && p.audio_t == 207,
           "oneshot totals");
    expect(p.phases[0].video_t0 == 0 && p.phases[0].video_t1 == 37,
           "oneshot one window");
    expect(p.phases[0].frame_stop == 124, "oneshot 124 frames");
    expect(h3_taomate_denoise_t0(&p, 0, 2) == 0, "oneshot ov unused");
    expect(p.phases[0].video_t1 - h3_taomate_denoise_t0(&p, 0, 2) == 37,
           "oneshot denoise 37 T");
    expect(h3_taomate_plan_build_oneshot(243, &p, err, sizeof(err)) == 0,
           "oneshot reject 10s");

    memset(&p, 0, sizeof(p));
    expect(h3_taomate_plan_build(124, &p, err, sizeof(err)) == 1, "plan 5s ov");
    {
        h3_taomate_ov_span sp[H3_TAOMATE_MAX_OV_SPANS];
        int n, extra, i;
        n = h3_taomate_overlap_spans(&p, 0, 2, 2, sp);
        expect(n == 0, "p0 no ov");
        n = h3_taomate_overlap_spans(&p, 1, 0, 0, sp);
        expect(n == 0, "disabled ov");
        n = h3_taomate_overlap_spans(&p, 1, 2, 2, sp);
        expect(n == 2, "p1 sink+hist");
        expect(sp[0].video_t0 == 0 && sp[0].video_t1 == 2, "sink first 2");
        expect(sp[1].video_t0 == 10 && sp[1].video_t1 == 12, "hist last 2 of p0");
        extra = 0;
        for (i = 0; i < n; i++)
            extra += sp[i].video_t1 - sp[i].video_t0;
        expect(extra == 4 && extra < p.video_t, "p1 extra T bounded");
        n = h3_taomate_overlap_spans(&p, 3, 2, 2, sp);
        extra = 0;
        for (i = 0; i < n; i++)
            extra += sp[i].video_t1 - sp[i].video_t0;
        expect(n == 2 && extra == 4 && extra < p.video_t, "last chunk not full Q");
        expect(sp[1].video_t1 == p.phases[2].video_t1, "hist from prev phase");
        n = h3_taomate_overlap_spans(&p, 1, 12, 2, sp);
        extra = 0;
        for (i = 0; i < n; i++)
            extra += sp[i].video_t1 - sp[i].video_t0;
        expect(extra == 12 && extra < p.video_t, "hist cap to prev chunk not clip");
        expect(h3_taomate_denoise_t0(&p, 0, 2) == 0, "p0 denoise t0");
        expect(h3_taomate_denoise_t0(&p, 1, 2) == 10, "p1 ov2 t0=10");
        expect(h3_taomate_denoise_t0(&p, 1, 4) == 8, "p1 ov4 t0=8");
        expect(p.phases[1].video_t1 - h3_taomate_denoise_t0(&p, 1, 2) == 12,
               "p1 window 12 T not 14");
        expect(p.phases[1].video_t1 - h3_taomate_denoise_t0(&p, 1, 4) == 14,
               "p1 ov4 window 14");
        expect(h3_taomate_denoise_t0(&p, 1, 4) >= p.phases[0].video_t0,
               "ov stays in prev phase");
    }
    {
        uint32_t idx[4096];
        size_t n_sink = 0, n, i;
        size_t spatial = 4;
        size_t audio_off = 10;
        size_t video_off = 10 + 20;
        n = h3_taomate_layer_kv_save_idx(&p, 0, 2, 2, audio_off, video_off,
                                         spatial, 1, 10, idx, 4096, &n_sink);
        expect(n > 0 && n_sink > 0 && n_sink < n, "p0 sink+hist idx");
        expect(idx[0] == (uint32_t)video_off, "p0 starts at video_off");
        expect(h3_taomate_layer_kv_budget(1008, 2, 2, 1) <
                   1008 * 12 + 1000,
               "budget << one 12T chunk");
        expect(h3_taomate_layer_kv_budget(1008, 2, 2, 1) ==
                   1008 * 4 + 4 * 16 + 32,
               "budget 4T video + audio slack");
        n = h3_taomate_layer_kv_save_idx(&p, 1, 2, 2, audio_off, video_off,
                                         spatial, 0, 0, idx, 4096, &n_sink);
        expect(n == 2 * spatial && n_sink == 0, "p1 hist video only");
        for (i = 0; i < n; i++)
            expect(idx[i] >= (uint32_t)video_off, "p1 idx in video");
    }
    memset(&p, 0, sizeof(p));
    expect(h3_taomate_plan_build(243, &p, err, sizeof(err)) == 1, "plan 10s ov");
    {
        h3_taomate_ov_span sp[H3_TAOMATE_MAX_OV_SPANS];
        int n, extra, i;
        n = h3_taomate_overlap_spans(&p, 4, 2, 2, sp);
        extra = 0;
        for (i = 0; i < n; i++)
            extra += sp[i].video_t1 - sp[i].video_t0;
        expect(n == 2 && extra == 4 && extra < p.video_t, "10s extra not 72 T");
    }

    return fails ? 1 : (printf("test_h3_taomate: PASS\n"), 0);
}
