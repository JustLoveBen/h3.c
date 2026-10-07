/* tests/test_h3_dit_w4a8_cond.c
 * INT8 打开路径对齐 tools/h3_cuda_generate.c:688-749。
 * 阶段 A：forward_av 自洽。阶段 B：avc(cond_seq=0) bit-for-bit；cond_seq>0 形状。 */
#include "h3_dit_int8.h"
#include "h3_dit_w4a8.h"
#include "h3_gpu_cuda.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VIDEO_PATCH 96
#define AUDIO_CH 32
#define HIDDEN 5376

/* Fixture matches Stage A: video_seq=16 but fill_metal_video_pos(2,4,4)
 * only writes 8 RoPE slots. Do not treat this as a packed video grid. */
#define TEXT_SEQ 8
#define AUDIO_SEQ 4
#define VIDEO_SEQ 16
#define COND_SEQ 8
#define N_BLOCKS 2

static int same_bits(const float *a, const float *b, size_t n) {
    return memcmp(a, b, n * sizeof(float)) == 0;
}

static int all_finite(const float *a, size_t n) {
    size_t i;
    for (i = 0; i < n; i++) {
        if (!isfinite(a[i])) return 0;
    }
    return 1;
}

/* Gold numbers from tests/test_h3.c / h3_host.h:101-106. */
static int test_reference_image_canvas(char *err, size_t es) {
    int width = 0, height = 0;
    if (!h3_dit_w4a8_reference_image_canvas(1920, 1080, 512, 512, 0, &width,
                                            &height) ||
        width != 672 || height != 384) {
        snprintf(err, es, "ref canvas 1920x1080 vs 512x512 got %dx%d want 672x384",
                 width, height);
        return 0;
    }
    if (!h3_dit_w4a8_reference_image_canvas(640, 480, 1024, 1024, 0, &width,
                                            &height) ||
        width != 640 || height != 480) {
        snprintf(err, es, "ref canvas must not upscale 640x480, got %dx%d",
                 width, height);
        return 0;
    }
    if (!h3_dit_w4a8_reference_image_canvas(4096, 2160, 512, 512, 2048, &width,
                                            &height) ||
        width != 3872 || height != 2048) {
        snprintf(err, es, "ref canvas max_short=2048 got %dx%d want 3872x2048",
                 width, height);
        return 0;
    }
    if (h3_dit_w4a8_reference_image_canvas(0, 480, 512, 512, 0, &width,
                                           &height)) {
        snprintf(err, es, "ref canvas accepted invalid 0-width");
        return 0;
    }
    puts("ok: reference_image_canvas gold (down-only, aspect-preserving)");
    return 1;
}

/* Ref2VA: pos_t = cursor (text_len, then +1). Not last-keyframe formula.
 * Spatial grid may differ from the 864x480 target (30x54). */
static int test_ref_image_rope_cursor(char *err, size_t es) {
    const int ref_lh = 40, ref_lw = 40; /* 640x640 canvas / 16 */
    const size_t rows = (size_t)(ref_lh / 2) * (size_t)(ref_lw / 2);
    const size_t text_seq = 128;
    const int latent_t = 7, tgt_h = 30, tgt_w = 54;
    const size_t video_seq =
        (size_t)latent_t * (size_t)(tgt_h / 2) * (size_t)(tgt_w / 2);
    const float last_kf =
        (float)text_seq + h3_dit_w4a8_video_span_sum(latent_t) - (5.0f / 3.0f);
    float *t0 = calloc(rows, sizeof(float));
    float *h0 = calloc(rows, sizeof(float));
    float *w0 = calloc(rows, sizeof(float));
    float *t1 = calloc(rows, sizeof(float));
    float *h1 = calloc(rows, sizeof(float));
    float *w1 = calloc(rows, sizeof(float));
    float *vt = calloc(video_seq, sizeof(float));
    float *vh = calloc(video_seq, sizeof(float));
    float *vw = calloc(video_seq, sizeof(float));
    size_t i;
    int ok = 0;
    if (rows != 400) {
        snprintf(err, es, "expected 400 ref rows for 40x40, got %zu", rows);
        goto done;
    }
    if (!t0 || !h0 || !w0 || !t1 || !h1 || !w1 || !vt || !vh || !vw) {
        snprintf(err, es, "OOM ref rope");
        goto done;
    }
    if (!h3_dit_w4a8_fill_metal_ref_image_pos((float)text_seq, ref_lh, ref_lw,
                                              t0, h0, w0, err, es) ||
        !h3_dit_w4a8_fill_metal_ref_image_pos((float)text_seq + 1.0f, ref_lh,
                                              ref_lw, t1, h1, w1, err, es))
        goto done;
    for (i = 0; i < rows; i++) {
        if (t0[i] != (float)text_seq) {
            snprintf(err, es, "ref0 pos_t[%zu]=%g want cursor %zu (not keyframe)",
                     i, (double)t0[i], text_seq);
            goto done;
        }
        if (t1[i] != (float)text_seq + 1.0f) {
            snprintf(err, es, "ref1 pos_t[%zu]=%g want cursor %zu+1", i,
                     (double)t1[i], text_seq);
            goto done;
        }
        if (t0[i] == last_kf) {
            snprintf(err, es, "ref image reused last-keyframe condition_time");
            goto done;
        }
    }
    if (h0[0] == h0[rows - 1] && w0[0] == w0[rows - 1]) {
        snprintf(err, es, "ref spatial grid collapsed");
        goto done;
    }
    /* After one image ref, target video starts at text_len+1. */
    if (!h3_dit_w4a8_fill_metal_video_pos(text_seq + 1, latent_t, tgt_h, tgt_w,
                                          vt, vh, vw, err, es))
        goto done;
    if (vt[0] != (float)text_seq + 1.0f) {
        snprintf(err, es, "video after 1 ref pos_t[0]=%g want text_len+1=%g",
                 (double)vt[0], (double)text_seq + 1.0);
        goto done;
    }
    if (vt[0] == (float)text_seq) {
        snprintf(err, es, "video RoPE still starts at text_len (FL2VA keyframe)");
        goto done;
    }
    ok = 1;
    printf("ok: REF_IMAGE cursor=%zu then +1; video starts at %zu (not last_kf "
           "%g)\n",
           text_seq, text_seq + 1, (double)last_kf);
done:
    free(t0);
    free(h0);
    free(w0);
    free(t1);
    free(h1);
    free(w1);
    free(vt);
    free(vh);
    free(vw);
    return ok;
}

static int test_fill_metal_cond_pos(char *err, size_t es) {
    const int latent_h = 30, latent_w = 54;
    const size_t rows = (size_t)(latent_h / 2) * (size_t)(latent_w / 2);
    const float condition_time = 128.0f;
    float *pt = calloc(rows, sizeof(float));
    float *ph = calloc(rows, sizeof(float));
    float *pw = calloc(rows, sizeof(float));
    size_t i;
    int ok = 0;
    if (rows != 405) {
        snprintf(err, es, "expected 405 cond rows, got %zu", rows);
        goto done;
    }
    if (!pt || !ph || !pw) {
        snprintf(err, es, "OOM fill_metal_cond_pos");
        goto done;
    }
    if (!h3_dit_w4a8_fill_metal_cond_pos(condition_time, latent_h, latent_w, pt,
                                         ph, pw, err, es))
        goto done;
    for (i = 0; i < rows; i++) {
        if (pt[i] != condition_time) {
            snprintf(err, es, "cond pos_t[%zu]=%g want %g (no frame_rescale)",
                     i, (double)pt[i], (double)condition_time);
            goto done;
        }
    }
    if (ph[0] == ph[rows - 1] && pw[0] == pw[rows - 1]) {
        snprintf(err, es, "cond spatial grid collapsed");
        goto done;
    }
    ok = 1;
    puts("ok: fill_metal_cond_pos 405 rows, pos_t pinned");
done:
    free(pt);
    free(ph);
    free(pw);
    return ok;
}

/* Metal last-keyframe: text_len + span_sum(T) - 5/3. T=7 theme canvas. */
static int test_video_span_sum_last_cond(char *err, size_t es) {
    const int latent_t = 7, latent_h = 30, latent_w = 54;
    const size_t video_seq =
        (size_t)latent_t * (size_t)(latent_h / 2) * (size_t)(latent_w / 2);
    const size_t frame_rows = (size_t)(latent_h / 2) * (size_t)(latent_w / 2);
    const size_t text_seq = 128;
    const float want_span = (5.0f / 3.0f) * 22.0f; /* 1+4+4+4+4+1+4 */
    const float want_last = (float)text_seq + want_span - (5.0f / 3.0f);
    float span, last_t;
    float *pt = calloc(video_seq, sizeof(float));
    float *ph = calloc(video_seq, sizeof(float));
    float *pw = calloc(video_seq, sizeof(float));
    float *ct = calloc(frame_rows, sizeof(float));
    float *ch = calloc(frame_rows, sizeof(float));
    float *cw = calloc(frame_rows, sizeof(float));
    size_t i;
    int ok = 0;
    if (!pt || !ph || !pw || !ct || !ch || !cw) {
        snprintf(err, es, "OOM video_span_sum");
        goto done;
    }
    span = h3_dit_w4a8_video_span_sum(latent_t);
    if (fabsf(span - want_span) > 1e-5f) {
        snprintf(err, es, "span_sum(7)=%g want %g", (double)span,
                 (double)want_span);
        goto done;
    }
    if (!h3_dit_w4a8_fill_metal_video_pos(text_seq, latent_t, latent_h, latent_w,
                                          pt, ph, pw, err, es))
        goto done;
    if (pt[0] != (float)text_seq) {
        snprintf(err, es, "video pos_t[0]=%g want text_len %zu", (double)pt[0],
                 text_seq);
        goto done;
    }
    last_t = (float)text_seq + span - (5.0f / 3.0f);
    if (fabsf(last_t - want_last) > 1e-5f) {
        snprintf(err, es, "last condition_time=%g want %g", (double)last_t,
                 (double)want_last);
        goto done;
    }
    if (!h3_dit_w4a8_fill_metal_cond_pos(last_t, latent_h, latent_w, ct, ch, cw,
                                         err, es))
        goto done;
    for (i = 0; i < frame_rows; i++) {
        if (ct[i] != last_t) {
            snprintf(err, es, "last cond pos_t[%zu]=%g want %g (pinned)", i,
                     (double)ct[i], (double)last_t);
            goto done;
        }
    }
    if (ct[0] == (float)text_seq) {
        snprintf(err, es, "last condition_time collapsed to first-frame text_len");
        goto done;
    }
    ok = 1;
    printf("ok: video_span_sum(7)=%g last condition_time=%g (not text_len)\n",
           (double)span, (double)last_t);
done:
    free(pt);
    free(ph);
    free(pw);
    free(ct);
    free(ch);
    free(cw);
    return ok;
}

static int test_av_self_consistent(h3_dit_w4a8 *dit,
                                   const h3_dit_w4a8_core *core, const float *xv,
                                   const float *xa, const float *th,
                                   const float *pt, const float *ph,
                                   const float *pw, const float *at,
                                   const float *ah, const float *aw, char *err,
                                   size_t es) {
    float *vv1 = calloc(VIDEO_SEQ * VIDEO_PATCH, sizeof(float));
    float *va1 = calloc(AUDIO_SEQ * AUDIO_CH, sizeof(float));
    float *vv2 = calloc(VIDEO_SEQ * VIDEO_PATCH, sizeof(float));
    float *va2 = calloc(AUDIO_SEQ * AUDIO_CH, sizeof(float));
    int ok = 0;
    if (!vv1 || !va1 || !vv2 || !va2) {
        snprintf(err, es, "OOM av self-consistent");
        goto done;
    }
    if (!h3_dit_w4a8_forward_av_f32(
            dit, core, 0, N_BLOCKS, 0.5f, 0.3f, xv, VIDEO_SEQ, xa, AUDIO_SEQ, th,
            TEXT_SEQ, pt, ph, pw, at, ah, aw, 1.0f, vv1, va1, 1, NULL, err, es))
        goto done;
    if (!h3_dit_w4a8_forward_av_f32(
            dit, core, 0, N_BLOCKS, 0.5f, 0.3f, xv, VIDEO_SEQ, xa, AUDIO_SEQ, th,
            TEXT_SEQ, pt, ph, pw, at, ah, aw, 1.0f, vv2, va2, 1, NULL, err, es))
        goto done;
    if (!same_bits(vv1, vv2, VIDEO_SEQ * VIDEO_PATCH) ||
        !same_bits(va1, va2, AUDIO_SEQ * AUDIO_CH)) {
        snprintf(err, es, "FAIL: forward_av not self-consistent");
        goto done;
    }
    ok = 1;
    puts("ok: forward_av self-consistent (stage A)");
done:
    free(vv1);
    free(va1);
    free(vv2);
    free(va2);
    return ok;
}

static int test_adaln_0p999_polarity_host(char *err, size_t es) {
    const size_t table_rows = 1000;
    size_t got, want_time, noisy_flow;
    /* sigma=1.0 (F2VA step 0): Metal time=0 → keep VIDEO@0.999, row_map=3. */
    if (!h3_dit_w4a8_visual_cond_adaln_active(COND_SEQ, 1.0f)) {
        snprintf(err, es,
                 "FAIL: sigma=1.0 must keep cond on VIDEO@0.999 (got inactive)");
        return 0;
    }
    if (h3_dit_w4a8_visual_cond_mod_row(COND_SEQ, 1.0f) != 3u) {
        snprintf(err, es, "FAIL: sigma=1.0 row_map want 3 got %u",
                 h3_dit_w4a8_visual_cond_mod_row(COND_SEQ, 1.0f));
        return 0;
    }
    /* sigma<=0.001: Metal time>=0.999 → fallback MOD_VIDEO. */
    if (h3_dit_w4a8_visual_cond_adaln_active(COND_SEQ, 0.001f)) {
        snprintf(err, es, "FAIL: sigma=0.001 must fallback MOD_VIDEO");
        return 0;
    }
    if (h3_dit_w4a8_visual_cond_mod_row(COND_SEQ, 0.001f) !=
        (uint32_t)H3_DIT_W4A8_MOD_VIDEO) {
        snprintf(err, es, "FAIL: sigma=0.001 mod_row want MOD_VIDEO=%d got %u",
                 (int)H3_DIT_W4A8_MOD_VIDEO,
                 h3_dit_w4a8_visual_cond_mod_row(COND_SEQ, 0.001f));
        return 0;
    }
    if (h3_dit_w4a8_visual_cond_adaln_active(COND_SEQ, 0.0f)) {
        snprintf(err, es, "FAIL: sigma=0 must fallback MOD_VIDEO");
        return 0;
    }
    if (!h3_dit_w4a8_visual_cond_adaln_active(COND_SEQ, 0.5f)) {
        snprintf(err, es, "FAIL: sigma=0.5 must stay on VIDEO@0.999");
        return 0;
    }
    /* Empty cond never enters 4-row AdaLN, even at σ=1.0. */
    if (h3_dit_w4a8_visual_cond_adaln_active(0, 1.0f) ||
        h3_dit_w4a8_visual_cond_mod_row(0, 1.0f) !=
            (uint32_t)H3_DIT_W4A8_MOD_VIDEO) {
        snprintf(err, es, "FAIL: empty cond must not enter 4-row AdaLN");
        return 0;
    }
    got = h3_dit_w4a8_visual_cond_t_row(table_rows);
    want_time = h3_dit_w4a8_t_row_from_sigma(1.0f - 0.999f, table_rows);
    noisy_flow = h3_dit_w4a8_t_row_from_sigma(0.999f, table_rows);
    if (got != want_time) {
        snprintf(err, es,
                 "FAIL: cond t_row=%zu want time=0.999 row=%zu (not flow-σ 0.999 "
                 "row=%zu)",
                 got, want_time, noisy_flow);
        return 0;
    }
    if (got == noisy_flow) {
        snprintf(err, es,
                 "FAIL: cond t_row used noisy flow-σ=0.999, want time=0.999");
        return 0;
    }
    puts("ok: AdaLN 0.999 polarity (host, Metal time axis)");
    return 1;
}

static int test_avc_empty_matches_av(h3_dit_w4a8 *dit,
                                     const h3_dit_w4a8_core *core,
                                     const float *xv, const float *xa,
                                     const float *th, const float *pt,
                                     const float *ph, const float *pw,
                                     const float *at, const float *ah,
                                     const float *aw, char *err, size_t es) {
    /* Limitation: forward_av is already forward_avc(NULL,0). This memcmp
     * cannot prove vs the pre-wrapper body. Load-bearing check is empty cond
     * at σ=1.0 still reports visual_cond_t_active=0 (no 4-row AdaLN). */
    float *vv_av = calloc(VIDEO_SEQ * VIDEO_PATCH, sizeof(float));
    float *va_av = calloc(AUDIO_SEQ * AUDIO_CH, sizeof(float));
    float *vv_avc = calloc(VIDEO_SEQ * VIDEO_PATCH, sizeof(float));
    float *va_avc = calloc(AUDIO_SEQ * AUDIO_CH, sizeof(float));
    h3_dit_w4a8_forward_stats st;
    int ok = 0;
    if (!vv_av || !va_av || !vv_avc || !va_avc) {
        snprintf(err, es, "OOM avc empty");
        goto done;
    }
    if (!h3_dit_w4a8_forward_av_f32(
            dit, core, 0, N_BLOCKS, 1.0f, 0.3f, xv, VIDEO_SEQ, xa, AUDIO_SEQ, th,
            TEXT_SEQ, pt, ph, pw, at, ah, aw, 1.0f, vv_av, va_av, 1, &st, err,
            es))
        goto done;
    if (st.visual_cond_t_active ||
        st.visual_cond_mod_row != (uint32_t)H3_DIT_W4A8_MOD_VIDEO) {
        snprintf(err, es,
                 "FAIL: empty cond at sigma=1.0 entered 4-row AdaLN "
                 "(active=%d mod=%u)",
                 st.visual_cond_t_active, st.visual_cond_mod_row);
        goto done;
    }
    if (!h3_dit_w4a8_forward_avc_f32(
            dit, core, 0, N_BLOCKS, 1.0f, 0.3f, xv, VIDEO_SEQ, xa, AUDIO_SEQ,
            NULL, 0, NULL, 0, th, TEXT_SEQ, pt, ph, pw, at, ah, aw, NULL, NULL,
            NULL, NULL, NULL, NULL, 1.0f, vv_avc, va_avc, 1, &st, err, es))
        goto done;
    if (st.visual_cond_t_active) {
        snprintf(err, es, "FAIL: avc cond_seq=0 visual_cond_t_active=%d",
                 st.visual_cond_t_active);
        goto done;
    }
    if (!same_bits(vv_av, vv_avc, VIDEO_SEQ * VIDEO_PATCH) ||
        !same_bits(va_av, va_avc, AUDIO_SEQ * AUDIO_CH)) {
        snprintf(err, es, "FAIL: cond_seq=0 avc != forward_av (not bit-for-bit)");
        goto done;
    }
    ok = 1;
    puts("ok: forward_avc cond_seq=0 wrapper-identical; 3-row AdaLN at sigma=1");
    fprintf(stderr,
            "note: empty-cond memcmp is wrapper tautology vs pre-B body; "
            "3-row AdaLN at sigma=1.0 is the load-bearing check\n");
done:
    free(vv_av);
    free(va_av);
    free(vv_avc);
    free(va_avc);
    return ok;
}

static int test_avc_cond_keeps_target_size(h3_dit_w4a8 *dit,
                                           const h3_dit_w4a8_core *core,
                                           const float *xv, const float *xa,
                                           const float *th, const float *pt,
                                           const float *ph, const float *pw,
                                           const float *at, const float *ah,
                                           const float *aw, char *err,
                                           size_t es) {
    float *cond = calloc(COND_SEQ * VIDEO_PATCH, sizeof(float));
    float *ct = calloc(COND_SEQ, sizeof(float));
    float *ch = calloc(COND_SEQ, sizeof(float));
    float *cw = calloc(COND_SEQ, sizeof(float));
    float *vv1 = calloc(VIDEO_SEQ * VIDEO_PATCH, sizeof(float));
    float *va1 = calloc(AUDIO_SEQ * AUDIO_CH, sizeof(float));
    float *vv2 = calloc(VIDEO_SEQ * VIDEO_PATCH, sizeof(float));
    float *va2 = calloc(AUDIO_SEQ * AUDIO_CH, sizeof(float));
    size_t i;
    int ok = 0;
    if (!cond || !ct || !ch || !cw || !vv1 || !va1 || !vv2 || !va2) {
        snprintf(err, es, "OOM avc cond");
        goto done;
    }
    for (i = 0; i < COND_SEQ * VIDEO_PATCH; i++) cond[i] = 0.1f;
    for (i = 0; i < COND_SEQ; i++) {
        ct[i] = (float)TEXT_SEQ; /* fake first-frame condition_time */
        ch[i] = (float)(i / 4);
        cw[i] = (float)(i % 4);
    }
    if (!h3_dit_w4a8_forward_avc_f32(
            dit, core, 0, N_BLOCKS, 0.5f, 0.3f, xv, VIDEO_SEQ, xa, AUDIO_SEQ,
            cond, COND_SEQ, NULL, 0, th, TEXT_SEQ, pt, ph, pw, at, ah, aw, ct,
            ch, cw, NULL, NULL, NULL, 1.0f, vv1, va1, 1, NULL, err, es))
        goto done;
    if (!all_finite(vv1, VIDEO_SEQ * VIDEO_PATCH) ||
        !all_finite(va1, AUDIO_SEQ * AUDIO_CH)) {
        snprintf(err, es, "FAIL: cond_seq>0 velocity not finite");
        goto done;
    }
    if (!h3_dit_w4a8_forward_avc_f32(
            dit, core, 0, N_BLOCKS, 0.5f, 0.3f, xv, VIDEO_SEQ, xa, AUDIO_SEQ,
            cond, COND_SEQ, NULL, 0, th, TEXT_SEQ, pt, ph, pw, at, ah, aw, ct,
            ch, cw, NULL, NULL, NULL, 1.0f, vv2, va2, 1, NULL, err, es))
        goto done;
    if (!same_bits(vv1, vv2, VIDEO_SEQ * VIDEO_PATCH) ||
        !same_bits(va1, va2, AUDIO_SEQ * AUDIO_CH)) {
        snprintf(err, es, "FAIL: cond_seq>0 not self-consistent");
        goto done;
    }
    ok = 1;
    puts("ok: forward_avc cond_seq>0 finite, target-sized, self-consistent");
done:
    free(cond);
    free(ct);
    free(ch);
    free(cw);
    free(vv1);
    free(va1);
    free(vv2);
    free(va2);
    return ok;
}

static int test_avc_adaln_0p999_polarity(h3_dit_w4a8 *dit,
                                         const h3_dit_w4a8_core *core,
                                         const float *xv, const float *xa,
                                         const float *th, const float *pt,
                                         const float *ph, const float *pw,
                                         const float *at, const float *ah,
                                         const float *aw, char *err,
                                         size_t es) {
    float *cond = calloc(COND_SEQ * VIDEO_PATCH, sizeof(float));
    float *ct = calloc(COND_SEQ, sizeof(float));
    float *ch = calloc(COND_SEQ, sizeof(float));
    float *cw = calloc(COND_SEQ, sizeof(float));
    float *vv = calloc(VIDEO_SEQ * VIDEO_PATCH, sizeof(float));
    float *va = calloc(AUDIO_SEQ * AUDIO_CH, sizeof(float));
    h3_dit_w4a8_forward_stats st;
    size_t i, want_row;
    int ok = 0;
    if (!cond || !ct || !ch || !cw || !vv || !va) {
        snprintf(err, es, "OOM avc adaln polarity");
        goto done;
    }
    for (i = 0; i < COND_SEQ * VIDEO_PATCH; i++) cond[i] = 0.1f;
    for (i = 0; i < COND_SEQ; i++) {
        ct[i] = (float)TEXT_SEQ;
        ch[i] = (float)(i / 4);
        cw[i] = (float)(i % 4);
    }
    want_row = h3_dit_w4a8_t_row_from_sigma(1.0f - 0.999f, core->t_table_rows);
    if (!h3_dit_w4a8_forward_avc_f32(
            dit, core, 0, N_BLOCKS, 1.0f, 0.3f, xv, VIDEO_SEQ, xa, AUDIO_SEQ,
            cond, COND_SEQ, NULL, 0, th, TEXT_SEQ, pt, ph, pw, at, ah, aw, ct,
            ch, cw, NULL, NULL, NULL, 1.0f, vv, va, 1, &st, err, es))
        goto done;
    if (!st.visual_cond_t_active || st.visual_cond_mod_row != 3u) {
        snprintf(err, es,
                 "FAIL: sigma=1.0 want row_map=3 / VIDEO@0.999 "
                 "(active=%d mod=%u)",
                 st.visual_cond_t_active, st.visual_cond_mod_row);
        goto done;
    }
    if (st.visual_cond_t_row != want_row) {
        snprintf(err, es, "FAIL: sigma=1.0 cond t_row=%zu want %zu",
                 st.visual_cond_t_row, want_row);
        goto done;
    }
    if (!h3_dit_w4a8_forward_avc_f32(
            dit, core, 0, N_BLOCKS, 0.001f, 0.3f, xv, VIDEO_SEQ, xa, AUDIO_SEQ,
            cond, COND_SEQ, NULL, 0, th, TEXT_SEQ, pt, ph, pw, at, ah, aw, ct,
            ch, cw, NULL, NULL, NULL, 1.0f, vv, va, 1, &st, err, es))
        goto done;
    if (st.visual_cond_t_active ||
        st.visual_cond_mod_row != (uint32_t)H3_DIT_W4A8_MOD_VIDEO) {
        snprintf(err, es,
                 "FAIL: sigma=0.001 want fallback MOD_VIDEO (active=%d mod=%u)",
                 st.visual_cond_t_active, st.visual_cond_mod_row);
        goto done;
    }
    ok = 1;
    puts("ok: AdaLN 0.999 polarity (sigma=1 row_map=3; sigma<=0.001 MOD_VIDEO)");
done:
    free(cond);
    free(ct);
    free(ch);
    free(cw);
    free(vv);
    free(va);
    return ok;
}

int main(int argc, char **argv) {
    const char *dit_path = argc > 1 ? argv[1] : NULL;
    char err[512];
    h3_cuda *cuda = NULL;
    h3_dit_int8_store *store = NULL;
    h3_dit_w4a8 *dit = NULL;
    h3_dit_w4a8_core core;
    h3_w4a8_cache_config cfg;
    float *xv, *xa, *th, *pt, *ph, *pw, *at, *ah, *aw;
    size_t i;
    int ok = 0;
    err[0] = '\0';
    if (!test_reference_image_canvas(err, sizeof(err))) {
        fprintf(stderr, "error: %s\n", err);
        return 1;
    }
    if (!test_fill_metal_cond_pos(err, sizeof(err))) {
        fprintf(stderr, "error: %s\n", err);
        return 1;
    }
    if (!test_video_span_sum_last_cond(err, sizeof(err))) {
        fprintf(stderr, "error: %s\n", err);
        return 1;
    }
    if (!test_ref_image_rope_cursor(err, sizeof(err))) {
        fprintf(stderr, "error: %s\n", err);
        return 1;
    }
    if (!test_adaln_0p999_polarity_host(err, sizeof(err))) {
        fprintf(stderr, "error: %s\n", err);
        return 1;
    }
    unsetenv("H3_DIT_ACT_F16");
    if (!h3_dit_w4a8_act_f16()) {
        fprintf(stderr, "error: act_f16 default should be on (AdaLN prescale)\n");
        return 1;
    }
    setenv("H3_DIT_ACT_F16", "0", 1);
    if (h3_dit_w4a8_act_f16()) {
        fprintf(stderr, "error: H3_DIT_ACT_F16=0 should disable\n");
        return 1;
    }
    setenv("H3_DIT_ACT_F16", "1", 1);
    if (!h3_dit_w4a8_act_f16()) {
        fprintf(stderr, "error: H3_DIT_ACT_F16=1 should enable\n");
        return 1;
    }
    unsetenv("H3_DIT_ACT_F16");
    puts("ok: H3_DIT_ACT_F16 default on / 0 disables");
    unsetenv("H3_DIT_FUSED_EPILOGUE");
    if (!h3_dit_w4a8_fused_epilogue()) {
        fprintf(stderr, "error: fused epi default should be on\n");
        return 1;
    }
    setenv("H3_DIT_FUSED_EPILOGUE", "0", 1);
    if (h3_dit_w4a8_fused_epilogue()) {
        fprintf(stderr, "error: H3_DIT_FUSED_EPILOGUE=0 should disable\n");
        return 1;
    }
    setenv("H3_DIT_FUSED_EPILOGUE", "1", 1);
    if (!h3_dit_w4a8_fused_epilogue()) {
        fprintf(stderr, "error: H3_DIT_FUSED_EPILOGUE=1 should enable\n");
        return 1;
    }
    unsetenv("H3_DIT_FUSED_EPILOGUE");
    puts("ok: H3_DIT_FUSED_EPILOGUE default on / 0 disables");
    unsetenv("H3_DIT_LORA_F16");
    if (h3_dit_w4a8_lora_f16()) {
        fprintf(stderr, "error: lora f16 default should be off (mixed n=64)\n");
        return 1;
    }
    setenv("H3_DIT_LORA_F16", "1", 1);
    if (!h3_dit_w4a8_lora_f16()) {
        fprintf(stderr, "error: H3_DIT_LORA_F16=1 should enable\n");
        return 1;
    }
    unsetenv("H3_DIT_LORA_F16");
    puts("ok: H3_DIT_LORA_F16 default off / 1 enables");
    unsetenv("H3_DIT_LORA_TF32");
    if (h3_dit_w4a8_lora_tf32()) {
        fprintf(stderr, "error: lora tf32 default should be off (clip1 NaN)\n");
        return 1;
    }
    setenv("H3_DIT_LORA_TF32", "1", 1);
    if (!h3_dit_w4a8_lora_tf32()) {
        fprintf(stderr, "error: H3_DIT_LORA_TF32=1 should enable\n");
        return 1;
    }
    unsetenv("H3_DIT_LORA_TF32");
    puts("ok: H3_DIT_LORA_TF32 default off / 1 enables");
    unsetenv("H3_DIT_LORA_OVERLAP");
    if (h3_dit_w4a8_lora_overlap()) {
        fprintf(stderr, "error: lora overlap default should be off (hot waffle)\n");
        return 1;
    }
    setenv("H3_DIT_LORA_OVERLAP", "1", 1);
    if (!h3_dit_w4a8_lora_overlap()) {
        fprintf(stderr, "error: H3_DIT_LORA_OVERLAP=1 should enable\n");
        return 1;
    }
    unsetenv("H3_DIT_LORA_OVERLAP");
    puts("ok: H3_DIT_LORA_OVERLAP default off / 1 enables");
    if (!dit_path) {
        fprintf(stderr, "usage: %s DIT_INT8.safetensors\n", argv[0]);
        return 2;
    }
    memset(&core, 0, sizeof(core));
    h3_dit_w4a8_config_init(&cfg);
    cuda = h3_cuda_create(0, err, sizeof(err));
    store = cuda ? h3_dit_int8_store_open(dit_path, err, sizeof(err)) : NULL;
    if (!store ||
        !h3_dit_w4a8_load_core_int8(store, &core, err, sizeof(err)) ||
        !(dit = h3_dit_w4a8_open_int8(cuda, store, &cfg, err, sizeof(err)))) {
        fprintf(stderr, "%s\n", err);
        h3_dit_w4a8_core_free(&core);
        h3_dit_int8_store_free(store);
        h3_cuda_free(cuda);
        return 1;
    }
    xv = calloc(VIDEO_SEQ * VIDEO_PATCH, sizeof(float));
    xa = calloc(AUDIO_SEQ * AUDIO_CH, sizeof(float));
    th = calloc(TEXT_SEQ * HIDDEN, sizeof(float));
    pt = calloc(VIDEO_SEQ, sizeof(float));
    ph = calloc(VIDEO_SEQ, sizeof(float));
    pw = calloc(VIDEO_SEQ, sizeof(float));
    at = calloc(AUDIO_SEQ, sizeof(float));
    ah = calloc(AUDIO_SEQ, sizeof(float));
    aw = calloc(AUDIO_SEQ, sizeof(float));
    if (!xv || !xa || !th || !pt || !ph || !pw || !at || !ah || !aw)
        goto done;
    for (i = 0; i < VIDEO_SEQ * VIDEO_PATCH; i++)
        xv[i] = (float)(i % 7) * 0.01f - 0.03f;
    for (i = 0; i < AUDIO_SEQ * AUDIO_CH; i++)
        xa[i] = (float)(i % 5) * 0.01f - 0.02f;
    for (i = 0; i < TEXT_SEQ * HIDDEN; i++)
        th[i] = (float)(i % 11) * 0.005f;
    /* RoPE fills 8 of 16 video slots on purpose (2×2×2 grid). */
    if (!h3_dit_w4a8_fill_metal_video_pos(TEXT_SEQ, 2, 4, 4, pt, ph, pw, err,
                                          sizeof(err)) ||
        !h3_dit_w4a8_fill_metal_audio_pos(TEXT_SEQ, (int)(AUDIO_SEQ / 2), 4, 4,
                                          at, ah, aw, err, sizeof(err)))
        goto done;
    if (!test_av_self_consistent(dit, &core, xv, xa, th, pt, ph, pw, at, ah, aw,
                                 err, sizeof(err)))
        goto done;
    if (!test_avc_empty_matches_av(dit, &core, xv, xa, th, pt, ph, pw, at, ah,
                                   aw, err, sizeof(err)))
        goto done;
    if (!test_avc_cond_keeps_target_size(dit, &core, xv, xa, th, pt, ph, pw, at,
                                         ah, aw, err, sizeof(err)))
        goto done;
    if (!test_avc_adaln_0p999_polarity(dit, &core, xv, xa, th, pt, ph, pw, at,
                                       ah, aw, err, sizeof(err)))
        goto done;
    ok = 1;
    puts("ok: stage B avc gates");
done:
    if (!ok && err[0]) fprintf(stderr, "error: %s\n", err);
    free(xv);
    free(xa);
    free(th);
    free(pt);
    free(ph);
    free(pw);
    free(at);
    free(ah);
    free(aw);
    h3_dit_w4a8_free(dit);
    h3_dit_w4a8_core_free(&core);
    h3_dit_int8_store_free(store);
    h3_cuda_free(cuda);
    return ok ? 0 : 1;
}
