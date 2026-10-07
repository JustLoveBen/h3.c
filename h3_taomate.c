#include "h3_taomate.h"

#include <stdio.h>
#include <string.h>

const int H3_TAOMATE_STATE_INDICES[H3_TAOMATE_N_ANCHORS] = {0, 16, 33, 49};
const int H3_TAOMATE_QUALITY_INDICES[H3_TAOMATE_QUALITY_N_SIGMAS] = {
    0, 8, 16, 24, 33, 41, 49};

static int fail_msg(char *err, size_t err_size, const char *msg) {
    if (err && err_size)
        snprintf(err, err_size, "%s", msg ? msg : "taomate error");
    return 0;
}

int h3_taomate_base_sigma(int index, int num_steps, float shift, float *out) {
    float base, denom;
    if (!out || num_steps < 2 || index < 0 || index >= num_steps ||
        !(shift > 0.0f))
        return 0;
    /* torch.linspace(1.0, 0.0, num_steps) */
    base = 1.0f - (float)index / (float)(num_steps - 1);
    denom = 1.0f + (shift - 1.0f) * base;
    if (!(denom > 0.0f))
        return 0;
    *out = shift * base / denom;
    return 1;
}

int h3_taomate_nfe_sigmas(int nfe, float shift, float *sigmas) {
    const int *idx;
    int n, i;
    if (!sigmas)
        return 0;
    if (nfe == H3_TAOMATE_NFE) {
        idx = H3_TAOMATE_STATE_INDICES;
        n = H3_TAOMATE_N_ANCHORS;
    } else if (nfe == H3_TAOMATE_QUALITY_NFE) {
        idx = H3_TAOMATE_QUALITY_INDICES;
        n = H3_TAOMATE_QUALITY_N_SIGMAS;
    } else
        return 0;
    for (i = 0; i < n; i++) {
        if (!h3_taomate_base_sigma(idx[i], H3_TAOMATE_BASE_STEPS, shift,
                                   &sigmas[i]))
            return 0;
    }
    sigmas[n - 1] = 0.0f;
    return 1;
}

int h3_taomate_distilled_sigmas(float shift, float *sigmas) {
    return h3_taomate_nfe_sigmas(H3_TAOMATE_NFE, shift, sigmas);
}

int h3_taomate_audio_latent_boundary(int frame) {
    /* round_half_even(frame * 40 / 24) = round_half_even(frame * 5 / 3) */
    long num, den, q, r, doubled;
    if (frame <= 0)
        return 0;
    num = (long)frame * 5L;
    den = 3L;
    q = num / den;
    r = num % den;
    if (r < 0) {
        q--;
        r += den;
    }
    doubled = r * 2L;
    if (doubled < den)
        return (int)q;
    if (doubled > den)
        return (int)q + 1;
    return (int)q + (int)(q & 1L);
}

static int append_request(h3_taomate_plan *out, int request_index,
                          int frame_origin, int video_origin) {
    const int groups[4] = {2, 2, 2, 1};
    int i;
    int frame_stop = 0;
    int video_stop = 0;

    for (i = 0; i < 4; i++) {
        h3_taomate_phase *ph;
        int frame_start = frame_stop;
        int video_start = video_stop;
        if (out->n_phases >= H3_TAOMATE_MAX_PHASES)
            return 0;
        frame_stop += groups[i] * H3_TAOMATE_GROUP_FRAMES;
        video_stop += groups[i] * H3_TAOMATE_GROUP_LATENTS;
        if (request_index == 0 && i == 0) {
            frame_stop += H3_TAOMATE_PREFIX_FRAMES;
            video_stop += H3_TAOMATE_PREFIX_LATENTS;
        }
        ph = &out->phases[out->n_phases++];
        ph->index = out->n_phases - 1;
        ph->group_count = groups[i];
        ph->frame_start = frame_origin + frame_start;
        ph->frame_stop = frame_origin + frame_stop;
        ph->video_t0 = video_origin + video_start;
        ph->video_t1 = video_origin + video_stop;
        /* Global audio indices into one concatenated clip (C generate). */
        ph->audio_t0 = h3_taomate_audio_latent_boundary(ph->frame_start);
        ph->audio_t1 = h3_taomate_audio_latent_boundary(ph->frame_stop);
    }
    out->native_frames = frame_origin + frame_stop;
    out->video_t = video_origin + video_stop;
    out->audio_t = h3_taomate_audio_latent_boundary(out->native_frames);
    return 1;
}

int h3_taomate_plan_build(int native_frames, h3_taomate_plan *out, char *err,
                          size_t err_size) {
    int remaining;
    int req;
    int frame_origin;
    int video_origin;

    if (err && err_size)
        err[0] = '\0';
    if (!out)
        return fail_msg(err, err_size, "taomate plan: null");
    memset(out, 0, sizeof(*out));
    if (native_frames < 124)
        return fail_msg(err, err_size, "taomate plan: need native 5s (124 frames)");
    remaining = native_frames;
    req = 0;
    frame_origin = 0;
    video_origin = 0;
    while (remaining > 0) {
        int chunk_frames = (req == 0) ? 124 : (124 - H3_TAOMATE_PREFIX_FRAMES);
        if (remaining < chunk_frames)
            return fail_msg(err, err_size,
                            "taomate plan: frames not 124+119k");
        if (!append_request(out, req, frame_origin, video_origin))
            return fail_msg(err, err_size, "taomate plan: too many phases");
        remaining -= chunk_frames;
        frame_origin = out->native_frames;
        video_origin = out->video_t;
        req++;
        if (req > 8)
            return fail_msg(err, err_size, "taomate plan: >8 requests");
    }
    if (out->native_frames != native_frames)
        return fail_msg(err, err_size, "taomate plan: frame mismatch");
    return 1;
}

int h3_taomate_plan_build_oneshot(int native_frames, h3_taomate_plan *out,
                                  char *err, size_t err_size) {
    h3_taomate_phase *ph;
    if (err && err_size)
        err[0] = '\0';
    if (!out)
        return fail_msg(err, err_size, "taomate oneshot: null");
    memset(out, 0, sizeof(*out));
    if (native_frames != 124)
        return fail_msg(err, err_size,
                        "taomate oneshot: only native 5s (124 frames)");
    ph = &out->phases[0];
    ph->index = 0;
    ph->group_count = 7; /* 2+2+2+1 codec groups, one denoise */
    ph->frame_start = 0;
    ph->frame_stop = 124;
    ph->video_t0 = 0;
    ph->video_t1 = 37;
    ph->audio_t0 = 0;
    ph->audio_t1 = h3_taomate_audio_latent_boundary(124);
    out->n_phases = 1;
    out->native_frames = 124;
    out->video_t = 37;
    out->audio_t = ph->audio_t1;
    return 1;
}

static void taomate_map_audio_span(const h3_taomate_phase *ph, int v0, int v1,
                                   int *a0, int *a1) {
    int vt, at;
    if (!ph || !a0 || !a1) {
        if (a0)
            *a0 = 0;
        if (a1)
            *a1 = 0;
        return;
    }
    vt = ph->video_t1 - ph->video_t0;
    at = ph->audio_t1 - ph->audio_t0;
    if (vt <= 0 || at <= 0) {
        *a0 = ph->audio_t0;
        *a1 = ph->audio_t0;
        return;
    }
    *a0 = ph->audio_t0 + (v0 - ph->video_t0) * at / vt;
    *a1 = ph->audio_t0 + (v1 - ph->video_t0) * at / vt;
    if (*a0 < ph->audio_t0)
        *a0 = ph->audio_t0;
    if (*a1 > ph->audio_t1)
        *a1 = ph->audio_t1;
    if (*a1 < *a0)
        *a1 = *a0;
}

int h3_taomate_denoise_t0(const h3_taomate_plan *plan, int phase_i, int ov) {
    int t0;
    if (!plan || phase_i < 0 || phase_i >= plan->n_phases)
        return 0;
    t0 = plan->phases[phase_i].video_t0;
    if (phase_i == 0 || ov <= 0)
        return t0;
    t0 -= ov;
    if (t0 < plan->phases[phase_i - 1].video_t0)
        t0 = plan->phases[phase_i - 1].video_t0;
    if (t0 < 0)
        t0 = 0;
    return t0;
}

int h3_taomate_denoise_a0(const h3_taomate_plan *plan, int phase_i, int ov) {
    int t0_ov, v_ov, a_ov, prev_vt, prev_at, a0;
    if (!plan || phase_i < 0 || phase_i >= plan->n_phases)
        return 0;
    a0 = plan->phases[phase_i].audio_t0;
    if (phase_i == 0 || ov <= 0)
        return a0;
    t0_ov = h3_taomate_denoise_t0(plan, phase_i, ov);
    v_ov = plan->phases[phase_i].video_t0 - t0_ov;
    prev_vt = plan->phases[phase_i - 1].video_t1 -
              plan->phases[phase_i - 1].video_t0;
    prev_at = plan->phases[phase_i - 1].audio_t1 -
              plan->phases[phase_i - 1].audio_t0;
    a_ov = 0;
    if (v_ov > 0 && prev_vt > 0)
        a_ov = (v_ov * prev_at) / prev_vt;
    if (a_ov < 1 && v_ov > 0)
        a_ov = 1;
    a0 -= a_ov;
    if (a0 < plan->phases[phase_i - 1].audio_t0)
        a0 = plan->phases[phase_i - 1].audio_t0;
    if (a0 < 0)
        a0 = 0;
    return a0;
}

int h3_taomate_overlap_spans(const h3_taomate_plan *plan, int phase_i,
                             int hist_t, int sink_t,
                             h3_taomate_ov_span *out) {
    int n = 0;
    const h3_taomate_phase *p0;
    const h3_taomate_phase *prev;
    if (!plan || !out || phase_i <= 0 || phase_i >= plan->n_phases)
        return 0;
    if (hist_t < 0)
        hist_t = 0;
    if (sink_t < 0)
        sink_t = 0;
    p0 = &plan->phases[0];
    prev = &plan->phases[phase_i - 1];
    if (sink_t > 0) {
        int vt = p0->video_t1 - p0->video_t0;
        int st = sink_t < vt ? sink_t : vt;
        if (st > 0) {
            out[n].video_t0 = p0->video_t0;
            out[n].video_t1 = p0->video_t0 + st;
            taomate_map_audio_span(p0, out[n].video_t0, out[n].video_t1,
                                   &out[n].audio_t0, &out[n].audio_t1);
            n++;
        }
    }
    if (hist_t > 0) {
        int vt = prev->video_t1 - prev->video_t0;
        int ht = hist_t < vt ? hist_t : vt;
        int v0 = prev->video_t1 - ht;
        int v1 = prev->video_t1;
        if (n > 0 && prev == p0 && v0 < out[0].video_t1)
            v0 = out[0].video_t1;
        if (v1 > v0) {
            out[n].video_t0 = v0;
            out[n].video_t1 = v1;
            taomate_map_audio_span(prev, v0, v1, &out[n].audio_t0,
                                   &out[n].audio_t1);
            n++;
        }
    }
    return n;
}

size_t h3_taomate_layer_kv_budget(size_t spatial, int hist_t, int sink_t,
                                  int pack_audio) {
    int vt;
    size_t audio;
    if (hist_t < 0)
        hist_t = 0;
    if (sink_t < 0)
        sink_t = 0;
    vt = hist_t + sink_t;
    /* ~5.6 audio latents/video T × 2 channels, plus slack. */
    audio = pack_audio ? (size_t)vt * 16u + 32u : 0;
    return (size_t)vt * spatial + audio;
}

static size_t kv_push_video(uint32_t *idx, size_t n, size_t cap, size_t video_off,
                            size_t spatial, int t_local, int t_n) {
    int t, s;
    if (t_n <= 0)
        return n;
    for (t = 0; t < t_n; t++) {
        for (s = 0; s < (int)spatial; s++) {
            if (n >= cap)
                return n;
            idx[n++] = (uint32_t)(video_off + (size_t)(t_local + t) * spatial +
                                  (size_t)s);
        }
    }
    return n;
}

static size_t kv_push_audio(uint32_t *idx, size_t n, size_t cap, size_t audio_off,
                            int at_win, int a_local, int a_n) {
    int t, ch;
    if (a_n <= 0 || at_win <= 0)
        return n;
    for (ch = 0; ch < 2; ch++) {
        for (t = 0; t < a_n; t++) {
            int loc = a_local + t;
            if (loc < 0 || loc >= at_win)
                continue;
            if (n >= cap)
                return n;
            idx[n++] = (uint32_t)(audio_off + (size_t)ch * (size_t)at_win +
                                  (size_t)loc);
        }
    }
    return n;
}

size_t h3_taomate_layer_kv_save_idx(const h3_taomate_plan *plan, int phase_i,
                                    int hist_t, int sink_t, size_t audio_off,
                                    size_t video_off, size_t spatial,
                                    int pack_audio, int audio_win,
                                    uint32_t *idx, size_t cap, size_t *n_sink) {
    const h3_taomate_phase *ph;
    size_t n = 0, sink_n = 0;
    int vt, ht, st;
    if (n_sink)
        *n_sink = 0;
    if (!plan || !idx || phase_i < 0 || phase_i >= plan->n_phases || cap == 0)
        return 0;
    if (hist_t < 0)
        hist_t = 0;
    if (sink_t < 0)
        sink_t = 0;
    ph = &plan->phases[phase_i];
    vt = ph->video_t1 - ph->video_t0;
    if (vt <= 0)
        return 0;
    if (phase_i == 0 && sink_t > 0) {
        int a0 = 0, a1 = 0;
        st = sink_t < vt ? sink_t : vt;
        n = kv_push_video(idx, n, cap, video_off, spatial, 0, st);
        taomate_map_audio_span(ph, ph->video_t0, ph->video_t0 + st, &a0, &a1);
        if (pack_audio)
            n = kv_push_audio(idx, n, cap, audio_off, audio_win, a0 - ph->audio_t0,
                              a1 - a0);
        sink_n = n;
    }
    ht = hist_t < vt ? hist_t : vt;
    if (ht > 0) {
        int t0 = vt - ht;
        int a0 = 0, a1 = 0;
        if (phase_i == 0 && sink_n > 0 && t0 < sink_t)
            t0 = sink_t < vt ? sink_t : vt;
        if (t0 < vt) {
            n = kv_push_video(idx, n, cap, video_off, spatial, t0, vt - t0);
            taomate_map_audio_span(ph, ph->video_t0 + t0, ph->video_t1, &a0, &a1);
            if (pack_audio)
                n = kv_push_audio(idx, n, cap, audio_off, audio_win,
                                  a0 - ph->audio_t0, a1 - a0);
        }
    }
    if (n_sink)
        *n_sink = sink_n;
    return n;
}
