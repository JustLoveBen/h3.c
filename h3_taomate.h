#ifndef H3_TAOMATE_H
#define H3_TAOMATE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * TaoMate-H3 distilled schedule + codec-chunk geometry (opt-in, 非金线).
 * Anchors are indices into linspace(1,0,50) then video/audio time-shift.
 * Source: third_party/accel_survey/TaoMate-H3/src/taomate_h3/denoise_schedule.py
 *         DISTILLED_STATE_INDICES = (0, 16, 33, 49)
 */

enum {
    H3_TAOMATE_BASE_STEPS = 50,
    H3_TAOMATE_N_ANCHORS = 4, /* 3 Euler intervals + terminal 0 */
    H3_TAOMATE_NFE = 3,
    H3_TAOMATE_QUALITY_NFE = 6, /* distilled anchors + midpoints */
    H3_TAOMATE_QUALITY_N_SIGMAS = 7,
    H3_TAOMATE_MAX_PHASES = 16,
    H3_TAOMATE_VIDEO_FPS = 24,
    H3_TAOMATE_AUDIO_LATENT_RATE = 40,
    H3_TAOMATE_PREFIX_FRAMES = 5,
    H3_TAOMATE_PREFIX_LATENTS = 2,
    H3_TAOMATE_GROUP_FRAMES = 17,
    H3_TAOMATE_GROUP_LATENTS = 5
};

extern const int H3_TAOMATE_STATE_INDICES[H3_TAOMATE_N_ANCHORS];
extern const int H3_TAOMATE_QUALITY_INDICES[H3_TAOMATE_QUALITY_N_SIGMAS];

typedef struct h3_taomate_phase {
    int index;
    int group_count;
    int frame_start;
    int frame_stop;
    int video_t0;
    int video_t1; /* exclusive */
    int audio_t0;
    int audio_t1; /* exclusive */
} h3_taomate_phase;

typedef struct h3_taomate_plan {
    int native_frames;
    int video_t;
    int audio_t;
    int n_phases;
    h3_taomate_phase phases[H3_TAOMATE_MAX_PHASES];
} h3_taomate_plan;

enum { H3_TAOMATE_MAX_OV_SPANS = 2 };

/* Bounded cond-token overlap: first-chunk sink prefix + previous-chunk suffix.
 * Extra video T is at most sink_t+hist_t (not full-clip Q / not full-chunk KV). */
typedef struct h3_taomate_ov_span {
    int video_t0;
    int video_t1; /* exclusive */
    int audio_t0;
    int audio_t1; /* exclusive */
} h3_taomate_ov_span;

/* linspace(1,0,num_steps)[index] then shift * b / (1+(shift-1)*b). */
int h3_taomate_base_sigma(int index, int num_steps, float shift, float *out);

/* 4 video or audio sigmas at DISTILLED_STATE_INDICES. Terminal is 0. */
int h3_taomate_distilled_sigmas(float shift, float *sigmas /*[4]*/);
/* nfe=3 distilled; nfe=6 inserts midpoints {0,8,16,24,33,41,49}. sigmas[nfe+1]. */
int h3_taomate_nfe_sigmas(int nfe, float shift, float *sigmas);

/* round_half_even(frame * 40 / 24); matches streaming/geometry.py. */
int h3_taomate_audio_latent_boundary(int frame);

/* Streaming codec chunks: 5s native 124 → 4 phases (12,10,10,5).
 * 10s 243 → 8 phases. Generate uses oneshot instead for duration≤5s. */
int h3_taomate_plan_build(int native_frames, h3_taomate_plan *out, char *err,
                          size_t err_size);

/* Comfy dense 5s: one phase covering the whole 124-frame / 37-T clip.
 * native_frames must be 124. Totals match plan_build(124). */
int h3_taomate_plan_build_oneshot(int native_frames, h3_taomate_plan *out,
                                  char *err, size_t err_size);

/* Inclusive start of the denoise window (phase t0 minus ov, not before prev). */
int h3_taomate_denoise_t0(const h3_taomate_plan *plan, int phase_i, int ov);
int h3_taomate_denoise_a0(const h3_taomate_plan *plan, int phase_i, int ov);

/* phase_i==0 or hist=sink=0 → 0. Out holds up to H3_TAOMATE_MAX_OV_SPANS spans. */
int h3_taomate_overlap_spans(const h3_taomate_plan *plan, int phase_i,
                             int hist_t, int sink_t,
                             h3_taomate_ov_span *out);

/* Resident layer-KV token budget: sink_t + hist_t video rows + bounded audio.
 * Not full-chunk / not full-clip. */
size_t h3_taomate_layer_kv_budget(size_t spatial, int hist_t, int sink_t,
                                  int pack_audio);

/* Packed-row indices in the current chunk [text|audio|video] to stage.
 * phase 0: sink prefix + hist suffix. later: hist suffix only (sink stays).
 * *n_sink is how many leading indices are the identity sink. */
size_t h3_taomate_layer_kv_save_idx(const h3_taomate_plan *plan, int phase_i,
                                    int hist_t, int sink_t, size_t audio_off,
                                    size_t video_off, size_t spatial,
                                    int pack_audio, int audio_win,
                                    uint32_t *idx, size_t cap, size_t *n_sink);

#ifdef __cplusplus
}
#endif

#endif
