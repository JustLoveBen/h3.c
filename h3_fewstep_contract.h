#ifndef H3_FEWSTEP_CONTRACT_H
#define H3_FEWSTEP_CONTRACT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * 少步合同自检（PDMD / DMAD）：错 preset / shift / sampler / 步数时 FAIL。
 * 未设合同（NONE）则与今日行为相同。详见
 * docs/research/2026-10-26-contract-selfcheck-design.md
 *
 * 旋钮：--fewstep-contract pdmd4|pdmd2|dmad4 或 H3_FEWSTEP_CONTRACT=
 * 一键：--fewstep-preset pdmd4|pdmd2|dmad4 或 H3_FEWSTEP_PRESET=
 *       （绑 steps/cfg/shift/sampler/contract；勿与 turbo4 叠）
 * 可选：H3_FEWSTEP_STRICT=1（pdmd2 audio∉{3,6} 升 FAIL）
 *       H3_FEWSTEP_ALLOW_NVFP4=1（允许合同×NVFP4 交叉，默认拒）
 */

typedef enum {
    H3_FEWSTEP_NONE = 0,
    H3_FEWSTEP_PDMD4 = 1,
    H3_FEWSTEP_PDMD2 = 2,
    H3_FEWSTEP_DMAD4 = 3
} h3_fewstep_id;

typedef struct h3_fewstep_contract_in {
    h3_fewstep_id id;
    int steps;
    /* 1 if --preset turbo4 (or other schedule-rewriting speed preset). */
    int preset_rewrites_sigma;
    const char *preset_name; /* may be NULL */
    const char *lora_path;   /* may be NULL */
    float video_shift;
    float audio_shift;
    float cfg_scale;
    int nvfp4_enabled;
    /* sampler_is_velocity_euler=1 for default Euler path. */
    int sampler_is_velocity_euler;
    /* has_lcm_renoise=1 when --sampler lcm_renoise (DMAD). */
    int has_lcm_renoise;
    /* After schedule build: length steps+1. NULL → skip vector check. */
    const float *video_sigmas;
    const float *audio_sigmas;
} h3_fewstep_contract_in;

/* Parse CLI/env token. Unknown → NONE and fills err if provided. */
h3_fewstep_id h3_fewstep_contract_parse(const char *name, char *err,
                                        size_t err_size);

/* CLI NULL → read H3_FEWSTEP_CONTRACT. Empty/unset → NONE. */
h3_fewstep_id h3_fewstep_contract_resolve(const char *cli_name, char *err,
                                         size_t err_size);

const char *h3_fewstep_contract_name(h3_fewstep_id id);

/* After preset apply. 1 = ok (WARN may print). 0 = FAIL, err set. */
int h3_fewstep_contract_check_early(const h3_fewstep_contract_in *in, char *err,
                                    size_t err_size);

/* After sigma[] printed. Checks student_grid golden vectors. */
int h3_fewstep_contract_check_sigmas(const h3_fewstep_contract_in *in, char *err,
                                     size_t err_size);

/* Log: fewstep_contract: id=... steps=... sampler=... shifts=... */
void h3_fewstep_contract_print(const h3_fewstep_contract_in *in);

/*
 * One-shot fewstep preset (--fewstep-preset / H3_FEWSTEP_PRESET):
 * expands steps + cfg + audio shift + sampler preference + contract id.
 * Does NOT touch --preset turbo4/taomate (caller must refuse that combo).
 * Explicit steps/cfg/sampler from CLI are kept; contract check may still FAIL.
 */
typedef struct h3_fewstep_preset_in {
    const char *name; /* CLI; NULL → H3_FEWSTEP_PRESET env */
    int steps_explicit;
    int steps;
    int cfg_explicit;
    float cfg_scale;
    int sampler_explicit;
    /* 1 if user/env already named --fewstep-contract / H3_FEWSTEP_CONTRACT */
    int contract_already_set;
    h3_fewstep_id contract_already;
} h3_fewstep_preset_in;

typedef struct h3_fewstep_preset_out {
    int applied; /* 1 if a named preset expanded */
    h3_fewstep_id id;
    int steps;
    float cfg_scale;
    float video_shift;
    float audio_shift;
    int want_lcm_renoise; /* 1 → dmad4 */
    /* 1 → caller should setenv(H3_AUDIO_SIGMA_SHIFT, audio_env_value, 0) */
    int set_audio_env;
    const char *audio_env_value; /* static "2" or "3"; may be NULL */
} h3_fewstep_preset_out;

/* 1 = ok. applied=0 when name/env empty. Unknown name → 0 + err. */
int h3_fewstep_preset_apply(const h3_fewstep_preset_in *in,
                            h3_fewstep_preset_out *out, char *err,
                            size_t err_size);

void h3_fewstep_preset_print(const h3_fewstep_preset_out *out);

/*
 * After --fewstep-preset expands: if H3_AUDIO_SIGMA_SHIFT is already set to a
 * value that differs from want_audio, FAIL with a clear conflict message.
 * Unset or matching (eps 0.051) → ok. Call before setenv(..., 0) so a stale
 * env cannot silently keep the wrong audio shift under a named preset.
 */
int h3_fewstep_preset_audio_env_compatible(float want_audio, char *err,
                                           size_t err_size);

#ifdef __cplusplus
}
#endif

#endif
