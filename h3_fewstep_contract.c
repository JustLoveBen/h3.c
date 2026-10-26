#include "h3_fewstep_contract.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* student_grid(nfe, shift): linspace(1,0,nfe+1) then SD3 shift s*t/(1+(s-1)*t).
 * Golden values for formal PDMD/DMAD acceptance (design §2–3). */

static float shifted_sigma(float base, float shift) {
    if (!(base > 0.0f))
        return 0.0f;
    return shift * base / (1.0f + (shift - 1.0f) * base);
}

static int env_truthy(const char *name) {
    const char *e = getenv(name);
    return e && e[0] && e[0] != '0';
}

static int fail_msg(char *err, size_t err_size, const char *msg) {
    if (err && err_size)
        snprintf(err, err_size, "%s", msg ? msg : "fewstep contract fail");
    return 0;
}

const char *h3_fewstep_contract_name(h3_fewstep_id id) {
    switch (id) {
    case H3_FEWSTEP_PDMD4:
        return "pdmd4";
    case H3_FEWSTEP_PDMD2:
        return "pdmd2";
    case H3_FEWSTEP_DMAD4:
        return "dmad4";
    case H3_FEWSTEP_NONE:
    default:
        return "none";
    }
}

h3_fewstep_id h3_fewstep_contract_parse(const char *name, char *err,
                                        size_t err_size) {
    if (err && err_size)
        err[0] = '\0';
    if (!name || !name[0] || !strcmp(name, "none") || !strcmp(name, "off"))
        return H3_FEWSTEP_NONE;
    if (!strcmp(name, "pdmd4") || !strcmp(name, "pdmd-4") ||
        !strcmp(name, "pdmd_4nfe") || !strcmp(name, "PDMD4"))
        return H3_FEWSTEP_PDMD4;
    if (!strcmp(name, "pdmd2") || !strcmp(name, "pdmd-2") ||
        !strcmp(name, "pdmd_2nfe") || !strcmp(name, "PDMD2"))
        return H3_FEWSTEP_PDMD2;
    if (!strcmp(name, "dmad4") || !strcmp(name, "dmad-4") ||
        !strcmp(name, "dmad") || !strcmp(name, "DMAD4"))
        return H3_FEWSTEP_DMAD4;
    if (err && err_size)
        snprintf(err, err_size,
                 "unknown --fewstep-contract '%s' (pdmd4|pdmd2|dmad4|none)",
                 name);
    return H3_FEWSTEP_NONE;
}

h3_fewstep_id h3_fewstep_contract_resolve(const char *cli_name, char *err,
                                          size_t err_size) {
    const char *name = cli_name;
    char perr[192];
    h3_fewstep_id id;

    if (err && err_size)
        err[0] = '\0';
    if (!name || !name[0]) {
        name = getenv("H3_FEWSTEP_CONTRACT");
        if (!name || !name[0])
            return H3_FEWSTEP_NONE;
    }
    perr[0] = '\0';
    id = h3_fewstep_contract_parse(name, perr, sizeof(perr));
    if (id == H3_FEWSTEP_NONE && perr[0]) {
        if (err && err_size)
            snprintf(err, err_size, "%s", perr);
        /* Treat unknown token as hard resolve failure via err non-empty;
         * caller should FAIL. Return NONE. */
        return H3_FEWSTEP_NONE;
    }
    return id;
}

static int audio_shift_ok_pdmd2(float a, int *is_warn_only) {
    /* Paper metrics: 3; look-feel tier: 6. */
    const float eps = 0.051f;
    *is_warn_only = 0;
    if (fabsf(a - 3.0f) <= eps || fabsf(a - 6.0f) <= eps)
        return 1;
    *is_warn_only = 1;
    return 0;
}

static int lora_path_is_ranked_down(const char *path) {
    if (!path || !path[0])
        return 0;
    if (strstr(path, "rank_39") || strstr(path, "rank39") ||
        strstr(path, "rank_57") || strstr(path, "rank57") ||
        strstr(path, "rank_38") || strstr(path, "rank38") ||
        strstr(path, "kijai") || strstr(path, "Kijai"))
        return 1;
    return 0;
}

void h3_fewstep_contract_print(const h3_fewstep_contract_in *in) {
    const char *sampler;
    if (!in || in->id == H3_FEWSTEP_NONE)
        return;
    if (in->has_lcm_renoise)
        sampler = "lcm+re-noise";
    else if (in->sampler_is_velocity_euler)
        sampler = "velocity_euler";
    else
        sampler = "unknown";
    printf("fewstep_contract: id=%s steps=%d sampler=%s "
           "video_shift=%.3g audio_shift=%.3g cfg=%.3g preset=%s "
           "nvfp4=%d lora=%s\n",
           h3_fewstep_contract_name(in->id), in->steps, sampler,
           (double)in->video_shift, (double)in->audio_shift,
           (double)in->cfg_scale,
           in->preset_name && in->preset_name[0] ? in->preset_name : "none",
           in->nvfp4_enabled ? 1 : 0,
           in->lora_path && in->lora_path[0] ? in->lora_path : "(none)");
    fflush(stdout);
}

int h3_fewstep_contract_check_early(const h3_fewstep_contract_in *in, char *err,
                                    size_t err_size) {
    char buf[384];
    int warn_only = 0;

    if (err && err_size)
        err[0] = '\0';
    if (!in || in->id == H3_FEWSTEP_NONE)
        return 1;

    h3_fewstep_contract_print(in);

    if (in->nvfp4_enabled && !env_truthy("H3_FEWSTEP_ALLOW_NVFP4")) {
        snprintf(buf, sizeof(buf),
                 "fewstep_contract=%s 禁止与 H3_DIT_NVFP4=1 交叉验收 "
                 "（设 H3_FEWSTEP_ALLOW_NVFP4=1 才放行）",
                 h3_fewstep_contract_name(in->id));
        return fail_msg(err, err_size, buf);
    }

    if (in->preset_rewrites_sigma) {
        snprintf(buf, sizeof(buf),
                 "fewstep_contract=%s 拒绝 --preset %s（会改写 sigma/sampler）；"
                 "请无 preset + --steps + H3_AUDIO_SIGMA_SHIFT 走 serving 格点",
                 h3_fewstep_contract_name(in->id),
                 in->preset_name && in->preset_name[0] ? in->preset_name
                                                       : "turbo4");
        return fail_msg(err, err_size, buf);
    }

    switch (in->id) {
    case H3_FEWSTEP_PDMD4:
        if (in->steps != 4) {
            snprintf(buf, sizeof(buf),
                     "fewstep_contract=pdmd4 要求 steps=4（got %d）", in->steps);
            return fail_msg(err, err_size, buf);
        }
        if (fabsf(in->video_shift - 12.0f) > 0.051f) {
            snprintf(buf, sizeof(buf),
                     "fewstep_contract=pdmd4 要求 video_shift=12（got %.3g）",
                     (double)in->video_shift);
            return fail_msg(err, err_size, buf);
        }
        if (fabsf(in->audio_shift - 3.0f) > 0.051f) {
            snprintf(buf, sizeof(buf),
                     "fewstep_contract=pdmd4 要求 audio_shift=3 "
                     "（H3_AUDIO_SIGMA_SHIFT；got %.3g）",
                     (double)in->audio_shift);
            return fail_msg(err, err_size, buf);
        }
        if (!(fabsf(in->cfg_scale - 1.0f) <= 0.051f)) {
            fprintf(stderr,
                    "WARN: fewstep_contract=pdmd4 cfg=%.3g（合同 cfg=1）\n",
                    (double)in->cfg_scale);
        }
        break;

    case H3_FEWSTEP_PDMD2:
        if (in->steps != 2) {
            snprintf(buf, sizeof(buf),
                     "fewstep_contract=pdmd2 要求 steps=2（got %d）", in->steps);
            return fail_msg(err, err_size, buf);
        }
        if (fabsf(in->video_shift - 12.0f) > 0.051f) {
            snprintf(buf, sizeof(buf),
                     "fewstep_contract=pdmd2 要求 video_shift=12（got %.3g）",
                     (double)in->video_shift);
            return fail_msg(err, err_size, buf);
        }
        if (!audio_shift_ok_pdmd2(in->audio_shift, &warn_only)) {
            snprintf(buf, sizeof(buf),
                     "fewstep_contract=pdmd2 audio_shift=%.3g 不在 {3,6} "
                     "（论文指标 3；观感档 6）",
                     (double)in->audio_shift);
            if (env_truthy("H3_FEWSTEP_STRICT") || !warn_only)
                return fail_msg(err, err_size, buf);
            fprintf(stderr, "WARN: %s\n", buf);
        } else if (fabsf(in->audio_shift - 6.0f) <= 0.051f) {
            fprintf(stderr,
                    "WARN: fewstep_contract=pdmd2 audio_shift=6 为观感档 "
                    "（论文指标用 3；META non_paper_lookfeel）\n");
        }
        break;

    case H3_FEWSTEP_DMAD4:
        if (in->steps != 4) {
            snprintf(buf, sizeof(buf),
                     "fewstep_contract=dmad4 要求 steps=4（got %d）", in->steps);
            return fail_msg(err, err_size, buf);
        }
        if (fabsf(in->video_shift - 12.0f) > 0.051f) {
            snprintf(buf, sizeof(buf),
                     "fewstep_contract=dmad4 要求 video_shift=12（got %.3g）",
                     (double)in->video_shift);
            return fail_msg(err, err_size, buf);
        }
        if (fabsf(in->audio_shift - 2.0f) > 0.051f) {
            snprintf(buf, sizeof(buf),
                     "fewstep_contract=dmad4 要求 audio_shift=2 "
                     "（H3_AUDIO_SIGMA_SHIFT=2；got %.3g）",
                     (double)in->audio_shift);
            return fail_msg(err, err_size, buf);
        }
        if (!in->has_lcm_renoise) {
            snprintf(buf, sizeof(buf),
                     "fewstep_contract=dmad4 拒绝：本仓尚无 lcm+re-noise 采样器 "
                     "（当前 velocity_euler 不算 DMAD 合同通过；见 "
                     "docs/research/2026-10-09-pdmd-dmad-loras.md）");
            return fail_msg(err, err_size, buf);
        }
        break;

    default:
        break;
    }

    if (lora_path_is_ranked_down(in->lora_path)) {
        fprintf(stderr,
                "WARN: fewstep_contract=%s LoRA 路径含 rank_*/kijai "
                "（降秩非金线；标 non_gold，不挡跑）: %s\n",
                h3_fewstep_contract_name(in->id), in->lora_path);
    }

    return 1;
}

static int sigmas_match(const float *got, const float *want, int n, float tol) {
    int i;
    if (!got || !want || n < 2)
        return 0;
    for (i = 0; i < n; i++) {
        if (!isfinite(got[i]) || !isfinite(want[i]))
            return 0;
        if (fabsf(got[i] - want[i]) > tol)
            return 0;
    }
    return 1;
}

static void fill_student_grid(int nfe, float shift, float *out) {
    int i;
    for (i = 0; i <= nfe; i++) {
        float base = 1.0f - (float)i / (float)nfe;
        out[i] = shifted_sigma(base, shift);
    }
    out[nfe] = 0.0f;
}

int h3_fewstep_contract_check_sigmas(const h3_fewstep_contract_in *in, char *err,
                                     size_t err_size) {
    float want_v[16];
    float want_a[16];
    int nfe;
    float a_shift;
    char buf[512];
    const float tol = 0.0025f; /* ~0.25% abs; printed logs use ~4g */

    if (err && err_size)
        err[0] = '\0';
    if (!in || in->id == H3_FEWSTEP_NONE)
        return 1;
    if (!in->video_sigmas || !in->audio_sigmas)
        return 1; /* early-only call site */

    switch (in->id) {
    case H3_FEWSTEP_PDMD4:
        nfe = 4;
        a_shift = 3.0f;
        break;
    case H3_FEWSTEP_PDMD2:
        nfe = 2;
        /* Accept either paper (3) or look-feel (6) grid. */
        if (fabsf(in->audio_shift - 6.0f) <= 0.051f)
            a_shift = 6.0f;
        else
            a_shift = 3.0f;
        break;
    case H3_FEWSTEP_DMAD4:
        nfe = 4;
        a_shift = 2.0f;
        break;
    default:
        return 1;
    }

    if (in->steps != nfe) {
        snprintf(buf, sizeof(buf),
                 "fewstep_contract=%s sigma 检查 steps=%d 期望 %d",
                 h3_fewstep_contract_name(in->id), in->steps, nfe);
        return fail_msg(err, err_size, buf);
    }
    if (nfe + 1 > (int)(sizeof(want_v) / sizeof(want_v[0])))
        return fail_msg(err, err_size, "fewstep_contract: nfe too large");

    fill_student_grid(nfe, 12.0f, want_v);
    fill_student_grid(nfe, a_shift, want_a);

    if (!sigmas_match(in->video_sigmas, want_v, nfe + 1, tol)) {
        snprintf(buf, sizeof(buf),
                 "fewstep_contract=%s video sigma 与 student_grid(%d,12) 不一致 "
                 "（拒 turbo4/错表冒充；期望首中末≈%.4g / %.4g / 0）",
                 h3_fewstep_contract_name(in->id), nfe, (double)want_v[0],
                 (double)want_v[nfe / 2]);
        return fail_msg(err, err_size, buf);
    }
    if (!sigmas_match(in->audio_sigmas, want_a, nfe + 1, tol)) {
        snprintf(buf, sizeof(buf),
                 "fewstep_contract=%s audio sigma 与 student_grid(%d,%.3g) 不一致 "
                 "（检查 H3_AUDIO_SIGMA_SHIFT）",
                 h3_fewstep_contract_name(in->id), nfe, (double)a_shift);
        return fail_msg(err, err_size, buf);
    }

    printf("fewstep_contract: sigma_ok student_grid(nfe=%d, video_shift=12, "
           "audio_shift=%.3g)\n",
           nfe, (double)a_shift);
    fflush(stdout);
    return 1;
}
