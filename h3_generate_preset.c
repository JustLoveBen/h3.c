#include "h3_generate_preset.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char k_sdpa_sage3[] = "sage3";

/* R-F 实测封档（docs/research/2026-10-07-deadend-audit-quality.md #1）：
 * turbo4 默认拒绝一切跨步缓存；仅当用户显式置 H3_TURBO4_ALLOW_CACHE=1 时
 * 解锁 mag/tea/fb 叠加（opt-in，默认零行为变化）。 */
static int turbo4_allow_cache(void) {
    const char *v = getenv("H3_TURBO4_ALLOW_CACHE");
    return v && v[0] == '1';
}

static int preset_fail(char *err, size_t err_size, const char *msg) {
    if (err && err_size)
        snprintf(err, err_size, "%s", msg ? msg : "preset error");
    return 0;
}

int h3_generate_preset_apply(const h3_generate_preset_in *in,
                             h3_generate_preset_out *out, char *err,
                             size_t err_size) {
    const char *name;
    const char *lora;
    h3_preset_id id = H3_PRESET_NONE;

    if (err && err_size)
        err[0] = '\0';
    if (!in || !out)
        return preset_fail(err, err_size, "preset: null args");
    memset(out, 0, sizeof(*out));
    out->steps = in->steps;
    out->sdpa = in->sdpa;
    out->cfg_scale = in->cfg_scale;
    out->lora_path = in->lora_path;
    out->lora_scale = 1.0f;
    out->id = H3_PRESET_NONE;
    out->non_gold = 0;

    name = in->preset_name;
    if (!name || !name[0]) {
        out->id = H3_PRESET_NONE;
        return 1;
    }
    if (!strcmp(name, "gold")) {
        out->id = H3_PRESET_GOLD;
        return 1;
    }
    if (!strcmp(name, "fast4")) {
        return preset_fail(err, err_size,
                           "preset fast4 已更名为 turbo4（非金线极速档）");
    }
    if (!strcmp(name, "lift") || !strcmp(name, "self-lift") ||
        !strcmp(name, "self_lift")) {
        if (in->magcache_thresh > 0.0f)
            return preset_fail(err, err_size,
                               "preset=lift 非金线：拒绝 magcache");
        if (in->step_cache_kind != 0)
            return preset_fail(err, err_size,
                               "preset=lift 非金线：拒绝 tea/fb/easy");
        if (in->reuse_interval > 1)
            return preset_fail(err, err_size,
                               "preset=lift 非金线：拒绝 reuse>1");
        out->id = H3_PRESET_LIFT;
        out->non_gold = 1;
        if (!in->steps_explicit)
            out->steps = 8;
        if (!in->sdpa_explicit)
            out->sdpa = "cudnn";
        if (!in->cfg_explicit)
            out->cfg_scale = 1.0f;
        return 1;
    }
    if (!strcmp(name, "taomate") || !strcmp(name, "taomate-lift") ||
        !strcmp(name, "taomate_lift")) {
        int want_lift = strcmp(name, "taomate") != 0;
        if (in->magcache_thresh > 0.0f)
            return preset_fail(err, err_size,
                               "preset=taomate 非金线：拒绝 magcache");
        if (in->step_cache_kind != 0)
            return preset_fail(err, err_size,
                               "preset=taomate 非金线：拒绝 tea/fb/easy");
        if (in->reuse_interval > 1)
            return preset_fail(err, err_size,
                               "preset=taomate 非金线：拒绝 reuse>1");
        if (in->steps_explicit && in->steps != 3 && in->steps != 5 &&
            in->steps != 6)
            return preset_fail(err, err_size,
                               "preset=taomate 非金线：3-step 锚点 {0,16,33,49}、"
                               "Comfy simple-5（SplitSigmas 3+2）、或 6-step 中点；"
                               "禁止 4-step sage");
        lora = in->lora_path;
        if (!lora || !lora[0]) {
            const char *env = getenv("H3_TAOMATE_LORA");
            if (env && env[0])
                lora = env;
        }
        if (!lora || !lora[0]) {
            return preset_fail(err, err_size,
                               "preset=taomate 非金线：必须 --lora FILE "
                               "（或 H3_TAOMATE_LORA）");
        }
        out->id = H3_PRESET_TAOMATE;
        out->non_gold = 1;
        out->lift = want_lift;
        out->lora_path = lora;
        out->steps = in->steps_explicit ? in->steps : 3;
        if (!in->sdpa_explicit)
            out->sdpa = k_sdpa_sage3;
        if (!in->cfg_explicit)
            out->cfg_scale = 1.0f;
        if (!out->sdpa)
            out->sdpa = k_sdpa_sage3;
        return 1;
    }
    if (!strcmp(name, "nvfp4")) {
        /* DISARM (2026-10-27): --preset nvfp4 was the Sep-13 host
         * dequant NVFP4→INT8 bypass (zero FP4 speedup). Do not re-enable.
         * Target path: INT8 ConvRot --dit + H3_DIT_NVFP4=1 +
         * H3_DIT_NVFP4_FILE=<nvfp4_w4a4.safetensors> (+ optional KEEP_INT8). */
        return preset_fail(err, err_size,
                           "preset=nvfp4 DISARMED (host dequant→INT8 bypass; "
                           "use H3_DIT_NVFP4=1 + NVFP4_FILE w4a4 native FP4)");
    }
    if (strcmp(name, "turbo4") != 0) {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "unknown --preset '%s' (gold|turbo4|nvfp4|lift|taomate|"
                 "taomate-lift)",
                 name);
        return preset_fail(err, err_size, buf);
    }

    id = H3_PRESET_TURBO4;
    out->id = id;
    out->non_gold = 1;

    lora = in->lora_path;
    if (!lora || !lora[0]) {
        const char *env = getenv("H3_TURBO_LORA");
        if (env && env[0])
            lora = env;
    }
    if (!lora || !lora[0]) {
        return preset_fail(err, err_size,
                           "preset=turbo4 非金线：必须 --lora FILE.safetensors "
                           "（或 H3_TURBO_LORA）；未蒸馏 --steps 4 不是产品");
    }
    out->lora_path = lora;

    if (in->magcache_thresh > 0.0f && !turbo4_allow_cache())
        return preset_fail(err, err_size,
                           "preset=turbo4 非金线：拒绝 magcache（极速档 cache 关）");
    if (in->step_cache_kind != 0 && !turbo4_allow_cache())
        return preset_fail(err, err_size,
                           "preset=turbo4 非金线：拒绝 tea/fb/easy（极速档 cache 关）");
    if (in->reuse_interval > 1)
        return preset_fail(err, err_size,
                           "preset=turbo4 非金线：拒绝 reuse>1（极速档全评）");

    if (!in->steps_explicit)
        out->steps = 4;
    if (!in->sdpa_explicit)
        out->sdpa = k_sdpa_sage3;
    if (!in->cfg_explicit)
        out->cfg_scale = 1.0f;
    if (!out->sdpa)
        out->sdpa = k_sdpa_sage3;
    return 1;
}
