#include "h3_generate_preset.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char k_sdpa_sage3[] = "sage3";

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
    if (strcmp(name, "turbo4") != 0) {
        char buf[256];
        snprintf(buf, sizeof(buf), "unknown --preset '%s' (gold|turbo4)", name);
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

    if (in->magcache_thresh > 0.0f)
        return preset_fail(err, err_size,
                           "preset=turbo4 非金线：拒绝 magcache（极速档 cache 关）");
    if (in->step_cache_kind != 0)
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
