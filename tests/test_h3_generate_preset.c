/* tests/test_h3_generate_preset.c */
#include "h3_generate_preset.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int fails;
static void expect(int c, const char *m) {
    if (!c) {
        fprintf(stderr, "FAIL %s\n", m);
        fails++;
    }
}
int main(void) {
    h3_generate_preset_in in;
    h3_generate_preset_out out;
    char err[256];
    memset(&in, 0, sizeof in);
    in.steps = 2;
    in.sdpa = "cudnn";
    in.cfg_scale = 5.0f;
    in.reuse_interval = 1;
    expect(h3_generate_preset_apply(&in, &out, err, sizeof err) == 1, "none");
    expect(out.id == H3_PRESET_NONE, "id none");
    expect(out.steps == 2 && strcmp(out.sdpa, "cudnn") == 0, "none no rewrite");
    expect(out.non_gold == 0, "none is gold path");

    in.preset_name = "gold";
    expect(h3_generate_preset_apply(&in, &out, err, sizeof err) == 1, "gold");
    expect(out.steps == 2 && strcmp(out.sdpa, "cudnn") == 0, "gold no rewrite");

    in.preset_name = "turbo4";
    in.lora_path = NULL;
    expect(h3_generate_preset_apply(&in, &out, err, sizeof err) == 0,
           "turbo no lora");
    expect(strstr(err, "非金线") != NULL, "stderr 非金线");

    in.lora_path = "/tmp/fake.safetensors";
    expect(h3_generate_preset_apply(&in, &out, err, sizeof err) == 1, "turbo ok");
    expect(out.id == H3_PRESET_TURBO4 && out.non_gold == 1, "turbo id");
    expect(out.steps == 4, "steps 4");
    expect(strcmp(out.sdpa, "sage3") == 0, "sage3");
    expect(out.cfg_scale == 1.0f, "cfg 1");

    in.steps_explicit = 1;
    in.steps = 8;
    expect(h3_generate_preset_apply(&in, &out, err, sizeof err) == 1,
           "override steps");
    expect(out.steps == 8, "explicit steps kept");

    in.steps_explicit = 0;
    in.steps = 2;
    in.sdpa_explicit = 1;
    in.sdpa = "cudnn";
    expect(h3_generate_preset_apply(&in, &out, err, sizeof err) == 1,
           "override sdpa");
    expect(strcmp(out.sdpa, "cudnn") == 0, "explicit sdpa kept");

    in.sdpa_explicit = 0;
    in.sdpa = "cudnn";
    in.magcache_thresh = 0.08f;
    expect(h3_generate_preset_apply(&in, &out, err, sizeof err) == 0, "turbo+mag");
    setenv("H3_TURBO4_ALLOW_CACHE", "1", 1);
    expect(h3_generate_preset_apply(&in, &out, err, sizeof err) == 1,
           "turbo+mag opt-in allow");
    unsetenv("H3_TURBO4_ALLOW_CACHE");
    expect(h3_generate_preset_apply(&in, &out, err, sizeof err) == 0,
           "turbo+mag re-locked after unsetenv");
    in.magcache_thresh = 0.0f;
    in.preset_name = "fast4";
    expect(h3_generate_preset_apply(&in, &out, err, sizeof err) == 0,
           "fast4 rejected");
    in.preset_name = "nvfp4";
    in.lora_path = NULL;
    in.magcache_thresh = 0.0f;
    in.sdpa_explicit = 0;
    in.sdpa = "sage3";
    in.steps_explicit = 0;
    in.steps = 2;
    in.cfg_explicit = 0;
    in.cfg_scale = 5.0f;
    expect(h3_generate_preset_apply(&in, &out, err, sizeof err) == 1, "nvfp4");
    expect(out.id == H3_PRESET_NVFP4 && out.non_gold == 1, "nvfp4 id");
    expect(out.steps == 8, "nvfp4 steps 8");
    expect(strcmp(out.sdpa, "cudnn") == 0, "nvfp4 cudnn");
    expect(out.cfg_scale == 1.0f, "nvfp4 cfg 1");
    expect(out.lora_path == NULL, "nvfp4 no lora required");

    in.preset_name = "lift";
    in.lora_path = NULL;
    in.magcache_thresh = 0.0f;
    in.sdpa_explicit = 0;
    in.sdpa = "sage3";
    in.steps_explicit = 0;
    in.steps = 2;
    in.cfg_explicit = 0;
    in.cfg_scale = 5.0f;
    expect(h3_generate_preset_apply(&in, &out, err, sizeof err) == 1, "lift");
    expect(out.id == H3_PRESET_LIFT && out.non_gold == 1, "lift id");
    expect(out.steps == 8, "lift steps 8");
    expect(strcmp(out.sdpa, "cudnn") == 0, "lift cudnn");
    expect(out.cfg_scale == 1.0f, "lift cfg 1");
    in.preset_name = "self-lift";
    expect(h3_generate_preset_apply(&in, &out, err, sizeof err) == 1,
           "self-lift alias");
    expect(out.id == H3_PRESET_LIFT, "self-lift id");

    in.preset_name = "taomate";
    in.lora_path = NULL;
    in.magcache_thresh = 0.0f;
    in.sdpa_explicit = 0;
    in.sdpa = "cudnn";
    in.steps_explicit = 0;
    in.steps = 8;
    in.cfg_explicit = 0;
    in.cfg_scale = 5.0f;
    expect(h3_generate_preset_apply(&in, &out, err, sizeof err) == 0,
           "taomate no lora");
    expect(strstr(err, "非金线") != NULL, "taomate 非金线");

    in.lora_path = "/tmp/taomate.safetensors";
    expect(h3_generate_preset_apply(&in, &out, err, sizeof err) == 1,
           "taomate ok");
    expect(out.id == H3_PRESET_TAOMATE && out.non_gold == 1, "taomate id");
    expect(out.steps == 3, "taomate 3-step");
    expect(out.lift == 0, "taomate no lift");
    expect(strcmp(out.sdpa, "sage3") == 0, "taomate sage3");
    expect(out.cfg_scale == 1.0f, "taomate cfg 1");

    in.preset_name = "taomate-lift";
    expect(h3_generate_preset_apply(&in, &out, err, sizeof err) == 1,
           "taomate-lift ok");
    expect(out.id == H3_PRESET_TAOMATE && out.lift == 1, "taomate-lift flag");
    expect(out.steps == 3, "taomate-lift keeps 3-step");
    in.preset_name = "taomate";

    in.steps_explicit = 1;
    in.steps = 4;
    expect(h3_generate_preset_apply(&in, &out, err, sizeof err) == 0,
           "taomate reject 4-step");
    expect(strstr(err, "3-step") != NULL || strstr(err, "锚点") != NULL ||
               strstr(err, "6-step") != NULL,
           "taomate 4-step err");
    in.steps = 5;
    expect(h3_generate_preset_apply(&in, &out, err, sizeof err) == 1,
           "taomate allow Comfy simple-5");
    expect(out.steps == 5, "taomate 5 kept");
    in.steps = 6;
    expect(h3_generate_preset_apply(&in, &out, err, sizeof err) == 1,
           "taomate allow 6-step");
    expect(out.steps == 6, "taomate 6 kept");
    in.steps_explicit = 0;

    in.preset_name = "nope";
    expect(h3_generate_preset_apply(&in, &out, err, sizeof err) == 0, "unknown");
    return fails ? 1 : (printf("test_h3_generate_preset: PASS\n"), 0);
}
