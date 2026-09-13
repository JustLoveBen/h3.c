/* tests/test_h3_generate_preset.c */
#include "h3_generate_preset.h"
#include <stdio.h>
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
    in.magcache_thresh = 0.0f;
    in.preset_name = "fast4";
    expect(h3_generate_preset_apply(&in, &out, err, sizeof err) == 0,
           "fast4 rejected");
    in.preset_name = "nope";
    expect(h3_generate_preset_apply(&in, &out, err, sizeof err) == 0, "unknown");
    return fails ? 1 : (printf("test_h3_generate_preset: PASS\n"), 0);
}
