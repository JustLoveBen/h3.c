/* Host-only fewstep contract tests (no CUDA). */
#include "h3_fewstep_contract.h"
#include "tests/fixtures/fewstep_student_grid_golden.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int fails;

static void expect(int cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "FAIL %s\n", msg);
        fails++;
    }
}

static void fill_grid(int nfe, float shift, float *out) {
    int i;
    for (i = 0; i <= nfe; i++) {
        float base = 1.0f - (float)i / (float)nfe;
        out[i] = (base > 0.0f)
                     ? shift * base / (1.0f + (shift - 1.0f) * base)
                     : 0.0f;
    }
    out[nfe] = 0.0f;
}

int main(void) {
    h3_fewstep_contract_in in;
    char err[256];
    float v[8], a[8];

    unsetenv("H3_FEWSTEP_CONTRACT");
    unsetenv("H3_FEWSTEP_STRICT");
    unsetenv("H3_FEWSTEP_ALLOW_NVFP4");

    expect(h3_fewstep_contract_parse(NULL, err, sizeof err) == H3_FEWSTEP_NONE,
           "parse null");
    expect(h3_fewstep_contract_parse("pdmd4", err, sizeof err) ==
               H3_FEWSTEP_PDMD4,
           "parse pdmd4");
    expect(h3_fewstep_contract_parse("pdmd2", err, sizeof err) ==
               H3_FEWSTEP_PDMD2,
           "parse pdmd2");
    expect(h3_fewstep_contract_parse("dmad4", err, sizeof err) ==
               H3_FEWSTEP_DMAD4,
           "parse dmad4");
    expect(h3_fewstep_contract_parse("turbo4", err, sizeof err) ==
               H3_FEWSTEP_NONE,
           "parse bad → none");
    expect(err[0] != '\0', "bad parse sets err");

    expect(h3_fewstep_contract_resolve(NULL, err, sizeof err) ==
               H3_FEWSTEP_NONE,
           "resolve unset");
    setenv("H3_FEWSTEP_CONTRACT", "pdmd4", 1);
    expect(h3_fewstep_contract_resolve(NULL, err, sizeof err) ==
               H3_FEWSTEP_PDMD4,
           "resolve env");
    unsetenv("H3_FEWSTEP_CONTRACT");

    /* NONE: always ok */
    memset(&in, 0, sizeof in);
    in.id = H3_FEWSTEP_NONE;
    expect(h3_fewstep_contract_check_early(&in, err, sizeof err) == 1,
           "none early");

    /* pdmd4 + turbo4 preset → FAIL */
    memset(&in, 0, sizeof in);
    in.id = H3_FEWSTEP_PDMD4;
    in.steps = 4;
    in.preset_rewrites_sigma = 1;
    in.preset_name = "turbo4";
    in.video_shift = 12.0f;
    in.audio_shift = 3.0f;
    in.cfg_scale = 1.0f;
    in.sampler_is_velocity_euler = 1;
    expect(h3_fewstep_contract_check_early(&in, err, sizeof err) == 0,
           "pdmd4+turbo4 fail");
    expect(strstr(err, "preset") != NULL, "pdmd4+turbo4 msg");

    /* pdmd4 wrong steps */
    in.preset_rewrites_sigma = 0;
    in.preset_name = NULL;
    in.steps = 8;
    expect(h3_fewstep_contract_check_early(&in, err, sizeof err) == 0,
           "pdmd4 steps fail");

    /* pdmd4 wrong audio shift */
    in.steps = 4;
    in.audio_shift = 2.0f;
    expect(h3_fewstep_contract_check_early(&in, err, sizeof err) == 0,
           "pdmd4 audio fail");

    /* pdmd4 ok */
    in.audio_shift = 3.0f;
    expect(h3_fewstep_contract_check_early(&in, err, sizeof err) == 1,
           "pdmd4 ok");

    /* pdmd4 × NVFP4 without allow → FAIL */
    in.nvfp4_enabled = 1;
    expect(h3_fewstep_contract_check_early(&in, err, sizeof err) == 0,
           "pdmd4 nvfp4 fail");
    setenv("H3_FEWSTEP_ALLOW_NVFP4", "1", 1);
    expect(h3_fewstep_contract_check_early(&in, err, sizeof err) == 1,
           "pdmd4 nvfp4 allow");
    unsetenv("H3_FEWSTEP_ALLOW_NVFP4");
    in.nvfp4_enabled = 0;

    /* dmad4 without lcm → FAIL even if shifts ok */
    memset(&in, 0, sizeof in);
    in.id = H3_FEWSTEP_DMAD4;
    in.steps = 4;
    in.video_shift = 12.0f;
    in.audio_shift = 2.0f;
    in.cfg_scale = 1.0f;
    in.sampler_is_velocity_euler = 1;
    in.has_lcm_renoise = 0;
    expect(h3_fewstep_contract_check_early(&in, err, sizeof err) == 0,
           "dmad4 no lcm fail");
    expect(strstr(err, "lcm") != NULL, "dmad4 lcm msg");

    /* dmad4 + lcm_renoise + shift 12/2 → PASS */
    in.has_lcm_renoise = 1;
    in.audio_shift = 2.0f;
    expect(h3_fewstep_contract_check_early(&in, err, sizeof err) == 1,
           "dmad4 lcm ok");

    /* dmad4 wrong audio */
    in.has_lcm_renoise = 1;
    in.audio_shift = 3.0f;
    expect(h3_fewstep_contract_check_early(&in, err, sizeof err) == 0,
           "dmad4 audio fail");

    /* pdmd2 audio 6 WARN but ok; strict fail on 5 */
    memset(&in, 0, sizeof in);
    in.id = H3_FEWSTEP_PDMD2;
    in.steps = 2;
    in.video_shift = 12.0f;
    in.audio_shift = 6.0f;
    in.cfg_scale = 1.0f;
    in.sampler_is_velocity_euler = 1;
    expect(h3_fewstep_contract_check_early(&in, err, sizeof err) == 1,
           "pdmd2 lookfeel ok");
    in.audio_shift = 5.0f;
    expect(h3_fewstep_contract_check_early(&in, err, sizeof err) == 1,
           "pdmd2 bad audio warn-only");
    setenv("H3_FEWSTEP_STRICT", "1", 1);
    expect(h3_fewstep_contract_check_early(&in, err, sizeof err) == 0,
           "pdmd2 strict fail");
    unsetenv("H3_FEWSTEP_STRICT");

    /* sigma golden fixtures: fill ≡ frozen tables */
    {
        float g[8];
        int i;
        h3_fewstep_student_grid_fill(4, 12.0f, g);
        for (i = 0; i < H3_FEWSTEP_GOLDEN_PDMD4_VIDEO_S12_LEN; i++)
            expect(g[i] == H3_FEWSTEP_GOLDEN_PDMD4_VIDEO_S12[i],
                   "fill≡pdmd4 video golden");
        h3_fewstep_student_grid_fill(4, 3.0f, g);
        for (i = 0; i < H3_FEWSTEP_GOLDEN_PDMD4_AUDIO_S3_LEN; i++)
            expect(g[i] == H3_FEWSTEP_GOLDEN_PDMD4_AUDIO_S3[i],
                   "fill≡pdmd4 audio golden");
        h3_fewstep_student_grid_fill(4, 2.0f, g);
        for (i = 0; i < H3_FEWSTEP_GOLDEN_DMAD4_AUDIO_S2_LEN; i++)
            expect(g[i] == H3_FEWSTEP_GOLDEN_DMAD4_AUDIO_S2[i],
                   "fill≡dmad4 audio golden");
        h3_fewstep_student_grid_fill(2, 12.0f, g);
        for (i = 0; i < H3_FEWSTEP_GOLDEN_PDMD2_VIDEO_S12_LEN; i++)
            expect(g[i] == H3_FEWSTEP_GOLDEN_PDMD2_VIDEO_S12[i],
                   "fill≡pdmd2 video golden");
        h3_fewstep_student_grid_fill(2, 3.0f, g);
        for (i = 0; i < H3_FEWSTEP_GOLDEN_PDMD2_AUDIO_S3_LEN; i++)
            expect(g[i] == H3_FEWSTEP_GOLDEN_PDMD2_AUDIO_S3[i],
                   "fill≡pdmd2 audio3 golden");
        h3_fewstep_student_grid_fill(2, 6.0f, g);
        for (i = 0; i < H3_FEWSTEP_GOLDEN_PDMD2_AUDIO_S6_LEN; i++)
            expect(g[i] == H3_FEWSTEP_GOLDEN_PDMD2_AUDIO_S6[i],
                   "fill≡pdmd2 audio6 golden");
    }

    /* sigma golden: pdmd4 against frozen table */
    memset(&in, 0, sizeof in);
    in.id = H3_FEWSTEP_PDMD4;
    in.steps = 4;
    in.audio_shift = 3.0f;
    in.video_sigmas = H3_FEWSTEP_GOLDEN_PDMD4_VIDEO_S12;
    in.audio_sigmas = H3_FEWSTEP_GOLDEN_PDMD4_AUDIO_S3;
    expect(h3_fewstep_contract_check_sigmas(&in, err, sizeof err) == 1,
           "pdmd4 sigma ok");
    /* Corrupt mid video sigma (turbo-lookalike mismatch simulation) */
    fill_grid(4, 12.0f, v);
    fill_grid(4, 3.0f, a);
    v[1] = 0.5f;
    in.video_sigmas = v;
    in.audio_sigmas = a;
    expect(h3_fewstep_contract_check_sigmas(&in, err, sizeof err) == 0,
           "pdmd4 sigma mismatch");
    expect(strstr(err, "student_grid") != NULL, "sigma msg");

    /* pdmd2 audio=3 and audio=6 both accept golden */
    memset(&in, 0, sizeof in);
    in.id = H3_FEWSTEP_PDMD2;
    in.steps = 2;
    in.audio_shift = 3.0f;
    in.video_sigmas = H3_FEWSTEP_GOLDEN_PDMD2_VIDEO_S12;
    in.audio_sigmas = H3_FEWSTEP_GOLDEN_PDMD2_AUDIO_S3;
    expect(h3_fewstep_contract_check_sigmas(&in, err, sizeof err) == 1,
           "pdmd2 sigma audio3 ok");
    in.audio_shift = 6.0f;
    in.audio_sigmas = H3_FEWSTEP_GOLDEN_PDMD2_AUDIO_S6;
    expect(h3_fewstep_contract_check_sigmas(&in, err, sizeof err) == 1,
           "pdmd2 sigma audio6 ok");
    /* wrong audio table under shift=3 */
    in.audio_shift = 3.0f;
    in.audio_sigmas = H3_FEWSTEP_GOLDEN_PDMD2_AUDIO_S6;
    expect(h3_fewstep_contract_check_sigmas(&in, err, sizeof err) == 0,
           "pdmd2 sigma wrong audio table");

    /* dmad4 golden */
    memset(&in, 0, sizeof in);
    in.id = H3_FEWSTEP_DMAD4;
    in.steps = 4;
    in.audio_shift = 2.0f;
    in.video_sigmas = H3_FEWSTEP_GOLDEN_PDMD4_VIDEO_S12;
    in.audio_sigmas = H3_FEWSTEP_GOLDEN_DMAD4_AUDIO_S2;
    expect(h3_fewstep_contract_check_sigmas(&in, err, sizeof err) == 1,
           "dmad4 sigma ok");
    in.audio_sigmas = H3_FEWSTEP_GOLDEN_PDMD4_AUDIO_S3;
    expect(h3_fewstep_contract_check_sigmas(&in, err, sizeof err) == 0,
           "dmad4 sigma wrong audio");

    /* kijai path → WARN only (still ok) */
    memset(&in, 0, sizeof in);
    in.id = H3_FEWSTEP_PDMD4;
    in.steps = 4;
    in.video_shift = 12.0f;
    in.audio_shift = 3.0f;
    in.cfg_scale = 1.0f;
    in.lora_path = "/models/kijai_rank_39_bf16.safetensors";
    expect(h3_fewstep_contract_check_early(&in, err, sizeof err) == 1,
           "kijai warn only");

    /* --- fewstep-preset one-shot expansion --- */
    {
        h3_fewstep_preset_in pin;
        h3_fewstep_preset_out pout;

        unsetenv("H3_FEWSTEP_PRESET");
        memset(&pin, 0, sizeof pin);
        pin.steps = 8;
        pin.cfg_scale = 5.0f;
        expect(h3_fewstep_preset_apply(&pin, &pout, err, sizeof err) == 1,
               "preset none ok");
        expect(pout.applied == 0 && pout.steps == 8, "preset none no rewrite");

        pin.name = "pdmd4";
        expect(h3_fewstep_preset_apply(&pin, &pout, err, sizeof err) == 1,
               "preset pdmd4");
        expect(pout.applied == 1 && pout.id == H3_FEWSTEP_PDMD4, "pdmd4 id");
        expect(pout.steps == 4 && pout.cfg_scale == 1.0f, "pdmd4 steps/cfg");
        expect(pout.audio_shift == 3.0f && pout.video_shift == 12.0f,
               "pdmd4 shifts");
        expect(pout.want_lcm_renoise == 0, "pdmd4 euler");
        expect(pout.set_audio_env == 1 && pout.audio_env_value &&
                   !strcmp(pout.audio_env_value, "3"),
               "pdmd4 audio env");

        pin.name = "pdmd2";
        expect(h3_fewstep_preset_apply(&pin, &pout, err, sizeof err) == 1,
               "preset pdmd2");
        expect(pout.steps == 2 && pout.id == H3_FEWSTEP_PDMD2, "pdmd2 steps");
        expect(pout.want_lcm_renoise == 0, "pdmd2 euler");

        pin.name = "dmad4";
        expect(h3_fewstep_preset_apply(&pin, &pout, err, sizeof err) == 1,
               "preset dmad4");
        expect(pout.id == H3_FEWSTEP_DMAD4 && pout.steps == 4, "dmad4 id");
        expect(pout.audio_shift == 2.0f && pout.want_lcm_renoise == 1,
               "dmad4 lcm+shift2");
        expect(pout.audio_env_value && !strcmp(pout.audio_env_value, "2"),
               "dmad4 audio env");

        /* explicit steps/cfg kept */
        pin.name = "pdmd4";
        pin.steps_explicit = 1;
        pin.steps = 8;
        pin.cfg_explicit = 1;
        pin.cfg_scale = 2.5f;
        expect(h3_fewstep_preset_apply(&pin, &pout, err, sizeof err) == 1,
               "preset keep explicit");
        expect(pout.steps == 8 && pout.cfg_scale == 2.5f, "explicit kept");
        pin.steps_explicit = 0;
        pin.cfg_explicit = 0;
        pin.steps = 8;
        pin.cfg_scale = 5.0f;

        /* conflict with other contract */
        pin.name = "pdmd4";
        pin.contract_already_set = 1;
        pin.contract_already = H3_FEWSTEP_DMAD4;
        expect(h3_fewstep_preset_apply(&pin, &pout, err, sizeof err) == 0,
               "preset vs contract conflict");
        expect(strstr(err, "冲突") != NULL, "conflict msg");
        pin.contract_already_set = 0;

        /* same contract ok */
        pin.contract_already_set = 1;
        pin.contract_already = H3_FEWSTEP_PDMD4;
        expect(h3_fewstep_preset_apply(&pin, &pout, err, sizeof err) == 1,
               "preset matching contract");
        pin.contract_already_set = 0;

        /* env H3_FEWSTEP_PRESET */
        pin.name = NULL;
        setenv("H3_FEWSTEP_PRESET", "dmad4", 1);
        expect(h3_fewstep_preset_apply(&pin, &pout, err, sizeof err) == 1,
               "preset env");
        expect(pout.applied == 1 && pout.id == H3_FEWSTEP_DMAD4, "env dmad4");
        unsetenv("H3_FEWSTEP_PRESET");

        pin.name = "turbo4";
        expect(h3_fewstep_preset_apply(&pin, &pout, err, sizeof err) == 0,
               "preset bad name");
        expect(strstr(err, "fewstep-preset") != NULL, "bad name msg");
    }


    /* --- fewstep-preset vs pre-set H3_AUDIO_SIGMA_SHIFT --- */
    {
        unsetenv("H3_AUDIO_SIGMA_SHIFT");
        expect(h3_fewstep_preset_audio_env_compatible(2.0f, err, sizeof err) ==
                   1,
               "audio env unset ok");
        setenv("H3_AUDIO_SIGMA_SHIFT", "2", 1);
        expect(h3_fewstep_preset_audio_env_compatible(2.0f, err, sizeof err) ==
                   1,
               "audio env match dmad");
        setenv("H3_AUDIO_SIGMA_SHIFT", "3.0", 1);
        expect(h3_fewstep_preset_audio_env_compatible(2.0f, err, sizeof err) ==
                   0,
               "audio env 3 vs dmad2 fail");
        expect(strstr(err, "冲突") != NULL, "audio env conflict msg");
        expect(h3_fewstep_preset_audio_env_compatible(3.0f, err, sizeof err) ==
                   1,
               "audio env 3 vs pdmd ok");
        setenv("H3_AUDIO_SIGMA_SHIFT", "not-a-float", 1);
        expect(h3_fewstep_preset_audio_env_compatible(3.0f, err, sizeof err) ==
                   0,
               "audio env garbage fail");
        unsetenv("H3_AUDIO_SIGMA_SHIFT");
    }

    /* Shared help text must mention the three CLI knobs (host; no generate.c). */
    {
        char tmp[] = "/tmp/h3_fewstep_help_XXXXXX";
        int fd = mkstemp(tmp);
        FILE *tf;
        char buf[4096];
        size_t n = 0;
        expect(fd >= 0, "help mkstemp");
        if (fd >= 0) {
            tf = fdopen(fd, "w+");
            expect(tf != NULL, "help fdopen");
            if (tf) {
                h3_fewstep_help_fprint(tf);
                fflush(tf);
                rewind(tf);
                n = fread(buf, 1, sizeof buf - 1, tf);
                buf[n] = '\0';
                fclose(tf);
                unlink(tmp);
                expect(strstr(buf, "--fewstep-contract") != NULL, "help contract");
                expect(strstr(buf, "--fewstep-preset") != NULL, "help preset");
                expect(strstr(buf, "lcm_renoise") != NULL, "help sampler");
            }
        }
    }

    if (fails) {
        fprintf(stderr, "%d failure(s)\n", fails);
        return 1;
    }

    printf("ok h3_fewstep_contract_test\n");
    return 0;
}
