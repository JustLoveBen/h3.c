/* Host-only unit tests for h3_env_contract (C061). */
#include "h3_env_contract.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail;

static void expect_ok(const char *what, int cond) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", what);
        g_fail = 1;
    }
}

int main(void) {
    char err[512];
    char buf[4096];
    FILE *mem;
    h3_env_contract_cli cli;

    unsetenv("H3_DIT_NVFP4");
    unsetenv("H3_DIT_NVFP4_FILE");
    unsetenv("H3_DIT_NVFP4_CHAIN");
    unsetenv("H3_DIT_NVFP4_CHAIN_FC1_F32");
    unsetenv("H3_DIT_NVFP4_CHAIN_DIAG");
    unsetenv("H3_SDPA_CUDNN_CUDA_GRAPH");
    unsetenv("H3_TEXT_H_LOAD");
    unsetenv("H3_DIAG_W_CKSUM");
    unsetenv("H3_DIAG_BLK0");
    unsetenv("H3_EXPECT_P0_VISUAL");

    expect_ok("p0 int8 clean", h3_env_contract_check_p0("int8", err, sizeof(err)));

    setenv("H3_DIT_NVFP4_CHAIN", "1", 1);
    expect_ok("p0 rejects chain",
              !h3_env_contract_check_p0("int8", err, sizeof(err)));
    expect_ok("chain msg", strstr(err, "H3_DIT_NVFP4_CHAIN") != NULL);
    unsetenv("H3_DIT_NVFP4_CHAIN");

    setenv("H3_SDPA_CUDNN_CUDA_GRAPH", "1", 1);
    expect_ok("p0 rejects sdpa graph",
              !h3_env_contract_check_p0("int8", err, sizeof(err)));
    unsetenv("H3_SDPA_CUDNN_CUDA_GRAPH");

    setenv("H3_DIT_NVFP4", "1", 1);
    setenv("H3_DIT_NVFP4_FILE", "/tmp/x.safetensors", 1);
    expect_ok("p0 nvfp4 arm ok",
              h3_env_contract_check_p0("nvfp4", err, sizeof(err)));
    expect_ok("p0 nvfp4 rejected on int8 arm",
              !h3_env_contract_check_p0("int8", err, sizeof(err)));
    unsetenv("H3_DIT_NVFP4");
    unsetenv("H3_DIT_NVFP4_FILE");

    memset(&cli, 0, sizeof(cli));
    cli.seed = 0x76800002ULL;
    cli.skip_audio = 1;
    cli.fewstep_preset = "pdmd4";
    cli.fewstep_contract = "pdmd4";
    cli.sampler = "velocity_euler";
    cli.sdpa = "cudnn";
    cli.arm = "int8";
    mem = tmpfile();
    expect_ok("tmpfile", mem != NULL);
    if (mem) {
        size_t n;
        h3_env_contract_fprint(mem, &cli);
        rewind(mem);
        n = fread(buf, 1, sizeof(buf) - 1, mem);
        buf[n] = '\0';
        fclose(mem);
        expect_ok("marker", strstr(buf, "# H3_ENV_CONTRACT") != NULL);
        expect_ok("seed field", strstr(buf, "seed=0x76800002") != NULL);
        expect_ok("skip_audio", strstr(buf, "skip_audio=1") != NULL);
        expect_ok("te live", strstr(buf, "te_path=live") != NULL);
        expect_ok("sdpa_graph=0", strstr(buf, "sdpa_graph=0") != NULL);
        expect_ok("te cache env",
                  strstr(buf, "env H3_TE_PROMPT_CACHE=") != NULL);
        expect_ok("te require_hit env",
                  strstr(buf, "env H3_TE_CACHE_REQUIRE_HIT=") != NULL);
        expect_ok("lifecycle trace env",
                  strstr(buf, "env H3_LIFECYCLE_TRACE=") != NULL);
        expect_ok("dump latent env",
                  strstr(buf, "env H3_DUMP_LATENT=") != NULL);
    }

    setenv("H3_EXPECT_P0_VISUAL", "1", 1);
    expect_ok("expect flag", h3_env_contract_expect_p0_visual() == 1);

    if (g_fail) {
        fprintf(stderr, "test_h3_env_contract: FAILED\n");
        return 1;
    }
    printf("ok test_h3_env_contract\n");
    return 0;
}
