/* Host-only: print shared H3 env contract block (C061). */
#include "h3_env_contract.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    h3_env_contract_cli cli;
    const char *arm = "int8";
    int expect = 0;
    int i;
    memset(&cli, 0, sizeof(cli));
    cli.seed = 0x483343554441ULL;
    cli.skip_audio = 0;
    cli.fewstep_preset = "none";
    cli.fewstep_contract = "none";
    cli.sampler = "velocity_euler";
    cli.sdpa = "cudnn";
    cli.arm = arm;

    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--expect-p0-visual")) {
            expect = 1;
            setenv("H3_EXPECT_P0_VISUAL", "1", 1);
        } else if (!strcmp(argv[i], "--arm") && i + 1 < argc) {
            arm = argv[++i];
            cli.arm = arm;
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            fprintf(stderr,
                    "usage: %s [--expect-p0-visual] [--arm int8|nvfp4]\n"
                    "  Print # H3_ENV_CONTRACT (same fields as generate).\n"
                    "  Python twin: tools/print_h3_env_contract.py\n",
                    argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown arg: %s\n", argv[i]);
            return 2;
        }
    }
    (void)expect;
    h3_env_contract_fprint(stdout, &cli);
    if (h3_env_contract_expect_p0_visual()) {
        char err[512];
        if (!h3_env_contract_check_p0(arm, err, sizeof(err))) {
            fprintf(stderr, "P0 visual contract violations: %s\n", err);
            return 1;
        }
    }
    return 0;
}
