/* Host-only: print shared --fewstep-* / --sampler help block. */
#include "h3_fewstep_contract.h"

#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    int i;
    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--json") || !strcmp(argv[i], "-j")) {
            h3_fewstep_contracts_json_fprint(stdout, 1);
            return 0;
        }
        if (!strcmp(argv[i], "--json-compact")) {
            h3_fewstep_contracts_json_fprint(stdout, 0);
            return 0;
        }
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            fprintf(stderr,
                    "usage: %s [--json|--json-compact]\n"
                    "  (default) print --fewstep-* / --sampler help text\n"
                    "  --json           pretty contract table JSON\n"
                    "  --json-compact   one-line JSON\n",
                    argv[0]);
            return 0;
        }
        fprintf(stderr, "unknown arg: %s\n", argv[i]);
        return 2;
    }
    h3_fewstep_help_fprint(stdout);
    return 0;
}
