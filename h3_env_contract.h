#ifndef H3_ENV_CONTRACT_H
#define H3_ENV_CONTRACT_H

#include <stddef.h>
#include <stdio.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Effective H3_* env + CLI contract dump (C061).
 * Mirrors tools/print_h3_env_contract.py CONTRACT + P0 rules.
 * Offline / host-safe: no CUDA.
 *
 * Log marker: lines under "# H3_ENV_CONTRACT" recognized by
 * tools/p0_protocol_check_log.py.
 */

typedef struct h3_env_contract_cli {
    uint64_t seed;
    int skip_audio; /* 1 if --skip-audio */
    const char *fewstep_preset;   /* may be NULL */
    const char *fewstep_contract; /* resolved name or NULL/"none" */
    const char *sampler;          /* may be NULL */
    const char *sdpa;             /* CLI/backend name; may be NULL */
    /* arm hint for P0 check: "int8" (default) or "nvfp4" */
    const char *arm;
} h3_env_contract_cli;

/* Print structured contract block to out (default stdout). */
void h3_env_contract_fprint(FILE *out, const h3_env_contract_cli *cli);

/*
 * P0 visual rules (same semantics as Python --expect-p0-visual).
 * arm: "int8" | "nvfp4" (NULL → int8).
 * Returns 1 ok, 0 fail (err set when provided).
 */
int h3_env_contract_check_p0(const char *arm, char *err, size_t err_size);

/* True when H3_EXPECT_P0_VISUAL=1 (first char). */
int h3_env_contract_expect_p0_visual(void);

#ifdef __cplusplus
}
#endif

#endif /* H3_ENV_CONTRACT_H */
