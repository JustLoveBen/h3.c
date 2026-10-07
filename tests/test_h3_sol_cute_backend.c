/* tests/test_h3_sol_cute_backend.c
 * CPU-only: official NVlabs sol-engine CuTe names vs fake/triton fallback.
 * Source of names: techniques/sparse_backends/sol_attn/interface.py
 *   (8,9)->cute_sm89 (9,0)->cute_sm90 (10,0)->cute_sm100 (12,0)->cute_sm120
 * Missing CuTe runtime returns "triton" — that is NOT a product kernel.
 */
#include "h3_sdpa.h"

#include <stdio.h>
#include <stdlib.h>

static int fails;
static void expect(int c, const char *m) {
    if (!c) {
        fprintf(stderr, "FAIL %s\n", m);
        fails++;
    }
}

int main(void) {
    expect(h3_sdpa_sol_cute_is_official("cute_sm89") == 1, "sm89 CuTe");
    expect(h3_sdpa_sol_cute_is_official("cute_sm90") == 1, "sm90 CuTe");
    expect(h3_sdpa_sol_cute_is_official("cute_sm100") == 1, "sm100 CuTe");
    expect(h3_sdpa_sol_cute_is_official("cute_sm120") == 1, "sm120 CuTe 5090");

    expect(h3_sdpa_sol_cute_is_official("triton") == 0, "triton fallback fake");
    expect(h3_sdpa_sol_cute_is_official("triton_ref") == 0, "triton_ref fake");
    expect(h3_sdpa_sol_cute_is_official("nested_loop") == 0, "nested_loop fake");
    expect(h3_sdpa_sol_cute_is_official("metal") == 0, "metal not 5090");
    expect(h3_sdpa_sol_cute_is_official("sage3") == 0, "sage3 is other plugin");
    expect(h3_sdpa_sol_cute_is_official("") == 0, "empty");
    expect(h3_sdpa_sol_cute_is_official(NULL) == 0, "null");
    expect(h3_sdpa_sol_cute_is_official("cute_sm121") == 0, "unknown cute");

    {
        size_t qkvo = h3_sol_cute_qkvo_bytes(39172, 56, 128);
        size_t f16 = h3_sol_cute_f16_qkv_scratch_bytes(39172, 56, 128);
        size_t cudnn4 = 4ull * 39172ull * 56ull * 128ull * 2ull;
        expect(qkvo == 2246279168ull, "hires QKVO 4*BF16");
        expect(f16 == 1684709376ull, "hires F16 QKV scratch 3*");
        expect(f16 < cudnn4, "sol scratch < cudnn 4*padded-class");
        expect(qkvo - f16 == 561569792ull, "QKVO is +1 tensor vs QKV");
    }

    unsetenv("H3_SOL_MAX_SEQ");
    expect(h3_sol_cute_max_seq() == 0, "unset max_seq = no cap");
    expect(h3_sol_cute_seq_allowed(10756) == 1, "low-res allowed");
    expect(h3_sol_cute_seq_allowed(39172) == 1, "hires allowed by default");
    setenv("H3_SOL_MAX_SEQ", "16000", 1);
    expect(h3_sol_cute_max_seq() == 16000, "max_seq 16000");
    expect(h3_sol_cute_seq_allowed(10756) == 1, "mixed keeps low-res");
    expect(h3_sol_cute_seq_allowed(39172) == 0, "mixed skips hires");
    setenv("H3_SOL_MAX_SEQ", "0", 1);
    expect(h3_sol_cute_seq_allowed(39172) == 1, "0 means no cap");
    unsetenv("H3_SOL_MAX_SEQ");

    return fails ? 1 : (printf("test_h3_sol_cute_backend: PASS\n"), 0);
}
