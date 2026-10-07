/* tests/test_h3_sage3_require.c
 * CPU-only: accel sage3 want>0 && hits==0 hard-fails unless
 * H3_SAGE3_REQUIRE_HIT=0 (legacy fallback). */
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
    unsetenv("H3_SAGE3_REQUIRE_HIT");
    expect(h3_sdpa_sage3_require_hit(10, 0) == 1, "default hard-fail");
    expect(h3_sdpa_sage3_require_hit(0, 0) == 0, "want=0 gold/cudnn ok");
    expect(h3_sdpa_sage3_require_hit(10, 1) == 0, "any hit ok");
    expect(h3_sdpa_sage3_require_hit(200, 0) == 1, "768p-like miss");

    setenv("H3_SAGE3_REQUIRE_HIT", "0", 1);
    expect(h3_sdpa_sage3_require_hit(10, 0) == 0, "env 0 allows fallback");
    expect(h3_sdpa_sage3_require_hit(200, 0) == 0, "env 0 768p fallback");

    setenv("H3_SAGE3_REQUIRE_HIT", "1", 1);
    expect(h3_sdpa_sage3_require_hit(10, 0) == 1, "env 1 still hard-fail");
    expect(h3_sdpa_sage3_require_hit(4, 4) == 0, "env 1 hits==want ok");

    unsetenv("H3_SAGE3_REQUIRE_HIT");

    unsetenv("H3_SDPA_STRICT");
    expect(h3_sdpa_strict() == 1, "strict default on");
    expect(h3_sdpa_vae_batched_fail_hard() == 1, "vae batched default hard");
    setenv("H3_SDPA_STRICT", "0", 1);
    expect(h3_sdpa_strict() == 0, "strict 0");
    expect(h3_sdpa_vae_batched_fail_hard() == 0, "vae batched allow loop");
    setenv("H3_SDPA_STRICT", "1", 1);
    expect(h3_sdpa_vae_batched_fail_hard() == 1, "strict 1 hard");
    unsetenv("H3_SDPA_STRICT");
    expect(h3_sdpa_sage3_require_hit(10, 0) == 1, "sage3 hard-fail unchanged");
    return fails ? 1 : (printf("test_h3_sage3_require: PASS\n"), 0);
}
