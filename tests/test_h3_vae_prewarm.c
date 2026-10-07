/* CPU-only: H3_GEN_VAE_PREWARM default on; 0 disables. */
#include "h3_vae_prewarm.h"

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
    unsetenv("H3_GEN_VAE_PREWARM");
    expect(h3_vae_prewarm_wanted() == 1, "default on");
    setenv("H3_GEN_VAE_PREWARM", "0", 1);
    expect(h3_vae_prewarm_wanted() == 0, "env 0 off");
    setenv("H3_GEN_VAE_PREWARM", "1", 1);
    expect(h3_vae_prewarm_wanted() == 1, "env 1 on");
    unsetenv("H3_GEN_VAE_PREWARM");
    expect(h3_vae_prewarm_wanted() == 1, "unset back on");
    return fails ? 1 : (printf("test_h3_vae_prewarm: PASS\n"), 0);
}
