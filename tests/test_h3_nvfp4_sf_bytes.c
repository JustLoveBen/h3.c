/* Host-only: NVFP4 SF byte formula must pad ncb to 32 tile-cols.
 * K=5376 → ncb=84 → ncb_rd=96 (the mosaic bug: create used unpadded 84).
 *   clang -std=c11 -O2 -I. -o /tmp/t_sf tests/test_h3_nvfp4_sf_bytes.c \\
 *         -DTEST_H3_NVFP4_SF_BYTES_STANDALONE
 */
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>

#ifndef TEST_H3_NVFP4_SF_BYTES_STANDALONE
#include "h3_gpu_cuda.h"
#else
/* Mirror of h3_cuda_nvfp4_sf_bytes for host-only proof without CUDA link. */
static size_t h3_cuda_nvfp4_sf_bytes(size_t rows, size_t k) {
    size_t ncb, ncb_rd, rb;
    if (rows == 0 || k == 0 || (k & 63u) != 0) return 0;
    ncb = (k / 16u) / 4u;
    ncb_rd = (ncb + 31u) & ~(size_t)31u;
    rb = (rows + 127u) / 128u;
    return rb * ncb_rd * 512u;
}
#endif

static size_t unpadded(size_t rows, size_t k) {
    size_t ncb = (k / 16u) / 4u;
    size_t rb = (rows + 127u) / 128u;
    return rb * ncb * 512u;
}

int main(void) {
    struct { const char *name; size_t n, k; size_t want_pad; int must_differ; } cases[] = {
        {"qkv", 21504, 5376, 8257536, 1},
        {"out", 5376, 5376, 2064384, 1},
        {"fc1", 28672, 5376, 11010048, 1},
        {"fc2", 5376, 14336, 4816896, 0},
    };
    int i, fails = 0;
    for (i = 0; i < 4; i++) {
        size_t got = h3_cuda_nvfp4_sf_bytes(cases[i].n, cases[i].k);
        size_t bad = unpadded(cases[i].n, cases[i].k);
        if (got != cases[i].want_pad) {
            printf("FAIL %s: got %zu want %zu\n", cases[i].name, got, cases[i].want_pad);
            fails++;
        } else if (cases[i].must_differ && got == bad) {
            printf("FAIL %s: padded==unpadded (%zu) — mosaic bug not gated\n",
                   cases[i].name, got);
            fails++;
        } else if (!cases[i].must_differ && got != bad) {
            printf("FAIL %s: expected no pad but padded=%zu unpadded=%zu\n",
                   cases[i].name, got, bad);
            fails++;
        } else {
            printf("ok %s sf_bytes=%zu (unpadded would be %zu)\n",
                   cases[i].name, got, bad);
        }
    }
    if (h3_cuda_nvfp4_sf_bytes(1, 63) != 0 || h3_cuda_nvfp4_sf_bytes(0, 64) != 0) {
        printf("FAIL: bad-args should return 0\n");
        fails++;
    }
    printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
