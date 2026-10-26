/* Host-only lock for h3_cuda_nvfp4_sf_bytes: rb * ncb * 512 (stride=ncb).
 * K=5376 → ncb=84 (NOT ncb_rd=96). e2e on sm_120 rejected ceil32 row-stride.
 */
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>

static size_t h3_cuda_nvfp4_sf_bytes(size_t rows, size_t k) {
    size_t ncb, rb;
    if (rows == 0 || k == 0 || (k & 63u) != 0) return 0;
    ncb = (k / 16u) / 4u;
    rb = (rows + 127u) / 128u;
    return rb * ncb * 512u;
}

int main(void) {
    struct { const char *name; size_t n, k; size_t want; } cases[] = {
        {"qkv K=5376 N=21504", 21504, 5376, 168ull * 84ull * 512ull},
        {"fc2 K=14336 N=5376", 5376, 14336, 42ull * 224ull * 512ull},
        {"tiny K=64 N=1", 1, 64, 1ull * 1ull * 512ull},
        {"K=512 N=256", 256, 512, 2ull * 8ull * 512ull},
    };
    int fail = 0;
    for (size_t i = 0; i < sizeof(cases)/sizeof(cases[0]); i++) {
        size_t got = h3_cuda_nvfp4_sf_bytes(cases[i].n, cases[i].k);
        if (got != cases[i].want) {
            printf("FAIL %s: got %zu want %zu\n", cases[i].name, got, cases[i].want);
            fail = 1;
        } else {
            printf("ok %s sf_bytes=%zu\n", cases[i].name, got);
        }
    }
    if (h3_cuda_nvfp4_sf_bytes(1, 63) != 0 || h3_cuda_nvfp4_sf_bytes(0, 64) != 0) {
        printf("FAIL: bad-args should return 0\n");
        fail = 1;
    }
    return fail;
}
