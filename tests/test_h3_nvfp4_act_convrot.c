/* Host algebraic proof: ConvRot-baked W_rot requires X@H before GEMM.
 * Mirrors the NVFP4 mosaic root cause (missing act ConvRot) vs the INT8
 * / fixed NVFP4 path. No CUDA. */
#include "h3_w4a8_dequant.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;

static void expect(int c, const char *m) {
    if (!c) {
        fprintf(stderr, "FAIL %s\n", m);
        fails++;
    } else {
        printf("ok %s\n", m);
    }
}

static void fill_rand(float *a, size_t n, unsigned seed) {
    size_t i;
    unsigned s = seed;
    for (i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        a[i] = ((float)(s & 0xffff) / 65535.0f) * 2.0f - 1.0f;
    }
}

/* Y = X @ W_rot^T  (broken NVFP4) vs Y = (X@H) @ W_rot^T (correct).
 * Also compare correct path to unrotated W = W_rot @ H  (equivalence). */
static void test_equivalence(void) {
    const size_t m = 4, n = 8, k = 256; /* gs=256, one group */
    const int gs = 256;
    float *x, *w_rot, *w, *x_rot, *y_broken, *y_fixed, *y_ref;
    float max_broken, max_fixed;
    double rmse_broken, rmse_fixed;
    char err[256];

    x = (float *)malloc(m * k * sizeof(float));
    w_rot = (float *)malloc(n * k * sizeof(float));
    w = (float *)malloc(n * k * sizeof(float));
    x_rot = (float *)malloc(m * k * sizeof(float));
    y_broken = (float *)malloc(m * n * sizeof(float));
    y_fixed = (float *)malloc(m * n * sizeof(float));
    y_ref = (float *)malloc(m * n * sizeof(float));
    expect(x && w_rot && w && x_rot && y_broken && y_fixed && y_ref, "alloc");
    if (!x)
        return;

    fill_rand(x, m * k, 1u);
    fill_rand(w_rot, n * k, 2u);

    /* W = W_rot @ H  (undo weight ConvRot) — same as apply_convrot path. */
    expect(h3_w4a8_convrot_weight_f32(w_rot, n, k, gs, w, err, sizeof(err)) ==
               1,
           "undo W_rot -> W");
    expect(h3_w4a8_convrot_act_f32(x, m, k, gs, x_rot, err, sizeof(err)) == 1,
           "act X -> X_rot");

    /* ref: X @ W^T */
    h3_w4a8_gemm_f32(x, w, m, n, k, y_ref);
    /* fixed: X_rot @ W_rot^T */
    h3_w4a8_gemm_f32(x_rot, w_rot, m, n, k, y_fixed);
    /* broken: X @ W_rot^T  (pre-fix NVFP4) */
    h3_w4a8_gemm_f32(x, w_rot, m, n, k, y_broken);

    expect(h3_w4a8_compare_f32(y_ref, y_fixed, m * n, &max_fixed, &rmse_fixed),
           "compare fixed");
    expect(h3_w4a8_compare_f32(y_ref, y_broken, m * n, &max_broken,
                               &rmse_broken),
           "compare broken");

    printf("fixed  max_abs=%.6g rmse=%.6g\n", max_fixed, rmse_fixed);
    printf("broken max_abs=%.6g rmse=%.6g\n", max_broken, rmse_broken);

    expect(max_fixed < 1e-4f, "fixed ≈ ref (Hadamard involution)");
    expect(max_broken > 0.1f, "broken far from ref (mosaic regime)");
    expect(max_broken > 10.0f * (max_fixed + 1e-8f),
           "broken >> fixed error");

    free(x);
    free(w_rot);
    free(w);
    free(x_rot);
    free(y_broken);
    free(y_fixed);
    free(y_ref);
}

int main(void) {
    test_equivalence();
    if (fails) {
        fprintf(stderr, "%d FAIL(s)\n", fails);
        return 1;
    }
    printf("ALL PASS\n");
    return 0;
}
