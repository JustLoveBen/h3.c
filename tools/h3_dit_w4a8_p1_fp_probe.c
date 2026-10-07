/* P1 FP verification probe (NVIDIA Sol-Engine Pillar-1 MAE gates).
 *
 * This repo's DiT QKV is ALREADY one merged wide GEMM (weight [3*INNER,
 * HIDDEN]) and QKNorm+RoPE is ALREADY a single fused kernel — the missing
 * P1 piece is the FP math-equivalence gate.  This probe measures:
 *
 *   1) MAE(merged wide GEMM, 3 narrow split GEMMs)  — expected 0 in FP32.
 *   2) MAE(fused QKNorm+RoPE, unfused two-pass reference) — expected ~1e-7.
 *
 * Gates (T2 promotion criteria from the dev plan): qkv MAE <= 1e-6,
 * fused-vs-unfused MAE <= 1e-5 (looser than the 1e-6 target because the
 * unfused reference rounds the normalized intermediate once extra).
 *
 * No GPU needed (pure host F32).  INT8 dequant associativity is evaluated
 * separately on 238 — H3_DIT_QKV_MERGE / H3_DIT_QKNORM_ROPE_FUSE stay
 * EXPERIMENTAL T1-only until then.
 *
 * Build:
 *   cmake --build build-cuda -j --target h3_dit_w4a8_p1_fp_probe
 *   ./build-cuda/h3_dit_w4a8_p1_fp_probe [--seq N] [--verbose]
 */
#include "h3_dit_w4a8.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

static float hash_to_unit(uint64_t seed, size_t index) {
    uint64_t h = mix64(seed ^ (uint64_t)(index + 1) * 0x9e3779b97f4a7c15ULL);
    const int32_t signed_bits = (int32_t)(h >> 32);
    return (float)signed_bits * (1.0f / 2147483648.0f);
}

static void fill(float *x, size_t n, uint64_t seed) {
    size_t i;
    for (i = 0; i < n; i++) x[i] = hash_to_unit(seed, i);
}

static void fill_positive(float *x, size_t n, uint64_t seed) {
    size_t i;
    for (i = 0; i < n; i++) x[i] = 0.5f + 0.5f * fabsf(hash_to_unit(seed, i));
}

static double mae_f32(const float *a, const float *b, size_t n) {
    double acc = 0.0;
    size_t i;
    for (i = 0; i < n; i++) acc += (double)fabsf(a[i] - b[i]);
    return n ? acc / (double)n : 0.0;
}

static float max_abs_f32(const float *a, const float *b, size_t n) {
    float m = 0.0f;
    size_t i;
    for (i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

/* One wide merged GEMM: y[m,3n] = x[m,k] @ wqkv[3n,k]^T (same K summation
 * order as the split3 reference so the only difference is loop layout). */
static void gemm_wide(const float *x, const float *wqkv, size_t m, size_t n,
                      size_t k, float *y) {
    size_t r, o, c;
    for (r = 0; r < m; r++) {
        const float *xr = x + r * k;
        float *yr = y + r * n * 3;
        for (o = 0; o < n * 3; o++) {
            const float *wrow = wqkv + o * k;
            float acc = 0.0f;
            for (c = 0; c < k; c++) acc += xr[c] * wrow[c];
            yr[o] = acc;
        }
    }
}

int main(int argc, char **argv) {
    const size_t heads = H3_DIT_W4A8_HEADS;
    const size_t head_dim = H3_DIT_W4A8_HEAD_DIM;
    const size_t inner = heads * head_dim;
    const size_t hidden = H3_DIT_W4A8_HIDDEN;
    const size_t rope_half = H3_DIT_W4A8_ROPE_HALF;
    const float eps = 1e-5f;
    size_t seq = 4;
    int verbose = 0;
    int i;
    char err[256];
    float *x = NULL, *wqkv = NULL, *y_wide = NULL, *y_split = NULL;
    float *q_norm = NULL, *k_norm = NULL, *rcos = NULL, *rsin = NULL;
    float *q_f = NULL, *k_f = NULL, *v_f = NULL;
    float *q_u = NULL, *k_u = NULL, *v_u = NULL;
    double mae_qkv, mae_q, mae_k, mae_v;
    float mx_qkv, mx_q, mx_k;
    int pass;

    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--seq") && i + 1 < argc) {
            seq = (size_t)strtoul(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--verbose")) {
            verbose = 1;
        }
    }

    x = (float *)malloc(seq * hidden * sizeof(float));
    wqkv = (float *)malloc(inner * 3 * hidden * sizeof(float));
    y_wide = (float *)malloc(seq * inner * 3 * sizeof(float));
    y_split = (float *)malloc(seq * inner * 3 * sizeof(float));
    q_norm = (float *)malloc(head_dim * sizeof(float));
    k_norm = (float *)malloc(head_dim * sizeof(float));
    rcos = (float *)malloc(seq * rope_half * sizeof(float));
    rsin = (float *)malloc(seq * rope_half * sizeof(float));
    q_f = (float *)malloc(seq * inner * sizeof(float));
    k_f = (float *)malloc(seq * inner * sizeof(float));
    v_f = (float *)malloc(seq * inner * sizeof(float));
    q_u = (float *)malloc(seq * inner * sizeof(float));
    k_u = (float *)malloc(seq * inner * sizeof(float));
    v_u = (float *)malloc(seq * inner * sizeof(float));
    if (!x || !wqkv || !y_wide || !y_split || !q_norm || !k_norm || !rcos ||
        !rsin || !q_f || !k_f || !v_f || !q_u || !k_u || !v_u) {
        fprintf(stderr, "probe: OOM\n");
        return 2;
    }

    fill(x, seq * hidden, 0x1234);
    fill(wqkv, inner * 3 * hidden, 0xbeef);
    /* Scale weights down so FP32 sums stay well-conditioned. */
    {
        size_t j, n = inner * 3 * hidden;
        for (j = 0; j < n; j++) wqkv[j] *= 0.05f;
    }
    fill_positive(q_norm, head_dim, 0x77);
    fill_positive(k_norm, head_dim, 0x88);
    fill(rcos, seq * rope_half, 0x99);
    fill(rsin, seq * rope_half, 0xaa);
    /* Normalize cos/sin to a unit rotation per row. */
    {
        size_t r, d;
        for (r = 0; r < seq; r++) {
            for (d = 0; d < rope_half; d++) {
                float c = rcos[r * rope_half + d];
                float s = rsin[r * rope_half + d];
                float nrm = sqrtf(c * c + s * s);
                if (nrm < 1e-6f) nrm = 1.0f;
                rcos[r * rope_half + d] = c / nrm;
                rsin[r * rope_half + d] = s / nrm;
            }
        }
    }

    /* Gate 1: merged wide GEMM vs 3 narrow GEMMs. */
    gemm_wide(x, wqkv, seq, inner, hidden, y_wide);
    if (!h3_dit_w4a8_fp_qkv_split3_f32(x, wqkv, seq, inner, hidden, y_split,
                                       err, sizeof(err))) {
        fprintf(stderr, "probe: split3 failed: %s\n", err);
        return 2;
    }
    mae_qkv = mae_f32(y_wide, y_split, seq * inner * 3);
    mx_qkv = max_abs_f32(y_wide, y_split, seq * inner * 3);

    /* Gate 2: fused QKNorm+RoPE vs unfused two-pass reference. */
    if (!h3_dit_w4a8_qkv_rope_f32(y_wide, q_norm, k_norm, rcos, rsin, seq,
                                  heads, head_dim, rope_half, eps, q_f, k_f,
                                  v_f, err, sizeof(err))) {
        fprintf(stderr, "probe: fused qkv_rope failed: %s\n", err);
        return 2;
    }
    if (!h3_dit_w4a8_fp_qknorm_rope_unfused_f32(
            y_wide, q_norm, k_norm, rcos, rsin, seq, heads, head_dim,
            rope_half, eps, q_u, k_u, v_u, err, sizeof(err))) {
        fprintf(stderr, "probe: unfused qknorm_rope failed: %s\n", err);
        return 2;
    }
    mae_q = mae_f32(q_f, q_u, seq * inner);
    mae_k = mae_f32(k_f, k_u, seq * inner);
    mae_v = mae_f32(v_f, v_u, seq * inner);
    mx_q = max_abs_f32(q_f, q_u, seq * inner);
    mx_k = max_abs_f32(k_f, k_u, seq * inner);

    printf("P1 FP MAE gates (seq=%zu, inner=%zu, hidden=%zu)\n", seq, inner,
           hidden);
    printf("  qkv merged-vs-split3 : MAE=%.3e max_abs=%.3e  (gate <=1e-6)\n",
           mae_qkv, mx_qkv);
    printf("  fused-vs-unfused Q   : MAE=%.3e max_abs=%.3e  (gate <=1e-5)\n",
           mae_q, mx_q);
    printf("  fused-vs-unfused K   : MAE=%.3e max_abs=%.3e  (gate <=1e-5)\n",
           mae_k, mx_k);
    printf("  fused-vs-unfused V   : MAE=%.3e (must be 0)\n", mae_v);
    if (verbose) {
        printf("  H3_DIT_QKV_MERGE=%d H3_DIT_QKNORM_ROPE_FUSE=%d\n",
               h3_dit_w4a8_fp_qkv_merge_enabled(),
               h3_dit_w4a8_fp_qknorm_rope_fuse_enabled());
    }

    pass = (mae_qkv <= 1e-6) && (mx_qkv <= 1e-6f) && (mae_q <= 1e-5) &&
           (mae_k <= 1e-5) && (mae_v == 0.0);
    printf("%s\n", pass ? "PASS" : "FAIL");

    free(x);
    free(wqkv);
    free(y_wide);
    free(y_split);
    free(q_norm);
    free(k_norm);
    free(rcos);
    free(rsin);
    free(q_f);
    free(k_f);
    free(v_f);
    free(q_u);
    free(k_u);
    free(v_u);
    return pass ? 0 : 1;
}
