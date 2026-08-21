/* P4: single linear W4A8 dequant + FP32 GEMM probe (CPU).
 * Build:
 *   make h3_w4a8_matmul_probe
 *   ./h3_w4a8_matmul_probe [--block N] [--suffix out|qkv|fc1|fc2] \
 *       [--batch M] [--no-convrot] [--check-act-convrot] FILE.safetensors
 *
 * --check-act-convrot: compare weight-undo GEMM vs act-ConvRot @ W_rot GEMM.
 */
#include "h3_w4a8_dequant.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
    /* Map to (-1, 1) deterministically. */
    const int32_t signed_bits = (int32_t)(h >> 32);
    return (float)signed_bits * (1.0f / 2147483648.0f);
}

static uint64_t checksum_f32(const float *data, size_t count) {
    uint64_t acc = 0x9e3779b97f4a7c15ULL;
    for (size_t i = 0; i < count; i++) {
        uint32_t bits;
        memcpy(&bits, &data[i], sizeof(bits));
        acc = mix64(acc ^ bits ^ (uint64_t)i);
    }
    return acc;
}

static void stats_f32(const float *data, size_t count, float *min_out,
                      float *max_out, float *absmax_out, double *sum_out,
                      size_t *nan_out, size_t *inf_out) {
    float min_v = data[0];
    float max_v = data[0];
    float absmax = 0.0f;
    double sum = 0.0;
    size_t nans = 0;
    size_t infs = 0;
    for (size_t i = 0; i < count; i++) {
        const float v = data[i];
        if (isnan(v)) {
            nans++;
            continue;
        }
        if (isinf(v)) {
            infs++;
            continue;
        }
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
        const float a = fabsf(v);
        if (a > absmax) absmax = a;
        sum += (double)v;
    }
    *min_out = min_v;
    *max_out = max_v;
    *absmax_out = absmax;
    *sum_out = sum;
    *nan_out = nans;
    *inf_out = infs;
}

static const char *resolve_suffix(const char *name) {
    if (!name || !strcmp(name, "out") || !strcmp(name, "out_proj"))
        return H3_W4A8_SUFFIX_OUT;
    if (!strcmp(name, "qkv") || !strcmp(name, "qkv_proj"))
        return H3_W4A8_SUFFIX_QKV;
    if (!strcmp(name, "fc1")) return H3_W4A8_SUFFIX_FC1;
    if (!strcmp(name, "fc2")) return H3_W4A8_SUFFIX_FC2;
    return NULL;
}

static int float_equal_bits(const float *a, const float *b, size_t count) {
    for (size_t i = 0; i < count; i++) {
        uint32_t ba, bb;
        memcpy(&ba, &a[i], sizeof(ba));
        memcpy(&bb, &b[i], sizeof(bb));
        if (ba != bb) return 0;
    }
    return 1;
}

int main(int argc, char **argv) {
    unsigned block = 0;
    const char *suffix_arg = "out";
    size_t batch = 4;
    int apply_convrot = 1;
    int check_act_convrot = 0;
    float atol = 1e-3f;
    float rtol = 1e-4f;
    const char *path = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--block") && i + 1 < argc) {
            block = (unsigned)strtoul(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--suffix") && i + 1 < argc) {
            suffix_arg = argv[++i];
        } else if (!strcmp(argv[i], "--batch") && i + 1 < argc) {
            batch = (size_t)strtoul(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--no-convrot")) {
            apply_convrot = 0;
        } else if (!strcmp(argv[i], "--check-act-convrot")) {
            check_act_convrot = 1;
        } else if (!strcmp(argv[i], "--atol") && i + 1 < argc) {
            atol = strtof(argv[++i], NULL);
        } else if (!strcmp(argv[i], "--rtol") && i + 1 < argc) {
            rtol = strtof(argv[++i], NULL);
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            fprintf(stderr,
                    "usage: %s [--block N] [--suffix out|qkv|fc1|fc2] "
                    "[--batch M] [--no-convrot] [--check-act-convrot] "
                    "[--atol A] [--rtol R] FILE.safetensors\n",
                    argv[0]);
            return 0;
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "unknown option: %s\n", argv[i]);
            return 2;
        } else {
            path = argv[i];
        }
    }
    if (!path || batch == 0) {
        fprintf(stderr,
                "usage: %s [--block N] [--suffix out|qkv|fc1|fc2] "
                "[--batch M] [--no-convrot] [--check-act-convrot] "
                "FILE.safetensors\n",
                argv[0]);
        return 2;
    }

    const char *suffix = resolve_suffix(suffix_arg);
    if (!suffix) {
        fprintf(stderr, "unknown suffix: %s\n", suffix_arg);
        return 2;
    }

    char error[512];
    h3_w4a8_store *store = h3_w4a8_store_open(path, error, sizeof(error));
    if (!store) {
        fprintf(stderr, "open failed: %s\n", error);
        return 1;
    }

    const h3_st_quant_info *quant = h3_w4a8_store_quant(store);
    int group_size = quant->group_size > 0 ? quant->group_size
                                           : H3_W4A8_DEFAULT_GROUP_SIZE;
    int convrot_gs = quant->convrot_groupsize > 0
                         ? quant->convrot_groupsize
                         : H3_W4A8_DEFAULT_CONVROT_GROUPSIZE;

    h3_w4a8_quad quad;
    if (!h3_w4a8_find_block_linear(store, block, suffix, &quad, error,
                                   sizeof(error))) {
        fprintf(stderr, "find linear failed: %s\n", error);
        h3_w4a8_store_free(store);
        return 1;
    }

    printf("path: %s\n", path);
    printf("linear: blocks.%u.%s\n", block, suffix);
    printf("format: %s\n", quant->format[0] ? quant->format : "(unknown)");
    printf("group_size: %d\n", group_size);
    printf("convrot_groupsize: %d\n", convrot_gs);
    printf("apply_weight_convrot_undo: %d\n", apply_convrot);
    printf("packed_shape: [%llu,%llu] (logical K=%llu)\n",
           (unsigned long long)quad.weight->shape[0],
           (unsigned long long)quad.weight->shape[1],
           (unsigned long long)quad.weight->shape[1] * 2ULL);

    clock_t t0 = clock();
    float *weight = NULL;
    size_t n = 0;
    size_t k = 0;
    if (!h3_w4a8_load_linear_f32(store, &quad, group_size, convrot_gs,
                                 apply_convrot, &weight, &n, &k, error,
                                 sizeof(error))) {
        fprintf(stderr, "dequant failed: %s\n", error);
        h3_w4a8_store_free(store);
        return 1;
    }
    clock_t t1 = clock();
    const double dequant_ms =
        1000.0 * (double)(t1 - t0) / (double)CLOCKS_PER_SEC;

    float wmin, wmax, wabs;
    double wsum;
    size_t wnan, winf;
    stats_f32(weight, n * k, &wmin, &wmax, &wabs, &wsum, &wnan, &winf);
    printf("dequant: N=%zu K=%zu time_ms=%.2f\n", n, k, dequant_ms);
    printf("weight_stats: min=%.6g max=%.6g absmax=%.6g sum=%.6g nan=%zu "
           "inf=%zu checksum=%016llx\n",
           wmin, wmax, wabs, wsum, wnan, winf,
           (unsigned long long)checksum_f32(weight, n * k));

    if (wnan || winf) {
        fprintf(stderr, "FAIL: dequant produced non-finite weights\n");
        free(weight);
        h3_w4a8_store_free(store);
        return 1;
    }

    float *x = (float *)malloc(batch * k * sizeof(float));
    float *y1 = (float *)malloc(batch * n * sizeof(float));
    float *y2 = (float *)malloc(batch * n * sizeof(float));
    if (!x || !y1 || !y2) {
        fprintf(stderr, "out of memory for GEMM buffers\n");
        free(x);
        free(y1);
        free(y2);
        free(weight);
        h3_w4a8_store_free(store);
        return 1;
    }

    const uint64_t seed = 0x48324d41544d4f4cULL; /* 'H3MATMUL' */
    for (size_t i = 0; i < batch * k; i++) {
        x[i] = hash_to_unit(seed, i);
    }

    t0 = clock();
    h3_w4a8_gemm_f32(x, weight, batch, n, k, y1);
    t1 = clock();
    const double gemm1_ms =
        1000.0 * (double)(t1 - t0) / (double)CLOCKS_PER_SEC;

    t0 = clock();
    h3_w4a8_gemm_f32(x, weight, batch, n, k, y2);
    t1 = clock();
    const double gemm2_ms =
        1000.0 * (double)(t1 - t0) / (double)CLOCKS_PER_SEC;

    float ymin, ymax, yabs;
    double ysum;
    size_t ynan, yinf;
    stats_f32(y1, batch * n, &ymin, &ymax, &yabs, &ysum, &ynan, &yinf);

    const int identical = float_equal_bits(y1, y2, batch * n);
    printf("gemm: batch=%zu out_features=%zu gemm1_ms=%.2f gemm2_ms=%.2f\n",
           batch, n, gemm1_ms, gemm2_ms);
    printf("output_stats: min=%.6g max=%.6g absmax=%.6g sum=%.6g nan=%zu "
           "inf=%zu checksum=%016llx\n",
           ymin, ymax, yabs, ysum, ynan, yinf,
           (unsigned long long)checksum_f32(y1, batch * n));
    printf("self_check_bit_identical: %s\n", identical ? "PASS" : "FAIL");
    printf("self_check_finite: %s\n",
           (ynan == 0 && yinf == 0) ? "PASS" : "FAIL");

    /* Second dequant must match first (deterministic). */
    float *weight_b = NULL;
    size_t n2 = 0, k2 = 0;
    int dequant_ok = h3_w4a8_load_linear_f32(store, &quad, group_size, convrot_gs,
                                             apply_convrot, &weight_b, &n2, &k2,
                                             error, sizeof(error));
    int weight_identical = 0;
    if (dequant_ok && n2 == n && k2 == k) {
        weight_identical = float_equal_bits(weight, weight_b, n * k);
    }
    printf("dequant_repeat_bit_identical: %s\n",
           weight_identical ? "PASS" : "FAIL");

    int act_pass = 1;
    if (check_act_convrot) {
        if (!apply_convrot) {
            fprintf(stderr,
                    "FAIL: --check-act-convrot requires weight undo "
                    "(omit --no-convrot)\n");
            act_pass = 0;
        } else {
            float *w_rot = NULL;
            size_t nr = 0, kr = 0;
            float *x_rot = (float *)malloc(batch * k * sizeof(float));
            float *y_act = (float *)malloc(batch * n * sizeof(float));
            int rot_ok = h3_w4a8_load_linear_f32(store, &quad, group_size,
                                                 convrot_gs, 0, &w_rot, &nr,
                                                 &kr, error, sizeof(error));
            int act_ok = 0;
            float max_abs = 0.0f;
            double rmse = 0.0;
            if (!x_rot || !y_act || !rot_ok || nr != n || kr != k) {
                fprintf(stderr, "act ConvRot setup failed: %s\n",
                        rot_ok ? "alloc/shape" : error);
                act_pass = 0;
            } else if (!h3_w4a8_convrot_act_f32(x, batch, k, convrot_gs, x_rot,
                                                error, sizeof(error))) {
                fprintf(stderr, "act ConvRot failed: %s\n", error);
                act_pass = 0;
            } else {
                h3_w4a8_gemm_f32(x_rot, w_rot, batch, n, k, y_act);
                h3_w4a8_compare_f32(y1, y_act, batch * n, &max_abs, &rmse);
                const float thresh =
                    atol + rtol * (yabs > 0.0f ? yabs : 1.0f);
                act_ok = max_abs <= thresh;
                act_pass = act_ok;
                printf("act_convrot_vs_weight_undo: max_abs=%.6g rmse=%.6g "
                       "thresh=%.6g %s\n",
                       max_abs, rmse, thresh, act_ok ? "PASS" : "FAIL");
            }
            free(w_rot);
            free(x_rot);
            free(y_act);
        }
    }

    const int pass = identical && ynan == 0 && yinf == 0 && weight_identical &&
                     dequant_ok && act_pass;
    printf("summary: %s\n", pass ? "PASS" : "FAIL");

    free(weight_b);
    free(x);
    free(y1);
    free(y2);
    free(weight);
    h3_w4a8_store_free(store);
    return pass ? 0 : 1;
}
