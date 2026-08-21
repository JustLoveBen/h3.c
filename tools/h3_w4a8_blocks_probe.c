/* P4 step-3: multi-block W4A8 smoke + CPU INT8 act/GEMM vs FP32.
 * Build (CPU):
 *   make h3_w4a8_blocks_probe
 *   ./h3_w4a8_blocks_probe [--blocks 0,1,24,49|0-3] [--batch M] \
 *       [--act-quant per-row|per-tensor] [--atol A] [--rtol R] \
 *       [--int8-atol A] [--int8-rtol R] FILE.safetensors
 *
 * Per block × {qkv,out,fc1,fc2} (found linears only):
 *   1) packed → INT8 dequant twice (bit-identical)
 *   2) Y_fp32 = (X@H) @ W_rot.T          (FP32 ref; W_rot keeps INT8*s_channel)
 *   3) Y_i8   = dequant( INT8(X@H) @ W_i8 ) with per-row/tensor act scales
 *   4) Y_qi   = (X_hat) @ W_rot.T where X_hat = X_i8 * act_scale
 *      → Y_i8 ≈ Y_qi (tight); Y_i8 ≈ Y_fp32 (looser INT8 quant tol)
 */
#include "h3_w4a8_dequant.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_BLOCKS 64
#define MAX_SUFFIXES 8

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

static uint64_t checksum_f32(const float *data, size_t count) {
    uint64_t acc = 0x9e3779b97f4a7c15ULL;
    for (size_t i = 0; i < count; i++) {
        uint32_t bits;
        memcpy(&bits, &data[i], sizeof(bits));
        acc = mix64(acc ^ bits ^ (uint64_t)i);
    }
    return acc;
}

static uint64_t checksum_i8(const int8_t *data, size_t count) {
    uint64_t acc = 0x9e3779b97f4a7c15ULL;
    for (size_t i = 0; i < count; i++) {
        acc = mix64(acc ^ (uint64_t)(uint8_t)data[i] ^ (uint64_t)i);
    }
    return acc;
}

static void stats_f32(const float *data, size_t count, float *absmax_out,
                      size_t *nan_out, size_t *inf_out) {
    float absmax = 0.0f;
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
        const float a = fabsf(v);
        if (a > absmax) absmax = a;
    }
    *absmax_out = absmax;
    *nan_out = nans;
    *inf_out = infs;
}

static int within_tol(float max_abs, float ref_absmax, float atol, float rtol) {
    const float thresh = atol + rtol * (ref_absmax > 0.0f ? ref_absmax : 1.0f);
    return max_abs <= thresh;
}

static int parse_blocks(const char *spec, unsigned *out, int *out_count) {
    *out_count = 0;
    if (!spec || !spec[0]) return 0;

    /* range: A-B */
    const char *dash = strchr(spec, '-');
    if (dash && !strchr(spec, ',')) {
        char *end = NULL;
        unsigned a = (unsigned)strtoul(spec, &end, 10);
        if (end != dash) return 0;
        unsigned b = (unsigned)strtoul(dash + 1, &end, 10);
        if (*end != '\0' || b < a) return 0;
        if ((int)(b - a + 1) > MAX_BLOCKS) return 0;
        for (unsigned i = a; i <= b; i++) out[(*out_count)++] = i;
        return 1;
    }

    const char *p = spec;
    while (*p) {
        if (*out_count >= MAX_BLOCKS) return 0;
        char *end = NULL;
        unsigned v = (unsigned)strtoul(p, &end, 10);
        if (end == p) return 0;
        out[(*out_count)++] = v;
        if (*end == '\0') break;
        if (*end != ',') return 0;
        p = end + 1;
    }
    return *out_count > 0;
}

static const char *k_suffixes[] = {
    H3_W4A8_SUFFIX_QKV,
    H3_W4A8_SUFFIX_OUT,
    H3_W4A8_SUFFIX_FC1,
    H3_W4A8_SUFFIX_FC2,
};

static const char *short_suffix(const char *s) {
    if (!strcmp(s, H3_W4A8_SUFFIX_QKV)) return "qkv";
    if (!strcmp(s, H3_W4A8_SUFFIX_OUT)) return "out";
    if (!strcmp(s, H3_W4A8_SUFFIX_FC1)) return "fc1";
    if (!strcmp(s, H3_W4A8_SUFFIX_FC2)) return "fc2";
    return s;
}

typedef struct {
    int pass;
    int fail;
    int skip;
} probe_tally;

static int run_one_linear(const h3_w4a8_store *store, unsigned block,
                          const char *suffix, int group_size, int convrot_gs,
                          size_t batch, h3_w4a8_act_quant_mode act_mode,
                          float atol, float rtol, float int8_atol,
                          float int8_rtol, probe_tally *tally) {
    char error[512];
    h3_w4a8_quad quad;
    if (!h3_w4a8_find_block_linear(store, block, suffix, &quad, error,
                                   sizeof(error))) {
        printf("  blocks.%u.%s: SKIP (%s)\n", block, short_suffix(suffix),
               error);
        tally->skip++;
        return 1; /* missing linear is skip, not hard fail */
    }

    clock_t t0 = clock();
    int8_t *w_i8 = NULL;
    float *s_channel = NULL;
    size_t n = 0, k = 0;
    if (!h3_w4a8_load_linear_int8(store, &quad, group_size, &w_i8, &s_channel,
                                  &n, &k, error, sizeof(error))) {
        printf("  blocks.%u.%s: FAIL load_int8: %s\n", block,
               short_suffix(suffix), error);
        tally->fail++;
        return 0;
    }

    int8_t *w_i8_b = NULL;
    float *s_channel_b = NULL;
    size_t n_b = 0, k_b = 0;
    if (!h3_w4a8_load_linear_int8(store, &quad, group_size, &w_i8_b,
                                  &s_channel_b, &n_b, &k_b, error,
                                  sizeof(error))) {
        printf("  blocks.%u.%s: FAIL load_int8 repeat: %s\n", block,
               short_suffix(suffix), error);
        free(w_i8);
        free(s_channel);
        tally->fail++;
        return 0;
    }
    const int dequant_ok =
        (n_b == n && k_b == k && memcmp(w_i8, w_i8_b, n * k) == 0 &&
         memcmp(s_channel, s_channel_b, n * sizeof(float)) == 0);
    free(w_i8_b);
    free(s_channel_b);
    if (!dequant_ok) {
        printf("  blocks.%u.%s: FAIL dequant_repeat bit-identical\n", block,
               short_suffix(suffix));
        free(w_i8);
        free(s_channel);
        tally->fail++;
        return 0;
    }

    float *w_f32 = (float *)malloc(n * k * sizeof(float));
    float *x = (float *)malloc(batch * k * sizeof(float));
    float *x_rot = (float *)malloc(batch * k * sizeof(float));
    float *y_fp32 = (float *)malloc(batch * n * sizeof(float));
    float *y_i8 = (float *)malloc(batch * n * sizeof(float));
    float *y_qi = (float *)malloc(batch * n * sizeof(float));
    int8_t *x_i8 = (int8_t *)malloc(batch * k);
    float *x_hat = (float *)malloc(batch * k * sizeof(float));
    const size_t n_scales =
        (act_mode == H3_W4A8_ACT_QUANT_PER_TENSOR) ? 1 : batch;
    float *act_scales = (float *)malloc(n_scales * sizeof(float));
    if (!w_f32 || !x || !x_rot || !y_fp32 || !y_i8 || !y_qi || !x_i8 || !x_hat ||
        !act_scales) {
        printf("  blocks.%u.%s: FAIL OOM\n", block, short_suffix(suffix));
        free(w_f32);
        free(x);
        free(x_rot);
        free(y_fp32);
        free(y_i8);
        free(y_qi);
        free(x_i8);
        free(x_hat);
        free(act_scales);
        free(w_i8);
        free(s_channel);
        tally->fail++;
        return 0;
    }

    if (!h3_w4a8_int8_scale_to_f32(w_i8, n, k, s_channel, w_f32, error,
                                   sizeof(error))) {
        printf("  blocks.%u.%s: FAIL scale_to_f32: %s\n", block,
               short_suffix(suffix), error);
        free(w_f32);
        free(x);
        free(x_rot);
        free(y_fp32);
        free(y_i8);
        free(y_qi);
        free(x_i8);
        free(x_hat);
        free(act_scales);
        free(w_i8);
        free(s_channel);
        tally->fail++;
        return 0;
    }

    const uint64_t seed =
        0x48324d41544d4f4cULL ^ ((uint64_t)block << 16) ^
        (uint64_t)(unsigned char)suffix[0];
    for (size_t i = 0; i < batch * k; i++) x[i] = hash_to_unit(seed, i);

    if (!h3_w4a8_convrot_act_f32(x, batch, k, convrot_gs, x_rot, error,
                                 sizeof(error))) {
        printf("  blocks.%u.%s: FAIL act ConvRot: %s\n", block,
               short_suffix(suffix), error);
        free(w_f32);
        free(x);
        free(x_rot);
        free(y_fp32);
        free(y_i8);
        free(y_qi);
        free(x_i8);
        free(x_hat);
        free(act_scales);
        free(w_i8);
        free(s_channel);
        tally->fail++;
        return 0;
    }

    h3_w4a8_gemm_f32(x_rot, w_f32, batch, n, k, y_fp32);

    if (!h3_w4a8_quantize_act_int8(x_rot, batch, k, act_mode, x_i8, act_scales,
                                   error, sizeof(error))) {
        printf("  blocks.%u.%s: FAIL act quant: %s\n", block,
               short_suffix(suffix), error);
        free(w_f32);
        free(x);
        free(x_rot);
        free(y_fp32);
        free(y_i8);
        free(y_qi);
        free(x_i8);
        free(x_hat);
        free(act_scales);
        free(w_i8);
        free(s_channel);
        tally->fail++;
        return 0;
    }

    const int per_tensor = (act_mode == H3_W4A8_ACT_QUANT_PER_TENSOR);
    h3_w4a8_gemm_i8_dequant_f32(x_i8, w_i8, act_scales, per_tensor, s_channel,
                                batch, n, k, y_i8);

    /* Reconstruct quantized activations for tight GEMM self-check. */
    for (size_t i = 0; i < batch; i++) {
        const float s = per_tensor ? act_scales[0] : act_scales[i];
        for (size_t t = 0; t < k; t++) {
            x_hat[i * k + t] = (float)x_i8[i * k + t] * s;
        }
    }
    h3_w4a8_gemm_f32(x_hat, w_f32, batch, n, k, y_qi);

    float ref_absmax = 0.0f;
    size_t nan_c = 0, inf_c = 0;
    stats_f32(y_fp32, batch * n, &ref_absmax, &nan_c, &inf_c);

    float qi_max = 0.0f, i8_max = 0.0f;
    double qi_rmse = 0.0, i8_rmse = 0.0;
    h3_w4a8_compare_f32(y_i8, y_qi, batch * n, &qi_max, &qi_rmse);
    h3_w4a8_compare_f32(y_i8, y_fp32, batch * n, &i8_max, &i8_rmse);

    size_t i8_nan = 0, i8_inf = 0;
    float i8_absmax = 0.0f;
    stats_f32(y_i8, batch * n, &i8_absmax, &i8_nan, &i8_inf);

    const int finite_ok = (nan_c == 0 && inf_c == 0 && i8_nan == 0 && i8_inf == 0);
    const int qi_ok = finite_ok && within_tol(qi_max, ref_absmax, atol, rtol);
    const int i8_ok =
        finite_ok && within_tol(i8_max, ref_absmax, int8_atol, int8_rtol);
    const int pass = qi_ok && i8_ok;

    clock_t t1 = clock();
    const double ms = 1000.0 * (double)(t1 - t0) / (double)CLOCKS_PER_SEC;

    printf("  blocks.%u.%s: N=%zu K=%zu batch=%zu absmax=%.4g "
           "ck_i8=%016llx ck_y=%016llx "
           "qi_max=%.3g qi_rmse=%.3g i8_max=%.3g i8_rmse=%.3g "
           "time_ms=%.1f %s\n",
           block, short_suffix(suffix), n, k, batch, ref_absmax,
           (unsigned long long)checksum_i8(w_i8, n * k),
           (unsigned long long)checksum_f32(y_fp32, batch * n), qi_max, qi_rmse,
           i8_max, i8_rmse, ms, pass ? "PASS" : "FAIL");
    if (!qi_ok)
        printf("    detail: int8_gemm_vs_qi FAIL (max_abs=%.6g tol≈%.6g)\n",
               qi_max, atol + rtol * (ref_absmax > 0 ? ref_absmax : 1.0f));
    if (!i8_ok)
        printf("    detail: int8_vs_fp32 FAIL (max_abs=%.6g tol≈%.6g)\n",
               i8_max,
               int8_atol + int8_rtol * (ref_absmax > 0 ? ref_absmax : 1.0f));

    free(w_f32);
    free(x);
    free(x_rot);
    free(y_fp32);
    free(y_i8);
    free(y_qi);
    free(x_i8);
    free(x_hat);
    free(act_scales);
    free(w_i8);
    free(s_channel);

    if (pass)
        tally->pass++;
    else
        tally->fail++;
    return pass;
}

int main(int argc, char **argv) {
    unsigned blocks[MAX_BLOCKS];
    int n_blocks = 0;
    /* default: representative layers */
    blocks[0] = 0;
    blocks[1] = 1;
    blocks[2] = 24;
    blocks[3] = 49;
    n_blocks = 4;

    size_t batch = 2;
    float atol = 1e-3f;
    float rtol = 1e-4f;
    /* INT8 act quant vs full FP32 act — looser (document in contract). */
    float int8_atol = 0.5f;
    float int8_rtol = 0.05f;
    h3_w4a8_act_quant_mode act_mode = H3_W4A8_ACT_QUANT_PER_ROW;
    const char *path = NULL;
    int only_out = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--blocks") && i + 1 < argc) {
            if (!parse_blocks(argv[++i], blocks, &n_blocks)) {
                fprintf(stderr, "bad --blocks spec (use 0,1,24,49 or 0-3)\n");
                return 2;
            }
        } else if (!strcmp(argv[i], "--batch") && i + 1 < argc) {
            batch = (size_t)strtoul(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--atol") && i + 1 < argc) {
            atol = strtof(argv[++i], NULL);
        } else if (!strcmp(argv[i], "--rtol") && i + 1 < argc) {
            rtol = strtof(argv[++i], NULL);
        } else if (!strcmp(argv[i], "--int8-atol") && i + 1 < argc) {
            int8_atol = strtof(argv[++i], NULL);
        } else if (!strcmp(argv[i], "--int8-rtol") && i + 1 < argc) {
            int8_rtol = strtof(argv[++i], NULL);
        } else if (!strcmp(argv[i], "--act-quant") && i + 1 < argc) {
            const char *m = argv[++i];
            if (!strcmp(m, "per-row") || !strcmp(m, "row"))
                act_mode = H3_W4A8_ACT_QUANT_PER_ROW;
            else if (!strcmp(m, "per-tensor") || !strcmp(m, "tensor"))
                act_mode = H3_W4A8_ACT_QUANT_PER_TENSOR;
            else {
                fprintf(stderr, "unknown --act-quant %s\n", m);
                return 2;
            }
        } else if (!strcmp(argv[i], "--only-out")) {
            only_out = 1;
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            fprintf(stderr,
                    "usage: %s [--blocks 0,1,24,49|0-3] [--batch M] "
                    "[--act-quant per-row|per-tensor] [--only-out] "
                    "[--atol A] [--rtol R] [--int8-atol A] [--int8-rtol R] "
                    "FILE.safetensors\n",
                    argv[0]);
            return 0;
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "unknown option: %s\n", argv[i]);
            return 2;
        } else {
            path = argv[i];
        }
    }
    if (!path || batch == 0 || n_blocks <= 0) {
        fprintf(stderr, "usage: %s [--blocks ...] FILE.safetensors\n",
                argv[0]);
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

    printf("path: %s\n", path);
    printf("blocks:");
    for (int i = 0; i < n_blocks; i++) printf(" %u", blocks[i]);
    printf("\n");
    printf("group_size: %d convrot_groupsize: %d batch: %zu\n", group_size,
           convrot_gs, batch);
    printf("act_quant: %s\n",
           act_mode == H3_W4A8_ACT_QUANT_PER_ROW ? "per-row" : "per-tensor");
    printf("tol_qi: atol=%.3g rtol=%.3g | tol_int8_vs_fp32: atol=%.3g "
           "rtol=%.3g\n",
           atol, rtol, int8_atol, int8_rtol);
    printf("note: uses W_rot (no weight Hadamard undo) + act ConvRot\n");

    probe_tally tally = {0, 0, 0};
    int all_ok = 1;

    for (int bi = 0; bi < n_blocks; bi++) {
        printf("block %u:\n", blocks[bi]);
        const int n_suf = only_out ? 1 : 4;
        for (int si = 0; si < n_suf; si++) {
            const char *suf = only_out ? H3_W4A8_SUFFIX_OUT : k_suffixes[si];
            if (!run_one_linear(store, blocks[bi], suf, group_size, convrot_gs,
                                batch, act_mode, atol, rtol, int8_atol,
                                int8_rtol, &tally)) {
                all_ok = 0;
            }
        }
    }

    printf("summary: pass=%d fail=%d skip=%d → %s\n", tally.pass, tally.fail,
           tally.skip, (all_ok && tally.fail == 0) ? "PASS" : "FAIL");

    h3_w4a8_store_free(store);
    return (all_ok && tally.fail == 0) ? 0 : 1;
}
