/* P4 step-4: CUDA INT8×INT8→INT32 GEMM (+ scale dequant) vs CPU INT8 ref.
 * Build (Linux / CUDA):
 *   cmake --build build-cuda -j --target h3_w4a8_int8_cuda_probe
 *   ./build-cuda/h3_w4a8_int8_cuda_probe [--block N] [--suffixes out,fc2] \
 *       [--batch M] [--atol A] [--rtol R] [--device I] FILE.safetensors
 *
 * Per linear:
 *   1) load W_i8 + s_channel (rotated basis)
 *   2) X → ConvRot → dynamic INT8 act (per-row)
 *   3) Y_cpu  = h3_w4a8_gemm_i8_dequant_f32(...)
 *   4) Y_cuda = h3_cuda_gemm_i8_dequant_f32(...)  (cuBLASLt / GemmEx / kernel)
 *   5) optional FP32 cuBLAS timing on same shapes (dequant weight to F32)
 */
#include "h3_gpu_cuda.h"
#include "h3_w4a8_dequant.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

static const char *resolve_suffix(const char *name) {
    if (!name || !strcmp(name, "out") || !strcmp(name, "out_proj"))
        return H3_W4A8_SUFFIX_OUT;
    if (!strcmp(name, "qkv") || !strcmp(name, "qkv_proj"))
        return H3_W4A8_SUFFIX_QKV;
    if (!strcmp(name, "fc1")) return H3_W4A8_SUFFIX_FC1;
    if (!strcmp(name, "fc2")) return H3_W4A8_SUFFIX_FC2;
    return NULL;
}

static const char *short_suffix(const char *s) {
    if (!strcmp(s, H3_W4A8_SUFFIX_QKV)) return "qkv";
    if (!strcmp(s, H3_W4A8_SUFFIX_OUT)) return "out";
    if (!strcmp(s, H3_W4A8_SUFFIX_FC1)) return "fc1";
    if (!strcmp(s, H3_W4A8_SUFFIX_FC2)) return "fc2";
    return s;
}

static int parse_suffixes(const char *spec, const char **out, int *out_count) {
    *out_count = 0;
    if (!spec || !spec[0]) return 0;
    const char *p = spec;
    while (*p) {
        if (*out_count >= MAX_SUFFIXES) return 0;
        char tok[32];
        size_t len = 0;
        while (*p && *p != ',' && len + 1 < sizeof(tok)) tok[len++] = *p++;
        tok[len] = '\0';
        if (*p == ',') p++;
        if (len == 0) continue;
        const char *s = resolve_suffix(tok);
        if (!s) return 0;
        out[(*out_count)++] = s;
    }
    return *out_count > 0;
}

typedef struct {
    int pass;
    int fail;
} probe_tally;

static int run_one(h3_cuda *cuda, const h3_w4a8_store *store, unsigned block,
                   const char *suffix, int group_size, int convrot_gs,
                   size_t batch, float atol, float rtol, int do_fp32_bench,
                   probe_tally *tally) {
    char error[512];
    h3_w4a8_quad quad;
    if (!h3_w4a8_find_block_linear(store, block, suffix, &quad, error,
                                   sizeof(error))) {
        printf("  blocks.%u.%s: FAIL find: %s\n", block, short_suffix(suffix),
               error);
        tally->fail++;
        return 0;
    }

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

    float *x = (float *)malloc(batch * k * sizeof(float));
    float *x_rot = (float *)malloc(batch * k * sizeof(float));
    int8_t *x_i8 = (int8_t *)malloc(batch * k);
    float *act_scales = (float *)malloc(batch * sizeof(float));
    float *y_cpu = (float *)malloc(batch * n * sizeof(float));
    float *y_cuda = (float *)malloc(batch * n * sizeof(float));
    if (!x || !x_rot || !x_i8 || !act_scales || !y_cpu || !y_cuda) {
        printf("  blocks.%u.%s: FAIL OOM\n", block, short_suffix(suffix));
        free(x);
        free(x_rot);
        free(x_i8);
        free(act_scales);
        free(y_cpu);
        free(y_cuda);
        free(w_i8);
        free(s_channel);
        tally->fail++;
        return 0;
    }

    const uint64_t seed = 0x4833494e54384355ULL; /* 'H3INT8CU' */
    for (size_t i = 0; i < batch * k; i++) x[i] = hash_to_unit(seed, i);

    if (!h3_w4a8_convrot_act_f32(x, batch, k, convrot_gs, x_rot, error,
                                 sizeof(error))) {
        printf("  blocks.%u.%s: FAIL act ConvRot: %s\n", block,
               short_suffix(suffix), error);
        free(x);
        free(x_rot);
        free(x_i8);
        free(act_scales);
        free(y_cpu);
        free(y_cuda);
        free(w_i8);
        free(s_channel);
        tally->fail++;
        return 0;
    }

    if (!h3_w4a8_quantize_act_int8(x_rot, batch, k, H3_W4A8_ACT_QUANT_PER_ROW,
                                   x_i8, act_scales, error, sizeof(error))) {
        printf("  blocks.%u.%s: FAIL act quant: %s\n", block,
               short_suffix(suffix), error);
        free(x);
        free(x_rot);
        free(x_i8);
        free(act_scales);
        free(y_cpu);
        free(y_cuda);
        free(w_i8);
        free(s_channel);
        tally->fail++;
        return 0;
    }

    clock_t t0 = clock();
    h3_w4a8_gemm_i8_dequant_f32(x_i8, w_i8, act_scales, 0, s_channel, batch, n,
                                k, y_cpu);
    clock_t t1 = clock();
    const double cpu_ms =
        1000.0 * (double)(t1 - t0) / (double)CLOCKS_PER_SEC;

    float ref_absmax = 0.0f;
    size_t nan_c = 0, inf_c = 0;
    stats_f32(y_cpu, batch * n, &ref_absmax, &nan_c, &inf_c);

    char backend[32] = {0};
    double cuda_ms = 0.0;
    /* Warmup */
    if (!h3_cuda_gemm_i8_dequant_f32(cuda, x_i8, w_i8, act_scales, 0, s_channel,
                                     batch, n, k, y_cuda, NULL, backend,
                                     sizeof(backend), error, sizeof(error))) {
        printf("  blocks.%u.%s: FAIL cuda INT8 warmup: %s\n", block,
               short_suffix(suffix), error);
        free(x);
        free(x_rot);
        free(x_i8);
        free(act_scales);
        free(y_cpu);
        free(y_cuda);
        free(w_i8);
        free(s_channel);
        tally->fail++;
        return 0;
    }
    if (!h3_cuda_gemm_i8_dequant_f32(cuda, x_i8, w_i8, act_scales, 0, s_channel,
                                     batch, n, k, y_cuda, &cuda_ms, backend,
                                     sizeof(backend), error, sizeof(error))) {
        printf("  blocks.%u.%s: FAIL cuda INT8 gemm: %s\n", block,
               short_suffix(suffix), error);
        free(x);
        free(x_rot);
        free(x_i8);
        free(act_scales);
        free(y_cpu);
        free(y_cuda);
        free(w_i8);
        free(s_channel);
        tally->fail++;
        return 0;
    }

    float max_abs = 0.0f;
    double rmse = 0.0;
    h3_w4a8_compare_f32(y_cpu, y_cuda, batch * n, &max_abs, &rmse);
    size_t cuda_nan = 0, cuda_inf = 0;
    float cuda_absmax = 0.0f;
    stats_f32(y_cuda, batch * n, &cuda_absmax, &cuda_nan, &cuda_inf);
    const int pass =
        (nan_c == 0 && inf_c == 0 && cuda_nan == 0 && cuda_inf == 0 &&
         within_tol(max_abs, ref_absmax, atol, rtol));

    double fp32_ms = -1.0;
    if (do_fp32_bench) {
        float *w_f32 = (float *)malloc(n * k * sizeof(float));
        float *y_fp = (float *)malloc(batch * n * sizeof(float));
        if (w_f32 && y_fp &&
            h3_w4a8_int8_scale_to_f32(w_i8, n, k, s_channel, w_f32, error,
                                      sizeof(error))) {
            /* Reconstruct quantized activations for fair FP32 GEMM timing. */
            float *x_hat = (float *)malloc(batch * k * sizeof(float));
            if (x_hat) {
                for (size_t r = 0; r < batch; r++) {
                    const float s = act_scales[r];
                    for (size_t c = 0; c < k; c++) {
                        x_hat[r * k + c] = (float)x_i8[r * k + c] * s;
                    }
                }
                (void)h3_cuda_gemm_f32(cuda, x_hat, w_f32, batch, n, k, y_fp,
                                       NULL, error, sizeof(error));
                (void)h3_cuda_gemm_f32(cuda, x_hat, w_f32, batch, n, k, y_fp,
                                       &fp32_ms, error, sizeof(error));
                free(x_hat);
            }
        }
        free(w_f32);
        free(y_fp);
    }

    printf("  blocks.%u.%s: N=%zu K=%zu batch=%zu backend=%s\n", block,
           short_suffix(suffix), n, k, batch, backend);
    printf("    cpu_i8_ms=%.2f cuda_i8_ms=%.3f", cpu_ms, cuda_ms);
    if (fp32_ms >= 0.0) printf(" cublas_f32_ms=%.3f", fp32_ms);
    printf("\n");
    printf("    max_abs=%.6g rmse=%.6g ref_absmax=%.6g checksum=%016llx %s\n",
           max_abs, rmse, ref_absmax,
           (unsigned long long)checksum_f32(y_cuda, batch * n),
           pass ? "PASS" : "FAIL");

    free(x);
    free(x_rot);
    free(x_i8);
    free(act_scales);
    free(y_cpu);
    free(y_cuda);
    free(w_i8);
    free(s_channel);

    if (pass) tally->pass++;
    else tally->fail++;
    return pass;
}

int main(int argc, char **argv) {
    unsigned block = 0;
    const char *suffix_spec = "out,fc2";
    size_t batch = 4;
    float atol = 1e-3f;
    float rtol = 1e-4f;
    int device = 0;
    int do_fp32_bench = 1;
    const char *path = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--block") && i + 1 < argc) {
            block = (unsigned)strtoul(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--suffixes") && i + 1 < argc) {
            suffix_spec = argv[++i];
        } else if (!strcmp(argv[i], "--suffix") && i + 1 < argc) {
            suffix_spec = argv[++i];
        } else if (!strcmp(argv[i], "--batch") && i + 1 < argc) {
            batch = (size_t)strtoul(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--atol") && i + 1 < argc) {
            atol = strtof(argv[++i], NULL);
        } else if (!strcmp(argv[i], "--rtol") && i + 1 < argc) {
            rtol = strtof(argv[++i], NULL);
        } else if (!strcmp(argv[i], "--device") && i + 1 < argc) {
            device = (int)strtol(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--no-fp32-bench")) {
            do_fp32_bench = 0;
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            fprintf(stderr,
                    "usage: %s [--block N] [--suffixes out,fc2] [--batch M] "
                    "[--atol A] [--rtol R] [--device I] [--no-fp32-bench] "
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
    if (!path || batch == 0) {
        fprintf(stderr,
                "usage: %s [--block N] [--suffixes out,fc2] FILE.safetensors\n",
                argv[0]);
        return 2;
    }

    const char *suffixes[MAX_SUFFIXES];
    int n_suffixes = 0;
    if (!parse_suffixes(suffix_spec, suffixes, &n_suffixes)) {
        fprintf(stderr, "bad --suffixes: %s\n", suffix_spec);
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

    h3_cuda *cuda = h3_cuda_create(device, error, sizeof(error));
    if (!cuda) {
        fprintf(stderr, "cuda create failed: %s\n", error);
        h3_w4a8_store_free(store);
        return 1;
    }
    h3_cuda_device_info info;
    h3_cuda_get_info(cuda, &info);

    printf("path: %s\n", path);
    printf("cuda_device: [%d] %s sm_%d%d\n", info.device_index, info.name,
           info.compute_major, info.compute_minor);
    printf("block: %u suffixes: %s batch: %zu\n", block, suffix_spec, batch);
    printf("group_size: %d convrot_groupsize: %d act_quant: per-row\n",
           group_size, convrot_gs);
    printf("tolerance (cuda_i8 vs cpu_i8): atol=%.3g rtol=%.3g\n", atol, rtol);
    printf("note: INT8 accumulate is exact; dequant FP32 should match tightly\n");

    probe_tally tally = {0, 0};
    int all_ok = 1;
    for (int i = 0; i < n_suffixes; i++) {
        if (!run_one(cuda, store, block, suffixes[i], group_size, convrot_gs,
                     batch, atol, rtol, do_fp32_bench, &tally)) {
            all_ok = 0;
        }
    }

    printf("summary: pass=%d fail=%d → %s\n", tally.pass, tally.fail,
           all_ok && tally.fail == 0 ? "PASS" : "FAIL");

    h3_cuda_free(cuda);
    h3_w4a8_store_free(store);
    return all_ok && tally.fail == 0 ? 0 : 1;
}
