/* P4 step-5: device W4A8 unpack → resident INT8 + GEMM vs host load_linear_int8.
 * Build (Linux / CUDA):
 *   cmake --build build-cuda -j --target h3_w4a8_device_probe
 *   ./build-cuda/h3_w4a8_device_probe [--block N] [--suffixes out,fc2] \
 *       [--batch M] [--iters I] [--device I] FILE.safetensors
 *
 * Per linear:
 *   1) host load_linear_int8 (CPU unpack baseline)
 *   2) upload packed + device unpack → resident W_i8 + s_channel
 *   3) compare device INT8 vs host (full max_abs; expect 0)
 *   4) act ConvRot + INT8 quant; GEMM via resident weight
 *   5) timing: one unpack + N gemms vs host-dequant+H2D gemm baseline
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

static int compare_i8(const int8_t *a, const int8_t *b, size_t count,
                      int *max_abs_out, size_t *mismatch_out) {
    int max_abs = 0;
    size_t mismatch = 0;
    if (!a || !b || count == 0) return 0;
    for (size_t i = 0; i < count; i++) {
        int d = (int)a[i] - (int)b[i];
        if (d < 0) d = -d;
        if (d > max_abs) max_abs = d;
        if (d != 0) mismatch++;
    }
    if (max_abs_out) *max_abs_out = max_abs;
    if (mismatch_out) *mismatch_out = mismatch;
    return 1;
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
                   size_t batch, int iters, float atol, float rtol,
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

    /* --- host INT8 reference --- */
    clock_t th0 = clock();
    int8_t *w_host = NULL;
    float *s_channel = NULL;
    size_t n = 0, k = 0;
    if (!h3_w4a8_load_linear_int8(store, &quad, group_size, &w_host, &s_channel,
                                  &n, &k, error, sizeof(error))) {
        printf("  blocks.%u.%s: FAIL load_int8: %s\n", block,
               short_suffix(suffix), error);
        tally->fail++;
        return 0;
    }
    clock_t th1 = clock();
    const double host_dequant_ms =
        1000.0 * (double)(th1 - th0) / (double)CLOCKS_PER_SEC;

    /* --- packed tensors for device unpack --- */
    size_t nbytes = 0;
    int8_t *packed =
        (int8_t *)h3_w4a8_dup_tensor(store, quad.weight, &nbytes, error,
                                     sizeof(error));
    float *codebook =
        (float *)h3_w4a8_dup_tensor(store, quad.weight_codebook, &nbytes, error,
                                    sizeof(error));
    uint8_t *s_rel =
        (uint8_t *)h3_w4a8_dup_tensor(store, quad.weight_s_rel, &nbytes, error,
                                      sizeof(error));
    if (!packed || !codebook || !s_rel) {
        printf("  blocks.%u.%s: FAIL dup packed: %s\n", block,
               short_suffix(suffix), error);
        free(packed);
        free(codebook);
        free(s_rel);
        free(w_host);
        free(s_channel);
        tally->fail++;
        return 0;
    }
    const size_t k_half = k / 2;

    double unpack_ms = 0.0, h2d_ms = 0.0;
    h3_cuda_w4a8_weight *dw = NULL;
    /* Warmup create/free once so first timed create is stable. */
    {
        h3_cuda_w4a8_weight *tmp = NULL;
        if (!h3_cuda_w4a8_weight_create(cuda, packed, n, k_half, codebook, s_rel,
                                        s_channel, group_size, &tmp, NULL, NULL,
                                        error, sizeof(error))) {
            printf("  blocks.%u.%s: FAIL device unpack warmup: %s\n", block,
                   short_suffix(suffix), error);
            free(packed);
            free(codebook);
            free(s_rel);
            free(w_host);
            free(s_channel);
            tally->fail++;
            return 0;
        }
        h3_cuda_w4a8_weight_free(tmp);
    }
    if (!h3_cuda_w4a8_weight_create(cuda, packed, n, k_half, codebook, s_rel,
                                    s_channel, group_size, &dw, &unpack_ms,
                                    &h2d_ms, error, sizeof(error))) {
        printf("  blocks.%u.%s: FAIL device unpack: %s\n", block,
               short_suffix(suffix), error);
        free(packed);
        free(codebook);
        free(s_rel);
        free(w_host);
        free(s_channel);
        tally->fail++;
        return 0;
    }
    free(packed);
    free(codebook);
    free(s_rel);

    int8_t *w_dev = (int8_t *)malloc(n * k);
    if (!w_dev ||
        !h3_cuda_w4a8_weight_download_int8(dw, w_dev, error, sizeof(error))) {
        printf("  blocks.%u.%s: FAIL download INT8: %s\n", block,
               short_suffix(suffix), error);
        free(w_dev);
        h3_cuda_w4a8_weight_free(dw);
        free(w_host);
        free(s_channel);
        tally->fail++;
        return 0;
    }

    int w_max_abs = 0;
    size_t w_mismatch = 0;
    compare_i8(w_host, w_dev, n * k, &w_max_abs, &w_mismatch);
    free(w_dev);

    /* activations */
    float *x = (float *)malloc(batch * k * sizeof(float));
    float *x_rot = (float *)malloc(batch * k * sizeof(float));
    int8_t *x_i8 = (int8_t *)malloc(batch * k);
    float *act_scales = (float *)malloc(batch * sizeof(float));
    float *y_cpu = (float *)malloc(batch * n * sizeof(float));
    float *y_res = (float *)malloc(batch * n * sizeof(float));
    float *y_base = (float *)malloc(batch * n * sizeof(float));
    if (!x || !x_rot || !x_i8 || !act_scales || !y_cpu || !y_res || !y_base) {
        printf("  blocks.%u.%s: FAIL OOM\n", block, short_suffix(suffix));
        free(x);
        free(x_rot);
        free(x_i8);
        free(act_scales);
        free(y_cpu);
        free(y_res);
        free(y_base);
        h3_cuda_w4a8_weight_free(dw);
        free(w_host);
        free(s_channel);
        tally->fail++;
        return 0;
    }

    const uint64_t seed = 0x4833444556494345ULL; /* 'H3DEVICE' */
    for (size_t i = 0; i < batch * k; i++) x[i] = hash_to_unit(seed, i);

    if (!h3_w4a8_convrot_act_f32(x, batch, k, convrot_gs, x_rot, error,
                                 sizeof(error)) ||
        !h3_w4a8_quantize_act_int8(x_rot, batch, k, H3_W4A8_ACT_QUANT_PER_ROW,
                                   x_i8, act_scales, error, sizeof(error))) {
        printf("  blocks.%u.%s: FAIL act path: %s\n", block,
               short_suffix(suffix), error);
        free(x);
        free(x_rot);
        free(x_i8);
        free(act_scales);
        free(y_cpu);
        free(y_res);
        free(y_base);
        h3_cuda_w4a8_weight_free(dw);
        free(w_host);
        free(s_channel);
        tally->fail++;
        return 0;
    }

    h3_w4a8_gemm_i8_dequant_f32(x_i8, w_host, act_scales, 0, s_channel, batch,
                                n, k, y_cpu);

    char backend[32] = {0};
    double gemm_res_ms = 0.0;
    /* Warmup + timed resident GEMMs */
    if (!h3_cuda_gemm_i8_dequant_f32_w(cuda, dw, x_i8, act_scales, 0, batch,
                                       y_res, NULL, backend, sizeof(backend),
                                       error, sizeof(error))) {
        printf("  blocks.%u.%s: FAIL resident gemm warmup: %s\n", block,
               short_suffix(suffix), error);
        free(x);
        free(x_rot);
        free(x_i8);
        free(act_scales);
        free(y_cpu);
        free(y_res);
        free(y_base);
        h3_cuda_w4a8_weight_free(dw);
        free(w_host);
        free(s_channel);
        tally->fail++;
        return 0;
    }
    double gemm_sum = 0.0;
    for (int i = 0; i < iters; i++) {
        double ms = 0.0;
        if (!h3_cuda_gemm_i8_dequant_f32_w(cuda, dw, x_i8, act_scales, 0, batch,
                                           y_res, &ms, backend, sizeof(backend),
                                           error, sizeof(error))) {
            printf("  blocks.%u.%s: FAIL resident gemm: %s\n", block,
                   short_suffix(suffix), error);
            free(x);
            free(x_rot);
            free(x_i8);
            free(act_scales);
            free(y_cpu);
            free(y_res);
            free(y_base);
            h3_cuda_w4a8_weight_free(dw);
            free(w_host);
            free(s_channel);
            tally->fail++;
            return 0;
        }
        gemm_sum += ms;
    }
    gemm_res_ms = gemm_sum / (double)iters;

    /* Baseline: host INT8 already loaded; each call H2D's W (old path) */
    double base_sum = 0.0;
    if (!h3_cuda_gemm_i8_dequant_f32(cuda, x_i8, w_host, act_scales, 0,
                                     s_channel, batch, n, k, y_base, NULL, NULL,
                                     0, error, sizeof(error))) {
        printf("  blocks.%u.%s: FAIL baseline gemm warmup: %s\n", block,
               short_suffix(suffix), error);
        free(x);
        free(x_rot);
        free(x_i8);
        free(act_scales);
        free(y_cpu);
        free(y_res);
        free(y_base);
        h3_cuda_w4a8_weight_free(dw);
        free(w_host);
        free(s_channel);
        tally->fail++;
        return 0;
    }
    /* Wall time including weight H2D for fair "reload every time" baseline. */
    clock_t tb0 = clock();
    for (int i = 0; i < iters; i++) {
        double ms = 0.0;
        if (!h3_cuda_gemm_i8_dequant_f32(cuda, x_i8, w_host, act_scales, 0,
                                         s_channel, batch, n, k, y_base, &ms,
                                         NULL, 0, error, sizeof(error))) {
            printf("  blocks.%u.%s: FAIL baseline gemm: %s\n", block,
                   short_suffix(suffix), error);
            free(x);
            free(x_rot);
            free(x_i8);
            free(act_scales);
            free(y_cpu);
            free(y_res);
            free(y_base);
            h3_cuda_w4a8_weight_free(dw);
            free(w_host);
            free(s_channel);
            tally->fail++;
            return 0;
        }
        base_sum += ms;
    }
    clock_t tb1 = clock();
    const double base_gemm_ms = base_sum / (double)iters;
    const double base_wall_ms =
        1000.0 * (double)(tb1 - tb0) / (double)CLOCKS_PER_SEC / (double)iters;

    float y_max_abs = 0.0f;
    double y_rmse = 0.0;
    h3_w4a8_compare_f32(y_cpu, y_res, batch * n, &y_max_abs, &y_rmse);
    float ref_absmax = 0.0f;
    for (size_t i = 0; i < batch * n; i++) {
        float a = fabsf(y_cpu[i]);
        if (a > ref_absmax) ref_absmax = a;
    }
    const float thresh =
        atol + rtol * (ref_absmax > 0.0f ? ref_absmax : 1.0f);
    const int pass_w = (w_max_abs == 0);
    const int pass_y = (y_max_abs <= thresh);
    const int pass = pass_w && pass_y;

    const double resident_total_once =
        h2d_ms + unpack_ms + gemm_res_ms * (double)iters;
    /* Baseline reload: host dequant once + (H2D W + gemm) * iters
     * (host dequant already done; wall includes H2D of W each iter). */
    const double baseline_total =
        host_dequant_ms + base_wall_ms * (double)iters;

    printf("  blocks.%u.%s: N=%zu K=%zu batch=%zu iters=%d backend=%s\n", block,
           short_suffix(suffix), n, k, batch, iters, backend);
    printf("    host_dequant_ms=%.2f  device_h2d_ms=%.3f  device_unpack_ms=%.3f\n",
           host_dequant_ms, h2d_ms, unpack_ms);
    printf("    weight_vs_host: max_abs=%d mismatch=%zu %s\n", w_max_abs,
           w_mismatch, pass_w ? "PASS" : "FAIL");
    printf("    gemm_resident_ms=%.3f  gemm_baseline_dev_ms=%.3f  "
           "baseline_wall_ms=%.3f\n",
           gemm_res_ms, base_gemm_ms, base_wall_ms);
    printf("    total_%d_gemms: resident(h2d+unpack+gemms)=%.2f  "
           "baseline(host_dq+wall)=%.2f\n",
           iters, resident_total_once, baseline_total);
    printf("    y_vs_cpu: max_abs=%.6g rmse=%.6g ref_absmax=%.6g "
           "checksum=%016llx %s\n",
           y_max_abs, y_rmse, ref_absmax,
           (unsigned long long)checksum_f32(y_res, batch * n),
           pass_y ? "PASS" : "FAIL");

    free(x);
    free(x_rot);
    free(x_i8);
    free(act_scales);
    free(y_cpu);
    free(y_res);
    free(y_base);
    h3_cuda_w4a8_weight_free(dw);
    free(w_host);
    free(s_channel);

    if (pass) tally->pass++;
    else tally->fail++;
    return pass;
}

int main(int argc, char **argv) {
    unsigned block = 0;
    const char *suffix_spec = "out,fc2";
    size_t batch = 4;
    int iters = 8;
    float atol = 1e-3f;
    float rtol = 1e-4f;
    int device = 0;
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
        } else if (!strcmp(argv[i], "--iters") && i + 1 < argc) {
            iters = (int)strtol(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--atol") && i + 1 < argc) {
            atol = strtof(argv[++i], NULL);
        } else if (!strcmp(argv[i], "--rtol") && i + 1 < argc) {
            rtol = strtof(argv[++i], NULL);
        } else if (!strcmp(argv[i], "--device") && i + 1 < argc) {
            device = (int)strtol(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            fprintf(stderr,
                    "usage: %s [--block N] [--suffixes out,fc2] [--batch M] "
                    "[--iters I] [--atol A] [--rtol R] [--device I] "
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
    if (!path || batch == 0 || iters <= 0) {
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
    printf("block: %u suffixes: %s batch: %zu iters: %d\n", block, suffix_spec,
           batch, iters);
    printf("group_size: %d convrot_groupsize: %d\n", group_size, convrot_gs);
    printf("tolerance (y cuda vs cpu): atol=%.3g rtol=%.3g\n", atol, rtol);
    printf("note: device unpack should match host INT8 bit-identical "
           "(max_abs=0)\n");

    probe_tally tally = {0, 0};
    int all_ok = 1;
    for (int i = 0; i < n_suffixes; i++) {
        if (!run_one(cuda, store, block, suffixes[i], group_size, convrot_gs,
                     batch, iters, atol, rtol, &tally)) {
            all_ok = 0;
        }
    }

    printf("summary: pass=%d fail=%d → %s\n", tally.pass, tally.fail,
           all_ok && tally.fail == 0 ? "PASS" : "FAIL");

    h3_cuda_free(cuda);
    h3_w4a8_store_free(store);
    return all_ok && tally.fail == 0 ? 0 : 1;
}
