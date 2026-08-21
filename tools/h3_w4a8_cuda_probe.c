/* P4 step-2: W4A8 act ConvRot equivalence + cuBLAS FP32 GEMM probe.
 * Build (Linux / CUDA):
 *   cmake -S . -B build-cuda -DH3_GPU_BACKEND=cuda && cmake --build build-cuda -j
 *   ./build-cuda/h3_w4a8_cuda_probe [--block N] [--suffix out|qkv|fc1|fc2] \
 *       [--batch M] [--atol A] [--rtol R] FILE.safetensors
 *
 * Checks:
 *   1) Y_ref  = X @ W_undo.T                         (CPU)
 *   2) Y_act  = (X @ H) @ W_rot.T                    (CPU; must ≈ Y_ref)
 *   3) Y_cuda = cuBLAS(X, W_undo)                    (must ≈ Y_ref)
 */
#include "h3_gpu_cuda.h"
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

int main(int argc, char **argv) {
    unsigned block = 0;
    const char *suffix_arg = "out";
    size_t batch = 4;
    float atol = 1e-3f;
    float rtol = 1e-4f;
    int device = 0;
    const char *path = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--block") && i + 1 < argc) {
            block = (unsigned)strtoul(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--suffix") && i + 1 < argc) {
            suffix_arg = argv[++i];
        } else if (!strcmp(argv[i], "--batch") && i + 1 < argc) {
            batch = (size_t)strtoul(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--atol") && i + 1 < argc) {
            atol = strtof(argv[++i], NULL);
        } else if (!strcmp(argv[i], "--rtol") && i + 1 < argc) {
            rtol = strtof(argv[++i], NULL);
        } else if (!strcmp(argv[i], "--device") && i + 1 < argc) {
            device = (int)strtol(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            fprintf(stderr,
                    "usage: %s [--block N] [--suffix out|qkv|fc1|fc2] "
                    "[--batch M] [--atol A] [--rtol R] [--device I] "
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
                "usage: %s [--block N] [--suffix out|qkv|fc1|fc2] "
                "[--batch M] FILE.safetensors\n",
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
    printf("group_size: %d convrot_groupsize: %d\n", group_size, convrot_gs);
    printf("packed_shape: [%llu,%llu] (logical K=%llu)\n",
           (unsigned long long)quad.weight->shape[0],
           (unsigned long long)quad.weight->shape[1],
           (unsigned long long)quad.weight->shape[1] * 2ULL);
    printf("tolerance: atol=%.3g rtol=%.3g\n", atol, rtol);

    clock_t t0 = clock();
    float *w_undo = NULL;
    size_t n = 0, k = 0;
    if (!h3_w4a8_load_linear_f32(store, &quad, group_size, convrot_gs, 1,
                                 &w_undo, &n, &k, error, sizeof(error))) {
        fprintf(stderr, "dequant (undo) failed: %s\n", error);
        h3_w4a8_store_free(store);
        return 1;
    }
    clock_t t1 = clock();
    const double dequant_undo_ms =
        1000.0 * (double)(t1 - t0) / (double)CLOCKS_PER_SEC;

    t0 = clock();
    float *w_rot = NULL;
    size_t n_rot = 0, k_rot = 0;
    if (!h3_w4a8_load_linear_f32(store, &quad, group_size, convrot_gs, 0,
                                 &w_rot, &n_rot, &k_rot, error,
                                 sizeof(error))) {
        fprintf(stderr, "dequant (rot) failed: %s\n", error);
        free(w_undo);
        h3_w4a8_store_free(store);
        return 1;
    }
    t1 = clock();
    const double dequant_rot_ms =
        1000.0 * (double)(t1 - t0) / (double)CLOCKS_PER_SEC;

    if (n_rot != n || k_rot != k) {
        fprintf(stderr, "shape mismatch between undo/rot dequant\n");
        free(w_undo);
        free(w_rot);
        h3_w4a8_store_free(store);
        return 1;
    }

    printf("dequant_undo: N=%zu K=%zu time_ms=%.2f\n", n, k, dequant_undo_ms);
    printf("dequant_rot:  N=%zu K=%zu time_ms=%.2f\n", n, k, dequant_rot_ms);

    float *x = (float *)malloc(batch * k * sizeof(float));
    float *x_rot = (float *)malloc(batch * k * sizeof(float));
    float *y_ref = (float *)malloc(batch * n * sizeof(float));
    float *y_act = (float *)malloc(batch * n * sizeof(float));
    float *y_cuda = (float *)malloc(batch * n * sizeof(float));
    if (!x || !x_rot || !y_ref || !y_act || !y_cuda) {
        fprintf(stderr, "out of memory for GEMM buffers\n");
        free(x);
        free(x_rot);
        free(y_ref);
        free(y_act);
        free(y_cuda);
        free(w_undo);
        free(w_rot);
        h3_w4a8_store_free(store);
        return 1;
    }

    const uint64_t seed = 0x48324d41544d4f4cULL; /* 'H3MATMUL' */
    for (size_t i = 0; i < batch * k; i++) x[i] = hash_to_unit(seed, i);

    if (!h3_w4a8_convrot_act_f32(x, batch, k, convrot_gs, x_rot, error,
                                 sizeof(error))) {
        fprintf(stderr, "act ConvRot failed: %s\n", error);
        free(x);
        free(x_rot);
        free(y_ref);
        free(y_act);
        free(y_cuda);
        free(w_undo);
        free(w_rot);
        h3_w4a8_store_free(store);
        return 1;
    }

    t0 = clock();
    h3_w4a8_gemm_f32(x, w_undo, batch, n, k, y_ref);
    t1 = clock();
    const double cpu_ref_ms =
        1000.0 * (double)(t1 - t0) / (double)CLOCKS_PER_SEC;

    t0 = clock();
    h3_w4a8_gemm_f32(x_rot, w_rot, batch, n, k, y_act);
    t1 = clock();
    const double cpu_act_ms =
        1000.0 * (double)(t1 - t0) / (double)CLOCKS_PER_SEC;

    float ref_absmax = 0.0f;
    size_t nan_c = 0, inf_c = 0;
    stats_f32(y_ref, batch * n, &ref_absmax, &nan_c, &inf_c);
    printf("cpu_ref_gemm: batch=%zu out=%zu time_ms=%.2f absmax=%.6g "
           "checksum=%016llx nan=%zu inf=%zu\n",
           batch, n, cpu_ref_ms, ref_absmax,
           (unsigned long long)checksum_f32(y_ref, batch * n), nan_c, inf_c);

    float act_max_abs = 0.0f;
    double act_rmse = 0.0;
    h3_w4a8_compare_f32(y_ref, y_act, batch * n, &act_max_abs, &act_rmse);
    const int act_pass =
        (nan_c == 0 && inf_c == 0 &&
         within_tol(act_max_abs, ref_absmax, atol, rtol));
    printf("act_convrot_vs_weight_undo: max_abs=%.6g rmse=%.6g time_ms=%.2f "
           "%s\n",
           act_max_abs, act_rmse, cpu_act_ms, act_pass ? "PASS" : "FAIL");

    h3_cuda *cuda = h3_cuda_create(device, error, sizeof(error));
    if (!cuda) {
        fprintf(stderr, "cuda create failed: %s\n", error);
        free(x);
        free(x_rot);
        free(y_ref);
        free(y_act);
        free(y_cuda);
        free(w_undo);
        free(w_rot);
        h3_w4a8_store_free(store);
        return 1;
    }
    h3_cuda_device_info info;
    h3_cuda_get_info(cuda, &info);
    printf("cuda_device: [%d] %s sm_%d%d\n", info.device_index, info.name,
           info.compute_major, info.compute_minor);

    double cuda_ms = 0.0;
    /* Warmup + timed run */
    if (!h3_cuda_gemm_f32(cuda, x, w_undo, batch, n, k, y_cuda, NULL, error,
                          sizeof(error))) {
        fprintf(stderr, "cublas warmup failed: %s\n", error);
        h3_cuda_free(cuda);
        free(x);
        free(x_rot);
        free(y_ref);
        free(y_act);
        free(y_cuda);
        free(w_undo);
        free(w_rot);
        h3_w4a8_store_free(store);
        return 1;
    }
    if (!h3_cuda_gemm_f32(cuda, x, w_undo, batch, n, k, y_cuda, &cuda_ms,
                          error, sizeof(error))) {
        fprintf(stderr, "cublas gemm failed: %s\n", error);
        h3_cuda_free(cuda);
        free(x);
        free(x_rot);
        free(y_ref);
        free(y_act);
        free(y_cuda);
        free(w_undo);
        free(w_rot);
        h3_w4a8_store_free(store);
        return 1;
    }

    float cuda_max_abs = 0.0f;
    double cuda_rmse = 0.0;
    h3_w4a8_compare_f32(y_ref, y_cuda, batch * n, &cuda_max_abs, &cuda_rmse);
    size_t cuda_nan = 0, cuda_inf = 0;
    float cuda_absmax = 0.0f;
    stats_f32(y_cuda, batch * n, &cuda_absmax, &cuda_nan, &cuda_inf);
    const int cuda_pass =
        (cuda_nan == 0 && cuda_inf == 0 &&
         within_tol(cuda_max_abs, ref_absmax, atol, rtol));
    printf("cublas_vs_cpu_ref: max_abs=%.6g rmse=%.6g gemm_ms=%.3f "
           "checksum=%016llx %s\n",
           cuda_max_abs, cuda_rmse, cuda_ms,
           (unsigned long long)checksum_f32(y_cuda, batch * n),
           cuda_pass ? "PASS" : "FAIL");

    const int pass = act_pass && cuda_pass;
    printf("summary: %s\n", pass ? "PASS" : "FAIL");

    h3_cuda_free(cuda);
    free(x);
    free(x_rot);
    free(y_ref);
    free(y_act);
    free(y_cuda);
    free(w_undo);
    free(w_rot);
    h3_w4a8_store_free(store);
    return pass ? 0 : 1;
}
