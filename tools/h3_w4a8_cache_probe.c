/* P4 step-6: 50-block W4A8 INT8 residency / swap probe.
 * Build (Linux / CUDA):
 *   cmake --build build-cuda -j --target h3_w4a8_cache_probe
 *   ./build-cuda/h3_w4a8_cache_probe [--policy resident_all|window|lru] \
 *       [--window N] [--blocks 50] [--sample 0,24,49] [--batch M] \
 *       [--device I] FILE.safetensors
 *
 * Reports: how many blocks fit on device, VRAM, warmup time, GEMM smoke.
 */
#include "h3_gpu_cuda.h"
#include "h3_w4a8_cache.h"
#include "h3_w4a8_dequant.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_SAMPLE 16

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

static const char *format_bytes(size_t bytes, char *buf, size_t buf_size) {
    const double gb = (double)bytes / (1024.0 * 1024.0 * 1024.0);
    const double mb = (double)bytes / (1024.0 * 1024.0);
    if (gb >= 1.0)
        snprintf(buf, buf_size, "%.3f GiB (%zu B)", gb, bytes);
    else
        snprintf(buf, buf_size, "%.1f MiB (%zu B)", mb, bytes);
    return buf;
}

static int parse_blocks_list(const char *spec, unsigned *out, int *out_count,
                             int max_count) {
    *out_count = 0;
    if (!spec || !spec[0]) return 0;
    const char *p = spec;
    while (*p) {
        if (*out_count >= max_count) return 0;
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

static h3_w4a8_cache_policy parse_policy(const char *s) {
    if (!s || !strcmp(s, "resident_all") || !strcmp(s, "all"))
        return H3_W4A8_CACHE_RESIDENT_ALL;
    if (!strcmp(s, "window")) return H3_W4A8_CACHE_WINDOW;
    if (!strcmp(s, "lru")) return H3_W4A8_CACHE_LRU;
    return (h3_w4a8_cache_policy)-1;
}

static int gemm_smoke(h3_cuda *cuda, h3_w4a8_cache *cache,
                      const h3_w4a8_store *store, unsigned block,
                      h3_w4a8_linear_kind kind, int group_size, int convrot_gs,
                      size_t batch, float atol, float rtol) {
    char error[512];
    const h3_cuda_w4a8_weight *dw =
        h3_w4a8_cache_get_linear(cache, block, kind, error, sizeof(error));
    if (!dw) {
        printf("  blocks.%u.%s: FAIL get: %s\n", block,
               h3_w4a8_linear_kind_name(kind), error);
        return 0;
    }
    size_t n = h3_cuda_w4a8_weight_n(dw);
    size_t k = h3_cuda_w4a8_weight_k(dw);

    /* Host INT8 reference for y compare */
    h3_w4a8_quad quad;
    if (!h3_w4a8_find_block_linear(store, block,
                                   h3_w4a8_linear_kind_suffix(kind), &quad,
                                   error, sizeof(error))) {
        printf("  blocks.%u.%s: FAIL find: %s\n", block,
               h3_w4a8_linear_kind_name(kind), error);
        return 0;
    }
    int8_t *w_host = NULL;
    float *s_channel = NULL;
    size_t hn = 0, hk = 0;
    if (!h3_w4a8_load_linear_int8(store, &quad, group_size, &w_host,
                                  &s_channel, &hn, &hk, error,
                                  sizeof(error))) {
        printf("  blocks.%u.%s: FAIL host int8: %s\n", block,
               h3_w4a8_linear_kind_name(kind), error);
        return 0;
    }

    float *x = (float *)malloc(batch * k * sizeof(float));
    float *x_rot = (float *)malloc(batch * k * sizeof(float));
    int8_t *x_i8 = (int8_t *)malloc(batch * k);
    float *act_scales = (float *)malloc(batch * sizeof(float));
    float *y_cpu = (float *)malloc(batch * n * sizeof(float));
    float *y_gpu = (float *)malloc(batch * n * sizeof(float));
    if (!x || !x_rot || !x_i8 || !act_scales || !y_cpu || !y_gpu) {
        printf("  blocks.%u.%s: FAIL OOM\n", block,
               h3_w4a8_linear_kind_name(kind));
        free(x);
        free(x_rot);
        free(x_i8);
        free(act_scales);
        free(y_cpu);
        free(y_gpu);
        free(w_host);
        free(s_channel);
        return 0;
    }

    const uint64_t seed = 0x4833434143484500ULL; /* H3CACHE */
    for (size_t i = 0; i < batch * k; i++) x[i] = hash_to_unit(seed, i);

    if (!h3_w4a8_convrot_act_f32(x, batch, k, convrot_gs, x_rot, error,
                                 sizeof(error)) ||
        !h3_w4a8_quantize_act_int8(x_rot, batch, k, H3_W4A8_ACT_QUANT_PER_ROW,
                                   x_i8, act_scales, error, sizeof(error))) {
        printf("  blocks.%u.%s: FAIL act: %s\n", block,
               h3_w4a8_linear_kind_name(kind), error);
        free(x);
        free(x_rot);
        free(x_i8);
        free(act_scales);
        free(y_cpu);
        free(y_gpu);
        free(w_host);
        free(s_channel);
        return 0;
    }

    h3_w4a8_gemm_i8_dequant_f32(x_i8, w_host, act_scales, 0, s_channel, batch,
                                n, k, y_cpu);

    char backend[32] = {0};
    double gemm_ms = 0.0;
    if (!h3_cuda_gemm_i8_dequant_f32_w(cuda, dw, x_i8, act_scales, 0, batch,
                                       y_gpu, &gemm_ms, backend, sizeof(backend),
                                       error, sizeof(error))) {
        printf("  blocks.%u.%s: FAIL gemm: %s\n", block,
               h3_w4a8_linear_kind_name(kind), error);
        free(x);
        free(x_rot);
        free(x_i8);
        free(act_scales);
        free(y_cpu);
        free(y_gpu);
        free(w_host);
        free(s_channel);
        return 0;
    }

    float max_abs = 0.0f;
    double rmse = 0.0;
    h3_w4a8_compare_f32(y_cpu, y_gpu, batch * n, &max_abs, &rmse);
    float ref_absmax = 0.0f;
    for (size_t i = 0; i < batch * n; i++) {
        float a = fabsf(y_cpu[i]);
        if (a > ref_absmax) ref_absmax = a;
    }
    const float thresh =
        atol + rtol * (ref_absmax > 0.0f ? ref_absmax : 1.0f);
    const int pass = (max_abs <= thresh);

    printf("  blocks.%u.%s: N=%zu K=%zu gemm_ms=%.3f backend=%s "
           "y_max_abs=%.6g %s\n",
           block, h3_w4a8_linear_kind_name(kind), n, k, gemm_ms, backend,
           max_abs, pass ? "PASS" : "FAIL");

    free(x);
    free(x_rot);
    free(x_i8);
    free(act_scales);
    free(y_cpu);
    free(y_gpu);
    free(w_host);
    free(s_channel);
    return pass;
}

/* Time sequential unpack of every block (load then free) for full-50 cost. */
static int timed_unpack_all(h3_cuda *cuda, h3_w4a8_store *store,
                            unsigned n_blocks, int group_size,
                            double *elapsed_ms, char *error,
                            size_t error_size) {
    clock_t t0 = clock();
    for (unsigned b = 0; b < n_blocks; b++) {
        for (int k = 0; k < H3_W4A8_LINEAR_COUNT; k++) {
            h3_w4a8_quad quad;
            if (!h3_w4a8_find_block_linear(
                    store, b,
                    h3_w4a8_linear_kind_suffix((h3_w4a8_linear_kind)k), &quad,
                    error, error_size))
                return 0;
            size_t nbytes = 0;
            int8_t *packed = (int8_t *)h3_w4a8_dup_tensor(
                store, quad.weight, &nbytes, error, error_size);
            float *codebook = (float *)h3_w4a8_dup_tensor(
                store, quad.weight_codebook, &nbytes, error, error_size);
            uint8_t *s_rel = (uint8_t *)h3_w4a8_dup_tensor(
                store, quad.weight_s_rel, &nbytes, error, error_size);
            float *s_channel = (float *)h3_w4a8_dup_tensor(
                store, quad.weight_s_channel, &nbytes, error, error_size);
            if (!packed || !codebook || !s_rel || !s_channel) {
                free(packed);
                free(codebook);
                free(s_rel);
                free(s_channel);
                return 0;
            }
            h3_cuda_w4a8_weight *dw = NULL;
            size_t n = (size_t)quad.weight->shape[0];
            size_t k_half = (size_t)quad.weight->shape[1];
            if (!h3_cuda_w4a8_weight_create(cuda, packed, n, k_half, codebook,
                                            s_rel, s_channel, group_size, &dw,
                                            NULL, NULL, error, error_size)) {
                free(packed);
                free(codebook);
                free(s_rel);
                free(s_channel);
                return 0;
            }
            h3_cuda_w4a8_weight_free(dw);
            free(packed);
            free(codebook);
            free(s_rel);
            free(s_channel);
        }
    }
    clock_t t1 = clock();
    if (elapsed_ms)
        *elapsed_ms = 1000.0 * (double)(t1 - t0) / (double)CLOCKS_PER_SEC;
    return 1;
}

int main(int argc, char **argv) {
    const char *policy_s = "resident_all";
    unsigned window = 0;
    unsigned n_blocks = H3_W4A8_CACHE_DEFAULT_BLOCKS;
    const char *sample_s = "0,24,49";
    size_t batch = 4;
    float atol = 1e-3f;
    float rtol = 1e-4f;
    int device = 0;
    int skip_unpack_all = 0;
    const char *path = NULL;
    char b1[64], b2[64], b3[64], b4[64];

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--policy") && i + 1 < argc) {
            policy_s = argv[++i];
        } else if (!strcmp(argv[i], "--window") && i + 1 < argc) {
            window = (unsigned)strtoul(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--blocks") && i + 1 < argc) {
            n_blocks = (unsigned)strtoul(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--sample") && i + 1 < argc) {
            sample_s = argv[++i];
        } else if (!strcmp(argv[i], "--batch") && i + 1 < argc) {
            batch = (size_t)strtoul(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--atol") && i + 1 < argc) {
            atol = strtof(argv[++i], NULL);
        } else if (!strcmp(argv[i], "--rtol") && i + 1 < argc) {
            rtol = strtof(argv[++i], NULL);
        } else if (!strcmp(argv[i], "--device") && i + 1 < argc) {
            device = (int)strtol(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--skip-unpack-all")) {
            skip_unpack_all = 1;
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            fprintf(stderr,
                    "usage: %s [--policy resident_all|window|lru] "
                    "[--window N] [--blocks 50] [--sample 0,24,49] "
                    "[--batch M] [--skip-unpack-all] FILE.safetensors\n",
                    argv[0]);
            return 0;
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "unknown option: %s\n", argv[i]);
            return 2;
        } else {
            path = argv[i];
        }
    }
    if (!path || n_blocks == 0 || batch == 0) {
        fprintf(stderr, "usage: %s FILE.safetensors\n", argv[0]);
        return 2;
    }

    h3_w4a8_cache_policy policy = parse_policy(policy_s);
    if ((int)policy < 0) {
        fprintf(stderr, "bad --policy: %s\n", policy_s);
        return 2;
    }

    unsigned samples[MAX_SAMPLE];
    int n_samples = 0;
    if (!parse_blocks_list(sample_s, samples, &n_samples, MAX_SAMPLE)) {
        fprintf(stderr, "bad --sample: %s\n", sample_s);
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
    size_t free0 = 0, total0 = 0;
    h3_cuda_mem_info(cuda, &free0, &total0, NULL, 0);

    printf("path: %s\n", path);
    printf("cuda_device: [%d] %s sm_%d%d\n", info.device_index, info.name,
           info.compute_major, info.compute_minor);
    printf("vram_before: free=%s  total=%s\n",
           format_bytes(free0, b1, sizeof(b1)),
           format_bytes(total0, b2, sizeof(b2)));

    h3_w4a8_cache_config cfg;
    h3_w4a8_cache_config_init(&cfg);
    cfg.n_blocks = n_blocks;
    cfg.policy = policy;
    cfg.window_size = window;

    h3_w4a8_cache *cache =
        h3_w4a8_cache_open(cuda, store, &cfg, error, sizeof(error));
    if (!cache) {
        fprintf(stderr, "cache open failed: %s\n", error);
        h3_cuda_free(cuda);
        h3_w4a8_store_free(store);
        return 1;
    }

    h3_w4a8_cache_stats st;
    if (!h3_w4a8_cache_get_stats(cache, &st, error, sizeof(error))) {
        fprintf(stderr, "stats failed: %s\n", error);
        h3_w4a8_cache_free(cache);
        h3_cuda_free(cuda);
        h3_w4a8_store_free(store);
        return 1;
    }

    printf("policy: requested=%s effective=%s%s window=%u\n",
           h3_w4a8_cache_policy_name(st.requested_policy),
           h3_w4a8_cache_policy_name(st.effective_policy),
           st.auto_downgraded ? " (auto-downgraded)" : "", st.window_size);
    printf("estimate: per_block=%s  all_%u=%s\n",
           format_bytes(st.bytes_per_block_est, b1, sizeof(b1)), n_blocks,
           format_bytes(st.bytes_estimated_all, b2, sizeof(b2)));
    printf("capacity: max_blocks_fit=%u / %u  budget=%s\n", st.max_blocks_fit,
           n_blocks, format_bytes(st.vram_budget_bytes, b3, sizeof(b3)));

    double unpack_all_ms = 0.0;
    if (!skip_unpack_all) {
        printf("unpack_all_%u: timing sequential load+free (cache still empty)...\n",
               n_blocks);
        if (!timed_unpack_all(cuda, store, n_blocks, group_size, &unpack_all_ms,
                              error, sizeof(error))) {
            fprintf(stderr, "unpack_all failed: %s\n", error);
            h3_w4a8_cache_free(cache);
            h3_cuda_free(cuda);
            h3_w4a8_store_free(store);
            return 1;
        }
        printf("unpack_all_%u: wall_ms=%.1f  (avg_block=%.1f ms)\n", n_blocks,
               unpack_all_ms, unpack_all_ms / (double)n_blocks);
    }

    double warm_ms = 0.0;
    unsigned warm_n = n_blocks;
    if (st.effective_policy != H3_W4A8_CACHE_RESIDENT_ALL)
        warm_n = st.window_size;
    if (!h3_w4a8_cache_prefetch(cache, 0, warm_n, &warm_ms, error,
                                sizeof(error))) {
        fprintf(stderr, "prefetch failed: %s\n", error);
        h3_w4a8_cache_free(cache);
        h3_cuda_free(cuda);
        h3_w4a8_store_free(store);
        return 1;
    }

    if (!h3_w4a8_cache_get_stats(cache, &st, error, sizeof(error))) {
        fprintf(stderr, "stats failed: %s\n", error);
        h3_w4a8_cache_free(cache);
        h3_cuda_free(cuda);
        h3_w4a8_store_free(store);
        return 1;
    }

    printf("warmup_resident: blocks=%u linears=%u wall_ms=%.1f\n",
           st.n_resident_blocks, st.n_resident_linears, warm_ms);
    printf("resident: %s\n",
           format_bytes(st.bytes_resident, b1, sizeof(b1)));
    printf("vram_after_warmup: free=%s  total=%s  used_delta≈%s\n",
           format_bytes(st.vram_free_bytes, b2, sizeof(b2)),
           format_bytes(st.vram_total_bytes, b3, sizeof(b3)),
           format_bytes(free0 > st.vram_free_bytes ? free0 - st.vram_free_bytes
                                                   : 0,
                        b4, sizeof(b4)));

    printf("gemm_smoke: sample=%s batch=%zu (out,fc2)\n", sample_s, batch);
    int pass = 0, fail = 0;
    for (int i = 0; i < n_samples; i++) {
        unsigned b = samples[i];
        if (b >= n_blocks) {
            printf("  sample block %u out of range — SKIP\n", b);
            continue;
        }
        if (gemm_smoke(cuda, cache, store, b, H3_W4A8_LINEAR_OUT, group_size,
                       convrot_gs, batch, atol, rtol))
            pass++;
        else
            fail++;
        if (gemm_smoke(cuda, cache, store, b, H3_W4A8_LINEAR_FC2, group_size,
                       convrot_gs, batch, atol, rtol))
            pass++;
        else
            fail++;
    }

    printf("summary: fit=%u/%u resident=%u warm_ms=%.1f unpack_all_ms=%.1f "
           "gemm_pass=%d fail=%d → %s\n",
           st.max_blocks_fit, n_blocks, st.n_resident_blocks, warm_ms,
           unpack_all_ms, pass, fail, fail == 0 ? "PASS" : "FAIL");

    h3_w4a8_cache_free(cache);
    h3_cuda_free(cuda);
    h3_w4a8_store_free(store);
    return fail == 0 ? 0 : 1;
}
