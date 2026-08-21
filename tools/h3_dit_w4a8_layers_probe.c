/* P4 M1: multi-block DiT W4A8 chain (CUDA linears + CPU non-quant ops).
 * Build:
 *   cmake --build build-cuda -j --target h3_dit_w4a8_layers_probe
 *   ./build-cuda/h3_dit_w4a8_layers_probe --layers 4 [--batch M] FILE.safetensors
 */
#include "h3_dit_w4a8.h"

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
    return (float)(int32_t)(h >> 32) * (1.0f / 2147483648.0f);
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

static int all_finite(const float *x, size_t n) {
    size_t i;
    for (i = 0; i < n; i++)
        if (!isfinite(x[i])) return 0;
    return 1;
}

static float absmax(const float *x, size_t n) {
    float m = 0.0f;
    size_t i;
    for (i = 0; i < n; i++) {
        float a = fabsf(x[i]);
        if (a > m) m = a;
    }
    return m;
}

int main(int argc, char **argv) {
    unsigned first = 0;
    unsigned layers = 4;
    size_t batch = 4;
    size_t t_row = 512;
    int device = 0;
    int policy_window = 0;
    unsigned window_size = 0;
    const char *path = NULL;
    char error[512];
    char b1[64], b2[64], b3[64];
    h3_w4a8_store *store = NULL;
    h3_cuda *cuda = NULL;
    h3_dit_w4a8 *ctx = NULL;
    h3_w4a8_cache_config cfg;
    h3_w4a8_cache_stats cst;
    h3_cuda_device_info info;
    h3_dit_w4a8_block_aux aux0;
    h3_dit_w4a8_layers_stats lst;
    float *x = NULL, *y = NULL, *cos = NULL, *sin = NULL;
    int *pt = NULL, *ph = NULL, *pw = NULL;
    uint32_t *row_map = NULL;
    size_t free0 = 0, total0 = 0;
    size_t i;
    int rc = 1;
    int iarg;

    memset(&aux0, 0, sizeof(aux0));

    for (iarg = 1; iarg < argc; iarg++) {
        if (!strcmp(argv[iarg], "--layers") && iarg + 1 < argc) {
            layers = (unsigned)strtoul(argv[++iarg], NULL, 10);
        } else if (!strcmp(argv[iarg], "--first") && iarg + 1 < argc) {
            first = (unsigned)strtoul(argv[++iarg], NULL, 10);
        } else if (!strcmp(argv[iarg], "--batch") && iarg + 1 < argc) {
            batch = (size_t)strtoul(argv[++iarg], NULL, 10);
        } else if (!strcmp(argv[iarg], "--t-row") && iarg + 1 < argc) {
            t_row = (size_t)strtoul(argv[++iarg], NULL, 10);
        } else if (!strcmp(argv[iarg], "--device") && iarg + 1 < argc) {
            device = (int)strtol(argv[++iarg], NULL, 10);
        } else if (!strcmp(argv[iarg], "--window") && iarg + 1 < argc) {
            policy_window = 1;
            window_size = (unsigned)strtoul(argv[++iarg], NULL, 10);
        } else if (!strcmp(argv[iarg], "-h") || !strcmp(argv[iarg], "--help")) {
            fprintf(stderr,
                    "usage: %s [--layers N] [--first B] [--batch M] "
                    "[--t-row R] [--window W] FILE.safetensors\n",
                    argv[0]);
            return 0;
        } else if (argv[iarg][0] == '-') {
            fprintf(stderr, "unknown option: %s\n", argv[iarg]);
            return 2;
        } else {
            path = argv[iarg];
        }
    }
    if (!path || layers == 0 || batch == 0) {
        fprintf(stderr, "usage: %s --layers N FILE.safetensors\n", argv[0]);
        return 2;
    }
    if ((size_t)first + (size_t)layers > H3_DIT_W4A8_N_BLOCKS) {
        fprintf(stderr, "block range %u..%u out of 0..%d\n", first,
                first + layers - 1, H3_DIT_W4A8_N_BLOCKS - 1);
        return 2;
    }

    store = h3_w4a8_store_open(path, error, sizeof(error));
    if (!store) {
        fprintf(stderr, "open failed: %s\n", error);
        return 1;
    }
    cuda = h3_cuda_create(device, error, sizeof(error));
    if (!cuda) {
        fprintf(stderr, "cuda create failed: %s\n", error);
        goto cleanup;
    }
    h3_cuda_get_info(cuda, &info);
    h3_cuda_mem_info(cuda, &free0, &total0, NULL, 0);

    h3_dit_w4a8_config_init(&cfg);
    if (policy_window) {
        cfg.policy = H3_W4A8_CACHE_WINDOW;
        cfg.window_size = window_size ? window_size : layers;
    } else {
        cfg.policy = H3_W4A8_CACHE_RESIDENT_ALL;
    }
    ctx = h3_dit_w4a8_open(cuda, store, &cfg, error, sizeof(error));
    if (!ctx) {
        fprintf(stderr, "dit w4a8 open failed: %s\n", error);
        goto cleanup;
    }

    printf("path: %s\n", path);
    printf("cuda_device: [%d] %s sm_%d%d\n", info.device_index, info.name,
           info.compute_major, info.compute_minor);
    printf("vram_before: free=%s  total=%s\n",
           format_bytes(free0, b1, sizeof(b1)),
           format_bytes(total0, b2, sizeof(b2)));
    printf("range: blocks.%u .. blocks.%u  (n=%u)  seq=%zu  t_row=%zu\n", first,
           first + layers - 1, layers, batch, t_row);

    if (!h3_dit_w4a8_load_block_aux(store, first, &aux0, error, sizeof(error))) {
        fprintf(stderr, "load aux failed: %s\n", error);
        goto cleanup;
    }
    if (t_row >= aux0.t_table_rows) {
        fprintf(stderr, "t_row %zu >= t_table_rows %zu\n", t_row,
                aux0.t_table_rows);
        goto cleanup;
    }

    x = (float *)malloc(batch * H3_DIT_W4A8_HIDDEN * sizeof(float));
    y = (float *)malloc(batch * H3_DIT_W4A8_HIDDEN * sizeof(float));
    cos = (float *)malloc(batch * H3_DIT_W4A8_ROPE_HALF * sizeof(float));
    sin = (float *)malloc(batch * H3_DIT_W4A8_ROPE_HALF * sizeof(float));
    pt = (int *)malloc(batch * sizeof(int));
    ph = (int *)malloc(batch * sizeof(int));
    pw = (int *)malloc(batch * sizeof(int));
    row_map = (uint32_t *)calloc(batch, sizeof(uint32_t));
    if (!x || !y || !cos || !sin || !pt || !ph || !pw || !row_map) {
        fprintf(stderr, "OOM\n");
        goto cleanup;
    }
    for (i = 0; i < batch * H3_DIT_W4A8_HIDDEN; i++)
        x[i] = hash_to_unit(0x4C415945525331ULL, i);
    for (i = 0; i < batch; i++) {
        pt[i] = (int)i;
        ph[i] = (int)(i % 2);
        pw[i] = (int)(i % 4);
    }
    if (!h3_dit_w4a8_rope_tables_f32(aux0.inv_freq, batch, pt, ph, pw, 1.0f, cos,
                                     sin, error, sizeof(error))) {
        fprintf(stderr, "rope failed: %s\n", error);
        goto cleanup;
    }

    memset(&lst, 0, sizeof(lst));
    if (!h3_dit_w4a8_layers_forward_f32(ctx, first, layers, t_row, x, batch,
                                        row_map, cos, sin, y, 1, &lst, error,
                                        sizeof(error))) {
        fprintf(stderr, "layers_forward failed: %s\n", error);
        goto cleanup;
    }
    if (!h3_dit_w4a8_get_cache_stats(ctx, &cst, error, sizeof(error))) {
        fprintf(stderr, "stats failed: %s\n", error);
        goto cleanup;
    }

    printf("prefetch_ms=%.1f  blocks_ms=%.1f  total_ms=%.1f  gemm_ms_sum=%.1f\n",
           lst.prefetch_ms, lst.blocks_ms, lst.total_ms, lst.gemm_ms_sum);
    printf("policy: requested=%s effective=%s  resident_blocks=%u "
           "resident=%s\n",
           h3_w4a8_cache_policy_name(cst.requested_policy),
           h3_w4a8_cache_policy_name(cst.effective_policy),
           cst.n_resident_blocks,
           format_bytes(cst.bytes_resident, b1, sizeof(b1)));
    printf("vram_after: free=%s  used_delta≈%s\n",
           format_bytes(cst.vram_free_bytes, b2, sizeof(b2)),
           format_bytes(free0 > cst.vram_free_bytes ? free0 - cst.vram_free_bytes
                                                    : 0,
                        b3, sizeof(b3)));
    printf("y: absmax=%.6g  finite=%d  checksum=0x%016llx\n",
           absmax(y, batch * H3_DIT_W4A8_HIDDEN),
           all_finite(y, batch * H3_DIT_W4A8_HIDDEN),
           (unsigned long long)h3_dit_w4a8_checksum_f32(
               y, batch * H3_DIT_W4A8_HIDDEN));

    if (lst.all_finite && all_finite(y, batch * H3_DIT_W4A8_HIDDEN)) {
        printf("summary: layers=%u pass=1 fail=0 → PASS\n", layers);
        rc = 0;
    } else {
        printf("summary: layers=%u → FAIL (non-finite)\n", layers);
        rc = 1;
    }

cleanup:
    free(x);
    free(y);
    free(cos);
    free(sin);
    free(pt);
    free(ph);
    free(pw);
    free(row_map);
    h3_dit_w4a8_block_aux_free(&aux0);
    h3_dit_w4a8_free(ctx);
    h3_cuda_free(cuda);
    h3_w4a8_store_free(store);
    return rc;
}
