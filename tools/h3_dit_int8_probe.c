/* Host-only INT8 ConvRot DiT probe (theme-follow secondary path).
 *   cc -std=c11 -O2 -I. -o h3_dit_int8_probe \
 *      tools/h3_dit_int8_probe.c h3_safetensors.c h3_dit_int8.c
 */
#include "h3_dit_int8.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_shape(const h3_st_tensor *tensor) {
    fputc('[', stdout);
    for (int i = 0; i < tensor->ndim; i++) {
        if (i) fputc(',', stdout);
        printf("%llu", (unsigned long long)tensor->shape[i]);
    }
    fputc(']', stdout);
}

static void print_tensor(const char *role, const h3_st_tensor *tensor) {
    if (!tensor) {
        printf("    %-12s (missing)\n", role);
        return;
    }
    printf("    %-12s %s  dtype=%-8s nbytes=%llu  shape=", role,
           tensor->name ? tensor->name : "?", h3_dtype_name(tensor->dtype),
           (unsigned long long)h3_st_tensor_nbytes(tensor));
    print_shape(tensor);
    fputc('\n', stdout);
}

static int probe_linear(h3_dit_int8_store *store, unsigned block,
                        const char *suffix, int *dequant_ok) {
    h3_dit_int8_linear lin;
    char error[512];
    float *dq;
    size_t sample;
    if (!h3_dit_int8_find_block_linear(store, block, suffix, &lin, error,
                                       sizeof(error))) {
        printf("  blocks.%u.%s: FAIL (%s)\n", block, suffix, error);
        return 0;
    }
    printf("  blocks.%u.%s: OK  N=%zu K=%zu convrot_gs=%d\n", block, suffix,
           lin.n, lin.k, h3_dit_int8_convrot_groupsize(store, lin.k));
    print_tensor("weight", lin.weight);
    print_tensor("scale", lin.scale);
    print_tensor("comfy_quant", lin.comfy_quant);

    /* Smoke: dequant first 8 rows only via full alloc would be huge for QKV;
     * just read scale host bytes. */
    sample = lin.n * sizeof(float);
    {
        float *scales = (float *)malloc(sample);
        if (!scales) {
            printf("    scale host read: FAIL OOM\n");
            *dequant_ok = 0;
            return 1;
        }
        if (!h3_dit_int8_read_tensor(store, lin.scale, scales, sample, error,
                                     sizeof(error))) {
            printf("    scale host read: FAIL (%s)\n", error);
            free(scales);
            *dequant_ok = 0;
            return 1;
        }
        printf("    scale host read: OK (first=%g)\n", (double)scales[0]);
        free(scales);
    }

    if (lin.n * lin.k <= 8ull * 1024ull * 1024ull) {
        dq = h3_dit_int8_dequant_f32(store, &lin, error, sizeof(error));
        if (!dq) {
            printf("    dequant: FAIL (%s)\n", error);
            *dequant_ok = 0;
        } else {
            printf("    dequant: OK (elem0=%g)\n", (double)dq[0]);
            free(dq);
        }
    } else {
        printf("    dequant: skipped (N*K=%zu too large for smoke)\n",
               lin.n * lin.k);
    }
    return 1;
}

int main(int argc, char **argv) {
    unsigned block = 0;
    const char *path = NULL;
    h3_dit_int8_store *store;
    char error[512];
    const h3_st_quant_info *q;
    const h3_st_header *h;
    int ok = 1;
    int dequant_ok = 1;
    static const char *suffixes[] = {
        H3_DIT_INT8_SUFFIX_QKV, H3_DIT_INT8_SUFFIX_OUT, H3_DIT_INT8_SUFFIX_FC1,
        H3_DIT_INT8_SUFFIX_FC2};
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--block") && i + 1 < argc) {
            block = (unsigned)strtoul(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            fprintf(stderr, "usage: %s [--block N] FILE.safetensors\n",
                    argv[0]);
            return 0;
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "unknown option: %s\n", argv[i]);
            return 2;
        } else {
            path = argv[i];
        }
    }
    if (!path) {
        fprintf(stderr, "usage: %s [--block N] FILE.safetensors\n", argv[0]);
        return 2;
    }
    store = h3_dit_int8_store_open(path, error, sizeof(error));
    if (!store) {
        fprintf(stderr, "open FAIL: %s\n", error);
        return 1;
    }
    h = h3_dit_int8_store_header(store);
    q = h3_dit_int8_store_quant(store);
    printf("INT8 DiT store OK\n");
    printf("  path=%s\n", path);
    printf("  tensors=%zu  format=%s  convrot=%d  convrot_gs=%d\n",
           h->tensor_count, q->format[0] ? q->format : "?", q->convrot,
           q->convrot_groupsize);
    for (size_t i = 0; i < sizeof(suffixes) / sizeof(suffixes[0]); i++) {
        if (!probe_linear(store, block, suffixes[i], &dequant_ok)) ok = 0;
    }
    h3_dit_int8_store_free(store);
    if (!ok || !dequant_ok) return 1;
    printf("ALL OK\n");
    return 0;
}
