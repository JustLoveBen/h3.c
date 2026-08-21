/* Host-only W4A8 / single-file safetensors probe (P3).
 * No Metal / CUDA / DiT runtime. Build:
 *   cc -std=c11 -O2 -I. -o h3_w4a8_probe \
 *      tools/h3_w4a8_probe.c h3_safetensors.c h3_w4a8.c
 */
#include "h3_w4a8.h"

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
    printf("    %-16s %s  dtype=%-8s nbytes=%llu  shape=",
           role,
           tensor->name ? tensor->name : "?",
           h3_dtype_name(tensor->dtype),
           (unsigned long long)h3_st_tensor_nbytes(tensor));
    print_shape(tensor);
    fputc('\n', stdout);
}

static int probe_quad(h3_w4a8_store *store, unsigned block, const char *suffix,
                      int *host_ok) {
    h3_w4a8_quad quad;
    char error[512];
    if (!h3_w4a8_find_block_linear(store, block, suffix, &quad, error,
                                   sizeof(error))) {
        printf("  blocks.%u.%s: FAIL (%s)\n", block, suffix, error);
        return 0;
    }
    printf("  blocks.%u.%s: OK\n", block, suffix);
    print_tensor("weight", quad.weight);
    print_tensor("codebook", quad.weight_codebook);
    print_tensor("s_channel", quad.weight_s_channel);
    print_tensor("s_rel", quad.weight_s_rel);

    /* Host copy smoke test on codebook (tiny F32[16]). */
    size_t bytes = 0;
    void *dup = h3_w4a8_dup_tensor(store, quad.weight_codebook, &bytes, error,
                                   sizeof(error));
    if (!dup) {
        printf("    codebook host read: FAIL (%s)\n", error);
        *host_ok = 0;
        return 1;
    }
    printf("    codebook host read: OK (%zu bytes)\n", bytes);
    free(dup);
    return 1;
}

int main(int argc, char **argv) {
    unsigned block = 0;
    const char *path = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--block") && i + 1 < argc) {
            block = (unsigned)strtoul(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            fprintf(stderr,
                    "usage: %s [--block N] FILE.safetensors\n",
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

    char error[512];
    h3_w4a8_store *store = h3_w4a8_store_open(path, error, sizeof(error));
    if (!store) {
        fprintf(stderr, "open failed: %s\n", error);
        return 1;
    }

    const h3_st_header *header = h3_w4a8_store_header(store);
    const h3_st_quant_info *quant = h3_w4a8_store_quant(store);

    printf("path: %s\n", path);
    printf("file_size: %llu\n", (unsigned long long)header->file_size);
    printf("header_size: %llu\n", (unsigned long long)header->header_size);
    printf("tensors: %zu\n", header->tensor_count);
    printf("has_metadata: %s\n", quant->has_metadata ? "yes" : "no");
    printf("format: %s\n", quant->format[0] ? quant->format : "(unknown)");
    printf("group_size: %d\n", quant->group_size);
    printf("convrot: %d\n", quant->convrot);
    printf("convrot_groupsize: %d\n", quant->convrot_groupsize);

    static const char *suffixes[] = {
        H3_W4A8_SUFFIX_QKV,
        H3_W4A8_SUFFIX_OUT,
        H3_W4A8_SUFFIX_FC1,
        H3_W4A8_SUFFIX_FC2,
    };
    printf("block %u linears:\n", block);
    int found = 0;
    int host_ok = 1;
    for (size_t i = 0; i < sizeof(suffixes) / sizeof(suffixes[0]); i++) {
        if (probe_quad(store, block, suffixes[i], &host_ok)) found++;
    }
    printf("summary: found %d/4 W4A8 quads for block %u; host_read=%s\n",
           found, block, host_ok ? "ok" : "fail");

    h3_w4a8_store_free(store);
    return found == 4 && host_ok ? 0 : 1;
}
