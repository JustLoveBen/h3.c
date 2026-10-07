/* Precompute NVFP4 → INT8 sidecar next to the pruned_nvfp4 file.
 * Independent of gold --dit. Probe failure hard-exits. Weights stay out of git. */
#include "h3_dit_nvfp4.h"
#include "h3_nvfp4.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *k_suffix[4] = {
    H3_DIT_NVFP4_SUFFIX_QKV,
    H3_DIT_NVFP4_SUFFIX_OUT,
    H3_DIT_NVFP4_SUFFIX_FC1,
    H3_DIT_NVFP4_SUFFIX_FC2,
};
static const char *k_name[4] = {"qkv", "out", "fc1", "fc2"};

int main(int argc, char **argv) {
    const char *path;
    h3_nvfp4_store store;
    char err[512];
    unsigned b, k;
    unsigned hits = 0, writes = 0, misses = 0;
    memset(&store, 0, sizeof(store));
    if (argc < 2) {
        fprintf(stderr,
                "usage: h3_nvfp4_int8_sidecar <pruned_nvfp4.safetensors>\n");
        return 2;
    }
    path = argv[1];
    if (!h3_nvfp4_store_open(path, &store, err, sizeof(err))) {
        fprintf(stderr, "open failed: %s\n", err);
        return 1;
    }
    if (!h3_dit_nvfp4_probe(&store, err, sizeof(err))) {
        fprintf(stderr, "%s\n", err);
        h3_nvfp4_store_free(&store);
        return 1;
    }
    for (b = 0; b < 50; b++) {
        for (k = 0; k < 4; k++) {
            h3_nvfp4_linear lin;
            int8_t *i8 = NULL;
            float *s = NULL;
            size_t n = 0, kk = 0;
            if (!h3_dit_nvfp4_find_block_linear(&store, b, k_suffix[k], &lin,
                                                err, sizeof(err))) {
                fprintf(stderr, "find b%u %s: %s\n", b, k_name[k], err);
                h3_nvfp4_store_free(&store);
                return 1;
            }
            n = lin.n;
            kk = lin.k;
            i8 = (int8_t *)malloc(n * kk);
            s = (float *)malloc(n * sizeof(float));
            if (i8 && s &&
                h3_dit_nvfp4_sidecar_try_load(path, b, k_name[k], n, kk, i8,
                                              s)) {
                hits++;
                free(i8);
                free(s);
                continue;
            }
            free(i8);
            free(s);
            i8 = NULL;
            s = NULL;
            misses++;
            if (!h3_dit_nvfp4_dequant_to_i8(&store, &lin, &i8, &s, &n, &kk, err,
                                            sizeof(err))) {
                fprintf(stderr, "dequant b%u %s: %s\n", b, k_name[k], err);
                h3_nvfp4_store_free(&store);
                return 1;
            }
            if (!h3_dit_nvfp4_sidecar_save(path, b, k_name[k], n, kk, i8, s, err,
                                           sizeof(err))) {
                fprintf(stderr, "save b%u %s: %s\n", b, k_name[k], err);
                free(i8);
                free(s);
                h3_nvfp4_store_free(&store);
                return 1;
            }
            writes++;
            free(i8);
            free(s);
        }
        fprintf(stderr, "sidecar block %u/50 hits=%u misses=%u writes=%u\n",
                b + 1, hits, misses, writes);
    }
    h3_nvfp4_store_free(&store);
    printf("nvfp4_int8_sidecar done hits=%u misses=%u writes=%u\n", hits, misses,
           writes);
    return 0;
}
