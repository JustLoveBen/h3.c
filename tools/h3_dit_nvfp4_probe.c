/* Host-first NVFP4 DiT header / probe (not gold).
 *   cmake --build build-cuda-sm120 -j --target h3_dit_nvfp4_probe
 *   ./build-cuda-sm120/h3_dit_nvfp4_probe FILE.safetensors
 */
#include "h3_dit_nvfp4.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_shape(const h3_st_tensor *t) {
    int i;
    fputc('[', stdout);
    for (i = 0; i < t->ndim; i++) {
        if (i) fputc(',', stdout);
        printf("%llu", (unsigned long long)t->shape[i]);
    }
    fputc(']', stdout);
}

static void print_tensor(const h3_st_header *hdr, const char *name) {
    const h3_st_tensor *t = h3_st_find(hdr, name);
    if (!t) {
        printf("  %-56s MISSING\n", name);
        return;
    }
    printf("  %-56s %-10s nbytes=%llu shape=", name, h3_dtype_name(t->dtype),
           (unsigned long long)h3_st_tensor_nbytes(t));
    print_shape(t);
    fputc('\n', stdout);
}

int main(int argc, char **argv) {
    const char *path = NULL;
    char error[512];
    h3_nvfp4_store store;
    h3_nvfp4_linear qkv;
    float *rows = NULL;
    size_t n_rows = 0;
    int i;
    static const char *keys[] = {
        "blocks.0.attn.qkv_proj.weight",
        "blocks.0.attn.qkv_proj.weight_scale",
        "blocks.0.attn.qkv_proj.weight_scale_2",
        "blocks.0.attn.out_proj.weight",
        "blocks.0.mlp.fc1.weight",
        "blocks.0.mlp.fc2.weight",
        "blocks.0.norm1.weight",
        "blocks.0.adaln_proj.linear.weight",
        "video_patch_proj.weight",
        "audio_patch_proj.weight",
        "final_layer.video_out.weight",
        "final_layer.audio_out.weight",
        "condition_proj.weight",
        "token_refiner.blocks.0.attn.qkv_proj.weight",
        "token_refiner.final_norm.weight",
        "adaln_t_table",
        "rope.inv_freq",
        NULL};

    memset(&store, 0, sizeof(store));
    store.map_fd = -1;
    for (i = 1; i < argc; i++) {
        if (argv[i][0] == '-') {
            fprintf(stderr, "unknown option: %s\n", argv[i]);
            return 2;
        }
        path = argv[i];
    }
    if (!path) {
        fprintf(stderr, "usage: %s FILE.safetensors\n", argv[0]);
        return 2;
    }
    if (!h3_nvfp4_store_open(path, &store, error, sizeof(error))) {
        fprintf(stderr, "open failed（不是金线；不回退 INT8）: %s\n", error);
        return 1;
    }
    printf("path: %s\n", path);
    printf("tensors: %zu  format=%s\n", store.header.tensor_count,
           store.quant.format[0] ? store.quant.format : "(none)");
    printf("key tensors:\n");
    for (i = 0; keys[i]; i++)
        print_tensor(&store.header, keys[i]);

    if (!h3_dit_nvfp4_probe(&store, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        h3_nvfp4_store_free(&store);
        return 1;
    }
    if (!h3_dit_nvfp4_find_block_linear(&store, 0, H3_DIT_NVFP4_SUFFIX_QKV, &qkv,
                                        error, sizeof(error))) {
        fprintf(stderr, "find qkv: %s\n", error);
        h3_nvfp4_store_free(&store);
        return 1;
    }
    rows = h3_nvfp4_dequant_rows_f32(&store, &qkv, 128, &n_rows, error,
                                     sizeof(error));
    if (!rows) {
        fprintf(stderr, "dequant qkv: %s\n", error);
        h3_nvfp4_store_free(&store);
        return 1;
    }
    printf("dequant_qkv: rows=%zu k=%zu elem0=%g (非金线)\n", n_rows, qkv.k,
           (double)rows[0]);
    free(rows);
    h3_nvfp4_store_free(&store);
    printf("dit_nvfp4_probe: PASS\n");
    return 0;
}
