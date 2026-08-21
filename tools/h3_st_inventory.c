/* Host-only safetensors header inventory + prequant detect (P3).
 * Pure C acceptance path (no scripting runtime).
 *
 *   make h3_st_inventory
 *   ./h3_st_inventory FILE.safetensors
 *   ./h3_st_inventory --w4a8-block 0 FILE.safetensors
 */
#include "h3_safetensors.h"
#include "h3_w4a8.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *needle;
    size_t count;
} h3_count_slot;

static int name_has(const char *name, const char *needle) {
    return name && needle && strstr(name, needle) != NULL;
}

static int name_ends_weight(const char *name) {
    size_t n;
    if (!name) return 0;
    n = strlen(name);
    return n >= 7 && !strcmp(name + n - 7, ".weight");
}

static void print_shape(const h3_st_tensor *tensor) {
    fputc('[', stdout);
    for (int i = 0; i < tensor->ndim; i++) {
        if (i) fputc(',', stdout);
        printf("%llu", (unsigned long long)tensor->shape[i]);
    }
    fputc(']', stdout);
}

static void bump_dtype(size_t *counts, size_t n, h3_dtype dtype) {
    if ((size_t)dtype < n) counts[dtype]++;
}

static void print_tensor_line(const h3_st_tensor *tensor) {
    printf("  %s  %s  nbytes=%llu  shape=",
           tensor->name ? tensor->name : "?",
           h3_dtype_name(tensor->dtype),
           (unsigned long long)h3_st_tensor_nbytes(tensor));
    print_shape(tensor);
    fputc('\n', stdout);
}

static int probe_w4a8_block(const char *path, unsigned block) {
    char error[512];
    h3_w4a8_store *store = h3_w4a8_store_open(path, error, sizeof(error));
    if (!store) {
        fprintf(stderr, "w4a8 open failed: %s\n", error);
        return 0;
    }
    static const char *suffixes[] = {
        H3_W4A8_SUFFIX_QKV,
        H3_W4A8_SUFFIX_OUT,
        H3_W4A8_SUFFIX_FC1,
        H3_W4A8_SUFFIX_FC2,
    };
    int found = 0;
    int host_ok = 1;
    printf("-- w4a8 block %u (load_block direction) --\n", block);
    for (size_t i = 0; i < sizeof(suffixes) / sizeof(suffixes[0]); i++) {
        h3_w4a8_quad quad;
        if (!h3_w4a8_find_block_linear(store, block, suffixes[i], &quad, error,
                                       sizeof(error))) {
            printf("  blocks.%u.%s: FAIL (%s)\n", block, suffixes[i], error);
            continue;
        }
        found++;
        printf("  blocks.%u.%s: OK\n", block, suffixes[i]);
        print_tensor_line(quad.weight);
        print_tensor_line(quad.weight_codebook);
        print_tensor_line(quad.weight_s_channel);
        print_tensor_line(quad.weight_s_rel);
        size_t bytes = 0;
        void *dup = h3_w4a8_dup_tensor(store, quad.weight_codebook, &bytes,
                                       error, sizeof(error));
        if (!dup) {
            printf("    codebook host read: FAIL (%s)\n", error);
            host_ok = 0;
        } else {
            printf("    codebook host read: OK (%zu bytes)\n", bytes);
            free(dup);
        }
    }
    printf("w4a8_summary: found %d/4 quads; host_read=%s\n", found,
           host_ok ? "ok" : "fail");
    h3_w4a8_store_free(store);
    return found == 4 && host_ok;
}

int main(int argc, char **argv) {
    const char *path = NULL;
    int do_w4a8 = 0;
    unsigned w4a8_block = 0;
    int list_limit = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--w4a8-block") && i + 1 < argc) {
            do_w4a8 = 1;
            w4a8_block = (unsigned)strtoul(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--list") && i + 1 < argc) {
            list_limit = (int)strtol(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            fprintf(stderr,
                    "usage: %s [--w4a8-block N] [--list K] FILE.safetensors\n",
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
        fprintf(stderr,
                "usage: %s [--w4a8-block N] [--list K] FILE.safetensors\n",
                argv[0]);
        return 2;
    }

    char error[512];
    h3_st_header header;
    if (!h3_st_read_header(path, &header, error, sizeof(error))) {
        fprintf(stderr, "header read failed: %s\n", error);
        return 1;
    }

    h3_st_quant_info quant;
    h3_st_quant_info_from_header(&header, &quant);

    size_t dtype_counts[32];
    memset(dtype_counts, 0, sizeof(dtype_counts));
    h3_count_slot affiliates[] = {
        {"weight_codebook", 0},
        {"weight_s_rel", 0},
        {"weight_s_channel", 0},
        {"comfy_quant", 0},
        {"qweight", 0},
        {"weight_scale", 0},
        {"weight_scale_2", 0},
        {"pre_quant_scale", 0},
    };
    size_t i8_weight = 0;
    size_t u8_weight = 0;
    size_t blocks_prefix = 0;
    size_t model_prefix = 0;
    size_t visual_prefix = 0;
    size_t diffusion_prefix = 0;

    for (size_t i = 0; i < header.tensor_count; i++) {
        const h3_st_tensor *t = &header.tensors[i];
        bump_dtype(dtype_counts, sizeof(dtype_counts) / sizeof(dtype_counts[0]),
                   t->dtype);
        const char *name = t->name;
        for (size_t a = 0; a < sizeof(affiliates) / sizeof(affiliates[0]); a++) {
            if (name_has(name, affiliates[a].needle)) affiliates[a].count++;
        }
        if (name_ends_weight(name)) {
            if (t->dtype == H3_DTYPE_I8) i8_weight++;
            if (t->dtype == H3_DTYPE_U8) u8_weight++;
        }
        if (name && !strncmp(name, "blocks.", 7)) blocks_prefix++;
        if (name && !strncmp(name, "model.", 6)) model_prefix++;
        if (name && !strncmp(name, "visual.", 7)) visual_prefix++;
        if (name && !strncmp(name, "diffusion_model.", 16)) diffusion_prefix++;
    }

    const char *kind = quant.format[0] ? quant.format : "unknown";
    if ((!quant.format[0] || !strcmp(quant.format, "unknown")) &&
        affiliates[0].count && affiliates[1].count && affiliates[2].count) {
        kind = "asym_w4a8_int8";
    } else if (u8_weight && dtype_counts[H3_DTYPE_F8_E4M3] &&
               affiliates[5].count) {
        kind = "nvfp4_awq_like";
    } else if (affiliates[5].count && i8_weight) {
        kind = "int8_scale_convrot_like";
    }

    printf("path: %s\n", path);
    printf("file_size: %llu\n", (unsigned long long)header.file_size);
    printf("header_size: %llu\n", (unsigned long long)header.header_size);
    printf("tensors: %zu\n", header.tensor_count);
    printf("has_metadata: %s\n", quant.has_metadata ? "yes" : "no");
    printf("format: %s\n", quant.format[0] ? quant.format : "(none)");
    printf("kind: %s\n", kind);
    printf("group_size: %d\n", quant.group_size);
    printf("convrot: %d\n", quant.convrot);
    printf("convrot_groupsize: %d\n", quant.convrot_groupsize);

    printf("-- dtypes --\n");
    for (size_t d = 0; d < sizeof(dtype_counts) / sizeof(dtype_counts[0]); d++) {
        if (!dtype_counts[d]) continue;
        printf("  %s: %zu\n", h3_dtype_name((h3_dtype)d), dtype_counts[d]);
    }

    printf("-- prefixes --\n");
    printf("  blocks.: %zu\n", blocks_prefix);
    printf("  model.: %zu\n", model_prefix);
    printf("  visual.: %zu\n", visual_prefix);
    printf("  diffusion_model.: %zu\n", diffusion_prefix);

    printf("-- affiliate_counts --\n");
    for (size_t a = 0; a < sizeof(affiliates) / sizeof(affiliates[0]); a++) {
        printf("  %s: %zu\n", affiliates[a].needle, affiliates[a].count);
    }
    printf("  i8_weight: %zu\n", i8_weight);
    printf("  u8_weight: %zu\n", u8_weight);

    /* Sample first I8 packed linear with W4A8 affiliates if present. */
    printf("-- sample_packed_linear --\n");
    int sample_printed = 0;
    for (size_t i = 0; i < header.tensor_count && !sample_printed; i++) {
        const h3_st_tensor *t = &header.tensors[i];
        if (!name_ends_weight(t->name) || t->dtype != H3_DTYPE_I8) continue;
        size_t n = strlen(t->name);
        char base[256];
        if (n < 8 || n - 7 >= sizeof(base)) continue;
        memcpy(base, t->name, n - 7);
        base[n - 7] = '\0';
        char key[288];
        snprintf(key, sizeof(key), "%s.weight_codebook", base);
        if (!h3_st_find(&header, key)) continue;
        print_tensor_line(t);
        snprintf(key, sizeof(key), "%s.weight_codebook", base);
        print_tensor_line(h3_st_find(&header, key));
        snprintf(key, sizeof(key), "%s.weight_s_channel", base);
        if (h3_st_find(&header, key)) print_tensor_line(h3_st_find(&header, key));
        snprintf(key, sizeof(key), "%s.weight_s_rel", base);
        if (h3_st_find(&header, key)) print_tensor_line(h3_st_find(&header, key));
        sample_printed = 1;
    }
    if (!sample_printed) {
        for (size_t i = 0; i < header.tensor_count && !sample_printed; i++) {
            const h3_st_tensor *t = &header.tensors[i];
            if (!name_ends_weight(t->name) || t->dtype != H3_DTYPE_U8) continue;
            print_tensor_line(t);
            sample_printed = 1;
        }
    }
    if (!sample_printed) printf("  (none)\n");

    if (list_limit > 0) {
        printf("-- tensors (first %d) --\n", list_limit);
        size_t limit = (size_t)list_limit;
        if (limit > header.tensor_count) limit = header.tensor_count;
        for (size_t i = 0; i < limit; i++) print_tensor_line(&header.tensors[i]);
    }

    int w4a8_ok = 1;
    if (do_w4a8) {
        w4a8_ok = probe_w4a8_block(path, w4a8_block);
    }

    h3_st_free_header(&header);
    return (do_w4a8 && !w4a8_ok) ? 1 : 0;
}
