#include "h3_dit_int8.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


struct h3_dit_int8_store {
    h3_st_header header;
    h3_st_quant_info quant;
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static void scan_comfy_quant_blob(h3_dit_int8_store *store) {
    size_t ti;
    if (!store) return;
    for (ti = 0; ti < store->header.tensor_count; ti++) {
        const h3_st_tensor *t = &store->header.tensors[ti];
        char blob[256];
        size_t nbytes;
        const char *p;
        if (!t->name || !strstr(t->name, "comfy_quant")) continue;
        nbytes = (size_t)h3_st_tensor_nbytes(t);
        if (nbytes == 0 || nbytes + 1 > sizeof(blob)) continue;
        if (!h3_st_read_data(&store->header, t, blob, nbytes, NULL, 0)) continue;
        blob[nbytes] = '\0';
        p = strstr(blob, "\"convrot\"");
        if (p && store->quant.convrot < 0) {
            if (strstr(p, "true")) store->quant.convrot = 1;
            else if (strstr(p, "false")) store->quant.convrot = 0;
        }
        p = strstr(blob, "\"convrot_groupsize\"");
        if (p && store->quant.convrot_groupsize < 0) {
            p = strchr(p, ':');
            if (p)
                store->quant.convrot_groupsize =
                    (int)strtol(p + 1, NULL, 10);
        }
        if (store->quant.convrot >= 0 && store->quant.convrot_groupsize > 0)
            break;
    }
    if (!store->quant.format[0]) {
        snprintf(store->quant.format, sizeof(store->quant.format),
                 "int8_scale_convrot_like");
    }
}

h3_dit_int8_store *h3_dit_int8_store_open(const char *path, char *error,
                                          size_t error_size) {
    h3_dit_int8_store *store;
    char detail[384];
    if (!path || !*path) {
        fail(error, error_size, "INT8 DiT weight path is required");
        return NULL;
    }
    store = calloc(1, sizeof(*store));
    if (!store) {
        fail(error, error_size, "out of memory opening INT8 DiT store");
        return NULL;
    }
    if (!h3_st_read_header(path, &store->header, detail, sizeof(detail))) {
        fail(error, error_size, "%s", detail);
        free(store);
        return NULL;
    }
    h3_st_quant_info_from_header(&store->header, &store->quant);
    scan_comfy_quant_blob(store);
    if (!h3_dit_int8_store_is_int8_convrot(store)) {
        fail(error, error_size,
             "not an INT8 ConvRot DiT (need I8+weight_scale; reject W4A8 "
             "codebook). format=%s",
             store->quant.format[0] ? store->quant.format : "?");
        h3_st_free_header(&store->header);
        free(store);
        return NULL;
    }
    return store;
}

void h3_dit_int8_store_free(h3_dit_int8_store *store) {
    if (!store) return;
    h3_st_free_header(&store->header);
    free(store);
}

const h3_st_header *h3_dit_int8_store_header(const h3_dit_int8_store *store) {
    return store ? &store->header : NULL;
}

const h3_st_quant_info *h3_dit_int8_store_quant(const h3_dit_int8_store *store) {
    return store ? &store->quant : NULL;
}

int h3_dit_int8_store_is_int8_convrot(const h3_dit_int8_store *store) {
    int has_scale = 0;
    int has_codebook = 0;
    int has_i8 = 0;
    size_t i;
    if (!store) return 0;
    for (i = 0; i < store->header.tensor_count; i++) {
        const h3_st_tensor *t = &store->header.tensors[i];
        if (!t->name) continue;
        if (strstr(t->name, "weight_codebook")) has_codebook = 1;
        if (strstr(t->name, "weight_scale")) has_scale = 1;
        if (t->dtype == H3_DTYPE_I8 && strstr(t->name, ".weight") &&
            !strstr(t->name, "weight_"))
            has_i8 = 1;
    }
    return has_i8 && has_scale && !has_codebook;
}

int h3_dit_int8_find_linear(const h3_dit_int8_store *store, const char *base,
                            h3_dit_int8_linear *out, char *error,
                            size_t error_size) {
    char wname[288], sname[288], qname[288];
    const h3_st_tensor *w, *s, *q;
    if (!store || !base || !out) {
        fail(error, error_size, "invalid INT8 DiT find_linear args");
        return 0;
    }
    memset(out, 0, sizeof(*out));
    snprintf(out->base_name, sizeof(out->base_name), "%s", base);
    snprintf(wname, sizeof(wname), "%s.weight", base);
    snprintf(sname, sizeof(sname), "%s.weight_scale", base);
    snprintf(qname, sizeof(qname), "%s.comfy_quant", base);
    w = h3_st_find(&store->header, wname);
    s = h3_st_find(&store->header, sname);
    q = h3_st_find(&store->header, qname);
    if (!w || !s) {
        fail(error, error_size, "missing %s.weight or .weight_scale", base);
        return 0;
    }
    if (w->dtype != H3_DTYPE_I8 || s->dtype != H3_DTYPE_F32 || w->ndim != 2) {
        fail(error, error_size, "%s unexpected weight/scale dtype or rank",
             base);
        return 0;
    }
    out->weight = w;
    out->scale = s;
    out->comfy_quant = q;
    out->n = (size_t)w->shape[0];
    out->k = (size_t)w->shape[1];
    if (out->n == 0 || out->k == 0) {
        fail(error, error_size, "%s empty linear shape", base);
        return 0;
    }
    if (h3_st_tensor_elements(s) != out->n &&
        !(s->ndim == 2 && (size_t)s->shape[0] == out->n &&
          (size_t)s->shape[1] == 1)) {
        fail(error, error_size, "%s weight_scale shape mismatch", base);
        return 0;
    }
    return 1;
}

int h3_dit_int8_find_block_linear(const h3_dit_int8_store *store,
                                  unsigned block_index, const char *suffix,
                                  h3_dit_int8_linear *out, char *error,
                                  size_t error_size) {
    char base[192];
    if (!suffix) {
        fail(error, error_size, "INT8 DiT suffix required");
        return 0;
    }
    snprintf(base, sizeof(base), "blocks.%u.%s", block_index, suffix);
    return h3_dit_int8_find_linear(store, base, out, error, error_size);
}

const h3_st_tensor *h3_dit_int8_find(const h3_dit_int8_store *store,
                                     const char *name) {
    return store ? h3_st_find(&store->header, name) : NULL;
}

int h3_dit_int8_read_tensor(const h3_dit_int8_store *store,
                            const h3_st_tensor *tensor, void *dst, size_t bytes,
                            char *error, size_t error_size) {
    if (!store || !tensor) {
        fail(error, error_size, "invalid INT8 DiT tensor read");
        return 0;
    }
    return h3_st_read_data(&store->header, tensor, dst, bytes, error,
                           error_size);
}

int h3_dit_int8_convrot_groupsize(const h3_dit_int8_store *store, size_t k) {
    const char *e;
    int gs;
    e = getenv("H3_DIT_INT8_NO_CONVROT");
    if (e && e[0] == '1') return 0;
    gs = store && store->quant.convrot_groupsize > 0
             ? store->quant.convrot_groupsize
             : H3_DIT_INT8_DEFAULT_CONVROT_GS;
    if (gs < 4 || k % (size_t)gs != 0) return 0;
    if (store && store->quant.convrot == 0) return 0;
    return gs;
}

float *h3_dit_int8_dequant_f32(const h3_dit_int8_store *store,
                               const h3_dit_int8_linear *lin, char *error,
                               size_t error_size) {
    int8_t *w_i8 = NULL;
    float *scales = NULL;
    float *out = NULL;
    size_t i, j;
    if (!store || !lin || !lin->weight || !lin->scale) {
        fail(error, error_size, "invalid INT8 DiT dequant args");
        return NULL;
    }
    w_i8 = (int8_t *)malloc(lin->n * lin->k);
    scales = (float *)malloc(lin->n * sizeof(float));
    out = (float *)malloc(lin->n * lin->k * sizeof(float));
    if (!w_i8 || !scales || !out) {
        free(w_i8);
        free(scales);
        free(out);
        fail(error, error_size, "OOM INT8 DiT dequant");
        return NULL;
    }
    if (!h3_st_read_data(&store->header, lin->weight, w_i8, lin->n * lin->k,
                         error, error_size) ||
        !h3_st_read_data(&store->header, lin->scale, scales,
                         lin->n * sizeof(float), error, error_size)) {
        free(w_i8);
        free(scales);
        free(out);
        return NULL;
    }
    for (i = 0; i < lin->n; i++) {
        float s = scales[i];
        const int8_t *row = w_i8 + i * lin->k;
        float *dst = out + i * lin->k;
        for (j = 0; j < lin->k; j++) dst[j] = (float)row[j] * s;
    }
    free(w_i8);
    free(scales);
    return out;
}
