#include "h3_w4a8.h"

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct h3_w4a8_store {
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

h3_w4a8_store *h3_w4a8_store_open(const char *path, char *error,
                                  size_t error_size) {
    if (!path || !*path) {
        fail(error, error_size, "W4A8 weight path is required");
        return NULL;
    }
    h3_w4a8_store *store = calloc(1, sizeof(*store));
    if (!store) {
        fail(error, error_size, "out of memory opening W4A8 store");
        return NULL;
    }
    char detail[384];
    if (!h3_st_read_header(path, &store->header, detail, sizeof(detail))) {
        fail(error, error_size, "%s", detail);
        free(store);
        return NULL;
    }
    h3_st_quant_info_from_header(&store->header, &store->quant);
    return store;
}

void h3_w4a8_store_free(h3_w4a8_store *store) {
    if (!store) return;
    h3_st_free_header(&store->header);
    free(store);
}

const h3_st_header *h3_w4a8_store_header(const h3_w4a8_store *store) {
    return store ? &store->header : NULL;
}

const h3_st_quant_info *h3_w4a8_store_quant(const h3_w4a8_store *store) {
    return store ? &store->quant : NULL;
}

static int require_tensor(const h3_st_header *header, const char *name,
                          h3_dtype dtype, const h3_st_tensor **out,
                          char *error, size_t error_size) {
    const h3_st_tensor *tensor = h3_st_find(header, name);
    if (!tensor) {
        fail(error, error_size, "missing W4A8 tensor: %s", name);
        return 0;
    }
    if (tensor->dtype != dtype) {
        fail(error, error_size, "%s has dtype %s, expected %s", name,
             h3_dtype_name(tensor->dtype), h3_dtype_name(dtype));
        return 0;
    }
    *out = tensor;
    return 1;
}

int h3_w4a8_find_quad(const h3_w4a8_store *store, const char *base_name,
                      h3_w4a8_quad *out, char *error, size_t error_size) {
    if (!store || !base_name || !out) {
        fail(error, error_size, "invalid W4A8 quad lookup");
        return 0;
    }
    memset(out, 0, sizeof(*out));
    snprintf(out->base_name, sizeof(out->base_name), "%s", base_name);

    char name[224];
    const h3_st_header *header = &store->header;

    snprintf(name, sizeof(name), "%s.weight", base_name);
    if (!require_tensor(header, name, H3_DTYPE_I8, &out->weight, error,
                        error_size))
        return 0;

    snprintf(name, sizeof(name), "%s.weight_codebook", base_name);
    if (!require_tensor(header, name, H3_DTYPE_F32, &out->weight_codebook,
                        error, error_size))
        return 0;
    if (out->weight_codebook->ndim != 1 ||
        out->weight_codebook->shape[0] != 16) {
        fail(error, error_size, "%s.weight_codebook expected F32[16]",
             base_name);
        return 0;
    }

    snprintf(name, sizeof(name), "%s.weight_s_channel", base_name);
    if (!require_tensor(header, name, H3_DTYPE_F32, &out->weight_s_channel,
                        error, error_size))
        return 0;

    snprintf(name, sizeof(name), "%s.weight_s_rel", base_name);
    if (!require_tensor(header, name, H3_DTYPE_F8_E4M3, &out->weight_s_rel,
                        error, error_size))
        return 0;

    return 1;
}

int h3_w4a8_find_block_linear(const h3_w4a8_store *store, unsigned block_index,
                              const char *suffix, h3_w4a8_quad *out,
                              char *error, size_t error_size) {
    if (!suffix) {
        fail(error, error_size, "W4A8 linear suffix is required");
        return 0;
    }
    char base[192];
    snprintf(base, sizeof(base), "blocks.%u.%s", block_index, suffix);
    return h3_w4a8_find_quad(store, base, out, error, error_size);
}

int h3_w4a8_read_tensor(const h3_w4a8_store *store, const h3_st_tensor *tensor,
                        void *dst, size_t bytes, char *error,
                        size_t error_size) {
    if (!store) {
        fail(error, error_size, "W4A8 store is required");
        return 0;
    }
    return h3_st_read_data(&store->header, tensor, dst, bytes, error,
                           error_size);
}

void *h3_w4a8_dup_tensor(const h3_w4a8_store *store, const h3_st_tensor *tensor,
                         size_t *out_bytes, char *error, size_t error_size) {
    if (out_bytes) *out_bytes = 0;
    if (!store || !tensor) {
        fail(error, error_size, "invalid W4A8 tensor dup");
        return NULL;
    }
    uint64_t nbytes = h3_st_tensor_nbytes(tensor);
    if (nbytes > SIZE_MAX) {
        fail(error, error_size, "tensor too large: %s",
             tensor->name ? tensor->name : "?");
        return NULL;
    }
    void *buffer = malloc((size_t)nbytes);
    if (!buffer) {
        fail(error, error_size, "out of memory reading %s",
             tensor->name ? tensor->name : "tensor");
        return NULL;
    }
    if (!h3_w4a8_read_tensor(store, tensor, buffer, (size_t)nbytes, error,
                             error_size)) {
        free(buffer);
        return NULL;
    }
    if (out_bytes) *out_bytes = (size_t)nbytes;
    return buffer;
}

const h3_st_tensor *h3_w4a8_find(const h3_w4a8_store *store, const char *name) {
    if (!store || !name) return NULL;
    return h3_st_find(&store->header, name);
}

static float bf16_bits_to_f32(uint16_t bits) {
    union {
        uint32_t u;
        float f;
    } v;
    v.u = ((uint32_t)bits) << 16;
    return v.f;
}

static float f16_bits_to_f32(uint16_t bits) {
    const uint32_t sign = (uint32_t)(bits & 0x8000u) << 16;
    const uint32_t exp = (bits >> 10) & 0x1fu;
    const uint32_t mant = bits & 0x3ffu;
    union {
        uint32_t u;
        float f;
    } v;
    if (exp == 0) {
        if (mant == 0) {
            v.u = sign;
            return v.f;
        }
        /* subnormal */
        float m = (float)mant / 1024.0f;
        float val = ldexpf(m, -14);
        v.f = val;
        if (sign) v.f = -v.f;
        return v.f;
    }
    if (exp == 31) {
        v.u = sign | 0x7f800000u | (mant << 13);
        return v.f;
    }
    v.u = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    return v.f;
}

float *h3_w4a8_load_f32(const h3_w4a8_store *store, const char *name,
                        size_t *out_elements, h3_dtype *out_dtype, char *error,
                        size_t error_size) {
    const h3_st_tensor *tensor;
    size_t elements;
    float *out = NULL;
    size_t i;

    if (out_elements) *out_elements = 0;
    if (out_dtype) *out_dtype = H3_DTYPE_UNKNOWN;
    if (!store || !name) {
        fail(error, error_size, "invalid W4A8 load_f32 args");
        return NULL;
    }
    tensor = h3_w4a8_find(store, name);
    if (!tensor) {
        fail(error, error_size, "missing tensor: %s", name);
        return NULL;
    }
    if (tensor->dtype != H3_DTYPE_F32 && tensor->dtype != H3_DTYPE_BF16 &&
        tensor->dtype != H3_DTYPE_F16) {
        fail(error, error_size, "%s dtype %s not convertible to F32", name,
             h3_dtype_name(tensor->dtype));
        return NULL;
    }
    elements = (size_t)h3_st_tensor_elements(tensor);
    if (elements == 0) {
        fail(error, error_size, "%s is empty", name);
        return NULL;
    }
    out = (float *)malloc(elements * sizeof(float));
    if (!out) {
        fail(error, error_size, "out of memory loading %s", name);
        return NULL;
    }
    if (tensor->dtype == H3_DTYPE_F32) {
        if (!h3_w4a8_read_tensor(store, tensor, out, elements * sizeof(float),
                                 error, error_size)) {
            free(out);
            return NULL;
        }
    } else {
        uint16_t *raw = (uint16_t *)malloc(elements * sizeof(uint16_t));
        if (!raw) {
            fail(error, error_size, "out of memory reading %s", name);
            free(out);
            return NULL;
        }
        if (!h3_w4a8_read_tensor(store, tensor, raw, elements * sizeof(uint16_t),
                                 error, error_size)) {
            free(raw);
            free(out);
            return NULL;
        }
        if (tensor->dtype == H3_DTYPE_BF16) {
            for (i = 0; i < elements; i++) out[i] = bf16_bits_to_f32(raw[i]);
        } else {
            for (i = 0; i < elements; i++) out[i] = f16_bits_to_f32(raw[i]);
        }
        free(raw);
    }
    if (out_elements) *out_elements = elements;
    if (out_dtype) *out_dtype = tensor->dtype;
    return out;
}
