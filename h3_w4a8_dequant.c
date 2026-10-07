#include "h3_w4a8_dequant.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

float h3_f8e4m3fn_to_f32(uint8_t bits) {
    const int sign = (bits >> 7) & 1;
    const int exp = (bits >> 3) & 0xF;
    const int mant = bits & 0x7;

    /* float8_e4m3fn: only NaN encoding is exp=all-ones, mant=all-ones. */
    if (exp == 0xF && mant == 0x7) {
        return sign ? -NAN : NAN;
    }

    float value;
    if (exp == 0) {
        if (mant == 0) {
            value = 0.0f;
        } else {
            /* subnormal: 2^(1-bias-mbits) * mant */
            value = ldexpf((float)mant, 1 - 7 - 3);
        }
    } else {
        value = ldexpf(1.0f + (float)mant / 8.0f, exp - 7);
    }
    return sign ? -value : value;
}

int h3_w4a8_unpack_nibbles(const int8_t *packed, size_t n, size_t k_half,
                           int8_t *indices_out, char *error, size_t error_size) {
    if (!packed || !indices_out || n == 0 || k_half == 0) {
        fail(error, error_size, "invalid nibble unpack arguments");
        return 0;
    }
    const size_t k = k_half * 2;
    for (size_t row = 0; row < n; row++) {
        const int8_t *src = packed + row * k_half;
        int8_t *dst = indices_out + row * k;
        for (size_t col = 0; col < k_half; col++) {
            const unsigned byte = (unsigned)(uint8_t)src[col];
            dst[col * 2] = (int8_t)(byte & 0xFu);
            dst[col * 2 + 1] = (int8_t)((byte >> 4) & 0xFu);
        }
    }
    return 1;
}

static float round_nearest_even_away(float x) {
    /* Match torch.round (banker's rounding toward even on .5). */
    return roundf(x);
}

int h3_w4a8_dequant_to_int8(const int8_t *packed, size_t n, size_t k_half,
                            const float *codebook16, const uint8_t *s_rel_f8,
                            int group_size, int8_t *int8_out, char *error,
                            size_t error_size) {
    if (!packed || !codebook16 || !s_rel_f8 || !int8_out) {
        fail(error, error_size, "invalid dequant_to_int8 arguments");
        return 0;
    }
    if (group_size < 4 || (group_size % 16 != 0 && 16 % group_size != 0)) {
        fail(error, error_size, "unsupported group_size=%d", group_size);
        return 0;
    }
    const size_t k = k_half * 2;
    if (k % (size_t)group_size != 0) {
        fail(error, error_size, "K=%zu not divisible by group_size=%d", k,
             group_size);
        return 0;
    }
    const size_t groups = k / (size_t)group_size;

    int8_t *indices = (int8_t *)malloc(n * k);
    if (!indices) {
        fail(error, error_size, "out of memory for nibble indices");
        return 0;
    }
    if (!h3_w4a8_unpack_nibbles(packed, n, k_half, indices, error, error_size)) {
        free(indices);
        return 0;
    }

    for (size_t row = 0; row < n; row++) {
        for (size_t g = 0; g < groups; g++) {
            const float scale =
                h3_f8e4m3fn_to_f32(s_rel_f8[row * groups + g]);
            for (int e = 0; e < group_size; e++) {
                const size_t col = g * (size_t)group_size + (size_t)e;
                const int idx = (int)indices[row * k + col];
                if (idx < 0 || idx > 15) {
                    free(indices);
                    fail(error, error_size, "nibble index out of range: %d",
                         idx);
                    return 0;
                }
                float value = codebook16[idx] * scale;
                value = round_nearest_even_away(value);
                if (value > 127.0f) value = 127.0f;
                if (value < -127.0f) value = -127.0f;
                int8_out[row * k + col] = (int8_t)value;
            }
        }
    }
    free(indices);
    return 1;
}

int h3_w4a8_int8_scale_to_f32(const int8_t *int8_w, size_t n, size_t k,
                              const float *s_channel, float *f32_out,
                              char *error, size_t error_size) {
    if (!int8_w || !s_channel || !f32_out || n == 0 || k == 0) {
        fail(error, error_size, "invalid int8_scale_to_f32 arguments");
        return 0;
    }
    for (size_t row = 0; row < n; row++) {
        const float scale = s_channel[row];
        const int8_t *src = int8_w + row * k;
        float *dst = f32_out + row * k;
        for (size_t col = 0; col < k; col++) {
            dst[col] = (float)src[col] * scale;
        }
    }
    return 1;
}

/* Normalized regular Hadamard (power-of-4 size) via Kronecker of H4. */
float *h3_w4a8_hadamard_normalized(int size, char *error,
                                   size_t error_size) {
    if (size < 4 || (size & (size - 1)) != 0) {
        fail(error, error_size, "Hadamard size must be power of 2 >=4, got %d",
             size);
        return NULL;
    }
    /* Regular Hadamard: size must be 4^p. */
    int t = size;
    while (t % 4 == 0) t /= 4;
    if (t != 1) {
        fail(error, error_size, "regular Hadamard size must be 4^p, got %d",
             size);
        return NULL;
    }

    static const float h4[16] = {
        1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1,
    };

    float *h = (float *)malloc((size_t)size * (size_t)size * sizeof(float));
    float *tmp = (float *)malloc((size_t)size * (size_t)size * sizeof(float));
    if (!h || !tmp) {
        free(h);
        free(tmp);
        fail(error, error_size, "out of memory building Hadamard");
        return NULL;
    }

    int current = 4;
    memcpy(h, h4, sizeof(h4));

    while (current < size) {
        const int next = current * 4;
        /* kron(h, h4): block (i,j) = h[i,j] * h4 */
        for (int i = 0; i < current; i++) {
            for (int j = 0; j < current; j++) {
                const float scale = h[i * current + j];
                for (int bi = 0; bi < 4; bi++) {
                    for (int bj = 0; bj < 4; bj++) {
                        tmp[(i * 4 + bi) * next + (j * 4 + bj)] =
                            scale * h4[bi * 4 + bj];
                    }
                }
            }
        }
        memcpy(h, tmp, (size_t)next * (size_t)next * sizeof(float));
        current = next;
    }

    const float inv_sqrt = 1.0f / sqrtf((float)size);
    const size_t elems = (size_t)size * (size_t)size;
    for (size_t i = 0; i < elems; i++) h[i] *= inv_sqrt;

    free(tmp);
    return h;
}

float *h3_w4a8_hadamard_sylvester_normalized(int size, char *error,
                                             size_t error_size) {
    float *h = NULL;
    float *tmp = NULL;
    int current;
    float inv_sqrt;
    size_t elems, i;
    if (size < 2 || (size & (size - 1)) != 0) {
        fail(error, error_size,
             "Sylvester Hadamard size must be power of 2 >=2, got %d", size);
        return NULL;
    }
    h = (float *)malloc((size_t)size * (size_t)size * sizeof(float));
    tmp = (float *)malloc((size_t)size * (size_t)size * sizeof(float));
    if (!h || !tmp) {
        free(h);
        free(tmp);
        fail(error, error_size, "out of memory building Sylvester Hadamard");
        return NULL;
    }
    h[0] = 1.0f;
    current = 1;
    while (current < size) {
        int next = current * 2;
        int r, c;
        for (r = 0; r < current; r++) {
            for (c = 0; c < current; c++) {
                float v = h[r * current + c];
                tmp[r * next + c] = v;
                tmp[r * next + (c + current)] = v;
                tmp[(r + current) * next + c] = v;
                tmp[(r + current) * next + (c + current)] = -v;
            }
        }
        memcpy(h, tmp, (size_t)next * (size_t)next * sizeof(float));
        current = next;
    }
    inv_sqrt = 1.0f / sqrtf((float)size);
    elems = (size_t)size * (size_t)size;
    for (i = 0; i < elems; i++) h[i] *= inv_sqrt;
    free(tmp);
    return h;
}

static void fwht_normalized_inplace(float *a, int n) {
    int h;
    float scale;
    for (h = 1; h < n; h *= 2) {
        int i;
        for (i = 0; i < n; i += h * 2) {
            int j;
            for (j = i; j < i + h; j++) {
                float x = a[j];
                float y = a[j + h];
                a[j] = x + y;
                a[j + h] = x - y;
            }
        }
    }
    scale = 1.0f / sqrtf((float)n);
    for (h = 0; h < n; h++) a[h] *= scale;
}

int h3_w4a8_fwht_act_f32(const float *act_in, size_t m, size_t k,
                         int convrot_groupsize, float *act_out, char *error,
                         size_t error_size) {
    size_t gsize, n_groups, row, g;
    if (!act_in || !act_out || m == 0 || k == 0) {
        fail(error, error_size, "invalid fwht_act arguments");
        return 0;
    }
    if (convrot_groupsize < 2 || (convrot_groupsize & (convrot_groupsize - 1)) ||
        k % (size_t)convrot_groupsize != 0) {
        fail(error, error_size, "invalid fwht groupsize=%d K=%zu",
             convrot_groupsize, k);
        return 0;
    }
    gsize = (size_t)convrot_groupsize;
    n_groups = k / gsize;
    if (act_out != act_in)
        memcpy(act_out, act_in, m * k * sizeof(float));
    for (row = 0; row < m; row++) {
        float *rowp = act_out + row * k;
        for (g = 0; g < n_groups; g++)
            fwht_normalized_inplace(rowp + g * gsize, (int)gsize);
    }
    return 1;
}

int h3_w4a8_convrot_weight_f32(const float *weight_in, size_t n, size_t k,
                               int convrot_groupsize, float *weight_out,
                               char *error, size_t error_size) {
    static float *cached_h = NULL;
    static int cached_gs = 0;
    float *h;
    size_t gsize, n_groups, g, row, i, j;
    float *pack = NULL;
    float *out_pack = NULL;

    if (!weight_in || !weight_out || n == 0 || k == 0) {
        fail(error, error_size, "invalid convrot_weight arguments");
        return 0;
    }
    /* gs<=0: identity (NVFP4 requant → INT8 GEMM; no ConvRot). */
    if (convrot_groupsize <= 0) {
        if (weight_in != weight_out)
            memcpy(weight_out, weight_in, n * k * sizeof(float));
        return 1;
    }
    if (convrot_groupsize < 4 || k % (size_t)convrot_groupsize != 0) {
        fail(error, error_size, "K=%zu not divisible by convrot_groupsize=%d",
             k, convrot_groupsize);
        return 0;
    }

    gsize = (size_t)convrot_groupsize;
    n_groups = k / gsize;
    if (cached_h && cached_gs == convrot_groupsize) {
        h = cached_h;
    } else {
        free(cached_h);
        cached_h = h3_w4a8_hadamard_normalized(convrot_groupsize, error,
                                               error_size);
        if (!cached_h) {
            cached_gs = 0;
            return 0;
        }
        cached_gs = convrot_groupsize;
        h = cached_h;
    }

    /* Pack each K-group to contiguous [n, gs], GEMM vs H, unpack.
     * Loop order favors H broadcast + row streaming. */
    pack = (float *)malloc(n * gsize * sizeof(float));
    out_pack = (float *)malloc(n * gsize * sizeof(float));
    if (!pack || !out_pack) {
        free(pack);
        free(out_pack);
        fail(error, error_size, "out of memory for convrot pack");
        return 0;
    }

    for (g = 0; g < n_groups; g++) {
        for (row = 0; row < n; row++) {
            memcpy(pack + row * gsize,
                   weight_in + row * k + g * gsize, gsize * sizeof(float));
        }
        memset(out_pack, 0, n * gsize * sizeof(float));
        /* out = pack @ H  (row-major); ikj for cache on H columns. */
        for (i = 0; i < gsize; i++) {
            for (j = 0; j < gsize; j++) {
                float hij = h[i * gsize + j];
                if (hij == 0.0f) continue;
                for (row = 0; row < n; row++) {
                    out_pack[row * gsize + j] +=
                        pack[row * gsize + i] * hij;
                }
            }
        }
        for (row = 0; row < n; row++) {
            memcpy(weight_out + row * k + g * gsize,
                   out_pack + row * gsize, gsize * sizeof(float));
        }
    }

    free(pack);
    free(out_pack);
    return 1;
}

int h3_w4a8_convrot_act_f32(const float *act_in, size_t m, size_t k,
                            int convrot_groupsize, float *act_out, char *error,
                            size_t error_size) {
    /* Same K-group Hadamard as weight undo; rows are batch tokens.
     * Optional Sylvester/FWHT via H3_CONVROT_SYLVESTER=1. */
    if (getenv("H3_CONVROT_SYLVESTER") &&
        getenv("H3_CONVROT_SYLVESTER")[0] == '1')
        return h3_w4a8_fwht_act_f32(act_in, m, k, convrot_groupsize, act_out,
                                    error, error_size);
    return h3_w4a8_convrot_weight_f32(act_in, m, k, convrot_groupsize, act_out,
                                      error, error_size);
}

int h3_w4a8_dequant_to_f32(const int8_t *packed, size_t n, size_t k_half,
                           const float *codebook16, const uint8_t *s_rel_f8,
                           const float *s_channel, int group_size,
                           int convrot_groupsize, int apply_convrot,
                           float *f32_out, char *error, size_t error_size) {
    if (!f32_out) {
        fail(error, error_size, "f32_out is required");
        return 0;
    }
    const size_t k = k_half * 2;
    int8_t *int8_w = (int8_t *)malloc(n * k);
    if (!int8_w) {
        fail(error, error_size, "out of memory for INT8 dequant");
        return 0;
    }
    if (!h3_w4a8_dequant_to_int8(packed, n, k_half, codebook16, s_rel_f8,
                                 group_size, int8_w, error, error_size)) {
        free(int8_w);
        return 0;
    }
    if (!h3_w4a8_int8_scale_to_f32(int8_w, n, k, s_channel, f32_out, error,
                                   error_size)) {
        free(int8_w);
        return 0;
    }
    free(int8_w);

    if (apply_convrot) {
        float *tmp = (float *)malloc(n * k * sizeof(float));
        if (!tmp) {
            fail(error, error_size, "out of memory for ConvRot undo");
            return 0;
        }
        memcpy(tmp, f32_out, n * k * sizeof(float));
        if (!h3_w4a8_convrot_weight_f32(tmp, n, k, convrot_groupsize, f32_out,
                                        error, error_size)) {
            free(tmp);
            return 0;
        }
        free(tmp);
    }
    return 1;
}

int h3_w4a8_load_linear_f32(const h3_w4a8_store *store, const h3_w4a8_quad *quad,
                            int group_size, int convrot_groupsize,
                            int apply_convrot, float **out_weight, size_t *out_n,
                            size_t *out_k, char *error, size_t error_size) {
    if (out_weight) *out_weight = NULL;
    if (out_n) *out_n = 0;
    if (out_k) *out_k = 0;
    if (!store || !quad || !out_weight) {
        fail(error, error_size, "invalid load_linear_f32 arguments");
        return 0;
    }
    if (quad->weight->ndim != 2 || quad->weight_s_rel->ndim != 2 ||
        quad->weight_s_channel->ndim != 1) {
        fail(error, error_size, "%s: unexpected tensor ranks",
             quad->base_name);
        return 0;
    }

    const size_t n = (size_t)quad->weight->shape[0];
    const size_t k_half = (size_t)quad->weight->shape[1];
    const size_t k = k_half * 2;
    if ((size_t)quad->weight_s_channel->shape[0] != n) {
        fail(error, error_size, "%s: s_channel length mismatch",
             quad->base_name);
        return 0;
    }
    if ((size_t)quad->weight_s_rel->shape[0] != n ||
        (size_t)quad->weight_s_rel->shape[1] * (size_t)group_size != k) {
        fail(error, error_size, "%s: s_rel shape mismatch for group_size=%d",
             quad->base_name, group_size);
        return 0;
    }

    size_t nbytes = 0;
    int8_t *packed =
        (int8_t *)h3_w4a8_dup_tensor(store, quad->weight, &nbytes, error,
                                     error_size);
    if (!packed) return 0;
    float *codebook =
        (float *)h3_w4a8_dup_tensor(store, quad->weight_codebook, &nbytes, error,
                                    error_size);
    if (!codebook) {
        free(packed);
        return 0;
    }
    float *s_channel =
        (float *)h3_w4a8_dup_tensor(store, quad->weight_s_channel, &nbytes,
                                    error, error_size);
    if (!s_channel) {
        free(packed);
        free(codebook);
        return 0;
    }
    uint8_t *s_rel =
        (uint8_t *)h3_w4a8_dup_tensor(store, quad->weight_s_rel, &nbytes, error,
                                      error_size);
    if (!s_rel) {
        free(packed);
        free(codebook);
        free(s_channel);
        return 0;
    }

    float *weight = (float *)malloc(n * k * sizeof(float));
    if (!weight) {
        free(packed);
        free(codebook);
        free(s_channel);
        free(s_rel);
        fail(error, error_size, "out of memory for dequant weight");
        return 0;
    }

    int ok = h3_w4a8_dequant_to_f32(packed, n, k_half, codebook, s_rel,
                                    s_channel, group_size, convrot_groupsize,
                                    apply_convrot, weight, error, error_size);
    free(packed);
    free(codebook);
    free(s_channel);
    free(s_rel);
    if (!ok) {
        free(weight);
        return 0;
    }

    *out_weight = weight;
    if (out_n) *out_n = n;
    if (out_k) *out_k = k;
    return 1;
}

void h3_w4a8_gemm_f32(const float *x, const float *weight, size_t m, size_t n,
                      size_t k, float *y) {
    /* y[m,n] = x[m,k] @ W[n,k]^T */
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float acc = 0.0f;
            const float *xi = x + i * k;
            const float *wj = weight + j * k;
            for (size_t t = 0; t < k; t++) {
                acc += xi[t] * wj[t];
            }
            y[i * n + j] = acc;
        }
    }
}

int h3_w4a8_compare_f32(const float *a, const float *b, size_t count,
                        float *max_abs_diff_out, double *rmse_out) {
    if (!a || !b || count == 0) return 0;
    float max_abs = 0.0f;
    double sum_sq = 0.0;
    for (size_t i = 0; i < count; i++) {
        const float d = fabsf(a[i] - b[i]);
        if (d > max_abs) max_abs = d;
        sum_sq += (double)d * (double)d;
    }
    if (max_abs_diff_out) *max_abs_diff_out = max_abs;
    if (rmse_out) *rmse_out = sqrt(sum_sq / (double)count);
    return 1;
}

int h3_w4a8_load_linear_int8(const h3_w4a8_store *store,
                             const h3_w4a8_quad *quad, int group_size,
                             int8_t **out_int8, float **out_s_channel,
                             size_t *out_n, size_t *out_k, char *error,
                             size_t error_size) {
    if (out_int8) *out_int8 = NULL;
    if (out_s_channel) *out_s_channel = NULL;
    if (out_n) *out_n = 0;
    if (out_k) *out_k = 0;
    if (!store || !quad || !out_int8 || !out_s_channel) {
        fail(error, error_size, "invalid load_linear_int8 arguments");
        return 0;
    }
    if (quad->weight->ndim != 2 || quad->weight_s_rel->ndim != 2 ||
        quad->weight_s_channel->ndim != 1) {
        fail(error, error_size, "%s: unexpected tensor ranks",
             quad->base_name);
        return 0;
    }

    const size_t n = (size_t)quad->weight->shape[0];
    const size_t k_half = (size_t)quad->weight->shape[1];
    const size_t k = k_half * 2;
    if ((size_t)quad->weight_s_channel->shape[0] != n) {
        fail(error, error_size, "%s: s_channel length mismatch",
             quad->base_name);
        return 0;
    }
    if ((size_t)quad->weight_s_rel->shape[0] != n ||
        (size_t)quad->weight_s_rel->shape[1] * (size_t)group_size != k) {
        fail(error, error_size, "%s: s_rel shape mismatch for group_size=%d",
             quad->base_name, group_size);
        return 0;
    }

    size_t nbytes = 0;
    int8_t *packed =
        (int8_t *)h3_w4a8_dup_tensor(store, quad->weight, &nbytes, error,
                                     error_size);
    if (!packed) return 0;
    float *codebook =
        (float *)h3_w4a8_dup_tensor(store, quad->weight_codebook, &nbytes,
                                    error, error_size);
    if (!codebook) {
        free(packed);
        return 0;
    }
    float *s_channel =
        (float *)h3_w4a8_dup_tensor(store, quad->weight_s_channel, &nbytes,
                                    error, error_size);
    if (!s_channel) {
        free(packed);
        free(codebook);
        return 0;
    }
    uint8_t *s_rel =
        (uint8_t *)h3_w4a8_dup_tensor(store, quad->weight_s_rel, &nbytes, error,
                                      error_size);
    if (!s_rel) {
        free(packed);
        free(codebook);
        free(s_channel);
        return 0;
    }

    int8_t *int8_w = (int8_t *)malloc(n * k);
    if (!int8_w) {
        free(packed);
        free(codebook);
        free(s_channel);
        free(s_rel);
        fail(error, error_size, "out of memory for INT8 weight");
        return 0;
    }

    int ok = h3_w4a8_dequant_to_int8(packed, n, k_half, codebook, s_rel,
                                     group_size, int8_w, error, error_size);
    free(packed);
    free(codebook);
    free(s_rel);
    if (!ok) {
        free(s_channel);
        free(int8_w);
        return 0;
    }

    *out_int8 = int8_w;
    *out_s_channel = s_channel;
    if (out_n) *out_n = n;
    if (out_k) *out_k = k;
    return 1;
}

int h3_w4a8_quantize_act_int8(const float *act_f32, size_t m, size_t k,
                              h3_w4a8_act_quant_mode mode, int8_t *act_i8,
                              float *scales_out, char *error,
                              size_t error_size) {
    if (!act_f32 || !act_i8 || !scales_out || m == 0 || k == 0) {
        fail(error, error_size, "invalid quantize_act_int8 arguments");
        return 0;
    }

    if (mode == H3_W4A8_ACT_QUANT_PER_TENSOR) {
        float absmax = 0.0f;
        for (size_t i = 0; i < m * k; i++) {
            const float a = fabsf(act_f32[i]);
            if (a > absmax) absmax = a;
        }
        const float scale = (absmax > 0.0f) ? (absmax / 127.0f) : 1.0f;
        scales_out[0] = scale;
        const float inv = 1.0f / scale;
        for (size_t i = 0; i < m * k; i++) {
            float q = roundf(act_f32[i] * inv);
            if (q > 127.0f) q = 127.0f;
            if (q < -127.0f) q = -127.0f;
            act_i8[i] = (int8_t)q;
        }
        return 1;
    }

    if (mode != H3_W4A8_ACT_QUANT_PER_ROW) {
        fail(error, error_size, "unknown act quant mode %d", (int)mode);
        return 0;
    }

    for (size_t row = 0; row < m; row++) {
        const float *src = act_f32 + row * k;
        int8_t *dst = act_i8 + row * k;
        float absmax = 0.0f;
        for (size_t col = 0; col < k; col++) {
            const float a = fabsf(src[col]);
            if (a > absmax) absmax = a;
        }
        const float scale = (absmax > 0.0f) ? (absmax / 127.0f) : 1.0f;
        scales_out[row] = scale;
        const float inv = 1.0f / scale;
        for (size_t col = 0; col < k; col++) {
            float q = roundf(src[col] * inv);
            if (q > 127.0f) q = 127.0f;
            if (q < -127.0f) q = -127.0f;
            dst[col] = (int8_t)q;
        }
    }
    return 1;
}

void h3_w4a8_gemm_i8_dequant_f32(const int8_t *x_i8, const int8_t *w_i8,
                                 const float *act_scales, int per_tensor_scale,
                                 const float *s_channel, size_t m, size_t n,
                                 size_t k, float *y_f32) {
    /* y[m,n] = a_scale * s_channel[n] * sum_k x_i8[m,k] * w_i8[n,k] */
    for (size_t i = 0; i < m; i++) {
        const float a_scale =
            per_tensor_scale ? act_scales[0] : act_scales[i];
        const int8_t *xi = x_i8 + i * k;
        for (size_t j = 0; j < n; j++) {
            const int8_t *wj = w_i8 + j * k;
            int32_t acc = 0;
            for (size_t t = 0; t < k; t++) {
                acc += (int32_t)xi[t] * (int32_t)wj[t];
            }
            y_f32[i * n + j] = a_scale * s_channel[j] * (float)acc;
        }
    }
}
