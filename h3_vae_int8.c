#include "h3_vae_int8.h"
#include "h3_w4a8_dequant.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static void fail(char *error, size_t error_size, const char *msg) {
    if (!error || !error_size) return;
    snprintf(error, error_size, "%s", msg ? msg : "h3_vae_int8 error");
}

static double elapsed_ms(clock_t t0, clock_t t1) {
    return 1000.0 * (double)(t1 - t0) / (double)CLOCKS_PER_SEC;
}

int h3_vae_int8_store_copy(const h3_vae_int8_store *store, size_t file_offset,
                           void *dst, size_t nbytes, char *error,
                           size_t error_size) {
    if (!store || (!dst && nbytes != 0)) {
        fail(error, error_size, "invalid vae store_copy args");
        return 0;
    }
    if (nbytes == 0) return 1;
    if (store->map && store->map_len > 0) {
        if (file_offset + nbytes > store->map_len) {
            fail(error, error_size, "vae store_copy past mmap end");
            return 0;
        }
        memcpy(dst, store->map + file_offset, nbytes);
        return 1;
    }
    {
        int fd = open(store->path, O_RDONLY);
        size_t done = 0;
        if (fd < 0) {
            fail(error, error_size, "vae store_copy open failed");
            return 0;
        }
        while (done < nbytes) {
            ssize_t n = pread(fd, (unsigned char *)dst + done, nbytes - done,
                              (off_t)(file_offset + done));
            if (n <= 0) {
                close(fd);
                fail(error, error_size, "vae store_copy pread failed");
                return 0;
            }
            done += (size_t)n;
        }
        close(fd);
    }
    return 1;
}

static float *load_f32_named(const h3_vae_int8_store *store, const char *name,
                             size_t want, char *error, size_t error_size) {
    const h3_st_tensor *t;
    float *buf;
    size_t i, n;
    uint16_t *u16 = NULL;
    if (!store || !name) {
        fail(error, error_size, "invalid load_f32_named");
        return NULL;
    }
    t = h3_st_find(&store->header, name);
    if (!t) {
        fail(error, error_size, "missing tensor");
        return NULL;
    }
    n = (size_t)h3_st_tensor_elements(t);
    if (want && n != want) {
        fail(error, error_size, "tensor element count mismatch");
        return NULL;
    }
    buf = (float *)malloc(n * sizeof(float));
    if (!buf) {
        fail(error, error_size, "OOM load_f32");
        return NULL;
    }
    if (t->dtype == H3_DTYPE_F32) {
        if (!h3_vae_int8_store_copy(store, (size_t)t->file_offset, buf,
                                    n * sizeof(float), error, error_size)) {
            free(buf);
            return NULL;
        }
        return buf;
    }
    if (t->dtype == H3_DTYPE_BF16 || t->dtype == H3_DTYPE_F16) {
        u16 = (uint16_t *)malloc(n * sizeof(uint16_t));
        if (!u16) {
            free(buf);
            fail(error, error_size, "OOM load half");
            return NULL;
        }
        if (!h3_vae_int8_store_copy(store, (size_t)t->file_offset, u16,
                                    n * sizeof(uint16_t), error,
                                    error_size)) {
            free(u16);
            free(buf);
            return NULL;
        }
        for (i = 0; i < n; i++) {
            if (t->dtype == H3_DTYPE_BF16) {
                uint32_t bits = ((uint32_t)u16[i]) << 16;
                memcpy(&buf[i], &bits, sizeof(float));
            } else {
                /* F16 → F32 */
                uint16_t h = u16[i];
                uint32_t sign = (uint32_t)(h >> 15) << 31;
                uint32_t exp = (h >> 10) & 0x1F;
                uint32_t mant = h & 0x3FF;
                uint32_t bits;
                if (exp == 0) {
                    if (mant == 0) bits = sign;
                    else {
                        exp = 127 - 15 + 1;
                        while ((mant & 0x400) == 0) {
                            mant <<= 1;
                            exp--;
                        }
                        mant &= 0x3FF;
                        bits = sign | (exp << 23) | (mant << 13);
                    }
                } else if (exp == 31) {
                    bits = sign | 0x7F800000u | (mant << 13);
                } else {
                    bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
                }
                memcpy(&buf[i], &bits, sizeof(float));
            }
        }
        free(u16);
        return buf;
    }
    free(buf);
    fail(error, error_size, "unsupported dtype for load_f32_named");
    return NULL;
}

static void rms_norm(const float *x, const float *w, size_t rows, size_t width,
                     float eps, float *out) {
    size_t r, c;
    for (r = 0; r < rows; r++) {
        const float *xr = x + r * width;
        float *yr = out + r * width;
        float sum = 0.0f, inv;
        for (c = 0; c < width; c++) sum += xr[c] * xr[c];
        inv = 1.0f / sqrtf(sum / (float)width + eps);
        for (c = 0; c < width; c++) yr[c] = xr[c] * inv * w[c];
    }
}

static void layer_norm_bias(const float *x, const float *w, const float *b,
                            size_t rows, size_t width, float eps, float *out) {
    size_t r, c;
    for (r = 0; r < rows; r++) {
        const float *xr = x + r * width;
        float *yr = out + r * width;
        float mean = 0.0f, var = 0.0f, inv;
        for (c = 0; c < width; c++) mean += xr[c];
        mean /= (float)width;
        for (c = 0; c < width; c++) {
            float d = xr[c] - mean;
            var += d * d;
        }
        inv = 1.0f / sqrtf(var / (float)width + eps);
        for (c = 0; c < width; c++)
            yr[c] = (xr[c] - mean) * inv * w[c] + b[c];
    }
}

static void silu_mul(const float *fused, size_t rows, size_t width, float *out) {
    size_t r, c;
    for (r = 0; r < rows; r++) {
        const float *gate = fused + r * width * 2;
        const float *up = gate + width;
        float *yr = out + r * width;
        for (c = 0; c < width; c++) {
            float g = gate[c];
            float s = g / (1.0f + expf(-g));
            yr[c] = s * up[c];
        }
    }
}

int h3_vae_int8_load_latent_stats(const h3_vae_int8_store *store,
                                  float mean[H3_VAE_INT8_LATENT_CH],
                                  float std_[H3_VAE_INT8_LATENT_CH],
                                  char *error, size_t error_size) {
    float *m = NULL, *s = NULL;
    int c;
    if (!store || !mean || !std_) {
        fail(error, error_size, "invalid load_latent_stats");
        return 0;
    }
    m = load_f32_named(store, "latents_mean", H3_VAE_INT8_LATENT_CH, error,
                       error_size);
    s = load_f32_named(store, "latents_std", H3_VAE_INT8_LATENT_CH, error,
                       error_size);
    if (!m || !s) {
        free(m);
        free(s);
        return 0;
    }
    for (c = 0; c < H3_VAE_INT8_LATENT_CH; c++) {
        if (!(s[c] > 0.0f) || !isfinite(m[c]) || !isfinite(s[c])) {
            fail(error, error_size, "invalid latents_mean/std");
            free(m);
            free(s);
            return 0;
        }
        mean[c] = m[c];
        std_[c] = s[c];
    }
    free(m);
    free(s);
    return 1;
}

int h3_vae_int8_prepare_rope(int t_len, int latent_h, int latent_w,
                             size_t n_patches, size_t seq, float *cos,
                             float *sin, char *error, size_t error_size) {
    size_t row = 0;
    int t, h, w;
    size_t index;
    if (!cos || !sin || t_len < 1 || latent_h < 1 || latent_w < 1 ||
        seq < n_patches ||
        n_patches != (size_t)t_len * (size_t)latent_h * (size_t)latent_w) {
        fail(error, error_size, "invalid prepare_rope args");
        return 0;
    }
    for (t = 0; t < t_len; t++)
        for (h = 0; h < latent_h; h++)
            for (w = 0; w < latent_w; w++, row++) {
                float axes[3] = {
                    2.0f * (((float)t + 0.5f) / (float)t_len) - 1.0f,
                    2.0f * (((float)h + 0.5f) / (float)latent_h) - 1.0f,
                    2.0f * (((float)w + 0.5f) / (float)latent_w) - 1.0f};
                unsigned axis, frequency;
                for (axis = 0; axis < 3; axis++)
                    for (frequency = 0; frequency < 8; frequency++) {
                        float inverse =
                            1.0f / powf(100.0f, (float)frequency * 0.125f);
                        float angle = 2.0f * 3.14159265358979323846f *
                                      axes[axis] * inverse;
                        size_t offset = row * H3_VAE_INT8_ROPE_HALF +
                                        (size_t)axis * 8 + frequency;
                        cos[offset] = cosf(angle);
                        sin[offset] = sinf(angle);
                    }
            }
    for (; row < seq; row++) {
        for (index = 0; index < H3_VAE_INT8_ROPE_HALF; index++) {
            cos[row * H3_VAE_INT8_ROPE_HALF + index] = 1.0f;
            sin[row * H3_VAE_INT8_ROPE_HALF + index] = 0.0f;
        }
    }
    return 1;
}

/* Video VAE RoPE: Metal half-split (d ↔ d+rope_half), matching
 * h3_video_qkv_rope_f32. Torch A/B on DiT latents: metal+regular-H >> split_half. */
static int video_qkv_rope_f32(const float *qkv, const float *rope_cos,
                              const float *rope_sin, size_t seq, size_t heads,
                              size_t head_dim, size_t rope_half, float eps,
                              float *query, float *key, float *value) {
    size_t row, head, d;
    size_t inner = heads * head_dim;
    if (!qkv || !rope_cos || !rope_sin || !query || !key || !value ||
        rope_half * 2 > head_dim)
        return 0;
    for (row = 0; row < seq; row++) {
        const float *row_base = qkv + row * inner * 3;
        const float *c_row = rope_cos + row * rope_half;
        const float *s_row = rope_sin + row * rope_half;
        for (head = 0; head < heads; head++) {
            const float *q_base = row_base + head * head_dim * 3;
            const float *k_base = q_base + head_dim;
            const float *v_base = k_base + head_dim;
            float *q_out = query + (row * heads + head) * head_dim;
            float *k_out = key + (row * heads + head) * head_dim;
            float *v_out = value + (row * heads + head) * head_dim;
            float q_sum = 0.0f, k_sum = 0.0f, qi, ki;
            for (d = 0; d < head_dim; d++) {
                q_sum += q_base[d] * q_base[d];
                k_sum += k_base[d] * k_base[d];
                v_out[d] = v_base[d];
            }
            qi = 1.0f / sqrtf(q_sum / (float)head_dim + eps);
            ki = 1.0f / sqrtf(k_sum / (float)head_dim + eps);
            for (d = 0; d < head_dim; d++) {
                float q0 = q_base[d] * qi;
                float k0 = k_base[d] * ki;
                if (d < rope_half) {
                    size_t pair = d + rope_half;
                    float q1 = q_base[pair] * qi;
                    float k1 = k_base[pair] * ki;
                    float c = c_row[d];
                    float s = s_row[d];
                    q0 = q0 * c - q1 * s;
                    k0 = k0 * c - k1 * s;
                } else if (d < rope_half * 2) {
                    size_t pair = d - rope_half;
                    float q1 = q_base[pair] * qi;
                    float k1 = k_base[pair] * ki;
                    float c = c_row[pair];
                    float s = s_row[pair];
                    q0 = q0 * c + q1 * s;
                    k0 = k0 * c + k1 * s;
                }
                q_out[d] = q0;
                k_out[d] = k0;
            }
        }
    }
    return 1;
}

static int sdpa_qkv_f32(const float *query, const float *key, const float *value,
                        size_t seq, size_t heads, size_t head_dim, float *out) {
    size_t s, h, t, d;
    float scale = 1.0f / sqrtf((float)head_dim);
    float *scores = (float *)malloc(seq * sizeof(float));
    if (!scores) return 0;
    for (s = 0; s < seq; s++) {
        for (h = 0; h < heads; h++) {
            const float *q = query + (s * heads + h) * head_dim;
            float *dst = out + (s * heads + h) * head_dim;
            float maxv = -1e30f;
            for (t = 0; t < seq; t++) {
                const float *k = key + (t * heads + h) * head_dim;
                float sum = 0.0f;
                for (d = 0; d < head_dim; d++) sum += q[d] * k[d];
                scores[t] = sum * scale;
                if (scores[t] > maxv) maxv = scores[t];
            }
            {
                float denom = 0.0f;
                for (t = 0; t < seq; t++) {
                    scores[t] = expf(scores[t] - maxv);
                    denom += scores[t];
                }
                for (d = 0; d < head_dim; d++) dst[d] = 0.0f;
                for (t = 0; t < seq; t++) {
                    const float *v = value + (t * heads + h) * head_dim;
                    float a = scores[t] / denom;
                    for (d = 0; d < head_dim; d++) dst[d] += a * v[d];
                }
            }
        }
    }
    free(scores);
    return 1;
}

int h3_vae_int8_store_open(const char *path, h3_vae_int8_store *store,
                           char *error, size_t error_size) {
    struct stat st;
    if (!path || !store) {
        fail(error, error_size, "invalid vae_int8 open args");
        return 0;
    }
    memset(store, 0, sizeof(*store));
    store->map_fd = -1;
    snprintf(store->path, sizeof(store->path), "%s", path);
    if (!h3_st_read_header(path, &store->header, error, error_size)) return 0;
    h3_st_quant_info_from_header(&store->header, &store->quant);
    store->map_fd = open(path, O_RDONLY);
    if (store->map_fd >= 0 && fstat(store->map_fd, &st) == 0 && st.st_size > 0) {
        void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE,
                         store->map_fd, 0);
        if (map != MAP_FAILED) {
            store->map = (const uint8_t *)map;
            store->map_len = (size_t)st.st_size;
#ifdef MADV_SEQUENTIAL
            (void)madvise((void *)store->map, store->map_len, MADV_SEQUENTIAL);
#endif
        } else {
            close(store->map_fd);
            store->map_fd = -1;
        }
    } else if (store->map_fd >= 0) {
        close(store->map_fd);
        store->map_fd = -1;
    }
    /* File-level metadata often omits convrot; scan a comfy_quant blob. */
    if (store->quant.convrot < 0 || store->quant.convrot_groupsize < 0) {
        size_t ti;
        for (ti = 0; ti < store->header.tensor_count; ti++) {
            const h3_st_tensor *t = &store->header.tensors[ti];
            char blob[256];
            size_t nbytes;
            if (!t->name || !strstr(t->name, "comfy_quant")) continue;
            nbytes = (size_t)h3_st_tensor_nbytes(t);
            if (nbytes == 0 || nbytes + 1 > sizeof(blob)) continue;
            if (!h3_st_read_data(&store->header, t, blob, nbytes, error,
                                 error_size))
                continue;
            blob[nbytes] = '\0';
            /* Re-scan into quant info (only fills unset fields). */
            {
                const char *p = strstr(blob, "\"convrot\"");
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
            }
            if (store->quant.convrot >= 0 && store->quant.convrot_groupsize > 0)
                break;
        }
    }
    /* Native F16/BF16/F32 VAE is not ConvRot-quantized. Defaulting ConvRot
     * ON (INT8 checkpoint quirk) would scramble fp16 weights into a grid. */
    {
        const h3_st_tensor *probe = h3_st_find(
            &store->header, "decoder.transformer_blocks.0.attn.to_qkv.weight");
        if (probe && probe->dtype != H3_DTYPE_I8)
            store->quant.convrot = 0;
    }
    return 1;
}

void h3_vae_int8_store_free(h3_vae_int8_store *store) {
    if (!store) return;
    if (store->map && store->map_len) {
        munmap((void *)store->map, store->map_len);
        store->map = NULL;
        store->map_len = 0;
    }
    if (store->map_fd >= 0) {
        close(store->map_fd);
        store->map_fd = -1;
    }
    h3_st_free_header(&store->header);
    memset(store, 0, sizeof(*store));
    store->map_fd = -1;
}

int h3_vae_int8_find_linear(const h3_vae_int8_store *store, const char *base,
                            h3_vae_int8_linear *out, char *error,
                            size_t error_size) {
    char wname[288], sname[288], bname[288];
    const h3_st_tensor *w, *s, *b;
    if (!store || !base || !out) {
        fail(error, error_size, "invalid find_linear args");
        return 0;
    }
    memset(out, 0, sizeof(*out));
    snprintf(out->base_name, sizeof(out->base_name), "%s", base);
    snprintf(wname, sizeof(wname), "%s.weight", base);
    snprintf(sname, sizeof(sname), "%s.weight_scale", base);
    snprintf(bname, sizeof(bname), "%s.bias", base);
    w = h3_st_find(&store->header, wname);
    s = h3_st_find(&store->header, sname);
    b = h3_st_find(&store->header, bname);
    if (!w) {
        fail(error, error_size, "missing weight");
        return 0;
    }
    if (w->ndim != 2) {
        fail(error, error_size, "unexpected weight rank");
        return 0;
    }
    out->weight = w;
    out->bias = b;
    out->n = (size_t)w->shape[0];
    out->k = (size_t)w->shape[1];
    if (out->n == 0 || out->k == 0) {
        fail(error, error_size, "empty linear shape");
        return 0;
    }
    if (w->dtype == H3_DTYPE_I8) {
        if (!s) {
            fail(error, error_size, "missing weight or weight_scale");
            return 0;
        }
        if (s->dtype != H3_DTYPE_F32) {
            fail(error, error_size, "unexpected weight/scale dtype or rank");
            return 0;
        }
        if (h3_st_tensor_elements(s) != out->n &&
            !(s->ndim == 2 && (size_t)s->shape[0] == out->n &&
              (size_t)s->shape[1] == 1)) {
            fail(error, error_size, "weight_scale shape mismatch");
            return 0;
        }
        out->scale = s;
        return 1;
    }
    if (w->dtype == H3_DTYPE_F16 || w->dtype == H3_DTYPE_BF16 ||
        w->dtype == H3_DTYPE_F32) {
        /* Native float VAE (e.g. minimax_h3_video_vae_fp16): no scale, no
         * ConvRot. Do not force the INT8 I8+weight_scale contract. */
        out->scale = NULL;
        return 1;
    }
    fail(error, error_size, "unexpected weight/scale dtype or rank");
    return 0;
}

/* Kijai VAE comfy_quant: {"format":"int8_tensorwise","convrot":true,
 * "convrot_groupsize":256}. Weights stay in rotated basis (W_rot); callers
 * must ConvRot activations (X@H) before GEMM. */
int h3_vae_int8_convrot_groupsize(const h3_vae_int8_store *store, size_t k) {
    const char *e;
    int gs;
    e = getenv("H3_VAE_NO_CONVROT");
    if (e && e[0] == '1') return 0;
    gs = store && store->quant.convrot_groupsize > 0
             ? store->quant.convrot_groupsize
             : H3_W4A8_DEFAULT_CONVROT_GROUPSIZE;
    if (gs < 4 || k % (size_t)gs != 0) return 0;
    if (store && store->quant.convrot == 0) return 0;
    return gs;
}

float *h3_vae_int8_dequant_f32(const h3_vae_int8_store *store,
                               const h3_vae_int8_linear *lin, char *error,
                               size_t error_size) {
    int8_t *w_i8 = NULL;
    float *scales = NULL;
    float *out = NULL;
    size_t i, j;
    if (!store || !lin || !lin->weight) {
        fail(error, error_size, "invalid dequant args");
        return NULL;
    }
    if (lin->weight->dtype == H3_DTYPE_F16 ||
        lin->weight->dtype == H3_DTYPE_BF16 ||
        lin->weight->dtype == H3_DTYPE_F32) {
        char wname[288];
        snprintf(wname, sizeof(wname), "%s.weight", lin->base_name);
        return load_f32_named(store, wname, lin->n * lin->k, error,
                              error_size);
    }
    if (lin->weight->dtype != H3_DTYPE_I8 || !lin->scale) {
        fail(error, error_size, "invalid dequant args");
        return NULL;
    }
    w_i8 = (int8_t *)malloc(lin->n * lin->k);
    scales = (float *)malloc(lin->n * sizeof(float));
    out = (float *)malloc(lin->n * lin->k * sizeof(float));
    if (!w_i8 || !scales || !out) {
        fail(error, error_size, "OOM dequant");
        goto fail;
    }
    if (!h3_vae_int8_store_copy(store, (size_t)lin->weight->file_offset, w_i8,
                                lin->n * lin->k, error, error_size) ||
        !h3_vae_int8_store_copy(store, (size_t)lin->scale->file_offset, scales,
                                lin->n * sizeof(float), error, error_size))
        goto fail;
    for (i = 0; i < lin->n; i++) {
        float s = scales[i];
        const int8_t *row = w_i8 + i * lin->k;
        float *dst = out + i * lin->k;
        for (j = 0; j < lin->k; j++) dst[j] = (float)row[j] * s;
    }
    free(w_i8);
    free(scales);
    return out;
fail:
    free(w_i8);
    free(scales);
    free(out);
    return NULL;
}

int h3_vae_int8_gemm_f32(const float *x, const float *w, size_t m, size_t n,
                         size_t k, float *y, char *error, size_t error_size) {
    size_t r, o, i;
    if (!x || !w || !y || m == 0 || n == 0 || k == 0) {
        fail(error, error_size, "invalid gemm args");
        return 0;
    }
    for (r = 0; r < m; r++) {
        const float *xr = x + r * k;
        float *yr = y + r * n;
        for (o = 0; o < n; o++) {
            const float *wr = w + o * k;
            float sum = 0.0f;
            for (i = 0; i < k; i++) sum += xr[i] * wr[i];
            yr[o] = sum;
        }
    }
    return 1;
}

int h3_vae_int8_linear_f32(const h3_vae_int8_store *store,
                           const h3_vae_int8_linear *lin, const float *x,
                           size_t m, float *y, char *error, size_t error_size) {
    float *w = NULL;
    float *bias = NULL;
    float *x_rot = NULL;
    const float *x_use = x;
    size_t i, o;
    int ok = 0;
    int gs;
    if (!store || !lin || !x || !y || m == 0) {
        fail(error, error_size, "invalid linear_f32 args");
        return 0;
    }
    w = h3_vae_int8_dequant_f32(store, lin, error, error_size);
    if (!w) return 0;
    gs = h3_vae_int8_convrot_groupsize(store, lin->k);
    if (gs > 0) {
        x_rot = (float *)malloc(m * lin->k * sizeof(float));
        if (!x_rot) {
            fail(error, error_size, "OOM act ConvRot");
            goto done;
        }
        if (!h3_w4a8_convrot_act_f32(x, m, lin->k, gs, x_rot, error,
                                     error_size))
            goto done;
        x_use = x_rot;
    }
    if (!h3_vae_int8_gemm_f32(x_use, w, m, lin->n, lin->k, y, error,
                              error_size))
        goto done;
    if (lin->bias) {
        char bname[288];
        snprintf(bname, sizeof(bname), "%s.bias", lin->base_name);
        bias = load_f32_named(store, bname, lin->n, error, error_size);
        if (!bias) goto done;
        for (i = 0; i < m; i++)
            for (o = 0; o < lin->n; o++) y[i * lin->n + o] += bias[o];
    }
    ok = 1;
done:
    free(w);
    free(bias);
    free(x_rot);
    return ok;
}

float *h3_vae_int8_unpack_dit_latent_f32(const float *packed, int latent_t,
                                         int latent_h, int latent_w,
                                         char *error, size_t error_size) {
    float *chw = NULL;
    int t, h, w, c;
    size_t tokens;
    if (!packed || latent_t < 1 || latent_h < 2 || latent_w < 2 ||
        (latent_h % 2) || (latent_w % 2)) {
        fail(error, error_size, "invalid unpack_dit_latent dims");
        return NULL;
    }
    tokens = (size_t)latent_t * (size_t)(latent_h / 2) * (size_t)(latent_w / 2);
    chw = (float *)calloc((size_t)H3_VAE_INT8_LATENT_CH * (size_t)latent_t *
                              (size_t)latent_h * (size_t)latent_w,
                          sizeof(float));
    if (!chw) {
        fail(error, error_size, "OOM unpack latent");
        return NULL;
    }
    /* DiT token packs 2x2×24 into 96 using Metal h3_dit_patchify_video order:
     * for c in 24: for dy in 2: for dx in 2  (channel-major over the 2x2). */
    for (t = 0; t < latent_t; t++) {
        int th = latent_h / 2, tw = latent_w / 2;
        int yh, xw;
        for (yh = 0; yh < th; yh++)
            for (xw = 0; xw < tw; xw++) {
                size_t tok =
                    ((size_t)t * (size_t)th + (size_t)yh) * (size_t)tw +
                    (size_t)xw;
                const float *src = packed + tok * 96;
                int dy, dx;
                size_t idx = 0;
                for (c = 0; c < H3_VAE_INT8_LATENT_CH; c++)
                    for (dy = 0; dy < 2; dy++)
                        for (dx = 0; dx < 2; dx++, idx++) {
                            int hh = yh * 2 + dy;
                            int ww = xw * 2 + dx;
                            size_t dst =
                                (((size_t)c * (size_t)latent_t + (size_t)t) *
                                     (size_t)latent_h +
                                 (size_t)hh) *
                                    (size_t)latent_w +
                                (size_t)ww;
                            chw[dst] = src[idx];
                        }
                (void)tokens;
            }
    }
    return chw;
}

static int run_block(const h3_vae_int8_store *store, unsigned block,
                     float *seq_h, size_t seq, const float *rope_cos,
                     const float *rope_sin, char *error, size_t error_size) {
    char base[128];
    h3_vae_int8_linear qkv, out, w1, w2;
    float *norm1 = NULL, *norm2 = NULL, *scale1 = NULL, *scale2 = NULL;
    float *tmp = NULL, *qkv_y = NULL, *attn = NULL, *branch = NULL;
    float *ff = NULL, *act = NULL;
    float *query = NULL, *key = NULL, *value = NULL;
    size_t hidden = H3_VAE_INT8_HIDDEN;
    size_t inner = (size_t)H3_VAE_INT8_HEADS * H3_VAE_INT8_HEAD_DIM;
    int ok = 0;

    snprintf(base, sizeof(base), "decoder.transformer_blocks.%u.attn.to_qkv",
             block);
    if (!h3_vae_int8_find_linear(store, base, &qkv, error, error_size))
        return 0;
    snprintf(base, sizeof(base), "decoder.transformer_blocks.%u.attn.to_out",
             block);
    if (!h3_vae_int8_find_linear(store, base, &out, error, error_size))
        return 0;
    snprintf(base, sizeof(base), "decoder.transformer_blocks.%u.ff.w1", block);
    if (!h3_vae_int8_find_linear(store, base, &w1, error, error_size)) return 0;
    snprintf(base, sizeof(base), "decoder.transformer_blocks.%u.ff.w2", block);
    if (!h3_vae_int8_find_linear(store, base, &w2, error, error_size)) return 0;

    snprintf(base, sizeof(base), "decoder.transformer_blocks.%u.norm1.weight",
             block);
    norm1 = load_f32_named(store, base, hidden, error, error_size);
    snprintf(base, sizeof(base), "decoder.transformer_blocks.%u.norm2.weight",
             block);
    norm2 = load_f32_named(store, base, hidden, error, error_size);
    snprintf(base, sizeof(base), "decoder.transformer_blocks.%u.scale1", block);
    scale1 = load_f32_named(store, base, hidden, error, error_size);
    snprintf(base, sizeof(base), "decoder.transformer_blocks.%u.scale2", block);
    scale2 = load_f32_named(store, base, hidden, error, error_size);
    tmp = (float *)malloc(seq * hidden * sizeof(float));
    qkv_y = (float *)malloc(seq * qkv.n * sizeof(float));
    query = (float *)malloc(seq * inner * sizeof(float));
    key = (float *)malloc(seq * inner * sizeof(float));
    value = (float *)malloc(seq * inner * sizeof(float));
    attn = (float *)malloc(seq * hidden * sizeof(float));
    branch = (float *)malloc(seq * hidden * sizeof(float));
    ff = (float *)malloc(seq * w1.n * sizeof(float));
    act = (float *)malloc(seq * H3_VAE_INT8_FFN * sizeof(float));
    if (!norm1 || !norm2 || !scale1 || !scale2 || !tmp || !qkv_y || !query ||
        !key || !value || !attn || !branch || !ff || !act) {
        fail(error, error_size, "OOM vae block");
        goto done;
    }

    rms_norm(seq_h, norm1, seq, hidden, 1e-5f, tmp);
    if (!h3_vae_int8_linear_f32(store, &qkv, tmp, seq, qkv_y, error,
                                error_size))
        goto done;
    if (!video_qkv_rope_f32(qkv_y, rope_cos, rope_sin, seq, H3_VAE_INT8_HEADS,
                            H3_VAE_INT8_HEAD_DIM, H3_VAE_INT8_ROPE_HALF, 1e-5f,
                            query, key, value)) {
        fail(error, error_size, "vae video_qkv_rope failed");
        goto done;
    }
    if (!sdpa_qkv_f32(query, key, value, seq, H3_VAE_INT8_HEADS,
                      H3_VAE_INT8_HEAD_DIM, attn)) {
        fail(error, error_size, "vae sdpa failed");
        goto done;
    }
    if (!h3_vae_int8_linear_f32(store, &out, attn, seq, branch, error,
                                error_size))
        goto done;
    {
        size_t i, c;
        for (i = 0; i < seq; i++)
            for (c = 0; c < hidden; c++)
                seq_h[i * hidden + c] +=
                    branch[i * hidden + c] * scale1[c];
    }

    rms_norm(seq_h, norm2, seq, hidden, 1e-5f, tmp);
    if (!h3_vae_int8_linear_f32(store, &w1, tmp, seq, ff, error, error_size))
        goto done;
    silu_mul(ff, seq, H3_VAE_INT8_FFN, act);
    if (!h3_vae_int8_linear_f32(store, &w2, act, seq, branch, error,
                                error_size))
        goto done;
    {
        size_t i, c;
        for (i = 0; i < seq; i++)
            for (c = 0; c < hidden; c++)
                seq_h[i * hidden + c] +=
                    branch[i * hidden + c] * scale2[c];
    }
    ok = 1;
done:
    free(norm1);
    free(norm2);
    free(scale1);
    free(scale2);
    free(tmp);
    free(qkv_y);
    free(query);
    free(key);
    free(value);
    free(attn);
    free(branch);
    free(ff);
    free(act);
    return ok;
}

int h3_vae_int8_decode_f32(const h3_vae_int8_store *store,
                           const float *latent_chw, int latent_t, int latent_h,
                           int latent_w, unsigned n_blocks,
                           h3_vae_int8_frames *frames,
                           h3_vae_int8_decode_stats *stats, char *error,
                           size_t error_size) {
    float *post_w = NULL, *post_b = NULL;
    float *embed_w = NULL, *embed_b = NULL;
    float *registers = NULL;
    float *norm_w = NULL, *norm_b = NULL;
    float *proj_w = NULL, *proj_b = NULL;
    float *patches = NULL, *seq_h = NULL, *projected = NULL, *rgb = NULL;
    float *rope_cos = NULL, *rope_sin = NULL;
    float latent_mean[H3_VAE_INT8_LATENT_CH];
    float latent_std[H3_VAE_INT8_LATENT_CH];
    size_t n_patches, seq, i, c, t, h, w;
    unsigned b;
    clock_t t0, t1, tall0, tall1;
    int ok = 0;
    int pixel_h, pixel_w, out_frames;

    if (stats) memset(stats, 0, sizeof(*stats));
    if (!store || !latent_chw || !frames || latent_t < 1 || latent_h < 1 ||
        latent_w < 1) {
        fail(error, error_size, "invalid decode args");
        return 0;
    }
    if (n_blocks > H3_VAE_INT8_N_BLOCKS) n_blocks = H3_VAE_INT8_N_BLOCKS;
    memset(frames, 0, sizeof(*frames));
    n_patches = (size_t)latent_t * (size_t)latent_h * (size_t)latent_w;
    seq = n_patches + H3_VAE_INT8_REGISTERS + 1; /* +1 zero token like Metal */
    pixel_h = latent_h * H3_VAE_INT8_SPATIAL_RATIO;
    pixel_w = latent_w * H3_VAE_INT8_SPATIAL_RATIO;
    out_frames = latent_t * 4; /* OUTPUT_PATCH packs 4 temporal */

    tall0 = clock();
    t0 = clock();
    if (!h3_vae_int8_load_latent_stats(store, latent_mean, latent_std, error,
                                       error_size))
        return 0;
    post_w = load_f32_named(store, "post_quant_conv.weight",
                            (size_t)H3_VAE_INT8_LATENT_CH *
                                H3_VAE_INT8_LATENT_CH,
                            error, error_size);
    post_b = load_f32_named(store, "post_quant_conv.bias",
                            H3_VAE_INT8_LATENT_CH, error, error_size);
    embed_w =
        load_f32_named(store, "decoder.x_embedder.weight",
                       (size_t)H3_VAE_INT8_HIDDEN * H3_VAE_INT8_LATENT_CH,
                       error, error_size);
    embed_b = load_f32_named(store, "decoder.x_embedder.bias",
                             H3_VAE_INT8_HIDDEN, error, error_size);
    registers = load_f32_named(store, "decoder.register_tokens",
                               (size_t)H3_VAE_INT8_REGISTERS *
                                   H3_VAE_INT8_HIDDEN,
                               error, error_size);
    norm_w = load_f32_named(store, "decoder.norm_out.weight",
                            H3_VAE_INT8_HIDDEN, error, error_size);
    norm_b = load_f32_named(store, "decoder.norm_out.bias", H3_VAE_INT8_HIDDEN,
                            error, error_size);
    proj_w =
        load_f32_named(store, "decoder.proj_out.weight",
                       (size_t)H3_VAE_INT8_OUTPUT_PATCH * H3_VAE_INT8_HIDDEN,
                       error, error_size);
    proj_b = load_f32_named(store, "decoder.proj_out.bias",
                            H3_VAE_INT8_OUTPUT_PATCH, error, error_size);
    patches = (float *)malloc(n_patches * H3_VAE_INT8_LATENT_CH * sizeof(float));
    seq_h = (float *)calloc(seq * H3_VAE_INT8_HIDDEN, sizeof(float));
    projected =
        (float *)malloc(seq * H3_VAE_INT8_OUTPUT_PATCH * sizeof(float));
    rope_cos = (float *)malloc(seq * H3_VAE_INT8_ROPE_HALF * sizeof(float));
    rope_sin = (float *)malloc(seq * H3_VAE_INT8_ROPE_HALF * sizeof(float));
    if (!post_w || !post_b || !embed_w || !embed_b || !registers || !norm_w ||
        !norm_b || !proj_w || !proj_b || !patches || !seq_h || !projected ||
        !rope_cos || !rope_sin) {
        fail(error, error_size, "OOM decode heads");
        goto done;
    }
    if (!h3_vae_int8_prepare_rope(latent_t, latent_h, latent_w, n_patches, seq,
                                  rope_cos, rope_sin, error, error_size))
        goto done;
    t1 = clock();
    if (stats) stats->load_ms = elapsed_ms(t0, t1);

    /* CHW → denorm (Metal prepare_input) → patch-major → post_quant (1x1). */
    i = 0;
    for (t = 0; t < (size_t)latent_t; t++)
        for (h = 0; h < (size_t)latent_h; h++)
            for (w = 0; w < (size_t)latent_w; w++, i++) {
                float x[H3_VAE_INT8_LATENT_CH];
                float y[H3_VAE_INT8_LATENT_CH];
                for (c = 0; c < H3_VAE_INT8_LATENT_CH; c++) {
                    size_t src =
                        ((c * (size_t)latent_t + t) * (size_t)latent_h + h) *
                            (size_t)latent_w +
                        w;
                    x[c] = latent_chw[src] * latent_std[c] + latent_mean[c];
                }
                for (c = 0; c < H3_VAE_INT8_LATENT_CH; c++) {
                    const float *wr =
                        post_w + c * H3_VAE_INT8_LATENT_CH;
                    float sum = post_b[c];
                    size_t k;
                    for (k = 0; k < H3_VAE_INT8_LATENT_CH; k++)
                        sum += wr[k] * x[k];
                    y[c] = sum;
                }
                memcpy(patches + i * H3_VAE_INT8_LATENT_CH, y,
                       H3_VAE_INT8_LATENT_CH * sizeof(float));
            }

    /* embed patches */
    if (!h3_vae_int8_gemm_f32(patches, embed_w, n_patches, H3_VAE_INT8_HIDDEN,
                              H3_VAE_INT8_LATENT_CH, seq_h, error,
                              error_size))
        goto done;
    for (i = 0; i < n_patches; i++)
        for (c = 0; c < H3_VAE_INT8_HIDDEN; c++)
            seq_h[i * H3_VAE_INT8_HIDDEN + c] += embed_b[c];
    memcpy(seq_h + n_patches * H3_VAE_INT8_HIDDEN, registers,
           (size_t)H3_VAE_INT8_REGISTERS * H3_VAE_INT8_HIDDEN * sizeof(float));
    /* trailing zero token already zeroed by calloc */

    t0 = clock();
    for (b = 0; b < n_blocks; b++) {
        if (!run_block(store, b, seq_h, seq, rope_cos, rope_sin, error,
                       error_size))
            goto done;
    }
    t1 = clock();
    if (stats) {
        stats->blocks_ms = elapsed_ms(t0, t1);
        stats->n_blocks = n_blocks;
    }

    {
        float *normed =
            (float *)malloc(seq * H3_VAE_INT8_HIDDEN * sizeof(float));
        if (!normed) {
            fail(error, error_size, "OOM norm_out");
            goto done;
        }
        layer_norm_bias(seq_h, norm_w, norm_b, seq, H3_VAE_INT8_HIDDEN, 1e-5f,
                        normed);
        if (!h3_vae_int8_gemm_f32(normed, proj_w, seq, H3_VAE_INT8_OUTPUT_PATCH,
                                  H3_VAE_INT8_HIDDEN, projected, error,
                                  error_size)) {
            free(normed);
            goto done;
        }
        for (i = 0; i < seq; i++)
            for (c = 0; c < H3_VAE_INT8_OUTPUT_PATCH; c++)
                projected[i * H3_VAE_INT8_OUTPUT_PATCH + c] += proj_b[c];
        free(normed);
    }

    /* Unpack patch tokens (exclude registers/zero) → RGB frames. */
    {
        static const float mean[] = {0.485f, 0.456f, 0.406f};
        static const float deviation[] = {0.229f, 0.224f, 0.225f};
        size_t rgb_n =
            (size_t)out_frames * (size_t)pixel_h * (size_t)pixel_w * 3;
        int frame, y, x, ch;
        rgb = (float *)malloc(rgb_n * sizeof(float));
        if (!rgb) {
            fail(error, error_size, "OOM rgb");
            goto done;
        }
        for (frame = 0; frame < out_frames; frame++) {
            /* Metal unpack_frame_range: decoded_t = frame + FRAME_OFFSET(3). */
            int decoded_t = frame + 3;
            int patch_t = decoded_t / 4;
            int within_t = decoded_t % 4;
            if (patch_t >= latent_t) patch_t = latent_t - 1;
            for (y = 0; y < pixel_h; y++) {
                int patch_y = y / 16, within_y = y % 16;
                for (x = 0; x < pixel_w; x++) {
                    int patch_x = x / 16, within_x = x % 16;
                    size_t patch =
                        ((size_t)patch_t * (size_t)latent_h +
                         (size_t)patch_y) *
                            (size_t)latent_w +
                        (size_t)patch_x;
                    for (ch = 0; ch < 3; ch++) {
                        size_t component =
                            ((((size_t)ch * 4 + (size_t)within_t) * 16 +
                              (size_t)within_y) *
                                 16 +
                             (size_t)within_x);
                        float value =
                            projected[patch * H3_VAE_INT8_OUTPUT_PATCH +
                                      component] *
                                deviation[ch] +
                            mean[ch];
                        size_t dst =
                            (((size_t)frame * (size_t)pixel_h + (size_t)y) *
                                 (size_t)pixel_w +
                             (size_t)x) *
                                3 +
                            (size_t)ch;
                        if (value < 0.0f) value = 0.0f;
                        if (value > 1.0f) value = 1.0f;
                        rgb[dst] = value;
                    }
                }
            }
        }
        frames->rgb = rgb;
        frames->frames = out_frames;
        frames->height = pixel_h;
        frames->width = pixel_w;
        rgb = NULL;
        h3_vae_int8_deblock16_rgb(frames->rgb, frames->frames, frames->height,
                                  frames->width);
    }

    tall1 = clock();
    if (stats) {
        size_t n =
            (size_t)frames->frames * (size_t)frames->height *
            (size_t)frames->width * 3;
        stats->total_ms = elapsed_ms(tall0, tall1);
        stats->all_finite = 1;
        for (i = 0; i < n; i++) {
            if (!isfinite(frames->rgb[i])) {
                stats->all_finite = 0;
                break;
            }
        }
    }
    ok = 1;
done:
    free(post_w);
    free(post_b);
    free(embed_w);
    free(embed_b);
    free(registers);
    free(norm_w);
    free(norm_b);
    free(proj_w);
    free(proj_b);
    free(patches);
    free(seq_h);
    free(projected);
    free(rope_cos);
    free(rope_sin);
    free(rgb);
    if (!ok) h3_vae_int8_frames_free(frames);
    return ok;
}

void h3_vae_int8_deblock16_rgb(float *rgb, int frames, int height, int width) {
    /* Soften residual VAE 16×16 patch seams. Strength from H3_VAE_DEBLOCK
     * (0 disables; default 0.28 — enough to clear b16≈1.35→<1.3 without
     * washing continuous texture). */
    const char *e = getenv("H3_VAE_DEBLOCK");
    float strength = 0.28f;
    int f, y, x, c;
    static int logged;
    if (e) {
        if (e[0] == '0' && e[1] == '\0') return;
        strength = (float)atof(e);
    }
    if (!(strength > 0.0f) || !rgb || frames < 1 || height < 17 || width < 17)
        return;
    if (strength > 0.75f) strength = 0.75f;
    if (!logged) {
        fprintf(stderr, "vae_deblock16: strength=%.2f (16px patch seams)\n",
                (double)strength);
        logged = 1;
    }
#ifdef _OPENMP
#pragma omp parallel for private(y, x, c)
#endif
    for (f = 0; f < frames; f++) {
        float *fr = rgb + (size_t)f * (size_t)height * (size_t)width * 3;
        for (y = 0; y < height; y++) {
            for (x = 16; x < width; x += 16) {
                for (c = 0; c < 3; c++) {
                    size_t i = ((size_t)y * (size_t)width + (size_t)x) * 3 +
                               (size_t)c;
                    size_t L = i - 3;
                    size_t R = (x + 1 < width) ? i + 3 : i;
                    float mid = 0.5f * (fr[L] + fr[R]);
                    fr[i] = fr[i] * (1.0f - strength) + mid * strength;
                    fr[L] = fr[L] * (1.0f - 0.5f * strength) +
                            (0.5f * (fr[L] + fr[i])) * (0.5f * strength);
                }
            }
        }
        for (x = 0; x < width; x++) {
            for (y = 16; y < height; y += 16) {
                for (c = 0; c < 3; c++) {
                    size_t i = ((size_t)y * (size_t)width + (size_t)x) * 3 +
                               (size_t)c;
                    size_t U = i - (size_t)width * 3;
                    size_t D =
                        (y + 1 < height) ? i + (size_t)width * 3 : i;
                    float mid = 0.5f * (fr[U] + fr[D]);
                    fr[i] = fr[i] * (1.0f - strength) + mid * strength;
                }
            }
        }
    }
}

void h3_vae_int8_frames_free(h3_vae_int8_frames *frames) {
    if (!frames) return;
    free(frames->rgb);
    memset(frames, 0, sizeof(*frames));
}

int h3_vae_int8_write_ppm(const h3_vae_int8_frames *frames, int frame_index,
                          const char *path, char *error, size_t error_size) {
    FILE *fp;
    const float *src;
    if (!frames || !frames->rgb || !path || frame_index < 0 ||
        frame_index >= frames->frames) {
        fail(error, error_size, "invalid write_ppm args");
        return 0;
    }
    fp = fopen(path, "wb");
    if (!fp) {
        fail(error, error_size, "cannot open ppm for write");
        return 0;
    }
    fprintf(fp, "P6\n%d %d\n255\n", frames->width, frames->height);
    src = frames->rgb +
          (size_t)frame_index * (size_t)frames->height * (size_t)frames->width *
              3;
    {
        size_t npx = (size_t)frames->width * (size_t)frames->height;
        unsigned char *buf = (unsigned char *)malloc(npx * 3);
        size_t p;
        if (!buf) {
            fclose(fp);
            fail(error, error_size, "OOM ppm frame buffer");
            return 0;
        }
        for (p = 0; p < npx; p++) {
            float r = src[p * 3 + 0] * 255.0f + 0.5f;
            float g = src[p * 3 + 1] * 255.0f + 0.5f;
            float b = src[p * 3 + 2] * 255.0f + 0.5f;
            buf[p * 3 + 0] = (unsigned char)r;
            buf[p * 3 + 1] = (unsigned char)g;
            buf[p * 3 + 2] = (unsigned char)b;
        }
        if (fwrite(buf, 1, npx * 3, fp) != npx * 3) {
            free(buf);
            fclose(fp);
            fail(error, error_size, "ppm fwrite short");
            return 0;
        }
        free(buf);
    }
    fclose(fp);
    return 1;
}
