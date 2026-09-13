#include "h3_vision_cuda.h"

#include "h3_gpu_cuda.h"
#include "h3_safetensors.h"
#include "h3_sdpa.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

enum {
    VISION_HIDDEN = 1152,
    VISION_INTERMEDIATE = 4304,
    VISION_LAYERS = 27,
    VISION_HEADS = 16,
    VISION_HEAD_DIM = 72,
    VISION_ROPE_HALF = 36,
    PATCH = 16,
    TEMPORAL_PATCH = 2,
    MERGE = 2,
    PATCH_DIM = 3 * TEMPORAL_PATCH * PATCH * PATCH,
    MERGE_DIM = VISION_HIDDEN * MERGE * MERGE,
    POSITION_SIDE = 48,
    POSITION_COUNT = POSITION_SIDE * POSITION_SIDE,
    OUTPUT_WIDTH = 5120
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static float *load_f32(const h3_st_header *header, const char *prefix,
                       const char *suffix, size_t expect, char *error,
                       size_t error_size) {
    char name[192];
    size_t n = 0;
    float *w;
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);
    w = h3_st_load_f32(header, name, &n, NULL, error, error_size);
    if (!w) return NULL;
    if (expect && n != expect) {
        free(w);
        fail(error, error_size, "%s size mismatch (got %zu want %zu)", name, n,
             expect);
        return NULL;
    }
    return w;
}

static int detect_prefix(const h3_st_header *header, char *prefix,
                         size_t prefix_size) {
    if (h3_st_find(header, "visual.patch_embed.proj.weight")) {
        snprintf(prefix, prefix_size, "visual.");
        return 1;
    }
    if (h3_st_find(header, "model.visual.patch_embed.proj.weight")) {
        snprintf(prefix, prefix_size, "model.visual.");
        return 1;
    }
    return 0;
}

static h3_cuda *g_vis_cuda = NULL;
static double g_vis_sdpa_ms = 0.0;
static int g_vis_sdpa_calls = 0;
static char g_vis_sdpa_backend[64] = "host";

static int vision_want_cuda(void) {
    const char *force = getenv("H3_VISION_HOST");
    return g_vis_cuda && !(force && force[0] == '1');
}

/* Default: cuDNN when CUDA bound. H3_VISION_SDPA=host keeps cublas GEMM but
 * host OpenMP SDPA (A/B / rollback of the SDPA knife only). */
static int vision_want_sdpa_cuda(void) {
    const char *e;
    if (!vision_want_cuda()) return 0;
    e = getenv("H3_VISION_SDPA");
    if (e && (e[0] == 'h' || e[0] == 'H') &&
        (e[1] == 'o' || e[1] == 'O'))
        return 0;
    return 1;
}

/* Vision shapes are H=16 D=72 — force Tier-0 cudnn (not DiT sage3/sol). */
static void vision_sdpa_push_cudnn(char *prev_out, size_t prev_n, int *had_prev) {
    const char *prev = getenv("H3_CUDA_SDPA_BACKEND");
    *had_prev = 0;
    if (prev_out && prev_n) prev_out[0] = '\0';
    if (prev && prev[0]) {
        *had_prev = 1;
        if (prev_out && prev_n)
            snprintf(prev_out, prev_n, "%s", prev);
    }
    setenv("H3_CUDA_SDPA_BACKEND", "cudnn", 1);
    h3_sdpa_cache_invalidate();
}

static void vision_sdpa_pop_cudnn(const char *prev, int had_prev) {
    if (had_prev && prev && prev[0])
        setenv("H3_CUDA_SDPA_BACKEND", prev, 1);
    else
        unsetenv("H3_CUDA_SDPA_BACKEND");
    h3_sdpa_cache_invalidate();
}

typedef struct {
    float *n1w, *n1b, *qkv_w, *qkv_b, *ow, *ob;
    float *n2w, *n2b, *fc1w, *fc1b, *fc2w, *fc2b;
} vis_layer_w;

typedef struct {
    float *nw, *nb, *fc1w, *fc1b, *fc2w, *fc2b;
} vis_merger_w;

typedef struct {
    char path[1024];
    char prefix[32];
    float *pos;
    float *patch_w;
    float *patch_b;
    vis_layer_w layer[VISION_LAYERS];
    vis_merger_w merger;
    vis_merger_w deepstack[3];
    int ready;
} vis_w_cache;

static vis_w_cache g_vis_w;

void h3_vision_encode_f32_bind_cuda(void *cuda) {
    g_vis_cuda = (h3_cuda *)cuda;
}

static void vis_w_free(vis_w_cache *c) {
    int i;
    if (!c) return;
    free(c->pos);
    free(c->patch_w);
    free(c->patch_b);
    free(c->merger.nw);
    free(c->merger.nb);
    free(c->merger.fc1w);
    free(c->merger.fc1b);
    free(c->merger.fc2w);
    free(c->merger.fc2b);
    for (i = 0; i < 3; i++) {
        free(c->deepstack[i].nw);
        free(c->deepstack[i].nb);
        free(c->deepstack[i].fc1w);
        free(c->deepstack[i].fc1b);
        free(c->deepstack[i].fc2w);
        free(c->deepstack[i].fc2b);
    }
    for (i = 0; i < VISION_LAYERS; i++) {
        vis_layer_w *L = &c->layer[i];
        free(L->n1w);
        free(L->n1b);
        free(L->qkv_w);
        free(L->qkv_b);
        free(L->ow);
        free(L->ob);
        free(L->n2w);
        free(L->n2b);
        free(L->fc1w);
        free(L->fc1b);
        free(L->fc2w);
        free(L->fc2b);
    }
    memset(c, 0, sizeof(*c));
}

static int load_merger_w(const h3_st_header *header, const char *prefix,
                         int deepstack, vis_merger_w *m, char *error,
                         size_t error_size) {
    size_t nw = deepstack ? (size_t)MERGE_DIM : (size_t)VISION_HIDDEN;
    m->nw = load_f32(header, prefix, "norm.weight", nw, error, error_size);
    m->nb = load_f32(header, prefix, "norm.bias", nw, error, error_size);
    m->fc1w = load_f32(header, prefix, "linear_fc1.weight",
                       (size_t)MERGE_DIM * MERGE_DIM, error, error_size);
    m->fc1b = load_f32(header, prefix, "linear_fc1.bias", MERGE_DIM, error,
                       error_size);
    m->fc2w = load_f32(header, prefix, "linear_fc2.weight",
                       (size_t)OUTPUT_WIDTH * MERGE_DIM, error, error_size);
    m->fc2b = load_f32(header, prefix, "linear_fc2.bias", OUTPUT_WIDTH, error,
                       error_size);
    return m->nw && m->nb && m->fc1w && m->fc1b && m->fc2w && m->fc2b;
}

static int vis_w_ensure(const h3_st_header *header, const char *path,
                        const char *prefix, char *error, size_t error_size) {
    int i;
    char bp[128];
    struct timespec t0, t1;
    if (g_vis_w.ready && g_vis_w.path[0] && path &&
        !strcmp(g_vis_w.path, path) && !strcmp(g_vis_w.prefix, prefix))
        return 1;
    vis_w_free(&g_vis_w);
    if (path)
        snprintf(g_vis_w.path, sizeof(g_vis_w.path), "%s", path);
    snprintf(g_vis_w.prefix, sizeof(g_vis_w.prefix), "%s", prefix);
    clock_gettime(CLOCK_MONOTONIC, &t0);
    g_vis_w.pos = load_f32(header, prefix, "pos_embed.weight",
                           (size_t)POSITION_COUNT * VISION_HIDDEN, error,
                           error_size);
    g_vis_w.patch_w = load_f32(header, prefix, "patch_embed.proj.weight",
                               (size_t)VISION_HIDDEN * PATCH_DIM, error,
                               error_size);
    g_vis_w.patch_b = load_f32(header, prefix, "patch_embed.proj.bias",
                               VISION_HIDDEN, error, error_size);
    if (!g_vis_w.pos || !g_vis_w.patch_w || !g_vis_w.patch_b) {
        vis_w_free(&g_vis_w);
        return 0;
    }
    for (i = 0; i < VISION_LAYERS; i++) {
        vis_layer_w *L = &g_vis_w.layer[i];
        snprintf(bp, sizeof(bp), "%sblocks.%d.", prefix, i);
        L->n1w = load_f32(header, bp, "norm1.weight", VISION_HIDDEN, error,
                          error_size);
        L->n1b = load_f32(header, bp, "norm1.bias", VISION_HIDDEN, error,
                          error_size);
        L->qkv_w = load_f32(header, bp, "attn.qkv.weight",
                            (size_t)VISION_HIDDEN * 3 * VISION_HIDDEN, error,
                            error_size);
        L->qkv_b = load_f32(header, bp, "attn.qkv.bias", VISION_HIDDEN * 3,
                            error, error_size);
        L->ow = load_f32(header, bp, "attn.proj.weight",
                         (size_t)VISION_HIDDEN * VISION_HIDDEN, error,
                         error_size);
        L->ob = load_f32(header, bp, "attn.proj.bias", VISION_HIDDEN, error,
                         error_size);
        L->n2w = load_f32(header, bp, "norm2.weight", VISION_HIDDEN, error,
                          error_size);
        L->n2b = load_f32(header, bp, "norm2.bias", VISION_HIDDEN, error,
                          error_size);
        L->fc1w = load_f32(header, bp, "mlp.linear_fc1.weight",
                           (size_t)VISION_INTERMEDIATE * VISION_HIDDEN, error,
                           error_size);
        L->fc1b = load_f32(header, bp, "mlp.linear_fc1.bias",
                           VISION_INTERMEDIATE, error, error_size);
        L->fc2w = load_f32(header, bp, "mlp.linear_fc2.weight",
                           (size_t)VISION_HIDDEN * VISION_INTERMEDIATE, error,
                           error_size);
        L->fc2b = load_f32(header, bp, "mlp.linear_fc2.bias", VISION_HIDDEN,
                            error, error_size);
        if (!L->n1w || !L->n1b || !L->qkv_w || !L->qkv_b || !L->ow || !L->ob ||
            !L->n2w || !L->n2b || !L->fc1w || !L->fc1b || !L->fc2w ||
            !L->fc2b) {
            vis_w_free(&g_vis_w);
            return 0;
        }
    }
    snprintf(bp, sizeof(bp), "%smerger.", prefix);
    if (!load_merger_w(header, bp, 0, &g_vis_w.merger, error, error_size)) {
        vis_w_free(&g_vis_w);
        return 0;
    }
    for (i = 0; i < 3; i++) {
        snprintf(bp, sizeof(bp), "%sdeepstack_merger_list.%d.", prefix, i);
        if (!load_merger_w(header, bp, 1, &g_vis_w.deepstack[i], error,
                           error_size)) {
            vis_w_free(&g_vis_w);
            return 0;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    g_vis_w.ready = 1;
    fprintf(stderr, "vision_w_cache: loaded layers=%d ms=%.1f\n",
            VISION_LAYERS,
            (t1.tv_sec - t0.tv_sec) * 1000.0 +
                (t1.tv_nsec - t0.tv_nsec) / 1e6);
    fflush(stderr);
    return 1;
}

static void gemm_nt_host(const float *x, const float *w, const float *bias,
                         size_t m, size_t n, size_t k, float *y) {
    size_t r, c, i;
#ifdef _OPENMP
#pragma omp parallel for private(c, i)
#endif
    for (r = 0; r < m; r++) {
        for (c = 0; c < n; c++) {
            const float *xr = x + r * k;
            const float *wr = w + c * k;
            float s = bias ? bias[c] : 0.0f;
            for (i = 0; i < k; i++)
                s += xr[i] * wr[i];
            y[r * n + c] = s;
        }
    }
}

static void gemm_nt(const float *x, const float *w, const float *bias,
                    size_t m, size_t n, size_t k, float *y) {
    int want_cuda = vision_want_cuda();
    if (want_cuda && m > 0 && n > 0 && k > 0) {
        char err[192];
        size_t r, c;
        err[0] = '\0';
        if (h3_cuda_gemm_f32(g_vis_cuda, x, w, m, n, k, y, NULL, err,
                             sizeof(err))) {
            if (bias) {
                for (r = 0; r < m; r++)
                    for (c = 0; c < n; c++)
                        y[r * n + c] += bias[c];
            }
            return;
        }
        fprintf(stderr, "vision_gemm: cuda fallback host (%s)\n",
                err[0] ? err : "unknown");
    }
    gemm_nt_host(x, w, bias, m, n, k, y);
}

static void layer_norm(const float *x, const float *w, const float *b,
                       size_t rows, size_t dim, float eps, float *y) {
    size_t r, d;
#ifdef _OPENMP
#pragma omp parallel for private(d)
#endif
    for (r = 0; r < rows; r++) {
        const float *xr = x + r * dim;
        float *yr = y + r * dim;
        float mean = 0.0f, var = 0.0f, inv;
        for (d = 0; d < dim; d++)
            mean += xr[d];
        mean /= (float)dim;
        for (d = 0; d < dim; d++) {
            float c = xr[d] - mean;
            var += c * c;
        }
        inv = 1.0f / sqrtf(var / (float)dim + eps);
        for (d = 0; d < dim; d++)
            yr[d] = (xr[d] - mean) * inv * w[d] + b[d];
    }
}

static float gelu_erf(float value) {
    float sign = value < 0.0f ? -1.0f : 1.0f;
    float x = fabsf(value);
    float t = 1.0f / (1.0f + 0.3275911f * x);
    float polynomial = (((((1.061405429f * t - 1.453152027f) * t) +
                           1.421413741f) * t - 0.284496736f) * t +
                          0.254829592f) * t;
    float erf = sign * (1.0f - polynomial * expf(-x * x));
    if (value <= -10.0f) return 0.0f;
    if (value >= 10.0f) return value;
    return 0.5f * value * (1.0f + erf);
}

static float gelu_tanh(float value) {
    float inner = 0.7978845608028654f *
                  (value + 0.044715f * value * value * value);
    if (inner <= -10.0f) return 0.0f;
    if (inner >= 10.0f) return value;
    return 0.5f * value * (1.0f + tanhf(inner));
}

static void gelu(float *x, size_t n, int approximate) {
    size_t i;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (i = 0; i < n; i++)
        x[i] = approximate ? gelu_tanh(x[i]) : gelu_erf(x[i]);
}

static int prepare_patch_rows(const float *pixels, int frames, int height,
                              int width, float *rows) {
    int grid_h = height / PATCH;
    int grid_w = width / PATCH;
    size_t output = 0;
    int block_h, block_w, inner_h, inner_w, channel, temporal, patch_h, patch_w;
    for (block_h = 0; block_h < grid_h / MERGE; block_h++)
        for (block_w = 0; block_w < grid_w / MERGE; block_w++)
            for (inner_h = 0; inner_h < MERGE; inner_h++)
                for (inner_w = 0; inner_w < MERGE; inner_w++)
                    for (channel = 0; channel < 3; channel++)
                        for (temporal = 0; temporal < TEMPORAL_PATCH;
                             temporal++)
                            for (patch_h = 0; patch_h < PATCH; patch_h++)
                                for (patch_w = 0; patch_w < PATCH; patch_w++) {
                                    int frame = frames == 1 ? 0 : temporal;
                                    int y = (block_h * MERGE + inner_h) * PATCH +
                                            patch_h;
                                    int x = (block_w * MERGE + inner_w) * PATCH +
                                            patch_w;
                                    size_t input =
                                        (((size_t)frame * 3u + (size_t)channel) *
                                             (size_t)height +
                                         (size_t)y) *
                                            (size_t)width +
                                        (size_t)x;
                                    rows[output++] = pixels[input] * 2.0f - 1.0f;
                                }
    return output == (size_t)grid_h * (size_t)grid_w * PATCH_DIM;
}

static int prepare_position_rows(const float *table, int grid_h, int grid_w,
                                 float *rows) {
    size_t output_row = 0;
    int block_h, block_w, inner_h, inner_w, channel, corner;
    for (block_h = 0; block_h < grid_h / MERGE; block_h++) {
        for (block_w = 0; block_w < grid_w / MERGE; block_w++) {
            for (inner_h = 0; inner_h < MERGE; inner_h++) {
                for (inner_w = 0; inner_w < MERGE; inner_w++) {
                    int row = block_h * MERGE + inner_h;
                    int column = block_w * MERGE + inner_w;
                    double hv = (double)row * (POSITION_SIDE - 1) /
                                (double)(grid_h - 1);
                    double wv = (double)column * (POSITION_SIDE - 1) /
                                (double)(grid_w - 1);
                    int h0 = (int)hv;
                    int w0 = (int)wv;
                    int h1 = h0 + 1 < POSITION_SIDE ? h0 + 1 : h0;
                    int w1 = w0 + 1 < POSITION_SIDE ? w0 + 1 : w0;
                    float dh = (float)(hv - h0);
                    float dw = (float)(wv - w0);
                    float coefficients[4] = {(1.0f - dh) * (1.0f - dw),
                                             (1.0f - dh) * dw,
                                             dh * (1.0f - dw), dh * dw};
                    int indices[4] = {h0 * POSITION_SIDE + w0,
                                      h0 * POSITION_SIDE + w1,
                                      h1 * POSITION_SIDE + w0,
                                      h1 * POSITION_SIDE + w1};
                    for (channel = 0; channel < VISION_HIDDEN; channel++) {
                        float sum = 0.0f;
                        for (corner = 0; corner < 4; corner++)
                            sum += table[(size_t)indices[corner] * VISION_HIDDEN +
                                         (size_t)channel] *
                                   coefficients[corner];
                        rows[output_row * VISION_HIDDEN + (size_t)channel] = sum;
                    }
                    output_row++;
                }
            }
        }
    }
    return output_row == (size_t)grid_h * (size_t)grid_w;
}

static void prepare_rope(int grid_h, int grid_w, float *cosines, float *sines) {
    size_t row = 0;
    int block_h, block_w, inner_h, inner_w, axis, index;
    for (block_h = 0; block_h < grid_h / MERGE; block_h++)
        for (block_w = 0; block_w < grid_w / MERGE; block_w++)
            for (inner_h = 0; inner_h < MERGE; inner_h++)
                for (inner_w = 0; inner_w < MERGE; inner_w++) {
                    float positions[2] = {(float)(block_h * MERGE + inner_h),
                                          (float)(block_w * MERGE + inner_w)};
                    for (axis = 0; axis < 2; axis++) {
                        for (index = 0; index < VISION_ROPE_HALF / 2; index++) {
                            float inverse = powf(10000.0f,
                                                 -(float)(index * 2) /
                                                     (float)VISION_ROPE_HALF);
                            float angle = positions[axis] * inverse;
                            size_t offset = row * VISION_ROPE_HALF +
                                            (size_t)axis * (VISION_ROPE_HALF / 2) +
                                            (size_t)index;
                            cosines[offset] = cosf(angle);
                            sines[offset] = sinf(angle);
                        }
                    }
                    row++;
                }
}

static void vision_qkv_rope(const float *qkv, const float *rope_cos,
                            const float *rope_sin, size_t rows, float *query,
                            float *key, float *value) {
    size_t row, head, dimension;
    const size_t inner = (size_t)VISION_HEADS * VISION_HEAD_DIM;
    for (row = 0; row < rows; row++) {
        size_t row_base = row * inner * 3;
        for (head = 0; head < (size_t)VISION_HEADS; head++) {
            size_t q_base = row_base + head * VISION_HEAD_DIM;
            size_t k_base = row_base + inner + head * VISION_HEAD_DIM;
            size_t v_base = row_base + inner * 2 + head * VISION_HEAD_DIM;
            for (dimension = 0; dimension < (size_t)VISION_HEAD_DIM;
                 dimension++) {
                size_t rope_index =
                    row * VISION_ROPE_HALF + dimension % VISION_ROPE_HALF;
                float c = rope_cos[rope_index];
                float s = rope_sin[rope_index];
                size_t pair = dimension < (size_t)VISION_ROPE_HALF
                                  ? dimension + VISION_ROPE_HALF
                                  : dimension - VISION_ROPE_HALF;
                float q0 = qkv[q_base + dimension];
                float k0 = qkv[k_base + dimension];
                float q1 = qkv[q_base + pair];
                float k1 = qkv[k_base + pair];
                float qr = dimension < (size_t)VISION_ROPE_HALF
                               ? q0 * c - q1 * s
                               : q0 * c + q1 * s;
                float kr = dimension < (size_t)VISION_ROPE_HALF
                               ? k0 * c - k1 * s
                               : k0 * c + k1 * s;
                size_t out = (row * VISION_HEADS + head) * VISION_HEAD_DIM +
                             dimension;
                query[out] = qr;
                key[out] = kr;
                value[out] = qkv[v_base + dimension];
            }
        }
    }
}

static void sdpa_host(const float *query, const float *key, const float *value,
                      size_t rows, float *out) {
    const float scale = 1.0f / sqrtf((float)VISION_HEAD_DIM);
    size_t head, row, col, d;
#ifdef _OPENMP
#pragma omp parallel for private(row, col, d)
#endif
    for (head = 0; head < (size_t)VISION_HEADS; head++) {
        float *scores = (float *)malloc(rows * sizeof(float));
        if (!scores) continue;
        for (row = 0; row < rows; row++) {
            const float *q = query + (row * VISION_HEADS + head) * VISION_HEAD_DIM;
            float maxv = -INFINITY, sum = 0.0f;
            for (col = 0; col < rows; col++) {
                const float *k = key + (col * VISION_HEADS + head) * VISION_HEAD_DIM;
                float s = 0.0f;
                for (d = 0; d < (size_t)VISION_HEAD_DIM; d++)
                    s += q[d] * k[d];
                scores[col] = s * scale;
                if (scores[col] > maxv) maxv = scores[col];
            }
            for (col = 0; col < rows; col++) {
                scores[col] = expf(scores[col] - maxv);
                sum += scores[col];
            }
            {
                float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
                float *o = out + (row * VISION_HEADS + head) * VISION_HEAD_DIM;
                for (d = 0; d < (size_t)VISION_HEAD_DIM; d++)
                    o[d] = 0.0f;
                for (col = 0; col < rows; col++) {
                    const float *v =
                        value + (col * VISION_HEADS + head) * VISION_HEAD_DIM;
                    float w = scores[col] * inv;
                    for (d = 0; d < (size_t)VISION_HEAD_DIM; d++)
                        o[d] += w * v[d];
                }
            }
        }
        free(scores);
    }
}

static void sdpa(const float *query, const float *key, const float *value,
                 size_t rows, float *out) {
    const float scale = 1.0f / sqrtf((float)VISION_HEAD_DIM);
    if (vision_want_sdpa_cuda() && rows > 0) {
        char err[256];
        char be[64];
        double ms = 0.0;
        err[0] = '\0';
        be[0] = '\0';
        /* Backend already forced to cudnn for the encode window. */
        if (h3_cuda_sdpa_f16_ex(g_vis_cuda, query, key, value, rows,
                                (size_t)VISION_HEADS, (size_t)VISION_HEAD_DIM,
                                scale, 0, out, &ms, be, sizeof(be), err,
                                sizeof(err))) {
            g_vis_sdpa_ms += ms;
            g_vis_sdpa_calls++;
            if (be[0])
                snprintf(g_vis_sdpa_backend, sizeof(g_vis_sdpa_backend), "%s",
                         be);
            return;
        }
        fprintf(stderr, "vision_sdpa: cuda fallback host (%s)\n",
                err[0] ? err : "unknown");
        fflush(stderr);
    }
    sdpa_host(query, key, value, rows, out);
    if (!g_vis_sdpa_calls)
        snprintf(g_vis_sdpa_backend, sizeof(g_vis_sdpa_backend), "%s", "host");
}

static int run_merger(const vis_merger_w *mw, const float *hidden,
                      size_t patch_rows, int deepstack, float *output,
                      char *error, size_t error_size) {
    size_t merged_rows = patch_rows / (MERGE * MERGE);
    uint32_t norm_width = deepstack ? MERGE_DIM : VISION_HIDDEN;
    uint32_t norm_rows = deepstack ? (uint32_t)merged_rows : (uint32_t)patch_rows;
    float *norm, *fc1;
    int ok = 0;
    if (!mw || !mw->nw || !mw->nb || !mw->fc1w || !mw->fc1b || !mw->fc2w ||
        !mw->fc2b) {
        fail(error, error_size, "Qwen vision merger weights missing");
        return 0;
    }
    norm = (float *)malloc((size_t)patch_rows * VISION_HIDDEN * sizeof(float));
    fc1 = (float *)malloc(merged_rows * MERGE_DIM * sizeof(float));
    if (!norm || !fc1) {
        fail(error, error_size, "cannot allocate Qwen vision merger");
        goto cleanup;
    }
    layer_norm(hidden, mw->nw, mw->nb, norm_rows, norm_width, 1e-6f, norm);
    gemm_nt(norm, mw->fc1w, mw->fc1b, merged_rows, MERGE_DIM, MERGE_DIM, fc1);
    gelu(fc1, merged_rows * MERGE_DIM, 0);
    gemm_nt(fc1, mw->fc2w, mw->fc2b, merged_rows, OUTPUT_WIDTH, MERGE_DIM,
            output);
    ok = 1;
cleanup:
    free(norm);
    free(fc1);
    return ok;
}

int h3_vision_encode_f32(
    const char *vision_weights,
    const float *pixels_t3hw,
    int frames, int height, int width,
    float *merged_out,
    float *deepstack_out[3],
    int *grid_h, int *grid_w, size_t *tokens,
    char *error, size_t error_size) {
    h3_st_header header;
    char vis_prefix[32];
    size_t gh, gw, rows, merged_rows, layer;
    float *patches = NULL, *position = NULL;
    float *rope_cos = NULL, *rope_sin = NULL;
    float *hidden = NULL;
    float *norm = NULL, *qkv = NULL, *query = NULL, *key = NULL, *value = NULL;
    float *heads = NULL, *branch = NULL, *fc1 = NULL;
    struct timespec t_enc0, t_enc1;
    int ok = 0;

    memset(&header, 0, sizeof(header));
    if (grid_h) *grid_h = 0;
    if (grid_w) *grid_w = 0;
    if (tokens) *tokens = 0;
    if (error && error_size) error[0] = '\0';
    if (!vision_weights || !pixels_t3hw || !merged_out || !deepstack_out ||
        !deepstack_out[0] || !deepstack_out[1] || !deepstack_out[2] ||
        !grid_h || !grid_w || !tokens ||
        (frames != 1 && frames != 2) || height < 32 || width < 32 ||
        height % 32 || width % 32) {
        fail(error, error_size, "invalid Qwen vision encoder arguments");
        return 0;
    }
    gh = (size_t)height / PATCH;
    gw = (size_t)width / PATCH;
    rows = gh * gw;
    merged_rows = rows / (MERGE * MERGE);
    if (!h3_st_read_header(vision_weights, &header, error, error_size))
        return 0;
    if (!detect_prefix(&header, vis_prefix, sizeof(vis_prefix))) {
        fail(error, error_size, "visual weights not found (tried visual. and "
                                "model.visual.)");
        goto done;
    }
    if (!vis_w_ensure(&header, vision_weights, vis_prefix, error, error_size)) {
        if (!error || !error[0])
            fail(error, error_size, "cannot cache Qwen vision weights");
        goto done;
    }
    patches = (float *)malloc(rows * PATCH_DIM * sizeof(float));
    position = (float *)malloc(rows * VISION_HIDDEN * sizeof(float));
    rope_cos = (float *)malloc(rows * VISION_ROPE_HALF * sizeof(float));
    rope_sin = (float *)malloc(rows * VISION_ROPE_HALF * sizeof(float));
    hidden = (float *)malloc(rows * VISION_HIDDEN * sizeof(float));
    if (!patches || !position || !rope_cos || !rope_sin || !hidden ||
        !prepare_patch_rows(pixels_t3hw, frames, height, width, patches) ||
        !prepare_position_rows(g_vis_w.pos, (int)gh, (int)gw, position)) {
        if (!error || !error[0])
            fail(error, error_size, "cannot prepare Qwen vision patch inputs");
        goto done;
    }
    prepare_rope((int)gh, (int)gw, rope_cos, rope_sin);
    gemm_nt(patches, g_vis_w.patch_w, g_vis_w.patch_b, rows, VISION_HIDDEN,
            PATCH_DIM, hidden);
    {
        size_t i;
        for (i = 0; i < rows * VISION_HIDDEN; i++)
            hidden[i] += position[i];
    }
    free(patches);
    patches = NULL;
    free(position);
    position = NULL;

    norm = (float *)malloc(rows * VISION_HIDDEN * sizeof(float));
    qkv = (float *)malloc(rows * VISION_HIDDEN * 3 * sizeof(float));
    query = (float *)malloc(rows * VISION_HIDDEN * sizeof(float));
    key = (float *)malloc(rows * VISION_HIDDEN * sizeof(float));
    value = (float *)malloc(rows * VISION_HIDDEN * sizeof(float));
    heads = (float *)malloc(rows * VISION_HIDDEN * sizeof(float));
    branch = (float *)malloc(rows * VISION_HIDDEN * sizeof(float));
    fc1 = (float *)malloc(rows * VISION_INTERMEDIATE * sizeof(float));
    if (!norm || !qkv || !query || !key || !value || !heads || !branch || !fc1) {
        fail(error, error_size, "cannot allocate Qwen vision activations");
        goto done;
    }
    g_vis_sdpa_ms = 0.0;
    g_vis_sdpa_calls = 0;
    snprintf(g_vis_sdpa_backend, sizeof(g_vis_sdpa_backend), "%s",
             vision_want_sdpa_cuda() ? "cudnn?" : "host");
    fprintf(stderr,
            "vision_encode_begin frames=%d %dx%d patch_rows=%zu "
            "merged=%zu layers=%d gemm=%s sdpa=%s\n",
            frames, height, width, rows, merged_rows, VISION_LAYERS,
            vision_want_cuda() ? "cublas" : "host",
            vision_want_sdpa_cuda() ? "cudnn" : "host");
    fflush(stderr);
    {
        char prev_be[64];
        int had_prev = 0;
        int push_cudnn = vision_want_sdpa_cuda();
        if (push_cudnn)
            vision_sdpa_push_cudnn(prev_be, sizeof(prev_be), &had_prev);
        clock_gettime(CLOCK_MONOTONIC, &t_enc0);
        for (layer = 0; layer < (size_t)VISION_LAYERS; layer++) {
            const vis_layer_w *L = &g_vis_w.layer[layer];
            size_t i;
            if (layer == 0 || layer + 1 == (size_t)VISION_LAYERS ||
                layer == 8 || layer == 16 || layer == 24)
                fprintf(stderr, "vision_layer %zu/%d rows=%zu\n", layer + 1,
                        VISION_LAYERS, rows);
            layer_norm(hidden, L->n1w, L->n1b, rows, VISION_HIDDEN, 1e-6f, norm);
            gemm_nt(norm, L->qkv_w, L->qkv_b, rows, VISION_HIDDEN * 3,
                    VISION_HIDDEN, qkv);
            vision_qkv_rope(qkv, rope_cos, rope_sin, rows, query, key, value);
            sdpa(query, key, value, rows, heads);
            gemm_nt(heads, L->ow, L->ob, rows, VISION_HIDDEN, VISION_HIDDEN,
                    branch);
            for (i = 0; i < rows * VISION_HIDDEN; i++)
                hidden[i] += branch[i];
            layer_norm(hidden, L->n2w, L->n2b, rows, VISION_HIDDEN, 1e-6f, norm);
            gemm_nt(norm, L->fc1w, L->fc1b, rows, VISION_INTERMEDIATE,
                    VISION_HIDDEN, fc1);
            gelu(fc1, rows * VISION_INTERMEDIATE, 1);
            gemm_nt(fc1, L->fc2w, L->fc2b, rows, VISION_HIDDEN,
                    VISION_INTERMEDIATE, branch);
            for (i = 0; i < rows * VISION_HIDDEN; i++)
                hidden[i] += branch[i];
            if (layer == 8 || layer == 16 || layer == 24) {
                int index = layer == 8 ? 0 : layer == 16 ? 1 : 2;
                if (!run_merger(&g_vis_w.deepstack[index], hidden, rows, 1,
                                deepstack_out[index], error, error_size)) {
                    if (push_cudnn)
                        vision_sdpa_pop_cudnn(prev_be, had_prev);
                    goto done;
                }
            }
        }
        if (!run_merger(&g_vis_w.merger, hidden, rows, 0, merged_out, error,
                        error_size)) {
            if (push_cudnn)
                vision_sdpa_pop_cudnn(prev_be, had_prev);
            goto done;
        }
        clock_gettime(CLOCK_MONOTONIC, &t_enc1);
        if (push_cudnn)
            vision_sdpa_pop_cudnn(prev_be, had_prev);
    }
    fprintf(stderr,
            "vision_encode: done layers_ms=%.1f gemm=%s sdpa=%s "
            "sdpa_ms=%.1f sdpa_calls=%d\n",
            (t_enc1.tv_sec - t_enc0.tv_sec) * 1000.0 +
                (t_enc1.tv_nsec - t_enc0.tv_nsec) / 1e6,
            vision_want_cuda() ? "cublas" : "host", g_vis_sdpa_backend,
            g_vis_sdpa_ms, g_vis_sdpa_calls);
    fflush(stderr);
    *grid_h = (int)gh;
    *grid_w = (int)gw;
    *tokens = merged_rows;
    ok = 1;
done:
    free(patches);
    free(position);
    free(rope_cos);
    free(rope_sin);
    free(hidden);
    free(norm);
    free(qkv);
    free(query);
    free(key);
    free(value);
    free(heads);
    free(branch);
    free(fc1);
    h3_st_free_header(&header);
    return ok;
}
