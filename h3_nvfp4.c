#include "h3_nvfp4.h"
#include "h3_nvfp4_cuda.h"

#include "h3_w4a8_dequant.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static void fail(char *error, size_t error_size, const char *msg) {
    if (!error || !error_size) return;
    snprintf(error, error_size, "%s", msg ? msg : "h3_nvfp4 error");
}

static void *g_linear_ctx = NULL;
static h3_nvfp4_linear_fn g_linear_fn = NULL;

typedef struct {
    int valid;
    char path[512];
    size_t tokens;
    size_t dim;
    int32_t *ids;
    float *emb;
} nvfp4_embed_cache;

static nvfp4_embed_cache g_embed_cache;

void h3_nvfp4_set_linear_backend(void *ctx, h3_nvfp4_linear_fn fn) {
    g_linear_ctx = ctx;
    g_linear_fn = fn;
}

void h3_nvfp4_embed_cache_clear(void) {
    free(g_embed_cache.ids);
    free(g_embed_cache.emb);
    memset(&g_embed_cache, 0, sizeof(g_embed_cache));
}

int h3_nvfp4_store_copy(const h3_nvfp4_store *store, size_t file_offset,
                        void *dst, size_t nbytes, char *error,
                        size_t error_size) {
    if (!store || (!dst && nbytes != 0)) {
        fail(error, error_size, "invalid store_copy args");
        return 0;
    }
    if (nbytes == 0) return 1;
    if (store->map && store->map_len > 0) {
        if (file_offset + nbytes > store->map_len) {
            fail(error, error_size, "store_copy past mmap end");
            return 0;
        }
        memcpy(dst, store->map + file_offset, nbytes);
        return 1;
    }
    {
        int fd = open(store->path, O_RDONLY);
        size_t done = 0;
        if (fd < 0) {
            fail(error, error_size, "store_copy open failed");
            return 0;
        }
        while (done < nbytes) {
            ssize_t n = pread(fd, (unsigned char *)dst + done, nbytes - done,
                              (off_t)(file_offset + done));
            if (n <= 0) {
                close(fd);
                fail(error, error_size, "store_copy pread failed");
                return 0;
            }
            done += (size_t)n;
        }
        close(fd);
        return 1;
    }
}

static uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

/* OCP FP4 E2M1: sign | exp(2) | mant(1) — table for nibbles 0..15. */
float h3_fp4_e2m1_to_f32(uint8_t nibble) {
    static const float table[16] = {
        0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    return table[nibble & 0xF];
}

static size_t nvfp4_ceil_div(size_t a, size_t b) {
    return b ? (a + b - 1) / b : 0;
}

/* Comfy-kitchen / cuBLAS NVFP4 block-scale unswizzle (from_blocked).
 * On-disk weight_scale is tiled; dequant needs logical [N, K/gs]. */
int h3_nvfp4_unswizzle_scales_u8(const uint8_t *blocked, size_t num_rows,
                                 size_t num_cols, uint8_t *out, char *error,
                                 size_t error_size) {
    size_t n_row_blocks, n_col_blocks, padded_rows, padded_cols;
    size_t rb, cb, i32, j16;
    if (!blocked || !out || num_rows == 0 || num_cols == 0) {
        fail(error, error_size, "invalid nvfp4 unswizzle args");
        return 0;
    }
    n_row_blocks = nvfp4_ceil_div(num_rows, 128);
    n_col_blocks = nvfp4_ceil_div(num_cols, 4);
    padded_rows = n_row_blocks * 128;
    padded_cols = n_col_blocks * 4;
    if (padded_rows * padded_cols != num_rows * num_cols &&
        (padded_rows != num_rows || padded_cols != num_cols)) {
        fail(error, error_size, "nvfp4 weight_scale not 128×4 padded");
        return 0;
    }
    /* Mirror comfy_kitchen.float_utils.from_blocked on (padded_rows,padded_cols).
     * blocked is row-major of the swizzled 2D matrix. */
    for (rb = 0; rb < n_row_blocks; rb++) {
        for (cb = 0; cb < n_col_blocks; cb++) {
            size_t tile = rb * n_col_blocks + cb;
            for (i32 = 0; i32 < 32; i32++) {
                for (j16 = 0; j16 < 16; j16++) {
                    /* storage viewed as (-1, 32, 16) then (-1, 32, 4, 4) */
                    size_t src = tile * 32 * 16 + i32 * 16 + j16;
                    size_t a = j16 / 4; /* was dim after transpose */
                    size_t b = i32;
                    size_t c = j16 % 4;
                    /* transpose(1,2) inverse: after[t,b,a,c] = before[t,a,b,c]
                     * before layout (-1, 4, 32, 4) → (nrb,ncb,128,4) with
                     * 128 = a*32+b */
                    size_t row128 = a * 32 + b;
                    size_t dst_r = rb * 128 + row128;
                    size_t dst_c = cb * 4 + c;
                    if (dst_r < num_rows && dst_c < num_cols)
                        out[dst_r * num_cols + dst_c] = blocked[src];
                }
            }
        }
    }
    return 1;
}

/* NVFP4 pack: hi_first (Comfy default) — high nibble → even K, low → odd. */
static void nvfp4_unpack_byte_hi_first(uint8_t byte, float s0, float s1,
                                       float *dst0, float *dst1) {
    *dst0 = h3_fp4_e2m1_to_f32((byte >> 4) & 0xF) * s0;
    *dst1 = h3_fp4_e2m1_to_f32(byte & 0xF) * s1;
}

int h3_nvfp4_store_open(const char *path, h3_nvfp4_store *store, char *error,
                        size_t error_size) {
    struct stat st;
    if (!path || !store) {
        fail(error, error_size, "invalid nvfp4 open args");
        return 0;
    }
    memset(store, 0, sizeof(*store));
    store->map_fd = -1;
    snprintf(store->path, sizeof(store->path), "%s", path);
    if (!h3_st_read_header(path, &store->header, error, error_size)) return 0;
    h3_st_quant_info_from_header(&store->header, &store->quant);
    /* Best-effort mmap: speeds TE linear/embed reads; falls back to pread. */
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
    return 1;
}

void h3_nvfp4_store_free(h3_nvfp4_store *store) {
    if (!store) return;
    /* TE device cache is keyed by safetensors offsets in this store. */
    h3_nvfp4_cuda_cache_clear();
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

int h3_nvfp4_find_linear(const h3_nvfp4_store *store, const char *base,
                         h3_nvfp4_linear *out, char *error, size_t error_size) {
    char wname[320], sname[320], s2name[320], pqname[320];
    const h3_st_tensor *w, *s, *s2, *pq;
    if (!store || !base || !out) {
        fail(error, error_size, "invalid find_linear args");
        return 0;
    }
    memset(out, 0, sizeof(*out));
    snprintf(out->base_name, sizeof(out->base_name), "%s", base);
    snprintf(wname, sizeof(wname), "%s.weight", base);
    snprintf(sname, sizeof(sname), "%s.weight_scale", base);
    snprintf(s2name, sizeof(s2name), "%s.weight_scale_2", base);
    snprintf(pqname, sizeof(pqname), "%s.pre_quant_scale", base);
    w = h3_st_find(&store->header, wname);
    s = h3_st_find(&store->header, sname);
    s2 = h3_st_find(&store->header, s2name);
    pq = h3_st_find(&store->header, pqname);
    if (!w || !s || !s2) {
        fail(error, error_size, "missing weight / weight_scale / weight_scale_2");
        return 0;
    }
    if (w->dtype != H3_DTYPE_U8 || s->dtype != H3_DTYPE_F8_E4M3 ||
        s2->dtype != H3_DTYPE_F32 || w->ndim != 2 || s->ndim != 2) {
        fail(error, error_size, "unexpected nvfp4 tensor dtypes/ranks");
        return 0;
    }
    out->weight = w;
    out->scale = s;
    out->scale_2 = s2;
    out->n = (size_t)w->shape[0];
    out->k = (size_t)w->shape[1] * 2;
    out->groups = (size_t)s->shape[1];
    if (out->n == 0 || out->k == 0 || out->groups == 0 ||
        (size_t)s->shape[0] != out->n || (out->k % out->groups) != 0) {
        fail(error, error_size, "nvfp4 shape inconsistency");
        return 0;
    }
    out->group_size = out->k / out->groups;
    /* AWQ smooth: present on o_proj / down_proj (Comfy ModelOpt). */
    if (pq) {
        if (pq->dtype != H3_DTYPE_BF16 || pq->ndim != 1 ||
            (size_t)pq->shape[0] != out->k) {
            fail(error, error_size, "pre_quant_scale dtype/shape mismatch");
            return 0;
        }
        out->pre_quant_scale = pq;
    }
    return 1;
}

int h3_nvfp4_find_embed(const h3_nvfp4_store *store, h3_nvfp4_embed *out,
                        char *error, size_t error_size) {
    const h3_st_tensor *w, *s;
    if (!store || !out) {
        fail(error, error_size, "invalid find_embed args");
        return 0;
    }
    memset(out, 0, sizeof(*out));
    w = h3_st_find(&store->header, "model.embed_tokens.weight");
    s = h3_st_find(&store->header, "model.embed_tokens.weight_scale");
    if (!w || !s) {
        fail(error, error_size, "missing model.embed_tokens weight/scale");
        return 0;
    }
    if (w->dtype != H3_DTYPE_I8 || s->dtype != H3_DTYPE_F32 || w->ndim != 2) {
        fail(error, error_size, "embed_tokens unexpected dtype/rank");
        return 0;
    }
    out->weight = w;
    out->scale = s;
    out->vocab = (size_t)w->shape[0];
    out->dim = (size_t)w->shape[1];
    if (out->vocab == 0 || out->dim == 0) {
        fail(error, error_size, "embed_tokens empty");
        return 0;
    }
    if (h3_st_tensor_elements(s) != out->vocab &&
        !(s->ndim == 2 && (size_t)s->shape[0] == out->vocab &&
          (size_t)s->shape[1] == 1)) {
        fail(error, error_size, "embed_tokens scale shape mismatch");
        return 0;
    }
    return 1;
}

float *h3_nvfp4_dequant_rows_f32(const h3_nvfp4_store *store,
                                 const h3_nvfp4_linear *lin, size_t max_rows,
                                 size_t *out_rows, char *error,
                                 size_t error_size) {
    size_t rows, k_half, i, j;
    uint8_t *packed = NULL;
    uint8_t *scales_f8 = NULL;
    float scale2 = 0.0f;
    float *out = NULL;
    if (!store || !lin || !lin->weight || !lin->scale || !lin->scale_2) {
        fail(error, error_size, "invalid nvfp4 dequant args");
        return NULL;
    }
    rows = lin->n;
    if (max_rows && max_rows < rows) rows = max_rows;
    k_half = lin->k / 2;
    packed = (uint8_t *)malloc(rows * k_half);
    scales_f8 = (uint8_t *)malloc(rows * lin->groups);
    out = (float *)malloc(rows * lin->k * sizeof(float));
    if (!packed || !scales_f8 || !out) {
        fail(error, error_size, "OOM nvfp4 dequant");
        goto fail;
    }
    if (!h3_nvfp4_store_copy(store, (size_t)lin->weight->file_offset, packed,
                             rows * k_half, error, error_size) ||
        !h3_nvfp4_store_copy(store, (size_t)lin->scale->file_offset, scales_f8,
                             rows * lin->groups, error, error_size) ||
        !h3_nvfp4_store_copy(store, (size_t)lin->scale_2->file_offset, &scale2,
                             sizeof(float), error, error_size))
        goto fail;

    {
        uint8_t *logical = (uint8_t *)malloc(rows * lin->groups);
        if (!logical) {
            fail(error, error_size, "OOM nvfp4 scale unswizzle");
            goto fail;
        }
        if (!h3_nvfp4_unswizzle_scales_u8(scales_f8, rows, lin->groups, logical,
                                          error, error_size)) {
            free(logical);
            goto fail;
        }
        free(scales_f8);
        scales_f8 = logical;
    }

    for (i = 0; i < rows; i++) {
        const uint8_t *prow = packed + i * k_half;
        const uint8_t *srow = scales_f8 + i * lin->groups;
        float *dst = out + i * lin->k;
        for (j = 0; j < k_half; j++) {
            uint8_t byte = prow[j];
            size_t col0 = j * 2;
            size_t col1 = col0 + 1;
            size_t g0 = col0 / lin->group_size;
            size_t g1 = col1 / lin->group_size;
            float s0 = h3_f8e4m3fn_to_f32(srow[g0]) * scale2;
            float s1 = h3_f8e4m3fn_to_f32(srow[g1]) * scale2;
            nvfp4_unpack_byte_hi_first(byte, s0, s1, &dst[col0], &dst[col1]);
        }
    }
    if (out_rows) *out_rows = rows;
    free(packed);
    free(scales_f8);
    return out;
fail:
    free(packed);
    free(scales_f8);
    free(out);
    return NULL;
}

float *h3_nvfp4_embed_lookup_f32(const h3_nvfp4_store *store,
                                 const h3_nvfp4_embed *emb,
                                 const int32_t *token_ids, size_t tokens,
                                 char *error, size_t error_size) {
    float *out = NULL;
    size_t t, d;
    int cache_hit = 0;
    if (!store || !emb || !emb->weight || !emb->scale || !token_ids ||
        tokens == 0) {
        fail(error, error_size, "invalid embed_lookup args");
        return NULL;
    }
    /* Process-local cache: same path+ids → clone cached embedding. */
    if (g_embed_cache.valid && g_embed_cache.tokens == tokens &&
        g_embed_cache.dim == emb->dim &&
        strcmp(g_embed_cache.path, store->path) == 0 && g_embed_cache.ids &&
        g_embed_cache.emb &&
        memcmp(g_embed_cache.ids, token_ids, tokens * sizeof(int32_t)) == 0) {
        out = (float *)malloc(tokens * emb->dim * sizeof(float));
        if (!out) {
            fail(error, error_size, "OOM embed_lookup cache clone");
            return NULL;
        }
        memcpy(out, g_embed_cache.emb, tokens * emb->dim * sizeof(float));
        return out;
    }

    out = (float *)malloc(tokens * emb->dim * sizeof(float));
    if (!out) {
        fail(error, error_size, "OOM embed_lookup");
        return NULL;
    }
    for (t = 0; t < tokens; t++) {
        int32_t id = token_ids[t];
        int8_t *row_i8 = NULL;
        float scale = 0.0f;
        size_t w_off, s_off;
        if (id < 0 || (size_t)id >= emb->vocab) {
            free(out);
            fail(error, error_size, "token id out of vocab");
            return NULL;
        }
        row_i8 = (int8_t *)malloc(emb->dim);
        if (!row_i8) {
            free(out);
            fail(error, error_size, "OOM embed row");
            return NULL;
        }
        w_off = (size_t)emb->weight->file_offset + (size_t)id * emb->dim;
        s_off = (size_t)emb->scale->file_offset + (size_t)id * sizeof(float);
        if (!h3_nvfp4_store_copy(store, w_off, row_i8, emb->dim, error,
                                 error_size) ||
            !h3_nvfp4_store_copy(store, s_off, &scale, sizeof(float), error,
                                 error_size)) {
            free(row_i8);
            free(out);
            return NULL;
        }
        for (d = 0; d < emb->dim; d++)
            out[t * emb->dim + d] = (float)row_i8[d] * scale;
        free(row_i8);
    }

    /* Refresh cache (best-effort). */
    {
        int32_t *ids = (int32_t *)malloc(tokens * sizeof(int32_t));
        float *emb_c = (float *)malloc(tokens * emb->dim * sizeof(float));
        if (ids && emb_c) {
            h3_nvfp4_embed_cache_clear();
            memcpy(ids, token_ids, tokens * sizeof(int32_t));
            memcpy(emb_c, out, tokens * emb->dim * sizeof(float));
            snprintf(g_embed_cache.path, sizeof(g_embed_cache.path), "%s",
                     store->path);
            g_embed_cache.tokens = tokens;
            g_embed_cache.dim = emb->dim;
            g_embed_cache.ids = ids;
            g_embed_cache.emb = emb_c;
            g_embed_cache.valid = 1;
        } else {
            free(ids);
            free(emb_c);
        }
    }
    (void)cache_hit;
    return out;
}

float *h3_nvfp4_synth_condition_f32(size_t tokens, uint64_t seed, char *error,
                                    size_t error_size) {
    float *out;
    size_t i, n;
    if (tokens == 0) {
        fail(error, error_size, "synth_condition tokens=0");
        return NULL;
    }
    n = tokens * (size_t)H3_NVFP4_TEXT_DIM;
    out = (float *)malloc(n * sizeof(float));
    if (!out) {
        fail(error, error_size, "OOM synth_condition");
        return NULL;
    }
    for (i = 0; i < n; i++) {
        uint64_t h =
            mix64(seed ^ (uint64_t)(i + 1) * 0x9e3779b97f4a7c15ULL);
        out[i] = (float)(int32_t)(h >> 32) * (1.0f / 2147483648.0f) * 0.02f;
    }
    return out;
}

int h3_nvfp4_gemm_f32(const float *x, const float *w, size_t m, size_t n,
                      size_t k, float *y, char *error, size_t error_size) {
    size_t r, o, i;
    if (!x || !w || !y || m == 0 || n == 0 || k == 0) {
        fail(error, error_size, "invalid nvfp4 gemm args");
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

float *h3_nvfp4_make_condition_f32(const h3_nvfp4_store *store,
                                   const int32_t *token_ids, size_t tokens,
                                   uint64_t synth_seed,
                                   const char *apply_linear_base,
                                   size_t *out_tokens, char *error,
                                   size_t error_size) {
    float *emb = NULL;
    float *out = NULL;
    h3_nvfp4_embed e;
    if (tokens == 0) {
        fail(error, error_size, "make_condition tokens=0");
        return NULL;
    }
    if (store && token_ids) {
        memset(&e, 0, sizeof(e));
        if (!h3_nvfp4_find_embed(store, &e, error, error_size)) return NULL;
        if (e.dim != (size_t)H3_NVFP4_TEXT_DIM) {
            fail(error, error_size, "embed dim != TEXT_DIM 5120");
            return NULL;
        }
        emb = h3_nvfp4_embed_lookup_f32(store, &e, token_ids, tokens, error,
                                        error_size);
    } else {
        emb = h3_nvfp4_synth_condition_f32(tokens, synth_seed, error,
                                           error_size);
    }
    if (!emb) return NULL;

    if (!apply_linear_base || !store) {
        if (out_tokens) *out_tokens = tokens;
        return emb;
    }

    /* Optional smoke: project through a truncated NVFP4 linear whose K matches
     * TEXT_DIM (e.g. o_proj / down_proj). Keeps [tokens, TEXT_DIM]. */
    {
        h3_nvfp4_linear lin;
        float *w = NULL;
        size_t rows = 0;
        size_t use_n;
        memset(&lin, 0, sizeof(lin));
        if (!h3_nvfp4_find_linear(store, apply_linear_base, &lin, error,
                                  error_size)) {
            free(emb);
            return NULL;
        }
        if (lin.k != (size_t)H3_NVFP4_TEXT_DIM) {
            free(emb);
            fail(error, error_size, "apply_linear K != TEXT_DIM");
            return NULL;
        }
        use_n = lin.n;
        if (use_n > (size_t)H3_NVFP4_TEXT_DIM) use_n = (size_t)H3_NVFP4_TEXT_DIM;
        w = h3_nvfp4_dequant_rows_f32(store, &lin, use_n, &rows, error,
                                      error_size);
        if (!w) {
            free(emb);
            return NULL;
        }
        out = (float *)malloc(tokens * (size_t)H3_NVFP4_TEXT_DIM * sizeof(float));
        if (!out) {
            free(w);
            free(emb);
            fail(error, error_size, "OOM make_condition out");
            return NULL;
        }
        memset(out, 0, tokens * (size_t)H3_NVFP4_TEXT_DIM * sizeof(float));
        if (!h3_nvfp4_gemm_f32(emb, w, tokens, rows, lin.k, out, error,
                               error_size)) {
            free(w);
            free(emb);
            free(out);
            return NULL;
        }
        free(w);
        free(emb);
        if (out_tokens) *out_tokens = tokens;
        return out;
    }
}

#include <math.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static float nvfp4_bf16_bits_to_f32(uint16_t bits) {
    union {
        uint32_t u;
        float f;
    } v;
    v.u = ((uint32_t)bits) << 16;
    return v.f;
}

static double nvfp4_elapsed_ms(clock_t t0, clock_t t1) {
    return 1000.0 * (double)(t1 - t0) / (double)CLOCKS_PER_SEC;
}

static float *nvfp4_load_bf16_f32(const h3_nvfp4_store *store, const char *name,
                                  size_t expect, char *error,
                                  size_t error_size) {
    const h3_st_tensor *t;
    uint16_t *raw = NULL;
    float *out = NULL;
    size_t i, n;
    t = h3_st_find(&store->header, name);
    if (!t || t->dtype != H3_DTYPE_BF16) {
        fail(error, error_size, "missing BF16 tensor");
        return NULL;
    }
    n = (size_t)h3_st_tensor_elements(t);
    if (expect && n != expect) {
        fail(error, error_size, "BF16 tensor size mismatch");
        return NULL;
    }
    raw = (uint16_t *)malloc(n * sizeof(uint16_t));
    out = (float *)malloc(n * sizeof(float));
    if (!raw || !out) {
        free(raw);
        free(out);
        fail(error, error_size, "OOM BF16 load");
        return NULL;
    }
    if (!h3_nvfp4_store_copy(store, (size_t)t->file_offset, raw,
                             n * sizeof(uint16_t), error, error_size)) {
        free(raw);
        free(out);
        return NULL;
    }
    for (i = 0; i < n; i++) out[i] = nvfp4_bf16_bits_to_f32(raw[i]);
    free(raw);
    return out;
}

static void nvfp4_rms_norm(const float *x, const float *w, size_t rows,
                           size_t dim, float eps, float *out) {
    size_t r, d;
    for (r = 0; r < rows; r++) {
        const float *xr = x + r * dim;
        float *yr = out + r * dim;
        float mean = 0.0f, inv;
        for (d = 0; d < dim; d++) mean += xr[d] * xr[d];
        inv = 1.0f / sqrtf(mean / (float)dim + eps);
        for (d = 0; d < dim; d++) yr[d] = xr[d] * inv * w[d];
    }
}

static void nvfp4_head_rms(float *x, const float *w, size_t tokens,
                           size_t heads, size_t head_dim, float eps) {
    size_t t, h, d;
    for (t = 0; t < tokens; t++) {
        for (h = 0; h < heads; h++) {
            float *xr = x + (t * heads + h) * head_dim;
            float mean = 0.0f, inv;
            for (d = 0; d < head_dim; d++) mean += xr[d] * xr[d];
            inv = 1.0f / sqrtf(mean / (float)head_dim + eps);
            for (d = 0; d < head_dim; d++) xr[d] = xr[d] * inv * w[d];
        }
    }
}

static void nvfp4_rope(float *q, float *k, size_t tokens, size_t q_heads,
                       size_t kv_heads, size_t head_dim, float theta) {
    size_t t, h, i;
    const size_t half = head_dim / 2;
    for (t = 0; t < tokens; t++) {
        for (h = 0; h < q_heads; h++) {
            float *xq = q + (t * q_heads + h) * head_dim;
            for (i = 0; i < half; i++) {
                float freq = 1.0f / powf(theta, (float)(2 * i) / (float)head_dim);
                float ang = (float)t * freq;
                float c = cosf(ang), s = sinf(ang);
                float a = xq[i], b = xq[i + half];
                xq[i] = a * c - b * s;
                xq[i + half] = a * s + b * c;
            }
        }
        for (h = 0; h < kv_heads; h++) {
            float *xk = k + (t * kv_heads + h) * head_dim;
            for (i = 0; i < half; i++) {
                float freq = 1.0f / powf(theta, (float)(2 * i) / (float)head_dim);
                float ang = (float)t * freq;
                float c = cosf(ang), s = sinf(ang);
                float a = xk[i], b = xk[i + half];
                xk[i] = a * c - b * s;
                xk[i + half] = a * s + b * c;
            }
        }
    }
}

/* Qwen3-VL mRoPE: first 60 of 64 rotary pairs split across 3 axes
 * (Metal h3_text_encoder.c). pos is axis-major [3, tokens]. */
static void nvfp4_rope_mrope(float *q, float *k, size_t tokens, size_t q_heads,
                             size_t kv_heads, size_t head_dim, float theta,
                             const uint32_t *pos) {
    size_t t, h, i;
    const size_t half = head_dim / 2;
    for (t = 0; t < tokens; t++) {
        for (h = 0; h < q_heads; h++) {
            float *xq = q + (t * q_heads + h) * head_dim;
            for (i = 0; i < half; i++) {
                size_t axis = 0;
                float freq, ang, c, s, a, b;
                if (i < 60 && i % 3 == 1) axis = 1;
                else if (i < 60 && i % 3 == 2) axis = 2;
                freq = 1.0f / powf(theta, (float)(2 * i) / (float)head_dim);
                ang = (float)pos[axis * tokens + t] * freq;
                c = cosf(ang);
                s = sinf(ang);
                a = xq[i];
                b = xq[i + half];
                xq[i] = a * c - b * s;
                xq[i + half] = a * s + b * c;
            }
        }
        for (h = 0; h < kv_heads; h++) {
            float *xk = k + (t * kv_heads + h) * head_dim;
            for (i = 0; i < half; i++) {
                size_t axis = 0;
                float freq, ang, c, s, a, b;
                if (i < 60 && i % 3 == 1) axis = 1;
                else if (i < 60 && i % 3 == 2) axis = 2;
                freq = 1.0f / powf(theta, (float)(2 * i) / (float)head_dim);
                ang = (float)pos[axis * tokens + t] * freq;
                c = cosf(ang);
                s = sinf(ang);
                a = xk[i];
                b = xk[i + half];
                xk[i] = a * c - b * s;
                xk[i + half] = a * s + b * c;
            }
        }
    }
}

static int nvfp4_build_mrope_pos(const h3_cuda_vision_span *spans,
                                 size_t span_count, size_t sequence,
                                 uint32_t *positions, char *error,
                                 size_t error_size) {
    int64_t offset = 0;
    size_t image, axis, index;
    for (image = 0; image < span_count; image++) {
        size_t vis_start = spans[image].start;
        size_t pad_n, pad_start, pad_end, merged_h, merged_w, length_max, cursor;
        int64_t base, next;
        if (spans[image].tokens < 2 || vis_start >= sequence ||
            vis_start + spans[image].tokens > sequence ||
            spans[image].grid_h < 2 || spans[image].grid_w < 2 ||
            (spans[image].grid_h & 1) || (spans[image].grid_w & 1)) {
            fail(error, error_size, "invalid Qwen vision span geometry");
            return 0;
        }
        pad_n = spans[image].tokens - 2;
        pad_start = vis_start + 1;
        pad_end = pad_start + pad_n;
        /* Metal build_positions: merged = patch grid / MERGE(2). */
        merged_h = (size_t)spans[image].grid_h / 2u;
        merged_w = (size_t)spans[image].grid_w / 2u;
        if (merged_w && merged_h > SIZE_MAX / merged_w) {
            fail(error, error_size, "Qwen merged grid overflow");
            return 0;
        }
        if (merged_h * merged_w != pad_n) {
            fail(error, error_size, "Qwen merged grid does not match token span");
            return 0;
        }
        if (image == 0) {
            for (axis = 0; axis < 3; axis++)
                for (index = 0; index < pad_start; index++)
                    positions[axis * sequence + index] = (uint32_t)index;
        }
        length_max = merged_h > merged_w ? merged_h : merged_w;
        base = (int64_t)pad_start + offset;
        next = (int64_t)pad_start + (int64_t)length_max + offset;
        if (base < 0 || next < 0) {
            fail(error, error_size, "multimodal mRoPE position overflow");
            return 0;
        }
        for (index = pad_start; index < pad_end; index++)
            positions[index] = (uint32_t)base;
        cursor = pad_start;
        {
            size_t row, column;
            for (row = 0; row < merged_h; row++) {
                for (column = 0; column < merged_w; column++) {
                    positions[sequence + cursor] =
                        (uint32_t)(base + (int64_t)row);
                    positions[2 * sequence + cursor] =
                        (uint32_t)(base + (int64_t)column);
                    cursor++;
                }
            }
        }
        if (cursor != pad_end) {
            fail(error, error_size, "Qwen merged grid does not match token span");
            return 0;
        }
        for (axis = 0; axis < 3; axis++)
            for (index = pad_end; index < sequence; index++)
                positions[axis * sequence + index] =
                    (uint32_t)(next + (int64_t)(index - pad_end));
        offset += (int64_t)length_max - (int64_t)pad_n;
    }
    return 1;
}

static int nvfp4_inject_vision(float *hidden, size_t tokens,
                               const h3_cuda_vision_span *spans,
                               size_t span_count, const float *src,
                               char *error, size_t error_size) {
    size_t s, off = 0;
    if (!src) {
        fail(error, error_size, "missing vision embeddings");
        return 0;
    }
    for (s = 0; s < span_count; s++) {
        size_t pad_start, pad_n;
        if (spans[s].tokens < 2 || spans[s].start >= tokens ||
            spans[s].start + spans[s].tokens > tokens) {
            fail(error, error_size, "invalid Qwen vision presentation span");
            return 0;
        }
        pad_start = spans[s].start + 1;
        pad_n = spans[s].tokens - 2;
        memcpy(hidden + pad_start * (size_t)H3_NVFP4_TEXT_DIM,
               src + off * (size_t)H3_NVFP4_TEXT_DIM,
               pad_n * (size_t)H3_NVFP4_TEXT_DIM * sizeof(float));
        off += pad_n;
    }
    return 1;
}

static int nvfp4_add_vision(float *hidden, size_t tokens,
                            const h3_cuda_vision_span *spans, size_t span_count,
                            const float *src, char *error, size_t error_size) {
    size_t s, off = 0, i;
    if (!src) {
        fail(error, error_size, "missing vision deepstack");
        return 0;
    }
    for (s = 0; s < span_count; s++) {
        size_t pad_start, pad_n, n;
        if (spans[s].tokens < 2 || spans[s].start >= tokens ||
            spans[s].start + spans[s].tokens > tokens) {
            fail(error, error_size, "invalid Qwen vision presentation span");
            return 0;
        }
        pad_start = spans[s].start + 1;
        pad_n = spans[s].tokens - 2;
        n = pad_n * (size_t)H3_NVFP4_TEXT_DIM;
        for (i = 0; i < n; i++)
            hidden[pad_start * (size_t)H3_NVFP4_TEXT_DIM + i] +=
                src[off * (size_t)H3_NVFP4_TEXT_DIM + i];
        off += pad_n;
    }
    return 1;
}

/* Causal GQA: Q[T,Hq,D] K/V[T,Hkv,D] → O[T,Hq,D] */
static int nvfp4_gqa_causal(const float *q, const float *k, const float *v,
                            size_t tokens, size_t q_heads, size_t kv_heads,
                            size_t head_dim, float *out, char *error,
                            size_t error_size) {
    const size_t rep = q_heads / kv_heads;
    const float scale = 1.0f / sqrtf((float)head_dim);
    int ok = 1;
    if (q_heads % kv_heads != 0) {
        fail(error, error_size, "GQA head mismatch");
        return 0;
    }
    {
        const char *force_host = getenv("H3_NVFP4_HOST_GQA");
        int want_host = force_host && force_host[0] == '1';
        if (!want_host && h3_nvfp4_cuda_enabled()) {
            if (h3_nvfp4_cuda_gqa_causal(q, k, v, tokens, q_heads, kv_heads,
                                         head_dim, out, error, error_size))
                return 1;
            return 0; /* keep GPU error; do not fall back to O(T^2) host */
        }
    }
    /* Host GQA is O(T^2 * heads * dim). Loop indices must be private —
     * sharing h/s/d across OpenMP threads races and yields NaN/Inf. */
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        float *scores = (float *)malloc(tokens * sizeof(float));
        if (!scores) {
#ifdef _OPENMP
#pragma omp critical
#endif
            {
                if (ok) {
                    fail(error, error_size, "OOM GQA scores");
                    ok = 0;
                }
            }
        } else {
            size_t t, h, s, d;
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 8)
#endif
            for (t = 0; t < tokens; t++) {
                if (!ok) continue;
                for (h = 0; h < q_heads; h++) {
                    size_t kv_h = h / rep;
                    const float *qq = q + (t * q_heads + h) * head_dim;
                    float *oo = out + (t * q_heads + h) * head_dim;
                    float maxv = -1e30f, sum = 0.0f;
                    for (s = 0; s <= t; s++) {
                        const float *kk = k + (s * kv_heads + kv_h) * head_dim;
                        float dot = 0.0f;
                        for (d = 0; d < head_dim; d++) dot += qq[d] * kk[d];
                        scores[s] = dot * scale;
                        if (scores[s] > maxv) maxv = scores[s];
                    }
                    for (s = 0; s <= t; s++) {
                        scores[s] = expf(scores[s] - maxv);
                        sum += scores[s];
                    }
                    for (d = 0; d < head_dim; d++) oo[d] = 0.0f;
                    for (s = 0; s <= t; s++) {
                        const float *vv = v + (s * kv_heads + kv_h) * head_dim;
                        float a = scores[s] / sum;
                        for (d = 0; d < head_dim; d++) oo[d] += a * vv[d];
                    }
                }
            }
            free(scores);
        }
    }
    return ok;
}

static void nvfp4_silu_mul(const float *gate, const float *up, size_t n,
                           float *out) {
    size_t i;
    for (i = 0; i < n; i++) {
        float g = gate[i];
        float sig = 1.0f / (1.0f + expf(-g));
        out[i] = (g * sig) * up[i];
    }
}

enum { H3_NVFP4_LINEAR_CHUNK = 512 };

/* Comfy ModelOpt AWQ: x = x * pre_quant_scale (per-K) before NVFP4 matmul. */
static float *nvfp4_apply_pre_quant(const h3_nvfp4_store *store,
                                    const h3_nvfp4_linear *lin, const float *x,
                                    size_t m, char *error, size_t error_size) {
    float *scale = NULL;
    float *out = NULL;
    uint16_t *raw = NULL;
    size_t r, d, n;
    if (!lin->pre_quant_scale) return NULL;
    n = lin->k;
    raw = (uint16_t *)malloc(n * sizeof(uint16_t));
    scale = (float *)malloc(n * sizeof(float));
    out = (float *)malloc(m * n * sizeof(float));
    if (!raw || !scale || !out) {
        free(raw);
        free(scale);
        free(out);
        fail(error, error_size, "OOM pre_quant_scale");
        return NULL;
    }
    if (!h3_nvfp4_store_copy(store, (size_t)lin->pre_quant_scale->file_offset,
                             raw, n * sizeof(uint16_t), error, error_size)) {
        free(raw);
        free(scale);
        free(out);
        return NULL;
    }
    for (d = 0; d < n; d++) scale[d] = nvfp4_bf16_bits_to_f32(raw[d]);
    free(raw);
    for (r = 0; r < m; r++) {
        const float *xr = x + r * n;
        float *yr = out + r * n;
        for (d = 0; d < n; d++) yr[d] = xr[d] * scale[d];
    }
    free(scale);
    return out;
}

int h3_nvfp4_linear_f32(const h3_nvfp4_store *store, const h3_nvfp4_linear *lin,
                        const float *x, size_t m, float *y, char *error,
                        size_t error_size) {
    size_t n0, chunk;
    float *wchunk = NULL;
    uint8_t *packed = NULL;
    uint8_t *scales_f8 = NULL;
    float scale2 = 0.0f;
    size_t k_half;
    float *x_scaled = NULL;
    const float *xin = x;
    int ok = 0;

    if (!store || !lin || !x || !y || m == 0) {
        fail(error, error_size, "invalid nvfp4_linear args");
        return 0;
    }
    /* CUDA backend applies AWQ pre_quant on-device (same math, no host
     * m*k scratch or extra x H2D of the scaled buffer). */
    if (g_linear_fn) {
        ok = g_linear_fn(g_linear_ctx, store, lin, x, m, y, error, error_size);
        return ok;
    }
    if (lin->pre_quant_scale) {
        x_scaled = nvfp4_apply_pre_quant(store, lin, x, m, error, error_size);
        if (!x_scaled) return 0;
        xin = x_scaled;
    }

    memset(y, 0, m * lin->n * sizeof(float));
    k_half = lin->k / 2;
    if (!h3_nvfp4_store_copy(store, (size_t)lin->scale_2->file_offset, &scale2,
                             sizeof(float), error, error_size)) {
        free(x_scaled);
        return 0;
    }

    /* Prefetch full packed+scales once (mmap/pread), then chunk GEMM. */
    packed = (uint8_t *)malloc(lin->n * k_half);
    scales_f8 = (uint8_t *)malloc(lin->n * lin->groups);
    if (!packed || !scales_f8) {
        free(packed);
        free(scales_f8);
        free(x_scaled);
        fail(error, error_size, "OOM linear packed");
        return 0;
    }
    if (!h3_nvfp4_store_copy(store, (size_t)lin->weight->file_offset, packed,
                             lin->n * k_half, error, error_size) ||
        !h3_nvfp4_store_copy(store, (size_t)lin->scale->file_offset, scales_f8,
                             lin->n * lin->groups, error, error_size)) {
        free(packed);
        free(scales_f8);
        free(x_scaled);
        return 0;
    }
    {
        uint8_t *logical = (uint8_t *)malloc(lin->n * lin->groups);
        if (!logical) {
            free(packed);
            free(scales_f8);
            free(x_scaled);
            fail(error, error_size, "OOM nvfp4 scale unswizzle");
            return 0;
        }
        if (!h3_nvfp4_unswizzle_scales_u8(scales_f8, lin->n, lin->groups, logical,
                                          error, error_size)) {
            free(logical);
            free(packed);
            free(scales_f8);
            free(x_scaled);
            return 0;
        }
        free(scales_f8);
        scales_f8 = logical;
    }

    for (n0 = 0; n0 < lin->n; n0 += (size_t)H3_NVFP4_LINEAR_CHUNK) {
        size_t i, j, r, o, i2;
        chunk = lin->n - n0;
        if (chunk > (size_t)H3_NVFP4_LINEAR_CHUNK)
            chunk = (size_t)H3_NVFP4_LINEAR_CHUNK;
        wchunk = (float *)malloc(chunk * lin->k * sizeof(float));
        if (!wchunk) {
            free(packed);
            free(scales_f8);
            free(x_scaled);
            fail(error, error_size, "OOM wchunk");
            return 0;
        }
        for (i = 0; i < chunk; i++) {
            const uint8_t *prow = packed + (n0 + i) * k_half;
            const uint8_t *srow = scales_f8 + (n0 + i) * lin->groups;
            float *dst = wchunk + i * lin->k;
            for (j = 0; j < k_half; j++) {
                uint8_t byte = prow[j];
                size_t col0 = j * 2;
                size_t col1 = col0 + 1;
                size_t g0 = col0 / lin->group_size;
                size_t g1 = col1 / lin->group_size;
                float s0 = h3_f8e4m3fn_to_f32(srow[g0]) * scale2;
                float s1 = h3_f8e4m3fn_to_f32(srow[g1]) * scale2;
                nvfp4_unpack_byte_hi_first(byte, s0, s1, &dst[col0],
                                           &dst[col1]);
            }
        }
        for (r = 0; r < m; r++) {
            const float *xr = xin + r * lin->k;
            float *yr = y + r * lin->n + n0;
            for (o = 0; o < chunk; o++) {
                const float *wr = wchunk + o * lin->k;
                float sum = 0.0f;
                for (i2 = 0; i2 < lin->k; i2++) sum += xr[i2] * wr[i2];
                yr[o] = sum;
            }
        }
        free(wchunk);
        wchunk = NULL;
    }
    free(packed);
    free(scales_f8);
    free(x_scaled);
    return 1;
}

static float *nvfp4_encode_layers_impl(
    const h3_nvfp4_store *store, const int32_t *token_ids, size_t tokens,
    const h3_cuda_vision_span *spans, size_t span_count, const float *merged,
    const float *deepstack[3], unsigned n_layers, h3_nvfp4_encode_stats *stats,
    char *error, size_t error_size) {
    const float eps = 1e-6f;
    const float rope_theta = 5000000.0f;
    h3_nvfp4_embed emb;
    float *hidden = NULL, *normed = NULL, *residual = NULL;
    float *q = NULL, *k = NULL, *v = NULL, *attn = NULL, *attn_out = NULL;
    float *gate = NULL, *up = NULL, *ffn = NULL;
    float *in_w = NULL, *post_w = NULL, *qn_w = NULL, *kn_w = NULL;
    uint32_t *mrope_pos = NULL;
    clock_t t_all0, t_all1, t0, t1;
    unsigned layer, env_layers = 0;
    const char *env;
    size_t i;
    int use_mm = spans && span_count > 0;

    if (stats) memset(stats, 0, sizeof(*stats));
    if (!store || !token_ids || tokens == 0) {
        fail(error, error_size, "invalid encode_layers args");
        return NULL;
    }
    env = getenv("H3_NVFP4_TE_LAYERS");
    if (env && env[0]) {
        env_layers = (unsigned)strtoul(env, NULL, 10);
        if (env_layers > 0) n_layers = env_layers;
    }
    if (n_layers > (unsigned)H3_NVFP4_N_LAYERS)
        n_layers = (unsigned)H3_NVFP4_N_LAYERS;

    t_all0 = clock();
    memset(&emb, 0, sizeof(emb));
    if (!h3_nvfp4_find_embed(store, &emb, error, error_size)) return NULL;
    if (emb.dim != (size_t)H3_NVFP4_TEXT_DIM) {
        fail(error, error_size, "embed dim != 5120");
        return NULL;
    }
    t0 = clock();
    if (stats) {
        stats->embed_cache_hit =
            (g_embed_cache.valid && g_embed_cache.tokens == tokens &&
             g_embed_cache.dim == (size_t)H3_NVFP4_TEXT_DIM &&
             strcmp(g_embed_cache.path, store->path) == 0 && g_embed_cache.ids &&
             memcmp(g_embed_cache.ids, token_ids, tokens * sizeof(int32_t)) ==
                 0)
                ? 1
                : 0;
        stats->used_cuda = g_linear_fn ? 1 : 0;
    }
    hidden = h3_nvfp4_embed_lookup_f32(store, &emb, token_ids, tokens, error,
                                       error_size);
    t1 = clock();
    if (!hidden) return NULL;
#ifdef MADV_WILLNEED
    /* Prefault TE mmap during first encode so packed-cache fill is less
     * page-fault bound. Same bytes as SEQUENTIAL open; lossless. */
    if (store->map && store->map_len > 0)
        (void)madvise((void *)store->map, store->map_len, MADV_WILLNEED);
#endif
    if (use_mm &&
        !nvfp4_inject_vision(hidden, tokens, spans, span_count, merged, error,
                             error_size)) {
        free(hidden);
        return NULL;
    }
    if (stats) stats->embed_ms = nvfp4_elapsed_ms(t0, t1);

    if (n_layers == 0) {
        if (stats) {
            stats->n_layers = 0;
            stats->total_ms = nvfp4_elapsed_ms(t_all0, clock());
            stats->all_finite = 1;
            for (i = 0; i < tokens * (size_t)H3_NVFP4_TEXT_DIM; i++) {
                if (!isfinite(hidden[i])) {
                    stats->all_finite = 0;
                    break;
                }
            }
        }
        return hidden;
    }

    /* Device-resident text TE is the default (H3_NVFP4_HOST_ACT=1 reverts). */
    if (!use_mm && h3_nvfp4_cuda_enabled()) {
        const char *host_act = getenv("H3_NVFP4_HOST_ACT");
        const char *dev_act = getenv("H3_NVFP4_DEV_ACT");
        int want_dev = !(host_act && host_act[0] == '1') &&
                       !(dev_act && dev_act[0] == '0');
        if (want_dev) {
            t0 = clock();
            if (h3_nvfp4_cuda_encode_text_layers(store, hidden, tokens, n_layers,
                                                 error, error_size)) {
                t1 = clock();
                if (stats) {
                    stats->n_layers = n_layers;
                    stats->layers_ms = nvfp4_elapsed_ms(t0, t1);
                    stats->total_ms = nvfp4_elapsed_ms(t_all0, clock());
                    stats->used_cuda = 1;
                    stats->all_finite = 1;
                    for (i = 0; i < tokens * (size_t)H3_NVFP4_TEXT_DIM; i++) {
                        if (!isfinite(hidden[i])) {
                            stats->all_finite = 0;
                            break;
                        }
                    }
                }
                return hidden;
            }
            if (error && error_size && error[0]) {
                free(hidden);
                return NULL;
            }
        }
    }

    normed = (float *)malloc(tokens * H3_NVFP4_TEXT_DIM * sizeof(float));
    residual = (float *)malloc(tokens * H3_NVFP4_TEXT_DIM * sizeof(float));
    q = (float *)malloc(tokens * H3_NVFP4_QUERY_DIM * sizeof(float));
    k = (float *)malloc(tokens * H3_NVFP4_KV_DIM * sizeof(float));
    v = (float *)malloc(tokens * H3_NVFP4_KV_DIM * sizeof(float));
    attn = (float *)malloc(tokens * H3_NVFP4_QUERY_DIM * sizeof(float));
    attn_out = (float *)malloc(tokens * H3_NVFP4_TEXT_DIM * sizeof(float));
    gate = (float *)malloc(tokens * H3_NVFP4_INTERMEDIATE * sizeof(float));
    up = (float *)malloc(tokens * H3_NVFP4_INTERMEDIATE * sizeof(float));
    ffn = (float *)malloc(tokens * H3_NVFP4_INTERMEDIATE * sizeof(float));
    if (!normed || !residual || !q || !k || !v || !attn || !attn_out || !gate ||
        !up || !ffn) {
        fail(error, error_size, "OOM encode_layers buffers");
        goto fail_enc;
    }
    if (use_mm) {
        if (n_layers > 0 &&
            (!deepstack || !deepstack[0] || !deepstack[1] || !deepstack[2])) {
            fail(error, error_size, "missing vision deepstack");
            goto fail_enc;
        }
        mrope_pos = (uint32_t *)calloc(3 * tokens, sizeof(uint32_t));
        if (!mrope_pos ||
            !nvfp4_build_mrope_pos(spans, span_count, tokens, mrope_pos, error,
                                   error_size)) {
            if (!mrope_pos)
                fail(error, error_size, "OOM mRoPE positions");
            goto fail_enc;
        }
    }

    t0 = clock();
    {
        struct timespec ts;
        double wall0;
        const char *v = getenv("H3_NVFP4_VERBOSE");
        int verbose = v && v[0] == '1';
        clock_gettime(CLOCK_MONOTONIC, &ts);
        wall0 = (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
        if (verbose) {
            fprintf(stderr,
                    "te_encode_begin layers=%u tokens=%zu cuda=%d omp=%d\n",
                    n_layers, tokens, g_linear_fn ? 1 : 0,
#ifdef _OPENMP
                    omp_get_max_threads()
#else
                    0
#endif
            );
            fflush(stderr);
        }
        (void)wall0;
    }
    for (layer = 0; layer < n_layers; layer++) {
        char base[192];
        h3_nvfp4_linear lin;
        struct timespec ts0, ts1;
        const char *v = getenv("H3_NVFP4_VERBOSE");
        int verbose = v && v[0] == '1';
        clock_gettime(CLOCK_MONOTONIC, &ts0);
        if (verbose) {
            fprintf(stderr, "te_layer_begin %u/%u tokens=%zu\n", layer + 1,
                    n_layers, tokens);
            fflush(stderr);
        }
        snprintf(base, sizeof(base), "model.layers.%u.input_layernorm.weight",
                 layer);
        in_w = nvfp4_load_bf16_f32(store, base, H3_NVFP4_TEXT_DIM, error,
                                  error_size);
        if (!in_w) goto fail_enc;
        nvfp4_rms_norm(hidden, in_w, tokens, H3_NVFP4_TEXT_DIM, eps, normed);
        free(in_w);
        in_w = NULL;

        snprintf(base, sizeof(base), "model.layers.%u.self_attn.q_proj", layer);
        if (!h3_nvfp4_find_linear(store, base, &lin, error, error_size) ||
            !h3_nvfp4_linear_f32(store, &lin, normed, tokens, q, error,
                                error_size))
            goto fail_enc;
        snprintf(base, sizeof(base), "model.layers.%u.self_attn.k_proj", layer);
        if (!h3_nvfp4_find_linear(store, base, &lin, error, error_size) ||
            !h3_nvfp4_linear_f32(store, &lin, normed, tokens, k, error,
                                error_size))
            goto fail_enc;
        snprintf(base, sizeof(base), "model.layers.%u.self_attn.v_proj", layer);
        if (!h3_nvfp4_find_linear(store, base, &lin, error, error_size) ||
            !h3_nvfp4_linear_f32(store, &lin, normed, tokens, v, error,
                                error_size))
            goto fail_enc;

        snprintf(base, sizeof(base), "model.layers.%u.self_attn.q_norm.weight",
                 layer);
        qn_w = nvfp4_load_bf16_f32(store, base, H3_NVFP4_HEAD_DIM, error,
                                   error_size);
        snprintf(base, sizeof(base), "model.layers.%u.self_attn.k_norm.weight",
                 layer);
        kn_w = nvfp4_load_bf16_f32(store, base, H3_NVFP4_HEAD_DIM, error,
                                   error_size);
        if (!qn_w || !kn_w) goto fail_enc;
        /* Optional TE layer-0 oracle dumps (H3_NVFP4_DUMP_DIR). */
        if (layer == 0) {
            const char *dd = getenv("H3_NVFP4_DUMP_DIR");
            if (dd && dd[0]) {
                char path[768];
                FILE *fp;
                snprintf(path, sizeof(path), "%s/c_L0_normed.f32", dd);
                fp = fopen(path, "wb");
                if (fp) {
                    fwrite(normed, sizeof(float),
                           tokens * (size_t)H3_NVFP4_TEXT_DIM, fp);
                    fclose(fp);
                }
                snprintf(path, sizeof(path), "%s/c_L0_q.f32", dd);
                fp = fopen(path, "wb");
                if (fp) {
                    fwrite(q, sizeof(float),
                           tokens * (size_t)H3_NVFP4_QUERY_DIM, fp);
                    fclose(fp);
                }
                snprintf(path, sizeof(path), "%s/c_L0_k.f32", dd);
                fp = fopen(path, "wb");
                if (fp) {
                    fwrite(k, sizeof(float), tokens * (size_t)H3_NVFP4_KV_DIM,
                           fp);
                    fclose(fp);
                }
                snprintf(path, sizeof(path), "%s/c_L0_v.f32", dd);
                fp = fopen(path, "wb");
                if (fp) {
                    fwrite(v, sizeof(float), tokens * (size_t)H3_NVFP4_KV_DIM,
                           fp);
                    fclose(fp);
                }
            }
        }
        nvfp4_head_rms(q, qn_w, tokens, H3_NVFP4_QUERY_HEADS, H3_NVFP4_HEAD_DIM,
                       eps);
        nvfp4_head_rms(k, kn_w, tokens, H3_NVFP4_KV_HEADS, H3_NVFP4_HEAD_DIM,
                       eps);
        free(qn_w);
        free(kn_w);
        qn_w = kn_w = NULL;

        if (use_mm)
            nvfp4_rope_mrope(q, k, tokens, H3_NVFP4_QUERY_HEADS,
                             H3_NVFP4_KV_HEADS, H3_NVFP4_HEAD_DIM, rope_theta,
                             mrope_pos);
        else
            nvfp4_rope(q, k, tokens, H3_NVFP4_QUERY_HEADS, H3_NVFP4_KV_HEADS,
                       H3_NVFP4_HEAD_DIM, rope_theta);
        if (!nvfp4_gqa_causal(q, k, v, tokens, H3_NVFP4_QUERY_HEADS,
                              H3_NVFP4_KV_HEADS, H3_NVFP4_HEAD_DIM, attn, error,
                              error_size))
            goto fail_enc;

        snprintf(base, sizeof(base), "model.layers.%u.self_attn.o_proj", layer);
        if (!h3_nvfp4_find_linear(store, base, &lin, error, error_size) ||
            !h3_nvfp4_linear_f32(store, &lin, attn, tokens, attn_out, error,
                                error_size))
            goto fail_enc;
        for (i = 0; i < tokens * (size_t)H3_NVFP4_TEXT_DIM; i++)
            hidden[i] += attn_out[i];

        snprintf(base, sizeof(base),
                 "model.layers.%u.post_attention_layernorm.weight", layer);
        post_w = nvfp4_load_bf16_f32(store, base, H3_NVFP4_TEXT_DIM, error,
                                    error_size);
        if (!post_w) goto fail_enc;
        nvfp4_rms_norm(hidden, post_w, tokens, H3_NVFP4_TEXT_DIM, eps, normed);
        free(post_w);
        post_w = NULL;

        snprintf(base, sizeof(base), "model.layers.%u.mlp.gate_proj", layer);
        if (!h3_nvfp4_find_linear(store, base, &lin, error, error_size) ||
            !h3_nvfp4_linear_f32(store, &lin, normed, tokens, gate, error,
                                error_size))
            goto fail_enc;
        snprintf(base, sizeof(base), "model.layers.%u.mlp.up_proj", layer);
        if (!h3_nvfp4_find_linear(store, base, &lin, error, error_size) ||
            !h3_nvfp4_linear_f32(store, &lin, normed, tokens, up, error,
                                error_size))
            goto fail_enc;
        nvfp4_silu_mul(gate, up, tokens * (size_t)H3_NVFP4_INTERMEDIATE, ffn);
        snprintf(base, sizeof(base), "model.layers.%u.mlp.down_proj", layer);
        if (!h3_nvfp4_find_linear(store, base, &lin, error, error_size) ||
            !h3_nvfp4_linear_f32(store, &lin, ffn, tokens, attn_out, error,
                                error_size))
            goto fail_enc;
        for (i = 0; i < tokens * (size_t)H3_NVFP4_TEXT_DIM; i++)
            hidden[i] += attn_out[i];
        if (use_mm && layer < 3 &&
            !nvfp4_add_vision(hidden, tokens, spans, span_count,
                              deepstack[layer], error, error_size))
            goto fail_enc;
        clock_gettime(CLOCK_MONOTONIC, &ts1);
        if (verbose) {
            fprintf(stderr, "te_layer_done %u/%u wall_ms=%.1f\n", layer + 1,
                    n_layers,
                    ((double)ts1.tv_sec * 1000.0 + (double)ts1.tv_nsec / 1e6) -
                        ((double)ts0.tv_sec * 1000.0 +
                         (double)ts0.tv_nsec / 1e6));
            fflush(stderr);
        }
    }
    t1 = clock();
    if (stats) {
        stats->n_layers = n_layers;
        stats->layers_ms = nvfp4_elapsed_ms(t0, t1);
        stats->total_ms = nvfp4_elapsed_ms(t_all0, clock());
        stats->used_cuda = g_linear_fn ? 1 : 0;
        stats->all_finite = 1;
        for (i = 0; i < tokens * (size_t)H3_NVFP4_TEXT_DIM; i++) {
            if (!isfinite(hidden[i])) {
                stats->all_finite = 0;
                break;
            }
        }
    }
    free(normed);
    free(residual);
    free(q);
    free(k);
    free(v);
    free(attn);
    free(attn_out);
    free(gate);
    free(up);
    free(ffn);
    free(mrope_pos);
    return hidden;

fail_enc:
    free(hidden);
    free(normed);
    free(residual);
    free(q);
    free(k);
    free(v);
    free(attn);
    free(attn_out);
    free(gate);
    free(up);
    free(ffn);
    free(in_w);
    free(post_w);
    free(qn_w);
    free(kn_w);
    free(mrope_pos);
    return NULL;
}

float *h3_nvfp4_encode_layers_f32(const h3_nvfp4_store *store,
                                  const int32_t *token_ids, size_t tokens,
                                  unsigned n_layers,
                                  h3_nvfp4_encode_stats *stats, char *error,
                                  size_t error_size) {
    return nvfp4_encode_layers_impl(store, token_ids, tokens, NULL, 0, NULL,
                                    NULL, n_layers, stats, error, error_size);
}

float *h3_nvfp4_encode_layers_multimodal_f32(
    const h3_nvfp4_store *store, const int32_t *token_ids, size_t tokens,
    const h3_cuda_vision_span *spans, size_t span_count, const float *merged,
    const float *deepstack[3], unsigned n_layers, h3_nvfp4_encode_stats *stats,
    char *error, size_t error_size) {
    if (!spans || span_count == 0)
        return h3_nvfp4_encode_layers_f32(store, token_ids, tokens, n_layers,
                                          stats, error, error_size);
    return nvfp4_encode_layers_impl(store, token_ids, tokens, spans, span_count,
                                    merged, deepstack, n_layers, stats, error,
                                    error_size);
}
