#include "h3_tae_cuda.h"
#include "h3_gpu_cuda.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef H3_TAE_HAVE_CUDA
#define H3_TAE_HAVE_CUDA 1
#endif

static void fail(char *err, size_t err_size, const char *msg) {
    if (err && err_size)
        snprintf(err, err_size, "%s", msg ? msg : "TAE CUDA error");
}

static int ck(cudaError_t st, char *err, size_t err_size, const char *what) {
    if (st == cudaSuccess) return 1;
    snprintf(err, err_size, "TAE CUDA %s: %s（不是金线成片）", what,
             cudaGetErrorString(st));
    return 0;
}

__global__ void k_relu(float *x, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && x[i] < 0.0f) x[i] = 0.0f;
}

__global__ void k_clamp_tanh3(float *x, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = tanhf(x[i] / 3.0f) * 3.0f;
}

__global__ void k_add_relu(float *d, const float *x, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = d[i] + x[i];
        d[i] = v < 0.0f ? 0.0f : v;
    }
}

/* Same-size conv: 3x3 pad1 or 1x1. Output spatial == input spatial. */
__global__ void k_conv_same(const float *in, const float *w, const float *b,
                            float *out, int n, int ic, int oc, int ih, int iw,
                            int kh, int kw, int pad) {
    int ox = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int oy = (int)blockIdx.y * (int)blockDim.y + (int)threadIdx.y;
    int noc = (int)blockIdx.z;
    if (ox >= iw || oy >= ih || noc >= n * oc) return;
    int ni = noc / oc;
    int oc_i = noc - ni * oc;
    float acc = b ? b[oc_i] : 0.0f;
    int c, ky, kx;
    for (c = 0; c < ic; c++) {
        const float *win = w + ((size_t)oc_i * (size_t)ic + (size_t)c) *
                                   (size_t)kh * (size_t)kw;
        const float *pin =
            in + ((size_t)ni * (size_t)ic + (size_t)c) * (size_t)ih *
                     (size_t)iw;
        for (ky = 0; ky < kh; ky++) {
            int iy = oy + ky - pad;
            if ((unsigned)iy >= (unsigned)ih) continue;
            for (kx = 0; kx < kw; kx++) {
                int ix = ox + kx - pad;
                if ((unsigned)ix >= (unsigned)iw) continue;
                acc += pin[(size_t)iy * (size_t)iw + (size_t)ix] *
                       win[(size_t)ky * (size_t)kw + (size_t)kx];
            }
        }
    }
    out[(((size_t)ni * (size_t)oc + (size_t)oc_i) * (size_t)ih + (size_t)oy) *
            (size_t)iw +
        (size_t)ox] = acc;
}

__global__ void k_upsample2(const float *in, float *out, int n, int c, int h,
                            int w) {
    int ox = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int oy = (int)blockIdx.y * (int)blockDim.y + (int)threadIdx.y;
    int nc = (int)blockIdx.z;
    int oh = h * 2, ow = w * 2;
    if (ox >= ow || oy >= oh || nc >= n * c) return;
    int ni = nc / c;
    int ch = nc - ni * c;
    out[(((size_t)ni * (size_t)c + (size_t)ch) * (size_t)oh + (size_t)oy) *
            (size_t)ow +
        (size_t)ox] =
        in[(((size_t)ni * (size_t)c + (size_t)ch) * (size_t)h +
            (size_t)(oy / 2)) *
               (size_t)w +
           (size_t)(ox / 2)];
}

__global__ void k_tgrow(const float *mid, float *out, int n, int oc, int stride,
                        int h, int w) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int c_out = oc / stride;
    size_t plane = (size_t)h * (size_t)w;
    size_t total = (size_t)n * (size_t)stride * (size_t)c_out * plane;
    if (i >= total) return;
    size_t rem = i;
    int x = (int)(rem % (size_t)w);
    rem /= (size_t)w;
    int y = (int)(rem % (size_t)h);
    rem /= (size_t)h;
    int ch = (int)(rem % (size_t)c_out);
    rem /= (size_t)c_out;
    int t = (int)(rem % (size_t)stride);
    int ni = (int)(rem / (size_t)stride);
    int src_c = t * c_out + ch;
    out[i] = mid[(((size_t)ni * (size_t)oc + (size_t)src_c) * (size_t)h +
                  (size_t)y) *
                     (size_t)w +
                 (size_t)x];
}

__global__ void k_pixel_shuffle2(const float *in, float *out, int n, int h,
                                 int w) {
    int ox = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int oy = (int)blockIdx.y * (int)blockDim.y + (int)threadIdx.y;
    int ni = (int)blockIdx.z;
    int oh = h * 2, ow = w * 2;
    if (ox >= ow || oy >= oh || ni >= n) return;
    int rh = oy % 2, rw = ox % 2;
    int c;
    for (c = 0; c < 3; c++) {
        int src_c = c * 4 + rh * 2 + rw;
        out[(((size_t)ni * 3 + (size_t)c) * (size_t)oh + (size_t)oy) *
                (size_t)ow +
            (size_t)ox] =
            in[(((size_t)ni * 12 + (size_t)src_c) * (size_t)h +
                (size_t)(oy / 2)) *
                   (size_t)w +
               (size_t)(ox / 2)];
    }
}

__global__ void k_cat_past(const float *x, const float *past, float *cat,
                           int n, int c, int h, int w) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t plane = (size_t)c * (size_t)h * (size_t)w;
    size_t total = (size_t)n * plane * 2;
    if (i >= total) return;
    size_t ni = i / (plane * 2);
    size_t rest = i - ni * plane * 2;
    if (rest < plane)
        cat[i] = x[ni * plane + rest];
    else
        cat[i] = past[ni * plane + (rest - plane)];
}

__global__ void k_shift_past(const float *x, float *past, int n, size_t plane) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)n * plane;
    if (i >= total) return;
    size_t t = i / plane;
    if (t == 0)
        past[i] = 0.0f;
    else
        past[i] = x[i - plane];
}

__global__ void k_chw_to_nchw(const float *chw, float *nchw, int t, int h,
                              int w) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t plane = (size_t)h * (size_t)w;
    size_t total = 24ull * (size_t)t * plane;
    if (i >= total) return;
    size_t rem = i;
    int ww = (int)(rem % (size_t)w);
    rem /= (size_t)w;
    int hh = (int)(rem % (size_t)h);
    rem /= (size_t)h;
    int tt = (int)(rem % (size_t)t);
    int ch = (int)(rem / (size_t)t);
    nchw[(((size_t)tt * 24 + (size_t)ch) * (size_t)h + (size_t)hh) *
             (size_t)w +
         (size_t)ww] =
        chw[(((size_t)ch * (size_t)t + (size_t)tt) * (size_t)h + (size_t)hh) *
                (size_t)w +
            (size_t)ww];
}

__global__ void k_nchw_to_hwc01(const float *nchw, float *hwc, int src, int h,
                                int w) {
    int ox = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int oy = (int)blockIdx.y * (int)blockDim.y + (int)threadIdx.y;
    if (ox >= w || oy >= h) return;
    int ch;
    for (ch = 0; ch < 3; ch++) {
        float v = nchw[(((size_t)src * 3 + (size_t)ch) * (size_t)h +
                        (size_t)oy) *
                           (size_t)w +
                       (size_t)ox];
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        hwc[((size_t)oy * (size_t)w + (size_t)ox) * 3 + (size_t)ch] = v;
    }
}

static void launch1(void (*k)(float *, size_t), float *x, size_t n) {
    int threads = 256;
    int blocks = (int)((n + 255) / 256);
    if (blocks < 1) blocks = 1;
    k<<<blocks, threads>>>(x, n);
}

typedef struct {
    float *ptr;
    size_t cap;
} tae_dbuf;

static int dbuf_fit(tae_dbuf *b, size_t n, char *err, size_t err_size) {
    if (n <= b->cap) return 1;
    if (b->ptr) cudaFree(b->ptr);
    b->ptr = NULL;
    b->cap = 0;
    if (!ck(cudaMalloc((void **)&b->ptr, n * sizeof(float)), err, err_size,
            "alloc"))
        return 0;
    b->cap = n;
    return 1;
}

static void dbuf_free(tae_dbuf *b) {
    if (b->ptr) cudaFree(b->ptr);
    b->ptr = NULL;
    b->cap = 0;
}

static int upload_conv(const tae_conv *c, float **dw, float **db, char *err,
                       size_t err_size) {
    size_t nw = (size_t)c->oc * (size_t)c->ic * (size_t)c->kh * (size_t)c->kw;
    *dw = NULL;
    *db = NULL;
    if (!ck(cudaMalloc((void **)dw, nw * sizeof(float)), err, err_size,
            "w"))
        return 0;
    if (!ck(cudaMemcpy(*dw, c->w, nw * sizeof(float), cudaMemcpyHostToDevice),
            err, err_size, "w H2D")) {
        cudaFree(*dw);
        *dw = NULL;
        return 0;
    }
    if (c->b) {
        if (!ck(cudaMalloc((void **)db, (size_t)c->oc * sizeof(float)), err,
                err_size, "b")) {
            cudaFree(*dw);
            *dw = NULL;
            return 0;
        }
        if (!ck(cudaMemcpy(*db, c->b, (size_t)c->oc * sizeof(float),
                           cudaMemcpyHostToDevice),
                err, err_size, "b H2D")) {
            cudaFree(*dw);
            cudaFree(*db);
            *dw = NULL;
            *db = NULL;
            return 0;
        }
    }
    return 1;
}

typedef struct {
    float *w, *b;
} dconv;

static void dconv_free(dconv *c) {
    if (!c) return;
    cudaFree(c->w);
    cudaFree(c->b);
    c->w = c->b = NULL;
}

static int conv_same_d(h3_cuda *cuda, const float *in, int n, int ih, int iw,
                       const tae_conv *cv, const dconv *d, float *out,
                       char *err, size_t err_size) {
    int pad = (cv->kh == 3) ? 1 : 0;
    if (cuda && d->w &&
        h3_cuda_conv3d_ncdhw_bias_dd(cuda, in, d->w, d->b, out, n, cv->ic,
                                     cv->oc, 1, ih, iw, 1, cv->kh, cv->kw, 0,
                                     pad, pad, 1, err, err_size)) {
        static int logged;
        if (!logged) {
            fprintf(stderr, "tae: conv backend=cudnn ncdhw-as-2d (非金线)\n");
            logged = 1;
        }
        return 1;
    }
    {
        dim3 block(16, 16);
        dim3 grid((iw + 15) / 16, (ih + 15) / 16, n * cv->oc);
        k_conv_same<<<grid, block>>>(in, d->w, d->b, out, n, cv->ic, cv->oc, ih,
                                     iw, cv->kh, cv->kw, pad);
        return ck(cudaGetLastError(), err, err_size, "conv");
    }
}

static int relu_d(float *x, size_t n, char *err, size_t err_size) {
    launch1(k_relu, x, n);
    return ck(cudaGetLastError(), err, err_size, "relu");
}

static int mem_parallel_d(h3_cuda *cuda, float **px, int n, int h, int w,
                          const tae_mem *m, const dconv *d0, const dconv *d2,
                          const dconv *d4, tae_dbuf *past, tae_dbuf *cat,
                          tae_dbuf *a, tae_dbuf *b, tae_dbuf *y, char *err,
                          size_t err_size) {
    int c = m->conv2.oc;
    size_t plane = (size_t)c * (size_t)h * (size_t)w;
    size_t nplane = (size_t)n * plane;
    int threads = 256;
    if (m->conv0.ic != c * 2) {
        fail(err, err_size, "TAE mem ic");
        return 0;
    }
    if (!dbuf_fit(past, nplane, err, err_size) ||
        !dbuf_fit(cat, nplane * 2, err, err_size) ||
        !dbuf_fit(a, nplane, err, err_size) ||
        !dbuf_fit(b, nplane, err, err_size) ||
        !dbuf_fit(y, nplane, err, err_size))
        return 0;
    {
        int blocks = (int)((nplane + 255) / 256);
        k_shift_past<<<blocks < 1 ? 1 : blocks, threads>>>(*px, past->ptr, n,
                                                           plane);
        if (!ck(cudaGetLastError(), err, err_size, "past")) return 0;
        blocks = (int)((nplane * 2 + 255) / 256);
        k_cat_past<<<blocks < 1 ? 1 : blocks, threads>>>(*px, past->ptr,
                                                         cat->ptr, n, c, h, w);
        if (!ck(cudaGetLastError(), err, err_size, "cat")) return 0;
    }
    if (!conv_same_d(cuda, cat->ptr, n, h, w, &m->conv0, d0, a->ptr, err,
                     err_size))
        return 0;
    if (!relu_d(a->ptr, nplane, err, err_size)) return 0;
    if (!conv_same_d(cuda, a->ptr, n, h, w, &m->conv2, d2, b->ptr, err,
                     err_size))
        return 0;
    if (!relu_d(b->ptr, nplane, err, err_size)) return 0;
    if (!conv_same_d(cuda, b->ptr, n, h, w, &m->conv4, d4, y->ptr, err,
                     err_size))
        return 0;
    {
        int blocks = (int)((nplane + 255) / 256);
        k_add_relu<<<blocks < 1 ? 1 : blocks, threads>>>(y->ptr, *px, nplane);
        if (!ck(cudaGetLastError(), err, err_size, "res")) return 0;
    }
    *px = y->ptr;
    return 1;
}

static int tae_extract_t(const float *src, int t, int h, int w, int t0,
                         int tlen, float *dst) {
    int c, tt;
    size_t plane = (size_t)h * (size_t)w;
    if (!src || !dst || tlen < 1 || t0 < 0 || t0 + tlen > t)
        return 0;
    for (c = 0; c < 24; c++)
        for (tt = 0; tt < tlen; tt++)
            memcpy(dst + ((size_t)c * (size_t)tlen + (size_t)tt) * plane,
                   src + ((size_t)c * (size_t)t + (size_t)(t0 + tt)) * plane,
                   plane * sizeof(float));
    return 1;
}

static int tae_cuda_decode_span(h3_tae_ctx *tc, const float *latent_chw, int t,
                                int h, int w, int want_frames, int skip_trim,
                                h3_vae_int8_frames *frames, char *err,
                                size_t err_size);

int h3_tae_cuda_decode_frames(h3_tae_ctx *tc, const float *latent_chw, int t,
                              int h, int w, int want_frames,
                              h3_vae_int8_frames *frames, char *err,
                              size_t err_size) {
    int chunk = 8, overlap = 1, t0, first = 1, take_total, start_drop;
    size_t plane_hw;
    float *slice = NULL;
    const char *e;
    if (!frames) {
        fail(err, err_size, "TAE frames null");
        return 0;
    }
    memset(frames, 0, sizeof(*frames));
    e = getenv("H3_TAE_TIME_CHUNK");
    if (e && *e) {
        int v = atoi(e);
        if (v >= 1) chunk = v;
    }
    if (getenv("H3_TAE_NO_CHUNK") && getenv("H3_TAE_NO_CHUNK")[0] == '1')
        chunk = t > 0 ? t : 1;
    /* 480p 30×54 fits one shot; 768P 48×84 peaks ~9GiB×2 at tgrow3. */
    if (t <= chunk || (size_t)h * (size_t)w < (size_t)40 * 70)
        return tae_cuda_decode_span(tc, latent_chw, t, h, w, want_frames, 0,
                                    frames, err, err_size);
    if (want_frames < 1) want_frames = t * 4;
    frames->height = h * 16;
    frames->width = w * 16;
    frames->frames = want_frames;
    plane_hw = (size_t)frames->height * (size_t)frames->width * 3;
    frames->rgb = (float *)malloc((size_t)want_frames * plane_hw * sizeof(float));
    if (!frames->rgb) {
        fail(err, err_size, "TAE CUDA OOM frames");
        return 0;
    }
    start_drop = 3; /* same global trim as the one-shot path */
    take_total = 0;
    for (t0 = 0; t0 < t; ) {
        int tlen = chunk;
        int skip = first ? 0 : overlap;
        h3_vae_int8_frames part;
        int src, dst_i, ncopy;
        if (!first) t0 -= overlap;
        if (t0 < 0) t0 = 0;
        if (t0 + tlen > t) tlen = t - t0;
        if (tlen < 1) break;
        slice = (float *)realloc(slice, 24ull * (size_t)tlen * (size_t)h *
                                            (size_t)w * sizeof(float));
        if (!slice || !tae_extract_t(latent_chw, t, h, w, t0, tlen, slice)) {
            fail(err, err_size, "TAE CUDA OOM chunk slice");
            free(slice);
            free(frames->rgb);
            memset(frames, 0, sizeof(*frames));
            return 0;
        }
        memset(&part, 0, sizeof(part));
        if (!tae_cuda_decode_span(tc, slice, tlen, h, w, tlen * 4, 1, &part,
                                  err, err_size)) {
            free(slice);
            free(part.rgb);
            free(frames->rgb);
            memset(frames, 0, sizeof(*frames));
            return 0;
        }
        src = skip * 4;
        if (first && start_drop > 0) {
            src += start_drop;
            start_drop = 0;
        }
        if (src < 0) src = 0;
        if (src > part.frames) src = part.frames;
        ncopy = part.frames - src;
        if (take_total + ncopy > want_frames)
            ncopy = want_frames - take_total;
        for (dst_i = 0; dst_i < ncopy; dst_i++)
            memcpy(frames->rgb + ((size_t)take_total + (size_t)dst_i) * plane_hw,
                   part.rgb + ((size_t)src + (size_t)dst_i) * plane_hw,
                   plane_hw * sizeof(float));
        take_total += ncopy;
        free(part.rgb);
        first = 0;
        t0 += tlen;
        if (take_total >= want_frames) break;
    }
    free(slice);
    frames->frames = take_total;
    fprintf(stderr,
            "tae: decoded %d frames %dx%d from latent T=%d "
            "(GPU TAEHV chunk=%d；非金线、画质有损)\n",
            frames->frames, frames->width, frames->height, t, chunk);
    return take_total > 0;
}

static int tae_cuda_decode_span(h3_tae_ctx *tc, const float *latent_chw, int t,
                                int h, int w, int want_frames, int skip_trim,
                                h3_vae_int8_frames *frames, char *err,
                                size_t err_size) {
    h3_cuda *cuda = (h3_cuda *)tc->cuda;
    dconv dc1 = {0}, dt1 = {0}, dc8 = {0}, dt2 = {0}, dc14 = {0}, dt3 = {0},
          dc20 = {0}, dc22 = {0};
    dconv dm256[9] = {{0}}, dm128[9] = {{0}}, dm64[9] = {{0}};
    tae_dbuf A = {0}, B = {0}, C = {0}, past = {0}, cat = {0}, tmpa = {0},
             tmpb = {0};
    float *x = NULL, *tmp = NULL;
    int n = t, i, ok = 0;
    int hh = h, ww = w;
    size_t plane24;
    int trim = skip_trim ? 0 : 3;

    if (!frames) {
        fail(err, err_size, "TAE frames null");
        return 0;
    }
    memset(frames, 0, sizeof(*frames));
    if (!cuda || !tc->ok || !latent_chw || t < 1) {
        fail(err, err_size, "TAE CUDA args（不是金线成片）");
        return 0;
    }
    if (want_frames < 1) want_frames = t * 4;
    if (!h3_cuda_bind_device(cuda, err, err_size)) return 0;

    if (!upload_conv(&tc->c1, &dc1.w, &dc1.b, err, err_size) ||
        !upload_conv(&tc->tgrow1, &dt1.w, &dt1.b, err, err_size) ||
        !upload_conv(&tc->c8, &dc8.w, &dc8.b, err, err_size) ||
        !upload_conv(&tc->tgrow2, &dt2.w, &dt2.b, err, err_size) ||
        !upload_conv(&tc->c14, &dc14.w, &dc14.b, err, err_size) ||
        !upload_conv(&tc->tgrow3, &dt3.w, &dt3.b, err, err_size) ||
        !upload_conv(&tc->c20, &dc20.w, &dc20.b, err, err_size) ||
        !upload_conv(&tc->c22, &dc22.w, &dc22.b, err, err_size))
        goto done;
    for (i = 0; i < 3; i++) {
        if (!upload_conv(&tc->mem256[i].conv0, &dm256[i * 3 + 0].w,
                         &dm256[i * 3 + 0].b, err, err_size) ||
            !upload_conv(&tc->mem256[i].conv2, &dm256[i * 3 + 1].w,
                         &dm256[i * 3 + 1].b, err, err_size) ||
            !upload_conv(&tc->mem256[i].conv4, &dm256[i * 3 + 2].w,
                         &dm256[i * 3 + 2].b, err, err_size) ||
            !upload_conv(&tc->mem128[i].conv0, &dm128[i * 3 + 0].w,
                         &dm128[i * 3 + 0].b, err, err_size) ||
            !upload_conv(&tc->mem128[i].conv2, &dm128[i * 3 + 1].w,
                         &dm128[i * 3 + 1].b, err, err_size) ||
            !upload_conv(&tc->mem128[i].conv4, &dm128[i * 3 + 2].w,
                         &dm128[i * 3 + 2].b, err, err_size) ||
            !upload_conv(&tc->mem64[i].conv0, &dm64[i * 3 + 0].w,
                         &dm64[i * 3 + 0].b, err, err_size) ||
            !upload_conv(&tc->mem64[i].conv2, &dm64[i * 3 + 1].w,
                         &dm64[i * 3 + 1].b, err, err_size) ||
            !upload_conv(&tc->mem64[i].conv4, &dm64[i * 3 + 2].w,
                         &dm64[i * 3 + 2].b, err, err_size))
            goto done;
    }

    plane24 = (size_t)h * (size_t)w;
    if (!dbuf_fit(&A, (size_t)t * 256 * plane24 * 4 + 16, err, err_size) ||
        !dbuf_fit(&B, (size_t)t * 256 * plane24 * 4 + 16, err, err_size))
        goto done;
    {
        size_t nlat = 24ull * (size_t)t * plane24;
        int blocks = (int)((nlat + 255) / 256);
        if (!ck(cudaMemcpy(A.ptr, latent_chw, nlat * sizeof(float),
                           cudaMemcpyHostToDevice),
                err, err_size, "latent H2D"))
            goto done;
        k_chw_to_nchw<<<blocks < 1 ? 1 : blocks, 256>>>(A.ptr, B.ptr, t, h, w);
        if (!ck(cudaGetLastError(), err, err_size, "layout")) goto done;
        k_clamp_tanh3<<<blocks < 1 ? 1 : blocks, 256>>>(B.ptr, nlat);
        if (!ck(cudaGetLastError(), err, err_size, "clamp")) goto done;
    }
    x = B.ptr;
    tmp = A.ptr;
    if (!conv_same_d(cuda, x, n, hh, ww, &tc->c1, &dc1, tmp, err, err_size))
        goto done;
    x = tmp;
    tmp = B.ptr;
    if (!relu_d(x, (size_t)n * 256 * plane24, err, err_size)) goto done;
    for (i = 0; i < 3; i++) {
        if (!mem_parallel_d(cuda, &x, n, hh, ww, &tc->mem256[i], &dm256[i * 3],
                            &dm256[i * 3 + 1], &dm256[i * 3 + 2], &past, &cat,
                            &tmpa, &tmpb, x == A.ptr ? &B : &A, err,
                            err_size))
            goto done;
    }
    {
        dim3 block(16, 16);
        dim3 grid((ww * 2 + 15) / 16, (hh * 2 + 15) / 16, n * 256);
        size_t need = (size_t)n * 256 * (size_t)(hh * 2) * (size_t)(ww * 2);
        if (!dbuf_fit(x == A.ptr ? &B : &A, need, err, err_size)) goto done;
        tmp = (x == A.ptr) ? B.ptr : A.ptr;
        k_upsample2<<<grid, block>>>(x, tmp, n, 256, hh, ww);
        if (!ck(cudaGetLastError(), err, err_size, "up1")) goto done;
        x = tmp;
        hh *= 2;
        ww *= 2;
    }
    {
        size_t need = (size_t)n * 256 * (size_t)hh * (size_t)ww;
        if (!dbuf_fit(x == A.ptr ? &B : &A, need, err, err_size)) goto done;
        tmp = (x == A.ptr) ? B.ptr : A.ptr;
        if (!conv_same_d(cuda, x, n, hh, ww, &tc->tgrow1, &dt1, tmp, err, err_size))
            goto done;
        x = tmp;
    }
    {
        size_t need = (size_t)n * 128 * (size_t)hh * (size_t)ww;
        if (!dbuf_fit(x == A.ptr ? &B : &A, need, err, err_size)) goto done;
        tmp = (x == A.ptr) ? B.ptr : A.ptr;
        if (!conv_same_d(cuda, x, n, hh, ww, &tc->c8, &dc8, tmp, err, err_size))
            goto done;
        x = tmp;
    }
    for (i = 0; i < 3; i++) {
        if (!mem_parallel_d(cuda, &x, n, hh, ww, &tc->mem128[i], &dm128[i * 3],
                            &dm128[i * 3 + 1], &dm128[i * 3 + 2], &past, &cat,
                            &tmpa, &tmpb, x == A.ptr ? &B : &A, err,
                            err_size))
            goto done;
    }
    {
        dim3 block(16, 16);
        dim3 grid((ww * 2 + 15) / 16, (hh * 2 + 15) / 16, n * 128);
        size_t need = (size_t)n * 128 * (size_t)(hh * 2) * (size_t)(ww * 2);
        if (!dbuf_fit(x == A.ptr ? &B : &A, need, err, err_size)) goto done;
        tmp = (x == A.ptr) ? B.ptr : A.ptr;
        k_upsample2<<<grid, block>>>(x, tmp, n, 128, hh, ww);
        if (!ck(cudaGetLastError(), err, err_size, "up2")) goto done;
        x = tmp;
        hh *= 2;
        ww *= 2;
    }
    {
        size_t mid_n = (size_t)n * 256 * (size_t)hh * (size_t)ww;
        size_t out_n = (size_t)n * 2 * 128 * (size_t)hh * (size_t)ww;
        tae_dbuf *midb = (x == A.ptr) ? &B : &A;
        if (!dbuf_fit(midb, mid_n, err, err_size)) goto done;
        if (!conv_same_d(cuda, x, n, hh, ww, &tc->tgrow2, &dt2, midb->ptr, err,
                         err_size))
            goto done;
        if (!dbuf_fit(&C, out_n > C.cap ? out_n : C.cap, err, err_size))
            goto done;
        {
            int blocks = (int)((out_n + 255) / 256);
            k_tgrow<<<blocks < 1 ? 1 : blocks, 256>>>(midb->ptr, C.ptr, n, 256,
                                                      2, hh, ww);
            if (!ck(cudaGetLastError(), err, err_size, "tgrow2")) goto done;
        }
        x = C.ptr;
        n *= 2;
    }
    {
        size_t need = (size_t)n * 64 * (size_t)hh * (size_t)ww;
        if (!dbuf_fit(&A, need, err, err_size)) goto done;
        if (!conv_same_d(cuda, x, n, hh, ww, &tc->c14, &dc14, A.ptr, err, err_size))
            goto done;
        x = A.ptr;
    }
    for (i = 0; i < 3; i++) {
        if (!mem_parallel_d(cuda, &x, n, hh, ww, &tc->mem64[i], &dm64[i * 3],
                            &dm64[i * 3 + 1], &dm64[i * 3 + 2], &past, &cat,
                            &tmpa, &tmpb, x == A.ptr ? &B : &A, err,
                            err_size))
            goto done;
    }
    {
        dim3 block(16, 16);
        dim3 grid((ww * 2 + 15) / 16, (hh * 2 + 15) / 16, n * 64);
        size_t need = (size_t)n * 64 * (size_t)(hh * 2) * (size_t)(ww * 2);
        if (!dbuf_fit(x == A.ptr ? &B : &A, need, err, err_size)) goto done;
        tmp = (x == A.ptr) ? B.ptr : A.ptr;
        k_upsample2<<<grid, block>>>(x, tmp, n, 64, hh, ww);
        if (!ck(cudaGetLastError(), err, err_size, "up3")) goto done;
        x = tmp;
        hh *= 2;
        ww *= 2;
    }
    {
        size_t mid_n = (size_t)n * 128 * (size_t)hh * (size_t)ww;
        size_t out_n = (size_t)n * 2 * 64 * (size_t)hh * (size_t)ww;
        if (!dbuf_fit(&C, mid_n > C.cap ? mid_n : C.cap, err, err_size))
            goto done;
        if (!conv_same_d(cuda, x, n, hh, ww, &tc->tgrow3, &dt3, C.ptr, err,
                         err_size))
            goto done;
        if (!dbuf_fit(&A, out_n, err, err_size)) goto done;
        {
            int blocks = (int)((out_n + 255) / 256);
            k_tgrow<<<blocks < 1 ? 1 : blocks, 256>>>(C.ptr, A.ptr, n, 128, 2,
                                                      hh, ww);
            if (!ck(cudaGetLastError(), err, err_size, "tgrow3")) goto done;
        }
        x = A.ptr;
        n *= 2;
    }
    {
        size_t need = (size_t)n * 64 * (size_t)hh * (size_t)ww;
        if (!dbuf_fit(&B, need, err, err_size)) goto done;
        if (!conv_same_d(cuda, x, n, hh, ww, &tc->c20, &dc20, B.ptr, err, err_size))
            goto done;
        x = B.ptr;
        if (!relu_d(x, need, err, err_size)) goto done;
        if (!dbuf_fit(&A, need, err, err_size)) goto done;
        if (!conv_same_d(cuda, x, n, hh, ww, &tc->c22, &dc22, A.ptr, err, err_size))
            goto done;
        x = A.ptr;
    }
    {
        dim3 block(16, 16);
        dim3 grid((ww * 2 + 15) / 16, (hh * 2 + 15) / 16, n);
        size_t need = (size_t)n * 3 * (size_t)(hh * 2) * (size_t)(ww * 2);
        if (!dbuf_fit(&B, need, err, err_size)) goto done;
        k_pixel_shuffle2<<<grid, block>>>(x, B.ptr, n, hh, ww);
        if (!ck(cudaGetLastError(), err, err_size, "shuffle")) goto done;
        x = B.ptr;
        hh *= 2;
        ww *= 2;
    }

    {
        int start = trim;
        int take = want_frames;
        int f;
        if (start + take > n) {
            if (n > take)
                start = n - take;
            else {
                start = 0;
                take = n;
            }
        }
        frames->frames = take;
        frames->height = hh;
        frames->width = ww;
        frames->rgb = (float *)malloc((size_t)take * (size_t)hh * (size_t)ww *
                                      3 * sizeof(float));
        if (!frames->rgb) {
            fail(err, err_size, "TAE CUDA OOM frames");
            goto done;
        }
        {
            size_t one = (size_t)hh * (size_t)ww * 3;
            if (!dbuf_fit(&C, one * (size_t)take, err, err_size)) goto done;
            for (f = 0; f < take; f++) {
                int src = start + f;
                dim3 block(16, 16);
                dim3 grid((ww + 15) / 16, (hh + 15) / 16);
                if (src >= n) src = n - 1;
                k_nchw_to_hwc01<<<grid, block>>>(x, C.ptr + (size_t)f * one, src,
                                                 hh, ww);
                if (!ck(cudaGetLastError(), err, err_size, "hwc")) goto done;
            }
            if (!ck(cudaMemcpy(frames->rgb, C.ptr,
                               one * (size_t)take * sizeof(float),
                               cudaMemcpyDeviceToHost),
                    err, err_size, "rgb D2H"))
                goto done;
        }
    }

    if (!ck(cudaDeviceSynchronize(), err, err_size, "sync")) goto done;
    fprintf(stderr,
            "tae: decoded %d frames %dx%d from latent T=%d "
            "(GPU TAEHV；非金线、画质有损)\n",
            frames->frames, frames->width, frames->height, t);
    ok = 1;
done:
    dconv_free(&dc1);
    dconv_free(&dt1);
    dconv_free(&dc8);
    dconv_free(&dt2);
    dconv_free(&dc14);
    dconv_free(&dt3);
    dconv_free(&dc20);
    dconv_free(&dc22);
    for (i = 0; i < 9; i++) {
        dconv_free(&dm256[i]);
        dconv_free(&dm128[i]);
        dconv_free(&dm64[i]);
    }
    dbuf_free(&A);
    dbuf_free(&B);
    dbuf_free(&C);
    dbuf_free(&past);
    dbuf_free(&cat);
    dbuf_free(&tmpa);
    dbuf_free(&tmpb);
    if (!ok) {
        free(frames->rgb);
        memset(frames, 0, sizeof(*frames));
    }
    return ok;
}
