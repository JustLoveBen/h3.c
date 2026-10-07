#include "h3_tae.h"
#include "h3_tae_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int h3_tae_allow_opt_in(int non_gold, const char *vae_decode,
                        const char *tae_path, char *err, size_t err_size) {
    if (!vae_decode || !vae_decode[0] || !strcmp(vae_decode, "official"))
        return 1;
    if (strcmp(vae_decode, "tae") != 0) {
        snprintf(err, err_size, "h3: --vae-decode must be official|tae");
        return 0;
    }
    if (!non_gold) {
        snprintf(err, err_size,
                 "TAE 不是金线成片（仅 --preset turbo4|nvfp4）");
        return 0;
    }
    if (!tae_path || !tae_path[0]) {
        snprintf(err, err_size,
                 "需要 --tae 权（不静默 official）");
        return 0;
    }
    return 1;
}

int h3_tae_bind_cuda(void *ctx, void *cuda, char *err, size_t err_size) {
    h3_tae_ctx *t = (h3_tae_ctx *)ctx;
    if (!t || !t->ok) {
        snprintf(err, err_size, "TAE bind: ctx missing（不是金线成片）");
        return 0;
    }
    t->cuda = cuda;
    if (!cuda) {
        fprintf(stderr, "tae: CUDA bind skipped (host decode；非金线、有损)\n");
        return 1;
    }
#ifdef H3_TAE_HAVE_CUDA
    fprintf(stderr, "tae: CUDA bound (GPU TAEHV taeh3；非金线、画质有损)\n");
    return 1;
#else
    (void)err;
    (void)err_size;
    fprintf(stderr,
            "tae: this binary has no TAE CUDA (host fallback；非金线、有损)\n");
    return 1;
#endif
}

static void fail(char *err, size_t err_size, const char *msg) {
    if (err && err_size)
        snprintf(err, err_size, "%s", msg ? msg : "TAE error");
}

static void conv_free(tae_conv *c) {
    if (!c) return;
    free(c->w);
    free(c->b);
    memset(c, 0, sizeof(*c));
}

static void mem_free(tae_mem *m) {
    if (!m) return;
    conv_free(&m->conv0);
    conv_free(&m->conv2);
    conv_free(&m->conv4);
}

static int load_conv(const h3_st_header *hdr, const char *wname,
                     const char *bname, tae_conv *c, char *err,
                     size_t err_size) {
    size_t ne = 0;
    h3_dtype dt;
    const h3_st_tensor *tw;
    memset(c, 0, sizeof(*c));
    tw = h3_st_find(hdr, wname);
    if (!tw || tw->ndim != 4) {
        snprintf(err, err_size, "TAE missing conv %s", wname);
        return 0;
    }
    c->oc = (int)tw->shape[0];
    c->ic = (int)tw->shape[1];
    c->kh = (int)tw->shape[2];
    c->kw = (int)tw->shape[3];
    c->w = h3_st_load_f32(hdr, wname, &ne, &dt, err, err_size);
    if (!c->w) return 0;
    if (bname) {
        size_t nb = 0;
        c->b = h3_st_load_f32(hdr, bname, &nb, &dt, err, err_size);
        if (!c->b || (int)nb != c->oc) {
            snprintf(err, err_size, "TAE bias %s", bname);
            conv_free(c);
            return 0;
        }
    }
    return 1;
}

static int load_mem(const h3_st_header *hdr, int idx, tae_mem *m, char *err,
                    size_t err_size) {
    char w0[64], b0[64], w2[64], b2[64], w4[64], b4[64];
    memset(m, 0, sizeof(*m));
    snprintf(w0, sizeof(w0), "decoder.%d.conv.0.weight", idx);
    snprintf(b0, sizeof(b0), "decoder.%d.conv.0.bias", idx);
    snprintf(w2, sizeof(w2), "decoder.%d.conv.2.weight", idx);
    snprintf(b2, sizeof(b2), "decoder.%d.conv.2.bias", idx);
    snprintf(w4, sizeof(w4), "decoder.%d.conv.4.weight", idx);
    snprintf(b4, sizeof(b4), "decoder.%d.conv.4.bias", idx);
    return load_conv(hdr, w0, b0, &m->conv0, err, err_size) &&
           load_conv(hdr, w2, b2, &m->conv2, err, err_size) &&
           load_conv(hdr, w4, b4, &m->conv4, err, err_size);
}

int h3_tae_open(const char *path, void **ctx, char *err, size_t err_size) {
    h3_tae_ctx *t;
    char detail[256];
    if (ctx) *ctx = NULL;
    if (!path || !path[0]) {
        fail(err, err_size,
             "TAE 无 --tae 权（仅 turbo/nvfp4；不是金线成片；不静默 official）");
        return 0;
    }
    t = (h3_tae_ctx *)calloc(1, sizeof(*t));
    if (!t) {
        fail(err, err_size, "TAE OOM");
        return 0;
    }
    if (!h3_st_read_header(path, &t->header, detail, sizeof(detail))) {
        snprintf(err, err_size, "TAE open: %s", detail);
        free(t);
        return 0;
    }
    if (!h3_st_find(&t->header, "decoder.1.weight") ||
        !h3_st_find(&t->header, "decoder.22.weight")) {
        h3_st_free_header(&t->header);
        free(t);
        fail(err, err_size,
             "TAE 权不是 TAEHV taeh3（缺 decoder.1/22；不是金线成片）");
        return 0;
    }
    if (!load_conv(&t->header, "decoder.1.weight", "decoder.1.bias", &t->c1,
                   err, err_size) ||
        !load_mem(&t->header, 3, &t->mem256[0], err, err_size) ||
        !load_mem(&t->header, 4, &t->mem256[1], err, err_size) ||
        !load_mem(&t->header, 5, &t->mem256[2], err, err_size) ||
        !load_conv(&t->header, "decoder.7.conv.weight", NULL, &t->tgrow1, err,
                   err_size) ||
        !load_conv(&t->header, "decoder.8.weight", NULL, &t->c8, err,
                   err_size) ||
        !load_mem(&t->header, 9, &t->mem128[0], err, err_size) ||
        !load_mem(&t->header, 10, &t->mem128[1], err, err_size) ||
        !load_mem(&t->header, 11, &t->mem128[2], err, err_size) ||
        !load_conv(&t->header, "decoder.13.conv.weight", NULL, &t->tgrow2, err,
                   err_size) ||
        !load_conv(&t->header, "decoder.14.weight", NULL, &t->c14, err,
                   err_size) ||
        !load_mem(&t->header, 15, &t->mem64[0], err, err_size) ||
        !load_mem(&t->header, 16, &t->mem64[1], err, err_size) ||
        !load_mem(&t->header, 17, &t->mem64[2], err, err_size) ||
        !load_conv(&t->header, "decoder.19.conv.weight", NULL, &t->tgrow3, err,
                   err_size) ||
        !load_conv(&t->header, "decoder.20.weight", NULL, &t->c20, err,
                   err_size) ||
        !load_conv(&t->header, "decoder.22.weight", "decoder.22.bias", &t->c22,
                   err, err_size)) {
        h3_tae_close(t);
        return 0;
    }
    if (t->c1.ic != 24 || t->c1.oc != 256 || t->c1.kh != 3 || t->c1.kw != 3) {
        snprintf(err, err_size,
                 "TAE latent C/shape 对不上官方 H3（decoder.1 要 [256,24,3,3]，"
                 "got [%d,%d,%d,%d]；禁止静默 official）",
                 t->c1.oc, t->c1.ic, t->c1.kh, t->c1.kw);
        h3_tae_close(t);
        return 0;
    }
    if (t->c22.oc != 12 || t->c22.ic != 64) {
        snprintf(err, err_size,
                 "TAE 出口对不上 taeh3 patch=2（decoder.22 要 [12,64,*,*]，"
                 "got [%d,%d]；禁止静默 official）",
                 t->c22.oc, t->c22.ic);
        h3_tae_close(t);
        return 0;
    }
    if (t->tgrow2.oc != t->tgrow2.ic * 2 || t->tgrow3.oc != t->tgrow3.ic * 2) {
        snprintf(err, err_size,
                 "TAE 时间上采样不是 taeh3 (1,2,2)（禁止静默 official）");
        h3_tae_close(t);
        return 0;
    }
    t->latent_ch = 24;
    t->spatial_ratio = 16;
    t->t_upscale = 4;
    t->ok = 1;
    if (ctx) *ctx = t;
    fprintf(stderr,
            "tae: opened %s (TAEHV taeh3 C=24 spatial=16 t_up=4；"
            "非金线、画质有损)\n",
            path);
    return 1;
}

void h3_tae_close(void *ctx) {
    h3_tae_ctx *t = (h3_tae_ctx *)ctx;
    int i;
    if (!t) return;
    conv_free(&t->c1);
    for (i = 0; i < 3; i++) mem_free(&t->mem256[i]);
    conv_free(&t->tgrow1);
    conv_free(&t->c8);
    for (i = 0; i < 3; i++) mem_free(&t->mem128[i]);
    conv_free(&t->tgrow2);
    conv_free(&t->c14);
    for (i = 0; i < 3; i++) mem_free(&t->mem64[i]);
    conv_free(&t->tgrow3);
    conv_free(&t->c20);
    conv_free(&t->c22);
    h3_st_free_header(&t->header);
    free(t);
}

static float *nchw_alloc(int n, int c, int h, int w) {
    return (float *)calloc((size_t)n * (size_t)c * (size_t)h * (size_t)w,
                           sizeof(float));
}

/* Same-size 3x3 pad1 or 1x1. ih/iw known. */
static int conv_same(const float *in, int n, int ih, int iw, const tae_conv *cv,
                     float **outp) {
    int stride = 1, pad = (cv->kh == 3) ? 1 : 0;
    int oh = ih, ow = iw;
    float *out = nchw_alloc(n, cv->oc, oh, ow);
    if (!out) return 0;
    {
        int ni, oc, ic, oy, ox, ky, kx;
        for (ni = 0; ni < n; ni++)
            for (oc = 0; oc < cv->oc; oc++)
                for (oy = 0; oy < oh; oy++)
                    for (ox = 0; ox < ow; ox++) {
                        float acc = cv->b ? cv->b[oc] : 0.0f;
                        for (ic = 0; ic < cv->ic; ic++)
                            for (ky = 0; ky < cv->kh; ky++) {
                                int iy = oy * stride + ky - pad;
                                for (kx = 0; kx < cv->kw; kx++) {
                                    int ix = ox * stride + kx - pad;
                                    if (iy < 0 || iy >= ih || ix < 0 ||
                                        ix >= iw)
                                        continue;
                                    acc += in[(((size_t)ni * (size_t)cv->ic +
                                                (size_t)ic) *
                                                   (size_t)ih +
                                               (size_t)iy) *
                                                  (size_t)iw +
                                              (size_t)ix] *
                                           cv->w[(((size_t)oc * (size_t)cv->ic +
                                                   (size_t)ic) *
                                                      (size_t)cv->kh +
                                                  (size_t)ky) *
                                                     (size_t)cv->kw +
                                                 (size_t)kx];
                                }
                            }
                        out[(((size_t)ni * (size_t)cv->oc + (size_t)oc) *
                                 (size_t)oh +
                             (size_t)oy) *
                                (size_t)ow +
                            (size_t)ox] = acc;
                    }
    }
    *outp = out;
    return 1;
}

static void relu_inplace(float *x, size_t n) {
    size_t i;
    for (i = 0; i < n; i++)
        if (x[i] < 0.0f) x[i] = 0.0f;
}

static void clamp_tanh3(float *x, size_t n) {
    size_t i;
    for (i = 0; i < n; i++)
        x[i] = tanhf(x[i] / 3.0f) * 3.0f;
}

static int upsample2(const float *in, int n, int c, int h, int w,
                     float **outp) {
    int oh = h * 2, ow = w * 2, ni, ch, y, x;
    float *out = nchw_alloc(n, c, oh, ow);
    if (!out) return 0;
    for (ni = 0; ni < n; ni++)
        for (ch = 0; ch < c; ch++)
            for (y = 0; y < oh; y++)
                for (x = 0; x < ow; x++)
                    out[(((size_t)ni * (size_t)c + (size_t)ch) * (size_t)oh +
                         (size_t)y) *
                            (size_t)ow +
                        (size_t)x] =
                        in[(((size_t)ni * (size_t)c + (size_t)ch) * (size_t)h +
                            (size_t)(y / 2)) *
                               (size_t)w +
                           (size_t)(x / 2)];
    *outp = out;
    return 1;
}

static int tgrow(const float *in, int n, int h, int w, const tae_conv *cv,
                 int stride, float **outp, int *n_out) {
    float *mid = NULL;
    float *out;
    int oc = cv->oc, c_out, ni, t, ch, y, x;
    if (!conv_same(in, n, h, w, cv, &mid)) return 0;
    if (stride <= 1) {
        *outp = mid;
        *n_out = n;
        return 1;
    }
    /* mid: [N, C*stride, H, W] → [N*stride, C, H, W] */
    c_out = oc / stride;
    out = nchw_alloc(n * stride, c_out, h, w);
    if (!out) {
        free(mid);
        return 0;
    }
    for (ni = 0; ni < n; ni++)
        for (t = 0; t < stride; t++)
            for (ch = 0; ch < c_out; ch++)
                for (y = 0; y < h; y++)
                    for (x = 0; x < w; x++) {
                        int src_c = t * c_out + ch;
                        out[((((size_t)ni * (size_t)stride + (size_t)t) *
                                  (size_t)c_out +
                              (size_t)ch) *
                                 (size_t)h +
                             (size_t)y) *
                                (size_t)w +
                            (size_t)x] =
                            mid[(((size_t)ni * (size_t)oc + (size_t)src_c) *
                                     (size_t)h +
                                 (size_t)y) *
                                    (size_t)w +
                                (size_t)x];
                    }
    free(mid);
    *outp = out;
    *n_out = n * stride;
    return 1;
}

static int memblock(const float *x, const float *past, int n, int h, int w,
                    const tae_mem *m, float **outp) {
    /* cat [x, past] on C → conv0 → relu → conv2 → relu → conv4 + x → relu */
    int c = m->conv2.oc;
    int c_in = m->conv0.ic;
    float *cat = NULL, *a = NULL, *b = NULL, *d = NULL;
    size_t plane = (size_t)h * (size_t)w;
    int ni, ch, i;
    if (c_in != c * 2) return 0;
    cat = nchw_alloc(n, c_in, h, w);
    if (!cat) return 0;
    for (ni = 0; ni < n; ni++) {
        memcpy(cat + ((size_t)ni * (size_t)c_in) * plane,
               x + ((size_t)ni * (size_t)c) * plane, (size_t)c * plane * 4);
        memcpy(cat + ((size_t)ni * (size_t)c_in + (size_t)c) * plane,
               past + ((size_t)ni * (size_t)c) * plane, (size_t)c * plane * 4);
    }
    if (!conv_same(cat, n, h, w, &m->conv0, &a)) {
        free(cat);
        return 0;
    }
    free(cat);
    relu_inplace(a, (size_t)n * (size_t)c * plane);
    if (!conv_same(a, n, h, w, &m->conv2, &b)) {
        free(a);
        return 0;
    }
    free(a);
    relu_inplace(b, (size_t)n * (size_t)c * plane);
    if (!conv_same(b, n, h, w, &m->conv4, &d)) {
        free(b);
        return 0;
    }
    free(b);
    for (i = 0, ni = n * c * (int)plane; i < ni; i++) {
        d[i] += x[i];
        if (d[i] < 0.0f) d[i] = 0.0f;
    }
    *outp = d;
    (void)ch;
    return 1;
}

static int mem_parallel(float **px, int n, int h, int w, const tae_mem *m) {
    float *x = *px;
    int c = m->conv2.oc;
    size_t plane = (size_t)c * (size_t)h * (size_t)w;
    float *past = nchw_alloc(n, c, h, w);
    float *y = NULL;
    int t;
    if (!past) return 0;
    /* past[t] = x[t-1], past[0]=0 */
    for (t = 1; t < n; t++)
        memcpy(past + (size_t)t * plane, x + (size_t)(t - 1) * plane,
               plane * sizeof(float));
    if (!memblock(x, past, n, h, w, m, &y)) {
        free(past);
        return 0;
    }
    free(past);
    free(x);
    *px = y;
    return 1;
}

static int pixel_shuffle2(const float *in, int n, int h, int w, float **outp) {
    /* in C=12 → out C=3, 2x spatial */
    int oh = h * 2, ow = w * 2, ni, c, y, x;
    float *out = nchw_alloc(n, 3, oh, ow);
    if (!out) return 0;
    for (ni = 0; ni < n; ni++)
        for (c = 0; c < 3; c++)
            for (y = 0; y < oh; y++)
                for (x = 0; x < ow; x++) {
                    int rh = y % 2, rw = x % 2;
                    int src_c = c * 4 + rh * 2 + rw;
                    out[(((size_t)ni * 3 + (size_t)c) * (size_t)oh +
                         (size_t)y) *
                            (size_t)ow +
                        (size_t)x] =
                        in[(((size_t)ni * 12 + (size_t)src_c) * (size_t)h +
                            (size_t)(y / 2)) *
                               (size_t)w +
                           (size_t)(x / 2)];
                }
    *outp = out;
    return 1;
}

int h3_tae_decode_frames(void *ctx, const float *latent_chw, int t, int h,
                         int w, int want_frames, h3_vae_int8_frames *frames,
                         char *err, size_t err_size) {
    h3_tae_ctx *tc = (h3_tae_ctx *)ctx;
    float *x = NULL, *tmp = NULL, *rgb = NULL;
    int n, ch, i, f, hh, ww;
    int trim = 3;
    size_t plane24;
    if (!frames) {
        fail(err, err_size, "TAE frames null");
        return 0;
    }
    memset(frames, 0, sizeof(*frames));
    if (!tc || !tc->ok || !latent_chw || t < 1 || h < 1 || w < 1) {
        fail(err, err_size, "TAE decode args（不是金线成片）");
        return 0;
    }
    if (tc->latent_ch != 24 || tc->spatial_ratio != 16) {
        fail(err, err_size,
             "TAE latent/spatial 与官方 H3 对不上（禁止静默 official）");
        return 0;
    }
#ifdef H3_TAE_HAVE_CUDA
    if (tc->cuda)
        return h3_tae_cuda_decode_frames(tc, latent_chw, t, h, w, want_frames,
                                         frames, err, err_size);
#endif
    if (want_frames < 1) want_frames = t * 4;
    /* CHW [24,T,H,W] → NCHW [T,24,H,W] */
    x = nchw_alloc(t, 24, h, w);
    if (!x) {
        fail(err, err_size, "TAE OOM latent");
        return 0;
    }
    plane24 = (size_t)h * (size_t)w;
    for (i = 0; i < t; i++)
        for (ch = 0; ch < 24; ch++)
            memcpy(x + ((size_t)i * 24 + (size_t)ch) * plane24,
                   latent_chw + ((size_t)ch * (size_t)t + (size_t)i) * plane24,
                   plane24 * sizeof(float));
    n = t;
    clamp_tanh3(x, (size_t)n * 24 * plane24);
    if (!conv_same(x, n, h, w, &tc->c1, &tmp)) goto oom;
    free(x);
    x = tmp;
    tmp = NULL;
    relu_inplace(x, (size_t)n * 256 * plane24);
    for (i = 0; i < 3; i++)
        if (!mem_parallel(&x, n, h, w, &tc->mem256[i])) goto oom;
    if (!upsample2(x, n, 256, h, w, &tmp)) goto oom;
    free(x);
    x = tmp;
    tmp = NULL;
    h *= 2;
    w *= 2;
    if (!tgrow(x, n, h, w, &tc->tgrow1, 1, &tmp, &n)) goto oom;
    free(x);
    x = tmp;
    tmp = NULL;
    if (!conv_same(x, n, h, w, &tc->c8, &tmp)) goto oom;
    free(x);
    x = tmp;
    tmp = NULL;
    for (i = 0; i < 3; i++)
        if (!mem_parallel(&x, n, h, w, &tc->mem128[i])) goto oom;
    if (!upsample2(x, n, 128, h, w, &tmp)) goto oom;
    free(x);
    x = tmp;
    tmp = NULL;
    h *= 2;
    w *= 2;
    if (!tgrow(x, n, h, w, &tc->tgrow2, 2, &tmp, &n)) goto oom;
    free(x);
    x = tmp;
    tmp = NULL;
    if (!conv_same(x, n, h, w, &tc->c14, &tmp)) goto oom;
    free(x);
    x = tmp;
    tmp = NULL;
    for (i = 0; i < 3; i++)
        if (!mem_parallel(&x, n, h, w, &tc->mem64[i])) goto oom;
    if (!upsample2(x, n, 64, h, w, &tmp)) goto oom;
    free(x);
    x = tmp;
    tmp = NULL;
    h *= 2;
    w *= 2;
    if (!tgrow(x, n, h, w, &tc->tgrow3, 2, &tmp, &n)) goto oom;
    free(x);
    x = tmp;
    tmp = NULL;
    if (!conv_same(x, n, h, w, &tc->c20, &tmp)) goto oom;
    free(x);
    x = tmp;
    tmp = NULL;
    relu_inplace(x, (size_t)n * 64 * (size_t)h * (size_t)w);
    if (!conv_same(x, n, h, w, &tc->c22, &tmp)) goto oom;
    free(x);
    x = tmp;
    tmp = NULL;
    if (!pixel_shuffle2(x, n, h, w, &tmp)) goto oom;
    free(x);
    x = tmp;
    tmp = NULL;
    h *= 2;
    w *= 2;
    /* RGB NCHW [n,3,H,W] in ~[0,1] after clamp */
    {
        int start = trim;
        int take = want_frames;
        if (start + take > n) {
            if (n > take)
                start = n - take;
            else {
                start = 0;
                take = n;
            }
        }
        frames->frames = take;
        frames->height = h;
        frames->width = w;
        frames->rgb = (float *)malloc((size_t)take * (size_t)h * (size_t)w * 3 *
                                      sizeof(float));
        if (!frames->rgb) goto oom;
        for (f = 0; f < take; f++) {
            int src = start + f;
            if (src >= n) src = n - 1;
            for (hh = 0; hh < h; hh++)
                for (ww = 0; ww < w; ww++)
                    for (ch = 0; ch < 3; ch++) {
                        float v = x[(((size_t)src * 3 + (size_t)ch) *
                                         (size_t)h +
                                     (size_t)hh) *
                                        (size_t)w +
                                    (size_t)ww];
                        if (v < 0.0f) v = 0.0f;
                        if (v > 1.0f) v = 1.0f;
                        frames->rgb[(((size_t)f * (size_t)h + (size_t)hh) *
                                         (size_t)w +
                                     (size_t)ww) *
                                        3 +
                                    (size_t)ch] = v;
                    }
        }
    }
    free(x);
    fprintf(stderr,
            "tae: decoded %d frames %dx%d from latent T=%d (host TAEHV；非金线)\n",
            frames->frames, frames->width, frames->height, t);
    return 1;
oom:
    free(x);
    free(tmp);
    fail(err, err_size, "TAE decode OOM/layer fail（不是金线成片）");
    return 0;
}

int h3_tae_decode_f32(void *ctx, const float *latent, int t, int h, int w,
                      uint8_t *rgb_out, char *err, size_t err_size) {
    h3_vae_int8_frames fr;
    size_t i, n;
    memset(&fr, 0, sizeof(fr));
    if (!h3_tae_decode_frames(ctx, latent, t, h, w, t * 4, &fr, err,
                              err_size))
        return 0;
    n = (size_t)fr.frames * (size_t)fr.height * (size_t)fr.width * 3;
    if (rgb_out) {
        for (i = 0; i < n; i++) {
            float v = fr.rgb[i] * 255.0f;
            if (v < 0.0f) v = 0.0f;
            if (v > 255.0f) v = 255.0f;
            rgb_out[i] = (uint8_t)(v + 0.5f);
        }
    }
    free(fr.rgb);
    return 1;
}
