#include "h3_gpu_cuda.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail(char *err, size_t err_size, const char *msg) {
    if (err && err_size)
        snprintf(err, err_size, "%s", msg ? msg : "upscaler cuda");
    return 0;
}

static int ck(cudaError_t e, char *err, size_t err_size, const char *what) {
    if (e == cudaSuccess)
        return 1;
    if (err && err_size)
        snprintf(err, err_size, "%s: %s", what ? what : "cuda",
                 cudaGetErrorString(e));
    return 0;
}

__global__ void k_conv3d(const float *in, float *out, const float *w,
                         const float *b, int n, int ic, int oc, int t, int h,
                         int ws, int kd, int kh, int kw, int pad_t, int pad_h,
                         int pad_w, int groups) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t nout = (size_t)n * (size_t)oc * (size_t)t * (size_t)h * (size_t)ws;
    int ic_g, oc_g, g, oc_i, n_i, ot, oh, ow, ic0, ii, kd_i, kh_i, kw_i;
    float acc;
    if (tid >= nout)
        return;
    {
        size_t rem = tid;
        ow = (int)(rem % (size_t)ws);
        rem /= (size_t)ws;
        oh = (int)(rem % (size_t)h);
        rem /= (size_t)h;
        ot = (int)(rem % (size_t)t);
        rem /= (size_t)t;
        oc_i = (int)(rem % (size_t)oc);
        n_i = (int)(rem / (size_t)oc);
    }
    oc_g = oc / groups;
    ic_g = ic / groups;
    g = oc_i / oc_g;
    ic0 = g * ic_g;
    acc = b ? b[oc_i] : 0.0f;
    for (ii = 0; ii < ic_g; ii++) {
        int ic_i = ic0 + ii;
        for (kd_i = 0; kd_i < kd; kd_i++) {
            int it = ot + kd_i - pad_t;
            if (it < 0 || it >= t)
                continue;
            for (kh_i = 0; kh_i < kh; kh_i++) {
                int ih = oh + kh_i - pad_h;
                if (ih < 0 || ih >= h)
                    continue;
                for (kw_i = 0; kw_i < kw; kw_i++) {
                    int iw = ow + kw_i - pad_w;
                    size_t in_i, w_i;
                    if (iw < 0 || iw >= ws)
                        continue;
                    in_i = ((((size_t)n_i * (size_t)ic + (size_t)ic_i) *
                                 (size_t)t +
                             (size_t)it) *
                                (size_t)h +
                            (size_t)ih) *
                               (size_t)ws +
                           (size_t)iw;
                    w_i = (((((size_t)oc_i * (size_t)ic_g + (size_t)ii) *
                                 (size_t)kd +
                             (size_t)kd_i) *
                                (size_t)kh +
                            (size_t)kh_i) *
                               (size_t)kw +
                           (size_t)kw_i);
                    acc += in[in_i] * w[w_i];
                }
            }
        }
    }
    out[tid] = acc;
}

__global__ void k_silu(float *x, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        x[i] = v / (1.0f + expf(-v));
    }
}

__global__ void k_add(float *y, const float *a, const float *b, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a[i] + b[i];
}

__global__ void k_gn_stats(const float *x, float *mean, float *var, int n,
                           int c, int spat, int groups) {
    int g = (int)blockIdx.x;
    int cg = c / groups;
    int n_i = g / groups;
    int gi = g % groups;
    int c0 = gi * cg;
    double s = 0.0, s2 = 0.0;
    int count = cg * spat;
    int i;
    if (n_i >= n)
        return;
    for (i = threadIdx.x; i < count; i += (int)blockDim.x) {
        int cc = c0 + i / spat;
        int sp = i % spat;
        size_t ix = ((size_t)n_i * (size_t)c + (size_t)cc) * (size_t)spat +
                    (size_t)sp;
        double v = (double)x[ix];
        s += v;
        s2 += v * v;
    }
    __shared__ double sh[512], sh2[512];
    sh[threadIdx.x] = s;
    sh2[threadIdx.x] = s2;
    __syncthreads();
    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) {
            sh[threadIdx.x] += sh[threadIdx.x + stride];
            sh2[threadIdx.x] += sh2[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        double m = sh[0] / (double)count;
        double v = sh2[0] / (double)count - m * m;
        if (v < 0.0)
            v = 0.0;
        mean[g] = (float)m;
        var[g] = (float)v;
    }
}

__global__ void k_gn_apply(float *x, const float *mean, const float *var,
                           const float *gamma, const float *beta, int n, int c,
                           int spat, int groups, float eps) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t ntot = (size_t)n * (size_t)c * (size_t)spat;
    int n_i, cc, gi, g;
    float inv;
    if (tid >= ntot)
        return;
    {
        size_t rem = tid;
        rem = tid / (size_t)spat;
        cc = (int)(rem % (size_t)c);
        n_i = (int)(rem / (size_t)c);
    }
    gi = cc / (c / groups);
    g = n_i * groups + gi;
    inv = rsqrtf(var[g] + eps);
    x[tid] = (x[tid] - mean[g]) * inv * gamma[cc] + beta[cc];
}

__global__ void k_adaln(float *x, const float *scale, const float *shift,
                        int n, int c, int spat) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t ntot = (size_t)n * (size_t)c * (size_t)spat;
    int cc;
    if (tid >= ntot)
        return;
    cc = (int)((tid / (size_t)spat) % (size_t)c);
    x[tid] = x[tid] * (1.0f + scale[cc]) + shift[cc];
}

__global__ void k_trilinear(const float *in, float *out, int n, int c, int it,
                            int ih, int iw, int ot, int oh, int ow) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t nout = (size_t)n * (size_t)c * (size_t)ot * (size_t)oh * (size_t)ow;
    int n_i, cc, tt, hh, ww;
    float ft, fh, fw;
    int t0, h0, w0, t1, h1, w1;
    if (tid >= nout)
        return;
    {
        size_t rem = tid;
        ww = (int)(rem % (size_t)ow);
        rem /= (size_t)ow;
        hh = (int)(rem % (size_t)oh);
        rem /= (size_t)oh;
        tt = (int)(rem % (size_t)ot);
        rem /= (size_t)ot;
        cc = (int)(rem % (size_t)c);
        n_i = (int)(rem / (size_t)c);
    }
    ft = (ot == 1) ? 0.0f
                   : ((float)tt + 0.5f) * (float)it / (float)ot - 0.5f;
    fh = (oh == 1) ? 0.0f
                   : ((float)hh + 0.5f) * (float)ih / (float)oh - 0.5f;
    fw = (ow == 1) ? 0.0f
                   : ((float)ww + 0.5f) * (float)iw / (float)ow - 0.5f;
    if (ft < 0.0f)
        ft = 0.0f;
    if (fh < 0.0f)
        fh = 0.0f;
    if (fw < 0.0f)
        fw = 0.0f;
    if (ft > (float)(it - 1))
        ft = (float)(it - 1);
    if (fh > (float)(ih - 1))
        fh = (float)(ih - 1);
    if (fw > (float)(iw - 1))
        fw = (float)(iw - 1);
    t0 = (int)ft;
    h0 = (int)fh;
    w0 = (int)fw;
    t1 = t0 + 1 < it ? t0 + 1 : t0;
    h1 = h0 + 1 < ih ? h0 + 1 : h0;
    w1 = w0 + 1 < iw ? w0 + 1 : w0;
    {
        float dt = ft - (float)t0, dh = fh - (float)h0, dw = fw - (float)w0;
        float acc = 0.0f;
        int a, b, d;
        for (a = 0; a < 2; a++) {
            int ti = a ? t1 : t0;
            float wt = a ? dt : (1.0f - dt);
            for (b = 0; b < 2; b++) {
                int hi = b ? h1 : h0;
                float wh = b ? dh : (1.0f - dh);
                for (d = 0; d < 2; d++) {
                    int wi = d ? w1 : w0;
                    float wwgt = d ? dw : (1.0f - dw);
                    size_t ix =
                        ((((size_t)n_i * (size_t)c + (size_t)cc) * (size_t)it +
                          (size_t)ti) *
                             (size_t)ih +
                         (size_t)hi) *
                            (size_t)iw +
                        (size_t)wi;
                    acc += in[ix] * wt * wh * wwgt;
                }
            }
        }
        out[tid] = acc;
    }
}

__global__ void k_scale_bias(float *x, const float *mean, const float *stdv,
                             int inv, int n, int c, int spat) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t ntot = (size_t)n * (size_t)c * (size_t)spat;
    int cc;
    if (tid >= ntot)
        return;
    cc = (int)((tid / (size_t)spat) % (size_t)c);
    if (inv)
        x[tid] = (x[tid] - mean[cc]) / stdv[cc];
    else
        x[tid] = x[tid] * stdv[cc] + mean[cc];
}

#ifdef __cplusplus
extern "C" {
#endif

static h3_cuda *g_ups_cuda;
static int g_ups_conv_logged;
static int g_ups_naive_logged;
static float *g_gn_mean;
static float *g_gn_var;
static size_t g_gn_cap;

void h3_ups_bind_cuda(h3_cuda *cuda) { g_ups_cuda = cuda; }

int h3_ups_conv3d_nchw(const float *in, float *out, const float *w,
                       const float *b, int n, int ic, int oc, int t, int h,
                       int w_s, int kd, int kh, int kw, int pad_t, int pad_h,
                       int pad_w, int groups, char *err, size_t err_size) {
    size_t nout;
    int threads = 256, blocks;
    const char *force;
    if (!in || !out || !w || n < 1 || ic < 1 || oc < 1 || t < 1 || h < 1 ||
        w_s < 1 || groups < 1 || (ic % groups) || (oc % groups))
        return fail(err, err_size, "conv3d bad dims");
    force = getenv("H3_LBH_CONV");
    if (!(force && strcmp(force, "naive") == 0) && g_ups_cuda) {
        char local[256];
        if (h3_cuda_conv3d_ncdhw_bias_dd(g_ups_cuda, in, w, b, out, n, ic, oc,
                                         t, h, w_s, kd, kh, kw, pad_t, pad_h,
                                         pad_w, groups, local, sizeof(local))) {
            if (!g_ups_conv_logged) {
                fprintf(stderr, "selflift: conv3d path=cudnn\n");
                fflush(stderr);
                g_ups_conv_logged = 1;
            }
            return ck(cudaGetLastError(), err, err_size, "conv3d cudnn");
        }
        if (!g_ups_naive_logged) {
            fprintf(stderr,
                    "selflift: conv3d cudnn miss (%s); fallback naive ic=%d "
                    "oc=%d t=%d %dx%d k=%dx%dx%d g=%d\n",
                    local, ic, oc, t, h, w_s, kd, kh, kw, groups);
            fflush(stderr);
            g_ups_naive_logged = 1;
        }
    }
    if (!g_ups_conv_logged && !g_ups_naive_logged) {
        fprintf(stderr, "selflift: conv3d path=naive\n");
        fflush(stderr);
        g_ups_conv_logged = 1;
    }
    nout = (size_t)n * (size_t)oc * (size_t)t * (size_t)h * (size_t)w_s;
    blocks = (int)((nout + (size_t)threads - 1) / (size_t)threads);
    k_conv3d<<<blocks, threads>>>(in, out, w, b, n, ic, oc, t, h, w_s, kd, kh,
                                  kw, pad_t, pad_h, pad_w, groups);
    return ck(cudaGetLastError(), err, err_size, "conv3d");
}

int h3_ups_groupnorm(float *x, const float *gamma, const float *beta, int n,
                     int c, int t, int h, int w, int groups, float eps,
                     char *err, size_t err_size) {
    int spat, ng, threads = 256;
    size_t ntot, need;
    if (!x || !gamma || !beta || n < 1 || c < 1 || (c % groups))
        return fail(err, err_size, "groupnorm bad dims");
    spat = t * h * w;
    ng = n * groups;
    ntot = (size_t)n * (size_t)c * (size_t)spat;
    need = (size_t)ng * sizeof(float);
    if (g_gn_cap < need) {
        cudaFree(g_gn_mean);
        cudaFree(g_gn_var);
        g_gn_mean = NULL;
        g_gn_var = NULL;
        g_gn_cap = 0;
        if (!ck(cudaMalloc((void **)&g_gn_mean, need), err, err_size,
                "gn mean") ||
            !ck(cudaMalloc((void **)&g_gn_var, need), err, err_size,
                "gn var")) {
            cudaFree(g_gn_mean);
            cudaFree(g_gn_var);
            g_gn_mean = NULL;
            g_gn_var = NULL;
            return 0;
        }
        g_gn_cap = need;
    }
    k_gn_stats<<<ng, threads>>>(x, g_gn_mean, g_gn_var, n, c, spat, groups);
    if (!ck(cudaGetLastError(), err, err_size, "gn stats"))
        return 0;
    {
        int blocks = (int)((ntot + (size_t)threads - 1) / (size_t)threads);
        k_gn_apply<<<blocks, threads>>>(x, g_gn_mean, g_gn_var, gamma, beta, n,
                                        c, spat, groups, eps);
    }
    return ck(cudaGetLastError(), err, err_size, "gn apply");
}

int h3_ups_silu(float *x, size_t n, char *err, size_t err_size) {
    int threads = 256;
    int blocks;
    if (!x)
        return fail(err, err_size, "silu null");
    if (n == 0)
        return 1;
    blocks = (int)((n + (size_t)threads - 1) / (size_t)threads);
    k_silu<<<blocks, threads>>>(x, n);
    return ck(cudaGetLastError(), err, err_size, "silu");
}

int h3_ups_add(float *y, const float *a, const float *b, size_t n, char *err,
               size_t err_size) {
    int threads = 256;
    int blocks;
    if (!y || !a || !b)
        return fail(err, err_size, "add null");
    if (n == 0)
        return 1;
    blocks = (int)((n + (size_t)threads - 1) / (size_t)threads);
    k_add<<<blocks, threads>>>(y, a, b, n);
    return ck(cudaGetLastError(), err, err_size, "add");
}

int h3_ups_adaln(float *x, const float *scale, const float *shift, int n,
                 int c, int t, int h, int w, char *err, size_t err_size) {
    int spat = t * h * w;
    size_t ntot = (size_t)n * (size_t)c * (size_t)spat;
    int threads = 256;
    int blocks;
    if (!x || !scale || !shift)
        return fail(err, err_size, "adaln null");
    blocks = (int)((ntot + (size_t)threads - 1) / (size_t)threads);
    k_adaln<<<blocks, threads>>>(x, scale, shift, n, c, spat);
    return ck(cudaGetLastError(), err, err_size, "adaln");
}

int h3_ups_trilinear(const float *in, float *out, int n, int c, int t, int ih,
                     int iw, int oh, int ow, char *err, size_t err_size) {
    size_t nout;
    int threads = 256, blocks;
    if (!in || !out)
        return fail(err, err_size, "trilinear null");
    nout = (size_t)n * (size_t)c * (size_t)t * (size_t)oh * (size_t)ow;
    blocks = (int)((nout + (size_t)threads - 1) / (size_t)threads);
    k_trilinear<<<blocks, threads>>>(in, out, n, c, t, ih, iw, t, oh, ow);
    return ck(cudaGetLastError(), err, err_size, "trilinear");
}

int h3_ups_scale_bias(float *x, const float *mean, const float *stdv, int inv,
                      int n, int c, int t, int h, int w, char *err,
                      size_t err_size) {
    int spat = t * h * w;
    size_t ntot = (size_t)n * (size_t)c * (size_t)spat;
    int threads = 256, blocks;
    if (!x || !mean || !stdv)
        return fail(err, err_size, "scale_bias null");
    blocks = (int)((ntot + (size_t)threads - 1) / (size_t)threads);
    k_scale_bias<<<blocks, threads>>>(x, mean, stdv, inv, n, c, spat);
    return ck(cudaGetLastError(), err, err_size, "scale_bias");
}

#ifdef __cplusplus
}
#endif
