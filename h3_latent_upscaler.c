#include "h3_latent_upscaler.h"
#include "h3_latent_upscaler_internal.h"
#include "h3_safetensors.h"

#include <cuda_runtime_api.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define H3_UPS_MAX_BLOCKS 48
#define H3_UPS_GN 32
#define H3_UPS_EPS 1e-5f
#define H3_UPS_EMB 64

typedef enum { H3_UPS_RES = 1, H3_UPS_TEMP = 2 } h3_ups_kind;

typedef struct {
    h3_ups_kind kind;
    float *in_gn_w, *in_gn_b;
    float *in_conv_w, *in_conv_b;
    float *emb_w, *emb_b;
    float *out_gn_w, *out_gn_b;
    float *out_conv_w, *out_conv_b;
    float *tn_w, *tn_b;
    float *dw_w, *dw_b;
    float *pw_w, *pw_b;
} h3_ups_block;

#define H3_UPS_SLOTS 12

struct h3_latent_upscaler {
    int in_ch, hid, tk, n_in, n_out;
    float *conv_in_w, *conv_in_b;
    float *emb0_w, *emb0_b, *emb2_w, *emb2_b;
    float *norm_out_w, *norm_out_b;
    float *conv_out_w, *conv_out_b;
    h3_ups_block in_b[H3_UPS_MAX_BLOCKS];
    h3_ups_block out_b[H3_UPS_MAX_BLOCKS];
    /* Persistent device WS: skip per-forward cudaMalloc on hot clips. */
    float *d_in, *d_x, *d_h, *d_o, *d_w, *d_mean, *d_std;
    size_t cap_in, cap_x, cap_h, cap_o, cap_w, cap_mean, cap_std;
    float *d_slot[H3_UPS_SLOTS];
    size_t slot_cap[H3_UPS_SLOTS];
    int slot_i;
};

static int fail(char *err, size_t err_size, const char *msg) {
    if (err && err_size)
        snprintf(err, err_size, "%s", msg ? msg : "latent upscaler");
    return 0;
}

static float *must_load(const h3_st_header *hdr, const char *name, size_t want,
                        char *err, size_t err_size) {
    size_t got = 0;
    h3_dtype dt = H3_DTYPE_UNKNOWN;
    float *p = h3_st_load_f32(hdr, name, &got, &dt, err, err_size);
    if (!p)
        return NULL;
    if (got != want) {
        if (err && err_size)
            snprintf(err, err_size, "upscaler %s elems %zu want %zu", name, got,
                     want);
        free(p);
        return NULL;
    }
    return p;
}

static int parse_idx(const char *name, const char *pref, int *idx) {
    const char *p;
    char *end = NULL;
    long v;
    size_t n = strlen(pref);
    if (strncmp(name, pref, n) != 0)
        return 0;
    p = name + n;
    v = strtol(p, &end, 10);
    if (end == p || *end != '.')
        return 0;
    if (v < 0 || v >= H3_UPS_MAX_BLOCKS)
        return 0;
    *idx = (int)v;
    return 1;
}

static void free_block(h3_ups_block *b) {
    if (!b)
        return;
    free(b->in_gn_w);
    free(b->in_gn_b);
    free(b->in_conv_w);
    free(b->in_conv_b);
    free(b->emb_w);
    free(b->emb_b);
    free(b->out_gn_w);
    free(b->out_gn_b);
    free(b->out_conv_w);
    free(b->out_conv_b);
    free(b->tn_w);
    free(b->tn_b);
    free(b->dw_w);
    free(b->dw_b);
    free(b->pw_w);
    free(b->pw_b);
    memset(b, 0, sizeof(*b));
}

static void ups_ws_release(h3_latent_upscaler *u) {
    int s;
    if (!u)
        return;
    cudaFree(u->d_in);
    cudaFree(u->d_x);
    cudaFree(u->d_h);
    cudaFree(u->d_o);
    cudaFree(u->d_w);
    cudaFree(u->d_mean);
    cudaFree(u->d_std);
    u->d_in = u->d_x = u->d_h = u->d_o = u->d_w = u->d_mean = u->d_std = NULL;
    u->cap_in = u->cap_x = u->cap_h = u->cap_o = u->cap_w = 0;
    u->cap_mean = u->cap_std = 0;
    for (s = 0; s < H3_UPS_SLOTS; s++) {
        cudaFree(u->d_slot[s]);
        u->d_slot[s] = NULL;
        u->slot_cap[s] = 0;
    }
    u->slot_i = 0;
}

void h3_latent_upscaler_release_device(h3_latent_upscaler *u) {
    ups_ws_release(u);
}

void h3_latent_upscaler_close(h3_latent_upscaler *u) {
    int i;
    if (!u)
        return;
    ups_ws_release(u);
    free(u->conv_in_w);
    free(u->conv_in_b);
    free(u->emb0_w);
    free(u->emb0_b);
    free(u->emb2_w);
    free(u->emb2_b);
    free(u->norm_out_w);
    free(u->norm_out_b);
    free(u->conv_out_w);
    free(u->conv_out_b);
    for (i = 0; i < u->n_in; i++)
        free_block(&u->in_b[i]);
    for (i = 0; i < u->n_out; i++)
        free_block(&u->out_b[i]);
    free(u);
}

static int load_res(const h3_st_header *hdr, const char *side, int i, int hid,
                    h3_ups_block *b, char *err, size_t err_size) {
    char n[128];
    size_t conv_n = (size_t)hid * (size_t)hid * 27u;
    size_t emb_n = (size_t)(2 * hid) * (size_t)H3_UPS_EMB;
    memset(b, 0, sizeof(*b));
    b->kind = H3_UPS_RES;
    snprintf(n, sizeof(n), "%s.%d.in_layers.0.weight", side, i);
    if (!(b->in_gn_w = must_load(hdr, n, (size_t)hid, err, err_size)))
        return 0;
    snprintf(n, sizeof(n), "%s.%d.in_layers.0.bias", side, i);
    if (!(b->in_gn_b = must_load(hdr, n, (size_t)hid, err, err_size)))
        return 0;
    snprintf(n, sizeof(n), "%s.%d.in_layers.2.weight", side, i);
    if (!(b->in_conv_w = must_load(hdr, n, conv_n, err, err_size)))
        return 0;
    snprintf(n, sizeof(n), "%s.%d.in_layers.2.bias", side, i);
    if (!(b->in_conv_b = must_load(hdr, n, (size_t)hid, err, err_size)))
        return 0;
    snprintf(n, sizeof(n), "%s.%d.emb_layers.1.weight", side, i);
    if (!(b->emb_w = must_load(hdr, n, emb_n, err, err_size)))
        return 0;
    snprintf(n, sizeof(n), "%s.%d.emb_layers.1.bias", side, i);
    if (!(b->emb_b = must_load(hdr, n, (size_t)(2 * hid), err, err_size)))
        return 0;
    snprintf(n, sizeof(n), "%s.%d.out_norm.weight", side, i);
    if (!(b->out_gn_w = must_load(hdr, n, (size_t)hid, err, err_size)))
        return 0;
    snprintf(n, sizeof(n), "%s.%d.out_norm.bias", side, i);
    if (!(b->out_gn_b = must_load(hdr, n, (size_t)hid, err, err_size)))
        return 0;
    snprintf(n, sizeof(n), "%s.%d.out_layers.2.weight", side, i);
    if (!(b->out_conv_w = must_load(hdr, n, conv_n, err, err_size)))
        return 0;
    snprintf(n, sizeof(n), "%s.%d.out_layers.2.bias", side, i);
    if (!(b->out_conv_b = must_load(hdr, n, (size_t)hid, err, err_size)))
        return 0;
    return 1;
}

static int load_temp(const h3_st_header *hdr, const char *side, int i, int hid,
                     int tk, h3_ups_block *b, char *err, size_t err_size) {
    char n[128];
    memset(b, 0, sizeof(*b));
    b->kind = H3_UPS_TEMP;
    snprintf(n, sizeof(n), "%s.%d.norm.weight", side, i);
    if (!(b->tn_w = must_load(hdr, n, (size_t)hid, err, err_size)))
        return 0;
    snprintf(n, sizeof(n), "%s.%d.norm.bias", side, i);
    if (!(b->tn_b = must_load(hdr, n, (size_t)hid, err, err_size)))
        return 0;
    snprintf(n, sizeof(n), "%s.%d.dwconv.weight", side, i);
    if (!(b->dw_w = must_load(hdr, n, (size_t)hid * (size_t)tk, err, err_size)))
        return 0;
    snprintf(n, sizeof(n), "%s.%d.dwconv.bias", side, i);
    if (!(b->dw_b = must_load(hdr, n, (size_t)hid, err, err_size)))
        return 0;
    snprintf(n, sizeof(n), "%s.%d.pwconv.weight", side, i);
    if (!(b->pw_w = must_load(hdr, n, (size_t)hid * (size_t)hid, err, err_size)))
        return 0;
    snprintf(n, sizeof(n), "%s.%d.pwconv.bias", side, i);
    if (!(b->pw_b = must_load(hdr, n, (size_t)hid, err, err_size)))
        return 0;
    return 1;
}

int h3_latent_upscaler_open(const char *path, h3_latent_upscaler **out,
                            char *err, size_t err_size) {
    h3_st_header hdr;
    h3_latent_upscaler *u = NULL;
    const h3_st_tensor *cin;
    int i, max_in = -1, max_out = -1;
    int kind_in[H3_UPS_MAX_BLOCKS], kind_out[H3_UPS_MAX_BLOCKS];
    memset(&hdr, 0, sizeof(hdr));
    memset(kind_in, 0, sizeof(kind_in));
    memset(kind_out, 0, sizeof(kind_out));
    if (!path || !out)
        return fail(err, err_size, "upscaler open null");
    *out = NULL;
    if (!h3_st_read_header(path, &hdr, err, err_size))
        return 0;
    cin = h3_st_find(&hdr, "conv_in.weight");
    if (!cin)
        cin = h3_st_find(&hdr, "upscaler.conv_in.weight");
    if (!cin || cin->ndim != 5) {
        h3_st_free_header(&hdr);
        return fail(err, err_size, "upscaler missing conv_in.weight [512,24,3,3,3]");
    }
    u = (h3_latent_upscaler *)calloc(1, sizeof(*u));
    if (!u) {
        h3_st_free_header(&hdr);
        return fail(err, err_size, "OOM upscaler");
    }
    u->hid = (int)cin->shape[0];
    u->in_ch = (int)cin->shape[1];
    u->tk = 5;
    if (u->in_ch != 24 || u->hid != 512 || cin->shape[2] != 3 ||
        cin->shape[3] != 3 || cin->shape[4] != 3) {
        snprintf(err, err_size,
                 "upscaler conv_in shape [%llu,%llu,%llu,%llu,%llu] "
                 "want [512,24,3,3,3]",
                 (unsigned long long)cin->shape[0],
                 (unsigned long long)cin->shape[1],
                 (unsigned long long)cin->shape[2],
                 (unsigned long long)cin->shape[3],
                 (unsigned long long)cin->shape[4]);
        h3_latent_upscaler_close(u);
        h3_st_free_header(&hdr);
        return 0;
    }
    for (i = 0; i < (int)hdr.tensor_count; i++) {
        const char *nm = hdr.tensors[i].name;
        int idx;
        if (strstr(nm, ".q.weight") || strstr(nm, ".attn.")) {
            h3_latent_upscaler_close(u);
            h3_st_free_header(&hdr);
            return fail(err, err_size,
                        "upscaler has attention keys; LBH H3 ckpt is attn=0");
        }
        if (parse_idx(nm, "in_blocks.", &idx)) {
            if (idx > max_in)
                max_in = idx;
            if (strstr(nm, ".in_layers."))
                kind_in[idx] = H3_UPS_RES;
            else if (strstr(nm, ".dwconv.weight")) {
                kind_in[idx] = H3_UPS_TEMP;
                if (hdr.tensors[i].ndim >= 3)
                    u->tk = (int)hdr.tensors[i].shape[2];
            }
        }
        if (parse_idx(nm, "out_blocks.", &idx)) {
            if (idx > max_out)
                max_out = idx;
            if (strstr(nm, ".in_layers."))
                kind_out[idx] = H3_UPS_RES;
            else if (strstr(nm, ".dwconv.weight")) {
                kind_out[idx] = H3_UPS_TEMP;
                if (hdr.tensors[i].ndim >= 3)
                    u->tk = (int)hdr.tensors[i].shape[2];
            }
        }
    }
    if (max_in < 0 || max_out < 0) {
        h3_latent_upscaler_close(u);
        h3_st_free_header(&hdr);
        return fail(err, err_size, "upscaler missing in_blocks/out_blocks");
    }
    u->n_in = max_in + 1;
    u->n_out = max_out + 1;
    if (!(u->conv_in_w =
              must_load(&hdr, "conv_in.weight",
                        (size_t)u->hid * (size_t)u->in_ch * 27u, err,
                        err_size)) ||
        !(u->conv_in_b =
              must_load(&hdr, "conv_in.bias", (size_t)u->hid, err, err_size)) ||
        !(u->emb0_w = must_load(&hdr, "embed.0.weight", (size_t)H3_UPS_EMB, err,
                                err_size)) ||
        !(u->emb0_b =
              must_load(&hdr, "embed.0.bias", (size_t)H3_UPS_EMB, err,
                        err_size)) ||
        !(u->emb2_w = must_load(&hdr, "embed.2.weight",
                                (size_t)H3_UPS_EMB * (size_t)H3_UPS_EMB, err,
                                err_size)) ||
        !(u->emb2_b =
              must_load(&hdr, "embed.2.bias", (size_t)H3_UPS_EMB, err,
                        err_size)) ||
        !(u->norm_out_w =
              must_load(&hdr, "norm_out.weight", (size_t)u->hid, err,
                        err_size)) ||
        !(u->norm_out_b =
              must_load(&hdr, "norm_out.bias", (size_t)u->hid, err, err_size)) ||
        !(u->conv_out_w =
              must_load(&hdr, "conv_out.weight",
                        (size_t)u->in_ch * (size_t)u->hid * 27u, err,
                        err_size)) ||
        !(u->conv_out_b = must_load(&hdr, "conv_out.bias", (size_t)u->in_ch, err,
                                    err_size))) {
        h3_latent_upscaler_close(u);
        h3_st_free_header(&hdr);
        return 0;
    }
    for (i = 0; i < u->n_in; i++) {
        if (kind_in[i] == H3_UPS_RES) {
            if (!load_res(&hdr, "in_blocks", i, u->hid, &u->in_b[i], err,
                          err_size)) {
                h3_latent_upscaler_close(u);
                h3_st_free_header(&hdr);
                return 0;
            }
        } else if (kind_in[i] == H3_UPS_TEMP) {
            if (!load_temp(&hdr, "in_blocks", i, u->hid, u->tk, &u->in_b[i],
                           err, err_size)) {
                h3_latent_upscaler_close(u);
                h3_st_free_header(&hdr);
                return 0;
            }
        } else {
            snprintf(err, err_size, "upscaler in_blocks.%d unknown kind", i);
            h3_latent_upscaler_close(u);
            h3_st_free_header(&hdr);
            return 0;
        }
    }
    for (i = 0; i < u->n_out; i++) {
        if (kind_out[i] == H3_UPS_RES) {
            if (!load_res(&hdr, "out_blocks", i, u->hid, &u->out_b[i], err,
                          err_size)) {
                h3_latent_upscaler_close(u);
                h3_st_free_header(&hdr);
                return 0;
            }
        } else if (kind_out[i] == H3_UPS_TEMP) {
            if (!load_temp(&hdr, "out_blocks", i, u->hid, u->tk, &u->out_b[i],
                           err, err_size)) {
                h3_latent_upscaler_close(u);
                h3_st_free_header(&hdr);
                return 0;
            }
        } else {
            snprintf(err, err_size, "upscaler out_blocks.%d unknown kind", i);
            h3_latent_upscaler_close(u);
            h3_st_free_header(&hdr);
            return 0;
        }
    }
    h3_st_free_header(&hdr);
    *out = u;
    return 1;
}

int h3_latent_upscaler_probe(const h3_latent_upscaler *u, int *in_ch, int *hid,
                             char *err, size_t err_size) {
    if (!u)
        return fail(err, err_size, "upscaler probe null");
    if (in_ch)
        *in_ch = u->in_ch;
    if (hid)
        *hid = u->hid;
    if (u->in_ch != 24 || u->hid != 512)
        return fail(err, err_size, "upscaler probe: not 24→512");
    return 1;
}

static int upload(float **d, const float *h, size_t n, char *err,
                  size_t err_size) {
    if (cudaMalloc((void **)d, n * sizeof(float)) != cudaSuccess)
        return fail(err, err_size, "OOM upscaler weight upload");
    if (cudaMemcpy(*d, h, n * sizeof(float), cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        cudaFree(*d);
        *d = NULL;
        return fail(err, err_size, "upscaler H2D");
    }
    return 1;
}

/* Persistent small H2D slots: no cudaMalloc/cudaFree per layer (those sync). */
static int upload_slot(h3_latent_upscaler *u, float **d, const float *h, size_t n,
                       char *err, size_t err_size) {
    int s;
    if (!u || u->slot_i >= H3_UPS_SLOTS)
        return fail(err, err_size, "upscaler slot exhausted");
    s = u->slot_i++;
    if (u->slot_cap[s] < n || !u->d_slot[s]) {
        cudaFree(u->d_slot[s]);
        u->d_slot[s] = NULL;
        u->slot_cap[s] = 0;
        if (cudaMalloc((void **)&u->d_slot[s], n * sizeof(float)) !=
            cudaSuccess) {
            (void)cudaGetLastError();
            u->d_slot[s] = NULL;
            return fail(err, err_size, "OOM upscaler slot");
        }
        u->slot_cap[s] = n;
    }
    *d = u->d_slot[s];
    if (cudaMemcpy(*d, h, n * sizeof(float), cudaMemcpyHostToDevice) !=
        cudaSuccess)
        return fail(err, err_size, "upscaler slot H2D");
    return 1;
}

static void gemm_vec(const float *w, const float *b, const float *x, float *y,
                     int out, int in) {
    int i, k;
    for (i = 0; i < out; i++) {
        float s = b ? b[i] : 0.0f;
        const float *wr = w + (size_t)i * (size_t)in;
        for (k = 0; k < in; k++)
            s += wr[k] * x[k];
        y[i] = s;
    }
}

static float silu_h(float v) { return v / (1.0f + expf(-v)); }

static int run_res3(h3_latent_upscaler *u, const h3_ups_block *b, float *d_x,
                    float *d_h, float *d_o, float *d_w, int hid, int t, int hh,
                    int ww, const float *emb, char *err, size_t err_size) {
    float e1[1024], scale[512], shift[512];
    float *d_gnw = NULL, *d_gnb = NULL, *d_sc = NULL, *d_sh = NULL, *d_bias = NULL;
    size_t vol = (size_t)hid * (size_t)t * (size_t)hh * (size_t)ww;
    size_t conv_n = (size_t)hid * (size_t)hid * 27u;
    int i;
    {
        float emb_s[H3_UPS_EMB];
        int e;
        for (e = 0; e < H3_UPS_EMB; e++)
            emb_s[e] = silu_h(emb[e]);
        gemm_vec(b->emb_w, b->emb_b, emb_s, e1, 2 * hid, H3_UPS_EMB);
    }
    for (i = 0; i < hid; i++) {
        scale[i] = e1[i];
        shift[i] = e1[hid + i];
    }
    if (cudaMemcpy(d_h, d_x, vol * sizeof(float), cudaMemcpyDeviceToDevice) !=
        cudaSuccess)
        return fail(err, err_size, "res copy");
    u->slot_i = 0;
    if (!upload_slot(u, &d_gnw, b->in_gn_w, (size_t)hid, err, err_size) ||
        !upload_slot(u, &d_gnb, b->in_gn_b, (size_t)hid, err, err_size))
        return 0;
    if (!h3_ups_groupnorm(d_h, d_gnw, d_gnb, 1, hid, t, hh, ww, H3_UPS_GN,
                          H3_UPS_EPS, err, err_size) ||
        !h3_ups_silu(d_h, vol, err, err_size))
        return 0;
    if (cudaMemcpy(d_w, b->in_conv_w, conv_n * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess)
        return fail(err, err_size, "res in_conv H2D");
    if (!upload_slot(u, &d_bias, b->in_conv_b, (size_t)hid, err, err_size))
        return 0;
    if (!h3_ups_conv3d_nchw(d_h, d_o, d_w, d_bias, 1, hid, hid, t, hh, ww, 3, 3,
                            3, 1, 1, 1, 1, err, err_size))
        return 0;
    if (!upload_slot(u, &d_gnw, b->out_gn_w, (size_t)hid, err, err_size) ||
        !upload_slot(u, &d_gnb, b->out_gn_b, (size_t)hid, err, err_size) ||
        !upload_slot(u, &d_sc, scale, (size_t)hid, err, err_size) ||
        !upload_slot(u, &d_sh, shift, (size_t)hid, err, err_size))
        return 0;
    if (!h3_ups_groupnorm(d_o, d_gnw, d_gnb, 1, hid, t, hh, ww, H3_UPS_GN,
                          H3_UPS_EPS, err, err_size) ||
        !h3_ups_adaln(d_o, d_sc, d_sh, 1, hid, t, hh, ww, err, err_size) ||
        !h3_ups_silu(d_o, vol, err, err_size))
        return 0;
    if (cudaMemcpy(d_w, b->out_conv_w, conv_n * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess)
        return fail(err, err_size, "res out_conv H2D");
    if (!upload_slot(u, &d_bias, b->out_conv_b, (size_t)hid, err, err_size))
        return 0;
    if (!h3_ups_conv3d_nchw(d_o, d_h, d_w, d_bias, 1, hid, hid, t, hh, ww, 3, 3,
                            3, 1, 1, 1, 1, err, err_size))
        return 0;
    if (!h3_ups_add(d_x, d_x, d_h, vol, err, err_size))
        return 0;
    return 1;
}

static int run_temp(h3_latent_upscaler *u, const h3_ups_block *b, float *d_x,
                    float *d_h, float *d_o, float *d_w, int hid, int tk, int t,
                    int hh, int ww, char *err, size_t err_size) {
    float *d_gnw = NULL, *d_gnb = NULL, *d_bias = NULL;
    size_t vol = (size_t)hid * (size_t)t * (size_t)hh * (size_t)ww;
    int pad = tk / 2;
    if (cudaMemcpy(d_h, d_x, vol * sizeof(float), cudaMemcpyDeviceToDevice) !=
        cudaSuccess)
        return fail(err, err_size, "temp copy");
    u->slot_i = 0;
    if (!upload_slot(u, &d_gnw, b->tn_w, (size_t)hid, err, err_size) ||
        !upload_slot(u, &d_gnb, b->tn_b, (size_t)hid, err, err_size))
        return 0;
    if (!h3_ups_groupnorm(d_h, d_gnw, d_gnb, 1, hid, t, hh, ww, H3_UPS_GN,
                          H3_UPS_EPS, err, err_size) ||
        !h3_ups_silu(d_h, vol, err, err_size))
        return 0;
    if (cudaMemcpy(d_w, b->dw_w, (size_t)hid * (size_t)tk * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess)
        return fail(err, err_size, "dw H2D");
    if (!upload_slot(u, &d_bias, b->dw_b, (size_t)hid, err, err_size))
        return 0;
    if (!h3_ups_conv3d_nchw(d_h, d_o, d_w, d_bias, 1, hid, hid, t, hh, ww, tk, 1,
                            1, pad, 0, 0, hid, err, err_size))
        return 0;
    if (cudaMemcpy(d_w, b->pw_w, (size_t)hid * (size_t)hid * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess)
        return fail(err, err_size, "pw H2D");
    if (!upload_slot(u, &d_bias, b->pw_b, (size_t)hid, err, err_size))
        return 0;
    if (!h3_ups_conv3d_nchw(d_o, d_h, d_w, d_bias, 1, hid, hid, t, hh, ww, 1, 1,
                            1, 0, 0, 0, 1, err, err_size))
        return 0;
    if (!h3_ups_add(d_x, d_x, d_h, vol, err, err_size))
        return 0;
    return 1;
}

static int run_blocks(h3_latent_upscaler *u, h3_ups_block *bs, int n, float *d_x,
                      float *d_h, float *d_o, float *d_w, int hid, int tk, int t,
                      int hh, int ww, const float *emb, char *err,
                      size_t err_size) {
    int i;
    for (i = 0; i < n; i++) {
        if (bs[i].kind == H3_UPS_RES) {
            if (!run_res3(u, &bs[i], d_x, d_h, d_o, d_w, hid, t, hh, ww, emb,
                          err, err_size))
                return 0;
        } else if (bs[i].kind == H3_UPS_TEMP) {
            if (!run_temp(u, &bs[i], d_x, d_h, d_o, d_w, hid, tk, t, hh, ww,
                          err, err_size))
                return 0;
        } else
            return fail(err, err_size, "unknown block kind");
    }
    return 1;
}

/* stop: 1=conv_in 2=first in_block 3=all in_blocks 4=trilinear 5=full net */
static int forward_seg(h3_latent_upscaler *u, const float *emb, float *d_in,
                       float *d_x, float *d_h, float *d_o, float *d_w, int t,
                       int h, int w, int out_h, int out_w, int stop, char *err,
                       size_t err_size) {
    float *d_cb = NULL, *d_nw = NULL, *d_nb = NULL, *d_ob = NULL;
    size_t vol_in = (size_t)u->hid * (size_t)t * (size_t)h * (size_t)w;
    size_t vol_out = (size_t)u->hid * (size_t)t * (size_t)out_h * (size_t)out_w;
    size_t cin_n = (size_t)u->hid * (size_t)u->in_ch * 27u;
    size_t cout_n = (size_t)u->in_ch * (size_t)u->hid * 27u;
    u->slot_i = 0;
    if (cudaMemcpy(d_w, u->conv_in_w, cin_n * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess)
        return fail(err, err_size, "conv_in H2D");
    if (!upload_slot(u, &d_cb, u->conv_in_b, (size_t)u->hid, err, err_size))
        return 0;
    if (!h3_ups_conv3d_nchw(d_in, d_x, d_w, d_cb, 1, u->in_ch, u->hid, t, h, w,
                            3, 3, 3, 1, 1, 1, 1, err, err_size))
        return 0;
    if (stop <= 1)
        return 1;
    if (stop == 2) {
        if (u->n_in < 1)
            return fail(err, err_size, "oracle no in_blocks");
        if (u->in_b[0].kind == H3_UPS_RES) {
            if (!run_res3(u, &u->in_b[0], d_x, d_h, d_o, d_w, u->hid, t, h, w,
                          emb, err, err_size))
                return 0;
        } else if (!run_temp(u, &u->in_b[0], d_x, d_h, d_o, d_w, u->hid, u->tk,
                             t, h, w, err, err_size))
            return 0;
        return 1;
    }
    if (!run_blocks(u, u->in_b, u->n_in, d_x, d_h, d_o, d_w, u->hid, u->tk, t, h,
                    w, emb, err, err_size))
        return 0;
    if (stop <= 3)
        return 1;
    if (!h3_ups_trilinear(d_x, d_h, 1, u->hid, t, h, w, out_h, out_w, err,
                          err_size))
        return 0;
    if (cudaMemcpy(d_x, d_h, vol_out * sizeof(float),
                   cudaMemcpyDeviceToDevice) != cudaSuccess)
        return fail(err, err_size, "interp copy");
    (void)vol_in;
    if (stop <= 4)
        return 1;
    if (!run_blocks(u, u->out_b, u->n_out, d_x, d_h, d_o, d_w, u->hid, u->tk, t,
                    out_h, out_w, emb, err, err_size))
        return 0;
    u->slot_i = 0;
    if (!upload_slot(u, &d_nw, u->norm_out_w, (size_t)u->hid, err, err_size) ||
        !upload_slot(u, &d_nb, u->norm_out_b, (size_t)u->hid, err, err_size))
        return 0;
    if (!h3_ups_groupnorm(d_x, d_nw, d_nb, 1, u->hid, t, out_h, out_w, H3_UPS_GN,
                          H3_UPS_EPS, err, err_size) ||
        !h3_ups_silu(d_x, vol_out, err, err_size))
        return 0;
    if (cudaMemcpy(d_w, u->conv_out_w, cout_n * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess)
        return fail(err, err_size, "conv_out H2D");
    if (!upload_slot(u, &d_ob, u->conv_out_b, (size_t)u->in_ch, err, err_size))
        return 0;
    if (!h3_ups_conv3d_nchw(d_x, d_h, d_w, d_ob, 1, u->hid, u->in_ch, t, out_h,
                            out_w, 3, 3, 3, 1, 1, 1, 1, err, err_size))
        return 0;
    return 1;
}

static void make_emb(const h3_latent_upscaler *u, float scale, float emb[64]) {
    float x0[64], h[64];
    int i;
    float s = scale - 1.0f;
    gemm_vec(u->emb0_w, u->emb0_b, &s, x0, H3_UPS_EMB, 1);
    for (i = 0; i < H3_UPS_EMB; i++)
        x0[i] = silu_h(x0[i]);
    gemm_vec(u->emb2_w, u->emb2_b, x0, h, H3_UPS_EMB, H3_UPS_EMB);
    memcpy(emb, h, sizeof(float) * H3_UPS_EMB);
}

static int ups_ws_grow_one(float **p, size_t *cap, size_t need, char *err,
                           size_t err_size) {
    if (*p && *cap >= need)
        return 1;
    cudaFree(*p);
    *p = NULL;
    *cap = 0;
    if (cudaMalloc((void **)p, need * sizeof(float)) != cudaSuccess) {
        (void)cudaGetLastError();
        *p = NULL;
        return fail(err, err_size, "OOM upscaler workspace");
    }
    *cap = need;
    return 1;
}

static int ups_ws_ensure(h3_latent_upscaler *u, size_t in_n, size_t hid_n,
                         size_t w_n, char *err, size_t err_size) {
    return ups_ws_grow_one(&u->d_in, &u->cap_in, in_n, err, err_size) &&
           ups_ws_grow_one(&u->d_x, &u->cap_x, hid_n, err, err_size) &&
           ups_ws_grow_one(&u->d_h, &u->cap_h, hid_n, err, err_size) &&
           ups_ws_grow_one(&u->d_o, &u->cap_o, hid_n, err, err_size) &&
           ups_ws_grow_one(&u->d_w, &u->cap_w, w_n, err, err_size) &&
           ups_ws_grow_one(&u->d_mean, &u->cap_mean, 24, err, err_size) &&
           ups_ws_grow_one(&u->d_std, &u->cap_std, 24, err, err_size);
}

int h3_latent_upscaler_forward(h3_latent_upscaler *u, h3_cuda *cuda,
                               const float *z_low, int t, int h, int w,
                               int out_h, int out_w, const float *mean24,
                               const float *std24, float *z_high, char *err,
                               size_t err_size) {
    float emb[64];
    float *d_in, *d_x, *d_h, *d_o, *d_w, *d_mean, *d_std;
    float scale;
    int chunk, overlap, ok = 0;
    size_t w_n, hid_n, in_n;
    const char *ce;
    if (!u || !cuda || !z_low || !z_high || !mean24 || !std24)
        return fail(err, err_size, "upscaler forward null");
    if (t < 1 || h < 1 || w < 1 || out_h < 1 || out_w < 1)
        return fail(err, err_size, "upscaler forward dims");
    if (!h3_cuda_bind_device(cuda, err, err_size))
        return 0;
    h3_ups_bind_cuda(cuda);
    scale = ((float)out_h / (float)h + (float)out_w / (float)w) * 0.5f;
    make_emb(u, scale, emb);
    overlap = u->tk;
    w_n = (size_t)u->hid * (size_t)u->hid * 27u;
    /* Default one-shot for T<=128 (10s T=72). H3_LBH_TIME_CHUNK=32 restores
     * overlap windows. */
    chunk = (t <= 128) ? t : 32;
    ce = getenv("H3_LBH_TIME_CHUNK");
    if (ce && ce[0]) {
        long v = strtol(ce, NULL, 10);
        if (v >= 8 && v <= 256)
            chunk = (int)v;
    }
retry_chunk:
    {
        int tmax = t <= chunk ? t : (t + 2 * overlap);
        if (tmax < t)
            tmax = t;
        hid_n = (size_t)u->hid * (size_t)tmax * (size_t)(out_h > h ? out_h : h) *
                (size_t)(out_w > w ? out_w : w);
        in_n = (size_t)u->in_ch * (size_t)tmax * (size_t)h * (size_t)w;
        if (in_n < (size_t)u->in_ch * (size_t)tmax * (size_t)out_h *
                       (size_t)out_w)
            in_n = (size_t)u->in_ch * (size_t)tmax * (size_t)out_h *
                   (size_t)out_w;
    }
    if (!ups_ws_ensure(u, in_n, hid_n, w_n, err, err_size)) {
        if (chunk > 32 && t > 32) {
            fprintf(stderr,
                    "selflift: oneshot OOM (t=%d chunk=%d); fallback "
                    "time_chunk=32\n",
                    t, chunk);
            fflush(stderr);
            ups_ws_release(u);
            chunk = 32;
            goto retry_chunk;
        }
        return 0;
    }
    d_in = u->d_in;
    d_x = u->d_x;
    d_h = u->d_h;
    d_o = u->d_o;
    d_w = u->d_w;
    d_mean = u->d_mean;
    d_std = u->d_std;
    fprintf(stderr, "selflift: time_chunk=%d t=%d oneshot=%d hid_ws_mb=%.1f\n",
            chunk, t, t <= chunk ? 1 : 0,
            (double)hid_n * 4.0 * 3.0 / (1024.0 * 1024.0));
    fflush(stderr);
    if (cudaMemcpy(d_mean, mean24, 24 * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_std, std24, 24 * sizeof(float), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
        fail(err, err_size, "upscaler stats H2D");
        goto done;
    }

    if (t <= chunk) {
        size_t n24 = (size_t)u->in_ch * (size_t)t * (size_t)h * (size_t)w;
        size_t n24o = (size_t)u->in_ch * (size_t)t * (size_t)out_h * (size_t)out_w;
        if (cudaMemcpy(d_in, z_low, n24 * sizeof(float),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            fail(err, err_size, "upscaler input H2D");
            goto done;
        }
        if (!h3_ups_scale_bias(d_in, d_mean, d_std, 1, 1, u->in_ch, t, h, w, err,
                               err_size))
            goto done;
        if (!forward_seg(u, emb, d_in, d_x, d_h, d_o, d_w, t, h, w, out_h, out_w,
                         5, err, err_size))
            goto done;
        if (!h3_ups_scale_bias(d_h, d_mean, d_std, 0, 1, u->in_ch, t, out_h,
                               out_w, err, err_size))
            goto done;
        if (cudaMemcpy(z_high, d_h, n24o * sizeof(float),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            fail(err, err_size, "upscaler D2H");
            goto done;
        }
        ok = 1;
        goto done;
    }

    /* T>32: replicate-pad + overlap windows (plugin _temporal_windows). */
    {
        int pad_t = t + 2 * overlap;
        size_t plane = (size_t)h * (size_t)w;
        float *padded = (float *)malloc((size_t)u->in_ch * (size_t)pad_t * plane *
                                        sizeof(float));
        float *acc = (float *)calloc((size_t)u->in_ch * (size_t)t *
                                         (size_t)out_h * (size_t)out_w,
                                     sizeof(float));
        float *wgt = (float *)calloc((size_t)t, sizeof(float));
        int start, c, tt, hh, ww;
        if (!padded || !acc || !wgt) {
            free(padded);
            free(acc);
            free(wgt);
            fail(err, err_size, "OOM upscaler temporal pad");
            goto done;
        }
        for (c = 0; c < u->in_ch; c++) {
            for (tt = 0; tt < pad_t; tt++) {
                int src = tt - overlap;
                if (src < 0)
                    src = 0;
                if (src >= t)
                    src = t - 1;
                memcpy(padded + ((size_t)c * (size_t)pad_t + (size_t)tt) * plane,
                       z_low + ((size_t)c * (size_t)t + (size_t)src) * plane,
                       plane * sizeof(float));
            }
        }
        for (start = 0; start < t; start += chunk) {
            int end = start + chunk;
            int out_start, out_end, lo, hi, seg_t, s0, n_valid, j;
            float *seg_out = NULL;
            if (end > t)
                end = t;
            out_start = start - overlap;
            if (out_start < 0)
                out_start = 0;
            out_end = end + overlap;
            if (out_end > t)
                out_end = t;
            lo = out_start;
            hi = out_end + 2 * overlap;
            seg_t = hi - lo;
            s0 = (out_start + overlap) - lo;
            n_valid = out_end - out_start;
            {
                size_t seg_in = (size_t)u->in_ch * (size_t)seg_t * plane;
                float *seg = (float *)malloc(seg_in * sizeof(float));
                size_t n24o;
                if (!seg) {
                    free(padded);
                    free(acc);
                    free(wgt);
                    fail(err, err_size, "OOM upscaler seg");
                    goto done;
                }
                for (c = 0; c < u->in_ch; c++)
                    memcpy(seg + (size_t)c * (size_t)seg_t * plane,
                           padded + ((size_t)c * (size_t)pad_t + (size_t)lo) *
                                        plane,
                           (size_t)seg_t * plane * sizeof(float));
                if (cudaMemcpy(d_in, seg, seg_in * sizeof(float),
                               cudaMemcpyHostToDevice) != cudaSuccess) {
                    free(seg);
                    free(padded);
                    free(acc);
                    free(wgt);
                    fail(err, err_size, "upscaler seg H2D");
                    goto done;
                }
                free(seg);
                if (!h3_ups_scale_bias(d_in, d_mean, d_std, 1, 1, u->in_ch,
                                       seg_t, h, w, err, err_size) ||
                    !forward_seg(u, emb, d_in, d_x, d_h, d_o, d_w, seg_t, h, w,
                                 out_h, out_w, 5, err, err_size) ||
                    !h3_ups_scale_bias(d_h, d_mean, d_std, 0, 1, u->in_ch,
                                       seg_t, out_h, out_w, err, err_size)) {
                    free(padded);
                    free(acc);
                    free(wgt);
                    goto done;
                }
                n24o = (size_t)u->in_ch * (size_t)seg_t * (size_t)out_h *
                       (size_t)out_w;
                seg_out = (float *)malloc(n24o * sizeof(float));
                if (!seg_out ||
                    cudaMemcpy(seg_out, d_h, n24o * sizeof(float),
                               cudaMemcpyDeviceToHost) != cudaSuccess) {
                    free(seg_out);
                    free(padded);
                    free(acc);
                    free(wgt);
                    fail(err, err_size, "OOM/D2H upscaler seg");
                    goto done;
                }
            }
            for (j = 0; j < n_valid; j++) {
                float wt = 1.0f;
                int dst_t = out_start + j;
                if (start > out_start && j < (start - out_start))
                    wt = (float)(j + 1) / (float)((start - out_start) + 1);
                if (out_end > end && j >= n_valid - (out_end - end)) {
                    int blend = out_end - end;
                    int back = n_valid - j;
                    wt = (float)back / (float)(blend + 1);
                }
                wgt[dst_t] += wt;
                for (c = 0; c < u->in_ch; c++)
                    for (hh = 0; hh < out_h; hh++)
                        for (ww = 0; ww < out_w; ww++) {
                            size_t so =
                                ((((size_t)c * (size_t)seg_t +
                                   (size_t)(s0 + j)) *
                                      (size_t)out_h +
                                  (size_t)hh) *
                                     (size_t)out_w +
                                 (size_t)ww);
                            size_t ao =
                                ((((size_t)c * (size_t)t + (size_t)dst_t) *
                                      (size_t)out_h +
                                  (size_t)hh) *
                                     (size_t)out_w +
                                 (size_t)ww);
                            acc[ao] += seg_out[so] * wt;
                        }
            }
            free(seg_out);
        }
        for (c = 0; c < u->in_ch; c++)
            for (tt = 0; tt < t; tt++) {
                float den = wgt[tt] > 1e-8f ? wgt[tt] : 1e-8f;
                for (hh = 0; hh < out_h; hh++)
                    for (ww = 0; ww < out_w; ww++) {
                        size_t ao =
                            ((((size_t)c * (size_t)t + (size_t)tt) *
                                  (size_t)out_h +
                              (size_t)hh) *
                                 (size_t)out_w +
                             (size_t)ww);
                        z_high[ao] = acc[ao] / den;
                    }
            }
        free(padded);
        free(acc);
        free(wgt);
        ok = 1;
    }

done:
    /* Device WS stays on `u` for the next clip / hot request. */
    return ok;
}

static int alloc_ws(float **d_in, float **d_x, float **d_h, float **d_o,
                    float **d_w, float **d_mean, float **d_std, size_t in_n,
                    size_t hid_n, size_t w_n, char *err, size_t err_size) {
    if (cudaMalloc((void **)d_in, in_n * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)d_x, hid_n * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)d_h, hid_n * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)d_o, hid_n * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)d_w, w_n * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)d_mean, 24 * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)d_std, 24 * sizeof(float)) != cudaSuccess)
        return fail(err, err_size, "OOM upscaler workspace");
    return 1;
}

int h3_latent_upscaler_conv_in_head(const h3_latent_upscaler *u, float *out8) {
    int i;
    if (!u || !u->conv_in_w || !out8)
        return 0;
    for (i = 0; i < 8; i++)
        out8[i] = u->conv_in_w[i];
    return 1;
}

int h3_latent_upscaler_oracle(h3_latent_upscaler *u, h3_cuda *cuda, int stage,
                              const float *z_low, int t, int h, int w,
                              int dst_h, int dst_w, const float *mean24,
                              const float *std24, float *out, int *got_c,
                              int *got_t, int *got_h, int *got_w, char *err,
                              size_t err_size) {
    float emb[64];
    float *d_in = NULL, *d_x = NULL, *d_h = NULL, *d_o = NULL, *d_w = NULL;
    float *d_mean = NULL, *d_std = NULL;
    float scale;
    int ok = 0, c_out, hh, ww;
    size_t w_n, hid_n, in_n, ncpy;
    const float *src;
    if (!u || !cuda || !z_low || !out || !mean24 || !std24)
        return fail(err, err_size, "upscaler oracle null");
    if (stage < 0 || stage > 5)
        return fail(err, err_size, "upscaler oracle stage");
    if (t < 1 || h < 1 || w < 1 || dst_h < 1 || dst_w < 1)
        return fail(err, err_size, "upscaler oracle dims");
    if (!h3_cuda_bind_device(cuda, err, err_size))
        return 0;
    h3_ups_bind_cuda(cuda);
    scale = ((float)dst_h / (float)h + (float)dst_w / (float)w) * 0.5f;
    make_emb(u, scale, emb);
    w_n = (size_t)u->hid * (size_t)u->hid * 27u;
    hid_n = (size_t)u->hid * (size_t)t * (size_t)(dst_h > h ? dst_h : h) *
            (size_t)(dst_w > w ? dst_w : w);
    in_n = (size_t)u->in_ch * (size_t)t * (size_t)h * (size_t)w;
    if (in_n < (size_t)u->in_ch * (size_t)t * (size_t)dst_h * (size_t)dst_w)
        in_n = (size_t)u->in_ch * (size_t)t * (size_t)dst_h * (size_t)dst_w;
    if (!alloc_ws(&d_in, &d_x, &d_h, &d_o, &d_w, &d_mean, &d_std, in_n, hid_n,
                  w_n, err, err_size))
        goto done;
    if (cudaMemcpy(d_mean, mean24, 24 * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_std, std24, 24 * sizeof(float), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
        fail(err, err_size, "oracle stats H2D");
        goto done;
    }
    {
        size_t n24 = (size_t)u->in_ch * (size_t)t * (size_t)h * (size_t)w;
        if (cudaMemcpy(d_in, z_low, n24 * sizeof(float),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            fail(err, err_size, "oracle input H2D");
            goto done;
        }
    }
    if (!h3_ups_scale_bias(d_in, d_mean, d_std, 1, 1, u->in_ch, t, h, w, err,
                           err_size))
        goto done;
    if (stage == 0) {
        c_out = u->in_ch;
        hh = h;
        ww = w;
        src = d_in;
        goto copy;
    }
    if (!forward_seg(u, emb, d_in, d_x, d_h, d_o, d_w, t, h, w, dst_h, dst_w,
                     stage <= 4 ? stage : 5, err, err_size))
        goto done;
    if (stage == 5) {
        if (!h3_ups_scale_bias(d_h, d_mean, d_std, 0, 1, u->in_ch, t, dst_h,
                               dst_w, err, err_size))
            goto done;
        c_out = u->in_ch;
        hh = dst_h;
        ww = dst_w;
        src = d_h;
        goto copy;
    }
    c_out = u->hid;
    hh = (stage >= 4) ? dst_h : h;
    ww = (stage >= 4) ? dst_w : w;
    src = d_x;
copy:
    ncpy = (size_t)c_out * (size_t)t * (size_t)hh * (size_t)ww;
    if (cudaMemcpy(out, src, ncpy * sizeof(float), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        fail(err, err_size, "oracle D2H");
        goto done;
    }
    if (got_c)
        *got_c = c_out;
    if (got_t)
        *got_t = t;
    if (got_h)
        *got_h = hh;
    if (got_w)
        *got_w = ww;
    ok = 1;
done:
    cudaFree(d_in);
    cudaFree(d_x);
    cudaFree(d_h);
    cudaFree(d_o);
    cudaFree(d_w);
    cudaFree(d_mean);
    cudaFree(d_std);
    return ok;
}
