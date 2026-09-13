/* 2-GPU Ulysses A2A (S_pad=64, H=56, D=8). Integer fill; max-abs==0 vs host ref. */
#include "h3_ulysses_comm.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
static void expect(int c, const char *m) {
    if (!c) {
        fprintf(stderr, "FAIL %s\n", m);
        fails++;
    }
}

static float qkv_in_at(int rank, int s, int qkv, int h, int d, int seq_local,
                       int heads, int head_dim) {
    /* Unique integer-valued fill. */
    int i = ((s * 3 + qkv) * heads + h) * head_dim + d;
    return (float)(rank * 100000 + i + 1);
}

int main(void) {
    h3_ulysses_cfg cfg;
    int devs[2] = {0, 1};
    int n_dev = 0;
    char err[256];
    const int seq_local = 32;
    const int heads = 56;
    const int head_dim = 8;
    const int u = 2;
    const int seq_pad = seq_local * u;
    const int h_local = heads / u;
    const size_t qkv_in_n = (size_t)seq_local * 3u * (size_t)heads * (size_t)head_dim;
    const size_t qkv_out_n = (size_t)seq_pad * 3u * (size_t)h_local * (size_t)head_dim;
    const size_t attn_in_n = (size_t)seq_pad * (size_t)h_local * (size_t)head_dim;
    const size_t attn_out_n = (size_t)seq_local * (size_t)heads * (size_t)head_dim;
    float *h_in[2] = {0};
    float *h_qkv[2] = {0};
    float *h_attn_in[2] = {0};
    float *h_attn_out[2] = {0};
    float *d_in[2] = {0};
    float *d_qkv[2] = {0};
    float *d_attn_in[2] = {0};
    float *d_attn_out[2] = {0};
    int r, s, qkv, h, d, hl, s_pad, src;
    float maxabs = 0.0f;

    cudaGetDeviceCount(&n_dev);
    if (n_dev < 2) {
        printf("test_h3_ulysses_a2a: SKIP (need 2 GPUs, have %d)\n", n_dev);
        return 0;
    }
    memset(&cfg, 0, sizeof(cfg));
    cfg.tp_size = 1;
    cfg.ulysses = 2;
    cfg.ring = 1;
    if (!h3_ulysses_comm_init(&cfg, devs, 2, err, sizeof(err))) {
        fprintf(stderr, "init: %s\n", err);
        return 1;
    }

    for (r = 0; r < 2; r++) {
        size_t i;
        cudaSetDevice(r);
        h_in[r] = (float *)malloc(qkv_in_n * sizeof(float));
        h_qkv[r] = (float *)malloc(qkv_out_n * sizeof(float));
        h_attn_in[r] = (float *)malloc(attn_in_n * sizeof(float));
        h_attn_out[r] = (float *)malloc(attn_out_n * sizeof(float));
        cudaMalloc(&d_in[r], qkv_in_n * sizeof(float));
        cudaMalloc(&d_qkv[r], qkv_out_n * sizeof(float));
        cudaMalloc(&d_attn_in[r], attn_in_n * sizeof(float));
        cudaMalloc(&d_attn_out[r], attn_out_n * sizeof(float));
        for (s = 0; s < seq_local; s++)
            for (qkv = 0; qkv < 3; qkv++)
                for (h = 0; h < heads; h++)
                    for (d = 0; d < head_dim; d++) {
                        size_t idx = ((((size_t)s * 3u + (size_t)qkv) *
                                       (size_t)heads) +
                                      (size_t)h) *
                                         (size_t)head_dim +
                                     (size_t)d;
                        h_in[r][idx] =
                            qkv_in_at(r, s, qkv, h, d, seq_local, heads, head_dim);
                    }
        cudaMemcpy(d_in[r], h_in[r], qkv_in_n * sizeof(float),
                   cudaMemcpyHostToDevice);
        expect(h3_ulysses_comm_bind(&cfg, r, d_in[r], d_qkv[r], err,
                                    sizeof(err)) == 1,
               "bind qkv");
        (void)i;
    }
    expect(h3_ulysses_a2a_qkv(&cfg, d_in[0], d_qkv[0], (size_t)seq_local,
                              (size_t)heads, (size_t)head_dim, 4, err,
                              sizeof(err)) == 1,
           "a2a_qkv");

    for (r = 0; r < 2; r++) {
        cudaSetDevice(r);
        cudaMemcpy(h_qkv[r], d_qkv[r], qkv_out_n * sizeof(float),
                   cudaMemcpyDeviceToHost);
        for (s_pad = 0; s_pad < seq_pad; s_pad++) {
            src = s_pad / seq_local;
            s = s_pad % seq_local;
            for (qkv = 0; qkv < 3; qkv++)
                for (hl = 0; hl < h_local; hl++)
                    for (d = 0; d < head_dim; d++) {
                        int hout = r * h_local + hl;
                        size_t idx = ((((size_t)s_pad * 3u + (size_t)qkv) *
                                       (size_t)h_local) +
                                      (size_t)hl) *
                                         (size_t)head_dim +
                                     (size_t)d;
                        float exp = qkv_in_at(src, s, qkv, hout, d, seq_local,
                                              heads, head_dim);
                        float dlt = fabsf(h_qkv[r][idx] - exp);
                        if (dlt > maxabs)
                            maxabs = dlt;
                    }
        }
    }
    expect(maxabs == 0.0f, "qkv a2a max-abs==0");
    printf("test_h3_ulysses_a2a: qkv_maxabs=%.6g\n", (double)maxabs);

    /* attn-out: each rank holds full seq, H/U heads (use Q slice of a2a_qkv). */
    maxabs = 0.0f;
    for (r = 0; r < 2; r++) {
        cudaSetDevice(r);
        for (s_pad = 0; s_pad < seq_pad; s_pad++)
            for (hl = 0; hl < h_local; hl++)
                for (d = 0; d < head_dim; d++) {
                    size_t src_idx = ((((size_t)s_pad * 3u + 0u) * (size_t)h_local) +
                                      (size_t)hl) *
                                         (size_t)head_dim +
                                     (size_t)d;
                    size_t dst_idx = ((size_t)s_pad * (size_t)h_local + (size_t)hl) *
                                         (size_t)head_dim +
                                     (size_t)d;
                    h_attn_in[r][dst_idx] = h_qkv[r][src_idx];
                }
        cudaMemcpy(d_attn_in[r], h_attn_in[r], attn_in_n * sizeof(float),
                   cudaMemcpyHostToDevice);
        expect(h3_ulysses_comm_bind(&cfg, r, d_attn_in[r], d_attn_out[r], err,
                                    sizeof(err)) == 1,
               "bind attn");
    }
    expect(h3_ulysses_a2a_attn_out(&cfg, d_attn_in[0], d_attn_out[0],
                                   (size_t)seq_local, (size_t)heads,
                                   (size_t)head_dim, 4, err, sizeof(err)) == 1,
           "a2a_attn_out");
    for (r = 0; r < 2; r++) {
        cudaSetDevice(r);
        cudaMemcpy(h_attn_out[r], d_attn_out[r], attn_out_n * sizeof(float),
                   cudaMemcpyDeviceToHost);
        for (s = 0; s < seq_local; s++)
            for (h = 0; h < heads; h++)
                for (d = 0; d < head_dim; d++) {
                    size_t idx = ((size_t)s * (size_t)heads + (size_t)h) *
                                     (size_t)head_dim +
                                 (size_t)d;
                    float exp = qkv_in_at(r, s, 0, h, d, seq_local, heads, head_dim);
                    float dlt = fabsf(h_attn_out[r][idx] - exp);
                    if (dlt > maxabs)
                        maxabs = dlt;
                }
    }
    expect(maxabs == 0.0f, "attn a2a max-abs==0");
    printf("test_h3_ulysses_a2a: attn_maxabs=%.6g\n", (double)maxabs);

    for (r = 0; r < 2; r++) {
        cudaSetDevice(r);
        cudaFree(d_in[r]);
        cudaFree(d_qkv[r]);
        cudaFree(d_attn_in[r]);
        cudaFree(d_attn_out[r]);
        free(h_in[r]);
        free(h_qkv[r]);
        free(h_attn_in[r]);
        free(h_attn_out[r]);
    }
    h3_ulysses_comm_destroy(&cfg);
    return fails ? 1 : (printf("test_h3_ulysses_a2a: PASS\n"), 0);
}
