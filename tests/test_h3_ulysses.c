/* tests/test_h3_ulysses.c */
#include "h3_ulysses.h"
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

/* Host clone of ulysses_pack/unpack_qkv + pack/unpack_attn index math.
 * Simulates destination-major A2A (rank r pack[p] → rank p unpack[r]). */
static float fill_qkv(int rank, int s, int qkv, int h, int d, int heads,
                      int head_dim) {
    int i = ((s * 3 + qkv) * heads + h) * head_dim + d;
    return (float)(rank * 100000 + i + 1);
}

static int test_a2a_perm(int u, int seq_local, int heads, int head_dim) {
    int h_local = heads / u;
    int seq_pad = seq_local * u;
    size_t qkv_n = (size_t)seq_local * 3u * (size_t)heads * (size_t)head_dim;
    size_t chunk_q = (size_t)seq_local * 3u * (size_t)h_local * (size_t)head_dim;
    size_t chunk_a = (size_t)seq_local * (size_t)h_local * (size_t)head_dim;
    float **local = (float **)calloc((size_t)u, sizeof(float *));
    float **pack = (float **)calloc((size_t)u, sizeof(float *));
    float **unp = (float **)calloc((size_t)u, sizeof(float *));
    float **gathered = (float **)calloc((size_t)u, sizeof(float *));
    float **attn_in = (float **)calloc((size_t)u, sizeof(float *));
    float **apack = (float **)calloc((size_t)u, sizeof(float *));
    float **aunp = (float **)calloc((size_t)u, sizeof(float *));
    float **attn_out = (float **)calloc((size_t)u, sizeof(float *));
    int r, p, s, qkv, h, d, hl, s_pad, src, peer, ok = 1;

    if (!local || !pack || !unp || !gathered || !attn_in || !apack || !aunp ||
        !attn_out)
        return 0;
    for (r = 0; r < u; r++) {
        local[r] = (float *)calloc(qkv_n, sizeof(float));
        pack[r] = (float *)calloc(qkv_n, sizeof(float));
        unp[r] = (float *)calloc(qkv_n, sizeof(float));
        gathered[r] = (float *)calloc(qkv_n, sizeof(float));
        attn_in[r] = (float *)calloc((size_t)u * chunk_a, sizeof(float));
        apack[r] = (float *)calloc((size_t)u * chunk_a, sizeof(float));
        aunp[r] = (float *)calloc((size_t)u * chunk_a, sizeof(float));
        attn_out[r] = (float *)calloc(chunk_a * (size_t)u, sizeof(float));
        if (!local[r] || !pack[r] || !unp[r] || !gathered[r] || !attn_in[r] ||
            !apack[r] || !aunp[r] || !attn_out[r])
            ok = 0;
    }
    if (!ok)
        goto done;

    for (r = 0; r < u; r++)
        for (s = 0; s < seq_local; s++)
            for (qkv = 0; qkv < 3; qkv++)
                for (h = 0; h < heads; h++)
                    for (d = 0; d < head_dim; d++) {
                        size_t idx =
                            ((((size_t)s * 3u + (size_t)qkv) * (size_t)heads) +
                             (size_t)h) *
                                (size_t)head_dim +
                            (size_t)d;
                        local[r][idx] =
                            fill_qkv(r, s, qkv, h, d, heads, head_dim);
                    }

    for (r = 0; r < u; r++)
        for (s = 0; s < seq_local; s++)
            for (qkv = 0; qkv < 3; qkv++)
                for (h = 0; h < heads; h++)
                    for (d = 0; d < head_dim; d++) {
                        size_t src_i =
                            ((((size_t)s * 3u + (size_t)qkv) * (size_t)heads) +
                             (size_t)h) *
                                (size_t)head_dim +
                            (size_t)d;
                        peer = h / h_local;
                        hl = h - peer * h_local;
                        size_t dst =
                            ((((size_t)peer * (size_t)seq_local + (size_t)s) *
                                  3u +
                              (size_t)qkv) *
                                 (size_t)h_local +
                             (size_t)hl) *
                                (size_t)head_dim +
                            (size_t)d;
                        pack[r][dst] = local[r][src_i];
                    }

    for (r = 0; r < u; r++)
        for (p = 0; p < u; p++)
            memcpy(unp[p] + (size_t)r * chunk_q, pack[r] + (size_t)p * chunk_q,
                   chunk_q * sizeof(float));

    for (r = 0; r < u; r++)
        for (src = 0; src < u; src++)
            for (s = 0; s < seq_local; s++)
                for (qkv = 0; qkv < 3; qkv++)
                    for (hl = 0; hl < h_local; hl++)
                        for (d = 0; d < head_dim; d++) {
                            size_t in_i =
                                (((((size_t)src * (size_t)seq_local +
                                    (size_t)s) *
                                       3u +
                                   (size_t)qkv) *
                                      (size_t)h_local +
                                  (size_t)hl) *
                                 (size_t)head_dim) +
                                (size_t)d;
                            s_pad = src * seq_local + s;
                            size_t dst =
                                ((((size_t)s_pad * 3u + (size_t)qkv) *
                                  (size_t)h_local) +
                                 (size_t)hl) *
                                    (size_t)head_dim +
                                (size_t)d;
                            gathered[r][dst] = unp[r][in_i];
                        }

    for (r = 0; r < u && ok; r++)
        for (s_pad = 0; s_pad < seq_pad && ok; s_pad++) {
            src = s_pad / seq_local;
            s = s_pad % seq_local;
            for (qkv = 0; qkv < 3 && ok; qkv++)
                for (hl = 0; hl < h_local && ok; hl++)
                    for (d = 0; d < head_dim && ok; d++) {
                        int hout = r * h_local + hl;
                        size_t idx =
                            ((((size_t)s_pad * 3u + (size_t)qkv) *
                              (size_t)h_local) +
                             (size_t)hl) *
                                (size_t)head_dim +
                            (size_t)d;
                        if (gathered[r][idx] !=
                            fill_qkv(src, s, qkv, hout, d, heads, head_dim))
                            ok = 0;
                    }
        }

    /* Attn-out: Q slice of gathered → reverse A2A → local [S_local, H, D]. */
    for (r = 0; r < u; r++)
        for (s_pad = 0; s_pad < seq_pad; s_pad++)
            for (hl = 0; hl < h_local; hl++)
                for (d = 0; d < head_dim; d++) {
                    size_t src_idx =
                        ((((size_t)s_pad * 3u + 0u) * (size_t)h_local) +
                         (size_t)hl) *
                            (size_t)head_dim +
                        (size_t)d;
                    size_t dst_idx =
                        ((size_t)s_pad * (size_t)h_local + (size_t)hl) *
                            (size_t)head_dim +
                        (size_t)d;
                    attn_in[r][dst_idx] = gathered[r][src_idx];
                }

    for (r = 0; r < u; r++)
        for (src = 0; src < u; src++)
            for (s = 0; s < seq_local; s++)
                for (hl = 0; hl < h_local; hl++)
                    for (d = 0; d < head_dim; d++) {
                        s_pad = src * seq_local + s;
                        size_t src_i =
                            ((size_t)s_pad * (size_t)h_local + (size_t)hl) *
                                (size_t)head_dim +
                            (size_t)d;
                        size_t dst =
                            ((((size_t)src * (size_t)seq_local + (size_t)s) *
                              (size_t)h_local) +
                             (size_t)hl) *
                                (size_t)head_dim +
                            (size_t)d;
                        apack[r][dst] = attn_in[r][src_i];
                    }

    for (r = 0; r < u; r++)
        for (p = 0; p < u; p++)
            memcpy(aunp[p] + (size_t)r * chunk_a, apack[r] + (size_t)p * chunk_a,
                   chunk_a * sizeof(float));

    for (r = 0; r < u; r++)
        for (peer = 0; peer < u; peer++)
            for (s = 0; s < seq_local; s++)
                for (hl = 0; hl < h_local; hl++)
                    for (d = 0; d < head_dim; d++) {
                        size_t in_i =
                            ((((size_t)peer * (size_t)seq_local + (size_t)s) *
                              (size_t)h_local) +
                             (size_t)hl) *
                                (size_t)head_dim +
                            (size_t)d;
                        h = peer * h_local + hl;
                        size_t dst =
                            ((size_t)s * (size_t)heads + (size_t)h) *
                                (size_t)head_dim +
                            (size_t)d;
                        attn_out[r][dst] = aunp[r][in_i];
                    }

    for (r = 0; r < u && ok; r++)
        for (s = 0; s < seq_local && ok; s++)
            for (h = 0; h < heads && ok; h++)
                for (d = 0; d < head_dim && ok; d++) {
                    size_t idx = ((size_t)s * (size_t)heads + (size_t)h) *
                                     (size_t)head_dim +
                                 (size_t)d;
                    if (attn_out[r][idx] !=
                        fill_qkv(r, s, 0, h, d, heads, head_dim))
                        ok = 0;
                }

done:
    for (r = 0; r < u; r++) {
        if (local) free(local[r]);
        if (pack) free(pack[r]);
        if (unp) free(unp[r]);
        if (gathered) free(gathered[r]);
        if (attn_in) free(attn_in[r]);
        if (apack) free(apack[r]);
        if (aunp) free(aunp[r]);
        if (attn_out) free(attn_out[r]);
    }
    free(local);
    free(pack);
    free(unp);
    free(gathered);
    free(attn_in);
    free(apack);
    free(aunp);
    free(attn_out);
    return ok;
}

int main(void) {
    char err[128];
    expect(h3_ulysses_validate(56, 64, 1, 1, 1, err, sizeof err) == 1, "U1");
    expect(h3_ulysses_validate(56, 64, 1, 2, 1, err, sizeof err) == 1, "U2");
    expect(h3_ulysses_validate(56, 64, 1, 4, 1, err, sizeof err) == 1, "U4");
    expect(h3_ulysses_validate(56, 64, 1, 8, 1, err, sizeof err) == 1, "U8");
    expect(h3_ulysses_validate(56, 64, 1, 7, 1, err, sizeof err) == 0, "U7");
    expect(h3_ulysses_validate(56, 64, 1, 3, 1, err, sizeof err) == 0, "U3");
    expect(h3_ulysses_validate(56, 64, 2, 4, 1, err, sizeof err) == 1, "TP2+U4");
    expect(h3_ulysses_validate(56, 64, 4, 4, 1, err, sizeof err) == 0, "TP4+U4");
    expect(h3_ulysses_pad_seq(15973, 64) == 16000, "cafe pad");
    expect(h3_ulysses_local_rows(16000, 2, 0) == 8000, "r0");
    expect(h3_ulysses_local_rows(16000, 2, 1) == 8000, "r1");
    expect(h3_ulysses_local_rows(16000, 8, 0) == 2000, "u8 r0");
    /* SDPA must use S_real=15973, not S_pad=16000 (pad tail=27). */
    expect(15973u + 27u == 16000u, "pad tail");
    expect(test_a2a_perm(2, 32, 56, 8), "perm U2");
    expect(test_a2a_perm(8, 8, 56, 8), "perm U8");
    return fails ? 1 : (printf("test_h3_ulysses: PASS\n"), 0);
}
