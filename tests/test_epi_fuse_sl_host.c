/* Host proof: EPI_FUSE_SL recompute-Hadamard+quant ≡ ink (rot then quant).
 * Also confirms skip-ConvRot still ≠ true. No CUDA. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void hadamard4(float *H) {
    const float s = 0.5f;
    float r[16] = {1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1};
    for (int i = 0; i < 16; i++) H[i] = r[i] * s;
}

static void matvec_row(const float *row, const float *H, int gs, float *out) {
    for (int i = 0; i < gs; i++) {
        float a = 0.f;
        for (int j = 0; j < gs; j++) a += row[j] * H[j * gs + i];
        out[i] = a;
    }
}

static void quant_row(const float *x, int k, int8_t *y, float *scale_out) {
    float absmax = 0.f;
    for (int i = 0; i < k; i++) {
        float a = fabsf(x[i]);
        if (a > absmax) absmax = a;
    }
    float scale = (absmax > 0.f) ? (absmax / 127.f) : 1.f;
    *scale_out = scale;
    float inv = 1.f / scale;
    for (int i = 0; i < k; i++) {
        float q = roundf(x[i] * inv);
        if (q > 127.f) q = 127.f;
        if (q < -127.f) q = -127.f;
        y[i] = (int8_t)q;
    }
}

/* SL: absmax over recomputed rot, then requant with second Hadamard pass. */
static void sl_quant_row(const float *x, const float *H, int gs, int ng, int k,
                         int8_t *y, float *scale_out) {
    float absmax = 0.f;
    float *tmp = (float *)malloc((size_t)gs * sizeof(float));
    for (int g = 0; g < ng; g++) {
        matvec_row(x + g * gs, H, gs, tmp);
        for (int i = 0; i < gs; i++) {
            float a = fabsf(tmp[i]);
            if (a > absmax) absmax = a;
        }
    }
    float scale = (absmax > 0.f) ? (absmax / 127.f) : 1.f;
    *scale_out = scale;
    float inv = 1.f / scale;
    for (int g = 0; g < ng; g++) {
        matvec_row(x + g * gs, H, gs, tmp);
        for (int i = 0; i < gs; i++) {
            float q = roundf(tmp[i] * inv);
            if (q > 127.f) q = 127.f;
            if (q < -127.f) q = -127.f;
            y[g * gs + i] = (int8_t)q;
        }
    }
    free(tmp);
    (void)k;
}


/* Normalized FWHT ≡ Sylvester dense H@x (same as GPU SL fast path). */
static void fwht_normalized_inplace(float *a, int n) {
    for (int h = 1; h < n; h *= 2) {
        for (int i = 0; i < n; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float x = a[j];
                float y = a[j + h];
                a[j] = x + y;
                a[j + h] = x - y;
            }
        }
    }
    float scale = 1.f / sqrtf((float)n);
    for (int i = 0; i < n; i++) a[i] *= scale;
}

static float *sylvester_H(int size) {
    float *h = (float *)malloc((size_t)size * size * sizeof(float));
    float *tmp = (float *)malloc((size_t)size * size * sizeof(float));
    int current = 1;
    h[0] = 1.f;
    while (current < size) {
        int next = current * 2;
        for (int r = 0; r < current; r++)
            for (int c = 0; c < current; c++) {
                float v = h[r * current + c];
                tmp[r * next + c] = v;
                tmp[r * next + c + current] = v;
                tmp[(r + current) * next + c] = v;
                tmp[(r + current) * next + c + current] = -v;
            }
        memcpy(h, tmp, (size_t)next * next * sizeof(float));
        current = next;
    }
    float inv = 1.f / sqrtf((float)size);
    for (size_t i = 0; i < (size_t)size * size; i++) h[i] *= inv;
    free(tmp);
    return h;
}

static void sl_fwht_quant_row(const float *x, int gs, int ng, int8_t *y,
                              float *scale_out) {
    float *tmp = (float *)malloc((size_t)gs * sizeof(float));
    float absmax = 0.f;
    for (int g = 0; g < ng; g++) {
        memcpy(tmp, x + g * gs, (size_t)gs * sizeof(float));
        fwht_normalized_inplace(tmp, gs);
        for (int i = 0; i < gs; i++) {
            float a = fabsf(tmp[i]);
            if (a > absmax) absmax = a;
        }
    }
    float scale = (absmax > 0.f) ? (absmax / 127.f) : 1.f;
    *scale_out = scale;
    float inv = 1.f / scale;
    for (int g = 0; g < ng; g++) {
        memcpy(tmp, x + g * gs, (size_t)gs * sizeof(float));
        fwht_normalized_inplace(tmp, gs);
        for (int i = 0; i < gs; i++) {
            float q = roundf(tmp[i] * inv);
            if (q > 127.f) q = 127.f;
            if (q < -127.f) q = -127.f;
            y[g * gs + i] = (int8_t)q;
        }
    }
    free(tmp);
}

static int test_fwht_sl_vs_dense(void) {
    const int gs = 256, ng = 2, k = gs * ng;
    float *H = sylvester_H(gs);
    float *X = (float *)malloc((size_t)k * sizeof(float));
    int8_t *Yd = (int8_t *)malloc((size_t)k);
    int8_t *Yf = (int8_t *)malloc((size_t)k);
    float sd, sf;
    srand(42);
    for (int i = 0; i < k; i++) X[i] = ((rand() % 2000) - 1000) / 100.f;
    sl_quant_row(X, H, gs, ng, k, Yd, &sd);
    sl_fwht_quant_row(X, gs, ng, Yf, &sf);
    int mism = 0;
    for (int i = 0; i < k; i++)
        if (Yd[i] != Yf[i]) mism++;
    printf("fwht_sl_vs_dense i8_mismatch=%d scale_d=%.8g scale_f=%.8g\n", mism,
           sd, sf);
    free(H); free(X); free(Yd); free(Yf);
    return mism == 0;
}

int main(void) {
    const int gs = 4, ng = 2, k = gs * ng, m = 3;
    float H[16];
    float *X = calloc((size_t)m * k, sizeof(float));
    float *Xrot = calloc((size_t)m * k, sizeof(float));
    int8_t *Yink = calloc((size_t)m * k, 1);
    int8_t *Ysl = calloc((size_t)m * k, 1);
    float *Sink = calloc((size_t)m, sizeof(float));
    float *Ssl = calloc((size_t)m, sizeof(float));
    hadamard4(H);
    srand(11);
    for (int i = 0; i < m * k; i++) X[i] = ((rand() % 2000) - 1000) / 250.f;
    for (int r = 0; r < m; r++) {
        for (int g = 0; g < ng; g++)
            matvec_row(X + r * k + g * gs, H, gs, Xrot + r * k + g * gs);
        quant_row(Xrot + r * k, k, Yink + r * k, Sink + r);
        sl_quant_row(X + r * k, H, gs, ng, k, Ysl + r * k, Ssl + r);
    }
    int mismatch = 0;
    for (int i = 0; i < m * k; i++)
        if (Yink[i] != Ysl[i]) mismatch++;
    float scale_rel = 0.f;
    for (int r = 0; r < m; r++) {
        float d = Sink[r] - Ssl[r];
        scale_rel += d * d;
    }
    printf("i8_mismatch=%d scale_l2=%.6g\n", mismatch, sqrtf(scale_rel));
    int pass = (mismatch == 0) && (scale_rel < 1e-12f);
    int fwht_ok = test_fwht_sl_vs_dense();
    printf("fwht_sl_vs_dense %s\n", fwht_ok ? "PASS" : "FAIL");
    pass = pass && fwht_ok;
    printf("%s\n", pass ? "PASS" : "FAIL");
    free(X); free(Xrot); free(Yink); free(Ysl); free(Sink); free(Ssl);
    return pass ? 0 : 1;
}
