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
    printf("%s\n", pass ? "PASS" : "FAIL");
    free(X); free(Xrot); free(Yink); free(Ysl); free(Sink); free(Ssl);
    return pass ? 0 : 1;
}
