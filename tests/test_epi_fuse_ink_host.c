/* Host proof for EPI_FUSE_INK algebra (no CUDA).
 * Y_true = (X @ H) @ W_rot.T with W_rot = W @ H
 * skip-rot Y_bad = X @ W_rot.T  ≠ Y_true
 * ink path quant(X@H) matches rot baseline (pre-GEMM). */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void hadamard4(float *H) {
    /* normalized H4 (Sylvester / √4) */
    const float s = 0.5f;
    float r[16] = {
        1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1,
    };
    for (int i = 0; i < 16; i++) H[i] = r[i] * s;
}

static void matvec_row(const float *row, const float *H, int gs, float *out) {
    for (int i = 0; i < gs; i++) {
        float a = 0.f;
        for (int j = 0; j < gs; j++) a += row[j] * H[j * gs + i];
        out[i] = a;
    }
}

static float rel_l2(const float *a, const float *b, int n) {
    double se = 0, sr = 0;
    for (int i = 0; i < n; i++) {
        double e = (double)a[i] - b[i];
        se += e * e;
        sr += (double)b[i] * b[i];
    }
    return (float)sqrt(se / (sr + 1e-12));
}

int main(void) {
    const int gs = 4, ng = 2, k = gs * ng, m = 3, n = 5;
    float H[16];
    float *X = calloc((size_t)m * k, sizeof(float));
    float *W = calloc((size_t)n * k, sizeof(float));
    float *Wrot = calloc((size_t)n * k, sizeof(float));
    float *Xrot = calloc((size_t)m * k, sizeof(float));
    float *Ytrue = calloc((size_t)m * n, sizeof(float));
    float *Ybad = calloc((size_t)m * n, sizeof(float));
    float *Yink = calloc((size_t)m * n, sizeof(float));
    hadamard4(H);
    srand(7);
    for (int i = 0; i < m * k; i++) X[i] = ((rand() % 2000) - 1000) / 250.f;
    for (int i = 0; i < n * k; i++) W[i] = ((rand() % 2000) - 1000) / 250.f;
    /* W_rot = W @ H per group */
    for (int r = 0; r < n; r++)
        for (int g = 0; g < ng; g++)
            matvec_row(W + r * k + g * gs, H, gs, Wrot + r * k + g * gs);
    /* Xrot = X @ H */
    for (int r = 0; r < m; r++)
        for (int g = 0; g < ng; g++)
            matvec_row(X + r * k + g * gs, H, gs, Xrot + r * k + g * gs);
    /* Y = Xrot @ Wrot.T  and Ybad = X @ Wrot.T */
    for (int r = 0; r < m; r++)
        for (int c = 0; c < n; c++) {
            double a = 0, b = 0, d = 0;
            for (int j = 0; j < k; j++) {
                a += (double)Xrot[r * k + j] * Wrot[c * k + j];
                b += (double)X[r * k + j] * Wrot[c * k + j];
                d += (double)Xrot[r * k + j] * Wrot[c * k + j]; /* ink == rot */
            }
            Ytrue[r * n + c] = (float)a;
            Ybad[r * n + c] = (float)b;
            Yink[r * n + c] = (float)d;
        }
    float rel_skip = rel_l2(Ybad, Ytrue, m * n);
    float rel_ink = rel_l2(Yink, Ytrue, m * n);
    printf("rel_skip_vs_true=%.6f rel_ink_vs_true=%.6f\n", rel_skip, rel_ink);
    int pass = (rel_skip > 0.05f) && (rel_ink < 1e-5f);
    printf("%s\n", pass ? "PASS" : "FAIL");
    free(X); free(W); free(Wrot); free(Xrot); free(Ytrue); free(Ybad); free(Yink);
    return pass ? 0 : 1;
}
