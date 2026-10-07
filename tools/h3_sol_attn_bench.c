/* Isolated Sol-Attn vs cuDNN SDPA at cafe packed shape.
 * Does not change generate defaults. Opt-in generate: --sdpa-backend sol.
 *
 *   cmake --build build-cuda -j --target h3_sol_attn_bench
 *   ./build-cuda/h3_sol_attn_bench [--seq 15973] [--heads 56] [--dim 128]
 */
#include "h3_gpu_cuda.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t xorshift(uint32_t *s) {
    uint32_t x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    return x;
}

static float unit_noise(uint32_t *s) {
    return ((int32_t)xorshift(s) / 2147483648.0f);
}

static void fill_qkv(float *q, float *k, float *v, size_t seq, size_t heads,
                     size_t dim, int gaussian) {
    size_t i, h, d;
    uint32_t rng = 0xC0FFEEu;
    for (i = 0; i < seq; i++) {
        for (h = 0; h < heads; h++) {
            float loc = sinf((float)i / 64.0f + 0.07f * (float)h);
            for (d = 0; d < dim; d++) {
                size_t idx = (i * heads + h) * dim + d;
                if (gaussian) {
                    q[idx] = unit_noise(&rng);
                    k[idx] = unit_noise(&rng);
                    v[idx] = unit_noise(&rng);
                } else {
                    float nd = 0.02f * (float)d / (float)dim;
                    q[idx] = loc + 0.05f * unit_noise(&rng) + nd;
                    k[idx] = loc + 0.05f * unit_noise(&rng) - nd;
                    v[idx] = loc * 0.4f + 0.08f * unit_noise(&rng);
                }
            }
        }
    }
}

static int time_backend(h3_cuda *cuda, const char *backend, float *dq, float *dk,
                        float *dv, float *dout, size_t seq, size_t heads,
                        size_t dim, float scale, int warmup, int iters,
                        double *ms_out, char *be_out, size_t be_n, char *err,
                        size_t err_n) {
    int i;
    double acc = 0.0;
    setenv("H3_CUDA_SDPA_BACKEND", backend, 1);
    if (!strcmp(backend, "cudnn")) {
        if (!h3_cuda_sdpa_prebuild(cuda, seq, heads, dim, scale, 0, err, err_n))
            fprintf(stderr, "prebuild %s: %s (continue)\n", backend, err);
    }
    for (i = 0; i < warmup; i++) {
        double ms = 0.0;
        if (!h3_cuda_sdpa_f16_dev_ex(cuda, dq, dk, dv, seq, heads, dim, scale, 0,
                                     dout, &ms, be_out, be_n, err, err_n)) {
            fprintf(stderr, "warmup %s failed: %s\n", backend, err);
            return 0;
        }
    }
    for (i = 0; i < iters; i++) {
        double ms = 0.0;
        if (!h3_cuda_sdpa_f16_dev_ex(cuda, dq, dk, dv, seq, heads, dim, scale, 0,
                                     dout, &ms, be_out, be_n, err, err_n)) {
            fprintf(stderr, "iter %s failed: %s\n", backend, err);
            return 0;
        }
        acc += ms;
    }
    *ms_out = acc / (double)iters;
    return 1;
}

static double rel_l2(const float *a, const float *b, size_t n) {
    double num = 0.0, den = 0.0;
    size_t i;
    for (i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        num += d * d;
        den += (double)b[i] * (double)b[i];
    }
    if (den < 1e-20) return num;
    return sqrt(num / den);
}

int main(int argc, char **argv) {
    size_t seq = 15973, heads = 56, dim = 128;
    int warmup = 2, iters = 4, gaussian = 0, iarg;
    int device = 0;
    int skip_sol = 0, skip_flash = 0;
    char error[512], be[64];
    h3_cuda *cuda = NULL;
    float *hq = NULL, *hk = NULL, *hv = NULL, *ho_ref = NULL, *ho_sol = NULL;
    float *dq = NULL, *dk = NULL, *dv = NULL, *dout = NULL;
    size_t elems, bytes;
    float scale;
    double ms_cudnn = -1.0, ms_sol = -1.0, ms_flash = -1.0;
    int ok_cudnn = 0, ok_sol = 0, ok_flash = 0;
    int rc = 1;

    for (iarg = 1; iarg < argc; iarg++) {
        if (!strcmp(argv[iarg], "--seq") && iarg + 1 < argc)
            seq = (size_t)strtoul(argv[++iarg], NULL, 10);
        else if (!strcmp(argv[iarg], "--heads") && iarg + 1 < argc)
            heads = (size_t)strtoul(argv[++iarg], NULL, 10);
        else if (!strcmp(argv[iarg], "--dim") && iarg + 1 < argc)
            dim = (size_t)strtoul(argv[++iarg], NULL, 10);
        else if (!strcmp(argv[iarg], "--warmup") && iarg + 1 < argc)
            warmup = atoi(argv[++iarg]);
        else if (!strcmp(argv[iarg], "--iters") && iarg + 1 < argc)
            iters = atoi(argv[++iarg]);
        else if (!strcmp(argv[iarg], "--gaussian"))
            gaussian = 1;
        else if (!strcmp(argv[iarg], "--skip-sol"))
            skip_sol = 1;
        else if (!strcmp(argv[iarg], "--skip-flash"))
            skip_flash = 1;
        else if (!strcmp(argv[iarg], "--device") && iarg + 1 < argc)
            device = atoi(argv[++iarg]);
        else {
            fprintf(stderr, "unknown arg %s\n", argv[iarg]);
            return 2;
        }
    }

    elems = seq * heads * dim;
    bytes = elems * sizeof(float);
    scale = 1.0f / sqrtf((float)dim);
    printf("h3_sol_attn_bench seq=%zu heads=%zu dim=%zu elems=%.2fM fill=%s "
           "warmup=%d iters=%d\n",
           seq, heads, dim, (double)elems / 1e6,
           gaussian ? "gaussian" : "local-corr", warmup, iters);

    cuda = h3_cuda_create(device, error, sizeof(error));
    if (!cuda) {
        fprintf(stderr, "h3_cuda_create: %s\n", error);
        return 1;
    }
    hq = (float *)malloc(bytes);
    hk = (float *)malloc(bytes);
    hv = (float *)malloc(bytes);
    ho_ref = (float *)malloc(bytes);
    ho_sol = (float *)malloc(bytes);
    if (!hq || !hk || !hv || !ho_ref || !ho_sol) {
        fprintf(stderr, "host OOM\n");
        goto done;
    }
    fill_qkv(hq, hk, hv, seq, heads, dim, gaussian);
    if (cudaMalloc((void **)&dq, bytes) != cudaSuccess ||
        cudaMalloc((void **)&dk, bytes) != cudaSuccess ||
        cudaMalloc((void **)&dv, bytes) != cudaSuccess ||
        cudaMalloc((void **)&dout, bytes) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc QKV/O failed\n");
        goto done;
    }
    cudaMemcpy(dq, hq, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dk, hk, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, hv, bytes, cudaMemcpyHostToDevice);

    ok_cudnn = time_backend(cuda, "cudnn", dq, dk, dv, dout, seq, heads, dim,
                            scale, warmup, iters, &ms_cudnn, be, sizeof(be),
                            error, sizeof(error));
    if (ok_cudnn) {
        cudaMemcpy(ho_ref, dout, bytes, cudaMemcpyDeviceToHost);
        printf("cudnn  backend=%s  %.2f ms/iter\n", be, ms_cudnn);
    }

    ok_sol = 0;
    if (!skip_sol)
        ok_sol = time_backend(cuda, "sol", dq, dk, dv, dout, seq, heads, dim, scale,
                              warmup, iters, &ms_sol, be, sizeof(be), error,
                              sizeof(error));
    if (ok_sol) {
        cudaMemcpy(ho_sol, dout, bytes, cudaMemcpyDeviceToHost);
        printf("sol    backend=%s  %.2f ms/iter  exact_frac=%.3f  rel_l2_vs_cudnn=%.4f\n",
               be, ms_sol, h3_cuda_sol_last_exact_frac(),
               ok_cudnn ? rel_l2(ho_sol, ho_ref, elems) : -1.0);
    }

    ok_flash = 0;
    if (!skip_flash)
        ok_flash = time_backend(cuda, "flash", dq, dk, dv, dout, seq, heads, dim,
                                scale, 1, 1, &ms_flash, be, sizeof(be), error,
                                sizeof(error));
    if (ok_flash)
        printf("flash  backend=%s  %.2f ms (1 iter, context only)\n", be,
               ms_flash);

    printf("\n=== verdict ===\n");
    if (!ok_cudnn) {
        printf("cuDNN failed; abort.\n");
        goto done;
    }
    if (skip_sol) {
        printf("cudnn-only  %.2f ms/iter  (nested-loop sol skipped)\n",
               ms_cudnn);
        rc = 0;
        goto done;
    }
    if (!ok_sol) {
        printf("sol kernel failed; do not wire as default. STOP.\n");
        goto done;
    }
    if (ms_sol + 1.0 < ms_cudnn * 0.92) {
        printf("sol faster than cuDNN (%.2f vs %.2f ms). Eligible for opt-in "
               "generate A/B. Default stays cuDNN.\n",
               ms_sol, ms_cudnn);
        rc = 0;
    } else {
        printf("sol NOT stably faster than cuDNN (%.2f vs %.2f ms, "
               "need <92%%). STOP. Do not open quality path. Default cuDNN.\n",
               ms_sol, ms_cudnn);
        rc = 0; /* bench itself succeeded */
    }

done:
    if (dq) cudaFree(dq);
    if (dk) cudaFree(dk);
    if (dv) cudaFree(dv);
    if (dout) cudaFree(dout);
    free(hq);
    free(hk);
    free(hv);
    free(ho_ref);
    free(ho_sol);
    h3_cuda_free(cuda);
    return rc;
}
