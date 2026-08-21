/* G7: minimal numeric quality gate for CUDA bypass.
 * Asserts SDPA canary + optional audio decode + optional PPM PSNR pair.
 * Pure C; no Python. */
#include "h3_audio_vae_cuda.h"
#include "h3_gpu_cuda.h"
#include "h3_gpu_cuda_audio.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int gate_sdpa_smoke(h3_cuda *cuda, char *error, size_t error_size) {
    /* Tiny causal-like SDPA via audio helper as a numeric canary. */
    enum { B = 1, S = 8, H = 2, D = 16 };
    size_t n = (size_t)B * S * H * D;
    float *hq = NULL, *hk = NULL, *hv = NULL, *ho = NULL;
    float *dq = NULL, *dk = NULL, *dv = NULL, *dout = NULL;
    size_t i;
    int ok = 0;
    double sum = 0.0;

    hq = malloc(n * sizeof(float));
    hk = malloc(n * sizeof(float));
    hv = malloc(n * sizeof(float));
    ho = malloc(n * sizeof(float));
    if (!hq || !hk || !hv || !ho) {
        snprintf(error, error_size, "OOM sdpa gate");
        goto done;
    }
    for (i = 0; i < n; i++) {
        hq[i] = 0.01f * (float)((i % 7) - 3);
        hk[i] = 0.01f * (float)((i % 5) - 2);
        hv[i] = 0.02f * (float)((i % 3) - 1);
    }
    if (cudaMalloc((void **)&dq, n * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&dk, n * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&dv, n * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&dout, n * sizeof(float)) != cudaSuccess) {
        snprintf(error, error_size, "cudaMalloc sdpa gate");
        goto done;
    }
    cudaMemcpy(dq, hq, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dk, hk, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dv, hv, n * sizeof(float), cudaMemcpyHostToDevice);
    if (!h3_cuda_audio_sdpa_causal_f32(cuda, dout, dq, dk, dv, B, S, H, D,
                                       1.0f / sqrtf((float)D), error,
                                       error_size))
        goto done;
    cudaMemcpy(ho, dout, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (i = 0; i < n; i++) {
        if (!isfinite(ho[i])) {
            snprintf(error, error_size, "non-finite SDPA output");
            goto done;
        }
        sum += fabs((double)ho[i]);
    }
    if (!(sum > 1e-8)) {
        snprintf(error, error_size, "SDPA all-near-zero");
        goto done;
    }
    ok = 1;
done:
    free(hq);
    free(hk);
    free(hv);
    free(ho);
    if (dq) cudaFree(dq);
    if (dk) cudaFree(dk);
    if (dv) cudaFree(dv);
    if (dout) cudaFree(dout);
    return ok;
}

/* Load raw P6 PPM (no comments) into malloc'd RGB24. */
static uint8_t *load_ppm_rgb24(const char *path, int *w, int *h, char *error,
                               size_t error_size) {
    FILE *fp;
    char magic[8];
    int maxv = 0;
    size_t n;
    uint8_t *buf;
    fp = fopen(path, "rb");
    if (!fp) {
        snprintf(error, error_size, "cannot open ppm %s", path);
        return NULL;
    }
    if (fscanf(fp, "%7s", magic) != 1 || strcmp(magic, "P6") != 0) {
        fclose(fp);
        snprintf(error, error_size, "not P6 ppm: %s", path);
        return NULL;
    }
    if (fscanf(fp, "%d %d %d", w, h, &maxv) != 3 || *w < 1 || *h < 1 ||
        maxv != 255) {
        fclose(fp);
        snprintf(error, error_size, "bad ppm header: %s", path);
        return NULL;
    }
    fgetc(fp); /* single whitespace after maxval */
    n = (size_t)(*w) * (size_t)(*h) * 3;
    buf = (uint8_t *)malloc(n);
    if (!buf) {
        fclose(fp);
        snprintf(error, error_size, "OOM ppm");
        return NULL;
    }
    if (fread(buf, 1, n, fp) != n) {
        free(buf);
        fclose(fp);
        snprintf(error, error_size, "short ppm read: %s", path);
        return NULL;
    }
    fclose(fp);
    return buf;
}

static int gate_ppm_psnr(const char *a_path, const char *b_path, double min_psnr,
                         char *error, size_t error_size) {
    int wa = 0, ha = 0, wb = 0, hb = 0;
    uint8_t *a = NULL, *b = NULL;
    size_t n, i;
    double mse = 0.0, psnr;
    int ok = 0;
    a = load_ppm_rgb24(a_path, &wa, &ha, error, error_size);
    if (!a) return 0;
    b = load_ppm_rgb24(b_path, &wb, &hb, error, error_size);
    if (!b) {
        free(a);
        return 0;
    }
    if (wa != wb || ha != hb) {
        snprintf(error, error_size, "ppm size mismatch %dx%d vs %dx%d", wa, ha,
                 wb, hb);
        goto done;
    }
    n = (size_t)wa * (size_t)ha * 3;
    for (i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        mse += d * d;
    }
    mse /= (double)n;
    if (mse < 1e-12)
        psnr = 99.0;
    else
        psnr = 10.0 * log10(255.0 * 255.0 / mse);
    printf("quality_gate ppm_psnr=%.3f (min=%.3f) %s vs %s\n", psnr, min_psnr,
           a_path, b_path);
    if (psnr + 1e-6 < min_psnr) {
        snprintf(error, error_size, "PSNR %.3f < min %.3f", psnr, min_psnr);
        goto done;
    }
    ok = 1;
done:
    free(a);
    free(b);
    return ok;
}

int main(int argc, char **argv) {
    const char *audio_weights = NULL;
    const char *ppm_a = NULL, *ppm_b = NULL;
    double min_psnr = 20.0;
    int i;
    char error[512];
    h3_cuda *cuda;
    int failed = 0;

    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--audio-vae") && i + 1 < argc)
            audio_weights = argv[++i];
        else if (!strcmp(argv[i], "--ppm-a") && i + 1 < argc)
            ppm_a = argv[++i];
        else if (!strcmp(argv[i], "--ppm-b") && i + 1 < argc)
            ppm_b = argv[++i];
        else if (!strcmp(argv[i], "--min-psnr") && i + 1 < argc)
            min_psnr = strtod(argv[++i], NULL);
    }

    cuda = h3_cuda_create(0, error, sizeof(error));
    if (!cuda) {
        fprintf(stderr, "FAIL create: %s\n", error);
        return 1;
    }

    if (!gate_sdpa_smoke(cuda, error, sizeof(error))) {
        fprintf(stderr, "FAIL sdpa_gate: %s\n", error);
        failed = 1;
    } else {
        printf("PASS quality_gate sdpa_causal_canary\n");
    }

    if (audio_weights && !failed) {
        float latent[32 * 2 * 2];
        h3_audio_vae_cuda_waveform wave = {0};
        h3_audio_vae_cuda_stats stats = {0};
        size_t j, n;
        int finite = 1;
        memset(latent, 0, sizeof(latent));
        for (j = 0; j < sizeof(latent) / sizeof(latent[0]); j++)
            latent[j] = 0.01f * sinf((float)j * 0.17f);
        if (!h3_audio_vae_cuda_decode(cuda, audio_weights, latent, 2, NULL, NULL,
                                      &wave, &stats, error, sizeof(error))) {
            fprintf(stderr, "FAIL audio_decode_gate: %s\n", error);
            failed = 1;
        } else {
            n = (size_t)wave.channels * (size_t)wave.samples;
            for (j = 0; j < n; j++)
                if (!isfinite(wave.pcm[j])) finite = 0;
            if (!finite || wave.samples != 1600) {
                fprintf(stderr, "FAIL audio_decode geometry/finite\n");
                failed = 1;
            } else {
                printf("PASS quality_gate audio_decode T=2 samples=%d decode_ms=%.1f\n",
                       wave.samples, stats.decode_ms);
            }
            h3_audio_vae_cuda_waveform_free(&wave);
        }
    }

    if (ppm_a && ppm_b && !failed) {
        if (!gate_ppm_psnr(ppm_a, ppm_b, min_psnr, error, sizeof(error))) {
            fprintf(stderr, "FAIL ppm_psnr_gate: %s\n", error);
            failed = 1;
        } else {
            printf("PASS quality_gate ppm_psnr\n");
        }
    }

    h3_cuda_free(cuda);
    if (failed) {
        fprintf(stderr, "FAIL h3_quality_gate_probe\n");
        return 1;
    }
    printf("PASS h3_quality_gate_probe\n");
    return 0;
}
