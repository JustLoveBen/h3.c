/* Au6: CUDA AudioVAE encode smoke (Comfy FP32). */
#include "h3_audio_vae_cuda.h"
#include "h3_gpu_cuda.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void on_progress(int done, int total, void *opaque) {
    (void)opaque;
    fprintf(stderr, "encode progress %d/%d\n", done, total);
}

int main(int argc, char **argv) {
    const char *weights = NULL;
    int samples = 1600;
    int i, ch, t;
    char error[512];
    h3_cuda *cuda = NULL;
    float *pcm = NULL;
    h3_audio_vae_cuda_latent latent = {0};
    h3_audio_vae_cuda_waveform wave = {0};
    h3_audio_vae_cuda_stats stats = {0};
    double sum = 0.0, sq = 0.0;
    size_t n;
    int finite = 1, nonzero = 0;

    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--weights") && i + 1 < argc)
            weights = argv[++i];
        else if (!strcmp(argv[i], "--samples") && i + 1 < argc)
            samples = atoi(argv[++i]);
    }
    if (!weights) {
        fprintf(stderr, "usage: %s --weights AUDIO_VAE.safetensors [--samples N]\n",
                argv[0]);
        return 2;
    }

    cuda = h3_cuda_create(0, error, sizeof(error));
    if (!cuda) {
        fprintf(stderr, "cuda create failed: %s\n", error);
        return 1;
    }
    pcm = calloc((size_t)2 * (size_t)samples, sizeof(*pcm));
    if (!pcm) {
        fprintf(stderr, "OOM pcm\n");
        return 1;
    }
    for (ch = 0; ch < 2; ch++)
        for (t = 0; t < samples; t++)
            pcm[(size_t)ch * samples + t] =
                0.25f * sinf(2.0f * 3.14159265f * 440.0f * (float)t / 32000.0f +
                             (float)ch);

    if (!h3_audio_vae_cuda_encode(cuda, weights, pcm, samples, on_progress, NULL,
                                  &latent, &stats, error, sizeof(error))) {
        fprintf(stderr, "encode failed: %s\n", error);
        free(pcm);
        h3_cuda_free(cuda);
        return 1;
    }

    n = (size_t)latent.channels * latent.stereo * latent.length;
    for (i = 0; i < (int)n; i++) {
        float v = latent.values[i];
        if (!isfinite(v)) finite = 0;
        if (fabsf(v) > 1e-6f) nonzero = 1;
        sum += v;
        sq += (double)v * v;
    }

    printf("PASS h3_audio_vae_cuda_encode_probe\n");
    printf("  latent=[%d,%d,%d] pcm_samples=%d\n", latent.channels, latent.stereo,
           latent.length, samples);
    printf("  load_ms=%.1f encode_ms=%.1f total_ms=%.1f\n", stats.load_ms,
           stats.encode_ms, stats.total_ms);
    printf("  mean=%.6g rms=%.6g finite=%d nonzero=%d\n", sum / (double)n,
           sqrt(sq / (double)n), finite, nonzero);
    printf("  audio_ms=%.1f\n", stats.encode_ms);

    if (!finite || !nonzero || latent.length != (samples + 799) / 800) {
        fprintf(stderr, "FAIL numeric/geometry gate\n");
        h3_audio_vae_cuda_latent_free(&latent);
        free(pcm);
        h3_cuda_free(cuda);
        return 1;
    }

    /* Optional roundtrip smoke: encode→decode must stay finite. */
    if (!h3_audio_vae_cuda_decode(cuda, weights, latent.values, latent.length,
                                  NULL, NULL, &wave, NULL, error,
                                  sizeof(error))) {
        fprintf(stderr, "roundtrip decode failed: %s\n", error);
        h3_audio_vae_cuda_latent_free(&latent);
        free(pcm);
        h3_cuda_free(cuda);
        return 1;
    }
    printf("  roundtrip_pcm_samples=%d\n", wave.samples);
    h3_audio_vae_cuda_waveform_free(&wave);
    h3_audio_vae_cuda_latent_free(&latent);
    free(pcm);
    h3_cuda_free(cuda);
    return 0;
}
