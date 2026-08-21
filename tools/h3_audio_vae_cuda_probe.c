/* Au2: CUDA AudioVAE decode smoke (Comfy FP32 safetensors). */
#include "h3_audio_vae_cuda.h"
#include "h3_gpu_cuda.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void die(const char *msg) {
    fprintf(stderr, "FAIL h3_audio_vae_cuda_probe: %s\n", msg);
    exit(1);
}

static void on_progress(int done, int total, void *opaque) {
    (void)opaque;
    fprintf(stderr, "audio_vae_cuda stage %d/%d\n", done, total);
}

int main(int argc, char **argv) {
    const char *weights =
        "/home/haiyang/h3_c/models/comfy-org/vae/minimax_h3_audio_vae_fp32.safetensors";
    int latent_t = 2;
    int device = 0;
    char error[512];
    h3_cuda *cuda = NULL;
    h3_audio_vae_cuda_waveform wave = {0};
    h3_audio_vae_cuda_stats stats = {0};
    float *latent = NULL;
    size_t n_lat, i;
    double absmax = 0.0, rms = 0.0;
    int finite = 1;
    int iarg;

    for (iarg = 1; iarg < argc; iarg++) {
        if (!strcmp(argv[iarg], "--weights") && iarg + 1 < argc)
            weights = argv[++iarg];
        else if (!strcmp(argv[iarg], "--latent-t") && iarg + 1 < argc)
            latent_t = atoi(argv[++iarg]);
        else if (!strcmp(argv[iarg], "--device") && iarg + 1 < argc)
            device = atoi(argv[++iarg]);
        else {
            fprintf(stderr,
                    "usage: %s [--weights PATH] [--latent-t N] [--device I]\n",
                    argv[0]);
            return 2;
        }
    }
    if (latent_t < 1) die("latent-t must be >= 1");

    n_lat = (size_t)H3_AUDIO_VAE_LATENT_CHANNELS * H3_AUDIO_VAE_STEREO *
            (size_t)latent_t;
    latent = calloc(n_lat, sizeof(*latent));
    if (!latent) die("OOM latent");
    /* Mild non-zero normalized latent for smoke (not silence). */
    for (i = 0; i < n_lat; i++)
        latent[i] = ((float)((i * 17) % 100) - 50.0f) * 0.01f;

    cuda = h3_cuda_create(device, error, sizeof(error));
    if (!cuda) die(error);

    if (!h3_audio_vae_cuda_decode(cuda, weights, latent, latent_t, on_progress,
                                  NULL, &wave, &stats, error, sizeof(error))) {
        fprintf(stderr, "decode failed: %s\n", error);
        return 1;
    }

    if (wave.channels != H3_AUDIO_VAE_STEREO ||
        wave.sample_rate != H3_AUDIO_VAE_SAMPLE_RATE ||
        wave.samples != latent_t * H3_AUDIO_VAE_HOP_LENGTH)
        die("unexpected waveform shape");

    for (i = 0; i < (size_t)wave.channels * (size_t)wave.samples; i++) {
        float v = wave.pcm[i];
        if (!isfinite(v)) finite = 0;
        if (fabs(v) > absmax) absmax = fabs(v);
        rms += (double)v * (double)v;
    }
    rms = sqrt(rms / ((double)wave.channels * wave.samples));

    printf("audio_vae_cuda PASS latent_t=%d samples=%d\n", latent_t,
           wave.samples);
    printf("  load_ms=%.1f decode_ms=%.1f total_ms=%.1f\n", stats.load_ms,
           stats.decode_ms, stats.total_ms);
    printf("  pcm absmax=%.6g rms=%.6g finite=%d\n", absmax, rms, finite);
    printf("  audio_ms=%.1f\n", stats.decode_ms);

    if (!finite) die("non-finite PCM");
    if (absmax > 1.0001) die("PCM outside clip range");

    h3_audio_vae_cuda_waveform_free(&wave);
    free(latent);
    h3_cuda_free(cuda);
    return 0;
}
