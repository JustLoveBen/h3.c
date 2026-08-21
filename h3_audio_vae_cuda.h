#ifndef H3_AUDIO_VAE_CUDA_H
#define H3_AUDIO_VAE_CUDA_H

#include "h3_audio_vae_layout.h"
#include "h3_gpu_cuda.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int channels;
    int samples;
    int sample_rate;
    /* Channel-major F32: [channels, samples], clipped to [-1,1]. */
    float *pcm;
} h3_audio_vae_cuda_waveform;

typedef struct {
    int channels;
    int stereo;
    int length;
    /* Channel-major F32: [32,2,length], normalized posterior means. */
    float *values;
} h3_audio_vae_cuda_latent;

typedef struct {
    double load_ms;
    double decode_ms;
    double encode_ms;
    double total_ms;
    int latent_length;
    int pcm_samples;
} h3_audio_vae_cuda_stats;

typedef void (*h3_audio_vae_cuda_progress)(int completed_stages,
                                           int total_stages, void *opaque);

/* Decode normalized H3 audio latents [32,2,T] → stereo PCM @ 32 kHz.
 * weights_path: Comfy FP32 .safetensors file OR a directory containing one.
 * Does not touch Metal h3_audio_vae. */
int h3_audio_vae_cuda_decode(h3_cuda *cuda, const char *weights_path,
                             const float *normalized_latent, int latent_length,
                             h3_audio_vae_cuda_progress progress,
                             void *progress_opaque,
                             h3_audio_vae_cuda_waveform *output,
                             h3_audio_vae_cuda_stats *stats, char *error,
                             size_t error_size);

/* Encode channel-major 32 kHz stereo PCM [2,samples] → [32,2,T] latents. */
int h3_audio_vae_cuda_encode(h3_cuda *cuda, const char *weights_path,
                             const float *pcm, int samples,
                             h3_audio_vae_cuda_progress progress,
                             void *progress_opaque,
                             h3_audio_vae_cuda_latent *output,
                             h3_audio_vae_cuda_stats *stats, char *error,
                             size_t error_size);

void h3_audio_vae_cuda_waveform_free(h3_audio_vae_cuda_waveform *waveform);
void h3_audio_vae_cuda_latent_free(h3_audio_vae_cuda_latent *latent);

#ifdef __cplusplus
}
#endif

#endif
