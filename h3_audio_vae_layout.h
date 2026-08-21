#ifndef H3_AUDIO_VAE_LAYOUT_H
#define H3_AUDIO_VAE_LAYOUT_H

#include <stdint.h>

/* Shared AudioVAE layout constants (Metal + CUDA). No GPU deps. */

enum {
    H3_AUDIO_VAE_LATENT_CHANNELS = 32,
    H3_AUDIO_VAE_LATENT_DIM = 2048,
    H3_AUDIO_VAE_DECODER_DIM = 1024,
    H3_AUDIO_VAE_STEREO = 2,
    H3_AUDIO_VAE_STAGES = 7,
    H3_AUDIO_VAE_RESBLOCKS = 3,
    H3_AUDIO_VAE_RESIDUAL_PAIRS = 3,
    H3_AUDIO_VAE_FILTER_SIZE = 12,
    H3_AUDIO_VAE_SAMPLE_RATE = 32000,
    H3_AUDIO_VAE_HOP_LENGTH = 800,
    H3_AUDIO_VAE_ENCODER_STAGES = 5,
    H3_AUDIO_VAE_ENCODER_RESIDUALS = 3,
    H3_AUDIO_VAE_ENCODER_HEADS = 8,
    H3_AUDIO_VAE_ENCODER_DIM = 64
};

static const uint32_t h3_audio_vae_encoder_strides[H3_AUDIO_VAE_ENCODER_STAGES] = {
    2, 4, 4, 5, 5
};
static const uint32_t
    h3_audio_vae_encoder_dilations[H3_AUDIO_VAE_ENCODER_RESIDUALS] = {1, 3, 9};

static const uint32_t h3_audio_vae_upsample_rates[H3_AUDIO_VAE_STAGES] = {
    5, 5, 2, 2, 2, 2, 2
};
static const uint32_t h3_audio_vae_upsample_kernels[H3_AUDIO_VAE_STAGES] = {
    9, 9, 4, 4, 4, 4, 4
};
static const uint32_t h3_audio_vae_residual_kernels[H3_AUDIO_VAE_RESBLOCKS] = {
    3, 7, 11
};
static const uint32_t h3_audio_vae_residual_dilations[H3_AUDIO_VAE_RESIDUAL_PAIRS] = {
    1, 3, 5
};

#endif
