#ifndef H3_TAE_CUDA_H
#define H3_TAE_CUDA_H

#include "h3_tae_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

int h3_tae_cuda_decode_frames(h3_tae_ctx *tc, const float *latent_chw, int t,
                              int h, int w, int want_frames,
                              h3_vae_int8_frames *frames, char *err,
                              size_t err_size);

#ifdef __cplusplus
}
#endif

#endif
