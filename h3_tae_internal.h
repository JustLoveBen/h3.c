#ifndef H3_TAE_INTERNAL_H
#define H3_TAE_INTERNAL_H

#include "h3_safetensors.h"
#include "h3_vae_int8.h"

#ifdef __cplusplus
extern "C" {
#endif

/* madebyollin TAEHV taeh3: patch=2, C=24, decoder n_f=[256,128,64,64].
 * Spatial 2^4=16 (3× nearest-2 + pixelshuffle-2). Temporal tgrow (1,2,2)=4×. */

typedef struct {
    float *w; /* [oc, ic, kh, kw] */
    float *b; /* [oc] or NULL */
    int oc, ic, kh, kw;
} tae_conv;

typedef struct {
    tae_conv conv0, conv2, conv4; /* n_in*2→n, n→n, n→n */
} tae_mem;

typedef struct {
    h3_st_header header;
    tae_conv c1;          /* 24→256 */
    tae_mem mem256[3];    /* 3,4,5 */
    tae_conv tgrow1;      /* 256→256 1x1 */
    tae_conv c8;          /* 256→128 */
    tae_mem mem128[3];    /* 9,10,11 */
    tae_conv tgrow2;      /* 128→256 1x1 */
    tae_conv c14;         /* 128→64 */
    tae_mem mem64[3];     /* 15,16,17 */
    tae_conv tgrow3;      /* 64→128 1x1 */
    tae_conv c20;         /* 64→64 */
    tae_conv c22;         /* 64→12 */
    int ok;
    int latent_ch;        /* must be 24 */
    int spatial_ratio;    /* must be 16 */
    int t_upscale;        /* 4 */
    void *cuda;           /* h3_cuda* after bind; NULL = host */
} h3_tae_ctx;

#ifdef H3_TAE_HAVE_CUDA
int h3_tae_cuda_decode_frames(h3_tae_ctx *tc, const float *latent_chw, int t,
                              int h, int w, int want_frames,
                              h3_vae_int8_frames *frames, char *err,
                              size_t err_size);
#endif

#ifdef __cplusplus
}
#endif

#endif
