#ifndef H3_LATENT_UPSCALER_INTERNAL_H
#define H3_LATENT_UPSCALER_INTERNAL_H

#include "h3_gpu_cuda.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void h3_ups_bind_cuda(h3_cuda *cuda);

int h3_ups_conv3d_nchw(const float *in, float *out, const float *w,
                       const float *b, int n, int ic, int oc, int t, int h,
                       int w_s, int kd, int kh, int kw, int pad_t, int pad_h,
                       int pad_w, int groups, char *err, size_t err_size);
int h3_ups_groupnorm(float *x, const float *gamma, const float *beta, int n,
                     int c, int t, int h, int w, int groups, float eps,
                     char *err, size_t err_size);
int h3_ups_silu(float *x, size_t n, char *err, size_t err_size);
int h3_ups_add(float *y, const float *a, const float *b, size_t n, char *err,
               size_t err_size);
int h3_ups_adaln(float *x, const float *scale, const float *shift, int n,
                 int c, int t, int h, int w, char *err, size_t err_size);
int h3_ups_trilinear(const float *in, float *out, int n, int c, int t, int ih,
                     int iw, int oh, int ow, char *err, size_t err_size);
int h3_ups_scale_bias(float *x, const float *mean, const float *std, int inv,
                      int n, int c, int t, int h, int w, char *err,
                      size_t err_size);

#ifdef __cplusplus
}
#endif

#endif
