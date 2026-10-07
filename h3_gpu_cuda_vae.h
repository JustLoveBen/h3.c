#ifndef H3_GPU_CUDA_VAE_H
#define H3_GPU_CUDA_VAE_H

#include "h3_gpu_cuda.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Fused bias + residual-scale: x += (branch + bias) * scale.
 * d_bias may be NULL. */
int h3_cuda_vae_bias_residual_scale_f32_dev(
    h3_cuda *cuda, float *d_x, const float *d_branch, const float *d_bias,
    const float *d_scale, size_t rows, size_t width, char *error,
    size_t error_size);

/* Fused bias + SiLU-mul. d_fused is [rows, 2*width]; d_bias [2*width] or
 * NULL; d_out is [rows, width]. */
int h3_cuda_vae_bias_silu_mul_f32_dev(h3_cuda *cuda, const float *d_fused,
                                      const float *d_bias, size_t rows,
                                      size_t width, float *d_out, char *error,
                                      size_t error_size);

/* M4 act-F16: RMS F32 residual → F16 (F32 acc). */
int h3_cuda_vae_rms_norm_f32_to_f16_dev(h3_cuda *cuda, const float *d_x,
                                        const float *d_w, size_t rows,
                                        size_t width, float eps,
                                        void *d_out_half, char *error,
                                        size_t error_size);

/* F16 y[r,c] += F32 bias[c]. */
int h3_cuda_vae_bias_add_f16_dev(h3_cuda *cuda, void *d_y_half,
                                 const float *d_bias, size_t rows, size_t width,
                                 char *error, size_t error_size);

/* F16 fused [rows, 2*width] + optional F32 bias → F16 [rows, width]. */
int h3_cuda_vae_bias_silu_mul_f16_dev(h3_cuda *cuda, const void *d_fused_half,
                                      const float *d_bias, size_t rows,
                                      size_t width, void *d_out_half,
                                      char *error, size_t error_size);

/* F32 residual += (F16 branch + optional F32 bias) * F32 scale. */
int h3_cuda_vae_bias_residual_scale_f16_dev(
    h3_cuda *cuda, float *d_x, const void *d_branch_half, const float *d_bias,
    const float *d_scale, size_t rows, size_t width, char *error,
    size_t error_size);

/* F16 QKV + tile-local RoPE + batched cuDNN; O stays F16 packed
 * [n_batch*seq, heads*dim]. Geometry 256/64 unchanged. */
int h3_cuda_vae_qkv_rope_sdpa_f16io_dev_layout_nb(
    h3_cuda *cuda, const void *d_qkv_half, const float *d_q_norm,
    const float *d_k_norm, const float *d_rope_cos, const float *d_rope_sin,
    size_t seq, size_t n_batch, size_t heads, size_t head_dim, size_t rope_half,
    float epsilon, float scale, int grouped, int split_half, void *d_attn_half,
    char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
