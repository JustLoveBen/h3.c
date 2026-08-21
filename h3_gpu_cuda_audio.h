#ifndef H3_GPU_CUDA_AUDIO_H
#define H3_GPU_CUDA_AUDIO_H

#include "h3_gpu_cuda.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* CUDA AudioVAE 1D ops. Semantics align with Metal h3_gpu_* audio APIs:
 * activations [batch, length, channels]; Conv1d weights OIK; Transpose IOK. */

int h3_cuda_audio_weight_norm_f32(h3_cuda *cuda, float *d_out,
                                  const float *d_vector,
                                  const float *d_magnitude, uint32_t outer,
                                  uint32_t inner, char *error,
                                  size_t error_size);

int h3_cuda_audio_conv1d_f32(h3_cuda *cuda, float *d_out, const float *d_in,
                             const float *d_weight, const float *d_bias,
                             uint32_t batch, uint32_t length,
                             uint32_t in_ch, uint32_t out_ch, uint32_t kernel,
                             uint32_t padding, uint32_t dilation, char *error,
                             size_t error_size);

/* Strided Conv1d; out_length must match encoder_conv_length formula. */
int h3_cuda_audio_conv1d_stride_f32(h3_cuda *cuda, float *d_out,
                                    const float *d_in, const float *d_weight,
                                    const float *d_bias, uint32_t batch,
                                    uint32_t in_length, uint32_t out_length,
                                    uint32_t in_ch, uint32_t out_ch,
                                    uint32_t kernel, uint32_t stride,
                                    uint32_t padding, uint32_t dilation,
                                    char *error, size_t error_size);

int h3_cuda_audio_conv_transpose1d_f32(h3_cuda *cuda, float *d_out,
                                       const float *d_in, const float *d_weight,
                                       const float *d_bias, uint32_t batch,
                                       uint32_t length, uint32_t in_ch,
                                       uint32_t out_ch, uint32_t kernel,
                                       uint32_t stride, uint32_t padding,
                                       char *error, size_t error_size);

int h3_cuda_audio_alias_free_snake_f32(h3_cuda *cuda, float *d_out,
                                       const float *d_in, const float *d_alpha,
                                       const float *d_beta,
                                       const float *d_up_filter,
                                       const float *d_down_filter,
                                       uint32_t batch, uint32_t length,
                                       uint32_t channels, char *error,
                                       size_t error_size);

/* Encoder Snake1d: y = x + sin(a*x)^2 / (a+eps); alpha per-channel. */
int h3_cuda_audio_snake1d_f32(h3_cuda *cuda, float *d_out, const float *d_in,
                              const float *d_alpha, uint32_t batch,
                              uint32_t length, uint32_t channels, char *error,
                              size_t error_size);

int h3_cuda_audio_layer_norm_f32(h3_cuda *cuda, float *d_out, const float *d_in,
                                 const float *d_weight, const float *d_bias,
                                 uint32_t rows, uint32_t width, float epsilon,
                                 char *error, size_t error_size);

/* y[M,N] = x[M,K] @ W[N,K]^T (+ optional bias[N]). */
int h3_cuda_audio_linear_f32(h3_cuda *cuda, float *d_out, const float *d_in,
                             const float *d_weight, const float *d_bias,
                             uint32_t rows, uint32_t in_dim, uint32_t out_dim,
                             char *error, size_t error_size);

int h3_cuda_audio_qkv_split_f32(h3_cuda *cuda, float *d_q, float *d_k,
                                float *d_v, const float *d_qkv,
                                const float *d_q_bias, const float *d_k_bias,
                                const float *d_v_bias, uint32_t batch,
                                uint32_t length, uint32_t heads,
                                uint32_t head_dim, char *error,
                                size_t error_size);

int h3_cuda_audio_sdpa_causal_f32(h3_cuda *cuda, float *d_out, const float *d_q,
                                  const float *d_k, const float *d_v,
                                  uint32_t batch, uint32_t sequence,
                                  uint32_t heads, uint32_t head_dim,
                                  float scale, char *error, size_t error_size);

int h3_cuda_audio_attention_pool_f32(h3_cuda *cuda, float *d_out,
                                     const float *d_attended, uint32_t batch,
                                     uint32_t length, uint32_t heads,
                                     uint32_t head_dim, uint32_t output_dim,
                                     char *error, size_t error_size);

int h3_cuda_audio_geglu_f32(h3_cuda *cuda, float *d_out, const float *d_gate,
                            const float *d_linear, uint32_t elements,
                            char *error, size_t error_size);

int h3_cuda_audio_add_scaled_f32(h3_cuda *cuda, float *d_out,
                                 const float *d_left, const float *d_right,
                                 float left_scale, float right_scale,
                                 uint32_t elements, char *error,
                                 size_t error_size);

int h3_cuda_audio_clip_f32(h3_cuda *cuda, float *d_out, const float *d_in,
                           uint32_t elements, float lo, float hi, char *error,
                           size_t error_size);

int h3_cuda_audio_copy_f32(h3_cuda *cuda, float *d_dst, const float *d_src,
                           size_t elements, char *error, size_t error_size);

/* Force backend for Au7 A/B: "naive" | "gemm" | NULL/auto. */
void h3_cuda_audio_set_conv_backend(const char *backend);
const char *h3_cuda_audio_conv_backend(void);

#ifdef __cplusplus
}
#endif

#endif
