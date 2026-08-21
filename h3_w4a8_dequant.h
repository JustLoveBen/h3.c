#ifndef H3_W4A8_DEQUANT_H
#define H3_W4A8_DEQUANT_H

#include "h3_w4a8.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* P4: asym_w4a8_int8 weight unpack + dequant (CPU reference) + act ConvRot.
 * Matches comfy-kitchen eager AsymW4A8Int8Layout decode:
 *   packed I8[N, K/2] → nibble indices → codebook[idx] * s_rel → INT8
 *   INT8 * s_channel → FP32 in ConvRot-rotated basis (W_rot)
 *   optional Hadamard undo (group=convrot_groupsize) → W = W_rot @ H
 *
 * Equivalence (H normalized regular Hadamard, involutory + symmetric):
 *   Y = X @ W.T  ==  (X @ H) @ W_rot.T
 * Activation ConvRot applies the same per-group Hadamard along K. */

#define H3_W4A8_DEFAULT_GROUP_SIZE 16
#define H3_W4A8_DEFAULT_CONVROT_GROUPSIZE 256

/* IEEE float8_e4m3fn → float32 (NaN when exp=0xF and mant=0x7). */
float h3_f8e4m3fn_to_f32(uint8_t bits);

/* Expand packed I8[N, K/2] to unsigned 4-bit indices in I8[N, K].
 * Layout: low nibble → even K, high nibble → odd K (comfy-kitchen). */
int h3_w4a8_unpack_nibbles(const int8_t *packed, size_t n, size_t k_half,
                           int8_t *indices_out, char *error, size_t error_size);

/* Decode packed W4A8 → INT8[N, K] in rotated weight space.
 * codebook: F32[16]; s_rel: F8_E4M3[N, K/group_size]; group_size typically 16. */
int h3_w4a8_dequant_to_int8(const int8_t *packed, size_t n, size_t k_half,
                            const float *codebook16, const uint8_t *s_rel_f8,
                            int group_size, int8_t *int8_out, char *error,
                            size_t error_size);

/* INT8[N,K] * s_channel[N] → FP32[N,K] (still rotated if apply_convrot=0). */
int h3_w4a8_int8_scale_to_f32(const int8_t *int8_w, size_t n, size_t k,
                              const float *s_channel, float *f32_out,
                              char *error, size_t error_size);

/* Build normalized regular Hadamard (order must be 4^p). Caller frees.
 * Used by ConvRot and CUDA act-prep (same matrix as host path). */
float *h3_w4a8_hadamard_normalized(int size, char *error, size_t error_size);

/* Sylvester / FWHT Hadamard (power-of-two order). Comfy VAE INT8 ConvRot. */
float *h3_w4a8_hadamard_sylvester_normalized(int size, char *error,
                                             size_t error_size);

/* Activation ConvRot via FWHT butterfly (matches comfy int8_linear convrot). */
int h3_w4a8_fwht_act_f32(const float *act_in, size_t m, size_t k,
                         int convrot_groupsize, float *act_out, char *error,
                         size_t error_size);

/* Out-of-place ConvRot along K: mat' = mat @ H per group.
 * Same kernel for weight undo (W_rot → W) and activation rotate (X → X_rot).
 * Hadamard is involutory when normalized → applying twice restores. */
int h3_w4a8_convrot_weight_f32(const float *weight_in, size_t n, size_t k,
                               int convrot_groupsize, float *weight_out,
                               char *error, size_t error_size);

/* Activation ConvRot: X_rot[M,K] = X @ H per group (alias of weight path). */
int h3_w4a8_convrot_act_f32(const float *act_in, size_t m, size_t k,
                            int convrot_groupsize, float *act_out, char *error,
                            size_t error_size);

/* Full path: packed + scales → FP32[N,K].
 * apply_convrot=1 undoes weight ConvRot (usable with unrotated activations). */
int h3_w4a8_dequant_to_f32(const int8_t *packed, size_t n, size_t k_half,
                           const float *codebook16, const uint8_t *s_rel_f8,
                           const float *s_channel, int group_size,
                           int convrot_groupsize, int apply_convrot,
                           float *f32_out, char *error, size_t error_size);

/* Load one linear from store and dequant to host FP32 (caller frees *out). */
int h3_w4a8_load_linear_f32(const h3_w4a8_store *store, const h3_w4a8_quad *quad,
                            int group_size, int convrot_groupsize,
                            int apply_convrot, float **out_weight, size_t *out_n,
                            size_t *out_k, char *error, size_t error_size);

/* y[M,N] = x[M,K] @ W[N,K]^T  (row-major, FP32). */
void h3_w4a8_gemm_f32(const float *x, const float *weight, size_t m, size_t n,
                      size_t k, float *y);

/* Compare two FP32 buffers; returns 0 if either pointer is NULL. */
int h3_w4a8_compare_f32(const float *a, const float *b, size_t count,
                        float *max_abs_diff_out, double *rmse_out);

/* --- P4 step-3: dynamic INT8 activation + INT8×INT8 → INT32 → FP32 --- */

typedef enum {
    H3_W4A8_ACT_QUANT_PER_TENSOR = 0, /* one scale for whole X[M,K] */
    H3_W4A8_ACT_QUANT_PER_ROW = 1     /* one scale per row (token); default */
} h3_w4a8_act_quant_mode;

/* Decode packed W4A8 → host INT8[N,K] in ConvRot-rotated weight space +
 * duplicate s_channel[N]. Caller frees *out_int8 / *out_s_channel. */
int h3_w4a8_load_linear_int8(const h3_w4a8_store *store,
                             const h3_w4a8_quad *quad, int group_size,
                             int8_t **out_int8, float **out_s_channel,
                             size_t *out_n, size_t *out_k, char *error,
                             size_t error_size);

/* Dynamic symmetric INT8 quantize of activations (after ConvRot).
 * scales_out length: 1 (per-tensor) or M (per-row).
 * x_q = clamp(round(x / scale), -127, 127); scale = absmax / 127
 * (zero row/tensor → scale=1, zeros). Aligns with comfy-kitchen / M5-style
 * dynamic act quant assumptions for A8. */
int h3_w4a8_quantize_act_int8(const float *act_f32, size_t m, size_t k,
                              h3_w4a8_act_quant_mode mode, int8_t *act_i8,
                              float *scales_out, char *error,
                              size_t error_size);

/* y[M,N] = act_scale[row] * s_channel[col] * sum_k x_i8[m,k] * w_i8[n,k]
 * INT32 accumulate then FP32 dequant. act_scales length 1 or M
 * (per_tensor_scale=1 → broadcast scales[0]). */
void h3_w4a8_gemm_i8_dequant_f32(const int8_t *x_i8, const int8_t *w_i8,
                                 const float *act_scales, int per_tensor_scale,
                                 const float *s_channel, size_t m, size_t n,
                                 size_t k, float *y_f32);

#ifdef __cplusplus
}
#endif

#endif
