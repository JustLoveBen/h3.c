#ifndef H3_DIT_ACT_F16_H
#define H3_DIT_ACT_F16_H

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* F16 max finite is 65504. Normalized Hadamard ||H||_∞ = sqrt(gs), so
 * ConvRot can amplify a row's ∞-norm by that factor. Scale F32 activations
 * so both the F32→F16 cast and Hx fit; INT8 act_scale must be divided by
 * the same factor after F16 quant (see h3_cuda_quantize_act_per_row_f16). */
#if defined(__CUDACC__)
__host__ __device__
#endif
static inline float h3_dit_act_f16_pre_scale(float absmax, int gs) {
    float amp;
    float room;
    if (!(absmax > 0.0f) || gs < 4)
        return 1.0f;
    /* Inf/NaN: NaN fails `> 0`; Inf fails `<= FLT_MAX`. Avoid isfinite so
     * this stays __device__-safe without extra CUDA headers. */
    if (!(absmax <= 3.402823466e+38f))
        return 1.0f;
    amp = sqrtf((float)gs);
    room = 65504.0f * 0.96f;
    if (absmax * amp <= room)
        return 1.0f;
    return room / (absmax * amp);
}

#ifdef __cplusplus
}
#endif

#endif
