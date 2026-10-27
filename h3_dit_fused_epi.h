#ifndef H3_DIT_FUSED_EPI_H
#define H3_DIT_FUSED_EPI_H

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* R31: fuse DiT post-GEMM F32 kernels (gate/AdaLN/SwiGLU). Default ON.
 * Rollback: H3_DIT_FUSED_EPILOGUE=0. */
static inline int h3_dit_fused_epilogue_wanted(void) {
    const char *e = getenv("H3_DIT_FUSED_EPILOGUE");
    if (e && e[0] == '0' && e[1] == '\0')
        return 0;
    return 1;
}

/* E2: intended to collapse "dequant(epi) → FP32 round-trip → requant" pairs
 * in the DiT block. opt-in, default 0: H3_DIT_EPI_FUSE=1.
 * 2026-10-27: P2 skip-ConvRot arm is DISARMED in h3_dit_w4a8.c (invalid with
 * packed W_rot). Env bit still readable for tests/logging; no numerical change. */
static inline int h3_dit_epi_fuse_wanted(void) {
    const char *e = getenv("H3_DIT_EPI_FUSE");
    return e && e[0] == '1' && e[1] == '\0';
}

/* E2 redesign: ConvRot+requant after FC1 SwiGLU, then FC2 pre_q (keeps
 * Y=(X@H)@W_rot.T). Opt-in H3_DIT_EPI_FUSE_INK=1; default OFF. Does NOT
 * arm the disarmed skip-ConvRot H3_DIT_EPI_FUSE path. Partial: uses
 * sgemm ConvRot + per-row INT8 quant (f32 algebra); act_f16-prescale
 * match and single-launch fuse still TODO before waffle-green default. */
static inline int h3_dit_epi_fuse_ink_wanted(void) {
    const char *e = getenv("H3_DIT_EPI_FUSE_INK");
    return e && e[0] == '1' && e[1] == '\0';
}

/* gated[r] = residual[r] + branch[r] * gate_mod; then AdaLN(gated) → normed.
 * Same math as h3_dit_w4a8_gate_f32 then h3_dit_w4a8_adaln_f32. */
static inline void h3_dit_gate_adaln_f32(
    const float *residual, const float *branch, const float *norm_w,
    const float *modulation, const uint32_t *row_map, size_t rows,
    size_t width, size_t slots, size_t gate_slot, size_t shift_slot,
    size_t scale_slot, float epsilon, float *gated, float *normed) {
    size_t r, c;
    if (!residual || !branch || !norm_w || !modulation || !gated || !normed ||
        width == 0 || slots == 0)
        return;
    for (r = 0; r < rows; r++) {
        uint32_t mrow = row_map ? row_map[r] : (uint32_t)r;
        const float *mod_base = modulation + (size_t)mrow * slots * width;
        const float *gate = mod_base + gate_slot * width;
        const float *shift = mod_base + shift_slot * width;
        const float *scale = mod_base + scale_slot * width;
        const float *res = residual + r * width;
        const float *br = branch + r * width;
        float *gr = gated + r * width;
        float *nr = normed + r * width;
        float sum = 0.0f;
        float inv;
        for (c = 0; c < width; c++) {
            float v = res[c] + br[c] * gate[c];
            gr[c] = v;
            sum += v * v;
        }
        inv = 1.0f / sqrtf(sum / (float)width + epsilon);
        for (c = 0; c < width; c++) {
            float n = gr[c] * inv * norm_w[c];
            nr[c] = n * (1.0f + scale[c]) + shift[c];
        }
    }
}

/* y = SiLU(dequant(acc_gate)+lora_gate) * (dequant(acc_up)+lora_up).
 * acc/lora are [rows, 2*width]. lora NULL → 0 (already-scaled LoRA delta). */
static inline void h3_dit_dequant_swiglu_f32(const int32_t *acc,
                                             const float *act_scales,
                                             int per_tensor,
                                             const float *s_channel,
                                             size_t rows, size_t width,
                                             const float *lora, float *out) {
    size_t r, c, n;
    if (!acc || !act_scales || !s_channel || !out || width == 0)
        return;
    n = width * 2u;
    for (r = 0; r < rows; r++) {
        float a = per_tensor ? act_scales[0] : act_scales[r];
        const int32_t *ar = acc + r * n;
        const float *lr = lora ? (lora + r * n) : NULL;
        float *yr = out + r * width;
        for (c = 0; c < width; c++) {
            float gate = a * s_channel[c] * (float)ar[c];
            float up = a * s_channel[width + c] * (float)ar[width + c];
            if (lr) {
                gate += lr[c];
                up += lr[width + c];
            }
            yr[c] = (gate / (1.0f + expf(-gate))) * up;
        }
    }
}

/* out = residual + (dequant(acc)+lora) * gate_mod. acc/lora/out [rows, width].
 * lora NULL → 0; caller applies lora_scale (product turbo = 0.75) into lora. */
static inline void h3_dit_dequant_gate_f32(
    const int32_t *acc, const float *act_scales, int per_tensor,
    const float *s_channel, const float *residual, const float *modulation,
    const uint32_t *row_map, size_t rows, size_t width, size_t slots,
    size_t gate_slot, const float *lora, float *out) {
    size_t r, c;
    if (!acc || !act_scales || !s_channel || !residual || !modulation ||
        !out || width == 0 || slots == 0)
        return;
    for (r = 0; r < rows; r++) {
        uint32_t mrow = row_map ? row_map[r] : (uint32_t)r;
        const float *gate =
            modulation + (size_t)mrow * slots * width + gate_slot * width;
        float a = per_tensor ? act_scales[0] : act_scales[r];
        const int32_t *ar = acc + r * width;
        const float *res = residual + r * width;
        const float *lr = lora ? (lora + r * width) : NULL;
        float *yr = out + r * width;
        for (c = 0; c < width; c++) {
            float br = a * s_channel[c] * (float)ar[c];
            if (lr)
                br += lr[c];
            yr[c] = res[c] + br * gate[c];
        }
    }
}

#ifdef __cplusplus
}
#endif

#endif
