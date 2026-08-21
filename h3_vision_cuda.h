#ifndef H3_VISION_CUDA_H
#define H3_VISION_CUDA_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define H3_CUDA_VISION_WIDTH 5120u
#define H3_CUDA_VISION_DEEPSTACKS 3u

/* Host F32 Qwen3-VL vision tower (CUDA-portable; no Metal).
 * pixels_t3hw: [T,3,H,W] in [0,1], T=1 for image (T=2 reserved for H3).
 * H,W multiples of 32. merged_out / deepstack_out[3] are [tokens, 5120]
 * with tokens = (H/32)*(W/32). grid_h/grid_w are the patch grid (H/16, W/16),
 * matching Metal h3_vision_output. */
int h3_vision_encode_f32(
    const char *vision_weights,
    const float *pixels_t3hw,
    int frames, int height, int width,
    float *merged_out,
    float *deepstack_out[3],
    int *grid_h, int *grid_w, size_t *tokens,
    char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
