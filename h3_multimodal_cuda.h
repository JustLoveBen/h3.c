#ifndef H3_MULTIMODAL_CUDA_H
#define H3_MULTIMODAL_CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define H3_CUDA_VISION_START UINT32_C(151652)
#define H3_CUDA_VISION_END   UINT32_C(151653)
#define H3_CUDA_IMAGE_PAD    UINT32_C(151655)
#define H3_CUDA_VIDEO_PAD    UINT32_C(151656)

/* Vision span on the packed multimodal id sequence.
 * start includes VISION_START; tokens = START + pads + END.
 * Embed replace / deepstack use the inner pad rows only
 * (start+1 .. start+tokens-2), matching Metal append_vision.
 * grid_h/grid_w are the patch grid (H/16, W/16) from h3_vision_encode_f32 /
 * Metal h3_vision_output. Pad count is vision.tokens = (grid_h/2)*(grid_w/2),
 * not grid_h*grid_w. */
typedef struct {
    size_t start;
    size_t tokens;
    int grid_h, grid_w;
} h3_cuda_vision_span;

/* Build [Picture-prefix | vision pads | prompt_ids].
 * Picture n UTF-8 prefix ids are a static table dumped from the released
 * Qwen tokenizer (same vocab as tests/test_tokenizer.c).
 * picture_count: F2VA=1, FL2VA=2.
 * vision_tokens: pad count per image (Metal vision->tokens, merged).
 * grid_h/grid_w: patch grid H/16, W/16 (even, >=2). mRoPE uses grid/2.
 * Requires (grid_h/2)*(grid_w/2) == vision_tokens[i].
 * Caller frees *ids_out. spans_out must hold picture_count entries. */
int h3_multimodal_cuda_build_fl2va_ids(
    size_t picture_count,
    const uint32_t *prompt_ids, size_t prompt_n,
    const size_t *vision_tokens,
    const int *grid_h, const int *grid_w,
    uint32_t **ids_out, size_t *n_out,
    h3_cuda_vision_span *spans_out,
    char *error, size_t error_size);

/* Ref2VA video presentation: [<Video n>: ][ <t seconds> VISION_START pads
 * VIDEO_PAD VISION_END ]* × blocks | prompt. timestamps follow Metal
 * ((first+second)/4) then snprintf("%.1f") digit dump (0.0–9.9).
 * spans_out holds block_count entries. Caller frees *ids_out. */
int h3_multimodal_cuda_build_ref_video_ids(
    size_t video_ordinal,
    size_t block_count,
    const double *timestamps,
    const uint32_t *prompt_ids, size_t prompt_n,
    const size_t *vision_tokens,
    const int *grid_h, const int *grid_w,
    uint32_t **ids_out, size_t *n_out,
    h3_cuda_vision_span *spans_out,
    char *error, size_t error_size);

/* Metal DiT text tags: 1=language, 0=Qwen vision span including START/END.
 * CUDA span.start already includes VISION_START; tokens = START+pads+END.
 * CUDA INT8 official F2VA generate defaults these OFF (Picture+vision stay
 * TEXT AdaLN — condition embeddings, not video latent). Pass
 * --dit-text-tags to apply this helper. That is a debug exception, not a
 * Metal bit-match claim. */
int h3_multimodal_cuda_dit_text_tags(size_t text_seq,
                                     const h3_cuda_vision_span *spans,
                                     size_t span_count, uint8_t *tags,
                                     char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
