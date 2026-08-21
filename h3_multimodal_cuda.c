#include "h3_multimodal_cuda.h"

#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Dumped from released Qwen tokenizer (same vocab as tests/test_tokenizer.c:
 * "A red fox walking through snow" → 32,2518,38835,11435,1526,11794).
 * Strings are Metal h3_multimodal.c: "<Picture %zu>: " with pad_empty=0. */
static const uint32_t k_picture_1_prefix[] = {
    21604u, 3826u, 220u, 16u, 26818u, 220u
};
static const uint32_t k_picture_2_prefix[] = {
    21604u, 3826u, 220u, 17u, 26818u, 220u
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static const uint32_t *picture_prefix(size_t ordinal, size_t *n) {
    if (ordinal == 1) {
        *n = sizeof(k_picture_1_prefix) / sizeof(k_picture_1_prefix[0]);
        return k_picture_1_prefix;
    }
    if (ordinal == 2) {
        *n = sizeof(k_picture_2_prefix) / sizeof(k_picture_2_prefix[0]);
        return k_picture_2_prefix;
    }
    *n = 0;
    return NULL;
}

static int patch_grid_ok(int grid_h, int grid_w, size_t vision_tokens,
                         char *error, size_t error_size) {
    size_t merged_h, merged_w;
    if (grid_h < 2 || grid_w < 2 || (grid_h & 1) || (grid_w & 1)) {
        fail(error, error_size, "invalid vision patch grid");
        return 0;
    }
    if (!vision_tokens) {
        fail(error, error_size, "invalid vision token count");
        return 0;
    }
    merged_h = (size_t)grid_h / 2u;
    merged_w = (size_t)grid_w / 2u;
    if (merged_w && merged_h > SIZE_MAX / merged_w) {
        fail(error, error_size, "vision merged grid overflow");
        return 0;
    }
    if (merged_h * merged_w != vision_tokens) {
        fail(error, error_size,
             "vision pad count must equal (patch grid/2)*(patch grid/2)");
        return 0;
    }
    return 1;
}

int h3_multimodal_cuda_build_fl2va_ids(
    size_t picture_count,
    const uint32_t *prompt_ids, size_t prompt_n,
    const size_t *vision_tokens,
    const int *grid_h, const int *grid_w,
    uint32_t **ids_out, size_t *n_out,
    h3_cuda_vision_span *spans_out,
    char *error, size_t error_size) {
    size_t image, total, cursor, pads, prefix_n;
    const uint32_t *prefix;
    uint32_t *ids;
    size_t p;

    if (ids_out) *ids_out = NULL;
    if (n_out) *n_out = 0;
    if (error && error_size) error[0] = '\0';
    if (!picture_count || picture_count > 2 || !prompt_ids || !prompt_n ||
        !vision_tokens || !grid_h || !grid_w || !ids_out || !n_out ||
        !spans_out) {
        fail(error, error_size, "invalid FL2VA id builder arguments");
        return 0;
    }
    total = prompt_n;
    for (image = 0; image < picture_count; image++) {
        if (!patch_grid_ok(grid_h[image], grid_w[image], vision_tokens[image],
                           error, error_size))
            return 0;
        prefix = picture_prefix(image + 1, &prefix_n);
        if (!prefix) {
            fail(error, error_size, "unsupported <Picture n> ordinal");
            return 0;
        }
        pads = vision_tokens[image];
        if (total > SIZE_MAX - prefix_n ||
            total + prefix_n > SIZE_MAX - (pads + 2)) {
            fail(error, error_size, "FL2VA id sequence overflow");
            return 0;
        }
        total += prefix_n + pads + 2;
    }
    ids = (uint32_t *)malloc(total * sizeof(uint32_t));
    if (!ids) {
        fail(error, error_size, "out of memory constructing FL2VA ids");
        return 0;
    }
    cursor = 0;
    for (image = 0; image < picture_count; image++) {
        prefix = picture_prefix(image + 1, &prefix_n);
        pads = vision_tokens[image];
        memcpy(ids + cursor, prefix, prefix_n * sizeof(uint32_t));
        cursor += prefix_n;
        ids[cursor] = H3_CUDA_VISION_START;
        spans_out[image].start = cursor;
        spans_out[image].tokens = pads + 2;
        spans_out[image].grid_h = grid_h[image];
        spans_out[image].grid_w = grid_w[image];
        cursor++;
        for (p = 0; p < pads; p++)
            ids[cursor++] = H3_CUDA_IMAGE_PAD;
        ids[cursor++] = H3_CUDA_VISION_END;
    }
    memcpy(ids + cursor, prompt_ids, prompt_n * sizeof(uint32_t));
    cursor += prompt_n;
    if (cursor != total) {
        free(ids);
        fail(error, error_size, "FL2VA id length mismatch");
        return 0;
    }
    *ids_out = ids;
    *n_out = total;
    return 1;
}

static const uint32_t k_video_1_prefix[] = {
    27u, 10724u, 220u, 16u, 26818u, 220u
};
static const uint32_t k_video_2_prefix[] = {
    27u, 10724u, 220u, 17u, 26818u, 220u
};

static const uint32_t *video_prefix(size_t ordinal, size_t *n) {
    if (ordinal == 1) {
        *n = sizeof(k_video_1_prefix) / sizeof(k_video_1_prefix[0]);
        return k_video_1_prefix;
    }
    if (ordinal == 2) {
        *n = sizeof(k_video_2_prefix) / sizeof(k_video_2_prefix[0]);
        return k_video_2_prefix;
    }
    *n = 0;
    return NULL;
}

/* Qwen digit ids: '0'→15 … '9'→24. Match Metal snprintf("<%.1f seconds>", ts). */
static int append_timestamp_ids(uint32_t *ids, size_t *cursor, double ts,
                                char *error, size_t error_size) {
    char buf[32];
    int whole = 0, frac = 0, n;
    if (!isfinite(ts) || ts < 0.0) {
        fail(error, error_size, "invalid video timestamp");
        return 0;
    }
    n = snprintf(buf, sizeof(buf), "%.1f", ts);
    if (n < 3 || (size_t)n >= sizeof(buf) ||
        sscanf(buf, "%d.%d", &whole, &frac) != 2 || whole < 0 || whole > 9 ||
        frac < 0 || frac > 9) {
        fail(error, error_size, "timestamp dump range is <0.0..9.9 seconds>");
        return 0;
    }
    ids[(*cursor)++] = 27u;
    ids[(*cursor)++] = 15u + (uint32_t)whole;
    ids[(*cursor)++] = 13u;
    ids[(*cursor)++] = 15u + (uint32_t)frac;
    ids[(*cursor)++] = 6486u;
    ids[(*cursor)++] = 29u;
    return 1;
}

int h3_multimodal_cuda_build_ref_video_ids(
    size_t video_ordinal,
    size_t block_count,
    const double *timestamps,
    const uint32_t *prompt_ids, size_t prompt_n,
    const size_t *vision_tokens,
    const int *grid_h, const int *grid_w,
    uint32_t **ids_out, size_t *n_out,
    h3_cuda_vision_span *spans_out,
    char *error, size_t error_size) {
    size_t block, total, cursor, pads, prefix_n, p;
    const uint32_t *prefix;
    uint32_t *ids;

    if (ids_out) *ids_out = NULL;
    if (n_out) *n_out = 0;
    if (error && error_size) error[0] = '\0';
    if (!video_ordinal || !block_count || !timestamps || !prompt_ids ||
        !prompt_n || !vision_tokens || !grid_h || !grid_w || !ids_out ||
        !n_out || !spans_out) {
        fail(error, error_size, "invalid Ref2VA video id builder arguments");
        return 0;
    }
    prefix = video_prefix(video_ordinal, &prefix_n);
    if (!prefix) {
        fail(error, error_size, "unsupported <Video n> ordinal");
        return 0;
    }
    total = prompt_n + prefix_n;
    for (block = 0; block < block_count; block++) {
        if (!patch_grid_ok(grid_h[block], grid_w[block], vision_tokens[block],
                           error, error_size))
            return 0;
        pads = vision_tokens[block];
        /* 6 timestamp ids + START + pads + END */
        if (total > SIZE_MAX - (6 + pads + 2)) {
            fail(error, error_size, "Ref2VA video id sequence overflow");
            return 0;
        }
        total += 6 + pads + 2;
    }
    ids = (uint32_t *)malloc(total * sizeof(uint32_t));
    if (!ids) {
        fail(error, error_size, "out of memory constructing Ref2VA video ids");
        return 0;
    }
    cursor = 0;
    memcpy(ids + cursor, prefix, prefix_n * sizeof(uint32_t));
    cursor += prefix_n;
    for (block = 0; block < block_count; block++) {
        pads = vision_tokens[block];
        if (!append_timestamp_ids(ids, &cursor, timestamps[block], error,
                                  error_size)) {
            free(ids);
            return 0;
        }
        ids[cursor] = H3_CUDA_VISION_START;
        spans_out[block].start = cursor;
        spans_out[block].tokens = pads + 2;
        spans_out[block].grid_h = grid_h[block];
        spans_out[block].grid_w = grid_w[block];
        cursor++;
        for (p = 0; p < pads; p++)
            ids[cursor++] = H3_CUDA_VIDEO_PAD;
        ids[cursor++] = H3_CUDA_VISION_END;
    }
    memcpy(ids + cursor, prompt_ids, prompt_n * sizeof(uint32_t));
    cursor += prompt_n;
    if (cursor != total) {
        free(ids);
        fail(error, error_size, "Ref2VA video id length mismatch");
        return 0;
    }
    *ids_out = ids;
    *n_out = total;
    return 1;
}

int h3_multimodal_cuda_dit_text_tags(size_t text_seq,
                                     const h3_cuda_vision_span *spans,
                                     size_t span_count, uint8_t *tags,
                                     char *error, size_t error_size) {
    size_t i, image, start, n;
    if (error && error_size) error[0] = '\0';
    if (!tags || !text_seq || (span_count && !spans)) {
        fail(error, error_size, "invalid DiT text tag arguments");
        return 0;
    }
    memset(tags, 1, text_seq);
    for (image = 0; image < span_count; image++) {
        start = spans[image].start;
        n = spans[image].tokens;
        if (start > text_seq || n > text_seq - start) {
            fail(error, error_size, "DiT text tag span overflows text_seq");
            return 0;
        }
        for (i = 0; i < n; i++) tags[start + i] = 0;
    }
    return 1;
}
