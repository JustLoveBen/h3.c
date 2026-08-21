/* tests/test_h3_multimodal_cuda.c
 * Stage D: vision + multimodal TE. No-image TE must match encode_layers_f32.
 * Does not claim a first-frame product path. */
#include "h3_gpu_cuda.h"
#include "h3_multimodal_cuda.h"
#include "h3_nvfp4.h"
#include "h3_nvfp4_cuda.h"
#include "h3_vision_cuda.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail;

static void fail_msg(const char *msg) {
    fprintf(stderr, "FAIL tests/test_h3_multimodal_cuda.c: %s\n", msg);
    g_fail = 1;
}

static int all_finite(const float *a, size_t n) {
    size_t i;
    for (i = 0; i < n; i++) {
        if (!isfinite(a[i])) return 0;
    }
    return 1;
}

static double cosine(const float *a, const float *b, size_t n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    size_t i;
    for (i = 0; i < n; i++) {
        dot += (double)a[i] * (double)b[i];
        na += (double)a[i] * (double)a[i];
        nb += (double)b[i] * (double)b[i];
    }
    if (na <= 0.0 || nb <= 0.0) return 0.0;
    return dot / (sqrt(na) * sqrt(nb));
}

static size_t count_image_pads(const uint32_t *ids, size_t n) {
    size_t i, pads = 0;
    for (i = 0; i < n; i++)
        if (ids[i] == H3_CUDA_IMAGE_PAD) pads++;
    return pads;
}

static void test_picture_ids(void) {
    const uint32_t prompt[3] = {32u, 2518u, 38835u};
    /* 4 pads come from merged tokens, not patch 2x2. Patch 4x4 → merged 2x2. */
    const size_t vision_tokens = 4;
    const int grid_h = 4, grid_w = 4;
    uint32_t *ids = NULL;
    size_t n = 0;
    h3_cuda_vision_span span;
    char err[256];
    size_t i, prefix_n;
    const uint32_t want_prefix[] = {21604u, 3826u, 220u, 16u, 26818u, 220u};

    memset(&span, 0, sizeof(span));
    if (!h3_multimodal_cuda_build_fl2va_ids(1, prompt, 3, &vision_tokens,
                                            &grid_h, &grid_w, &ids, &n, &span,
                                            err, sizeof(err))) {
        fail_msg(err[0] ? err : "build_fl2va_ids returned 0");
        return;
    }
    prefix_n = sizeof(want_prefix) / sizeof(want_prefix[0]);
    if (n != prefix_n + 2 + 4 + 3) {
        char buf[128];
        snprintf(buf, sizeof(buf), "id length %zu want %zu", n,
                 prefix_n + 9);
        fail_msg(buf);
        free(ids);
        return;
    }
    for (i = 0; i < prefix_n; i++) {
        if (ids[i] != want_prefix[i]) {
            fail_msg("sequence does not start with <Picture 1>: prefix");
            free(ids);
            return;
        }
    }
    if (ids[prefix_n] != H3_CUDA_VISION_START ||
        ids[prefix_n + 5] != H3_CUDA_VISION_END) {
        fail_msg("VISION_START/END missing");
        free(ids);
        return;
    }
    for (i = 0; i < 4; i++) {
        if (ids[prefix_n + 1 + i] != H3_CUDA_IMAGE_PAD) {
            fail_msg("expected four IMAGE_PAD tokens");
            free(ids);
            return;
        }
    }
    if (ids[n - 3] != prompt[0] || ids[n - 2] != prompt[1] ||
        ids[n - 1] != prompt[2]) {
        fail_msg("prompt ids not appended");
        free(ids);
        return;
    }
    if (span.start != prefix_n || span.tokens != 6 || span.grid_h != 4 ||
        span.grid_w != 4 || span.tokens - 2 != vision_tokens) {
        fail_msg("vision span geometry mismatch");
        free(ids);
        return;
    }
    free(ids);
    puts("ok: FL2VA ids Picture prefix + 151652/151655x4/151653 + prompt");
}

static void test_dit_text_tags(void) {
    const uint32_t prompt[3] = {32u, 2518u, 38835u};
    const size_t vision_tokens = 4;
    const int grid_h = 4, grid_w = 4;
    uint32_t *ids = NULL;
    size_t n = 0, i, video_n = 0;
    h3_cuda_vision_span span;
    uint8_t *tags = NULL;
    char err[256];

    memset(&span, 0, sizeof(span));
    if (!h3_multimodal_cuda_build_fl2va_ids(1, prompt, 3, &vision_tokens,
                                            &grid_h, &grid_w, &ids, &n, &span,
                                            err, sizeof(err))) {
        fail_msg(err[0] ? err : "build_fl2va_ids for tags returned 0");
        return;
    }
    tags = (uint8_t *)malloc(n);
    if (!tags) {
        fail_msg("OOM dit text tags");
        free(ids);
        return;
    }
    if (!h3_multimodal_cuda_dit_text_tags(n, &span, 1, tags, err, sizeof(err))) {
        fail_msg(err[0] ? err : "dit_text_tags returned 0");
        free(tags);
        free(ids);
        return;
    }
    for (i = 0; i < n; i++) {
        if (tags[i] == 0) video_n++;
        else if (tags[i] != 1) {
            fail_msg("dit text tags must be 0 or 1");
            free(tags);
            free(ids);
            return;
        }
    }
    if (video_n != span.tokens) {
        fail_msg("vision span rows must be tag=0 (VIDEO)");
        free(tags);
        free(ids);
        return;
    }
    if (span.start == 0 || tags[span.start - 1] != 1 ||
        tags[span.start] != 0 ||
        tags[span.start + span.tokens - 1] != 0 ||
        (span.start + span.tokens < n && tags[span.start + span.tokens] != 1)) {
        fail_msg("tag=0 must cover START+pads+END only");
        free(tags);
        free(ids);
        return;
    }
    free(tags);
    free(ids);
    puts("ok: DiT text tags tag=0 on vision span, tag=1 elsewhere");
}

/* Pad count follows vision.tokens; grid is the patch grid (Metal).
 * 32x32 → tokens=1, patch 2x2. 864x480 → tokens=405, patch 30x54.
 * Must not emit grid_h*grid_w pads (1620 for 480p). */
static void test_span_pads_follow_vision_tokens(void) {
    const uint32_t prompt[3] = {32u, 2518u, 38835u};
    char err[256];

    {
        const size_t vision_tokens = 1;
        const int grid_h = 2, grid_w = 2;
        uint32_t *ids = NULL;
        size_t n = 0;
        h3_cuda_vision_span span;
        memset(&span, 0, sizeof(span));
        if (!h3_multimodal_cuda_build_fl2va_ids(1, prompt, 3, &vision_tokens,
                                                &grid_h, &grid_w, &ids, &n,
                                                &span, err, sizeof(err))) {
            fail_msg(err[0] ? err : "32x32 builder returned 0");
            return;
        }
        if (span.tokens - 2 != vision_tokens ||
            count_image_pads(ids, n) != vision_tokens ||
            span.grid_h != 2 || span.grid_w != 2) {
            fail_msg("32x32 span.tokens-2 must equal vision.tokens=1");
            free(ids);
            return;
        }
        free(ids);
    }
    {
        const size_t vision_tokens = 405;
        const int grid_h = 30, grid_w = 54;
        uint32_t *ids = NULL;
        size_t n = 0;
        h3_cuda_vision_span span;
        memset(&span, 0, sizeof(span));
        if (!h3_multimodal_cuda_build_fl2va_ids(1, prompt, 3, &vision_tokens,
                                                &grid_h, &grid_w, &ids, &n,
                                                &span, err, sizeof(err))) {
            fail_msg(err[0] ? err : "864x480 builder returned 0");
            return;
        }
        if (span.tokens - 2 != 405 || count_image_pads(ids, n) != 405 ||
            span.grid_h != 30 || span.grid_w != 54) {
            fail_msg("864x480 must emit 405 pads, not grid_h*grid_w=1620");
            free(ids);
            return;
        }
        /* prefix(6) + START + 405 pads + END + prompt(3) */
        if (n != 6 + 2 + 405 + 3) {
            fail_msg("864x480 id length used patch-grid pads");
            free(ids);
            return;
        }
        free(ids);
    }
    {
        const size_t vision_tokens = 1;
        const int grid_h = 1, grid_w = 1;
        uint32_t *ids = NULL;
        size_t n = 0;
        h3_cuda_vision_span span;
        memset(&span, 0, sizeof(span));
        if (h3_multimodal_cuda_build_fl2va_ids(1, prompt, 3, &vision_tokens,
                                               &grid_h, &grid_w, &ids, &n,
                                               &span, err, sizeof(err))) {
            fail_msg("builder must reject merged 1x1 passed as patch grid");
            free(ids);
            return;
        }
        free(ids);
    }
    puts("ok: span pads follow vision.tokens (32x32=1, 864x480=405, not grid product)");
}

static int test_no_image_te(const char *te_path) {
    char err[512];
    h3_cuda *cuda = NULL;
    h3_nvfp4_store store;
    h3_nvfp4_encode_stats st0, st1;
    float *a = NULL, *b = NULL;
    const size_t tokens = 8;
    const unsigned n_layers = 1;
    int32_t ids[8];
    size_t i;
    int ok = 0;

    memset(&store, 0, sizeof(store));
    for (i = 0; i < tokens; i++)
        ids[i] = (int32_t)(1 + i);
    cuda = h3_cuda_create(0, err, sizeof(err));
    if (!cuda) {
        fail_msg(err);
        return 0;
    }
    if (!h3_nvfp4_cuda_enable(cuda, err, sizeof(err))) {
        fail_msg(err);
        goto done;
    }
    if (!h3_nvfp4_store_open(te_path, &store, err, sizeof(err))) {
        fail_msg(err);
        goto done;
    }
    a = h3_nvfp4_encode_layers_f32(&store, ids, tokens, n_layers, &st0, err,
                                   sizeof(err));
    b = h3_nvfp4_encode_layers_multimodal_f32(&store, ids, tokens, NULL, 0,
                                              NULL, NULL, n_layers, &st1, err,
                                              sizeof(err));
    if (!a || !b) {
        fail_msg(err[0] ? err : "TE encode returned NULL");
        goto done;
    }
    if (memcmp(a, b, tokens * (size_t)H3_NVFP4_TEXT_DIM * sizeof(float)) != 0) {
        fail_msg("no-image multimodal TE != encode_layers_f32");
        goto done;
    }
    if (!all_finite(a, tokens * (size_t)H3_NVFP4_TEXT_DIM)) {
        fail_msg("no-image TE non-finite");
        goto done;
    }
    ok = 1;
    puts("ok: no-image multimodal TE memcmp-identical to encode_layers_f32");
done:
    free(a);
    free(b);
    h3_nvfp4_store_free(&store);
    h3_nvfp4_cuda_disable();
    h3_cuda_free(cuda);
    return ok;
}

static void fill_pixels(float *px, int frames, int height, int width) {
    int t, c, y, x;
    for (t = 0; t < frames; t++)
        for (c = 0; c < 3; c++)
            for (y = 0; y < height; y++)
                for (x = 0; x < width; x++) {
                    size_t idx =
                        ((((size_t)t * 3u + (size_t)c) * (size_t)height +
                          (size_t)y) *
                         (size_t)width) +
                        (size_t)x;
                    px[idx] = (float)((x * 13 + y * 7 + t * 3 + c * 17) % 251) /
                              250.0f;
                }
}

static int test_vision_encode(const char *te_path, const char *ref_merged) {
    const int frames = 1, height = 32, width = 32;
    const size_t tokens_want = (size_t)(height / 32) * (size_t)(width / 32);
    const size_t row = (size_t)H3_CUDA_VISION_WIDTH;
    float *pixels, *merged, *ds[3], *ref = NULL;
    char err[512];
    int grid_h = 0, grid_w = 0;
    size_t tokens = 0, i;
    int ok = 0;

    pixels = (float *)malloc((size_t)frames * 3u * height * width *
                             sizeof(float));
    merged = (float *)malloc(tokens_want * row * sizeof(float));
    ds[0] = (float *)malloc(tokens_want * row * sizeof(float));
    ds[1] = (float *)malloc(tokens_want * row * sizeof(float));
    ds[2] = (float *)malloc(tokens_want * row * sizeof(float));
    if (!pixels || !merged || !ds[0] || !ds[1] || !ds[2]) {
        fail_msg("OOM vision buffers");
        goto done;
    }
    fill_pixels(pixels, frames, height, width);
    if (!h3_vision_encode_f32(te_path, pixels, frames, height, width, merged,
                              ds, &grid_h, &grid_w, &tokens, err,
                              sizeof(err))) {
        fail_msg(err[0] ? err : "vision_encode_f32 failed");
        goto done;
    }
    if (grid_h != height / 16 || grid_w != width / 16 || tokens != tokens_want) {
        fail_msg("vision geometry mismatch (patch grid H/16, tokens H/32)");
        goto done;
    }
    if (!all_finite(merged, tokens * row) || !all_finite(ds[0], tokens * row) ||
        !all_finite(ds[1], tokens * row) || !all_finite(ds[2], tokens * row)) {
        fail_msg("vision output non-finite");
        goto done;
    }
    {
        float acc = 0.0f;
        for (i = 0; i < tokens * row; i++)
            acc += fabsf(merged[i]);
        if (acc < 1e-6f) {
            fail_msg("vision merged is all zeros");
            goto done;
        }
    }
    if (ref_merged) {
        FILE *fp = fopen(ref_merged, "rb");
        size_t got;
        double num = 0.0, den = 0.0, max_abs = 0.0;
        if (!fp) {
            fail_msg("cannot open --ref-merged");
            goto done;
        }
        ref = (float *)malloc(tokens * row * sizeof(float));
        if (!ref) {
            fclose(fp);
            fail_msg("OOM ref-merged");
            goto done;
        }
        got = fread(ref, sizeof(float), tokens * row, fp);
        fclose(fp);
        if (got != tokens * row) {
            fail_msg("ref-merged size mismatch");
            goto done;
        }
        for (i = 0; i < tokens * row; i++) {
            double d = (double)merged[i] - (double)ref[i];
            double a = fabs(d);
            num += d * d;
            den += (double)ref[i] * (double)ref[i];
            if (a > max_abs) max_abs = a;
        }
        {
            double rel_l2 = sqrt(num / (den > 1e-24 ? den : 1e-24));
            /* Host F32 vs Metal BF16. Plan nails rel-L2 < 5e-2. */
            printf("vision vs Metal ref: max_abs=%.6g rel-L2=%.6g "
                   "(threshold 5e-2)\n",
                   max_abs, rel_l2);
            if (rel_l2 >= 5e-2) {
                fail_msg("vision rel-L2 vs Metal exceeds 5e-2");
                goto done;
            }
        }
    }
    ok = 1;
    puts("ok: vision encode 32x32 finite, patch grid, merged tokens");
done:
    free(pixels);
    free(merged);
    free(ds[0]);
    free(ds[1]);
    free(ds[2]);
    free(ref);
    return ok;
}

static int test_with_image_te(const char *te_path) {
    char err[512];
    h3_cuda *cuda = NULL;
    h3_nvfp4_store store;
    uint32_t prompt[3] = {32u, 2518u, 38835u};
    uint32_t *ids32 = NULL;
    int32_t *ids = NULL;
    size_t n = 0, i;
    h3_cuda_vision_span span;
    float *pixels, *merged, *ds[3];
    const float *deepstack[3];
    float *plain = NULL, *mm = NULL;
    int vh = 0, vw = 0;
    size_t vtok = 0;
    const unsigned n_layers = 1;
    int ok = 0;
    double cosv;

    memset(&store, 0, sizeof(store));
    memset(&span, 0, sizeof(span));
    pixels = (float *)malloc(3u * 32u * 32u * sizeof(float));
    merged = (float *)malloc((size_t)H3_CUDA_VISION_WIDTH * sizeof(float));
    ds[0] = (float *)malloc((size_t)H3_CUDA_VISION_WIDTH * sizeof(float));
    ds[1] = (float *)malloc((size_t)H3_CUDA_VISION_WIDTH * sizeof(float));
    ds[2] = (float *)malloc((size_t)H3_CUDA_VISION_WIDTH * sizeof(float));
    if (!pixels || !merged || !ds[0] || !ds[1] || !ds[2]) {
        fail_msg("OOM with-image buffers");
        goto done;
    }
    fill_pixels(pixels, 1, 32, 32);
    if (!h3_vision_encode_f32(te_path, pixels, 1, 32, 32, merged, ds, &vh, &vw,
                              &vtok, err, sizeof(err))) {
        fail_msg(err[0] ? err : "vision for TE failed");
        goto done;
    }
    if (vtok != 1 || vh != 32 / 16 || vw != 32 / 16) {
        fail_msg("expected 32x32 vision: patch 2x2, 1 merged token");
        goto done;
    }
    /* Wire the real vision returns. Do not hardcode merged 1x1 as grid. */
    if (!h3_multimodal_cuda_build_fl2va_ids(1, prompt, 3, &vtok, &vh, &vw,
                                            &ids32, &n, &span, err,
                                            sizeof(err))) {
        fail_msg(err);
        goto done;
    }
    if (span.tokens < 2 || span.tokens - 2 != vtok ||
        span.grid_h != vh || span.grid_w != vw) {
        fail_msg("span.tokens-2 must equal vision.tokens; grid is patch H/16");
        goto done;
    }
    printf("wired 32x32 vision: tokens=%zu patch=%dx%d span.tokens-2=%zu\n",
           vtok, vh, vw, span.tokens - 2);
    ids = (int32_t *)malloc(n * sizeof(int32_t));
    if (!ids) {
        fail_msg("OOM ids");
        goto done;
    }
    for (i = 0; i < n; i++)
        ids[i] = (int32_t)ids32[i];
    cuda = h3_cuda_create(0, err, sizeof(err));
    if (!cuda || !h3_nvfp4_cuda_enable(cuda, err, sizeof(err)) ||
        !h3_nvfp4_store_open(te_path, &store, err, sizeof(err))) {
        fail_msg(err[0] ? err : "TE open failed");
        goto done;
    }
    deepstack[0] = ds[0];
    deepstack[1] = ds[1];
    deepstack[2] = ds[2];
    plain = h3_nvfp4_encode_layers_f32(&store, ids, n, n_layers, NULL, err,
                                       sizeof(err));
    mm = h3_nvfp4_encode_layers_multimodal_f32(&store, ids, n, &span, 1, merged,
                                               deepstack, n_layers, NULL, err,
                                               sizeof(err));
    if (!plain || !mm) {
        fail_msg(err[0] ? err : "with-image TE encode failed");
        goto done;
    }
    if (!all_finite(mm, n * (size_t)H3_NVFP4_TEXT_DIM)) {
        fail_msg("with-image TE non-finite");
        goto done;
    }
    cosv = cosine(plain, mm, n * (size_t)H3_NVFP4_TEXT_DIM);
    printf("with-image vs pad-embed TE cosine=%.6f (must not be ~1)\n", cosv);
    if (cosv > 0.999) {
        fail_msg("with-image TE collapsed onto no-vision hidden");
        goto done;
    }
    ok = 1;
    puts("ok: with-image TE distinct and contains Picture presentation");
done:
    free(pixels);
    free(merged);
    free(ds[0]);
    free(ds[1]);
    free(ds[2]);
    free(ids32);
    free(ids);
    free(plain);
    free(mm);
    h3_nvfp4_store_free(&store);
    h3_nvfp4_cuda_disable();
    h3_cuda_free(cuda);
    return ok;
}

int main(int argc, char **argv) {
    const char *te_path = NULL;
    const char *ref_merged = NULL;
    int i;
    g_fail = 0;
    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--te") && i + 1 < argc)
            te_path = argv[++i];
        else if (!strcmp(argv[i], "--ref-merged") && i + 1 < argc)
            ref_merged = argv[++i];
        else if (argv[i][0] != '-')
            te_path = argv[i];
    }
    test_picture_ids();
    test_dit_text_tags();
    test_span_pads_follow_vision_tokens();
    if (!te_path) {
        if (g_fail) return 1;
        fprintf(stderr, "usage: %s --te TE.safetensors [--ref-merged F32]\n",
                argv[0]);
        return 2;
    }
    test_no_image_te(te_path);
    test_vision_encode(te_path, ref_merged);
    test_with_image_te(te_path);
    return g_fail ? 1 : 0;
}
