#include "h3_vae_encode_f32.h"

#include "h3_safetensors.h"

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    RGB_CHANNELS = 3,
    LATENT_CHANNELS = 24,
    MOMENT_CHANNELS = 48,
    LEVELS = 6,
    BLOCKS = 2,
    GROUPS = 32,
    SPATIAL_RATIO = 16,
    TILE_PIXELS = 256,
    TILE_OVERLAP_MIN = 64
};

static const uint32_t level_channels[LEVELS] = {128, 256, 256, 512, 512, 1024};
static const uint32_t space_strides[LEVELS] = {2, 2, 2, 2, 1, 1};
static const uint32_t time_strides[LEVELS] = {1, 2, 2, 1, 1, 1};

typedef struct {
    uint64_t state;
    uint64_t increment;
    float spare;
    int has_spare;
} vae_rng;

typedef struct {
    float *weight;
    float *bias;
    uint32_t input_channels;
    uint32_t output_channels;
    uint32_t kernel;
    uint32_t stride_t;
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t depth_front;
    uint32_t height_before;
    uint32_t height_after;
    uint32_t width_before;
    uint32_t width_after;
} encoder_conv;

typedef struct {
    float *weight;
    float *bias;
} encoder_norm;

typedef struct {
    encoder_norm norm1;
    encoder_conv conv1;
    encoder_norm norm2;
    encoder_conv conv2;
    encoder_conv shortcut;
    int has_shortcut;
} encoder_block;

typedef struct {
    encoder_block blocks[BLOCKS];
    encoder_conv downsample;
    int has_downsample;
} encoder_level;

typedef struct {
    encoder_conv conv_in;
    encoder_level levels[LEVELS];
    encoder_norm norm_out;
    encoder_conv conv_out;
    encoder_conv quant;
    float latent_mean[LATENT_CHANNELS];
    float latent_std[LATENT_CHANNELS];
} encoder_context;

typedef struct {
    int count;
    int length;
    int *starts;
    int *overlaps;
} tile_axis;

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static uint32_t vae_rng_u32(vae_rng *rng) {
    uint64_t old_state = rng->state;
    rng->state = old_state * UINT64_C(6364136223846793005) + rng->increment;
    uint32_t shifted = (uint32_t)(((old_state >> 18u) ^ old_state) >> 27u);
    uint32_t rotation = (uint32_t)(old_state >> 59u);
    return (shifted >> rotation) | (shifted << ((-(int32_t)rotation) & 31));
}

static void vae_rng_seed(vae_rng *rng, uint64_t seed) {
    memset(rng, 0, sizeof(*rng));
    rng->increment = (seed << 1u) | 1u;
    (void)vae_rng_u32(rng);
    rng->state += seed ^ UINT64_C(0x9e3779b97f4a7c15);
    (void)vae_rng_u32(rng);
}

static float vae_rng_normal(vae_rng *rng) {
    double u1, u2, radius, angle;
    if (rng->has_spare) {
        rng->has_spare = 0;
        return rng->spare;
    }
    u1 = ((double)vae_rng_u32(rng) + 1.0) / 4294967297.0;
    u2 = ((double)vae_rng_u32(rng) + 0.5) / 4294967296.0;
    radius = sqrt(-2.0 * log(u1));
    angle = 2.0 * 3.14159265358979323846 * u2;
    rng->spare = (float)(radius * sin(angle));
    rng->has_spare = 1;
    return (float)(radius * cos(angle));
}

void h3_vae_augment_0p999(float *rows, size_t count, uint64_t seed) {
    vae_rng rng;
    size_t index;
    if (!rows || !count) return;
    vae_rng_seed(&rng, seed);
    for (index = 0; index < count; index++)
        rows[index] = 0.999f * rows[index] + 0.001f * vae_rng_normal(&rng);
}

static int video_shape(int channels, int time, int height, int width,
                       size_t *latent_count, size_t *row_count) {
    if (channels <= 0 || time <= 0 || height <= 0 || width <= 0 ||
        (height & 1) || (width & 1))
        return 0;
    if ((size_t)channels > SIZE_MAX / (size_t)time ||
        (size_t)channels * (size_t)time > SIZE_MAX / (size_t)height ||
        (size_t)channels * (size_t)time * (size_t)height > SIZE_MAX / (size_t)width)
        return 0;
    *latent_count = (size_t)channels * (size_t)time * (size_t)height *
                    (size_t)width;
    *row_count = (size_t)time * (size_t)(height / 2) * (size_t)(width / 2) *
                 (size_t)channels * 4u;
    return 1;
}

int h3_vae_patchify_video_96(const float *latent_24thw, int time,
                             int height, int width, float *rows_seq96,
                             size_t row_elements) {
    size_t latent_count, expected, output = 0;
    int t, h, w, c, dh, dw;
    if (!latent_24thw || !rows_seq96 ||
        !video_shape(LATENT_CHANNELS, time, height, width, &latent_count,
                     &expected) ||
        row_elements != expected || latent_count != expected)
        return 0;
    for (t = 0; t < time; t++)
        for (h = 0; h < height; h += 2)
            for (w = 0; w < width; w += 2)
                for (c = 0; c < LATENT_CHANNELS; c++)
                    for (dh = 0; dh < 2; dh++)
                        for (dw = 0; dw < 2; dw++) {
                            size_t input = (((size_t)c * (size_t)time +
                                (size_t)t) * (size_t)height + (size_t)(h + dh)) *
                                (size_t)width + (size_t)(w + dw);
                            rows_seq96[output++] = latent_24thw[input];
                        }
    return output == row_elements;
}

void h3_vae_latent_f32_free(h3_vae_latent_f32 *latent) {
    if (!latent) return;
    free(latent->values);
    memset(latent, 0, sizeof(*latent));
}

static size_t tensor_elements(uint32_t depth, uint32_t height, uint32_t width,
                              uint32_t channels) {
    return (size_t)depth * (size_t)height * (size_t)width * (size_t)channels;
}

static int reflect_coord(int coordinate, int length) {
    if (coordinate < 0) return -coordinate;
    if (coordinate >= length) return 2 * length - coordinate - 2;
    return coordinate;
}

static float *load_named_f32(const h3_st_header *header, const char *name,
                             size_t want, char *error, size_t error_size) {
    size_t got = 0;
    h3_dtype dtype = H3_DTYPE_UNKNOWN;
    float *data = h3_st_load_f32(header, name, &got, &dtype, error, error_size);
    if (!data) return NULL;
    if (got != want) {
        fail(error, error_size, "%s has %zu elements, expected %zu", name, got,
             want);
        free(data);
        return NULL;
    }
    return data;
}

static void free_conv(encoder_conv *conv) {
    if (!conv) return;
    free(conv->weight);
    free(conv->bias);
    memset(conv, 0, sizeof(*conv));
}

static void free_norm(encoder_norm *norm) {
    if (!norm) return;
    free(norm->weight);
    free(norm->bias);
    memset(norm, 0, sizeof(*norm));
}

static void cleanup_encoder(encoder_context *encoder) {
    int level, block;
    if (!encoder) return;
    free_conv(&encoder->conv_in);
    for (level = 0; level < LEVELS; level++) {
        for (block = 0; block < BLOCKS; block++) {
            encoder_block *item = &encoder->levels[level].blocks[block];
            free_norm(&item->norm1);
            free_conv(&item->conv1);
            free_norm(&item->norm2);
            free_conv(&item->conv2);
            free_conv(&item->shortcut);
        }
        free_conv(&encoder->levels[level].downsample);
    }
    free_norm(&encoder->norm_out);
    free_conv(&encoder->conv_out);
    free_conv(&encoder->quant);
    memset(encoder, 0, sizeof(*encoder));
}

static int load_conv(const h3_st_header *header, encoder_conv *conv,
                     const char *prefix, uint32_t input_channels,
                     uint32_t output_channels, uint32_t kernel,
                     uint32_t stride_t, uint32_t stride_h, uint32_t stride_w,
                     uint32_t depth_front, uint32_t height_before,
                     uint32_t height_after, uint32_t width_before,
                     uint32_t width_after, char *error, size_t error_size) {
    char name[192];
    size_t weight_n = (size_t)output_channels * (size_t)input_channels *
                      (size_t)kernel * (size_t)kernel * (size_t)kernel;
    conv->input_channels = input_channels;
    conv->output_channels = output_channels;
    conv->kernel = kernel;
    conv->stride_t = stride_t;
    conv->stride_h = stride_h;
    conv->stride_w = stride_w;
    conv->depth_front = depth_front;
    conv->height_before = height_before;
    conv->height_after = height_after;
    conv->width_before = width_before;
    conv->width_after = width_after;
    snprintf(name, sizeof(name), "%s.weight", prefix);
    conv->weight = load_named_f32(header, name, weight_n, error, error_size);
    if (!conv->weight) return 0;
    snprintf(name, sizeof(name), "%s.bias", prefix);
    conv->bias = load_named_f32(header, name, output_channels, error,
                                error_size);
    return conv->bias != NULL;
}

static int load_norm(const h3_st_header *header, encoder_norm *norm,
                     const char *prefix, uint32_t channels, char *error,
                     size_t error_size) {
    char name[192];
    snprintf(name, sizeof(name), "%s.weight", prefix);
    norm->weight = load_named_f32(header, name, channels, error, error_size);
    if (!norm->weight) return 0;
    snprintf(name, sizeof(name), "%s.bias", prefix);
    norm->bias = load_named_f32(header, name, channels, error, error_size);
    return norm->bias != NULL;
}

static int load_weights(encoder_context *encoder, const h3_st_header *header,
                        char *error, size_t error_size) {
    uint32_t previous = 128;
    int level, block;
    size_t i;
    if (!load_conv(header, &encoder->conv_in, "encoder.conv_in", 3, 128, 3,
                   1, 1, 1, 2, 1, 1, 1, 1, error, error_size))
        return 0;
    for (level = 0; level < LEVELS; level++) {
        uint32_t channels = level_channels[level];
        for (block = 0; block < BLOCKS; block++) {
            encoder_block *item = &encoder->levels[level].blocks[block];
            uint32_t input_channels = block ? channels : previous;
            char prefix[192];
            char name[224];
            snprintf(prefix, sizeof(prefix), "encoder.down.%d.block.%d", level,
                     block);
            snprintf(name, sizeof(name), "%s.norm1", prefix);
            if (!load_norm(header, &item->norm1, name, input_channels, error,
                           error_size))
                return 0;
            snprintf(name, sizeof(name), "%s.conv1", prefix);
            if (!load_conv(header, &item->conv1, name, input_channels, channels,
                           3, 1, 1, 1, 2, 1, 1, 1, 1, error, error_size))
                return 0;
            snprintf(name, sizeof(name), "%s.norm2", prefix);
            if (!load_norm(header, &item->norm2, name, channels, error,
                           error_size))
                return 0;
            snprintf(name, sizeof(name), "%s.conv2", prefix);
            if (!load_conv(header, &item->conv2, name, channels, channels, 3,
                           1, 1, 1, 2, 1, 1, 1, 1, error, error_size))
                return 0;
            if (input_channels != channels) {
                snprintf(name, sizeof(name), "%s.nin_shortcut", prefix);
                if (!load_conv(header, &item->shortcut, name, input_channels,
                               channels, 1, 1, 1, 1, 0, 0, 0, 0, 0, error,
                               error_size))
                    return 0;
                item->has_shortcut = 1;
            }
        }
        if (space_strides[level] * time_strides[level] > 1) {
            char name[192];
            uint32_t spatial_tail = space_strides[level] == 2 ? 1 : 0;
            snprintf(name, sizeof(name), "encoder.down.%d.downsample.conv",
                     level);
            if (!load_conv(header, &encoder->levels[level].downsample, name,
                           channels, channels, 3, time_strides[level],
                           space_strides[level], space_strides[level], 2, 0,
                           spatial_tail, 0, spatial_tail, error, error_size))
                return 0;
            encoder->levels[level].has_downsample = 1;
        }
        previous = channels;
    }
    if (!load_norm(header, &encoder->norm_out, "encoder.norm_out", 1024, error,
                   error_size) ||
        !load_conv(header, &encoder->conv_out, "encoder.conv_out", 1024,
                   MOMENT_CHANNELS, 3, 1, 1, 1, 2, 1, 1, 1, 1, error,
                   error_size) ||
        !load_conv(header, &encoder->quant, "quant_conv", MOMENT_CHANNELS,
                   MOMENT_CHANNELS, 1, 1, 1, 1, 0, 0, 0, 0, 0, error,
                   error_size))
        return 0;
    {
        float *mean = load_named_f32(header, "latents_mean", LATENT_CHANNELS,
                                     error, error_size);
        float *std = load_named_f32(header, "latents_std", LATENT_CHANNELS,
                                    error, error_size);
        if (!mean || !std) {
            free(mean);
            free(std);
            return 0;
        }
        memcpy(encoder->latent_mean, mean, sizeof(encoder->latent_mean));
        memcpy(encoder->latent_std, std, sizeof(encoder->latent_std));
        free(mean);
        free(std);
    }
    for (i = 0; i < LATENT_CHANNELS; i++)
        if (!(encoder->latent_std[i] > 0.0f) ||
            !isfinite(encoder->latent_mean[i]) ||
            !isfinite(encoder->latent_std[i])) {
            fail(error, error_size, "video VAE latent deviation is invalid");
            return 0;
        }
    return 1;
}

static int causal_pad(const float *input, uint32_t depth, uint32_t height,
                      uint32_t width, uint32_t channels,
                      uint32_t depth_front, uint32_t height_before,
                      uint32_t height_after, uint32_t width_before,
                      uint32_t width_after, float *output) {
    uint32_t out_d = depth + depth_front;
    uint32_t out_h = height + height_before + height_after;
    uint32_t out_w = width + width_before + width_after;
    uint32_t t, y, x, c;
    for (t = 0; t < out_d; t++)
        for (y = 0; y < out_h; y++)
            for (x = 0; x < out_w; x++) {
                size_t dest_base = (((size_t)t * out_h + y) * out_w + x) *
                                   channels;
                if (t < depth_front) {
                    for (c = 0; c < channels; c++) output[dest_base + c] = 0.0f;
                    continue;
                }
                {
                    uint32_t src_t = t - depth_front;
                    int src_y = reflect_coord((int)y - (int)height_before,
                                              (int)height);
                    int src_x = reflect_coord((int)x - (int)width_before,
                                              (int)width);
                    size_t src_base =
                        (((size_t)src_t * height + (uint32_t)src_y) * width +
                         (uint32_t)src_x) *
                        channels;
                    memcpy(output + dest_base, input + src_base,
                           (size_t)channels * sizeof(float));
                }
            }
    return 1;
}

static int conv3d_nhwc(const float *input, uint32_t depth, uint32_t height,
                       uint32_t width, const encoder_conv *conv,
                       float *output) {
    uint32_t kd = conv->kernel, kh = conv->kernel, kw = conv->kernel;
    uint32_t cin = conv->input_channels, cout = conv->output_channels;
    uint32_t od = (depth - kd) / conv->stride_t + 1;
    uint32_t oh = (height - kh) / conv->stride_h + 1;
    uint32_t ow = (width - kw) / conv->stride_w + 1;
    int ot, oy, ox, oc;
    if (depth < kd || height < kh || width < kw) return 0;
    if (kd == 1 && kh == 1 && kw == 1 && conv->stride_t == 1 &&
        conv->stride_h == 1 && conv->stride_w == 1) {
        int spatial = (int)depth * (int)height * (int)width;
        int n;
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (n = 0; n < spatial; n++) {
            uint32_t o;
            const float *in = input + (size_t)n * cin;
            float *out = output + (size_t)n * cout;
            for (o = 0; o < cout; o++) {
                const float *w = conv->weight + (size_t)o * cin;
                float acc = conv->bias[o];
                uint32_t ic;
                for (ic = 0; ic < cin; ic++) acc += in[ic] * w[ic];
                out[o] = acc;
            }
        }
        return 1;
    }
#ifdef _OPENMP
#pragma omp parallel for collapse(3) private(oc)
#endif
    for (ot = 0; ot < (int)od; ot++)
        for (oy = 0; oy < (int)oh; oy++)
            for (ox = 0; ox < (int)ow; ox++) {
                uint32_t in_t = (uint32_t)ot * conv->stride_t;
                uint32_t in_y = (uint32_t)oy * conv->stride_h;
                uint32_t in_x = (uint32_t)ox * conv->stride_w;
                float *out = output +
                             (((size_t)ot * oh + (uint32_t)oy) * ow +
                              (uint32_t)ox) * cout;
                for (oc = 0; oc < (int)cout; oc++) {
                    const float *wbase = conv->weight +
                        (size_t)oc * cin * kd * kh * kw;
                    float acc = conv->bias[oc];
                    uint32_t ic, dt, dy, dx;
                    for (ic = 0; ic < cin; ic++)
                        for (dt = 0; dt < kd; dt++)
                            for (dy = 0; dy < kh; dy++)
                                for (dx = 0; dx < kw; dx++) {
                                    size_t in_i =
                                        ((((size_t)(in_t + dt) * height +
                                           (in_y + dy)) *
                                          width + (in_x + dx)) *
                                         cin) + ic;
                                    size_t w_i =
                                        ((((size_t)ic * kd + dt) * kh + dy) *
                                         kw) + dx;
                                    acc += input[in_i] * wbase[w_i];
                                }
                    out[oc] = acc;
                }
            }
    return 1;
}

static int run_conv(const encoder_conv *conv, const float *input,
                    uint32_t depth, uint32_t height, uint32_t width,
                    float **output, uint32_t *out_d, uint32_t *out_h,
                    uint32_t *out_w, char *error, size_t error_size) {
    uint32_t padded_d = depth + conv->depth_front;
    uint32_t padded_h = height + conv->height_before + conv->height_after;
    uint32_t padded_w = width + conv->width_before + conv->width_after;
    int needs_pad = conv->depth_front || conv->height_before ||
                    conv->height_after || conv->width_before ||
                    conv->width_after;
    float *padded = NULL;
    float *out;
    const float *source = input;
    uint32_t src_d = depth, src_h = height, src_w = width;
    *out_d = (padded_d - conv->kernel) / conv->stride_t + 1;
    *out_h = (padded_h - conv->kernel) / conv->stride_h + 1;
    *out_w = (padded_w - conv->kernel) / conv->stride_w + 1;
    if (needs_pad) {
        padded = malloc(tensor_elements(padded_d, padded_h, padded_w,
                                        conv->input_channels) *
                        sizeof(*padded));
        if (!padded) {
            fail(error, error_size, "cannot allocate visual encoder padding");
            return 0;
        }
        causal_pad(input, depth, height, width, conv->input_channels,
                   conv->depth_front, conv->height_before, conv->height_after,
                   conv->width_before, conv->width_after, padded);
        source = padded;
        src_d = padded_d;
        src_h = padded_h;
        src_w = padded_w;
    }
    out = malloc(tensor_elements(*out_d, *out_h, *out_w, conv->output_channels) *
                 sizeof(*out));
    if (!out) {
        free(padded);
        fail(error, error_size, "cannot allocate visual encoder convolution");
        return 0;
    }
    if (!conv3d_nhwc(source, src_d, src_h, src_w, conv, out)) {
        free(padded);
        free(out);
        fail(error, error_size, "visual encoder Conv3d failed");
        return 0;
    }
    free(padded);
    *output = out;
    return 1;
}

static int group_norm_silu(const float *input, uint32_t depth, uint32_t height,
                           uint32_t width, uint32_t channels,
                           const float *weight, const float *bias,
                           float *output) {
    uint32_t cpg = channels / GROUPS;
    int t, g;
    uint32_t spatial = height * width;
    uint32_t elements = spatial * cpg;
    if (!cpg || (channels % GROUPS) != 0) return 0;
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (t = 0; t < (int)depth; t++)
        for (g = 0; g < GROUPS; g++) {
            double sum = 0.0, sq = 0.0;
            float mean, inv;
            uint32_t s, cg;
            for (s = 0; s < spatial; s++) {
                const float *pix = input + ((size_t)t * spatial + s) * channels +
                                   (size_t)g * cpg;
                for (cg = 0; cg < cpg; cg++) sum += (double)pix[cg];
            }
            mean = (float)(sum / (double)elements);
            for (s = 0; s < spatial; s++) {
                const float *pix = input + ((size_t)t * spatial + s) * channels +
                                   (size_t)g * cpg;
                for (cg = 0; cg < cpg; cg++) {
                    double d = (double)pix[cg] - (double)mean;
                    sq += d * d;
                }
            }
            inv = 1.0f / sqrtf((float)(sq / (double)elements) + 1e-6f);
            for (s = 0; s < spatial; s++) {
                const float *pix = input + ((size_t)t * spatial + s) * channels +
                                   (size_t)g * cpg;
                float *out = output + ((size_t)t * spatial + s) * channels +
                             (size_t)g * cpg;
                for (cg = 0; cg < cpg; cg++) {
                    uint32_t ch = g * cpg + cg;
                    float value = (pix[cg] - mean) * inv * weight[ch] + bias[ch];
                    out[cg] = value / (1.0f + expf(-value));
                }
            }
        }
    return 1;
}

static int run_block(const encoder_block *block, const float *input,
                     uint32_t depth, uint32_t height, uint32_t width,
                     uint32_t input_channels, uint32_t output_channels,
                     float **output, char *error, size_t error_size) {
    size_t in_n = tensor_elements(depth, height, width, input_channels);
    size_t out_n = tensor_elements(depth, height, width, output_channels);
    float *norm1 = malloc(in_n * sizeof(*norm1));
    float *hidden = NULL;
    float *norm2 = malloc(out_n * sizeof(*norm2));
    float *conv2 = NULL;
    float *shortcut = NULL;
    uint32_t dummy_d, dummy_h, dummy_w;
    int ok = 0;
    size_t i;
    if (!norm1 || !norm2) {
        fail(error, error_size, "cannot allocate visual encoder residual block");
        goto done;
    }
    if (!group_norm_silu(input, depth, height, width, input_channels,
                         block->norm1.weight, block->norm1.bias, norm1)) {
        fail(error, error_size, "visual encoder group norm 1 failed");
        goto done;
    }
    if (!run_conv(&block->conv1, norm1, depth, height, width, &hidden, &dummy_d,
                  &dummy_h, &dummy_w, error, error_size))
        goto done;
    if (!group_norm_silu(hidden, depth, height, width, output_channels,
                         block->norm2.weight, block->norm2.bias, norm2)) {
        fail(error, error_size, "visual encoder group norm 2 failed");
        goto done;
    }
    if (!run_conv(&block->conv2, norm2, depth, height, width, &conv2, &dummy_d,
                  &dummy_h, &dummy_w, error, error_size))
        goto done;
    if (block->has_shortcut) {
        if (!run_conv(&block->shortcut, input, depth, height, width, &shortcut,
                      &dummy_d, &dummy_h, &dummy_w, error, error_size))
            goto done;
        for (i = 0; i < out_n; i++) conv2[i] += shortcut[i];
    } else {
        for (i = 0; i < out_n; i++) conv2[i] += input[i];
    }
    ok = 1;
done:
    free(norm1);
    free(hidden);
    free(norm2);
    free(shortcut);
    if (!ok) {
        free(conv2);
        return 0;
    }
    *output = conv2;
    return 1;
}

static float *encode_tile(const encoder_context *encoder, const float *pixels,
                          int frames, int height, int width, int *latent_time,
                          char *error, size_t error_size) {
    static const float mean[] = {0.485f, 0.456f, 0.406f};
    static const float deviation[] = {0.229f, 0.224f, 0.225f};
    size_t pixel_count = (size_t)frames * height * width * RGB_CHANNELS;
    float *normalized = malloc(pixel_count * sizeof(*normalized));
    float *hidden = NULL, *next = NULL;
    uint32_t depth, h, w, next_d, next_h, next_w, previous;
    int level, block;
    size_t dest = 0;
    float *norm = NULL, *moments = NULL, *quant = NULL, *latent = NULL;
    int time, y, x, channel;
    uint32_t oc_t, oc_y, oc_x, oc_c;
    if (!normalized) {
        fail(error, error_size, "out of memory normalizing visual anchor");
        return NULL;
    }
    for (time = 0; time < frames; time++)
        for (y = 0; y < height; y++)
            for (x = 0; x < width; x++)
                for (channel = 0; channel < RGB_CHANNELS; channel++) {
                    size_t source = (((size_t)channel * frames + time) *
                                         height + y) * width + x;
                    normalized[dest++] =
                        (pixels[source] - mean[channel]) / deviation[channel];
                }
    depth = (uint32_t)frames;
    h = (uint32_t)height;
    w = (uint32_t)width;
    if (!run_conv(&encoder->conv_in, normalized, depth, h, w, &hidden, &next_d,
                  &next_h, &next_w, error, error_size)) {
        free(normalized);
        return NULL;
    }
    free(normalized);
    depth = next_d;
    h = next_h;
    w = next_w;
    previous = 128;
    for (level = 0; level < LEVELS; level++) {
        uint32_t channels = level_channels[level];
        for (block = 0; block < BLOCKS; block++) {
            uint32_t input_channels = block ? channels : previous;
            if (!run_block(&encoder->levels[level].blocks[block], hidden, depth,
                           h, w, input_channels, channels, &next, error,
                           error_size)) {
                free(hidden);
                return NULL;
            }
            free(hidden);
            hidden = next;
            next = NULL;
        }
        if (encoder->levels[level].has_downsample) {
            if (!run_conv(&encoder->levels[level].downsample, hidden, depth, h,
                          w, &next, &next_d, &next_h, &next_w, error,
                          error_size)) {
                free(hidden);
                return NULL;
            }
            free(hidden);
            hidden = next;
            next = NULL;
            depth = next_d;
            h = next_h;
            w = next_w;
        }
        previous = channels;
    }
    {
        size_t hidden_n = tensor_elements(depth, h, w, 1024);
        size_t latent_n = (size_t)LATENT_CHANNELS * depth * h * w;
        uint32_t dummy_d, dummy_h, dummy_w;
        norm = malloc(hidden_n * sizeof(*norm));
        if (!norm ||
            !group_norm_silu(hidden, depth, h, w, 1024, encoder->norm_out.weight,
                             encoder->norm_out.bias, norm)) {
            fail(error, error_size, "visual encoder output norm failed");
            free(hidden);
            free(norm);
            return NULL;
        }
        free(hidden);
        hidden = NULL;
        if (!run_conv(&encoder->conv_out, norm, depth, h, w, &moments, &dummy_d,
                      &dummy_h, &dummy_w, error, error_size)) {
            free(norm);
            return NULL;
        }
        free(norm);
        norm = NULL;
        if (!run_conv(&encoder->quant, moments, depth, h, w, &quant, &dummy_d,
                      &dummy_h, &dummy_w, error, error_size)) {
            free(moments);
            return NULL;
        }
        free(moments);
        latent = malloc(latent_n * sizeof(*latent));
        if (!latent) {
            fail(error, error_size, "cannot allocate visual encoder latent");
            free(quant);
            return NULL;
        }
        for (oc_c = 0; oc_c < LATENT_CHANNELS; oc_c++)
            for (oc_t = 0; oc_t < depth; oc_t++)
                for (oc_y = 0; oc_y < h; oc_y++)
                    for (oc_x = 0; oc_x < w; oc_x++) {
                        size_t source = (((size_t)oc_t * h + oc_y) * w + oc_x) *
                                            MOMENT_CHANNELS + oc_c;
                        size_t target = (((size_t)oc_c * depth + oc_t) * h +
                                         oc_y) * w + oc_x;
                        latent[target] =
                            (quant[source] - encoder->latent_mean[oc_c]) /
                            encoder->latent_std[oc_c];
                    }
        free(quant);
        *latent_time = (int)depth;
        return latent;
    }
}

static void tile_axis_free(tile_axis *axis) {
    if (!axis) return;
    free(axis->starts);
    free(axis->overlaps);
    memset(axis, 0, sizeof(*axis));
}

static int tile_axis_build(int extent, tile_axis *axis, char *error,
                           size_t error_size) {
    memset(axis, 0, sizeof(*axis));
    if (extent < SPATIAL_RATIO || extent % SPATIAL_RATIO) {
        fail(error, error_size,
             "visual encoder extent must be a multiple of %d", SPATIAL_RATIO);
        return 0;
    }
    if (extent <= TILE_PIXELS) {
        axis->count = 1;
        axis->length = extent;
        axis->starts = calloc(1, sizeof(*axis->starts));
        if (!axis->starts) {
            fail(error, error_size, "out of memory constructing encoder tiles");
            return 0;
        }
        return 1;
    }
    {
        int count = (extent + TILE_PIXELS - 1) / TILE_PIXELS;
        int index, remaining, unit;
        while (TILE_PIXELS * count - TILE_OVERLAP_MIN * (count - 1) < extent)
            count++;
        axis->starts = calloc((size_t)count, sizeof(*axis->starts));
        axis->overlaps = malloc((size_t)(count - 1) * sizeof(*axis->overlaps));
        if (!axis->starts || !axis->overlaps) {
            tile_axis_free(axis);
            fail(error, error_size, "out of memory constructing encoder tiles");
            return 0;
        }
        for (index = 0; index < count - 1; index++)
            axis->overlaps[index] = TILE_OVERLAP_MIN;
        remaining = TILE_PIXELS * count - TILE_OVERLAP_MIN * (count - 1) -
                    extent;
        for (unit = 0; unit < remaining / SPATIAL_RATIO; unit++)
            axis->overlaps[unit % (count - 1)] += SPATIAL_RATIO;
        for (index = 1; index < count; index++)
            axis->starts[index] = axis->starts[index - 1] + TILE_PIXELS -
                                  axis->overlaps[index - 1];
        axis->count = count;
        axis->length = TILE_PIXELS;
        return 1;
    }
}

static float *extract_pixel_tile(const float *pixels, int frames, int full_h,
                                 int full_w, int start_y, int start_x,
                                 int tile_h, int tile_w, char *error,
                                 size_t error_size) {
    size_t count = (size_t)RGB_CHANNELS * frames * tile_h * tile_w;
    float *tile = malloc(count * sizeof(*tile));
    int channel, time, y;
    if (!tile) {
        fail(error, error_size, "out of memory extracting visual anchor tile");
        return NULL;
    }
    for (channel = 0; channel < RGB_CHANNELS; channel++)
        for (time = 0; time < frames; time++)
            for (y = 0; y < tile_h; y++) {
                size_t source = (((size_t)channel * frames + time) * full_h +
                                  start_y + y) * full_w + start_x;
                size_t destination = (((size_t)channel * frames + time) *
                                       tile_h + y) * tile_w;
                memcpy(tile + destination, pixels + source,
                       (size_t)tile_w * sizeof(*tile));
            }
    return tile;
}

static int stitch_latents(float **tiles, int latent_time,
                          const tile_axis *y_axis, const tile_axis *x_axis,
                          h3_vae_latent_f32 *output, char *error,
                          size_t error_size) {
    int tile_h = y_axis->length / SPATIAL_RATIO;
    int tile_w = x_axis->length / SPATIAL_RATIO;
    int full_h = (y_axis->starts[y_axis->count - 1] + y_axis->length) /
                 SPATIAL_RATIO;
    int full_w = (x_axis->starts[x_axis->count - 1] + x_axis->length) /
                 SPATIAL_RATIO;
    size_t count = (size_t)LATENT_CHANNELS * latent_time * full_h * full_w;
    float *values = malloc(count * sizeof(*values));
    int tile_y, tile_x;
    if (!values) {
        fail(error, error_size, "out of memory stitching visual latents");
        return 0;
    }
    for (tile_y = 0; tile_y < y_axis->count; tile_y++)
        for (tile_x = 0; tile_x < x_axis->count; tile_x++) {
            int index = tile_y * x_axis->count + tile_x;
            const float *current = tiles[index];
            const float *above = tile_y ? tiles[index - x_axis->count] : NULL;
            const float *left = tile_x ? tiles[index - 1] : NULL;
            int overlap_y = tile_y ?
                y_axis->overlaps[tile_y - 1] / SPATIAL_RATIO : 0;
            int overlap_x = tile_x ?
                x_axis->overlaps[tile_x - 1] / SPATIAL_RATIO : 0;
            int keep_h = tile_h - (tile_y + 1 < y_axis->count ?
                y_axis->overlaps[tile_y] / SPATIAL_RATIO : 0);
            int keep_w = tile_w - (tile_x + 1 < x_axis->count ?
                x_axis->overlaps[tile_x] / SPATIAL_RATIO : 0);
            int destination_y = y_axis->starts[tile_y] / SPATIAL_RATIO;
            int destination_x = x_axis->starts[tile_x] / SPATIAL_RATIO;
            int channel, time, y, x;
            for (channel = 0; channel < LATENT_CHANNELS; channel++)
                for (time = 0; time < latent_time; time++)
                    for (y = 0; y < keep_h; y++)
                        for (x = 0; x < keep_w; x++) {
                            size_t local = (((size_t)channel * latent_time +
                                time) * tile_h + y) * tile_w + x;
                            float value = current[local];
                            if (above && y < overlap_y) {
                                size_t source = (((size_t)channel * latent_time +
                                    time) * tile_h + tile_h - overlap_y + y) *
                                    tile_w + x;
                                float weight = (float)y / (float)overlap_y;
                                value = above[source] * (1.0f - weight) +
                                        value * weight;
                            }
                            if (left && x < overlap_x) {
                                size_t source = (((size_t)channel * latent_time +
                                    time) * tile_h + y) * tile_w +
                                    tile_w - overlap_x + x;
                                float weight = (float)x / (float)overlap_x;
                                value = left[source] * (1.0f - weight) +
                                        value * weight;
                            }
                            {
                                size_t destination =
                                    (((size_t)channel * latent_time + time) *
                                     full_h + destination_y + y) * full_w +
                                    destination_x + x;
                                values[destination] = value;
                            }
                        }
        }
    output->time = latent_time;
    output->height = full_h;
    output->width = full_w;
    output->values = values;
    return 1;
}

int h3_vae_encode_f32(const char *vae_path, const float *rgb01_chtw,
                      int frames, int height, int width,
                      h3_vae_latent_f32 *out, char *error, size_t error_size) {
    h3_st_header header;
    encoder_context encoder;
    tile_axis y_axis, x_axis;
    float **tiles = NULL;
    int ok, tile_count, latent_time = 0, y, x, completed;
    memset(&header, 0, sizeof(header));
    memset(&encoder, 0, sizeof(encoder));
    memset(&y_axis, 0, sizeof(y_axis));
    memset(&x_axis, 0, sizeof(x_axis));
    if (error && error_size) error[0] = '\0';
    if (out) memset(out, 0, sizeof(*out));
    if (!vae_path || !*vae_path || !rgb01_chtw || !out || frames < 1 ||
        height < 32 || width < 32 || height % SPATIAL_RATIO ||
        width % SPATIAL_RATIO) {
        fail(error, error_size, "invalid visual encoder arguments");
        return 0;
    }
    ok = tile_axis_build(height, &y_axis, error, error_size) &&
         tile_axis_build(width, &x_axis, error, error_size);
    if (!ok) {
        tile_axis_free(&y_axis);
        tile_axis_free(&x_axis);
        return 0;
    }
    if (!h3_st_read_header(vae_path, &header, error, error_size) ||
        !load_weights(&encoder, &header, error, error_size)) {
        cleanup_encoder(&encoder);
        h3_st_free_header(&header);
        tile_axis_free(&y_axis);
        tile_axis_free(&x_axis);
        return 0;
    }
    tile_count = y_axis.count * x_axis.count;
    tiles = calloc((size_t)tile_count, sizeof(*tiles));
    if (!tiles) {
        fail(error, error_size, "out of memory allocating visual encoder tiles");
        cleanup_encoder(&encoder);
        h3_st_free_header(&header);
        tile_axis_free(&y_axis);
        tile_axis_free(&x_axis);
        return 0;
    }
    ok = 1;
    for (y = 0, completed = 0; ok && y < y_axis.count; y++)
        for (x = 0; ok && x < x_axis.count; x++, completed++) {
            float *tile = extract_pixel_tile(
                rgb01_chtw, frames, height, width, y_axis.starts[y],
                x_axis.starts[x], y_axis.length, x_axis.length, error,
                error_size);
            int current_time = 0;
            if (tile)
                tiles[completed] = encode_tile(&encoder, tile, frames,
                                               y_axis.length, x_axis.length,
                                               &current_time, error,
                                               error_size);
            free(tile);
            ok = tiles[completed] != NULL &&
                 (!latent_time || latent_time == current_time);
            if (ok) latent_time = current_time;
        }
    if (ok)
        ok = stitch_latents(tiles, latent_time, &y_axis, &x_axis, out, error,
                            error_size);
    if (tiles) {
        int index;
        for (index = 0; index < tile_count; index++) free(tiles[index]);
    }
    free(tiles);
    cleanup_encoder(&encoder);
    h3_st_free_header(&header);
    tile_axis_free(&y_axis);
    tile_axis_free(&x_axis);
    if (!ok) h3_vae_latent_f32_free(out);
    return ok;
}
