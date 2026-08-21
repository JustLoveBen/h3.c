#include "h3_audio_vae_cuda.h"

#include "h3_gpu_cuda_audio.h"
#include "h3_safetensors.h"

#include <cuda_runtime.h>
#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif

enum {
    LATENT_CHANNELS = H3_AUDIO_VAE_LATENT_CHANNELS,
    LATENT_DIM = H3_AUDIO_VAE_LATENT_DIM,
    DECODER_DIM = H3_AUDIO_VAE_DECODER_DIM,
    STEREO = H3_AUDIO_VAE_STEREO,
    STAGES = H3_AUDIO_VAE_STAGES,
    RESBLOCKS = H3_AUDIO_VAE_RESBLOCKS,
    RESIDUAL_PAIRS = H3_AUDIO_VAE_RESIDUAL_PAIRS,
    FILTER_SIZE = H3_AUDIO_VAE_FILTER_SIZE,
    SAMPLE_RATE = H3_AUDIO_VAE_SAMPLE_RATE,
    HOP_LENGTH = H3_AUDIO_VAE_HOP_LENGTH
};

typedef struct {
    float *d_weight;
    float *d_bias;
    uint32_t in_ch;
    uint32_t out_ch;
    uint32_t kernel;
    uint32_t padding;
    uint32_t dilation;
    uint32_t stride;
    int transpose;
    int has_bias;
} audio_conv;

typedef struct {
    float *d_alpha;
    float *d_beta;
} audio_act;

typedef struct {
    audio_act acts[RESIDUAL_PAIRS * 2];
    audio_conv convs1[RESIDUAL_PAIRS];
    audio_conv convs2[RESIDUAL_PAIRS];
} audio_resblock;

typedef struct {
    audio_conv upsample;
    audio_resblock blocks[RESBLOCKS];
} audio_stage;

typedef struct {
    h3_cuda *cuda;
    h3_st_header header;
    float *d_up_filter;
    float *d_down_filter;
    float *d_hidden;
    uint32_t length;
    float mean[LATENT_CHANNELS];
    float std[LATENT_CHANNELS];
} audio_ctx;

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static void fail(char *error, size_t error_size, const char *fmt, ...) {
    if (!error || !error_size) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(error, error_size, fmt, ap);
    va_end(ap);
}

static int is_safetensors_path(const char *path) {
    size_t n;
    if (!path) return 0;
    n = strlen(path);
    return n > 12 && strcmp(path + n - 12, ".safetensors") == 0;
}

static int resolve_weights_file(const char *path, char *out, size_t out_size,
                                char *error, size_t error_size) {
    struct stat st;
    if (!path || !*path) {
        fail(error, error_size, "audio VAE weights path required");
        return 0;
    }
    if (stat(path, &st) != 0) {
        fail(error, error_size, "cannot stat audio VAE weights %s: %s", path,
             strerror(errno));
        return 0;
    }
    if (S_ISREG(st.st_mode) && is_safetensors_path(path)) {
        if (strlen(path) + 1 > out_size) {
            fail(error, error_size, "audio VAE path too long");
            return 0;
        }
        memcpy(out, path, strlen(path) + 1);
        return 1;
    }
    if (S_ISDIR(st.st_mode)) {
        /* Prefer known Comfy filename, else first *.safetensors. */
        static const char *candidates[] = {
            "minimax_h3_audio_vae_fp32.safetensors",
            "model.safetensors",
            NULL
        };
        size_t i;
        for (i = 0; candidates[i]; i++) {
            snprintf(out, out_size, "%s/%s", path, candidates[i]);
            if (stat(out, &st) == 0 && S_ISREG(st.st_mode)) return 1;
        }
        fail(error, error_size,
             "no audio VAE safetensors in directory %s", path);
        return 0;
    }
    fail(error, error_size, "audio VAE path is neither file nor directory");
    return 0;
}

static void free_dev(float **p) {
    if (p && *p) {
        cudaFree(*p);
        *p = NULL;
    }
}

static int alloc_dev(float **out, size_t n, char *error, size_t error_size) {
    cudaError_t st;
    if (!out || !n) {
        fail(error, error_size, "invalid device alloc");
        return 0;
    }
    st = cudaMalloc((void **)out, n * sizeof(float));
    if (st != cudaSuccess) {
        fail(error, error_size, "cudaMalloc audio: %s", cudaGetErrorString(st));
        *out = NULL;
        return 0;
    }
    return 1;
}

static int upload_f32(float **d_out, const float *host, size_t n, char *error,
                      size_t error_size) {
    cudaError_t st;
    if (!alloc_dev(d_out, n, error, error_size)) return 0;
    st = cudaMemcpy(*d_out, host, n * sizeof(float), cudaMemcpyHostToDevice);
    if (st != cudaSuccess) {
        fail(error, error_size, "cudaMemcpy H2D: %s", cudaGetErrorString(st));
        free_dev(d_out);
        return 0;
    }
    return 1;
}

static const h3_st_tensor *find_t(const h3_st_header *h, const char *name) {
    return h3_st_find(h, name);
}

static int load_tensor_f32(audio_ctx *ctx, const char *name, size_t expect,
                           float **d_out, char *error, size_t error_size) {
    const h3_st_tensor *t = find_t(&ctx->header, name);
    float *host = NULL;
    size_t n;
    int ok = 0;
    if (!t) {
        fail(error, error_size, "missing tensor %s", name);
        return 0;
    }
    n = (size_t)h3_st_tensor_elements(t);
    if (expect && n != expect) {
        fail(error, error_size, "%s elements %zu != %zu", name, n, expect);
        return 0;
    }
    if (t->dtype != H3_DTYPE_F32) {
        fail(error, error_size, "%s dtype not F32", name);
        return 0;
    }
    host = malloc(n * sizeof(float));
    if (!host) {
        fail(error, error_size, "OOM loading %s", name);
        return 0;
    }
    if (!h3_st_read_data(&ctx->header, t, host, n * sizeof(float), error,
                         error_size)) {
        free(host);
        return 0;
    }
    ok = upload_f32(d_out, host, n, error, error_size);
    free(host);
    return ok;
}

static int load_host_f32(audio_ctx *ctx, const char *name, size_t expect,
                         float *out, char *error, size_t error_size) {
    const h3_st_tensor *t = find_t(&ctx->header, name);
    size_t n;
    if (!t) {
        fail(error, error_size, "missing tensor %s", name);
        return 0;
    }
    n = (size_t)h3_st_tensor_elements(t);
    if (n != expect || t->dtype != H3_DTYPE_F32) {
        fail(error, error_size, "bad tensor %s", name);
        return 0;
    }
    return h3_st_read_data(&ctx->header, t, out, n * sizeof(float), error,
                           error_size);
}

static void free_conv(audio_conv *c) {
    if (!c) return;
    free_dev(&c->d_weight);
    free_dev(&c->d_bias);
    memset(c, 0, sizeof(*c));
}

static void free_act(audio_act *a) {
    if (!a) return;
    free_dev(&a->d_alpha);
    free_dev(&a->d_beta);
    memset(a, 0, sizeof(*a));
}

static void free_stage(audio_stage *s) {
    int b, p, a;
    if (!s) return;
    free_conv(&s->upsample);
    for (b = 0; b < RESBLOCKS; b++) {
        for (a = 0; a < RESIDUAL_PAIRS * 2; a++)
            free_act(&s->blocks[b].acts[a]);
        for (p = 0; p < RESIDUAL_PAIRS; p++) {
            free_conv(&s->blocks[b].convs1[p]);
            free_conv(&s->blocks[b].convs2[p]);
        }
    }
    memset(s, 0, sizeof(*s));
}

static void cleanup(audio_ctx *ctx) {
    if (!ctx) return;
    free_dev(&ctx->d_up_filter);
    free_dev(&ctx->d_down_filter);
    free_dev(&ctx->d_hidden);
    h3_st_free_header(&ctx->header);
    memset(ctx, 0, sizeof(*ctx));
}

static int load_plain_conv(audio_ctx *ctx, audio_conv *c, const char *prefix,
                           uint32_t in_ch, uint32_t out_ch, uint32_t kernel,
                           uint32_t padding, uint32_t dilation, uint32_t stride,
                           int transpose, int has_bias, char *error,
                           size_t error_size) {
    char name[192];
    size_t w_elems;
    c->in_ch = in_ch;
    c->out_ch = out_ch;
    c->kernel = kernel;
    c->padding = padding;
    c->dilation = dilation;
    c->stride = stride;
    c->transpose = transpose;
    c->has_bias = has_bias;
    /* Transpose stores IOK; plain Conv1d stores OIK. */
    if (transpose)
        w_elems = (size_t)in_ch * out_ch * kernel;
    else
        w_elems = (size_t)out_ch * in_ch * kernel;
    snprintf(name, sizeof(name), "%s.weight", prefix);
    if (!load_tensor_f32(ctx, name, w_elems, &c->d_weight, error, error_size))
        return 0;
    if (has_bias) {
        snprintf(name, sizeof(name), "%s.bias", prefix);
        if (!load_tensor_f32(ctx, name, out_ch, &c->d_bias, error, error_size))
            return 0;
    }
    return 1;
}

static int load_act(audio_ctx *ctx, audio_act *a, const char *prefix,
                    uint32_t channels, char *error, size_t error_size) {
    char name[224];
    snprintf(name, sizeof(name), "%s.act.alpha", prefix);
    if (!load_tensor_f32(ctx, name, channels, &a->d_alpha, error, error_size))
        return 0;
    snprintf(name, sizeof(name), "%s.act.beta", prefix);
    return load_tensor_f32(ctx, name, channels, &a->d_beta, error, error_size);
}

static int load_filters(audio_ctx *ctx, char *error, size_t error_size) {
    return load_tensor_f32(ctx, "decoder.activation_post.upsample.filter",
                           FILTER_SIZE, &ctx->d_up_filter, error, error_size) &&
           load_tensor_f32(
               ctx, "decoder.activation_post.downsample.lowpass.filter",
               FILTER_SIZE, &ctx->d_down_filter, error, error_size);
}

static int load_latent_stats(audio_ctx *ctx, char *error, size_t error_size) {
    int i;
    if (!load_host_f32(ctx, "latents_mean", LATENT_CHANNELS, ctx->mean, error,
                       error_size) ||
        !load_host_f32(ctx, "latents_std", LATENT_CHANNELS, ctx->std, error,
                       error_size))
        return 0;
    for (i = 0; i < LATENT_CHANNELS; i++) {
        if (!(ctx->std[i] > 0.0f) || !isfinite(ctx->mean[i]) ||
            !isfinite(ctx->std[i])) {
            fail(error, error_size, "invalid latents_mean/std");
            return 0;
        }
    }
    return 1;
}

static int run_conv(audio_ctx *ctx, float *d_out, const float *d_in,
                    const audio_conv *c, uint32_t length, char *error,
                    size_t error_size) {
    if (c->transpose)
        return h3_cuda_audio_conv_transpose1d_f32(
            ctx->cuda, d_out, d_in, c->d_weight, c->d_bias, STEREO, length,
            c->in_ch, c->out_ch, c->kernel, c->stride, c->padding, error,
            error_size);
    return h3_cuda_audio_conv1d_f32(ctx->cuda, d_out, d_in, c->d_weight,
                                    c->d_bias, STEREO, length, c->in_ch,
                                    c->out_ch, c->kernel, c->padding,
                                    c->dilation, error, error_size);
}

static int run_act(audio_ctx *ctx, float *d_out, const float *d_in,
                   const audio_act *a, uint32_t length, uint32_t channels,
                   char *error, size_t error_size) {
    return h3_cuda_audio_alias_free_snake_f32(
        ctx->cuda, d_out, d_in, a->d_alpha, a->d_beta, ctx->d_up_filter,
        ctx->d_down_filter, STEREO, length, channels, error, error_size);
}

static int prepare_input(audio_ctx *ctx, const float *input, char *error,
                         size_t error_size) {
    size_t elements = (size_t)STEREO * ctx->length * LATENT_CHANNELS;
    float *rows = NULL;
    size_t dest = 0;
    int stereo, ch;
    uint32_t t;
    audio_conv projection = {0}, pre = {0};
    float *d_latent = NULL, *d_projected = NULL, *d_hidden = NULL;
    int ok = 0;

    rows = malloc(elements * sizeof(*rows));
    if (!rows) {
        fail(error, error_size, "OOM audio latent transpose");
        return 0;
    }
    /* Host layout [32,2,T] → activation [2,T,32] with denorm. */
    for (stereo = 0; stereo < STEREO; stereo++)
        for (t = 0; t < ctx->length; t++)
            for (ch = 0; ch < LATENT_CHANNELS; ch++) {
                size_t src =
                    ((size_t)ch * STEREO + (size_t)stereo) * ctx->length + t;
                rows[dest++] =
                    input[src] * ctx->std[ch] + ctx->mean[ch];
            }

    ok = upload_f32(&d_latent, rows, elements, error, error_size);
    free(rows);
    if (!ok) return 0;

    ok = load_plain_conv(ctx, &projection, "dec_in_proj", LATENT_CHANNELS,
                         LATENT_DIM, 1, 0, 1, 1, 0, 1, error, error_size) &&
         load_plain_conv(ctx, &pre, "decoder.conv_pre", LATENT_DIM, DECODER_DIM,
                         7, 3, 1, 1, 0, 1, error, error_size);
    if (!ok) goto done;
    if (!alloc_dev(&d_projected, (size_t)STEREO * ctx->length * LATENT_DIM,
                   error, error_size) ||
        !alloc_dev(&d_hidden, (size_t)STEREO * ctx->length * DECODER_DIM, error,
                   error_size)) {
        ok = 0;
        goto done;
    }
    ok = h3_cuda_audio_conv1d_f32(ctx->cuda, d_projected, d_latent,
                                  projection.d_weight, projection.d_bias,
                                  STEREO, ctx->length, LATENT_CHANNELS,
                                  LATENT_DIM, 1, 0, 1, error, error_size) &&
         h3_cuda_audio_conv1d_f32(ctx->cuda, d_hidden, d_projected, pre.d_weight,
                                  pre.d_bias, STEREO, ctx->length, LATENT_DIM,
                                  DECODER_DIM, 7, 3, 1, error, error_size);
    if (ok) {
        ctx->d_hidden = d_hidden;
        d_hidden = NULL;
    }

done:
    free_dev(&d_latent);
    free_dev(&d_projected);
    free_dev(&d_hidden);
    free_conv(&projection);
    free_conv(&pre);
    return ok;
}

static int load_stage(audio_ctx *ctx, audio_stage *stage, int index,
                      char *error, size_t error_size) {
    uint32_t in_ch = DECODER_DIM >> index;
    uint32_t out_ch = DECODER_DIM >> (index + 1);
    uint32_t rate = h3_audio_vae_upsample_rates[index];
    uint32_t kernel = h3_audio_vae_upsample_kernels[index];
    uint32_t padding = (kernel - rate) / 2;
    char prefix[192];
    int block_index, pair, which;

    snprintf(prefix, sizeof(prefix), "decoder.ups.%d.0", index);
    if (!load_plain_conv(ctx, &stage->upsample, prefix, in_ch, out_ch, kernel,
                         padding, 1, rate, 1, 1, error, error_size))
        return 0;

    for (block_index = 0; block_index < RESBLOCKS; block_index++) {
        int global = index * RESBLOCKS + block_index;
        audio_resblock *block = &stage->blocks[block_index];
        uint32_t residual_kernel = h3_audio_vae_residual_kernels[block_index];
        for (pair = 0; pair < RESIDUAL_PAIRS; pair++) {
            uint32_t dilation = h3_audio_vae_residual_dilations[pair];
            uint32_t dilated_padding = dilation * (residual_kernel - 1) / 2;
            for (which = 0; which < 2; which++) {
                int act = pair * 2 + which;
                snprintf(prefix, sizeof(prefix),
                         "decoder.resblocks.%d.activations.%d", global, act);
                if (!load_act(ctx, &block->acts[act], prefix, out_ch, error,
                              error_size))
                    return 0;
            }
            snprintf(prefix, sizeof(prefix), "decoder.resblocks.%d.convs1.%d",
                     global, pair);
            if (!load_plain_conv(ctx, &block->convs1[pair], prefix, out_ch,
                                 out_ch, residual_kernel, dilated_padding,
                                 dilation, 1, 0, 1, error, error_size))
                return 0;
            snprintf(prefix, sizeof(prefix), "decoder.resblocks.%d.convs2.%d",
                     global, pair);
            if (!load_plain_conv(ctx, &block->convs2[pair], prefix, out_ch,
                                 out_ch, residual_kernel,
                                 (residual_kernel - 1) / 2, 1, 1, 0, 1, error,
                                 error_size))
                return 0;
        }
    }
    return 1;
}

static int run_stage(audio_ctx *ctx, audio_stage *stage, int index,
                     char *error, size_t error_size) {
    uint32_t input_length = ctx->length;
    uint32_t channels = DECODER_DIM >> (index + 1);
    uint32_t kernel = h3_audio_vae_upsample_kernels[index];
    uint32_t stride = h3_audio_vae_upsample_rates[index];
    uint32_t padding = (kernel - stride) / 2;
    uint64_t expanded =
        (uint64_t)(input_length - 1) * stride + kernel - 2ull * padding;
    uint32_t output_length;
    size_t elements;
    float *d_up = NULL, *d_sum = NULL, *d_work = NULL, *d_act = NULL,
          *d_branch = NULL;
    int ok = 0;
    int block_index, pair;

    if (expanded == 0 || expanded > UINT32_MAX) {
        fail(error, error_size, "AudioVAE CUDA stage length overflow");
        return 0;
    }
    output_length = (uint32_t)expanded;
    elements = (size_t)STEREO * output_length * channels;
    if (!alloc_dev(&d_up, elements, error, error_size) ||
        !alloc_dev(&d_sum, elements, error, error_size) ||
        !alloc_dev(&d_work, elements, error, error_size) ||
        !alloc_dev(&d_act, elements, error, error_size) ||
        !alloc_dev(&d_branch, elements, error, error_size))
        goto done;

    ok = run_conv(ctx, d_up, ctx->d_hidden, &stage->upsample, input_length,
                  error, error_size);
    for (block_index = 0; ok && block_index < RESBLOCKS; block_index++) {
        audio_resblock *block = &stage->blocks[block_index];
        float *target = block_index == 0 ? d_sum : d_work;
        ok = h3_cuda_audio_copy_f32(ctx->cuda, target, d_up, elements, error,
                                    error_size);
        for (pair = 0; ok && pair < RESIDUAL_PAIRS; pair++) {
            ok = run_act(ctx, d_act, target, &block->acts[pair * 2],
                         output_length, channels, error, error_size) &&
                 run_conv(ctx, d_branch, d_act, &block->convs1[pair],
                          output_length, error, error_size) &&
                 run_act(ctx, d_act, d_branch, &block->acts[pair * 2 + 1],
                         output_length, channels, error, error_size) &&
                 run_conv(ctx, d_branch, d_act, &block->convs2[pair],
                          output_length, error, error_size) &&
                 h3_cuda_audio_add_scaled_f32(ctx->cuda, target, target,
                                              d_branch, 1.0f, 1.0f,
                                              (uint32_t)elements, error,
                                              error_size);
        }
        if (ok && block_index == 1)
            ok = h3_cuda_audio_add_scaled_f32(ctx->cuda, d_sum, d_sum, d_work,
                                              1.0f, 1.0f, (uint32_t)elements,
                                              error, error_size);
        if (ok && block_index == 2)
            ok = h3_cuda_audio_add_scaled_f32(
                ctx->cuda, d_sum, d_sum, d_work, 1.0f / 3.0f, 1.0f / 3.0f,
                (uint32_t)elements, error, error_size);
    }
    if (ok) {
        free_dev(&ctx->d_hidden);
        ctx->d_hidden = d_sum;
        d_sum = NULL;
        ctx->length = output_length;
    }

done:
    free_dev(&d_up);
    free_dev(&d_sum);
    free_dev(&d_work);
    free_dev(&d_act);
    free_dev(&d_branch);
    return ok;
}

static int decode_output(audio_ctx *ctx, h3_audio_vae_cuda_waveform *output,
                         char *error, size_t error_size) {
    audio_act act = {0};
    audio_conv post = {0};
    float *d_act = NULL, *d_wave = NULL, *host = NULL;
    size_t hidden_elems = (size_t)STEREO * ctx->length * 8;
    size_t wave_elems = (size_t)STEREO * ctx->length;
    int ok = 0;
    cudaError_t st;

    ok = load_act(ctx, &act, "decoder.activation_post", 8, error, error_size) &&
         load_plain_conv(ctx, &post, "decoder.conv_post", 8, 1, 7, 3, 1, 1, 0,
                         0, error, error_size);
    if (!ok) goto done;
    if (!alloc_dev(&d_act, hidden_elems, error, error_size) ||
        !alloc_dev(&d_wave, wave_elems, error, error_size)) {
        ok = 0;
        goto done;
    }
    ok = run_act(ctx, d_act, ctx->d_hidden, &act, ctx->length, 8, error,
                 error_size) &&
         run_conv(ctx, d_wave, d_act, &post, ctx->length, error, error_size) &&
         h3_cuda_audio_clip_f32(ctx->cuda, d_wave, d_wave, (uint32_t)wave_elems,
                                -1.0f, 1.0f, error, error_size);
    if (!ok) goto done;
    /* Device layout [2,T,1] → host channel-major [2,T]. */
    host = malloc(wave_elems * sizeof(*host));
    if (!host) {
        fail(error, error_size, "OOM waveform host");
        ok = 0;
        goto done;
    }
    st = cudaMemcpy(host, d_wave, wave_elems * sizeof(float),
                    cudaMemcpyDeviceToHost);
    if (st != cudaSuccess) {
        fail(error, error_size, "waveform D2H: %s", cudaGetErrorString(st));
        free(host);
        ok = 0;
        goto done;
    }
    /* Flatten [batch=stereo, length, ch=1] already contiguous as [2,T]. */
    output->pcm = host;
    output->channels = STEREO;
    output->samples = (int)ctx->length;
    output->sample_rate = SAMPLE_RATE;
    host = NULL;
    ok = 1;

done:
    free(host);
    free_dev(&d_act);
    free_dev(&d_wave);
    free_act(&act);
    free_conv(&post);
    return ok;
}

void h3_audio_vae_cuda_waveform_free(h3_audio_vae_cuda_waveform *waveform) {
    if (!waveform) return;
    free(waveform->pcm);
    memset(waveform, 0, sizeof(*waveform));
}

void h3_audio_vae_cuda_latent_free(h3_audio_vae_cuda_latent *latent) {
    if (!latent) return;
    free(latent->values);
    memset(latent, 0, sizeof(*latent));
}

enum {
    ENCODER_STAGES = H3_AUDIO_VAE_ENCODER_STAGES,
    ENCODER_RESIDUALS = H3_AUDIO_VAE_ENCODER_RESIDUALS,
    ENCODER_HEADS = H3_AUDIO_VAE_ENCODER_HEADS,
    ENCODER_DIM = H3_AUDIO_VAE_ENCODER_DIM
};

static int encoder_conv_length(uint32_t length, uint32_t kernel,
                               uint32_t stride, uint32_t padding,
                               uint32_t dilation, uint32_t *result) {
    uint64_t effective = (uint64_t)dilation * (kernel - 1) + 1;
    uint64_t padded = (uint64_t)length + 2 * padding;
    if (!result || !stride || padded < effective) return 0;
    uint64_t output = (padded - effective) / stride + 1;
    if (!output || output > UINT32_MAX) return 0;
    *result = (uint32_t)output;
    return 1;
}

static int load_alpha(audio_ctx *ctx, const char *name, uint32_t channels,
                      float **d_out, char *error, size_t error_size) {
    return load_tensor_f32(ctx, name, channels, d_out, error, error_size);
}

static int encoder_initial(audio_ctx *ctx, const float *pcm, int samples,
                           char *error, size_t error_size) {
    uint32_t padded =
        (uint32_t)(((uint64_t)(uint32_t)samples + HOP_LENGTH - 1) / HOP_LENGTH *
                   HOP_LENGTH);
    size_t input_elements = (size_t)STEREO * padded;
    float *input_rows = calloc(input_elements, sizeof(*input_rows));
    audio_conv conv = {0};
    float *d_in = NULL, *d_hidden = NULL;
    int ok = 0;
    int ch;
    if (!input_rows) {
        fail(error, error_size, "OOM padding reference audio");
        return 0;
    }
    for (ch = 0; ch < STEREO; ch++)
        memcpy(input_rows + (size_t)ch * padded, pcm + (size_t)ch * samples,
               (size_t)samples * sizeof(*pcm));
    ok = upload_f32(&d_in, input_rows, input_elements, error, error_size);
    free(input_rows);
    if (!ok) return 0;
    ok = load_plain_conv(ctx, &conv, "encoder.block.0", 1, ENCODER_DIM, 7, 3, 1,
                         1, 0, 1, error, error_size) &&
         alloc_dev(&d_hidden, (size_t)STEREO * padded * ENCODER_DIM, error,
                   error_size) &&
         h3_cuda_audio_conv1d_f32(ctx->cuda, d_hidden, d_in, conv.d_weight,
                                  conv.d_bias, STEREO, padded, 1, ENCODER_DIM,
                                  7, 3, 1, error, error_size);
    if (ok) {
        ctx->d_hidden = d_hidden;
        d_hidden = NULL;
        ctx->length = padded;
    }
    free_dev(&d_in);
    free_dev(&d_hidden);
    free_conv(&conv);
    return ok;
}

static int encoder_residual(audio_ctx *ctx, int stage, int residual,
                            uint32_t channels, char *error,
                            size_t error_size) {
    char name[224];
    float *d_a1 = NULL, *d_a2 = NULL, *d_act = NULL, *d_branch = NULL;
    audio_conv c1 = {0}, c2 = {0};
    uint32_t dilation = h3_audio_vae_encoder_dilations[residual];
    size_t elements = (size_t)STEREO * ctx->length * channels;
    int ok;
    snprintf(name, sizeof(name), "encoder.block.%d.block.%d.block.0.alpha",
             stage, residual);
    ok = load_alpha(ctx, name, channels, &d_a1, error, error_size);
    snprintf(name, sizeof(name), "encoder.block.%d.block.%d.block.1", stage,
             residual);
    if (ok)
        ok = load_plain_conv(ctx, &c1, name, channels, channels, 7,
                             3 * dilation, dilation, 1, 0, 1, error,
                             error_size);
    snprintf(name, sizeof(name), "encoder.block.%d.block.%d.block.2.alpha",
             stage, residual);
    if (ok) ok = load_alpha(ctx, name, channels, &d_a2, error, error_size);
    snprintf(name, sizeof(name), "encoder.block.%d.block.%d.block.3", stage,
             residual);
    if (ok)
        ok = load_plain_conv(ctx, &c2, name, channels, channels, 1, 0, 1, 1, 0,
                             1, error, error_size);
    if (ok)
        ok = alloc_dev(&d_act, elements, error, error_size) &&
             alloc_dev(&d_branch, elements, error, error_size);
    if (ok)
        ok = h3_cuda_audio_snake1d_f32(ctx->cuda, d_act, ctx->d_hidden, d_a1,
                                       STEREO, ctx->length, channels, error,
                                       error_size) &&
             h3_cuda_audio_conv1d_f32(ctx->cuda, d_branch, d_act, c1.d_weight,
                                      c1.d_bias, STEREO, ctx->length, channels,
                                      channels, 7, 3 * dilation, dilation,
                                      error, error_size) &&
             h3_cuda_audio_snake1d_f32(ctx->cuda, d_act, d_branch, d_a2, STEREO,
                                       ctx->length, channels, error,
                                       error_size) &&
             h3_cuda_audio_conv1d_f32(ctx->cuda, d_branch, d_act, c2.d_weight,
                                      c2.d_bias, STEREO, ctx->length, channels,
                                      channels, 1, 0, 1, error, error_size) &&
             h3_cuda_audio_add_scaled_f32(ctx->cuda, ctx->d_hidden, ctx->d_hidden,
                                          d_branch, 1.0f, 1.0f,
                                          (uint32_t)elements, error,
                                          error_size);
    free_dev(&d_a1);
    free_dev(&d_a2);
    free_dev(&d_act);
    free_dev(&d_branch);
    free_conv(&c1);
    free_conv(&c2);
    return ok;
}

static int encoder_downsample(audio_ctx *ctx, int stage, uint32_t channels,
                              char *error, size_t error_size) {
    char name[192];
    float *d_alpha = NULL, *d_act = NULL, *d_down = NULL;
    audio_conv conv = {0};
    uint32_t stride = h3_audio_vae_encoder_strides[stage - 1];
    uint32_t kernel = stride * 2;
    uint32_t padding = (stride + 1) / 2;
    uint32_t out_len = 0;
    size_t in_elems = (size_t)STEREO * ctx->length * channels;
    size_t out_elems;
    int ok;
    snprintf(name, sizeof(name), "encoder.block.%d.block.3.alpha", stage);
    ok = load_alpha(ctx, name, channels, &d_alpha, error, error_size);
    snprintf(name, sizeof(name), "encoder.block.%d.block.4", stage);
    if (ok)
        ok = load_plain_conv(ctx, &conv, name, channels, channels * 2, kernel,
                             padding, 1, stride, 0, 1, error, error_size) &&
             encoder_conv_length(ctx->length, kernel, stride, padding, 1,
                                 &out_len);
    out_elems = (size_t)STEREO * out_len * channels * 2;
    if (ok)
        ok = alloc_dev(&d_act, in_elems, error, error_size) &&
             alloc_dev(&d_down, out_elems, error, error_size);
    if (ok)
        ok = h3_cuda_audio_snake1d_f32(ctx->cuda, d_act, ctx->d_hidden, d_alpha,
                                       STEREO, ctx->length, channels, error,
                                       error_size) &&
             h3_cuda_audio_conv1d_stride_f32(
                 ctx->cuda, d_down, d_act, conv.d_weight, conv.d_bias, STEREO,
                 ctx->length, out_len, channels, channels * 2, kernel, stride,
                 padding, 1, error, error_size);
    if (ok) {
        free_dev(&ctx->d_hidden);
        ctx->d_hidden = d_down;
        d_down = NULL;
        ctx->length = out_len;
    }
    free_dev(&d_alpha);
    free_dev(&d_act);
    free_dev(&d_down);
    free_conv(&conv);
    return ok;
}

static int encoder_final_conv(audio_ctx *ctx, char *error, size_t error_size) {
    float *d_alpha = NULL, *d_act = NULL, *d_hidden = NULL;
    audio_conv conv = {0};
    size_t elements = (size_t)STEREO * ctx->length * LATENT_DIM;
    int ok = load_alpha(ctx, "encoder.block.6.alpha", LATENT_DIM, &d_alpha,
                        error, error_size) &&
             load_plain_conv(ctx, &conv, "encoder.block.7", LATENT_DIM,
                             LATENT_DIM, 3, 1, 1, 1, 0, 1, error, error_size) &&
             alloc_dev(&d_act, elements, error, error_size) &&
             alloc_dev(&d_hidden, elements, error, error_size);
    if (ok)
        ok = h3_cuda_audio_snake1d_f32(ctx->cuda, d_act, ctx->d_hidden, d_alpha,
                                       STEREO, ctx->length, LATENT_DIM, error,
                                       error_size) &&
             h3_cuda_audio_conv1d_f32(ctx->cuda, d_hidden, d_act, conv.d_weight,
                                      conv.d_bias, STEREO, ctx->length,
                                      LATENT_DIM, LATENT_DIM, 3, 1, 1, error,
                                      error_size);
    if (ok) {
        free_dev(&ctx->d_hidden);
        ctx->d_hidden = d_hidden;
        d_hidden = NULL;
    }
    free_dev(&d_alpha);
    free_dev(&d_act);
    free_dev(&d_hidden);
    free_conv(&conv);
    return ok;
}

static int load_vec(audio_ctx *ctx, const char *name, uint32_t n, float **d_out,
                    char *error, size_t error_size) {
    return load_tensor_f32(ctx, name, n, d_out, error, error_size);
}

static int load_mat(audio_ctx *ctx, const char *name, uint32_t rows,
                    uint32_t cols, float **d_out, char *error,
                    size_t error_size) {
    return load_tensor_f32(ctx, name, (size_t)rows * cols, d_out, error,
                           error_size);
}

static int encoder_projection_branch(audio_ctx *ctx, float **d_base,
                                     char *error, size_t error_size) {
    float *nw = NULL, *nb = NULL, *pw = NULL, *pb = NULL;
    float *d_norm = NULL, *d_proj = NULL;
    uint32_t rows = STEREO * ctx->length;
    int ok =
        load_vec(ctx, "pre_block.norm3.weight", LATENT_DIM, &nw, error,
                 error_size) &&
        load_vec(ctx, "pre_block.norm3.bias", LATENT_DIM, &nb, error,
                 error_size) &&
        load_mat(ctx, "pre_block.proj.weight", LATENT_CHANNELS, LATENT_DIM, &pw,
                 error, error_size) &&
        load_vec(ctx, "pre_block.proj.bias", LATENT_CHANNELS, &pb, error,
                 error_size) &&
        alloc_dev(&d_norm, (size_t)rows * LATENT_DIM, error, error_size) &&
        alloc_dev(&d_proj, (size_t)rows * LATENT_CHANNELS, error, error_size);
    if (ok)
        ok = h3_cuda_audio_layer_norm_f32(ctx->cuda, d_norm, ctx->d_hidden, nw,
                                          nb, rows, LATENT_DIM, 1e-5f, error,
                                          error_size) &&
             h3_cuda_audio_linear_f32(ctx->cuda, d_proj, d_norm, pw, pb, rows,
                                      LATENT_DIM, LATENT_CHANNELS, error,
                                      error_size);
    if (ok) {
        *d_base = d_proj;
        d_proj = NULL;
    }
    free_dev(&nw);
    free_dev(&nb);
    free_dev(&pw);
    free_dev(&pb);
    free_dev(&d_norm);
    free_dev(&d_proj);
    return ok;
}

static int encoder_attention_branch(audio_ctx *ctx, float *d_base, char *error,
                                    size_t error_size) {
    float *nw = NULL, *nb = NULL, *qkv_w = NULL;
    float *qb = NULL, *kb = NULL, *vb = NULL, *pw = NULL, *pb = NULL;
    float *d_norm = NULL, *d_qkv = NULL, *d_q = NULL, *d_k = NULL, *d_v = NULL;
    float *d_attn = NULL, *d_pool = NULL, *d_proj = NULL;
    uint32_t rows = STEREO * ctx->length;
    uint32_t head_dim = LATENT_DIM / ENCODER_HEADS;
    size_t attn_elems = (size_t)rows * LATENT_DIM;
    int ok =
        load_vec(ctx, "pre_block.norm1.weight", LATENT_DIM, &nw, error,
                 error_size) &&
        load_vec(ctx, "pre_block.norm1.bias", LATENT_DIM, &nb, error,
                 error_size) &&
        load_mat(ctx, "pre_block.attn.qkv.weight", LATENT_DIM * 3, LATENT_DIM,
                 &qkv_w, error, error_size) &&
        load_vec(ctx, "pre_block.attn.q_bias", LATENT_DIM, &qb, error,
                 error_size) &&
        load_vec(ctx, "pre_block.attn.zero_k_bias", LATENT_DIM, &kb, error,
                 error_size) &&
        load_vec(ctx, "pre_block.attn.v_bias", LATENT_DIM, &vb, error,
                 error_size) &&
        load_mat(ctx, "pre_block.attn.proj.weight", LATENT_CHANNELS,
                 LATENT_CHANNELS, &pw, error, error_size) &&
        load_vec(ctx, "pre_block.attn.proj.bias", LATENT_CHANNELS, &pb, error,
                 error_size);
    if (ok)
        ok = alloc_dev(&d_norm, attn_elems, error, error_size) &&
             alloc_dev(&d_qkv, attn_elems * 3, error, error_size) &&
             alloc_dev(&d_q, attn_elems, error, error_size) &&
             alloc_dev(&d_k, attn_elems, error, error_size) &&
             alloc_dev(&d_v, attn_elems, error, error_size) &&
             alloc_dev(&d_attn, attn_elems, error, error_size) &&
             alloc_dev(&d_pool, (size_t)rows * LATENT_CHANNELS, error,
                       error_size) &&
             alloc_dev(&d_proj, (size_t)rows * LATENT_CHANNELS, error,
                       error_size);
    if (ok)
        ok = h3_cuda_audio_layer_norm_f32(ctx->cuda, d_norm, ctx->d_hidden, nw,
                                          nb, rows, LATENT_DIM, 1e-5f, error,
                                          error_size) &&
             h3_cuda_audio_linear_f32(ctx->cuda, d_qkv, d_norm, qkv_w, NULL,
                                      rows, LATENT_DIM, LATENT_DIM * 3, error,
                                      error_size) &&
             h3_cuda_audio_qkv_split_f32(ctx->cuda, d_q, d_k, d_v, d_qkv, qb, kb,
                                         vb, STEREO, ctx->length, ENCODER_HEADS,
                                         head_dim, error, error_size) &&
             h3_cuda_audio_sdpa_causal_f32(
                 ctx->cuda, d_attn, d_q, d_k, d_v, STEREO, ctx->length,
                 ENCODER_HEADS, head_dim,
                 1.0f / sqrtf((float)head_dim), error, error_size) &&
             h3_cuda_audio_attention_pool_f32(
                 ctx->cuda, d_pool, d_attn, STEREO, ctx->length, ENCODER_HEADS,
                 head_dim, LATENT_CHANNELS, error, error_size) &&
             h3_cuda_audio_linear_f32(ctx->cuda, d_proj, d_pool, pw, pb, rows,
                                      LATENT_CHANNELS, LATENT_CHANNELS, error,
                                      error_size) &&
             h3_cuda_audio_add_scaled_f32(ctx->cuda, d_base, d_base, d_proj,
                                          1.0f, 1.0f, rows * LATENT_CHANNELS,
                                          error, error_size);
    free_dev(&nw);
    free_dev(&nb);
    free_dev(&qkv_w);
    free_dev(&qb);
    free_dev(&kb);
    free_dev(&vb);
    free_dev(&pw);
    free_dev(&pb);
    free_dev(&d_norm);
    free_dev(&d_qkv);
    free_dev(&d_q);
    free_dev(&d_k);
    free_dev(&d_v);
    free_dev(&d_attn);
    free_dev(&d_pool);
    free_dev(&d_proj);
    return ok;
}

static int encoder_mlp(audio_ctx *ctx, float *d_base, char *error,
                       size_t error_size) {
    float *n2w = NULL, *n2b = NULL, *nmw = NULL, *nmb = NULL;
    float *w0 = NULL, *b0 = NULL, *w1 = NULL, *b1 = NULL, *w2 = NULL, *b2 = NULL;
    float *d_n2 = NULL, *d_nm = NULL, *d_gate = NULL, *d_lin = NULL, *d_geglu = NULL,
          *d_branch = NULL;
    uint32_t rows = STEREO * ctx->length;
    int ok =
        load_vec(ctx, "pre_block.norm2.weight", LATENT_CHANNELS, &n2w, error,
                 error_size) &&
        load_vec(ctx, "pre_block.norm2.bias", LATENT_CHANNELS, &n2b, error,
                 error_size) &&
        load_vec(ctx, "pre_block.mlp.norm.weight", LATENT_CHANNELS, &nmw, error,
                 error_size) &&
        load_vec(ctx, "pre_block.mlp.norm.bias", LATENT_CHANNELS, &nmb, error,
                 error_size) &&
        load_mat(ctx, "pre_block.mlp.w0.weight", LATENT_CHANNELS * 2,
                 LATENT_CHANNELS, &w0, error, error_size) &&
        load_vec(ctx, "pre_block.mlp.w0.bias", LATENT_CHANNELS * 2, &b0, error,
                 error_size) &&
        load_mat(ctx, "pre_block.mlp.w1.weight", LATENT_CHANNELS * 2,
                 LATENT_CHANNELS, &w1, error, error_size) &&
        load_vec(ctx, "pre_block.mlp.w1.bias", LATENT_CHANNELS * 2, &b1, error,
                 error_size) &&
        load_mat(ctx, "pre_block.mlp.w2.weight", LATENT_CHANNELS,
                 LATENT_CHANNELS * 2, &w2, error, error_size) &&
        load_vec(ctx, "pre_block.mlp.w2.bias", LATENT_CHANNELS, &b2, error,
                 error_size);
    if (ok)
        ok = alloc_dev(&d_n2, (size_t)rows * LATENT_CHANNELS, error,
                       error_size) &&
             alloc_dev(&d_nm, (size_t)rows * LATENT_CHANNELS, error,
                       error_size) &&
             alloc_dev(&d_gate, (size_t)rows * LATENT_CHANNELS * 2, error,
                       error_size) &&
             alloc_dev(&d_lin, (size_t)rows * LATENT_CHANNELS * 2, error,
                       error_size) &&
             alloc_dev(&d_geglu, (size_t)rows * LATENT_CHANNELS * 2, error,
                       error_size) &&
             alloc_dev(&d_branch, (size_t)rows * LATENT_CHANNELS, error,
                       error_size);
    if (ok)
        ok = h3_cuda_audio_layer_norm_f32(ctx->cuda, d_n2, d_base, n2w, n2b,
                                          rows, LATENT_CHANNELS, 1e-5f, error,
                                          error_size) &&
             h3_cuda_audio_layer_norm_f32(ctx->cuda, d_nm, d_n2, nmw, nmb, rows,
                                          LATENT_CHANNELS, 1e-5f, error,
                                          error_size) &&
             h3_cuda_audio_linear_f32(ctx->cuda, d_gate, d_nm, w0, b0, rows,
                                      LATENT_CHANNELS, LATENT_CHANNELS * 2,
                                      error, error_size) &&
             h3_cuda_audio_linear_f32(ctx->cuda, d_lin, d_nm, w1, b1, rows,
                                      LATENT_CHANNELS, LATENT_CHANNELS * 2,
                                      error, error_size) &&
             h3_cuda_audio_geglu_f32(ctx->cuda, d_geglu, d_gate, d_lin,
                                     rows * LATENT_CHANNELS * 2, error,
                                     error_size) &&
             h3_cuda_audio_linear_f32(ctx->cuda, d_branch, d_geglu, w2, b2, rows,
                                      LATENT_CHANNELS * 2, LATENT_CHANNELS,
                                      error, error_size) &&
             h3_cuda_audio_add_scaled_f32(ctx->cuda, d_base, d_base, d_branch,
                                          1.0f, 1.0f, rows * LATENT_CHANNELS,
                                          error, error_size);
    free_dev(&n2w);
    free_dev(&n2b);
    free_dev(&nmw);
    free_dev(&nmb);
    free_dev(&w0);
    free_dev(&b0);
    free_dev(&w1);
    free_dev(&b1);
    free_dev(&w2);
    free_dev(&b2);
    free_dev(&d_n2);
    free_dev(&d_nm);
    free_dev(&d_gate);
    free_dev(&d_lin);
    free_dev(&d_geglu);
    free_dev(&d_branch);
    return ok;
}

static int encoder_output(audio_ctx *ctx, float *d_base, const float *mean,
                          const float *deviation,
                          h3_audio_vae_cuda_latent *output, char *error,
                          size_t error_size) {
    audio_conv conv = {0};
    float *d_lat = NULL, *rows = NULL;
    size_t elements = (size_t)STEREO * ctx->length * LATENT_CHANNELS;
    int ok = load_plain_conv(ctx, &conv, "mean_proj", LATENT_CHANNELS,
                             LATENT_CHANNELS, 1, 0, 1, 1, 0, 1, error,
                             error_size) &&
             alloc_dev(&d_lat, elements, error, error_size);
    if (ok)
        ok = h3_cuda_audio_conv1d_f32(ctx->cuda, d_lat, d_base, conv.d_weight,
                                      conv.d_bias, STEREO, ctx->length,
                                      LATENT_CHANNELS, LATENT_CHANNELS, 1, 0, 1,
                                      error, error_size);
    if (ok) {
        rows = malloc(elements * sizeof(*rows));
        output->values = malloc(elements * sizeof(*output->values));
        if (!rows || !output->values) {
            fail(error, error_size, "OOM encoder output");
            ok = 0;
        } else {
            cudaError_t st = cudaMemcpy(rows, d_lat, elements * sizeof(float),
                                        cudaMemcpyDeviceToHost);
            if (st != cudaSuccess) {
                fail(error, error_size, "encoder D2H: %s",
                     cudaGetErrorString(st));
                ok = 0;
            }
        }
    }
    if (ok) {
        int ch, stereo;
        uint32_t t;
        for (ch = 0; ch < LATENT_CHANNELS; ch++)
            for (stereo = 0; stereo < STEREO; stereo++)
                for (t = 0; t < ctx->length; t++) {
                    size_t src =
                        ((size_t)stereo * ctx->length + t) * LATENT_CHANNELS +
                        (size_t)ch;
                    size_t dst = ((size_t)ch * STEREO + (size_t)stereo) *
                                     ctx->length +
                                 t;
                    output->values[dst] =
                        (rows[src] - mean[ch]) / deviation[ch];
                }
        output->channels = LATENT_CHANNELS;
        output->stereo = STEREO;
        output->length = (int)ctx->length;
    }
    free(rows);
    free_dev(&d_lat);
    free_conv(&conv);
    return ok;
}

int h3_audio_vae_cuda_encode(h3_cuda *cuda, const char *weights_path,
                             const float *pcm, int samples,
                             h3_audio_vae_cuda_progress progress,
                             void *progress_opaque,
                             h3_audio_vae_cuda_latent *output,
                             h3_audio_vae_cuda_stats *stats, char *error,
                             size_t error_size) {
    audio_ctx ctx = {0};
    char file[1024];
    double t0, t_load0, t_enc0;
    float *d_base = NULL;
    uint32_t channels;
    int stage, residual, ok = 0;
    uint32_t expect_t;

    if (error && error_size) error[0] = '\0';
    if (output) memset(output, 0, sizeof(*output));
    if (stats) memset(stats, 0, sizeof(*stats));
    if (!cuda || !weights_path || !pcm || !output || samples < 1 ||
        samples > SAMPLE_RATE * 15) {
        fail(error, error_size, "invalid AudioVAE CUDA encode args");
        return 0;
    }

    t0 = now_ms();
    ctx.cuda = cuda;
    t_load0 = now_ms();
    if (!resolve_weights_file(weights_path, file, sizeof(file), error,
                              error_size) ||
        !h3_st_read_header(file, &ctx.header, error, error_size) ||
        !load_latent_stats(&ctx, error, error_size))
        goto done;
    if (stats) stats->load_ms = now_ms() - t_load0;

    t_enc0 = now_ms();
    ok = encoder_initial(&ctx, pcm, samples, error, error_size);
    channels = ENCODER_DIM;
    for (stage = 1; ok && stage <= ENCODER_STAGES; stage++) {
        for (residual = 0; ok && residual < ENCODER_RESIDUALS; residual++)
            ok = encoder_residual(&ctx, stage, residual, channels, error,
                                  error_size);
        if (ok) ok = encoder_downsample(&ctx, stage, channels, error, error_size);
        channels *= 2;
        if (ok && progress)
            progress(stage, ENCODER_STAGES + 2, progress_opaque);
    }
    expect_t = (uint32_t)((samples + HOP_LENGTH - 1) / HOP_LENGTH);
    if (ok && (channels != LATENT_DIM || ctx.length != expect_t)) {
        fail(error, error_size,
             "AudioVAE CUDA encoder geometry mismatch (ch=%u len=%u expect_t=%u)",
             channels, ctx.length, expect_t);
        ok = 0;
    }
    if (ok) ok = encoder_final_conv(&ctx, error, error_size);
    if (ok && progress)
        progress(ENCODER_STAGES + 1, ENCODER_STAGES + 2, progress_opaque);
    if (ok)
        ok = encoder_projection_branch(&ctx, &d_base, error, error_size) &&
             encoder_attention_branch(&ctx, d_base, error, error_size) &&
             encoder_mlp(&ctx, d_base, error, error_size) &&
             encoder_output(&ctx, d_base, ctx.mean, ctx.std, output, error,
                            error_size);
    if (ok) {
        cudaError_t st = cudaDeviceSynchronize();
        if (st != cudaSuccess) {
            fail(error, error_size, "cudaDeviceSynchronize: %s",
                 cudaGetErrorString(st));
            ok = 0;
        }
    }
    if (ok && progress)
        progress(ENCODER_STAGES + 2, ENCODER_STAGES + 2, progress_opaque);
    if (stats) {
        stats->encode_ms = now_ms() - t_enc0;
        stats->total_ms = now_ms() - t0;
        stats->latent_length = ok ? output->length : 0;
        stats->pcm_samples = samples;
    }

done:
    free_dev(&d_base);
    if (!ok) h3_audio_vae_cuda_latent_free(output);
    cleanup(&ctx);
    return ok;
}

int h3_audio_vae_cuda_decode(h3_cuda *cuda, const char *weights_path,
                             const float *normalized_latent, int latent_length,
                             h3_audio_vae_cuda_progress progress,
                             void *progress_opaque,
                             h3_audio_vae_cuda_waveform *output,
                             h3_audio_vae_cuda_stats *stats, char *error,
                             size_t error_size) {
    audio_ctx ctx = {0};
    char file[1024];
    double t0, t_load0, t_dec0;
    int index, ok = 0;

    if (error && error_size) error[0] = '\0';
    if (output) memset(output, 0, sizeof(*output));
    if (stats) memset(stats, 0, sizeof(*stats));
    if (!cuda || !weights_path || !normalized_latent || !output ||
        latent_length < 1 ||
        latent_length > INT32_MAX / HOP_LENGTH) {
        fail(error, error_size, "invalid AudioVAE CUDA decode args");
        return 0;
    }

    t0 = now_ms();
    ctx.cuda = cuda;
    ctx.length = (uint32_t)latent_length;

    t_load0 = now_ms();
    if (!resolve_weights_file(weights_path, file, sizeof(file), error,
                              error_size) ||
        !h3_st_read_header(file, &ctx.header, error, error_size) ||
        !load_latent_stats(&ctx, error, error_size) ||
        !load_filters(&ctx, error, error_size))
        goto done;
    if (stats) stats->load_ms = now_ms() - t_load0;

    t_dec0 = now_ms();
    ok = prepare_input(&ctx, normalized_latent, error, error_size);
    for (index = 0; ok && index < STAGES; index++) {
        audio_stage stage = {0};
        ok = load_stage(&ctx, &stage, index, error, error_size) &&
             run_stage(&ctx, &stage, index, error, error_size);
        free_stage(&stage);
        if (ok && progress) progress(index + 1, STAGES, progress_opaque);
    }
    if (ok) ok = ctx.length == (uint32_t)latent_length * HOP_LENGTH;
    if (!ok && error && error_size && !error[0])
        fail(error, error_size, "AudioVAE CUDA wrong sample count (%u)",
             ctx.length);
    if (ok) ok = decode_output(&ctx, output, error, error_size);
    if (ok) {
        cudaError_t st = cudaDeviceSynchronize();
        if (st != cudaSuccess) {
            fail(error, error_size, "cudaDeviceSynchronize: %s",
                 cudaGetErrorString(st));
            ok = 0;
        }
    }
    if (stats) {
        stats->decode_ms = now_ms() - t_dec0;
        stats->total_ms = now_ms() - t0;
        stats->latent_length = latent_length;
        stats->pcm_samples = ok ? output->samples : 0;
    }

done:
    if (!ok) h3_audio_vae_cuda_waveform_free(output);
    cleanup(&ctx);
    return ok;
}
