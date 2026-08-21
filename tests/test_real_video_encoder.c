#include "h3_safetensors.h"
#include "h3_video_encoder.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_real_video_encoder.c: %s\n", message);
    exit(1);
}

static int latent_t_from_frames(int frames) {
    return frames > 0 ? (frames + 3) / 4 : 0;
}

static int axis_tiles(int extent) {
    if (extent <= 256) return 1;
    int count = (extent + 255) / 256;
    while (256 * count - 64 * (count - 1) < extent) count++;
    return count;
}

static void progress(int completed, int total, void *opaque) {
    (void)opaque;
    fprintf(stderr, "native visual encoder tile: %d/%d\n", completed, total);
}

static int keep_encoder_tensor(const char *name) {
    if (!name) return 0;
    if (!strncmp(name, "encoder.", 8)) return 1;
    if (!strncmp(name, "quant_conv.", 11)) return 1;
    return !strcmp(name, "latents_mean") || !strcmp(name, "latents_std");
}

static int mkdir_p(const char *path) {
    if (mkdir(path, 0755) == 0 || errno == EEXIST) return 1;
    return 0;
}

static int file_nonempty(const char *path) {
    struct stat st;
    return stat(path, &st) == 0 && st.st_size > 0;
}

static int write_latents_config(const char *path, const float *mean,
                                const float *std) {
    FILE *file = fopen(path, "wb");
    int i;
    if (!file) return 0;
    fputs("{\n  \"latents_mean\": [", file);
    for (i = 0; i < 24; i++)
        fprintf(file, "%s%.9g", i ? ", " : "", (double)mean[i]);
    fputs("],\n  \"latents_std\": [", file);
    for (i = 0; i < 24; i++)
        fprintf(file, "%s%.9g", i ? ", " : "", (double)std[i]);
    fputs("]\n}\n", file);
    fclose(file);
    return 1;
}

/* Metal h3_weight_load_f32 only accepts F32 shards. Official MiniMax-H3
 * video_vae/source is not in this worktree; upcast the same fp16 encoder
 * tensors CUDA uses so Darwin can run h3_video_vae_encode / encode_tile. */
static int materialize_f32_encoder_store(const char *fp16_path,
                                         const char *store_dir,
                                         char *error, size_t error_size) {
    char source_dir[1024], out_path[1024], cfg_path[1024];
    h3_st_header header;
    FILE *out = NULL;
    char *json = NULL;
    size_t json_cap = 0, json_len = 0;
    uint64_t data_off = 0, header_size;
    size_t i, n_keep = 0;
    float mean[24], stdv[24];
    int have_mean = 0, have_std = 0;
    snprintf(source_dir, sizeof(source_dir), "%s/source", store_dir);
    snprintf(out_path, sizeof(out_path), "%s/encoder_f32.safetensors",
             source_dir);
    snprintf(cfg_path, sizeof(cfg_path), "%s/config.json", store_dir);
    if (file_nonempty(out_path) && file_nonempty(cfg_path)) {
        fprintf(stderr, "reusing F32 encoder store %s\n", out_path);
        return 1;
    }
    if (!mkdir_p(store_dir) || !mkdir_p(source_dir)) {
        snprintf(error, error_size, "cannot mkdir %s", store_dir);
        return 0;
    }
    if (!h3_st_read_header(fp16_path, &header, error, error_size)) return 0;
    for (i = 0; i < header.tensor_count; i++)
        if (keep_encoder_tensor(header.tensors[i].name)) n_keep++;
    if (!n_keep) {
        snprintf(error, error_size, "no encoder tensors in %s", fp16_path);
        h3_st_free_header(&header);
        return 0;
    }
    json_cap = n_keep * 160 + 8;
    json = malloc(json_cap);
    if (!json) {
        snprintf(error, error_size, "OOM json header");
        h3_st_free_header(&header);
        return 0;
    }
    json[0] = '{';
    json_len = 1;
    for (i = 0; i < header.tensor_count; i++) {
        const h3_st_tensor *t = &header.tensors[i];
        uint64_t elements, nbytes;
        char piece[256];
        int d, n;
        if (!keep_encoder_tensor(t->name)) continue;
        elements = h3_st_tensor_elements(t);
        nbytes = elements * 4u;
        n = snprintf(piece, sizeof(piece),
                     "%s\"%s\":{\"dtype\":\"F32\",\"shape\":[",
                     json_len > 1 ? "," : "", t->name);
        if (n < 0 || json_len + (size_t)n + 128 >= json_cap) {
            snprintf(error, error_size, "json header overflow");
            free(json);
            h3_st_free_header(&header);
            return 0;
        }
        memcpy(json + json_len, piece, (size_t)n);
        json_len += (size_t)n;
        for (d = 0; d < t->ndim; d++) {
            n = snprintf(piece, sizeof(piece), "%s%llu", d ? "," : "",
                         (unsigned long long)t->shape[d]);
            memcpy(json + json_len, piece, (size_t)n);
            json_len += (size_t)n;
        }
        n = snprintf(piece, sizeof(piece), "],\"data_offsets\":[%llu,%llu]}",
                     (unsigned long long)data_off,
                     (unsigned long long)(data_off + nbytes));
        memcpy(json + json_len, piece, (size_t)n);
        json_len += (size_t)n;
        data_off += nbytes;
    }
    json[json_len++] = '}';
    header_size = json_len;
    while (header_size % 8u) header_size++;
    out = fopen(out_path, "wb");
    if (!out) {
        snprintf(error, error_size, "cannot write %s", out_path);
        free(json);
        h3_st_free_header(&header);
        return 0;
    }
    if (fwrite(&header_size, 8, 1, out) != 1 ||
        fwrite(json, 1, json_len, out) != json_len) {
        snprintf(error, error_size, "cannot write safetensors header");
        fclose(out);
        free(json);
        h3_st_free_header(&header);
        return 0;
    }
    while (json_len < header_size) {
        if (fputc(' ', out) == EOF) {
            snprintf(error, error_size, "cannot pad safetensors header");
            fclose(out);
            free(json);
            h3_st_free_header(&header);
            return 0;
        }
        json_len++;
    }
    free(json);
    for (i = 0; i < header.tensor_count; i++) {
        const h3_st_tensor *t = &header.tensors[i];
        size_t got = 0;
        h3_dtype dtype = H3_DTYPE_UNKNOWN;
        float *data;
        if (!keep_encoder_tensor(t->name)) continue;
        data = h3_st_load_f32(&header, t->name, &got, &dtype, error,
                              error_size);
        if (!data) {
            fclose(out);
            h3_st_free_header(&header);
            return 0;
        }
        if (fwrite(data, sizeof(float), got, out) != got) {
            snprintf(error, error_size, "cannot write tensor %s", t->name);
            free(data);
            fclose(out);
            h3_st_free_header(&header);
            return 0;
        }
        if (!strcmp(t->name, "latents_mean") && got == 24) {
            memcpy(mean, data, sizeof(mean));
            have_mean = 1;
        }
        if (!strcmp(t->name, "latents_std") && got == 24) {
            memcpy(stdv, data, sizeof(stdv));
            have_std = 1;
        }
        free(data);
    }
    fclose(out);
    h3_st_free_header(&header);
    if (!have_mean || !have_std || !write_latents_config(cfg_path, mean, stdv)) {
        snprintf(error, error_size, "cannot write %s", cfg_path);
        return 0;
    }
    fprintf(stderr, "wrote F32 encoder store %s (%.1f MiB data)\n", out_path,
            (double)data_off / (1024.0 * 1024.0));
    return 1;
}

/* Same synthetic pixels as tests/test_h3_vae_encode_f32.c fill_tile_rgb. */
static void fill_tile_rgb(float *rgb, int frames, int height, int width) {
    int c, t, y, x;
    for (c = 0; c < 3; c++)
        for (t = 0; t < frames; t++)
            for (y = 0; y < height; y++)
                for (x = 0; x < width; x++) {
                    size_t idx = (((size_t)c * (size_t)frames + (size_t)t) *
                                      (size_t)height + (size_t)y) *
                                     (size_t)width +
                                 (size_t)x;
                    float u = (float)((x * 13 + y * 7 + t * 3 + c * 17) % 251) /
                              250.0f;
                    rgb[idx] = u;
                }
}

static float *load_rgb24_chw(const char *path, int height, int width) {
    size_t n = (size_t)height * (size_t)width * 3u;
    unsigned char *raw = malloc(n);
    float *rgb = malloc(n * sizeof(*rgb));
    FILE *file = fopen(path, "rb");
    size_t got;
    int y, x, c;
    if (!raw || !rgb || !file) {
        free(raw);
        free(rgb);
        if (file) fclose(file);
        return NULL;
    }
    got = fread(raw, 1, n, file);
    fclose(file);
    if (got != n) {
        free(raw);
        free(rgb);
        return NULL;
    }
    for (c = 0; c < 3; c++)
        for (y = 0; y < height; y++)
            for (x = 0; x < width; x++) {
                size_t src = ((size_t)y * (size_t)width + (size_t)x) * 3u +
                             (size_t)c;
                size_t dst = (((size_t)c * (size_t)height) + (size_t)y) *
                                 (size_t)width +
                             (size_t)x;
                rgb[dst] = (float)raw[src] / 255.0f;
            }
    free(raw);
    return rgb;
}

static int save_raw_f32(const char *path, const float *src, size_t n) {
    FILE *file = fopen(path, "wb");
    size_t put;
    if (!file) return 0;
    put = fwrite(src, sizeof(float), n, file);
    fclose(file);
    return put == n;
}

static void dump_metal_tile(const char *weights, const float *pixels,
                            const char *dump_path, const char *tag) {
    const int frames = 1, height = 256, width = 256;
    const int want_t = latent_t_from_frames(frames);
    const int want_h = height / 16, want_w = width / 16;
    const size_t lat_n = (size_t)24 * (size_t)want_t * (size_t)want_h *
                         (size_t)want_w;
    h3_video_latent got;
    char error[512];
    size_t i;
    double mean = 0.0, sq = 0.0;
    memset(&got, 0, sizeof(got));
    error[0] = '\0';
    if (!h3_video_vae_encode(weights, "h3_shaders.metal", pixels, frames,
                             height, width, progress, NULL, &got, error,
                             sizeof(error)))
        die(error);
    if (got.time != want_t || got.height != want_h || got.width != want_w ||
        !got.values)
        die("Metal h3_video_vae_encode returned the wrong shape");
    for (i = 0; i < lat_n; i++) {
        if (!isfinite(got.values[i]))
            die("Metal h3_video_vae_encode returned non-finite values");
        mean += (double)got.values[i];
        sq += (double)got.values[i] * (double)got.values[i];
    }
    mean /= (double)lat_n;
    sq = sqrt(sq / (double)lat_n - mean * mean);
    printf("metal h3_video_vae_encode %s: shape [24,%d,%d,%d] mean=%.5g "
           "std=%.5g\n",
           tag, got.time, got.height, got.width, mean, sq);
    printf("metal encoder: %.3f GiB allocated, %.3f GPU seconds, "
           "%llu MPS convolutions, %llu submissions\n",
           (double)got.gpu_stats.allocated_bytes / (1024.0 * 1024.0 * 1024.0),
           got.gpu_stats.gpu_seconds,
           (unsigned long long)got.gpu_stats.mps_conv_dispatches,
           (unsigned long long)got.gpu_stats.submissions);
    if (got.gpu_stats.mps_conv_dispatches != 34 ||
        got.gpu_stats.submissions != 18)
        die("Metal encode_tile dispatch structure changed unexpectedly");
    if (!save_raw_f32(dump_path, got.values, lat_n))
        die("cannot write --dump-latent");
    printf("wrote %s (%zu F32, channel-major 24x1x16x16)\n", dump_path, lat_n);
    h3_video_latent_free(&got);
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s [MODEL_ROOT] [FIXTURE]\n"
            "       %s --dump-latent OUT.f32 --vae-fp16 FILE [--synthetic]\n"
            "          [--rgb24 FILE] [--store-dir DIR]\n",
            argv0, argv0);
}

static int run_fixture(const char *model_root, const char *fixture_path) {
    char error[512];
    h3_st_header fixture;
    if (!h3_st_read_header(fixture_path, &fixture, error, sizeof(error)))
        die(error);
    const h3_st_tensor *pixel_tensor = h3_st_find(&fixture, "x.pixels");
    const h3_st_tensor *latent_tensor = h3_st_find(&fixture, "x.latent");
    if (!pixel_tensor || pixel_tensor->dtype != H3_DTYPE_F32 ||
        (pixel_tensor->ndim != 4 && pixel_tensor->ndim != 5) ||
        pixel_tensor->shape[0] != 1 ||
        pixel_tensor->shape[1] != 3 || !latent_tensor ||
        latent_tensor->dtype != H3_DTYPE_F32 || latent_tensor->ndim != 5 ||
        latent_tensor->shape[0] != 1 || latent_tensor->shape[1] != 24 ||
        latent_tensor->shape[3] * 16 !=
            pixel_tensor->shape[pixel_tensor->ndim - 2] ||
        latent_tensor->shape[4] * 16 !=
            pixel_tensor->shape[pixel_tensor->ndim - 1])
        die("malformed visual encoder fixture");
    int frames = pixel_tensor->ndim == 5 ? (int)pixel_tensor->shape[2] : 1;
    int height = (int)pixel_tensor->shape[pixel_tensor->ndim - 2];
    int width = (int)pixel_tensor->shape[pixel_tensor->ndim - 1];
    if (latent_tensor->shape[2] != (uint64_t)latent_t_from_frames(frames))
        die("visual encoder fixture has unexpected temporal compression");
    size_t pixel_count = h3_st_tensor_elements(pixel_tensor);
    size_t latent_count = h3_st_tensor_elements(latent_tensor);
    float *pixels = malloc(pixel_count * sizeof(*pixels));
    float *want = malloc(latent_count * sizeof(*want));
    if (!pixels || !want) die("out of memory loading visual encoder fixture");
    if (!h3_st_read_data(&fixture, pixel_tensor, pixels,
                         pixel_count * sizeof(*pixels), error, sizeof(error)) ||
        !h3_st_read_data(&fixture, latent_tensor, want,
                         latent_count * sizeof(*want), error, sizeof(error)))
        die(error);
    char weights[1024];
    snprintf(weights, sizeof(weights), "%s/FL2VA/video_vae/source", model_root);
    h3_video_latent got;
    if (!h3_video_vae_encode(weights, "h3_shaders.metal", pixels, frames,
                             height, width,
                             progress, NULL, &got, error, sizeof(error)))
        die(error);
    if (got.time != latent_t_from_frames(frames) ||
        got.height != height / 16 ||
        got.width != width / 16)
        die("native visual encoder returned the wrong shape");
    double maximum = 0.0, square_error = 0.0, square_value = 0.0;
    for (size_t index = 0; index < latent_count; index++) {
        if (!isfinite(got.values[index]))
            die("native visual encoder returned non-finite values");
        double delta = (double)got.values[index] - want[index];
        if (fabs(delta) > maximum) maximum = fabs(delta);
        square_error += delta * delta;
        square_value += (double)want[index] * want[index];
    }
    double relative_l2 = sqrt(square_error /
                              (square_value > 1e-24 ? square_value : 1e-24));
    printf("visual encoder: max abs %.7g, relative L2 %.7g\n",
           maximum, relative_l2);
    printf("visual encoder: %.3f GiB allocated, %.3f GPU seconds, "
           "%llu MPS convolutions, %llu submissions\n",
           (double)got.gpu_stats.allocated_bytes / (1024.0 * 1024.0 * 1024.0),
           got.gpu_stats.gpu_seconds,
           (unsigned long long)got.gpu_stats.mps_conv_dispatches,
           (unsigned long long)got.gpu_stats.submissions);
    if (maximum >= 2e-3 || relative_l2 >= 0.02)
        die("native visual encoder exceeds MLX parity bound");
    int tiles = axis_tiles(height) * axis_tiles(width);
    if (got.gpu_stats.mps_conv_dispatches != (uint64_t)(34 * tiles) ||
        got.gpu_stats.submissions != (uint64_t)(18 * tiles))
        die("native visual encoder dispatch structure changed unexpectedly");
    h3_video_latent_free(&got);
    h3_st_free_header(&fixture);
    free(pixels);
    free(want);
    puts("ok: native Metal visual encoder matches the MLX anchor latent");
    return 0;
}

int main(int argc, char **argv) {
    const char *model_root = "MiniMax-H3";
    const char *fixture_path =
        "misc/fixtures/h3_real_video_encoder_256.safetensors";
    const char *dump_latent = NULL, *vae_fp16 = NULL, *rgb24 = NULL;
    const char *store_dir = "out/verify_stage_c/metal_vae_f32";
    int synthetic = 0, positional = 0, i;
    char error[512];
    char weights[1024];
    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--dump-latent") && i + 1 < argc)
            dump_latent = argv[++i];
        else if (!strcmp(argv[i], "--vae-fp16") && i + 1 < argc)
            vae_fp16 = argv[++i];
        else if (!strcmp(argv[i], "--rgb24") && i + 1 < argc)
            rgb24 = argv[++i];
        else if (!strcmp(argv[i], "--store-dir") && i + 1 < argc)
            store_dir = argv[++i];
        else if (!strcmp(argv[i], "--synthetic"))
            synthetic = 1;
        else if (!strcmp(argv[i], "--help")) {
            usage(argv[0]);
            return 0;
        } else if (argv[i][0] == '-') {
            usage(argv[0]);
            return 2;
        } else if (positional == 0) {
            model_root = argv[i];
            positional++;
        } else if (positional == 1) {
            fixture_path = argv[i];
            positional++;
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (!dump_latent)
        return run_fixture(model_root, fixture_path);
    if (!vae_fp16)
        die("dump mode requires --vae-fp16 (Metal loader wants F32 shards)");
    if (!synthetic && !rgb24)
        die("dump mode requires --synthetic and/or --rgb24");
    error[0] = '\0';
    if (!materialize_f32_encoder_store(vae_fp16, store_dir, error,
                                       sizeof(error)))
        die(error);
    snprintf(weights, sizeof(weights), "%s/source", store_dir);
    if (synthetic) {
        const int frames = 1, height = 256, width = 256;
        size_t rgb_n = (size_t)3 * (size_t)frames * (size_t)height *
                       (size_t)width;
        float *rgb = malloc(rgb_n * sizeof(*rgb));
        if (!rgb) die("OOM synthetic rgb");
        fill_tile_rgb(rgb, frames, height, width);
        dump_metal_tile(weights, rgb, dump_latent, "fill_tile_rgb");
        free(rgb);
        puts("ok: dumped Metal h3_video_vae_encode latent (synthetic)");
        return 0;
    }
    {
        float *rgb = load_rgb24_chw(rgb24, 256, 256);
        if (!rgb) die("cannot read --rgb24");
        dump_metal_tile(weights, rgb, dump_latent, "tile256.rgb24");
        free(rgb);
        puts("ok: dumped Metal h3_video_vae_encode latent (rgb24)");
        return 0;
    }
}
