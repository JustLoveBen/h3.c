#ifndef H3_FFMPEG_H
#define H3_FFMPEG_H

#include "h3_av_sync.h"

#include <stddef.h>
#include <stdint.h>

typedef enum {
    H3_IMAGE_FIT_STRETCH = 0,
    H3_IMAGE_FIT_COVER = 1
} h3_image_fit;

/* Inspect the first visual stream without decoding it. H3_FFPROBE may select
 * an explicit ffprobe-compatible executable. */
int h3_ffprobe_visual_size(const char *path, int *width, int *height,
                           char *error, size_t error_size);

/* Decode one visual stream through FFmpeg. The caller owns channel-major F32
 * [3,height,width] RGB in [0,1]. */
int h3_ffmpeg_read_image_f32(const char *path, int width, int height,
                             h3_image_fit fit, float **pixels,
                             char *error, size_t error_size);

/* Decode a 24 fps visual stream to channel-major F32 [3,T,H,W] in [0,1].
 * The returned frame count is trimmed down to the released 5+17k cadence. */
int h3_ffmpeg_read_video_f32(const char *path, int width, int height,
                             int max_frames, float **pixels, int *frames,
                             char *error, size_t error_size);

/* Decode the first audio stream as channel-major stereo F32 at 32 kHz.
 * max_samples bounds allocation. truncate_at_limit is used for a video's
 * soundtrack; standalone clips report an error instead of silently trimming. */
int h3_ffmpeg_read_audio_f32(const char *path, int max_samples,
                             int truncate_at_limit,
                             float **pcm, int *samples,
                             char *error, size_t error_size);

int h3_ffmpeg_write_rgb24(const char *path, const uint8_t *frames,
                          int frame_count, int width, int height, int fps,
                          char *error, size_t error_size);

/* Encode RGB24 video and channel-major F32 PCM through two concurrent pipes.
 * No intermediate uncompressed media file is created. */
int h3_ffmpeg_write_av_rgb24_f32(const char *path, const uint8_t *frames,
                                 int frame_count, int width, int height,
                                 int fps, const float *pcm, int samples,
                                 int channels, int sample_rate,
                                 char *error, size_t error_size);

int h3_ffprobe_av_streams(const char *path, h3_av_probe *out, char *error,
                          size_t error_size);

typedef struct {
    double pipe_wait_ms;
    double encode_ms;
    int frames_pushed;
} h3_ffmpeg_av_stats;

typedef struct h3_ffmpeg_av_enc h3_ffmpeg_av_enc;

typedef struct {
    int width;
    int height;
    int fps;
    int sample_rate;
    int channels;
    int has_audio;
    const char *path;
} h3_ffmpeg_enc_opts;

/* H3_FFMPEG_STREAM=0 → one write_all. Default 1 (per-frame push). */
int h3_ffmpeg_stream_wanted(void);
int h3_ffmpeg_threads(void);
const char *h3_ffmpeg_venc(void);
/* libx264 -preset: H3_X264_PRESET 显式优先；否则按 H3_X264_PRESET_TIER
 * quality(默认)=veryfast / preview=ultrafast。封装层改动,latent 不变。 */
const char *h3_ffmpeg_x264_preset(void);
/* libx264 -crf: H3_X264_CRF 显式优先；否则 preview→23，hires
 * (W×H≥1152×640)→14，其余→18。封装层；不改 latent。 */
const char *h3_ffmpeg_x264_crf(int width, int height);

/* Fill argv with NULL-terminated ffmpeg encode command. scratch holds
 * size/rate/threads strings. Returns argc or -1. */
int h3_ffmpeg_build_encode_argv(const h3_ffmpeg_enc_opts *opts, char **argv,
                                size_t cap, char *scratch, size_t scratch_n);

int h3_ffmpeg_av_open(const char *path, int width, int height, int fps,
                      int sample_rate, int channels, h3_ffmpeg_av_enc **enc,
                      char *error, size_t error_size);
int h3_ffmpeg_av_push_frames(h3_ffmpeg_av_enc *enc, const uint8_t *rgb24,
                             int n, char *error, size_t error_size);
int h3_ffmpeg_av_push_pcm(h3_ffmpeg_av_enc *enc, const float *pcm, int samples,
                          char *error, size_t error_size);
int h3_ffmpeg_av_close(h3_ffmpeg_av_enc *enc, h3_ffmpeg_av_stats *stats,
                       char *error, size_t error_size);

int h3_ffmpeg_write_av_rgb24_f32_ex(const char *path, const uint8_t *frames,
                                    int frame_count, int width, int height,
                                    int fps, const float *pcm, int samples,
                                    int channels, int sample_rate,
                                    h3_ffmpeg_av_stats *stats, char *error,
                                    size_t error_size);

#endif
