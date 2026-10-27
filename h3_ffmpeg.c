#include "h3_ffmpeg.h"

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <signal.h>
#include <spawn.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#ifndef SSIZE_MAX
#define SSIZE_MAX ((ssize_t)(((size_t)-1) >> 1))
#endif

extern char **environ;

static const char *ffmpeg_program(void) {
    const char *override = getenv("H3_FFMPEG");
    return override && *override ? override : "ffmpeg";
}

static const char *ffprobe_program(void) {
    const char *override = getenv("H3_FFPROBE");
    return override && *override ? override : "ffprobe";
}

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int make_parents(const char *path, char *error, size_t error_size) {
    char *copy = strdup(path);
    if (!copy) {
        fail(error, error_size, "out of memory resolving output directory");
        return 0;
    }
    for (char *cursor = copy + 1; *cursor; cursor++) {
        if (*cursor != '/') continue;
        *cursor = '\0';
        if (mkdir(copy, 0755) != 0 && errno != EEXIST) {
            fail(error, error_size, "cannot create output directory %s: %s",
                 copy, strerror(errno));
            free(copy);
            return 0;
        }
        *cursor = '/';
    }
    free(copy);
    return 1;
}

static int write_all(int descriptor, const uint8_t *data, size_t bytes,
                     char *error, size_t error_size) {
    while (bytes) {
        ssize_t written = write(descriptor, data,
                                bytes > (size_t)SSIZE_MAX ?
                                (size_t)SSIZE_MAX : bytes);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) {
            fail(error, error_size, "cannot stream RGB frames to FFmpeg: %s",
                 written < 0 ? strerror(errno) : "short write");
            return 0;
        }
        data += (size_t)written;
        bytes -= (size_t)written;
    }
    return 1;
}

int h3_ffprobe_visual_size(const char *path, int *width, int *height,
                           char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (width) *width = 0;
    if (height) *height = 0;
    if (!path || !*path || !width || !height) {
        fail(error, error_size, "invalid FFprobe visual-size arguments");
        return 0;
    }
    int stream[2];
    if (pipe(stream) != 0) {
        fail(error, error_size, "cannot create FFprobe pipe: %s",
             strerror(errno));
        return 0;
    }
    char *arguments[] = {
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x",
        (char *)path, NULL
    };
    posix_spawn_file_actions_t actions;
    int code = posix_spawn_file_actions_init(&actions);
    if (!code) code = posix_spawn_file_actions_adddup2(
        &actions, stream[1], STDOUT_FILENO);
    if (!code) code = posix_spawn_file_actions_addclose(&actions, stream[0]);
    if (!code) code = posix_spawn_file_actions_addclose(&actions, stream[1]);
    pid_t child = -1;
    if (!code) code = posix_spawnp(&child, ffprobe_program(), &actions, NULL,
                                    arguments, environ);
    posix_spawn_file_actions_destroy(&actions);
    close(stream[1]);
    if (code) {
        close(stream[0]);
        fail(error, error_size, "cannot start FFprobe: %s", strerror(code));
        return 0;
    }
    char output[128];
    size_t received = 0;
    int overflow = 0;
    while (1) {
        char byte;
        ssize_t amount = read(stream[0], &byte, 1);
        if (amount < 0 && errno == EINTR) continue;
        if (amount <= 0) break;
        if (received + 1 < sizeof(output)) output[received++] = byte;
        else overflow = 1;
    }
    close(stream[0]);
    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
        if (errno == EINTR) continue;
        fail(error, error_size, "cannot wait for FFprobe: %s", strerror(errno));
        return 0;
    }
    output[received] = '\0';
    int parsed_width = 0, parsed_height = 0, consumed = 0;
    if (overflow || !WIFEXITED(status) || WEXITSTATUS(status) != 0 ||
        sscanf(output, "%dx%d%n", &parsed_width, &parsed_height, &consumed) != 2) {
        fail(error, error_size, "FFprobe could not inspect visual stream %s",
             path);
        return 0;
    }
    for (char *cursor = output + consumed; *cursor; cursor++) {
        if (*cursor != ' ' && *cursor != '\t' && *cursor != '\r' &&
            *cursor != '\n') {
            fail(error, error_size, "FFprobe returned an invalid visual size");
            return 0;
        }
    }
    if (parsed_width < 1 || parsed_height < 1) {
        fail(error, error_size, "visual stream has invalid dimensions %dx%d",
             parsed_width, parsed_height);
        return 0;
    }
    *width = parsed_width;
    *height = parsed_height;
    return 1;
}

int h3_ffmpeg_stream_wanted(void) {
    const char *e = getenv("H3_FFMPEG_STREAM");
    if (e && e[0] == '0' && e[1] == '\0') return 0;
    return 1;
}

int h3_ffmpeg_threads(void) {
    const char *e = getenv("H3_FFMPEG_THREADS");
    if (!e || !e[0]) return 0;
    return (int)strtol(e, NULL, 10);
}

const char *h3_ffmpeg_venc(void) {
    const char *e = getenv("H3_FFMPEG_VENC");
    return (e && e[0]) ? e : "libx264";
}

/* 画档位: quality(默认,veryfast) / preview(T1 极速预览,ultrafast).
 * H3_X264_PRESET 显式值始终优先。 */
static int x264_tier_preview(void) {
    const char *e = getenv("H3_X264_PRESET_TIER");
    return e && !strcmp(e, "preview");
}

const char *h3_ffmpeg_x264_preset(void) {
    const char *e = getenv("H3_X264_PRESET");
    if (e && e[0]) return e;
    return x264_tier_preview() ? "ultrafast" : "veryfast";
}

const char *h3_ffmpeg_x264_crf(int width, int height) {
    const char *e = getenv("H3_X264_CRF");
    if (e && e[0]) return e;
    if (x264_tier_preview()) return "23";
    /* 1344×768 CRF18+veryfast → ~1.3Mbps talking-head macroblocks.
     * Raise quality floor for ≥1152×640 eval/product mux. */
    if (width > 0 && height > 0 &&
        (long)width * (long)height >= 1152L * 640L)
        return "14";
    return "18";
}

static double ffmpeg_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static int ffprobe_csv(const char *path, const char *select,
                       const char *entries, char *out, size_t out_n,
                       char *error, size_t error_size) {
    int stream[2];
    char *arguments[16];
    posix_spawn_file_actions_t actions;
    pid_t child = -1;
    int code;
    size_t received = 0;
    int status = 0;
    if (!path || !select || !entries || !out || out_n < 2) {
        fail(error, error_size, "invalid ffprobe csv args");
        return 0;
    }
    if (pipe(stream) != 0) {
        fail(error, error_size, "cannot create FFprobe pipe: %s",
             strerror(errno));
        return 0;
    }
    arguments[0] = (char *)"ffprobe";
    arguments[1] = (char *)"-v";
    arguments[2] = (char *)"error";
    arguments[3] = (char *)"-select_streams";
    arguments[4] = (char *)select;
    arguments[5] = (char *)"-show_entries";
    arguments[6] = (char *)entries;
    arguments[7] = (char *)"-of";
    arguments[8] = (char *)"default=noprint_wrappers=1";
    arguments[9] = (char *)path;
    arguments[10] = NULL;
    code = posix_spawn_file_actions_init(&actions);
    if (!code) code = posix_spawn_file_actions_adddup2(
        &actions, stream[1], STDOUT_FILENO);
    if (!code) code = posix_spawn_file_actions_addclose(&actions, stream[0]);
    if (!code) code = posix_spawn_file_actions_addclose(&actions, stream[1]);
    if (!code) code = posix_spawnp(&child, ffprobe_program(), &actions, NULL,
                                    arguments, environ);
    posix_spawn_file_actions_destroy(&actions);
    close(stream[1]);
    if (code) {
        close(stream[0]);
        fail(error, error_size, "cannot start FFprobe: %s", strerror(code));
        return 0;
    }
    while (received + 1 < out_n) {
        ssize_t amount = read(stream[0], out + received, out_n - received - 1);
        if (amount < 0 && errno == EINTR) continue;
        if (amount <= 0) break;
        received += (size_t)amount;
    }
    close(stream[0]);
    while (waitpid(child, &status, 0) < 0) {
        if (errno == EINTR) continue;
        fail(error, error_size, "cannot wait for FFprobe: %s", strerror(errno));
        return 0;
    }
    out[received] = '\0';
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        fail(error, error_size, "FFprobe failed on %s", path);
        return 0;
    }
    return 1;
}

static double ff_kv_num(char *line, const char *key, double fallback) {
    char *p, *comma, *eq;
    size_t klen;
    if (!line || !key) return fallback;
    klen = strlen(key);
    p = line;
    while (p && *p) {
        comma = strchr(p, ',');
        if (comma) *comma = '\0';
        eq = strchr(p, '=');
        if (eq && (size_t)(eq - p) == klen && !strncmp(p, key, klen)) {
            double v = fallback;
            char *end = NULL;
            if (strcmp(eq + 1, "N/A") != 0) {
                v = strtod(eq + 1, &end);
                if (end == eq + 1) v = fallback;
            }
            if (comma) *comma = ',';
            return v;
        }
        if (comma) {
            *comma = ',';
            p = comma + 1;
        } else
            break;
    }
    return fallback;
}

int h3_ffprobe_av_streams(const char *path, h3_av_probe *out, char *error,
                          size_t error_size) {
    char vbuf[256], abuf[256];
    if (error && error_size) error[0] = '\0';
    if (out) memset(out, 0, sizeof(*out));
    if (!path || !*path || !out) {
        fail(error, error_size, "invalid ffprobe av args");
        return 0;
    }
    if (!ffprobe_csv(path, "v:0", "stream=start_time,duration,width,height",
                     vbuf, sizeof(vbuf), error, error_size))
        return 0;
    {
        char *p;
        for (p = vbuf; *p; p++)
            if (*p == '\n' || *p == '\r') *p = ',';
    }
    out->video_start_s = ff_kv_num(vbuf, "start_time", 0.0);
    out->video_duration_s = ff_kv_num(vbuf, "duration", 0.0);
    out->width = (int)ff_kv_num(vbuf, "width", 0.0);
    out->height = (int)ff_kv_num(vbuf, "height", 0.0);
    if (!ffprobe_csv(path, "a:0",
                     "stream=start_time,duration,sample_rate,channels", abuf,
                     sizeof(abuf), error, error_size))
        return 0;
    {
        char *p;
        for (p = abuf; *p; p++)
            if (*p == '\n' || *p == '\r') *p = ',';
    }
    out->audio_start_s = ff_kv_num(abuf, "start_time", 0.0);
    out->audio_duration_s = ff_kv_num(abuf, "duration", 0.0);
    out->sample_rate = (int)ff_kv_num(abuf, "sample_rate", 0.0);
    out->channels = (int)ff_kv_num(abuf, "channels", 0.0);
    if (out->width < 1 || out->height < 1 || out->video_duration_s <= 0.0) {
        fail(error, error_size, "FFprobe video stream incomplete");
        return 0;
    }
    return 1;
}

int h3_ffmpeg_build_encode_argv(const h3_ffmpeg_enc_opts *opts, char **argv,
                                size_t cap, char *scratch, size_t scratch_n) {
    char *size, *rate, *ar, *ac, *ain, *th;
    const char *venc;
    int threads;
    size_t n = 0, used = 0;
    if (!opts || !argv || cap < 40 || !scratch || scratch_n < 160 ||
        !opts->path)
        return -1;
    venc = h3_ffmpeg_venc();
    threads = h3_ffmpeg_threads();
    size = scratch + used;
    used += 64;
    rate = scratch + used;
    used += 16;
    ar = scratch + used;
    used += 16;
    ac = scratch + used;
    used += 16;
    ain = scratch + used;
    used += 16;
    th = scratch + used;
    used += 16;
    if (used > scratch_n) return -1;
    snprintf(size, 64, "%dx%d", opts->width, opts->height);
    snprintf(rate, 16, "%d", opts->fps);
    snprintf(ar, 16, "%d", opts->sample_rate > 0 ? opts->sample_rate : 32000);
    snprintf(ac, 16, "%d", opts->channels > 0 ? opts->channels : 2);
    snprintf(ain, 16, "pipe:3");
    snprintf(th, 16, "%d", threads);
#define ARG(s)                                                             \
    do {                                                                   \
        if (n + 1 >= cap) return -1;                                       \
        argv[n++] = (char *)(s);                                           \
    } while (0)
    ARG("ffmpeg");
    ARG("-y");
    ARG("-loglevel");
    ARG("error");
    ARG("-f");
    ARG("rawvideo");
    ARG("-pixel_format");
    ARG("rgb24");
    ARG("-video_size");
    ARG(size);
    ARG("-framerate");
    ARG(rate);
    ARG("-i");
    ARG("pipe:0");
    if (opts->has_audio) {
        ARG("-f");
        ARG("f32le");
        ARG("-ar");
        ARG(ar);
        ARG("-ac");
        ARG(ac);
        ARG("-i");
        ARG(ain);
        ARG("-map");
        ARG("0:v:0");
        ARG("-map");
        ARG("1:a:0");
    } else {
        ARG("-an");
    }
    ARG("-c:v");
    ARG(venc);
    if (!strcmp(venc, "h264_nvenc")) {
        ARG("-preset");
        ARG("p4");
    } else {
        ARG("-preset");
        ARG(h3_ffmpeg_x264_preset());
        ARG("-crf");
        ARG(h3_ffmpeg_x264_crf(opts->width, opts->height));
    }
    ARG("-pix_fmt");
    ARG("yuv420p");
    if (opts->has_audio) {
        ARG("-c:a");
        ARG("aac");
        ARG("-b:a");
        ARG("192k");
        ARG("-shortest");
    }
    if (threads > 0) {
        ARG("-threads");
        ARG(th);
    }
    ARG("-movflags");
    ARG("+faststart");
    ARG(opts->path);
    ARG(NULL);
#undef ARG
    return (int)n - 1;
}

int h3_ffmpeg_read_image_f32(const char *path, int width, int height,
                             h3_image_fit fit, float **pixels,
                             char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (pixels) *pixels = NULL;
    if (!path || !*path || !pixels || width < 1 || height < 1 ||
        (fit != H3_IMAGE_FIT_STRETCH && fit != H3_IMAGE_FIT_COVER)) {
        fail(error, error_size, "invalid FFmpeg image input arguments");
        return 0;
    }
    if ((size_t)width > SIZE_MAX / (size_t)height) {
        fail(error, error_size, "decoded image size overflows");
        return 0;
    }
    size_t area = (size_t)width * (size_t)height;
    if (area > SIZE_MAX / 3 ||
        area * 3 > SIZE_MAX / sizeof(float)) {
        fail(error, error_size, "decoded image size overflows");
        return 0;
    }
    size_t bytes = area * 3;
    uint8_t *rgb = malloc(bytes);
    float *channel_major = malloc(bytes * sizeof(*channel_major));
    if (!rgb || !channel_major) {
        free(rgb);
        free(channel_major);
        fail(error, error_size, "out of memory decoding input image");
        return 0;
    }
    char filter[256];
    if (fit == H3_IMAGE_FIT_STRETCH) {
        snprintf(filter, sizeof(filter), "scale=%d:%d:flags=lanczos",
                 width, height);
    } else {
        snprintf(filter, sizeof(filter),
                 "scale=%d:%d:force_original_aspect_ratio=increase:flags=lanczos,"
                 "crop=%d:%d", width, height, width, height);
    }
    int stream[2];
    if (pipe(stream) != 0) {
        free(rgb);
        free(channel_major);
        fail(error, error_size, "cannot create FFmpeg image pipe: %s",
             strerror(errno));
        return 0;
    }
    char *arguments[] = {
        "ffmpeg", "-v", "error", "-i", (char *)path,
        "-frames:v", "1", "-vf", filter,
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1", NULL
    };
    posix_spawn_file_actions_t actions;
    int code = posix_spawn_file_actions_init(&actions);
    if (!code) code = posix_spawn_file_actions_adddup2(
        &actions, stream[1], STDOUT_FILENO);
    if (!code) code = posix_spawn_file_actions_addclose(&actions, stream[0]);
    if (!code) code = posix_spawn_file_actions_addclose(&actions, stream[1]);
    pid_t child = -1;
    if (!code) code = posix_spawnp(&child, ffmpeg_program(), &actions, NULL,
                                    arguments, environ);
    posix_spawn_file_actions_destroy(&actions);
    close(stream[1]);
    if (code) {
        close(stream[0]);
        free(rgb);
        free(channel_major);
        fail(error, error_size, "cannot start FFmpeg: %s", strerror(code));
        return 0;
    }
    size_t received = 0;
    while (received < bytes) {
        ssize_t amount = read(stream[0], rgb + received, bytes - received);
        if (amount < 0 && errno == EINTR) continue;
        if (amount <= 0) break;
        received += (size_t)amount;
    }
    uint8_t extra;
    ssize_t trailing;
    do trailing = read(stream[0], &extra, 1);
    while (trailing < 0 && errno == EINTR);
    close(stream[0]);
    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
        if (errno == EINTR) continue;
        free(rgb);
        free(channel_major);
        fail(error, error_size, "cannot wait for FFmpeg: %s", strerror(errno));
        return 0;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0 ||
        received != bytes || trailing != 0) {
        free(rgb);
        free(channel_major);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
            fail(error, error_size, "FFmpeg could not decode image %s (status %d)",
                 path, WIFEXITED(status) ? WEXITSTATUS(status) : -1);
        else
            fail(error, error_size,
                 "FFmpeg decoded %zu bytes for %dx%d image, expected %zu",
                 received + (trailing > 0 ? 1u : 0u), width, height, bytes);
        return 0;
    }
    const float scale = 1.0f / 255.0f;
    for (size_t pixel = 0; pixel < area; pixel++) {
        channel_major[pixel] = (float)rgb[3 * pixel] * scale;
        channel_major[area + pixel] = (float)rgb[3 * pixel + 1] * scale;
        channel_major[2 * area + pixel] = (float)rgb[3 * pixel + 2] * scale;
    }
    free(rgb);
    *pixels = channel_major;
    return 1;
}

int h3_ffmpeg_read_video_f32(const char *path, int width, int height,
                             int max_frames, float **pixels, int *frames,
                             char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (pixels) *pixels = NULL;
    if (frames) *frames = 0;
    if (!path || !*path || !pixels || !frames || width < 1 || height < 1 ||
        max_frames < 5 || (size_t)width > SIZE_MAX / (size_t)height) {
        fail(error, error_size, "invalid FFmpeg video input arguments");
        return 0;
    }
    size_t area = (size_t)width * (size_t)height;
    if (area > SIZE_MAX / 3) {
        fail(error, error_size, "decoded video frame size overflows");
        return 0;
    }
    size_t frame_bytes = area * 3;
    if ((size_t)max_frames > SIZE_MAX / frame_bytes) {
        fail(error, error_size, "decoded video size overflows");
        return 0;
    }
    size_t capacity = (size_t)max_frames * frame_bytes;
    uint8_t *rgb = malloc(capacity);
    if (!rgb) {
        fail(error, error_size, "out of memory decoding input video");
        return 0;
    }
    char filter[256], frame_limit[32];
    snprintf(filter, sizeof(filter),
             "fps=24,scale=%d:%d:flags=lanczos,setsar=1", width, height);
    snprintf(frame_limit, sizeof(frame_limit), "%d", max_frames);
    int stream[2];
    if (pipe(stream) != 0) {
        free(rgb);
        fail(error, error_size, "cannot create FFmpeg video pipe: %s",
             strerror(errno));
        return 0;
    }
    char *arguments[] = {
        "ffmpeg", "-v", "error", "-i", (char *)path,
        "-map", "0:v:0", "-an", "-vf", filter,
        "-frames:v", frame_limit, "-f", "rawvideo", "-pix_fmt", "rgb24",
        "pipe:1", NULL
    };
    posix_spawn_file_actions_t actions;
    int code = posix_spawn_file_actions_init(&actions);
    if (!code) code = posix_spawn_file_actions_adddup2(
        &actions, stream[1], STDOUT_FILENO);
    if (!code) code = posix_spawn_file_actions_addclose(&actions, stream[0]);
    if (!code) code = posix_spawn_file_actions_addclose(&actions, stream[1]);
    pid_t child = -1;
    if (!code) code = posix_spawnp(&child, ffmpeg_program(), &actions, NULL,
                                    arguments, environ);
    posix_spawn_file_actions_destroy(&actions);
    close(stream[1]);
    if (code) {
        close(stream[0]);
        free(rgb);
        fail(error, error_size, "cannot start FFmpeg: %s", strerror(code));
        return 0;
    }
    size_t received = 0;
    while (received < capacity) {
        ssize_t amount = read(stream[0], rgb + received, capacity - received);
        if (amount < 0 && errno == EINTR) continue;
        if (amount <= 0) break;
        received += (size_t)amount;
    }
    uint8_t extra;
    ssize_t trailing;
    do trailing = read(stream[0], &extra, 1);
    while (trailing < 0 && errno == EINTR);
    close(stream[0]);
    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
        if (errno == EINTR) continue;
        free(rgb);
        fail(error, error_size, "cannot wait for FFmpeg: %s", strerror(errno));
        return 0;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0 || trailing != 0 ||
        received % frame_bytes) {
        free(rgb);
        fail(error, error_size, "FFmpeg could not decode bounded video %s",
             path);
        return 0;
    }
    int frame_count = (int)(received / frame_bytes);
    if (frame_count < 5) {
        free(rgb);
        fail(error, error_size,
             "reference videos require at least 5 decoded frames");
        return 0;
    }
    while (frame_count % 17 != 5) frame_count--;
    if ((size_t)frame_count > SIZE_MAX / 3 / area ||
        (size_t)frame_count * 3 * area > SIZE_MAX / sizeof(float)) {
        free(rgb);
        fail(error, error_size, "converted video size overflows");
        return 0;
    }
    size_t values = (size_t)frame_count * 3 * area;
    float *channel_major = malloc(values * sizeof(*channel_major));
    if (!channel_major) {
        free(rgb);
        fail(error, error_size, "out of memory converting input video");
        return 0;
    }
    const float scale = 1.0f / 255.0f;
    for (int time = 0; time < frame_count; time++)
        for (size_t pixel = 0; pixel < area; pixel++)
            for (size_t channel = 0; channel < 3; channel++) {
                size_t source = ((size_t)time * area + pixel) * 3 + channel;
                size_t destination = (channel * (size_t)frame_count +
                                      (size_t)time) * area + pixel;
                channel_major[destination] = (float)rgb[source] * scale;
            }
    free(rgb);
    *pixels = channel_major;
    *frames = frame_count;
    return 1;
}

int h3_ffmpeg_read_audio_f32(const char *path, int max_samples,
                             int truncate_at_limit,
                             float **pcm, int *samples,
                             char *error, size_t error_size) {
    enum { AUDIO_RATE = 32000, AUDIO_CHANNELS = 2, MIN_SAMPLES = 64000 };
    if (error && error_size) error[0] = '\0';
    if (pcm) *pcm = NULL;
    if (samples) *samples = 0;
    if (!path || !*path || !pcm || !samples || max_samples < MIN_SAMPLES ||
        max_samples > AUDIO_RATE * 15 ||
        (truncate_at_limit != 0 && truncate_at_limit != 1)) {
        fail(error, error_size, "invalid FFmpeg audio input arguments");
        return 0;
    }
    size_t elements = (size_t)max_samples * AUDIO_CHANNELS;
    if (elements > SIZE_MAX / sizeof(float)) {
        fail(error, error_size, "decoded audio size overflows");
        return 0;
    }
    float *interleaved = malloc(elements * sizeof(*interleaved));
    if (!interleaved) {
        fail(error, error_size, "out of memory decoding reference audio");
        return 0;
    }
    char duration[64];
    double seconds = (double)max_samples / (double)AUDIO_RATE;
    if (!truncate_at_limit) seconds += 1.0 / (double)AUDIO_RATE;
    snprintf(duration, sizeof(duration), "%.9f", seconds);
    int stream[2];
    if (pipe(stream) != 0) {
        free(interleaved);
        fail(error, error_size, "cannot create FFmpeg audio pipe: %s",
             strerror(errno));
        return 0;
    }
    char *arguments[] = {
        "ffmpeg", "-v", "error", "-i", (char *)path,
        "-map", "0:a:0", "-vn", "-ac", "2", "-ar", "32000",
        "-t", duration, "-f", "f32le", "pipe:1", NULL
    };
    posix_spawn_file_actions_t actions;
    int code = posix_spawn_file_actions_init(&actions);
    if (!code) code = posix_spawn_file_actions_adddup2(
        &actions, stream[1], STDOUT_FILENO);
    if (!code) code = posix_spawn_file_actions_addclose(&actions, stream[0]);
    if (!code) code = posix_spawn_file_actions_addclose(&actions, stream[1]);
    pid_t child = -1;
    if (!code) code = posix_spawnp(&child, ffmpeg_program(), &actions, NULL,
                                    arguments, environ);
    posix_spawn_file_actions_destroy(&actions);
    close(stream[1]);
    if (code) {
        close(stream[0]);
        free(interleaved);
        fail(error, error_size, "cannot start FFmpeg: %s", strerror(code));
        return 0;
    }
    size_t capacity = elements * sizeof(*interleaved);
    size_t received = 0;
    while (received < capacity) {
        ssize_t amount = read(stream[0], (uint8_t *)interleaved + received,
                              capacity - received);
        if (amount < 0 && errno == EINTR) continue;
        if (amount <= 0) break;
        received += (size_t)amount;
    }
    uint8_t extra;
    ssize_t trailing;
    do trailing = read(stream[0], &extra, 1);
    while (trailing < 0 && errno == EINTR);
    close(stream[0]);
    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
        if (errno == EINTR) continue;
        free(interleaved);
        fail(error, error_size, "cannot wait for FFmpeg: %s", strerror(errno));
        return 0;
    }
    size_t frame_bytes = AUDIO_CHANNELS * sizeof(float);
    if (!truncate_at_limit && trailing > 0) {
        free(interleaved);
        fail(error, error_size,
             "reference audio exceeds the 15 second total limit");
        return 0;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0 ||
        received % frame_bytes) {
        free(interleaved);
        fail(error, error_size,
             "FFmpeg could not decode a stereo soundtrack from %s", path);
        return 0;
    }
    int sample_count = (int)(received / frame_bytes);
    if (sample_count < MIN_SAMPLES) {
        free(interleaved);
        fail(error, error_size,
             "reference audio requires at least 2 seconds at 32 kHz");
        return 0;
    }
    size_t output_elements = (size_t)sample_count * AUDIO_CHANNELS;
    float *channel_major = malloc(output_elements * sizeof(*channel_major));
    if (!channel_major) {
        free(interleaved);
        fail(error, error_size, "out of memory converting reference audio");
        return 0;
    }
    for (int sample = 0; sample < sample_count; sample++)
        for (int channel = 0; channel < AUDIO_CHANNELS; channel++)
            channel_major[(size_t)channel * (size_t)sample_count +
                          (size_t)sample] =
                interleaved[(size_t)sample * AUDIO_CHANNELS +
                            (size_t)channel];
    free(interleaved);
    *pcm = channel_major;
    *samples = sample_count;
    return 1;
}

int h3_ffmpeg_write_rgb24(const char *path, const uint8_t *frames,
                          int frame_count, int width, int height, int fps,
                          char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!path || !*path || !frames || frame_count < 1 || width < 2 ||
        height < 2 || fps < 1 || width % 2 || height % 2) {
        fail(error, error_size, "invalid FFmpeg RGB output arguments");
        return 0;
    }
    size_t pixels = (size_t)width * (size_t)height;
    if (pixels > SIZE_MAX / 3 ||
        (size_t)frame_count > SIZE_MAX / (pixels * 3)) {
        fail(error, error_size, "RGB video size overflows");
        return 0;
    }
    if (!make_parents(path, error, error_size)) return 0;
    int stream[2];
    if (pipe(stream) != 0) {
        fail(error, error_size, "cannot create FFmpeg pipe: %s", strerror(errno));
        return 0;
    }
    char size[64], rate[32];
    snprintf(size, sizeof(size), "%dx%d", width, height);
    snprintf(rate, sizeof(rate), "%d", fps);
    char *arguments[] = {
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", size, "-framerate", rate,
        "-i", "pipe:0", "-an", "-c:v", "libx264",
        "-preset", (char *)h3_ffmpeg_x264_preset(),
        "-crf", (char *)h3_ffmpeg_x264_crf(width, height),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", (char *)path, NULL
    };
    posix_spawn_file_actions_t actions;
    int code = posix_spawn_file_actions_init(&actions);
    if (!code) code = posix_spawn_file_actions_adddup2(&actions, stream[0], STDIN_FILENO);
    if (!code) code = posix_spawn_file_actions_addclose(&actions, stream[0]);
    if (!code) code = posix_spawn_file_actions_addclose(&actions, stream[1]);
    pid_t child = -1;
    if (!code) code = posix_spawnp(&child, ffmpeg_program(), &actions, NULL,
                                    arguments, environ);
    posix_spawn_file_actions_destroy(&actions);
    close(stream[0]);
    if (code) {
        close(stream[1]);
        fail(error, error_size, "cannot start FFmpeg: %s", strerror(code));
        return 0;
    }
    struct sigaction ignore, previous;
    memset(&ignore, 0, sizeof(ignore));
    ignore.sa_handler = SIG_IGN;
    sigemptyset(&ignore.sa_mask);
    sigaction(SIGPIPE, &ignore, &previous);
    int ok = write_all(stream[1], frames,
                       (size_t)frame_count * pixels * 3,
                       error, error_size);
    close(stream[1]);
    sigaction(SIGPIPE, &previous, NULL);
    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
        if (errno == EINTR) continue;
        if (ok) fail(error, error_size, "cannot wait for FFmpeg: %s",
                     strerror(errno));
        return 0;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        if (ok) fail(error, error_size, "FFmpeg exited with status %d",
                     WIFEXITED(status) ? WEXITSTATUS(status) : -1);
        return 0;
    }
    return ok;
}

static int close_action(posix_spawn_file_actions_t *actions, int descriptor,
                        int keep_a, int keep_b) {
    return descriptor == keep_a || descriptor == keep_b ? 0 :
           posix_spawn_file_actions_addclose(actions, descriptor);
}

struct h3_ffmpeg_av_enc {
    pid_t child;
    int video_fd;
    int audio_fd;
    int width;
    int height;
    int channels;
    size_t pixels;
    double pipe_wait_ms;
    int frames_pushed;
    struct sigaction prev_pipe;
    int have_prev_pipe;
};

int h3_ffmpeg_av_open(const char *path, int width, int height, int fps,
                      int sample_rate, int channels, h3_ffmpeg_av_enc **enc,
                      char *error, size_t error_size) {
    h3_ffmpeg_av_enc *e;
    h3_ffmpeg_enc_opts opts;
    char *arguments[64];
    char scratch[192];
    int video_pipe[2] = {-1, -1}, audio_pipe[2] = {-1, -1};
    int audio_target;
    posix_spawn_file_actions_t actions;
    int code;
    pid_t child = -1;
    if (error && error_size) error[0] = '\0';
    if (!enc) {
        fail(error, error_size, "invalid ffmpeg av open");
        return 0;
    }
    *enc = NULL;
    if (!path || !*path || width < 2 || height < 2 || fps < 1 ||
        width % 2 || height % 2 || sample_rate < 1 || channels < 1) {
        fail(error, error_size, "invalid FFmpeg A/V output arguments");
        return 0;
    }
    if (!make_parents(path, error, error_size)) return 0;
    memset(&opts, 0, sizeof(opts));
    opts.width = width;
    opts.height = height;
    opts.fps = fps;
    opts.sample_rate = sample_rate;
    opts.channels = channels;
    opts.has_audio = 1;
    opts.path = path;
    if (h3_ffmpeg_build_encode_argv(&opts, arguments, 64, scratch,
                                    sizeof(scratch)) < 0) {
        fail(error, error_size, "cannot build ffmpeg argv");
        return 0;
    }
    if (pipe(video_pipe) != 0 || pipe(audio_pipe) != 0) {
        fail(error, error_size, "cannot create FFmpeg A/V pipes: %s",
             strerror(errno));
        if (video_pipe[0] >= 0) close(video_pipe[0]);
        if (video_pipe[1] >= 0) close(video_pipe[1]);
        if (audio_pipe[0] >= 0) close(audio_pipe[0]);
        if (audio_pipe[1] >= 0) close(audio_pipe[1]);
        return 0;
    }
    audio_target = video_pipe[0];
    if (video_pipe[1] > audio_target) audio_target = video_pipe[1];
    if (audio_pipe[0] > audio_target) audio_target = audio_pipe[0];
    if (audio_pipe[1] > audio_target) audio_target = audio_pipe[1];
    audio_target++;
    {
        int i;
        for (i = 0; arguments[i]; i++) {
            if (!strcmp(arguments[i], "pipe:3")) {
                static char ain[32];
                snprintf(ain, sizeof(ain), "pipe:%d", audio_target);
                arguments[i] = ain;
                break;
            }
        }
    }
    code = posix_spawn_file_actions_init(&actions);
    if (!code) code = posix_spawn_file_actions_adddup2(
        &actions, video_pipe[0], STDIN_FILENO);
    if (!code) code = posix_spawn_file_actions_adddup2(
        &actions, audio_pipe[0], audio_target);
    {
        int descriptors[] = {video_pipe[0], video_pipe[1], audio_pipe[0],
                             audio_pipe[1]};
        size_t index;
        for (index = 0; !code && index < sizeof(descriptors) / sizeof(*descriptors);
             index++)
            code = close_action(&actions, descriptors[index], STDIN_FILENO,
                                audio_target);
    }
    if (!code) code = posix_spawnp(&child, ffmpeg_program(), &actions, NULL,
                                    arguments, environ);
    posix_spawn_file_actions_destroy(&actions);
    close(video_pipe[0]);
    close(audio_pipe[0]);
    if (code) {
        close(video_pipe[1]);
        close(audio_pipe[1]);
        fail(error, error_size, "cannot start FFmpeg: %s", strerror(code));
        return 0;
    }
    e = (h3_ffmpeg_av_enc *)calloc(1, sizeof(*e));
    if (!e) {
        close(video_pipe[1]);
        close(audio_pipe[1]);
        (void)waitpid(child, NULL, 0);
        fail(error, error_size, "OOM ffmpeg enc");
        return 0;
    }
    e->child = child;
    e->video_fd = video_pipe[1];
    e->audio_fd = audio_pipe[1];
    e->width = width;
    e->height = height;
    e->channels = channels;
    e->pixels = (size_t)width * (size_t)height;
    memset(&e->prev_pipe, 0, sizeof(e->prev_pipe));
    {
        struct sigaction ignore;
        memset(&ignore, 0, sizeof(ignore));
        ignore.sa_handler = SIG_IGN;
        sigemptyset(&ignore.sa_mask);
        sigaction(SIGPIPE, &ignore, &e->prev_pipe);
        e->have_prev_pipe = 1;
    }
    *enc = e;
    return 1;
}

int h3_ffmpeg_av_push_frames(h3_ffmpeg_av_enc *enc, const uint8_t *rgb24, int n,
                             char *error, size_t error_size) {
    double t0;
    size_t bytes;
    if (!enc || !rgb24 || n < 1 || enc->video_fd < 0) {
        fail(error, error_size, "invalid ffmpeg push frames");
        return 0;
    }
    bytes = (size_t)n * enc->pixels * 3;
    t0 = ffmpeg_now_ms();
    if (!write_all(enc->video_fd, rgb24, bytes, error, error_size)) return 0;
    enc->pipe_wait_ms += ffmpeg_now_ms() - t0;
    enc->frames_pushed += n;
    return 1;
}

int h3_ffmpeg_av_push_pcm(h3_ffmpeg_av_enc *enc, const float *pcm, int samples,
                          char *error, size_t error_size) {
    float *interleaved;
    size_t pcm_elements;
    double t0;
    int sample, channel;
    int ok;
    if (!enc || !pcm || samples < 1 || enc->audio_fd < 0 || enc->channels < 1) {
        fail(error, error_size, "invalid ffmpeg push pcm");
        return 0;
    }
    pcm_elements = (size_t)samples * (size_t)enc->channels;
    interleaved = (float *)malloc(pcm_elements * sizeof(*interleaved));
    if (!interleaved) {
        fail(error, error_size, "out of memory interleaving generated PCM");
        return 0;
    }
    for (sample = 0; sample < samples; sample++)
        for (channel = 0; channel < enc->channels; channel++)
            interleaved[(size_t)sample * (size_t)enc->channels +
                        (size_t)channel] =
                pcm[(size_t)channel * (size_t)samples + (size_t)sample];
    t0 = ffmpeg_now_ms();
    ok = write_all(enc->audio_fd, (const uint8_t *)interleaved,
                   pcm_elements * sizeof(*interleaved), error, error_size);
    enc->pipe_wait_ms += ffmpeg_now_ms() - t0;
    free(interleaved);
    return ok;
}

int h3_ffmpeg_av_close(h3_ffmpeg_av_enc *enc, h3_ffmpeg_av_stats *stats,
                       char *error, size_t error_size) {
    int status = 0;
    double t0;
    if (!enc) {
        fail(error, error_size, "invalid ffmpeg close");
        return 0;
    }
    if (enc->video_fd >= 0) {
        close(enc->video_fd);
        enc->video_fd = -1;
    }
    if (enc->audio_fd >= 0) {
        close(enc->audio_fd);
        enc->audio_fd = -1;
    }
    t0 = ffmpeg_now_ms();
    while (waitpid(enc->child, &status, 0) < 0) {
        if (errno == EINTR) continue;
        if (enc->have_prev_pipe) sigaction(SIGPIPE, &enc->prev_pipe, NULL);
        if (stats) {
            stats->pipe_wait_ms = enc->pipe_wait_ms;
            stats->encode_ms = 0.0;
            stats->frames_pushed = enc->frames_pushed;
        }
        fail(error, error_size, "cannot wait for FFmpeg: %s", strerror(errno));
        free(enc);
        return 0;
    }
    if (enc->have_prev_pipe) sigaction(SIGPIPE, &enc->prev_pipe, NULL);
    if (stats) {
        stats->pipe_wait_ms = enc->pipe_wait_ms;
        stats->encode_ms = ffmpeg_now_ms() - t0;
        stats->frames_pushed = enc->frames_pushed;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        fail(error, error_size, "FFmpeg exited with status %d",
             WIFEXITED(status) ? WEXITSTATUS(status) : -1);
        free(enc);
        return 0;
    }
    free(enc);
    return 1;
}

typedef struct {
    h3_ffmpeg_av_enc *enc;
    const uint8_t *frames;
    int frame_count;
    int per_frame;
    char err[256];
    int ok;
} ffmpeg_vid_job;

typedef struct {
    h3_ffmpeg_av_enc *enc;
    const float *pcm;
    int samples;
    char err[256];
    int ok;
} ffmpeg_pcm_job;

static void ffmpeg_close_fd(int *fd) {
    if (fd && *fd >= 0) {
        close(*fd);
        *fd = -1;
    }
}

static void *ffmpeg_vid_th(void *arg) {
    ffmpeg_vid_job *j = (ffmpeg_vid_job *)arg;
    if (!j || !j->enc || !j->frames) return NULL;
    if (j->per_frame) {
        size_t fb = j->enc->pixels * 3;
        int f;
        for (f = 0; f < j->frame_count; f++) {
            if (!h3_ffmpeg_av_push_frames(j->enc, j->frames + (size_t)f * fb, 1,
                                          j->err, sizeof(j->err))) {
                j->ok = 0;
                ffmpeg_close_fd(&j->enc->video_fd);
                return NULL;
            }
        }
        j->ok = 1;
        ffmpeg_close_fd(&j->enc->video_fd);
        return NULL;
    }
    j->ok = h3_ffmpeg_av_push_frames(j->enc, j->frames, j->frame_count, j->err,
                                     sizeof(j->err));
    ffmpeg_close_fd(&j->enc->video_fd);
    return NULL;
}

static void *ffmpeg_pcm_th(void *arg) {
    ffmpeg_pcm_job *j = (ffmpeg_pcm_job *)arg;
    if (!j || !j->enc || !j->pcm) return NULL;
    j->ok = h3_ffmpeg_av_push_pcm(j->enc, j->pcm, j->samples, j->err,
                                  sizeof(j->err));
    ffmpeg_close_fd(&j->enc->audio_fd);
    return NULL;
}

int h3_ffmpeg_write_av_rgb24_f32_ex(const char *path, const uint8_t *frames,
                                    int frame_count, int width, int height,
                                    int fps, const float *pcm, int samples,
                                    int channels, int sample_rate,
                                    h3_ffmpeg_av_stats *stats, char *error,
                                    size_t error_size) {
    h3_ffmpeg_av_enc *enc = NULL;
    ffmpeg_vid_job vj;
    ffmpeg_pcm_job aj;
    pthread_t vt, at;
    int vcode, acode, ok = 0;
    if (stats) memset(stats, 0, sizeof(*stats));
    if (!frames || frame_count < 1 || !pcm) {
        fail(error, error_size, "invalid FFmpeg A/V output arguments");
        return 0;
    }
    if (!h3_ffmpeg_av_open(path, width, height, fps, sample_rate, channels,
                           &enc, error, error_size))
        return 0;
    memset(&vj, 0, sizeof(vj));
    memset(&aj, 0, sizeof(aj));
    vj.enc = enc;
    vj.frames = frames;
    vj.frame_count = frame_count;
    vj.per_frame = h3_ffmpeg_stream_wanted();
    aj.enc = enc;
    aj.pcm = pcm;
    aj.samples = samples;
    vcode = pthread_create(&vt, NULL, ffmpeg_vid_th, &vj);
    acode = pthread_create(&at, NULL, ffmpeg_pcm_th, &aj);
    if (vcode || acode) {
        fail(error, error_size, "cannot start FFmpeg stream thread");
        if (!vcode) pthread_join(vt, NULL);
        if (!acode) pthread_join(at, NULL);
        (void)h3_ffmpeg_av_close(enc, stats, NULL, 0);
        return 0;
    }
    pthread_join(vt, NULL);
    pthread_join(at, NULL);
    if (!vj.ok) {
        fail(error, error_size, "%s", vj.err[0] ? vj.err : "video pipe failed");
        (void)h3_ffmpeg_av_close(enc, stats, NULL, 0);
        return 0;
    }
    if (!aj.ok) {
        fail(error, error_size, "%s", aj.err[0] ? aj.err : "audio pipe failed");
        (void)h3_ffmpeg_av_close(enc, stats, NULL, 0);
        return 0;
    }
    ok = 1;
    if (!h3_ffmpeg_av_close(enc, stats, error, error_size)) ok = 0;
    return ok;
}

int h3_ffmpeg_write_av_rgb24_f32(const char *path, const uint8_t *frames,
                                 int frame_count, int width, int height,
                                 int fps, const float *pcm, int samples,
                                 int channels, int sample_rate,
                                 char *error, size_t error_size) {
    return h3_ffmpeg_write_av_rgb24_f32_ex(path, frames, frame_count, width,
                                           height, fps, pcm, samples, channels,
                                           sample_rate, NULL, error,
                                           error_size);
}

