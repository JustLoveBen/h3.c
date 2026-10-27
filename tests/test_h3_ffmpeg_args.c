#include "h3_ffmpeg.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
static void expect(int c, const char *m) {
    if (!c) {
        fprintf(stderr, "FAIL %s\n", m);
        fails++;
    }
}

static int argv_has(char **argv, const char *s) {
    int i;
    for (i = 0; argv[i]; i++)
        if (!strcmp(argv[i], s)) return 1;
    return 0;
}

int main(void) {
    h3_ffmpeg_enc_opts opts;
    char *argv[64];
    char scratch[192];
    int n;
    memset(&opts, 0, sizeof(opts));
    opts.width = 864;
    opts.height = 480;
    opts.fps = 24;
    opts.sample_rate = 32000;
    opts.channels = 2;
    opts.has_audio = 1;
    opts.path = "/tmp/h3_args.mp4";
    unsetenv("H3_FFMPEG_THREADS");
    unsetenv("H3_FFMPEG_VENC");
    unsetenv("H3_X264_PRESET");
    unsetenv("H3_X264_PRESET_TIER");
    unsetenv("H3_X264_CRF");
    n = h3_ffmpeg_build_encode_argv(&opts, argv, 64, scratch, sizeof(scratch));
    expect(n > 10, "argc");
    expect(argv_has(argv, "-c:v"), "-c:v");
    expect(argv_has(argv, "libx264"), "default libx264");
    expect(argv_has(argv, "veryfast"), "default x264 veryfast (quality tier)");
    expect(argv_has(argv, "18"), "480p default crf 18");
    expect(!argv_has(argv, "-threads"), "no threads default");
    opts.width = 1344;
    opts.height = 768;
    n = h3_ffmpeg_build_encode_argv(&opts, argv, 64, scratch, sizeof(scratch));
    expect(argv_has(argv, "14"), "hires default crf 14");
    setenv("H3_X264_CRF", "12", 1);
    n = h3_ffmpeg_build_encode_argv(&opts, argv, 64, scratch, sizeof(scratch));
    expect(argv_has(argv, "12"), "H3_X264_CRF overrides hires default");
    unsetenv("H3_X264_CRF");
    opts.width = 864;
    opts.height = 480;
    setenv("H3_X264_PRESET_TIER", "preview", 1);
    n = h3_ffmpeg_build_encode_argv(&opts, argv, 64, scratch, sizeof(scratch));
    expect(argv_has(argv, "ultrafast"), "preview tier ultrafast");
    setenv("H3_X264_PRESET", "medium", 1);
    n = h3_ffmpeg_build_encode_argv(&opts, argv, 64, scratch, sizeof(scratch));
    expect(argv_has(argv, "medium") && !argv_has(argv, "ultrafast"),
           "explicit H3_X264_PRESET wins over tier");
    unsetenv("H3_X264_PRESET");
    unsetenv("H3_X264_PRESET_TIER");
    setenv("H3_FFMPEG_THREADS", "4", 1);
    n = h3_ffmpeg_build_encode_argv(&opts, argv, 64, scratch, sizeof(scratch));
    expect(n > 10 && argv_has(argv, "-threads") && argv_has(argv, "4"),
           "threads 4");
    setenv("H3_FFMPEG_VENC", "h264_nvenc", 1);
    n = h3_ffmpeg_build_encode_argv(&opts, argv, 64, scratch, sizeof(scratch));
    expect(n > 10 && argv_has(argv, "h264_nvenc"), "nvenc");
    unsetenv("H3_FFMPEG_THREADS");
    unsetenv("H3_FFMPEG_VENC");
    return fails ? 1 : (printf("test_h3_ffmpeg_args: PASS\n"), 0);
}
