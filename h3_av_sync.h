#ifndef H3_AV_SYNC_H
#define H3_AV_SYNC_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double video_start_s;
    double video_duration_s;
    double audio_start_s;
    double audio_duration_s;
    int sample_rate;
    int channels;
    int width;
    int height;
} h3_av_probe;

/* 1 if video duration is within tol_s of expect_s and |v-a| <= tol_s. */
int h3_av_sync_check(const h3_av_probe *p, double expect_s, double tol_s,
                     char *why, size_t n);

/* H3_AV_SYNC_GATE=0 → warn only. Default 1. */
int h3_av_sync_gate_wanted(void);

/* Probe path with ffprobe, print av_sync: … PASS|FAIL.
 * Returns 1 on PASS (or gate off). 0 on FAIL with gate on. */
int h3_av_sync_log_file(const char *path, double expect_s, double tol_s,
                        char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
