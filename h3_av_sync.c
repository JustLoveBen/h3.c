#include "h3_av_sync.h"
#include "h3_ffmpeg.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int h3_av_sync_gate_wanted(void) {
    const char *e = getenv("H3_AV_SYNC_GATE");
    if (e && e[0] == '0' && e[1] == '\0') return 0;
    return 1;
}

int h3_av_sync_check(const h3_av_probe *p, double expect_s, double tol_s,
                     char *why, size_t n) {
    double drift;
    if (why && n) why[0] = '\0';
    if (!p || expect_s <= 0.0 || tol_s < 0.0) {
        if (why && n) snprintf(why, n, "invalid av_sync args");
        return 0;
    }
    if (p->width < 1 || p->height < 1 || p->video_duration_s <= 0.0) {
        if (why && n) snprintf(why, n, "missing video stream");
        return 0;
    }
    if (p->audio_duration_s <= 0.0 || p->sample_rate < 1 || p->channels < 1) {
        if (why && n) snprintf(why, n, "missing audio stream");
        return 0;
    }
    if (fabs(p->video_duration_s - expect_s) > tol_s) {
        if (why && n)
            snprintf(why, n, "video duration %.3f != expect %.3f ±%.3f",
                     p->video_duration_s, expect_s, tol_s);
        return 0;
    }
    drift = (p->video_duration_s - p->audio_duration_s) * 1000.0;
    if (fabs(p->video_duration_s - p->audio_duration_s) > tol_s) {
        if (why && n)
            snprintf(why, n, "av drift %.1f ms > %.1f ms", drift, tol_s * 1000.0);
        return 0;
    }
    if (why && n) snprintf(why, n, "ok");
    return 1;
}

int h3_av_sync_log_file(const char *path, double expect_s, double tol_s,
                        char *error, size_t error_size) {
    h3_av_probe p;
    char why[256];
    int ok;
    double drift;
    memset(&p, 0, sizeof(p));
    if (error && error_size) error[0] = '\0';
    if (!path || !path[0]) {
        if (error && error_size) snprintf(error, error_size, "av_sync: no path");
        return 0;
    }
    if (!h3_ffprobe_av_streams(path, &p, why, sizeof(why))) {
        printf("av_sync: video=0.000 audio=0.000 drift_ms=0.0 FAIL (%s)\n", why);
        fflush(stdout);
        if (error && error_size) snprintf(error, error_size, "%s", why);
        return h3_av_sync_gate_wanted() ? 0 : 1;
    }
    ok = h3_av_sync_check(&p, expect_s, tol_s, why, sizeof(why));
    drift = (p.video_duration_s - p.audio_duration_s) * 1000.0;
    printf("av_sync: video=%.3f audio=%.3f drift_ms=%.1f %s%s%s\n",
           p.video_duration_s, p.audio_duration_s, drift, ok ? "PASS" : "FAIL",
           ok ? "" : " reason=", ok ? "" : why);
    fflush(stdout);
    if (!ok && error && error_size) snprintf(error, error_size, "%s", why);
    if (ok) return 1;
    return h3_av_sync_gate_wanted() ? 0 : 1;
}
