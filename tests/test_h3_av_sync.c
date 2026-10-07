#include "h3_av_sync.h"

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

int main(void) {
    h3_av_probe p;
    char why[128];
    memset(&p, 0, sizeof(p));
    p.width = 864;
    p.height = 480;
    p.video_duration_s = 5.000;
    p.audio_duration_s = 5.000;
    p.sample_rate = 32000;
    p.channels = 2;
    expect(h3_av_sync_check(&p, 5.0, 0.05, why, sizeof(why)) == 1, "5.000 PASS");
    p.audio_duration_s = 5.02;
    expect(h3_av_sync_check(&p, 5.0, 0.05, why, sizeof(why)) == 1, "5.02 PASS");
    p.audio_duration_s = 4.9;
    expect(h3_av_sync_check(&p, 5.0, 0.05, why, sizeof(why)) == 0, "4.9 FAIL");
    p.audio_duration_s = 5.0;
    p.video_duration_s = 4.9;
    expect(h3_av_sync_check(&p, 5.0, 0.05, why, sizeof(why)) == 0,
           "video 4.9 FAIL");
    unsetenv("H3_AV_SYNC_GATE");
    expect(h3_av_sync_gate_wanted() == 1, "gate default on");
    setenv("H3_AV_SYNC_GATE", "0", 1);
    expect(h3_av_sync_gate_wanted() == 0, "gate 0 off");
    unsetenv("H3_AV_SYNC_GATE");
    return fails ? 1 : (printf("test_h3_av_sync: PASS\n"), 0);
}
