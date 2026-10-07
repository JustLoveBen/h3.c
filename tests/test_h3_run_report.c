#include "h3_run_report.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int fails;
static void expect(int c, const char *m) {
    if (!c) {
        fprintf(stderr, "FAIL %s\n", m);
        fails++;
    }
}

int main(void) {
    h3_run_report_in in;
    char err[128];
    char buf[2048];
    FILE *fp;
    const char *path = "/tmp/h3_run_report_test.json";
    memset(&in, 0, sizeof(in));
    in.path = path;
    in.preset = "turbo4";
    in.sdpa_backend = "cudnn";
    in.mode = "t2va";
    in.quant = "int8_convrot";
    in.width = 864;
    in.height = 480;
    in.seconds = 5;
    in.hot = 1;
    in.clip = 2;
    in.clip_n = 2;
    in.te_ms = 0.0;
    in.dit_ms = 100.0;
    in.vae_load_ms = 0.0;
    in.vae_blocks_ms = 50.0;
    in.audio_ms = 10.0;
    in.mux_ms = 20.0;
    in.total_ms = 200.0;
    expect(h3_run_report_write(&in, err, sizeof(err)) == 1, "write");
    fp = fopen(path, "r");
    expect(fp != NULL, "open json");
    if (fp) {
        size_t n = fread(buf, 1, sizeof(buf) - 1, fp);
        buf[n] = '\0';
        fclose(fp);
        expect(strstr(buf, "\"phase_ms\"") != NULL, "phase_ms key");
        expect(strstr(buf, "\"vae_load\"") != NULL, "vae_load key");
        expect(strstr(buf, "\"sdpa_backend\"") != NULL, "sdpa key");
        expect(strstr(buf, "\"vae_attn_backend\"") != NULL, "vae attn key");
        expect(strstr(buf, "\"total\"") != NULL, "total key");
        expect(strstr(buf, "HOME") == NULL, "no extra env");
    }
    unlink(path);
    return fails ? 1 : (printf("test_h3_run_report: PASS\n"), 0);
}
