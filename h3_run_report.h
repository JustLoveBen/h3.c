#ifndef H3_RUN_REPORT_H
#define H3_RUN_REPORT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char *path;
    const char *preset;
    const char *sdpa_backend;
    const char *vae_attn_backend;
    const char *mode;
    const char *quant;
    int width;
    int height;
    int seconds;
    int hot;
    int clip;
    int clip_n;
    double te_ms;
    double dit_ms;
    double vae_load_ms;
    double vae_blocks_ms;
    double audio_ms;
    double mux_ms;
    double total_ms;
} h3_run_report_in;

/* Write a small KPI JSON. Missing path → 0. */
int h3_run_report_write(const h3_run_report_in *in, char *error,
                        size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
