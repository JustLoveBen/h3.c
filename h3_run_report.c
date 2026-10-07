#include "h3_run_report.h"

#include <stdio.h>
#include <string.h>

static void json_esc(FILE *fp, const char *s) {
    if (!s) {
        fputs("", fp);
        return;
    }
    for (; *s; s++) {
        if (*s == '"' || *s == '\\') {
            fputc('\\', fp);
            fputc(*s, fp);
        } else if ((unsigned char)*s < 32)
            fprintf(fp, "\\u%04x", (unsigned char)*s);
        else
            fputc(*s, fp);
    }
}

int h3_run_report_write(const h3_run_report_in *in, char *error,
                        size_t error_size) {
    FILE *fp;
    if (error && error_size) error[0] = '\0';
    if (!in || !in->path || !in->path[0]) {
        if (error && error_size) snprintf(error, error_size, "report: no path");
        return 0;
    }
    fp = fopen(in->path, "w");
    if (!fp) {
        if (error && error_size)
            snprintf(error, error_size, "report: cannot write %s", in->path);
        return 0;
    }
    fputs("{\n", fp);
    fputs("  \"phase_ms\": {", fp);
    fprintf(fp,
            "\"te\": %.1f, \"dit\": %.1f, \"vae_load\": %.1f, "
            "\"vae_blocks\": %.1f, \"audio\": %.1f, \"mux\": %.1f, "
            "\"total\": %.1f",
            in->te_ms, in->dit_ms, in->vae_load_ms, in->vae_blocks_ms,
            in->audio_ms, in->mux_ms, in->total_ms);
    fputs("},\n", fp);
    fputs("  \"sdpa_backend\": \"", fp);
    json_esc(fp, in->sdpa_backend ? in->sdpa_backend : "");
    fputs("\",\n", fp);
    fputs("  \"vae_attn_backend\": \"", fp);
    json_esc(fp, in->vae_attn_backend ? in->vae_attn_backend : "");
    fputs("\",\n", fp);
    fputs("  \"preset\": \"", fp);
    json_esc(fp, in->preset ? in->preset : "");
    fputs("\",\n", fp);
    fputs("  \"mode\": \"", fp);
    json_esc(fp, in->mode ? in->mode : "t2va");
    fputs("\",\n", fp);
    fputs("  \"quant\": \"", fp);
    json_esc(fp, in->quant ? in->quant : "int8_convrot");
    fprintf(fp,
            "\",\n  \"width\": %d,\n  \"height\": %d,\n  \"seconds\": %d,\n"
            "  \"hot\": %d,\n  \"clip\": %d,\n  \"clip_n\": %d\n}\n",
            in->width, in->height, in->seconds, in->hot, in->clip, in->clip_n);
    fclose(fp);
    return 1;
}
