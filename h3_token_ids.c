#include "h3_token_ids.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define H3_PAD_TOKEN_ID_LOCAL 151643

static void fail(char *error, size_t error_size, const char *msg) {
    if (error && error_size) {
        snprintf(error, error_size, "%s", msg ? msg : "token_ids error");
    }
}

static int push_id(int32_t **ids, size_t *count, size_t *cap, long v,
                   char *error, size_t error_size) {
    int32_t *nbuf;
    if (v < 0 || v > 2147483647L) {
        fail(error, error_size, "token id out of range");
        return 0;
    }
    if (*count >= *cap) {
        size_t ncap = *cap ? *cap * 2 : 64;
        nbuf = (int32_t *)realloc(*ids, ncap * sizeof(int32_t));
        if (!nbuf) {
            fail(error, error_size, "OOM token ids");
            return 0;
        }
        *ids = nbuf;
        *cap = ncap;
    }
    (*ids)[(*count)++] = (int32_t)v;
    return 1;
}

static int parse_stream(const char *text, int32_t **ids, size_t *count,
                        char *error, size_t error_size) {
    const char *p = text;
    size_t cap = 0;
    *ids = NULL;
    *count = 0;
    if (!text) {
        fail(error, error_size, "null token list");
        return 0;
    }
    while (*p) {
        char *end = NULL;
        long v;
        while (*p && (isspace((unsigned char)*p) || *p == ',' || *p == ';'))
            p++;
        if (!*p) break;
        v = strtol(p, &end, 10);
        if (end == p) {
            fail(error, error_size, "invalid token id token");
            free(*ids);
            *ids = NULL;
            *count = 0;
            return 0;
        }
        if (!push_id(ids, count, &cap, v, error, error_size)) {
            free(*ids);
            *ids = NULL;
            *count = 0;
            return 0;
        }
        p = end;
    }
    if (*count == 0) {
        fail(error, error_size, "empty token id list");
        free(*ids);
        *ids = NULL;
        return 0;
    }
    return 1;
}

int h3_token_ids_parse_list(const char *text, int32_t **ids, size_t *count,
                            char *error, size_t error_size) {
    return parse_stream(text, ids, count, error, error_size);
}

int h3_token_ids_load_file(const char *path, int32_t **ids, size_t *count,
                           char *error, size_t error_size) {
    FILE *fp;
    long sz;
    char *buf = NULL;
    int ok;
    if (!path || !*path) {
        fail(error, error_size, "token-ids path empty");
        return 0;
    }
    fp = fopen(path, "rb");
    if (!fp) {
        fail(error, error_size, "cannot open token-ids file");
        return 0;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        fail(error, error_size, "token-ids seek failed");
        return 0;
    }
    sz = ftell(fp);
    if (sz < 0 || sz > 8 * 1024 * 1024) {
        fclose(fp);
        fail(error, error_size, "token-ids file too large/invalid");
        return 0;
    }
    rewind(fp);
    buf = (char *)malloc((size_t)sz + 1);
    if (!buf) {
        fclose(fp);
        fail(error, error_size, "OOM reading token-ids");
        return 0;
    }
    if (fread(buf, 1, (size_t)sz, fp) != (size_t)sz) {
        free(buf);
        fclose(fp);
        fail(error, error_size, "token-ids read failed");
        return 0;
    }
    fclose(fp);
    buf[sz] = '\0';
    ok = parse_stream(buf, ids, count, error, error_size);
    free(buf);
    return ok;
}

int h3_token_ids_canary(const char *name, int32_t **ids, size_t *count,
                        char *error, size_t error_size) {
    /* Matches tests/test_tokenizer.c against released Qwen vocabulary. */
    static const int32_t fox[] = {32, 2518, 38835, 11435, 1526, 11794};
    static const int32_t cinematic[] = {32, 64665, 3265, 5239, 315, 264, 8866,
                                        1778, 11958, 4633, 10971, 13};
    static const int32_t im_start[] = {151644};
    const int32_t *src = fox;
    size_t n = sizeof(fox) / sizeof(fox[0]);
    int32_t *out;
    size_t i;

    if (name && *name) {
        if (!strcmp(name, "fox")) {
            src = fox;
            n = sizeof(fox) / sizeof(fox[0]);
        } else if (!strcmp(name, "cinematic")) {
            src = cinematic;
            n = sizeof(cinematic) / sizeof(cinematic[0]);
        } else if (!strcmp(name, "im_start")) {
            src = im_start;
            n = sizeof(im_start) / sizeof(im_start[0]);
        } else {
            fail(error, error_size, "unknown canary (fox|cinematic|im_start)");
            return 0;
        }
    }
    out = (int32_t *)malloc(n * sizeof(int32_t));
    if (!out) {
        fail(error, error_size, "OOM canary ids");
        return 0;
    }
    for (i = 0; i < n; i++) out[i] = src[i];
    *ids = out;
    *count = n;
    return 1;
}

int h3_token_ids_fit(int32_t **ids, size_t *count, size_t want, char *error,
                     size_t error_size) {
    int32_t *nbuf;
    size_t i;
    if (!ids || !*ids || !count || want == 0) {
        fail(error, error_size, "invalid token_ids_fit args");
        return 0;
    }
    if (*count == want) return 1;
    nbuf = (int32_t *)realloc(*ids, want * sizeof(int32_t));
    if (!nbuf) {
        fail(error, error_size, "OOM token_ids_fit");
        return 0;
    }
    *ids = nbuf;
    if (*count < want) {
        for (i = *count; i < want; i++)
            nbuf[i] = (int32_t)H3_PAD_TOKEN_ID_LOCAL;
    }
    *count = want;
    return 1;
}
