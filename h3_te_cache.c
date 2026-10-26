#include "h3_te_cache.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static const char k_magic[8] = {'H', '3', 'T', 'E', '0', '1', '\n', 0};

static int tc_fail(char *err, size_t err_size, const char *msg) {
    if (err && err_size)
        snprintf(err, err_size, "%s", msg ? msg : "te_cache error");
    return 0;
}

static uint64_t fnv1a_update(uint64_t h, const void *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    size_t i;
    for (i = 0; i < n; i++) {
        h ^= (uint64_t)b[i];
        h *= 1099511628211ull;
    }
    return h;
}

int h3_te_cache_key(const int32_t *ids, size_t n, unsigned te_layers,
                    const char *te_path, char *hex, size_t hex_size,
                    char *err, size_t err_size) {
    uint64_t h = 14695981039346656037ull;
    struct stat st;
    uint64_t meta[4];
    unsigned i;
    if (!ids || n < 1 || !hex || hex_size < 17)
        return tc_fail(err, err_size, "te_cache: bad key args");
    memset(meta, 0, sizeof(meta));
    meta[0] = (uint64_t)n;
    meta[1] = (uint64_t)te_layers;
    meta[2] = (uint64_t)H3_TE_CACHE_TEXT_DIM; /* C062: dim in key */
    meta[3] = 0;
    if (te_path && te_path[0] && stat(te_path, &st) == 0) {
        meta[3] = (uint64_t)st.st_mtime ^ ((uint64_t)st.st_size << 1);
        h = fnv1a_update(h, te_path, strlen(te_path));
    } else if (te_path && te_path[0]) {
        h = fnv1a_update(h, te_path, strlen(te_path));
    }
    h = fnv1a_update(h, meta, sizeof(meta));
    h = fnv1a_update(h, ids, n * sizeof(int32_t));
    for (i = 0; i < 16 && (size_t)i + 1 < hex_size; i++) {
        static const char *d = "0123456789abcdef";
        unsigned shift = (15u - i) * 4u;
        hex[i] = d[(h >> shift) & 0xfu];
    }
    hex[i] = '\0';
    return 1;
}

static int cache_path(const char *dir, const char *hex, char *out, size_t n) {
    if (!dir || !dir[0] || !hex || !hex[0])
        return 0;
    snprintf(out, n, "%s/h3te_%s.bin", dir, hex);
    return 1;
}

int h3_te_cache_load(const char *dir, const char *hex, size_t tokens,
                     unsigned te_layers, float **out, char *err,
                     size_t err_size) {
    char path[768];
    FILE *fp;
    char magic[8];
    uint32_t ver = 0, layers = 0, dim = 0, reserved = 0;
    uint64_t tok = 0;
    float *emb;
    size_t n;
    if (!out)
        return tc_fail(err, err_size, "te_cache: null out");
    *out = NULL;
    if (!cache_path(dir, hex, path, sizeof(path)))
        return tc_fail(err, err_size, "te_cache: bad dir/key");
    fp = fopen(path, "rb");
    if (!fp)
        return 0;
    if (fread(magic, 1, 8, fp) != 8 || memcmp(magic, k_magic, 8) != 0 ||
        fread(&ver, 4, 1, fp) != 1 || ver != 1 ||
        fread(&layers, 4, 1, fp) != 1 || layers != te_layers ||
        fread(&tok, 8, 1, fp) != 1 || tok != (uint64_t)tokens ||
        fread(&dim, 4, 1, fp) != 1 || dim != (uint32_t)H3_TE_CACHE_TEXT_DIM ||
        fread(&reserved, 4, 1, fp) != 1) {
        fclose(fp);
        return tc_fail(err, err_size, "te_cache: header mismatch");
    }
    n = tokens * (size_t)H3_TE_CACHE_TEXT_DIM;
    emb = (float *)malloc(n * sizeof(float));
    if (!emb) {
        fclose(fp);
        return tc_fail(err, err_size, "te_cache: OOM");
    }
    if (fread(emb, sizeof(float), n, fp) != n) {
        free(emb);
        fclose(fp);
        return tc_fail(err, err_size, "te_cache: short read");
    }
    fclose(fp);
    *out = emb;
    return 1;
}

int h3_te_cache_save(const char *dir, const char *hex, const float *emb,
                     size_t tokens, unsigned te_layers, char *err,
                     size_t err_size) {
    char path[768];
    FILE *fp;
    uint32_t ver = 1, layers = te_layers, dim = (uint32_t)H3_TE_CACHE_TEXT_DIM;
    uint32_t reserved = 0;
    uint64_t tok = (uint64_t)tokens;
    size_t n;
    if (!emb || tokens < 1)
        return tc_fail(err, err_size, "te_cache: bad save args");
    if (!dir || !dir[0])
        return 1;
    if (mkdir(dir, 0755) != 0) {
        struct stat st;
        if (stat(dir, &st) != 0 || !S_ISDIR(st.st_mode))
            return tc_fail(err, err_size, "te_cache: cannot mkdir");
    }
    if (!cache_path(dir, hex, path, sizeof(path)))
        return tc_fail(err, err_size, "te_cache: bad dir/key");
    fp = fopen(path, "wb");
    if (!fp)
        return tc_fail(err, err_size, "te_cache: cannot write");
    n = tokens * (size_t)H3_TE_CACHE_TEXT_DIM;
    if (fwrite(k_magic, 1, 8, fp) != 8 || fwrite(&ver, 4, 1, fp) != 1 ||
        fwrite(&layers, 4, 1, fp) != 1 || fwrite(&tok, 8, 1, fp) != 1 ||
        fwrite(&dim, 4, 1, fp) != 1 || fwrite(&reserved, 4, 1, fp) != 1 ||
        fwrite(emb, sizeof(float), n, fp) != n) {
        fclose(fp);
        return tc_fail(err, err_size, "te_cache: short write");
    }
    fclose(fp);
    return 1;
}
