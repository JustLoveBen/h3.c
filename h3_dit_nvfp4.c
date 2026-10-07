#include "h3_dit_nvfp4.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static void fail(char *error, size_t error_size, const char *msg) {
    if (!error || !error_size) return;
    snprintf(error, error_size, "%s", msg ? msg : "dit nvfp4 error");
}

int h3_dit_nvfp4_path_hints(const char *path) {
    if (!path || !path[0]) return 0;
    if (strstr(path, "nvfp4") || strstr(path, "NVFP4") ||
        strstr(path, "Nvfp4"))
        return 1;
    return 0;
}

int h3_dit_nvfp4_find_block_linear(const h3_nvfp4_store *store,
                                   unsigned block_index, const char *suffix,
                                   h3_nvfp4_linear *out, char *error,
                                   size_t error_size) {
    char base[192];
    if (!suffix) {
        fail(error, error_size, "NVFP4 DiT suffix required");
        return 0;
    }
    snprintf(base, sizeof(base), "blocks.%u.%s", block_index, suffix);
    return h3_nvfp4_find_linear(store, base, out, error, error_size);
}

int h3_dit_nvfp4_probe(const h3_nvfp4_store *store, char *error,
                       size_t error_size) {
    h3_nvfp4_linear qkv;
    char detail[384];
    if (!store) {
        fail(error, error_size,
             "NVFP4 DiT probe failed（不是金线；不回退 INT8）: null store");
        return 0;
    }
    if (!h3_dit_nvfp4_find_block_linear(store, 0, H3_DIT_NVFP4_SUFFIX_QKV, &qkv,
                                        detail, sizeof(detail))) {
        snprintf(error, error_size,
                 "NVFP4 DiT probe failed（不是金线；不回退 INT8）: %s",
                 detail);
        return 0;
    }
    if (qkv.n != 21504 || qkv.k != 5376) {
        snprintf(error, error_size,
                 "NVFP4 DiT probe failed（不是金线；不回退 INT8）: "
                 "qkv shape N=%zu K=%zu want 21504x5376",
                 qkv.n, qkv.k);
        return 0;
    }
    if (!qkv.weight || qkv.weight->dtype != H3_DTYPE_U8 || !qkv.scale ||
        qkv.scale->dtype != H3_DTYPE_F8_E4M3) {
        fail(error, error_size,
             "NVFP4 DiT probe failed（不是金线；不回退 INT8）: "
             "need U8 weight + F8_E4M3 scale");
        return 0;
    }
    fprintf(stderr,
            "dit_nvfp4_probe: tensors=%zu qkv=U8[%zu,%zu] scale=F8[%zu,%zu] "
            "gs=%zu (非金线 dequant→INT8 GEMM)\n",
            store->header.tensor_count, qkv.n, qkv.k, qkv.n, qkv.groups,
            qkv.group_size);
    fflush(stderr);
    return 1;
}

int h3_dit_nvfp4_dequant_to_i8(const h3_nvfp4_store *store,
                               const h3_nvfp4_linear *lin, int8_t **out_i8,
                               float **out_scale, size_t *out_n, size_t *out_k,
                               char *error, size_t error_size) {
    float *f32 = NULL;
    int8_t *i8 = NULL;
    float *scales = NULL;
    size_t rows = 0, n, k, r, c;
    if (out_i8) *out_i8 = NULL;
    if (out_scale) *out_scale = NULL;
    if (!store || !lin || !out_i8 || !out_scale) {
        fail(error, error_size, "invalid nvfp4 dequant_to_i8 args");
        return 0;
    }
    f32 = h3_nvfp4_dequant_rows_f32(store, lin, 0, &rows, error, error_size);
    if (!f32) return 0;
    n = rows;
    k = lin->k;
    i8 = (int8_t *)malloc(n * k);
    scales = (float *)malloc(n * sizeof(float));
    if (!i8 || !scales) {
        free(f32);
        free(i8);
        free(scales);
        fail(error, error_size, "OOM nvfp4 requant INT8");
        return 0;
    }
    for (r = 0; r < n; r++) {
        const float *row = f32 + r * k;
        float amax = 0.0f;
        float s;
        for (c = 0; c < k; c++) {
            float a = fabsf(row[c]);
            if (a > amax) amax = a;
        }
        s = amax / 127.0f;
        if (!(s > 0.0f) || !isfinite(s)) s = 1.0f;
        scales[r] = s;
        for (c = 0; c < k; c++) {
            float q = nearbyintf(row[c] / s);
            if (q > 127.0f) q = 127.0f;
            if (q < -127.0f) q = -127.0f;
            i8[r * k + c] = (int8_t)q;
        }
    }
    free(f32);
    *out_i8 = i8;
    *out_scale = scales;
    if (out_n) *out_n = n;
    if (out_k) *out_k = k;
    return 1;
}

#define H3_NVFP4_SIDECAR_MAGIC "H3I8SID1"
#define H3_NVFP4_SIDECAR_VER 1u

static int sidecar_disabled(void) {
    const char *e = getenv("H3_NVFP4_INT8_SIDECAR");
    return e && e[0] == '0' && e[1] == '\0';
}

int h3_dit_nvfp4_sidecar_dir(const char *nvfp4_path, char *out_dir, size_t n) {
    const char *env;
    const char *slash;
    if (!nvfp4_path || !nvfp4_path[0] || !out_dir || n < 8) return 0;
    env = getenv("H3_NVFP4_INT8_SIDECAR");
    if (env && env[0] && !(env[0] == '0' && env[1] == '\0') &&
        !(env[0] == '1' && env[1] == '\0')) {
        snprintf(out_dir, n, "%s", env);
        return 1;
    }
    slash = strrchr(nvfp4_path, '/');
    if (slash)
        snprintf(out_dir, n, "%.*s/%s.int8_sidecar", (int)(slash - nvfp4_path),
                 nvfp4_path, slash + 1);
    else
        snprintf(out_dir, n, "%s.int8_sidecar", nvfp4_path);
    return 1;
}

static int sidecar_src_key(const char *nvfp4_path, uint64_t *sz, int64_t *mt) {
    struct stat st;
    if (!nvfp4_path || stat(nvfp4_path, &st) != 0) return 0;
    *sz = (uint64_t)st.st_size;
    *mt = (int64_t)st.st_mtime;
    return 1;
}

static int rw_full(int fd, void *buf, size_t bytes, int do_write) {
    unsigned char *p = (unsigned char *)buf;
    size_t done = 0;
    while (done < bytes) {
        size_t chunk = bytes - done;
        ssize_t n;
        if (chunk > (size_t)(1u << 30))
            chunk = (size_t)(1u << 30);
        n = do_write ? write(fd, p + done, chunk) : read(fd, p + done, chunk);
        if (n <= 0)
            return 0;
        done += (size_t)n;
    }
    return 1;
}

static int sidecar_path(const char *nvfp4_path, unsigned block,
                        const char *kind_name, char *out, size_t n) {
    char dir[768];
    if (!kind_name || !kind_name[0]) return 0;
    if (!h3_dit_nvfp4_sidecar_dir(nvfp4_path, dir, sizeof(dir))) return 0;
    snprintf(out, n, "%s/b%02u_%s.sid", dir, block, kind_name);
    return 1;
}

int h3_dit_nvfp4_sidecar_try_load(const char *nvfp4_path, unsigned block,
                                  const char *kind_name, size_t n, size_t k,
                                  int8_t *i8, float *scales) {
    char path[896];
    char magic[8];
    uint32_t ver = 0;
    uint64_t src_sz = 0, file_sz = 0, nn = 0, kk = 0;
    int64_t src_mt = 0, file_mt = 0;
    int fd;
    size_t w_bytes, s_bytes;
    if (sidecar_disabled() || !i8 || !scales || !n || !k) return 0;
    if (!sidecar_src_key(nvfp4_path, &src_sz, &src_mt)) return 0;
    if (!sidecar_path(nvfp4_path, block, kind_name, path, sizeof(path)))
        return 0;
    fd = open(path, O_RDONLY);
    if (fd < 0) return 0;
    if (read(fd, magic, 8) != 8 || memcmp(magic, H3_NVFP4_SIDECAR_MAGIC, 8) != 0 ||
        read(fd, &ver, 4) != 4 || ver != H3_NVFP4_SIDECAR_VER ||
        read(fd, &file_sz, 8) != 8 || read(fd, &file_mt, 8) != 8 ||
        read(fd, &nn, 8) != 8 || read(fd, &kk, 8) != 8) {
        close(fd);
        return 0;
    }
    if (file_sz != src_sz || file_mt != src_mt || nn != (uint64_t)n ||
        kk != (uint64_t)k) {
        close(fd);
        return 0;
    }
    w_bytes = n * k;
    s_bytes = n * sizeof(float);
    if (!rw_full(fd, i8, w_bytes, 0) || !rw_full(fd, scales, s_bytes, 0)) {
        close(fd);
        return 0;
    }
    close(fd);
    return 1;
}

int h3_dit_nvfp4_sidecar_save(const char *nvfp4_path, unsigned block,
                              const char *kind_name, size_t n, size_t k,
                              const int8_t *i8, const float *scales,
                              char *error, size_t error_size) {
    char dir[768], path[896], tmp[920];
    uint64_t src_sz = 0, nn, kk;
    int64_t src_mt = 0;
    uint32_t ver = H3_NVFP4_SIDECAR_VER;
    int fd;
    size_t w_bytes, s_bytes;
    if (sidecar_disabled()) return 1;
    if (!i8 || !scales || !n || !k) {
        fail(error, error_size, "invalid nvfp4 sidecar save");
        return 0;
    }
    if (!sidecar_src_key(nvfp4_path, &src_sz, &src_mt) ||
        !h3_dit_nvfp4_sidecar_dir(nvfp4_path, dir, sizeof(dir)) ||
        !sidecar_path(nvfp4_path, block, kind_name, path, sizeof(path))) {
        fail(error, error_size, "nvfp4 sidecar path");
        return 0;
    }
    if (mkdir(dir, 0755) != 0 && errno != EEXIST) {
        snprintf(error, error_size, "nvfp4 sidecar mkdir: %s", strerror(errno));
        return 0;
    }
    snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        snprintf(error, error_size, "nvfp4 sidecar open: %s", strerror(errno));
        return 0;
    }
    nn = (uint64_t)n;
    kk = (uint64_t)k;
    w_bytes = n * k;
    s_bytes = n * sizeof(float);
    if (write(fd, H3_NVFP4_SIDECAR_MAGIC, 8) != 8 ||
        write(fd, &ver, 4) != 4 || write(fd, &src_sz, 8) != 8 ||
        write(fd, &src_mt, 8) != 8 || write(fd, &nn, 8) != 8 ||
        write(fd, &kk, 8) != 8 || !rw_full(fd, (void *)i8, w_bytes, 1) ||
        !rw_full(fd, (void *)scales, s_bytes, 1)) {
        close(fd);
        unlink(tmp);
        snprintf(error, error_size, "nvfp4 sidecar write: %s", strerror(errno));
        return 0;
    }
    if (close(fd) != 0) {
        unlink(tmp);
        fail(error, error_size, "nvfp4 sidecar close");
        return 0;
    }
    if (rename(tmp, path) != 0) {
        unlink(tmp);
        snprintf(error, error_size, "nvfp4 sidecar rename: %s", strerror(errno));
        return 0;
    }
    return 1;
}
