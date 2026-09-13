#include "h3_cuda_device_caps.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int h3_caps_probe_lib(const char *env_key, const char *default_soname,
                             char *tried, size_t tried_n) {
    const char *path = getenv(env_key);
    void *h;
    if (!path || !path[0]) path = default_soname;
    if (tried && tried_n) {
        snprintf(tried, tried_n, "%s", path ? path : "(null)");
    }
    if (!path || !path[0]) return 0;
    h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!h) return 0;
    dlclose(h);
    return 1;
}

int h3_cuda_query_caps(int device_index, h3_cuda_device_caps *out, char *error,
                       size_t error_size) {
    int major;
    int minor;
    if (!out) {
        if (error && error_size)
            snprintf(error, error_size, "h3_cuda_query_caps: null out");
        return 0;
    }
    memset(out, 0, sizeof(*out));
    if (!h3_cuda_probe(device_index, &out->info, error, error_size)) return 0;

    major = out->info.runtime_version / 1000;
    minor = (out->info.runtime_version % 1000) / 10;
    out->cuda_runtime_major = major;
    out->cuda_runtime_minor = minor;

    /* Fatbin typically ships 89 (+120 when toolchain allows). Best-effort. */
    out->arch_bin_ok = 1;
    if (out->info.compute_major == 12 && out->info.compute_minor == 0 &&
        major < 13) {
        out->arch_bin_ok = 0; /* 5090 needs CUDA 13+ runtime */
    }

    /* NVFP4 Tensor Core: Blackwell sm_120+. Capability only — never auto sage3. */
    out->has_fp4_tc =
        (out->info.compute_major > 12) ||
        (out->info.compute_major == 12 && out->info.compute_minor >= 0);

#if defined(H3_HAVE_CUDNN)
    out->has_cudnn_sdpa = 1;
#else
    out->has_cudnn_sdpa = 0;
#endif

    out->has_sage3_plugin = h3_caps_probe_lib(
        "H3_SAGE3_LIB", "libh3_sage3.so", out->sage3_lib_tried,
        sizeof(out->sage3_lib_tried));
    out->has_sol_cute_plugin = h3_caps_probe_lib(
        "H3_SOL_CUTE_LIB", "libh3_sol_cute.so", out->sol_cute_lib_tried,
        sizeof(out->sol_cute_lib_tried));
    return 1;
}

void h3_cuda_caps_format(const h3_cuda_device_caps *caps, char *buf,
                         size_t buf_size) {
    if (!buf || !buf_size) return;
    if (!caps) {
        snprintf(buf, buf_size, "(null caps)");
        return;
    }
    snprintf(buf, buf_size,
             "dev=%d sm_%d%d cuda=%d.%d fp4=%d cudnn_sdpa=%d sage3_lib=%d "
             "sol_cute_lib=%d arch_ok=%d name=%s",
             caps->info.device_index, caps->info.compute_major,
             caps->info.compute_minor, caps->cuda_runtime_major,
             caps->cuda_runtime_minor, caps->has_fp4_tc, caps->has_cudnn_sdpa,
             caps->has_sage3_plugin, caps->has_sol_cute_plugin,
             caps->arch_bin_ok, caps->info.name);
}
