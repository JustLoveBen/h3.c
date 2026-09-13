#ifndef H3_CUDA_DEVICE_CAPS_H
#define H3_CUDA_DEVICE_CAPS_H

#include "h3_gpu_cuda.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* One-shot device capability snapshot for SDPA / plugin selection.
 * Generate / DiT must not branch on sm==120; ask caps + registry instead. */

typedef struct {
    h3_cuda_device_info info;
    int cuda_runtime_major; /* e.g. 13 → Blackwell path legal */
    int cuda_runtime_minor;
    int arch_bin_ok;        /* 1 if we assume fatbin covers this SM (best-effort) */
    int has_fp4_tc;         /* sm_120+ capability bit; does NOT auto-enable sage3 */
    int has_cudnn_sdpa;     /* compile-time + runtime handle may still fail later */
    int has_sage3_plugin;   /* dlopen probe of H3_SAGE3_LIB / libh3_sage3.so */
    int has_sol_cute_plugin;
    char sage3_lib_tried[512];
    char sol_cute_lib_tried[512];
} h3_cuda_device_caps;

/* Probe device + optional plugin .so presence (dlopen, then dlclose).
 * Missing plugins set has_*_plugin=0; they are not hard errors. */
int h3_cuda_query_caps(int device_index, h3_cuda_device_caps *out, char *error,
                       size_t error_size);

/* Human-readable one-liner for logs (no hostnames, no secrets). */
void h3_cuda_caps_format(const h3_cuda_device_caps *caps, char *buf,
                         size_t buf_size);

#ifdef __cplusplus
}
#endif

#endif /* H3_CUDA_DEVICE_CAPS_H */
