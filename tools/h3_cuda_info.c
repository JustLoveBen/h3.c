#include "h3_gpu_cuda.h"
#include "h3_cuda_device_caps.h"
#include "h3_sdpa.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

/* h3_gpu_cuda.cu.o pulls ConvRot Hadamard helpers; this probe never uses them. */
float *h3_w4a8_hadamard_normalized(int size, char *error, size_t error_size) {
    (void)size;
    if (error && error_size)
        snprintf(error, error_size, "h3_cuda_info: hadamard unused");
    return NULL;
}

float *h3_w4a8_hadamard_sylvester_normalized(int size, char *error,
                                             size_t error_size) {
    (void)size;
    if (error && error_size)
        snprintf(error, error_size, "h3_cuda_info: hadamard unused");
    return NULL;
}

static const char *h3_cuda_arch_note(int major, int minor) {
    if (major == 8 && minor == 9) return "Ada (sm_89), e.g. RTX 4090";
    if (major == 12 && minor == 0) return "Blackwell (sm_120), e.g. RTX 5090 class";
    if (major >= 12) return "Blackwell-or-newer family";
    if (major == 8) return "Ampere/Ada family";
    if (major == 9) return "Hopper family";
    return "see CUDA compute capability docs";
}

static void h3_cuda_print_version(const char *label, int version) {
    if (version <= 0) {
        printf("%s: unknown\n", label);
        return;
    }
    printf("%s: %d.%d\n", label, version / 1000, (version % 1000) / 10);
}

static void h3_cuda_print_env_path(const char *key) {
    const char *v = getenv(key);
    if (!v || !v[0]) {
        printf("  %s: unset\n", key);
        return;
    }
    printf("  %s: set path=%s exists=%d\n", key, v, access(v, F_OK) == 0);
}

static void h3_cuda_print_plugin_inventory(void) {
    printf("plugin inventory (so presence ≠ kernel; see "
           "tools/h3_sol_sage3_inventory.py):\n");
    h3_cuda_print_env_path("H3_SAGE3_LIB");
    h3_cuda_print_env_path("H3_SAGE3_ROOT");
    h3_cuda_print_env_path("H3_SAGE3_TOOLS");
    h3_cuda_print_env_path("H3_SOL_CUTE_LIB");
    h3_cuda_print_env_path("H3_SOL_ATTN_ROOT");
    h3_cuda_print_env_path("H3_SOL_CUTE_TOOLS");
    h3_cuda_print_env_path("H3_TRITON_OVERLAY");
    printf("  official CuTe names: cute_sm89 cute_sm90 cute_sm100 cute_sm120 "
           "(5090=cute_sm120); triton=fake\n");
    printf("  cute_sm120 official=%d triton official=%d\n",
           h3_sdpa_sol_cute_is_official("cute_sm120"),
           h3_sdpa_sol_cute_is_official("triton"));
}

int main(void) {
    char error[512];
    int count;
    int index;

    count = h3_cuda_device_count(error, sizeof(error));
    if (count < 0) {
        fprintf(stderr, "h3_cuda_info: %s\n", error);
        fprintf(stderr,
                "hint: install a CUDA Toolkit with nvcc, then reconfigure with "
                "-DH3_GPU_BACKEND=cuda (see docs/nvidia_build.md)\n");
        return 1;
    }
    if (count == 0) {
        fprintf(stderr, "h3_cuda_info: no CUDA devices visible\n");
        return 1;
    }

    printf("CUDA devices: %d\n", count);
    {
        const char *req = getenv("H3_CUDA_SDPA_BACKEND");
        h3_sdpa_backend_id id;
        int avail = 0;
        char reason[320];
        h3_sdpa_resolve(req && req[0] ? req : "auto", NULL, &id, &avail, reason,
                        sizeof(reason));
        printf("SDPA registry: request=%s → %s (auto always Tier-0 cudnn; "
               "Tier-1 = sage3|sol_cute)\n",
               (req && req[0]) ? req : "auto", h3_sdpa_backend_id_name(id));
        printf("  resolve: %s\n", reason);
    }
    h3_cuda_print_plugin_inventory();
    for (index = 0; index < count; index++) {
        h3_cuda_device_info info;
        h3_cuda_device_caps caps;
        char caps_line[384];
        h3_sdpa_backend_id id;
        int avail = 0;
        char reason[320];
        const char *req = getenv("H3_CUDA_SDPA_BACKEND");
        if (!h3_cuda_probe(index, &info, error, sizeof(error))) {
            fprintf(stderr, "h3_cuda_info: device %d: %s\n", index, error);
            return 1;
        }
        printf("\n[%d] %s\n", info.device_index, info.name);
        printf("  compute capability: sm_%d%d (%s)\n",
               info.compute_major, info.compute_minor,
               h3_cuda_arch_note(info.compute_major, info.compute_minor));
        printf("  VRAM total: %.2f GiB\n",
               (double)info.total_memory_bytes / (1024.0 * 1024.0 * 1024.0));
        printf("  VRAM free:  %.2f GiB\n",
               (double)info.free_memory_bytes / (1024.0 * 1024.0 * 1024.0));
        printf("  SMs: %d\n", info.multiprocessor_count);
        h3_cuda_print_version("  driver", info.driver_version);
        h3_cuda_print_version("  runtime", info.runtime_version);
        if (h3_cuda_query_caps(index, &caps, error, sizeof(error))) {
            h3_cuda_caps_format(&caps, caps_line, sizeof(caps_line));
            printf("  caps: %s\n", caps_line);
            h3_sdpa_resolve(req && req[0] ? req : "auto", &caps, &id, &avail,
                            reason, sizeof(reason));
            printf("  sdpa_select: %s available=%d (%s)\n",
                   h3_sdpa_backend_id_name(id), avail, reason);
            if (caps.has_fp4_tc && !req)
                printf("  note: FP4 capable — sage3 is Tier-1 opt-in, not auto\n");
            if (info.compute_major == 8 && info.compute_minor == 9)
                printf("  note: Ada sm_89 — prefer sol_cute for Tier-1 accel\n");
        } else {
            printf("  caps: probe failed: %s\n", error);
        }
    }
    return 0;
}
