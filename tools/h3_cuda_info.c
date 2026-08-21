#include "h3_gpu_cuda.h"

#include <stdio.h>

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
    for (index = 0; index < count; index++) {
        h3_cuda_device_info info;
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
    }
    return 0;
}
