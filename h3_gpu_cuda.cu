#include "h3_gpu_cuda.h"
#include "h3_sdpa.h"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <dirent.h>
#include <dlfcn.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(H3_HAVE_CUDNN)
#include <cudnn.h>
#endif

/* Avoid including h3_w4a8_dequant.h here (its include chain + extern "C"
 * fights CUDA fp16/bf16 operator overloads under nvcc). */
#ifdef __cplusplus
extern "C" {
#endif
float *h3_w4a8_hadamard_normalized(int size, char *error, size_t error_size);
float *h3_w4a8_hadamard_sylvester_normalized(int size, char *error,
                                             size_t error_size);
#ifdef __cplusplus
}
#endif

enum { H3_CUDA_N_STREAMS = 4 };

/* Comfy/Kijai: contiguous [Q|K|V]. Metal: interleaved via H3_DIT_QKV_GROUPED=1. */
static int h3_cuda_dit_qkv_grouped(void) {
    const char *e = getenv("H3_DIT_QKV_GROUPED");
    return e && e[0] == '1' && e[1] == '\0';
}

/* Default Comfy first/second-half split. H3_DIT_ROPE_HALF_OFFSET=1 or
 * H3_DIT_ROPE_SPLIT_HALF=0 → consecutive-pair rotate (legacy). */
static int h3_cuda_dit_rope_split_half(void) {
    const char *half_off = getenv("H3_DIT_ROPE_HALF_OFFSET");
    const char *e = getenv("H3_DIT_ROPE_SPLIT_HALF");
    if (half_off && half_off[0] == '1' && half_off[1] == '\0') return 0;
    if (e && e[0] == '0' && e[1] == '\0') return 0;
    if (e && e[0] == '1' && e[1] == '\0') return 1;
    return 1;
}

/* H3_SOL_PROFILE=1: millisecond breakdown. Extra EventSynchronize; off by
 * default. Compile time is recorded separately and excluded from the gap. */
static int h3_sol_profile_on(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("H3_SOL_PROFILE");
        cached = (e && e[0] && e[0] != '0' && strcmp(e, "false") != 0 &&
                  strcmp(e, "off") != 0 && strcmp(e, "no") != 0)
                     ? 1
                     : 0;
    }
    return cached;
}

extern "C" int h3_cuda_sol_profile_enabled(void) {
    return h3_sol_profile_on();
}

typedef struct {
    double sdpa_cute, sdpa_cudnn, sdpa_dense;
    double prep, layout, o_cast, h2d, d2h;
    double gemm, adaln, rope, swiglu, gate, unpack;
    double compile, host_py;
    double step_ms;
    unsigned long long n_cute, n_cudnn, n_dense, n_compile;
    unsigned long long n_prep, n_gemm, n_adaln, n_rope;
} h3_sol_prof;

static h3_sol_prof g_sp_step, g_sp_tot;
static int g_sp_banner = 0;
static int g_sp_last_finished = -1;

static void h3_sp_banner(void) {
    if (g_sp_banner || !h3_sol_profile_on()) return;
    g_sp_banner = 1;
    fprintf(stderr,
            "[h3] sol_profile ON (H3_SOL_PROFILE=1) extra EventSynchronize; "
            "cute.compile excluded from speedup gap; "
            "layout=F16→BF16 identity copy into BTHD (B=1 THD≡BTHD)\n");
    fflush(stderr);
}

static void h3_sp_add_n(double *step_f, double *tot_f, unsigned long long *ns,
                        unsigned long long *nt, double ms) {
    if (!h3_sol_profile_on() || !(ms > 0.0)) return;
    h3_sp_banner();
    *step_f += ms;
    *tot_f += ms;
    if (ns) (*ns)++;
    if (nt) (*nt)++;
}

static void h3_sp_add(double *step_f, double *tot_f, double ms) {
    h3_sp_add_n(step_f, tot_f, NULL, NULL, ms);
}

extern "C" void h3_cuda_sol_profile_add_unpack(double ms) {
    h3_sp_add(&g_sp_step.unpack, &g_sp_tot.unpack, ms);
}

struct h3_cuda {
    int device_index;
    h3_cuda_device_info info;
    cublasHandle_t cublas;
    cublasLtHandle_t cublasLt;
    float *d_hadamard; /* cached ConvRot H[gs*gs], gs==hadamard_gs */
    int hadamard_gs;
    float *d_hadamard_fwht; /* Sylvester/FWHT H for Comfy VAE */
    int hadamard_fwht_gs;
    /* Reused SDPA scratch (bytes) to avoid per-call cudaMalloc. */
    void *d_sdpa_scratch;
    size_t sdpa_scratch_bytes;
    /* Temp for packing QKV from raw seq into cuDNN-aligned padded layout. */
    void *d_sdpa_expand;
    size_t sdpa_expand_bytes;
    /* Grow-only TE GQA staging (KV-expand / host QKVO). Avoid per-layer
     * cudaMalloc+Free which serialized 50L encodes. */
    void *d_gqa_stage;
    size_t gqa_stage_bytes;
    /* Reused act ConvRot/quant scratch (separate from SDPA). */
    void *d_act_scratch;
    size_t act_scratch_bytes;
    /* Persistent INT8 GEMM acc + cuBLASLt workspace (CUDA Graph friendly). */
    void *d_gemm_acc;
    size_t gemm_acc_bytes;
    void *d_lt_workspace;
    size_t lt_workspace_bytes;
    int device_bound;
    /* Reuse cuBLASLt INT8 descriptors + heuristic across 50L×4 GEMMs. */
    struct {
        size_t m, n, k;
        int state; /* 0 empty, 1 ready, -1 no-algo */
        cublasLtMatmulDesc_t desc;
        cublasLtMatrixLayout_t a, b, c;
        cublasLtMatmulPreference_t pref;
        cublasLtMatmulHeuristicResult_t heur;
    } lt_i8[8];
    int lt_i8_n;
    int lt_i8_logged;
    /* When set: skip cudaEventSynchronize / avoid fresh cudaMalloc.
     * quiet: skip event timing syncs but still allow buffer growth.
     * graph_capture: also forbid realloc (CUDA Graph capture safety). */
    int quiet;
    int graph_capture;
    /* Borrowed. NULL = legacy default stream (U=1). Ulysses sets a
     * per-rank non-blocking stream so GEMM/SDPA can overlap across GPUs. */
    cudaStream_t op_stream;
    cudaStream_t streams[H3_CUDA_N_STREAMS];
    int streams_ready;
    /* Reused W4A8 unpack H2D scratch (avoids per-weight temp VRAM spike). */
    void *d_unpack_scratch;
    size_t unpack_scratch_bytes;
    /* Grow-only VAE-encode Conv3d scratch (NHWC in / PyTorch W / out / bias). */
    void *d_conv3d_in;
    void *d_conv3d_w;
    void *d_conv3d_y;
    void *d_conv3d_bias;
    size_t conv3d_in_bytes;
    size_t conv3d_w_bytes;
    size_t conv3d_y_bytes;
    size_t conv3d_bias_bytes;
    /* Set by h3_cuda_pregrow_dit_scratch. Refit must not subtract another
     * 8GiB from remaining free (cafe_p2 window=19). */
    int dit_scratch_pregrown;
#if defined(H3_HAVE_CUDNN)
    cudnnHandle_t cudnn;
    int cudnn_ok;
#endif
};

static int h3_cuda_want_events(const h3_cuda *cuda) {
    if (!cuda || cuda->graph_capture) return 0;
    if (h3_sol_profile_on()) return 1;
    return !cuda->quiet;
}

static cudaStream_t h3_cuda_op_st(const h3_cuda *cuda) {
    return (cuda && cuda->op_stream) ? cuda->op_stream : (cudaStream_t)0;
}

#if defined(H3_HAVE_CUDNN_FRONTEND)
extern "C" int h3_cuda_sdpa_cudnn_f16_run(
    void *cudnn_handle, const void *d_q_half, const void *d_k_half,
    const void *d_v_half, void *d_o_half_out, float *d_o_f32, size_t seq,
    size_t heads, size_t head_dim, float scale, double *sdpa_ms, char *error,
    size_t error_size);
extern "C" int h3_cuda_sdpa_cudnn_f16_run_ex(
    void *cudnn_handle, const void *d_q_half, const void *d_k_half,
    const void *d_v_half, void *d_o_half_out, float *d_o_f32, size_t seq,
    size_t heads, size_t head_dim, float scale, int causal, size_t logical_seq,
    int use_mask, double *sdpa_ms, char *error, size_t error_size);
extern "C" size_t h3_cuda_sdpa_cudnn_align_seq(size_t seq);
extern "C" int h3_cuda_sdpa_cudnn_shape_blocked(size_t seq, size_t heads,
                                                size_t head_dim, int causal);
extern "C" int h3_cuda_sdpa_cudnn_want_padded(size_t seq, size_t heads,
                                              size_t head_dim, int causal);
extern "C" void h3_cuda_sdpa_cudnn_note_variant(size_t seq, size_t heads,
                                                size_t head_dim, int causal,
                                                int variant);
extern "C" int h3_cuda_sdpa_cudnn_variant_failed(size_t seq, size_t heads,
                                                 size_t head_dim, int causal,
                                                 int variant);
extern "C" void h3_cuda_sdpa_cudnn_note_variant_fail(size_t seq, size_t heads,
                                                     size_t head_dim, int causal,
                                                     int variant);
extern "C" int h3_cuda_sdpa_cudnn_prebuild(
    void *cudnn_handle, size_t seq, size_t heads, size_t head_dim, float scale,
    int causal, int use_mask, char *error, size_t error_size);
extern "C" void h3_cuda_sdpa_cudnn_reset(void);
extern "C" void h3_cuda_sdpa_cudnn_bind_bias(const void *d_bias_half,
                                             size_t seq);
extern "C" void h3_cuda_sdpa_cudnn_log_fallback(size_t seq, size_t padded,
                                                size_t head_dim, int causal,
                                                const char *msg);
enum {
    H3_CUDNN_VAR_RAW = 1,
    H3_CUDNN_VAR_PAD_MASK = 2,
    H3_CUDNN_VAR_PAD_CAUSAL = 3,
    H3_CUDNN_VAR_FAIL = 4
};
#endif

struct h3_cuda_w4a8_weight {
    h3_cuda *cuda;
    size_t n;
    size_t k;
    int8_t *d_w_i8;   /* [N, K] ConvRot-rotated INT8 */
    float *d_s_channel; /* [N] */
    cudaEvent_t ready_event;
    int ready_event_valid;
    /* INT8 create_from_i8: host buffers kept until side-stream H2D completes
     * so upload can overlap default-stream compute (no default-stream sync). */
    int8_t *h_w_owned;
    float *h_s_owned;
};

/* INT8×INT8 → INT32 accumulate (row-major Y[m,n] = X[m,k] @ W[n,k]^T). */
__global__ void h3_cuda_gemm_i8_kernel(const int8_t *x, const int8_t *w,
                                       int32_t *y, size_t m, size_t n,
                                       size_t k) {
    size_t row = (size_t)blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;
    const int8_t *xi = x + row * k;
    const int8_t *wj = w + col * k;
    int32_t acc = 0;
    for (size_t t = 0; t < k; t++) {
        acc += (int32_t)xi[t] * (int32_t)wj[t];
    }
    y[row * n + col] = acc;
}

/* y_f32 = act_scale[row] * s_channel[col] * (float)acc
 * 2D grid: (ceil(n/256), m) — avoids 64-bit div/mod on every element. */
__global__ void h3_cuda_dequant_i32_kernel(const int32_t *acc,
                                           const float *act_scales,
                                           int per_tensor, const float *s_channel,
                                           float *y, size_t m, size_t n) {
    size_t col = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t row = (size_t)blockIdx.y;
    if (row >= m || col >= n) return;
    size_t idx = row * n + col;
    float a_scale = per_tensor ? act_scales[0] : act_scales[row];
    y[idx] = a_scale * s_channel[col] * (float)acc[idx];
}

/* IEEE float8_e4m3fn → float32 (matches host h3_f8e4m3fn_to_f32). */
__device__ float h3_cuda_f8e4m3fn_to_f32(uint8_t bits) {
    const int sign = (bits >> 7) & 1;
    const int exp = (bits >> 3) & 0xF;
    const int mant = bits & 0x7;
    if (exp == 0xF && mant == 0x7) {
        return sign ? -nanf("") : nanf("");
    }
    float value;
    if (exp == 0) {
        if (mant == 0) {
            value = 0.0f;
        } else {
            value = ldexpf((float)mant, 1 - 7 - 3);
        }
    } else {
        value = ldexpf(1.0f + (float)mant / 8.0f, exp - 7);
    }
    return sign ? -value : value;
}

/* One thread per packed byte → two INT8 outputs (low/high nibble).
 * Layout: low nibble → even K, high → odd K (comfy-kitchen). */
__global__ void h3_cuda_w4a8_unpack_kernel(const int8_t *packed,
                                           const float *codebook16,
                                           const uint8_t *s_rel_f8,
                                           int8_t *int8_out, size_t n,
                                           size_t k_half, int group_size) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = n * k_half;
    if (idx >= total) return;

    const size_t row = idx / k_half;
    const size_t col_half = idx % k_half;
    const size_t k = k_half * 2;
    const size_t groups = k / (size_t)group_size;

    const unsigned byte = (unsigned)(uint8_t)packed[idx];
    const int idx0 = (int)(byte & 0xFu);
    const int idx1 = (int)((byte >> 4) & 0xFu);
    const size_t col0 = col_half * 2;
    const size_t col1 = col0 + 1;
    const size_t g0 = col0 / (size_t)group_size;
    const size_t g1 = col1 / (size_t)group_size;

    float s0 = h3_cuda_f8e4m3fn_to_f32(s_rel_f8[row * groups + g0]);
    float s1 = h3_cuda_f8e4m3fn_to_f32(s_rel_f8[row * groups + g1]);

    float v0 = roundf(codebook16[idx0] * s0);
    float v1 = roundf(codebook16[idx1] * s1);
    if (v0 > 127.0f) v0 = 127.0f;
    if (v0 < -127.0f) v0 = -127.0f;
    if (v1 > 127.0f) v1 = 127.0f;
    if (v1 < -127.0f) v1 = -127.0f;

    int8_out[row * k + col0] = (int8_t)v0;
    int8_out[row * k + col1] = (int8_t)v1;
}

/* Definitions match the extern "C" declarations in h3_gpu_cuda.h. */

static void h3_cuda_set_error(char *error, size_t error_size, const char *message) {
    if (!error || !error_size) return;
    snprintf(error, error_size, "%s", message ? message : "CUDA error");
}

static void h3_cuda_set_error_cuda(char *error, size_t error_size,
                                   const char *prefix, cudaError_t status) {
    if (!error || !error_size) return;
    snprintf(error, error_size, "%s: %s",
             prefix ? prefix : "CUDA", cudaGetErrorString(status));
}

static int h3_cuda_fill_info(int device_index, h3_cuda_device_info *info,
                             char *error, size_t error_size) {
    cudaDeviceProp prop;
    cudaError_t status;
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    int driver_version = 0;
    int runtime_version = 0;

    if (!info) {
        h3_cuda_set_error(error, error_size, "info is NULL");
        return 0;
    }
    memset(info, 0, sizeof(*info));

    status = cudaGetDeviceProperties(&prop, device_index);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaGetDeviceProperties",
                               status);
        return 0;
    }

    status = cudaSetDevice(device_index);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaSetDevice", status);
        return 0;
    }

    status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaMemGetInfo", status);
        return 0;
    }

    (void)cudaDriverGetVersion(&driver_version);
    (void)cudaRuntimeGetVersion(&runtime_version);

    info->device_index = device_index;
    snprintf(info->name, sizeof(info->name), "%s", prop.name);
    info->compute_major = prop.major;
    info->compute_minor = prop.minor;
    info->total_memory_bytes = total_bytes ? total_bytes : prop.totalGlobalMem;
    info->free_memory_bytes = free_bytes;
    info->multiprocessor_count = prop.multiProcessorCount;
    info->driver_version = driver_version;
    info->runtime_version = runtime_version;
    return 1;
}

int h3_cuda_device_count(char *error, size_t error_size) {
    int count = 0;
    cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaGetDeviceCount", status);
        return -1;
    }
    return count;
}

int h3_cuda_probe(int device_index, h3_cuda_device_info *info,
                  char *error, size_t error_size) {
    int count = h3_cuda_device_count(error, error_size);
    if (count < 0) return 0;
    if (count == 0) {
        h3_cuda_set_error(error, error_size, "no CUDA devices");
        return 0;
    }
    if (device_index < 0) device_index = 0;
    if (device_index >= count) {
        h3_cuda_set_error(error, error_size, "CUDA device index out of range");
        return 0;
    }
    return h3_cuda_fill_info(device_index, info, error, error_size);
}

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

static char g_nvrtc_path[PATH_MAX];
static int g_nvrtc_ok;
static int g_nvrtc_checked;

static int h3_cuda_dlopen_ok(const char *path) {
    return path && path[0] && dlopen(path, RTLD_NOW | RTLD_GLOBAL) != NULL;
}

static int h3_cuda_load_nvrtc_dir(const char *dir) {
    DIR *d;
    struct dirent *ent;
    char builtins[PATH_MAX];
    char nvrtc[PATH_MAX];
    if (!dir || !dir[0]) return 0;
    d = opendir(dir);
    if (!d) return 0;
    builtins[0] = 0;
    nvrtc[0] = 0;
    while ((ent = readdir(d)) != NULL) {
        if (!strncmp(ent->d_name, "libnvrtc-builtins.so", 20)) {
            snprintf(builtins, sizeof(builtins), "%s/%s", dir, ent->d_name);
        } else if (!strcmp(ent->d_name, "libnvrtc.so.13") ||
                   !strcmp(ent->d_name, "libnvrtc.so.12") ||
                   !strcmp(ent->d_name, "libnvrtc.so")) {
            snprintf(nvrtc, sizeof(nvrtc), "%s/%s", dir, ent->d_name);
        }
    }
    closedir(d);
    if (builtins[0]) (void)h3_cuda_dlopen_ok(builtins);
    if (nvrtc[0] && h3_cuda_dlopen_ok(nvrtc)) {
        snprintf(g_nvrtc_path, sizeof(g_nvrtc_path), "%s", nvrtc);
        g_nvrtc_ok = 1;
        return 1;
    }
    return 0;
}

static void h3_cuda_preload_nvrtc(void) {
    const char *conda;
    const char *cuda_home;
    char dir[PATH_MAX];
    if (g_nvrtc_checked) return;
    g_nvrtc_checked = 1;
    g_nvrtc_ok = 0;
    g_nvrtc_path[0] = 0;

    /* cuDNN 9.10.x engines dlopen so.12 for sm_120 JIT. so.13 on ldconfig
     * must not skip this — gold cafe had cuda_nvrtc on DT_RPATH. */
    {
        const char *nv12[] = {
            "/home/user/opt/py310/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib",
            NULL
        };
        const char *conda12 = getenv("CONDA_PREFIX");
        char d12[PATH_MAX];
        (void)h3_cuda_load_nvrtc_dir(nv12[0]);
        if (conda12 && conda12[0]) {
            snprintf(d12, sizeof(d12),
                     "%s/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib",
                     conda12);
            (void)h3_cuda_load_nvrtc_dir(d12);
            snprintf(d12, sizeof(d12),
                     "%s/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib",
                     conda12);
            (void)h3_cuda_load_nvrtc_dir(d12);
        }
        if (h3_cuda_dlopen_ok("libnvrtc.so.12"))
            fprintf(stderr, "[h3] nvrtc: preloaded libnvrtc.so.12 for cuDNN JIT\n");
    }

    if (h3_cuda_dlopen_ok("libnvrtc.so.13")) {
        snprintf(g_nvrtc_path, sizeof(g_nvrtc_path), "%s", "libnvrtc.so.13");
        g_nvrtc_ok = 1;
        fprintf(stderr, "[h3] nvrtc: loaded libnvrtc.so.13 via default loader\n");
        fflush(stderr);
        return;
    }
    if (h3_cuda_dlopen_ok("libnvrtc.so.12")) {
        snprintf(g_nvrtc_path, sizeof(g_nvrtc_path), "%s", "libnvrtc.so.12");
        g_nvrtc_ok = 1;
        fprintf(stderr, "[h3] nvrtc: loaded libnvrtc.so.12 via default loader\n");
        fflush(stderr);
        return;
    }
    if (h3_cuda_dlopen_ok("libnvrtc.so")) {
        snprintf(g_nvrtc_path, sizeof(g_nvrtc_path), "%s", "libnvrtc.so");
        g_nvrtc_ok = 1;
        fprintf(stderr, "[h3] nvrtc: loaded libnvrtc.so via default loader\n");
        fflush(stderr);
        return;
    }

    conda = getenv("CONDA_PREFIX");
    cuda_home = getenv("CUDA_HOME");
    if (!cuda_home || !cuda_home[0]) cuda_home = getenv("CUDA_PATH");

    if (conda && conda[0]) {
        snprintf(dir, sizeof(dir),
                 "%s/lib/python3.12/site-packages/nvidia/cu13/lib", conda);
        if (h3_cuda_load_nvrtc_dir(dir)) goto found;
        snprintf(dir, sizeof(dir),
                 "%s/lib/python3.11/site-packages/nvidia/cu13/lib", conda);
        if (h3_cuda_load_nvrtc_dir(dir)) goto found;
        snprintf(dir, sizeof(dir),
                 "%s/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib", conda);
        if (h3_cuda_load_nvrtc_dir(dir)) goto found;
        snprintf(dir, sizeof(dir),
                 "%s/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib", conda);
        if (h3_cuda_load_nvrtc_dir(dir)) goto found;
        snprintf(dir, sizeof(dir),
                 "%s/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib", conda);
        if (h3_cuda_load_nvrtc_dir(dir)) goto found;
        snprintf(dir, sizeof(dir), "%s/lib", conda);
        if (h3_cuda_load_nvrtc_dir(dir)) goto found;
        snprintf(dir, sizeof(dir), "%s/targets/x86_64-linux/lib", conda);
        if (h3_cuda_load_nvrtc_dir(dir)) goto found;
    }
    if (cuda_home && cuda_home[0]) {
        snprintf(dir, sizeof(dir), "%s/lib64", cuda_home);
        if (h3_cuda_load_nvrtc_dir(dir)) goto found;
        snprintf(dir, sizeof(dir), "%s/lib", cuda_home);
        if (h3_cuda_load_nvrtc_dir(dir)) goto found;
    }
    if (h3_cuda_load_nvrtc_dir("/usr/local/cuda/lib64")) goto found;

    fprintf(stderr,
            "[h3] nvrtc: MISSING libnvrtc.so.13/so.12 (cuDNN JIT will fail). "
            "Install CUDA 13 NVRTC (nvidia-cu13 / toolkit lib) or CUDA 12 "
            "nvidia-cuda-nvrtc and put it on LD_LIBRARY_PATH / rpath.\n");
    fflush(stderr);
    return;

found:
    fprintf(stderr, "[h3] nvrtc: %s\n", g_nvrtc_path);
    fflush(stderr);
}

const char *h3_cuda_nvrtc_path(void) {
    h3_cuda_preload_nvrtc();
    return g_nvrtc_ok ? g_nvrtc_path : NULL;
}

int h3_cuda_nvrtc_ok(void) {
    h3_cuda_preload_nvrtc();
    return g_nvrtc_ok;
}

static void h3_cuda_sol_cute_maybe_warmup(void);

h3_cuda *h3_cuda_create(int device_index, char *error, size_t error_size) {
    h3_cuda *cuda;
    h3_cuda_device_info info;
    cublasStatus_t cublas_status;

    h3_cuda_preload_nvrtc();
    if (!h3_cuda_probe(device_index, &info, error, error_size)) return NULL;
    h3_cuda_sdpa_log_selection(device_index);
    cuda = (h3_cuda *)calloc(1, sizeof(*cuda));
    if (!cuda) {
        h3_cuda_set_error(error, error_size, "out of memory");
        return NULL;
    }
    cuda->device_index = info.device_index;
    cuda->info = info;
    if (cudaSetDevice(cuda->device_index) == cudaSuccess)
        cuda->device_bound = 1;
    cublas_status = cublasCreate(&cuda->cublas);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        free(cuda);
        h3_cuda_set_error(error, error_size, "cublasCreate failed");
        return NULL;
    }
    cublas_status = cublasLtCreate(&cuda->cublasLt);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        cublasDestroy(cuda->cublas);
        free(cuda);
        h3_cuda_set_error(error, error_size, "cublasLtCreate failed");
        return NULL;
    }
    {
        int s;
        cuda->streams_ready = 1;
        for (s = 0; s < H3_CUDA_N_STREAMS; s++) {
            if (cudaStreamCreateWithFlags(&cuda->streams[s],
                                          cudaStreamNonBlocking) !=
                cudaSuccess) {
                cuda->streams_ready = 0;
                break;
            }
        }
        if (!cuda->streams_ready) {
            for (s = 0; s < H3_CUDA_N_STREAMS; s++) {
                if (cuda->streams[s]) cudaStreamDestroy(cuda->streams[s]);
                cuda->streams[s] = NULL;
            }
        }
    }
#if defined(H3_HAVE_CUDNN)
    cuda->cudnn_ok = (cudnnCreate(&cuda->cudnn) == CUDNN_STATUS_SUCCESS);
#endif
    h3_cuda_sol_cute_maybe_warmup();
    return cuda;
}

static void h3_cuda_lt_i8_reset(h3_cuda *cuda) {
    int i;
    if (!cuda) return;
    for (i = 0; i < cuda->lt_i8_n && i < 8; i++) {
        if (cuda->lt_i8[i].pref)
            cublasLtMatmulPreferenceDestroy(cuda->lt_i8[i].pref);
        if (cuda->lt_i8[i].c) cublasLtMatrixLayoutDestroy(cuda->lt_i8[i].c);
        if (cuda->lt_i8[i].b) cublasLtMatrixLayoutDestroy(cuda->lt_i8[i].b);
        if (cuda->lt_i8[i].a) cublasLtMatrixLayoutDestroy(cuda->lt_i8[i].a);
        if (cuda->lt_i8[i].desc) cublasLtMatmulDescDestroy(cuda->lt_i8[i].desc);
        memset(&cuda->lt_i8[i], 0, sizeof(cuda->lt_i8[i]));
    }
    cuda->lt_i8_n = 0;
}

static int h3_cuda_bind(h3_cuda *cuda, char *error, size_t error_size) {
    cudaError_t status;
    int cur = -1;
    if (!cuda) {
        h3_cuda_set_error(error, error_size, "cuda handle is NULL");
        return 0;
    }
    /* cudaSetDevice is illegal during stream capture. */
    if (cuda->graph_capture) return 1;
    if (cudaGetDevice(&cur) == cudaSuccess && cur == cuda->device_index) {
        cuda->device_bound = 1;
        return 1;
    }
    status = cudaSetDevice(cuda->device_index);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaSetDevice", status);
        return 0;
    }
    cuda->device_bound = 1;
    return 1;
}

void h3_cuda_free(h3_cuda *cuda) {
    int s;
    if (!cuda) return;
    h3_cuda_sol_cute_dump_stats();
    h3_cuda_dist_attn_clear();
    h3_cuda_lt_i8_reset(cuda);
#if defined(H3_HAVE_CUDNN_FRONTEND)
    h3_cuda_sdpa_cudnn_reset();
#endif
    if (cuda->d_hadamard) cudaFree(cuda->d_hadamard);
    if (cuda->d_hadamard_fwht) cudaFree(cuda->d_hadamard_fwht);
    if (cuda->d_sdpa_scratch) cudaFree(cuda->d_sdpa_scratch);
    if (cuda->d_sdpa_expand) cudaFree(cuda->d_sdpa_expand);
    if (cuda->d_gqa_stage) cudaFree(cuda->d_gqa_stage);
    if (cuda->d_act_scratch) cudaFree(cuda->d_act_scratch);
    if (cuda->d_gemm_acc) cudaFree(cuda->d_gemm_acc);
    if (cuda->d_lt_workspace) cudaFree(cuda->d_lt_workspace);
    if (cuda->d_unpack_scratch) cudaFree(cuda->d_unpack_scratch);
    if (cuda->d_conv3d_in) cudaFree(cuda->d_conv3d_in);
    if (cuda->d_conv3d_w) cudaFree(cuda->d_conv3d_w);
    if (cuda->d_conv3d_y) cudaFree(cuda->d_conv3d_y);
    if (cuda->d_conv3d_bias) cudaFree(cuda->d_conv3d_bias);
    for (s = 0; s < H3_CUDA_N_STREAMS; s++) {
        if (cuda->streams[s]) cudaStreamDestroy(cuda->streams[s]);
    }
#if defined(H3_HAVE_CUDNN)
    if (cuda->cudnn_ok) cudnnDestroy(cuda->cudnn);
#endif
    if (cuda->cublasLt) cublasLtDestroy(cuda->cublasLt);
    if (cuda->cublas) cublasDestroy(cuda->cublas);
    free(cuda);
}

void h3_cuda_release_heavy_workspace(h3_cuda *cuda) {
    if (!cuda) return;
    /* Dist-attn bias is DiT-sized; drop it before VAE so VAE SDPA is not
     * scored with video grid coords. */
    h3_cuda_dist_attn_clear();
    /* Keep cuDNN SDPA plans (TE/DiT/VAE slots). Workspace is typically 0;
     * resetting here forced a full DiT rebuild after TE. */
    if (cuda->d_sdpa_scratch) {
        cudaFree(cuda->d_sdpa_scratch);
        cuda->d_sdpa_scratch = NULL;
        cuda->sdpa_scratch_bytes = 0;
    }
    if (cuda->d_sdpa_expand) {
        cudaFree(cuda->d_sdpa_expand);
        cuda->d_sdpa_expand = NULL;
        cuda->sdpa_expand_bytes = 0;
    }
    if (cuda->d_gqa_stage) {
        cudaFree(cuda->d_gqa_stage);
        cuda->d_gqa_stage = NULL;
        cuda->gqa_stage_bytes = 0;
    }
    if (cuda->d_act_scratch) {
        cudaFree(cuda->d_act_scratch);
        cuda->d_act_scratch = NULL;
        cuda->act_scratch_bytes = 0;
    }
    if (cuda->d_gemm_acc) {
        cudaFree(cuda->d_gemm_acc);
        cuda->d_gemm_acc = NULL;
        cuda->gemm_acc_bytes = 0;
    }
    if (cuda->d_lt_workspace) {
        cudaFree(cuda->d_lt_workspace);
        cuda->d_lt_workspace = NULL;
        cuda->lt_workspace_bytes = 0;
    }
    if (cuda->d_unpack_scratch) {
        cudaFree(cuda->d_unpack_scratch);
        cuda->d_unpack_scratch = NULL;
        cuda->unpack_scratch_bytes = 0;
    }
    if (cuda->d_conv3d_in) {
        cudaFree(cuda->d_conv3d_in);
        cuda->d_conv3d_in = NULL;
        cuda->conv3d_in_bytes = 0;
    }
    if (cuda->d_conv3d_w) {
        cudaFree(cuda->d_conv3d_w);
        cuda->d_conv3d_w = NULL;
        cuda->conv3d_w_bytes = 0;
    }
    if (cuda->d_conv3d_y) {
        cudaFree(cuda->d_conv3d_y);
        cuda->d_conv3d_y = NULL;
        cuda->conv3d_y_bytes = 0;
    }
    if (cuda->d_conv3d_bias) {
        cudaFree(cuda->d_conv3d_bias);
        cuda->d_conv3d_bias = NULL;
        cuda->conv3d_bias_bytes = 0;
    }
    cuda->dit_scratch_pregrown = 0;
}

static int h3_cuda_ensure_gemm_acc(h3_cuda *cuda, size_t bytes, char *error,
                                   size_t error_size) {
    if (!cuda) {
        h3_cuda_set_error(error, error_size, "cuda handle is NULL");
        return 0;
    }
    if (bytes == 0) return 1;
    if (cuda->d_gemm_acc && cuda->gemm_acc_bytes >= bytes) return 1;
    if (cuda->graph_capture) {
        h3_cuda_set_error(error, error_size,
                          "gemm acc realloc during graph capture");
        return 0;
    }
    if (!h3_cuda_bind(cuda, error, error_size)) return 0;
    if (cuda->d_gemm_acc) {
        cudaFree(cuda->d_gemm_acc);
        cuda->d_gemm_acc = NULL;
        cuda->gemm_acc_bytes = 0;
    }
    if (cudaMalloc(&cuda->d_gemm_acc, bytes) != cudaSuccess) {
        h3_cuda_set_error(error, error_size, "cudaMalloc gemm acc failed");
        return 0;
    }
    cuda->gemm_acc_bytes = bytes;
    return 1;
}

static int h3_cuda_ensure_lt_workspace(h3_cuda *cuda, size_t bytes, char *error,
                                       size_t error_size) {
    if (!cuda) {
        h3_cuda_set_error(error, error_size, "cuda handle is NULL");
        return 0;
    }
    if (bytes == 0) return 1;
    if (cuda->d_lt_workspace && cuda->lt_workspace_bytes >= bytes) return 1;
    if (cuda->graph_capture) {
        h3_cuda_set_error(error, error_size,
                          "lt workspace realloc during graph capture");
        return 0;
    }
    if (!h3_cuda_bind(cuda, error, error_size)) return 0;
    if (cuda->d_lt_workspace) {
        cudaFree(cuda->d_lt_workspace);
        cuda->d_lt_workspace = NULL;
        cuda->lt_workspace_bytes = 0;
    }
    if (cudaMalloc(&cuda->d_lt_workspace, bytes) != cudaSuccess) {
        h3_cuda_set_error(error, error_size, "cudaMalloc lt workspace failed");
        return 0;
    }
    cuda->lt_workspace_bytes = bytes;
    return 1;
}

static int h3_cuda_ensure_scratch(h3_cuda *cuda, size_t bytes, char *error,
                                  size_t error_size) {
    cudaError_t status;
    if (!cuda) {
        h3_cuda_set_error(error, error_size, "cuda handle is NULL");
        return 0;
    }
    if (bytes == 0) return 1;
    if (cuda->d_sdpa_scratch && cuda->sdpa_scratch_bytes >= bytes) return 1;
    if (!h3_cuda_bind(cuda, error, error_size)) return 0;
    if (cuda->d_sdpa_scratch) {
        cudaFree(cuda->d_sdpa_scratch);
        cuda->d_sdpa_scratch = NULL;
        cuda->sdpa_scratch_bytes = 0;
    }
    status = cudaMalloc(&cuda->d_sdpa_scratch, bytes);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaMalloc sdpa scratch",
                               status);
        return 0;
    }
    cuda->sdpa_scratch_bytes = bytes;
    return 1;
}

static int h3_cuda_ensure_expand(h3_cuda *cuda, size_t bytes, char *error,
                                 size_t error_size) {
    cudaError_t status;
    if (!cuda) {
        h3_cuda_set_error(error, error_size, "cuda handle is NULL");
        return 0;
    }
    if (bytes == 0) return 1;
    if (cuda->d_sdpa_expand && cuda->sdpa_expand_bytes >= bytes) return 1;
    if (cuda->graph_capture) {
        h3_cuda_set_error(error, error_size,
                          "sdpa expand realloc during graph capture");
        return 0;
    }
    if (cuda->d_sdpa_expand) {
        cudaFree(cuda->d_sdpa_expand);
        cuda->d_sdpa_expand = NULL;
        cuda->sdpa_expand_bytes = 0;
    }
    status = cudaMalloc(&cuda->d_sdpa_expand, bytes);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaMalloc sdpa expand",
                               status);
        return 0;
    }
    cuda->sdpa_expand_bytes = bytes;
    return 1;
}

static int h3_cuda_ensure_gqa_stage(h3_cuda *cuda, size_t bytes, char *error,
                                    size_t error_size) {
    cudaError_t status;
    if (!cuda) {
        h3_cuda_set_error(error, error_size, "cuda handle is NULL");
        return 0;
    }
    if (bytes == 0) return 1;
    if (cuda->d_gqa_stage && cuda->gqa_stage_bytes >= bytes) return 1;
    if (cuda->graph_capture) {
        h3_cuda_set_error(error, error_size,
                          "gqa stage realloc during graph capture");
        return 0;
    }
    if (cuda->d_gqa_stage) {
        cudaFree(cuda->d_gqa_stage);
        cuda->d_gqa_stage = NULL;
        cuda->gqa_stage_bytes = 0;
    }
    status = cudaMalloc(&cuda->d_gqa_stage, bytes);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaMalloc gqa stage",
                               status);
        return 0;
    }
    cuda->gqa_stage_bytes = bytes;
    return 1;
}

__global__ void h3_cuda_zero_pad_tails_kernel(__half *q, __half *k, __half *v,
                                              size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        q[i] = __float2half(0.0f);
        k[i] = __float2half(0.0f);
        v[i] = __float2half(0.0f);
    }
}

static void h3_cuda_zero_pad_tails(const h3_cuda *cuda, __half *q, __half *k,
                                   __half *v, size_t n) {
    if (n == 0) return;
    unsigned thr = 256;
    unsigned blk = (unsigned)((n + (size_t)thr - 1) / (size_t)thr);
    h3_cuda_zero_pad_tails_kernel<<<blk, thr, 0, h3_cuda_op_st(cuda)>>>(q, k, v,
                                                                        n);
}

/* Repack Q/K/V in scratch from raw [seq] to padded [align] layout.
 * Scratch must already hold 4 * padded * heads * d halves. */
/* Packed [Q|K|V] @ seq → padded [Q|K|V] @ padded. One kernel; tmp holds the
 * packed source so in-place dest (P>S) cannot clobber unread K/V. */
__global__ void h3_cuda_sdpa_expand_qkv_kernel(const __half *__restrict__ src,
                                               __half *__restrict__ dst,
                                               size_t S, size_t P) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t n = 3u * P;
    size_t t, o;
    if (idx >= n) return;
    t = idx / P;
    o = idx - t * P;
    dst[idx] = (o < S) ? src[t * S + o] : __float2half(0.0f);
}

__global__ void h3_cuda_sdpa_pack_pad_kernel(const __half *__restrict__ src,
                                            __half *__restrict__ dst, size_t S,
                                            size_t P) {
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= P) return;
    dst[i] = (i < S) ? src[i] : __float2half(0.0f);
}

static int h3_cuda_sdpa_expand_qkv_in_scratch(h3_cuda *cuda, __half *base,
                                              size_t seq, size_t padded,
                                              size_t heads, size_t head_dim,
                                              char *error, size_t error_size) {
    size_t S = seq * heads * head_dim;
    size_t P = padded * heads * head_dim;
    __half *tmp;
    const size_t packed_bytes = 3u * S * sizeof(__half);
    const size_t kMaxSnap = 192u * 1024u * 1024u;
    unsigned nblk;
    if (padded <= seq) return 1;
    if (packed_bytes <= kMaxSnap) {
        if (!h3_cuda_ensure_expand(cuda, packed_bytes, error, error_size))
            return 0;
        tmp = (__half *)cuda->d_sdpa_expand;
        if (cudaMemcpy(tmp, base, packed_bytes, cudaMemcpyDeviceToDevice) !=
            cudaSuccess) {
            h3_cuda_set_error(error, error_size, "sdpa expand snapshot QKV");
            return 0;
        }
        nblk = (unsigned)((3u * P + 255u) / 256u);
        if (nblk == 0) nblk = 1;
        h3_cuda_sdpa_expand_qkv_kernel<<<nblk, 256>>>(tmp, base, S, P);
        if (cudaGetLastError() != cudaSuccess) {
            h3_cuda_set_error(error, error_size, "sdpa expand kernel");
            return 0;
        }
        return 1;
    }
    /* Long-seq: one S-row tmp, pack V then K, zero Q tail. */
    if (!h3_cuda_ensure_expand(cuda, S * sizeof(__half), error, error_size))
        return 0;
    tmp = (__half *)cuda->d_sdpa_expand;
    nblk = (unsigned)((P + 255u) / 256u);
    if (nblk == 0) nblk = 1;
    if (cudaMemcpy(tmp, base + 2u * S, S * sizeof(__half),
                   cudaMemcpyDeviceToDevice) != cudaSuccess) {
        h3_cuda_set_error(error, error_size, "sdpa expand copy V");
        return 0;
    }
    h3_cuda_sdpa_pack_pad_kernel<<<nblk, 256>>>(tmp, base + 2u * P, S, P);
    if (cudaMemcpy(tmp, base + S, S * sizeof(__half),
                   cudaMemcpyDeviceToDevice) != cudaSuccess) {
        h3_cuda_set_error(error, error_size, "sdpa expand copy K");
        return 0;
    }
    h3_cuda_sdpa_pack_pad_kernel<<<nblk, 256>>>(tmp, base + P, S, P);
    if (P > S)
        h3_cuda_sdpa_pack_pad_kernel<<<nblk, 256>>>(base, base, S, P);
    if (cudaGetLastError() != cudaSuccess) {
        h3_cuda_set_error(error, error_size, "sdpa expand pack");
        return 0;
    }
    return 1;
}

static int h3_cuda_ensure_act_scratch(h3_cuda *cuda, size_t bytes, char *error,
                                      size_t error_size) {
    cudaError_t status;
    if (!cuda) {
        h3_cuda_set_error(error, error_size, "cuda handle is NULL");
        return 0;
    }
    if (bytes == 0) return 1;
    if (cuda->d_act_scratch && cuda->act_scratch_bytes >= bytes) return 1;
    if (!h3_cuda_bind(cuda, error, error_size)) return 0;
    if (cuda->d_act_scratch) {
        cudaFree(cuda->d_act_scratch);
        cuda->d_act_scratch = NULL;
        cuda->act_scratch_bytes = 0;
    }
    status = cudaMalloc(&cuda->d_act_scratch, bytes);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaMalloc act scratch",
                               status);
        return 0;
    }
    cuda->act_scratch_bytes = bytes;
    return 1;
}

int h3_cuda_pregrow_dit_scratch(h3_cuda *cuda, size_t seq, size_t hidden,
                                size_t heads, size_t head_dim, size_t ffn,
                                char *error, size_t error_size) {
    size_t m_pad, n_fc1, k_max, acc_bytes, act_bytes, sdpa_bytes, padded;
    (void)hidden;
    if (!cuda || seq == 0 || heads == 0 || head_dim == 0 || ffn == 0) {
        h3_cuda_set_error(error, error_size, "invalid pregrow_dit_scratch");
        return 0;
    }
    if (!h3_cuda_bind(cuda, error, error_size)) return 0;
    m_pad = (seq + 15u) & ~(size_t)15u;
    n_fc1 = ffn * 2u; /* SwiGLU fused */
    k_max = ffn;
    acc_bytes = m_pad * n_fc1 * sizeof(int32_t);
    act_bytes = seq * k_max * sizeof(float) + m_pad * k_max + m_pad * sizeof(float);
    padded = (seq + 63u) & ~(size_t)63u;
    sdpa_bytes = padded * heads * head_dim * 4u * sizeof(__half);
    if (!h3_cuda_ensure_gemm_acc(cuda, acc_bytes, error, error_size)) return 0;
    if (!h3_cuda_ensure_act_scratch(cuda, act_bytes, error, error_size))
        return 0;
    if (!h3_cuda_ensure_scratch(cuda, sdpa_bytes, error, error_size)) return 0;
    if (!h3_cuda_ensure_lt_workspace(cuda, (size_t)64u << 20, error,
                                     error_size))
        return 0;
    cuda->dit_scratch_pregrown = 1;
    return 1;
}

int h3_cuda_dit_scratch_pregrown(const h3_cuda *cuda) {
    if (!cuda) return 0;
    if (cuda->dit_scratch_pregrown) return 1;
    return cuda->gemm_acc_bytes > 0 && cuda->sdpa_scratch_bytes > 0 &&
           cuda->act_scratch_bytes > 0;
}

int h3_cuda_bind_device(h3_cuda *cuda, char *error, size_t error_size) {
    return h3_cuda_bind(cuda, error, error_size);
}

static void h3_cuda_drop_if_wrong_dev(void **p, size_t *bytes, int want) {
    struct cudaPointerAttributes a;
    cudaError_t ce;
    if (!p || !*p)
        return;
    memset(&a, 0, sizeof(a));
    ce = cudaPointerGetAttributes(&a, *p);
    if (ce != cudaSuccess) {
        cudaGetLastError();
        return;
    }
    if (a.device == want)
        return;
    (void)cudaSetDevice(a.device);
    (void)cudaFree(*p);
    *p = NULL;
    if (bytes)
        *bytes = 0;
}

int h3_cuda_dit_scratch_rehome(h3_cuda *cuda, char *error, size_t error_size) {
    int want;
    if (!h3_cuda_bind(cuda, error, error_size))
        return 0;
    want = cuda->device_index;
    /* Always drop Hadamard here (U>1 only). Attributes can report the
     * current device for a pointer that was allocated on another GPU. */
    if (cuda->d_hadamard) {
        cudaFree(cuda->d_hadamard);
        cuda->d_hadamard = NULL;
        cuda->hadamard_gs = 0;
    }
    h3_cuda_drop_if_wrong_dev(&cuda->d_act_scratch, &cuda->act_scratch_bytes,
                              want);
    h3_cuda_drop_if_wrong_dev(&cuda->d_gemm_acc, &cuda->gemm_acc_bytes, want);
    h3_cuda_drop_if_wrong_dev(&cuda->d_lt_workspace, &cuda->lt_workspace_bytes,
                              want);
    h3_cuda_drop_if_wrong_dev(&cuda->d_sdpa_scratch, &cuda->sdpa_scratch_bytes,
                              want);
    if (!h3_cuda_bind(cuda, error, error_size))
        return 0;
    return 1;
}

static int h3_cuda_ptr_on_dev(const void *p, int want, const char *name,
                              char *error, size_t error_size) {
    struct cudaPointerAttributes a;
    cudaError_t ce;
    if (!p)
        return 1;
    memset(&a, 0, sizeof(a));
    ce = cudaPointerGetAttributes(&a, p);
    if (ce != cudaSuccess) {
        snprintf(error, error_size, "%s: %s", name, cudaGetErrorString(ce));
        return 0;
    }
    if (a.type != cudaMemoryTypeDevice && a.type != cudaMemoryTypeManaged) {
        snprintf(error, error_size, "%s: type=%d", name, (int)a.type);
        return 0;
    }
    if (a.device != want) {
        snprintf(error, error_size, "%s on dev %d want %d", name, a.device,
                 want);
        return 0;
    }
    return 1;
}

int h3_cuda_dit_scratch_check(h3_cuda *cuda, char *error, size_t error_size) {
    int want;
    if (!cuda) {
        h3_cuda_set_error(error, error_size, "cuda handle is NULL");
        return 0;
    }
    want = cuda->device_index;
    if (!h3_cuda_ptr_on_dev(cuda->d_hadamard, want, "d_hadamard", error,
                            error_size) ||
        !h3_cuda_ptr_on_dev(cuda->d_act_scratch, want, "d_act_scratch", error,
                            error_size) ||
        !h3_cuda_ptr_on_dev(cuda->d_gemm_acc, want, "d_gemm_acc", error,
                            error_size) ||
        !h3_cuda_ptr_on_dev(cuda->d_lt_workspace, want, "d_lt_workspace", error,
                            error_size) ||
        !h3_cuda_ptr_on_dev(cuda->d_sdpa_scratch, want, "d_sdpa_scratch", error,
                            error_size))
        return 0;
    return 1;
}

int h3_cuda_recreate_blas(h3_cuda *cuda, char *error, size_t error_size) {
    cublasStatus_t st;
    if (!cuda) {
        h3_cuda_set_error(error, error_size, "cuda handle is NULL");
        return 0;
    }
    if (!h3_cuda_bind(cuda, error, error_size))
        return 0;
    h3_cuda_lt_i8_reset(cuda);
    if (cuda->cublasLt) {
        cublasLtDestroy(cuda->cublasLt);
        cuda->cublasLt = NULL;
    }
    if (cuda->cublas) {
        cublasDestroy(cuda->cublas);
        cuda->cublas = NULL;
    }
    st = cublasCreate(&cuda->cublas);
    if (st != CUBLAS_STATUS_SUCCESS) {
        h3_cuda_set_error(error, error_size, "cublasCreate failed");
        return 0;
    }
    st = cublasLtCreate(&cuda->cublasLt);
    if (st != CUBLAS_STATUS_SUCCESS) {
        h3_cuda_set_error(error, error_size, "cublasLtCreate failed");
        return 0;
    }
    if (cuda->op_stream &&
        cublasSetStream(cuda->cublas, cuda->op_stream) !=
            CUBLAS_STATUS_SUCCESS) {
        h3_cuda_set_error(error, error_size, "cublasSetStream failed");
        return 0;
    }
    return 1;
}

int h3_cuda_get_info(const h3_cuda *cuda, h3_cuda_device_info *info) {
    if (!cuda || !info) return 0;
    *info = cuda->info;
    return 1;
}

int h3_cuda_mem_info(h3_cuda *cuda, size_t *free_bytes, size_t *total_bytes,
                     char *error, size_t error_size) {
    cudaError_t status;
    size_t free_b = 0;
    size_t total_b = 0;
    if (!cuda) {
        h3_cuda_set_error(error, error_size, "cuda handle is NULL");
        return 0;
    }
    status = cudaSetDevice(cuda->device_index);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaSetDevice", status);
        return 0;
    }
    status = cudaMemGetInfo(&free_b, &total_b);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaMemGetInfo", status);
        return 0;
    }
    cuda->info.free_memory_bytes = free_b;
    cuda->info.total_memory_bytes = total_b;
    if (free_bytes) *free_bytes = free_b;
    if (total_bytes) *total_bytes = total_b;
    return 1;
}

int h3_cuda_gemm_f32(h3_cuda *cuda, const float *x, const float *weight,
                     size_t m, size_t n, size_t k, float *y, double *gemm_ms,
                     char *error, size_t error_size) {
    float *d_x = NULL;
    float *d_w = NULL;
    float *d_y = NULL;
    cudaError_t status;
    cublasStatus_t cublas_status;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cudaEvent_t ev_start = NULL;
    cudaEvent_t ev_stop = NULL;
    float elapsed_ms = 0.0f;
    int ok = 0;

    if (gemm_ms) *gemm_ms = 0.0;
    if (!cuda || !x || !weight || !y || m == 0 || n == 0 || k == 0) {
        h3_cuda_set_error(error, error_size, "invalid h3_cuda_gemm_f32 args");
        return 0;
    }

    status = cudaSetDevice(cuda->device_index);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaSetDevice", status);
        return 0;
    }

    status = cudaMalloc((void **)&d_x, m * k * sizeof(float));
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaMalloc x", status);
        goto cleanup;
    }
    status = cudaMalloc((void **)&d_w, n * k * sizeof(float));
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaMalloc W", status);
        goto cleanup;
    }
    status = cudaMalloc((void **)&d_y, m * n * sizeof(float));
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaMalloc y", status);
        goto cleanup;
    }

    status = cudaMemcpy(d_x, x, m * k * sizeof(float), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaMemcpy x H2D", status);
        goto cleanup;
    }
    status =
        cudaMemcpy(d_w, weight, n * k * sizeof(float), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaMemcpy W H2D", status);
        goto cleanup;
    }

    status = cudaEventCreate(&ev_start);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaEventCreate start",
                               status);
        goto cleanup;
    }
    status = cudaEventCreate(&ev_stop);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaEventCreate stop",
                               status);
        goto cleanup;
    }

    /* Row-major Y = X @ W^T  ↔  column-major:
     *   Y^T = W @ X^T
     * Using row-major buffers as column-major with transposed ops:
     *   cublasSgemm(OP_T, OP_N, N, M, K, W(ld=K), X(ld=K), Y(ld=N)) */
    status = cudaEventRecord(ev_start, 0);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaEventRecord start",
                               status);
        goto cleanup;
    }
    cublas_status = cublasSgemm(
        cuda->cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)n, (int)m, (int)k, &alpha,
        d_w, (int)k, d_x, (int)k, &beta, d_y, (int)n);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        h3_cuda_set_error(error, error_size, "cublasSgemm failed");
        goto cleanup;
    }
    status = cudaEventRecord(ev_stop, 0);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaEventRecord stop",
                               status);
        goto cleanup;
    }
    status = cudaEventSynchronize(ev_stop);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaEventSynchronize",
                               status);
        goto cleanup;
    }
    status = cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaEventElapsedTime",
                               status);
        goto cleanup;
    }
    if (gemm_ms) *gemm_ms = (double)elapsed_ms;

    status = cudaMemcpy(y, d_y, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaMemcpy y D2H", status);
        goto cleanup;
    }
    ok = 1;

cleanup:
    if (ev_start) cudaEventDestroy(ev_start);
    if (ev_stop) cudaEventDestroy(ev_stop);
    if (d_x) cudaFree(d_x);
    if (d_w) cudaFree(d_w);
    if (d_y) cudaFree(d_y);
    return ok;
}

/* One thread per output element. Matches host conv3d_nhwc in
 * h3_vae_encode_f32.c (NHWC in, PyTorch [Cout,Cin,Kd,Kh,Kw] weight). */
__global__ void h3_cuda_conv3d_nhwc_bias_kernel(
    const float *__restrict__ input, const float *__restrict__ weight,
    const float *__restrict__ bias, float *__restrict__ output, uint32_t depth,
    uint32_t height, uint32_t width, uint32_t cin, uint32_t cout, uint32_t od,
    uint32_t oh, uint32_t ow, uint32_t kd, uint32_t kh, uint32_t kw,
    uint32_t stride_t, uint32_t stride_h, uint32_t stride_w) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = (size_t)od * (size_t)oh * (size_t)ow * (size_t)cout;
    uint32_t oc, ox, oy, ot, in_t, in_y, in_x, ic, dt, dy, dx;
    size_t tmp;
    const float *wbase;
    float acc;
    if (idx >= n) return;
    oc = (uint32_t)(idx % (size_t)cout);
    tmp = idx / (size_t)cout;
    ox = (uint32_t)(tmp % (size_t)ow);
    tmp /= (size_t)ow;
    oy = (uint32_t)(tmp % (size_t)oh);
    ot = (uint32_t)(tmp / (size_t)oh);
    in_t = ot * stride_t;
    in_y = oy * stride_h;
    in_x = ox * stride_w;
    acc = bias[oc];
    wbase = weight + (size_t)oc * (size_t)cin * (size_t)kd * (size_t)kh *
                         (size_t)kw;
    for (ic = 0; ic < cin; ic++)
        for (dt = 0; dt < kd; dt++)
            for (dy = 0; dy < kh; dy++)
                for (dx = 0; dx < kw; dx++) {
                    size_t in_i =
                        ((((size_t)(in_t + dt) * height + (in_y + dy)) * width +
                          (in_x + dx)) *
                         cin) +
                        ic;
                    size_t w_i =
                        ((((size_t)ic * kd + dt) * kh + dy) * kw) + dx;
                    acc += input[in_i] * wbase[w_i];
                }
    output[idx] = acc;
}

static int h3_cuda_grow_scratch(void **ptr, size_t *cap, size_t need,
                                char *error, size_t error_size,
                                const char *name) {
    cudaError_t status;
    if (*cap >= need) return 1;
    if (*ptr) {
        cudaFree(*ptr);
        *ptr = NULL;
        *cap = 0;
    }
    status = cudaMalloc(ptr, need);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, name, status);
        return 0;
    }
    *cap = need;
    return 1;
}

int h3_cuda_conv3d_nhwc_bias_f32(
    h3_cuda *cuda, const float *input, uint32_t depth, uint32_t height,
    uint32_t width, const float *weight, const float *bias, uint32_t cin,
    uint32_t cout, uint32_t kd, uint32_t kh, uint32_t kw, uint32_t stride_t,
    uint32_t stride_h, uint32_t stride_w, float *output, char *error,
    size_t error_size) {
    uint32_t od, oh, ow;
    size_t in_n, w_n, out_n;
    int threads = 256;
    int blocks;
    cudaError_t status;

    if (!cuda || !input || !weight || !bias || !output || !depth || !height ||
        !width || !cin || !cout || !kd || !kh || !kw || !stride_t ||
        !stride_h || !stride_w || depth < kd || height < kh || width < kw) {
        h3_cuda_set_error(error, error_size, "invalid conv3d_nhwc args");
        return 0;
    }
    od = (depth - kd) / stride_t + 1;
    oh = (height - kh) / stride_h + 1;
    ow = (width - kw) / stride_w + 1;
    if (!od || !oh || !ow) {
        h3_cuda_set_error(error, error_size, "conv3d_nhwc empty output");
        return 0;
    }
    in_n = (size_t)depth * height * width * cin;
    w_n = (size_t)cout * cin * kd * kh * kw;
    out_n = (size_t)od * oh * ow * cout;
    if (!h3_cuda_bind(cuda, error, error_size)) return 0;
    if (!h3_cuda_grow_scratch(&cuda->d_conv3d_in, &cuda->conv3d_in_bytes,
                              in_n * sizeof(float), error, error_size,
                              "conv3d in") ||
        !h3_cuda_grow_scratch(&cuda->d_conv3d_w, &cuda->conv3d_w_bytes,
                              w_n * sizeof(float), error, error_size,
                              "conv3d W") ||
        !h3_cuda_grow_scratch(&cuda->d_conv3d_y, &cuda->conv3d_y_bytes,
                              out_n * sizeof(float), error, error_size,
                              "conv3d y") ||
        !h3_cuda_grow_scratch(&cuda->d_conv3d_bias, &cuda->conv3d_bias_bytes,
                              (size_t)cout * sizeof(float), error, error_size,
                              "conv3d bias"))
        return 0;
    status = cudaMemcpy(cuda->d_conv3d_in, input, in_n * sizeof(float),
                        cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "conv3d in H2D", status);
        return 0;
    }
    status = cudaMemcpy(cuda->d_conv3d_w, weight, w_n * sizeof(float),
                        cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "conv3d W H2D", status);
        return 0;
    }
    status = cudaMemcpy(cuda->d_conv3d_bias, bias, (size_t)cout * sizeof(float),
                        cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "conv3d bias H2D", status);
        return 0;
    }
    blocks = (int)((out_n + (size_t)threads - 1) / (size_t)threads);
    if (blocks < 1) blocks = 1;
    h3_cuda_conv3d_nhwc_bias_kernel<<<blocks, threads>>>(
        (const float *)cuda->d_conv3d_in, (const float *)cuda->d_conv3d_w,
        (const float *)cuda->d_conv3d_bias, (float *)cuda->d_conv3d_y, depth,
        height, width, cin, cout, od, oh, ow, kd, kh, kw, stride_t, stride_h,
        stride_w);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "conv3d kernel", status);
        return 0;
    }
    status = cudaMemcpy(output, cuda->d_conv3d_y, out_n * sizeof(float),
                        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "conv3d y D2H", status);
        return 0;
    }
    return 1;
}

extern "C" int h3_cuda_gemm_f16_dd(h3_cuda *cuda, const void *d_x_half,
                                   const void *d_weight_half, size_t m,
                                   size_t n, size_t k, void *d_y_half,
                                   double *gemm_ms, char *error,
                                   size_t error_size) {
    cublasStatus_t st;
    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);
    if (gemm_ms) *gemm_ms = 0.0;
    if (!cuda || !d_x_half || !d_weight_half || !d_y_half || m == 0 || n == 0 ||
        k == 0) {
        h3_cuda_set_error(error, error_size, "invalid gemm_f16_dd");
        return 0;
    }
    st = cublasGemmEx(cuda->cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)n, (int)m,
                      (int)k, &alpha, d_weight_half, CUDA_R_16F, (int)k,
                      d_x_half, CUDA_R_16F, (int)k, &beta, d_y_half, CUDA_R_16F,
                      (int)n, CUBLAS_COMPUTE_16F,
                      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (st != CUBLAS_STATUS_SUCCESS) {
        h3_cuda_set_error(error, error_size, "cublasGemmEx f16 failed");
        return 0;
    }
    return 1;
}

int h3_cuda_gemm_f32_dd_ex(h3_cuda *cuda, const float *d_x,
                           const float *d_weight, size_t m, size_t n, size_t k,
                           float *d_y, float alpha, float beta, double *gemm_ms,
                           char *error, size_t error_size) {
    cublasStatus_t cublas_status;
    cudaEvent_t ev_start = NULL, ev_stop = NULL;
    float elapsed_ms = 0.0f;
    int quiet;

    if (gemm_ms) *gemm_ms = 0.0;
    if (!cuda || !d_x || !d_weight || !d_y || m == 0 || n == 0 || k == 0) {
        h3_cuda_set_error(error, error_size, "invalid h3_cuda_gemm_f32_dd args");
        return 0;
    }
    if (!h3_cuda_bind(cuda, error, error_size)) return 0;
    /* No EventSynchronize unless timing was requested. */
    quiet = cuda->quiet || cuda->graph_capture || !gemm_ms;
    if (!quiet) {
        if (cudaEventCreate(&ev_start) != cudaSuccess ||
            cudaEventCreate(&ev_stop) != cudaSuccess) {
            h3_cuda_set_error(error, error_size, "gemm_f32_dd event failed");
            if (ev_start) cudaEventDestroy(ev_start);
            if (ev_stop) cudaEventDestroy(ev_stop);
            return 0;
        }
        cudaEventRecord(ev_start, 0);
    }
    cublas_status = cublasSgemm(cuda->cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)n,
                                (int)m, (int)k, &alpha, d_weight, (int)k, d_x,
                                (int)k, &beta, d_y, (int)n);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        if (ev_start) cudaEventDestroy(ev_start);
        if (ev_stop) cudaEventDestroy(ev_stop);
        h3_cuda_set_error(error, error_size, "cublasSgemm dd failed");
        return 0;
    }
    if (!quiet) {
        cudaEventRecord(ev_stop, 0);
        cudaEventSynchronize(ev_stop);
        cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop);
        if (gemm_ms) *gemm_ms = (double)elapsed_ms;
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_stop);
    }
    return 1;
}

int h3_cuda_gemm_f32_dd(h3_cuda *cuda, const float *d_x, const float *d_weight,
                        size_t m, size_t n, size_t k, float *d_y,
                        double *gemm_ms, char *error, size_t error_size) {
    return h3_cuda_gemm_f32_dd_ex(cuda, d_x, d_weight, m, n, k, d_y, 1.0f, 0.0f,
                                  gemm_ms, error, error_size);
}

int h3_cuda_gemm_f32_dw(h3_cuda *cuda, const float *x, const float *d_weight,
                        size_t m, size_t n, size_t k, float *y, double *gemm_ms,
                        char *error, size_t error_size) {
    float *d_x = NULL;
    float *d_y = NULL;
    cudaError_t status;
    cublasStatus_t cublas_status;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cudaEvent_t ev_start = NULL, ev_stop = NULL;
    float elapsed_ms = 0.0f;
    int ok = 0;

    if (gemm_ms) *gemm_ms = 0.0;
    if (!cuda || !x || !d_weight || !y || m == 0 || n == 0 || k == 0) {
        h3_cuda_set_error(error, error_size, "invalid h3_cuda_gemm_f32_dw args");
        return 0;
    }
    status = cudaSetDevice(cuda->device_index);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaSetDevice", status);
        return 0;
    }
    if (cudaMalloc((void **)&d_x, m * k * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&d_y, m * n * sizeof(float)) != cudaSuccess) {
        h3_cuda_set_error(error, error_size, "gemm_f32_dw OOM");
        goto cleanup;
    }
    if (cudaMemcpy(d_x, x, m * k * sizeof(float), cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        h3_cuda_set_error(error, error_size, "gemm_f32_dw x H2D failed");
        goto cleanup;
    }
    {
        int quiet = cuda->quiet || cuda->graph_capture || !gemm_ms;
        if (!quiet) {
            if (cudaEventCreate(&ev_start) != cudaSuccess ||
                cudaEventCreate(&ev_stop) != cudaSuccess) {
                h3_cuda_set_error(error, error_size,
                                  "gemm_f32_dw event create failed");
                goto cleanup;
            }
            cudaEventRecord(ev_start, 0);
        }
        cublas_status = cublasSgemm(cuda->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                    (int)n, (int)m, (int)k, &alpha, d_weight,
                                    (int)k, d_x, (int)k, &beta, d_y, (int)n);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            h3_cuda_set_error(error, error_size, "cublasSgemm dw failed");
            goto cleanup;
        }
        if (!quiet) {
            cudaEventRecord(ev_stop, 0);
            cudaEventSynchronize(ev_stop);
            cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop);
            if (gemm_ms) *gemm_ms = (double)elapsed_ms;
        }
    }
    if (cudaMemcpy(y, d_y, m * n * sizeof(float), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        h3_cuda_set_error(error, error_size, "gemm_f32_dw y D2H failed");
        goto cleanup;
    }
    ok = 1;
cleanup:
    if (ev_start) cudaEventDestroy(ev_start);
    if (ev_stop) cudaEventDestroy(ev_stop);
    if (d_x) cudaFree(d_x);
    if (d_y) cudaFree(d_y);
    return ok;
}

static void h3_cuda_set_error_cublas(char *error, size_t error_size,
                                     const char *prefix, cublasStatus_t st) {
    if (!error || !error_size) return;
    snprintf(error, error_size, "%s: cublasStatus=%d",
             prefix ? prefix : "cuBLAS", (int)st);
}

/* Try cuBLASLt INT8×INT8→INT32: same layout as cublasSgemm(OP_T,OP_N).
 * Descriptors + heuristic are cached per (m,n,k) — 50L×4 GEMMs must not
 * rebuild the plan every call. Returns 1 on success, 0 if no algo. */
static int h3_cuda_lt_gemm_i8_i32(h3_cuda *cuda, const int8_t *d_w,
                                  const int8_t *d_x, int32_t *d_acc, size_t m,
                                  size_t n, size_t k, char *error,
                                  size_t error_size) {
    void *workspace = NULL;
    size_t workspace_bytes = 64 * 1024 * 1024;
    const int32_t alpha = 1;
    const int32_t beta = 0;
    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;
    cublasStatus_t st;
    char ws_err[128];
    int slot = -1;
    int i;
    int returned = 0;

    if (!h3_cuda_ensure_lt_workspace(cuda, workspace_bytes, ws_err,
                                     sizeof(ws_err))) {
        h3_cuda_set_error(error, error_size, ws_err[0] ? ws_err
                                                       : "lt workspace");
        return 0;
    }
    workspace = cuda->d_lt_workspace;

    for (i = 0; i < cuda->lt_i8_n; i++) {
        if (cuda->lt_i8[i].m == m && cuda->lt_i8[i].n == n &&
            cuda->lt_i8[i].k == k) {
            slot = i;
            break;
        }
    }
    if (slot >= 0 && cuda->lt_i8[slot].state < 0) {
        h3_cuda_set_error(error, error_size, "cublasLt INT8 cached miss");
        return 0;
    }
    if (slot < 0) {
        if (cuda->lt_i8_n >= 8) {
            h3_cuda_set_error(error, error_size, "cublasLt INT8 cache full");
            return 0;
        }
        slot = cuda->lt_i8_n;
        memset(&cuda->lt_i8[slot], 0, sizeof(cuda->lt_i8[slot]));
        cuda->lt_i8[slot].m = m;
        cuda->lt_i8[slot].n = n;
        cuda->lt_i8[slot].k = k;
        st = cublasLtMatmulDescCreate(&cuda->lt_i8[slot].desc,
                                      CUBLAS_COMPUTE_32I, CUDA_R_32I);
        if (st != CUBLAS_STATUS_SUCCESS) {
            h3_cuda_set_error_cublas(error, error_size, "cublasLtMatmulDescCreate",
                                     st);
            goto fail_slot;
        }
        st = cublasLtMatmulDescSetAttribute(cuda->lt_i8[slot].desc,
                                            CUBLASLT_MATMUL_DESC_TRANSA,
                                            &transa, sizeof(transa));
        if (st != CUBLAS_STATUS_SUCCESS) {
            h3_cuda_set_error_cublas(error, error_size, "set TRANSA", st);
            goto fail_slot;
        }
        st = cublasLtMatmulDescSetAttribute(cuda->lt_i8[slot].desc,
                                            CUBLASLT_MATMUL_DESC_TRANSB,
                                            &transb, sizeof(transb));
        if (st != CUBLAS_STATUS_SUCCESS) {
            h3_cuda_set_error_cublas(error, error_size, "set TRANSB", st);
            goto fail_slot;
        }
        st = cublasLtMatrixLayoutCreate(&cuda->lt_i8[slot].a, CUDA_R_8I,
                                        (uint64_t)k, (uint64_t)n, (int64_t)k);
        if (st != CUBLAS_STATUS_SUCCESS) {
            h3_cuda_set_error_cublas(error, error_size, "layout A", st);
            goto fail_slot;
        }
        st = cublasLtMatrixLayoutCreate(&cuda->lt_i8[slot].b, CUDA_R_8I,
                                        (uint64_t)k, (uint64_t)m, (int64_t)k);
        if (st != CUBLAS_STATUS_SUCCESS) {
            h3_cuda_set_error_cublas(error, error_size, "layout B", st);
            goto fail_slot;
        }
        st = cublasLtMatrixLayoutCreate(&cuda->lt_i8[slot].c, CUDA_R_32I,
                                        (uint64_t)n, (uint64_t)m, (int64_t)n);
        if (st != CUBLAS_STATUS_SUCCESS) {
            h3_cuda_set_error_cublas(error, error_size, "layout C", st);
            goto fail_slot;
        }
        st = cublasLtMatmulPreferenceCreate(&cuda->lt_i8[slot].pref);
        if (st != CUBLAS_STATUS_SUCCESS) {
            h3_cuda_set_error_cublas(error, error_size, "preferenceCreate", st);
            goto fail_slot;
        }
        st = cublasLtMatmulPreferenceSetAttribute(
            cuda->lt_i8[slot].pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_bytes, sizeof(workspace_bytes));
        if (st != CUBLAS_STATUS_SUCCESS) {
            h3_cuda_set_error_cublas(error, error_size, "set workspace pref",
                                     st);
            goto fail_slot;
        }
        {
            cublasLtMatmulHeuristicResult_t heurs[16];
            int hi, pick = -1;
            st = cublasLtMatmulAlgoGetHeuristic(
                cuda->cublasLt, cuda->lt_i8[slot].desc, cuda->lt_i8[slot].a,
                cuda->lt_i8[slot].b, cuda->lt_i8[slot].c, cuda->lt_i8[slot].c,
                cuda->lt_i8[slot].pref, 16, heurs, &returned);
            if (st != CUBLAS_STATUS_SUCCESS || returned <= 0) {
                h3_cuda_set_error(error, error_size,
                                  "cublasLtMatmulAlgoGetHeuristic: no INT8 algo");
                goto fail_slot;
            }
            for (hi = 0; hi < returned; hi++) {
                if (heurs[hi].workspaceSize <= workspace_bytes) {
                    pick = hi;
                    break;
                }
            }
            if (pick < 0) {
                h3_cuda_set_error(error, error_size,
                                  "cublasLt algo needs workspace > 64MiB");
                goto fail_slot;
            }
            cuda->lt_i8[slot].heur = heurs[pick];
        }
        cuda->lt_i8[slot].state = 1;
        cuda->lt_i8_n++;
        if (!cuda->lt_i8_logged) {
            fprintf(stderr,
                    "[h3] int8_gemm: cublasLt cached m=%zu n=%zu k=%zu "
                    "ws=%zu\n",
                    m, n, k, (size_t)cuda->lt_i8[slot].heur.workspaceSize);
            cuda->lt_i8_logged = 1;
        }
    }

    st = cublasLtMatmul(cuda->cublasLt, cuda->lt_i8[slot].desc, &alpha, d_w,
                        cuda->lt_i8[slot].a, d_x, cuda->lt_i8[slot].b, &beta,
                        d_acc, cuda->lt_i8[slot].c, d_acc, cuda->lt_i8[slot].c,
                        &cuda->lt_i8[slot].heur.algo, workspace,
                        cuda->lt_i8[slot].heur.workspaceSize,
                        h3_cuda_op_st(cuda));
    if (st != CUBLAS_STATUS_SUCCESS) {
        h3_cuda_set_error_cublas(error, error_size, "cublasLtMatmul", st);
        return 0;
    }
    return 1;

fail_slot:
    if (cuda->lt_i8[slot].pref)
        cublasLtMatmulPreferenceDestroy(cuda->lt_i8[slot].pref);
    if (cuda->lt_i8[slot].c) cublasLtMatrixLayoutDestroy(cuda->lt_i8[slot].c);
    if (cuda->lt_i8[slot].b) cublasLtMatrixLayoutDestroy(cuda->lt_i8[slot].b);
    if (cuda->lt_i8[slot].a) cublasLtMatrixLayoutDestroy(cuda->lt_i8[slot].a);
    if (cuda->lt_i8[slot].desc)
        cublasLtMatmulDescDestroy(cuda->lt_i8[slot].desc);
    memset(&cuda->lt_i8[slot], 0, sizeof(cuda->lt_i8[slot]));
    cuda->lt_i8[slot].m = m;
    cuda->lt_i8[slot].n = n;
    cuda->lt_i8[slot].k = k;
    cuda->lt_i8[slot].state = -1;
    if (slot == cuda->lt_i8_n) cuda->lt_i8_n++;
    return 0;
}

/* Fallback: cublasGemmEx INT8→INT32 with same OP_T/OP_N layout. */
static int h3_cuda_ex_gemm_i8_i32(h3_cuda *cuda, const int8_t *d_w,
                                  const int8_t *d_x, int32_t *d_acc, size_t m,
                                  size_t n, size_t k, char *error,
                                  size_t error_size) {
    const int32_t alpha = 1;
    const int32_t beta = 0;
    cublasStatus_t st = cublasGemmEx(
        cuda->cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)n, (int)m, (int)k, &alpha,
        d_w, CUDA_R_8I, (int)k, d_x, CUDA_R_8I, (int)k, &beta, d_acc,
        CUDA_R_32I, (int)n, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) {
        h3_cuda_set_error_cublas(error, error_size, "cublasGemmEx INT8", st);
        return 0;
    }
    return 1;
}

static void h3_cuda_kernel_gemm_i8_i32(const h3_cuda *cuda, const int8_t *d_x,
                                       const int8_t *d_w, int32_t *d_acc,
                                       size_t m, size_t n, size_t k) {
    dim3 block(16, 16);
    dim3 grid((unsigned)((n + block.x - 1) / block.x),
              (unsigned)((m + block.y - 1) / block.y));
    h3_cuda_gemm_i8_kernel<<<grid, block, 0, h3_cuda_op_st(cuda)>>>(
        d_x, d_w, d_acc, m, n, k);
}

/* Device-pointer INT8 GEMM + scale dequant. d_w / d_sc already on device.
 * Uploads x + act_scales; downloads y. *gemm_ms = GEMM+dequant only. */
static int h3_cuda_gemm_i8_dequant_f32_dev(
    h3_cuda *cuda, const int8_t *x_i8, const int8_t *d_w,
    const float *act_scales, int per_tensor_scale, const float *d_sc, size_t m,
    size_t n, size_t k, float *y_f32, double *gemm_ms, char *backend_out,
    size_t backend_out_size, char *error, size_t error_size) {
    int8_t *d_x = NULL;
    int32_t *d_acc = NULL;
    float *d_y = NULL;
    float *d_act = NULL;
    cudaError_t status;
    cudaEvent_t ev_start = NULL;
    cudaEvent_t ev_stop = NULL;
    float elapsed_ms = 0.0f;
    char gemm_err[256];
    int gemm_ok = 0;
    const char *gemm_backend = "unknown";
    int ok = 0;
    size_t act_count;

    if (gemm_ms) *gemm_ms = 0.0;
    if (backend_out && backend_out_size) backend_out[0] = '\0';
    if (!cuda || !x_i8 || !d_w || !act_scales || !d_sc || !y_f32 || m == 0 ||
        n == 0 || k == 0) {
        h3_cuda_set_error(error, error_size,
                          "invalid h3_cuda_gemm_i8_dequant_f32 args");
        return 0;
    }
    act_count = per_tensor_scale ? 1 : m;

    status = cudaSetDevice(cuda->device_index);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaSetDevice", status);
        return 0;
    }

    status = cudaMalloc((void **)&d_x, m * k);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaMalloc x_i8", status);
        goto cleanup;
    }
    status = cudaMalloc((void **)&d_acc, m * n * sizeof(int32_t));
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaMalloc acc", status);
        goto cleanup;
    }
    status = cudaMalloc((void **)&d_y, m * n * sizeof(float));
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaMalloc y", status);
        goto cleanup;
    }
    status = cudaMalloc((void **)&d_act, act_count * sizeof(float));
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaMalloc act_scales",
                               status);
        goto cleanup;
    }

    status = cudaMemcpy(d_x, x_i8, m * k, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaMemcpy x_i8", status);
        goto cleanup;
    }
    status = cudaMemcpy(d_act, act_scales, act_count * sizeof(float),
                        cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaMemcpy act_scales",
                               status);
        goto cleanup;
    }

    status = cudaEventCreate(&ev_start);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaEventCreate start",
                               status);
        goto cleanup;
    }
    status = cudaEventCreate(&ev_stop);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaEventCreate stop",
                               status);
        goto cleanup;
    }

    status = cudaEventRecord(ev_start, 0);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaEventRecord start",
                               status);
        goto cleanup;
    }

    gemm_err[0] = '\0';
    if (h3_cuda_lt_gemm_i8_i32(cuda, d_w, d_x, d_acc, m, n, k, gemm_err,
                               sizeof(gemm_err))) {
        gemm_ok = 1;
        gemm_backend = "cublasLt";
    } else if (h3_cuda_ex_gemm_i8_i32(cuda, d_w, d_x, d_acc, m, n, k, gemm_err,
                                      sizeof(gemm_err))) {
        gemm_ok = 1;
        gemm_backend = "cublasGemmEx";
    } else {
        h3_cuda_kernel_gemm_i8_i32(cuda, d_x, d_w, d_acc, m, n, k);
        status = cudaGetLastError();
        if (status != cudaSuccess) {
            h3_cuda_set_error_cuda(error, error_size, "INT8 GEMM kernel",
                                   status);
            goto cleanup;
        }
        gemm_ok = 1;
        gemm_backend = "cuda_kernel";
    }

    if (!gemm_ok) {
        h3_cuda_set_error(error, error_size,
                          gemm_err[0] ? gemm_err : "INT8 GEMM failed");
        goto cleanup;
    }

    {
        unsigned threads = 256;
        dim3 dq_grid((unsigned)((n + (size_t)threads - 1) / (size_t)threads),
                     (unsigned)m, 1);
        h3_cuda_dequant_i32_kernel<<<dq_grid, threads, 0, h3_cuda_op_st(cuda)>>>(
            d_acc, d_act, per_tensor_scale ? 1 : 0, d_sc, d_y, m, n);
        status = cudaGetLastError();
        if (status != cudaSuccess) {
            h3_cuda_set_error_cuda(error, error_size, "dequant kernel", status);
            goto cleanup;
        }
    }

    status = cudaEventRecord(ev_stop, 0);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaEventRecord stop",
                               status);
        goto cleanup;
    }
    status = cudaEventSynchronize(ev_stop);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaEventSynchronize",
                               status);
        goto cleanup;
    }
    status = cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaEventElapsedTime",
                               status);
        goto cleanup;
    }
    if (gemm_ms) *gemm_ms = (double)elapsed_ms;
    if (backend_out && backend_out_size) {
        snprintf(backend_out, backend_out_size, "%s", gemm_backend);
    }

    status =
        cudaMemcpy(y_f32, d_y, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaMemcpy y D2H", status);
        goto cleanup;
    }
    ok = 1;

cleanup:
    if (ev_start) cudaEventDestroy(ev_start);
    if (ev_stop) cudaEventDestroy(ev_stop);
    if (d_x) cudaFree(d_x);
    if (d_acc) cudaFree(d_acc);
    if (d_y) cudaFree(d_y);
    if (d_act) cudaFree(d_act);
    return ok;
}

int h3_cuda_gemm_i8_dequant_f32(h3_cuda *cuda, const int8_t *x_i8,
                                const int8_t *w_i8, const float *act_scales,
                                int per_tensor_scale, const float *s_channel,
                                size_t m, size_t n, size_t k, float *y_f32,
                                double *gemm_ms, char *backend_out,
                                size_t backend_out_size, char *error,
                                size_t error_size) {
    int8_t *d_w = NULL;
    float *d_sc = NULL;
    cudaError_t status;
    int ok = 0;

    if (gemm_ms) *gemm_ms = 0.0;
    if (backend_out && backend_out_size) backend_out[0] = '\0';
    if (!cuda || !x_i8 || !w_i8 || !act_scales || !s_channel || !y_f32 ||
        m == 0 || n == 0 || k == 0) {
        h3_cuda_set_error(error, error_size,
                          "invalid h3_cuda_gemm_i8_dequant_f32 args");
        return 0;
    }

    status = cudaSetDevice(cuda->device_index);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaSetDevice", status);
        return 0;
    }

    status = cudaMalloc((void **)&d_w, n * k);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaMalloc w_i8", status);
        goto cleanup;
    }
    status = cudaMalloc((void **)&d_sc, n * sizeof(float));
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaMalloc s_channel",
                               status);
        goto cleanup;
    }
    status = cudaMemcpy(d_w, w_i8, n * k, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaMemcpy w_i8", status);
        goto cleanup;
    }
    status = cudaMemcpy(d_sc, s_channel, n * sizeof(float),
                        cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaMemcpy s_channel",
                               status);
        goto cleanup;
    }

    ok = h3_cuda_gemm_i8_dequant_f32_dev(
        cuda, x_i8, d_w, act_scales, per_tensor_scale, d_sc, m, n, k, y_f32,
        gemm_ms, backend_out, backend_out_size, error, error_size);

cleanup:
    if (d_w) cudaFree(d_w);
    if (d_sc) cudaFree(d_sc);
    return ok;
}

int h3_cuda_w4a8_weight_create(h3_cuda *cuda, const int8_t *packed_host,
                               size_t n, size_t k_half,
                               const float *codebook16,
                               const uint8_t *s_rel_f8,
                               const float *s_channel, int group_size,
                               h3_cuda_w4a8_weight **out, double *unpack_ms,
                               double *h2d_ms, char *error, size_t error_size) {
    h3_cuda_w4a8_weight *weight = NULL;
    int8_t *d_packed = NULL;
    float *d_codebook = NULL;
    uint8_t *d_s_rel = NULL;
    cudaError_t status;
    cudaEvent_t ev_h2d0 = NULL, ev_h2d1 = NULL;
    cudaEvent_t ev_u0 = NULL, ev_u1 = NULL;
    float ms_h2d = 0.0f, ms_unpack = 0.0f;
    const size_t k = k_half * 2;
    const size_t groups = k / (size_t)group_size;
    int ok = 0;

    if (out) *out = NULL;
    if (unpack_ms) *unpack_ms = 0.0;
    if (h2d_ms) *h2d_ms = 0.0;
    if (!cuda || !packed_host || !codebook16 || !s_rel_f8 || !s_channel ||
        !out || n == 0 || k_half == 0 || group_size < 4) {
        h3_cuda_set_error(error, error_size,
                          "invalid h3_cuda_w4a8_weight_create args");
        return 0;
    }
    if (k % (size_t)group_size != 0) {
        h3_cuda_set_error(error, error_size, "K not divisible by group_size");
        return 0;
    }

    status = cudaSetDevice(cuda->device_index);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaSetDevice", status);
        return 0;
    }

    weight = (h3_cuda_w4a8_weight *)calloc(1, sizeof(*weight));
    if (!weight) {
        h3_cuda_set_error(error, error_size, "out of memory");
        return 0;
    }
    weight->cuda = cuda;
    weight->n = n;
    weight->k = k;

    status = cudaMalloc((void **)&d_packed, n * k_half);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaMalloc packed", status);
        goto cleanup;
    }
    status = cudaMalloc((void **)&d_codebook, 16 * sizeof(float));
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaMalloc codebook",
                               status);
        goto cleanup;
    }
    status = cudaMalloc((void **)&d_s_rel, n * groups);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaMalloc s_rel", status);
        goto cleanup;
    }
    status = cudaMalloc((void **)&weight->d_w_i8, n * k);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaMalloc w_i8", status);
        goto cleanup;
    }
    status = cudaMalloc((void **)&weight->d_s_channel, n * sizeof(float));
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaMalloc s_channel",
                               status);
        goto cleanup;
    }

    /* Prefetch / quiet path: no per-weight EventSynchronize (was serializing
     * mid-window unpack against in-flight compute). Optional timing still
     * uses events. Prefer side stream so H2D+unpack can overlap compute. */
    {
        const int want_time = (h2d_ms != NULL) || (unpack_ms != NULL);
        cudaStream_t st =
            (cuda->streams_ready && cuda->streams[0]) ? cuda->streams[0]
                                                       : (cudaStream_t)0;

        if (want_time) {
            status = cudaEventCreate(&ev_h2d0);
            if (status != cudaSuccess) {
                h3_cuda_set_error_cuda(error, error_size, "cudaEventCreate",
                                       status);
                goto cleanup;
            }
            status = cudaEventCreate(&ev_h2d1);
            if (status != cudaSuccess) {
                h3_cuda_set_error_cuda(error, error_size, "cudaEventCreate",
                                       status);
                goto cleanup;
            }
            status = cudaEventCreate(&ev_u0);
            if (status != cudaSuccess) {
                h3_cuda_set_error_cuda(error, error_size, "cudaEventCreate",
                                       status);
                goto cleanup;
            }
            status = cudaEventCreate(&ev_u1);
            if (status != cudaSuccess) {
                h3_cuda_set_error_cuda(error, error_size, "cudaEventCreate",
                                       status);
                goto cleanup;
            }
            status = cudaEventRecord(ev_h2d0, st);
            if (status != cudaSuccess) {
                h3_cuda_set_error_cuda(error, error_size, "cudaEventRecord",
                                       status);
                goto cleanup;
            }
        }

        status = cudaMemcpyAsync(d_packed, packed_host, n * k_half,
                                 cudaMemcpyHostToDevice, st);
        if (status != cudaSuccess) {
            h3_cuda_set_error_cuda(error, error_size, "cudaMemcpy packed",
                                   status);
            goto cleanup;
        }
        status = cudaMemcpyAsync(d_codebook, codebook16, 16 * sizeof(float),
                                 cudaMemcpyHostToDevice, st);
        if (status != cudaSuccess) {
            h3_cuda_set_error_cuda(error, error_size, "cudaMemcpy codebook",
                                   status);
            goto cleanup;
        }
        status = cudaMemcpyAsync(d_s_rel, s_rel_f8, n * groups,
                                 cudaMemcpyHostToDevice, st);
        if (status != cudaSuccess) {
            h3_cuda_set_error_cuda(error, error_size, "cudaMemcpy s_rel",
                                   status);
            goto cleanup;
        }
        status = cudaMemcpyAsync(weight->d_s_channel, s_channel,
                                 n * sizeof(float), cudaMemcpyHostToDevice, st);
        if (status != cudaSuccess) {
            h3_cuda_set_error_cuda(error, error_size, "cudaMemcpy s_channel",
                                   status);
            goto cleanup;
        }
        if (want_time) {
            status = cudaEventRecord(ev_h2d1, st);
            if (status != cudaSuccess) {
                h3_cuda_set_error_cuda(error, error_size, "cudaEventRecord",
                                       status);
                goto cleanup;
            }
            status = cudaEventRecord(ev_u0, st);
            if (status != cudaSuccess) {
                h3_cuda_set_error_cuda(error, error_size, "cudaEventRecord",
                                       status);
                goto cleanup;
            }
        }
        {
            unsigned threads = 256;
            size_t total = n * k_half;
            unsigned blocks =
                (unsigned)((total + (size_t)threads - 1) / (size_t)threads);
            h3_cuda_w4a8_unpack_kernel<<<blocks, threads, 0, st>>>(
                d_packed, d_codebook, d_s_rel, weight->d_w_i8, n, k_half,
                group_size);
            status = cudaGetLastError();
            if (status != cudaSuccess) {
                h3_cuda_set_error_cuda(error, error_size, "W4A8 unpack kernel",
                                       status);
                goto cleanup;
            }
        }
        if (want_time) {
            status = cudaEventRecord(ev_u1, st);
            if (status != cudaSuccess) {
                h3_cuda_set_error_cuda(error, error_size, "cudaEventRecord",
                                       status);
                goto cleanup;
            }
            status = cudaEventSynchronize(ev_u1);
            if (status != cudaSuccess) {
                h3_cuda_set_error_cuda(error, error_size,
                                       "cudaEventSynchronize", status);
                goto cleanup;
            }
            status = cudaEventElapsedTime(&ms_h2d, ev_h2d0, ev_h2d1);
            if (status != cudaSuccess) {
                h3_cuda_set_error_cuda(error, error_size,
                                       "cudaEventElapsedTime h2d", status);
                goto cleanup;
            }
            status = cudaEventElapsedTime(&ms_unpack, ev_u0, ev_u1);
            if (status != cudaSuccess) {
                h3_cuda_set_error_cuda(error, error_size,
                                       "cudaEventElapsedTime unpack", status);
                goto cleanup;
            }
            if (h2d_ms) *h2d_ms = (double)ms_h2d;
            if (unpack_ms) *unpack_ms = (double)ms_unpack;
        } else {
            /* Drain copy stream before freeing H2D temps. Overlap with
             * default-stream compute when caller already launched work. */
            status = cudaStreamSynchronize(st);
            if (status != cudaSuccess) {
                h3_cuda_set_error_cuda(error, error_size,
                                       "cudaStreamSynchronize copy", status);
                goto cleanup;
            }
        }
    }

    *out = weight;
    weight = NULL;
    ok = 1;

cleanup:
    if (ev_h2d0) cudaEventDestroy(ev_h2d0);
    if (ev_h2d1) cudaEventDestroy(ev_h2d1);
    if (ev_u0) cudaEventDestroy(ev_u0);
    if (ev_u1) cudaEventDestroy(ev_u1);
    if (d_packed) cudaFree(d_packed);
    if (d_codebook) cudaFree(d_codebook);
    if (d_s_rel) cudaFree(d_s_rel);
    if (weight) {
        if (weight->d_w_i8) cudaFree(weight->d_w_i8);
        if (weight->d_s_channel) cudaFree(weight->d_s_channel);
        if (weight->ready_event_valid && weight->ready_event)
            cudaEventDestroy(weight->ready_event);
        free(weight);
    }
    return ok;
}

void *h3_cuda_host_alloc(size_t bytes, int *pinned_out) {
    void *p = NULL;
    if (pinned_out) *pinned_out = 0;
    if (bytes == 0) return NULL;
    if (cudaHostAlloc(&p, bytes, cudaHostAllocDefault) == cudaSuccess) {
        if (pinned_out) *pinned_out = 1;
        return p;
    }
    p = malloc(bytes);
    return p;
}

void h3_cuda_host_free(void *p, int pinned) {
    if (!p) return;
    if (pinned)
        cudaFreeHost(p);
    else
        free(p);
}

int h3_cuda_w4a8_weight_create_from_i8(h3_cuda *cuda, int8_t *w_i8, size_t n,
                                       size_t k, float *s_channel,
                                       h3_cuda_w4a8_weight **out,
                                       double *h2d_ms, int borrow_host,
                                       char *error, size_t error_size) {
    h3_cuda_w4a8_weight *weight = NULL;
    cudaError_t status;
    cudaEvent_t ev0 = NULL, ev1 = NULL;
    float ms = 0.0f;
    int ok = 0;

    if (out) *out = NULL;
    if (h2d_ms) *h2d_ms = 0.0;
    if (!cuda || !w_i8 || !s_channel || !out || n == 0 || k == 0) {
        h3_cuda_set_error(error, error_size,
                          "invalid h3_cuda_w4a8_weight_create_from_i8 args");
        return 0;
    }
    status = cudaSetDevice(cuda->device_index);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaSetDevice", status);
        return 0;
    }
    weight = (h3_cuda_w4a8_weight *)calloc(1, sizeof(*weight));
    if (!weight) {
        h3_cuda_set_error(error, error_size, "out of memory");
        return 0;
    }
    weight->cuda = cuda;
    weight->n = n;
    weight->k = k;
    status = cudaMalloc((void **)&weight->d_w_i8, n * k);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaMalloc w_i8", status);
        goto cleanup;
    }
    status = cudaMalloc((void **)&weight->d_s_channel, n * sizeof(float));
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaMalloc s_channel",
                               status);
        goto cleanup;
    }
    {
        const int want_time = (h2d_ms != NULL);
        /* Prefer side stream so INT8 H2D can overlap default-stream compute.
         * Never sync the default stream for unpack/upload. */
        cudaStream_t st =
            (cuda->streams_ready && cuda->streams[0]) ? cuda->streams[0]
                                                       : (cudaStream_t)0;
        if (want_time) {
            status = cudaEventCreate(&ev0);
            if (status != cudaSuccess) {
                h3_cuda_set_error_cuda(error, error_size, "cudaEventCreate",
                                       status);
                goto cleanup;
            }
            status = cudaEventCreate(&ev1);
            if (status != cudaSuccess) {
                h3_cuda_set_error_cuda(error, error_size, "cudaEventCreate",
                                       status);
                goto cleanup;
            }
            status = cudaEventRecord(ev0, st);
            if (status != cudaSuccess) {
                h3_cuda_set_error_cuda(error, error_size, "cudaEventRecord",
                                       status);
                goto cleanup;
            }
        }
        status = cudaMemcpyAsync(weight->d_w_i8, w_i8, n * k,
                                 cudaMemcpyHostToDevice, st);
        if (status != cudaSuccess) {
            h3_cuda_set_error_cuda(error, error_size, "cudaMemcpyAsync w_i8",
                                   status);
            goto cleanup;
        }
        status = cudaMemcpyAsync(weight->d_s_channel, s_channel,
                                 n * sizeof(float), cudaMemcpyHostToDevice, st);
        if (status != cudaSuccess) {
            h3_cuda_set_error_cuda(error, error_size,
                                   "cudaMemcpyAsync s_channel", status);
            goto cleanup;
        }
        if (want_time) {
            status = cudaEventRecord(ev1, st);
            if (status != cudaSuccess) {
                h3_cuda_set_error_cuda(error, error_size, "cudaEventRecord",
                                       status);
                goto cleanup;
            }
            status = cudaEventSynchronize(ev1);
            if (status != cudaSuccess) {
                h3_cuda_set_error_cuda(error, error_size,
                                       "cudaEventSynchronize", status);
                goto cleanup;
            }
            status = cudaEventElapsedTime(&ms, ev0, ev1);
            if (status == cudaSuccess && h2d_ms) *h2d_ms = (double)ms;
            /* Timing path drained the copy; free only if we own host. */
            if (!borrow_host) {
                free(w_i8);
                free(s_channel);
            }
        } else if (borrow_host) {
            /* Caller (INT8 host cache) retains staging; only track ready. */
            status = cudaEventCreateWithFlags(&weight->ready_event,
                                              cudaEventDisableTiming);
            if (status != cudaSuccess) {
                h3_cuda_set_error_cuda(error, error_size, "cudaEventCreate",
                                       status);
                goto cleanup;
            }
            status = cudaEventRecord(weight->ready_event, st);
            if (status != cudaSuccess) {
                cudaEventDestroy(weight->ready_event);
                h3_cuda_set_error_cuda(error, error_size, "cudaEventRecord",
                                       status);
                goto cleanup;
            }
            weight->ready_event_valid = 1;
            w_i8 = NULL;
            s_channel = NULL;
        } else {
            /* Keep host alive until GEMM wait_ready / weight_free. Record
             * ready on the copy stream (side stream preferred). */
            weight->h_w_owned = w_i8;
            weight->h_s_owned = s_channel;
            status = cudaEventCreateWithFlags(&weight->ready_event,
                                              cudaEventDisableTiming);
            if (status != cudaSuccess) {
                h3_cuda_set_error_cuda(error, error_size, "cudaEventCreate",
                                       status);
                weight->h_w_owned = NULL;
                weight->h_s_owned = NULL;
                goto cleanup;
            }
            status = cudaEventRecord(weight->ready_event, st);
            if (status != cudaSuccess) {
                cudaEventDestroy(weight->ready_event);
                weight->h_w_owned = NULL;
                weight->h_s_owned = NULL;
                h3_cuda_set_error_cuda(error, error_size, "cudaEventRecord",
                                       status);
                goto cleanup;
            }
            weight->ready_event_valid = 1;
            w_i8 = NULL;
            s_channel = NULL;
        }
    }
    *out = weight;
    weight = NULL;
    ok = 1;

cleanup:
    if (ev0) cudaEventDestroy(ev0);
    if (ev1) cudaEventDestroy(ev1);
    if (weight) {
        if (weight->d_w_i8) cudaFree(weight->d_w_i8);
        if (weight->d_s_channel) cudaFree(weight->d_s_channel);
        if (weight->ready_event_valid && weight->ready_event)
            cudaEventDestroy(weight->ready_event);
        free(weight->h_w_owned);
        free(weight->h_s_owned);
        free(weight);
    }
    return ok;
}

static void h3_cuda_w4a8_weight_release_host(h3_cuda_w4a8_weight *weight) {
    if (!weight) return;
    free(weight->h_w_owned);
    free(weight->h_s_owned);
    weight->h_w_owned = NULL;
    weight->h_s_owned = NULL;
}

int h3_cuda_copy_stream_synchronize(h3_cuda *cuda, char *error,
                                    size_t error_size) {
    cudaStream_t st;
    cudaError_t status;
    if (!cuda) {
        h3_cuda_set_error(error, error_size, "cuda is NULL");
        return 0;
    }
    st = (cuda->streams_ready && cuda->streams[0]) ? cuda->streams[0]
                                                    : (cudaStream_t)0;
    status = cudaStreamSynchronize(st);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaStreamSynchronize copy",
                               status);
        return 0;
    }
    return 1;
}

int h3_cuda_w4a8_weight_on_device(const h3_cuda_w4a8_weight *weight,
                                  int want_dev, char *error, size_t error_size) {
    struct cudaPointerAttributes a;
    cudaError_t ce;
    const void *ptrs[2];
    const char *names[2];
    int i;
    if (!weight || !weight->d_w_i8 || !weight->d_s_channel) {
        h3_cuda_set_error(error, error_size, "weight missing device ptr");
        return 0;
    }
    if (weight->cuda && weight->cuda->device_index != want_dev) {
        snprintf(error, error_size, "weight handle dev %d want %d",
                 weight->cuda->device_index, want_dev);
        return 0;
    }
    ptrs[0] = weight->d_w_i8;
    ptrs[1] = weight->d_s_channel;
    names[0] = "d_w_i8";
    names[1] = "d_s_channel";
    for (i = 0; i < 2; i++) {
        memset(&a, 0, sizeof(a));
        ce = cudaPointerGetAttributes(&a, ptrs[i]);
        if (ce != cudaSuccess) {
            h3_cuda_set_error_cuda(error, error_size, names[i], ce);
            return 0;
        }
        if (a.device != want_dev) {
            snprintf(error, error_size, "%s on dev %d want %d", names[i],
                     a.device, want_dev);
            return 0;
        }
    }
    return 1;
}

int h3_cuda_w4a8_weight_ensure_ready(h3_cuda_w4a8_weight *weight, char *error,
                                     size_t error_size) {
    if (!weight) return 1;
    if (!weight->ready_event_valid || !weight->ready_event) {
        h3_cuda_w4a8_weight_release_host(weight);
        return 1;
    }
    /* GPU-side wait only. EventSynchronize here used to drain the default
     * stream on every newly uploaded linear (window reload). Wait once,
     * then drop the event so later GEMMs on this weight skip the wait.
     * Must wait on op_stream too: Ulysses GEMM is on a non-blocking
     * stream, and waiting only on legacy default lets that GEMM read
     * weights still in flight (IMA at join). U=1 op_stream is NULL. */
    if (cudaStreamWaitEvent((cudaStream_t)0, weight->ready_event, 0) !=
        cudaSuccess) {
        h3_cuda_set_error(error, error_size, "StreamWaitEvent weight ready");
        return 0;
    }
    if (weight->cuda && weight->cuda->op_stream &&
        cudaStreamWaitEvent(weight->cuda->op_stream, weight->ready_event, 0) !=
            cudaSuccess) {
        h3_cuda_set_error(error, error_size, "StreamWaitEvent weight ready (op)");
        return 0;
    }
    cudaEventDestroy(weight->ready_event);
    weight->ready_event = NULL;
    weight->ready_event_valid = 0;
    h3_cuda_w4a8_weight_release_host(weight);
    return 1;
}

static int h3_cuda_w4a8_weight_wait_ready(const h3_cuda_w4a8_weight *weight,
                                          char *error, size_t error_size) {
    return h3_cuda_w4a8_weight_ensure_ready((h3_cuda_w4a8_weight *)weight,
                                            error, error_size);
}

void h3_cuda_w4a8_weight_free(h3_cuda_w4a8_weight *weight) {
    if (!weight) return;
    if (weight->ready_event_valid && weight->ready_event) {
        cudaEventSynchronize(weight->ready_event);
        cudaEventDestroy(weight->ready_event);
        weight->ready_event_valid = 0;
    }
    h3_cuda_w4a8_weight_release_host(weight);
    if (weight->d_w_i8) cudaFree(weight->d_w_i8);
    if (weight->d_s_channel) cudaFree(weight->d_s_channel);
    free(weight);
}

size_t h3_cuda_w4a8_weight_n(const h3_cuda_w4a8_weight *weight) {
    return weight ? weight->n : 0;
}

size_t h3_cuda_w4a8_weight_k(const h3_cuda_w4a8_weight *weight) {
    return weight ? weight->k : 0;
}

int h3_cuda_w4a8_weight_download_int8(const h3_cuda_w4a8_weight *weight,
                                      int8_t *host_out, char *error,
                                      size_t error_size) {
    cudaError_t status;
    if (!weight || !host_out || !weight->d_w_i8) {
        h3_cuda_set_error(error, error_size, "invalid download_int8 args");
        return 0;
    }
    if (!h3_cuda_w4a8_weight_wait_ready(weight, error, error_size)) return 0;
    status = cudaSetDevice(weight->cuda->device_index);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaSetDevice", status);
        return 0;
    }
    status = cudaMemcpy(host_out, weight->d_w_i8, weight->n * weight->k,
                        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaMemcpy w_i8 D2H",
                               status);
        return 0;
    }
    return 1;
}

int h3_cuda_gemm_i8_dequant_f32_w(h3_cuda *cuda,
                                  const h3_cuda_w4a8_weight *weight,
                                  const int8_t *x_i8, const float *act_scales,
                                  int per_tensor_scale, size_t m, float *y_f32,
                                  double *gemm_ms, char *backend_out,
                                  size_t backend_out_size, char *error,
                                  size_t error_size) {
    if (!weight || !weight->d_w_i8 || !weight->d_s_channel) {
        h3_cuda_set_error(error, error_size, "invalid resident weight");
        return 0;
    }
    if (cuda && weight->cuda && cuda != weight->cuda) {
        h3_cuda_set_error(error, error_size, "weight cuda handle mismatch");
        return 0;
    }
    if (!h3_cuda_w4a8_weight_wait_ready(weight, error, error_size)) return 0;
    return h3_cuda_gemm_i8_dequant_f32_dev(
        weight->cuda, x_i8, weight->d_w_i8, act_scales, per_tensor_scale,
        weight->d_s_channel, m, weight->n, weight->k, y_f32, gemm_ms,
        backend_out, backend_out_size, error, error_size);
}

/* ---- Act ConvRot + per-row INT8 quant (device) ---- */

static int h3_cuda_ensure_hadamard(h3_cuda *cuda, int gs, char *error,
                                   size_t error_size) {
    float *host_h = NULL;
    cudaError_t status;
    char detail[256];

    if (!cuda || gs < 4) {
        h3_cuda_set_error(error, error_size, "invalid hadamard request");
        return 0;
    }
    if (cuda->d_hadamard && cuda->hadamard_gs == gs) return 1;

    if (cuda->d_hadamard) {
        cudaFree(cuda->d_hadamard);
        cuda->d_hadamard = NULL;
        cuda->hadamard_gs = 0;
    }

    detail[0] = '\0';
    if (getenv("H3_CONVROT_SYLVESTER") &&
        getenv("H3_CONVROT_SYLVESTER")[0] == '1')
        host_h =
            h3_w4a8_hadamard_sylvester_normalized(gs, detail, sizeof(detail));
    else
        host_h = h3_w4a8_hadamard_normalized(gs, detail, sizeof(detail));
    if (!host_h) {
        snprintf(error, error_size, "hadamard: %s",
                 detail[0] ? detail : "build failed");
        return 0;
    }

    status = cudaSetDevice(cuda->device_index);
    if (status != cudaSuccess) {
        free(host_h);
        h3_cuda_set_error_cuda(error, error_size, "cudaSetDevice", status);
        return 0;
    }
    status = cudaMalloc((void **)&cuda->d_hadamard,
                        (size_t)gs * (size_t)gs * sizeof(float));
    if (status != cudaSuccess) {
        free(host_h);
        h3_cuda_set_error_cuda(error, error_size, "cudaMalloc hadamard",
                               status);
        return 0;
    }
    status = cudaMemcpy(cuda->d_hadamard, host_h,
                        (size_t)gs * (size_t)gs * sizeof(float),
                        cudaMemcpyHostToDevice);
    free(host_h);
    if (status != cudaSuccess) {
        cudaFree(cuda->d_hadamard);
        cuda->d_hadamard = NULL;
        h3_cuda_set_error_cuda(error, error_size, "cudaMemcpy hadamard",
                               status);
        return 0;
    }
    cuda->hadamard_gs = gs;
    return 1;
}

int h3_cuda_pregrow_hadamard(h3_cuda *cuda, int gs, char *error,
                             size_t error_size) {
    return h3_cuda_ensure_hadamard(cuda, gs, error, error_size);
}

int h3_cuda_convrot_act_f32_dd(h3_cuda *cuda, const float *d_x, size_t m,
                               size_t k, int convrot_groupsize, float *d_y,
                               char *error, size_t error_size) {
    size_t n_groups, g;
    int gs;
    const float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t st;
    if (!cuda || !d_x || !d_y || m == 0 || k == 0 || convrot_groupsize < 4) {
        h3_cuda_set_error(error, error_size, "invalid convrot_act_dd args");
        return 0;
    }
    if (d_x == d_y) {
        h3_cuda_set_error(error, error_size, "convrot_act_dd aliases");
        return 0;
    }
    gs = convrot_groupsize;
    if (k % (size_t)gs != 0) {
        h3_cuda_set_error(error, error_size, "K not divisible by convrot_gs");
        return 0;
    }
    if (!h3_cuda_ensure_hadamard(cuda, gs, error, error_size)) return 0;
    n_groups = k / (size_t)gs;
    st = cublasSgemmStridedBatched(
        cuda->cublas, CUBLAS_OP_N, CUBLAS_OP_N, gs, (int)m, gs, &alpha,
        cuda->d_hadamard, gs, 0LL, d_x, (int)k, (long long)gs, &beta, d_y,
        (int)k, (long long)gs, (int)n_groups);
    if (st == CUBLAS_STATUS_SUCCESS) return 1;
    for (g = 0; g < n_groups; g++) {
        st = cublasSgemm(cuda->cublas, CUBLAS_OP_N, CUBLAS_OP_N, gs, (int)m, gs,
                         &alpha, cuda->d_hadamard, gs, d_x + g * (size_t)gs,
                         (int)k, &beta, d_y + g * (size_t)gs, (int)k);
        if (st != CUBLAS_STATUS_SUCCESS) {
            h3_cuda_set_error(error, error_size,
                              "cublasSgemm convrot_act_dd failed");
            return 0;
        }
    }
    return 1;
}

static int h3_cuda_ensure_hadamard_fwht(h3_cuda *cuda, int gs, char *error,
                                        size_t error_size) {
    float *host_h;
    char detail[256];
    cudaError_t status;
    if (!cuda || gs < 2 || (gs & (gs - 1)) != 0) {
        h3_cuda_set_error(error, error_size, "invalid fwht hadamard request");
        return 0;
    }
    if (cuda->d_hadamard_fwht && cuda->hadamard_fwht_gs == gs) return 1;
    if (cuda->d_hadamard_fwht) {
        cudaFree(cuda->d_hadamard_fwht);
        cuda->d_hadamard_fwht = NULL;
        cuda->hadamard_fwht_gs = 0;
    }
    host_h = h3_w4a8_hadamard_sylvester_normalized(gs, detail, sizeof(detail));
    if (!host_h) {
        snprintf(error, error_size, "fwht hadamard: %s",
                 detail[0] ? detail : "build failed");
        return 0;
    }
    status = cudaMalloc((void **)&cuda->d_hadamard_fwht,
                        (size_t)gs * (size_t)gs * sizeof(float));
    if (status != cudaSuccess) {
        free(host_h);
        h3_cuda_set_error_cuda(error, error_size, "cudaMalloc fwht hadamard",
                               status);
        return 0;
    }
    status = cudaMemcpy(cuda->d_hadamard_fwht, host_h,
                        (size_t)gs * (size_t)gs * sizeof(float),
                        cudaMemcpyHostToDevice);
    free(host_h);
    if (status != cudaSuccess) {
        cudaFree(cuda->d_hadamard_fwht);
        cuda->d_hadamard_fwht = NULL;
        h3_cuda_set_error_cuda(error, error_size, "cudaMemcpy fwht hadamard",
                               status);
        return 0;
    }
    cuda->hadamard_fwht_gs = gs;
    return 1;
}

int h3_cuda_fwht_act_f32_dd(h3_cuda *cuda, const float *d_x, size_t m, size_t k,
                            int convrot_groupsize, float *d_y, char *error,
                            size_t error_size) {
    size_t n_groups, g;
    int gs;
    const float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t st;
    if (!cuda || !d_x || !d_y || m == 0 || k == 0 || convrot_groupsize < 2) {
        h3_cuda_set_error(error, error_size, "invalid fwht_act_dd args");
        return 0;
    }
    if (d_x == d_y) {
        h3_cuda_set_error(error, error_size, "fwht_act_dd aliases");
        return 0;
    }
    gs = convrot_groupsize;
    if (k % (size_t)gs != 0) {
        h3_cuda_set_error(error, error_size, "K not divisible by fwht_gs");
        return 0;
    }
    if (!h3_cuda_ensure_hadamard_fwht(cuda, gs, error, error_size)) return 0;
    n_groups = k / (size_t)gs;
    st = cublasSgemmStridedBatched(
        cuda->cublas, CUBLAS_OP_N, CUBLAS_OP_N, gs, (int)m, gs, &alpha,
        cuda->d_hadamard_fwht, gs, 0LL, d_x, (int)k, (long long)gs, &beta, d_y,
        (int)k, (long long)gs, (int)n_groups);
    if (st == CUBLAS_STATUS_SUCCESS) return 1;
    for (g = 0; g < n_groups; g++) {
        st = cublasSgemm(cuda->cublas, CUBLAS_OP_N, CUBLAS_OP_N, gs, (int)m, gs,
                         &alpha, cuda->d_hadamard_fwht, gs,
                         d_x + g * (size_t)gs, (int)k, &beta,
                         d_y + g * (size_t)gs, (int)k);
        if (st != CUBLAS_STATUS_SUCCESS) {
            h3_cuda_set_error(error, error_size,
                              "cublasSgemm fwht_act_dd failed");
            return 0;
        }
    }
    return 1;
}

/* One block per (row, group); blockDim.x == gs (e.g. 256). */
__global__ void h3_cuda_convrot_act_kernel(const float *x, const float *H,
                                           float *y, size_t m, size_t k,
                                           int gs) {
    const size_t row = (size_t)blockIdx.y;
    const size_t group = (size_t)blockIdx.x;
    const int tid = (int)threadIdx.x;
    extern __shared__ float smem[];
    float *x_g = smem; /* gs */

    if (row >= m || tid >= gs) return;
    const size_t n_groups = k / (size_t)gs;
    if (group >= n_groups) return;

    const float *src = x + row * k + group * (size_t)gs;
    float *dst = y + row * k + group * (size_t)gs;
    x_g[tid] = src[tid];
    __syncthreads();

    float acc = 0.0f;
    for (int i = 0; i < gs; i++) {
        acc += x_g[i] * H[i * gs + tid];
    }
    dst[tid] = acc;
}

/* One block per row; reduce absmax then quantize. Vectorized float4 when
 * K%4==0 (DiT hidden/ffn always are). Extra padded rows (row>=m) write 0. */
__global__ void h3_cuda_quantize_act_per_row_kernel(const float *x, int8_t *y,
                                                    float *scales, size_t m,
                                                    size_t k) {
    const size_t row = (size_t)blockIdx.x;
    const int tid = (int)threadIdx.x;
    const int nthreads = (int)blockDim.x;
    int8_t *dst = y + row * k;
    if (row >= m) {
        for (size_t col = (size_t)tid; col < k; col += (size_t)nthreads)
            dst[col] = 0;
        return;
    }
    extern __shared__ float red[];
    const float *src = x + row * k;

    float local = 0.0f;
    if ((k & 3u) == 0) {
        const float4 *src4 = (const float4 *)src;
        size_t n4 = k >> 2;
        for (size_t col = (size_t)tid; col < n4; col += (size_t)nthreads) {
            float4 v = src4[col];
            float a0 = fabsf(v.x), a1 = fabsf(v.y), a2 = fabsf(v.z),
                  a3 = fabsf(v.w);
            float a = a0 > a1 ? a0 : a1;
            a = a > a2 ? a : a2;
            a = a > a3 ? a : a3;
            if (a > local) local = a;
        }
    } else {
        for (size_t col = (size_t)tid; col < k; col += (size_t)nthreads) {
            float a = fabsf(src[col]);
            if (a > local) local = a;
        }
    }
    red[tid] = local;
    __syncthreads();
    for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride && red[tid + stride] > red[tid])
            red[tid] = red[tid + stride];
        __syncthreads();
    }
    float absmax = red[0];
    float scale = (absmax > 0.0f) ? (absmax / 127.0f) : 1.0f;
    if (tid == 0) scales[row] = scale;
    __syncthreads();
    float inv = 1.0f / scale;
    if ((k & 3u) == 0) {
        const float4 *src4 = (const float4 *)src;
        size_t n4 = k >> 2;
        for (size_t col = (size_t)tid; col < n4; col += (size_t)nthreads) {
            float4 v = src4[col];
            float q0 = roundf(v.x * inv);
            float q1 = roundf(v.y * inv);
            float q2 = roundf(v.z * inv);
            float q3 = roundf(v.w * inv);
            if (q0 > 127.0f) q0 = 127.0f;
            if (q0 < -127.0f) q0 = -127.0f;
            if (q1 > 127.0f) q1 = 127.0f;
            if (q1 < -127.0f) q1 = -127.0f;
            if (q2 > 127.0f) q2 = 127.0f;
            if (q2 < -127.0f) q2 = -127.0f;
            if (q3 > 127.0f) q3 = 127.0f;
            if (q3 < -127.0f) q3 = -127.0f;
            size_t base = col << 2;
            dst[base] = (int8_t)q0;
            dst[base + 1] = (int8_t)q1;
            dst[base + 2] = (int8_t)q2;
            dst[base + 3] = (int8_t)q3;
        }
    } else {
        for (size_t col = (size_t)tid; col < k; col += (size_t)nthreads) {
            float q = roundf(src[col] * inv);
            if (q > 127.0f) q = 127.0f;
            if (q < -127.0f) q = -127.0f;
            dst[col] = (int8_t)q;
        }
    }
}

/* Device-pointer GEMM+dequant (x already on device).
 * m_gemm >= m: Tensor-Core INT8 wants M%16==0; extra act rows must be zero
 * so they cannot change the first m output rows. Dequant writes only m. */
static int h3_cuda_gemm_i8_dequant_f32_dev_dx(
    h3_cuda *cuda, const int8_t *d_x, const int8_t *d_w, const float *d_act,
    int per_tensor_scale, const float *d_sc, size_t m, size_t n, size_t k,
    float *d_y, size_t m_gemm, double *gemm_ms, char *backend_out,
    size_t backend_out_size, char *error, size_t error_size) {
    int32_t *d_acc = NULL;
    cudaError_t status;
    cudaEvent_t ev_start = NULL;
    cudaEvent_t ev_stop = NULL;
    float elapsed_ms = 0.0f;
    char gemm_err[256];
    int gemm_ok = 0;
    const char *gemm_backend = "unknown";
    int ok = 0;
    int quiet = 0;

    if (gemm_ms) *gemm_ms = 0.0;
    if (backend_out && backend_out_size) backend_out[0] = '\0';
    if (!cuda || !d_x || !d_w || !d_act || !d_sc || !d_y || m == 0 || n == 0 ||
        k == 0) {
        h3_cuda_set_error(error, error_size, "invalid device gemm args");
        return 0;
    }
    if (m_gemm < m) m_gemm = m;
    quiet = !h3_cuda_want_events(cuda);

    if (!h3_cuda_ensure_gemm_acc(cuda, m_gemm * n * sizeof(int32_t), error,
                                 error_size))
        return 0;
    d_acc = (int32_t *)cuda->d_gemm_acc;

    if (!quiet) {
        status = cudaEventCreate(&ev_start);
        if (status != cudaSuccess) {
            h3_cuda_set_error_cuda(error, error_size, "cudaEventCreate start",
                                   status);
            return 0;
        }
        status = cudaEventCreate(&ev_stop);
        if (status != cudaSuccess) {
            cudaEventDestroy(ev_start);
            h3_cuda_set_error_cuda(error, error_size, "cudaEventCreate stop",
                                   status);
            return 0;
        }
        status = cudaEventRecord(ev_start, 0);
        if (status != cudaSuccess) {
            h3_cuda_set_error_cuda(error, error_size, "cudaEventRecord start",
                                   status);
            goto cleanup;
        }
    }

    gemm_err[0] = '\0';
    if (h3_cuda_lt_gemm_i8_i32(cuda, d_w, d_x, d_acc, m_gemm, n, k, gemm_err,
                               sizeof(gemm_err))) {
        gemm_ok = 1;
        gemm_backend = "cublasLt";
    } else if (h3_cuda_ex_gemm_i8_i32(cuda, d_w, d_x, d_acc, m_gemm, n, k,
                                      gemm_err, sizeof(gemm_err))) {
        gemm_ok = 1;
        gemm_backend = "cublasGemmEx";
    } else {
        h3_cuda_kernel_gemm_i8_i32(cuda, d_x, d_w, d_acc, m_gemm, n, k);
        status = cudaGetLastError();
        if (status != cudaSuccess) {
            h3_cuda_set_error_cuda(error, error_size, "INT8 GEMM kernel",
                                   status);
            goto cleanup;
        }
        gemm_ok = 1;
        gemm_backend = "cuda_kernel";
    }
    if (!gemm_ok) {
        h3_cuda_set_error(error, error_size,
                          gemm_err[0] ? gemm_err : "INT8 GEMM failed");
        goto cleanup;
    }

    {
        unsigned threads = 256;
        dim3 dq_grid((unsigned)((n + (size_t)threads - 1) / (size_t)threads),
                     (unsigned)m, 1);
        h3_cuda_dequant_i32_kernel<<<dq_grid, threads, 0, h3_cuda_op_st(cuda)>>>(
            d_acc, d_act, per_tensor_scale ? 1 : 0, d_sc, d_y, m, n);
        status = cudaGetLastError();
        if (status != cudaSuccess) {
            h3_cuda_set_error_cuda(error, error_size, "dequant kernel", status);
            goto cleanup;
        }
    }

    if (!quiet) {
        status = cudaEventRecord(ev_stop, 0);
        if (status != cudaSuccess) {
            h3_cuda_set_error_cuda(error, error_size, "cudaEventRecord stop",
                                   status);
            goto cleanup;
        }
        status = cudaEventSynchronize(ev_stop);
        if (status != cudaSuccess) {
            h3_cuda_set_error_cuda(error, error_size, "cudaEventSynchronize",
                                   status);
            goto cleanup;
        }
        status = cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop);
        if (status != cudaSuccess) {
            h3_cuda_set_error_cuda(error, error_size, "cudaEventElapsedTime",
                                   status);
            goto cleanup;
        }
        if (gemm_ms) *gemm_ms = (double)elapsed_ms;
    }
    if (backend_out && backend_out_size)
        snprintf(backend_out, backend_out_size, "%s", gemm_backend);
    ok = 1;

cleanup:
    if (ev_start) cudaEventDestroy(ev_start);
    if (ev_stop) cudaEventDestroy(ev_stop);
    return ok;
}

/* Device-resident ConvRot+quant+INT8 GEMM. d_x / d_y already on GPU.
 * ConvRot = per-group x @ H via cuBLAS (TC-friendly) instead of O(gs^2)
 * naive kernel; temporary rot/i8/scales from act scratch. */
int h3_cuda_gemm_i8_dequant_f32_w_act_dev(
    h3_cuda *cuda, const h3_cuda_w4a8_weight *weight, const float *d_x_f32,
    int convrot_groupsize, size_t m, float *d_y_f32, double *prep_ms,
    double *gemm_ms, char *backend_out, size_t backend_out_size, char *error,
    size_t error_size) {
    float *d_rot = NULL;
    int8_t *d_xi8 = NULL;
    float *d_act = NULL;
    cudaError_t status;
    cudaEvent_t ev0 = NULL, ev1 = NULL;
    float prep_elapsed = 0.0f;
    size_t n, k, tmp_bytes, n_groups, m_pad;
    int gs;
    int ok = 0;
    int quiet;
    static int m_pad_logged = 0;

    if (prep_ms) *prep_ms = 0.0;
    if (gemm_ms) *gemm_ms = 0.0;
    if (backend_out && backend_out_size) backend_out[0] = '\0';
    if (!cuda || !weight || !weight->d_w_i8 || !weight->d_s_channel ||
        !d_x_f32 || !d_y_f32 || m == 0 || convrot_groupsize < 4) {
        h3_cuda_set_error(error, error_size, "invalid w_act_dev gemm args");
        return 0;
    }
    if (weight->cuda && weight->cuda != cuda) {
        h3_cuda_set_error(error, error_size, "weight cuda handle mismatch");
        return 0;
    }
    if (!h3_cuda_w4a8_weight_wait_ready(weight, error, error_size)) return 0;
    n = weight->n;
    k = weight->k;
    gs = convrot_groupsize;
    if (k % (size_t)gs != 0) {
        h3_cuda_set_error(error, error_size, "K not divisible by convrot_gs");
        return 0;
    }
    if (!h3_cuda_ensure_hadamard(cuda, gs, error, error_size)) return 0;

    quiet = !h3_cuda_want_events(cuda);
    /* Quiet used to skip SetDevice (U=1 already on the handle's GPU).
     * U>1 rank 1 IMA'd when scratch/W were used on the wrong current
     * device. bind is GetDevice+no-op when already correct (U=1). */
    if (!h3_cuda_bind(cuda, error, error_size)) return 0;

    /* INT8 Tensor Cores want M%16==0. Pad extra act rows with zeros. */
    m_pad = (m + 15u) & ~(size_t)15u;
    if (m_pad < m) m_pad = m;
    if (m_pad > m && !m_pad_logged) {
        fprintf(stderr, "[h3] int8_gemm: pad M %zu → %zu (TC align 16)\n", m,
                m_pad);
        m_pad_logged = 1;
    }
    tmp_bytes = m * k * sizeof(float) + m_pad * k + m_pad * sizeof(float);
    if (!h3_cuda_ensure_act_scratch(cuda, tmp_bytes, error, error_size))
        return 0;
    {
        char *base = (char *)cuda->d_act_scratch;
        d_rot = (float *)base;
        d_xi8 = (int8_t *)(d_rot + m * k);
        d_act = (float *)(d_xi8 + m_pad * k);
    }

    if (!quiet) {
        status = cudaEventCreate(&ev0);
        if (status != cudaSuccess) {
            h3_cuda_set_error_cuda(error, error_size, "cudaEventCreate",
                                   status);
            goto cleanup;
        }
        status = cudaEventCreate(&ev1);
        if (status != cudaSuccess) {
            h3_cuda_set_error_cuda(error, error_size, "cudaEventCreate",
                                   status);
            goto cleanup;
        }
        status = cudaEventRecord(ev0, 0);
        if (status != cudaSuccess) {
            h3_cuda_set_error_cuda(error, error_size, "cudaEventRecord",
                                   status);
            goto cleanup;
        }
    }

    /* Per-group ConvRot via strided-batched Sgemm: one launch for all groups.
     * Y[:, g*gs:(g+1)*gs] = X[:, g*gs:(g+1)*gs] @ H  (H shared, strideA=0). */
    n_groups = k / (size_t)gs;
    {
        const float alpha = 1.0f, beta = 0.0f;
        cublasStatus_t st = cublasSgemmStridedBatched(
            cuda->cublas, CUBLAS_OP_N, CUBLAS_OP_N, gs, (int)m, gs, &alpha,
            cuda->d_hadamard, gs, 0LL, d_x_f32, (int)k, (long long)gs, &beta,
            d_rot, (int)k, (long long)gs, (int)n_groups);
        if (st != CUBLAS_STATUS_SUCCESS) {
            /* Fallback: per-group Sgemm (older drivers / edge sizes). */
            size_t g;
            for (g = 0; g < n_groups; g++) {
                st = cublasSgemm(
                    cuda->cublas, CUBLAS_OP_N, CUBLAS_OP_N, gs, (int)m, gs,
                    &alpha, cuda->d_hadamard, gs, d_x_f32 + g * (size_t)gs,
                    (int)k, &beta, d_rot + g * (size_t)gs, (int)k);
                if (st != CUBLAS_STATUS_SUCCESS) {
                    h3_cuda_set_error(error, error_size,
                                      "cublasSgemm convrot group failed");
                    goto cleanup;
                }
            }
        }
    }
    {
        unsigned threads = 256;
        size_t smem = (size_t)threads * sizeof(float);
        unsigned qrows = (unsigned)(m_pad > m ? m_pad : m);
        h3_cuda_quantize_act_per_row_kernel<<<qrows, threads, smem,
                                              h3_cuda_op_st(cuda)>>>(
            d_rot, d_xi8, d_act, m, k);
        status = cudaGetLastError();
        if (status != cudaSuccess) {
            h3_cuda_set_error_cuda(error, error_size, "quant kernel", status);
            goto cleanup;
        }
    }

    if (ev0 && ev1) {
        status = cudaEventRecord(ev1, 0);
        if (status != cudaSuccess) {
            h3_cuda_set_error_cuda(error, error_size, "cudaEventRecord",
                                   status);
            goto cleanup;
        }
        status = cudaEventSynchronize(ev1);
        if (status != cudaSuccess) {
            h3_cuda_set_error_cuda(error, error_size, "cudaEventSynchronize",
                                   status);
            goto cleanup;
        }
        status = cudaEventElapsedTime(&prep_elapsed, ev0, ev1);
        if (status != cudaSuccess) {
            h3_cuda_set_error_cuda(error, error_size, "cudaEventElapsedTime",
                                   status);
            goto cleanup;
        }
        if (prep_ms) *prep_ms = (double)prep_elapsed;
    }

    if (!h3_cuda_gemm_i8_dequant_f32_dev_dx(
            cuda, d_xi8, weight->d_w_i8, d_act, 0, weight->d_s_channel, m, n, k,
            d_y_f32, m_pad, gemm_ms, backend_out, backend_out_size, error,
            error_size))
        goto cleanup;
    ok = 1;
    if (h3_sol_profile_on()) {
        double g = 0.0, p = 0.0;
        if (gemm_ms) g = *gemm_ms;
        if (prep_ms) p = *prep_ms;
        h3_sp_add_n(&g_sp_step.gemm, &g_sp_tot.gemm, &g_sp_step.n_gemm,
                    &g_sp_tot.n_gemm, p + g);
    }

cleanup:
    if (ev0) cudaEventDestroy(ev0);
    if (ev1) cudaEventDestroy(ev1);
    return ok;
}

int h3_cuda_gemm_i8_dequant_f32_w_act(
    h3_cuda *cuda, const h3_cuda_w4a8_weight *weight, const float *x_f32,
    int convrot_groupsize, size_t m, float *y_f32, double *prep_ms,
    double *gemm_ms, char *backend_out, size_t backend_out_size, char *error,
    size_t error_size) {
    float *d_x = NULL;
    float *d_y = NULL;
    cudaError_t status;
    size_t n, k;
    int ok = 0;

    if (prep_ms) *prep_ms = 0.0;
    if (gemm_ms) *gemm_ms = 0.0;
    if (backend_out && backend_out_size) backend_out[0] = '\0';
    if (!cuda || !weight || !x_f32 || !y_f32 || m == 0) {
        h3_cuda_set_error(error, error_size, "invalid w_act gemm args");
        return 0;
    }
    n = weight->n;
    k = weight->k;
    status = cudaSetDevice(cuda->device_index);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaSetDevice", status);
        return 0;
    }
    if (cudaMalloc((void **)&d_x, m * k * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&d_y, m * n * sizeof(float)) != cudaSuccess) {
        h3_cuda_set_error(error, error_size, "w_act alloc OOM");
        goto cleanup;
    }
    status =
        cudaMemcpy(d_x, x_f32, m * k * sizeof(float), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaMemcpy x H2D", status);
        goto cleanup;
    }
    if (!h3_cuda_gemm_i8_dequant_f32_w_act_dev(
            cuda, weight, d_x, convrot_groupsize, m, d_y, prep_ms, gemm_ms,
            backend_out, backend_out_size, error, error_size))
        goto cleanup;
    status =
        cudaMemcpy(y_f32, d_y, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaMemcpy y D2H", status);
        goto cleanup;
    }
    ok = 1;
cleanup:
    if (d_x) cudaFree(d_x);
    if (d_y) cudaFree(d_y);
    return ok;
}

/* ---- FlashAttention-style tiled SDPA (FP32) ---- */

enum { H3_CUDA_SDPA_BR = 32, H3_CUDA_SDPA_BC = 32, H3_CUDA_SDPA_DMAX = 128 };

/* grid: (heads, ceil(seq/Br)); block: Br threads (one query row each).
 * smem: Q[Br*D] + K[Bc*D] + V[Bc*D] + O[Br*D] */
__global__ void h3_cuda_sdpa_flash_kernel(const float *__restrict__ Q,
                                          const float *__restrict__ K,
                                          const float *__restrict__ V,
                                          float *__restrict__ Out, int seq,
                                          int heads, int D, float scale) {
    const int Br = H3_CUDA_SDPA_BR;
    const int Bc = H3_CUDA_SDPA_BC;
    const int h = (int)blockIdx.x;
    const int q0 = (int)blockIdx.y * Br;
    const int t = (int)threadIdx.x; /* 0 .. Br-1 */
    extern __shared__ float sm[];
    float *Qs = sm;
    float *Ks = Qs + Br * D;
    float *Vs = Ks + Bc * D;
    float *Os = Vs + Bc * D;

    if (t >= Br) return;
    const int qi = q0 + t;
    const int q_valid = (qi < seq);

    /* Load Q row; zero O; init online softmax state. */
    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    if (q_valid) {
        const float *q_row = Q + ((size_t)qi * (size_t)heads + (size_t)h) * (size_t)D;
        for (int d = 0; d < D; d++) {
            Qs[t * D + d] = q_row[d];
            Os[t * D + d] = 0.0f;
        }
    } else {
        for (int d = 0; d < D; d++) {
            Qs[t * D + d] = 0.0f;
            Os[t * D + d] = 0.0f;
        }
    }
    __syncthreads();

    for (int k0 = 0; k0 < seq; k0 += Bc) {
        /* Cooperative load K/V tiles (Br threads cover Bc rows). */
        for (int r = t; r < Bc; r += Br) {
            const int ki = k0 + r;
            if (ki < seq) {
                const float *k_row =
                    K + ((size_t)ki * (size_t)heads + (size_t)h) * (size_t)D;
                const float *v_row =
                    V + ((size_t)ki * (size_t)heads + (size_t)h) * (size_t)D;
                for (int d = 0; d < D; d++) {
                    Ks[r * D + d] = k_row[d];
                    Vs[r * D + d] = v_row[d];
                }
            } else {
                for (int d = 0; d < D; d++) {
                    Ks[r * D + d] = 0.0f;
                    Vs[r * D + d] = 0.0f;
                }
            }
        }
        __syncthreads();

        if (q_valid) {
            float scores[H3_CUDA_SDPA_BC];
            float row_max = -FLT_MAX;
            const int kc = (k0 + Bc <= seq) ? Bc : (seq - k0);
            for (int j = 0; j < kc; j++) {
                float dot = 0.0f;
                for (int d = 0; d < D; d++)
                    dot += Qs[t * D + d] * Ks[j * D + d];
                scores[j] = dot * scale;
                if (scores[j] > row_max) row_max = scores[j];
            }
            float m_new = (row_max > m_i) ? row_max : m_i;
            float alpha = expf(m_i - m_new);
            float l_new = l_i * alpha;
            for (int d = 0; d < D; d++) Os[t * D + d] *= alpha;

            for (int j = 0; j < kc; j++) {
                float p = expf(scores[j] - m_new);
                l_new += p;
                for (int d = 0; d < D; d++) Os[t * D + d] += p * Vs[j * D + d];
            }
            m_i = m_new;
            l_i = l_new;
        }
        __syncthreads();
    }

    if (q_valid) {
        float inv = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
        float *o_row =
            Out + ((size_t)qi * (size_t)heads + (size_t)h) * (size_t)D;
        for (int d = 0; d < D; d++) o_row[d] = Os[t * D + d] * inv;
    }
}

int h3_cuda_sdpa_f32(h3_cuda *cuda, const float *query, const float *key,
                     const float *value, size_t seq, size_t heads,
                     size_t head_dim, float scale, float *out, double *sdpa_ms,
                     char *error, size_t error_size) {
    float *d_q = NULL, *d_k = NULL, *d_v = NULL, *d_o = NULL;
    cudaError_t status;
    cudaEvent_t ev0 = NULL, ev1 = NULL;
    float elapsed = 0.0f;
    size_t elems;
    size_t smem;
    int ok = 0;
    static int smem_attr_set = 0;

    if (sdpa_ms) *sdpa_ms = 0.0;
    if (!cuda || !query || !key || !value || !out || seq == 0 || heads == 0 ||
        head_dim == 0) {
        h3_cuda_set_error(error, error_size, "invalid h3_cuda_sdpa_f32 args");
        return 0;
    }
    if (head_dim > (size_t)H3_CUDA_SDPA_DMAX) {
        h3_cuda_set_error(error, error_size, "head_dim > 128 not supported");
        return 0;
    }

    status = cudaSetDevice(cuda->device_index);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaSetDevice", status);
        return 0;
    }

    elems = seq * heads * head_dim;
    status = cudaMalloc((void **)&d_q, elems * sizeof(float));
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaMalloc Q", status);
        goto cleanup;
    }
    status = cudaMalloc((void **)&d_k, elems * sizeof(float));
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaMalloc K", status);
        goto cleanup;
    }
    status = cudaMalloc((void **)&d_v, elems * sizeof(float));
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaMalloc V", status);
        goto cleanup;
    }
    status = cudaMalloc((void **)&d_o, elems * sizeof(float));
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaMalloc O", status);
        goto cleanup;
    }

    status = cudaMemcpy(d_q, query, elems * sizeof(float),
                        cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaMemcpy Q", status);
        goto cleanup;
    }
    status =
        cudaMemcpy(d_k, key, elems * sizeof(float), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaMemcpy K", status);
        goto cleanup;
    }
    status = cudaMemcpy(d_v, value, elems * sizeof(float),
                        cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaMemcpy V", status);
        goto cleanup;
    }

    smem = (size_t)(2 * (H3_CUDA_SDPA_BR + H3_CUDA_SDPA_BC)) * head_dim *
           sizeof(float);
    if (!smem_attr_set) {
        status = cudaFuncSetAttribute(
            h3_cuda_sdpa_flash_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        if (status != cudaSuccess) {
            h3_cuda_set_error_cuda(error, error_size,
                                   "cudaFuncSetAttribute smem", status);
            goto cleanup;
        }
        smem_attr_set = 1;
    }

    status = cudaEventCreate(&ev0);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaEventCreate", status);
        goto cleanup;
    }
    status = cudaEventCreate(&ev1);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaEventCreate", status);
        goto cleanup;
    }

    {
        unsigned q_tiles =
            (unsigned)((seq + (size_t)H3_CUDA_SDPA_BR - 1) / (size_t)H3_CUDA_SDPA_BR);
        dim3 grid((unsigned)heads, q_tiles, 1);
        status = cudaEventRecord(ev0, 0);
        if (status != cudaSuccess) {
            h3_cuda_set_error_cuda(error, error_size, "cudaEventRecord",
                                   status);
            goto cleanup;
        }
        h3_cuda_sdpa_flash_kernel<<<grid, H3_CUDA_SDPA_BR, smem>>>(
            d_q, d_k, d_v, d_o, (int)seq, (int)heads, (int)head_dim, scale);
        status = cudaGetLastError();
        if (status != cudaSuccess) {
            h3_cuda_set_error_cuda(error, error_size, "sdpa flash kernel",
                                   status);
            goto cleanup;
        }
        status = cudaEventRecord(ev1, 0);
        if (status != cudaSuccess) {
            h3_cuda_set_error_cuda(error, error_size, "cudaEventRecord",
                                   status);
            goto cleanup;
        }
        status = cudaEventSynchronize(ev1);
        if (status != cudaSuccess) {
            h3_cuda_set_error_cuda(error, error_size, "cudaEventSynchronize",
                                   status);
            goto cleanup;
        }
        status = cudaEventElapsedTime(&elapsed, ev0, ev1);
        if (status != cudaSuccess) {
            h3_cuda_set_error_cuda(error, error_size, "cudaEventElapsedTime",
                                   status);
            goto cleanup;
        }
    }
    if (sdpa_ms) *sdpa_ms = (double)elapsed;

    status =
        cudaMemcpy(out, d_o, elems * sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaMemcpy O D2H", status);
        goto cleanup;
    }
    ok = 1;

cleanup:
    if (ev0) cudaEventDestroy(ev0);
    if (ev1) cudaEventDestroy(ev1);
    if (d_q) cudaFree(d_q);
    if (d_k) cudaFree(d_k);
    if (d_v) cudaFree(d_v);
    if (d_o) cudaFree(d_o);
    return ok;
}

/* ---- FP16 Flash / TC / window SDPA ---- */

enum {
    H3_CUDA_SDPA_F16_RQ = 8,
    H3_CUDA_SDPA_F16_BC = 128,
    H3_CUDA_SDPA_F16_D = 128
};

const char *h3_cuda_sdpa_backend_name(void) {
    const char *e = getenv("H3_CUDA_SDPA_BACKEND");
    if (!e || !e[0]) return "auto";
    return e;
}

/* Registry-backed selectors — no sm==120 branching; see h3_sdpa.h. */
static h3_sdpa_backend_id h3_cuda_sdpa_id(void) {
    return h3_sdpa_cached_id();
}

static int h3_cuda_sdpa_want_tc(void) {
    return h3_cuda_sdpa_id() == H3_SDPA_ID_TC;
}

static int h3_cuda_sdpa_want_cudnn(void) {
    h3_sdpa_backend_id id = h3_cuda_sdpa_id();
    /* Tier-0 cudnn, and dense fallback for Tier-1 sage3 / sol_cute. */
    return id == H3_SDPA_ID_CUDNN || id == H3_SDPA_ID_SAGE3 ||
           id == H3_SDPA_ID_SOL_CUTE;
}

static int h3_cuda_sdpa_want_sage2(void) {
    /* In-repo reference — NOT product / NOT Sage3. */
    return h3_cuda_sdpa_id() == H3_SDPA_ID_SAGE2;
}

static int h3_cuda_sdpa_want_sage3(void) {
    return h3_cuda_sdpa_id() == H3_SDPA_ID_SAGE3;
}

static int h3_cuda_sdpa_want_sol(void) {
    /* Nested-loop fake — forbidden as product; not sol_cute. */
    return h3_cuda_sdpa_id() == H3_SDPA_ID_SOL;
}

static int h3_cuda_sdpa_want_sol_cute(void) {
    return h3_cuda_sdpa_id() == H3_SDPA_ID_SOL_CUTE;
}

extern "C" void h3_cuda_sdpa_log_selection(int device_index) {
    h3_cuda_device_caps caps;
    char err[256];
    h3_sdpa_cache_invalidate();
    if (!h3_cuda_query_caps(device_index, &caps, err, sizeof(err))) {
        fprintf(stderr, "[h3] sdpa_select: caps probe failed: %s\n", err);
        h3_sdpa_log_selection(NULL);
        return;
    }
    h3_sdpa_log_selection(&caps);
}

static int h3_sol_sink_tokens_env(void);

static int g_sol_cute_step = -1;
static int g_sol_cute_layer = 0;
static int g_sol_cute_inited = 0;
static int g_sol_cute_failed = 0;
static int g_sol_cute_atexit = 0;
static size_t g_sol_cute_seq = 0, g_sol_cute_heads = 0, g_sol_cute_dim = 0;
static void *g_sol_cute_q = NULL, *g_sol_cute_k = NULL, *g_sol_cute_v = NULL,
            *g_sol_cute_o = NULL;
static void *g_sol_cute_lib = NULL;
static char g_sol_cute_backend[64] = "cute_sm89";
static unsigned long long g_sol_cute_hits = 0;
static unsigned long long g_sol_cute_dense_skip = 0;
static unsigned long long g_sol_cute_fallback = 0;
static unsigned long long g_sol_cute_want = 0;
static unsigned long long g_sol_cute_hits_at_step = 0;
static unsigned long long g_sol_cute_dense_at_step = 0;

typedef int (*h3_sol_cute_init_fn)(const char *, int, int, int, float, int, int,
                                   void **, void **, void **, void **, char *,
                                   size_t, char *, size_t);
typedef int (*h3_sol_cute_fwd_fn)(char *, size_t);
typedef int (*h3_sol_cute_fwd_ex_fn)(char *, size_t, double *, double *,
                                     double *, double *, double *);
static h3_sol_cute_init_fn g_sol_cute_init_fn = NULL;
static h3_sol_cute_fwd_fn g_sol_cute_fwd_fn = NULL;
static h3_sol_cute_fwd_ex_fn g_sol_cute_fwd_ex_fn = NULL;

static double h3_sp_sum_other(const h3_sol_prof *p) {
    return p->adaln + p->rope + p->swiglu + p->gate + p->unpack;
}

static void h3_sp_print_totals(void) {
    double sdpa, layout, other, accounted, gap, theory;
    double cute_avg, cudnn_avg;
    if (!h3_sol_profile_on()) return;
    sdpa = g_sp_tot.sdpa_cute + g_sp_tot.sdpa_cudnn;
    layout = g_sp_tot.layout + g_sp_tot.o_cast;
    other = h3_sp_sum_other(&g_sp_tot);
    accounted = sdpa + g_sp_tot.gemm + layout + g_sp_tot.prep + other;
    gap = g_sp_tot.step_ms - accounted - g_sp_tot.compile;
    cute_avg = (g_sp_tot.n_cute > 0)
                   ? (g_sp_tot.sdpa_cute / (double)g_sp_tot.n_cute)
                   : 0.0;
    cudnn_avg = (g_sp_tot.n_cudnn > 0)
                    ? (g_sp_tot.sdpa_cudnn / (double)g_sp_tot.n_cudnn)
                    : 0.0;
    theory = (g_sp_tot.n_cute > 0 && cudnn_avg > cute_avg)
                 ? (cudnn_avg - cute_avg) * (double)g_sp_tot.n_cute
                 : 0.0;
    fprintf(stderr,
            "[h3] sol_profile TOTAL step_ms=%.1f accounted_ms=%.1f gap_ms=%.1f "
            "(sync/GIL/launch; compile_ms=%.1f n=%llu excluded from gap)\n",
            g_sp_tot.step_ms, accounted, gap, g_sp_tot.compile,
            g_sp_tot.n_compile);
    fprintf(stderr,
            "[h3] sol_profile SDPA cute_ms=%.1f n=%llu avg=%.2f "
            "cudnn_ms=%.1f n=%llu avg=%.2f dense_ms=%.1f n=%llu "
            "theory_save_ms=%.1f (if those cute hits had been cuDNN avg)\n",
            g_sp_tot.sdpa_cute, g_sp_tot.n_cute, cute_avg, g_sp_tot.sdpa_cudnn,
            g_sp_tot.n_cudnn, cudnn_avg, g_sp_tot.sdpa_dense, g_sp_tot.n_dense,
            theory);
    fprintf(stderr,
            "[h3] sol_profile PREP triton_ms=%.1f n=%llu "
            "layout_f16bf16_ms=%.1f o_cast_bf16f32_ms=%.1f permute_ms=0 "
            "h2d_ms=%.1f d2h_ms=%.1f host_py_ms=%.1f "
            "gil_ms=%.1f\n",
            g_sp_tot.prep, g_sp_tot.n_prep, g_sp_tot.layout, g_sp_tot.o_cast,
            g_sp_tot.h2d, g_sp_tot.d2h, g_sp_tot.host_py,
            g_sp_tot.host_py - g_sp_tot.prep - g_sp_tot.sdpa_cute -
                g_sp_tot.o_cast);
    fprintf(stderr,
            "[h3] sol_profile NONATTN gemm_ms=%.1f n=%llu "
            "(INT8 ConvRot+quant+GEMM+dequant QKV/O/FC1/FC2) "
            "adaln_ms=%.1f rope_ms=%.1f swiglu_ms=%.1f gate_ms=%.1f "
            "unpack_ms=%.1f other_ms=%.1f\n",
            g_sp_tot.gemm, g_sp_tot.n_gemm, g_sp_tot.adaln, g_sp_tot.rope,
            g_sp_tot.swiglu, g_sp_tot.gate, g_sp_tot.unpack, other);
    fprintf(stderr,
            "[h3] sol_profile SHARE sdpa=%.1f%% gemm=%.1f%% layout+prep=%.1f%% "
            "other=%.1f%% gap=%.1f%% of step_ms\n",
            (g_sp_tot.step_ms > 0.0) ? (100.0 * sdpa / g_sp_tot.step_ms) : 0.0,
            (g_sp_tot.step_ms > 0.0) ? (100.0 * g_sp_tot.gemm / g_sp_tot.step_ms)
                                     : 0.0,
            (g_sp_tot.step_ms > 0.0)
                ? (100.0 * (layout + g_sp_tot.prep) / g_sp_tot.step_ms)
                : 0.0,
            (g_sp_tot.step_ms > 0.0) ? (100.0 * other / g_sp_tot.step_ms) : 0.0,
            (g_sp_tot.step_ms > 0.0) ? (100.0 * gap / g_sp_tot.step_ms) : 0.0);
    fflush(stderr);
}

extern "C" void h3_cuda_sol_cute_dump_stats(void) {
    fprintf(stderr,
            "[h3] sol_cute stats backend=%s hits=%llu dense_skip=%llu "
            "fallback=%llu want=%llu failed=%d inited=%d "
            "layout=BF16_BTHD D=128 nested_loop=never\n",
            g_sol_cute_backend[0] ? g_sol_cute_backend : "(none)",
            g_sol_cute_hits, g_sol_cute_dense_skip, g_sol_cute_fallback,
            g_sol_cute_want, g_sol_cute_failed, g_sol_cute_inited);
    h3_sp_print_totals();
    fflush(stderr);
}

extern "C" void h3_cuda_sol_profile_finish_step(int step, double step_ms) {
    double sdpa, layout, other, accounted, gap;
    unsigned long long hits, dense;
    if (!h3_sol_profile_on()) return;
    if (step == g_sp_last_finished) return;
    g_sp_last_finished = step;
    g_sp_step.step_ms = step_ms;
    g_sp_tot.step_ms += step_ms;
    sdpa = g_sp_step.sdpa_cute + g_sp_step.sdpa_cudnn;
    layout = g_sp_step.layout + g_sp_step.o_cast;
    other = h3_sp_sum_other(&g_sp_step);
    accounted = sdpa + g_sp_step.gemm + layout + g_sp_step.prep + other;
    gap = step_ms - accounted - g_sp_step.compile;
    hits = g_sol_cute_hits - g_sol_cute_hits_at_step;
    dense = g_sol_cute_dense_skip - g_sol_cute_dense_at_step;
    fprintf(stderr,
            "[h3] sol_profile step= %d sdpa_ms= %.1f gemm_ms= %.1f "
            "layout_ms= %.1f prep_ms= %.1f other_ms= %.1f step_ms= %.1f "
            "hits= %llu dense= %llu gap_ms= %.1f cute_ms= %.1f cudnn_ms= %.1f "
            "dense_ms= %.1f compile_ms= %.1f host_py_ms= %.1f\n",
            step, sdpa, g_sp_step.gemm, layout, g_sp_step.prep, other, step_ms,
            hits, dense, gap, g_sp_step.sdpa_cute, g_sp_step.sdpa_cudnn,
            g_sp_step.sdpa_dense, g_sp_step.compile, g_sp_step.host_py);
    fflush(stderr);
    memset(&g_sp_step, 0, sizeof(g_sp_step));
}

extern "C" void h3_cuda_sol_cute_set_step(int step) {
    int s = step < 0 ? 0 : step;
    if (s != g_sol_cute_step) {
        if (g_sol_cute_step >= 0 && g_sol_cute_want > 0)
            fprintf(stderr,
                    "[h3] sol_cute step_done=%d hits=%llu dense_skip=%llu "
                    "fallback=%llu step_hits=%llu\n",
                    g_sol_cute_step, g_sol_cute_hits, g_sol_cute_dense_skip,
                    g_sol_cute_fallback,
                    g_sol_cute_hits - g_sol_cute_hits_at_step);
        g_sol_cute_hits_at_step = g_sol_cute_hits;
        g_sol_cute_dense_at_step = g_sol_cute_dense_skip;
        g_sol_cute_step = s;
        g_sol_cute_layer = 0;
    }
    if (!g_sol_cute_atexit) {
        g_sol_cute_atexit = 1;
        atexit(h3_cuda_sol_cute_dump_stats);
    }
}

static int h3_sol_cute_dense_steps(void) {
    const char *e = getenv("H3_SOL_FIRST_DENSE_STEPS");
    if (!e || !e[0]) e = getenv("SOL_ATTN_FIRST_DENSE_STEPS");
    /* Official 50-step recipe keeps 10 dense = 20%. 8-step → 2. */
    if (!e || !e[0]) return 2;
    return atoi(e);
}

static int h3_sol_cute_dense_layers(void) {
    const char *e = getenv("H3_SOL_FIRST_DENSE_LAYERS");
    if (!e || !e[0]) e = getenv("SOL_ATTN_FIRST_DENSE_LAYERS");
    if (!e || !e[0]) return 2;
    return atoi(e);
}

static float h3_sol_cute_tau(void) {
    const char *e = getenv("H3_SOL_TAU");
    if (!e || !e[0]) e = getenv("SOL_ATTN_TAU");
    if (!e || !e[0]) return 1.0f;
    return (float)atof(e);
}

__global__ void h3_f16_to_bf16_kernel(const __half *__restrict__ in,
                                      __nv_bfloat16 *__restrict__ out,
                                      size_t n) {
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i < n)
        out[i] = __float2bfloat16(__half2float(in[i]));
}

__global__ void h3_bf16_to_f32_kernel(const __nv_bfloat16 *__restrict__ in,
                                      float *__restrict__ out, size_t n) {
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i < n) out[i] = __bfloat162float(in[i]);
}

static int h3_sol_cute_load_lib(char *error, size_t error_size) {
    const char *path;
    if (g_sol_cute_lib) return 1;
    path = getenv("H3_SOL_CUTE_LIB");
    if (!path || !path[0]) path = "libh3_sol_cute.so";
    g_sol_cute_lib = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
    if (!g_sol_cute_lib) {
        snprintf(error, error_size,
                 "dlopen sol_cute (%s): %s — set H3_SOL_CUTE_LIB to "
                 "libh3_sol_cute.so (Tier-1; no machine-local default path)",
                 path, dlerror());
        return 0;
    }
    g_sol_cute_init_fn =
        (h3_sol_cute_init_fn)dlsym(g_sol_cute_lib, "h3_sol_cute_init");
    g_sol_cute_fwd_fn =
        (h3_sol_cute_fwd_fn)dlsym(g_sol_cute_lib, "h3_sol_cute_fwd");
    g_sol_cute_fwd_ex_fn =
        (h3_sol_cute_fwd_ex_fn)dlsym(g_sol_cute_lib, "h3_sol_cute_fwd_ex");
    if (!g_sol_cute_init_fn || !g_sol_cute_fwd_fn) {
        snprintf(error, error_size, "dlsym sol_cute missing init/fwd");
        return 0;
    }
    return 1;
}

static int h3_cuda_sdpa_sol_cute_dev(h3_cuda *cuda, const __half *d_q,
                                     const __half *d_k, const __half *d_v,
                                     float *d_o, size_t seq, size_t heads,
                                     size_t dim, char *error,
                                     size_t error_size) {
    const char *pkg;
    size_t elems;
    unsigned nblk;
    const int thr = 256;
    int sink;
    int prof = h3_sol_profile_on();
    cudaEvent_t ev0 = NULL, ev1 = NULL;
    float layout_ms = 0.0f, ocast_ms = 0.0f;
    double prep = 0.0, kern = 0.0, copy = 0.0, compile = 0.0, host = 0.0;
    (void)cuda;
    if (g_sol_cute_failed) return 0;
    if (dim != 128) {
        snprintf(error, error_size, "sol_cute requires D=128");
        return 0;
    }
    if (!h3_sol_cute_load_lib(error, error_size)) {
        g_sol_cute_failed = 1;
        return 0;
    }
    elems = seq * heads * dim;
    if (!g_sol_cute_inited || g_sol_cute_seq != seq ||
        g_sol_cute_heads != heads || g_sol_cute_dim != dim) {
        pkg = getenv("H3_SOL_ATTN_ROOT");
        if (!pkg || !pkg[0]) {
            snprintf(error, error_size,
                     "sol_cute requires H3_SOL_ATTN_ROOT (path to "
                     "techniques/sparse_backends); no machine-local default");
            g_sol_cute_failed = 1;
            return 0;
        }
        sink = h3_sol_sink_tokens_env();
        if (!g_sol_cute_init_fn(pkg, (int)seq, (int)heads, (int)dim,
                                h3_sol_cute_tau(), 0, sink, &g_sol_cute_q,
                                &g_sol_cute_k, &g_sol_cute_v, &g_sol_cute_o,
                                g_sol_cute_backend, sizeof(g_sol_cute_backend),
                                error, error_size)) {
            g_sol_cute_failed = 1;
            return 0;
        }
        g_sol_cute_inited = 1;
        g_sol_cute_seq = seq;
        g_sol_cute_heads = heads;
        g_sol_cute_dim = dim;
        fprintf(stderr,
                "[h3] official sol_cute backend=%s seq=%zu H=%zu D=%zu "
                "layout=BF16_BTHD qkv=F16_THD_copy tau=%.2f sink=%d "
                "dense_steps=%d dense_layers=%d "
                "(INT8 ConvRot is QKV-proj GEMM only; attn is BF16 Sol)\n",
                g_sol_cute_backend, seq, heads, dim, h3_sol_cute_tau(), sink,
                h3_sol_cute_dense_steps(), h3_sol_cute_dense_layers());
    }
    nblk = (unsigned)((elems + (size_t)thr - 1) / (size_t)thr);
    if (prof) {
        cudaEventCreate(&ev0);
        cudaEventCreate(&ev1);
        if (ev0) cudaEventRecord(ev0, 0);
    }
    h3_f16_to_bf16_kernel<<<nblk, thr>>>(d_q, (__nv_bfloat16 *)g_sol_cute_q,
                                         elems);
    h3_f16_to_bf16_kernel<<<nblk, thr>>>(d_k, (__nv_bfloat16 *)g_sol_cute_k,
                                         elems);
    h3_f16_to_bf16_kernel<<<nblk, thr>>>(d_v, (__nv_bfloat16 *)g_sol_cute_v,
                                         elems);
    if (prof && ev0 && ev1) {
        cudaEventRecord(ev1, 0);
        cudaEventSynchronize(ev1);
        cudaEventElapsedTime(&layout_ms, ev0, ev1);
        h3_sp_add(&g_sp_step.layout, &g_sp_tot.layout, (double)layout_ms);
    }
    {
        int ok_fwd = 0;
        if (g_sol_cute_fwd_ex_fn)
            ok_fwd = g_sol_cute_fwd_ex_fn(error, error_size, &prep, &kern,
                                          &copy, &compile, &host);
        else
            ok_fwd = g_sol_cute_fwd_fn(error, error_size);
        if (!ok_fwd) {
            g_sol_cute_failed = 1;
            (void)cudaGetLastError();
            if (ev0) cudaEventDestroy(ev0);
            if (ev1) cudaEventDestroy(ev1);
            return 0;
        }
    }
    if (prof) {
        h3_sp_add_n(&g_sp_step.prep, &g_sp_tot.prep, &g_sp_step.n_prep,
                    &g_sp_tot.n_prep, prep);
        h3_sp_add_n(&g_sp_step.sdpa_cute, &g_sp_tot.sdpa_cute, &g_sp_step.n_cute,
                    &g_sp_tot.n_cute, kern);
        h3_sp_add(&g_sp_step.o_cast, &g_sp_tot.o_cast, copy);
        if (compile > 0.0) {
            h3_sp_add_n(&g_sp_step.compile, &g_sp_tot.compile,
                        &g_sp_step.n_compile, &g_sp_tot.n_compile, compile);
        }
        h3_sp_add(&g_sp_step.host_py, &g_sp_tot.host_py, host);
        if (ev0) cudaEventRecord(ev0, 0);
    }
    h3_bf16_to_f32_kernel<<<nblk, thr>>>((__nv_bfloat16 *)g_sol_cute_o, d_o,
                                         elems);
    if (prof && ev0 && ev1) {
        cudaEventRecord(ev1, 0);
        cudaEventSynchronize(ev1);
        cudaEventElapsedTime(&ocast_ms, ev0, ev1);
        h3_sp_add(&g_sp_step.o_cast, &g_sp_tot.o_cast, (double)ocast_ms);
        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
        ev0 = ev1 = NULL;
    }
    if (ev0) cudaEventDestroy(ev0);
    if (ev1) cudaEventDestroy(ev1);
    return 1;
}

static void h3_cuda_sol_cute_maybe_warmup(void) {
    const char *be;
    const char *es;
    const char *pkg;
    int seq, heads, dim, sink;
    char error[768];
    typedef int (*h3_sol_cute_warmup_fn)(const char *, int, int, int, float,
                                         int, int, char *, size_t);
    h3_sol_cute_warmup_fn wfn;

    be = getenv("H3_CUDA_SDPA_BACKEND");
    if (!be || (strcmp(be, "sol_cute") && strcmp(be, "sol-cute") &&
                strcmp(be, "sol-engine") && strcmp(be, "cute_sm89")))
        return;
    es = getenv("H3_SOL_WARMUP_SEQ");
    seq = (es && es[0]) ? atoi(es) : 0;
    if (seq < 4096) return;
    heads = 56;
    dim = 128;
    es = getenv("H3_SOL_WARMUP_HEADS");
    if (es && es[0] && atoi(es) > 0) heads = atoi(es);
    es = getenv("H3_SOL_WARMUP_DIM");
    if (es && es[0] && atoi(es) > 0) dim = atoi(es);
    error[0] = '\0';
    if (!h3_sol_cute_load_lib(error, sizeof(error))) {
        fprintf(stderr, "[h3] sol_cute warmup skip: %s\n", error);
        return;
    }
    wfn = (h3_sol_cute_warmup_fn)dlsym(g_sol_cute_lib, "h3_sol_cute_warmup");
    if (!wfn) {
        fprintf(stderr, "[h3] sol_cute warmup skip: no h3_sol_cute_warmup\n");
        return;
    }
    pkg = getenv("H3_SOL_ATTN_ROOT");
    if (!pkg || !pkg[0]) {
        fprintf(stderr,
                "[h3] sol_cute warmup skip: set H3_SOL_ATTN_ROOT to official "
                "techniques/sparse_backends (no sparse/pytorch_fallback "
                "default)\n");
        return;
    }
    sink = h3_sol_sink_tokens_env();
    fprintf(stderr,
            "[h3] sol_cute warmup compile seq=%d H=%d D=%d tau=%.2f sink=%d "
            "root=%s\n",
            seq, heads, dim, (double)h3_sol_cute_tau(), sink, pkg);
    if (!wfn(pkg, seq, heads, dim, h3_sol_cute_tau(), 0, sink, error,
             sizeof(error))) {
        fprintf(stderr, "[h3] sol_cute warmup failed: %s\n",
                error[0] ? error : "(no msg)");
        (void)cudaGetLastError();
        return;
    }
    g_sol_cute_inited = 0;
    g_sol_cute_q = NULL;
    g_sol_cute_k = NULL;
    g_sol_cute_v = NULL;
    g_sol_cute_o = NULL;
    g_sol_cute_seq = 0;
    fprintf(stderr, "[h3] sol_cute warmup ok (compiled, QKVO released)\n");
}

/* ---- SageAttention3 Blackwell FP4 plugin (dlopen libh3_sage3.so) ---- */
__global__ void h3_cuda_f16_to_f32_kernel(const __half *__restrict__ in,
                                          float *__restrict__ out, size_t n);

static int g_sage3_inited = 0;
static int g_sage3_failed = 0;
static int g_sage3_atexit = 0;
static size_t g_sage3_seq = 0, g_sage3_heads = 0, g_sage3_dim = 0;
static void *g_sage3_q = NULL, *g_sage3_k = NULL, *g_sage3_v = NULL,
            *g_sage3_o = NULL;
static void *g_sage3_lib = NULL;
static char g_sage3_backend[64] = "sage3_blackwell_fp4";
static unsigned long long g_sage3_hits = 0;
static unsigned long long g_sage3_fallback = 0;
static unsigned long long g_sage3_want = 0;

typedef int (*h3_sage3_init_fn)(const char *, int, int, int, float, int, int,
                                void **, void **, void **, void **, char *,
                                size_t, char *, size_t);
typedef int (*h3_sage3_fwd_fn)(char *, size_t);
typedef int (*h3_sage3_fwd_ex_fn)(char *, size_t, double *, double *, double *,
                                  double *, double *);
static h3_sage3_init_fn g_sage3_init_fn = NULL;
static h3_sage3_fwd_fn g_sage3_fwd_fn = NULL;
static h3_sage3_fwd_ex_fn g_sage3_fwd_ex_fn = NULL;

/* THD (T,H,D) → BHTD (1,H,T,D) for sage3; contiguous F16. */
__global__ void h3_f16_thd_to_bhtd_kernel(const __half *__restrict__ in,
                                          __half *__restrict__ out, size_t T,
                                          size_t H, size_t D) {
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t n = T * H * D;
    size_t d, h, t, tmp;
    if (i >= n) return;
    d = i % D;
    tmp = i / D;
    h = tmp % H;
    t = tmp / H;
    out[(h * T + t) * D + d] = in[i];
}

/* BHTD F16 → THD F32 (combine transpose + cast; drops Python permute). */
__global__ void h3_f16_bhtd_to_f32_thd_kernel(const __half *__restrict__ in,
                                              float *__restrict__ out,
                                              size_t T, size_t H, size_t D) {
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t n = T * H * D;
    size_t d, h, t, tmp;
    if (i >= n) return;
    d = i % D;
    tmp = i / D;
    t = tmp % T;
    h = tmp / T;
    out[(t * H + h) * D + d] = __half2float(in[i]);
}

extern "C" void h3_cuda_sage3_dump_stats(void) {
    fprintf(stderr,
            "[h3] sage3 stats backend=%s hits=%llu fallback=%llu want=%llu "
            "failed=%d inited=%d layout=F16_BHTD prep=cuda_thd_bhtd D=128 "
            "nested_loop=never ref_sage2=never\n",
            g_sage3_backend[0] ? g_sage3_backend : "(none)", g_sage3_hits,
            g_sage3_fallback, g_sage3_want, g_sage3_failed, g_sage3_inited);
    fflush(stderr);
}

static int h3_sage3_load_lib(char *error, size_t error_size) {
    const char *path;
    if (g_sage3_lib) return 1;
    path = getenv("H3_SAGE3_LIB");
    if (!path || !path[0]) path = "libh3_sage3.so";
    g_sage3_lib = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
    if (!g_sage3_lib) {
        snprintf(error, error_size,
                 "dlopen sage3 (%s): %s — set H3_SAGE3_LIB to libh3_sage3.so "
                 "(Tier-1; no machine-local default path)",
                 path, dlerror());
        return 0;
    }
    g_sage3_init_fn = (h3_sage3_init_fn)dlsym(g_sage3_lib, "h3_sage3_init");
    g_sage3_fwd_fn = (h3_sage3_fwd_fn)dlsym(g_sage3_lib, "h3_sage3_fwd");
    g_sage3_fwd_ex_fn =
        (h3_sage3_fwd_ex_fn)dlsym(g_sage3_lib, "h3_sage3_fwd_ex");
    if (!g_sage3_init_fn || !g_sage3_fwd_fn) {
        snprintf(error, error_size, "dlsym sage3 missing init/fwd");
        return 0;
    }
    if (!g_sage3_atexit) {
        g_sage3_atexit = 1;
        atexit(h3_cuda_sage3_dump_stats);
    }
    return 1;
}

static int h3_cuda_sdpa_sage3_dev(h3_cuda *cuda, const __half *d_q,
                                  const __half *d_k, const __half *d_v,
                                  float *d_o, size_t seq, size_t heads,
                                  size_t dim, float scale, char *error,
                                  size_t error_size) {
    const char *pkg;
    size_t elems;
    unsigned nblk;
    const int thr = 256;
    double prep = 0.0, kern = 0.0, copy = 0.0, compile = 0.0, host = 0.0;
    (void)cuda;
    if (g_sage3_failed) return 0;
    if (dim != 128) {
        snprintf(error, error_size, "sage3 requires D=128");
        return 0;
    }
    if (!h3_sage3_load_lib(error, error_size)) {
        g_sage3_failed = 1;
        return 0;
    }
    elems = seq * heads * dim;
    if (!g_sage3_inited || g_sage3_seq != seq || g_sage3_heads != heads ||
        g_sage3_dim != dim) {
        pkg = getenv("H3_SAGE3_ROOT");
        if (!pkg || !pkg[0]) {
            snprintf(error, error_size,
                     "sage3 requires H3_SAGE3_ROOT (thu-ml "
                     "sageattention3_blackwell tree); no machine-local default");
            g_sage3_failed = 1;
            return 0;
        }
        if (!g_sage3_init_fn(pkg, (int)seq, (int)heads, (int)dim, scale, 0, 0,
                             &g_sage3_q, &g_sage3_k, &g_sage3_v, &g_sage3_o,
                             g_sage3_backend, sizeof(g_sage3_backend), error,
                             error_size)) {
            g_sage3_failed = 1;
            return 0;
        }
        g_sage3_inited = 1;
        g_sage3_seq = seq;
        g_sage3_heads = heads;
        g_sage3_dim = dim;
        fprintf(stderr,
                "[h3] official sage3 backend=%s seq=%zu H=%zu D=%zu "
                "layout=F16_BHTD prep=cuda_thd_bhtd scale=%.6f "
                "(thu-ml FP4; not in-repo sage2 reference)\n",
                g_sage3_backend, seq, heads, dim, scale);
    }
    /* DiT scratch is F16 THD; plugin buffers are F16 BHTD. */
    nblk = (unsigned)((elems + (size_t)thr - 1) / (size_t)thr);
    {
        cudaEvent_t e0 = NULL, e1 = NULL;
        float prep_cuda_ms = 0.0f, post_cuda_ms = 0.0f;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        if (e0) cudaEventRecord(e0, 0);
        h3_f16_thd_to_bhtd_kernel<<<nblk, thr>>>(d_q, (__half *)g_sage3_q, seq,
                                                 heads, dim);
        h3_f16_thd_to_bhtd_kernel<<<nblk, thr>>>(d_k, (__half *)g_sage3_k, seq,
                                                 heads, dim);
        h3_f16_thd_to_bhtd_kernel<<<nblk, thr>>>(d_v, (__half *)g_sage3_v, seq,
                                                 heads, dim);
        if (e0 && e1) {
            cudaEventRecord(e1, 0);
            cudaEventSynchronize(e1);
            cudaEventElapsedTime(&prep_cuda_ms, e0, e1);
        }
        if (cudaGetLastError() != cudaSuccess) {
            if (e0) cudaEventDestroy(e0);
            if (e1) cudaEventDestroy(e1);
            snprintf(error, error_size, "sage3 THD→BHTD failed");
            g_sage3_failed = 1;
            return 0;
        }
        {
            int ok_fwd = 0;
            if (g_sage3_fwd_ex_fn)
                ok_fwd = g_sage3_fwd_ex_fn(error, error_size, &prep, &kern,
                                           &copy, &compile, &host);
            else
                ok_fwd = g_sage3_fwd_fn(error, error_size);
            if (!ok_fwd) {
                if (e0) cudaEventDestroy(e0);
                if (e1) cudaEventDestroy(e1);
                g_sage3_failed = 1;
                (void)cudaGetLastError();
                return 0;
            }
        }
        if (e0) cudaEventRecord(e0, 0);
        h3_f16_bhtd_to_f32_thd_kernel<<<nblk, thr>>>((__half *)g_sage3_o, d_o,
                                                     seq, heads, dim);
        if (e0 && e1) {
            cudaEventRecord(e1, 0);
            cudaEventSynchronize(e1);
            cudaEventElapsedTime(&post_cuda_ms, e0, e1);
        }
        if (cudaGetLastError() != cudaSuccess) {
            if (e0) cudaEventDestroy(e0);
            if (e1) cudaEventDestroy(e1);
            snprintf(error, error_size, "sage3 BHTD→THD F32 failed");
            g_sage3_failed = 1;
            return 0;
        }
        if (g_sage3_hits + 1 <= 3 || ((g_sage3_hits + 1) % 50ull) == 0) {
            fprintf(stderr,
                    "[h3] sage3 cuda_layout hit#%llu prep_ms=%.2f "
                    "py_kern_ms=%.2f post_ms=%.2f (Python permute removed)\n",
                    g_sage3_hits + 1, (double)prep_cuda_ms, kern,
                    (double)post_cuda_ms);
        }
        if (e0) cudaEventDestroy(e0);
        if (e1) cudaEventDestroy(e1);
    }
    return 1;
}

int h3_cuda_sdpa_prebuild(h3_cuda *cuda, size_t seq, size_t heads,
                          size_t head_dim, float scale, int causal,
                          char *error, size_t error_size) {
#if defined(H3_HAVE_CUDNN_FRONTEND)
    size_t padded;
    int want_pad;
    int mask;
    size_t tensor_seq;
    if (!cuda || !cuda->cudnn_ok || seq == 0 || heads == 0 || head_dim == 0) {
        h3_cuda_set_error(error, error_size, "sdpa prebuild unavailable");
        return 0;
    }
    if (!h3_cuda_sdpa_want_cudnn()) return 1;
    padded = h3_cuda_sdpa_cudnn_align_seq(seq);
    want_pad = h3_cuda_sdpa_cudnn_want_padded(seq, heads, head_dim, causal);
    tensor_seq = (want_pad && padded > seq) ? padded : seq;
    /* DiT bidirectional pad uses a padding mask; causal TE pad does not. */
    mask = (!causal && want_pad && padded > seq) ? 1 : 0;
    if (!h3_cuda_sdpa_cudnn_prebuild((void *)cuda->cudnn, tensor_seq, heads,
                                     head_dim, scale, causal, mask, error,
                                     error_size))
        return 0;
    h3_cuda_sdpa_cudnn_note_variant(
        seq, heads, head_dim, causal,
        mask ? H3_CUDNN_VAR_PAD_MASK
             : (want_pad ? H3_CUDNN_VAR_PAD_CAUSAL : H3_CUDNN_VAR_RAW));
    return 1;
#else
    (void)cuda;
    (void)seq;
    (void)heads;
    (void)head_dim;
    (void)scale;
    (void)causal;
    (void)error;
    (void)error_size;
    return 1;
#endif
}

void h3_cuda_set_graph_capture(h3_cuda *cuda, int enabled) {
    if (cuda) {
        cuda->graph_capture = enabled ? 1 : 0;
        /* Capture implies quiet; clearing capture also clears quiet. */
        cuda->quiet = enabled ? 1 : 0;
    }
}

int h3_cuda_graph_capture(const h3_cuda *cuda) {
    return cuda ? cuda->graph_capture : 0;
}

void h3_cuda_set_quiet(h3_cuda *cuda, int enabled) {
    if (cuda) cuda->quiet = enabled ? 1 : 0;
}

int h3_cuda_set_op_stream(h3_cuda *cuda, void *stream, char *error,
                          size_t error_size) {
    cudaStream_t st = (cudaStream_t)stream;
    if (!cuda) {
        h3_cuda_set_error(error, error_size, "cuda handle is NULL");
        return 0;
    }
    if (!h3_cuda_bind(cuda, error, error_size))
        return 0;
    cuda->op_stream = st;
    if (cublasSetStream(cuda->cublas, st) != CUBLAS_STATUS_SUCCESS) {
        h3_cuda_set_error(error, error_size, "cublasSetStream failed");
        return 0;
    }
#if defined(H3_HAVE_CUDNN)
    if (cuda->cudnn_ok && cudnnSetStream(cuda->cudnn, st) != CUDNN_STATUS_SUCCESS) {
        h3_cuda_set_error(error, error_size, "cudnnSetStream failed");
        return 0;
    }
#endif
    return 1;
}

void *h3_cuda_op_stream(const h3_cuda *cuda) {
    return cuda ? (void *)cuda->op_stream : NULL;
}

int h3_cuda_quiet(const h3_cuda *cuda) {
    return cuda ? (cuda->quiet || cuda->graph_capture) : 0;
}

void *h3_cuda_stream(h3_cuda *cuda, int index) {
    if (!cuda || !cuda->streams_ready || index < 0 ||
        index >= H3_CUDA_N_STREAMS)
        return NULL;
    return (void *)cuda->streams[index];
}

int h3_cuda_stream_count(void) { return H3_CUDA_N_STREAMS; }

int h3_cuda_event_record_default(h3_cuda *cuda, void **out_event, char *error,
                                 size_t error_size) {
    cudaEvent_t ev = NULL;
    cudaError_t st;
    if (!cuda || !out_event) {
        h3_cuda_set_error(error, error_size, "invalid event_record args");
        return 0;
    }
    *out_event = NULL;
    st = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    if (st != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaEventCreate", st);
        return 0;
    }
    st = cudaEventRecord(ev, 0);
    if (st != cudaSuccess) {
        cudaEventDestroy(ev);
        h3_cuda_set_error_cuda(error, error_size, "cudaEventRecord", st);
        return 0;
    }
    *out_event = (void *)ev;
    return 1;
}

int h3_cuda_stream_wait_event(void *stream, void *event, char *error,
                              size_t error_size) {
    cudaError_t st;
    if (!stream || !event) {
        h3_cuda_set_error(error, error_size, "invalid stream_wait_event");
        return 0;
    }
    st = cudaStreamWaitEvent((cudaStream_t)stream, (cudaEvent_t)event, 0);
    if (st != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaStreamWaitEvent", st);
        return 0;
    }
    return 1;
}

void h3_cuda_event_destroy(void *event) {
    if (event) cudaEventDestroy((cudaEvent_t)event);
}

__global__ void h3_cuda_f32_to_f16_kernel(const float *__restrict__ in,
                                          __half *__restrict__ out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

/* One launch for Q/K/V cast (same n elems each). */
__global__ void h3_cuda_f32_to_f16_3_kernel(const float *__restrict__ a,
                                            const float *__restrict__ b,
                                            const float *__restrict__ c,
                                            __half *__restrict__ oa,
                                            __half *__restrict__ ob,
                                            __half *__restrict__ oc, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        oa[i] = __float2half(a[i]);
        ob[i] = __float2half(b[i]);
        oc[i] = __float2half(c[i]);
    }
}

__global__ void h3_cuda_f16_to_f32_kernel(const __half *__restrict__ in,
                                          float *__restrict__ out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(in[i]);
}

extern "C" void h3_cuda_launch_f16_to_f32(const void *in_half, float *out,
                                          size_t n) {
    if (!in_half || !out || n == 0) return;
    const int thr = 256;
    const unsigned nblk = (unsigned)((n + (size_t)thr - 1) / (size_t)thr);
    h3_cuda_f16_to_f32_kernel<<<nblk, thr>>>((const __half *)in_half, out, n);
}

/* YCNodes-style Gaussian RF on video-video scores. Prefix (text/audio/cond)
 * rows and columns stay 0 so conditioning is not sparsified. */
struct h3_dist_attn_c {
    int on;
    int v0;
    int vn;
    int T;
    int H;
    int W;
    float scale;
    float tw;
};

__constant__ h3_dist_attn_c g_dist_c;

static h3_dist_attn_c g_dist_host;
static __half *g_d_dist_bias = NULL;
static size_t g_dist_bias_seq = 0;
static int g_dist_logged;
static int g_dist_cudnn_fail;

__device__ __forceinline__ float h3_dist_score_bias(int qi, int ki) {
    if (!g_dist_c.on || g_dist_c.scale <= 0.0f) return 0.0f;
    if (qi < g_dist_c.v0 || ki < g_dist_c.v0) return 0.0f;
    int q = qi - g_dist_c.v0;
    int k = ki - g_dist_c.v0;
    if (q >= g_dist_c.vn || k >= g_dist_c.vn) return 0.0f;
    int hw = g_dist_c.H * g_dist_c.W;
    if (hw <= 0 || g_dist_c.W <= 0) return 0.0f;
    int qt = q / hw;
    int qr = q - qt * hw;
    int qy = qr / g_dist_c.W;
    int qx = qr - qy * g_dist_c.W;
    int kt = k / hw;
    int kr = k - kt * hw;
    int ky = kr / g_dist_c.W;
    int kx = kr - ky * g_dist_c.W;
    float dt = (float)(qt - kt) * g_dist_c.tw;
    float dy = (float)(qy - ky);
    float dx = (float)(qx - kx);
    return -sqrtf(dt * dt + dy * dy + dx * dx) / g_dist_c.scale;
}

__global__ void h3_dist_bias_fill_kernel(__half *bias, int seq, int v0, int vn,
                                         int T, int H, int W, float scale,
                                         float tw) {
    int i = (int)blockIdx.y * (int)blockDim.y + (int)threadIdx.y;
    int j = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (i >= seq || j >= seq) return;
    float v = 0.0f;
    if (scale > 0.0f && i >= v0 && j >= v0) {
        int q = i - v0;
        int k = j - v0;
        int hw = H * W;
        if (q < vn && k < vn && hw > 0 && W > 0) {
            int qt = q / hw, qr = q - qt * hw, qy = qr / W, qx = qr - qy * W;
            int kt = k / hw, kr = k - kt * hw, ky = kr / W, kx = kr - ky * W;
            (void)T;
            float dt = (float)(qt - kt) * tw;
            float dy = (float)(qy - ky);
            float dx = (float)(qx - kx);
            v = -sqrtf(dt * dt + dy * dy + dx * dx) / scale;
        }
    }
    bias[(size_t)i * (size_t)seq + (size_t)j] = __float2half(v);
}

void h3_cuda_dist_attn_clear(void) {
#if defined(H3_HAVE_CUDNN_FRONTEND)
    h3_cuda_sdpa_cudnn_bind_bias(NULL, 0);
#endif
    if (g_d_dist_bias) {
        cudaFree(g_d_dist_bias);
        g_d_dist_bias = NULL;
    }
    g_dist_bias_seq = 0;
    memset(&g_dist_host, 0, sizeof(g_dist_host));
    g_dist_logged = 0;
    g_dist_cudnn_fail = 0;
    cudaMemcpyToSymbol(g_dist_c, &g_dist_host, sizeof(g_dist_host));
}

int h3_cuda_dist_attn_active(void) { return g_dist_host.on != 0; }

int h3_cuda_dist_attn_configure(h3_cuda *cuda, size_t seq, size_t video_start,
                                size_t video_seq, int grid_t, int grid_h,
                                int grid_w, float scale, float temporal_weight,
                                int enabled, char *error, size_t error_size) {
    h3_dist_attn_c cfg;
    size_t padded = seq;
    memset(&cfg, 0, sizeof(cfg));
    if (!enabled || video_seq == 0 || grid_t < 1 || grid_h < 1 || grid_w < 1 ||
        scale <= 0.0f) {
        cfg.on = 0;
        g_dist_host = cfg;
        cudaMemcpyToSymbol(g_dist_c, &cfg, sizeof(cfg));
#if defined(H3_HAVE_CUDNN_FRONTEND)
        h3_cuda_sdpa_cudnn_bind_bias(NULL, 0);
#endif
        return 1;
    }
    if (video_start + video_seq > seq) {
        h3_cuda_set_error(error, error_size, "dist-attn video span exceeds seq");
        return 0;
    }
    if ((size_t)grid_t * (size_t)grid_h * (size_t)grid_w != video_seq) {
        h3_cuda_set_error(error, error_size, "dist-attn grid != video_seq");
        return 0;
    }
    if (cuda && !h3_cuda_bind(cuda, error, error_size)) return 0;
    cfg.on = 1;
    cfg.v0 = (int)video_start;
    cfg.vn = (int)video_seq;
    cfg.T = grid_t;
    cfg.H = grid_h;
    cfg.W = grid_w;
    cfg.scale = scale;
    cfg.tw = temporal_weight > 0.0f ? temporal_weight : 1.0f;
#if defined(H3_HAVE_CUDNN_FRONTEND)
    padded = h3_cuda_sdpa_cudnn_align_seq(seq);
#endif
    {
        int need_alloc = (!g_d_dist_bias || g_dist_bias_seq != padded);
        int need_fill =
            need_alloc || g_dist_host.v0 != cfg.v0 || g_dist_host.vn != cfg.vn ||
            g_dist_host.T != cfg.T || g_dist_host.H != cfg.H ||
            g_dist_host.W != cfg.W || g_dist_host.scale != cfg.scale ||
            g_dist_host.tw != cfg.tw;
        if (need_alloc) {
            if (g_d_dist_bias) {
                cudaFree(g_d_dist_bias);
                g_d_dist_bias = NULL;
                g_dist_bias_seq = 0;
            }
            if (cudaMalloc((void **)&g_d_dist_bias,
                           padded * padded * sizeof(__half)) != cudaSuccess) {
                fprintf(stderr,
                        "[h3] dist-attn bias OOM (pad=%zu); flash on-the-fly only\n",
                        padded);
                g_d_dist_bias = NULL;
                g_dist_bias_seq = 0;
                g_dist_cudnn_fail = 1;
                need_fill = 0;
            } else {
                g_dist_bias_seq = padded;
                g_dist_cudnn_fail = 0;
            }
        }
        if (need_fill && g_d_dist_bias) {
            dim3 block(16, 16);
            dim3 grid((unsigned)((padded + 15) / 16),
                      (unsigned)((padded + 15) / 16));
            h3_dist_bias_fill_kernel<<<grid, block>>>(
                g_d_dist_bias, (int)padded, cfg.v0, cfg.vn, cfg.T, cfg.H, cfg.W,
                cfg.scale, cfg.tw);
            if (cudaGetLastError() != cudaSuccess) {
                h3_cuda_set_error(error, error_size, "dist-attn fill kernel");
                return 0;
            }
        }
    }
    g_dist_host = cfg;
    cudaMemcpyToSymbol(g_dist_c, &cfg, sizeof(cfg));
#if defined(H3_HAVE_CUDNN_FRONTEND)
    if (g_d_dist_bias)
        h3_cuda_sdpa_cudnn_bind_bias(g_d_dist_bias, padded);
    else
        h3_cuda_sdpa_cudnn_bind_bias(NULL, 0);
#endif
    if (!g_dist_logged) {
        fprintf(stderr,
                "[h3] dist-attn video_off=%d video_seq=%d grid=%dx%dx%d "
                "scale=%.3g tw=%.3g seq=%zu pad=%zu bias_mib=%.1f "
                "(video-video only; prefix unpenalized)\n",
                cfg.v0, cfg.vn, cfg.T, cfg.H, cfg.W, (double)cfg.scale,
                (double)cfg.tw, seq, padded,
                g_d_dist_bias
                    ? (double)(padded * padded * sizeof(__half)) /
                          (1024.0 * 1024.0)
                    : 0.0);
        fflush(stderr);
        g_dist_logged = 1;
    }
    return 1;
}

/* Single-pass online softmax; scores tile in smem. attn_window==0 → global. */
__global__ void h3_cuda_sdpa_flash_f16_d128_kernel(
    const __half *__restrict__ Q, const __half *__restrict__ K,
    const __half *__restrict__ V, float *__restrict__ Out, int seq, int heads,
    float scale, int attn_window) {
    const int D = H3_CUDA_SDPA_F16_D;
    const int Bc = H3_CUDA_SDPA_F16_BC;
    const int RQ = H3_CUDA_SDPA_F16_RQ;
    const int h = (int)blockIdx.x;
    const int q_base = (int)blockIdx.y * RQ;
    const int lane = (int)threadIdx.x;
    const int warp = (int)threadIdx.y;
    const int qi = q_base + warp;
    const int q_valid = (qi < seq);

    extern __shared__ char sm_raw[];
    __half *Ks = (__half *)sm_raw;
    __half *Vs = Ks + (size_t)Bc * (size_t)D;
    float *Ss = (float *)(Vs + (size_t)Bc * (size_t)D); /* [RQ][Bc] */

    const int d0 = lane * 4;
    float q_reg[4];
    float o_reg[4];
    float m_i = -FLT_MAX;
    float l_i = 0.0f;
#pragma unroll
    for (int u = 0; u < 4; u++) {
        q_reg[u] = 0.0f;
        o_reg[u] = 0.0f;
    }

    if (q_valid) {
        const __half *q_row =
            Q + ((size_t)qi * (size_t)heads + (size_t)h) * (size_t)D;
        float2 q01 = __half22float2(
            *reinterpret_cast<const __half2 *>(q_row + d0));
        float2 q23 = __half22float2(
            *reinterpret_cast<const __half2 *>(q_row + d0 + 2));
        q_reg[0] = q01.x;
        q_reg[1] = q01.y;
        q_reg[2] = q23.x;
        q_reg[3] = q23.y;
    }
    __syncthreads();

    /* CTA-uniform KV range so __syncthreads stays aligned across warps. */
    int cta_k_lo = 0, cta_k_hi = seq;
    if (attn_window > 0) {
        cta_k_lo = q_base - attn_window;
        cta_k_hi = q_base + RQ - 1 + attn_window + 1;
        if (cta_k_lo < 0) cta_k_lo = 0;
        if (cta_k_hi > seq) cta_k_hi = seq;
    }
    const int k_lo =
        (attn_window > 0 && q_valid) ? (qi - attn_window) : 0;
    const int k_hi =
        (attn_window > 0 && q_valid) ? (qi + attn_window + 1) : seq;
    int k0_start = 0;
    if (attn_window > 0) k0_start = (cta_k_lo / Bc) * Bc;

    for (int k0 = k0_start; k0 < seq; k0 += Bc) {
        if (attn_window > 0) {
            if (k0 + Bc <= cta_k_lo) continue;
            if (k0 >= cta_k_hi) break;
        }
        const int tid = warp * 32 + lane;
        const int nthreads = RQ * 32;
        for (int idx2 = tid; idx2 < (Bc * D) / 2; idx2 += nthreads) {
            const int idx = idx2 * 2;
            const int r = idx / D;
            const int d = idx % D;
            const int ki = k0 + r;
            __half2 *ks2 = reinterpret_cast<__half2 *>(Ks + idx);
            __half2 *vs2 = reinterpret_cast<__half2 *>(Vs + idx);
            if (ki < seq) {
                const size_t base =
                    ((size_t)ki * (size_t)heads + (size_t)h) * (size_t)D +
                    (size_t)d;
                *ks2 = *reinterpret_cast<const __half2 *>(K + base);
                *vs2 = *reinterpret_cast<const __half2 *>(V + base);
            } else {
                *ks2 = __float2half2_rn(0.0f);
                *vs2 = __float2half2_rn(0.0f);
            }
        }
        __syncthreads();

        if (q_valid) {
            const int kc = (k0 + Bc <= seq) ? Bc : (seq - k0);
            float row_max = -FLT_MAX;
            for (int j = 0; j < kc; j++) {
                const int ki = k0 + j;
                int in_win = 1;
                if (attn_window > 0) {
                    in_win = (ki >= k_lo && ki < k_hi);
                }
                float s = -FLT_MAX;
                if (in_win) {
                    float2 k01 = __half22float2(
                        *reinterpret_cast<const __half2 *>(&Ks[j * D + d0]));
                    float2 k23 = __half22float2(
                        *reinterpret_cast<const __half2 *>(&Ks[j * D + d0 + 2]));
                    float partial = q_reg[0] * k01.x + q_reg[1] * k01.y +
                                    q_reg[2] * k23.x + q_reg[3] * k23.y;
#pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1)
                        partial +=
                            __shfl_down_sync(0xffffffffu, partial, offset);
                    s = __shfl_sync(0xffffffffu, partial, 0) * scale;
                    s += h3_dist_score_bias(qi, ki);
                }
                if (lane == 0) Ss[warp * Bc + j] = s;
                if (s > row_max) row_max = s;
            }
            __syncwarp();
            float m_new = (row_max > m_i) ? row_max : m_i;
            float alpha = __expf(m_i - m_new);
            float l_new = l_i * alpha;
#pragma unroll
            for (int u = 0; u < 4; u++) o_reg[u] *= alpha;

            for (int j = 0; j < kc; j++) {
                float s = Ss[warp * Bc + j];
                if (!(s > -FLT_MAX / 4)) continue;
                float p = __expf(s - m_new);
                l_new += p;
                float2 v01 = __half22float2(
                    *reinterpret_cast<const __half2 *>(&Vs[j * D + d0]));
                float2 v23 = __half22float2(
                    *reinterpret_cast<const __half2 *>(&Vs[j * D + d0 + 2]));
                o_reg[0] += p * v01.x;
                o_reg[1] += p * v01.y;
                o_reg[2] += p * v23.x;
                o_reg[3] += p * v23.y;
            }
            m_i = m_new;
            l_i = l_new;
        }
        __syncthreads();
    }

    if (q_valid) {
        float inv = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
        float *o_row =
            Out + ((size_t)qi * (size_t)heads + (size_t)h) * (size_t)D;
#pragma unroll
        for (int u = 0; u < 4; u++) o_row[d0 + u] = o_reg[u] * inv;
    }
}

/* WMMA TC flash: CTA computes Br=64 queries for one head; K/V tiles Bc=64.
 * Uses tensor cores for QK^T and PV (FA2-class tiling). */
enum { H3_CUDA_SDPA_TC_BR = 64, H3_CUDA_SDPA_TC_BC = 64 };

__global__ void __launch_bounds__(256, 2) h3_cuda_sdpa_flash_f16_tc_d128_kernel(
    const __half *__restrict__ Q, const __half *__restrict__ K,
    const __half *__restrict__ V, float *__restrict__ Out, int seq, int heads,
    float scale, int attn_window) {
    using namespace nvcuda::wmma;
    const int D = 128;
    const int Br = H3_CUDA_SDPA_TC_BR;
    const int Bc = H3_CUDA_SDPA_TC_BC;
    const int h = (int)blockIdx.x;
    const int q0 = (int)blockIdx.y * Br;
    const int tid = (int)threadIdx.x;

    extern __shared__ char sm_tc[];
    __half *Qs = (__half *)sm_tc;
    __half *Ks = Qs + Br * D;
    __half *Vs = Ks + Bc * D;
    float *Os = (float *)(Vs + Bc * D); /* Br * D */
    float *Ms = Os + Br * D;            /* Br */
    float *Ls = Ms + Br;                /* Br */
    float *Ss = Ls + Br;                /* Br * Bc */

    for (int i = tid; i < Br; i += blockDim.x) {
        Ms[i] = -FLT_MAX;
        Ls[i] = 0.0f;
        const int qi = q0 + i;
        if (qi < seq) {
            const __half *q_row =
                Q + ((size_t)qi * (size_t)heads + (size_t)h) * (size_t)D;
            for (int d = 0; d < D; d++) Qs[i * D + d] = q_row[d];
            for (int d = 0; d < D; d++) Os[i * D + d] = 0.0f;
        } else {
            for (int d = 0; d < D; d++) {
                Qs[i * D + d] = __float2half(0.0f);
                Os[i * D + d] = 0.0f;
            }
        }
    }
    __syncthreads();

    for (int k0 = 0; k0 < seq; k0 += Bc) {
        /* Window skip for whole tile when all queries share a coarse bound. */
        if (attn_window > 0) {
            int any = 0;
            for (int i = 0; i < Br; i++) {
                int qi = q0 + i;
                if (qi >= seq) continue;
                int lo = qi - attn_window;
                int hi = qi + attn_window + 1;
                if (lo < 0) lo = 0;
                if (hi > seq) hi = seq;
                if (!(k0 + Bc <= lo || k0 >= hi)) {
                    any = 1;
                    break;
                }
            }
            if (!any) continue;
        }

        for (int i = tid; i < Bc * D; i += blockDim.x) {
            int r = i / D;
            int d = i % D;
            int ki = k0 + r;
            if (ki < seq) {
                size_t base =
                    ((size_t)ki * (size_t)heads + (size_t)h) * (size_t)D +
                    (size_t)d;
                Ks[i] = K[base];
                Vs[i] = V[base];
            } else {
                Ks[i] = __float2half(0.0f);
                Vs[i] = __float2half(0.0f);
            }
        }
        __syncthreads();

        /* QK via WMMA 16x16x16 — one warp per output tile. */
        {
            const int warp_id = tid / 32;
            const int nwarps = blockDim.x / 32;
            const int tiles_m = Br / 16;
            const int tiles_n = Bc / 16;
            const int ntiles = tiles_m * tiles_n;
            for (int tile = warp_id; tile < ntiles; tile += nwarps) {
                const int tm = tile / tiles_n;
                const int tn = tile - tm * tiles_n;
                const int qm = tm * 16;
                const int kn = tn * 16;
                fragment<matrix_a, 16, 16, 16, __half, row_major> a_frag;
                fragment<matrix_b, 16, 16, 16, __half, col_major> b_frag;
                fragment<accumulator, 16, 16, 16, float> c_frag;
                fill_fragment(c_frag, 0.0f);
                for (int kk = 0; kk < D; kk += 16) {
                    load_matrix_sync(a_frag, Qs + qm * D + kk, D);
                    load_matrix_sync(b_frag, Ks + kn * D + kk, D);
                    mma_sync(c_frag, a_frag, b_frag, c_frag);
                }
                store_matrix_sync(Ss + qm * Bc + kn, c_frag, Bc,
                                  mem_row_major);
            }
        }
        __syncthreads();

        /* Scale, mask, online softmax + PV accumulate (thread per query). */
        for (int i = tid; i < Br; i += blockDim.x) {
            int qi = q0 + i;
            if (qi >= seq) continue;
            int lo = 0, hi = seq;
            if (attn_window > 0) {
                lo = qi - attn_window;
                hi = qi + attn_window + 1;
                if (lo < 0) lo = 0;
                if (hi > seq) hi = seq;
            }
            float row_max = -FLT_MAX;
            const int kc = (k0 + Bc <= seq) ? Bc : (seq - k0);
            for (int j = 0; j < kc; j++) {
                int ki = k0 + j;
                float s = Ss[i * Bc + j] * scale;
                if (ki < lo || ki >= hi) s = -FLT_MAX;
                else s += h3_dist_score_bias(qi, ki);
                Ss[i * Bc + j] = s;
                if (s > row_max) row_max = s;
            }
            float m_i = Ms[i];
            float m_new = (row_max > m_i) ? row_max : m_i;
            float alpha = __expf(m_i - m_new);
            float l_new = Ls[i] * alpha;
            for (int d = 0; d < D; d++) Os[i * D + d] *= alpha;
            for (int j = 0; j < kc; j++) {
                float s = Ss[i * Bc + j];
                if (!(s > -FLT_MAX / 4)) continue;
                float p = __expf(s - m_new);
                l_new += p;
                for (int d = 0; d < D; d++)
                    Os[i * D + d] += p * __half2float(Vs[j * D + d]);
            }
            Ms[i] = m_new;
            Ls[i] = l_new;
        }
        __syncthreads();
    }

    for (int i = tid; i < Br; i += blockDim.x) {
        int qi = q0 + i;
        if (qi >= seq) continue;
        float inv = (Ls[i] > 0.0f) ? (1.0f / Ls[i]) : 0.0f;
        float *o_row =
            Out + ((size_t)qi * (size_t)heads + (size_t)h) * (size_t)D;
        for (int d = 0; d < D; d++) o_row[d] = Os[i * D + d] * inv;
    }
}

enum { H3_CUDA_SDPA_F16_BR = 64, H3_CUDA_SDPA_F16_BC_GEN = 64 };

__global__ void h3_cuda_sdpa_flash_f16_generic_kernel(
    const __half *__restrict__ Q, const __half *__restrict__ K,
    const __half *__restrict__ V, float *__restrict__ Out, int seq, int heads,
    int D, float scale, int attn_window) {
    const int Br = H3_CUDA_SDPA_F16_BR;
    const int Bc = H3_CUDA_SDPA_F16_BC_GEN;
    const int h = (int)blockIdx.x;
    const int q0 = (int)blockIdx.y * Br;
    const int t = (int)threadIdx.x;
    extern __shared__ __half smg[];
    __half *Qs = smg;
    __half *Ks = Qs + Br * D;
    __half *Vs = Ks + Bc * D;
    float *Os = (float *)(Vs + Bc * D);

    if (t >= Br) return;
    const int qi = q0 + t;
    const int q_valid = (qi < seq);
    float m_i = -FLT_MAX;
    float l_i = 0.0f;

    if (q_valid) {
        const __half *q_row =
            Q + ((size_t)qi * (size_t)heads + (size_t)h) * (size_t)D;
        for (int d = 0; d < D; d++) {
            Qs[t * D + d] = q_row[d];
            Os[t * D + d] = 0.0f;
        }
    } else {
        for (int d = 0; d < D; d++) {
            Qs[t * D + d] = __float2half(0.0f);
            Os[t * D + d] = 0.0f;
        }
    }
    __syncthreads();

    int k_lo = 0, k_hi = seq;
    if (attn_window > 0 && q_valid) {
        k_lo = qi - attn_window;
        k_hi = qi + attn_window + 1;
        if (k_lo < 0) k_lo = 0;
        if (k_hi > seq) k_hi = seq;
    }

    for (int k0 = 0; k0 < seq; k0 += Bc) {
        if (attn_window > 0 && q_valid && (k0 + Bc <= k_lo || k0 >= k_hi)) {
            if (k0 >= k_hi) break;
            continue;
        }
        for (int r = t; r < Bc; r += Br) {
            const int ki = k0 + r;
            if (ki < seq) {
                const __half *k_row =
                    K + ((size_t)ki * (size_t)heads + (size_t)h) * (size_t)D;
                const __half *v_row =
                    V + ((size_t)ki * (size_t)heads + (size_t)h) * (size_t)D;
                for (int d = 0; d < D; d++) {
                    Ks[r * D + d] = k_row[d];
                    Vs[r * D + d] = v_row[d];
                }
            } else {
                for (int d = 0; d < D; d++) {
                    Ks[r * D + d] = __float2half(0.0f);
                    Vs[r * D + d] = __float2half(0.0f);
                }
            }
        }
        __syncthreads();

        if (q_valid) {
            float scores[H3_CUDA_SDPA_F16_BC_GEN];
            float row_max = -FLT_MAX;
            const int kc = (k0 + Bc <= seq) ? Bc : (seq - k0);
            for (int j = 0; j < kc; j++) {
                int ki = k0 + j;
                if (attn_window > 0 && (ki < k_lo || ki >= k_hi)) {
                    scores[j] = -FLT_MAX;
                    continue;
                }
                float dot = 0.0f;
                int d = 0;
                for (; d + 1 < D; d += 2) {
                    float2 qv = __half22float2(
                        *reinterpret_cast<const __half2 *>(&Qs[t * D + d]));
                    float2 kv = __half22float2(
                        *reinterpret_cast<const __half2 *>(&Ks[j * D + d]));
                    dot += qv.x * kv.x + qv.y * kv.y;
                }
                for (; d < D; d++)
                    dot += __half2float(Qs[t * D + d]) *
                           __half2float(Ks[j * D + d]);
                scores[j] = dot * scale;
                scores[j] += h3_dist_score_bias(qi, k0 + j);
                if (scores[j] > row_max) row_max = scores[j];
            }
            float m_new = (row_max > m_i) ? row_max : m_i;
            float alpha = expf(m_i - m_new);
            float l_new = l_i * alpha;
            for (int d = 0; d < D; d++) Os[t * D + d] *= alpha;
            for (int j = 0; j < kc; j++) {
                if (!(scores[j] > -FLT_MAX / 4)) continue;
                float p = expf(scores[j] - m_new);
                l_new += p;
                for (int d = 0; d < D; d++)
                    Os[t * D + d] += p * __half2float(Vs[j * D + d]);
            }
            m_i = m_new;
            l_i = l_new;
        }
        __syncthreads();
    }

    if (q_valid) {
        float inv = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
        float *o_row =
            Out + ((size_t)qi * (size_t)heads + (size_t)h) * (size_t)D;
        for (int d = 0; d < D; d++) o_row[d] = Os[t * D + d] * inv;
    }
}

/* F1/G8: SageAttention2-style opt-in — INT8 Q/K per-row + FP16 V.
 * Not a full Sage2++ MMA port; gate via check_sdpa (looser tol). */
enum { H3_CUDA_SAGE2_BR = 64, H3_CUDA_SAGE2_BC = 64 };

__global__ void h3_cuda_sdpa_sage2_kernel(const __half *__restrict__ Q,
                                          const __half *__restrict__ K,
                                          const __half *__restrict__ V,
                                          float *__restrict__ O, int seq,
                                          int heads, int D, float scale,
                                          int attn_window) {
    const int Br = H3_CUDA_SAGE2_BR;
    const int Bc = H3_CUDA_SAGE2_BC;
    const int head = blockIdx.y;
    const int tile_r = blockIdx.x;
    const int tid = threadIdx.x;
    if (head >= heads) return;
    const int qo0 = tile_r * Br;
    if (qo0 >= seq) return;
    const int q_rows = min(Br, seq - qo0);

    extern __shared__ char smem_sage[];
    __half *sQ = (__half *)smem_sage;
    __half *sK = sQ + Br * D;
    __half *sV = sK + Bc * D;
    float *sAcc = (float *)(sV + Bc * D);
    float *sM = sAcc + Br * D;
    float *sL = sM + Br;
    int8_t *sQi8 = (int8_t *)(sL + Br);
    int8_t *sKi8 = sQi8 + Br * D;
    float *sQs = (float *)(sKi8 + Bc * D);
    float *sKs = sQs + Br;

    for (int r = tid; r < q_rows * D; r += blockDim.x)
        sQ[r] = Q[((size_t)(qo0 + r / D) * (size_t)heads + head) * D + (r % D)];
    for (int r = tid; r < q_rows * D; r += blockDim.x) sAcc[r] = 0.0f;
    for (int r = tid; r < q_rows; r += blockDim.x) {
        sM[r] = -1e30f;
        sL[r] = 0.0f;
    }
    __syncthreads();

    for (int r = tid; r < q_rows; r += blockDim.x) {
        float amax = 1e-8f;
        for (int d = 0; d < D; d++) {
            float vv = fabsf(__half2float(sQ[r * D + d]));
            if (vv > amax) amax = vv;
        }
        sQs[r] = amax / 127.0f;
        float inv = 127.0f / amax;
        for (int d = 0; d < D; d++) {
            int q = (int)lrintf(__half2float(sQ[r * D + d]) * inv);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            sQi8[r * D + d] = (int8_t)q;
        }
    }
    __syncthreads();

    for (int kc = 0; kc < seq; kc += Bc) {
        const int kv_cols = min(Bc, seq - kc);
        if (attn_window > 0) {
            int mid = qo0 + q_rows / 2;
            if (kc + kv_cols - 1 < mid - attn_window || kc > mid + attn_window)
                continue;
        }
        for (int r = tid; r < kv_cols * D; r += blockDim.x) {
            int t = kc + r / D;
            int d = r % D;
            sK[r] = K[((size_t)t * (size_t)heads + head) * D + d];
            sV[r] = V[((size_t)t * (size_t)heads + head) * D + d];
        }
        __syncthreads();
        for (int c = tid; c < kv_cols; c += blockDim.x) {
            float amax = 1e-8f;
            for (int d = 0; d < D; d++) {
                float vv = fabsf(__half2float(sK[c * D + d]));
                if (vv > amax) amax = vv;
            }
            sKs[c] = amax / 127.0f;
            float inv = 127.0f / amax;
            for (int d = 0; d < D; d++) {
                int q = (int)lrintf(__half2float(sK[c * D + d]) * inv);
                if (q > 127) q = 127;
                if (q < -127) q = -127;
                sKi8[c * D + d] = (int8_t)q;
            }
        }
        __syncthreads();

        for (int r = tid; r < q_rows; r += blockDim.x) {
            float m_i = sM[r];
            float l_i = sL[r];
            float scores[H3_CUDA_SAGE2_BC];
            float m_tile = -1e30f;
            for (int c = 0; c < kv_cols; c++) {
                int dot = 0;
                const int8_t *qi = &sQi8[r * D];
                const int8_t *ki = &sKi8[c * D];
                int d = 0;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
                for (; d + 3 < D; d += 4) {
                    int a = *reinterpret_cast<const int *>(qi + d);
                    int b = *reinterpret_cast<const int *>(ki + d);
                    dot = __dp4a(a, b, dot);
                }
#endif
                for (; d < D; d++) dot += (int)qi[d] * (int)ki[d];
                float s = (float)dot * sQs[r] * sKs[c] * scale;
                scores[c] = s;
                if (s > m_tile) m_tile = s;
            }
            float m_new = m_i > m_tile ? m_i : m_tile;
            float alpha = (m_i > -1e29f) ? expf(m_i - m_new) : 0.0f;
            float l_tile = 0.0f;
            for (int c = 0; c < kv_cols; c++) {
                scores[c] = expf(scores[c] - m_new);
                l_tile += scores[c];
            }
            float l_new = alpha * l_i + l_tile;
            for (int d = 0; d < D; d++) {
                float acc = sAcc[r * D + d] * alpha;
                for (int c = 0; c < kv_cols; c++)
                    acc += scores[c] * __half2float(sV[c * D + d]);
                sAcc[r * D + d] = acc;
            }
            sM[r] = m_new;
            sL[r] = l_new;
        }
        __syncthreads();
    }

    for (int r = tid; r < q_rows; r += blockDim.x) {
        float inv = 1.0f / fmaxf(sL[r], 1e-20f);
        for (int d = 0; d < D; d++)
            O[((size_t)(qo0 + r) * (size_t)heads + head) * D + d] =
                sAcc[r * D + d] * inv;
    }
}

static int h3_cuda_sdpa_sage2_dev(h3_cuda *cuda, const __half *d_qh,
                                  const __half *d_kh, const __half *d_vh,
                                  float *d_o, size_t seq, size_t heads,
                                  size_t head_dim, float scale, int attn_window,
                                  double *sdpa_ms, char *error,
                                  size_t error_size) {
    const int Br = H3_CUDA_SAGE2_BR;
    const int Bc = H3_CUDA_SAGE2_BC;
    size_t smem;
    dim3 grid, block;
    cudaError_t st;
    cudaEvent_t ev0 = NULL, ev1 = NULL;
    float elapsed = 0.0f;
    static int smem_set = 0;

    if (head_dim > 128 || head_dim == 0) {
        h3_cuda_set_error(error, error_size, "sage2 head_dim unsupported");
        return 0;
    }
    smem = (size_t)(Br + 2 * Bc) * head_dim * sizeof(__half) +
           (size_t)Br * head_dim * sizeof(float) +
           (size_t)Br * 2 * sizeof(float) +
           (size_t)(Br + Bc) * head_dim * sizeof(int8_t) +
           (size_t)(Br + Bc) * sizeof(float);
    if (!smem_set) {
        st = cudaFuncSetAttribute(h3_cuda_sdpa_sage2_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  (int)smem);
        if (st != cudaSuccess) {
            h3_cuda_set_error_cuda(error, error_size, "sage2 smem attr", st);
            return 0;
        }
        smem_set = 1;
    }
    grid.x = (unsigned)((seq + Br - 1) / Br);
    grid.y = (unsigned)heads;
    grid.z = 1;
    block.x = 128;
    block.y = block.z = 1;
    if (!(cuda->quiet || cuda->graph_capture) && sdpa_ms) {
        cudaEventCreate(&ev0);
        cudaEventCreate(&ev1);
        if (ev0) cudaEventRecord(ev0, 0);
    }
    h3_cuda_sdpa_sage2_kernel<<<grid, block, smem>>>(
        d_qh, d_kh, d_vh, d_o, (int)seq, (int)heads, (int)head_dim, scale,
        attn_window);
    st = cudaGetLastError();
    if (st != cudaSuccess) {
        if (ev0) cudaEventDestroy(ev0);
        if (ev1) cudaEventDestroy(ev1);
        h3_cuda_set_error_cuda(error, error_size, "sage2 kernel", st);
        return 0;
    }
    if (ev0 && ev1) {
        cudaEventRecord(ev1, 0);
        cudaEventSynchronize(ev1);
        cudaEventElapsedTime(&elapsed, ev0, ev1);
        if (sdpa_ms) *sdpa_ms = (double)elapsed;
        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
    } else if (sdpa_ms) {
        *sdpa_ms = 0.0;
    }
    return 1;
}

#if defined(H3_HAVE_CUDNN)
/* cuDNN fused Flash SDPA. Q/K/V live in d_sdpa_scratch.
 * qkv_is_padded: caller already packed tensors at align_seq strides. */
static int h3_cuda_sdpa_cudnn_dev(h3_cuda *cuda, const __half *d_qh,
                                  const __half *d_kh, const __half *d_vh,
                                  float *d_o, size_t seq, size_t heads,
                                  size_t head_dim, float scale, int causal,
                                  int qkv_is_padded, double *sdpa_ms,
                                  char *error, size_t error_size) {
#if defined(H3_HAVE_CUDNN_FRONTEND)
    size_t padded = h3_cuda_sdpa_cudnn_align_seq(seq);
    size_t S = seq * heads * head_dim;
    size_t P = padded * heads * head_dim;
    size_t need = (padded > seq ? P : S) * 4 * sizeof(__half);
    __half *base;
    __half *qh = (__half *)d_qh;
    __half *kh = (__half *)d_kh;
    __half *vh = (__half *)d_vh;
    __half *d_oh;
    int causal_flag = causal ? 1 : 0;
    char local_err[256];

    if (!cuda || !cuda->cudnn_ok) {
        h3_cuda_set_error(error, error_size, "cudnn handle not ready");
        return 0;
    }
    if (h3_cuda_sdpa_cudnn_shape_blocked(seq, heads, head_dim, causal_flag)) {
        h3_cuda_set_error(error, error_size, "cudnn sdpa cached fail");
        return 0;
    }
    if (padded > seq) need = P * 4 * sizeof(__half);
    else need = S * 4 * sizeof(__half);
    if (!cuda->d_sdpa_scratch || cuda->sdpa_scratch_bytes < need) {
        h3_cuda_set_error(error, error_size, "cudnn sdpa scratch too small");
        return 0;
    }
    base = (__half *)cuda->d_sdpa_scratch;
    local_err[0] = '\0';

    /* Known-good padded layout from a previous layer: do not retry RAW. */
    if (qkv_is_padded && padded > seq) {
        int mask = causal_flag ? 0 : 1;
        d_oh = base + 3 * P;
        if (h3_cuda_sdpa_cudnn_f16_run_ex(
                (void *)cuda->cudnn, qh, kh, vh, d_oh, d_o, padded, heads,
                head_dim, scale, causal_flag, seq, mask, sdpa_ms, error,
                error_size))
            return 1;
        if (causal_flag &&
            h3_cuda_sdpa_cudnn_f16_run_ex(
                (void *)cuda->cudnn, qh, kh, vh, d_oh, d_o, padded, heads,
                head_dim, scale, causal_flag, seq, 1, sdpa_ms, error,
                error_size)) {
            h3_cuda_sdpa_cudnn_note_variant(seq, heads, head_dim, causal_flag,
                                            H3_CUDNN_VAR_PAD_MASK);
            return 1;
        }
        /* H/U (28, 7) has no pad-mask engine on this Blackwell stack.
         * The first `seq` THD rows of the padded buffer are a valid RAW
         * tensor; try that before branding the shape dead (gold H=56
         * still hits PAD_MASK above and never reaches here). */
        if (!h3_cuda_sdpa_cudnn_variant_failed(seq, heads, head_dim,
                                               causal_flag, H3_CUDNN_VAR_RAW)) {
            if (error && error_size) error[0] = '\0';
            if (h3_cuda_sdpa_cudnn_f16_run_ex(
                    (void *)cuda->cudnn, qh, kh, vh, d_oh, d_o, seq, heads,
                    head_dim, scale, causal_flag, seq, 0, sdpa_ms, error,
                    error_size)) {
                h3_cuda_sdpa_cudnn_note_variant(seq, heads, head_dim,
                                                causal_flag, H3_CUDNN_VAR_RAW);
                return 1;
            }
            if (!(error && strstr(error, "OOM")))
                h3_cuda_sdpa_cudnn_note_variant_fail(seq, heads, head_dim,
                                                     causal_flag,
                                                     H3_CUDNN_VAR_RAW);
        }
        if (!(error && strstr(error, "OOM")))
            h3_cuda_sdpa_cudnn_note_variant(seq, heads, head_dim, causal_flag,
                                            H3_CUDNN_VAR_FAIL);
        h3_cuda_sdpa_cudnn_log_fallback(seq, padded, head_dim, causal_flag,
                                        (error && error[0]) ? error : local_err);
        return 0;
    }

    if (!h3_cuda_sdpa_cudnn_variant_failed(seq, heads, head_dim, causal_flag,
                                           H3_CUDNN_VAR_RAW)) {
        d_oh = base + 3 * S;
        if (h3_cuda_sdpa_cudnn_f16_run_ex(
                (void *)cuda->cudnn, qh, kh, vh, d_oh, d_o, seq, heads,
                head_dim, scale, causal_flag, seq, 0, sdpa_ms, error,
                error_size)) {
            h3_cuda_sdpa_cudnn_note_variant(seq, heads, head_dim, causal_flag,
                                            H3_CUDNN_VAR_RAW);
            return 1;
        }
        if (!(error && strstr(error, "OOM")))
            h3_cuda_sdpa_cudnn_note_variant_fail(seq, heads, head_dim,
                                                 causal_flag, H3_CUDNN_VAR_RAW);
        if (error && error[0]) {
            snprintf(local_err, sizeof(local_err), "%s", error);
            error[0] = '\0';
        }
    }

    if (padded > seq) {
        if (!h3_cuda_sdpa_expand_qkv_in_scratch(cuda, base, seq, padded, heads,
                                                head_dim, error, error_size))
            return 0;
        qh = base;
        kh = base + P;
        vh = base + 2 * P;
        d_oh = base + 3 * P;

        if (causal_flag &&
            !h3_cuda_sdpa_cudnn_variant_failed(seq, heads, head_dim,
                                               causal_flag,
                                               H3_CUDNN_VAR_PAD_CAUSAL)) {
            if (h3_cuda_sdpa_cudnn_f16_run_ex(
                    (void *)cuda->cudnn, qh, kh, vh, d_oh, d_o, padded, heads,
                    head_dim, scale, causal_flag, seq, 0, sdpa_ms, error,
                    error_size)) {
                    h3_cuda_sdpa_cudnn_note_variant(seq, heads, head_dim,
                                                causal_flag,
                                                H3_CUDNN_VAR_PAD_CAUSAL);
                return 1;
            }
            if (!(error && strstr(error, "OOM")))
                h3_cuda_sdpa_cudnn_note_variant_fail(seq, heads, head_dim,
                                                     causal_flag,
                                                     H3_CUDNN_VAR_PAD_CAUSAL);
            if (error && error[0]) {
                snprintf(local_err, sizeof(local_err), "%s", error);
                error[0] = '\0';
            }
        }

        if (!h3_cuda_sdpa_cudnn_variant_failed(seq, heads, head_dim, causal_flag,
                                               H3_CUDNN_VAR_PAD_MASK)) {
            if (h3_cuda_sdpa_cudnn_f16_run_ex(
                    (void *)cuda->cudnn, qh, kh, vh, d_oh, d_o, padded, heads,
                    head_dim, scale, causal_flag, seq, 1, sdpa_ms, error,
                    error_size)) {
                    h3_cuda_sdpa_cudnn_note_variant(seq, heads, head_dim,
                                                causal_flag,
                                                H3_CUDNN_VAR_PAD_MASK);
                return 1;
            }
            if (!(error && strstr(error, "OOM")))
                h3_cuda_sdpa_cudnn_note_variant_fail(seq, heads, head_dim,
                                                     causal_flag,
                                                     H3_CUDNN_VAR_PAD_MASK);
            if (error && error[0]) {
                snprintf(local_err, sizeof(local_err), "%s", error);
                error[0] = '\0';
            }
        }
    }

    h3_cuda_sdpa_cudnn_log_fallback(
        seq, padded, head_dim, causal_flag,
        local_err[0] ? local_err : ((error && error[0]) ? error : "(no msg)"));
    if (!((local_err[0] && strstr(local_err, "OOM")) ||
          (error && strstr(error, "OOM")))) {
        h3_cuda_sdpa_cudnn_note_variant(seq, heads, head_dim, causal_flag,
                                        H3_CUDNN_VAR_FAIL);
        if (error && error_size)
            snprintf(error, error_size, "cudnn sdpa cached fail");
    }
    return 0;
#else
    (void)cuda;
    (void)d_qh;
    (void)d_kh;
    (void)d_vh;
    (void)d_o;
    (void)seq;
    (void)heads;
    (void)head_dim;
    (void)scale;
    (void)causal;
    (void)qkv_is_padded;
    (void)sdpa_ms;
    h3_cuda_set_error(error, error_size,
                      "built without cudnn_frontend; use flash/tc");
    return 0;
#endif
}
#endif

/* ---- Sol-Attn (opt-in). Nested-loop: pooled-K routing + exact tiles +
 * pooled-V correction. Isolated A/B vs cuDNN; default generate stays
 * fused cuDNN. Block=64, tau/sink via env. D=128 only. */
enum { H3_SOL_B = 64, H3_SOL_RQ = 8 };

static float *g_d_sol_kpool = NULL;
static float *g_d_sol_vpool = NULL;
static unsigned long long *g_d_sol_stats = NULL;
static size_t g_sol_pool_cap = 0;
static double g_sol_last_exact_frac = -1.0;
static int g_sol_logged = 0;

extern "C" double h3_cuda_sol_last_exact_frac(void) {
    return g_sol_last_exact_frac;
}

static float h3_sol_tau_env(void) {
    const char *e = getenv("H3_SOL_TAU");
    if (!e || !e[0]) return 1.3f;
    return (float)atof(e);
}

static int h3_sol_sink_tokens_env(void) {
    const char *e = getenv("H3_SOL_SINK_TOKENS");
    if (!e || !e[0]) return 988; /* cafe text 574 + audio 414 */
    int v = atoi(e);
    return v < 0 ? 0 : v;
}

__global__ void h3_cuda_sol_pool_kv_kernel(const __half *__restrict__ K,
                                           const __half *__restrict__ V,
                                           float *__restrict__ Kpool,
                                           float *__restrict__ Vpool, int seq,
                                           int heads, int nB, int B) {
    const int D = H3_CUDA_SDPA_F16_D;
    const int jb = (int)blockIdx.x;
    const int h = (int)blockIdx.y;
    const int d = (int)threadIdx.x;
    if (jb >= nB || h >= heads || d >= D) return;
    const int k0 = jb * B;
    int n = seq - k0;
    if (n > B) n = B;
    if (n <= 0) return;
    float ks = 0.0f, vs = 0.0f;
    for (int i = 0; i < n; i++) {
        const size_t idx =
            ((size_t)(k0 + i) * (size_t)heads + (size_t)h) * (size_t)D +
            (size_t)d;
        ks += __half2float(K[idx]);
        vs += __half2float(V[idx]);
    }
    const float inv = 1.0f / (float)n;
    const size_t po =
        ((size_t)h * (size_t)nB + (size_t)jb) * (size_t)D + (size_t)d;
    Kpool[po] = ks * inv;
    Vpool[po] = vs * inv;
}

/* One CTA = RQ queries × 1 head. Lane-split D=128 like flash_f16_d128. */
__global__ void h3_cuda_sol_fwd_d128_kernel(
    const __half *__restrict__ Q, const __half *__restrict__ K,
    const __half *__restrict__ V, const float *__restrict__ Kpool,
    const float *__restrict__ Vpool, float *__restrict__ Out, int seq,
    int heads, float scale, float tau, int sink_blocks, int nB,
    unsigned long long *__restrict__ stats) {
    const int D = H3_CUDA_SDPA_F16_D;
    const int B = H3_SOL_B;
    const int RQ = H3_SOL_RQ;
    const int h = (int)blockIdx.x;
    const int q_base = (int)blockIdx.y * RQ;
    const int lane = (int)threadIdx.x;
    const int warp = (int)threadIdx.y;
    const int qi = q_base + warp;
    const int q_valid = (qi < seq);
    const int d0 = lane * 4;

    extern __shared__ char sm_sol[];
    __half *Ks = (__half *)sm_sol;
    __half *Vs = Ks + (size_t)B * (size_t)D;
    float *Pscore = (float *)(Vs + (size_t)B * (size_t)D); /* [RQ][nB] */
    unsigned char *Route =
        (unsigned char *)(Pscore + (size_t)RQ * (size_t)nB);
    size_t route_pad = ((size_t)RQ * (size_t)nB + 3u) & ~3u;
    int *any_exact = (int *)(Route + route_pad);
    float *Ss = (float *)(any_exact + 8); /* [RQ][B] exact-tile scores */

    float q_reg[4];
    float o_reg[4];
    float m_i = -FLT_MAX;
    float l_i = 0.0f;
#pragma unroll
    for (int u = 0; u < 4; u++) {
        q_reg[u] = 0.0f;
        o_reg[u] = 0.0f;
    }
    if (q_valid) {
        const __half *q_row =
            Q + ((size_t)qi * (size_t)heads + (size_t)h) * (size_t)D;
        float2 q01 =
            __half22float2(*reinterpret_cast<const __half2 *>(q_row + d0));
        float2 q23 = __half22float2(
            *reinterpret_cast<const __half2 *>(q_row + d0 + 2));
        q_reg[0] = q01.x;
        q_reg[1] = q01.y;
        q_reg[2] = q23.x;
        q_reg[3] = q23.y;
    }
    __syncthreads();

    for (int j = 0; j < nB; j++) {
        float s = -FLT_MAX;
        if (q_valid) {
            const float *kp =
                Kpool + ((size_t)h * (size_t)nB + (size_t)j) * (size_t)D +
                (size_t)d0;
            float partial = q_reg[0] * kp[0] + q_reg[1] * kp[1] +
                            q_reg[2] * kp[2] + q_reg[3] * kp[3];
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                partial += __shfl_down_sync(0xffffffffu, partial, offset);
            s = __shfl_sync(0xffffffffu, partial, 0) * scale;
            if (lane == 0) Pscore[(size_t)warp * (size_t)nB + (size_t)j] = s;
        }
    }
    __syncthreads();

    if (q_valid && lane == 0) {
        float mu = 0.0f;
        for (int j = 0; j < nB; j++)
            mu += Pscore[(size_t)warp * (size_t)nB + (size_t)j];
        mu /= (float)nB;
        float var = 0.0f;
        for (int j = 0; j < nB; j++) {
            float d = Pscore[(size_t)warp * (size_t)nB + (size_t)j] - mu;
            var += d * d;
        }
        float sigma = sqrtf(var / (float)nB);
        float thr = mu + tau * sigma;
        int q_block = qi / B;
        for (int j = 0; j < nB; j++) {
            int neigh = q_block - j;
            if (neigh < 0) neigh = -neigh;
            unsigned char r =
                (Pscore[(size_t)warp * (size_t)nB + (size_t)j] > thr) ||
                (neigh <= 1) || (j < sink_blocks);
            Route[(size_t)warp * (size_t)nB + (size_t)j] = r;
        }
    }
    __syncthreads();

    for (int j = 0; j < nB; j++) {
        if (warp == 0 && lane == 0) {
            int any = 0;
            for (int w = 0; w < RQ; w++) {
                int qj = q_base + w;
                if (qj < seq && Route[(size_t)w * (size_t)nB + (size_t)j])
                    any = 1;
            }
            any_exact[0] = any;
        }
        __syncthreads();
        const int do_exact = any_exact[0];
        const int k0 = j * B;
        const int kc = (k0 + B <= seq) ? B : (seq - k0);
        const int my_route =
            q_valid && Route[(size_t)warp * (size_t)nB + (size_t)j];

        if (do_exact) {
            const int tid = warp * 32 + lane;
            const int nthreads = RQ * 32;
            for (int idx2 = tid; idx2 < (B * D) / 2; idx2 += nthreads) {
                const int idx = idx2 * 2;
                const int r = idx / D;
                const int d = idx % D;
                const int ki = k0 + r;
                __half2 *ks2 = reinterpret_cast<__half2 *>(Ks + idx);
                __half2 *vs2 = reinterpret_cast<__half2 *>(Vs + idx);
                if (r < kc && ki < seq) {
                    const size_t base =
                        ((size_t)ki * (size_t)heads + (size_t)h) * (size_t)D +
                        (size_t)d;
                    *ks2 = *reinterpret_cast<const __half2 *>(K + base);
                    *vs2 = *reinterpret_cast<const __half2 *>(V + base);
                } else {
                    *ks2 = __float2half2_rn(0.0f);
                    *vs2 = __float2half2_rn(0.0f);
                }
            }
            __syncthreads();
            if (q_valid && my_route) {
                float row_max = -FLT_MAX;
                for (int t = 0; t < kc; t++) {
                    float2 k01 = __half22float2(
                        *reinterpret_cast<const __half2 *>(&Ks[t * D + d0]));
                    float2 k23 = __half22float2(
                        *reinterpret_cast<const __half2 *>(
                            &Ks[t * D + d0 + 2]));
                    float partial = q_reg[0] * k01.x + q_reg[1] * k01.y +
                                    q_reg[2] * k23.x + q_reg[3] * k23.y;
#pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1)
                        partial +=
                            __shfl_down_sync(0xffffffffu, partial, offset);
                    float s = __shfl_sync(0xffffffffu, partial, 0) * scale;
                    if (lane == 0) Ss[(size_t)warp * (size_t)B + (size_t)t] = s;
                    if (s > row_max) row_max = s;
                }
                __syncwarp();
                float m_new = (row_max > m_i) ? row_max : m_i;
                float alpha = __expf(m_i - m_new);
                float l_new = l_i * alpha;
#pragma unroll
                for (int u = 0; u < 4; u++) o_reg[u] *= alpha;
                for (int t = 0; t < kc; t++) {
                    float p = __expf(Ss[(size_t)warp * (size_t)B + (size_t)t] -
                                     m_new);
                    l_new += p;
                    float2 v01 = __half22float2(
                        *reinterpret_cast<const __half2 *>(&Vs[t * D + d0]));
                    float2 v23 = __half22float2(
                        *reinterpret_cast<const __half2 *>(
                            &Vs[t * D + d0 + 2]));
                    o_reg[0] += p * v01.x;
                    o_reg[1] += p * v01.y;
                    o_reg[2] += p * v23.x;
                    o_reg[3] += p * v23.y;
                }
                m_i = m_new;
                l_i = l_new;
                if (lane == 0 && stats) atomicAdd(stats, 1ULL);
            } else if (q_valid) {
                float s = Pscore[(size_t)warp * (size_t)nB + (size_t)j];
                const float *vp =
                    Vpool + ((size_t)h * (size_t)nB + (size_t)j) * (size_t)D +
                    (size_t)d0;
                float m_new = (s > m_i) ? s : m_i;
                float alpha = __expf(m_i - m_new);
                float p = __expf(s - m_new);
#pragma unroll
                for (int u = 0; u < 4; u++)
                    o_reg[u] = o_reg[u] * alpha + p * vp[u];
                l_i = l_i * alpha + p;
                m_i = m_new;
            }
            if (lane == 0 && q_valid && stats) atomicAdd(stats + 1, 1ULL);
            __syncthreads();
        } else if (q_valid) {
            float s = Pscore[(size_t)warp * (size_t)nB + (size_t)j];
            const float *vp =
                Vpool + ((size_t)h * (size_t)nB + (size_t)j) * (size_t)D +
                (size_t)d0;
            float m_new = (s > m_i) ? s : m_i;
            float alpha = __expf(m_i - m_new);
            float p = __expf(s - m_new);
#pragma unroll
            for (int u = 0; u < 4; u++)
                o_reg[u] = o_reg[u] * alpha + p * vp[u];
            l_i = l_i * alpha + p;
            m_i = m_new;
            if (lane == 0 && stats) atomicAdd(stats + 1, 1ULL);
        }
        __syncthreads();
    }

    if (q_valid) {
        float inv = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
        float *o_row =
            Out + ((size_t)qi * (size_t)heads + (size_t)h) * (size_t)D;
#pragma unroll
        for (int u = 0; u < 4; u++) o_row[d0 + u] = o_reg[u] * inv;
    }
}

static int h3_cuda_sdpa_sol_dev(h3_cuda *cuda, const __half *d_qh,
                                const __half *d_kh, const __half *d_vh,
                                float *d_o, size_t seq, size_t heads,
                                size_t head_dim, float scale, double *sdpa_ms,
                                char *error, size_t error_size) {
    const int B = H3_SOL_B;
    const int RQ = H3_SOL_RQ;
    const int D = H3_CUDA_SDPA_F16_D;
    int nB;
    int sink_blocks;
    float tau;
    size_t pool_n;
    cudaError_t st;
    cudaEvent_t ev0 = NULL, ev1 = NULL;
    float elapsed = 0.0f;
    size_t smem;
    dim3 grid, block;
    unsigned long long stats_h[2];
    static int smem_set = 0;

    (void)cuda;
    if (head_dim != (size_t)D) {
        h3_cuda_set_error(error, error_size, "sol attn requires head_dim=128");
        return 0;
    }
    nB = (int)((seq + (size_t)B - 1) / (size_t)B);
    if (nB <= 0) {
        h3_cuda_set_error(error, error_size, "sol attn empty seq");
        return 0;
    }
    tau = h3_sol_tau_env();
    sink_blocks = (h3_sol_sink_tokens_env() + B - 1) / B;
    pool_n = (size_t)heads * (size_t)nB * (size_t)D;
    if (pool_n > g_sol_pool_cap) {
        if (g_d_sol_kpool) cudaFree(g_d_sol_kpool);
        if (g_d_sol_vpool) cudaFree(g_d_sol_vpool);
        g_d_sol_kpool = NULL;
        g_d_sol_vpool = NULL;
        g_sol_pool_cap = 0;
        if (cudaMalloc((void **)&g_d_sol_kpool, pool_n * sizeof(float)) !=
                cudaSuccess ||
            cudaMalloc((void **)&g_d_sol_vpool, pool_n * sizeof(float)) !=
                cudaSuccess) {
            if (g_d_sol_kpool) cudaFree(g_d_sol_kpool);
            if (g_d_sol_vpool) cudaFree(g_d_sol_vpool);
            g_d_sol_kpool = NULL;
            g_d_sol_vpool = NULL;
            h3_cuda_set_error(error, error_size, "sol pool cudaMalloc");
            return 0;
        }
        g_sol_pool_cap = pool_n;
    }
    if (!g_d_sol_stats) {
        if (cudaMalloc((void **)&g_d_sol_stats,
                       2 * sizeof(unsigned long long)) != cudaSuccess) {
            h3_cuda_set_error(error, error_size, "sol stats cudaMalloc");
            return 0;
        }
    }

    smem = (size_t)B * (size_t)D * 2 * sizeof(__half) +
           (size_t)RQ * (size_t)nB * sizeof(float) +
           (((size_t)RQ * (size_t)nB + 3u) & ~3u) + 32u +
           (size_t)RQ * (size_t)B * sizeof(float);
    if (!smem_set) {
        st = cudaFuncSetAttribute(h3_cuda_sol_fwd_d128_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  65536);
        if (st != cudaSuccess) {
            h3_cuda_set_error_cuda(error, error_size, "sol smem attr", st);
            return 0;
        }
        smem_set = 1;
    }
    if (smem > 65536) {
        h3_cuda_set_error(error, error_size, "sol smem too large");
        return 0;
    }

    st = cudaMemset(g_d_sol_stats, 0, 2 * sizeof(unsigned long long));
    if (st != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "sol stats memset", st);
        return 0;
    }

    if (sdpa_ms) {
        cudaEventCreate(&ev0);
        cudaEventCreate(&ev1);
        if (ev0) cudaEventRecord(ev0, 0);
    }

    {
        dim3 pgrid((unsigned)nB, (unsigned)heads, 1);
        h3_cuda_sol_pool_kv_kernel<<<pgrid, D>>>(d_kh, d_vh, g_d_sol_kpool,
                                                 g_d_sol_vpool, (int)seq,
                                                 (int)heads, nB, B);
    }
    grid.x = (unsigned)heads;
    grid.y = (unsigned)((seq + (size_t)RQ - 1) / (size_t)RQ);
    grid.z = 1;
    block.x = 32;
    block.y = (unsigned)RQ;
    block.z = 1;
    h3_cuda_sol_fwd_d128_kernel<<<grid, block, smem>>>(
        d_qh, d_kh, d_vh, g_d_sol_kpool, g_d_sol_vpool, d_o, (int)seq,
        (int)heads, scale, tau, sink_blocks, nB, g_d_sol_stats);
    st = cudaGetLastError();
    if (st != cudaSuccess) {
        if (ev0) cudaEventDestroy(ev0);
        if (ev1) cudaEventDestroy(ev1);
        h3_cuda_set_error_cuda(error, error_size, "sol kernel", st);
        return 0;
    }
    if (ev0 && ev1) {
        cudaEventRecord(ev1, 0);
        cudaEventSynchronize(ev1);
        cudaEventElapsedTime(&elapsed, ev0, ev1);
        if (sdpa_ms) *sdpa_ms = (double)elapsed;
        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
    } else if (sdpa_ms) {
        cudaDeviceSynchronize();
        *sdpa_ms = 0.0;
    }
    if (sdpa_ms || !g_sol_logged) {
        stats_h[0] = stats_h[1] = 0;
        cudaMemcpy(stats_h, g_d_sol_stats, 2 * sizeof(unsigned long long),
                   cudaMemcpyDeviceToHost);
        if (stats_h[1] > 0)
            g_sol_last_exact_frac = (double)stats_h[0] / (double)stats_h[1];
        else
            g_sol_last_exact_frac = -1.0;
        if (!g_sol_logged) {
            fprintf(stderr,
                    "[h3] sol-attn tau=%.2f sink_blocks=%d nB=%d exact_frac=%.3f "
                    "ms=%.1f seq=%zu heads=%zu\n",
                    (double)tau, sink_blocks, nB, g_sol_last_exact_frac,
                    sdpa_ms ? *sdpa_ms : 0.0, seq, heads);
            g_sol_logged = 1;
        }
    }
    return 1;
}

static int h3_cuda_sdpa_f16_launch_dev_ex(
    h3_cuda *cuda, const float *d_q, const float *d_k, const float *d_v,
    size_t seq, size_t heads, size_t head_dim, float scale, int attn_window,
    float *d_o, double *sdpa_ms, char *backend_out, size_t backend_out_size,
    char *error, size_t error_size, int qkv_already_half) {
    __half *d_qh = NULL, *d_kh = NULL, *d_vh = NULL;
    cudaError_t status;
    cudaEvent_t ev0 = NULL, ev1 = NULL;
    float elapsed = 0.0f;
    size_t elems;
    int ok = 0;
    const int use_d128 = (head_dim == (size_t)H3_CUDA_SDPA_F16_D);
    const int use_tc = use_d128 && h3_cuda_sdpa_want_tc();
    const char *backend = "cuda_flash_f16";
    /* cudaFuncSetAttribute is per-device. A process-wide flag made rank1+
     * skip the >48KiB dynamic-smem raise and launch with invalid argument. */
    enum { H3_SDPA_SMEM_DEV_MAX = 16 };
    static int smem_attr_set_d128[H3_SDPA_SMEM_DEV_MAX];
    static int smem_attr_set_tc[H3_SDPA_SMEM_DEV_MAX];
    static int smem_attr_set_gen[H3_SDPA_SMEM_DEV_MAX];
    int smem_dev;
    int this_dense = 0;
    int cute_timed = 0;

    if (sdpa_ms) *sdpa_ms = 0.0;
    if (backend_out && backend_out_size)
        snprintf(backend_out, backend_out_size, "%s", "unset");
    if (!cuda || !d_o || seq == 0 || heads == 0 || head_dim == 0 ||
        (!qkv_already_half && (!d_q || !d_k || !d_v))) {
        h3_cuda_set_error(error, error_size, "invalid sdpa_f16_dev args");
        return 0;
    }
    if (head_dim > 128) {
        h3_cuda_set_error(error, error_size, "head_dim > 128 not supported");
        return 0;
    }
    if (attn_window < 0) attn_window = 0;

    if (!h3_cuda_bind(cuda, error, error_size)) return 0;
    smem_dev = cuda->device_index;
    if (smem_dev < 0 || smem_dev >= H3_SDPA_SMEM_DEV_MAX) smem_dev = 0;

    elems = seq * heads * head_dim;
    /* Reserve Q/K/V (+ optional O half for cuDNN fused path). Size for the
     * aligned seq so a raw→pad retry does not realloc under the kernel. */
    {
        size_t half_n = elems * 3;
#if defined(H3_HAVE_CUDNN_FRONTEND)
        if (h3_cuda_sdpa_want_cudnn()) {
            size_t padded = h3_cuda_sdpa_cudnn_align_seq(seq);
            size_t pe = padded * heads * head_dim;
            half_n = pe * 4;
        }
#endif
        if (!h3_cuda_ensure_scratch(cuda, half_n * sizeof(__half), error,
                                    error_size))
            return 0;
    }
    {
        char *base = (char *)cuda->d_sdpa_scratch;
        size_t stride = elems;
#if defined(H3_HAVE_CUDNN_FRONTEND)
        if (h3_cuda_sdpa_want_cudnn() &&
            h3_cuda_sdpa_cudnn_want_padded(seq, heads, head_dim, 0))
            stride = h3_cuda_sdpa_cudnn_align_seq(seq) * heads * head_dim;
#endif
        d_qh = (__half *)base;
        d_kh = d_qh + stride;
        d_vh = d_kh + stride;
    }

    if (h3_cuda_want_events(cuda)) {
        status = cudaEventCreate(&ev0);
        if (status != cudaSuccess) {
            h3_cuda_set_error_cuda(error, error_size, "cudaEventCreate",
                                   status);
            return 0;
        }
        status = cudaEventCreate(&ev1);
        if (status != cudaSuccess) {
            cudaEventDestroy(ev0);
            h3_cuda_set_error_cuda(error, error_size, "cudaEventCreate",
                                   status);
            return 0;
        }
    }

    {
        const int thr = 256;
        const unsigned nblk =
            (unsigned)((elems + (size_t)thr - 1) / (size_t)thr);
        if (ev0) {
            status = cudaEventRecord(ev0, 0);
            if (status != cudaSuccess) {
                h3_cuda_set_error_cuda(error, error_size, "cudaEventRecord",
                                       status);
                goto cleanup;
            }
        }
        if (!qkv_already_half) {
            h3_cuda_f32_to_f16_3_kernel<<<nblk, thr, 0, h3_cuda_op_st(cuda)>>>(
                d_q, d_k, d_v, d_qh, d_kh, d_vh, elems);
#if defined(H3_HAVE_CUDNN_FRONTEND)
            if (h3_cuda_sdpa_want_cudnn() &&
                h3_cuda_sdpa_cudnn_want_padded(seq, heads, head_dim, 0)) {
                size_t padded = h3_cuda_sdpa_cudnn_align_seq(seq);
                size_t P = padded * heads * head_dim;
                if (P > elems) {
                    h3_cuda_zero_pad_tails(cuda, d_qh + elems, d_kh + elems,
                                           d_vh + elems, P - elems);
                }
            }
#endif
        }
        (void)nblk;
        (void)thr;

        if (!h3_cuda_dist_attn_active() && h3_cuda_sdpa_want_sol_cute() &&
            head_dim == (size_t)H3_CUDA_SDPA_F16_D && seq >= 4096) {
            int layer = g_sol_cute_layer++;
            int dense = (g_sol_cute_step < h3_sol_cute_dense_steps()) ||
                        (layer < h3_sol_cute_dense_layers());
            g_sol_cute_want++;
            if (!dense) {
                if (h3_cuda_sdpa_sol_cute_dev(cuda, d_qh, d_kh, d_vh, d_o, seq,
                                              heads, head_dim, error,
                                              error_size)) {
                    backend = g_sol_cute_backend[0] ? g_sol_cute_backend
                                                    : "cute_sm89";
                    g_sol_cute_hits++;
                    cute_timed = 1;
                    if (g_sol_cute_hits <= 3 || (g_sol_cute_hits % 50ull) == 0)
                        fprintf(stderr,
                                "[h3] sol_cute hit#%llu step=%d layer=%d "
                                "seq=%zu H=%zu backend=%s\n",
                                g_sol_cute_hits, g_sol_cute_step, layer, seq,
                                heads, backend);
                    ok = 1;
                    goto timed_done;
                }
                g_sol_cute_fallback++;
                fprintf(stderr,
                        "[h3] sol_cute fallback#%llu seq=%zu: %s\n",
                        g_sol_cute_fallback, seq,
                        (error && error[0]) ? error : "(no msg)");
                if (error && error_size) error[0] = '\0';
            } else {
                g_sol_cute_dense_skip++;
                this_dense = 1;
            }
        }
        if (!h3_cuda_dist_attn_active() && h3_cuda_sdpa_want_sol() &&
            head_dim == (size_t)H3_CUDA_SDPA_F16_D) {
            double *sol_ms = h3_cuda_want_events(cuda) ? sdpa_ms : NULL;
            if (h3_cuda_sdpa_sol_dev(cuda, d_qh, d_kh, d_vh, d_o, seq, heads,
                                     head_dim, scale, sol_ms, error,
                                     error_size)) {
                backend = "cuda_sol_attn";
                ok = 1;
                goto timed_done;
            }
            fprintf(stderr, "[h3] sol-attn fallback seq=%zu d=%zu: %s\n", seq,
                    head_dim, (error && error[0]) ? error : "(no msg)");
            if (error && error_size) error[0] = '\0';
        }
        if (!h3_cuda_dist_attn_active() && h3_cuda_sdpa_want_sage3() &&
            head_dim == (size_t)H3_CUDA_SDPA_F16_D && seq >= 128) {
            g_sage3_want++;
            if (h3_cuda_sdpa_sage3_dev(cuda, d_qh, d_kh, d_vh, d_o, seq, heads,
                                       head_dim, scale, error, error_size)) {
                backend = g_sage3_backend[0] ? g_sage3_backend
                                             : "sage3_blackwell_fp4";
                g_sage3_hits++;
                if (g_sage3_hits <= 3 || (g_sage3_hits % 50ull) == 0)
                    fprintf(stderr,
                            "[h3] sage3 hit#%llu seq=%zu H=%zu backend=%s\n",
                            g_sage3_hits, seq, heads, backend);
                ok = 1;
                goto timed_done;
            }
            g_sage3_fallback++;
            fprintf(stderr, "[h3] sage3 fallback#%llu seq=%zu: %s\n",
                    g_sage3_fallback, seq,
                    (error && error[0]) ? error : "(no msg)");
            if (error && error_size) error[0] = '\0';
        }
        if (!h3_cuda_dist_attn_active() && h3_cuda_sdpa_want_sage2() &&
            head_dim <= 128) {
            double *sage_ms = h3_cuda_want_events(cuda) ? sdpa_ms : NULL;
            if (h3_cuda_sdpa_sage2_dev(cuda, d_qh, d_kh, d_vh, d_o, seq, heads,
                                       head_dim, scale, attn_window, sage_ms,
                                       error, error_size)) {
                backend = "cuda_sage2";
                ok = 1;
                goto timed_done;
            }
            fprintf(stderr, "[h3] sage2 sdpa fallback seq=%zu d=%zu: %s\n", seq,
                    head_dim, (error && error[0]) ? error : "(no msg)");
            if (error && error_size) error[0] = '\0';
        }
#if defined(H3_HAVE_CUDNN)
        if (h3_cuda_sdpa_want_cudnn() && cuda->cudnn_ok && head_dim <= 128 &&
            !(h3_cuda_dist_attn_active() && g_dist_cudnn_fail)) {
            double *cudnn_ms = h3_cuda_want_events(cuda) ? sdpa_ms : NULL;
            int qkv_pad = 0;
            int blocked = 0;
            size_t padded = seq;
#if defined(H3_HAVE_CUDNN_FRONTEND)
            padded = h3_cuda_sdpa_cudnn_align_seq(seq);
            qkv_pad = h3_cuda_sdpa_cudnn_want_padded(seq, heads, head_dim, 0);
            blocked = h3_cuda_sdpa_cudnn_shape_blocked(seq, heads, head_dim, 0);
#endif
            if (h3_cuda_sdpa_cudnn_dev(cuda, d_qh, d_kh, d_vh, d_o, seq, heads,
                                       head_dim, scale, 0, qkv_pad, cudnn_ms,
                                       error, error_size)) {
                backend = h3_cuda_dist_attn_active() ? "cuda_cudnn_sdpa_dist"
                                                     : "cuda_cudnn_sdpa";
                ok = 1;
                goto timed_done;
            }
            /* First failure is logged inside the plan cache. Do not reprint
             * 400× for the same shape; still fall through to flash. */
            if (h3_cuda_dist_attn_active()) g_dist_cudnn_fail = 1;
            if (error && error[0] && !strstr(error, "cached fail"))
                fprintf(stderr,
                        "[h3] cudnn sdpa fallback seq=%zu d=%zu: %s\n", seq,
                        head_dim, error);
            if (!blocked && padded > seq &&
                !(error && strstr(error, "expand"))) {
                __half *base = (__half *)cuda->d_sdpa_scratch;
                size_t P = padded * heads * head_dim;
                d_qh = base;
                d_kh = base + P;
                d_vh = base + 2 * P;
            }
            if (error && error_size) error[0] = '\0';
            /* fall through to TC/flash */
        }
#else
        (void)h3_cuda_sdpa_want_cudnn;
#endif

        if (use_tc) {
            const int D = H3_CUDA_SDPA_F16_D;
            size_t smem =
                (size_t)(H3_CUDA_SDPA_TC_BR + 2 * H3_CUDA_SDPA_TC_BC) *
                    (size_t)D * sizeof(__half) +
                (size_t)H3_CUDA_SDPA_TC_BR * (size_t)D * sizeof(float) +
                (size_t)H3_CUDA_SDPA_TC_BR * 2 * sizeof(float) +
                (size_t)H3_CUDA_SDPA_TC_BR * (size_t)H3_CUDA_SDPA_TC_BC *
                    sizeof(float);
            if (!smem_attr_set_tc[smem_dev]) {
                status = cudaFuncSetAttribute(
                    h3_cuda_sdpa_flash_f16_tc_d128_kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
                if (status != cudaSuccess) {
                    h3_cuda_set_error_cuda(error, error_size,
                                           "cudaFuncSetAttribute tc smem",
                                           status);
                    goto cleanup;
                }
                smem_attr_set_tc[smem_dev] = 1;
            }
            unsigned q_tiles =
                (unsigned)((seq + (size_t)H3_CUDA_SDPA_TC_BR - 1) /
                           (size_t)H3_CUDA_SDPA_TC_BR);
            dim3 grid((unsigned)heads, q_tiles, 1);
            /* Multi-stream over head shards (stable grid shape per stream). */
            if (cuda->streams_ready && heads >= (size_t)H3_CUDA_N_STREAMS) {
                unsigned s;
                for (s = 0; s < (unsigned)H3_CUDA_N_STREAMS; s++) {
                    unsigned h_begin = (unsigned)((heads * s) / H3_CUDA_N_STREAMS);
                    unsigned h_end =
                        (unsigned)((heads * (s + 1)) / H3_CUDA_N_STREAMS);
                    if (h_end <= h_begin) continue;
                    dim3 g((unsigned)(h_end - h_begin), q_tiles, 1);
                    /* Offset Q/K/V/O by head — layout is [seq, heads, D],
                     * so per-head contiguous slice is not simple; keep full
                     * heads grid on default stream instead when layout blocks
                     * sharding. */
                    (void)g;
                    (void)h_begin;
                }
            }
            h3_cuda_sdpa_flash_f16_tc_d128_kernel<<<grid, 256, smem,
                                                    h3_cuda_op_st(cuda)>>>(
                d_qh, d_kh, d_vh, d_o, (int)seq, (int)heads, scale,
                attn_window);
            backend = attn_window > 0 ? "cuda_flash_f16_tc_win"
                                      : "cuda_flash_f16_tc";
        } else if (use_d128) {
            size_t smem =
                (size_t)H3_CUDA_SDPA_F16_BC * (size_t)H3_CUDA_SDPA_F16_D * 2 *
                    sizeof(__half) +
                (size_t)H3_CUDA_SDPA_F16_RQ * (size_t)H3_CUDA_SDPA_F16_BC *
                    sizeof(float);
            if (!smem_attr_set_d128[smem_dev]) {
                status = cudaFuncSetAttribute(
                    h3_cuda_sdpa_flash_f16_d128_kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
                if (status != cudaSuccess) {
                    h3_cuda_set_error_cuda(error, error_size,
                                           "cudaFuncSetAttribute smem", status);
                    goto cleanup;
                }
                smem_attr_set_d128[smem_dev] = 1;
            }
            unsigned q_tiles =
                (unsigned)((seq + (size_t)H3_CUDA_SDPA_F16_RQ - 1) /
                           (size_t)H3_CUDA_SDPA_F16_RQ);
            dim3 grid((unsigned)heads, q_tiles, 1);
            dim3 block(32, H3_CUDA_SDPA_F16_RQ, 1);
            h3_cuda_sdpa_flash_f16_d128_kernel<<<grid, block, smem,
                                                h3_cuda_op_st(cuda)>>>(
                d_qh, d_kh, d_vh, d_o, (int)seq, (int)heads, scale,
                attn_window);
            backend = attn_window > 0 ? "cuda_flash_f16_win" : "cuda_flash_f16";
        } else {
            size_t smem =
                ((size_t)H3_CUDA_SDPA_F16_BR +
                 2 * (size_t)H3_CUDA_SDPA_F16_BC_GEN) *
                    head_dim * sizeof(__half) +
                (size_t)H3_CUDA_SDPA_F16_BR * head_dim * sizeof(float);
            if (!smem_attr_set_gen[smem_dev]) {
                status = cudaFuncSetAttribute(
                    h3_cuda_sdpa_flash_f16_generic_kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
                if (status != cudaSuccess) {
                    h3_cuda_set_error_cuda(error, error_size,
                                           "cudaFuncSetAttribute smem", status);
                    goto cleanup;
                }
                smem_attr_set_gen[smem_dev] = 1;
            }
            unsigned q_tiles =
                (unsigned)((seq + (size_t)H3_CUDA_SDPA_F16_BR - 1) /
                           (size_t)H3_CUDA_SDPA_F16_BR);
            dim3 grid((unsigned)heads, q_tiles, 1);
            h3_cuda_sdpa_flash_f16_generic_kernel<<<grid, H3_CUDA_SDPA_F16_BR,
                                                    smem, h3_cuda_op_st(cuda)>>>(
                d_qh, d_kh, d_vh, d_o, (int)seq, (int)heads, (int)head_dim,
                scale, attn_window);
            backend = attn_window > 0 ? "cuda_flash_f16_win" : "cuda_flash_f16";
        }
        status = cudaGetLastError();
        if (status != cudaSuccess) {
            h3_cuda_set_error_cuda(error, error_size, "sdpa f16 kernel",
                                   status);
            goto cleanup;
        }
        if (h3_cuda_want_events(cuda)) {
            status = cudaEventRecord(ev1, 0);
            if (status != cudaSuccess) {
                h3_cuda_set_error_cuda(error, error_size, "cudaEventRecord",
                                       status);
                goto cleanup;
            }
            status = cudaEventSynchronize(ev1);
            if (status != cudaSuccess) {
                h3_cuda_set_error_cuda(error, error_size,
                                       "cudaEventSynchronize", status);
                goto cleanup;
            }
            status = cudaEventElapsedTime(&elapsed, ev0, ev1);
            if (status != cudaSuccess) {
                h3_cuda_set_error_cuda(error, error_size,
                                       "cudaEventElapsedTime", status);
                goto cleanup;
            }
        }
    }
    if (sdpa_ms) *sdpa_ms = (double)elapsed;
    if (backend_out && backend_out_size) {
        if (h3_cuda_dist_attn_active() && !strstr(backend, "dist"))
            snprintf(backend_out, backend_out_size, "%s_dist", backend);
        else
            snprintf(backend_out, backend_out_size, "%s", backend);
    }
    ok = 1;
    goto cleanup;

timed_done:
    if (h3_cuda_want_events(cuda)) {
        status = cudaEventRecord(ev1, 0);
        if (status == cudaSuccess) cudaEventSynchronize(ev1);
        if (sdpa_ms && ev0 && ev1) {
            float e2 = 0.0f;
            if (cudaEventElapsedTime(&e2, ev0, ev1) == cudaSuccess)
                *sdpa_ms = (double)e2;
        }
    }
    if (h3_sol_profile_on() && !cute_timed) {
        double ms = (sdpa_ms && *sdpa_ms > 0.0) ? *sdpa_ms : (double)elapsed;
        if (backend && strstr(backend, "cudnn") && ms > 0.0) {
            h3_sp_add_n(&g_sp_step.sdpa_cudnn, &g_sp_tot.sdpa_cudnn,
                        &g_sp_step.n_cudnn, &g_sp_tot.n_cudnn, ms);
            if (this_dense)
                h3_sp_add_n(&g_sp_step.sdpa_dense, &g_sp_tot.sdpa_dense,
                            &g_sp_step.n_dense, &g_sp_tot.n_dense, ms);
        }
    }
    if (backend_out && backend_out_size) {
        if (h3_cuda_dist_attn_active() && !strstr(backend, "dist"))
            snprintf(backend_out, backend_out_size, "%s_dist", backend);
        else
            snprintf(backend_out, backend_out_size, "%s", backend);
    }

cleanup:
    if (ev0) cudaEventDestroy(ev0);
    if (ev1) cudaEventDestroy(ev1);
    return ok;
}

static int h3_cuda_sdpa_f16_launch_dev(h3_cuda *cuda, const float *d_q,
                                       const float *d_k, const float *d_v,
                                       size_t seq, size_t heads,
                                       size_t head_dim, float scale,
                                       float *d_o, double *sdpa_ms,
                                       char *error, size_t error_size) {
    return h3_cuda_sdpa_f16_launch_dev_ex(cuda, d_q, d_k, d_v, seq, heads,
                                          head_dim, scale, 0, d_o, sdpa_ms,
                                          NULL, 0, error, error_size, 0);
}

int h3_cuda_sdpa_f16_dev_ex(h3_cuda *cuda, const float *d_query,
                            const float *d_key, const float *d_value,
                            size_t seq, size_t heads, size_t head_dim,
                            float scale, int attn_window, float *d_out,
                            double *sdpa_ms, char *backend_out,
                            size_t backend_out_size, char *error,
                            size_t error_size) {
    return h3_cuda_sdpa_f16_launch_dev_ex(
        cuda, d_query, d_key, d_value, seq, heads, head_dim, scale, attn_window,
        d_out, sdpa_ms, backend_out, backend_out_size, error, error_size, 0);
}

int h3_cuda_sdpa_f16_dev(h3_cuda *cuda, const float *d_query,
                         const float *d_key, const float *d_value, size_t seq,
                         size_t heads, size_t head_dim, float scale,
                         float *d_out, double *sdpa_ms, char *error,
                         size_t error_size) {
    return h3_cuda_sdpa_f16_launch_dev(cuda, d_query, d_key, d_value, seq,
                                       heads, head_dim, scale, d_out, sdpa_ms,
                                       error, error_size);
}

int h3_cuda_sdpa_f16_ex(h3_cuda *cuda, const float *query, const float *key,
                        const float *value, size_t seq, size_t heads,
                        size_t head_dim, float scale, int attn_window,
                        float *out, double *sdpa_ms, char *backend_out,
                        size_t backend_out_size, char *error,
                        size_t error_size) {
    float *d_q = NULL, *d_k = NULL, *d_v = NULL, *d_o = NULL;
    cudaError_t status;
    size_t elems;
    int ok = 0;

    if (sdpa_ms) *sdpa_ms = 0.0;
    if (!cuda || !query || !key || !value || !out || seq == 0 || heads == 0 ||
        head_dim == 0) {
        h3_cuda_set_error(error, error_size, "invalid h3_cuda_sdpa_f16_ex args");
        return 0;
    }
    status = cudaSetDevice(cuda->device_index);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaSetDevice", status);
        return 0;
    }
    elems = seq * heads * head_dim;
    if (cudaMalloc((void **)&d_q, elems * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&d_k, elems * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&d_v, elems * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&d_o, elems * sizeof(float)) != cudaSuccess) {
        h3_cuda_set_error(error, error_size, "sdpa f16 staging OOM");
        goto cleanup;
    }
    if (cudaMemcpy(d_q, query, elems * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_k, key, elems * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_v, value, elems * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        h3_cuda_set_error(error, error_size, "sdpa f16 H2D failed");
        goto cleanup;
    }
    if (!h3_cuda_sdpa_f16_launch_dev_ex(cuda, d_q, d_k, d_v, seq, heads,
                                        head_dim, scale, attn_window, d_o,
                                        sdpa_ms, backend_out, backend_out_size,
                                        error, error_size, 0))
        goto cleanup;
    status =
        cudaMemcpy(out, d_o, elems * sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaMemcpy O D2H", status);
        goto cleanup;
    }
    ok = 1;
cleanup:
    if (d_q) cudaFree(d_q);
    if (d_k) cudaFree(d_k);
    if (d_v) cudaFree(d_v);
    if (d_o) cudaFree(d_o);
    return ok;
}

int h3_cuda_sdpa_f16(h3_cuda *cuda, const float *query, const float *key,
                     const float *value, size_t seq, size_t heads,
                     size_t head_dim, float scale, float *out, double *sdpa_ms,
                     char *error, size_t error_size) {
    return h3_cuda_sdpa_f16_ex(cuda, query, key, value, seq, heads, head_dim,
                               scale, 0, out, sdpa_ms, NULL, 0, error,
                               error_size);
}

/* Repeat KV heads for GQA → MHA: in[T,Hkv,D] → out[T,Hq,D], Hq % Hkv == 0. */
__global__ void h3_cuda_repeat_kv_heads_kernel(const float *__restrict__ in,
                                               float *__restrict__ out,
                                               size_t seq, size_t kv_heads,
                                               size_t q_heads,
                                               size_t head_dim) {
    size_t n = seq * q_heads * head_dim;
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= n) return;
    size_t d = i % head_dim;
    size_t tmp = i / head_dim;
    size_t h = tmp % q_heads;
    size_t t = tmp / q_heads;
    size_t rep = q_heads / kv_heads;
    size_t kv_h = h / rep;
    out[i] = in[(t * kv_heads + kv_h) * head_dim + d];
}

/* Naive causal GQA (one thread per query token×head). Fallback if cuDNN
 * causal fused SDPA is unavailable. Layout [T,H,D]. */
__global__ void h3_cuda_gqa_causal_naive_kernel(
    const float *__restrict__ Q, const float *__restrict__ K,
    const float *__restrict__ V, float *__restrict__ O, uint32_t seq,
    uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim, float scale) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = seq * q_heads;
    if (tid >= total) return;
    uint32_t h = tid % q_heads;
    uint32_t t = tid / q_heads;
    uint32_t rep = q_heads / kv_heads;
    uint32_t kv_h = h / rep;
    const float *q = Q + ((size_t)t * q_heads + h) * head_dim;
    float m = -1e30f;
    float l = 0.0f;
    float acc[128];
    uint32_t d, s;
    if (head_dim > 128) return;
    for (d = 0; d < head_dim; d++) acc[d] = 0.0f;
    for (s = 0; s <= t; s++) {
        const float *krow = K + ((size_t)s * kv_heads + kv_h) * head_dim;
        const float *vrow = V + ((size_t)s * kv_heads + kv_h) * head_dim;
        float dot = 0.0f;
        float m_new, alpha, p;
        for (d = 0; d < head_dim; d++)
            dot = fmaf(q[d], krow[d], dot);
        float score = dot * scale;
        m_new = score > m ? score : m;
        alpha = (m > -1e29f) ? expf(m - m_new) : 0.0f;
        p = expf(score - m_new);
        l = alpha * l + p;
        for (d = 0; d < head_dim; d++)
            acc[d] = acc[d] * alpha + p * vrow[d];
        m = m_new;
    }
    {
        float inv = 1.0f / fmaxf(l, 1e-20f);
        float *o = O + ((size_t)t * q_heads + h) * head_dim;
        for (d = 0; d < head_dim; d++) o[d] = acc[d] * inv;
    }
}

int h3_cuda_sdpa_gqa_causal_f16_dev(h3_cuda *cuda, const float *d_query,
                                    const float *d_key, const float *d_value,
                                    size_t seq, size_t q_heads, size_t kv_heads,
                                    size_t head_dim, float scale, float *d_out,
                                    double *sdpa_ms, char *backend_out,
                                    size_t backend_out_size, char *error,
                                    size_t error_size) {
    float *d_k_exp = NULL, *d_v_exp = NULL;
    cudaError_t status;
    size_t q_elems;
    int ok = 0;
    const char *backend = "cuda_gqa_causal_naive";
    unsigned nblk;
    const int thr = 256;

    if (sdpa_ms) *sdpa_ms = 0.0;
    if (backend_out && backend_out_size)
        snprintf(backend_out, backend_out_size, "%s", "unset");
    if (!cuda || !d_query || !d_key || !d_value || !d_out || seq == 0 ||
        q_heads == 0 || kv_heads == 0 || head_dim == 0) {
        h3_cuda_set_error(error, error_size, "invalid gqa causal dev args");
        return 0;
    }
    if (q_heads % kv_heads != 0) {
        h3_cuda_set_error(error, error_size, "GQA head mismatch");
        return 0;
    }
    if (head_dim > 128) {
        h3_cuda_set_error(error, error_size, "gqa causal head_dim>128");
        return 0;
    }
    if (!h3_cuda_bind(cuda, error, error_size)) return 0;
    q_elems = seq * q_heads * head_dim;

#if defined(H3_HAVE_CUDNN) && defined(H3_HAVE_CUDNN_FRONTEND)
    if (cuda->cudnn_ok &&
        !h3_cuda_sdpa_cudnn_shape_blocked(seq, q_heads, head_dim, 1)) {
        size_t padded = h3_cuda_sdpa_cudnn_align_seq(seq);
        size_t P = padded * q_heads * head_dim;
        size_t half_n = P * 4;
        int qkv_pad =
            h3_cuda_sdpa_cudnn_want_padded(seq, q_heads, head_dim, 1);
        size_t stride = qkv_pad ? P : q_elems;
        __half *d_qh, *d_kh, *d_vh;
        unsigned exp_blk =
            (unsigned)((q_elems + (size_t)thr - 1) / (size_t)thr);
        unsigned cvt_blk =
            (unsigned)((q_elems + (size_t)thr - 1) / (size_t)thr);
        if (h3_cuda_ensure_gqa_stage(cuda, 2 * q_elems * sizeof(float), error,
                                     error_size) &&
            h3_cuda_ensure_scratch(cuda, half_n * sizeof(__half), error,
                                   error_size)) {
            d_k_exp = (float *)cuda->d_gqa_stage;
            d_v_exp = d_k_exp + q_elems;
            h3_cuda_repeat_kv_heads_kernel<<<exp_blk, thr>>>(
                d_key, d_k_exp, seq, kv_heads, q_heads, head_dim);
            h3_cuda_repeat_kv_heads_kernel<<<exp_blk, thr>>>(
                d_value, d_v_exp, seq, kv_heads, q_heads, head_dim);
            d_qh = (__half *)cuda->d_sdpa_scratch;
            d_kh = d_qh + stride;
            d_vh = d_kh + stride;
            h3_cuda_f32_to_f16_3_kernel<<<cvt_blk, thr>>>(
                d_query, d_k_exp, d_v_exp, d_qh, d_kh, d_vh, q_elems);
            if (qkv_pad && P > q_elems) {
                h3_cuda_zero_pad_tails(cuda, d_qh + q_elems, d_kh + q_elems,
                                       d_vh + q_elems, P - q_elems);
            }
            if (h3_cuda_sdpa_cudnn_dev(cuda, d_qh, d_kh, d_vh, d_out, seq,
                                       q_heads, head_dim, scale, 1, qkv_pad,
                                       sdpa_ms, error, error_size)) {
                backend = "cuda_cudnn_gqa_causal";
                ok = 1;
                goto done_dev;
            }
            if (error && error[0] && !strstr(error, "cached fail"))
                fprintf(stderr,
                        "[h3] te_gqa cudnn fallback seq=%zu d=%zu: %s\n", seq,
                        head_dim, error);
            if (error && error_size) error[0] = '\0';
        }
    }
#endif

    nblk = (unsigned)((seq * q_heads + (size_t)thr - 1) / (size_t)thr);
    h3_cuda_gqa_causal_naive_kernel<<<nblk, thr>>>(
        d_query, d_key, d_value, d_out, (uint32_t)seq, (uint32_t)q_heads,
        (uint32_t)kv_heads, (uint32_t)head_dim, scale);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "gqa causal naive launch",
                               status);
        goto done_dev;
    }
    backend = "cuda_gqa_causal_naive";
    ok = 1;

done_dev:
    if (ok && backend_out && backend_out_size)
        snprintf(backend_out, backend_out_size, "%s", backend);
    (void)d_k_exp;
    (void)d_v_exp;
    return ok;
}

int h3_cuda_sdpa_gqa_causal_f16(h3_cuda *cuda, const float *query,
                                const float *key, const float *value,
                                size_t seq, size_t q_heads, size_t kv_heads,
                                size_t head_dim, float scale, float *out,
                                double *sdpa_ms, char *backend_out,
                                size_t backend_out_size, char *error,
                                size_t error_size) {
    float *d_q = NULL, *d_k = NULL, *d_v = NULL, *d_k_exp = NULL,
          *d_v_exp = NULL, *d_o = NULL;
    cudaError_t status;
    size_t q_elems, kv_elems;
    int ok = 0;
    const char *backend = "cuda_gqa_causal_naive";
    unsigned nblk;
    const int thr = 256;

    if (sdpa_ms) *sdpa_ms = 0.0;
    if (backend_out && backend_out_size)
        snprintf(backend_out, backend_out_size, "%s", "unset");
    if (!cuda || !query || !key || !value || !out || seq == 0 || q_heads == 0 ||
        kv_heads == 0 || head_dim == 0) {
        h3_cuda_set_error(error, error_size, "invalid gqa causal args");
        return 0;
    }
    if (q_heads % kv_heads != 0) {
        h3_cuda_set_error(error, error_size, "GQA head mismatch");
        return 0;
    }
    if (head_dim > 128) {
        h3_cuda_set_error(error, error_size, "gqa causal head_dim>128");
        return 0;
    }
    status = cudaSetDevice(cuda->device_index);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaSetDevice", status);
        return 0;
    }

    q_elems = seq * q_heads * head_dim;
    kv_elems = seq * kv_heads * head_dim;
    {
        /* [q][o][k][v][k_exp][v_exp] — grow-only, no per-layer malloc. */
        size_t stage_n = 4 * q_elems + 2 * kv_elems;
        if (!h3_cuda_ensure_gqa_stage(cuda, stage_n * sizeof(float), error,
                                      error_size))
            goto cleanup;
        d_q = (float *)cuda->d_gqa_stage;
        d_o = d_q + q_elems;
        d_k = d_o + q_elems;
        d_v = d_k + kv_elems;
        d_k_exp = d_v + kv_elems;
        d_v_exp = d_k_exp + q_elems;
    }
    if (cudaMemcpy(d_q, query, q_elems * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_k, key, kv_elems * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_v, value, kv_elems * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        h3_cuda_set_error(error, error_size, "gqa causal H2D failed");
        goto cleanup;
    }

#if defined(H3_HAVE_CUDNN) && defined(H3_HAVE_CUDNN_FRONTEND)
    if (cuda->cudnn_ok &&
        !h3_cuda_sdpa_cudnn_shape_blocked(seq, q_heads, head_dim, 1)) {
        size_t padded = h3_cuda_sdpa_cudnn_align_seq(seq);
        size_t P = padded * q_heads * head_dim;
        size_t half_n = P * 4;
        int qkv_pad =
            h3_cuda_sdpa_cudnn_want_padded(seq, q_heads, head_dim, 1);
        size_t stride = qkv_pad ? P : q_elems;
        __half *d_qh, *d_kh, *d_vh;
        unsigned exp_blk =
            (unsigned)((q_elems + (size_t)thr - 1) / (size_t)thr);
        unsigned cvt_blk =
            (unsigned)((q_elems + (size_t)thr - 1) / (size_t)thr);
        if (d_k_exp && d_v_exp &&
            h3_cuda_ensure_scratch(cuda, half_n * sizeof(__half), error,
                                   error_size)) {
            h3_cuda_repeat_kv_heads_kernel<<<exp_blk, thr>>>(
                d_k, d_k_exp, seq, kv_heads, q_heads, head_dim);
            h3_cuda_repeat_kv_heads_kernel<<<exp_blk, thr>>>(
                d_v, d_v_exp, seq, kv_heads, q_heads, head_dim);
            d_qh = (__half *)cuda->d_sdpa_scratch;
            d_kh = d_qh + stride;
            d_vh = d_kh + stride;
            h3_cuda_f32_to_f16_3_kernel<<<cvt_blk, thr>>>(
                d_q, d_k_exp, d_v_exp, d_qh, d_kh, d_vh, q_elems);
            if (qkv_pad && P > q_elems) {
                h3_cuda_zero_pad_tails(cuda, d_qh + q_elems, d_kh + q_elems,
                                       d_vh + q_elems, P - q_elems);
            }
            if (h3_cuda_sdpa_cudnn_dev(cuda, d_qh, d_kh, d_vh, d_o, seq,
                                       q_heads, head_dim, scale, 1, qkv_pad,
                                       sdpa_ms, error, error_size)) {
                backend = "cuda_cudnn_gqa_causal";
                ok = 1;
                goto download;
            }
            if (error && error[0] && !strstr(error, "cached fail"))
                fprintf(stderr,
                        "[h3] te_gqa cudnn fallback seq=%zu d=%zu: %s\n", seq,
                        head_dim, error);
            if (error && error_size) error[0] = '\0';
        }
    }
#endif

    nblk = (unsigned)((seq * q_heads + (size_t)thr - 1) / (size_t)thr);
    h3_cuda_gqa_causal_naive_kernel<<<nblk, thr>>>(
        d_q, d_k, d_v, d_o, (uint32_t)seq, (uint32_t)q_heads,
        (uint32_t)kv_heads, (uint32_t)head_dim, scale);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "gqa causal naive launch",
                               status);
        goto cleanup;
    }
    backend = "cuda_gqa_causal_naive";
    ok = 1;

download:
    if (ok) {
        status = cudaMemcpy(out, d_o, q_elems * sizeof(float),
                            cudaMemcpyDeviceToHost);
        if (status != cudaSuccess) {
            h3_cuda_set_error_cuda(error, error_size, "gqa causal D2H",
                                   status);
            ok = 0;
        }
    }
    if (ok && backend_out && backend_out_size)
        snprintf(backend_out, backend_out_size, "%s", backend);
cleanup:
    return ok;
}

/* ---- DiT elementwise CUDA (AdaLN / gate / SwiGLU / QKV-RoPE) ---- */

__global__ void h3_cuda_adaln_kernel(const float *__restrict__ x,
                                     const float *__restrict__ norm_w,
                                     const float *__restrict__ modulation,
                                     const uint32_t *__restrict__ row_map,
                                     float *__restrict__ out, size_t rows,
                                     size_t width, size_t slots,
                                     size_t shift_slot, size_t scale_slot,
                                     float epsilon) {
    size_t r = (size_t)blockIdx.x;
    if (r >= rows) return;
    const float *xr = x + r * width;
    float *yr = out + r * width;
    uint32_t mrow = row_map ? row_map[r] : (uint32_t)r;
    const float *mod_base = modulation + (size_t)mrow * slots * width;
    const float *shift = mod_base + shift_slot * width;
    const float *scale = mod_base + scale_slot * width;

    float sum = 0.0f;
    for (size_t c = threadIdx.x; c < width; c += blockDim.x)
        sum += xr[c] * xr[c];
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    __shared__ float warp_sums[32];
    __shared__ float ssum;
    int warp = (int)(threadIdx.x >> 5);
    int lane = (int)(threadIdx.x & 31);
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        int nwarps = ((int)blockDim.x + 31) >> 5;
        for (int w = 0; w < nwarps; w++) total += warp_sums[w];
        ssum = total;
    }
    __syncthreads();
    float inv = rsqrtf(ssum / (float)width + epsilon);
    for (size_t c = threadIdx.x; c < width; c += blockDim.x) {
        float n = xr[c] * inv * norm_w[c];
        yr[c] = n * (1.0f + scale[c]) + shift[c];
    }
}

__global__ void h3_cuda_gate_kernel(const float *__restrict__ residual,
                                    const float *__restrict__ branch,
                                    const float *__restrict__ modulation,
                                    const uint32_t *__restrict__ row_map,
                                    float *__restrict__ out, size_t rows,
                                    size_t width, size_t slots,
                                    size_t gate_slot) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = rows * width;
    if (idx >= total) return;
    size_t r = idx / width;
    size_t c = idx % width;
    uint32_t mrow = row_map ? row_map[r] : (uint32_t)r;
    float g =
        modulation[(size_t)mrow * slots * width + gate_slot * width + c];
    out[idx] = residual[idx] + branch[idx] * g;
}

__global__ void h3_cuda_swiglu_kernel(const float *__restrict__ fused,
                                      float *__restrict__ out, size_t rows,
                                      size_t width) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = rows * width;
    if (idx >= total) return;
    size_t r = idx / width;
    size_t c = idx % width;
    const float *row = fused + r * (width * 2);
    float gate = row[c];
    float up = row[width + c];
    out[idx] = (gate / (1.0f + expf(-gate))) * up;
}

__global__ void h3_cuda_qkv_rope_kernel(
    const float *__restrict__ qkv, const float *__restrict__ q_norm,
    const float *__restrict__ k_norm, const float *__restrict__ rope_cos,
    const float *__restrict__ rope_sin, float *__restrict__ query,
    float *__restrict__ key, float *__restrict__ value, size_t seq,
    size_t heads, size_t head_dim, size_t rope_half, float epsilon,
    int grouped, int split_half) {
    size_t row = (size_t)blockIdx.y;
    size_t head = (size_t)blockIdx.x;
    if (row >= seq || head >= heads) return;
    size_t inner = heads * head_dim;
    const float *qkv_row = qkv + row * inner * 3;
    const float *c_row = rope_cos + row * rope_half;
    const float *s_row = rope_sin + row * rope_half;
    /* grouped=1: Metal interleaved Q|K|V per head; 0: contiguous [Q|K|V]. */
    const float *q_base;
    const float *k_base;
    const float *v_base;
    if (grouped) {
        q_base = qkv_row + head * head_dim * 3;
        k_base = q_base + head_dim;
        v_base = k_base + head_dim;
    } else {
        q_base = qkv_row + head * head_dim;
        k_base = q_base + inner;
        v_base = q_base + inner * 2;
    }
    float *q_out = query + (row * heads + head) * head_dim;
    float *k_out = key + (row * heads + head) * head_dim;
    float *v_out = value + (row * heads + head) * head_dim;

    float q_sum = 0.0f, k_sum = 0.0f;
    for (size_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        q_sum += q_base[d] * q_base[d];
        k_sum += k_base[d] * k_base[d];
        v_out[d] = v_base[d];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        q_sum += __shfl_down_sync(0xffffffffu, q_sum, offset);
        k_sum += __shfl_down_sync(0xffffffffu, k_sum, offset);
    }
    __shared__ float qk[2];
    __shared__ float wq[8], wk[8];
    int warp = (int)(threadIdx.x >> 5);
    int lane = (int)(threadIdx.x & 31);
    if (lane == 0) {
        wq[warp] = q_sum;
        wk[warp] = k_sum;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float tq = 0.0f, tk = 0.0f;
        int nwarps = ((int)blockDim.x + 31) >> 5;
        for (int w = 0; w < nwarps; w++) {
            tq += wq[w];
            tk += wk[w];
        }
        qk[0] = tq;
        qk[1] = tk;
    }
    __syncthreads();
    float q_inv = rsqrtf(qk[0] / (float)head_dim + epsilon);
    float k_inv = rsqrtf(qk[1] / (float)head_dim + epsilon);

    if (rope_half > 0 && split_half) {
        /* Comfy apply_rope_split_half: pair d with d+rope_half. */
        for (size_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float q0 = q_base[d] * q_inv * q_norm[d];
            float k0 = k_base[d] * k_inv * k_norm[d];
            if (d < rope_half) {
                size_t pair = d + rope_half;
                float q1 = q_base[pair] * q_inv * q_norm[pair];
                float k1 = k_base[pair] * k_inv * k_norm[pair];
                float c = c_row[d];
                float s = s_row[d];
                q0 = q0 * c - q1 * s;
                k0 = k0 * c - k1 * s;
            } else if (d < rope_half * 2) {
                size_t pair = d - rope_half;
                float q1 = q_base[pair] * q_inv * q_norm[pair];
                float k1 = k_base[pair] * k_inv * k_norm[pair];
                float c = c_row[pair];
                float s = s_row[pair];
                q0 = q0 * c + q1 * s;
                k0 = k0 * c + k1 * s;
            }
            q_out[d] = q0;
            k_out[d] = k0;
        }
    } else if (rope_half > 0) {
        /* Consecutive-pair rotate (legacy). */
        for (size_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            q_out[d] = q_base[d] * q_inv * q_norm[d];
            k_out[d] = k_base[d] * k_inv * k_norm[d];
        }
        __syncthreads();
        for (size_t p = threadIdx.x; p < rope_half; p += blockDim.x) {
            size_t i0 = p * 2u;
            size_t i1 = i0 + 1u;
            float c = c_row[p];
            float s = s_row[p];
            float q0 = q_out[i0], q1 = q_out[i1];
            float k0 = k_out[i0], k1 = k_out[i1];
            q_out[i0] = q0 * c - q1 * s;
            q_out[i1] = q0 * s + q1 * c;
            k_out[i0] = k0 * c - k1 * s;
            k_out[i1] = k0 * s + k1 * c;
        }
    } else {
        for (size_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            q_out[d] = q_base[d] * q_inv * q_norm[d];
            k_out[d] = k_base[d] * k_inv * k_norm[d];
        }
    }
}

/* Same math as f32 rope, writes FP16 Q/K/V (feeds SDPA without cast). */
__global__ void h3_cuda_qkv_rope_f16_kernel(
    const float *__restrict__ qkv, const float *__restrict__ q_norm,
    const float *__restrict__ k_norm, const float *__restrict__ rope_cos,
    const float *__restrict__ rope_sin, __half *__restrict__ query,
    __half *__restrict__ key, __half *__restrict__ value, size_t seq,
    size_t seq_padded, size_t heads, size_t head_dim, size_t rope_half,
    float epsilon, int grouped, int split_half) {
    size_t row = (size_t)blockIdx.y;
    size_t head = (size_t)blockIdx.x;
    if (head >= heads) return;
    if (seq_padded < seq) seq_padded = seq;
    if (row >= seq_padded) return;
    __half *q_out = query + (row * heads + head) * head_dim;
    __half *k_out = key + (row * heads + head) * head_dim;
    __half *v_out = value + (row * heads + head) * head_dim;
    if (row >= seq) {
        for (size_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            q_out[d] = __float2half(0.0f);
            k_out[d] = __float2half(0.0f);
            v_out[d] = __float2half(0.0f);
        }
        return;
    }
    size_t inner = heads * head_dim;
    const float *qkv_row = qkv + row * inner * 3;
    const float *c_row = rope_cos + row * rope_half;
    const float *s_row = rope_sin + row * rope_half;
    const float *q_base;
    const float *k_base;
    const float *v_base;
    if (grouped) {
        q_base = qkv_row + head * head_dim * 3;
        k_base = q_base + head_dim;
        v_base = k_base + head_dim;
    } else {
        q_base = qkv_row + head * head_dim;
        k_base = q_base + inner;
        v_base = q_base + inner * 2;
    }
    __shared__ float sq[128], sk[128];

    float q_sum = 0.0f, k_sum = 0.0f;
    for (size_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        q_sum += q_base[d] * q_base[d];
        k_sum += k_base[d] * k_base[d];
        v_out[d] = __float2half(v_base[d]);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        q_sum += __shfl_down_sync(0xffffffffu, q_sum, offset);
        k_sum += __shfl_down_sync(0xffffffffu, k_sum, offset);
    }
    __shared__ float qk[2];
    __shared__ float wq[8], wk[8];
    int warp = (int)(threadIdx.x >> 5);
    int lane = (int)(threadIdx.x & 31);
    if (lane == 0) {
        wq[warp] = q_sum;
        wk[warp] = k_sum;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float tq = 0.0f, tk = 0.0f;
        int nwarps = ((int)blockDim.x + 31) >> 5;
        for (int w = 0; w < nwarps; w++) {
            tq += wq[w];
            tk += wk[w];
        }
        qk[0] = tq;
        qk[1] = tk;
    }
    __syncthreads();
    float q_inv = rsqrtf(qk[0] / (float)head_dim + epsilon);
    float k_inv = rsqrtf(qk[1] / (float)head_dim + epsilon);

    if (rope_half > 0 && split_half) {
        /* Comfy apply_rope_split_half: pair d with d+rope_half. */
        for (size_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float q0 = q_base[d] * q_inv * q_norm[d];
            float k0 = k_base[d] * k_inv * k_norm[d];
            if (d < rope_half) {
                size_t pair = d + rope_half;
                float q1 = q_base[pair] * q_inv * q_norm[pair];
                float k1 = k_base[pair] * k_inv * k_norm[pair];
                float c = c_row[d];
                float s = s_row[d];
                q0 = q0 * c - q1 * s;
                k0 = k0 * c - k1 * s;
            } else if (d < rope_half * 2) {
                size_t pair = d - rope_half;
                float q1 = q_base[pair] * q_inv * q_norm[pair];
                float k1 = k_base[pair] * k_inv * k_norm[pair];
                float c = c_row[pair];
                float s = s_row[pair];
                q0 = q0 * c + q1 * s;
                k0 = k0 * c + k1 * s;
            }
            q_out[d] = __float2half(q0);
            k_out[d] = __float2half(k0);
        }
    } else if (rope_half > 0) {
        /* Consecutive-pair rotate (legacy). */
        for (size_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            sq[d] = q_base[d] * q_inv * q_norm[d];
            sk[d] = k_base[d] * k_inv * k_norm[d];
        }
        __syncthreads();
        for (size_t p = threadIdx.x; p < rope_half; p += blockDim.x) {
            size_t i0 = p * 2u;
            size_t i1 = i0 + 1u;
            float c = c_row[p];
            float s = s_row[p];
            float q0 = sq[i0], q1 = sq[i1];
            float k0 = sk[i0], k1 = sk[i1];
            sq[i0] = q0 * c - q1 * s;
            sq[i1] = q0 * s + q1 * c;
            sk[i0] = k0 * c - k1 * s;
            sk[i1] = k0 * s + k1 * c;
        }
        __syncthreads();
        for (size_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            q_out[d] = __float2half(sq[d]);
            k_out[d] = __float2half(sk[d]);
        }
    } else {
        for (size_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            q_out[d] = __float2half(q_base[d] * q_inv * q_norm[d]);
            k_out[d] = __float2half(k_base[d] * k_inv * k_norm[d]);
        }
    }
}

static int h3_cuda_op_begin(h3_cuda *cuda, cudaEvent_t *ev0, cudaEvent_t *ev1,
                            char *error, size_t error_size) {
    cudaError_t status;
    *ev0 = NULL;
    *ev1 = NULL;
    /* cudaSetDevice is illegal during stream capture. Quiet used to skip
     * bind (U=1 already on the handle). U>1 AdaLN/gate then launched on
     * the previous rank's current device. bind is GetDevice no-op on U=1. */
    if (!cuda->graph_capture && !h3_cuda_bind(cuda, error, error_size))
        return 0;
    if (!h3_cuda_want_events(cuda)) {
        return 1; /* quiet: no events */
    }
    status = cudaEventCreate(ev0);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaEventCreate", status);
        return 0;
    }
    status = cudaEventCreate(ev1);
    if (status != cudaSuccess) {
        cudaEventDestroy(*ev0);
        *ev0 = NULL;
        h3_cuda_set_error_cuda(error, error_size, "cudaEventCreate", status);
        return 0;
    }
    return 1;
}

static int h3_cuda_op_end(cudaEvent_t ev0, cudaEvent_t ev1, double *op_ms,
                          char *error, size_t error_size) {
    cudaError_t status;
    float elapsed = 0.0f;
    if (!ev0 || !ev1) {
        if (op_ms) *op_ms = 0.0;
        return 1; /* quiet / graph-capture mode */
    }
    status = cudaEventRecord(ev1, 0);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaEventRecord", status);
        return 0;
    }
    status = cudaEventSynchronize(ev1);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaEventSynchronize",
                               status);
        return 0;
    }
    status = cudaEventElapsedTime(&elapsed, ev0, ev1);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaEventElapsedTime",
                               status);
        return 0;
    }
    if (op_ms) *op_ms = (double)elapsed;
    return 1;
}

int h3_cuda_adaln_f32(h3_cuda *cuda, const float *x, const float *norm_w,
                      const float *modulation, const uint32_t *row_map,
                      size_t rows, size_t width, size_t slots, size_t mod_rows,
                      size_t shift_slot, size_t scale_slot, float epsilon,
                      float *out, double *op_ms, char *error,
                      size_t error_size) {
    float *d_x = NULL, *d_nw = NULL, *d_mod = NULL, *d_out = NULL;
    uint32_t *d_map = NULL;
    cudaEvent_t ev0 = NULL, ev1 = NULL;
    cudaError_t status;
    size_t mod_elems;
    int ok = 0;
    if (op_ms) *op_ms = 0.0;
    if (!cuda || !x || !norm_w || !modulation || !out || rows == 0 ||
        width == 0 || slots == 0 || mod_rows == 0) {
        h3_cuda_set_error(error, error_size, "invalid adaln args");
        return 0;
    }
    mod_elems = mod_rows * slots * width;
    if (!h3_cuda_op_begin(cuda, &ev0, &ev1, error, error_size)) return 0;

    status = cudaMalloc((void **)&d_x, rows * width * sizeof(float));
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaMalloc x", status);
        goto done;
    }
    status = cudaMalloc((void **)&d_nw, width * sizeof(float));
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaMalloc nw", status);
        goto done;
    }
    status = cudaMalloc((void **)&d_mod, mod_elems * sizeof(float));
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaMalloc mod", status);
        goto done;
    }
    status = cudaMalloc((void **)&d_out, rows * width * sizeof(float));
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaMalloc out", status);
        goto done;
    }
    if (row_map) {
        status = cudaMalloc((void **)&d_map, rows * sizeof(uint32_t));
        if (status != cudaSuccess) {
            h3_cuda_set_error_cuda(error, error_size, "cudaMalloc map", status);
            goto done;
        }
        status = cudaMemcpy(d_map, row_map, rows * sizeof(uint32_t),
                            cudaMemcpyHostToDevice);
        if (status != cudaSuccess) {
            h3_cuda_set_error_cuda(error, error_size, "H2D map", status);
            goto done;
        }
    }
    if (cudaMemcpy(d_x, x, rows * width * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_nw, norm_w, width * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_mod, modulation, mod_elems * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        h3_cuda_set_error(error, error_size, "adaln H2D failed");
        goto done;
    }
    status = cudaEventRecord(ev0, 0);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaEventRecord", status);
        goto done;
    }
    h3_cuda_adaln_kernel<<<(unsigned)rows, 256>>>(
        d_x, d_nw, d_mod, d_map, d_out, rows, width, slots, shift_slot,
        scale_slot, epsilon);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "adaln kernel", status);
        goto done;
    }
    if (!h3_cuda_op_end(ev0, ev1, op_ms, error, error_size)) goto done;
    status = cudaMemcpy(out, d_out, rows * width * sizeof(float),
                        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "adaln D2H", status);
        goto done;
    }
    ok = 1;
done:
    if (ev0) cudaEventDestroy(ev0);
    if (ev1) cudaEventDestroy(ev1);
    if (d_x) cudaFree(d_x);
    if (d_nw) cudaFree(d_nw);
    if (d_mod) cudaFree(d_mod);
    if (d_out) cudaFree(d_out);
    if (d_map) cudaFree(d_map);
    return ok;
}

int h3_cuda_gate_f32(h3_cuda *cuda, const float *residual, const float *branch,
                     const float *modulation, const uint32_t *row_map,
                     size_t rows, size_t width, size_t slots, size_t mod_rows,
                     size_t gate_slot, float *out, double *op_ms, char *error,
                     size_t error_size) {
    float *d_res = NULL, *d_br = NULL, *d_mod = NULL, *d_out = NULL;
    uint32_t *d_map = NULL;
    cudaEvent_t ev0 = NULL, ev1 = NULL;
    cudaError_t status;
    size_t mod_elems, total;
    int ok = 0;
    if (op_ms) *op_ms = 0.0;
    if (!cuda || !residual || !branch || !modulation || !out || rows == 0 ||
        width == 0 || mod_rows == 0) {
        h3_cuda_set_error(error, error_size, "invalid gate args");
        return 0;
    }
    mod_elems = mod_rows * slots * width;
    total = rows * width;
    if (!h3_cuda_op_begin(cuda, &ev0, &ev1, error, error_size)) return 0;
    if (cudaMalloc((void **)&d_res, total * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&d_br, total * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&d_mod, mod_elems * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&d_out, total * sizeof(float)) != cudaSuccess) {
        h3_cuda_set_error(error, error_size, "gate OOM");
        goto done;
    }
    if (row_map) {
        if (cudaMalloc((void **)&d_map, rows * sizeof(uint32_t)) !=
                cudaSuccess ||
            cudaMemcpy(d_map, row_map, rows * sizeof(uint32_t),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            h3_cuda_set_error(error, error_size, "gate map H2D");
            goto done;
        }
    }
    if (cudaMemcpy(d_res, residual, total * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_br, branch, total * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_mod, modulation, mod_elems * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        h3_cuda_set_error(error, error_size, "gate H2D");
        goto done;
    }
    status = cudaEventRecord(ev0, 0);
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "cudaEventRecord", status);
        goto done;
    }
    {
        unsigned nblk = (unsigned)((total + 255) / 256);
        h3_cuda_gate_kernel<<<nblk, 256>>>(d_res, d_br, d_mod, d_map, d_out,
                                           rows, width, slots, gate_slot);
    }
    if (cudaGetLastError() != cudaSuccess) {
        h3_cuda_set_error(error, error_size, "gate kernel");
        goto done;
    }
    if (!h3_cuda_op_end(ev0, ev1, op_ms, error, error_size)) goto done;
    if (cudaMemcpy(out, d_out, total * sizeof(float), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        h3_cuda_set_error(error, error_size, "gate D2H");
        goto done;
    }
    ok = 1;
done:
    if (ev0) cudaEventDestroy(ev0);
    if (ev1) cudaEventDestroy(ev1);
    if (d_res) cudaFree(d_res);
    if (d_br) cudaFree(d_br);
    if (d_mod) cudaFree(d_mod);
    if (d_out) cudaFree(d_out);
    if (d_map) cudaFree(d_map);
    return ok;
}

int h3_cuda_swiglu_f32(h3_cuda *cuda, const float *fused, size_t rows,
                       size_t width, float *out, double *op_ms, char *error,
                       size_t error_size) {
    float *d_f = NULL, *d_o = NULL;
    cudaEvent_t ev0 = NULL, ev1 = NULL;
    size_t total, fused_n;
    int ok = 0;
    if (op_ms) *op_ms = 0.0;
    if (!cuda || !fused || !out || rows == 0 || width == 0) {
        h3_cuda_set_error(error, error_size, "invalid swiglu args");
        return 0;
    }
    total = rows * width;
    fused_n = rows * width * 2;
    if (!h3_cuda_op_begin(cuda, &ev0, &ev1, error, error_size)) return 0;
    if (cudaMalloc((void **)&d_f, fused_n * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&d_o, total * sizeof(float)) != cudaSuccess ||
        cudaMemcpy(d_f, fused, fused_n * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        h3_cuda_set_error(error, error_size, "swiglu alloc/H2D");
        goto done;
    }
    if (cudaEventRecord(ev0, 0) != cudaSuccess) {
        h3_cuda_set_error(error, error_size, "swiglu event");
        goto done;
    }
    {
        unsigned nblk = (unsigned)((total + 255) / 256);
        h3_cuda_swiglu_kernel<<<nblk, 256>>>(d_f, d_o, rows, width);
    }
    if (cudaGetLastError() != cudaSuccess ||
        !h3_cuda_op_end(ev0, ev1, op_ms, error, error_size) ||
        cudaMemcpy(out, d_o, total * sizeof(float), cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
        h3_cuda_set_error(error, error_size, "swiglu kernel/D2H");
        goto done;
    }
    ok = 1;
done:
    if (ev0) cudaEventDestroy(ev0);
    if (ev1) cudaEventDestroy(ev1);
    if (d_f) cudaFree(d_f);
    if (d_o) cudaFree(d_o);
    return ok;
}

int h3_cuda_qkv_rope_f32(h3_cuda *cuda, const float *qkv, const float *q_norm,
                         const float *k_norm, const float *rope_cos,
                         const float *rope_sin, size_t seq, size_t heads,
                         size_t head_dim, size_t rope_half, float epsilon,
                         float *query, float *key, float *value, double *op_ms,
                         char *error, size_t error_size) {
    float *d_qkv = NULL, *d_qn = NULL, *d_kn = NULL, *d_cos = NULL, *d_sin = NULL;
    float *d_q = NULL, *d_k = NULL, *d_v = NULL;
    cudaEvent_t ev0 = NULL, ev1 = NULL;
    size_t inner, qkv_n, rope_n;
    int ok = 0;
    if (op_ms) *op_ms = 0.0;
    if (!cuda || !qkv || !q_norm || !k_norm || !rope_cos || !rope_sin ||
        !query || !key || !value || seq == 0 || heads == 0 || head_dim == 0) {
        h3_cuda_set_error(error, error_size, "invalid qkv_rope args");
        return 0;
    }
    if (rope_half * 2 > head_dim) {
        h3_cuda_set_error(error, error_size, "rope_half too large");
        return 0;
    }
    inner = heads * head_dim;
    qkv_n = seq * inner * 3;
    rope_n = seq * rope_half;
    if (!h3_cuda_op_begin(cuda, &ev0, &ev1, error, error_size)) return 0;
    if (cudaMalloc((void **)&d_qkv, qkv_n * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&d_qn, head_dim * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&d_kn, head_dim * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&d_cos, rope_n * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&d_sin, rope_n * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&d_q, seq * inner * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&d_k, seq * inner * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&d_v, seq * inner * sizeof(float)) != cudaSuccess) {
        h3_cuda_set_error(error, error_size, "qkv_rope OOM");
        goto done;
    }
    if (cudaMemcpy(d_qkv, qkv, qkv_n * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_qn, q_norm, head_dim * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_kn, k_norm, head_dim * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_cos, rope_cos, rope_n * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_sin, rope_sin, rope_n * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        h3_cuda_set_error(error, error_size, "qkv_rope H2D");
        goto done;
    }
    if (cudaEventRecord(ev0, 0) != cudaSuccess) {
        h3_cuda_set_error(error, error_size, "qkv_rope event");
        goto done;
    }
    {
        dim3 grid((unsigned)heads, (unsigned)seq, 1);
        int grouped = h3_cuda_dit_qkv_grouped();
        int split_half = h3_cuda_dit_rope_split_half();
        h3_cuda_qkv_rope_kernel<<<grid, 128>>>(
            d_qkv, d_qn, d_kn, d_cos, d_sin, d_q, d_k, d_v, seq, heads,
            head_dim, rope_half, epsilon, grouped, split_half);
    }
    if (cudaGetLastError() != cudaSuccess ||
        !h3_cuda_op_end(ev0, ev1, op_ms, error, error_size)) {
        h3_cuda_set_error(error, error_size, "qkv_rope kernel");
        goto done;
    }
    if (cudaMemcpy(query, d_q, seq * inner * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(key, d_k, seq * inner * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(value, d_v, seq * inner * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        h3_cuda_set_error(error, error_size, "qkv_rope D2H");
        goto done;
    }
    ok = 1;
done:
    if (ev0) cudaEventDestroy(ev0);
    if (ev1) cudaEventDestroy(ev1);
    if (d_qkv) cudaFree(d_qkv);
    if (d_qn) cudaFree(d_qn);
    if (d_kn) cudaFree(d_kn);
    if (d_cos) cudaFree(d_cos);
    if (d_sin) cudaFree(d_sin);
    if (d_q) cudaFree(d_q);
    if (d_k) cudaFree(d_k);
    if (d_v) cudaFree(d_v);
    return ok;
}

/* ---- Device-pointer DiT ops (no H2D/D2H) ---- */

int h3_cuda_adaln_f32_dev(h3_cuda *cuda, const float *d_x, const float *d_norm_w,
                          const float *d_modulation, const uint32_t *d_row_map,
                          size_t rows, size_t width, size_t slots,
                          size_t shift_slot, size_t scale_slot, float epsilon,
                          float *d_out, double *op_ms, char *error,
                          size_t error_size) {
    cudaEvent_t ev0 = NULL, ev1 = NULL;
    cudaError_t status;
    if (op_ms) *op_ms = 0.0;
    if (!cuda || !d_x || !d_norm_w || !d_modulation || !d_out || rows == 0 ||
        width == 0 || slots == 0) {
        h3_cuda_set_error(error, error_size, "invalid adaln_dev args");
        return 0;
    }
    if (!h3_cuda_op_begin(cuda, &ev0, &ev1, error, error_size)) return 0;
    if (ev0) {
        status = cudaEventRecord(ev0, 0);
        if (status != cudaSuccess) {
            h3_cuda_set_error_cuda(error, error_size, "cudaEventRecord",
                                   status);
            cudaEventDestroy(ev0);
            cudaEventDestroy(ev1);
            return 0;
        }
    }
    h3_cuda_adaln_kernel<<<(unsigned)rows, 256, 0, h3_cuda_op_st(cuda)>>>(
        d_x, d_norm_w, d_modulation, d_row_map, d_out, rows, width, slots,
        shift_slot, scale_slot, epsilon);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        h3_cuda_set_error_cuda(error, error_size, "adaln_dev kernel", status);
        if (ev0) cudaEventDestroy(ev0);
        if (ev1) cudaEventDestroy(ev1);
        return 0;
    }
    if (!h3_cuda_op_end(ev0, ev1, op_ms, error, error_size)) {
        if (ev0) cudaEventDestroy(ev0);
        if (ev1) cudaEventDestroy(ev1);
        return 0;
    }
    if (op_ms)
        h3_sp_add_n(&g_sp_step.adaln, &g_sp_tot.adaln, &g_sp_step.n_adaln,
                    &g_sp_tot.n_adaln, *op_ms);
    if (ev0) cudaEventDestroy(ev0);
    if (ev1) cudaEventDestroy(ev1);
    return 1;
}

int h3_cuda_gate_f32_dev(h3_cuda *cuda, const float *d_residual,
                         const float *d_branch, const float *d_modulation,
                         const uint32_t *d_row_map, size_t rows, size_t width,
                         size_t slots, size_t gate_slot, float *d_out,
                         double *op_ms, char *error, size_t error_size) {
    cudaEvent_t ev0 = NULL, ev1 = NULL;
    size_t total;
    if (op_ms) *op_ms = 0.0;
    if (!cuda || !d_residual || !d_branch || !d_modulation || !d_out ||
        rows == 0 || width == 0) {
        h3_cuda_set_error(error, error_size, "invalid gate_dev args");
        return 0;
    }
    total = rows * width;
    if (!h3_cuda_op_begin(cuda, &ev0, &ev1, error, error_size)) return 0;
    if (ev0 && cudaEventRecord(ev0, 0) != cudaSuccess) {
        h3_cuda_set_error(error, error_size, "gate_dev event");
        if (ev0) cudaEventDestroy(ev0);
        if (ev1) cudaEventDestroy(ev1);
        return 0;
    }
    {
        unsigned nblk = (unsigned)((total + 255) / 256);
        h3_cuda_gate_kernel<<<nblk, 256, 0, h3_cuda_op_st(cuda)>>>(
            d_residual, d_branch, d_modulation, d_row_map, d_out, rows, width,
            slots, gate_slot);
    }
    if (cudaGetLastError() != cudaSuccess ||
        !h3_cuda_op_end(ev0, ev1, op_ms, error, error_size)) {
        h3_cuda_set_error(error, error_size, "gate_dev kernel");
        if (ev0) cudaEventDestroy(ev0);
        if (ev1) cudaEventDestroy(ev1);
        return 0;
    }
    if (op_ms) h3_sp_add(&g_sp_step.gate, &g_sp_tot.gate, *op_ms);
    if (ev0) cudaEventDestroy(ev0);
    if (ev1) cudaEventDestroy(ev1);
    return 1;
}

int h3_cuda_swiglu_f32_dev(h3_cuda *cuda, const float *d_fused, size_t rows,
                           size_t width, float *d_out, double *op_ms,
                           char *error, size_t error_size) {
    cudaEvent_t ev0 = NULL, ev1 = NULL;
    size_t total;
    if (op_ms) *op_ms = 0.0;
    if (!cuda || !d_fused || !d_out || rows == 0 || width == 0) {
        h3_cuda_set_error(error, error_size, "invalid swiglu_dev args");
        return 0;
    }
    total = rows * width;
    if (!h3_cuda_op_begin(cuda, &ev0, &ev1, error, error_size)) return 0;
    if (ev0 && cudaEventRecord(ev0, 0) != cudaSuccess) {
        if (ev0) cudaEventDestroy(ev0);
        if (ev1) cudaEventDestroy(ev1);
        return 0;
    }
    {
        unsigned nblk = (unsigned)((total + 255) / 256);
        h3_cuda_swiglu_kernel<<<nblk, 256, 0, h3_cuda_op_st(cuda)>>>(
            d_fused, d_out, rows, width);
    }
    if (cudaGetLastError() != cudaSuccess ||
        !h3_cuda_op_end(ev0, ev1, op_ms, error, error_size)) {
        h3_cuda_set_error(error, error_size, "swiglu_dev kernel");
        if (ev0) cudaEventDestroy(ev0);
        if (ev1) cudaEventDestroy(ev1);
        return 0;
    }
    if (op_ms) h3_sp_add(&g_sp_step.swiglu, &g_sp_tot.swiglu, *op_ms);
    if (ev0) cudaEventDestroy(ev0);
    if (ev1) cudaEventDestroy(ev1);
    return 1;
}

int h3_cuda_ffn_swiglu_i8_dev(
    h3_cuda *cuda, const h3_cuda_w4a8_weight *w_fc2, const float *d_fused,
    float *d_act, int convrot_groupsize, size_t m, size_t ffn, float *d_y,
    double *swiglu_ms, double *prep_ms, double *gemm_ms, char *backend_out,
    size_t backend_out_size, char *error, size_t error_size) {
    if (swiglu_ms) *swiglu_ms = 0.0;
    if (!h3_cuda_swiglu_f32_dev(cuda, d_fused, m, ffn, d_act, swiglu_ms, error,
                                error_size))
        return 0;
    return h3_cuda_gemm_i8_dequant_f32_w_act_dev(
        cuda, w_fc2, d_act, convrot_groupsize, m, d_y, prep_ms, gemm_ms,
        backend_out, backend_out_size, error, error_size);
}

int h3_cuda_qkv_rope_f32_dev(h3_cuda *cuda, const float *d_qkv,
                             const float *d_q_norm, const float *d_k_norm,
                             const float *d_rope_cos, const float *d_rope_sin,
                             size_t seq, size_t heads, size_t head_dim,
                             size_t rope_half, float epsilon, float *d_query,
                             float *d_key, float *d_value, double *op_ms,
                             char *error, size_t error_size) {
    cudaEvent_t ev0 = NULL, ev1 = NULL;
    if (op_ms) *op_ms = 0.0;
    if (!cuda || !d_qkv || !d_q_norm || !d_k_norm || !d_rope_cos ||
        !d_rope_sin || !d_query || !d_key || !d_value || seq == 0 ||
        heads == 0 || head_dim == 0) {
        h3_cuda_set_error(error, error_size, "invalid qkv_rope_dev args");
        return 0;
    }
    if (rope_half * 2 > head_dim) {
        h3_cuda_set_error(error, error_size, "rope_half too large");
        return 0;
    }
    if (!h3_cuda_op_begin(cuda, &ev0, &ev1, error, error_size)) return 0;
    if (ev0 && cudaEventRecord(ev0, 0) != cudaSuccess) {
        if (ev0) cudaEventDestroy(ev0);
        if (ev1) cudaEventDestroy(ev1);
        return 0;
    }
    {
        dim3 grid((unsigned)heads, (unsigned)seq, 1);
        int grouped = h3_cuda_dit_qkv_grouped();
        int split_half = h3_cuda_dit_rope_split_half();
        h3_cuda_qkv_rope_kernel<<<grid, 128>>>(
            d_qkv, d_q_norm, d_k_norm, d_rope_cos, d_rope_sin, d_query, d_key,
            d_value, seq, heads, head_dim, rope_half, epsilon, grouped,
            split_half);
    }
    if (cudaGetLastError() != cudaSuccess ||
        !h3_cuda_op_end(ev0, ev1, op_ms, error, error_size)) {
        h3_cuda_set_error(error, error_size, "qkv_rope_dev kernel");
        if (ev0) cudaEventDestroy(ev0);
        if (ev1) cudaEventDestroy(ev1);
        return 0;
    }
    if (ev0) cudaEventDestroy(ev0);
    if (ev1) cudaEventDestroy(ev1);
    return 1;
}

__global__ void h3_cuda_qkv_rope_split_half_kernel(
    const float *__restrict__ qkv, const float *__restrict__ q_norm,
    const float *__restrict__ k_norm, const float *__restrict__ rope_cos,
    const float *__restrict__ rope_sin, float *__restrict__ query,
    float *__restrict__ key, float *__restrict__ value, size_t seq,
    size_t heads, size_t head_dim, size_t rope_half, float epsilon) {
    size_t row = (size_t)blockIdx.y;
    size_t head = (size_t)blockIdx.x;
    if (row >= seq || head >= heads) return;
    size_t inner = heads * head_dim;
    const float *qkv_row = qkv + row * inner * 3;
    const float *c_row = rope_cos + row * rope_half;
    const float *s_row = rope_sin + row * rope_half;
    /* Contiguous [Q|K|V] (Comfy); matches split-half RoPE path. */
    const float *q_base = qkv_row + head * head_dim;
    const float *k_base = q_base + inner;
    const float *v_base = q_base + inner * 2;
    float *q_out = query + (row * heads + head) * head_dim;
    float *k_out = key + (row * heads + head) * head_dim;
    float *v_out = value + (row * heads + head) * head_dim;

    float q_sum = 0.0f, k_sum = 0.0f;
    for (size_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        q_sum += q_base[d] * q_base[d];
        k_sum += k_base[d] * k_base[d];
        v_out[d] = v_base[d];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        q_sum += __shfl_down_sync(0xffffffffu, q_sum, offset);
        k_sum += __shfl_down_sync(0xffffffffu, k_sum, offset);
    }
    __shared__ float qk[2];
    __shared__ float wq[8], wk[8];
    int warp = (int)(threadIdx.x >> 5);
    int lane = (int)(threadIdx.x & 31);
    if (lane == 0) {
        wq[warp] = q_sum;
        wk[warp] = k_sum;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float tq = 0.0f, tk = 0.0f;
        int nwarps = ((int)blockDim.x + 31) >> 5;
        for (int w = 0; w < nwarps; w++) {
            tq += wq[w];
            tk += wk[w];
        }
        qk[0] = tq;
        qk[1] = tk;
    }
    __syncthreads();
    float q_inv = rsqrtf(qk[0] / (float)head_dim + epsilon);
    float k_inv = rsqrtf(qk[1] / (float)head_dim + epsilon);

    for (size_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float q0 = q_base[d] * q_inv * q_norm[d];
        float k0 = k_base[d] * k_inv * k_norm[d];
        if (rope_half > 0 && d < rope_half) {
            size_t pair = d + rope_half;
            float q1 = q_base[pair] * q_inv * q_norm[pair];
            float k1 = k_base[pair] * k_inv * k_norm[pair];
            float c = c_row[d];
            float s = s_row[d];
            q0 = q0 * c - q1 * s;
            k0 = k0 * c - k1 * s;
        } else if (rope_half > 0 && d < rope_half * 2) {
            size_t pair = d - rope_half;
            float q1 = q_base[pair] * q_inv * q_norm[pair];
            float k1 = k_base[pair] * k_inv * k_norm[pair];
            float c = c_row[pair];
            float s = s_row[pair];
            q0 = q0 * c + q1 * s;
            k0 = k0 * c + k1 * s;
        }
        q_out[d] = q0;
        k_out[d] = k0;
    }
}

int h3_cuda_qkv_rope_split_half_f32_dev(
    h3_cuda *cuda, const float *d_qkv, const float *d_q_norm,
    const float *d_k_norm, const float *d_rope_cos, const float *d_rope_sin,
    size_t seq, size_t heads, size_t head_dim, size_t rope_half, float epsilon,
    float *d_query, float *d_key, float *d_value, double *op_ms, char *error,
    size_t error_size) {
    cudaEvent_t ev0 = NULL, ev1 = NULL;
    if (op_ms) *op_ms = 0.0;
    if (!cuda || !d_qkv || !d_q_norm || !d_k_norm || !d_rope_cos ||
        !d_rope_sin || !d_query || !d_key || !d_value || seq == 0 ||
        heads == 0 || head_dim == 0) {
        h3_cuda_set_error(error, error_size, "invalid qkv_rope_split_half args");
        return 0;
    }
    if (rope_half * 2 > head_dim) {
        h3_cuda_set_error(error, error_size, "rope_half too large");
        return 0;
    }
    if (!h3_cuda_op_begin(cuda, &ev0, &ev1, error, error_size)) return 0;
    if (ev0 && cudaEventRecord(ev0, 0) != cudaSuccess) {
        if (ev0) cudaEventDestroy(ev0);
        if (ev1) cudaEventDestroy(ev1);
        return 0;
    }
    {
        dim3 grid((unsigned)heads, (unsigned)seq, 1);
        h3_cuda_qkv_rope_split_half_kernel<<<grid, 128>>>(
            d_qkv, d_q_norm, d_k_norm, d_rope_cos, d_rope_sin, d_query, d_key,
            d_value, seq, heads, head_dim, rope_half, epsilon);
    }
    if (cudaGetLastError() != cudaSuccess ||
        !h3_cuda_op_end(ev0, ev1, op_ms, error, error_size)) {
        h3_cuda_set_error(error, error_size, "qkv_rope_split_half kernel");
        if (ev0) cudaEventDestroy(ev0);
        if (ev1) cudaEventDestroy(ev1);
        return 0;
    }
    if (ev0) cudaEventDestroy(ev0);
    if (ev1) cudaEventDestroy(ev1);
    return 1;
}

/* RoPE → FP16 scratch + SDPA (skips f32 QKV materialize + cast). */
int h3_cuda_qkv_rope_sdpa_f16_dev_layout(
    h3_cuda *cuda, const float *d_qkv, const float *d_q_norm,
    const float *d_k_norm, const float *d_rope_cos, const float *d_rope_sin,
    size_t seq, size_t heads, size_t head_dim, size_t rope_half, float epsilon,
    float scale, int attn_window, int grouped, int split_half,
    float *d_attn_out, double *rope_ms, double *sdpa_ms, char *backend_out,
    size_t backend_out_size, char *error, size_t error_size) {
    __half *d_qh, *d_kh, *d_vh;
    size_t elems;
    size_t seq_padded = seq;
    cudaError_t status;

    if (rope_ms) *rope_ms = 0.0;
    if (sdpa_ms) *sdpa_ms = 0.0;
    if (!cuda || !d_qkv || !d_q_norm || !d_k_norm || !d_rope_cos ||
        !d_rope_sin || !d_attn_out || seq == 0 || heads == 0 || head_dim == 0) {
        h3_cuda_set_error(error, error_size, "invalid qkv_rope_sdpa args");
        return 0;
    }
    if (rope_half * 2 > head_dim) {
        h3_cuda_set_error(error, error_size, "rope_half too large");
        return 0;
    }
    elems = seq * heads * head_dim;
    {
        size_t half_n = elems * 3;
        size_t stride = elems;
#if defined(H3_HAVE_CUDNN_FRONTEND)
        if (h3_cuda_sdpa_want_cudnn()) {
            size_t padded = h3_cuda_sdpa_cudnn_align_seq(seq);
            size_t pe = padded * heads * head_dim;
            half_n = pe * 4;
            if (h3_cuda_sdpa_cudnn_want_padded(seq, heads, head_dim, 0)) {
                stride = pe;
                seq_padded = padded;
            }
        }
#endif
        if (!h3_cuda_ensure_scratch(cuda, half_n * sizeof(__half), error,
                                    error_size))
            return 0;
        d_qh = (__half *)cuda->d_sdpa_scratch;
        d_kh = d_qh + stride;
        d_vh = d_kh + stride;
    }

    {
        dim3 grid((unsigned)heads, (unsigned)seq_padded, 1);
        cudaEvent_t rev0 = NULL, rev1 = NULL;
        if (h3_sol_profile_on()) {
            cudaEventCreate(&rev0);
            cudaEventCreate(&rev1);
            if (rev0) cudaEventRecord(rev0, 0);
        }
        h3_cuda_qkv_rope_f16_kernel<<<grid, 128, 0, h3_cuda_op_st(cuda)>>>(
            d_qkv, d_q_norm, d_k_norm, d_rope_cos, d_rope_sin, d_qh, d_kh, d_vh,
            seq, seq_padded, heads, head_dim, rope_half, epsilon, grouped,
            split_half);
        status = cudaGetLastError();
        if (status != cudaSuccess) {
            if (rev0) cudaEventDestroy(rev0);
            if (rev1) cudaEventDestroy(rev1);
            h3_cuda_set_error_cuda(error, error_size, "qkv_rope_f16", status);
            return 0;
        }
        if (rev0 && rev1) {
            float rms = 0.0f;
            cudaEventRecord(rev1, 0);
            cudaEventSynchronize(rev1);
            if (cudaEventElapsedTime(&rms, rev0, rev1) == cudaSuccess) {
                if (rope_ms) *rope_ms = (double)rms;
                h3_sp_add_n(&g_sp_step.rope, &g_sp_tot.rope, &g_sp_step.n_rope,
                            &g_sp_tot.n_rope, (double)rms);
            }
            cudaEventDestroy(rev0);
            cudaEventDestroy(rev1);
        }
    }

    return h3_cuda_sdpa_f16_launch_dev_ex(
        cuda, NULL, NULL, NULL, seq, heads, head_dim, scale, attn_window,
        d_attn_out, sdpa_ms, backend_out, backend_out_size, error, error_size,
        1);
}

int h3_cuda_qkv_rope_sdpa_f16_dev(
    h3_cuda *cuda, const float *d_qkv, const float *d_q_norm,
    const float *d_k_norm, const float *d_rope_cos, const float *d_rope_sin,
    size_t seq, size_t heads, size_t head_dim, size_t rope_half, float epsilon,
    float scale, int attn_window, float *d_attn_out, double *rope_ms,
    double *sdpa_ms, char *backend_out, size_t backend_out_size, char *error,
    size_t error_size) {
    return h3_cuda_qkv_rope_sdpa_f16_dev_layout(
        cuda, d_qkv, d_q_norm, d_k_norm, d_rope_cos, d_rope_sin, seq, heads,
        head_dim, rope_half, epsilon, scale, attn_window,
        h3_cuda_dit_qkv_grouped(), h3_cuda_dit_rope_split_half(), d_attn_out,
        rope_ms, sdpa_ms, backend_out, backend_out_size, error, error_size);
}

/* ---- VAE / generic device helpers ---- */

__global__ void h3_cuda_rms_norm_kernel(const float *__restrict__ x,
                                        const float *__restrict__ w,
                                        float *__restrict__ out, size_t rows,
                                        size_t width, float eps) {
    size_t r = (size_t)blockIdx.x;
    if (r >= rows) return;
    const float *xr = x + r * width;
    float *yr = out + r * width;
    float sum = 0.0f;
    for (size_t c = threadIdx.x; c < width; c += blockDim.x)
        sum += xr[c] * xr[c];
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    __shared__ float warp_sums[32];
    __shared__ float ssum;
    int warp = (int)(threadIdx.x >> 5);
    int lane = (int)(threadIdx.x & 31);
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        int nwarps = ((int)blockDim.x + 31) >> 5;
        for (int i = 0; i < nwarps; i++) total += warp_sums[i];
        ssum = total;
    }
    __syncthreads();
    float inv = rsqrtf(ssum / (float)width + eps);
    for (size_t c = threadIdx.x; c < width; c += blockDim.x)
        yr[c] = xr[c] * inv * w[c];
}

__global__ void h3_cuda_layer_norm_bias_kernel(const float *__restrict__ x,
                                               const float *__restrict__ w,
                                               const float *__restrict__ b,
                                               float *__restrict__ out,
                                               size_t rows, size_t width,
                                               float eps) {
    size_t r = (size_t)blockIdx.x;
    if (r >= rows) return;
    const float *xr = x + r * width;
    float *yr = out + r * width;
    float sum = 0.0f, sumsq = 0.0f;
    for (size_t c = threadIdx.x; c < width; c += blockDim.x) {
        float v = xr[c];
        sum += v;
        sumsq += v * v;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
        sumsq += __shfl_down_sync(0xffffffffu, sumsq, offset);
    }
    __shared__ float warp_sum[32], warp_sq[32];
    __shared__ float ssum, ssumsq;
    int warp = (int)(threadIdx.x >> 5);
    int lane = (int)(threadIdx.x & 31);
    if (lane == 0) {
        warp_sum[warp] = sum;
        warp_sq[warp] = sumsq;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float ts = 0.0f, tq = 0.0f;
        int nwarps = ((int)blockDim.x + 31) >> 5;
        for (int i = 0; i < nwarps; i++) {
            ts += warp_sum[i];
            tq += warp_sq[i];
        }
        ssum = ts;
        ssumsq = tq;
    }
    __syncthreads();
    float mean = ssum / (float)width;
    float var = ssumsq / (float)width - mean * mean;
    if (var < 0.0f) var = 0.0f;
    float inv = rsqrtf(var + eps);
    for (size_t c = threadIdx.x; c < width; c += blockDim.x)
        yr[c] = (xr[c] - mean) * inv * w[c] + b[c];
}

__global__ void h3_cuda_silu_mul_kernel(const float *__restrict__ fused,
                                        float *__restrict__ out, size_t rows,
                                        size_t width) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = rows * width;
    if (idx >= total) return;
    size_t r = idx / width;
    size_t c = idx % width;
    float g = fused[r * width * 2 + c];
    float u = fused[r * width * 2 + width + c];
    out[idx] = (g / (1.0f + expf(-g))) * u;
}

__global__ void h3_cuda_residual_scale_kernel(float *__restrict__ x,
                                              const float *__restrict__ branch,
                                              const float *__restrict__ scale,
                                              size_t rows, size_t width) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = rows * width;
    if (idx >= total) return;
    size_t c = idx % width;
    x[idx] += branch[idx] * scale[c];
}

__global__ void h3_cuda_bias_add_kernel(float *__restrict__ y,
                                        const float *__restrict__ bias,
                                        size_t rows, size_t width) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = rows * width;
    if (idx >= total) return;
    y[idx] += bias[idx % width];
}

__global__ void h3_cuda_qkv_thirds_kernel(const float *__restrict__ qkv,
                                          float *__restrict__ q,
                                          float *__restrict__ k,
                                          float *__restrict__ v, size_t seq,
                                          size_t heads, size_t head_dim) {
    size_t inner = heads * head_dim;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = seq * inner;
    if (idx >= total) return;
    size_t s = idx / inner;
    size_t within = idx % inner;
    size_t h = within / head_dim;
    size_t d = within % head_dim;
    const float *row = qkv + s * inner * 3;
    size_t dst = (s * heads + h) * head_dim + d;
    q[dst] = row[h * head_dim + d];
    k[dst] = row[inner + h * head_dim + d];
    v[dst] = row[2 * inner + h * head_dim + d];
}

int h3_cuda_rms_norm_f32_dev(h3_cuda *cuda, const float *d_x, const float *d_w,
                             size_t rows, size_t width, float eps, float *d_out,
                             char *error, size_t error_size) {
    if (!cuda || !d_x || !d_w || !d_out || rows == 0 || width == 0) {
        h3_cuda_set_error(error, error_size, "invalid rms_norm_dev");
        return 0;
    }
    h3_cuda_rms_norm_kernel<<<(unsigned)rows, 256>>>(d_x, d_w, d_out, rows,
                                                     width, eps);
    if (cudaGetLastError() != cudaSuccess) {
        h3_cuda_set_error(error, error_size, "rms_norm_dev kernel");
        return 0;
    }
    return 1;
}

int h3_cuda_layer_norm_bias_f32_dev(h3_cuda *cuda, const float *d_x,
                                    const float *d_w, const float *d_b,
                                    size_t rows, size_t width, float eps,
                                    float *d_out, char *error,
                                    size_t error_size) {
    if (!cuda || !d_x || !d_w || !d_b || !d_out || rows == 0 || width == 0) {
        h3_cuda_set_error(error, error_size, "invalid layer_norm_dev");
        return 0;
    }
    h3_cuda_layer_norm_bias_kernel<<<(unsigned)rows, 256>>>(d_x, d_w, d_b, d_out,
                                                            rows, width, eps);
    if (cudaGetLastError() != cudaSuccess) {
        h3_cuda_set_error(error, error_size, "layer_norm_dev kernel");
        return 0;
    }
    return 1;
}

int h3_cuda_silu_mul_f32_dev(h3_cuda *cuda, const float *d_fused, size_t rows,
                             size_t width, float *d_out, char *error,
                             size_t error_size) {
    size_t total;
    if (!cuda || !d_fused || !d_out || rows == 0 || width == 0) {
        h3_cuda_set_error(error, error_size, "invalid silu_mul_dev");
        return 0;
    }
    total = rows * width;
    h3_cuda_silu_mul_kernel<<<(unsigned)((total + 255) / 256), 256>>>(
        d_fused, d_out, rows, width);
    if (cudaGetLastError() != cudaSuccess) {
        h3_cuda_set_error(error, error_size, "silu_mul_dev kernel");
        return 0;
    }
    return 1;
}

int h3_cuda_residual_scale_f32_dev(h3_cuda *cuda, float *d_x,
                                   const float *d_branch, const float *d_scale,
                                   size_t rows, size_t width, char *error,
                                   size_t error_size) {
    size_t total;
    if (!cuda || !d_x || !d_branch || !d_scale || rows == 0 || width == 0) {
        h3_cuda_set_error(error, error_size, "invalid residual_scale_dev");
        return 0;
    }
    total = rows * width;
    h3_cuda_residual_scale_kernel<<<(unsigned)((total + 255) / 256), 256>>>(
        d_x, d_branch, d_scale, rows, width);
    if (cudaGetLastError() != cudaSuccess) {
        h3_cuda_set_error(error, error_size, "residual_scale_dev kernel");
        return 0;
    }
    return 1;
}

int h3_cuda_qkv_thirds_to_heads_f32_dev(h3_cuda *cuda, const float *d_qkv,
                                        size_t seq, size_t heads,
                                        size_t head_dim, float *d_q, float *d_k,
                                        float *d_v, char *error,
                                        size_t error_size) {
    size_t total;
    if (!cuda || !d_qkv || !d_q || !d_k || !d_v || seq == 0 || heads == 0 ||
        head_dim == 0) {
        h3_cuda_set_error(error, error_size, "invalid qkv_thirds_dev");
        return 0;
    }
    total = seq * heads * head_dim;
    h3_cuda_qkv_thirds_kernel<<<(unsigned)((total + 255) / 256), 256>>>(
        d_qkv, d_q, d_k, d_v, seq, heads, head_dim);
    if (cudaGetLastError() != cudaSuccess) {
        h3_cuda_set_error(error, error_size, "qkv_thirds_dev kernel");
        return 0;
    }
    return 1;
}

int h3_cuda_bias_add_f32_dev(h3_cuda *cuda, float *d_y, const float *d_bias,
                             size_t rows, size_t width, char *error,
                             size_t error_size) {
    size_t total;
    if (!cuda || !d_y || !d_bias || rows == 0 || width == 0) {
        h3_cuda_set_error(error, error_size, "invalid bias_add_dev");
        return 0;
    }
    total = rows * width;
    h3_cuda_bias_add_kernel<<<(unsigned)((total + 255) / 256), 256>>>(
        d_y, d_bias, rows, width);
    if (cudaGetLastError() != cudaSuccess) {
        h3_cuda_set_error(error, error_size, "bias_add_dev kernel");
        return 0;
    }
    return 1;
}

__global__ void h3_cuda_dequant_i8_rowscale_kernel(
    const int8_t *__restrict__ w_i8, const float *__restrict__ scale,
    float *__restrict__ out, size_t n, size_t k) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = n * k;
    if (idx >= total) return;
    out[idx] = (float)w_i8[idx] * scale[idx / k];
}

extern "C" int h3_cuda_dequant_i8_rowscale_f32(const int8_t *d_i8,
                                               const float *d_scale, size_t n,
                                               size_t k, float *d_out,
                                               void *stream, char *error,
                                               size_t error_size) {
    size_t total;
    const int thr = 256;
    unsigned nblk;
    cudaStream_t s = stream ? (cudaStream_t)stream : 0;
    if (!d_i8 || !d_scale || !d_out || n == 0 || k == 0) {
        h3_cuda_set_error(error, error_size, "invalid i8 rowscale dequant");
        return 0;
    }
    total = n * k;
    nblk = (unsigned)((total + (size_t)thr - 1) / (size_t)thr);
    h3_cuda_dequant_i8_rowscale_kernel<<<nblk, thr, 0, s>>>(d_i8, d_scale, d_out,
                                                            n, k);
    if (cudaGetLastError() != cudaSuccess) {
        h3_cuda_set_error(error, error_size, "i8 rowscale dequant kernel");
        return 0;
    }
    return 1;
}

extern "C" int h3_cuda_f32_to_f16_dev(h3_cuda *cuda, const float *d_in,
                                      void *d_out_half, size_t n, char *error,
                                      size_t error_size) {
    if (!cuda || !d_in || !d_out_half || n == 0) {
        h3_cuda_set_error(error, error_size, "invalid f32_to_f16_dev");
        return 0;
    }
    {
        const int thr = 256;
        const unsigned nblk = (unsigned)((n + (size_t)thr - 1) / (size_t)thr);
        h3_cuda_f32_to_f16_kernel<<<nblk, thr>>>(d_in, (__half *)d_out_half, n);
    }
    if (cudaGetLastError() != cudaSuccess) {
        h3_cuda_set_error(error, error_size, "f32_to_f16_dev kernel");
        return 0;
    }
    return 1;
}

extern "C" int h3_cuda_f16_to_f32_dev(h3_cuda *cuda, const void *d_in_half,
                                      float *d_out, size_t n, char *error,
                                      size_t error_size) {
    /* cuda handle is unused (launch is default-stream). NULL is OK so VAE
     * weights_load can convert native F16→F32 without a generate handle. */
    (void)cuda;
    if (!d_in_half || !d_out || n == 0) {
        h3_cuda_set_error(error, error_size, "invalid f16_to_f32_dev");
        return 0;
    }
    (void)cudaGetLastError(); /* drain stale hostreg / memcpy status */
    h3_cuda_launch_f16_to_f32(d_in_half, d_out, n);
    {
        cudaError_t st = cudaGetLastError();
        if (st != cudaSuccess) {
            h3_cuda_set_error_cuda(error, error_size, "f16_to_f32_dev kernel",
                                   st);
            return 0;
        }
    }
    return 1;
}

/* ---- Token-reduction pool / expand (Metal pair + delta semantics, F32) ---- */

__global__ void h3_cuda_token_pool_f32_kernel(
    const float *__restrict__ input, const uint32_t *__restrict__ pairs,
    const uint32_t *__restrict__ baseline_idx, float *__restrict__ output,
    float *__restrict__ original, float *__restrict__ baseline,
    size_t reduced_rows, size_t width) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t total = reduced_rows * width;
    size_t row, col;
    uint32_t first, second, bi;
    float a, pooled;
    if (idx >= total) return;
    row = idx / width;
    col = idx % width;
    first = pairs[row * 2u];
    second = pairs[row * 2u + 1u];
    a = input[(size_t)first * width + col];
    original[(size_t)first * width + col] = a;
    pooled = a;
    if (first != second) {
        float b = input[(size_t)second * width + col];
        original[(size_t)second * width + col] = b;
        pooled = 0.5f * (a + b);
    }
    output[row * width + col] = pooled;
    bi = baseline_idx[row];
    if (bi != 0xffffffffu && baseline)
        baseline[(size_t)bi * width + col] = pooled;
}

__global__ void h3_cuda_token_expand_f32_kernel(
    const float *__restrict__ original, const float *__restrict__ reduced,
    const float *__restrict__ baseline, const uint32_t *__restrict__ baseline_idx,
    const uint32_t *__restrict__ parents, float *__restrict__ output,
    size_t full_rows, size_t width, uint32_t video_start, uint32_t video_rows,
    float update_scale) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t total = full_rows * width;
    size_t row, col;
    uint32_t parent, bi;
    float red;
    if (idx >= total) return;
    row = idx / width;
    col = idx % width;
    parent = parents[row];
    red = reduced[(size_t)parent * width + col];
    if (row < (size_t)video_start ||
        row >= (size_t)video_start + (size_t)video_rows ||
        (bi = baseline_idx[parent]) == 0xffffffffu) {
        output[idx] = red;
    } else {
        float orig = original[row * width + col];
        float base = baseline[(size_t)bi * width + col];
        output[idx] = orig + update_scale * (red - base);
    }
}

int h3_cuda_token_pool_f32(h3_cuda *cuda, const float *input, size_t full_rows,
                           const uint32_t *pairs, const uint32_t *baseline_idx,
                           size_t reduced_rows, size_t baseline_rows,
                           size_t width, float *output, float **d_original_out,
                           float **d_baseline_out, double *op_ms, char *error,
                           size_t error_size) {
    float *d_in = NULL, *d_out = NULL, *d_orig = NULL, *d_base = NULL;
    uint32_t *d_pairs = NULL, *d_bidx = NULL;
    cudaEvent_t ev0 = NULL, ev1 = NULL;
    size_t full_n, red_n, base_n;
    unsigned nblk;
    int ok = 0;
    if (op_ms) *op_ms = 0.0;
    if (d_original_out) *d_original_out = NULL;
    if (d_baseline_out) *d_baseline_out = NULL;
    if (!cuda || !input || !pairs || !baseline_idx || !output || !d_original_out ||
        !d_baseline_out || full_rows == 0 || reduced_rows == 0 || width == 0) {
        h3_cuda_set_error(error, error_size, "invalid token_pool args");
        return 0;
    }
    full_n = full_rows * width;
    red_n = reduced_rows * width;
    base_n = baseline_rows * width;
    if (!h3_cuda_op_begin(cuda, &ev0, &ev1, error, error_size)) return 0;
    if (cudaMalloc((void **)&d_in, full_n * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&d_out, red_n * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&d_orig, full_n * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&d_pairs, reduced_rows * 2u * sizeof(uint32_t)) !=
            cudaSuccess ||
        cudaMalloc((void **)&d_bidx, reduced_rows * sizeof(uint32_t)) !=
            cudaSuccess) {
        h3_cuda_set_error(error, error_size, "token_pool OOM");
        goto done;
    }
    if (baseline_rows > 0) {
        if (cudaMalloc((void **)&d_base, base_n * sizeof(float)) !=
            cudaSuccess) {
            h3_cuda_set_error(error, error_size, "token_pool baseline OOM");
            goto done;
        }
    }
    if (cudaMemcpy(d_in, input, full_n * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_pairs, pairs, reduced_rows * 2u * sizeof(uint32_t),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_bidx, baseline_idx, reduced_rows * sizeof(uint32_t),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        h3_cuda_set_error(error, error_size, "token_pool H2D");
        goto done;
    }
    if (ev0 && cudaEventRecord(ev0, 0) != cudaSuccess) {
        h3_cuda_set_error(error, error_size, "token_pool event");
        goto done;
    }
    nblk = (unsigned)((red_n + 255u) / 256u);
    h3_cuda_token_pool_f32_kernel<<<nblk, 256>>>(
        d_in, d_pairs, d_bidx, d_out, d_orig, d_base, reduced_rows, width);
    if (cudaGetLastError() != cudaSuccess) {
        h3_cuda_set_error(error, error_size, "token_pool kernel");
        goto done;
    }
    if (!h3_cuda_op_end(ev0, ev1, op_ms, error, error_size)) goto done;
    if (cudaMemcpy(output, d_out, red_n * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        h3_cuda_set_error(error, error_size, "token_pool D2H");
        goto done;
    }
    *d_original_out = d_orig;
    d_orig = NULL;
    *d_baseline_out = d_base;
    d_base = NULL;
    ok = 1;
done:
    if (ev0) cudaEventDestroy(ev0);
    if (ev1) cudaEventDestroy(ev1);
    if (d_in) cudaFree(d_in);
    if (d_out) cudaFree(d_out);
    if (d_orig) cudaFree(d_orig);
    if (d_base) cudaFree(d_base);
    if (d_pairs) cudaFree(d_pairs);
    if (d_bidx) cudaFree(d_bidx);
    if (!ok && d_original_out && *d_original_out) {
        cudaFree(*d_original_out);
        *d_original_out = NULL;
    }
    if (!ok && d_baseline_out && *d_baseline_out) {
        cudaFree(*d_baseline_out);
        *d_baseline_out = NULL;
    }
    return ok;
}

int h3_cuda_token_expand_f32(h3_cuda *cuda, const float *d_original,
                             const float *reduced, const float *d_baseline,
                             const uint32_t *baseline_idx,
                             const uint32_t *parents, size_t full_rows,
                             size_t reduced_rows, size_t baseline_rows,
                             size_t width, uint32_t video_start,
                             uint32_t video_rows, float update_scale,
                             float *output, double *op_ms, char *error,
                             size_t error_size) {
    float *d_red = NULL, *d_out = NULL;
    uint32_t *d_bidx = NULL, *d_parents = NULL;
    cudaEvent_t ev0 = NULL, ev1 = NULL;
    size_t full_n, red_n;
    unsigned nblk;
    int ok = 0;
    if (op_ms) *op_ms = 0.0;
    if (!cuda || !d_original || !reduced || !baseline_idx || !parents ||
        !output || full_rows == 0 || reduced_rows == 0 || width == 0) {
        h3_cuda_set_error(error, error_size, "invalid token_expand args");
        return 0;
    }
    if (baseline_rows > 0 && !d_baseline) {
        h3_cuda_set_error(error, error_size, "token_expand missing baseline");
        return 0;
    }
    full_n = full_rows * width;
    red_n = reduced_rows * width;
    if (!h3_cuda_op_begin(cuda, &ev0, &ev1, error, error_size)) return 0;
    if (cudaMalloc((void **)&d_red, red_n * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&d_out, full_n * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&d_bidx, reduced_rows * sizeof(uint32_t)) !=
            cudaSuccess ||
        cudaMalloc((void **)&d_parents, full_rows * sizeof(uint32_t)) !=
            cudaSuccess) {
        h3_cuda_set_error(error, error_size, "token_expand OOM");
        goto done;
    }
    if (cudaMemcpy(d_red, reduced, red_n * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_bidx, baseline_idx, reduced_rows * sizeof(uint32_t),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_parents, parents, full_rows * sizeof(uint32_t),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        h3_cuda_set_error(error, error_size, "token_expand H2D");
        goto done;
    }
    if (ev0 && cudaEventRecord(ev0, 0) != cudaSuccess) {
        h3_cuda_set_error(error, error_size, "token_expand event");
        goto done;
    }
    nblk = (unsigned)((full_n + 255u) / 256u);
    h3_cuda_token_expand_f32_kernel<<<nblk, 256>>>(
        d_original, d_red, d_baseline, d_bidx, d_parents, d_out, full_rows,
        width, video_start, video_rows, update_scale);
    if (cudaGetLastError() != cudaSuccess) {
        h3_cuda_set_error(error, error_size, "token_expand kernel");
        goto done;
    }
    if (!h3_cuda_op_end(ev0, ev1, op_ms, error, error_size)) goto done;
    if (cudaMemcpy(output, d_out, full_n * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        h3_cuda_set_error(error, error_size, "token_expand D2H");
        goto done;
    }
    ok = 1;
done:
    if (ev0) cudaEventDestroy(ev0);
    if (ev1) cudaEventDestroy(ev1);
    if (d_red) cudaFree(d_red);
    if (d_out) cudaFree(d_out);
    if (d_bidx) cudaFree(d_bidx);
    if (d_parents) cudaFree(d_parents);
    return ok;
}

int h3_cuda_token_pool_f32_dev(h3_cuda *cuda, const float *d_input,
                               size_t full_rows, const uint32_t *pairs,
                               const uint32_t *baseline_idx, size_t reduced_rows,
                               size_t baseline_rows, size_t width,
                               float *d_output, float *d_original,
                               float *d_baseline, double *op_ms, char *error,
                               size_t error_size) {
    uint32_t *d_pairs = NULL, *d_bidx = NULL;
    cudaEvent_t ev0 = NULL, ev1 = NULL;
    size_t red_n;
    unsigned nblk;
    int ok = 0;
    if (op_ms) *op_ms = 0.0;
    if (!cuda || !d_input || !pairs || !baseline_idx || !d_output ||
        !d_original || full_rows == 0 || reduced_rows == 0 || width == 0) {
        h3_cuda_set_error(error, error_size, "invalid token_pool_dev args");
        return 0;
    }
    if (baseline_rows > 0 && !d_baseline) {
        h3_cuda_set_error(error, error_size, "token_pool_dev missing baseline");
        return 0;
    }
    red_n = reduced_rows * width;
    if (!h3_cuda_op_begin(cuda, &ev0, &ev1, error, error_size)) return 0;
    if (cudaMalloc((void **)&d_pairs, reduced_rows * 2u * sizeof(uint32_t)) !=
            cudaSuccess ||
        cudaMalloc((void **)&d_bidx, reduced_rows * sizeof(uint32_t)) !=
            cudaSuccess) {
        h3_cuda_set_error(error, error_size, "token_pool_dev OOM");
        goto done;
    }
    if (cudaMemcpy(d_pairs, pairs, reduced_rows * 2u * sizeof(uint32_t),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_bidx, baseline_idx, reduced_rows * sizeof(uint32_t),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        h3_cuda_set_error(error, error_size, "token_pool_dev idx H2D");
        goto done;
    }
    if (ev0 && cudaEventRecord(ev0, 0) != cudaSuccess) {
        h3_cuda_set_error(error, error_size, "token_pool_dev event");
        goto done;
    }
    nblk = (unsigned)((red_n + 255u) / 256u);
    h3_cuda_token_pool_f32_kernel<<<nblk, 256>>>(
        d_input, d_pairs, d_bidx, d_output, d_original, d_baseline, reduced_rows,
        width);
    if (cudaGetLastError() != cudaSuccess) {
        h3_cuda_set_error(error, error_size, "token_pool_dev kernel");
        goto done;
    }
    if (!h3_cuda_op_end(ev0, ev1, op_ms, error, error_size)) goto done;
    ok = 1;
done:
    if (ev0) cudaEventDestroy(ev0);
    if (ev1) cudaEventDestroy(ev1);
    if (d_pairs) cudaFree(d_pairs);
    if (d_bidx) cudaFree(d_bidx);
    return ok;
}

int h3_cuda_token_expand_f32_dev(
    h3_cuda *cuda, const float *d_original, const float *d_reduced,
    const float *d_baseline, const uint32_t *baseline_idx,
    const uint32_t *parents, size_t full_rows, size_t reduced_rows,
    size_t baseline_rows, size_t width, uint32_t video_start,
    uint32_t video_rows, float update_scale, float *d_output, double *op_ms,
    char *error, size_t error_size) {
    uint32_t *d_bidx = NULL, *d_parents = NULL;
    cudaEvent_t ev0 = NULL, ev1 = NULL;
    size_t full_n;
    unsigned nblk;
    int ok = 0;
    if (op_ms) *op_ms = 0.0;
    if (!cuda || !d_original || !d_reduced || !baseline_idx || !parents ||
        !d_output || full_rows == 0 || reduced_rows == 0 || width == 0) {
        h3_cuda_set_error(error, error_size, "invalid token_expand_dev args");
        return 0;
    }
    if (baseline_rows > 0 && !d_baseline) {
        h3_cuda_set_error(error, error_size, "token_expand_dev missing baseline");
        return 0;
    }
    full_n = full_rows * width;
    if (!h3_cuda_op_begin(cuda, &ev0, &ev1, error, error_size)) return 0;
    if (cudaMalloc((void **)&d_bidx, reduced_rows * sizeof(uint32_t)) !=
            cudaSuccess ||
        cudaMalloc((void **)&d_parents, full_rows * sizeof(uint32_t)) !=
            cudaSuccess) {
        h3_cuda_set_error(error, error_size, "token_expand_dev OOM");
        goto done;
    }
    if (cudaMemcpy(d_bidx, baseline_idx, reduced_rows * sizeof(uint32_t),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_parents, parents, full_rows * sizeof(uint32_t),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        h3_cuda_set_error(error, error_size, "token_expand_dev idx H2D");
        goto done;
    }
    if (ev0 && cudaEventRecord(ev0, 0) != cudaSuccess) {
        h3_cuda_set_error(error, error_size, "token_expand_dev event");
        goto done;
    }
    nblk = (unsigned)((full_n + 255u) / 256u);
    h3_cuda_token_expand_f32_kernel<<<nblk, 256>>>(
        d_original, d_reduced, d_baseline, d_bidx, d_parents, d_output, full_rows,
        width, video_start, video_rows, update_scale);
    if (cudaGetLastError() != cudaSuccess) {
        h3_cuda_set_error(error, error_size, "token_expand_dev kernel");
        goto done;
    }
    if (!h3_cuda_op_end(ev0, ev1, op_ms, error, error_size)) goto done;
    ok = 1;
done:
    if (ev0) cudaEventDestroy(ev0);
    if (ev1) cudaEventDestroy(ev1);
    if (d_bidx) cudaFree(d_bidx);
    if (d_parents) cudaFree(d_parents);
    return ok;
}
