#include "h3_vae_prewarm.h"
#include "h3_vae_int8_cuda.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

static double prewarm_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

void *h3_vae_prewarm_th(void *arg) {
    h3_vae_prewarm_job *j = (h3_vae_prewarm_job *)arg;
    double t0;
    if (!j || !j->path || !j->store) return NULL;
    t0 = prewarm_now_ms();
    j->err[0] = '\0';
    if (!(j->opened && *j->opened)) {
        if (!h3_vae_int8_store_open(j->path, j->store, j->err, sizeof(j->err))) {
            j->ok = 0;
            j->ms = prewarm_now_ms() - t0;
            return NULL;
        }
        if (j->opened) *j->opened = 1;
    }
    if (j->store->map && j->store->map_len > 0) {
#ifdef MADV_WILLNEED
        (void)madvise((void *)j->store->map, j->store->map_len, MADV_WILLNEED);
#endif
    }
    if (j->want_hostreg && j->device >= 0) {
        if (cudaSetDevice(j->device) != cudaSuccess) {
            (void)cudaGetLastError();
            snprintf(j->err, sizeof(j->err), "cudaSetDevice(%d) failed",
                     j->device);
        } else if (h3_vae_int8_cuda_hostreg_prepare(j->store, j->err,
                                                    sizeof(j->err))) {
            j->hostreg_ok = 1;
        }
    }
    if (j->want_dev && j->device >= 0 && j->store && j->n_blocks > 0 &&
        !h3_vae_int8_cuda_device_resident()) {
        h3_vae_int8_cuda_weights *w = NULL;
        double lms = 0.0;
        if (j->device >= 0 && cudaSetDevice(j->device) != cudaSuccess)
            (void)cudaGetLastError();
        if (h3_vae_int8_cuda_weights_load(j->store, j->n_blocks, &w, &lms,
                                          j->err, sizeof(j->err))) {
            j->device_ok = 1;
            j->load_ms = lms;
        } else {
            j->device_ok = 0;
            j->load_ms = lms;
        }
    } else if (h3_vae_int8_cuda_device_resident()) {
        j->device_ok = 1;
        j->load_ms = 0.0;
    }
    j->ok = 1;
    j->ms = prewarm_now_ms() - t0;
    return NULL;
}
