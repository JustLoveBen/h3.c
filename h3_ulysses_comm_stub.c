#include "h3_ulysses_comm.h"

#include <stdio.h>
#include <string.h>

int h3_ulysses_comm_init(h3_ulysses_cfg *cfg, const int *device_ids, int n_dev,
                         char *err, size_t err_size) {
    (void)device_ids;
    if (cfg) {
        cfg->comm = NULL;
        cfg->world_size = n_dev < 1 ? 1 : n_dev;
    }
    if (n_dev <= 1)
        return 1;
    if (err && err_size)
        snprintf(err, err_size, "NCCL not linked; --ulysses > 1 unavailable");
    return 0;
}

void h3_ulysses_comm_destroy(h3_ulysses_cfg *cfg) {
    if (cfg)
        cfg->comm = NULL;
}

int h3_ulysses_comm_set_compute(h3_ulysses_cfg *cfg, int rank, void *stream,
                                char *err, size_t err_size) {
    (void)cfg;
    (void)rank;
    (void)stream;
    if (err && err_size)
        snprintf(err, err_size, "NCCL not linked");
    return 0;
}

int h3_ulysses_comm_bind(h3_ulysses_cfg *cfg, int rank, const void *d_in,
                         void *d_out, char *err, size_t err_size) {
    (void)cfg;
    (void)rank;
    (void)d_in;
    (void)d_out;
    if (err && err_size)
        snprintf(err, err_size, "NCCL not linked");
    return 0;
}

int h3_ulysses_a2a_qkv(h3_ulysses_cfg *cfg, const void *d_in, void *d_out,
                       size_t seq_local, size_t heads, size_t head_dim,
                       int dtype_bytes, char *err, size_t err_size) {
    (void)cfg;
    (void)d_in;
    (void)d_out;
    (void)seq_local;
    (void)heads;
    (void)head_dim;
    (void)dtype_bytes;
    if (err && err_size)
        snprintf(err, err_size, "NCCL not linked");
    return 0;
}

int h3_ulysses_a2a_attn_out(h3_ulysses_cfg *cfg, const void *d_in, void *d_out,
                            size_t seq_local, size_t heads, size_t head_dim,
                            int dtype_bytes, char *err, size_t err_size) {
    (void)cfg;
    (void)d_in;
    (void)d_out;
    (void)seq_local;
    (void)heads;
    (void)head_dim;
    (void)dtype_bytes;
    if (err && err_size)
        snprintf(err, err_size, "NCCL not linked");
    return 0;
}
