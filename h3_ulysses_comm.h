#ifndef H3_ULYSSES_COMM_H
#define H3_ULYSSES_COMM_H

#include "h3_ulysses.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Device pointers. dtype_bytes = 4（本仓 GEMM 后 F32 QKV）或 2. */
int h3_ulysses_comm_init(h3_ulysses_cfg *cfg, const int *device_ids, int n_dev,
                         char *err, size_t err_size);
void h3_ulysses_comm_destroy(h3_ulysses_cfg *cfg);

/* Bind per-rank device buffers, then call a2a once (single-process). */
int h3_ulysses_comm_bind(h3_ulysses_cfg *cfg, int rank, const void *d_in,
                         void *d_out, char *err, size_t err_size);

/* Borrow per-rank GEMM/SDPA stream so A2A pack waits on it, not stream 0. */
int h3_ulysses_comm_set_compute(h3_ulysses_cfg *cfg, int rank, void *stream,
                                char *err, size_t err_size);

/* Bound d_in [seq_local, 3*heads*head_dim] → d_out [seq_pad, 3*(heads/U)*head_dim] */
int h3_ulysses_a2a_qkv(h3_ulysses_cfg *cfg, const void *d_in, void *d_out,
                       size_t seq_local, size_t heads, size_t head_dim,
                       int dtype_bytes, char *err, size_t err_size);

/* Bound d_in [seq_pad, (heads/U)*head_dim] → d_out [seq_local, heads*head_dim] */
int h3_ulysses_a2a_attn_out(h3_ulysses_cfg *cfg, const void *d_in, void *d_out,
                            size_t seq_local, size_t heads, size_t head_dim,
                            int dtype_bytes, char *err, size_t err_size);

#ifdef __cplusplus
}
#endif

#endif
