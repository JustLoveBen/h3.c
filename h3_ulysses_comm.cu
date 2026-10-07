#include "h3_ulysses_comm.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define H3_ULYSSES_MAX_DEV 8

typedef struct h3_ulysses_comm_state {
    int n_dev;
    int *devs;
    ncclComm_t *comms;
    cudaStream_t *streams;
    cudaEvent_t *ev_compute; /* recorded on default (compute) stream */
    cudaEvent_t *ev_comm;    /* recorded on NCCL/peer stream after unpack */
    cudaEvent_t *ev_pack;    /* recorded on comm stream after pack (peer) */
    void **pack;
    void **unpack;
    size_t buf_bytes;
    const void *bound_in[H3_ULYSSES_MAX_DEV];
    void *bound_out[H3_ULYSSES_MAX_DEV];
    cudaStream_t compute[H3_ULYSSES_MAX_DEV]; /* borrowed; GEMM/SDPA stream */
    int peer_ok;
    int peer_fail;
    int peer_a2a; /* 1 = U=2 full-mesh peer; 0 = NCCL */
} h3_ulysses_comm_state;

static int comm_fail(char *err, size_t err_size, const char *msg) {
    if (err && err_size)
        snprintf(err, err_size, "%s", msg ? msg : "ulysses comm error");
    return 0;
}

void h3_ulysses_comm_destroy(h3_ulysses_cfg *cfg) {
    h3_ulysses_comm_state *st;
    int i;
    if (!cfg || !cfg->comm)
        return;
    st = (h3_ulysses_comm_state *)cfg->comm;
    if (st->comms) {
        for (i = 0; i < st->n_dev; i++) {
            if (st->comms[i])
                ncclCommDestroy(st->comms[i]);
        }
    }
    if (st->streams) {
        for (i = 0; i < st->n_dev; i++) {
            if (st->streams[i]) {
                cudaSetDevice(st->devs[i]);
                cudaStreamDestroy(st->streams[i]);
            }
        }
    }
    if (st->ev_compute || st->ev_comm || st->ev_pack) {
        for (i = 0; i < st->n_dev; i++) {
            if (st->devs)
                cudaSetDevice(st->devs[i]);
            if (st->ev_compute && st->ev_compute[i])
                cudaEventDestroy(st->ev_compute[i]);
            if (st->ev_comm && st->ev_comm[i])
                cudaEventDestroy(st->ev_comm[i]);
            if (st->ev_pack && st->ev_pack[i])
                cudaEventDestroy(st->ev_pack[i]);
        }
    }
    if (st->pack) {
        for (i = 0; i < st->n_dev; i++) {
            if (st->pack[i]) {
                cudaSetDevice(st->devs[i]);
                cudaFree(st->pack[i]);
            }
        }
    }
    if (st->unpack) {
        for (i = 0; i < st->n_dev; i++) {
            if (st->unpack[i]) {
                cudaSetDevice(st->devs[i]);
                cudaFree(st->unpack[i]);
            }
        }
    }
    free(st->comms);
    free(st->streams);
    free(st->ev_compute);
    free(st->ev_comm);
    free(st->ev_pack);
    free(st->pack);
    free(st->unpack);
    free(st->devs);
    free(st);
    cfg->comm = NULL;
}

static int ensure_buf(h3_ulysses_comm_state *st, size_t bytes, char *err,
                      size_t err_size) {
    int i;
    if (bytes <= st->buf_bytes)
        return 1;
    for (i = 0; i < st->n_dev; i++) {
        cudaSetDevice(st->devs[i]);
        if (st->pack[i])
            cudaFree(st->pack[i]);
        if (st->unpack[i])
            cudaFree(st->unpack[i]);
        st->pack[i] = st->unpack[i] = NULL;
        if (cudaMalloc(&st->pack[i], bytes) != cudaSuccess ||
            cudaMalloc(&st->unpack[i], bytes) != cudaSuccess)
            return comm_fail(err, err_size, "ulysses a2a OOM");
    }
    st->buf_bytes = bytes;
    return 1;
}

int h3_ulysses_comm_init(h3_ulysses_cfg *cfg, const int *device_ids, int n_dev,
                         char *err, size_t err_size) {
    h3_ulysses_comm_state *st;
    int i, j;
    ncclResult_t nr;

    if (err && err_size)
        err[0] = '\0';
    if (!cfg)
        return comm_fail(err, err_size, "ulysses comm: null cfg");
    if (n_dev <= 1) {
        cfg->world_size = 1;
        if (cfg->ulysses < 1)
            cfg->ulysses = 1;
        cfg->comm = NULL;
        return 1;
    }
    if (n_dev > H3_ULYSSES_MAX_DEV)
        return comm_fail(err, err_size, "ulysses comm: n_dev > 8");
    if (!device_ids)
        return comm_fail(err, err_size, "ulysses comm: null device_ids");
    cfg->ulysses = n_dev;
    cfg->world_size = n_dev;
    if (cfg->tp_size < 1)
        cfg->tp_size = 1;
    if (cfg->ring < 1)
        cfg->ring = 1;

    st = (h3_ulysses_comm_state *)calloc(1, sizeof(*st));
    if (!st)
        return comm_fail(err, err_size, "ulysses comm: OOM");
    st->n_dev = n_dev;
    st->devs = (int *)malloc((size_t)n_dev * sizeof(int));
    st->comms = (ncclComm_t *)calloc((size_t)n_dev, sizeof(ncclComm_t));
    st->streams = (cudaStream_t *)calloc((size_t)n_dev, sizeof(cudaStream_t));
    st->ev_compute = (cudaEvent_t *)calloc((size_t)n_dev, sizeof(cudaEvent_t));
    st->ev_comm = (cudaEvent_t *)calloc((size_t)n_dev, sizeof(cudaEvent_t));
    st->ev_pack = (cudaEvent_t *)calloc((size_t)n_dev, sizeof(cudaEvent_t));
    st->pack = (void **)calloc((size_t)n_dev, sizeof(void *));
    st->unpack = (void **)calloc((size_t)n_dev, sizeof(void *));
    if (!st->devs || !st->comms || !st->streams || !st->ev_compute ||
        !st->ev_comm || !st->ev_pack || !st->pack || !st->unpack) {
        cfg->comm = st;
        h3_ulysses_comm_destroy(cfg);
        return comm_fail(err, err_size, "ulysses comm: OOM");
    }
    memcpy(st->devs, device_ids, (size_t)n_dev * sizeof(int));

    /* GeForce + no NVLink: cudaDeviceCanAccessPeer is usually 0 even on
     * PCIe Gen5. Do not call EnablePeerAccess / MemcpyPeer unless can=1.
     * NCCL can still bounce via SHM/host; that is not the same as P2P. */
    st->peer_ok = st->peer_fail = 0;
    st->peer_a2a = 0;
    for (i = 0; i < n_dev; i++) {
        for (j = 0; j < n_dev; j++) {
            int can = 0;
            cudaError_t en;
            if (i == j)
                continue;
            if (cudaDeviceCanAccessPeer(&can, st->devs[i], st->devs[j]) ==
                    cudaSuccess &&
                can) {
                cudaSetDevice(st->devs[i]);
                en = cudaDeviceEnablePeerAccess(st->devs[j], 0);
                if (en == cudaSuccess ||
                    en == cudaErrorPeerAccessAlreadyEnabled) {
                    (void)cudaGetLastError();
                    st->peer_ok++;
                } else {
                    st->peer_fail++;
                    fprintf(stderr,
                            "h3_ulysses: EnablePeerAccess %d->%d failed: %s\n",
                            st->devs[i], st->devs[j], cudaGetErrorString(en));
                }
            } else {
                st->peer_fail++;
            }
            if (n_dev <= 2)
                fprintf(stderr, "h3_ulysses: canAccessPeer %d->%d can=%d\n",
                        st->devs[i], st->devs[j], can);
        }
    }
    st->peer_a2a = (n_dev == 2 && st->peer_ok == 2 && st->peer_fail == 0) ? 1 : 0;
    fprintf(stderr,
            "h3_ulysses: peer_access ok=%d fail=%d peer_a2a=%d "
            "(need canAccessPeer; PCIe w/o NVLink often 0; no NCCL_P2P_DISABLE)\n",
            st->peer_ok, st->peer_fail, st->peer_a2a);

    for (i = 0; i < n_dev; i++) {
        if (cudaSetDevice(st->devs[i]) != cudaSuccess ||
            cudaStreamCreateWithFlags(&st->streams[i], cudaStreamNonBlocking) !=
                cudaSuccess ||
            cudaEventCreateWithFlags(&st->ev_compute[i],
                                     cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&st->ev_comm[i],
                                     cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&st->ev_pack[i],
                                     cudaEventDisableTiming) != cudaSuccess) {
            cfg->comm = st;
            h3_ulysses_comm_destroy(cfg);
            return comm_fail(err, err_size, "ulysses comm: stream/event create failed");
        }
    }

    nr = ncclCommInitAll(st->comms, n_dev, st->devs);
    if (nr != ncclSuccess) {
        cfg->comm = st;
        h3_ulysses_comm_destroy(cfg);
        return comm_fail(err, err_size, "ncclCommInitAll failed");
    }
    cfg->comm = st;
    return 1;
}

int h3_ulysses_comm_bind(h3_ulysses_cfg *cfg, int rank, const void *d_in,
                         void *d_out, char *err, size_t err_size) {
    h3_ulysses_comm_state *st;
    if (!cfg || !cfg->comm)
        return comm_fail(err, err_size, "ulysses bind: not inited");
    st = (h3_ulysses_comm_state *)cfg->comm;
    if (rank < 0 || rank >= st->n_dev)
        return comm_fail(err, err_size, "ulysses bind: bad rank");
    st->bound_in[rank] = d_in;
    st->bound_out[rank] = d_out;
    return 1;
}

int h3_ulysses_comm_set_compute(h3_ulysses_cfg *cfg, int rank, void *stream,
                                char *err, size_t err_size) {
    h3_ulysses_comm_state *st;
    if (!cfg || !cfg->comm)
        return comm_fail(err, err_size, "ulysses set_compute: not inited");
    st = (h3_ulysses_comm_state *)cfg->comm;
    if (rank < 0 || rank >= st->n_dev)
        return comm_fail(err, err_size, "ulysses set_compute: bad rank");
    st->compute[rank] = (cudaStream_t)stream;
    return 1;
}

__global__ void ulysses_pack_qkv_f32(const float *in, float *out, int seq,
                                     int heads, int head_dim, int u) {
    size_t n = (size_t)seq * 3u * (size_t)heads * (size_t)head_dim;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    int h_local = heads / u;
    for (; idx < n; idx += stride) {
        int d = (int)(idx % (size_t)head_dim);
        size_t t = idx / (size_t)head_dim;
        int h = (int)(t % (size_t)heads);
        t /= (size_t)heads;
        int qkv = (int)(t % 3u);
        int s = (int)(t / 3u);
        int peer = h / h_local;
        int hl = h - peer * h_local;
        size_t dst = ((((size_t)peer * (size_t)seq + (size_t)s) * 3u +
                       (size_t)qkv) *
                          (size_t)h_local +
                      (size_t)hl) *
                         (size_t)head_dim +
                     (size_t)d;
        out[dst] = in[idx];
    }
}

__global__ void ulysses_unpack_qkv_f32(const float *in, float *out, int seq_local,
                                       int heads, int head_dim, int u) {
    int h_local = heads / u;
    size_t n = (size_t)u * (size_t)seq_local * 3u * (size_t)h_local *
               (size_t)head_dim;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (; idx < n; idx += stride) {
        int d = (int)(idx % (size_t)head_dim);
        size_t t = idx / (size_t)head_dim;
        int hl = (int)(t % (size_t)h_local);
        t /= (size_t)h_local;
        int qkv = (int)(t % 3u);
        t /= 3u;
        int s = (int)(t % (size_t)seq_local);
        int src = (int)(t / (size_t)seq_local);
        int s_pad = src * seq_local + s;
        size_t dst = ((((size_t)s_pad * 3u + (size_t)qkv) * (size_t)h_local) +
                      (size_t)hl) *
                         (size_t)head_dim +
                     (size_t)d;
        out[dst] = in[idx];
    }
}

__global__ void ulysses_pack_attn_f32(const float *in, float *out, int seq_local,
                                      int h_local, int head_dim, int u) {
    size_t n = (size_t)u * (size_t)seq_local * (size_t)h_local * (size_t)head_dim;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (; idx < n; idx += stride) {
        int d = (int)(idx % (size_t)head_dim);
        size_t t = idx / (size_t)head_dim;
        int hl = (int)(t % (size_t)h_local);
        t /= (size_t)h_local;
        int s = (int)(t % (size_t)seq_local);
        int src = (int)(t / (size_t)seq_local);
        int s_pad = src * seq_local + s;
        size_t src_i = (((size_t)s_pad * (size_t)h_local) + (size_t)hl) *
                           (size_t)head_dim +
                       (size_t)d;
        out[idx] = in[src_i];
    }
}

__global__ void ulysses_unpack_attn_f32(const float *in, float *out, int seq_local,
                                        int heads, int head_dim, int u) {
    int h_local = heads / u;
    size_t n = (size_t)u * (size_t)seq_local * (size_t)h_local * (size_t)head_dim;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (; idx < n; idx += stride) {
        int d = (int)(idx % (size_t)head_dim);
        size_t t = idx / (size_t)head_dim;
        int hl = (int)(t % (size_t)h_local);
        t /= (size_t)h_local;
        int s = (int)(t % (size_t)seq_local);
        int peer = (int)(t / (size_t)seq_local);
        int h = peer * h_local + hl;
        size_t dst = (((size_t)s * (size_t)heads) + (size_t)h) * (size_t)head_dim +
                     (size_t)d;
        out[dst] = in[idx];
    }
}

__global__ void ulysses_f32_to_bf16(const float *in, __nv_bfloat16 *out,
                                    size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (; idx < n; idx += stride)
        out[idx] = __float2bfloat16(in[idx]);
}

__global__ void ulysses_bf16_to_f32(const __nv_bfloat16 *in, float *out,
                                    size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (; idx < n; idx += stride)
        out[idx] = __bfloat162float(in[idx]);
}

static void launch_n(size_t n, dim3 *grid, dim3 *block) {
    int t = 256;
    int g = (int)((n + (size_t)t - 1u) / (size_t)t);
    if (g < 1)
        g = 1;
    if (g > 2048)
        g = 2048;
    block->x = (unsigned)t;
    block->y = block->z = 1;
    grid->x = (unsigned)g;
    grid->y = grid->z = 1;
}

/* GEMM / SDPA run on the default stream. Pack launches on a non-blocking
 * comm stream. Record compute-done on the default stream and wait it on
 * the comm stream — same happens-after as DeviceSynchronize, without a
 * host drain (~800/run). After unpack, record comm-done and wait it on
 * the default stream so the next phase sees gathered data. */
static int wait_compute(h3_ulysses_comm_state *st, int r, char *err,
                        size_t err_size) {
    if (cudaSetDevice(st->devs[r]) != cudaSuccess)
        return comm_fail(err, err_size, "ulysses a2a: set device");
    /* Pack must wait for this rank's GEMM/SDPA. Those now run on the
     * borrowed op_stream (non-blocking). Recording on legacy default
     * alone would let pack start before compute finished. */
    {
        cudaStream_t src = (cudaStream_t)0;
        if (st->compute[r])
            src = st->compute[r];
        if (cudaEventRecord(st->ev_compute[r], src) != cudaSuccess)
            return comm_fail(err, err_size, "ulysses a2a: record compute");
    }
    if (cudaStreamWaitEvent(st->streams[r], st->ev_compute[r], 0) !=
        cudaSuccess)
        return comm_fail(err, err_size, "ulysses a2a: wait compute");
    return 1;
}

static int publish_comm(h3_ulysses_comm_state *st, int r, char *err,
                        size_t err_size) {
    if (cudaSetDevice(st->devs[r]) != cudaSuccess)
        return comm_fail(err, err_size, "ulysses a2a: set device");
    if (cudaEventRecord(st->ev_comm[r], st->streams[r]) != cudaSuccess)
        return comm_fail(err, err_size, "ulysses a2a: record comm");
    if (cudaStreamWaitEvent(0, st->ev_comm[r], 0) != cudaSuccess)
        return comm_fail(err, err_size, "ulysses a2a: wait comm");
    return 1;
}

static int a2a_sendrecv(h3_ulysses_comm_state *st, size_t chunk, int dtype_bytes,
                        char *err, size_t err_size) {
    int r, p;
    size_t ebytes = (dtype_bytes == 2) ? 2u : 4u;
    ncclDataType_t dt = (dtype_bytes == 2) ? ncclBfloat16 : ncclFloat32;
    if (ncclGroupStart() != ncclSuccess)
        return comm_fail(err, err_size, "ncclGroupStart");
    for (r = 0; r < st->n_dev; r++) {
        for (p = 0; p < st->n_dev; p++) {
            char *s = (char *)st->pack[r] + p * chunk * ebytes;
            char *d = (char *)st->unpack[r] + p * chunk * ebytes;
            if (ncclSend(s, chunk, dt, p, st->comms[r], st->streams[r]) !=
                    ncclSuccess ||
                ncclRecv(d, chunk, dt, p, st->comms[r], st->streams[r]) !=
                    ncclSuccess) {
                ncclGroupEnd();
                return comm_fail(err, err_size, "nccl send/recv a2a failed");
            }
        }
    }
    if (ncclGroupEnd() != ncclSuccess)
        return comm_fail(err, err_size, "ncclGroupEnd");
    return 1;
}

/* U=2 only, and only when canAccessPeer+Enable succeeded both ways.
 * Destination-major pack: unpack[r][p] = pack[p][r]. */
static int a2a_peer_u2(h3_ulysses_comm_state *st, size_t chunk, int dtype_bytes,
                       char *err, size_t err_size) {
    int r, p;
    size_t ebytes = (dtype_bytes == 2) ? 2u : 4u;
    size_t bytes = chunk * ebytes;
    for (r = 0; r < 2; r++) {
        if (cudaSetDevice(st->devs[r]) != cudaSuccess)
            return comm_fail(err, err_size, "ulysses peer: set device");
        if (cudaEventRecord(st->ev_pack[r], st->streams[r]) != cudaSuccess)
            return comm_fail(err, err_size, "ulysses peer: record pack");
    }
    for (r = 0; r < 2; r++) {
        if (cudaSetDevice(st->devs[r]) != cudaSuccess)
            return comm_fail(err, err_size, "ulysses peer: set device");
        p = 1 - r;
        if (cudaStreamWaitEvent(st->streams[r], st->ev_pack[p], 0) !=
            cudaSuccess)
            return comm_fail(err, err_size, "ulysses peer: wait pack");
        if (cudaMemcpyAsync((char *)st->unpack[r] + (size_t)r * bytes,
                            (char *)st->pack[r] + (size_t)r * bytes, bytes,
                            cudaMemcpyDeviceToDevice,
                            st->streams[r]) != cudaSuccess)
            return comm_fail(err, err_size, "ulysses peer: local copy");
        if (cudaMemcpyPeerAsync((char *)st->unpack[r] + (size_t)p * bytes,
                                st->devs[r],
                                (char *)st->pack[p] + (size_t)r * bytes,
                                st->devs[p], bytes,
                                st->streams[r]) != cudaSuccess)
            return comm_fail(err, err_size, "ulysses peer: MemcpyPeerAsync");
    }
    return 1;
}

static int a2a_exchange(h3_ulysses_comm_state *st, size_t chunk, int dtype_bytes,
                        char *err, size_t err_size) {
    if (st->peer_a2a)
        return a2a_peer_u2(st, chunk, dtype_bytes, err, err_size);
    return a2a_sendrecv(st, chunk, dtype_bytes, err, err_size);
}

int h3_ulysses_a2a_qkv(h3_ulysses_cfg *cfg, const void *d_in, void *d_out,
                       size_t seq_local, size_t heads, size_t head_dim,
                       int dtype_bytes, char *err, size_t err_size) {
    h3_ulysses_comm_state *st;
    int u, r, orig = -1;
    size_t elems, chunk, bytes;
    dim3 grid, block;

    if (!cfg || !cfg->comm)
        return comm_fail(err, err_size, "ulysses a2a_qkv: not inited");
    if (dtype_bytes != 4 && dtype_bytes != 2)
        return comm_fail(err, err_size, "ulysses a2a_qkv: dtype_bytes 2 or 4");
    st = (h3_ulysses_comm_state *)cfg->comm;
    u = st->n_dev;
    if (u < 2 || heads % (size_t)u != 0)
        return comm_fail(err, err_size, "ulysses a2a_qkv: bad U/heads");
    /* Optional single-rank bind via the call args when rank buffers unset. */
    if (d_in && d_out && cfg->rank >= 0 && cfg->rank < u) {
        if (!st->bound_in[cfg->rank])
            st->bound_in[cfg->rank] = d_in;
        if (!st->bound_out[cfg->rank])
            st->bound_out[cfg->rank] = d_out;
    }
    for (r = 0; r < u; r++) {
        if (!st->bound_in[r] || !st->bound_out[r])
            return comm_fail(err, err_size, "ulysses a2a_qkv: bind all ranks");
    }
    elems = seq_local * 3u * heads * head_dim;
    chunk = seq_local * 3u * (heads / (size_t)u) * head_dim;
    bytes = elems * sizeof(float);
    if (!ensure_buf(st, bytes, err, err_size))
        return 0;
    cudaGetDevice(&orig);
    launch_n(elems, &grid, &block);
    for (r = 0; r < u; r++) {
        if (!wait_compute(st, r, err, err_size))
            return 0;
        ulysses_pack_qkv_f32<<<grid, block, 0, st->streams[r]>>>(
            (const float *)st->bound_in[r], (float *)st->pack[r],
            (int)seq_local, (int)heads, (int)head_dim, u);
        if (dtype_bytes == 2)
            ulysses_f32_to_bf16<<<grid, block, 0, st->streams[r]>>>(
                (const float *)st->pack[r], (__nv_bfloat16 *)st->pack[r],
                elems);
    }
    if (!a2a_exchange(st, chunk, dtype_bytes, err, err_size))
        return 0;
    for (r = 0; r < u; r++) {
        cudaSetDevice(st->devs[r]);
        if (dtype_bytes == 2) {
            ulysses_bf16_to_f32<<<grid, block, 0, st->streams[r]>>>(
                (const __nv_bfloat16 *)st->unpack[r], (float *)st->pack[r],
                elems);
            ulysses_unpack_qkv_f32<<<grid, block, 0, st->streams[r]>>>(
                (const float *)st->pack[r], (float *)st->bound_out[r],
                (int)seq_local, (int)heads, (int)head_dim, u);
        } else {
            ulysses_unpack_qkv_f32<<<grid, block, 0, st->streams[r]>>>(
                (const float *)st->unpack[r], (float *)st->bound_out[r],
                (int)seq_local, (int)heads, (int)head_dim, u);
        }
        if (!publish_comm(st, r, err, err_size))
            return 0;
    }
    if (orig >= 0)
        cudaSetDevice(orig);
    return 1;
}

int h3_ulysses_a2a_attn_out(h3_ulysses_cfg *cfg, const void *d_in, void *d_out,
                            size_t seq_local, size_t heads, size_t head_dim,
                            int dtype_bytes, char *err, size_t err_size) {
    h3_ulysses_comm_state *st;
    int u, r, orig = -1;
    size_t h_local, elems, chunk;
    dim3 grid, block;

    if (!cfg || !cfg->comm)
        return comm_fail(err, err_size, "ulysses a2a_attn: not inited");
    if (dtype_bytes != 4 && dtype_bytes != 2)
        return comm_fail(err, err_size, "ulysses a2a_attn: dtype_bytes 2 or 4");
    st = (h3_ulysses_comm_state *)cfg->comm;
    u = st->n_dev;
    if (d_in && d_out && cfg->rank >= 0 && cfg->rank < u) {
        if (!st->bound_in[cfg->rank])
            st->bound_in[cfg->rank] = d_in;
        if (!st->bound_out[cfg->rank])
            st->bound_out[cfg->rank] = d_out;
    }
    for (r = 0; r < u; r++) {
        if (!st->bound_in[r] || !st->bound_out[r])
            return comm_fail(err, err_size, "ulysses a2a_attn: bind all ranks");
    }
    h_local = heads / (size_t)u;
    elems = (size_t)u * seq_local * h_local * head_dim;
    chunk = seq_local * h_local * head_dim;
    if (!ensure_buf(st, elems * sizeof(float), err, err_size))
        return 0;
    cudaGetDevice(&orig);
    launch_n(elems, &grid, &block);
    for (r = 0; r < u; r++) {
        if (!wait_compute(st, r, err, err_size))
            return 0;
        ulysses_pack_attn_f32<<<grid, block, 0, st->streams[r]>>>(
            (const float *)st->bound_in[r], (float *)st->pack[r],
            (int)seq_local, (int)h_local, (int)head_dim, u);
        if (dtype_bytes == 2)
            ulysses_f32_to_bf16<<<grid, block, 0, st->streams[r]>>>(
                (const float *)st->pack[r], (__nv_bfloat16 *)st->pack[r],
                elems);
    }
    if (!a2a_exchange(st, chunk, dtype_bytes, err, err_size))
        return 0;
    for (r = 0; r < u; r++) {
        cudaSetDevice(st->devs[r]);
        if (dtype_bytes == 2) {
            ulysses_bf16_to_f32<<<grid, block, 0, st->streams[r]>>>(
                (const __nv_bfloat16 *)st->unpack[r], (float *)st->pack[r],
                elems);
            ulysses_unpack_attn_f32<<<grid, block, 0, st->streams[r]>>>(
                (const float *)st->pack[r], (float *)st->bound_out[r],
                (int)seq_local, (int)heads, (int)head_dim, u);
        } else {
            ulysses_unpack_attn_f32<<<grid, block, 0, st->streams[r]>>>(
                (const float *)st->unpack[r], (float *)st->bound_out[r],
                (int)seq_local, (int)heads, (int)head_dim, u);
        }
        if (!publish_comm(st, r, err, err_size))
            return 0;
    }
    if (orig >= 0)
        cudaSetDevice(orig);
    return 1;
}
