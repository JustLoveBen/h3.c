/*
 * Standalone P2P / host-bounce bandwidth probe.
 * Does not link generate. Does not rmmod. Does not write secrets.
 *
 *   nvcc -O2 -std=c++17 -o h3_p2p_bw_probe tools/h3_p2p_bw_probe.cu
 *   ./h3_p2p_bw_probe            # GPU 0 -> 1, 64 MiB
 *   ./h3_p2p_bw_probe 0 1 64
 *
 * Prints canAccessPeer both ways, then:
 *   - if can=1 both ways: uni / bi cudaMemcpyPeer GB/s
 *   - always: D2H+H2D host-bounce GB/s on src (baseline, not P2P)
 *
 * Gate used by the extreme plan: peer=1 AND uni >= ~20 GB/s.
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void die(const char *msg) {
    fprintf(stderr, "h3_p2p_bw_probe: %s\n", msg);
    std::exit(2);
}

static void ck(cudaError_t e, const char *what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "h3_p2p_bw_probe: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

static double gbps(size_t bytes, float ms) {
    if (ms <= 0.0f)
        return 0.0;
    return (double)bytes / ((double)ms * 1e6);
}

int main(int argc, char **argv) {
    int src = 0;
    int dst = 1;
    int mib = 64;
    int n_dev = 0;
    int can_sd = 0, can_ds = 0;
    size_t bytes;
    void *d_src = NULL, *d_dst = NULL, *h = NULL;
    cudaEvent_t a, b;
    float ms = 0.0f;
    int i;

    if (argc >= 3) {
        src = atoi(argv[1]);
        dst = atoi(argv[2]);
    }
    if (argc >= 4)
        mib = atoi(argv[3]);
    if (src < 0 || dst < 0 || src == dst || mib < 1)
        die("usage: h3_p2p_bw_probe [src dst] [MiB]");

    ck(cudaGetDeviceCount(&n_dev), "cudaGetDeviceCount");
    if (src >= n_dev || dst >= n_dev)
        die("src/dst out of range");

    bytes = (size_t)mib * 1024ull * 1024ull;
    ck(cudaDeviceCanAccessPeer(&can_sd, src, dst), "canAccessPeer src->dst");
    ck(cudaDeviceCanAccessPeer(&can_ds, dst, src), "canAccessPeer dst->src");
    printf("n_dev=%d src=%d dst=%d bytes=%zu (%.1f MiB)\n", n_dev, src, dst,
           bytes, (double)mib);
    printf("canAccessPeer %d->%d = %d\n", src, dst, can_sd);
    printf("canAccessPeer %d->%d = %d\n", dst, src, can_ds);

    ck(cudaSetDevice(src), "set src");
    ck(cudaMalloc(&d_src, bytes), "malloc src");
    ck(cudaMemset(d_src, 1, bytes), "memset src");
    ck(cudaSetDevice(dst), "set dst");
    ck(cudaMalloc(&d_dst, bytes), "malloc dst");
    ck(cudaMemset(d_dst, 2, bytes), "memset dst");

    if (can_sd && can_ds) {
        cudaError_t en;
        ck(cudaSetDevice(src), "set src peer");
        en = cudaDeviceEnablePeerAccess(dst, 0);
        if (en != cudaSuccess && en != cudaErrorPeerAccessAlreadyEnabled)
            ck(en, "EnablePeerAccess src->dst");
        (void)cudaGetLastError();
        ck(cudaSetDevice(dst), "set dst peer");
        en = cudaDeviceEnablePeerAccess(src, 0);
        if (en != cudaSuccess && en != cudaErrorPeerAccessAlreadyEnabled)
            ck(en, "EnablePeerAccess dst->src");
        (void)cudaGetLastError();

        ck(cudaSetDevice(src), "set src copy");
        ck(cudaEventCreate(&a), "ev a");
        ck(cudaEventCreate(&b), "ev b");
        /* warmup */
        ck(cudaMemcpyPeer(d_dst, dst, d_src, src, bytes), "peer warmup");
        ck(cudaDeviceSynchronize(), "sync warmup");
        ck(cudaEventRecord(a, 0), "rec a");
        for (i = 0; i < 8; i++)
            ck(cudaMemcpyPeer(d_dst, dst, d_src, src, bytes), "peer uni");
        ck(cudaEventRecord(b, 0), "rec b");
        ck(cudaEventSynchronize(b), "sync uni");
        ck(cudaEventElapsedTime(&ms, a, b), "elapsed uni");
        printf("p2p_uni_src_to_dst_GBps=%.2f (n=8)\n", gbps(bytes * 8ull, ms));

        ck(cudaSetDevice(dst), "set dst copy");
        ck(cudaMemcpyPeer(d_src, src, d_dst, dst, bytes), "peer warmup2");
        ck(cudaDeviceSynchronize(), "sync warmup2");
        ck(cudaEventRecord(a, 0), "rec a2");
        for (i = 0; i < 8; i++)
            ck(cudaMemcpyPeer(d_src, src, d_dst, dst, bytes), "peer uni2");
        ck(cudaEventRecord(b, 0), "rec b2");
        ck(cudaEventSynchronize(b), "sync uni2");
        ck(cudaEventElapsedTime(&ms, a, b), "elapsed uni2");
        printf("p2p_uni_dst_to_src_GBps=%.2f (n=8)\n", gbps(bytes * 8ull, ms));

        /* bidirectional: two streams, one on each device */
        {
            cudaStream_t s0, s1;
            ck(cudaSetDevice(src), "set src bi");
            ck(cudaStreamCreate(&s0), "s0");
            ck(cudaSetDevice(dst), "set dst bi");
            ck(cudaStreamCreate(&s1), "s1");
            ck(cudaSetDevice(src), "set src bi2");
            ck(cudaEventRecord(a, s0), "bi a");
            for (i = 0; i < 8; i++) {
                ck(cudaMemcpyPeerAsync(d_dst, dst, d_src, src, bytes, s0),
                   "bi src->dst");
                ck(cudaSetDevice(dst), "bi dst");
                ck(cudaMemcpyPeerAsync(d_src, src, d_dst, dst, bytes, s1),
                   "bi dst->src");
                ck(cudaSetDevice(src), "bi src");
            }
            ck(cudaSetDevice(dst), "bi join dst");
            ck(cudaStreamSynchronize(s1), "sync s1");
            ck(cudaSetDevice(src), "bi join src");
            ck(cudaEventRecord(b, s0), "bi b");
            ck(cudaStreamSynchronize(s0), "sync s0");
            ck(cudaEventElapsedTime(&ms, a, b), "elapsed bi");
            /* two directions in the same window */
            printf("p2p_bi_sum_GBps=%.2f (n=8 each way)\n",
                   gbps(bytes * 16ull, ms));
            cudaStreamDestroy(s0);
            cudaStreamDestroy(s1);
        }
        cudaEventDestroy(a);
        cudaEventDestroy(b);
        printf("peer_gate=1  (plan: uni >= ~20 GB/s to schedule U=2 short)\n");
    } else {
        printf("p2p_uni_src_to_dst_GBps=n/a (canAccessPeer=0)\n");
        printf("p2p_uni_dst_to_src_GBps=n/a (canAccessPeer=0)\n");
        printf("p2p_bi_sum_GBps=n/a (canAccessPeer=0)\n");
        printf("peer_gate=0  (do not claim P2P; host bounce only)\n");
    }

    /* Host-bounce baseline on src (always). */
    h = malloc(bytes);
    if (!h)
        die("host malloc");
    memset(h, 3, bytes);
    ck(cudaSetDevice(src), "set src host");
    ck(cudaEventCreate(&a), "host ev a");
    ck(cudaEventCreate(&b), "host ev b");
    ck(cudaMemcpy(h, d_src, bytes, cudaMemcpyDeviceToHost), "d2h warmup");
    ck(cudaMemcpy(d_src, h, bytes, cudaMemcpyHostToDevice), "h2d warmup");
    ck(cudaEventRecord(a, 0), "host a");
    for (i = 0; i < 8; i++) {
        ck(cudaMemcpy(h, d_src, bytes, cudaMemcpyDeviceToHost), "d2h");
        ck(cudaMemcpy(d_src, h, bytes, cudaMemcpyHostToDevice), "h2d");
    }
    ck(cudaEventRecord(b, 0), "host b");
    ck(cudaEventSynchronize(b), "host sync");
    ck(cudaEventElapsedTime(&ms, a, b), "host elapsed");
    /* one bounce = D2H + H2D = 2 * bytes */
    printf("host_bounce_roundtrip_GBps=%.2f (n=8, D2H+H2D)\n",
           gbps(bytes * 16ull, ms));
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    free(h);
    cudaSetDevice(src);
    cudaFree(d_src);
    cudaSetDevice(dst);
    cudaFree(d_dst);
    return (can_sd && can_ds) ? 0 : 1;
}
