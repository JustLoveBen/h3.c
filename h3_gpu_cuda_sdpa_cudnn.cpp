/* cuDNN Frontend fused SDPA (Flash) for H3 CUDA bypass.
 *
 * One execution-plan cache keyed by (tensor_seq, heads, d, causal, mask).
 * A separate per-logical-shape fail cache stops TE/DiT/VAE from rebuilding
 * a graph that already returned "No execution plans" — without branding the
 * whole cuDNN handle as dead (VAE seq=1797 must still try after DiT 16k
 * fails).
 *
 * Unaligned seq is retried at ALIGN (default 64) with a padding mask so pad
 * tokens do not change bidirectional attention. Causal pad-at-end does not
 * need a mask: real tokens never attend into the pad. */

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cudnn.h>
#include <cudnn_frontend.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fe = cudnn_frontend;

enum {
    H3_CUDNN_SDPA_Q = 1,
    H3_CUDNN_SDPA_K = 2,
    H3_CUDNN_SDPA_V = 3,
    H3_CUDNN_SDPA_O = 4,
    H3_CUDNN_SDPA_SEQ_Q = 5,
    H3_CUDNN_SDPA_SEQ_KV = 6,
    H3_CUDNN_SDPA_BIAS = 8
};

enum {
    H3_CUDNN_VAR_UNKNOWN = 0,
    H3_CUDNN_VAR_RAW = 1,
    H3_CUDNN_VAR_PAD_MASK = 2,
    H3_CUDNN_VAR_PAD_CAUSAL = 3,
    H3_CUDNN_VAR_FAIL = 4
};

struct h3_cudnn_sdpa_plan {
    int64_t tensor_seq = 0;
    int64_t heads = 0;
    int64_t head_dim = 0;
    float scale = 0.0f;
    int causal = 0;
    int use_mask = 0;
    int use_bias = 0;
    int device = -1;
    std::shared_ptr<fe::graph::Graph> graph;
    int64_t workspace_size = 0;
    void *d_workspace = nullptr;
    int32_t *d_seq_len = nullptr;
};

/* Per-device slots: graph + workspace + seq_len are allocated on the GPU
 * that first built the shape. A process-wide cache keyed only by
 * (seq,H,D) made U>1 rank1 execute rank0's plan → IMA in rope_sdpa.
 * U=1 still sees 4 slots on device 0 (same eviction as before). */
enum { H3_CUDNN_PLAN_DEV_MAX = 8 };
enum { H3_CUDNN_PLAN_SLOTS = 4 };
static h3_cudnn_sdpa_plan g_plans[H3_CUDNN_PLAN_DEV_MAX][H3_CUDNN_PLAN_SLOTS];
static int g_n_plans[H3_CUDNN_PLAN_DEV_MAX];
static h3_cudnn_sdpa_plan *g_plan = nullptr;

static int plan_cur_dev(void) {
    int d = 0;
    if (cudaGetDevice(&d) != cudaSuccess || d < 0)
        return -1;
    return d;
}

struct ShapeKey {
    uint64_t seq;
    uint32_t heads;
    uint32_t head_dim;
    int causal;
    bool operator==(const ShapeKey &o) const {
        return seq == o.seq && heads == o.heads && head_dim == o.head_dim &&
               causal == o.causal;
    }
};

struct ShapeKeyHash {
    size_t operator()(const ShapeKey &k) const {
        uint64_t h = k.seq;
        h ^= (uint64_t)k.heads * 0x9E3779B97F4A7C15ULL;
        h ^= (uint64_t)k.head_dim * 0xBF58476D1CE4E5B9ULL;
        h ^= (uint64_t)(unsigned)k.causal << 48;
        return (size_t)h;
    }
};

static std::unordered_map<ShapeKey, int, ShapeKeyHash> g_shape_var;
static std::unordered_set<uint64_t> g_var_fail;
static int g_logged_fail;
static int g_logged_ok;
static const void *g_bind_bias = nullptr;
static int64_t g_bind_bias_seq = 0;

extern "C" void h3_cuda_launch_f16_to_f32(const void *in_half, float *out,
                                          size_t n);

extern "C" void h3_cuda_sdpa_cudnn_bind_bias(const void *d_bias_half,
                                             size_t seq) {
    g_bind_bias = d_bias_half;
    g_bind_bias_seq = d_bias_half ? (int64_t)seq : 0;
}

static uint64_t var_fail_key(size_t seq, size_t heads, size_t head_dim,
                             int causal, int variant) {
    uint64_t h = (uint64_t)seq;
    h ^= (uint64_t)heads << 32;
    h ^= (uint64_t)head_dim << 44;
    h ^= (uint64_t)(unsigned)causal << 52;
    h ^= (uint64_t)(unsigned)variant << 56;
    return h;
}

static int align_from_env(void) {
    const char *e = getenv("H3_CUDNN_SDPA_ALIGN");
    if (e && e[0]) {
        int a = atoi(e);
        if (a <= 0) return 0;
        return a;
    }
    return 64;
}

extern "C" size_t h3_cuda_sdpa_cudnn_align_seq(size_t seq) {
    int align = align_from_env();
    if (align < 1 || seq == 0) return seq;
    return ((seq + (size_t)align - 1) / (size_t)align) * (size_t)align;
}

extern "C" int h3_cuda_sdpa_cudnn_shape_blocked(size_t seq, size_t heads,
                                                size_t head_dim, int causal) {
    ShapeKey k{(uint64_t)seq, (uint32_t)heads, (uint32_t)head_dim, causal ? 1 : 0};
    auto it = g_shape_var.find(k);
    return it != g_shape_var.end() && it->second == H3_CUDNN_VAR_FAIL;
}

extern "C" int h3_cuda_sdpa_cudnn_want_padded(size_t seq, size_t heads,
                                              size_t head_dim, int causal) {
    ShapeKey k{(uint64_t)seq, (uint32_t)heads, (uint32_t)head_dim, causal ? 1 : 0};
    auto it = g_shape_var.find(k);
    if (it != g_shape_var.end()) {
        if (it->second == H3_CUDNN_VAR_FAIL) return 0;
        return it->second == H3_CUDNN_VAR_PAD_MASK ||
               it->second == H3_CUDNN_VAR_PAD_CAUSAL;
    }
    /* Unaligned seq: skip RAW on the first attempt. cuDNN often has no plan
     * for seq % 64 != 0; trying RAW just burns a failed graph build. */
    return h3_cuda_sdpa_cudnn_align_seq(seq) > seq ? 1 : 0;
}

extern "C" void h3_cuda_sdpa_cudnn_note_variant(size_t seq, size_t heads,
                                                size_t head_dim, int causal,
                                                int variant) {
    ShapeKey k{(uint64_t)seq, (uint32_t)heads, (uint32_t)head_dim, causal ? 1 : 0};
    g_shape_var[k] = variant;
}

extern "C" int h3_cuda_sdpa_cudnn_variant_failed(size_t seq, size_t heads,
                                                 size_t head_dim, int causal,
                                                 int variant) {
    return g_var_fail.count(var_fail_key(seq, heads, head_dim, causal, variant))
               ? 1
               : 0;
}

extern "C" void h3_cuda_sdpa_cudnn_note_variant_fail(size_t seq, size_t heads,
                                                     size_t head_dim, int causal,
                                                     int variant) {
    g_var_fail.insert(var_fail_key(seq, heads, head_dim, causal, variant));
}

static void plan_reset(h3_cudnn_sdpa_plan *p) {
    int orig = -1;
    if (!p) return;
    if (p->device >= 0)
        cudaGetDevice(&orig);
    if (p->device >= 0)
        cudaSetDevice(p->device);
    if (p->d_workspace) {
        cudaFree(p->d_workspace);
        p->d_workspace = nullptr;
    }
    if (p->d_seq_len) {
        cudaFree(p->d_seq_len);
        p->d_seq_len = nullptr;
    }
    if (orig >= 0)
        cudaSetDevice(orig);
    p->graph.reset();
    p->workspace_size = 0;
    p->tensor_seq = p->heads = p->head_dim = 0;
    p->scale = 0.0f;
    p->causal = 0;
    p->use_mask = 0;
    p->use_bias = 0;
    p->device = -1;
}

extern "C" void h3_cuda_sdpa_cudnn_reset(void) {
    int d, i;
    for (d = 0; d < H3_CUDNN_PLAN_DEV_MAX; d++) {
        for (i = 0; i < g_n_plans[d]; i++)
            plan_reset(&g_plans[d][i]);
        g_n_plans[d] = 0;
    }
    g_plan = nullptr;
    g_logged_ok = 0;
    g_logged_fail = 0;
}

static int plan_matches(const h3_cudnn_sdpa_plan *p, int64_t tensor_seq,
                        int64_t heads, int64_t head_dim, float scale,
                        int causal, int use_mask, int use_bias) {
    return p && p->graph && p->tensor_seq == tensor_seq && p->heads == heads &&
           p->head_dim == head_dim && p->scale == scale && p->causal == causal &&
           p->use_mask == use_mask && p->use_bias == use_bias;
}

static h3_cudnn_sdpa_plan *plan_find(int dev, int64_t tensor_seq, int64_t heads,
                                     int64_t head_dim, float scale, int causal,
                                     int use_mask, int use_bias) {
    int i;
    if (dev < 0 || dev >= H3_CUDNN_PLAN_DEV_MAX)
        return nullptr;
    for (i = 0; i < g_n_plans[dev]; i++) {
        if (plan_matches(&g_plans[dev][i], tensor_seq, heads, head_dim, scale,
                         causal, use_mask, use_bias))
            return &g_plans[dev][i];
    }
    return nullptr;
}

static h3_cudnn_sdpa_plan *plan_alloc_slot(int dev) {
    int i, evict = 0;
    if (dev < 0 || dev >= H3_CUDNN_PLAN_DEV_MAX)
        return nullptr;
    if (g_n_plans[dev] < H3_CUDNN_PLAN_SLOTS)
        return &g_plans[dev][g_n_plans[dev]++];
    /* Full on this device: evict the shortest seq (usually TE 576). */
    for (i = 1; i < g_n_plans[dev]; i++) {
        if (g_plans[dev][i].tensor_seq < g_plans[dev][evict].tensor_seq)
            evict = i;
    }
    plan_reset(&g_plans[dev][evict]);
    return &g_plans[dev][evict];
}

static int build_plan(cudnnHandle_t handle, int64_t tensor_seq, int64_t heads,
                      int64_t head_dim, float scale, int causal, int use_mask,
                      char *error, size_t error_size) {
    int use_bias =
        (g_bind_bias && g_bind_bias_seq == tensor_seq) ? 1 : 0;
    int dev = plan_cur_dev();
    h3_cudnn_sdpa_plan *p;
    if (dev < 0 || dev >= H3_CUDNN_PLAN_DEV_MAX) {
        if (error && error_size)
            snprintf(error, error_size, "cudnn sdpa: bad current device");
        return 0;
    }
    p = plan_find(dev, tensor_seq, heads, head_dim, scale, causal, use_mask,
                  use_bias);
    if (p) {
        g_plan = p;
        return 1;
    }
    p = plan_alloc_slot(dev);
    if (!p) {
        if (error && error_size)
            snprintf(error, error_size, "cudnn sdpa: no plan slot");
        return 0;
    }
    g_plan = p;

    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    const int64_t b = 1;
    const int64_t s = tensor_seq;
    const int64_t h = heads;
    const int64_t d = head_dim;
    const std::vector<int64_t> dim_qkv = {b, h, s, d};
    const std::vector<int64_t> stride_qkv = {h * s * d, d, h * d, 1};

    auto Q = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("Q")
                               .set_uid(H3_CUDNN_SDPA_Q)
                               .set_dim(dim_qkv)
                               .set_stride(stride_qkv));
    auto K = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("K")
                               .set_uid(H3_CUDNN_SDPA_K)
                               .set_dim(dim_qkv)
                               .set_stride(stride_qkv));
    auto V = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("V")
                               .set_uid(H3_CUDNN_SDPA_V)
                               .set_dim(dim_qkv)
                               .set_stride(stride_qkv));

    auto sdpa_options = fe::graph::SDPA_attributes()
                            .set_name(causal ? "h3_flash_sdpa_causal"
                                             : "h3_flash_sdpa")
                            .set_is_inference(true)
                            .set_attn_scale(scale)
                            .set_causal_mask(causal != 0);

    if (use_mask) {
        auto seq_q = graph->tensor(
            fe::graph::Tensor_attributes()
                .set_name("seq_q")
                .set_uid(H3_CUDNN_SDPA_SEQ_Q)
                .set_dim({b, 1, 1, 1})
                .set_stride({1, 1, 1, 1})
                .set_data_type(fe::DataType_t::INT32));
        auto seq_kv = graph->tensor(
            fe::graph::Tensor_attributes()
                .set_name("seq_kv")
                .set_uid(H3_CUDNN_SDPA_SEQ_KV)
                .set_dim({b, 1, 1, 1})
                .set_stride({1, 1, 1, 1})
                .set_data_type(fe::DataType_t::INT32));
        sdpa_options.set_padding_mask(true)
            .set_seq_len_q(seq_q)
            .set_seq_len_kv(seq_kv);
    }
    if (use_bias) {
        const std::vector<int64_t> dim_b = {1, 1, s, s};
        const std::vector<int64_t> stride_b = {s * s, s * s, s, 1};
        auto Bias = graph->tensor(fe::graph::Tensor_attributes()
                                      .set_name("Bias")
                                      .set_uid(H3_CUDNN_SDPA_BIAS)
                                      .set_dim(dim_b)
                                      .set_stride(stride_b)
                                      .set_data_type(fe::DataType_t::HALF));
        sdpa_options.set_bias(Bias);
    }

    auto [O, Stats] = graph->sdpa(Q, K, V, sdpa_options);
    (void)Stats;
    O->set_output(true)
        .set_dim(dim_qkv)
        .set_stride(stride_qkv)
        .set_uid(H3_CUDNN_SDPA_O);

    cudaEvent_t ev0 = nullptr, ev1 = nullptr;
    float build_ms = 0.0f;
    const int want_build_time = getenv("H3_CUDNN_PLAN_BUILD_MS") &&
                                getenv("H3_CUDNN_PLAN_BUILD_MS")[0] == '1';
    if (want_build_time) {
        cudaEventCreate(&ev0);
        cudaEventCreate(&ev1);
        if (ev0) cudaEventRecord(ev0, 0);
    }

    auto st = graph->build(handle, {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK});
    if (want_build_time && ev0 && ev1) {
        cudaEventRecord(ev1, 0);
        cudaEventSynchronize(ev1);
        cudaEventElapsedTime(&build_ms, ev0, ev1);
    }
    if (ev0) cudaEventDestroy(ev0);
    if (ev1) cudaEventDestroy(ev1);

    if (!st.is_good()) {
        if (error && error_size) {
            snprintf(error, error_size, "cudnn sdpa build failed: %s",
                     st.get_message().c_str());
        }
        plan_reset(p);
        return 0;
    }

    int64_t ws = 0;
    st = graph->get_workspace_size(ws);
    if (!st.is_good()) {
        if (error && error_size)
            snprintf(error, error_size, "cudnn sdpa workspace query failed");
        plan_reset(p);
        return 0;
    }
    if (ws > 0) {
        if (cudaMalloc(&p->d_workspace, (size_t)ws) != cudaSuccess) {
            if (error && error_size)
                snprintf(error, error_size, "cudnn sdpa workspace OOM");
            plan_reset(p);
            return 0;
        }
    }
    if (use_mask) {
        if (cudaMalloc((void **)&p->d_seq_len, 2 * sizeof(int32_t)) !=
            cudaSuccess) {
            if (error && error_size)
                snprintf(error, error_size, "cudnn sdpa seq_len OOM");
            plan_reset(p);
            return 0;
        }
    }

    p->graph = graph;
    p->workspace_size = ws;
    p->tensor_seq = tensor_seq;
    p->heads = heads;
    p->head_dim = head_dim;
    p->scale = scale;
    p->causal = causal;
    p->use_mask = use_mask;
    p->use_bias = use_bias;
    p->device = dev;

    fprintf(stderr,
            "[h3] cudnn sdpa plan tensor_seq=%lld heads=%lld d=%lld "
            "causal=%d mask=%d bias=%d device=%d ws_mib=%.1f slots=%d%s\n",
            (long long)tensor_seq, (long long)heads, (long long)head_dim,
            causal, use_mask, use_bias, dev, (double)ws / (1024.0 * 1024.0),
            g_n_plans[dev], want_build_time ? "" : "");
    if (want_build_time)
        fprintf(stderr, "[h3] cudnn_plan_build_ms=%.1f\n", (double)build_ms);
    fflush(stderr);
    g_logged_ok = 1;
    return 1;
}

extern "C" int h3_cuda_sdpa_cudnn_available(void) { return 1; }

extern "C" int h3_cuda_sdpa_cudnn_prebuild(
    void *cudnn_handle, size_t seq, size_t heads, size_t head_dim, float scale,
    int causal, int use_mask, char *error, size_t error_size) {
    if (!cudnn_handle || seq == 0 || heads == 0 || head_dim == 0) {
        if (error && error_size)
            snprintf(error, error_size, "invalid cudnn prebuild args");
        return 0;
    }
    return build_plan((cudnnHandle_t)cudnn_handle, (int64_t)seq, (int64_t)heads,
                      (int64_t)head_dim, scale, causal ? 1 : 0,
                      use_mask ? 1 : 0, error, error_size);
}

extern "C" int h3_cuda_sdpa_cudnn_f16_run_ex(
    void *cudnn_handle, const void *d_q_half, const void *d_k_half,
    const void *d_v_half, void *d_o_half_out, float *d_o_f32, size_t seq,
    size_t heads, size_t head_dim, float scale, int causal, size_t logical_seq,
    int use_mask, double *sdpa_ms, char *error, size_t error_size);

extern "C" int h3_cuda_sdpa_cudnn_f16_run(
    void *cudnn_handle, const void *d_q_half, const void *d_k_half,
    const void *d_v_half, void *d_o_half_out, float *d_o_f32, size_t seq,
    size_t heads, size_t head_dim, float scale, double *sdpa_ms, char *error,
    size_t error_size) {
    return h3_cuda_sdpa_cudnn_f16_run_ex(cudnn_handle, d_q_half, d_k_half,
                                         d_v_half, d_o_half_out, d_o_f32, seq,
                                         heads, head_dim, scale, 0, seq, 0,
                                         sdpa_ms, error, error_size);
}

extern "C" int h3_cuda_sdpa_cudnn_f16_run_ex(
    void *cudnn_handle, const void *d_q_half, const void *d_k_half,
    const void *d_v_half, void *d_o_half_out, float *d_o_f32, size_t seq,
    size_t heads, size_t head_dim, float scale, int causal, size_t logical_seq,
    int use_mask, double *sdpa_ms, char *error, size_t error_size) {
    if (sdpa_ms) *sdpa_ms = 0.0;
    if (logical_seq == 0) logical_seq = seq;
    if (logical_seq > seq) logical_seq = seq;
    if (!cudnn_handle || !d_q_half || !d_k_half || !d_v_half || !d_o_half_out ||
        !d_o_f32 || seq == 0 || heads == 0 || head_dim == 0) {
        if (error && error_size)
            snprintf(error, error_size, "invalid cudnn sdpa args");
        return 0;
    }
    if (head_dim > 128) {
        if (error && error_size)
            snprintf(error, error_size, "cudnn sdpa head_dim>128 unsupported");
        return 0;
    }
    if (use_mask && logical_seq == seq) use_mask = 0;

    cudnnHandle_t handle = (cudnnHandle_t)cudnn_handle;
    if (!build_plan(handle, (int64_t)seq, (int64_t)heads, (int64_t)head_dim,
                    scale, causal ? 1 : 0, use_mask ? 1 : 0, error, error_size))
        return 0;

    h3_cudnn_sdpa_plan *p = g_plan;
    if (!p || !p->graph) {
        if (error && error_size)
            snprintf(error, error_size, "cudnn sdpa plan missing");
        return 0;
    }
    if (p->device != plan_cur_dev()) {
        if (error && error_size)
            snprintf(error, error_size, "cudnn sdpa plan device mismatch");
        return 0;
    }
    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *> vp = {
        {H3_CUDNN_SDPA_Q, const_cast<void *>(d_q_half)},
        {H3_CUDNN_SDPA_K, const_cast<void *>(d_k_half)},
        {H3_CUDNN_SDPA_V, const_cast<void *>(d_v_half)},
        {H3_CUDNN_SDPA_O, d_o_half_out},
    };
    if (p->use_mask) {
        int32_t host_len[2] = {(int32_t)logical_seq, (int32_t)logical_seq};
        if (cudaMemcpy(p->d_seq_len, host_len, sizeof(host_len),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            if (error && error_size)
                snprintf(error, error_size, "cudnn sdpa seq_len H2D failed");
            return 0;
        }
        vp[H3_CUDNN_SDPA_SEQ_Q] = p->d_seq_len;
        vp[H3_CUDNN_SDPA_SEQ_KV] = p->d_seq_len + 1;
    }
    if (p->use_bias) {
        if (!g_bind_bias || g_bind_bias_seq != p->tensor_seq) {
            if (error && error_size)
                snprintf(error, error_size, "cudnn sdpa bias not bound");
            return 0;
        }
        vp[H3_CUDNN_SDPA_BIAS] = const_cast<void *>(g_bind_bias);
    }

    cudaEvent_t ev0 = nullptr, ev1 = nullptr;
    float elapsed = 0.0f;
    const int want_time = (sdpa_ms != nullptr);
    if (want_time) {
        if (cudaEventCreate(&ev0) != cudaSuccess ||
            cudaEventCreate(&ev1) != cudaSuccess) {
            if (error && error_size)
                snprintf(error, error_size, "cudnn sdpa event create failed");
            if (ev0) cudaEventDestroy(ev0);
            if (ev1) cudaEventDestroy(ev1);
            return 0;
        }
        cudaEventRecord(ev0, 0);
    }

    auto st = p->graph->execute(handle, vp, p->d_workspace);
    if (!st.is_good()) {
        if (error && error_size) {
            snprintf(error, error_size, "cudnn sdpa execute failed: %s",
                     st.get_message().c_str());
        }
        if (ev0) cudaEventDestroy(ev0);
        if (ev1) cudaEventDestroy(ev1);
        /* Execute fail is not a "no plan" fail — drop this graph so the
         * next shape (VAE) is not stuck with a poisoned plan object. */
        plan_reset(p);
        return 0;
    }

    h3_cuda_launch_f16_to_f32(d_o_half_out, d_o_f32, logical_seq * heads * head_dim);

    if (want_time) {
        cudaEventRecord(ev1, 0);
        cudaEventSynchronize(ev1);
        cudaEventElapsedTime(&elapsed, ev0, ev1);
        *sdpa_ms = (double)elapsed;
        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
    }
    return 1;
}

extern "C" void h3_cuda_sdpa_cudnn_log_fallback(size_t seq, size_t padded,
                                                size_t head_dim, int causal,
                                                const char *msg) {
    if (g_logged_fail) return;
    fprintf(stderr,
            "[h3] cudnn sdpa fallback seq=%zu pad=%zu d=%zu causal=%d: %s\n",
            seq, padded, head_dim, causal, (msg && msg[0]) ? msg : "(no msg)");
    fflush(stderr);
    g_logged_fail = 1;
}
