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

/* H3_VAE_DETERMINISTIC=1 (registered in tools/h3_plugin.h): force a
 * deterministic SDPA engine so VAE decode is bit-stable across runs. The
 * default HeurMode_t::A autotune can pick a different engine per process,
 * which reorders the softmax/value reduction and breaks run-to-run bit
 * parity (observed: async-off VAE differs in 124/124 frames between runs
 * on a fixed latent input). */
static int h3_vae_deterministic_sdpa(void) {
    const char *e = getenv("H3_VAE_DETERMINISTIC");
    return e && e[0] == '1' && e[1] == '\0';
}

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

/* cuDNN frontend CUDA-graph slot. Q/K/V/O live in DiT sdpa scratch at
 * fixed offsets for a given seq, so populate once and launch across
 * blocks. Replay is not run-to-run deterministic on sm_120 (INT8 DiT
 * step-1 velocity diverges; eager execute() is bit-stable). Default off.
 * Opt in: H3_SDPA_CUDNN_CUDA_GRAPH=1. */
struct h3_sdpa_cugraph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    void *q = nullptr;
    void *k = nullptr;
    void *v = nullptr;
    void *o = nullptr;
    int ok = 0;
    int fail = 0;
    unsigned long long launches = 0;
};

struct h3_cudnn_sdpa_plan {
    int64_t tensor_seq = 0;
    int64_t heads = 0;
    int64_t head_dim = 0;
    int64_t batch = 1;
    float scale = 0.0f;
    int causal = 0;
    int use_mask = 0;
    int use_bias = 0;
    int device = -1;
    std::shared_ptr<fe::graph::Graph> graph;
    int64_t workspace_size = 0;
    void *d_workspace = nullptr;
    int32_t *d_seq_len = nullptr;
    int32_t cached_logical_seq = -1;
    h3_sdpa_cugraph cug;
};

/* Per-device slots: graph + workspace + seq_len are allocated on the GPU
 * that first built the shape. A process-wide cache keyed only by
 * (seq,H,D) made U>1 rank1 execute rank0's plan → IMA in rope_sdpa.
 * U=1 still sees 4 slots on device 0 (same eviction as before). */
enum { H3_CUDNN_PLAN_DEV_MAX = 8 };
enum { H3_CUDNN_PLAN_SLOTS = 6 };
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
static unsigned long long g_sdpa_graph_launches;
static unsigned long long g_sdpa_graph_eager;
static int g_sdpa_graph_logged_hit;
static int g_sdpa_graph_logged_fail;

static int sdpa_cugraph_wanted(void) {
    static int inited, want;
    if (!inited) {
        const char *e = getenv("H3_SDPA_CUDNN_CUDA_GRAPH");
        inited = 1;
        /* Default off: cudaGraphLaunch of the populated cuDNN SDPA graph
         * is nondeterministic on sm_120. Eager execute() is bit-stable.
         * H3_SDPA_CUDNN_CUDA_GRAPH=1 opts back in. */
        want = (e && e[0] == '1' && e[1] == '\0') ? 1 : 0;
        fprintf(stderr, "sdpa_graph: wanted=%d (H3_SDPA_CUDNN_CUDA_GRAPH)\n",
                want);
        fflush(stderr);
    }
    return want;
}

static void cugraph_reset(h3_sdpa_cugraph *s) {
    if (!s) return;
    if (s->exec) {
        cudaGraphExecDestroy(s->exec);
        s->exec = nullptr;
    }
    if (s->graph) {
        cudaGraphDestroy(s->graph);
        s->graph = nullptr;
    }
    s->q = s->k = s->v = s->o = nullptr;
    s->ok = 0;
}

static int cugraph_vp_same(const h3_sdpa_cugraph *s,
                           const std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *> &vp) {
    auto q = vp.find(H3_CUDNN_SDPA_Q);
    auto k = vp.find(H3_CUDNN_SDPA_K);
    auto v = vp.find(H3_CUDNN_SDPA_V);
    auto o = vp.find(H3_CUDNN_SDPA_O);
    if (q == vp.end() || k == vp.end() || v == vp.end() || o == vp.end())
        return 0;
    return s->q == q->second && s->k == k->second && s->v == v->second &&
           s->o == o->second;
}

static void cugraph_store_vp(h3_sdpa_cugraph *s,
                             const std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *> &vp) {
    auto q = vp.find(H3_CUDNN_SDPA_Q);
    auto k = vp.find(H3_CUDNN_SDPA_K);
    auto v = vp.find(H3_CUDNN_SDPA_V);
    auto o = vp.find(H3_CUDNN_SDPA_O);
    s->q = (q != vp.end()) ? q->second : nullptr;
    s->k = (k != vp.end()) ? k->second : nullptr;
    s->v = (v != vp.end()) ? v->second : nullptr;
    s->o = (o != vp.end()) ? o->second : nullptr;
}

static int cugraph_populate(h3_sdpa_cugraph *s,
                            const std::shared_ptr<fe::graph::Graph> &fe_graph,
                            cudnnHandle_t handle,
                            std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *> &vp,
                            void *workspace, int64_t seq, const char *tag) {
    cudaGraph_t g = nullptr;
    cudaGraphExec_t ge = nullptr;
    if (!s || !fe_graph || s->fail || !sdpa_cugraph_wanted())
        return 0;
    cugraph_reset(s);
    if (cudaGraphCreate(&g, 0) != cudaSuccess) {
        s->fail = 1;
        if (!g_sdpa_graph_logged_fail) {
            fprintf(stderr, "sdpa_graph: eager reason=create_fail tag=%s seq=%lld\n",
                    tag ? tag : "?", (long long)seq);
            fflush(stderr);
            g_sdpa_graph_logged_fail = 1;
        }
        return 0;
    }
    auto st = fe_graph->populate_cuda_graph(handle, vp, workspace, g);
    if (!st.is_good()) {
        cudaGraphDestroy(g);
        s->fail = 1;
        if (!g_sdpa_graph_logged_fail) {
            fprintf(stderr,
                    "sdpa_graph: eager reason=populate_fail tag=%s seq=%lld "
                    "msg=%s\n",
                    tag ? tag : "?", (long long)seq, st.get_message().c_str());
            fflush(stderr);
            g_sdpa_graph_logged_fail = 1;
        }
        return 0;
    }
    if (cudaGraphInstantiate(&ge, g, 0) != cudaSuccess) {
        cudaGraphDestroy(g);
        s->fail = 1;
        if (!g_sdpa_graph_logged_fail) {
            fprintf(stderr,
                    "sdpa_graph: eager reason=instantiate_fail tag=%s seq=%lld\n",
                    tag ? tag : "?", (long long)seq);
            fflush(stderr);
            g_sdpa_graph_logged_fail = 1;
        }
        return 0;
    }
    s->graph = g;
    s->exec = ge;
    s->ok = 1;
    cugraph_store_vp(s, vp);
    fprintf(stderr, "sdpa_graph: populated tag=%s seq=%lld\n",
            tag ? tag : "?", (long long)seq);
    fflush(stderr);
    return 1;
}

static int cugraph_launch(h3_sdpa_cugraph *s,
                          const std::shared_ptr<fe::graph::Graph> &fe_graph,
                          cudnnHandle_t handle,
                          std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *> &vp,
                          void *workspace) {
    cudaStream_t st = nullptr;
    if (!s || !s->ok || !s->exec || !fe_graph)
        return 0;
    if (!cugraph_vp_same(s, vp)) {
        auto ust = fe_graph->update_cuda_graph(handle, vp, workspace, s->graph);
        if (!ust.is_good()) {
            cugraph_reset(s);
            s->fail = 1;
            if (!g_sdpa_graph_logged_fail) {
                fprintf(stderr, "sdpa_graph: eager reason=update_fail msg=%s\n",
                        ust.get_message().c_str());
                fflush(stderr);
                g_sdpa_graph_logged_fail = 1;
            }
            return 0;
        }
        cudaGraphExecUpdateResultInfo info;
        memset(&info, 0, sizeof(info));
        if (cudaGraphExecUpdate(s->exec, s->graph, &info) != cudaSuccess) {
            cudaGraphExec_t ge = nullptr;
            if (s->exec) {
                cudaGraphExecDestroy(s->exec);
                s->exec = nullptr;
            }
            if (cudaGraphInstantiate(&ge, s->graph, 0) != cudaSuccess) {
                cugraph_reset(s);
                s->fail = 1;
                if (!g_sdpa_graph_logged_fail) {
                    fprintf(stderr,
                            "sdpa_graph: eager reason=reinstantiate_fail\n");
                    fflush(stderr);
                    g_sdpa_graph_logged_fail = 1;
                }
                return 0;
            }
            s->exec = ge;
        }
        cugraph_store_vp(s, vp);
    }
    if (cudnnGetStream(handle, &st) != CUDNN_STATUS_SUCCESS)
        st = nullptr;
    if (cudaGraphLaunch(s->exec, st) != cudaSuccess) {
        cugraph_reset(s);
        s->fail = 1;
        if (!g_sdpa_graph_logged_fail) {
            fprintf(stderr, "sdpa_graph: eager reason=launch_fail\n");
            fflush(stderr);
            g_sdpa_graph_logged_fail = 1;
        }
        return 0;
    }
    s->launches++;
    g_sdpa_graph_launches++;
    if (!g_sdpa_graph_logged_hit ||
        (g_sdpa_graph_launches % 100ull) == 0) {
        fprintf(stderr,
                "sdpa_graph: graph=hit launches=%llu eager=%llu\n",
                g_sdpa_graph_launches, g_sdpa_graph_eager);
        fflush(stderr);
        g_sdpa_graph_logged_hit = 1;
    }
    return 1;
}

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
    cugraph_reset(&p->cug);
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
    p->batch = 1;
    p->scale = 0.0f;
    p->causal = 0;
    p->use_mask = 0;
    p->use_bias = 0;
    p->device = -1;
    p->cached_logical_seq = -1;
    p->cug.fail = 0;
    p->cug.launches = 0;
}

struct h3_cudnn_rect_plan {
    int64_t seq_q = 0;
    int64_t seq_k = 0;
    int64_t heads = 0;
    int64_t head_dim = 0;
    float scale = 0.0f;
    int device = -1;
    std::shared_ptr<fe::graph::Graph> graph;
    int64_t workspace_size = 0;
    void *d_workspace = nullptr;
    h3_sdpa_cugraph cug;
};

static h3_cudnn_rect_plan g_rect[H3_CUDNN_PLAN_DEV_MAX];

/* Separate F32 IO cache: must not evict DiT F16 plans. VAE full-frame
 * seq~29k needs true FP32 (no TF32) or 16px waffle returns. */
static h3_cudnn_sdpa_plan g_f32_plans[H3_CUDNN_PLAN_DEV_MAX];
static int g_f32_tf32_off;

static void rect_reset(h3_cudnn_rect_plan *p) {
    int orig = -1;
    if (!p) return;
    cudaGetDevice(&orig);
    if (p->device >= 0)
        cudaSetDevice(p->device);
    cugraph_reset(&p->cug);
    if (p->d_workspace) {
        cudaFree(p->d_workspace);
        p->d_workspace = nullptr;
    }
    if (orig >= 0)
        cudaSetDevice(orig);
    p->graph.reset();
    p->workspace_size = 0;
    p->seq_q = p->seq_k = p->heads = p->head_dim = 0;
    p->scale = 0.0f;
    p->device = -1;
    p->cug.fail = 0;
    p->cug.launches = 0;
}

extern "C" void h3_cuda_sdpa_cudnn_reset(void) {
    int d, i;
    if (g_sdpa_graph_launches || g_sdpa_graph_eager) {
        fprintf(stderr,
                "sdpa_graph: summary launches=%llu eager=%llu\n",
                g_sdpa_graph_launches, g_sdpa_graph_eager);
        fflush(stderr);
    }
    for (d = 0; d < H3_CUDNN_PLAN_DEV_MAX; d++) {
        for (i = 0; i < g_n_plans[d]; i++)
            plan_reset(&g_plans[d][i]);
        g_n_plans[d] = 0;
        rect_reset(&g_rect[d]);
        plan_reset(&g_f32_plans[d]);
    }
    g_plan = nullptr;
    g_logged_ok = 0;
    g_logged_fail = 0;
}

static int plan_matches(const h3_cudnn_sdpa_plan *p, int64_t tensor_seq,
                        int64_t heads, int64_t head_dim, int64_t batch,
                        float scale, int causal, int use_mask, int use_bias) {
    return p && p->graph && p->tensor_seq == tensor_seq && p->heads == heads &&
           p->head_dim == head_dim && p->batch == batch && p->scale == scale &&
           p->causal == causal && p->use_mask == use_mask &&
           p->use_bias == use_bias;
}

static h3_cudnn_sdpa_plan *plan_find(int dev, int64_t tensor_seq, int64_t heads,
                                     int64_t head_dim, int64_t batch,
                                     float scale, int causal, int use_mask,
                                     int use_bias) {
    int i;
    if (dev < 0 || dev >= H3_CUDNN_PLAN_DEV_MAX)
        return nullptr;
    for (i = 0; i < g_n_plans[dev]; i++) {
        if (plan_matches(&g_plans[dev][i], tensor_seq, heads, head_dim, batch,
                         scale, causal, use_mask, use_bias))
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
                      int64_t head_dim, int64_t batch, float scale, int causal,
                      int use_mask, char *error, size_t error_size) {
    int use_bias =
        (g_bind_bias && g_bind_bias_seq == tensor_seq) ? 1 : 0;
    int dev = plan_cur_dev();
    h3_cudnn_sdpa_plan *p;
    if (batch < 1) batch = 1;
    if (dev < 0 || dev >= H3_CUDNN_PLAN_DEV_MAX) {
        if (error && error_size)
            snprintf(error, error_size, "cudnn sdpa: bad current device");
        return 0;
    }
    p = plan_find(dev, tensor_seq, heads, head_dim, batch, scale, causal,
                  use_mask, use_bias);
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

    const int64_t b = batch;
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

    fe::error_t st = fe::error_t{};
    {
        const int det = h3_vae_deterministic_sdpa();
        if (det) {
            /* Deterministic: pin plan index 0 so the chosen engine cannot
             * drift run-to-run under HeurMode_t::A autotune (which reorders
             * the softmax/value reduction and breaks bit parity). Full chain:
             * build_operation_graph → create_execution_plans → check_support
             * → build_plan_at_index. Fall back to autotune if any step fails. */
            st = graph->build_operation_graph(handle);
            if (st.is_good())
                st = graph->create_execution_plans({fe::HeurMode_t::A});
            if (st.is_good())
                st = graph->check_support(handle);
            if (st.is_good())
                st = graph->build_plan_at_index(handle, 0);
            if (!st.is_good()) {
                fprintf(stderr,
                        "[h3] cudnn sdpa deterministic plan0 unavailable (%s); "
                        "fallback to autotune build\n",
                        st.get_message().c_str());
                st = graph->build(
                    handle, {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK});
            }
        } else {
            st = graph->build(handle,
                              {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK});
        }
    }
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
        if (cudaMalloc((void **)&p->d_seq_len,
                       2 * (size_t)b * sizeof(int32_t)) != cudaSuccess) {
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
    p->batch = batch;
    p->scale = scale;
    p->causal = causal;
    p->use_mask = use_mask;
    p->use_bias = use_bias;
    p->device = dev;

    fprintf(stderr,
            "[h3] cudnn sdpa plan tensor_seq=%lld heads=%lld d=%lld batch=%lld "
            "causal=%d mask=%d bias=%d device=%d ws_mib=%.1f slots=%d%s\n",
            (long long)tensor_seq, (long long)heads, (long long)head_dim,
            (long long)batch, causal, use_mask, use_bias, dev,
            (double)ws / (1024.0 * 1024.0), g_n_plans[dev],
            want_build_time ? "" : "");
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
                      (int64_t)head_dim, 1, scale, causal ? 1 : 0,
                      use_mask ? 1 : 0, error, error_size);
}

extern "C" int h3_cuda_sdpa_cudnn_f16_run_batched(
    void *cudnn_handle, const void *d_q_half, const void *d_k_half,
    const void *d_v_half, void *d_o_half_out, float *d_o_f32, size_t seq,
    size_t heads, size_t head_dim, float scale, int causal, size_t logical_seq,
    int use_mask, size_t batch, double *sdpa_ms, char *error,
    size_t error_size);

extern "C" int h3_cuda_sdpa_cudnn_f16_run(
    void *cudnn_handle, const void *d_q_half, const void *d_k_half,
    const void *d_v_half, void *d_o_half_out, float *d_o_f32, size_t seq,
    size_t heads, size_t head_dim, float scale, double *sdpa_ms, char *error,
    size_t error_size) {
    return h3_cuda_sdpa_cudnn_f16_run_batched(
        cudnn_handle, d_q_half, d_k_half, d_v_half, d_o_half_out, d_o_f32, seq,
        heads, head_dim, scale, 0, seq, 0, 1, sdpa_ms, error, error_size);
}

extern "C" int h3_cuda_sdpa_cudnn_f16_run_ex(
    void *cudnn_handle, const void *d_q_half, const void *d_k_half,
    const void *d_v_half, void *d_o_half_out, float *d_o_f32, size_t seq,
    size_t heads, size_t head_dim, float scale, int causal, size_t logical_seq,
    int use_mask, double *sdpa_ms, char *error, size_t error_size) {
    return h3_cuda_sdpa_cudnn_f16_run_batched(
        cudnn_handle, d_q_half, d_k_half, d_v_half, d_o_half_out, d_o_f32, seq,
        heads, head_dim, scale, causal, logical_seq, use_mask, 1, sdpa_ms,
        error, error_size);
}

extern "C" int h3_cuda_sdpa_cudnn_f16_run_batched(
    void *cudnn_handle, const void *d_q_half, const void *d_k_half,
    const void *d_v_half, void *d_o_half_out, float *d_o_f32, size_t seq,
    size_t heads, size_t head_dim, float scale, int causal, size_t logical_seq,
    int use_mask, size_t batch, double *sdpa_ms, char *error,
    size_t error_size) {
    if (sdpa_ms) *sdpa_ms = 0.0;
    if (logical_seq == 0) logical_seq = seq;
    if (logical_seq > seq) logical_seq = seq;
    if (batch < 1) batch = 1;
    if (!cudnn_handle || !d_q_half || !d_k_half || !d_v_half || !d_o_half_out ||
        seq == 0 || heads == 0 || head_dim == 0) {
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
                    (int64_t)batch, scale, causal ? 1 : 0, use_mask ? 1 : 0,
                    error, error_size))
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
        if (p->cached_logical_seq != (int32_t)logical_seq) {
            std::vector<int32_t> host_len(2 * batch);
            size_t i;
            for (i = 0; i < 2 * batch; i++)
                host_len[i] = (int32_t)logical_seq;
            if (cudaMemcpy(p->d_seq_len, host_len.data(),
                           host_len.size() * sizeof(int32_t),
                           cudaMemcpyHostToDevice) != cudaSuccess) {
                if (error && error_size)
                    snprintf(error, error_size, "cudnn sdpa seq_len H2D failed");
                return 0;
            }
            p->cached_logical_seq = (int32_t)logical_seq;
        }
        vp[H3_CUDNN_SDPA_SEQ_Q] = p->d_seq_len;
        vp[H3_CUDNN_SDPA_SEQ_KV] = p->d_seq_len + batch;
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
    int used_graph = 0;
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

    if (sdpa_cugraph_wanted() && p->cug.ok && !p->cug.fail) {
        used_graph = cugraph_launch(&p->cug, p->graph, handle, vp, p->d_workspace);
    }
    if (!used_graph) {
        auto st = p->graph->execute(handle, vp, p->d_workspace);
        if (!st.is_good()) {
            if (error && error_size) {
                snprintf(error, error_size, "cudnn sdpa execute failed: %s",
                         st.get_message().c_str());
            }
            if (ev0) cudaEventDestroy(ev0);
            if (ev1) cudaEventDestroy(ev1);
            plan_reset(p);
            return 0;
        }
        g_sdpa_graph_eager++;
        if (sdpa_cugraph_wanted() && !p->cug.ok && !p->cug.fail) {
            cudaStream_t wst = nullptr;
            if (cudnnGetStream(handle, &wst) != CUDNN_STATUS_SUCCESS)
                wst = nullptr;
            (void)cudaStreamSynchronize(wst);
            (void)cugraph_populate(&p->cug, p->graph, handle, vp, p->d_workspace,
                                   p->tensor_seq, "square");
        }
    }

    if (d_o_f32) {
        size_t thd = heads * head_dim;
        size_t src_row = seq * thd;
        size_t dst_row = logical_seq * thd;
        const __half *oh = (const __half *)d_o_half_out;
        size_t bi;
        if (batch == 1 && logical_seq == seq) {
            h3_cuda_launch_f16_to_f32(oh, d_o_f32, dst_row);
        } else {
            for (bi = 0; bi < batch; bi++)
                h3_cuda_launch_f16_to_f32(oh + bi * src_row,
                                          d_o_f32 + bi * dst_row, dst_row);
        }
    }

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

/* Rectangular flash: Q=seq_q, K/V=seq_k. Separate plan so square DiT/TE
 * cache is untouched. THD strides match the square graph. */
extern "C" int h3_cuda_sdpa_cudnn_f16_run_rect(
    void *cudnn_handle, const void *d_q_half, const void *d_k_half,
    const void *d_v_half, void *d_o_half_out, float *d_o_f32, size_t seq_q,
    size_t seq_k, size_t heads, size_t head_dim, float scale, double *sdpa_ms,
    char *error, size_t error_size) {
    int dev;
    h3_cudnn_rect_plan *p;
    cudnnHandle_t handle;
    if (sdpa_ms) *sdpa_ms = 0.0;
    if (!cudnn_handle || !d_q_half || !d_k_half || !d_v_half || !d_o_half_out ||
        !d_o_f32 || seq_q == 0 || seq_k == 0 || heads == 0 || head_dim == 0) {
        if (error && error_size)
            snprintf(error, error_size, "invalid cudnn rect sdpa args");
        return 0;
    }
    if (head_dim > 128) {
        if (error && error_size)
            snprintf(error, error_size, "cudnn rect head_dim>128");
        return 0;
    }
    handle = (cudnnHandle_t)cudnn_handle;
    dev = plan_cur_dev();
    if (dev < 0 || dev >= H3_CUDNN_PLAN_DEV_MAX) {
        if (error && error_size)
            snprintf(error, error_size, "cudnn rect bad device");
        return 0;
    }
    p = &g_rect[dev];
    if (!(p->graph && p->seq_q == (int64_t)seq_q && p->seq_k == (int64_t)seq_k &&
          p->heads == (int64_t)heads && p->head_dim == (int64_t)head_dim &&
          p->scale == scale && p->device == dev)) {
        rect_reset(p);
        auto graph = std::make_shared<fe::graph::Graph>();
        graph->set_io_data_type(fe::DataType_t::HALF)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);
        const int64_t b = 1;
        const int64_t h = (int64_t)heads;
        const int64_t d = (int64_t)head_dim;
        const int64_t sq = (int64_t)seq_q;
        const int64_t sk = (int64_t)seq_k;
        const std::vector<int64_t> dim_q = {b, h, sq, d};
        const std::vector<int64_t> dim_k = {b, h, sk, d};
        const std::vector<int64_t> stride_q = {h * sq * d, d, h * d, 1};
        const std::vector<int64_t> stride_k = {h * sk * d, d, h * d, 1};
        auto Q = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("Q")
                                   .set_uid(H3_CUDNN_SDPA_Q)
                                   .set_dim(dim_q)
                                   .set_stride(stride_q));
        auto K = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("K")
                                   .set_uid(H3_CUDNN_SDPA_K)
                                   .set_dim(dim_k)
                                   .set_stride(stride_k));
        auto V = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("V")
                                   .set_uid(H3_CUDNN_SDPA_V)
                                   .set_dim(dim_k)
                                   .set_stride(stride_k));
        auto sdpa_options = fe::graph::SDPA_attributes()
                                .set_name("h3_flash_sdpa_rect")
                                .set_is_inference(true)
                                .set_attn_scale(scale)
                                .set_causal_mask(false);
        auto [O, Stats] = graph->sdpa(Q, K, V, sdpa_options);
        (void)Stats;
        O->set_output(true).set_dim(dim_q).set_stride(stride_q).set_uid(
            H3_CUDNN_SDPA_O);
        fe::error_t st = fe::error_t{};
        if (h3_vae_deterministic_sdpa()) {
            st = graph->build_operation_graph(handle);
            if (st.is_good())
                st = graph->create_execution_plans({fe::HeurMode_t::A});
            if (st.is_good())
                st = graph->check_support(handle);
            if (st.is_good())
                st = graph->build_plan_at_index(handle, 0);
            if (!st.is_good())
                st = graph->build(
                    handle, {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK});
        } else {
            st = graph->build(handle,
                              {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK});
        }
        if (!st.is_good()) {
            if (error && error_size)
                snprintf(error, error_size, "cudnn rect build failed: %s",
                         st.get_message().c_str());
            rect_reset(p);
            return 0;
        }
        int64_t ws = 0;
        st = graph->get_workspace_size(ws);
        if (!st.is_good()) {
            if (error && error_size)
                snprintf(error, error_size, "cudnn rect workspace query failed");
            rect_reset(p);
            return 0;
        }
        if (ws > 0) {
            if (cudaMalloc(&p->d_workspace, (size_t)ws) != cudaSuccess) {
                if (error && error_size)
                    snprintf(error, error_size, "cudnn rect workspace OOM");
                rect_reset(p);
                return 0;
            }
        }
        p->graph = graph;
        p->workspace_size = ws;
        p->seq_q = sq;
        p->seq_k = sk;
        p->heads = h;
        p->head_dim = d;
        p->scale = scale;
        p->device = dev;
        fprintf(stderr,
                "[h3] cudnn sdpa rect plan seq_q=%lld seq_k=%lld heads=%lld "
                "d=%lld device=%d ws_mib=%.1f\n",
                (long long)sq, (long long)sk, (long long)h, (long long)d, dev,
                (double)ws / (1024.0 * 1024.0));
        fflush(stderr);
    }

    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *> vp = {
        {H3_CUDNN_SDPA_Q, const_cast<void *>(d_q_half)},
        {H3_CUDNN_SDPA_K, const_cast<void *>(d_k_half)},
        {H3_CUDNN_SDPA_V, const_cast<void *>(d_v_half)},
        {H3_CUDNN_SDPA_O, d_o_half_out},
    };
    int used_graph = 0;
    if (sdpa_cugraph_wanted() && p->cug.ok && !p->cug.fail)
        used_graph = cugraph_launch(&p->cug, p->graph, handle, vp, p->d_workspace);
    if (!used_graph) {
        auto st = p->graph->execute(handle, vp, p->d_workspace);
        if (!st.is_good()) {
            if (error && error_size)
                snprintf(error, error_size, "cudnn rect execute failed: %s",
                         st.get_message().c_str());
            rect_reset(p);
            return 0;
        }
        g_sdpa_graph_eager++;
        if (sdpa_cugraph_wanted() && !p->cug.ok && !p->cug.fail) {
            cudaStream_t wst = nullptr;
            if (cudnnGetStream(handle, &wst) != CUDNN_STATUS_SUCCESS)
                wst = nullptr;
            (void)cudaStreamSynchronize(wst);
            (void)cugraph_populate(&p->cug, p->graph, handle, vp, p->d_workspace,
                                   (int64_t)seq_q, "rect");
        }
    }
    h3_cuda_launch_f16_to_f32(d_o_half_out, d_o_f32, seq_q * heads * head_dim);
    return 1;
}

static void h3_cudnn_disable_tf32(cudnnHandle_t handle) {
    /* Ampere+ TF32 is on by default for FP32 tensor-core math. The
     * driver override is the documented way to force true FP32 in fused
     * SDPA graphs (no handle-level FMA_MATH). Same intent as ncdhw
     * CUDNN_FMA_MATH on conv descriptors. */
    (void)handle;
    if (!g_f32_tf32_off) {
        setenv("NVIDIA_TF32_OVERRIDE", "0", 1);
        g_f32_tf32_off = 1;
        fprintf(stderr, "[h3] cudnn sdpa f32: NVIDIA_TF32_OVERRIDE=0\n");
        fflush(stderr);
    }
}

static int build_plan_f32(cudnnHandle_t handle, int64_t tensor_seq,
                          int64_t heads, int64_t head_dim, float scale,
                          int causal, int use_mask, char *error,
                          size_t error_size) {
    int dev = plan_cur_dev();
    h3_cudnn_sdpa_plan *p;
    if (dev < 0 || dev >= H3_CUDNN_PLAN_DEV_MAX) {
        if (error && error_size)
            snprintf(error, error_size, "cudnn sdpa f32: bad current device");
        return 0;
    }
    p = &g_f32_plans[dev];
    if (plan_matches(p, tensor_seq, heads, head_dim, 1, scale, causal, use_mask,
                     0))
        return 1;
    plan_reset(p);
    h3_cudnn_disable_tf32(handle);

    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::FLOAT)
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
                            .set_name(causal ? "h3_flash_sdpa_f32_causal"
                                             : "h3_flash_sdpa_f32")
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

    auto [O, Stats] = graph->sdpa(Q, K, V, sdpa_options);
    (void)Stats;
    O->set_output(true)
        .set_dim(dim_qkv)
        .set_stride(stride_qkv)
        .set_uid(H3_CUDNN_SDPA_O);

    fe::error_t st = fe::error_t{};
    if (h3_vae_deterministic_sdpa()) {
        st = graph->build_operation_graph(handle);
        if (st.is_good())
            st = graph->create_execution_plans({fe::HeurMode_t::A});
        if (st.is_good())
            st = graph->check_support(handle);
        if (st.is_good())
            st = graph->build_plan_at_index(handle, 0);
        if (!st.is_good())
            st = graph->build(handle,
                              {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK});
    } else {
        st = graph->build(handle,
                          {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK});
    }
    if (!st.is_good()) {
        if (error && error_size)
            snprintf(error, error_size, "cudnn sdpa f32 build failed: %s",
                     st.get_message().c_str());
        plan_reset(p);
        return 0;
    }
    int64_t ws = 0;
    st = graph->get_workspace_size(ws);
    if (!st.is_good()) {
        if (error && error_size)
            snprintf(error, error_size, "cudnn sdpa f32 workspace query failed");
        plan_reset(p);
        return 0;
    }
    if (ws > 0) {
        if (cudaMalloc(&p->d_workspace, (size_t)ws) != cudaSuccess) {
            if (error && error_size)
                snprintf(error, error_size, "cudnn sdpa f32 workspace OOM");
            plan_reset(p);
            return 0;
        }
    }
    if (use_mask) {
        if (cudaMalloc((void **)&p->d_seq_len, 2 * sizeof(int32_t)) !=
            cudaSuccess) {
            if (error && error_size)
                snprintf(error, error_size, "cudnn sdpa f32 seq_len OOM");
            plan_reset(p);
            return 0;
        }
    }
    p->graph = graph;
    p->workspace_size = ws;
    p->tensor_seq = tensor_seq;
    p->heads = heads;
    p->head_dim = head_dim;
    p->batch = 1;
    p->scale = scale;
    p->causal = causal;
    p->use_mask = use_mask;
    p->use_bias = 0;
    p->device = dev;
    fprintf(stderr,
            "[h3] cudnn sdpa f32 plan tensor_seq=%lld heads=%lld d=%lld "
            "causal=%d mask=%d device=%d ws_mib=%.1f tf32=off\n",
            (long long)tensor_seq, (long long)heads, (long long)head_dim,
            causal, use_mask, dev, (double)ws / (1024.0 * 1024.0));
    fflush(stderr);
    return 1;
}

extern "C" int h3_cuda_sdpa_cudnn_f32_run_ex(
    void *cudnn_handle, const float *d_q, const float *d_k, const float *d_v,
    float *d_o, size_t seq, size_t heads, size_t head_dim, float scale,
    int causal, size_t logical_seq, int use_mask, double *sdpa_ms, char *error,
    size_t error_size) {
    int dev;
    h3_cudnn_sdpa_plan *p;
    cudnnHandle_t handle;
    if (sdpa_ms) *sdpa_ms = 0.0;
    if (logical_seq == 0) logical_seq = seq;
    if (logical_seq > seq) logical_seq = seq;
    if (!cudnn_handle || !d_q || !d_k || !d_v || !d_o || seq == 0 ||
        heads == 0 || head_dim == 0) {
        if (error && error_size)
            snprintf(error, error_size, "invalid cudnn sdpa f32 args");
        return 0;
    }
    if (head_dim > 128) {
        if (error && error_size)
            snprintf(error, error_size, "cudnn sdpa f32 head_dim>128");
        return 0;
    }
    if (use_mask && logical_seq == seq) use_mask = 0;

    handle = (cudnnHandle_t)cudnn_handle;
    h3_cudnn_disable_tf32(handle);
    if (!build_plan_f32(handle, (int64_t)seq, (int64_t)heads, (int64_t)head_dim,
                        scale, causal ? 1 : 0, use_mask ? 1 : 0, error,
                        error_size))
        return 0;

    dev = plan_cur_dev();
    if (dev < 0 || dev >= H3_CUDNN_PLAN_DEV_MAX) {
        if (error && error_size)
            snprintf(error, error_size, "cudnn sdpa f32 bad device");
        return 0;
    }
    p = &g_f32_plans[dev];
    if (!p->graph || p->device != dev) {
        if (error && error_size)
            snprintf(error, error_size, "cudnn sdpa f32 plan missing");
        return 0;
    }

    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *> vp = {
        {H3_CUDNN_SDPA_Q, const_cast<float *>(d_q)},
        {H3_CUDNN_SDPA_K, const_cast<float *>(d_k)},
        {H3_CUDNN_SDPA_V, const_cast<float *>(d_v)},
        {H3_CUDNN_SDPA_O, d_o},
    };
    if (p->use_mask) {
        int32_t host_len[2] = {(int32_t)logical_seq, (int32_t)logical_seq};
        if (cudaMemcpy(p->d_seq_len, host_len, sizeof(host_len),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            if (error && error_size)
                snprintf(error, error_size, "cudnn sdpa f32 seq_len H2D failed");
            return 0;
        }
        vp[H3_CUDNN_SDPA_SEQ_Q] = p->d_seq_len;
        vp[H3_CUDNN_SDPA_SEQ_KV] = p->d_seq_len + 1;
    }

    cudaEvent_t ev0 = nullptr, ev1 = nullptr;
    float elapsed = 0.0f;
    const int want_time = (sdpa_ms != nullptr);
    if (want_time) {
        if (cudaEventCreate(&ev0) != cudaSuccess ||
            cudaEventCreate(&ev1) != cudaSuccess) {
            if (error && error_size)
                snprintf(error, error_size, "cudnn sdpa f32 event create failed");
            if (ev0) cudaEventDestroy(ev0);
            if (ev1) cudaEventDestroy(ev1);
            return 0;
        }
        cudaEventRecord(ev0, 0);
    }

    auto st = p->graph->execute(handle, vp, p->d_workspace);
    if (!st.is_good()) {
        if (error && error_size)
            snprintf(error, error_size, "cudnn sdpa f32 execute failed: %s",
                     st.get_message().c_str());
        if (ev0) cudaEventDestroy(ev0);
        if (ev1) cudaEventDestroy(ev1);
        plan_reset(p);
        return 0;
    }
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
