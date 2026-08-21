/* P4 step-8: DiT block-0 full-ish forward (W4A8 linears + CPU AdaLN/RoPE/SDPA).
 * Build (Linux / CUDA):
 *   cmake --build build-cuda -j --target h3_dit_w4a8_block_probe
 *   ./build-cuda/h3_dit_w4a8_block_probe [--block N] [--batch M] FILE.safetensors
 *
 * 1) Four quantized linears vs host CPU INT8
 * 2) CPU op self-checks (AdaLN / gate / QK-RoPE / SDPA)
 * 3) Full block chain: AdaLN → qkv → rope → SDPA → out → gate →
 *    AdaLN → fc1 → SwiGLU → fc2 → gate  (CUDA linears vs CPU INT8 linears)
 */
#include "h3_dit_w4a8.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const h3_w4a8_linear_kind k_kinds[H3_W4A8_LINEAR_COUNT] = {
    H3_W4A8_LINEAR_QKV, H3_W4A8_LINEAR_OUT, H3_W4A8_LINEAR_FC1,
    H3_W4A8_LINEAR_FC2};

typedef struct {
    int8_t *w_i8;
    float *s_channel;
    size_t n;
    size_t k;
} host_linear;

static uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

static float hash_to_unit(uint64_t seed, size_t index) {
    uint64_t h = mix64(seed ^ (uint64_t)(index + 1) * 0x9e3779b97f4a7c15ULL);
    const int32_t signed_bits = (int32_t)(h >> 32);
    return (float)signed_bits * (1.0f / 2147483648.0f);
}

static const char *format_bytes(size_t bytes, char *buf, size_t buf_size) {
    const double gb = (double)bytes / (1024.0 * 1024.0 * 1024.0);
    const double mb = (double)bytes / (1024.0 * 1024.0);
    if (gb >= 1.0)
        snprintf(buf, buf_size, "%.3f GiB (%zu B)", gb, bytes);
    else
        snprintf(buf, buf_size, "%.1f MiB (%zu B)", mb, bytes);
    return buf;
}

static int within_tol(const float *ref, const float *got, size_t count,
                      float atol, float rtol, float *max_abs_out,
                      double *rmse_out, float *ref_absmax_out) {
    float max_abs = 0.0f;
    float ref_absmax = 0.0f;
    double rmse = 0.0;
    size_t i;
    h3_w4a8_compare_f32(ref, got, count, &max_abs, &rmse);
    for (i = 0; i < count; i++) {
        float a = fabsf(ref[i]);
        if (a > ref_absmax) ref_absmax = a;
    }
    if (max_abs_out) *max_abs_out = max_abs;
    if (rmse_out) *rmse_out = rmse;
    if (ref_absmax_out) *ref_absmax_out = ref_absmax;
    return max_abs <= (atol + rtol * (ref_absmax > 0.0f ? ref_absmax : 1.0f));
}

static int all_finite(const float *x, size_t n) {
    size_t i;
    for (i = 0; i < n; i++)
        if (!isfinite(x[i])) return 0;
    return 1;
}

static void fill_x(float *x, size_t count, uint64_t seed) {
    size_t i;
    for (i = 0; i < count; i++) x[i] = hash_to_unit(seed, i);
}

static int load_host(const h3_w4a8_store *store, unsigned block,
                     h3_w4a8_linear_kind kind, int group_size,
                     host_linear *out, char *error, size_t error_size) {
    h3_w4a8_quad quad;
    memset(out, 0, sizeof(*out));
    if (!h3_w4a8_find_block_linear(store, block,
                                   h3_w4a8_linear_kind_suffix(kind), &quad,
                                   error, error_size))
        return 0;
    return h3_w4a8_load_linear_int8(store, &quad, group_size, &out->w_i8,
                                    &out->s_channel, &out->n, &out->k, error,
                                    error_size);
}

static void free_host(host_linear *h) {
    if (!h) return;
    free(h->w_i8);
    free(h->s_channel);
    memset(h, 0, sizeof(*h));
}

static int cpu_linear(const host_linear *h, const float *x, size_t m,
                      int convrot_gs, float *y, char *error,
                      size_t error_size) {
    float *x_rot = NULL;
    int8_t *x_i8 = NULL;
    float *scales = NULL;
    int ok = 0;
    if (!h || !h->w_i8 || !x || !y) {
        snprintf(error, error_size, "cpu_linear: invalid args");
        return 0;
    }
    x_rot = (float *)malloc(m * h->k * sizeof(float));
    x_i8 = (int8_t *)malloc(m * h->k);
    scales = (float *)malloc(m * sizeof(float));
    if (!x_rot || !x_i8 || !scales) {
        snprintf(error, error_size, "cpu_linear: OOM");
        goto done;
    }
    if (!h3_w4a8_convrot_act_f32(x, m, h->k, convrot_gs, x_rot, error,
                                 error_size) ||
        !h3_w4a8_quantize_act_int8(x_rot, m, h->k, H3_W4A8_ACT_QUANT_PER_ROW,
                                   x_i8, scales, error, error_size))
        goto done;
    h3_w4a8_gemm_i8_dequant_f32(x_i8, h->w_i8, scales, 0, h->s_channel, m,
                                h->n, h->k, y);
    ok = 1;
done:
    free(x_rot);
    free(x_i8);
    free(scales);
    return ok;
}

static int smoke_one(h3_dit_w4a8 *ctx, const host_linear *host,
                     unsigned block, h3_w4a8_linear_kind kind, size_t batch,
                     float atol, float rtol, uint64_t seed) {
    char error[512];
    size_t n = 0, k = 0;
    float *x = NULL, *y_gpu = NULL, *y_cpu = NULL;
    h3_dit_w4a8_linear_stats st;
    float max_abs = 0.0f, ref_absmax = 0.0f;
    double rmse = 0.0;
    int pass = 0;
    const char *name = h3_w4a8_linear_kind_name(kind);

    if (!h3_dit_w4a8_linear_shape(ctx, block, kind, &n, &k, error,
                                  sizeof(error))) {
        printf("  blocks.%u.%s: FAIL shape: %s\n", block, name, error);
        return 0;
    }
    if (host->n != n || host->k != k) {
        printf("  blocks.%u.%s: FAIL shape mismatch host=%zux%zu gpu=%zux%zu\n",
               block, name, host->n, host->k, n, k);
        return 0;
    }

    x = (float *)malloc(batch * k * sizeof(float));
    y_gpu = (float *)malloc(batch * n * sizeof(float));
    y_cpu = (float *)malloc(batch * n * sizeof(float));
    if (!x || !y_gpu || !y_cpu) {
        printf("  blocks.%u.%s: FAIL OOM\n", block, name);
        goto done;
    }
    fill_x(x, batch * k, seed);

    memset(&st, 0, sizeof(st));
    if (!h3_dit_w4a8_linear_f32(ctx, block, kind, x, batch, y_gpu, &st, error,
                                sizeof(error))) {
        printf("  blocks.%u.%s: FAIL gpu: %s\n", block, name, error);
        goto done;
    }
    if (!cpu_linear(host, x, batch, h3_dit_w4a8_convrot_groupsize(ctx), y_cpu,
                    error, sizeof(error))) {
        printf("  blocks.%u.%s: FAIL cpu: %s\n", block, name, error);
        goto done;
    }

    pass = within_tol(y_cpu, y_gpu, batch * n, atol, rtol, &max_abs, &rmse,
                      &ref_absmax);
    printf("  blocks.%u.%s: N=%zu K=%zu gemm_ms=%.3f backend=%s max_abs=%.6g "
           "rmse=%.6g %s\n",
           block, name, n, k, st.gemm_ms, st.backend[0] ? st.backend : "?",
           max_abs, rmse, pass ? "PASS" : "FAIL");
done:
    free(x);
    free(y_gpu);
    free(y_cpu);
    return pass;
}

static int check_ops_self(const h3_dit_w4a8_block_aux *aux, size_t seq) {
    const size_t hidden = H3_DIT_W4A8_HIDDEN;
    const size_t heads = H3_DIT_W4A8_HEADS;
    const size_t head_dim = H3_DIT_W4A8_HEAD_DIM;
    const size_t inner = H3_DIT_W4A8_INNER;
    const size_t slots = H3_DIT_W4A8_SLOTS;
    const size_t rope_half = H3_DIT_W4A8_ROPE_HALF;
    char error[512];
    float *x = NULL, *mod = NULL, *a = NULL, *b = NULL;
    float *qkv = NULL, *q1 = NULL, *k1 = NULL, *v1 = NULL;
    float *q2 = NULL, *k2 = NULL, *v2 = NULL;
    float *cos = NULL, *sin = NULL, *sdpa1 = NULL, *sdpa2 = NULL;
    int *pt = NULL, *ph = NULL, *pw = NULL;
    uint32_t *row_map = NULL;
    float max_abs = 0.0f, ref_absmax = 0.0f;
    double rmse = 0.0;
    int pass = 1;
    size_t i;

    printf("op_self_check: seq=%zu (CPU F32 vs CPU F32)\n", seq);

    x = (float *)malloc(seq * hidden * sizeof(float));
    mod = (float *)malloc(H3_DIT_W4A8_ADALN_OUT * sizeof(float));
    a = (float *)malloc(seq * hidden * sizeof(float));
    b = (float *)malloc(seq * hidden * sizeof(float));
    qkv = (float *)malloc(seq * inner * 3 * sizeof(float));
    q1 = (float *)malloc(seq * inner * sizeof(float));
    k1 = (float *)malloc(seq * inner * sizeof(float));
    v1 = (float *)malloc(seq * inner * sizeof(float));
    q2 = (float *)malloc(seq * inner * sizeof(float));
    k2 = (float *)malloc(seq * inner * sizeof(float));
    v2 = (float *)malloc(seq * inner * sizeof(float));
    cos = (float *)malloc(seq * rope_half * sizeof(float));
    sin = (float *)malloc(seq * rope_half * sizeof(float));
    sdpa1 = (float *)malloc(seq * inner * sizeof(float));
    sdpa2 = (float *)malloc(seq * inner * sizeof(float));
    pt = (int *)malloc(seq * sizeof(int));
    ph = (int *)malloc(seq * sizeof(int));
    pw = (int *)malloc(seq * sizeof(int));
    row_map = (uint32_t *)calloc(seq, sizeof(uint32_t));
    if (!x || !mod || !a || !b || !qkv || !q1 || !k1 || !v1 || !q2 || !k2 ||
        !v2 || !cos || !sin || !sdpa1 || !sdpa2 || !pt || !ph || !pw ||
        !row_map) {
        printf("  ops: FAIL OOM\n");
        pass = 0;
        goto done;
    }

    fill_x(x, seq * hidden, 0x4F505331ULL);
    fill_x(mod, H3_DIT_W4A8_ADALN_OUT, 0x4F505332ULL);
    fill_x(qkv, seq * inner * 3, 0x4F505333ULL);
    for (i = 0; i < seq; i++) {
        pt[i] = (int)i;
        ph[i] = 0;
        pw[i] = (int)(i % 3);
    }

    h3_dit_w4a8_adaln_f32(x, aux->norm1, mod, row_map, seq, hidden, slots, 0, 1,
                          1e-5f, a);
    h3_dit_w4a8_adaln_f32(x, aux->norm1, mod, row_map, seq, hidden, slots, 0, 1,
                          1e-5f, b);
    if (!within_tol(a, b, seq * hidden, 0.0f, 0.0f, &max_abs, &rmse,
                    &ref_absmax) ||
        !all_finite(a, seq * hidden)) {
        printf("  adaln: FAIL max_abs=%.6g finite=%d\n", max_abs,
               all_finite(a, seq * hidden));
        pass = 0;
    } else {
        printf("  adaln: max_abs=0 finite absmax=%.4g PASS\n", ref_absmax);
    }

    h3_dit_w4a8_gate_f32(x, a, mod, row_map, seq, hidden, slots, 2, b);
    {
        float *c = (float *)malloc(seq * hidden * sizeof(float));
        if (!c) {
            printf("  gate: FAIL OOM\n");
            pass = 0;
        } else {
            h3_dit_w4a8_gate_f32(x, a, mod, row_map, seq, hidden, slots, 2, c);
            if (!within_tol(b, c, seq * hidden, 0.0f, 0.0f, &max_abs, &rmse,
                            &ref_absmax) ||
                !all_finite(b, seq * hidden)) {
                printf("  gate: FAIL max_abs=%.6g\n", max_abs);
                pass = 0;
            } else {
                printf("  gate: max_abs=0 finite absmax=%.4g PASS\n",
                       ref_absmax);
            }
            free(c);
        }
    }

    if (!h3_dit_w4a8_rope_tables_f32(aux->inv_freq, seq, pt, ph, pw, 1.0f, cos,
                                     sin, error, sizeof(error))) {
        printf("  rope_tables: FAIL %s\n", error);
        pass = 0;
        goto done;
    }
    if (!h3_dit_w4a8_qkv_rope_f32(qkv, aux->q_norm, aux->k_norm, cos, sin, seq,
                                  heads, head_dim, rope_half, 1e-5f, q1, k1, v1,
                                  error, sizeof(error)) ||
        !h3_dit_w4a8_qkv_rope_f32(qkv, aux->q_norm, aux->k_norm, cos, sin, seq,
                                  heads, head_dim, rope_half, 1e-5f, q2, k2, v2,
                                  error, sizeof(error))) {
        printf("  qkv_rope: FAIL %s\n", error);
        pass = 0;
        goto done;
    }
    if (!within_tol(q1, q2, seq * inner, 0.0f, 0.0f, &max_abs, &rmse,
                    &ref_absmax) ||
        !within_tol(k1, k2, seq * inner, 0.0f, 0.0f, &max_abs, &rmse,
                    &ref_absmax) ||
        !within_tol(v1, v2, seq * inner, 0.0f, 0.0f, &max_abs, &rmse,
                    &ref_absmax) ||
        !all_finite(q1, seq * inner)) {
        printf("  qkv_rope: FAIL\n");
        pass = 0;
    } else {
        printf("  qkv_rope: max_abs=0 finite q_absmax=%.4g PASS\n", ref_absmax);
    }

    if (!h3_dit_w4a8_sdpa_f32(q1, k1, v1, seq, heads, head_dim,
                              1.0f / sqrtf((float)head_dim), sdpa1, error,
                              sizeof(error)) ||
        !h3_dit_w4a8_sdpa_f32(q1, k1, v1, seq, heads, head_dim,
                              1.0f / sqrtf((float)head_dim), sdpa2, error,
                              sizeof(error))) {
        printf("  sdpa: FAIL %s\n", error);
        pass = 0;
        goto done;
    }
    if (!within_tol(sdpa1, sdpa2, seq * inner, 0.0f, 0.0f, &max_abs, &rmse,
                    &ref_absmax) ||
        !all_finite(sdpa1, seq * inner)) {
        printf("  sdpa: FAIL max_abs=%.6g\n", max_abs);
        pass = 0;
    } else {
        printf("  sdpa: max_abs=0 finite absmax=%.4g PASS\n", ref_absmax);
    }

done:
    free(x);
    free(mod);
    free(a);
    free(b);
    free(qkv);
    free(q1);
    free(k1);
    free(v1);
    free(q2);
    free(k2);
    free(v2);
    free(cos);
    free(sin);
    free(sdpa1);
    free(sdpa2);
    free(pt);
    free(ph);
    free(pw);
    free(row_map);
    return pass;
}

/* CPU-linear twin of block_forward for CUDA-vs-CPU comparison. */
static int cpu_block_forward(const host_linear *hosts, int convrot_gs,
                             const h3_dit_w4a8_block_aux *aux, const float *x,
                             size_t seq, const float *modulation,
                             const uint32_t *row_map, const float *rope_cos,
                             const float *rope_sin, float *y, char *error,
                             size_t error_size) {
    const size_t hidden = H3_DIT_W4A8_HIDDEN;
    const size_t heads = H3_DIT_W4A8_HEADS;
    const size_t head_dim = H3_DIT_W4A8_HEAD_DIM;
    const size_t inner = H3_DIT_W4A8_INNER;
    const size_t ffn = H3_DIT_W4A8_FFN;
    const size_t slots = H3_DIT_W4A8_SLOTS;
    const float eps = 1e-5f;
    float *mod_attn = NULL, *qkv = NULL, *query = NULL, *key = NULL, *value = NULL;
    float *attn = NULL, *attn_out = NULL, *after_attn = NULL;
    float *mod_mlp = NULL, *fc1 = NULL, *act = NULL, *mlp_out = NULL;
    int ok = 0;

    mod_attn = (float *)malloc(seq * hidden * sizeof(float));
    qkv = (float *)malloc(seq * inner * 3 * sizeof(float));
    query = (float *)malloc(seq * inner * sizeof(float));
    key = (float *)malloc(seq * inner * sizeof(float));
    value = (float *)malloc(seq * inner * sizeof(float));
    attn = (float *)malloc(seq * inner * sizeof(float));
    attn_out = (float *)malloc(seq * hidden * sizeof(float));
    after_attn = (float *)malloc(seq * hidden * sizeof(float));
    mod_mlp = (float *)malloc(seq * hidden * sizeof(float));
    fc1 = (float *)malloc(seq * ffn * 2 * sizeof(float));
    act = (float *)malloc(seq * ffn * sizeof(float));
    mlp_out = (float *)malloc(seq * hidden * sizeof(float));
    if (!mod_attn || !qkv || !query || !key || !value || !attn || !attn_out ||
        !after_attn || !mod_mlp || !fc1 || !act || !mlp_out) {
        snprintf(error, error_size, "cpu_block OOM");
        goto done;
    }

    h3_dit_w4a8_adaln_f32(x, aux->norm1, modulation, row_map, seq, hidden,
                          slots, 0, 1, eps, mod_attn);
    if (!cpu_linear(&hosts[H3_W4A8_LINEAR_QKV], mod_attn, seq, convrot_gs, qkv,
                    error, error_size))
        goto done;
    if (!h3_dit_w4a8_qkv_rope_f32(qkv, aux->q_norm, aux->k_norm, rope_cos,
                                  rope_sin, seq, heads, head_dim,
                                  H3_DIT_W4A8_ROPE_HALF, eps, query, key, value,
                                  error, error_size))
        goto done;
    if (!h3_dit_w4a8_sdpa_f32(query, key, value, seq, heads, head_dim,
                              1.0f / sqrtf((float)head_dim), attn, error,
                              error_size))
        goto done;
    if (!cpu_linear(&hosts[H3_W4A8_LINEAR_OUT], attn, seq, convrot_gs, attn_out,
                    error, error_size))
        goto done;
    h3_dit_w4a8_gate_f32(x, attn_out, modulation, row_map, seq, hidden, slots,
                         2, after_attn);
    h3_dit_w4a8_adaln_f32(after_attn, aux->norm2, modulation, row_map, seq,
                          hidden, slots, 3, 4, eps, mod_mlp);
    if (!cpu_linear(&hosts[H3_W4A8_LINEAR_FC1], mod_mlp, seq, convrot_gs, fc1,
                    error, error_size))
        goto done;
    h3_dit_w4a8_swiglu_f32(fc1, seq, ffn, act);
    if (!cpu_linear(&hosts[H3_W4A8_LINEAR_FC2], act, seq, convrot_gs, mlp_out,
                    error, error_size))
        goto done;
    h3_dit_w4a8_gate_f32(after_attn, mlp_out, modulation, row_map, seq, hidden,
                         slots, 5, y);
    ok = 1;
done:
    free(mod_attn);
    free(qkv);
    free(query);
    free(key);
    free(value);
    free(attn);
    free(attn_out);
    free(after_attn);
    free(mod_mlp);
    free(fc1);
    free(act);
    free(mlp_out);
    return ok;
}

static int run_full_block(h3_dit_w4a8 *ctx, const host_linear *hosts,
                          const h3_dit_w4a8_block_aux *aux, unsigned block,
                          size_t seq, float atol, float rtol) {
    char error[512];
    const size_t hidden = H3_DIT_W4A8_HIDDEN;
    const size_t rope_half = H3_DIT_W4A8_ROPE_HALF;
    float time_feat[H3_DIT_W4A8_TIME_FEAT];
    float *modulation = NULL;
    float *x = NULL, *y_gpu = NULL, *y_cpu = NULL;
    float *cos = NULL, *sin = NULL;
    int *pt = NULL, *ph = NULL, *pw = NULL;
    uint32_t *row_map = NULL;
    h3_dit_w4a8_block_stats st;
    float max_abs = 0.0f, ref_absmax = 0.0f;
    double rmse = 0.0;
    size_t i;
    size_t t_row;
    int pass = 0;

    printf("full_block: AdaLN→qkv→RoPE→SDPA→out→gate→AdaLN→fc1→SwiGLU→fc2→"
           "gate\n");
    printf("  aux: adaln_w=[%zu,%zu] t_table_rows=%zu inv_freq=%d\n",
           aux->adaln_out, aux->adaln_in, aux->t_table_rows,
           aux->inv_freq ? 1 : 0);

    modulation = (float *)malloc(H3_DIT_W4A8_ADALN_OUT * sizeof(float));
    x = (float *)malloc(seq * hidden * sizeof(float));
    y_gpu = (float *)malloc(seq * hidden * sizeof(float));
    y_cpu = (float *)malloc(seq * hidden * sizeof(float));
    cos = (float *)malloc(seq * rope_half * sizeof(float));
    sin = (float *)malloc(seq * rope_half * sizeof(float));
    pt = (int *)malloc(seq * sizeof(int));
    ph = (int *)malloc(seq * sizeof(int));
    pw = (int *)malloc(seq * sizeof(int));
    row_map = (uint32_t *)calloc(seq, sizeof(uint32_t));
    if (!modulation || !x || !y_gpu || !y_cpu || !cos || !sin || !pt || !ph ||
        !pw || !row_map) {
        printf("  full_block: FAIL OOM\n");
        goto done;
    }

    t_row = aux->t_table_rows > 1 ? aux->t_table_rows / 2 : 0;
    if (!h3_dit_w4a8_time_feat_from_table(aux, t_row, time_feat, error,
                                          sizeof(error))) {
        printf("  time_feat: FAIL %s\n", error);
        goto done;
    }
    if (!h3_dit_w4a8_adaln_project_f32(aux, time_feat, modulation, error,
                                       sizeof(error))) {
        printf("  adaln_project: FAIL %s\n", error);
        goto done;
    }
    if (!all_finite(modulation, H3_DIT_W4A8_ADALN_OUT)) {
        printf("  adaln_project: FAIL non-finite\n");
        goto done;
    }
    printf("  modulation: from t_table[%zu] via F16 adaln_proj  finite PASS\n",
           t_row);

    fill_x(x, seq * hidden, 0x424C4B30ULL + block);
    for (i = 0; i < seq; i++) {
        pt[i] = (int)i;
        ph[i] = (int)(i % 2);
        pw[i] = (int)(i % 4);
    }
    if (!h3_dit_w4a8_rope_tables_f32(aux->inv_freq, seq, pt, ph, pw, 1.0f, cos,
                                     sin, error, sizeof(error))) {
        printf("  rope: FAIL %s\n", error);
        goto done;
    }

    memset(&st, 0, sizeof(st));
    if (!h3_dit_w4a8_block_forward_f32(ctx, block, aux, x, seq, modulation,
                                       row_map, cos, sin, y_gpu, &st, error,
                                       sizeof(error))) {
        printf("  cuda_block: FAIL %s\n", error);
        goto done;
    }
    if (!cpu_block_forward(hosts, h3_dit_w4a8_convrot_groupsize(ctx), aux, x,
                           seq, modulation, row_map, cos, sin, y_cpu, error,
                           sizeof(error))) {
        printf("  cpu_block: FAIL %s\n", error);
        goto done;
    }

    pass = within_tol(y_cpu, y_gpu, seq * hidden, atol, rtol, &max_abs, &rmse,
                      &ref_absmax) &&
           all_finite(y_gpu, seq * hidden);
    printf("  y: max_abs=%.6g rmse=%.6g ref_absmax=%.4g finite=%d %s\n",
           max_abs, rmse, ref_absmax, all_finite(y_gpu, seq * hidden),
           pass ? "PASS" : "FAIL");
    printf("  gemm_ms: qkv=%.3f out=%.3f fc1=%.3f fc2=%.3f backend=%s\n",
           st.qkv_ms, st.out_ms, st.fc1_ms, st.fc2_ms,
           st.qkv_backend[0] ? st.qkv_backend : "?");
    printf("  cpu_ops_ms: adaln_attn=%.2f rope=%.2f sdpa=%.2f gate_attn=%.2f "
           "adaln_mlp=%.2f swiglu=%.2f gate_mlp=%.2f\n",
           st.adaln_attn_ms, st.rope_ms, st.sdpa_ms, st.gate_attn_ms,
           st.adaln_mlp_ms, st.swiglu_ms, st.gate_mlp_ms);

done:
    free(modulation);
    free(x);
    free(y_gpu);
    free(y_cpu);
    free(cos);
    free(sin);
    free(pt);
    free(ph);
    free(pw);
    free(row_map);
    return pass;
}

int main(int argc, char **argv) {
    unsigned block = 0;
    size_t batch = 4;
    float atol = 1e-3f;
    float rtol = 1e-4f;
    int device = 0;
    const char *path = NULL;
    char error[512];
    char b1[64], b2[64], b3[64], b4[64];
    h3_w4a8_store *store = NULL;
    h3_cuda *cuda = NULL;
    h3_dit_w4a8 *ctx = NULL;
    h3_w4a8_cache_config cfg;
    h3_w4a8_cache_stats st;
    h3_cuda_device_info info;
    h3_dit_w4a8_block_aux aux;
    host_linear hosts[H3_W4A8_LINEAR_COUNT];
    size_t free0 = 0, total0 = 0;
    double prefetch_ms = 0.0;
    int pass = 0, fail = 0;
    int i;
    int rc = 1;

    memset(hosts, 0, sizeof(hosts));
    memset(&aux, 0, sizeof(aux));

    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--block") && i + 1 < argc) {
            block = (unsigned)strtoul(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--batch") && i + 1 < argc) {
            batch = (size_t)strtoul(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--atol") && i + 1 < argc) {
            atol = strtof(argv[++i], NULL);
        } else if (!strcmp(argv[i], "--rtol") && i + 1 < argc) {
            rtol = strtof(argv[++i], NULL);
        } else if (!strcmp(argv[i], "--device") && i + 1 < argc) {
            device = (int)strtol(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            fprintf(stderr,
                    "usage: %s [--block N] [--batch M] [--device I] "
                    "FILE.safetensors\n",
                    argv[0]);
            return 0;
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "unknown option: %s\n", argv[i]);
            return 2;
        } else {
            path = argv[i];
        }
    }
    if (!path || batch == 0) {
        fprintf(stderr, "usage: %s FILE.safetensors\n", argv[0]);
        return 2;
    }

    store = h3_w4a8_store_open(path, error, sizeof(error));
    if (!store) {
        fprintf(stderr, "open failed: %s\n", error);
        return 1;
    }
    cuda = h3_cuda_create(device, error, sizeof(error));
    if (!cuda) {
        fprintf(stderr, "cuda create failed: %s\n", error);
        goto cleanup;
    }
    h3_cuda_get_info(cuda, &info);
    h3_cuda_mem_info(cuda, &free0, &total0, NULL, 0);

    h3_dit_w4a8_config_init(&cfg);
    ctx = h3_dit_w4a8_open(cuda, store, &cfg, error, sizeof(error));
    if (!ctx) {
        fprintf(stderr, "dit w4a8 open failed: %s\n", error);
        goto cleanup;
    }

    printf("path: %s\n", path);
    printf("cuda_device: [%d] %s sm_%d%d\n", info.device_index, info.name,
           info.compute_major, info.compute_minor);
    printf("vram_before: free=%s  total=%s\n",
           format_bytes(free0, b1, sizeof(b1)),
           format_bytes(total0, b2, sizeof(b2)));
    printf("block: %u  batch/seq: %zu  act_quant: per-row\n", block, batch);
    printf("convrot_groupsize: %d  group_size: %d\n",
           h3_dit_w4a8_convrot_groupsize(ctx), h3_dit_w4a8_group_size(ctx));
    printf("tol: atol=%g rtol=%g (cuda vs cpu INT8 on linears)\n", atol, rtol);
    printf("scope: 4 linears + CPU AdaLN/RoPE/SDPA/gate full block; "
           "NOT generate / patch / Metal\n");

    if (!h3_dit_w4a8_load_block_aux(store, block, &aux, error, sizeof(error))) {
        fprintf(stderr, "load block aux failed: %s\n", error);
        goto cleanup;
    }
    printf("block_aux: norm1/2 BF16, q/k_norm BF16, adaln_proj F16[%zu,%zu], "
           "rope.inv_freq F32, t_table F32[%zu,%d]\n",
           aux.adaln_out, aux.adaln_in, aux.t_table_rows,
           H3_DIT_W4A8_TIME_FEAT);

    if (!h3_dit_w4a8_prefetch(ctx, block, 1, &prefetch_ms, error,
                              sizeof(error))) {
        fprintf(stderr, "prefetch failed: %s\n", error);
        goto cleanup;
    }
    if (!h3_dit_w4a8_get_cache_stats(ctx, &st, error, sizeof(error))) {
        fprintf(stderr, "stats failed: %s\n", error);
        goto cleanup;
    }
    printf("prefetch_block_%u: wall_ms=%.1f  resident=%s  linears=%u\n", block,
           prefetch_ms, format_bytes(st.bytes_resident, b1, sizeof(b1)),
           st.n_resident_linears);
    printf("vram_after_prefetch: free=%s  used_delta≈%s\n",
           format_bytes(st.vram_free_bytes, b2, sizeof(b2)),
           format_bytes(free0 > st.vram_free_bytes ? free0 - st.vram_free_bytes
                                                   : 0,
                        b3, sizeof(b3)));

    printf("host_int8_ref: loading 4 linears for CPU GEMM...\n");
    for (i = 0; i < H3_W4A8_LINEAR_COUNT; i++) {
        if (!load_host(store, block, k_kinds[i], h3_dit_w4a8_group_size(ctx),
                       &hosts[i], error, sizeof(error))) {
            fprintf(stderr, "host int8 %s failed: %s\n",
                    h3_w4a8_linear_kind_name(k_kinds[i]), error);
            goto cleanup;
        }
        printf("  host %s: N=%zu K=%zu\n", h3_w4a8_linear_kind_name(k_kinds[i]),
               hosts[i].n, hosts[i].k);
    }

    printf("separate_smoke: blocks.%u × {qkv,out,fc1,fc2}\n", block);
    for (i = 0; i < H3_W4A8_LINEAR_COUNT; i++) {
        uint64_t seed = 0x4833444954573400ULL + (uint64_t)i;
        if (smoke_one(ctx, &hosts[i], block, k_kinds[i], batch, atol, rtol,
                      seed))
            pass++;
        else
            fail++;
    }

    if (check_ops_self(&aux, batch))
        pass++;
    else
        fail++;

    if (run_full_block(ctx, hosts, &aux, block, batch, atol, rtol))
        pass++;
    else
        fail++;

    if (!h3_dit_w4a8_get_cache_stats(ctx, &st, error, sizeof(error))) {
        fprintf(stderr, "stats failed: %s\n", error);
        goto cleanup;
    }
    printf("vram_after: free=%s  resident=%s\n",
           format_bytes(st.vram_free_bytes, b4, sizeof(b4)),
           format_bytes(st.bytes_resident, b1, sizeof(b1)));
    printf("summary: separate=4 ops=1 full_block=1 pass=%d fail=%d → %s\n",
           pass, fail, fail == 0 ? "PASS" : "FAIL");
    rc = fail == 0 ? 0 : 1;

cleanup:
    for (i = 0; i < H3_W4A8_LINEAR_COUNT; i++) free_host(&hosts[i]);
    h3_dit_w4a8_block_aux_free(&aux);
    h3_dit_w4a8_free(ctx);
    h3_cuda_free(cuda);
    h3_w4a8_store_free(store);
    return rc;
}
