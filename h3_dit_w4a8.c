#include "h3_dit_w4a8.h"

#include "h3_dit_fused_epi.h"
#include "h3_dit_int8.h"
#include "h3_dit_int8_opt.h"
#include "h3_dit_nvfp4.h"
#include "h3_lora.h"

#include <cuda_runtime_api.h>

#include "h3_cuda_hot_sync.h"

#include <limits.h>
#include <stdint.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>
#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif

/* 16-byte pad so AdaLN `d_a` stays float4-safe. Odd seq used to leave
 * map+8 and break NVFP4 gs==0 quantize (misaligned). Gold layout +8 max. */
#define H3_WS_MAP_BYTES(seq) \
    (((size_t)(seq) * sizeof(uint32_t) + 15u) & ~(size_t)15u)
#define H3_LORA_EV_MAX 256

static int env_flag_is_one(const char *name) {
    const char *e = getenv(name);
    return e && e[0] == '1' && e[1] == '\0';
}

/* Kill-switch: restore broken NVFP4 (W_rot @ unrotated act) for A/B.
 * Default OFF — act ConvRot is required when sidecar weights are
 * ConvRot-baked (tools/quant_dit_nvfp4.py). */
static int nvfp4_skip_act_convrot(void) {
    return env_flag_is_one("H3_DIT_NVFP4_SKIP_ACT_CONVROT");
}

/* H3_DIT_NVFP4_KEEP_INT8=<csv>: no-retrain hybrid — listed linears stay on
 * INT8 ConvRot GEMM; others use the NVFP4 sidecar GEMM.
 * Tokens: qkv,out,fc1,fc2 + aliases attn(=qkv,out), mlp(=fc1,fc2), all.
 * Unset/empty = full FP4. Product default remains INT8 (H3_DIT_NVFP4 unset). */
static unsigned nvfp4_parse_keep_int8(void) {
    const char *e = getenv("H3_DIT_NVFP4_KEEP_INT8");
    unsigned mask = 0;
    char buf[128];
    char *tok, *save = NULL;
    size_t n;
    if (!e || !e[0])
        return 0;
    n = strlen(e);
    if (n >= sizeof(buf))
        n = sizeof(buf) - 1;
    memcpy(buf, e, n);
    buf[n] = '\0';
    for (tok = strtok_r(buf, ",;+|/ ", &save); tok;
         tok = strtok_r(NULL, ",;+|/ ", &save)) {
        while (*tok == ' ' || *tok == '\t')
            tok++;
        if (!tok[0])
            continue;
        if (!strcasecmp(tok, "qkv") || !strcasecmp(tok, "attn.qkv_proj"))
            mask |= 1u << 0;
        else if (!strcasecmp(tok, "out") || !strcasecmp(tok, "attn.out_proj") ||
                 !strcasecmp(tok, "out_proj"))
            mask |= 1u << 1;
        else if (!strcasecmp(tok, "fc1") || !strcasecmp(tok, "mlp.fc1"))
            mask |= 1u << 2;
        else if (!strcasecmp(tok, "fc2") || !strcasecmp(tok, "mlp.fc2"))
            mask |= 1u << 3;
        else if (!strcasecmp(tok, "attn"))
            mask |= (1u << 0) | (1u << 1);
        else if (!strcasecmp(tok, "mlp") || !strcasecmp(tok, "ffn"))
            mask |= (1u << 2) | (1u << 3);
        else if (!strcasecmp(tok, "all") || !strcasecmp(tok, "1") ||
                 !strcasecmp(tok, "*"))
            mask |= 0xFu;
        else
            fprintf(stderr,
                    "dit_nvfp4: KEEP_INT8 unknown token '%s' "
                    "(want qkv,out,fc1,fc2,attn,mlp,all)\n",
                    tok);
    }
    return mask & 0xFu;
}

static void nvfp4_log_keep_int8(unsigned keep) {
    static int logged;
    char names[64];
    size_t o = 0;
    unsigned k;
    static const char *nm[4] = {"qkv", "out", "fc1", "fc2"};
    if (logged)
        return;
    logged = 1;
    names[0] = '\0';
    for (k = 0; k < 4; k++) {
        if (!(keep & (1u << k)))
            continue;
        if (o && o + 1 < sizeof(names))
            names[o++] = ',';
        if (o + strlen(nm[k]) < sizeof(names)) {
            memcpy(names + o, nm[k], strlen(nm[k]));
            o += strlen(nm[k]);
            names[o] = '\0';
        }
    }
    if (!keep)
        fprintf(stderr, "dit_nvfp4: KEEP_INT8=<unset> — full FP4 GEMM\n");
    else
        fprintf(stderr,
                "dit_nvfp4: KEEP_INT8=%s (fp4_mask=0x%x) — hybrid no-retrain\n",
                names, (~keep) & 0xFu);
    fflush(stderr);
}

/* P4.2 Phase-1: FC1→FC2 chained direct-output. Requires H3_DIT_NVFP4=1 AND
 * the chain build (H3_ENABLE_NVFP4_CHAIN). Env H3_DIT_NVFP4_CHAIN=1 enables. */
static int h3_dit_nvfp4_chain_on(void) {
#ifdef H3_ENABLE_NVFP4_CHAIN
    return env_flag_is_one("H3_DIT_NVFP4") &&
           env_flag_is_one("H3_DIT_NVFP4_CHAIN");
#else
    return 0;
#endif
}

/* U>1 only: uly_do_phase sets these so block_phase_* can name the kernel
 * that IMA'd. U=1 leaves rank<0 (no logs, no extra sync). */
static int g_uly_log_rank = -1;
static int g_uly_log_phase = 0;
static unsigned g_uly_log_block = 0;
static int g_uly_phase_sync = 0;
static int g_dit_gemm_epi;

static void uly_klog(const char *kernel, const char *state) {
    if (g_uly_log_rank < 0 || !kernel)
        return;
    fprintf(stderr, "uly_phase: phase%d r%d block%u kernel=%s %s\n",
            g_uly_log_phase, g_uly_log_rank, g_uly_log_block, kernel,
            state ? state : "");
    fflush(stderr);
}

static int uly_ksync(const char *kernel, char *err, size_t errn) {
    cudaError_t ce;
    if (g_uly_log_rank < 0)
        return 1;
    uly_klog(kernel, "launched");
    if (!g_uly_phase_sync)
        return 1;
    ce = h3_cuda_device_sync_class(H3_SYNC_DEBUG, "uly_ksync");
    if (ce != cudaSuccess) {
        snprintf(err, errn, "r%d phase%d %s: %s", g_uly_log_rank,
                 g_uly_log_phase, kernel, cudaGetErrorString(ce));
        uly_klog(kernel, "IMA");
        return 0;
    }
    uly_klog(kernel, "sync_ok");
    return 1;
}

static int env_int_default(const char *name, int def) {
    const char *e = getenv(name);
    if (!e || !e[0]) return def;
    return (int)strtol(e, NULL, 10);
}

static float env_float_default(const char *name, float def) {
    const char *e = getenv(name);
    if (!e || !e[0]) return def;
    return (float)strtod(e, NULL);
}

/* YCNodes Distance Attention: infer t-major (T,H,W) from video RoPE pos. */
static int dist_attn_maybe_configure(h3_cuda *cuda, size_t seq,
                                     size_t video_off, size_t video_seq,
                                     const float *v_pos_t, const float *v_pos_h,
                                     float sigma, char *error,
                                     size_t error_size) {
    size_t w, h;
    int T, H, W, on;
    float scale, tw, start_s, end_s;
    if (!env_flag_is_one("H3_DIT_DIST_ATTN")) {
        if (h3_cuda_dist_attn_active()) h3_cuda_dist_attn_clear();
        return 1;
    }
    if (!cuda || !v_pos_t || !v_pos_h || video_seq < 1) {
        if (error && error_size)
            snprintf(error, error_size, "dist-attn missing video pos");
        return 0;
    }
    w = 1;
    while (w < video_seq && fabsf(v_pos_h[w] - v_pos_h[0]) < 1e-5f &&
           fabsf(v_pos_t[w] - v_pos_t[0]) < 1e-5f)
        w++;
    h = 1;
    while (h * w < video_seq &&
           fabsf(v_pos_t[h * w] - v_pos_t[0]) < 1e-5f)
        h++;
    if (w < 1 || h < 1 || (video_seq % (w * h)) != 0) {
        if (error && error_size)
            snprintf(error, error_size, "dist-attn cannot infer video grid");
        return 0;
    }
    W = (int)w;
    H = (int)h;
    T = (int)(video_seq / (w * h));
    scale = env_float_default("H3_DIT_DIST_SCALE", 15.0f);
    /* YCNodes node default 2, UI max 10. 3 is slightly tighter in time than
     * the node; --dist-temporal raises this (identity lock vs far frames). */
    tw = env_float_default("H3_DIT_DIST_TEMPORAL", 3.0f);
    start_s = env_float_default("H3_DIT_DIST_START_SIGMA", 2.5f);
    end_s = env_float_default("H3_DIT_DIST_END_SIGMA", 0.0f);
    if (!(scale > 0.0f) || !isfinite(scale)) scale = 15.0f;
    if (!(tw > 0.0f) || !isfinite(tw)) tw = 3.0f;
    on = (start_s >= sigma && sigma >= end_s) ? 1 : 0;
    return h3_cuda_dist_attn_configure(cuda, seq, video_off, video_seq, T, H, W,
                                       scale, tw, on, error, error_size);
}

/* Comfy/Metal default text AdaLN modality tag=1 (MOD_TEXT). Override with
 * H3_DIT_TEXT_MOD_ROW=0|1|2 for ablation (0=VIDEO may strengthen text tokens). */
static uint32_t dit_text_mod_row(void) {
    int v = env_int_default("H3_DIT_TEXT_MOD_ROW", (int)H3_DIT_W4A8_MOD_TEXT);
    if (v < 0 || v >= (int)H3_DIT_W4A8_MODALITIES)
        v = (int)H3_DIT_W4A8_MOD_TEXT;
    return (uint32_t)v;
}

/* Comfy MiniMax H3 / Kijai W4A8: qkv_proj.split → contiguous [Q|K|V].
 * Metal BF16 path uses interleaved (grouped=1). Default OFF for this checkpoint.
 * Set H3_DIT_QKV_GROUPED=1 only for Metal-layout exports. */
static int dit_qkv_grouped_layout(void) {
    return env_flag_is_one("H3_DIT_QKV_GROUPED");
}

/* Comfy MiniMaxH3: contiguous QKV + rms_rope_split_half (first/second half).
 * Default: that Comfy pairing. H3_DIT_ROPE_HALF_OFFSET=1 (or
 * H3_DIT_ROPE_SPLIT_HALF=0) → consecutive-pair rotate (legacy mis-impl). */
static int dit_rope_split_half(void) {
    const char *half_off = getenv("H3_DIT_ROPE_HALF_OFFSET");
    const char *split = getenv("H3_DIT_ROPE_SPLIT_HALF");
    if (half_off && half_off[0] == '1') return 0;
    if (split && split[0] == '0') return 0;
    if (split && split[0] == '1') return 1;
    return 1;
}

struct h3_dit_w4a8 {
    h3_cuda *cuda;
    h3_w4a8_store *store;
    h3_dit_int8_store *int8_store; /* mutually exclusive with store */
    h3_nvfp4_store *nvfp4_store;   /* NVFP4 DiT; mutually exclusive */
    h3_w4a8_cache *cache;
    int convrot_gs;
    int group_size;
    h3_w4a8_act_quant_mode act_quant;
    float *x_rot;
    int8_t *x_i8;
    float *act_scales;
    size_t cap_mk;
    size_t cap_m;
    /* Persistent device workspace for GPU-resident block forward. */
    void *d_block_ws;
    size_t d_block_ws_bytes;
    /* Host aux cache: load_block_aux hits safetensors+BF16 convert; keep once. */
    h3_dit_w4a8_block_aux aux_cache[H3_DIT_W4A8_N_BLOCKS];
    unsigned char aux_valid[H3_DIT_W4A8_N_BLOCKS];
    /* CUDA Graph for stable-shape resident block (optional).
     * Keyed by (block, seq); rebuilt when either changes. Requires fixed
     * cuBLASLt workspace (see h3_gpu_cuda). Env: H3_DIT_W4A8_CUDA_GRAPH=1. */
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    size_t graph_seq;
    unsigned graph_block;
    int graph_ok;
    int graph_warmed;
    /* Whole-step CUDA Graph: one capture for all n_blocks of the all-aux
     * device chain (only when the weight window covers the full range).
     * Keyed by (first_block, n_blocks, seq); the captured device pointers
     * must stay stable across steps, so any realloc/reload invalidates.
     * Env: H3_DIT_W4A8_CUDA_GRAPH=1 (shared with the block graph). */
    cudaGraph_t step_graph;
    cudaGraphExec_t step_graph_exec;
    size_t step_graph_seq;
    unsigned step_graph_first;
    unsigned step_graph_n;
    float *d_graph_aux_base; /* d_block_ws offset snapshot at capture */
    int step_graph_ok;
    /* Metal-compatible token-reduction (opt-in; default off). */
    int token_reduction;
    unsigned tr_begin;
    unsigned tr_end;
    unsigned tr_early_steps;
    unsigned tr_early_end;
    float tr_scale;
    int denoise_step; /* for early token-reduction schedule */
    /* Resident F32 patch/final weights (lazy H2D; freed with ctx). */
    float *d_video_patch_w;
    float *d_video_patch_b;
    float *d_final_video_w;
    float *d_final_video_b;
    float *d_final_norm;
    float *d_audio_patch_w;
    float *d_audio_patch_b;
    float *d_final_audio_w;
    float *d_final_audio_b;
    /* Reused final-head scratch (x/mod/normed/y/map); grown as needed. */
    void *d_final_ws;
    size_t d_final_ws_bytes;
    /* End-to-end forward activation ping-pong (patch→layers→final resident). */
    float *d_fwd_act_a;
    float *d_fwd_act_b;
    size_t d_fwd_act_cap; /* floats per buffer (= cap_seq * HIDDEN) */
    /* Token-reduction device scratch (original/baseline/reduced). */
    float *d_tr_original;
    float *d_tr_baseline;
    float *d_tr_reduced_a;
    float *d_tr_reduced_b;
    size_t d_tr_full_cap;     /* floats (= cap_full * H) */
    size_t d_tr_base_cap;     /* floats */
    size_t d_tr_reduced_cap;  /* floats per reduced buffer */
    /* Transient layers IO (set by resident forward; cleared after call). */
    float *io_d_act_a;
    float *io_d_act_b;
    float *io_d_result; /* out: device ptr holding layers result */
    int io_flags;       /* 1=input on device in act_a; 2=leave result on device */
    /* Persistent RoPE tables on device (same seq across denoise steps). */
    float *d_rope_cos;
    float *d_rope_sin;
    size_t d_rope_seq;
    int d_rope_valid; /* 1 after successful H2D for d_rope_seq */
    /* Metal/vLLM: audio AdaLN uses a separate sigma (shift=3) from video
     * (shift=12). When audio_t_active≠0, block AdaLN splices MOD_AUDIO from
     * audio_t_row while video+text keep t_row. */
    size_t audio_t_row;
    int audio_t_active;
    float *adaln_audio_scratch; /* [ADALN_OUT] lazy host scratch for splice */
    size_t visual_cond_t_row;
    int visual_cond_t_active; /* cond_seq>0 且 Metal time=1-σ < 0.999 */
    float *adaln_visual_cond_scratch; /* 0.999 投影；VIDEO 切片拷到 row 3 */
    size_t audio_cond_t_row;
    int audio_cond_t_active; /* audio_cond_seq>0 且 target audio time < 1 */
    float *adaln_audio_cond_scratch; /* time=1.0 投影；AUDIO 切片 */
    /* Borrowed Metal text tags (0=VIDEO vision span, 1=TEXT). NULL → TEXT. */
    const uint8_t *text_mod_tags;
    size_t text_mod_tag_n;
    /* Host aux disk loads (safetensors+BF16). Should be ≈n_blocks once across
     * steps — not n_blocks×steps. */
    unsigned aux_disk_loads;
    unsigned char aux_disk_logged;
    /* 0=unset, 1=all-aux prestaged, 2=dual-bank fallback. */
    unsigned char aux_prestage_mode;
    /* Turbo LoRA residual (NULL = gold; no extra kernels). */
    const h3_lora_set *lora;
    float *d_lora_A[H3_DIT_W4A8_N_BLOCKS][4];
    float *d_lora_B[H3_DIT_W4A8_N_BLOCKS][4];
    int lora_r[H3_DIT_W4A8_N_BLOCKS][4];
    int lora_n[H3_DIT_W4A8_N_BLOCKS][4];
    int lora_k[H3_DIT_W4A8_N_BLOCKS][4];
    float *d_lora_t;
    size_t d_lora_t_cap;
    int lora_max_r;
    unsigned char lora_f16; /* A/B stored as __half; mixed GEMM. */
    cudaEvent_t *lora_ev0;
    cudaEvent_t *lora_ev1;
    int lora_ev_n;
    int lora_ev_cap;
    cudaEvent_t lora_x_ev;
    cudaEvent_t lora_done_ev;
    /* Ulysses SP group (only on rank 0 ctx; uly_ctx[0] == this). Rank ctxs
     * are owned by the caller (generate). uly_u<=1 → today's path. */
    struct h3_dit_w4a8 *uly_ctx[8];
    int uly_dev[8];
    int uly_u;
    h3_dit_w4a8_sp_comm uly_comm;
    /* Layer-wise clean AV KV (TaoMate). NULL when disabled. */
    int ckv_on;
    int ckv_mode; /* 0 off, 1 noisy, 2 commit */
    unsigned ckv_layer;
    size_t ckv_max;
    size_t ckv_n;
    size_t ckv_save_n;
    size_t ckv_save_dst;
    void *d_ckv_k;
    void *d_ckv_v;
    uint32_t *d_ckv_idx;
    size_t ckv_idx_cap;
    /* NVFP4 P1.4/P1.5 (H3_DIT_NVFP4=1): FP4 cuBLASLt GEMM track. FP4
     * weights come from a sidecar safetensors ({base}.weight_fp4 /
     * .weight_sf / .weight_scale2, swizzled-SF layout identical to the
     * runtime act quantizer). INT8 track stays the fallback per-block. */
    int nvfp4_on;
    unsigned nvfp4_keep_int8; /* bit0=qkv bit1=out bit2=fc1 bit3=fc2 */
    unsigned nvfp4_fp4_mask;  /* (~keep)&0xF when nvfp4_on else 0 */
    int nvfp4_chain_on;
    int nvfp4_ready;
    h3_st_header nvfp4_hdr;
    h3_cuda_nvfp4_weight *nvfp4_w[H3_DIT_W4A8_N_BLOCKS][4];
    uint8_t *d_nvfp4_scratch;
    size_t d_nvfp4_scratch_bytes;
    size_t nvfp4_scratch_cap_seq;
    /* Chained FC1→FC2 (P4.2 Phase-1): separate scratch so the non-chain FP4
     * path layout is untouched when the chain toggles off. */
    uint8_t *d_nvfp4_chain_scratch;
    size_t d_nvfp4_chain_scratch_bytes;
    size_t nvfp4_chain_scratch_cap_seq;
    /* Act ConvRot scratch for NVFP4 (W_rot needs X@H before FP4 quant).
     * Sized for seq * max_k (ffn=14336). Separate from gemm scratch so
     * d_x / scratch cannot alias through the quantizer. */
    float *d_nvfp4_rot;
    size_t d_nvfp4_rot_bytes;
    size_t nvfp4_rot_cap_seq;
    int nvfp4_act_convrot_logged;
};

static int nvfp4_kind_use_fp4(const h3_dit_w4a8 *ctx, unsigned kind) {
    if (!ctx || !ctx->nvfp4_on || kind > 3u)
        return 0;
    return (ctx->nvfp4_fp4_mask & (1u << kind)) != 0;
}

static uint32_t dit_text_row_mod(const h3_dit_w4a8 *ctx, size_t i,
                                 size_t text_seq) {
    uint32_t text_mod = dit_text_mod_row();
    if (ctx && ctx->text_mod_tags && ctx->text_mod_tag_n == text_seq &&
        i < text_seq) {
        uint8_t tag = ctx->text_mod_tags[i];
        if (tag < (uint8_t)H3_DIT_W4A8_MODALITIES) return (uint32_t)tag;
    }
    return text_mod;
}

/* Empty cond keeps 3 AdaLN rows (bit-for-bit). Extra rows: VIDEO@0.999,
 * AUDIO@1.0 for Ref2VA visual/audio condition segments. */
static size_t dit_adaln_mod_rows(const h3_dit_w4a8 *ctx) {
    size_t rows = (size_t)H3_DIT_W4A8_MODALITIES;
    if (ctx && ctx->visual_cond_t_active) rows++;
    if (ctx && ctx->audio_cond_t_active) rows++;
    return rows;
}

static size_t dit_adaln_mod_elems(const h3_dit_w4a8 *ctx) {
    return dit_adaln_mod_rows(ctx) * (size_t)H3_DIT_W4A8_SLOTS *
           (size_t)H3_DIT_W4A8_HIDDEN;
}

static void destroy_block_graph(h3_dit_w4a8 *ctx);
static void destroy_step_graph(h3_dit_w4a8 *ctx);

static void fail(char *error, size_t error_size, const char *msg) {
    if (!error || !error_size) return;
    snprintf(error, error_size, "%s", msg ? msg : "h3_dit_w4a8 error");
}

static double elapsed_ms(clock_t t0, clock_t t1) {
    return 1000.0 * (double)(t1 - t0) / (double)CLOCKS_PER_SEC;
}

void h3_dit_w4a8_config_init(h3_w4a8_cache_config *cfg) {
    h3_w4a8_cache_config_init(cfg);
}

h3_dit_w4a8 *h3_dit_w4a8_open(h3_cuda *cuda, h3_w4a8_store *store,
                              const h3_w4a8_cache_config *cfg, char *error,
                              size_t error_size) {
    h3_w4a8_cache_config local;
    h3_dit_w4a8 *ctx;
    const h3_st_quant_info *quant;
    char detail[384];

    if (!cuda || !store) {
        fail(error, error_size, "cuda and W4A8 store are required");
        return NULL;
    }

    ctx = (h3_dit_w4a8 *)calloc(1, sizeof(*ctx));
    if (!ctx) {
        fail(error, error_size, "out of memory opening dit w4a8");
        return NULL;
    }
    ctx->cuda = cuda;
    ctx->store = store;
    ctx->int8_store = NULL;
    ctx->act_quant = H3_W4A8_ACT_QUANT_PER_ROW;

    quant = h3_w4a8_store_quant(store);
    ctx->group_size = (quant && quant->group_size > 0)
                          ? quant->group_size
                          : H3_W4A8_DEFAULT_GROUP_SIZE;
    ctx->convrot_gs = (quant && quant->convrot_groupsize > 0)
                          ? quant->convrot_groupsize
                          : H3_W4A8_DEFAULT_CONVROT_GROUPSIZE;

    if (cfg) {
        local = *cfg;
    } else {
        h3_w4a8_cache_config_init(&local);
    }

    ctx->cache = h3_w4a8_cache_open(cuda, store, &local, detail, sizeof(detail));
    if (!ctx->cache) {
        snprintf(error, error_size, "cache open: %s", detail);
        free(ctx);
        return NULL;
    }
    ctx->token_reduction = 0;
    ctx->tr_begin = 4;
    ctx->tr_end = 30;
    ctx->tr_early_steps = 10;
    ctx->tr_early_end = 40;
    ctx->tr_scale = 1.0f;
    return ctx;
}

h3_dit_w4a8 *h3_dit_w4a8_open_int8(h3_cuda *cuda, h3_dit_int8_store *store,
                                   const h3_w4a8_cache_config *cfg, char *error,
                                   size_t error_size) {
    h3_w4a8_cache_config local;
    h3_dit_w4a8 *ctx;
    const h3_st_quant_info *quant;
    char detail[384];

    if (!cuda || !store) {
        fail(error, error_size, "cuda and INT8 DiT store are required");
        return NULL;
    }

    ctx = (h3_dit_w4a8 *)calloc(1, sizeof(*ctx));
    if (!ctx) {
        fail(error, error_size, "out of memory opening dit int8");
        return NULL;
    }
    ctx->cuda = cuda;
    ctx->store = NULL;
    ctx->int8_store = store;
    ctx->act_quant = H3_W4A8_ACT_QUANT_PER_ROW;
    ctx->group_size = 0;
    quant = h3_dit_int8_store_quant(store);
    ctx->convrot_gs = h3_dit_int8_convrot_groupsize(store, H3_DIT_INT8_HIDDEN);
    if (ctx->convrot_gs <= 0) {
        ctx->convrot_gs = (quant && quant->convrot_groupsize > 0)
                              ? quant->convrot_groupsize
                              : H3_DIT_INT8_DEFAULT_CONVROT_GS;
    }

    if (cfg) {
        local = *cfg;
    } else {
        h3_w4a8_cache_config_init(&local);
    }

    ctx->cache =
        h3_w4a8_cache_open_int8(cuda, store, &local, detail, sizeof(detail));
    if (!ctx->cache) {
        snprintf(error, error_size, "INT8 cache open: %s", detail);
        free(ctx);
        return NULL;
    }
    ctx->token_reduction = 0;
    ctx->tr_begin = 4;
    ctx->tr_end = 30;
    ctx->tr_early_steps = 10;
    ctx->tr_early_end = 40;
    ctx->tr_scale = 1.0f;
    ctx->nvfp4_on = env_flag_is_one("H3_DIT_NVFP4");
    ctx->nvfp4_keep_int8 = ctx->nvfp4_on ? nvfp4_parse_keep_int8() : 0u;
    ctx->nvfp4_fp4_mask =
        ctx->nvfp4_on ? ((~ctx->nvfp4_keep_int8) & 0xFu) : 0u;
    if (ctx->nvfp4_on) {
        nvfp4_log_keep_int8(ctx->nvfp4_keep_int8);
        h3_w4a8_cache_set_nvfp4_int8_skip(ctx->cache, 1);
        fprintf(stderr, "dit_nvfp4: H3_DIT_NVFP4=1 — FP4 cuBLASLt GEMM track "
                        "(INT8 weights load lazily only on FP4 dispatch "
                        "miss)\n");
        if (ctx->convrot_gs > 0 && !nvfp4_skip_act_convrot())
            fprintf(stderr,
                    "dit_nvfp4: act_convrot policy=ON gs=%d "
                    "(per-linear; chain auto-skipped)\n",
                    ctx->convrot_gs);
        else if (ctx->convrot_gs > 0)
            fprintf(stderr,
                    "dit_nvfp4: act_convrot policy=SKIP "
                    "(H3_DIT_NVFP4_SKIP_ACT_CONVROT=1)\n");
        fflush(stderr);
    }
    ctx->nvfp4_chain_on = h3_dit_nvfp4_chain_on();
    if (ctx->nvfp4_chain_on) {
        fprintf(stderr, "dit_nvfp4: H3_DIT_NVFP4_CHAIN=1 — FC1->FC2 chained "
                        "direct-output (CUTLASS sm_120a)\n");
        fflush(stderr);
    }
    return ctx;
}

h3_dit_w4a8 *h3_dit_w4a8_open_nvfp4(h3_cuda *cuda, h3_nvfp4_store *store,
                                    const h3_w4a8_cache_config *cfg,
                                    char *error, size_t error_size) {
    h3_w4a8_cache_config local;
    h3_dit_w4a8 *ctx;
    char detail[384];

    if (!cuda || !store) {
        fail(error, error_size, "cuda and NVFP4 DiT store are required");
        return NULL;
    }
    if (!h3_dit_nvfp4_probe(store, error, error_size)) return NULL;

    ctx = (h3_dit_w4a8 *)calloc(1, sizeof(*ctx));
    if (!ctx) {
        fail(error, error_size, "out of memory opening dit nvfp4");
        return NULL;
    }
    ctx->cuda = cuda;
    ctx->store = NULL;
    ctx->int8_store = NULL;
    ctx->nvfp4_store = store;
    ctx->act_quant = H3_W4A8_ACT_QUANT_PER_ROW;
    ctx->group_size = 0;
    ctx->convrot_gs = 0; /* pure NVFP4 store: no INT8 ConvRot metadata;
                          * overlay open (W4A8+sidecar) sets gs from INT8 */

    if (cfg)
        local = *cfg;
    else
        h3_w4a8_cache_config_init(&local);

    ctx->cache =
        h3_w4a8_cache_open_nvfp4(cuda, store, &local, detail, sizeof(detail));
    if (!ctx->cache) {
        snprintf(error, error_size, "NVFP4 cache open: %s", detail);
        free(ctx);
        return NULL;
    }
    ctx->token_reduction = 0;
    ctx->tr_begin = 4;
    ctx->tr_end = 30;
    ctx->tr_early_steps = 10;
    ctx->tr_early_end = 40;
    ctx->tr_scale = 1.0f;
    ctx->nvfp4_on = env_flag_is_one("H3_DIT_NVFP4");
    ctx->nvfp4_keep_int8 = ctx->nvfp4_on ? nvfp4_parse_keep_int8() : 0u;
    ctx->nvfp4_fp4_mask =
        ctx->nvfp4_on ? ((~ctx->nvfp4_keep_int8) & 0xFu) : 0u;
    if (ctx->nvfp4_on) {
        nvfp4_log_keep_int8(ctx->nvfp4_keep_int8);
        h3_w4a8_cache_set_nvfp4_int8_skip(ctx->cache, 1);
        fprintf(stderr, "dit_nvfp4: H3_DIT_NVFP4=1 — FP4 cuBLASLt GEMM track "
                        "(INT8 weights load lazily only on FP4 dispatch "
                        "miss)\n");
        if (ctx->convrot_gs > 0 && !nvfp4_skip_act_convrot())
            fprintf(stderr,
                    "dit_nvfp4: act_convrot policy=ON gs=%d "
                    "(per-linear; chain auto-skipped)\n",
                    ctx->convrot_gs);
        else if (ctx->convrot_gs > 0)
            fprintf(stderr,
                    "dit_nvfp4: act_convrot policy=SKIP "
                    "(H3_DIT_NVFP4_SKIP_ACT_CONVROT=1)\n");
        fflush(stderr);
    }
    ctx->nvfp4_chain_on = h3_dit_nvfp4_chain_on();
    if (ctx->nvfp4_chain_on) {
        fprintf(stderr, "dit_nvfp4: H3_DIT_NVFP4_CHAIN=1 — FC1->FC2 chained "
                        "direct-output (CUTLASS sm_120a)\n");
        fflush(stderr);
    }
    return ctx;
}

static void dit_rope_memo_free_all(void);

void h3_dit_w4a8_free(h3_dit_w4a8 *ctx) {
    unsigned i;
    if (!ctx) return;
    h3_w4a8_cache_free(ctx->cache);
    free(ctx->x_rot);
    free(ctx->x_i8);
    free(ctx->act_scales);
    destroy_block_graph(ctx);
    destroy_step_graph(ctx);
    if (ctx->d_block_ws) cudaFree(ctx->d_block_ws);
    if (ctx->d_video_patch_w) cudaFree(ctx->d_video_patch_w);
    if (ctx->d_video_patch_b) cudaFree(ctx->d_video_patch_b);
    if (ctx->d_final_video_w) cudaFree(ctx->d_final_video_w);
    if (ctx->d_final_video_b) cudaFree(ctx->d_final_video_b);
    if (ctx->d_final_norm) cudaFree(ctx->d_final_norm);
    if (ctx->d_audio_patch_w) cudaFree(ctx->d_audio_patch_w);
    if (ctx->d_audio_patch_b) cudaFree(ctx->d_audio_patch_b);
    if (ctx->d_final_audio_w) cudaFree(ctx->d_final_audio_w);
    if (ctx->d_final_audio_b) cudaFree(ctx->d_final_audio_b);
    if (ctx->d_final_ws) cudaFree(ctx->d_final_ws);
    if (ctx->d_fwd_act_a) cudaFree(ctx->d_fwd_act_a);
    if (ctx->d_fwd_act_b) cudaFree(ctx->d_fwd_act_b);
    if (ctx->d_tr_original) cudaFree(ctx->d_tr_original);
    if (ctx->d_tr_baseline) cudaFree(ctx->d_tr_baseline);
    if (ctx->d_tr_reduced_a) cudaFree(ctx->d_tr_reduced_a);
    if (ctx->d_tr_reduced_b) cudaFree(ctx->d_tr_reduced_b);
    if (ctx->d_rope_cos) cudaFree(ctx->d_rope_cos);
    if (ctx->d_rope_sin) cudaFree(ctx->d_rope_sin);
    {
        unsigned b, p;
        for (b = 0; b < H3_DIT_W4A8_N_BLOCKS; b++) {
            for (p = 0; p < 4; p++) {
                if (ctx->d_lora_A[b][p]) cudaFree(ctx->d_lora_A[b][p]);
                if (ctx->d_lora_B[b][p]) cudaFree(ctx->d_lora_B[b][p]);
            }
        }
    }
    if (ctx->d_lora_t) cudaFree(ctx->d_lora_t);
    h3_dit_w4a8_clean_kv_free(ctx);
    free(ctx->adaln_audio_scratch);
    free(ctx->adaln_visual_cond_scratch);
    free(ctx->adaln_audio_cond_scratch);
    for (i = 0; i < H3_DIT_W4A8_N_BLOCKS; i++) {
        if (ctx->aux_valid[i]) h3_dit_w4a8_block_aux_free(&ctx->aux_cache[i]);
    }
    {
        unsigned b, p;
        for (b = 0; b < H3_DIT_W4A8_N_BLOCKS; b++)
            for (p = 0; p < 4; p++)
                if (ctx->nvfp4_w[b][p])
                    h3_cuda_nvfp4_weight_free(ctx->cuda, ctx->nvfp4_w[b][p]);
    }
    if (ctx->d_nvfp4_scratch) cudaFree(ctx->d_nvfp4_scratch);
    if (ctx->d_nvfp4_chain_scratch) cudaFree(ctx->d_nvfp4_chain_scratch);
    if (ctx->d_nvfp4_rot) cudaFree(ctx->d_nvfp4_rot);
    if (ctx->nvfp4_ready) h3_st_free_header(&ctx->nvfp4_hdr);
    free(ctx);
    dit_rope_memo_free_all();
}

static const h3_lora_pair *dit_lora_pair(const h3_lora_block *b, int which) {
    if (!b) return NULL;
    if (which == 0) return &b->qkv;
    if (which == 1) return &b->out;
    if (which == 2) return &b->fc1;
    if (which == 3) return &b->fc2;
    return NULL;
}

static void dit_lora_gpu_clear(h3_dit_w4a8 *ctx) {
    unsigned b, p;
    if (!ctx) return;
    for (b = 0; b < H3_DIT_W4A8_N_BLOCKS; b++) {
        for (p = 0; p < 4; p++) {
            if (ctx->d_lora_A[b][p]) {
                cudaFree(ctx->d_lora_A[b][p]);
                ctx->d_lora_A[b][p] = NULL;
            }
            if (ctx->d_lora_B[b][p]) {
                cudaFree(ctx->d_lora_B[b][p]);
                ctx->d_lora_B[b][p] = NULL;
            }
            ctx->lora_r[b][p] = ctx->lora_n[b][p] = ctx->lora_k[b][p] = 0;
        }
    }
    if (ctx->d_lora_t) {
        cudaFree(ctx->d_lora_t);
        ctx->d_lora_t = NULL;
    }
    ctx->d_lora_t_cap = 0;
    ctx->lora_max_r = 0;
    ctx->lora_f16 = 0;
    if (ctx->lora_ev0 && ctx->lora_ev1) {
        int i;
        for (i = 0; i < ctx->lora_ev_cap; i++) {
            if (ctx->lora_ev0[i]) cudaEventDestroy(ctx->lora_ev0[i]);
            if (ctx->lora_ev1[i]) cudaEventDestroy(ctx->lora_ev1[i]);
        }
    }
    free(ctx->lora_ev0);
    free(ctx->lora_ev1);
    ctx->lora_ev0 = ctx->lora_ev1 = NULL;
    ctx->lora_ev_n = ctx->lora_ev_cap = 0;
    if (ctx->lora_x_ev)
        cudaEventDestroy(ctx->lora_x_ev);
    if (ctx->lora_done_ev)
        cudaEventDestroy(ctx->lora_done_ev);
    ctx->lora_x_ev = ctx->lora_done_ev = NULL;
    ctx->lora = NULL;
}

int h3_dit_w4a8_set_lora(h3_dit_w4a8 *ctx, const h3_lora_set *lora, char *error,
                         size_t error_size) {
    unsigned b, p;
    int max_r = 0;
    size_t pre;
    if (!ctx) {
        fail(error, error_size, "dit set_lora: null ctx");
        return 0;
    }
    dit_lora_gpu_clear(ctx);
    if (!lora)
        return 1;
    if (ctx->cuda && !h3_cuda_bind_device(ctx->cuda, error, error_size))
        return 0;
    ctx->lora = lora;
    for (b = 0; b < H3_LORA_N_BLOCKS; b++) {
        for (p = 0; p < 4; p++) {
            const h3_lora_pair *pair = dit_lora_pair(&lora->blocks[b], (int)p);
            size_t a_n, b_n;
            if (!pair || !pair->A || !pair->B || pair->r < 1) {
                dit_lora_gpu_clear(ctx);
                fail(error, error_size, "dit set_lora: incomplete pair");
                return 0;
            }
            a_n = (size_t)pair->r * (size_t)pair->k;
            b_n = (size_t)pair->n * (size_t)pair->r;
            if (cudaMalloc((void **)&ctx->d_lora_A[b][p],
                           a_n * sizeof(float)) != cudaSuccess ||
                cudaMalloc((void **)&ctx->d_lora_B[b][p],
                           b_n * sizeof(float)) != cudaSuccess) {
                dit_lora_gpu_clear(ctx);
                fail(error, error_size, "dit set_lora: OOM uploading A/B");
                return 0;
            }
            if (cudaMemcpy(ctx->d_lora_A[b][p], pair->A, a_n * sizeof(float),
                           cudaMemcpyHostToDevice) != cudaSuccess ||
                cudaMemcpy(ctx->d_lora_B[b][p], pair->B, b_n * sizeof(float),
                           cudaMemcpyHostToDevice) != cudaSuccess) {
                dit_lora_gpu_clear(ctx);
                fail(error, error_size, "dit set_lora: H2D A/B failed");
                return 0;
            }
            ctx->lora_r[b][p] = pair->r;
            ctx->lora_n[b][p] = pair->n;
            ctx->lora_k[b][p] = pair->k;
            if (pair->r > max_r)
                max_r = pair->r;
        }
    }
    ctx->lora_max_r = max_r;
    if (h3_dit_w4a8_lora_f16()) {
        for (b = 0; b < H3_LORA_N_BLOCKS; b++) {
            for (p = 0; p < 4; p++) {
                size_t a_n = (size_t)ctx->lora_r[b][p] * (size_t)ctx->lora_k[b][p];
                size_t b_n = (size_t)ctx->lora_n[b][p] * (size_t)ctx->lora_r[b][p];
                void *d_h = NULL;
                char detail[256];
                if (cudaMalloc(&d_h, a_n * 2u) != cudaSuccess ||
                    !h3_cuda_f32_to_f16_dev(ctx->cuda, ctx->d_lora_A[b][p], d_h,
                                            a_n, detail, sizeof(detail)) ||
                    h3_cuda_device_sync_class(H3_SYNC_LIFECYCLE,
                                              "dit_set_lora_A_f16") !=
                        cudaSuccess) {
                    if (d_h) cudaFree(d_h);
                    dit_lora_gpu_clear(ctx);
                    fail(error, error_size, "dit set_lora: A f16");
                    return 0;
                }
                cudaFree(ctx->d_lora_A[b][p]);
                ctx->d_lora_A[b][p] = (float *)d_h;
                d_h = NULL;
                if (cudaMalloc(&d_h, b_n * 2u) != cudaSuccess ||
                    !h3_cuda_f32_to_f16_dev(ctx->cuda, ctx->d_lora_B[b][p], d_h,
                                            b_n, detail, sizeof(detail)) ||
                    h3_cuda_device_sync_class(H3_SYNC_LIFECYCLE,
                                              "dit_set_lora_B_f16") !=
                        cudaSuccess) {
                    if (d_h) cudaFree(d_h);
                    dit_lora_gpu_clear(ctx);
                    fail(error, error_size, "dit set_lora: B f16");
                    return 0;
                }
                cudaFree(ctx->d_lora_B[b][p]);
                ctx->d_lora_B[b][p] = (float *)d_h;
            }
        }
        ctx->lora_f16 = 1;
        fprintf(stderr, "dit_lora_gemm: f16w max_r=%d\n", max_r);
        fflush(stderr);
    }
    pre = 16384u * (size_t)max_r;
    if (cudaMalloc((void **)&ctx->d_lora_t, pre * sizeof(float)) != cudaSuccess) {
        dit_lora_gpu_clear(ctx);
        fail(error, error_size, "dit set_lora: OOM T workspace");
        return 0;
    }
    ctx->d_lora_t_cap = pre;
    ctx->lora_ev0 = (cudaEvent_t *)calloc((size_t)H3_LORA_EV_MAX, sizeof(cudaEvent_t));
    ctx->lora_ev1 = (cudaEvent_t *)calloc((size_t)H3_LORA_EV_MAX, sizeof(cudaEvent_t));
    ctx->lora_ev_n = 0;
    ctx->lora_ev_cap = 0;
    if (ctx->lora_ev0 && ctx->lora_ev1) {
        int i;
        for (i = 0; i < H3_LORA_EV_MAX; i++) {
            if (cudaEventCreate(&ctx->lora_ev0[i]) != cudaSuccess ||
                cudaEventCreate(&ctx->lora_ev1[i]) != cudaSuccess)
                break;
            ctx->lora_ev_cap = i + 1;
        }
    }
    if (cudaEventCreateWithFlags(&ctx->lora_x_ev, cudaEventDisableTiming) !=
            cudaSuccess ||
        cudaEventCreateWithFlags(&ctx->lora_done_ev, cudaEventDisableTiming) !=
            cudaSuccess) {
        dit_lora_gpu_clear(ctx);
        fail(error, error_size, "dit set_lora: overlap events");
        return 0;
    }
    return 1;
}

int h3_dit_w4a8_set_ulysses(h3_dit_w4a8 *ctx, h3_dit_w4a8 *const *rank_ctx,
                            const int *devices, int u,
                            const h3_dit_w4a8_sp_comm *comm, char *error,
                            size_t error_size) {
    int i;
    if (!ctx) {
        fail(error, error_size, "set_ulysses: null ctx");
        return 0;
    }
    memset(ctx->uly_ctx, 0, sizeof(ctx->uly_ctx));
    memset(ctx->uly_dev, 0, sizeof(ctx->uly_dev));
    memset(&ctx->uly_comm, 0, sizeof(ctx->uly_comm));
    ctx->uly_u = 0;
    if (u <= 1)
        return 1;
    if (u > 8) {
        fail(error, error_size, "set_ulysses: u>8");
        return 0;
    }
    if (!rank_ctx || !devices || !comm || !comm->cfg || !comm->bind ||
        !comm->a2a_qkv || !comm->a2a_attn_out) {
        fail(error, error_size, "set_ulysses: missing rank/comm");
        return 0;
    }
    if (rank_ctx[0] != ctx) {
        fail(error, error_size, "set_ulysses: rank_ctx[0] must be ctx");
        return 0;
    }
    for (i = 0; i < u; i++) {
        if (!rank_ctx[i]) {
            fail(error, error_size, "set_ulysses: null rank ctx");
            return 0;
        }
        ctx->uly_ctx[i] = rank_ctx[i];
        ctx->uly_dev[i] = devices[i];
    }
    ctx->uly_comm = *comm;
    ctx->uly_u = u;
    return 1;
}

static int dit_lora_residual_dev(h3_dit_w4a8 *ctx, unsigned block, int which,
                                 const float *d_x, float *d_y, size_t seq,
                                 float beta, char *error, size_t error_size) {
    int r, n, k;
    size_t need;
    char detail[256];
    float scale;
    int slot = -1;
    if (!ctx || !ctx->lora)
        return 1;
    if (block >= H3_DIT_W4A8_N_BLOCKS || which < 0 || which > 3) {
        fail(error, error_size, "lora residual: bad block/which");
        return 0;
    }
    r = ctx->lora_r[block][which];
    n = ctx->lora_n[block][which];
    k = ctx->lora_k[block][which];
    if (r < 1 || n < 1 || k < 1 || !ctx->d_lora_A[block][which] ||
        !ctx->d_lora_B[block][which]) {
        fail(error, error_size, "lora residual: missing GPU pair");
        return 0;
    }
    need = seq * (size_t)r;
    if (need > ctx->d_lora_t_cap) {
        if (h3_cuda_graph_capture(ctx->cuda)) {
            fail(error, error_size, "lora residual: T ws too small for graph");
            return 0;
        }
        if (ctx->cuda && !h3_cuda_bind_device(ctx->cuda, error, error_size))
            return 0;
        if (ctx->d_lora_t)
            cudaFree(ctx->d_lora_t);
        ctx->d_lora_t = NULL;
        if (cudaMalloc((void **)&ctx->d_lora_t, need * sizeof(float)) !=
            cudaSuccess) {
            ctx->d_lora_t_cap = 0;
            fail(error, error_size, "lora residual: OOM T");
            return 0;
        }
        ctx->d_lora_t_cap = need;
    }
    scale = ctx->lora->scale;
    if (ctx->lora_ev0 && ctx->lora_ev1 && ctx->lora_ev_n < ctx->lora_ev_cap) {
        cudaStream_t st = ctx->cuda
                               ? (cudaStream_t)h3_cuda_cublas_stream(ctx->cuda)
                               : (cudaStream_t)0;
        slot = ctx->lora_ev_n++;
        cudaEventRecord(ctx->lora_ev0[slot], st);
    }
    if (ctx->lora_f16) {
        if (!h3_cuda_gemm_f32in_f16w_f32out_dd_ex(
                ctx->cuda, d_x, ctx->d_lora_A[block][which], seq, (size_t)r,
                (size_t)k, ctx->d_lora_t, 1.0f, 0.0f, detail, sizeof(detail))) {
            snprintf(error, error_size, "lora XA^T f16: %s", detail);
            return 0;
        }
        if (!h3_cuda_gemm_f32in_f16w_f32out_dd_ex(
                ctx->cuda, ctx->d_lora_t, ctx->d_lora_B[block][which], seq,
                (size_t)n, (size_t)r, d_y, scale, beta, detail,
                sizeof(detail))) {
            snprintf(error, error_size, "lora TB^T f16: %s", detail);
            return 0;
        }
    } else if (h3_dit_w4a8_lora_tf32()) {
        static int tf32_logged;
        if (!tf32_logged) {
            tf32_logged = 1;
            fprintf(stderr, "dit_lora_gemm: tf32 r=%d\n", r);
            fflush(stderr);
        }
        if (!h3_cuda_gemm_f32_dd_tf32_ex(ctx->cuda, d_x,
                                         ctx->d_lora_A[block][which], seq,
                                         (size_t)r, (size_t)k, ctx->d_lora_t,
                                         1.0f, 0.0f, detail, sizeof(detail))) {
            snprintf(error, error_size, "lora XA^T tf32: %s", detail);
            return 0;
        }
        if (!h3_cuda_gemm_f32_dd_tf32_ex(
                ctx->cuda, ctx->d_lora_t, ctx->d_lora_B[block][which], seq,
                (size_t)n, (size_t)r, d_y, scale, beta, detail,
                sizeof(detail))) {
            snprintf(error, error_size, "lora TB^T tf32: %s", detail);
            return 0;
        }
    } else {
        if (!h3_cuda_gemm_f32_dd(ctx->cuda, d_x, ctx->d_lora_A[block][which],
                                 seq, (size_t)r, (size_t)k, ctx->d_lora_t, NULL,
                                 detail, sizeof(detail))) {
            snprintf(error, error_size, "lora XA^T: %s", detail);
            return 0;
        }
        if (!h3_cuda_gemm_f32_dd_ex(ctx->cuda, ctx->d_lora_t,
                                    ctx->d_lora_B[block][which], seq,
                                    (size_t)n, (size_t)r, d_y, scale, beta, NULL,
                                    detail, sizeof(detail))) {
            snprintf(error, error_size, "lora TB^T: %s", detail);
            return 0;
        }
    }
    if (slot >= 0) {
        cudaStream_t st = ctx->cuda
                               ? (cudaStream_t)h3_cuda_cublas_stream(ctx->cuda)
                               : (cudaStream_t)0;
        cudaEventRecord(ctx->lora_ev1[slot], st);
    }
    return 1;
}

/* 1 = LoRA running on side stream; 0 = serial; -1 = error. */
static int dit_lora_overlap_begin(h3_dit_w4a8 *ctx, char *error,
                                  size_t error_size) {
    void *side;
    static int logged;
    if (!ctx || !ctx->cuda || !h3_dit_lora_overlap_wanted())
        return 0;
    if (!ctx->lora_x_ev || !ctx->lora_done_ev)
        return 0;
    if (h3_cuda_graph_capture(ctx->cuda) || ctx->uly_u > 1)
        return 0;
    side = h3_cuda_stream(ctx->cuda, 1);
    if (!side) {
        if (!logged) {
            logged = 1;
            fprintf(stderr, "dit_lora_overlap: off reason=no_stream\n");
            fflush(stderr);
        }
        return 0;
    }
    if (!h3_cuda_event_record(h3_cuda_op_stream(ctx->cuda), ctx->lora_x_ev,
                              error, error_size))
        return -1;
    if (!h3_cuda_stream_wait_event(side, ctx->lora_x_ev, error, error_size))
        return -1;
    if (!h3_cuda_cublas_use_side(ctx->cuda, 1, error, error_size))
        return 0;
    if (!logged) {
        logged = 1;
        fprintf(stderr, "dit_lora_overlap: on side=1 cublas_side=1\n");
        fflush(stderr);
    }
    return 1;
}

static int dit_lora_overlap_finish(h3_dit_w4a8 *ctx, int ov, char *error,
                                   size_t error_size) {
    void *side;
    if (ov <= 0)
        return ov < 0 ? 0 : 1;
    side = h3_cuda_stream(ctx->cuda, 1);
    if (!h3_cuda_event_record(side, ctx->lora_done_ev, error, error_size)) {
        h3_cuda_cublas_use_side(ctx->cuda, 0, NULL, 0);
        return 0;
    }
    if (!h3_cuda_cublas_use_side(ctx->cuda, 0, error, error_size))
        return 0;
    return 1;
}

static int dit_lora_kick(h3_dit_w4a8 *ctx, unsigned block, int which,
                         const float *d_x, float *d_y, size_t seq, float beta,
                         int overlap, void **lora_ready, char *error,
                         size_t error_size) {
    int ov = 0;
    if (lora_ready)
        *lora_ready = NULL;
    if (overlap)
        ov = dit_lora_overlap_begin(ctx, error, error_size);
    if (ov < 0)
        return 0;
    if (!dit_lora_residual_dev(ctx, block, which, d_x, d_y, seq, beta, error,
                               error_size)) {
        dit_lora_overlap_finish(ctx, ov, error, error_size);
        return 0;
    }
    if (!dit_lora_overlap_finish(ctx, ov, error, error_size))
        return 0;
    if (ov == 1 && lora_ready)
        *lora_ready = (void *)ctx->lora_done_ev;
    return 1;
}

static double dit_lora_harvest_ms(h3_dit_w4a8 *ctx) {
    int i;
    double sum = 0.0;
    if (!ctx || ctx->lora_ev_n <= 0 || !ctx->lora_ev0 || !ctx->lora_ev1)
        return 0.0;
    if (cudaEventSynchronize(ctx->lora_ev1[ctx->lora_ev_n - 1]) != cudaSuccess) {
        ctx->lora_ev_n = 0;
        return 0.0;
    }
    for (i = 0; i < ctx->lora_ev_n; i++) {
        float ms = 0.0f;
        if (cudaEventElapsedTime(&ms, ctx->lora_ev0[i], ctx->lora_ev1[i]) ==
            cudaSuccess)
            sum += (double)ms;
    }
    ctx->lora_ev_n = 0;
    return sum;
}

static void dit_lora_log_ms(h3_dit_w4a8 *ctx, h3_dit_w4a8_layers_stats *stats) {
    double ms;
    if (!ctx || !ctx->lora)
        return;
    ms = dit_lora_harvest_ms(ctx);
    if (stats)
        stats->lora_ms = ms;
    fprintf(stderr, "dit_lora: scale=%.4g gemm_epi=%d ms=%.1f\n",
            ctx->lora->scale, g_dit_gemm_epi, ms);
    fflush(stderr);
}

void h3_dit_w4a8_set_token_reduction(h3_dit_w4a8 *ctx, int enabled) {
    unsigned begin = 4, end = 30, early_steps = 10, early_end = 40;
    float scale = 1.0f;
    const char *range, *early, *scale_text;
    if (!ctx) return;
    ctx->token_reduction = enabled ? 1 : 0;
    if (!ctx->token_reduction) return;
    range = getenv("H3_TOKEN_REDUCTION_BLOCKS");
    if (range && *range) {
        unsigned long b = 0, e = 0;
        char *mid = NULL, *tail = NULL;
        b = strtoul(range, &mid, 10);
        if (mid && *mid == ':') e = strtoul(mid + 1, &tail, 10);
        if (mid && *mid == ':' && tail && !*tail && b < e &&
            e <= H3_DIT_W4A8_N_BLOCKS) {
            begin = (unsigned)b;
            end = (unsigned)e;
        }
    }
    early = getenv("H3_TOKEN_REDUCTION_EARLY");
    if (early && *early) {
        if (!strcmp(early, "0")) {
            early_steps = 0;
            early_end = end;
        } else {
            unsigned long s = 0, e = 0;
            char *mid = NULL, *tail = NULL;
            s = strtoul(early, &mid, 10);
            if (mid && *mid == ':') e = strtoul(mid + 1, &tail, 10);
            if (mid && *mid == ':' && tail && !*tail && s > 0 && e > end &&
                e <= H3_DIT_W4A8_N_BLOCKS) {
                early_steps = (unsigned)s;
                early_end = (unsigned)e;
            }
        }
    } else if (end >= 40) {
        early_steps = 0;
        early_end = end;
    }
    scale_text = getenv("H3_TOKEN_REDUCTION_SCALE");
    if (scale_text && *scale_text) {
        char *tail = NULL;
        float s = strtof(scale_text, &tail);
        if (tail != scale_text && (!tail || !*tail) && isfinite(s) && s >= 0.0f &&
            s <= 2.0f)
            scale = s;
    }
    ctx->tr_begin = begin;
    ctx->tr_end = end;
    ctx->tr_early_steps = early_steps;
    ctx->tr_early_end = early_end;
    ctx->tr_scale = scale;
}

int h3_dit_w4a8_token_reduction_enabled(const h3_dit_w4a8 *ctx) {
    return ctx && ctx->token_reduction;
}

void h3_dit_w4a8_set_denoise_step(h3_dit_w4a8 *ctx, int step) {
    if (ctx) ctx->denoise_step = step < 0 ? 0 : step;
    h3_cuda_sol_cute_set_step(step);
}

void h3_dit_w4a8_clean_kv_free(h3_dit_w4a8 *ctx) {
    if (!ctx) return;
    h3_cuda_sdpa_clear_kvext();
    if (ctx->d_ckv_k) cudaFree(ctx->d_ckv_k);
    if (ctx->d_ckv_v) cudaFree(ctx->d_ckv_v);
    if (ctx->d_ckv_idx) cudaFree(ctx->d_ckv_idx);
    ctx->d_ckv_k = NULL;
    ctx->d_ckv_v = NULL;
    ctx->d_ckv_idx = NULL;
    ctx->ckv_on = 0;
    ctx->ckv_mode = 0;
    ctx->ckv_max = 0;
    ctx->ckv_n = 0;
    ctx->ckv_save_n = 0;
    ctx->ckv_idx_cap = 0;
}

int h3_dit_w4a8_clean_kv_enable(h3_dit_w4a8 *ctx, size_t max_tokens,
                                char *error, size_t error_size) {
    size_t layer_bytes, idx_bytes;
    cudaError_t st;
    if (!ctx || max_tokens == 0) {
        snprintf(error, error_size, "clean-kv enable: bad args");
        return 0;
    }
    h3_dit_w4a8_clean_kv_free(ctx);
    destroy_block_graph(ctx);
    layer_bytes = max_tokens * (size_t)H3_DIT_W4A8_HEADS *
                  (size_t)H3_DIT_W4A8_HEAD_DIM * sizeof(uint16_t);
    layer_bytes *= (size_t)H3_DIT_W4A8_N_BLOCKS;
    idx_bytes = max_tokens * sizeof(uint32_t);
    st = cudaMalloc(&ctx->d_ckv_k, layer_bytes);
    if (st != cudaSuccess) {
        snprintf(error, error_size, "clean-kv K cudaMalloc %zu: %s",
                 layer_bytes, cudaGetErrorString(st));
        h3_dit_w4a8_clean_kv_free(ctx);
        return 0;
    }
    st = cudaMalloc(&ctx->d_ckv_v, layer_bytes);
    if (st != cudaSuccess) {
        snprintf(error, error_size, "clean-kv V cudaMalloc %zu: %s",
                 layer_bytes, cudaGetErrorString(st));
        h3_dit_w4a8_clean_kv_free(ctx);
        return 0;
    }
    st = cudaMalloc((void **)&ctx->d_ckv_idx, idx_bytes);
    if (st != cudaSuccess) {
        snprintf(error, error_size, "clean-kv idx cudaMalloc: %s",
                 cudaGetErrorString(st));
        h3_dit_w4a8_clean_kv_free(ctx);
        return 0;
    }
    cudaMemset(ctx->d_ckv_k, 0, layer_bytes);
    cudaMemset(ctx->d_ckv_v, 0, layer_bytes);
    ctx->ckv_idx_cap = max_tokens;
    ctx->ckv_max = max_tokens;
    ctx->ckv_n = 0;
    ctx->ckv_on = 1;
    ctx->ckv_mode = 0;
    fprintf(stderr,
            "[h3] layer_kv enable max_tokens=%zu bytes_kv=%zu (Q unchanged; "
            "K/V hist only)\n",
            max_tokens, layer_bytes * 2u);
    return 1;
}

int h3_dit_w4a8_clean_kv_enabled(const h3_dit_w4a8 *ctx) {
    return ctx && ctx->ckv_on && ctx->d_ckv_k && ctx->d_ckv_v;
}

size_t h3_dit_w4a8_clean_kv_max_tokens(const h3_dit_w4a8 *ctx) {
    return ctx ? ctx->ckv_max : 0;
}

size_t h3_dit_w4a8_clean_kv_hist_tokens(const h3_dit_w4a8 *ctx) {
    return ctx ? ctx->ckv_n : 0;
}

int h3_dit_w4a8_clean_kv_begin(h3_dit_w4a8 *ctx, int mode,
                               const uint32_t *save_idx, size_t save_n,
                               size_t save_dst, char *error, size_t error_size) {
    if (!h3_dit_w4a8_clean_kv_enabled(ctx)) {
        snprintf(error, error_size, "clean-kv begin: not enabled");
        return 0;
    }
    if (mode != 1 && mode != 2) {
        snprintf(error, error_size, "clean-kv begin: mode %d", mode);
        return 0;
    }
    ctx->ckv_mode = mode;
    ctx->ckv_save_n = 0;
    ctx->ckv_save_dst = 0;
    if (mode == 2) {
        if (!save_idx || save_n == 0) {
            snprintf(error, error_size, "clean-kv commit needs save idx");
            return 0;
        }
        if (save_n + save_dst > ctx->ckv_max) {
            snprintf(error, error_size,
                     "clean-kv save %zu+%zu exceeds max %zu", save_dst, save_n,
                     ctx->ckv_max);
            return 0;
        }
        if (save_n > ctx->ckv_idx_cap) {
            snprintf(error, error_size, "clean-kv idx cap %zu < %zu",
                     ctx->ckv_idx_cap, save_n);
            return 0;
        }
        if (cudaMemcpy(ctx->d_ckv_idx, save_idx, save_n * sizeof(uint32_t),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            snprintf(error, error_size, "clean-kv idx H2D failed");
            return 0;
        }
        ctx->ckv_save_n = save_n;
        ctx->ckv_save_dst = save_dst;
    }
    return 1;
}

void h3_dit_w4a8_clean_kv_finish(h3_dit_w4a8 *ctx, size_t hist_n) {
    if (!ctx) return;
    h3_cuda_sdpa_clear_kvext();
    if (ctx->ckv_mode == 2 && hist_n <= ctx->ckv_max)
        ctx->ckv_n = hist_n;
    ctx->ckv_mode = 0;
    ctx->ckv_save_n = 0;
}

void h3_dit_w4a8_set_text_mod_tags(h3_dit_w4a8 *ctx, const uint8_t *tags,
                                   size_t n) {
    if (!ctx) return;
    ctx->text_mod_tags = tags;
    ctx->text_mod_tag_n = tags ? n : 0;
}

static void dit_aux_log_disk_loads(h3_dit_w4a8 *ctx) {
    const char *mode;
    if (!ctx || ctx->aux_disk_logged) return;
    ctx->aux_disk_logged = 1;
    mode = ctx->aux_prestage_mode == 1
               ? "all"
               : (ctx->aux_prestage_mode == 2 ? "dual_bank" : "unset");
    fprintf(stderr,
            "dit_aux_cache: disk_loads=%u prestage=%s "
            "(expect ≈unique blocks once, not blocks×steps)\n",
            ctx->aux_disk_loads, mode);
    fflush(stderr);
    if (ctx->int8_store || ctx->nvfp4_store)
        h3_w4a8_cache_log_int8_host(ctx->cache);
}

/* Lazily load + cache block aux (avoids per-layer safetensors+BF16 convert). */
static const h3_dit_w4a8_block_aux *dit_aux_cached(h3_dit_w4a8 *ctx,
                                                    unsigned block,
                                                    char *error,
                                                    size_t error_size) {
    if (!ctx || block >= H3_DIT_W4A8_N_BLOCKS) {
        fail(error, error_size, "invalid aux cache block");
        return NULL;
    }
    if (!ctx->aux_valid[block]) {
        const h3_st_header *hdr = NULL;
        if (ctx->int8_store)
            hdr = h3_dit_int8_store_header(ctx->int8_store);
        else if (ctx->nvfp4_store)
            hdr = &ctx->nvfp4_store->header;
        else if (ctx->store)
            hdr = h3_w4a8_store_header(ctx->store);
        if (!hdr) {
            fail(error, error_size, "dit store missing for aux load");
            return NULL;
        }
        if (!h3_dit_w4a8_load_block_aux_hdr(hdr, block, &ctx->aux_cache[block],
                                            error, error_size))
            return NULL;
        ctx->aux_valid[block] = 1;
        ctx->aux_disk_loads++;
    }
    return &ctx->aux_cache[block];
}

const h3_dit_w4a8_block_aux *h3_dit_w4a8_cached_block_aux(
    h3_dit_w4a8 *ctx, unsigned block, char *error, size_t error_size) {
    return dit_aux_cached(ctx, block, error, error_size);
}

static int ensure_block_ws(h3_dit_w4a8 *ctx, size_t bytes, char *error,
                           size_t error_size) {
    void *fresh = NULL;
    size_t alloc;
    const size_t slack = (size_t)64 << 20;
    if (!ctx) {
        fail(error, error_size, "dit ctx NULL");
        return 0;
    }
    if (bytes == 0) return 1;
    if (ctx->d_block_ws && ctx->d_block_ws_bytes >= bytes) return 1;
    /* The step graph hard-codes d_block_ws-carved pointers. */
    destroy_step_graph(ctx);
    if (ctx->cuda && !h3_cuda_bind_device(ctx->cuda, error, error_size))
        return 0;
    /* Allocate first, free old only on success. Free-then-malloc after
     * cuDNN graphs were built fragmented cafe VRAM and triggered
     * make_room window 29→16. 64MiB slack absorbs seq-pad growth. */
    alloc = bytes;
    if (bytes <= (size_t)-1 - slack) alloc = bytes + slack;
    if (cudaMalloc(&fresh, alloc) != cudaSuccess) {
        /* Grow-in-place: alloc-first can fail while an older smaller WS still
         * occupies the only large contiguous hole (taomate overlap 12304→14346).
         * Free old then retry once. Do not make_room here (frozen INT8 window). */
        if (ctx->d_block_ws) {
            cudaFree(ctx->d_block_ws);
            ctx->d_block_ws = NULL;
            ctx->d_block_ws_bytes = 0;
            if (cudaMalloc(&fresh, alloc) == cudaSuccess) {
                ctx->d_block_ws = fresh;
                ctx->d_block_ws_bytes = alloc;
                return 1;
            }
        }
        fail(error, error_size, "cudaMalloc block workspace failed");
        return 0;
    }
    if (ctx->d_block_ws) cudaFree(ctx->d_block_ws);
    ctx->d_block_ws = fresh;
    ctx->d_block_ws_bytes = alloc;
    return 1;
}

static int ensure_fwd_act(h3_dit_w4a8 *ctx, size_t seq, char *error,
                          size_t error_size) {
    size_t need;
    if (!ctx || seq == 0) {
        fail(error, error_size, "ensure_fwd_act bad args");
        return 0;
    }
    need = seq * (size_t)H3_DIT_W4A8_HIDDEN;
    if (ctx->d_fwd_act_a && ctx->d_fwd_act_b && ctx->d_fwd_act_cap >= need)
        return 1;
    destroy_step_graph(ctx);
    if (ctx->d_fwd_act_a) {
        cudaFree(ctx->d_fwd_act_a);
        ctx->d_fwd_act_a = NULL;
    }
    if (ctx->d_fwd_act_b) {
        cudaFree(ctx->d_fwd_act_b);
        ctx->d_fwd_act_b = NULL;
    }
    ctx->d_fwd_act_cap = 0;
    if (cudaMalloc((void **)&ctx->d_fwd_act_a, need * sizeof(float)) !=
            cudaSuccess ||
        cudaMalloc((void **)&ctx->d_fwd_act_b, need * sizeof(float)) !=
            cudaSuccess) {
        if (ctx->d_fwd_act_a) cudaFree(ctx->d_fwd_act_a);
        ctx->d_fwd_act_a = NULL;
        fail(error, error_size, "cudaMalloc fwd act failed");
        return 0;
    }
    ctx->d_fwd_act_cap = need;
    return 1;
}

static int ensure_tr_ws(h3_dit_w4a8 *ctx, size_t full_seq, size_t reduced_seq,
                        size_t baseline_rows, char *error, size_t error_size) {
    size_t full_n = full_seq * (size_t)H3_DIT_W4A8_HIDDEN;
    size_t red_n = reduced_seq * (size_t)H3_DIT_W4A8_HIDDEN;
    size_t base_n = baseline_rows * (size_t)H3_DIT_W4A8_HIDDEN;
    if (!ctx || full_seq == 0 || reduced_seq == 0) {
        fail(error, error_size, "ensure_tr_ws bad args");
        return 0;
    }
    if (!ctx->d_tr_original || ctx->d_tr_full_cap < full_n) {
        if (ctx->d_tr_original) cudaFree(ctx->d_tr_original);
        ctx->d_tr_original = NULL;
        ctx->d_tr_full_cap = 0;
        if (cudaMalloc((void **)&ctx->d_tr_original, full_n * sizeof(float)) !=
            cudaSuccess) {
            fail(error, error_size, "cudaMalloc tr original failed");
            return 0;
        }
        ctx->d_tr_full_cap = full_n;
    }
    if (baseline_rows > 0 &&
        (!ctx->d_tr_baseline || ctx->d_tr_base_cap < base_n)) {
        if (ctx->d_tr_baseline) cudaFree(ctx->d_tr_baseline);
        ctx->d_tr_baseline = NULL;
        ctx->d_tr_base_cap = 0;
        if (cudaMalloc((void **)&ctx->d_tr_baseline, base_n * sizeof(float)) !=
            cudaSuccess) {
            fail(error, error_size, "cudaMalloc tr baseline failed");
            return 0;
        }
        ctx->d_tr_base_cap = base_n;
    }
    if (!ctx->d_tr_reduced_a || !ctx->d_tr_reduced_b ||
        ctx->d_tr_reduced_cap < red_n) {
        if (ctx->d_tr_reduced_a) cudaFree(ctx->d_tr_reduced_a);
        if (ctx->d_tr_reduced_b) cudaFree(ctx->d_tr_reduced_b);
        ctx->d_tr_reduced_a = NULL;
        ctx->d_tr_reduced_b = NULL;
        ctx->d_tr_reduced_cap = 0;
        if (cudaMalloc((void **)&ctx->d_tr_reduced_a, red_n * sizeof(float)) !=
                cudaSuccess ||
            cudaMalloc((void **)&ctx->d_tr_reduced_b, red_n * sizeof(float)) !=
                cudaSuccess) {
            if (ctx->d_tr_reduced_a) cudaFree(ctx->d_tr_reduced_a);
            ctx->d_tr_reduced_a = NULL;
            fail(error, error_size, "cudaMalloc tr reduced failed");
            return 0;
        }
        ctx->d_tr_reduced_cap = red_n;
    }
    return 1;
}

h3_w4a8_cache *h3_dit_w4a8_cache(h3_dit_w4a8 *ctx) {
    return ctx ? ctx->cache : NULL;
}

void h3_dit_w4a8_release_block_workspace(h3_dit_w4a8 *ctx) {
    if (!ctx || !ctx->d_block_ws)
        return;
    destroy_step_graph(ctx);
    cudaFree(ctx->d_block_ws);
    ctx->d_block_ws = NULL;
    ctx->d_block_ws_bytes = 0;
}

/* Keep in sync with h3_dit_w4a8_layers_forward_f32 all-aux `need`. */
static size_t layers_dev_chain_ws_bytes(const h3_dit_w4a8 *ctx, size_t seq,
                                        unsigned n_blocks, int use_ext_act) {
    const size_t hidden = H3_DIT_W4A8_HIDDEN;
    const size_t inner = H3_DIT_W4A8_INNER;
    const size_t ffn = H3_DIT_W4A8_FFN;
    const size_t slots = H3_DIT_W4A8_SLOTS;
    const size_t mod_rows = dit_adaln_mod_rows(ctx);
    const size_t rope_half = (size_t)H3_DIT_W4A8_ROPE_HALF;
    const size_t mod_elems = mod_rows * slots * hidden;
    const size_t head_dim = H3_DIT_W4A8_HEAD_DIM;
    const size_t wide = ffn * 2;
    const size_t aux_stride = mod_elems + hidden * 2 + head_dim * 2;
    const size_t act_bytes =
        use_ext_act ? 0 : (seq * hidden * sizeof(float) * 2);
    size_t need = 0;
    if (n_blocks < 1)
        n_blocks = 1;
    if (n_blocks > (unsigned)H3_DIT_W4A8_N_BLOCKS)
        n_blocks = (unsigned)H3_DIT_W4A8_N_BLOCKS;
    need += act_bytes;
    need += H3_WS_MAP_BYTES(seq);
    need += seq * rope_half * sizeof(float) * 2;
    need += (size_t)n_blocks * aux_stride * sizeof(float);
    need += seq * hidden * sizeof(float) * 2;
    need += seq * inner * 3 * sizeof(float);
    need += seq * wide * sizeof(float);
    return need;
}

int h3_dit_w4a8_pregrow_layers_ws(h3_dit_w4a8 *ctx, size_t seq,
                                  unsigned n_blocks, char *error,
                                  size_t error_size) {
    size_t need;
    if (!ctx || seq == 0) {
        fail(error, error_size, "pregrow_layers_ws bad args");
        return 0;
    }
    if (!ensure_fwd_act(ctx, seq, error, error_size))
        return 0;
    need = layers_dev_chain_ws_bytes(ctx, seq, n_blocks, 1);
    if (!ensure_block_ws(ctx, need, error, error_size))
        return 0;
    printf("dit_pregrow_layers_ws: seq=%zu n_blocks=%u ws_giB=%.2f "
           "fwd_act_giB=%.2f\n",
           seq, n_blocks < 1 ? 1u : (n_blocks > (unsigned)H3_DIT_W4A8_N_BLOCKS
                                         ? (unsigned)H3_DIT_W4A8_N_BLOCKS
                                         : n_blocks),
           (double)ctx->d_block_ws_bytes / (1024.0 * 1024.0 * 1024.0),
           (double)(seq * (size_t)H3_DIT_W4A8_HIDDEN * sizeof(float) * 2) /
               (1024.0 * 1024.0 * 1024.0));
    fflush(stdout);
    return 1;
}

h3_cuda *h3_dit_w4a8_cuda(h3_dit_w4a8 *ctx) {
    return ctx ? ctx->cuda : NULL;
}

const h3_w4a8_store *h3_dit_w4a8_store(const h3_dit_w4a8 *ctx) {
    return ctx ? ctx->store : NULL;
}

int h3_dit_w4a8_convrot_groupsize(const h3_dit_w4a8 *ctx) {
    return ctx ? ctx->convrot_gs : 0;
}

int h3_dit_w4a8_act_f16(void) {
    const char *e = getenv("H3_DIT_ACT_F16");
    if (e && e[0] == '0' && e[1] == '\0') return 0;
    return 1;
}

int h3_dit_w4a8_fused_epilogue(void) {
    return h3_dit_fused_epilogue_wanted();
}

int h3_dit_w4a8_lora_f16(void) {
    const char *e = getenv("H3_DIT_LORA_F16");
    return e && e[0] == '1' && e[1] == '\0';
}

int h3_dit_w4a8_lora_tf32(void) {
    const char *e = getenv("H3_DIT_LORA_TF32");
    return e && e[0] == '1' && e[1] == '\0';
}

int h3_dit_w4a8_lora_overlap(void) {
    return h3_dit_lora_overlap_wanted();
}

int h3_dit_w4a8_group_size(const h3_dit_w4a8 *ctx) {
    return ctx ? ctx->group_size : 0;
}

static void copy_layer_stats(h3_dit_w4a8_forward_stats *st,
                             const h3_dit_w4a8_layers_stats *lst) {
    if (!st || !lst) return;
    st->layers_ms = lst->total_ms;
    st->layers_prefetch_ms = lst->prefetch_ms;
    st->layers_h2d_ms = lst->h2d_ms;
    st->layers_compute_ms = lst->compute_ms;
    st->layers_d2h_ms = lst->d2h_ms;
    st->layers_prefetch_wait_ms = lst->prefetch_wait_ms;
}

/* U=1: next-block only. Full-window prefetch drains H2D before compute. */
static int prefetch_next_u1(h3_dit_w4a8 *ctx, unsigned next_block,
                            unsigned last_block, char *error,
                            size_t error_size) {
    double ignore = 0.0;
    if (!ctx || next_block >= last_block) return 1;
    return h3_dit_w4a8_prefetch(ctx, next_block, 1, &ignore, error,
                                error_size);
}

int h3_dit_w4a8_prefetch(h3_dit_w4a8 *ctx, unsigned first_block,
                         unsigned n_blocks, double *elapsed_ms_out,
                         char *error, size_t error_size) {
    int ok;
    if (!ctx || !ctx->cache) {
        fail(error, error_size, "dit w4a8 is NULL");
        return 0;
    }
    ok = h3_w4a8_cache_prefetch(ctx->cache, first_block, n_blocks,
                                elapsed_ms_out, error, error_size);
    if (ok && elapsed_ms_out && *elapsed_ms_out > 0.0)
        h3_cuda_sol_profile_add_unpack(*elapsed_ms_out);
    return ok;
}

int h3_dit_w4a8_stage_host(h3_dit_w4a8 *ctx, unsigned first_block,
                           unsigned n_blocks, double *elapsed_ms_out,
                           char *error, size_t error_size) {
    if (!ctx || !ctx->cache) {
        fail(error, error_size, "dit w4a8 is NULL");
        return 0;
    }
    return h3_w4a8_cache_stage_host_int8(ctx->cache, first_block, n_blocks,
                                         elapsed_ms_out, error, error_size);
}

int h3_dit_w4a8_get_cache_stats(h3_dit_w4a8 *ctx, h3_w4a8_cache_stats *stats,
                                char *error, size_t error_size) {
    if (!ctx || !ctx->cache) {
        fail(error, error_size, "dit w4a8 is NULL");
        return 0;
    }
    return h3_w4a8_cache_get_stats(ctx->cache, stats, error, error_size);
}

static const h3_cuda_w4a8_weight *get_weight(h3_dit_w4a8 *ctx, unsigned block,
                                             h3_w4a8_linear_kind kind,
                                             char *error, size_t error_size) {
    const h3_cuda_w4a8_weight *w;
    if (!ctx || !ctx->cache) {
        fail(error, error_size, "dit w4a8 is NULL");
        return NULL;
    }
    w = h3_w4a8_cache_get_linear(ctx->cache, block, kind, error, error_size);
    return w;
}

int h3_dit_w4a8_linear_shape(h3_dit_w4a8 *ctx, unsigned block,
                             h3_w4a8_linear_kind kind, size_t *n_out,
                             size_t *k_out, char *error, size_t error_size) {
    const h3_cuda_w4a8_weight *w = get_weight(ctx, block, kind, error,
                                              error_size);
    if (!w) return 0;
    if (n_out) *n_out = h3_cuda_w4a8_weight_n(w);
    if (k_out) *k_out = h3_cuda_w4a8_weight_k(w);
    return 1;
}

static int ensure_scratch(h3_dit_w4a8 *ctx, size_t m, size_t k, char *error,
                          size_t error_size) {
    size_t mk = m * k;
    if (mk > ctx->cap_mk) {
        float *rot = (float *)realloc(ctx->x_rot, mk * sizeof(float));
        int8_t *qi;
        if (!rot) {
            fail(error, error_size, "out of memory (dit w4a8 scratch rot)");
            return 0;
        }
        ctx->x_rot = rot;
        qi = (int8_t *)realloc(ctx->x_i8, mk);
        if (!qi) {
            fail(error, error_size, "out of memory (dit w4a8 scratch i8)");
            return 0;
        }
        ctx->x_i8 = qi;
        ctx->cap_mk = mk;
    }
    if (m > ctx->cap_m) {
        float *sc = (float *)realloc(ctx->act_scales, m * sizeof(float));
        if (!sc) {
            fail(error, error_size, "out of memory (dit w4a8 scratch M)");
            return 0;
        }
        ctx->act_scales = sc;
        ctx->cap_m = m;
    }
    return 1;
}

int h3_dit_w4a8_linear_f32(h3_dit_w4a8 *ctx, unsigned block,
                           h3_w4a8_linear_kind kind, const float *x_f32,
                           size_t m, float *y_f32,
                           h3_dit_w4a8_linear_stats *stats, char *error,
                           size_t error_size) {
    const h3_cuda_w4a8_weight *w;
    size_t n, k;
    clock_t t0, t1;
    char detail[384];
    int per_tensor;

    if (stats) memset(stats, 0, sizeof(*stats));
    if (!x_f32 || !y_f32 || m == 0) {
        fail(error, error_size, "invalid dit w4a8 linear args");
        return 0;
    }

    w = get_weight(ctx, block, kind, error, error_size);
    if (!w) return 0;
    n = h3_cuda_w4a8_weight_n(w);
    k = h3_cuda_w4a8_weight_k(w);
    if (n == 0 || k == 0) {
        fail(error, error_size, "resident weight has empty shape");
        return 0;
    }
    if (!ensure_scratch(ctx, m, k, error, error_size)) return 0;

    /* Prefer device ConvRot + per-row INT8 quant (avoids host rotate/quant). */
    if (!env_flag_is_one("H3_DIT_W4A8_FORCE_CPU_ACT") &&
        ctx->act_quant == H3_W4A8_ACT_QUANT_PER_ROW) {
        double prep = 0.0;
        if (!h3_cuda_gemm_i8_dequant_f32_w_act(
                ctx->cuda, w, x_f32, ctx->convrot_gs, m, y_f32, &prep,
                stats ? &stats->gemm_ms : NULL, stats ? stats->backend : NULL,
                stats ? sizeof(stats->backend) : 0, detail, sizeof(detail))) {
            snprintf(error, error_size, "gemm_act: %s", detail);
            return 0;
        }
        if (stats) {
            stats->convrot_ms = prep;
            stats->quant_ms = 0.0;
            stats->m = m;
            stats->n = n;
            stats->k = k;
        }
        return 1;
    }

    t0 = clock();
    if (!h3_w4a8_convrot_act_f32(x_f32, m, k, ctx->convrot_gs, ctx->x_rot,
                                 error, error_size))
        return 0;
    t1 = clock();
    if (stats) stats->convrot_ms = elapsed_ms(t0, t1);

    t0 = clock();
    if (!h3_w4a8_quantize_act_int8(ctx->x_rot, m, k, ctx->act_quant, ctx->x_i8,
                                   ctx->act_scales, error, error_size))
        return 0;
    t1 = clock();
    if (stats) stats->quant_ms = elapsed_ms(t0, t1);

    per_tensor = (ctx->act_quant == H3_W4A8_ACT_QUANT_PER_TENSOR);
    if (!h3_cuda_gemm_i8_dequant_f32_w(
            ctx->cuda, w, ctx->x_i8, ctx->act_scales, per_tensor, m, y_f32,
            stats ? &stats->gemm_ms : NULL, stats ? stats->backend : NULL,
            stats ? sizeof(stats->backend) : 0, detail, sizeof(detail))) {
        snprintf(error, error_size, "gemm: %s", detail);
        return 0;
    }
    if (stats) {
        stats->m = m;
        stats->n = n;
        stats->k = k;
    }
    return 1;
}

void h3_dit_w4a8_swiglu_f32(const float *fused, size_t rows, size_t width,
                            float *out) {
    size_t r, c;
    if (!fused || !out || width == 0) return;
    for (r = 0; r < rows; r++) {
        const float *row = fused + r * (width * 2);
        float *dst = out + r * width;
        for (c = 0; c < width; c++) {
            float gate = row[c];
            float up = row[width + c];
            dst[c] = (gate / (1.0f + expf(-gate))) * up;
        }
    }
}

int h3_dit_w4a8_qkv_take_v(const float *qkv, size_t m, size_t inner,
                           float *v_out, char *error, size_t error_size) {
    size_t r;
    if (!qkv || !v_out || m == 0 || inner == 0) {
        fail(error, error_size, "invalid qkv_take_v args");
        return 0;
    }
    for (r = 0; r < m; r++) {
        memcpy(v_out + r * inner, qkv + r * (inner * 3) + inner * 2,
               inner * sizeof(float));
    }
    return 1;
}

void h3_dit_w4a8_block_aux_free(h3_dit_w4a8_block_aux *aux) {
    if (!aux) return;
    free(aux->norm1);
    free(aux->norm2);
    free(aux->q_norm);
    free(aux->k_norm);
    free(aux->adaln_w);
    free(aux->adaln_b);
    free(aux->inv_freq);
    free(aux->t_table);
    memset(aux, 0, sizeof(*aux));
}

static int load_named_f32_hdr(const h3_st_header *hdr, const char *name,
                              size_t expect, float **out, char *error,
                              size_t error_size) {
    size_t n = 0;
    h3_dtype dtype = H3_DTYPE_UNKNOWN;
    float *buf = h3_st_load_f32(hdr, name, &n, &dtype, error, error_size);
    if (!buf) return 0;
    if (expect && n != expect) {
        snprintf(error, error_size, "%s: got %zu elements, expected %zu", name,
                 n, expect);
        free(buf);
        return 0;
    }
    *out = buf;
    return 1;
}

static int load_named_f32(const h3_w4a8_store *store, const char *name,
                          size_t expect, float **out, char *error,
                          size_t error_size) {
    return load_named_f32_hdr(h3_w4a8_store_header(store), name, expect, out,
                              error, error_size);
}

int h3_dit_w4a8_load_block_aux_hdr(const h3_st_header *hdr, unsigned block,
                                   h3_dit_w4a8_block_aux *aux, char *error,
                                   size_t error_size) {
    char name[192];
    const h3_st_tensor *tw = NULL, *tb = NULL, *tt = NULL;
    size_t n = 0;
    h3_dtype dtype = H3_DTYPE_UNKNOWN;

    if (!hdr || !aux) {
        fail(error, error_size, "invalid load_block_aux args");
        return 0;
    }
    memset(aux, 0, sizeof(*aux));

    snprintf(name, sizeof(name), "blocks.%u.norm1.weight", block);
    if (!load_named_f32_hdr(hdr, name, H3_DIT_W4A8_HIDDEN, &aux->norm1, error,
                            error_size))
        goto fail;
    snprintf(name, sizeof(name), "blocks.%u.norm2.weight", block);
    if (!load_named_f32_hdr(hdr, name, H3_DIT_W4A8_HIDDEN, &aux->norm2, error,
                            error_size))
        goto fail;
    snprintf(name, sizeof(name), "blocks.%u.attn.q_norm.weight", block);
    if (!load_named_f32_hdr(hdr, name, H3_DIT_W4A8_HEAD_DIM, &aux->q_norm,
                            error, error_size))
        goto fail;
    snprintf(name, sizeof(name), "blocks.%u.attn.k_norm.weight", block);
    if (!load_named_f32_hdr(hdr, name, H3_DIT_W4A8_HEAD_DIM, &aux->k_norm,
                            error, error_size))
        goto fail;
    if (!load_named_f32_hdr(hdr, "rope.inv_freq", H3_DIT_W4A8_ROPE_FREQS,
                            &aux->inv_freq, error, error_size))
        goto fail;

    snprintf(name, sizeof(name), "blocks.%u.adaln_proj.linear.weight", block);
    tw = h3_st_find(hdr, name);
    snprintf(name, sizeof(name), "blocks.%u.adaln_proj.linear.bias", block);
    tb = h3_st_find(hdr, name);
    if (!tw || !tb) {
        fail(error, error_size, "missing adaln_proj weight/bias");
        goto fail;
    }
    if (tw->ndim != 2 || tb->ndim != 1) {
        fail(error, error_size, "adaln_proj unexpected rank");
        goto fail;
    }
    aux->adaln_out = (size_t)tw->shape[0];
    aux->adaln_in = (size_t)tw->shape[1];
    if (aux->adaln_out != H3_DIT_W4A8_ADALN_OUT ||
        aux->adaln_in != H3_DIT_W4A8_TIME_FEAT ||
        (size_t)tb->shape[0] != aux->adaln_out) {
        snprintf(error, error_size,
                 "adaln_proj shape [%zu,%zu]/%zu; expected [%zu,%d]/%zu "
                 "(W4A8 mixed contract)",
                 aux->adaln_out, aux->adaln_in, (size_t)tb->shape[0],
                 (size_t)H3_DIT_W4A8_ADALN_OUT, H3_DIT_W4A8_TIME_FEAT,
                 (size_t)H3_DIT_W4A8_ADALN_OUT);
        goto fail;
    }
    snprintf(name, sizeof(name), "blocks.%u.adaln_proj.linear.weight", block);
    aux->adaln_w = h3_st_load_f32(hdr, name, &n, &dtype, error, error_size);
    if (!aux->adaln_w || n != aux->adaln_out * aux->adaln_in) goto fail;
    snprintf(name, sizeof(name), "blocks.%u.adaln_proj.linear.bias", block);
    aux->adaln_b = h3_st_load_f32(hdr, name, &n, &dtype, error, error_size);
    if (!aux->adaln_b || n != aux->adaln_out) goto fail;

    tt = h3_st_find(hdr, "adaln_t_table");
    if (tt) {
        if (tt->ndim != 2 || (size_t)tt->shape[1] != H3_DIT_W4A8_TIME_FEAT) {
            fail(error, error_size, "adaln_t_table unexpected shape");
            goto fail;
        }
        aux->t_table_rows = (size_t)tt->shape[0];
        aux->t_table =
            h3_st_load_f32(hdr, "adaln_t_table", &n, &dtype, error, error_size);
        if (!aux->t_table || n != aux->t_table_rows * H3_DIT_W4A8_TIME_FEAT)
            goto fail;
    }
    return 1;

fail:
    h3_dit_w4a8_block_aux_free(aux);
    return 0;
}

int h3_dit_w4a8_load_block_aux(const h3_w4a8_store *store, unsigned block,
                               h3_dit_w4a8_block_aux *aux, char *error,
                               size_t error_size) {
    if (!store) {
        fail(error, error_size, "invalid load_block_aux args");
        return 0;
    }
    return h3_dit_w4a8_load_block_aux_hdr(h3_w4a8_store_header(store), block,
                                          aux, error, error_size);
}

/* Metal schedule_row_map: video/text ← video time_row; audio ← audio time_row.
 * Kijai adaln_proj emits all 3 modalities for one time_feat; splice MOD_AUDIO
 * from a second projection when audio_t_active. */
static int project_block_adaln_metal(h3_dit_w4a8 *ctx,
                                     const h3_dit_w4a8_block_aux *aux,
                                     size_t t_row, float *modulation,
                                     char *error, size_t error_size) {
    float time_feat[H3_DIT_W4A8_TIME_FEAT];
    const size_t slice =
        (size_t)H3_DIT_W4A8_SLOTS * (size_t)H3_DIT_W4A8_HIDDEN;
    if (!h3_dit_w4a8_time_feat_from_table(aux, t_row, time_feat, error,
                                          error_size) ||
        !h3_dit_w4a8_adaln_project_f32(aux, time_feat, modulation, error,
                                       error_size))
        return 0;
    if (ctx && ctx->audio_t_active && ctx->audio_t_row != t_row) {
        if (!ctx->adaln_audio_scratch) {
            ctx->adaln_audio_scratch =
                (float *)malloc(H3_DIT_W4A8_ADALN_OUT * sizeof(float));
            if (!ctx->adaln_audio_scratch) {
                fail(error, error_size, "OOM adaln audio scratch");
                return 0;
            }
        }
        if (!h3_dit_w4a8_time_feat_from_table(aux, ctx->audio_t_row, time_feat,
                                              error, error_size) ||
            !h3_dit_w4a8_adaln_project_f32(aux, time_feat,
                                           ctx->adaln_audio_scratch, error,
                                           error_size))
            return 0;
        memcpy(modulation + (size_t)H3_DIT_W4A8_MOD_AUDIO * slice,
               ctx->adaln_audio_scratch +
                   (size_t)H3_DIT_W4A8_MOD_AUDIO * slice,
               slice * sizeof(float));
    }
    /* visual_cond_t_active==0: do not enter — empty cond stays 3-row path. */
    if (ctx && ctx->visual_cond_t_active) {
        if (!ctx->adaln_visual_cond_scratch) {
            ctx->adaln_visual_cond_scratch =
                (float *)malloc(H3_DIT_W4A8_ADALN_OUT * sizeof(float));
            if (!ctx->adaln_visual_cond_scratch) {
                fail(error, error_size, "OOM adaln visual cond scratch");
                return 0;
            }
        }
        if (!h3_dit_w4a8_time_feat_from_table(aux, ctx->visual_cond_t_row,
                                              time_feat, error, error_size) ||
            !h3_dit_w4a8_adaln_project_f32(aux, time_feat,
                                           ctx->adaln_visual_cond_scratch,
                                           error, error_size))
            return 0;
        memcpy(modulation + 3 * slice,
               ctx->adaln_visual_cond_scratch +
                   (size_t)H3_DIT_W4A8_MOD_VIDEO * slice,
               slice * sizeof(float));
    }
    if (ctx && ctx->audio_cond_t_active) {
        const size_t row_idx =
            3u + (ctx->visual_cond_t_active ? 1u : 0u);
        if (!ctx->adaln_audio_cond_scratch) {
            ctx->adaln_audio_cond_scratch =
                (float *)malloc(H3_DIT_W4A8_ADALN_OUT * sizeof(float));
            if (!ctx->adaln_audio_cond_scratch) {
                fail(error, error_size, "OOM adaln audio cond scratch");
                return 0;
            }
        }
        if (!h3_dit_w4a8_time_feat_from_table(aux, ctx->audio_cond_t_row,
                                              time_feat, error, error_size) ||
            !h3_dit_w4a8_adaln_project_f32(aux, time_feat,
                                           ctx->adaln_audio_cond_scratch,
                                           error, error_size))
            return 0;
        memcpy(modulation + row_idx * slice,
               ctx->adaln_audio_cond_scratch +
                   (size_t)H3_DIT_W4A8_MOD_AUDIO * slice,
               slice * sizeof(float));
    }
    return 1;
}

/* Comfy AdalnProj: apply_silu=False when use_adaln_curves (adaln_t_table).
 * Kijai W4A8 is curve-form (table [1025,8] → linear); SiLU here OOD and was
 * implicated in split_half black-frame collapses. H3_DIT_ADALN_SILU=1 restores
 * the non-curve TimeEmbedder path (linear(SiLU(t_emb))). */
static int dit_adaln_apply_silu(void) {
    return env_flag_is_one("H3_DIT_ADALN_SILU");
}

static void adaln_silu_time_feat(const float *time_feat, float *activated,
                                 size_t n) {
    size_t i;
    for (i = 0; i < n; i++) {
        float x = time_feat[i];
        activated[i] = x / (1.0f + expf(-x));
    }
}

int h3_dit_w4a8_adaln_project_f32(const h3_dit_w4a8_block_aux *aux,
                                  const float *time_feat, float *modulation,
                                  char *error, size_t error_size) {
    float act[H3_DIT_W4A8_TIME_FEAT];
    const float *feat;
    size_t o, i;
    if (!aux || !aux->adaln_w || !aux->adaln_b || !time_feat || !modulation) {
        fail(error, error_size, "invalid adaln_project args");
        return 0;
    }
    if (aux->adaln_in != H3_DIT_W4A8_TIME_FEAT) {
        fail(error, error_size, "adaln_in != TIME_FEAT");
        return 0;
    }
    if (dit_adaln_apply_silu()) {
        adaln_silu_time_feat(time_feat, act, H3_DIT_W4A8_TIME_FEAT);
        feat = act;
    } else {
        feat = time_feat;
    }
    for (o = 0; o < aux->adaln_out; o++) {
        const float *row = aux->adaln_w + o * aux->adaln_in;
        float sum = aux->adaln_b[o];
        for (i = 0; i < aux->adaln_in; i++) sum += row[i] * feat[i];
        modulation[o] = sum;
    }
    return 1;
}

int h3_dit_w4a8_time_feat_from_table(const h3_dit_w4a8_block_aux *aux,
                                     size_t table_row, float *time_feat,
                                     char *error, size_t error_size) {
    if (!aux || !aux->t_table || !time_feat) {
        fail(error, error_size, "adaln_t_table unavailable");
        return 0;
    }
    if (table_row >= aux->t_table_rows) {
        fail(error, error_size, "adaln_t_table row out of range");
        return 0;
    }
    memcpy(time_feat, aux->t_table + table_row * H3_DIT_W4A8_TIME_FEAT,
           H3_DIT_W4A8_TIME_FEAT * sizeof(float));
    return 1;
}

/* Phase-A profile counters (RoPE table build). Zero-cost unless env var set. */
static double g_dit_rope_profile_ms = 0.0;
static unsigned long g_dit_rope_profile_calls = 0;
static unsigned long g_dit_rope_memo_hits = 0;
static unsigned long g_dit_rope_memo_misses = 0;

void h3_dit_w4a8_rope_profile_get(double *ms, unsigned long *calls) {
    if (ms) *ms = g_dit_rope_profile_ms;
    if (calls) *calls = g_dit_rope_profile_calls;
}

void h3_dit_w4a8_rope_profile_get_ex(double *ms, unsigned long *calls,
                                     unsigned long *hits,
                                     unsigned long *misses) {
    if (ms) *ms = g_dit_rope_profile_ms;
    if (calls) *calls = g_dit_rope_profile_calls;
    if (hits) *hits = g_dit_rope_memo_hits;
    if (misses) *misses = g_dit_rope_memo_misses;
}

void h3_dit_w4a8_rope_profile_reset(void) {
    g_dit_rope_profile_ms = 0.0;
    g_dit_rope_profile_calls = 0;
    g_dit_rope_memo_hits = 0;
    g_dit_rope_memo_misses = 0;
}

/* Per-shape memo cache for DiT RoPE tables.
 * Key = (seq, spatial_scale, content-hash of pos_t/pos_h/pos_w).
 * Value = precomputed cos/sin table [seq, ROPE_HALF].
 * Grow-only, process-lifetime. Thread-safe under GIL-style single-threaded
 * generation (h3_cuda_generate is single-threaded for DiT forward). */
typedef struct {
    size_t seq;
    float spatial_scale;
    uint64_t pos_hash;
    float *cos_table;
    float *sin_table;
} dit_rope_memo_entry;

static dit_rope_memo_entry *g_dit_rope_memo = NULL;
static size_t g_dit_rope_memo_n = 0;
static size_t g_dit_rope_memo_cap = 0;

static uint64_t dit_rope_hash_pos(const float *pos, size_t n) {
    /* FNV-1a over raw bytes; position arrays are deterministic floats. */
    uint64_t h = 1469598103934665603ULL;
    const unsigned char *p = (const unsigned char *)pos;
    size_t bytes = n * sizeof(float);
    size_t i;
    for (i = 0; i < bytes; i++) {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static uint64_t dit_rope_hash_all(const float *pos_t, const float *pos_h,
                                  const float *pos_w, size_t seq,
                                  float spatial_scale) {
    uint64_t h = dit_rope_hash_pos(pos_t, seq);
    h = dit_rope_hash_pos(pos_h, seq) ^ (h << 1);
    h = dit_rope_hash_pos(pos_w, seq) ^ (h << 2);
    h ^= (uint64_t)(spatial_scale * 1000.0f);
    return h;
}

static int dit_rope_memo_grow(size_t need, char *error, size_t error_size) {
    if (need <= g_dit_rope_memo_cap) return 1;
    size_t new_cap = g_dit_rope_memo_cap ? g_dit_rope_memo_cap * 2 : 8;
    if (new_cap < need) new_cap = need;
    dit_rope_memo_entry *n = (dit_rope_memo_entry *)realloc(
        g_dit_rope_memo, new_cap * sizeof(dit_rope_memo_entry));
    if (!n) {
        fail(error, error_size, "OOM rope memo grow");
        return 0;
    }
    g_dit_rope_memo = n;
    g_dit_rope_memo_cap = new_cap;
    return 1;
}

static dit_rope_memo_entry *dit_rope_memo_find(size_t seq, float spatial_scale,
                                                uint64_t pos_hash) {
    size_t i;
    for (i = 0; i < g_dit_rope_memo_n; i++) {
        dit_rope_memo_entry *e = &g_dit_rope_memo[i];
        if (e->seq == seq && e->spatial_scale == spatial_scale &&
            e->pos_hash == pos_hash)
            return e;
    }
    return NULL;
}

static dit_rope_memo_entry *dit_rope_memo_alloc(size_t seq,
                                                 float spatial_scale,
                                                 uint64_t pos_hash,
                                                 char *error,
                                                 size_t error_size) {
    if (!dit_rope_memo_grow(g_dit_rope_memo_n + 1, error, error_size))
        return NULL;
    dit_rope_memo_entry *e = &g_dit_rope_memo[g_dit_rope_memo_n];
    e->seq = seq;
    e->spatial_scale = spatial_scale;
    e->pos_hash = pos_hash;
    e->cos_table = (float *)malloc(seq * H3_DIT_W4A8_ROPE_HALF * sizeof(float));
    e->sin_table = (float *)malloc(seq * H3_DIT_W4A8_ROPE_HALF * sizeof(float));
    if (!e->cos_table || !e->sin_table) {
        free(e->cos_table);
        free(e->sin_table);
        fail(error, error_size, "OOM rope memo table");
        return NULL;
    }
    g_dit_rope_memo_n++;
    return e;
}

static void dit_rope_memo_free_all(void) {
    size_t i;
    for (i = 0; i < g_dit_rope_memo_n; i++) {
        free(g_dit_rope_memo[i].cos_table);
        free(g_dit_rope_memo[i].sin_table);
    }
    free(g_dit_rope_memo);
    g_dit_rope_memo = NULL;
    g_dit_rope_memo_n = 0;
    g_dit_rope_memo_cap = 0;
}

/* Cleanup hook: dit_rope_memo_free_all() is called from h3_dit_w4a8_free. */

int h3_dit_w4a8_rope_tables_f32(const float *inv_freq, size_t seq,
                                const float *pos_t, const float *pos_h,
                                const float *pos_w, float spatial_scale,
                                float *cos_out, float *sin_out, char *error,
                                size_t error_size) {
    size_t row, axis, f;
    clock_t t0 = 0, t1 = 0;
    const int profile = env_flag_is_one("H3_ROPE_PROFILE");
    /* Explicit H3_ROPE_MEMO=0 disables host memo (fair KPI). Default on. */
    const char *memo_env = getenv("H3_ROPE_MEMO");
    const int use_memo = !(memo_env && memo_env[0] == '0' && memo_env[1] == '\0');
    uint64_t pos_hash = 0;
    dit_rope_memo_entry *memo = NULL;
    if (profile) t0 = clock();
    if (!inv_freq || !pos_t || !pos_h || !pos_w || !cos_out || !sin_out ||
        seq == 0) {
        fail(error, error_size, "invalid rope_tables args");
        return 0;
    }
    if (!isfinite(spatial_scale) || spatial_scale <= 0.0f) {
        fail(error, error_size, "invalid spatial_scale");
        return 0;
    }
    pos_hash = dit_rope_hash_all(pos_t, pos_h, pos_w, seq, spatial_scale);
    if (use_memo)
        memo = dit_rope_memo_find(seq, spatial_scale, pos_hash);
    if (use_memo && memo) {
        g_dit_rope_memo_hits++;
        memcpy(cos_out, memo->cos_table,
               seq * H3_DIT_W4A8_ROPE_HALF * sizeof(float));
        memcpy(sin_out, memo->sin_table,
               seq * H3_DIT_W4A8_ROPE_HALF * sizeof(float));
    } else {
        if (use_memo) {
            g_dit_rope_memo_misses++;
            memo = dit_rope_memo_alloc(seq, spatial_scale, pos_hash, error,
                                       error_size);
            if (!memo) return 0;
        }
        for (row = 0; row < seq; row++) {
            float axes[3] = {pos_t[row], pos_h[row] * spatial_scale,
                             pos_w[row] * spatial_scale};
            for (axis = 0; axis < 3; axis++) {
                for (f = 0; f < H3_DIT_W4A8_ROPE_FREQS; f++) {
                    size_t index =
                        row * H3_DIT_W4A8_ROPE_HALF +
                        axis * H3_DIT_W4A8_ROPE_FREQS + f;
                    float angle = axes[axis] * inv_freq[f];
                    float c = cosf(angle), s = sinf(angle);
                    cos_out[index] = c;
                    sin_out[index] = s;
                    if (memo) {
                        memo->cos_table[index] = c;
                        memo->sin_table[index] = s;
                    }
                }
            }
        }
    }
    if (profile) {
        t1 = clock();
        g_dit_rope_profile_ms += elapsed_ms(t0, t1);
        g_dit_rope_profile_calls++;
    }
    return 1;
}

/* Metal h3_host.c frame/video grid constants. */
static const int h3_dit_frame_per_token[5] = {1, 4, 4, 4, 4};
static const float h3_dit_frame_rescale = 5.0f / 3.0f;

static float h3_dit_video_time_step(int t) {
    return h3_dit_frame_rescale * (float)h3_dit_frame_per_token[t % 5];
}

float h3_dit_w4a8_video_span_sum(int latent_t) {
    float sum = 0.0f;
    int t;
    if (latent_t < 1) return 0.0f;
    for (t = 0; t < latent_t; t++)
        sum += h3_dit_video_time_step(t);
    return sum;
}

int h3_dit_w4a8_fill_metal_video_pos(size_t text_len, int latent_t,
                                     int latent_h, int latent_w, float *pos_t,
                                     float *pos_h, float *pos_w, char *error,
                                     size_t error_size) {
    int nh, nw, t, yh, xw;
    size_t idx = 0;
    float area, ratio_h, ratio_w, step_h, step_w, base_h, base_w, time;
    if (!pos_t || !pos_h || !pos_w || latent_t < 1 || latent_h < 2 ||
        latent_w < 2 || (latent_h % 2) || (latent_w % 2)) {
        fail(error, error_size, "invalid fill_metal_video_pos args");
        return 0;
    }
    nh = latent_h / 2;
    nw = latent_w / 2;
    area = sqrtf((float)latent_h * (float)latent_w);
    ratio_h = (float)latent_h / area;
    ratio_w = (float)latent_w / area;
    step_h = ratio_h / (float)nh;
    step_w = ratio_w / (float)nw;
    base_h = (1.0f - ratio_h) * 0.5f;
    base_w = (1.0f - ratio_w) * 0.5f;
    time = (float)text_len;
    for (t = 0; t < latent_t; t++) {
        for (yh = 0; yh < nh; yh++) {
            float hh = ((float)yh * step_h + base_h) * 32.0f;
            for (xw = 0; xw < nw; xw++, idx++) {
                float ww = ((float)xw * step_w + base_w) * 32.0f;
                pos_t[idx] = time;
                pos_h[idx] = hh;
                pos_w[idx] = ww;
            }
        }
        time += h3_dit_video_time_step(t);
    }
    return 1;
}

int h3_dit_w4a8_reference_image_canvas(int width, int height,
                                       int target_width, int target_height,
                                       int max_short_edge, int *adapted_w,
                                       int *adapted_h) {
    double scale, out_w, out_h;
    if (width < 1 || height < 1 || target_width < 1 || target_height < 1 ||
        max_short_edge < 0 || !adapted_w || !adapted_h)
        return 0;
    if (max_short_edge) {
        int short_edge = width < height ? width : height;
        scale = fmin(1.0, (double)max_short_edge / (double)short_edge);
    } else {
        double target_area = (double)target_width * (double)target_height;
        double source_area = (double)width * (double)height;
        scale = fmin(1.0, sqrt(target_area / source_area));
    }
    out_w = nearbyint((double)width * scale / H3_DIT_W4A8_CANVAS_MULTIPLE) *
            H3_DIT_W4A8_CANVAS_MULTIPLE;
    out_h = nearbyint((double)height * scale / H3_DIT_W4A8_CANVAS_MULTIPLE) *
            H3_DIT_W4A8_CANVAS_MULTIPLE;
    if (out_w < H3_DIT_W4A8_CANVAS_MULTIPLE)
        out_w = H3_DIT_W4A8_CANVAS_MULTIPLE;
    if (out_h < H3_DIT_W4A8_CANVAS_MULTIPLE)
        out_h = H3_DIT_W4A8_CANVAS_MULTIPLE;
    if (out_w > (double)INT_MAX || out_h > (double)INT_MAX) return 0;
    *adapted_w = (int)out_w;
    *adapted_h = (int)out_h;
    return 1;
}

int h3_dit_w4a8_fill_metal_ref_image_pos(float cursor, int latent_h,
                                         int latent_w, float *pos_t,
                                         float *pos_h, float *pos_w,
                                         char *error, size_t error_size) {
    return h3_dit_w4a8_fill_metal_cond_pos(cursor, latent_h, latent_w, pos_t,
                                           pos_h, pos_w, error, error_size);
}

int h3_dit_w4a8_reference_video_canvas(int width, int height, int *adapted_w,
                                       int *adapted_h) {
    const double max_pixels = 768.0 * 1344.0;
    double ratio, nominal_w, nominal_h, pixels, source_area, target_area;
    int out_w, out_h;
    if (width < 1 || height < 1 || !adapted_w || !adapted_h) return 0;
    ratio = (double)width / (double)height;
    if (ratio >= 1.0) {
        nominal_w = 768.0 * ratio;
        nominal_h = 768.0;
    } else {
        nominal_w = 768.0;
        nominal_h = 768.0 / ratio;
    }
    pixels = nominal_w * nominal_h;
    if (pixels > max_pixels) {
        double scale = sqrt(max_pixels / pixels);
        nominal_w *= scale;
        nominal_h *= scale;
    }
    out_w = (int)(nearbyint(nominal_w / H3_DIT_W4A8_CANVAS_MULTIPLE) *
                  H3_DIT_W4A8_CANVAS_MULTIPLE);
    out_h = (int)(nearbyint(nominal_h / H3_DIT_W4A8_CANVAS_MULTIPLE) *
                  H3_DIT_W4A8_CANVAS_MULTIPLE);
    *adapted_w = out_w < H3_DIT_W4A8_CANVAS_MULTIPLE ? H3_DIT_W4A8_CANVAS_MULTIPLE
                                                    : out_w;
    *adapted_h = out_h < H3_DIT_W4A8_CANVAS_MULTIPLE ? H3_DIT_W4A8_CANVAS_MULTIPLE
                                                    : out_h;
    source_area = (double)width * (double)height;
    target_area = (double)(*adapted_w) * (double)(*adapted_h);
    if (source_area < target_area) {
        out_w = (int)(nearbyint((double)width / H3_DIT_W4A8_CANVAS_MULTIPLE) *
                      H3_DIT_W4A8_CANVAS_MULTIPLE);
        out_h = (int)(nearbyint((double)height / H3_DIT_W4A8_CANVAS_MULTIPLE) *
                      H3_DIT_W4A8_CANVAS_MULTIPLE);
        *adapted_w = out_w < H3_DIT_W4A8_CANVAS_MULTIPLE
                         ? H3_DIT_W4A8_CANVAS_MULTIPLE
                         : out_w;
        *adapted_h = out_h < H3_DIT_W4A8_CANVAS_MULTIPLE
                         ? H3_DIT_W4A8_CANVAS_MULTIPLE
                         : out_h;
    }
    return 1;
}

int h3_dit_w4a8_fill_metal_ref_video_pos(float cursor, int latent_t,
                                         int latent_h, int latent_w,
                                         float *pos_t, float *pos_h,
                                         float *pos_w, char *error,
                                         size_t error_size) {
    /* Same as fill_metal_video_pos but cursor is float (Ref layout). */
    int nh, nw, t, yh, xw;
    size_t idx = 0;
    float area, ratio_h, ratio_w, step_h, step_w, base_h, base_w, time;
    if (!pos_t || !pos_h || !pos_w || latent_t < 1 || latent_h < 2 ||
        latent_w < 2 || (latent_h % 2) || (latent_w % 2)) {
        fail(error, error_size, "invalid fill_metal_ref_video_pos args");
        return 0;
    }
    nh = latent_h / 2;
    nw = latent_w / 2;
    area = sqrtf((float)latent_h * (float)latent_w);
    ratio_h = (float)latent_h / area;
    ratio_w = (float)latent_w / area;
    step_h = ratio_h / (float)nh;
    step_w = ratio_w / (float)nw;
    base_h = (1.0f - ratio_h) * 0.5f;
    base_w = (1.0f - ratio_w) * 0.5f;
    time = cursor;
    for (t = 0; t < latent_t; t++) {
        for (yh = 0; yh < nh; yh++) {
            float hh = ((float)yh * step_h + base_h) * 32.0f;
            for (xw = 0; xw < nw; xw++, idx++) {
                float ww = ((float)xw * step_w + base_w) * 32.0f;
                pos_t[idx] = time;
                pos_h[idx] = hh;
                pos_w[idx] = ww;
            }
        }
        time += h3_dit_video_time_step(t);
    }
    return 1;
}

int h3_dit_w4a8_fill_metal_cond_pos(float condition_time, int latent_h,
                                    int latent_w, float *pos_t, float *pos_h,
                                    float *pos_w, char *error,
                                    size_t error_size) {
    int nh, nw, yh, xw;
    size_t idx = 0;
    float area, ratio_h, ratio_w, step_h, step_w, base_h, base_w;
    if (!pos_t || !pos_h || !pos_w || latent_h < 2 || latent_w < 2 ||
        (latent_h % 2) || (latent_w % 2)) {
        fail(error, error_size, "invalid fill_metal_cond_pos args");
        return 0;
    }
    nh = latent_h / 2;
    nw = latent_w / 2;
    area = sqrtf((float)latent_h * (float)latent_w);
    ratio_h = (float)latent_h / area;
    ratio_w = (float)latent_w / area;
    step_h = ratio_h / (float)nh;
    step_w = ratio_w / (float)nw;
    base_h = (1.0f - ratio_h) * 0.5f;
    base_w = (1.0f - ratio_w) * 0.5f;
    for (yh = 0; yh < nh; yh++) {
        float hh = ((float)yh * step_h + base_h) * 32.0f;
        for (xw = 0; xw < nw; xw++, idx++) {
            float ww = ((float)xw * step_w + base_w) * 32.0f;
            pos_t[idx] = condition_time;
            pos_h[idx] = hh;
            pos_w[idx] = ww;
        }
    }
    return 1;
}

int h3_dit_w4a8_fill_metal_ref_audio_pos(float cursor, int audio_t, int latent_h,
                                         int latent_w, float *pos_t,
                                         float *pos_h, float *pos_w,
                                         char *error, size_t error_size) {
    int nw, index;
    float area, ratio_w, step_w, base_w, low, high;
    if (!pos_t || !pos_h || !pos_w || audio_t < 1 || latent_h < 2 ||
        latent_w < 2 || (latent_w % 2)) {
        fail(error, error_size, "invalid fill_metal_ref_audio_pos args");
        return 0;
    }
    nw = latent_w / 2;
    area = sqrtf((float)latent_h * (float)latent_w);
    ratio_w = (float)latent_w / area;
    step_w = ratio_w / (float)nw;
    base_w = (1.0f - ratio_w) * 0.5f;
    low = (0.0f * step_w + base_w) * 32.0f;
    high = ((float)(nw - 1) * step_w + base_w) * 32.0f;
    for (index = 0; index < audio_t; index++) {
        pos_t[index] = cursor + (float)index;
        pos_h[index] = 0.0f;
        pos_w[index] = low;
        pos_t[audio_t + index] = cursor + (float)index;
        pos_h[audio_t + index] = 0.0f;
        pos_w[audio_t + index] = high;
    }
    return 1;
}

int h3_dit_w4a8_fill_metal_audio_pos(size_t text_len, int audio_t, int latent_h,
                                     int latent_w, float *pos_t, float *pos_h,
                                     float *pos_w, char *error,
                                     size_t error_size) {
    int nw, index;
    float area, ratio_w, step_w, base_w, low, high;
    if (!pos_t || !pos_h || !pos_w || audio_t < 1 || latent_h < 2 ||
        latent_w < 2 || (latent_w % 2)) {
        fail(error, error_size, "invalid fill_metal_audio_pos args");
        return 0;
    }
    nw = latent_w / 2;
    area = sqrtf((float)latent_h * (float)latent_w);
    ratio_w = (float)latent_w / area;
    step_w = ratio_w / (float)nw;
    base_w = (1.0f - ratio_w) * 0.5f;
    low = (0.0f * step_w + base_w) * 32.0f;
    high = ((float)(nw - 1) * step_w + base_w) * 32.0f;
    for (index = 0; index < audio_t; index++) {
        pos_t[index] = (float)text_len + (float)index;
        pos_h[index] = 0.0f;
        pos_w[index] = low;
        pos_t[audio_t + index] = (float)text_len + (float)index;
        pos_h[audio_t + index] = 0.0f;
        pos_w[audio_t + index] = high;
    }
    return 1;
}

void h3_dit_w4a8_rms_norm_f32(const float *x, const float *weight, size_t rows,
                              size_t width, float epsilon, float *out) {
    size_t r, c;
    if (!x || !weight || !out || width == 0) return;
    for (r = 0; r < rows; r++) {
        const float *xr = x + r * width;
        float *yr = out + r * width;
        float sum = 0.0f;
        float inv;
        for (c = 0; c < width; c++) sum += xr[c] * xr[c];
        inv = 1.0f / sqrtf(sum / (float)width + epsilon);
        for (c = 0; c < width; c++) yr[c] = xr[c] * inv * weight[c];
    }
}

void h3_dit_w4a8_adaln_f32(const float *x, const float *norm_w,
                           const float *modulation, const uint32_t *row_map,
                           size_t rows, size_t width, size_t slots,
                           size_t shift_slot, size_t scale_slot, float epsilon,
                           float *out) {
    size_t r, c;
    if (!x || !norm_w || !modulation || !out || width == 0 || slots == 0)
        return;
    for (r = 0; r < rows; r++) {
        const float *xr = x + r * width;
        float *yr = out + r * width;
        uint32_t mrow = row_map ? row_map[r] : (uint32_t)r;
        const float *mod_base = modulation + (size_t)mrow * slots * width;
        const float *shift = mod_base + shift_slot * width;
        const float *scale = mod_base + scale_slot * width;
        float sum = 0.0f;
        float inv;
        for (c = 0; c < width; c++) sum += xr[c] * xr[c];
        inv = 1.0f / sqrtf(sum / (float)width + epsilon);
        for (c = 0; c < width; c++) {
            float n = xr[c] * inv * norm_w[c];
            yr[c] = n * (1.0f + scale[c]) + shift[c];
        }
    }
}

void h3_dit_w4a8_gate_f32(const float *residual, const float *branch,
                          const float *modulation, const uint32_t *row_map,
                          size_t rows, size_t width, size_t slots,
                          size_t gate_slot, float *out) {
    size_t r, c;
    if (!residual || !branch || !modulation || !out || width == 0 || slots == 0)
        return;
    for (r = 0; r < rows; r++) {
        uint32_t mrow = row_map ? row_map[r] : (uint32_t)r;
        const float *gate =
            modulation + (size_t)mrow * slots * width + gate_slot * width;
        const float *res = residual + r * width;
        const float *br = branch + r * width;
        float *yr = out + r * width;
        for (c = 0; c < width; c++) yr[c] = res[c] + br[c] * gate[c];
    }
}

int h3_dit_w4a8_qkv_rope_f32(const float *qkv, const float *q_norm,
                             const float *k_norm, const float *rope_cos,
                             const float *rope_sin, size_t seq, size_t heads,
                             size_t head_dim, size_t rope_half, float epsilon,
                             float *query, float *key, float *value,
                             char *error, size_t error_size) {
    size_t row, head, d;
    size_t inner;
    /* rope_half==0 (token_refiner): skip RoPE; cos/sin may be NULL. */
    if (!qkv || !q_norm || !k_norm || !query || !key || !value || seq == 0 ||
        heads == 0 || head_dim == 0 ||
        (rope_half > 0 && (!rope_cos || !rope_sin))) {
        fail(error, error_size, "invalid qkv_rope args");
        return 0;
    }
    if (rope_half * 2 > head_dim) {
        fail(error, error_size, "rope_half too large for head_dim");
        return 0;
    }
    inner = heads * head_dim;
    for (row = 0; row < seq; row++) {
        const float *qkv_row = qkv + row * inner * 3;
        const float *c_row =
            (rope_half > 0) ? (rope_cos + row * rope_half) : NULL;
        const float *s_row =
            (rope_half > 0) ? (rope_sin + row * rope_half) : NULL;
        for (head = 0; head < heads; head++) {
            /* Default: contiguous [Q|K|V] (Comfy/Kijai). GROUPED=1 → interleaved
             * (Metal BF16 export). Do not special-case refiner: Kijai W4A8 keeps
             * the same QKV layout as the DiT blocks. */
            const float *q_base;
            const float *k_base;
            const float *v_base;
            if (dit_qkv_grouped_layout()) {
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
            float q_inv, k_inv;
            for (d = 0; d < head_dim; d++) {
                q_sum += q_base[d] * q_base[d];
                k_sum += k_base[d] * k_base[d];
                v_out[d] = v_base[d];
            }
            q_inv = 1.0f / sqrtf(q_sum / (float)head_dim + epsilon);
            k_inv = 1.0f / sqrtf(k_sum / (float)head_dim + epsilon);
            if (dit_rope_split_half() && rope_half > 0) {
                /* Comfy apply_rope_split_half: pair dim d with d+rope_half. */
                for (d = 0; d < head_dim; d++) {
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
                /* Consecutive-pair rotate (legacy / non-Comfy). */
                for (d = 0; d < head_dim; d++) {
                    q_out[d] = q_base[d] * q_inv * q_norm[d];
                    k_out[d] = k_base[d] * k_inv * k_norm[d];
                }
                for (d = 0; d < rope_half; d++) {
                    size_t i0 = d * 2u;
                    size_t i1 = i0 + 1u;
                    float c = c_row[d];
                    float s = s_row[d];
                    float q0 = q_out[i0], q1 = q_out[i1];
                    float k0 = k_out[i0], k1 = k_out[i1];
                    q_out[i0] = q0 * c - q1 * s;
                    q_out[i1] = q0 * s + q1 * c;
                    k_out[i0] = k0 * c - k1 * s;
                    k_out[i1] = k0 * s + k1 * c;
                }
            } else {
                for (d = 0; d < head_dim; d++) {
                    q_out[d] = q_base[d] * q_inv * q_norm[d];
                    k_out[d] = k_base[d] * k_inv * k_norm[d];
                }
            }
        }
    }
    return 1;
}

/* P1 FP verification: split merged [3n,k] weight into three narrow [n,k]
 * GEMMs. Output-split means each output column of the wide GEMM is an
 * independent dot product, so merged vs split3 is bit-identical in exact
 * arithmetic; in FP32 the dot-product summation order is the same too
 * (same K order), so MAE should be exactly 0. */
int h3_dit_w4a8_fp_qkv_split3_f32(const float *x, const float *wqkv,
                                  size_t m, size_t n, size_t k, float *y,
                                  char *error, size_t error_size) {
    size_t r, part, o, c;
    if (!x || !wqkv || !y || m == 0 || n == 0 || k == 0) {
        fail(error, error_size, "invalid fp_qkv_split3 args");
        return 0;
    }
    for (r = 0; r < m; r++) {
        const float *xr = x + r * k;
        float *yr = y + r * n * 3;
        for (part = 0; part < 3; part++) {
            const float *wp = wqkv + part * n * k;
            float *yp = yr + part * n;
            for (o = 0; o < n; o++) {
                const float *wrow = wp + o * k;
                float acc = 0.0f;
                for (c = 0; c < k; c++) acc += xr[c] * wrow[c];
                yp[o] = acc;
            }
        }
    }
    return 1;
}

/* P1 FP verification: unfused QK-RMSNorm (materialized intermediate) then
 * RoPE. Same layout/grouped/split-half conventions as the fused kernel. */
int h3_dit_w4a8_fp_qknorm_rope_unfused_f32(
    const float *qkv, const float *q_norm, const float *k_norm,
    const float *rope_cos, const float *rope_sin, size_t seq, size_t heads,
    size_t head_dim, size_t rope_half, float epsilon, float *query, float *key,
    float *value, char *error, size_t error_size) {
    size_t row, head, d;
    size_t inner;
    float *q_tmp = NULL, *k_tmp = NULL;
    int ok = 0;
    if (!qkv || !q_norm || !k_norm || !query || !key || !value || seq == 0 ||
        heads == 0 || head_dim == 0 ||
        (rope_half > 0 && (!rope_cos || !rope_sin))) {
        fail(error, error_size, "invalid fp_qknorm_rope_unfused args");
        return 0;
    }
    if (rope_half * 2 > head_dim) {
        fail(error, error_size, "rope_half too large for head_dim");
        return 0;
    }
    inner = heads * head_dim;
    q_tmp = (float *)malloc(inner * sizeof(float));
    k_tmp = (float *)malloc(inner * sizeof(float));
    if (!q_tmp || !k_tmp) {
        fail(error, error_size, "out of memory (fp_qknorm_rope_unfused)");
        goto done;
    }
    for (row = 0; row < seq; row++) {
        const float *qkv_row = qkv + row * inner * 3;
        const float *c_row =
            (rope_half > 0) ? (rope_cos + row * rope_half) : NULL;
        const float *s_row =
            (rope_half > 0) ? (rope_sin + row * rope_half) : NULL;
        for (head = 0; head < heads; head++) {
            const float *q_base;
            const float *k_base;
            const float *v_base;
            if (dit_qkv_grouped_layout()) {
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
            float q_inv, k_inv;
            /* Pass 1: RMS normalize into materialized q_tmp/k_tmp (the
             * intermediate the fused kernel keeps in registers). */
            for (d = 0; d < head_dim; d++) {
                q_sum += q_base[d] * q_base[d];
                k_sum += k_base[d] * k_base[d];
                v_out[d] = v_base[d];
            }
            q_inv = 1.0f / sqrtf(q_sum / (float)head_dim + epsilon);
            k_inv = 1.0f / sqrtf(k_sum / (float)head_dim + epsilon);
            for (d = 0; d < head_dim; d++) {
                q_tmp[d] = q_base[d] * q_inv * q_norm[d];
                k_tmp[d] = k_base[d] * k_inv * k_norm[d];
            }
            /* Pass 2: RoPE from the materialized buffer. */
            if (dit_rope_split_half() && rope_half > 0) {
                for (d = 0; d < head_dim; d++) {
                    float q0 = q_tmp[d];
                    float k0 = k_tmp[d];
                    if (d < rope_half) {
                        size_t pair = d + rope_half;
                        float q1 = q_tmp[pair];
                        float k1 = k_tmp[pair];
                        float c = c_row[d];
                        float s = s_row[d];
                        q0 = q0 * c - q1 * s;
                        k0 = k0 * c - k1 * s;
                    } else if (d < rope_half * 2) {
                        size_t pair = d - rope_half;
                        float q1 = q_tmp[pair];
                        float k1 = k_tmp[pair];
                        float c = c_row[pair];
                        float s = s_row[pair];
                        q0 = q0 * c + q1 * s;
                        k0 = k0 * c + k1 * s;
                    }
                    q_out[d] = q0;
                    k_out[d] = k0;
                }
            } else if (rope_half > 0) {
                for (d = 0; d < head_dim; d++) {
                    q_out[d] = q_tmp[d];
                    k_out[d] = k_tmp[d];
                }
                for (d = 0; d < rope_half; d++) {
                    size_t i0 = d * 2u;
                    size_t i1 = i0 + 1u;
                    float c = c_row[d];
                    float s = s_row[d];
                    float q0 = q_out[i0], q1 = q_out[i1];
                    float k0 = k_out[i0], k1 = k_out[i1];
                    q_out[i0] = q0 * c - q1 * s;
                    q_out[i1] = q0 * s + q1 * c;
                    k_out[i0] = k0 * c - k1 * s;
                    k_out[i1] = k0 * s + k1 * c;
                }
            } else {
                for (d = 0; d < head_dim; d++) {
                    q_out[d] = q_tmp[d];
                    k_out[d] = k_tmp[d];
                }
            }
        }
    }
    ok = 1;
done:
    free(q_tmp);
    free(k_tmp);
    return ok;
}

int h3_dit_w4a8_fp_qkv_merge_enabled(void) {
    return env_flag_is_one("H3_DIT_QKV_MERGE");
}

int h3_dit_w4a8_fp_qknorm_rope_fuse_enabled(void) {
    return env_flag_is_one("H3_DIT_QKNORM_ROPE_FUSE");
}

int h3_dit_w4a8_sdpa_f32(const float *query, const float *key,
                         const float *value, size_t seq, size_t heads,
                         size_t head_dim, float scale, float *out, char *error,
                         size_t error_size) {
    size_t h, qi, ki, d;
    float *scores = NULL;
    if (!query || !key || !value || !out || seq == 0 || heads == 0 ||
        head_dim == 0) {
        fail(error, error_size, "invalid sdpa args");
        return 0;
    }
    scores = (float *)malloc(seq * sizeof(float));
    if (!scores) {
        fail(error, error_size, "out of memory (sdpa scores)");
        return 0;
    }
    for (h = 0; h < heads; h++) {
        for (qi = 0; qi < seq; qi++) {
            const float *q = query + (qi * heads + h) * head_dim;
            float *o = out + (qi * heads + h) * head_dim;
            float max_s = -INFINITY;
            float sum_e = 0.0f;
            for (ki = 0; ki < seq; ki++) {
                const float *k = key + (ki * heads + h) * head_dim;
                float dot = 0.0f;
                for (d = 0; d < head_dim; d++) dot += q[d] * k[d];
                scores[ki] = dot * scale;
                if (scores[ki] > max_s) max_s = scores[ki];
            }
            for (ki = 0; ki < seq; ki++) {
                scores[ki] = expf(scores[ki] - max_s);
                sum_e += scores[ki];
            }
            for (d = 0; d < head_dim; d++) o[d] = 0.0f;
            for (ki = 0; ki < seq; ki++) {
                const float *v = value + (ki * heads + h) * head_dim;
                float w = scores[ki] / sum_e;
                for (d = 0; d < head_dim; d++) o[d] += w * v[d];
            }
        }
    }
    free(scores);
    return 1;
}

/* ---- NVFP4 P1.4/P1.5 dispatch (H3_DIT_NVFP4=1) ----
 * Weight sidecar: one safetensors holding per-linear
 *   {base}.weight_fp4    U8 [N, K/2]  (packed E2M1, ConvRot basis)
 *   {base}.weight_sf     U8 [rb*ncb*512]  (swizzled UE4M3 SF, same layout
 *                        as h3_cuda_quantize_act_nvfp4_kernel)
 *   {base}.weight_scale2 F32 [1] (per-tensor second-level scale)
 * Lookup mirrors the INT8 loader's base-name derivation (blocks.{i}.<suffix>
 * from ctx->int8_store / nvfp4_store); the file is env-selected:
 *   H3_DIT_NVFP4_FILE=<path>   (explicit)
 *   unset → int8 store path with ".safetensors" → ".nvfp4.safetensors"
 * FP4 GEMM produces BF16 [M,N]; epilogues apply w_scale2 * act_global
 * (act_global=1.0 today: act quantization is dynamic per-16 SF only), then
 * run the same SwiGLU / AdaLN-gate fusion structure as the INT8 R31
 * epilogues. epi_fuse INT8 requant chaining is bypassed on this track. */

static const char *nvfp4_linear_suffix(unsigned kind) {
    switch (kind) {
    case 0: return "attn.qkv_proj";
    case 1: return "attn.out_proj";
    case 2: return "mlp.fc1";
    case 3: return "mlp.fc2";
    default: return NULL;
    }
}

static int nvfp4_read_tensor(const h3_st_header *hdr, const char *name,
                             void *dst, size_t bytes, char *error,
                             size_t error_size) {
    const h3_st_tensor *t = h3_st_find(hdr, name);
    if (!t) {
        snprintf(error, error_size, "nvfp4 sidecar missing %s", name);
        return 0;
    }
    if ((size_t)(t->data_end - t->data_begin) != bytes) {
        snprintf(error, error_size, "nvfp4 sidecar %s size mismatch", name);
        return 0;
    }
    return h3_st_read_data(hdr, t, dst, bytes, error, error_size);
}

static int nvfp4_load_block_weights(h3_dit_w4a8 *ctx, unsigned block,
                                    char *error, size_t error_size) {
    unsigned p;
    size_t n_expect[4], k_expect[4];
    const char *base_path = NULL;
    char sidecar[768];
    char detail[384];
    static const size_t n_fix[4] = {21504u, 5376u, 28672u, 5376u};
    static const size_t k_fix[4] = {5376u, 7168u, 5376u, 14336u};

    for (p = 0; p < 4; p++) {
        n_expect[p] = n_fix[p];
        k_expect[p] = k_fix[p];
    }
    if (ctx->int8_store) {
        h3_dit_int8_linear lin;
        base_path = h3_dit_int8_store_header(ctx->int8_store)->path;
        for (p = 0; p < 4; p++) {
            if (h3_dit_int8_find_block_linear(ctx->int8_store, block,
                                              nvfp4_linear_suffix(p), &lin,
                                              detail, sizeof(detail))) {
                n_expect[p] = lin.n;
                k_expect[p] = lin.k;
            }
        }
    } else if (ctx->nvfp4_store) {
        h3_nvfp4_linear lin;
        base_path = ctx->nvfp4_store->path;
        for (p = 0; p < 4; p++) {
            if (h3_dit_nvfp4_find_block_linear(ctx->nvfp4_store, block,
                                               nvfp4_linear_suffix(p), &lin,
                                               detail, sizeof(detail))) {
                n_expect[p] = lin.n;
                k_expect[p] = lin.k;
            }
        }
    }
    if (!ctx->nvfp4_ready) {
        const char *env_file = getenv("H3_DIT_NVFP4_FILE");
        const char *dot;
        if (env_file && env_file[0]) {
            snprintf(sidecar, sizeof(sidecar), "%s", env_file);
        } else if (base_path && (dot = strrchr(base_path, '.')) &&
                   !strcmp(dot, ".safetensors")) {
            snprintf(sidecar, sizeof(sidecar), "%.*s.nvfp4.safetensors",
                     (int)(dot - base_path), base_path);
        } else {
            fail(error, error_size,
                 "H3_DIT_NVFP4=1 needs H3_DIT_NVFP4_FILE (no derivable path)");
            return 0;
        }
        if (!h3_st_read_header(sidecar, &ctx->nvfp4_hdr, detail,
                               sizeof(detail))) {
            snprintf(error, error_size, "nvfp4 sidecar open %s: %s", sidecar,
                     detail);
            return 0;
        }
        fprintf(stderr, "dit_nvfp4: sidecar=%s tensors=%zu\n", sidecar,
                ctx->nvfp4_hdr.tensor_count);
        fflush(stderr);
        ctx->nvfp4_ready = 1;
    }
    for (p = 0; p < 4; p++) {
        char wname[320], sname[320], s2name[320];
        uint8_t *h_w = NULL, *h_sf = NULL;
        float scale2 = 1.0f;
        size_t n = n_expect[p], k = k_expect[p];
        size_t w_bytes = n * (k / 2u);
        /* Same rb*ncb*512 as h3_cuda_nvfp4_sf_bytes / quant_dit_nvfp4.py. */
        size_t sf_bytes = h3_cuda_nvfp4_sf_bytes(n, k);
        const h3_st_tensor *s2t;
        if (ctx->nvfp4_w[block][p]) continue;
        if (!nvfp4_kind_use_fp4(ctx, p))
            continue; /* KEEP_INT8: leave NULL; INT8 path loads lazily */
        snprintf(wname, sizeof(wname), "blocks.%u.%s.weight_fp4", block,
                 nvfp4_linear_suffix(p));
        snprintf(sname, sizeof(sname), "blocks.%u.%s.weight_sf", block,
                 nvfp4_linear_suffix(p));
        snprintf(s2name, sizeof(s2name), "blocks.%u.%s.weight_scale2", block,
                 nvfp4_linear_suffix(p));
        s2t = h3_st_find(&ctx->nvfp4_hdr, s2name);
        if (s2t &&
            !h3_st_read_data(&ctx->nvfp4_hdr, s2t, &scale2, sizeof(float),
                             detail, sizeof(detail))) {
            snprintf(error, error_size, "nvfp4 %s: %s", s2name, detail);
            return 0;
        }
        h_w = (uint8_t *)malloc(w_bytes);
        h_sf = (uint8_t *)malloc(sf_bytes);
        if (!h_w || !h_sf) {
            free(h_w);
            free(h_sf);
            fail(error, error_size, "OOM nvfp4 host staging");
            return 0;
        }
        if (!nvfp4_read_tensor(&ctx->nvfp4_hdr, wname, h_w, w_bytes, detail,
                               sizeof(detail)) ||
            !nvfp4_read_tensor(&ctx->nvfp4_hdr, sname, h_sf, sf_bytes, detail,
                               sizeof(detail))) {
            free(h_w);
            free(h_sf);
            snprintf(error, error_size, "%s", detail);
            return 0;
        }
        /* Loader takes ownership of h_w / h_sf on success. */
        if (!h3_cuda_nvfp4_weight_create(ctx->cuda, h_w, h_sf, scale2, n, k,
                                         &ctx->nvfp4_w[block][p], NULL,
                                         detail, sizeof(detail))) {
            free(h_w);
            free(h_sf);
            snprintf(error, error_size, "nvfp4 weight create b%u.%s: %s",
                     block, nvfp4_linear_suffix(p), detail);
            return 0;
        }
    }
    return 1;
}

/* Ensure d_nvfp4_rot can hold [seq, k] f32 (k <= FFN). */
static int nvfp4_ensure_rot(h3_dit_w4a8 *ctx, size_t seq, size_t k,
                            char *error, size_t error_size) {
    size_t need;
    const size_t max_k = (size_t)H3_DIT_W4A8_FFN;
    if (k == 0 || k > max_k) {
        snprintf(error, error_size, "nvfp4 rot: bad k=%zu (max %zu)", k, max_k);
        return 0;
    }
    if (ctx->d_nvfp4_rot && seq <= ctx->nvfp4_rot_cap_seq)
        return 1;
    need = seq * max_k * sizeof(float);
    if (need > ctx->d_nvfp4_rot_bytes) {
        if (ctx->d_nvfp4_rot) {
            cudaFree(ctx->d_nvfp4_rot);
            ctx->d_nvfp4_rot = NULL;
        }
        if (cudaMalloc((void **)&ctx->d_nvfp4_rot, need) != cudaSuccess) {
            ctx->d_nvfp4_rot_bytes = 0;
            ctx->nvfp4_rot_cap_seq = 0;
            snprintf(error, error_size, "nvfp4 rot scratch OOM (%zu bytes)", need);
            return 0;
        }
        ctx->d_nvfp4_rot_bytes = need;
    }
    ctx->nvfp4_rot_cap_seq = seq;
    return 1;
}

/* When weights are ConvRot-baked (ctx->convrot_gs > 0), rotate acts into
 * d_nvfp4_rot and return that pointer. Otherwise return d_x unchanged.
 * H3_DIT_NVFP4_SKIP_ACT_CONVROT=1 forces the old broken path. */
static const float *nvfp4_act_maybe_convrot(h3_dit_w4a8 *ctx, const float *d_x,
                                            size_t seq, size_t k, char *error,
                                            size_t error_size) {
    int gs = ctx->convrot_gs;
    char detail[384];
    if (gs <= 0 || nvfp4_skip_act_convrot()) {
        if (gs > 0 && !ctx->nvfp4_act_convrot_logged) {
            fprintf(stderr,
                    "dit_nvfp4: act_convrot=0 SKIP_ACT_CONVROT "
                    "(W_rot @ unrotated act — expected mosaic)\n");
            fflush(stderr);
            ctx->nvfp4_act_convrot_logged = 1;
        }
        return d_x;
    }
    if (!nvfp4_ensure_rot(ctx, seq, k, error, error_size))
        return NULL;
    if (!h3_cuda_convrot_act_f32_dd(ctx->cuda, d_x, seq, k, gs,
                                    ctx->d_nvfp4_rot, detail,
                                    sizeof(detail))) {
        snprintf(error, error_size, "nvfp4 act convrot: %s", detail);
        return NULL;
    }
    if (!ctx->nvfp4_act_convrot_logged) {
        fprintf(stderr,
                "dit_nvfp4: act_convrot=1 gs=%d "
                "(sidecar W_rot; SKIP_ACT_CONVROT=1 for broken A/B)\n",
                gs);
        fflush(stderr);
        ctx->nvfp4_act_convrot_logged = 1;
    }
    return ctx->d_nvfp4_rot;
}

static int nvfp4_ensure_scratch(h3_dit_w4a8 *ctx, size_t seq, char *error,
                                size_t error_size) {
    size_t need, widest_n;
    if (ctx->d_nvfp4_scratch && seq <= ctx->nvfp4_scratch_cap_seq)
        return 1;
    widest_n = 28672u; /* max(21504 qkv, 28672 fc1, 5376 out/fc2) */
    need = 0;
    need += h3_cuda_nvfp4_gemm_scratch_bytes(seq, 21504u, 5376u);
    {
        size_t b = h3_cuda_nvfp4_gemm_scratch_bytes(seq, widest_n, 14336u);
        if (b > need) need = b;
    }
    if (need > ctx->d_nvfp4_scratch_bytes) {
        if (ctx->d_nvfp4_scratch) {
            cudaFree(ctx->d_nvfp4_scratch);
            ctx->d_nvfp4_scratch = NULL;
        }
        if (cudaMalloc((void **)&ctx->d_nvfp4_scratch, need) != cudaSuccess) {
            ctx->d_nvfp4_scratch_bytes = 0;
            ctx->nvfp4_scratch_cap_seq = 0;
            fail(error, error_size, "nvfp4 scratch OOM");
            return 0;
        }
        ctx->d_nvfp4_scratch_bytes = need;
    }
    ctx->nvfp4_scratch_cap_seq = seq;
    return 1;
}

/* P4.2 Phase-1: ensure the chained FC1→FC2 scratch slab. Separate from
 * d_nvfp4_scratch so toggling the chain never disturbs the fallback layout. */
static int nvfp4_chain_ensure_scratch(h3_dit_w4a8 *ctx, size_t seq,
                                      char *error, size_t error_size) {
    size_t need;
    if (ctx->d_nvfp4_chain_scratch && seq <= ctx->nvfp4_chain_scratch_cap_seq)
        return 1;
    need = h3_cuda_nvfp4_chain_scratch_bytes(seq, H3_DIT_W4A8_HIDDEN,
                                             H3_DIT_W4A8_FFN);
    if (need > ctx->d_nvfp4_chain_scratch_bytes) {
        if (ctx->d_nvfp4_chain_scratch) {
            cudaFree(ctx->d_nvfp4_chain_scratch);
            ctx->d_nvfp4_chain_scratch = NULL;
        }
        if (cudaMalloc((void **)&ctx->d_nvfp4_chain_scratch, need) !=
            cudaSuccess) {
            ctx->d_nvfp4_chain_scratch_bytes = 0;
            ctx->nvfp4_chain_scratch_cap_seq = 0;
            fail(error, error_size, "nvfp4 chain scratch OOM");
            return 0;
        }
        ctx->d_nvfp4_chain_scratch_bytes = need;
    }
    ctx->nvfp4_chain_scratch_cap_seq = seq;
    return 1;
}

/* P4.2 Phase-1 (方案 B): chained FC1→SwiGLU→requant→FC2 with the FP32 acc
 * never leaving the device in BF16/quantized-transit form.
 *
 *   quantize(d_a) -> fc1_in FP4+SF           (reuse the standard act quant)
 *   CUTLASS FC1 GEMM -> fc1_acc FP32 [seq, 2*ffn]   (no BF16 rounding)
 *   swiglu+requant -> act FP4+SF [seq, ffn]  (fused, no FP32 transit)
 *   FC2 pre_q GEMM + AdaLN gate -> d_y FP32  (skips re-quantize)
 *
 * Returns 1 on success (chain ran), 0 on hard failure (caller falls back to
 * the standard per-linear nvfp4_gemm_dispatch path). */
static int nvfp4_chain_fc1_fc2(h3_dit_w4a8 *ctx, unsigned block,
                               const float *d_a, const float *d_b,
                               const float *d_mod, const uint32_t *d_map,
                               const float *d_lora_fc1, const float *d_lora_fc2,
                               float *d_y, size_t seq, double *prep_ms,
                               double *gemm_ms, char *error,
                               size_t error_size) {
    const size_t kdim = H3_DIT_W4A8_HIDDEN;
    const size_t ffn = H3_DIT_W4A8_FFN;
    const h3_cuda_nvfp4_weight *w_fc1, *w_fc2;
    uint8_t *base, *fc1_in_fp4, *fc1_in_sf, *act_fp4, *act_sf;
    float *fc1_acc;
    size_t m_pad, sf_rb, sf_ncb_kdim, sf_ncb_ffn;
    size_t off_in_fp4, off_in_sf, off_acc, off_act_fp4, off_act_sf;
    double g1 = 0.0, g2 = 0.0;
    char detail[384];

    if (prep_ms) *prep_ms = 0.0;
    if (gemm_ms) *gemm_ms = 0.0;
    if (!nvfp4_load_block_weights(ctx, block, error, error_size))
        return 0;
    if (!nvfp4_chain_ensure_scratch(ctx, seq, error, error_size))
        return 0;
    w_fc1 = ctx->nvfp4_w[block][2];
    w_fc2 = ctx->nvfp4_w[block][3];
    if (!w_fc1 || !w_fc2) {
        fail(error, error_size, "nvfp4 chain: fc1/fc2 weight missing");
        return 0;
    }

    /* Carve the slab. */
    m_pad = (seq + 127u) & ~(size_t)127u;
    sf_rb = m_pad / 128u;
    sf_ncb_kdim = (((kdim / 16u) + 3u) / 4u + 31u) & ~(size_t)31u;
    sf_ncb_ffn = (((ffn / 16u) + 3u) / 4u + 31u) & ~(size_t)31u;
    base = ctx->d_nvfp4_chain_scratch;
    off_in_fp4 = 0;
    off_in_sf = off_in_fp4 + m_pad * (kdim / 2u);
    off_acc = off_in_sf + sf_rb * sf_ncb_kdim * 512u;
    off_act_fp4 = off_acc + m_pad * (2u * ffn) * 4u;
    off_act_sf = off_act_fp4 + m_pad * (ffn / 2u);
    fc1_in_fp4 = base + off_in_fp4;
    fc1_in_sf = base + off_in_sf;
    fc1_acc = (float *)(base + off_acc);
    act_fp4 = base + off_act_fp4;
    act_sf = base + off_act_sf;

    /* 1. Quantize the FC1 input activation (d_a, [seq, kdim]) -> FP4+SF. */
    if (!h3_cuda_quantize_act_nvfp4_dev(ctx->cuda, d_a, seq, kdim, fc1_in_fp4,
                                        fc1_in_sf, detail, sizeof(detail))) {
        snprintf(error, error_size, "nvfp4 chain fc1 quant: %s", detail);
        return 0;
    }
    if (m_pad > seq) {
        size_t x_tail = (m_pad - seq) * (kdim / 2u);
        if (x_tail)
            cudaMemsetAsync(fc1_in_fp4 + seq * (kdim / 2u), 0, x_tail,
                            (cudaStream_t)h3_cuda_op_stream(ctx->cuda));
    }

    /* 2. CUTLASS FC1 GEMM -> FP32 acc [seq, 2*ffn] (direct output). */
    if (!h3_cutlass_nvfp4_fc1_f32_run(
            ctx->cuda, h3_cuda_nvfp4_weight_packed(w_fc1),
            h3_cuda_nvfp4_weight_sf(w_fc1), fc1_in_fp4, fc1_in_sf, seq,
            2u * ffn, kdim, fc1_acc, &g1, detail, sizeof(detail))) {
        snprintf(error, error_size, "nvfp4 chain fc1 gemm: %s", detail);
        return 0;
    }

    /* 3. Fused SwiGLU + requant -> act FP4+SF [seq, ffn]. */
    if (!h3_nvfp4_swiglu_requant_dev(ctx->cuda, fc1_acc,
                                     h3_cuda_nvfp4_weight_scale2(w_fc1),
                                     d_lora_fc1, seq, ffn, act_fp4, act_sf,
                                     detail, sizeof(detail))) {
        snprintf(error, error_size, "nvfp4 chain swiglu_requant: %s", detail);
        return 0;
    }
    if (m_pad > seq) {
        size_t x_tail = (m_pad - seq) * (ffn / 2u);
        if (x_tail)
            cudaMemsetAsync(act_fp4 + seq * (ffn / 2u), 0, x_tail,
                            (cudaStream_t)h3_cuda_op_stream(ctx->cuda));
    }

    /* 4. FC2 pre_q GEMM + AdaLN gate (slot 5, residual d_b) -> d_y. Reuse the
     * bf16 region of the standard FP4 scratch as the FC2 GEMM output. */
    if (!nvfp4_ensure_scratch(ctx, seq, error, error_size))
        return 0;
    {
        size_t sf_ncb_w = (((ffn / 16u) + 3u) / 4u + 31u) & ~(size_t)31u;
        size_t sf_rb_w = m_pad / 128u;
        void *d_bf16 = ctx->d_nvfp4_scratch + m_pad * (ffn / 2u) +
                       sf_rb_w * sf_ncb_w * 512u;
        if (!h3_cuda_gemm_fp4_pre_q_dev(ctx->cuda, w_fc2, act_fp4, act_sf, seq,
                                        d_b, d_mod, d_map,
                                        (size_t)H3_DIT_W4A8_SLOTS, 5u,
                                        d_lora_fc2, d_y, d_bf16, &g2, detail,
                                        sizeof(detail), "chain.fc2")) {
            snprintf(error, error_size, "nvfp4 chain fc2: %s", detail);
            return 0;
        }
    }

    if (prep_ms) *prep_ms = 0.0; /* quantize is folded; no separate prep timer */
    if (gemm_ms) *gemm_ms = g1 + g2;
    return 1;
}

/* Per-linear FP4 dispatch. kind: 0=QKV 1=OUT 2=FC1 3=FC2. Handles the act
 * quant + FP4 GEMM + BF16 epilogue (swiglu for FC1, AdaLN gate for
 * OUT/FC2). Returns 0 only on hard error; caller keeps stats. */
static int nvfp4_gemm_dispatch(h3_dit_w4a8 *ctx, unsigned block, unsigned kind,
                               const float *d_x, float *d_y, size_t seq,
                               const float *d_residual, const float *d_mod,
                               const uint32_t *d_map, size_t gate_slot,
                               const float *d_lora, double *prep_ms,
                               double *gemm_ms, char *error,
                               size_t error_size) {
    const h3_cuda_nvfp4_weight *w;
    const float *d_x_use;
    size_t k;
    char detail[384];
    char tag[32];
    if (!nvfp4_load_block_weights(ctx, block, error, error_size))
        return 0;
    if (!nvfp4_ensure_scratch(ctx, seq, error, error_size))
        return 0;
    w = ctx->nvfp4_w[block][kind];
    if (!w) {
        fail(error, error_size, "nvfp4 weight missing");
        return 0;
    }
    k = h3_cuda_nvfp4_weight_k(w);
    /* Sidecar FP4 weights are ConvRot-baked (same basis as INT8 W_rot).
     * Rotate activations before FP4 quant — matches INT8 gemm_f32_dd path.
     * Missing this was the C001 NVFP4 mosaic root cause (2026-10-27). */
    d_x_use = nvfp4_act_maybe_convrot(ctx, d_x, seq, k, error, error_size);
    if (!d_x_use)
        return 0;
    snprintf(tag, sizeof(tag), "b%u.%s", block, nvfp4_linear_suffix(kind));
    if (h3_nvfp4_debug_nan_dev()) {
        h3_nvfp4_debug_scan_dev(ctx->cuda, d_x_use, seq * k, "f32", tag,
                                "dispatch_in");
    }
    if (kind == 2u) {
        if (!h3_cuda_gemm_fp4_w_act_dev_swiglu(
                ctx->cuda, w, d_x_use, 1.0f, seq, d_lora, d_y,
                ctx->d_nvfp4_scratch, prep_ms, gemm_ms, detail,
                sizeof(detail), tag)) {
            snprintf(error, error_size, "nvfp4 fc1: %s", detail);
            return 0;
        }
        return 1;
    }
    if (kind == 1u || kind == 3u) {
        if (!h3_cuda_gemm_fp4_w_act_dev_gate(
                ctx->cuda, w, d_x_use, 1.0f, seq, d_residual, d_mod, d_map,
                (size_t)H3_DIT_W4A8_SLOTS, gate_slot, d_lora, d_y,
                ctx->d_nvfp4_scratch, prep_ms, gemm_ms, detail,
                sizeof(detail), tag)) {
            snprintf(error, error_size, "nvfp4 %s: %s",
                     kind == 1u ? "out" : "fc2", detail);
            return 0;
        }
        return 1;
    }
    if (!h3_cuda_gemm_fp4_w_act_dev(ctx->cuda, w, d_x_use, 1.0f, seq, d_y,
                                    ctx->d_nvfp4_scratch, prep_ms, gemm_ms,
                                    detail, sizeof(detail), tag)) {
        snprintf(error, error_size, "nvfp4 qkv: %s", detail);
        return 0;
    }
    return 1;
}

/* Phase 1: AdaLN(attn) + QKV GEMM (+LoRA residual) on local rows.
 * Ulysses insertion point is directly after this (A2A on d_qkv). */
/* nvfp4 track: the 4 FP4-ized linears are not pre-resident; resolve each
 * INT8 fallback lazily right before its (skipped-by-FP4) GEMM phase. A
 * successful FP4 dispatch jumps past the resolve, so nothing loads. */
static const h3_cuda_w4a8_weight *dit_lazy_int8(h3_dit_w4a8 *ctx,
                                                unsigned block,
                                                h3_w4a8_linear_kind kind,
                                                char *error,
                                                size_t error_size) {
    if (!ctx->nvfp4_on) return NULL;
    return get_weight(ctx, block, kind, error, error_size);
}

static int block_phase_qkv(h3_dit_w4a8 *ctx, unsigned block,
                           const h3_cuda_w4a8_weight *w_qkv, float *d_x,
                           float *d_mod, uint32_t *d_map, float *d_n1,
                           float *d_a, float *d_qkv, size_t seq,
                           h3_dit_w4a8_block_stats *stats, char *error,
                           size_t error_size) {
    const size_t hidden = H3_DIT_W4A8_HIDDEN;
    const size_t slots = H3_DIT_W4A8_SLOTS;
    const float eps = 1e-5f;
    double prep = 0.0, gemm = 0.0, op = 0.0;
    char backend[32];
    char detail[384];

    uly_klog("adaln_attn", "enter");
    if (!h3_cuda_adaln_f32_dev(ctx->cuda, d_x, d_n1, d_mod, d_map, seq, hidden,
                               slots, 0, 1, eps, d_a, &op, detail,
                               sizeof(detail))) {
        snprintf(error, error_size, "adaln_attn: %s", detail);
        return 0;
    }
    if (!uly_ksync("adaln_attn", error, error_size))
        return 0;
    if (stats) stats->adaln_attn_ms = op;

    prep = gemm = 0.0;
    uly_klog("qkv_gemm", "enter");
    if (nvfp4_kind_use_fp4(ctx, 0u)) {
        /* FP4 track: act quant + FP4 cuBLASLt GEMM + BF16→F32 dequant. */
        snprintf(backend, sizeof(backend), "%s", "cublasLt_fp4");
        if (nvfp4_gemm_dispatch(ctx, block, 0u, d_a, d_qkv, seq, NULL, NULL,
                                NULL, 0u, NULL, &prep, &gemm, detail,
                                sizeof(detail))) {
            goto qkv_done;
        }
        fprintf(stderr, "dit_nvfp4: qkv b%u FP4 dispatch failed (%s); INT8\n",
                block, detail);
        fflush(stderr);
        backend[0] = '\0';
        prep = gemm = 0.0;
    }
    if (ctx->nvfp4_on && !w_qkv) {
        w_qkv = dit_lazy_int8(ctx, block, H3_W4A8_LINEAR_QKV, error,
                              error_size);
        if (!w_qkv) return 0;
    }
    if (!h3_cuda_gemm_i8_dequant_f32_w_act_dev(
            ctx->cuda, w_qkv, d_a, ctx->convrot_gs, seq, d_qkv, &prep, &gemm,
            backend, sizeof(backend), detail, sizeof(detail))) {
        snprintf(error, error_size, "qkv: %s", detail);
        return 0;
    }
qkv_done:
    if (!uly_ksync("qkv_gemm", error, error_size))
        return 0;
    if (ctx->lora) {
        uly_klog("qkv_lora", "enter");
        if (!dit_lora_residual_dev(ctx, block, 0, d_a, d_qkv, seq, 1.0f, error,
                                   error_size))
            return 0;
        if (!uly_ksync("qkv_lora", error, error_size))
            return 0;
    }
    if (stats) {
        stats->qkv_ms = gemm;
        stats->act_prep_ms += prep;
        snprintf(stats->qkv_backend, sizeof(stats->qkv_backend), "%s", backend);
    }
    return 1;
}

/* Phase 2: RoPE + SDPA. seq_attn/heads_attn are the attention geometry
 * (U=1: local seq × full heads; Ulysses: S_real gathered rows × H/U heads —
 * pad rows are excluded here so zero-K rows never pollute softmax). */
static int block_phase_attn(h3_dit_w4a8 *ctx, unsigned block, const float *d_qkv,
                            const float *d_qn, const float *d_kn,
                            const float *d_cos, const float *d_sin,
                            size_t seq_attn, size_t heads_attn, float *d_attn,
                            h3_dit_w4a8_block_stats *stats, char *error,
                            size_t error_size) {
    const size_t head_dim = H3_DIT_W4A8_HEAD_DIM;
    const size_t rope_half = (size_t)H3_DIT_W4A8_ROPE_HALF;
    const float eps = 1e-5f;
    const float attn_scale = 1.0f / sqrtf((float)head_dim);
    int attn_window = env_int_default("H3_DIT_W4A8_ATTN_WINDOW", 0);
    char be[32];
    char detail[384];
    double rope_ms = 0.0, op = 0.0;
    be[0] = '\0';
    if (ctx && ctx->ckv_on && ctx->ckv_mode && ctx->d_ckv_k &&
        block < (unsigned)H3_DIT_W4A8_N_BLOCKS) {
        size_t row = (size_t)H3_DIT_W4A8_HEADS * (size_t)H3_DIT_W4A8_HEAD_DIM;
        size_t layer_bytes = ctx->ckv_max * row * sizeof(uint16_t);
        char *k_base = (char *)ctx->d_ckv_k + (size_t)block * layer_bytes;
        char *v_base = (char *)ctx->d_ckv_v + (size_t)block * layer_bytes;
        if (ctx->ckv_n > 0)
            h3_cuda_sdpa_bind_kvcat(k_base, v_base, ctx->ckv_n);
        else
            h3_cuda_sdpa_bind_kvcat(NULL, NULL, 0);
        if (ctx->ckv_mode == 2 && ctx->ckv_save_n > 0) {
            size_t off = ctx->ckv_save_dst * row * sizeof(uint16_t);
            h3_cuda_sdpa_bind_kvsave(k_base + off, v_base + off, ctx->d_ckv_idx,
                                     ctx->ckv_save_n);
        }
    }
    /* RoPE → FP16 scratch + SDPA (skip f32 Q/K/V materialize + cast). */
    uly_klog("rope_sdpa", "enter");
    if (!h3_cuda_qkv_rope_sdpa_f16_dev(
            ctx->cuda, d_qkv, d_qn, d_kn, d_cos, d_sin, seq_attn, heads_attn,
            head_dim, rope_half, eps, attn_scale, attn_window, d_attn, &rope_ms,
            &op, be, sizeof(be), detail, sizeof(detail))) {
        h3_cuda_sdpa_clear_kvext();
        snprintf(error, error_size, "rope_sdpa: %s", detail);
        return 0;
    }
    h3_cuda_sdpa_clear_kvext();
    if (!uly_ksync("rope_sdpa", error, error_size))
        return 0;
    if (stats) {
        stats->rope_ms = rope_ms;
        stats->sdpa_ms = op;
        snprintf(stats->sdpa_backend, sizeof(stats->sdpa_backend), "%s",
                 be[0] ? be : "cuda_rope_sdpa_f16");
    }
    return 1;
}

/* Phase 3: out-proj (+LoRA) + gates + MLP (+LoRA) on local rows.
 * d_attn is the attention output for the local rows ([seq, H*D]). */
static int block_phase_out_mlp(
    h3_dit_w4a8 *ctx, unsigned block, const h3_cuda_w4a8_weight *w_out,
    const h3_cuda_w4a8_weight *w_fc1, const h3_cuda_w4a8_weight *w_fc2,
    float *d_x, float *d_y, float *d_mod, uint32_t *d_map, float *d_n2,
    float *d_a, float *d_b, float *d_qkv, float *d_wide, float *d_attn,
    size_t seq, h3_dit_w4a8_block_stats *stats, char *error,
    size_t error_size) {
    const size_t hidden = H3_DIT_W4A8_HIDDEN;
    const size_t ffn = H3_DIT_W4A8_FFN;
    const size_t slots = H3_DIT_W4A8_SLOTS;
    const float eps = 1e-5f;
    double prep = 0.0, gemm = 0.0, op = 0.0;
    char backend[32];
    char detail[384];
    int fused = h3_dit_w4a8_fused_epilogue();
    /* FP4 track fuses out-gate / swiglu / fc2-gate in its own BF16
     * epilogues, which take over the INT8 R31 epilogue's role; keep the
     * INT8 GEMM epilogue off so the non-FP4 fallback paths stay separate. */
    /* KEEP_INT8=all → fp4_mask=0 → INT8 fused epi allowed (BIT_ID path). */
    int gemm_epi = fused && !ctx->nvfp4_fp4_mask;
    /* E2 opt-in (H3_DIT_EPI_FUSE=1): intended to collapse dequant→requant pairs.
     * Requires R31 gemm_epi + act_f16 + convrot.
     *
     * P1 (OUT→FC1) NOT wired: AdaLN sits between OUT gate and FC1; requant of
     * d_x skipped AdaLN → NaN on 238 (see c003 nan-fix).
     *
     * P2 (FC1 swiglu→FC2) skip-ConvRot pre_q is DISARMED: packed weights are
     * W_rot=W@H, so Y=X@W.T == (X@H)@W_rot.T. Feeding quant(X) into pre_q
     * (skip act ConvRot) computes quant(X)@W_rot.T ≠ baseline. DESIGN §3
     * equated those; host Hadamard proof + 238 fuse↔ctrl latent rel_l2≈1.49
     * refute it. Keep env bit for logging; do not arm requant_mode/pre_q until
     * an in-kernel ConvRot+requant redesign. Default remains OFF. */
    int epi_fuse_req = gemm_epi && h3_dit_epi_fuse_wanted() &&
                       h3_dit_w4a8_act_f16() && ctx->convrot_gs > 0;
    const int epi_fuse = 0; /* skip-ConvRot DISARMED — never arm */
    /* Redesign arm: after FC1 SwiGLU, ConvRot+requant then FC2 pre_q.
     * Requires gemm_epi + convrot; independent of H3_DIT_EPI_FUSE.
     * Default OFF. Prefer H3_DIT_ACT_F16=0 for A/B (ink uses f32 algebra). */
    int epi_fuse_sl = gemm_epi && h3_dit_epi_fuse_sl_wanted() &&
                      ctx->convrot_gs > 0 && !ctx->nvfp4_fp4_mask;
    /* SL implies ink algebra (pre_q after fused ConvRot+requant). */
    int epi_fuse_ink = gemm_epi && ctx->convrot_gs > 0 && !ctx->nvfp4_fp4_mask &&
                       (h3_dit_epi_fuse_ink_wanted() || epi_fuse_sl);
    int8_t *fc2_pre_x = NULL;
    float *fc2_pre_act = NULL;
    h3_cuda_dit_gemm_epi epi;
    static int fused_logged;

    g_dit_gemm_epi = gemm_epi;
    if (!fused_logged) {
        fused_logged = 1;
        fprintf(stderr,
                "dit_fused_epi: %s gate_adaln=%d gemm_epi=%d lora=%d scale=%.4g "
                "overlap=%d epi_fuse_req=%d epi_fuse=%d epi_fuse_ink=%d "
                "epi_fuse_sl=%d "
                "(P2 skip-ConvRot DISARMED; ink/SL=ConvRot+requant) nvfp4=%d\n",
                fused ? "on" : "off", fused ? 1 : 0, gemm_epi ? 1 : 0,
                ctx->lora ? 1 : 0, ctx->lora ? ctx->lora->scale : 0.0,
                ctx->lora ? h3_dit_w4a8_lora_overlap() : 0,
                epi_fuse_req ? 1 : 0, epi_fuse ? 1 : 0,
                epi_fuse_ink ? 1 : 0, epi_fuse_sl ? 1 : 0,
                ctx->nvfp4_on ? 1 : 0);
        fflush(stderr);
    }
    memset(&epi, 0, sizeof(epi));

    prep = gemm = 0.0;
    uly_klog("out_gemm", "enter");
    if (nvfp4_kind_use_fp4(ctx, 1u)) {
        /* FP4 track: GEMM + AdaLN gate (slot 2) fused on the BF16 acc;
         * LoRA delta is added inside the BF16 gate epilogue (no pre-kick). */
        double fp4_prep = 0.0, fp4_gemm = 0.0;
        snprintf(backend, sizeof(backend), "%s", "cublasLt_fp4");
        if (nvfp4_gemm_dispatch(ctx, block, 1u, d_attn, d_b, seq, d_x, d_mod,
                                d_map, 2u, ctx->lora ? d_a : NULL, &fp4_prep,
                                &fp4_gemm, detail, sizeof(detail))) {
            prep = fp4_prep;
            gemm = fp4_gemm;
            goto out_done;
        }
        fprintf(stderr, "dit_nvfp4: out b%u FP4 dispatch failed (%s); INT8\n",
                block, detail);
        fflush(stderr);
        backend[0] = '\0';
        prep = gemm = 0.0;
    }
    if (ctx->nvfp4_on && !w_out) {
        w_out = dit_lazy_int8(ctx, block, H3_W4A8_LINEAR_OUT, error,
                              error_size);
        if (!w_out) return 0;
    }
    if (gemm_epi && ctx->lora) {
        void *ready = NULL;
        uly_klog("out_lora", "enter");
        if (!dit_lora_kick(ctx, block, 1, d_attn, d_a, seq, 0.0f, 1, &ready,
                           error, error_size))
            return 0;
        if (!ready && !uly_ksync("out_lora", error, error_size))
            return 0;
        epi.d_lora = d_a;
        epi.lora_ready = ready;
    }
    if (gemm_epi) {
        epi.gate = 1;
        epi.swiglu = 0;
        epi.d_residual = d_x;
        epi.d_modulation = d_mod;
        epi.d_row_map = d_map;
        epi.slots = slots;
        epi.gate_slot = 2;
        /* P1 disabled: do not requant d_x here (AdaLN → d_a is FC1's act). */
        if (!h3_cuda_gemm_i8_dequant_f32_w_act_dev_epi(
                ctx->cuda, w_out, d_attn, ctx->convrot_gs, seq, d_b, &prep,
                &gemm, backend, sizeof(backend), &epi, detail,
                sizeof(detail))) {
            snprintf(error, error_size, "out: %s", detail);
            return 0;
        }
    } else if (!h3_cuda_gemm_i8_dequant_f32_w_act_dev(
                   ctx->cuda, w_out, d_attn, ctx->convrot_gs, seq, d_a, &prep,
                   &gemm, backend, sizeof(backend), detail, sizeof(detail))) {
        snprintf(error, error_size, "out: %s", detail);
        return 0;
    }
out_done:
    if (!uly_ksync("out_gemm", error, error_size))
        return 0;
    if (!gemm_epi && ctx->lora) {
        uly_klog("out_lora", "enter");
        if (!dit_lora_residual_dev(ctx, block, 1, d_attn, d_a, seq, 1.0f, error,
                                   error_size))
            return 0;
        if (!uly_ksync("out_lora", error, error_size))
            return 0;
    }
    if (stats) {
        stats->out_ms = gemm;
        stats->act_prep_ms += prep;
        snprintf(stats->out_backend, sizeof(stats->out_backend), "%s",
                 backend);
    }

    if (gemm_epi) {
        uly_klog("adaln_mlp", "enter");
        if (!h3_cuda_adaln_f32_dev(ctx->cuda, d_b, d_n2, d_mod, d_map, seq,
                                   hidden, slots, 3, 4, eps, d_a, &op, detail,
                                   sizeof(detail))) {
            snprintf(error, error_size, "adaln_mlp: %s", detail);
            return 0;
        }
        if (!uly_ksync("adaln_mlp", error, error_size))
            return 0;
        if (stats) {
            stats->gate_attn_ms = 0.0;
            stats->adaln_mlp_ms = op;
        }
    } else if (fused) {
        uly_klog("gate_adaln", "enter");
        if (!h3_cuda_gate_adaln_f32_dev(
                ctx->cuda, d_x, d_a, d_n2, d_mod, d_map, seq, hidden, slots, 2,
                3, 4, eps, d_b, d_a, &op, detail, sizeof(detail))) {
            snprintf(error, error_size, "gate_adaln: %s", detail);
            return 0;
        }
        if (!uly_ksync("gate_adaln", error, error_size))
            return 0;
        if (stats) {
            stats->gate_attn_ms = 0.0;
            stats->adaln_mlp_ms = op;
        }
    } else {
        uly_klog("gate_attn", "enter");
        if (!h3_cuda_gate_f32_dev(ctx->cuda, d_x, d_a, d_mod, d_map, seq, hidden,
                                  slots, 2, d_b, &op, detail, sizeof(detail))) {
            snprintf(error, error_size, "gate_attn: %s", detail);
            return 0;
        }
        if (!uly_ksync("gate_attn", error, error_size))
            return 0;
        if (stats) stats->gate_attn_ms = op;

        uly_klog("adaln_mlp", "enter");
        if (!h3_cuda_adaln_f32_dev(ctx->cuda, d_b, d_n2, d_mod, d_map, seq,
                                   hidden, slots, 3, 4, eps, d_a, &op, detail,
                                   sizeof(detail))) {
            snprintf(error, error_size, "adaln_mlp: %s", detail);
            return 0;
        }
        if (!uly_ksync("adaln_mlp", error, error_size))
            return 0;
        if (stats) stats->adaln_mlp_ms = op;
    }

    {
        float *d_fc1 = d_wide;
        float *d_act = d_qkv;
        int fc1_fused = 0;

        prep = gemm = 0.0;
        uly_klog("fc1_gemm", "enter");
        if (ctx->nvfp4_chain_on &&
            !(ctx->convrot_gs > 0 && !nvfp4_skip_act_convrot())) {
            /* P4.2 Phase-1: chained FC1→SwiGLU→requant→FC2. Runs BOTH FC1 and
             * FC2 (writes d_y), so on success we skip the standalone swiglu
             * and the FC2 dispatch below. Falls back to the per-linear FP4
             * path (then INT8) on any chain failure.
             *
             * When convrot_gs>0 the fused swiglu+requant cannot ConvRot the
             * FC2 activation before FP4 quant — skip chain and use the
             * corrected per-linear path instead. */
            snprintf(backend, sizeof(backend), "%s", "cutlass_fp4_chain");
            if (nvfp4_chain_fc1_fc2(ctx, block, d_a, d_b, d_mod, d_map,
                                    ctx->lora ? d_fc1 : NULL,
                                    ctx->lora ? d_a : NULL, d_y, seq,
                                    &prep, &gemm, detail, sizeof(detail))) {
                if (stats) {
                    stats->fc1_ms = gemm;
                    stats->act_prep_ms += prep;
                    snprintf(stats->fc1_backend, sizeof(stats->fc1_backend),
                             "%s", backend);
                    stats->swiglu_ms = 0.0;
                    stats->fc2_ms = 0.0;
                    snprintf(stats->fc2_backend, sizeof(stats->fc2_backend),
                             "%s", backend);
                }
                if (!uly_ksync("fc1_gemm", error, error_size))
                    return 0;
                if (!uly_ksync("fc2_gemm", error, error_size))
                    return 0;
                return 1; /* whole MLP (fc1+swiglu+fc2) done */
            }
            fprintf(stderr,
                    "dit_nvfp4: fc1 b%u chain failed (%s); per-linear FP4\n",
                    block, detail);
            fflush(stderr);
            backend[0] = '\0';
            prep = gemm = 0.0;
        }
        if (nvfp4_kind_use_fp4(ctx, 2u)) {
            /* FP4 FC1: GEMM [seq,2*ffn] + fused SwiGLU (+lora) on BF16 acc;
             * d_act holds the gated half [seq,ffn]. */
            snprintf(backend, sizeof(backend), "%s", "cublasLt_fp4");
            if (nvfp4_gemm_dispatch(ctx, block, 2u, d_a, d_act, seq, NULL,
                                    NULL, NULL, 0u,
                                    ctx->lora ? d_fc1 : NULL, &prep, &gemm,
                                    detail, sizeof(detail))) {
                fc1_fused = 1; /* d_act already holds SwiGLU(FC1) */
                goto fc1_done;
            }
            fprintf(stderr,
                    "dit_nvfp4: fc1 b%u FP4 dispatch failed (%s); INT8\n",
                    block, detail);
            fflush(stderr);
            backend[0] = '\0';
            prep = gemm = 0.0;
        }
        if (ctx->nvfp4_on && !w_fc1) {
            w_fc1 = dit_lazy_int8(ctx, block, H3_W4A8_LINEAR_FC1, error,
                                  error_size);
            if (!w_fc1) return 0;
        }
        if (gemm_epi && ctx->lora) {
            void *ready = NULL;
            uly_klog("fc1_lora", "enter");
            if (!dit_lora_kick(ctx, block, 2, d_a, d_fc1, seq, 0.0f, 1, &ready,
                               error, error_size))
                return 0;
            if (!ready && !uly_ksync("fc1_lora", error, error_size))
                return 0;
            memset(&epi, 0, sizeof(epi));
            epi.swiglu = 1;
            epi.d_lora = d_fc1;
            epi.lora_ready = ready;
            if (epi_fuse) {
                epi.requant_mode = 1;
                epi.requant_gs = (int)ffn; /* P2: swiglu out width */
            }
            if (!h3_cuda_gemm_i8_dequant_f32_w_act_dev_epi(
                    ctx->cuda, w_fc1, d_a, ctx->convrot_gs, seq, d_act, &prep,
                    &gemm, backend, sizeof(backend), &epi, detail,
                    sizeof(detail))) {
                snprintf(error, error_size, "fc1: %s", detail);
                return 0;
            }
            if (epi_fuse) {
                fc2_pre_x = epi.d_rq_x;
                fc2_pre_act = epi.d_rq_act;
            }
        } else if (gemm_epi) {
            memset(&epi, 0, sizeof(epi));
            epi.swiglu = 1;
            if (epi_fuse) {
                epi.requant_mode = 1;
                epi.requant_gs = (int)ffn;
            }
            if (!h3_cuda_gemm_i8_dequant_f32_w_act_dev_epi(
                    ctx->cuda, w_fc1, d_a, ctx->convrot_gs, seq, d_act, &prep,
                    &gemm, backend, sizeof(backend), &epi, detail,
                    sizeof(detail))) {
                snprintf(error, error_size, "fc1: %s", detail);
                return 0;
            }
            if (epi_fuse) {
                fc2_pre_x = epi.d_rq_x;
                fc2_pre_act = epi.d_rq_act;
            }
        } else if (!h3_cuda_gemm_i8_dequant_f32_w_act_dev(
                       ctx->cuda, w_fc1, d_a, ctx->convrot_gs, seq, d_fc1,
                       &prep, &gemm, backend, sizeof(backend), detail,
                       sizeof(detail))) {
            snprintf(error, error_size, "fc1: %s", detail);
            return 0;
        }
fc1_done:
        if (!uly_ksync("fc1_gemm", error, error_size))
            return 0;
        if (!gemm_epi && ctx->lora) {
            uly_klog("fc1_lora", "enter");
            if (!dit_lora_residual_dev(ctx, block, 2, d_a, d_fc1, seq, 1.0f,
                                       error, error_size))
                return 0;
            if (!uly_ksync("fc1_lora", error, error_size))
                return 0;
        }
        if (stats) {
            stats->fc1_ms = gemm;
            stats->act_prep_ms += prep;
            snprintf(stats->fc1_backend, sizeof(stats->fc1_backend), "%s",
                     backend);
        }

        /* FP4 FC1 already wrote SwiGLU into d_act; a standalone swiglu here would
         * re-read the stale d_fc1 and overwrite d_act (FC2 then consumed garbage). */
        if (!gemm_epi && !fc1_fused) {
            prep = gemm = 0.0;
            uly_klog("swiglu", "enter");
            if (!h3_cuda_swiglu_f32_dev(ctx->cuda, d_fc1, seq, ffn, d_act, &op,
                                        detail, sizeof(detail))) {
                snprintf(error, error_size, "swiglu: %s", detail);
                return 0;
            }
            if (!uly_ksync("swiglu", error, error_size))
                return 0;
            if (stats) stats->swiglu_ms = op;
        } else if (stats) {
            stats->swiglu_ms = 0.0;
        }

        /* EPI_FUSE_INK: d_act is SwiGLU(FC1) f32. ConvRot+requant into rq
         * scratch; FC2 pre_q skips act ConvRot (weights are W_rot). */
        if (epi_fuse_ink && gemm_epi) {
            char ink_err[256];
            if (!h3_cuda_epi_fuse_ink_prep_fc2(
                    ctx->cuda, d_act, seq, ffn, ctx->convrot_gs, &fc2_pre_x,
                    &fc2_pre_act, ink_err, sizeof(ink_err))) {
                snprintf(error, error_size, "epi_fuse_ink: %s", ink_err);
                return 0;
            }
        }

        prep = gemm = 0.0;
        uly_klog("fc2_gemm", "enter");
        if (nvfp4_kind_use_fp4(ctx, 3u)) {
            /* FP4 FC2: GEMM + AdaLN gate (slot 5, residual d_b) fused on
             * the BF16 acc; writes d_y directly. */
            snprintf(backend, sizeof(backend), "%s", "cublasLt_fp4");
            if (nvfp4_gemm_dispatch(ctx, block, 3u, d_act, d_y, seq, d_b,
                                    d_mod, d_map, 5u, ctx->lora ? d_a : NULL,
                                    &prep, &gemm, detail, sizeof(detail))) {
                goto fc2_done;
            }
            fprintf(stderr,
                    "dit_nvfp4: fc2 b%u FP4 dispatch failed (%s); INT8\n",
                    block, detail);
            fflush(stderr);
            backend[0] = '\0';
            prep = gemm = 0.0;
        }
        if (ctx->nvfp4_on && !w_fc2) {
            w_fc2 = dit_lazy_int8(ctx, block, H3_W4A8_LINEAR_FC2, error,
                                  error_size);
            if (!w_fc2) return 0;
        }
        if (gemm_epi && ctx->lora) {
            void *ready = NULL;
            uly_klog("fc2_lora", "enter");
            if (!dit_lora_kick(ctx, block, 3, d_act, d_a, seq, 0.0f, 1, &ready,
                               error, error_size))
                return 0;
            if (!ready && !uly_ksync("fc2_lora", error, error_size))
                return 0;
            memset(&epi, 0, sizeof(epi));
            epi.gate = 1;
            epi.d_residual = d_b;
            epi.d_modulation = d_mod;
            epi.d_row_map = d_map;
            epi.slots = slots;
            epi.gate_slot = 5;
            epi.d_lora = d_a;
            epi.lora_ready = ready;
            if ((epi_fuse || epi_fuse_ink) && fc2_pre_x && fc2_pre_act) {
                if (!h3_cuda_gemm_i8_dequant_f32_dev_pre_q(
                        ctx->cuda, w_fc2, fc2_pre_x, fc2_pre_act, seq, d_y,
                        &prep, &gemm, backend, sizeof(backend), &epi, detail,
                        sizeof(detail))) {
                    snprintf(error, error_size, "fc2: %s", detail);
                    return 0;
                }
            } else if (!h3_cuda_gemm_i8_dequant_f32_w_act_dev_epi(
                           ctx->cuda, w_fc2, d_act, ctx->convrot_gs, seq, d_y,
                           &prep, &gemm, backend, sizeof(backend), &epi,
                           detail, sizeof(detail))) {
                snprintf(error, error_size, "fc2: %s", detail);
                return 0;
            }
        } else if (gemm_epi) {
            memset(&epi, 0, sizeof(epi));
            epi.gate = 1;
            epi.d_residual = d_b;
            epi.d_modulation = d_mod;
            epi.d_row_map = d_map;
            epi.slots = slots;
            epi.gate_slot = 5;
            if ((epi_fuse || epi_fuse_ink) && fc2_pre_x && fc2_pre_act) {
                if (!h3_cuda_gemm_i8_dequant_f32_dev_pre_q(
                        ctx->cuda, w_fc2, fc2_pre_x, fc2_pre_act, seq, d_y,
                        &prep, &gemm, backend, sizeof(backend), &epi, detail,
                        sizeof(detail))) {
                    snprintf(error, error_size, "fc2: %s", detail);
                    return 0;
                }
            } else if (!h3_cuda_gemm_i8_dequant_f32_w_act_dev_epi(
                           ctx->cuda, w_fc2, d_act, ctx->convrot_gs, seq, d_y,
                           &prep, &gemm, backend, sizeof(backend), &epi,
                           detail, sizeof(detail))) {
                snprintf(error, error_size, "fc2: %s", detail);
                return 0;
            }
        } else if (!h3_cuda_gemm_i8_dequant_f32_w_act_dev(
                       ctx->cuda, w_fc2, d_act, ctx->convrot_gs, seq, d_a,
                       &prep, &gemm, backend, sizeof(backend), detail,
                       sizeof(detail))) {
            snprintf(error, error_size, "fc2: %s", detail);
            return 0;
        }
fc2_done:
        if (!uly_ksync("fc2_gemm", error, error_size))
            return 0;
        if (!gemm_epi && ctx->lora) {
            uly_klog("fc2_lora", "enter");
            if (!dit_lora_residual_dev(ctx, block, 3, d_act, d_a, seq, 1.0f,
                                       error, error_size))
                return 0;
            if (!uly_ksync("fc2_lora", error, error_size))
                return 0;
        }
        if (stats) {
            stats->fc2_ms = gemm;
            stats->act_prep_ms += prep;
            snprintf(stats->fc2_backend, sizeof(stats->fc2_backend), "%s",
                     backend);
        }
    }

    /* FP4 fc2 already applied the MLP gate in its BF16 epilogue (it writes
     * d_y), so skip the standalone gate like the INT8 R31 path does. */
    if (gemm_epi || ctx->nvfp4_on) {
        if (stats) stats->gate_mlp_ms = 0.0;
        return 1;
    }
    uly_klog("gate_mlp", "enter");
    if (!h3_cuda_gate_f32_dev(ctx->cuda, d_b, d_a, d_mod, d_map, seq, hidden,
                              slots, 5, d_y, &op, detail, sizeof(detail))) {
        snprintf(error, error_size, "gate_mlp: %s", detail);
        return 0;
    }
    if (!uly_ksync("gate_mlp", error, error_size))
        return 0;
    if (stats) stats->gate_mlp_ms = op;
    return 1;
}

/* Device compute for one resident block (no H2D/D2H). Used by eager + graph.
 * Pure composition of the three phases — identical instruction stream to the
 * pre-split single function (U=1 gold path unchanged). */
static int block_device_compute(
    h3_dit_w4a8 *ctx, unsigned block, const h3_cuda_w4a8_weight *w_qkv,
    const h3_cuda_w4a8_weight *w_out, const h3_cuda_w4a8_weight *w_fc1,
    const h3_cuda_w4a8_weight *w_fc2, float *d_x, float *d_y, float *d_mod,
    uint32_t *d_map, float *d_cos, float *d_sin, float *d_n1, float *d_n2,
    float *d_qn, float *d_kn, float *d_a, float *d_b, float *d_qkv,
    float *d_wide, size_t seq, h3_dit_w4a8_block_stats *stats, char *error,
    size_t error_size) {
    if (!block_phase_qkv(ctx, block, w_qkv, d_x, d_mod, d_map, d_n1, d_a,
                         d_qkv, seq, stats, error, error_size))
        return 0;
    if (!block_phase_attn(ctx, block, d_qkv, d_qn, d_kn, d_cos, d_sin, seq,
                          (size_t)H3_DIT_W4A8_HEADS, d_wide, stats, error,
                          error_size))
        return 0;
    return block_phase_out_mlp(ctx, block, w_out, w_fc1, w_fc2, d_x, d_y,
                               d_mod, d_map, d_n2, d_a, d_b, d_qkv, d_wide,
                               d_wide, seq, stats, error, error_size);
}

static void destroy_block_graph(h3_dit_w4a8 *ctx) {
    if (!ctx) return;
    if (ctx->graph_exec) {
        cudaGraphExecDestroy(ctx->graph_exec);
        ctx->graph_exec = NULL;
    }
    if (ctx->graph) {
        cudaGraphDestroy(ctx->graph);
        ctx->graph = NULL;
    }
    ctx->graph_ok = 0;
    ctx->graph_seq = 0;
    ctx->graph_block = 0;
}

static void destroy_step_graph(h3_dit_w4a8 *ctx) {
    if (!ctx) return;
    if (ctx->step_graph_exec) {
        cudaGraphExecDestroy(ctx->step_graph_exec);
        ctx->step_graph_exec = NULL;
    }
    if (ctx->step_graph) {
        cudaGraphDestroy(ctx->step_graph);
        ctx->step_graph = NULL;
    }
    ctx->step_graph_ok = 0;
    ctx->step_graph_seq = 0;
    ctx->step_graph_first = 0;
    ctx->step_graph_n = 0;
    ctx->d_graph_aux_base = NULL;
}

/* P2b whole-step graph: only the static weight-resident range is
 * graph-eligible. Layers above that boundary keep the eager sliding-window
 * loop; they are appended after the graph replay so one layers_forward call
 * can still span the full range. */
static int all_blocks_resident(h3_dit_w4a8 *ctx, unsigned first_block,
                               unsigned n_blocks) {
    unsigned b;
    if (ctx->nvfp4_on)
        return 1; /* FP4 weights own the blocks; INT8 residency not required */
    for (b = 0; b < n_blocks; b++) {
        unsigned block = first_block + b;
        if (!h3_w4a8_cache_block_is_resident(ctx->cache, block))
            return 0;
    }
    return 1;
}

/* Whole-step graph diagnostics (same Tu latch pattern as
 * h3_w4a8_cache_last_plan_reuse). mode: 0 off, 1 eager fallback, 2 replay. */
static int s_step_graph_mode;
static char s_step_graph_reason[64];
static void step_graph_note(int mode, const char *reason) {
    s_step_graph_mode = mode;
    snprintf(s_step_graph_reason, sizeof(s_step_graph_reason), "%s",
             reason ? reason : "");
}

void h3_dit_w4a8_cuda_graph_last(int *mode, const char **reason) {
    if (mode) *mode = s_step_graph_mode;
    if (reason) *reason = s_step_graph_reason;
}

static const char *step_graph_block_reason(h3_dit_w4a8 *ctx,
                                           unsigned n_blocks) {
    if (ctx->uly_u > 1) return "ulysses";
    if (ctx->token_reduction) return "token_reduction";
    if (ctx->lora) return "lora";
    if (h3_cuda_sdpa_backend_name() &&
        !strcmp(h3_cuda_sdpa_backend_name(), "cudnn"))
        return "cudnn";
    {
        unsigned win = h3_w4a8_cache_window_size(ctx->cache);
        if (win > 0 && win < n_blocks) return "weight_window";
    }
    return NULL;
}

static int run_all_blocks_device(
    h3_dit_w4a8 *ctx, unsigned first_block, unsigned n_blocks, float *d_x,
    float *d_y, float *d_aux_base, size_t aux_stride, size_t mod_elems,
    uint32_t *d_map, float *d_cos, float *d_sin, float *d_a, float *d_b,
    float *d_qkv, float *d_wide, size_t seq, char *error, size_t error_size) {
    const size_t hidden = H3_DIT_W4A8_HIDDEN;
    const size_t head_dim = H3_DIT_W4A8_HEAD_DIM;
    unsigned b;
    char detail[384];
    for (b = 0; b < n_blocks; b++) {
        unsigned block = first_block + b;
        const h3_cuda_w4a8_weight *w_qkv = NULL, *w_out = NULL, *w_fc1 = NULL,
                                  *w_fc2 = NULL;
        float *layer = d_aux_base + (size_t)b * aux_stride;
        float *d_mod = layer;
        float *d_n1 = layer + mod_elems;
        float *d_n2 = d_n1 + hidden;
        float *d_qn = d_n2 + hidden;
        float *d_kn = d_qn + head_dim;
        if (!ctx->nvfp4_on) {
            w_qkv = get_weight(ctx, block, H3_W4A8_LINEAR_QKV, error,
                               error_size);
            w_out = w_qkv ? get_weight(ctx, block, H3_W4A8_LINEAR_OUT, error,
                                       error_size)
                          : NULL;
            w_fc1 = w_out ? get_weight(ctx, block, H3_W4A8_LINEAR_FC1, error,
                                       error_size)
                          : NULL;
            w_fc2 = w_fc1 ? get_weight(ctx, block, H3_W4A8_LINEAR_FC2, error,
                                       error_size)
                          : NULL;
            if (!w_fc2) return 0;
        }
        if (!block_device_compute(ctx, block, w_qkv, w_out, w_fc1, w_fc2, d_x,
                                  d_y, d_mod, d_map, d_cos, d_sin, d_n1, d_n2,
                                  d_qn, d_kn, d_a, d_b, d_qkv, d_wide, seq,
                                  NULL, detail, sizeof(detail))) {
            snprintf(error, error_size, "%s", detail);
            return 0;
        }
        {
            float *tmp = d_x;
            d_x = d_y;
            d_y = tmp;
        }
    }
    return 1;
}

/* GPU-resident block: one H2D of inputs + one D2H of y; no per-linear PCIe. */
static int block_forward_gpu_resident(
    h3_dit_w4a8 *ctx, unsigned block, const h3_dit_w4a8_block_aux *aux,
    const float *x, size_t seq, const float *modulation,
    const uint32_t *row_map, const float *rope_cos, const float *rope_sin,
    float *y, h3_dit_w4a8_block_stats *stats, char *error, size_t error_size) {
    const size_t hidden = H3_DIT_W4A8_HIDDEN;
    const size_t inner = H3_DIT_W4A8_INNER;
    const size_t ffn = H3_DIT_W4A8_FFN;
    const size_t slots = H3_DIT_W4A8_SLOTS;
    const size_t mod_rows = dit_adaln_mod_rows(ctx);
    const size_t rope_half = (size_t)H3_DIT_W4A8_ROPE_HALF;
    const size_t mod_elems = mod_rows * slots * hidden;
    const size_t head_dim = H3_DIT_W4A8_HEAD_DIM;
    const size_t wide = ffn * 2; /* == inner*4 */

    float *d_x, *d_y, *d_mod, *d_cos, *d_sin, *d_n1, *d_n2, *d_qn, *d_kn;
    float *d_a, *d_b, *d_qkv, *d_wide;
    uint32_t *d_map;
    const h3_cuda_w4a8_weight *w_qkv, *w_out, *w_fc1, *w_fc2;
    char *base;
    size_t off = 0;
    size_t need;
    char detail[384];

    if (wide != inner * 4) {
        fail(error, error_size, "unexpected ffn/inner geometry");
        return 0;
    }

    w_qkv = w_out = w_fc1 = w_fc2 = NULL;
    if (!ctx->nvfp4_on) {
        w_qkv = get_weight(ctx, block, H3_W4A8_LINEAR_QKV, error, error_size);
        if (!w_qkv) return 0;
        w_out = get_weight(ctx, block, H3_W4A8_LINEAR_OUT, error, error_size);
        if (!w_out) return 0;
        w_fc1 = get_weight(ctx, block, H3_W4A8_LINEAR_FC1, error, error_size);
        if (!w_fc1) return 0;
        w_fc2 = get_weight(ctx, block, H3_W4A8_LINEAR_FC2, error, error_size);
        if (!w_fc2) return 0;
    }

    need = 0;
    need += seq * hidden * sizeof(float);                 /* x */
    need += seq * hidden * sizeof(float);                 /* y */
    need += mod_elems * sizeof(float);                    /* mod */
    need += H3_WS_MAP_BYTES(seq);  /* map */
    need += seq * rope_half * sizeof(float) * 2;          /* cos/sin */
    need += (hidden * 2 + head_dim * 2) * sizeof(float);  /* norms */
    need += seq * hidden * sizeof(float);                 /* a */
    need += seq * hidden * sizeof(float);                 /* b */
    need += seq * inner * 3 * sizeof(float);              /* qkv / act */
    need += seq * wide * sizeof(float);                   /* qkva / fc1 */
    if (!ensure_block_ws(ctx, need, error, error_size)) return 0;

    base = (char *)ctx->d_block_ws;
#define H3_WS_F32(n)                                                           \
    (off += (n) * sizeof(float), (float *)(base + (off - (n) * sizeof(float))))
    d_x = H3_WS_F32(seq * hidden);
    d_y = H3_WS_F32(seq * hidden);
    d_mod = H3_WS_F32(mod_elems);
    {
        size_t map_bytes = H3_WS_MAP_BYTES(seq);
        d_map = (uint32_t *)(base + off);
        off += map_bytes;
    }
    d_cos = H3_WS_F32(seq * rope_half);
    d_sin = H3_WS_F32(seq * rope_half);
    d_n1 = H3_WS_F32(hidden);
    d_n2 = H3_WS_F32(hidden);
    d_qn = H3_WS_F32(head_dim);
    d_kn = H3_WS_F32(head_dim);
    d_a = H3_WS_F32(seq * hidden);
    d_b = H3_WS_F32(seq * hidden);
    d_qkv = H3_WS_F32(seq * inner * 3);
    d_wide = H3_WS_F32(seq * wide);
#undef H3_WS_F32
    if (off > need) {
        fail(error, error_size, "block workspace packing overflow");
        return 0;
    }

    if (cudaMemcpy(d_x, x, seq * hidden * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_mod, modulation, mod_elems * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_cos, rope_cos, seq * rope_half * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_sin, rope_sin, seq * rope_half * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_n1, aux->norm1, hidden * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_n2, aux->norm2, hidden * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_qn, aux->q_norm, head_dim * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_kn, aux->k_norm, head_dim * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        fail(error, error_size, "block resident H2D failed");
        return 0;
    }
    if (row_map) {
        if (cudaMemcpy(d_map, row_map, seq * sizeof(uint32_t),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            fail(error, error_size, "block resident map H2D failed");
            return 0;
        }
    } else {
        d_map = NULL;
    }

    if (env_flag_is_one("H3_DIT_W4A8_FORCE_SDPA_F32")) {
        fail(error, error_size,
             "resident path requires FP16 SDPA (unset FORCE_SDPA_F32)");
        return 0;
    }

    {
        int want_graph = env_flag_is_one("H3_DIT_W4A8_CUDA_GRAPH");
        /* Default quiet: skip per-op EventSynchronize (huge host gap).
         * H3_DIT_W4A8_VERBOSE_TIMING=1 restores per-op ms breakdown. */
        const int want_quiet = !env_flag_is_one("H3_DIT_W4A8_VERBOSE_TIMING");
        int use_graph = 0;
        int quiet_set = 0;
        cudaEvent_t wall0 = NULL, wall1 = NULL;

        /* cuDNN fused SDPA is often uncapturable; keep quiet path instead. */
        if (want_graph) {
            const char *be = h3_cuda_sdpa_backend_name();
            if (be && !strcmp(be, "cudnn")) {
                static int logged_eager;
                want_graph = 0;
                if (!logged_eager) {
                    fprintf(stderr,
                            "dit_graph: eager reason=cudnn_uncapturable\n");
                    logged_eager = 1;
                }
            }
        } else {
            static int logged_off;
            if (!logged_off) {
                fprintf(stderr, "dit_graph: eager reason=off\n");
                logged_off = 1;
            }
        }
        if (want_graph) {
            /* Warm scratch/hadamard/lt workspace once outside capture. */
            if (!ctx->graph_warmed) {
                h3_cuda_set_graph_capture(ctx->cuda, 1);
                quiet_set = 1;
                if (!block_device_compute(
                        ctx, block, w_qkv, w_out, w_fc1, w_fc2, d_x, d_y, d_mod, d_map,
                        d_cos, d_sin, d_n1, d_n2, d_qn, d_kn, d_a, d_b, d_qkv,
                        d_wide, seq, NULL, detail, sizeof(detail))) {
                    /* Graph optional — fall back to eager without aborting. */
                    detail[0] = '\0';
                } else if (cudaMemcpy(d_x, x, seq * hidden * sizeof(float),
                                      cudaMemcpyHostToDevice) != cudaSuccess) {
                    h3_cuda_set_graph_capture(ctx->cuda, 0);
                    fail(error, error_size, "graph warm re-H2D failed");
                    return 0;
                }
                h3_cuda_set_graph_capture(ctx->cuda, 0);
                quiet_set = 0;
                ctx->graph_warmed = 1;
                (void)cudaGetLastError(); /* clear sticky */
            }

            if (!(ctx->graph_ok && ctx->graph_exec && ctx->graph_seq == seq &&
                  ctx->graph_block == block)) {
                cudaError_t gst;
                cudaGraph_t g = NULL;
                destroy_block_graph(ctx);
                (void)cudaGetLastError();
                /* Relaxed mode: allow some legacy sync; fixed 64MiB lt ws. */
                h3_cuda_set_graph_capture(ctx->cuda, 1);
                quiet_set = 1;
                gst = cudaStreamBeginCapture(0, cudaStreamCaptureModeRelaxed);
                if (gst == cudaSuccess) {
                    int cok = block_device_compute(
                        ctx, block, w_qkv, w_out, w_fc1, w_fc2, d_x, d_y, d_mod, d_map,
                        d_cos, d_sin, d_n1, d_n2, d_qn, d_kn, d_a, d_b, d_qkv,
                        d_wide, seq, NULL, detail, sizeof(detail));
                    gst = cudaStreamEndCapture(0, &g);
                    h3_cuda_set_graph_capture(ctx->cuda, 0);
                    quiet_set = 0;
                    if (cok && gst == cudaSuccess && g) {
                        cudaGraphExec_t ge = NULL;
                        if (cudaGraphInstantiate(&ge, g, 0) == cudaSuccess) {
                            ctx->graph = g;
                            ctx->graph_exec = ge;
                            ctx->graph_seq = seq;
                            ctx->graph_block = block;
                            ctx->graph_ok = 1;
                            use_graph = 1;
                            goto graph_done; /* capture already computed */
                        }
                        cudaGraphDestroy(g);
                    } else if (g) {
                        cudaGraphDestroy(g);
                    }
                    /* Capture failed: restore inputs and eager-run below. */
                    (void)cudaGetLastError();
                    if (cudaMemcpy(d_x, x, seq * hidden * sizeof(float),
                                   cudaMemcpyHostToDevice) != cudaSuccess) {
                        fail(error, error_size, "graph fallback H2D failed");
                        return 0;
                    }
                } else {
                    h3_cuda_set_graph_capture(ctx->cuda, 0);
                    quiet_set = 0;
                    (void)cudaGetLastError();
                }
            } else {
                use_graph = 1;
            }
        }

        if (use_graph && ctx->graph_exec) {
            static int logged_hit;
            if (cudaGraphLaunch(ctx->graph_exec, 0) != cudaSuccess) {
                fail(error, error_size, "cudaGraphLaunch failed");
                return 0;
            }
            if (!logged_hit) {
                fprintf(stderr, "dit_graph: graph=hit seq=%zu block=%u\n", seq,
                        block);
                logged_hit = 1;
            }
            if (stats) {
                snprintf(stats->sdpa_backend, sizeof(stats->sdpa_backend), "%s",
                         h3_cuda_sdpa_last_backend_name());
            }
        } else {
            if (want_quiet) {
                h3_cuda_set_quiet(ctx->cuda, 1);
                quiet_set = 1;
                if (stats && cudaEventCreate(&wall0) == cudaSuccess &&
                    cudaEventCreate(&wall1) == cudaSuccess)
                    cudaEventRecord(wall0, 0);
            }
            if (!block_device_compute(
                    ctx, block, w_qkv, w_out, w_fc1, w_fc2, d_x, d_y, d_mod, d_map,
                    d_cos, d_sin, d_n1, d_n2, d_qn, d_kn, d_a, d_b, d_qkv,
                    d_wide, seq, want_quiet ? NULL : stats, detail,
                    sizeof(detail))) {
                if (quiet_set) h3_cuda_set_quiet(ctx->cuda, 0);
                if (wall0) cudaEventDestroy(wall0);
                if (wall1) cudaEventDestroy(wall1);
                snprintf(error, error_size, "%s", detail);
                return 0;
            }
            if (want_quiet) {
                if (wall0 && wall1) {
                    float wall_ms = 0.0f;
                    cudaEventRecord(wall1, 0);
                    cudaEventSynchronize(wall1);
                    cudaEventElapsedTime(&wall_ms, wall0, wall1);
                    if (stats) {
                        /* Quiet whole-block GPU wall — NOT true SDPA share. */
                        stats->block_gpu_ms = (double)wall_ms;
                        stats->sdpa_ms = 0.0;
                        snprintf(stats->sdpa_backend,
                                 sizeof(stats->sdpa_backend), "%s",
                                 h3_cuda_sdpa_last_backend_name());
                    }
                    cudaEventDestroy(wall0);
                    cudaEventDestroy(wall1);
                }
                h3_cuda_set_quiet(ctx->cuda, 0);
                quiet_set = 0;
            }
        }
        (void)quiet_set;
    }
graph_done:

    if (cudaMemcpy(y, d_y, seq * hidden * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        fail(error, error_size, "block resident D2H failed");
        return 0;
    }
    return 1;
}

static int block_forward_host_path(
    h3_dit_w4a8 *ctx, unsigned block, const h3_dit_w4a8_block_aux *aux,
    const float *x, size_t seq, const float *modulation,
    const uint32_t *row_map, const float *rope_cos, const float *rope_sin,
    float *y, h3_dit_w4a8_block_stats *stats, char *error, size_t error_size) {
    const size_t hidden = H3_DIT_W4A8_HIDDEN;
    const size_t heads = H3_DIT_W4A8_HEADS;
    const size_t head_dim = H3_DIT_W4A8_HEAD_DIM;
    const size_t inner = H3_DIT_W4A8_INNER;
    const size_t ffn = H3_DIT_W4A8_FFN;
    const size_t slots = H3_DIT_W4A8_SLOTS;
    const float eps = 1e-5f;
    const float attn_scale = 1.0f / sqrtf((float)head_dim);

    float *mod_attn = NULL, *qkv = NULL, *query = NULL, *key = NULL, *value = NULL;
    float *attn = NULL, *attn_out = NULL, *after_attn = NULL;
    float *mod_mlp = NULL, *fc1 = NULL, *act = NULL, *mlp_out = NULL;
    h3_dit_w4a8_linear_stats lst;
    clock_t t0, t1;
    int ok = 0;
    const int force_cpu_ops = env_flag_is_one("H3_DIT_W4A8_FORCE_CPU_OPS");
    const size_t mod_rows = dit_adaln_mod_rows(ctx);
    double gpu_ms = 0.0;

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
        fail(error, error_size, "out of memory (block_forward)");
        goto done;
    }

    t0 = clock();
    if (!force_cpu_ops && ctx->cuda) {
        if (!h3_cuda_adaln_f32(ctx->cuda, x, aux->norm1, modulation, row_map,
                               seq, hidden, slots, mod_rows, 0, 1, eps, mod_attn,
                               &gpu_ms, error, error_size))
            goto done;
        if (stats)
            stats->adaln_attn_ms =
                (gpu_ms > 0.0) ? gpu_ms : elapsed_ms(t0, clock());
    } else {
        h3_dit_w4a8_adaln_f32(x, aux->norm1, modulation, row_map, seq, hidden,
                              slots, 0, 1, eps, mod_attn);
        t1 = clock();
        if (stats) stats->adaln_attn_ms = elapsed_ms(t0, t1);
    }

    memset(&lst, 0, sizeof(lst));
    if (!h3_dit_w4a8_linear_f32(ctx, block, H3_W4A8_LINEAR_QKV, mod_attn, seq,
                                qkv, &lst, error, error_size))
        goto done;
    if (stats) {
        stats->qkv_ms = lst.gemm_ms;
        stats->act_prep_ms += lst.convrot_ms + lst.quant_ms;
        snprintf(stats->qkv_backend, sizeof(stats->qkv_backend), "%s",
                 lst.backend);
    }

    t0 = clock();
    gpu_ms = 0.0;
    if (!force_cpu_ops && ctx->cuda) {
        if (!h3_cuda_qkv_rope_f32(ctx->cuda, qkv, aux->q_norm, aux->k_norm,
                                  rope_cos, rope_sin, seq, heads, head_dim,
                                  H3_DIT_W4A8_ROPE_HALF, eps, query, key, value,
                                  &gpu_ms, error, error_size))
            goto done;
        if (stats)
            stats->rope_ms = (gpu_ms > 0.0) ? gpu_ms : elapsed_ms(t0, clock());
    } else {
        if (!h3_dit_w4a8_qkv_rope_f32(qkv, aux->q_norm, aux->k_norm, rope_cos,
                                      rope_sin, seq, heads, head_dim,
                                      H3_DIT_W4A8_ROPE_HALF, eps, query, key,
                                      value, error, error_size))
            goto done;
        t1 = clock();
        if (stats) stats->rope_ms = elapsed_ms(t0, t1);
    }

    t0 = clock();
    gpu_ms = 0.0;
    if (!env_flag_is_one("H3_DIT_W4A8_FORCE_CPU_SDPA")) {
        int use_f32 = env_flag_is_one("H3_DIT_W4A8_FORCE_SDPA_F32");
        char be[32] = "";
        int sdpa_ok =
            use_f32 ? h3_cuda_sdpa_f32(ctx->cuda, query, key, value, seq, heads,
                                       head_dim, attn_scale, attn, &gpu_ms,
                                       error, error_size)
                    : h3_cuda_sdpa_f16_ex(ctx->cuda, query, key, value, seq,
                                          heads, head_dim, attn_scale, 0, attn,
                                          &gpu_ms, be, sizeof(be), error,
                                          error_size);
        if (!sdpa_ok) goto done;
        t1 = clock();
        if (stats) {
            stats->sdpa_ms = (gpu_ms > 0.0) ? gpu_ms : elapsed_ms(t0, t1);
            snprintf(stats->sdpa_backend, sizeof(stats->sdpa_backend), "%s",
                     use_f32 ? "cuda_flash_f32"
                             : (be[0] ? be : "cuda_flash_f16"));
        }
    } else {
        if (!h3_dit_w4a8_sdpa_f32(query, key, value, seq, heads, head_dim,
                                  attn_scale, attn, error, error_size))
            goto done;
        t1 = clock();
        if (stats) {
            stats->sdpa_ms = elapsed_ms(t0, t1);
            snprintf(stats->sdpa_backend, sizeof(stats->sdpa_backend),
                     "cpu_naive");
        }
    }

    memset(&lst, 0, sizeof(lst));
    if (!h3_dit_w4a8_linear_f32(ctx, block, H3_W4A8_LINEAR_OUT, attn, seq,
                                attn_out, &lst, error, error_size))
        goto done;
    if (stats) {
        stats->out_ms = lst.gemm_ms;
        stats->act_prep_ms += lst.convrot_ms + lst.quant_ms;
        snprintf(stats->out_backend, sizeof(stats->out_backend), "%s",
                 lst.backend);
    }

    t0 = clock();
    gpu_ms = 0.0;
    if (!force_cpu_ops && ctx->cuda) {
        if (!h3_cuda_gate_f32(ctx->cuda, x, attn_out, modulation, row_map, seq,
                              hidden, slots, mod_rows, 2, after_attn, &gpu_ms,
                              error, error_size))
            goto done;
        if (stats)
            stats->gate_attn_ms =
                (gpu_ms > 0.0) ? gpu_ms : elapsed_ms(t0, clock());
    } else {
        h3_dit_w4a8_gate_f32(x, attn_out, modulation, row_map, seq, hidden,
                             slots, 2, after_attn);
        t1 = clock();
        if (stats) stats->gate_attn_ms = elapsed_ms(t0, t1);
    }

    t0 = clock();
    gpu_ms = 0.0;
    if (!force_cpu_ops && ctx->cuda) {
        if (!h3_cuda_adaln_f32(ctx->cuda, after_attn, aux->norm2, modulation,
                               row_map, seq, hidden, slots, mod_rows, 3, 4, eps,
                               mod_mlp, &gpu_ms, error, error_size))
            goto done;
        if (stats)
            stats->adaln_mlp_ms =
                (gpu_ms > 0.0) ? gpu_ms : elapsed_ms(t0, clock());
    } else {
        h3_dit_w4a8_adaln_f32(after_attn, aux->norm2, modulation, row_map, seq,
                              hidden, slots, 3, 4, eps, mod_mlp);
        t1 = clock();
        if (stats) stats->adaln_mlp_ms = elapsed_ms(t0, t1);
    }

    memset(&lst, 0, sizeof(lst));
    if (!h3_dit_w4a8_linear_f32(ctx, block, H3_W4A8_LINEAR_FC1, mod_mlp, seq,
                                fc1, &lst, error, error_size))
        goto done;
    if (stats) {
        stats->fc1_ms = lst.gemm_ms;
        stats->act_prep_ms += lst.convrot_ms + lst.quant_ms;
        snprintf(stats->fc1_backend, sizeof(stats->fc1_backend), "%s",
                 lst.backend);
    }

    t0 = clock();
    gpu_ms = 0.0;
    if (!force_cpu_ops && ctx->cuda) {
        if (!h3_cuda_swiglu_f32(ctx->cuda, fc1, seq, ffn, act, &gpu_ms, error,
                                error_size))
            goto done;
        if (stats)
            stats->swiglu_ms = (gpu_ms > 0.0) ? gpu_ms : elapsed_ms(t0, clock());
    } else {
        h3_dit_w4a8_swiglu_f32(fc1, seq, ffn, act);
        t1 = clock();
        if (stats) stats->swiglu_ms = elapsed_ms(t0, t1);
    }

    memset(&lst, 0, sizeof(lst));
    if (!h3_dit_w4a8_linear_f32(ctx, block, H3_W4A8_LINEAR_FC2, act, seq,
                                mlp_out, &lst, error, error_size))
        goto done;
    if (stats) {
        stats->fc2_ms = lst.gemm_ms;
        stats->act_prep_ms += lst.convrot_ms + lst.quant_ms;
        snprintf(stats->fc2_backend, sizeof(stats->fc2_backend), "%s",
                 lst.backend);
    }

    t0 = clock();
    gpu_ms = 0.0;
    if (!force_cpu_ops && ctx->cuda) {
        if (!h3_cuda_gate_f32(ctx->cuda, after_attn, mlp_out, modulation,
                              row_map, seq, hidden, slots, mod_rows, 5, y,
                              &gpu_ms, error, error_size))
            goto done;
        if (stats)
            stats->gate_mlp_ms =
                (gpu_ms > 0.0) ? gpu_ms : elapsed_ms(t0, clock());
    } else {
        h3_dit_w4a8_gate_f32(after_attn, mlp_out, modulation, row_map, seq,
                             hidden, slots, 5, y);
        t1 = clock();
        if (stats) stats->gate_mlp_ms = elapsed_ms(t0, t1);
    }

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

int h3_dit_w4a8_block_forward_f32(
    h3_dit_w4a8 *ctx, unsigned block, const h3_dit_w4a8_block_aux *aux,
    const float *x, size_t seq, const float *modulation,
    const uint32_t *row_map, const float *rope_cos, const float *rope_sin,
    float *y, h3_dit_w4a8_block_stats *stats, char *error, size_t error_size) {
    if (stats) memset(stats, 0, sizeof(*stats));
    if (!ctx || !aux || !x || !modulation || !rope_cos || !rope_sin || !y ||
        seq == 0) {
        fail(error, error_size, "invalid block_forward args");
        return 0;
    }
    if (!aux->norm1 || !aux->norm2 || !aux->q_norm || !aux->k_norm) {
        fail(error, error_size, "block aux missing norms");
        return 0;
    }

    /* Default: keep activations resident on GPU (kill per-linear PCIe). */
    if (ctx->cuda && !env_flag_is_one("H3_DIT_W4A8_FORCE_HOST_BLOCK") &&
        !env_flag_is_one("H3_DIT_W4A8_FORCE_CPU_SDPA") &&
        !env_flag_is_one("H3_DIT_W4A8_FORCE_CPU_OPS") &&
        !env_flag_is_one("H3_DIT_W4A8_FORCE_CPU_ACT") &&
        !env_flag_is_one("H3_DIT_W4A8_FORCE_SDPA_F32") &&
        ctx->act_quant == H3_W4A8_ACT_QUANT_PER_ROW) {
        return block_forward_gpu_resident(ctx, block, aux, x, seq, modulation,
                                          row_map, rope_cos, rope_sin, y, stats,
                                          error, error_size);
    }
    return block_forward_host_path(ctx, block, aux, x, seq, modulation, row_map,
                                   rope_cos, rope_sin, y, stats, error,
                                   error_size);
}

static void refiner_block_free(h3_dit_w4a8_refiner_block *b) {
    if (!b) return;
    free(b->norm1);
    free(b->norm2);
    free(b->q_norm);
    free(b->k_norm);
    free(b->qkv_w);
    free(b->out_w);
    free(b->fc1_w);
    free(b->fc2_w);
    memset(b, 0, sizeof(*b));
}

void h3_dit_w4a8_core_free(h3_dit_w4a8_core *core) {
    if (!core) return;
    free(core->video_patch_w);
    free(core->video_patch_b);
    free(core->audio_patch_w);
    free(core->audio_patch_b);
    free(core->final_norm);
    free(core->final_video_w);
    free(core->final_video_b);
    free(core->final_audio_w);
    free(core->final_audio_b);
    free(core->final_adaln_w);
    free(core->final_adaln_b);
    free(core->t_table);
    free(core->inv_freq);
    free(core->condition_w);
    free(core->condition_b);
    refiner_block_free(&core->refiner[0]);
    refiner_block_free(&core->refiner[1]);
    free(core->refiner_final_norm);
    memset(core, 0, sizeof(*core));
}

int h3_dit_w4a8_load_core_hdr(const h3_st_header *hdr, h3_dit_w4a8_core *core,
                              char *error, size_t error_size) {
    const h3_st_tensor *tt = NULL, *fw = NULL, *fb = NULL;
    size_t n = 0;
    h3_dtype dtype = H3_DTYPE_UNKNOWN;
    const size_t final_adaln_out =
        (size_t)H3_DIT_W4A8_FINAL_SLOTS * H3_DIT_W4A8_HIDDEN;

    if (!hdr || !core) {
        fail(error, error_size, "invalid load_core args");
        return 0;
    }
    memset(core, 0, sizeof(*core));

    if (!load_named_f32_hdr(hdr, "video_patch_proj.weight",
                            (size_t)H3_DIT_W4A8_HIDDEN * H3_DIT_W4A8_VIDEO_PATCH,
                            &core->video_patch_w, error, error_size) ||
        !load_named_f32_hdr(hdr, "video_patch_proj.bias", H3_DIT_W4A8_HIDDEN,
                            &core->video_patch_b, error, error_size) ||
        !load_named_f32_hdr(hdr, "final_layer.norm.weight", H3_DIT_W4A8_HIDDEN,
                            &core->final_norm, error, error_size) ||
        !load_named_f32_hdr(hdr, "final_layer.video_out.weight",
                            (size_t)H3_DIT_W4A8_VIDEO_PATCH * H3_DIT_W4A8_HIDDEN,
                            &core->final_video_w, error, error_size) ||
        !load_named_f32_hdr(hdr, "final_layer.video_out.bias",
                            H3_DIT_W4A8_VIDEO_PATCH, &core->final_video_b, error,
                            error_size) ||
        !load_named_f32_hdr(hdr, "rope.inv_freq", H3_DIT_W4A8_ROPE_FREQS,
                            &core->inv_freq, error, error_size))
        goto fail;

    fw = h3_st_find(hdr, "final_layer.adaln_proj.linear.weight");
    fb = h3_st_find(hdr, "final_layer.adaln_proj.linear.bias");
    if (!fw || !fb || fw->ndim != 2 || fb->ndim != 1 ||
        (size_t)fw->shape[0] != final_adaln_out ||
        (size_t)fw->shape[1] != H3_DIT_W4A8_TIME_FEAT ||
        (size_t)fb->shape[0] != final_adaln_out) {
        fail(error, error_size, "final_layer.adaln_proj unexpected shape");
        goto fail;
    }
    core->final_adaln_w = h3_st_load_f32(
        hdr, "final_layer.adaln_proj.linear.weight", &n, &dtype, error,
        error_size);
    if (!core->final_adaln_w || n != final_adaln_out * H3_DIT_W4A8_TIME_FEAT)
        goto fail;
    core->final_adaln_b = h3_st_load_f32(
        hdr, "final_layer.adaln_proj.linear.bias", &n, &dtype, error,
        error_size);
    if (!core->final_adaln_b || n != final_adaln_out) goto fail;

    tt = h3_st_find(hdr, "adaln_t_table");
    if (!tt || tt->ndim != 2 ||
        (size_t)tt->shape[1] != H3_DIT_W4A8_TIME_FEAT) {
        fail(error, error_size, "adaln_t_table missing/unexpected");
        goto fail;
    }
    core->t_table_rows = (size_t)tt->shape[0];
    core->t_table =
        h3_st_load_f32(hdr, "adaln_t_table", &n, &dtype, error, error_size);
    if (!core->t_table || n != core->t_table_rows * H3_DIT_W4A8_TIME_FEAT)
        goto fail;

    if (h3_st_find(hdr, "condition_proj.weight") &&
        h3_st_find(hdr, "condition_proj.bias")) {
        if (!load_named_f32_hdr(
                hdr, "condition_proj.weight",
                (size_t)H3_DIT_W4A8_HIDDEN * H3_DIT_W4A8_TEXT_DIM,
                &core->condition_w, error, error_size) ||
            !load_named_f32_hdr(hdr, "condition_proj.bias", H3_DIT_W4A8_HIDDEN,
                                &core->condition_b, error, error_size))
            goto fail;
    }

    if (h3_st_find(hdr, "token_refiner.blocks.0.attn.qkv_proj.weight") &&
        h3_st_find(hdr, "token_refiner.final_norm.weight")) {
        unsigned bi;
        for (bi = 0; bi < 2; bi++) {
            h3_dit_w4a8_refiner_block *rb = &core->refiner[bi];
            char prefix[96];
            char name[160];
            snprintf(prefix, sizeof(prefix), "token_refiner.blocks.%u.", bi);
#define H3_REF_LOAD(field, suffix, expect)                                     \
    do {                                                                       \
        snprintf(name, sizeof(name), "%s%s", prefix, suffix);                  \
        if (!load_named_f32_hdr(hdr, name, (expect), &rb->field, error,        \
                                error_size))                                   \
            goto fail;                                                         \
    } while (0)
            H3_REF_LOAD(norm1, "norm1.weight", H3_DIT_W4A8_HIDDEN);
            H3_REF_LOAD(norm2, "norm2.weight", H3_DIT_W4A8_HIDDEN);
            H3_REF_LOAD(q_norm, "attn.q_norm.weight", H3_DIT_W4A8_HEAD_DIM);
            H3_REF_LOAD(k_norm, "attn.k_norm.weight", H3_DIT_W4A8_HEAD_DIM);
            H3_REF_LOAD(qkv_w, "attn.qkv_proj.weight",
                        (size_t)H3_DIT_W4A8_INNER * 3 * H3_DIT_W4A8_HIDDEN);
            H3_REF_LOAD(out_w, "attn.out_proj.weight",
                        (size_t)H3_DIT_W4A8_HIDDEN * H3_DIT_W4A8_INNER);
            H3_REF_LOAD(fc1_w, "mlp.fc1.weight",
                        (size_t)H3_DIT_W4A8_FFN * 2 * H3_DIT_W4A8_HIDDEN);
            H3_REF_LOAD(fc2_w, "mlp.fc2.weight",
                        (size_t)H3_DIT_W4A8_HIDDEN * H3_DIT_W4A8_FFN);
#undef H3_REF_LOAD
        }
        if (!load_named_f32_hdr(hdr, "token_refiner.final_norm.weight",
                                H3_DIT_W4A8_HIDDEN, &core->refiner_final_norm,
                                error, error_size))
            goto fail;
        core->has_refiner = 1;
    }

    if (h3_st_find(hdr, "audio_patch_proj.weight") &&
        h3_st_find(hdr, "audio_patch_proj.bias")) {
        if (!load_named_f32_hdr(
                hdr, "audio_patch_proj.weight",
                (size_t)H3_DIT_W4A8_HIDDEN * H3_DIT_W4A8_AUDIO_CH,
                &core->audio_patch_w, error, error_size) ||
            !load_named_f32_hdr(hdr, "audio_patch_proj.bias", H3_DIT_W4A8_HIDDEN,
                                &core->audio_patch_b, error, error_size))
            goto fail;
    }
    if (h3_st_find(hdr, "final_layer.audio_out.weight") &&
        h3_st_find(hdr, "final_layer.audio_out.bias")) {
        if (!load_named_f32_hdr(
                hdr, "final_layer.audio_out.weight",
                (size_t)H3_DIT_W4A8_AUDIO_CH * H3_DIT_W4A8_HIDDEN,
                &core->final_audio_w, error, error_size) ||
            !load_named_f32_hdr(hdr, "final_layer.audio_out.bias",
                                H3_DIT_W4A8_AUDIO_CH, &core->final_audio_b,
                                error, error_size))
            goto fail;
    }
    return 1;

fail:
    h3_dit_w4a8_core_free(core);
    return 0;
}

int h3_dit_w4a8_load_core(const h3_w4a8_store *store, h3_dit_w4a8_core *core,
                          char *error, size_t error_size) {
    if (!store) {
        fail(error, error_size, "invalid load_core args");
        return 0;
    }
    return h3_dit_w4a8_load_core_hdr(h3_w4a8_store_header(store), core, error,
                                     error_size);
}

int h3_dit_w4a8_load_core_nvfp4(const h3_nvfp4_store *store,
                                h3_dit_w4a8_core *core, char *error,
                                size_t error_size) {
    if (!store || !core) {
        fail(error, error_size, "invalid load_core_nvfp4 args");
        return 0;
    }
    return h3_dit_w4a8_load_core_hdr(&store->header, core, error, error_size);
}

int h3_dit_w4a8_load_core_int8(const h3_dit_int8_store *store,
                               h3_dit_w4a8_core *core, char *error,
                               size_t error_size) {
    if (!store) {
        fail(error, error_size, "invalid load_core_int8 args");
        return 0;
    }
    return h3_dit_w4a8_load_core_hdr(h3_dit_int8_store_header(store), core,
                                     error, error_size);
}

size_t h3_dit_w4a8_t_row_from_sigma(float sigma, size_t t_table_rows) {
    double s;
    size_t row;
    float t;
    if (t_table_rows == 0) return 0;
    if (!isfinite(sigma)) sigma = 0.0f;
    if (sigma < 0.0f) sigma = 0.0f;
    if (sigma > 1.0f) sigma = 1.0f;
    /* Metal h3_dit_schedule: time_embedder input is (1 - sigma).
     * Kijai adaln_t_table is baked over that same [0,1] axis, so index by
     * (1-sigma). H3_DIT_T_ROW_BY_SIGMA=1 restores the old inverted mapping. */
    if (env_flag_is_one("H3_DIT_T_ROW_BY_SIGMA"))
        t = sigma;
    else
        t = 1.0f - sigma;
    s = (double)t * (double)(t_table_rows - 1);
    row = (size_t)llround(s);
    if (row >= t_table_rows) row = t_table_rows - 1;
    return row;
}

int h3_dit_w4a8_visual_cond_adaln_active(size_t cond_seq, float sigma) {
    float time;
    if (cond_seq == 0) return 0;
    if (!isfinite(sigma)) sigma = 0.0f;
    /* Same axis as t_row_from_sigma / Metal schedule: time = 1 - flow_σ
     * (clean end). Metal fallback when time >= 0.999.
     * Also gate σ > 0.001 so the fixture sigma<=0.001 is fallback:
     * (1-0.001f) < 0.999f is true in float and would miss it. */
    time = 1.0f - sigma;
    if (sigma <= 0.001f) return 0;
    return time < 0.999f ? 1 : 0;
}

uint32_t h3_dit_w4a8_visual_cond_mod_row(size_t cond_seq, float sigma) {
    return h3_dit_w4a8_visual_cond_adaln_active(cond_seq, sigma)
               ? 3u
               : (uint32_t)H3_DIT_W4A8_MOD_VIDEO;
}

size_t h3_dit_w4a8_visual_cond_t_row(size_t t_table_rows) {
    /* Metal times[visual_condition] = 0.999 (time axis). t_row_from_sigma
     * indexes by (1-flow_σ), so pass flow_σ = 1-0.999. */
    return h3_dit_w4a8_t_row_from_sigma(1.0f - 0.999f, t_table_rows);
}

int h3_dit_w4a8_audio_cond_adaln_active(size_t audio_cond_seq,
                                        float audio_sigma) {
    float time;
    if (audio_cond_seq == 0) return 0;
    if (!isfinite(audio_sigma)) audio_sigma = 0.0f;
    /* Metal schedule: audio_condition when target audio time < 1.0
     * (1 - audio_sigma < 1). At clean audio (σ<=0) fall back to MOD_AUDIO. */
    time = 1.0f - audio_sigma;
    return time < 1.0f ? 1 : 0;
}

uint32_t h3_dit_w4a8_audio_cond_mod_row(size_t audio_cond_seq, float audio_sigma,
                                        int visual_cond_active) {
    if (!h3_dit_w4a8_audio_cond_adaln_active(audio_cond_seq, audio_sigma))
        return (uint32_t)H3_DIT_W4A8_MOD_AUDIO;
    return visual_cond_active ? 4u : 3u;
}

size_t h3_dit_w4a8_audio_cond_t_row(size_t t_table_rows) {
    /* Metal times[audio_condition] = 1.0 (clean audio end). */
    return h3_dit_w4a8_t_row_from_sigma(0.0f, t_table_rows);
}

int h3_dit_w4a8_dense_f32(const float *x, const float *weight,
                          const float *bias, size_t m, size_t n, size_t k,
                          float *y, char *error, size_t error_size) {
    size_t r, o, i;
    if (!x || !weight || !y || m == 0 || n == 0 || k == 0) {
        fail(error, error_size, "invalid dense_f32 args");
        return 0;
    }
    for (r = 0; r < m; r++) {
        const float *xr = x + r * k;
        float *yr = y + r * n;
        for (o = 0; o < n; o++) {
            const float *wrow = weight + o * k;
            float sum = bias ? bias[o] : 0.0f;
            for (i = 0; i < k; i++) sum += xr[i] * wrow[i];
            yr[o] = sum;
        }
    }
    return 1;
}

static float *ensure_d_weight(float **slot, const float *host, size_t nelem,
                              char *error, size_t error_size) {
    cudaError_t st;
    if (!slot || !host || nelem == 0) {
        fail(error, error_size, "ensure_d_weight bad args");
        return NULL;
    }
    if (*slot) return *slot;
    st = cudaMalloc((void **)slot, nelem * sizeof(float));
    if (st != cudaSuccess) {
        fail(error, error_size, "cudaMalloc dense weight");
        *slot = NULL;
        return NULL;
    }
    st = cudaMemcpy(*slot, host, nelem * sizeof(float), cudaMemcpyHostToDevice);
    if (st != cudaSuccess) {
        cudaFree(*slot);
        *slot = NULL;
        fail(error, error_size, "cudaMemcpy dense weight H2D");
        return NULL;
    }
    return *slot;
}

/* cuBLAS dense y=x@W^T (+ host bias). H3_DIT_W4A8_HOST_DENSE=1 → host. */
static int dense_f32_cuda(h3_cuda *cuda, float **d_w_slot, float **d_b_slot,
                          const float *x, const float *weight, const float *bias,
                          size_t m, size_t n, size_t k, float *y, char *error,
                          size_t error_size) {
    float *d_w;
    size_t r, o;
    if (env_flag_is_one("H3_DIT_W4A8_HOST_DENSE") || !cuda)
        return h3_dit_w4a8_dense_f32(x, weight, bias, m, n, k, y, error,
                                     error_size);
    d_w = ensure_d_weight(d_w_slot, weight, n * k, error, error_size);
    if (!d_w) return 0;
    if (bias && d_b_slot &&
        !ensure_d_weight(d_b_slot, bias, n, error, error_size))
        return 0;
    if (!h3_cuda_gemm_f32_dw(cuda, x, d_w, m, n, k, y, NULL, error, error_size))
        return 0;
    if (bias) {
        for (r = 0; r < m; r++) {
            float *yr = y + r * n;
            for (o = 0; o < n; o++) yr[o] += bias[o];
        }
    }
    return 1;
}

/* Device-out dense: H2D x → GEMM+bias → d_y (no activation D2H). */
static int dense_f32_cuda_to_dev(h3_cuda *cuda, float **d_w_slot,
                                 float **d_b_slot, const float *x,
                                 const float *weight, const float *bias,
                                 size_t m, size_t n, size_t k, float *d_y,
                                 char *error, size_t error_size) {
    float *d_w, *d_b = NULL, *d_x = NULL;
    int ok = 0;
    if (!cuda || !x || !weight || !d_y || m == 0 || n == 0 || k == 0) {
        fail(error, error_size, "invalid dense_to_dev args");
        return 0;
    }
    d_w = ensure_d_weight(d_w_slot, weight, n * k, error, error_size);
    if (!d_w) return 0;
    if (bias) {
        d_b = ensure_d_weight(d_b_slot, bias, n, error, error_size);
        if (!d_b) return 0;
    }
    if (cudaMalloc((void **)&d_x, m * k * sizeof(float)) != cudaSuccess) {
        fail(error, error_size, "dense_to_dev x OOM");
        return 0;
    }
    if (cudaMemcpy(d_x, x, m * k * sizeof(float), cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        fail(error, error_size, "dense_to_dev x H2D");
        goto done;
    }
    if (!h3_cuda_gemm_f32_dd(cuda, d_x, d_w, m, n, k, d_y, NULL, error,
                             error_size))
        goto done;
    if (d_b && !h3_cuda_bias_add_f32_dev(cuda, d_y, d_b, m, n, error,
                                         error_size))
        goto done;
    ok = 1;
done:
    if (d_x) cudaFree(d_x);
    return ok;
}

int h3_dit_w4a8_patch_video_f32(const h3_dit_w4a8_core *core, const float *x,
                                size_t seq, float *y, char *error,
                                size_t error_size) {
    if (!core || !core->video_patch_w || !core->video_patch_b) {
        fail(error, error_size, "core missing video_patch");
        return 0;
    }
    return h3_dit_w4a8_dense_f32(x, core->video_patch_w, core->video_patch_b,
                                 seq, H3_DIT_W4A8_HIDDEN,
                                 H3_DIT_W4A8_VIDEO_PATCH, y, error, error_size);
}

int h3_dit_w4a8_patch_video_cuda(h3_dit_w4a8 *ctx, const h3_dit_w4a8_core *core,
                                 const float *x, size_t seq, float *y,
                                 char *error, size_t error_size) {
    if (!core || !core->video_patch_w || !core->video_patch_b) {
        fail(error, error_size, "core missing video_patch");
        return 0;
    }
    if (!ctx || !ctx->cuda)
        return h3_dit_w4a8_patch_video_f32(core, x, seq, y, error, error_size);
    return dense_f32_cuda(ctx->cuda, &ctx->d_video_patch_w, &ctx->d_video_patch_b,
                          x, core->video_patch_w, core->video_patch_b, seq,
                          H3_DIT_W4A8_HIDDEN, H3_DIT_W4A8_VIDEO_PATCH, y, error,
                          error_size);
}

int h3_dit_w4a8_patch_audio_f32(const h3_dit_w4a8_core *core, const float *x,
                                size_t seq, float *y, char *error,
                                size_t error_size) {
    if (!core || !core->audio_patch_w || !core->audio_patch_b) {
        fail(error, error_size, "core missing audio_patch (Au3 weights?)");
        return 0;
    }
    return h3_dit_w4a8_dense_f32(x, core->audio_patch_w, core->audio_patch_b,
                                 seq, H3_DIT_W4A8_HIDDEN, H3_DIT_W4A8_AUDIO_CH,
                                 y, error, error_size);
}

int h3_dit_w4a8_patch_audio_cuda(h3_dit_w4a8 *ctx, const h3_dit_w4a8_core *core,
                                 const float *x, size_t seq, float *y,
                                 char *error, size_t error_size) {
    if (!core || !core->audio_patch_w || !core->audio_patch_b) {
        fail(error, error_size, "core missing audio_patch (Au3 weights?)");
        return 0;
    }
    if (!ctx || !ctx->cuda)
        return h3_dit_w4a8_patch_audio_f32(core, x, seq, y, error, error_size);
    return dense_f32_cuda(ctx->cuda, &ctx->d_audio_patch_w, &ctx->d_audio_patch_b,
                          x, core->audio_patch_w, core->audio_patch_b, seq,
                          H3_DIT_W4A8_HIDDEN, H3_DIT_W4A8_AUDIO_CH, y, error,
                          error_size);
}

int h3_dit_w4a8_pack_audio_latent(const float *latent_c2t, int audio_t,
                                  float *rows_seq32, char *error,
                                  size_t error_size) {
    int stream, t, ch;
    size_t out = 0;
    if (!latent_c2t || !rows_seq32 || audio_t < 1) {
        fail(error, error_size, "invalid pack_audio_latent");
        return 0;
    }
    for (stream = 0; stream < 2; stream++)
        for (t = 0; t < audio_t; t++)
            for (ch = 0; ch < H3_DIT_W4A8_AUDIO_CH; ch++) {
                size_t in = ((size_t)ch * 2 + (size_t)stream) * (size_t)audio_t +
                            (size_t)t;
                rows_seq32[out++] = latent_c2t[in];
            }
    return 1;
}

int h3_dit_w4a8_unpack_audio_latent(const float *rows_seq32, int audio_t,
                                    float *latent_c2t, char *error,
                                    size_t error_size) {
    int stream, t, ch;
    size_t in = 0;
    if (!rows_seq32 || !latent_c2t || audio_t < 1) {
        fail(error, error_size, "invalid unpack_audio_latent");
        return 0;
    }
    for (stream = 0; stream < 2; stream++)
        for (t = 0; t < audio_t; t++)
            for (ch = 0; ch < H3_DIT_W4A8_AUDIO_CH; ch++) {
                size_t out =
                    ((size_t)ch * 2 + (size_t)stream) * (size_t)audio_t +
                    (size_t)t;
                latent_c2t[out] = rows_seq32[in++];
            }
    return 1;
}

int h3_dit_w4a8_final_adaln_project_f32(const h3_dit_w4a8_core *core,
                                        const float *time_feat,
                                        float *modulation, char *error,
                                        size_t error_size) {
    const size_t out =
        (size_t)H3_DIT_W4A8_FINAL_SLOTS * H3_DIT_W4A8_HIDDEN;
    float act[H3_DIT_W4A8_TIME_FEAT];
    const float *feat;
    size_t o, i;
    if (!core || !core->final_adaln_w || !core->final_adaln_b || !time_feat ||
        !modulation) {
        fail(error, error_size, "invalid final_adaln_project args");
        return 0;
    }
    if (dit_adaln_apply_silu()) {
        adaln_silu_time_feat(time_feat, act, H3_DIT_W4A8_TIME_FEAT);
        feat = act;
    } else {
        feat = time_feat;
    }
    for (o = 0; o < out; o++) {
        const float *row = core->final_adaln_w + o * H3_DIT_W4A8_TIME_FEAT;
        float sum = core->final_adaln_b[o];
        for (i = 0; i < H3_DIT_W4A8_TIME_FEAT; i++)
            sum += row[i] * feat[i];
        modulation[o] = sum;
    }
    return 1;
}

int h3_dit_w4a8_final_video_f32(const h3_dit_w4a8_core *core, const float *x,
                                size_t seq, const float *time_feat, float *y,
                                char *error, size_t error_size) {
    float *modulation = NULL;
    float *normed = NULL;
    int ok = 0;
    if (!core || !core->final_norm || !core->final_video_w ||
        !core->final_video_b || !x || !time_feat || !y || seq == 0) {
        fail(error, error_size, "invalid final_video args");
        return 0;
    }
    modulation = (float *)malloc((size_t)H3_DIT_W4A8_FINAL_SLOTS *
                                 H3_DIT_W4A8_HIDDEN * sizeof(float));
    normed = (float *)malloc(seq * H3_DIT_W4A8_HIDDEN * sizeof(float));
    if (!modulation || !normed) {
        fail(error, error_size, "out of memory (final_video)");
        goto done;
    }
    if (!h3_dit_w4a8_final_adaln_project_f32(core, time_feat, modulation, error,
                                             error_size))
        goto done;
    /* row_map NULL → identity; for video-only use mrow=0 via zero map. */
    {
        uint32_t *zeros = (uint32_t *)calloc(seq, sizeof(uint32_t));
        if (!zeros) {
            fail(error, error_size, "out of memory (final row_map)");
            goto done;
        }
        h3_dit_w4a8_adaln_f32(x, core->final_norm, modulation, zeros, seq,
                              H3_DIT_W4A8_HIDDEN, H3_DIT_W4A8_FINAL_SLOTS, 0, 1,
                              1e-5f, normed);
        free(zeros);
    }
    if (!h3_dit_w4a8_dense_f32(normed, core->final_video_w, core->final_video_b,
                               seq, H3_DIT_W4A8_VIDEO_PATCH, H3_DIT_W4A8_HIDDEN,
                               y, error, error_size))
        goto done;
    ok = 1;
done:
    free(modulation);
    free(normed);
    return ok;
}

/* Final head: AdaLN + dense on device (host AdaLN was ~300ms/step @ 48k tok).
 * H3_DIT_W4A8_HOST_DENSE=1 falls back to full host final_*.
 * If d_x_in != NULL, skip hidden H2D and read activations from device. */
static int final_head_cuda(h3_dit_w4a8 *ctx, const h3_dit_w4a8_core *core,
                           const float *x, const float *d_x_in, size_t seq,
                           const float *time_feat, float **d_w_slot,
                           float **d_b_slot, const float *weight,
                           const float *bias, size_t out_dim, float *y,
                           char *error, size_t error_size) {
    float *modulation = NULL;
    float *d_x, *d_mod, *d_normed, *d_y;
    uint32_t *d_map;
    float *d_w, *d_b, *d_norm;
    const size_t hidden = H3_DIT_W4A8_HIDDEN;
    const size_t mod_n =
        (size_t)H3_DIT_W4A8_FINAL_SLOTS * H3_DIT_W4A8_HIDDEN;
    size_t need, off;
    char *base;
    int ok = 0;
    const int x_on_dev = d_x_in != NULL;
    if (!ctx || !ctx->cuda) {
        fail(error, error_size, "final_head_cuda requires device path");
        return 0;
    }
    if (!x_on_dev && !x) {
        fail(error, error_size, "final_head_cuda missing x");
        return 0;
    }
    modulation = (float *)malloc(mod_n * sizeof(float));
    if (!modulation) {
        fail(error, error_size, "OOM final modulation");
        return 0;
    }
    if (!h3_dit_w4a8_final_adaln_project_f32(core, time_feat, modulation, error,
                                             error_size))
        goto done;
    d_norm = ensure_d_weight(&ctx->d_final_norm, core->final_norm, hidden, error,
                             error_size);
    d_w = ensure_d_weight(d_w_slot, weight, out_dim * hidden, error, error_size);
    d_b = ensure_d_weight(d_b_slot, bias, out_dim, error, error_size);
    if (!d_norm || !d_w || !d_b) goto done;
    need = (x_on_dev ? 0 : seq * hidden * sizeof(float)) +
           seq * hidden * sizeof(float) + /* normed */
           mod_n * sizeof(float) + seq * out_dim * sizeof(float) +
           H3_WS_MAP_BYTES(seq);
    if (!ctx->d_final_ws || ctx->d_final_ws_bytes < need) {
        void *fresh = NULL;
        if (cudaMalloc(&fresh, need) != cudaSuccess) {
            if (ctx->cache) {
                char room_err[256];
                size_t keep = need + ((size_t)256ULL << 20);
                (void)h3_w4a8_cache_make_room(ctx->cache, keep, room_err,
                                              sizeof(room_err));
            }
            if (cudaMalloc(&fresh, need) != cudaSuccess) {
                fail(error, error_size, "final_head_cuda OOM");
                goto done;
            }
        }
        if (ctx->d_final_ws) cudaFree(ctx->d_final_ws);
        ctx->d_final_ws = fresh;
        ctx->d_final_ws_bytes = need;
    }
    base = (char *)ctx->d_final_ws;
    off = 0;
    if (x_on_dev) {
        d_x = (float *)(uintptr_t)d_x_in;
    } else {
        d_x = (float *)(base + off);
        off += seq * hidden * sizeof(float);
    }
    d_normed = (float *)(base + off);
    off += seq * hidden * sizeof(float);
    d_mod = (float *)(base + off);
    off += mod_n * sizeof(float);
    d_y = (float *)(base + off);
    off += seq * out_dim * sizeof(float);
    d_map = (uint32_t *)(base + off);
    /* zeros map → all tokens read modulation row 0 (Metal/host final semantics).
     * NULL row_map would be identity and OOB on FINAL_SLOTS modulation. */
    if (cudaMemset(d_map, 0, seq * sizeof(uint32_t)) != cudaSuccess ||
        cudaMemcpy(d_mod, modulation, mod_n * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        fail(error, error_size, "final_head_cuda H2D");
        goto done;
    }
    if (!x_on_dev) {
        if (cudaMemcpy(d_x, x, seq * hidden * sizeof(float),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            fail(error, error_size, "final_head_cuda x H2D");
            goto done;
        }
    }
    if (!h3_cuda_adaln_f32_dev(ctx->cuda, d_x, d_norm, d_mod, d_map, seq, hidden,
                               H3_DIT_W4A8_FINAL_SLOTS, 0, 1, 1e-5f, d_normed,
                               NULL, error, error_size))
        goto done;
    if (!h3_cuda_gemm_f32_dd(ctx->cuda, d_normed, d_w, seq, out_dim, hidden, d_y,
                             NULL, error, error_size))
        goto done;
    if (!h3_cuda_bias_add_f32_dev(ctx->cuda, d_y, d_b, seq, out_dim, error,
                                  error_size))
        goto done;
    if (cudaMemcpy(y, d_y, seq * out_dim * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        fail(error, error_size, "final_head_cuda D2H");
        goto done;
    }
    ok = 1;
done:
    free(modulation);
    return ok;
}

int h3_dit_w4a8_final_video_cuda(h3_dit_w4a8 *ctx, const h3_dit_w4a8_core *core,
                                 const float *x, size_t seq,
                                 const float *time_feat, float *y, char *error,
                                 size_t error_size) {
    if (!core || !core->final_norm || !core->final_video_w ||
        !core->final_video_b || !x || !time_feat || !y || seq == 0) {
        fail(error, error_size, "invalid final_video args");
        return 0;
    }
    if (!ctx || !ctx->cuda || env_flag_is_one("H3_DIT_W4A8_HOST_DENSE"))
        return h3_dit_w4a8_final_video_f32(core, x, seq, time_feat, y, error,
                                           error_size);
    return final_head_cuda(ctx, core, x, NULL, seq, time_feat,
                           &ctx->d_final_video_w, &ctx->d_final_video_b,
                           core->final_video_w, core->final_video_b,
                           H3_DIT_W4A8_VIDEO_PATCH, y, error, error_size);
}

int h3_dit_w4a8_final_audio_f32(const h3_dit_w4a8_core *core, const float *x,
                                size_t seq, const float *time_feat, float *y,
                                char *error, size_t error_size) {
    float *modulation = NULL;
    float *normed = NULL;
    int ok = 0;
    if (!core || !core->final_norm || !core->final_audio_w ||
        !core->final_audio_b || !x || !time_feat || !y || seq == 0) {
        fail(error, error_size, "invalid final_audio args");
        return 0;
    }
    modulation = (float *)malloc((size_t)H3_DIT_W4A8_FINAL_SLOTS *
                                 H3_DIT_W4A8_HIDDEN * sizeof(float));
    normed = (float *)malloc(seq * H3_DIT_W4A8_HIDDEN * sizeof(float));
    if (!modulation || !normed) {
        fail(error, error_size, "out of memory (final_audio)");
        goto done;
    }
    if (!h3_dit_w4a8_final_adaln_project_f32(core, time_feat, modulation, error,
                                             error_size))
        goto done;
    {
        uint32_t *zeros = (uint32_t *)calloc(seq, sizeof(uint32_t));
        if (!zeros) {
            fail(error, error_size, "out of memory (final audio row_map)");
            goto done;
        }
        h3_dit_w4a8_adaln_f32(x, core->final_norm, modulation, zeros, seq,
                              H3_DIT_W4A8_HIDDEN, H3_DIT_W4A8_FINAL_SLOTS, 0, 1,
                              1e-5f, normed);
        free(zeros);
    }
    if (!h3_dit_w4a8_dense_f32(normed, core->final_audio_w, core->final_audio_b,
                               seq, H3_DIT_W4A8_AUDIO_CH, H3_DIT_W4A8_HIDDEN, y,
                               error, error_size))
        goto done;
    ok = 1;
done:
    free(modulation);
    free(normed);
    return ok;
}

int h3_dit_w4a8_final_audio_cuda(h3_dit_w4a8 *ctx, const h3_dit_w4a8_core *core,
                                 const float *x, size_t seq,
                                 const float *time_feat, float *y, char *error,
                                 size_t error_size) {
    if (!core || !core->final_norm || !core->final_audio_w ||
        !core->final_audio_b || !x || !time_feat || !y || seq == 0) {
        fail(error, error_size, "invalid final_audio args");
        return 0;
    }
    if (!ctx || !ctx->cuda || env_flag_is_one("H3_DIT_W4A8_HOST_DENSE"))
        return h3_dit_w4a8_final_audio_f32(core, x, seq, time_feat, y, error,
                                           error_size);
    return final_head_cuda(ctx, core, x, NULL, seq, time_feat,
                           &ctx->d_final_audio_w, &ctx->d_final_audio_b,
                           core->final_audio_w, core->final_audio_b,
                           H3_DIT_W4A8_AUDIO_CH, y, error, error_size);
}

static int uly_set_dev(const h3_dit_w4a8 *ctx, int rank) {
    if (!ctx || rank < 0 || rank >= ctx->uly_u)
        return 0;
    return cudaSetDevice(ctx->uly_dev[rank]) == cudaSuccess;
}

typedef struct {
    float *d_x, *d_y, *d_mod, *d_cos, *d_sin;
    float *d_n1, *d_n2, *d_qn, *d_kn;
    float *d_a, *d_b, *d_qkv, *d_wide;
    uint32_t *d_map;
} uly_rank_ws;

enum { ULY_PH_QKV = 1, ULY_PH_ATTN = 2, ULY_PH_MLP = 3 };

typedef struct {
    h3_dit_w4a8 *root;
    h3_dit_w4a8 *rc; /* uly_ctx[rank] only; never root's device buffers */
    uly_rank_ws *ws;
    h3_dit_w4a8_sp_comm *comm;
    int rank;
    int phase;
    unsigned block;
    const float *modulation;
    const h3_dit_w4a8_block_aux *aux;
    const h3_cuda_w4a8_weight *w_qkv, *w_out, *w_fc1, *w_fc2;
    size_t seq_local, seq_real, seq_pad;
    size_t heads, heads_attn, head_dim, hidden, mod_elems;
    void *compute_stream;
    void *ev_comp;
    void *ev_host;
    int ok;
    char err[384];
} uly_job;

static int uly_ptr_dev(int rank, int want, const void *p, const char *name,
                       char *err, size_t errn);

/* Per-rank GEMM/SDPA. Each thread binds its own device + ctx (U=1 never
 * enters this file path). A2A stays a collection point on the caller. */
static int uly_do_phase(uly_job *j) {
    h3_dit_w4a8 *rc;
    uly_rank_ws *w;
    char detail[384];
    int want;
    if (!j || !j->root || !j->rc || !j->ws) {
        if (j)
            snprintf(j->err, sizeof j->err, "ulysses: bad rank job");
        return 0;
    }
    if (!uly_set_dev(j->root, j->rank)) {
        snprintf(j->err, sizeof j->err, "ulysses: cudaSetDevice");
        return 0;
    }
    rc = j->rc;
    w = j->ws;
    want = j->root->uly_dev[j->rank];
    if (!h3_cuda_bind_device(rc->cuda, j->err, sizeof j->err))
        return 0;
    if (!h3_cuda_dit_scratch_check(rc->cuda, j->err, sizeof j->err))
        return 0;
    if (!uly_ptr_dev(j->rank, want, w->d_x, "d_x", j->err, sizeof j->err) ||
        !uly_ptr_dev(j->rank, want, w->d_qkv, "d_qkv", j->err, sizeof j->err) ||
        !uly_ptr_dev(j->rank, want, w->d_wide, "d_wide", j->err, sizeof j->err) ||
        !uly_ptr_dev(j->rank, want, w->d_a, "d_a", j->err, sizeof j->err) ||
        !uly_ptr_dev(j->rank, want, w->d_mod, "d_mod", j->err, sizeof j->err) ||
        !uly_ptr_dev(j->rank, want, w->d_n1, "d_n1", j->err, sizeof j->err) ||
        !uly_ptr_dev(j->rank, want, w->d_map, "d_map", j->err, sizeof j->err))
        return 0;
    if (rc->lora) {
        int which = (j->phase == ULY_PH_QKV) ? 0 : 1;
        if (!uly_ptr_dev(j->rank, want, rc->d_lora_A[j->block][which], "lora_A",
                         j->err, sizeof j->err) ||
            !uly_ptr_dev(j->rank, want, rc->d_lora_B[j->block][which], "lora_B",
                         j->err, sizeof j->err) ||
            !uly_ptr_dev(j->rank, want, rc->d_lora_t, "lora_T", j->err,
                         sizeof j->err))
            return 0;
    }
    g_uly_log_rank = j->rank;
    g_uly_log_phase = j->phase;
    g_uly_log_block = j->block;
    g_uly_phase_sync = env_flag_is_one("H3_ULYSSES_PHASE_SYNC");
    uly_klog(j->phase == ULY_PH_QKV
                 ? "phase1_qkv"
                 : (j->phase == ULY_PH_ATTN ? "phase2_attn" : "phase3_mlp"),
             "begin");
    detail[0] = '\0';
    /* Prefetch / cache_refit / x-scatter land on the default stream.
     * Compute is a non-blocking stream, so enqueue a default→compute
     * wait before any GEMM (no host drain). */
    if (j->compute_stream && j->ev_host &&
        (cudaEventRecord((cudaEvent_t)j->ev_host, 0) != cudaSuccess ||
         cudaStreamWaitEvent((cudaStream_t)j->compute_stream,
                             (cudaEvent_t)j->ev_host, 0) != cudaSuccess)) {
        snprintf(j->err, sizeof j->err, "ulysses: default→compute fence");
        return 0;
    }
    if (j->phase == ULY_PH_QKV) {
        if (!j->w_qkv) {
            snprintf(j->err, sizeof j->err, "ulysses: missing QKV weight");
            return 0;
        }
        if (!block_phase_qkv(rc, j->block, j->w_qkv, w->d_x, w->d_mod, w->d_map,
                             w->d_n1, w->d_a, w->d_qkv, j->seq_local, NULL,
                             detail, sizeof(detail))) {
            snprintf(j->err, sizeof j->err, "ulysses r%d phase1: %s", j->rank,
                     detail);
            return 0;
        }
        return 1;
    }
    if (j->phase == ULY_PH_ATTN) {
        size_t pad_elems;
        if (!block_phase_attn(rc, j->block, w->d_qkv, w->d_qn, w->d_kn, w->d_cos,
                              w->d_sin, j->seq_real, j->heads_attn, w->d_wide,
                              NULL, detail, sizeof(detail))) {
            snprintf(j->err, sizeof j->err, "ulysses r%d phase2: %s", j->rank,
                     detail);
            return 0;
        }
        pad_elems = (j->seq_pad - j->seq_real) * j->heads_attn * j->head_dim;
        if (pad_elems &&
            cudaMemsetAsync(
                w->d_wide + j->seq_real * j->heads_attn * j->head_dim, 0,
                pad_elems * sizeof(float),
                (cudaStream_t)j->compute_stream) != cudaSuccess) {
            snprintf(j->err, sizeof j->err, "ulysses: pad attn zero");
            return 0;
        }
        return 1;
    }
    if (j->phase == ULY_PH_MLP) {
        if (!j->w_out || !j->w_fc1 || !j->w_fc2) {
            snprintf(j->err, sizeof j->err, "ulysses: missing MLP weights");
            return 0;
        }
        if (!block_phase_out_mlp(rc, j->block, j->w_out, j->w_fc1, j->w_fc2,
                                 w->d_x, w->d_y, w->d_mod, w->d_map, w->d_n2,
                                 w->d_a, w->d_b, w->d_qkv, w->d_wide, w->d_wide,
                                 j->seq_local, NULL, detail, sizeof(detail))) {
            snprintf(j->err, sizeof j->err, "ulysses r%d phase3: %s", j->rank,
                     detail);
            return 0;
        }
        {
            float *tmp = w->d_x;
            w->d_x = w->d_y;
            w->d_y = tmp;
        }
        return 1;
    }
    snprintf(j->err, sizeof j->err, "ulysses: bad phase");
    return 0;
}

static double uly_now_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
        return 0.0;
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

/* After enqueue: default stream waits for this rank's compute stream so
 * A2A pack (comm stream ← default event) sees GEMM/SDPA. */
static int uly_join_compute_to_default(uly_job *jobs, int u, char *error,
                                       size_t error_size) {
    int r;
    for (r = 0; r < u; r++) {
        if (!uly_set_dev(jobs[r].root, r)) {
            fail(error, error_size, "ulysses: set device");
            return 0;
        }
        if (jobs[r].compute_stream && jobs[r].ev_comp) {
            if (cudaEventRecord((cudaEvent_t)jobs[r].ev_comp,
                                (cudaStream_t)jobs[r].compute_stream) !=
                    cudaSuccess ||
                cudaStreamWaitEvent(0, (cudaEvent_t)jobs[r].ev_comp, 0) !=
                    cudaSuccess) {
                fail(error, error_size, "ulysses: compute→default");
                return 0;
            }
        }
    }
    return 1;
}

/* After A2A: compute stream waits for default (unpack landed). */
static int uly_join_default_to_compute(uly_job *jobs, int u, char *error,
                                       size_t error_size) {
    int r;
    for (r = 0; r < u; r++) {
        if (!uly_set_dev(jobs[r].root, r)) {
            fail(error, error_size, "ulysses: set device");
            return 0;
        }
        if (jobs[r].compute_stream && jobs[r].ev_host) {
            if (cudaEventRecord((cudaEvent_t)jobs[r].ev_host, 0) !=
                    cudaSuccess ||
                cudaStreamWaitEvent((cudaStream_t)jobs[r].compute_stream,
                                    (cudaEvent_t)jobs[r].ev_host, 0) !=
                    cudaSuccess) {
                fail(error, error_size, "ulysses: default→compute");
                return 0;
            }
        }
    }
    return 1;
}

/* Why GEMM/SDPA did not overlap (old pthread + per-rank host wait):
 *   1. uly_do_phase was launch+wait per rank (sync memcpy / join too
 *      early). Rank r+1 did not enqueue until rank r's GPU finished.
 *   2. A host sync on rank r drained that device before cudaSetDevice
 *      of rank r+1, so the two default streams never ran together.
 *   3. Per-phase pthread_create/join re-attached the CUDA ctx every
 *      block and held the runtime lock across those waits.
 * Fix: enqueue every rank first (setDevice + kernels, no host wait),
 * THEN DeviceSynchronize all ranks. Side streams IMA'd cublasLt/cuDNN
 * on sm_120; default stream matches A2A and U=1. U=1 never enters. */
static int uly_join_ranks(uly_job *jobs, int u, char *error, size_t error_size) {
    int r;
    for (r = 0; r < u; r++) {
        if (!uly_set_dev(jobs[r].root, r)) {
            fail(error, error_size, "ulysses: set device");
            return 0;
        }
        if (jobs[r].compute_stream) {
            cudaError_t ce =
                cudaStreamSynchronize((cudaStream_t)jobs[r].compute_stream);
            if (ce != cudaSuccess) {
                snprintf(error, error_size, "ulysses: join r%d: %s", r,
                         cudaGetErrorString(ce));
                return 0;
            }
        } else {
            /* Default stream only — DeviceSynchronize drained copy/other. */
            cudaError_t ce = cudaStreamSynchronize((cudaStream_t)0);
            if (ce != cudaSuccess) {
                snprintf(error, error_size, "ulysses: join r%d: %s", r,
                         cudaGetErrorString(ce));
                return 0;
            }
        }
    }
    return 1;
}

/* Pageable aux H2D must be single-threaded: two threads sharing the
 * runtime bounce buffer IMA'd rank 1. Call before spawning workers. */
static int uly_h2d_aux(uly_job *j) {
    uly_rank_ws *w;
    if (!j || !j->root || !j->ws)
        return 0;
    if (j->phase != ULY_PH_QKV)
        return 1;
    if (!uly_set_dev(j->root, j->rank)) {
        snprintf(j->err, sizeof j->err, "ulysses: aux setDevice");
        return 0;
    }
    w = j->ws;
    {
        cudaStream_t st =
            j->compute_stream ? (cudaStream_t)j->compute_stream : (cudaStream_t)0;
        if (cudaMemcpyAsync(w->d_mod, j->modulation,
                            j->mod_elems * sizeof(float), cudaMemcpyHostToDevice,
                            st) != cudaSuccess ||
            cudaMemcpyAsync(w->d_n1, j->aux->norm1, j->hidden * sizeof(float),
                            cudaMemcpyHostToDevice, st) != cudaSuccess ||
            cudaMemcpyAsync(w->d_n2, j->aux->norm2, j->hidden * sizeof(float),
                            cudaMemcpyHostToDevice, st) != cudaSuccess ||
            cudaMemcpyAsync(w->d_qn, j->aux->q_norm,
                            j->head_dim * sizeof(float), cudaMemcpyHostToDevice,
                            st) != cudaSuccess ||
            cudaMemcpyAsync(w->d_kn, j->aux->k_norm,
                            j->head_dim * sizeof(float), cudaMemcpyHostToDevice,
                            st) != cudaSuccess) {
            snprintf(j->err, sizeof j->err, "ulysses: aux H2D");
            return 0;
        }
    }
    return 1;
}

/* Cache lookup + weight ready-wait on the main thread. Workers must not
 * load_block / ensure_ready (pageable unpack H2D + EventDestroy). */
static int uly_resolve_weights(uly_job *j) {
    h3_dit_w4a8 *rc;
    if (!j || !j->root) {
        if (j)
            snprintf(j->err, sizeof j->err, "ulysses: bad resolve");
        return 0;
    }
    if (!uly_set_dev(j->root, j->rank)) {
        snprintf(j->err, sizeof j->err, "ulysses: resolve setDevice");
        return 0;
    }
    rc = j->rc;
    if (!rc) {
        snprintf(j->err, sizeof j->err, "ulysses: resolve missing rc");
        return 0;
    }
    j->w_qkv = j->w_out = j->w_fc1 = j->w_fc2 = NULL;
    if (j->phase == ULY_PH_ATTN)
        return 1;
    if (j->phase == ULY_PH_QKV) {
        if (rc->nvfp4_on) return 1;
        j->w_qkv = get_weight(rc, j->block, H3_W4A8_LINEAR_QKV, j->err,
                              sizeof j->err);
        if (!j->w_qkv)
            return 0;
        /* Drain this rank's copy stream only. ensure_ready EventDestroy
         * here left rank-1 GEMM reading W still on streams[0] (IMA).
         * GEMM wait_ready keeps the event. U=1 never enters. */
        if (!h3_cuda_copy_stream_synchronize(rc->cuda, j->err, sizeof j->err))
            return 0;
        if (!h3_cuda_w4a8_weight_on_device(j->w_qkv, j->root->uly_dev[j->rank],
                                           j->err, sizeof j->err))
            return 0;
        return 1;
    }
    if (j->phase == ULY_PH_MLP) {
        if (rc->nvfp4_on) return 1;
        j->w_out = get_weight(rc, j->block, H3_W4A8_LINEAR_OUT, j->err,
                              sizeof j->err);
        j->w_fc1 = j->w_out ? get_weight(rc, j->block, H3_W4A8_LINEAR_FC1,
                                         j->err, sizeof j->err)
                            : NULL;
        j->w_fc2 = j->w_fc1 ? get_weight(rc, j->block, H3_W4A8_LINEAR_FC2,
                                         j->err, sizeof j->err)
                            : NULL;
        if (!j->w_fc2)
            return 0;
        if (!h3_cuda_copy_stream_synchronize(rc->cuda, j->err, sizeof j->err) ||
            !h3_cuda_w4a8_weight_on_device(j->w_out, j->root->uly_dev[j->rank],
                                           j->err, sizeof j->err) ||
            !h3_cuda_w4a8_weight_on_device(j->w_fc1, j->root->uly_dev[j->rank],
                                           j->err, sizeof j->err) ||
            !h3_cuda_w4a8_weight_on_device(j->w_fc2, j->root->uly_dev[j->rank],
                                           j->err, sizeof j->err))
            return 0;
        return 1;
    }
    snprintf(j->err, sizeof j->err, "ulysses: bad phase");
    return 0;
}

/* Bind A2A after all ranks have joined compute. Doing this from workers
 * raced on comm->cfg while kernels were still in flight. */
static int uly_bind_phase(uly_job *jobs, int u, int phase, char *error,
                          size_t error_size) {
    int r;
    if (phase != ULY_PH_QKV && phase != ULY_PH_ATTN)
        return 1;
    /* After join only: writing shared comm->cfg while GEMM still uses
     * d_qkv/d_wide IMA'd rank 1 (pointer registered on the wrong device). */
    for (r = 0; r < u; r++) {
        uly_rank_ws *w = jobs[r].ws;
        const void *p = (phase == ULY_PH_QKV) ? (const void *)w->d_qkv
                                              : (const void *)w->d_wide;
        const char *name = (phase == ULY_PH_QKV) ? "d_qkv" : "d_wide";
        int want = jobs[r].root->uly_dev[r];
        if (!uly_set_dev(jobs[r].root, r)) {
            fail(error, error_size, "ulysses: bind setDevice");
            return 0;
        }
        if (!uly_ptr_dev(r, want, p, name, error, error_size))
            return 0;
        if (!jobs[r].comm->bind(jobs[r].comm->cfg, jobs[r].rank, p, (void *)p,
                                error, error_size))
            return 0;
    }
    return 1;
}

static int uly_ptr_dev(int rank, int want, const void *p, const char *name,
                       char *err, size_t errn) {
    struct cudaPointerAttributes a;
    cudaError_t ce;
    if (!p)
        return 1;
    memset(&a, 0, sizeof(a));
    ce = cudaPointerGetAttributes(&a, p);
    if (ce != cudaSuccess) {
        snprintf(err, errn, "ulysses r%d %s: %s", rank, name,
                 cudaGetErrorString(ce));
        return 0;
    }
    if (a.type != cudaMemoryTypeDevice && a.type != cudaMemoryTypeManaged) {
        snprintf(err, errn, "ulysses r%d %s: type=%d", rank, name, (int)a.type);
        return 0;
    }
    if (a.device != want) {
        snprintf(err, errn, "ulysses r%d %s: on dev %d want %d", rank, name,
                 a.device, want);
        return 0;
    }
    return 1;
}

static int uly_sync_rank(uly_job *j, char *error, size_t error_size) {
    cudaError_t ce;
    if (!uly_set_dev(j->root, j->rank)) {
        fail(error, error_size, "ulysses: set device");
        return 0;
    }
    /* Per-rank compute stream (or event), not DeviceSynchronize — that
     * drained copy/other streams every phase. */
    if (j->compute_stream)
        ce = cudaStreamSynchronize((cudaStream_t)j->compute_stream);
    else if (j->ev_comp)
        ce = cudaEventSynchronize((cudaEvent_t)j->ev_comp);
    else
        ce = cudaStreamSynchronize((cudaStream_t)0);
    if (ce != cudaSuccess) {
        snprintf(error, error_size, "ulysses: join r%d phase%d: %s", j->rank,
                 j->phase, cudaGetErrorString(ce));
        uly_klog("join", "IMA");
        return 0;
    }
    uly_klog("join", "ok");
    return 1;
}

/* Main thread: serial aux H2D + weight resolve (pageable / cache).
 * H3_ULYSSES_SERIAL=1: one rank to completion before the next (isolate
 * shared-handle vs bad pointer). Default: enqueue all, then join all.
 * Each job uses uly_ctx[rank] cublas/scratch/weight cache only. */
static int uly_run_ranks(uly_job *jobs, int u, int phase, char *error,
                         size_t error_size, int join_each) {
    int r;
    const int serial = env_flag_is_one("H3_ULYSSES_SERIAL");
    if (!jobs || u < 1 || u > 8) {
        fail(error, error_size, "ulysses: bad rank fanout");
        return 0;
    }
    (void)join_each;
    for (r = 0; r < u; r++) {
        jobs[r].phase = phase;
        jobs[r].ok = 0;
        jobs[r].err[0] = '\0';
        if (!uly_h2d_aux(&jobs[r])) {
            snprintf(error, error_size, "%s",
                     jobs[r].err[0] ? jobs[r].err : "ulysses: aux H2D");
            return 0;
        }
        if (!uly_resolve_weights(&jobs[r])) {
            snprintf(error, error_size, "%s",
                     jobs[r].err[0] ? jobs[r].err : "ulysses: resolve");
            return 0;
        }
        if (serial) {
            fprintf(stderr, "uly_run: serial phase%d r%d block%u enqueue\n",
                    phase, r, jobs[r].block);
            fflush(stderr);
            if (!uly_do_phase(&jobs[r])) {
                snprintf(error, error_size, "%s",
                         jobs[r].err[0] ? jobs[r].err : "ulysses: enqueue");
                return 0;
            }
            if (!uly_sync_rank(&jobs[r], error, error_size))
                return 0;
            g_uly_log_rank = -1;
        }
    }
    if (serial)
        return 1;
    for (r = 0; r < u; r++) {
        fprintf(stderr, "uly_run: concurrent phase%d r%d block%u enqueue\n",
                phase, r, jobs[r].block);
        fflush(stderr);
        if (!uly_do_phase(&jobs[r])) {
            snprintf(error, error_size, "%s",
                     jobs[r].err[0] ? jobs[r].err : "ulysses: enqueue");
            g_uly_log_rank = -1;
            return 0;
        }
    }
    for (r = 0; r < u; r++) {
        g_uly_log_rank = jobs[r].rank;
        g_uly_log_phase = phase;
        g_uly_log_block = jobs[r].block;
        if (!uly_sync_rank(&jobs[r], error, error_size)) {
            g_uly_log_rank = -1;
            return 0;
        }
    }
    g_uly_log_rank = -1;
    return 1;
}

/* U>1 SP loop. Pad rows never enter SDPA (S_real only). Same-block ranks
 * run GEMM/SDPA concurrently; A2A is still a barrier. */
static int layers_forward_ulysses(
    h3_dit_w4a8 *ctx, unsigned first_block, unsigned n_blocks, size_t t_row,
    const float *x, size_t seq, const uint32_t *row_map, const float *rope_cos,
    const float *rope_sin, float *y, int auto_prefetch,
    h3_dit_w4a8_layers_stats *stats, char *error, size_t error_size) {
    const int u = ctx->uly_u;
    const size_t hidden = H3_DIT_W4A8_HIDDEN;
    const size_t heads = H3_DIT_W4A8_HEADS;
    const size_t head_dim = H3_DIT_W4A8_HEAD_DIM;
    const size_t inner = H3_DIT_W4A8_INNER;
    const size_t ffn = H3_DIT_W4A8_FFN;
    const size_t wide = ffn * 2;
    const size_t rope_half = (size_t)H3_DIT_W4A8_ROPE_HALF;
    const size_t seq_real = seq;
    const size_t seq_pad = (seq + 63u) / 64u * 64u;
    const size_t seq_local = seq_pad / (size_t)u;
    const size_t heads_attn = heads / (size_t)u;
    const size_t mod_elems = dit_adaln_mod_elems(ctx);
    uly_rank_ws ws[8];
    uly_job jobs[8];
    cudaStream_t uly_st[8];
    cudaEvent_t uly_ev_c[8], uly_ev_h[8];
    float *modulation = NULL;
    int mod_pinned = 0;
    h3_dit_w4a8_sp_comm *comm = &ctx->uly_comm;
    int orig = -1, ok = 0, r;
    double ms_qkv = 0.0, ms_a2a_q = 0.0, ms_attn = 0.0, ms_a2a_a = 0.0,
           ms_mlp = 0.0;
    int uly_warmed = 0;
    unsigned b;
    clock_t t0, t1, t_all0, t_all1;
    char detail[384];
    const int want_quiet = !env_flag_is_one("H3_DIT_W4A8_VERBOSE_TIMING");
    const int rank_serial = env_flag_is_one("H3_ULYSSES_SERIAL");

    memset(ws, 0, sizeof(ws));
    memset(jobs, 0, sizeof(jobs));
    memset(uly_st, 0, sizeof(uly_st));
    memset(uly_ev_c, 0, sizeof(uly_ev_c));
    memset(uly_ev_h, 0, sizeof(uly_ev_h));
    cudaGetDevice(&orig);
    t_all0 = clock();
    if (ctx->token_reduction) {
        fail(error, error_size, "ulysses: token-reduction not supported");
        return 0;
    }
    if (!rope_cos || !rope_sin) {
        fail(error, error_size, "ulysses: rope tables required");
        return 0;
    }
    {
        int have_dev_in = ctx->io_d_act_a && (ctx->io_flags & 1);
        int have_dev_out = ctx->io_d_act_b && (ctx->io_flags & 2);
        if (!x && !have_dev_in) {
            fail(error, error_size, "ulysses: missing host x / device act_a");
            return 0;
        }
        if (!y && !have_dev_out) {
            fail(error, error_size, "ulysses: missing host y / device act_b");
            return 0;
        }
    }
    if (u < 2 || heads % (size_t)u != 0 || seq_pad % (size_t)u != 0) {
        fail(error, error_size, "ulysses: bad U/heads/pad geometry");
        return 0;
    }
    if (!comm->bind || !comm->a2a_qkv || !comm->a2a_attn_out || !comm->cfg) {
        fail(error, error_size, "ulysses: comm vtable incomplete");
        return 0;
    }
    if (cudaMallocHost((void **)&modulation, mod_elems * sizeof(float)) ==
        cudaSuccess) {
        mod_pinned = 1;
    } else {
        modulation = (float *)malloc(mod_elems * sizeof(float));
    }
    if (!modulation) {
        fail(error, error_size, "ulysses: OOM modulation");
        return 0;
    }

    for (r = 0; r < u; r++) {
        h3_dit_w4a8 *rc = ctx->uly_ctx[r];
        char *base;
        size_t off = 0, need = 0;
        size_t start, nreal;
        if (!uly_set_dev(ctx, r)) {
            fail(error, error_size, "ulysses: cudaSetDevice");
            goto done;
        }
        need += seq_local * hidden * sizeof(float) * 2;
        need += mod_elems * sizeof(float);
        need += H3_WS_MAP_BYTES(seq_local);
        need += seq_real * rope_half * sizeof(float) * 2;
        need += (hidden * 2 + head_dim * 2) * sizeof(float);
        need += seq_local * hidden * sizeof(float) * 2;
        need += seq_local * inner * 3 * sizeof(float);
        need += seq_local * wide * sizeof(float);
        if (!ensure_block_ws(rc, need, error, error_size))
            goto done;
        if ((rc->int8_store || rc->nvfp4_store) && rc->cache &&
            !h3_w4a8_cache_refit_after_workspace(rc->cache, detail,
                                                 sizeof(detail))) {
            snprintf(error, error_size, "INT8 cache refit: %s", detail);
            goto done;
        }
        /* Drop any scratch that landed on another GPU during TE/open,
         * then recreate cublas on this rank's device. U=1 never enters. */
        if (!h3_cuda_bind_device(rc->cuda, detail, sizeof(detail))) {
            snprintf(error, error_size, "ulysses bind: %s", detail);
            goto done;
        }
        h3_cuda_release_heavy_workspace(rc->cuda);
        if (!h3_cuda_dit_scratch_rehome(rc->cuda, detail, sizeof(detail))) {
            snprintf(error, error_size, "scratch rehome: %s", detail);
            goto done;
        }
        if (!h3_cuda_recreate_blas(rc->cuda, detail, sizeof(detail))) {
            snprintf(error, error_size, "ulysses blas: %s", detail);
            goto done;
        }
        if (!h3_cuda_pregrow_dit_scratch(rc->cuda, seq_real, hidden, heads_attn,
                                         head_dim, ffn, detail,
                                         sizeof(detail))) {
            snprintf(error, error_size, "dit scratch: %s", detail);
            goto done;
        }
        if (rc->convrot_gs > 0 &&
            !h3_cuda_pregrow_hadamard(rc->cuda, rc->convrot_gs, detail,
                                      sizeof(detail))) {
            snprintf(error, error_size, "hadamard: %s", detail);
            goto done;
        }
        base = (char *)rc->d_block_ws;
#define H3_ULY_F32(n)                                                          \
    (off += (n) * sizeof(float), (float *)(base + (off - (n) * sizeof(float))))
        ws[r].d_x = H3_ULY_F32(seq_local * hidden);
        ws[r].d_y = H3_ULY_F32(seq_local * hidden);
        ws[r].d_mod = H3_ULY_F32(mod_elems);
        {
            size_t map_bytes = H3_WS_MAP_BYTES(seq_local);
            ws[r].d_map = (uint32_t *)(base + off);
            off += map_bytes;
        }
        ws[r].d_cos = H3_ULY_F32(seq_real * rope_half);
        ws[r].d_sin = H3_ULY_F32(seq_real * rope_half);
        ws[r].d_n1 = H3_ULY_F32(hidden);
        ws[r].d_n2 = H3_ULY_F32(hidden);
        ws[r].d_qn = H3_ULY_F32(head_dim);
        ws[r].d_kn = H3_ULY_F32(head_dim);
        ws[r].d_a = H3_ULY_F32(seq_local * hidden);
        ws[r].d_b = H3_ULY_F32(seq_local * hidden);
        ws[r].d_qkv = H3_ULY_F32(seq_local * inner * 3);
        ws[r].d_wide = H3_ULY_F32(seq_local * wide);
#undef H3_ULY_F32
        start = (size_t)r * seq_local;
        nreal = 0;
        if (start < seq_real) {
            nreal = seq_real - start;
            if (nreal > seq_local)
                nreal = seq_local;
        }
        if (nreal) {
            cudaError_t ce;
            if (x) {
                ce = cudaMemcpy(ws[r].d_x, x + start * hidden,
                                nreal * hidden * sizeof(float),
                                cudaMemcpyHostToDevice);
            } else {
                /* Rank-0 full act → shard. Peer if enabled, else driver
                 * stages through host (GeForce PCIe typically no P2P). */
                ce = cudaMemcpy(ws[r].d_x,
                                ctx->io_d_act_a + start * hidden,
                                nreal * hidden * sizeof(float),
                                cudaMemcpyDefault);
            }
            if (ce != cudaSuccess) {
                fail(error, error_size, "ulysses: x scatter");
                goto done;
            }
        }
        if (nreal < seq_local &&
            cudaMemset(ws[r].d_x + nreal * hidden, 0,
                       (seq_local - nreal) * hidden * sizeof(float)) !=
                cudaSuccess) {
            fail(error, error_size, "ulysses: x pad");
            goto done;
        }
        if (row_map) {
            if (nreal &&
                cudaMemcpy(ws[r].d_map, row_map + start,
                           nreal * sizeof(uint32_t),
                           cudaMemcpyHostToDevice) != cudaSuccess) {
                fail(error, error_size, "ulysses: map H2D");
                goto done;
            }
            if (nreal < seq_local &&
                cudaMemset(ws[r].d_map + nreal, 0,
                           (seq_local - nreal) * sizeof(uint32_t)) !=
                    cudaSuccess) {
                fail(error, error_size, "ulysses: map pad");
                goto done;
            }
        } else {
            ws[r].d_map = NULL;
        }
        if (cudaMemcpy(ws[r].d_cos, rope_cos,
                       seq_real * rope_half * sizeof(float),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(ws[r].d_sin, rope_sin,
                       seq_real * rope_half * sizeof(float),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            fail(error, error_size, "ulysses: rope H2D");
            goto done;
        }
        if (want_quiet)
            h3_cuda_set_quiet(rc->cuda, 1);
        jobs[r].root = ctx;
        jobs[r].rc = rc;
        jobs[r].ws = &ws[r];
        jobs[r].comm = comm;
        jobs[r].rank = r;
        jobs[r].seq_local = seq_local;
        jobs[r].seq_real = seq_real;
        jobs[r].seq_pad = seq_pad;
        jobs[r].heads = heads;
        jobs[r].heads_attn = heads_attn;
        jobs[r].head_dim = head_dim;
        jobs[r].hidden = hidden;
        jobs[r].mod_elems = mod_elems;
        /* Default stream per device (same as U=1). Overlap comes from
         * enqueue-all-then-join-all, not from a side stream: cublasLt /
         * cuDNN IMA'd on cudaStreamNonBlocking. A2A pack already waits
         * on stream 0. */
        jobs[r].compute_stream = NULL;
        jobs[r].ev_comp = NULL;
        jobs[r].ev_host = NULL;
    }
    printf("ulysses_rank_parallel: U=%d serial=%d enqueue=%s streams=0 "
           "phase_sync=%d\n",
           u, rank_serial, rank_serial ? "per-rank" : "all-then-join",
           env_flag_is_one("H3_ULYSSES_PHASE_SYNC"));
    fflush(stdout);

    if (auto_prefetch) {
        for (r = 0; r < u; r++) {
            double ignore = 0.0;
            unsigned n_pref = n_blocks;
            unsigned win;
            if (!uly_set_dev(ctx, r))
                goto done;
            win = h3_w4a8_cache_window_size(ctx->uly_ctx[r]->cache);
            if (win > 0 && win < n_pref)
                n_pref = win;
            if (!h3_dit_w4a8_prefetch(ctx->uly_ctx[r], first_block, n_pref,
                                      &ignore, error, error_size))
                goto done;
            if (stats && r == 0)
                stats->prefetch_ms = ignore;
        }
    }

    t0 = clock();
    for (b = 0; b < n_blocks; b++) {
        unsigned block = first_block + b;
        const h3_dit_w4a8_block_aux *aux =
            dit_aux_cached(ctx, block, error, error_size);
        if (!aux)
            goto done;
        if (!project_block_adaln_metal(ctx, aux, t_row, modulation, error,
                                       error_size))
            goto done;
        if (b + 1 < n_blocks) {
            for (r = 0; r < u; r++) {
                double ignore = 0.0;
                if (!uly_set_dev(ctx, r))
                    goto done;
                if (!h3_dit_w4a8_prefetch(ctx->uly_ctx[r], block + 1, 1,
                                          &ignore, error, error_size))
                    goto done;
            }
        }
        for (r = 0; r < u; r++) {
            jobs[r].block = block;
            jobs[r].modulation = modulation;
            jobs[r].aux = aux;
        }
        {
            double t0p = uly_now_ms(), t1p;
            if (!uly_run_ranks(jobs, u, ULY_PH_QKV, error, error_size,
                               !uly_warmed))
                goto done;
            uly_warmed = 1;
            if (!uly_join_compute_to_default(jobs, u, error, error_size))
                goto done;
            if (!uly_bind_phase(jobs, u, ULY_PH_QKV, error, error_size))
                goto done;
            t1p = uly_now_ms();
            ms_qkv += t1p - t0p;
            t0p = t1p;
            if (!comm->a2a_qkv(comm->cfg, seq_local, heads, head_dim, error,
                               error_size))
                goto done;
            if (!uly_join_default_to_compute(jobs, u, error, error_size))
                goto done;
            t1p = uly_now_ms();
            ms_a2a_q += t1p - t0p;
            t0p = t1p;
            if (!uly_run_ranks(jobs, u, ULY_PH_ATTN, error, error_size, 0))
                goto done;
            if (!uly_join_compute_to_default(jobs, u, error, error_size))
                goto done;
            if (!uly_bind_phase(jobs, u, ULY_PH_ATTN, error, error_size))
                goto done;
            t1p = uly_now_ms();
            ms_attn += t1p - t0p;
            t0p = t1p;
            if (!comm->a2a_attn_out(comm->cfg, seq_local, heads, head_dim,
                                    error, error_size))
                goto done;
            if (!uly_join_default_to_compute(jobs, u, error, error_size))
                goto done;
            t1p = uly_now_ms();
            ms_a2a_a += t1p - t0p;
            t0p = t1p;
            if (!uly_run_ranks(jobs, u, ULY_PH_MLP, error, error_size, 0))
                goto done;
            t1p = uly_now_ms();
            ms_mlp += t1p - t0p;
        }
    }

    if (!uly_join_compute_to_default(jobs, u, error, error_size))
        goto done;
    for (r = 0; r < u; r++) {
        size_t start = (size_t)r * seq_local;
        size_t nreal = 0;
        if (start < seq_real) {
            nreal = seq_real - start;
            if (nreal > seq_local)
                nreal = seq_local;
        }
        if (!nreal)
            continue;
        if (!uly_set_dev(ctx, r))
            goto done;
        {
            cudaError_t ce;
            if (y) {
                ce = cudaMemcpy(y + start * hidden, ws[r].d_x,
                                nreal * hidden * sizeof(float),
                                cudaMemcpyDeviceToHost);
            } else {
                ce = cudaMemcpy(ctx->io_d_act_b + start * hidden, ws[r].d_x,
                                nreal * hidden * sizeof(float),
                                cudaMemcpyDefault);
            }
            if (ce != cudaSuccess) {
                fail(error, error_size, "ulysses: y gather");
                goto done;
            }
        }
    }
    if (!y && ctx->io_d_act_b)
        ctx->io_d_result = ctx->io_d_act_b;

    t1 = clock();
    t_all1 = clock();
    if (stats) {
        size_t i, n = seq_real * hidden;
        size_t stride = n > 4096 ? (n / 4096) : 1;
        stats->first_block = first_block;
        stats->n_blocks = n_blocks;
        stats->blocks_ms = elapsed_ms(t0, t1);
        stats->total_ms = elapsed_ms(t_all0, t_all1);
        stats->all_finite = 1;
        if (y) {
            for (i = 0; i < n; i += stride) {
                if (!isfinite(y[i])) {
                    stats->all_finite = 0;
                    break;
                }
            }
        }
    }
    printf("ulysses_split: U=%d qkv_enq=%.1f a2a_q(+qkv_gpu)=%.1f "
           "attn_enq=%.1f a2a_a(+sdpa_gpu)=%.1f mlp_enq=%.1f ms\n",
           u, ms_qkv, ms_a2a_q, ms_attn, ms_a2a_a, ms_mlp);
    fflush(stdout);
    ok = 1;
done:
    for (r = 0; r < u; r++) {
        char ign[64];
        if (ctx->uly_ctx[r] && ctx->uly_ctx[r]->cuda) {
            h3_cuda_set_quiet(ctx->uly_ctx[r]->cuda, 0);
            (void)h3_cuda_set_op_stream(ctx->uly_ctx[r]->cuda, NULL, ign,
                                        sizeof(ign));
        }
        if (comm->set_compute)
            (void)comm->set_compute(comm->cfg, r, NULL, ign, sizeof(ign));
        if (uly_st[r] || uly_ev_c[r] || uly_ev_h[r]) {
            if (ctx->uly_dev[r] >= 0)
                cudaSetDevice(ctx->uly_dev[r]);
            if (uly_st[r])
                cudaStreamDestroy(uly_st[r]);
            if (uly_ev_c[r])
                cudaEventDestroy(uly_ev_c[r]);
            if (uly_ev_h[r])
                cudaEventDestroy(uly_ev_h[r]);
        }
    }
    if (orig >= 0)
        cudaSetDevice(orig);
    if (mod_pinned)
        cudaFreeHost(modulation);
    else
        free(modulation);
    return ok;
}

int h3_dit_w4a8_layers_forward_f32(
    h3_dit_w4a8 *ctx, unsigned first_block, unsigned n_blocks, size_t t_row,
    const float *x, size_t seq, const uint32_t *row_map, const float *rope_cos,
    const float *rope_sin, float *y, int auto_prefetch,
    h3_dit_w4a8_layers_stats *stats, char *error, size_t error_size) {
    float *buf_a = NULL, *buf_b = NULL;
    float *modulation = NULL;
    float time_feat[H3_DIT_W4A8_TIME_FEAT];
    const float *cur;
    float *nxt;
    unsigned b;
    clock_t t0, t1, t_all0, t_all1;
    double gemm_sum = 0.0;
    int ok = 0;
    const int force_host = env_flag_is_one("H3_DIT_W4A8_FORCE_HOST_BLOCK");
    const int use_dev_chain =
        ctx && ctx->cuda && !force_host && n_blocks >= 1 &&
        !env_flag_is_one("H3_DIT_W4A8_FORCE_SDPA_F32");
    const int use_ext_act =
        use_dev_chain && ctx->io_d_act_a && ctx->io_d_act_b;
    const int skip_act_h2d = use_ext_act && (ctx->io_flags & 1);
    const int leave_act_dev = use_ext_act && (ctx->io_flags & 2);

    if (stats) memset(stats, 0, sizeof(*stats));
    if (ctx) ctx->io_d_result = NULL;
    if (!ctx || (!x && !skip_act_h2d) || (!y && !leave_act_dev) ||
        !rope_cos || !rope_sin || seq == 0 || n_blocks == 0) {
        fail(error, error_size, "invalid layers_forward args");
        return 0;
    }
    if ((size_t)first_block + (size_t)n_blocks > H3_DIT_W4A8_N_BLOCKS) {
        fail(error, error_size, "layers_forward block range out of bounds");
        return 0;
    }
    if (ctx->uly_u > 1)
        return layers_forward_ulysses(ctx, first_block, n_blocks, t_row, x, seq,
                                      row_map, rope_cos, rope_sin, y,
                                      auto_prefetch, stats, error, error_size);

    if (ctx->cache)
        h3_w4a8_cache_set_async_h2d(ctx->cache, 1);

    if (stats) {
        stats->first_block = first_block;
        stats->n_blocks = n_blocks;
    }
    t_all0 = clock();

    /* WS-first: on the device-chained path, activation workspace
     * (ensure_block_ws / ensure_fwd_act) must be allocated before weight
     * residency. Defer auto_prefetch to the warm-window call after WS.
     * Host / per-block path still prefetches up front. */
    if (auto_prefetch && !use_dev_chain) {
        double pms = 0.0;
        if (!h3_dit_w4a8_prefetch(ctx, first_block, n_blocks, &pms, error,
                                  error_size))
            return 0;
        if (stats) stats->prefetch_ms = pms;
    }

    /* Device-chained path: one H2D of x/rope/all-aux + one D2H of y.
     * Aux is cached on host and prestaged for all layers (no per-layer
     * safetensors reload / dual-bank StreamSynchronize).
     * Resident forward may supply external act ping-pong (skip act PCIe). */
    if (use_dev_chain) {
        const size_t hidden = H3_DIT_W4A8_HIDDEN;
        const size_t inner = H3_DIT_W4A8_INNER;
        const size_t ffn = H3_DIT_W4A8_FFN;
        const size_t slots = H3_DIT_W4A8_SLOTS;
        const size_t mod_rows = dit_adaln_mod_rows(ctx);
        const size_t rope_half = (size_t)H3_DIT_W4A8_ROPE_HALF;
        const size_t mod_elems = mod_rows * slots * hidden;
        const size_t head_dim = H3_DIT_W4A8_HEAD_DIM;
        const size_t wide = ffn * 2;
        const size_t aux_stride =
            mod_elems + hidden * 2 + head_dim * 2; /* floats per layer */
        const size_t act_bytes =
            use_ext_act ? 0 : (seq * hidden * sizeof(float) * 2);
        float *d_x, *d_y, *d_cos, *d_sin, *d_a, *d_b, *d_qkv, *d_wide;
        float *d_aux_base;
        uint32_t *d_map;
        float *h_aux_pack = NULL;
        int h_aux_pinned = 0;
        char *base;
        size_t off = 0, need;
        char detail[384];
        void *copy_stream;
        const int want_quiet =
            !env_flag_is_one("H3_DIT_W4A8_VERBOSE_TIMING");

        need = 0;
        need += act_bytes;
        need += H3_WS_MAP_BYTES(seq);
        need += seq * rope_half * sizeof(float) * 2;
        need += n_blocks * aux_stride * sizeof(float);
        need += seq * hidden * sizeof(float) * 2;
        need += seq * inner * 3 * sizeof(float);
        need += seq * wide * sizeof(float);
        /* Do not proactively evict for (WS+8GiB) — that collapsed cafe
         * window 29→16 after step 1. Allocate WS; only make_room when
         * there is no existing block WS at all. A slightly-short old WS
         * falls through to dual-bank instead of dropping 13 layers. */
        if (!ensure_block_ws(ctx, need, detail, sizeof(detail))) {
            if ((ctx->int8_store || ctx->nvfp4_store) && ctx->cache &&
                !ctx->d_block_ws &&
                !env_flag_is_one("H3_DIT_NO_WS_MAKE_ROOM")) {
                char room_err[256];
                size_t keep = need + ((size_t)512ULL << 20);
                if (!h3_w4a8_cache_make_room(ctx->cache, keep, room_err,
                                             sizeof(room_err)))
                    fprintf(stderr, "dit_int8_cache_make_room: %s\n",
                            room_err);
                if (ensure_block_ws(ctx, need, detail, sizeof(detail)))
                    goto ws_ready_allaux;
            }
            size_t need_bank = 0;
            need_bank += act_bytes;
            need_bank += mod_elems * sizeof(float) * 2;
            need_bank += H3_WS_MAP_BYTES(seq);
            need_bank += seq * rope_half * sizeof(float) * 2;
            need_bank += (hidden * 2 + head_dim * 2) * sizeof(float) * 2;
            need_bank += seq * hidden * sizeof(float) * 2;
            need_bank += seq * inner * 3 * sizeof(float);
            need_bank += seq * wide * sizeof(float);
            if (!ensure_block_ws(ctx, need_bank, error, error_size)) goto done;
            /* Dual-bank path with cached host aux (no safetensors reload). */
            ctx->aux_prestage_mode = 2;
            if (!h3_cuda_pregrow_dit_scratch(
                    ctx->cuda, seq, hidden, H3_DIT_W4A8_HEADS, head_dim, ffn,
                    detail, sizeof(detail))) {
                snprintf(error, error_size, "dit scratch: %s", detail);
                goto done;
            }
            if ((ctx->int8_store || ctx->nvfp4_store) &&
                !h3_w4a8_cache_refit_after_workspace(ctx->cache, detail,
                                                     sizeof(detail))) {
                snprintf(error, error_size, "INT8 cache refit: %s", detail);
                goto done;
            }
            {
                float *d_mod[2], *d_n1[2], *d_n2[2], *d_qn[2], *d_kn[2];
                int bank = 0, next_ready = 0;
                off = 0;
                base = (char *)ctx->d_block_ws;
#define H3_WS_F32(n)                                                           \
    (off += (n) * sizeof(float),                                               \
     (float *)(base + (off - (n) * sizeof(float))))
                if (use_ext_act) {
                    d_x = ctx->io_d_act_a;
                    d_y = ctx->io_d_act_b;
                } else {
                    d_x = H3_WS_F32(seq * hidden);
                    d_y = H3_WS_F32(seq * hidden);
                }
                d_mod[0] = H3_WS_F32(mod_elems);
                d_mod[1] = H3_WS_F32(mod_elems);
                {
                    size_t map_bytes = H3_WS_MAP_BYTES(seq);
                    d_map = (uint32_t *)(base + off);
                    off += map_bytes;
                }
                d_cos = H3_WS_F32(seq * rope_half);
                d_sin = H3_WS_F32(seq * rope_half);
                d_n1[0] = H3_WS_F32(hidden);
                d_n2[0] = H3_WS_F32(hidden);
                d_qn[0] = H3_WS_F32(head_dim);
                d_kn[0] = H3_WS_F32(head_dim);
                d_n1[1] = H3_WS_F32(hidden);
                d_n2[1] = H3_WS_F32(hidden);
                d_qn[1] = H3_WS_F32(head_dim);
                d_kn[1] = H3_WS_F32(head_dim);
                d_a = H3_WS_F32(seq * hidden);
                d_b = H3_WS_F32(seq * hidden);
                d_qkv = H3_WS_F32(seq * inner * 3);
                d_wide = H3_WS_F32(seq * wide);
#undef H3_WS_F32
                modulation =
                    (float *)malloc(dit_adaln_mod_elems(ctx) * sizeof(float));
                if (!modulation) {
                    fail(error, error_size, "out of memory (layers_forward)");
                    goto done;
                }
                copy_stream = h3_cuda_stream(ctx->cuda, 0);
                if (!skip_act_h2d) {
                    if (cudaMemcpy(d_x, x, seq * hidden * sizeof(float),
                                   cudaMemcpyHostToDevice) != cudaSuccess) {
                        fail(error, error_size, "layers resident H2D failed");
                        goto done;
                    }
                }
                if (cudaMemcpy(d_cos, rope_cos, seq * rope_half * sizeof(float),
                               cudaMemcpyHostToDevice) != cudaSuccess ||
                    cudaMemcpy(d_sin, rope_sin, seq * rope_half * sizeof(float),
                               cudaMemcpyHostToDevice) != cudaSuccess) {
                    fail(error, error_size, "layers resident H2D failed");
                    goto done;
                }
                if (row_map) {
                    if (cudaMemcpy(d_map, row_map, seq * sizeof(uint32_t),
                                   cudaMemcpyHostToDevice) != cudaSuccess) {
                        fail(error, error_size, "layers map H2D failed");
                        goto done;
                    }
                } else {
                    d_map = NULL;
                }
                t0 = clock(); /* placeholder; reset after warm prefetch */
                if (want_quiet) h3_cuda_set_quiet(ctx->cuda, 1);
                /* WS-first: only first two blocks so compute overlaps later H2D. */
                {
                    unsigned win = h3_w4a8_cache_window_size(ctx->cache);
                    unsigned n_pref = 2;
                    double ignore = 0.0;
                    if (win > 0 && win < n_pref) n_pref = win;
                    if (n_pref > n_blocks) n_pref = n_blocks;
                    if (!h3_dit_w4a8_prefetch(ctx, first_block, n_pref, &ignore,
                                              error, error_size)) {
                        if (want_quiet) h3_cuda_set_quiet(ctx->cuda, 0);
                        goto done;
                    }
                    if (stats && auto_prefetch) stats->prefetch_ms = ignore;
                }
                t0 = clock(); /* blocks_ms excludes initial weight unpack */
                for (b = 0; b < n_blocks; b++) {
                    unsigned block = first_block + b;
                    const h3_dit_w4a8_block_aux *aux;
                    const h3_cuda_w4a8_weight *w_qkv = NULL, *w_out = NULL,
                                                *w_fc1 = NULL, *w_fc2 = NULL;
                    unsigned win = h3_w4a8_cache_window_size(ctx->cache);
                    int cur = bank;
                    (void)win;
                    if (!prefetch_next_u1(ctx, block + 1, first_block + n_blocks,
                                          error, error_size)) {
                        if (want_quiet) h3_cuda_set_quiet(ctx->cuda, 0);
                        goto done;
                    }
                    aux = dit_aux_cached(ctx, block, error, error_size);
                    if (!aux) {
                        if (want_quiet) h3_cuda_set_quiet(ctx->cuda, 0);
                        goto done;
                    }
                    if (!project_block_adaln_metal(ctx, aux, t_row, modulation,
                                                   error, error_size)) {
                        if (want_quiet) h3_cuda_set_quiet(ctx->cuda, 0);
                        goto done;
                    }
                    if (ctx->nvfp4_on) goto have_weights_dual;
                    w_qkv = get_weight(ctx, block, H3_W4A8_LINEAR_QKV, error,
                                       error_size);
                    w_out = w_qkv
                                ? get_weight(ctx, block, H3_W4A8_LINEAR_OUT,
                                             error, error_size)
                                : NULL;
                    w_fc1 = w_out
                                ? get_weight(ctx, block, H3_W4A8_LINEAR_FC1,
                                             error, error_size)
                                : NULL;
                    w_fc2 = w_fc1
                                ? get_weight(ctx, block, H3_W4A8_LINEAR_FC2,
                                             error, error_size)
                                : NULL;
                    if (!w_fc2) {
                        if (want_quiet) h3_cuda_set_quiet(ctx->cuda, 0);
                        goto done;
                    }
                have_weights_dual:
                    if (!next_ready) {
                        cudaStream_t st = copy_stream
                                              ? (cudaStream_t)copy_stream
                                              : (cudaStream_t)0;
                        if (cudaMemcpyAsync(d_mod[cur], modulation,
                                            mod_elems * sizeof(float),
                                            cudaMemcpyHostToDevice,
                                            st) != cudaSuccess ||
                            cudaMemcpyAsync(d_n1[cur], aux->norm1,
                                            hidden * sizeof(float),
                                            cudaMemcpyHostToDevice,
                                            st) != cudaSuccess ||
                            cudaMemcpyAsync(d_n2[cur], aux->norm2,
                                            hidden * sizeof(float),
                                            cudaMemcpyHostToDevice,
                                            st) != cudaSuccess ||
                            cudaMemcpyAsync(d_qn[cur], aux->q_norm,
                                            head_dim * sizeof(float),
                                            cudaMemcpyHostToDevice,
                                            st) != cudaSuccess ||
                            cudaMemcpyAsync(d_kn[cur], aux->k_norm,
                                            head_dim * sizeof(float),
                                            cudaMemcpyHostToDevice,
                                            st) != cudaSuccess ||
                            (st && cudaStreamSynchronize(st) != cudaSuccess)) {
                            if (want_quiet) h3_cuda_set_quiet(ctx->cuda, 0);
                            fail(error, error_size, "layers aux H2D failed");
                            goto done;
                        }
                    }
                    next_ready = 0;
                    if (!block_device_compute(
                            ctx, block, w_qkv, w_out, w_fc1, w_fc2, d_x, d_y,
                            d_mod[cur], d_map, d_cos, d_sin, d_n1[cur],
                            d_n2[cur], d_qn[cur], d_kn[cur], d_a, d_b, d_qkv,
                            d_wide, seq, NULL, detail, sizeof(detail))) {
                        if (want_quiet) h3_cuda_set_quiet(ctx->cuda, 0);
                        snprintf(error, error_size, "%s", detail);
                        goto done;
                    }
                    if (b + 1 < n_blocks && copy_stream) {
                        unsigned nb = first_block + b + 1;
                        const h3_dit_w4a8_block_aux *naux =
                            dit_aux_cached(ctx, nb, detail, sizeof(detail));
                        int nb_bank = 1 - cur;
                        if (naux &&
                            project_block_adaln_metal(ctx, naux, t_row,
                                                      modulation, detail,
                                                      sizeof(detail))) {
                            cudaStream_t st = (cudaStream_t)copy_stream;
                            if (cudaMemcpyAsync(d_mod[nb_bank], modulation,
                                                mod_elems * sizeof(float),
                                                cudaMemcpyHostToDevice,
                                                st) == cudaSuccess &&
                                cudaMemcpyAsync(d_n1[nb_bank], naux->norm1,
                                                hidden * sizeof(float),
                                                cudaMemcpyHostToDevice,
                                                st) == cudaSuccess &&
                                cudaMemcpyAsync(d_n2[nb_bank], naux->norm2,
                                                hidden * sizeof(float),
                                                cudaMemcpyHostToDevice,
                                                st) == cudaSuccess &&
                                cudaMemcpyAsync(d_qn[nb_bank], naux->q_norm,
                                                head_dim * sizeof(float),
                                                cudaMemcpyHostToDevice,
                                                st) == cudaSuccess &&
                                cudaMemcpyAsync(d_kn[nb_bank], naux->k_norm,
                                                head_dim * sizeof(float),
                                                cudaMemcpyHostToDevice,
                                                st) == cudaSuccess &&
                                cudaStreamSynchronize(st) == cudaSuccess) {
                                next_ready = 1;
                                bank = nb_bank;
                            } else {
                                bank = 1 - cur;
                            }
                        } else {
                            bank = 1 - cur;
                        }
                    } else {
                        bank = 1 - cur;
                    }
                    {
                        float *tmp = d_x;
                        d_x = d_y;
                        d_y = tmp;
                    }
                }
                if (want_quiet) h3_cuda_set_quiet(ctx->cuda, 0);
                dit_lora_log_ms(ctx, stats);
                if (leave_act_dev) {
                    ctx->io_d_result = d_x;
                } else if (cudaMemcpy(y, d_x, seq * hidden * sizeof(float),
                                      cudaMemcpyDeviceToHost) != cudaSuccess) {
                    fail(error, error_size, "layers resident D2H failed");
                    goto done;
                }
                t1 = clock();
                t_all1 = clock();
                if (stats) {
                    stats->blocks_ms = elapsed_ms(t0, t1);
                    stats->total_ms = elapsed_ms(t_all0, t_all1);
                    stats->gemm_ms_sum = 0.0;
                    stats->all_finite = 1;
                    if (!leave_act_dev && y) {
                        size_t i, n = seq * hidden;
                        size_t stride = n > 4096 ? (n / 4096) : 1;
                        for (i = 0; i < n; i += stride) {
                            if (!isfinite(y[i])) {
                                stats->all_finite = 0;
                                break;
                            }
                        }
                    }
                }
                dit_aux_log_disk_loads(ctx);
                ok = 1;
                goto done;
            }
        }

        /* All-aux prestaged path (preferred; INT8 must not skip this). */
    ws_ready_allaux:
        ctx->aux_prestage_mode = 1;
        if (!h3_cuda_pregrow_dit_scratch(
                ctx->cuda, seq, hidden, H3_DIT_W4A8_HEADS, head_dim, ffn,
                detail, sizeof(detail))) {
            snprintf(error, error_size, "dit scratch: %s", detail);
            goto done;
        }
        if ((ctx->int8_store || ctx->nvfp4_store) &&
            !h3_w4a8_cache_refit_after_workspace(ctx->cache, detail,
                                                 sizeof(detail))) {
            snprintf(error, error_size, "INT8 cache refit: %s", detail);
            goto done;
        }
        base = (char *)ctx->d_block_ws;
#define H3_WS_F32(n)                                                           \
    (off += (n) * sizeof(float),                                               \
     (float *)(base + (off - (n) * sizeof(float))))
        if (use_ext_act) {
            d_x = ctx->io_d_act_a;
            d_y = ctx->io_d_act_b;
        } else {
            d_x = H3_WS_F32(seq * hidden);
            d_y = H3_WS_F32(seq * hidden);
        }
        {
            size_t map_bytes = H3_WS_MAP_BYTES(seq);
            d_map = (uint32_t *)(base + off);
            off += map_bytes;
        }
        /* RoPE lives in the block workspace (already in `need`). Extra
         * persistent cudaMalloc here OOMs on 5s/10s 24fps seq after INT8
         * cache + activations fill 24GB (dual-bank path already carves WS). */
        d_cos = H3_WS_F32(seq * rope_half);
        d_sin = H3_WS_F32(seq * rope_half);
        ctx->d_rope_valid = 0;
        d_aux_base = H3_WS_F32(n_blocks * aux_stride);
        d_a = H3_WS_F32(seq * hidden);
        d_b = H3_WS_F32(seq * hidden);
        d_qkv = H3_WS_F32(seq * inner * 3);
        d_wide = H3_WS_F32(seq * wide);
#undef H3_WS_F32

        /* Host pack: [mod | n1 | n2 | qn | kn] × n_blocks (pinned if possible). */
        {
            size_t pack_bytes = n_blocks * aux_stride * sizeof(float);
            if (cudaMallocHost((void **)&h_aux_pack, pack_bytes) ==
                cudaSuccess) {
                h_aux_pinned = 1;
            } else {
                h_aux_pack = (float *)malloc(pack_bytes);
            }
            if (!h_aux_pack) {
                fail(error, error_size, "out of memory (aux pack)");
                goto done;
            }
        }
        for (b = 0; b < n_blocks; b++) {
            unsigned block = first_block + b;
            const h3_dit_w4a8_block_aux *aux =
                dit_aux_cached(ctx, block, error, error_size);
            float *dst = h_aux_pack + b * aux_stride;
            if (!aux) goto done_dev;
            if (!project_block_adaln_metal(ctx, aux, t_row, dst, error,
                                           error_size))
                goto done_dev;
            memcpy(dst + mod_elems, aux->norm1, hidden * sizeof(float));
            memcpy(dst + mod_elems + hidden, aux->norm2,
                   hidden * sizeof(float));
            memcpy(dst + mod_elems + hidden * 2, aux->q_norm,
                   head_dim * sizeof(float));
            memcpy(dst + mod_elems + hidden * 2 + head_dim, aux->k_norm,
                   head_dim * sizeof(float));
        }

        copy_stream = h3_cuda_stream(ctx->cuda, 0);
        {
            cudaEvent_t ev_a = NULL, ev_b = NULL;
            float ms = 0.0f;
            cudaStream_t st =
                copy_stream ? (cudaStream_t)copy_stream : (cudaStream_t)0;
            if (stats) {
                cudaEventCreate(&ev_a);
                cudaEventCreate(&ev_b);
                if (ev_a) cudaEventRecord(ev_a, st);
            }
            if (!skip_act_h2d) {
                if (cudaMemcpyAsync(d_x, x, seq * hidden * sizeof(float),
                                    cudaMemcpyHostToDevice, st) != cudaSuccess) {
                    if (ev_a) cudaEventDestroy(ev_a);
                    if (ev_b) cudaEventDestroy(ev_b);
                    fail(error, error_size, "layers resident H2D failed");
                    goto done_dev;
                }
            }
            if (cudaMemcpyAsync(d_cos, rope_cos,
                                seq * rope_half * sizeof(float),
                                cudaMemcpyHostToDevice, st) != cudaSuccess ||
                cudaMemcpyAsync(d_sin, rope_sin,
                                seq * rope_half * sizeof(float),
                                cudaMemcpyHostToDevice, st) !=
                    cudaSuccess) {
                if (ev_a) cudaEventDestroy(ev_a);
                if (ev_b) cudaEventDestroy(ev_b);
                fail(error, error_size, "layers rope H2D failed");
                goto done_dev;
            }
            /* A fresh aux pack can land at a different d_block_ws offset than
             * the one baked into the step graph (e.g. TR pool/expand grew the
             * workspace in between). Invalidate when the layout moved. */
            if (ctx->step_graph_ok && d_aux_base != ctx->d_graph_aux_base)
                destroy_step_graph(ctx);
            if (cudaMemcpyAsync(d_aux_base, h_aux_pack,
                                n_blocks * aux_stride * sizeof(float),
                                cudaMemcpyHostToDevice, st) != cudaSuccess) {
                if (ev_a) cudaEventDestroy(ev_a);
                if (ev_b) cudaEventDestroy(ev_b);
                fail(error, error_size, "layers resident H2D failed");
                goto done_dev;
            }
            if (row_map) {
                if (cudaMemcpyAsync(d_map, row_map, seq * sizeof(uint32_t),
                                    cudaMemcpyHostToDevice,
                                    st) != cudaSuccess) {
                    if (ev_a) cudaEventDestroy(ev_a);
                    if (ev_b) cudaEventDestroy(ev_b);
                    fail(error, error_size, "layers map H2D failed");
                    goto done_dev;
                }
            } else {
                d_map = NULL;
            }
            if (st) {
                if (cudaStreamSynchronize(st) != cudaSuccess) {
                    if (ev_a) cudaEventDestroy(ev_a);
                    if (ev_b) cudaEventDestroy(ev_b);
                    fail(error, error_size, "layers H2D sync failed");
                    goto done_dev;
                }
            }
            /* Default-stream cudaMemcpyAsync is ordered with later default
             * kernels; cudaDeviceSynchronize would wait for other streams. */
            ctx->d_rope_valid = 0;
            if (stats && ev_a && ev_b) {
                cudaEventRecord(ev_b, st);
                cudaEventSynchronize(ev_b);
                cudaEventElapsedTime(&ms, ev_a, ev_b);
                stats->h2d_ms = (double)ms;
            }
            if (ev_a) cudaEventDestroy(ev_a);
            if (ev_b) cudaEventDestroy(ev_b);
        }

        /* P2b whole-step CUDA Graph: first eligible call eager-runs the block
         * loop below to settle cuBLASLt plans / Hadamard / act-scratch sizes,
         * the next call captures, subsequent same-shape calls replay. The
         * captured region contains kernel launches only — all cross-step
         * inputs (act, modulation, rope, map) live in fixed device buffers
         * that are re-uploaded above before the replay. */
        step_graph_note(0, "off");
        if (env_flag_is_one("H3_DIT_W4A8_CUDA_GRAPH")) {
            const char *block_reason =
                step_graph_block_reason(ctx, n_blocks);
            if (block_reason) {
                step_graph_note(1, block_reason);
            } else if (!use_ext_act || !ctx->io_d_act_a || !ctx->io_d_act_b) {
                /* The graph bakes d_x/d_y = io act buffers; only the resident
                 * forward pins those (WS-carved d_x could move on realloc). */
                step_graph_note(1, "no_resident_act");
            } else if (!all_blocks_resident(ctx, first_block, n_blocks)) {
                step_graph_note(1, "weights_not_resident");
            } else if (!(ctx->step_graph_ok && ctx->step_graph_exec &&
                         ctx->step_graph_seq == seq &&
                         ctx->step_graph_first == first_block &&
                         ctx->step_graph_n == n_blocks &&
                         ctx->d_graph_aux_base == d_aux_base)) {
                cudaError_t gst;
                cudaGraph_t g = NULL;
                destroy_step_graph(ctx);
                (void)cudaGetLastError();
                h3_cuda_set_graph_capture(ctx->cuda, 1);
                gst = cudaStreamBeginCapture(0, cudaStreamCaptureModeRelaxed);
                if (gst == cudaSuccess) {
                    int cok = run_all_blocks_device(
                        ctx, first_block, n_blocks, d_x, d_y, d_aux_base,
                        aux_stride, mod_elems, d_map, d_cos, d_sin, d_a, d_b,
                        d_qkv, d_wide, seq, detail, sizeof(detail));
                    gst = cudaStreamEndCapture(0, &g);
                    h3_cuda_set_graph_capture(ctx->cuda, 0);
                    if (cok && gst == cudaSuccess && g) {
                        cudaGraphExec_t ge = NULL;
                        if (cudaGraphInstantiate(&ge, g, 0) == cudaSuccess) {
                            ctx->step_graph = g;
                            ctx->step_graph_exec = ge;
                            ctx->step_graph_seq = seq;
                            ctx->step_graph_first = first_block;
                            ctx->step_graph_n = n_blocks;
                            ctx->d_graph_aux_base = d_aux_base;
                            ctx->step_graph_ok = 1;
                            step_graph_note(2, "capture");
                        } else {
                            cudaGraphDestroy(g);
                            step_graph_note(1, "instantiate_fail");
                        }
                    } else {
                        if (g) cudaGraphDestroy(g);
                        step_graph_note(1, "capture_fail");
                    }
                    (void)cudaGetLastError();
                } else {
                    h3_cuda_set_graph_capture(ctx->cuda, 0);
                    (void)cudaGetLastError();
                    step_graph_note(1, "begin_capture_fail");
                }
                /* Capture executed the block chain once with current inputs.
                 * On failure the activation ping-pong may have advanced —
                 * restore it so the eager fallback below sees pristine d_x. */
                if (!ctx->step_graph_ok) {
                    float *in_act =
                        (n_blocks & 1u) ? ctx->io_d_act_b : ctx->io_d_act_a;
                    if (use_ext_act && in_act != ctx->io_d_act_a) {
                        if (cudaMemcpy(ctx->io_d_act_a, in_act,
                                       seq * hidden * sizeof(float),
                                       cudaMemcpyDeviceToDevice) !=
                            cudaSuccess) {
                            fail(error, error_size,
                                 "step graph fallback act restore failed");
                            goto done_dev;
                        }
                    }
                }
            } else {
                step_graph_note(2, "replay");
            }
        }

        if (ctx->step_graph_ok && ctx->step_graph_exec &&
            ctx->step_graph_seq == seq &&
            ctx->step_graph_first == first_block &&
            ctx->step_graph_n == n_blocks &&
            ctx->d_graph_aux_base == d_aux_base) {
            if (cudaGraphLaunch(ctx->step_graph_exec, 0) != cudaSuccess) {
                fail(error, error_size, "step cudaGraphLaunch failed");
                goto done_dev;
            }
            /* Result buffer after n_blocks ping-pong swaps (see loop below). */
            d_x = (n_blocks & 1u) ? ctx->io_d_act_b : ctx->io_d_act_a;
            if (want_quiet) h3_cuda_set_quiet(ctx->cuda, 1);
            goto blocks_chain_done;
        }

        t0 = clock(); /* placeholder; reset after warm prefetch */
        if (want_quiet) h3_cuda_set_quiet(ctx->cuda, 1);
        /* WS-first: warm 2 blocks; remaining H2D overlaps compute (U=1). */
        {
            unsigned win = h3_w4a8_cache_window_size(ctx->cache);
            unsigned n_pref = 2;
            double unpack_ms = 0.0;
            clock_t tp0 = clock(), tp1;
            if (win > 0 && win < n_pref) n_pref = win;
            if (n_pref > n_blocks) n_pref = n_blocks;
            if (!h3_dit_w4a8_prefetch(ctx, first_block, n_pref, &unpack_ms, error,
                                      error_size)) {
                if (want_quiet) h3_cuda_set_quiet(ctx->cuda, 0);
                goto done_dev;
            }
            tp1 = clock();
            if (stats) {
                /* Coarse: host wait for warm unpack (API ms if set, else wall). */
                stats->prefetch_wait_ms =
                    unpack_ms > 0.0 ? unpack_ms : elapsed_ms(tp0, tp1);
                if (auto_prefetch) stats->prefetch_ms = stats->prefetch_wait_ms;
            }
        }
        t0 = clock(); /* blocks_ms excludes initial weight unpack */
        {
            cudaEvent_t ev_c0 = NULL, ev_c1 = NULL;
            if (stats) {
                cudaEventCreate(&ev_c0);
                cudaEventCreate(&ev_c1);
                if (ev_c0) cudaEventRecord(ev_c0, 0);
            }
        for (b = 0; b < n_blocks; b++) {
            unsigned block = first_block + b;
            const h3_cuda_w4a8_weight *w_qkv = NULL, *w_out = NULL,
                                        *w_fc1 = NULL, *w_fc2 = NULL;
            float *layer = d_aux_base + b * aux_stride;
            float *d_mod = layer;
            float *d_n1 = layer + mod_elems;
            float *d_n2 = d_n1 + hidden;
            float *d_qn = d_n2 + hidden;
            float *d_kn = d_qn + head_dim;

            /* Next-block H2D on the copy stream; GEMM wait_ready fences. */
            if (!prefetch_next_u1(ctx, block + 1, first_block + n_blocks,
                                  error, error_size)) {
                if (want_quiet) h3_cuda_set_quiet(ctx->cuda, 0);
                goto done_dev;
            }

            if (!ctx->nvfp4_on) {
                w_qkv = get_weight(ctx, block, H3_W4A8_LINEAR_QKV, error,
                                   error_size);
                w_out = w_qkv ? get_weight(ctx, block, H3_W4A8_LINEAR_OUT,
                                           error, error_size)
                              : NULL;
                w_fc1 = w_out ? get_weight(ctx, block, H3_W4A8_LINEAR_FC1,
                                           error, error_size)
                              : NULL;
                w_fc2 = w_fc1 ? get_weight(ctx, block, H3_W4A8_LINEAR_FC2,
                                           error, error_size)
                              : NULL;
                if (!w_fc2) {
                    if (want_quiet) h3_cuda_set_quiet(ctx->cuda, 0);
                    goto done_dev;
                }
            }

            if (!block_device_compute(ctx, block, w_qkv, w_out, w_fc1, w_fc2, d_x, d_y,
                                      d_mod, d_map, d_cos, d_sin, d_n1, d_n2,
                                      d_qn, d_kn, d_a, d_b, d_qkv, d_wide, seq,
                                      NULL, detail, sizeof(detail))) {
                if (want_quiet) h3_cuda_set_quiet(ctx->cuda, 0);
                snprintf(error, error_size, "%s", detail);
                goto done_dev;
            }

            {
                float *tmp = d_x;
                d_x = d_y;
                d_y = tmp;
            }
        }
            if (stats && ev_c0 && ev_c1) {
                float ms = 0.0f;
                cudaEventRecord(ev_c1, 0);
                cudaEventSynchronize(ev_c1);
                cudaEventElapsedTime(&ms, ev_c0, ev_c1);
                stats->compute_ms = (double)ms;
            }
            if (ev_c0) cudaEventDestroy(ev_c0);
            if (ev_c1) cudaEventDestroy(ev_c1);
        }
        if (want_quiet) h3_cuda_set_quiet(ctx->cuda, 0);
    blocks_chain_done:
        dit_lora_log_ms(ctx, stats);
        /* After loop, result is in d_x (swapped). */
        {
            cudaEvent_t ev_d0 = NULL, ev_d1 = NULL;
            if (leave_act_dev) {
                ctx->io_d_result = d_x;
            } else {
                if (stats) {
                    cudaEventCreate(&ev_d0);
                    cudaEventCreate(&ev_d1);
                    if (ev_d0) cudaEventRecord(ev_d0, 0);
                }
                if (cudaMemcpy(y, d_x, seq * hidden * sizeof(float),
                               cudaMemcpyDeviceToHost) != cudaSuccess) {
                    if (ev_d0) cudaEventDestroy(ev_d0);
                    if (ev_d1) cudaEventDestroy(ev_d1);
                    fail(error, error_size, "layers resident D2H failed");
                    goto done_dev;
                }
                if (stats && ev_d0 && ev_d1) {
                    float ms = 0.0f;
                    cudaEventRecord(ev_d1, 0);
                    cudaEventSynchronize(ev_d1);
                    cudaEventElapsedTime(&ms, ev_d0, ev_d1);
                    stats->d2h_ms = (double)ms;
                }
                if (ev_d0) cudaEventDestroy(ev_d0);
                if (ev_d1) cudaEventDestroy(ev_d1);
            }
        }
        t1 = clock();
        t_all1 = clock();
        if (stats) {
            stats->blocks_ms = elapsed_ms(t0, t1);
            stats->total_ms = elapsed_ms(t_all0, t_all1);
            stats->gemm_ms_sum = 0.0;
            /* Sampled finite check — full 80M scan was ~80ms on host. */
            stats->all_finite = 1;
            if (!leave_act_dev && y) {
                size_t i, n = seq * hidden;
                size_t stride = n > 4096 ? (n / 4096) : 1;
                for (i = 0; i < n; i += stride) {
                    if (!isfinite(y[i])) {
                        stats->all_finite = 0;
                        break;
                    }
                }
            }
        }
        dit_aux_log_disk_loads(ctx);
        ok = 1;
    done_dev:
        if (h_aux_pack) {
            if (h_aux_pinned)
                cudaFreeHost(h_aux_pack);
            else
                free(h_aux_pack);
        }
        goto done;
    }

    buf_a = (float *)malloc(seq * H3_DIT_W4A8_HIDDEN * sizeof(float));
    buf_b = (float *)malloc(seq * H3_DIT_W4A8_HIDDEN * sizeof(float));
    modulation = (float *)malloc(dit_adaln_mod_elems(ctx) * sizeof(float));
    if (!buf_a || !buf_b || !modulation) {
        fail(error, error_size, "out of memory (layers_forward)");
        goto done;
    }
    memcpy(buf_a, x, seq * H3_DIT_W4A8_HIDDEN * sizeof(float));
    cur = buf_a;
    nxt = buf_b;

    t0 = clock();
    for (b = 0; b < n_blocks; b++) {
        unsigned block = first_block + b;
        const h3_dit_w4a8_block_aux *aux;
        h3_dit_w4a8_block_stats bst;
        /* Multi-block prefetch: keep next block warm while current runs. */
        if (b + 1 < n_blocks) {
            double ignore = 0.0;
            if (!h3_dit_w4a8_prefetch(ctx, block + 1, 1, &ignore, error,
                                      error_size))
                goto done;
        }
        aux = dit_aux_cached(ctx, block, error, error_size);
        if (!aux) goto done;
        if (!project_block_adaln_metal(ctx, aux, t_row, modulation, error,
                                       error_size))
            goto done;
        memset(&bst, 0, sizeof(bst));
        if (!h3_dit_w4a8_block_forward_f32(ctx, block, aux, cur, seq,
                                           modulation, row_map, rope_cos,
                                           rope_sin, nxt, &bst, error,
                                           error_size))
            goto done;
        gemm_sum += bst.qkv_ms + bst.out_ms + bst.fc1_ms + bst.fc2_ms;
        {
            const float *tmp = cur;
            cur = nxt;
            nxt = (float *)tmp;
        }
    }
    t1 = clock();
    memcpy(y, cur, seq * H3_DIT_W4A8_HIDDEN * sizeof(float));
    t_all1 = clock();

    if (stats) {
        stats->blocks_ms = elapsed_ms(t0, t1);
        stats->total_ms = elapsed_ms(t_all0, t_all1);
        stats->gemm_ms_sum = gemm_sum;
        stats->all_finite = 1;
        {
            size_t i, n = seq * H3_DIT_W4A8_HIDDEN;
            for (i = 0; i < n; i++) {
                if (!isfinite(y[i])) {
                    stats->all_finite = 0;
                    break;
                }
            }
        }
    }
    dit_aux_log_disk_loads(ctx);
    dit_lora_log_ms(ctx, stats);
    ok = 1;
done:
    if (ctx && ctx->lora_ev_n > 0)
        ctx->lora_ev_n = 0;
    if (ok && ctx && ctx->cache) h3_w4a8_cache_freeze_window(ctx->cache);
    free(buf_a);
    free(buf_b);
    free(modulation);
    return ok;
}

int h3_dit_w4a8_project_text_f32(const h3_dit_w4a8_core *core,
                                 const float *text_emb, size_t tokens,
                                 float *text_hidden, char *error,
                                 size_t error_size) {
    if (!core || !core->condition_w || !core->condition_b || !text_emb ||
        !text_hidden || tokens == 0) {
        fail(error, error_size, "invalid project_text args / missing "
                                "condition_proj");
        return 0;
    }
    return h3_dit_w4a8_dense_f32(text_emb, core->condition_w, core->condition_b,
                                 tokens, H3_DIT_W4A8_HIDDEN,
                                 H3_DIT_W4A8_TEXT_DIM, text_hidden, error,
                                 error_size);
}

static int refine_dense(h3_dit_w4a8 *ctx, const float *x, const float *weight,
                        const float *bias, size_t m, size_t n, size_t k,
                        float *y, char *error, size_t error_size) {
    size_t r, o;
    if (!ctx || !ctx->cuda || env_flag_is_one("H3_DIT_W4A8_HOST_DENSE"))
        return h3_dit_w4a8_dense_f32(x, weight, bias, m, n, k, y, error,
                                     error_size);
    if (!h3_cuda_gemm_f32(ctx->cuda, x, weight, m, n, k, y, NULL, error,
                          error_size))
        return 0;
    if (bias) {
        for (r = 0; r < m; r++) {
            float *yr = y + r * n;
            for (o = 0; o < n; o++) yr[o] += bias[o];
        }
    }
    return 1;
}

static int run_refiner_block_f32(h3_dit_w4a8 *ctx,
                                 const h3_dit_w4a8_refiner_block *rb,
                                 unsigned rbi, float *hidden, size_t tokens, float *norm,
                                 float *qkv, float *query, float *key,
                                 float *value, float *heads, float *branch,
                                 float *fc1, float *act, char *error,
                                 size_t error_size) {
    const size_t hidden_w = H3_DIT_W4A8_HIDDEN;
    const size_t inner = H3_DIT_W4A8_INNER;
    const size_t ffn = H3_DIT_W4A8_FFN;
    const size_t heads_n = H3_DIT_W4A8_HEADS;
    const size_t head_dim = H3_DIT_W4A8_HEAD_DIM;
    const float eps = 1e-5f;
    size_t i, n;
    double ignore = 0.0;

    h3_dit_w4a8_rms_norm_f32(hidden, rb->norm1, tokens, hidden_w, eps, norm);
    if (!refine_dense(ctx, norm, rb->qkv_w, NULL, tokens, inner * 3, hidden_w,
                      qkv, error, error_size))
        return 0;
    if (ctx && ctx->lora && ctx->lora->n_refiner_pairs > 0 &&
        rbi < (unsigned)H3_LORA_N_REFINER) {
        if (!h3_lora_apply_f32(&ctx->lora->refiner[rbi].qkv, ctx->lora->scale,
                               norm, qkv, tokens, error, error_size))
            return 0;
    }
    /* Comfy TokenRefiner Attention: contiguous QKV + Q/K RMS, no RoPE. */
    if (!h3_dit_w4a8_qkv_rope_f32(qkv, rb->q_norm, rb->k_norm, NULL, NULL,
                                  tokens, heads_n, head_dim, 0, eps, query,
                                  key, value, error, error_size))
        return 0;
    if (ctx && ctx->cuda) {
        if (!h3_cuda_sdpa_f32(ctx->cuda, query, key, value, tokens, heads_n,
                              head_dim, 1.0f / sqrtf((float)head_dim), heads,
                              &ignore, error, error_size))
            return 0;
    } else if (!h3_dit_w4a8_sdpa_f32(query, key, value, tokens, heads_n,
                                    head_dim, 1.0f / sqrtf((float)head_dim),
                                    heads, error, error_size)) {
        return 0;
    }
    if (!refine_dense(ctx, heads, rb->out_w, NULL, tokens, hidden_w, inner,
                      branch, error, error_size))
        return 0;
    if (ctx && ctx->lora && ctx->lora->n_refiner_pairs > 0 &&
        rbi < (unsigned)H3_LORA_N_REFINER) {
        if (!h3_lora_apply_f32(&ctx->lora->refiner[rbi].out, ctx->lora->scale,
                               heads, branch, tokens, error, error_size))
            return 0;
    }
    n = tokens * hidden_w;
    for (i = 0; i < n; i++) hidden[i] += branch[i];

    h3_dit_w4a8_rms_norm_f32(hidden, rb->norm2, tokens, hidden_w, eps, norm);
    if (!refine_dense(ctx, norm, rb->fc1_w, NULL, tokens, ffn * 2, hidden_w,
                      fc1, error, error_size))
        return 0;
    if (ctx && ctx->lora && ctx->lora->n_refiner_pairs > 0 &&
        rbi < (unsigned)H3_LORA_N_REFINER) {
        if (!h3_lora_apply_f32(&ctx->lora->refiner[rbi].fc1, ctx->lora->scale,
                               norm, fc1, tokens, error, error_size))
            return 0;
    }
    if (ctx && ctx->cuda) {
        if (!h3_cuda_swiglu_f32(ctx->cuda, fc1, tokens, ffn, act, &ignore,
                                error, error_size))
            return 0;
    } else {
        h3_dit_w4a8_swiglu_f32(fc1, tokens, ffn, act);
    }
    if (!refine_dense(ctx, act, rb->fc2_w, NULL, tokens, hidden_w, ffn, branch,
                      error, error_size))
        return 0;
    if (ctx && ctx->lora && ctx->lora->n_refiner_pairs > 0 &&
        rbi < (unsigned)H3_LORA_N_REFINER) {
        if (!h3_lora_apply_f32(&ctx->lora->refiner[rbi].fc2, ctx->lora->scale,
                               act, branch, tokens, error, error_size))
            return 0;
    }
    for (i = 0; i < n; i++) hidden[i] += branch[i];
    return 1;
}

int h3_dit_w4a8_refine_text_f32(h3_dit_w4a8 *ctx, const h3_dit_w4a8_core *core,
                                const float *text_emb, size_t tokens,
                                float *text_hidden, char *error,
                                size_t error_size) {
    float *norm = NULL, *qkv = NULL, *query = NULL, *key = NULL, *value = NULL;
    float *heads = NULL, *branch = NULL, *fc1 = NULL, *act = NULL;
    const size_t hidden_w = H3_DIT_W4A8_HIDDEN;
    const size_t inner = H3_DIT_W4A8_INNER;
    const size_t ffn = H3_DIT_W4A8_FFN;
    int ok = 0;
    unsigned bi;

    if (!core || !core->condition_w || !core->condition_b || !text_emb ||
        !text_hidden || tokens == 0) {
        fail(error, error_size, "invalid refine_text args / missing "
                                "condition_proj");
        return 0;
    }
    if (!refine_dense(ctx, text_emb, core->condition_w, core->condition_b,
                      tokens, hidden_w, H3_DIT_W4A8_TEXT_DIM, text_hidden,
                      error, error_size))
        return 0;
    if (!core->has_refiner || env_flag_is_one("H3_DIT_NO_REFINER"))
        return 1;

    norm = (float *)malloc(tokens * hidden_w * sizeof(float));
    qkv = (float *)malloc(tokens * inner * 3 * sizeof(float));
    query = (float *)malloc(tokens * inner * sizeof(float));
    key = (float *)malloc(tokens * inner * sizeof(float));
    value = (float *)malloc(tokens * inner * sizeof(float));
    heads = (float *)malloc(tokens * inner * sizeof(float));
    branch = (float *)malloc(tokens * hidden_w * sizeof(float));
    fc1 = (float *)malloc(tokens * ffn * 2 * sizeof(float));
    act = (float *)malloc(tokens * ffn * sizeof(float));
    if (!norm || !qkv || !query || !key || !value || !heads || !branch ||
        !fc1 || !act) {
        fail(error, error_size, "OOM token_refiner workspace");
        goto done;
    }

    for (bi = 0; bi < 2; bi++) {
        if (!run_refiner_block_f32(ctx, &core->refiner[bi], bi, text_hidden, tokens,
                                   norm, qkv, query, key, value, heads, branch,
                                   fc1, act, error, error_size))
            goto done;
    }
    h3_dit_w4a8_rms_norm_f32(text_hidden, core->refiner_final_norm, tokens,
                             hidden_w, 1e-5f, text_hidden);
    ok = 1;
done:
    free(norm);
    free(qkv);
    free(query);
    free(key);
    free(value);
    free(heads);
    free(branch);
    free(fc1);
    free(act);
    return ok;
}

/* --- Metal-compatible horizontal token reduction (host F32 pool/expand) --- */

typedef struct {
    uint32_t video_start;
    uint32_t video_rows;
    uint32_t audio_rows;
    uint32_t full_seq;
    uint32_t spatial_w;
    uint32_t spatial_h;
    uint32_t latent_t;
    uint32_t reduced_video;
    uint32_t reduced_seq;
    uint32_t baseline_rows;
    uint32_t *pairs;
    uint32_t *baseline_idx;
    uint32_t *parents;
} h3_tr_maps;

static void tr_maps_free(h3_tr_maps *m) {
    if (!m) return;
    free(m->pairs);
    free(m->baseline_idx);
    free(m->parents);
    memset(m, 0, sizeof(*m));
}

static void tr_pool_sources(const h3_tr_maps *m, uint32_t reduced_row,
                            uint32_t *first, uint32_t *second) {
    uint32_t local, source, reduced_w;
    if (reduced_row < m->video_start) {
        *first = reduced_row;
        *second = reduced_row;
        return;
    }
    if (reduced_row >= m->video_start + m->reduced_video) {
        /* audio suffix: identity from full audio region */
        uint32_t a = reduced_row - (m->video_start + m->reduced_video);
        *first = m->video_start + m->video_rows + a;
        *second = *first;
        return;
    }
    reduced_w = (m->spatial_w + 1u) / 2u;
    local = reduced_row - m->video_start;
    source = m->video_start + (local / reduced_w) * m->spatial_w +
             (local % reduced_w) * 2u;
    *first = source;
    *second = source +
              (((source - m->video_start) % m->spatial_w + 1u < m->spatial_w)
                   ? 1u
                   : 0u);
}

static uint32_t tr_parent(const h3_tr_maps *m, uint32_t full_row) {
    uint32_t reduced_w, local;
    if (full_row < m->video_start) return full_row;
    if (full_row >= m->video_start + m->video_rows) {
        uint32_t a = full_row - (m->video_start + m->video_rows);
        return m->video_start + m->reduced_video + a;
    }
    reduced_w = (m->spatial_w + 1u) / 2u;
    local = full_row - m->video_start;
    return m->video_start + (local / m->spatial_w) * reduced_w +
           (local % m->spatial_w) / 2u;
}

static int tr_build_maps(h3_tr_maps *m, uint32_t video_start, uint32_t video_rows,
                         uint32_t audio_rows, uint32_t spatial_w,
                         uint32_t spatial_h, uint32_t latent_t, char *error,
                         size_t error_size) {
    uint32_t reduced_w, row, baseline_row = 0;
    uint64_t reduced_video;
    memset(m, 0, sizeof(*m));
    if (!spatial_w || !spatial_h || !latent_t ||
        (uint64_t)latent_t * spatial_h * spatial_w != video_rows) {
        fail(error, error_size, "token-reduction spatial dims mismatch");
        return 0;
    }
    reduced_w = (spatial_w + 1u) / 2u;
    reduced_video = (uint64_t)latent_t * spatial_h * reduced_w;
    if (reduced_video > UINT32_MAX) {
        fail(error, error_size, "token-reduction reduced_video overflow");
        return 0;
    }
    m->video_start = video_start;
    m->video_rows = video_rows;
    m->audio_rows = audio_rows;
    m->full_seq = video_start + video_rows + audio_rows;
    m->spatial_w = spatial_w;
    m->spatial_h = spatial_h;
    m->latent_t = latent_t;
    m->reduced_video = (uint32_t)reduced_video;
    m->reduced_seq = video_start + m->reduced_video + audio_rows;
    m->baseline_rows = video_rows - m->reduced_video;
    m->pairs = (uint32_t *)malloc((size_t)m->reduced_seq * 2u * sizeof(uint32_t));
    m->baseline_idx =
        (uint32_t *)malloc((size_t)m->reduced_seq * sizeof(uint32_t));
    m->parents = (uint32_t *)malloc((size_t)m->full_seq * sizeof(uint32_t));
    if (!m->pairs || !m->baseline_idx || !m->parents) {
        tr_maps_free(m);
        fail(error, error_size, "OOM token-reduction maps");
        return 0;
    }
    for (row = 0; row < m->reduced_seq; row++) {
        tr_pool_sources(m, row, &m->pairs[(size_t)row * 2],
                        &m->pairs[(size_t)row * 2 + 1]);
        if (row >= m->video_start && row < m->video_start + m->reduced_video &&
            m->pairs[(size_t)row * 2] != m->pairs[(size_t)row * 2 + 1])
            m->baseline_idx[row] = baseline_row++;
        else
            m->baseline_idx[row] = UINT32_MAX;
    }
    if (baseline_row != m->baseline_rows) {
        tr_maps_free(m);
        fail(error, error_size, "token-reduction baseline map inconsistent");
        return 0;
    }
    for (row = 0; row < m->full_seq; row++) m->parents[row] = tr_parent(m, row);
    return 1;
}

static int tr_infer_spatial(const float *pos_t, const float *pos_h,
                            const float *pos_w, size_t video_seq,
                            uint32_t *out_t, uint32_t *out_h, uint32_t *out_w) {
    size_t i;
    int mt = 0, mh = 0, mw = 0;
    if (!pos_t || !pos_h || !pos_w || !out_t || !out_h || !out_w ||
        video_seq == 0)
        return 0;
    for (i = 0; i < video_seq; i++) {
        /* Token-reduction only supports the old integer patch grid. Metal
         * float layout positions are incompatible — keep TR off. */
        if (pos_t[i] < 0.0f || pos_h[i] < 0.0f || pos_w[i] < 0.0f ||
            fabsf(pos_t[i] - roundf(pos_t[i])) > 1e-3f ||
            fabsf(pos_h[i] - roundf(pos_h[i])) > 1e-3f ||
            fabsf(pos_w[i] - roundf(pos_w[i])) > 1e-3f)
            return 0;
        if ((int)lroundf(pos_t[i]) > mt) mt = (int)lroundf(pos_t[i]);
        if ((int)lroundf(pos_h[i]) > mh) mh = (int)lroundf(pos_h[i]);
        if ((int)lroundf(pos_w[i]) > mw) mw = (int)lroundf(pos_w[i]);
    }
    *out_t = (uint32_t)mt + 1u;
    *out_h = (uint32_t)mh + 1u;
    *out_w = (uint32_t)mw + 1u;
    return (uint64_t)(*out_t) * (*out_h) * (*out_w) == video_seq;
}

static void tr_pool_f32(const h3_tr_maps *m, const float *input, float *output,
                        float *original, float *baseline) {
    const size_t W = H3_DIT_W4A8_HIDDEN;
    uint32_t row, c;
    for (row = 0; row < m->reduced_seq; row++) {
        uint32_t first = m->pairs[(size_t)row * 2];
        uint32_t second = m->pairs[(size_t)row * 2 + 1];
        const float *a = input + (size_t)first * W;
        float *out = output + (size_t)row * W;
        float *orig_a = original + (size_t)first * W;
        for (c = 0; c < W; c++) {
            orig_a[c] = a[c];
            out[c] = a[c];
        }
        if (first != second) {
            const float *b = input + (size_t)second * W;
            float *orig_b = original + (size_t)second * W;
            for (c = 0; c < W; c++) {
                orig_b[c] = b[c];
                out[c] = 0.5f * (a[c] + b[c]);
            }
        }
        if (m->baseline_idx[row] != UINT32_MAX) {
            float *base = baseline + (size_t)m->baseline_idx[row] * W;
            for (c = 0; c < W; c++) base[c] = out[c];
        }
    }
}

static void tr_expand_f32(const h3_tr_maps *m, const float *original,
                          const float *reduced, const float *baseline,
                          float *output, float scale) {
    const size_t W = H3_DIT_W4A8_HIDDEN;
    uint32_t row, c;
    for (row = 0; row < m->full_seq; row++) {
        uint32_t parent = m->parents[row];
        const float *red = reduced + (size_t)parent * W;
        float *out = output + (size_t)row * W;
        if (row < m->video_start ||
            row >= m->video_start + m->video_rows ||
            m->baseline_idx[parent] == UINT32_MAX) {
            for (c = 0; c < W; c++) out[c] = red[c];
        } else {
            const float *orig = original + (size_t)row * W;
            const float *base =
                baseline + (size_t)m->baseline_idx[parent] * W;
            for (c = 0; c < W; c++)
                out[c] = orig[c] + scale * (red[c] - base[c]);
        }
    }
}

static int tr_build_reduced_rope(const h3_tr_maps *m, const float *inv_freq,
                                 const float *full_pt, const float *full_ph,
                                 const float *full_pw, float spatial_scale,
                                 float *cos_out, float *sin_out, char *error,
                                 size_t error_size) {
    float *pt = NULL, *ph = NULL, *pw = NULL;
    uint32_t row;
    int ok = 0;
    pt = (float *)malloc((size_t)m->reduced_seq * sizeof(float));
    ph = (float *)malloc((size_t)m->reduced_seq * sizeof(float));
    pw = (float *)malloc((size_t)m->reduced_seq * sizeof(float));
    if (!pt || !ph || !pw) {
        fail(error, error_size, "OOM reduced rope pos");
        goto done;
    }
    for (row = 0; row < m->reduced_seq; row++) {
        uint32_t first, second;
        tr_pool_sources(m, row, &first, &second);
        pt[row] = 0.5f * (full_pt[first] + full_pt[second]);
        ph[row] = 0.5f * (full_ph[first] + full_ph[second]);
        pw[row] = 0.5f * (full_pw[first] + full_pw[second]);
    }
    ok = h3_dit_w4a8_rope_tables_f32(inv_freq, m->reduced_seq, pt, ph, pw,
                                     spatial_scale, cos_out, sin_out, error,
                                     error_size);
done:
    free(pt);
    free(ph);
    free(pw);
    return ok;
}

/* Transient layers IO helpers for end-to-end device-resident forward. */
static void layers_io_set(h3_dit_w4a8 *ctx, float *a, float *b, int flags) {
    if (!ctx) return;
    ctx->io_d_act_a = a;
    ctx->io_d_act_b = b;
    ctx->io_flags = flags;
    ctx->io_d_result = NULL;
}

static void layers_io_clear(h3_dit_w4a8 *ctx) {
    if (!ctx) return;
    ctx->io_d_act_a = NULL;
    ctx->io_d_act_b = NULL;
    ctx->io_flags = 0;
    ctx->io_d_result = NULL;
}

/* After leave-device layers, make the result buffer the next input (act_a). */
static void layers_io_promote_result(h3_dit_w4a8 *ctx) {
    float *res, *other;
    if (!ctx || !ctx->io_d_result) return;
    res = ctx->io_d_result;
    other = (res == ctx->io_d_act_a) ? ctx->io_d_act_b : ctx->io_d_act_a;
    ctx->io_d_act_a = res;
    ctx->io_d_act_b = other;
    ctx->io_d_result = NULL;
}

/* Split layers around token-reduction window (Metal begin/end semantics).
 * Pool/expand default to CUDA (original/baseline stay on device); set
 * H3_DIT_W4A8_HOST_TOKEN_TR=1 to force host F32 loops.
 * resident≠0: activations already on device via layers_io_set; no host hidden. */
static int layers_with_token_reduction(
    h3_dit_w4a8 *ctx, const h3_dit_w4a8_core *core, unsigned first_block,
    unsigned n_blocks, size_t t_row, float *hidden, float *hidden2,
    size_t text_seq, size_t cond_seq, size_t audio_cond_seq, size_t video_seq,
    size_t audio_seq, const uint32_t *row_map, const float *cos,
    const float *sin, const float *full_pt, const float *full_ph,
    const float *full_pw, float spatial_scale, int auto_prefetch, int resident,
    h3_dit_w4a8_layers_stats *lst, char *error, size_t error_size) {
    h3_tr_maps maps;
    float *reduced = NULL, *reduced2 = NULL, *original = NULL, *baseline = NULL;
    float *d_original = NULL, *d_baseline = NULL;
    float *rcos = NULL, *rsin = NULL;
    uint32_t *rmap = NULL;
    uint32_t spat_t = 0, spat_h = 0, spat_w = 0;
    unsigned begin, end, last;
    unsigned prefix_n, mid_n, suffix_n;
    double layers_ms = 0.0;
    int use_cuda_tr;
    int use_resident_tr;
    int ok = 0;
    size_t full_seq = text_seq + cond_seq + audio_cond_seq + audio_seq + video_seq;
    size_t i;
    float *d_full_a = NULL, *d_full_b = NULL;

    memset(&maps, 0, sizeof(maps));
    if (!ctx->token_reduction) {
        if (resident) {
            /* io_* already bound by caller */
            ok = h3_dit_w4a8_layers_forward_f32(
                ctx, first_block, n_blocks, t_row, NULL, full_seq, row_map, cos,
                sin, NULL, auto_prefetch, lst, error, error_size);
            return ok;
        }
        return h3_dit_w4a8_layers_forward_f32(
            ctx, first_block, n_blocks, t_row, hidden, full_seq, row_map, cos,
            sin, hidden2, auto_prefetch, lst, error, error_size);
    }
    /* Packed order is text | cond | audio_cond | audio | video. Infer the
     * spatial grid from target video only (after video_off). cond/audio_cond
     * are unpoolable prefixes. */
    {
        size_t video_off = text_seq + cond_seq + audio_cond_seq + audio_seq;
        if (!tr_infer_spatial(full_pt + video_off, full_ph + video_off,
                              full_pw + video_off, video_seq, &spat_t, &spat_h,
                              &spat_w)) {
            fail(error, error_size,
                 "token-reduction: cannot infer spatial grid");
            return 0;
        }
    }
    /* TR maps assume [prefix | video | audio_suffix]. Fold cond+audio into
     * the prefix and pass audio_rows=0. */
    if (!tr_build_maps(&maps, (uint32_t)(text_seq + cond_seq + audio_seq),
                       (uint32_t)video_seq, /*audio_suffix*/ 0u, spat_w, spat_h,
                       spat_t, error, error_size))
        return 0;

    begin = ctx->tr_begin;
    end = ctx->tr_end;
    if (ctx->tr_early_steps && (unsigned)ctx->denoise_step < ctx->tr_early_steps)
        end = ctx->tr_early_end;
    last = first_block + n_blocks;
    if (begin < first_block) begin = first_block;
    if (end > last) end = last;
    if (begin >= end || begin >= last) {
        if (resident) {
            ok = h3_dit_w4a8_layers_forward_f32(
                ctx, first_block, n_blocks, t_row, NULL, full_seq, row_map, cos,
                sin, NULL, auto_prefetch, lst, error, error_size);
        } else {
            ok = h3_dit_w4a8_layers_forward_f32(
                ctx, first_block, n_blocks, t_row, hidden, full_seq, row_map, cos,
                sin, hidden2, auto_prefetch, lst, error, error_size);
        }
        tr_maps_free(&maps);
        return ok;
    }

    prefix_n = begin > first_block ? begin - first_block : 0;
    mid_n = end - begin;
    suffix_n = last > end ? last - end : 0;
    use_cuda_tr = ctx->cuda && !env_flag_is_one("H3_DIT_W4A8_HOST_TOKEN_TR");
    use_resident_tr = resident && use_cuda_tr;

    if (!use_resident_tr) {
        reduced = (float *)malloc((size_t)maps.reduced_seq * H3_DIT_W4A8_HIDDEN *
                                  sizeof(float));
        reduced2 = (float *)malloc((size_t)maps.reduced_seq * H3_DIT_W4A8_HIDDEN *
                                   sizeof(float));
        if (!use_cuda_tr) {
            original =
                (float *)malloc(full_seq * H3_DIT_W4A8_HIDDEN * sizeof(float));
            baseline =
                (float *)malloc((size_t)maps.baseline_rows * H3_DIT_W4A8_HIDDEN *
                                sizeof(float));
        }
    }
    rcos = (float *)malloc((size_t)maps.reduced_seq * H3_DIT_W4A8_ROPE_HALF *
                           sizeof(float));
    rsin = (float *)malloc((size_t)maps.reduced_seq * H3_DIT_W4A8_ROPE_HALF *
                           sizeof(float));
    rmap = (uint32_t *)malloc((size_t)maps.reduced_seq * sizeof(uint32_t));
    if ((!use_resident_tr && (!reduced || !reduced2)) ||
        (!use_cuda_tr && (!original || !baseline)) || !rcos || !rsin || !rmap) {
        fail(error, error_size, "OOM token-reduction buffers");
        goto done;
    }
    for (i = 0; i < maps.reduced_seq; i++) {
        uint32_t first, second;
        tr_pool_sources(&maps, (uint32_t)i, &first, &second);
        (void)second;
        rmap[i] = row_map[first];
    }
    if (!tr_build_reduced_rope(&maps, core->inv_freq, full_pt, full_ph, full_pw,
                               spatial_scale, rcos, rsin, error, error_size))
        goto done;

    if (use_resident_tr) {
        d_full_a = ctx->io_d_act_a;
        d_full_b = ctx->io_d_act_b;
        if (!d_full_a || !d_full_b) {
            fail(error, error_size, "resident TR missing full act buffers");
            goto done;
        }
        if (!ensure_tr_ws(ctx, full_seq, maps.reduced_seq, maps.baseline_rows,
                          error, error_size))
            goto done;

        if (prefix_n) {
            h3_dit_w4a8_layers_stats s0;
            memset(&s0, 0, sizeof(s0));
            if (!h3_dit_w4a8_layers_forward_f32(
                    ctx, first_block, prefix_n, t_row, NULL, full_seq, row_map,
                    cos, sin, NULL, auto_prefetch, &s0, error, error_size))
                goto done;
            layers_ms += s0.total_ms;
            layers_io_promote_result(ctx);
            d_full_a = ctx->io_d_act_a;
            d_full_b = ctx->io_d_act_b;
        }

        if (!h3_cuda_token_pool_f32_dev(
                ctx->cuda, ctx->io_d_act_a, full_seq, maps.pairs,
                maps.baseline_idx, maps.reduced_seq, maps.baseline_rows,
                H3_DIT_W4A8_HIDDEN, ctx->d_tr_reduced_a, ctx->d_tr_original,
                maps.baseline_rows ? ctx->d_tr_baseline : NULL, NULL, error,
                error_size))
            goto done;

        layers_io_set(ctx, ctx->d_tr_reduced_a, ctx->d_tr_reduced_b, 1 | 2);
        {
            h3_dit_w4a8_layers_stats s1;
            memset(&s1, 0, sizeof(s1));
            if (!h3_dit_w4a8_layers_forward_f32(
                    ctx, begin, mid_n, t_row, NULL, maps.reduced_seq, rmap, rcos,
                    rsin, NULL, auto_prefetch, &s1, error, error_size))
                goto done;
            layers_ms += s1.total_ms;
        }
        if (!ctx->io_d_result) {
            fail(error, error_size, "resident TR mid missing device result");
            goto done;
        }

        if (!h3_cuda_token_expand_f32_dev(
                ctx->cuda, ctx->d_tr_original, ctx->io_d_result,
                maps.baseline_rows ? ctx->d_tr_baseline : NULL, maps.baseline_idx,
                maps.parents, full_seq, maps.reduced_seq, maps.baseline_rows,
                H3_DIT_W4A8_HIDDEN, maps.video_start, maps.video_rows,
                ctx->tr_scale, d_full_a, NULL, error, error_size))
            goto done;

        layers_io_set(ctx, d_full_a, d_full_b, 1 | 2);
        if (suffix_n) {
            h3_dit_w4a8_layers_stats s2;
            memset(&s2, 0, sizeof(s2));
            if (!h3_dit_w4a8_layers_forward_f32(
                    ctx, end, suffix_n, t_row, NULL, full_seq, row_map, cos, sin,
                    NULL, auto_prefetch, &s2, error, error_size))
                goto done;
            layers_ms += s2.total_ms;
        } else {
            ctx->io_d_result = d_full_a;
        }
    } else {
        if (prefix_n) {
            h3_dit_w4a8_layers_stats s0;
            memset(&s0, 0, sizeof(s0));
            if (!h3_dit_w4a8_layers_forward_f32(
                    ctx, first_block, prefix_n, t_row, hidden, full_seq, row_map,
                    cos, sin, hidden2, auto_prefetch, &s0, error, error_size))
                goto done;
            layers_ms += s0.total_ms;
            memcpy(hidden, hidden2,
                   full_seq * H3_DIT_W4A8_HIDDEN * sizeof(float));
        }

        if (use_cuda_tr) {
            if (!h3_cuda_token_pool_f32(
                    ctx->cuda, hidden, full_seq, maps.pairs, maps.baseline_idx,
                    maps.reduced_seq, maps.baseline_rows, H3_DIT_W4A8_HIDDEN,
                    reduced, &d_original, &d_baseline, NULL, error, error_size))
                goto done;
        } else {
            tr_pool_f32(&maps, hidden, reduced, original, baseline);
        }

        {
            h3_dit_w4a8_layers_stats s1;
            memset(&s1, 0, sizeof(s1));
            if (!h3_dit_w4a8_layers_forward_f32(
                    ctx, begin, mid_n, t_row, reduced, maps.reduced_seq, rmap,
                    rcos, rsin, reduced2, auto_prefetch, &s1, error,
                    error_size))
                goto done;
            layers_ms += s1.total_ms;
        }

        if (use_cuda_tr) {
            if (!h3_cuda_token_expand_f32(
                    ctx->cuda, d_original, reduced2, d_baseline,
                    maps.baseline_idx, maps.parents, full_seq, maps.reduced_seq,
                    maps.baseline_rows, H3_DIT_W4A8_HIDDEN, maps.video_start,
                    maps.video_rows, ctx->tr_scale, hidden, NULL, error,
                    error_size))
                goto done;
        } else {
            tr_expand_f32(&maps, original, reduced2, baseline, hidden,
                          ctx->tr_scale);
        }

        if (suffix_n) {
            h3_dit_w4a8_layers_stats s2;
            memset(&s2, 0, sizeof(s2));
            if (!h3_dit_w4a8_layers_forward_f32(
                    ctx, end, suffix_n, t_row, hidden, full_seq, row_map, cos,
                    sin, hidden2, auto_prefetch, &s2, error, error_size))
                goto done;
            layers_ms += s2.total_ms;
        } else {
            memcpy(hidden2, hidden,
                   full_seq * H3_DIT_W4A8_HIDDEN * sizeof(float));
        }
    }

    if (lst) {
        memset(lst, 0, sizeof(*lst));
        lst->first_block = first_block;
        lst->n_blocks = n_blocks;
        lst->total_ms = layers_ms;
        lst->all_finite = 1;
    }
    ok = 1;
done:
    free(reduced);
    free(reduced2);
    free(original);
    free(baseline);
    free(rcos);
    free(rsin);
    free(rmap);
    if (d_original) cudaFree(d_original);
    if (d_baseline) cudaFree(d_baseline);
    tr_maps_free(&maps);
    return ok;
}

int h3_dit_w4a8_forward_video_f32(
    h3_dit_w4a8 *ctx, const h3_dit_w4a8_core *core, unsigned first_block,
    unsigned n_blocks, float sigma, const float *latent, size_t video_seq,
    const float *text_hidden, size_t text_seq, const float *pos_t,
    const float *pos_h, const float *pos_w, float spatial_scale, float *velocity,
    int auto_prefetch, h3_dit_w4a8_forward_stats *stats, char *error,
    size_t error_size) {
    float *hidden = NULL, *hidden2 = NULL;
    float *cos = NULL, *sin = NULL;
    uint32_t *row_map = NULL;
    float *full_pt = NULL, *full_ph = NULL, *full_pw = NULL;
    float time_feat[H3_DIT_W4A8_TIME_FEAT];
    h3_dit_w4a8_layers_stats lst;
    size_t t_row, seq, i;
    clock_t t0, t1, tall0, tall1;
    int ok = 0;
    const int use_resident =
        ctx && ctx->cuda &&
        !env_flag_is_one("H3_DIT_W4A8_HOST_FWD") &&
        !env_flag_is_one("H3_DIT_W4A8_HOST_DENSE") &&
        !env_flag_is_one("H3_DIT_W4A8_FORCE_HOST_BLOCK");
    float *d_final_x = NULL;

    if (stats) memset(stats, 0, sizeof(*stats));
    if (!ctx || !core || !latent || !velocity || !pos_t || !pos_h || !pos_w ||
        video_seq == 0 || n_blocks == 0) {
        fail(error, error_size, "invalid forward_video args");
        return 0;
    }
    if (text_seq && !text_hidden) {
        fail(error, error_size, "text_seq>0 but text_hidden NULL");
        return 0;
    }
    if (!core->t_table || !core->inv_freq) {
        fail(error, error_size, "core missing t_table/inv_freq");
        return 0;
    }

    seq = text_seq + video_seq;
    t_row = h3_dit_w4a8_t_row_from_sigma(sigma, core->t_table_rows);
    if (stats) {
        stats->first_block = first_block;
        stats->n_blocks = n_blocks;
        stats->t_row = t_row;
    }
    tall0 = clock();

    if (!use_resident) {
        hidden = (float *)malloc(seq * H3_DIT_W4A8_HIDDEN * sizeof(float));
        hidden2 = (float *)malloc(seq * H3_DIT_W4A8_HIDDEN * sizeof(float));
    }
    cos = (float *)malloc(seq * H3_DIT_W4A8_ROPE_HALF * sizeof(float));
    sin = (float *)malloc(seq * H3_DIT_W4A8_ROPE_HALF * sizeof(float));
    row_map = (uint32_t *)calloc(seq, sizeof(uint32_t));
    full_pt = (float *)calloc(seq, sizeof(float));
    full_ph = (float *)calloc(seq, sizeof(float));
    full_pw = (float *)calloc(seq, sizeof(float));
    if ((!use_resident && (!hidden || !hidden2)) || !cos || !sin || !row_map ||
        !full_pt || !full_ph || !full_pw) {
        fail(error, error_size, "out of memory (forward_video)");
        goto done;
    }

    memcpy(time_feat, core->t_table + t_row * H3_DIT_W4A8_TIME_FEAT,
           H3_DIT_W4A8_TIME_FEAT * sizeof(float));

    /* Metal/Comfy: text RoPE t=token_index,h=0,w=0; AdaLN tag default MOD_TEXT.
     * H3_DIT_TEXT_MOD_ROW overrides tag for gating ablations. */
    {
        for (i = 0; i < text_seq; i++) {
            row_map[i] = dit_text_row_mod(ctx, i, text_seq);
            full_pt[i] = (float)i;
            full_ph[i] = 0.0f;
            full_pw[i] = 0.0f;
        }
    }
    for (i = 0; i < video_seq; i++) {
        row_map[text_seq + i] = (uint32_t)H3_DIT_W4A8_MOD_VIDEO;
        full_pt[text_seq + i] = pos_t[i];
        full_ph[text_seq + i] = pos_h[i];
        full_pw[text_seq + i] = pos_w[i];
    }

    if (!dist_attn_maybe_configure(ctx->cuda, seq, text_seq, video_seq, pos_t,
                                   pos_h, sigma, error, error_size))
        goto done;

    if (!h3_dit_w4a8_rope_tables_f32(core->inv_freq, seq, full_pt, full_ph,
                                     full_pw, spatial_scale, cos, sin, error,
                                     error_size))
        goto done;

    if (use_resident) {
        /* Pack on device: text H2D + patch→dev; layers leave-device; final
         * from device (only velocity D2H). Eliminates ~1GB×3 PCIe/step. */
        if (!ensure_fwd_act(ctx, seq, error, error_size)) goto done;
        t0 = clock();
        if (text_seq) {
            if (cudaMemcpy(ctx->d_fwd_act_a, text_hidden,
                           text_seq * H3_DIT_W4A8_HIDDEN * sizeof(float),
                           cudaMemcpyHostToDevice) != cudaSuccess) {
                fail(error, error_size, "forward text H2D failed");
                goto done;
            }
        }
        if (!dense_f32_cuda_to_dev(
                ctx->cuda, &ctx->d_video_patch_w, &ctx->d_video_patch_b, latent,
                core->video_patch_w, core->video_patch_b, video_seq,
                H3_DIT_W4A8_HIDDEN, H3_DIT_W4A8_VIDEO_PATCH,
                ctx->d_fwd_act_a + text_seq * H3_DIT_W4A8_HIDDEN, error,
                error_size))
            goto done;
        t1 = clock();
        if (stats) stats->patch_ms = elapsed_ms(t0, t1);

        layers_io_set(ctx, ctx->d_fwd_act_a, ctx->d_fwd_act_b, 1 | 2);
        memset(&lst, 0, sizeof(lst));
        if (!layers_with_token_reduction(
                ctx, core, first_block, n_blocks, t_row, NULL, NULL, text_seq,
                0, 0, video_seq, 0, row_map, cos, sin, full_pt, full_ph, full_pw,
                spatial_scale, auto_prefetch, 1, &lst, error, error_size)) {
            layers_io_clear(ctx);
            goto done;
        }
        if (stats) copy_layer_stats(stats, &lst);
        d_final_x = ctx->io_d_result;
        if (!d_final_x) {
            layers_io_clear(ctx);
            fail(error, error_size, "resident forward missing layers result");
            goto done;
        }

        t0 = clock();
        if (!final_head_cuda(
                ctx, core, NULL,
                d_final_x + text_seq * H3_DIT_W4A8_HIDDEN, video_seq, time_feat,
                &ctx->d_final_video_w, &ctx->d_final_video_b, core->final_video_w,
                core->final_video_b, H3_DIT_W4A8_VIDEO_PATCH, velocity, error,
                error_size)) {
            layers_io_clear(ctx);
            goto done;
        }
        layers_io_clear(ctx);
        t1 = clock();
    } else {
        /* Pack: [text_hidden | patch(video_latent)]. */
        t0 = clock();
        if (text_seq)
            memcpy(hidden, text_hidden,
                   text_seq * H3_DIT_W4A8_HIDDEN * sizeof(float));
        if (!h3_dit_w4a8_patch_video_cuda(
                ctx, core, latent, video_seq,
                hidden + text_seq * H3_DIT_W4A8_HIDDEN, error, error_size))
            goto done;
        t1 = clock();
        if (stats) stats->patch_ms = elapsed_ms(t0, t1);

        memset(&lst, 0, sizeof(lst));
        if (!layers_with_token_reduction(
                ctx, core, first_block, n_blocks, t_row, hidden, hidden2,
                text_seq, 0, 0, video_seq, 0, row_map, cos, sin, full_pt, full_ph,
                full_pw, spatial_scale, auto_prefetch, 0, &lst, error,
                error_size))
            goto done;
        if (stats) copy_layer_stats(stats, &lst);

        t0 = clock();
        if (!h3_dit_w4a8_final_video_cuda(
                ctx, core, hidden2 + text_seq * H3_DIT_W4A8_HIDDEN, video_seq,
                time_feat, velocity, error, error_size))
            goto done;
        t1 = clock();
    }

    tall1 = clock();
    if (stats) {
        size_t n = video_seq * H3_DIT_W4A8_VIDEO_PATCH;
        stats->final_ms = elapsed_ms(t0, t1);
        stats->total_ms = elapsed_ms(tall0, tall1);
        stats->all_finite = 1;
        for (i = 0; i < n; i++) {
            if (!isfinite(velocity[i])) {
                stats->all_finite = 0;
                break;
            }
        }
    }
    ok = 1;
done:
    layers_io_clear(ctx);
    free(hidden);
    free(hidden2);
    free(cos);
    free(sin);
    free(row_map);
    free(full_pt);
    free(full_ph);
    free(full_pw);
    return ok;
}

int h3_dit_w4a8_forward_av_f32(
    h3_dit_w4a8 *ctx, const h3_dit_w4a8_core *core, unsigned first_block,
    unsigned n_blocks, float sigma, float audio_sigma, const float *video_latent,
    size_t video_seq, const float *audio_latent, size_t audio_seq,
    const float *text_hidden, size_t text_seq, const float *v_pos_t,
    const float *v_pos_h, const float *v_pos_w, const float *a_pos_t,
    const float *a_pos_h, const float *a_pos_w, float spatial_scale,
    float *video_velocity, float *audio_velocity, int auto_prefetch,
    h3_dit_w4a8_forward_stats *stats, char *error, size_t error_size) {
    /* Thin wrapper: cond_seq=0. av≡avc memcmp is tautological vs this wrapper,
     * not vs the pre-B function body. Empty cond must not enter 4-row AdaLN. */
    return h3_dit_w4a8_forward_avc_f32(
        ctx, core, first_block, n_blocks, sigma, audio_sigma, video_latent,
        video_seq, audio_latent, audio_seq, NULL, 0, NULL, 0, text_hidden,
        text_seq, v_pos_t, v_pos_h, v_pos_w, a_pos_t, a_pos_h, a_pos_w, NULL,
        NULL, NULL, NULL, NULL, NULL, spatial_scale, video_velocity,
        audio_velocity, auto_prefetch, stats, error, error_size);
}

int h3_dit_w4a8_forward_avc_f32(
    h3_dit_w4a8 *ctx, const h3_dit_w4a8_core *core, unsigned first_block,
    unsigned n_blocks, float sigma, float audio_sigma,
    const float *video_latent, size_t video_seq, const float *audio_latent,
    size_t audio_seq, const float *cond_latent, size_t cond_seq,
    const float *audio_cond_latent, size_t audio_cond_seq,
    const float *text_hidden, size_t text_seq, const float *v_pos_t,
    const float *v_pos_h, const float *v_pos_w, const float *a_pos_t,
    const float *a_pos_h, const float *a_pos_w, const float *c_pos_t,
    const float *c_pos_h, const float *c_pos_w, const float *ac_pos_t,
    const float *ac_pos_h, const float *ac_pos_w, float spatial_scale,
    float *video_velocity, float *audio_velocity, int auto_prefetch,
    h3_dit_w4a8_forward_stats *stats, char *error, size_t error_size) {
    float *hidden = NULL, *hidden2 = NULL;
    float *cos = NULL, *sin = NULL;
    uint32_t *row_map = NULL;
    float *full_pt = NULL, *full_ph = NULL, *full_pw = NULL;
    float time_feat[H3_DIT_W4A8_TIME_FEAT];
    float time_feat_audio[H3_DIT_W4A8_TIME_FEAT];
    h3_dit_w4a8_layers_stats lst;
    size_t t_row, t_row_audio, seq, i, cond_off, audio_cond_off, video_off,
        audio_off;
    uint32_t cond_mod, audio_cond_mod;
    clock_t t0, t1, tall0, tall1;
    int ok = 0;
    const int use_resident =
        ctx && ctx->cuda &&
        !env_flag_is_one("H3_DIT_W4A8_HOST_FWD") &&
        !env_flag_is_one("H3_DIT_W4A8_HOST_DENSE") &&
        !env_flag_is_one("H3_DIT_W4A8_FORCE_HOST_BLOCK");
    float *d_final_x = NULL;

    if (stats) memset(stats, 0, sizeof(*stats));
    if (!ctx || !core || !video_latent || !audio_latent || !video_velocity ||
        !audio_velocity || !v_pos_t || !v_pos_h || !v_pos_w || !a_pos_t ||
        !a_pos_h || !a_pos_w || video_seq == 0 || audio_seq == 0 ||
        n_blocks == 0) {
        fail(error, error_size, "invalid forward_avc args");
        return 0;
    }
    if (text_seq && !text_hidden) {
        fail(error, error_size, "text_seq>0 but text_hidden NULL");
        return 0;
    }
    if (cond_seq && (!cond_latent || !c_pos_t || !c_pos_h || !c_pos_w)) {
        fail(error, error_size, "cond_seq>0 but cond_latent/c_pos NULL");
        return 0;
    }
    if (audio_cond_seq &&
        (!audio_cond_latent || !ac_pos_t || !ac_pos_h || !ac_pos_w)) {
        fail(error, error_size,
             "audio_cond_seq>0 but audio_cond_latent/ac_pos NULL");
        return 0;
    }
    if (!core->audio_patch_w || !core->final_audio_w) {
        fail(error, error_size, "core missing audio patch/final weights");
        return 0;
    }
    if (!core->t_table || !core->inv_freq) {
        fail(error, error_size, "core missing t_table/inv_freq");
        return 0;
    }

    /* Metal Ref2VA / keyframe order: text | cond | audio_cond | audio | video */
    seq = text_seq + cond_seq + audio_cond_seq + audio_seq + video_seq;
    cond_off = text_seq;
    audio_cond_off = text_seq + cond_seq;
    audio_off = text_seq + cond_seq + audio_cond_seq;
    video_off = text_seq + cond_seq + audio_cond_seq + audio_seq;
    t_row = h3_dit_w4a8_t_row_from_sigma(sigma, core->t_table_rows);
    t_row_audio = h3_dit_w4a8_t_row_from_sigma(audio_sigma, core->t_table_rows);
    /* Dual AdaLN timesteps (Metal schedule_row_map / vLLM flow_shift pair). */
    ctx->audio_t_row = t_row_audio;
    ctx->audio_t_active = (t_row_audio != t_row) ? 1 : 0;
    /* Empty cond must not enter extra AdaLN rows (bit-for-bit). */
    ctx->visual_cond_t_row =
        cond_seq ? h3_dit_w4a8_visual_cond_t_row(core->t_table_rows) : 0;
    ctx->visual_cond_t_active =
        h3_dit_w4a8_visual_cond_adaln_active(cond_seq, sigma);
    ctx->audio_cond_t_row = audio_cond_seq
                                ? h3_dit_w4a8_audio_cond_t_row(core->t_table_rows)
                                : 0;
    ctx->audio_cond_t_active =
        h3_dit_w4a8_audio_cond_adaln_active(audio_cond_seq, audio_sigma);
    cond_mod = h3_dit_w4a8_visual_cond_mod_row(cond_seq, sigma);
    audio_cond_mod = h3_dit_w4a8_audio_cond_mod_row(
        audio_cond_seq, audio_sigma, ctx->visual_cond_t_active);
    if (stats) {
        stats->first_block = first_block;
        stats->n_blocks = n_blocks;
        stats->t_row = t_row;
        stats->visual_cond_t_active = ctx->visual_cond_t_active;
        stats->visual_cond_mod_row = cond_mod;
        stats->visual_cond_t_row = ctx->visual_cond_t_row;
    }
    tall0 = clock();

    if (!use_resident) {
        hidden = (float *)malloc(seq * H3_DIT_W4A8_HIDDEN * sizeof(float));
        hidden2 = (float *)malloc(seq * H3_DIT_W4A8_HIDDEN * sizeof(float));
    }
    cos = (float *)malloc(seq * H3_DIT_W4A8_ROPE_HALF * sizeof(float));
    sin = (float *)malloc(seq * H3_DIT_W4A8_ROPE_HALF * sizeof(float));
    row_map = (uint32_t *)calloc(seq, sizeof(uint32_t));
    full_pt = (float *)calloc(seq, sizeof(float));
    full_ph = (float *)calloc(seq, sizeof(float));
    full_pw = (float *)calloc(seq, sizeof(float));
    if ((!use_resident && (!hidden || !hidden2)) || !cos || !sin || !row_map ||
        !full_pt || !full_ph || !full_pw) {
        fail(error, error_size, "out of memory (forward_avc)");
        goto done;
    }

    memcpy(time_feat, core->t_table + t_row * H3_DIT_W4A8_TIME_FEAT,
           H3_DIT_W4A8_TIME_FEAT * sizeof(float));
    memcpy(time_feat_audio, core->t_table + t_row_audio * H3_DIT_W4A8_TIME_FEAT,
           H3_DIT_W4A8_TIME_FEAT * sizeof(float));

    {
        for (i = 0; i < text_seq; i++) {
            row_map[i] = dit_text_row_mod(ctx, i, text_seq);
            full_pt[i] = (float)i;
            full_ph[i] = 0.0f;
            full_pw[i] = 0.0f;
        }
    }
    for (i = 0; i < cond_seq; i++) {
        row_map[cond_off + i] = cond_mod;
        full_pt[cond_off + i] = c_pos_t[i];
        full_ph[cond_off + i] = c_pos_h[i];
        full_pw[cond_off + i] = c_pos_w[i];
    }
    for (i = 0; i < audio_cond_seq; i++) {
        row_map[audio_cond_off + i] = audio_cond_mod;
        full_pt[audio_cond_off + i] = ac_pos_t[i];
        full_ph[audio_cond_off + i] = ac_pos_h[i];
        full_pw[audio_cond_off + i] = ac_pos_w[i];
    }
    for (i = 0; i < audio_seq; i++) {
        row_map[audio_off + i] = (uint32_t)H3_DIT_W4A8_MOD_AUDIO;
        full_pt[audio_off + i] = a_pos_t[i];
        full_ph[audio_off + i] = a_pos_h[i];
        full_pw[audio_off + i] = a_pos_w[i];
    }
    for (i = 0; i < video_seq; i++) {
        row_map[video_off + i] = (uint32_t)H3_DIT_W4A8_MOD_VIDEO;
        full_pt[video_off + i] = v_pos_t[i];
        full_ph[video_off + i] = v_pos_h[i];
        full_pw[video_off + i] = v_pos_w[i];
    }

    if (!dist_attn_maybe_configure(ctx->cuda, seq, video_off, video_seq, v_pos_t,
                                   v_pos_h, sigma, error, error_size))
        goto done;

    if (!h3_dit_w4a8_rope_tables_f32(core->inv_freq, seq, full_pt, full_ph,
                                     full_pw, spatial_scale, cos, sin, error,
                                     error_size))
        goto done;

    if (use_resident) {
        if (!ensure_fwd_act(ctx, seq, error, error_size)) goto done;
        t0 = clock();
        if (text_seq) {
            if (cudaMemcpy(ctx->d_fwd_act_a, text_hidden,
                           text_seq * H3_DIT_W4A8_HIDDEN * sizeof(float),
                           cudaMemcpyHostToDevice) != cudaSuccess) {
                fail(error, error_size, "forward_avc text H2D failed");
                goto done;
            }
        }
        if (cond_seq &&
            !dense_f32_cuda_to_dev(
                ctx->cuda, &ctx->d_video_patch_w, &ctx->d_video_patch_b,
                cond_latent, core->video_patch_w, core->video_patch_b, cond_seq,
                H3_DIT_W4A8_HIDDEN, H3_DIT_W4A8_VIDEO_PATCH,
                ctx->d_fwd_act_a + cond_off * H3_DIT_W4A8_HIDDEN, error,
                error_size))
            goto done;
        if (audio_cond_seq &&
            !dense_f32_cuda_to_dev(
                ctx->cuda, &ctx->d_audio_patch_w, &ctx->d_audio_patch_b,
                audio_cond_latent, core->audio_patch_w, core->audio_patch_b,
                audio_cond_seq, H3_DIT_W4A8_HIDDEN, H3_DIT_W4A8_AUDIO_CH,
                ctx->d_fwd_act_a + audio_cond_off * H3_DIT_W4A8_HIDDEN, error,
                error_size))
            goto done;
        if (!dense_f32_cuda_to_dev(
                ctx->cuda, &ctx->d_video_patch_w, &ctx->d_video_patch_b,
                video_latent, core->video_patch_w, core->video_patch_b, video_seq,
                H3_DIT_W4A8_HIDDEN, H3_DIT_W4A8_VIDEO_PATCH,
                ctx->d_fwd_act_a + video_off * H3_DIT_W4A8_HIDDEN, error,
                error_size) ||
            !dense_f32_cuda_to_dev(
                ctx->cuda, &ctx->d_audio_patch_w, &ctx->d_audio_patch_b,
                audio_latent, core->audio_patch_w, core->audio_patch_b, audio_seq,
                H3_DIT_W4A8_HIDDEN, H3_DIT_W4A8_AUDIO_CH,
                ctx->d_fwd_act_a + audio_off * H3_DIT_W4A8_HIDDEN, error,
                error_size))
            goto done;
        t1 = clock();
        if (stats) stats->patch_ms = elapsed_ms(t0, t1);

        layers_io_set(ctx, ctx->d_fwd_act_a, ctx->d_fwd_act_b, 1 | 2);
        memset(&lst, 0, sizeof(lst));
        if (!layers_with_token_reduction(
                ctx, core, first_block, n_blocks, t_row, NULL, NULL, text_seq,
                cond_seq, audio_cond_seq, video_seq, audio_seq, row_map, cos, sin,
                full_pt, full_ph, full_pw, spatial_scale, auto_prefetch, 1, &lst,
                error, error_size)) {
            layers_io_clear(ctx);
            goto done;
        }
        if (stats) copy_layer_stats(stats, &lst);
        d_final_x = ctx->io_d_result;
        if (!d_final_x) {
            layers_io_clear(ctx);
            fail(error, error_size, "resident forward_avc missing layers result");
            goto done;
        }
        t0 = clock();
        if (!final_head_cuda(
                ctx, core, NULL, d_final_x + video_off * H3_DIT_W4A8_HIDDEN,
                video_seq, time_feat, &ctx->d_final_video_w,
                &ctx->d_final_video_b, core->final_video_w, core->final_video_b,
                H3_DIT_W4A8_VIDEO_PATCH, video_velocity, error, error_size) ||
            !final_head_cuda(
                ctx, core, NULL, d_final_x + audio_off * H3_DIT_W4A8_HIDDEN,
                audio_seq, time_feat_audio, &ctx->d_final_audio_w,
                &ctx->d_final_audio_b, core->final_audio_w, core->final_audio_b,
                H3_DIT_W4A8_AUDIO_CH, audio_velocity, error, error_size)) {
            layers_io_clear(ctx);
            goto done;
        }
        layers_io_clear(ctx);
        t1 = clock();
    } else {
        t0 = clock();
        if (text_seq)
            memcpy(hidden, text_hidden,
                   text_seq * H3_DIT_W4A8_HIDDEN * sizeof(float));
        if (cond_seq &&
            !h3_dit_w4a8_patch_video_cuda(
                ctx, core, cond_latent, cond_seq,
                hidden + cond_off * H3_DIT_W4A8_HIDDEN, error, error_size))
            goto done;
        if (audio_cond_seq &&
            !h3_dit_w4a8_patch_audio_cuda(
                ctx, core, audio_cond_latent, audio_cond_seq,
                hidden + audio_cond_off * H3_DIT_W4A8_HIDDEN, error,
                error_size))
            goto done;
        if (!h3_dit_w4a8_patch_video_cuda(
                ctx, core, video_latent, video_seq,
                hidden + video_off * H3_DIT_W4A8_HIDDEN, error, error_size) ||
            !h3_dit_w4a8_patch_audio_cuda(
                ctx, core, audio_latent, audio_seq,
                hidden + audio_off * H3_DIT_W4A8_HIDDEN, error, error_size))
            goto done;
        t1 = clock();
        if (stats) stats->patch_ms = elapsed_ms(t0, t1);

        memset(&lst, 0, sizeof(lst));
        if (!layers_with_token_reduction(
                ctx, core, first_block, n_blocks, t_row, hidden, hidden2,
                text_seq, cond_seq, audio_cond_seq, video_seq, audio_seq, row_map,
                cos, sin, full_pt, full_ph, full_pw, spatial_scale,
                auto_prefetch, 0, &lst, error, error_size))
            goto done;
        if (stats) copy_layer_stats(stats, &lst);

        t0 = clock();
        if (!h3_dit_w4a8_final_video_cuda(
                ctx, core, hidden2 + video_off * H3_DIT_W4A8_HIDDEN, video_seq,
                time_feat, video_velocity, error, error_size) ||
            !h3_dit_w4a8_final_audio_cuda(
                ctx, core, hidden2 + audio_off * H3_DIT_W4A8_HIDDEN, audio_seq,
                time_feat_audio, audio_velocity, error, error_size))
            goto done;
        t1 = clock();
    }

    tall1 = clock();
    if (stats) {
        size_t nv = video_seq * H3_DIT_W4A8_VIDEO_PATCH;
        size_t na = audio_seq * H3_DIT_W4A8_AUDIO_CH;
        stats->final_ms = elapsed_ms(t0, t1);
        stats->total_ms = elapsed_ms(tall0, tall1);
        stats->all_finite = 1;
        for (i = 0; i < nv; i++) {
            if (!isfinite(video_velocity[i])) {
                stats->all_finite = 0;
                break;
            }
        }
        for (i = 0; stats->all_finite && i < na; i++) {
            if (!isfinite(audio_velocity[i])) stats->all_finite = 0;
        }
    }
    ok = 1;
done:
    if (ctx) {
        ctx->audio_t_active = 0;
        ctx->audio_t_row = 0;
        ctx->visual_cond_t_active = 0;
        ctx->visual_cond_t_row = 0;
        ctx->audio_cond_t_active = 0;
        ctx->audio_cond_t_row = 0;
    }
    layers_io_clear(ctx);
    free(hidden);
    free(hidden2);
    free(cos);
    free(sin);
    free(row_map);
    free(full_pt);
    free(full_ph);
    free(full_pw);
    return ok;
}

int h3_dit_w4a8_euler_step_f32(float *sample, const float *velocity,
                               size_t count, float sigma, float sigma_next,
                               char *error, size_t error_size) {
    size_t i;
    float delta;
    if (!sample || !velocity || count == 0) {
        fail(error, error_size, "invalid euler_step args");
        return 0;
    }
    if (!isfinite(sigma) || !isfinite(sigma_next) || !(sigma > sigma_next) ||
        sigma_next < 0.0f) {
        fail(error, error_size, "euler_step rejected sigma pair");
        return 0;
    }
    delta = sigma - sigma_next;
    for (i = 0; i < count; i++) sample[i] += delta * velocity[i];
    return 1;
}

int h3_dit_w4a8_reuse_schedule(int steps, int reuse_interval, uint8_t *selected,
                               size_t selected_count) {
    int step, count = 0;
    if (steps < 1 || reuse_interval < 1 || reuse_interval > 32 || !selected ||
        selected_count < (size_t)steps)
        return -1;
    memset(selected, 0, (size_t)steps);
    for (step = 0; step < steps; step++) {
        if (reuse_interval == 1 || step == 0 || step == steps - 1 ||
            step % reuse_interval == 0) {
            selected[step] = 1;
            count++;
        }
    }
    return count;
}

static float h3_dit_w4a8_extrapolation_ratio(float current_sigma,
                                             float last_sigma,
                                             float previous_sigma,
                                             int have_previous) {
    float denominator;
    float ratio;
    if (!have_previous) return 0.0f;
    denominator = last_sigma - previous_sigma;
    if (!(fabsf(denominator) > 1e-8f)) return 0.0f;
    ratio = (current_sigma - last_sigma) / denominator;
    if (!isfinite(ratio)) return 0.0f;
    return ratio;
}

void h3_dit_w4a8_extrapolate_velocity_f32(float *out, const float *last,
                                          const float *previous, size_t count,
                                          float current_sigma, float last_sigma,
                                          float previous_sigma,
                                          int have_previous) {
    size_t i;
    float ratio;
    if (!out || !last || count == 0) return;
    if (!have_previous || !previous) {
        memcpy(out, last, count * sizeof(*out));
        return;
    }
    ratio = h3_dit_w4a8_extrapolation_ratio(current_sigma, last_sigma,
                                            previous_sigma, 1);
    for (i = 0; i < count; i++)
        out[i] = last[i] + ratio * (last[i] - previous[i]);
}

float h3_dit_w4a8_velocity_rel_l1_f32(const float *a, const float *b,
                                      size_t count) {
    size_t i;
    double num = 0.0, den = 0.0;
    if (!a || !b || count == 0) return 1.0f;
    for (i = 0; i < count; i++) {
        num += fabs((double)a[i] - (double)b[i]);
        den += fabs((double)b[i]);
    }
    if (!(den > 1e-12)) return 1.0f;
    return (float)(num / den);
}

float h3_dit_w4a8_velocity_mean_abs_f32(const float *x, size_t count) {
    size_t i;
    double s = 0.0;
    if (!x || count == 0) return 0.0f;
    for (i = 0; i < count; i++) s += fabs((double)x[i]);
    return (float)(s / (double)count);
}

uint64_t h3_dit_w4a8_checksum_f32(const float *x, size_t n) {
    uint64_t h = 0xcbf29ce484222325ULL;
    size_t i;
    if (!x) return 0;
    for (i = 0; i < n; i++) {
        uint32_t bits = 0;
        memcpy(&bits, &x[i], sizeof(bits));
        h ^= (uint64_t)bits;
        h *= 0x100000001b3ULL;
    }
    return h;
}
