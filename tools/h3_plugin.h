#ifndef H3_PLUGIN_H
#define H3_PLUGIN_H

#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* H3 plugin registry: every H3_* env switch read via getenv()/h3_plugin_enabled()
 * must be registered here. tools/check_plugin_registry.sh verifies this.
 *
 * Status values:
 *   STABLE       — product path, default ON or OFF as specified
 *   EXPERIMENTAL — opt-in only, may change semantics
 *   DEAD_END_VERIFIED — verified dead end, kept for documentation
 *
 * Stage tags: DIT, VAE, MUX, SOL, TE, AUDIO, GUARD, MISC
 *
 * Default: "0" = off, "1" = on.
 */

#define H3_PLUGIN_BUILTIN_DEFS(X) \
    X("H3_AUDIO_CONV_BACKEND", "EXPERIMENTAL", "AUDIO", "0") \
    X("H3_AUDIO_ENCODER_TRACE", "EXPERIMENTAL", "AUDIO", "0") \
    X("H3_AUDIO_VAE", "EXPERIMENTAL", "AUDIO", "0") \
    X("H3_AUDIO_VAE_OVERLAP", "EXPERIMENTAL", "AUDIO", "0") \
    X("H3_AV_SYNC_GATE", "EXPERIMENTAL", "MUX", "0") \
    X("H3_BENCH_INT8_ATTENTION_OUT_AB", "EXPERIMENTAL", "DIT", "0") \
    X("H3_BENCH_INT8_MLP_AB", "EXPERIMENTAL", "DIT", "0") \
    X("H3_BENCH_INT8_QKV_AB", "EXPERIMENTAL", "DIT", "0") \
    X("H3_CUDA_L2_PIN", "EXPERIMENTAL", "DIT", "0") \
    X("H3_CONVROT_SYLVESTER", "EXPERIMENTAL", "DIT", "0") \
    X("H3_CPU_SAMPLER", "EXPERIMENTAL", "MISC", "0") \
    X("H3_CUDA_SDPA_BACKEND", "EXPERIMENTAL", "DIT", "auto") \
    X("H3_DEBUG_VISION", "EXPERIMENTAL", "MISC", "0") \
    X("H3_DEBUG_VISION_STEPS", "EXPERIMENTAL", "MISC", "0") \
    X("H3_DISABLE_DIT_ACTIVATION_ALIAS", "EXPERIMENTAL", "DIT", "0") \
    X("H3_DISABLE_FUSED_CROSS_BLOCK_ADALN", "EXPERIMENTAL", "DIT", "0") \
    X("H3_DISABLE_FUSED_FINAL_HEAD", "EXPERIMENTAL", "DIT", "0") \
    X("H3_DISABLE_FUSED_FINAL_SLICE", "EXPERIMENTAL", "DIT", "0") \
    X("H3_DISABLE_FUSED_GATE_ADALN", "EXPERIMENTAL", "DIT", "0") \
    X("H3_DISABLE_FUSED_INT8_MLP_INPUT", "EXPERIMENTAL", "DIT", "0") \
    X("H3_DISABLE_FUSED_INT8_QKV_INPUT", "EXPERIMENTAL", "DIT", "0") \
    X("H3_DISABLE_FUSED_MLP", "EXPERIMENTAL", "DIT", "0") \
    X("H3_DISABLE_FUSED_PATCH_CAST", "EXPERIMENTAL", "DIT", "0") \
    X("H3_DISABLE_FUSED_PATCH_PACK", "EXPERIMENTAL", "DIT", "0") \
    X("H3_DISABLE_FUSED_TOKEN_ADALN", "EXPERIMENTAL", "DIT", "0") \
    X("H3_DISABLE_FUSED_TOKEN_POOL_ADALN", "EXPERIMENTAL", "DIT", "0") \
    X("H3_DISABLE_HEAD_MAJOR_ATTENTION_OUTPUT", "EXPERIMENTAL", "DIT", "0") \
    X("H3_DISABLE_INT8_ATTENTION_OUT", "EXPERIMENTAL", "DIT", "0") \
    X("H3_DISABLE_INT8_MLP", "EXPERIMENTAL", "DIT", "0") \
    X("H3_DISABLE_INT8_QKV", "EXPERIMENTAL", "DIT", "0") \
    X("H3_DISABLE_NAX_MLP", "EXPERIMENTAL", "DIT", "0") \
    X("H3_DISABLE_TOKEN_REDUCTION", "EXPERIMENTAL", "DIT", "0") \
    X("H3_DIT_ACT_F16", "EXPERIMENTAL", "DIT", "0") \
    X("H3_DIT_ADALN_SILU", "EXPERIMENTAL", "DIT", "0") \
    X("H3_DIT_COMMAND_BLOCKS", "EXPERIMENTAL", "DIT", "0") \
    X("H3_DIT_DIST_ATTN", "EXPERIMENTAL", "DIT", "0") \
    X("H3_DIT_F32_FINAL", "EXPERIMENTAL", "DIT", "0") \
    X("H3_DIT_FUSED_EPILOGUE", "STABLE", "DIT", "1") \
    X("H3_DIT_HOST_STAGE_TE", "EXPERIMENTAL", "DIT", "0") \
    X("H3_DIT_HOST_STAGE_THREADS", "EXPERIMENTAL", "DIT", "0") \
    X("H3_DIT_HOST_STAGE_VERBOSE", "EXPERIMENTAL", "DIT", "0") \
    X("H3_DIT_HOST_STAGE_WAIT_BLOCKS", "EXPERIMENTAL", "DIT", "0") \
    X("H3_DIT_INT8_ALGO_BEST", "EXPERIMENTAL", "DIT", "0") \
    X("H3_DIT_INT8_ALLOW_SHRINK", "EXPERIMENTAL", "DIT", "0") \
    X("H3_DIT_INT8_NO_CONVROT", "EXPERIMENTAL", "DIT", "0") \
    X("H3_DIT_LAYER_POLICY", "EXPERIMENTAL", "DIT", "0") \
    X("H3_DIT_LORA_F16", "EXPERIMENTAL", "DIT", "0") \
    X("H3_DIT_LORA_OVERLAP", "EXPERIMENTAL", "DIT", "0") \
    X("H3_DIT_LORA_TF32", "EXPERIMENTAL", "DIT", "0") \
    X("H3_DIT_NO_REFINER", "EXPERIMENTAL", "DIT", "0") \
    X("H3_DIT_PLAN_REUSE", "EXPERIMENTAL", "DIT", "0") \
    X("H3_DIT_QKV_GROUPED", "EXPERIMENTAL", "DIT", "0") \
    X("H3_DIT_QKV_MERGE", "EXPERIMENTAL", "DIT", "0") \
    X("H3_DIT_QKNORM_ROPE_FUSE", "EXPERIMENTAL", "DIT", "0") \
    X("H3_DIT_ROPE_HALF_OFFSET", "EXPERIMENTAL", "DIT", "0") \
    X("H3_DIT_ROPE_SPLIT_HALF", "EXPERIMENTAL", "DIT", "0") \
    X("H3_DIT_TEXT_GAIN", "EXPERIMENTAL", "DIT", "0") \
    X("H3_DIT_TEXT_MOD_ROW", "EXPERIMENTAL", "DIT", "0") \
    X("H3_DIT_TE_OVERLAP_BLOCKS", "EXPERIMENTAL", "DIT", "0") \
    X("H3_DIT_W4A8_ATTN_WINDOW", "EXPERIMENTAL", "DIT", "0") \
    X("H3_DIT_W4A8_CUDA_GRAPH", "EXPERIMENTAL", "DIT", "0") \
    X("H3_DUMP_DIR", "EXPERIMENTAL", "MISC", "0") \
    X("H3_DUMP_LATENT", "EXPERIMENTAL", "VAE", "0") \
    X("H3_DUMP_ONLY", "EXPERIMENTAL", "MISC", "0") \
    X("H3_LIFECYCLE_TRACE", "EXPERIMENTAL", "MISC", "0") \
    X("H3_FFMPEG", "EXPERIMENTAL", "MUX", "0") \
    X("H3_FFMPEG_STREAM", "EXPERIMENTAL", "MUX", "0") \
    X("H3_FFMPEG_THREADS", "EXPERIMENTAL", "MUX", "0") \
    X("H3_FFMPEG_VENC", "EXPERIMENTAL", "MUX", "0") \
    X("H3_X264_PRESET", "STABLE", "MUX", "veryfast") \
    X("H3_X264_PRESET_TIER", "STABLE", "MUX", "quality") \
    X("H3_FFPROBE", "EXPERIMENTAL", "MUX", "0") \
    X("H3_GEN_MEM_TRACE", "EXPERIMENTAL", "MISC", "0") \
    X("H3_GEN_VAE_PREWARM", "EXPERIMENTAL", "VAE", "0") \
    X("H3_VAE_PREWARM_DEV", "EXPERIMENTAL", "VAE", "auto") \
    X("H3_GPU_SAMPLER", "EXPERIMENTAL", "MISC", "0") \
    X("H3_GPU_SAMPLER_WINDOW", "EXPERIMENTAL", "MISC", "0") \
    X("H3_INT8_GEMM_ACC_MIB", "EXPERIMENTAL", "DIT", "0") \
    X("H3_INT8_GEMM_TILE_M", "EXPERIMENTAL", "DIT", "0") \
    X("H3_INT8_KEEP_BF16_ATTENTION_OUT", "EXPERIMENTAL", "DIT", "0") \
    X("H3_INT8_KEEP_BF16_MLP", "EXPERIMENTAL", "DIT", "0") \
    X("H3_INT8_KEEP_BF16_QKV", "EXPERIMENTAL", "DIT", "0") \
    X("H3_INT8_LT_WORKSPACE_MIB", "EXPERIMENTAL", "DIT", "0") \
    X("H3_INT8_MLP_STAGE", "EXPERIMENTAL", "DIT", "0") \
    X("H3_LATENT_UPSCALER", "EXPERIMENTAL", "MISC", "0") \
    X("H3_LAYER_KV_ALLOW_TC", "EXPERIMENTAL", "DIT", "0") \
    X("H3_LBH_CONV", "EXPERIMENTAL", "MISC", "0") \
    X("H3_LBH_TIME_CHUNK", "EXPERIMENTAL", "MISC", "0") \
    X("H3_LIFT_HIRES_WINDOW", "EXPERIMENTAL", "MISC", "0") \
    X("H3_LIFT_Z0_ONLY", "EXPERIMENTAL", "MISC", "0") \
    X("H3_LIFT_ZLOW_ONLY", "EXPERIMENTAL", "MISC", "0") \
    X("H3_MAGCACHE_MAX_CONSEC", "EXPERIMENTAL", "DIT", "0") \
    X("H3_NSYS_HIRES", "EXPERIMENTAL", "MISC", "0") \
    X("H3_NVFP4_DEV_ACT", "EXPERIMENTAL", "DIT", "0") \
    X("H3_NVFP4_DISABLE_DEV_CACHE", "EXPERIMENTAL", "DIT", "0") \
    X("H3_NVFP4_DUMP_DIR", "EXPERIMENTAL", "DIT", "0") \
    X("H3_NVFP4_HOST_ACT", "EXPERIMENTAL", "DIT", "0") \
    X("H3_NVFP4_HOST_GEMM", "EXPERIMENTAL", "DIT", "0") \
    X("H3_NVFP4_HOST_GQA", "EXPERIMENTAL", "DIT", "0") \
    X("H3_NVFP4_INT8_SIDECAR", "EXPERIMENTAL", "DIT", "0") \
    X("H3_NVFP4_TE_LAYERS", "EXPERIMENTAL", "DIT", "0") \
    X("H3_NVFP4_VERBOSE", "EXPERIMENTAL", "DIT", "0") \
    X("H3_PROFILE", "EXPERIMENTAL", "MISC", "0") \
    X("H3_QWEN_PREFETCH", "EXPERIMENTAL", "TE", "0") \
    X("H3_QWEN_PREFETCH_DEPTH", "EXPERIMENTAL", "TE", "0") \
    X("H3_REPLAY_LATENT", "EXPERIMENTAL", "VAE", "0") \
    X("H3_REUSE_STEPS", "EXPERIMENTAL", "DIT", "0") \
    X("H3_SAGE3_LIB", "EXPERIMENTAL", "SOL", "0") \
    X("H3_SAGE3_REQUIRE_HIT", "EXPERIMENTAL", "SOL", "0") \
    X("H3_SAGE3_ROOT", "EXPERIMENTAL", "SOL", "0") \
    X("H3_SAGE3_TOOLS", "EXPERIMENTAL", "SOL", "0") \
    X("H3_SCALAR_PATCH", "EXPERIMENTAL", "DIT", "0") \
    X("H3_SDPA_STRICT", "EXPERIMENTAL", "DIT", "0") \
    X("H3_SKIP_PPM", "EXPERIMENTAL", "MISC", "0") \
    X("H3_SOL_ATTN_ROOT", "EXPERIMENTAL", "SOL", "0") \
    X("H3_SOL_CUTE_LIB", "EXPERIMENTAL", "SOL", "0") \
    X("H3_SOL_CUTE_TOOLS", "EXPERIMENTAL", "SOL", "0") \
    X("H3_SOL_FIRST_DENSE_LAYERS", "EXPERIMENTAL", "SOL", "0") \
    X("H3_SOL_FIRST_DENSE_STEPS", "EXPERIMENTAL", "SOL", "0") \
    X("H3_SOL_HIRES_QKVO", "EXPERIMENTAL", "SOL", "0") \
    X("H3_SOL_MAX_SEQ", "EXPERIMENTAL", "SOL", "0") \
    X("H3_SOL_PROFILE", "EXPERIMENTAL", "SOL", "0") \
    X("H3_SOL_SINK_TOKENS", "EXPERIMENTAL", "SOL", "0") \
    X("H3_SOL_TAU", "EXPERIMENTAL", "SOL", "0") \
    X("H3_SOL_WARMUP_DIM", "EXPERIMENTAL", "SOL", "0") \
    X("H3_SOL_WARMUP_HEADS", "EXPERIMENTAL", "SOL", "0") \
    X("H3_SOL_WARMUP_SEQ", "EXPERIMENTAL", "SOL", "0") \
    X("H3_T2AV_FORCE_PER_BLOCK", "EXPERIMENTAL", "DIT", "0") \
    X("H3_TAE_NO_CHUNK", "EXPERIMENTAL", "VAE", "0") \
    X("H3_TAE_TIME_CHUNK", "EXPERIMENTAL", "VAE", "0") \
    X("H3_TAOMATE_CHUNK_OVERLAP", "EXPERIMENTAL", "DIT", "0") \
    X("H3_TAOMATE_CLEAN_KV", "EXPERIMENTAL", "DIT", "0") \
    X("H3_TAOMATE_FORCE_CHUNKS", "EXPERIMENTAL", "DIT", "0") \
    X("H3_TAOMATE_LAYER_KV", "EXPERIMENTAL", "DIT", "0") \
    X("H3_TAOMATE_LORA", "EXPERIMENTAL", "DIT", "0") \
    X("H3_TAOMATE_SINK_T", "EXPERIMENTAL", "DIT", "0") \
    X("H3_TAOMATE_STAGE1_H", "EXPERIMENTAL", "DIT", "0") \
    X("H3_TAOMATE_STAGE1_W", "EXPERIMENTAL", "DIT", "0") \
    X("H3_TE_MMAP_HOSTREG", "EXPERIMENTAL", "TE", "0") \
    X("H3_TE_MMAP_POPULATE", "EXPERIMENTAL", "TE", "0") \
    X("H3_TE_PACKED_RESIDENT", "EXPERIMENTAL", "TE", "0") \
    X("H3_TE_PREFAULT", "EXPERIMENTAL", "TE", "0") \
    X("H3_TE_PROMPT_CACHE", "EXPERIMENTAL", "TE", "0") \
    X("H3_TE_CACHE_REQUIRE_HIT", "EXPERIMENTAL", "TE", "0") \
    X("H3_TE_CACHE_AUTO", "EXPERIMENTAL", "TE", "0") \
    X("H3_TOKEN_REDUCTION", "EXPERIMENTAL", "DIT", "0") \
    X("H3_TOKEN_REDUCTION_BLOCKS", "EXPERIMENTAL", "DIT", "0") \
    X("H3_TOKEN_REDUCTION_EARLY", "EXPERIMENTAL", "DIT", "0") \
    X("H3_TOKEN_REDUCTION_SCALE", "EXPERIMENTAL", "DIT", "0") \
    X("H3_TURBO_LORA", "EXPERIMENTAL", "DIT", "0") \
    X("H3_ULYSSES_A2A_F32", "EXPERIMENTAL", "DIT", "0") \
    X("H3_VAE_ACT_F16", "EXPERIMENTAL", "VAE", "0") \
    X("H3_VAE_BLOCKS_2STREAM", "EXPERIMENTAL", "VAE", "0") \
    X("H3_VAE_CUDA_GRAPH", "EXPERIMENTAL", "VAE", "0") \
    X("H3_VAE_CUDA_W_RESIDENT", "EXPERIMENTAL", "VAE", "0") \
    X("H3_VAE_DEBLOCK", "EXPERIMENTAL", "VAE", "0") \
    X("H3_VAE_DETERMINISTIC", "EXPERIMENTAL", "VAE", "0") \
    X("H3_VAE_ENCODE_HOST", "EXPERIMENTAL", "VAE", "0") \
    X("H3_VAE_FUSED_EPILOGUE", "EXPERIMENTAL", "VAE", "0") \
    X("H3_VAE_GEMM_F16_DIRECT", "EXPERIMENTAL", "VAE", "0") \
    X("H3_VAE_HOSTREG", "EXPERIMENTAL", "VAE", "0") \
    X("H3_VAE_NO_CONVROT", "EXPERIMENTAL", "VAE", "0") \
    X("H3_VAE_NO_SPATIAL_TILE", "EXPERIMENTAL", "VAE", "0") \
    X("H3_VAE_QKV_LAYOUT", "EXPERIMENTAL", "VAE", "0") \
    X("H3_VAE_REPEAT_RELEASE", "EXPERIMENTAL", "VAE", "0") \
    X("H3_VAE_SDPA_F32", "EXPERIMENTAL", "VAE", "0") \
    X("H3_VAE_TILE_BATCH", "EXPERIMENTAL", "VAE", "0") \
    X("H3_VAE_TILE_PIXELS", "EXPERIMENTAL", "VAE", "0") \
    X("H3_VAE_UNPATCH_ASYNC", "STABLE", "VAE", "1") \
    X("H3_VAE_UNPATCH_PINNED", "STABLE", "VAE", "1") \
    X("H3_VAE_WEIGHT_CACHE", "EXPERIMENTAL", "VAE", "0") \
    X("H3_VISION_HOST", "EXPERIMENTAL", "MISC", "0") \
    X("H3_VISION_SDPA", "EXPERIMENTAL", "MISC", "0") \
    X("H3_WAFFLE_COLD_GUARD", "EXPERIMENTAL", "GUARD", "0") \
    X("H3_WAFFLE_COLD_MAX", "EXPERIMENTAL", "GUARD", "2.0") \
    X("H3_W4A8_CACHE_REFIT_RESERVE_MIB", "EXPERIMENTAL", "DIT", "0") \
    X("H3_W4A8_CACHE_RESERVE_MIB", "EXPERIMENTAL", "DIT", "0")

typedef struct {
    const char *name;
    const char *status;
    const char *stage;
    const char *default_value;
} h3_plugin_def;

#define H3_PLUGIN_DEF_ITEM(n, s, g, d) {n, s, g, d},
static const h3_plugin_def h3_plugin_builtins[] = {
    H3_PLUGIN_BUILTIN_DEFS(H3_PLUGIN_DEF_ITEM)
};
#undef H3_PLUGIN_DEF_ITEM

static const size_t h3_plugin_builtin_count =
    sizeof(h3_plugin_builtins) / sizeof(h3_plugin_builtins[0]);

static const h3_plugin_def *h3_plugin_find(const char *name) {
    size_t i;
    if (!name) return NULL;
    for (i = 0; i < h3_plugin_builtin_count; i++) {
        if (strcmp(h3_plugin_builtins[i].name, name) == 0)
            return &h3_plugin_builtins[i];
    }
    return NULL;
}

static int h3_plugin_enabled(const char *name) {
    const char *e = getenv(name);
    const h3_plugin_def *def = h3_plugin_find(name);
    const char *val = (e && e[0]) ? e : (def ? def->default_value : "0");
    if (!val || !val[0]) return 0;
    return strcmp(val, "0") != 0 && strcmp(val, "false") != 0 &&
           strcmp(val, "off") != 0 && strcmp(val, "no") != 0;
}

#ifdef __cplusplus
}
#endif

#endif
