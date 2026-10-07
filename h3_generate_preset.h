#ifndef H3_GENERATE_PRESET_H
#define H3_GENERATE_PRESET_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    H3_PRESET_NONE = 0, /* 未传 --preset，与今日一致 */
    H3_PRESET_GOLD,
    H3_PRESET_TURBO4,
    H3_PRESET_NVFP4, /* 8-step + NVFP4 DiT 权；非金线 */
    H3_PRESET_LIFT,  /* LBH 3D mid-σ；非金线加速档 */
    H3_PRESET_TAOMATE /* 3-step {0,16,33,49} + TaoMate LoRA；非金线 */
} h3_preset_id;

typedef struct h3_generate_preset_in {
    const char *preset_name; /* NULL 或 "" → NONE */
    const char *lora_path;   /* --lora 或 NULL */
    int steps_explicit;
    int steps;
    int sdpa_explicit;
    const char *sdpa; /* 调用方现有指针，可改写为内部静态名 */
    int cfg_explicit;
    float cfg_scale;
    float magcache_thresh;
    int step_cache_kind; /* h3_step_cache_kind 的 int 值 */
    int reuse_interval;  /* effective reuse; 1 = 全评 */
} h3_generate_preset_in;

typedef struct h3_generate_preset_out {
    h3_preset_id id;
    int steps;
    const char *sdpa;
    float cfg_scale;
    const char *lora_path;
    float lora_scale; /* 默认 1.0；本任务只透传，A2 使用 */
    int non_gold;     /* turbo4 = 1 */
    int lift;         /* taomate-lift：3-step 上叠 LBH；金线为 0 */
} h3_generate_preset_out;

/* 0 = 失败（填 err）。NONE/GOLD 不改 steps/sdpa/cfg。 */
int h3_generate_preset_apply(const h3_generate_preset_in *in,
                             h3_generate_preset_out *out, char *err,
                             size_t err_size);

#ifdef __cplusplus
}
#endif

#endif
