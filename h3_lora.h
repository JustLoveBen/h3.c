#ifndef H3_LORA_H
#define H3_LORA_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * LoRA × INT8 数值合同（禁止 merge 进金线 INT8 文件）：
 *
 *   Y = GEMM_i8(X, W)
 *   Y += lora_scale * (X @ A^T) @ B^T
 *     PEFT:  A [r, K], B [N, r]
 *     Kohya: lora_down = A, lora_up = B
 *
 * VSA / gate / sm100a 键：忽略并 log，5090 不接训练 VSA。
 * 缺 blocks.{0..49} 的 qkv/out/fc1/fc2 任一条所需 A/B：load 硬失败。
 */

enum { H3_LORA_N_BLOCKS = 50, H3_LORA_N_REFINER = 2 };

typedef enum {
    H3_LORA_SCOPE_DIT = 0,
    H3_LORA_SCOPE_REFINER = 1
} h3_lora_scope;

typedef struct h3_lora_pair {
    float *A; /* [r, K] row-major */
    float *B; /* [N, r] */
    int r, n, k;
} h3_lora_pair;

typedef struct h3_lora_block {
    h3_lora_pair qkv, out, fc1, fc2;
} h3_lora_block;

typedef struct h3_lora_set {
    char path[768];
    float scale;
    h3_lora_block blocks[H3_LORA_N_BLOCKS];
    h3_lora_block refiner[H3_LORA_N_REFINER];
    int n_pairs;
    int n_refiner_pairs;
    int ignored_vsa_tensors;
} h3_lora_set;

int h3_lora_name_is_vsa(const char *tensor_name);

/* Map HF/PEFT/Kohya name → block_index + suffix in {qkv,out,fc1,fc2} + AB.
 * 0 = 不识别（调用方：VSA 则 ignore，其它未知键 log+continue 或按合同 fail）。 */
int h3_lora_parse_name(const char *name, unsigned *block, const char **suffix,
                       int *is_a, char *err, size_t err_size);

/* TaoMate flat `*.lora_a` / `*.lora_b` plus token_refiner. DiT parse_name
 * still rejects token_refiner so turbo4 cannot collide on blocks.N. */
int h3_lora_parse_tensor(const char *name, int *scope, unsigned *block,
                         const char **suffix, int *is_a, char *err,
                         size_t err_size);

#define H3_LORA_REQUIRE_REFINER 1u
int h3_lora_load(const char *path, float scale, h3_lora_set *out, char *err,
                 size_t err_size);
int h3_lora_load_flags(const char *path, float scale, unsigned flags,
                       h3_lora_set *out, char *err, size_t err_size);
void h3_lora_free(h3_lora_set *set);

/* Host F32 residual: y[M,N] += scale * (x[M,K] @ A[r,K]^T) @ B[N,r]^T */
int h3_lora_apply_f32(const h3_lora_pair *p, float scale, const float *x,
                      float *y, size_t m, char *err, size_t err_size);

#ifdef __cplusplus
}
#endif

#endif
