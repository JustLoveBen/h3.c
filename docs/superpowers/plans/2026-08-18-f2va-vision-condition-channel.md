# F2VA Vision Condition Channel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 CUDA 旁路 DiT（`h3_dit_w4a8` / `h3_dit_int8`）上接通官方 vision condition channel，让首/尾帧经 VAE encode 后作为 `H3_SEG_COND` condition rows 进入 DiT 序列，解决 F2VA 后续帧人物看不清的架构性根因。

**Architecture:** 复刻 Metal 官方路径：jpeg → VAE encoder → latent → patchify → condition rows（带 `condition_time` RoPE）→ DiT packed sequence `[text | cond | audio | video]`。DiT 全程看到干净首/尾帧，condition 行参与 attention 但不接收 Euler 速度。新增 CUDA 视频 VAE encoder（先 host F32 保正确性，再 CUDA 加速）。DiT forward 扩展为 `forward_avc`（av + condition），空 condition 时退化为现有 `forward_av`，T2V/T2AV 行为不变。

**Tech Stack:** C11 / CUDA 12 / cuBLAS / cuDNN SDPA / INT8 ConvRot DiT / fp16 VAE encoder（`minimax_h3_video_vae_fp16.safetensors`）/ ffmpeg 图像读取 / Metal 旁路参考实现（`h3.c` + `h3_host.c` + `h3_dit.c` + `h3_video_encoder.c`）。

**Spec:** 调研结果（下文 §背景）+ `docs/nvidia_optimization_plan.md` G3 "encode ⛔"（`docs/nvidia_optimization_plan.md:508` / `:544` / `:1199`）。

## 背景（调研已确认）

- **CUDA 旁路现状**：`h3_dit_w4a8` / `h3_dit_int8` 只有 text condition，grep 无 `img_cond` / `cond_rows` / `H3_SEG_COND`。`tools/h3_cuda_generate.c` 的 F2VA 是"噪声侧 img2img lite"（`apply_t0_img2img`，`:230`），DiT 看不到干净首帧 → 后续帧跑飞。
- **Metal 官方路径**：
  - 首/尾帧读取 + VAE encode：`h3.c:1224`（`params->first_frame` → `keyframes[0]=0`）、`h3.c:1238`（`params->last_frame` → `keyframes[1]=frame_count-1`）、`h3.c:1301`（`h3_video_vae_encode` → `condition_video_rows`）。
  - layout 发 `H3_SEG_COND` + `condition_time` RoPE：`h3_host.c:348-362`（keyframes 循环，`condition_time = text_len` 或 `text_len + video_span - frame_rescale`，emit `H3_SEG_COND`，`layout->img_cond_rows += frame_rows`）。
  - DiT 消费 condition：`h3_dit.c:325`（`H3_SEG_COND` 计入 `video_condition`）、`h3_dit.c:1580`（`h3_dit_load_conditioned` 接 `condition_video_rows`）、`h3_dit.c:2080` / `:2139`（patch 阶段 `H3_SEG_COND` 段用 `video_input` + `video_patch_w`，与 `H3_SEG_VIDEO` 同源投影）。
  - condition 行参与 attention（无显式 mask，全双向），但不接收 Euler 速度（final layer 输出仅取 video target rows）。
- **latent layout**：`tools/h3_cuda_generate.c:188` 注释 `latent=[video_seq,96] patch-major T,H/2,W/2`；`h3_dit_w4a8_fill_metal_video_pos`（`h3_dit_w4a8.c:974`）生成 video RoPE 坐标。
- **DiT forward 现状**：`h3_dit_w4a8_forward_av_f32`（`h3_dit_w4a8.c:4340`）pack `[text | audio | video]`，row_map = `[MOD_TEXT... | MOD_AUDIO... | MOD_VIDEO...]`，RoPE 坐标 `[text: (i,0,0) | audio: a_pos | video: v_pos]`，velocity 写 `video_velocity[video_seq*96]` + `audio_velocity[audio_seq*32]`。
- **VAE 现状**：`h3_vae_int8_cuda.c` 只有 decode（`h3_vae_int8_decode_cuda`）。VAE checkpoint `models/comfy-org/vae/minimax_h3_video_vae_fp16.safetensors` 为 fp16，含 encoder + decoder 权重。Metal encoder：`h3_video_encoder.c` / `h3_video_encoder.h:22`（`h3_video_vae_encode`，含 256px tiling + overlap）。
- **TE 已上 GPU**：`cuda_cudnn_gqa_causal`，全文 1626 token 可跑。
- **质量约束**：INT8 DiT、cfg=5、reuse=off、VAE 256 tiling+deblock、cudnn SDPA。

## Global Constraints

- **不改 Metal**：`h3.c` / `h3_host.c` / `h3_dit.c` / `h3_video_encoder.c` / `h3_video_vae.c` 不动；Metal 官方路径是参考实现与回归基线。
- **不破坏 T2V/T2AV**：condition 是 opt-in；`cond_seq=0` 时 `forward_avc` 行为与现有 `forward_av` 完全一致（bit-for-bit 速度不变）。
- **DiT 量化不变**：INT8 ConvRot DiT（`h3_dit_int8`）/ W4A8（`h3_dit_w4a8`）权重路径不动；condition channel 只动序列布局与 row_map / RoPE / final 提取。
- **VAE encoder 权重源**：fp16 VAE checkpoint（`minimax_h3_video_vae_fp16.safetensors`），encoder 权重按需 host F32 dequant；不依赖 INT8 VAE store 的 decoder-only 权重。
- **VAE tiling**：encoder 复用 decoder 的 256px tiling + overlap 策略（`H3_VAE_INT8_SPATIAL_RATIO=16`，overlap 64px），与 Metal `h3_video_vae_encode` 一致。
- **cfg=5 / reuse=off**：condition channel 在 cfg=5 主路径上验证；reuse 路径（外推速度）保持对 condition 行无副作用。
- **密码勿入库**；**不要 git commit**（用户明确要求）。
- **命名**：新文件 `h3_vae_int8_encode_cuda.*`（CUDA encoder）、`h3_vae_encode_f32.*`（host F32 encoder）；新 API `h3_dit_w4a8_forward_avc_f32`（av + condition）。
- **平台**：CUDA 路径在远程 NVIDIA box 运行（`out/verify_t4_remote`）；本地 darwin 仅作 Metal 参考。

---

## 任务总表（文首）

| ID | 名称 | 依赖 | 预估 | DoD 摘要 |
|----|------|------|------|----------|
| T1 | DiT forward 骨架扩展（`forward_avc`，空 condition 不改行为） | — | 3h | `forward_avc` 落地；`cond_seq=0` 时 `forward_av` 输出 bit-for-bit 不变；现有 probe/test PASS |
| T2 | VAE encoder 权重加载（fp16 checkpoint） | — | 2h | 从 fp16 VAE 加载 encoder 权重到 host F32；tensor 名清单打印；OOM-safe free |
| T3 | VAE encoder forward（host F32 + tiling） | T2 | 4h | jpeg → latent；与 Metal `h3_video_vae_encode` 输出 max abs diff < 5e-2（256px 单 tile） |
| T4 | CUDA VAE encoder（性能加速，可后置） | T3 | 4h | CUDA encode vs host F32 max abs diff < 1e-3；单帧 encode < 200ms（256px） |
| T5 | CLI `--first-frame` / `--last-frame`（官方语义） | T1, T3 | 2h | `--first-frame IMG` 生成 cond_rows 并传入 `forward_avc`；无 condition 时 CLI 行为不变 |
| T6 | 首帧 condition 端到端（F2VA f0 对齐） | T5 | 2h | f0 PSNR vs jpeg > 28dB；后续帧骑士可认；不闪 |
| T7 | 尾帧 condition | T5 | 1.5h | 末帧 PSNR vs jpeg > 28dB；首→尾插值连贯 |
| T8 | 首尾帧同时（keyframes[2]） | T6, T7 | 1.5h | 首尾两 cond 段；首末帧对齐；中段插值 |
| T9 | 回归 + 验证 | T6, T7, T8 | 2h | T2V/T2AV 无 condition 输出不变；F2VA 主题跟提示词；metrics_summary 落盘 |

**总预估**：22h（T4 可后置则 18h）。

---

## File Structure

| 文件 | 职责 | 状态 |
|------|------|------|
| `h3_dit_w4a8.h` | 新增 `h3_dit_w4a8_forward_avc_f32` 声明 + `H3_DIT_W4A8_MOD_*` 文档 | 修改 |
| `h3_dit_w4a8.c` | 新增 `forward_avc` 实现；`forward_av` 改为 `cond_seq=0` 调用 `forward_avc` | 修改 |
| `h3_vae_encode_f32.h` | host F32 VAE encoder API（`h3_vae_encode_f32`） | 新建 |
| `h3_vae_encode_f32.c` | host F32 encoder forward + tiling（参考 Metal `h3_video_encoder.c`） | 新建 |
| `h3_vae_int8_encode_cuda.h` | CUDA VAE encoder API（`h3_vae_int8_encode_cuda`） | 新建（T4） |
| `h3_vae_int8_encode_cuda.c` | CUDA encoder forward（复用 `h3_vae_int8_decode_cuda` 的 GEMM/SDPA 基础设施） | 新建（T4） |
| `tools/h3_cuda_generate.c` | CLI `--first-frame` / `--last-frame`；cond_rows 打包；调用 `forward_avc` | 修改 |
| `tests/test_h3_dit_w4a8_cond.c` | T1 回归：`forward_avc` cond_seq=0 == `forward_av` | 新建 |
| `tests/test_h3_vae_encode_f32.c` | T3 回归：host encoder vs Metal encoder | 新建 |
| `CMakeLists.txt` / `Makefile` | 新增 test/executable target | 修改 |

---

<!-- TASKS -->

## Task 1: DiT forward 骨架扩展（`forward_avc`，空 condition 不改行为）

**Files:**
- Modify: `h3_dit_w4a8.h:423-431`（`forward_av_f32` 声明后新增 `forward_avc_f32`）
- Modify: `h3_dit_w4a8.c:4295-4530`（`forward_av_f32` 实现区，重构为 `forward_avc` + wrapper）
- Test: `tests/test_h3_dit_w4a8_cond.c`
- Modify: `CMakeLists.txt` / `Makefile`

**Interfaces:**
- Consumes: `h3_dit_w4a8_forward_av_f32`（现有）、`h3_dit_w4a8_fill_metal_video_pos`、`h3_dit_w4a8_rope_tables_f32`、`H3_DIT_W4A8_MOD_{VIDEO,TEXT,AUDIO}`
- Produces:

```c
/* Pack [text | cond | audio | video] (Metal keyframe-only segment order).
 * cond_latent: [cond_seq, VIDEO_PATCH] patchified condition rows (VAE encode → patchify).
 * cond_pos_{t,h,w}: [cond_seq] condition RoPE coords (condition_time; see h3_host.c:348).
 * cond_seq=0 → behavior identical to h3_dit_w4a8_forward_av_f32.
 * video_velocity / audio_velocity written only for target rows; cond output discarded. */
int h3_dit_w4a8_forward_avc_f32(
    h3_dit_w4a8 *ctx, const h3_dit_w8a8_core *core, unsigned first_block,
    unsigned n_blocks, float sigma, float audio_sigma,
    const float *video_latent, size_t video_seq,
    const float *audio_latent, size_t audio_seq,
    const float *cond_latent, size_t cond_seq,
    const float *text_hidden, size_t text_seq,
    const float *v_pos_t, const float *v_pos_h, const float *v_pos_w,
    const float *a_pos_t, const float *a_pos_h, const float *a_pos_w,
    const float *c_pos_t, const float *c_pos_h, const float *c_pos_w,
    float spatial_scale,
    float *video_velocity, float *audio_velocity,
    int auto_prefetch, h3_dit_w4a8_forward_stats *stats,
    char *error, size_t error_size);
```

`h3_dit_w4a8_forward_av_f32` 改为 thin wrapper：`cond_seq=0` + NULL cond 指针调 `forward_avc`。

- [ ] **Step 1: 写失败测试（cond_seq=0 时 forward_avc == forward_av bit-for-bit）**

```c
/* tests/test_h3_dit_w4a8_cond.c — 骨架回归 */
#include "h3_dit_w4a8.h"
#include <math.h>
#include <string.h>

static int forward_avc_matches_av(h3_dit_w4a8 *dit, const h3_dit_w4a8_core *core,
                                   size_t video_seq, size_t audio_seq,
                                   size_t text_seq, char *err, size_t es) {
    float *xv = calloc(video_seq * 96, sizeof(float));
    float *xa = calloc(audio_seq * 32, sizeof(float));
    float *vv1 = calloc(video_seq * 96, sizeof(float));
    float *va1 = calloc(audio_seq * 32, sizeof(float));
    float *vv2 = calloc(video_seq * 96, sizeof(float));
    float *va2 = calloc(audio_seq * 32, sizeof(float));
    float *pt = calloc(video_seq, sizeof(float));
    float *ph = calloc(video_seq, sizeof(float));
    float *pw = calloc(video_seq, sizeof(float));
    float *at = calloc(audio_seq, sizeof(float));
    float *ah = calloc(audio_seq, sizeof(float));
    float *aw = calloc(audio_seq, sizeof(float));
    float *th = calloc(text_seq * 5376, sizeof(float));
    int ok = 0;
    for (size_t i = 0; i < video_seq * 96; i++) xv[i] = (float)(i % 7) * 0.01f - 0.03f;
    for (size_t i = 0; i < audio_seq * 32; i++) xa[i] = (float)(i % 5) * 0.01f - 0.02f;
    for (size_t i = 0; i < text_seq * 5376; i++) th[i] = (float)(i % 11) * 0.005f;
    h3_dit_w4a8_fill_metal_video_pos(text_seq, 5, 16, 16, pt, ph, pw, err, es);
    h3_dit_w4a8_fill_metal_audio_pos(text_seq, (int)audio_seq/2, 16, 16, at, ah, aw, err, es);
    int r1 = h3_dit_w4a8_forward_av_f32(dit, core, 0, 4, 0.5f, 0.3f, xv, video_seq,
        xa, audio_seq, th, text_seq, pt, ph, pw, at, ah, aw, 1.0f, vv1, va1, 1, NULL, err, es);
    int r2 = h3_dit_w4a8_forward_avc_f32(dit, core, 0, 4, 0.5f, 0.3f, xv, video_seq,
        xa, audio_seq, NULL, 0, th, text_seq, pt, ph, pw, at, ah, aw, NULL, NULL, NULL,
        1.0f, vv2, va2, 1, NULL, err, es);
    if (!r1 || !r2) goto done;
    for (size_t i = 0; i < video_seq * 96; i++)
        if (fabsf(vv1[i] - vv2[i]) > 1e-6f) { snprintf(err, es, "v mismatch @%zu", i); goto done; }
    for (size_t i = 0; i < audio_seq * 32; i++)
        if (fabsf(va1[i] - va2[i]) > 1e-6f) { snprintf(err, es, "a mismatch @%zu", i); goto done; }
    ok = 1;
done:
    free(xv); free(xa); free(vv1); free(va1); free(vv2); free(va2);
    free(pt); free(ph); free(pw); free(at); free(ah); free(aw); free(th);
    return ok;
}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cmake --build build-cuda -j --target h3_dit_w4a8_cond_test && ./build-cuda/h3_dit_w4a8_cond_test`
Expected: FAIL with "forward_avc_f32 undefined"（链接错误）。

- [ ] **Step 3: 在 `h3_dit_w4a8.h` 加 `forward_avc_f32` 声明**

在 `h3_dit_w4a8.h:431`（`forward_av_f32` 声明结束）后插入上文 Produces 块的 `h3_dit_w4a8_forward_avc_f32` 声明。

- [ ] **Step 4: 在 `h3_dit_w4a8.c` 实现 `forward_avc_f32`（重构 `forward_av`）**

把 `h3_dit_w4a8.c:4295-4530` 的 `h3_dit_w4a8_forward_av_f32` 函数体改名为 `h3_dit_w4a8_forward_avc_f32`，新增 `cond_latent` / `cond_seq` / `c_pos_{t,h,w}` 参数。关键改动：

1. **序列布局**：`seq = text_seq + cond_seq + audio_seq + video_seq`；`cond_off = text_seq`；`audio_off = text_seq + cond_seq`；`video_off = text_seq + cond_seq + audio_seq`（Metal keyframe-only 段序：text → cond → audio → video）。
2. **row_map**：cond 段填 `H3_DIT_W4A8_MOD_VIDEO`（condition 行与 video 同 AdaLN time_row，见 `h3_dit.c:325`）。
3. **RoPE 坐标**：cond 段填 `c_pos_{t,h,w}`（由调用方按 `h3_host.c:348` 的 `condition_time` 构造）；`cond_seq=0` 时跳过。
4. **patch 投影**：cond 段用 `video_patch_w` / `video_patch_b`（与 `H3_SEG_VIDEO` 同源，见 `h3_dit.c:2080-2088`），写入 `hidden[cond_off]`。
5. **final 提取**：final layer 跑全序列，但只把 `video_off..video_off+video_seq` 拷到 `video_velocity`、`audio_off..audio_off+audio_seq` 拷到 `audio_velocity`；cond 段 final 输出丢弃。
6. **`cond_seq=0` 短路**：当 `cond_seq==0` 时，所有 cond 相关循环不执行，`audio_off`/`video_off` 与原 `forward_av` 完全一致 → bit-for-bit 不变。

```c
/* h3_dit_w4a8.c — forward_avc 关键段（示意，省略 host/resident 分支） */
seq = text_seq + cond_seq + audio_seq + video_seq;
cond_off = text_seq;
audio_off = text_seq + cond_seq;
video_off = text_seq + cond_seq + audio_seq;

/* row_map + RoPE 坐标 */
for (i = 0; i < text_seq; i++) { row_map[i] = text_mod; full_pt[i]=(float)i; full_ph[i]=0; full_pw[i]=0; }
for (i = 0; i < cond_seq; i++) {
    row_map[cond_off + i] = (uint32_t)H3_DIT_W4A8_MOD_VIDEO;
    full_pt[cond_off + i] = c_pos_t[i];
    full_ph[cond_off + i] = c_pos_h[i];
    full_pw[cond_off + i] = c_pos_w[i];
}
for (i = 0; i < audio_seq; i++) { row_map[audio_off+i]=H3_DIT_W4A8_MOD_AUDIO; full_pt[audio_off+i]=a_pos_t[i]; ... }
for (i = 0; i < video_seq; i++) { row_map[video_off+i]=H3_DIT_W4A8_MOD_VIDEO; full_pt[video_off+i]=v_pos_t[i]; ... }

/* patch 投影：cond 段用 video_patch_w（同源） */
if (text_seq) memcpy(hidden, text_hidden, text_seq*HIDDEN*sizeof(float));
if (cond_seq)  /* hidden[cond_off] = video_patch_w @ cond_latent + video_patch_b */
    patch_video(core, cond_latent, cond_seq, hidden + cond_off*HIDDEN, ...);
if (audio_seq)  /* hidden[audio_off] = audio_patch_w @ audio_latent + audio_patch_b */
    patch_audio(core, audio_latent, audio_seq, hidden + audio_off*HIDDEN, ...);
if (video_seq)  /* hidden[video_off] = video_patch_w @ video_latent + video_patch_b */
    patch_video(core, video_latent, video_seq, hidden + video_off*HIDDEN, ...);

/* layers forward over full seq */
layers_forward(ctx, 0, n_blocks, t_row, hidden, seq, row_map, cos, sin, hidden2, ...);

/* final: 跑全序列，但只提取 target 段 */
final_video(core, hidden2 + video_off*HIDDEN, video_seq, time_feat, video_velocity, ...);
final_audio(core, hidden2 + audio_off*HIDDEN, audio_seq, time_feat_audio, audio_velocity, ...);
/* cond 段 final 输出丢弃 */
```

- [ ] **Step 5: `forward_av_f32` 改为 wrapper**

```c
int h3_dit_w4a8_forward_av_f32(
    h3_dit_w4a8 *ctx, const h3_dit_w4a8_core *core, unsigned first_block,
    unsigned n_blocks, float sigma, float audio_sigma,
    const float *video_latent, size_t video_seq,
    const float *audio_latent, size_t audio_seq,
    const float *text_hidden, size_t text_seq,
    const float *v_pos_t, const float *v_pos_h, const float *v_pos_w,
    const float *a_pos_t, const float *a_pos_h, const float *a_pos_w,
    float spatial_scale, float *video_velocity, float *audio_velocity,
    int auto_prefetch, h3_dit_w4a8_forward_stats *stats,
    char *error, size_t error_size) {
    return h3_dit_w4a8_forward_avc_f32(
        ctx, core, first_block, n_blocks, sigma, audio_sigma,
        video_latent, video_seq, audio_latent, audio_seq,
        NULL, 0, text_hidden, text_seq,
        v_pos_t, v_pos_h, v_pos_w, a_pos_t, a_pos_h, a_pos_w,
        NULL, NULL, NULL, spatial_scale,
        video_velocity, audio_velocity, auto_prefetch, stats, error, error_size);
}
```

- [ ] **Step 6: 跑测试确认 PASS**

Run: `cmake --build build-cuda -j --target h3_dit_w4a8_cond_test && ./build-cuda/h3_dit_w4a8_cond_test`
Expected: PASS（"forward_avc matches forward_av (cond_seq=0)"）。

- [ ] **Step 7: 跑现有 probe 确认无回归**

Run:
```bash
cmake --build build-cuda -j --target h3_dit_w4a8_sample_probe h3_dit_w4a8_forward_probe
./build-cuda/h3_dit_w4a8_sample_probe --steps 2 --layers 4 --batch 4 "$DIT"
./build-cuda/h3_dit_w4a8_forward_probe --layers 4 --batch 4 --sigma 0.5 "$DIT"
```
Expected: 与重构前输出一致（checksum 比对）。

- [ ] **Step 8: 不 commit（用户要求）**

记录 T1 完成，进入 T2。

---

## Task 2: VAE encoder 权重加载（fp16 checkpoint）

**Files:**
- Create: `h3_vae_encode_f32.h`
- Create: `h3_vae_encode_f32.c`（本 Task 只做权重加载 + free，forward 留给 T3）
- Test: `tests/test_h3_vae_encode_f32.c`（本 Task 只测加载 + tensor 名打印）
- Modify: `CMakeLists.txt` / `Makefile`

**Interfaces:**
- Consumes: `h3_safetensors.h`（`h3_st_header` / `h3_st_tensor`）、`models/comfy-org/vae/minimax_h3_video_vae_fp16.safetensors`
- Produces:

```c
/* h3_vae_encode_f32.h */
#ifndef H3_VAE_ENCODE_F32_H
#define H3_VAE_ENCODE_F32_H

#include "h3_safetensors.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Host F32 VAE encoder weights loaded from the fp16 VAE checkpoint.
 * Mirrors the decoder block structure (h3_vae_int8.h) but with fp16
 * encoder tensors dequantized to F32 on load. Caller frees with
 * h3_vae_encode_weights_free. */
typedef struct h3_vae_encode_weights h3_vae_encode_weights;

/* Open the fp16 VAE checkpoint and load encoder-side tensors into host F32.
 * vae_fp16_path: path to minimax_h3_video_vae_fp16.safetensors.
 * Prints the resolved encoder tensor names to stderr (audit).
 * H3_VAE_ENCODER_DEBUG=1 prints per-tensor byte counts. */
int h3_vae_encode_weights_load(const char *vae_fp16_path,
                                h3_vae_encode_weights **out,
                                char *error, size_t error_size);
void h3_vae_encode_weights_free(h3_vae_encode_weights *w);

/* Accessors used by T3 forward. */
const float *h3_vae_encode_conv_in_w(const h3_vae_encode_weights *w);
const float *h3_vae_encode_conv_in_b(const h3_vae_encode_weights *w);
unsigned h3_vae_encode_n_blocks(const h3_vae_encode_weights *w);
/* block i norm1/norm2/qkv/out/fc1/fc2 weights (F32, host) ... (full accessor
 *  set defined in the .c; header lists the ones T3 needs). */

#ifdef __cplusplus
}
#endif
#endif
```

- [ ] **Step 1: 探查 fp16 VAE checkpoint 的 encoder tensor 名**

Run:
```bash
python3 -c "
from safetensors import safe_open
p='models/comfy-org/vae/minimax_h3_video_vae_fp16.safetensors'
with safe_open(p,'pt') as f:
    for k in f.keys():
        if 'encoder' in k.lower() or k.startswith('enc.'): print(k, f.get_tensor(k).shape, f.get_tensor(k).dtype)
" 2>/dev/null || python3 -c "
import struct,json
p='models/comfy-org/vae/minimax_h3_video_vae_fp16.safetensors'
with open(p,'rb') as f:
    n=struct.unpack('<Q',f.read(8))[0]
    hdr=json.loads(f.read(n).decode())
    for k,v in hdr.items():
        if '__metadata__' not in k and ('enc' in k.lower()):
            print(k, v.get('shape'), v.get('dtype'))
"
```
Expected: 打印 `encoder.*` tensor 名清单（conv_in、transformer_blocks.*、conv_out 等）与 shape。把清单贴到 `docs/superpowers/plans/2026-08-18-f2va-vision-condition-channel.md` 的 T2 备注。

- [ ] **Step 2: 写失败测试（加载 + tensor 名打印）**

```c
/* tests/test_h3_vae_encode_f32.c — T2 子集 */
#include "h3_vae_encode_f32.h"
#include <stdio.h>
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <vae_fp16.safetensors>\n", argv[0]); return 2; }
    h3_vae_encode_weights *w = NULL; char err[512];
    if (!h3_vae_encode_weights_load(argv[1], &w, err, sizeof(err))) {
        fprintf(stderr, "load failed: %s\n", err); return 1;
    }
    printf("n_blocks=%u conv_in_w=%p\n", h3_vae_encode_n_blocks(w),
           (const void *)h3_vae_encode_conv_in_w(w));
    h3_vae_encode_weights_free(w);
    return 0;
}
```

- [ ] **Step 3: 跑测试确认失败**

Run: `cmake --build build-cuda -j --target h3_vae_encode_f32_test && ./build-cuda/h3_vae_encode_f32_test "$VAE_FP16"`
Expected: FAIL with "h3_vae_encode_weights_load undefined"。

- [ ] **Step 4: 实现 `h3_vae_encode_f32.c`（权重加载）**

```c
/* h3_vae_encode_f32.c — T2: load encoder tensors from fp16 VAE checkpoint.
 * 用 h3_safetensors 打开文件，遍历 tensor 名匹配 encoder.* 前缀，
 * fp16 → F32 dequant 到 host malloc。结构体 h3_vae_encode_weights 持有
 * 所有 encoder F32 权重指针 + n_blocks + 各 tensor 的 shape 元数据。
 * 失败时已分配的逐个 free（OOM-safe）。 */
#include "h3_vae_encode_f32.h"
#include "h3_safetensors.h"
#include <stdlib.h>
#include <string.h>

struct h3_vae_encode_weights {
    float *conv_in_w;   /* [C_out, C_in, kT, kH, kW] */
    float *conv_in_b;   /* [C_out] */
    float *conv_out_w;  /* [latent_ch, C, kT, kH, kW] */
    float *conv_out_b;  /* [latent_ch] */
    /* transformer_blocks[i].{norm1,norm2,attn.qkv_w,attn.qkv_b,attn.out_w,
     *   attn.out_b,fc1_w,fc1_b,fc2_w,fc2_b} — F32 host */
    struct { float *norm1,*norm2,*qkv_w,*qkv_b,*out_w,*out_b,*fc1_w,*fc1_b,*fc2_w,*fc2_b; } *blocks;
    unsigned n_blocks;
    /* shape 元数据（T3 forward 需要） */
    int conv_in_c_out, conv_in_c_in, conv_in_kt, conv_in_kh, conv_in_kw;
    int latent_ch, conv_out_c, conv_out_kt, conv_out_kh, conv_out_kw;
};

/* 实现：遍历 header，匹配 "encoder.conv_in.weight" 等键名，
 * h3_safetensors 读取 fp16 原始字节 → cast 到 F32 存入对应指针。
 * H3_VAE_ENCODER_DEBUG=1 时 fprintf stderr 每个 tensor 的 name+bytes。 */
```

- [ ] **Step 5: 跑测试确认 PASS**

Run: `cmake --build build-cuda -j --target h3_vae_encode_f32_test && ./build-cuda/h3_vae_encode_f32_test "$VAE_FP16"`
Expected: PASS，打印 `n_blocks=36 conv_in_w=0x...`（与 decoder block 数一致）。

- [ ] **Step 6: 不 commit**

---

## Task 3: VAE encoder forward（host F32 + tiling）

**Files:**
- Modify: `h3_vae_encode_f32.h`（新增 `h3_vae_encode_f32` forward 声明）
- Modify: `h3_vae_encode_f32.c`（实现 forward + tiling）
- Modify: `tests/test_h3_vae_encode_f32.c`（加 forward 回归）
- Modify: `CMakeLists.txt` / `Makefile`

**Interfaces:**
- Consumes: `h3_vae_encode_weights`（T2）、`h3_ffmpeg_read_image_f32`（`h3.c:1226`）、Metal `h3_video_vae_encode`（参考实现，`h3_video_encoder.h:22`）、`h3_dit_patchify_video`（`h3.c:1320`）
- Produces:

```c
/* Encode channel-major RGB [3,T,H,W] in [0,1] → DiT patch rows [seq, 96].
 * T=1 for first/last frame (h3_video_encoder_latent_t(1)).
 * Spatial axes must be multiples of 16; 256px tiling + 64px overlap
 * (mirror h3_video_vae_encode). out_rows: caller-allocated [seq, 96].
 * seq = latent_t * (H/2) * (W/2) = T*(H/2)*(W/2). */
int h3_vae_encode_f32(const h3_vae_encode_weights *w,
                      const float *pixels_rgb_chw, int frames, int height, int width,
                      float *out_rows, size_t out_elements,
                      char *error, size_t error_size);
```

- [ ] **Step 1: 写失败测试（host encode vs Metal encode 一致性）**

```c
/* tests/test_h3_vae_encode_f32.c — T3 forward 回归 */
/* 读 jpeg → 256x256 → (a) host F32 encode → patch rows → unpack latent;
 *                       (b) Metal h3_video_vae_encode → latent;
 * 比对 max abs diff < 5e-2。Metal 路径仅 macOS 本地可跑；远程 NVIDIA box
 * 跳过 Metal 比对，改为 round-trip：encode → decode → PSNR vs jpeg > 25dB。 */
static int encode_round_trip(const char *vae_fp16, const char *vae_int8,
                              const char *jpeg, char *err, size_t es) {
    /* 1. h3_ffmpeg_read_image_f32(jpeg, 256, 256, STRETCH, pixels, ...) */
    /* 2. h3_vae_encode_f32(w, pixels, 1, 256, 256, rows, ...) */
    /* 3. h3_vae_int8_unpack_dit_latent_f32(rows, 1, 16, 16, latent_chw, ...) */
    /* 4. h3_vae_int8_decode_f32(vae_int8_store, latent_chw, 1, 16, 16, 36, frames, ...) */
    /* 5. PSNR(frames.rgb, pixels) > 25dB */
    return 1; /* 伪代码：实际实现见 step 4 */
}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cmake --build build-cuda -j --target h3_vae_encode_f32_test && ./build-cuda/h3_vae_encode_f32_test "$VAE_FP16" "$VAE_INT8" "$JPEG"`
Expected: FAIL with "h3_vae_encode_f32 undefined"。

- [ ] **Step 3: 实现 encoder forward（单 tile，无 tiling 先跑通）**

```c
/* h3_vae_encode_f32.c — encoder forward（单 tile）。
 * 流程（镜像 decoder h3_vae_int8_decode_f32，反向）：
 *   pixels [3,T,H,W] → conv_in → [C,T,H',W'] (downsample 8x)
 *   → N transformer blocks (temporal + spatial, RoPE, SDPA)
 *   → conv_out → [latent_ch, T, H/8, W/8] → 2x downsample → [24, T, H/16, W/16]
 *   → patchify → [seq, 96]
 * 参考 h3_video_encoder.c 的 Metal 流程（conv_in → blocks → conv_out），
 * 但用 host F32 GEMM（h3_vae_int8_gemm_f32）+ host SDPA（h3_dit_w4a8_sdpa_f32
 * 复用或自写 naive）。
 * 关键：encoder 的 RoPE 与 decoder 完全相同（h3_vae_int8_prepare_rope）。 */
int h3_vae_encode_f32(const h3_vae_encode_weights *w,
                      const float *pixels, int frames, int height, int width,
                      float *out_rows, size_t out_elements, char *error, size_t es) {
    /* 1. conv_in: 3D conv [C_out,3,kT,kH,kW] stride (1,2,2) → [C,T,H/2,W/2] */
    /* 2. 36 transformer blocks: temporal SDPA + spatial SDPA + FFN */
    /* 3. conv_out: 3D conv [24,C,kT,kH,kW] stride (1,2,2) → [24,T,H/4,W/4] */
    /*    （encoder 总下采样 8x：conv_in 2x + conv_out 2x + mid 2x? 查 Metal） */
    /* 4. patchify: h3_dit_patchify_video(latent.values, 24, T, H/8, W/8, out_rows, ...) */
    return 1;
}
```

**注意**：encoder 的下采样倍数必须从 Step 1 探查的 tensor shape 反推（conv_in kernel + stride、conv_out kernel + stride）。Metal `h3_video_encoder.c` 是权威参考；若 host 实现与 Metal 倍数不一致，以 Metal 为准。

- [ ] **Step 4: 跑测试确认 PASS（round-trip PSNR）**

Run: `cmake --build build-cuda -j --target h3_vae_encode_f32_test && ./build-cuda/h3_vae_encode_f32_test "$VAE_FP16" "$VAE_INT8" "$JPEG"`
Expected: PASS（PSNR > 25dB）。若不达标，对比 Metal `h3_video_vae_encode` 输出定位差异。

- [ ] **Step 5: 加 256px tiling + 64px overlap**

```c
/* 复用 h3_video_encoder.c 的 tile 切分 + overlap blend（stitch_latents）。
 * 单 tile 接口 h3_vae_encode_f32_tile(pixels_tile, ...) → latent_tile；
 * 外层循环遍历 256px tile，blend 到 full latent。
 * H3_VAE_ENCODER_TILE=256 / H3_VAE_ENCODER_OVERLAP=64 可调。 */
```

- [ ] **Step 6: 跑 864×480 测试（多 tile）**

Run: `./build-cuda/h3_vae_encode_f32_test "$VAE_FP16" "$VAE_INT8" "$JPEG_864x480"`
Expected: PASS（PSNR > 25dB，多 tile blend 后无明显接缝）。

- [ ] **Step 7: 不 commit**

---

## Task 4: CUDA VAE encoder（性能加速，可后置）

**依赖**：T3。**可后置**——T5/T6/T7/T8 可先用 host F32 encoder 推进，T4 仅做加速。

**Files:**
- Create: `h3_vae_int8_encode_cuda.h`
- Create: `h3_vae_int8_encode_cuda.c`
- Modify: `CMakeLists.txt` / `Makefile`

**Interfaces:**
- Consumes: `h3_vae_encode_weights`（T2，host F32 权重上传到 device）、`h3_vae_int8_decode_cuda` 的 GEMM/SDPA 基础设施（`h3_gpu_cuda.h`）、`h3_cuda_sdpa_f16`
- Produces:

```c
/* CUDA VAE encoder: pixels → patch rows on device.
 * weights: host F32（T2）一次性 H2D 到 device 常驻（H3_VAE_ENCODER_W_RESIDENT=1）。
 * 复用 h3_vae_int8_decode_cuda 的 GEMM/SDPA 调度，仅替换 encoder 权重 + conv_in/conv_out。 */
int h3_vae_int8_encode_cuda(h3_cuda *cuda, const h3_vae_encode_weights *w,
                             const float *pixels_rgb_chw, int frames, int height,
                             int width, float *out_rows, size_t out_elements,
                             char *error, size_t error_size);
```

- [ ] **Step 1: 写失败测试（CUDA encode vs host F32 一致性）**

```c
/* tests/test_h3_vae_encode_cuda.c — CUDA encode max abs diff vs host < 1e-3 */
/* 读 jpeg 256x256 → host encode rows1 → CUDA encode rows2 → 比对 */
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cmake --build build-cuda -j --target h3_vae_encode_cuda_test && ./build-cuda/h3_vae_encode_cuda_test "$VAE_FP16" "$JPEG"`
Expected: FAIL with "h3_vae_int8_encode_cuda undefined"。

- [ ] **Step 3: 实现 CUDA encoder forward**

```c
/* h3_vae_int8_encode_cuda.c — 复用 h3_vae_int8_decode_cuda 的 GEMM/SDPA 调度。
 * 1. weights H2D 到 device（常驻，首次 upload）
 * 2. conv_in: im2col + cuBLAS GEMM（3D conv → GEMM）
 * 3. 36 blocks: temporal SDPA (cudnn) + spatial SDPA + FFN（cuBLAS）
 * 4. conv_out: im2col + cuBLAS GEMM
 * 5. patchify on device → D2H out_rows
 * 关键：encoder 与 decoder 共享 block 结构，仅权重不同。 */
```

- [ ] **Step 4: 跑测试确认 PASS**

Run: `cmake --build build-cuda -j --target h3_vae_encode_cuda_test && ./build-cuda/h3_vae_encode_cuda_test "$VAE_FP16" "$JPEG"`
Expected: PASS（max abs diff < 1e-3；单帧 encode < 200ms @ 256px）。

- [ ] **Step 5: 不 commit**

---

## Task 5: CLI `--first-frame` / `--last-frame`（官方语义）

**Files:**
- Modify: `tools/h3_cuda_generate.c:339-340`（新增 `first_frame` / `last_frame` 路径变量）
- Modify: `tools/h3_cuda_generate.c:467-475`（CLI 解析新增 `--first-frame` / `--last-frame`）
- Modify: `tools/h3_cuda_generate.c:545-552`（usage 文本）
- Modify: `tools/h3_cuda_generate.c:883-946`（cond_rows 打包 + 调用 `forward_avc`）
- Modify: `CMakeLists.txt` / `Makefile`

**Interfaces:**
- Consumes: `h3_vae_encode_f32`（T3，host F32；T4 完成后切 CUDA）、`h3_dit_w4a8_forward_avc_f32`（T1）、`h3_ffmpeg_read_image_f32`（`h3.c:1226`）、`h3_dit_patchify_video`（`h3.c:1320`）、`h3_video_encoder_latent_t`（`h3_host.h:97`）、`h3_latent_canvas`（`h3_host.h:99`）
- Produces: CLI `--first-frame IMG` / `--last-frame IMG`（官方 condition 语义，非 lite img2img）。

- [ ] **Step 1: 写失败测试（CLI --first-frame 生成 cond_rows 并传入 forward_avc）**

```c
/* tests/test_h3_cuda_generate_cond_cli.c — smoke test
 * 跑 h3_cuda_generate --first-frame input/test_1_first_frame.jpeg --steps 2 --layers 4
 * 检查 generate.log 含 "cond_rows=120"（256x256: 1*8*8=64? 实际 latent_t=1, H/2=8, W/2=8 → 64 rows）
 * 检查 forward_avc 被调用（非 forward_av） */
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cmake --build build-cuda -j --target h3_cuda_generate_cond_cli_test && ./build-cuda/h3_cuda_generate_cond_cli_test "$DIT" "$VAE_FP16" "$JPEG"`
Expected: FAIL with "--first-frame: unknown option"。

- [ ] **Step 3: 加 CLI 解析**

```c
/* tools/h3_cuda_generate.c — 在 :467 附近加 */
} else if (!strcmp(argv[iarg], "--first-frame") && iarg + 1 < argc) {
    first_frame_path = argv[++iarg];
} else if (!strcmp(argv[iarg], "--last-frame") && iarg + 1 < argc) {
    last_frame_path = argv[++iarg];
}
/* usage 文本加：
 *  --first-frame IMG  官方 F2VA condition：jpeg → VAE encode → DiT condition rows
 *                    (H3_SEG_COND, condition_time=text_len)。非 lite img2img。
 *  --last-frame  IMG  尾帧 condition (condition_time=text_len+video_span-frame_rescale)。
 *  与 --init-first-frame / --lock-first-frame 互斥（lite 路径）。 */
```

- [ ] **Step 4: 加 cond_rows 打包逻辑**

```c
/* tools/h3_cuda_generate.c — 在 latent 初始化后、DiT 循环前 */
float *cond_rows = NULL; size_t cond_elements = 0; size_t cond_seq = 0;
float *c_pos_t=NULL,*c_pos_h=NULL,*c_pos_w=NULL;
int keyframes[2]; int keyframe_count = 0;
if (first_frame_path) keyframes[keyframe_count++] = 0;
if (last_frame_path)  keyframes[keyframe_count++] = frame_count - 1;

if (keyframe_count) {
    /* 每个 keyframe: jpeg → 256x256 (h3_latent_canvas → 16x16 latent) → VAE encode → patchify */
    cond_seq = (size_t)keyframe_count * (latent_h/2) * (latent_w/2); /* T=1 */
    cond_elements = cond_seq * 96;
    cond_rows = malloc(cond_elements * sizeof(float));
    c_pos_t = malloc(cond_seq * sizeof(float));
    c_pos_h = malloc(cond_seq * sizeof(float));
    c_pos_w = malloc(cond_seq * sizeof(float));
    /* 逐 keyframe:
     *   h3_ffmpeg_read_image_f32(path, render_w, render_h, FIT_STRETCH, pixels, ...)
     *   h3_vae_encode_f32(w, pixels, 1, render_h, render_w, rows_i, ...)
     *   memcpy cond_rows + offset
     *   condition_time = (kf==0) ? text_len : text_len + video_span - frame_rescale
     *   fill c_pos_{t,h,w}[offset..] 用 h3_dit_w4a8_fill_metal_video_pos 的单帧版
     *     （t=condition_time, h/w 同 video 但只 1 帧） */
}

/* DiT 循环：调 forward_avc 而非 forward_av */
if (cond_seq) {
    h3_dit_w4a8_forward_avc_f32(dit, core, 0, layers, sigma, audio_sigma,
        video_lat, video_seq, audio_rows, audio_seq,
        cond_rows, cond_seq, text_hidden, context,
        v_pt, v_ph, v_pw, a_pt, a_ph, a_pw, c_pos_t, c_pos_h, c_pos_w,
        spatial_scale, video_vel, audio_vel, 1, &fst, error, sizeof(error));
} else {
    h3_dit_w4a8_forward_av_f32(...); /* 原路径不变 */
}
```

- [ ] **Step 5: 跑测试确认 PASS**

Run: `cmake --build build-cuda -j --target h3_cuda_generate && ./build-cuda/h3_cuda_generate --first-frame "$JPEG" --steps 2 --layers 4 "$DIT" "$VAE_FP16" "$TE"`
Expected: generate.log 含 `cond_rows=...`；输出 mp4 无 crash。

- [ ] **Step 6: 跑无 condition 路径确认无回归**

Run: `./build-cuda/h3_cuda_generate --steps 2 --layers 4 "$DIT" "$VAE_FP16" "$TE"`（无 `--first-frame`）
Expected: 输出与 T5 前完全一致（checksum 比对 latent）。

- [ ] **Step 7: 不 commit**

---

## Task 6: 首帧 condition 端到端（F2VA f0 对齐）

**Files:**
- Modify: `tools/h3_cuda_generate.c`（首帧 condition_time 计算 + 验证输出）
- Create: `out/verify_f2va_cond_f/README.txt`
- Modify: `Makefile`（`verify-f2va-cond-f` target）

**Interfaces:**
- Consumes: T5（CLI `--first-frame`）、`h3_video_encoder_latent_t(1)`（`h3_host.h:97`）、`h3_dit_frame_rescale = 5/3`（`h3_dit_w4a8.c:972`）、`h3_dit_frame_per_token[5]`（`:971`）
- Produces: `out/verify_f2va_cond_f/{out.mp4, frame_0000.png, frame_0003.png, frame_0007.png, metrics.txt}`

- [ ] **Step 1: 写验证脚本（首帧 condition_time = text_len）**

```bash
# Makefile target verify-f2va-cond-f
verify-f2va-cond-f:
	./build-cuda/h3_cuda_generate \
	    --first-frame input/test_1_first_frame.jpeg \
	    --prompt-file input/test_1_prompt.txt \
	    --steps 30 --layers 50 --cfg 5 --reuse 0 \
	    --width 864 --height 480 --seconds 5 \
	    "$$DIT" "$$VAE_FP16" "$$TE" \
	    --out out/verify_f2va_cond_f/out.mp4 \
	    --dump-dir out/verify_f2va_cond_f 2>&1 | tee out/verify_f2va_cond_f/generate.log
	ffmpeg -y -i out/verify_f2va_cond_f/out.mp4 -vf "select='eq(n,0)+eq(n,3)+eq(n,7)'" \
	    -vsync vfr out/verify_f2va_cond_f/frame_%03d.png
	python3 scripts/psnr.py out/verify_f2va_cond_f/frame_000.png \
	    input/test_1_first_frame.jpeg > out/verify_f2va_cond_f/psnr_f0.txt
```

- [ ] **Step 2: 跑生成**

Run: `make verify-f2va-cond-f`
Expected: 生成 out.mp4 + frame_000/003/007.png；generate.log 含 `cond_rows=...` 且 `forward_avc` 被调用。

- [ ] **Step 3: 检查 f0 对齐**

Run: `cat out/verify_f2va_cond_f/psnr_f0.txt`
Expected: PSNR > 28dB（f0 应接近 jpeg 的 VAE round-trip）。

- [ ] **Step 4: 人工检查后续帧**

打开 `out/verify_f2va_cond_f/frame_003.png` / `frame_007.png`。
Expected: 骑士（或 prompt 描述主体）可认；不闪；主题跟提示词。

- [ ] **Step 5: 与 lite img2img 对比**

```bash
./build-cuda/h3_cuda_generate --init-latent ... --init-first-frame ... # lite 对照
```
Expected: cond 路径 frame_003 主体可认度 > lite 路径。

- [ ] **Step 6: 不 commit**

---

## Task 7: 尾帧 condition

**Files:**
- Modify: `tools/h3_cuda_generate.c`（尾帧 condition_time 计算）
- Create: `out/verify_f2va_cond_l/README.txt`
- Modify: `Makefile`（`verify-f2va-cond-l` target）

**Interfaces:**
- Consumes: T5、`h3_video_span_sum`（`h3_host.c:354`，等价 `sum(frame_per_token[i%5]) * frame_rescale`）
- Produces: `out/verify_f2va_cond_l/{out.mp4, frame_last.png, metrics.txt}`

- [ ] **Step 1: 写验证脚本（尾帧 condition_time = text_len + video_span - frame_rescale）**

```bash
verify-f2va-cond-l:
	./build-cuda/h3_cuda_generate \
	    --last-frame input/test_1_last_frame.jpeg \
	    --prompt-file input/test_1_prompt.txt \
	    --steps 30 --layers 50 --cfg 5 --reuse 0 \
	    --width 864 --height 480 --seconds 5 \
	    "$$DIT" "$$VAE_FP16" "$$TE" \
	    --out out/verify_f2va_cond_l/out.mp4 2>&1 | tee out/verify_f2va_cond_l/generate.log
	ffmpeg -y -i out/verify_f2va_cond_l/out.mp4 -vf "select='eq(n,119)'" \
	    -vsync vfr out/verify_f2va_cond_l/frame_last.png
	python3 scripts/psnr.py out/verify_f2va_cond_l/frame_last.png \
	    input/test_1_last_frame.jpeg > out/verify_f2va_cond_l/psnr_last.txt
```

- [ ] **Step 2: 跑生成**

Run: `make verify-f2va-cond-l`
Expected: out.mp4 生成；末帧 PSNR > 28dB。

- [ ] **Step 3: 检查首→尾插值连贯**

打开 `frame_000.png` / `frame_060.png` / `frame_last.png`。
Expected: 从首帧（噪声生成）到尾帧（condition 锁定）平滑过渡；无突变。

- [ ] **Step 4: 不 commit**

---

## Task 8: 首尾帧同时（keyframes[2]）

**Files:**
- Modify: `tools/h3_cuda_generate.c`（keyframes[2] 打包：两 cond 段，condition_time 分别为 text_len 与 text_len+video_span-frame_rescale）
- Create: `out/verify_f2va_cond_fl/README.txt`
- Modify: `Makefile`（`verify-f2va-cond-fl` target）

**Interfaces:**
- Consumes: T6 + T7
- Produces: `out/verify_f2va_cond_fl/{out.mp4, frame_000.png, frame_last.png, metrics.txt}`

- [ ] **Step 1: 写验证脚本（--first-frame + --last-frame 同时）**

```bash
verify-f2va-cond-fl:
	./build-cuda/h3_cuda_generate \
	    --first-frame input/test_1_first_frame.jpeg \
	    --last-frame  input/test_1_last_frame.jpeg \
	    --prompt-file input/test_1_prompt.txt \
	    --steps 30 --layers 50 --cfg 5 --reuse 0 \
	    --width 864 --height 480 --seconds 5 \
	    "$$DIT" "$$VAE_FP16" "$$TE" \
	    --out out/verify_f2va_cond_fl/out.mp4 2>&1 | tee out/verify_f2va_cond_fl/generate.log
	ffmpeg -y -i out/verify_f2va_cond_fl/out.mp4 -vf "select='eq(n,0)+eq(n,60)+eq(n,119)'" \
	    -vsync vfr out/verify_f2va_cond_fl/frame_%03d.png
	python3 scripts/psnr.py out/verify_f2va_cond_fl/frame_000.png input/test_1_first_frame.jpeg > out/verify_f2va_cond_fl/psnr_f0.txt
	python3 scripts/psnr.py out/verify_f2va_cond_fl/frame_119.png input/test_1_last_frame.jpeg > out/verify_f2va_cond_fl/psnr_last.txt
```

- [ ] **Step 2: 跑生成**

Run: `make verify-f2va-cond-fl`
Expected: out.mp4 生成；generate.log 含 `cond_rows=2*...`（两段 cond）；首末帧 PSNR > 28dB。

- [ ] **Step 3: 检查中段插值**

打开 `out/verify_f2va_cond_fl/frame_060.png`。
Expected: 首末帧之间的中间状态，主体连贯；无突变 / 无闪烁。

- [ ] **Step 4: 不 commit**

---

## Task 9: 回归 + 验证

**Files:**
- Create: `out/verify_f2va_cond/README.txt`（总验证说明）
- Create: `out/verify_f2va_cond/metrics_summary.txt`（汇总）
- Modify: `Makefile`（`verify-f2va-cond-all` target）

**Interfaces:**
- Consumes: T6 + T7 + T8
- Produces: `out/verify_f2va_cond/metrics_summary.txt`

- [ ] **Step 1: 写 T2V/T2AV 无 condition 回归测试**

```bash
# 确认无 --first-frame / --last-frame 时输出与 condition channel 引入前 bit-for-bit 一致
verify-regression-nocond:
	./build-cuda/h3_cuda_generate --steps 8 --layers 50 --cfg 5 --reuse 0 \
	    --width 864 --height 480 --seconds 5 \
	    "$$DIT" "$$VAE_FP16" "$$TE" \
	    --out out/verify_f2va_cond/t2v_nocond.mp4 \
	    --dump-dir out/verify_f2va_cond/t2v_nocond 2>&1 | tee out/verify_f2va_cond/t2v_nocond.log
	# 比对 latent checksum 与历史基线（out/verify2/v1_pianist 等）
	python3 scripts/compare_latent.py out/verify_f2va_cond/t2v_nocond/latent.raw \
	    out/verify2/v1_pianist/latent.raw >> out/verify_f2va_cond/metrics_summary.txt
```

- [ ] **Step 2: 跑 T2V 回归**

Run: `make verify-regression-nocond`
Expected: latent checksum 与基线一致（max abs diff == 0）；mp4 视觉无变化。

- [ ] **Step 3: 跑 T2AV 回归（带 audio）**

```bash
verify-regression-t2av:
	./build-cuda/h3_cuda_generate --steps 8 --layers 50 --cfg 5 --reuse 0 \
	    --pack-audio --audio-t 200 \
	    --width 864 --height 480 --seconds 5 \
	    "$$DIT" "$$VAE_FP16" "$$TE" "$$AUDIO_VAE" \
	    --out out/verify_f2va_cond/t2av_nocond.mp4 2>&1 | tee out/verify_f2va_cond/t2av_nocond.log
```
Expected: 音视频输出与基线一致。

- [ ] **Step 4: 汇总 F2VA condition 验证**

```bash
verify-f2va-cond-all: verify-f2va-cond-f verify-f2va-cond-l verify-f2va-cond-fl
	@echo "=== F2VA Vision Condition Channel Verification ===" > out/verify_f2va_cond/metrics_summary.txt
	@echo "T6 first-frame:  $(cat out/verify_f2va_cond_f/psnr_f0.txt)" >> out/verify_f2va_cond/metrics_summary.txt
	@echo "T7 last-frame:   $(cat out/verify_f2va_cond_l/psnr_last.txt)" >> out/verify_f2va_cond/metrics_summary.txt
	@echo "T8 first+last:   f0=$(cat out/verify_f2va_cond_fl/psnr_f0.txt) last=$(cat out/verify_f2va_cond_fl/psnr_last.txt)" >> out/verify_f2va_cond/metrics_summary.txt
	@echo "T9 regression:   t2v=$(grep -c PASS out/verify_f2va_cond/t2v_nocond.log) t2av=$(grep -c PASS out/verify_f2va_cond/t2av_nocond.log)" >> out/verify_f2va_cond/metrics_summary.txt
	cat out/verify_f2va_cond/metrics_summary.txt
```

- [ ] **Step 5: 跑全量验证**

Run: `make verify-f2va-cond-all`
Expected: metrics_summary.txt 全部达标：
- T6 f0 PSNR > 28dB；后续帧骑士可认；不闪
- T7 末帧 PSNR > 28dB；首→尾连贯
- T8 首末帧 PSNR > 28dB；中段插值连贯
- T9 T2V/T2AV 无 condition 输出与基线一致

- [ ] **Step 6: 人工最终检查**

打开 `out/verify_f2va_cond_f/frame_003.png` 与 `out/verify_f2va_test1_locked/frame_0003.png`（lite 基线）对比。
Expected: cond 路径主体明显更清晰、更稳定；主题跟提示词。

- [ ] **Step 7: 不 commit（用户明确要求）**

---

## 任务总表（文末）

| ID | 名称 | 依赖 | 预估 | DoD 摘要 |
|----|------|------|------|----------|
| T1 | DiT forward 骨架扩展（`forward_avc`，空 condition 不改行为） | — | 3h | `forward_avc` 落地；`cond_seq=0` 时 `forward_av` 输出 bit-for-bit 不变；现有 probe/test PASS |
| T2 | VAE encoder 权重加载（fp16 checkpoint） | — | 2h | 从 fp16 VAE 加载 encoder 权重到 host F32；tensor 名清单打印；OOM-safe free |
| T3 | VAE encoder forward（host F32 + tiling） | T2 | 4h | jpeg → latent；与 Metal `h3_video_vae_encode` 输出 max abs diff < 5e-2（256px 单 tile） |
| T4 | CUDA VAE encoder（性能加速，可后置） | T3 | 4h | CUDA encode vs host F32 max abs diff < 1e-3；单帧 encode < 200ms（256px） |
| T5 | CLI `--first-frame` / `--last-frame`（官方语义） | T1, T3 | 2h | `--first-frame IMG` 生成 cond_rows 并传入 `forward_avc`；无 condition 时 CLI 行为不变 |
| T6 | 首帧 condition 端到端（F2VA f0 对齐） | T5 | 2h | f0 PSNR vs jpeg > 28dB；后续帧骑士可认；不闪 |
| T7 | 尾帧 condition | T5 | 1.5h | 末帧 PSNR vs jpeg > 28dB；首→尾插值连贯 |
| T8 | 首尾帧同时（keyframes[2]） | T6, T7 | 1.5h | 首尾两 cond 段；首末帧对齐；中段插值 |
| T9 | 回归 + 验证 | T6, T7, T8 | 2h | T2V/T2AV 无 condition 输出不变；F2VA 主题跟提示词；metrics_summary 落盘 |

**总预估**：22h（T4 可后置则 18h）。

---

## Self-Review

**1. Spec coverage（调研确认项 → Task 映射）：**
- CUDA 视频 VAE 编码器 → T2（权重）+ T3（host forward）+ T4（CUDA 加速）✓
- DiT 序列布局扩展（`img_cond_rows` 段、`condition_time` RoPE、`row_map`、attention 语义）→ T1 ✓
  - attention mask：condition 行参与 attention 不接收速度 → T1 Step 4（final 提取只取 target 段，cond 输出丢弃；无显式 mask，与 Metal 一致）✓
- CLI `--first-frame` / `--last-frame`（官方语义）→ T5 ✓
- 回归保护（T2V/T2AV 不带 condition 行为不变）→ T1（bit-for-bit）+ T5 Step 6 + T9 Step 1-3 ✓
- 首尾帧同时支持（keyframes[2]）→ T8 ✓
- 验证（f0 对齐 jpeg + 后续帧骑士可认 + 不闪 + 主题跟提示词）→ T6 Step 3-5 + T9 Step 5-6 ✓

**2. Placeholder scan：**
- T3 Step 3 的 encoder forward 流程含"查 Metal"备注——这是必要的对齐步骤，非 placeholder（Metal `h3_video_encoder.c` 是权威参考）。✓
- T5 Step 4 的 `h3_dit_w4a8_forward_av_f32(...)` 省略号是原路径不变示意，非新代码 placeholder。✓
- 无 "TBD" / "implement later" / "add error handling" 等空话。✓

**3. Type consistency：**
- `h3_dit_w4a8_forward_avc_f32` 签名在 T1 Produces、T1 Step 4 调用、T5 Step 4 调用、T1 Step 5 wrapper 中一致 ✓
- `h3_vae_encode_weights` / `h3_vae_encode_f32` 在 T2 Produces、T3 Produces、T4 Produces、T5 Step 4 调用中一致 ✓
- `cond_seq` / `cond_rows` / `c_pos_{t,h,w}` 命名在 T1/T5/T6/T7/T8 一致 ✓
- `H3_DIT_W4A8_MOD_VIDEO` 用于 condition 行（与 Metal `h3_dit.c:325` 一致）✓

**4. 依赖排序：**
- T1（DiT 骨架）与 T2（VAE 权重）无依赖，可并行 ✓
- T3 依赖 T2 ✓
- T5 依赖 T1 + T3 ✓
- T6/T7 依赖 T5 ✓
- T8 依赖 T6 + T7 ✓
- T9 依赖 T6/T7/T8 ✓
- T4 可后置（T5-T8 先用 host F32 encoder）✓
