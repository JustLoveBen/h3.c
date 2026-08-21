# H3 Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不破坏 Metal、不破坏无 condition CUDA T2V 的前提下，按官方 packed 语义把 CUDA 旁路从文生竖切到 F2VA / L2VA / FL2VA / Ref2VA，速度优化单独立项。

**Architecture:** 先冻 T2V 对照面，再让 CUDA DiT 在 `cond_seq=0` 时与今日 `forward_av` bit-for-bit，再共享 VAE encode 与 vision/multimodal TE，最后按首帧 → 尾帧 → 首尾 → 参考接线。禁止用 `--init-latent` lite 冒充任一条件模态。

**Tech Stack:** C11；Metal 为金标准（只读对照）；CUDA 12 / cuDNN SDPA / INT8 ConvRot DiT / NVFP4 TE / fp16 Video VAE encoder 权 / INT8 或 fp16 Video VAE decode / Audio VAE FP32。构建：Darwin `Makefile`（不动默认路径）；Linux `cmake -DH3_GPU_BACKEND=cuda`（`docs/nvidia_build.md`）。

**Spec:**
- 架构真源：[`2026-08-18-h3-generation-architecture.md`](2026-08-18-h3-generation-architecture.md)
- 竖切路线图：[`2026-08-18-h3-generation-roadmap.md`](2026-08-18-h3-generation-roadmap.md)
- 算力账本（G3-lite 以降级为准）：`docs/nvidia_optimization_plan.md` §15 / §22
- 权契约 / 构建：`docs/nvidia_quant_contract.md`、`docs/nvidia_build.md`
- 速度计划（**不要在 A–H 执行**）：[`2026-08-18-nvidia-accel-r1-r3.md`](2026-08-18-nvidia-accel-r1-r3.md)
- 旧 F2VA 计划仅作接口形状参考，**禁止按其 Task 1–9 当执行序**：[`2026-08-18-f2va-vision-condition-channel.md`](2026-08-18-f2va-vision-condition-channel.md)

## Global Constraints

- 不改 Metal 生产语义（`h3.c` / `h3_host.c` / `h3_dit.c` / `h3_video_encoder.c` / `h3_vision_encoder.c` / `h3_multimodal.c` 只读对照）。允许的例外只有：CUDA 侧无法链接 Darwin-only 符号时，**新建**可移植文件，而不是改 Metal 数学。
- 无 condition 时 CUDA DiT **bit-for-bit** 不变（第一道门）。`cond_seq=0` 的 `forward_avc` 必须与现 `h3_dit_w4a8_forward_av_f32` 逐 bit 相同。
- 主题默认：INT8 ConvRot DiT + 修复 NVFP4 TE（`hi_first` + unswizzle + `pre_quant`）+ cuDNN SDPA；cfg=5；reuse / magcache / Sage / token-reduction **默认关**。W4A8 **禁止**当主题默认。
- 禁止 `--init-latent` / `--lock-first-frame` / `--init-first-frame` / 调 `img_w` 冒充 F2VA / L2VA / FL2VA / Ref2VA。这些开关最多保留为 debug，help 必须写「非官方 condition」。
- A/B 完成前 **不要**把 `--first-frame` 做成产品默认。
- 速度（R1–R3、主题 ≤35s）= **阶段 S**，不与 condition 缠在一起；不得改 pack / AdaLN 换速度。
- 不 git commit（除非用户当轮明确要求）；密码不上库；不把「再跑一条 4090 碰运气」当收口。
- CUDA 文档与 CLI 必须自称 **旁路**，不得写成已替换 Metal `h3_generate`。

---

## 任务总表

| 阶段 | 名称 | 依赖 | 预估 | DoD |
| --- | --- | --- | --- | --- |
| **A** | T2V 回归门 | — | 0.5d | 主题命令文档化；`forward_av` 自洽夹具可跑；W4A8 不是主题锚点 |
| **B** | 官方 condition 骨架 | A | 2d | `forward_avc`；`cond_seq=0` bit-for-bit；`cond_seq>0` 时 Euler 不写 cond；AdaLN 0.999 |
| **C** | 共享 VAE encode | B 建议合入后；E 前必须 | 2d | jpeg/帧 → `[24,T,H,W]` → patchify 96；相对 Metal encoder 阈值内；**不宣称 F2VA** |
| **D** | vision + multimodal TE | E 前必须；可与 C 部分∥ | 2.5d | 无图时 TE 与 A 一致；有图含 `<Picture n>`；不回退 G1 契约 |
| **E** | F2VA 竖切 | B+C+D | 1d | 官方 `--first-frame`；A 门仍过；f0 对齐且后续可认；lite 降为 debug |
| **F** | L2VA 竖切 | E | 0.5d | `--last-frame`；cover-crop；末帧对齐；第 0 帧不是尾帧复制 |
| **G** | FL2VA 竖切 | E+F | 0.5d | 两段 COND；首尾同时对齐；中段插值；与 ref 互斥 |
| **H1** | Ref2VA image | B+C+D；建议 G 后 | 1.5d | 切 Ref ckpt；`--ref-image`；与 keyframes 互斥 |
| **H2** | Ref2VA audio | H1 | 0.75d | `--ref-audio` 须配图/视频；接已有 encode；目标音轨仍去噪 |
| **H3** | Ref2VA video | H1 | 1.5d | `--ref-video`；因果 encode；vision 两帧 + timestamps |
| **S** | 速度 | 功能宣称完成之后；A 持续为回归 | 见速度计划 | 不改 pack/AdaLN；橙猫/水下仍 PASS；≤35s 失败不得回滚 condition |

```text
A 回归门
 └── B condition 骨架（空 cond = 今日 T2VA）
      ├── C VAE encode payload
      └── D vision + multimodal TE
           ├── E F2VA
           ├── F L2VA
           ├── G FL2VA
           └── H Ref2VA（H1 image → H2 audio → H3 video）
S 速度 ── 与 B–H 代码隔离；H 完成后再作为默认质量路径收口
```

竖切原则：**不要**「先调 img_w → 再加 `--first-frame` → 再补 encoder」。E/F/G/H 只接线与验收；数学在 B–D。

---

## File Structure

| 文件 | 职责 | 本计划 |
| --- | --- | --- |
| `h3_dit_w4a8.h` / `h3_dit_w4a8.c` | CUDA DiT pack / AdaLN / RoPE / Euler | 改：B 起 `forward_avc`；`forward_av` 变 thin wrapper |
| `h3_dit_int8.c` | 主题默认 INT8 ConvRot 权 | 只读；继续走同一 `forward_*` |
| `tools/h3_cuda_generate.c` | CUDA 旁路 CLI | A 只读；E 起官方旗标；lite 降级文案 |
| `tests/test_h3_dit_w4a8_cond.c` | cond 骨架回归 | 新建（A 自洽，B 扩 avc） |
| `h3_vae_encode_f32.h` / `h3_vae_encode_f32.c` | 共享视觉 encode（host F32） | 新建（C） |
| `tests/test_h3_vae_encode_f32.c` | encode / patchify / augment | 新建（C） |
| `h3_vae_int8_encode_cuda.*` | encode CUDA 加速 | C 附录，**不是 S** |
| `h3_vision_cuda.h` / `h3_vision_cuda.c` | Qwen vision tower（先 host F32） | 新建（D） |
| `h3_multimodal_cuda.h` / `h3_multimodal_cuda.c` | `<Picture n>` presentation + span | 新建（D） |
| `h3_nvfp4.c` / `h3_nvfp4.h` | TE 层 | D：无图路径不变；有图走 multimodal inject |
| `h3_token_ids.c` | 纯文本 ids | 保持纯文本；D 在其外拼接 Picture 前缀 |
| `h3_host.c` / `h3_dit_schedule.c` | Metal layout / AdaLN 0.999 | **只读**；CUDA 复刻公式，不链接 Accelerate |
| `h3_ffmpeg.c` | 读图 STRETCH/COVER | E/F 调用已有 API |
| `h3_audio_vae_cuda.c` | 参考音频 encode | H2 接入 generate（probe 已有） |
| `CMakeLists.txt` | CUDA 目标 | 加测试与新 .c；**不改** Darwin `Makefile` 默认 |
| `docs/nvidia_optimization_plan.md` | 算力账本 | E 后把 G3-lite 表述降级为 debug（文档，非代码） |

CUDA **不要**链接整个 `h3_host.c`（`#include <Accelerate/Accelerate.h>`）或 `h3_dit.c` / `h3_video_encoder.c`（Metal GPU）。需要的公式抄到 CUDA 可移植文件，并用 `tests/test_h3.c` 的金数字做对照。

---

## 共享常量与冻结接口

主题回归锚点（P3，`out/verify_r1r3_p3_vae/`；**不要为碰运气重跑 ≤35s**）：

| 项 | 值 |
| --- | --- |
| 分辨率 / 时长 | 864×480，5s，`--frames 17` → latent `7×30×54` |
| DiT / TE / VAE | `--layers 50 --te-layers 50 --vae-blocks 36` |
| 采样 | `--steps 8 --cfg 5`；reuse/magcache/token-reduction **关** |
| SDPA | `H3_CUDA_SDPA_BACKEND=cudnn` / `--sdpa-backend cudnn` |
| 橙猫 | `--seed 17`，`out/verify2/tokens/v4_orange_cat.txt`（20 tok） |
| 水下 | `--seed 91`，`out/verify_official_prompt/b_underwater.ids`（55 tok） |
| 成片对照 | `out/verify_theme_ok/` 与 `out/verify_r1r3_p3_vae/{cat,uw}/` |
| latent 指纹 | 橙猫 `mean=-0.1121 std=1.285`；水下 `mean=-0.1058 std=1.679` |
| DiT 权 | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` |
| TE 权 | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` |
| Video VAE decode | 主题已 PASS 的 INT8/fp16 路径（P3 用 INT8 VAE） |
| Video VAE **encode** | `minimax_h3_video_vae_fp16.safetensors` 的 `encoder.*`（INT8 VAE 偏 decoder，禁止假设能 encode） |

今日冻结的 T2VA forward（A/B 对照面）：

```c
int h3_dit_w4a8_forward_av_f32(
    h3_dit_w4a8 *ctx, const h3_dit_w4a8_core *core, unsigned first_block,
    unsigned n_blocks, float sigma, float audio_sigma,
    const float *video_latent, size_t video_seq,
    const float *audio_latent, size_t audio_seq,
    const float *text_hidden, size_t text_seq,
    const float *v_pos_t, const float *v_pos_h, const float *v_pos_w,
    const float *a_pos_t, const float *a_pos_h, const float *a_pos_w,
    float spatial_scale,
    float *video_velocity, float *audio_velocity,
    int auto_prefetch, h3_dit_w4a8_forward_stats *stats,
    char *error, size_t error_size);
```

pack 今日写死 `[text | audio | video]`（`h3_dit_w4a8.c:4342`）。AdaLN 只有 `H3_DIT_W4A8_MOD_{VIDEO,TEXT,AUDIO}` × 单一 video `t_row`，音频用 `audio_t_active` splice。

Metal 金数字（CUDA 不得改这些测试）：`tests/test_h3.c` `test_layout_fl2va` — 5s 480P、text=128、双 keyframe、`seq_len=8009`、段序 `TEXT, COND, COND, AUDIO, VIDEO`、bounds `{0,128},{128,533},{533,938},{938,1124},{1124,8009}`。

每阶段合入后的共同回归（复制执行）：

1. 无 `--first-frame` / 无 `--ref-*` / `cond_seq=0` → T2V 主题门（或 bit-for-bit 夹具）。
2. 不改 `tests/test_h3.c` layout 金数字。
3. 不 commit 除非用户要求；命令文档不写远程口令。

---

### Task A: T2V 回归门（契约）

**Files:**
- Read: `tools/h3_cuda_generate.c`、`h3_dit_w4a8.c:4296-4534`（`forward_av_f32`）、`out/verify_r1r3_p3_vae/README.txt`、`docs/nvidia_build.md`
- Create: `tests/test_h3_dit_w4a8_cond.c`（本阶段只测 `forward_av` 两次相同输入）
- Modify: `CMakeLists.txt`（加 `h3_dit_w4a8_cond_test` 可执行文件）
- Create: `docs/superpowers/plans/h3-generation-regression-anchor.md`（回归命令；**不写密码**）

**Interfaces:**
- Consumes: 现有 `h3_dit_w4a8_forward_av_f32`、`h3_dit_w4a8_fill_metal_video_pos`、`h3_dit_w4a8_fill_metal_audio_pos`、`h3_dit_w4a8_open_int8`
- Produces: 冻结的主题 CLI 与「后续 pack 改动必须先过此门」的测试入口。本阶段 **不** 新增 `forward_avc`。

- [x] **Step 1: 把主题回归命令写进锚点文档**

在 `docs/superpowers/plans/h3-generation-regression-anchor.md` 写入（路径用 4090 上的实际权文件，**不要**写 SSH 密码）：

标题：`# CUDA T2V 回归锚点（阶段 A）`。正文写：构建见 `docs/nvidia_build.md`；主题默认 INT8 ConvRot，**不是** W4A8。

橙猫（对照 `out/verify_r1r3_p3_vae/cat/`）：

```sh
H3_CUDA_SDPA_BACKEND=cudnn H3_AUDIO_CONV_BACKEND=gemm \
./build-cuda/h3_cuda_generate \
  --dit "$DIT_INT8" --vae "$VAE" --te "$TE" \
  --out /tmp/h3_reg_cat --width 864 --height 480 --seconds 5 \
  --frames 17 --steps 8 --layers 50 --te-layers 50 --vae-blocks 36 \
  --cfg 5 --seed 17 --sdpa-backend cudnn \
  --token-ids-file out/verify2/tokens/v4_orange_cat.txt
```

水下（对照 `out/verify_r1r3_p3_vae/uw/`）：同上，改 `--seed 91` 与 `--token-ids-file out/verify_official_prompt/b_underwater.ids`。

禁止：`--init-latent`、`--reuse`、`--magcache-*`、W4A8 DiT、`--cfg 1`。
日志必须出现：`dit_format: INT8 ConvRot`、`cfg: scale=5`、`reuse_schedule: disabled`、`token_red=0`。

- [x] **Step 2: 写 `forward_av` 自洽测试（两次调用逐 bit 相同）**

创建 `tests/test_h3_dit_w4a8_cond.c`。本阶段断言的是 **现有** API，证明夹具能跑；B 再加 `forward_avc`。

```c
/* tests/test_h3_dit_w4a8_cond.c
 * INT8 打开路径对齐 tools/h3_cuda_generate.c:688-749。 */
#include "h3_dit_int8.h"
#include "h3_dit_w4a8.h"
#include "h3_gpu_cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VIDEO_PATCH 96
#define AUDIO_CH 32
#define HIDDEN 5376

static int same_bits(const float *a, const float *b, size_t n) {
    return memcmp(a, b, n * sizeof(float)) == 0;
}

int main(int argc, char **argv) {
    const char *dit_path = argc > 1 ? argv[1] : NULL;
    char err[512];
    h3_cuda *cuda = NULL;
    h3_dit_int8_store *store = NULL;
    h3_dit_w4a8 *dit = NULL;
    h3_dit_w4a8_core core;
    h3_w4a8_cache_config cfg;
    const size_t text_seq = 8, audio_seq = 4, video_seq = 16;
    const unsigned n_blocks = 2;
    float *xv, *xa, *th, *vv1, *va1, *vv2, *va2, *pt, *ph, *pw, *at, *ah, *aw;
    size_t i;
    int ok = 0;
    if (!dit_path) {
        fprintf(stderr, "usage: %s DIT_INT8.safetensors\n", argv[0]);
        return 2;
    }
    memset(&core, 0, sizeof(core));
    h3_dit_w4a8_config_init(&cfg);
    cuda = h3_cuda_create(0, err, sizeof(err));
    store = cuda ? h3_dit_int8_store_open(dit_path, err, sizeof(err)) : NULL;
    if (!store ||
        !h3_dit_w4a8_load_core_int8(store, &core, err, sizeof(err)) ||
        !(dit = h3_dit_w4a8_open_int8(cuda, store, &cfg, err, sizeof(err)))) {
        fprintf(stderr, "%s\n", err);
        h3_dit_w4a8_core_free(&core);
        h3_dit_int8_store_free(store);
        h3_cuda_free(cuda);
        return 1;
    }
    xv = calloc(video_seq * VIDEO_PATCH, sizeof(float));
    xa = calloc(audio_seq * AUDIO_CH, sizeof(float));
    th = calloc(text_seq * HIDDEN, sizeof(float));
    vv1 = calloc(video_seq * VIDEO_PATCH, sizeof(float));
    va1 = calloc(audio_seq * AUDIO_CH, sizeof(float));
    vv2 = calloc(video_seq * VIDEO_PATCH, sizeof(float));
    va2 = calloc(audio_seq * AUDIO_CH, sizeof(float));
    pt = calloc(video_seq, sizeof(float));
    ph = calloc(video_seq, sizeof(float));
    pw = calloc(video_seq, sizeof(float));
    at = calloc(audio_seq, sizeof(float));
    ah = calloc(audio_seq, sizeof(float));
    aw = calloc(audio_seq, sizeof(float));
    if (!xv || !xa || !th || !vv1 || !va1 || !vv2 || !va2 || !pt || !ph ||
        !pw || !at || !ah || !aw)
        goto done;
    for (i = 0; i < video_seq * VIDEO_PATCH; i++)
        xv[i] = (float)(i % 7) * 0.01f - 0.03f;
    for (i = 0; i < audio_seq * AUDIO_CH; i++)
        xa[i] = (float)(i % 5) * 0.01f - 0.02f;
    for (i = 0; i < text_seq * HIDDEN; i++)
        th[i] = (float)(i % 11) * 0.005f;
    if (!h3_dit_w4a8_fill_metal_video_pos(text_seq, 2, 4, 4, pt, ph, pw,
                                          err, sizeof(err)) ||
        !h3_dit_w4a8_fill_metal_audio_pos(text_seq, (int)(audio_seq / 2), 4, 4,
                                          at, ah, aw, err, sizeof(err)))
        goto done;
    if (!h3_dit_w4a8_forward_av_f32(
            dit, &core, 0, n_blocks, 0.5f, 0.3f, xv, video_seq, xa, audio_seq,
            th, text_seq, pt, ph, pw, at, ah, aw, 1.0f, vv1, va1, 1, NULL,
            err, sizeof(err)))
        goto done;
    if (!h3_dit_w4a8_forward_av_f32(
            dit, &core, 0, n_blocks, 0.5f, 0.3f, xv, video_seq, xa, audio_seq,
            th, text_seq, pt, ph, pw, at, ah, aw, 1.0f, vv2, va2, 1, NULL,
            err, sizeof(err)))
        goto done;
    if (!same_bits(vv1, vv2, video_seq * VIDEO_PATCH) ||
        !same_bits(va1, va2, audio_seq * AUDIO_CH)) {
        fprintf(stderr, "FAIL: forward_av not self-consistent\n");
        goto done;
    }
    ok = 1;
    puts("ok: forward_av self-consistent (stage A)");
done:
    if (!ok) fprintf(stderr, "error: %s\n", err);
    free(xv); free(xa); free(th); free(vv1); free(va1); free(vv2); free(va2);
    free(pt); free(ph); free(pw); free(at); free(ah); free(aw);
    h3_dit_w4a8_free(dit);
    h3_dit_w4a8_core_free(&core);
    h3_dit_int8_store_free(store);
    h3_cuda_free(cuda);
    return ok ? 0 : 1;
}
```

- [x] **Step 3: 把测试挂进 CMake**

在 `CMakeLists.txt` 的 `h3_dit_w4a8_audio_probe` 目标旁增加：

```cmake
add_executable(h3_dit_w4a8_cond_test
  tests/test_h3_dit_w4a8_cond.c
  h3_dit_w4a8.c
  h3_dit_int8.c
  h3_safetensors.c
  h3_w4a8.c
  h3_w4a8_dequant.c
  h3_w4a8_cache.c)
target_include_directories(h3_dit_w4a8_cond_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_dit_w4a8_cond_test PRIVATE c_std_11)
target_link_libraries(h3_dit_w4a8_cond_test PRIVATE h3_gpu_cuda m)
```

并把该 target 加入底部 `install(TARGETS ...)` 列表。

- [x] **Step 4: 在 4090 上编译并跑夹具**

```sh
cmake --build build-cuda -j --target h3_dit_w4a8_cond_test h3_cuda_generate
./build-cuda/h3_dit_w4a8_cond_test "$DIT_INT8"
```

Expected: `ok: forward_av self-consistent (stage A)`，exit 0。

- [x] **Step 5: 确认主题路径仍可复现（若本轮未改 generate：对照 P3 日志即可，不必重跑 8-step e2e）**

读 `out/verify_r1r3_p3_vae/cat/generate.log`：必须已有 `dit_format: INT8 ConvRot`、`cfg: scale=5`、`reuse_schedule: disabled`、`phase_ms` 橙猫 total≈63.75s。本阶段 **不做** 新的 50L 出片，除非 generate 被改到。在锚点文档写明：「A 的 e2e 金标准是 P3 目录，不是 W4A8。」

**本阶段不做：** 改 TE、改 VAE、改 F2VA lite、加 `--first-frame`、冲 ≤35s。

**执行记录（阶段 A，2026-08-19）**

- 冻结主题命令：`docs/superpowers/plans/h3-generation-regression-anchor.md`（INT8 ConvRot，非 W4A8；橙猫 seed=17 / 水下 seed=91；禁止 lite / reuse / magcache / cfg=1）。
- 夹具：4090 `/home/haiyang/h3_c/src_sync/build-cuda/h3_dit_w4a8_cond_test` + `DIT_INT8` → stdout `ok: forward_av self-consistent (stage A)`，exit 0（00:10:29–00:10:33 CST）。日志副本 `out/verify_stage_a/cond_test.log`。
- generate **未改**；未重跑 8-step e2e。P3 对照：`dit_format: INT8 ConvRot`、`cfg: scale=5`、`reuse_schedule: disabled`、`token_red=0`；橙猫 `phase_ms ... total=63753.0`（≈63.75s），水下 `total=68027.9`。
- 未加产品 `--first-frame`，未引入 `forward_avc`，未做 B+。未 commit。

---

### Task B: 官方 condition 骨架

**Files:**
- Modify: `h3_dit_w4a8.h`（`forward_av_f32` 声明后新增 `forward_avc_f32` + cond RoPE helper）
- Modify: `h3_dit_w4a8.c`
  - `h3_dit_w4a8` 结构体（约 `:123-128`）增加 visual-condition AdaLN 状态
  - `project_block_adaln_metal`（`:844-874`）在 `cond` 激活时扩展 modulation
  - `layers_with_token_reduction`（`:3853`）`full_seq` 计入 cond；video 网格仍只从 **target video** 推断
  - `forward_av_f32` 改为 `cond_seq=0` 调 `forward_avc`
  - 新增 `h3_dit_w4a8_forward_avc_f32`、`h3_dit_w4a8_fill_metal_cond_pos`
- Modify: `tests/test_h3_dit_w4a8_cond.c`（加 avc 门）
- Modify: `tools/h3_cuda_generate.c` **仅当** 需要把现有调用换成 `forward_avc(..., NULL, 0, ...)`；**禁止** 本阶段加 `--first-frame`

**Interfaces:**
- Consumes: Task A 冻结的 `forward_av_f32` 行为；Metal 对照 `h3_host.c:347-362`、`h3_dit_schedule.c:115-147`、`h3_dit_schedule.c:458-466`
- Produces:

```c
/* Pack [text | cond | audio | video] — Metal keyframe 顺序。
 * cond_seq=0 → 与 h3_dit_w4a8_forward_av_f32 逐 bit 相同。
 * cond 行参与 attention；velocity 只写 video_seq*96 与 audio_seq*32。 */
int h3_dit_w4a8_forward_avc_f32(
    h3_dit_w4a8 *ctx, const h3_dit_w4a8_core *core, unsigned first_block,
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

/* 单帧 COND 网格：空间与 fill_metal_video_pos 同一 canvas，
 * 全部 pos_t = condition_time（首帧=text_len；尾帧见 Task F）。 */
int h3_dit_w4a8_fill_metal_cond_pos(
    float condition_time, int latent_h, int latent_w,
    float *pos_t, float *pos_h, float *pos_w,
    char *error, size_t error_size);
```

AdaLN 规则（对齐 Metal，**禁止**让 cond 与 target 共用同一个 video timestep）：

- `cond_seq=0`：保持今日 3-行 modulation（`H3_DIT_W4A8_ADALN_OUT`），`row_map ∈ {0,1,2}`。这是 bit-for-bit 的关键。
- `cond_seq>0`：与 `t_row_from_sigma` 同一轴，`time = 1.f - sigma`（Metal `video = 1 - sigmas.video`，干净端）。额外算 `t_row_cond = h3_dit_w4a8_t_row_from_sigma(1.f - 0.999f, core->t_table_rows)`（time=0.999 的 VIDEO 切片，**不是** flow-σ=0.999）。若 `time >= 0.999f`（即 `sigma <= 0.001f`），cond 回退到普通 `MOD_VIDEO`（与 `h3_dit_schedule.c:123-124` 的 `video >= 0.999f` 相同）。否则 cond 行 `row_map = 3`，modulation 扩成 4 行：`[V_σ, T_σ, A_σ, V_0.999]`。`V_0.999` 取 time=0.999 投影的 **VIDEO 切片**（tag=0），不是 TEXT。F2VA 第 0 步 `sigmas[0]=1.0` 必须走 row_map=3，不得收成 noisy `MOD_VIDEO`。
- modality：cond = video tag 0；不是 text。
- 音频 splice（`audio_t_active`）继续只改 MOD_AUDIO 切片，与 cond 正交。

Pack / 偏移：

```text
seq        = text_seq + cond_seq + audio_seq + video_seq
cond_off   = text_seq
audio_off  = text_seq + cond_seq
video_off  = text_seq + cond_seq + audio_seq
```

cond 与 video **共用** `video_patch_*`（`dense_f32_cuda_to_dev` / `h3_dit_w4a8_patch_video_cuda`）。final head **只**从 `video_off` / `audio_off` 取 target。

`layers_with_token_reduction`：今日 `full_seq = text_seq + video_seq + audio_seq`（`:3853`）。改成计入 `cond_seq`。token-reduction 的空间网格仍从 **target video** 的 `v_pos_*` 推断（`video_off` 之后），cond 当作不可池化前缀（与 text 一样）。主题默认 `token_red=0`，走 `h3_dit_w4a8_layers_forward_f32` 的 full_seq 路径。

RoPE：cond 用调用方 `c_pos_*`。本阶段测试可用假坐标；E/F 再填 keyframe 公式。

- [x] **Step 1: 先写会失败的 avc 测试（cond_seq=0 bit-for-bit + cond_seq>0 形状）**

在 `tests/test_h3_dit_w4a8_cond.c` 追加（保留 A 的自洽测试）：

```c
static int test_avc_empty_matches_av(h3_dit_w4a8 *dit,
                                     const h3_dit_w4a8_core *core,
                                     char *err, size_t es) {
    /* 与 A 相同的缓冲；r1 = forward_av；r2 = forward_avc(..., NULL, 0, ...) */
    /* memcmp video_velocity 与 audio_velocity 必须逐 bit 相同 */
}

static int test_avc_cond_keeps_target_size(h3_dit_w4a8 *dit,
                                           const h3_dit_w4a8_core *core,
                                           char *err, size_t es) {
    const size_t text_seq = 8, audio_seq = 4, video_seq = 16, cond_seq = 8;
    float *cond = calloc(cond_seq * 96, sizeof(float));
    float *ct = calloc(cond_seq, sizeof(float));
    float *ch = calloc(cond_seq, sizeof(float));
    float *cw = calloc(cond_seq, sizeof(float));
    /* fill cond with 0.1f；c_pos_t = text_seq（假首帧） */
    /* call forward_avc；要求：
     *  1. 返回 1 且 velocity 全 finite
     *  2. 只写 video_seq*96 与 audio_seq*32（输出缓冲不要 oversized 依赖）
     *  3. 第二次相同输入 bit-for-bit 自洽
     */
}
```

- [x] **Step 2: 编译测试，确认 `forward_avc` 尚未声明时失败**

```sh
cmake --build build-cuda -j --target h3_dit_w4a8_cond_test
```

Expected: FAIL，`h3_dit_w4a8_forward_avc_f32` undeclared / linker undefined。

- [x] **Step 3: 在头文件声明 avc 与 cond RoPE helper**

`h3_dit_w4a8.h` 在 `forward_av_f32`（`:423-431`）之后插入上面 **Produces** 的两个声明。注释写明 pack 顺序与「cond 不进 Euler」。

在 `h3_dit_w4a8.c` 的 ctx 结构体 `audio_t_active` 旁增加：

```c
    size_t visual_cond_t_row;
    int visual_cond_t_active; /* cond_seq>0 且 time=1-σ < 0.999（σ > 0.001） */
    float *adaln_visual_cond_scratch; /* time=0.999 投影；VIDEO 切片拷到 row 3 */
```

`forward_avc` 结束时与 audio 一样清零 `visual_cond_t_active`。

- [x] **Step 4: 实现 `h3_dit_w4a8_fill_metal_cond_pos`**

复制 `h3_dit_w4a8_fill_metal_video_pos`（`:974-1009`）的 **单帧** 空间公式（`nh=latent_h/2` 等），循环只有一层 `yh,xw`，所有 `pos_t[i] = condition_time`。不要加 `h3_dit_frame_rescale` 时间步进（那是目标视频轴）。

对照：`tests/test_h3.c` `test_layout_fl2va` 首段 COND 405 行（128..533），`latent_h=30`、`latent_w=54` → `frame_rows = 15*27 = 405`。

- [x] **Step 5: 实现 `forward_avc`；把 `forward_av` 收成 wrapper**

`forward_av_f32` 末尾改为：

```c
    return h3_dit_w4a8_forward_avc_f32(
        ctx, core, first_block, n_blocks, sigma, audio_sigma,
        video_latent, video_seq, audio_latent, audio_seq,
        NULL, 0, text_hidden, text_seq,
        v_pos_t, v_pos_h, v_pos_w, a_pos_t, a_pos_h, a_pos_w,
        NULL, NULL, NULL, spatial_scale,
        video_velocity, audio_velocity, auto_prefetch, stats,
        error, error_size);
```

`forward_avc` 从现 `forward_av` 体复制，然后：

1. `cond_seq>0` 时要求 `cond_latent` 与 `c_pos_*` 非 NULL。
2. 按上面偏移 pack hidden / row_map / RoPE。
3. patch：text memcpy；cond 与 video 都走 `video_patch_*`；audio 走 `audio_patch_*`。
4. `visual_cond_t_active` 仅在 `cond_seq>0 && time < 0.999f` 时置 1（`time = 1.f - sigma`，即 `sigma > 0.001f`）。**禁止**用 flow-σ `sigma < 0.999f`（第 0 步 σ=1 会把 cond 收成 noisy `MOD_VIDEO`）。
5. final 仍只写 target video/audio。

AdaLN：扩展 `project_block_adaln_metal`。`visual_cond_t_active==0` 时 **一个分支都不进**，保证空 cond 与今日二进制路径一致（包括 modulation 分配大小仍是 `H3_DIT_W4A8_ADALN_OUT`）。激活时：

```c
    /* after existing video/text/audio project + audio splice */
    if (ctx && ctx->visual_cond_t_active) {
        if (!h3_dit_w4a8_time_feat_from_table(aux, ctx->visual_cond_t_row,
                time_feat, error, error_size) ||
            !h3_dit_w4a8_adaln_project_f32(aux, time_feat,
                ctx->adaln_visual_cond_scratch, error, error_size))
            return 0;
        memcpy(modulation + 3 * slice,
               ctx->adaln_visual_cond_scratch +
                   (size_t)H3_DIT_W4A8_MOD_VIDEO * slice,
               slice * sizeof(float));
    }
```

CUDA AdaLN kernel / `h3_dit_w4a8_adaln_f32` 必须允许 `row_map` 值为 3。检查 device AdaLN 是否写死 `mod_rows=3`；若写死，仅在 `visual_cond_t_active` 时传 `mod_rows=4`。**空 cond 必须仍传 3。**

- [x] **Step 6: 跑 cond 测试**

```sh
cmake --build build-cuda -j --target h3_dit_w4a8_cond_test
./build-cuda/h3_dit_w4a8_cond_test "$DIT_INT8"
```

Expected: A 自洽仍 PASS；`cond_seq=0` avc≡av bit-for-bit；`cond_seq>0` finite 且自洽。

- [x] **Step 7: generate 无 condition 接线（若 Step 5 把 `forward_av` 收成 wrapper，本步只重建 generate）**

```sh
cmake --build build-cuda -j --target h3_cuda_generate
```

用 **短步冒烟**（不要当主题收口）：`--steps 2 --layers 4 --te-layers 1 --skip-vae`，同一 seed 两次，比较 `latent.raw` 或 velocity checksum。Expected：与包装前一致。完整 8-step 主题门放到 E 的「A 门」；B 只要证明 wrapper 无数值漂移。

- [x] **Step 8: 确认 CLI 仍无产品 `--first-frame`**

`rg "first-frame" tools/h3_cuda_generate.c` 不得出现官方产品旗标。lite 开关可仍在，但本阶段不改其默认。

- [x] **Fix round: AdaLN 0.999 极性对齐 Metal time 轴**

夹具（host + `forward_avc` stats）：`sigma=1.0` → `row_map=3` / 第 4 行 VIDEO@time=0.999；`sigma<=0.001` → 回退 `MOD_VIDEO`。TDD：先 RED（flow-σ `sigma<0.999` 在 σ=1.0 上 inactive），再 GREEN。

- [x] **Fix round: 空 cond 夹具不再假装能证明包装前金标**

`forward_av` 已是 `forward_avc(NULL,0)` wrapper，memcmp 写明局限。负载不变量：空 cond 在 σ=1.0 仍 `visual_cond_t_active=0`、不进 4 行 AdaLN。无预包装金标文件可对照。

**本阶段不做：** jpeg、VAE encoder、vision、Ref2VA ckpt、改 lite `img_w`。

**执行记录（阶段 B，2026-08-19）**

- TDD RED：夹具调用 `forward_avc` / `fill_metal_cond_pos` 时 linker undefined（`h3_dit_w4a8_forward_avc_f32`、`h3_dit_w4a8_fill_metal_cond_pos`）。
- 实现：`h3_dit_w4a8_forward_avc_f32` pack `[text | cond | audio | video]`；`forward_av` 收成 `cond_seq=0` wrapper。空 cond 保持 3 行 AdaLN（`visual_cond_t_active=0`）；`cond_seq>0 && time<0.999`（`time=1-σ`，即 `σ>0.001`）时 row_map=3、modulation 扩 4 行，VIDEO 切片取 **time=0.999**（`t_row_from_sigma(1-0.999)`）。final / Euler 只写 target video×96 与 audio×32。
- 夹具仍用 `video_seq=16` + `fill_metal_video_pos(2,4,4)`（只填 8 槽），未当成填满 video 网格。
- 4090 `h3_dit_w4a8_cond_test` 初版（00:22:20–00:22:23 CST）exit 0（当时 AdaLN 极性仍用 flow-σ，已被 fix round 覆盖）。
- generate 源码未加 `--first-frame`；仅重建。短步冒烟 `--steps 2 --layers 4 --te-layers 1 --skip-vae --cfg 5 seed=17` 两次 `latent.raw` `cmp` 相同（`dit_format: INT8 ConvRot`、`cfg: scale=5`、`reuse_schedule: disabled`、`token_red=0`）。
- 未改主题锚点、未切 W4A8、未做 C+、未 commit。

**执行记录（阶段 B fix round，2026-08-19）**

- Reviewer REQUEST_CHANGES：AdaLN 0.999 回退极性 vs Metal；空 cond memcmp 同义反复。
- TDD RED（00:33 CST）：`FAIL: sigma=1.0 must keep cond on VIDEO@0.999 (got inactive)`（当时 `sigma < 0.999f` flow-σ 门）。
- GREEN：`h3_dit_w4a8_visual_cond_adaln_active` 用 `time=1-σ`，`active = cond_seq>0 && σ>0.001`；`visual_cond_t_row = t_row_from_sigma(1-0.999)`。
- 4090 `h3_dit_w4a8_cond_test`（00:34:12–00:34:16 CST）exit 0：
  - `ok: fill_metal_cond_pos 405 rows, pos_t pinned`
  - `ok: AdaLN 0.999 polarity (host, Metal time axis)`
  - `ok: forward_av self-consistent (stage A)`
  - `ok: forward_avc cond_seq=0 wrapper-identical; 3-row AdaLN at sigma=1`
  - `ok: forward_avc cond_seq>0 finite, target-sized, self-consistent`
  - `ok: AdaLN 0.999 polarity (sigma=1 row_map=3; sigma<=0.001 MOD_VIDEO)`
  - `ok: stage B avc gates`
  - 日志 `out/verify_stage_b/cond_test.log`（副本 `cond_test_fix_round.log`）
- 空 cond：无包装前金标；注释/stderr 写明 wrapper tautology；σ=1.0 仍 3 行 AdaLN。
- 未加产品 `--first-frame`，未做 C，未 commit。

---

### Task C: 共享视觉条件 payload

**Files:**
- Create: `h3_vae_encode_f32.h`、`h3_vae_encode_f32.c`
- Create: `tests/test_h3_vae_encode_f32.c`
- Modify: `CMakeLists.txt`（增加 `h3_vae_encode_f32_test`：`tests/test_h3_vae_encode_f32.c` + `h3_vae_encode_f32.c` + `h3_safetensors.c`，链 `h3_gpu_cuda m`，**不链** DiT）
- Read: `h3_video_encoder.c`（`TILE_PIXELS=256`、`TILE_OVERLAP_MIN=64`、`SPATIAL_RATIO=16`、`encoder.*` / `quant_conv` 键）、`h3_dit.c:3077-3097`（patchify 循环）、`h3.c:311-317`（augment）
- Optional appendix: `h3_vae_int8_encode_cuda.h` / `.c`（仍属 C，**不是 S**）

**Interfaces:**
- Consumes: fp16 VAE `models/comfy-org/vae/minimax_h3_video_vae_fp16.safetensors`；`h3_safetensors.c`；Metal `h3_video_latent` 布局（`h3_video_encoder.h:8-15`）
- Produces:

```c
typedef struct {
    int time;    /* h3_video_encoder_latent_t(frames) = ceil(frames/4) */
    int height;  /* pixels/16 */
    int width;
    float *values; /* channel-major [24,time,height,width] */
} h3_vae_latent_f32;

int h3_vae_encode_f32(
    const char *vae_path,
    const float *rgb01_chtw, int frames, int height, int width,
    h3_vae_latent_f32 *out,
    char *error, size_t error_size);
void h3_vae_latent_f32_free(h3_vae_latent_f32 *latent);

/* 与 Metal h3_dit_patchify_video 同一循环顺序：t, h+=2, w+=2, c, dh, dw */
int h3_vae_patchify_video_96(const float *latent_24thw, int time,
                             int height, int width, float *rows_seq96,
                             size_t row_elements);

/* generate 路径调用；不要写进 encoder */
void h3_vae_augment_0p999(float *rows, size_t count, uint64_t seed);
```

`h3_vae_augment_0p999` 必须与 `h3.c:311-317` 相同：`h3_rng_seed` + `0.999 * x + 0.001 * N(seed)`。CUDA 不要链接 `h3_host.c`；把 `h3_rng_*` 的实现 **复制** 到 `h3_vae_encode_f32.c`（算法已在 `h3_host.c:500+`），或抽一个无 Accelerate 的 `h3_rng.c`。优先复制最小 RNG，避免动 Metal 文件。

几何：

- 输入 RGB：channel-major `[3,T,H,W]`，`[0,1]`，`H,W` 为 16 的倍数。
- `T_lat = (frames + 3) / 4`（`h3_video_encoder_latent_t`）。
- tiling：256px tile、≥64 overlap，与 `h3_video_encoder.c` `tile_axis_build` 一致。
- 权键：`encoder.conv_in`、`encoder.down.{0..5}.block.{0,1}`、`encoder.down.{l}.downsample.conv`、`encoder.norm_out`、`encoder.conv_out`、`quant_conv`；再按 Metal `load_normalization` 做 latent mean/std。执行时用 `./build-cuda/h3_st_inventory` 打印 fp16 文件，核对键名后再写 loader。

- [x] **Step 1: 写会失败的 encode 形状测试（不依赖 Metal GPU）**

`tests/test_h3_vae_encode_f32.c`：

```c
static void test_patchify_order(void) {
    /* 1×2×2 latent，C=24 全唯一值；断言 rows[0] 来自 c=0,t=0,h=0,w=0 */
}
static void test_augment_formula(void) {
    float x = 1.0f;
    h3_vae_augment_0p999(&x, 1, 1);
    /* 与手算 0.999 + 0.001*h3_rng_normal(seed=1 的第一个样本) 一致 */
}
```

- [x] **Step 2: 跑测试确认失败**

```sh
cmake --build build-cuda -j --target h3_vae_encode_f32_test
./build-cuda/h3_vae_encode_f32_test
```

Expected: FAIL（符号不存在）。

- [x] **Step 3: 实现 patchify + augment + 空 encode stub**

`h3_vae_patchify_video_96` 逐行复制 `h3_dit.c:3085-3095`。`h3_vae_encode_f32` 先只检查参数并返回「not implemented」，让形状测试先绿。

- [x] **Step 3b: CMake 目标**

```cmake
add_executable(h3_vae_encode_f32_test
  tests/test_h3_vae_encode_f32.c
  h3_vae_encode_f32.c
  h3_safetensors.c)
target_include_directories(h3_vae_encode_f32_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_vae_encode_f32_test PRIVATE c_std_11)
target_link_libraries(h3_vae_encode_f32_test PRIVATE h3_gpu_cuda m)
```

- [x] **Step 4: 实现 host F32 encoder forward**

按 `h3_video_encoder.c` 的 block/level/stride（`level_channels`、`space_strides`、`time_strides`）用 cblas/naive conv 跑通 **单 tile 256×256、1 帧**。Conv 数值以 Metal `encode_tile` 为对照：在 Darwin 用现有 `tests/test_real_video_encoder.c` 导出同一 jpeg 的 latent；在 4090 跑 host encoder，比较 max abs。

阈值（路线图已给量级，本计划钉死）：

- 单 256 tile、1 帧：`max_abs < 5e-2`；若实测更小，把数字写进测试注释，不要放宽。
- `out.time == (frames+3)/4`，`out.height == height/16`，`out.width == width/16`。

- [x] **Step 5: 在 4090 跑 encode 单测**

```sh
./build-cuda/h3_vae_encode_f32_test --vae "$VAE_FP16" --ref-latent /path/to/metal_tile.raw
```

Expected: PASS。日志不得出现 `F2VA`。

4090 `h3_vae_encode_f32_test --vae fp16 --ref-latent`（合成 `fill_tile_rgb`，对照 Darwin Metal `h3_video_vae_encode` dump）：`max_abs=6.50e-6`，shape `[24,1,16,16]`，64×96 condition rows。日志无 `F2VA`。附录 Step 6 CUDA encode 后置。

- [ ] **Step 6（附录，可后置）：CUDA encode**

仅当 host 已过门。新 API `h3_vae_encode_cuda` vs host `max_abs < 1e-3`。这是 C 的加速，验收仍是 payload 正确，**不是** 阶段 S 的 ≤35s。

**本阶段不做：** 产品 `--first-frame`；不把 encode 结果写成「已经是 F2VA」；不做 vision。

**执行记录（阶段 C，2026-08-19）**

- host F32 encoder + patchify 96 + augment 0.999；4090 形状/阈值门先用 torch Metal-pad 对照（已被 fix round 替换）。
- 不宣称 F2VA；无产品 `--first-frame`；未做 D/E；未 commit。

**执行记录（阶段 C fix round，2026-08-19）**

- Reviewer REQUEST_CHANGES：数值金标必须是 Metal `h3_video_vae_encode` / `encode_tile` dump，不是 torch Metal-pad。
- Darwin M4：`tests/test_real_video_encoder.c` dump 模式调用 `h3_video_vae_encode`（34 MPS conv / 18 submissions = 单 tile `encode_tile`）。本机无 MiniMax-H3 `FL2VA/video_vae/source` F32 树，把同一份 fp16 encoder 权升到 F32 shard 后加载（不改 Metal 数学）。
- 同一 `fill_tile_rgb` dump `[24,1,16,16]` → `out/verify_stage_c/metal_fill_tile_rgb_24x1x16x16.f32`（mean=-0.13052 std=0.96949）。同一 `tile256.rgb24` dump → `metal_tile256_rgb24_24x1x16x16.f32`。
- 4090 `h3_vae_encode_f32_test --vae fp16 --ref-latent metal_fill_tile_rgb_24x1x16x16.f32`：`max_abs=6.496906e-06`（阈值 5e-2），exit 0。stdout `out/verify_stage_c/h3_vae_encode_f32_test_4090.log`，无 `F2VA`。
- 未改 pack/AdaLN；未加产品 `--first-frame`；未做 E；未 commit。

---

### Task D: 共享 vision + multimodal TE

**Files:**
- Create: `h3_multimodal_cuda.h`、`h3_multimodal_cuda.c`（presentation ids + span）
- Create: `h3_vision_cuda.h`、`h3_vision_cuda.c`（Qwen vision，先 host F32）
- Modify: `h3_nvfp4.h` / `h3_nvfp4.c`（multimodal inject：embed 替换 span + deepstack after layers 0/1/2）
- Create: `tests/test_h3_multimodal_cuda.c`
- Modify: `CMakeLists.txt`
- Read: `h3_multimodal.c`（`H3_VISION_START=151652`、`H3_VISION_END=151653`、`H3_IMAGE_PAD=151655`）、`h3_text_encoder.h:40-52`、`h3_vision_encoder.c`、`h3_nvfp4.c` `h3_nvfp4_encode_layers_f32`
- Keep: `h3_token_ids.c` 纯文本路径不动

**Interfaces:**
- Consumes: 现有 `h3_nvfp4_encode_layers_f32(store, token_ids, tokens, n_layers, ...)`；用户 prompt 仍来自 `--token-ids-file`
- Produces:

```c
#define H3_CUDA_VISION_START UINT32_C(151652)
#define H3_CUDA_VISION_END   UINT32_C(151653)
#define H3_CUDA_IMAGE_PAD    UINT32_C(151655)

typedef struct {
    size_t start;     /* 含 VISION_START 的 span 起点 */
    size_t tokens;    /* START + pads + END */
    int grid_h, grid_w;
} h3_cuda_vision_span;

/* 拼 [Picture-prefix | vision pads | prompt_ids]。
 * Picture n 的 UTF-8 前缀 ids 用静态表（Metal tokenizer 一次 dump），
 * 不在本阶段移植完整 HF BPE（架构禁止把纯 C tokenizer 塞进 F2VA 竖切）。 */
int h3_multimodal_cuda_build_fl2va_ids(
    size_t picture_count,          /* F2VA=1；FL2VA=2 */
    const uint32_t *prompt_ids, size_t prompt_n,
    const int *grid_h, const int *grid_w, /* 每图 vision token 网格 */
    uint32_t **ids_out, size_t *n_out,
    h3_cuda_vision_span *spans_out,
    char *error, size_t error_size);

int h3_vision_encode_f32(
    const char *vision_weights,    /* FL2VA text_encoder 或独立 visual 文件 */
    const float *pixels_t3hw,      /* [T,3,H,W] in [0,1], T=1 for image */
    int frames, int height, int width, /* H,W multiples of 32 */
    float *merged_out,             /* [tokens, 5120] */
    float *deepstack_out[3],       /* 同形 */
    int *grid_h, int *grid_w, size_t *tokens,
    char *error, size_t error_size);

/* 无 span 时必须 ≡ h3_nvfp4_encode_layers_f32 */
float *h3_nvfp4_encode_layers_multimodal_f32(
    const h3_nvfp4_store *store,
    const int32_t *token_ids, size_t tokens,
    const h3_cuda_vision_span *spans, size_t span_count,
    const float *merged,           /* 每 span 的 vision 行，按 tokens 拼接 */
    const float *deepstack[3],
    unsigned n_layers,
    h3_nvfp4_encode_stats *stats, char *error, size_t error_size);
```

无图：`span_count=0` → 直接调现有 `h3_nvfp4_encode_layers_f32`。有图：embed 后把 span 内（不含 START/END 的 pad 段，对齐 Metal `append_vision`）换成 `merged`；language layers 0/1/2 之后加 deepstack（对齐 `h3_text_encode_multimodal_bf16`）。

Picture 前缀静态表：在 Darwin 用 `h3_tokenizer` 对 `"<Picture 1>: "` / `"<Picture 2>: "` dump ids，写入 `h3_multimodal_cuda.c` 的 `k_picture_prefix_ids[n]`。执行本任务时先跑 dump，再贴真实数组（禁止留空表）。

Vision 权：用 `h3_st_inventory` 扫 FL2VA `text_encoder` 与 NVFP4 单文件。**禁止**假设 INT8 DiT 含 visual。若 visual 只在 Metal 目录，CUDA encode 读那份 fp16/bf16。

- [x] **Step 1: 写无图 TE 回归（必须先绿）**

```c
/* 同一 token-ids、同一 n_layers：
 * multimodal(..., spans=NULL, 0) memcmp 或 cosine==1 vs encode_layers_f32 */
```

- [x] **Step 2: 跑测试，确认 multimodal 符号不存在时失败；实现 wrapper 后无图 PASS**

无图路径 = 直接 `return h3_nvfp4_encode_layers_f32(...)`。G1 `hi_first` / unswizzle / `pre_quant` 不得改。

- [x] **Step 3: dump Picture 前缀 ids，实现 `h3_multimodal_cuda_build_fl2va_ids`**

单测：`picture_count=1`、`prompt_n=3`、`grid=2×2` → 序列以 Picture 前缀开头，随后 `151652`、四个 `151655`、`151653`、再 3 个 prompt id。

- [x] **Step 4: 实现 vision encode（先 host，单图 32 倍数）**

对照 Darwin `tests/test_real_qwen_vision.c` 导出的 merged 行。阈值：相对 Metal `rel-L2 < 5e-2` 或执行时实测钉死（写入测试注释）。本阶段 **不** 接 Ref 视频双帧（H3）。

- [x] **Step 5: 有图 TE 可区分性**

同一 prompt ids，有 vision vs 无 vision 的 hidden **不得** cosine≈1。有图序列必须含 Picture presentation。不接 DiT。

- [x] **Step 6: 确认纯文本 generate 未改默认**

`h3_cuda_generate` 无 `--first-frame` 时仍只走 `h3_nvfp4_encode_layers_f32`。本阶段可以先不改 generate。

**本阶段不做：** 把 vision 做成 F2VA 专用分叉；不接 Ref `<Video n>`；不实现完整 HF BPE。

**执行记录（阶段 D，2026-08-19）**

- Picture 前缀：Qwen tokenizer（与 `tests/test_tokenizer.c` fox 金标 `32,2518,38835,11435,1526,11794` 一致）dump `"<Picture 1>: "` → `21604,3826,220,16,26818,220`；`"<Picture 2>: "` 仅把 `16` 换成 `17`。静态表写入 `h3_multimodal_cuda.c`。
- 无图：`h3_nvfp4_encode_layers_multimodal_f32(..., spans=NULL, 0)` 直接调 `encode_layers_f32`。G1 `hi_first` / unswizzle / `pre_quant` 未改。
- 有图：embed 后替换 START/END 内侧 IMAGE_PAD 行为 `merged`；language layers 0/1/2 之后加 deepstack；mRoPE 按 Metal `build_positions`（CUDA span.start 含 VISION_START）。
- Vision：host F32，权键 `visual.*`（NVFP4 TE 单文件 351 个 BF16 visual 张量）；亦接受 `model.visual.*`。本阶段不接双帧 Ref 视频。
- 4090 `h3_multimodal_cuda_test --te "$TE"`（01:21 CST）exit 0：
  - `ok: FL2VA ids Picture prefix + 151652/151655x4/151653 + prompt`
  - `ok: no-image multimodal TE memcmp-identical to encode_layers_f32`
  - `ok: vision encode 32x32 finite, patch grid, merged tokens`
  - `with-image vs pad-embed TE cosine=0.545499`
  - `ok: with-image TE distinct and contains Picture presentation`
  - 日志 `out/verify_stage_d/multimodal_test.log`
- Metal merged dump 本环境不可得（Darwin 无 MiniMax-H3 text_encoder；4090 无 Metal）。夹具对 32×32 收 geometry+finite；`--ref-merged` 已接 `rel-L2 < 5e-2`。
- `h3_cuda_generate` 源码未加产品 `--first-frame`；无图仍走 `h3_nvfp4_encode_layers_f32`。未改 `h3_token_ids.c`。CMake 只追加 `h3_multimodal_cuda_test`，保留 C 的 `h3_vae_encode_f32_test`。未做 E、未 commit。

**执行记录（阶段 D fix round，2026-08-19）**

- Reviewer REQUEST_CHANGES：`grid_h/grid_w` 契约分裂。vision 返回 patch `H/16,W/16` + `tokens=(H/32)*(W/32)`；builder 曾用 `pads=grid_h*grid_w`；`h3_nvfp4` mRoPE 把 span.grid 当 merged（不 /2）。32×32 夹具硬编码 `grid=1×1` 把 2×2 vs 1 token 藏起来；864×480 会 1620 pad vs 405 merged。
- 修法跟 Metal：builder pad = `vision.tokens`；span.grid 存 patch；mRoPE `grid/2`。`build_fl2va_ids` 增加 `vision_tokens[]`；拒绝奇数/`<2` patch grid，且要求 `(grid/2)*(grid/2)==tokens`。
- 夹具：32×32 接线用 **真实** `h3_vision_encode_f32` 返回值，断言 `span.tokens-2 == vision.tokens`；禁止硬编码 1×1。另测 864×480 → 405 pads（不是 1620）。Picture 前缀单测改为 patch 4×4 + `tokens=4`（仍四个 IMAGE_PAD）。
- 4090 `h3_multimodal_cuda_test --te "$TE"` exit 0：
  - `ok: span pads follow vision.tokens (32x32=1, 864x480=405, not grid product)`
  - `wired 32x32 vision: tokens=1 patch=2x2 span.tokens-2=1`
  - 无图 memcmp 仍 identical；有图 cosine=0.545499（与修前 1-pad 32×32 相同：2×2/2 仍是 1×1 mRoPE）
  - 日志副本 `out/verify_stage_d/multimodal_test_fix_round.log`
- Metal `--ref-merged` **未跑**：本机无 MiniMax-H3 `FL2VA/text_encoder`；4090 无 Metal。不把 geometry 冒充 Step 4。
- 无产品 `--first-frame`；未做 E；未 commit。

---

### Task E: F2VA 竖切（首帧）

**Files:**
- Modify: `tools/h3_cuda_generate.c`（官方 `--first-frame`；lite 降级文案；调用 `forward_avc`）
- Use: `h3_ffmpeg_read_image_f32(..., H3_IMAGE_FIT_STRETCH, ...)`（`h3_ffmpeg.h:19-21`，对齐 Metal `h3.c:1228`）
- Use: Task C encode+patchify+augment；Task D multimodal TE；Task B pack
- Modify: help 字符串（`:514+`）

**Interfaces:**
- Consumes: `forward_avc`、`h3_vae_encode_f32`、`h3_vae_patchify_video_96`、`h3_vae_augment_0p999`、`h3_multimodal_cuda_build_fl2va_ids`、`h3_vision_encode_f32`、`h3_nvfp4_encode_layers_multimodal_f32`
- Produces: CUDA 产品入口 `--first-frame IMG`；keyframes=`[0]`；RoPE `condition_time = text_len`（**multimodal 之后的真实 text_seq**，不是裸 prompt 长度）

接线（官方路径，禁止走 `apply_t0_img2img`）：

1. `h3_ffmpeg_read_image_f32(path, render_w, render_h, H3_IMAGE_FIT_STRETCH, &rgb, ...)`
2. `h3_vae_encode_f32` → patchify → augment(seed)
3. vision + Picture presentation → TE hidden
4. `h3_dit_w4a8_fill_metal_cond_pos((float)text_seq, latent_h, latent_w, ...)`
5. 每步 `forward_avc`；Euler **只**更新 target video/audio（现有 `h3_dit_w4a8_euler_step_f32` 本来就只拿 target 缓冲）
6. 日志：`cond_seq=`、`keyframes=0`、`official_condition=1`。**禁止**用 `img_w=` 当官方路径标记

降级：若仍保留 `--init-latent` / `--lock-first-frame` / `--init-first-frame`，启动时打印 `non-official G3-lite`，help 写「非官方 condition」。`--first-frame` 与 lite 同时出现 → 拒绝。

- [x] **Step 1: 加 CLI 解析与互斥校验（先不接 encode）**

```c
const char *first_frame_path = NULL;
/* argv: --first-frame PATH */
if (first_frame_path && (init_latent_path || lock_first_frame || init_first_frame)) {
    fprintf(stderr, "--first-frame is official condition; do not mix with G3-lite\n");
    return 1;
}
```

无 `--first-frame` 时控制流与今日完全相同（仍可调用 `forward_avc` 且 `cond_seq=0`）。

- [x] **Step 2: 无 `--first-frame` 短步 checksum（A 门冒烟）**

`--steps 2 --layers 4 --skip-vae` 两次。Expected：与 E 之前 bit-for-bit。

- [x] **Step 3: 接 encode + cond pack + multimodal TE**

按上面 1–5 接线。`text_seq` 必须是 multimodal 后的长度。Metal `h3.c` 无 CFG（蒸馏路径）。CUDA cfg=5 保持今日 generate 的 `te_pad_refiner` uncond：**cond 行在 cond/uncond 两次 forward 里都在**（DiT 始终看见干净首帧），uncond 只把 text hidden 换成 pad。不要把 COND 从 uncond pack 里拿掉。

- [x] **Step 4: 官方 F2VA e2e（4090）**（接线/日志过；质量门未满，见执行记录）

输入：`input/test_1_first_frame.jpeg`（或同等骑士/人物首帧）。命令骨架：

```sh
H3_CUDA_SDPA_BACKEND=cudnn ./build-cuda/h3_cuda_generate \
  --dit "$DIT_INT8" --vae "$VAE_DEC" --vae-encode "$VAE_FP16" --te "$TE" \
  --vision "$VISION" \
  --out /tmp/h3_f2va_official --width 864 --height 480 --seconds 5 \
  --frames 17 --steps 8 --layers 50 --te-layers 50 --vae-blocks 36 \
  --cfg 5 --seed 17 --sdpa-backend cudnn \
  --token-ids-file "$PROMPT_IDS" \
  --first-frame input/test_1_first_frame.jpeg
```

`--vae-encode` / `--vision` 若与 decode/TE 同文件可省略，但 help 必须写清。

验收：

- 日志 `official_condition=1`、`keyframes=0`、`cond_seq>0`，无 `g3_lite`
- f0 vs jpeg：目视可认，且 `h3_quality_gate_probe` PSNR **≥ 28 dB**（**禁止** `--lock-first-frame` 过门）
- 后续帧同一主体、允许运动（禁止站桩）
- 无整帧闪
- 主题 prompt 仍跟随（有首帧不能塌成「只有照片」或「照片消失」）

- [x] **Step 5: 主题 A 门（无 first-frame）**

若 Step 3 动了默认 pack 路径，用锚点命令跑橙猫或对照 P3 `latent_stats`。Expected：仍 PASS / 指纹一致。

- [x] **Step 6: lite 文案**

help 与运行时 warning 含 `non-official G3-lite`。`docs/nvidia_optimization_plan.md` 中「G3 ✅ lite」改为「debug only；官方 G3 = VAE encode → COND → AdaLN 0.999」。

**本阶段不做：** `--last-frame`、Ref、速度、把 lite 当默认。

### 执行记录（阶段 E，2026-08-19）

- 接线：`tools/h3_cuda_generate.c` 官方 `--first-frame` = ffmpeg STRETCH → vision TE（`build_fl2va_ids(vision.tokens, patch grid)`，禁止 merged 1×1）→ `h3_vae_encode_f32` → patchify → AdaLN 0.999 → `fill_metal_cond_pos(text_seq)` → `forward_avc`。无 first-frame 时 `cond_seq=0` 仍走 `forward_avc`。`--init-latent` / `--init-first-frame` / `--lock-first-frame` 标 debug 并与 `--first-frame` 互斥。
- A 门：两次 `--steps 2 --layers 4 --te-layers 1 --skip-vae --cfg 5 --seed 17` + 橙猫 20 tok，无 `--first-frame`：latent **bit-identical**；`cond_seq=0`；无 `official_condition` / `g3_lite`。
- 互斥：`--first-frame` + `--init-latent` 报 `--first-frame is official condition; do not mix with G3-lite`。lite 单独启动打印 `warning: non-official G3-lite`。
- 官方 e2e（只跑一刀）：INT8 DiT / cfg=5 / reuse=off / VAE 256+deblock / 全文 1626 ids / 864×480 / 8 step / 50L。`--vae-encode`=fp16 VAE。日志：`vision.tokens=405 patch_grid=30x54`（非 1×1）、`span.tokens=407`、`text_seq=2039`、`official_condition=1 keyframes=0 cond_seq=405`、TE `finite=1`、`reuse_schedule: disabled`、无 `g3_lite` / `img_w=`。host VAE encode ~42 min（无 CUDA encode API）。
- 质量：**未过门**。f0 vs stretch jpeg **PSNR=26.134 < 28**。f0 可认骑士+蓝电剑+城堡。f3 人影糊、青光过曝；f7+ 中心亮黄白块、主体不可认。未用 `--lock-first-frame`。generate 因 `--mux` 无 `--audio-vae` rc≠0；事后 ffmpeg 合成 video-only `out.mp4`。
- 产物：`out/verify_stage_e_f2va/`。不进入 F，不 commit。

### 执行记录（阶段 E fix round，2026-08-19）

Reviewer REQUEST_CHANGES：f0 26.13<28；f7 塌缩；补 A 门日志；mux 无 audio-vae 应 rc=0。

- 修：CFG uncond 保留 Picture+vision（只 pad prompt）；mux video-only；A 门日志 `out/verify_stage_e_f2va_fix/a_gate/`（checksum `0xd46d4aebe7d15817` bit-identical）；AdaLN step0 active=1。
- 官方 cfg=5+Metal VIDEO tags：f0 **26.936**；f7 仍塌；mux **rc=0**。
- 消融（同一 `cond_rows` checksum `0xf3c1a4e830ec4089`）：cfg=1+tags f0 **27.376** f7 仍塌；cfg=1+`--no-dit-text-tags` f0 **27.319** 且 f7/f16 骑士可认；cfg=1+`--no-vision-inject` f0 **27.412** 后期可认。vision 不是 f0 瓶颈；VIDEO@σ tags 是后期塌缩开关。官方默认仍对齐 Metal tags（未升诊断旗标）。
- 质量门仍未满。产物 `out/verify_stage_e_f2va_fix/`。不进入 F，不 commit。

### 执行记录（阶段 E fix2，2026-08-19）

Parent rulings：保留 vision；CUDA INT8 官方默认不要把 Picture+vision 打成 VIDEO@σ；同 latent 试 fp16 decode；cfg=5 / reuse=off / 官方 `--first-frame` / tile+deblock / 全文 / 骑士；禁止 lite/lock 过门。

- 默认：vision span 留 TEXT AdaLN。`--dit-text-tags` 才恢复 Metal-like VIDEO@σ（debug，非 bit-match）。README / `docs/nvidia_build.md` G3 已写 CUDA INT8 例外。
- 官方一刀：cfg=5、vision on、tags off、`--cond-rows` checksum `0xf3c1a4e830ec4089`、`--vae` fp16、mux video-only **rc=0**。日志 `dit_text_tags: off`、`vae_ckpt: F16 native`、`official_condition=1`。产物 `out/verify_stage_e_f2va_fix2/`。
- 同 latent INT8 decode：f0 **26.551**；fp16 **26.544**。decode Δ ≈ 0.007 dB，**未到 28**（差 1.45 dB）。不把官方 decode 默认切 fp16；INT8 仍是速度档。t0 vs COND cosine=0.9875。
- 目视：f0/f3 骑士可认且有运动；**f7/f16 在 cfg=5 下仍是亮斑**（INT8/fp16 decode 同塌，根因在 DiT/CFG，不是 decode）。上一轮 cfg=1+tags OFF 后期可认，本轮按裁定未降 cfg。
- 未用 `--lock-first-frame`。未做 F。未 commit。

### 执行记录（阶段 E fix3，2026-08-19）

Parent rulings：T2VA 默认仍 cfg=5；官方 `--first-frame` 默认 cfg=1；保留 vision；CUDA INT8 默认不打 VIDEO tag；先测 VAE roundtrip；禁止 lite/lock；一刀骑士全文 cfg=1 reuse=off tile+deblock。

- CLI：无 `--cfg` 时 `--first-frame` → cfg=1；T2VA 仍 5。help 写清两条默认。`--skip-dit --first-frame` = VAE encode→decode 天花板（非 lite）。
- Roundtrip（`out/verify_stage_e_f2va_fix3/roundtrip/`）：COND still checksum `0xf3c1a4e830ec4089` 铺满 T=7，INT8 tile+deblock decode。f0 vs canvas **PSNR=27.013 < 28**。f0 门改为 **≥ 26.513**（roundtrip−0.5）。28 记为 encode/decode 限制。未到 30，不查 DiT t=0。
- 官方一刀：未传 `--cfg`（`explicit=0 this_run=1`）、vision on、tags off、reuse off、全文 1626、mux video-only。f0 **27.275 ≥ 26.513 PASS**。f7/f16 骑士可认、非中心亮斑。generate ssh rc=0，`summary: fl2va pack PASS`。
- 未用 `--lock-first-frame`。未做 F。未 commit。产物 `out/verify_stage_e_f2va_fix3/`。

---

### Task F: L2VA 竖切（尾帧）

**Files:**
- Modify: `tools/h3_cuda_generate.c`
- Read: `h3.c:1238-1242`（`H3_IMAGE_FIT_COVER`）、`h3_host.c:352-354`（尾帧 `condition_time`）

**Interfaces:**
- Consumes: 与 E 同一 encode / vision / `forward_avc`（不要复制 encoder）
- Produces: `--last-frame PATH`；keyframes=`[frame_count-1]`

尾帧 RoPE（`h3_host.c:352-354`，CUDA 用 float）：

```c
float condition_time = (float)text_seq
    + h3_dit_w4a8_video_span_sum(latent_t)  /* fill_metal_video_pos 里同一 rescale 累加 */
    - (5.0f / 3.0f);                        /* h3_dit_frame_rescale */
```

在 `h3_dit_w4a8.c` 把 `fill_metal_video_pos` 的时间累加抽成 `h3_dit_w4a8_video_span_sum(int latent_t)`，避免 E/F 各写一套。像素：`h3_ffmpeg_read_image_f32(..., H3_IMAGE_FIT_COVER, ...)`。

- [x] **Step 1: CLI `--last-frame`；与 `--ref-*` 互斥（H 未做时可先只拒 first+ref 的占位）**

- [x] **Step 2: 无 first/last 时 A 门冒烟**（短步 checksum）

- [x] **Step 3: e2e `--last-frame`**

验收：末帧对 jpeg 目视可认、PSNR ≥ 28 dB（绝对 28 被 VAE 天花板卡住则用 roundtrip−0.5）；**第 0 帧不是尾帧复制**；时间方向走向尾锚，不是整段锁死。

**本阶段不做：** 双锚（那是 G）；不新开 encoder。

### 执行记录（阶段 F，2026-08-19）

- 接线：`tools/h3_cuda_generate.c` 官方 `--last-frame` = ffmpeg COVER → 与 E 同一 vision TE / VAE encode / AdaLN 0.999 / `forward_avc`。`h3_dit_w4a8_video_span_sum` 从 `fill_metal_video_pos` 抽出。keyframes=`aligned_frame_count-1`（`--frames 17` → 21）。condition_time=`text_seq+span_sum-5/3`。无 `--cfg` 时 L2VA 默认 cfg=1。`--first-frame`+`--last-frame` 拒绝（G）；`--ref-*` 占位拒绝；与 lite 互斥。
- A 门：两次短步无 first/last，latent checksum `0xd46d4aebe7d15817` bit-identical（同 E）。
- Roundtrip：`--skip-dit --last-frame` COVER still。图像天花板 f0 vs COVER **27.076 < 28** → 末帧门 **≥ 26.576**。
- 官方一刀：骑士 jpeg 当 last-frame（无独立尾帧静帧；README 写清）、全文 1626、cfg=1、tags off、reuse off。日志 `official_condition=1 keyframes=21 cond_seq=405 condition_time=2074`。f21 vs COVER **26.902 ≥ 26.576 PASS**。f0 vs COVER **13.702**（不是尾帧复印）。未用 lite/lock。mux video-only rc=0。
- 产物：`out/verify_stage_f_l2va/`。未做 G。未 commit。

---

### Task G: FL2VA 竖切（首尾）

**Files:**
- Modify: `tools/h3_cuda_generate.c`（同时接受 `--first-frame` + `--last-frame`）
- Read: `tests/test_h3.c:188-211` 段序

**Interfaces:**
- Consumes: E+F
- Produces: 两段 COND 拼接进同一个 `cond_latent` / `c_pos_*`：先首帧（`condition_time=text_seq`），再尾帧（F 的公式）。`cond_seq = 2 * frame_rows`。

段序必须是 `[text | COND | COND | audio | video]`，与 `test_layout_fl2va` 一致。AdaLN 仍是一条 visual_condition 0.999 行（两段 COND 都 tag=0）。

- [x] **Step 1: 允许双旗标；`img_cond_rows` 加倍；日志 `keyframes=0,last`**

- [x] **Step 2: 与 `--ref-image` 互斥**（即使 H 未完成，也先拒绝「first+ref」）

- [x] **Step 3: e2e 双锚**

验收：首尾同时对齐（各 ≥ 28 dB / 目视可认）；中段连贯，不是两张照片硬切；A 门（无旗标）仍过。

### 执行记录（阶段 G，2026-08-19）

Parent rulings：E/F 已过；不改 F 冻结的末帧公式；双旗标两段 COND（首 `condition_time=text_seq`，尾用 F 公式）；必须两张不同静帧（骑士 jpeg STRETCH + 橙猫 png COVER）；PSNR 相对门 roundtrip−0.5；图像条件默认 cfg=1；禁止 lite/lock；不 commit；不做 H。

- 接线：`tools/h3_cuda_generate.c` 同时接受 `--first-frame`+`--last-frame`。`cond_seq=2*frame_rows`；pack `[text|cond|cond|audio|video]`；日志 `keyframes=0,last`。首帧 `fill_metal_cond_pos(text_seq)`，尾帧 `text_seq+video_span_sum-5/3`。`HARD_MAX` 提到 4096（双 Picture+1626≈2452）。`--ref-*` 仍拒绝。无 `--cfg` 时 FL2VA 默认 1。
- A 门：清 GPU 后两次短步无旗标，latent checksum **`0xd46d4aebe7d15817`** bit-identical（同 E/F）。08:09 污染对已替换。
- Roundtrip 尾：`--skip-dit --last-frame` 橙猫 COVER still。f0 vs COVER **33.962** → 相对门 **33.462**。首帧天花板沿用 E 骑士 STRETCH **27.013** → **26.513**。
- 官方一刀：骑士 jpeg + `input/test_g_last_frame_cat.png`，全文 1626，cfg=1，tags off，reuse off。日志 `official_condition=1 keyframes=0,last cond_seq=810 text_seq=2452 condition_time0=2452 condition_time1=2487`。`--cond-rows` checksum `0x9173b592191b8be7`（2×405）。mux video-only，`summary: fl2va pack PASS`。
- PSNR：f0 vs STRETCH **27.455 ≥ 26.513 PASS**。f21 vs COVER **30.441 ≥ 28 PASS**，相对 33.462 **FAIL −3.02 dB**（橙猫 VAE 天花板过高；目视已是那只猫）。f0 vs COVER **9.841**（不是尾帧复印）。未用 lite/lock。
- 目视：开头骑士对齐 first，结尾橙猫对齐 last；中段骑士运动，猫锁在末锚。
- 产物：`out/verify_stage_g_fl2va/`。未做 H。未 commit。

---

### Task H1: Ref2VA image

**Files:**
- Modify: DiT 权加载（Ref2VA safetensors；注意 `comfy_quant` 键，见 `docs/nvidia_quant_contract.md`）
- Modify: `h3_dit_w4a8.c` pack：text 之后按请求顺序插 `REF_IMAGE`（可复用 cond 缓冲，但 **AdaLN 仍走 visual_condition**；日志写 `H3_SEG_REF_IMAGE` 而不是 keyframe）
- Modify: `tools/h3_cuda_generate.c`：`--ref-image`；canvas 用 Metal 对照 `h3_reference_image_canvas`（只缩小、保比例）。CUDA 重实现该函数到 `h3_dit_w4a8.c` 或 `h3_multimodal_cuda.c`，用 `tests/test_h3.c` / `h3_host.h:101-106` 注释钉行为
- Read: `h3_host.c:364-378`、`tests/test_h3.c:214-243`

**Interfaces:**
- Consumes: B 的可插前缀、C encode、D Picture presentation（有序 `<Picture N>`）
- Produces: 切 **另一棵** DiT（Ref2VA ckpt）。缺权文件 → 明确报错，**禁止**静默回退 FL2VA + init-latent。
- 硬规则：`keyframes` 与 `references` 互斥（`h3_cli.c:313`）。

Pack 相对 FL2VA：REF 段不是 `condition_time` 钉在 0 / last，而是 `h3_host.c` 的 cursor（text_len 起每次 +1）。RoPE 用 cursor，不要复用 E 的 keyframe 公式。

- [x] **Step 1: 权加载探测** — `h3_st_inventory` 列出 Ref2VA 键；INT8/W4A8 以契约为准。打不开就 fail。

- [x] **Step 2: CLI `--ref-image` + 与 first/last 互斥**

- [x] **Step 3: 无 ref 时仍走 FL2VA INT8 主题权（A 门）**

- [x] **Step 4: e2e 单图参考**

验收：身份/场景跟随参考图；有序 Picture；A 门（无 ref）仍过。

### 执行记录（阶段 H1，2026-08-19）

- 远程开工时 **没有** Ref2VA 权；从 hf-mirror 拉 `minimax_h3_ref2va_pruned_int8_convrot.safetensors`（20970379616 B；932 tensors；I8+scale+comfy_quant；**不是** FL2VA 改名）。`--dit` 含 `fl2va` / 不含 `ref2va` / 缺文件均硬拒绝。
- Canvas：CUDA `h3_dit_w4a8_reference_image_canvas` 钉 `tests/test_h3.c` 金数字（1920×1080 vs 512×512 → 672×384）。骑士 jpeg 2048×1152 → 864×480（只缩小）。
- Pack：复用 cond 缓冲；日志 `H3_SEG_REF_IMAGE`；RoPE cursor=`text_seq` 然后 +1；目标 A/V 从 `text_seq+n` 起。AdaLN 仍 visual_condition。
- A 门：无 `--ref-image`，FL2VA INT8，checksum **`0xd46d4aebe7d15817`** bit-identical（同 E/F/G）。
- 官方一刀：骑士 jpeg + 1626 tok，cfg=1。日志 `official_reference=1 H3_SEG_REF_IMAGE n=1 text_seq=2039 cond_seq=405 target_rope_t0=2040`。`summary: ref2va pack PASS`。
- PSNR f0 vs canvas **26.865**（身份跟随）；f7 **12.056** / f0 vs f21 **11.237**（运动，不是静帧锁）。未用 lite。
- 产物：`out/verify_stage_h1_ref_image/`。未做 H2/H3。未 commit。

---

### Task H2: Ref2VA audio

**Files:**
- Modify: `tools/h3_cuda_generate.c`
- Use: 已有 `h3_audio_vae_cuda_encode`（`h3_audio_vae_cuda.h:53-59`）；probe：`h3_audio_vae_cuda_encode_probe`
- Modify: `forward_avc` 或新增 audio-cond 参数：`ref_audio_latent[audio_cond_seq, 32]`，pack 进 `H3_SEG_REF_AUDIO`，AdaLN **audio_condition time=1.0**（`h3_dit_schedule.c:126-127,147`），tag=2
- Read: Metal README 约束：参考音频 2–15s，最多 3 段，总时长 ≤15s；standalone `--ref-audio` 必须配图像或视频

**Interfaces:**
- Consumes: H1；`h3_dit_w4a8_pack_audio_latent`
- Produces: `--ref-audio WAV`（须已有 `--ref-image` 或 `--ref-video`）

- [x] **Step 1: CLI 校验 standalone audio → 报错**

- [x] **Step 2: encode → pack REF_AUDIO → AdaLN audio_condition**

实现上对称于 visual_cond：`audio_cond_t_active` + modulation 再扩一行 AUDIO_1.0，或在已有 audio splice 上当 `audio_sigma==1` 且该行是 ref 时走 condition 行。**target 音频**仍用当前 `audio_sigma`。

- [x] **Step 3: e2e**

验收：参考声进入条件行；目标音轨仍去噪生成，不是整段复制（除非 Metal 金标准就是 pin——以 `h3.c` 为准）。

### 执行记录（阶段 H2，2026-08-20）

- DiT：`forward_avc` 扩 `audio_cond_seq`；pack `[text|cond|audio_cond|audio|video]`；AdaLN 额外 AUDIO@time=1.0 行；`fill_metal_ref_audio_pos`。
- CLI：`--ref-audio` 须配 `--ref-image`/`--ref-video`；与 keyframes/lite 互斥；接 `h3_audio_vae_cuda_encode`。
- A 门：无 ref FL2VA 短步 checksum 一致。Standalone reject 文案正确。
- 官方一刀：骑士 + 3s sine，`H3_SEG_REF_AUDIO n=1 audio_cond_seq=240`，mux A/V，`summary: ref2va pack PASS`。产物 `out/verify_stage_h2_ref_audio/`。
- 未做 H3；未 commit。

---

### Task H3: Ref2VA video

**Files:**
- Modify: `tools/h3_cuda_generate.c`（`--ref-video` / `--ref-silent-video`）
- Use: `h3_ffmpeg_read_video_f32`、C 的因果 encode `T_lat = ceil(T/4)`、D 的 vision **两帧**采样 + timestamps
- Read: `h3_multimodal.c` Ref2VA `<Video N>`；`h3_host.c` `H3_LAYOUT_REF_VIDEO`

**Interfaces:**
- Consumes: H1 的 Ref ckpt 与互斥规则；C encode；扩展 D 的 presentation（本阶段才做 Video 标签）
- Produces: 参考视频视觉网格进 `H3_SEG_REF_IMAGE`（Metal 把视频视觉 token 放在 REF_IMAGE 段，见架构 §4.1）；可选内嵌/外挂音轨走 H2

- [x] **Step 1: CLI；`--ref-silent-video` 忽略声轨**

- [x] **Step 2: 每 12 帧抽一对，两帧 vision block + timestamp**（对照 `h3_vision_encode_bf16` `frames=1 or 2`）

- [x] **Step 3: e2e**

验收：运动延续；缺 Ref 权时失败信息明确；不与 first/last 混用。

### 执行记录（阶段 H3，2026-08-20）

- CLI：`--ref-video` / `--ref-silent-video`；与 keyframes / `--ref-image` 互斥；Ref2VA DiT 门。
- Vision：每 12 帧抽一对 + `<Video n>` / timestamp ids（`h3_multimodal_cuda_build_ref_video_ids`）。
- Pack：视觉进 `H3_SEG_REF_IMAGE`（Metal）；日志 `H3_SEG_REF_VIDEO→H3_SEG_REF_IMAGE`；可选嵌入音轨走 H2。
- A 门：`0xd46d4aebe7d15817`。互斥拒绝正确。
- 官方一刀：5 帧 silent 640×360，`cond_seq=440`，`summary: ref2va pack PASS`，mux `e2e/out.mp4`。host F32 encode ≈71 min（`encode_ms≈4.27e6`）；留下 `cond_rows.raw` 可复用。
- 产物：`out/verify_stage_h3_ref_video/`。未 commit。

---

### Task S: 速度（功能完整之后）

**Files:** 不要在本任务改 pack / AdaLN / cond 段逻辑。执行真源是 [`2026-08-18-nvidia-accel-r1-r3.md`](2026-08-18-nvidia-accel-r1-r3.md)。触及：`h3_nvfp4_cuda.cu`、`h3_dit_w4a8.c` INT8 调度、`h3_vae_int8_cuda.c` load；`tools/h3_cuda_generate.c` 仅计时/overlap。

**Interfaces:**
- Consumes: A 的主题门；若已接 E，F2VA 官方路径
- Produces: TE 暴露下降；主题 ≤35s **再议**。失败不得回滚 condition channel。

P3 基线（不是本阶段要「先调 img_w」的理由）：橙猫 total 63.75s（TE 23.77s）；水下 68.03s（TE 22.31s）。

- [x] **Step 1: 确认功能宣称已完成**（E 官方 F2VA 至少已合入；H 按产品是否宣称 Ref）

- [x] **Step 2: 按速度计划执行 R1/R2/R3；PR 不得夹带 `forward_avc` pack 变更**

- [~] **Step 3: 回归**

橙猫/水下主题仍 PASS（见 `out/verify_r1r3_closeout/` / P3）；≤35s **未达标**（橙猫 63.75s / 水下 68.03s；`out/verify_r1r3_final/` 亦 FAIL）。禁止 magcache / reuse / Sage / W4A8 伪收口。本轮不重跑碰运气。

**Stage S 续作（2026-08-20，`out/verify_stage_s/`）：**
- [x] Host packed 权重缓存（CFG 第二遍跳过 mmap→staging；契约不变）
- [x] Device-resident text TE（RMS/RoPE/GQA/FFN）已实现，**默认关**（`H3_NVFP4_DEV_ACT=1` opt-in；默认路径未切）
- [x] 双主题视觉 PASS（橙猫窗台；水下海龟+珊瑚无人）
- [ ] ≤35s：**仍 FAIL**（S 橙猫 total≈69.6s / TE≈27.4s；水下≈67.6s / TE≈28.7s）。剩余大头仍是 350× packed H2D+dequant+skinny GEMM；即便 TE→0，dit≈18s+vae≈4–7s 也压不到 35s。未伪收口。
---

## 每阶段共同回归（执行清单）

1. `cond_seq=0` / 无 `--first-frame` / 无 `--ref-*` → T2V 主题门或 A 夹具 bit-for-bit。
2. `tests/test_h3.c` layout 金数字不因 CUDA 工作而改。
3. 不 commit 除非用户要求；不把远程口令写进命令文档。
4. 主题默认仍是 INT8 ConvRot + cfg=5 + cudnn；W4A8 不是回归锚点。

---

## 明确不做（本计划全期）

- 为单模态复制第二套 VAE / vision / DiT pack。
- 用 G3-lite 顶掉 E–H 的 DoD。
- 把 Sage MMA、多卡、MagCache 校准、HF-BPE 纯 C tokenizer 塞进 F2VA 竖切。
- 修改 Darwin 默认 `make` 路径来「顺便」接 CUDA。
- 在 A/B 完成前把 `--first-frame` 做成产品默认。
- 把 R1–R3 / ≤35s 缠进 B–H。
- 把「f0 对得上 jpeg」当成 F2VA 完成（后续帧必须在看见干净 COND 的前提下运动）。

---

## Spec coverage（对照架构 / 路线图）

| 架构 / 路线图要求 | 本计划任务 |
| --- | --- |
| T2V 无 cond 时 CUDA DiT bit-for-bit | A + B Step 1/6 |
| packed `[text\|COND\|audio\|video]`，空 cond 退化 T2VA | B |
| AdaLN cond=0.999 / Euler 不写 cond / 共用 video_patch | B |
| 共享 VAE encode + patchify + 0.999 augment | C |
| vision + `<Picture n>` multimodal TE；无图 TE 不变 | D |
| 官方 `--first-frame` STRETCH、keyframes=`[0]` | E |
| `--last-frame` COVER、尾帧 condition_time | F |
| 双 COND、`test_layout_fl2va` 段序 | G |
| Ref 另一棵 DiT、`H3_SEG_REF_*`、与 keyframes 互斥 | H1–H3 |
| 禁止 lite 冒充官方条件 | E Step 6 + Global Constraints |
| 速度独立 | S；不在 A–H |
| 不改 Metal 生产语义 | File Structure；CUDA 不链 Accelerate/Metal GPU |
