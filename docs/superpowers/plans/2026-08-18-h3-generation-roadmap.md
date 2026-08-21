# H3 生成能力路线图

> **For agentic workers:** 先读架构真源 [`2026-08-18-h3-generation-architecture.md`](2026-08-18-h3-generation-architecture.md)。本文件按**能力竖切**排期，不是 54 个微步骤，也不是补丁清单。推荐执行：superpowers:subagent-driven-development（一阶段一子代理）或 executing-plans。

**Goal:** 在不破坏 Metal 的前提下，让 CUDA 旁路按官方语义交付 T2V/T2AV、F2VA、L2VA、FL2VA、Ref2VA（image/audio/video），并在功能完整后单独收口主题路径速度。

**Architecture:** 先把 CUDA DiT 扩成可插入 condition 的 packed 序列（空 condition 与今日 T2VA bit-for-bit），再共享 VAE encode / vision+multimodal TE，再按首帧 → 尾帧 → 首尾 → 参考竖切。禁止用 `--init-latent` lite 冒充任一条件模态。

**Tech Stack:** C11；Metal 为金标准（只读参考）；CUDA 12 / cuDNN SDPA / INT8 ConvRot DiT / NVFP4 TE / fp16 或 INT8 Video VAE / Audio VAE FP32。

**Spec:** [`2026-08-18-h3-generation-architecture.md`](2026-08-18-h3-generation-architecture.md)（主）；`docs/nvidia_optimization_plan.md` §15/§22（算力与主题基线，G3-lite 以降级为准）；`docs/nvidia_quant_contract.md`、`docs/nvidia_build.md`。

## Global Constraints

- 不改 Metal 生产语义（`h3.c` / `h3_host.c` / `h3_dit.c` / encoder 仅作对照；除非架构文档明确允许的只读测试夹具）。
- 无 condition 时 CUDA DiT **bit-for-bit** 不变（第一道门）。
- 主题默认：INT8 ConvRot DiT + 修复 NVFP4 TE + cuDNN SDPA；cfg=5；reuse/magcache/Sage **默认关**。
- 禁止 `--init-latent` lite 冒充 F2VA/L2VA/FL2VA/Ref2VA。
- 速度（R1–R3、≤35s）= **阶段 H**，不与 condition 缠在一起。
- 不 git commit（除非用户当轮明确要求）；密码不上库；不把「再跑一条 4090 碰运气」当收口。
- 旧计划 `2026-08-18-f2va-vision-condition-channel.md` 可参考接口形状，**不要按其 Task 1–9 当执行序**。

---

## 任务总表

| 阶段 | 名称 | 依赖 | 产物 | DoD 摘要 |
| --- | --- | --- | --- | --- |
| **A** | T2V 回归门（契约） | — | 无 condition 对照夹具 + 文档化命令 | 今日主题路径可复现；后续任何 pack 改动必须先过此门 |
| **B** | 官方 condition 骨架 | A | CUDA pack 可插 COND；`cond_seq=0` ≡ `forward_av` | 空 condition bit-for-bit；有 condition 时 Euler 不写 cond 行 |
| **C** | 共享视觉条件 payload | B | 视频 VAE encode + patchify + 0.999 augment | jpeg/视频帧 → 96-d 行；几何与 Metal encoder 对齐 |
| **D** | 共享 vision + multimodal TE | C（可部分∥ B） | CUDA 能把图编进 TE presentation | FL2VA `<Picture n>` 与纯文本 TE 可切换；无图时 TE 不变 |
| **E** | F2VA 竖切 | B+C+D | `--first-frame` 官方路径 | 主题不回归；f0 对齐；后续可认；不闪；**禁用 lite 作为默认** |
| **F** | L2VA 竖切 | E 的 cond 骨架 | `--last-frame` | 尾帧对齐；开头非尾帧复制 |
| **G** | FL2VA 竖切 | E+F | 双 keyframe | 首尾同时对齐；中段插值 |
| **H** | Ref2VA 竖切 | B+C+D，建议 G 后 | Ref ckpt + `H3_SEG_REF_*` | image → audio → video 三刀；与 keyframes 互斥 |
| **S** | 速度（R1–R3 残余） | **功能宣称完成之后**；至少 A 持续为回归 | TE 暴露下降；主题 ≤35s 再议 | 不得改 pack/AdaLN 换速度；橙猫/水下仍 PASS |

竖切原则：**不要**「先调 img_w → 再加 `--first-frame` → 再补 encoder」。E/F/G/H 只接线与验收；数学在 B–D。

```text
A 回归门
 └── B condition 骨架（空 cond = 今日 T2VA）
      ├── C VAE encode payload
      └── D vision + multimodal TE
           ├── E F2VA
           ├── F L2VA
           ├── G FL2VA
           └── H Ref2VA（image → audio → video）
S 速度 ── 与 B–H 代码隔离；H 完成后再作为默认质量路径收口
```

---

## 阶段 A — T2V 回归门

**目标：** 把「无 condition 的 CUDA DiT」冻成后续所有工作的对照面。没有这道门，插入 COND 无法证明没破坏文生。

**依赖：** 无。主题 PASS 已存在（`out/verify_r1r3_p3_vae/`、`out/verify_theme_ok/`）。

**主要文件：**

- 只读：`tools/h3_cuda_generate.c`、`h3_dit_w4a8.c`（`forward_av_f32`）
- 新建测试：`tests/test_h3_dit_w4a8_cond.c`（本阶段可只测 `forward_av` 自洽；B 再扩 `forward_avc`）
- 文档：在 generate 日志或测试 README 写清「回归锚点」命令（不写密码）

**接口（冻结）：**

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

pack 必须保持 `[text | audio | video]`。

**验收：**

- [ ] 固定 seed + token-ids + INT8 DiT + cfg=5 + reuse=off，橙猫/水下主题仍 PASS（或与 P3 锚点像素差在既有容差内）。
- [ ] 无 `--init-latent` / 无 first-frame 时，阶段 B 合入后 velocity **bit-for-bit**（见 B）。
- [ ] 文档写明：W4A8 不是主题回归锚点。

**不做：** 改 TE、改 VAE、改 F2VA lite、冲 ≤35s。

---

## 阶段 B — 官方 condition 骨架

**目标：** CUDA DiT 学会官方段顺序与 AdaLN/RoPE/Euler 规则。此阶段 **不读 jpeg、不加产品 CLI**。

**依赖：** A。

**主要文件：**

- 修改：`h3_dit_w4a8.h` / `h3_dit_w4a8.c`（`forward_av` 改为 `cond_seq=0` 的 `forward_avc` 包装）
- 修改：AdaLN `row_map` 生成（今日只有 MOD_TEXT/AUDIO/VIDEO × 单一 video t_row）
- 可选复用：`h3_host.c` 的 `h3_layout_build`（CUDA 链接 host 布局，避免手写第二套段逻辑）
- 测试：`tests/test_h3_dit_w4a8_cond.c`
- **不改** `tools/h3_cuda_generate.c` 的产品默认（除非 `cond_seq=0` 接线）

**接口（本阶段产出，后续阶段只填数据）：**

```c
/* Pack [text | cond | audio | video] — Metal keyframe 顺序。
 * cond_seq=0 → 与 h3_dit_w4a8_forward_av_f32 逐 bit 相同。
 * cond 行参与 attention；velocity 只写 video/audio target。 */
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
```

AdaLN（对齐 `h3_dit_schedule.c`）：

- 有 `cond_seq>0` 时增加 visual_condition 时间行，`times = 0.999f`。
- 当前 video σ ≥ 0.999 时条件行可与 video 行合并（与 Metal 相同）。
- modality：cond 行 tag=0（video），不是 text。

RoPE：cond 用调用方提供的 `c_pos_*`（阶段 E/F 按 keyframe 公式填；本阶段测试可用假坐标）。

**验收：**

- [ ] `cond_seq=0`：`forward_avc` 与现 `forward_av` 输出 bit-for-bit（同一输入缓冲）。
- [ ] `cond_seq>0`：`seq = text+cond+audio+video`；final 只写 `video_seq*96` 与 `audio_seq*32`。
- [ ] 现有 T2V/T2AV generate（不传 cond）墙钟与数值不因包装函数回退。
- [ ] 无 jpeg 依赖即可在 4090 或 host 测试上证明。

**不做：** `--first-frame`、VAE encoder、vision、Ref2VA ckpt、改 lite img_w。

---

## 阶段 C — 共享视觉条件 payload

**目标：** 任意「需要像素条件」的模态（F2/L/FL/Ref 图/视频）共用一条 encode 链：RGB → VAE encoder → `[24,T,H,W]` → patchify 96 → 0.999 augment。

**依赖：** 无硬依赖 B，但 E 之前必须完成。建议 B 并行后合入。

**主要文件：**

- 新建：`h3_vae_encode_f32.h` / `h3_vae_encode_f32.c`（正确性优先，host F32）
- 后续加速（仍属本阶段或本阶段附录，**不是**阶段 S）：`h3_vae_int8_encode_cuda.*` 或 fp16 encoder CUDA
- 只读对照：`h3_video_encoder.c`（256px tile、64 overlap、`h3_video_encoder_latent_t`）
- 复用：`h3_dit_patchify_video`（Metal 已有；CUDA 侧应对齐同一 layout）
- 权：`minimax_h3_video_vae_fp16.safetensors` 的 **encoder.***（INT8 VAE 文件偏 decoder，不要假设能 encode）
- 构建：`CMakeLists.txt`

**接口：**

```c
int h3_vae_encode_f32(
    const char *vae_path,
    const float *rgb01_chtw, int frames, int height, int width,
    h3_video_latent *out, /* [24,T,H,W] 与 Metal 一致 */
    char *error, size_t error_size);
```

augment 不要写进 encoder；generate 路径调用与 Metal 相同的 `0.999 + 0.001 * N(seed)`。

**验收：**

- [ ] 单帧 256 对齐 canvas：相对 Metal `h3_video_vae_encode` max abs 在文档阈值内（计划执行时用实测钉死，目标量级 5e-2 单 tile / 再收）。
- [ ] 输出 `T = h3_video_encoder_latent_t(frames)`，空间为 16 倍数。
- [ ] 无 DiT 也能单测。
- [ ] **不**把 encode 结果写成「已经是 F2VA」。

**不做：** 产品 `--first-frame` 默认走这条就宣布完成；不在本阶段做 vision。

---

## 阶段 D — 共享 vision + multimodal TE

**目标：** 官方 F2VA/Ref **不是**「只有 VAE 行」。TE 必须能吃 `<Picture n>` / 后续 `<Video n>`。无视觉输入时，现有 token-ids TE **不变**。

**依赖：** 现有 `h3_nvfp4*` 文本层。不必等 C 的数值收敛，但 E 的产品验收依赖 D。

**主要文件：**

- 对照：`h3_vision_encoder.c`、`h3_multimodal.c`
- 新建 CUDA/host 视觉塔（可先 host F32 正确性，再进 GPU）
- TE：把 vision tokens 与 prompt ids 拼成 Metal 同款 presentation，再走 NVFP4 层
- `h3_token_ids.c` 保持纯文本路径

**验收：**

- [ ] 无图：与阶段 A 的 TE hidden 一致（或 cos≈1 的既有 TE 契约）。
- [ ] 有单图：序列含 Picture presentation；hidden 与「纯改 prompt 字符串假装有图」可区分。
- [ ] 不接 DiT 也能测 TE。
- [ ] G1 的 hi_first / unswizzle / pre_quant **不回退**。

**不做：** 把 vision 做成 F2VA 专用分叉；不在本阶段接 Ref 视频双帧采样（放到 H）。

---

## 阶段 E — F2VA 竖切（首帧）

**目标：** CUDA 产品入口使用 **官方** `--first-frame IMG`：C 的 cond 行 + D 的 multimodal TE + B 的 pack；keyframes=`[0]`；RoPE `condition_time = text_len`。

**依赖：** B + C + D。

**主要文件：**

- `tools/h3_cuda_generate.c`：新增 `--first-frame`；**默认不再推荐** `--init-latent` 当 F2VA
- 调用 `forward_avc`；cond 行不进 Euler 写回
- ffmpeg 读图：stretch 到 render canvas（对齐 Metal `H3_IMAGE_FIT_STRETCH`）
- help 文本：lite 开关标为 debug

**验收：**

- [ ] **A 门：** 不传 `--first-frame` 时与阶段 A bit-for-bit / 主题 PASS。
- [ ] f0 对 jpeg 目视可认且 PSNR 门槛（执行时钉数字；不要靠 `--lock-first-frame`）。
- [ ] 后续帧同一主体、允许运动（骑士类镜头不能站桩）。
- [ ] 闪烁门：无整帧闪。
- [ ] 主题 prompt 仍跟随（不能因为有首帧就塌成「只有照片、无视文本」或「只有文本、照片消失」）。
- [ ] 日志写 `cond_seq=` / `keyframes=0`，不得写 `img_w=` 作为官方路径。

**降级：** `--init-latent` / `--lock-first-frame` / `--init-first-frame` 保留则必须打印 `non-official G3-lite`。

---

## 阶段 F — L2VA 竖切（尾帧）

**目标：** `--last-frame` → 单段 COND；`keyframes=[frame_count-1]`；`condition_time = text_len + video_span - frame_rescale`；像素 **cover+center crop**。

**依赖：** E（同一 cond 骨架；不要复制 encoder）。

**主要文件：** `tools/h3_cuda_generate.c`；RoPE 填法对齐 `h3_host.c:348-358`。

**验收：**

- [ ] 无 last/first 时 A 门仍过。
- [ ] 末帧对 jpeg 对齐；第 0 帧不是尾帧复制。
- [ ] 时间方向从噪声走向尾锚，而不是整段锁死。

---

## 阶段 G — FL2VA 竖切（首尾）

**目标：** `--first-frame` + `--last-frame` → **两段** `H3_SEG_COND`；`img_cond_rows` 加倍；与 `tests/test_h3.c` `test_layout_fl2va` 段序一致（text, cond, cond, audio, video）。

**依赖：** E + F。

**验收：**

- [ ] 首尾同时对齐。
- [ ] 中段连贯，不是两张照片硬切。
- [ ] 与 references 互斥（若已开始做 H，拒绝混用）。

---

## 阶段 H — Ref2VA 竖切

**目标：** 有序参考走 **另一棵** DiT（Ref2VA ckpt），pack 为 `H3_SEG_REF_IMAGE` / `H3_SEG_REF_AUDIO`，不是再加 keyframe。

建议内部三刀（仍是同一阶段，可分开验收，但不要拆成与骨架无关的补丁仓）：

| 刀 | 输入 | 额外工作 |
| --- | --- | --- |
| H1 image | `--ref-image` | 切 Ref2VA 权；参考 canvas 只缩小；有序 Picture |
| H2 audio | `--ref-audio`（须配图/视频） | 接已有 `h3_audio_vae_cuda_encode` → REF_AUDIO；AdaLN audio_condition time=1.0 |
| H3 video | `--ref-video` / silent / video-audio | 因果 encode `ceil(T/4)`；vision 两帧采样 + timestamps |

**依赖：** B+C+D；建议 G 完成后再做，避免 FL2VA 与 Ref 段逻辑搅在一起。H1 不依赖 G 的双 keyframe。

**主要文件：**

- 权加载：Ref2VA safetensors（注意 `comfy_quant` 键差异，见 quant 契约）
- `h3_layout_build` 复用
- generate CLI 与 Metal `main.c` 旗标对齐
- **禁止** references + first/last 同时开

**验收：**

- [ ] H1：身份/场景跟随参考图；A 门（无 ref）仍过。
- [ ] H2：参考声进入条件行；目标音轨仍可生成而非整段复制（除非官方语义就是 pin——以 Metal 为准）。
- [ ] H3：运动延续；`--ref-silent-video` 忽略声轨。
- [ ] 缺 Ref2VA 权文件时失败信息明确，不静默回退 FL2VA+init-latent。

---

## 阶段 S — 速度（功能完整之后）

**目标：** 在 **不改 condition 语义** 的前提下，处理 R1–R3 残余：TE 暴露、主题路径 ≤35s、VAE 冷启。这是独立阶段，不是 E 的子任务。

**依赖：** 至少 A 持续为回归。对外宣称「F2VA 已官方化」之后再把 S 当默认质量收口；S 的代码审阅不得夹带 pack 变更。

**主要文件：** `h3_nvfp4_cuda.cu`、`h3_dit_w4a8.c` INT8 路径、`h3_vae_int8_cuda.c` load；`tools/h3_cuda_generate.c` 仅计时/overlap。

**验收：**

- [ ] 橙猫/水下主题仍 PASS。
- [ ] 若已接 E：F2VA 官方路径不回退。
- [ ] ≤35s：单独记录；**失败不得回滚 condition channel**。
- [ ] 禁止 magcache/reuse/Sage/W4A8 伪收口。

历史基线见 `nvidia_optimization_plan.md` §22 与 `2026-08-18-r1r3-closeout-fixes.md`（P3：~64–68s，TE ~23s）。那些数字是速度问题，不是「先把 img_w 调对」的理由。

---

## 每阶段共同回归（复制到执行清单）

1. `cond_seq=0` / 无 `--first-frame` / 无 `--ref-*` → T2V 主题门。
2. Metal 测试 `tests/test_h3.c` layout 金数字不因 CUDA 工作而改（本路线默认不改 Metal）。
3. 不 commit 除非用户要求；不把远程口令写进命令文档。

---

## 明确不做（本路线全期）

- 为单模态复制第二套 VAE/vision/DiT pack。
- 用 G3-lite 顶掉 E–H 的 DoD。
- 把 Sage MMA、多卡、MagCache 校准、HF-BPE 纯 C tokenizer 塞进 F2VA 竖切。
- 修改 Darwin 默认 `make` 路径来「顺便」接 CUDA。

---

## 执行方式

计划已保存。架构边界以 `2026-08-18-h3-generation-architecture.md` 为准。

1. **Subagent-Driven（推荐）** — 一阶段一子代理，阶段结束对照本文验收表。
2. **Inline** — 本会话按 A→B→… 执行，每阶段停下来过门。

从 **阶段 A** 开始；在 A/B 完成前不要实装 `--first-frame` 产品默认。
