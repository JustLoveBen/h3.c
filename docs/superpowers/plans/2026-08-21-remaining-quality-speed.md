# 剩余保质量加速菜单 + Latent Upscaler 对照

日期：2026-08-21  
仓库：`h3.c`（纯 C/CUDA MiniMax-H3 T2VA）  
硬件：RTX 4090 24GB  
本文不改代码、不跑 generate。数字来自 cafe 金线与公开源码。

**结论先说：**

1. [Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) **认真看过源码和工作流，不只是 README 标题。**
2. **现在不要接到 h3.c 质量默认。** 它不是「跳过整段 480P DiT」，而是「低分一采 + 神经网络 latent 放大 + 目标分辨率再采」。对本仓已经钉死的 864×480 / 8 step 金线，二采若也跑满 8 步会更慢；二采若只跑 3 步，墙钟可能略降，但小脸/身份/口型对白会先碎。Comfy 社区要的是 **480p→720p/2K 比原生高分便宜**，不是把已经 480p 的 8 步再砍一刀。

**2026-08-22 速度刀（质量冻结，人脸/MCU/dist-attn 不动）：**

有声 cafe q1 金标 **75.4s**（te 11.6 / **dit 47.5** / vae 9.3，checksum `0x3b3464adf6c39df2`）。同 prompt/ids/seed 的 `t2va_complex_5s_cafe_q1_spd/`：**75.3s**，checksum **相同**，ffprobe 5.000s/120。TE mmap 直传 −0.7s；PPM 整帧 fwrite + rgb24 复用（ppm 0.91 / mux 0.42）。**墙钟在噪声里。**

拆开不可破的：

| 段 | 秒 | 能否再无损压 |
| --- | ---: | --- |
| DiT 8-step dense cuDNN | **47.5** | 否。step1 10.3（unpack+plan）+ 7×5.31。Sol 已输 cuDNN 7.5× |
| TE packed H2D 12.8GiB | **~10** | 15GB mmap 无法 pin；pageable H2D ≈1.3GiB/s。进程内常驻已做 |
| VAE fp16 tile 272 4×2 + deblock | **9.4** | 勿设 `H3_VAE_TILE_PIXELS=256`（会变成 5×3=15 块，VAE 15.3s） |
| 开档 / refiner / ppm / mux | **~7** | PPM 已从「按像素 fwrite」收掉；再挤不到十秒 |

**有声 480P 真 5s 质量路径 floor ≈75s。40–50s 必须减少步数或换掉 cuDNN 注意力，不能靠再扫工程间隙假装。**

对照：[YCNodes MiniMax-H3](https://github.com/yichengup/ComfyUI-YCNodes-MiniMax-H3)（Tail / Tiled Sampler / Sigma Refiner / Distance Attention）。LightWan 侧已落地项见 [2026-08-21-lightwan-h3-speed.md](./2026-08-21-lightwan-h3-speed.md)。

**Sol-Attn 与市面加速全表**（kijai Triton / 论文 arXiv:2607.24027 / Saganaki22、Sage、cache、CAB、Optimization Suite）：见 **[2026-08-21-h3-accel-survey.md](./2026-08-21-h3-accel-survey.md)**。一句话：孤立 CUDA Sol 在 cafe `seq=15973` **比本仓 cuDNN 慢 7.5×（351 vs 47 ms）**，已停；**不进质量默认**，未跑 q2 generate。

---

## 0. 本仓现状（必须算进账，不要装没发生过）

质量金线：`t2va_complex_5s_cafe_p3`，**90.5s**，真 5.000s / 120 帧 @24fps / 864×480，无 MagCache。

| 刀 | 墙钟 | 备注 |
| --- | ---: | --- |
| 悬崖 cafe（cuDNN/NVRTC 坏） | 24.1 min | flash 慢路径；已修 |
| 修好后 p0 | ~119s | |
| p2 | 112s | |
| **p3 质量默认** | **90.5s** | TE 16.4 + DiT 50.5 + VAE 15.6；`dit_evals=8 dit_skips=0`；checksum `0x3b3464adf6c39df2` |
| MagCache A/B thresh=0.08 | 79s | skips=3；帧 RGB MAE **52–68**；**已回退，不当默认** |
| p5 无损尝试 | 99–100s | pthread cuDNN prebuild 等；比 p3 慢，已回退变慢项 |
| 8-step 4090 **无损下限实测** | **82–90s** | TE~16 + DiT~50 + VAE~16。p3 已贴这条带 |
| 有声 av8 | 109s | 现已默认 audio；相对无声金线约 +18s |
| 有声 av20 | 140s | **20 步不是同片更清**；整段加步会拉歪 8-step 蒸馏轨迹 |
| **有声 q1（A1 packed TE + audio overlap）** | **75.4s** | te 11.6 / dit 47.5 / vae 9.3；checksum 与 p3 相同，无损。**不要重跑 q1。** |
| **有声 q1_spd（mmap 直传 + PPM 整帧写）** | **75.3s** | 同 checksum `0x3b3464adf6c39df2`；te 10.8 / dit 47.7 / vae 9.4 / ppm 0.91 / mux 0.42。相对 q1 在噪声里 |
| 8-step 4090 **有声质量 floor（实测）** | **≈75s** | DiT 47.5 已是 dense cuDNN 下限。**40–50s 需减步或换注意力** |
| Sol-Attn 孤立 bench | — | cafe T=15973 H=56：cuDNN **46.7ms** vs CUDA Sol **351ms**。未跑 q2。默认 cuDNN |

禁止当质量：W4A8、关 deblock、`--frames 17` 冒充 5s、默认 magcache、TinyVAE 成片、未蒸馏强行 4 步。

产品形状（和 Upscaler / YCNodes 对账用）：

- 像素 864×480，VAE 空间 /16 → latent 30×54；DiT 2×2 patch → `video_seq = 37 × 15 × 27 = 14985`
- 5s：`frames_req=120` → align 124 → `latent_t=37`；audio `round(124×40/24)=207`，`audio_seq=414`
- cafe packed `seq=15973`（text 574 + audio 414 + video 14985）
- 8 step、cfg=1（CFG 蒸馏）、INT8 ConvRot DiT、fp16 VAE tile 256 + deblock 0.28
- 音频：Comfy `EmptyMiniMaxH3LatentAV` 同款 **pack 默认开**；`--skip-audio` 才是无声。`h3.c` 已有 `--init-latent`（二采续跑的钩子）

到 40–50s 必须跨红线。下面只列 **还没做完 / 还可测 / 已证伪** 的，不重复 p0–p3 已合入项。

---

## 1. 剩余方案分档

### A. 质量几乎无损、尚未做或未做完（对 h3.c 可落地）

这些不改采样数学，只砍 PCIe / host 间隙 / 重复反量化。预期合计大概还能挤 **数秒到十几秒**，到不了 40s。

| ID | 方案 | 原理 | 预期 | 状态 |
| --- | --- | --- | ---: | --- |
| A1 | TE NVFP4 packed 常驻 GPU | 现在每个 linear 仍 H2D packed + device dequant。15.7GB 无法 `cudaHostRegister`，每层 PCIe 是 TE 16s 的主因 | TE 16s → **8–12s**（契约 cos≈1） | LightWan 文档 P1，**未做** |
| A2 | VAE decode 核融合 | load 已摊到 1.0s；剩下 blocks ~9–15s。tile 内 SDPA 已 cuDNN，看能否再融 GEMM/AdaLN | VAE 15.6s 再砍 **1–3s** | 未做；p5 的 FFN 包装已回退 |
| A3 | 音频 VAE 与视频 decode 重叠 / 减 sync | 默认有声后 av8=109s vs p3=90.5s，差额主要在 audio pack+decode，不是「多 12 步」 | 有声墙钟向 90s 金线靠 | **未做完**；无声金线不受益 |
| A4 | 热路径再减 sync | p3 已 quiet TE/VAE 刷屏；cuDNN SDPA **不可**进 CUDA graph（p5 prebuild 已证更慢） | 可能 &lt;1s | 只剩零碎；不要再碰 SDPA prebuild |
| A5 | INT8 权窗保持 37 / 禁止无谓换页 | 已是 p3 金线 | 守住 50.5s DiT | **已做，禁止回退** |
| A6 | H3 分段 AdaLN / gate 融合（bit-exact） | Sol-Engine 称 AdaLN 是 H3 最大无损刀；Saganaki22 fusion 在 38k token 上 scale/shift 1.91×。本仓 p5 乱融 FFN 已更慢，这条要单独设计 | DiT 可能 **1–3s** | 未做；见加速普查 §2.5 |

**不要再空跑：** NVRTC/seq pad、cuDNN 失败缓存、TE device-resident act、VAE fp16 GPU upcast、window freeze、pthread SDPA prebuild。

### B. 可测的质量换速度（必须 cafe A/B，默认关）

| ID | 方案 | 原理 | 质量风险 | 对 h3.c |
| --- | --- | --- | --- | --- |
| B1 | 更保守 MagCache | 已有 `--magcache-thresh`；0.08 已炸 | 帧 MAE 50+；H3 8-step 蒸馏后再 skip 几乎必漂（[LightX2V #579](https://github.com/ModelTC/LightX2V/issues/579)） | **可再试更小 thresh**，但不要期待质量过线 |
| B2 | `--reuse N` 速度外推 | Metal 同款；隔步复用 velocity | 色块/轨迹；旧测 DiT −30–40% 但主题 PSNR 差 | CLI 已有，**质量默认关** |
| B3 | `--token-reduction` | 空间 pool 再 expand | 与 metal float RoPE 质量档不兼容（旧日志 `tr_infer_spatial` 失败） | **不进质量默认** |
| B4 | 低分一采 + latent 上采样 + 短二采 | 见第 3 章 | 小脸/身份在一采就没了；二采低 sigma 补不出 20px 五官；口型与一采音频错位 | **可 Comfy 侧 A/B，不进 C 默认**；移植成本高 |
| B5 | YCNodes Distance Attention | 时空高斯感受野，挡背景吞五官 | 不加速，略增 mask 开销；可能改善 cafe 小脸 | **质量刀，不是速度刀** |
| B6 | YCNodes Sigma Refiner | 低 sigma 尾部**加步** | 更慢。整段 8→20 已证伪（av20 140s 且不更清）；Refiner 只加密尾巴，对「中前期身份融化」无效 | **质量刀，默认会变慢** |
| B7 | 稀疏 attn / Sage2（泛称） | 跳 token 或量化 attn | 本仓 sage2 满 seq 比 cuDNN 慢数量级；稀疏伤细部 | 仅研究档 |
| B8 | **Sol-Attn**（kijai Triton / 论文） | training-free 阈值路由 + pooled-KV 近似校正；H3 需 sink 保 text/audio | 小脸/细部。**2026-08-21 孤立 CUDA Sol 输给 cuDNN 7.5×**，未跑 q2 | **已停**。不进质量默认。专章：[加速普查 §1.10](./2026-08-21-h3-accel-survey.md) |

### C. 已证伪 / 不适用于本仓

| 项 | 为什么已经死 |
| --- | --- |
| MagCache 0.08 当质量 | 79s，RGB MAE 52–68，已回退 |
| 把 `--steps` 改成 4（无 H3 4-step 蒸馏权重） | LightWan 80× 的主力在蒸馏；H3 已是 8-step CFG-distilled |
| 20 步当「更清」 | av20 140s，同片不更清；整段加步拉歪蒸馏轨迹 |
| W4A8 DiT 当主题 | 主题崩 |
| NVFP4 DiT / 5090 故事 | 4090=Ada |
| 默认 Sage2 / 稀疏 90% | 更慢或伤脸 |
| Sol-Attn 当质量默认 / 调 Triton 侧车 | 孤立 CUDA Sol 351ms vs cuDNN 47ms；Python 侧车架构禁止；小脸风险 |
| TinyVAE / 关 deblock / 无 tile 成片 | waffle |
| `--frames 17` 冒充 5s | mux ≈0.92s |
| cfg=5 质量档 | 过曝；H3 已蒸馏 cfg=1 |
| 多卡 Ulysses / CFG 并行 | 单卡 4090；cfg=1 无第二路 |
| p5 cuDNN pthread prebuild | checksum 相同但 90.5→98.8s |
| 把生成栈改成 Python/Comfy 节点 | 架构禁止 |
| YCNodes Tiled Sampler 加速 480p | 为 768p/2K **显存** 分块，cosine 融合；480p 再切 tile ≈ 乘 DiT 次数。作者写明不适用从纯噪声重去噪 |
| YCNodes Sigma Refiner / Distance Attention 当加速 | 一个加步、一个加 mask，都不减 NFE |

---

## 2. YCNodes 四件套对照（用户刚问过 Refiner）

源码目录：[yichengup/ComfyUI-YCNodes-MiniMax-H3/py](https://github.com/yichengup/ComfyUI-YCNodes-MiniMax-H3/tree/main/py)

| 节点 | 源码实际做什么 | 加速？ | 迁到 h3.c |
| --- | --- | --- | --- |
| **Image to Video (Tail)** | 官方 I2V **总是吐空 AV latent**。Tail 把一采 latent **透传**，按 T 重算关键帧锚点，width/height 对齐二采像素。[`minimax_i2v_tail.py`](https://github.com/yichengup/ComfyUI-YCNodes-MiniMax-H3/blob/main/py/minimax_i2v_tail.py) | 不加速；是二采**接线** | 本仓 `--init-latent` 已是续跑钩子。T2VA 无关键帧时更接近 `apply_keyframes=disable` |
| **Tiled Sampler** | 沿 H 或 W 切空间块，独立采样后 cosine 融合；**音频 passthrough 不参与采样**。[`h3_tiled_sampler.py`](https://github.com/yichengup/ComfyUI-YCNodes-MiniMax-H3/blob/main/py/h3_tiled_sampler.py) | 480p **更慢**；720p/2K 才是显存刚需 | 不要为 864×480 做。若未来出 720p 二采再考虑 |
| **Sigma Refiner** | 高噪头不动，从 `start_at_sigma`（默认 0.7）把尾巴重采样得更长。[`h3_sigma_refiner.py`](https://github.com/yichengup/ComfyUI-YCNodes-MiniMax-H3/blob/main/py/h3_sigma_refiner.py) `extra_steps` 默认 **+1** | **加步 = 更慢** | 可作质量实验（运动边缘颗粒），**不是修脸专用**，也救不了 cafe 中前期蜡脸 |
| **Distance Attention** | 时空高斯感受野，中前期挡住大背景同化肢体/脸。默认 `original_width=864 height=480` | 不加速 | 若要动注意力支撑域，这比稀疏 90% 更对症；需 CUDA mask + cafe A/B |

推荐 Comfy 接线（官方 README）本身就是：**一采低分 + Tail 续跑 + Tiled 高清精修**，和 Latent Upscaler 是同一条「两段采样」故事，不是同一段神经网络。

Refiner 与本仓 8 vs 20 步的关系：整段加步会重排构图；Refiner 只加密低 sigma 尾巴，理论上更可能补五官颗粒而不是重排戏。但 cafe f119 那种蜡脸是**中前期身份被背景吃掉**，对症的是 Distance Attention 或先特写，不是 Refiner，更不是 20 步。

---

## 3. Latent Upscaler 专章

仓库：https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler  
权重：https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler  
用户本地副本：[uploads/Comfyui_Minimax_h3_latent_Upscaler-0.md](/Users/digenman/.cursor/projects/Volumes-Ben-Digen-code-h3-c/uploads/Comfyui_Minimax_h3_latent_Upscaler-0.md)

读过的文件（不是只看标题）：

- `README.md` / `README_zh.md`
- [`nodes/minimax_h3_latent_upscaler_2d.py`](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/blob/main/nodes/minimax_h3_latent_upscaler_2d.py)
- [`nodes/minimax_h3_latent_upscaler_3d.py`](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/blob/main/nodes/minimax_h3_latent_upscaler_3d.py)
- 示例工作流 [`minimax_h3_r2v_Latent Upscaler example workflow3.json`](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/blob/main/workflow_templates/minimax_h3_r2v_Latent%20Upscaler%20example%20workflow3.json)
- Issues #2, #3, #7, #8, #9, #13–#21（NestedTensor、二采 shape、边缘色条、OOM、画质回退）

### 3.1 原理（源码，不是营销句）

这是一个 **24 通道 H3 VAE latent 的学习型空间放大 CNN**，不是 DiT，也不是像素超分。

**2D 节点** `VideoLatentResizer`：

- `LatentResizer`：12+12 `ResBlockEmb`（AdaLN 吃 `scale-1` embedding），中间 `F.interpolate(..., mode="bilinear")`，默认 `channels=640`
- 时间：`TemporalConv` depthwise 3D（k=5）按 `temporal_every=2` 插在 in/out 块；**T 不变，只放 H×W**
- 推理强制 `attn=False`

**3D 节点** `LatentResizer3D`：

- 全程 `Conv3d` ResBlock + TemporalConv，中间 `trilinear`
- 默认 `channels=512`，issue #16 日志：**345,280,216 params**
- `VAE_DOWNSAMPLE = 16`（与本仓 `width/16` 一致）
- 三种尺寸：`scale` / `target dimensions`（默认 UI 1280×704）/ `megapixels`；输出再对齐 `align`（默认 32）并卡回 VAE 网格
- 只支持放大；`scale=1` 原样返回

前后处理：训练用的 24 维 mean/std 逐通道归一化再反归一化。权重不进 git，放 `ComfyUI/models/latent_upscale_models/`。

它**明确不是**「decode → 像素超分 → encode」。省的是 H3 ~5B VAE 那一趟往返。README 原文：节省的是**时间不是显存**，因为精修仍在目标分辨率跑。

### 3.2 是不是「一采低分 + 上采样 + 二采」？会不会跳过 480P DiT？

**是两段采样。不是跳过目标分辨率 DiT。**

README 写了两条路：

- 预览：`[Latent] → [Upscaler] → [VAE Decode]`（**没有**二采 DiT；质量差）
- 推荐高品质：`[低清 Latent] → [Upscaler] → [Refine / Re-sample] → [VAE Decode]`

官方 r2v 示例工作流（workflow3）接线是：

1. `ResolutionSelector` **0.2 MP / 16:9 / multiple 32 → 608×352**，length 124
2. 一采 `SamplerCustomAdvanced`：`BasicScheduler simple 8 steps` + `SplitSigmas step=4` → **只跑高噪 4 步**
3. `LTXVSeparateAVLatent` 拆出 video；`MinimaxH3LatentUpscaler3D` **megapixels=1.0、align=32** → 约 1.0MP（表内 16:9 ≈ 1376×768，约 2.2×）
4. `LTXVConcatAVLatent` 把**未放大的 audio latent 拼回去**
5. 二采 `SamplerCustomAdvanced`：`ManualSigmas` **3 步** `0.9035, 0.6316, 0.3158, 0.0000`（从 σ≈0.90 的低噪尾巴精修）
6. 高分 `VAEDecode` + `VAEDecodeAudio`

所以：

- 一采 DiT 在 **608×352**，不是 480p
- 放大是 **345M 3D CNN**，不是 DiT
- 二采 DiT 在 **~1MP / 720p 级**，3 步，**没有跳过**
- 预览 TinyVAE（`taeh3`）只给 `ModelPreviewOverrideKJ`，不是成片

Issue [#8](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/8) 还指出示例里 scheduler 用的是未 patch 的 shift 12 表，而模型走 MiniMaxH3SigmaShift 6/3——二采 sigma 与一采切分点可能对不齐。这是工作流成熟度问题，不是「免费跳过 480p」。

### 3.3 和「先低分再超分」的墙钟账（会不会反而更慢）

本仓 480p 一步的代价已经测过：step1 ≈11s（INT8 窗 unpack + cuDNN 建图，**几乎不随分辨率线性掉**），稳态 ≈5.33s（随 packed seq 的 attn/GEMM）。

视频 token 近似 ∝ H×W（T 相同）。注意力 ~ seq²，但 packed 里还有 text 574 + audio 414 常数项。

| 配方 | DiT 账（4090 粗算） | 其它 | 相对 p3 90.5s |
| --- | --- | --- | --- |
| **金线：864×480 × 8 步** | 11 + 7×5.33 ≈ **50s** | TE 16 + VAE 16 | **90.5s** |
| 608×352 × 4 步 + CNN + **864×480 × 3 步** | 一采 step1 仍 ~11s + 3×~2s ≈ 17s；二采换 seq 再建图 ~11s + 2×5.33 ≈ 22s；CNN ~2–5s | TE 16 + VAE 16（只解一次 480p） | **约 73–80s**，可能略快 |
| 608×352 × 8 步 + CNN + 480p × **8 步** | ~23 + 50 + CNN | TE+VAE | **~108s，更慢** |
| 864×480 × 8 步 + CNN + **720p × 3 步** | 50 + 720p 每步 seq 大 2.2–2.6×，attn 更疼，3 步很容易 60s+ | VAE 更大 | **明显更慢**；产品分辨率变了，不是加速 480p |
| 朴素 decode→像素超分→encode→二采 | 同上 + **两趟 5B VAE** | 本仓 VAE 已 16s/趟 | 正是 Upscaler 声称省掉的；对本仓 480p 成片无意义 |

关键点：

- **step1 unpack 11s 会付两次**（两个不同 seq 的 cuDNN 图），除非工程上缓存两套图。
- 加速只成立于：**一采少步 + 二采少步 + 一采空间 token 明显更少**。
- 若二采仍是满 8 步 480p，算术上 **不可能** 快过直接 480p 8 步。
- 社区收益来自「原生 720p 8 步太贵，改成 0.2MP 4 步 + 3 步 720p」。**本仓金线已经是 480p**，抄这条会把 480p 变成「超分目标」——等于主动降一采分辨率。

Issue [#18](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/18)：15s 视频 3D 512 通道全 T 前向在 16GB 上 OOM。本仓 5s `latent_t=37` 在 4090 上大概率能进，但 10s 要时间分块。Issue [#20](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/20)：3D 节点合并 resize 模式后有人报底边光带、**生成变慢、画质下降**。

### 3.4 质量风险：小脸 / 声音

**小脸**

- cafe 480p 上 20–30px 的脸已经在高频残差区。一采若是 608×352，脸再缩到 ~14–20px。4 步高噪只定大形。
- 3 步从 σ≈0.90 精修补的是纹理，**补不出一采没有的身份**。这和「Distance Attention 对中前期同化」是同一层问题。
- Issue [#2](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/2)：作者对比图里 **像素 VAE 超分看起来比 latent 超分更好**——连 README 配图都承认学习型 latent 不是画质上限。
- 非整数倍 / 非 32 对齐会边缘色条（[#3](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/3)、[#6](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/6)）。本仓硬性 `pixel % 32 == 0`、latent H/W 偶数；432×240 **非法**，608×352 / 448×256 才合法。
- I2V/F2V 二采条件空间对不齐会直接 shape mismatch（[#7](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/7)：`[300,96]` vs `[1200,96]`）。T2VA 无 keyframe 会好一些，但仍要重算 packed 位置。

**声音**

- H3 latent 是 `NestedTensor(video, audio)`。节点默认当普通 tensor，会 `clone`/`dim` 崩（[#13](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/13)、[#15](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/15)）。
- 正确做法：拆 AV → 只放大 video → audio **passthrough**（[#9](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/pull/9) 未合并进当前 3D 节点；示例工作流用 `LTXVSeparateAVLatent` / `Concat`）。
- 音频时间维跟帧数走（40 fps latent），**不跟空间分辨率走**。一采 608×352 的对白可以「还能听」，但二采改了嘴型之后 **口型-音频是一采锁定的**。本仓默认 audio 的产品不能假装这件事不存在。
- EasyCache/MagCache 伤 H3 音频已有社区记录；两段采样再叠加 skip 更危险。

### 3.5 对 h3.c C/CUDA 可否迁移

| 层 | 判定 |
| --- | --- |
| 数学可复现 | 可以。24ch mean/std、ResBlock+interpolate、T 不变，都是常规 CNN。 |
| 工程成本 | **高。** 345M 3D 权要从 HF 拉进本仓加载器；C/CUDA 新网 vs 架构禁止的 Python 侧车。2D 稍轻但仍是新模型。 |
| 已有钩子 | `--init-latent` ≈ Tail；缺的是 CNN 放大和「从 σ=0.90 只跑 3 步」的调度 CLI。 |
| 质量默认 | **不迁。** 金线是 864×480 满 8 步；把一采降到 0.2MP 等于主动放弃 cafe 小脸。 |
| 若只想要 720p 产品 | 这条才对症；那是换产品定义，不是加速现有 480p 金线。 |
| 预览档 | 「放大后直接 VAE」可当草稿，红线禁止当质量。 |

**一句话：** 社区方案对「想出 720p、显存不够原生 8 步」有意义；对「h3.c 已经 90.5s 真 480p 8 步、还要保 cafe 小脸和默认音频」**不值得接进质量路径**。若做，只能当 B 档 Comfy A/B，对照 cafe_p3 帧 MAE / 口型，而不是 C 默认。

---

## 4. 建议顺序（仍不改代码）

1. **A1 / A3 已落地**（q1 有声 75.4s）。mmap 直传不再有量级收益。
2. **不要**把 Latent Upscaler、Tiled Sampler、Sigma Refiner、MagCache、20 步、Sol-Attn 接进质量默认。
3. 小脸是质量刀（MCU / Distance Attention），**不是速度刀**；本轮已冻结。
4. **有声 480P 5s 质量 floor ≈75s**（DiT 47.5s）。40–50s 必须减步或换掉 cuDNN 注意力。

---

## 5. URL

- https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler
- https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/blob/main/nodes/minimax_h3_latent_upscaler_2d.py
- https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/blob/main/nodes/minimax_h3_latent_upscaler_3d.py
- https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/blob/main/workflow_templates/minimax_h3_r2v_Latent%20Upscaler%20example%20workflow3.json
- https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler
- https://github.com/Ttl/ComfyUi_NNLatentUpscale （架构祖先）
- https://github.com/yichengup/ComfyUI-YCNodes-MiniMax-H3
- https://github.com/ModelTC/LightX2V/issues/579
- 本仓金线：`out/verify_480p_quality_full/t2va_complex_5s_cafe_p3/`
- LightWan 已落地项：[2026-08-21-lightwan-h3-speed.md](./2026-08-21-lightwan-h3-speed.md)
- Sol-Attn + 市面加速普查：[2026-08-21-h3-accel-survey.md](./2026-08-21-h3-accel-survey.md)
- https://github.com/kijai/ComfyUI-SolAttn_triton
- https://arxiv.org/abs/2607.24027
- https://github.com/Saganaki22/ComfyUI-sol-attn
