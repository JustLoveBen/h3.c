# h3.c 多卡 RTX 5090 机制级加速普查（五方案对照）

> **后续优化方案见** [2026-09-11-h3c-next-opt-plan.md](./2026-09-11-h3c-next-opt-plan.md)  
> **机制深潜（代码级，含 RH Lightning × 多卡 RTX 6000D 专章）：** [2026-09-11-h3c-accel-survey-tech-deepdive.md](./2026-09-11-h3c-accel-survey-tech-deepdive.md)  
> **少步为何好看（8-step CFG 蒸馏 vs 4-step 学生 vs draft+refine）：** [2026-09-11-h3-few-step-quality.md](./2026-09-11-h3-few-step-quality.md)
>
> **For agentic workers:** 本文是调研 + 可执行技术路线，**禁止**改产品默认路径 / `h3_cuda_generate` 默认 / 单卡金线 KPI。后续落地用 superpowers:subagent-driven-development（推荐）或 executing-plans；步骤用 checkbox。  
> **冲突裁决：** 金线 KPI 以 [next-tech-plan](./2026-08-26-h3c-next-tech-plan.md) 为准；机制 ROI 以 [tech-deepdive](./2026-08-25-h3c-vs-sglang-vllm-tech-deepdive.md) 为准；多卡另册以 [multi-gpu-arch](./2026-08-25-h3c-multi-gpu-arch-speed.md) 为准；五方案代码级拆解以 [accel-survey-tech-deepdive](./2026-09-11-h3c-accel-survey-tech-deepdive.md) 为准。本文不覆盖、不改写 75s/66s。

**Goal:** 在守住 INT8 8-step + cuDNN 金线（MAE∩有声）的前提下，把 FastVideo / Sol Super Accel / VDN / Fal H3 Max / RunningHub Lightning 拆到机制层，给出多卡 5090 可抄 / 永不碰清单与分阶段对照实验。

**Architecture:** 金线永远是单卡 dense cuDNN；多卡 Ulysses/TP 与少步 LoRA / 稀疏 attn / TinyVAE / draft+refine 一律 opt-in 另册 KPI。对照只认 cafe 同 prompt/seed 的 PPM MAE + 口型/有声，禁止用对方 SSIM 或 50-step 倍数改写 8-step。

**Tech Stack:** 本仓 C/CUDA；对照仓均为 PyTorch（FastVideo / Diffusers / SGLang / Sol-Engine）。隔离下载在 `third_party/accel_survey/`（gitignore 大树，权重未下）。

**Spec / 底稿（短链，勿大改旧文）：**

| 文档 | 用途 |
| --- | --- |
| [accel-survey-tech-deepdive](./2026-09-11-h3c-accel-survey-tech-deepdive.md) | 五方案代码级深潜；RH 6000D 多卡效率拆账 |
| [pcie-multigpu-lessons](./2026-09-11-h3c-pcie-multigpu-lessons.md) | 无 NVLink 实际链路、少 TP 多 Ulysses、5090 可抄/不抄 |
| [few-step-quality](./2026-09-11-h3-few-step-quality.md) | 官方 8-step CFG 蒸馏 vs 社区 4-step / draft+refine；为何不能再砍金线 |
| [tech-deepdive](./2026-08-25-h3c-vs-sglang-vllm-tech-deepdive.md) | 误差预算、Amdahl、框架 vs 本仓 |
| [next-tech-plan](./2026-08-26-h3c-next-tech-plan.md) | P0–P3 + 多卡另册 |
| [multi-gpu-arch](./2026-08-25-h3c-multi-gpu-arch-speed.md) | SDPA Tier、5090 sage3 已测 |
| [sla-vs-h3c](./2026-08-25-sla-vs-h3c.md) | 4-step Turbo LoRA 禁接金线 |
| [4090-accel-plan](./2026-08-25-h3c-4090-accel-plan.md) | 官方 4.44× 怎么读 |
| [8-21 普查](./2026-08-21-h3-accel-survey.md) | Sol-Attn 论文底稿（已被 8-25 superseded） |

## Global Constraints（全文隐含）

- 质量默认：**INT8 ConvRot DiT + fp16 VAE tile+deblock，cfg=1，steps=8，864×480，有声**；`sdpa=cudnn`；reuse / mag / tea / fb / sage / sol **默认关**
- 单卡金线：**4090 cafe ~75s**；**5090 cafe cudnn warm ~66s（DiT ~31.4s）**
- 已测不过门：Tea/Mag skip 有 DiT 收益但 MAE 未过门 / 历史 FAIL；sage3 DiT −17% MAE~27；sol_cute MAE~35
- **禁止**把官方 4.44× / Super Accel **22.2×** / FastH3 14× / RH 12.2× / Fal ~35× 直接写成 8-step 金线倍数
- 多卡另册 KPI，不污染 75s/66s
- 真 `--seconds`；设备现网 4090 `haiyang@192.168.31.34`、5090 `yongbin@192.168.31.238`
- 密码不入库；未要求不 commit / 不 push

---

## 0. 结论先行（多卡 5090 + 保效果）

目标是：**多卡 RTX 5090 上机制级加速，同时守住金线视觉/有声。** 五方案里真正能同时满足这两句的，几乎没有「整栈平移」。能做的是拆刀、分账、另册。

### 该抄（机制，opt-in / 另册）

1. **多卡拓扑思想（RunningHub，最接近消费卡 PCIe）**  
   无 NVLink 时 **TP2 + Ulysses4 优于 TP4 + Ulysses2**（他们 15s 768p：54.0s → 48.2s，约 12% 且少 ~14 GiB）。本仓若做 2/4/8×5090，第一刀应是 **序列并行切 packed SDPA**，权重仍 INT8 8-step dense，质量门相对单卡 5090 cudnn。这是唯一不换采样轨迹、有机会过 MAE∩有声的大延迟杠杆。

2. **VDN 的「保效果」哲学（不要抄进默认核）**  
   他们故意 **停在 8-NFE**、拒绝激进 4-step；hybrid 是 **局部 softmax 精确 + 远端线性**，text/audio 仍 dense，边界锚点只加 3.57% 密度。对本仓：8-step 金线已经是「少步合同」；可研究的是 **有声 packed 上的 window+anchor 思想**，但必须当独立 attn 后端、过 MAE 才谈。近无损是他们相对 **dense BF16 H3** 的口头合同，**不是**相对本仓 INT8+cudnn 的 MAE 门。

3. **Sol Super Accel 的流水线分账（当独立 SKU，不当金线）**  
   低分草稿 4-step → 轻量 decode（TAEH3）→ 高分 3-step refine；**最终音频用 Stage1 H3 PCM mux**，不靠 refine 轨口型。若产品要「快档 768p」，这是机制模板；金线 864×480 8-step **永不**改成 draft+LTX。

4. **Fal 的质量门纪律（流程，无代码）**  
   「优化只有在质量评测站住才留下」——对齐本仓「失败即关、不过门不进默认」。秒数与栈不可复现，只抄门，不抄 2.5s 叙事。

5. **FastH3 的稀疏蒸馏配方（研究档）**  
   DMD2 4-call + 训练期 VSA（video-video 90% 稀疏，text/audio dense）。可对照、可在隔离目录跑 PyTorch；**不可**把 VSA 学生 LoRA 插进本仓 INT8 8-step 金线（换权 + 换 attn + 换步数，三重越界）。

### 永不碰（金线 / 默认路径）

| 禁 | 为什么 |
| --- | --- |
| 4-step Turbo / LightX2V / FastH3 / RH 蒸馏 LoRA 接 INT8 金线 | 换采样轨迹；SLA 专章已判 C；本仓无产品级 LoRA 加载器进默认 |
| 默认开 sage3 / sol_cute / VSA / Cache-DiT / Mag / Tea | 已测 MAE 27–35 或历史 FAIL；8-step 跳步预算极窄 |
| TAEH3 / TAEHV / TinyVAE 当金线成片解码 | Super Accel 自己也承认轻量解码非官方 VAE；本仓 fp16 tile+deblock 是质量合同 |
| 把 22.2× / 4.44× / 12.2× / 14× / 35× 写进 66s | 分母是 50-step BF16 或多卡 SGLang / 官方 API，不是 8-step INT8+cudnn |
| `torch.compile` / Python serving 当质量栈 | 本仓静态 C 管道；RH/FastH3 自己也承认 compile 改数值 |
| Super Accel 整栈当 5090 默认 | 合同钉 **2×GB200 + `cute_sm100`**；消费卡 32GB 装不下 H3+LTX 双居民 |
| VDN FP8 66GB DiT 假装能常驻 5090 | 单卡 B200 峰值 ~65–85GB；5090 32GB 只能 offload，延迟故事另一把尺子 |
| 静默叠刀（少步×稀疏×cache×量化） | 误差预算单点；金线预算已花在 INT8 权 |

### 一句话总判

**保效果的多卡 5090：抄 Ulysses/SP 切 packed 序列（同 8-step INT8+cudnn），KPI 另册。**  
**要墙钟数量级下降：必须换配方（4/8-step LoRA 或 draft+refine），那是快档 SKU，不是金线。**  
**五方案里没有一个是「无损加速本仓 66s」的现成补丁。**

---

## 1. 五方案机制对照表

口径警告：下表秒数是**各方自己的分母**。禁止横比成本仓 8-step 倍数。

| 轴 | FastVideo FastH3 | Sol Super Acceleration | VDN-H3 | Fal H3 Max | RunningHub Lightning |
| --- | --- | --- | --- | --- | --- |
| **主机制** | DMD2 **4 DiT calls** + **可训练 VSA**（video tile 90% 稀疏，text/audio dense） | **两阶段**：H3+LoRA 4-step 896×512 草稿 → upsample → **LTX 3-step Sol-Attn** 精修 | **Hybrid attn**：chunk 窗 softmax（精确局部）+ **Video Delta 线性**（远端）+ 边界锚；另加 DMD **8-NFE** turbo LoRA | **闭源后训练 + 自研扩散 serving 共设计**（推断：少步+系统优化；步数/核未公开） | **4-step 蒸馏** + SageAttn2 + Cache-DiT + `torch.compile` + **TP2+Ulysses4** |
| **少步** | 是（5 个 sigma 点 = 4 次 forward） | 是（4+3，且换模型） | 50-NFE 近无损档 + **8-NFE** 快档（故意不做 4） | 未开源；5s 768p `timings.inference≈2.5s` **推断**极短轨迹 | 是（宣传 4；高动态建议 8） |
| **稀疏/量化 attn** | VSA 训练稀疏；sm100a CUDA tile-64 或 Triton；**不是** training-free Sol | Stage2：**Sol-Attn `cute_sm100`**，layer0 dense，tau 1.0/1.25/1.5 | 窗 softmax（Flex/分解核）+ 线性；FP8 Linear | 未开源 | SageAttention2（内部量化 attn）；主干 **BF16 权** |
| **步/块 cache** | 默认配方不靠 Tea/Mag 吃饭 | Stage1 **cache 关**；Stage2 禁用 algorithm cache | 文档有 kernel+caching；8-NFE 主收益来自少步+hybrid+Ulysses | 未开源 | **Cache-DiT 默认开**（RH 称 quality-validated） |
| **多卡切分** | SP / 复制 DiT；`--num-gpus` 必须整除 **56 heads**；8×B200 另开 SP8 VAE | **不是 TP/SP**：每请求 **2 GPU**（S1\|\|S2）；4 GPU=两对独立流水 | Ulysses 8 路 + **5+3 分支并行**（softmax 5 卡 / VDA 3 卡，B200） | 训练/服务在 **GB200 NVL72**；对外 serverless | **TP2+Ulysses4**（PCIe、无 NVLink）；8× RTX 6000D |
| **VAE / TAE** | 官方 H3 VAE；编译 + 序列并行 decode | **TAEH3** 草稿解码；Stage2 **原版 LTX VAE encode** + **TAEHV** 成片 | 官方 H3 VAE；计时**排除** VAE（只报 denoise） | 未开源 | 官方 VAE，上游精度 |
| **两阶段 draft+refine** | 否 | **是（核心）** | 否 | 未声明 | 否 |
| **调度** | FastVideo launcher：regional DiT compile、pinned offload | compile Stage1 DiT/VAE + Stage2 48 block Inductor；Sol 内核 eager 边界 | Triton/FA4 Flex；推理核与训练核分离 | 自研引擎 | SGLang `multimodal_gen` serve |
| **质量合同** | **非无损**；T2VA only；运动/细节/部分音频可低于 Base；`profile=all` **承认 compile 改 FP 序** | 页面 **明确非无损**；仓内 **`perceptual_quality_gate_run: false`**、`speedup_claim: null` | 宣称 **near-lossless vs dense H3**（相对 50-step BF16）；8-NFE 是另档；门不是本仓 MAE | 人评 Elo / Design Arena；自称质量优先再压速度；**无 MAE/SSIM 公开门** | **不宣称无损**；Sage+蒸馏+Cache 改变计算；RH 自训权未公开 |
| **有声/口型** | 原生 T2VA 立体声 | **最终音频 = Stage1 H3 PCM**（丢掉 Stage2 音频） | 原生 AV；FL2VA/I2VA/L2VA 同 ckpt | 原生同步音；口型靠 prompt | 客户端强制探测 video+audio 双流 |
| **硬件假设** | 公开数字 **B200**；VSA 快核 **sm100a/sm103**；5090 **仓内零命中** | 博客 1×GB200 串行；仓内 v2 **2×GB200**；Stage2 钉 sm100 | **H200 / B200** 8 卡；单卡需 ~65–85GB | **GB200 NVL72** | **8× RTX 6000D sm_120、85GB、PCIe** |
| **落到多卡 5090？** | 权硬件无关；**快核不是 sm_120**。32GB 要 offload/INT8，数字对不上 B200 | **不能**按 6.85s 复现。5090 只有另一条 50-step Sol-Engine 4.52×（1045→231s @768p），仍非金线 | 机制可参考；**显存不够常驻 BF16/FP8 全模型**；Flex/FA4 在 sm_120 需另测 | **不能本地复现** | **拓扑最近**：同 Blackwell 消费互连哲学。但 6000D 85GB ≠ 5090 32GB；RH 权未公开 |
| **对本仓可抄** | 隔离跑 4-step 对照；VSA「text/audio dense」思想 | 快档 SKU 的 draft+refine 分账；TAE 仅预览 | 8-NFE 不碰金线；window+anchor 作研究 attn；Ulysses 切 packed | 质量门纪律 | **PCIe TP+Ulysses 配比**；Cache-DiT 当 P2 实验（默认关） |
| **对本仓不可抄** | 4-step+VSA 进默认；sm100a 核当 5090 默认 | 22.2× 写 66s；TAE 成片；LTX 进金线 | 线性支路替换 cuDNN 默认；单卡 51s/8-NFE 当无损 | 任何闭源秒数 | 4-step+Sage+Cache+compile 叠进默认；12.2× 写金线 |
| **开源完整度** | **高**：推理+LoRA+VSA 核；FastH3 训练「即将」；Apache-2.0 代码 | **中**：Sol-Engine **有** `super_acceleration/` 编排；权重/容器/质量门不随仓；博客倍数 ≠ 仓内 claim | **高**：训练 A1/A2/B/DMD + 推理 + Ulysses；权在 HF（Community License） | **无公开仓** | **中高**：配方+pinned SGLang 快照完整；**RH 加速权未公开**（公开替代 larryvrh / LightX2V） |
| **依赖** | PyTorch、uv、`fastvideo-kernel`、可选 FA4 | SGLang 运行时 + CuTe Sol + LTX-2.5 + 双模型 | PyTorch 2.13、**flash-attn-4**、FlexAttention、Triton 3.7、patched Diffusers | fal API only | PyTorch 2.13、SGLang、SageAttn 2.2、Cache-DiT 1.3、Triton |

### 1.1 秒数怎么读（禁止焊金线）

| 方案宣称 | 实际分母 | 对本仓 8-step 的合法读法 |
| --- | --- | --- |
| FastH3 1×B200 15s 768p 47.2s（14.38×） | Base H3 dense FA4 **~49 DiT** @ 678.7s | 「4-call+VSA 相对 50-step BF16」；**不是** 66s÷14 |
| Super Accel 6.85s / 22.2× | 博客 vs **SGLang 152.3s** @ 5s 768p GB200；仓内 formal **6.76s 且不声称倍数** | 换模型+换分辨率+换步数+轻量 VAE；**禁止 22×** |
| Sol-Engine H3 4.44× / 4.52× | 4090/5090 **50-step BF16 768p**（2239→504 / 1045→231） | 旧文已拆：Tea+Sol+compile 整栈；**不是** 8-step INT8 |
| VDN 8×B200 8-NFE 11.23s | **仅 denoise**，14.4s 768p；排除 load/warmup/VAE/mux | 相对 dense 50-NFE 单卡 13.95min 的 74.5× 叙事；与 8-step 金线无关 |
| Fal ~2.5s / ~35× | 官方 MiniMax **托管 endpoint** 吞吐 | 闭源共设计；无法在 5090 复盘 |
| RH 12.2×（348.8→28.7s） | 8×6000D、**50-step BF16 → 4-step** + Sage+Cache+compile | 他们自己写明倍数含步数变化 |

---

## 2. 各方案机制深拆

### 2.1 FastVideo FastH3

**机制。** Base H3 约 49 次 DiT；FastH3 用 DMD2（frozen teacher + critic，score 差作信号）蒸成 **4 次 forward**。推荐 ckpt **VSA / Data-Free**（只吃 prompt、无目标视频，step 1300）：学生在 video-video 上保留约 10% tile（tile=64），**text/audio 查询永远 dense**；teacher/critic 走 dense，给学生一个全注意力靶。`to_gate_compress` 在基座 H3 不存在——loader 先零初始化；蒸馏学生才带训练门。VSA LoRA **不能**丢到 dense 路径（launcher 硬拒绝 `--no-vsa`）。

**核。** 快路径是 `fastvideo_kernel.block_sparse_attn_sm100a`（sm_100/sm_103，opt-in `FASTVIDEO_VSA_SM100A=1`）；否则 Triton block-sparse。tile-256 另有 FA4 CuTe。消费 Blackwell **sm_120 不是这条 CUDA 核的资格集合**。

**质量。** Preview 自称 T2VA only（FL2VA/Ref2VA 未蒸）。`profile=all` 明确是 **report-only、非 parity**（fusion+compile 改运算序）。下一步才考虑 8-step 质量档。

**5090。** 仓内 grep 无 RTX 5090。博客写「权硬件无关，正在准备 RTX 配方」。多卡要求 GPU 数整除 56 heads → 2/4/7/8 合法，3/5/6 不合法。

**开源。** 本轮已落到 `third_party/accel_survey/FastVideo/`（codeload tarball，~180MB 树，无权重）。

### 2.2 Sol-H3 Super Acceleration（及必须分开的两条 NVIDIA 线）

三条 NVIDIA 线经常被混写，必须拆开：

| 线 | 页 | 是否少步 | 质量口头 | 硬件 | 本仓关系 |
| --- | --- | --- | --- | --- | --- |
| Sol-Engine H3 | [H3](https://nvlabs.github.io/Sana/Sol-Engine/H3/) / OnDevice | **否**（50-step） | near-lossless vs 同拓扑 baseline | 含 **RTX 5090 4.52×** | 旧文 4.44×；Tea+Sol+fusion；**仍非 8-step 金线** |
| **Super Acceleration** | [H3-Super-Acceleration](https://nvlabs.github.io/Sana/Sol-Engine/H3-Super-Acceleration/) | **是**（4+3，换 LTX） | **明确非无损** | GB200 | 本文主对象 |
| Sol-H3 | [Sol-H3](https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3/) | **是**（默认 FastH3 4-step） | 4-step vs 50-step 分母不同 | 8×B300 5s **1.653s** | 少步+Ulysses+INT8 QKV 传输；另册 |

**Super Accel 机制（仓内 v2，比博客更硬）。**

1. Stage1（1 GPU）：FL2VA 首帧条件，**896×512 / 124f / 4 Transformer forwards**，LightX2V `4step_v0.1_544p` LoRA **scale=0.0625**，**cache 禁用**，DiT/VAE compile，**TAEH3** 出草稿。  
2. Handoff：认证 loopback TCP，BF16 像素 `[1,3,121,384,672]` + FP32 PCM；**禁止中间 MP4**（避免 H.264 再损）。  
3. Stage2（另 1 GPU）：**原版 LTX-2.5 Video VAE encode**（refiner 训在该 latent 上，故 encoder 不能换 TAE）+ x2 latent upsample → 1344×768；3 updates（sigma `0.909375, 0.725, 0.421875, 0.0`）；layer0 dense，1–47 Sol-Attn **`cute_sm100`**；**TAEHV** decode。  
4. **音频合同：扔掉 Stage2 联合去噪音频，mux 回 Stage1 的 H3 PCM。** 口型/声画对齐的「保效果」靠的是草稿轨音频，不是 refine。  
5. 拓扑：每请求 2 GPU；4 GPU 节点 = 两对流水，**零模型并行**。

**开源完整度（纠正「可能没代码」）。**  
`NVlabs/Sana` `sol-engine` **有**完整编排目录。拿得到：脚本、合同、kernel 计数门（3 dense / 141 Sol / 141 sm100）。**拿不到：** 权重与容器、感知质量门、与博客 22.2× 同源的 matched baseline（JSON 写死 `speedup_claim: null`）。整树未 clone；摘录在 `third_party/accel_survey/sol-engine-excerpts/`。

**5090。** Super Accel 不能落。5090 上能落的是 **另一条** 50-step Sol-Engine（layerwise offload，231s @768p 5s）——比本仓 480p 8-step 66s **更慢且不同合同**，只说明「消费卡也能跑 768p 50-step」，不能用来打金线。

### 2.3 VDN（OpenVDN/vdn-minimax-h3）

**机制。** 把 video-video 注意力拆成：

- **Softmax 窗：** 按 H3 VAE 的 5-latent-frame chunk，每帧看 **完整的前/当前/后 chunk**（`chunk=K` 对齐，不是中心半径那种会切碎 chunk 的窗）。text/audio 作 global，双向 dense。  
- **边界锚：** 首末帧列/行 dense，约 +3.57% 密度，换长程主体稳定。锚为 `both` 时线性支路 `skip_ends`，两支划分精确、不双计。  
- **Video Delta 线性：** 帧内对 `(I+A)^{-1}` 联合求解（`VdnDelta`，Cholesky），非 SANA-WM 的加性 `I-A/U`。双向 scan + 文本半写入两个状态以免双计。  
- **门控融合：** softmax 门（防局部质量过满）+ 线性 RMSNorm 门 + 独立 `to_out_linear`；只加在 **video 行**。

**训练。** 冻骨干：A1 逐层对齐线性支（200）→ A2 端到端支（500）→ B QKVO LoRA+线性（2000）→ DMD2 8-step turbo（250，无 GAN），turbo 初始化自 **larryvrh MiniMax-H3-Turbo-Lora**。推理可 merge，不改骨干文件。

**多卡。** 标准 Ulysses（序列切 residual，attn 内 all-to-all 变 head-shard）。B200 上再 **5+3 分支并行**（softmax 5 卡 / VDA 3 卡）相对均匀 Ulysses 再降 ~13% NFE。H200  twin 是 6+2（算力平衡不同）。**无 KV cache、无近似通信。**

**质量。** 口头 near-lossless vs **dense MiniMax H3**；计时排除 VAE。8-NFE 相对 50-NFE 是明确的质量-速度折中。门不是本仓 PPM MAE。

**5090。** 依赖 FlexAttention 的 **FA4 CuTe**（`flash-attn-4`）与 B200/H200 核；sm_120 需单独资格测试。显存：单卡 FP8 transformer ~45GB + 工作集约 65GB —— **32GB 5090 不能常驻**。多卡 5090 若每卡复制 33B BF16/FP8 同样不够；只能切序列 + 权量化/offload，那已不是他们的 11.23s 故事。

### 2.4 Fal MiniMax H3 Max（未开源 / 推断）

**能核实。** 官方页/API/博客：fal **后训练** 开源 H3，强化遵循与美学，RL + 人评 Elo；与 **四年自研扩散引擎共设计**；训练与服务在 **GB200 NVL72**；5s 768p **&lt;3s 墙钟**，`timings.inference≈2.5s`；相对官方 MiniMax 托管约 **35× 吞吐**；480p/768p、5–15s、T2V/I2V/Ref2V；**无公开权重、无 Git 仓、无步数、无核、无并行拓扑。**

**推断（必须打标）。** 要在 GB200 上把 5s 768p denoise 打到 ~2.5s，信息论上几乎必然是 **极少 NFE（4 左右或更少）+ 重度系统优化（常驻、融合、可能稀疏/量化 attn、可能多卡）**。fal 原文反对「只减步/降精度就报倍数」，但**没有**证明其 2.5s 路径与开源 50-step 或本仓 8-step INT8 同分布。Design Arena「50× speed」同样是对 **托管 H3** 的偏好-速度前沿，不是对本仓金线。

**5090。** 不能本地落地。可抄的只有：**质量评测一票否决加速刀**。

### 2.5 RunningHub H3 Lightning

**机制。** 三层显式相乘（他们自己承认 12.2× 含步数变化）：

1. 少步：RH 自训加速权 **4-step**（未公开）；公开复现用 larryvrh turbo v4 EMA 或 LightX2V ref2v **8-step**。  
2. 执行：SageAttention2 + Cache-DiT + torch.compile。  
3. 多卡：SGLang `tp=2, ulysses=4`，针对 **PCIe 无 NVLink**。

**质量。** README 明确 BF16 权 ≠ 无损（Sage 内部量化 + 蒸馏 + Cache 改计算）。验收清单含主体/细节/运动/遵循/**音画同步**，但无公开 MAE 表。RH 权与公开 LoRA **不是同一实验**。

**为何对多卡 5090 最有工程参考价值。** 6000D 与 5090 同为 **Blackwell sm_120 + 消费级互连**。差异是显存（85GB vs 32GB）和本仓已是 INT8 8-step 而非 BF16 50-step。因此：**抄拓扑，不抄 4-step 默认，不抄 12.2×。**

Pinned runtime：`sglang/RH-PIN.md` → SGLang `f8cbf000f4a5bfd86d3fb7c1e2d6c8fb12339d0e`。

---

## 3. 与 h3.c 现状差距

单卡金线（已钉，勿改）：4090 ~75s / 5090 ~66s（DiT ~31.4s），INT8+cudnn，8-step，864×480 有声。

| 能力 | h3.c 现在 | 五方案里谁有 | 差距性质 |
| --- | --- | --- | --- |
| Dense fused SDPA | **默认 cudnn**，后融合 | VDN/RH 基线也是 cuDNN/FA；他们加速刀是换 attn | 本仓已赢「eager BF16 attn」；再抄 Sage/Sol/VSA **MAE 不过门** |
| INT8 ConvRot 权 | **默认** | RH/VDN/FastH3 宣传路径多为 BF16/FP8 激活 | 金线误差预算已花在权上；勿再叠稀疏当默认 |
| 少步蒸馏 LoRA | **无产品加载器进默认** | 五方案里四个靠它拿数量级 | 另开 SKU；SLA 文已禁接金线 |
| 训练稀疏 attn | 无 | FastH3 VSA | 研究档；sm100a 核不服务 5090 |
| Training-free Sol | **Tier-1 插件 `sol_cute`，默认关，MAE~35** | Super Accel Stage2 / Sol-Engine 50-step | 继续投资质量，不改默认 |
| Sage3 | **Tier-1，5090 DiT −17%，MAE~27** | RH 用 Sage2 | 商用可切，质量路径否 |
| 步/块 cache | Mag FAIL；Tea 未独立 | RH Cache-DiT；Sol-Engine Tea | 对齐 next-tech-plan P1/P2，8-step 预期 skip 少 |
| 轻量 VAE | 无（金线 fp16 tile+deblock） | Super Accel TAEH3/TAEHV | 预览可以，成片不可以 |
| 多卡 SP/TP | **产品未成**（文档另册） | RH / VDN / FastH3 / Sol-H3 | **本轮最大合法加速面** |
| HTTP serving | 无 | RH/Fal/SGLang | 需要时调框架网关，本仓守引擎 |
| 双模型 refine | 无 | Super Accel H3→LTX | 快档 SKU；两套权+两套 VAE，32GB 不友好 |

Amdahl（5090 cafe，已测分解量级）：e2e ~66s ≈ TE 12–14 + **DiT 31.4** + VAE ~10.5 + 其它。  
- **同轨迹多卡切 DiT**：理论上能打 DiT 墙钟，TE/VAE 仍在（除非也切 VAE tile）。  
- **换 4-step**：DiT 次数 8→4，但 LoRA 改轨迹 → 质量另测。  
- **attn 替换**：单卡已测 −14–17% DiT，MAE 不过 → 金线 e2e 有效天花板 ≈0。

---

## 4. 推荐技术路线（分阶段；MAE∩有声；多卡独立 KPI）

原则：金线进程零污染；每阶段一张对照表；失败即关。

### KPI 分列

| 册 | 配方 | 门 | 成功长什么样 |
| --- | --- | --- | --- |
| **A 金线** | 现默认，单卡 4090/5090 | 相对同机 cudnn 不可辨 + 有声；墙钟 ~75/~66 | 任何 PR 后仍可复现；默认 log `sdpa=cudnn` |
| **B 多卡延迟** | **同一 8-step INT8+cudnn**，2/4/8×5090 | 相对 **单卡 5090 金线成片** MAE∩有声；另报 e2e/DiT | 例如 2 卡 DiT 明显下降且过门；**不改 66s 承诺** |
| **C 快档 SKU** | 公开 8-step turbo LoRA 或 4-step，显式 CLI | 自建门（可宽于金线，但必须写清「非金线」） | 产品能切档；默认仍 A |

### 阶段 M0 — 尺子与隔离仓（本文交付，0 改默认）

- [x] 五方案源码/摘录落到 `third_party/accel_survey/`（权重未下）
- [x] 对照文档（本文）
- [ ] 在 5090 单卡把「公开 8-step turbo LoRA / 4-step」标成 **禁接 generate 默认**（help 一句即可，**不要**实现加载器除非单独任务）

### 阶段 M1 — 多卡 5090：同轨迹 Ulysses/SP（主推荐）

**抄 RH 拓扑思想，不抄 RH 4-step。** 先 2×5090（32GB×2 仍可能常驻 INT8 864×480；768p 另测）。

- [ ] 设计：只切 DiT packed SDPA 的序列维（Ulysses：attn 前 all-to-all，head-shard 计算，再还原）。MLP/INT8 GEMM 是否切权（TP）**第二刀**——RH 在 PCIe 上显示 **少 TP、多 Ulysses** 更优。
- [ ] 约束：`num_gpus` 整除 56 heads（FastH3 同款硬规则）→ 优先 2/4/8。
- [ ] 质量门：cafe 同 seed，相对单卡 5090 cudnn PPM MAE + 音频波形/口型；不过门则 SP 实现留在实验分支。
- [ ] 计时：warm median；排除 load；**另册**写「2×5090 e2e / DiT」，禁止除 66s。
- [ ] 真 `--seconds`；现网 `yongbin@192.168.31.238` 若只有单卡则等第二台 5090，不拿 4090 多卡冒充。

**预期（机制，非承诺）：** 通信税在 PCIe 上会吃掉理想 2×；RH 8 卡才把 50-step 打到 28s 级，且含少步。本仓 8-step 2 卡若 DiT 从 31s → ~20s 量级已算赢；e2e 仍被 TE+VAE 托底。

### 阶段 M2 — 金线内剩余刀（对齐 next-tech-plan，不因本文加码）

- [ ] 独立 TeaCache + AV veto（P1）：8-step 从严扫；预期可能 FAIL，FAIL 也要填表。
- [ ] 保守 FBCache/块级残差（P2）：默认关。
- [ ] Sage3 / sol_cute 继续当 Tier-1 开关，**auto 永不选**。

### 阶段 M3 — 快档 SKU（可选，永不默认）

优先级（保效果递减、墙钟递增）：

1. **公开 8-step turbo LoRA**（larryvrh / VDN turbo）+ 本仓或 SGLang 侧车 —— 与金线 8-step 同 NFE，改的是权，不是步数幻觉。  
2. VDN hybrid **仅当** 有 80GB 级卡或可接受 offload；5090 多卡不够就不要承诺 11s。  
3. FastH3 4-step VSA：隔离 FastVideo 跑对照；本仓最多当研究 attn，sm100a 核不迁 5090 默认。  
4. Super Accel：需要第二模型 LTX + 双 GPU 流水 + 接受非无损与 TAE 成片；**产品上必须标快档**。

- [ ] 任一快档：独立 CLI 名（如 `--preset fast8`），help 写「非金线」；矩阵新行；MAE 阈值不得冒充金线。

### 阶段 M4 — 明确不做

- [ ] 不把 Fal 35× / Super Accel 22.2× 写进发布说明。  
- [ ] 不把 TAE 换成金线 VAE。  
- [ ] 不在默认路径 merge 任何社区 LoRA。  
- [ ] 不把 MagCache 回潮默认。

---

## 5. 可执行对照实验（有卡再跑；本文不跑 generate）

现网：4090 `haiyang@192.168.31.34`；5090 `yongbin@192.168.31.238`。密码仅 env。

| ID | 目的 | 命令意图 | 通过 |
| --- | --- | --- | --- |
| G0 | 金线未漂 | 本仓 cafe 8-step cudnn 单卡 5090 | ~66s 量级；`sdpa=cudnn` |
| B1 | 多卡同轨迹 | 2×5090 SP + 同配方 | MAE∩有声 vs G0；报 DiT 墙钟 |
| C8 | 8-step 公开 LoRA（侧车） | SGLang 或 FastVideo **非本仓默认** | 另表 MAE；禁止写入金线 |
| C4 | 4-step 公开 LoRA | 同上 | 必 FAIL 金线门则停 |
| F1 | FastH3 VSA 隔离 | `FastVideo/examples/inference/basic/run_fasth3_lora_preview_vsa_datafree.sh`；5090 加 `--vsa-kernel triton --no-fa4` | 能出片即对照；不要求过本仓 MAE |
| V1 | VDN 8-NFE | 需 ≫32GB；5090 单卡预计 OOM → 记「硬件不够」而非质量 FAIL | |

---

## 6. 各仓本地路径

`github.com:443` 的 `git clone` 本轮失败（空响应/超时）；改用 **`codeload.github.com` tarball**。权重均未下载。

| 方案 | 路径 | 体积（约） | 备注 |
| --- | --- | --- | --- |
| 索引 | `third_party/accel_survey/README.md` | — | |
| FastVideo | `third_party/accel_survey/FastVideo/` | 180MB 树 | FastH3：`examples/inference/basic/run_fasth3_lora_preview_vsa_datafree.sh`；VSA：`fastvideo/attention/backends/video_sparse_attn_h3.py` |
| VDN | `third_party/accel_survey/vdn-minimax-h3/` | 70MB | 混合注意力 `src/models/hybrid_attention.py`；Ulysses `src/inference/utils/ulysses.py`；B200 8-NFE `configs/inference/8nfe_tuned_fp8_ulysses_b200.yaml` |
| RunningHub | `third_party/accel_survey/MiniMax-H3-MultiGPU-Lightning/` | 125MB（含 SGLang 快照） | `scripts/start.sh` TP2+U4；`sglang/RH-PIN.md` |
| Sol Super Accel 摘录 | `third_party/accel_survey/sol-engine-excerpts/` | 小 | 完整代码在上游 `models/minimax_h3/super_acceleration/`；本仓未整树 clone Sol-Engine |
| Fal | 无仓 | — | https://fal.ai/minimax-h3-max · https://blog.fal.ai/introducing-h3-max-by-fal/ |

`.gitignore` 已忽略三棵大树（VDN / RH / FastVideo），避免误 commit。

上游 URL：

- FastVideo https://github.com/hao-ai-lab/FastVideo  
- Super Accel 页 https://nvlabs.github.io/Sana/Sol-Engine/H3-Super-Acceleration/ · 代码 https://github.com/NVlabs/Sana/tree/sol-engine/models/minimax_h3/super_acceleration  
- VDN https://github.com/OpenVDN/vdn-minimax-h3 · 博客 https://openvdn.github.io/  
- Fal https://fal.ai/models/minimax/h3-max/text-to-video  
- RH https://github.com/RH-RunningHub/MiniMax-H3-MultiGPU-Lightning/

---

## 7. 给执行者的硬停止句

1. 不要改 `tools/h3_cuda_generate.c` 默认 backend / steps / mag / sage / sol。  
2. 不要把任何外部 4-step LoRA merge 进 INT8 金线权。  
3. 不要在发布材料使用 22.2× / 4.44× / 12.2× / 14× / 35× 作为 8-step 加速比。  
4. 多卡数字写进独立 REPORT（如 `out/verify_5090_sp2/`），不要改 next-tech-plan 的 66s 行。  
5. 需要 serving 时优先 **调用** SGLang/RH 配方做网关，本仓继续守 C 引擎。
