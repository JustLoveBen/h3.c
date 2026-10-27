# 加速且提质：行业机理 × MiniMax-H3/h3.c 杠杆 × 可证伪下一刀

> **日期：** 2026-10-28（Asia/Shanghai）。**性质：** 研究笔记（分析+文档；本轮不抢 GPU）。  
> **用户杠：** 每个细节要深挖；要对行业方案有真实认识；对本仓加速方案有深刻理解；要从**理论**上知道如何在加速的同时提升质量（不只是速度–质量权衡）。  
> **产品两轨：** **Track A** 近无损加速（waffle / BIT_ID，默认可开）；**Track B** 有损但可接受（显式质量下限，opt-in）。  
> **硬约束回响：** INT8 ConvRot 产品默认；注意力 = cuDNN SDPA；主臂 DMAD-4 / Euler-8；PDMD 对照；FaceRefine **STOPPED**；NVFP4 **terminal BLOCK**；勿与 C064/C015 抢卡。  
> **仓内姊妹文（必读）：** [`acceleration-mechanisms`](2026-10-05-acceleration-mechanisms.md)（机理八族）、[`quality-preserving-acceleration`](2026-10-05-quality-preserving-acceleration.md)、[`sdpa-backends-compared`](2026-10-28-sdpa-backends-compared.md)、[`4090-opt-menu-beyond-dmad`](2026-10-28-4090-opt-menu-beyond-dmad.md)、[`face-lora-vs-facerefine`](2026-10-28-face-lora-vs-facerefine.md)、[`beyond-dmad-and-lossless`](2026-10-27-beyond-dmad-and-lossless.md)、[`nvfp4-keep-int8-gate-bug-rca`](2026-10-27-nvfp4-keep-int8-gate-bug-rca.md)。

---

## 0. 一句话

**加速 ≠ 权衡质量。** 真正可同时提速提质的路径只有几类：(1) 消掉**本不该存在的数值/工程错误**；(2) 用更匹配教师的离散化/蒸馏合同，让少步**截断误差更小**；(3) 用精确算子腾出预算去做**更多空间 token（抬画布）或末步 refine**；(4) 精确不变量缓存与融合——质量零杠杆但墙钟真降。其余（稀疏注意力、跨步特征缓存、W4A4、FaceRefine stitch）在少步蒸馏栈上多数是 **Track B**，且常在 NFE=4 时失效或伤运动。本仓 4090 金线墙钟主轴仍是 **NFE（DMAD≈Euler 暖 DiT 1.61×）**；内核层近峰后 Track A 只剩薄肉。

---

## A. 行业地图（机理 + 质量风险 + 何时能「提质」）

组织轴沿用仓内 [`acceleration-mechanisms`](2026-10-05-acceleration-mechanisms.md) 的八族。**禁止**把 50-step BF16 官方倍数直接换算到本仓 4-step INT8。下列数字一律带来源链接；无仓内复现则标「行业宣称 / 未在本栈复测」。

### A.1 注意力：精确核 vs 近似

| 方法 | 机制（2–4 句） | 质量风险 | 何时能 IMPROVE 质量（不只加速） | 链接 |
| --- | --- | --- | --- | --- |
| **FlashAttention / FA2 / FA3** | 精确 softmax 注意力的 IO-aware tiling：中间结果留在 SRAM，少写 HBM；数学上等价于标准注意力（浮点结合序差异除外）。FA3 面向 Hopper 异步等硬件特性。 | 极低（数值 ≈ dense）；错误实现或错误 dtype 才伤质。 | 使**更长序列可训/可推**成为可能（原论文 Path-X / Path-256），从而用更多上下文换质量——这是能力轴，不是「同一短序列糊一点换快」。 | [FA](https://arxiv.org/abs/2205.14135) · [FA2](https://arxiv.org/abs/2307.08691) · [FA3](https://arxiv.org/abs/2407.08608) |
| **cuDNN SDPA** | 厂商融合 dense SDPA；本仓 `auto`→cudnn，Tier-0 质量默认；`sdpa_strict` 要求 hits==want。 | 个别 F32 计划失败可落到 naive F32 flash（慢路径），主 DiT 仍应是 cudnn。 | 稳定精确后端是其他近似臂的**对照金线**；本身不「提质」，但避免错误后端冒充加速。 | NVIDIA cuDNN；仓内 [`sdpa-backends-compared`](2026-10-28-sdpa-backends-compared.md) |
| **SageAttention / Sage2 / Sage3** | 利用注意力矩阵可中心化等结构做低比特 QK/PV（INT8→INT4/FP8→**FP4 microscaling**）。Sage3 明确依赖 Blackwell FP4 Tensor Core。官方自认**不保证所有模型无损**，敏感层/时间步建议混 Sage2++。 | 量化误差；首尾步与文本条件行更敏感；Ada 无原生 FP4 时挂 Sage3 名是错位。 | 几乎不「提质」；仅当低精度注意力意外抑制病态数值、或腾出算力去做更多步/更高分时间接提质。行业宣称 5090 上相对最快 FA 可达约 5× **算子级**吞吐（[Sage3](https://arxiv.org/abs/2505.11594)）——**非**本仓端到端合同。 | [Sage3](https://arxiv.org/abs/2505.11594) · [SageAttention 仓库](https://github.com/thu-ml/SageAttention) |
| **稀疏注意力（STA / Radial / SVG / Sol-Attn / MInference 类）** | Softmax 质量高度集中 → 只算高能量 QK 块（静态窗、径向、头分工、动态路由）。Sol-Attn 另加块稀疏 + 低阶 Taylor 补偿。MInference 类偏 LLM 预填稀疏，视频 DiT 需重校准索引。 | 丢块 = 有界近似误差；快运动、远距离依赖、少步轨迹更脆。全稀疏在少步蒸馏上可**改 ODE 轨迹**。 | 长序列下若 dense OOM，稀疏让「本来跑不了的高分/长片」跑通 → **能力提质**。带 **dense 护栏**（前 N 步/层）时，行业与本仓 5090 曾见 waffle 不劣于 dense 的个例——仍属 Track B 直到同日金线复测。 | Sol 官方叙事见仓内 [`sol-h3-official-reanalysis`](2026-10-05-sol-h3-official-reanalysis.md)；稀疏前沿 [`video-dit-attention-frontier`](2026-10-07-video-dit-attention-frontier.md) |
| **量化注意力（与上重叠）** | 注意力内部 GEMM 降位宽拿 Tensor Core；与权重 W4A4 **不同族**。 | 与 Sage 同；另有累加器/缩放因子实现坑。 | 同 Sage：间接为主。 | 同上 |

### A.2 少步 / 蒸馏

| 方法 | 机制 | 质量风险 | 何时能 IMPROVE 质量 | 链接 |
| --- | --- | --- | --- | --- |
| **Progressive Distillation** | 学生学教师「两步并一步」的净映射，迭代把 NFE 减半。 | 误差沿蒸馏链累积；视频运动易塌。 | 设计良好时可逼近教师分布；极少系统性「优于多步教师」。 | [Salimans & Ho 2022](https://arxiv.org/abs/2202.00512) |
| **Consistency / LCM** | Consistency：任意时刻映到轨迹终点；LCM 把一致性蒸馏接到引导扩散，少步采样。 | 过饱和、细节糊、CFG 自由度丢失。 | 末步 consistency/refiner 思想可**修补**过饱和少步输出（本仓 C035 叙事）。 | [Consistency](https://arxiv.org/abs/2303.01469) · [LCM](https://arxiv.org/abs/2310.04378) |
| **DMD / DMD2** | 分布匹配蒸馏：学生输出分布对齐教师，不强制逐步轨迹一一对应。DMD2 去掉昂贵回归对、加 GAN、支持多步，并报**一步 FID 可超过教师**（图像域）。 | 视频上中间步监督不足 → 运动/动态退化（Phased DMD / SGTS 类文献警示）；学生天花板被教师与数据锁。 | **理论上升质路径：** 分布匹配 + 真数据 GAN 可修正教师 score 偏差，使少步学生在感知指标上**超过**多步教师——这是「加速且提质」的正例，但是**训练合同**，不是推理开关。 | [DMD](https://arxiv.org/abs/2311.18828) · [DMD2](https://arxiv.org/abs/2405.14867) |
| **CFG 蒸馏 / Decoupled DMD 类** | 把 CFG 增广项内化进学生，推理免双前向。 | 过引导、风格漂。 | 去掉双倍 CFG 算力的同时稳住引导 → 同墙钟下可把预算还给步数或分辨率。 | 见仓内机理文对 Decoupled DMD 的综述段落 |
| **MeanFlow / 流匹配少步族** | 在流/速度场上直接回归平均流或大步映射。 | 与 DMD 同类：少步截断 + 容量。 | 更好的流参数化可降低少步截断误差（理论），本仓未独立复现。 | 行业综述向；落地前须绑定本栈合同 |
| **高阶求解器（DPM-Solver / UniPC）** | 利用半线性结构解析线性项，高阶积分剩余非线性。训练-free。 | NFE≪8 时失效；难打过已蒸馏 4NFE 学生。 | 在**未蒸馏**多步教师上，更好离散化可在同 NFE 提高保真——「同算力更高质量」。叠已蒸馏学生时收益通常薄。 | [DPM-Solver](https://arxiv.org/abs/2206.00927) · UniPC 等 |

### A.3 缓存（Tea / Mag / DeepCache 类）

| 方法 | 机制 | 质量风险 | 何时能 IMPROVE 质量 | 链接 |
| --- | --- | --- | --- | --- |
| **TeaCache** | 用 timestep embedding 差异代理预测输出变化，跳过/复用中间特征。需校准。 | 阈值过松 → 运动发粘、细节糊；prompt 过拟合。 | 一般不提质。 | [TeaCache](https://github.com/ali-vilab/TeaCache) · CVPR 2025 |
| **MagCache** | 相邻残差幅度比单调衰减律 → 自适应跳步；宣称单样本校准。行业报约 2.1–2.7×（多模型）。 | 同陈旧误差累积。 | 一般不提质。 | [MagCache](https://arxiv.org/abs/2506.09045) |
| **DeepCache 类** | 复用高阶 U-Net/DiT 特征，廉价更新低阶。 | 架构假设强；DiT 块划分不当则花纹/身份漂。 | 一般不提质。 | [DeepCache](https://arxiv.org/abs/2312.00858) |
| **精确不变量缓存** | 文本 emb、RoPE 表、真常量 AdaLN 条件等**数学常量**只算一次。 | **无**质量杠杆（实现正确时）。 | 不直接提质；腾出墙钟可换更高分/更多步 → 产品层提质。 | 本仓 TE cache / RoPE memo |

**少步失效机理（对本仓极关键）：** 4–8 NFE 蒸馏后，相邻步相对变化大，跨步特征相似前提塌缩。仓内与官方 4-step 栈、IonDen 相对 L1、DisCa 横向表形成三角旁证 → **主题禁把 Mag/Tea 当 turbo4/DMAD 默认**（见机理文附录）。

### A.4 量化（W8A8 / FP8 / NVFP4）

| 方法 | 机制 | 质量风险 | 何时能 IMPROVE 质量 | 备注 |
| --- | --- | --- | --- | --- |
| **W8A8 / INT8 GEMM** | 权重+激活 INT8 原生 Tensor Core；须融合 quant/epilogue 才兑现墙钟。 | 激活 outlier；错误 dequant/累加序。 | 极少直接提质；稳定 INT8 链路是质量地板。 | 4090 甜点 |
| **FP8** | Hopper/Blackwell 原生；动态范围优于 INT8。 | 平台绑定。 | 同左。 | |
| **NVFP4 / W4A4** | 每块 micro-scaling 的 4-bit 权重/激活。 | **能量塌缩、马赛克、栅格**；实现 bug 可伪装成「量化理论失败」。 | 几乎不；修 bug 后「从崩坏恢复到可辨」是假提质（相对 bug），相对 INT8 金线仍常 FAIL。 | 本仓 C001 **BLOCK** |

### A.5 VAE / latent 超分 / tiled decode

| 方法 | 机制 | 质量风险 | 何时能 IMPROVE 质量 |
| --- | --- | --- | --- |
| **Tiled / batched VAE（本仓 Kitchen 名）** | 大画布分块 decode；批 cuDNN。**不是** DiT SDPA 后端。 | 接缝、色偏、能量门。 | 让高分装得下 → 能力提质。 |
| **异步 unpatch / cast 融合** | 消 D2H 串行与多余 cast。 | 实现正确则 bit-id。 | 间接。 |
| **Latent 超分 / 二采** | 低分草稿 + 条件超分或二次 DiT。 | 小脸仍可能在二采合同里被伤；非无损金线。 | **直接提质**（更多像素/细节），墙钟通常↑或结构重排——属产品 SKU，不是「无损加速」。 |

### A.6 并行 / 重叠 / 异步 TE

| 方法 | 机制 | 质量风险 | 何时能 IMPROVE 质量 |
| --- | --- | --- | --- |
| **序列并行 / CP（Ulysses、Ring）** | 多卡切序列；通信敏感。 | 通信主导时加卡变慢。 | 装下更长/更高分。 |
| **TE↔DiT↔VAE 生命周期重叠** | 预取/错峰释放。 | 竞态则数值错 → 假质量事故。 | 正确实现零质量杠杆。 |
| **异步 TE / dump-replay** | 文本编码与 DiT 解耦；dump latent 后多次 VAE/打分。 | 合同漂移。 | 测量卫生 + 同 latent 多评分 → **评测提质**（更好选片），非生成提质。 |

---

## B. MiniMax-H3 / 本仓杠杆：宣称 vs 实测（分轨）

### B.1 Track A（近无损 / 默认可安全）

| 杠杆 | 宣称机制 | 实测（证据） | 默认？ |
| --- | --- | --- | --- |
| **INT8 ConvRot 主链路** | Ada 上 INT8 GEMM 近峰；质量合同绑 INT8 | 产品默认；GEMM 峰 98–99%（deadend-operators）；链路 cast/quant 仍有工程肉 | **是** |
| **cuDNN SDPA** | Tier-0 精确 dense | C040/DMAD：`sdpa_select=cudnn`，`sage3_lib=0` `sol_cute_lib=0`；[`sdpa-backends-compared`](2026-10-28-sdpa-backends-compared.md) | **是** |
| **C073 dump→replay + VAE async** | 同 latent 可 waffle；async 应 bit-id | 480p Euler-8：waffle Δ=0；rgb checksum async on≡off（`5d8c9cd` 笔记） | 测量/流水工具；async 可开 |
| **C041 `H3_DIT_INT8_WINDOW`** | 缩权重窗降 peak VRAM | Euler −3.1 GiB / DMAD −2.2 GiB；DiT 略慢或平（`5e994fc`） | **否**（opt-in 余量） |
| **C062 `H3_TE_CACHE_AUTO`** | repeat≥2 时 TE HIT | clip2 te=0 已确认；默认仍 off（`5e6b8e2` / beyond-dmad） | **否** |
| **C064 TE GPU prefetch overlap** | TE∥DiT 重叠 | 4090：dit **+7–9%** vs HOST_STAGE；NO_WALLCLOCK_WIN；opt-in（`cc522ff` / c064 笔记） | **否** |
| **C017 L2 pin** | AccessPolicyWindow | BIT_ID PASS；fair hot Δ≈0（`95ab073`） | **否** |
| **C068/C069 RoPE memo** | 宿主 memo / VAE H2D skip | BIT_ID；墙钟噪声级；memo default-on | memo **是** |
| **C003 EPI_FUSE\*** | 融合 ConvRot+requant | SL：BIT_ID 但 hot dit **+46%** → keep OFF（`ed1ce72`） | **否（禁默认）** |
| **C070 forbid hot sync** | 卫生 | KPI 门禁用 | 工具 |

### B.2 Track B（有损可接受 / opt-in 或产品档）

| 杠杆 | 宣称 | 实测 | 默认？ |
| --- | --- | --- | --- |
| **PDMD / DMAD LoRA 少步合同** | 4NFE 分布/轨迹蒸馏 | 4090：DMAD vs Euler 暖 dit **1.61×** / e2e **1.43×**（dmad4-vs-euler8 笔记）；论文/仓内 PDMD 常优于同 NFE DMAD（beyond-dmad） | **产品少步臂**（有损相对 Euler-8 金线） |
| **sol_cute** | 块稀疏加速 | 4090 历史 8-step：DiT −15% 但 MAE/视觉 FAIL；5090 turbo4 全稀疏 waffle 退化，护栏臂曾速度+waffle 双正；**当前 4090 产品 lib=0 未复测 DMAD** | **否** |
| **Sage3 路径** | FP4 attn | 5090：DiT −17%，旧 MAE 高 → 不过默认；**非 4090 目标** | **否** |
| **NVFP4 W4A4** | 每步更快 | 能量塌缩 → terminal BLOCK；KEEP 门控曾是**实现 bug**（修后 mlp/qkv 不再炸，但仍 mosaic FAIL）（`ebd77a5` / gate-bug RCA） | **BLOCK** |
| **C035 / `H3_REFINER_LORA`** | 末 N 步细节 LoRA | 接线有；权重/A/B 未当金线关闭 | opt-in 实验 |
| **抬画布 SKU** | 更多 face/token | 远脸主解；1344 Euler 脸最好；PDMD≤1152（ghost/facerefine 笔记） | **质量档**（墙钟↑） |
| **FaceRefine** | crop→F2VA→stitch | 贵、缝、加步不救 34px → **STOPPED** | **停** |

### B.3 轴分离（禁止合成一个验收）

| 轴 | 消什么 | 本仓尺 |
| --- | --- | --- |
| Kernel/engine | 算子/中转/launch | 同配置 waffle / BIT_ID |
| Sampler/schedule | NFE、σ、renoise | `--fewstep-contract` + 目检 |
| Architecture | attn/窗/streaming | 独立 SKU |
| Training | 学生/补丁权重 | 合同绑定；**禁止**跨臂 waffle 当金线 |

---

## C. 理论：如何「加速且提质」（非纯权衡）

### C.1 非权衡路径（机制）

1. **更好的 ODE/SDE 离散化 / 与教师调度对齐**  
   扩散生成 = 数值积分。局部截断误差随步长幂次增长。DPM/UniPC 解析半线性部分 → 同 NFE 更低误差 = **同算力更高保真**。蒸馏学生若学的是「错误调度上的教师」，少步会系统偏；**合同内 lcm_renoise / 正确 shift** 是质量门，不是装饰。

2. **蒸馏缩小相对教师的有效截断误差（少步 > 多步噪声的情形）**  
   多步采样也有离散化噪声与 score 估计误差。DMD2 类用分布匹配 + 真数据判别，可在图像域报出**优于教师**的 FID——机制是学生不再被锁死在教师逐步路径上，并能修正教师 score 偏差。视频上运动塌缩风险高 → **理论成立、本栈证据弱**，须独立合同与人眼/运动指标。

3. **更高画布 / 更多 face token（质量↑ 墙钟↑）作产品 SKU**  
   远脸 ~34px ≈ 1 个 DiT patch：信息论上限，不是风格缺口。抬到 960/1152/1344 增加 face patches → **直接提质**。这不是「加速」，但是「用算力换质量」的正确杠杆；与少步加速正交组 SKU。LoRA **不能**发明不存在的空间 token（[`face-lora-vs-facerefine`](2026-10-28-face-lora-vs-facerefine.md)）。

4. **消除数值/控制流 bug（「假量化质量」）**  
   NVFP4 KEEP 门控 RCA：`KEEP=qkv`≡full FP4 是 AdaLN 重门控擦除 out 结果，不是「QKV 必须 FP4」。修 bug 后能量恢复、视觉仍可能 mosaic——说明必须先排除实现bug，再谈量化理论。**修 bug = 同时「加速路径可用」与「质量从崩坏回到可评」**。

5. **精确注意力 vs 近似；近似何时伤运动**  
   Dense cudnn：轨迹与训练一致。全稀疏改每步向量场 → 少步下误差不可被后续步平均掉 → 运动发粘/拖影。护栏（前步/前层 dense）是把近似限制在「教师也较平滑」的区域——**质量保全机制**，不是营销开关。

6. **投机 / 级联：廉价草稿 + 末步 refine（C035）**  
   前 N−k 步用快学生，末 k 步切 detail/refiner LoRA 或更高精度算子。机制：高频细节对末段轨迹更敏感；把贵算力集中在高敏感区间 → **同总 NFE 下可感质量↑**。须防风格渗漏与合同漂移。

7. **数据/课程 for detail LoRA**  
   成对教师（高分 Euler）vs 学生（480 DMAD）、多尺度桶、face-bbox 加权：让适配器学「少步过饱和后的残差」，而非静物棚拍。成功线须对照「抬画布」收益；若抬画布碾压则停训。

### C.2 对本栈理论仍弱 / 勿夸大处

| 命题 | 为何弱 |
| --- | --- |
| 「再叠 MagCache 到 DMAD-4」 | 少步冗余已榨干；与族 1 张力；主题禁默认 |
| 「Sage3 加速 4090」 | 无 FP4 TC；应谈 Sage2++ MMA 而非挂名 |
| 「NVFP4 修完就能默认」 | 门控 bug 已修；mlp/qkv/full 视觉仍 FAIL |
| 「FaceRefine / 脸 LoRA 救远脸」 | 像素预算上限；stitch 已停 |
| 「官方 Sol 1.48× → 本仓 4-step」 | 合同：50-step BF16 ≠ turbo/DMAD INT8 |
| 「CUDA Graph / QKV merge 必有秒级」 | 部分已陈旧或噪声；须 nsys 后再宣称 |
| 「线性注意力免训练提质」 | VDN/SLA 需训或微调；合同外 |

### C.3 合成公式（战略）

\[
\text{墙钟} \approx \underbrace{\mathrm{NFE}}_{\text{Track B 主杠杆}} \times \underbrace{T_{\text{step}}(\text{精确核}+\text{量化}+\text{融合})}_{\text{Track A}} + T_{\text{TE/VAE}}
\]

\[
\text{感知质量} \approx f(\underbrace{\text{合同/蒸馏}}_{\text{轨迹}},\, \underbrace{\text{空间 token}}_{\text{画布}},\, \underbrace{\text{无 bug 数值}}_{\text{实现}},\, \underbrace{\text{末步 refine}}_{\text{可选}})
\]

**同时改善两者的投资顺序：** 先 Track A 消浪费与修 bug → 再选对少步合同（PDMD/DMAD）→ 再用画布/末步 refine 做质量 SKU → 最后才 opt-in 近似注意力/缓存。

---

## D. 严谨清单（未来每刀切片模板）

每条 claim 必须填满下表，否则不得进默认、不得写进用户「已验证」。

| 字段 | 要求 |
| --- | --- |
| **Claim** | 一句话：加速多少 / 质量怎样 / 在何合同 |
| **Mechanism** | 消的是 NFE / token / 位宽 / launch / bug / 哪一族 |
| **Falsifiable metric** | 暖 dit_ms、e2e、peak_dit、face-px、waffle mean_abs_delta、BIT_ID checksum、运动类目检项——**事先写阈值** |
| **Gate** | 同配置 waffle 或 BIT_ID（Track A）；Track B 另写视觉/OR 下限与「不可接受」伪影列表 |
| **Controls** | 同 seed、同 prompt、同 TE HIT、burn-in、禁止跨臂 waffle |
| **Default vs opt-in** | 明确；FAIL → 保持 off，写进关闭表 |
| **Non-goals** | 本刀不做的事（防范围蔓延） |
| **SHA / artifacts** | 代码 SHA、out/ 路径、labeled mp4 |
| **Track** | A 或 B；禁止混 KPI |

**红线：** 无机理归族、无失败阈值、用 50-step 倍数讲 4-step、把 Kitchen 当 DiT attn、把 FaceRefine 当质量主解、把实现 bug 当成量化理论。

---

## E. 分轨下一刀（按说服力排序）

### E.1 Track A（近无损）

| 优先级 | Dig | 会说服我们的证据 |
| ---: | --- | --- |
| 1 | **C015/C042 act-FP16 中转**（他机可能在做；勿抢卡） | peak_dit ≥−0.5 GiB；**同配置 waffle PASS**；无马赛克；暖 dit 回归 <阈值 |
| 2 | **C011 VAE cast/prescale 融合**（需盘+nsys） | nsys 证明 cast 段可融；BIT_ID；暖 e2e 可测降 |
| 3 | **C018 QKV merge 现状 diff** | 先证明「仍有三次瘦 GEMM」缺口；再估 −0.6~−1.2s；INT8 序非 bit 则改 waffle 门 |
| 4 | **C064 演进：HOST_STAGE 与小块 GPU pref 并存** | 当前 ov 互斥导致 dit 回归；若并存后 dit≤基线且无竞态 BIT_ID → 再谈默认 |
| 5 | **测量卫生固化（C073 类）** | 任意新核默认要求 dump-replay waffle 可复现 |

### E.2 Track B（有损可接受）

| 优先级 | Dig | 会说服我们的证据 |
| ---: | --- | --- |
| 1 | **抬画布正式 SKU 表**（960/1152/1344 × DMAD/Euler） | 固定 p10：远景 face-px↑ + 人眼五官可辨；墙钟单独报价，不叫「加速失败」 |
| 2 | **4090 sol_cute+护栏 vs cudnn · DMAD-4 · 480p** | DiT 有稳定增益；waffle/人眼 ≥ 下限；无运动塌；否则关闭叙事 |
| 3 | **C035 refiner LoRA scale sweep** | 同画布下 smear/过饱和↓，无全局糊；且收益接近「半档抬画布」才值得产品化 |
| 4 | **PDMD vs DMAD 同日 4090 视觉+KPI**（PDMD 仍对照政策时只做研究臂） | 同 NFE 下运动/脸分；不自动改默认除非产品改合同 |
| 5 | **远脸 LoRA 最小验证**（仅硬 KPI 锁 480p 时） | 见 face-lora 笔记 §8；抬画布碾压则 **停训** |

### E.3 明确不再挖（本窗口）

- NVFP4 默认 / host-dequant 旁路 / chain  
- EPI_FUSE_SL 默认开  
- FaceRefine stitch 主线  
- MagCache/TeaCache @ DMAD-4 默认  
- 无 WC 的 L2 再诉讼  
- 把 Kitchen 并进「四大注意力」对比  

---

---

## G. 竞品吸收：机制 → Track A/B → 可装船下一刀

> 用户加码（t79u）：主动深挖 **sol-h3 / FastH3 / VDN / DMD·DMAD·PDMD 族及同侪**，合成**对本栈最佳、可落地**的方案——借长处、能装船，不是文献摘要。  
> 一手仓内依据：[`sol-h3-official-reanalysis`](2026-10-05-sol-h3-official-reanalysis.md)、[`taomate-fasth3-vdn-reanalysis`](2026-10-05-taomate-fasth3-vdn-reanalysis.md)、[`v0-v1-sol-ab-results`](2026-10-05-v0-v1-sol-ab-results.md)、[`vdn-direct-use-reanalysis`](2026-10-07-vdn-direct-use-reanalysis.md)、[`methods-mechanism-deepdive`](2026-10-05-methods-mechanism-deepdive.md)、[`beyond-dmad-and-lossless`](2026-10-27-beyond-dmad-and-lossless.md)。

### G.0 一句话裁决

| 竞品 | 对我们真正有用的「长处」 | 不要整包照搬的原因 |
| --- | --- | --- |
| **NVIDIA Sol-Engine / Sol-H3** | **Pillar-1 融合**（QKV-merge、QKNorm+RoPE、graph capture）= Track A；**MMDiT dense 护栏**（前 N 层/步 dense）= 让 sol 从「全稀疏退化」变成 Track B 可接受 | 官方 4.52× 主成分是 **TeaCache @ 50-step**；4-step Sol-H3 官方自己也改用蒸馏不用 cache——与我们判断同构 |
| **Sol-Attn (sol_cute)** | 长序列块稀疏 + Taylor 补偿；**护栏配方**已在 5090 turbo4 上把 waffle 从 1.321 拉回 ~0.81（优于 dense 0.849） | 全稀疏 @ 少步蒸馏 **实测退化**（FastWan 论点成立）；480p 收益薄；4090 产品当前 lib=0 |
| **FastH3** | 「**稀疏训进权**」配方（VSA）+ 官方自认 **8-step 质量档** | 4-step 权本身：arena 中下、运动塌缩被 VDN/社区复现；Comfy 常需 6 步才可用——**不装权，装思想** |
| **VDN** | 混合注意力（近窗 softmax + 远线性状态）**保运动 > FastH3**；线性分支缓解长序列 OOM；权重是「分支+LoRA 挂冻结 H3」可插拔形态 | 现成模型 @ 5090 社区栈 5s 768p **~2× 慢于本仓 T4**（8 NFE）；官方 FA4 栈不吃 sm_120；C 路径移植 4–8 周 |
| **TaoMate-H3** | **refiner 收尾**（0.7 强度 / 末几步）治 turbo/少步过饱和——社区高票；流式+clean-KV 本仓已有接线 | 整段 3-step 生成 WhichH3 **垫底**；全量权含 `token_refiner` 会被本仓 parser 拒——用 Kijai rank-19 等适配版 |
| **DMD / DMD2 / DMAD / PDMD** | 少步分布匹配是 **墙钟主杠杆**；DMD2「可超教师」是训练合同级提质；本仓 **DMAD-4 暖 dit 1.61× vs Euler-8** 已落地；PDMD 论文/仓内常优于同 NFE DMAD | 少步 ≠ 无损；加步救 turbo 学生常失败（mode-seeking + 校准网格外推）；与 MagCache **互斥甜区** |
| **LoSA / SLA** | **fidelity-first 阈值**（冻支撑不冻值）与 budget-first 稀疏正交；SLA 稀疏+线性补偿治快运动 | 非即插即用；需索引重校准或微调——立项级 Track B |
| **Tea/Mag/EasyCache** | 精确不变量缓存永远该做（已部分落地） | **近似跨步缓存 @ DMAD/turbo4：主题禁默认**（冗余已榨干） |

---

### G.1 机制 → 吸收进哪一轨

| # | 吸收项（机制级） | 来源 | Track | 本仓现状 | 装船形态 |
| ---: | --- | --- | --- | --- | --- |
| 1 | QKV 三瘦→一宽 GEMM | Sol Pillar-1 | **A** | **已落地**（C018 stale closed；[`exp-g3`](2026-10-28-exp-g3-pillar1-gap-proof.md)） | 停止 −0.6~−1.2s；余量→C015 |
| 2 | QKNorm+RoPE 融合 | Sol Pillar-1 | **A** | **已落地**（C019 stale closed；`qkv_rope_f16_kernel`） | 勿重开 |
| 3 | CUDA Graph / 区域捕获 | Sol Spark 1.39× 成分 | **A** | 默认关（Z05 噪声） | 固定 shape 图；先测 launch 税占比再开 |
| 4 | GEMM+epilogue / cast 消隐 | Sol + 本仓 INT8 链路 | **A** | 部分；C011/C015 待 | nsys 定点；waffle |
| 5 | 精确 TE / RoPE 不变量缓存 | 全行业 + 本仓 | **A** | C062/C068 已有 | AUTO 默认可议；勿与近似 cache 混谈 |
| 6 | Sol **FIRST_DENSE_LAYERS=2**（± STEPS） | Sol 官方 MMDiT + V1 实测 | **B→或许 A′** | 5090 turbo4 已证；**4090 DMAD 未复测** | opt-in 产品候选；须改 `sdpa_strict` 承认 dense_skip |
| 7 | 文本 query 行 dense | Sol 官方另一半接线 | **B** | **未接线** | V1-a 无它已救回质量 → 优先级低于护栏层表 |
| 8 | DMAD/PDMD 少步合同 | DMD 族 | **B（产品少步）** | DMAD 主臂；PDMD 对照政策 | 保持合同门；PDMD 升默认须产品改合同 |
| 9 | 末步 / 低强度 **refiner LoRA**（TaoMate 0.7 类） | 社区 TaoMate | **B** | `H3_REFINER_LORA` 已接线（C035） | scale sweep；禁整段 3-step 当默认 |
| 10 | 「稀疏感知蒸馏」配方（VSA/SLA 思想） | FastH3 / SLA | **B（训练）** | 观察 | 不装 FastH3 权；若开训则 steps-aware 目标 |
| 11 | 混合注意力保运动（VDN/SLA） | VDN | **B（长片/质量档）** | 否决整模当速度解；机制重开 | 路径 A：外部 Comfy 对照质量；C 仅长片 SKU |
| 12 | 抬画布 / 二采 SKU | 本仓 RCA + 行业 latent upscaler | **B 质量档** | 证据强 | 正式报价表，不叫加速失败 |
| 13 | TeaCache / MagCache 近似跳步 | Sol-H3 50-step 柱 | **禁 @ 少步默认** | 主题禁 | 仅研究：Euler-8+ 若 rel-L1 证明有跳步空间再开 |
| 14 | FastH3 4-step 权整包 | FastH3 | **不装** | 已评差 | — |
| 15 | VDN 现成 8-step 当 T2/T4 速度解 | VDN | **不装为速度默认** | ~2× 慢 | 质量参照 / 长片候选 |

---

### G.2 「最佳可行落地栈」（shippable synthesis）

面向 **4090 产品硬约束**（INT8、cudnn 默认、DMAD/Euler、FaceRefine 停、NVFP4 BLOCK）：

#### 栈 S0 — 今日可卖（已落地）

- DiT：**INT8 ConvRot** + **cuDNN SDPA**  
- 轨迹：**DMAD-4**（速度）或 **Euler-8**（质量金线）  
- 卫生：TE HIT / RoPE memo / C073 dump-replay 可复现  
- 远脸：抬画布 SKU（PDMD/DMAD≤1152；Euler@1344），**不** FaceRefine  

#### 栈 S1 — Track A 下一装船包（目标：同合同墙钟↓ 或 VRAM↑，waffle/BIT_ID）

按 ROI 串行（**不与他机 C015 抢卡**）：

1. C015/C042 act-FP16（若 PASS）→ 768p 余量  
2. ~~C018 QKV-merge~~ **stale closed**（Exp-G3）；改 **C015** 消中转  
3. QKNorm+RoPE fuse  
4. C011 VAE cast 融合（nsys 后）  
5. Graph capture 仅当 launch 税 ≥ 阈值  

**预期诚实量级：** DiT 侧合计约 **5–15%** 量级（Sol 官方 Pillar-1 全栈仅 ~1.08× @ 50-step；少步下 launch 税占比更高但绝非倍数）。墙钟主轴仍是 NFE。

#### 栈 S2 — Track B 质量/速度 Pareto（opt-in，显式下限）

| 层 | 选择 | 门 |
| --- | --- | --- |
| 少步 | 保持 DMAD；研究臂 PDMD 同日对照 | fewstep-contract + 人眼运动 |
| 注意力 | **仅当** lib 武装后：`sol_cute` + `FIRST_DENSE_LAYERS=2`（优先于 STEPS） | waffle ≤ dense+δ；人眼无首尾糊；dit 仍明显快于 dense |
| 抛光 | `H3_REFINER_LORA` 末 1–2 步 · scale∈{0.5,0.7} | 过饱和↓、无全局糊；收益 ≥ 半档抬画布才产品化 |
| 空间 | 抬画布 / 二采正式 SKU | face-px + 可辨五官 |
| 训练演进 | 稀疏感知学生 / SLA（非 FastH3 权） | 新合同；RAFT/运动指标 ≥ dense 少步基线 |

#### 栈 S3 — 明确不装船（本窗口）

整包 FastH3-4、TeaCache@DMAD-4、NVFP4 默认、FaceRefine、VDN 当 480p 速度默认、无护栏全稀疏 sol。

---

### G.3 具体下一实验（falsifiable）

> 协议共性：固定 seed/prompt（优先 `p10_street_wave_dolly`）、burn-in、暖 KPI、`--skip-audio` 可选、记录 SHA/backend hits、**声明 Track**。不抢 C015/C064 GPU。

#### Exp-G1 · Track B · 4090「Sol 护栏 vs cudnn」@ DMAD-4（最高信息密度）

| 字段 | 内容 |
| --- | --- |
| **Claim** | 武装 `sol_cute` + `FIRST_DENSE_LAYERS=2` 后，DMAD-4@480p 相对 cudnn：dit 更快且 waffle/人眼不低于显式下限 |
| **Mechanism** | 族2 稀疏 + 官方 dense 护栏（V1-a 在 5090 turbo4 已救退化） |
| **Arms** | (1) cudnn (2) sol 全稀疏 FIRST_DENSE_*=0 (3) sol LAYERS=2 (4) 可选 STEPS=1 |
| **PASS** | 臂3 waffle 均值 ≤ 臂1 + 0.10；人眼无运动塌/首尾糊；臂3 暖 dit ≤ 臂1 × 0.92；hits 解释得清（dense_skip 计入） |
| **FAIL** | 臂2 相对臂1 waffle 超 +0.15（复现 V0）→ 确认少步全稀疏禁默认；臂3 不达 PASS → **关闭 4090 sol 产品叙事**，仅留 5090/长序列 |
| **前置** | `libh3_sol_cute.so` + cute_sm89；修/绕过 `sdpa_strict` 与 dense_skip 冲突 |

#### Exp-G2 · Track B · TaoMate/细节 refiner @ DMAD-4（C035 落地）

| 字段 | 内容 |
| --- | --- |
| **Claim** | 末 1–2 步 refiner LoRA（scale 0.5–0.7）降低少步过饱和/糊边，且不引入全局风格塌 |
| **Mechanism** | 高频对末段敏感；社区 TaoMate-as-refiner（非整段 3-step） |
| **Arms** | DMAD-4 裸；+refiner STEPS=1/2 × scale∈{0.5,0.7}；对照抬画布一档（质量锚） |
| **PASS** | 同画布人眼：过饱和/ smear 明显改善；无 identity/色偏灾难；主观增益 ≥「半档抬画布」的一半 |
| **FAIL** | 无改善或全局糊 → 维持 C035 idea，不产品化；**绝不**回退 FaceRefine stitch |

#### Exp-G3 · Track A · Sol Pillar-1「免费核」缺口证明（无长跑也可开）

| 字段 | 内容 |
| --- | --- |
| **Claim** | 当前 INT8 路径仍存在可融的三次 QKV 瘦 GEMM / 未融 QKNorm+RoPE（否则关闭 C018 叙事） |
| **Mechanism** | 族6 工程浪费 |
| **Method** | nsys 或 kernel 名计数 + 源码 diff；对比 Sol 官方 Pillar-1 列表 |
| **PASS** | 证明缺口存在 → 开工票；预估 WC 与 waffle 门写入切片 |
| **FAIL** | 已合并/不可达 → catalog 标 **stale closed**，停止估计 −0.6~−1.2s |
| **结果 2026-10-28** | **FAIL（缺口不存在）** — 见 [`exp-g3`](2026-10-28-exp-g3-pillar1-gap-proof.md)；DeepSeek Absorb/QKV 叙事同步关闭；下一刀 **C015** |

#### Exp-G4 · Track B · PDMD vs DMAD 同日 4090（研究臂）

| 字段 | 内容 |
| --- | --- |
| **Claim** | 同 4NFE 下 PDMD 运动/脸分 ≥ DMAD（beyond-dmad 论文序） |
| **PASS** | 合同 early_ok + 人眼运动不劣 + 无新伪影；墙钟 ≈ |
| **FAIL** | 不优于 DMAD → 保持对照政策，不升默认 |
| **政策** | 用户政策 PDMD=对照时本实验**不自动改产品默认** |

#### Exp-G5 · Track B · VDN 质量参照（零/近零集成）

| 字段 | 内容 |
| --- | --- |
| **Claim** | VDN 8-step 社区栈在固定 prompt 上运动保真明显优于 FastH3/本仓过饱和少步，可作为质量锚 |
| **Method** | Comfy 外部路径 A（vdn-direct-use §四）；**不**进 C CUDA |
| **PASS** | 人眼/RAFT 级运动 ≥ Euler-8@同画布的可接受带；记录 e2e 慢多少 |
| **FAIL / 定位** | 若仅慢且质不惊艳 → 维持「机制重开、模型不当速度解」；若质惊艳 → 仅开 T3 质量档讨论，**不**替换 DMAD 速度默认 |

#### Exp-G6 · 禁做对照（防手滑）

| 禁 | 原因 |
| --- | --- |
| Mag/Tea @ DMAD-4 默认开 | 少步 rel-L1 旁证零跳步；官方 4-step 也不靠 cache |
| 无护栏 sol 进默认 | V0 waffle +0.47 |
| FastH3 权当金线 | 官方+arena+运动三方差评 |
| 用官方 1.48×/4.52× 讲本仓 4-step | 合同不符 |

---

### G.4 与 §E 的合并优先级（执行序）

1. **不抢卡：** 等 C015 结果 → 并入 S1。  
2. **信息密度最高的 GPU 刀（有 lib 时）：Exp-G1**（4090 sol 护栏 vs cudnn @ DMAD）——决定 Track B 注意力叙事生死。  
3. **无 lib / 错峰：Exp-G2** refiner + **Exp-G3** Pillar-1 静态证明。  
4. **质量档并行文档化：** 抬画布 SKU 表（已有证据，差产品包装）。  
5. **训练级（慢）：** 稀疏感知学生 / SLA——仅当 G1 显示推理期稀疏上限不够。

---

## F. 修订

| 日期 | 说明 |
| --- | --- |
| 2026-10-28 | 初稿：行业地图（带链接）、本仓 Track A/B 证据表、加速且提质理论、严谨模板、分轨下一刀。分析-only，无长 GPU。 |
| 2026-10-28 | **§G 增补：** sol-h3 / FastH3 / VDN / TaoMate / DMD 族竞品吸收 → Track A/B 映射 → shippable 栈 S0–S3 → Exp-G1…G6 可证伪门。分析-only。 |
| 2026-10-28 | **Exp-G3 FAIL**：C018/C019 stale closed；C015 dig+；[`deepseek-derived-track-a-status`](2026-10-28-deepseek-derived-track-a-status.md)。 |
