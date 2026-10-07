# H3.c 方案选型：Sol-H3 / 社区加速 / 本仓配方（2026-09-16）

> **For agentic workers:** 选型备忘，不是施工单。禁止改推理、禁止改金线默认、禁止 commit、禁止开 cafe。密码不入库。执行某一模块时另开子方案，先读 [768P 评测协议](./2026-09-13-h3c-768p-eval-protocol.md) 与 [分模块枢纽](./2026-09-15-h3c-module-opt-plan.md)。

**Goal:** 核清 NVIDIA Sol-H3「DGX Spark 768P 56s」与社区「4090 有望 50s」的尺子，对照本仓 8×5090 C/CUDA 栈，决定默认路径 vs 场景切换，而不是换一个魔改 binary。

**Architecture:** 产品拆成 **配方表**（金线保真 / 热加速 5s 768P / 吞吐 8 卡），挂在同一套 generate 内核上。内核可插拔（attn / VAE / 量化 / 并行），配方互斥，金线零分叉。

**Tech Stack:** 本仓 C/CUDA `h3_cuda_generate` + `--preset`。对照只读：Sol-Engine / Sol-H3-Spark、TaoMate、SelfLift/LBH、RunningHub Lightning、VDN、aikitoria P2P。

**Spec:** [768P 协议](./2026-09-13-h3c-768p-eval-protocol.md) · [模块枢纽（含 VDN §2.5、kitchen VAE）](./2026-09-15-h3c-module-opt-plan.md) · [TaoMate](./2026-09-13-h3c-taomate-h3-analysis.md) · [SelfLift](./2026-09-13-h3c-selflift-comfy-analysis.md) · [Lightning vs Ulysses](./2026-09-13-h3c-minimax-lightning-survey.md) · [NVFP4](./2026-09-13-h3c-model-format-5090.md) · [二采](./2026-09-11-h3c-two-stage-768p.md) · [极致施工图 / aikitoria](./2026-09-13-h3c-5090-extreme-full-plan.md)

## Global Constraints

- 本仓 768P 横屏尺子是 **5s · 1344×768 · 全含 e2e**（TE + DiT + VAE + mux）。禁止和过时 480p 金线横比。
- 金线默认：U=1、8-step、`sdpa=cudnn`、官方 VAE、无 LoRA、`--te-cache` 关。
- 有损刀（少步 / lift / TAE / sage3 / NVFP4 DiT / LTX 二采）进加速档，不进 `gold`。
- 公开倍数的分母几乎都是 **50-step BF16** 或 **GB200 SGLang**，禁止除进本仓秒数。

---

## 0. 先给硬话

**Sol-H3 不是更适合我们 5090 的通吃方案。** NVIDIA 的 56s 是 **单台 DGX Spark、5 秒、1344×768、H3 4-step 384p 草稿 + LTX-2.5 3-step 精修、含新鲜 Qwen 与 MP4**；成片 decode 是 **LTX 卷积 VAE**，中间 **故意跳过 H3 官方 ViT VAE**。独立复现同一图在 **5090 上已经更快**（热 30.6 s），Spark 只是靠 **128 GB 统一内存**把两套模型常驻。社区「4090 50s 内」**没有一手 768P 全含测量**；公开 4090 是分钟级，且 Stage1 FP8 H3 ~33.5 GiB **塞不进 24 GB**。

**该采用的默认栈：** 金线仍是本仓 INT8 ConvRot + 8-step cudnn + 官方 VAE（kitchen 核追上后再换核、不换合同）。热 5s 768P 加速档继续 **lift（低 3 + 高 1 + LBH）**，P0 是 **kitchen VAE ~4s** 和 **修好 768P 注意力（sage3 禁止假 fallback）**，不是换 Sol-H3 runtime。8 卡吞吐走 **请求级 DP**，Ulysses 冻结到本机 P2P 成立。

**能否切换：** 能，而且必须切换。不存在一个 binary 通吃保真 / 热延迟 / 8 卡吞吐。现有 `--preset` 已经能挂金线 / turbo4 / lift / taomate / nvfp4；缺的是 **VAE 可插拔、attn 可插拔（真失败）、配方表写进日志**，不是再做一个魔改引擎。

---

## 1. 方案全表

栏位约定：**公开尺子**写硬件 / 画布 / 时长 / 步数 / 是否含 TE·VAE·load。**5090 能否跑**指 sm_120 + 32 GB 消费卡，不是「论文里出现过 Blackwell」。**打哪一段**指 Amdahl：TE / DiT / VAE / 通信 / 换模型。

| 方案 | 做什么 | 公开尺子（一手） | 5090 能否跑 | 打哪一段 | 质量红线 | 和本仓关系 |
| --- | --- | --- | --- | --- | --- | --- |
| **官方金线** | MiniMax-H3 发布轨迹：8-step、cfg=1、官方 ViT3D VAE tile 256/64 | 本仓已测 5s 1344×768 单卡 5090 热 **e2e 159.38 s**（TE 13.14 / DiT 112.23 / VAE 23.57）。官方 serving 常写 50-step BF16，那是另一条分母 | **能。** 金线路径就是它 | 全栈保真，不加速 | waffle / 无声 / 假帧数 = 事故。MAE 金线门 | **默认。** `--preset gold` 或不传 preset。U=1 cudnn INT8 |
| **turbo4** | 4-step 蒸馏 LoRA + sage3，cache 关 | 本仓 5s 768P turbo4+TE+TAE **80.48 s**；sage3 **hits=0 fallback=200** @ `seq≈37788`。社区 4-step 分母多为 50-step | **能跑命令。** 768P sage3 **假加速**（退 dense） | DiT NFE（50→4 的故事我们吃不到，金线已是 8） | 未蒸馏 `--steps 4` 禁止。金线永不 turbo | `--preset turbo4`。P0 是修 attn，不是再砍步 |
| **TaoMate 3+2+LoRA** | 3-step 区间蒸馏 + 4 小块 Q + clean-KV；用户栈另叠双采样 | 官方表：**8×H20、480×864、10s、TP2×U4、Pure DiT 14.810 s，不含 load/TE/VAE/mux/音频 teacher**。用户 **单卡 5090 10s 768P ≈27s** 不是开源 CLI | **开源树不能。** 只认 FA3/Hopper；`--gpus`∈{4,8}。本仓 `--preset taomate` 是 C 栈 opt-in，不是他们的 FA3 runtime | DiT NFE + 短 Q；768P 主场是 chunk 不是 3 vs 4 | 金线不动。权必须配 `{0,16,33,49}`。token_refiner 本仓曾拒名 | 已接线 `taomate` / `taomate-lift`。真对齐还缺 chunk+KV。公开 11.45× **禁止**除进 159s |
| **SelfLift / LBH** | 同一 Euler 日程：低分前缀 → 3D latent lift → 高分后缀 | 本仓 `--preset lift` 5s 768P 热 **47–53 s** 全含（47.09 / 50.39；施工图工人后 ~53）。低 3 + 高 1 DiT ≈22 s；LBH **4.63 s**。论文 SelfLift-rich **不在** Comfy 插件里 | **能。** 已是热加速档 | 多数 NFE 落在 ~1/4 token；成片仍付官方 VAE | `gold` 永不 lift。像素锚 ≠ TAE。糊/过饱和是 LBH+少高分步上限 | `h3_selflift.c` + `h3_latent_upscaler.*`（LBH-123-AI 3D resizer）。rho=0 跳 VAE 往返 |
| **kitchen VAE** | 官方 36 层 ViT 解码器，换融合核（RMS/RoPE/attn/FFN），tile-local RoPE 256/64 | Comfy 同画布 **~4 s** decode。本仓官方 **~18–23.57 s**。差在核，不在再减窗 | **能，且必须。** 这是 5090 成片 decode 的 P0 | **VAE only** | 整幅 F16 / canvas RoPE / tile>320 → 16px waffle。H3#46 | **未追上。** 枢纽 P0。不是另训 VAE，不是 VDN |
| **TAE** | Tiny 预览解码器替换官方 VAE | 本仓 TAE **7.03 s** 但彩色闪。Kijai「not the greatest」；holdout PSNR ≈25.5 dB | **能跑，金线禁** | VAE 墙钟 | 成片闪/抹/色漂。禁止用来宣称「VAE 已做到最好」 | `h3_tae*`。只许 turbo/nvfp4 预览档 |
| **NVFP4** | TE 已是 NVFP4-AWQ；DiT 社区 `pruned_nvfp4` | Comfy 同骨干 INT8 2.17 → NVFP4 1.90 **s/it（~1.14×/step）**。Sol-H3-Spark Stage1 用 NVFP4 Qwen + FP8 DiT | **sm_120 原生吃 NVFP4。** 本仓 DiT 热路径仍 dequant→INT8，禁止 CPU requant 进 generate | 单步 GEMM / 显存（12.5 vs 21 GB） | 非无损。probe 失败必须硬退出，禁止静默回 INT8 | `--preset nvfp4`。价值是与少步相乘 + 8 卡每卡一份权更松，**单独追不上 50s** |
| **Ulysses** | 一条请求切序列到多卡，注意力前 A2A 换头 | 本仓 8×5090 `canAccessPeer=0`：U=2 **更慢**，U=8 DiT **4.3× 更慢**（480p cafe；768P 总线更差） | **代码能跑，公路付不起。** 消费卡官方无 GPU P2P | 通信，不是算力 | 多卡 vs 单卡同配方 MAE 到 sage 量级 = bug | `h3_ulysses_comm.cu`。**冻结**到 P2P≥20 GB/s 或放弃切请求 |
| **请求级 DP** | 8 路各跑一条，U=1，不切层 | 无公开「H3 8-way DP」金表。施工图：聚合目标 ≥6.5× 热/电 | **能。** 不靠 P2P | 吞吐（视频/小时），单条延迟≈单卡 | TE cache 按 `gpu$i` 分目录，防脏 key | **8 卡 P0。** 尚未做 `h3_dp_generate.sh` |
| **VDN** | DiT hybrid attn（窗 softmax + Video Delta）+ 8-NFE turbo LoRA | 头牌 **8×B200、denoise 11.23 s / 更新 6.9 s**；README **排除 load / warmup / VAE / mux**。单卡 B200 8-NFE denoise 仍 **51 s**。Comfy 5090 ~**17 s/it**，作者写 headline 拿不到、**FA4 不服务 sm_120** | **不成片主路径。** 32 GB 常驻不了 B200 FP8 配方 | DiT attn 形状，**不改 VAE** | 口径不含 VAE 要拆穿。金线禁换 attn 合同 | 研究档。见枢纽 §2.5。不要当 kitchen 替代 |
| **Sol-H3**（家族，下一节拆页） | Sol-Engine 运行时：融合核 + Sol-Attn + cache；Spark 页另加 H3→LTX 二采 | 见 §2。56 s = Spark 5s 768P **两阶段全含**；8×B300 4-step **1.653 s**；5090 50-step 近无损 **231.2 s**；GB200 Super **6.85 s** 是另一台机器 | **核：Sol-Attn 官方表含 sm_120 CuTe。运行时：Spark 常驻 116.9 GiB，5090 装不下整栈。** Stage2 钉过 `cute_sm100` | Spark 页：换 LTX + 跳过 H3 VAE + 少步；OnDevice 页：50-step 近无损 | 二采是异模型，不是 H3 金线。TAE/LTX 成片 ≠ 官方 VAE | **不替换本仓默认。** 可借鉴 Sol-Attn（768P 注意力）与「草稿+精修」思想；不要把 Spark 56s 当 5090 KPI |
| **RunningHub Lightning** | 50→4 蒸馏 + Sage2 + Cache-DiT + compile + TP2×Ulysses4 | [README](https://github.com/RH-RunningHub/MiniMax-H3-MultiGPU-Lightning)：**8×RTX 6000D sm_120 85 GB**，5s 1344×768 **warm 生成 28.7 s**（相对同机 50-step 348.8 s = 12.2×）。**不含 load / 首次 compile / 排队。** 15s t2va **48.2 s**。自家加速权 **未公开** | **拓扑可参考，数字不能抄。** 6000D 有 PCIe P2P；本仓 5090 `peer=0`、32 GB、金线已 8-step | 少步占 12.2× 的 **95%**；多卡只是同 8 卡上的切分 | BF16 权 ≠ 无损（Sage/Cache/蒸馏改算术） | 抄「少 TP、多 Ulysses、别关 P2P」；禁止把 12.2× 写成 8 卡÷1 卡 |
| **aikitoria P2P** | 非官方 open kernel 打开 GeForce BAR1 P2P | 作者机 8×5090+EPYC：单向 ~**55 GB/s**。vLLM 论坛另一台 2×5090：**~14.1 GB/s**。NVIDIA 官方：**50 系无 GPU P2P**，产品页 [NVLink (SLI-Ready): No](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/) | **未在本机验证。** 驱动必须锁 `595.45.04-p2p`；缺 `iommu=pt` / 有人 generate **禁止装** | 只打 Ulysses 公路，不打 TE/VAE | hang / GSP / 覆盖 `nvidia.ko`。`can=1` 但拷贝坏 → 位级不过就回滚 | [施工图 Phase 3](./2026-09-13-h3c-5090-extreme-full-plan.md)。**不接 C 栈。** 源：https://github.com/aikitoria/open-gpu-kernel-modules |

补充（表内已覆盖机制，避免重复行）：**H3 Super Acceleration** = Sol 家族的 GB200 二采（896×512 4-step + LTX 3-step，博客 5s **6.85 s** / 10s **14.93 s**，分母已发表 SGLang，且独立评测指出「分段相加、不含 load/warmup」）。**本仓无 LTX，不能按 6.85 s 复现。**

---

## 2. Sol-H3 专节：数字真假、Spark vs 5090、是否通吃

### 2.1 名字不要混

| 名字 | 一手页 | 实际是什么 |
| --- | --- | --- |
| **Sol-Engine** | https://github.com/NVlabs/Sana/tree/sol-engine · https://nvlabs.github.io/Sana/Sol-Engine/ | 训练无关加速框架（cache / 稀疏 attn / 量化 / 融合 / token prune），跑在 SGLang `multimodal_gen` |
| **Sol-Attn** | arXiv:2607.24027 · Engine README | 在线 softmax 里做 block 选择的稀疏注意力。官方 backend：**4090 SM89 / H100 SM90 / B200 SM100 / 5090 SM120 = CuTe**；无 CuTe 或更旧卡走 Triton。**GB10 Spark 是 sm_121**：CuTe 表停在 sm_120，官方包装 **fallback Triton** |
| **H3 On-Device** | https://nvlabs.github.io/Sana/Sol-Engine/H3-OnDevice/ | **仍 50-step、无 LoRA**。Spark 测 **480p**，5090 测 **768p**。近无损 4.52× |
| **H3 Super Acceleration** | https://nvlabs.github.io/Sana/Sol-Engine/H3-Super-Acceleration/ | H3 少步草稿 + **LTX-2.5** 精修。头牌 **1×GB200** |
| **Sol-H3** | https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3/ | 8×B300 上「比播放还快」的 4-step serving（5s **1.653 s**） |
| **Sol-H3-Spark** | https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3-Spark/ | **用户问的 56s 就是这一页** |

论文：Sol-Engine [arXiv:2606.23743](https://arxiv.org/abs/2606.23743)；Sol-Attn [arXiv:2607.24027](https://arxiv.org/abs/2607.24027)。

### 2.2 56s 尺子——逐项核清

来源：Sol-H3-Spark 页原文（2026 检索）。

| 项 | 官方口径 | 拆穿 / 对照 |
| --- | --- | --- |
| **时长** | **5s @ 24 fps**，1344×768，T2VA/FL2VA | **不是 10s。** 10s 头牌在 Super Accel 页，是 GB200 14.93 s |
| **分辨率** | 输出 1344×768；草稿 **672×384（384p）** 124 帧 | 与本仓 768P 官方桶一致。草稿不是 768P dense |
| **步数** | Stage1 **H3 4-step LoRA**；Stage2 **LTX 3-step** | 不是 8-step 金线，也不是 50-step On-Device |
| **精度** | NVFP4 Qwen + **FP8 H3 DiT** + 低 bit 视频件 | 换精度 + 换模型 |
| **单卡/多卡** | **1× DGX Spark** | 不是 8 卡，不是 4090 |
| **含 TE？** | **含。** 「fresh Qwen encoding」；分解 **Qwen 2.4%** | 热路径仍每次编 prompt（Stage2 Gemma **缓存掉**） |
| **含 H3 VAE？** | **不含成片 H3 VAE。** 学到的 VAE Adapter 做 H3→LTX latent，**Inter-stage VAE 2→0** | **口径陷阱。** 最后 decode 是 **LTX Conv VAE（分解 12.5%）**，不是本仓 36 层 ViT |
| **含 mux？** | **含。** 「completed MP4 with audio」；mux **1.3%** | 与本仓 `phase_ms total` 同类：请求入口 → 可播文件 |
| **含 load？** | **热常驻。** 设计目标是 116.9 GiB 整栈留在 119.68 GiB UMA，避免每请求切权 | 冷启动 / 第一次 compile **不在 56s 里**。独立评测冷 118.8 s vs 热 73.3 s |
| **sm 覆盖** | Spark GPU = **GB10 / sm_121**，不是 5090 的 **sm_120** | Sol-Attn CuTe 官方列 **sm_120**；Spark 实测常走 Triton。本仓 5090 编译目标是 sm_120，**不要假设 Spark cubin 能直接跑 5090** |

分解（官方把 56.17 s 摊开）：H3 4-step **35.2%** ≈ 19.8 s；LTX 3-step **44.0%** ≈ 24.7 s；LTX VAE **12.5%** ≈ 7.0 s；upscaler+adapter **3.8%**；Qwen **2.4%**；mux **1.3%**；overhead **0.9%**。

独立复现（同一 672×384→1344×768 图，[ai-muninn 2026-09-14](https://ai-muninn.com/en/blog/h3-ltx-two-stage-dgx-spark)）：

| | DGX Spark | RTX 5090 |
| --- | ---: | ---: |
| Native H3 1344×768 4-step 冷 / 热 | 237.7 / 230.1 s | **51.5 / 41.5 s** |
| 两阶段 H3+LTX 冷 / 热 | 118.8 / 73.3 s | **40.3 / 30.6 s** |
| 热 + Sol-Attn + 解时钟帽 | **66.1 s**（仍慢于官方 56.17，约 1.18×） | — |

**结论：** 56s **不是假新闻**，但是 **Spark 专用两阶段配方的热 e2e**，复现落到 66–73 s。把它说成「MiniMax-H3 官方 VAE、8-step、单段 768P」是假口径。

### 2.3 「4090 有望 50s」——一手找不到

检索（2026-09-16）：Sol-H3、MiniMax H3、768p、4090、50s / 50 seconds。**没有**与 Spark 页同等级的 4090 768P ≤50 s 全含表。

找得到的 4090 一手 / 准一手：

| 来源 | 尺子 | 秒 | 为何不是「50s 内」 |
| --- | --- | --- | --- |
| [Pedro Alonso](https://www.pedroalonso.net/blog/minimax-h3-local-rtx-4090/) | 1×4090，1344×768，turbo **12** step，5s | 热 **~300 s**，冷 ~330 s；10s turbo4 **275 s** 且质量崩 | 分钟级 |
| [yume-arasaki recipe](https://github.com/yume-arasaki/RTX-4090-3090-Minimax-H3-15s) | 4090 24 GB，**832×480** 8-step | 5s **~7 min**；15s 25–30 min | 480p 都不是 50s |
| [awesome-minimax-H3 performance](https://github.com/wildminder/awesome-minimax-H3/blob/main/guides/minimax-h3-performance.md) | 24 GB 级汇总 | ~4 min（1 MP）– 12 min（10 s） | 社区共识分钟级 |
| Sol-Engine README | MiniMax-H3 **列出** 4090 为支持卡（SM89 CuTe） | **未给 768P 50s** | 支持 ≠ 56s 配方 |

容易串味的数字（都不是 4090 768P 金线）：

- **5090** native 4-step 768P 热 **41.5 s** / 冷 51.5 s（muninn）——最像「50s」谣言的源头。
- **5090** 两阶段热 **30.6 s**（muninn）。Stage1 权 **33.5 GiB**，作者写 **5090 31.84 GiB 都装不下 Stage1、每轮重载**；**4090 24 GB 更装不下**。
- RunningHub **15s** 竖屏 768P **48.2 s** = **8×6000D**，4-step，warm 生成、不含 load。
- Spark **56 s** 被口播成「消费卡也能」。

**拆穿：4090 50s 内若属实，必须是「不含 H3 VAE / 不含 TE / 480p / 不计 load / 或其实是 5090」。按现公开测量，不属实。**

### 2.4 DGX Spark vs RTX 5090（我们的箱子）

| | DGX Spark（GB10） | GeForce RTX 5090 |
| --- | --- | --- |
| 架构 | Grace Blackwell Superchip，GPU **sm_121** | Blackwell 消费卡 **sm_120** |
| 内存 | **128 GB** LPDDR5x **统一** CPU+GPU，带宽 **273 GB/s**（[NVIDIA Spark 页](https://www.nvidia.com/en-in/products/workstations/dgx-spark/)） | **32 GB GDDR7** 专用显存（[5090 产品页](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)） |
| 功耗 | GB10 TDP **140 W**，电源 **240 W** | TGP **575 W**；NVLink **No** |
| 峰值叙事 | 最多 1 PFLOP **sparse FP4**；LMSYS 档位常被写成 5070 级 | 21760 CUDA core，显存带宽约 **1.8 TB/s 级**（~6.5× Spark） |
| Sol-H3 角色 | **容量机器**：两套模型 + adapter 常驻 116.9 / 119.68 GiB | **算力机器**：同图两阶段已 **快于 Spark 2.4–5.5×**，但 **32 GB 留不下 Spark 常驻栈** |
| 对本仓 | 买 Spark 是买 UMA 做实验床，不是买 768P 延迟冠军 | 8×5090 已在手。单卡延迟应打 5090 表，不是 Spark 56s |

**sm_120：** 本仓 `h3_cuda_device_caps` / `h3_cuda_info` 把 5090 认成 sm_120；sage3、NVFP4 能力位按 sm_120+。Sol-Attn 官方 **为 5090 提供 CuTe sm_120**。VDN/FA4、FA3/Hopper、DeepGEMM SM90/100、FastH3 `block_sparse_attn_sm100a` **不是** sm_120 资格集。Spark 的 sm_121 与 5090 **不能当同一份 cubin**。

### 2.5 Sol-H3 是否通吃？

**不通吃。** 它是「特定模型 × 特定硬件 × 特定时长/分辨率」的搜索结果（On-Device 页自己写：knob 不迁移）。对我们：

1. **保真金线** — Sol-H3-Spark 换了 LTX 成片 + 4-step 学生，过不了官方 VAE / 8-step 合同。
2. **热 5s 768P 延迟** — 方法（低分草稿 + 少步精修）我们 **已经有 lift**；异模型 LTX 本仓没有，且 32 GB 常驻双栈会变成 **换权税**（muninn：5090 每轮 reload Stage1）。Spark 56s 在 5090 上不是更优 KPI，**同方法已经 30s 级**（Comfy 两阶段），本仓 lift 热 ~53s 全含，差在 **VAE 18s + 768P dense attn**，不在「没买 Spark」。
3. **8 卡吞吐** — Sol-H3 8×B300 是 NVLink/NVL 世界；抄到 PCIe 5090 = 再演 U=8 变慢。

可偷的三块，按优先级：**Sol-Attn sm_120 核（修 768P 注意力）> 融合 AdaLN/RoPE（lossless，On-Device 第一柱）> 不要把 FirstBlockCache 叠进 4/8-step 蒸馏档**（本仓已测 8-step 几乎不跳，turbo4 直接拒绝 cache）。

---

## 3. 我们该采用什么

### 3.1 三条产品公路（必须切换，不要平均成一条）

| 场景 | 配方 | 质量尺 | 延迟/吞吐 KPI | 明确不做 |
| --- | --- | --- | --- | --- |
| **A 金线保真** | INT8 ConvRot · 8-step · cudnn · 官方 VAE 256/64 tile-local RoPE · U=1 | 协议 MAE∩有声∩无 waffle | 墙钟诚实报全含。VAE 追上 Comfy ~4s 后 e2e 才有资格和外部 27s 比 | LoRA / lift / TAE / sage3 / U>1 / Sol-H3 / VDN |
| **B 热加速 5s 768P** | `--preset lift`（低 3+高 1+LBH）+ **kitchen 官方 VAE** + 真快 attn（修 sage3 或 chunk Q）+ `--te-cache` 热开 | 能认人、有声、禁止花屏；允许糊/过饱和 | 今日热 **~47–53 s** 全含。P0 把 VAE 18s→~4s 后，同配方应落到 **~35 s 量级（回代）**；再修高分 15s dense | 用 TAE 冒充赢了；把 Spark 56s 当验收；未蒸馏 4-step |
| **C 吞吐 8 卡** | 8× `CUDA_VISIBLE_DEVICES=i` 请求级 DP，每路 = A 或 B | 单条不差于单卡 10% | 聚合 ≥6.5×。Ulysses **不是**这条公路 | U=8 cafe；TP2×U4 抄 H20/6000D |

TaoMate 3-step、turbo4、nvfp4、H3→LTX 二采 = **B 的实验支路**，各自 `--preset`，过门再升正式加速档。

### 3.2 硬建议（三条）

1. **不要把 Sol-H3 / Spark 56s 当 5090 默认栈。** 56s 是 5s 两阶段 + LTX VAE + 128 GB 常驻。我们 5090 上同思想已经有独立 30s 级（Comfy）和本仓 lift ~53s；缺的是 kitchen VAE 与 768P attn，不是换 NVIDIA runtime，也不是买 GB10。
2. **默认产品路径继续分列：A 金线不动；B 热路径锁 lift + kitchen VAE（P0）+ 768P 真稀疏/分块注意力（P0）；C 八卡只做 DP。** turbo4 在 sage3 fallback 修好之前不要当 768P 头牌。TaoMate 只许 chunk+KV 对齐后的 opt-in。
3. **架构上采用「内核可插拔 + 配方表」，禁止通吃 binary。** 一个 generate 挂多 `--preset`；VAE / attn / 量化 / 并行都是失败可见的 backend。Sol-Attn、kitchen、NVFP4 MMA 以插件进 B，永不污染 A 的日志合同。

### 3.3 执行顺序（仍不在本文改代码）

```text
P0-VAE kitchen（无 waffle） ──► 重测 lift 热 5s 768P 的 vae= 段
P0-DiT 768P attn（禁止 hits=0 还报 sage3）── 可与 VAE 并行写，不开 cafe 冲 27s
P1-请求级 DP 启动器（8 卡吞吐）
P1-Sol-Attn sm_120 仅加速档对照（对金线 MAE 门）
P2-TaoMate chunk+KV；NVFP4 原生 MMA
多卡 Ulysses / aikitoria ── 冻结，直到本机 P2P 探针或换 6000D
Sol-H3-Spark / LTX 二采 ── 附录。无 LTX 权与 32 GB 常驻方案前不做产品
```

---

## 4. 架构健壮性：现有 `--preset` 能不能挂多配方？

### 4.1 已经有的（够挂「配方」，不够挂「后端」）

`h3_generate_preset.c` / `tools/h3_cuda_generate.c` 今日：

| `--preset` | 行为 | 金线污染 |
| --- | --- | --- |
| 不传 / `gold` | 不改 steps/sdpa/cfg | 无 |
| `turbo4` | 4-step + 必须 LoRA + 默认 sage3；拒 cache | `non_gold=1` |
| `nvfp4` | 8-step cudnn；拒 cache | 非金线 |
| `lift` | 默认 8-step cudnn；拒 cache | 非金线 |
| `taomate` / `taomate-lift` | 默认 3-step sage3 + 必须 LoRA；lift 位置位 | 非金线 |

互斥已经写进 preset：turbo/lift/taomate/nvfp4 拒绝 mag/tea/fb/reuse。日志有 `non_gold` 警示。**这一层够按场景切换配方**（步数、LoRA、是否 lift）。

### 4.2 缺什么（不够「内核可插拔」）

| 缺口 | 今天 | 配方表需要 |
| --- | --- | --- |
| **VAE backend** | 官方 C+cuDNN tile vs TAE 开关；kitchen 未到顶 | `vae=official_tile \| kitchen \| tae`，金线只允许前两者中的合同对齐实现；TAE 加速档。失败硬退，禁止静默 TAE |
| **Attn backend** | `sdpa=cudnn\|sage3\|…`；768P sage3 **hits=0 fallback=200 仍继续** | 缺插件或 shape 不支持 = **硬失败**。加速档可挂 Sol-Attn sm_120 / chunk-Q。禁止假 sage |
| **量化 backend** | INT8 金线；NVFP4 probe 合同已写「失败硬退」 | 保持。generate 按 safetensors 布局选，不把格式焊死 |
| **并行 backend** | Ulysses U∈{1,2,4,8}；无 DP 启动器 | `parallel=none \| ulysses \| dp`。默认 none。U>1 在 `peer=0` 上应警告 |
| **二采 / 异模型** | lift = H3→H3 LBH；无 LTX | `refine=none \| lbh \| ltx` 仅加速档。LTX 未落地前枚举存在但 CLI 拒 |
| **口径** | `phase_ms` 有 te/dit/vae/mux | 强制：冷热、是否含 TE/VAE/load、步数、画布、U。对外禁止只报 DiT |

### 4.3 通吃方案存在吗？

**不存在。** 金线要的是发布轨迹 + 官方 VAE；热延迟要的是少 token × 少 NFE × 快 decode；8 卡 PCIe 要的是 **不通信**（DP）。Sol-H3-Spark 用「换 LTX + 128 GB 常驻」把前两件事绑在一台 GB10 上，**绑死硬件与质量尺**，对我们 8×5090 是错的通吃。

正确形态：

```text
h3_cuda_generate
  ├── kernel: TE | DiT(quant, attn) | VAE | mux | (optional lift)
  ├── backends: 失败可见、可测、可回滚
  └── recipes (--preset): gold | lift | turbo4 | taomate | nvfp4 | dp-wrap
```

一个魔改 binary（把 Sol-Engine / Comfy / SGLang 嵌进生产）会把金线合同、waffle 红线、INT8 数值门全部弄脏。**内核可插拔 + 配方表，而不是一个通吃 binary。**

现有 `--preset` **能挂多配方的「步数/LoRA/lift」轴**；要把 Sol-Attn、kitchen、DP 变成可切换场景，还差 backend 接口，不差再写一个引擎。

---

## 5. 本仓现状锚（禁止和 480p 金线横比）

| 锚 | 5s · 1344×768 · 1×5090 | 出处 |
| --- | --- | --- |
| 金线 e2e | **159.38 s** | TaoMate 分析 §3.1 |
| 热 lift | **47–53 s** 全含 | 同上 + 用户现行 |
| 官方 VAE | **23.57 s** / 196 窗 **~18 s** vs Comfy kitchen **~4 s** | 模块枢纽 |
| TAE | **7.03 s** 彩色闪 | 枢纽 |
| 金线步数 | **8-step** | 协议 |
| 768P RoPE | **tile-local**；canvas denom = waffle | `h3_vae_int8.h` |
| sage3 768P | **fallback** | 枢纽 §3.3 |
| 8 卡 Ulysses | 更慢；`peer_access=0` | Lightning 文 |

对照用户「10s 768P ≈27s」：那是 **另一条栈**（少步+分块+双采样+快 decode，口径未钉死是否全含）。本仓 5s lift ~53s 按每秒视频仍大约 3×+。主因不是 Sol-H3 没接上，是 **VAE 核 + 768P 仍付 S≈37k dense**。

---

## 6. 一手链接

- Sol-H3-Spark（**56s 原页**）：https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3-Spark/
- Sol-H3 8×B300：https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3/
- H3 On-Device（5090 50-step 231.2 s）：https://nvlabs.github.io/Sana/Sol-Engine/H3-OnDevice/
- H3 Super Acceleration（GB200 6.85 s）：https://nvlabs.github.io/Sana/Sol-Engine/H3-Super-Acceleration/
- Sol-Engine 代码：https://github.com/NVlabs/Sana/tree/sol-engine
- 独立 Spark vs 5090 两阶段：https://ai-muninn.com/en/blog/h3-ltx-two-stage-dgx-spark
- 独立 5090 两阶段：https://ai-muninn.com/en/blog/minimax-h3-ltx-hybrid-refine-rtx5090
- DGX Spark 规格：https://www.nvidia.com/en-in/products/workstations/dgx-spark/
- RTX 5090 规格（32 GB、575 W、无 NVLink）：https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/
- MiniMax-H3：https://github.com/MiniMax-AI/MiniMax-H3
- RunningHub Lightning：https://github.com/RH-RunningHub/MiniMax-H3-MultiGPU-Lightning
- TaoMate：https://github.com/TaoLiveAIGC/TaoMate-H3
- VDN：https://openvdn.github.io/ · https://github.com/OpenVDN/vdn-minimax-h3
- aikitoria：https://github.com/aikitoria/open-gpu-kernel-modules
- 4090 分钟级对照：https://www.pedroalonso.net/blog/minimax-h3-local-rtx-4090/

---

## 7. 不要做的事

- 不要 commit。不要改 `h3_cuda_generate` 默认。不要开 5s/10s cafe 冲 27s / 56s。
- 不要把 Spark 56s、RH 28.7 s、GB200 6.85 s、VDN 14.4 s 和本仓 53s 画在同一根轴上而不标尺子。
- 不要为「通吃」嵌 Sol-Engine Python。不要生产箱装 aikitoria（缺窗口 / 缺 `iommu=pt` / 有人 generate）。
- 密码、HF token、SSH 不入库。
