# DeepSeek 公开性能优化：技术清单 + 对本仓可迁移性

> **日期：** 2026-09-13。调研文档。未改推理代码，未 commit，密码未入库。不与正在跑的 host_stage / TAE / NVFP4 抢同一批源文件。
> **范围：** 一手（[V3 技术报告](https://arxiv.org/abs/2412.19437) §3、[Day 6 推理概览](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)、DeepEP / DeepGEMM / FlashMLA / DualPipe / EPLB / 3FS README）。二手只作索引。驱动结论已钉死，见 [ulysses-driver-options §8](./2026-09-13-h3c-5090-ulysses-driver-options.md)：他们挖的是用户态，没改 `nvidia.ko`。
> **本仓尺子（不推翻）：** 本地 C/CUDA 视频 DiT 推理，不是 MoE 训练。8×5090 PCIe，`peer_access=0`，无 NVLink。cafe 864×480 5s：金线 e2e ~79 s / DiT 31.67 s；turbo4 ~64 s / DiT 16.47 s；turbo4+TE HIT ~50 s。瓶颈：DiT、TE、`dit_host_stage` ~13 s、VAE ~10 s。已测：TE cache 无损；Ulysses 正确但 U=2 更慢、U=8 大亏。

**一句话：** DeepSeek 的课是用户态把通信体积、重叠和 kernel 抠到极限，不是改驱动。能搬的是「少字节、少同步、重叠独立阶段」；不能搬的是 NVLink/IB 上的 MoE 训练图。

---

## 0. 他们在什么硬件、什么图上抠

| 项 | DeepSeek-V3 / R1（公开） | 本仓 |
| --- | --- | --- |
| 卡 | 训练 2048× **H800**；推理也是 H800 节点 | 8× **RTX 5090** |
| 节点内 | **NVLink + NVSwitch**，论文写 ~160 GB/s | **无 NVLink**，`canAccessPeer=0`，A2A 走 host bounce |
| 节点间 | IB（CX7 量级 ~50 GB/s），解码用 **IBGDA** | 单机；Ulysses 不走网卡 |
| 模型 | 671B MoE，每 token 激活 37B；MLA + 路由专家 | MiniMax H3 视频 DiT，单卡 INT8 已常驻 |
| 热路径 | 训练：PP16 × EP64 × ZeRO-1 DP，**不用 TP**。推理：PD 分离 + 跨节点 EP + 双 micro-batch 重叠 | 单请求 generate：TE → DiT 8-step → VAE。Ulysses 只切 DiT |
| 精度 | 训练/推理 GEMM+dispatch **FP8**；MLA 核与 combine **BF16** | 金线 DiT INT8 ConvRot；TE 已是 NVFP4-AWQ（热路径仍 dequant→GEMM）；A2A **F32** |

H800 被砍的是相对 H100 的**跨节点/互联档位**，不是节点内 NVLink。他们的「软件路由」站在 **已经有 P2P** 的箱子上。5090 没有这条公路。

Open Source Week（2025-02，[open-infra-index](https://github.com/deepseek-ai/open-infra-index)）：Day1 FlashMLA → Day2 DeepEP → Day3 DeepGEMM → Day4 DualPipe + EPLB + profile-data → Day5 3FS / Smallpond → Day6 推理系统概览。后面 DeepGEMM 扩到 FP4 / Mega MoE；DeepEP V2 换 NCCL Gin。

---

## 1. DeepSeek 优化全表（层 / 技术 / 墙 / 依赖）

每条：做什么、解决什么墙、开源位置、是否依赖 NVLink / H800 / 训练图。

### 1.1 Kernel / PTX

| # | 技术 | 做什么 | 墙 | 开源 | 依赖 | 推理也这样？ |
| --- | --- | --- | --- | --- | --- | --- |
| K1 | **DeepGEMM** | 用户态 JIT：FP8/FP4/BF16 GEMM、MoE grouped GEMM、Mega MoE（dispatch+两段线性+SwiGLU+combine 熔进一核）。WGMMA / TMA / warp specialization / persistent schedule / 非对齐 tile；旧版 FFMA SASS interleave（NVCC 12.9+ 官方已做，他们关掉后处理） | Hopper/Blackwell 上通用 cuBLAS 吃不满细粒度 block-scale FP8 | [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | **SM90 / SM100**，CUDA 12.9+，CUTLASS 4。**不是 SM120** | 是。训练前向/反向 + 推理 prefill（contiguous grouped）+ decode（masked grouped，对接 DeepEP 低延迟输出） |
| K2 | **FlashMLA** | MLA / 稀疏 MLA 的 prefill+decode 手写核；paged KV；FP8/FP4 KV cache；V4.1 还融 norm+RoPE+attn+cast | 标准 MHA/FlashAttn 对「缩成 latent 的 KV + 吸收后的形状」不匹配；decode 带宽墙 | [deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA)；NVIDIA 提了 [SM100 MHA PR #76](https://github.com/deepseek-ai/FlashMLA/pull/76) | **SM90 / SM100**。H800 数字：dense decode 最高 ~3000 GB/s / 660 TFLOPS | **是，而且主要是推理 decode。** 训练反向另有 SM100 MHA |
| K3 | **手写 PTX / cache hint** | DeepEP：`ld.global.nc.L1::no_allocate.L2::256B` 读 volatile（文档自称 UB，Hopper 上测过）；论文：customized PTX + chunk auto-tune，减通信核砸计算 SM 的 L2。DeepGEMM：TMA + WGMMA + CUDA-core promotion 做 block-scale | 通信核和 GEMM 抢 L2；默认 `volatile` 加载编不出 | DeepEP [docs/legacy.md](https://github.com/deepseek-ai/DeepEP/blob/main/docs/legacy.md)；V3 论文 §3.2.2 | Hopper 语义。`DISABLE_AGGRESSIVE_PTX_INSTRS=1` 可关。DeepEP #21：PTX **与 OS driver 无关** | 通信核：训练/prefill 用。Decode 低延迟核发完 RDMA 就退，不长期占 SM |
| K4 | **20 SM 留给通信** | 论文：20 SM / 10 channel，warp specialization 拆 IB 发、IB→NVLink 转发、NVLink 收。DeepEP V1 默认 `Buffer.set_num_sms(24)`。V2 分析式选 SM，V3 类训练 **24→4–6** | DualPipe 要「通信核常驻 + 计算核并行」，通吃 IB+NVLink 又不能把 132 SM 全占掉 | 论文 §3.2.2；DeepEP README / [issue #179](https://github.com/deepseek-ai/DeepEP/issues/179) | **节点内 NVLink + 节点间 IB**。H800 有 132 SM，扣 20 还剩 112 | **训练/prefill：是**（normal 核整段占 SM，因为要排队 + NVLink 转发）。**Decode：不是长期占 20 SM。** 低延迟核：发指令 ~us 后退；`return_recv_hook` 期间 **0 SM**，数据走 RDMA。论文解码段：只给 dispatch+MoE+combine 留**一小撮** SM，把 SM 留给 MLA |
| K5 | **论文 §3.5 自己说没做成的** | 他们**明确希望**以后芯片把 IB↔NVLink 转发、RDMA↔用户缓冲、combine reduce 从 SM 卸走（当 co-processor）。20 SM 是妥协，不是目标。还点名：Tensor Core FP8 累加只有 ~14 bit；没有原生 tile/block scale；没有 online quant 融进 TMA | SM 做通信 = tensor core 空转 | 论文 §3.5，没有对应仓库 | 硬件愿景 | 训练+推理都适用的抱怨 |

**不要学的读法：** 「留 20 SM 给通信」在 5090 上是负优化。我们没有 NVLink 通信核可重叠；NCCL SHM A2A 已经在占带宽。再扣 SM 只会饿 INT8 GEMM / cuDNN SDPA。

### 1.2 通信（DeepEP）

| # | 技术 | 做什么 | 墙 | 开源 | 依赖 |
| --- | --- | --- | --- | --- | --- |
| C1 | **Normal / 高吞吐核** | MoE **dispatch / combine** all-to-all。训练 + 推理 **prefill**。FP8 dispatch、BF16 combine。可控 SM 数。`async_with_compute_stream` | 跨节点 EP 的 A2A 和计算 1:1，通用 NCCL 吃不满 IB+NVLink 不对称带宽 | [DeepEP](https://github.com/deepseek-ai/DeepEP) README；V1 [legacy.md](https://github.com/deepseek-ai/DeepEP/blob/main/docs/legacy.md) | **NVLink 节点内 + RDMA 节点间**。V1 还要 NVSHMEM |
| C2 | **Low-latency 核** | 纯 RDMA、最坏情况预分配缓冲、CUDA Graph 友好。decode 用。表：EP8 dispatch 77 µs | decode 批量小（论文：每专家通常 ≤256 token），要的是延迟不是吞吐 | 同上；V1 `low_latency_dispatch/combine` | IBGDA / RDMA。V2 写明 **0-SM RDMA 低延迟 EP 不再支持**（换 Gin 路径） |
| C3 | **NVSHMEM / IBGDA → NCCL Gin** | V1：NVSHMEM + 官方 `NVreg_EnableStreamMemOPs` / `PeerMappingOverride`（[docs/nvshmem.md](https://github.com/deepseek-ai/DeepEP/blob/main/docs/nvshmem.md)——标题「Configure NVIDIA driver」= **官方 modprobe 键**，不是 fork ko）。V2：NCCL Gin，复用已有 communicator | GPU 直接发 IB，砍 CPU doorbell 延迟 | DeepEP V2 README | 数据中心 RDMA NIC。本机虽有 `mlx5_*`，Ulysses **不走**这块 |
| C4 | **IB→NVLink 软件路由** | 每 token 最多去 **4 节点**；先 IB 到对端同号 GPU，再 NVLink 转发到真正的专家。IB 与 NVLink 重叠。8 路由专家可扩到理论 13 而不加通信价 | H800 节点间 IB 50 GB/s vs 节点内 NVLink 160 GB/s（约 3.2×） | 论文 §3.2.2 | **必须同时有 IB 和 NVLink** |
| C5 | **计算重叠接口** | `EventOverlap` / `previous_event` / decode 的 `return_recv_hook`（钩子触发前 RDMA 在后台走、不占 SM） | 通信和 GEMM 串行 | DeepEP README + legacy 低延迟例 | 有独立的计算流；Ulysses 语义上 A2A **必须等齐**才能 SDPA |
| C6 | **FP8 通信** | dispatch 激活打 FP8（scale 用 2 的幂）；combine 留 BF16 | A2A 字节墙 | 论文 §3.3.3；DeepEP native FP8 dispatch | 与 FP8 GEMM 合同对齐 |
| C7 | **V2 其它** | ElasticBuffer；EP 到 2048；0-SM Engram/PP（RDMA）、0-SM CP（Copy Engine）；实验性 DP/TP all-gather / reduce-scatter。第三方：Tencent zero-copy、AntGroup SM-free RDMA | V1 占 SM 太多 | DeepEP README News | 仍要 NVLink 或 RDMA |

**MoE A2A ≠ Ulysses A2A。** DeepEP 切的是「token → 专家所在卡」（稀疏、不等长、可 FP8）。我们切的是「整段 packed 序列的行 ↔ 头」（稠密、每层两次、F32、必须全到齐）。形状不能照搬。

### 1.3 流水（DualPipe / DualPipeV / 推理双 batch）

| # | 技术 | 做什么 | 墙 | 开源 | 依赖 | 推理有没有对应物？ |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | **DualPipe** | 双向 PP：微批从管道两端同时喂。一块切成 attn / A2A dispatch / MLP / A2A combine；反向再拆 input-bwd / weight-bwd。一对 fwd+bwd 重叠，把 A2A 和 PP 通信藏进另一方向的计算 | 训练 EP 跨节点，算:通信 ≈ 1:1；1F1B / ZB1P bubble 大 | [deepseek-ai/DualPipe](https://github.com/deepseek-ai/DualPipe)；论文 §3.2.1；[profile-data](https://github.com/deepseek-ai/profile-data) | **训练图**（要反向）。双份参数。PP 偶数。PyTorch+NCCL | **没有。** DualPipe README 写明是 V3/R1 **training** |
| P2 | **DualPipeV** | Sea AI Lab「cut-in-half」：同样 bubble 公式，设备数减半（每卡两段）。他们承认 Dual 的 2× 参数浪费 | DualPipe 参数翻倍 | DualPipe 仓内 `example_dualpipev.py`；[SAIL 博文](https://sail.sea.com/blog/articles/63) | 仍是训练 PP+EP | 无 |
| P3 | **推理：双 micro-batch 重叠** | **不是 DualPipe。** Prefill：两个差不多大的 micro-batch 交错，一个的 attn/MoE 盖住另一个的 dispatch/combine。Decode：attn 拆两步，五段流水；一个 batch 的 attn 盖住另一个的 dispatch+MoE+combine | 推理没有反向可藏；decode 各段时长不齐 | 论文 §3.4；Day 6 概览 + profile-data 图 | 大 EP 批量、跨节点、PD 分离 | **这才是推理对应物。** 前提：有两个独立、可交错的计算块，且通信能在后台走 |
| P4 | **他们没做的 PP 用法** | 推理不用 DualPipe，不用「TE 在 0 卡、DiT 在 1–7 卡」那种层流水。训练刻意 **不用 TP**（§3.2 开头：内存抠完就避免 costly TP） | TP all-reduce 在他们的 EP 图上更亏 | 论文原文 | — | 本仓 `cfg=1` 也没有第二条臂可 PP |

对本仓：Ulysses 的 QKV-A2A 和 SDPA **有数据依赖**，不能把 184 GB bounce 藏进 GEMM。能重叠的是 **与 DiT 无依赖的阶段**（TE 计算 vs `dit_host_stage` 读盘、下一步权预取）。那是 host_stage 那条线，不是再写一套 DualPipe。

### 1.4 数值（FP8 / block-scale）

| # | 技术 | 做什么 | 墙 | 开源 | 依赖 |
| --- | --- | --- | --- | --- | --- |
| N1 | **FP8 混合精度训练** | 线性 Fprop/Dgrad/Wgrad 走 FP8；embedding / 输出头 / MoE gate / norm / **attention 保持 BF16/FP32**。主权重、梯度、优化器主副本较高精度 | 671B 用 BF16 算不动、装不下 | 论文 §3.3 | Hopper FP8 Tensor Core。他们自己修累加精度 |
| N2 | **细粒度 scale** | 激活 **1×128 tile**；权 **128×128 block**。在线算 max（不用 delayed/历史 max）。全用 E4M3（不用 E5M2 反向） | 张量级 scale 被 outlier 打爆 | 论文 §3.3.2；DeepGEMM 要求 TMA-aligned scale（SM90=FP32 SF，SM100=packed UE8M0） | 非标准 FP8 GEMM，要自研核 |
| N3 | **WGMMA 间隔提升到 CUDA core** | Tensor Core 累加约 14 bit；每 \(N_C=128\) 把部分和拷到 FP32 CUDA core，顺带乘 group scale | H800 FP8 MMA 精度墙（论文测 K=4096 相对误差可到 ~2%） | 论文 §3.3.2 / §3.5.2 | Hopper 缺陷的软件补。Blackwell 原生 microscaling 才是他们点名的下一代 |
| N4 | **低精度存储/通信** | 激活缓存 FP8；Adam 一/二阶 BF16；MoE dispatch FP8、combine BF16。attn 后进入线性的激活用定制 E5M6 | 显存 + A2A 带宽 | 论文 §3.3.3 | 训练图 |
| N5 | **推理对齐训练精度** | Day 6：**GEMM + dispatch = FP8；MLA + combine = BF16**。56.3% 输入 token 打中盘上 KV cache | 服务精度和训练一致，同时吃 FP8 吞吐 | Day 6 | H800 集群 + 3FS |

**和本仓 NVFP4 / W4A8 的关系（不要写成「他们也是 4-bit」）：**

| | DeepSeek FP8 | 本仓金线 INT8 ConvRot | 本仓 NVFP4 | 本仓 W4A8（Kijai） |
| --- | --- | --- | --- | --- |
| 角色 | 训练+推理主 GEMM | **DiT 金线** | TE 已用；DiT 加速档社区权 | 禁止主题默认 |
| 粒度 | 1×128 / 128×128 | ConvRot 后 INT8 | 分组 FP8 scale（`h3_nvfp4.h`） | 4-bit 权 + 8-bit 激活 |
| 热路径 | **原生** FP8 Tensor Core（DeepGEMM） | 常驻 INT8 GEMM | TE/DiT 现状多是 **dequant → F32/BF16/INT8 GEMM** | dequant 或混合核 |
| 可搬思想 | 细粒度 scale + **别解包再乘** | 已落地 | P1：sm_120 原生 FP4 GEMM（CUTLASS/cuBLASLt），不是再写一套 Hopper DeepGEMM | 不升默认 |

DeepGEMM 现在有 FP8×FP4 / FP4 indexer，但 arch 门是 **SM90/SM100**。5090 = **sm_120**。仓库不能当二进制丢进来。思想：权已经是 NVFP4 就不要再解成 FP32 乘。

### 1.5 并行策略

| # | 技术 | 做什么 | 墙 | 开源 | 和 DiT 是否同类 |
| --- | --- | --- | --- | --- | --- |
| A1 | **专家并行 EP** | 训练 EP64 / 8 节点；推理 prefill EP32（4 节点）、decode EP144–320（论文/Day6 数字随部署变）。冗余专家 + 周期重排 | 256 专家只激 8 个，要靠超大全局 batch 把每专家 batch 喂饱 | 论文 §3.2 / §3.4；EPLB | **不是。** 我们没有路由专家。INT8 单卡已装下，没有「切权才能装」的理由 |
| A2 | **MLA 压 KV** | 每 token 只缓存 \(c^{KV}\) + RoPE 解耦 key，不是整头 K/V。decode 显存从 MHA 量级压下去 | 自回归 KV 随长度线性涨 | 论文 §2.1；FlashMLA | **不是同类。** DiT 每步对当前 packed token（cafe ~1.6 万）做**双向** SDPA，下一步噪声图重算 QKV，**没有跨步增长的 KV cache**。没有「吸收 \(W^{UK}\) 进 Q」的 decode 循环 |
| A3 | **MLA absorb** | 结合律：decode 不先升满 K/V，而是 `q_nope @ w_kc`、`attn_out @ w_vc`。SGLang 默认开；还有 FP8 BMM absorb | decode 升维再 attn 又算又占带宽 | SGLang 文档 [Weight Absorption](https://github.com/sgl-project/sglang/blob/main/docs/basic_usage/deepseek_v3.md)；官方 Day0 推荐 SGLang | **搬不来形状。** 可搬的只有「能乘进去的矩阵就别物化」——对本仓更像「A2A 前先 BF16/FP8 pack」，不是改 cuDNN SDPA |
| A4 | **MTP** | 训练多深度 next-token 辅助损失。推理：**可以扔掉** MTP 模块；也可以拿来做 speculative decode（SGLang nextn / vLLM DeepSeek MTP） | 训练信号稀、decode 逐 token | 论文 §2.2 / §2.5；vLLM `deepseek_mtp.py` | **不是。** 视频 DiT 没有「草稿 token → 大模型一次校验」。少步蒸馏（turbo4）才是我们的步数刀 |
| A5 | **EPLB** | 按在线统计复制高负载专家、节点内重排，减 dispatch 接收倾斜 | 热专家把一张卡打满、别的卡空转 | [deepseek-ai/eplb](https://github.com/deepseek-ai/eplb) | 无专家，无此墙 |
| A6 | **冗余专家 / 动态路由** | Prefill 每卡 8+1 冗余；他们还在探「每卡多放、每步只激 9 个」 | 负载随请求变 | 论文 §3.4.1 | 无 |
| A7 | **明确不做** | 训练 **不 drop token**。训练 **不用 TP**。推理 attn 用小 TP4+SP，MoE 用大 EP——SGLang 后来也写 MLA 模型优先 DP attention 而不是纯 TP | TP 通信在 PCIe/跨节点更亏 | 论文 §3.2、§3.4 | 与本仓「不要把 TP 当 5090 主轴」一致（lightning-survey / driver-options P2.1） |

### 1.6 推理栈（V3/R1、SGLang、vLLM）

| # | 技术 | 做什么 | 开源 / 位置 | 依赖 |
| --- | --- | --- | --- | --- |
| I1 | **PD 分离 + 跨节点 EP** | Prefill 与 decode 不同并行度、不同重叠配方；靠 EP 把全局 batch 做大 | 论文 §3.4；Day 6 | 多机 H800 + IB |
| I2 | **官方推 SGLang** | [DeepSeek-V3 README](https://github.com/deepseek-ai/DeepSeek-V3) Day0 推荐 SGLang。SGLang：MLA absorb、FA3/FlashMLA/FlashInfer/CutlassMLA/TRT-LLM MLA、W8A8 FP8、KV FP8、DeepGEMM 集成、MTP nextn | [sglang#2591](https://github.com/sgl-project/sglang/issues/2591) 路线图（FP8 GEMM tune / CUTLASS / FlashMLA / DeepGEMM） | Python 服务栈 |
| I3 | **vLLM** | DeepSeek MTP 模块、FP8 CUTLASS、MLA 后端 | vLLM `deepseek_mtp.py` 等 | 同上 |
| I4 | **盘上 KV cache** | Day 6：24h 内 56.3% 输入 token 打中 on-disk KV；3FS 单客户端 KV lookup >40 GiB/s | 3FS + Day 6 | 集群文件系统 |
| I5 | **夜间缩容** | 高峰全节点推理、低谷把卡还给训练 | Day 6 | 多租集群 |

对本仓：没有 serving batch、没有 KV cache、没有 PD 分离。TE cache 已经是「别重复算同一句 prompt」的同构刀，而且 **已测无损**。不要再接一套 SGLang/vLLM。

### 1.7 系统（H800 被砍之后）

| # | 技术 | 做什么 | 墙 | 开源 | 依赖 |
| --- | --- | --- | --- | --- | --- |
| S1 | **节点内 NVLink 怎么用** | 专家落在同节点不同卡：IB 到「同号 GPU」后 **立刻 NVLink 转发**，不在 IB 落地再等。通信核 warp 按负载动态切发送/转发/接收 | 只用 IB 或只用 NVLink 都会空一半带宽 | 论文 §3.2.2 | NVSwitch 全连接 |
| S2 | **节点间软件路由** | 限 4 节点、组限制门控（group-limited gating），让 IB 域和 NVLink 域不对称转发匹配 | 出口管制后的 H800 互联档位 | 论文 §3.2；集群描述 | IB + NVLink，**不是**把 GeForce `canAccessPeer` 拧开 |
| S3 | **3FS / Smallpond** | 训练数据、ckpt、推理 KV 的并行 FS；RDMA + SSD | 存储带宽成为第二墙 | [3FS](https://github.com/deepseek-ai/3FS) | 多节点 RDMA 存储 |
| S4 | **HAI-LLM** | 自研训练框架（论文 §3.2）。推理引擎他们写过「准备开源」索引页，主体仍是上述库 + SGLang | 通用框架叠 PP+EP+FP8+DualPipe 太重 | 论文；[open-infra-index 202504](https://github.com/deepseek-ai/open-infra-index) | 内部栈 |
| S5 | **官方模块参数** | IBGDA 要的 `NVreg_*` / GDRCopy `gdrdrv` | GPU 直接发 IB | DeepEP nvshmem.md | **官方驱动开关**。和 aikitoria 改 `nvidia.ko` 开消费卡 P2P 不是一类事（§8 已写，此处不展开） |

### 1.8 他们明确说「没做」或只适用于训练

| 项 | 原文位置 | 含义 | 对本仓 |
| --- | --- | --- | --- |
| 训练不用 TP | 论文 §3.2 | 内存抠完就避免 TP | 不要为 480p 再加 TP all-reduce |
| 不 drop token | §2 / §3 | 负载均衡够好就不丢 | 无关 |
| DualPipe 仅训练 | DualPipe README；推理改用双 micro-batch | 不要把 fwd/bwd 调度抄进 generate | 抄「重叠独立阶段」，不抄双向 PP |
| MTP 推理可丢 | 论文 §2.5 | MTP 首先是训练目标 | 不要做视频 MTP |
| Attention 不走 FP8 | §3.3.1；Day 6 MLA=BF16 | 数值敏感算子留高精度 | 金线 SDPA 继续 cuDNN/BF16 合同，不要为学 FP8 改注意力核 |
| Combine 不走 FP8 | §3.3.3 | 聚合精度 | A2A 若降精度，先 BF16，不要第一步就 FP8 |
| 不用 delayed quant | §3.3.2 | 在线 per-tile max | 量化策略细节，不是我们的墙 |
| 20 SM 是妥协 | §3.5 | 他们想卸掉，不是炫技 | 5090 **不要留 20 SM** |
| 没改 nvidia.ko | 全仓库 / §8 | 用户态 + 官方 regkey | 不要学「改驱动」 |
| DeepGEMM/FlashMLA 不支持 SM120 | 两仓 Requirements | 不能当 5090 即插库 | 思想可学，二进制不能搬 |
| V1 低延迟核「完全不占 SM」有条件 | DeepEP #179 | 只在 hook 发出之后；发指令那 ~us 仍占 | 不要写「通信 0 成本」 |

---

## 2. 可借鉴对照表（P0 / P1 / P2）

对照已有刀：去 DeviceSync、BF16 A2A、host_stage 双缓冲、cuDNN SDPA vs 手写、NVFP4 dequant vs 原生 FP4、TE cache。

秒数是 cafe 864×480 5s **回代**，不是本轮新测。U=2 相对金线 DiT 31.67 s。

### 2.1 可直接借鉴（P0）——现有 C/CUDA 能做，且打中已测瓶颈

| # | 搬什么（DeepSeek 原句思想） | 改哪 | 打哪一段秒数 | 风险 |
| --- | --- | --- | --- | --- |
| **P0.1** | dispatch 用低精度砍 A2A 字节（他们 FP8 dispatch / BF16 combine） | `h3_ulysses_comm.cu`：放开 `dtype_bytes==4`，pack 前 F32→BF16，unpack 升回。只开 U=2 实验臂 | U=2 通信体积约 184 GB/卡 → ~92 GB。回代 **DiT ~25–28 s**（通信 ~7–8 + 计算 15.8 + 宿主 ~2）。**不改驱动时首次有希望快于 31.67。** e2e 仍约 72–74 s（TE+VAE 没切）。U=8 642/2 GB 仍可能 50–80 s，**不做** | **非位级等价**，另开 MAE 门。不要第一步 FP8 |
| **P0.2** | 用 event / hook，不要把整机 `cudaDeviceSynchronize` 当通信完成（他们 `EventOverlap`，不用 DeviceSync） | `h3_dit_w4a8.c`：`uly_sync_rank` 去掉每相 DeviceSync；A2A 前 per-rank event。A2A 已是 event，不要改回 `StreamSynchronize`。`uly_h2d_aux` 改 async | 单独 **~32–33 s**（不够翻正）。与 P0.1 叠才有 25–28。打的是宿主空转，不是 184 GB | 曾 IMA；保持 per-device cuDNN plan。Ulysses 语义仍须 A2A 等齐，**不要写 overlap 后 U=2≈16 s** |
| **P0.3** | 「能缓存的中间量不要重算」（Day 6 盘上 KV 56% hit；3FS） | **已经落地：** `--te-cache` / `H3_TE_PROMPT_CACHE`。保持 opt-in，金线默认关 | 已测：TE HIT 金线 e2e **61.41**；turbo4+TE HIT **~49.7–50 s**。打的是 TE ~15 s，HIT 后 host_stage 会记进 DiT（~9–13 s），不要用 HIT 的 DiT 和 31.67 横比 | 脏 key 前科（`63ba422170cf015a`）。cache 目录与评片隔离 |
| **P0.4** | 8 卡正当用法是 **DP 吞吐**，不是单请求 EP/SP（Day 6 的 DP attention + 我们已有结论） | 调度层：8 路各跑完整 generate，卡间零 A2A。不改 Ulysses 核 | 单条仍 ~79 / ~64 / ~50；聚合 **×8 吞吐**。这是 `peer=0` 上唯一稳的 8 卡 KPI | 单请求延迟不变。产品先选 KPI |

**P0 不做：** 留 20 SM、接 DeepEP、为 U=8 赌 P2P、动 `h3_gpu_cuda_sdpa_cudnn.cpp` 换 FlashMLA。

### 2.2 可借鉴但要改形状（P1）

| # | 思想 | 为什么不能照搬 | 本仓落点 | 谁在做（别抢） |
| --- | --- | --- | --- | --- |
| P1.1 | **重叠独立阶段**（推理双 micro-batch，不是 DualPipe） | MoE dispatch 可和「另一个 batch 的 attn」并行；Ulysses A2A 和本步 SDPA 不能 | 重叠的是 **TE ↔ `dit_host_stage`（disk→host）**、步间权预取、将来 TAE 与 mux。双缓冲，不要 DeviceSync 排干 | **host_stage 已在另路。** 本文不改 `tools/h3_cuda_generate.c` |
| P1.2 | **细粒度量化 + 原生低精度 Tensor Core**（DeepGEMM，不解包） | 他们的核是 SM90/SM100 FP8 WGMMA；我们是 sm_120 NVFP4 | TE/DiT NVFP4 热路径从 dequant→GEMM 换成 **原生 FP4 MMA**（CUTLASS / cuBLASLt），scale 合同对齐 `h3_nvfp4.h`。预期：社区 s/it 约 1.14× 量级，金线 DiT 31.67→~28 **远小于** turbo4 已拿到的 16.47；价值是与少步相乘、8 卡每卡一份权更松 | **NVFP4 已在另路**（`h3_dit_nvfp4.c` / `h3_w4a8_cache.c`）。本文不改 |
| P1.3 | **通信核少干扰计算**（PTX cache hint、chunk 调谐、少占 SM） | 他们有常驻 NVLink 通信核 | 我们最多：NCCL 路径少同步、BF16 pack、同 NUMA U=2。不要手写 Hopper UB PTX 到 5090 | 跟 P0.1/P0.2 走 |
| P1.4 | **注意力手写核只在形状特殊时才值** | FlashMLA 是为 MLA decode / 稀疏 top-k。本仓 packed 双向 SDPA，金线 cuDNN 已建成 U=1 heads=56、U=2 heads=28 | 保持 `h3_gpu_cuda_sdpa_cudnn.cpp`。手写/Sage 只当加速档、且已有测评纪律。不要为「学 FlashMLA」重写 SDPA | 不新开 |
| P1.5 | **能融合的小核就融**（V4.1 fused norm-RoPE-attn-cast） | 他们融的是 Q_b / Wv 排列后的 MLA 小核串 | 我们若融，对象是 **dequant+GEMM** 或 host_stage 拷贝，不是 RoPE-MLA | NVFP4 / W4A8 dequant 另路（`h3_w4a8_dequant.c`） |
| P1.6 | **减集体次数** | 他们靠门控限制跨节点跳数 | Ulysses：只在 attn 边界 A2A；MLP 不要多余相同步。不要把 QKV / attn-out 两次 A2A 合成一次除非证明等价 | `h3_dit_w4a8.c` `layers_forward_ulysses`，实验臂 |

### 2.3 学了也搬不来（P2）

| # | 技术 | 卡在哪 |
| --- | --- | --- |
| P2.1 | DeepEP / NVSHMEM / IBGDA / NCCL Gin / Mega MoE | 要 NVLink 或 RDMA A2A。本机 `peer=0`，Ulysses 走 SHM。MoE 形状不对 |
| P2.2 | DualPipe / DualPipeV | 训练 fwd+bwd。单请求 8-step DiT 没有微批管道 |
| P2.3 | 20 SM 通信常驻、IB→NVLink 4 节点路由 | 没有 NVLink，没有跨节点专家 |
| P2.4 | EP / EPLB / 冗余专家 / PD 分离 / 跨节点 EP144 | 不是 MoE serving；480p 单卡装得下 |
| P2.5 | MLA / absorb / MTP / FlashMLA / paged KV | DiT 无增长 KV、无草稿 token。FlashMLA 无 SM120 |
| P2.6 | DeepGEMM 二进制、Hopper WGMMA 精度补丁 | arch 是 SM90/100，不是 5090 sm_120 |
| P2.7 | 3FS | 我们的墙是单机 `dit_host_stage` ~13 s，不是 180 节点 6.6 TiB/s |
| P2.8 | 巨型 decode batch（靠 EP 把每专家 batch 喂到几百） | 单条 cafe，没有 serving batch |
| P2.9 | 训练不用但推理用的小 TP4 | PCIe 上 per-block all-reduce 比 A2A 更亏（RH 自己测过） |

### 2.4 不要学

| 项 | 为什么 |
| --- | --- |
| 改 `nvidia.ko` / GSP / aikitoria「因为 DeepSeek 也改驱动」 | 他们没改。见 driver-options §8。一句带过。 |
| 为单请求 U=8 赌 P2P | 已测 137 s。软件刀救不回 642 GB host bounce。 |
| 把 20 SM 划给 NCCL | 5090 计算 SM 更金贵，且没有可重叠的 NVLink 通信核。 |
| 用 FlashMLA 替换 cuDNN SDPA | 形状不是 MLA；arch 不是 SM90。 |
| 把 DualPipe 画进 generate | 没有反向，没有 PP rank。 |

---

## 3. 建议落地顺序（只写不抢文件的刀）

host_stage / TAE / NVFP4 继续走现有分支。下面是**本清单建议的顺序**，3–6 条，具体到模块。

1. **U=2 BF16 A2A（P0.1）** — `h3_ulysses_comm.cu`（`dtype_bytes != 4` 硬拒处、pack/unpack）。只跑同 NUMA GPU0,1、cafe warm，和金线 DiT 31.67 比。过门（DiT ≤ 28 s + 另阈 MAE）当加速档开关；不过门关掉。**不要开 U=8。**
2. **去掉每相 `uly_sync_rank` DeviceSync（P0.2）** — `h3_dit_w4a8.c` 的 `uly_sync_rank` / `uly_run_ranks`。与 (1) 同一实验臂。A2A 保持 stream event。预期单独不够翻正，是 (1) 的配套。
3. **e2e 继续用已测配方，不靠 Ulysses** — turbo4 + `--te-cache`（HIT ~50 s）。金线默认 U=1、cache 关。这是当前单请求诚实上限，不是 5–8 s。
4. **8 卡 KPI 改口为请求级 DP（P0.4）** — 调度，不碰通信核。吞吐 ×8。
5. **不要新开：** DeepEP 移植、FlashMLA、DualPipe、`h3_gpu_cuda_sdpa_cudnn.cpp` 重写、为 P2P 动驱动。
6. **留给已有另路（只对齐思想，不在本清单改文件）：** `tools/h3_cuda_generate.c` 的 `dit_host_stage` 双缓冲 ≈ 推理双 batch 重叠；`h3_dit_nvfp4.c` / 将来 sm_120 FP4 GEMM ≈ DeepGEMM「别 dequant 再乘」。

U=2 软件刀做完：DiT 最好 25–27 s，e2e 仍被 TE+VAE+host 托在 ~70 s 档。要 50 s 以下走 (3)，不走 U=8。

---

## 4. 和已有刀的一张对照（防以后再争论）

| 本仓已有 / 已规划 | DeepSeek 对应物 | 结论 |
| --- | --- | --- |
| 去 DeviceSync | `EventOverlap` / recv hook | **学。** P0.2 |
| BF16 A2A | FP8 dispatch + BF16 combine | **学形状，降一档精度。** P0.1。他们能 FP8 是因为 GEMM 也是 FP8 |
| host_stage 双缓冲 | 推理双 micro-batch，不是 DualPipe | **学思想。** 另路，别抢 `h3_cuda_generate.c` |
| cuDNN SDPA vs 手写 | FlashMLA | **不换。** 他们手写是因为 MLA；我们形状 cuDNN 已经吃 | 
| NVFP4 dequant vs 原生 FP4 | DeepGEMM 原生 FP8/FP4 | **学思想，另路实现。** 不要 git submodule DeepGEMM 当 5090 后端 |
| TE cache | 盘上 KV cache + 3FS | **已经在打同一类墙。** 保持 |
| Ulysses U=2/8 | DeepEP A2A / 节点内 NVLink | **公路不同。** 算法对、5090 上慢。软件只救 U=2 一小截 |

---

## 5. 一句话

DeepSeek 的课是用户态把通信和 kernel 抠到极限（字节、重叠、占用、精度合同），不是改驱动。

---

## 附录 A. 一手链接

| 源 | 用途 |
| --- | --- |
| [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) §3.2–3.5、§2 MLA/MTP | 训练框架、通信核、FP8、推理部署、硬件建议 |
| [Day 6 推理概览](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md) | PD 分离、双 batch 重叠、负载均衡、FP8/BF16 合同、KV hit |
| [DeepEP](https://github.com/deepseek-ai/DeepEP) + [legacy.md](https://github.com/deepseek-ai/DeepEP/blob/main/docs/legacy.md) + [issue #179](https://github.com/deepseek-ai/DeepEP/issues/179) | normal vs 低延迟、SM 占用、PTX |
| [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | JIT GEMM、arch 门、Mega MoE |
| [FlashMLA](https://github.com/deepseek-ai/FlashMLA) | MLA 核、SM 矩阵、NVIDIA PR |
| [DualPipe](https://github.com/deepseek-ai/DualPipe) | 训练 PP；DualPipeV 来自 SAIL |
| [EPLB](https://github.com/deepseek-ai/eplb) · [3FS](https://github.com/deepseek-ai/3FS) · [profile-data](https://github.com/deepseek-ai/profile-data) | 负载、存储、重叠剖面 |
| [SGLang DeepSeek 文档](https://github.com/sgl-project/sglang/blob/main/docs/basic_usage/deepseek_v3.md) | MLA absorb、官方推荐引擎 |

二手索引（不当事实源）：SemiAnalysis DeepSeek Debates；Tom's Hardware「bypasses CUDA」标题（已证伪，§8）。

## 附录 B. 本仓锚（只读，未改）

```468:469:h3_ulysses_comm.cu
    if (dtype_bytes != 4)
        return comm_fail(err, err_size, "ulysses a2a_qkv: only f32 in P1");
```

`h3_dit_w4a8.c`：`uly_sync_rank` → `cudaDeviceSynchronize`；`uly_run_ranks` 全 rank enqueue 后再逐 rank sync。

`tools/h3_cuda_generate.c`：`dit_host_stage_th` / `kick_dit_host_stage`（另路）。

`h3_gpu_cuda_sdpa_cudnn.cpp`：金线 SDPA。

`h3_dit_nvfp4.c` / `h3_w4a8_dequant.c`：NVFP4 / 解包（另路）。

密码未入库。未改推理代码，未 commit。
