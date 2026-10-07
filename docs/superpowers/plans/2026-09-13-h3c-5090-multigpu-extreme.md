# 8×RTX 5090 多卡「极致」路线（社区对照 + 已测回代）

> **已被完整方案取代（2026-09-13）：** 施工图改读 [2026-09-13-h3c-5090-extreme-full-plan.md](./2026-09-13-h3c-5090-extreme-full-plan.md)。本文保留作文献对照，不再当施工单。

> **For agentic workers:** 本文是战略/回代文档，不是施工单。不要装社区内核，不要改推理代码，不要 commit。若以后要落地软件刀，再用 superpowers:subagent-driven-development 或 executing-plans 按任务拆。
>
> **日期：** 2026-09-13。联网检索 2025–2026 新材料后写。密码未入库。未 SSH 装驱动。拓扑沿用 [ulysses-driver-options 附录 A](./2026-09-13-h3c-5090-ulysses-driver-options.md)（2026-09-13 07:47 UTC 只读）。
>
> **Goal:** 在不推翻已钉死事实的前提下，把「多卡 5090 性能」拆成延迟 KPI 与吞吐 KPI，并对照社区新方案给出极致上限。
>
> **Architecture:** 公路仍是 PCIe + host bounce（`peer_access=0`）。单请求延迟靠少字节通信 / 少步 / 少 decode；8 卡吞吐靠请求级 DP。社区内核只当副机可选实验。
>
> **Tech Stack:** 本仓 C/CUDA；对照 xDiT / LightX2V / FastVideo / Cache-DiT / vLLM / 开源内核 fork。
>
> **Spec:** 已测尺子见 [lightning-survey](./2026-09-13-h3c-minimax-lightning-survey.md)、[driver-options](./2026-09-13-h3c-5090-ulysses-driver-options.md)、[deepseek-opt-lessons](./2026-09-13-h3c-deepseek-opt-lessons.md)、[plan-results-audit](./2026-09-13-h3c-plan-results-audit.md)。本文不复述那三份的论证，只引用秒数。
>
> **钉死（无反证不推翻）：** 8×5090，PCIe，`peer_access=0`，无 NVLink。U=1 DiT **31.67 s**；U=2 **34.66 s**；U=8 **137 s**。官方 50 系 GeForce 不支持 GPU↔GPU P2P。DeepSeek 主力刀不适用于这台箱。RunningHub 12×几乎是减步，卡能 P2P。

**一句话：** 极致不是再拧 U=8，也不是生产箱刷社区 `nvidia.ko`。延迟极致 = 单卡配方（turbo4+TE+TAE）叠 U=2 **低精度 A2A**（学 Cache-DiT / LightX2V，不是学 DeepSeek 改驱动）；吞吐极致 = 8 路请求级 DP。2026 年 NVIDIA 态度没有变。

---

## Global Constraints

- 8× GeForce RTX 5090（`10de:2b85`），驱动 **595.45.04** Open Kernel Module，`topo -p2p r` 全 **CNS**。
- 金线默认 U=1、8-step INT8、`cfg=1`、官方 VAE。加速档可以改并行，不能把社区内核写进生产路径。
- 本文秒数是 cafe 864×480 5s **回代**，不是本轮新测。
- 不 commit、不改推理代码、不在远端装补丁。

---

## 0. 先答：必须上社区内核吗？

**不必，也不该当第一刀。**

2026 年没有找到官方态度变化。NVIDIA 员工 Yuki_Ni 在 [论坛 326776](https://forums.developer.nvidia.com/t/p2p-issue-using-two-rtx-5090-gpus/326776/8)（**2025-03-13**）写明 50 系 GeForce **不支持 GPU↔GPU P2P**；产品页仍标 [NVLink (SLI-Ready): No](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)（现行）。NCCL 维护者在 [#1637](https://github.com/NVIDIA/nccl/issues/1637)（2025-03，2026 仍有 5090 跟帖）重复：P2P **不是 NCCL 功能**，**recent GeForce 不支持**；2.26+ 只修 SM120 SHM 上限。

社区 *能* 把 `canAccessPeer` 拧到 1，但：

| 2026 新实锤 | 含义 |
| --- | --- |
| [vLLM 论坛 2026-08-11](https://discuss.vllm.ai/t/dual-rtx-5090-tp-2-shm-vs-patched-bar1-p2p-cumem-single-pass-2-7-mean-point-estimate/2870)：2×5090 + aikitoria 风格 610.43.03，P2P **OK**，`~16 MiB` 带宽仅 **~14.1 GB/s**（该机 host 路径 ~1.9 GB/s）；vLLM TP=2 相对 SHM 只 **+2.7%**（有一格 −1.9%） | 即使门开了，**不是每台 5090 都有 aikitoria README 的 55 GB/s**。LLM 小集体几乎吃不满 PCIe P2P。 |
| [llama.cpp PR #21910](https://github.com/ggml-org/llama.cpp/pull/21910)（2026）：`canAccessPeer=1` 在部分主板/BIOS 上 **崩溃或静默坏输出**，改成 **显式 opt-in** | `can=1` ≠ 拷贝正确。 |
| aikitoria issue #9（2025-10）：5090 补丁后 BAR 分配失败 / GSP 起不来 | 生产 8 卡箱已评估过，**有风险，没试**。 |

我们的 Ulysses 载荷是 **184 GB/卡（U=2）/ 642 GB 总线（U=8）**，比 vLLM decode all-reduce 大两个数量级，所以 **若** 副机真测到 40–55 GB/s，U=2 仍可能从 34.66 掉到 ~20–22 s。那是可选上限，不是「必须刷内核才能做极致」。

**极致的第一刀在用户态。** 社区 2026 年真正新增、且点名「没 NVLink」的，是 **压通信体积**（Cache-DiT `ulysses_float8`、LightX2V `use_fp8_comm` / `ulysses-4090`），不是再 fork 一份 `nvidia.ko`。

---

## 1. 社区方案地图（2025–2026，带链接）

每行：**适用 5090 / 仅数据中心 / 仅 Python 框架**。点名 5090 单独标。

### 1.1 消费卡多卡视频 / DiT

| 方案 | 日期 / 源 | 点名 5090？ | 切法 | 对本仓 | 桶 |
| --- | --- | --- | --- | --- | --- |
| **xDiT / xfuser USP** | 论文 [arXiv:2411.01738](https://arxiv.org/html/2411.01738v1)（2024-11）；runner 现行支持 MiniMax-H3（[runner.md](https://github.com/xdit-project/xDiT/blob/main/docs/runner/runner.md)） | 公开表主写 H100/H20/L40。Step-Video 文档一句「Consumer GPUs (5090/5090D)」是 **训练装得下**，不是 8 卡单请求赢单卡 | Ulysses × Ring × **PipeFusion** × CFG × DP | PipeFusion = **按层切 + 步间陈旧激活**，不是每层 A2A。CFG 并行我们 `cfg=1` 无第二条臂。不要把 Python USP 嵌进 C 栈 | **仅 Python 框架**；算法思想可搬 |
| **PipeFusion** | [arXiv:2405.14430](https://arxiv.org/html/2405.14430)（2024-05；v2 HTML 现行） | 实验是 **8×L40 PCIe Gen4**，没点 5090 | 层流水 + patch；通信是 **步间激活 P2P**，不是每层 A2A | 8-step INT8 上「陈旧 KV」质量未知；实现量大。480p 流水气泡可能比 A2A 还亏 | 思想：**不靠 P2P 的切法**候选；**仅 Python** |
| **Wan2.1 官方** | [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) README（现行） | 否 | FSDP + `--ulysses_size` / `--ring_size` | 为 14B **装得下 + 长序列**。480p 单卡 INT8 已常驻，FSDP 是答错题 | **仅 Python** |
| **Raylight**（Comfy + xDiT） | [komikndr/raylight](https://github.com/komikndr/raylight) README（现行） | **是。** 表：Wan 2.1 14B，Ulysses=2，**5090** 3.05 s/it vs 双 2000 ADA 5.18；FSDP 只救小显存卡 | USP + CFG 并行 + FSDP | 自己写 USP「really want a very fast connection」。5090 数字是 **单卡 s/it**，不是 8 卡相对 1 卡。Comfy 工作流，不能当 C 后端 | **适用 5090（Python/Comfy）** |
| **ComfyUI-MultiGPU / DisTorch** | [pollockjj/ComfyUI-MultiGPU](https://github.com/pollockjj/ComfyUI-MultiGPU)；Reddit [2025-01](https://www.reddit.com/r/comfyui/comments/1ic0mzt/comfyui_gguf_and_multigpu_making_your_unet_a_2net/) | 否（讲 4090/FLUX/Hunyuan） | 权/VAE **捐赠显存**、Block Swap | **不降单请求墙钟**。我们 32 GB 已够 | 显存刀，不是延迟刀 |
| **TeaCache** | [welltop-cn/ComfyUI-TeaCache](https://github.com/welltop-cn/ComfyUI-TeaCache)（2025-01–03 接 Hunyuan/Cog/Wan） | 否 | 步间特征缓存，宣称 1.5–2× | 和 TE cache 同类墙，打的是 **重算** 不是 A2A。8-step 上跳步窗口窄 | 思想可搬；**仅 Python** |
| **MagCache / AdaCache / TaylorSeer** | LightX2V [feature caching](https://deepwiki.com/ModelTC/lightx2v/9.4-feature-caching-strategies)（现行） | 框架另文点 5090（见下） | 幅度/自适应跳块 | 加速档实验臂。金线禁止 | **仅 Python**；思想 = 跳块 |
| **Cache-DiT**（vipshop）+ **SGLang Diffusion** | [SGLang cache_dit 文档](https://docs.sglang.io/docs/sglang-diffusion/cache_dit)（现行）；并行 PR [sglang#15163](https://github.com/sgl-project/sglang/pull/15163) | 文档列 **MiniMax-H3**。**明确写：没 NVLink 就开 `ulysses_float8`**。`ulysses_async` 只支持 FLUX/Qwen/Ovis/Z-Image，**不是 H3** | DBCache + TaylorSeer + SCM；Ulysses/Ring/TP；FP8 A2A | **这是 2026 最该偷的用户态刀。** SCM 要求 ≥8 step，H3 金线 8-step 卡在门槛上；turbo4=4 会自动关 SCM | **仅 Python 框架**；FP8 A2A 思想 **适用 5090** |
| **LightX2V** | 并行文档 [parallel.html](https://lightx2v-en.readthedocs.io/en/latest/method_tutorials/parallel.html)；[LightX2V-BLOG](https://github.com/ModelTC/LightX2V-BLOG/commit/ec1ca6415ba16315e2c85f388d8b10dc713e1f9d)；核 `ulysses-4090`（[ulysses_attn.py](https://github.com/ModelTC/lightx2v/blob/836e6f65/lightx2v/common/ops/attn/ulysses_attn.py)）；SM120 MXFP8 PR [#1090](https://github.com/ModelTC/LightX2V/pull/1090) | **是。** 博文：5090 上 Hunyuan 1.5-8B I2V 480p + CFG，宣称消费卡互连仍有并行效率；相对其他开源框架 5090 上 **2.73×**（这是框架对比，**不是 8 卡÷1 卡**）。PR #1090：5090 fused FFN | Ulysses / Ring / CFG；**`ulysses-4090` = 手工 send/recv 替 NCCL A2A**；`use_fp8_comm` / **`use_fp4_comm`** | U=2 软件刀的社区原形。CFG 并行我们没有。不要整仓 Python 进来 | **适用 5090（Python）**；通信配方可搬进 C |
| **FastVideo / FastWan-QAD / Attn-QAT** | 博文 [2026-06-15](https://haoailab.com/blogs/fastwan-qad/)；权重 [FastWan-QAD-1.3B](https://huggingface.co/FastVideo/FastWan-QAD-1.3B)；论文 Attn-QAT [arXiv:2603.00040](https://arxiv.org/abs/2603.00040)（2026） | **是。点名单卡 RTX 5090：5s 480p e2e 1.78 s**（3-step + 原生 NVFP4 + SageAttention3 FP4 + TAEHV，无 CFG） | **几乎不靠多卡。** 序列并行是可选附件 | 1.78 s 是 **1.3B Wan 换配方**，不是 H3 8-step。课是：延迟极致在 **步数×原生低比特×小 VAE**，不在 U=8 | **适用 5090（单卡配方）**；**仅 Python** |
| **TAEHV / LightTAE** | FastWan 用 [madebyollin/taehv](https://github.com/madebyollin/taehv)；LightX2V Hunyuan LightTAE（2025-11 README） | FastWan 表在 5090 | 小 VAE 换官方 decode | 本仓 TAE 已有 C 路径（`h3_tae.c`），金线禁止成片。与多卡正交 | 思想已在另路 |
| **DistVAE / ParaVAE** | [xdit-project/DistVAE](https://github.com/xdit-project/DistVAE)；[RiseAI-Sys/ParaVAE](https://github.com/RiseAI-Sys/ParaVAE)（Wan2.1 3D VAE） | 否 | VAE patch/halo 并行，主要砍 **峰值显存** | cafe VAE ~10 s。480p 不是 8K OOM。8 卡切 VAE 最多砍这段，**动不了 DiT 31.67** | **仅 Python**；加速档可考虑 tile，不是主刀 |
| **FlowCache** | ICLR 2026，[mikeallen39/FlowCache](https://github.com/mikeallen39/FlowCache)（代码 2026-02） | 否（A800 / MAGI-1） | 自回归 chunk 缓存 | H3 是双向 8-step DiT，**不是** MAGI 自回归。搬不来形状 | 仅数据中心实验 / 错模型 |
| **MAGI-1 官方** | SandAI 8 卡预设 | 常见 H100×8 / 4090×8 | 为 1080p 装得下 | 不是 5090 480p 单请求证人 | 容量刀 |

**负结果（按要求搜过）：** 没有第二家在 **5090 + `peer=0`** 上公布「同一条 8-step 请求比单卡快」的可引用数字。Raylight / LightX2V / FastVideo 的 5090 数字，要么是单卡 s/it，要么是少步/量化配方，要么是框架横比。

### 1.2 5090 / Blackwell 多卡实锤（P2P 与带宽）

| 源 | 日期 | 点名 5090？ | 带宽 / 结论 |
| --- | --- | --- | --- |
| NVIDIA 产品页 | 现行 | 是 | **无 NVLink 桥。** 50 系对照表全列 NVLink=No（[compare](https://www.nvidia.com/en-us/geforce/graphics-cards/compare/)） |
| Yuki_Ni 论坛 326776 | 2025-03-13 | 是 | 官方：**GPU↔GPU P2P 不支持** |
| NCCL #1637 | 2025-03 → 2026 跟帖（驱动 595.71.05） | 是 | 官方栈走 **SHM**；`NCCL_P2P_DISABLE=1` 本来就没效果 |
| [aikitoria](https://github.com/aikitoria/open-gpu-kernel-modules) | README 末推 2026-07-13；`595.45.04-p2p` 有 5090 提交 | **是（8×5090 + PRO 6000，EPYC）** | 单向 P2P ≈ **55–56 GB/s**，双向 ≈111。P2P 关时该机 host 仍 ≈43 GB/s（**不是本机 ~10–12**） |
| [tinygrad 570.148.08-p2p](https://github.com/tinygrad/open-gpu-kernel-modules/tree/570.148.08-p2p) | 发布 **2025-06-06** | README 写 4090/5090 | 多卡 5090 上有人报不如 aikitoria fork（Reddit [2025-08-30](https://www.reddit.com/r/LocalLLaMA/comments/1n3qcqn/patched_p2p_nvidia_driver_now_works_with_multiple/)） |
| [QuixiAI](https://github.com/QuixiAI/open-gpu-kernel-modules) | 建仓 2026-08-26，驱动 610.57.04 | 文档写 5090 | **验证段是 8×3090**。自称 610 已有原生 BAR1 HAL，只是 GeForce 不选 |
| [ec-jt](https://github.com/ec-jt/open-gpu-kernel-modules) | 610.43.02-p2p | README 点 5090 | aikitoria 线再 fork |
| [Duanyll](https://github.com/Duanyll/open-gpu-kernel-modules) | `590.48.01-p2p` | 点 4090 **和 5090** | 另做 48GB 魔改 4090 的 BAR1<VRAM；**5090 默认 BAR 小** 是他们写的 caveat（本机 BAR1 已 32 GiB，这条对我们不是墙） |
| [kacper-daftcode/rtx-p2p](https://github.com/kacper-daftcode/rtx-p2p) | 宣称 590.x+，测 590.48.01 | **是（GB202）** | **不改内核**，`RMDisableFeatureDisablement` 反汇编 GSP。8 star。**比改 ko 更危险。排除。** |
| vLLM 论坛上文 | **2026-08-11** | 是 | 补丁后 P2P **14.1 GB/s**；TP=2 只 +2.7% |
| llama.cpp #21910 | 2026 | 泛 CUDA，不单 5090 | P2P 可坏输出 → opt-in |
| 2×5090 装机笔记 | [hotchpotch.dev](https://hotchpotch.dev/articles/rtx5090x2-pc/)（约 2025 年末） | 是 | 无 NVLink；PCIe 5.0 x8 约 20–30 GB/s 量级；训练弱相关任务才 ~1.8× |

**vLLM / SGLang / llama.cpp 都没有「官方驱动打开 5090 P2P」的补丁。** 他们只是：探测到 `can=1` 就走 NCCL P2P，否则 SHM。tinygrad 是内核 fork 的源头，不是推理引擎补丁。

### 1.3 不靠 P2P 的切法（2026 社区实际在用的）

| 切法 | 社区落点 | 通信次数 | 本仓能不能用 |
| --- | --- | --- | --- |
| **CFG 并行**（正负 prompt 两卡，每步一次 latent all-gather） | xDiT [cfg_parallel.md](https://github.com/xdit-project/xDiT/blob/main/docs/methods/cfg_parallel.md)；LightX2V `cfg_p_size`；Comfy Worksplit | **每步 1 次**，不是每层 | **`cfg=1`，零收益。** 不要做 |
| **PipeFusion / 按层 PP** | xDiT | 步间激活，可 overlap | 8-step 气泡大；要改层调度。P2 研究项，不是立刻做 |
| **Ring attention**（切 token，KV 绕环，无 Ulysses A2A） | Wan `--ring_size`；LightX2V `seq_p_attn_type=ring`；xDiT `ring_degree` | 每层 hop，可藏进 FA | 本仓 packed cuDNN SDPA **不是** FA 在线 softmax。重写核才能接。480p 仍可能亏 |
| **Ulysses 但压字节** | Cache-DiT `ulysses_float8`；LightX2V `use_fp8_comm` / `use_fp4_comm` | 仍每层 2×A2A，体积 /2 或 /4 | **立刻该偷。** 本仓已有 F32 硬拒（`h3_ulysses_comm.cu`）；头文件已写 `dtype_bytes=2` |
| **Ulysses 但手工 send/recv**（`ulysses-4090`） | LightX2V 专为消费卡 | 避开 NCCL A2A 启动税 | U=2 值得对照；U=8 642 GB 仍救不回 |
| **异步 Ulysses CP** | Cache-DiT `ulysses_async` | 宣称 overlap | **文档写明不支持 MiniMax-H3。** Ulysses 语义上 A2A 必须等齐才能 SDPA。不要写「overlap 后 U=2≈16 s」 |
| **请求级 DP** | xDiT `--data_parallel_degree`；所有 serving 栈的默认正确用法 | **零 A2A** | **8 卡唯一稳的极致。** |
| **pinned host / 双缓冲 bounce** | 通用 CUDA | 不改变体积 | 可削几个百分点。GDRCopy：**对 GeForce GPU↔GPU 无效**（它是 CPU 映射 GPU BAR，见 [GDRCopy](https://github.com/NVIDIA/gdrcopy)）；GPUDirect RDMA 文档写 **Tesla + Quadro**（[CUDA 13.3](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html)） |
| **NCCL Simple vs LL / LL128** | [ETH 综述](https://spcl.inf.ethz.ch/Publications/.pdf/hu-demystifying-nccl.pdf) | LL 吃不满大消息带宽；LL128 要 NVLink 原子 128B | 我们 A2A 是 **百 MB 级大消息** → 应走 **Simple**。不要为「低延迟协议」把 184 GB 换成 LL。`NCCL_PROTO=Simple` 最多当关书实验 |
| **NUMA 绑定** | 本机 GPU0–3 NODE / GPU4–7 SYS | U=2 已在同域仍亏 3 s | 保持。不要用跨 SYS 的 U=8「再证明该快」 |

### 1.4 吞吐 KPI vs 延迟 KPI（必须拆开）

| | 延迟极致（一条 cafe 墙钟） | 吞吐极致（8 卡视频/小时） |
| --- | --- | --- |
| 正确手段 | 单卡配方 + 最多 U=2 压字节 | **8 路完整 generate，卡间零通信** |
| 错误手段 | U=8 host bounce；把 RH 12×当除数 | 一条请求切 8 卡还说「8 卡都用上了」 |
| **MIG** | [RunPod 5090 页](https://www.runpod.io/articles/guides/nvidia-rtx-5090) / 工作站对照：**5090 无 MIG**。MIG 在 RTX PRO 6000 Blackwell | 不能靠 MIG 把一张 5090 切成多租户 |
| **MPS** | CUDA MPS 可减同卡多进程 launch 税，**无显存隔离** | 一张 5090 32 GB 跑一份 INT8 DiT 已紧。**不要** 2 作业/卡。8 卡 = 8 进程，每卡一个，**不用 MPS** |
| 连续 batch | 视频 generate 是一条长作业，不是 LLM decode token | 队列里叠 8 路独立请求即「连续」。不要接 vLLM scheduler |

社区「5090 极致」样板 **FastWan-QAD 1.78 s** 是 **单卡延迟 KPI**，用的是换模型+3-step+NVFP4+TAEHV。把它抄成「8 卡 H3 也该 1.8 s」是类别错误。

### 1.5 社区内核 / 驱动：除 aikitoria 还有谁？2026 NVIDIA 有无松口？

**没有松口。** 2026 检索仍是 2025-03 官方句 + NCCL「GeForce 不支持」+ 产品页 No NVLink。没有「某日官方 .run 让 5090 `can=1`」的实锤。

比 aikitoria 新或旁支（都是同一条 BAR1 强制选择，不是更安全的「只开 BAR1 官方开关」）：

| 仓 | 相对 aikitoria | 更安全？ | 生产箱 |
| --- | --- | --- | --- |
| tinygrad `570.148.08-p2p` | 源头；5090 多人改用 fork | 否 | 否 |
| QuixiAI 610.57.04 | 更少文件、自称走官方原生 BAR1 HAL | 相对「手建 PTE」干净一点；仍改 ko | 否（验证是 3090） |
| ec-jt 610.43.02 | 再发行 | 否 | 否 |
| Duanyll 590.48.01 | 旧驱动线 + 48G 4090 | 否 | 否；本机是 595 |
| rtx-p2p | **不改内核**，改 GSP 键 | **更不安全**（固件逆向 + EULA） | **排除** |

**没有找到「只开 BAR1、不动 P2P 选择、官方认可」的 2026 方案。** BAR1=32 GiB 本机已经有；墙在 SKU 门，不在 BAR 大小。

### 1.6 硬件 / 拓扑（核实）

| 问题 | 答案 | 源 |
| --- | --- | --- |
| 5090 有无官方 NVLink 桥？ | **没有。** 规格页 NVLink (SLI-Ready)=No。30 系才有桥 | NVIDIA 5090 页；[30 系 NVLink Bridge](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090-3090ti/) |
| PCIe 5.0 拓扑 | 本机双路 Xeon：GPU0–3 NUMA0 `NODE`，GPU4–7 NUMA1 `SYS`。跨域 A2A 更亏 | 2026-09-13 只读 topo |
| 同 root complex | U=2 已钉 GPU0,1。官方栈上同域仍亏 | 已测 |
| GPUDirect 对 GeForce | **GPUDirect RDMA（GPU↔NIC）名单是 Tesla/Quadro**。`nvidia-peermem` 不打开 5090↔5090 | [GPUDirect RDMA 13.3](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html) |
| 本机有 `mlx5_*` | 在 NUMA0。Ulysses **不走网卡**。DeepEP/IBGDA 用不上 | driver-options 附录 A |

---

## 2. 「极致」两条路（用已测分段回代）

尺子（cafe 864×480 5s，不推翻）：

| 段 | 金线 | turbo4 | 备注 |
| --- | ---: | ---: | --- |
| e2e | **78.74 s** | ~64 s | turbo4+TE HIT **~49.7–50 s** |
| DiT | **31.67 s** | **16.47 s** | HIT 时 host_stage 会记进 DiT，**不要和 31.67 横比** |
| TE | ~15–17 s | 同左 | HIT 金线 e2e **61.41 s** |
| `dit_host_stage` | **13.2 s** | 同量级 | TE 并行时藏在 TE；HIT 后暴露 |
| VAE | ~10–10.6 s | 同左 | TAE 未成片；砍掉则 e2e −8～10 s |

Ulysses 通信体积（F32，已测对得上 34.66 / 137）：U=2 每卡发 **184 GB**；U=8 总线 **~642 GB**。host 有效带宽按本机 **~10–12 GB/s**。

### 路 A — 单请求延迟

目标：一条 cafe 墙钟。U=8 **不参与**。

| 档 | 动作 | DiT 回代 | e2e 回代 | 依据 |
| --- | --- | ---: | ---: | --- |
| A0 现状金线 | U=1 | 31.67 | 78.74 | 已测 |
| A1 已有加速档 | turbo4+TE HIT | （含 host，~25） | **~50** | 已测。当前诚实延迟上限 |
| A2 软件刀 U=2 | 去 DeviceSync + **BF16 A2A**（学 Cache-DiT float8 的降档） | **25–28** | ~72–74（金线其余段未切） | 184→92 GB ÷12 GB/s ≈7–8 s 通信 + 15.8 计算 + ~2 宿主。**首次有希望 < 31.67** |
| A2′ 再降 FP8 A2A | 学 LightX2V `use_fp8_comm` / Cache-DiT `ulysses_float8` | **~22–25**（乐观） | ~70–72 | 体积再半。**非位级**，MAE 另阈。不要第一步 FP4 通信（`use_fp4_comm` 社区有开关，数值风险高于收益） |
| A3 叠 turbo4？ | U=2 + 4-step | 计算 ~8.2 + 通信 ~7.5 + 宿主 ~2 ≈ **~18** | 可能 **慢于** turbo4 U=1 的 16.47 | **480p 4-step 上 U=2 很可能仍亏。** 路 A 的 U=2 只绑 **8-step 金线/实验臂**，不要绑 turbo4 默认 |
| A4 +TAE | 加速档官方 VAE→TAE | 不动 DiT | A1 的 50 **→ ~40** | 砍 VAE ~10。与 Ulysses 正交。权/stub 另路 |
| A5 副机 P2P（可选） | aikitoria 类补丁，**一对卡**，测到 ≥40 GB/s 再谈 | U=2 **~20–22**（55 GB/s 表）；若只有 vLLM 那台的 **14 GB/s**，则 ~15.8+13≈**29**，几乎不赢 | e2e 仍被 TE+VAE+host 托在 **40–50** | **本机 Xeon 不要抄 EPYC 55 GB/s。** 先测带宽再承诺 |
| A6 幻觉 | U=8 除以 8 → DiT 4 s；e2e 5–8 s | 已测 **137** | 已测 **212** | 作废 |

**路 A 天花板（诚实）：**

- **今天就能宣称：** ~50 s（turbo4+TE HIT）。
- **用户态做到头：** ~40 s（再加 TAE）；U=2 BF16 只把 **金线 DiT** 拧到 25–28，e2e 仍 ~70+，**赢不了 turbo4**。
- **副机 P2P 做到头：** 金线 DiT ~20–22 s（前提是真 40+ GB/s），e2e 仍 ~40–50。
- **社区「1.8 s」做到头：** 必须换模型+3-step 蒸馏+原生 NVFP4 核+TAEHV，那是另一条产品线（FastWan-QAD），不是这台箱上的 H3 金线。

### 路 B — 8 卡吞吐

目标：卡均视频/小时。卡间 **零 A2A**。

| 配方 | 单条 e2e | 理想 8 路聚合（条/小时） | 热/电打折 0.8× 后 |
| --- | ---: | ---: | ---: |
| 金线 U=1 | 78.74 | 8 × 3600/78.74 ≈ **366** | ~293 |
| turbo4 | ~64 | ≈ **450** | ~360 |
| turbo4+TE HIT | ~50 | ≈ **576** | ~461 |
| +TAE（回代） | ~40 | ≈ **720** | ~576 |

单卡金线只有 **3600/78.74 ≈ 46** 条/小时。路 B 是 **今天就能宣称的 8 卡突破**（约 **8×**，再扣电源/温度）。

不要：一条请求 U=8（0.23×，吞吐变负）；不要 MPS 双作业；不要 MIG。

实现形态：8 个 `h3_cuda_generate`，`CUDA_VISIBLE_DEVICES=i`，队列扇出。NUMA：作业 0–3 绑 GPU0–3 / node0，4–7 绑 node1。权重缓存每卡一份（已经是单卡常驻模型）。

---

## 3. 立刻做的 3 件事 / 明确不要做

### 立刻做（具体到模块）

1. **路 B 调度（吞吐极致，零通信）**  
   - 新建薄启动器（建议 `tools/h3_dp_generate.sh` 或小 C wrapper），**不要改** `h3_dit_w4a8.c` / `h3_ulysses_comm.cu`。  
   - 调现有 `tools/h3_cuda_generate.c`：8 路、每路 `--ulysses 1`、可选 `--preset turbo4 --te-cache`。  
   - 验收：8 条并发 cafe，单条 e2e 不差于单卡 10%，聚合 ≥ 6.5×（留热/电余量）。

2. **路 A 软件刀：只做 U=2，只打金线 DiT < 31.67**  
   - `h3_ulysses_comm.cu`：放开 `dtype_bytes==4` 硬拒；pack 前 F32→BF16（下一步才实验 FP8，对齐 Cache-DiT `ulysses_float8` / LightX2V `use_fp8_comm`）。头文件 `h3_ulysses_comm.h` 已写 `dtype_bytes=2`。  
   - `h3_dit_w4a8.c`：`uly_sync_rank` 去掉每相 `cudaDeviceSynchronize`，A2A 前 per-rank event；`uly_h2d_aux` 改 async。  
   - 只跑同 NUMA GPU0,1、cafe warm。过门（DiT ≤ 28 s + BF16 MAE 另阈）当加速档开关；不过门关掉。**禁止 U=8。**

3. **把 e2e 地板从 50 推向 40：TAE 成片（与多卡正交）**  
   - `h3_tae.c` / `h3_tae_cuda.cu` / `h3_tae.h`：加速档 decode，金线仍官方 VAE。  
   - 同时不要抢 `tools/h3_cuda_generate.c` 里已有的 `dit_host_stage` 双缓冲（另路）。  
   - 验收：turbo4+TE HIT+TAE，e2e 回代 ~40 s，不是 5 s。

### 明确不要做

- 生产箱装 aikitoria / QuixiAI / tinygrad / Duanyll / rtx-p2p。副机 + IPMI + 备份 ko 之前，连 `595.45.04-p2p` 都不装。
- 为 P2P 升官方驱动、换 CUDA toolkit、设 `NCCL_P2P_DISABLE`、指望 GDRCopy / `nvidia-peermem` / Fabric Manager。
- U=8、TP、CFG 并行、Ring（未换 FA 核）、`HOST_FWD=1`、把 20 SM 划给 NCCL。
- 把 FastWan-QAD 1.78 s、RH 12.2×、explainer 5–8 s 写成 H3 8 卡目标。
- git submodule xDiT / LightX2V / FastVideo / DeepEP / FlashMLA 当 C 后端。
- NVFP4 **CPU dequant 424 s** 当加速档（已出片的是正确性，不是性能）。
- 自研 `nvidia.ko`。

---

## 4. 若用社区方案：用哪一个、风险、怎么接 C 栈

**选「偷配方进 C」，不选「把 Python 引擎嵌进来」。**

| 优先级 | 用哪一个 | 风险 | C 栈接法 |
| --- | --- | --- | --- |
| **P0** | **Cache-DiT `ulysses_float8` + LightX2V `use_fp8_comm` / `ulysses-4090`** | BF16/FP8 A2A 非位级；FP8 要单独 MAE。`ulysses-4090` 的手工 send/recv 在 U=2 可能比 NCCL A2A 少启动税，U=8 无意义 | **不链 Python。** 在 `h3_ulysses_comm.cu` 加 pack/unpack + 现有 NCCL send/recv（或 U=2 已写的 `a2a_peer_u2`，`peer=0` 时仍走不通 peer）。对照文件：LightX2V `ulysses_attn.py` 的 quant+P2P 环，只抄体积和握手，不抄 FlashAttn |
| **P0** | **请求级 DP**（xDiT `data_parallel_degree` / 任何 serving 的默认） | 几乎无。队列不公平、缓存目录要按 GPU 分片 | 启动器 + `CUDA_VISIBLE_DEVICES`。TE cache 目录加 `gpu$i` 后缀，避免脏 key（前科 `63ba422170cf015a`） |
| **P1** | **FastWan-QAD / Attn-QAT 思想**（不是权重） | 换 3-step + 原生 NVFP4 注意力 = 新配方，质量门重开。Sage3 本仓已有探测 `tools/h3_sage3_embed.c`，不是金线 | `h3_dit_nvfp4.c` 走 **sm_120 MMA**，禁止 CPU dequant 路径进 generate。注意力保持 cuDNN，除非加速档另开 Sage3 且过 MAE |
| **P1** | **MagCache / TeaCache / DBCache 跳块** | 8-step 窗口窄；SCM 对 <8 step 会关。画质漂 | 只进 turbo 实验臂的 `layers_forward` skip，默认关 |
| **P2 副机** | **aikitoria `595.45.04-p2p`**（版本对得上本机）或 QuixiAI（若愿升 610 整包） | hang / 误写 / EULA / 重启残留。见 driver-options §7 | **不接 C 栈。** userspace 已有 `canAccessPeer` 探测。副机：`simpleP2P` + `nvbandwidth` + 本仓 U=2 短测。生产箱保持 595 官方 |
| **不做** | rtx-p2p、DeepEP、FlashMLA、xDiT 整仓、Comfy DisTorch、FlowCache | 法律 / 错硬件 / 错模型 / 错 KPI | — |

**和 C 栈的边界一句话：** 社区给的是 **通信体积合同**（BF16/FP8 A2A、消费卡手工 P2P 环）和 **单卡配方合同**（少步+原生 FP4+小 VAE）。公路（`canAccessPeer`）仍由官方驱动决定。C 栈已经会探测 peer；缺的是 pack 精度和 DP 启动器，不是再嵌一个 Python USP。

---

## 5. 检索日志（防「只用仓库旧三家」）

本轮 **WebSearch / WebFetch** 新点名、旧文未当主角的：

- FastWan-QAD / Attn-QAT / TAEHV / SageAttention3 FP4（Hao AI Lab，2026-06，**点名 5090 1.78 s**）
- Cache-DiT `ulysses_float8` / `ulysses_async` / DBCache / TaylorSeer / SCM（SGLang Diffusion，列 MiniMax-H3）
- LightX2V `ulysses-4090`、`use_fp8_comm`、`use_fp4_comm`、SM120 MXFP8 PR #1090、5090 框架横比 2.73×
- Raylight（Comfy USP，**5090 vs 2000 ADA 表**）
- ParaVAE、FlowCache（ICLR 2026）、DisTorch
- vLLM 2×5090 P2P **+2.7% / 14.1 GB/s**（2026-08-11）
- llama.cpp P2P opt-in PR #21910
- QuixiAI / ec-jt / Duanyll / rtx-p2p（aikitoria 旁支）
- PipeFusion 8×L40；xDiT runner 现含 MiniMax-H3
- NVIDIA 对照表 50 系全无 NVLink；GPUDirect RDMA 仍 Tesla/Quadro；5090 无 MIG

旧三家（aikitoria / DeepSeek / RunningHub）只作尺子，不展开。

---

## 附录 A. 本机拓扑（只读，不重复论证）

| 项 | 值 |
| --- | --- |
| 卡 | 8×5090，32 GB，无 NVLink |
| 驱动 | 595.45.04 Open Kernel Module |
| 拓扑 | GPU0–3 NUMA0 `NODE`；GPU4–7 NUMA1 `SYS` |
| P2P | `topo -p2p r` 全 CNS；`peer_ok=0` |
| IOMMU | 翻译模式开（`dmar0–19`），无 `iommu=pt` |
| BAR1 | 32 GiB prefetchable |

未本轮 SSH。未装补丁。密码未入库。

---

## 6. 给执行者的选择（不自动开工）

本文不是实现计划。若要落地：

1. **吞吐：** 做 §3.1 DP 启动器。  
2. **金线 DiT 再快一点：** 做 §3.2 U=2 BF16。  
3. **e2e 50→40：** 做 §3.3 TAE。  
4. **P2P：** 只开副机窗口，源用 aikitoria `595.45.04-p2p`，门槛见 driver-options §7。

未改推理代码，未 commit。
