# 8×RTX 5090：还能不能救 Ulysses、要不要换/自研驱动

> **日期：** 2026-09-13。战略+技术调研。未改推理代码，未 commit，密码未入库。未装社区内核，未 reload 驱动。
> **本轮 SSH（只读，2026-09-13 07:47 UTC）：** `nvidia-smi` 驱动 **595.45.04** Open Kernel Module；`topo -p2p r` 全 **CNS**；GPU0 有别人的 `h3_cuda_generate --preset nvfp4`（~19 GiB，cafe 1s/2-step）。aikitoria 实机试 **没做**，见 §7。
> **钉死事实（不推翻）：** 8×5090、PCIe、无 NVLink、`peer_access=0`；U=1/U=2/U=8 同 binary 位级对齐；金线 DiT **31.67 s**，U=2 **34.66 s**，U=8 **137 s**。根因见 [lightning-survey](./2026-09-13-h3c-minimax-lightning-survey.md)；审计见 [plan-results-audit](./2026-09-13-h3c-plan-results-audit.md)。
> **代码锚：** `h3_ulysses_comm.cu`（`canAccessPeer` / NCCL send-recv / F32 硬拒）、`h3_dit_w4a8.c`（`uly_run_ranks` / `uly_sync_rank` → `cudaDeviceSynchronize`）。

**一句话：** 问题是公路（GeForce 官方关 GPU↔GPU P2P → A2A 走 GPU→host→GPU），不是算法。U=2 用软件刀有机会略快于 31.67 s（约 25–28 s），到不了「除以 2」。U=8 在这台机器上救不回来。社区有 **非官方** 内核补丁在 5090 上把 `canAccessPeer` 拧到 1（有带宽表），**没有**「换官方驱动就 peer=1」的 5090 实锤。自研驱动建议 **不做**。8 卡突破应改定义为吞吐 ×8，或承认单请求延迟的诚实上限大约 e2e 50 s 档（turbo4+TE），不是 5–8 s。

---

## 1. 问题是公路还是算法

**公路。** 切分已经对了。

| 证据 | 含义 |
| --- | --- |
| 同 binary U=1 ≡ U=2 ≡ U=8（远端 mp4/ppm/wav 位级相同） | 行切 + 头换手没有画坏图 |
| cafe U=2 DiT 34.66 vs 金线 31.67 | 计算本该 ≈15.8 s，被 ~184 GB/卡 F32 host bounce 吃光 |
| cafe U=8 DiT 137 vs 31.67 | 总线约 642 GB 挤主机 + 跨 NUMA `SYS`，「除以 8」变成「乘以 4」 |
| A2A 改 stream event 后墙钟几乎不动 | 税不在 A2A 函数末尾那次 drain，在体积 + host 路径 + 每相 `uly_sync_rank` |

本仓路径（磁盘现状）：

```text
canAccessPeer=0
  → 不 EnablePeerAccess / 不 MemcpyPeer
  → peer_a2a=0
  → ncclSend/Recv 拼 destination-major A2A（F32）
  → NCCL 走 SHM / host bounce
每层 QKV → A2A → SDPA → A2A → MLP
每相 uly_run_ranks：全 rank enqueue 后对每个 rank cudaDeviceSynchronize
```

Ulysses 论文（DeepSpeed，[arXiv:2309.14509](https://arxiv.org/abs/2309.14509)）和 xDiT USP（[arXiv:2405.07719](https://arxiv.org/abs/2405.07719)）假定的是 **NVLink / 数据中心 P2P** 量级的 all-to-all。算法没错；把同一套 A2A 放到 GeForce + host bounce 上，480p 8-step 的计算藏不住换手。

RunningHub 不能当反证：他们的 12× 几乎全是 50→4 步；卡是 **RTX 6000D**（工作站，PCIe P2P 通常 `can=1`）；活是 768p BF16。三条我们都不占。见 lightning-survey §2。

---

## 2. 社区 / 驱动现状（带链接）

### 2.1 官方：5090 没有 NVLink；GeForce 50 系不支持 GPU↔GPU P2P

| 来源 | 日期 | 说了什么 |
| --- | --- | --- |
| NVIDIA 5090 规格页 [NVLink (SLI-Ready): **No**](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/) | 现行产品页 | 硬件上就没有 NVLink 桥 |
| NVIDIA 员工 Yuki_Ni，[开发者论坛 326776](https://forums.developer.nvidia.com/t/p2p-issue-using-two-rtx-5090-gpus/326776/8) | **2025-03-13** | 「RTX 50-series are Geforce… **p2p is not supported for GPU↔GPU P2P**. For GPU lists that can support p2p, there are mainly 2 series, Quadro RTX and Data Center GPUs.」机器是 2×5090、驱动 570.124、IOMMU 已关、BAR1=32 GiB，**仍被官方判不支持** |
| NCCL 维护者（Stephen DeVuyst），[NCCL #1637](https://github.com/NVIDIA/nccl/issues/1637) | 2025-03 | P2P 不是 NCCL 功能；「**not supported on any recent GeForce hardware**」。`NCCL_P2P_DISABLE=1` 在 5090 上本来就没效果——NCCL 探测到没有 P2P 会自己关。2.26+ 修的是 **SM120 共享内存上限**，让 SHM 通路能跑，**不是**打开 P2P |
| NVIDIA Engineering（4090），[论坛 233202](https://forums.developer.nvidia.com/t/standard-nvidia-cuda-tests-fail-with-dual-rtx-4090-linux-box/233202/21) | 2023-02 | 「Peer to Peer is **not supported on 4090**。」4090 做成过的社区补丁 **不能外推** 成「官方 5090 也能开」 |
| GPUDirect RDMA 文档 | 现行 | [「available on both Tesla and Quadro」](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html)——名单里没有 GeForce |
| NCCL [GPU troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html) | 现行 | NCCL 只在 CUDA 报告 `canAccessPeer` 时走 P2P。IOMMU/ACS 是「**已经有 P2P** 时为什么慢/坏」的排查，不是「消费卡怎么打开 P2P」的开关 |

**本仓对照：** U=8 日志 `peer_access ok=0 fail=56 peer_a2a=0`。`h3_ulysses_comm.cu:164-202` 已按官方合同探测：`can=0` 就不 `EnablePeerAccess`。这不是忘了开。

`cudaDeviceCanAccessPeer` 的语义见 [CUDA Runtime Peer Access](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PEER.html)：返回 0 就禁止 `EnablePeerAccess`。GeForce 上这个 0 是 **SKU / 驱动策略**，不是 ACS 开关没拨。

### 2.2 「换官方驱动就能 peer=1」：5090 上没有实锤

检索范围：NVIDIA 论坛、NCCL issues、vLLM 5090 TP=2 帖、CUDA samples。

找到的 5090 **官方栈**数据点全部是 `canAccessPeer=0` / NCCL SHM：

- 2025-03：2×5090 Threadripper 7960X，驱动 570.124.06，官方回复不支持 P2P（上表）。
- 2025–2026：NCCL #1637 后续（2026 仍有人报）2×5090、驱动 595.71.05，**「PYNCCL shared-memory transport (no P2P, … as expected on GeForce)」**。
- 本机：驱动 **595.45.04 Open Kernel Module**（`docs/nvidia_build.md`），U=8 `peer_ok=0`。

**没有**找到「某日、某张 5090、**未改内核模块**、只换官方 .run / 只关 IOMMU / 只关 ACS / 只开 ReBAR，`cudaDeviceCanAccessPeer` 变成 1」的报告。4090 上 tinygrad 补丁的成功 **不算** 5090 官方解。

R545 曾误报 40 系 `can=1` 但拷贝是坏的，NVIDIA 在 R550 修回「正确报告不支持」（[论坛 283264](https://forums.developer.nvidia.com/t/direct-gpu-gpu-communication-does-not-seem-to-work-properly/283264)）。把官方驱动升到「看起来 can=1」而不验证拷贝，是已知坑。

### 2.3 社区私货：5090 上 *有* 带宽表，但是改内核 / 改 GSP 键

这是和「换官方驱动」完全不同的一条路。硬件 BAR1 能走 DMA；官方在 GeForce SKU 上关掉选择。补丁是把选择强制打开。

| 项目 | 卡 / 日期 | 证据强度 | 备注 |
| --- | --- | --- | --- |
| [tinygrad/open-gpu-kernel-modules](https://github.com/tinygrad/open-gpu-kernel-modules) `550.54.15-p2p` | 先 4090/3090（2024） | 40 系实锤，**不是** 5090 | geohot 路线：证明「消费卡 P2P 是驱动策略不是硅残」 |
| [aikitoria/open-gpu-kernel-modules](https://github.com/aikitoria/open-gpu-kernel-modules) `610.43.03-p2p` | **8×5090 + 1×RTX PRO 6000**，双路 EPYC 9575F；README 最后推 **2026-07-13** | **5090 实锤（非官方）** | `p2pBandwidthLatencyTest`：P2P 矩阵全 1；单向 P2P≈**55–56 GB/s**，双向≈**111 GB/s**；P2P 关时单向仍≈43 GB/s（那是很好的 EPYC Gen5 host 路径，**不是**我们这台 ~10–12 GB/s） |
| [QuixiAI/open-gpu-kernel-modules](https://github.com/QuixiAI/open-gpu-kernel-modules) | 610.57.04；宣称 GB202；建仓 **2026-08-26** | 文档写 5090，**验证段落写的是 8×3090** | 自己承认 610 已有原生 BAR1 HAL，只是 GeForce 不选 |
| [kacper-daftcode/rtx-p2p](https://github.com/kacper-daftcode/rtx-p2p) | 宣称 5090、GSP 寄存器 `RMDisableFeatureDisablement`、**不改内核** | 8 star、README 薄 | **反汇编 GSP 固件**。法律/稳定风险高于 aikitoria。不当生产候选 |

aikitoria 的前提（他们自己写的）：`iommu=pt`（翻译模式 DMA 会坏）、ACS 关掉否则 P2P 仍绕 CPU、ReBAR、内核模块版本必须和 userspace/GSP **锁死同一 610.43.03**。

**结论：** 社区 *有* 5090 做成 P2P 的例子，但是 **改开源内核模块 + 钉死驱动版本**，不是「apt upgrade nvidia-driver」。本机当前 595.45.04 官方模块走不通这条路。**2026-09-13 实机试：有风险，没试。** 见 §7。

### 2.4 容易幻觉的配件（对 GPU↔GPU P2P 无效或答错题）

| 东西 | 实际干什么 | 对本仓 A2A |
| --- | --- | --- |
| `nvidia-peermem` | GPU↔**网卡** GPUDirect RDMA | 不打开 5090↔5090 |
| [GDRCopy](https://github.com/NVIDIA/gdrcopy) | CPU 映射 GPU 显存，加速 host 登记 | 可能略减 bounce 拷贝税，**变不成**卡直写卡 |
| Fabric Manager | NVSwitch / HGX | 这台没有 NVSwitch |
| 换 CUDA toolkit（12.8↔13.2） | 用户态运行时 | `canAccessPeer` 在驱动/GSP，不在 nvcc |
| `NCCL_P2P_DISABLE=0`（默认） | 「没关 P2P」 | `can=0` 时 NCCL 本来就不会走 P2P。**不要设 DISABLE=1**（和 RH 一样） |
| CUDA IPC / 自己写 pack+MemcpyPeer | 本仓 U=2 快路径已写（`a2a_peer_u2`） | **`peer_a2a` 仍为 0 就走不通**。IPC 不能绕过 `canAccessPeer` |

### 2.5 社区视频 DiT 多卡：谁在 5090 上宣称单条请求加速

| 项目 | 多卡机制 | 公开数字的硬件 | 5090 单请求加速？ |
| --- | --- | --- | --- |
| **xDiT / USP** | Ulysses × Ring × 可选 PipeFusion | HunyuanVideo 官方表是 **H100 / H20 / L20**（720p 50-step：H100 1→8 卡 1904→338 s）。L40 PCIe 上 8 卡相对 4 卡几乎不降（[hunyuandit.md](https://github.com/xdit-project/xDiT/blob/main/docs/performance/hunyuandit.md)：「low communication bandwidth across sockets」） | **没有**找到 8×5090 单条请求赢单卡的报告 |
| **DeepSpeed Ulysses** | 训练向 SP | 数据中心 | 论文不是消费卡推理合同 |
| **Hunyuan / Wan 官方** | FSDP + Ulysses | 表里常见 Hopper + FlashAttn3 | 多卡是为了 **装得下 + 长序列**；不是 5090 480p 8-step 故事 |
| **FastVideo** | 自动选并行 + Sage / TeaCache | 对 Wan 宣称相对官方实现最多 ~3× | 少步/近似核，不是「无损 SP」 |
| **Comfy 多卡** | Raylight USP（要快互连）；Worksplit=CFG 对半；MultiGPU=显存捐赠；Distributed=多请求 | Reddit 2025-11 自己承认 USP「really want a very fast connection」 | CFG 并行我们 `cfg=1` 无第二条臂。捐赠显存不降墙钟 |
| **MAGI-2** | 8 卡预设面向 Hopper 级显存 | 1080p 装不下单卡 | 不是 5090 单请求加速证人 |
| **RunningHub Lightning** | TP2×U4，NCCL A2A | **8× RTX 6000D**（有 PCIe P2P） | 见 survey：12× 是减步 |
| **本仓** | 纯 U=2/U=8 | 8×5090 | **已测：更慢** |

社区共识和我们一致：Ulysses 在 **NVLink / 工作站 P2P + 足够长的序列** 上赚钱；在 **GeForce + 短 480p** 上赔钱。没有第二家在 5090 `peer=0` 上把「单条请求比单卡快」做成可引用数字。

---

## 3. 不换驱动也能做的软件刀（按 ROI）

金线默认保持 U=1。下列只进加速档实验臂。秒数是 **cafe 864×480 5s / 8-step** 相对金线 DiT **31.67 s** 的回代，不是再测。

### P0 — 先做 / 先停

| # | 改哪里 | 预期 U=2 DiT | 预期 U=8 DiT | 风险 |
| --- | --- | --- | --- | --- |
| P0.1 | **停止把 U=8 当主路径。** 文档改口（本文 + explainer 5–8 s / DiT 2.5–4 s） | 不变 | 不变 | 无。省机时 |
| P0.2 | 加速档继续 **turbo4 + `--te-cache`**（已测 e2e ~49.7–50 s HIT） | 不靠 Ulysses | 同左 | cache 默认仍关；turbo 非金线 |
| P0.3 | 请求级 **DP**：8 卡各跑一条完整 generate，卡间零 A2A | 单条仍 ~31.7 / e2e ~79 | n/a | 单请求延迟不变；吞吐 ×8。这是 `peer=0` 上唯一稳的 8 卡用法 |

### P1 — 还值得做一次的 U=2（实验臂）

| # | 改哪里 | 预期 U=2 DiT | 预期 U=8 DiT | 风险 |
| --- | --- | --- | --- | --- |
| P1.1 | `h3_dit_w4a8.c`：`uly_sync_rank` 去掉每相 `cudaDeviceSynchronize`，A2A 前只用 per-rank event。A2A 已是 event，不要改回 `StreamSynchronize`。`uly_h2d_aux` 改 async | **32–33 s**（单独不够翻正） | 仍 ≫ 32（只削宿主税） | 曾 IMA；须保持 per-device cuDNN plan |
| P1.2 | `h3_ulysses_comm.cu`：放开 `dtype_bytes==4`，pack 前 F32→BF16，unpack 后升回。字节减半 | **25–28 s**（回代：通信 ~7–8 + 计算 15.8 + 宿主 ~2）。**这是不改驱动时首次有希望快于 31.67 的刀** | 642/2 GB 仍可能 **50–80 s** | **非位级等价**，须另开 MAE 门 |
| P1.3 | 减集体：融合相邻层只在 attn 边界 A2A；MLP 不要多余相同步 | 含在 P1.1 的 1–3 s | 同理 | 正确性；别把 QKV/attn-out 两次 A2A 合成一次除非证明等价 |
| P1.4 | **不要**再赌 `MemcpyPeer` / CUDA IPC | 0 | 0 | `can=0` 必失败。代码快路径已在，探测失败即死 |

P1.1+P1.2 做完：**U=2 有机会 25–27 s**（约 0.8× 金线 DiT），e2e 仍约 72–74 s（TE+VAE 没切）。**U=8 仍然救不回来。**

计算-通信重叠：Ulysses 语义上 A2A 必须等齐（RH 也是 `waits immediately`）。去掉 DeviceSync 只能让宿主少空转、让下一相 enqueue 早点，**不能**把 184 GB bounce 藏进 GEMM。不要写「overlap 之后 U=2≈16 s」。

### P2 — 换并行轴，不是继续拧 U

| # | 策略 | 预期 | 风险 |
| --- | --- | --- | --- |
| P2.1 | **TP 为主** | PCIe 上 per-block all-reduce 比 A2A 更亏（RH 自己 TP4+U2 比 TP2+U4 慢 12%；SGLang 文档把 TP 排在最不友好）。本仓 INT8 单卡已常驻，**没有容量理由** | 明确不建议 |
| P2.2 | **U=2 封顶**，U=4 仅同 NUMA GPU0–3 再评估（最好 768p） | 480p U=4 未测；按体积介于 367–642 GB，大概率仍亏 | 不要用 U=8 验证「多卡该快」 |
| P2.3 | CFG 并行 | `cfg=1` 无第二条臂 | 零收益 |
| P2.4 | 「切序列但不切 head」（Ring / 只分 token、attn 用 KV 切片） | Ring 要 FA 风格在线 softmax；本仓 packed cuDNN 合同不同。PCIe 上 Ring hop 也要带宽 | 工作量大，480p 仍可能亏 |
| P2.5 | 多请求 DP vs 单请求 SP | DP：吞吐 ×8，延迟不变。SP：只有 P1 或 P2P 之后才谈延迟 | 产品先选 KPI，见 §6 |

### 预期汇总

| 动作 | U=2 DiT | U=8 DiT |
| --- | ---: | ---: |
| 现状 | 34.66（亏） | 137（大亏） |
| 只去 DeviceSync | ~32–33 | 仍远慢 |
| + BF16 A2A | **~25–28（可能首次快于金线）** | 仍 ≫ 32 |
| 社区补丁 P2P ≈55 GB/s（§2.3，非官方） | ~20–22 | 乐观 25–40，**480p 仍可能亏**；不承诺 |
| 官方工作站卡（6000D 类，P2P 合法） | ~20–22 | 768p 才值得试 U=4；480p U=8 仍险 |
| 搬 RH Sage/4-step | 换配方，另册 | 同左 |

e2e：只切 DiT 时 U=2 理想也到不了 40 s 以下（TE+VAE+host_stage 仍 ~26–40 s）。要 50 s 以下走已测的 turbo4+TE cache，不走 U=8。

---

## 4. 换官方驱动 / 改 BIOS / 拓扑：值不值得试

半天实验预算怎么花：

| 动作 | 预期对 `canAccessPeer` | 预期对墙钟 | 值不值得 |
| --- | --- | --- | --- |
| 官方驱动 595 → 更新的官方 595/610 **原版** 模块 | **仍为 0**（官方政策，§2.1–2.2） | NCCL 2.26+ 只修 SM120 能跑，我们 A2A 已经能跑 | **不值得为 P2P 升。** 安全补丁另说 |
| 换 CUDA 12.8/13.x toolkit | 0 | 0 | 幻觉 |
| `iommu=pt` / 关 IOMMU（官方模块） | 0（SKU 门在前） | host bounce 可能好几个百分点 | **关书用：半小时确认 `can` 仍 0。** 以后若装社区补丁，这项变成 **必做** |
| BIOS 关 ACS / `pcie_acs_override`（官方模块） | 0 | 无 P2P 时 A2A 已经走 host，ACS 主要伤害的是「已有 P2P」的直达 | 官方栈 **低 ROI**。社区补丁后变成必做 |
| ReBAR / Above-4G | 0（官方仍关 P2P） | BAR1=32 GiB 已在 5090 规格里 | 确认开着即可；不是开关 |
| 把 U=2 钉在同 NUMA GPU0,1 | 已如此 | 同域仍亏 3 s | 已经做了。跨 SYS 的 U=8 更差，不要再测来「证明该快」 |
| 官方 NCCL 升级 | 不改变 P2P | 避免 SHM 回归 | 随 CUDA 走即可 |

**幻觉：** 「换个 Game Ready / Studio / open kernel 官方包，5090 就 peer=1。」论坛 + NCCL + 本机日志三重打脸。

**唯一「换驱动」有物理意义的动作：** 装 **aikitoria / QuixiAI 这类改过的 open kernel modules**（§2.3），userspace 锁到对应 610.x。这不是官方驱动。代价见 §5。收益上限见 §3 表「社区补丁 P2P」行：U=2 DiT ~20–22 s，不是 5 s。

若只想花半天且不碰内核：**不要改驱动。** 把半天给 P1.1+P1.2 的实现和一次 cafe warm。

---

## 5. 自研驱动：结论必须明确

**建议不做。** 没有找到决定性反证。

分清三件被混为一谈的事：

1. **从零写 NVIDIA 兼容驱动**（替代 `nvidia.ko` + GSP + CUDA userspace）  
   要复刻 RM、GSP RISC-V 固件协议、UVM、显示、CUDA 上下文、与 cuDNN/NCCL/nvrtc 的 ABI。NVIDIA 用数百人年维护。本仓收益是「让 8 张消费卡 BAR1 DMA 被 CUDA 承认」——社区 fork 已经把这一刀切完了。  
   **量级：10² 人年。不要做。**

2. **自己维护一份「打开 GeForce P2P」的内核 fork**（复刻 aikitoria，不从零写）  
   开源内核模块是 **MIT/GPLv2**（[NVIDIA/open-gpu-kernel-modules COPYING](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/580.76.05/COPYING)），改源码分发在内核许可下说得通。但必须和 **专有** userspace + **GSP 固件** 锁版本；固件 EULA 禁止翻译/逆向（[NVIDIA Linux Driver License](https://www.nvidia.com/en-eu/drivers/nvidia-license/linux/)）。Secure Boot 要自签；每个上游安全公告都要 rebase；错一次是整机 8 卡挂死或静默坏显存。  
   **量级：上手 2–4 人月 + 每年 1–2 人月跟版。** 产出不超过 aikitoria 已经贴出的 55 GB/s。对 U=8 480p 仍可能赢不了单卡。

3. **GSP 寄存器「一键开 P2P」（rtx-p2p 路线）**  
   反汇编专有固件、关 SKU feature gate。法律风险明显高于 (2)，样本面极小。  
   **不要在生产 8 卡箱做。**

**自研比社区 fork 多挖不出什么。** 硅上没有藏着的 NVLink。能挖的就是官方关掉的 **BAR1 P2P** 和拷贝引擎调度；前者社区已挖，后者在 `can=0` 时 CUDA 不会把对端显存交给 CE。绕过 host 的唯一合法硬件解是：**换有 P2P 的卡（6000D / Pro / 数据中心）或换有 NVLink/NVSwitch 的箱。**

若执意要 P2P：用 **现成社区模块** 在 **牺牲机 / 一对卡** 上跑 `simpleP2P` + `nvbandwidth` + 本仓 U=2 短测，不要从零写。生产箱保持官方 595，直到补丁在副机上 cafe 数字和稳定性都过门。

---

## 6. 「8 卡 5090 要突破」的正确定义

先选 KPI，再谈突破。现在的失败模式是：用 **单请求延迟** 当 KPI，却拿 **8 张 peer=0 的卡做 Ulysses**，再用 explainer 的 5–8 s 当成功线。

### 6.1 两把尺子，不要混

| KPI | 正当用法 | 成功长什么样（这台机器、cafe 5s） |
| --- | --- | --- |
| **单请求延迟**（一条视频墙钟） | 金线 U=1；加速档 turbo4+TE（已有）；U=2 仅当 P1 测完 **DiT < 31.67** | **P1 成功：** U=2 DiT ≤ 28 s 且同 binary MAE≈0（BF16 另阈）。**P2P 成功（若将来有）：** U=2 DiT ≤ 22 s。**U=8 成功：** DiT < 31.67——480p `peer=0` 上 **不要设这个门**。e2e 地板在 TE+VAE+host，诚实目标 **~50 s（已有 HIT）或 ~40 s（再加 TAE）**，不是 5–8 |
| **吞吐**（卡均视频/小时） | 8 路 DP，互不通信 | **成功：** 8 条并发 cafe，单条 e2e 仍 ~79 s（金线）或 ~50 s（turbo+TE），聚合 ≈ **8× 单卡吞吐**。这是 8 张 5090 **今天就能宣称的突破** |

### 6.2 什么不叫突破

- U=8 成片正确但 137 s（正确、更慢）。
- 相对 50-step 基线的 12×（RH 的分母，不是我们的）。
- 「8 卡线性摊薄 DiT 31.67→4 s」。
- 换官方驱动之后 `nvidia-smi` 版本号变了但 `can` 仍为 0。

### 6.3 下一件具体该干什么（按目标选一条）

1. **若目标是单请求再快一点：** 实现 P1.1+P1.2（去 DeviceSync + BF16 A2A），只跑 **U=2、同 NUMA、cafe warm**，和 31.67 比。过门当加速档开关；不过门关掉。**不要再开 U=8。**
2. **若目标是「8 张卡不能闲着」：** 做请求级 DP 调度，用现有单卡金线/turbo 二进制 ×8。这是突破成功的正确定义。
3. **若目标是单请求 20 s 级 DiT：** 先完成 (1)；仍不够再评估 **换 6000D/Pro** 或 **副机试 aikitoria**，不要自研驱动，不要在生产箱赌 GSP 私货。
4. **不要做：** 为 P2P 升官方驱动、在生产箱自研/刷社区内核当第一刀、把 TP 当主轴、把 5–8 s 写回对外页。

---

## 7. aikitoria 风险与是否实机试（2026-09-13）

用户原话：「没有风险就应该试试。」**有风险。这次不试。** 门槛任一不满足就只写评估——本节是评估，不是施工单。

### 7.1 原始仓库 / 帖（已读 README + issue + 对比官方树）

| 源 | 角色 |
| --- | --- |
| **[aikitoria/open-gpu-kernel-modules](https://github.com/aikitoria/open-gpu-kernel-modules)** | 正源。tinygrad/geohot 路线的简化移植。点名 **RTX 3090 / 4090 / 5090**，P2P 路径写 **PCIe BAR1**；同代可混（5090 ↔ RTX PRO 6000 **Blackwell**）。不是官方包。 |
| 默认分支 `610.57.04-p2p-v2`（README 末推 **2026-07-13**） | 8×5090 + 1×RTX PRO 6000、双路 **EPYC 9575F**：`p2pBandwidthLatencyTest` 矩阵全 1；单向 P2P ≈ **55–56 GB/s**，双向 ≈ **111 GB/s**。P2P 关时单向仍 ≈43 GB/s（那是很好的 EPYC Gen5 host 路径，**不是**本机）。 |
| 分支 `610.43.03-p2p` | 上一档钉死版本；本机 **不是** 610。 |
| 分支 **`595.45.04-p2p`** | README 标题就是「595.45.04 with P2P for **4090 and 5090**」。和本机驱动 **数字对得上**。5090 支持提交：`7810415`（nimlgen，2025-05）。 |
| 二次传播 | [smcleod.net 2026-02](https://smcleod.net/2026/02/patching-nvidias-driver-and-vllm-to-enable-p2p-on-consumer-gpus/)（博客，跟 aikitoria 装）；[QuixiAI/open-gpu-kernel-modules](https://github.com/QuixiAI/open-gpu-kernel-modules)（610.57.04，文档写 5090，**自己验证段是 8×3090**）；[kacper-daftcode/rtx-p2p](https://github.com/kacper-daftcode/rtx-p2p)（反汇编 GSP，**更危险，排除**）。 |
| 开 issue（未关） | [#9 Resizable BAR not working on 5090](https://github.com/aikitoria/open-gpu-kernel-modules/issues/9)（2025-10，Ubuntu 22.04 + 5090）：补丁后 **BAR1 分配失败 / `RmInitAdapter failed` / GSP 起不来**。[#38](https://github.com/aikitoria/open-gpu-kernel-modules/issues/38) 跨代 4090↔6000 Ada 失败；[#43](https://github.com/aikitoria/open-gpu-kernel-modules/issues/43) 混代无 P2P。 |

**卡型合同：** README 和 5090 提交都点名 **RTX 5090 / Blackwell / GB202**，走 BAR1 DMA，不是 NVLink。没有单独写 `sm_120` 字符串——P2P 在内核 RM，不在 nvcc arch。本机 PCI ID `10de:2b85` 就是 5090。

### 7.2 它具体改什么（不是「只拧 canAccessPeer 白名单」）

对比官方同版本树（`NVIDIA/open-gpu-kernel-modules` ↔ aikitoria）：

| 层 | 改了什么 | 595.45.04-p2p | 610.57.04-p2p-v2 |
| --- | --- | --- | --- |
| 交付 | **自己编译的 out-of-tree `nvidia.ko` + uvm/modeset/drm**，`./install.sh`：`rmmod` 四模块 → `make modules` → `make modules_install` → `depmod`。**不是** DKMS 配方，**不是** 只改 userspace | 有 | 有 |
| BIF / RM 默认 | `kernel_bif.c`：**写死** `p2pOverride=0x11`、`forceP2PType=PCIEP2P`、`pcieP2PType=BAR1`（官方默认是 DEFAULT/AUTO，GeForce 上等于关） | 有 | 有 |
| P2P 能力 | `p2p_caps.c` / bus HAL（`kern_bus_gp100.c` 等）让 BAR1 P2P mapping 过门 | 较少 | 更大（+ `kern_bus_tu102.c` / `kern_bus_gm200.c`） |
| DMA / MMU | `nv-dma.c`、`io_vaspace.c`、`nv_gpu_ops.c`、`osmemdesc.c`：对端显存按物理地址 DMA 映射 | 有 | 有 |
| 用户态 | — | 无单独脚本 | 另有 `patch-libcuda-p2p.py`（**补 libcuda**） |
| 顺带私货 | `os-mlock.c`：**实验性** 1G hugepage `cudaHostRegister`，「可能在官方能处理的边上误行为」 | **绑在同一分支** | 同左 |
| 主机前提 | README：**必须** `iommu=pt`（翻译模式 DMA 会坏）；慢则关 ACS；要重启 | 写明 | 写明；610 还要 **整包 userspace/GSP 锁 610.57.04** |

**一句话：** 改的是开源内核模块里的 **BAR1 P2P 选择 + DMA/MMU 映射**，让 CUDA 的 `cudaDeviceCanAccessPeer` 从 0 变成 1。不是 BIOS 开关，不是 `NCCL_P2P_DISABLE`，不是只改白名单布尔。`can=1` 只是门开了；拷贝对不对取决于 BAR1 是否覆盖整帧、IOMMU/ACS、以及 GSP 有没有按官方 SKU 把门再关回去。

回滚**不是**「rmmod 就回官方」：`modules_install` **覆盖** `/lib/modules/$(uname -r)/.../nvidia.ko`。要恢复：重装发行版/官方 `.run` 的 **595.45.04 open** 模块（或事先备份 ko）。GRUB 的 `iommu=pt` **重启后残留**，必须改回 cmdline 再 reboot。Secure Boot 若开着，自编译模块还要签名。

### 7.3 本机只读（2026-09-13 07:47 UTC）

| 项 | 实测 |
| --- | --- |
| 驱动 | **595.45.04** `NVIDIA UNIX Open Kernel Module`，`nvidia.ko` 在 `/lib/modules/5.15.0-25-generic/kernel/drivers/video/nvidia.ko` |
| CUDA | `nvidia-smi` 报 13.2 |
| 内核 cmdline | `root=UUID=... ro`——**没有** `iommu=pt` / `intel_iommu=` / `pcie_acs_override` |
| IOMMU | `/sys/class/iommu` 有 **dmar0–dmar19**（Intel VT-d **翻译模式开着**）。作者自己写：这状态下 DMA **会坏** |
| P2P | `nvidia-smi topo -p2p r`：**全部 CNS**（Chipset not supported）。无 NVLink |
| 拓扑 | GPU0–3 = NUMA0 `NODE`；GPU4–7 = NUMA1；跨域 `SYS`。CPU 是 **2× Xeon**，不是 aikitoria 带宽表那台 EPYC |
| BAR1 | GPU0 等 `Region 1` **32G** prefetchable（ReBAR 看起来已经够大） |
| 作业 | **GPU0 19 158 MiB，PID 285611，`./build-cuda-sm120/h3_cuda_generate --preset nvfp4`**，cafe 864×480 **1s / 2-step**，已跑 7+ min。这就是三路里的 NVFP4，不是闲箱 |

### 7.4 风险清单

| 风险 | 会不会发生 | 后果 | 能否回滚 |
| --- | --- | --- | --- |
| 机器 hang / 进不了系统 | **中。** issue #9：5090 上补丁后 BAR 分配失败、`RmInitAdapter failed`、GSP 起不来。再加 GRUB `iommu=pt` + 可能的 ACS override，启动路径变了 | 8 卡全灭或 SSH 在驱动初始化卡死；机房要 IPMI/人去 | **部分。** 能进系统就重装官方 595 模块 + 改回 GRUB。进不了系统要救援盘/IPMI。不是一键 |
| 驱动装不上、CUDA/NCCL/cuDNN 全挂 | **中高。** `install.sh` 先 `rmmod`；模块与 GSP/userspace 必须锁同一版本。默认分支要升到 **610.57.04 整包**；即使用 `595.45.04-p2p` 也对上本机数字，仍要重编 8 卡生产模块 | 三路开发全部停；金线 78.74 / DiT 31.67 基线作废直到装回 | 重装官方 `.run`/`nvidia-open` **同小版本** 通常能回。失败窗口按小时计 |
| 和现有驱动不匹配 | **默认分支：不匹配。** 本机 595.45.04 ≠ 610.57.04。`595.45.04-p2p` 数字匹配，但 README 仍要 **重启 + IOMMU pt**，且同分支绑了实验 hugepage 路径 | 硬装 610 等于换整栈。混装 595 userspace + 610 模块是已知自杀 | 不匹配则 **按门槛不要试** |
| NVIDIA 许可 / 保修 / 生产箱 | **政策风险确定。** 开源模块本身 MIT/GPLv2，改源码分发说得通。但 userspace/GSP 仍是 [NVIDIA Linux EULA](https://www.nvidia.com/en-eu/drivers/nvidia-license/linux/)：软件 AS-IS、**GeForce 软件「not licensed for datacenter deployment」**、固件禁止翻译/逆向。官方论坛已写 50 系 **不支持 GPU↔GPU P2P** | 出问题 NVIDIA/板卡商可拒软件支持；生产箱政策不允许私货内核当第一刀 | 卸模块回官方，许可问题不会「自动消失」，只是不再继续违规运行 |
| 数据损坏、多卡误写 | **中。** 作者警告 IOMMU 翻译模式下 DMA 失败「very dangerous」。R545 曾 `can=1` 但拷贝是坏的。QuixiAI 写小 BAR1 时 `can`/`topo` 仍 OK、**NCCL 集体挂死**。静默写错显存比 hang 更难查 | 坏视频当「质量回归」；最坏跨卡踩内存 | 回滚驱动停出血。已经写坏的输出不能靠回滚救 |
| 抢三路 / reload 杀进程 | **这次一定会发生。** GPU0 正在 NVFP4 generate。`rmmod`/`install.sh` 会杀光所有 CUDA 进程 | NVFP4 / TAE / host_stage 作业丢；可能留下半截权重缓存 | 杀了就没了。门槛：**有作业只写评估** |
| 重启残留 / 一键卸掉 | **会残留。** 覆盖的 `nvidia.ko` + GRUB `iommu=pt` 重启后还在。没有「一键卸 DKMS」 | 箱变成非官方栈，下一个人不知道 | 「一键」= 重装官方 595.45.04 open 模块 + 还原 GRUB + reboot。要事先备份 ko。做不到「只卸模块、不重启」 |

**收益上限（就算 peer 真变成 1）：** 本机双路 Xeon + 跨 NUMA `SYS`，**不要**把 EPYC 表上的 55 GB/s 抄过来。§3 已写：社区补丁乐观 U=2 DiT ~20–22 s，U=8 480p 仍可能亏。换不来 5–8 s。

### 7.5 试的门槛对照 → 结论

| 门槛 | 这次 |
| --- | --- |
| 补丁源码看过，有明确 unload/回滚 | **半过。** 源码和 `install.sh` 看过。回滚要 **重装官方模块 + 还原 GRUB**，不是卸模块即可 |
| 驱动主版本与补丁声明匹配，否则不要试 | **默认分支不过。** 610 ≠ 595。`595.45.04-p2p` 数字匹配，但绑定重启/IOMMU/实验 hugepage |
| `nvidia-smi` 上没有别人的 generate/cafe | **不过。** GPU0 正在 `h3_cuda_generate --preset nvfp4` |
| 只做最小 `canAccessPeer` / 秒级带宽，失败立刻回滚 | 没资格开步：上面已失败 |
| 不把私货内核写进仓库、不 commit、不改推理 | 遵守。本节只追加文档 |

**结论：有风险，这次不试，因为（1）方案本身就有 hang / 装坏 / 误写 / 许可 / 残留风险，不是「没风险」；（2）GPU0 有正在跑的 NVFP4 generate，reload 会杀光三路；（3）默认补丁钉 610，本机是 595；要上还得改 GRUB IOMMU 并重启生产箱。**

未测：`cudaDeviceCanAccessPeer` 仍按官方 0 / `topo` CNS 计。没有 0→1，没有带宽数，没有回滚（因为没装）。

### 7.6 下一步（按目标选一条，不要为「试试」拆箱）

1. **单请求再快一点：** 做 §3 P1.1+P1.2（去 DeviceSync + BF16 A2A），只跑 **U=2 软件刀**，和 31.67 比。不过门关掉。
2. **等三路（host_stage / TAE / NVFP4）做完、箱空、有 IPMI、先备份官方 ko：** 再单独开窗口评估 `595.45.04-p2p`——仍然要改 `intel_iommu=on iommu=pt`、reboot、只测一对 GPU 的 `can` + 秒级带宽。**不要**把生产 8 卡当第一次实验机；副机更好。
3. **放弃内核私货：** 8 卡突破改口为请求级 DP（吞吐 ×8）。单请求 20 s 级 DiT 认硬件：换 6000D/Pro 或换有 NVLink 的箱。

---

## 附录 A. 本机拓扑 / peer 现状（只读，2026-09-13 07:47 UTC SSH）

来源：本轮只读 SSH（2026-09-13 07:47 UTC）+ `docs/nvidia_build.md` + U=8 日志。与 2026-09-11 一致处不再解释。

| 项 | 值 |
| --- | --- |
| 卡 | 8× GeForce RTX 5090（`10de:2b85`），32 GB，TGP 575 W |
| 驱动 | **595.45.04 Open Kernel Module**（官方，`/lib/modules/5.15.0-25-generic/.../nvidia.ko`） |
| CUDA | `nvidia-smi` 13.2（`nvcc` 13.2.51） |
| CPU / 内存 | 2× Xeon Gold 6530，~314 GiB |
| cmdline / IOMMU | `ro`，无 `iommu=pt`；`/sys/class/iommu` **dmar0–19**（翻译模式） |
| NVLink | 不支持 |
| 拓扑 | GPU0–3 + `mlx5_*` = NUMA 0（`NODE`）；GPU4–7 = NUMA 1；跨域 `SYS` |
| BAR1 | 已是 32 GiB prefetchable |
| `topo -p2p r` | **全 CNS** |
| `peer_access` | U=8 日志：`ok=0 fail=56 peer_a2a=0` |
| 当时作业 | GPU0：`h3_cuda_generate --preset nvfp4`（~19 GiB） |

密码未入库。未改推理代码，未 commit，未装补丁。

---

## 附录 B. 本仓通信合同（防以后再争论）

```164:202:h3_ulysses_comm.cu
    /* GeForce + no NVLink: cudaDeviceCanAccessPeer is usually 0 even on
     * PCIe Gen5. Do not call EnablePeerAccess / MemcpyPeer unless can=1.
     * NCCL can still bounce via SHM/host; that is not the same as P2P. */
    ...
    st->peer_a2a = (n_dev == 2 && st->peer_ok == 2 && st->peer_fail == 0) ? 1 : 0;
```

```468:469:h3_ulysses_comm.cu
    if (dtype_bytes != 4)
        return comm_fail(err, err_size, "ulysses a2a_qkv: only f32 in P1");
```

```3857:3936:h3_dit_w4a8.c
static int uly_sync_rank(...) { cudaDeviceSynchronize(); ... }
static int uly_run_ranks(...) {
    /* enqueue all ranks, then uly_sync_rank each → DeviceSynchronize */
}
```

未改这些文件。密码未入库。

---

## 8. DeepSeek 有没有改驱动

> **日期：** 2026-09-13。只追加，不改 §1–§7 / 附录评估，不改推理代码，不 commit。
> **问法：** 「DeepSeek 是否通过修改驱动来提升 GPU 性能？毕竟官方版本是一个平衡版本。」——和本仓 8×5090 / aikitoria 内核补丁讨论绑在一起时，必须拆开。
> **范围：** 公开一手材料（论文、官方仓库、NVIDIA 文档/博客、可核链的媒体）。不传谣。

**一句话：没有。** DeepSeek 公开挖的是 **用户态**（手写 CUDA/PTX kernel、DeepEP 通信、DualPipe 重叠、FP8）。**没有**公开材料写他们改了 `nvidia.ko` / NVIDIA kernel driver。把这件事和 aikitoria 开 GeForce P2P 绑在一起，是把两类完全不同的工程混了。

### 8.1 一手材料有没有写「改 nvidia.ko」

| 来源 | 写了什么 | 有没有改内核驱动 |
| --- | --- | --- |
| [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) §3.2 | DualPipe；跨节点 all-to-all；**warp specialization**；20 SM / 10 channel；**customized PTX** + chunk auto-tune 减 L2 干扰；FP8 + WGMMA 精度修补；H800 节点内 **NVLink/NVSwitch**、节点间 IB；解码用 **IBGDA** | **无。** 全文无 `nvidia.ko` / kernel driver / open-gpu-kernel-modules。论文里的 “driver” 一词都搜不到 |
| [deepseek-ai/DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) | 权重 + 推理 demo；推 SGLang / vLLM / TRT-LLM | 无驱动补丁 |
| [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP) | 用户态 JIT CUDA kernel；MoE all-to-all；V1=NVSHMEM/IBGDA，V2=NCCL Gin | **无改 nvidia.ko。** 要求官方 CUDA / NVLink / RDMA |
| [DeepEP docs/nvshmem.md](https://github.com/deepseek-ai/DeepEP/blob/main/docs/nvshmem.md) | 标题就叫 **「Configure NVIDIA driver」**：`options nvidia NVreg_EnableStreamMemOPs=1 NVreg_RegistryDwords="PeerMappingOverride=1;"`；或装官方 [GDRCopy](https://github.com/NVIDIA/gdrcopy) `gdrdrv` | **官方模块参数 / 官方旁路模块**，不是 fork `nvidia.ko`。文档自己写：改不了 regkey 就走 CPU-assisted IBGDA |
| [DeepEP docs/legacy.md](https://github.com/deepseek-ai/DeepEP/blob/main/docs/legacy.md) | **Undefined-behavior PTX**：`ld.global.nc.L1::no_allocate.L2::256B` 读 volatile；不对就 `DISABLE_AGGRESSIVE_PTX_INSTRS=1` | 用户态内联汇编，NVCC/PTXAS 编译 |
| [DeepEP #21](https://github.com/deepseek-ai/DeepEP/issues/21) | 有人问「H800 上 PTX 要哪个 **GPU driver**」 | 回复大意：**PTX 和 OS driver 无关**，是 Hopper 上的 cache 修饰符用法 |
| [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | 用户态 JIT GEMM（FP8/FP4、TMA/WGMMA 风格、Mega MoE）；学 CUTLASS/CuTe，不重内核 | 要求官方 CUDA 12.9+、SM90/SM100 |
| [deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA) | 用户态 MLA/MHA kernel；H800 数字 | NVIDIA 还给 SM100 提了 [PR #76](https://github.com/deepseek-ai/FlashMLA/pull/76)——官方当 CUDA 库合作，不当「改驱动客户」 |
| [deepseek-ai/DualPipe](https://github.com/deepseek-ai/DualPipe) | PyTorch + NCCL 的 PP 调度 | 无驱动 |
| [deepseek-ai/DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) | `cuda11.8+torch2.6` 推理配方 | 无驱动。5090 上炸是 **缺 sm_120 kernel**，不是没改驱动 |

**负结果：** 检索范围内没有 DeepSeek 发布过 `open-gpu-kernel-modules` fork、没有「我们改了 nvidia.ko 开了某某门」的 issue/README/论文句。

最容易被误读成「改驱动」的一句，是 DeepEP 自己的 **「Configure NVIDIA driver」**：那是 NVIDIA 文档里就有的 IBGDA **modprobe 开关**，和 aikitoria 改源码强制 `canAccessPeer=1` 不是一类事。

### 8.2 他们实际吹过 / 开源过的层

全部是 **用户态 + 调度**，站在官方 CUDA 栈之上（或之内）：

| 层 | 一手出处 | 干什么 | 是不是改驱动 |
| --- | --- | --- | --- |
| 手写 PTX / 通信 kernel | 论文 §3.2.2；DeepEP `utils.cuh` | warp specialization；TMA/WGMMA 在 DeepGEMM；cache 修饰符减通信对计算 SM 的 L2 干扰 | 否。PTX 是 CUDA 的稳定 ISA，[NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) 自己写：给 library / 手写性能 kernel 用 |
| DeepEP | GitHub + 论文 all-to-all | MoE dispatch/combine；NVLink 域 ↔ IB 域转发；FP8；IBGDA / 后来 NCCL Gin | 否。要求 **节点内已有 NVLink** |
| DualPipe | 论文 §3.2.1 + DualPipe 仓 | 双向 PP，把 A2A / PP 通信藏进另一方向的计算 | 否 |
| FP8 训练 | 论文 §3.3 | 细粒度量化 + Tensor Core 累加精度不够时提到 CUDA core 提升 | 否 |
| H800「软件绕过」 | 论文 §3.2 + 集群描述 | H800 **节点内仍有 NVLink**（他们写 160 GB/s），被砍的是相对 H100 的互联档位。绕法是：token 最多去 4 节点、先 IB 再 NVLink 转发、20 SM 做通信、DualPipe 重叠 | **软件拓扑 + kernel，不是开 SKU 门。** 他们本来就有 P2P/NVLink，不需要 aikitoria |
| IBGDA | 论文解码段；DeepEP nvshmem.md | GPU 直接发 IB 降低延迟 | 官方 GPUDirect 路径 + 官方 regkey，不是改 ko |

H800 不是 5090。论文原句：*「Each node in the H800 cluster contains 8 GPUs connected by NVLink and NVSwitch within nodes. Across different nodes, InfiniBand (IB) interconnects are utilized.」*

### 8.3 「DeepSeek 改驱动」从哪来的

**不是 SemiAnalysis，也不是 机器之心 的一手断言。** SemiAnalysis [DeepSeek Debates](https://newsletter.semianalysis.com/p/deepseek-debates)（2025-01-31）写的是卡数、H800 **互联带宽被砍**、成本叙事；没写他们 fork 了 `nvidia.ko`。本轮也没找到 机器之心 原文声称「改了内核驱动」。

可核的传播链是：

1. **论文自己**（合法）：「customized PTX」「partition 20 SMs」。这是 **kernel 占用 SM**，不是改 Linux 驱动。
2. **Mirae Asset Securities Korea**（2025-01，经 [@Jukanlosreve](https://www.reddit.com/r/NVDA_Stock/comments/1iann4i/mirae_securities_report_on_nvda_and_large_scale/) 传出）：把上面翻译成 “customized parts of the GPU's SMs”“rebuilt everything from the ground up”“PTX level”。
3. **[Tom's Hardware, 2025-01-28](https://www.tomshardware.com/tech-industry/artificial-intelligence/deepseeks-ai-breakthrough-bypasses-industry-standard-cuda-uses-assembly-like-ptx-programming-instead)**：标题写成 **「bypasses CUDA… uses PTX」**，正文「**reconfigured** Nvidia's H800 GPUs」。这是英文圈最大的放大器。PTX **就是 CUDA 栈的一截**，不是绕过驱动。
4. **中文转载**（例：[智源社区「绕开 CUDA 的垄断」](https://hub.baai.ac.cn/view/43021)）：「对 H800 **重新配置**」「从底层重建」。
5. **再变体**（例：[头条「虚拟硬件层」](https://www.toutiao.com/article/7645644893506830883/)）：滑到「解锁**官方驱动和编译器**无法触及的性能」。**「驱动」二字从这里开始和内核模块缠在一起。**
6. **DeepEP #21**：用户自己把 PTX 问成「要哪个 GPU driver」——说明这个混淆在工程圈也存在。
7. **DeepEP 文档标题 “Configure NVIDIA driver”**：扫一眼会以为他们在改驱动；点进去是官方 `NVreg_*`。

口语「改驱动」= 把 **(3)(4)(5)** 的「reconfigure / 绕过 CUDA / 官方驱动」和 **(7)** 的文档标题叠在一起。没有一手仓库支撑「改了 nvidia.ko」。

### 8.4 NVIDIA 官方 / 员工怎么说 PTX

**没有**找到 NVIDIA 官方或具名员工声明「DeepSeek 改了我们的 kernel driver」。找到的官方动作全是把 DeepSeek 当 **CUDA 生态用户**：

| 来源 | 日期 | 说了什么 |
| --- | --- | --- |
| [NVIDIA：Handwritten PTX](https://developer.nvidia.com/blog/advanced-nvidia-cuda-kernel-optimization-techniques-handwritten-ptx/)（Jonathan Bentz / Tony Scudiero） | 2025-07-02 | PTX 是 CUDA 栈最底层的 **支持路径**（inline PTX / `cuda::ptx`）；CUTLASS 自己也手写 PTX；「last resort」，但 **合法、在驱动之上** |
| [NVIDIA 用 DeepSeek-R1 自动生成 kernel](https://developer.nvidia.com/blog/automating-gpu-kernel-generation-with-deepseek-r1-and-inference-time-scaling/) | 2025-02 | 官方工程师用 R1 写 CUDA attention kernel |
| GTC 2025 Dynamo 新闻稿 | 2025-03-18 | 在 Blackwell 上服 DeepSeek-R1；黄仁勋谈的是推理软件栈，**不是**「他们改了驱动」 |
| FlashMLA [NVIDIA PR #76](https://github.com/deepseek-ai/FlashMLA/pull/76) | 2025-08 | 官方给 DeepSeek 仓库提 SM100 kernel |
| 论坛 [“Does DeepSeek use handwritten PTX?”](https://forums.developer.nvidia.com/t/advanced-nvidia-cuda-kernel-optimization-techniques-handwritten-ptx/337880) | 2025-11 | 有人问；**没有**官方工程师在该帖回答「他们改了驱动」 |

NVIDIA 的姿态是：手写 PTX = 官方允许的用户态深挖；DeepSeek = 用他们的卡和栈的客户。这和「闭源驱动是平衡版，大厂都改 ko」对不上。

### 8.5 和本仓 aikitoria / 8×5090：必须拆开

| | DeepSeek 实际在做的 | aikitoria / 消费卡开 P2P（§2.3 / §7） |
| --- | --- | --- |
| 层 | **用户态**：`.cu` / 内联 PTX / NCCL 插件 / NVSHMEM | **内核态**：改开源 `nvidia.ko`，把 GeForce SKU 的 `canAccessPeer` 拧到 1 |
| 硬件前提 | H800/H100 类：**节点内已有 NVLink/NVSwitch，节点间 IB**；`canAccessPeer` 本来就是 1 | 5090：**官方关 GPU↔GPU P2P**，本机 `topo` 全 CNS |
| 合法/可维护 | 官方 CUDA 合同内；跟 toolkit 走；NVIDIA 自己也手写 PTX | 非官方模块 + 锁死 userspace/GSP；EULA / 生产箱风险见 §5、§7.4 |
| 要解决的问题 | H800 互联档位被砍 → 用路由 + 重叠 + 通信 kernel 把 IB/NVLink 吃满 | 消费卡策略门 → 没有合法 P2P 公路 |
| 对本仓 | **该学这一层**：通信体积、dtype、重叠、少 DeviceSync（§3 P1）；不是学「改驱动」 | 和 DeepSeek **无关**。他们不需要这补丁 |

**「官方是平衡版」对在哪、错在哪：**

- **对：** 默认 CUDA C++ / cuBLAS / 通用 NCCL 路径确实不是某张卡、某个通信形状的极限。DeepSeek、CUTLASS、NVIDIA 自己都承认：热点可以下到 PTX / 手写 kernel。官方驱动对 **GeForce 关 P2P** 也是明确的产品策略（§2.1），不是「忘了开」。
- **错：** 「平衡」不等于「内核里藏着一档性能，改 `nvidia.ko` 就能挖出来」。DeepSeek 挖的不是这一档。他们的集群 **已经有** 数据中心 P2P。把「官方折中」理解成「大厂都改驱动」，是把用户态 kernel 工作和内核 SKU 门焊在一起。

**对本仓 8×5090 的硬结论：** 学 DeepSeek = 学用户态通信/kernel（P1.1/P1.2，以及将来合法卡上的 A2A）。**不要**把 aikitoria 说成「DeepSeek 同款」。DeepSeek 没走那条路，他们的硬件也不需要那条路。§7 的「这次不试」评估保持不动。
