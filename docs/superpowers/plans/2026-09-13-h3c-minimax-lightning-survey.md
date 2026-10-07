# RunningHub Lightning vs 本仓 Ulysses：为什么 8×5090 上 U=2/U=8 比单卡慢

> **日期：** 2026-09-13（相对初稿重写：C2 插点已接上，U=2/U=8 cafe 已测）。只读调研，未改推理代码，未 commit。
> **源码树：** `third_party/accel_survey/MiniMax-H3-MultiGPU-Lightning/`（SGLang pin `f8cbf000f4a5`，含 `README_CN.md`）
> **上游页：** https://github.com/RH-RunningHub/MiniMax-H3-MultiGPU-Lightning/
> **本仓实现：** `h3_dit_w4a8.c`（`layers_forward_ulysses` / `uly_run_ranks`）、`h3_ulysses_comm.cu`、`h3_gpu_cuda_sdpa_cudnn.cpp`、`tools/h3_cuda_generate.c`
> **秒数：** `out/verify_turbo_impl/PROGRESS_PHASE2.md`、`out/verify_5090x8_gold/REPORT.md`
> **原理：** [turbo-explainer](./2026-09-13-h3c-turbo-explainer.md) · [principle-matrix](./2026-09-11-h3c-accel-principle-matrix.md) · [pcie-lessons](./2026-09-11-h3c-pcie-multigpu-lessons.md)
> **合同：** 金线默认不动（U=1、`sdpa=cudnn`、8-step INT8）。加速档可以改并行策略。禁止把 RH 的 12.2× 写成「8 卡相对 1 卡」。

**一句话：** RunningHub 在无 NVLink 的 8 卡上「快」，主因是 **50 步蒸成 4 步**（占他们宣传倍数的 95%），多卡只是同一 8 卡上的 **DiT 切分**；他们的卡（RTX 6000D）能走 **PCIe P2P**，激活还是 **BF16**。我们的 8×5090 `peer_access=0`，A2A 只能 **绕主机内存**，再叠加 F32 载荷和每层 `cudaDeviceSynchronize`。所以 **U=2 略亏、U=8 大亏**——不是「多卡理论上不该加速」，是这条公路付不起这次换手。

---

## 0. 先把两套数字放在同一张桌上

本仓 8×RTX 5090，PCIe，无 NVLink，`peer_access=0`。cafe 864×480 5s、8-step、`cfg=1`、`sdpa=cudnn`、seed=`0x11`。

| 臂 | e2e | DiT | 相对金线 DiT | MAE vs 金线 |
| --- | ---: | ---: | ---: | ---: |
| 金线 GPU0（warm） | **78.74 s** | **31.67 s** | 1.00× | — |
| U=2 cafe（warm） | 83.40 s | **34.66 s** | **0.91×（更慢）** | 11.159 |
| U=8 cafe（n=1 冷） | 212.29 s | **137.43 s** | **0.23×（更慢 4.3×）** | 11.159 |

短测（1s / 2-step，同一二进制）：U=2 DiT **12.66 s**，U=8 DiT **106.05 s**。U=8 日志：`peer_access ok=0 fail=56 peer_a2a=0`。

质量：U=2 与 U=8 五帧 MAE **完全相同**（11.159），与当前 binary 无 cache 对金线同组。**不是切分算错。** pad 行已不进 softmax（`seq_real` + `heads_attn=H/U`）。A2A 从 `cudaDeviceSynchronize` 改成 stream event 后，**墙钟几乎没下来**——因为真正的 host drain 不在 A2A 函数里（见 §3.3）。

RH 自己公布的表（8× RTX 6000D，**同样 8 卡**，5s 1344×768 t2va）：

| 配置 | 步数 | 秒 | 相对他们基线 |
| --- | ---: | ---: | ---: |
| MiniMax H3 BF16 | 50 | 348.8 | 1.0× |
| + 后训练加速权 | 4 | 43.0 | **8.1×（占 348.8→28.7 节省的 95.5%）** |
| + Sage2 + Cache-DiT + compile | 4 | 28.7 | 12.2× |

另表（仍是同一 8 卡、同一 4-step）：TP4+U2 **54.0 s** → TP2+U4 **48.2 s**（约 12%）。**他们没有公布「同配方 1 卡 vs 8 卡」。**

---

## 1. 他们做什么 / 我们做什么

先用大白话：一次生成 = 翻译菜单（TE）→ 主厨炒 8 轮（DiT）→ 摆盘（VAE）。多卡只能帮「炒菜」。两边都是这个结构。

### 1.1 对照表

| 维度 | RunningHub Lightning | 本仓现在（U>1） |
| --- | --- | --- |
| 运行时 | Python / SGLang `multimodal_gen`，**每 GPU 一进程** | C 栈，**单进程** `ncclCommInitAll` 管全部卡 |
| 并行配方 | **TP2 × Ulysses4**（`scripts/start.sh` 钉死）。CFG 并行关、Ring=1、PP 关、DP 关 | **TP=1，纯 Ulysses**。U=2 / U=8 已跑。CFG/PP/DP 都没有 |
| 切什么 | TP 切**权**（每卡 28 个头）+ Ulysses 切 **packed 序列行**；注意力前 A2A 换成「全长序列、少头」（7 个头） | 只切序列行。注意力前 A2A：`[S_local, 3HD] → [S_pad, 3(H/U)D]`，输出再反向。U=2 头=28，U=8 头=7 |
| 是不是「8 个进程各跑一条视频」 | **不是。** 一条请求切到 8 张卡上一起算 | 也不是。一条请求、多卡协同 |
| 通信原语 | Ulysses：**NCCL `all_to_all_single`**（`usp.py::_usp_all_to_all_single`），立刻 wait、**不 overlap**。QKV 打成一次 destination-major。U=2 且 TP=1 且 `can_device_access_peer` 才走 CUDA-IPC 直写。**生产 TP2+U4 走不到 IPC。** TP：每块 **all-reduce** | U=2 本想 `MemcpyPeerAsync`，但 `canAccessPeer=0` → `peer_a2a=0` → **NCCL send/recv 拼 A2A**。注释写明：GeForce 无 NVLink 时这是 **SHM / host bounce，不是 P2P** |
| 载荷精度 | 主干 **BF16**，A2A 跟激活走 BF16 | A2A **硬编码 F32**（`dtype_bytes != 4` 直接拒） |
| 注意力核 | 默认 **SageAttention2**（小头也能跑） | 金线 **cuDNN**。cafe 上 U=2/U=8 都建成了 plan（`heads=28` / `heads=7`）；其它 shape 常拒 `heads=H/U` 退 flash |
| TE / VAE | **不并行** | **不并行**（只切 DiT） |
| 权怎么放 | BF16 按 TP 切开；768p 峰值约 60–66 GiB/卡（85GB 卡） | INT8 **每卡一份完整副本**（`h3_dit_w4a8_open_int8` × U）。480p 单卡 32GB 已常驻，复制是显存税，不是容量刚需 |
| 12.2× 从哪来 | **几乎全是少步**，卡数是常量 | 不适用。本仓金线已经是官方 8-step |

### 1.2 RunningHub 切的就是「序列并行 + 一点张量并行」，不是流水、不是多请求

公式（SGLang Diffusion）：

```text
num_gpus = cfg × tp × ulysses × ring
```

RH：`1 × 2 × 4 × 1 = 8`。

- **Ulysses（序列并行 / SP）：** 残差和 MLP 按 token 行切开；注意力必须「人人看人人」，所以先 A2A 把头换手——每张卡拿到**整段序列**，但只算自己那几个头。算完再 A2A 换回来。这和我们 `layers_forward_ulysses` 是同一类切法。
- **TP（张量并行）：** 把线性层的宽切开，每块用 all-reduce 把 hidden 加回来。RH 只用 T=2，因为 PCIe 上 all-reduce 比 A2A 更亏（他们用 TP4+U2 vs TP2+U4 的 12% 证明了这一点）。
- **CFG 并行：** 关。H3 已经是 CFG 蒸馏（`cfg=1`），没有正负两条分支可拆。我们一样。
- **流水线并行（PP）：** 关。不是「TE 在 0 卡、DiT 在 1–7 卡」那种流水。
- **数据并行：** 关。不是 8 个进程各生成一条视频。

通信：**不是**「自己实现的 host bounce API」。他们调的是 NCCL。6000D 工作站卡在无 NVLink 时通常 **`canAccessPeer=1`**，NCCL A2A 走 **PCIe P2P**（卡直接写卡，不进主机内存）。我们这台 5090 消费卡 **`canAccessPeer=0`**，同一份 NCCL 代码会退化成 **GPU → 主机内存 → GPU**。名字都叫 NCCL，路不是一条。

---

## 2. 他们为什么还能「加速」，我们为什么不能

两套机器都是 Blackwell、都无 NVLink、都走 PCIe。差别不在营销，在四件事。

### 2.1 他们宣传的快，多半不是多卡赚的

README 自己写：「12.2× combines step distillation and execution optimizations, including a change in generation step count。」

同一 8 卡：50 步 348.8s → 4 步 43.0s，再叠 Sage/Cache/compile 到 28.7s。**卡数没变。** 真正属于「怎么切多卡」的公开数字只有 TP4+U2 → TP2+U4 的 12%。

所以「他们在 8 张消费级哲学的卡上加速了」有两层意思，必须拆开：

1. **相对 50 步基线快 12×** → 少步 + 近似核。本仓金线已经是 8 步 INT8+cuDNN，这 12× **抄不来，也不该抄进金线**。
2. **相对「只用 1 张 6000D 跑同一条 768p BF16」** → 他们没发这个数。8 卡动机是把 768p 长序列的 DiT 算力切开（4 卡对照里 BF16 已能常驻 57 GiB/卡，不是「不切就得 offload」）。

本仓问的是另一道题：**同一条 8-step INT8 480p，U=2/U=8 能不能比单卡 31.67s 的 DiT 更快。** 实测答案：不能。

### 2.2 他们通信走 P2P，我们走主机内存

| | RH 6000D | 本仓 5090 |
| --- | --- | --- |
| `cudaDeviceCanAccessPeer` | 工作站卡，PCIe 上通常为 1 | 日志 **ok=0 fail=56**（U=8） |
| NCCL A2A 实际路径 | 卡↔卡（PCIe P2P） | **SHM / host bounce** |
| 有效带宽量级 | Gen5 x16 P2P，双向大约几十 GB/s | 多卡挤主机内存 + 根复合体，常掉到约 **10 GB/s 量级** |
| 载荷 | BF16 | F32（字节数 ×2） |

`h3_ulysses_comm.cu` 写得很直白：没 `can=1` 就不要 `EnablePeerAccess` / `MemcpyPeer`；NCCL 仍能通，但「that is not the same as P2P」。

U=2 的 IPC 直写（RH 的 `ipc_a2a.py`、我们的 `a2a_peer_u2`）**两边都走不到**：他们生产是 TP>1，我们是 `peer_a2a=0`。

### 2.3 他们的计算够大，藏得住换手；我们的 480p 藏不住

Ulysses 每层两次 A2A，**立刻等齐**（RH 注释：`waits immediately`；我们每层也是 QKV → A2A → SDPA → A2A → MLP）。换手和计算**不能重叠**。

768p 序列大约是 480p 的 2.3 倍，注意力大约 ×5。同样 150 MB 的 A2A，在 768p BF16 上是「炒菜 10 分钟、传盘子 30 秒」；在 480p INT8 上是「炒菜 4 秒、传盘子 15 秒」。PCIe 税占单步的比例完全不同。

他们还用 Sage，**7 个头照样有核**。我们 U=8 也是 7 个头：cafe 建成了 cuDNN plan，但小头占用和 56 头不是一回事；其它 shape 还常退 flash。

### 2.4 进程模型：他们每卡自己 enqueue，我们一根宿主线程扫 8 张卡

RH：8 个进程，各绑一张卡，NCCL 在进程间集合。没有「for r in 0..7: cudaSetDevice; 同步拷 aux; 等权; DeviceSynchronize」。

我们：`uly_run_ranks` 在**主线程**里对每个 rank 做同步 aux H2D、权 resolve（还 `copy_stream_synchronize`），enqueue 完再对每个 rank `cudaDeviceSynchronize`（`uly_sync_rank`）。`jobs[r].compute_stream = NULL`，所以 `uly_join_*` 是空操作。A2A 改 event **改变不了**「每一相算完先把整机 GPU 排干」这件事。这就是「800 次 sync 改 event、墙钟几乎不动」的根。

早期 `H3_DIT_W4A8_HOST_FWD=1` 还曾让 Ulysses 再慢 6–8×（层间 hidden 走主机）。**已关，不要再开。**

---

## 3. 我们慢的根因（按贡献排序，用现有数字）

下面用 cafe 几何：`S=15973` pad 到 `16000`，`H=56`，`D=128`，50 block × 8 step = **400 次层**，每层 **2 次 A2A**。

### 3.1 第一名：A2A 字节 × 次数 × host bounce（U=8 的主因，U=2 的大半）

每层、每 rank 发出去的量（F32，destination-major，与代码 `chunk` 一致）：

| | U=2（`S_local=8000`，头 28） | U=8（`S_local=2000`，头 7） |
| --- | ---: | ---: |
| QKV A2A 每 rank 发送 | **344 MB** | **151 MB** |
| attn-out A2A 每 rank 发送 | **115 MB** | **50 MB** |
| 每层每 rank 发送合计 | 459 MB | 201 MB |
| 全程每 rank 发送（×400） | **184 GB** | **80 GB** |
| 全程总线合计（×U） | **~367 GB** | **~642 GB** |

没有 P2P 时，这几百 GB 都要 **进主机再出来**。多卡还抢同一条 CPU 内存和 PCIe 根。

回代墙钟（数量级，不是再测的秒）：

- **U=2：** 理想计算 ≈ 31.67 / 2 ≈ **15.8 s**。184 GB ÷ ~12 GB/s ≈ **15 s 通信**。算+传串行 ≈ **31 s**，再加同步/aux ≈ **34.7 s**。与实测 **34.66 s** 对得上。
- **U=8：** 理想计算 ≈ 31.67 / 8 ≈ **4 s**。642 GB 挤 host，有效带宽再掉（跨 NUMA `SYS`）到 ~8–12 GB/s → 通信 **50–80 s**。再加 8 倍宿主串行，落到 **137 s** 不奇怪。短测 2-step 就要 106 s，说明税在**每层**上，不是 cafe 冷启动一次。

早期 explainer 里「U=8 每次 A2A ~33 MB、全程约 1 s」少算了 **QKV×3 和 F32**，还默认了 P2P 50 GB/s。那条 1 秒税**作废**。

**A2A 改 stream event 几乎没掉墙钟：** 通信体积还在，host bounce 还在；event 只去掉了 A2A 函数末尾的 drain，算完仍被 `uly_sync_rank` 排干（§3.3）。

### 3.2 第二名：PCIe 无 P2P（消费卡合同，不是调参能开）

`peer_access=0` 是 5090 + 无 NVLink 的常见结果，不是忘了 `EnablePeerAccess`。代码已经探测了，U=2 的 peer 快路径直接关。

**不要设 `NCCL_P2P_DISABLE`**（和 RH 一样）。但「没关 P2P」≠「有 P2P」。没有 `canAccessPeer`，NCCL 只能绕主机。

U=8 还跨两个 NUMA。同域 U=2（GPU0,1）已经略亏；跨域 8 路 A2A 只会更差。

### 3.3 第三名：每相 `cudaDeviceSynchronize`（不是 A2A 里那 800 次）

`uly_run_ranks` 默认路径：enqueue 全部 rank → 对每个 rank `uly_sync_rank` → `cudaDeviceSynchronize`。三相（QKV / ATTN / MLP）每层都做。

| | U=2 | U=8 |
| --- | ---: | ---: |
| DeviceSynchronize 次数 | 3 × 50 × 8 × 2 = **2400** | 3 × 50 × 8 × 8 = **9600** |

A2A 那 800 次（2 × 50 × 8）已经改成 event，所以墙钟不动。这些 DeviceSynchronize 还在，而且 **挡住「算和下一次通信重叠」**——Ulysses 本身也 overlap 不了 A2A，但至少不该在 A2A 之前再无谓地让宿主空转。

权 resolve 里还有 `h3_cuda_copy_stream_synchronize`（每层每 rank 的 QKV/MLP）。

贡献：U=2 大概 **1–3 s**（回代里 34.66 − 31 的那截）；U=8 放大到十几秒级宿主开销，但盖不过 642 GB bounce。

### 3.4 第四名：头被切碎（U=8 明显，U=2 轻微）

注意力 FLOP 按 `H/U` 除，**总计算量**该降。降的是「56 个粗头」换成「7 个细头」：

- cafe U=2：`tensor_seq=16000 heads=28`，device 0 和 1 各一条 plan → **cuDNN 接住了**。
- cafe U=8：`heads=7`，8 张卡各一条 → 也建成了。用户说的「常拒退 flash」发生在**别的 shape**（例如短测 `seq=5568 heads=7` 一类），不是 cafe 5s 的主日志。
- 7 头占用、tile 效率仍差于 56 头。RH 用 Sage 就是为了躲开「官方融合核不爱小头」。

这一项解释不了 U=2 的 +3 s，也解释不了 U=8 的 +100 s；它是 **U=8 计算侧的附加税**，排在通信后面。

### 3.5 第五名：单线程串行 enqueue + 每层同步 aux H2D

每层、每个 rank：`cudaMemcpy` 调制/norm（`uly_h2d_aux`，同步）、`get_weight`、再 `cudaSetDevice` 发射。U=8 是 U=2 的四倍宿主往返。权重在 warmup 后应已在各自 cache；U=8 cafe 只有冷跑，权首次 H2D 会贵一截，但短测 2-step 已经 106 s，**不是「冷加载一次」能解释的。**

### 3.6 第六名：HOST 路径（历史坑，已关）

`HOST_FWD=1` 让层间 hidden 走主机，Ulysses 再慢 6–8×。默认已关，device-resident。**不要当加速旋钮拧回去。**

### 3.7 第七名：权重复制（显存税，不是这轮墙钟主因）

U 张卡各 `open_int8` 一份完整 DiT。480p 装得下。冷启动 8 份 H2D 会拉长第一次，warm U=2 只比 cold 快 ~1.2 s（35.88 → 34.66），说明**稳态墙钟不是复制权在吃**。复制的真实代价是：8× 显存、8× prefetch，换不到通信更便宜。

### 3.8 一张「税怎么叠到 34.66 / 137」的简表

| 项 | U=2 对 31.67 的贡献（估） | U=8 对 31.67 的贡献（估） |
| --- | --- | --- |
| 计算本应降到 | −15.8 s | −27.7 s |
| host-bounce A2A（F32） | **+15 s** | **+80–110 s** |
| DeviceSync + aux + 串行 enqueue | +2–4 s | +15–30 s |
| 小头 / 偶发 flash | ~0（cafe 已建 plan） | +数秒～十几秒 |
| HOST_FWD | 0（关） | 0 |
| 权复制（warm） | ~0 | 冷跑可能 +数秒 |
| **净结果** | **+3.0 s（34.66）** | **+105.8 s（137.43）** |

U=2：计算省下的几乎被 bounce 通信吃光，再被同步推过单卡。  
U=8：通信体积×争用把「除以 8」变成「乘以 4」。

---

## 4. 建议动作（加速档可改并行；金线默认不动）

### 明确不该再空拧的

- **不要再把 U=8 当主路径调参**（换 event、关 HOST_FWD、换 SERIAL）。公路是 host bounce，拧的是闸门不是车道。
- **不要抄 RH 的 TP2+U4 进 5090 金线。** 他们 T=2 是 85GB BF16 768p 的容量折中；我们 INT8 单卡已常驻，再加 per-block all-reduce 是 PCIe 上最亏的轴。
- **不要用 12.2× 或 explainer 里「U=8 DiT 2.5–4 s」当目标。** 后者假设了 P2P + 低估了 F32 A2A。
- **不要开 Ring / CFG 并行 / HOST_FWD。** Ring 要 FA；`cfg=1` 无第二分支；HOST_FWD 已证明 6–8× 更慢。
- **不要为了「多卡好看」把金线默认改成 U>1。**

### P0 — 值得做的下一刀（不靠 Ulysses）

| # | 做什么 | 改哪里 | 风险 | 相对金线 DiT 31.67 / e2e 78.74 的预期 |
| --- | --- | --- | --- | --- |
| P0.1 | **承认 8×5090 PCIe 上 Ulysses 不是主加速路径**；文档和路线图改口（本文 + explainer 的 2.5–4 s 推断作废） | 只改文档 | 无 | 停止在错误分母上烧机时 |
| P0.2 | **加速档继续走已测刀**：turbo4（少步）+ opt-in `--te-cache` | 已有。HIT 金线 e2e **61.41 s**；turbo4+cache warm e2e **~49.7 s**（`PROGRESS_PHASE2`） | cache 默认仍关；turbo 非金线 | DiT 不靠多卡也能到 25 s 档（turbo4）；e2e 砍的是 TE 15 s |
| P0.3 | TAE 权落到再做 decode（B1 仍缺权+stub） | `h3_tae.c`，仅 `--preset turbo4` | 金线禁止 TAE 成片 | 砍 VAE ~10 s，与 Ulysses 正交 |

P0 的下一刀是 **TE 缓存进加速档工作流** 和 **TAE**，不是再跑一轮 U=8。

### P1 — 若还要救 U=2（实验臂，不是主路径）

前提：`peer_access` 仍为 0。目标从「接近 15.8 s」改成「**能否低于 31.67 s**」。

| # | 做什么 | 改哪里 | 风险 | 预期（相对 31.67） |
| --- | --- | --- | --- | --- |
| P1.1 | 去掉 `uly_run_ranks` 里每相 `cudaDeviceSynchronize`；只用 event 在 A2A 前等待该相算完。A2A 已是 event，不要改回去 | `h3_dit_w4a8.c` `uly_sync_rank` / `uly_run_ranks` | 曾 IMA；须按 rank 分 cuDNN plan（已做） | U=2：**−1～3 s**（34.66 → ~32–33）。**单独不够翻正。** U=8 同理只削宿主税 |
| P1.2 | A2A 载荷改 BF16（pack 前 downcast，unpack 后 upcast） | `h3_ulysses_comm.cu` 放开 `dtype_bytes==4` | **非位级等价**，须单独过 MAE 门 | 通信字节减半。U=2 回代：通信 ~7–8 s + 计算 15.8 + 宿主 ~2 → **DiT ~25–27 s**（约 0.8×）。e2e ~72–74 s。U=8：642/2 GB 仍可能 **DiT 50–80 s**，仍慢于单卡 |
| P1.3 | aux H2D 改 async + 复用，不要每层同步 `cudaMemcpy` | `uly_h2d_aux` | 低 | 含在 P1.1 的 1–3 s 里 |
| P1.4 | **不要**再赌 5090 上 `MemcpyPeer` | — | `can=0` 时必失败 | 0 |

P1 做完，**U=2 有机会略快于金线 DiT**（大约 25–28 s，不是 16 s）。**U=8 仍然救不回来。**

### P2 — 加速档可以换的并行，而不是继续拧 Ulysses

| # | 策略 | 改哪里（概念） | 风险 | 预期 |
| --- | --- | --- | --- | --- |
| P2.1 | **请求级数据并行**：8 卡各跑一条完整生成，卡间零 A2A | generate / serving 调度，不进金线单请求 | 单条延迟不变 | **吞吐 ×8**，e2e 单条仍 ~79 s。这是 8×5090 在 `peer_access=0` 上唯一稳的多卡用法 |
| P2.2 | VAE tile / 时间片分卡 | VAE 路径，加速档 | 拼回要正确 | 砍 e2e 里 ~10 s 的一部分；不碰 DiT 31.67 |
| P2.3 | 有 P2P 的机器（6000D / 带 NVLink 的箱）再开 U=2/U=4 | 同一套 Ulysses，探测 `peer_a2a=1` | 换箱 | U=2 才接近旧估计：DiT **~20–22 s**（15.8 + 4–5 s P2P）。U=8 在纯 PCIe 上仍险，768p 才可能藏住 |
| P2.4 | 768p / 更长序列再评估 U=4（同 NUMA） | 加速档 | 480p 已证伪 U=8 | 序列越长，计算/通信比越好；**480p 不要用 U=8 验证「多卡该快」** |

### 预期汇总（诚实）

相对金线 **DiT 31.67 s**：

| 动作 | U=2 | U=8 |
| --- | --- | --- |
| 现状 | 34.66（亏） | 137（大亏） |
| 只去 DeviceSync（P1.1） | ~32–33（仍可能亏或打平） | 仍远慢于 32 s |
| + BF16 A2A（P1.2） | **~25–28（可能首次快于金线）** | 仍 ≫ 32（救不了 PCIe 8 卡 bounce） |
| 若未来有 P2P | ~20–22 | 乐观 10–15，不承诺；480p 仍可能亏 |
| 搬 RH 的 Sage/4-step/compile | 换配方，不是救 Ulysses；进加速档另册 | 同左 |

e2e：只切 DiT 时，U=2 理想也到不了 40 s 以下（TE+VAE 仍 ~26 s）。要 50 s 以下走 **turbo4 + TE cache**（已测 ~49.7 s），不走 U=8。

---

## 5. 结论：8×5090 PCIe 上 Ulysses 还值不值得当主路径

**不值得。**

1. **切分是对的，路是错的。** 和 RH 同一类 Ulysses（行切 + 头换手 + packed QKV 一次 A2A）。MAE 11.159 与 U 无关，说明置换没有把图画坏。慢在 **5090 不能 P2P + F32 体积 + 每层屏障**。
2. **RH 不能当「8 张消费卡上 Ulysses 一定更快」的证人。** 他们的 12.2× 是少步；他们的卡能 P2P；他们的活是 768p BF16，计算藏得住 A2A。三条我们都不占。
3. **U=2 是唯一还值得做一次性能迭代的 U**（P1：去 DeviceSync + BF16 载荷），预期最多把 DiT 从 34.66 拧到 **25–28 s**，e2e 仍在 70 s 出头。过门可以留作加速档开关，**默认保持 U=1**。
4. **U=8 在这台机器上不要再当主路径。** 642 GB host bounce 不是工程微调能填的坑。8 卡的正当用法是 **多请求 DP**（吞吐）或等 **有 P2P / 更长序列** 的箱子。
5. **金线继续单卡。** 加速档的下一刀是已经测过的 TE cache / turbo4，以及还没权的 TAE——不是再空拧 Ulysses。

未改代码，未 commit。密码未入库。
