# h3.c 五方案加速机制深潜（RH Lightning × 多卡 6000 专章）

> **后续优化方案见** [2026-09-11-h3c-next-opt-plan.md](./2026-09-11-h3c-next-opt-plan.md)  
> **For agentic workers:** 本文是浅层普查的**机制加深**，不是实现计划。禁止改 `h3_cuda_generate` 默认 / 单卡金线 KPI / 默认 `sdpa=cudnn`。不 commit。密码不入库。  
> **浅层入口：** [2026-09-11-h3c-accel-survey-multigpu-5090.md](./2026-09-11-h3c-accel-survey-multigpu-5090.md)  
> **无 NVLink 多卡链路 + 5090 可抄清单：** [2026-09-11-h3c-pcie-multigpu-lessons.md](./2026-09-11-h3c-pcie-multigpu-lessons.md)  
> **少步为何好看（CFG 蒸馏 vs 4-step 学生 vs draft+refine）：** [2026-09-11-h3-few-step-quality.md](./2026-09-11-h3-few-step-quality.md)  
> **旧深潜（框架 vs 本仓）：** [2026-08-25-h3c-vs-sglang-vllm-tech-deepdive.md](./2026-08-25-h3c-vs-sglang-vllm-tech-deepdive.md) · [2026-08-25-h3c-multi-gpu-arch-speed.md](./2026-08-25-h3c-multi-gpu-arch-speed.md)

**Goal:** 把 FastVideo / Sol Super Accel / VDN / Fal H3 Max / RunningHub Lightning 拆到计算图、注意力核、步数、cache、多卡切分点与质量门；并单独讲透 **RH Lightning 在多卡 RTX 6000D 上效率从哪来**（禁止把 4-step 倍数算进「多卡效率」）。

**口径：** 各方秒数只在其自己的分母下引用。本仓金线仍是 **INT8 ConvRot + cuDNN SDPA + 8-step + 864×480 有声**；5090 cafe warm **~66s（DiT ~31.4s）**。未在本仓复测的数字一律标来源，不编造。

**源码锚点（本轮已 clone / 摘录，权重未下）：**

| 方案 | 本地树 |
| --- | --- |
| RH Lightning | `third_party/accel_survey/MiniMax-H3-MultiGPU-Lightning/`（含 pinned SGLang `f8cbf000f4a5`） |
| FastVideo | `third_party/accel_survey/FastVideo/` |
| VDN | `third_party/accel_survey/vdn-minimax-h3/` |
| Sol Super Accel | `third_party/accel_survey/sol-engine-excerpts/`（非整仓） |
| Fal | 无仓 |

---

## 0. 结论先行（技术，非营销）

1. **「多卡 6000 快 12×」不是多卡效率。** RH 的 12.2×（348.8s → 28.7s）是 **同一套 8× RTX 6000D** 上、**50-step BF16 → 4-step 蒸馏** 为主、Sage2+Cache-DiT+compile 为辅的整栈倍数。两边都已经是 8 卡。仓内自己写明：「12.2× combines step distillation and execution optimizations, including a change in generation step count。」把 12.2× 写成「8 卡相对 1 卡」或「相对本仓 66s」都是类别错误。

2. **真正属于「多卡拓扑」的公开数字只有一档、且不含步数变化：** 同一 8 卡、同一 4-step 配方下，15s 768p 竖屏 t2va 从 **TP4+Ulysses2 的 54.0s → TP2+Ulysses4 的 48.2s（约 12%，少 ~14 GiB）**。这才是 PCIe 无 NVLink 上「怎么切」的证据。RH **没有公布** 同配方单卡 6000D 秒数，因此「8 卡相对 1 卡加速比」**未知**，本文不编。

3. **卡型不是 Ada RTX 6000，也不是带 NVLink 的全球版 RTX PRO 6000 96GB。** README / `nvidia-smi` 实测钉死：**NVIDIA RTX 6000D，Blackwell `sm_120`，单卡可用显存 85,651 MiB，PCIe Gen5 x16，无 NVLink，SM 2430 MHz，600 W。** 这是中国市场裁剪版 GB202 工作站卡（约 84GB GDDR7），与消费卡 5090 **同互连哲学（PCIe、无 NVLink）、同架构代（sm_120）**，但显存差 2.6×、算力/带宽也不是 5090 的平移。

4. **多卡在 6000D 上快，主因是 DiT 算力叠加 + BF16 常驻，不是「避免 offload」。** 4 卡对照里 BF16 50-step 峰值 57.1 GiB/卡，已小于 85GB；8 卡 TP2+U4 峰值 60–66 GiB/卡。他们用 8 卡不是因为单卡装不下，而是因为 768p 长序列的注意力/MLP 要切。TE / VAE / mux **不在** TP2+Ulysses4 的主切分里；H3 金线又是 cfg 蒸馏（`cfg=1`），**CFG 并行闲置**；**PP 未开**。

5. **对本仓 2×/4×5090 保效果：只抄 Ulysses/SP 切 packed 序列，不抄 4-step / Sage2 / Cache-DiT / compile。** 本仓 INT8 已在 **单卡 32GB 常驻** 864×480 8-step；5090 多卡的合法故事是「同轨迹砍 DiT 墙钟」，不是「显存换常驻」。Amdahl 托底：e2e ~66s ≈ TE 12–14 + **DiT 31.4** + VAE ~10.5；只切 DiT 时 e2e 理想上界远小于卡数倍。PCIe 上 **少 TP、多 Ulysses** 仍然划算；TP 的 per-block all-reduce 在无 NVLink 上是最差轴。

6. **五方案没有一个是「无损加速本仓 66s」的补丁。** 墙钟数量级下降必须换配方（少步 / draft+refine / 稀疏学生权）。金线要保 MAE∩有声，就只能做同轨迹并行，并接受 TE+VAE 地板。

---

## 1. 专章：RH Lightning × 多卡 6000 效率从哪来

### 1.1 卡型鉴定（不要猜错）

证据链全部来自仓内 README / README_CN 的「Measured hardware parameters (from `nvidia-smi`)」，不是营销页推断：

| 项 | 实测记录 |
| --- | --- |
| 型号字符串 | **NVIDIA RTX 6000D** |
| 架构 | Blackwell，**`sm_120`** |
| 可用显存 | **85,651 MiB**（`nvidia-smi` 口径） |
| 功耗上限 | **600 W**（默认 600 W，可调下限 200 W） |
| 互连 | **PCIe Gen5 x16，无 NVLink** |
| 时钟 | SM 2430 MHz / memory 12,481 MHz |
| 拓扑 | 8 卡跨两个 NUMA（同域 NODE，跨域 SYS） |
| 主机 | 2× EPYC 9354，~1007 GiB RAM |

**排除：**

| 容易猜错的卡 | 为什么不是 |
| --- | --- |
| RTX 6000 Ada | Ada `sm_89`、48GB；与 `sm_120` / 85GB 矛盾 |
| 全球版 RTX PRO 6000 Blackwell Workstation 96GB | 部分 SKU 带 NVLink / 96GB；RH 明确 **无 NVLink** 且 smi 可用 ~83.6 GiB，对齐中国裁剪版 **6000D ~84GB GDDR7** |
| RTX 5090 | 同 `sm_120`、同无 NVLink，但 **32GB**、消费级带宽/功耗；RH 数字不能直接当 5090 秒数 |

启动脚本把这套部署写成硬编码假设：`scripts/start.sh` 注释「Mirrors the validated 8x RTX 6000D deployment: TP2 + Ulysses4」，`CUDA_VISIBLE_DEVICES=0..7`，`--num-gpus 8 --tp-size 2 --ulysses-degree 4`。

### 1.2 并行拓扑（切什么张量、什么通信）

SGLang Diffusion 的乘积公式（`sglang/docs/docs/sglang-diffusion/parallelism.mdx`）：

```text
num_gpus = cfg_parallel_degree × tp_size × sp_degree
sp_degree = ulysses_degree × ring_degree
```

RH 默认配方：**cfg=1（闲置）、tp=2、ulysses=4、ring=1、pp=1、dp=1 → 8 GPU。** 没有层切 PP，没有 CFG 双分支并行。

#### 1.2.1 各轴切什么

| 轴 | RH 是否开 | 切的维 | 通信 | H3 上的形状 |
| --- | --- | --- | --- | --- |
| **Ulysses SP** | **4** | 残差流按 **packed 序列行** 切；注意力内部 all-to-all 改成 **头维切** | **每层注意力两次 all-to-all**（输入 QKV + 输出）；NCCL `all_to_all_single`，2 卡时可走 CUDA-IPC | `[B, S/(U·R), H/T, D] → A2A → [B, S/R, H/(T·U), D]` |
| **TP** | **2** | **权**（Column/Row Parallel Linear）+ 投影后的 **头** | **每 transformer 块 all-reduce**（RowParallel 输出） | QKV `MergedColumnParallelLinear`；`out_proj` / MLP `fc2` `RowParallelLinear` |
| **Ring SP** | 关 | 注意力内部按 **行** 切，K/V 邻接旋转 | 邻居 hop，可与计算重叠 | H3 packed varlen 支持，但 RH 启动未开 |
| **CFG 并行** | 关 | 正/负分支 | 每步一次 combine | H3 金线 cfg 蒸馏，**无第二分支可切** |
| **PP（层切）** | 关 | 块流水 | send/recv 激活 | 未用于 Lightning |
| **DP** | 关 | 请求副本 | 请求路径零通信 | serving 吞吐轴，与单请求延迟无关 |
| **VAE 并行** | 非主叙事 | decode tile / 时间 chunk | all-gather 拼回 | H3 VAE 类有 tiling 开关；RH 宣传层不靠它吃饭 |
| **TE / encoder** | 非主叙事 | encoder DP 可与 TP 组合 | 启动期为主 | 文本编码器 ~46GB BF16；6000D 85GB 可常驻，**不是** 8 卡的延迟理由 |

Ulysses 与 TP **串行组合、都碰头维**：先 TP 切投影权（每卡 `H/T` 个头），再 Ulysses 切 **TP-local 头**。H3 `num_attention_heads=56` 的合法性：

```
H % T == 0
(H / T) % U == 0
PACKED_SEQUENCE_ALIGNMENT(64) % (U × R) == 0
```

因此 **TP2+Ulysses4 合法**（28 % 4 = 0），**TP4+Ulysses4 非法**（14 % 4 ≠ 0）。这不是口味，是 `MiniMaxH3DiTModel._validate_sequence_parallel_config` 硬约束。

行切顺序（`build_rope_cache` / `forward` 注释）：**Ring 外层连续块，Ulysses 内层**。RH 的 R=1，于是每卡 residual 持有 `S/4` 行、`H/2=28` 个 TP-local 头；进注意力前 A2A 变成全序列、`28/4=7` 个头。

#### 1.2.2 集合通信实现（代码级）

初始化：`initialize_model_parallel()`（`runtime/distributed/parallel_state.py`）按 `"tp-sp-pp-cfg-dp"` 生成 RankGenerator；SP 子组再由 `set_seq_parallel_pg_by_sp_groups()`（`parallel_groups.py`）拆成 **Ulysses 连续 rank 组** 与 **Ring 跨步组**。文档原话：Ulysses 要 full-bisection，应留在节点内 NVLink；**RH 机器没有 NVLink**，所以 Ulysses 流量走 **PCIe peer**。他们仍然选 U=4 而不是把 8 卡全给 TP，因为下一小节的通信形态更差的是 TP。

注意力内 A2A：

- `_usp_all_to_all_single()` → `torch.distributed.all_to_all_single(..., group=ulysses_pg)`
- H3 packed 路径：`_minimax_h3_attention_core_impl` 调 `_usp_input_all_to_all_packed_qkv(q,k,v)`，把三次输入交换打成 **一次 destination-major 集体**（`pack_qkv_destination_major`），再 `_usp_output_all_to_all`
- 2 卡且 P2P 可用时，`SGLANG_DIFFUSION_IPC_A2A=1`（默认）用 CUDA-IPC 直写 peer staging，**置换不是归约，与 NCCL 位级相同**

TP 通信：`tensor_model_parallel_all_reduce()` → `get_tp_group().all_reduce`。SGLang 文档把 TP 标成跨互连 **最差轴**：「hidden-sized reduction per transformer block… Highest frequency, no overlap, already ~70% of sharded kernel time on NVLink。」PCIe 上只会更差。这就是 RH 用 **小 TP、大 Ulysses** 的机制原因，而不只是「显存不够」。

#### 1.2.3 PCIe vs NVLink 假设

RH **显式按无 NVLink 机器选拓扑**。同一 8 卡上 TP4+U2 → TP2+U4 的 12% 收益，就是这条假设的实验。SGLang 通用建议「Ulysses 留 NVLink」在这里被改写为：「没有 NVLink 时，宁可 Ulysses 吃 PCIe all-to-all，也不要把 rank 数堆到 TP all-reduce 上。」

2 卡特例：IPC A2A 比 NCCL rendezvous 更适合 PCIe 对等；**4 卡 Ulysses4 仍走 NCCL all-to-all**（IPC 路径要求 ulysses world_size==2）。

### 1.3 为何多卡更快：算力、显存、offload、Amdahl

把「8 卡 28.7s」拆成四个独立问题。

#### A. 算力叠加（是，但只作用于 DiT 可切部分）

Ulysses 把 packed 序列的 SDPA 从 `O(S² · H)` 摊到 `U` 张卡上（每卡算全序列、少头）；MLP/其余 GEMM 仍在 **序列分片** 上做，相当于按行切激活。TP 把 GEMM 的 K 维/头维切开，但每块要 all-reduce 还原 hidden。

**不是** batch 维数据并行（单请求 batch=1）。**不是** 8 路 CFG。

#### B. 显存换常驻（6000D 上不是主因）

| 配置 | 峰值/卡（RH 记录） | 相对 85GB |
| --- | ---: | --- |
| 8× TP2+U4 | 60–66 GiB | 常驻，有余量 |
| 4 卡 BF16 50-step | 57.1 GiB | 仍常驻 |
| 4 卡 INT8-ConvRot | 36.8 GiB | 常驻 |
| 4 卡 NVFP4 | 33.2 GiB | 常驻 |
| 4 卡定型（turbo+Sage2+Cache+compile） | 37.3 GiB | 常驻 |

BF16 主干在 6000D 上 **单卡也装得下 DiT 工作集**（4 卡 57 GiB 的对照说明权+激活在 ≤4 卡拓扑下已 <85GB）。8 卡的动机是 **768p 序列的注意力墙钟**，不是「不切就得 offload」。对比：SGLang cookbook 的 **2×5090 BF16** 必须 layerwise（20 层 resident + 主机 ~384 GiB RAM），50-step 768p **559.67s**——那才是「显存换可跑」，和 Lightning 的 28.7s 不在一个问题里。

RH 主干 **BF16 权、无 INT8/NVFP4**；VAE 上游精度。Sage 内部量化注意力 ≠ 权量化。

#### C. 避免 offload/layerwise？（这台机器上基本否）

`scripts/start.sh` **没有** `--layerwise-offload-components`。6000D 85GB + 8 卡切分后，DiT/VAE/TE 走常驻。offload 是 5090/4090 cookbook 的故事，不要倒灌进 6000D 叙事。

#### D. Amdahl：哪些并行、哪些仍串行

一次 t2va 请求的粗图（SGLang `multimodal_gen` H3 管道）：

```
TE (Qwen3-VL) ──► pack [text|audio|video] ──► DiT denoise loop ──► VAE decode ──► mux
     ▲ 常驻/可 encoder-DP                         ▲ TP×Ulysses                     ▲ tile，默认不切进 TP2+U4
     │ 单请求几乎串行                             │ 每步 50 block × (attn A2A + MLP AR)
     └─ 不在 12% 拓扑对照的变量里                 └─ Lightning 多卡刀只砍这里
```

| 阶段 | RH 8 卡是否并行 | 对本仓含义 |
| --- | --- | --- |
| TE | 基本串行（encoder DP 是副本/切 batch，单请求无收益） | 5090 金线 TE 12–14s **不会**因 Ulysses 下降 |
| DiT | **是**：序列/头/权 | 唯一合法的同轨迹杠杆 |
| VAE | 可选 tile/chunk；Lightning 宣传不靠它 | 5090 VAE ~10.5s 仍是地板，除非另做 decode 并行 |
| mux / 写盘 | 串行 | 噪声 |
| Cache-DiT | 步/块跳过，**不是**卡间并行 | 改计算；4-step 上可跳空间极窄 |
| compile | 降单卡核开销 | 改数值；本仓静态 C 无此刀 |

**没有单卡 6000D 对照，不能把 28.7s 当成「8 卡加速比的分子」。** 只能说：在已经 8 卡的前提下，再换拓扑只再抠 12%。

### 1.4 与 4-step / Sage2 / Cache-DiT / torch.compile 解耦（禁止混账）

RH 自己的 5s 1344×768 t2va、**同一 8×6000D** 三行表：

| 配置 | 请求步数 | 生成耗时 | 相对 50-step 8 卡基线 |
| --- | ---: | ---: | ---: |
| MiniMax H3 BF16 baseline | 50 | **348.8 s** | 1.0× |
| + RH 后训练加速权 | 4 | **43.0 s** | **8.1×** |
| + SageAttention2 + Cache-DiT + torch.compile | 4 | **28.7 s** | **12.2×** |

分账（只使用表内数字，不引入未测单卡）：

| 增量 | 秒数变化 | 倍数 | 占「348.8→28.7 节省」 |
| --- | ---: | ---: | ---: |
| **步数蒸馏 50→4**（权变了） | 348.8 → 43.0（−305.8 s） | **8.1×** | **95.5%** |
| Sage2 + Cache-DiT + compile（仍 4-step） | 43.0 → 28.7（−14.3 s） | **再 ×1.50** | **4.5%** |
| **卡数** | 两行都是 8 卡 | **1.0×（此表无卡数变量）** | **0%** |

理论 NFE 比 50/4=12.5，实测蒸馏臂只有 8.1×：说明 4-step 并没有把非 DiT 开销按 12.5 砍掉（TE/VAE/调度/通信是步数不敏感项），且蒸馏权改变计算密度。这与 Amdahl 一致。

**拓扑 12% 是另一张表**（15s、4-step、8 卡内部 TP4+U2 vs TP2+U4），RH 原文：「This result is independent of the 12.2× comparison above。」禁止与 12.2× 相乘。

质量合同（README 原文）：BF16 权 **≠** 无损；Sage 内部量化 + 蒸馏 + Cache-DiT 都改计算。验收清单含主体/细节/运动/遵循/**音画同步**，**无公开 MAE 表**。RH 自训权 **未公开**；公开复现用 larryvrh turbo v4 EMA 或 LightX2V ref2v 8-step，**不是** 12.2× 那次实验。

Cache-DiT 默认（SGLang `CacheDiTConfig`）：`Fn_compute_blocks=1`，`Bn=0`，`max_warmup_steps=4`，`residual_diff_threshold=0.24`，`max_continuous_cached_steps=3`，**TaylorSeer 默认关**（少步蒸馏不适合）。H3 上 Cache-DiT 还 patch 了 SP/TP 组上的 residual 均值 all-reduce（`patched_similarity`），否则分片 residual 会误判。`MiniMaxH3DiTBlock.preserve_input_for_cache_dit`：首个 gated residual 不能 in-place，以免 Pattern 探测恒 0——这是已知工程坑，不是营销。

`torch.compile`：RH `--performance-mode speed` + `--enable-torch-compile`。SGLang cookbook 对 H3 **明文 compile 改数值**，不作 consistency GT。本仓静态 C 管道没有对等物。

SageAttention2：`sage_attn.py` 调 `sageattention.sageattn`；pin `d9704247`。这是 **量化注意力**，不是本仓 Tier-1 `sage3`（FP4 Blackwell 插件）。本仓 sage3 已测 DiT −17%、MAE~27 **不过门**；RH Sage2 没有对本仓尺子的 MAE。

### 1.5 关键函数 / 文件

路径均相对 `third_party/accel_survey/MiniMax-H3-MultiGPU-Lightning/`。

| 职责 | 文件 | 符号 |
| --- | --- | --- |
| 8 卡启动 | `scripts/start.sh` | `--num-gpus 8 --tp-size 2 --ulysses-degree 4 --attention-backend sage_attn --enable-torch-compile` |
| pin | `sglang/RH-PIN.md` | SGLang `f8cbf000f4a5bfd86d3fb7c1e2d6c8fb12339d0e` |
| 并行组 | `sglang/python/sglang/multimodal_gen/runtime/distributed/parallel_state.py` | `initialize_model_parallel`，`get_ulysses_ctx`，`get_tp_group` |
| Ulysses/Ring 子组 | `.../distributed/parallel_groups.py` | `set_seq_parallel_pg_by_sp_groups` |
| 通信原语 | `.../distributed/communication_op.py` | `sequence_model_parallel_all_to_all_4D`，`tensor_model_parallel_all_reduce` |
| A2A 实现 | `.../runtime/layers/usp.py` | `_usp_all_to_all_single`，`_usp_input_all_to_all_packed_qkv`，`_usp_output_all_to_all`，IPC 分支 |
| TP Linear | `.../runtime/layers/linear.py` | `ColumnParallelLinear`，`RowParallelLinear`，`MergedColumnParallelLinear` |
| H3 注意力+Ulysses | `.../runtime/models/dits/minimax_h3.py` | `_minimax_h3_attention_core_impl`，`MiniMaxH3Attention`，`MiniMaxH3DiTBlock.forward`，`_validate_sequence_parallel_config`，`build_rope_cache` |
| packed 对齐 | `.../configs/models/dits/minimax_h3.py` | `MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT = 64` |
| denoise | `.../pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py` | `minimax_h3_denoise_loop`（cfg 蒸馏，无双分支） |
| Sage2 | `.../runtime/layers/attention/backends/sage_attn.py` | `SageAttentionImpl` |
| Cache-DiT | `.../runtime/cache/cache_dit_integration.py` | `enable_cache_on_transformer`，`CacheDiTConfig`，`patched_similarity` |
| VAE | `.../runtime/models/vaes/minimax_h3.py` | `MiniMaxH3VideoVAE`（encoder 常驻；decoder tile 可选） |

### 1.6 数字与配方（只引用仓内）

**5s 文生，8×6000D，1344×768，t2va，warmup 后 generation latency，不含 load/compile/排队/下载：**

- 50-step BF16：348.8 s
- 4-step RH 权：43.0 s
- 4-step + Sage2 + Cache-DiT + compile：28.7 s
- 有声：客户端强制探测 video+audio 双流；H3 24 fps、`17n+5` 桶，5s → 124 帧

**15s，8 卡 TP2+Ulysses4，768×1344：**

| 任务 | 步数 | 延迟 |
| --- | ---: | ---: |
| t2va | 4 | 48.2 s |
| ref2va | 4 | 73.0 s |
| t2va 高动态 | 8 | 89.3 s |
| ref2va 高动态 | 8 | 134.3 s |

高动态建议 8-step：蒸馏 4-step **不是**无条件更快更好。

**没有：** 单卡 6000D 同分辨率同步数秒数；本仓 864×480 8-step INT8 在 6000D 上的秒数；Sage2 单独 ablation；Cache-DiT 单独 ablation。

### 1.7 落到本仓 2× / 4× RTX 5090（保效果、不改采样）

| 维 | RH 6000D | 本仓 5090 |
| --- | --- | --- |
| 互连 | PCIe Gen5，无 NVLink | 消费卡同样无 NVLink（同哲学） |
| 显存 | ~85GB，BF16 常驻 | **32GB**；BF16 H3 要 offload（cookbook 2×5090 559s @50-step 768p） |
| 权 | BF16 + 4-step LoRA | **INT8 ConvRot 已常驻**，8-step 金线 |
| 注意力 | Sage2 默认 | **cuDNN dense** 默认；sage3 MAE FAIL |
| 步数 | 宣传 4 | **8，禁止默认改 4** |
| compile / Cache-DiT | 开 | 默认关；compile 无 C 对等物 |

**仍然划算的通信模式（同 8-step INT8+cudnn，KPI 另册）：**

1. **2×5090：纯 Ulysses2，TP=1。** 56%2=0，64%2=0。每层两次（或 packed 一次）A2A；2 卡可走 IPC。不切权 → 无 per-block all-reduce。INT8 权每卡一份副本（32GB 上 864×480 已证明单卡常驻，副本是显存税但换的是算力）。  
2. **4×5090：纯 Ulysses4，或不得已时 TP2+Ulysses2。** 优先 Ulysses。TP2+Ulysses4 需要 8 卡。  
3. **不要** 在 2/4 卡 PCIe 上拉高 TP：RH 自己的 8 卡对照已经显示少 TP 更快。  
4. **不要** 默认开 Ring（H3 ring 还要求 FA backend，与 cuDNN 金线冲突）。  
5. **VAE/TE 另刀**：Ulysses 不自动加速它们。若 e2e 要明显低于 66s 量级，需要独立的 VAE tile 并行（FastVideo 有 bit-exact chunk 调度可参考），TE 单请求几乎无并行收益。

**机制预期（非承诺、非实测）：** 只切 DiT 时，通信税会吃掉理想 2×。5090 cafe DiT ~31.4s；2 卡若落到 ~20s 量级已算拓扑赢；e2e 仍被 TE+VAE 托在 ~45–55s 带。任何「2 卡 → 33s」的口号都忽略了 Amdahl。未测不算。

**质量门：** cafe 同 seed，相对 **单卡 5090 cudnn** PPM MAE ∩ 有声。不过门则 SP 留实验分支。禁止把 Sage/Cache/4-step 叠进这扇门。

---

## 2. 五方案机制深潜

浅层表停在「一句话主机制」。下面按统一七轴写：计算图、注意力核、步数/蒸馏、cache 判据、多卡切分点、质量门、5090 可移植性。

### 2.1 FastVideo FastH3

**计算图。** 基座 H3 ~49 次 DiT forward；FastH3 用 DMD2（frozen teacher + critic）蒸成 **5 个 sigma 点 = 4 次 Transformer forward**。推荐 ckpt VSA / Data-Free（只吃 prompt）。Launcher：`tp_size=1`，`sp_size=num_gpus`，多卡默认可 `use_fsdp_inference`（切 DiT 权副本），TE/VAE **pin CPU offload**。可选 `vae_parallel_decode` + strategy `gather`（时间 chunk 轮询，拼回 rank 上 bit-exact）。

**注意力核。** `VIDEO_SPARSE_ATTN_H3`：packed `[text|cond|audio|video]`；**非 video 查询永远 dense**；video-video 保留约 10% tile（训练 sparsity 0.9，tile=64 的 `(4,4,4)`）。`to_gate_compress` 基座不存在，loader 零初始化；学生 LoRA 才带训练门。快路径 `fastvideo_kernel.block_sparse_attn_sm100a`（**sm_100 / sm_103**）；否则 Triton block-sparse。**sm_120 不在 CUDA 核资格集。** 有 `vsa_dense_first_n_steps / vsa_dense_layers` 把前几步/层打回 dense。

**步数。** 4 DiT calls。公开材料写下一步才考虑 8-step 质量档。Preview **T2VA only**。

**Cache。** 默认配方不靠 Tea/Mag；速度来自少步 + VSA。

**多卡。** `arch.num_attention_heads % sp_world_size == 0`（`fastvideo/models/dits/minimax_h3.py`）→ 56 头 ⇒ **合法 1/2/4/7/8，非法 3/5/6**。SP 是 Ulysses 同类：序列切 + 注意力内 A2A（`sequence_model_parallel_all_to_all_4D`）。8×B200 可再开 SP8 VAE。

**质量门。** **非无损**；运动/细节/部分音频可低于 Base。`profile=all` **承认 compile 改 FP 序**，report-only。门不是本仓 MAE。

**5090。** 权硬件无关；快核不是 sm_120。32GB 要 offload 或本仓 INT8，数字对不上 B200 47s/14×。可隔离跑 Triton VSA 对照；**不可**把 VSA 学生 LoRA 接 INT8 8-step 金线（换权+换 attn+换步数）。

### 2.2 Sol Super Acceleration（必须与 Sol-Engine 50-step 拆开）

三条 NVIDIA 线：

| 线 | 少步？ | 硬件 | 与本仓 |
| --- | --- | --- | --- |
| Sol-Engine H3 | 否，50-step | 含 5090 4.52×（1045→231s @768p） | Tea+Sol+fusion；**仍非 8-step 金线** |
| **Super Accel** | **是，4+3 且换 LTX** | **2×GB200** | 本节对象 |
| Sol-H3 | 是，默认 FastH3 4-step | 8×B300 | 另一 SKU |

**计算图（仓内 v2，`sol-engine-excerpts/super_acceleration/`）。** 每请求 **2 GPU，零模型并行**：

1. Stage1（GPU A）：FL2VA，**896×512 / 124f / 4 Transformer forwards**，LightX2V `4step_v0.1_544p` LoRA **scale=0.0625**，**cache 关**，DiT/VAE compile，**TAEH3** 出草稿。  
2. Handoff：认证 loopback TCP，BF16 像素 `[1,3,121,384,672]` + FP32 PCM；**禁止中间 MP4**。Stage1 直到 Stage2 ack 张量已在其 CUDA 上才返回。  
3. Stage2（GPU B）：**原版 LTX-2.5 Video VAE encode**（refiner 训在该 latent 上，encoder 不能换 TAE）+ x2 latent upsample → 1344×768；3 updates（sigma `0.909375, 0.725, 0.421875, 0.0`）；layer0 dense，1–47 Sol-Attn **`cute_sm100`**；**TAEHV** 成片。  
4. **扔掉 Stage2 联合去噪音频，mux 回 Stage1 H3 PCM。** 口型合同在草稿轨，不在 refine。

4 GPU 节点 = **两对独立流水**，不是 TP/SP。

**注意力。** Stage2：48 layer；layer0 dense；其余 Sol-Attn τ=1.0/1.25/1.5；cross/audio attn dense。合同：每请求 **3 dense + 141 Sol + 141 实际 `cute_sm100` 核**。Sol 内层 eager，只 compile 48 个 block。

**步数。** 4（H3 草稿）+ 3（LTX），且 **换模型**。

**Cache。** 两阶段都禁用 algorithm cache。少步轨迹上 cache 不是他们的刀。

**质量门。** 页面明确 **非无损**。仓内 `perceptual_quality_gate_run: false`，`speedup_claim: null`。博客 6.85s / 22.2× 的分母是 **已发表 SGLang 152.3s @5s 768p GB200**，不是 50-step 2239s，更不是本仓 66s。

**两阶段延迟分解（正式 job 6304303，20 hot requests，不含 load/compile/warmup）：**

| 边界 | 中位 |
| --- | ---: |
| H3 Stage1 → 最终 Stage2 MP4 | **6.761 s** |
| Stage1 wall | **4.292 s** |
| Stage2 resident service | **2.447 s** |

博客页 6.85s 与正式 6.76s 同量级；**4.292+2.447 已含 handoff 等待**（S1 不提前结束）。二者之和 6.739 与 e2e 6.761 的差是成片/mux 边。  
**为何不可与本仓 8-step 横比：** (1) 分辨率草稿 896×512 vs 金线 864×480，成片 1344×768；(2) 步数 4+3 vs 8，且 Stage2 是另一个 DiT；(3) 解码 TAE 非官方 H3 VAE；(4) 2×GB200 sm100 vs 1×5090 sm120 INT8；(5) 音频来自草稿 PCM。把 66s/6.76≈9.8 写成加速比，是把快档 SKU 焊进金线。

**5090。** Super Accel **不能落**（`cute_sm100` + 双模型常驻）。5090 上能跑的是另一条 50-step Sol-Engine layerwise，231s @768p——比金线 66s **更慢且不同合同**。

### 2.3 VDN-H3

**计算图。** 冻骨干 H3；加线性支路 + 小 LoRA。推理 merge，不改骨干文件。训练：A1 逐层对齐线性支（200）→ A2 端到端（500）→ B QKVO LoRA+线性（2000）→ DMD2 **8-NFE** turbo（250，无 GAN），turbo 初始化自 larryvrh。**故意不做 4-step。**

**注意力核。** `HybridAttention`（`src/models/hybrid_attention.py`）：

- Softmax 窗：按 VAE **5-latent-frame chunk** 对齐（`window.py` `chunk=K`），每帧看完整的前/当前/后 chunk；**中心半径窗会切碎 chunk，他们不用。** text/audio 作 global，双向 dense。  
- 边界锚：`anchor_frames` in {none, columns, rows, both}；both 时线性支 `skip_ends`，两支划分精确、不双计。锚约 +3.57% 密度。  
- Video Delta 线性：`BidirectionalLinearBranch`，帧内对 `(I+A)^{-1}` 联合求解（Cholesky），非 SANA-WM 加性近似；双向 scan + 文本半写入。  
- 融合：softmax 门 + 线性 RMSNorm 门 + 独立 `to_out_linear`，**只加在 video 行**。  
- 推理核：窗走 FlexAttention FA4 或 **decomposed**；B200 配方显式 `softmax_backend: decomposed`。

**步数。** 50-NFE「近无损档」+ **8-NFE 快档**。`num_inference_steps` 计 sigma 格点，9 点 = 8 次模型评估。

**Cache。** 文档有 kernel+caching 字样；公开推理主收益是少步 + hybrid + Ulysses，不是 Tea 族。**无 KV cache、无近似通信。**

**多卡。** 标准 Ulysses：残差按 packed 行切；attn 内 all-to-all 变头切（`ulysses.py` 模块文档）。B200 再开 **5+3 分支并行**：`parallel.softmax_ranks: 5` → 5 卡跑窗 softmax、3 卡跑线性支，QKV 只算一次，不均匀 A2A 把 processed QKV 送到 softmax 组、raw QKV 送到线性组（`_branch_parallel_attention_forward`，`UlyssesRuntime.dispatch_fields_to_branches_overlapped`）。相对均匀 8 路再降 ~13% NFE（yaml 注释：B200 5+3 = 1.425 s/NFE vs 6+2 = 1.629）。H200 twin 是 **6+2**（算力平衡不同）。`install_ulysses` 要求 `heads % world_size == 0`。

**质量门。** 口头 near-lossless vs **dense MiniMax H3 BF16**；计时 **排除 VAE / load / warmup / mux**。8-NFE 是明确折中。不是本仓 PPM MAE。

**5090。** 单卡 FP8 transformer ~45GB、工作集 ~65GB → **32GB 不能常驻**。Flex/FA4 sm_120 需另测。多卡 5090 若每卡复制 FP8 同样不够；只能切序列 + 本仓已有的 INT8，那已不是他们的 11.23s（8×B200、仅 denoise、14.4s 768p 8-NFE）故事。

可抄思想：**停在 8-NFE**；window+anchor 当研究 attn（独立后端、过 MAE 才谈）。不可抄：线性支替换 cuDNN 默认；把 11.23s 当无损金线。

### 2.4 Fal H3 Max（闭源；机制推断 + 证据链）

**能核实（官方页 / API / 博客，2026-09 公开材料）：**

- fal **后训练**开源 H3，RL + 人评 Elo / Design Arena；与「四年自研扩散引擎」共设计。  
- 训练与服务在 **GB200 NVL72**。  
- 5s 768p 墙钟 **&lt;3s**，`timings.inference ≈ 2.5s`；相对官方 MiniMax **托管**约 **35× 吞吐**。  
- 产品面：480p/768p、5–15s、T2V/I2V/Ref2V，原生同步音。  
- **无权重、无 Git、无步数、无核、无并行拓扑、无 MAE。**

**推断（必须打标，不是事实）：** 要在 GB200 上把 5s 768p inference 打到 ~2.5s，信息论上几乎必然是 **极少 NFE（约 4 或更少）+ 常驻 + 融合 + 可能稀疏/量化 attn + 可能多卡**。fal 原文反对「只减步/降精度就报倍数」，但 **没有**证明 2.5s 路径与开源 50-step 或本仓 8-step INT8 同分布。Design Arena「50× speed」是对 **托管 H3** 的偏好-速度前沿。

**质量门（可抄的唯一样条）：** 「优化只有在质量评测站住才留下」——对齐本仓失败即关。秒数不可复现。

**5090：** 不能本地落地。禁止把 2.5s / 35× 写入 66s 叙事。

### 2.5 RunningHub Lightning（机制总览；拓扑细节见 §1）

**计算图。** SGLang `multimodal_gen` serve：TE → packed H3 DiT → 官方 VAE → mux。BF16 权 + 可选 LoRA merge。`--performance-mode speed`。

**注意力。** 默认 `sage_attn`（SageAttention2.2）；可回退 `fa`。不是 Sol，不是 VSA。

**步数。** 宣传 4（RH 自训权未公开）；公开复现 larryvrh 4 或 LightX2V ref2v **8**。高动态建议 8。

**Cache。** Cache-DiT 1.3 默认开（`SGLANG_CACHE_DIT_ENABLED=true`），RH 称 quality-validated；判据见 §1.4。

**多卡。** **TP2+Ulysses4**，专为 PCIe 无 NVLink。详见 §1。

**质量门。** 不宣称无损；无 MAE 表；强制音视频双流。

**5090。** 拓扑最近；显存与权精度差一代故事。抄切分，不抄 12.2×，不抄 4-step 默认。

---

## 3. 对本仓 2× / 4×5090 保效果路线（通信 / 切分 / 不改采样）

原则与浅层 M1 一致，这里落到「做哪条通信、不做哪条」。

### 3.1 合同

- 配方锁死：INT8 8-step，`sdpa=cudnn`，cfg=1，864×480 有声，真 `--seconds`。  
- KPI 另册（例如 `out/verify_5090_sp2/`），**不改 66s 行**。  
- 门：相对单卡 5090 cudnn cafe 同 seed 的 PPM MAE ∩ 口型/有声。  
- `num_gpus` ∈ {2,4,8}（整除 56 头与 packed 对齐 64）。

### 3.2 切分顺序（抄 RH 的 PCIe 教训，不抄 RH 配方）

| 优先级 | 拓扑 | 切什么 | 通信 | 为何先做 |
| --- | --- | --- | --- | --- |
| **P0** | 2× **Ulysses2 / TP1** | packed 序列行；attn 内头维 A2A | 每层 packed QKV A2A + 输出 A2A；2 卡优先 IPC | 无 all-reduce；INT8 权复制仍常驻 |
| **P1** | 4× **Ulysses4 / TP1** | 同上，U=4 | NCCL all-to-all（无 4 卡 IPC 快路径） | 仍避免 TP |
| **P2** | 4× TP2+Ulysses2 | 权半切 + U=2 | per-block AR + A2A | 仅当 P1 显存/头约束爆了（864×480 INT8 预期用不上） |
| 不做（金线） | 任何 4-step / Sage / Cache / compile / TAE | 改轨迹或改核 | — | 快档 SKU 另 CLI |

CFG 并行：金线 cfg=1，**无分支可切**。PP：单请求 8-step 50 层，bubble 会吃掉收益，RH 也没用。

### 3.3 通信是否划算（机制，非本仓实测）

Ulysses A2A 是 **置换**，体积 ~ packed QKV（BF16/FP16 激活，不是 INT8 权）。PCIe Gen4/5 x16 在 2 卡 P2P 上，对 ~16k token、H=56、D=128、50 层×8 步的量级，通信税会可见但通常小于「把 TP all-reduce 打满」。RH 8 卡 12% 的拓扑差已经证明：**在无 NVLink 上，A2A 形状优于 AR 形状。** 2×5090 应先验证 Ulysses2，不要从 TP2 起步（cookbook 的 5090 TP2 是 **BF16+offload 容量解**，本仓 INT8 不需要那条）。

权 INT8 已常驻：多卡 **不会**再通过「避免 layerwise」拿到 6000D 那种「从不能跑到能跑」的收益。唯一杠杆是 DiT 墙钟。TE 12–14s + VAE ~10.5s 是 e2e 地板，除非另开 VAE chunk 并行（FastVideo `minimax_h3_parallel.py` 的 bit-exact 合同可当设计参考，仍须 MAE 门）。

### 3.4 明确不把对方秒数写进金线

| 对方数字 | 合法读法 |
| --- | --- |
| RH 12.2× / 28.7s | 8×6000D、768p、4-step+Sage+Cache+compile vs 同机 50-step |
| RH 48.2 vs 54.0s | **同 4-step** 的拓扑差 12% |
| FastH3 14× / 47s | 1×B200、4-call+VSA vs ~49-step BF16 |
| Super Accel 6.76s / 22.2× | 2×GB200 draft+LTX；正式仓不声称倍数 |
| VDN 11.23s | 8×B200、仅 denoise、8-NFE 768p 14.4s |
| Fal ~2.5s / ~35× | 托管 GB200；不可复盘 |
| Sol-Engine 4.52× @5090 | **50-step BF16 768p** 231s，不是 8-step 66s |

---

## 4. 给执行者的硬停止句（与浅层一致，加深一条）

1. 不要改 generate 默认 backend / steps / mag / sage / sol。  
2. 不要把 4-step LoRA merge 进 INT8 金线。  
3. 不要把 12.2× 解释成「多卡效率」或「相对 66s」。多卡效率只引用 **同配方拓扑表**（54.0→48.2）。  
4. 不要假设卡是 Ada 6000 或带 NVLink 的 96GB PRO 6000；RH 机器是 **RTX 6000D sm_120 PCIe**。  
5. 2×/4×5090 第一刀是 **Ulysses、TP=1、cudnn、8-step**；失败即关，数字另册。

---

## 5. 公开材料不足 / 推断清单

| 项 | 状态 |
| --- | --- |
| RH 单卡 6000D 同 5s 768p 秒数 | **无**；8 卡相对 1 卡加速比未知 |
| Sage2、Cache-DiT、compile 各自 ablation | **无**；只有「蒸馏」vs「蒸馏+三件套」两行 |
| RH 权 vs larryvrh 公开 LoRA 的质量差 | **未公开** |
| 本仓 Ulysses2 在 2×5090 上的 DiT/e2e | **未测**（现网若仍单卡 5090 则等第二台） |
| Super Accel 感知质量 vs 金线 MAE | 仓内门未跑 |
| Fal 步数/核/拓扑 | 闭源 |
| 4.17+2.68 作为 Super Accel 分解 | **本仓库内摘录是 4.292+2.447（job 6304303）**；若他处博客有 4.17+2.68，以正式 JSON 为准，不把两套数加总 |

未 git commit。密码未写入。默认路径未改。
