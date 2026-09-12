# MiniMax-H3-MultiGPU-Lightning 源码调研：对本仓 8×5090 Ulysses（C1–C4）的对比与行动项

> **日期：** 2026-09-13。只读调研产物，未改任何代码，未 commit。
> **前置文档：** [2026-09-11-h3c-pcie-multigpu-lessons.md](./2026-09-11-h3c-pcie-multigpu-lessons.md)（PCIe 教训短补）· [2026-09-11-h3c-5090-turbo-implementation.md](./2026-09-11-h3c-5090-turbo-implementation.md)（C1–C4 计划）
> **源码树（gitignore）：** `third_party/accel_survey/MiniMax-H3-MultiGPU-Lightning/`（SGLang pin `f8cbf000f4a5`）
> **For agentic workers：** 本文是调研 + 行动项，不是实现计划。禁止改 generate 默认 / 单卡金线 KPI / 默认 `sdpa=cudnn`。C3/C4 执行细则以 turbo-implementation 计划为准，本文 §5 行动项是其前置补丁。

**一句话：** RH Lightning 在同样「PCIe 无 NVLink + Blackwell 8 卡」环境下验证了本仓已选路线（少 TP、Ulysses 切序列、NCCL P2P 不关、只切 DiT、蒸馏与并行正交）；对我们的实质增量是 **一个 correctness 教训（pad 行污染 softmax）**、**一个进度事实（C2 的 DiT 插点未接上）**、和三个过门后的性能迭代项（U=2 peer 直写、去 host sync、BF16 传输）。

---

## 1. 该仓库方案概述

**本质：** RunningHub（RH）没有自研运行时，而是 pin 了一份 SGLang `multimodal_gen` 快照（`f8cbf000f4a5`），核心价值在配方选择与实测数据。8× RTX 6000D（Blackwell sm_120，PCIe Gen5 x16，**无 NVLink**，8 卡跨两个 NUMA 域）上 5s 视频 348.8s → 28.7s（12.2×）。

### 1.1 并行组合：TP2 + Ulysses4

`scripts/start.sh` 钉死 `--num-gpus 8 --tp-size 2 --ulysses-degree 4`；Ring=1、CFG 并行=关、PP=关。

- 组划分 `num_gpus = cfg × tp × ulysses × ring`，rank 顺序 `tp-sp-pp-cfg-dp`（`runtime/distributed/parallel_state.py`）；SP 子组由 `set_seq_parallel_pg_by_sp_groups`（`runtime/distributed/parallel_groups.py`，`use_ulysses_low=True`）拆成 **Ulysses 连续 rank** + Ring 跨步 rank——同 NUMA 优先。
- CFG 轴闲置原因和本仓相同：蒸馏后 cfg=1，denoise 单分支（`denoise_loop.py`）。

### 1.2 通信原语与拓扑假设

**不假设 NVLink。** 主路径是 **NCCL all-to-all 走 PCIe P2P**（launcher 未设 `NCCL_P2P_DISABLE` / `NCCL_SHM_DISABLE`）。

| 路径 | 条件 | 实现 |
| --- | --- | --- |
| NCCL A2A | Ulysses 组（RH 默认 U=4） | `_usp_all_to_all_single` → `torch.distributed.all_to_all_single`（`runtime/layers/usp.py`），注释明确「立即 wait，无 overlap」 |
| QKV 单集合 | 输入侧 | `pack_qkv_destination_major` 把 Q/K/V 打成一次 destination-major 集体（`_usp_input_all_to_all_packed_qkv`），grow-only staging buffer（`_a2a_staging_buffer`）消除每层分配 |
| CUDA-IPC 直写 | **仅** U=2 且 TP=1 且同主机且 `can_device_access_peer` | `device_communicators/ipc_a2a.py`：staging buffer 经 IPC handle 映射到对端，peer 直写 + GPU 侧 spin/signal 序号同步，CUDA-graph 可捕获。**RH 生产 TP2+U4 走不到这条路**（`ipc_a2a_ready` 在 TP>1 时直接 False） |
| Ring 变体 | 默认关 | `_ring_attention_varlen`：KV 按 P2P 旋转、hop 与 compute overlap，但只支持 FA backend |

有数据支撑的拓扑结论：同 8 卡同任务（15s 768p t2va 4-step），**TP4+U2 54.0s → TP2+U4 48.2s（约 12%，省 ~14 GiB/卡）**——PCIe 上「宁可 A2A 也少 TP all-reduce」。

### 1.3 pad 处理（RH 侧）

`runtime/distributed/sp_shard_utils.py`：ceil-div 均分，**pad 全部落在最后一个 rank 的本地尾部** → Ulysses gather 后 pad 是一个全局尾块；`tail_attn_meta` 把 pad 做成独立 varlen 段，注意力**零成本跳过**（无 repack、无 mask 计算）；joint attention 用 `join_seqs` 把文本 pad 搬到全局尾。文件头写明设计动机：「padding always sits at the end of the LAST rank's local chunk」。

### 1.4 少步蒸馏与多卡：完全正交

turbo LoRA 用 `--lora-merge-mode auto` 在**加载时 merge 进 BF16 权重**，之后 TP/Ulysses 对切分后权重无感知。12.2× 拆账（同 8 卡，卡数是常量）：4-step 蒸馏占节省的 95.5%（348.8→43.0s），Sage2+Cache-DiT+compile 再 ×1.5（43.0→28.7s）。**禁止把 12.2× 乘进「多卡效率」叙事**（教训文档 §5 已锁）。

---

## 2. 与本仓 h3.c Ulysses（C1/C2）逐点对比

| 维度 | RH (SGLang) | 本仓（`h3_ulysses.c` / `h3_ulysses_comm.cu`） | 评价 |
| --- | --- | --- | --- |
| 切分维度 | packed 序列行切 U 份，注意力前 A2A 换头维（heads→H/U、seq→全长），输出反向 A2A | 相同（`h3_ulysses_a2a_qkv`: `[S_local, 3HD]` → `[S_pad, 3(H/U)D]`；`h3_ulysses_a2a_attn_out` 反向） | 一致 |
| head 整除约束 | `(H/T) % U == 0`、`64 % (U×R) == 0`（`minimax_h3.py::_validate_sequence_parallel_config`，H=56、对齐 64） | `h3_ulysses_validate` 三条规则逐字对应；U=7/U=3/TP4+U4 均拒（`tests/test_h3_ulysses.c`） | 一致（C1 本就照它建） |
| pad 处理 | pad 落最后 rank 尾部→全局尾块；`tail_attn_meta` varlen 段跳过 pad | pad 到 64 对齐（cafe 15973→16000，27 行），`ulysses_unpack_qkv_f32` 后 pad 同在全局尾。**但注意力对 padded 全序列直接算，无跳过机制** | ⚠️ 实质差距，见 §5 行动项 2 |
| A2A 实现 | 每 GPU 一进程 + torch.distributed；QKV 单集合；staging 复用；U=2 IPC 直写；stream-ordered **无 host sync** | 单进程 `ncclCommInitAll` 驱动全部卡；自写 destination-major pack/unpack 核；grow-only `ensure_buf`（parity）；QKV 单集合（parity）；**每次 A2A 每 rank `cudaStreamSynchronize`**；payload 硬编码 **F32**（`dtype_bytes != 4` 直接拒） | 结构相似，两处性能差距（§4） |
| VAE / TE | 都**不并行**（`encoder_parallel`/`decoder_parallel` 关；TP×U 只砍 DiT） | 计划相同（只切 DiT） | 一致，互相印证 |
| LoRA × 多卡 | merge 进权重，对并行透明 | 合同禁止 merge 进 INT8（`h3_lora.h` 头注释），runtime residual；A2A 插点在 LoRA residual 之后（turbo 计划 C2 已写明） | 无冲突，同样正交 |

### 2.1 关键事实：本仓 C2 的 DiT 插点未接上

`h3_ulysses_a2a_qkv` / `h3_ulysses_a2a_attn_out` 在本仓**只被合成测试 `tests/test_h3_ulysses_a2a.cu` 调用**；`h3_dit_w4a8.c` 里没有任何 ulysses 引用；`tools/h3_cuda_generate.c` 在 `--ulysses N>1` 时只做 `h3_ulysses_comm_init` 并打印：

```1792:1794:tools/h3_cuda_generate.c
        printf("ulysses: N=%d NCCL inited (experimental; default still U=1; "
               "DiT SP in block A2A not driving this denoise yet)\n",
               ulysses);
```

即 turbo 计划 Task C2 的「Modify: `h3_dit_w4a8.c` `block_device_compute`」尚未落地。**C3 的真正前置工作是补插点，不是直接跑短测。** 详见 §5 行动项 1。

---

## 3. 值得借鉴 / 不适用清单

### 3.1 值得借鉴（按优先级）

1. **pad 行必须从注意力 K/V 中剔除（C3 correctness 头号风险）。** zero-pad 的 K 行 logit=0 → softmax 权重 \(e^0/Z\) **非零**，15973 个真实 query 全被 27 行 pad 污染。RH 有两套机制（`tail_attn_meta` varlen、ring 的 `remote_used` 截断）；本仓一个都没有。最简修复见 §5 行动项 2。**不修则 C3 MAE 门必挂，且会被误判为通信 bug。**
2. **U=2 走直写 P2P，跳过 NCCL。** RH 的 IPC A2A 证明 2 卡置换用 peer 直写 + GPU 计数器比 NCCL rendezvous 省。本仓单进程，比 RH 更简单：不需要 IPC handle，直接 `cudaMemcpyPeerAsync` / peer 指针写（`h3_ulysses_comm_init` 已 `cudaDeviceEnablePeerAccess`）。C3 过门后的性能迭代项。
3. **去掉每次 A2A 的 host sync。** 本仓每 block 两次 A2A、每次对每 rank `cudaStreamSynchronize`（`h3_ulysses_comm.cu` a2a 尾部），8-step×50-block×2 = 800 次 host 往返 + 流水线断裂。RH 全程 stream-ordered。单进程下用 event 串 NCCL 流与 compute 流，只在 step 边界 sync。
4. **BF16 传输 payload。** 本仓 `dtype_bytes==4`（F32）硬编码，PCIe 字节数是 RH（BF16）的两倍。置换本身无损，但 F32→BF16→F32 downcast 非位级等价，需过质量门后作可选项（预计通信墙钟减半）。
5. **拓扑结论直接采信：** 少 TP 多 Ulysses（TP4→TP2 的 12% 实验）、Ulysses 组连续 rank/同 NUMA、不关 P2P。与教训文档一致，本次源码调研确认实现层面无隐藏私货。

### 3.2 不适用

| 禁 / 不抄 | 为什么 |
| --- | --- |
| TP2+U4 默认配方 | 那是 6000D 85GB 跑 BF16 768p 的容量折中（峰值 60–66 GiB/卡）；本仓 INT8 常驻单卡 32GB，TP 的每 block all-reduce 在 PCIe 上纯亏。仅 U=8 爆了才 fallback（C4 已规定） |
| Ring attention | 只配 FA backend，与金线 `sdpa=cudnn` 冲突 |
| IPC handle + spin-kernel 跨进程机制 | 本仓单进程用不着，直接 peer copy 更简单 |
| 4-step LoRA merge / Sage2 / Cache-DiT / torch.compile 进金线 | 换轨迹或改数值，与多卡正交；教训文档 §7 已禁 |
| `tail_attn_meta` 的 varlen 零拷贝框架 | 他们 pad 可能上千行才值得；本仓 27 行 pad 用「slice 到真实长度」即可 |

---

## 4. 对 C3/C4 短测的具体建议

### 4.1 C3（U=2）步骤

1. **先跑 C2 欠的 GPU 短测**：远端 `CUDA_VISIBLE_DEVICES=0,1`（同 NUMA 0）跑 `h3_ulysses_test_a2a`，确认日志 `peer_access ok=2 fail=0`、往返 maxabs==0。这一步从未在 GPU 上验证过。
2. **补 DiT 插点**（§5 行动项 1）+ **pad 截断**（§5 行动项 2）。
3. **correctness 递进**：先 `--seconds 1 --steps 2` 冒烟比对单卡同 seed（Ulysses 是纯置换 + FP 归约序差异，MAE 应接近位级；落到 sage3~27 量级即实现 bug，不许改阈值）；再 cafe 金线 8-step warm×3 过 MAE∩有声门（对照 `out/verify_5090x8_gold/`）。
4. **首轮不做性能项**：F32 + NCCL + 现有 sync 先过门；BF16 传输 / peer 直写 / 去 sync 作为过门后独立迭代，每项单独量化。

### 4.2 预期收益上限（估算，未测不算数）

cafe U=2、F32、8-step gold：S=15973→16000，S_local=8000，H=56，D=128。

- QKV A2A：每 rank 发 8000×3×7168×4B×(1/2) ≈ **344 MB**；attn-out A2A：8000×28×128×4B ≈ **115 MB**。每 block ≈ 459 MB egress/rank，×50 block ×8 step ≈ **184 GB**。
- PCIe Gen5 x16 有效 ~40–50 GB/s（双向同时）→ 通信墙钟约 **4–5 s**。
- DiT ≈ 31.67/2 + 4.5 ≈ **20–21 s**（1.5–1.6×）；gold e2e 约 78.7 → **~67 s**。BF16 传输后通信减半，DiT 可到 ~18 s。turbo4（4-step）通信同比减半。

### 4.3 C4（U=4 / U=8）

- U=4 同 NUMA：每 rank ~344 MB/block（略少于 U=2 但 rendezvous 变 4 方）。
- U=8：理想 DiT compute ~4 s + 通信 3–5 s（跨 NUMA 有效带宽可能掉到 15–25 GB/s）→ DiT 约 **8–10 s**、e2e ~56 s；但 TE+VAE 地板 ~47 s，Amdahl 收益递减明显，符合「480p 甜点可能是 U=4，U=8 留 768p」预判。
- U=8 爆了才试 TP2+U4，且需 C3+U4 先过门（turbo 计划 C4 原文）。

---

## 5. 行动项（执行者必读，C3 前置）

### 行动项 1：补 C2 的 DiT 插点——**当前 `--ulysses N>1` 不驱动 denoise**

**现状证据：**

- `rg -ln "h3_ulysses_a2a_qkv|h3_ulysses_a2a_attn_out"` 只命中 `h3_ulysses_comm.{h,cu}`、stub、`tests/test_h3_ulysses_a2a.cu` 和计划文档；`h3_dit_w4a8.c` **零命中**。
- `tools/h3_cuda_generate.c` L1782–1795：`--ulysses N>1` 只调 `h3_ulysses_comm_init`（devices `0..N-1`），随后打印 `DiT SP in block A2A not driving this denoise yet`；L4208 只有 destroy。

**要做：** 按 turbo 计划 Task C2 的插点图，修改 `h3_dit_w4a8.c` `block_device_compute`：

```
QKV GEMM on local rows [S_local, 3*H*D]
→ (+ LoRA residual if turbo)                    /* 插点必须在 residual 之后 */
→ (U>1) h3_ulysses_a2a_qkv → [S_pad, 3*(H/U)*D]
→ rope+sdpa heads=H/U（见行动项 2 的截断）        /* 金线保持 cudnn */
→ (U>1) h3_ulysses_a2a_attn_out → [S_local, H*D]
→ out-proj + MLP on local rows
```

结构性难点（与 RH 每卡一进程的最大差异）：本仓单进程 `ncclCommInitAll`，`h3_ulysses_a2a_qkv` 要求**所有 rank 都已 `h3_ulysses_comm_bind`**（`h3_ulysses_comm.cu` 里逐 rank 校验 `bound_in/bound_out`）才发集合——即两卡各持一份完整 INT8 权重、同一 denoise 循环逐 rank 驱动。U=1 必须走现路径、零 extra A2A（C2 验收原文）。

**验证：** ① `--ulysses 1` 短冒烟与改前一致（`sdpa=cudnn`、无 NCCL init）；② `--ulysses 2 --seconds 1 --steps 2` 出片且与单卡同 seed MAE 接近位级；③ `--ulysses 7` 非零退出、stderr 含 `packed align`。

### 行动项 2：pad 行截断到真实长度——**否则污染 softmax，C3 必挂**

**问题机理：** `h3_ulysses_pad_seq(15973, 64) = 16000`，27 行 zero-pad 经 `ulysses_unpack_qkv_f32` 落在 gathered 序列全局尾部。若 SDPA 对 S_pad=16000 全长直接算：pad 的 K 向量为零 → 每个真实 query 对它 logit=0 → softmax 权重 \(e^0/Z\) **非零** → 全部 15973 个真实 token 的输出被污染。这不是「pad 输出丢弃就行」能解决的——污染发生在**真实行**上。

**RH 的对策（参考，不抄框架）：** `sp_shard_utils.py::tail_attn_meta` 把 `[valid | pad]` 做成 varlen 两段，注意力天然不跨段；`usp.py::_ring_attention_varlen` 用 `remote_used` 截断每 hop 的真实前缀，注释明言「Padding rows never contribute to any KV chunk」。

**本仓最简修复（利用 pad 在全局尾这一不变量）：** 插点内 SDPA 只对前 `S_real=15973` 行做 q/k/v（单卡路径本就跑 15973，cudnn 不要求 64 对齐——64 对齐只是 A2A 的通信粒度）；pad 行输出置零后再进 `h3_ulysses_a2a_attn_out`，保证反向 A2A 形状仍按 S_pad 对齐。

**验证：** U=2 短冒烟对比单卡同 seed 的 MAE 必须接近位级（置换 + FP 归约序差异量级）。若 MAE 落到 sage3~27 / sol~35 量级，先查此项，**不许改阈值**。可加一个廉价单测：构造 S_real 非 64 倍数的小张量，断言真实行输出与不 pad 的参考实现逐位一致。

### 行动项 3（过门后，性能迭代，每项单独量化）

| 项 | 位置 | 预期 |
| --- | --- | --- |
| 去 per-call host sync | `h3_ulysses_comm.cu` `h3_ulysses_a2a_qkv` / `a2a_attn_out` 尾部的 `cudaStreamSynchronize` → event 串流，step 边界才 sync | 消 800 次 host 往返/run |
| U=2 peer 直写 | 单进程直接 `cudaMemcpyPeerAsync`（peer access 已开），跳过 NCCL rendezvous | 每 block 少一次集合延迟 |
| BF16 传输 payload | 放开 `dtype_bytes==4` 硬编码，A2A 前后转换 | 通信墙钟约减半；**非位级等价，需单独过质量门** |

---

## 6. 结论

对本仓有实质帮助，但不在「抄架构」而在三点：① **方向验证**——同环境实测证明纯 Ulysses 优先、TP 尽量少、NCCL P2P 不关、只切 DiT、蒸馏正交的路线是对的（含 TP4 vs TP2 的 12% 反例数据）；② **一个能直接避免 C3 翻车的 correctness 教训**——pad 行 softmax 污染必须处理（§5 行动项 2），现有计划文档未覆盖；③ **三个低风险性能迭代**（§5 行动项 3）。另外本次调研确认 C2 的 DiT 插点实际未落地（§5 行动项 1），C3 的真正前置是补插点而非直接短测。

未改代码，未 commit。
