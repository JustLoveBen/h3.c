# RH Lightning：无 NVLink 多卡推理怎么走、对本仓 5090 抄什么

> **后续优化方案见** [2026-09-11-h3c-next-opt-plan.md](./2026-09-11-h3c-next-opt-plan.md)  
> **极速 SKU 实现：** [2026-09-11-h3c-5090-turbo-implementation.md](./2026-09-11-h3c-5090-turbo-implementation.md)  
> **For agentic workers:** 短补，不是实现计划。禁止改 `h3_cuda_generate` 默认 / 单卡金线 KPI / 默认 `sdpa=cudnn`。不 commit。  
> **RH 专章（数字拆账）：** [2026-09-11-h3c-accel-survey-tech-deepdive.md](./2026-09-11-h3c-accel-survey-tech-deepdive.md) §1  
> **源码树（gitignore）：** `third_party/accel_survey/MiniMax-H3-MultiGPU-Lightning/`（SGLang pin `f8cbf000f4a5`）

**一句话：** RH 没有发明一条「绕过 PCIe」的私有总线。8× RTX 6000D 上就是 **NCCL 走 PCIe Gen5 P2P**；他们赢在 **少 TP、多 Ulysses**，以及把 12.2× 从「多卡」里拆出去。

路径均相对 `third_party/accel_survey/MiniMax-H3-MultiGPU-Lightning/`。

---

## 1. 没有 NVLink 时实际走什么链路

**物理：** README / README_CN 实测钉死 PCIe Gen5 x16、无 NVLink；8 卡跨两个 NUMA（同域 NODE，跨域 SYS）。`scripts/start.sh` **没有**设 `NCCL_P2P_DISABLE` / `NCCL_SHM_DISABLE` / `NCCL_IB_*`。

**软件 backend：** CUDA 上 `current_platform.get_torch_distributed_backend_str()` → `"nccl"`（`sglang/python/sglang/multimodal_gen/runtime/platforms/interface.py`）。`init_distributed_environment` / `initialize_model_parallel` 把这个 backend 传给 `torch.distributed.init_process_group` 和各 `GroupCoordinator`。

**P2P 没关，反而会开：**

| 路径 | 何时 | 实现 |
| --- | --- | --- |
| NCCL all-to-all | Ulysses 组（RH 默认 U=4） | `_usp_all_to_all_single` → `torch.distributed.all_to_all_single(..., group=ulysses_pg)`（`runtime/layers/usp.py`） |
| NCCL / pynccl all-reduce | TP 组（RH 默认 T=2） | `RowParallelLinear.forward` → `tensor_model_parallel_all_reduce` → `GroupCoordinator.all_reduce`（可能走 SRT custom AR，否则 `torch.distributed.all_reduce`） |
| CUDA-IPC A2A | **仅** `SGLANG_DIFFUSION_IPC_A2A=1`（默认 true）且 **TP=1** 且 Ulysses `world_size==2` 且 `can_device_access_peer` | `ipc_a2a_ready`；`cudaDeviceEnablePeerAccess`；peer staging 直写（`runtime/distributed/device_communicators/ipc_a2a.py`） |

RH 生产配方是 **`--tp-size 2 --ulysses-degree 4`**。`ipc_a2a_ready` 在 `get_tp_world_size() > 1` 时直接 `return False`。因此 **8 卡 Lightning 主路径没有走 CUDA-IPC A2A**，Ulysses4 是 NCCL all-to-all，TP2 是 NCCL/custom all-reduce。IPC 是 2 卡、TP=1 的快路径，不是他们 8 卡的主叙事。

`ipc_a2a.py` 文件头写「over NVLink」是 SGLang 通用注释；在这台 6000D 上 peer mapping 的物理载体仍是 **PCIe**。置换不是归约，IPC 与 NCCL 位级相同（`parallelism.mdx` The Ulysses transport）。

**Device mesh：** H3 serve 路径 **不用** `torch.distributed.DeviceMesh`。并行组是 `RankGenerator(..., order="tp-sp-pp-cfg-dp")`（`runtime/distributed/parallel_state.py` `initialize_model_parallel`），SP 子组再由 `set_seq_parallel_pg_by_sp_groups`（`parallel_groups.py`）拆成 **Ulysses 连续 rank** 与 **Ring 跨步 rank**。DeviceMesh 只出现在 FSDP load / Hunyuan / ComfyUI 旁路，不是 Lightning H3。

---

## 2. 为何少 TP、多 Ulysses（PCIe 上 all-reduce vs all-to-all）

SGLang `sglang/docs/docs/sglang-diffusion/parallelism.mdx` 把轴按跨互连友好度排序：**TP all-reduce 最差**（每 transformer 块一次 hidden-size 归约，频率最高、无 overlap，NVLink 上已约占 sharded kernel 时间 ~70%）；Ulysses all-to-all 次差（每层注意力两次、需要 full-bisection，文档建议留 NVLink）。RH **没有 NVLink**，但仍选 U=4、T=2，而不是把 8 卡全给 TP。

机制差异：

- **TP：** 切权。`ColumnParallelLinear` / `MergedColumnParallelLinear` 切输出；`RowParallelLinear.forward` 在 `reduce_results and tp_size > 1` 时 **阻塞 all-reduce 还原 hidden**（`runtime/layers/linear.py`）。H3 50 block ×（attn `out_proj` + MLP `fc2`）每步多次 AR。PCIe 延迟对「小而密」的归约极不友好。
- **Ulysses：** 切 packed **序列行**；注意力前把头维 A2A 成「全序列、少头」。体积是 **激活置换**（QKV），不是权归约。H3 packed 路径把三次输入打成一次 destination-major 集体：`_minimax_h3_attention_core_impl` → `_usp_input_all_to_all_packed_qkv`（`pack_qkv_destination_major`）+ `_usp_output_all_to_all`。`_usp_all_to_all_single` 注释写明 USP **立刻 wait**，不能像 Ring hop 那样藏进计算。

公开证据（同 8 卡、同 4-step、15s 768p 竖屏 t2va，**与 12.2× 表独立**）：TP4+Ulysses2 **54.0 s** → TP2+Ulysses4 **48.2 s**（约 12%，少 ~14 GiB）。这是「无 NVLink 时宁可 A2A 吃 PCIe、也不要把 rank 堆到 TP AR」的实验，不是口味。

---

## 3. 切分公式与 H3 头数约束

```text
num_gpus = cfg_parallel_degree × tp_size × sp_degree
sp_degree = ulysses_degree × ring_degree
```

RH `scripts/start.sh`：**cfg=1、tp=2、ulysses=4、ring=1、pp=1、dp=1 → 8 GPU。** 未传 `--enable-cfg-parallel` / `--ring-degree` / `--pp`。H3 金线是 cfg 蒸馏（`denoise_loop.py`：「positive presentation is forwarded exactly once」），**CFG 轴闲置**。

硬约束在 `MiniMaxH3DiTModel._validate_tp_config` / `_validate_sequence_parallel_config`（`runtime/models/dits/minimax_h3.py`），对齐常数 `MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT = 64`、`num_attention_heads = 56`（`configs/models/dits/minimax_h3.py`）：

```
H % T == 0
(H / T) % U == 0          # 切的是 TP-local 头，不是全局 H % U
64 % (U × R) == 0
hidden/ffn/... % T == 0   # TP 还切 GEMM 宽
```

因此 **TP2+Ulysses4 合法**（28 % 4 = 0），**TP4+Ulysses4 非法**（14 % 4 ≠ 0）。行切：Ring 外层连续块、Ulysses 内层（`build_rope_cache` / `forward` 的 `row_start`）。RH 的 R=1 → 每卡 residual `S/4` 行、`H/2=28` 个 TP-local 头；进注意力 A2A 后全序列、`28/4=7` 个头。

本仓 TP=1 时合法 Ulysses：`{1,2,4,8}`。**U=7 因 64%7≠0 非法**（不要抄 FastVideo 只检查 `heads % sp` 的 7）。

---

## 4. 哪些阶段根本没切（TE / VAE）

一次 t2va：`TE (Qwen3-VL) → pack [text|audio|video] → DiT denoise loop → VAE decode → mux`。Lightning 的 TP×Ulysses **只砍 DiT**。

| 阶段 | RH 8 卡主配方 | 代码 |
| --- | --- | --- |
| **TE** | 不在 TP2+U4 主切分里。`--encoder-parallel` 默认 `auto`；单请求 batch=1 时 encoder DP 无收益。fold 是另一轴（encoder GEMM + 每层 AR），`start.sh` 未钉 fold。文档：encode 时整个 DiT replica 闲、denoise 时 encoder 闲（`encoder_parallel.mdx`） | `MiniMaxH3Qwen3VLEncoder`；`parallel_state._ENCODER_DP` |
| **VAE** | `encoder_parallel=False`、`decoder_parallel=False`；tile 可选。`start.sh` 无 `--vae-config` / parallel-decode | `MiniMaxH3VideoVAE.__init__` |
| CFG / PP / DP / Ring | 关 | denoise 单分支；launcher 未开 |

6000D 85GB 上 BF16 常驻；8 卡动机是 **768p 序列墙钟**，不是「不切就得 offload」。`start.sh` 无 `--layerwise-offload-components`。

---

## 5. 与「快」解耦：4-step 不是无 NVLink 的功劳

同一 8×6000D、5s 1344×768 t2va（README 原文含步数变化）：

| 配置 | 步数 | 延迟 |
| --- | ---: | ---: |
| BF16 baseline | 50 | 348.8 s |
| RH 蒸馏权 | 4 | 43.0 s（**8.1×，占节省 95.5%**） |
| + Sage2 + Cache-DiT + compile | 4 | 28.7 s（再 ×1.50，**4.5%**） |

**卡数在此表是常量。** 拓扑 12% 是另一张表。禁止把 12.2× 乘进「无 NVLink 多卡效率」。

---

## 6. 对本仓 2× / 4× / 8× 5090 的硬经验（保效果）

合同锁死：INT8 ConvRot + `sdpa=cudnn` + 8-step + cfg=1 + 864×480 有声。KPI 另册。门：相对单卡 5090 cudnn cafe 同 seed 的 PPM MAE ∩ 有声。

1. **先 TP=1，用 Ulysses 切 packed 序列。** 2× → U=2；4× → U=4；8× → U=8（56%8=0，64%8=0）。不要从 cookbook 的 5090 TP2+offload 起步——那是 BF16 容量解，本仓 INT8 单卡已常驻。
2. **复制 INT8 权，不要切权。** TP=1 无 per-block all-reduce。32GB 上 864×480 已证明一份权常驻；多卡付的是显存税，换的是 DiT 算力。
3. **2 卡优先 CUDA-IPC A2A（P2P 开着），4+/U>2 走 NCCL all-to-all。** 不要设 `NCCL_P2P_DISABLE`。IPC 要求同主机、两卡、`can_device_access_peer`；失败则退 NCCL，数值位级相同。
4. **输入 QKV 打成一次 destination-major 集体**，输出再一次 A2A；复用 staging，避免每层三次 rendezvous。Ulysses A2A **立刻 wait**（`_usp_all_to_all_single`），不要幻想 Ring 那种 hop/compute overlap——金线也不要开 Ring（H3 ring 要 FA backend，与 cuDNN 冲突）。
5. **Ulysses 组放连续 rank / 同 NUMA。** `set_seq_parallel_pg_by_sp_groups(..., use_ulysses_low=True)` 把 Ulysses 放 SP 组内连续下标。跨 NUMA 的 SYS 仍是 PCIe，只是 hop 更差。
6. **合法度先算头与 64 对齐**，再谈卡数。非法组合直接 fail-fast（抄 `_validate_sequence_parallel_config`），不要 silent pad 把头切碎。
7. **质量门独立于墙钟。** 不过门则 SP 留实验分支。禁止叠 Sage / Cache / 4-step 来「先把秒数做漂亮」。
8. **Amdahl：只切 DiT。** 5090 cafe e2e ~66s ≈ TE 12–14 + DiT ~31.4 + VAE ~10.5。2 卡理想也到不了 33s。e2e 要再砍，VAE tile/chunk 并行是另一把刀（FastVideo bit-exact chunk 可参考），TE 单请求几乎无收益。
9. **8×5090 不要默认抄 RH 的 TP2+U4。** 他们 T=2 是 6000D BF16 768p 工作集（60–66 GiB/卡）下的折中；本仓 INT8+cudnn 应先量纯 Ulysses8。只有 U=8 显存/实现爆了才考虑 TP2+U4。
10. **通信税可见、但形状优于 AR。** 未测不算。2×5090 若 DiT 落到 ~20s 量级已算拓扑赢；「2 卡 → 33s e2e」忽略了 TE+VAE 地板。

---

## 7. 不能抄

| 禁 | 为什么 |
| --- | --- |
| 4-step / turbo LoRA 进金线 | 换轨迹；12.2× 的 95.5% 来自这里，与 PCIe 无关 |
| Sage2 / sage3 / Cache-DiT / `torch.compile` 进默认 | 改核或改数值；本仓 sage3 MAE~27 已 FAIL；compile 无 C 对等物 |
| Python serving / SGLang `multimodal_gen` 当质量栈 | 本仓静态 C 管道 |
| 把 28.7s / 12.2× 写成 8 卡相对 1 卡或相对 66s | 分母是同机 50-step 8 卡 |
| 默认 Ring / CFG 并行 / PP | Ring↔FA；cfg=1 无第二分支；PP bubble |
| 关 P2P 当「无 NVLink 最佳实践」 | RH 没关；2 卡 IPC 还显式 `cudaDeviceEnablePeerAccess` |
| FastVideo 的 U=7 | H3 packed 64 对齐在 SGLang 路径非法 |

未 git commit。默认路径未改。
