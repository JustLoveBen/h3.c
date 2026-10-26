# DeepInfer 二次深挖 → h3.c（模块级 + 非显而易见工程）

> **日期：** 2026-10-26（Asia/Singapore）。  
> **对象：** MiniMax-H3 原生推理仓 `h3.c`（INT8 默认 DiT、NVFP4 实验臂、sm_120 / 5090、单请求离线 CLI）。  
> **方法：** 对照 [DeepInfer README](https://github.com/deepseek-ai/DeepInfer)、[openlm.ai/deepinfer](https://openlm.ai/deepinfer/)、[DeepInfer Explained](https://explaindeepinfer.projects.thomaspaul.dev/)（commit `9bd9cf2`）、公开新闻（The New Stack 2026-10-20）。  
> **前置（不重复诉讼）：**  
> - [`2026-10-26-deepinfer-to-h3c-takeaways.md`](2026-10-26-deepinfer-to-h3c-takeaways.md) — 一句话裁决 + A/B/C/D 粗筛  
> - [`2026-10-07-llm-to-dit-transfer.md`](2026-10-07-llm-to-dit-transfer.md) — MLA/MoE/Marlin/MPK/FA4/JIT 等死路  
> **标签：** 【新启发】本轮新抬；【已覆盖】取舍文或仓内已落地；【仍 N-A】DiT / 本产品形态无同构。

---

## 0. 一句话

第一轮取舍文没有写错：**算子与 serving 核仍搬不动**。二次深挖的增量几乎全在 **工程元纪律**——env 合同、sync 守卫、测量卫生、graph 政策、资源延后释放、黄金轨迹对照——这些才是 DeepInfer 开源版真正对 `h3.c` 值钱的部分。

---

## 1. 模块逐项

每条格式：DeepInfer 机理 → DiT 是否同构 → h3.c 可行动 → 标签。

### 1.1 Scheduling（ZOS / 步循环）

| | |
|---|---|
| **机理** | `DEEPINFER_ZERO_OVERHEAD_SCHEDULING`：CPU 先 `schedule(N+1)` + 入队 forward，再 post-process(N)；device 侧 `compute_seqlens_k_zos` 修正草稿接受数；`step()` 晚一拍返回 token；DP 投票改走 Gloo，避免 `.item()` 拆掉重叠。 |
| **同构？** | 部分。我们无 decode 步循环 / MTP 草稿，但有 **多阶段流水线**：TE encode ∥ DiT host_stage、DiT step ∥ VAE prewarm。 |
| **可行动** | 保持 / 强化既有门控（NVFP4 禁止 host_stage 与 DiT overlap；2NFE 禁止 VAE device-W 抢 resident）。不引入 ZOS 调度器。见 [`2026-10-26-te-lifecycle-audit.md`](2026-10-26-te-lifecycle-audit.md)。 |
| **标签** | 【已覆盖】流水线意识；【仍 N-A】ZOS/seqlen 修正本体 |

### 1.2 Batching / continuous batch / P/D 分离

| | |
|---|---|
| **机理** | 一步内不混 prefill+decode；chunked prefill；continuous batching；生产侧 E/P/D 拆分（开源版缺 KV store / LB）。 |
| **同构？** | 当前 CLI **否**。阶段隔离思想：TE / DiT / VAE 硬相位，勿在 DiT step0 偷塞 VAE device alloc。 |
| **可行动** | 产品若变多并发出片再议；现阶段只记观察项。 |
| **标签** | 【已覆盖】观察项 D；【仍 N-A】continuous batch / PagedAttention |

### 1.3 Memory（KV pages / pools / DeferredRelease）

| | |
|---|---|
| **机理** | BlockManager 分页 KV；`DeferredRelease`：step 产出的 metadata / RequestOutput **延后一轮**在 `pre_sampler_sync` 回收，与 GPU sample 重叠；多模态 pinned host pool（`DEEPINFER_USE_MULTIMODAL_POOL`）。 |
| **同构？** | 无 KV 页。【新启发】「**缓冲多活一轮，释放与下一阶段 launch 重叠**」对 TE scratch / TE store 释放节奏是同构纪律。 |
| **可行动** | TE audit 已要求 encode 后立刻 `free(text_emb)` + store_free；可选：把「允许延后到 DiT step0 event 之后再 free 的对象」写成白名单，避免过早 free 造成 UAF，也避免过晚 free 顶掉 resident。多模态 pinned pool：**仅当** TE/VAE host 碎片成为实测问题时再议。 |
| **标签** | 【新启发】DeferredRelease 纪律；【仍 N-A】KV pages；【部分】pinned pool |

### 1.4 Kernels（FlashMLA / DeepGEMM / DeepEP / TileLang / PDL）

| | |
|---|---|
| **机理** | 子模块核库；worker 启动时 `deep_gemm.set_pdl(True)` + `tile_kernels.set_pdl(True)`；batch-invariant 时切 deterministic algorithms。 |
| **同构？** | FlashMLA/DeepEP **N-A**（无跨步 KV / 无 MoE）。PDL / GEMM+epilogue 融合 = transfer 文【部分迁移】，非 DeepInfer 独有。 |
| **可行动** | 不追核；PDL 仍排在融合审计后。 |
| **标签** | 【已覆盖】否定清单 + transfer §3 |

### 1.5 Quantization

| | |
|---|---|
| **机理** | V4.1 路径含 FP8 KV 等；与 DeepGEMM deterministic 开关耦合。 |
| **同构？** | 本仓主杆是 **INT8 / NVFP4 DiT**，裁决已在 transfer + 10-09 主线。 |
| **可行动** | 不从 DeepInfer 搬量化格式；继续 P0 视觉门。 |
| **标签** | 【已覆盖】 |

### 1.6 Speculative decoding（DSpark / MTP / DFlash）

| | |
|---|---|
| **机理** | `DEEPINFER_USE_NEXTN_DECODE` + confidence-scheduled `MTP_VARLEN`；草稿预算跨请求共享 → **破坏** deterministic sampling，故确定性配方必须 `MTP_VARLEN=0`。 |
| **同构？** | 4 NFE 基数塌缩；无 draft 权重。ASDSV 类扩散投机 = transfer 观察项。 |
| **可行动** | 不进施工；**新启发（旁证）**：任何「可变长草稿 / 可变验证长度」的加速臂，必须从确定性合同里 **显式开除**（与 fewstep 合同隔离同构）。 |
| **标签** | 【仍 N-A】默认路径；【新启发】可变长臂开除确定性 |

### 1.7 Observability（StatsLogger / profile HTTP / memory tracing）

| | |
|---|---|
| **机理** | `StatsLogger` 滑窗 MFU/SPS/TPS；异步 log worker；`DEEPINFER_LOGGING_INTERVAL_STEP=0` 关日志；benchmark **双窗**（先有 log 半段读 moe_lb，再静默半段出数）；丢弃与 LB stats flush 碰撞的 interval；`PROFILE=1` 后仍有 7–11% 残余减速，**禁止**在 profiler 后再测。 |
| **同构？** | **高度同构**于本仓墙钟 / waffle / nsys 纪律。 |
| **可行动** | 强化 timing 账本规则：测墙钟时关 `H3_GEN_MEM_TRACE` / 重 debug printf；nsys/EVT **后**不得当 KPI；`timing_append_row` 仅 PROTO_OK 后。可选：generate 日志头打印合同 env 摘要（见 §3 行动 #1）。 |
| **标签** | 【新启发】测量卫生 / 双窗 / flush 碰撞 |

### 1.8 Determinism / batch invariance

| | |
|---|---|
| **机理** | `BATCH_INVARIANT=1` → deterministic DeepGEMM/TileKernels + FlashMLA `enable_batch_invariant` +（条件）DeepEP 单次 reduction。确定性采样再加 `MTP_VARLEN=0` + 固定 `seq_seed`。测试：`test_batch_invariant` **batch vs single + golden_token_ids**（比同一 token 轨迹上的 logprobs，`==` 零容差）；`test_kl` decode vs 全量 prefill KL≤0。 |
| **同构？** | 本仓无「跨请求组批」；有 **跨配置 / 跨 TE 路径 / 跨 graph 状态** 的伪对照问题。golden 轨迹思想直接迁移。 |
| **可行动** | P0 已要求 graph 关 + TE 路径标明。**【新启发】** 速度场 / 1-step md5：强制 `H3_TEXT_H_LOAD` 同一 refine dump（golden conditioning），禁止 live TE 漂移伪装成 DiT 差；视觉门仍用 live TE（产品路径）。 |
| **标签** | 【已覆盖】A 纪律；【新启发】golden conditioning / 零容差轨迹 |

### 1.9 Multi-device / EP / CP / Ascend

| | |
|---|---|
| **机理** | Attention DP / balanced CP / MoE EP+EPLB；SM100 + Ascend 950 并列；开源缺 replica LB。 |
| **同构？** | 单卡 5090 默认路径 **N-A**。多卡旁路若存在，不从 DeepInfer 抄 EPLB。 |
| **可行动** | 无。 |
| **标签** | 【仍 N-A】 |

### 1.10 CUDA graphs（政策，非「再开 graph」）

| | |
|---|---|
| **机理** | 仅 decode 捕获；`ENFORCE_EAGER` 总开关；按 batch size 列表 pad+replay；**共享 graph memory pool**（先大后小）；ZOS 时 input 已在 device、图内无 H2D copy；`forbid_cuda_sync` 包住 replay；关机先毁 graph 再毁 NCCL 防挂死；DFlash 独立 graph runner。 |
| **同构？** | 本仓 **刻意默认关** SDPA cuDNN graph（sm_120 可复现）。政策层同构：graph 是 **合同字段**，不是隐式加速。 |
| **可行动** | 已有 `sdpa_graph: wanted=0` 日志硬门。【新启发】若未来对固定 shape DiT 子图再开 graph：① 独立 env 开关；② 捕获前 clean 派生字段；③ 销毁顺序写进关机路径；④ waffle/视觉门默认仍要求 off。 |
| **标签** | 【已覆盖】默认 off；【新启发】graph 生命周期 / pool / 销毁顺序清单 |

### 1.11 Config-as-contract（environ.py）

| | |
|---|---|
| **机理** | 59 个 `DEEPINFER_*` 集中 `EnvVar` 描述符：首次读解析并缓存；`rearm`；bool 仅 `0/"" /false/False` 为假（**`off`/`no` 反而是真**）；错误带变量名；文档明确「先设 env 再 import」；建议打印 `{k: getattr(envs,k) for k in _descriptors}`。 |
| **同构？** | **高度同构且本仓缺口大**：约 **285** 个分散 `getenv("H3_*")`，无注册表、无启动摘要、易「以为关了其实 on」。 |
| **可行动** | **离线优先：** `tools/print_h3_env_contract.py`（P0/少步/生命周期关键键 + 生效值 + 期望）。长期可选：薄封装 `h3_env_*` 集中解析（不挡主线）。 |
| **标签** | 【新启发】**本轮最高价值复用** |

### 1.12 Failure taxonomy / sync guard

| | |
|---|---|
| **机理** | `forbid_cuda_sync()`：forward/sample launch 内禁止 blocking sync / blocking memcpy；允许预定 `Event.synchronize`；`DISABLE_SYNC_GUARD=1` 仅 debug。Env 页「What can go wrong」把 `off`=on、设晚了静默忽略、YOCO keep 太小等列成可检索故障表。 |
| **同构？** | C/CUDA 无 Python 上下文管理器，但 **「热路径禁止意外 host sync」** 同构；故障表同构。 |
| **可行动** | 扩 TE audit / followup：列出允许的 sync 点（TE join、VAE wait、nsys 边界）与禁止点（DiT 层间 `cudaDeviceSynchronize` 当 debug 遗留）。可选 debug：`H3_FORBID_HOT_SYNC=1` 用宏包一层（工作量大，P3）。先做文档清单。 |
| **标签** | 【新启发】sync 白名单 + 故障表 |

### 1.13 Structured outputs / recipe / eval 两段式

| | |
|---|---|
| **机理** | xgrammar 约束解码；deepseek-recipe 复现 API；eval **生成与打分分离**，`meta.json` 存完整配置；PPL baseline 精确逐样本。 |
| **同构？** | 结构化输出 N-A。**【新启发】** dump-once / score-many：`H3_DUMP_LATENT` + `H3_REPLAY_LATENT` 已有雏形 → 视觉/waffle 应用「一次出片、多次门」时写清 META。 |
| **可行动** | Runbook META.txt 已接近；env 合同打印机补齐可机读摘要。 |
| **标签** | 【新启发】生成/评分分离；【仍 N-A】structured outputs |

### 1.14 其它边角

| 项 | 标签 | 备注 |
|---|---|---|
| `CUDA_SCALE_LAUNCH_QUEUES=4x`（import 时强制） | 【新启发·弱观察】 | 原因未文档化；仅当 launch 队列成为 nsys 热点再试 |
| Prefill attempt ALL/ANY 投票 | 【仍 N-A】 | 多 DP 服务语义 |
| YOCO / SWA bounded replay | 【仍 N-A】 | 无 SWA |
| Engram / multimodal pool GiB | 【仍 N-A】/【部分】 | 见 §1.3 |
| Platform dual（NVIDIA+Ascend） | 【仍 N-A】 | 本仓 sm_120；不追 SM100/昇腾路径 |

---

## 2. 非显而易见启发汇总（相对取舍文 §2）

取舍文已有：A 可复现、B 异步重叠、C 热路径少决策、D serving 观察。  
本轮新抬（按对离线开发的价值排序）：

| # | 启发 | 为何第一轮没写透 | 离线可落地？ |
|---|---|---|
| N1 | **Config-as-contract + 启动打印生效 env** | 只写了「合同绑定」，没挖 `environ.py` 机制与 bool 陷阱 | ✅ `print_h3_env_contract.py` |
| N2 | **测量卫生**（关日志半段、丢 flush 碰撞、profiler 后不作 KPI） | 只写了「耗时追加 timing 文档」 | ✅ 文档规则 + 可选检查 |
| N3 | **Golden conditioning / 同轨迹对照**（batch_invariant 测法） | 只写了 seed/prompt，没写「强制同一 text_h」 | ✅ runbook 一句 + 已有 `H3_TEXT_H_LOAD` |
| N4 | **Sync 白名单 / forbid hot sync** | 未提 `forbid_cuda_sync` | ✅ 清单；代码守卫 P3 |
| N5 | **DeferredRelease：缓冲 +1 步寿命** | B 只写了重叠检查点 | ✅ TE audit 补一句 |
| N6 | **Graph 政策：pool、clean 派生字段、销毁顺序** | 只写默认 off | ✅ 政策备忘 |
| N7 | **可变长加速臂开除确定性合同** | 投机解码整段 N-A | ✅ 合同文档一句 |
| N8 | **生成/评分分离 + meta 完整配置** | 未提 eval 两段式 | ✅ META / env dump |

---

## 3. 排序后的新行动项（离线优先）

| 优先级 | 行动 | 形态 | 依赖 |
|---|---|---|---|
| **P0-dev** | `tools/print_h3_env_contract.py`：打印 P0/少步/生命周期关键 `H3_*` 的 raw + 解释；`--expect-p0-visual` 对违例 exit≠0 | 代码（本轮落地） | 无 GPU |
| **P0-doc** | P0 runbook / timing：测墙钟时禁止 mem_trace；速度场对照强制同一 `H3_TEXT_H_LOAD` | 文档 | 无 |
| **P1-doc** | TE audit 增补：DeferredRelease 白名单 + 允许 sync 点表 | 文档 | 无 |
| **P1-doc** | Graph 政策备忘（若再开子图）：pool / clean / destroy-before-NCCL | 文档（可附本文件 §1.10） | 无 |
| **P2** | 可选 `H3_FORBID_HOT_SYNC` 宏（工作量大） | 代码 | 真机验证 |
| **P3** | 集中 `h3_env` 注册表（替代 285 处裸 getenv） | 重构 | 不挡主线 |
| **不做** | 搬 ZOS / FlashMLA / DeepEP / DSpark / SM100 / continuous batch | — | — |

主线不变：INT8 稳 → NVFP4 视觉门；少步 PDMD/DMAD 隔离臂。DeepInfer 二次深挖 **仍不新增算力主线**。

---

## 4. 外部锚点

| 资源 | 用途 |
|---|---|
| https://github.com/deepseek-ai/DeepInfer | 源码；README 特性表 |
| https://openlm.ai/deepinfer/ | 官方说明镜像 + benchmark/eval 细节 |
| https://explaindeepinfer.projects.thomaspaul.dev/ | 机制深潜（ZOS、determinism、env、CUDA graphs）@ `9bd9cf2` |
| The New Stack 2026-10-20 | 发布背景：离线引擎、缺 KV store / E-P-D LB；SM100+Ascend |

---


## 6. 算子 / serving 核：质量与效率（专项，回应「搬不动之外」）

> **问题：** DeepInfer 的算子与 serving 核，在**它们自己的领域**里质量和效率到底怎样？为何对 h3.c 仍「搬不动」？有没有不搬核也能带走的薄启发？  
> **证据源：** DeepInfer / FlashMLA / DeepGEMM / DeepEP / TileKernels README（公开自称数字）；ExplainDeepInfer 目录图（`9bd9cf2`）；本仓 C5b / transfer / deadend-operators。**未在本机复跑他们的微基准。**

### 6.1 他们在优化什么（负载形状）

| 目标负载 | DeepInfer 栈在消什么 |
|---|---|
| **自回归 decode** | 每步 1（或少量）token × 全历史 **KV 带宽**；小 kernel 海量、launch/气泡占比高 |
| **Prefill（长上下文）** | 稀疏 MLA + FP8/FP4 KV；融合 Q-norm/RoPE/attn/cast 消小核往返 |
| **MoE** | Expert 路由 + **EP all-to-all**（DeepEP）与 grouped GEMM（DeepGEMM Mega MoE）重叠 |
| **多请求 serving** | Continuous batch / 一步内不混 P+D / 异步调度（ZOS）/ 草稿投机（DSpark） |
| **硬件** | **SM100（B200 等）+ Ascend 950**；README 明确不把 GeForce sm_120 当支持面 |

一句话：这是 **「decode 带宽 + MoE 通信 + 多请求调度」** 的生产栈，不是「固定大 shape、双向全注意力、无跨步状态」的视频 DiT 栈。

### 6.2 质量（determinism / 数值路径 / 成熟度与风险）

**领域内：高质量，且把「可复现」当产品能力。**

- **Determinism 分层清楚：** `BATCH_INVARIANT=1` → DeepGEMM / TileKernels / FlashMLA 切 deterministic；再要采样可复现必须 `MTP_VARLEN=0` + 固定 `seq_seed`。测试用 batch vs single **零容差** token 轨迹（golden_token_ids），不是「差不多就行」。
- **数值路径：** 为 V4.1-Flash 定制——FP8/FP4 KV、稀疏 indexer、Mega-MoE、fused TileLang ops；质量合同是 **LLM token/logprob 轨迹**，不是视频 waffle/视觉门。
- **成熟度证据（公开自称）：** FlashMLA 报 B200 fused sparse 达 **1460 TFLOPS prefill / 950 decode**；Ascend sparse prefill **95% 硬件峰**、decode **83%**；DeepGEMM 自称在多 shape 上匹配或超过 expert-tuned，历史 H800 峰值宣传到 **1550 TFLOPS**；TileKernels 自称「多数近算力或带宽上限」且已进内部训练/推理。DeepInfer 自身给了 4×SM100 prefill / 64× decode 的 Pareto 图（需按 `run_sweep.sh` 复现）。
- **风险（诚实）：**
  1. **硬件锁死 SM100/Ascend** —— 对 5090/sm_120 **零直接质量背书**（FlashMLA 2026.09 甚至砍掉 Hopper 与旧模型兼容）。
  2. **模型锁死 V4.1-Flash** —— 不是通用 DiT/视频引擎。
  3. **开源不完整** —— vendor 子模块空克隆、缺生产 KV store / E-P-D LB；「生产同款」≠「clone 即跑满」。
  4. **加速臂与确定性互斥** —— MTP/DSpark 跨请求共享草稿预算会破坏 deterministic；他们自己写进配方，不是隐藏坑。
  5. **自称 TFLOPS 多为目标硬件微基准** —— 未在本仓交叉验证；引用时只作「领域天花板」证据，不作 h3.c KPI。

**对 h3.c 的质量含义：** 他们的「高质量」= 同输入同 token + batch 组成不变。我们可偷的是 **合同纪律**，不是他们的数值格式或注意力数学。

### 6.3 效率（Amdahl：赢从哪来）

公开叙事里的加速，拆开几乎全落在 LLM serving 的 Amdahl 分母上：

| 赢点 | 机理 | 在 LLM 里为何大 | 在 h3.c DiT 里 |
|---|---|---|---|
| FlashMLA / 稀疏 attn + 低精度 KV | 砍 decode/prefill 注意力带宽与 FLOPs | decode 主导延迟 | **无 KV**；注意力是当步双向全量，属另一族问题（sol / SDPA） |
| DeepGEMM FP8/FP4 + Mega MoE | 更高峰值档 GEMM + 算通重叠 | MoE 层 + 变长 token 组 | **无 MoE**；INT8 GEMM 已 **98–99% sm_120 峰**（C5b），同精度 GEMM **无余量** |
| DeepEP | EP dispatch/combine 低延迟 | 多卡 MoE 通信墙 | 单卡默认路径 **N/A** |
| TileKernels 融合 quant/SwiGLU/RoPE | 消小核与中转 | decode 小核海 | **部分同构**：我们痛的是 quant/dequant/cast **链路往返**（~数秒级账本），不是缺一个 LLM 专用 TileLang 核 |
| ZOS / 异步调度 | CPU 调度与 device 重叠 | 每 token 一步、请求多 | CLI 单请求；阶段重叠已有 TE∥DiT / VAE 门控，ZOS 本体过重 |
| DSpark 投机 | 少跑 verified forward | decode token 税 | 4 NFE 基数已塌；无 draft 权重 |

**Amdahl 对照本仓（480p 量级记忆）：** DiT 热路径里 INT8 GEMM 本征已近峰；端到端大头还在 SDPA、量化中转、VAE、以及（少步臂上的）步数。DeepInfer 核堆解决的「KV + MoE + 多请求 decode」在这条分母里 **权重接近零**。

### 6.4 与 h3.c 现实对打

| 维度 | DeepInfer 栈 | h3.c（5090 / sm_120） |
|---|---|---|
| 形状 | 变长 seq、变 batch、decode 极瘦 | **固定大 shape**（如 16k×大 K） |
| 状态 | 跨步 KV +（常）MoE 路由 | **无跨步 KV、无 MoE** |
| GEMM 精度档 | FP8/FP4 原生（SM100） | 默认 **INT8** 已近峰；NVFP4 是**另一更高峰值档**的实验臂 |
| 产品形态 | 多副本、P/D、LB（开源缺后半） | 离线 **单请求 CLI** |
| 质量尺 | token / logprob / batch-invariant | waffle / 视觉门 / velocity md5 |
| 硬件 | SM100 + Ascend | **sm_120**（寄存器 MMA，无 TMEM 子集） |

所以：不是「他们的核差」，是 **问题同构性为零的高效解**。把 FlashMLA/DeepEP 搬进 h3.c = 用错药。

### 6.5 诚实裁决

1. **在 DeepSeek LLM 生产域内：算子与 serving 核可以评为高质量、高效率**——有峰值利用率自称、确定性产品开关、融合与通信重叠设计、且绑死自家模型/硬件做端到端 Pareto。  
2. **对 h3.c：效率不可迁移的主因是 Amdahl + 硬件 + 无 KV/MoE**，不是「开源质量差所以别碰」。  
3. **仍值得带走、且不搬核的薄启发（质量/效率卫生）：**
   - **把确定性当一等公民**（对照门用 golden conditioning，加速臂显式开除确定性合同）——已在 N3/N7。  
   - **大核旁的小 op 融合 / 消中转**（他们用 fused MLA+cast；我们对应 quant↔GEMM↔SwiGLU 链路与 EVT 候选）——属 transfer【部分迁移】，非 DeepInfer 独家。  
   - **测量与 env 合同**（测速关日志；profiler 后不作 KPI）——N1/N2。  
   - **不要在已近峰的同精度 GEMM 上再找「更快的核」**——DeepGEMM 的故事是换精度档/换 shape 族；我们同档已闭环（C5b）。

**一句话补强取舍文：** DeepInfer 核堆 **在域内又快又稳**；对视频 DiT **又快又稳地解决错问题**。继续主线 INT8 稳 → NVFP4 视觉门；算力杆不在搬他们的 serving 核。

---
---

## 7. 策略层补强（纠正「LLM ⇒ 无价值」）

§1–§6 解决的是：**模块能不能搬、核在域内好不好、Amdahl 是否同构**。  
那不等于「DeepInfer 作为世界级 LLM 加速工程没有可学的速度策略」。

完整策略分析（S1–S12，keep/adapt/defer）见：

- [`2026-10-26-deepinfer-speed-strategies.md`](2026-10-26-deepinfer-speed-strategies.md)
- 落点整合：[`2026-10-26-optimization-roadmap.md`](2026-10-26-optimization-roadmap.md)

与 §6.5 的关系：§6.5「又快又稳地解决错问题」针对的是 **KV/MoE/多请求核堆**；策略层明确 **S4 量化共设计、S5 融合、S1 重叠、S9 测量** 等仍 adapt/keep。

## 5. 修订记录

| 日期 | 说明 |
|---|---|
| 2026-10-26 | 初稿：模块级标签 + 8 条非显而易见启发 + 排序行动；落地 env 合同打印机。 |
| 2026-10-26 | §6：算子/serving 核质量与效率专项（域内高、Amdahl 不迁移、薄启发）。 |
| 2026-10-26 | §7：链到策略层文 + 优化路线图；澄清同构否决 ≠ 策略否决。 |
