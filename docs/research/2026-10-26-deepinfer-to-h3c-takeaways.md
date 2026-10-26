# DeepInfer → h3.c 取舍清单

> **日期：** 2026-10-26。  
> **对象：** MiniMax-H3 原生推理仓 `h3.c`（INT8 默认 DiT、NVFP4 实验臂、sm_120 / 5090 CUDA 旁路、单请求离线 CLI）。  
> **范围：** DeepSeek [DeepInfer](https://github.com/deepseek-ai/DeepInfer) 整套方案里，**哪些只对我们有启发或可复用**；哪些明确不搬。  
> **前置：** 与 `2026-10-07-llm-to-dit-transfer.md` 裁决一致；不重复诉讼该文已闭环死路。  
> **身份澄清：** DeepInfer 是 **大规模 LLM 推理框架**（公开宣传面含 DeepSeek-V4.1-Flash、NVIDIA SM100 / 昇腾等），**不是** 视频 / DiT / few-step 蒸馏方案。本仓无 `DeepInfer` 字符串；口语若另有所指，需另开笔记。

---

## 1. 一句话裁决

DeepInfer 对 `h3.c` **可复用的是工程纪律与流水线意识**，不是算子与 serving 核。  
算力主杆仍在本仓既有计划：**NVFP4 视觉门** + **768p 显存 / layerwise streaming**；少步质量另走 **PDMD/DMAD 合同臂**（见 `2026-10-09-pdmd-dmad-loras.md`），与 DeepInfer 正交。

---

## 2. 值得留的（启发 / 可复用）

| # | DeepInfer 侧能力 | 对 h3.c 的映射 | 落地形态 | 优先级 |
|---|---|---|---|---|
| A | Batch-invariant + deterministic sampling（同输入同输出、跨 batch 组成不变） | 对齐「cuDNN SDPA CUDA graph 默认关」后的 velocity md5 / 公平对照 | 对照硬门：固定 seed、prompt、token-ids；显式记录 `sdpa_graph` / TE 路径（`H3_TEXT_H_LOAD` 等）；跨配置禁止混用 waffle 金线 | **P0 纪律**（持续） |
| B | Async zero-overhead：CPU 调度与 GPU 执行重叠 | TE encode 释放 ↔ DiT 权重上传；VAE 预热 ↔ DiT 首步（PDMD r1 已踩预热 OOM） | 流水线 / 生命周期审计；不引入 DeepInfer 调度器代码 | **P2 工程** |
| C | 「热路径少决策」的反向启发（他们用 plan-run/JIT 服务多 shape；我们 shape 固定） | INT8 GEMM 已近 sm_120 峰值 → 热路径继续 **单 binary + 预设合同**，勿塞运行时分支 | 与 preset（INT8 / NVFP4 / PDMD-4 / DMAD-4）绑定；错合同加载 FAIL | **P1 合同** |
| D | Prefill/Decode 分离、continuous batching 等 serving 形态 | **仅当** 产品变成多并发出片服务时再议 | 记产品选项；当前 CLI **刻意不做** | **观察项** |

### 2.1 A — 可复现门禁（最小清单）

对照实验写入硬门，缺一项即不算「公平」：

1. `H3_SDPA_CUDNN_CUDA_GRAPH` 默认关（`321538a`）；若临时打开必须写入报告标题。  
2. 同 seed / 同 prompt / 同（或明确记录的）token-ids。  
3. TE 路径标明：live encode vs refine/`H3_TEXT_H_LOAD`；`c_te_raw` 偶发不确定时速度场诊断优先 refine 路径。  
4. 量化臂隔离：INT8 金线 vs `H3_DIT_NVFP4=1`；少步 LoRA **先钉 INT8**，再可选叠 NVFP4。  
5. waffle：**仅同配置接缝回归**；跨权重 / 跨合同禁止当质量门。

### 2.2 B — 异步重叠（检查点，非新框架）

- VAE 预热与 resident LoRA / 大分辨率同时驻留的显存峰值。  
- TE 输出生命周期：encode 完成后尽快释放、与 DiT 上传重叠是否安全。  
- 768p 全序列 OOM：优先显存账本 + streaming，而不是「上 DeepInfer 调度」。

### 2.3 C — 合同绑定（与 DeepInfer 无关但由其叙事反推）

- PDMD：Euler + 指定 sigma/shift；**禁止** turbo4 混用。  
- DMAD：需 `lcm`/re-noise 等价采样器后方可诚实验收。  
- 速度比宣传：仅同合同、同 graph/TE 状态；错合同计时作废。

---

## 3. 相邻但非 DeepInfer 独有（本仓已标【部分迁移】）

以下常被 LLM 推理栈一起讨论，**不全是 DeepInfer 专有**，却对 `h3.c` 更实在。裁决沿用 `2026-10-07-llm-to-dit-transfer.md`：

| 项 | 裁决 | 备注 |
|---|---|---|
| NVFP4 / FP4（sm_120 原生） | 【部分迁移】 | 最大算力杆；**视觉/waffle 门未过前不作产品默认** |
| Layerwise weight streaming | 【部分迁移】 | 768p 能力解锁优先于墙钟 |
| PDL（自写 kernel 边界） | 【部分迁移】 | 小收益；排在 GEMM+epilogue 融合后；cublasLt 不暴露 PDL |
| Hadamard 激活旋转 | 【部分迁移】 | 跟 NVFP4/W4A4 捆绑，非独立项 |
| FA3 式 warp specialization（dense SDPA 段） | 【部分迁移】 | sol_cute 线演进，非新开 DeepInfer 线 |

---

## 4. 明确不搬（否定清单）

| DeepInfer / LLM 能力 | 原因 |
|---|---|
| FlashMLA / GQA / MQA / 长 KV | DiT **无跨步 KV cache** |
| DeepEP / Wide EP / MoE 路由 | DiT **无 MoE** |
| SWA bounded replay | 无 SWA / KV 持久态可重放 |
| DSpark 类投机解码进 **默认路径** | 4 NFE 基数紧、无 draft 权重；最多观察项 |
| DeepGEMM JIT / FlashInfer plan-run 当主加速 | 固定 shape；热路径零决策已是本仓优势；同精度 GEMM 余量见 transfer 文 |
| Mirage 式整模型 megakernel | 负载形态相反；launch 税本仓已测很小 |
| SM100 / TMEM / tcgen05 专属路径 | 硬件面 ≠ 默认 sm_120 / 5090 |
| PagedAttention / continuous batching（当前产品） | 单请求离线 CLI；无 KV 分页对象 |

---

## 5. 与 10-09 主线的衔接

1. **不打断** `2026-10-09-dev-plan-and-progress.md`：先 INT8 稳 → NVFP4 视觉门。  
2. DeepInfer 取舍 **不新增算力主线**；只强化对照纪律与可选异步审计。  
3. T3（PDMD/DMAD）保持 **隔离臂**；采样合同错误不得归因于「没上 DeepInfer」。  
4. 磁盘 / nsys / EVT 清理仍 **等用户授权**（与本清单无关）。

---

## 6. 外部链接

- DeepInfer 仓库：https://github.com/deepseek-ai/DeepInfer  
- 说明汇总：https://openlm.ai/deepinfer/  
- 本仓 LLM→DiT 裁决：`docs/research/2026-10-07-llm-to-dit-transfer.md`  
- 进展主文档：`docs/research/2026-10-09-dev-plan-and-progress.md`  
- PDMD/DMAD：`docs/research/2026-10-09-pdmd-dmad-loras.md`

---

## 7. 修订记录

| 日期 | 说明 |
|---|---|
| 2026-10-26 | 初稿：从 DeepInfer 整案筛「启发/复用」与否定清单，对齐既有 transfer 裁决。 |
