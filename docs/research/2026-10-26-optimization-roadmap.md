# h3.c 后续优化方案（整合路线图 · 2026-10-26）

> **日期：** 2026-10-26（Asia/Singapore）。  
> **对象：** MiniMax-H3 原生推理仓 `h3.c`（默认 INT8 DiT、NVFP4 实验臂、T3 少步 LoRA、sm_120 / 5090 CUDA 旁路）。  
> **性质：** 把「算力 / 质量 / 显存 / 工程门禁」收成一页**优化方案**；执行清单仍以 [`2026-10-26-followup-dev-plan.md`](2026-10-26-followup-dev-plan.md) 为准，**本文不改其优先级与否定清单**。  
> **DeepInfer 速度策略层：** [`2026-10-26-deepinfer-speed-strategies.md`](2026-10-26-deepinfer-speed-strategies.md)。  
> **耗时账本：** 墙钟只追加 [`2026-10-09-generation-timing.md`](2026-10-09-generation-timing.md)。

---

## 0. 一句话

优化分四条正交轴，**禁止合成一个 SKU 验收**：

1. **每步算力**（INT8 稳 → NVFP4 视觉门 → 可选链式）  
2. **步数 / 合同**（PDMD-4 → DMAD `lcm_renoise`）  
3. **显存能力**（768p / streaming；能力优先于墙钟）  
4. **工程兑现**（可复现门禁、测量卫生、阶段重叠、融合消中转）——DeepInfer 世界级 LLM 加速的**策略**主要落在这条轴，不是「搬 FlashMLA」

总目标不变：四档 SKU（见 [`../superpowers/plans/2026-10-05-h3-optimization-roadmap.md`](../superpowers/plans/2026-10-05-h3-optimization-roadmap.md)）；默认路径零行为变化。

---

## 1. 原则（与 followup-dev-plan §1 对齐）

| # | 原则 | 含义 |
|---|---|---|
| 1 | 不打断主线 | INT8 稳 → NVFP4 视觉门 → 耗时另册 → 链式靠后 |
| 2 | 臂隔离 | INT8 / NVFP4 / chain / PDMD / DMAD 独立验收；禁跨臂 waffle / 墙钟当质量门 |
| 3 | 合同绑定 | 少步钉死 steps/shift/sampler/LoRA；错合同 FAIL |
| 4 | 投产门 = 视觉 / waffle | §12 cosine 只是调试尺 |
| 5 | graph 默认关 | 公平对照；临时开必须写进报告标题 |
| 6 | 授权项 | 清盘 / nsys / push / 链式提交 / 升默认 —— 等用户点头 |
| 7 | 策略可学、核勿盲搬 | LLM 加速的**速度策略**可迁移；问题不同构的**算子本体**不进主线（见速度策略文） |

---

## 2. 当前基线（优化起点）

| 轴 | 状态 | 阻塞 |
|---|---|---|
| INT8 默认 | graph 关后 1-step velocity md5 可复现；GEMM ~98–99% sm_120 峰 | 生产 TE `c_te_raw` 偶发不确定（速度场优先 refine/`H3_TEXT_H_LOAD`） |
| NVFP4 | 能跑完；公平 1-step vs INT8 轨迹差大；**P2.2 视觉门未按现基线在 238 重验** | 视觉门；链式 WIP 未提交 |
| 少步 T3 | 合同自检 / preset / `lcm_renoise` / host 测已齐 | **GPU 视觉验收未做** |
| 768p | 全序列两侧易 OOM | streaming / 显存账本；盘满时勿 nsys |
| 工程门禁 | env 合同打印、P0 日志硬门、waffle 同配置门、TE audit 笔记已落地 | 真机跑数与门禁串联 |

---

## 3. 优化组合（按 ROI / 依赖）

### 3.1 P0 — 质量地基 + NVFP4 算力门（主杠杆）

| 项 | 做什么 | 通过标准 | 依赖 | 明确不做 |
|---|---|---|---|---|
| **INT8 对照纪律** | graph 关；同 seed/prompt/token-ids；TE 路径标明；env 合同打印 | 1-step velocity md5 可复现；报告含 `sdpa_graph` / TE 路径 | `321538a`；`print_h3_env_contract.py`；P0 runbook | 开 graph 当默认；跨 TE 路径混 waffle |
| **NVFP4 P2.2 视觉门** | 对照**现在**稳定 INT8；chain 关；`--skip-audio` | 目检 + **同配置** waffle；不要求 bit-identical / cosine>0.99 | INT8 金线；旧片子作废 | 旧 SSIM/目检当通过；视觉未过升默认 |
| **测量卫生** | 记墙钟关 `H3_GEN_MEM_TRACE`；profiler 后不作 KPI | timing 行可审计 | `timing_append_row.py` | 用 graph-on / 错合同数字宣传 |

**为何是主杠杆：** 同精度 INT8 GEMM 已近峰；下一档算力来自 **NVFP4 精度档**（策略：量化与硬件共设计），不是再换一个「更快 INT8 核」。

### 3.2 P1 — 账本 + 少步合同最短路径

| 项 | 做什么 | 通过标准 | 依赖 | 明确不做 |
|---|---|---|---|---|
| **耗时账本** | 同协议 ≥2 次、标冷启动；只追加 timing 文档 | 可复述配置 | 视觉门前后均可记 | 墙钟替代视觉门 |
| **PDMD-4NFE** | 原件 + Euler + shift 12/3 + cfg 1 + INT8；`--fewstep-preset pdmd4` | 合同自检 PASS；480p 目检 | LoRA 路径；禁止 turbo4 | 用 2NFE 冲清晰度当 T3 通过 |
| **合同自检（已落地）** | GPU 少步跑时打开 `--fewstep-contract` / preset | 错合同 FAIL | host 测已过 | 错 sampler 冒充 DMAD |

### 3.3 P2 — DMAD + 阶段重叠 / 融合卫生

| 项 | 做什么 | 通过标准 | 依赖 | 明确不做 |
|---|---|---|---|---|
| **DMAD 验收** | `lcm_renoise` + `full_critic`；preset `dmad4` | 视觉门 + 合同 | 采样器已实现 | Euler 冒充 DMAD |
| **阶段重叠审计→可选实现** | TE↔DiT↔VAE 峰值与 DeferredRelease；安全后再重叠 H2D/encode | 审计笔记 → 无回归峰值 OOM | [`2026-10-26-te-lifecycle-audit.md`](2026-10-26-te-lifecycle-audit.md) | 引入 LLM 调度器框架 |
| **大核旁消中转** | quant↔GEMM↔激活 epilogue / 融合候选（EVT 等） | 墙钟有增益且 waffle 同配置过 | nsys 需授权；盘够 | 未剖析就大改热路径 |

### 3.4 P3 — 显存能力 + 链式（门控）

| 项 | 做什么 | 通过标准 | 依赖 | 明确不做 |
|---|---|---|---|---|
| **NVFP4 链式** | 视觉门通过 + 用户确认后再提交 WIP | 端到端可解释；768p 策略另议 | P0 视觉结论 | 视觉未过强推链式 |
| **Layerwise / streaming** | 768p 全序列能力 | 能跑完优先于加速比 | 显存账本 | 多卡进默认 |
| **sync 白名单守卫** | 可选 `H3_FORBID_HOT_SYNC` | 不增回归 | 真机 | 挡主线 |

---

## 4. DeepInfer 策略 → 本仓落点（摘要）

完整策略表见 [`2026-10-26-deepinfer-speed-strategies.md`](2026-10-26-deepinfer-speed-strategies.md)。优化方案里只钉**已映射到动作的**：

| 策略抽象 | 本仓动作 | 档位 |
|---|---|---|
| 可复现 / batch-invariant 当产品能力 | P0 对照硬门、env 合同、同配置 waffle | P0 |
| 测量卫生与回归 harness | timing 规则、日志协议检、fewstep host 测 | P0–P1 |
| 量化与硬件共设计（换精度档） | NVFP4 视觉门（非「再抠 INT8」） | P0 |
| 阶段重叠 / DeferredRelease | TE lifecycle → 可选重叠 | P2 |
| 融合与 epilogue（消中转） | quant/cast 链路 + EVT 候选 | P2 |
| 按 phase 特化（eager vs graph 政策） | SDPA graph 默认关；政策备忘 | 持续 |
| 投机 / 早退的**合同变体** | 少步 LoRA（PDMD/DMAD），非 decode 投机核 | P1–P2 |
| 多请求 batching / EP / KV | **defer**（CLI 单请求）；产品变 serving 再议 | 观察 |

---

## 5. 近两周执行顺序（与 followup §6 同步）

有序；`∥` = 可与主线并行的离线/少步臂：

1. [ ] **P0 视觉门**（238 授权后）：INT8 金线 → NVFP4 P2.2 目检 + waffle  
2. [ ] **P1 耗时**追加 timing（不挡视觉）  
3. [ ] **∥ PDMD-4NFE** GPU 合同跑  
4. [x] 合同自检 / preset / `lcm_renoise` / host 门禁（开发已齐）  
5. [ ] **P2 DMAD** `full_critic` GPU（授权后）  
6. [ ] **P2** 重叠/融合：审计已有 → 授权后 nsys → 小改  
7. [ ] **P3** 视觉结论后请示：链式提交 / 清盘 / streaming  

**本轮文档不触发 238 / 不碰链式 WIP。**

---

## 6. 明确不做（与 followup §5 一致，不扩诉）

- 本仓重训 PDMD/DMAD；2NFE 冲观感当通过  
- W4A8 DiT 主题默认；有损 magcache/reuse、Sage MMA、多卡默认  
- **把「LLM ⇒ 无价值」当借口停学策略**——停的是**不同构算子盲搬**，不是策略分析  
- 默认开 SDPA CUDA graph；§12 cosine 当产品完成标准  
- 热路径多 shape JIT 当主加速（固定 shape 已是优势）

---

## 7. 相关文档

| 路径 | 用途 |
|---|---|
| [`2026-10-26-followup-dev-plan.md`](2026-10-26-followup-dev-plan.md) | **执行 checklist 权威** |
| [`2026-10-26-deepinfer-speed-strategies.md`](2026-10-26-deepinfer-speed-strategies.md) | 速度策略层分析 |
| [`2026-10-26-deepinfer-to-h3c-takeaways.md`](2026-10-26-deepinfer-to-h3c-takeaways.md) | 取舍清单 |
| [`2026-10-26-deepinfer-deeper-pass.md`](2026-10-26-deepinfer-deeper-pass.md) | 模块级 + 核质量效率 |
| [`2026-10-26-p0-visual-gate-runbook.md`](2026-10-26-p0-visual-gate-runbook.md) | P0 命令 |
| [`2026-10-09-dev-plan-and-progress.md`](2026-10-09-dev-plan-and-progress.md) | 总目标上游 |
| [`2026-10-05-acceleration-mechanisms.md`](2026-10-05-acceleration-mechanisms.md) | 视频扩散机理族 |
| [`2026-10-07-llm-to-dit-transfer.md`](2026-10-07-llm-to-dit-transfer.md) | 算子同构裁决（≠ 策略否决） |
| [`2026-10-27-additional-optimization-opportunities.md`](2026-10-27-additional-optimization-opportunities.md) | 增量机会扫掠（E2/VRAM/TE/…） |

---


## 8. 修订记录

| 日期 | 说明 |
|---|---|
| 2026-10-26 | 初稿：整合 P0–P3 优化轴 + DeepInfer 策略落点；与 followup-dev-plan 对齐、不改其优先级。 |
| 2026-10-27 | 增 §9 指针 → [`2026-10-27-additional-optimization-opportunities.md`](2026-10-27-additional-optimization-opportunities.md)（E2/显存账本/TE cache 等增量，不改 P0）。 |
| 2026-10-27 | 增 §10 指针 → [`2026-10-27-full-optimization-catalog.md`](2026-10-27-full-optimization-catalog.md)（长周期全量目录）。 |

---

## 9. 增量机会指针（2026-10-27）

主线（§3 P0–P3）不变。另见扫掠文档 [`2026-10-27-additional-optimization-opportunities.md`](2026-10-27-additional-optimization-opportunities.md)：把 10-05/10-07 已论证、却未进入 10-26 执行表（或只写了一句「融合」）的项重新排序。

**最值得并行于主线、且不替代 NVFP4 视觉门的五条：**

| ID | 项 | 为何现在提 |
|---|---|---|
| A1 | `H3_DIT_EPI_FUSE`（E2）真机 A/B | 代码与 `ab.sh` 已在仓；同精度链路秒级；S5 实体 |
| A2 | 768p 显存账本插桩 | streaming/减窗/FP16 中转的决策前提 |
| A3 | TE `te_cache`/常驻产品化 | TE 7–58s 摆动污染加速比 |
| A4 | 激活 FP16 中转（账本后） | 768p 能力备胎 |
| A5 | sol autotune 是否 per-clip | 可能 cold −1s，或诚实关闭 |

完整 A1–A15、与 S1–S12 对照、否定清单见该文。**仍不跑 GPU 则优先可开发项：A10（generate 内联 env 合同）。**


## 10. 全量优化目录指针（2026-10-27）

长周期、穷尽式机会目录（去重 taxonomy + Closed 节 + C001+ 全表 + 2–8 周 backlog + 长地平线）见：

[`2026-10-27-full-optimization-catalog.md`](2026-10-27-full-optimization-catalog.md)

**本文（roadmap）与 followup-dev-plan 仍是近周执行权威**；全量目录用于「还有没有漏的」与数月排期，不替代 P0 视觉门优先级。

