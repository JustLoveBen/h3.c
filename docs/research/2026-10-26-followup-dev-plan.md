# 后续开发计划（2026-10-26）

> **日期：** 2026-10-26。  
> **对象：** `h3.c` CUDA / sm_120 旁路：INT8 默认 DiT、NVFP4 实验臂、T3 少步 LoRA、可选工程纪律项。  
> **性质：** 把 `2026-10-09-dev-plan-and-progress.md` 的「往后怎么做」与 DeepInfer / PDMD·DMAD 取舍收成一页可执行路线图；**不改**总目标（四档 SKU、默认零行为变化）。  
> **耗时：** 墙钟只追加到 [`2026-10-09-generation-timing.md`](2026-10-09-generation-timing.md)，本文不记加速比结论。

---

## 1. 原则

1. **不打断主线。** 顺序仍是：默认 INT8 稳 → 隔离臂 NVFP4 视觉门 → 耗时另册 → 链式靠后。  
2. **臂隔离。** INT8 金线、`H3_DIT_NVFP4=1`、`H3_DIT_NVFP4_CHAIN`、PDMD、DMAD 各自独立验收；禁止跨臂混用 waffle / 墙钟当质量门。  
3. **合同绑定。** 少步臂必须钉死 steps / shift / sampler / LoRA 原件身份；错合同加载应 FAIL（或明确标「非金线」）。量化压每步算力、LoRA 压步数——正交，不合成一个 SKU 验收。  
4. **投产门 = 视觉 / waffle，不是 cosine。** 链式文档 §12 的 cosine>0.99 / rel<0.05 仍是调试尺，不是产品完成标准。  
5. **graph 关。** 公平对照默认 `H3_SDPA_CUDNN_CUDA_GRAPH` 关（`321538a`）；临时打开必须写入报告标题。  
6. **授权项不擅动。** 磁盘清理、nsys、EVT、推送 origin、链式提交，均等用户明确授权。

---

## 2. 当前基线快照

- `main` @ origin-synced after `44e51c3`；本轮离线续作见 §2.1 新行（TE checksum / P0 硬门 / audio-env 冲突）。  
- 默认路径仍是 **INT8 DiT**；`H3_DIT_NVFP4=1` 可切实验臂；基线 `fc1_fused` 守卫已修（`fcd930c`）。INT8 1-step velocity md5 在 graph 关下可复现（`321538a` graph 默认关）。  
- NVFP4：**能跑完、latent 有限**；公平 1-step 相对稳定 INT8 轨迹差大（velocity rel ~0.94）；**P2.2 视觉门仍未在 238 按现基线重验**；链式 WIP 在工作区（端到端 latent rel ~0.33 未解释），**未提交，等视觉门 + 用户确认**。  
- T3（PDMD/DMAD）：**仍标未验证（无 GPU 视觉验收）**。PDMD-2NFE 合同复验过；**PDMD-4NFE 原件正式验收未做**。开发侧：`h3_fewstep_contract`（`7fe119c`）、`lcm_renoise`（`e037d42`）、`--fewstep-preset`（`8359dae`）已落地；禁止用 Euler/turbo4 冒充 DMAD 通过。Metal `h3` CLI **未**接 fewstep（无 LoRA/sampler 路径；验收走 `h3_cuda_generate`）。  
- DeepInfer：只取可复现门禁 + 可选异步重叠审计（**审计笔记已写**）；**不新增算力主线**（见 `2026-10-26-deepinfer-to-h3c-takeaways.md`）。
- student_grid：pdmd2/4·dmad4 冻结 σ 表进 `tests/fixtures/`；合同 `check_sigmas` 对照该表。  
- 工作区脏：链式源码、`h3_lora.c` 等 + 多份 10-09 研究文档未入库；238 盘接近满；**清盘 / nsys / push / 链式提交均未做**。

### 2.1 已落地（开发，Asia/Singapore）

| 提交 | 时间（UTC+8） | 内容 |
| --- | --- | --- |
| `ef5f732` | 2026-10-26 21:36 | DeepInfer 取舍 + follow-up 开发计划 |
| `675e878` | 2026-10-26 21:40 | P0 视觉门 runbook + 合同自检设计 |
| `5949d37` | 2026-10-26 21:40 | `tools/p0_protocol_check_log.py`（离线协议检） |
| `7fe119c` | 2026-10-26 21:44 | `h3_fewstep_contract`：`--fewstep-contract` / `H3_FEWSTEP_CONTRACT`，错合同 FAIL；host 单测 |
| `e037d42` | 2026-10-26 21:51 | `h3_sampler`：`--sampler lcm_renoise`（DMAD）；dmad4 自动选 lcm；host 单测 |
| `8359dae` | 2026-10-26 22:27 | `--fewstep-preset` / `H3_FEWSTEP_PRESET`：一键绑 steps/cfg/shift/sampler/contract；host 单测 |
| `9356fa7` / `b7c08f0` | 2026-10-26 | fewstep audio-env 冲突提前 FAIL；P0 `--expect-seed/skip-audio/live-te`；`te_f32_checksum.py`；`timing_append_row.py`；[`2026-10-26-te-checksum-and-p0-gates.md`](2026-10-26-te-checksum-and-p0-gates.md) |

**仍未做：** 238 上 P0 视觉门跑数、`git push` origin、链式 WIP 提交、磁盘清理 / nsys。

---

## 3. 优先级路线图（P0 → P3）

| 优先级 | 目标 | 通过标准 | 依赖 | 明确不做 |
| --- | --- | --- | --- | --- |
| **P0** | 默认 INT8 对照纪律固化 + 稳定金线 | graph 关；同 seed/prompt/token-ids；TE 路径标明（live vs `H3_TEXT_H_LOAD`）；1-step velocity md5 可复现；报告写明 `sdpa_graph` | `321538a` 已在；生产 TE `c_te_raw` 偶发不确定记为已知债，速度场诊断优先 refine 路径 | 打开 graph 当默认；跨 TE 路径混 waffle |
| **P0** | NVFP4 实验臂 **P2.2 视觉门**（对照**现在这份**稳定 INT8） | 同协议：graph 关、chain 关、同 seed/prompt、`--skip-audio`；解码后视频目检 + 同配置 waffle；**不要求** bit-identical / cosine>0.99 | P0 INT8 金线；旧 SSIM/目检（graph 修复前）作废 | 用旧片子当通过；把 §12 cosine 当投产门；视觉未过就开默认路径 |
| **P1** | 耗时账本补齐（仅记录，不改产品结论） | 同协议、每臂 ≥2 次、标冷启动；graph 关后数字只追加到 timing 文档 | 视觉门前后都可记；墙钟不得替代视觉门 | 把 graph-on「12%」或错合同计时当宣传加速比 |
| **P1** | PDMD-**4NFE** 原件合同验收（T3 最短路径） | 未截断 `pdmd_4NFE_lora`；4 步 **Euler**；video/audio shift **12/3**；cfg 1；INT8 底模；480p 同 seed；无网格噪、有限 latent、成片可审；waffle 仅同配置 | LoRA 通路已通；勿叠 NVFP4 做首验 | 用 2NFE 冲观感当 T3 通过；turbo4 sigma；Kijai 降秩当金线 |
| **P2** | DMAD 采样器（`lcm_renoise`）+ `full_critic` 诚实验收 | **采样器已提交**；GPU 验收：shift 12/2、cfg 1、原件非降秩；Euler 跑 DMAD **不算**通过 | P1 PDMD-4 合同纪律可复用 | 未视觉验收前标 T3 通过；turbo4 错合同计加速 |
| **P2** | （可选）异步重叠审计 | **笔记已交** [`2026-10-26-te-lifecycle-audit.md`](2026-10-26-te-lifecycle-audit.md)；真机峰值仍待授权跑 | 不引入 DeepInfer 调度器 | 当多并发 serving 重做 |
| **P3** | 链式 FC1→FC2 提交决策 | 仅当 **NVFP4 视觉门有结论** 且 **用户确认**；端到端 latent rel 0.33 有解释或接受为臂内差 | P0 视觉门 | 视觉前强行 commit；把链式当 INT8/NVFP4 切换或 SKU 主线 |
| **P3** | 768p / layerwise streaming、PDL、Hadamard（跟 NVFP4） | 沿用 transfer【部分迁移】；768p 全序列 OOM 优先显存账本 | 视觉门后；用户授权剖析工具 | DeepInfer / SM100 / MoE / FlashMLA 进默认路径 |

---

## 4. 并行隔离臂

下列可与 P0 视觉门**并行**，但不得抢默认路径或污染 INT8 waffle：

| 臂 | 内容 | 停手条件 |
| --- | --- | --- |
| **T3 · PDMD-4NFE** | 原件转换（QKV 合成 / refiner 标记）→ 480p 合同跑 → 目检 | 错合同或 OOM 未按 r2 类预热修复前，不标通过 |
| **T3 · DMAD** | `lcm_renoise` 已实现（`e037d42`）；再 `full_critic` GPU 验收 | 未视觉验收则整臂暂停 |
| **工程 · 异步审计** | 生命周期与显存峰值笔记（DeepInfer 取舍 B） | 只出审计笔记，不改热路径分支 |
| **合同自检** | 错 preset / 错 shift / 错 sampler 加载 FAIL（`7fe119c`） | **代码已落地**；GPU 少步验收时务必打开合同旋钮 |

量化臂（NVFP4）与少步臂（LoRA）**禁止**首轮交叉验收。

---

## 5. 明确不做 / 需用户授权

### 不做（否定清单，延续既有裁决）

- 本仓重训 PDMD/DMAD；拿 2NFE 冲「清晰度」当 T3 通过。  
- W4A8 DiT 作主题默认；有损 magcache / reuse、Sage MMA、多卡进默认。  
- DeepInfer 算子与 serving 核：FlashMLA、DeepEP、投机解码进默认、DeepGEMM JIT 当主加速、SM100 专属、continuous batching（当前 CLI）。  
- 热路径塞运行时多 shape / 计划 JIT；默认打开 SDPA CUDA graph。  
- 链式文档 §12 当产品完成标准。

### 需用户授权后再做

| 项 | 原因 |
| --- | --- |
| 磁盘清理（238 `/mnt/nvme1n1` ~99%、根分区紧） | 未授权不删资产 |
| nsys / EVT 剖析 | 盘满时不写 trace；需授权 |
| `git push` origin | 已在 `44e51c3` 齐平；本轮新提交需再授权 push |
| 提交链式源码与脏工作区（`h3_dit_nvfp4_chain.cu` 等） | 视觉门 + 明确确认 |
| 把 NVFP4 升为默认或并入 golden | 视觉门通过后的产品决策 |

---

## 6. 建议近两周执行顺序

有序 checklist（完成一项勾一项；并行臂标「∥」）：

1. [x] **P0 纪律**：对照模板与命令块见 [`2026-10-26-p0-visual-gate-runbook.md`](2026-10-26-p0-visual-gate-runbook.md)；离线日志检 `tools/p0_protocol_check_log.py`。（2026-10-26）  
2. [ ] **P0 视觉门**：graph 关、chain 关，NVFP4 vs 稳定 INT8，P2.2 协议重做目检 + 同配置 waffle；结论写入进展文档（通过 / 不可用 / 需改核），**作废** graph 修复前旧片子。  
3. [ ] **P1 耗时**：仅追加 timing 文档缺项（同协议重复、标明冷启动）；不挡视觉结论。  
4. [ ] **∥ T3 PDMD-4NFE**：原件 → h3repo → 864×480、4-step Euler、shift 12/3、cfg 1、INT8；记录 sigma 与 `refiner_pairs`；目检。  
5. [x] **∥ 合同自检**：设计 [`2026-10-26-contract-selfcheck-design.md`](2026-10-26-contract-selfcheck-design.md) + 代码 `7fe119c`（`--fewstep-contract` / `H3_FEWSTEP_CONTRACT`）；host 单测过。GPU 少步跑时务必启用。  
5b. [x] **∥ 少步一键 preset**：`--fewstep-preset` / `H3_FEWSTEP_PRESET`（`8359dae`；[`2026-10-26-fewstep-preset.md`](2026-10-26-fewstep-preset.md)）；绑 steps/cfg/shift/sampler/contract；勿叠 turbo4。离线日志识别见 `tools/p0_protocol_check_log.py --expect-fewstep`。  
6. [x] **P2 DMAD 采样器**：`h3_sampler` / `--sampler lcm_renoise`（`e037d42`，2026-10-26）；[ ] `full_critic` GPU 验收（仅在你点头后；238 未跑）。  
7. [x] **P2 可选**：TE/VAE/DiT 生命周期与预热显存审计一页笔记 ([`2026-10-26-te-lifecycle-audit.md`](2026-10-26-te-lifecycle-audit.md))。  
8. [ ] **P3 门控**：视觉门结论出来后，单独请示：链式是否提交、是否 push 本轮新提交、是否清盘/nsys。  
9. [ ] **生产债（不插队）**：`c_te_raw` 偶发不确定单独开修；修完前速度场对照继续优先 refine/`H3_TEXT_H_LOAD`。离线辅助已落地：`tools/te_f32_checksum.py` + P0 `--expect-live-te`/`--expect-te-load`（2026-10-26）。
9b. [x] **∥ P0 日志硬门扩展 + timing 行模板**：seed/skip-audio/live-te/chain；`timing_append_row.py`（不自动改账本）。
9c. [x] **∥ fewstep-preset × 错误 audio shift 提前 FAIL**：`h3_fewstep_preset_audio_env_compatible`。
9d. [x] **∥ student_grid golden fixture**：`tests/fixtures/fewstep_student_grid_golden.h` + `tools/dump_fewstep_student_grid`；`check_sigmas` 对照冻结表 (pdmd2/4、dmad4；pdmd2 audio 3|6)。
9e. [x] **∥ `make fewstep_host_tests`**：打包 contract/sampler/dump/waffle host 测 + `waffle_same_config_gate.py --self-test`。
9f. [x] **∥ 同配置 waffle 接缝门**：[`2026-10-26-waffle-same-config-gate.md`](2026-10-26-waffle-same-config-gate.md) + 离线脚本（禁跨配置比）。
9g. [x] **∥ Metal/CUDA fewstep 边界 + TODO 表**：[`2026-10-26-fewstep-metal-boundary.md`](2026-10-26-fewstep-metal-boundary.md)。

---

## 7. 相关文档

| 路径 | 用途 |
| --- | --- |
| [`2026-10-09-dev-plan-and-progress.md`](2026-10-09-dev-plan-and-progress.md) | 总目标与 10-09 进展（本文上游） |
| [`2026-10-09-generation-timing.md`](2026-10-09-generation-timing.md) | 唯一耗时账本 |
| [`2026-10-09-pdmd-dmad-loras.md`](2026-10-09-pdmd-dmad-loras.md) | T3 合同、2NFE 复验、T3 未验证 |
| [`2026-10-26-deepinfer-to-h3c-takeaways.md`](2026-10-26-deepinfer-to-h3c-takeaways.md) | DeepInfer 取舍（纪律 / 否定清单） |
| [`2026-10-26-p0-visual-gate-runbook.md`](2026-10-26-p0-visual-gate-runbook.md) | P0 INT8/NVFP4 对照命令与纪律 |
| [`2026-10-26-contract-selfcheck-design.md`](2026-10-26-contract-selfcheck-design.md) | 少步合同 FAIL 矩阵；Phase-1 已落地 |
| [`2026-10-07-llm-to-dit-transfer.md`](2026-10-07-llm-to-dit-transfer.md) | LLM→DiT 迁移裁决表 |
| [`2026-10-07-nvfp4-w4a4-path.md`](2026-10-07-nvfp4-w4a4-path.md) | NVFP4 P2.2 视觉协议 |
| [`2026-10-26-dmad-lcm-renoise-sampler.md`](2026-10-26-dmad-lcm-renoise-sampler.md) | DMAD lcm+simple 公式与 API |
| [`2026-10-26-fewstep-preset.md`](2026-10-26-fewstep-preset.md) | `--fewstep-preset` 一键绑 PDMD/DMAD 合同 |
| [`2026-10-08-nvfp4-chained-direct-output.md`](2026-10-08-nvfp4-chained-direct-output.md) | 链式；§12 降为调试尺 |
| [`../superpowers/plans/2026-10-05-h3-optimization-roadmap.md`](../superpowers/plans/2026-10-05-h3-optimization-roadmap.md) | T1–T4 SKU |
| [`../nvidia_optimization_plan.md`](../nvidia_optimization_plan.md) | INT8 主路径 / 主题禁项 |

---

## 8. 修订记录

| 日期 | 说明 |
| --- | --- |
| 2026-10-26 | 初稿：收束 10-09 主线 + PDMD/DMAD + DeepInfer 取舍为 P0–P3 与近两周 checklist。 |
| 2026-10-26 | P2：CUDA `lcm_renoise` 采样器落地；GPU `full_critic` 验收仍待授权。 |
| 2026-10-26 | 勾 §6.1；链到 P0 runbook + 合同自检设计 + `tools/p0_protocol_check_log.py`。 |
| 2026-10-26 | 勾 §6.5（`7fe119c`）；§6.6 采样器 hash `e037d42`；新增 §2.1 已落地表；基线改为 ahead 20；明确 238 视觉门 / push / 链式 / 清盘仍未做。 |
| 2026-10-26 | 勾 §6.5b：`--fewstep-preset` 一键合同；GPU 验收仍未做。 |
| 2026-10-26 | §2.1 记入 `8359dae`；离线协议检识别 fewstep_preset/bound/sampler；Metal CLI 明确不接 fewstep。 |

| 2026-10-26 | 离线：TE checksum + P0 硬门扩展 + fewstep audio-env 冲突文案；未开 238 / 未碰链式 WIP。 |
| 2026-10-26 | 离线：`make fewstep_host_tests` + waffle 同配置门 + Metal 边界 TODO；commit 后 push。 |
