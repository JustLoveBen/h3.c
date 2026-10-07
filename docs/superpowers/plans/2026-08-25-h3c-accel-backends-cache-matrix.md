# h3.c 加速后端 × 步间缓存 统一对比矩阵 Implementation Plan

> **后续开发方案见** [2026-08-26-h3c-next-tech-plan.md](./2026-08-26-h3c-next-tech-plan.md)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 Sol-Attn / Sage3 / cuDNN 与 EasyCache / MagCache / TeaCache / FBCache / reuse 等加速方案放进同一套可执行对比协议，完成缺口实现与视觉门验收，且**不破坏质量默认路径**。

**Architecture:** 注意力类走现有 `h3_sdpa` Tier-0/1 registry；步间缓存类统一挂在 `h3_cuda_generate` 采样环（与 MagCache/reuse 同插点），各自 opt-in CLI，默认全关。对比只认 cafe 金线配方 + warm mean + PPM MAE，禁止与官方 Sol-Engine 4.44× 整栈混谈。

**Tech Stack:** C/CUDA `h3_cuda_generate`，`libh3_sol_cute.so` / `libh3_sage3.so` Python embed，cuDNN SDPA，4090 `build-cuda` / 5090 `build-cuda-sm120`。

**Spec / 底稿（冲突以本文 + 下列为准）：**
- [2026-08-25-h3c-4090-accel-plan.md](./2026-08-25-h3c-4090-accel-plan.md) — 4090 三列重判、4.44× 读法
- [2026-08-25-h3c-multi-gpu-arch-speed.md](./2026-08-25-h3c-multi-gpu-arch-speed.md) — SDPA Tier 分层
- [2026-08-21-h3-accel-survey.md](./2026-08-21-h3-accel-survey.md) — 市面加速普查
- 对比骨架：[`out/verify_accel_matrix/REPORT.md`](../../../out/verify_accel_matrix/REPORT.md)
- **外部框架对照（短链）：** [2026-08-25-h3c-vs-sglang-vllm-omni.md](./2026-08-25-h3c-vs-sglang-vllm-omni.md) — h3.c vs SGLang Diffusion vs vLLM-Omni（MiniMax-H3）；禁止与 50-step 多卡秒数横比

## Global Constraints

- 质量默认：**INT8 ConvRot DiT + fp16 VAE tile+deblock，cfg=1，steps=8，864×480**；**reuse / magcache / sage / sol 全部 off**；SDPA=`cudnn`
- 真时长：`--seconds 5` → 120 frames @24fps；**禁止** `--frames` 假时长
- SDPA：`--sdpa-backend cudnn|sage3|sol_cute`；缺 `LIB`/`ROOT` **硬失败**；禁止 nested-loop `sol`、假 sage2 静默替代
- 金线 cafe：574 ids，seed `0x11`，有声；4090 cudnn ~75.4s；5090 cudnn warm ~66s
- 视觉门不过 → **不得改质量默认**（含 MagCache / Tea / Easy / FB / Sol / Sage3）
- 官方 Sol-Engine **4.44×** = 50-step BF16 + TeaCache + compile + Sol **整栈**，勿与「Sol alone」或本仓 8-step INT8 金线横比
- 密码绝不写入 repo/docs；用户未要求则不 commit / 不 push

---

## 0. 方案分类（勿混指标）

| 类 | 方案 | 作用面 | 与官方 Sol-Engine 关系 |
| --- | --- | --- | --- |
| **A. 注意力** | `cudnn` / `sage3` / `sol_cute` | 每步每层 SDPA 核 | Sol 是整栈里的**注意力臂**；官方 4.44× 的主力是 **TeaCache 类 skip**，不是 Sol 单核 |
| **B. 步间缓存** | MagCache / TeaCache / EasyCache / FBCache | 跳过整步 DiT（或首块探测后跳） | 官方 4.44× 的**主力臂**；本仓 8-step 蒸馏上 skip 空间极小且易炸视觉/音频 |
| **C. 其它** | `--reuse N`（固定间隔外推 velocity） | 固定 schedule skip | 非论文 cache；与 magcache 互斥（`reuse>1` 时 magcache 忽略） |

**禁止混谈示例：** 「Sol 给了 4×」✗；正确：「官方 50 步 BF16 768p 上 TeaCache≈3.18× + Sol≈1.22× ≈4.44×」。本仓 cafe 8-step 上 `sol_cute` DiT −15%、e2e≈0、MAE~35。

---

## 1. 现状盘点（代码真相，2026-08-25）

| 方案 | 状态 | CLI / env | 证据 / 禁用原因 |
| --- | --- | --- | --- |
| **cudnn SDPA** | **已实现** · 质量默认 | `--sdpa-backend cudnn`；`H3_CUDA_SDPA_BACKEND`；`auto`→cudnn | `h3_sdpa.h` Tier-0；4090 q1 ~75.4s；5090 warm ~66s |
| **sol_cute（官方 CuTe Sol-Attn）** | **已实现** · Tier-1 | `--sdpa-backend sol_cute`；`H3_SOL_CUTE_LIB` + `H3_SOL_ATTN_ROOT`（缺则硬失败） | `tools/h3_sol_cute_embed.{c,py}`、`h3_sdpa_registry.c`；`out/verify_sol_vis_gate`：DiT −15%、e2e≈0、**MAE 35.2 视觉不过** → 不当默认 |
| **sage3（thu-ml FP4）** | **已实现** · Tier-1（5090） | `--sdpa-backend sage3`；`H3_SAGE3_LIB` + `H3_SAGE3_ROOT` | `tools/h3_sage3_embed.{c,py}`；`out/verify_5090_bench`：DiT −14%、**MAE ~27 视觉不过** → 不当默认 |
| **sol nested-loop** | **已禁用（产品）** | 请求 `sol` → resolve 拒绝 | 孤立 bench 曾比 cuDNN 慢 7.5×；`forbidden_product` |
| **sage2 参考核** | **已禁用（产品）** | 禁止当产品后端 | 假核 / 满 seq 曾远慢于 cuDNN |
| **MagCache（本仓 lite）** | **已实现 · 半成品相对论文** | `--magcache-thresh T`；`H3_MAGCACHE_MAX_CONSEC`；默认 thresh=0 | `h3_cuda_generate.c` 采样环；p4 thresh=0.08：79.2s、evals=5/skips=3、**帧 MAE 52–68** → **明确不当质量默认**；代码保留 opt-in |
| **TeaCache** | **半成品** | `--magcache-calibrated` = TeaCache 风格 poly(rel_l1)；无独立 `--teacache` | 注释写 vLLM H3 TeaCache ref~0.17；**不是**独立 TeaCache 实现；官方 50 步 0.10 不可抄到 8 步无损 |
| **EasyCache** | **未实现** | — | 仅文档提及；Comfy #15326 伤 H3 音频 → 有声路径高风险 |
| **FBCache / First Block Cache** | **未实现** | — | 勿与 `h3_w4a8_cache` 的 `first_block`（权重窗）混淆；官方 Spark/5090 页用 FBC，本仓无 CLI |
| **reuse** | **已实现** · 默认关 | `--reuse N`（1–32）；与 magcache 互斥 | `h3_dit_w4a8_reuse_schedule`；质量默认 `reuse_schedule: disabled` |

**已有 harness（可复用，勿推倒）：**
- `scripts/h3_sdpa_backend_smoke.sh` + `h3_sdpa_backend_smoke_matrix.sh` — 短步冒烟（1s/2step），**无 MagCache**
- `out/verify_sdpa_smoke/`、`out/verify_sol_vis_gate/`、`out/verify_5090_bench/`、`out/verify_480p_quality_full/`
- 本计划新增：`scripts/h3_accel_matrix_smoke.sh`（草稿）、`out/verify_accel_matrix/REPORT.md`

---

## 2. 目标 / 非目标

### 目标

1. 一张表能回答：每个方案在本仓是「已实现 / 半成品 / 未实现 / 已禁用」，以及相对 cafe cudnn 金线的 **e2e / DiT / 视觉 MAE / 是否允许进质量默认**。
2. 注意力类：保持 Tier-1 可切换、可测、可回退；继续抠质量（MAE→不可辨）优先于再砍墙钟。
3. 缓存类：补齐 **独立 TeaCache / 保守 FBC /（可选）EasyCache AV 分账** 的实现缺口，全部默认关；同一 harness 跑 A/B。
4. 统一对比协议（§3）成为后续所有加速 PR 的验收入口。

### 非目标

- 不把任何损质后端改成质量默认。
- 不实现 / 不宣传 nested-loop `sol`、假 `sage2`。
- 不重跑官方 50 步 BF16 768p 去「复现 4.44×」冒充本仓进度。
- 一次 PR 不实现全部缓存；本轮交付以计划 + 矩阵骨架 + harness 草稿为主。
- 不把 SLA / Radial / SVG2 / W4A8 / TinyVAE / `--frames 17` 塞进本矩阵主表（另文）。

---

## 3. 统一对比协议（所有臂必须遵守）

| 项 | 固定值 |
| --- | --- |
| Prompt | cafe 金线 ids（574 tokens），与 `t2va_complex_5s_cafe_q1` / `input/cafe/prompt.ids` 同份 |
| Seed | `0x11` |
| 分辨率 | 864×480 |
| 时长 | `--seconds 5`（ffprobe 5.000s / 120 @24fps） |
| 音频 | 有声 pack（默认） |
| 配方 | INT8 ConvRot DiT，fp16 VAE tile+deblock，cfg=1，steps=8，layers=50 |
| 基线 | `--sdpa-backend cudnn`，reuse/magcache off |
| 计时 | **warm mean**（建议 cold 1 + warm ≥2）；报 `phase_ms`：e2e + TE + DiT + VAE；另报 `dit_evals` / `dit_skips` |
| 视觉 | 相对同机 cudnn warm 的 PPM 抽帧 RGB MAE（至少 f0/24/48/90/119）；口型/道具目视；有声时注意音频漂移 |
| 质量默认？ | 仅当视觉「与金线不可辨」**且**无音频回归时写「候选」；否则一律 **否** |
| 硬件分列 | 4090 主测 `sol_cute` + cache；5090 主测 `sage3` + cache；cudnn 双卡都报 |

**输出目录约定：** `out/verify_accel_matrix/<gpu>_<arm>/` + 总表 `REPORT.md`。链接已有报告，**禁止编造数字**。

---

## 4. 每方案实现缺口

| 方案 | 缺口 | 建议接口（草案） |
| --- | --- | --- |
| cudnn / sage3 / sol_cute | 产品接线基本齐；缺：统一 cafe 全矩阵一键脚本、持续质量改进 | 沿用 `--sdpa-backend`；矩阵脚本填表 |
| MagCache lite | 已接线；缺：更严 thresh 扫、与 Tea/FBC 并列命名澄清 | 保留 `--magcache-thresh`；文档标明「本仓 magnitude skip，非完整论文 MagCache」 |
| TeaCache | 无独立模块；calibrated 曲线绑在 magcache 旗下 | 后续：`--teacache-thresh` 或 `--cache-backend teacache`；默认关；8 步 thresh 从**严于** 0.08 起扫 |
| EasyCache | 零代码 | 新：`--easycache-thresh`；**必须 AV 分账 / 音频 veto**（防 #15326）；默认关 |
| FBCache | 零代码 | 新：`--fbcache-thresh`；首块 residual 探测后决定 skip；默认关；8 步预期悲观 |
| reuse | 已齐 | 矩阵保留一列对照；不当质量默认 |

---

## 5. 分阶段任务清单

并行标注：`‖` = 可与同阶段其它任务并行。

### Phase 0 — 盘点与骨架（本轮）

- [x] **P0.1** 代码/CLI 盘点 → 本文 §1
- [x] **P0.2** 写本计划 + `out/verify_accel_matrix/REPORT.md` 骨架（填入已有数字）
- [x] **P0.3** `scripts/h3_accel_matrix_smoke.sh` 草稿（不改质量默认；需 GPU 主机才真跑）

### Phase 1 — 注意力矩阵填满（优先，已近完成） ‖

**Files:** 复用 `scripts/h3_sdpa_backend_smoke.sh`；扩展矩阵脚本写 cafe 全长臂；不改默认。

- [ ] **P1.1** 4090：cudnn warm 对照（可引用 q1 / sol_vis_gate，勿重复空跑）写入 `REPORT.md`
- [ ] **P1.2** 4090：`sol_cute` cafe 8-step warm（已有 vis_gate）→ 矩阵标记视觉 FAIL
- [ ] **P1.3** 5090：cudnn / sage3 cafe warm（已有 `verify_5090_bench`）→ 矩阵
- [ ] **P1.4**（可选质量）`sol_cute` / `sage3` 质量旋钮实验（dense 前缀步、sink 等）——**仅当**目标是过视觉门；过不了不改默认

**验收：** 矩阵 A 类四格（4090 cudnn/sol、5090 cudnn/sage3）均有 warm 数字 + MAE 链接；默认仍 cudnn。

### Phase 2 — MagCache / calibrated 统一 harness ‖（相对 P3）

**Files:** `scripts/h3_accel_matrix_smoke.sh`；`tools/h3_cuda_generate.c`（仅文档注释澄清，逻辑默认不动）。

- [ ] **P2.1** harness 增加 arm=`magcache`：`--magcache-thresh` 可配（默认试 0.04 / 0.08，**不**写进质量路径）
- [ ] **P2.2** arm=`magcache_cal`：`--magcache-thresh` + `--magcache-calibrated`
- [ ] **P2.3** 同一 cafe ids，报 evals/skips + MAE vs cudnn；填 `REPORT.md`
- [ ] **P2.4** 文档一句：本仓 MagCache ≠ 完整论文实现；calibrated ≈ TeaCache 风格，非独立 TeaCache

**验收：** p4 级结果可复现或引用；任何 MAE≫不可辨 → 矩阵写 FAIL / 默认否。

### Phase 3 — TeaCache 独立接线（实现缺口）

**Files:**
- Modify: `tools/h3_cuda_generate.c`（采样环旁路，仿 magcache 插点）
- Modify: help / `docs/nvidia_build.md` 一小节
- Test: harness arm=`teacache`

- [ ] **P3.1** 设计：`--cache-backend off|mag|tea|fb|easy` **或** 独立 `--teacache-thresh`（二选一，实现前锁定一种；推荐独立 flag 少动 mag 语义）
- [ ] **P3.2** 实现 TeaCache 决策（rel_L1 累积 / poly）；首末步强制 eval；`H3_TEACACHE_MAX_CONSEC`
- [ ] **P3.3** 有声：音频 residual 可 veto skip（对齐现有 mag AV split）
- [ ] **P3.4** cafe A/B：thresh 从严扫；填矩阵；视觉/音频任一门不过 → 默认否

**验收：** CLI 可开关；默认 off；矩阵有至少一档数字（即使 FAIL）。

### Phase 4 — 保守 FBCache（实现缺口）‖ 可与 P3 后半并行测

**Files:** `tools/h3_cuda_generate.c` + DiT 首块 hook（若需 `h3_dit_w4a8_*` 暴露 first-block residual）

- [ ] **P4.1** 定义探测：step 内只跑 first block 得 residual 相似度，再决定 full eval vs reuse velocity
- [ ] **P4.2** `--fbcache-thresh`；默认 off
- [ ] **P4.3** cafe A/B；8 步预期 skip 少或视觉 FAIL —— 诚实填表

**验收：** 可跑通；不过门不进默认。

### Phase 5 — EasyCache（最后，有声高风险）

**Files:** 新逻辑 + 强制 AV 分账

- [ ] **P5.1** 读 Comfy EasyCache 行为与 #15326；设计音频独立判据
- [ ] **P5.2** `--easycache-thresh`；默认 off；无 AV 分账则拒绝合并
- [ ] **P5.3** cafe 有声门；音频回归 → 永久标记「有声禁止默认」

**验收：** 代码存在且默认关；矩阵有结论行。

### Phase 6 — 组合臂（可选，低优先级）

- [ ] **P6.1** `sol_cute`+Tea / `sage3`+Tea 等组合：**仅在单臂矩阵填完后**；官方整栈启发，不承诺 4×
- [ ] **P6.2** 组合不过门 → 文档写「叠乘更危险」，不进默认

---

## 6. 验收标准（总）

| 门 | 标准 |
| --- | --- |
| 功能 | 每臂 CLI 可显式开；缺插件硬失败；默认路径 log 仍 `sdpa=cudnn`、`magcache=0`、`reuse off` |
| 速度 | 报 warm e2e/TE/DiT/VAE；注意力比看 DiT；缓存比看 DiT + skips |
| 视觉 | PPM MAE + 目视；不可辨才允许「质量候选」 |
| 默认 | 候选须单独产品决策；本计划**不自动改默认** |
| 文档 | `REPORT.md` 无编造；4.44× 脚注保留 |

---

## 7. 风险

| 风险 | 缓解 |
| --- | --- |
| 8-step 蒸馏后再 skip → 轨迹漂（LightX2V #579 / MagCache p4） | thresh 从严；max_consec 低；失败即关 |
| EasyCache 伤 H3 音频（#15326） | AV 分账 + 有声门；无分账不合并 |
| Sol/Sage 加速但 MAE 高 | 保持 Tier-1 opt-in；质量默认 cudnn |
| 指标混用 4.44× | 矩阵页强制脚注；禁止「Sol=4×」 |
| TE I/O 抖动吃掉 DiT 收益 | 同时报 DiT；e2e 用 warm mean±std |
| SSH 主机 / 密码进文档 | 主机用 `H3_SSH_4090`/`H3_SSH_5090`（默认见 `docs/nvidia_build.md`）；密码仅 `H3_SSH_PASS_*`；脚本不写死密码 |

---

## 8. 建议下一阶段优先顺序（带理由）

1. **P1 填满注意力矩阵（引用已有报告）** — 成本最低，产品 Tier-1 已接线，表先完整。
2. **P2 MagCache/calibrated harness** — 代码已有，缺统一 cafe 对比入口；可快速复现 FAIL 证据。
3. **P3 独立 TeaCache** — 官方 4.44× 主力臂；本仓需诚实测 8-step 是否还有 skip 空间。
4. **P4 FBCache** — 与 Tea 同类，官方部分页用 FBC；实现成本中等。
5. **P5 EasyCache** — 有声回归风险最高，最后做。
6. **注意力质量改进（sol/sage MAE）** — 与缓存并行的长期线；过门前不改默认。

---

## 9. 执行交接

Plan complete and saved to `docs/superpowers/plans/2026-08-25-h3c-accel-backends-cache-matrix.md`.

**执行选项：**
1. **Subagent-Driven（推荐）** — 每 Phase/Task 新 subagent + 两段 review
2. **Inline Execution** — 本会话 executing-plans，批量 + checkpoint

本轮已完成 Phase 0；未跑 GPU generate（需 4090/5090 与插件 ROOT）。
