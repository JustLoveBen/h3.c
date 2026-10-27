# cuDNN / Sage3 / sol_cute / Kitchen：是否对比过？

> **日期：** 2026-10-28。**性质：** 只读汇总（答用户「这几个方案是否对比过」）。  
> **机器语境：** 产品 4090 = Ada sm_89 · cuDNN SDPA；5090/238 = Blackwell sm_120。  
> **结论先行：** **对比过（历史矩阵 + 5090 sol 护栏 A/B + 理论审计），但不是当前 4090 产品合同（DMAD-4/Euler-8、libs=0）下的同日头对头复测。** Kitchen **不是** DiT 注意力后端。

---

## 0. 四名一词对齐

| 名字 | 在本仓是什么 | Tier |
| --- | --- | --- |
| **cuDNN SDPA** | DiT dense 注意力；`auto`→cudnn；`quality_safe=1` | **Tier-0 质量默认** |
| **Sage3** | SageAttention3 **Blackwell FP4** 插件（`libh3_sage3.so` + `H3_SAGE3_ROOT`）；Ada 上叙事应走 Sage2/2++，真 MMA 未达 | Tier-1 opt-in |
| **sol_cute** | 官方 Sol-Attn CuTe（4090=`cute_sm89`；5090=`cute_sm120`）；块稀疏+零阶 Taylor 补偿，**近似非 bit-exact** | Tier-1 opt-in |
| **Kitchen / vae_kitchen** | **VAE tile 批调度 / batched cuDNN 入口**（`vae_kitchen_sdpa`），**不是** DiT SDPA 后端；Comfy kitchen 的 `int8_attention` **未**当本仓 DiT 默认 | N/A（VAE） |

证据：`h3_sdpa.h` Tier 注释；`nvidia_build.md` §SDPA；C040 日志 `sage3_lib=0 sol_cute_lib=0`；opt-menu「Sage3/SOL/Kitchen 不在 4090 产品路径」。

---

## 1. 理论要点（为何默认 cudnn）

1. **cuDNN dense**：精确 softmax 注意力；`auto` 永不挑有损后端；KPI `sdpa_strict` 要求 hits==want。
2. **Sage3 宣称**：注意力段低比特 MMA（FP4 microscaling + 两级量化）；相对 FA2/dense 可数倍 attn FLOPs；论文/社区自认「不保证每模型无损」，首尾步更敏感。本仓接线偏 **sm_120 + FP4**；4090（无原生 FP4）不该把 Sage3 当默认。
3. **sol_cute 宣称**：长序列（16k–128k）块稀疏省 FLOPs；官方 5090 阶梯在 **50-step BF16** 上 Sol-Attn ~1.48×（相对核优化基线）；少步蒸馏上需 **dense 护栏**（前 N 步/前 N 层），全稀疏会改 ODE 轨迹。
4. **Kitchen**：社区 Comfy 侧融合/批 VAE 与部分 attn 实验名；本仓同名符号多指 **VAE 批 cuDNN**，勿与 DiT SDPA 混谈。

机理长文：`2026-10-05-acceleration-mechanisms.md`、`2026-10-05-sol-h3-official-reanalysis.md`、`2026-10-07-video-dit-attention-frontier.md`、`2026-10-07-sol-attn-negative-audit.md`。

---

## 2. 实测数字（已有）

### 2.1 4090 · 8-step cafe 480p（2026-08；非 DMAD）

| 臂 | warm DiT | warm e2e | 质量 |
| --- | ---: | ---: | --- |
| **cudnn** | ~47.5–48.3s | ~75–79s | 基线 / 默认 |
| **sol_cute**（全稀疏） | **40.9s（−15%）** | ~79s（≈0） | **MAE 35.2 · 视觉门 FAIL** |

来源：`out/verify_accel_matrix/REPORT.md` · `out/verify_sol_vis_gate/REPORT.md`。  
冒烟（1s/2step）：cudnn 12.14s PASS；sol_cute 18.06s PASS（hits 满，非质量门）。

### 2.2 5090 · 8-step cafe 480p warm n=3（2026-08-26）

| 臂 | DiT | e2e | 备注 |
| --- | ---: | ---: | --- |
| **cudnn** | **31.37±0.05s** | 66.06±5.80 | 默认 |
| **sage3** | **26.02±0.72s（−17%）** | 60.15±2.33 | hits=400/400；旧 MAE~26–27 → **不过质量默认** |

来源：`out/verify_accel_matrix/REPORT.md` §F。Tea/Mag 等同报告另表（步间缓存，非 SDPA）。

### 2.3 5090 · turbo4 + sol 护栏（2026-10-05/07）

| 臂 | hot total | waffle | 裁决 |
| --- | ---: | ---: | --- |
| dense cudnn | 19.37s | 0.849 | 对照 |
| sol 全稀疏 `FIRST_DENSE_*=0` | — | **1.321（+0.472）** | **退化（V0）** |
| sol V1-a 护栏 steps=2/layers=2 | **17.48s（−1.89s）** | **0.831** | **速度+质量双正（R1）** |
| 768P 全 sol | — | — | **32GB OOM**（硬负面） |
| 768P 混合臂 vs cudnn | 同量级 | — | ±3s 噪声内不可分 |

来源：`2026-10-05-v0-v1-sol-ab-results.md` · `2026-10-07-sol-attn-negative-audit.md`。  
官方 1.48× **禁止**换算到本仓 4-step 金线（合同：50-step BF16 vs turbo4 INT8）。

### 2.4 当前 4090 产品路径（2026-10-28）

- C040/DMAD KPI：**仅 cudnn**；`sage3_lib=0` · `sol_cute_lib=0` · `want=0`。  
- **没有** DMAD-4 / Euler-8 下 cudnn×sage3×sol_cute 同日 A/B。  
- opt-menu 明确：**先别在 4090 武装 sol/Sage 当产品加速**。

---

## 3. 架构约束速查

| 后端 | sm_89 (4090) | sm_120 (5090) |
| --- | --- | --- |
| cuDNN SDPA | 产品默认 | 产品默认 |
| sol_cute | `cute_sm89` 可真 hit（历史） | `cute_sm120`；768P 全 sol OOM |
| Sage3 FP4 | **非目标**（无 FP4 TC；`fp4` cap=0） | Tier-1；需 ROOT+LIB |
| Sage2/2++ Ada | 计划/接线有，**真 MMA 未达**（微基准曾远慢于 cudnn） | N/A 主线 |
| Kitchen DiT attn | **未接为 SDPA 后端** | 同左；VAE kitchen ≠ DiT |

---

## 4. 建议：要不要补测？

| 优先级 | 建议 |
| --- | --- |
| **默认** | **不改产品默认**；不把 Kitchen 当 DiT 加速叙事 |
| 若只补一条 | 4090 上 **sol_cute + 官方护栏**（FIRST_DENSE_STEPS/LAYERS=2）vs cudnn，固定 DMAD-4 · 480p · waffle/人眼；预期：DiT 可有 ~10–15% 级，e2e/质量需同日门 |
| Sage3 @4090 | **低优先**（Blackwell FP4）；真要 Ada 量化注意力应重新评估 Sage2++ MMA，不是挂 Sage3 名 |
| Sage3 @5090 | 已有 −17% DiT；质量门未过默认 → 保持 opt-in |
| Kitchen | 只谈 VAE 批/融合；勿并入「四个注意力方案」对比 |

**一句话：** 方案目录与历史 A/B **有**；当前 4090 金线 **只跑 cudnn**；Sage/SOL 有速度数字但质量默认不过门（SOL 全稀疏曾退化，护栏臂在 5090 turbo4 双正）；Kitchen 是 VAE 不是 DiT attn。
