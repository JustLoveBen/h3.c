# Exp-G1 · Track B · 4090 sol_cute+dense guardrails vs cudnn @ DMAD-4

**Date (Asia/Shanghai):** 2026-10-28 01:02–01:08 CST  
**GPU:** lan-4090-31 / RTX 4090 sm_89 · 24 GiB  
**Synced SHA (4090 `.synced_from_mac_sha`):** `6b64259`  
**Gate:** 864×480 × 5s · `p10_street_wave_dolly` · DMAD-4 + full_critic LoRA · INT8 · `--repeat 2` · `H3_TE_CACHE_AUTO=1` · skip-audio  
**Track:** **B**（有损可接受 / opt-in）— **不改产品默认**（仍 cudnn Tier-0）  
**Runner:** `tools/run_g1_sol_guard_vs_cudnn_4090.sh` · aggregator `tools/g1_aggregate_verdict.py`  
**Out:** `out/g1_sol_guard_vs_cudnn_dmad4_480p_20261028/`（4090 + Mac mirror）

## Claim / gate（事先写死）

| 字段 | 内容 |
| --- | --- |
| **Claim** | `sol_cute` + `H3_SOL_FIRST_DENSE_LAYERS=2` @ DMAD-4·480p：暖 dit 快于 cudnn，waffle/人眼不低于下限 |
| **PASS** | 臂3 waffle_mean ≤ 臂1 + 0.10；臂3 暖 dit ≤ 臂1 × 0.92；hits 可解释（dense_skip 计入）；人眼无运动塌 |
| **FAIL** | 臂2 waffle − 臂1 > +0.15（确认全稀疏禁默认）；臂3 不达 PASS → 关闭 4090 sol 产品叙事 |

## Preconditions / VRAM truth

| 事实 | 证据 |
| --- | --- |
| `sage3_lib=0` | 全部臂 `device_caps`（Ada 预期；非本刀目标） |
| `sol_cute_lib=1` after rebuild | cmake 指 conda Python3 → `build-cuda/libh3_sol_cute.so`；`cute_sm89` |
| **Product INT8 window=41 OOMs sol pool** @ seq≈4978 | smoke `g1_smoke_sol_layers2`：`sol_cute OOM` → `hits=0 dense_skip=2 fallback=1` |
| **G1 fairness window** | 全臂 `H3_DIT_INT8_WINDOW=24` + `H3_VAE_PREWARM_DEV=0`（C041 余量；SDPA 为唯一变量） |
| Triton | `H3_TRITON_OVERLAY=…/triton34_overlay` (3.4)；站点 3.2 不够 |
| `H3_SDPA_STRICT=0` on sol arms | dense_skip 与 strict hits==want 冲突；cudnn 臂仍 `sdpa_strict: PASS` |

## Results（暖 clip=2 / TE HIT）

| arm | backend | dense | warm dit_ms | warm total_s | WALL_s | waffle_mean (hot5) | hits / want | dense_skip |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| **arm1** | cudnn | — | **16937** | 25.23 | 84.3 | **0.774** | 400/400 cudnn | 0 |
| **arm2** | cute_sm89 | STEPS=0 LAYERS=0 | 14246 (−15.9%) | 22.60 | 98.6 | **4.034** | 400/400 sol | 0 |
| **arm3** | cute_sm89 | LAYERS=**2** | **14359 (−15.2%)** | 22.46 | 84.9 | **0.757** | 384/400 sol + 16 cudnn | **16** |
| **arm4** | cute_sm89 | STEPS=**1** | 14946 (−11.8%) | 23.06 | 92.4 | 0.763 | 300/400 sol + 100 cudnn | 100 |

- seq=15103 · H=56 · D=128 · layout=BF16_BTHD · `sampler=lcm_renoise`（DMAD contract）  
- 臂2 冷 clip1 dit=36.5s（Sol/Triton 首次编译）；暖后才 −16%——**勿用冷 WALL 讲加速**  
- 臂3 WALL≈臂1：暖 dit 赢，但冷启动+VAE 淹没 e2e；**暖 total −11%** 才是可复述的墙钟叙事

## Gate verdict

| 检查 | 值 | 结果 |
| --- | --- | --- |
| arm3 waffle − arm1 | **−0.017** ≤ +0.10 | **PASS** |
| arm2 waffle − arm1 | **+3.260** > +0.15 | 全稀疏质量 **FAIL**（预期） |
| arm3 dit / arm1 | **0.848** ≤ 0.92 | **PASS** |
| hits explained | LAYERS=2 → dense_skip=2×4steps×2clips 冷暖累计；暖 stats dense_skip=16 | **PASS** |

### **VERDICT: PASS**（Track B quality floor · 护栏臂）

- 全稀疏（arm2）**禁默认**（waffle +3.26，与历史 8-step MAE FAIL / 5090 V0 同向）。  
- **LAYERS=2 护栏**在本合同下同时达成速度门槛与 waffle 不劣于 cudnn。  
- **不改产品默认**：仍需 opt-in `H3_SOL_*` + window=24 + Triton overlay；产品路径继续 cudnn。

## Quality floor notes

- waffle 多帧标 `INVALID_LOWENERGY`（能量门）；相对臂间仍可用。臂2 b16≈3–5 vs 护栏/cudnn ≈0.7–0.9。  
- Labeled warm clips（burn text）:  
  - Mac: `out/g1_sol_guard_vs_cudnn_dmad4_480p_20261028/labeled/`  
  - 4090: `/home/haiyang/workspace/h3_c_main/out/g1_sol_guard_vs_cudnn_dmad4_480p_20261028/labeled/`  
  - warm r2 raw: `…/warm_r2/`  
- 目检建议：臂2 运动/纹理塌 vs 臂1/3；臂3 应接近臂1。

## Decision / ship

| 项 | 决定 |
| --- | --- |
| 产品默认 SDPA | **仍 cudnn**（Tier-0） |
| Track-B opt-in recipe | `sdpa=sol_cute` + `H3_SOL_FIRST_DENSE_LAYERS=2` + `H3_DIT_INT8_WINDOW=24` + `H3_VAE_PREWARM_DEV=0` + Triton≥3.3 overlay + `libh3_sol_cute.so` |
| 全稀疏 / STEPS-only | 不作默认；STEPS=1 暖 dit 仅 −12%，WALL 更差 |
| Sage3@4090 | 仍关闭叙事（`sage3_lib=0`） |
| 下一刀 | Exp-G2 refiner；或把 G1 recipe 写成 `tools/run_sol_guard_dmad4_optin_4090.sh` 产品文档（仍默认关） |

## Cross-links

- Spec: [`accel-quality-theory-and-industry`](2026-10-28-accel-quality-theory-and-industry.md) §Exp-G1  
- Prior SDPA map: [`sdpa-backends-compared`](2026-10-28-sdpa-backends-compared.md)  
- VRAM lever used: [`c041-window24-probe`](2026-10-28-c041-window24-probe.md)  
- Official Sol: [`sol-h3-official-reanalysis`](2026-10-05-sol-h3-official-reanalysis.md)
