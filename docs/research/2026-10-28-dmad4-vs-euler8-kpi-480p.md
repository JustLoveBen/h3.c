# DMAD-4 vs Euler-8 INT8 KPI (480p×5s complex p10)

**Date (Asia/Singapore):** 2026-10-28 00:22–00:25 CST  
**GPU:** lan-4090-31 / RTX 4090 sm_89  
**Gate:** 864×480 × 5s · `p10_street_wave_dolly` · INT8 · `--repeat 2` · `H3_TE_CACHE_AUTO=1`  
**Policy (t71u):** **DMAD or Euler primary**; **PDMD skipped** (contrast-only). FaceRefine STOPPED; NVFP4 BLOCK.  
**Base SHA:** `5d8c9cd` binary on 4090.

## Arms

| arm | steps | sampler | LoRA |
|-----|-------|---------|------|
| Euler-8 | 8 | velocity_euler | none |
| DMAD-4 | 4 | lcm_renoise (`--fewstep-preset dmad4`) | `dmad_minimax_h3_4step_full_critic.h3repo` scale=1 |

## Results

| arm | WALL_s (×2) | warm_s | cold_s | warm dit_ms | cold dit_ms | warm te_ms | TE reuse |
|-----|-------------|--------|--------|-------------|-------------|------------|----------|
| euler8_int8_teauto | 89.60 | 35.77 | 53.15 | 27324.5 | 27379.0 | 0.0 | yes (clip2 te=0) |
| dmad4_int8_teauto | 74.24 | 25.02 | 48.12 | 16982.2 | 17298.5 | 0.0 | yes (clip2 te=0) |

- warm dit speedup Euler→DMAD: **1.61×** (27324/16982)
- warm e2e speedup: **1.43×** (35.77/25.02)

## C007

`sol_cute` want=0 hits=0 lib=0 under `--sdpa-backend cudnn` → **N/A** on this product path; no autotune persistence work.

## Artifacts

- Runner: `tools/run_dmad4_vs_euler8_kpi_4090.sh`
- Out: `out/dmad4_vs_euler8_kpi_480p_20261028/`
- Share: `~/grokbot-share/dmad4_vs_euler8_kpi_480p_20261028/` (+ Mac Downloads + box)
- Labeled warm r2: `labeled/{euler8,dmad4}_int8_teauto_labeled.mp4`
