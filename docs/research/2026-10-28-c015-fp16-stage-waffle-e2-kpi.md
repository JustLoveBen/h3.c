# C015 H3_DIT_ACT_FP16_STAGE — 480p waffle + E2 KPI

**Date:** 2026-10-28 Asia/Shanghai  
**GPU:** lan-4090-31 / RTX 4090 sm_89 (24 GiB)  
**Synced tree:** `/home/haiyang/workspace/h3_c_main`  
**Construction SHA:** `a3d9b74` (build); tip tools polish `e04b656`  
**Binary:** `build-cuda/h3_cuda_generate` sm_89 (C015 symbols present)

## Protocol

- Gate: 864×480 × 5s, complex `p10_street_wave_dolly`, seed `0x76800002`
- Scheme: Euler-8 INT8, **no LoRA** (so wide F16 can arm), `--sdpa-backend cudnn`, `--repeat 2`, `--skip-audio`
- Env common: `H3_VRAM_LEDGER=1 H3_TE_CACHE_AUTO=1 H3_SKIP_PPM=1`
- Arms: `base_euler8_off` (env unset) vs `cand_euler8_stage` (`H3_DIT_ACT_FP16_STAGE=1`)
- Script: `tools/run_c015_fp16_stage_waffle_4090.sh`
- Out: `out/c015_fp16_stage_waffle_480p_20261028/`

## KPI

| Metric | base (OFF) | cand (STAGE=1) | Δ |
|---|---:|---:|---:|
| peak_dit cold used | 22.3427 GiB | 20.9306 GiB | **−1.4121 GiB** |
| peak_dit warm (r2) | 19.4267 GiB | 18.0146 GiB | **−1.4121 GiB** |
| dit_ms hot | 27308.5 | 27307.4 | ~0 |
| WALL_SEC (r2 path) | 96.65 | 89.22 | (TE-cache / noise) |

Theoretical half-save @480p (qkv+wide) ≈ **1.412 GiB** — measured cold/warm drop matches.

### Waffle same-config

```
mean_abs_delta=0.0148 max_abs_delta=0.0230
tol mean<=0.15 max<=0.4
waffle_same_config_gate: PASS
```

INVALID_LOWENERGY frames agree base/cand.

### E2

E2 = waffle PASS **and** peak_dit drop ≥ 0.5 GiB → **PASS** (drop 1.412 GiB).

## Product default

**Remains OFF** (`H3_DIT_ACT_FP16_STAGE=1` opt-in). Gates cleared; flip-to-default (unset→ON, `=0` rollback) is a separate product call. LoRA path keeps wide F32 (qkv-only F16); U>1 / nvfp4 / !fused_epi refuse arm.

## Not reopened

C018 / C003-SL / FlashMLA. GPU serialized after G1 sol_guard finished.
