# Track-B S2 sellable recipe (opt-in) — 4090

> **Asia/Shanghai 2026-10-28.** Product-facing packaging.  
> **Does not flip** product SDPA default (**cudnn** Tier-0 stays).  
> Evidence: Exp-G1 PASS [`exp-g1`](2026-10-28-exp-g1-sol-guard-vs-cudnn-dmad4-4090.md) @ synced `eb805b4` / journal; C041 window24; C015 default ON @`b99d56b`.

## One-liner presets (user-facing)

**Track-B S2 speed (opt-in sol):**
```bash
source tools/h3_track_b_s2_env.sh
# then on generate:
H3_DIT_INT8_WINDOW=24 H3_SOL_FIRST_DENSE_LAYERS=2 H3_SOL_FIRST_DENSE_STEPS=0 \
  --sdpa-backend sol_cute --fewstep-preset dmad4 ...
# + H3_SOL_CUTE_LIB / H3_SOL_ATTN_ROOT / H3_TRITON_OVERLAY (≥3.3)
```

**VRAM stack only (cudnn default; near-lossless Track-A-safe):**
```bash
H3_DIT_INT8_WINDOW=24   # C041 opt-in; C015 act-FP16 already default ON
# do NOT set sol / do NOT change --sdpa-backend
```

**Opt-out C015 if needed:**
```bash
H3_DIT_ACT_FP16_STAGE=0
```

## Recipe bill of materials

| knob | value | why | default? |
|---|---|---|---|
| SDPA | `sol_cute` + `cute_sm89` | Track-B WC (−15% warm dit @G1) | **NO** — product stays **cudnn** |
| `H3_SOL_FIRST_DENSE_LAYERS` | **2** | G1 arm3 PASS; full-sparse FAIL waffle +3.26 | opt-in |
| `H3_SOL_FIRST_DENSE_STEPS` | **0** | LAYERS-only; STEPS=1 weaker | opt-in |
| `H3_DIT_INT8_WINDOW` | **24** | sol pool OOM at product window 41 @seq≈4978 | opt-in (C041) |
| `H3_DIT_ACT_FP16_STAGE` | **ON** (unset) | C015 decision dig YES @`b99d56b`; −1.41 GiB Euler | **YES** product default |
| `H3_VAE_PREWARM_DEV` | `0` (G1 fairness) | optional; keeps SDPA comparison clean | leave product as-is unless probing |
| `H3_SDPA_STRICT` | `0` on sol arms | dense_skip ≠ hits==want | sol arms only |
| Fewstep | **DMAD-4** + full_critic LoRA | S2 speed axis | contract |
| Forbidden | full-sparse `LAYERS=0 STEPS=0`; Sage3@4090; FaceRefine; NVFP4 | G1/G catalog | — |

## Proven KPI (G1, DMAD-4 · 480p · p10 · window=24)

| arm | warm dit vs cudnn | waffle vs cudnn |
|---|---|---|
| sol LAYERS=2 | **0.848×** (−15%) | Δ **−0.017** PASS |
| sol full-sparse | 0.841× | Δ **+3.26** FAIL — never ship |

Warm e2e total ≈ −11% vs cudnn; cold WALL not a speed claim (Sol/Triton first compile).

## Helper

```bash
source tools/h3_track_b_s2_env.sh
```

Exports window + LAYERS=2 + STRICT=0 + VAE_PREWARM hint. **Does not** set `H3_CUDA_SDPA_BACKEND` — you must pass `--sdpa-backend sol_cute` (or export backend only in that shell) so the product default path stays untouched in code.

## Stack smoke (optional)

DMAD-4 + window=24 + C015 ON (env=1 or unset on ≥`b99d56b` binary) under **cudnn** confirms VRAM stack without arming sol. Full sol smoke needs lib + attn_root + Triton overlay on the box.

**2026-10-28 smoke @4090** (`tools/run_track_b_s2_vram_stack_smoke_4090.sh`, binary `84e75b4` + `H3_DIT_ACT_FP16_STAGE=1`):

| field | value |
|---|---|
| RUN_RC | **0** |
| window | `dit_int8_window_cap: 41→24` |
| sdpa | cudnn strict PASS (hits==want) |
| warm dit_ms | 16778.7 |
| peak_dit (warm) | 19100.9 MiB (~18.7 GiB) · free ~4.9 GiB |
| WALL_SEC | 81.63 |

Stack OK — no OOM; cudnn default path unchanged.

## Raise-canvas (quality SKU, separate doc)

Far-face is **pixel budget**, not FaceRefine. See [`raise-canvas-sku-policy`](2026-10-28-raise-canvas-sku-policy.md): DMAD/PDMD ≤1152×640; Euler-8 @1344×768 for max face budget.

## Non-goals

- Flip cudnn → sol as product default  
- Reopen C018 / C011 / C016 / C043  
- Ship full-sparse sol or Sage3 on 4090  
