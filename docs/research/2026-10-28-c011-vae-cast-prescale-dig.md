# C011 dig — VAE cast/prescale fusion remainder (Track-A)

**Date:** 2026-10-28 Asia/Shanghai  
**Trigger:** Exp-G2 C035 **BLOCK** (no local refiner weights) → pivot Track-A.  
**Status:** dig+ (no construction this slice; needs nsys attribution before code).  
**≠** DiT `H3_DIT_ACT_F16` / C015 act FP16 staging.

## Claim → mechanism → metric

| | |
|---|---|
| **Claim** | Historical nsys “cast/prescale ~2.29s” still contains a **quality-preserving** fuseable subset on the VAE (and leftover DiT act-prep) path after product `f16_direct` / fused epi / act_f16 landings. |
| **Mechanism** | HBM dtype roundtrips: F32⇄F16 around GEMM/RMS/ConvRot when fused epi / `rms_norm_f32_to_f16` / `f16_direct` are not armed; SIMT sgemm ~1.15s called out as VAE-family in sm120 frontier §5.3. |
| **Metric (E2)** | nsys (or CUPTI) Δ on cast/prescale + vae_blocks; warm e2e/vae_ms drop; **BIT_ID or same-config waffle PASS** if fuse-only. Target: prove ≥0.3 s warm recoverable before product default. |

## Already landed (do not re-litigate)

| Piece | Evidence |
|---|---|
| VAE `f16_direct` GEMM | `h3_vae_int8_cuda.c` `vae_gemm: f16_direct=…`; product logs show `f16_direct=1` |
| Fused RMS→F16 | `h3_cuda_vae_rms_norm_f32_to_f16_dev` |
| Fused epi / act_f16 on VAE | `vae_gemm: fused_epi=1 act_f16=1` in 4090 INT8 generate logs |
| DiT AdaLN F16 prescale | `dit_gemm_path: act_f16 prescale` (default ON) — **DiT**, not C011 |

## Residual hypotheses (need nsys to rank)

1. **fallback_cast** path still hit on some tiles/shapes (`vae_gemm_path: fallback_cast`) — F32→F16→GEMM→F16→F32 HBM.  
2. **Weight cast_w_to_f16** at load/decode boundary still billed under cast bucket.  
3. **SIMT sgemm ~1.15s** inside historical cast/prescale rollup — may be C012 (TF32/sgemm) not pure cast; split before fusing.  
4. Stale 2.29s number from sm120/238 era — **re-baseline on 4090 sm_89** after C015/product stack.

## Construction checklist (next engineering)

1. **Re-baseline** 480p Euler-8 INT8 warm: `vae_ms` / phase breakdown + optional `nsys profile` (disk OK: 4090 workspace ~256 GiB free).  
2. Attribute kernels in cast/prescale bucket → table (name, ms, bytes).  
3. Fuse candidates ranked by ms; prefer BIT_ID (no algebra change).  
4. A/B waffle + vae_ms KPI; default OFF until PASS.  
5. Do **not** mix C012 TF32 quality risk into C011 fuse-only slice.

## Gate

| Result | Action |
|---|---|
| PASS (≥0.3 s warm, waffle/BIT_ID) | opt-in → consider default |
| NO_WC_WIN / no fuseable remainder | catalog **stale / closed** like C064 |
| Needs TF32 for win | hand to **C012**, keep C011 fuse-only clean |

## Env / hooks (existing)

- `H3_VAE_*` / `h3_vae_int8_gemm_f16_direct` / `h3_vae_int8_act_f16` / `h3_vae_int8_fused_epilogue` — product mostly ON.  
- New env only if a *new* fuse lands (`H3_VAE_CAST_FUSE` sketch; not added this dig).

## Out of scope

- C035 download; FaceRefine; C015 default flip; NVFP4 VAE.
