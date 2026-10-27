# C015/C042 · Track-A dig · act-FP16 workspace staging

> **Asia/Shanghai 2026-10-28.** Track-A near-lossless (waffle gate). Default-safe: **OFF** until E2 PASS.  
> **≠** `H3_DIT_ACT_F16` (ConvRot/half path, default ON).  
> **DeepSeek link:** 消中转 / Absorb-style eliminate F32 materialize (DeepInfer «fused MLA+cast» strategy → DiT block WS), not FlashMLA.

## Claim

Storing `d_qkv` and `d_wide` as FP16 in the DiT block workspace cuts ~½ of those buffers’ VRAM; peak_dit can drop enough for 768p headroom; WC is **not** guaranteed positive without fused cast / half-C GEMM.

## Mechanism

| Piece | Detail |
|---|---|
| Geometry | `HIDDEN=5376` `INNER=7168` `FFN=14336` `wide=2·FFN=28672` |
| Hot carve | `h3_dit_w4a8.c` block pack: `d_qkv = seq·inner·3·f32`, `d_wide = seq·wide·f32` |
| Writers | INT8 GEMM `h3_cuda_gemm_i8_*` → **F32 C** today |
| Readers | `block_phase_attn` / MLP expect `float *` |
| Industry parallel | FA/TE keep attn scratch F16; LLM stacks fuse cast into epi (DeepGEMM/TileKernels spirit) |
| H3 constraint | Must add F16 carve + cast (or F16 C) on **all** writers/readers + Ulysses pack; invasive ~2–3d |

## Metric (formula — live `dit_ws_ledger` when `H3_VRAM_LEDGER=1` / `H3_DIT_WS_LEDGER=1`)

```
qkv_f32  = seq · INNER · 3 · 4
wide_f32 = seq · (2·FFN) · 4
C015_half_save = (qkv_f32 + wide_f32) / 2
```

| Res | seq (packed) | qkv F32 | wide F32 | **half-save** |
|---|---:|---:|---:|---:|
| 480p | 15103 | 1.210 GiB | 1.613 GiB | **1.412 GiB** |
| 768p | ~37414 | 2.997 GiB | 3.996 GiB | **3.497 GiB** |

**Peak_dit:** C040 @4090 ≈22.4 GiB (480p) / ≈22.7 GiB (768p). Half-save is **upper bound** on activation delta; weights still dominate — E2 asks ≥0.5 GiB measured peak drop.

**WC upper bound (naive BW):** overstates. INT8 GEMM is compute-bound; extra F32→F16 cast without epi fuse can **regress** dit. Expected Track-A: VRAM win primary; WC ~0±few % until half-C or fused epi.

## E2 gate (unchanged)

768p Euler-8; stage on vs F32 baseline; **waffle PASS** (non BIT_ID OK); no mosaic; peak_dit ≤ −0.5 GiB. FAIL → keep OFF.

## Landed construction (default OFF)

| Item | Status |
|---|---|
| `dit_ws_ledger` component print | landed; shows stage/wide_f16/save |
| `H3_DIT_ACT_FP16_STAGE` | env; arms pack/phases when fused_epi + U=1 + !nvfp4 |
| F16 `d_qkv` carve | landed; GEMM `epi.y_f16` dequant + F16-input rope |
| F16 `d_wide` carve | landed when **!lora** (LoRA FC1 needs F32 wide) |
| FC2 F16→F32 cast into wide prefix | landed |
| Ulysses pack sizes | byte helpers shared; **U>1 refuses arm** (A2A half TBD) |
| Default | **OFF** until waffle PASS + E2 peak_dit ≤ −0.5 GiB |

## Construction checklist (v1 done → gate)

1. Dual-size WS bytes + pack — **done** (`dit_qkv_ws_bytes` / `dit_wide_ws_bytes`).  
2. GEMM F16 C via `h3_cuda_dit_gemm_epi.y_f16` — **done** (plain + SwiGLU).  
3. F16-input rope `…_f16qkv` — **done**.  
4. FC2 cast F16→F32 — **done**.  
5. Ulysses sizes — pack math ready; runtime arm U=1 only.  
6. 480p waffle + E2 — **PASS** ([kpi](2026-10-28-c015-fp16-stage-waffle-e2-kpi.md): Δb16 mean 0.015, peak_dit −1.412 GiB). Default still OFF.

## Related closed / non-goals

- Exp-G3 / C018/C019: merge+rope fuse **already present** — do not reopen as C015.  
- C003 EPI_FUSE_SL: keep OFF (+46% dit).  
- FlashMLA / DSA / MLA KV: **N-A** (no cross-step KV).
