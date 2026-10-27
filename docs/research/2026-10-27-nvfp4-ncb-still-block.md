# NVFP4 after SF stride=ncb — still BLOCK (2026-10-27)

## What fixed
- Row-block SF stride `ncb` (not `ncb_rd`) + prior 32×4 in-tile + act ConvRot.
- Sidecar reswizzled; host unit + 500-sample logical (r,c) map match linear backup.
- C001 protocol RC=0; same-config waffle Δ=0 both arms.

## What did not
- Visual: OR + frames → **blank muddy** (not athlete). Latent std 0.31 vs INT8 1.03.
- Failure mode changed: rainbow mosaic → low-energy blank (waffle INVALID_LOWENERGY).

## Residual hypotheses (not proven)
1. DiT-shape GEMM/SF interaction beyond K=512 e2e (standalone K=5376 → cublasLt INVALID_VALUE in harness; prod still runs).
2. Act ConvRot ↔ FP4 quant interaction under real seq lengths.
3. scale2 / epilogue magnitude (less likely — SF values map-correct).

## Policy
- NVFP4 visual arm **BLOCK**; no chain WIP; EPI_FUSE stays DISARMED.
- Next honest pass: production-path single-layer A/B (INT8 vs NVFP4) with `H3_NVFP4_DEBUG_NAN` + DiT-K e2e that matches prod layout helpers.

## Extra evidence (this pass)
- Latent energy: NVFP4 std≈0.31 vs INT8 ≈1.03 (~3× attenuation) — not NaN collapse.
- Frame pixel std≈7 (muddy blank) vs prior mosaic std≈60.
- Weight SF reswizzle logical map: 0 mismatches / 500 samples; SF nonzero_frac=1.0.
- Standalone e2e still hardcoded-proven at M=N=256 K=512 `32x4+stride_ncb` (rel_l2≈0.0017).
- Extended e2e harness now accepts `M N K` argv for DiT-shape A/B on 238.
