# C001 NVFP4 mosaic — SF 32×4 swizzle root cause (2026-10-27)

## Verdict
**Root cause found and fixed (non-chain arm):** act/weight SF was written with
linear in-tile layout `(r%128)*4+(c%4)`, but cuBLASLt `VEC16_UE4M3` on sm_120
reads **32×4-swizzled** bytes:
`16*((r%128)%32) + 4*((r%128)/32) + (c%4)`.
504/512 slots in a 128×4 tile disagree → scales attach to the wrong rows →
visual mosaic. INT8 path unaffected (no SF).

## Evidence
1. OR free VL: INT8 PASS, NVFP4 FAIL (mosaic) on `out/p0_visual_20261027_*`.
2. Missing act ConvRot was a real bug (landed earlier) but insufficient —
   latent checksum stayed `0x37f14b6cfc58dced` after ConvRot + SF ncb_rd pad.
3. Weight-create SF byte-count pad A/B (padded ≡ unpadded checksum) → **not** residual.
4. `diag_sf_layout.cu` single-marker probe on 238:
   `phys=3→row0, phys=7→row32, phys=19→row1, ...` matches 32×4, not linear.
5. Offline weight dequant with *linear* unswizzle still showed ~0.095 rel_l2 vs
   INT8 floats (quant error only) — dequant used the same wrong formula, so it
   looked "fine" while cuBLASLt scrambled scales.

## Fix
- `h3_gpu_cuda.cu`: `h3_nvfp4_sf_off_in_tile()`; act quant writes use it.
- `tools/quant_dit_nvfp4.py`: emit 32×4 swizzle.
- `tools/reswizzle_nvfp4_sf.py`: permute existing sidecar SF without re-quant.
- `bench/nvfp4_phase0/test_fp4_e2e.cu`: align layout + use K%256==0 shape.
- Chain WIP file has the same formula bug; left untouched for commit policy
  (no chain enable / no chain WIP commit). Re-apply there before any chain trial.

## Non-causes (this batch)
- SF H2D size truncate (ncb vs ncb_rd) — necessary hygiene, not mosaic residual.
- `scale2=1.0` — unchanged.
- Pack nibble hi_first — matches Comfy; weight dequant coherent under same pack.

## Gate
After reswizzle + rebuild: P0 NVFP4 vs INT8, same-config waffle, OR VL.
Keep chain BLOCK until visual+user say otherwise.


## Update: row-block stride
See `2026-10-27-nvfp4-sf-row-stride-ncb.md` — 32x4 in-tile was necessary but
stride must be `ncb` not `ncb_rd`. That residual caused M>=128 Y=0 mosaic.
