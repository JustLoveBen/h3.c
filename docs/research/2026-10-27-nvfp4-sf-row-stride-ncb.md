# NVFP4 mosaic residual — SF row-block stride = ncb (2026-10-27)

## Verdict
**Root cause (post 32×4 in-tile fix):** SF row-block stride used `ncb_rd =
ceil(ncb/32)*32`, but cuBLASLt on sm_120 indexes
`tile = (r//128)*ncb + (c//4)` (**stride = ncb**). With `ncb_rd` stride, every
M≥128 act/weight row lands on SF tiles cuBLASLt never reads (and the tiles it
does read for those rows stay 0) → Y=0 for those rows → full-frame mosaic.

In-tile **32×4** swizzle remains correct (`diag_sf_layout.cu`).

## Evidence (`bench/nvfp4_phase0/test_fp4_e2e.cu` on 238)
| mode | rel_l2 | row128 |
| --- | --- | --- |
| 32x4 + stride_ncb_rd | 0.87 | Y=0 FAIL |
| **32x4 + stride_ncb** | **0.0017** | **PASS** |
| linear + stride_ncb | 0.14 | FAIL (in-tile) |

## Fix
- `h3_cuda_nvfp4_sf_bytes` → `rb * ncb * 512`
- act quant `tile_base = (row>>7)*ncb`
- `tools/quant_dit_nvfp4.py` + `tools/reswizzle_nvfp4_sf.py` emit/convert same
- Re-layout sidecar on 238 from `*.preswizzle_linear` or current 32x4+ncb_rd

## Non-goals
- Chain still OFF / no chain WIP commit
- EPI_FUSE still DISARMED until its own waffle green
