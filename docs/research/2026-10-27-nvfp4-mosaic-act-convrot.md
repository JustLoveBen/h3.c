# NVFP4 mosaic root cause — missing act ConvRot (2026-10-27)

## Symptom
C001 P0 visual (864×480, 8-step, skip-audio):
- INT8: OR free VL **PASS** (clear frames)
- NVFP4 (`H3_DIT_NVFP4=1`, chain unset): OR free VL **FAIL** (full-frame mosaic)
- Same-config waffle on NVFP4 frames is meaningless (both sides INVALID_LOWENERGY);
  latent checksums were **deterministic** (`0xe00e004dcdcbc4e1`) → not a race.

Historical Oct-7 P2.2 “healthy” 768p NVFP4 frames under `out/` / review dumps
are also mosaic under the same VL criteria — that doc overstated quality.

## Root cause (verified)
Sidecar FP4 weights from `tools/quant_dit_nvfp4.py` are **ConvRot-baked**
(same basis as INT8 `W_rot`). Algebra:

```
Y_correct = (X @ H) @ W_rot^T   ==   X @ W^T
Y_broken  =  X @ W_rot^T        # pre-fix NVFP4 dispatch
```

`nvfp4_gemm_dispatch` passed unrotated `d_x` into `h3_cuda_fp4_gemm_*`,
which quantizes acts with **no** ConvRot. INT8 path always ConvRots via
`h3_cuda_w4a8_gemm_f32_dd(..., convrot_gs, ...)`.

Host unit `tests/test_h3_nvfp4_act_convrot.c` (no CUDA):
- fixed max_abs ~1e-5 vs ref
- broken max_abs ~20 (mosaic regime)

## Fix
In `h3_dit_w4a8.c`:
- `nvfp4_act_maybe_convrot` → `h3_cuda_convrot_act_f32_dd` into `d_nvfp4_rot`
  when `ctx->convrot_gs > 0`
- All per-linear NVFP4 GEMMs use the rotated act
- Kill-switch: `H3_DIT_NVFP4_SKIP_ACT_CONVROT=1` restores broken path for A/B
- Chain (`H3_DIT_NVFP4_CHAIN`) auto-skipped when act ConvRot is required:
  fused `swiglu+requant` cannot ConvRot FC2 acts yet (still WIP; do not enable)

## Non-goals / still BLOCK until re-visual
- Do **not** commit NVFP4 **chain WIP** or default-on chain
- Do **not** flip catalog to landed until 238 P0 + OR re-review PASS
