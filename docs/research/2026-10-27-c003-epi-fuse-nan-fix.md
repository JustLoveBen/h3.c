# C003 EPI_FUSE NaN fix (2026-10-27)

## Symptom (238 prior run)
- `H3_DIT_EPI_FUSE=1`: `latent_stats: mean=nan`, mux ~100KB vs ctrl ~2–3MB
- dit hot clip2 ~8.4% faster but quality FAIL — do not default-on

## Root cause
1. **Wrong P1 act source:** fused OUT requant fed quantized `d_x` into FC1 via `pre_q`, skipping AdaLN (`d_b`→`d_a`). Network diverged to NaN across blocks.
2. **`h3_cuda_gemm_i8_dequant_f32_dev_pre_q`:** after redirecting `epi` to a stack `epi_local`, surface-back mutated the local copy — caller never received `d_rq_x`/`d_rq_act` (P2 often silently fell back).
3. **Scratch width:** requant mode-1 buffers sized with GEMM `K` (hidden) but wrote `n/2` (ffn=14336 > 5376).

## Fix
- Disable P1 (OUT→FC1). Keep P2 only: FC1 still consumes post-AdaLN `d_a`; fused requant of swiglu out for FC2 `pre_q`.
- Set `requant_gs = ffn` on mode-1.
- Size requant scratch by `rq_width` (mode1 `n/2`, else `n`).
- Surface `d_rq_*` to the **caller** epi in `dev_pre_q`.
- Default remains OFF (`H3_DIT_EPI_FUSE=1` opt-in) until 238 re-verify.

## Files
- `h3_dit_w4a8.c`, `h3_gpu_cuda.cu`
- `out/review-2026-10-07/e2-epi-fuse/DESIGN.md` §7
