# 1344 Euler-8「一块一块」RCA (2026-10-27)

## Verdict
**(c) encoder macroblocks** — not DiT mosaic, not VAE 320px tile seams, not INT8 window≈17.

## Hard evidence
- `talking_head_1344x768_s8.mp4`: H.264 ~1.32 Mbps @ 1344×768 (bpp≈0.053).
- Same-res FAIL_NOISE: ~25 Mbps (noise forces CRF bitrate up).
- 1152 PDMD: ~2.1 Mbps (bpp≈0.119) — less blocky.
- Log: `dit_int8_cache_refit … window=17` (PASS waffle mean_b16=0.637); VAE `tiles 5x3` + `deblock16 strength=0.28`.
- Mux used hardcoded `libx264 -preset veryfast -crf 18` (`h3_ffmpeg.c`). CRF18 + smooth talking-head → bitrate starvation → classic 8×8 macroblocks on skin/bokeh.
- Decoded-frame within-8 variance: s8_1344≈33 vs pdmd_1152≈83 (detail crushed by encode).

## Fix
- `h3_ffmpeg_x264_crf()`: `H3_X264_CRF` override; else hires (≥1152×640) CRF **14**; else 18; preview tier 23.
- Eval clips: `tools/burn_eval_label.sh` TOP-LEFT scheme burn-in.
- Verify via `H3_REPLAY_LATENT` on same `latent.raw` (bit-identical RGB path; encode-only delta).
