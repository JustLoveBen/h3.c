# EPI_FUSE_SL — single-launch ConvRot+requant (2026-10-27)

Asia/Shanghai. Goal: **true mid-buffer elimination** (not ink's sgemm+quant multi-launch).

## What landed (default OFF)

| Item | Detail |
|---|---|
| Env | `H3_DIT_EPI_FUSE_SL=1` (`h3_dit_epi_fuse_sl_wanted`) |
| Kernel | `h3_cuda_epi_fuse_sl_convrot_quant_kernel` — one launch |
| Algebra | Same as ink: `Y = quant(X@H)` with `W_rot=W@H` → FC2 `pre_q` |
| Mid-buffer | **No** f32 `d_rot` in rq scratch (i8+scales only) |
| Trick | Recompute Hadamard for absmax, then again for quant (2× compute, 0× rot HBM) |
| Ink path | Unchanged when only `H3_DIT_EPI_FUSE_INK=1` (sgemm + quant) |
| P2 | `H3_DIT_EPI_FUSE` skip-ConvRot remains **DISARMED** |
| NVFP4 | Not armed (`!nvfp4_on`); chain untouched |

## Host proof

`tests/test_epi_fuse_sl_host.c`: SL recompute quant ≡ ink rot+quant (`i8_mismatch=0`).

## Incomplete / remaining

1. **Wall-clock:** naive `O(gs²)` Hadamard recompute **regresses** vs cublas ConvRot (~+46% hot dit). Keep OFF until tiled/TC Hadamard (or accept launch-count/graph trade with documented WC loss).
2. **act_f16:** still f32 algebra (match ink); A/B used `H3_DIT_ACT_F16=0`.
3. Device vs ctrl: **BIT_ID** (same hot latent checksum) + same-config waffle **PASS**.
4. Do **not** default-on (WC regression).

## Policy

- Production default: **OFF**.
- NVFP4 stays terminal visual BLOCK (`ebd77a5` energy RCA) — do not green.
- EPI_FUSE_INK remains valid partial (BIT_ID PASS, ~0% WC) — prefer ink over SL for any opt-in fuse today.

## 238 A/B (2026-10-27 Asia/Shanghai, SHA `ed1ce72`)

Config: turbo4 INT8, 864×480, seed `0x48000001`, `--repeat 2`, `H3_DIT_ACT_F16=0`, chain unset. Out: `out/c003_epi_fuse_sl_20261027/`. Host: `h3_epi_fuse_sl_host_test` PASS (`i8_mismatch=0`).

| arm | SL | ink* | hot dit ms | hot latent | waffle | RUN_RC |
|---|---|---|---|---|---|---|
| ctrl_r1/r2 | 0 | 0 | 14275.6 / 14366.9 | `0xc36483432681f772` | PASS (ctrl) | 0 |
| sl_r1/r2 | 1 | 1* | 20898.9 / 20910.4 | `0xc36483432681f772` | PASS (sl) | 0 |

\* SL implies ink algebra / pre_q (`epi_fuse_ink=1` in log when `epi_fuse_sl=1`).

- ctrl vs SL hot latent: **BIT_ID equal**.
- Wall-clock: **~+46% hot dit** (SL slower; expected naive Hadamard). **Default remains OFF.**
- P2 `H3_DIT_EPI_FUSE` stays DISARMED. NVFP4 untouched.

## Tiled / FWHT path (in progress, 2026-10-27 Asia/Shanghai)

Naive O(gs²) recompute caused **~+46% hot dit**. Replacement in `h3_cuda_epi_fuse_sl_convrot_quant_kernel`:

1. **Fast path (default when gs power-of-2):** in-smem **normalized FWHT** — ≡ Sylvester `H@x` (`d_hadamard` is always Sylvester via `h3_cuda_ensure_hadamard`). Complexity O(gs log gs) per group per pass.
2. **Fallback:** smem-resident `x_g` + dense matvec (same sum order as `h3_cuda_convrot_act_kernel`) when gs is not a power of two.
3. Smem layout: `buf[gs] + red[threads]`. Launch smem bumped accordingly.
4. Host proof extended: `tests/test_epi_fuse_sl_host.c` — dense SL≡ink (gs=4) **and** `fwht_sl_vs_dense` at gs=256 (`i8_mismatch=0`).

**Still default OFF.** Next: sync+rebuild 238, INT8 turbo4 A/B (ctrl vs SL) for BIT_ID + WC. Do not default-on until WC ≤ ctrl and waffle green.

## FWHT A/B + dense-tiled default (2026-10-27 Asia/Shanghai)

Out: `out/c003_epi_fuse_sl_fwht_20261027/` SHA binary `c506524e…` (FWHT-as-default briefly). Protocol: INT8, 864×480, seed `0x48000001`, `H3_DIT_ACT_F16=0`, 8 steps.

| arm | SL | dit_ms | latent checksum | std |
|---|---|---|---|---|
| ctrl_r1/r2 | 0 | 25251.7 / 25200.1 | `0x349ec79e4d1cd0e1` | 0.9915 |
| sl_r1/r2 (FWHT) | 1 | 25317.7 / 25104.8 | `0xf8b53ac7ba008c61` | 1.337 |

- **WC:** FWHT ≈ ctrl (~0%, vs naive +46%).  
- **BIT_ID:** **FAIL** (checksum≠ctrl; std drift). Host `fwht_sl_vs_dense` still PASS on one-shot quant — DiT amplifies float≠dense.  
- **Fix:** default SL path = **smem-x dense tiled** (ink algebra). FWHT only if `H3_DIT_EPI_FUSE_SL_FWHT=1`. Still default OFF overall (`H3_DIT_EPI_FUSE_SL`).

