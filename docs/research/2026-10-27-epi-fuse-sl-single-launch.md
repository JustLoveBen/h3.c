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

1. **Wall-clock:** custom `O(gs²)` Hadamard per group is not cublas-strided-batched; expect **Δ≈0 or regression** vs ink/ctrl until tiled/TC Hadamard. Report honest dit ms on 238.
2. **act_f16:** still f32 algebra (match ink); `H3_DIT_ACT_F16=0` for A/B.
3. **Device BIT_ID vs ink:** production A/B (SL vs ctrl, and SL vs ink) — see table below / append.
4. Do **not** default-on until waffle≈PASS and WC ≥ ink (or documented trade for launch-count / graph).

## Policy

- Production default: **OFF**.
- NVFP4 stays terminal visual BLOCK (`ebd77a5` energy RCA) — do not green.
- EPI_FUSE_INK remains valid partial (BIT_ID PASS, ~0% WC).

## 238 A/B (fill after rebuild)

Config: turbo4 INT8, 864×480, seed `0x48000001`, `--repeat 2`, `H3_DIT_ACT_F16=0`, chain unset.

| arm | SL | ink | hot dit ms | hot latent | waffle | notes |
|---|---|---|---|---|---|---|
| ctrl | 0 | 0 | TBD | TBD | — | |
| sl | 1 | 0 | TBD | TBD | TBD | |

