# EPI_FUSE_INK partial vertical slice (2026-10-27)

## Goal
Re-arm FC1→FC2 fuse **without** skipping Hadamard: after SwiGLU, `quant(X@H)` then FC2 `pre_q` against `W_rot`.

## Landed (default OFF)
- Env: `H3_DIT_EPI_FUSE_INK=1` (`h3_dit_epi_fuse_ink_wanted`).
- API: `h3_cuda_epi_fuse_ink_prep_fc2` — sgemm ConvRot + per-row INT8 into rq scratch.
- DiT: when ink+gemm_epi, prep after FC1 SwiGLU; FC2 uses `pre_q`.
- Old `H3_DIT_EPI_FUSE` skip-ConvRot path remains **DISARMED** (`epi_fuse=0`).
- Host algebra: `tests/test_epi_fuse_ink_host.c` (skip≠true, ink≡rot).

## Incomplete (do not default-on)
- Not a single-launch epilogue kernel (still sgemm + quant).
- Matches **f32** act-prep, not `act_f16` prescale — A/B with `H3_DIT_ACT_F16=0`.
- No 238 waffle green yet; keep OFF until same-config waffle PASS.

## Policy
EPI_FUSE (any arm) stays OFF in production until waffle green.

## 238 A/B (2026-10-27 Asia/Shanghai, SHA `1c0e8db`)

Config: turbo4 INT8, 864×480, seed `0x48000001`, `--repeat 2`, `H3_DIT_ACT_F16=0`, chain unset. Out: `out/c003_epi_fuse_ink_20261027/`.

| arm | epi_fuse_ink | hot dit ms | hot latent checksum | RUN_RC |
|---|---|---|---|---|
| ctrl_r1/r2 | 0 | 14271 / 14348 | `0xc36483432681f772` | 0 |
| ink_r1/r2 | 1 | 14321 / 14351 | `0xc36483432681f772` | 0 |

- Same-config waffle: **PASS** (ink + ctrl, meanΔ=0).
- ctrl vs ink hot latent: **BIT_ID equal**.
- Wall-clock: **~0%** (f32 ConvRot+requant prep still separate launches; not a net win yet).
- **Default remains OFF.** Old `H3_DIT_EPI_FUSE` stays DISARMED. True fused act_f16/single-launch still future work.

## Follow-on

True single-launch: `H3_DIT_EPI_FUSE_SL` — see `2026-10-27-epi-fuse-sl-single-launch.md`.
