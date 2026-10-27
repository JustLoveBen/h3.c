# KEEP_INT8 hybrid（正路径）

**正路径优先级（2026-10-27）：** 现成 w4a4 原生 FP4 + KEEP_INT8 找最大可过覆盖；convert DISARMED。

# NVFP4 KEEP_INT8 hybrid (no-retrain)

**Date:** 2026-10-27 (Asia/Shanghai)  
**Status:** KEEP landed; **out/fc2 gate hybrid bug fixed** (see gate-bug RCA). Postfix energy OK on mlp/qkv/full; multi-scene visual pending.  
**SHA base:** a7d7375 (KEEP+DISARM on origin/main)

## Why

User: NVFP4 weights already exist — do **not** default to QAT/retrain.
C001 full W4A4 still energy-collapses (~13% rel/linear → latent std~0.31).
Honest no-retrain next step: **keep error-sensitive linears on INT8**,
run remaining linears on the existing NVFP4 sidecar.

## Env

| Var | Meaning |
|-----|---------|
| `H3_DIT_NVFP4=1` | enable FP4 GEMM track (unchanged; product default still unset/INT8) |
| `H3_DIT_NVFP4_FILE=` | existing w4a4 sidecar (unchanged) |
| `H3_DIT_NVFP4_KEEP_INT8=` | CSV: `qkv,out,fc1,fc2` or aliases `attn`(=qkv+out), `mlp`(=fc1+fc2), `all` |

Unset/empty KEEP → full FP4 (prior C001 behavior).  
`KEEP_INT8=all` → all four linears INT8 (BIT_ID / sanity vs gold INT8 when overlay).

Requires **INT8 store + NVFP4 sidecar overlay** (lazy INT8 via `dit_lazy_int8`).
Pure `h3_dit_w4a8_open_nvfp4` has no INT8 fallback — do not use KEEP there.

## Code

- `h3_dit_w4a8.c`: parse/log, `nvfp4_fp4_mask`, per-kind `nvfp4_kind_use_fp4`,
  skip FP4 weight load for kept kinds, dispatch gates, gemm_epi/ink/SL use
  `!nvfp4_fp4_mask` (so KEEP=all can exercise fused INT8 epi).
- `h3_env_contract.c` + `tools/print_h3_env_contract.py`: contract entry.

## Ladder (target)

Same turbo4 INT8 864×480 protocol as C001/energy-collapse docs:

1. ctrl INT8 (no NVFP4)
2. NVFP4 full (KEEP unset) — expect INVALID_LOWENERGY / blank
3. KEEP=all — expect near-INT8 (BIT_ID / waffle green)
4. KEEP=attn → KEEP=attn,fc1 → KEEP=out,fc2 → KEEP=qkv … find max FP4 coverage
   that still clears waffle / latent std

Product default stays INT8 until waffle+visual dual-green on a hybrid or full pack.

## Ready weights (inventory)

- DiT sidecar used by C001: `minimax_h3_fl2va_nvfp4_w4a4.safetensors` (naive INT8→NVFP4 requant — known energy limit)
- TE AWQ: `models/comfy-org/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` (text encoder, not DiT)
- Other mixed/codebook packs on disk are not a drop-in fix for DiT W4A4 blank

## Non-goals

- No `H3_DIT_NVFP4_CHAIN`
- No wipe / retrain narrative
- No silent default-on NVFP4
