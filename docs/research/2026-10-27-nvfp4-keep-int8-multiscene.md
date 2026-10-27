# NVFP4 KEEP_INT8 multi-scene visual gate (2026-10-27 Asia/Shanghai)

**Protocol:** 864×480 5s 8step cfg1 seed `0x76800002` cudnn skip-audio; TE AWQ + INT8 ConvRot DiT + NVFP4 w4a4 sidecar.  
**Prompts:** p01_sprint / p02_talking_head / p03_corridor / p05_neon_sign / p07_market_track (`evals/768p/ids/`).  
**Arms:** `int8` control; `KEEP=attn`; `KEEP=attn,fc1`; `KEEP=attn,fc2`.  
**Binary:** postfix gate-fix `a4eef5b9…` on 238. Artifacts: `out/keep_int8_multiscene_20261027/` (238) → Mac `out/keep_int8_multiscene_20261027_from238/` + `~/grokbot-share/keep_int8_multiscene_20261027/`.

## Latent energy table (all RUN_RC=0)

| prompt | int8 std / cksum | k_attn std / cksum | k_attn_fc1 | k_attn_fc2 |
|---|---|---|---|---|
| p01_sprint | 1.026 / `0x75ef…1ead` | 1.024 / `0xc45b…d7d1` | 1.023 / `0x7819…4325` | 1.028 / `0xc517…9a29` |
| p02_talking_head | 1.113 / `0x404b…ed1b` | 1.122 / `0xd213…a509` | 1.113 / `0x6d3e…fc26` | 1.112 / `0xf280…5c38` |
| p03_corridor | 1.021 / `0x556d…a201` | 1.023 / `0xdecf…7ebc` | 1.024 / `0xa62e…a90e` | 1.028 / `0x03ca…c5a6` |
| p05_neon_sign | 1.176 / `0xbfd3…ebb1` | 1.146 / `0xae50…59f1` | 1.149 / `0x00e3…2e2b` | 1.167 / `0x560b…7268` |
| p07_market_track | 0.987 / `0xa627…5a8a` | 0.961 / `0x4a53…54e8` | 0.962 / `0xea0b…0ac9` | 0.983 / `0x33a1…623a` |

No collapse/explode on any arm×prompt. Cold waffle `INVALID_LOWENERGY` still fires on some INT8 gold clips (known FP) — ignore for pass/fail.

## DiT wall-clock (ms, 8 evals; warm after first)

| arm | typical dit_ms | notes |
|---|---|---|
| int8 | ~24000–24100 | gold |
| k_attn | ~24200–27800 | ≈ INT8 once warm; MLP stays FP4 |
| k_attn_fc1 | ~26500–49000 | first clip cold; only fc2 FP4 |
| k_attn_fc2 | ~24800–30500 | only fc1 FP4 |
| k_full (postfix) | ~23500 | fastest when routing OK |

## Visual spot-check (t≈2s frames; agent read — OR :free 401 this turn)

| scene | int8 | k_attn | note |
|---|---|---|---|
| p01_sprint (postfix) | recognizable | alive, more blocky | attn_fc1 looked sharpest among hybrids |
| p02_talking_head | sharp face | recognizable, softer | no mosaic |
| p03_corridor | clear colonnade | structure OK, painterly/waxy | no collapse |
| p05 / p07 | (pulled) | energy OK | share dir for human review |

Postfix mlp/qkv (p01 only): mosaic/screen-door **FAIL** despite energy OK — see gate-bug RCA. Not re-run in multi-scene (quality, not routing).

## Recommendation

- **Product default:** stay INT8.
- **Experimental arm (document, never silent default-on):** `H3_DIT_NVFP4=1` + `H3_DIT_NVFP4_KEEP_INT8=attn` (optionally `attn,fc1`). Passes multi-scene energy; visual alive vs INT8 with mild quality loss.
- **Do not ship:** full FP4 / KEEP=mlp / KEEP=qkv until naive w4a4 quality fixed (or better calib).
- Gate hybrid AdaLN bug fixed in `a1b7aff` — required for any hybrid/full FP4 energy.

## Open items

- Refresh OpenRouter key → :free VL per clip (nemotron/dots).
- Human phone review via `~/grokbot-share/`.
- Optional: single-GEMM rel_l2 host probe for out/fc* SF (not blocking; routing proven).
