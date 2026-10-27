# C011 CLOSE — VAE cast/prescale fuse (NO_WC_WIN)

**Date:** 2026-10-28 Asia/Shanghai  
**Host:** lan-4090-31 RTX 4090 sm_89  
**Synced binary SHA:** `84e75b418091bf131d3f5ddcc9f617d01a8fca4d` (product stack after C015)  
**Dig:** [`c011-dig`](2026-10-28-c011-vae-cast-prescale-dig.md) @`775cd09`  
**Status:** **stale / closed (NO_WC_WIN)** — no construction; no new env; default unchanged.

## Claim tested

Historical nsys “cast/prescale ~2.29s” still contains ≥0.3 s **VAE** fuseable remainder after product `f16_direct` / `act_f16` / `fused_epi`.

## Product path (confirmed)

| signal | value |
|---|---|
| `vae_gemm` | `f16_direct=1 fused_epi=1 act_f16=1` |
| `vae_gemm_path` | `act_f16` (not `fallback_cast`) |
| `phase_ms` | te=14022.9 dit=27240.7 **vae_blocks=612.1** vae_load=675.0 total=53990.7 |
| WALL | 56.97 s (nsys overhead included) |
| scheme | Euler-8 INT8 864×480×5s p10; TE_CACHE_AUTO; --repeat 1; no LoRA |

## nsys attribution (cuda_gpu_kern_sum + CUPTI gridX split)

| bucket | ms | C011? |
|---|---:|---|
| DiT `h3_cuda_f32_to_f16_row_prescale` (AdaLN) | **1341.5** | No (DiT; default ON) |
| DiT SDPA `h3_cuda_f16_to_f32` (gridX=422884, 400×) | **271.7** | No (SDPA F16→F32; C015-adjacent) |
| already-fused `h3_vae_rms_f32_to_f16` | 112.3 | No (prior win) |
| SIMT/TF32 sgemm family | 214.3 | No → **C012** handoff |
| **VAE weight F16→F32** (`load_linear_f16_gpu`, 144× weight grids) | **13.8** | Yes |
| **VAE weight F32→F16** (`cast_w_to_f16` / `ensure_block_f16_weights`, 144×) | **15.5** | Yes |
| **C011 fuseable meat (weight roundtrip)** | **≈29.3** | — |

Naive script summed all pure `f16↔f32` elementwise → 301.3 ms and printed `HAS_MEAT`. GridX split shows **≈90% of that is DiT SDPA out-cast**, not VAE.

### Weight-grid match (36 blocks × 4 linears)

| gridX | ≈elems | role | f16→f32 ms | f32→f16 ms |
|---|---:|---|---:|---:|
| 131072 | 33.6M | w1 (2×FFN) | 7.32 | 7.69 |
| 65536 | 16.8M | w2 | 3.32 | 3.86 |
| 49152 | 12.6M | qkv | 2.39 | 2.90 |
| 16384 | 4.2M | out | 0.80 | 1.00 |

## Gate

| rule | result |
|---|---|
| ≥300 ms warm VAE fuseable cast | **FAIL** (29.3 ms) |
| waffle / BIT_ID construction | **not started** (no meat) |
| new `H3_VAE_CAST_FUSE` | **not added** |

**Verdict: `NO_WC_WIN` → catalog stale/closed** (same class as C064).

## Mechanism note (why remainder is tiny)

Product already arms VAE M4: fused RMS→F16, act_f16 GEMM, fused epi. Residual VAE dtype tax is almost only the **load-time F16 ckpt → F32 W → F16 W** roundtrip (~29 ms). Skipping it would be a micro-opt below the C011 WC gate; optional future micro under a different id if ever needed — **not** worth a Track-A slice now.

Historical 2.29 s was a **mixed DiT+VAE rollup** (prescale + SDPA cast + sgemm); re-baseline on 4090 after product stack invalidates that number for C011 scope.

## Out of scope (honored)

- No FaceRefine; no LoRA download; no C015 default flip; no C012 TF32 in this slice.

## Next DeepSeek checklist item

Per [`deepseek-derived-track-a-status`](2026-10-28-deepseek-derived-track-a-status.md) priority after C015 + C011:

1. ~~C015~~ built OFF + gate PASS  
2. ~~C011~~ **closed NO_WC_WIN** (this doc)  
3. **C016 PDL** — dig only after a self-written short GEMM→dequant chain exists; expected WC low (−0.1~−0.3 s); do **not** reopen C018/C019/C003-SL/FlashMLA  

Pointer: C012 (VAE SIMT sgemm ~214 ms) remains **idea** with TF32 quality risk — separate from C011 fuse-only; not the DeepSeek #3 pick.

## Artifacts (4090)

- `out/c011_vae_cast_nsys_480p_20261028/` — META, generate.log, nsys-rep/sqlite, REPORT, VERDICT (naive), KPI_extract  
- Tools: `tools/run_c011_vae_cast_nsys_4090.sh`, `tools/c011_nsys_attribute.py`
