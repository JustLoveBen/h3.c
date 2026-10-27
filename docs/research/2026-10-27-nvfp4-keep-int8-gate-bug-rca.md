# NVFP4 KEEP_INT8 mlp/qkv FAIL — implementation bug RCA (2026-10-27 Asia/Shanghai)

**Status:** root cause confirmed (implementation bug, not theory); fix in `h3_dit_w4a8.c`; postfix ladder **PASS** on 238 (`out/keep_int8_ladder_postfix_20261027/`). Multi-scene visual gate running.

## Symptom (pre-fix ladder)

| arm | KEEP | fp4_mask | latent cksum | std | note |
|---|---|---|---|---|---|
| k_all | all | 0x0 | `0x75efbb7057231ead` | 1.026 | BIT_ID vs INT8 gold |
| k_attn | attn (qkv,out) | 0xc | `0xc45bd33eb5b2d7d1` | 1.024 | energy OK |
| k_mlp | mlp (fc1,fc2) | 0x3 | `0xef9705a253f47c82` | **12.12** | explode |
| k_qkv | qkv | 0xe | `0x95c5882d174bced9` | **0.310** | **≡ k_full** |
| k_full | unset | 0xf | `0x95c5882d174bced9` | **0.310** | collapse |

Mask parse / log were **correct** (`attn→qkv,out mask=0xc`, `mlp→fc1,fc2 mask=0x3`, `qkv→mask=0xe`). Kind enum matches bits (`QKV=0 OUT=1 FC1=2 FC2=3`). Weight load skip for KEEP kinds is correct. **Not** a theory-of-quantization failure — hybrid path control-flow bug.

## Smoking gun

`KEEP=qkv` and full FP4 share **bit-identical** latent checksum. Under correct kind gating that is impossible unless the paths recombine before the divergence can matter.

Trace of `block_phase_out_mlp` when `nvfp4_fp4_mask != 0` (any hybrid with ≥1 FP4 linear → `gemm_epi = fused && !fp4_mask` is **0**):

1. FP4 `out` dispatch writes **already-gated** residual into `d_b` (`nvfp4_gemm_dispatch` gate slot 2), then `goto out_done`.
2. Post-`out_done` still took the `else if (fused)` branch:
   `gate_adaln(d_x, d_a → d_b, d_a)`.
3. At that point `d_a` is still the **pre-qkv AdaLN activation** (written in `block_phase_qkv`), **not** the out-proj result. The FP4 out result in `d_b` is wiped.
4. Because AdaLN→`d_a` runs **before** QKV, wiping makes post-attn residual **independent of whether QKV was INT8 or FP4** → `KEEP=qkv` ≡ full FP4 checksum.

Second bug for `KEEP=mlp` (fc2 INT8 under `nvfp4_on`):

5. INT8 `fc2` (non-`gemm_epi`) writes into `d_a`.
6. Trailer was `if (gemm_epi || ctx->nvfp4_on) return 1` — skipped `gate_mlp`, left **`d_y` unwritten** (stale). Explains std≈12 energy explode, not "MLP FP4 is theoretically toxic".

`KEEP=attn` survived because **out stayed INT8** (correct `d_a` + `gate_adaln`) and **fc2 stayed FP4** (writes gated `d_y` itself).

## Fix

In `h3_dit_w4a8.c` `block_phase_out_mlp`:

- `out_fp4_gated` set on successful FP4 out; post-out uses `if (gemm_epi || out_fp4_gated) adaln_mlp(d_b→d_a)` (no re-gate).
- `fc2_fp4_gated` set on successful FP4 fc2; trailer `if (gemm_epi || fc2_fp4_gated) return` — INT8 fc2 under hybrid still runs `gate_mlp`.

## Expected after fix (postfix ladder)

- `k_all` still BIT_ID gold.
- `k_qkv` **must diverge** from `k_full` checksum (hard regression gate).
- `k_mlp` should leave explode-if-pure-bug; may still be weaker than attn if naive w4a4 attn error remains — judge by std/checksum/visual, not theory.
- `k_attn` should remain energy-OK (control).

## Non-goals

No convert / chain / EPI default-on. Product default stays INT8 until multi-scene visual + waffle clear.

## Postfix ladder (238, binary `a4eef5b9…`, Asia/Shanghai ~13:04–13:10)

Hard gates:

| arm | KEEP | fp4_mask | checksum | std | vs pre-fix |
|---|---|---|---|---|---|
| k_all | all | 0x0 | `0x75efbb7057231ead` | 1.026 | still BIT_ID gold |
| k_attn | attn | 0xc | `0xc45bd33eb5b2d7d1` | 1.024 | unchanged control |
| k_mlp | mlp | 0x3 | `0xb6b84af72babd4c2` | **1.05** | was std **12.12** explode |
| k_qkv | qkv | 0xe | `0x4a7cc3d4c2a8a984` | **1.02** | was ≡ full / std 0.31 |
| k_full | unset | 0xf | `0x9a6ccab3b1c6c10b` | **1.036** | was collapse std 0.31 |
| k_attn_fc1 | attn,fc1 | 0x8 | `0x7819b566c9554325` | 1.023 | energy OK |
| k_attn_fc2 | attn,fc2 | 0x4 | `0xc5173989c8869a29` | 1.028 | waffle 0/5 INVALID |

**PASS hard regression:** `k_qkv` checksum **≠** `k_full`.  
**PASS:** `k_mlp` no longer energy-explodes.  
**Bonus:** full FP4 energy restored — C001 blank/collapse was the same out-gate wipe, not (only) naive w4a4 math. Visual/OR still required before any product default-on.

Artifacts: `out/keep_int8_ladder_postfix_20261027/{k_*}/`.

## Visual probe (postfix p01_sprint t≈2s, Asia/Shanghai 13:14)

Frame pull: `out/keep_int8_ladder_postfix_20261027_from238/frames/` + `~/grokbot-share/postfix_frames/`.
PSNR vs INT8 gold (ffmpeg, avg): attn 17.16 / mlp 17.12 / qkv 17.38 / full 18.33 / attn_fc1 18.36 / attn_fc2 18.55 — all diverge (expected); PSNR alone does not rank usability.

| arm | latent energy | human/agent frame read | verdict |
|---|---|---|---|
| k_all | BIT_ID gold | recognizable sprinter, stadium, mild gen fuzz | gold |
| k_attn | OK | recognizable; more blocky/painterly than gold | alive (experimental) |
| k_attn_fc1 | OK | clear wet-track sprint, structure intact | alive candidate |
| k_attn_fc2 | OK | vertical smear / pixelation around runner | weak |
| k_mlp | OK (fixed) | **systematic mosaic/grid overlay** | FAIL visual |
| k_qkv | OK (≠full) | **screen-door grid + chroma fringe** | FAIL visual |
| k_full | OK (fixed) | content present but heavy noise/distortion | FAIL visual (not blank) |

**Mechanism after gate fix:** mlp/qkv no longer explode/collapse via AdaLN wipe — residual FAIL is **FP4 GEMM quality on those kinds** (naive w4a4 / SF path), not another KEEP control-flow bug. Mask/kind/weight-skip were already correct pre-fix; the only hard implementation bug was the dual gate hybrid. Product still stays INT8 default; recommend experimental arm = `KEEP=attn` or `attn,fc1` pending multi-scene.

OpenRouter :free VL blocked this turn (`401 User not found` on env key) — local frame read used instead; re-run OR when key refreshed.

## Mask / path hard evidence (postfix logs)

| arm | KEEP_INT8 log | fp4_mask | gemm_epi | act_convrot | dit_ms (8step) |
|---|---|---|---|---|---|
| k_all | qkv,out,fc1,fc2 | 0x0 | 1 | ON | 38345 |
| k_attn | qkv,out | 0xc | 0 | ON | 37233 |
| k_mlp | fc1,fc2 | 0x3 | 0 | ON | 39295 |
| k_qkv | qkv | 0xe | 0 | ON | 30511 |
| k_full | unset | 0xf | 0 | ON | **23545** |
| k_attn_fc1 | qkv,out,fc1 | 0x8 | 0 | ON | 40137 |
| k_attn_fc2 | qkv,out,fc2 | 0x4 | 0 | ON | 29876 |

Bits: QKV=0 OUT=1 FC1=2 FC2=3; `fp4_mask=(~keep)&0xF`. Logs match intended GEMM split. `gemm_epi=0` whenever any FP4 linear (hybrid) — consistent with `fused && !fp4_mask`. No evidence of wrong-kind miss or act ConvRot asymmetry across arms. Weight path: INT8 ConvRot store + FP4 sidecar (1332 tensors) for FP4 kinds.

Host/device A/B at clip level = this postfix ladder (same seed/ids/protocol). Single-GEMM rel_l2 probe not required to close the gate-wipe bug; residual mlp/qkv mosaic remains a **quality** (naive w4a4) issue under correct routing.
