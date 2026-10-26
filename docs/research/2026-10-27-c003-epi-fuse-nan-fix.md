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

## 2026-10-27 retest (still NaN) → scratch alias fix

After disabling P1, 238 retest at `c11357b` still had fuse `latent_stats: mean=nan`, mux ~100KB, dit ~11% faster. Root cause #2:

**Producer (`w_act_dev_epi`) requant layout skipped f16 ConvRot scratch** when `requant_mode!=0`, sizing only `m_pad*rq_width` and setting `d_xi8=base`. Act-f16 prep still wrote `d_xh`/`d_rot_h` at `base` (needs ~`4*K` bytes/row; FC1 K=5376 vs ffn rq_width=14336) → buffer overrun. Tiled GEMM could also scramble later input rows when requant reused `d_xi8` with wider stride.

**Fix:** always size `d_act_scratch` for this GEMM's act prep (K-based). Put fused requant **output** in dedicated `d_rq_scratch` (already existed for `pre_q`). Default still OFF until re-verify.

## 2026-10-27 scratchfix retest @ `1b37f28` (238)

Tree: `/mnt/nvme1n1/workspace/h3_c_main_20261027`  
Binary sha256: `2cd0d12e17c7a1685b26d86b8e9c12d0bd11089d335ab11e5acf7746f17b7e64`  
Out: `out/c003_epi_fuse_20261027_scratchfix/` → Mac `out/c003_epi_fuse_20261027_scratchfix_from238/`

### Hot clip2 dit_ms (turbo4 INT8 480p, seed `0x48000001`)

| arm | cold dit_ms | hot dit_ms | hot latent mean / checksum |
| --- | ---: | ---: | --- |
| ctrl_r1 | 14054.6 | 13814.2 | −0.07335 / `0x1752bd77f232dbb6` |
| ctrl_r2 | 14190.9 | 13882.9 | −0.07335 / `0x1752bd77f232dbb6` |
| fuse_r1 | 13734.8 | 13419.5 | +0.07044 / `0xc7fd9102e37ffd7b` |
| fuse_r2 | 13781.0 | 13453.9 | +0.07044 / `0xc7fd9102e37ffd7b` |

- **NaN gone:** fuse latents finite both runs; checksums stable r1↔r2.
- **mux sane:** fuse hot ~5.9–6.0 MB (was ~100 KB); ctrl hot ~3.8 MB.
- **dit speedup:** hot mean ctrl 13848.6 → fuse 13436.7 ≈ **2.97%** (prior ~8.4% was the broken NaN path skipping real work).
- **not bit-identical to ctrl** (expected): fuse mean sign-flipped vs ctrl; latent checksum differs.
- **waffle same-config (hot-only last 5 lines):** ctrl **PASS** Δ=0; fuse **FAIL** mean|Δb16|=0.393 max=0.534 (tol 0.15/0.4). Fuse b16≈7–10 vs ctrl≈0.73–0.82; all frames still `INVALID_LOWENERGY`.
- Fuse mux sizes also drift r1↔r2 despite identical latent checksums (VAE/encode nondeterminism under these latents).

### Verdict
- Corruption-to-NaN: **FIXED**.
- Quality / determinism gate: **FAIL** — keep `H3_DIT_EPI_FUSE` **default OFF**; do not treat as landed.
- Next (not this batch): residual numerical error on P2 path (fused requant vs baseline ConvRot+quant), or accept non-default research arm only after human visual on `fuse_*_r2.mp4`.


## 2026-10-27 P2 skip-ConvRot unreachable (waffle Δ root cause)

### Evidence
- After NaN fixes (`1b37f28`), fuse DiT was run-to-run deterministic (identical latent checksum + `latent.raw` hash r1↔r2) but **same-config waffle FAIL** (mean|Δb16|≈0.39).
- Fuse vs ctrl latent (same seed): **rel_l2≈1.49**, sign_agree≈0.50 — catastrophic, not f16 noise.
- Waffle compares mux_rgb after VAE; bad latents amplify VAE/uint8 nondeterminism → mp4 drift despite stable DiT checksums.

### Root cause (algebra)
Packed INT8 weights are ConvRot-rotated: `W_rot = W @ H`. Correct identity (`h3_w4a8_dequant.h`):

`Y = X @ W.T == (X @ H) @ W_rot.T`

P2 fused requant fed `quant(X)` into `pre_q` (skip act ConvRot) → `quant(X) @ W_rot.T`, which is **not** equivalent. DESIGN §3 proved `quant(X·H)·W_rot.T ≡ …` then incorrectly claimed skip-rot is OK.

Host Hadamard (H4 normalized): skip-rot vs true rel_l2≈0.40; quant+skip vs quant+rot≈0.40.

### Fix / boundary
- **Disarm** P2 in `h3_dit_w4a8.c`: `epi_fuse` hard-forced 0; log keeps `epi_fuse_req` when env set. No `requant_mode` / no `pre_q`.
- Default remains **OFF**. Env=1 is currently a **no-op** (fuse≡ctrl) until in-kernel ConvRot+requant redesign.
- Tracked note + catalog; DESIGN §8 errata under gitignored `out/review-…/DESIGN.md`.

### Not race / scratch
DiT fuse path itself was deterministic; prior scratch/AdaLN bugs were separate and already fixed. Remaining waffle FAIL was numerical wrongness of skip-ConvRot, not a race.


## 2026-10-27 disarm retest @ `3145ba0` (238)

P2 skip-ConvRot **DISARMED**. fuse≡ctrl checksum `0x1752bd77f232dbb6`; waffle fuse+ctrl **PASS**; dit ≈0% delta. See [`skip-convrot-unreachable`](2026-10-27-c003-epi-fuse-skip-convrot-unreachable.md) + gpu-unified addendum.
