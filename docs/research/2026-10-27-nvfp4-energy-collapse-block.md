# NVFP4 energy/std collapse → visual blank — terminal BLOCK (2026-10-27)

Asia/Shanghai. Host evidence on `yongbin@192.168.0.238`. Parent SHA at probe: `7f3cbdf`.

## Symptom (unchanged after SF stride=ncb)
- C001 P0 NVFP4: protocol RC=0, waffle `INVALID_LOWENERGY`, OR VL **blank muddy**.
- Latent `std≈0.3097` (checksum `0x95c5…`) vs INT8 `std≈1.026` (`0x75ef…`) — ~3.3× attenuation.
- Prior mosaic FIXED by act ConvRot + SF 32×4 + stride=`ncb`. Failure mode is now **low energy**, not rainbow.

## Ruled out (instrumented)
| Hypothesis | Evidence | Verdict |
|---|---|---|
| SF row-stride / in-tile wrong | DiT-K e2e `32x4+stride_ncb` PASS at K=512/5376, N up to 21504, rel_l2≈0.0017; `stride_ncb_rd` FAIL | not SF-GEMM |
| `weight_scale2≡1.0` forgotten | Sidecar all 1.0 by design — per-row INT8 scale folded into UE4M3 SF in `quant_linear` | expected contract |
| Weight reconstruction attenuates | Host probe `probe_nvfp4_weight_norm.py`: `\|\|W_fp4\|\|/\|\|W_i8\|\|≈1.00`, rel_l2≈0.095 across qkv/out/fc1/fc2/b25 | not weight energy |
| Act ConvRot missing | Log: `act_convrot=1 gs=256` on overlay INT8+sidecar path | ON |
| Single-linear output energy gap | Host `probe_nvfp4_gemm_ab.py` (ConvRot + INT8 per-row vs FP4 per-16 dequant GEMM): **fp4/i8_norm≈0.999** for unit/LN/large/small(0.05); tiny(0.005) only ~2% with SF underflow | **not single-GEMM scale bug** |
| NaN wipe | Finite latent; DEBUG_NAN / waffle low-energy not NaN | finite attenuation |

## Root cause class (honest)
Offline W4A4 requant from INT8 ConvRot introduces **~13% rel_l2 per linear** on the host A/B path (INT8 path ~0.9%). Energy (RMS) is preserved per GEMM, but the **directional / spectral error** accumulates across 50 blocks × 4 linears × 8 diffusion steps and steers the ODE into a **low-energy attractor** (blank muddy, latent std 0.31).

This is an **arm/contract gap**, not a missing `acc_scale` / dequant / SF-layout bug:
- No production default.
- No fake PASS / no green visual gate.
- Chain (`H3_DIT_NVFP4_CHAIN`) stays gated — C001 visual still FAIL.
- EPI_FUSE / EPI_FUSE_INK remain **DISARMED / opt-in** (ink BIT_ID already PASS ~0% WC; unrelated to this blank).

## Harden landed (not a visual fix)
`h3_cuda_quantize_act_nvfp4_kernel`: clamp UE4M3 SF underflow to byte `1` (same as `tools/quant_dit_nvfp4.py` weight `np.maximum(sf, 1)`), and guard `inv` against div0. Closes act/weight SF contract gap for pathological tiny micro-blocks; **does not** recover C001 blank (single-layer energy already matched).

Chain WIP file `h3_dit_nvfp4_chain.cu` **not touched**.

## What would be needed for a real NVFP4 visual arm
Not in this knife (would be a new contract, not a one-line fix):
- Calibration / SmoothQuant / GPTQ-style NVFP4, or QAT;
- Or mixed precision (keep sensitive linears INT8);
- Or accept a separate SKU with its own visual gate — never silently replace INT8 default.

## Policy
- **C001 NVFP4 visual = terminal BLOCK** for product default.
- INT8 ConvRot remains the theme-follow / gold path.
- Do not open chain; do not default-on EPI from this result.
- EPI single-launch fuse may be evaluated as a **separate** ticket (ink already green) — not blocked by NVFP4 math, but also not a substitute for NVFP4 visual.

## Artifacts
- `/tmp/probe_nvfp4_weight_norm.py` (238)
- `/tmp/probe_nvfp4_gemm_ab.py` (238)
- Prior: `docs/research/2026-10-27-nvfp4-ncb-still-block.md`
