# C003 EPI_FUSE P2 skip-ConvRot — unreachable boundary (2026-10-27)

## Claim
The E2 P2 optimization that requants SwiGLU output **without** act ConvRot and feeds `pre_q` into FC2 is **numerically invalid** with packed ConvRot weights. It cannot land as a quality-preserving default.

## Math
Weights on device are `W_rot = W @ H` (normalized Hadamard). Baseline:

```
Y = (X @ H) @ W_rot.T = X @ W.T
```

Skip-ConvRot fuse:

```
Y_bad = X @ W_rot.T = X @ H.T @ W.T ≠ X @ W.T
```

DESIGN §3 proved the **rotated** path identity, then jumped to the unrotated act path — that jump is false even in exact arithmetic (before quant). Per-row absmax does not repair it.

## Measurements
| check | result |
| --- | --- |
| Host H4 skip vs true rel_l2 | ≈0.40 |
| 238 fuse↔ctrl latent rel_l2 (`1b37f28`) | ≈1.49 |
| sign_agree | ≈0.50 |
| fuse r1↔r2 DiT checksum | identical (deterministic wrong path) |
| fuse same-config waffle | FAIL meanΔ≈0.39 |
| ctrl same-config waffle | PASS Δ=0 |

## Disposition
- Code: P2 arm **DISARMED** (`epi_fuse=0` even when `H3_DIT_EPI_FUSE=1`); log `epi_fuse_req` vs `epi_fuse`.
- Default **OFF**; env=1 → behavior identical to ctrl (no speedup claim).
- Redesign if revisit: fuse **ConvRot+requant** into the swiglu epilogue (keep algebra); do not skip Hadamard.
- P1 stays disabled (AdaLN); P3 unwired.

See also: [`2026-10-27-c003-epi-fuse-nan-fix.md`](2026-10-27-c003-epi-fuse-nan-fix.md).


## 238 confirm (`3145ba0`)

After disarm: all four arms hot latent checksum `0x1752bd77f232dbb6` (fuse≡ctrl).  
waffle fuse PASS meanΔ=0.013; ctrl PASS meanΔ=0.004.  
`epi_fuse_req=1` / `epi_fuse=0 DISARMED` on fuse arms. Out: `out/c003_epi_fuse_20261027_disarm_from238/`.
