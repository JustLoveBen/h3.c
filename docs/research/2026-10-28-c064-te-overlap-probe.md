# C064 TE∥DiT GPU prefetch opt-in probe (4090) — Track-A

**Track:** **A** (near-lossless / BIT_ID weights path / default-safe). Do **not** mix KPIs with Track-B (lossy-but-acceptable, e.g. sol_cute+guardrails).

**Date (Asia/Shanghai):** 2026-10-28 00:46–00:49 CST  
**GPU:** lan-4090-31 / RTX 4090 sm_89  
**Gate:** 864×480 × 5s · `p10_street_wave_dolly` · INT8 · skip-audio · cudnn SDPA  
**Policy:** Euler-8 + DMAD-4; PDMD skipped; FaceRefine STOPPED; NVFP4 BLOCK; **C003 EPI_FUSE_SL keep OFF**  
**Hook:** `H3_DIT_TE_OVERLAP_BLOCKS=N>0` + `H3_DIT_HOST_STAGE_TE=0` overrides `latent_t≥32` default skip. Product default unchanged.

## Why this over C003 / C015

- **C003 (Track-A):** EPI_FUSE_SL already OFF (+46% dit) — do not reopen.
- **C015/C042 (Track-A):** act FP16 staging unbuilt; waffle non bit-identical; not small this slice.
- **C064 (Track-A):** whitelist+trace landed; true overlap gated — opt-in unlock only.

## Code change

- `tools/h3_cuda_generate.c`: explicit `H3_DIT_TE_OVERLAP_BLOCKS=N>0` allows TE GPU prefetch even when `latent_t≥32`; still mutually exclusive with HOST_STAGE (warn if both armed).
- `h3_lifecycle_whitelist.h`: notes updated.
- Default: identical (skip GPU prefetch when env unset and `latent_t≥32`; HOST_STAGE on).

## Results (single cold clip / arm; primary KPI = dit_ms)

| arm | te_ms | dit_ms | WALL_s | peak_dit_MiB | after_te_used | note |
|-----|------:|-------:|-------:|-------------:|--------------:|------|
| euler8_hs | 18005.0 | **27371.8** | 60.23 | 22878.9 | 520.9 | HOST_STAGE default |
| euler8_ov8 | 9760.6 | 29795.9 | 56.38 | 22878.9 | 3468.9 | GPU pref=8 |
| dmad4_hs | 12022.8 | **17437.6** | 46.12 | 22706.9 | 4444.9 | HOST_STAGE default |
| dmad4_ov8 | 11699.6 | 18732.1 | 46.45 | 22706.9 | 7392.9 | GPU pref=8 |

- All `RUN_RC=0`; no OOM; logs show `C064 opt-in TE-overlap gate` on ov arms.
- dit Δ ov−hs: Euler **+2424 ms (+8.9%)**; DMAD **+1295 ms (+7.4%)**.
- peak_dit unchanged; after_te +~3 GiB on ov8 (=8×~386 MiB) as expected.
- TE/WALL cold-noise — **do not claim WALL win**.

## Decision

- **Probe closed (Track-A).** CLAIM: **NO_WALLCLOCK_WIN** vs HOST_STAGE.
- Keep **opt-in**; do **not** default-on.
- Product default remains HOST_STAGE_TE on + latent_t≥32 GPU-prefetch skip.
- Root cause: ov8 forces `HOST_STAGE_TE=0` (mutual exclusive) → loses disk→host hide; 8-block GPU pref does not repay.
- Future (still Track-A, separate slice): concurrent small GPU pref **with** HOST_STAGE if VRAM gate allows — not this land.

## Artifacts

- Code: `tools/h3_cuda_generate.c`, `h3_lifecycle_whitelist.h`
- Runner: `tools/run_c064_te_overlap_probe_4090.sh`
- Out: `out/c064_te_overlap_480p_20261028/` (+ SUMMARY.md, labeled mp4)
- Share: `~/Downloads/grokbot-share/c064_te_overlap_480p_20261028/`
- Box: `/workspace/c064_te_overlap_480p_20261028/`
