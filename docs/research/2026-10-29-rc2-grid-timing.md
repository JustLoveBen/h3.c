# RC2 16px grid RCA + RC0 “warm 49.78s” KPI mislabel (2026-10-29)

## Scope
- User t88u: RC2 labeled clip shows heavy grid overlay.
- User t89u: `a1_rc0_smoke_labeled.mp4` burn-in claimed **warm 49.78s** for a **1s** clip — absurd; do not wait for users to spot KPI lies.

Evidence path (no videoReview): ffmpeg frame extract + FFT / 16px SNR / overlay on **unlabeled** sources.

Artifacts: `~/grokbot-share/rc2_grid_rca_20261029/` (Mac) and `/workspace/rc2_grid_rca_20261029/` (box).

---

## Part A — RC2 grid (hard evidence)

### Period / locus
| Check | Result |
|---|---|
| Present on unlabeled? | **Yes** — not burn-in artifact |
| Dominant period | **~16 px** (VAE latent cell / patch seam scale; 1344/84 = 16) |
| Encode starvation? | **No** — RC2 encode CRF14 hires, ~11.4 Mbps, bpp~0.46 (prior 1344 blockiness was encode; this is not) |
| Spatial tile seams (~256–320 px)? | **No** — overlay / SNR peaks at 16 px, not tile pitch |
| NVFP4? | **No** — path is INT8 ConvRot + C015 FP16 staging |
| Waffle / deblock16 | mean_b16 ~0.69 soft grid; `H3_VAE_DEBLOCK` default 0.28 still leaves 16 px structure on smooth sky/sidewalk |

### Cross-SKU
- RC0 / RC1 also show 16 px SNR peaks; RC2 most visible on large smooth regions (sky, sidewalk).
- INT8/C015 vs BF16 A/B and raising `H3_VAE_DEBLOCK` left as follow-up dig (not required to close labeling/KPI).

### Annotated frames
See RCA dir: unlabeled stills + FFT magnitude + 16 px period overlay.

---

## Part B — RC0 “warm 49.78s” timing RCA (t89u)

### Was duration really 1s?
**Yes.** Gate is `sec=1` / 24f. Probe of unlabeled / labeled mp4 duration ≈ 1.0 s. The absurdity is the **label text**, not clip length.

### Cold vs warm / TE load
From `a1_rc0_smoke/generate.log` (repeat=1, hot=0 only):

```
phase_ms: te=26922.4 dit=5022.7 vae=7659.8 vae_load=5764.3 ... total=49292.7 ... hot=0 clip=1/1
WALL_SEC=49.78
```

| Phase | ~s | Notes |
|---|---|---|
| TE | 26.9 | dominates cold |
| DiT | 5.0 | actual 1s DMAD-4 solve |
| VAE (+load) | ~7.7 (+5.8 load) | cold VAE path |
| Process WALL | **49.78** | `/usr/bin/time` over whole generate |

**Cold TE+load, not warm product KPI.** There is no `hot=1` phase when `repeat=1`.

### Wrong arm?
No — RC0 864×480 DMAD-4 INT8 C015 is the intended A1 smoke arm. Wrong **metric arm**: runner treated process WALL as “warm”.

### Mislabeled burn-in (root cause)
`tools/run_a1a2a3_product_gates_4090.sh` `extract_warm()` took last `WALL_SEC=` and burned:

`RC0 864x480 DMAD-4 INT8 C015 | warm ${WARM}s`

So cold process wall became “warm 49.78s”. Same bug on RC1/RC2: labeled “warm” was **full process WALL ≈ cold+hot sum** (e.g. RC2 WALL≈279s while hot phase≈125s).

### Fix (this change)
1. Replace `extract_warm` with `extract_timing` (proc_wall + cold/hot phase_ms → seconds).
2. Burn-in rules:
   - If `hot=1` present → `hot Xs (dit Ys)`; if proc ≫ hot, append `proc Zs cold+hot`.
   - Else → `COLD Xs (te Ys) NOT warm-KPI` — never the word “warm”.
   - Always include clip seconds (`1s` / `5s`) in the label.
3. `flag_absurd_timing`: self-alert when `sec≤2` and `proc_wall≥30`, or no hot phase — write `ALERT_KPI_TIMING` to META (do not wait for user).
4. A1 smoke `repeat=1` → `repeat=2` so next gates can quote true hot KPI for 1s smoke.
5. `burn_eval_label.sh` comment updated: never burn cold WALL as warm; 1s ≠ 5s product KPI.

Expected re-label for existing RC0 log (reburned on share):  
`RC0 864x480 DMAD-4 INT8 | 1s COLD 49.78s te26.9s NOT-warm`

Expected RC1 / RC2 style:  
`RC1 … | 5s hot 50.69s dit36.7s wall131.73s`  
`RC2 … | 5s hot 125.38s dit105.4s wall279.06s`

`burn_eval_label.sh` now picks fontsize from frame width × label length so 480p does not clip.

---

## Policy
**后续这种问题不要自己发现 → agent 必须自己发现：** any burn-in / META KPI where wall ≫ clip duration × plausible warm rate (e.g. 1s smoke with ≥30s “warm”) must trip `ALERT_KPI_TIMING` before share.


## Follow-up (same day)
- `flag_absurd_timing` now **refuses SHARE warm post** when no `hot=1` (COLD-only → `SHARE/refused/`); still burns honest `COLD … NOT-warm` locally.
- Also refuse when `hot_total > 12× clip_seconds` (vs DMAD ~25s/5s ≈ 5s/s).
- Dedicated re-verify: `tools/run_rc0_dmad_warm_kpi_4090.sh` (RC0 DMAD 1s+5s, repeat=2, hot-only labels).
