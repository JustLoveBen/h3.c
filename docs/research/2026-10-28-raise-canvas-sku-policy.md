# Raise-canvas SKU policy (product doc)

> **Asia/Shanghai 2026-10-28.** Product policy only — no new kernels.  
> Evidence: ghost/face A/B [`ghosting`](2026-10-27-ghosting-and-small-face.md) @`47b4cd8`; 4090 profile (W/H %32==0).

## One-liner

**Far face = canvas pixels.** Do not sell FaceRefine stitch or “one more generic LoRA” as the fix. SKU the resolution + sampler.

## Sellable matrix (4090 · INT8 · cudnn)

| SKU | resolution | sampler | intent | gate note |
|---|---|---|---|---|
| **Speed** | 864×480 (480p) | **DMAD-4** | wall-clock hero | waffle vs Euler optional |
| **Speed+face** | **≤1152×640** | **DMAD-4** (or PDMD-4) | more face-px without Euler cost | 1152×640 PDMD PASS; above → noise risk |
| **Max face** | **1344×768** | **Euler-8** | best far-face budget | 1344×PDMD/DMAD FAIL (noise / INT8 window≈6); Euler-8 PASS |
| **Avoid** | 1280×720 | any | W/H must **%32==0** | rejected by geometry |

## Policy rules

1. **DMAD/PDMD ceiling:** ship fewstep marketing only up to **1152×640** on 4090 INT8.  
2. **1344×768** = **Euler-8** quality SKU (not fewstep). Quote longer wall-clock honestly.  
3. **480p DMAD** remains the speed demo; do not claim it fixes distant faces.  
4. **FaceRefine / crop→F2VA→stitch:** STOPPED — not a SKU.  
5. **Detail LoRA / refiner (C035):** optional polish inside the same canvas; never a substitute for raise-canvas.  
6. **Track-B S2 sol** ([`s2-recipe`](2026-10-28-track-b-s2-sellable-recipe.md)) is an opt-in WC knob at 480p; it does not raise face-px.

## Copy for sales / README

- “Fast preview: DMAD-4 @480p–1152.”  
- “Best faces in frame: Euler-8 @1344×768.”  
- “Sparse-attn turbo (opt-in): sol + LAYERS=2 + window=24 — not default.”

## Cross-links

- [`ghosting-and-small-face`](2026-10-27-ghosting-and-small-face.md)  
- [`accel-quality` S0](2026-10-28-accel-quality-theory-and-industry.md)  
- [`track-b-s2-recipe`](2026-10-28-track-b-s2-sellable-recipe.md)  
