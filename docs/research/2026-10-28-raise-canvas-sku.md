# Raise-canvas SKU productization (4090)

> **Asia/Shanghai 2026-10-28.** Product packaging of the raise-canvas policy that shipped doc-only after S2 @`3e4414b`.  
> **Supersedes-as-product:** [`raise-canvas-sku-policy`](2026-10-28-raise-canvas-sku-policy.md) (policy sketch kept for history).  
> **Prior axis done:** NFE ladder @`7cc2a0f` — [`nfe-fewstep-contracts`](2026-10-28-nfe-fewstep-contracts.md).  
> **Policy:** primary arms **DMAD / Euler**; **PDMD = contrast-only**; FaceRefine **STOPPED**; no invent LoRA (G2 C035 BLOCK); Track-A near-lossless vs Track-B lossy **separate**.

## 0. One-liner

**Far face = canvas pixels.** SKU the resolution × sampler/NFE. Do not sell FaceRefine stitch or “one more generic LoRA” as the fix. Compose with NFE ladder + optional S2 WC; do not waffle across resolutions or across Euler↔DMAD.

## 1. Inventory (docs + evidence SHAs)

| Artifact | SHA / path | Role |
|---|---|---|
| Ghost/face A/B ledger | @`47b4cd8` · [`ghosting-and-small-face`](2026-10-27-ghosting-and-small-face.md) | 1152×640 PDMD PASS; 1344 PDMD FAIL noise (window≈6); 1344 Euler-8 PASS best face |
| Face LoRA vs FaceRefine | [`face-lora-vs-facerefine`](2026-10-28-face-lora-vs-facerefine.md) | Raise-canvas primary; FaceRefine STOP; face LoRA auxiliary only |
| Opt menu | @`9739646`+ · [`4090-opt-menu`](2026-10-28-4090-opt-menu-beyond-dmad.md) §#3 | Raise-canvas listed as quality SKU |
| S2 recipe | @`3e4414b` · [`track-b-s2`](2026-10-28-track-b-s2-sellable-recipe.md) + `tools/h3_track_b_s2_env.sh` | Opt-in WC; does **not** raise face-px |
| NFE ladder | @`7cc2a0f` · [`nfe-fewstep-contracts`](2026-10-28-nfe-fewstep-contracts.md) + `tools/h3_nfe_product_ladder_env.sh` | Euler-8 / DMAD-4 / PDMD-4 contrast |
| Policy sketch (doc-only) | @`3e4414b` · [`raise-canvas-sku-policy`](2026-10-28-raise-canvas-sku-policy.md) | Matrix before this productization |
| C015 default ON | @`b99d56b` | Act-FP16 stage product default |
| C041 window24 | @`5e994fc` | Opt-in VRAM; required for sol headroom |
| Clips (reuse) | `~/grokbot-share/ghost_face_ab_20261027/labeled/` | `talking_head_1152x640_pdmd4_labeled.mp4`, `talking_head_1344x768_s8_labeled.mp4`, `…_pdmd4_FAIL_NOISE_labeled.mp4` |

## 2. Sellable SKU table (4090 · INT8 · 24G)

| SKU id | Resolution | Sampler / NFE | Track | When to use | VRAM note (4090) | Quality / face note |
|---|---|---|---|---|---|---|
| **RC0** `480p_default` | **864×480** | **Euler-8** (quality) **or** **DMAD-4** (speed) | A / B | Speed demo + waffle gold / speed fewstep | Comfortable; C015 ON; optional `WINDOW=24` | **Not** a far-face fix (~34px face on wide p10) |
| **RC1** `1152_dmad_speed_face` | **1152×640** | **DMAD-4** (+ full_critic LoRA) | B | Far-face **speed** SKU — more face-px without Euler WC | Usable; INT8 window stays healthy (~18 hist.) | Face ≫ 480p; weaker than 1344 Euler |
| **RC1c** `1152_pdmd_contrast` | **1152×640** | **PDMD-4** (contrast) | B contrast | Paper/OR contrast at same canvas | PASS @ghost A/B | **Not** product primary (t71u) |
| **RC2** `1344_euler_max_face` | **1344×768** | **Euler-8** (no LoRA) | A | **Max face budget** quality SKU | Tight but PASS (~143s e2e hist.); prefer C015 ON + optional window24 | Best far-face; quote longer WC honestly |
| **RC960** `960_optional` | **960×544** | DMAD-4 or Euler-8 | B / A | Intermediate step-up from 480p | Easy | Optional rung; not required for ship |

**Geometry rule:** W and H must each be **multiples of 32**.

## 3. Hard rejects (refuse to sell / helper refuses)

| Combo | Why |
|---|---|
| **1280×720** (any sampler) | W/H not %32==0 → geometry reject |
| **1344×768 + PDMD-4** (4090 INT8) | Auto-downgrade window≈6 → **FAIL noise mosaic** (RUN_RC=0 but unusable) @`47b4cd8` |
| **1344×768 + DMAD-4** (4090 INT8) | Same class risk as PDMD at 1344 (fewstep + tiny window); **do not sell** — use Euler-8 |
| **FaceRefine** crop→F2VA→stitch | STOPPED — not a SKU |
| **C035 / detail LoRA as face fix** | G2 BLOCK — no TaoMate/refiner weights on disk; auxiliary polish only if weights appear later |
| Cross-arm **waffle** Euler↔DMAD or 480↔1344 | Invalid gate — Track-A vs Track-B and canvas SKUs are separate |

## 4. Composition with NFE / S2 / C041 / C015

```text
source tools/h3_nfe_product_ladder_env.sh {euler8|dmad4}   # NFE arm
source tools/h3_raise_canvas_sku_env.sh {480p|1152_dmad|1344_euler}  # canvas SKU
# optional WC (Track-B only, usually on 480p/1152 DMAD — not a face-px lever):
source tools/h3_track_b_s2_env.sh   # WINDOW=24 + sol LAYERS=2; still need --sdpa-backend sol_cute
# C015: product default ON (unset). Opt-out: H3_DIT_ACT_FP16_STAGE=0
# C041 alone (cudnn, no sol): export H3_DIT_INT8_WINDOW=24
```

| Stack piece | Default? | Raises face-px? | Notes |
|---|---|---|---|
| C015 act-FP16 | **YES** (unset) | No | VRAM headroom for higher canvas |
| C041 `WINDOW=24` | NO (opt-in) | No | Helps 1344 Euler / sol pool; does not invent face tokens |
| S2 sol LAYERS=2 | NO | No | Track-B WC only; cudnn product default unchanged |
| NFE Euler-8 / DMAD-4 | product arms | No | Choose per Track; then pick canvas SKU |
| Raise-canvas SKU | this doc | **YES** | Primary far-face lever |

## 5. Product entry (preferred)

```bash
tools/h3_product_generate.sh --sku rc0 --prompt-ids evals/768p/ids/p10_street_wave_dolly.ids --out out/rc0
tools/h3_product_generate.sh --sku rc1 --prompt-ids … --out out/rc1
tools/h3_product_generate.sh --sku rc2 --prompt-ids … --out out/rc2
# refuses: --canvas 1280x720 / --nfe dmad4 --canvas 1344
tools/h3_product_generate_smoke.sh
```

Full wiring: [`product-generate-entry`](2026-10-28-product-generate-entry.md). Env-compose examples below remain for digs.

## 5b. Copy-paste CLI examples (env compose)


Assume model root + generate binary already on PATH; adjust `-d` / binary name to the machine.

### 5.1 RC0 — 480p default (speed = DMAD)

```bash
source tools/h3_nfe_product_ladder_env.sh dmad4
source tools/h3_raise_canvas_sku_env.sh 480p
# generate … --width 864 --height 480 --duration 5 \
#   --fewstep-preset dmad4 --lora "$H3_DMAD_LORA" --lora-scale 1 \
#   --cfg 1 --sdpa-backend cudnn --skip-audio
```

Quality gold at 480p (Track-A):

```bash
source tools/h3_nfe_product_ladder_env.sh euler8
source tools/h3_raise_canvas_sku_env.sh 480p
# generate … --width 864 --height 480 --steps 8 --cfg 1 --sdpa-backend cudnn
```

### 5.2 RC1 — 1152 DMAD speed-face

```bash
source tools/h3_nfe_product_ladder_env.sh dmad4
source tools/h3_raise_canvas_sku_env.sh 1152_dmad
# generate … --width 1152 --height 640 --duration 5 \
#   --fewstep-preset dmad4 --lora "$H3_DMAD_LORA" --lora-scale 1 \
#   --cfg 1 --sdpa-backend cudnn --skip-audio
```

### 5.3 RC2 — 1344 Euler max-face

```bash
source tools/h3_nfe_product_ladder_env.sh euler8
source tools/h3_raise_canvas_sku_env.sh 1344_euler
# generate … --width 1344 --height 768 --duration 5 \
#   --steps 8 --cfg 1 --sdpa-backend cudnn --skip-audio
# optional: export H3_DIT_INT8_WINDOW=24
```

### 5.4 Helper refusals (expected)

```bash
source tools/h3_raise_canvas_sku_env.sh 1280x720     # → return 2: %32 geometry
source tools/h3_raise_canvas_sku_env.sh 1344_pdmd    # → refuse: 1344+PDMD FAIL noise
```

## 6. Evidence this turn

| Claim | Status | Source |
|---|---|---|
| 1152×640 fewstep usable | **Prior PASS** | ghost A/B @`47b4cd8` (PDMD-4; DMAD same ceiling by product policy) |
| 1344×768 Euler-8 best face | **Prior PASS** | same ledger · ~142.9s warm e2e |
| 1344+PDMD FAIL noise | **Prior FAIL** | window≈6 · labeled `_FAIL_NOISE` |
| Fresh GPU smoke | **Skipped / reuse** | No new claim; clips copied from `ghost_face_ab_20261027` |
| FaceRefine / C035 as SKU | **Not claimed** | STOPPED / G2 BLOCK |

Labeled share (this packaging): `~/grokbot-share/raise_canvas_sku_20261028/labeled/` (copies of prior labeled mp4s).

## 7. What is NOT claimed

- Flipping product SDPA default to sol.  
- Selling 1344 fewstep on 4090 INT8.  
- Reopening FaceRefine, NVFP4, C011/C016/C043/C018.  
- Inventing detail/face LoRA weights.  
- That S2 / window24 / C015 alone fixes far-face.

## 8. Related

- Helper: `tools/h3_raise_canvas_sku_env.sh`  
- [`nfe-fewstep-contracts`](2026-10-28-nfe-fewstep-contracts.md)  
- [`track-b-s2-sellable-recipe`](2026-10-28-track-b-s2-sellable-recipe.md)  
- [`4090-opt-menu-beyond-dmad`](2026-10-28-4090-opt-menu-beyond-dmad.md)  
- [`ghosting-and-small-face`](2026-10-27-ghosting-and-small-face.md)  
- [`face-lora-vs-facerefine`](2026-10-28-face-lora-vs-facerefine.md)
