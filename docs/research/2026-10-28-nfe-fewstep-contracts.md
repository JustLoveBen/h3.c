# NFE / few-step product contracts (ladder)

> **Asia/Shanghai 2026-10-28.** Product packaging after Track-A C015 default-ON (`b99d56b`) + Track-B S2 (`3e4414b`).  
> **NFE axis packaged @`7cc2a0f`.** Raise-canvas productized in [`raise-canvas-sku`](2026-10-28-raise-canvas-sku.md) + `tools/h3_raise_canvas_sku_env.sh` (policy sketch: [`raise-canvas-sku-policy`](2026-10-28-raise-canvas-sku-policy.md)).  
> **Policy (t71u / memory):** primary eval/product arms = **DMAD or Euler**; **PDMD = contrast-only**. FaceRefine STOPPED. NVFP4 BLOCK. No inventing LoRA not on disk (G2 C035 BLOCK; no pdmd2 weights on 4090).

## 0. One-liner

**Wall-clock axis after kernel Track-A/B is still NFE.** Ship two defaults: **Euler-8** (Track-A near-lossless fidelity) and **DMAD-4** (Track-B speed fewstep). Do not sell PDMD-2 / turbo4 / MagCache / FaceRefine as product steps. Raise-canvas is the next quality SKU axis — not this doc.

## 1. Inventory (code + docs @ `3e4414b`+)

| Artifact | Path / knob | What it freezes |
|---|---|---|
| Contract enum | `h3_fewstep_contract.h` (`pdmd4`/`pdmd2`/`dmad4`) | steps, sampler, shifts, cfg, LCM flag |
| Machine-readable table | `tests/fixtures/fewstep_contracts.json` (+ schema) | same as C dump |
| Dump / help | `tools/dump_fewstep_contracts.c`, `tools/print_fewstep_help.c` | host JSON / CLI help |
| Validators | `tools/fewstep_contracts_validate.py`, `fewstep_forbidden_combo_check.py` | offline; forbid turbo4×fewstep |
| CLI | `--fewstep-preset` / `--fewstep-contract` · `H3_FEWSTEP_PRESET` / `H3_FEWSTEP_CONTRACT` | expands + self-check |
| Strict / NVFP4 cross | `H3_FEWSTEP_STRICT`, `H3_FEWSTEP_ALLOW_NVFP4` (default reject) | pdmd2 audio∈{3,6}; NVFP4 off |
| Student σ grid | `h3_fewstep_student_grid_fill` + golden header | schedule vector check |
| KPI runner | `tools/run_dmad4_vs_euler8_kpi_4090.sh` | 480p×5s p10 DMAD-4 vs Euler-8 |
| Prior notes | [`fewstep-preset`](2026-10-26-fewstep-preset.md), [`fewstep-contracts-json`](2026-10-26-fewstep-contracts-json.md), [`dmad4-vs-euler8-kpi`](2026-10-28-dmad4-vs-euler8-kpi-480p.md), [`beyond-dmad`](2026-10-27-beyond-dmad-and-lossless.md), [`opt-menu`](2026-10-28-4090-opt-menu-beyond-dmad.md) | history |
| Weights on 4090 (`/home/haiyang/workspace/models/loras`) | `dmad_minimax_h3_4step_full_critic.h3repo.safetensors`, `pdmd_4NFE_lora_model_0.h3repo.safetensors` | **no** pdmd2 / TaoMate / detail-refiner |

Frozen contract rows (fixture):

| id | NFE/steps | sampler | video/audio shift | cfg | LoRA required |
|---|---:|---|---|---:|---|
| `pdmd4` | 4 | `velocity_euler` | 12 / 3 | 1 | yes |
| `pdmd2` | 2 | `velocity_euler` | 12 / 3 (alt 6) | 1 | yes |
| `dmad4` | 4 | `lcm_renoise` | 12 / **2** | 1 | yes |

**Forbidden speed presets with fewstep:** `turbo4`, `taomate`, `fast4` (rewrite σ → contract FAIL).

**Not a fewstep contract:** Euler-8 (steps=8, `velocity_euler`, no LoRA) — Track-A quality gold / waffle baseline.

## 2. Product ladder (Track-A vs Track-B)

Keep ranking and gates **separate**. Same-config waffle / BIT_ID only for Track-A near-lossless engine knobs — **never** cross-arm waffle between Euler↔DMAD↔PDMD.

| rung | arm | NFE | Track | Product default? | How to invoke | Warm e2e (480p×5s) | Quality note |
|---|---|---:|---|---|---|---|---|
| **L0** | **Euler-8** INT8 cudnn | 8 | **A** near-lossless fidelity | **YES** (quality / waffle gold) | `--steps 8` (no `--fewstep-*`, no LoRA) | **~35.8 s** warm @p10 (`3cee9ff`) | Highest fidelity; NFE dominates WC; use when quality > latency |
| **L1** | **DMAD-4** INT8 cudnn | 4 | **B** lossy-acceptable speed | **YES** (speed fewstep) | `--fewstep-preset dmad4 --lora <full_critic.h3repo> --lora-scale 1` | **~25.0 s** warm @p10 (**1.43×** vs Euler-8; dit **1.61×**) | Contract-bound `lcm_renoise` 12/2; visual PASS complex 480p; smear possible vs Euler |
| **L2** | **PDMD-4** INT8 | 4 | **B** contrast-only | **NO** (opt-in contrast) | `--fewstep-preset pdmd4 --lora <pdmd_4NFE…>` | ~similar 4-NFE band (historical) | Paper/OR often ≥ DMAD at same NFE; **not** product primary after t71u |
| **L3** | Track-B **S2** on top of L1 | 4 | **B** opt-in WC | **NO** | `source tools/h3_track_b_s2_env.sh` + `--sdpa-backend sol_cute` + dmad4 | dit **−15%** vs cudnn @G1 (`eb805b4`) | Needs window=24; cudnn default unchanged |
| — | **PDMD-2** | 2 | B speculative | **NO — not sellable** | contract exists (`pdmd2`) | — | **BLOCK ship:** no 2NFE LoRA on 4090; inherent σ-jump blur (C033 boundary) |
| — | turbo4 / MagCache / UniPC as default | ≤4 | — | **NO** | — | — | Forbidden or idea-only; wrong σ / unvalidated |
| — | C035 detail/refiner LoRA | 4 | B polish | **NO** | — | — | **G2 BLOCK** — no TaoMate/Kijai weights on disk |

### Track-A gate (near-lossless)

- Same-config **waffle / BIT_ID** vs Euler-8 INT8 ctrl (engine knobs only: C015, window, etc.).
- Do **not** use DMAD/PDMD pixels as Track-A reference.

### Track-B gate (lossy-acceptable fewstep)

- `--fewstep-contract` / preset **early_ok** + `fewstep_bound` + sigma_ok.
- Complex **480p×5s** visual PASS (motion+expression+camera); burn-in scheme + warm e2e.
- Optional VL rubric; **no** mosaic/energy-collapse; **no** FaceRefine stitch to “fix” quality.
- Floor vs Euler-8: may smear / oversaturate; must remain recognizable talking-head / complex action — not a waffle claim.

## 3. Evidence this turn

| Claim | Status | Source |
|---|---|---|
| DMAD-4 vs Euler-8 WC | **Measured** | [`dmad4-vs-euler8-kpi`](2026-10-28-dmad4-vs-euler8-kpi-480p.md) @`3cee9ff` · warm 25.0 vs 35.8 s |
| Complex 480p visual PASS (DMAD/Euler/PDMD) | **Prior PASS** | complex_480p_20261027 / ghost_face labeled shares |
| New fewer-than-4 sellable arm | **Not claimed** | pdmd2 LoRA **absent** on 4090 models/loras (2026-10-28 inventory) |
| Fresh GPU A/B this turn | **Skipped** | 4090 reachable & idle via Mac LAN, but **no new claimable arm**; reuse KPI rather than reburn wall-clock |

Labeled clips (share): `~/grokbot-share/nfe_fewstep_contracts_20261028/labeled/`.

## 4. Product entry (preferred)

```bash
# One-liner — no env soup
tools/h3_product_generate.sh --sku rc0_euler --prompt-ids evals/768p/ids/p10_street_wave_dolly.ids --out out/rc0_euler
tools/h3_product_generate.sh --sku rc0 --prompt-ids … --out out/rc0_dmad   # DMAD-4 speed
tools/h3_product_generate.sh --nfe dmad4 --canvas 480 --dry-run
tools/h3_product_generate_smoke.sh
```

Doc: [`product-generate-entry`](2026-10-28-product-generate-entry.md). Env helpers below remain valid for digs.

## 4b. Helper script (source-only)


```bash
# Track-A quality gold (prints recommended flags; does not invent LoRA)
source tools/h3_nfe_product_ladder_env.sh euler8

# Track-B speed fewstep (sets H3_FEWSTEP_PRESET=dmad4; still need --lora path)
source tools/h3_nfe_product_ladder_env.sh dmad4

# Contrast-only
source tools/h3_nfe_product_ladder_env.sh pdmd4
```

Does **not** export `H3_CUDA_SDPA_BACKEND` (cudnn stays product default). Compose with S2 only explicitly: `source tools/h3_track_b_s2_env.sh` after `dmad4`.

## 5. What is NOT claimed

- PDMD-2 / any 2-NFE product SKU without on-disk LoRA + new Track-B visual floor.  
- Flipping product primary back to PDMD (historical beyond-DMAD ranking superseded by t71u).  
- ~~Raise-canvas productization~~ → [`raise-canvas-sku`](2026-10-28-raise-canvas-sku.md) (done after this NFE doc).  
- C035 refiner, FaceRefine, NVFP4, MagCache, turbo4×fewstep.  
- Reopening C011 / C016 / C043 / C018.

## 6. Sequential axes

1. ~~NFE / few-step contracts~~ **this doc** @`7cc2a0f`  
2. ~~Raise-canvas SKU productization~~ → [`raise-canvas-sku`](2026-10-28-raise-canvas-sku.md) + `tools/h3_raise_canvas_sku_env.sh`

## 7. Related

- [`raise-canvas-sku`](2026-10-28-raise-canvas-sku.md) + `tools/h3_raise_canvas_sku_env.sh`  
- [`track-b-s2-sellable-recipe`](2026-10-28-track-b-s2-sellable-recipe.md)  
- [`accel-quality-theory-and-industry`](2026-10-28-accel-quality-theory-and-industry.md) § NFE still WC axis  
- [`4090-opt-menu-beyond-dmad`](2026-10-28-4090-opt-menu-beyond-dmad.md)
