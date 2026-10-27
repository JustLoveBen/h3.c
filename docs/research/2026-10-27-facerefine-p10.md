# FaceRefine on p10 distant-face mush (2026-10-27)

> Asia/Singapore. Gate remains 480p×5s INT8 labels. Scene B = `p10_street_wave_dolly`.

## RCA (confirmed)

Distant-face collapse is **head-size-in-frame pixel budget**, not Ref2VA/RefMod.

| frame | raw face area (YuNet) | ~side px |
| --- | ---: | ---: |
| f000 (wide) | ≈1176 | ≈34 |
| f119 (MCU after dolly) | ≈30015 | ≈173 |

864×480 → VAE/16 → few DiT patches on a ~34px face → features unrecoverable. Raising canvas helps globally; FaceRefine raises **face** pixels only.

## Minimal path landed

`tools/face_refine_crop_h3_stitch.py` + `tools/run_face_refine_p10_4090.sh`:

1. **detect** — YuNet track, pad square, export `first_frame_face.jpg` @544×544 + `track.json`
2. **H3 F2VA** — `--first-frame` + face-CU ids, PDMD-4, 544×544×5s (**do not** `--skip-audio`; F2VA requires audio pack)
3. **stitch** — feathered elliptical paste of refined face video onto base PDMD clip
4. **classical** — Lanczos+unsharp enhance baseline (no generative)

## A/B products

`out/facerefine_p10_20261027/labeled/` and `~/grokbot-share/facerefine_p10_20261027/labeled/`:

- BEFORE PDMD-4
- AFTER FaceEnhance classical
- AFTER FaceRefine H3-PDMD4 (warm ≈76.5s, dit_ms≈13194)
- face-CU 544 alone

## Gaps

- No CodeFormer/GFPGAN weights this slice
- H3 path is **F2VA regen**, not pixel-locked img2img denoise → temporal identity can drift vs body
- Single-subject tracker only

## Grok Video @480p vs H3 p10 (doc scan)

No in-repo `docs/` comparison naming Grok Video / xAI / Aurora face-budget behavior was found this turn. Public RCA for parent: likely different training mix, face-aware objectives / latent upscalers / post FaceDetailer in product stack, or prompts that keep faces larger — **not** evidence that 480p physics differs. H3 INT8 gold is CFG-distilled few-step T2V without a FaceRefine stage; product Grok Video may not be the same pipeline.

## YuNet weights (not committed)

OpenCV zoo ONNX expected at `models/face_detect/face_detection_yunet_2023mar.onnx` on the 4090 tree (or `$H3_YUNET`). Fetch from OpenCV zoo if missing; runner `test -f` fails closed.


## Steps A/B (pt69u — is NFE the limit?)

Follow-up on same p10 BEFORE base (`tools/run_face_refine_euler_ab_4090.sh`):

| face-CU arm | sampler | NFE | LoRA |
| --- | --- | ---: | --- |
| prior PDMD-4 | velocity_euler + fewstep | 4 | pdmd_4NFE |
| Euler-8 | velocity_euler gold | 8 | none |
| Euler-16 | velocity_euler | 16 | none |

All still **official F2VA** (`--first-frame` COND → full regen), then elliptical stitch @864×480.

### Why not Comfy img2img denoise yet

`h3_cuda_generate` official image path is F2VA/L2VA/Ref2VA condition packs. `--init-latent` + `--init-first-frame` is documented **G3-lite debug**, not a per-frame crop denoise (no denoise-strength / start-sigma API for FaceDetailer-style 0.35–0.45×8). Building true crop-locked img2img needs encode(crop frames)→partial denoise→decode→stitch outside current F2VA contract.


### Steps A/B results (4090, Asia/Singapore 2026-10-27 ~23:56)

| arm | warm WALL | dit_ms | NFE |
| --- | ---: | ---: | ---: |
| PDMD-4 F2VA face-CU | 76.52s | 13193.6 | 4 |
| Euler-8 F2VA face-CU | 82.60s | 20763.4 | 8 |
| Euler-16 F2VA face-CU | 95.64s | 41273.2 | 16 |

**Verdict vs pt69u (“步数太少?”): NFE is not the main limit.**

Hard evidence on same BEFORE base:

1. **Wide / early (t≈0.5, face ~34px):** BEFORE mush; PDMD4 stitch shows sharp elliptical paste (teeth/eyes) but halo. Euler-8/16 stitches do **not** look clearly sharper in-frame — once the 544 CU is downscaled into the ~34px slot, HF detail is gone regardless of 8 vs 16 NFE.
2. **MCU (t≈4.5):** PDMD4 stitch stays roughly head-locked. Euler-8/16 show **worse** framing drift (“bobblehead” oversized circular face vs body) — F2VA regen re-composes the face inside 544²; stitch still assumes track bbox = face canvas. More steps amplified identity/framing freedom, not pixel lock.
3. **Architecture > steps:** Official F2VA = COND + full regen from noise. No denoise-strength / start-σ API for Comfy-style 0.35–0.45 img2img on the crop. `--init-latent`+`--init-first-frame` is G3-lite debug, not product FaceDetailer.

Next quality lever (not done this slice): true crop-locked partial denoise, or at least warp/align face-CU to track bbox before stitch; raising base canvas still helps global pixel budget.
