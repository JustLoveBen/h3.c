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
