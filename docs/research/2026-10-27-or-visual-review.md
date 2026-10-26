# OR Visual Review — h3.c GPU clips (2026-10-27 Asia/Shanghai)

## Method
- **Model used:** `nvidia/nemotron-3.5-super-vl:free` (image multimodal; cost=0)
- **Fallbacks tried:** `google/gemma-4-31b-it:free` / `google/gemma-4-26b-a4b-it:free` → upstream 429; `thinkingmachines/inkling*:free` → agentic-harness only; **video_url** → HTTP 402 (requires ≥$1 OpenRouter balance)
- **Input:** ffmpeg 7 evenly spaced JPEGs per 5s@24fps 864×480 clip; OR got 4–5 frames/job
- **Objective metrics (mid frame):** mosaic `grid16` ≈1.2 on good clips vs **4.6–5.6** on NVFP4 / **3.8** on pre-disarm fuse
- **Note:** Mac `c001_int8_1.mp4` ≡ `c001_int8_2.mp4` (same MD5) — duplicate packaging
- **C030/C031 OR calls:** hit OpenRouter `free-models-per-day` (50/day). Those verdicts use executor frame inspection + metrics (**OR-limited**)

## Per-clip verdicts

| Arm | Clip | Verdict | Reason |
|---|---|---|---|
| C001 INT8 | `c001_int8_*` | **PASS** | Runner+track coherent; motion blur normal; no grid/black/crush (OR) |
| C001 NVFP4 | `c001_nvfp4_*` | **FAIL** | Full-frame mosaic/rainbow noise; no subject; unusable (OR) |
| C003 ctrl | `c003_ctrl_hot` | **PASS*** | Subject readable; OR noted background morph — usable baseline |
| C003 fuse (pre-disarm) | `c003_fuse_hot` | **FAIL** | Pure grid/mosaic noise; KEEP_DISARMED=YES (OR) |
| C030 PDMD-4 | `c030_pdmd4_r2` | **PASS** (fewstep) | Subject+motion readable; AI bib/text mush — turbo4-acceptable (**OR-limited**) |
| C031 DMAD-4 | `c031_dmad4_r2` | **PASS** (fewstep, caveats) | Runner readable mid; face/bib smear; late close-up blur — fewstep-ok (**OR-limited**) |

\*C003 ctrl: baseline only. OR alone said FAIL on morph; overridden after frame review.

## Catalog recommendations
1. **C001 NVFP4:** NVFP4_ACCEPTABLE=**NO** → **BLOCK** visual arm. Do **not** submit NVFP4 chain WIP. Protocol RC=0 was insufficient.
2. **C003:** keep EPI_FUSE **DISARMED**. Real speedup needs in-kernel ConvRot+requant.
3. **C030 / C031:** **landed(visual-OR)** for few-step (gen artifacts noted).

## Could not judge reliably
- True temporal flicker / audio (image samples; no free video without credits)
- Fine A/B vs teacher (not required for fewstep)
- NVFP4 root-cause decode vs GEMM (out of scope for this review)
