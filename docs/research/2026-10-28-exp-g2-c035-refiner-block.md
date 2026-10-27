# Exp-G2 · Track-B · C035 TaoMate/detail refiner — **BLOCK** (no weights)

**Date:** 2026-10-28 Asia/Shanghai  
**Host checked:** lan-4090-31 (`/home/haiyang/workspace/models`) + Mac `h3.c/models/loras`  
**Wiring SHA baseline:** tip before this note (C015 closed `84e75b4`; default OFF untouched)  
**Claim (theory §G Exp-G2):** 末 1–2 步 refiner LoRA (scale 0.5–0.7) @ DMAD-4 reduces few-step oversat/smear without global style collapse.

## Verdict: **BLOCK — weights absent**

| Check | Result |
|---|---|
| `H3_REFINER_LORA` / `STEPS` / `SCALE` in `tools/h3_cuda_generate.c` | **present** (tail-step switch) |
| TaoMate / Kijai rank-19 / detail refiner `.safetensors` under 4090 `models/` | **absent** |
| Mac `h3.c/models/loras/` | PDMD only (no taomate/refiner) |
| 4090 `models/loras/` | `dmad_*full_critic.h3repo` + `pdmd_4NFE_*.h3repo` only |
| Download authorized this turn? | **No** (policy: no weights → skip, do not download long-run) |

Searched: `*taomate*`, `*refiner*`, `*kijai*` under `/home/haiyang/workspace/models` (maxdepth 3) and `~/…/*taomate*.safetensors` — empty for LoRA weights. Comfy workflow JSON/clips referencing TaoMate exist under `comfyui_vace/` but are **not** h3.c product weights.

## What was *not* run

- No DMAD-4 / Euler-8 × STEPS∈{1,2} × scale∈{0.5,0.7} sweep  
- No labeled clips / quality-floor A/B  
- No FaceRefine reopen  

## Prior related evidence (not this gate)

- 2026-10-07 R-C on **238** (weights then present): Kijai rank-19 loaded; refiner tail waffle moved 1.321→1.207 (1-step) / 1.102 (2-step) — **human review pending**; not a 4090 product close.  
- HF path (for future restore, **not fetched**): `Kijai/MiniMax-H3_comfy` → `loras/minimax_h3_taomate_3step_lora_avg_rank_19_bf16.safetensors` (~182 MiB). Official full TaoMate with `token_refiner` is rejected by `h3_lora_parse_name`.

## Unblock recipe (when weights land)

1. Place Kijai rank-19 (or other parser-clean detail LoRA) at e.g. `models/loras/minimax_h3_taomate_3step_lora_avg_rank_19_bf16.safetensors`.  
2. Arms @480p×5s complex p10: base DMAD-4; +`H3_REFINER_LORA=… STEPS=1|2 SCALE=0.5|0.7`; optional Euler-8 polish arm.  
3. Labeled clips + explicit quality floor (oversat/smear↓; no identity/color disaster; subjective ≥ half of “half-canvas raise”).  
4. PASS → opt-in product doc; **FAIL → keep idea, never FaceRefine stitch**.

## Pivot (this turn)

Per parent: **no weights → document BLOCK and pivot Track-A C011** (VAE cast/prescale remainder). See [`c011-dig`](2026-10-28-c011-vae-cast-prescale-dig.md).  
**C015 default remains OFF** (`84e75b4` gate PASS; no flip).
