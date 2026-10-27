# C015 default-on decision — PASS → default ON

> **Asia/Shanghai 2026-10-28.** Host `digen-MS-7C80` / RTX 4090 sm_89.  
> Binary under test: synced SHA `84e75b4` (C015 built OFF + prior waffle/E2).  
> Dig out: `out/c015_default_decision_dig_480p_20261028/` (`tools/run_c015_default_decision_dig_4090.sh`).

## Claim

Decide whether `H3_DIT_ACT_FP16_STAGE` may become **product default ON**.  
**No silent flip** — only after multi-prompt waffle PASS + warm WC no regression; Euler peak drop ≥0.5 GiB is the VRAM gate; DMAD LoRA (qkv-only, wide stays F32) is informational.

## Arms

| scheme | prompts | steps | LoRA | stage A/B |
|---|---|---|---|---|
| Euler-8 INT8 | p02, p06, p10 | 8 | none | unset vs `=1` |
| DMAD-4 | p02, p10 | 4 | dmad full-critic | unset vs `=1` |

480p 864×480×5s, seed `0x76800002`, `--repeat 2`, cudnn SDPA, fused_epi default, no NVFP4 / FaceRefine / KEEP=attn.

## PASS table (raw)

| criterion | result | evidence |
|---|---|---|
| multi-prompt waffle all PASS | **PASS** | 5/5 pairs; mean\|Δ\| ≤0.0416, max\|Δ\| ≤0.109 (tol 0.15 / 0.40) |
| warm dit no material regression | **PASS** | all Δdit_ms ≤0 (slightly faster ON) |
| Euler no-LoRA peak_dit drop ≥0.5 GiB | **PASS** | drops 1.408 / 1.406 / 1.412 GiB |
| DMAD LoRA peak drop ≥0.25 GiB (informational) | **soft FAIL** | cold max-peak −0.12 GiB (wide stays F32); warm peak still ~−0.6 GiB |
| **recommend default ON** | **YES** | primary gates green; DMAD soft note only |

Full table: [`out/.../DECISION_REPORT.md`](../../out/c015_default_decision_dig_480p_20261028/DECISION_REPORT.md).

## Decision

**YES — flip default ON.**

- Opt-out: `H3_DIT_ACT_FP16_STAGE=0`.
- Pack still refuses `nvfp4` / `uly_u>1` / `!fused_epi` (unchanged).
- LoRA path: qkv F16 only; d_wide stays F32 (by design) — do not expect Euler-class VRAM drop under DMAD.
- **Parent/user policy:** user authorized proceed on rational judgment if multi-prompt waffle PASS and no WC regression. Both held; commit flips default with this note. Revert with env `=0` or revert commit if product policy disagrees.

## Code change (this commit)

- `dit_fp16_stage_env()`: unset → ON; `=0` → OFF (same pattern as `H3_DIT_ACT_F16`).
- `h3_env_contract.c` help string updated.
- Tools: `run_c015_default_decision_dig_4090.sh`, `c015_default_decision_aggregate.py`.

## Next (Track-A)

C015 ship-complete for near-lossless default path. Remaining DeepSeek-derived Track-A items are hygiene / defer (C012 idea, no reopen C018/C019/C003-SL/C016/C043). Pick next from catalog / product wall-clock (NFE/Track-B) — not another C015 dig.
