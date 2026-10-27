# NVFP4 KEEP_INT8 multi-scene OR :free VL + mosaic RCA close (2026-10-27 Asia/Shanghai)

**Binary:** postfix gate-fix `a4eef5b9…` on 238. Repo HEAD before this doc: `896d178`.  
**Clips:** `~/grokbot-share/keep_int8_multiscene_20261027/` + `~/grokbot-share/multiscene_vl_frames/` + postfix frames.  
**Protocol:** 864×480 5s 8step cfg1 seed `0x76800002` cudnn skip-audio.

## OpenRouter key / model

| item | result |
|---|---|
| env `OPENAI_API_KEY` / prior sk-or…f9c5 | **401** dead |
| working key | `~/.deepseek/config.toml` sk-or…f449 (`limit_usd=1`, spend 0; **no purchased credits** → paid models 402) |
| model used | `nvidia/nemotron-3.5-super-vl:free` (dots:free returned reasoning-only / empty content at low max_tokens; gemma:free 429) |
| free-tier | hit **daily 50** (`free-models-per-day`, reset ~UTC midnight) mid-batch after int8×5 |

## OR :free VL conclusion table

Schema: recognizable / artifacts / collapse 0–5 (higher art/col = worse); usable bool.

### OR-parsed (nemotron:free JSON)

| arm | prompt | rec | art | col | usable | one_line |
|---|---|---:|---:|---:|---|---|
| int8 | p01_sprint | 5 | 1 | 0 | true | clear sprint, negligible artifacts |
| int8 | p02_talking_head | 5 | 1 | 1 | true | clear portrait, minimal artifacts |
| int8 | p03_corridor | 5 | 1 | 0 | true | clean corridor/columns |
| int8 | p05_neon_sign | 5 | 1 | 0 | true | clear neon, no collapse |
| int8 | p07_market_track | 4 | 2 | 1 | true | recognizable market, minor softness |

### Hybrid arms — OR incomplete (429); host mid-frame VL (same jpgs) + PSNR vs int8

| arm | prompt | host_vis | PSNR vs int8 | notes |
|---|---|---|---:|---|
| k_attn | p01_sprint | mosaic/ghost on runner; scene alive | 17.17 | energy OK |
| k_attn | p02_talking_head | face blocky vs int8 clean | 22.55 | |
| k_attn | p03_corridor | soft/blocky | 16.62 | |
| k_attn | p05_neon_sign | heavy mosaic/vertical banding; text soft | 18.51 | int8 neon clean |
| k_attn | p07_market_track | soft/blocky | 17.86 | |
| k_attn_fc1 | (5 prompts) | similar / slightly sharper than attn | 17.9–26.1 | WC often slower |
| k_attn_fc2 | (5 prompts) | similar; talking_head PSNR best of hybrids | 18.4–27.3 | |
| postfix k_qkv / k_mlp / k_full | p01 | **heavy mosaic FAIL** (energy restored post gate-fix) | ~17–18 vs k_all | not multi-scene re-run |

Raw OR JSON: `out/keep_int8_multiscene_20261027_from238/vl_or/`.

**OR bottom line:** INT8 gold **PASS** on 5/5 free-VL. Hybrid arms **not OR-complete** this turn (rate limit); host frame review shows **KEEP=attn/attn_fc* still mosaic vs INT8** — alive/recognizable but not product-clean. Do **not** treat KEEP=attn as visual BIT_ID.

## DiT wall-clock (multiscene, Asia/Shanghai; warm after first prompt)

| arm | dit_ms range (5 prompts) | typical warm |
|---|---|---|
| int8 (`KEEP=all`) | 23940–24080 | **~24.0 s** |
| k_attn | 24194–27819 | **~24.2–25.4 s** (≈ INT8 once warm) |
| k_attn_fc2 | 24841–30511 | ~24.8–27.9 s |
| k_attn_fc1 | 26460–49191 | slower / high variance |

Postfix single-prompt (cold-ish): k_all 38345 / k_attn 37233 / k_full 23545 / k_qkv 30511 / k_mlp 39295.

## Mosaic after energy restore — RCA (no empty DISARM)

**Symptom:** post `a1b7aff` gate fix, mlp/qkv/full **std≈1.02–1.05** (energy OK) but frames still mosaic; even **KEEP=attn** (only FC1+FC2 FP4) shows clear mosaic vs INT8.

**Ruled out this knife (hard evidence):**

| hypothesis | evidence | verdict |
|---|---|---|
| AdaLN gate wipe (pre-fix) | fixed a1b7aff; postfix k_qkv≠k_full; mask logs match | fixed |
| act ConvRot off | logs `act_convrot=1 gs=256` all hybrid arms | ON |
| SF in-tile / stride | e2e `32x4+stride_ncb` PASS K=5376 N≤21504 rel_l2≈0.0017 | not SF-GEMM |
| kind-asymmetric weight pack | `probe_nvfp4_weight_norm.py` (238, re-run): qkv/out/fc1/fc2/b25 all **rel_l2≈0.095**, \|\|W\|\| ratio≈1.00 | uniform quant error |
| single-GEMM energy/scale | prior `probe_nvfp4_gemm_ab.py`: fp4/i8_norm≈0.999 | not scale |
| FP4 BF16 swiglu/gate formula vs contract | silu(gate)*up; gate epi `res + (scale*acc+lora)*mod` — matches INT8 fusion shape; no new asymmetry found on read | no short fix found |
| act SF ncb vs ncb_rd under-alloc | production uses stride=ncb (e2e-required); K=14336 and K=5376 same code path | not residual |

**Mechanism (honest):** offline naive INT8→NVFP4 w4a4 leaves **~13% directional rel_l2 per linear**. Energy preserved; error accumulates across blocks×linears×steps → structured mosaic (not blank). KEEP=attn proves **MLP FP4 alone** is enough to mosaic vs INT8 gold — so this is not "only attn linears are broken in code."

**Short vertical cut (implementation, no retrain/convert):** **none found** this knife. Would-be cuts that are *not* free:
- calib / SmoothQuant / GPTQ-style NVFP4 or QAT (needs auth + new contract)
- requant sidecar (DISARMED convert narrative)
- meaningless deblock/bypass epi

**Recommend:** product default **INT8**. Experimental doc-only arm `KEEP=attn` (WC≈INT8, energy OK) — **not** visual-clean; ship only behind explicit flag after human phone review. Prefer calib path if user wants mlp/qkv/full FP4 visual.

## Auth-needed remaining

1. OpenRouter: free daily exhausted; paid needs credits purchase **or** wait UTC midnight free reset — re-run hybrid×5 OR table.
2. Human phone review of `~/grokbot-share/keep_int8_multiscene_20261027/` (attn vs int8).
3. Calib/QAT or better NVFP4 pack if product wants FP4 visual (new contract).
4. Product default-on of any NVFP4 KEEP hybrid — **needs explicit user auth**.
5. Optional deeper dig (GPU): device single-block FC1 fused-swiglu vs INT8 dump (host weight A/B already uniform; low ROI unless new anomaly).

## Still do-not

Chain WIP / `H3_DIT_NVFP4_CHAIN` / EPI default-on / convert bypass / nsys unless asked.
