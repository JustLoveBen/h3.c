# GPU unified test results (238 · 2026-10-27 Asia/Shanghai)

Host: `yongbin@192.168.0.238` (hostname `digen`, RTX 5090 sm_120).  
Tree: `/mnt/nvme1n1/workspace/h3_c_main_20261027` (clean sync; **no** NVFP4 chain WIP).  
Git SHA synced: `abd2194` (`abd219447f655a5e669cc2612f62ebecbbe3d7ec`).  
Binary: `build-cuda-sm120/h3_cuda_generate`  
`sha256: 09204928302893e5ceb05966dfa6567e567f264faf344910e51c5efd36e485ff`  
Build: `cmake -S . -B build-cuda-sm120 -DH3_GPU_BACKEND=cuda` + system `/usr/bin/g++`, `H3_CUTLASS_DIR=/home/yongbin/deps/cutlass`, `H3_ENABLE_NVFP4_CHAIN=ON` (link only; **runtime chain unset**). BUILD_RC=0 after C062 scope fix landed in `abd2194`.

Disk at end: `/mnt/nvme1n1` ~99% / ~58G free (no wipe). ComfyUI was stopped before runs; VRAM free between items.

Mac pulls:

- `out/p0_visual_20261027_from238/`
- `out/c003_epi_fuse_20261027_from238/`
- `out/c030_pdmd4_20261027_from238/`

Converter added: `tools/convert_pdmd_lora_h3repo.py` (2NFE round-trip maxdiff 0.0 → then 4NFE).

---

## C001 — P0 visual gate (`docs/research/2026-10-26-p0-visual-gate-runbook.md`)

Out (238): `out/p0_visual_20261027/{int8_1,int8_2,nvfp4_1,nvfp4_2}`  
Config: 864×480, 5s, 8 steps, cfg1, seed `0x76800002`, cudnn, skip-audio, live TE, graph unset, **chain unset**. NVFP4 arms: `H3_DIT_NVFP4=1` + sidecar `minimax_h3_fl2va_nvfp4_w4a4.safetensors`.

| arm | RUN_RC | PROTO | phase_ms dit | total_ms | notes |
| --- | ---: | --- | ---: | ---: | --- |
| int8_1 | 0 | PROTO_OK | 31440.2 | 57771.6 | cold TE |
| int8_2 | 0 | PROTO_OK | 23967.9 | 34493.1 | |
| nvfp4_1 | 0 | PROTO_OK (WARN: log still prints `dit_format: INT8` + `dit_nvfp4:` banner) | 22709.8 | 32159.9 | |
| nvfp4_2 | 0 | PROTO_OK (same WARN) | 22268.9 | 31815.2 | |

Waffle same-config (logs):

- INT8 `int8_1` vs `int8_2`: **PASS** mean Δ=0.000 max=0.000 (all frames `INVALID_LOWENERGY` — gate “门失效” note; b16 meaningless)
- NVFP4 `nvfp4_1` vs `nvfp4_2`: **FAIL** mean |Δb16|=12.16 max=16.15 (also INVALID_LOWENERGY)

**Human visual review still required** for both INT8 and NVFP4 mp4s (protocol ≠ quality).

---

## C003 — E2 `H3_DIT_EPI_FUSE` A/B

Out (238): `out/c003_epi_fuse_20261027/`  
Binary/out = new tree. turbo4 + LoRA, 864×480, 5s, seed `0x48000001`, `--repeat 2`, alternating ctrl/fuse ×2.

Hot clip2 `phase_ms` **dit**:

| arm | hot dit_ms |
| --- | ---: |
| ctrl_r1 | 13957.8 |
| fuse_r1 | 12759.5 |
| ctrl_r2 | 13886.5 |
| fuse_r2 | 12741.3 |

- ctrl mean 13922.1 · fuse mean 12750.4 · **~8.4% dit speedup**
- same-config waffle: ctrl PASS (Δ~0.001); fuse PASS but **b16=0.000 all frames**
- mux sizes: ctrl ~2.3–3.6 MB; **fuse ~100 KB** → fuse arm looks visually collapsed / near-empty
- fuse log smoking gun: `latent_stats: mean=nan std=0 min=nan max=nan` (both clips); ctrl audio peak ~0.26–0.39 vs fuse peak=1.0 (clipped)

**Verdict:** ~8.4% dit speedup is real, but **EPI_FUSE corrupts latents to NaN — treat as FAIL for quality, do not enable by default**. Root-cause next (epi_fuse path), not nsys/EVT first.

---

## C030 — PDMD-4 (`--fewstep-preset pdmd4`)

Asset path:

1. Downloaded original `pdmd2026/pdmd_4NFE_lora` `lora_model_0.safetensors` (rank 128, 624 keys) → `/mnt/nvme1n1/workspace/models/loras/_pdmd4_dl/`
2. Converted with `tools/convert_pdmd_lora_h3repo.py` (rules verified exact vs existing 2NFE h3repo) →  
   `/mnt/nvme1n1/workspace/models/loras/pdmd_4NFE_lora_model_0.h3repo.safetensors` (1.9 G, bf16, `h3_load_refiner=1`)
3. **Did not** use Kijai `minimax_h3_pdmd_4step_lora_avg_rank_57_bf16.safetensors` (truncated, not gold)

Run: INT8 base, `--fewstep-preset pdmd4` (steps=4, Euler, video_shift=12, audio_shift=3, cfg=1), 864×480, seed `0x76800002`, live TE, cudnn, chain unset. Out: `out/c030_pdmd4_20261027/{r1,r2}`

| arm | RUN_RC | PROTO (`--arm fewstep --expect-fewstep pdmd4`) | dit_ms | total_ms |
| --- | ---: | --- | ---: | ---: |
| r1 | 0 | PROTO_OK (cold TE MISS warn) | 16538.1 | 28248.2 |
| r2 | 0 | PROTO_OK | 15349.5 | 24430.7 |

Waffle r1 vs r2: **PASS** mean Δ=0.014 max=0.027 (frames still flagged INVALID_LOWENERGY).

Contract lines present: `fewstep_preset: id=pdmd4 … sampler=velocity_euler`, `sigma_ok student_grid(nfe=4, video_shift=12, audio_shift=3)`, `av_sync … PASS`.

**Human visual review required** (protocol/waffle ≠ aesthetic gold).

---


---

## Addendum — C003 scratchfix + C031 DMAD (238 · same day, SHA `1b37f28`)

Binary: `build-cuda-sm120/h3_cuda_generate`  
`sha256: 2cd0d12e17c7a1685b26d86b8e9c12d0bd11089d335ab11e5acf7746f17b7e64`  
Mac pulls: `out/c003_epi_fuse_20261027_scratchfix_from238/`, `out/c031_dmad4_20261027_from238/`  
Disk end: still ~99% / ~58G free; VRAM idle after runs. Chain WIP untouched.

### C003 retest (NaN fix)

See [`2026-10-27-c003-epi-fuse-nan-fix.md`](2026-10-27-c003-epi-fuse-nan-fix.md). Summary:

| metric | ctrl hot | fuse hot | note |
| --- | ---: | ---: | --- |
| dit_ms mean (r1/r2) | 13848.6 | 13436.7 | **~3.0%** dit (not prior 8.4%) |
| latent finite | yes | **yes** | NaN fixed |
| latent checksum r1=r2 | yes | yes | |
| mux size | ~3.8 MB | ~6.0 MB | no longer ~100 KB |
| waffle same-config | PASS Δ=0 | **FAIL** meanΔ=0.39 | keep default OFF |

**Verdict:** NaN fixed; quality/determinism **not** green — **do not default-on**.

### C031 — DMAD-4 GPU accept (`lcm_renoise`)

Asset (original non-truncated, mirror C030 discipline):

- `/mnt/nvme1n1/workspace/models/loras/dmad_minimax_h3_4step_full_critic.h3repo.safetensors` (1 880 927 608 B)
- `--fewstep-preset dmad4` → contract `dmad4`, sampler **`lcm_renoise`** (Euler ≠ pass), video_shift=12, audio_shift=2, cfg=1
- INT8 ConvRot base, 864×480, 5s, seed `0x76800003`, cudnn, chain unset
- Out 238: `out/c031_dmad4_20261027/{r1,r2}`

| arm | RUN_RC | PROTO | dit_ms | total_ms (report) | mux |
| --- | ---: | --- | ---: | ---: | ---: |
| r1 | 0 | PROTO_OK (cold TE MISS warn) | 18001.4 | 29854.6 | 1.74 MB |
| r2 | 0 | PROTO_OK | 15409.1 | (see log) | 1.68 MB |

- latent_stats identical r1/r2: mean=0.03177 checksum=`0x4c6cb54664fa9586`
- waffle same-config (generate.log): **PASS** mean|Δb16|=0.053 max=0.147 (WAFFLE_RC=0); frames still `INVALID_LOWENERGY`
- log confirms `sampler: selected=lcm_renoise` and `fewstep_contract: id=dmad4 … sampler=lcm+re-noise`

**Protocol verdict:** PASS. **Human visual still required** (protocol ≠ quality).



### C062 — te_cache HIT hard-gate smoke (optional same batch)

Reuse C031 warm cache `out/c031_dmad4_20261027/te_cache/` (key=`a150f97b2a58968e`).  
`H3_TE_CACHE_REQUIRE_HIT=1` + `--te-cache` + protocol `--expect-te-cache-hit`. Same dmad4/INT8/480p/seed as C031.

| metric | value |
| --- | --- |
| RUN_RC | 0 |
| te_prompt_cache | **HIT** require_hit=1 |
| PROTO (`--expect-te-cache-hit`) | **PROTO_OK** |
| te_ms | 294.2 (C031 cold was ~3566) |
| dit_ms | 15320.4 |
| latent checksum | `0x4c6cb54664fa9586` (= C031) |

Mac pull: `out/c062_te_cache_hit_20261027_from238/`. Default te_cache policy remains **off**.

## Still needs human eyes (updated)

1. C001 INT8 / NVFP4 mp4s (`out/p0_visual_20261027_from238/`) — NVFP4 waffle FAIL + both INVALID_LOWENERGY
2. C003 fuse vs ctrl hot mp4s (`out/c003_epi_fuse_20261027_scratchfix_from238/fuse_*/*_r2.mp4`) — finite but waffle FAIL / latent diverge
3. C030 PDMD-4 mp4s (`out/c030_pdmd4_20261027_from238/`)
4. C031 DMAD-4 mp4s (`out/c031_dmad4_20261027_from238/`)

## Not done / out of scope this pass

- NVFP4 chain WIP untouched (local dirty on Mac; not synced / not committed)
- No nsys/EVT; no disk wipe; ComfyUI / eval-matrix pause flags left alone
- C040 VRAM ledger not run (disk ~99%; skipped). C062 HIT smoke **done** (see above).
