# Beyond DMAD + lossless acceleration (2026-10-27)

> Asia/Shanghai. Goal: rank **quality-preserving / bit-identical** speedups (Axes A/D) and **few-step alternatives better than DMAD** (Axis B); land one lossless code slice this batch; keep EPI_FUSE DISARMED.

## 1. Executive ranking

### 1.1 Beyond-DMAD (Axis B) — quality–speed Pareto

Evidence: paper Table 2 ([`pdmd-dmad-loras`](2026-10-09-pdmd-dmad-loras.md)); 238 GPU ([`gpu-unified`](2026-10-27-gpu-unified-test-results.md)); OR-limited visual ([`or-visual-review`](2026-10-27-or-visual-review.md)).

| Rank | Option | NFE | Hot dit_ms (238) | Quality signal | Status | Verdict |
| ---: | --- | ---: | ---: | --- | --- | --- |
| **1** | **PDMD-4** Euler 12/3 | 4 | ~15350 | Paper best auto among 4NFE; OR PASS fewstep | landed(visual-OR) C030 | **Preferred fewstep SKU** vs DMAD at same NFE |
| 2 | DMAD-4 `lcm_renoise` 12/2 | 4 | ~15409 | Paper below PDMD; OR PASS w/ smear | landed(visual-OR) C031 | Keep as alternate contract; not default over PDMD |
| 3 | turbo4 + detail/refiner LoRA (C035) | 4 | (turbo baseline) | Quality recovery on oversaturated turbo | idea | A/B only; not gold |
| 4 | 8-step INT8 gold + lossless stack | 8 | ~2× 4-step | Highest fidelity; NFE dominates wall clock | gold path | Use when quality > latency |
| 5 | Higher-order solvers UniPC/DPM++ (C036) | ≤8 | TBD | Training-free; hard to beat distill at 4NFE | idea | Observe; orthogonal to LoRA |
| 6 | SelfLift / VDN / TAE (C080) | SKU | — | Product tiers, not gold | idea | Separate SKU |
| — | PDMD-2 | 2 | faster | Inherent blur (σ jump) | boundary C033 | Contract regression only |
| — | NVFP4 4/8-step | any | faster/step | Mosaic FAIL | C001 BLOCK | No chain / no default |

**Pass criteria (fewstep):** `--fewstep-contract` early_ok + PROTO + same-config waffle PASS + visual (OR or human) without mosaic/grid. Wrong sampler/shift = FAIL (do not credit speed).

**Code this batch (B):** CLI help ranks `pdmd4` above `dmad4`; `tools/fewstep_pareto_report.py` prints the table from known 238 numbers. No new LoRA download.

### 1.2 Lossless / quality-preserving (Axes A/D)

Prefer **bit-identical** or **same-config waffle PASS vs INT8 ctrl**.

| Rank | ID | Slice | Lossless? | Est. WC | Status after this batch | Notes |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | **C017** | `H3_CUDA_L2_PIN=1` Hadamard AccessPolicyWindow | bit-id | −0.05–0.15s/clip (sm120) | **landed(opt-in)** | Default off; log `l2_pin:` |
| 2 | **C062** | `H3_TE_CACHE_AUTO=1` + `--repeat>=2` | bit-id emb | te 3.5s→~0.3s on HIT | **partial+** | Still not gold default |
| 3 | C070 | `H3_FORBID_HOT_SYNC` | n/a (hygiene) | low | landed(forbid) | KPI runs |
| 4 | C068/C069 | RoPE host memo + VAE H2D skip | bit-id | low; GPU WC TBD | landed / partial | Host done; WC verify deferred if smoke tight |
| 5 | C011 | VAE cast/prescale fuse | bit-id if fuse-only | mid | idea | Needs nsys; disk 99% |
| 6 | C003 redesign | in-kernel ConvRot+requant | bit-id if algebra kept | mid (was fake +3–8%) | **DISARMED** | Skip-ConvRot unreachable; no default-on |
| 7 | C018 QKV merge | maybe non-bit if INT8 order | mid | idea/maybe stale | Diff before build |
| 8 | Z05 CUDA Graph default | bit-id launch | <0.5s noise | closed default-off | Keep off |

**Deferred this batch:** real EPI_FUSE ConvRot-in-epilogue (large); VAE cast nsys (disk); RoPE GPU WC dedicated KPI; C040 wipe-free ledger run.

## 2. Landed code (this batch)

| Change | Files | Default |
| --- | --- | --- |
| C017 L2 pin Hadamard | `h3_gpu_cuda.cu` | `H3_CUDA_L2_PIN` off |
| C062 TE cache auto | `tools/h3_cuda_generate.c` | `H3_TE_CACHE_AUTO` off |
| Env contract / plugin | `h3_env_contract.c`, `tools/print_h3_env_contract.py`, `tools/h3_plugin.h` | listed |
| Beyond-DMAD CLI hint | `tools/h3_cuda_generate.c` help | text only |
| Pareto report | `tools/fewstep_pareto_report.py` | host |

## 3. Smoke plan (238)

1. rsync Mac → `/mnt/nvme1n1/workspace/h3_c_main_20261027` (no `--delete`, no models).
2. Rebuild `build-cuda-sm120/h3_cuda_generate`.
3. Short INT8 4-step smoke ×2: ctrl vs `H3_CUDA_L2_PIN=1`; expect `l2_pin: OK`, PROTO, same latent checksum if seed/contract fixed (bit-id), dit_ms delta noted (may be noise).
4. Optional: `H3_TE_CACHE_AUTO=1 --repeat 2` → clip2 HIT.
5. No NVFP4 chain WIP; no disk wipe.

## 4. Next blockers

- Disk ~99% / ~58G — blocks nsys / large multi-arm outs.
- C001 NVFP4 visual BLOCK — no chain.
- C003 true fuse needs new kernel design (not re-enable skip-ConvRot).
- C068/C069 wall-clock still needs a quiet KPI pass with `H3_ROPE_PROFILE=1`.
- Human aesthetic polish on PDMD/DMAD optional; OR already PASS fewstep.

## 5. Related

- Catalog: [`2026-10-27-full-optimization-catalog.md`](2026-10-27-full-optimization-catalog.md)
- Quality-preserving survey: [`2026-10-05-quality-preserving-acceleration.md`](2026-10-05-quality-preserving-acceleration.md)
- C003 boundary: [`2026-10-27-c003-epi-fuse-skip-convrot-unreachable.md`](2026-10-27-c003-epi-fuse-skip-convrot-unreachable.md)
