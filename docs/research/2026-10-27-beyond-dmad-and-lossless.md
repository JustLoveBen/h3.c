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
| 1 | **C017** | `H3_CUDA_L2_PIN=1` Hadamard AccessPolicyWindow | bit-id | −0.05–0.15s/clip (sm120) | **landed(opt-in)** @ hot-KPI | fair hot Δ≈0; BIT_ID PASS; **keep opt-in** |
| 2 | **C062** | `H3_TE_CACHE_AUTO=1` + `--repeat>=2` | bit-id emb | te 3.5s→~0.3s on HIT | **partial+** @ `95ab073` | AUTO MISS→reuse OK; still not gold default |
| 3 | C070 | `H3_FORBID_HOT_SYNC` | n/a (hygiene) | low | landed(forbid) | KPI runs |
| 4 | C068/C069 | RoPE host memo + VAE H2D skip | bit-id | rope host −9ms; dit Δ noise | landed / partial+ | Fair hot memo on/off done; default-on |
| 5 | C011 | VAE cast/prescale fuse | bit-id if fuse-only | mid | idea | Needs nsys; disk 99% |
| 6 | C003 redesign | in-kernel ConvRot+requant (`H3_DIT_EPI_FUSE_SL`) | bit-id (238 PASS) | mid (was fake +3–8%) | **landed OFF** | Naive SL hot dit **+46%**; ink ~0%; P2 DISARMED; keep OFF |
| 7 | C018 QKV merge | maybe non-bit if INT8 order | mid | idea/maybe stale | Diff before build |
| 8 | Z05 CUDA Graph default | bit-id launch | <0.5s noise | closed default-off | Keep off |

**Deferred / auth-gated:** disk wipe; nsys full; NVFP4 retrain/new weights; NVFP4 chain; VAE cast nsys; SL tiled/TC Hadamard (naive SL **+46%** dit).

## 2. Landed code (this batch)

| Change | Files | Default |
| --- | --- | --- |
| C017 L2 pin Hadamard | `h3_gpu_cuda.cu` | `H3_CUDA_L2_PIN` off |
| C062 TE cache auto | `tools/h3_cuda_generate.c` | `H3_TE_CACHE_AUTO` off |
| Env contract / plugin | `h3_env_contract.c`, `tools/print_h3_env_contract.py`, `tools/h3_plugin.h` | listed |
| Beyond-DMAD CLI hint | `tools/h3_cuda_generate.c` help | text only |
| Pareto report | `tools/fewstep_pareto_report.py` | host |

## 3. Smoke results (238, Asia/Shanghai ~2026-10-27 06:35+)

Work tree `h3_c_main_20261027` @ `95ab073`. Script `out/scripts/run_c017_l2_pin.sh`. Summaries on Mac: `out/c017_l2_pin_20261027_from238/`.

| Arm | te_ms | dit_ms | total_ms | latent checksum | Notes |
| --- | ---: | ---: | ---: | --- | --- |
| ctrl (TE HIT, cold proc) | 5725.9 | 22000.3 | 37624.0 | `0x4c6cb54664fa9586` | PROTO_OK; no `l2_pin` |
| pin `H3_CUDA_L2_PIN=1` | 4501.6 | 15381.1 | 28947.3 | `0x4c6cb54664fa9586` | PROTO_OK; `l2_pin: OK hadamard_f32` |
| AUTO clip1 MISS | 3516.0 | 15379.0 | 26684.1 | same | `H3_TE_CACHE_AUTO=1 --repeat 2` wrote `…/te_cache_auto` |
| AUTO clip2 hot | **0.0** | 15090.5 | 20103.7 | (hot seed path) | emb reuse; RUN_RC=0 |

- **Bit-identical:** ctrl≡pin latent checksum → **BIT_ID=PASS**.
- **WC claim:** ctrl dit 22s is **cold-first** confound; pin/AUTO ~15.4s matches prior DMAD hot — **no measurable L2 win** in this single A/B (est. −0.05–0.15s within noise). Keep opt-in.
- **C062 AUTO:** MISS→write then clip2 te=0 (in-process reuse). Still default-off.
- Disk still ~99% / 58G; no wipe; no NVFP4 chain.

## 4. Next blockers

- Disk ~99% / ~58G — blocks nsys / large multi-arm outs.
- C001 NVFP4 visual **terminal BLOCK** (energy RCA: single-GEMM OK; W4A4 multilayer attractor) — no chain; see `nvfp4-energy-collapse-block`.
- C003: INK ~0% WC; SL single-launch BIT_ID but **+46%** dit — needs tiled/TC Hadamard (not re-enable skip-ConvRot).
- C068/C069 fair hot done (Δ≈0 material); memo default-on.
- Human aesthetic polish on PDMD/DMAD optional; OR already PASS fewstep.


## 6. Fair hot KPI (238, Asia/Shanghai 2026-10-27)

Protocol: dmad4 INT8, TE HIT, `H3_FORBID_HOT_SYNC=1`, `--repeat`≥3, **discard clip1**, mean dit of hot clips. Same seed/contract. Summaries: `out/c017_hot_kpi_20261027_from238/`, `out/c068_rope_memo_kpi_20261027_from238/`.

### 6.1 C017 L2 pin (`H3_CUDA_L2_PIN`)

| Arm | hot dit mean (n=3) | vals | BIT_ID |
| --- | ---: | --- | --- |
| base | **15076.4** | 15042.6 / 15080.1 / 15106.6 | — |
| pin | **15138.8** | 15132.6 / 15141.0 / 15142.8 | per-clip seq **PASS** |

- Δ = **+62.4 ms (+0.41%)** → pin not faster.
- **CLAIM: NO_WALLCLOCK_WIN** — keep **opt-in**, do not default-on.
- `l2_pin: OK hadamard_f32` on pin arm only.

### 6.2 C068/C069 RoPE memo (`H3_ROPE_MEMO`, default on; `=0` disables)

| Arm | hot dit mean (n=2) | dit_rope hot avg total_ms | memo hits (hot) | BIT_ID |
| --- | ---: | ---: | --- | --- |
| memo_on | **15079.3** | **2.2** | 4/0 | PASS |
| memo_off | **15130.8** | **11.4** | 0/0 | PASS |

- Host rope saves ~9 ms/clip; dit Δ **+51.5 ms (+0.34%)** within noise vs 15.1s dit.
- **CLAIM: NO_MATERIAL_WALLCLOCK** — memo stays **default-on**; `H3_ROPE_MEMO=0` is KPI-only.
- VAE rope hot: on ≈0.02 ms (HIT) vs off ≈0.5 ms — also ≪ VAE blocks.

### 6.3 Round closure

**本轮开发收口条件已满足** (auto ROI exhausted under constraints):

| Done | Status |
| --- | --- |
| Beyond-DMAD rank + PDMD preferred | docs + CLI |
| C017 L2 pin | landed opt-in; fair hot Δ≈0 |
| C062 TE_CACHE_AUTO | partial+; default off |
| C068/C069 fair hot | memo default-on; WC material no |
| C003 EPI_FUSE / INK / SL | P2 **DISARMED**; INK green ~0%; SL @ `ed1ce72` BIT_ID+waffle PASS, hot dit **+46%** → **keep OFF** |
| C070 forbid | landed |
| C001 NVFP4 | visual **terminal BLOCK** (`ebd77a5` energy RCA) — no chain |

**剩余需授权 / 非自动 ROI（无更高自动 ROI → 开发收口）：**

1. **清盘**（~99%）— 阻塞 nsys / 大矩阵 outs  
2. **nsys** VAE cast/prescale（C011）— 需盘+授权  
3. **NVFP4** 重训/新权重 / 混精视觉门 — 非本轮；chain 仍 gated  
4. **EPI_FUSE_SL** tiled/TC Hadamard — 当前 naive 回归；勿 default-on  
5. 可选人眼抛光 PDMD/DMAD；扩 L2 pin（无证据不优先）


## 7. EPI_FUSE_SL 238 A/B (Asia/Shanghai 2026-10-27)

SHA `ed1ce72` (feat SL + cmake fix). Out: `out/c003_epi_fuse_sl_20261027/`. Host `i8_mismatch=0`.

| arm | SL | hot dit ms | hot latent | waffle |
| --- | --- | ---: | --- | --- |
| ctrl_r1/r2 | 0 | 14275.6 / 14366.9 | `0xc36483432681f772` | PASS |
| sl_r1/r2 | 1 | 20898.9 / 20910.4 | `0xc36483432681f772` | PASS |

BIT_ID vs ctrl; **~+46%** hot dit → default **OFF**. See [`epi-fuse-sl-single-launch`](2026-10-27-epi-fuse-sl-single-launch.md).

## 5. Related

- Catalog: [`2026-10-27-full-optimization-catalog.md`](2026-10-27-full-optimization-catalog.md)
- Quality-preserving survey: [`2026-10-05-quality-preserving-acceleration.md`](2026-10-05-quality-preserving-acceleration.md)
- C003 boundary: [`2026-10-27-c003-epi-fuse-skip-convrot-unreachable.md`](2026-10-27-c003-epi-fuse-skip-convrot-unreachable.md)
