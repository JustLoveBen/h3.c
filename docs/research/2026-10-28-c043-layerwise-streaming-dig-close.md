# C043 layerwise weight streaming dig → CLOSE (NO_NEW_MEAT / covered by window+C041)

**Date:** 2026-10-28 (Asia/Shanghai)  
**Host / product:** lan-4090-31 RTX 4090 sm_89 · INT8 ConvRot product path  
**Tip at dig start:** `1b504a5` (after C016 CLOSE)  
**Track:** A near-lossless · relate C041 · **no construction**

---

## 1. Claim → mechanism → metric

| Layer | Content |
|---|---|
| **Origin claim** | DeepSeek / SGLang Diffusion layerwise offload: keep 1–2 layers on GPU, prefetch next; report −30 GB VRAM and sometimes +perf when overlap is good (transfer §4.3). |
| **Mechanism** | Do not pin full DiT weights in HBM; stream block weights host→device on a side stream; overlap H2D with GEMM; evict LRU / sliding window. |
| **Metric** | Peak DiT VRAM (GiB) and warm dit_ms / WALL. Construction only if **clear VRAM or WC beyond C041 window=24** (already probed). Bit-identical if schedule-only. |

## 2. Overlap with existing INT8 window cache (already shipped)

| Piece | Status in tree | Notes |
|---|---|---|
| Sliding residence window + LRU eviction | **landed** `h3_w4a8_cache` `H3_W4A8_CACHE_WINDOW` | Auto-fit from free VRAM; host INT8 staging kept across device evict (reload = H2D only). |
| Side-stream H2D + `ensure_ready` | **landed** | Frontier M3: 「weight streaming overlap **已做**」. |
| Prefetch next block during compute | **landed** | `prefetch_next_u1` / WS-first warm **2** blocks then overlap. |
| Async H2D (U=1) | **landed** | `h3_w4a8_cache_set_async_h2d` on device-chained path. |
| **C041** `H3_DIT_INT8_WINDOW=N` (8..50) | **probed** [`c041`](2026-10-28-c041-window24-probe.md) | w24 @480p: Euler peak **−3.1 GiB**, DMAD **−2.2 GiB**; default **unchanged**. |
| C045 SSD BF16 streaming | **landed** (other path) | Mac/BF16; **互斥** with CUDA INT8 window story. |

**Conclusion:** The DeepSeek/SGLang “layerwise streaming” *mechanism* is already the product INT8 weight-window + prefetch subsystem. C043 is not a green-field feature.

## 3. Is there clear VRAM/WC *beyond* C041?

### 3.1 What “beyond C041” would mean

| Candidate | Beyond C041? | Verdict |
|---|---|---|
| New streaming runtime / pointer-swap framework | Would duplicate window cache | **No build** |
| Extreme window W=1..2 (true 1-layer) | Env floor today is **8** (`h3_dit_int8_window_env_cap`) | One-line floor change ⊂ **C041 lever**, not a C043 project; WC likely **negative** (PCIe) per transfer §4.3 |
| W=8 via existing env (vs probed 24) | Already legal | Extra ~16×~368 MiB ≈ **~5.9 GiB** VRAM vs w24 **without new code**; WC risk same family as C041 |
| Host SSD INT8 stream (C045-like) | New I/O path | High work; WC usually worse; 768p cudnn **already fits** |

### 3.2 Stacked levers already cover 768p headroom narrative

| Lever | Measured / estimated |
|---|---|
| C040 768p cudnn INT8 free@peak | **~0.8 GiB** (fits 24 GiB; folklore OOM wrong for this path) |
| C041 w24 | **+2.2–3.1 GiB** peak relief @480p (weight) |
| C015 FP16 stage (built OFF, gate PASS) | **+1.412 GiB** @480p (act) |
| Stacked (rough) | **~4+ GiB** free class — enough to try sol / thicker window **without** new streaming code |

Opt-menu deferred C043 until C041/C015 numbers — **those numbers exist**. They do **not** show a hole that only a new C043 runtime can fill.

### 3.3 WC

Transfer §4.3: streaming is a **768p capability** tool; speed **neutral to −2 s** if prefetch misses. C041 Euler already showed DiT **+1.4 s** at w24. Further shrinking window is **not** a Track-A WC win candidate.

## 4. waffle / default policy

| Policy | Decision |
|---|---|
| waffle | N/A (no impl). Window/prefetch path already BIT_ID vs full-resident when numerics equal. |
| Default window | **Unchanged** (C041 decision stands). |
| New `H3_*_STREAM` / C043 flag | **Do not add**. |
| FaceRefine / fake LoRA | Untouched. |
| C015 default | Untouched this slice. |

## 5. Verdict

**CLOSE — NO_NEW_MEAT / ALREADY_COVERED (window + prefetch + C041).**

| Criterion | Result |
|---|---|
| Claim new vs shipped window+prefetch? | **Duplicate** |
| Clear VRAM beyond C041 requiring new code? | **No** (use C041 N∈[8,50] and/or C015) |
| Clear WC beyond C041? | **No** (likely worse) |
| Construction? | **None** |

Catalog C043 → **stale closed (covered)**. Extreme W&lt;8, if ever needed, is a **C041 env-floor** micro-change after a dedicated probe — not a streaming rewrite.

## 6. Next DeepSeek checklist item

**C015 default-on decision dig** (evidence-only): multi-prompt waffle + warm WC + 768p VRAM stack with optional C041; **still no silent default flip** until PASS criteria written and met.  

Not C012 this turn (TF32 quality risk; non-DeepSeek). Not reopen C018/C019/C003-SL/C001/C016/C043.

## 7. Evidence anchors

- [`c041-window24-probe`](2026-10-28-c041-window24-probe.md)  
- [`c015-fp16-stage-kpi`](2026-10-28-c015-fp16-stage-waffle-e2-kpi.md)  
- [`c040-768p-ledger`](2026-10-28-c040-768p-ledger.md)  
- [`llm-to-dit-transfer`](2026-10-07-llm-to-dit-transfer.md) §4.3  
- [`sm120-operator-frontier`](2026-10-07-sm120-operator-frontier.md) §5.3 M3「已做」  
- Code: `h3_w4a8_cache.c` / `prefetch_next_u1` / WS-first in `h3_dit_w4a8.c`

