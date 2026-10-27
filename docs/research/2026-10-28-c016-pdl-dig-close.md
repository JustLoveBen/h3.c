# C016 PDL dig → CLOSE (NO_WC_WIN / HW_UNAVAILABLE @4090)

**Date:** 2026-10-28 (Asia/Shanghai)  
**Host:** lan-4090-31 RTX 4090 **sm_89 / CC 8.9** (verified `nvidia-smi --query-gpu=compute_cap`)  
**Tip at dig start:** `1db8a30` (after C011 CLOSE)  
**Track:** A near-lossless · waffle/BIT_ID · default OFF until WC win — **no construction**

---

## 1. Claim → mechanism → metric

| Layer | Content |
|---|---|
| **Origin claim** | DeepGEMM / DeepInfer `set_pdl(True)`；vLLM DeepSeek 报 PDL 相关 ~4% e2e；transfer §3.2「部分迁移」估 −0.1~−0.3 s |
| **Mechanism** | Programmatic Dependent Launch (CUDA ≥ **CC 9.0**): same-stream secondary launched early via `cudaLaunchAttributeProgrammaticStreamSerialization`; primary `cudaTriggerProgrammaticLaunchCompletion`; secondary `cudaGridDependencySynchronize`. Overlaps consumer prologue with producer tail — **launch/tail bubble only**, not memory round-trips. |
| **Metric (ours)** | Wall-clock DiT clip (Euler-8 / DMAD-4 480p); gate for construction ≈ **≥0.3 s** fuseable/launch meat (same rigor as C011). Bit-identical if pure schedule. |

## 2. Ada / 4090 feasibility — HARD FAIL

NVIDIA CUDA Programming Guide §4.5: PDL **requires compute capability 9.0+** (Hopper / Blackwell).  
CUTLASS docs: “Hopper and Blackwell architectures support PDL.”  
**RTX 4090 = CC 8.9** → APIs do not provide overlapping dependent launch on the **product GPU**.

| Target | CC | PDL HW |
|---|---|---|
| lan-4090-31 (product) | 8.9 | **No** |
| sm_90 Hopper | 9.0 | Yes |
| sm_120 Blackwell | 12.0 | Yes |

Catalog/transfer lines that say “sm120 L2 / sm_120 ✓” are **not** a 4090 product path. Track-A ship target here is Ada.

## 3. Is there a real fuse/short-chain to hang PDL on?

### 3.1 Producer GEMM = cuBLASLt (no PDL switch)

Product INT8 DiT path: `cublasLtMatmul` (persistent `lt_i8[]` descriptors in `h3_gpu_cuda.cu`).  
FlashInfer / transfer already note: **cuBLASLt does not expose PDL**.  
→ Cannot hang PDL on the GEMM→dequant edge at the **producer** that dominates time.

### 3.2 Self-written consumers exist — but are not the WC meat

Post-GEMM self-written kernels present (`h3_cuda_dequant_*`, `h3_cuda_quantize_act_per_row*`, row_prescale, EPI_FUSE requant variants).  
`cudaLaunchKernelEx` / any Programmatic attribute: **absent** from product tree (grep clean outside `third_party`).

### 3.3 Fusion already covers PDL’s target pairs

`H3_DIT_EPI_FUSE` / `*_requant_*` kernels **collapse dequant→requant** into one launch — the exact bubble PDL would chase between those two. Path is **default OFF** (C003 boundary; SL historically +46% dit). PDL would be a *weaker* substitute for a fuse we already have and keep off for WC reasons.

### 3.4 Even on sm90+, WC upper bound is below gate

Prior **L2** assessment ([`sm120-operator-frontier`](2026-10-07-sm120-operator-frontier.md) §4.2): ~3 μs × ~4000 boundaries ≈ **−0.01~−0.05 s**.  
P2b whole-step **CUDA Graph** only **−0.16 s** on turbo4 (same family: kill launch tax) — PDL on a *subset* of edges cannot beat that.  
Transfer’s −0.1~−0.3 s was **optimistic**; dig locks the stricter L2/P2b bound. **≪ 0.3 s construction gate.**

## 4. waffle / default policy

| Policy | Decision |
|---|---|
| waffle / BIT_ID | Would be bit-identical *if* implemented (schedule-only). **N/A** — no impl. |
| Default | **Do not add** `H3_*_PDL` env. No opt-in stub. |
| FaceRefine / LoRA download | Untouched (out of scope). |
| C015 default | Untouched. |

## 5. Verdict

**CLOSE — NO_WC_WIN + HW_UNAVAILABLE @4090.**

| Criterion | Result |
|---|---|
| Product CC ≥ 9.0? | **No** (8.9) |
| Hangable GEMM→epi PDL edge? | **No** (cublasLt producer) |
| Residual self-written bubble after EPI_FUSE design? | Already fuse-collapsible; EPI keep OFF |
| Expected WC if forced on sm90+ | ≤ ~0.05 s (L2); ≪ graph −0.16 s ceiling |
| Construction? | **None** |

Aligns catalog **Z11** (“低优先尾料”) and frontier L2 “已评估关闭”; DeepSeek checklist item C016 now **stale closed**.

## 6. Next DeepSeek checklist item

**C043 — layerwise weight streaming dig** (checklist row「权重窗 / streaming」; C041 window24 already probed, default unchanged).  
Not C012 (TF32 quality risk; checklist: 非 DeepSeek #3).  
Not reopen C018/C019/C003-SL/FlashMLA/C001.  
Not C015 default flip this slice.

## 7. Evidence anchors

- `nvidia-smi --query-gpu=name,compute_cap` → `NVIDIA GeForce RTX 4090, 8.9` @ lan-4090-31  
- CUDA Programming Guide §4.5 PDL (CC 9.0+)  
- `h3_gpu_cuda.cu`: cublasLt INT8 path; no `cudaLaunchKernelEx` / Programmatic attrs  
- `h3_dit_fused_epi.h`: EPI_FUSE default OFF  
- [`2026-10-07-sm120-operator-frontier.md`](2026-10-07-sm120-operator-frontier.md) §4.2 L2 close  
- [`2026-10-07-llm-to-dit-transfer.md`](2026-10-07-llm-to-dit-transfer.md) §3.2 (optimistic estimate superseded for Ada)

