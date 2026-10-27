# Exp-G3 · Track-A · Sol Pillar-1 gap proof (static)

> **Asia/Shanghai 2026-10-28.** No long GPU run (serialize vs Exp-G1 Track-B).  
> **Protocol:** claim → mechanism → metric. Source SHA at land: see commit.  
> **Also:** closes DeepSeek/Sol «Absorb / QKV-merge» narrative for this repo.

## Claim (under test)

Current INT8 hot path still has (a) three thin QKV GEMMs and/or (b) unfused QKNorm+RoPE — else **close C018/C019** and stop quoting −0.6~−1.2 s.

## Mechanism (Sol Pillar-1 vs h3.c)

| Pillar-1 item | Sol / DeepSeek-adjacent idea | h3.c hot path (INT8) | Verdict |
|---|---|---|---|
| QKV 3→1 wide GEMM | Merge three projections (launch+tile tax) | Single `H3_W4A8_LINEAR_QKV`; one `h3_cuda_gemm_i8_dequant_f32_w_act_dev` → `d_qkv[seq, inner·3]` in `block_phase_qkv` | **Already landed** |
| Fused QKNorm+RoPE (+cast) | Absorb small ops beside big attn | One `h3_cuda_qkv_rope_f16_kernel` does RMSNorm(Q/K)×γ + RoPE + F16 Q/K/V emit; then SDPA (`h3_cuda_qkv_rope_sdpa_f16_dev_layout`) | **Already landed** |
| Graph capture | Launch amortize | Default **off** (Z05 / cuDNN graph policy); not reopened here | Separate; not C018 |
| GEMM→rope without F32 `d_qkv` | Absorb / 消中转 | Still materialize F32 `d_qkv` then rope kernel | **Residual → C015 family**, not C018 |

Prior deadend note (`2026-10-07-sm120-operator-frontier.md`): «QKV merge/QKNorm+RoPE 融合（本就有，MAE=0）».

## Metric (source counters — no nsys required for FAIL)

Per DiT block, INT8 device path:

1. **QKV GEMM launches:** 1 (`uly_klog("qkv_gemm")` → single dequant GEMM). Linear kinds enum has **no** separate Q/K/V (`H3_W4A8_LINEAR_QKV|OUT|FC1|FC2` only).
2. **QKNorm / RoPE / F16 cast launches before SDPA:** 1 fused kernel (`h3_cuda_qkv_rope_f16_kernel`), not three host phases.
3. **Historical WC claim −0.6~−1.2 s:** attributed to missing merge — **invalid once (1)(2) hold**. Sol official Pillar-1 full stack ~1.08× @50-step; do not re-estimate merge alone.

## Result

| ID | Old status | New status |
|---|---|---|
| **C018** QKV merge | idea / maybe stale | **stale closed** (gap absent) |
| **C019** QKNorm+RoPE fuse | idea | **stale closed** (hot path fused; FP `H3_DIT_QKNORM_ROPE_FUSE` is host/FP probe only) |
| Exp-G3 | open | **FAIL (gap not present)** → stop merge narrative |

## What remains (do **not** call C018)

- **C015/C042** FP16 workspace staging of `d_qkv`/`d_wide` (消中转 / Absorb spirit; waffle, not BIT_ID).
- Optional: GEMM epi writing F16 C directly into rope inputs (construction ⊂ C015).
- CUDA Graph: keep default off; measure launch tax before reopening.

## DeepSeek mapping

MLA **Absorb** (fold up-proj into neighbor GEMM) is the algebraic cousin of QKV-merge.  
`llm-to-dit-transfer` already marked MLA **N-A** for KV-cache reasons and noted QKV wide GEMM **already closed**. Exp-G3 is the catalog-facing proof of that closure.
