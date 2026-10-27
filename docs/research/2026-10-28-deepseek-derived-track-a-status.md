# DeepSeek-derived schemes → Track-A status (h3.c)

> **Asia/Shanghai 2026-10-28.** User t80u: continue implementing DeepSeek-inspired plans already borrowed — **not** blind FlashMLA.  
> Authority: `deepinfer-to-h3c-takeaways`, `deepinfer-thinking-to-h3c` (T01–T14), `llm-to-dit-transfer`, `deepinfer-speed-strategies` S1–S12, catalog C*.

## One-line

DeepSeek/DeepInfer value here is **strategy** (消中转、融合、DeferredRelease、合同/测量卫生、PDL toolbox) — **not** FlashMLA / DeepEP / DSA-as-default. Track-A only below.

## Landed vs pending (Track-A)

| DeepSeek / DeepInfer idea | h3 mapping | Catalog | Status | Notes |
|---|---|---|---|---|
| Absorb / merge projections | QKV 3→1 GEMM | C018 | **stale closed** | Exp-G3 proof 2026-10-28 |
| Fused Q-norm+RoPE+cast | `qkv_rope_f16_kernel` | C019 | **stale closed** | Same proof |
| 消中转 / act dtype staging | `d_qkv`/`d_wide` FP16 WS | **C015/C042** | **built OFF** | waffle+E2 before default; U=1 |
| GEMM+epilogue / ConvRot+requant | EPI_FUSE / INK / SL | C003 | **landed OFF** | SL +46% dit; ink ~0%; keep OFF |
| DeferredRelease overlap | TE↔DiT prefetch | C064 | **partial+ NO_WC_WIN** | Opt-in only |
| Config-as-contract / golden TE | env contract, te_cache | C061/C062 | **landed+** | Hygiene |
| VRAM phase ledger | `H3_VRAM_LEDGER` | C040 | **landed+** | Unblocks C015 |
| Forbid hot sync | sync audit | C070 | partial | P3 |
| PDL (DeepGEMM `set_pdl`) | GEMM→dequant bubble | C016 | **idea** | After real fuse; cublasLt no PDL |
| L2 pin small tables | Hadamard pin | C017 | landed opt-in | No WC win |
| FlashMLA / MLA / GQA / DSA-as-attn | — | Z40/Z41 | **N-A** | No cross-step KV |
| DeepEP / MoE | — | Z40 | **N-A** | No MoE |
| NVFP4 co-design | FP4 DiT | C001 | **terminal BLOCK** visual | Not Track-A default |

## Priority next (Track-A waffleable, DeepSeek-derived)

1. **C015 construction** (消中转) — only Track-A item with expected **VRAM** metric and Absorb kinship; gate waffle+E2.  
2. **C011** VAE cast/prescale fuse — S5; needs nsys; WC mid.  
3. **C016 PDL** — only after a self-written short chain exists; low WC.  
4. **Do not** reopen C018/C019/C003-SL/FlashMLA.

Pillar-1 (Sol) ∩ DeepSeek Absorb → **closed** by Exp-G3; remaining «DeepSeek path» for near-lossless **is C015**, not MLA kernels.

## This commit deliverables

- Exp-G3 evidence doc; catalog C018/C019 → stale closed.  
- C015 dig + ledger + stub (prior).  
- **C015 construction:** F16 qkv(+wide!lora) pack, `epi.y_f16`, F16qkv rope, FC2 cast; env default OFF.  
- This status table.
