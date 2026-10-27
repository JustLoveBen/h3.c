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
| 消中转 / act dtype staging | `d_qkv`/`d_wide` FP16 WS | **C015/C042** | **default ON** | decision dig 2026-10-28 YES; Euler −1.41 GiB; multi-prompt waffle PASS; opt-out `=0` |
| GEMM+epilogue / ConvRot+requant | EPI_FUSE / INK / SL | C003 | **landed OFF** | SL +46% dit; ink ~0%; keep OFF |
| DeferredRelease overlap | TE↔DiT prefetch | C064 | **partial+ NO_WC_WIN** | Opt-in only |
| Config-as-contract / golden TE | env contract, te_cache | C061/C062 | **landed+** | Hygiene |
| VRAM phase ledger | `H3_VRAM_LEDGER` | C040 | **landed+** | Unblocks C015 |
| Forbid hot sync | sync audit | C070 | partial | P3 |
| PDL (DeepGEMM `set_pdl`) | GEMM→dequant bubble | C016 | **stale closed** | HW CC≥9.0；4090=8.9；[`c016-close`](2026-10-28-c016-pdl-dig-close.md) |
| L2 pin small tables | Hadamard pin | C017 | landed opt-in | No WC win |
| FlashMLA / MLA / GQA / DSA-as-attn | — | Z40/Z41 | **N-A** | No cross-step KV |
| DeepEP / MoE | — | Z40 | **N-A** | No MoE |
| NVFP4 co-design | FP4 DiT | C001 | **terminal BLOCK** visual | Not Track-A default |

## Priority next (Track-A waffleable, DeepSeek-derived)

1. ~~**C015**~~ **default ON** ([`c015-default-on`](2026-10-28-c015-default-on-decision.md)).  
2. ~~**C011**~~ VAE cast/prescale — **stale closed NO_WC_WIN** [`c011-nsys-close`](2026-10-28-c011-vae-cast-prescale-nsys-close.md) (VAE roundtrip ≈29 ms).  
3. ~~**C016 PDL**~~ — **CLOSED** @4090（无 HW + 无 cublasLt 挂点）。  
4. ~~**C043 streaming**~~ — **CLOSED** covered by INT8 window+prefetch+C041.  
5. ~~**C015 default-on decision**~~ — **CLOSED YES → flipped**.  
6. **Do not** reopen C018/C019/C003-SL/FlashMLA. Next Track-A pick from catalog / product WC (not another C015 dig).

Pillar-1 (Sol) ∩ DeepSeek Absorb → **closed** by Exp-G3; C015 near-lossless path **shipped default ON**.

## This commit deliverables

- C015 default-on decision dig + PASS table + **default ON flip**.  
- Checklist / track-a-status / env contract updated.  
- Tools: `run_c015_default_decision_dig_4090.sh`, `c015_default_decision_aggregate.py`.
