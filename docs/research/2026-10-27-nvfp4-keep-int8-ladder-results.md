# NVFP4 KEEP_INT8 ladder results (2026-10-27 Asia/Shanghai)

**WS:** `/mnt/nvme1n1/workspace/h3_c_main_20261027` SHA `a7d7375`  
**Out:** `out/keep_int8_ladder_20261027/`  
**Protocol:** 864×480, 5s, 8 steps, cfg1, seed `0x76800002`, cudnn, skip-audio, live TE, `H3_DIT_NVFP4=1` + w4a4 sidecar, INT8 ConvRot store. Chain unset. EPI OFF.  
**Ladder done:** `2026-10-27T04:45:48+00:00` (UTC; Asia/Shanghai = +8 → 12:45).

**INT8 gold checksum (same protocol):** `0x75efbb7057231ead`  
Cold-clip `waffle_guard INVALID_LOWENERGY` is a **known false positive** on INT8 gold (`p0_visual_20261027/int8_1`) — do not fail on cold WARN alone.

| arm | KEEP | fp4_mask | dit_ms | latent checksum | std | waffle_cold | mp4 | verdict |
|---|---|---|---|---|---|---|---|---|
| k_all | all | 0x0 | 35936 | **`0x75efbb7057231ead`** | 1.026 | 5/5 INVALID (=gold) | 1.2M | **PASS BIT_ID vs INT8 gold** |
| k_attn | attn (qkv,out) | 0xc mlp FP4 | 30762 | `0xc45bd33eb5b2d7d1` | 1.024 | 3/5; max_b16=0.884 | 2.3M | energy OK; **human visual TBD** |
| k_mlp | mlp (fc1,fc2) | 0x3 attn FP4 | 37324 | `0xef9705a253f47c82` | **12.12** | 0/5; max_b16=27.3 | 7.7M | **FAIL** energy explode |
| k_qkv | qkv | 0xe | 29581 | `0x95c5882d174bced9` | **0.310** | 5/5 INVALID | 2.1M | **FAIL** energy collapse (=full) |
| k_full | unset | full FP4 | 23214 | `0x95c5882d174bced9` | **0.310** | 5/5 INVALID | 2.1M | **FAIL** C001 collapse |

## Ranking (max FP4 coverage that still looks alive)

1. **KEEP=all** — BIT_ID gold (sanity of overlay). No FP4 coverage.
2. **KEEP=attn** (MLP on FP4) — only arm with std≈1.0 and ≠collapse/explode. **Best candidate for human review / further relax** (e.g. try KEEP=attn,fc1 next — not in this ladder).
3. KEEP=mlp / qkv / full — unusable (explode or collapse).

**Sensitive linears:** attention (`qkv`/`out`) on this naive w4a4 pack. Opening attn to FP4 while keeping MLP INT8 **explodes** (std 12). Keeping only `qkv` INT8 collapses identically to full FP4 → **`out` (and/or fc*) FP4 is enough to match full-collapse checksum**.

## Product

- Default stays INT8 until hybrid clears human visual + waffle (non-cold).
- Next ladder slice (optional): `KEEP=attn,fc1`, `KEEP=attn,fc2`, `KEEP=qkv,out,fc1` to peek more FP4 without losing attn INT8.
- No convert, no chain, no EPI default-on.

## Artifacts

`out/keep_int8_ladder_20261027/{k_all,k_attn,k_mlp,k_qkv,k_full}/{generate.log,summary.txt,latent.raw,p01_sprint.mp4}`
