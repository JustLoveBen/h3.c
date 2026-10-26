# C064 — DeferredRelease whitelist (offline)

Date: 2026-10-27 (Asia/Shanghai). **No hot-path overlap change.**

## Link

| Artifact | Role |
|---|---|
| [`2026-10-26-te-lifecycle-audit.md`](2026-10-26-te-lifecycle-audit.md) | Audit authority (peaks, gates, PDMD OOM) |
| `h3_lifecycle_whitelist.h` | **Code whitelist** (object / produce / release / event / safe_overlap) |
| `tests/test_h3_lifecycle_whitelist.c` | Host table sanity |
| `tools/h3_cuda_generate.c` | Optional `H3_LIFECYCLE_TRACE=1` prints transitions at **existing** release/join points |

## What landed

1. Structured whitelist rows (TE emb/store, DiT host_stage INT8 vs NVFP4, TE GPU prefetch, VAE prewarm host vs device-W, VAE rope).
2. Allowed host-sync documentation list (join / VAE wait / nsys / exit).
3. Debug env `H3_LIFECYCLE_TRACE=1`: dumps `# H3_LIFECYCLE_WHITELIST` once at generate start and prints `lifecycle_trace:` at TE release / host_stage wait|join. **Does not enable new overlap.**

## What did NOT land

- New TE∥DiT or VAE∥DiT overlap.
- Automatic DeferredRelease allocator.
- `H3_FORBID_HOT_SYNC` assert (C070 / P3).

## safe_overlap legend

| Value | Meaning |
|---|---|
| 0 | Free/join before next stage |
| 1 | May overlap next stage with documented wait/event |
| -1 | Gated / experimental (default skip or arm forbid) |

Real overlap remains gated by existing generate / prewarm logic (e.g. NVFP4 join, `latent_t≥32` skip prefetch, VAE `resident=0` → host-only).
