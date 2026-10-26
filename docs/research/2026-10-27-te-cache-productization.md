# C062 — TE prompt cache productization (offline)

Date: 2026-10-27 (Asia/Shanghai). Scope: **code + docs, no GPU**.

## Status

Opt-in path only. Gold / default production behavior stays **cache off**.
This slice hardens the opt-in path so P0 / fewstep / KPI scripts can grep HIT
and optionally require a warm cache.

## Key fields (cache identity)

| Field | Source | In key hash? | In file header? |
|---|---|---|---|
| token ids | TE tokenizer output | yes | tokens count |
| `te_layers` | CLI / encode path | yes | yes |
| `text_dim` (`H3_TE_CACHE_TEXT_DIM=5120`) | compile-time | yes (C062) | yes |
| TE path string | `--te` / resolved path | yes | no |
| TE file mtime⊕size | `stat(te_path)` | yes | no |

Vision / multimodal spans: **never cached** (`SKIP reason=vision_span`).

Changing `text_dim` or layers invalidates keys (cold MISS → rewrite). Old
on-disk blobs with a different key hash are simply unused; header still
rejects dim/layers/token mismatches.

## Default strategy (proposal)

1. **Ship default: off** (`--te-cache` / `H3_TE_PROMPT_CACHE` unset).
2. **Dev / KPI warm path:**
   - Run 1 (cold): `--te-cache DIR` only → expect `MISS wrote`.
   - Run 2 (warm): same DIR + `H3_TE_CACHE_REQUIRE_HIT=1` → expect `HIT` or
     `tag=FAIL_ENV`.
3. **P0 visual gate:** do **not** force cache on. Soft-recognize log lines;
   hard gate only with `--expect-te-cache-hit` when a warm DIR is intentional.
4. Process lifetime: TE encode is typically once per generate process; HIT
   gate is primarily **cross-process** warm KPI, not intra-process second clip.

## Log contract (grep-friendly)

```
te_prompt_cache: dir=DIR require_hit=0|1 (opt-in; gold default off)
te_prompt_cache: HIT dir=DIR key=HEX tokens=N layers=N dim=N
te_prompt_cache: MISS wrote dir=DIR key=HEX tokens=N layers=N dim=N
te_prompt_cache: MISS dir=DIR key=none tokens=N layers=N dim=N reason=no_key
te_prompt_cache: SKIP dir=DIR reason=vision_span
te_prompt_cache: REQUIRE_HIT failed (H3_TE_CACHE_REQUIRE_HIT=1; warm cache first)
```

`tools/p0_protocol_check_log.py --expect-te-cache-hit` requires a HIT line.

## Env contract

Printed under `# H3_ENV_CONTRACT` (info rules, not P0-forbidden):

- `H3_TE_PROMPT_CACHE` — cache directory (same as `--te-cache`)
- `H3_TE_CACHE_REQUIRE_HIT` — `1` → MISS is `FAIL_ENV` when cache enabled

## HIT hard gate for P0

- Soft: lines recognized; MISS alone is a warning.
- Hard: `--expect-te-cache-hit` or runtime `H3_TE_CACHE_REQUIRE_HIT=1`.
- Do not enable REQUIRE_HIT on a cold DIR; warm first.

## Files

- `h3_te_cache.h` / `h3_te_cache.c` — key completeness
- `tools/h3_cuda_generate.c` — logs + REQUIRE_HIT
- `h3_env_contract.c` + `tools/print_h3_env_contract.py` + `tools/h3_plugin.h`
- `tools/p0_protocol_check_log.py` — recognition / `--expect-te-cache-hit`
- `tests/test_h3_te_cache.c` — layers affect key

## Out of scope (this slice)

- Turning cache on by default
- GPU timing of HIT vs MISS
- Vision span caching
