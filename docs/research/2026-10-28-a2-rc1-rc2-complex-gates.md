# A2 RC1/RC2 complex-prompt gate expansion

> **Asia/Shanghai 2026-10-29.** After A1 product entry @`9de1d16`.
> **Entry:** `tools/h3_product_generate.sh --sku rc1|rc2`
> **Runner:** `tools/run_a1a2a3_product_gates_4090.sh`
> **Prompt:** `evals/768p/ids/p10_street_wave_dolly.ids` (complex 480p×5s style)
> **Policy:** burn-in scheme + warm WALL; Track-B (RC1) vs Track-A (RC2) separate — no cross-arm waffle.
> **Also:** A1 480p default path smoke PASS via same entry (`--sku rc0 --seconds 1`).

## Arms

| Arm | SKU | Canvas | NFE | WINDOW=24 | Repeat | RUN_RC |
|---|---|---|---|---|---|---|
| A1 smoke | `rc0` | 864×480 | DMAD-4 | off | 1 | **0** |
| A2-RC1 | `rc1` | 1152×640 | DMAD-4 | ON (A3) | 2 | **0** |
| A2-RC2 | `rc2` | 1344×768 | Euler-8 | ON (A3) | 2 | **0** |

## Gate results @4090 (`synced 9de1d16`)

| Arm | cold clip1 total | **warm clip2 total** | dit warm | Label |
|---|---:|---:|---:|---|
| RC0 1s | 49.3 s (load-heavy) | n/a (repeat=1) | 5.0 s | `RC0 864x480 DMAD-4 INT8 C015` |
| RC1 5s | 79.8 s | **50.7 s** | 36.7 s | `RC1 1152x640 DMAD-4 INT8 C015 w24` |
| RC2 5s | 153.0 s | **125.4 s** | 105.4 s | `RC2 1344x768 Euler-8 INT8 C015 w24` |

- `sdpa_actual=cudnn` both arms; fewstep_bound OK on RC1; Euler sampler on RC2.
- Visual: RUN_RC=0, no FAIL/mosaic in logs; labeled warm `_r2` clips burned.
- Share: `~/grokbot-share/a1a2a3_product_gates_20261028/labeled/`

## Related

- [`product-generate-entry`](2026-10-28-product-generate-entry.md) @`9de1d16`
- [`a3-c015-window24-ledger`](2026-10-28-a3-c015-window24-ledger.md)
- [`raise-canvas-sku`](2026-10-28-raise-canvas-sku.md)
