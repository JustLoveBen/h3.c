# Product generate entry (NFE × raise-canvas)

> **Asia/Shanghai 2026-10-28.** A1 product wiring: connect NFE ladder + RC0/RC1/RC2 into the **real generation entry**, not only `source` env helpers.
> **Binary:** `build-cuda/h3_cuda_generate` (`tools/h3_cuda_generate.c`).
> **Product wrapper:** `tools/h3_product_generate.sh` (this doc).
> **Prior axes:** [`nfe-fewstep-contracts`](2026-10-28-nfe-fewstep-contracts.md) @`7cc2a0f` · [`raise-canvas-sku`](2026-10-28-raise-canvas-sku.md) @`cd3daf2`.

## 0. One-liner

```bash
# RC0 speed (480p DMAD) — default demo path
tools/h3_product_generate.sh --sku rc0 \
  --prompt-ids evals/768p/ids/p10_street_wave_dolly.ids \
  --out out/product_rc0 --repeat 2

# RC1 speed-face / RC2 max-face
tools/h3_product_generate.sh --sku rc1 --prompt-ids … --out out/rc1
tools/h3_product_generate.sh --sku rc2 --prompt-ids … --out out/rc2

# Compose without SKU alias
tools/h3_product_generate.sh --nfe dmad4 --canvas 1152 --dry-run

# Shell unit smoke (no GPU)
tools/h3_product_generate_smoke.sh
```

Operators no longer hand-export `H3_FEWSTEP_PRESET` + `H3_WIDTH` soup. Illegal combos **exit 2** before exec.

## 1. Why a wrapper (not only env scripts)

| Layer | Role |
|---|---|
| `h3_cuda_generate` | Real CUDA bypass CLI (weights, SDPA, fewstep contract) |
| `h3_nfe_product_ladder_env.sh` / `h3_raise_canvas_sku_env.sh` | Opt-in `source` helpers (still valid for ad-hoc digs) |
| **`h3_product_generate.sh`** | **Product entry** — resolves SKU → argv, refuses illegals, execs generate |

`--preset gold|turbo4|…` remains the older non-SKU path. Product RC / NFE picks go through this wrapper (or explicit compose flags).

## 2. SKU matrix (wired)

| `--sku` | Canvas | NFE | Notes |
|---|---|---|---|
| `rc0` | 864×480 | DMAD-4 | default speed @480p |
| `rc0_euler` | 864×480 | Euler-8 | quality / waffle gold |
| `rc1` | 1152×640 | DMAD-4 | speed-face |
| `rc1c` | 1152×640 | PDMD-4 | contrast-only |
| `rc2` | 1344×768 | Euler-8 | max face |
| `rc960` | 960×544 | DMAD-4 | optional |

Compose: `--nfe euler8|dmad4|pdmd4` + `--canvas 480|1152|1344|960|WxH`.

## 3. Hard refuses (exit 2 + `REFUSED`)

- `1280×720` / any W or H not `%32==0`
- `1344×768 + dmad4|pdmd4` (4090 INT8 fewstep noise @`47b4cd8`)
- `pdmd2` (no 2NFE LoRA on disk)
- Overriding `rc2` with `--nfe dmad4` still refused

## 4. Orthogonal knobs

| Flag / env | Default | Notes |
|---|---|---|
| `--sdpa-backend` | `cudnn` | product default unchanged |
| `--s2` | off | sources `h3_track_b_s2_env.sh`; sets backend `sol_cute` if unset |
| `--window24` | off | `H3_DIT_INT8_WINDOW=24` only |
| C015 act-FP16 | ON (unset) | opt-out `H3_DIT_ACT_FP16_STAGE=0` |
| `--skip-audio` | on | `--with-audio` to keep audio |
| `H3_GENERATE` / `H3_MODELS` / `H3_DMAD_LORA` | 4090 paths | override per machine |

## 5. Smoke

```bash
tools/h3_product_generate_smoke.sh   # 17 checks: legal dry-run + refuse asserts
```

GPU smoke (4090): `--sku rc0 --seconds 1 --repeat 1` should not break the 480p default path.

## 6. Related

- [`nfe-fewstep-contracts`](2026-10-28-nfe-fewstep-contracts.md)
- [`raise-canvas-sku`](2026-10-28-raise-canvas-sku.md)
- [`track-b-s2-sellable-recipe`](2026-10-28-track-b-s2-sellable-recipe.md)
- Helpers still usable: `tools/h3_nfe_product_ladder_env.sh`, `tools/h3_raise_canvas_sku_env.sh`

## 7. Follow-on gates (same entry)

- A2 RC1/RC2 complex: [`a2-rc1-rc2-complex-gates`](2026-10-28-a2-rc1-rc2-complex-gates.md)
- A3 C015+WINDOW24 ledger: [`a3-c015-window24-ledger`](2026-10-28-a3-c015-window24-ledger.md)
- Runner: `tools/run_a1a2a3_product_gates_4090.sh`
