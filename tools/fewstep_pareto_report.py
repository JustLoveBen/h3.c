#!/usr/bin/env python3
"""Print beyond-DMAD fewstep Pareto from documented 238 numbers (host-only)."""
from __future__ import annotations

ROWS = [
    # rank, id, nfe, hot_dit_ms, quality, status, note
    (1, "pdmd4", 4, 15349.5, "paper best 4NFE auto; OR PASS", "C030 landed(visual-OR)",
     "PREFERRED over dmad4 at same NFE"),
    (2, "dmad4/lcm_renoise", 4, 15409.1, "OR PASS w/ smear; paper < PDMD", "C031 landed(visual-OR)",
     "alternate contract; wrong sampler = FAIL"),
    (3, "turbo4+refiner (C035)", 4, None, "unvalidated quality recovery", "idea",
     "A/B only; not gold"),
    (4, "8-step INT8 gold", 8, None, "highest fidelity", "gold",
     "NFE dominates; stack lossless on top"),
    (5, "UniPC/DPM++ (C036)", None, None, "training-free", "idea",
     "orthogonal; unlikely to beat distill @4"),
    (6, "SelfLift/VDN/TAE", None, None, "SKU tiers", "C080 idea",
     "not gold line"),
    ("—", "pdmd2", 2, None, "inherent blur", "C033 boundary",
     "regression only"),
    ("—", "NVFP4", None, None, "mosaic FAIL", "C001 BLOCK",
     "no chain / no default"),
]


def main() -> int:
    print("beyond-DMAD fewstep Pareto (docs/research/2026-10-27-beyond-dmad-and-lossless.md)")
    print(f"{'rank':>4}  {'preset':<22} {'NFE':>4}  {'hot_dit_ms':>10}  quality / status")
    print("-" * 88)
    for rank, pid, nfe, dit, qual, status, note in ROWS:
        nfe_s = "-" if nfe is None else str(nfe)
        dit_s = "-" if dit is None else f"{dit:.1f}"
        print(f"{str(rank):>4}  {pid:<22} {nfe_s:>4}  {dit_s:>10}  {qual}")
        print(f"{'':>4}  {'':<22} {'':>4}  {'':>10}  → {status}; {note}")
    print()
    print("Pass bar: fewstep-contract + PROTO + same-config waffle + visual (no mosaic).")
    print("Lossless stack (C017 L2 pin, C062 te_cache, C070 forbid) is orthogonal — apply on top.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
