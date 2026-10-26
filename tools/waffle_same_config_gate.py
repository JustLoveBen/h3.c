#!/usr/bin/env python3
"""Same-config waffle seam regression gate (offline; no GPU required).

Compares `waffle: frame=N b16=...` lines from two generate.log files that
MUST be the same protocol (same seed/steps/arm/graph/…). Fail when:

  - frame sets differ
  - any INVALID_LOWENERGY mismatch
  - mean |Δb16| or max |Δb16| exceeds thresholds
  - optional --require-frames count not met

This is a *seam* / regression helper, not a cross-arm quality claim.
Cross-config waffle (INT8 vs NVFP4, PDMD vs DMAD, turbo4 vs dense) is
explicitly out of scope — see docs/research/2026-10-07-dmad-noise-diagnosis.md.

Exit 0 = pass; 1 = fail; 2 = usage / parse error.
See docs/research/2026-10-26-waffle-same-config-gate.md.
"""
from __future__ import annotations

import argparse
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

_WAFFLE_RE = re.compile(
    r"^waffle:\s*frame=(?P<frame>\d+)\s+b16=(?P<b16>[-+0-9.eE]+)"
    r"(?:\s+(?P<invalid>INVALID_LOWENERGY)\b.*)?\s*$",
    re.M,
)


@dataclass(frozen=True)
class WaffleRow:
    frame: int
    b16: float
    invalid: bool


def parse_waffle(text: str) -> dict[int, WaffleRow]:
    out: dict[int, WaffleRow] = {}
    for m in _WAFFLE_RE.finditer(text):
        fr = int(m.group("frame"))
        b16 = float(m.group("b16"))
        inv = m.group("invalid") is not None
        # last line wins if duplicates (hot clip usually last)
        out[fr] = WaffleRow(frame=fr, b16=b16, invalid=inv)
    return out


def compare(
    base: dict[int, WaffleRow],
    cand: dict[int, WaffleRow],
    *,
    mean_tol: float,
    max_tol: float,
    require_frames: int | None,
) -> tuple[list[str], list[str]]:
    errs: list[str] = []
    notes: list[str] = []

    if not base:
        errs.append("baseline: no waffle: lines parsed")
    if not cand:
        errs.append("candidate: no waffle: lines parsed")
    if errs:
        return errs, notes

    if require_frames is not None and len(base) < require_frames:
        errs.append(
            f"baseline has {len(base)} waffle frames (need >= {require_frames})"
        )
    if require_frames is not None and len(cand) < require_frames:
        errs.append(
            f"candidate has {len(cand)} waffle frames (need >= {require_frames})"
        )

    bf, cf = set(base), set(cand)
    if bf != cf:
        errs.append(
            f"frame set mismatch: baseline={sorted(bf)} candidate={sorted(cf)}"
        )
        return errs, notes

    deltas: list[float] = []
    for fr in sorted(bf):
        b, c = base[fr], cand[fr]
        if b.invalid != c.invalid:
            errs.append(
                f"frame={fr}: INVALID_LOWENERGY mismatch "
                f"baseline={b.invalid} candidate={c.invalid}"
            )
        d = abs(b.b16 - c.b16)
        deltas.append(d)
        notes.append(
            f"frame={fr} base={b.b16:.3f} cand={c.b16:.3f} "
            f"abs_delta={d:.3f} invalid={b.invalid}/{c.invalid}"
        )

    if not deltas:
        errs.append("no overlapping frames to compare")
        return errs, notes

    mean_d = sum(deltas) / len(deltas)
    max_d = max(deltas)
    notes.append(f"mean_abs_delta={mean_d:.4f} max_abs_delta={max_d:.4f}")
    notes.append(f"tol mean<={mean_tol} max<={max_tol}")

    if mean_d > mean_tol:
        errs.append(
            f"mean |Δb16|={mean_d:.4f} exceeds --mean-tol={mean_tol}"
        )
    if max_d > max_tol:
        errs.append(
            f"max |Δb16|={max_d:.4f} exceeds --max-tol={max_tol}"
        )
    return errs, notes


def _self_test() -> int:
    base = """
waffle: frame=0 b16=1.100
waffle: frame=24 b16=0.900
waffle: frame=48 b16=0.850
"""
    ok = """
waffle: frame=0 b16=1.120
waffle: frame=24 b16=0.910
waffle: frame=48 b16=0.860
"""
    bad = """
waffle: frame=0 b16=2.500
waffle: frame=24 b16=0.910
waffle: frame=48 b16=0.860
"""
    inv_mismatch = """
waffle: frame=0 b16=1.100 INVALID_LOWENERGY block_ratio=1.0 gate=10.00
waffle: frame=24 b16=0.900
waffle: frame=48 b16=0.850
"""
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        (td_p / "base.log").write_text(base)
        (td_p / "ok.log").write_text(ok)
        (td_p / "bad.log").write_text(bad)
        (td_p / "inv.log").write_text(inv_mismatch)

        e, _ = compare(
            parse_waffle(base), parse_waffle(ok),
            mean_tol=0.05, max_tol=0.10, require_frames=3,
        )
        if e:
            print("SELF-TEST FAIL: expected ok pair to pass:", e, file=sys.stderr)
            return 1

        e, _ = compare(
            parse_waffle(base), parse_waffle(bad),
            mean_tol=0.05, max_tol=0.10, require_frames=3,
        )
        if not e:
            print("SELF-TEST FAIL: expected bad pair to fail", file=sys.stderr)
            return 1

        e, _ = compare(
            parse_waffle(base), parse_waffle(inv_mismatch),
            mean_tol=0.05, max_tol=0.10, require_frames=3,
        )
        if not any("INVALID_LOWENERGY" in x for x in e):
            print("SELF-TEST FAIL: expected INVALID mismatch:", e, file=sys.stderr)
            return 1

    print("waffle_same_config_gate: self-test PASS")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Same-config waffle seam regression (offline)."
    )
    p.add_argument("baseline", nargs="?", type=Path, help="baseline generate.log")
    p.add_argument("candidate", nargs="?", type=Path, help="candidate generate.log")
    p.add_argument(
        "--mean-tol",
        type=float,
        default=0.15,
        help="fail if mean |Δb16| exceeds this (default 0.15; matches historical seam)",
    )
    p.add_argument(
        "--max-tol",
        type=float,
        default=0.40,
        help="fail if any frame |Δb16| exceeds this (default 0.40)",
    )
    p.add_argument(
        "--require-frames",
        type=int,
        default=5,
        help="minimum waffle frames required in each log (default 5)",
    )
    p.add_argument(
        "--self-test",
        action="store_true",
        help="run built-in synthetic regression checks and exit",
    )
    args = p.parse_args(argv)

    if args.self_test:
        return _self_test()

    if not args.baseline or not args.candidate:
        p.error("baseline and candidate logs required (or pass --self-test)")

    if not args.baseline.is_file():
        print(f"missing baseline: {args.baseline}", file=sys.stderr)
        return 2
    if not args.candidate.is_file():
        print(f"missing candidate: {args.candidate}", file=sys.stderr)
        return 2

    base = parse_waffle(args.baseline.read_text(errors="replace"))
    cand = parse_waffle(args.candidate.read_text(errors="replace"))
    errs, notes = compare(
        base,
        cand,
        mean_tol=args.mean_tol,
        max_tol=args.max_tol,
        require_frames=args.require_frames,
    )
    for n in notes:
        print(n)
    if errs:
        for e in errs:
            print(f"FAIL: {e}", file=sys.stderr)
        return 1
    print("waffle_same_config_gate: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
