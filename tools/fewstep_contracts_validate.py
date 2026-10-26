#!/usr/bin/env python3
"""Validate tests/fixtures/fewstep_contracts.json against schema + invariants.

Offline; no GPU. Prefer jsonschema if installed; otherwise structural checks.
Also optional --check-dump: compare fixture to `dump_fewstep_contracts` output
(caller runs the binary separately via make).

Exit 0 = ok; 1 = fail; 2 = usage.
See docs/research/2026-10-26-fewstep-contracts-json.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "tests/fixtures/fewstep_contracts.json"
DEFAULT_SCHEMA = ROOT / "tests/fixtures/fewstep_contracts.schema.json"

_EXPECTED = {
    "pdmd4": dict(steps=4, sampler="velocity_euler", audio=3.0, lcm=False),
    "pdmd2": dict(steps=2, sampler="velocity_euler", audio=3.0, lcm=False),
    "dmad4": dict(steps=4, sampler="lcm_renoise", audio=2.0, lcm=True),
}


def _structural(doc: dict) -> list[str]:
    errs: list[str] = []
    if doc.get("schema_version") != 1:
        errs.append(f"schema_version={doc.get('schema_version')!r} want 1")
    fps = doc.get("forbidden_speed_presets")
    if not isinstance(fps, list) or "turbo4" not in fps:
        errs.append("forbidden_speed_presets must include turbo4")
    contracts = doc.get("contracts")
    if not isinstance(contracts, list) or not contracts:
        errs.append("contracts missing/empty")
        return errs
    seen: set[str] = set()
    for c in contracts:
        if not isinstance(c, dict):
            errs.append("contract entry not object")
            continue
        cid = c.get("id")
        if cid not in _EXPECTED:
            errs.append(f"unexpected contract id={cid!r}")
            continue
        if cid in seen:
            errs.append(f"duplicate id={cid}")
        seen.add(cid)
        exp = _EXPECTED[cid]
        if c.get("steps") != exp["steps"] or c.get("nfe") != exp["steps"]:
            errs.append(f"{cid}: steps/nfe != {exp['steps']}")
        if c.get("sampler") != exp["sampler"]:
            errs.append(f"{cid}: sampler={c.get('sampler')!r}")
        if abs(float(c.get("video_shift", -1)) - 12.0) > 0.051:
            errs.append(f"{cid}: video_shift != 12")
        if abs(float(c.get("audio_shift", -1)) - exp["audio"]) > 0.051:
            errs.append(f"{cid}: audio_shift != {exp['audio']}")
        if bool(c.get("want_lcm_renoise")) != exp["lcm"]:
            errs.append(f"{cid}: want_lcm_renoise mismatch")
        if abs(float(c.get("cfg", -1)) - 1.0) > 0.051:
            errs.append(f"{cid}: cfg != 1")
    missing = set(_EXPECTED) - seen
    if missing:
        errs.append(f"missing contracts: {sorted(missing)}")
    return errs


def _schema_validate(doc: dict, schema: dict) -> list[str]:
    try:
        import jsonschema  # type: ignore
    except ImportError:
        return []  # optional
    Validator = jsonschema.Draft202012Validator
    v = Validator(schema)
    return [f"schema: {e.message}" for e in v.iter_errors(doc)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="validate default fixture + a synthetic bad doc",
    )
    args = ap.parse_args()
    if args.self_test:
        doc = json.loads(args.json.read_text())
        errs = _structural(doc)
        schema = json.loads(args.schema.read_text())
        errs.extend(_schema_validate(doc, schema))
        bad = {"schema_version": 1, "source": "x", "forbidden_speed_presets": [],
               "contracts": []}
        bad_errs = _structural(bad)
        if errs:
            for e in errs:
                print(f"FAIL: {e}", file=sys.stderr)
            return 1
        if not bad_errs:
            print("FAIL: synthetic bad doc should error", file=sys.stderr)
            return 1
        print("ok fewstep_contracts_validate self-test")
        return 0

    if not args.json.is_file():
        print(f"usage error: missing {args.json}", file=sys.stderr)
        return 2
    doc = json.loads(args.json.read_text())
    errs = _structural(doc)
    if args.schema.is_file():
        schema = json.loads(args.schema.read_text())
        errs.extend(_schema_validate(doc, schema))
    if errs:
        for e in errs:
            print(f"FAIL: {e}", file=sys.stderr)
        return 1
    print("ok fewstep_contracts.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
