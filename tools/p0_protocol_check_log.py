#!/usr/bin/env python3
"""Offline protocol checker for P0 INT8 vs NVFP4 visual-gate generate.log files.

Does not run the model. Exit 0 = protocol OK; 1 = fail; 2 = usage error.
See docs/research/2026-10-26-p0-visual-gate-runbook.md.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def _read(path: Path) -> str:
    return path.read_text(errors="replace")


def check(text: str, arm: str) -> list[str]:
    errs: list[str] = []
    warnings: list[str] = []

    def need(pat: str, label: str) -> re.Match | None:
        m = re.search(pat, text, re.M)
        if not m:
            errs.append(f"missing: {label} (/{pat}/)")
        return m

    # Graph must be off
    m = re.search(r"sdpa_graph:\s*wanted=(\d+)", text)
    if not m:
        errs.append("missing: sdpa_graph: wanted=N")
    elif m.group(1) != "0":
        errs.append(f"sdpa_graph wanted={m.group(1)} (need 0 for P0)")

    # Chain must be off
    if re.search(r"H3_DIT_NVFP4_CHAIN\s*=\s*1", text) or re.search(
        r"dit_nvfp4:.*CHAIN=1", text
    ):
        errs.append("NVFP4 chain appears enabled (P0 forbids H3_DIT_NVFP4_CHAIN)")

    # Sigma / shift printed (discipline)
    if not re.search(r"sigma\[0\.\.", text):
        errs.append("missing: sigma[0..N] schedule print")
    if not re.search(r"adaln_dual_sigma:.*video_shift=", text):
        errs.append("missing: adaln_dual_sigma video_shift print")

    if arm == "int8":
        if re.search(r"dit_nvfp4:\s*H3_DIT_NVFP4=1", text):
            errs.append("INT8 arm log contains dit_nvfp4: H3_DIT_NVFP4=1")
        if not re.search(r"dit_format:\s*INT8", text):
            # soft: some builds may omit; warn
            warnings.append("no dit_format: INT8 line (confirm arm manually)")
    elif arm == "nvfp4":
        if not re.search(r"dit_nvfp4:\s*H3_DIT_NVFP4=1", text):
            errs.append("NVFP4 arm missing dit_nvfp4: H3_DIT_NVFP4=1")
        if not re.search(r"dit_nvfp4:\s*sidecar=", text):
            warnings.append("no dit_nvfp4: sidecar= line (confirm H3_DIT_NVFP4_FILE)")
    else:
        errs.append(f"unknown --arm {arm!r} (want int8|nvfp4)")

    # Optional RUN_RC footer
    mrc = re.search(r"^RUN_RC=(\d+)\s*$", text, re.M)
    if mrc and mrc.group(1) != "0":
        errs.append(f"RUN_RC={mrc.group(1)}")

    # phase_ms presence (e2e completeness hint)
    if not re.search(r"phase_ms:", text):
        warnings.append("missing phase_ms (run may have aborted before summary)")

    for w in warnings:
        print(f"WARN: {w}", file=sys.stderr)
    return errs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", type=Path, help="path to generate.log")
    ap.add_argument(
        "--arm",
        required=True,
        choices=("int8", "nvfp4"),
        help="expected DiT arm",
    )
    args = ap.parse_args()
    if not args.log.is_file():
        print(f"usage error: not a file: {args.log}", file=sys.stderr)
        return 2
    errs = check(_read(args.log), args.arm)
    if errs:
        for e in errs:
            print(f"FAIL: {e}", file=sys.stderr)
        print("PROTO_FAIL", file=sys.stderr)
        return 1
    print("PROTO_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
