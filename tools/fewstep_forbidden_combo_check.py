#!/usr/bin/env python3
"""Offline sanity: FAIL generate.log patterns that mix turbo4/taomate with fewstep.

Catches wrong-contract attempts that may not have a clean fewstep_bound line
(partial logs, older binaries, shell transcripts). Complements
p0_protocol_check_log.py (which already rejects speed_preset=turbo4 on bound).

FAIL when ANY of:
  - fewstep_bound speed_preset in {turbo4, taomate, fast4}
  - fewstep_contract: ... preset=turbo4|taomate|fast4  AND no hard-reject line
  - both a successful speed preset banner AND fewstep_preset:/fewstep_bound:
  - argv / command echo containing --preset turbo4|taomate AND --fewstep-

PASS (correct refusal) when the only turbo4×fewstep evidence is an error line
such as "拒绝 --preset turbo4" / "互斥" / "fewstep_contract=... 拒绝".

Exit 0 = clean; 1 = forbidden combo; 2 = usage.
See docs/research/2026-10-26-fewstep-forbidden-combo.md
"""
from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

_FORBIDDEN_PRESETS = ("turbo4", "taomate", "fast4")
_REJECT_RE = re.compile(
    r"(拒绝\s*--preset|互斥|fewstep_contract=\S+\s+拒绝|"
    r"fewstep-preset .* 与 --preset|会改写 sigma)",
    re.I,
)
_BOUND_RE = re.compile(
    r"^fewstep_bound:.*\bspeed_preset=(\S+)", re.M
)
_CONTRACT_PRESET_RE = re.compile(
    r"^fewstep_contract:.*\bpreset=(turbo4|taomate|fast4)\b", re.M
)
_FEWSTEP_LINE_RE = re.compile(
    r"^(fewstep_preset:|fewstep_bound:|fewstep_contract:)", re.M
)
_SPEED_BANNER_RE = re.compile(
    r"(?:^preset:\s*(turbo4|taomate|fast4)\b)|"
    r"(?:^h3_generate_preset:\s*id=(turbo4|taomate)\b)|"
    r"(?:--preset\s+(turbo4|taomate|fast4)\b)",
    re.M | re.I,
)
_ARGV_BOTH_RE = re.compile(
    r"(?:--preset\s+(?:turbo4|taomate|fast4).{0,200}--fewstep-(?:preset|contract))"
    r"|(?:--fewstep-(?:preset|contract).{0,200}--preset\s+(?:turbo4|taomate|fast4))",
    re.I | re.S,
)


def check(text: str) -> list[str]:
    errs: list[str] = []
    has_reject = bool(_REJECT_RE.search(text))

    for m in _BOUND_RE.finditer(text):
        sp = m.group(1)
        if sp in _FORBIDDEN_PRESETS:
            errs.append(
                f"fewstep_bound speed_preset={sp!r} "
                "(forbidden sigma-rewriting preset)"
            )

    if _CONTRACT_PRESET_RE.search(text) and not has_reject:
        errs.append(
            "fewstep_contract printed with preset=turbo4|taomate|fast4 "
            "and no reject line (bad combo reached check)"
        )

    if _FEWSTEP_LINE_RE.search(text) and _SPEED_BANNER_RE.search(text):
        if not has_reject:
            errs.append(
                "fewstep_* lines co-exist with turbo4/taomate/fast4 preset "
                "banner/argv and no reject line"
            )

    if _ARGV_BOTH_RE.search(text) and not has_reject:
        # Transcript that shows both flags without a refusal — still flag.
        # If refuse line present, OK.
        errs.append(
            "command/transcript has both --preset turbo4|taomate|fast4 "
            "and --fewstep-* without reject line"
        )

    return errs


def _self_test() -> int:
    clean = (
        "fewstep_preset: id=pdmd4 steps=4 cfg=1 video_shift=12 audio_shift=3 "
        "sampler=velocity_euler contract=pdmd4\n"
        "fewstep_bound: contract=pdmd4 steps=4 cfg=1 video_shift=12 "
        "audio_shift=3 sampler=velocity_euler speed_preset=none\n"
        "fewstep_contract: id=pdmd4 steps=4 sampler=velocity_euler "
        "video_shift=12 audio_shift=3 cfg=1 preset=none nvfp4=0 lora=x\n"
    )
    assert not check(clean), check(clean)

    bad_bound = clean.replace("speed_preset=none", "speed_preset=turbo4")
    assert check(bad_bound), "expected FAIL on speed_preset=turbo4"

    refused = (
        "h3: --fewstep-preset pdmd4 与 --preset turbo4 互斥"
        "（会改写 sigma；请无 speed preset + serving shift）\n"
    )
    assert not check(refused), "refuse line alone should PASS"

    mixed_refuse = (
        "--preset turbo4 --fewstep-preset pdmd4\n" + refused
    )
    assert not check(mixed_refuse), "argv+refuse should PASS"

    mixed_bad = "--preset turbo4 --fewstep-preset pdmd4\nout/generate.log\n"
    assert check(mixed_bad), "argv both without refuse should FAIL"

    contract_turbo = (
        "fewstep_contract: id=pdmd4 steps=4 sampler=velocity_euler "
        "video_shift=12 audio_shift=3 cfg=1 preset=turbo4 nvfp4=0 lora=x\n"
    )
    assert check(contract_turbo), "contract+turbo4 without refuse FAIL"
    assert not check(
        contract_turbo + "fewstep_contract=pdmd4 拒绝 --preset turbo4\n"
    )

    print("ok fewstep_forbidden_combo_check self-test")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", type=Path, nargs="?", help="generate.log or transcript")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return _self_test()
    if args.log is None:
        ap.error("log required unless --self-test")
    if not args.log.is_file():
        print(f"usage error: not a file: {args.log}", file=sys.stderr)
        return 2
    text = args.log.read_text(errors="replace")
    errs = check(text)
    if errs:
        for e in errs:
            print(f"FAIL: {e}", file=sys.stderr)
        print("FORBIDDEN_COMBO", file=sys.stderr)
        return 1
    print("COMBO_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
