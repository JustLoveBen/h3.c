#!/usr/bin/env python3
"""Offline protocol checker for P0 INT8 vs NVFP4 visual-gate generate.log files.

Also recognizes T3 fewstep log lines from --fewstep-preset / --fewstep-contract:
  fewstep_preset: ...
  fewstep_contract: ...
  fewstep_bound: ...
  sampler: selected=...

Does not run the model. Exit 0 = protocol OK; 1 = fail; 2 = usage error.
See docs/research/2026-10-26-p0-visual-gate-runbook.md and
docs/research/2026-10-26-fewstep-preset.md.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_FEWSTEP_STEPS = {"pdmd4": 4, "pdmd2": 2, "dmad4": 4}
_FEWSTEP_SAMPLER = {
    "pdmd4": "velocity_euler",
    "pdmd2": "velocity_euler",
    "dmad4": "lcm_renoise",
}
_FEWSTEP_AUDIO = {"pdmd4": 3.0, "pdmd2": 3.0, "dmad4": 2.0}


def _read(path: Path) -> str:
    return path.read_text(errors="replace")


def _parse_float(s: str) -> float | None:
    try:
        return float(s)
    except ValueError:
        return None


def check_fewstep(text: str, expect: str | None) -> tuple[list[str], list[str]]:
    """Validate fewstep_* / sampler lines.

    expect:
      None  — if any fewstep lines present, soft-check consistency
      none  — require absence of fewstep_preset / fewstep_bound
      pdmd4|pdmd2|dmad4 — require matching preset/bound/contract/sampler
    """
    errs: list[str] = []
    warnings: list[str] = []

    m_preset = re.search(
        r"^fewstep_preset:\s*id=(\S+)\s+steps=(\d+)\s+cfg=([0-9.eE+-]+)\s+"
        r"video_shift=([0-9.eE+-]+)\s+audio_shift=([0-9.eE+-]+)\s+"
        r"sampler=(\S+)\s+contract=(\S+)\s*$",
        text,
        re.M,
    )
    m_bound = re.search(
        r"^fewstep_bound:\s*contract=(\S+)\s+steps=(\d+)\s+cfg=([0-9.eE+-]+)\s+"
        r"video_shift=([0-9.eE+-]+)\s+audio_shift=([0-9.eE+-]+)\s+"
        r"sampler=(\S+)\s+speed_preset=(\S+)\s*$",
        text,
        re.M,
    )
    m_contract = re.search(
        r"^fewstep_contract:\s*id=(\S+)\s+steps=(\d+)\s+sampler=(\S+)\s+"
        r"video_shift=([0-9.eE+-]+)\s+audio_shift=([0-9.eE+-]+)",
        text,
        re.M,
    )
    m_sampler = re.search(
        r"^sampler:\s*selected=(\S+)\s+\(explicit=(\d+)\s+fewstep=(\S+)\)\s*$",
        text,
        re.M,
    )
    m_sigma_ok = re.search(r"^fewstep_contract:\s*sigma_ok\b", text, re.M)

    any_few = bool(m_preset or m_bound or m_contract)

    if expect == "none":
        if m_preset or m_bound:
            errs.append(
                "fewstep_preset/fewstep_bound present but --expect-fewstep=none"
            )
        return errs, warnings

    if expect in _FEWSTEP_STEPS:
        want = expect
        if not m_bound and not m_preset:
            errs.append(
                f"missing fewstep_preset / fewstep_bound for --expect-fewstep={want}"
            )
        if not m_contract:
            errs.append(f"missing fewstep_contract: id=... for {want}")
        if not m_sampler:
            errs.append(f"missing sampler: selected=... fewstep=... for {want}")

        def _id_ok(got: str | None, label: str) -> None:
            if got is None:
                return
            if got != want:
                errs.append(f"{label} id/contract={got!r} (want {want})")

        if m_preset:
            _id_ok(m_preset.group(1), "fewstep_preset")
            _id_ok(m_preset.group(7), "fewstep_preset.contract")
            steps = int(m_preset.group(2))
            if steps != _FEWSTEP_STEPS[want]:
                errs.append(
                    f"fewstep_preset steps={steps} (want {_FEWSTEP_STEPS[want]})"
                )
            samp = m_preset.group(6)
            if samp != _FEWSTEP_SAMPLER[want]:
                errs.append(
                    f"fewstep_preset sampler={samp!r} "
                    f"(want {_FEWSTEP_SAMPLER[want]})"
                )
            a = _parse_float(m_preset.group(5))
            v = _parse_float(m_preset.group(4))
            if v is not None and abs(v - 12.0) > 0.051:
                errs.append(f"fewstep_preset video_shift={v} (want 12)")
            if a is not None and abs(a - _FEWSTEP_AUDIO[want]) > 0.051:
                errs.append(
                    f"fewstep_preset audio_shift={a} "
                    f"(want {_FEWSTEP_AUDIO[want]})"
                )

        if m_bound:
            _id_ok(m_bound.group(1), "fewstep_bound")
            steps = int(m_bound.group(2))
            if steps != _FEWSTEP_STEPS[want]:
                errs.append(
                    f"fewstep_bound steps={steps} (want {_FEWSTEP_STEPS[want]})"
                )
            samp = m_bound.group(6)
            if samp != _FEWSTEP_SAMPLER[want]:
                errs.append(
                    f"fewstep_bound sampler={samp!r} "
                    f"(want {_FEWSTEP_SAMPLER[want]})"
                )
            a = _parse_float(m_bound.group(5))
            v = _parse_float(m_bound.group(4))
            if v is not None and abs(v - 12.0) > 0.051:
                errs.append(f"fewstep_bound video_shift={v} (want 12)")
            if a is not None and abs(a - _FEWSTEP_AUDIO[want]) > 0.051:
                errs.append(
                    f"fewstep_bound audio_shift={a} "
                    f"(want {_FEWSTEP_AUDIO[want]})"
                )
            speed = m_bound.group(7)
            if speed in ("turbo4", "taomate", "fast4"):
                errs.append(
                    f"fewstep_bound speed_preset={speed!r} "
                    "(fewstep forbids sigma-rewriting presets)"
                )

        if m_contract:
            _id_ok(m_contract.group(1), "fewstep_contract")
            # contract print uses "lcm+re-noise" vs sampler name "lcm_renoise"
            csamp = m_contract.group(3)
            want_c = (
                "lcm+re-noise"
                if want == "dmad4"
                else "velocity_euler"
            )
            if csamp != want_c and csamp != _FEWSTEP_SAMPLER[want]:
                errs.append(
                    f"fewstep_contract sampler={csamp!r} (want {want_c})"
                )

        if m_sampler:
            sel = m_sampler.group(1)
            fs = m_sampler.group(3)
            if fs != want:
                errs.append(f"sampler fewstep={fs!r} (want {want})")
            if sel != _FEWSTEP_SAMPLER[want]:
                errs.append(
                    f"sampler selected={sel!r} "
                    f"(want {_FEWSTEP_SAMPLER[want]})"
                )

        if not m_sigma_ok:
            warnings.append(
                "no fewstep_contract: sigma_ok (run may have failed before "
                "schedule print, or older binary)"
            )

        if re.search(r"dit_nvfp4:\s*H3_DIT_NVFP4=1", text):
            errs.append(
                "fewstep arm log has H3_DIT_NVFP4=1 "
                "(cross-arm forbidden unless H3_FEWSTEP_ALLOW_NVFP4)"
            )

        return errs, warnings

    # expect is None: soft recognition when lines present
    if any_few:
        if m_bound and m_preset:
            if m_bound.group(1) != m_preset.group(1):
                errs.append(
                    f"fewstep_bound contract={m_bound.group(1)!r} != "
                    f"fewstep_preset id={m_preset.group(1)!r}"
                )
        if m_bound and m_sampler:
            if m_bound.group(1) != m_sampler.group(3):
                warnings.append(
                    f"fewstep_bound contract={m_bound.group(1)!r} vs "
                    f"sampler fewstep={m_sampler.group(3)!r}"
                )
        if m_bound:
            speed = m_bound.group(7)
            if speed in ("turbo4", "taomate", "fast4"):
                errs.append(
                    f"fewstep_bound speed_preset={speed!r} "
                    "(fewstep forbids sigma-rewriting presets)"
                )
        if m_preset and not m_bound:
            warnings.append("fewstep_preset without fewstep_bound")
        if m_bound and not m_contract:
            warnings.append("fewstep_bound without fewstep_contract print")
    return errs, warnings


def check(text: str, arm: str, expect_fewstep: str | None) -> list[str]:
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
            warnings.append("no dit_format: INT8 line (confirm arm manually)")
    elif arm == "nvfp4":
        if not re.search(r"dit_nvfp4:\s*H3_DIT_NVFP4=1", text):
            errs.append("NVFP4 arm missing dit_nvfp4: H3_DIT_NVFP4=1")
        if not re.search(r"dit_nvfp4:\s*sidecar=", text):
            warnings.append("no dit_nvfp4: sidecar= line (confirm H3_DIT_NVFP4_FILE)")
    elif arm == "fewstep":
        # DiT arm not asserted; fewstep contract is the gate.
        if expect_fewstep is None or expect_fewstep == "none":
            errs.append(
                "--arm fewstep requires --expect-fewstep pdmd4|pdmd2|dmad4"
            )
    else:
        errs.append(f"unknown --arm {arm!r} (want int8|nvfp4|fewstep)")

    # P0 arms: warn if T3 fewstep lines leak in (arm isolation)
    if arm in ("int8", "nvfp4") and expect_fewstep is None:
        if re.search(r"^fewstep_bound:", text, re.M) or re.search(
            r"^fewstep_preset:", text, re.M
        ):
            warnings.append(
                "fewstep_preset/bound present on P0 arm log "
                "(量化臂与少步臂应隔离；用 --expect-fewstep 显式验收 T3)"
            )

    fe, fw = check_fewstep(text, expect_fewstep)
    errs.extend(fe)
    warnings.extend(fw)

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


def _self_test() -> int:
    """Host-only fixture checks (no GPU)."""
    base = """sdpa_graph: wanted=0
sigma[0..4]=1.000 ...
adaln_dual_sigma: video_shift=12 audio_shift=3
dit_format: INT8
phase_ms: total=1
RUN_RC=0
"""
    # P0 INT8 clean
    e = check(base, "int8", None)
    assert not e, e

    # fewstep pdmd4 happy path
    pdmd = base + """
fewstep_preset: id=pdmd4 steps=4 cfg=1 video_shift=12 audio_shift=3 sampler=velocity_euler contract=pdmd4
fewstep_contract: id=pdmd4 steps=4 sampler=velocity_euler video_shift=12 audio_shift=3 cfg=1 preset=none nvfp4=0 lora=(none)
sampler: selected=velocity_euler (explicit=0 fewstep=pdmd4)
fewstep_bound: contract=pdmd4 steps=4 cfg=1 video_shift=12 audio_shift=3 sampler=velocity_euler speed_preset=none
fewstep_contract: sigma_ok student_grid(nfe=4, video_shift=12, audio_shift=3)
"""
    e = check(pdmd, "fewstep", "pdmd4")
    assert not e, e

    # dmad4 wants lcm_renoise
    dmad_bad = base + """
fewstep_preset: id=dmad4 steps=4 cfg=1 video_shift=12 audio_shift=2 sampler=velocity_euler contract=dmad4
fewstep_contract: id=dmad4 steps=4 sampler=velocity_euler video_shift=12 audio_shift=2 cfg=1 preset=none nvfp4=0 lora=x
sampler: selected=velocity_euler (explicit=0 fewstep=dmad4)
fewstep_bound: contract=dmad4 steps=4 cfg=1 video_shift=12 audio_shift=2 sampler=velocity_euler speed_preset=none
"""
    e = check(dmad_bad, "fewstep", "dmad4")
    assert any("sampler" in x for x in e), e

    dmad_ok = base + """
fewstep_preset: id=dmad4 steps=4 cfg=1 video_shift=12 audio_shift=2 sampler=lcm_renoise contract=dmad4
fewstep_contract: id=dmad4 steps=4 sampler=lcm+re-noise video_shift=12 audio_shift=2 cfg=1 preset=none nvfp4=0 lora=x
sampler: selected=lcm_renoise (explicit=0 fewstep=dmad4)
fewstep_bound: contract=dmad4 steps=4 cfg=1 video_shift=12 audio_shift=2 sampler=lcm_renoise speed_preset=none
fewstep_contract: sigma_ok student_grid(nfe=4, video_shift=12, audio_shift=2)
"""
    e = check(dmad_ok, "fewstep", "dmad4")
    assert not e, e

    # turbo4 + fewstep must fail
    turbo = pdmd.replace("speed_preset=none", "speed_preset=turbo4")
    e = check(turbo, "fewstep", "pdmd4")
    assert any("turbo4" in x for x in e), e

    # expect none rejects bound
    e = check(pdmd, "int8", "none")
    assert any("expect-fewstep=none" in x for x in e), e

    print("ok p0_protocol_check_log self-test")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", type=Path, nargs="?", help="path to generate.log")
    ap.add_argument(
        "--arm",
        choices=("int8", "nvfp4", "fewstep"),
        help="expected DiT / acceptance arm",
    )
    ap.add_argument(
        "--expect-fewstep",
        choices=("none", "pdmd4", "pdmd2", "dmad4"),
        default=None,
        help="require/forbid fewstep_preset+bound+sampler lines "
        "(T3 offline check; default: soft-recognize if present)",
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="run host fixtures and exit (no log file)",
    )
    args = ap.parse_args()
    if args.self_test:
        return _self_test()
    if args.log is None or args.arm is None:
        ap.error("log and --arm are required unless --self-test")
    if not args.log.is_file():
        print(f"usage error: not a file: {args.log}", file=sys.stderr)
        return 2
    errs = check(_read(args.log), args.arm, args.expect_fewstep)
    if errs:
        for e in errs:
            print(f"FAIL: {e}", file=sys.stderr)
        print("PROTO_FAIL", file=sys.stderr)
        return 1
    print("PROTO_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
