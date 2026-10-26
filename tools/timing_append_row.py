#!/usr/bin/env python3
"""Extract one markdown timing-ledger row from a generate.log (host-only).

Appends nothing to the ledger by itself — prints a table row for humans to
paste into docs/research/2026-10-09-generation-timing.md after a protocol-OK
run. Does not invent acceleration ratios.

Usage:
  python3 tools/timing_append_row.py --arm INT8 --id F4 path/to/generate.log
  python3 tools/timing_append_row.py --self-test
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def parse(text: str) -> dict:
    out: dict = {}
    m = re.search(
        r"^shape:.*?(\d+)x(\d+).*?(\d+)s\b.*?steps=(\d+).*?\bsdpa=(\S+)"
        r".*?\bseed=(0x[0-9a-fA-F]+)",
        text,
        re.M,
    )
    if m:
        out["width"], out["height"] = m.group(1), m.group(2)
        out["seconds"], out["steps"] = m.group(3), m.group(4)
        out["sdpa"], out["seed"] = m.group(5), m.group(6)
    m = re.search(r"sdpa_graph:\s*wanted=(\d+)", text)
    out["graph"] = m.group(1) if m else "?"
    out["nvfp4"] = bool(re.search(r"dit_nvfp4:\s*H3_DIT_NVFP4=1", text))
    out["chain"] = bool(
        re.search(r"H3_DIT_NVFP4_CHAIN\s*=\s*1", text)
        or re.search(r"FC1->FC2 chained", text)
    )
    m = re.search(
        r"^phase_ms:\s*te=([0-9.]+)\s+dit=([0-9.]+).*?\s+vae=([0-9.]+)"
        r".*?\s+audio=([0-9.]+).*?\s+mux=([0-9.]+).*?\s+total=([0-9.]+)",
        text,
        re.M,
    )
    if m:
        out["te"], out["dit"], out["vae"] = m.group(1), m.group(2), m.group(3)
        out["audio"], out["mux"], out["total"] = (
            m.group(4),
            m.group(5),
            m.group(6),
        )
    m = re.search(r"^RUN_RC=(\d+)\s*$", text, re.M)
    out["rc"] = m.group(1) if m else "?"
    out["skip_audio"] = (
        "audio" in out
        and float(out["audio"]) <= 1.0
        and not re.search(r"adaln_dual_sigma:", text)
    )
    return out


def row(d: dict, arm: str, rid: str, log_label: str) -> str:
    chain = "开" if d.get("chain") else "关"
    res = (
        f"{d.get('width','?')}×{d.get('height','?')}"
        if "width" in d
        else "?"
    )
    phase = (
        f"{d.get('te','?')} / {d.get('dit','?')} / {d.get('vae','?')} / "
        f"{d.get('mux','?')} / {d.get('total','?')}"
        if "te" in d
        else "无 phase_ms"
    )
    return (
        f"| {rid} | {log_label} | {arm} | {res} | {d.get('seconds','?')} | "
        f"{d.get('steps','?')} | `{d.get('seed','?')}` | wanted={d.get('graph','?')} | "
        f"{chain} | {phase} | — | {d.get('rc','?')} | "
        f"{'skip-audio' if d.get('skip_audio') else 'audio-on'}; paste after PROTO_OK |"
    )


def _self_test() -> int:
    sample = """
sdpa_graph: wanted=0
shape: 864x480 5s → latent 1x30x54 steps=8 sdpa=cudnn attn_window=0 dist_attn=off overlap=0 cfg=1 seed=0x76800002 context_budget=0
phase_ms: te=10.0 dit=20.0 lift_switch=0.0 vae=3.0 vae_load=0.0 vae_blocks=0.0 audio=0.0 ppm=0.0 mux=1.0 total=34.0 dit_evals=8 dit_skips=0 hot=0 clip=1/1
RUN_RC=0
"""
    d = parse(sample)
    assert d["seed"] == "0x76800002" and d["graph"] == "0"
    assert d["te"] == "10.0" and d["dit"] == "20.0"
    r = row(d, "INT8", "T1", "`out/example/generate.log`")
    assert r.startswith("| T1 |") and "wanted=0" in r
    print("ok timing_append_row self-test")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", type=Path, nargs="?", help="generate.log")
    ap.add_argument("--arm", default="INT8", help="arm label for the row")
    ap.add_argument("--id", default="NEW", help="row id (F4, G3, …)")
    ap.add_argument(
        "--label",
        default=None,
        help="date+path cell; default = log path",
    )
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
    d = parse(text)
    label = args.label or f"`{args.log}`"
    print(row(d, args.arm, args.id, label))
    return 0


if __name__ == "__main__":
    sys.exit(main())
