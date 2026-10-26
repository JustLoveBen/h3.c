#!/usr/bin/env python3
"""Offline checker for C073 dump-once / score-many artifacts.

Validates an H3LAT1 latent dump header, and optionally greps a generate.log
for dump_once_score_many / # H3_DUMP_ONCE / # H3_SCORE_MANY markers.

Usage:
  python3 tools/dump_once_score_many_check.py /path/to/latent.h3lat
  python3 tools/dump_once_score_many_check.py --log generate.log
  python3 tools/dump_once_score_many_check.py --self-test
"""
from __future__ import annotations

import argparse
import re
import struct
import sys
from pathlib import Path

HDR_N = 64
MAGIC = b"H3LAT1"


def check_latent(path: Path) -> list[str]:
    errs: list[str] = []
    data = path.read_bytes()
    if len(data) < HDR_N:
        return [f"file too short ({len(data)} < {HDR_N})"]
    hdr = data[:HDR_N]
    try:
        text = hdr.split(b"\0", 1)[0].decode("ascii", errors="replace").strip()
    except Exception as e:  # noqa: BLE001
        return [f"header decode: {e}"]
    parts = text.split()
    if len(parts) < 5 or parts[0] != "H3LAT1":
        errs.append(f"bad magic/header: {text!r}")
        return errs
    try:
        t, h, w, n = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
    except ValueError:
        errs.append(f"bad ints in header: {text!r}")
        return errs
    expect_n = t * (h // 2) * (w // 2) * 96
    if h < 2 or w < 2 or (h % 2) or (w % 2):
        errs.append(f"geometry h,w must be even >=2; got {h}x{w}")
    if n != expect_n:
        errs.append(f"n={n} != t*(h/2)*(w/2)*96={expect_n}")
    body = len(data) - HDR_N
    if body != n * 4:
        errs.append(f"payload bytes={body} want n*4={n * 4}")
    if not errs:
        print(f"OK H3LAT1 t={t} h={h} w={w} n={n} bytes={len(data)}")
    return errs


def check_log(text: str) -> list[str]:
    errs: list[str] = []
    has_dump = bool(re.search(r"^# H3_DUMP_ONCE\b", text, re.M))
    has_score = bool(re.search(r"^# H3_SCORE_MANY\b", text, re.M))
    has_banner = bool(re.search(r"^dump_once_score_many:", text, re.M))
    if not (has_dump or has_score or has_banner):
        errs.append(
            "no dump_once_score_many / # H3_DUMP_ONCE / # H3_SCORE_MANY markers"
        )
    if "do_not_retune_gold=1" not in text and has_banner:
        errs.append("dump_once_score_many banner missing do_not_retune_gold=1")
    if has_dump or has_score or has_banner:
        print(
            f"OK log markers dump={int(has_dump)} score={int(has_score)} "
            f"banner={int(has_banner)}"
        )
    return errs


def _self_test() -> int:
    import tempfile

    t, h, w = 5, 30, 52
    n = t * (h // 2) * (w // 2) * 96
    hdr = f"H3LAT1 {t} {h} {w} {n}\n".encode("ascii")
    hdr = hdr + b"\0" * (HDR_N - len(hdr))
    payload = b"\0" * (n * 4)
    with tempfile.NamedTemporaryFile(suffix=".h3lat", delete=False) as f:
        f.write(hdr + payload)
        path = Path(f.name)
    try:
        e = check_latent(path)
        assert not e, e
    finally:
        path.unlink(missing_ok=True)

    log = (
        "dump_latent: clip=1 wrote 1 f32 (5x30x52) checksum=0x1 → /tmp/x\n"
        "# H3_DUMP_ONCE path=/tmp/x tokens_n/a checksum=0x1 shape=5x30x52 "
        "layout=THW96\n"
        "dump_once_score_many: mode=dump path=/tmp/x do_not_retune_gold=1\n"
    )
    e = check_log(log)
    assert not e, e
    e = check_log("no markers here\n")
    assert e, e
    print("ok dump_once_score_many_check self-test")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("latent", type=Path, nargs="?", help="H3LAT1 dump path")
    ap.add_argument("--log", type=Path, help="generate.log to scan")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return _self_test()
    if args.latent is None and args.log is None:
        ap.error("latent and/or --log required (or --self-test)")
    errs: list[str] = []
    if args.latent is not None:
        if not args.latent.is_file():
            print(f"usage error: not a file: {args.latent}", file=sys.stderr)
            return 2
        errs.extend(check_latent(args.latent))
    if args.log is not None:
        if not args.log.is_file():
            print(f"usage error: not a file: {args.log}", file=sys.stderr)
            return 2
        errs.extend(check_log(args.log.read_text(errors="replace")))
    if errs:
        for e in errs:
            print(f"FAIL: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
