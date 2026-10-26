#!/usr/bin/env python3
"""Host-only FNV-1a64 checksum for dumped TE / text_h f32 blobs.

Matches h3_dit_w4a8_checksum_f32 (same offset basis + prime). Use on
H3_DUMP_DIR outputs (c_te_raw.f32, c_te_refined.f32) to compare live TE
runs without GPU.

Guidance (P0 / c_te_raw debt):
  - Visual gate: live TE (do NOT set H3_TEXT_H_LOAD).
  - Velocity / 1-step md5 diagnostics: dump refined text_h once, then
    H3_TEXT_H_LOAD=<path/to/c_te_refined.f32> so INT8 vs NVFP4 share
    identical conditioning (isolates DiT-side drift from TE nondeterminism).
  - Do not mix live-TE waffle with load-TE waffle into one gold line.

Exit 0 on success; 1 if --expect mismatches or two files differ; 2 usage.
See docs/research/2026-10-26-p0-visual-gate-runbook.md §TE.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3


def checksum_f32(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if len(data) % 4 != 0:
        raise ValueError(f"{path}: size {len(data)} not multiple of 4")
    n = len(data) // 4
    h = _FNV_OFFSET
    for i in range(n):
        (bits,) = struct.unpack_from("<I", data, i * 4)
        h ^= bits
        h = (h * _FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h, n


def _self_test() -> int:
    # Empty → 0 per C (null/empty returns 0); we treat missing as error.
    # Single float 1.0f bits 0x3f800000
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "one.f32"
        p.write_bytes(struct.pack("<f", 1.0))
        h, n = checksum_f32(p)
        assert n == 1
        # hand-rolled
        expect = _FNV_OFFSET
        expect ^= 0x3F800000
        expect = (expect * _FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
        assert h == expect, hex(h)
    print("ok te_f32_checksum self-test")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", type=Path, help="one or two .f32 files")
    ap.add_argument(
        "--expect",
        default=None,
        help="require checksum == this hex (with or without 0x)",
    )
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument(
        "--guide",
        action="store_true",
        help="print H3_TEXT_H_LOAD / live-TE guidance and exit",
    )
    args = ap.parse_args()
    if args.self_test:
        return _self_test()
    if args.guide or not args.paths:
        print(
            "TE path guide:\n"
            "  live TE (visual gate): unset H3_TEXT_H_LOAD\n"
            "  shared refine dump:    H3_TEXT_H_LOAD=/path/c_te_refined.f32\n"
            "  dump once:             H3_DUMP_DIR=... then checksum this tool\n"
            "  compare two dumps:     te_f32_checksum.py a.f32 b.f32\n"
            "Log lines to record: text_h checksum=0x... ; "
            "text_h: override loaded from ..."
        )
        if not args.paths and not args.guide:
            ap.print_usage(sys.stderr)
            return 2
        if args.guide and not args.paths:
            return 0

    results: list[tuple[Path, int, int]] = []
    for p in args.paths:
        if not p.is_file():
            print(f"usage error: not a file: {p}", file=sys.stderr)
            return 2
        h, n = checksum_f32(p)
        results.append((p, h, n))
        print(f"{p}: checksum=0x{h:016x} n={n}")

    if args.expect is not None:
        exp = int(args.expect, 16)
        if results[0][1] != exp:
            print(
                f"FAIL: got 0x{results[0][1]:016x} want 0x{exp:016x}",
                file=sys.stderr,
            )
            return 1
        print("EXPECT_OK")

    if len(results) == 2:
        if results[0][1] != results[1][1] or results[0][2] != results[1][2]:
            print("DIFF", file=sys.stderr)
            return 1
        print("MATCH")
    return 0


if __name__ == "__main__":
    sys.exit(main())
