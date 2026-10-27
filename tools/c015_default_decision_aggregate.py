#!/usr/bin/env python3
"""Aggregate C015 default-on decision dig arms -> DECISION_REPORT.md + RECOMMEND.txt"""
from __future__ import annotations
import re, subprocess, sys
from pathlib import Path

def peak_gib(log: Path):
    t = log.read_text(encoding="utf-8", errors="replace")
    vals = [float(m.group(1)) / 1024.0 for m in re.finditer(
        r"vram_ledger:\s*point=peak_dit\b[^\n]*?used_mib=([0-9.]+)", t)]
    return max(vals) if vals else None

def warm_dit(log: Path):
    t = log.read_text(encoding="utf-8", errors="replace")
    vals = re.findall(r"^phase_ms:.*?\bdit=([0-9.]+)", t, re.M)
    return float(vals[-1]) if vals else None

def pack_flags(log: Path):
    t = log.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"pack qkv_f16=(\d+) wide_f16=(\d+)", t)
    return (m.group(1), m.group(2)) if m else ("?", "?")

def fmt4(x):
    return f"{x:.4f}" if x is not None else "NA"

def fmtn(x):
    if x is None:
        return "NA"
    if isinstance(x, float):
        return f"{x:.1f}"
    return str(x)

def main():
    out, root, sha = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    gate = root / "tools" / "waffle_same_config_gate.py"
    pairs = []
    for off in sorted(out.iterdir()):
        if not off.is_dir() or not off.name.endswith("_off"):
            continue
        on = out / (off.name[:-4] + "_on")
        if on.is_dir():
            pairs.append((off.name, off, on))

    lines = [
        "# C015 default-on decision dig — raw report",
        "",
        f"git_sha_synced: `{sha}`",
        f"out: `{out}`",
        "",
        "| pair | waffle | mean_abs | max_abs | peak_off | peak_on | drop | warm_dit_off | warm_dit_on | d_dit_ms | pack qkv/wide |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    waffle_all = True
    wc_ok = True
    euler_drops, dmad_drops = [], []

    for off_n, off_d, on_d in pairs:
        blog, clog = off_d / "generate.log", on_d / "generate.log"
        wr = subprocess.run(
            ["python3", str(gate), "--mean-tol", "0.15", "--max-tol", "0.40",
             "--require-frames", "5", str(blog), str(clog)],
            capture_output=True, text=True)
        wtxt = wr.stdout + wr.stderr
        (out / f"waffle_{off_n}_vs_on.txt").write_text(wtxt)
        mean_m = re.search(r"mean_abs_delta=([0-9.]+)", wtxt)
        max_m = re.search(r"max_abs_delta=([0-9.]+)", wtxt)
        mean_v = mean_m.group(1) if mean_m else "NA"
        max_v = max_m.group(1) if max_m else "NA"
        wpass = wr.returncode == 0
        if not wpass:
            waffle_all = False
        pb, pc = peak_gib(blog), peak_gib(clog)
        drop = (pb - pc) if (pb is not None and pc is not None) else None
        if drop is not None:
            (euler_drops if off_n.startswith("euler_") else dmad_drops).append(drop)
        db, dc = warm_dit(blog), warm_dit(clog)
        dd = (dc - db) if (db is not None and dc is not None) else None
        if dd is not None and db and dd > max(500.0, 0.03 * db):
            wc_ok = False
        qkv, wide = pack_flags(clog)
        lines.append(
            f"| `{off_n}` | {'PASS' if wpass else 'FAIL'} | {mean_v} | {max_v} | "
            f"{fmt4(pb)} | {fmt4(pc)} | {fmt4(drop)} | "
            f"{fmtn(db)} | {fmtn(dc)} | {fmtn(dd)} | {qkv}/{wide} |"
        )

    euler_vram_ok = bool(euler_drops) and min(euler_drops) >= 0.5
    dmad_vram_ok = (not dmad_drops) or (min(dmad_drops) >= 0.25)
    recommend = "YES" if (waffle_all and wc_ok and euler_vram_ok) else "NO"
    reasons = []
    if not waffle_all:
        reasons.append("waffle FAIL on >=1 pair")
    if not wc_ok:
        reasons.append("warm dit regression >3% and >500ms")
    if not euler_vram_ok:
        reasons.append("Euler peak drop <0.5 GiB")
    if dmad_drops and not dmad_vram_ok:
        reasons.append("DMAD LoRA peak drop <0.25 GiB")

    lines += [
        "",
        "## PASS table",
        "",
        "| criterion | result |",
        "|---|---|",
        f"| multi-prompt waffle all PASS | {'PASS' if waffle_all else 'FAIL'} |",
        f"| warm dit no material regression | {'PASS' if wc_ok else 'FAIL'} |",
        f"| Euler no-LoRA peak drop >=0.5 GiB | {'PASS' if euler_vram_ok else 'FAIL'} |",
        f"| DMAD LoRA peak drop >=0.25 GiB (qkv-only) | {'PASS' if dmad_vram_ok else 'FAIL/NA'} |",
        f"| **recommend default ON** | **{recommend}** |",
        "",
        f"reasons: {'; '.join(reasons) if reasons else 'all gates green'}",
    ]
    (out / "DECISION_REPORT.md").write_text("\n".join(lines) + "\n")
    (out / "RECOMMEND.txt").write_text(recommend + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
