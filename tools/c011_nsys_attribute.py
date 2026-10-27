#!/usr/bin/env python3
"""Attribute nsys cuda_gpu_kern_sum for C011 cast/prescale residual."""
import csv, re, sys
from pathlib import Path

out = Path(sys.argv[1])
sha = sys.argv[2] if len(sys.argv) > 2 else "unknown"
csv_p = out / "cuda_gpu_kern_sum.csv"
txt_p = out / "cuda_gpu_kern_sum.txt"
log = (out / "generate.log").read_text(encoding="utf-8", errors="replace")
kpi = {
    "vae_gemm": re.search(r"vae_gemm: ([^\n]+)", log),
    "vae_path": re.search(r"vae_gemm_path: ([^\n]+)", log),
    "phase": re.search(r"phase_ms: ([^\n]+)", log),
    "vae_line": re.search(r"^vae: ([^\n]+)", log, re.M),
    "vae_repeat": re.search(r"vae_repeat 1/1: ([^\n]+)", log),
    "fallback": "fallback_cast" in log,
}

rows = []
if csv_p.exists() and csv_p.stat().st_size > 0:
    text = csv_p.read_text(encoding="utf-8", errors="replace").splitlines()
    header_i = next(
        (i for i, l in enumerate(text) if "Name" in l and ("Total Time" in l or "Time" in l)),
        None,
    )
    if header_i is not None:
        rdr = csv.DictReader(text[header_i:])
        for r in rdr:
            name = (r.get("Name") or r.get("Kernel Name") or "").strip()
            tns = None
            for k, v in r.items():
                if not v:
                    continue
                if "Total Time" in (k or ""):
                    try:
                        tns = float(str(v).replace(",", ""))
                    except ValueError:
                        pass
            if tns is None:
                for k, v in r.items():
                    if k in ("Name", "Kernel Name"):
                        continue
                    try:
                        tns = float(str(v).replace(",", ""))
                        break
                    except ValueError:
                        pass
            if name and tns is not None:
                rows.append((tns, name))

if not rows and txt_p.exists():
    for line in txt_p.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"\s*([\d.]+)\s+([\d,]+)\s+(\d+)\s+.*?(\S.+)$", line)
        if m:
            name = m.group(4).strip()
            if name and not name.startswith("Time") and "----" not in name:
                rows.append((float(m.group(2).replace(",", "")), name))

rows.sort(reverse=True)

CAST_RE = re.compile(
    r"f32_to_f16|f16_to_f32|f16_to_bf16|bf16_to_f32|row_prescale|"
    r"h3_cuda_f32_to_f16|h3_cuda_f16_to_f32|h3_f16_to_bf16|h3_bf16_to_f32|"
    r"h3_vae_rms_f32_to_f16|h3_f16_bhtd_to_f32|h3_f16_thd_to_bhtd",
    re.I,
)
SGEMM_RE = re.compile(r"sgemm|gemm_32f|ampere_sgemm|volta_sgemm|tf32", re.I)
VAE_OWN_RE = re.compile(r"h3_vae_|vae_", re.I)
PURE_RE = re.compile(
    r"h3_cuda_f32_to_f16_kernel|h3_cuda_f16_to_f32_kernel|h3_cuda_f32_to_f16_3|"
    r"h3_f16_to_bf16_kernel|h3_bf16_to_f32_kernel",
    re.I,
)

cast_rows = [(t, n) for t, n in rows if CAST_RE.search(n)]
sgemm_rows = [(t, n) for t, n in rows if SGEMM_RE.search(n)]
pure_cast = [(t, n) for t, n in cast_rows if PURE_RE.search(n)]
prescale_dit = [(t, n) for t, n in cast_rows if "prescale" in n.lower()]
already_fused = [
    (t, n)
    for t, n in cast_rows
    if "rms_f32_to_f16" in n.lower() or "vae_rms" in n.lower()
]


def sum_ms(rs):
    return sum(t for t, _ in rs) / 1e6


def ms(ns):
    return ns / 1e6


cast_ms = sum_ms(cast_rows)
pure_ms = sum_ms(pure_cast)
sgemm_ms = sum_ms(sgemm_rows)
prescale_ms = sum_ms(prescale_dit)
already_ms = sum_ms(already_fused)
residual_bucket = max(0.0, cast_ms - prescale_ms - already_ms)
# VAE weight grids (36 blocks): qkv/out/w1(2*FFN)/w2 — not DiT SDPA
weight_grids = {49152, 16384, 65536, 131072}
vae_w_up = sum(t for t, n in pure_cast if "f16_to_f32" in n.lower())  # refined below via note
# Prefer reporting pure_ms as naive; corrected meat needs gridX (sqlite). Document both.
c011_meat_ms = pure_ms
GATE = 300.0
verdict = "NO_WC_WIN" if c011_meat_ms < GATE else "HAS_MEAT_NAIVE"
# Note: HAS_MEAT_NAIVE may be DiT SDPA — see close doc gridX split before construction.

lines = []
lines.append("# C011 nsys re-baseline (4090 sm_89)\n")
lines.append(f"- git_sha_synced: `{sha}`\n")
lines.append(f"- vae_gemm: {(kpi['vae_gemm'].group(1) if kpi['vae_gemm'] else 'NA')}\n")
lines.append(f"- vae_gemm_path: {(kpi['vae_path'].group(1) if kpi['vae_path'] else 'NA')}\n")
lines.append(f"- fallback_cast_logged: {kpi['fallback']}\n")
lines.append(f"- phase_ms: {(kpi['phase'].group(1) if kpi['phase'] else 'NA')}\n")
lines.append(f"- vae_repeat: {(kpi['vae_repeat'].group(1) if kpi['vae_repeat'] else 'NA')}\n")
lines.append(f"- vae: {(kpi['vae_line'].group(1) if kpi['vae_line'] else 'NA')}\n")
lines.append("\n## Kernel bucket totals (GPU kernel time, ms)\n")
lines.append("| bucket | ms |\n|---|---:|\n")
lines.append(f"| all CAST_RE match | {cast_ms:.1f} |\n")
lines.append(f"| DiT row_prescale (≠C011) | {prescale_ms:.1f} |\n")
lines.append(f"| already-fused vae_rms_f32_to_f16 | {already_ms:.1f} |\n")
lines.append(f"| pure HBM f32↔f16 elementwise (C011 meat) | {c011_meat_ms:.1f} |\n")
lines.append(f"| SIMT/TF32 sgemm (→C012 handoff) | {sgemm_ms:.1f} |\n")
lines.append(f"| cast residual after −prescale −rms_fuse | {residual_bucket:.1f} |\n")
lines.append(
    f"\n**Gate:** ≥300 ms pure cast for construction. "
    f"**Verdict: `{verdict}`** (meat={c011_meat_ms:.1f} ms).\n"
)
lines.append("\n## Top pure-cast kernels\n| ms | name |\n|---:|---|\n")
for t, n in sorted(pure_cast, reverse=True)[:25]:
    lines.append(f"| {ms(t):.2f} | `{n[:120]}` |\n")
if not pure_cast:
    lines.append("| — | *(none matched)* |\n")
lines.append("\n## Top CAST_RE kernels (all)\n| ms | name |\n|---:|---|\n")
for t, n in sorted(cast_rows, reverse=True)[:30]:
    lines.append(f"| {ms(t):.2f} | `{n[:120]}` |\n")
lines.append("\n## Top 20 kernels overall\n| ms | name |\n|---:|---|\n")
for t, n in rows[:20]:
    lines.append(f"| {ms(t):.2f} | `{n[:120]}` |\n")

(out / "REPORT.md").write_text("".join(lines), encoding="utf-8")
(out / "VERDICT.txt").write_text(
    f"VERDICT={verdict}\nC011_MEAT_MS={c011_meat_ms:.3f}\n"
    f"CAST_ALL_MS={cast_ms:.3f}\nPRESCALE_DIT_MS={prescale_ms:.3f}\n"
    f"ALREADY_FUSED_RMS_MS={already_ms:.3f}\nSGEMM_MS={sgemm_ms:.3f}\n"
    f"RESIDUAL_BUCKET_MS={residual_bucket:.3f}\n",
    encoding="utf-8",
)
print("".join(lines))
print("---")
print((out / "VERDICT.txt").read_text())
