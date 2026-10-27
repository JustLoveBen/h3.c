#!/usr/bin/env python3
"""Aggregate Exp-G1 Track-B KPIs and write VERDICT."""
import math, os, pathlib, re
root = pathlib.Path(os.environ["OUT_ROOT"])
arms = [
    "arm1_cudnn_dmad4",
    "arm2_sol_fullsparse_dmad4",
    "arm3_sol_layers2_dmad4",
    "arm4_sol_steps1_dmad4",
]

def parse(arm):
    p = root / arm / "generate.log"
    t = p.read_text(errors="replace") if p.exists() else ""
    wall = re.findall(r"^WALL_SEC=([0-9.]+)", t, re.M)
    phases = re.findall(
        r"^phase_ms:.*?\bdit=([0-9.]+).*?\btotal=([0-9.]+)",
        t, re.M,
    )
    waffles = re.findall(r"^waffle:\s*frame=(\d+)\s+b16=([-+0-9.eE]+)", t, re.M)
    hot_w = [(int(f), float(b)) for f, b in waffles[-5:]] if len(waffles) >= 5 else [
        (int(f), float(b)) for f, b in waffles
    ]
    mean_w = sum(b for _, b in hot_w) / len(hot_w) if hot_w else float("nan")
    caps = re.search(r"^device_caps:\s*(.*)$", t, re.M)
    sel = re.search(r"^sdpa_select:\s*(.*)$", t, re.M)
    strict = re.search(r"^sdpa_strict:\s*(.*)$", t, re.M)
    sol_stats = [
        ln for ln in t.splitlines()
        if ("sol_cute" in ln.lower() or "dense_skip" in ln.lower()
            or ln.startswith("sdpa_") or "device_caps:" in ln)
    ]
    hits = re.findall(r"hits[=:](\d+)(?:.*?want[=:](\d+))?", t)
    dit_hot = float(phases[-1][0]) if phases else float("nan")
    tot_hot = float(phases[-1][1]) / 1000.0 if phases else float("nan")
    return {
        "wall": float(wall[-1]) if wall else float("nan"),
        "dit_ms": dit_hot,
        "warm_s": tot_hot,
        "waffle_mean": mean_w,
        "waffle_frames": hot_w,
        "caps": caps.group(1) if caps else "",
        "select": sel.group(1) if sel else "",
        "strict": strict.group(1) if strict else "",
        "sol_lines": sol_stats[-16:],
        "hits": hits[-3:] if hits else [],
        "rc": "RUN_RC=0" in t,
    }

rows = {a: parse(a) for a in arms}
print("# Exp-G1 Track B · sol_cute+guard vs cudnn @ DMAD-4 · 4090")
print()
print(f"out: `{root}`")
print()
print("| arm | WALL_s | warm_s | dit_ms | waffle_mean | hits_tail | select |")
print("|-----|--------|--------|--------|-------------|-----------|--------|")
for a in arms:
    r = rows[a]
    def fmt(x, nd=1):
        return f"{x:.{nd}f}" if x == x else "NA"
    print(
        f"| {a} | {fmt(r['wall'],1)} | {fmt(r['warm_s'],2)} | {fmt(r['dit_ms'],0)} | "
        f"{fmt(r['waffle_mean'],3)} | {r['hits']} | `{r['select'][:50]}` |"
    )
print()
a1, a2, a3 = rows[arms[0]], rows[arms[1]], rows[arms[2]]
print("## Gate")
ok = True
reasons = []
if any(math.isnan(x["waffle_mean"]) for x in (a1, a3)):
    ok = False
    reasons.append("waffle missing on arm1 or arm3")
else:
    d31 = a3["waffle_mean"] - a1["waffle_mean"]
    print(f"- arm3 waffle - arm1 = {d31:+.3f} (PASS <= +0.10)")
    if d31 > 0.10:
        ok = False
        reasons.append(f"arm3 waffle delta {d31:+.3f} > +0.10")
    if not math.isnan(a2["waffle_mean"]):
        d21 = a2["waffle_mean"] - a1["waffle_mean"]
        print(f"- arm2 waffle - arm1 = {d21:+.3f} (full-sparse FAIL line > +0.15)")
        if d21 > 0.15:
            reasons.append(f"arm2 full-sparse degradation confirmed (+{d21:.3f})")
if any(math.isnan(x["dit_ms"]) for x in (a1, a3)) or a1["dit_ms"] <= 0:
    ok = False
    reasons.append("dit_ms missing")
else:
    ratio = a3["dit_ms"] / a1["dit_ms"]
    print(f"- arm3 dit / arm1 dit = {ratio:.3f} (PASS <= 0.92)")
    if ratio > 0.92:
        ok = False
        reasons.append(f"arm3 dit ratio {ratio:.3f} > 0.92 (no speed win)")
print()
print("## device_caps / sdpa / sol lines")
for a in arms:
    r = rows[a]
    print(f"### {a}")
    print(f"- caps: `{r['caps']}`")
    print(f"- select: `{r['select']}`")
    print(f"- strict: `{r['strict']}`")
    for ln in r["sol_lines"]:
        print(f"- `{ln}`")
    print(f"- waffle hot frames: {r['waffle_frames']}")
    print()
verdict = "PASS" if ok else "FAIL"
print(f"## VERDICT: **{verdict}** (Track B quality floor)")
for x in reasons:
    print(f"- {x}")
(root / "VERDICT.txt").write_text(verdict + "\n" + "\n".join(reasons) + "\n")
print()
print("labeled clips under labeled/")
