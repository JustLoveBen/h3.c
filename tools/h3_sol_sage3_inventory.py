#!/usr/bin/env python3
"""Inventory true sage3 / NVlabs sol-engine CuTe kernels. Never fakes hits.

This probe does not run DiT. Missing kernels print MISSING and how to fetch
the real tree. Exit 0 always unless --require or --self-test fails.

Official Sol-Attn CuTe names (sol-engine interface.py _CUTE_BACKENDS):
  (8,9)->cute_sm89  (9,0)->cute_sm90  (10,0)->cute_sm100  (12,0)->cute_sm120
Missing cutlass.cute returns backend \"triton\" (triton_ref) — not a kernel.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys

OFFICIAL_CUTE = ("cute_sm89", "cute_sm90", "cute_sm100", "cute_sm120")
FAKE_SOL = ("triton", "triton_ref", "nested_loop", "metal")

SAGE3_HOW = """
how_to_get sage3 (thu-ml SageAttention3 Blackwell FP4, sm_100/sm_120):
  git clone https://github.com/thu-ml/SageAttention.git
  cd SageAttention/sageattention3_blackwell
  python setup.py install   # CUDA >=12.8; 5090 needs sm_120
  export H3_SAGE3_ROOT=<that install / source root that imports sageattn3>
  export H3_SAGE3_LIB=$PWD/build-cuda-sm120/libh3_sage3.so
  export H3_SAGE3_TOOLS=$PWD/tools
  prove: python -c "from sageattn3 import sageattn3_blackwell; import fp4attn_cuda"
  do not point H3_SAGE3_ROOT at build-cuda-sm120 — that is the embed .so, not the kernel
""".strip()

SOL_HOW = """
how_to_get sol_cute (NVlabs Sana sol-engine CuTe, 5090 = cute_sm120):
  git clone --branch sol-engine --depth 1 https://github.com/NVlabs/Sana.git
  export H3_SOL_ATTN_ROOT=$PWD/Sana/techniques/sparse_backends
  export H3_SOL_CUTE_LIB=$PWD/build-cuda-sm120/libh3_sol_cute.so
  export H3_SOL_CUTE_TOOLS=$PWD/tools
  pip install nvidia-cutlass-dsl   # provides cutlass.cute; also cuda.bindings.driver
  Triton >=3.3 with triton.tools.tensor_descriptor.TensorDescriptor
    (H3_TRITON_OVERLAY if site-packages is too old)
  prove: python -c "from sol_attn import get_sol_attn_backend; print(get_sol_attn_backend('cuda'))"
  expect cute_sm120 on RTX 5090. If it prints triton, CuTe runtime is missing —
  that is NOT a product kernel; do not count hits.
  ROOT must be official preprocess (no _prepare_torch fallback).
""".strip()


def _env_line(key: str) -> str:
    v = os.environ.get(key, "").strip()
    if not v:
        return f"{key}: unset"
    exists = os.path.exists(v)
    return f"{key}: set path={v} exists={int(exists)}"


def _try_import(name: str):
    try:
        mod = importlib.import_module(name)
        return 1, getattr(mod, "__file__", "?")
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}"


def _sage3_kernel_present(root: str) -> int:
    if not root:
        return 0
    if os.path.isdir(os.path.join(root, "sageattn3")):
        return 1
    if os.path.isdir(os.path.join(root, "sageattention3_blackwell")):
        return 1
    dist = os.path.join(root, "dist")
    if os.path.isdir(dist):
        for name in os.listdir(dist):
            if name.startswith("sageattn3") and name.endswith(".egg"):
                return 1
    return 0


def inventory(out=None) -> dict:
    log = out or sys.stdout
    rows = {}

    def emit(s: str):
        log.write(s + "\n")

    emit("h3_sol_sage3_inventory: probe only; hits are not claimed")
    for key in (
        "H3_SAGE3_LIB",
        "H3_SAGE3_ROOT",
        "H3_SAGE3_TOOLS",
        "H3_SOL_CUTE_LIB",
        "H3_SOL_ATTN_ROOT",
        "H3_SOL_CUTE_TOOLS",
        "H3_TRITON_OVERLAY",
    ):
        line = _env_line(key)
        emit("  " + line)
        rows[key] = os.environ.get(key, "").strip()

    sage_root = rows["H3_SAGE3_ROOT"]
    rows["sage3_tree"] = _sage3_kernel_present(sage_root)
    emit(f"  sage3_tree: {rows['sage3_tree']} (sageattn3/ under ROOT)")

    ok, detail = _try_import("sageattn3")
    rows["sageattn3"] = ok
    emit(f"  import sageattn3: {ok} ({detail})")
    ok, detail = _try_import("fp4attn_cuda")
    rows["fp4attn_cuda"] = ok
    emit(f"  import fp4attn_cuda: {ok} ({detail})")

    ok, detail = _try_import("cutlass.cute")
    rows["cutlass_cute"] = ok
    emit(f"  import cutlass.cute: {ok} ({detail})")
    ok, detail = _try_import("cuda.bindings.driver")
    rows["cuda_bindings"] = ok
    emit(f"  import cuda.bindings.driver: {ok} ({detail})")

    sol_root = rows["H3_SOL_ATTN_ROOT"]
    if sol_root and sol_root not in sys.path:
        sys.path.insert(0, sol_root)
    ok, detail = _try_import("sol_attn")
    rows["sol_attn"] = ok
    emit(f"  import sol_attn: {ok} ({detail})")
    backend = ""
    if ok:
        try:
            from sol_attn import get_sol_attn_backend

            backend = str(get_sol_attn_backend("cuda"))
        except Exception as e:
            backend = f"error:{type(e).__name__}:{e}"
    rows["sol_backend"] = backend
    official = int(backend in OFFICIAL_CUTE)
    fake = int(backend in FAKE_SOL)
    emit(
        f"  sol_backend: {backend or 'n/a'} official_cute={official} "
        f"fake_or_triton={fake}"
    )
    rows["sol_official"] = official
    rows["sage3_kernel"] = int(rows["sageattn3"] and rows["fp4attn_cuda"])
    emit(f"  sage3_kernel: {rows['sage3_kernel']} (sageattn3 AND fp4attn_cuda)")
    emit(f"  sol_cute_kernel: {official} (official CuTe name only)")
    if not rows["sage3_kernel"]:
        emit(SAGE3_HOW)
    if not official:
        emit(SOL_HOW)
    emit(
        "verdict: do not report sage3/sol hits unless sage3_kernel=1 or "
        "sol_cute_kernel=1. want>0 && hits==0 must hard-fail."
    )
    return rows


def self_test() -> int:
    fails = 0

    def expect(c, m):
        nonlocal fails
        if not c:
            print(f"FAIL {m}", file=sys.stderr)
            fails += 1

    expect("cute_sm120" in OFFICIAL_CUTE, "5090 name in allowlist")
    expect("cute_sm89" in OFFICIAL_CUTE, "4090 name in allowlist")
    expect("triton" not in OFFICIAL_CUTE, "triton not official")
    expect("nested_loop" in FAKE_SOL, "nested_loop fake")
    here = os.path.dirname(os.path.abspath(__file__))
    embed = os.path.join(here, "h3_sol_cute_embed.py")
    with open(embed, encoding="utf-8") as f:
        text = f.read()
    expect("/home/haiyang" not in text, "embed has no haiyang default")
    expect("cute_sm120" in text, "embed mentions cute_sm120")
    expect("def drop_pool" in text, "embed can drop QKVO")
    c_embed = os.path.join(here, "h3_sol_cute_embed.c")
    with open(c_embed, encoding="utf-8") as f:
        ctext = f.read()
    expect("/home/haiyang" not in ctext, "C embed has no haiyang default")
    expect("H3_SOL_CUTE_TOOLS" in ctext, "C embed uses H3_SOL_CUTE_TOOLS")
    if fails:
        return 1
    print("h3_sol_sage3_inventory --self-test: PASS")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument(
        "--require",
        choices=("sage3", "sol_cute"),
        help="exit 2 if that kernel is missing (still does not fake hits)",
    )
    args = p.parse_args()
    if args.self_test:
        return self_test()
    rows = inventory()
    if args.require == "sage3" and not rows.get("sage3_kernel"):
        return 2
    if args.require == "sol_cute" and not rows.get("sol_official"):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
