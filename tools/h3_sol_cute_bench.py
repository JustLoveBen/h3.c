#!/usr/bin/env python3
"""Official NVlabs Sol-Attn SM89 CuTe kernel bench at cafe packed shape.

Not the h3.c nested-loop CUDA Sol. Times BF16 BTHD sol_attn vs torch SDPA.
cuDNN gold remains the C fused path (~46.7 ms); this script only measures
the released kernel.

  PYTHONPATH=.../techniques/sparse_backends python tools/h3_sol_cute_bench.py
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time


def fill_qkv(seq: int, heads: int, dim: int, gaussian: bool):
    import torch

    q = torch.empty((1, seq, heads, dim), dtype=torch.bfloat16, device="cuda")
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(0xC0FFEE)
    noise = torch.rand((3, seq, heads, dim), generator=rng)
    noise = (noise * 2.0 - 1.0).to(device="cuda", dtype=torch.float32)
    i = torch.arange(seq, device="cuda", dtype=torch.float32).view(seq, 1, 1)
    h = torch.arange(heads, device="cuda", dtype=torch.float32).view(1, heads, 1)
    d = torch.arange(dim, device="cuda", dtype=torch.float32).view(1, 1, dim)
    loc = torch.sin(i / 64.0 + 0.07 * h)
    nd = 0.02 * d / float(dim)
    if gaussian:
        q[0] = noise[0].to(torch.bfloat16)
        k[0] = noise[1].to(torch.bfloat16)
        v[0] = noise[2].to(torch.bfloat16)
    else:
        q[0] = (loc + 0.05 * noise[0] + nd).to(torch.bfloat16)
        k[0] = (loc + 0.05 * noise[1] - nd).to(torch.bfloat16)
        v[0] = (loc * 0.4 + 0.08 * noise[2]).to(torch.bfloat16)
    return q.contiguous(), k.contiguous(), v.contiguous()


def time_cuda(fn, warmup: int, iters: int) -> float:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / float(iters)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seq", type=int, default=15973)
    p.add_argument("--heads", type=int, default=56)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=4)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--sink-tokens", type=int, default=988)
    p.add_argument("--sink-start", type=int, default=0)
    p.add_argument("--thresh-type", default="diag")
    p.add_argument("--gaussian", action="store_true")
    p.add_argument(
        "--pkg-root",
        default=os.environ.get("H3_SOL_ATTN_ROOT", ""),
    )
    args = p.parse_args()
    if not args.pkg_root:
        print(
            "h3_sol_cute_bench: set --pkg-root or H3_SOL_ATTN_ROOT to official "
            "NVlabs sol-engine techniques/sparse_backends (no machine default)",
            file=sys.stderr,
        )
        return 2
    sys.path.insert(0, args.pkg_root)

    import torch
    from sol_attn import get_sol_attn_backend, sol_attn

    backend = get_sol_attn_backend("cuda")
    print(
        f"h3_sol_cute_bench seq={args.seq} heads={args.heads} dim={args.dim} "
        f"fill={'gaussian' if args.gaussian else 'local-corr'} "
        f"tau={args.tau} thresh={args.thresh_type} sink={args.sink_start}+"
        f"{args.sink_tokens} backend={backend} torch={torch.__version__} "
        f"cap={torch.cuda.get_device_capability()}",
        flush=True,
    )
    official = ("cute_sm89", "cute_sm90", "cute_sm100", "cute_sm120")
    if backend not in official:
        print(
            f"h3_sol_cute_bench: refusing non-CuTe backend {backend!r} "
            "(need cute_sm120 on 5090; triton/triton_ref is not a product kernel)",
            file=sys.stderr,
        )
        return 2
    if args.dim != 128:
        print("official sol_attn requires D=128", file=sys.stderr)
        return 2

    q, k, v = fill_qkv(args.seq, args.heads, args.dim, args.gaussian)

    def run_sol():
        return sol_attn(
            q,
            k,
            v,
            tau=args.tau,
            thresh_type=args.thresh_type,
            sink_start=args.sink_start,
            sink_tokens=args.sink_tokens,
        )

    print("compiling/warming official sol_attn (first call may take minutes)...",
          flush=True)
    t0 = time.time()
    out = run_sol()
    torch.cuda.synchronize()
    print(f"first_call_s={time.time() - t0:.1f} out={tuple(out.shape)} "
          f"{out.dtype}", flush=True)

    ms_sol = time_cuda(run_sol, args.warmup, args.iters)
    print(f"sol_cute  backend={backend}  {ms_sol:.2f} ms/iter", flush=True)

    ms_sdpa = -1.0
    try:
        def run_sdpa():
            return torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
            ).transpose(1, 2)

        run_sdpa()
        torch.cuda.synchronize()
        ms_sdpa = time_cuda(run_sdpa, args.warmup, args.iters)
        print(f"torch_sdpa  {ms_sdpa:.2f} ms/iter  (not the C cuDNN gold)",
              flush=True)
        rel = (out.float() - run_sdpa().float()).norm() / (
            run_sdpa().float().norm() + 1e-12
        )
        print(f"rel_l2_vs_torch_sdpa={float(rel):.4f}", flush=True)
    except Exception as exc:
        print(f"torch_sdpa skipped: {type(exc).__name__}: {exc}", flush=True)

    gold = 46.7
    need = gold * 0.92
    print("\n=== verdict vs C fused cuDNN gold 46.7 ms ===", flush=True)
    print(
        f"sol_cute={ms_sol:.2f} ms  need_stably_faster_than={need:.2f} ms "
        f"({0.92:.2f}x of 46.7)",
        flush=True,
    )
    if ms_sol + 1.0 < need:
        print("official SM89 CuTe IS stably faster than cuDNN gold.", flush=True)
    else:
        print(
            "official SM89 CuTe NOT stably faster than cuDNN gold. "
            "Kernel itself has no speedup vs this repo's fused cuDNN.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
