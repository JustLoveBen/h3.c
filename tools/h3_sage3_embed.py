"""Official thu-ml SageAttention3 (Blackwell FP4) embed for h3.c SDPA plugin.

Opt-in via --sdpa-backend sage3 / H3_CUDA_SDPA_BACKEND=sage3.

Persistent BHTD FP16 Q/K/V/O. C does THD↔BHTD on device (no Python permute);
fwd() only runs sageattn3_blackwell.

True path only: sageattn3.sageattn3_blackwell (fp4attn_cuda). Never the
in-repo nested-loop sage2 reference kernel.
"""
from __future__ import annotations

import atexit
import os
import sys
import time

_STATE = {}
_POOL = {}  # (t, h, d) -> q/k/v/o tensors (keep alive across chunk seqs)
_FWD_N = 0
_BOOTSTRAPPED = 0


def _dump_embed():
    print(
        f"[h3] sage3 embed fwd_total={_FWD_N} backend={_STATE.get('backend', '?')}",
        file=sys.stderr,
        flush=True,
    )


atexit.register(_dump_embed)


def _bootstrap_paths(pkg_root: str) -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    roots = []
    env_root = os.environ.get("H3_SAGE3_ROOT", "").strip()
    for root in (pkg_root, env_root):
        if not root:
            continue
        roots.append(root)
        build_lib = os.path.join(
            root, "build", "lib.linux-x86_64-cpython-310"
        )
        if os.path.isdir(build_lib):
            roots.append(build_lib)
        # Also accept other cpython tags if present.
        build_dir = os.path.join(root, "build")
        if os.path.isdir(build_dir):
            for name in os.listdir(build_dir):
                if name.startswith("lib.linux-x86_64-cpython-"):
                    roots.append(os.path.join(build_dir, name))
        dist = os.path.join(root, "dist")
        if os.path.isdir(dist):
            for name in os.listdir(dist):
                if name.endswith(".egg"):
                    roots.append(os.path.join(dist, name))
    for root in reversed(roots):
        if root and root not in sys.path:
            sys.path.insert(0, root)
    _BOOTSTRAPPED = 1


def init(pkg_root: str, t: int, h: int, d: int, _scale: float,
         _unused0: int, _unused1: int):
    """Allocate persistent BHTD Q/K/V/O. Returns (q,k,v,o_ptr,backend).

    Layout exposed to C is F16 BHTD (1,H,T,D). C owns THD↔BHTD transpose.
    """
    _bootstrap_paths(pkg_root or "")
    import torch
    from sageattn3 import sageattn3_blackwell

    if d != 128:
        raise RuntimeError(f"sage3 embed requires D=128, got {d}")
    if t < 1 or h < 1:
        raise RuntimeError(f"sage3 embed bad shape T={t} H={h}")

    key = (int(t), int(h), int(d))
    slot = _POOL.get(key)
    if slot is None:
        q = torch.empty((1, h, t, d), dtype=torch.float16, device="cuda")
        k = torch.empty_like(q)
        v = torch.empty_like(q)
        o = torch.empty_like(q)
        slot = {"q": q, "k": k, "v": v, "o": o}
        _POOL[key] = slot
        reused = 0
    else:
        q, k, v, o = slot["q"], slot["k"], slot["v"], slot["o"]
        reused = 1

    _STATE.update(
        torch=torch,
        sageattn3_blackwell=sageattn3_blackwell,
        q=q,
        k=k,
        v=v,
        o=o,
        backend="sage3_blackwell_fp4",
        t=int(t),
        h=int(h),
        d=int(d),
        layout="BHTD",
        sage3_file=getattr(sageattn3_blackwell, "__module__", "sageattn3"),
        pool=len(_POOL),
        reused=reused,
    )
    print(
        f"[h3] sage3 embed backend=sage3_blackwell_fp4 "
        f"dtype=float16 layout=BHTD shape_bhtd=(1,{h},{t},{d}) "
        f"prep=cuda_thd_bhtd module={sageattn3_blackwell.__module__} "
        f"pool={len(_POOL)} reused={reused} "
        f"nested_loop=never ref_sage2=never",
        file=sys.stderr,
        flush=True,
    )
    return (
        int(q.data_ptr()),
        int(k.data_ptr()),
        int(v.data_ptr()),
        int(o.data_ptr()),
        "sage3_blackwell_fp4",
    )


def fwd():
    """sageattn3_blackwell on pre-laid-out BHTD QKV; write BHTD O in-place."""
    global _FWD_N
    torch = _STATE["torch"]
    fn = _STATE["sageattn3_blackwell"]
    q, k, v, o = _STATE["q"], _STATE["k"], _STATE["v"], _STATE["o"]

    # prep is owned by C (CUDA THD→BHTD); report 0 here.
    prep = 0.0

    t1 = time.perf_counter()
    out = fn(q, k, v, is_causal=False)
    if out is not o:
        o.copy_(out)
    torch.cuda.synchronize()
    kern = (time.perf_counter() - t1) * 1000.0

    # post-copy (BHTD→THD F32) is owned by C.
    copy = 0.0

    _FWD_N += 1
    host = prep + kern + copy
    if _FWD_N <= 3 or (_FWD_N % 50) == 0:
        print(
            f"[h3] sage3 fwd#{_FWD_N} prep_ms={prep:.2f} kern_ms={kern:.2f} "
            f"copy_ms={copy:.2f} host_ms={host:.2f} layout=BHTD "
            f"cuda_prep=1",
            file=sys.stderr,
            flush=True,
        )
    # ok, prep, kern, copy, compile, host — same tuple as sol_cute
    return (1.0, prep, kern, copy, 0.0, host)


def warmup(pkg_root: str, t: int, h: int, d: int, scale: float,
           unused0: int, unused1: int):
    init(pkg_root, t, h, d, scale, unused0, unused1)
    _STATE["q"].normal_()
    _STATE["k"].normal_()
    _STATE["v"].normal_()
    fwd()
    return 1
