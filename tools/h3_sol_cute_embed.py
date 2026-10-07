"""Official NVlabs sol-engine CuTe sol_attn embed for h3.c SDPA.

Holds persistent BTHD BF16 tensors. C copies F16 THD into Q/K/V pointers,
calls fwd(), then copies O back.

True-path requirements (fail loud, do not silently fall back):
  - Triton >= 3.3 with triton.tools.tensor_descriptor.TensorDescriptor
  - official sol_attn.preprocess.prepare (Triton reduce, not _prepare_torch)
  - official CuTe backend: cute_sm89 / cute_sm90 / cute_sm100 / cute_sm120
    (NVlabs sol-engine interface.py _CUTE_BACKENDS). 5090 = cute_sm120.
  - missing CuTe runtime returns "triton" (triton_ref) — that is NOT a kernel.

No machine-local default paths. Set H3_SOL_ATTN_ROOT / H3_TRITON_OVERLAY /
H3_SOL_CUTE_TOOLS.
"""
from __future__ import annotations

import atexit
import importlib
import os
import sys
import time

_STATE = {}
_FWD_N = 0
_LOGGED = 0
_BOOTSTRAPPED = 0

# Keep in sync with h3_sdpa_sol_cute_is_official() / official interface.py.
_OFFICIAL_CUTE = ("cute_sm89", "cute_sm90", "cute_sm100", "cute_sm120")


def _dump_embed():
    be = _STATE.get("backend", "?")
    print(
        f"[h3] sol_attn embed fwd_total={_FWD_N} backend={be}",
        file=sys.stderr,
        flush=True,
    )


atexit.register(_dump_embed)


def _require_triton_prepare() -> bool:
    e = os.environ.get("H3_SOL_REQUIRE_TRITON_PREPARE", "1")
    return e not in ("0", "false", "False", "no")


def _want_profile() -> bool:
    e = os.environ.get("H3_SOL_PROFILE", "0")
    return e not in ("", "0", "false", "False", "off", "no")


def _bootstrap_paths(pkg_root: str) -> None:
    """Put Triton overlay + official sol_attn ahead of site-packages.

    Overlay is env-only. A missing H3_TRITON_OVERLAY must not silently
    pick a 4090 developer path.
    """
    global _BOOTSTRAPPED
    overlay = os.environ.get("H3_TRITON_OVERLAY", "").strip()
    roots = []
    if overlay and os.path.isdir(overlay):
        roots.append(overlay)
    if pkg_root:
        roots.append(pkg_root)
    for root in reversed(roots):
        if root and root not in sys.path:
            sys.path.insert(0, root)
    _BOOTSTRAPPED = 1


def _read_origin(pkg_root: str) -> str:
    env = os.environ.get("H3_SOL_GIT_SHA", "").strip()
    candidates = []
    if pkg_root:
        candidates.append(os.path.join(pkg_root, "ORIGIN.txt"))
        candidates.append(os.path.join(pkg_root, "..", "..", "ORIGIN.txt"))
        candidates.append(os.path.join(pkg_root, "..", "ORIGIN.txt"))
    sha = env
    extra = ""
    for path in candidates:
        path = os.path.normpath(path)
        if os.path.isfile(path):
            text = open(path, encoding="utf-8").read().strip()
            extra = " ".join(text.splitlines())
            if not sha:
                for line in text.splitlines():
                    if line.startswith("GIT_SHA="):
                        sha = line.split("=", 1)[1].strip()
                        break
            break
    return sha or "unknown", extra


def _prove_official_prepare(pp) -> str:
    """Return preprocess tag or raise. Never accept pytorch_fallback."""
    td = getattr(pp, "TensorDescriptor", None)
    if td is None:
        raise RuntimeError(
            "sol_attn preprocess TensorDescriptor is None; "
            "need Triton>=3.3 official prepare (not pytorch_fallback)"
        )
    if _require_triton_prepare() and hasattr(pp, "_prepare_torch"):
        raise RuntimeError(
            "sol_attn preprocess still ships _prepare_torch; "
            "point H3_SOL_ATTN_ROOT at official NVlabs preprocess.py "
            "(unconditional TensorDescriptor import)"
        )
    import inspect

    src = inspect.getsource(pp.prepare)
    if "_prepare_torch" in src:
        raise RuntimeError(
            "sol_attn.preprocess.prepare still branches to _prepare_torch"
        )
    if "TensorDescriptor" not in src and "_reduce_kv" not in src:
        raise RuntimeError(
            "sol_attn.preprocess.prepare does not look like official Triton path"
        )
    return "triton_prepare"


def _prove_official_arch(backend: str):
    """Import the official arch module. Do not invent a kernel."""
    if backend == "cute_sm89":
        sm89_ml = importlib.import_module("sol_attn.sm89.mainloop")
        if getattr(sm89_ml, "SolAttnForwardSm89", None) is None:
            raise RuntimeError(
                "SolAttnForwardSm89 missing from sol_attn.sm89.mainloop"
            )
        return "SolAttnForwardSm89", sm89_ml.__file__
    if backend == "cute_sm100":
        mod = importlib.import_module("sol_attn.sm100")
        if not hasattr(mod, "forward"):
            raise RuntimeError("sol_attn.sm100.forward missing")
        return "sm100.forward", mod.__file__
    arch = {
        "cute_sm90": "sol_attn.sm90",
        "cute_sm120": "sol_attn.sm120",
    }.get(backend)
    if not arch:
        raise RuntimeError(f"unsupported official backend {backend!r}")
    mod = importlib.import_module(arch)
    if not hasattr(mod, "make_kernel"):
        raise RuntimeError(f"{arch}.make_kernel missing")
    return f"{arch}.make_kernel", mod.__file__


def init(pkg_root: str, t: int, h: int, d: int, tau: float,
         sink_start: int, sink_tokens: int):
    _bootstrap_paths(pkg_root)
    import triton
    from triton.tools.tensor_descriptor import TensorDescriptor  # noqa: F401
    import torch
    import sol_attn
    from sol_attn import get_sol_attn_backend, sol_attn as sol_attn_fn
    import sol_attn.preprocess as pp

    ver = getattr(triton, "__version__", "?")
    if tuple(int(x) for x in ver.split(".")[:2] if x.isdigit()) < (3, 3):
        raise RuntimeError(
            f"Triton {ver} has no TensorDescriptor; need >=3.3 (overlay 3.4)"
        )
    prep = _prove_official_prepare(pp)
    backend = get_sol_attn_backend("cuda")
    if backend not in _OFFICIAL_CUTE:
        raise RuntimeError(
            f"sol_cute requires official CuTe backend "
            f"(cute_sm89|cute_sm90|cute_sm100|cute_sm120), got {backend!r} "
            "(triton/nested_loop/triton_ref=never; need cutlass.cute + "
            "cuda.bindings.driver on 5090 for cute_sm120)"
        )
    kernel_name, sm89_file = _prove_official_arch(backend)

    q = torch.zeros((1, t, h, d), dtype=torch.bfloat16, device="cuda")
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    o = torch.empty_like(q)
    sha, origin_extra = _read_origin(pkg_root)
    _STATE.update(
        torch=torch,
        sol_attn=sol_attn_fn,
        q=q,
        k=k,
        v=v,
        o=o,
        tau=float(tau),
        sink_start=int(sink_start),
        sink_tokens=int(sink_tokens),
        backend=backend,
        preprocess=prep,
        triton_ver=ver,
        triton_file=triton.__file__,
        interface_file=sol_attn.__file__,
        preprocess_file=pp.__file__,
        kernel=kernel_name,
        arch_file=sm89_file,
        git_sha=sha,
        t=int(t),
        h=int(h),
        d=int(d),
    )
    print(
        f"[h3] sol_attn origin git={sha} {origin_extra}".rstrip(),
        file=sys.stderr,
        flush=True,
    )
    print(
        f"[h3] sol_attn files interface={sol_attn.__file__} "
        f"preprocess={pp.__file__} arch={sm89_file} "
        f"embed={__file__} triton={triton.__file__}",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"[h3] sol_attn embed backend={backend} kernel={kernel_name} "
        f"dtype={q.dtype} shape={tuple(q.shape)} layout=BTHD "
        f"preprocess={prep} triton={ver} TensorDescriptor=ok "
        f"tau={float(tau)} sink={int(sink_start)}+{int(sink_tokens)} "
        f"qkv=INT8_GEMM_F16_THD_copy_to_BF16_BTHD nested_loop=never",
        file=sys.stderr,
        flush=True,
    )
    trim_cache()
    return (
        int(q.data_ptr()),
        int(k.data_ptr()),
        int(v.data_ptr()),
        int(o.data_ptr()),
        str(backend),
    )


def _log_fwd_count():
    import sol_attn.interface as iface

    keys = list(getattr(iface, "_compiled", {}).keys())
    arch_tag = {
        "cute_sm89": (8, 9),
        "cute_sm90": (9, 0),
        "cute_sm100": (10, 0),
        "cute_sm120": (12, 0),
    }.get(_STATE.get("backend"), (8, 9))
    want = (
        _STATE["q"].device.index,
        arch_tag,
        1,
        _STATE["t"],
        _STATE["h"],
        1,
    )
    match = "MATCH" if want in keys or any(
        k[3] == _STATE["t"] and k[4] == _STATE["h"] for k in keys
    ) else "MISMATCH"
    print(
        f"[h3] sol_attn fwd_count={_FWD_N} compiled_keys={keys} "
        f"generate_THD=({_STATE['t']},{_STATE['h']},{_STATE['d']}) "
        f"compile_vs_generate={match} kernel={_STATE.get('kernel')}",
        file=sys.stderr,
        flush=True,
    )


def _fwd_sol_attn():
    return _STATE["sol_attn"](
        _STATE["q"],
        _STATE["k"],
        _STATE["v"],
        tau=_STATE["tau"],
        thresh_type=os.environ.get("SOL_ATTN_THRESH_TYPE", "diag"),
        sink_start=_STATE["sink_start"],
        sink_tokens=_STATE["sink_tokens"],
    )


def _fwd_sm89_timed():
    """Split Triton prepare vs SolAttnForwardSm89 vs O copy. Same ops as
    sol_attn.interface._sol_attn_cute for SM89."""
    import sol_attn.preprocess as pp
    import sol_attn.interface as iface

    torch = _STATE["torch"]
    q, k, v = _STATE["q"], _STATE["k"], _STATE["v"]
    tau = _STATE["tau"]
    sink_start = _STATE["sink_start"]
    sink_tokens = _STATE["sink_tokens"]
    thresh = os.environ.get("SOL_ATTN_THRESH_TYPE", "diag")
    scale = q.shape[-1] ** -0.5
    batch, tokens, heads, _ = q.shape
    arch = (8, 9)
    prep_ms = kern_ms = copy_ms = compile_ms = 0.0

    e_prep0 = torch.cuda.Event(enable_timing=True)
    e_prep1 = torch.cuda.Event(enable_timing=True)
    e_kern0 = torch.cuda.Event(enable_timing=True)
    e_kern1 = torch.cuda.Event(enable_timing=True)
    e_copy0 = torch.cuda.Event(enable_timing=True)
    e_copy1 = torch.cuda.Event(enable_timing=True)

    with torch.cuda.device(q.device):
        e_prep0.record()
        kc, vc, threshold = pp.prepare(
            q, k, v, scale=scale, tau=tau, thresh_type=thresh
        )
        e_prep1.record()

        output = torch.empty_like(v)
        lse = torch.empty(
            (batch, tokens, heads), device=q.device, dtype=torch.float32
        )
        stream = iface._stream(q.device)
        key = (q.device.index, arch, batch, tokens, heads, 1)
        sink_start_block, sink_end_block = iface._sink_block_range(
            tokens, sink_start, sink_tokens
        )
        tensors = [q, k, v, output, kc, vc, threshold, lse]
        compiled = iface._compiled.get(key)
        if compiled is None:
            t_c0 = time.perf_counter()
            compiled, args = iface._compile_sm89(
                key, tensors, scale, sink_start_block, sink_end_block, stream
            )
            torch.cuda.synchronize()
            compile_ms = (time.perf_counter() - t_c0) * 1000.0
            print(
                f"[h3] sol_profile cute.compile ms={compile_ms:.1f} key={key} "
                f"(cold only; exclude from speedup)",
                file=sys.stderr,
                flush=True,
            )
        else:
            args = iface._to_cute_tensors(tensors)
        e_kern0.record()
        compiled(
            *args,
            scale,
            sink_start_block,
            sink_end_block,
            stream=stream,
        )
        e_kern1.record()
        e_copy0.record()
        _STATE["o"].copy_(output)
        e_copy1.record()
        torch.cuda.synchronize()

    prep_ms = float(e_prep0.elapsed_time(e_prep1))
    kern_ms = float(e_kern0.elapsed_time(e_kern1))
    copy_ms = float(e_copy0.elapsed_time(e_copy1))
    return prep_ms, kern_ms, copy_ms, compile_ms


def fwd():
    global _FWD_N, _LOGGED
    t_host0 = time.perf_counter()
    prep_ms = kern_ms = copy_ms = compile_ms = 0.0

    if _STATE.get("q") is None:
        raise RuntimeError("sol_cute fwd before init")
    if _STATE.get("preprocess") != "triton_prepare":
        raise RuntimeError(
            f"refusing fwd with preprocess={_STATE.get('preprocess')!r}"
        )
    if not _LOGGED:
        print(
            f"[h3] sol_attn embed backend={_STATE['backend']} dtype="
            f"{_STATE['q'].dtype} shape={tuple(_STATE['q'].shape)} "
            f"layout=BTHD preprocess={_STATE['preprocess']} "
            f"tau={_STATE['tau']} sink={_STATE['sink_start']}+"
            f"{_STATE['sink_tokens']}",
            file=sys.stderr,
            flush=True,
        )
        _LOGGED = 1

    if _want_profile() and _STATE.get("backend") == "cute_sm89":
        try:
            prep_ms, kern_ms, copy_ms, compile_ms = _fwd_sm89_timed()
        except Exception as e:
            print(
                f"[h3] sol_profile split-path failed ({type(e).__name__}: {e}); "
                f"falling back to sol_attn() lumped kernel time",
                file=sys.stderr,
                flush=True,
            )
            torch = _STATE["torch"]
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            out = _fwd_sol_attn()
            _STATE["o"].copy_(out)
            e1.record()
            torch.cuda.synchronize()
            kern_ms = float(e0.elapsed_time(e1))
    else:
        out = _fwd_sol_attn()
        _STATE["o"].copy_(out)
        _STATE["torch"].cuda.synchronize()

    _FWD_N += 1
    if _FWD_N <= 3 or (_FWD_N % 50) == 0:
        _log_fwd_count()
    host_ms = (time.perf_counter() - t_host0) * 1000.0
    return (1.0, prep_ms, kern_ms, copy_ms, compile_ms, host_ms)


def warmup(pkg_root: str, t: int, h: int, d: int, tau: float,
           sink_start: int, sink_tokens: int):
    """Compile official CuTe while VRAM is empty, then drop QKVO.

    The compiled kernel stays in sol_attn._compiled keyed by
    (device, arch, B, T, H, kv_splits). Later init()/fwd() reallocates
    tensors of the same shape without NVRTC.
    """
    global _LOGGED, _FWD_N
    init(pkg_root, t, h, d, tau, sink_start, sink_tokens)
    fwd()
    torch = _STATE.get("torch")
    for key in ("q", "k", "v", "o"):
        _STATE.pop(key, None)
    if torch is not None:
        torch.cuda.empty_cache()
    _LOGGED = 0
    # The compile fwd above already exercised the real kernel (on zeros).
    # Reset the embed counter so the FIRST real-data hit still logs via
    # _log_fwd_count and takes the same lazy-init branch as a true cold run.
    _FWD_N = 0
    return 1


def trim_cache():
    """Return unused caching-allocator blocks to the driver (keep QKVO).

    Native cudaMalloc w_i8 cannot use torch reserved memory. Call after
    init / compile and before INT8 window fill.
    """
    torch = _STATE.get("torch")
    if torch is None:
        try:
            import torch as _torch
            torch = _torch
        except Exception:
            return 1
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass
    return 1


def drop_pool():
    """Release persistent QKVO so a 768P realloc can fit on 32GB."""
    global _LOGGED
    torch = _STATE.get("torch")
    for key in ("q", "k", "v", "o"):
        _STATE.pop(key, None)
    _LOGGED = 0
    trim_cache()
    return 1
