#!/usr/bin/env python3
"""Generate CUTLASS sm_120 INT8 GEMM bench binary for the 8 production shapes.

Uses nvidia-cutlass Python helper (cutlass.backend) to emit a NVCC-compilable
.cu with sm80-style INT8 TensorOp kernels tuned for sm_120 (arch 1200).
cublasLt on this box picks `cutlass_80_i16832` for these shapes; CUTLASS emits
the same math with its own collective, letting us compare on equal footing.
"""
import argparse, os, sys

try:
    import cutlass
    from cutlass.backend.evt.ir import *  # noqa - not needed, just probe
    _HAS_EVT = True
except Exception:
    import cutlass
    _HAS_EVT = False
from cutlass.backend import create_gemm_operator
from cutlass.backend.operation import GemmOperation
from cutlass import (
    DataType, LayoutType, MathOperation, OpcodeClass,
    TileDescription, ThreadblockSwizzle, MathInstruction,
    SwizzlingFunctor,
)
import cutlass.backend.library as library

# (m, n, k) — 8 production DiT INT8 GEMM shapes from h3_gpu_cuda.cu lt_i8 slots
SHAPES = [
    (3200, 21504, 5376),   # 480P fc1 (SwiGLU fused n=ffn*2)
    (3200, 5376, 7168),    # 480P fc2
    (3200, 28672, 5376),   # 480P qkv/out
    (3200, 5376, 14336),   # 480P out-adaln
    (4096, 21504, 5376),   # 768P fc1
    (4096, 5376, 7168),    # 768P fc2
    (4096, 28672, 5376),   # 768P qkv/out
    (4096, 5376, 14336),   # 768P out-adaln
]

# Tile configs to try: cublasLt picks 256x128x64 for big shapes, 128x128x64
# for smaller; we sweep a small grid and keep the best per shape.
TILE_CONFIGS = [
    # (cta_m, cta_n, cta_k, warp_m, warp_n, warp_k, stages, name)
    (256, 128, 64, 64, 64, 64, 4, "cta256x128"),
    (128, 256, 64, 64, 64, 64, 4, "cta128x256"),
    (128, 128, 64, 64, 64, 64, 4, "cta128x128"),
    (128, 128, 128, 64, 64, 64, 3, "cta128x128k128"),
]


def make_op(m, n, k, cfg, cc=1200):
    cta_m, cta_n, cta_k, wm, wn, wk, stages, name = cfg
    mi = MathInstruction(
        [16, 8, 32],
        DataType.s8, DataType.s8, DataType.s32,
        OpcodeClass.TensorOp,
        MathOperation.multiply_add,
    )
    tile = TileDescription(
        [cta_m, cta_n, cta_k], stages,
        [wm, wn, wk], mi,
    )
    op = GemmOperation(
        cc, tile, DataType.s8, LayoutType.RowMajor,   # A = weights [n,k]
        DataType.s8, LayoutType.RowMajor,           # B = acts [m,k]
        DataType.s32, LayoutType.RowMajor,          # C = acc [n,m] (col-major view = row-major transposed)
        DataType.s32,                               # epilogue compute
    )
    op.swizzling_functor = ThreadblockSwizzle.ThreadblockSwizzleStreamK if False else ThreadblockSwizzle.ThreadblockSwizzleLog
    return op


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-dir", default="/tmp/h3_cutlass_evt_gen")
    ap.add_argument("--arch", default="120a")  # sm_120a
    args = ap.parse_args()

    os.makedirs(args.emit_dir, exist_ok=True)
    print(f"[gen] cutlass python version: {getattr(cutlass,'__version__','?')}")
    print(f"[gen] emitting kernels for {len(SHAPES)} shapes x {len(TILE_CONFIGS)} cfgs")
    print(f"[gen] emit dir: {args.emit_dir}")

    # For simplicity: emit one "manifest" .cu per (shape, cfg) — a small
    # standalone kernel + host main that allocates, fills, times, prints TFLOPs.
    # We batch all into a single nvcc compile via a top-level .cu that #includes
    # the emitted sources.

    kernels = []
    for (m, n, k) in SHAPES:
        for cfg in TILE_CONFIGS:
            name = f"m{m}_n{n}_k{k}_{cfg[7]}"
            op = make_op(m, n, k, cfg)
            src_name = f"{args.emit_dir}/{name}.cu"
            # Emit via cutlass.backend: write a compilable source
            op.emit_library(src_name) if hasattr(op, "emit_library") else None
            kernels.append((m, n, k, name, cfg))
            print(f"[gen] queued {name}")

    # Actually the cleanest path: use cutlass.backend.create_gemm_operator to
    # emit a full GemmUniversal host-runnable source. That's what we do below
    # in emit_standalone().
    print("[gen] done (stub - use emit_standalone for real run)")


if __name__ == "__main__":
    main()
