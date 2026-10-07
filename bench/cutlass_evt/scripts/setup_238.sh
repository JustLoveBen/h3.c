#!/bin/bash
# R-D: CUTLASS sm_120 INT8 GEMM bench environment setup on 238
# Run once. Idempotent.
set -e

WS=/mnt/nvme1n1/workspace
CUTLASS_DIR=$WS/cutlass

echo "=== 1. CUTLASS source (header-only) ==="
if [ ! -d "$CUTLASS_DIR/include/cutlass" ]; then
    git clone --depth 1 --branch v4.2.1 \
        https://github.com/NVIDIA/cutlass.git "$CUTLASS_DIR" 2>&1 | tail -2 || {
        echo "git clone failed (network?), retry with shallow any-branch"
        git clone --depth 1 https://github.com/NVIDIA/cutlass.git "$CUTLASS_DIR" 2>&1 | tail -2
    }
fi
ls "$CUTLASS_DIR/include/cutlass/gemm/collective/builders/" | head
echo "cutlass dir ok"

echo "=== 2. nvidia-cutlass pip package (Python helper API) ==="
PY=/home/yongbin/miniconda3/envs/yongbin/bin/python
PIP=/home/yongbin/miniconda3/envs/yongbin/bin/pip
$PY -c "import cutlass; print('cutlass already:', getattr(cutlass,'__version__','?'))" 2>/dev/null || {
    $PIP install nvidia-cutlass==3.9.* 2>&1 | tail -3
    $PY -c "import cutlass; print('cutlass installed:', cutlass.__version__)"
}

echo "=== 3. nvcc check ==="
/usr/local/cuda/bin/nvcc --version | tail -1
echo "setup done"
