#!/usr/bin/env bash
# run_phase0.sh — one-shot P4.2 Phase-0 runner for machine "238"
# (Linux x86_64, CUDA 13.1, RTX 5090 sm_120a). See design doc
# docs/research/2026-10-08-nvfp4-chained-direct-output.md §9 Phase 0.
set -euo pipefail
cd "$(dirname "$0")"

CUTLASS_TAG="v4.8.0"
TS="$(date +%Y%m%d-%H%M%S)"
mkdir -p results
LOG="results/phase0_${TS}.log"
exec > >(tee "$LOG") 2>&1

echo "== phase0 bench @ ${TS} =="

# --- 1. toolchain / GPU checks ----------------------------------------------
# Prefer CUDA 12.8 toolchain (238 has 11.5/12.4/12.8; FP4 block-scaled needs 12.8+).
if [ -x /usr/local/cuda-12.8/bin/nvcc ]; then
  export PATH="/usr/local/cuda-12.8/bin:$PATH"
  export CUDACXX="/usr/local/cuda-12.8/bin/nvcc"
fi
if ! command -v nvcc >/dev/null 2>&1; then
  echo "DECISION: NO-GO (nvcc not found)"; exit 1
fi
NVCC_VER="$(nvcc --version | awk '/release/ {print $NF}' | tr -d ',')"
NVCC_MAJOR="${NVCC_VER%%.*}"
NVCC_MINOR="$(echo "${NVCC_VER}" | cut -d. -f2)"
echo "nvcc version: ${NVCC_VER} ($(command -v nvcc))"
# Need >= 12.8 for sm_120a + FP4 block-scaled (VEC16_UE4M3).
if [ "${NVCC_MAJOR}" -lt 12 ] || { [ "${NVCC_MAJOR}" -eq 12 ] && [ "${NVCC_MINOR}" -lt 8 ]; }; then
  echo "DECISION: NO-GO (nvcc ${NVCC_VER} < 12.8)"; exit 1
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
GPU_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1)"
echo "GPU: ${GPU_NAME} (sm_${GPU_CC//./})"
if [ "${GPU_CC%.*}" != "12" ]; then
  echo "DECISION: NO-GO (GPU compute capability ${GPU_CC}, need sm_120)"; exit 1
fi

# --- 2. CUTLASS --------------------------------------------------------------
# Reuse pre-cloned CUTLASS at ~/deps/cutlass (main @0b55a2f, already on 238)
# to save the ~13G-free disk; otherwise shallow-clone the pinned tag.
if [ -d "$HOME/deps/cutlass/include/cutlass" ]; then
  echo "reusing pre-cloned CUTLASS at \$HOME/deps/cutlass (main @0b55a2f)"
  mkdir -p _deps
  ln -sfn "$HOME/deps/cutlass" _deps/cutlass
fi
if [ ! -d _deps/cutlass/.git ]; then
  echo "cloning CUTLASS ${CUTLASS_TAG} ..."
  mkdir -p _deps
  git clone --depth 1 --branch "${CUTLASS_TAG}" \
    https://github.com/NVIDIA/cutlass.git _deps/cutlass
fi

# --- 3. build ----------------------------------------------------------------
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120a \
  -DCUTLASS_DIR="$PWD/_deps/cutlass"
cmake --build build -j"$(nproc)"

# --- 4. SFD byte-exact gate --------------------------------------------------
echo "--- sfd_compare ---"
SFD_OUT="results/sfd_compare_${TS}.log"
set +e
./build/sfd_compare | tee "$SFD_OUT"
SFD_RC=$?
set -e
if [ "${SFD_RC}" -ne 0 ] || ! grep -q 'sfd_compare,ALL,.*,PASS' "$SFD_OUT"; then
  echo "DECISION: NO-GO (SFD layout mismatch — CUTLASS tile_atom_to_shape_SFD" \
       "!= h3 swizzle; see ${SFD_OUT})"
  exit 2
fi
echo "SFD gate: PASS"

# --- 5. GEMM bench -------------------------------------------------------------
echo "--- bench_gemm ---"
BENCH_OUT="results/bench_gemm_${TS}.csv"
./build/bench_gemm 50 | tee "$BENCH_OUT"

# --- 6. decision (design doc §9 step 4) ---------------------------------------
python3 - "$BENCH_OUT" <<'PY'
import csv, sys

rows = [r for r in csv.reader(open(sys.argv[1])) if r and r[0] != "impl"]
times = {}
for r in rows:
    impl, shape = r[0], r[1]
    if impl in ("cutlass79b", "cublaslt_fp4"):
        times[(impl, shape)] = float(r[5])

fc1_cut = times.get(("cutlass79b", "FC1"))
fc1_lt = times.get(("cublaslt_fp4", "FC1"))
fc2_cut = times.get(("cutlass79b", "FC2"))
fc2_lt = times.get(("cublaslt_fp4", "FC2"))

if fc1_cut and fc1_lt:
    r1 = fc1_cut / fc1_lt
    print(f"FC1 ratio cutlass/cublaslt = {r1:.3f}")
else:
    print("missing FC1 rows"); sys.exit(3)
if fc2_cut and fc2_lt:
    r2 = fc2_cut / fc2_lt
    print(f"FC2 ratio cutlass/cublaslt = {r2:.3f}")

if r1 <= 1.05:
    print("DECISION: GO (full CUTLASS path)")
else:
    print("DECISION: GO-DEGRADED (cublasLt GEMM + FP32 passthrough swiglu-requant)")
PY
