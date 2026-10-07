# P4.2 Phase-0 — NVFP4 Chained Direct-Output Feasibility Bench

Offline feasibility bench for the NVFP4 chained direct-output project.
Design doc: `docs/research/2026-10-08-nvfp4-chained-direct-output.md` (§9 Phase 0).

**Target machine**: "238" — Linux x86_64, CUDA 13.1, RTX 5090 (sm_120a).
Nothing here compiles or runs on the local Mac; this directory is meant to be
scp'd to 238.

## What it does (§9 Phase 0, 4 steps)

1. **Build**: standalone CMake project against CUTLASS 4.x headers (cloned on
   demand into `./_deps/cutlass`).
2. **Bench** (`bench_gemm`): pure GEMM time of CUTLASS 79b-style sm_120
   NVFP4-in→NVFP4-out (with SFD generation) vs the extracted cublasLt FP4
   block-scaled GEMM (BF16 output, exactly the `h3_gpu_cuda.cu:5325-5561`
   configuration), on the production FC1/FC2 shapes. Plus a rel_l2 spot-check
   that both compute the same math on identical inputs.
3. **SFD gate** (`sfd_compare`): byte-exact comparison of the CUTLASS epilogue
   SFD buffer vs the h3.c production quantize kernel's SF output
   (`h3_gpu_cuda.cu:4403-4483`, extracted in `quant_act_extracted.cuh`), over
   FC1/FC2 shapes plus a ragged-M (15973) and a K-tail stress case.
   **100% byte match required.**
4. **Decision** (`run_phase0.sh`): GO / GO-DEGRADED / NO-GO per §9 step 4.

## Copy to 238

From the repo root on the local Mac:

```bash
rsync -av --exclude _deps --exclude build --exclude results \
  /Volumes/Ben/Digen/code/h3.c/experiments/nvfp4-chain-phase0/ \
  238:~/nvfp4-chain-phase0/
# or: scp -r /Volumes/Ben/Digen/code/h3.c/experiments/nvfp4-chain-phase0 238:~/
```

## Run on 238 (one shot)

```bash
ssh 238
cd ~/nvfp4-chain-phase0
./run_phase0.sh
```

The script checks nvcc ≥ 13 and sm_120, clones CUTLASS, builds, runs the SFD
gate (aborts NO-GO if not byte-exact), runs the bench, and prints the
decision. All output goes to `results/phase0_<timestamp>.log` plus
`results/sfd_compare_<ts>.log` and `results/bench_gemm_<ts>.csv`.

## Manual steps (if you want to do it piecemeal)

```bash
# 1. CUTLASS (pinned tag v4.8.0 — see "CUTLASS pin" below)
mkdir -p _deps
git clone --depth 1 --branch v4.8.0 https://github.com/NVIDIA/cutlass.git _deps/cutlass

# 2. Configure + build (sm_120a is required for the sm_120 UMMA kernels)
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120a \
  -DCUTLASS_DIR=$PWD/_deps/cutlass
cmake --build build -j$(nproc)

# 3. SFD layout gate — must print "sfd_compare,ALL,...,PASS"
./build/sfd_compare

# 4. GEMM bench — prints CSV rows impl,shape,M,N,K,ms_avg
./build/bench_gemm 50        # argument = timing iterations (default 50)
```

You can also point at an existing CUTLASS checkout with `-DCUTLASS_DIR=/path/to/cutlass`.

## Expected outputs

`bench_gemm` CSV (stdout):

```
impl,shape,M,N,K,ms_avg
cutlass79b,FC1,16000,28672,5376,<ms>
cublaslt_fp4,FC1,16000,28672,5376,<ms>
spotcheck,FC1,16000,28672,5376,rel_l2=<small, e.g. <0.02>
cutlass79b,FC2,16000,5376,14336,<ms>
cublaslt_fp4,FC2,16000,5376,14336,<ms>
spotcheck,FC2,...,rel_l2=<small>
```

`sfd_compare` prints per-shape `PASS/FAIL` with mismatch counts and the first
16 mismatches decoded as `(tile_row, tile_col, r128, cc)` per the §1.3 layout
formula `off = (tile_row*ncb_rd + tile_col)*512 + r128*4 + cc`.

## Decision criteria (design doc §9 Phase 0 step 4)

- **NO-GO**: toolchain/GPU check fails, build fails, or `sfd_compare` is not
  100% byte-exact (SFD layout mismatch — CUTLASS path cannot feed the
  downstream cublasLt GEMM; fall back to a swiglu-requant fusion kernel only,
  ~10% upside).
- **GO (full CUTLASS path)**: SFD gate PASS **and**
  `cutlass79b FC1 time <= 1.05 × cublaslt_fp4 FC1 time`.
- **GO-DEGRADED**: SFD gate PASS but CUTLASS FC1 GEMM is >5% slower than
  cublasLt → keep cublasLt as the GEMM provider and do
  "cublasLt GEMM (BF16 out) + FP32 passthrough swiglu-requant" (design doc §7
  degradation note, still ~150ms of the ~180–210ms upside).

The decision is computed and printed by `run_phase0.sh` as the last line.

## Notes / assumptions

- M is rounded up to 16000 (= 125×128) for the bench shapes; the ragged
  M=15973 case is covered by `sfd_compare` (`FC1_M15973`), where padding rows
  must carry SF=0 exactly.
- K=5376 and K=14336 are both multiples of 64 (required by the swizzle:
  `ncb = K/64` tile-columns) but not of 2048, so both already exercise the
  K-tail padded SF extent (`round_up(K/64, 32)` tile-columns); `KTAIL` adds a
  small-M explicit stress.
- The SFD byte-exact test makes the GEMM an identity over K-columns
  (weight = per-column identity), so the 79b epilogue and the h3 quantize
  kernel quantize *identical* FP32 values; any layout or rounding difference
  shows up as a byte mismatch. D data bytes are poisoned with 0xEE before the
  run to detect unwritten regions.
- `bench_gemm`'s spot-check uses `norm_constant = 1.0` (not the example's 2.0)
  so the CUTLASS dequantized output and the cublasLt BF16 output are the same
  math modulo epilogue rounding (FP4 requant of the accumulator on the CUTLASS
  side — this is exactly the "extra FP4 rounding" discussed in design doc
  §4.1, so rel_l2 on the order of 1e-2 is expected, not a bug).
- A `sfd_compare` FAIL means the CUTLASS epilogue and the h3 kernel differ in
  either the SF *layout* (offsets) or the quantization *recipe* (scale
  selection/rounding). The mismatch printout decodes offsets into
  `(tile_row, tile_col, r128, cc)`: a structured pattern (e.g. whole tile-cols
  shifted) indicates a layout difference; scattered same-offset value
  differences indicate a recipe difference. Both are NO-GO per the design
  doc, but the distinction decides whether a layout shim kernel could rescue
  the CUTLASS path.

## CUTLASS pin

Pinned tag: **v4.8.0** (released 2026-08-25), verified via
<https://github.com/NVIDIA/cutlass/releases> and
<https://docs.nvidia.com/cutlass/latest/CHANGELOG.html> on 2026-10-08.
Example 79b (`examples/79_blackwell_geforce_gemm/79b_blackwell_geforce_nvfp4_nvfp4_gemm.cu`)
at that tag was fetched and the collective/builder configuration in
`bench_gemm.cu` / `sfd_compare.cu` copied from it 1:1 (ping-pong schedule,
`LinCombBlockScaleFactor`, `Sm1xxBlockScaledOutputConfig::tile_atom_to_shape_SFD`).
Any 4.x ≥ 4.0 should work (the example and `Sm1xxBlockScaledOutputConfig`
exist since 4.0), but only v4.8.0 was source-verified.
