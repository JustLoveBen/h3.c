# R-D：CUTLASS sm_120 INT8 GEMM 8 形状离线 bench 结果

**日期：** 2026-10-07。**机器：** 192.168.0.238（digen），RTX 5090 32GB sm_120，CUDA 13.1。
**任务：** 算子审计 E1 第 0 步——判定 CUTLASS 4.x EVT 融合 dequant+swiglu 是否值得进施工。

## 1. 判定结论：**E1 证伪（不进施工）**

8 个生产 GEMM 形状全部 **CUTLASS 慢于 cublasLt 41%–50%**（比值 0.50–0.59），远低于「进施工 ≥95% / 证伪 <90%」的阈值。

## 2. 8 形状对照表

| idx | m | n | k | 用途 | cublasLt TFLOP/s | CUTLASS TFLOP/s (best cfg) | 比值 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 3200 | 21504 | 5376 | 480P fc1 (SwiGLU fused) | 847.8 | 472.0 (128x128x64s4) | **0.557** |
| 1 | 3200 | 5376 | 7168 | 480P fc2 | 793.4 | 424.6 (128x128x64s4) | **0.535** |
| 2 | 3200 | 28672 | 5376 | 480P qkv/out | 854.5 | 474.6 (128x128x64s4) | **0.556** |
| 3 | 3200 | 5376 | 14336 | 480P out-adaln | 819.1 | 434.1 (128x128x128s3) | **0.530** |
| 4 | 4096 | 21504 | 5376 | 768P fc1 | 853.5 | 475.8 (256x128x64s4) | **0.557** |
| 5 | 4096 | 5376 | 7168 | 768P fc2 | 829.4 | 474.5 (128x256x64s4) | **0.572** |
| 6 | 4096 | 28672 | 5376 | 768P qkv/out | 858.2 | 472.5 (128x128x64s4) | **0.551** |
| 7 | 4096 | 5376 | 14336 | 768P out-adaln | 841.2 | 486.0 (128x256x64s4) | **0.578** |

每形状 300 次迭代（cublasLt）/300 次（CUTLASS），均带 3s GEMM 预热推 boost 时钟。
CUTLASS 每形状扫 4 个 tile 配置取最优。

## 3. 根因（硬件/软件钉死）

CUTLASS 4.x 公开 header 在 sm_120 上**没有 m16n8k32 的 `cutlass::arch::Mma` INT8 特化**：

- `include/cutlass/arch/mma_sm80.h` 里 INT8 m16n8k16 仅有 `OpMultiplyAddSaturate` 版本；
- m16n8k32 INT8 只在 `cute::` 命名空间下（CuTe DSL 用），不接入 `cutlass::gemm::device::GemmUniversal`；
- cublasLt 内部用的 `cutlass_80_i16832` 是 NVIDIA 私有的二进制 kernel，**公开 CUTLASS 源码树里没有对应模板**。

所以公开 CUTLASS 在 sm_120 上 INT8 GEMM 本征峰值就是 m16n8k16（~470 TFLOP/s），是 cublasLt m16n8k32（~840）的 **56%**。这不是 tile 配置/autotune 能救的——是指令吞吐 1/2 的硬约束。

仓里 C5b 三层验证的第二层（CUTLASS issue #2717：SM120 CollectiveBuilder static_assert 只支持 F8F6F4、不支持 INT8）与本 bench 实测完全互洽。

## 4. 对 roadmap 的影响

- **E1（CUTLASS EVT 融合 dequant+swiglu）：证伪。** GEMM 本征慢 45%（−0.6~−1.0s 收益被 −3.6s GEMM 本征损失完全淹没）。
- **E2（cublasLt 原地 + dequant kernel 合并减重，备胎 −0.2~−0.4s）：升为 T1-e 唯一算子层主刀。** 不动 GEMM 库、纯 kernel 工程、低风险，bit-identical 易证。
- **NVFP4 原生 GEMM（frontier 第 2 项，中期 2–3 周工程）的相对吸引力上升**——它是 sm_120 上唯一硬件级 2× GEMM 本征路径，且不依赖 INT8。

## 5. bench 代码路径

- `bench/cutlass_evt/bench_cublaslt_i8.c` — cublasLt 基线（生产配置逐字节对齐 `h3_gpu_cuda.cu:2441-2479`）
- `bench/cutlass_evt/bench_cutlass_int8_sm120.cu` — CUTLASS 4.x sm_120 bench（m16n8k16 Saturate，4 tile 配置扫描）
- `bench/cutlass_evt/scripts/setup_238.sh` — 238 环境 setup（CUTLASS clone + nvidia-cutlass pip）
- 238 上 CUTLASS 源：`/mnt/nvme1n1/workspace/cutlass`（v4.x，`CUTLASS_MAJOR=4`）
- 238 上 binary：`/mnt/nvme1n1/workspace/bench_cublaslt_i8` / `bench_cutlass_int8_sm120`

## 6. 复现命令（238 上）

```bash
NVCC=/home/yongbin/miniconda3/envs/yongbin/bin/nvcc
CU13=/home/yongbin/miniconda3/envs/yongbin/lib/python3.11/site-packages/nvidia/cu13
cd /mnt/nvme1n1/workspace

# cublasLt 基线（需 CUDA 13 nvcc + cu13 cublasLt）
$NVCC -O3 -x cu -std=c++17 -arch=sm_120a -I$CU13/include \
  bench_cublaslt_i8.c -L$CU13/lib -lcublasLt \
  -Xlinker -rpath -Xlinker $CU13/lib -o bench_cublaslt_i8
./bench_cublaslt_i8

# CUTLASS sm_120
$NVCC -O3 -std=c++17 -arch=sm_120a \
  -I/mnt/nvme1n1/workspace/cutlass/include \
  -I/mnt/nvme1n1/workspace/cutlass/tools/util/include \
  bench_cutlass_int8_sm120.cu -o bench_cutlass_int8_sm120
./bench_cutlass_int8_sm120
```

## 7. 过程中发现的两个工程坑（备忘）

1. **cublasLt 在 CUDA 12.4 nvcc 下对 sm_120 拒载**（heuristic returned=1 但 waves=-1.00，matmul=NOT_SUPPORTED）——必须用 CUDA 13 nvcc + cu13 cublasLt。生产 binary 链接的是 `nvidia/cu13/lib/libcublasLt.so.13`（pip 包），不是 `/usr/local/cuda-12.4`。
2. **GPU 时钟对 bench 影响 2 倍**：不预热时 cublasLt ~400 TFLOP/s，3s GEMM 预热后 793–858。生产 binary 自然预热（50 层 × 4 步），离线 bench 必须显式预热。
