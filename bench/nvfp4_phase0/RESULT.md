# Phase 0：cuBLASLt NVFP4 GEMM sm_120 探测结果

**日期：** 2026-10-07（238 实测回填）。**机器：** 192.168.0.238（digen），RTX 5090 32GB sm_120，CUDA 13.1。
**任务：** NVFP4 W4A4 路径 Phase 0——判定 cuBLASLt FP4 GEMM 启发式在 sm_120 是否命中原生 block-scaled 硬件核（调研：`docs/research/2026-10-07-nvfp4-w4a4-path.md`）。

## 1. 判定结论：**GO（决定性）**

8/8 形状全部命中原生 FP4 block-scaled 硬件核，平均加速 **1.73×**（区间 1.71×–1.76×），远超 1.5× GO 阈值。FP4 吞吐 ~1480 TFLOP/s ≈ INT8（~850 TOPS）的 1.73 倍，确认非 dequant→BF16 fallback。进入 Phase 1。

判据（与 E1 同构）：

- `cublasLtMatmulAlgoGetHeuristic` 返回 0 算法，或 FP4 性能 < 0.8× INT8 → **NO-GO（cuBLASLt FP4 死）**，转 CUTLASS 79a 路径再测一轮；
- **≥ 1.5× INT8 → GO 到 Phase 1**（为激活量化开销留 0.5× 余量）；
- 0.8×–1.5× → 算入激活量化开销后净收益不足 → NO-GO 或转 MXFP4/MXFP8 混合再测。

## 2. 8 形状对照表（待回填）

| idx | m | n | k | 用途 | INT8 TFLOP/s | FP4 TFLOP/s | 比值 (FP4/INT8) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 3200 | 21504 | 5376 | 480P fc1 (SwiGLU fused) | 861.6 | 1487.0 | **1.73×** |
| 1 | 3200 | 5376 | 7168 | 480P fc2 | 797.9 | 1404.8 | **1.76×** |
| 2 | 3200 | 28672 | 5376 | 480P qkv/out | 857.0 | 1482.5 | **1.73×** |
| 3 | 3200 | 5376 | 14336 | 480P out-adaln | 823.3 | 1415.0 | **1.72×** |
| 4 | 4096 | 21504 | 5376 | 768P fc1 | 856.9 | 1486.1 | **1.73×** |
| 5 | 4096 | 5376 | 7168 | 768P fc2 | 835.7 | 1451.9 | **1.74×** |
| 6 | 4096 | 28672 | 5376 | 768P qkv/out | 863.4 | 1479.5 | **1.71×** |
| 7 | 4096 | 5376 | 14336 | 768P out-adaln | 846.7 | 1474.8 | **1.74×** |

**平均加速：1.73×（区间 1.71×–1.76×）**

每形状 500 次迭代 + 200 次预热，另有 3s GEMM burn 推 boost 时钟（与 E1 同协议）。
INT8 臂配置与生产逐字节一致（`CUBLAS_COMPUTE_32I`，复用 `bench/cutlass_evt/bench_cublaslt_i8.c`）。

## 3. 启发式行为（原生核 vs fallback 的关键证据，已实测）

- `cublasLtMatmulAlgoGetHeuristic` returned = **6**（每形状 6 个算法可选；E1 INT8 仅 algos=1）
- algo0 `workspaceSize` = **0**（无额外工作区；`wavesEstimated` 在 CUDA 13.1 头文件已移除，编译时省去该字段）
- `CUBLASLT_MATMUL_DESC_A_SCALE_MODE = VEC16_UE4M3` 设置返回码 = **SUCCESS**（受理，未报 INVALID_VALUE）
- `CUBLASLT_POINTER_MODE_DEVICE` 是否必需 = **是**（alpha/beta 用 device 指针，已按此实现）
- FP4 绝对吞吐 ~**1480 TFLOP/s**，为 INT8 臂（~850 TOPS）的 **1.73×**，也显著高于 C5b 的 INT8 峰值 745.7 TOPS
  - 远超 1100 TFLOP/s 量级 → **原生 m16n8k64 block-scaled 核命中确认**
  - 非 dequant→BF16 fallback（fallback 只会有 ~1.0× 且无吞吐翻倍）

## 4. SF 布局验证（2026-10-27 回填）

- **结论：生产曾用的线性 `(r%128)*4+(c%4)` 在 cuBLASLt sm_120 下不正确。**
- `diag_sf_layout.cu` 单点探针：`phys=3→row0, phys=7→row32, phys=19→row1` → **32×4 swizzle**
  `off = tile*512 + 16*((r%128)%32) + 4*((r%128)/32) + (c%4)`（tile 仍按 ncb_rd 铺）。
- 线性配方与实测差 504/512 槽 → C001 NVFP4 mosaic 根因；已修（见 `docs/research/2026-10-27-nvfp4-mosaic-sf-swizzle.md`）。
- Phase 0 吞吐 GO 仍成立（吞吐不依赖布局正确性）。

## 5. 对 roadmap 的影响（已判定 GO）

- **GO 成立 → 进入 Phase 1**：GEMM 后端接入 cuBLASLt FP4，权重离线 NVFP4 量化 + 激活在线量化 kernel + `h3_dit_w4a8.c` 4 调用点 env 切换。
- Phase 0 已证 cuBLASLt FP4 路径可用且命中原生核，无需转 CUTLASS 79a 自建核（省时）。
- 预期 DiT GEMM 时间约降至 1/1.73（GEMM 占比高时 DiT 总时长有望 −25%~−40%，需 Phase 1 接入后实测验证，含激活量化开销）。

## 6. bench 代码路径

- `bench/nvfp4_phase0/bench_cublaslt_fp4.cu` — FP4 臂 + INT8 对照臂（单文件双轨）
- `bench/nvfp4_phase0/Makefile` — 238 构建（CUDA 13 nvcc + cu13 cublasLt，sm_120a）

## 7. 复现命令（238 上）

```bash
cd /mnt/nvme1n1/workspace   # 或本仓同步目录
cd bench/nvfp4_phase0
make            # NVCC/CU13 默认值已对齐 238 的 CUDA 13 pip 栈，可用 make NVCC=... 覆盖
./bench_cublaslt_fp4            # fp4 + int8 双臂，8 形状
./bench_cublaslt_fp4 fp4        # 仅 FP4 臂
./bench_cublaslt_fp4 int8       # 仅 INT8 基线臂
```

输出末尾直接打印 markdown 加速比表，粘贴进本文件第 2 节。

## 8. 已知注意事项（写入时未实测）

1. FP4 API 需 cuBLASLt ≥ 12.8；238 生产 binary 链接 `nvidia/cu13/lib/libcublasLt.so.13`（pip 包），Makefile 默认指向该路径（与 E1 一致）。
2. `CUDA_R_4F_E2M1` 打包为 2 元素/字节，packed buffer 尺寸 = N×K/2；本 bench 用 lo-nibble=偶数 K 约定（纯计时无所谓，Phase 1 需与权重打包约定对齐）。
3. 若 heuristic 返回 0 算法：先确认 nvcc/运行时确为 CUDA 13（E1 踩过 CUDA 12.4 对 sm_120 拒载的坑），再判 NO-GO。
