# P4.2 NVFP4 Chained Direct-Output（CUTLASS 79b 式）— 消除激活量化中转

- 日期：2026-10-08
- 状态：设计（无代码改动）
- 目标硬件：RTX 5090（sm_120, Blackwell GeForce）
- 前置：P4.1 全 W4A4 DiT 已上线（200/200 dispatch、零 NaN、8 步 1344×768 5s 91.5s vs INT8 104.0s）
- 相关：`docs/research/2026-10-07-nvfp4-w4a4-path.md`（P2.1/P2.2 精度结论）、`docs/nvidia_quant_contract.md`、`docs/nvidia_optimization_plan.md`

---

## 1. 现状机理：激活量化中转（transit）

### 1.1 当前每个 FP4 GEMM 的数据流

`h3_cuda_fp4_gemm_run`（`h3_gpu_cuda.cu:5964-6119`）：

```
FP32 act d_x ──(memsetAsync d_sf_x)──┐
                                     ▼
              h3_cuda_quantize_act_nvfp4_dev   ← 146.4ms 全家桶（quantize）
              FP32 → packed E2M1 + swizzled UE4M3 SF
                                     ▼
              h3_cuda_gemm_fp4_bs_dev (cublasLt FP4 block-scaled MMA)
              A=w_fp4[T] × B=x_fp4[N] → C = BF16 [m,n]    ← 678.7ms（45.2%）
                                     ▼
              BF16 epilogue（swiglu / gate / dequant，h3_gpu_cuda.cu:631-682）
              BF16 → FP32 d_y_f32                     ← 105.8ms（fp4_dequant）
                                                       + 90.0ms（fp4_swiglu）
                                                       + 92.2ms（fp4_gate）
                                     ▼
              下一个 GEMM 再把这个 FP32 量化回 FP4   ← 这就是中转
```

每个 GEMM 独立闭环：`FP32 in → quantize → GEMM → BF16 out → epilogue → FP32 out`。
两个相邻 GEMM 之间存在一次 **BF16→FP32→requantize** 的精度/带宽往返。

### 1.2 nsys 实测（2-step / 1s / 768P，DiT GPU busy 1500ms）

| 类别 | kernel | 时间 | 占 DiT busy |
|---|---|---:|---:|
| GEMM | cublasLt FP4 MMA | 678.7 ms | 45.2% |
| 窄中转 | quantize (146.4) + fp4_dequant (105.8) | **252.2 ms** | **16.8%** |
| 宽中转 | 窄中转 + fp4_gate (92.2) + fp4_swiglu (90.0) | 434.4 ms | 29.0% |
| SDPA | cudnn flash | ~18 ms | 1.2% |
| 其他 | swiglu 8.8%, qkv_rope 6.7%, gate_adaln 4.6%, adaln 2.2%, casts 2.2% | — | — |

窄中转 252ms / 1500ms ≈ 17%：如果完全消除，8 步 91.5s 基线投影到 DiT ≈ 76–80s。
P4.2 的目标就是消灭这条 `quantize + BF16 落盘 + dequant` 链。

### 1.3 一个「下一个 GEMM 可直接吃」的 FP4 激活 buffer 长什么样

由 `h3_cuda_quantize_act_nvfp4_kernel`（`h3_gpu_cuda.cu:4403-4483`）与
`h3_cuda_lt_gemm_fp4_bf16`（`h3_gpu_cuda.cu:5325+`）共同定义：

- 数据：`m_pad * (k/2)` 字节，`m_pad = round_up(m,128)`，每字节高低两个 E2M1 nibble
  （row-major，列内偶数 lane 收 odd lane 的低 nibble 打包）。
- SF（UE4M3）：按 128 行 × 2048-K-elem 的 tile 网格 swizzle。对每 128 行 tile：
  `sf_ncb = round_up(k/64, 32)` 个 tile 列，每 tile 列 512 字节
  （128 行 × 4 字节），即 offset = `(tile_base + tc)*512 + r128*4 + cc`。
  行间 major、微块按 `g/4` 分 tile 列、`g%4` 为字节内 cc。
- NaN 防护：padding 行与 K-tail 的 SF 必须显式置 0（`>=0xFE` 的 UE4M3 字节是 NaN），
  所以每个 GEMM 前有 `cudaMemsetAsync(d_sf_x, 0, ...)` 兜底，量化 kernel 自己也覆盖
  有效区 + padding 行 + K-tail。
- 输入侧 cublasLt 要求：`CUDA_R_4F_E2M1` + `CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3`，
  M 必须 128 对齐（否则 heuristic st=7 INVALID_VALUE）。
- 精度约定：per-16 微块 SF 在 MMA 内部消费；per-tensor 第二级 scale
  （`weight->scale2 × act_scale`，当前 act_scale=1.0）留在 epilogue 里乘。

**任何「直接输出 FP4」的方案，其 SFD（输出 SF）布局必须与上面完全一致**，
否则下一级 cublasLt 输入会读到错位的 scale 或直接 NaN。这是 P4.2 最大的工程约束。

---

## 2. 硬件/栈可行性：cublasLt 能不能 FP4 出？还是必须 CUTLASS 79b？

### 2.1 cublasLt：输入侧 FP4 ✓，输出侧 FP4（带 SF 生成）✗（截至 CUDA 12.8/12.9 公开 API）

已验证现状（本仓代码 + 外部证据）：

- 输入侧：`VEC16_UE4M3` scale mode 与 `CUDA_R_4F_E2M1` 是 CUDA 12.8 引入的 Blackwell
  原生 block-scaled 模式（[CUDALibrarySamples#310](https://github.com/NVIDIA/CUDALibrarySamples/issues/310)），本仓已跑通。
- 输出侧：TensorRT-LLM 的 cuBLASLt NVFP4 后端（[PR #7943](https://github.com/NVIDIA/TensorRT-LLM/pull/7943)、
  [commit 928247a](https://github.com/NVIDIA/TensorRT-LLM/commit/928247a3f9288dd84857e1d545c33b248f16cf72)）
  对 D 端设置的是：
  - `D_SCALE_MODE = SCALAR_32F`、`D_OUT_SCALE_MODE = SCALAR_32F`，
  - `D_SCALE_POINTER / D_OUT_SCALE_POINTER = nullptr`。

  即 **cublasLt 的 NVFP4 路径不把 D 量化为 FP4、也不生成 per-16 输出 SF**；
  D 端只有 per-tensor 标量语义。即使把 C layout 建成 `CUDA_R_4F_E2M1`，
  也没有 `VEC16_UE4M3` 的 D/SFD 模式可设——无法产出「下一级 GEMM 可直接消费」的
  FP4+swizzled-SF 对。
- nvmath-python 例 37（[example37_nvfp4_scale_layout.py](https://github.com/NVIDIA/nvmath-python/blob/main/examples/linalg/advanced/matmul/example37_nvfp4_scale_layout.py)）
  演示的 `result_type=CUDA_R_4F_E2M1, block_scaling=True` 确实能拿到 FP4 输出 +
  `aux["d_out_scale"]`（per-16 UE4M3），但那是 **cuBLASDx 设备端 API**，
  且其 `d_out_scale` 是线性布局（需 `expand_block_scale` 广播），
  **不是** cublasLt 期望的 128×2048 swizzle tile 布局；也不是 host 侧
  `cublasLtMatmul` 的能力。

**结论：cublasLt 路线（现 GEMM 提供方）无法完成 direct-output。**

### 2.2 CUTLASS 4.x example 79b：sm_120 NVFP4-in → NVFP4-out + SFD 生成 ✓

[CUTLASS example 79b `79b_blackwell_geforce_nvfp4_nvfp4_gemm.cu`](https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79b_blackwell_geforce_nvfp4_nvfp4_gemm.cu)
（RTX 50 系列 / sm_120 专用）正是这个目标内核：

- 注释原文：*"The kernel outputs quantized fp4 values with scale factors that will be
  the input of another GEMM … optimized for the GeForce RTX 50 series GPUs."*
- 机制：epilogue 用 `LinCombBlockScaleFactor`，在写 FP4 的 D 的同时生成 SFD：
  ```cpp
  using Sm1xxBlockScaledOutputConfig = cutlass::detail::Sm1xxBlockScaledOutputConfig;
  auto layout_SFD = SfdOutputCfg::tile_atom_to_shape_SFD({M,N,K,1});
  arguments.epilogue.thread.block_scale_factor_ptr = block_SFD.device_data();
  arguments.epilogue.thread.norm_constant_ptr  = block_Normconst.device_data();
  ```
  `norm_constant` 是全矩阵归一化常数（例中取 2），SFD 经 `tile_atom_to_shape_SFD`
  生成的布局即 sm_1xx block-scaled 标准 swizzle——与 cublasLt 输入 SF 的
  128-row × 2048-K-elem tile 语义同族（79a/b 的 SFA/SFB 走同一个
  `Sm1xxBlkScaledConfig`，本仓量化 kernel 正是按该布局实测对齐的）。
  Phase 0 必须用逐字节对比证明 `layout_SFD` == 本仓 SF 布局，不能假设。
- 约束（79 系列注释）：sm_120 不支持 TMA multicast（cluster 必须 1x1x1）、
  不支持 dynamic datatypes；warp-specialized persistent kernel（cooperative/ping-pong）。
- 旁证：example 72b 是同款内核的 sm_100 版本；[cutlass#3343](https://github.com/NVIDIA/cutlass/issues/3343)
  确认 79d（grouped 变体）在 sm_120 上可实例化并运行，进一步佐证 sm_120
  NVFP4 block-scaled 内核族可用。

**结论：direct-output 必须走 CUTLASS 79b 式内核（或等价自研 sm_120 UMMA 内核），
cublasLt 保持为非链式 GEMM 的 fallback。**

### 2.3 为什么不自研 UMMA epilogue

本仓已有 swizzle 布局的完整知识与 E2M1/UE4M3 编解码（`h3_e2m1_nibble`、
`h3_f32_to_ue4m3`），自研 epilogue 写 FP4+SF 本身不难；难的是要在 CUTLASS
级别的 tensor-core 主循环旁重写一个性能相当的 FP4 MMA。79b 已给出 sm_120 调优
（persistent + warp-specialized + cluster-launch-control 调度），直接集成它的
collective 比自己写 mainloop 风险低一个量级。自研仅在「CUTLASS 依赖不可接受」
时作为退路（见 §8）。

---

## 3. 可链边分析：哪些 GEMM→GEMM 边真的能链

### 3.1 DiT block 数据流（50 blocks；HIDDEN=5376, INNER=7168(56×128), FFN=14336）

依据 `block_phase_qkv` / `block_phase_attn` / `block_phase_out_mlp`
（`h3_dit_w4a8.c:3047-3560`）：

```
d_x ─adaln_attn(slot0/1)→ d_a ─QKV GEMM [5376→21504]→ d_qkv
                                    │  (+LoRA residual, FP32 域)
                                    ▼
                      qkv_rope + RMSNorm → FP16 → SDPA (cudnn) → d_attn (FP32)
                                    │
d_x(res) ───────────────┐           ▼
                        └─OUT GEMM [7168→5376]+gate(slot2) epilogue → d_b
                                    │
d_b ─adaln_mlp(slot3/4)─────────► d_a ─FC1 GEMM [5376→28672]+swiglu epilogue → d_act [.,14336]
                                                            │  (+LoRA, BF16 域)
                                                            ▼
d_b(res) ───────────────┐      FC2 GEMM [14336→5376]+gate(slot5) epilogue → d_y
                        └──────────────────────────────────────────────┘
```

每个 GEMM 的 epilogue 类型（`nvfp4_gemm_dispatch` kind 0..3）：
QKV=dequant、OUT=gate(residual=d_x, slot2)、FC1=swiglu、FC2=gate(residual=d_b, slot5)。

### 3.2 逐边判定

| 边 | 中间算子 | 可链？ | 理由 |
|---|---|---|---|
| QKV → OUT | qkv_rope + RMSNorm + FP16 cast + **SDPA** | ✗ | SDPA 是非线性全局算子且走 FP16/cudnn；OUT 的输入是注意力输出 d_attn，与 QKV 输出无逐元素关系。链式直出对此边无意义。 |
| OUT → FC1 | **AdaLN(mlp)**（LayerNorm + 逐行 scale/shift，slot3/4） | ✗（本期不链） | 中间是 row-wise normalization：需要整行统计量（mean/var），SF 是 per-16 局部量，把 AdaLN 折进 FP4 域等于在 FP4 上做 LayerNorm——误差与实现复杂度都不划算。**Phase 2 候选**：OUT epilogue 在 FP32 域完成 gate 后直接做 AdaLN 并输出 FP4+SF（一次 kernel 融合，不经 d_b/d_a 落 FP32 后再量化），可行但精度风险集中在 LN 的 FP4 化，需单独验证门。 |
| **FC1 → FC2** | **swiglu（逐元素）** | **✓（本期主目标）** | 唯一「GEMM→逐元素→GEMM」结构。FC1 输出 [m, 2·FFN]，swiglu 逐列配对 `silu(g)*u`，FC2 立即消费 [m, FFN]。逐元素算子可以干净地放在 direct-output epilogue 之后、FP4 域内完成。 |
| FC2 → 下一 block QKV | adaln_attn（LN） | ✗ | 同 OUT→FC1，LN 阻断。 |

### 3.3 结论：P4.2 的可施工边 = FC1→FC2（每 block 1 条，共 50 条）

- 每 block 恰好一条 FC1→FC2 链，50 blocks → 每步省 50 次「FC1 侧 quantize +
  BF16 落盘 + dequant」中的 requantize 部分（见 §7 收益拆分）。
- 同时消灭独立的 `swiglu` kernel（nsys 中 swiglu 8.8% 里的 FC1→FC2 份）——
  它并入 FC1 直出 epilogue 的后续 kernel（FP4 域 swiglu），
  fp4_swiglu（90.0ms，宽中转的一部分）整个被替代。

---

## 4. 精度风险评估：FP4 域 swiglu 与「FP4 驻留更久」

### 4.1 FP4 域 swiglu 的数值链路

拟议链路：`FC1 GEMM(FP4×FP4→FP32 acc) → [epilogue: +LoRA, 不落盘] →
FP4域swiglu kernel: 读 gate/up 两半的 FP4+SF → 反量化到 FP32 寄存器 →
silu(g)*u → 按 per-16 amax 重量化 → 写 FP4+SF（FC2 输入布局）→ FC2 GEMM`。

相对现状的变化点：

1. **少一次往返，多一次驻留**：现状是 acc(BF16)→FP32→requantize(FP4)。
   链式是 acc(FP32 寄存器)→swiglu(FP32 寄存器)→quantize(FP4)。
   **量化次数反而相同**（都是一次 per-16 量化喂 FC2）——因为 swiglu 输出本来就要
   量化给 FC2。链式把「先 BF16 化 acc、再 FP32 化、再量化」换成
   「FP32 寄存器直通、量化一次」。**精度不降反微升**：省掉了 acc 的 BF16 舍入
   （BF16 8 位尾数 vs FP32 24 位），swiglu 输入更精确。
2. **风险点不在 swiglu，在 epilogue 的 FP4 直出**：若让 FC1 的 GEMM epilogue
   直接输出 FP4（79b 式，gate/up 两半各带 SFD），则 acc 先被 FP4 化，
   swiglu 在 FP4 化的 gate/up 上运算 → **多一次 FP4 舍入**。
   E2M1 有效尾数约 2–3 bit，per-16 块内 SNR 约 14–18dB。
   silu(g)·u 是两个含噪量相乘，误差近似相加（相对误差 ~2×）。
   
   **因此设计取 4.2 的两段式而非一段式**：GEMM epilogue 保持 FP32/BF16 高精度
   （甚至可以不落盘，寄存器传给下一段的语义由 kernel 边界保证），
   FP4 化只发生在 swiglu 出口。即 **swiglu 内核 = FP4-in 是假象，
   实际是 acc 直通**（见 §5.2 方案 B）。一段式（79b 原样直出 gate/up 的 FP4）
   作为 Phase 2 的可选实验。

3. **Chaining 是否放大 8 步发散**：P2.1/P2.2（`2026-10-07-nvfp4-w4a4-path.md`）
   已定论——单步 rel_l2 0.0206 健康、8 步 rel_l2 1.33 是**混沌放大但视觉有效**
   （scene-score max 0.4184≈gold 0.4216，无结构坍塌）。P4.2 改变的是
   「每步速度场的实现方式」，单步误差预算与现状同量级（甚至更小，见第 1 点），
   不引入新的误差类别。验证门照 P2.2 的视觉协议，不做 bit-identical 要求。

### 4.2 误差预算

| 项 | 现状（P4.1） | P4.2 链式 | 判定 |
|---|---|---|---|
| acc 舍入 | FP32→BF16（8bit 尾数） | FP32 寄存器直通 | 改善 |
| swiglu 输入精度 | BF16 级 | FP32 级 | 改善 |
| FC2 输入量化 | per-16 amax→E2M1，一次 | 同，一次 | 持平 |
| 单步 rel_l2 期望 | 0.0206 | ≈0.020（±10%） | 验证门确认 |

---

## 5. 施工面（file-by-file）

### 5.1 新文件：`h3_dit_nvfp4_chain.cu`（CUTLASS 集成层）

承载 79b 式 sm_120 内核的实例化与本仓 cublasLt 风格的 host 包装。
新建独立 .cu 而非塞进 `h3_gpu_cuda.cu`：CUTLASS headers 编译慢、
且需要 C++17 + sm_120 arch flag，隔离编译单元避免拖慢主文件。

```cpp
// ---- 79b collective 类型别名（细节以 CUTLASS 4.x example 79b 为准）----
// using ChainGemmFC1 = cutlass::gemm::device::GemmUniversalAdapter<...sm120 nvfp4 nvfp4 ...>;

// FC1 直出（方案 A，Phase 2 可选）：
//   acc → (+LoRA) → FP4[gate|up] + SFD_gate + SFD_up
// FC1 直出（方案 B，Phase 1 主选）：
//   acc → (+LoRA) → FP32 紧排 scratch [m, 2*ffn]（不落全局 FP32 大数组则走 SMEM→
//   受 99KB 限制见 §8，故 Phase 1 仍落一次 FP32，但省掉 BF16 化与 requantize）
int h3_cutlass_nvfp4_fc1_run(h3_cuda *cuda,
        const h3_cuda_nvfp4_weight *w_fc1,   // FP4 权重 + SF + scale2（复用）
        const uint8_t *d_x_fp4, const uint8_t *d_sf_x,  // 已量化激活（AdaLN 输出）
        size_t m,                            // seq（内核内做 128 对齐）
        const float *d_lora,                 // 可空
        float *d_acc_f32,                    // [m, 2*ffn] 方案 B 输出
        uint8_t *d_y_fp4, uint8_t *d_sf_y,   // 方案 A 输出（Phase 2）
        char *error, size_t error_size);

// FP4 域 swiglu（方案 B 的配套；读 FP32 acc 写 FC2-ready FP4+SF）：
//   复用 h3_cuda_quantize_act_nvfp4_kernel 的 SF swizzle 写逻辑，
//   每个 16 元素微块：读 acc 的 gate/up 对 → silu(g)*u → amax→UE4M3→E2M1。
int h3_nvfp4_swiglu_requant_dev(h3_cuda *cuda,
        const float *d_acc_f32, size_t m, size_t ffn,
        uint8_t *d_act_fp4, uint8_t *d_sf_act,
        char *error, size_t error_size);
```

### 5.2 修改 `h3_gpu_cuda.cu`

- `h3_cuda_fp4_gemm_run`：不动。非链式 GEMM（QKV/OUT）与 fallback 全走它。
- `h3_cuda_gemm_fp4_bs_dev` / `h3_cuda_lt_gemm_fp4_bf16`：不动（FC2 的 GEMM 输入
  从「d_act FP32」换成「d_act_fp4 + d_sf_act」，走一条新的 pre-quantized 入口）：

```c
// FC2 预量化入口：跳过 memset+quantize，直接 GEMM+gate epilogue。
int h3_cuda_gemm_fp4_pre_q_dev(h3_cuda *cuda,
        const h3_cuda_nvfp4_weight *weight,
        const uint8_t *d_x_fp4, const uint8_t *d_sf_x,
        size_t m, const float *d_residual, const float *d_modulation,
        const uint32_t *d_row_map, size_t slots, size_t gate_slot,
        const float *d_lora, float *d_y_f32,
        void *d_bf16_scratch, double *gemm_ms,
        char *error, size_t error_size, const char *tag);
```

（复用现有 gate epilogue kernel，仅省去 quantize 段；scratch 布局中
x_fp4/sf_x 由调用方提供，本函数只用 bf16 区。）

### 5.3 修改 `h3_dit_w4a8.c`（调度与内存）

- `nvfp4_ensure_scratch`：扩 scratch——追加 `2*ffn` FP32 acc（方案 B）+
  FP4 激活对（`m_pad*ffn/2` + SF 区，SF 按 K=14336 算）。当前 widest 已按
  `nvfp4_gemm_scratch_bytes(seq, 28672, 14336)` 预留，新增约
  `seq*28672*4 + seq*14336/2 + SF(14336)` 字节；seq≈15973 时约 +230MB，
  9.3GiB 峰值 / 22GiB 余量下无压力。
- `block_phase_out_mlp` 的 FC1/FC2 段（`h3_dit_w4a8.c:3361-3498`）：
  在 `ctx->nvfp4_on && h3_dit_nvfp4_chain_on()` 分支下：

```
fc1: h3_cutlass_nvfp4_fc1_run(..., d_a 的 FP4 量化仍由现有 quantize 完成,
     → d_acc_f32 [m,2ffn])
     h3_nvfp4_swiglu_requant_dev(d_acc_f32 → d_act_fp4 + d_sf_act)
fc2: h3_cuda_gemm_fp4_pre_q_dev(..., d_act_fp4, d_sf_act, gate slot5 → d_y)
```

  原有 `nvfp4_gemm_dispatch(kind=2/3)` 路径完整保留作为 fallback。
- 开关：`H3_DIT_NVFP4_CHAIN`（默认 0）。`h3_dit_nvfp4_chain_on()` 读 env，
  与 `ctx->nvfp4_on` 正交；链式任一步失败（CUTLASS 未编译/heuristic 失败）
  打 stderr 回退到 P4.1 路径，语义与现有 `dit_lazy_int8` fallback 一致。

### 5.4 构建系统

- `CMakeLists.txt`：新增 `h3_dit_nvfp4_chain.cu` 编译单元，
  `target_compile_options` 加 `-gencode arch=compute_120a,code=sm_120a`
  （79b 需要 sm_120a 的 UMMA 特性宏 `CUTLASS_ARCH_MMA_SM120_SUPPORTED`）。
- CUTLASS 以 header-only submodule / FetchContent 引入（pin 一个 4.x tag，
  例如 v4.0+；79b 需要含 `Sm1xxBlockScaledOutputConfig` 的版本）。
- 仅当 `H3_ENABLE_NVFP4_CHAIN=ON`（默认 ON，detect CUTLASS 失败自动 OFF）
  时编译该单元，保证无 CUTLASS 环境仍可构建。

---

## 6. 验证门（不做 bit-identical）

沿袭 P2.2 的协议（latent 指标对混沌采样无判别力，只信解码后视频）：

1. **数值烟测（每次开发迭代）**：链式 FC1→FC2 与现状路径单 block 对比，
   `rel_l2(d_y_chain, d_y_base) < 0.03`（单步误差预算 0.0206 的 1.5 倍余量）。
   超阈即阻断合入。
2. **端到端 A/B（合入门）**：同 seed / 同 prompt（p01_sprint 等 waffle 集）、
   1344×768/5s/8 步/cfg1，`H3_DIT_NVFP4_CHAIN=0/1` 两臂：
   - 单帧质量：与 gold（INT8）帧的 SSIM 分布对比——链式臂不得出现
     gold 臂 SSIM 分布之外的低分异常帧（沿用 P2.2 的判定方式，
     不要求对 gold 逐帧高 SSIM，要求无结构坍塌/色块/噪声帧）。
   - 时域：ffmpeg scene-score 序列，max/mean 与 gold 偏差 < 15%。
3. **性能门**：8 步 1344×768 5s DiT 时间 ≤ 84s（对应 ≥8% 端到端改善；
   理论上限见 §7）。达不到则记录原因（CUTLASS 内核若比 cublasLt 慢会吃掉收益）。
4. **回退门**：`H3_DIT_NVFP4_CHAIN=0` 时输出与 P4.1 完全一致
   （该路径代码未被触碰，自动满足）。

---

## 7. 预期收益（从 nsys 数字推）

每 block 4 个 GEMM，50 blocks；FC1→FC2 链占全部 GEMM 边的 1/3（4 GEMM 中
涉及 2 个）。窄中转 252.2ms 的可消除份额：

| 项 | 现状归属 | 链式后 | 节省 |
|---|---|---|---|
| FC2 输入 quantize（146.4ms 的 ~1/4：4 个 GEMM 中 FC2 的 K=14336 较大，按字节估 ~30–35%） | quantize | 并入 swiglu-requant kernel（仍需量化，但读 FP32 寄存器级数据、免一次全局往返） | ≈ 半省：量化计算仍在，省的是独立 kernel 的 launch + FP32 中间数组读写 ≈ 25–40ms |
| FC1 侧 BF16 落盘 + swiglu 读回（fp4_swiglu 90.0ms + swiglu kernel 的部分） | 宽中转 | CUTLASS epilogue 直接产 FP32 紧排（或 Phase 2 直接 FP4），swiglu-requant 一趟完成 | 90ms 的 fp4_swiglu 全免；独立 swiglu（8.8%≈132ms 中的 FC1 份）全免，合并为一个轻 kernel（估 30–50ms） |
| FC1 的 BF16 acc 写 + FP32 化（fp4_dequant 的一部分不计，FC1 走 swiglu 不走 dequant） | — | acc 直接 FP32 落盘（方案 B 仍落一次 FP32）或不落盘（方案 A） | 方案 B 小额；方案 A 显著 |

**现实估计（方案 B，Phase 1）**：节省 ≈ (fp4_swiglu 90 + 独立 swiglu ~130 +
quantize 中 FC2 份的部分 ~30) − swiglu-requant 新 kernel ~40 ≈ **180–210ms**，
即 DiT busy 的 **12–14%**；8 步 91.5s → **79–81s**。
窄中转名义上限 17% 打七折的原因：quantize 计算本身不可免（FC2 输入必须
FP4+SF），只能免其 kernel 边界与中间数组往返；且 CUTLASS FC1 内核若比
cublasLt 慢 3–5% 会再吃掉 20–30ms，Phase 0 的 bench 必须先确认
CUTLASS ≥ cublasLt 的 FC1 GEMM 速度，否则方案退化为「cublasLt GEMM（BF16 out）+
FP32 直通 swiglu-requant」（仍能拿 ~150ms，只是 epilogue 不再融合 LoRA/swiglu）。

**方案 A（Phase 2，全 79b 直出）**：再免 FP32 acc 落盘与 swiglu-requant 的
读取，逼近 220–250ms ≈ 15–17%，但引入 §4.1 第 2 点的额外 FP4 舍入，
需单独过验证门。

---

## 8. 风险与回退

| 风险 | 影响 | 缓解/回退 |
|---|---|---|
| **CUTLASS 依赖**：引入 header-only 但 ~200MB 的 submodule；编译时间显著上升；pin 版本升级带来 API 漂移（79b 用 `Sm1xxBlockScaledOutputConfig`，4.x 内部 detail 命名空间可能改名） | 构建/维护 | 独立编译单元 + `H3_ENABLE_NVFP4_CHAIN` 开关；无 CUTLASS 自动 OFF；运行时 `H3_DIT_NVFP4_CHAIN=0` 完全回到 P4.1 |
| **SFD 布局不匹配**：CUTLASS `tile_atom_to_shape_SFD` 与本仓实测 swizzle 若在任何 corner（K-tail、M-padding、cc 字节序）不一致，下一级 cublasLt 读出 NaN 或错 scale | 正确性 | Phase 0 逐字节对比 harness（同输入跑 CUTLASS SFD vs 本仓 quantize kernel，要求 100% 一致才进入 Phase 1）；`h3_nvfp4_debug_scan_sf` 已有现成扫描器 |
| **CUTLASS FC1 GEMM 比 cublasLt 慢**：cublasLt FP4 是 NVIDIA 调优库，CUTLASS 示例内核在 K=5376/N=28672 形状上未必赢 | 收益被吃 | Phase 0 bench 先行；慢 >5% 则采用「cublasLt GEMM + FP32 直通」退化方案（不引 CUTLASS） |
| **sm_120 SMEM 约束**：sm_120 每 block 可用 SMEM ~99KB（GeForce 无 SM100 的 228KB），79b tile 选择受限；M=15973 的 persistent 调度占用需实测 | 性能/可行性 | Phase 0 用 79b 默认 tile 实测；若 SMEM 爆，降到更小 TileShape（示例提供多档），性能损失计入 §7 折扣 |
| **FP4 域 swiglu 精度**（仅方案 A） | 画质 | §4 误差预算 + §6 验证门；方案 B 不受影响 |
| **LoRA 路径**：FC1 的 LoRA 目前在 BF16 swiglu epilogue 内加；链式方案 B 改在 CUTLASS epilogue（或 acc 落盘后的 requant kernel）内加，行为变更点需对齐 | 正确性 | 无 LoRA 配置先验证；LoRA 配置纳入数值烟测 |

**总回退**：`H3_DIT_NVFP4_CHAIN` 默认 0，P4.1 路径代码零改动，
任何阶段出问题关 env 即回到 91.5s 基线。

---

## 9. 分阶段施工步骤

### Phase 0 — 离线可行性 bench（预计 0.5–1 天，纯实验不改主仓逻辑）
1. 拉 CUTLASS 4.x，独立小工程编译 example 79b（sm_120a）。
2. Bench：FC1 形状（M≈15973, N=28672, K=5376）与 FC2（N=5376, K=14336）
   上 CUTLASS 79b vs 本仓 cublasLt FP4 的 GEMM 纯时间。
3. SFD 布局逐字节对比：同随机输入，CUTLASS SFD vs `h3_cuda_quantize_act_nvfp4_kernel`
   输出（含 K-tail/padding 行），100% 一致才放行。
4. 产出：GO / GO-退化（cublasLt+FP32 直通）/ NO-GO（放弃 CUTLASS，只做
   swiglu-requant 融合 kernel，收益限于 ~10%）。

### Phase 1 — 方案 B：FC1→FC2 单链（主施工，预计 2–3 天）
1. `h3_dit_nvfp4_chain.cu`：CUTLASS FC1（acc FP32 直出）+ swiglu-requant kernel。
2. `h3_gpu_cuda.cu`：`h3_cuda_gemm_fp4_pre_q_dev`。
3. `h3_dit_w4a8.c`：`H3_DIT_NVFP4_CHAIN` 分支 + scratch 扩容 + fallback。
4. 数值烟测（§6.1）→ 端到端 A/B（§6.2）→ nsys 复测确认中转时间下降 → 性能门。

### Phase 2 — 拓宽与深化（各独立可关，预计各 1–2 天）
1. **方案 A**：FC1 epilogue 直出 FP4+SFD（79b 原义），swiglu 改 FP4-in/FP4-out，
   消灭 FP32 acc 落盘。独立 env `H3_DIT_NVFP4_CHAIN=2`，过 §6 全部门。
2. **OUT→FC1 探索**：OUT gate epilogue 融合 AdaLN(mlp) 并直出 FP4+SF
   （FP32 域算 LN，出口量化一次——不违反 §3.2 的精度顾虑，因为 LN 在 FP32 算）。
   这条若成，把第 2 条链纳入，再省 ~40–60ms。
3. QKV/OUT 保持 cublasLt 非链式（SDPA 阻断，无可链对象）。

### 完成判据
- `H3_DIT_NVFP4_CHAIN=1`：8 步 1344×768 5s DiT ≤ 84s，§6 验证门全过，
  nsys 显示 fp4_swiglu/swiglu kernel 消失、quantize 总时长下降 ≥25%。
- 默认配置（flag off）行为与性能与 P4.1 完全一致。
