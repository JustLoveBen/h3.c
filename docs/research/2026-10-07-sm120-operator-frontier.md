# RTX 5090（sm_120）算子/Kernel 侧剩余优化空间盘点

> **日期：** 2026-10-07。**机器基准：** 192.168.0.238，1×RTX 5090 32GB，sm_120，CUDA 13。
> **定位：** 机理级调研文档——盘点 H3.c 在 **算子/kernel 层**尚未关闭的全部优化空间，逐项给：机理 / 预期收益量级 / 施工面（文件级定位）/ 风险 / 与已闭环死路的区别。
> **输入账本：** 480P 热 DiT ~10.4s（INT8 GEMM 已达硬件峰值 98–99%，C5b 三层验证）；quant/dequant 中转 1.465s（nsys）；sol 窗口外 aux 12232 小核（但其中 DiT 稳态窗口内真实权重仅 ~0.09s/clip，见 §4）；768P INT8 权重窗+激活 28.8GiB 显存墙。
> **禁重提死路（roadmap §0.3/§5）：** QKV merge/QKNorm+RoPE 融合（本就有，MAE=0）· FFN epilogue **vec4**（对齐坑整管发散）· sol KV fuse（访存饱和零收益）· INT8 autotune（cublasLt 无备选核）· 手写 sm_120 INT8 GEMM（98–99% 峰值无 15% 余量）· CUDA Graph 整步（−0.16s 微薄，turbo4 仅 4 步）。本文所有方案均论证「为什么不是重复诉讼」。

---

## 1. sm_120 硬件能力钉死（不再猜）

### 1.1 Tensor Core 峰值表（dense / 2:4 sparse，FP32 累加口径）

| 精度 | RTX 5090（sm_120，GB202） | 备注 |
| --- | --- | --- |
| FP4（NVFP4 block-scaled） | **1676 / 3352 TFLOPS** | 第五代 Tensor Core 新能力；NVFP4 = E2M1 值 + E4M3 每 16 值块 scale + FP32 二级 scale |
| FP8（E4M3/E5M2） | **838 / 1676 TFLOPS**（FP16 累加）；**419 / 838**（FP32 累加） | 与 INT8 同速率（dense） |
| INT8 | **838 / 1676 TOPS** | 本仓实测 745.7 TOPS = dense 峰值的 89%，cublasLt 已达 98–99% 可达峰值 |
| FP16/BF16 | 209.5 / 419 TFLOPS（FP32 累加） | |
| TF32 | 104.8 / 209.5 TFLOPS | |
| INT4 | **无 Tensor 路径**（消费 Blackwell 砍掉 INT4/binary） | 低比特只有 FP4 一条路 |

**来源：** NVIDIA RTX Blackwell 架构白皮书 + zolotukhin.ai 推理参考表交叉一致；5090 官方「3352 AI TOPS」即 FP4 sparse 数。

### 1.2 消费级 sm_120 vs 数据中心 sm_100（B200）砍掉的功能

| 功能 | sm_100（B200） | sm_120（RTX 5090） | 影响 |
| --- | --- | --- | --- |
| MMA 指令 | tcgen05.mma（UMMA，TMEM 描述符） | `mma.sync.aligned.m16n8k32`（Ampere 式 warp 级，寄存器操作数） | **vLLM/TRT-LLM 的 NVFP4 快路径 gate 在 tcgen05 上，5090 直接 fall back 或拒载**——消费卡 NVFP4 是「field of one」（rfriedmann/imp 实测） |
| TMEM（Tensor Memory） | 有 | **没有** | SM100 内核设计（FA4、wgmma 流水线）在 sm_120 不可编译或崩溃 |
| TMA（cp.async.bulk） | 有 | **有**（继承自 sm_90） | GMEM→SMEM 异步批量拷贝可用 |
| TMA multicast | 有 | **没有**（cluster 只能 1×1×1） | 每个 CTA 自己加载，无跨 CTA 组播省带宽 |
| Thread block cluster | 有（含 cluster launch control） | **仅 1×1×1** | CUTLASS sm_120 collective 全部强制 1×1×1 |
| Block-scaled MMA（`kind::mxf8f6f4.block_scale`） | 有（TMEM 描述符送 scale） | **有**（scale 走寄存器，`scale_vec::1X` UE8M0） | **FP4 micro-scaling 在消费卡是硬件原生可用的**——这是本文最重要的钉死结论 |
| SMEM/SM | 228 KB | **99 KB** | M128/M256 大 tile 配置会崩；FlashInfer grouped GEMM 需要专门 patch |
| FP64 | 1/2 FP32 | 微量（仅为程序正确性） | 与本仓无关 |

### 1.3 软件栈支持矩阵（2026-10 现状）

| 栈 | sm_120 NVFP4 支持 | 备注 |
| --- | --- | --- |
| **CUTLASS 4.x** | ✅ 完整：`cutlass::arch::Sm120` + `OpClassBlockScaledTensorOp`，dense/sparse/grouped/NVFP4-in→NVFP4-out（带 scale 生成）全套；examples 79a/79b/79d/87 | 需 CUDA ≥12.8；**含 SM120 epilogue 与「full set of EVT fusions」**（4.x changelog 原文）——EVT 在消费 Blackwell 可用 |
| **cuBLASLt** | INT8 epilogue 仅 DEFAULT/BIAS/RELU/GELU(±AUX±BIAS)/BGRAD 系——**无 dequant、无 SwiGLU、无自定义 scale** | 本仓 INT8 链路无法用 cuBLASLt epilogue 消中转，钉死 |
| **TRT-LLM** | 支持矩阵含 sm_120（NVFP4/MXFP4/FP8 per-tensor），但无 rowwise/block-scaling FP8、无 FP8 KV cache | LLM 语境；vLLM 实际 gate tcgen05 |
| **TensorRT Model Optimizer** | NVFP4 PTQ/QAT/SVDQuant 全套，FLUX DiT 官方配方 | 质量证据见 §3 |
| **PDL** | sm_90+ 硬件支持，**Linux 消费卡可用**；Windows WDDM 有已知限制（本仓 Linux 238 不受影响）；CUDA Graph 内也可用（edge data） | §5 用 |
| **CUDA Graph** | 支持 | 本仓已交付整步捕获，收益 −0.16s 微薄 |

---

## 2. quant/dequant 中转 1.465s：消除方案（T1-e 核心靶子）

### 2.0 现状机理（先钉死这 1.465s 是什么）

本仓 INT8 链路（`h3_gpu_cuda.cu`）：cublasLt IMMA 写 **INT32 累加器 `d_acc` 到 HBM** → 独立 elementwise kernel 读回 `d_acc`、乘 `act_scale[row]×s_channel[col]`、写 FP32。往返流量 = `2 × m×n×4B`（读 INT32 + 写 FP32），外加 LoRA 加和时二次读写。nsys 实测：DiT quant/dequant（`dequant_i32_*` + `quantize_act`）**1.465s/clip，4000 个 kernel 实例**；另有 cast/prescale 2.29s 中约 0.9–1.2s 属同族可融余量（§1.9 C2 复核修正口径）。

R31 已交付的 `h3_cuda_dit_gemm_epi`（`h3_gpu_cuda.h:366-386`）只是**把独立 dequant/swiglu/gate kernel 换成更聪明的独立 kernel**（swiglu 版/gate 版），**INT32 acc 的 HBM 往返一次都没省**——dequant 仍发生在 GEMM 完成之后、acc 落盘之后。真正的消除 = dequant **在 GEMM epilogue 内、acc 还在寄存器/SMEM 时完成**。

### 2.1 方案 E1：CUTLASS 3.x EVT 融合 dequant+swiglu/gate（推荐主刀）

**机理：** CUTLASS EVT（Epilogue Visitor Tree）把 per-token scale × per-channel scale × acc + LoRA + SiLU(gate)×up 表达为 visitor 图，在 GEMM epilogue 内以寄存器级 acc 直接产出 FP32（甚至 FP16）输出——`d_acc` 永不落 HBM。vLLM 的 `ScaledEpilogue*` 族（w8a8 CUTLASS 路径）证明 per-token×per-channel INT8 dequant 是 EVT 标准做法；SwiGLU 可用自定义 visitor（gate/up 配对 + SiLU + mul + 写出 N/2），CUTLASS 4.x changelog 明确 **sm_120 有 full EVT 支持**。

**预期收益：** 消除 INT32 acc 往返 + 独立 dequant/swiglu kernel 的 launch+读带宽。按 nsys 1.465s 的可归属部分（dequant 系）+ swiglu/gate 独立 kernel 时间，**预估回收 −0.6 ~ −1.0s/clip**（打七折保守口径；上限为 1.465s 全回收但 quantize_act 部分不在此刀范围）。GEMM 计算本身不变（cublasLt 98–99% 峰值换成 CUTLASS sm_80 式 IMMA collective，DiT 这些大 K 形状下 CUTLASS INT8 与 cublasLt 通常持平 ±3%——**必须先做形状级 benchmark 兜底**）。

**施工面：** 新建 `h3_dit_cutlass_gemm.cu`（封装 CUTLASS 3.x `Sm80`/`Sm120` INT8 collective + 自定义 EVT：ScaledDequant→(+LoRA)→(SwiGLU|GateAdaLN)）；接线点 `h3_dit_w4a8.c` 的 `block_phase_out_mlp`（`h3_cuda_gemm_i8_dequant_f32_w_act_dev_epi` 四个调用点：out/fc1/fc2 + qkv 路径 `:1590/:1622`）；LoRA overlap 的 `lora_ready` event 语义保留（EVT 前置 streamWaitEvent）。开关 `H3_DIT_GEMM_CUTLASS_EVT`（默认 0，A/B 后定）。

**风险：** ① CUTLASS INT8 在这些 M/N/K 上若慢于 cublasLt >3%，收益被吃掉——**第 0 步离线 bench 全部 8 形状**；② EVT 引入浮点结合序变化（dequant 乘法顺序），bit-identical 可能破——INT32 acc 本身是精确的，dequant 只有两次乘法，**大概率仍 bit-identical**，但 swiglu 版 expf 顺序需逐位验证；③ 新依赖 CUTLASS 4.x 头文件（header-only，无 binary 依赖）。

**与死路区别：** FFN epilogue **vec4** 死的是「独立 kernel 里的 vec4 向量化撞非对齐指针」；E1 不写独立 kernel、不假设生产指针对齐（EVT 内部按 GEMM tile 天然对齐）。手写 INT8 GEMM 死的是「GEMM 本征更快」；E1 **不追求 GEMM 更快，只消链路往返**——正是死路清单里显式标注「未关闭方向：消除 GEMM↔dequant/swiglu 内存往返」的那条。

### 2.2 方案 E2：cublasLt 原地 + dequant kernel 合并减重（备胎，低成本）

**机理：** 不换 GEMM 库，只把现有 4000 个中转 kernel 做「三个合并」：① dequant+swiglu+LoRA 已合（R31），再把 **quantize_act（下一步 GEMM 的输入量化）与上一个 dequant/gate 合并**——gate 输出 FP32 后立即同 kernel 内重量化成下一步的 INT8+act_scale，省一次 FP32 往返；② ConvRot（cublasSgemmStridedBatched）与 quantize 融合；③ 用 `cudaMallocAsync` 池化 d_acc 消 alloc 抖动。

**预期收益：** **−0.2 ~ −0.4s/clip**（保守；只省部分往返，acc 落盘仍在）。

**施工面：** `h3_gpu_cuda.cu`（新 fused kernel：`dequant_gate_requant`）+ `h3_dit_w4a8.c` 调用点同 E1。

**风险：** 低（纯 kernel 工程，GEMM 不动）；重量化路径引入一次额外 round，bit-identical 需验证（数学上等价，FP 序可控）。

**与死路区别：** 死路关闭的是「GEMM+epilogue 的 vec4 实现」和「KV fuse」；E2 是「quant↔dequant 两个独立 kernel 的对撞融合」，从未施工过。

### 2.3 方案 E3：cuBLASLt epilogue 边界结论（钉死，不做）

cuBLASLt INT8 epilogue 只支持 BIAS/RELU/GELU(±AUX)/BGRAD——**无任意 scale dequant、无 SwiGLU**。`CUBLASLT_MATMUL_DESC_EPILOGUE` 不可表达 `act_scale[row]*s_channel[col]`。**结论：cuBLASLt epilogue 路线对 dequant 中转物理不可表达，关闭此问，不再查。**

---

## 3. NVFP4 路径：W4A4 的理论倍数与现实约束

### 3.1 硬件与栈（已钉死）

- sm_120 有**原生 block-scaled FP4 MMA**（`mma.sync.aligned.kind::mxf8f6f4.block_scale`，scale 走寄存器），dense FP4 = **2× FP8/INT8**。
- CUTLASS 4.x 有完整 sm_120 NVFP4 GEMM（examples 79a/79b：NVFP4→BF16、NVFP4→NVFP4 带 scale 生成——**后者意味着 GEMM 输出可直接再量化进下一个 GEMM，链式 FP4**）。
- 现成生产栈空缺：vLLM/TRT-LLM 快路径 gate tcgen05；**消费卡 NVFP4 推理目前是「field of one」**（imp，30B 300 tok/s 单 5090，手写 CUTLASS sm_120 核）——既是风险（无现成轮子）也是护城河（做出来了就是独家）。

### 3.2 理论倍数（DiT GEMM 侧）

| 项 | INT8 W4A8（现状） | NVFP4 W4A4 | 倍数 |
| --- | --- | --- | --- |
| GEMM 峰值 | 838 TOPS（实测 745） | 1676 TFLOPS | **GEMM 本征 ~2×** |
| 权重显存 | INT8 1B/param（+scale） | FP4 0.5B/param（+1/16 E4M3 scale ≈ 4.5bit） | **权重 −50%**（768P 权重窗直接减半，见 §6） |
| 激活显存 | FP32 中转 | 可 FP4/FP8 链式 | −50~75% |

480P DiT 10.4s 中 INT8 GEMM 本征 3.59s（nsys）→ 理想 1.8s，**但 Amdahl**：ConvRot/quant/dequant/SDPA/sol 不动，**全链路乐观预估 DiT −1.5 ~ −2.5s/clip**；且 FP4 的 quant 中转（act 量化到 NVFP4+scale 生成）比 INT8 更重，若不用 CUTLASS 79b 式「NVFP4 直出」链式，中转可能吃掉一半收益。

### 3.3 质量证据（激活 FP4 的风险）

- NVIDIA 官方 FLUX FP4：**PTQ 裸量化掉质量**（小字/图形劣化，ImageReward 1.118→1.096）；**QAT 或 SVDQuant 恢复到 BF16 持平**（1.119/1.108）。所有 FC 层 FP4、MHA 保持 FP8、首尾层排除（selective quantization）。
- ModelOpt 官方矩阵：W4A8 AWQ 标注「experimental，accuracy penalty 更高」；NVFP4(+SVDQuant) 是一等公民。
- TorchAO/diffusers（B200）：NVFP4 e2e 1.59×，**必须 selective quantization**；QwenImage 比 FLUX 更敏感（LPIPS 0.34 vs 0.11）。
- **对 turbo4 4-step 蒸馏的推断：** 蒸馏轨迹对扰动更敏感（FastWan 论点本仓已实证于稀疏），激活 FP4 大概率需要 SVDQuant/QAT 级修复——**纯 PTQ 直接上 W4A4 的质量风险判「高」**；W4A8-FP4（权 FP4、激活 INT8/FP8 混合）不在硬件甜点（block-scaled MMA 要求两边同族），混合路径需实测。

### 3.4 本仓 `h3_dit_nvfp4` 探索线现状（代码确认）

`h3_dit_nvfp4.h/c`：解析社区 pruned_nvfp4 DiT（U8 nibble 权重 + F8_E4M3 scale + F32 scale_2，vonkaiser/lilcheaty Comfy 格式），probe 失败硬退出（不回退 INT8 金线）；`dequant_to_i8` = **NVFP4→FP32→per-row INT8 转码** + INT8 sidecar 缓存（`h3_nvfp4_int8_sidecar.c`）。**即：当前 NVFP4 线只是「把 FP4 权重转成 INT8 跑现有 cublasLt 链路」——权重来源多样化，零硬件 FP4 收益。** 真正的原生 NVFP4 GEMM（CUTLASS sm_120 block-scaled）**未施工**。

**施工面（若立项）：** 新 `h3_dit_nvfp4_gemm.cu`（CUTLASS 4.x Sm120 block-scaled collective，examples 79a/79b 起步）；量化侧需 ModelOpt/SVDQuant 离线产权重（弃社区 pruned_nvfp4）；验证门=waffle 金线区间（非 bit-identical，T1/T3 档先行）。**预估工程量大（2–3 周级），建议排在 E1 之后、以 E1 的 EVT 经验为垫脚石。**

---

## 4. sol aux 12232 小核：launch 层的真实余量

### 4.0 先把口径钉死（防止第三次误判）

§1.6.1 已用 stream 分桶实证：12232 实例中 **DiT 稳态窗口内只有 ~0.09s/clip**（kc/vc/diag/stats 各 199×）；其余 = ① warmup/autotune 一次性 ~1.05s（每 clip 都有？——**关键未决**：triton autotune 按 T sweep 的开销是 per-process 还是 per-clip，若是 cold-only 则热 e2e 账本里根本不存在）+ ② VAE 段 FillFunctor 0.70s（与 sol 无关）。**「合并 50% 小核」若指 DiT 稳态部分，理论回收 ≤0.05s——不值得做任何 megakernel。** 唯一有肉的 launch 层靶子 = warmup/autotune 的 1.05s 是否每次 clip 都付。

### 4.1 方案 L1：sol warmup/autotune 结果持久化

**机理：** triton autotune 每进程重 sweep（8 配置×2 核×每 T），产生 ~6600 个一次性小核。把 autotune 结果按 (T, 形状, GPU) 落盘缓存，二次 clip 直接命中——**cold 启动税 −1.0s 量级**，hot clip 无收益（本来就没有）。

**施工面：** 238 sol-engine `preprocess.py` 侧（仓外依赖）；或本仓 `h3_generate_preset.c` 预热阶段复用。**风险：** 低。**与死路区别：** C1 关的是「DiT 窗口内 aux 融合」；此项是窗口外一次性开销的缓存，从未施工。

### 4.2 方案 L2：PDL（Programmatic Dependent Launch）挂 GEMM→dequant→quant 链

**机理：** sm_90+ 支持，Linux 消费卡可用（WDDM 限制不影响 238）。secondary kernel 用 `cudaLaunchAttributeProgrammaticStreamSerialization` 提前占坑，`cudaGridDependencySynchronize` 后读上游结果——**消 kernel 间隐式 barrier 的气泡**，对本仓 GEMM→dequant→swiglu→quant 这种 4000 实例的短链每个省 2–5μs。

**预期收益：** 若 DiT 稳态 kernel 间气泡按 3μs×~4000 估 ≈ **−0.01~−0.05s——量级太小**，且 P2b CUDA Graph 已证「turbo4 4 步 launch 税被 cold 噪声掩盖」（整步 graph 才 −0.16s）。**判不立项，登记为「已评估关闭」**：PDL 的收益模型与 CUDA Graph 同族（都是消 launch/气泡），graph 整步都只 −0.16s，PDL 部分链必然更小。

### 4.3 方案 L3：megakernel 化（Mirage MPK 式）——判不适用

MPK 论文目标场景是 LLM decode（数千 kernel/iteration、kernel 1–10μs）；本仓 DiT 稳态 kernel 数量少（4 步×~50 层×~10 核 ≈ 2000/clip 但单个多为 ms 级 GEMM），launch-bound 证据不足（P2b 实测已否定）。且 MPK 公开结果在 A100/H100/B200，sm_120 无现成运行时。**判不适用，登记关闭。**

---

## 5. 内存侧：768P 显存墙与 L2/流式余量

### 5.1 方案 M1：NVFP4 权重窗（与 §3 同一刀，显存维度收益）

768P OOM 账本 = native INT8 权重窗 + 激活 28.8GiB。权重 NVFP4 化 → **权重部分 −50%**（粗估省 6–9GiB 量级，取决于窗内权重/激活比例），可能直接把「768P 高分全 sol」从 OOM 墙下放出来（T4-e [审计依赖] 的显存前提改写）。**这是 NVFP4 对 T4 的战略价值，大于 480P 速度价值。**

### 5.2 方案 M2：激活 checkpointing / 原位计算

DiT 推理无反向，「checkpointing」退化为**逐层不复用缓冲**——本仓 d_a/d_b/d_qkv/d_wide/d_attn 已是固定槽位复用（`block_phase_out_mlp` 签名可见），**激活侧已接近原位**。剩余空间：FFN 的 `[seq, 2*FFN]` 宽中转（d_fc1 用 d_wide 槽）若改 FP16/BF16 存储，**激活 −50%**；精度风险：中转 FP16 化影响 dequant 输出表示，**非 bit-identical**，需 waffle 验证；收益只对 768P 显存墙有意义（480P 无墙）。**预估可腾 2–4GiB，风险中，建议作为 M1 失败时的备胎。**

### 5.3 方案 M3：weight streaming 与 GEMM overlap / L2 residency

- 本仓 INT8 权重窗已有 side-stream H2D + `ensure_ready` event overlap（`h3_cuda_w4a8_weight_create_from_i8` 注释明确「no default-stream sync」）——**weight streaming overlap 已做**。
- L2 residency（`cudaStreamSetAttribute` accessPolicyWindow）：5090 L2 = 96MB，DiT 逐层权重（~100–200MB/层 INT8）超出 L2，但 **ConvRot 旋转矩阵、RoPE 表、modulation 表、s_channel** 这类每层复读的小热数据可 pin L2——**预估 −0.05 ~ −0.15s/clip，低成本（纯 host 侧属性设置）**，bit-identical。施工面：`h3_gpu_cuda.cu` op stream 初始化处。
- **cast/prescale 2.29s 中的 SIMT sgemm 1.15s**（VAE 段，C3 已查）：VAE f16 GEMM/conv 的 TF32 档衍生刀口已立项未施工——属 VAE 族非本文算子层范围，登记指针。

---

## 6. TOP 排序（预期收益 × 置信度）

| # | 方案 | 预期收益 | 置信度 | 一句话施工定位 |
| --- | --- | --- | --- | --- |
| 1 | **E1 CUTLASS EVT 融合 dequant+swiglu/gate** | −0.6~−1.0s/clip（480P DiT） | 中高（EVT 是 vLLM 成熟做法；风险在 CUTLASS vs cublasLt 形状级快慢） | 新 `h3_dit_cutlass_gemm.cu` + `h3_dit_w4a8.c` 四个 `*_dev_epi` 调用点换后端，先离线 bench 8 形状 |
| 2 | **NVFP4 原生 GEMM（W4A4+SVDQuant）** | 480P DiT −1.5~−2.5s；**768P 权重窗 −50% 可能解 OOM** | 中（硬件/CUTLASS 钉死可用；质量需 SVDQuant/QAT，工程 2–3 周） | 新 `h3_dit_nvfp4_gemm.cu`（CUTLASS 79a/79b 起步）+ ModelOpt 离线产权；现 `h3_dit_nvfp4.c` 仅 INT8 转码线 |
| 3 | **E2 dequant+requant 对撞融合（备胎）** | −0.2~−0.4s/clip | 高（纯 kernel 工程） | `h3_gpu_cuda.cu` 新 fused kernel，不动 GEMM 库 |
| 4 | **L1 sol autotune 持久化** | cold −1.0s（hot 无收益） | 中（需确认 1.05s 是否 per-clip） | sol-engine preprocess 侧 autotune cache 落盘 |
| 5 | **M2 激活 FP16 中转（768P 备胎）** | 768P 腾 2–4GiB | 中（非 bit-identical 需 waffle 门） | `h3_dit_w4a8.c` d_wide/d_qkv 槽 dtype 参数化 |
| 6 | **M3 L2 residency pin 小热表** | −0.05~−0.15s/clip | 中 | `h3_gpu_cuda.cu` op stream accessPolicyWindow |
| 关闭 | L2 PDL / L3 megakernel / E3 cuBLASLt epilogue | <0.05s 或物理不可表达 | 高 | 登记为已评估关闭，不再诉讼 |

---

## 7. 结论

1. **算子层最大未关闭单项 = quant/dequant 中转（E1）**，且它是 T1/T2 目标重估的 [审计依赖] 关键变量——推荐作为下一拍主刀，第 0 步只做 8 形状 CUTLASS-vs-cublasLt bench（半天）即可证伪/证实。
2. **sm_120 的 FP4 是硬件真能力、软件真空区**——NVFP4 是中期最大杠杆（速度 2× GEMM 本征 + 768P 显存墙唯一算法级出口），但质量必须走 SVDQuant/QAT 配方，且当前仓内 NVFP4 线只是 INT8 转码，原生路径从零施工。
3. **launch 层在 turbo4 4-step 下已被 P2b 实证封顶**——PDL/megakernel 登记关闭；唯一残余是 sol warmup autotune 的 cold 税。
4. **显存墙的算子级出口排序：** NVFP4 权重（M1，治本）> 激活 FP16 中转（M2，备胎）> 现状已尽的 streaming/overlap。

## 参考源

- NVIDIA RTX Blackwell GPU 架构白皮书（GB202 峰值表）；zolotukhin.ai Ampere/Ada/Blackwell 推理参考（sm_120 无 INT4、FP4=2×FP8 交叉验证）
- CUTLASS 4.2.1 changelog + examples 79a/79b/79d/87（sm_120 block-scaled GEMM + full EVT）
- lna-lab/blackwell-geforce-nvfp4-gemm（sm_120 架构事实：99KB SMEM、1×1×1 cluster、无 TMEM/tcgen05、寄存器 block-scale）
- rfriedmann.de「Serving 30B at 300 tok/s on single RTX 5090」（消费卡 NVFP4 field-of-one 证据）
- NVIDIA Technical Blog：NVFP4 介绍 / TensorRT FP4 FLUX（PTQ 掉质、QAT/SVDQuant 恢复 BF16 持平）；PyTorch blog TorchAO MXFP8/NVFP4（selective quantization 必要性）
- CUDA Programming Guide §4.5 PDL；NVIDIA/cccl#6925（WDDM PDL 限制，Linux 不受影响）
- vLLM `csrc/quantization/w8a8/cutlass/Epilogues.md`（EVT dequant epilogue 成熟先例）
- 本仓：`h3_dit_w4a8.c` / `h3_gpu_cuda.cu:289-386` / `h3_gpu_cuda.h:366-386` / `h3_dit_nvfp4.c` / roadmap §0.3/§5 / 480p-10s-plugin-plan §1.5–1.9
