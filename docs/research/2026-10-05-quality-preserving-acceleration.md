# 质量保持型加速技术全景调研（H3 视频生成器）

> 来源：子代理 `ce4a7cc5-a122-4159-aaba-87cc0de628bc` 交付（只读，未改代码）。核心命题：减步/蒸馏 LoRA 必掉质量，加速且保质是差异化核心。战场 = 注意力机制 + 算子层。

## 一、分层技术地图：lossless-ness 评级 + 实测加速 + 对本仓适用性

### 1. 注意力层（核心战场）

| 技术 | lossless-ness | 实测加速 | 对 H3 适用性 |
| --- | --- | --- | --- |
| **sol_cute 稀疏注意力**（本仓现有） | **近似稀疏**（非 bit-exact vs dense；但 400/400 fallback=0，金线内 waffle 1.04–1.71，属「近无损感知」档） | 480P DiT 10.37s 内已含 | ✅ 已落地，主力路径 |
| **LoSA（Near-Lossless Sparse Attention）** | **近无损构造性保证**：固定 99% 注意力质量保留阈值，非固定稀疏率 | Wan2.1-1.3B 单用 1.36×，VBench −0.06；叠加特征缓存 HunyuanVideo 3.2× | 🔶 高价值调研方向：可移植「质量阈值冻结」思想到 sol 调窗；需重校准索引，非即插即用 |
| **NSA（Native Sparse Attention）** | 训练内生稀疏，非推理无损 | 3.6% 注意力预算达 128K token | ❌ 需端到端训练，不适用推理加速 |
| **SageAttention3 / FP4 注意力** | **非全模型无损**：官方明确不保证所有模型无损，推荐敏感层/时间步用 Sage2++ 混合 | RTX 5090 上 1038 TOPS，5× vs 最快 FlashAttention | 🔶 opt-in 精度档：T1 极速档可试，T2 金线档禁用 |
| **FlashAttention-4** | **算法无损**（精确注意力，仅 kernel 优化） | B200 1.3× vs cuDNN 9.13；**sm_120 缺 tcgen05/TMEM 硬件单元，FA4 在本代退化为 FA2 水平，5090 上 D128 仅 1.03–1.07×** | 🔶 **硬件代差，非技术无效**：FA4 算法本身无损且正确，收益受限只因 5090 缺 Blackwell 数据中心级 tensor core 代际特性（B200 1.3× 为证）。「cuDNN 9.13+ 已吸收 FA4 技术」一说**已证伪**（FA4 论文基线就是 cuDNN 9.13 且在其上还有 1.3×，若已吸收此数不可能存在）；且 sm_120 上 FA4 不可表达非软件问题——可移植的是软件仿真指数+条件 softmax rescaling 两个不依赖 TMEM 的 trick。注意：1.03–1.07× 的无损 kernel 优化仍值得做——量级与已列 P1 项 QKNorm+RoPE 融合的 ~0.42s 相当 |
| **KV-cache 压缩/量化** | 有界误差，非无损 | LLM 场景 2–4× 显存压缩 | ❌ 本仓无 KV-cache（每步全新 QKV），不适用 |

### 2. 算子/Kernel 层（纯工程无损主战场）

| 技术 | lossless-ness | 实测加速 | 对 H3 适用性 |
| --- | --- | --- | --- |
| **QKV 三次瘦 GEMM → 一次宽 GEMM** | INT8 合并若改累加顺序则非 bit-identical；FP 路径可无损 | 外部 RTX5090 pillar1 已证；本仓预估 −0.6~−1.2s | ✅ **P1**：50 层 K/V 各自独立 QKV-proj 可合并 |
| **QKNorm+RoPE 融合核** | **bit-identical 可达**（纯 kernel 融合无数值变化） | rope 0.25s+adaln 0.17s，预估可回收 ~0.42s | ✅ **缺失项**：已有 RMS-AdaLN，缺 QKNorm+RoPE |
| **GEMM+epilogue 融合（quant/dequant 消隐）** | **bit-identical 可达**（dequant 数学等价） | quant/dequant 中转 1.465s+cast 2.29s | ✅ **高价值**：INT8 量化链隐藏开销 |
| **CUDA Graph 整步捕获** | **完全无损**（消除 CPU launch 开销，不改数值） | TRT-LLM LTX-2 每步 −4.1ms（−2%）；FastUSP 端到端 9–16%；sglang Qwen-Image −8%；**kernel 越短、launch 越多，收益越大** | ✅ **本仓缺失**：4-step×50 层×多 kernel = launch-bound 风险高；sol pre/post aux 12232 实例大量小核 |
| **VAE unpatch_d2h pinned+async+stream 流水** | **bit-identical 可达**（纯传输优化） | 480P 串行尾 1.6s，预估 −1.3~−1.6s；768P −3.5~−5.5s | ✅ **P0 最高优先**（作用域 = VAE 4.96s 中的族#6 零成本浪费子集 ~2.3s，见表下注释） |
| **VAE blocks 2-stream 并行** | **bit-identical**（计算图不变） | 768P 预估 −1.5~−2.5s | ✅ T4 专属 |
| **VAE blocks 主体量化收割（fp16→int8 fused decode）** | **非零风险**：fp16 未量化真实计算（族#5 未收割），有 VAE 量化误差质量杠杆 | 480P vae_blocks 主体 ~2.6s | 🔶 **升级定性：族#5 在 VAE 上的二次收割**（非纯工程项）——需误差结构分析 + waffle A/B（同 DiT INT8 流程）；T3-c cold-clip 间歇高噪疑似其既有症状 |
| **冗余 zeros→empty / 内存复用** | **bit-identical**（须验证全写覆盖） | FillFunctor 0.70s（VAE 段 F32 canvas 零初始化） | ✅ 已部分落地（与 d2h 串行尾同属 VAE 族#6 浪费子集） |

> **VAE 4.96s 归因修正（依据 `2026-10-05-acceleration-mechanisms.md` 机理族框架交叉核对）：** VAE decode 4.96s **不是**整块「确定性、无质量代价」目标。其中仅 ~2.3s（d2h 串行尾 ~1.6s + FillFunctor ~0.7s）属**族#6 零成本工程浪费**（bit-identical 可达，P0 靶子）；vae_blocks 主体 ~2.6s 是 **fp16 未量化的真实计算**（族#5 未收割部分，带 VAE 量化误差质量杠杆）。VAE int8 融合 decode 属族#5×#6 纠缠，须按量化项走 A/B，不得按零风险工程项处理。

### 3. 算法层（非减步路径）

| 技术 | lossless-ness | 实测加速 | 对 H3 适用性 |
| --- | --- | --- | --- |
| **UniPC / DPM-solver++** | **非蒸馏减步**：高阶 ODE 求解器，训练-free | UniPC 5–10 NFE 优于 DPM-solver++ | 🔶 理论可试但受限：4-step 下 UniPC 难敌蒸馏学生；非差异化方向 |
| **CFG-free / 引导嵌入** | 若已 CFG-free 则无损 | — | ✅ 已 turbo4 蒸馏内化 CFG |
| **跨步精确缓存（非 TeaCache 近似）** | **bit-identical**：缓存不变量（text emb、AdaLN 若 step-independent、RoPE 表） | 依缓存命中率 | 🔶 有限空间：4-step 冗余已被蒸馏榨干 |

## 二、本仓缺失的 lossless 技术清单（Gap List）

| 优先级 | 缺失项 | 预估收益 | 状态 |
| --- | --- | --- | --- |
| **P0** | VAE unpatch_d2h pinned+async+stream 流水 | 480P −1.3~−1.6s / 768P −3.5~−5.5s | roadmap，待施工 |
| **P1** | QKV merge（3 瘦 GEMM → 1 宽 GEMM） | −0.6~−1.2s | 外部已证 |
| **P1** | QKNorm+RoPE 融合核 | ~0.42s | roadmap 附属 |
| **P2** | CUDA Graph 整步捕获（DiT 4-step 全图） | −2%~−16%（预估 −0.5~−1.5s） | **新增缺口** |
| **P2** | INT8 quant/dequant epilogue 融合 | 1.465s 量化中转可回收部分 | 未精确量化 |
| **P3** | LoSA 质量阈值稀疏思想移植 | 1.2–1.36× 注意力加速 | 调研项，非即插即用 |
| **P3** | FP4 注意力（sage3_blackwell）opt-in | 注意力段 3–5× | T1 实验项 |

## 三、战略综合：能否重构为「fast high-NFE」？

### 3.1 核心数学：4-step turbo4 vs 8-step golden

| 路径 | NFE | DiT 时间 | 质量（waffle） | 每步成本 |
| --- | --- | --- | --- | --- |
| 当前 turbo4 | 4 | 10.37s | 1.04–1.71 | 2.59s/step |
| 8-step golden（同等优化） | 8 | ~20.7s（线性外推） | 0.79–1.34（DMAD 区间） | 2.59s/step |
| 全 lossless 优化叠加后 8-step | 8 | ~16–17s | 0.79–1.34 | ~2.0–2.1s/step |
| 全 lossless 优化叠加后 4-step | 4 | ~6–8s | 1.04–1.71 | ~1.5–2.0s/step |

**关键洞察：** 即使 8-step 做到每步比 4-step 更快（2.0s vs 2.59s），总时间仍 ~16s vs ~7s——**NFE 结构差距（2×）无法被 per-step lossless 优化逆转**。

### 3.2 「fast high-NFE」成立条件

- **768P/长序列**：序列 15973→~64K 时，注意力从线性主导转平方主导，稀疏收益从 1.3× 升至 3×+，sol 近似误差在长序列下相对缩小。此时「8-step+极致稀疏」可能打平「4-step+稠密」。
- **质量敏感交付**：DMAD 4-step（waffle 0.79–1.34）优于 turbo4；若叠加 lossless 优化使 4-step DMAD 从 +11% 降到 −10%，则「高质量 4-step」成独立 SKU。

### 3.3 战略结论

1. **不放弃 4-step 蒸馏主线**：turbo4 已把 NFE 压到结构极限，lossless 优化应叠加在 4-step 上而非替代。
2. **差异化不在「多步快」，而在「同质量下无可匹敌的快」**：护城河 = sol_cute 稀疏 + 官方 VAE + 32GB 单卡垂直整合 + INT8 链路工程深度。**注意区分作用域**——INT8 **GEMM 本征**已 98–99% 硬件峰值（无 raw-FLOPS 余量，C5b 已证）；但 INT8 **链路**的 quant/dequant/cast 中转（§2 合计 ~3.75s：1.465s 量化中转 + 2.29s cast）仍是 **bit-identical 可达的高价值工程靶子**（P2 epilogue 融合）。「98–99% 峰值」仅指 GEMM kernel 层，**不代表 INT8 链路已无可压榨**——两者不可混用。
3. **新增最高价值缺口：CUDA Graph**（launch 开销被低估，预估 0.5–1.5s）。
4. **768P 是稀疏注意力真正战场**：32GB OOM 墙是唯一阻塞。
5. **FP4 注意力是 T1 极速档 opt-in 实验**（T2 金线禁用）。

### 3.4 一句话战略

> **质量保持型加速的终极形态 = 4-step 蒸馏（NFE 结构极限）× 全栈 lossless 工程（CUDA Graph + QKV-merge + RoPE 融合 + VAE 流水 + epilogue 消隐）× 场景化稀疏（480P 固定窗/768P 自适应阈值）× opt-in 精度档（FP4 注意力/TF32）。** 「fast high-NFE」在 480P 不成立（数学不闭合），在 768P 长序列待显存墙突破后可能成立；当前差异化应聚焦「同 NFE 下最快、同速度下质量金线逐位一致」。
