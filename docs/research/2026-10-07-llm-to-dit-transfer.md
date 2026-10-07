# LLM 推理栈 → 视频 DiT 迁移可行性机理分析

> **日期：** 2026-10-07。**对象：** 本仓 MiniMax H3 33B MMDiT（turbo4 4-step 蒸馏 LoRA，INT8 W4A8，单卡 RTX 5090 sm_120）。
> **方法：** 每条按「机理 → LLM 语境收益 → DiT 是否同构 → 本仓可迁移性裁决」四段式。裁决标签：**【可迁移】**（有明确落地路径）/ **【部分迁移】**（需改造或有条件）/ **【N-A】**（DiT 无同构问题）/ **【已闭环】**（本仓已证死路或已交付）。
> **前置纪律：** 不重复诉讼 roadmap §0.3/§5 已闭环死路；DiT INT8 GEMM 已达 sm_120 硬件峰值 98-99%（745.7 TOPS），任何「更快的同精度 GEMM」数学上无余量。
> **外部证据检索日期：** 2026-10-07，覆盖 2025–2026 一手源（NVIDIA 官方/CUTLASS/FA4 论文/Mirage OSDI'26/NeurIPS'25 等）。

---

## 1. 注意力架构级：MLA / GQA / MQA

### 1.1 机理（LLM 语境）

- **GQA/MQA**：多 query 头共享少量 KV 头，消的是 **decode 期 KV cache 的 HBM 读取量**（带宽 bound），不减 FLOPs。
- **MLA（DeepSeek V2/V3）**：K/V 压进共享低秩 latent（每 token 4.5·d_h），inference 时把 up-projection **吸收进 query 投影**（Absorb），cache 只存 latent。消的是 decode 期 KV cache 体积与读取带宽，附带 TP 不友好副作用。

### 1.2 LLM 语境收益

MLA 的收益语境是 **自回归 decode**：每步只算 1 个新 token 的注意力，但要从 HBM 读全部历史 KV → 纯带宽 bound，cache 体积直接等于每 token 延迟。TransMLA（NeurIPS'25）实测 LLaMA2-7B 8K 上下文 10×。

### 1.3 DiT 是否同构

**不同构，结构上 N/A。** 视频 DiT 是**双向全注意力、无 KV cache**：每一步去噪对整个 latent（480P ~16k tokens）做一次完整前向，Q/K/V 当步现算现用，不存在「跨步增量读取的历史状态」。MLA/GQA 消的那个量（decode KV 带宽）在 DiT 里根本不存在——这是范畴差异，不是程度差异。

有一个**伪装成同构**的相邻问题需要拆穿：DiT 的注意力 FLOPs 是 O(N²d)，KV 物化也是 O(N·d)，但它们是**当步计算图内部的张量**，不是跨步持久状态；压缩它等于改注意力本身的数学（那就进入稀疏/线性注意力族，见本仓机理文档族 2/2b），而不是「KV cache 优化」。

### 1.4 本仓裁决：**【N-A】**

- MMDiT 是否有可借鉴的低秩/头共享结构？——作为**架构改动**（重训练/微调）理论存在，但它消的是 DiT 注意力 FLOPs/参数，与 MLA 消 decode 带宽是两个不同目标；且合同红线内「不动质量尺、不改默认路径」，架构级改动=换权重=族 8 换合同范畴。**不进入优化候选表。**
- 唯一诚实的迁移价值：MLA 的 **Absorb 技巧**（把 up-projection 折进相邻 GEMM 省一次中间物化）是通用代数手法——但本仓 QKV 已是宽 GEMM 合并（P1 探针 MAE=0 闭环），无可吸收对象。
- 参考：扩散语言模型侧的 LLaDA-Hybrid（arXiv 2608.06628）走的是 block-hybrid 线性注意力路线（属族 2b），不是 MLA。

---

## 2. 量化前沿

### 2.1 NVFP4 / FP4 原生 micro-scaling

**机理**：Blackwell 五代 tensor core 原生 block-scaled FP4 MMA：16 值 micro-block 共享一个 FP8(E4M3) scale + 每 tensor 一个 FP32 二级 scale，4.5 bit/值。消的是**位宽冗余**，且是原生 MMA（非 dequant-then-compute），吞吐相对 FP8 ~2×、相对 FP16 更高。

**LLM 语境**：TensorRT-LLM/vLLM 已支持 NVFP4 部署（DeepSeek-R1、Llama3、**FLUX.1-dev** 已有官方 NVFP4 预量化 checkpoint）。NVIDIA 官方定位 NVFP4 为「准确且紧凑」的推理格式。

**sm_120 可用性（2026-10 查证）**：**关键硬件事实**——sm_120（GeForce Blackwell）**不是** sm_100（B200）子集：
- sm_120 用 SM80 系 `mma.sync.aligned.kind::mxf8f6f4.block_scale`（warp 级、寄存器到寄存器、m16n8k64），**无 tcgen05/TMEM/UMMA**；
- sm_100 的 TMEM 直供路径在 sm_120 上必须走显式 `ldmatrix`（SMEM→寄存器）；
- SMEM 上限 99KB（sm_100 为 228KB），cluster 仅 1×1×1（无 TMA multicast）；
- CUTLASS 已有官方 sm_120 NVFP4 GEMM 示例（79b，warp-specialized persistent + pingpong schedule），CUDA ≥12.8（本仓 CUDA 13 ✓）；CuTe DSL sm_120 MXF4/NVFP4 native-TMA 路径已合入（cutlass#3273）。

**理论倍数 vs 本仓 INT8**：FP4 MMA 相对 FP8 ~2×；INT8 与 FP8 在 Blackwell 同档峰值 → **FP4 理论峰值 ≈ 2× INT8**。本仓 INT8 已实测 745.7 TOPS（98-99% 峰值），FP4 理论天花板 ~1.5 POPS。注意 sm_120 的 FP4 走寄存器 MMA + ldmatrix，实测兑现率需打问号（QuACK 报能达 CUTLASS 示例的 95%，但那是 GEMM 微基准）。

**DiT 同构性**：DiT GEMM 与 LLM GEMM 数学同构；**激活 outlier 问题 DiT 比 LLM 更严重且多一维**：LRQ-DiT/Q-DiT/DiRotQ 均确认 DiT 激活有 salient channel outlier，且**随 denoising timestep 漂移**（LLM 无此维度）。FLUX.1-dev 官方 NVFP4 checkpoint 的存在 = NVFP4-on-DiT 已被 NVIDIA 官方背书可行。

**本仓裁决：【部分迁移】——方向开放，但有两个前置门。**
- 本仓已有 `h3_dit_nvfp4.c/h` 探索线，方向本身已在仓内立项，本条提供的是外部可行性证据升级：CUTLASS sm_120 官方示例 + CuTe DSL native 路径已可用，不再是「手写核赌博」。
- **前置门 1（质量）**：FP4 激活量化在 4-step 蒸馏轨迹上的质量损失谱未验证；DiT timestep 漂移意味着 per-step scale 校准必需。对照实验可在 SKU 隔离臂做（合同允许换权臂）。
- **前置门 2（兑现率）**：sm_120 寄存器 MMA 路径下，NVFP4 GEMM 实测 TOPS 需先 probe（对照 cublasLt INT8 745.7）；若实测兑现率 <1.4× INT8，扣除 quant/dequant 中转后净收益可能不覆盖迁移成本。
- **量级预估（诚实区间）**：GEMM 理论 2×，但 DiT 全链路含注意力/归一化/elementwise（Amdahl），端到端 DiT 段净收益预估 1.2–1.5×（**预估**，依据 FLUX NVFP4 官方存在性 + sm_120 寄存器路径损耗折扣）。若兑现，是已识别未关闭项里量级最大的单杆。
- **不做的事**：不在默认路径直接替换 INT8（质量尺红线）；不与「INT8 GEMM 已 98-99% 峰值」矛盾——那是 INT8 的峰值，FP4 是**另一个更高的峰值档**。

### 2.2 QuaRot / Hadamard 旋转抑制激活 outlier

**机理**：对激活右乘正交 Hadamard 矩阵 H、对权重左乘 Hᵀ，数学恒等（XW = (XH)(HᵀW)），把通道 outlier 的能量摊平到全通道，动态范围骤降 → 低 bit 激活量化从「糊图」变「可用」。代价：一次 O(K²) 全局旋转（可降为 block-wise O(K·N₀)）。

**LLM 语境**：QuaRot（2024）端到端 W4A4；SpinQuant 学旋转。

**DiT 同构**：**已有明确的 LLM→DiT 直接迁移先例**（2025-2026 密集出现）：
- **ConvRot**（arXiv 2512.03673）：group-wise regular Hadamard，W4A4 DiT，明确指出全局 Hadamard O(K²) 对大 DiT 太贵、改分组；
- **LRQ-DiT**（2508.03485）：区分 mild/salient 两类 outlier，自适应 Hadamard/outlier-aware 旋转；
- **CLQ**（2509.24416）：block Hadamard smoothing，W4A4 报 3.95× speedup/3.98× 显存；
- **DiRotQ**（2605.16732）：per-layer PCA + random residual rotation，明确指出 LLM 旋转法不能直接搬，因 DiT 激活统计随 timestep 漂移。

**本仓裁决：【已闭环为部分实现 + 部分迁移（激活侧待收割）】。**
- **本仓已实现**：`h3_w4a8_dequant.c` 已有 ConvRot 式 regular Hadamard（Kronecker H4）+ Sylvester Hadamard 构建，权重侧旋转 undo 已在线——即本仓 W4A8 的「ConvRot」名号来源，**权重侧已收割**。
- **未收割部分**：激活侧是 per-tensor/per-channel 运行时量化（`h3_dit_w4a8.c:1621`），未做激活侧在线旋转。但注意 INT8 激活本身动态范围宽裕，outlier 问题在 W4A8 档**不致命**——旋转的真正价值在 **W4A4/FP4 激活档**（配合 §2.1 NVFP4 才把激活压到 4 bit）。
- **结论**：不是独立收益项，是 **NVFP4/W4A4 迁移的前置技术件**。若走 NVFP4 臂，激活旋转（block-wise，参考 ConvRot 分组控制 O(K·N₀) 成本）是标配；当前 INT8 档下单独上加法旋转无收益（已够准）。
- 附注：SmoothQuant（激活↔权重难度迁移）同理——INT8 档不需要，FP4 档是候选工具之一。

### 2.3 AWQ / GPTQ / W4A16 weight-only in-register dequant

**机理**：只压权重到 4 bit，激活保持 FP16，kernel 内 register 级 dequant（Marlin：INT4→FP16 位运算 unpack + tensor core 喂料流水）。消的是**权重 HBM 读取带宽**（~4×），不碰计算精度路径。

**LLM 语境收益边界（关键）**：Marlin 官方明确——接近理想 4× 加速**仅在 batch ≤16-32 tokens 的 decode 带宽 bound 区间**；batch 一大（prefill/计算 bound）收益趋零。

**DiT 同构性**：**不同构于本仓负载**。DiT 每步是 batch=全 latent（16k-64k tokens）的大 GEMM，**纯计算 bound**——这正是本仓机理文档族 5 已注明的「反例：纯 W4A16 反量化在计算 bound DiT 大 GEMM 上端到端收益常 <10%（数字多来自 LLM decode 带宽 bound 语境）」。

**本仓裁决：【已闭环】。**
- 本仓已是 W4A8（权重 W4 + 激活 INT8 原生 MMA），比 W4A16 更激进且已兑现峰值。Marlin 式 in-register dequant 对本仓是**倒退**（放弃 INT8 原生 MMA 换回 FP16 计算）。
- 预防性登记：任何「W4A16/Marlin 搬到 DiT」的外部建议，直接引用本条驳回——收益语境是 decode 带宽 bound，本仓无此负载形态。

---

## 3. Kernel 工程

### 3.1 Persistent megakernel（Mirage MPK 单 kernel 整模型）

**机理**：把整个模型推理编译成**一个常驻 kernel**：SM 级任务图 + kernel 内去中心化调度，跨算子软件流水、计算/通信细粒度重叠，launch 税归零。OSDI'26 论文报对 vLLM/SGLang 端到端 1.0–1.7×（README 宣传 1.2–6.7× 含小模型极端情形）。

**LLM 语境**：收益在 **decode 期大量小 kernel**（每个 token 几十层、每层多个小算子，launch/边界气泡占比大）。MPK 目前支持 A100/H100/B200 的 LLM 负载（RMSNorm/MatMul/MoE/FP8），**无 DiT/视频负载先例**，MoE 支持仍在施工，sm_120 不在支持列表。

**DiT 同构性**：**负载形态相反**。turbo4 仅 4 步 × 每步 ~30 层，每层是**少量巨型 kernel**（16k×4096 级 GEMM），不是 LLM decode 的海量小 kernel。本仓实测证据：CUDA Graph 整步捕获（消灭全部 launch 税）e2e 仅 **−0.16s**（P2b 闭环）——这是「launch 税占本仓负载比例」的实测上界。MPK 比 CUDA Graph 多的部分是跨算子流水/重叠，但本仓 GEMM 已 98-99% 峰值，无空闲 SM 可填。

**本仓裁决：【N-A（本负载）】。** megakernel 消的冗余（launch 税+小 kernel 气泡）本仓实测仅 0.16s 量级且已被 CUDA Graph 可选项覆盖。唯二例外观察点：① sol aux 12232 小核 launch 开销（属已识别未关闭项）——但这是**一个稀疏算子的内部结构问题**，修它用合并/重写 aux 核即可，不需要整模型 megakernel；② MPK 的 SM 级任务图思想对「GEMM↔dequant/swiglu 内存往返 1.465s」有启发——但落点是**局部的 GEMM+epilogue 融合**（roadmap 已列为未关闭方向），不是全模型 megakernel 编译器。**不把 MPK 本体列为迁移候选。**

### 3.2 Programmatic Dependent Launch（PDL）

**机理**：同 stream 内次级 kernel 提前 launch，producer 执行 `cudaTriggerProgrammaticLaunchCompletion` 后 consumer 用 `cudaGridDependencySynchronize` 等待数据就绪——消**相邻 kernel 间的串行气泡**（launch 延迟 + tail effect）。DeepGEMM 2026.04 已加 `set_pdl`；FlashInfer TGV/TinyGEMM/CuTeDSL 后端支持 PDL；vLLM 对 DeepSeek 报 PDL 相关 4%+ e2e。

**DiT 同构性**：比 megakernel 现实——PDL 是**逐 kernel 对的局部机制**，CUDA ≥sm_90 可用（sm_120 ✓），本仓手写/CUTLASS kernel 可直接加。同构点：本仓 DiT 链路是 GEMM→dequant→swiglu→GEMM 的严格串行链，**正是 PDL 的目标形态**（consumer 的 prologue 可与 producer 的 mainloop 重叠）。

**本仓裁决：【部分迁移】——作为「quant/dequant 中转 1.465s」关闭手段的工具箱成员。**
- 量级预估：消的是 kernel 边界气泡，不是内存往返本身；在 1.465s 中转账本里，PDL 可回收的是 launch/tail 子集，**预估 −0.1~−0.3s**（按 kernel 对数量 × 每对数十微秒级气泡估，需 nsys 验证）。
- 前置条件：cublasLt **不暴露 PDL**（FlashInfer 文档实证 cuBLASLt 无此开关）→ 只对自写 kernel（dequant/swiglu/aux）与 CUTLASS 路径有效；这意味着 PDL 与「GEMM+epilogue 融合」是**替代/互补关系**——融合消往返更彻底，PDL 是不改 GEMM 库时的次优刀。
- 风险低（纯调度、bit-identical 可达）。列入族 6 工具箱，排期在 GEMM+epilogue 融合审计结论之后。

### 3.3 TMA / warp specialization / FA4 pingpong 调度

**机理**：FA3/FA4 的核心不是算法是**调度**：warp 分角色（MMA warp / softmax warpgroup / TMA warp），两个 Q tile 乒乓（一个算 MMA 时另一个算 softmax），Blackwell 上累加器驻 TMEM 消寄存器压力，软件模拟 exp 削减非 matmul 开销。FA4 对 cuDNN 9.13 达 1.3×。

**sm_120 可用性（硬约束）**：FA4 的 TMEM/tcgen05/2-CTA MMA **sm_120 全都没有**（见 §2.1 硬件事实）。sm_120 上可行的子集：TMA（sm_120 有）、warp specialization（寄存器 MMA 版）、软件 exp。**FA4 本体 N/A，FA3 风格调度思想可迁移。**

**DiT 同构性**：注意力调度与 LLM prefill 完全同构（都是长序列全注意力前向）。但本仓注意力现状：sol_cute cute_sm120 真核已在跑 + cudnn SDPA 兜底。

**本仓裁决：【部分迁移（思想层）/ N-A（FA4 本体）】。**
- 本仓 sol_cute 已是 cute 实现，warp specialization/TMA 是否已用取决于 cute_sm120 核内部（属仓内实现细节）；若未用 pingpong 调度，** dense 注意力段存在 1.1–1.3× 算子级空间**（FA4 对 cuDNN 的实测幅度折扣估）——但注意 dense 注意力在 turbo4 默认护栏臂里只占部分层/步（dense_steps=2/layers=2），Amdahl 折扣后端到端量级小。
- 高价值场景在 **768P 高分 cudnn SDPA 段**（roadmap §0.2 最大单段）：若 cute_sm120 dense 核采用 FA3 式 warp specialization 重写并覆盖该段，是候选刀——但这属于已有 sol_cute 线的演进，不是新迁移项。
- **判 FA4 移植为 N-A**：无 tcgen05/TMEM，重写量等于新写，不追。

### 3.4 DeepGEMM 式 JIT / FlashInfer plan-run 分离

**机理**：① DeepGEMM：GEMM 核按 shape JIT 编译，运行时无模板爆炸；② FlashInfer：plan（离线算调度参数/tiling）与 run（热路径零决策）分离，把 autotune/heuristic 移出热路径。

**DiT 同构性**：plan/run 分离是**负载形态无关的工程模式**，同构。

**本仓裁决：【已闭环（事实达成）】。**
- 本仓是 C 单 binary + 固定 shape（480P/768P 两档），**全部 plan 都在编译期/init 期完成**，热路径无任何运行时决策——已是 plan/run 分离的极致形态（比 FlashInfer 更彻底）。JIT 对固定 shape 无意义。
- cublasLt autotune 死路（algos=1）已闭环，不重新诉讼。
- 登记为负面知识：LLM 侧 JIT/plan-run 的复杂度来自**动态 shape**（变长序列/batch 波动），本仓无此问题。

---

## 4. 调度 / 系统级

### 4.1 vLLM PagedAttention / continuous batching

**机理**：PagedAttention=KV cache 的分页内存管理（消显存碎片+前缀共享）；continuous batching=请求级动态组批（消 decode 期 GPU 空闲）。

**本仓裁决：【N-A】。** ① 无 KV cache（§1.3），分页对象不存在；② continuous batching 是多请求 serving 技术，本仓是单请求离线生成 CLI——SGLang Diffusion 官方博客（2026.01）明说「Serving Diffusion Models Is Nothing Like LLM Serving……没有 KV cache 等价物」。仅当未来转 serving 形态（多并发出片）时 continuous batching 才有意义，届时是产品决策非优化项。

### 4.2 SGLang radix cache → 视频侧不变量

**机理**：radix tree 缓存共享前缀的 KV，多请求/多轮对话复用。**视频扩散的对应不变量是什么？** 逐项排查：
- 文本 embedding：全程真常量 → **本仓已缓存**（精确不变量缓存，族 4 精确子类，机理文档「永远该做」项）；
- RoPE 表/AdaLN conditioning：同上当常量处理；
- 跨请求 prompt 前缀：视频生成无「对话式前缀」形态，radix 无对应物；
- 跨步 latent：是状态不是 cache，相邻步不同（turbo4 4-step 每步变化大，TeaCache 类已三方印证闭环）。

**本仓裁决：【N-A（radix 本体）/ 已闭环（其不变量思想）】。** radix cache 消的冗余（跨请求前缀复用）在本仓负载形态不存在；其思想内核（精确不变量白捡）本仓已收割。

### 4.3 Layerwise weight streaming / offload

**机理**：不必全模型常驻——逐层流式载入，prefetch 下一层与当前层计算重叠。SGLang Diffusion 默认开启视频模型 layerwise offload，报峰值 VRAM −30GB 且**性能反升 58%**（消除 copy 等待）；PR#36727 对 **MiniMax-H3 的 9.7GiB VAE 在 RTX 5090** 上消掉重复 D2H（4.85s→~0，immutable 权重指针重绑定）——**注意这是同模型同卡的外部实证**。

**DiT 同构性**：**直接同构**，且是本仓 768P VRAM 墙（28.8GiB/32GB）的正对解。

**本仓裁决：【部分迁移——价值在 768P 能力轴，不在 480P 速度轴】。**
- 对 768P 高分全 sol OOM（已识别未关闭项）：layerwise streaming 让 INT8 窗+激活不必全常驻，**理论上可解开 32GB 墙**——但付 PCIe 带宽（5090 PCIe 5.0 x16 ~64GB/s vs HBM ~1.8TB/s），33B W4 权重 ~18GiB 逐层流式，每步全量过一遍 → 4 步 × 18GiB / 64GB/s ≈ 4.5s 额外传输，若不能与计算重叠则劣于混合臂现状；SGLang 报「反升 58%」的前提是 prefetch 重叠良好。**量级预估：中性到 −2s，且是能力解锁（全 sol@768P）不是纯速度。**
- SGLang PR#36727 的 **immutable 权重指针重绑定**（消重复 D2H）对本仓 VAE 臂有直接借鉴——本仓 vae_load 余尾 0.3s 已判不可闭合（显存无 headroom），但其「host store 保留 mmap 视图、swap 只重绑指针」的手法比本仓 drop/reload 更优，可作为 T4 臂备案参考。
- 排期：依赖 768P 显存账本审计；不阻塞 480P 任何档位。

### 4.4 Speculative decoding → 扩散轨迹

**机理**：LLM 投机解码=draft 小模型连猜 K 个 token，target 一次前向并行验证，接受前缀。消的是**串行 decode 步数**（把内存 bound 的逐 token 变成计算 bound 的批量验证），期望加速 = 接受长度 / (1+draft 成本)。

**DiT 同构性——关键数学差异**：LLM 验证是**精确的**（token 级分布拒绝采样，输出分布不变）；扩散轨迹是**连续高维状态**，无 token 级精确验证物。已有迁移先例（2025-2026）：
- **ASDSV**（NeurIPS'25）：对去噪步做 draft-verify——draft 模型猜 K 步，**只校验初态/末态对齐**（近似验证），K 随去噪相位自适应（早期动荡小 K、后期稳定大 K），Flux/Wan2.1 报 1.77–3.01×，VBench 降 0.3-0.4%；
- **SDVG**（arXiv 2604.17397）：块级自回归视频扩散，1.3B drafter + ImageReward 路由器验收块，1.59–2.09× 保 98.1% 质量，training-free。

**本仓裁决：【部分迁移——但当前负载下收益被双重压缩，列观察项】。**
- **压缩 1（基数塌缩）**：LLM 投机消的是几十~几百步串行 decode；turbo4 只有 **4 步**。理论上限：draft 猜对全部 4 步 = 4×，扣除 draft 成本与验证开销，乐观净收益 ≤1.5–2× DiT 段。ASDSV 的 1.77–3.01× 基线是 20-50 步模型。
- **压缩 2（draft 从哪来）**：需要比 turbo4 更便宜的 draft——候选：turbo4 自身降分辨率/降层版本、或 2-step 更激进蒸馏。本仓无现成 draft 权重。
- **与已闭环项的关系**：近似验证（只验末态）放宽了「bit-identical」，T2 默认路径不可用；T1 预览档（豁免位一致）是合法试验场。
- **量级预估**：T1 臂 DiT 段 −20~−40%（**预估**，依据 ASDSV 低步数区间折扣），前置条件=draft 权重 + 验证准则调参。列为**研究观察项，不进施工序列**；若未来 NFE 回到 8 步金线档，重估优先级上调。

---

## 5. LLM→扩散已有迁移实例盘点（2025–2026 实证）

| 技术 | 迁移先例 | 对本仓的印证 |
|---|---|---|
| Hadamard 旋转量化 | ConvRot / LRQ-DiT / CLQ / DiRotQ（W4A4 DiT 四连） | 本仓权重侧已收割（ConvRot 名号来源）；激活侧是 NVFP4 前置件 |
| NVFP4 | **FLUX.1-dev 官方 NVFP4 checkpoint**（NVIDIA/HF） | NVFP4-on-DiT 官方背书；sm_120 CUTLASS 示例已可用 |
| 投机解码 | ASDSV（NeurIPS'25，Flux/Wan2.1 1.77–3.01×）、SDVG（块级 AR 视频 1.59–2.09×） | 4-step 基数塌缩，T1 臂观察项 |
| Megakernel | Mirage MPK（OSDI'26，LLM 1.0–1.7×）——**无 DiT 先例** | 负载形态相反，N-A |
| Layerwise streaming | SGLang Diffusion 默认开启（视频模型 VRAM −30GB、性能 +58%）；PR#36727 同模型同卡（H3 VAE/5090）实证 | 768P 能力轴候选，指针重绑定手法可借鉴 |
| Cache-DiT/TeaCache | SGLang Diffusion 集成（1.69×）；**SCM 对 <8 步自动禁用** | 旁证 turbo4 4-step 缓存空间塌缩（本仓已闭环）的又一独立印证 |
| PDL | DeepGEMM 2026.04 `set_pdl`；FlashInfer TGV/TinyGEMM；vLLM DeepSeek +4% e2e | 本仓 GEMM↔dequant 串行链的候选工具，cublasLt 不支持是约束 |
| FA4 pingpong | FA4（MLSys'26）对 cuDNN 1.3×，**sm_100 专属**（TMEM/tcgen05） | sm_120 无 TMEM，本体 N-A；FA3 式调度思想对 768P cudnn SDPA 段可迁移 |
| Medusa 多头预测 | LLM 专属（预测未来 token 头） | 扩散无 token 序列，N-A；近似物=ASDSV 多步 draft |

---

## 6. 合成裁决表

| # | 技术 | 裁决 | 预期量级（本仓语境） | 前置条件 |
|---|---|---|---|---|
| 2.1 | **NVFP4/FP4（sm_120 原生）** | **【部分迁移】** | DiT 段 1.2–1.5×（预估，Amdahl 折扣后）；已识别未关闭项中最大单杆 | ① SKU 隔离臂质量验证（4-step 轨迹 FP4 激活）② sm_120 寄存器 MMA 兑现率 probe（vs INT8 745.7 TOPS）③ per-step scale 校准 |
| 4.3 | **Layerwise weight streaming** | **【部分迁移】** | 768P 能力解锁（全 sol OOM 墙）＞速度；中性到 −2s | 768P 显存账本审计；prefetch 重叠工程 |
| 3.2 | **PDL（Programmatic Dependent Launch）** | **【部分迁移】** | −0.1~−0.3s（kernel 边界气泡子集） | cublasLt 不暴露 PDL → 仅自写 kernel；排在 GEMM+epilogue 融合审计后 |
| 2.2 | **Hadamard 激活旋转** | **【部分迁移】** | 非独立收益——NVFP4/W4A4 的标配前置件 | 随 2.1 捆绑 |
| 3.3 | **FA3 式 warp specialization（768P dense 段）** | **【部分迁移】** | 768P 高分 cudnn SDPA 段算子级 1.1–1.3×（预估） | sol_cute 线演进；非新项 |
| 4.4 | **扩散投机解码（ASDSV 式）** | **【部分迁移·观察项】** | T1 臂 DiT 段 −20~−40%（预估） | draft 权重不存在；4-step 基数塌缩；不进施工序列 |
| 1 | MLA/GQA/MQA | 【N-A】 | — | DiT 无 KV cache |
| 2.3 | W4A16/Marlin | 【已闭环】 | — | 计算 bound 负载，本仓已是更激进的 W4A8 |
| 3.1 | Mirage MPK megakernel | 【N-A】 | — | 负载形态相反；launch 税实测仅 0.16s |
| 3.4 | DeepGEMM JIT / plan-run 分离 | 【已闭环】 | — | 固定 shape 单 binary，热路径零决策 |
| 4.1 | PagedAttention/continuous batching | 【N-A】 | — | 无 KV cache、单请求 |
| 4.2 | Radix cache | 【N-A】 | — | 不变量思想已收割（文本/RoPE 缓存） |
| 3.3b | FA4 本体 | 【N-A】 | — | sm_120 无 tcgen05/TMEM |
| 5 | Medusa 多头 | 【N-A】 | — | 无 token 序列 |

---

## 7. 关键判断依据备查（外部源）

- **sm_120 ≠ sm_100**：`mma.sync.aligned.kind::mxf8f6f4.block_scale`（寄存器 MMA，m16n8k64），无 tcgen05/TMEM/UMMA，SMEM 99KB，cluster 1×1×1；来源：lna-lab/blackwell-geforce-nvfp4-gemm `docs/sm120-architecture.md` + CUTLASS example 79b + cutlass#3273（CuTe DSL sm_120 NVFP4 native-TMA）。
- **NVFP4 格式与 DiT 背书**：NVIDIA Technical Blog「Introducing NVFP4」；FLUX.1-dev NVFP4 预量化 checkpoint（HF）。
- **FA4**：MLSys'26 论文（arXiv 2603.05451）+ PyTorch 官方 blog（MXFP8 FA4）；pingpong 双 Q tile + TMEM 累加器 + 专用 MMA/TMA warpgroup。
- **MPK**：OSDI'26（arXiv 2512.22219），对 vLLM/SGLang 1.0–1.7×；roadmap（mirage#498）确认无 sm_120、无 DiT。
- **PDL**：CUDA Programming Guide §4.5；DeepGEMM public-release-260416（2026.04）`set_pdl`；FlashInfer gemm_base.py（TGV/TinyGEMM/CuTeDSL 支持，cuBLASLt/CUTLASS 不支持）；vLLM DeepSeek +4% e2e（flashinfer#2887）。
- **DiT 量化迁移**：ConvRot（2512.03673）/ LRQ-DiT（2508.03485）/ CLQ（2509.24416）/ DiRotQ（2605.16732）/ Q-DiT（CVPR'25）——共同确认 DiT 激活 outlier + timestep 漂移。
- **扩散投机**：ASDSV（NeurIPS'25）、SDVG（2604.17397）。
- **SGLang Diffusion**：LMSYS blog 2026-01-16（layerwise offload 默认、Cache-DiT 1.69×、SCM <8 步禁用）+ PR#36727（H3 VAE/5090 指针重绑定实证）。
- **Marlin**：IST-DASLab/marlin README（近 4× 仅 batch ≤16-32 decode 区间）。

## 8. 修订记录

| 日期 | 变更 |
|---|---|
| 2026-10-07 | 初稿：13 项 LLM 推理技术逐条机理级迁移裁决。核心结论：NVFP4@sm_120 是最大开放单杆（部分迁移，双前置门）；layerwise streaming 对 768P 能力轴；PDL 进族 6 工具箱；MLA/GQA/Marlin/MPK/FA4 本体/radix/paging 判 N-A 或已闭环。 |
