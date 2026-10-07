# 2025–2026 视频 DiT 注意力加速全景与本仓适配裁决

> 调研日期：2026-10-07
> 前置阅读：`2026-10-05-acceleration-mechanisms.md`（机理族框架）、`2026-10-05-vdn-linear-attention.md`（VDN 深拆）、`2026-10-05-taomate-fasth3-vdn-reanalysis.md`（FastH3/VDN 实证）
> 本仓约束：MiniMax H3 33B MMDiT，turbo4 4-step 蒸馏，480P seq~16k / 768P seq~64k，单卡 RTX 5090 32GB sm_120；红线=不减窗、不动质量尺、不改 NFE；768P 高分全 sol 撞 32GB OOM。

---

## 一、推理期稀疏族：谁有 kernel，谁在 4-step 上验证过

### 1.1 方案清单与机理

| 方案 | 消什么冗余 | 训练依赖 | <10 步蒸馏实测 | 16k 收益 | 64k 收益 | sm_120 kernel |
|---|---|---|---|---|---|---|
| **SVG / SVG2** | 头部分工（空间头/时间头）+ 语义聚类置换 | 免训练，1% profiling | **未验证**（官方基线全是 50-step Wan2.1/Hunyuan） | 1.5–2.3× | 随 N 涨，64k 更优 | FlashInfer/Triton，无官方 sm_120 原生核 |
| **Radial** | 时空能量指数衰减（固定 pattern） | 免训练 | **未验证**（官方 50-step 基线） | 1.9× | 3.7×（长视频甜区） | 静态 mask 易实现，无官方 sm_120 核 |
| **SpargeAttn** | 通用块稀疏+量化 | 免训练 | **未验证** | ~2× | ~2× | CUDA 核开源，sm_120 需自编译验证 |
| **MInference** | A-shape/vertical-slash 动态模式 | 免训练 | **N/A**（LLM prefill 设计，非视频 DiT） | 模式不匹配 | 模式不匹配 | Triton/PIT，视频 DiT 上无可用性 |
| **FlexAttention 块稀疏** | PyTorch 通用块稀疏抽象 | 免训练 | **未验证** | 编译期 overhead 高 | 长序列可组合 | PyTorch 2.5+ 支持 sm_120，但性能非最优 |
| **Sparse-vDiT** | 离线搜索 diagonal/multi-diagonal/vertical-stripe | 免训练，离线搜索 | **未验证**（Wan2.1 基线 50-step） | 1.58× | 1.58× | 自定义 CUDA 核，sm_120 未验证 |
| **ScalingAttention** | 权重编码的内在稀疏拓扑（权重决定稀疏模式，非输入） | 免训练，离线拓扑提取 | **未验证** | 1.9× | 1.9× | 位级块稀疏核，sm_120 未验证 |
| **sol_cute（本仓）** | LoSA 同族，训练无关块稀疏 | 免训练 | **turbo4 上判退化**（本仓实证） | 护栏臂有效 | 768P 全 sol OOM | cute_sm120 真核（已有） |

### 1.2 关键发现

**推理期稀疏族在 4-step 蒸馏上的实证空白是普遍性的，不是本仓独有。** SVG/SVG2/Radial/SpargeAttn 全部在 50-step 基线上验证，FastWan 的「训练无关稀疏 <10 步退化」论点目前**没有公开反例**。本仓 turbo4 上 sol_cute 退化是**第一个公开实证的退化案例**，与 FastWan 论点互证。

**SVG2 的语义聚类置换（SAP）是机理上最有趣的变体**：它解决的是「critical token 散落导致 tensor core 利用率崩塌」这一稀疏通病。但 SAP 的 k-means 聚类开销在 4-step 场景下占比会被放大（4 步 × 每步聚类 vs 50 步摊还），且 kernel 是 FlashInfer/Triton，sm_120 上需重写。

**MInference 的 A-shape/vertical-slash 模式是 LLM prefill 特有的**（局部+初始 token、垂直条纹），视频 DiT 的 3D 时空注意力没有对应结构，**不适用**。

### 1.3 与本仓 sol_cute 的关系裁决

| 方案 | 与 sol_cute 关系 | 治 480P 速度 | 治 768P 内存墙 | 裁决 |
|---|---|---|---|---|
| SVG2 SAP | **替代**（同族不同路由） | 可能更优（语义聚类 vs 启发式） | 同 OOM（仍 O(Nk) 物化） | 不值得：路由改进但 kernel 无 sm_120，4-step 未验证 |
| Radial | **替代**（静态 pattern） | 更差（无头部分工） | 同 OOM | 否决：机理上不如 SVG 系列 |
| SpargeAttn | **替代**（通用块稀疏） | 类似 | 同 OOM | 否决：无增量，kernel 需重验 |
| ScalingAttention | **替代**（权重编码拓扑） | 理论上更稳（权重决定，非输入决定） | 同 OOM | 观察：机理有趣但 kernel 未成熟 |
| **任何推理期稀疏** | — | — | **不治** | **768P 内存墙需要 O(N) 或恒定内存方案，稀疏给不出** |

---

## 二、训练进权稀疏：FastH3 VSA / SLA / SLA2 / NSA / MoBA

### 2.1 方案清单与机理

| 方案 | 消什么冗余 | 训练成本 | <10 步蒸馏实测 | 16k 收益 | 64k 收益 | sm_120 kernel |
|---|---|---|---|---|---|---|
| **FastH3 VSA** | 可训练 tile 选择（粗粒度→细粒度） | 与蒸馏联合训练，~数千 GPU 时 | **官方 4-step 验证**（Wan2.1-14B 4-step） | 6× 注意力 | 6× 注意力 | **B200 sm_100a 专属**，5090 无 |
| **SLA** | 稀疏+线性补偿（critical 精确 / marginal 线性 / negligible 跳过） | 少量微调（"a few fine-tuning steps"） | **未在 4-step 验证**（Wan2.1-1.3B 基线） | 13.7× 注意力 | 恒定内存（线性分支） | 开源 CUDA 核，sm_120 需适配 |
| **SLA2** | SLA + 可学习路由 + QAT 低比特 | 与 SLA 类似，加 QAT | **未在 4-step 验证** | 18.6× 注意力 | 恒定内存 | 开源，sm_120 需适配 |
| **NSA** | 原生可训练层次稀疏（压缩+选择+滑窗） | 预训练从头训 | **N/A**（LLM 设计，未用于视频 DiT） | — | — | Triton 核，MHA 友好版存在 |
| **MoBA / VMoBA** | MoBA 视频适配（1D/2D/3D 块分区+全局选择+阈值选择） | 从头预训练 | **未在 4-step 验证**（ICLR 2026，训练加速为主） | 2.92× FLOPs | 长序列甜区 | 需自定义 kernel |

### 2.2 关键发现

**VSA 是目前唯一在 4-step 蒸馏上公开验证的稀疏方案**，但：
- 它的验证语境是「与蒸馏联合训练」（sparse-distillation concurrent），不是「蒸馏后插拔」
- kernel 是 B200 sm_100a tile-64 专属，5090 sm_120 **无法直接复用**
- 官方自认「4-step 是最低延迟档，8-step 质量更好」——VSA 没有解决少步质量问题，只是让稀疏在少步下不崩溃

**SLA/SLA2 的「稀疏+线性补偿」是机理上最契合本仓的方向**：
- critical 块走精确 softmax（保高频细节）→ 对应 sol_cute 的保留块
- marginal 块走线性（O(N)，成本 <0.5%）→ **这是恒定内存分支，治 768P OOM 的关键**
- negligible 跳过 → 推高稀疏度 70%→95%
- **SLA2 的可学习路由比 SLA 的启发式路由上限更高**，且 QAT 版本在 97% 稀疏下质量不输 full attention

**训练成本量级**（基于论文和工程常识估计）：
- SLA 微调：论文称 "a few fine-tuning steps"，估计 **10–100 GPU 时**（单卡 5090 可承受）
- SLA2 微调：类似量级，加 QAT 开销
- VSA 联合训练：与蒸馏同步，估计 **数百到数千 GPU 时**（本仓不可承受）
- VMoBA 从头预训练：**数千 GPU 时**（本仓不可承受）

### 2.3 与本仓栈的关系裁决

| 方案 | 与 sol_cute 关系 | 治 480P 速度 | 治 768P 内存墙 | 裁决 |
|---|---|---|---|---|
| FastH3 VSA | **替代**（换训练范式） | 4-step 验证有效 | 不治 | 否决：kernel 硬件外 + 训练成本不可承受 |
| **SLA** | **叠加**（在 sol 保留块上加线性分支） | 推高稀疏度→更快速度 | **线性分支恒定内存，治 OOM** | **优先推荐**：见第四节 |
| **SLA2** | **叠加**（SLA + 可学习路由 + QAT） | 上限更高 | 同 SLA | **演进方向**：需训练资源时升级 |
| NSA/MoBA/VMoBA | **替代**（换架构） | 未验证 | 不治 | 否决：训练成本不可承受，非少步设计 |

---

## 三、线性/混合注意力：VDN / SLA 式 / DeltaNet

### 3.1 定位（VDN 已深拆，此处只定位）

**VDN-the-model**：维持否决（训练+换权+B200 kernel），见 `2026-10-05-vdn-linear-attention.md` 第六节。

**VDN 机理贡献**：「近窗精确 softmax + 远帧线性状态」的时间分工混合，**线性分支恒定 d×d 状态**是稀疏给不出的性质。

### 3.2 SLA 式稀疏+线性补偿：本仓最优先方向

**机理**：按注意力权重分三级——critical（大权重，高秩）走 O(N²) 精确稀疏、marginal（中权重，低秩）走 O(N) 线性、negligible（小权重）跳过。线性分支把整个 marginal KV 压进固定 d×d 状态，**不物化 O(N²)/O(Nk) 矩阵**。

**对 768P OOM 墙的突破形态**：
- 768P seq~64k，当前全 sol 需要物化 INT8 窗+激活 28.8GiB，OOM
- SLA 式混合：critical 块保留精确稀疏（假设 5% 块），marginal 95% 块走线性（恒定状态）
- 内存：critical 块 O(N·k) 但 k 极小（5%），线性分支 O(d²) 恒定 → **总内存从 O(Nk) 降到 O(N·k_critical + d²)**，可解 768P OOM
- 计算：线性分支成本 <0.5%，critical 稀疏计算量随稀疏度降

**对 480P 的收益**：推高稀疏度从当前 ~70% 到 95%，注意力 FLOPs 降 20×，端到端 ~2.2×（SLA 论文数字）。

**训练依赖**：SLA 论文称 "a few fine-tuning steps" 即可，SLA2 需额外训可学习路由。**本仓 turbo4 上需验证**：4-step 蒸馏轨迹上，marginal 块的线性近似是否被网络放大（FastWan 退化机制是否对线性补偿同样成立）。

### 3.3 DeltaNet 族

**Gated DeltaNet**（2412.06464）：delta 规则+衰减门，逐 token 更新状态。**纯线性**，VDN 的 VDA 是其视频适配（逐帧联合最小二乘）。**纯 DeltaNet 直接用在视频 DiT 上必掉质量**（VDN 自认），必须混合。

**与本仓关系**：DeltaNet 是 SLA 线性分支的候选实现之一。SLA 的线性分支具体用什么核函数/更新规则，论文未强制绑定 DeltaNet，可用更简单的线性注意力变体。

---

## 四、注意力之外的序列压缩：token merge/prune

### 4.1 方案清单

| 方案 | 消什么冗余 | 训练依赖 | GPU 效率 | H3 FFN 占比下的潜力 |
|---|---|---|---|---|
| **ToMeSD** | token 相似合并 | 免训练 | **差**（匹配/排序/散写吃收益） | 理论高（FFN 也减），实际差 |
| **AsymRnR** | 非对称 Q/KV 缩减+恢复 | 免训练 | 1.13–1.30×（A100 实测） | 中等 |
| **ToMA** | GPU 友好重写 ToMe | 免训练 | 24%/23%（SDXL/Flux） | 低 |
| **ORBIS** | 输出引导 token 缩减+硬件协同 | 需专用硬件 | 4.5×（A100 对比） | **N/A**（硬件外） |
| **MaMe** | 矩阵化 token 合并 | 免训练 | GPU 友好矩阵操作 | 未在视频 DiT 验证 |

### 4.2 关键发现

**token merge 的 GPU 效率陷阱是结构性的**：ToMA（ICML'25）证明即使重写 GPU 友好版本，SDXL/Flux 上端到端收益仅 24%/23%。原因：merge/split 的散写操作与 FlashAttention 的连续内存访问模式根本冲突。

**AsymRnR 的 1.13–1.30× 是本族真实天花板**（A100 实测，非理论 FLOPs）。对 H3 FFN 占比高的特点，理论上 token 缩减比注意力稀疏更有效（FFN O(Nd²) 也减），但**实际收益被工程开销吃掉**。

**与 sol_cute 关系**：**强重叠**（都减注意力有效 N），叠加预期收益递减。且 token merge 需要改 RoPE 索引、处理 3D 时空位置，工程侵入性远高于注意力稀疏。

**裁决**：**不优先**。作为 sol_cute 的叠加手段，预期叠加收益 <1.2×，工程成本高，且 4-step 上未验证。

---

## 五、2026 新进展

### 5.1 SageAttention3 FP4（NeurIPS 2025，arXiv:2505.11594）

- **Blackwell 原生 FP4 microscaling**：QK^T 和 PV 用 1×16 块 FP4 + FP8 E4M3 scale，两级 P 量化
- **RTX 5090 实测 1038 TOPS，5× FlashAttention**（官方数字）
- **与本仓关系**：**叠加**（量化与稀疏正交）。sol_cute 决定算哪些块，SageAttention3 决定每块多快
- **风险**：① 非 lossless，论文自认 "accuracy not guaranteed for every model"；② 需 CUDA 12.8+ / PyTorch 2.8+ / Python 3.13；③ 与本仓 h3_dit_nvfp4 探索同线，但 SageAttention3 是注意力专用，NVFP4 是 GEMM 通用
- **裁决**：**高优先级验证**。sm_120 原生支持，官方 RTX 5090 实测。需 A/B：SageAttention3 vs 本仓当前 cudnn attention vs sol_cute 组合。

### 5.2 FlashAttention-4（arXiv:2603.05451，2026-03 beta）

- **Blackwell 算法-硬件协同设计**：异步 MMA + 大 tile + 软件模拟指数 + 条件 softmax rescaling
- **B200 BF16 1613 TFLOPs/s（71% 利用率），1.1–1.3× cuDNN 9.13**
- **sm_120 消费卡**：PR #2634 已集成，5090 上 D256 dense fwd 1.03–1.11× FA2（**增益有限**），D128 基本无增益
- **与本仓关系**：**替代**（dense attention 路径）。当前 768P 高分用 cudnn，FA4 在 5090 上增益 ~5–10%，**不显著**
- **裁决**：**观察**。FA4 的主要优势在 B200/GB200，5090 消费卡上增益有限。新版 cuDNN 已吸收 FA4 技术，等 cuDNN 更新即可。

### 5.3 ICLR 2026 新工作

- **SLA**（已详述）：稀疏+线性混合，13.7× 注意力，2.2× 端到端，开源 kernel
- **VMoBA**（已详述）：MoBA 视频适配，训练加速为主
- **QuantSparse**：量化+稀疏统一框架，Multi-Scale Salient Attention Distillation 对齐注意力偏移，3.68× 存储压缩+1.88× 端到端。**与本仓 W4A8+sol 栈同向**，但需训练
- **ScalingAttention**（已详述）：权重编码稀疏拓扑

### 5.4 其他 2026 趋势

- **稀疏+量化联合训练**成为主流（SLA2 QAT、QuantSparse、VSA+蒸馏联合）
- **「训练进权」全面压倒「推理期插拔」**：FastWan/VSA/SLA/SLA2/VMoBA 全部强调可训练，推理期免训练方案在少步蒸馏上集体失效
- **消费卡 kernel 生态滞后**：B200/sm_100a 优先，sm_120 消费卡支持滞后 3–6 个月

---

## 六、综合裁决：turbo4 4-step 蒸馏轨迹上，哪类注意力优化有实证不退化

### 6.1 实证证据汇总

| 类别 | 方案 | 4-step 实证 | 结果 |
|---|---|---|---|
| 推理期稀疏 | SVG/SVG2/Radial/SpargeAttn | **无公开验证** | — |
| 推理期稀疏 | sol_cute（本仓） | **本仓实证** | **退化**（FastWan 论点成立） |
| 训练进权稀疏 | FastH3 VSA | **官方验证** | **不退化**（但 4-step 质量本身低于 8-step，稀疏不背锅） |
| 稀疏+线性混合 | SLA/SLA2 | **未在 4-step 验证** | 待验证 |
| 纯线性/混合 | VDN | **官方验证** | **保运动>FastH3**（RAFT 11.71 vs 9.19），但需全套训练 |

### 6.2 核心结论

**在 <10 步蒸馏模型上，目前唯一有公开实证不退化的注意力优化是「训练进权」方案（VSA 联合蒸馏、VDN 混合训练）。** 所有「推理期免训练稀疏」方案要么未验证，要么（如本仓 sol_cute）实证退化。

**但 VSA/VDN 的「不退化」有条件**：
- VSA：与蒸馏**联合训练**，不是蒸馏后插拔；kernel 硬件外
- VDN：全套四阶段训练（线性对齐→端到端→LoRA→蒸馏），非即插即用

**对 FastWan 退化论点的反例**：VSA 和 VDN 都是反例，但它们通过**训练把稀疏/线性适应进权重**来避免退化，而非证明推理期稀疏在少步下有效。

### 6.3 对 turbo4 护栏臂的叠加形态

turbo4 当前默认=护栏臂（dense_steps=2/layers=2），即前 2 步+前 2 层 dense，其余 sol。叠加 SLA 式的可能形态：

1. **保守叠加**：护栏臂不动，sol 臂内 marginal 块加线性分支 → 推高 sol 臂稀疏度，护栏臂保质量
2. **激进叠加**：护栏臂也改 SLA（critical 精确 + marginal 线性），仅保留前 2 层 full dense → 需微调验证
3. **768P 专用臂**：768P 高分当前 cudnn，改 SLA 式（critical 稀疏 + marginal 线性恒定内存）→ 解 OOM

---

## 七、如果只做一件事：推荐路径

### 7.1 推荐：SLA 式「稀疏+线性补偿」叠加在 sol_cute 上

**理由**：
1. **唯一同时治 480P 速度和 768P 内存墙的方案**：线性分支恒定内存解 OOM，稀疏度推高提速度
2. **与 sol_cute 是叠加非替代**：critical 块=sol 保留块，marginal 块=新增线性分支，negligible 块=sol 已跳过
3. **训练成本可承受**：SLA 论文称 "a few fine-tuning steps"，估计 10–100 GPU 时，单卡 5090 可跑
4. **kernel 有开源基础**：SLA 官方 GitHub（thu-ml/SLA）开源 CUDA 核，需适配 sm_120
5. **机理上被三方背书**：VDN（保运动>FastH3）、SLA（95% 稀疏不掉质量）、FastH3（稀疏训进权上限高）共同指向

### 7.2 施工草案

**Phase 0：验证门（1–2 天，零训练）**
- 在 turbo4 480P 上，用当前 sol_cute 的 marginal 块（被跳过的块）做**离线线性近似误差分析**
- 方法：对 100 个 prompt，跑 dense attention，记录 marginal 块的注意力输出；用线性注意力（φ(Q)(φ(K)ᵀV)，ELU+1 核）近似，算 L2 误差
- **验证门**：若 marginal 块线性近似误差 <5%（相对于 dense 输出的相对误差），则线性补偿在 turbo4 上可行，进入 Phase 1；若误差 >10%，则 FastWan 退化机制对线性同样成立，SLA 路径需重新评估

**Phase 1：免训练原型（3–5 天）**
- 在 sol_cute 的 cute_sm120 kernel 上，给 marginal 块加线性分支（不训权重，用随机初始化的线性投影）
- 线性分支实现：参考 SLA 开源 kernel 的线性部分，适配 sm_120
- 验证：480P 上稀疏度从 ~70% 推到 90%，看质量是否掉（waffle/人眼/PSNR）
- **验证门**：90% 稀疏下 PSNR >28（vs dense），则进入 Phase 2；若 PSNR <25，需训练

**Phase 2：稀疏感知微调（1–2 周，需训练资源）**
- 用 SLA 式微调：冻结 backbone，只训线性分支投影层+可选 QKV/O LoRA
- 数据：用本仓 evals/768p 的 8 prompt × 多 seed，或爬取公开视频
- 步数：估计 500–2000 步，单卡 5090 约 1–2 天
- **验证门**：95% 稀疏下 VBench 指标不输 dense turbo4，且 768P 高分不 OOM

**Phase 3：768P 集成（1 周）**
- 768P 高分臂从 cudnn 改 SLA 混合：critical 5% 块精确稀疏 + marginal 95% 块线性
- 内存验证：确认 28.8GiB → <24GiB（留出 sol fwd 空间）
- **验证门**：768P 全 sol SLA 混合不 OOM，且质量不输当前混合臂（低分 sol+高分 cudnn）

### 7.3 风险与备选

| 风险 | 概率 | 备选 |
|---|---|---|
| turbo4 上 marginal 块线性近似误差大（FastWan 退化对线性同样成立） | 中 | 改 VDN 式「近窗 softmax+远帧线性」，线性只背远帧全局上下文，高频全留精确 |
| SLA kernel sm_120 适配困难 | 低 | 用 FlexAttention/Triton 原型验证，性能损失但功能正确 |
| 微调后质量仍掉 | 低 | 退回护栏臂扩大（dense_steps=4/layers=4），或接受 90% 而非 95% 稀疏 |
| SageAttention3 FP4 先成熟 | 中 | SageAttention3 是叠加不是竞争，可先做 SageAttention3（零训练），再做 SLA |

### 7.4 备选快速 win：SageAttention3 FP4 验证（零训练，1–2 天）

若 Phase 0 验证门不过（线性近似误差大），**立即转 SageAttention3**：
- 5090 原生 FP4，官方实测 5× FA
- 与 sol_cute 正交叠加（稀疏定块，FP4 定每块速度）
- 零训练，即插即用
- 风险：非 lossless，需 A/B 验证 turbo4 上质量

---

## 八、TOP 方案排序总表

| 排名 | 方案 | 与 sol_cute 关系 | 治 480P 速度 | 治 768P 内存墙 | 前置条件 | 预期收益 |
|---|---|---|---|---|---|---|
| **1** | **SLA 式稀疏+线性叠加** | 叠加 | ✓（稀疏度 70→95%） | **✓（线性恒定内存解 OOM）** | Phase 0 验证门 + 少量微调 | 注意力 13.7×，端到端 ~2.2× |
| **2** | **SageAttention3 FP4** | 叠加（正交） | ✓（每块 5×） | 部分（FP4 省内存） | 零训练，A/B 验证质量 | 注意力 ~5×，端到端 ~1.5–2× |
| **3** | **SLA2（可学习路由+QAT）** | 叠加 | ✓（上限更高） | ✓ | 更多训练资源 | 注意力 18.6×，端到端 ~2.5× |
| 4 | VDN 式远帧线性（仅 768P） | 叠加（768P 专用） | — | ✓ | 需训练 | 768P 内存从 O(Nk)→O(N+d²) |
| 5 | SVG2 SAP | 替代 | 可能更优 | 不治 | kernel 重写 sm_120 | 不确定 |
| 6 | ScalingAttention | 替代 | 理论更稳 | 不治 | kernel 未成熟 | 1.9× |
| 7 | token merge（AsymRnR） | 强重叠 | 弱叠加 | 不治 | 工程侵入高 | <1.3× |
| 8 | FA4 dense | 替代（dense 路径） | ~5–10% | 不治 | 等 cuDNN 更新 | 微弱 |
| 9 | FastH3 VSA | 替代 | 4-step 验证有效 | 不治 | 训练成本不可承受 + kernel 硬件外 | — |
| 10 | VMoBA/NSA | 替代 | 未验证 | 不治 | 从头预训练 | — |

---

## 九、一句话总结

**推理期免训练稀疏在 4-step 蒸馏上已判死刑（本仓实证+FastWan 论点互证），「训练进权」是唯一有实证不退化的路径；其中 SLA 式「稀疏+线性补偿」是唯一同时治 480P 速度和 768P 内存墙、训练成本可承受、与 sol_cute 叠加非替代的方案，推荐作为「如果只做一件事」的优先路径，施工从 Phase 0 免训练验证门开始。**
