# 十二项加速方法机理级深拆（VDN 标尺，只读）

> 来源：子代理 `913a36ea-9889-4e54-a724-1cf6495b3f51` 交付（只读）。每项回答：真实架构与数学、丢什么/得什么、带语境数字、有无被 gloss 的「线性式隐藏机制」、误分类检查、产品 vs 机制分离裁决。一手来源：arXiv 2608.12032(LoSA)/2412.11706(AsymRnR)/2505.11594(Sage3)/2603.05451(FA4)/2411.19108(TeaCache)/2506.09045(MagCache)/2507.02860(EasyCache)/2509.23202(NVFP4 实测)。

## 隐藏核心机制汇总（最重要）

| 方法 | 此前被 gloss 的隐藏机制 | 严重度 |
|---|---|---|
| **LoSA** | 「冻结支撑不冻结值」+「包络收缩」+「per-step 估计器在近无损区 miscalibrated」——把 sol 调保守≠复刻 LoSA | **高（VDN 级）** |
| **AsymRnR** | 「在 RoPE 之后缩减」是 ToMe 死因的精确反面；敏感度层级 V>H>K≫Q；只有族3 形式、族2 作用域（不碰 FFN） | 中 |
| **sage3/FP4** | 精度来源不是 FP4 而是**两级量化**（scale 撑满 E4M3）；「不保证无损」精确位置=首尾时间步 | 中 |
| **FA4** | 软件仿真指数/条件 rescaling 两 trick **不依赖 TMEM** 可移植 5090；「cuDNN 吸收 FA4」无出处应删 | 中 |
| **TeaCache/MagCache/EasyCache** | 三个不同假设（多项式相关/方向冻结+幅度律/一阶平稳），校准成本谱 50-prompt/单样本/零；MagCache「方向冻结+幅度衰减律」最强、误差界最紧 | **高** |
| **DPM-solver-v3** | EMS=用实测统计校准求解器参数化（框架此前完全未写） | 中 |
| **蒸馏族（自查盲区）** | 「turbo4 4-step>8-step」机理=DMD mode-seeking+校准网格外推，非数据怪癖 | **高** |
| CUDA Graph/融合族/NVFP4/offload/vLLM/W8A8 | 无隐藏机制，机制确认与文档一致 | — |

## 判定存活 vs 需修订

**存活**（机理检验后不变）：AsymRnR（≤1.15× 二级杠杆）、sage3（T1 opt-in/T2 禁用）、CUDA Graph（P2）、QKV-merge/融合（P1，INT8 caveat 源头=风险在 dequant 浮点序非 INT8 累加）、NVFP4（反量化宇宙不买墙钟）、CPU offload（768P capability 限定）、vLLM（单 clip N/A+形态触发条款）、W8A8（已等价，仅剩 SmoothQuant 质量项）、TeaCache 系@turbo4（预期零跳步，IonDen 先验）。

**需修订**：
1. **删**「cuDNN 9.13+ 已吸收 FA4」——被 FA4 论文自身基线证伪。
2. **LoSA 升级**：从「P3 调研项非即插即用」改为「fidelity-first 阈值化+支撑冻结是 sol 调窗的正交替代假设，值得实验（非调参）对待」。
3. **框架族2 补轴**：固定稀疏率（budget-first）vs 固定质量（fidelity-first）；族3 补注 AsymRnR K/V-only 丧失 FFN 优势。
4. **缓存族内排序**：turbo4 A/B 优先 **MagCache**（单样本校准、误差界紧）非 TeaCache。
5. **求解器族补记**：DPM-Solver-v3 EMS 为 golden 8-step 档合法无损候选，优先级高于 UniPC。
6. **新增盲区**：蒸馏族「步数-质量非单调」机理——turbo4 勿指望加步救质量；8-step 学生需重蒸馏（TDM 式 steps-aware 目标）。

## 分方法精要

**1. LoSA**：对头/query 块在 t₀=3（dense 步，质量精确算）按质量降序取最短前缀达 θ=0.99，**冻结的只是块索引（支撑集）非值**——之后每步 Q/K/V 与权重照常重算。机制=「包络收缩」：注意力从早期弥散逐步收缩，后期 top-99% 区域是 t₀ 建区域的子集（支撑收缩非迁移），末步 recall 98.7%。这引入族2 新轴：**固定稀疏率(budget-first) vs 固定质量(fidelity-first)**。

**2. AsymRnR**：Q/K/V 投影**之后**独立缩减，K/V 共享、Q 独立，注意力后只恢复 Q。**dodge 3D-RoPE 的机制**：ToMe 在映射前缩减 H 而 3D-RoPE 要求原始序列长度→结构性不兼容；AsymRnR 在 RoPE 之后缩减，RoPE 已在完整序列施加完毕。敏感度层级 V>H>K≫Q。

**3. sage3 FP4**：NVFP4（E2M1+1×16 块+E4M3 scale）做两个 MMA。核心=**两级量化**：先 per-token 把 P 每行归一到 [0, 448×6] 撑满 E4M3 范围，再 FP4 微缩——精度来源是这个 scale-range 工程非 FP4 本身。「不保证无损」精确位置=首尾时间步（首定语义布局、末定高频细节），官方配方首尾用 Sage2++。

**4. FA4**：收益来自 Blackwell 非对称扩展（B200 tensor core 翻倍但 MUFU/SMEM 没涨），全部依赖 sm_100 的 TMEM/tcgen05/2-CTA——sm_120 物理无 TMEM，FA4 不可表达（非软件问题）。「cuDNN 吸收 FA4」**证伪**（FA4 论文基线就是 cuDNN 9.13 且还有 1.3×）。可移植：软件仿真指数+条件 rescaling 不依赖 TMEM。

**5. 缓存三族**：TeaCache=embedding 输入差↔输出差多项式拟合（50-prompt 离线校准）；MagCache=残差**幅度比单调衰减律+方向冻结**（前 80% 步余弦距离≈0），单样本校准、误差界最紧；EasyCache=变换率一阶局部平稳，零离线 profiling。误差累积非独立相加（被后续网络放大），需累计阈值+最大连跳 K 双门控。**turbo4 上 IonDen 实证 8-step 蒸馏相邻 rel-L1 全≥25%、阈值 0.20 零跳步**——机制性封档证据。

**6. UniPC/DPM-solver++/v3**：半线性 ODE 线性项常数变易法解析解出（零离散误差），误差只来自神经场 Taylor 截断。**v3 隐藏机制=EMS 经验模型统计校准求解器参数化**（观察到 noise/data-prediction 非最优参数化）。高阶在 NFE≥8 有效、≤4 失效（CFG 强引导下神经场非光滑，高阶外推过冲）。

**7. CUDA Graph**：录制整段 kernel 序列单次提交，bit-identical（重放相同 kernel/参数/地址/顺序，连浮点结合序都不变）。静态形状约束→每 shape 一图。对 4-step 杠杆比 50-step 更大（总 kernel 减半但 launch 税不变，占比反升）。

**8. QKV-merge/融合/epilogue**：消 HBM 往返（三次瘦 GEMM→一次宽；norm→RoPE 中间张量留 SMEM；quant/dequant 作 GEMM epilogue）。INT8 caveat 源头：风险不在 INT8 累加（INT32 精确），在 dequant scale 应用的浮点结合序。

**9. NVFP4**：精度优势全来自 E4M3 scale 的尾数位（比 E8M0 高 3-4dB QSNR）。原生 block_scale MMA（sm_120 有）vs 反量化（本仓现状，只省存储带宽）。社区 NVFP4 checkpoint 不买墙钟的机制根源=只给存储格式没给原生 FP4 GEMM 路径。

**10. CPU offload**：能力杠杆非速度。768P 账=能不能跑（不 offload 则 OOM 跑不了），480P 纯税。

**11. vLLM/SGLang**：单 clip 无对象（batching/paged/prefix 全 N/A）。可转移残留：文本 emb 跨请求缓存（已有）、形态触发条款（批量/并发时重开）。

**12. W8A8**：本仓已 INT8 权+INT8 激活 GEMM=W8A8-equivalent（98-99% 峰值）。剩余=SmoothQuant 式激活平滑（质量鲁棒性非速度）。

**13. 蒸馏族盲区（自查）**：「turbo4 4-step>8-step」机理=DMD reverse-KL mode-seeking（推向少数模态）+学生只在校准步数网格训练、加步=外推区误差被逐步放大。推论：turbo4 勿指望加步救质量；8-step 学生需重蒸馏（TDM 式 steps-aware 目标）。SwiftVideo ≤8 步加步质量升不矛盾（连续时间一致性+轨迹对齐训练，显式多步覆盖）。
