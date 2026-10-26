# DeepInfer 速度优化策略 → h3.c（策略层分析）

> **日期：** 2026-10-26（Asia/Singapore）。  
> **动机：** 纠正过窄表述——「因为是 LLM 所以对 DiT 没参考价值」。DeepInfer / DeepSeek 推理栈在**全世界范围内 LLM 加速的第一梯队**；优秀工程的**速度策略**可以迁移，即使具体核（FlashMLA、DeepEP…）与视频 DiT **问题不同构**。  
> **方法：** 每条策略用固定五栏：`LLM 里买到什么 → 抽象原则 → DiT/h3.c 类比或改编 → 工作量/风险 → keep | adapt | defer`。  
> **不做：** 把核名字列成拒绝清单当作分析终点（那是 [`2026-10-07-llm-to-dit-transfer.md`](2026-10-07-llm-to-dit-transfer.md) 的同构裁决）；本文补的是 **策略层**。  
> **上游：** [DeepInfer](https://github.com/deepseek-ai/DeepInfer)、[deployment QA](https://raw.githubusercontent.com/deepseek-ai/DeepInfer/main/doc/inference_deployment_qa.md)、[ExplainDeepInfer](https://explaindeepinfer.projects.thomaspaul.dev/)、本仓 deeper-pass / takeaways / 加速机理文。



> **2026-10-27 续：** 源码级思维模式（T01–T14，含文件/开关/坑/Cxxx）见 [`2026-10-27-deepinfer-thinking-to-h3c.md`](2026-10-27-deepinfer-thinking-to-h3c.md)。本文仍是策略抽象权威。

---

## 0. 框架校正

| 错误框架 | 正确框架 |
|---|---|
| LLM 负载 ≠ DiT ⇒ DeepInfer 无价值 | LLM 负载 ≠ DiT ⇒ **算子本体**不可盲搬；**速度策略**（重叠、层级、量化共设计、融合、特化、测量、合同）仍可学 |
| 「搬不动核」= 分析结束 | 「搬不动核」= 排除错误主线；**策略表**才是继续深挖的对象 |
| 学 DeepInfer = 上 FlashMLA / continuous batch | 学 DeepInfer = 问「它消的是 launch、带宽、精度冗余、调度气泡还是步数？」再映射到 h3.c 分母 |

与视频侧机理文对齐：DeepInfer 策略大多落在 **族 5（量化）+ 族 6（工程浪费）+ 系统重叠**；少步蒸馏（族 1/8）是另一条合同轴，DeepInfer 不提供视频蒸馏，但「用合同换 NFE」与「用投机换 decode 步」在策略上同族——都是 **少做昂贵前向**。

---

## 1. 策略表

### S1 — 计算 / 通信（或 H2D）重叠

| 栏 | 内容 |
|---|---|
| **LLM 买到** | ZOS：CPU `schedule/postprocess` 与 GPU `forward/sample` 错开；EP 上 DeepEP 与 grouped GEMM 重叠；消 decode 步间气泡 |
| **抽象原则** | 关键路径上串行等待是税；用流水线让「准备下一份工作」与「执行当前工作」同时发生 |
| **DiT / h3.c** | 无多请求 decode 环，但有 **TE encode ↔ DiT 权重/激活 H2D**、**VAE 预热 ↔ DiT 首步**、可选 **CPU 拼片 ↔ GPU 下一 block**。DeferredRelease（缓冲多活一轮）是同族手段 |
| **工作量 / 风险** | 中；峰值显存与生命周期竞态（PDMD 已踩预热 OOM） |
| **裁决** | **adapt**（阶段重叠审计 → 小步实现）；**defer** 整套 ZOS / 多请求调度器 |

### S2 — 内存层级与 workspace 复用

| 栏 | 内容 |
|---|---|
| **LLM 买到** | KV pages / 固定 graph 缓冲 / pool；捕获地址固定；避免每步 alloc |
| **抽象原则** | 热路径零分配；大块 workspace 按峰值复用；长寿命缓冲与短寿命激活分层 |
| **DiT / h3.c** | 固定 shape 本就利于 **resident workspace**；768p 瓶颈是 **权重驻留** → layerwise streaming / offload（transfer【部分迁移】）；SDPA / GEMM 临时缓冲复用审计 |
| **工作量 / 风险** | streaming：高收益·高工；workspace 复用：中 |
| **裁决** | **keep** 固定 shape + resident 原则；**adapt** 768p streaming；**defer** paged KV 语义 |

### S3 — Batching / packing（即使当前单请求）

| 栏 | 内容 |
|---|---|
| **LLM 买到** | Continuous batch、padding 到 graph 档位、EP 拉大批次喂满矩阵单元 → 吞吐 |
| **抽象原则** | 硬件喜欢「够胖、够齐」的工作；稀疏/碎片请求要 **打包或填充** 才能吃满算力 |
| **DiT / h3.c** | CLI 单请求：**不做** continuous batch。仍可学——(1) **token/序列维对齐与 tiling** 减少尾部；(2) 多 clip / 多段导出时的 **pack 队列**（产品若变成批处理）；(3) 少步臂勿用错误 padding 合同冒充质量 |
| **工作量 / 风险** | 产品批处理：高；tiling 卫生：低 |
| **裁决** | **adapt** 对齐/tiling 卫生；**defer** serving batch（观察项 D） |

### S4 — 量化与硬件共设计

| 栏 | 内容 |
|---|---|
| **LLM 买到** | FP8/FP4 KV + DeepGEMM / TileKernels；换精度档抬峰值，而不是同档死磕微内核 |
| **抽象原则** | 当同精度算子已近硬件峰，下一档墙钟来自 **数值格式 × 张量核甜点** 的共设计，并配上质量合同 |
| **DiT / h3.c** | INT8 GEMM ~峰 ⇒ **NVFP4 实验臂**正是此策略；Hadamard/旋转跟 quant 绑；质量尺必须是 **视觉/waffle** 不是 token 轨迹 |
| **工作量 / 风险** | 已投入；风险在视觉门失败与链式数值 |
| **裁决** | **keep / 主杠杆**（P0 视觉门）；不要回头在同档 INT8 上「再找更快核」 |

### S5 — Kernel 融合与 epilogue

| 栏 | 内容 |
|---|---|
| **LLM 买到** | Fused MLA + cast/norm/RoPE；TileKernels 消小核海；decode 下 launch 税占比极高 |
| **抽象原则** | 大核之间的 cast/quant/bias/激活往返是带宽税；融合或 epilogue 挂接把「数学上分开的 op」合成一次访存 |
| **DiT / h3.c** | 痛点更像 **quant ↔ GEMM ↔ SwiGLU/cast 链路**（账本秒级）而非缺 MLA 核；EVT / 自写 epilogue / PDL 边界消气泡（小收益，排融合后） |
| **工作量 / 风险** | 中高；需 nsys 归因；数值回归 |
| **裁决** | **adapt**（P2，授权剖析后）；**不搬** FlashMLA 本体 |

### S6 — 按 shape / phase 特化

| 栏 | 内容 |
|---|---|
| **LLM 买到** | Prefill eager vs Decode CUDA graph；按 batch 档位多图；P/D 分离部署 |
| **抽象原则** | 不同相位的瓶颈不同（算力 vs launch vs 带宽）；**一套通吃核**往往两头不讨好 → 分相位策略 + 显式开关 |
| **DiT / h3.c** | 相位 ≈ TE / DiT 步循环 / VAE；SDPA **graph 默认关**（确定性优先）是特化政策；少步 vs 多步、480p vs 768p 用 **preset 合同** 特化，而非运行时 JIT 多 shape |
| **工作量 / 风险** | 政策已部分落地；再开 graph 子图需 pool/clean/destroy 纪律 |
| **裁决** | **keep** phase 合同与 graph 政策；**adapt** 更深的阶段计时归因；**defer** 多 shape plan-run 主加速 |

### S7 — 投机 / 早退 / 「少做前向」

| 栏 | 内容 |
|---|---|
| **LLM 买到** | DSpark / MTP：草稿 token + 置信调度验证，少跑完整 decode forward |
| **抽象原则** | 用 **便宜预测 + 选择性验证** 或 **改写积分合同** 减少昂贵前向次数 |
| **DiT / h3.c** | 直接搬投机解码 **N/A**（无 KV、4 NFE 基数紧）。策略同构落在 **PDMD/DMAD 少步蒸馏合同**（合法少 NFE）与（有损、本仓默认禁）特征缓存/reuse。早退式「步间置信停」属高风险观察项 |
| **工作量 / 风险** | 少步臂开发已齐，差 GPU 验收；有损缓存与主题禁项冲突 |
| **裁决** | **adapt** → T3 少步合同验收；**defer** decode 投机核与默认路径有损 cache |

### S8 — 调度 / 优先级流 / 禁止热路径 sync

| 栏 | 内容 |
|---|---|
| **LLM 买到** | 多 stream；forward 内 forbid sync（破 graph / 插气泡）；置信异步回写 |
| **抽象原则** | 热路径上的设备同步是隐形串行化；允许的 sync 应进 **白名单** |
| **DiT / h3.c** | 审计现有 `cudaDeviceSynchronize` / 事件等待；拷贝流与计算流分离；可选 `H3_FORBID_HOT_SYNC`（P3） |
| **工作量 / 风险** | 守卫：中；乱改易挂调试 |
| **裁决** | **adapt**（清单 → 可选宏）；不引入完整 async batch scheduler |

### S9 — 测量与回归 harness

| 栏 | 内容 |
|---|---|
| **LLM 买到** | batch-invariant 零容差 token 轨；配方与 env 合同；profile 与 KPI 分离 |
| **抽象原则** | **测不准就优化不对**；加速臂必须声明是否仍在确定性合同内；回归要同配置金线 |
| **DiT / h3.c** | `print_h3_env_contract`、P0 日志硬门、waffle 同配置门、fewstep host 测、velocity md5、timing 追加规则——已在落。继续把「加速臂开除确定性」写进报告标题 |
| **工作量 / 风险** | 低（纪律）；高（若省略） |
| **裁决** | **keep**（持续加硬） |

### S10 — 硬件感知路径

| 栏 | 内容 |
|---|---|
| **LLM 买到** | SM100 / Ascend 分路径；峰值宣传绑死硬件 |
| **抽象原则** | 优化必须声明 **目标微架构**；跨代核不能当通用真理 |
| **DiT / h3.c** | 默认 **sm_120 / 5090**；NVFP4 原生、寄存器 MMA、无 TMEM 子集——与 SM100 故事脱钩；任何外部 TFLOPS 数字只作天花板参考 |
| **工作量 / 风险** | 低（文档纪律） |
| **裁决** | **keep** |

### S11 — 配置即合同 / 失败分类

| 栏 | 内容 |
|---|---|
| **LLM 买到** | `environ.py` 集中解析；bool 陷阱文档化；sync guard；图销毁顺序防挂 |
| **抽象原则** | 生效配置必须可打印、可断言；失败要分「合同错 / OOM / 数值 / 环境」 |
| **DiT / h3.c** | fewstep 错合同 FAIL；audio shift 禁配；env 打印机；远期集中 `h3_env` 注册表 |
| **工作量 / 风险** | 注册表重构：高；现有 FAIL：已部分完成 |
| **裁决** | **keep / adapt** |

### S12 — 生成与评分分离

| 栏 | 内容 |
|---|---|
| **LLM 买到** | dump-once / score-many；structured output 与采样解耦 |
| **抽象原则** | 昂贵生成跑一遍，指标与对照在离线重复算 |
| **DiT / h3.c** | 保存 META + latent/速度场 + log；waffle/目检/md5 离线复评；禁止「边改配置边改金线」 |
| **工作量 / 风险** | 低 |
| **裁决** | **keep** |

---

## 2. 排序：对 h3.c 最值钱的可迁移策略（Top）

1. **S4 量化共设计** — 主算力杠杆（NVFP4 视觉门）  
2. **S9 + S11 测量 / 合同 / 回归** — 没有它所有加速比不可信  
3. **S5 融合消中转** — INT8 近峰后的工程秒级空间  
4. **S1 + S2 重叠与层级** — 阶段气泡 + 768p 能力  
5. **S7 少做前向（合同版）** — PDMD/DMAD，与量化正交  
6. **S6 phase 特化** — graph/preset 政策，防两头不讨好  
7. **S8 sync 纪律** — 小而硬的卫生  
8. **S3 / serving batch** — 产品形态未变前 defer  

---

## 3. 与「核搬不动」的关系（改写）

| 说法 | 地位 |
|---|---|
| FlashMLA / DeepEP / DSpark **算子**不进 h3.c 默认路径 | **仍然正确**（同构裁决） |
| 因此 DeepInfer **没什么可学** | **错误** —— 否定的是盲搬，不是策略 |
| DeepInfer 对 h3.c **只剩打印 env** | **过窄** —— 至少 S1–S12 中有 8+ 条 adapt/keep |
| 学策略 = 立刻新开算力主线 | **错误** —— 策略落地仍服从 followup：**INT8 → NVFP4 视觉门**；策略强化的是怎么做对，不是另起炉灶 |

---

## 4. 建议的后续分析（仍可不跑 GPU）

1. 把 S5 做成「中转税」检查单：对一次 480p step 的 nsys 类别预算（授权后）对照融合候选表。  
2. 把 S1 写成 TE/VAE 重叠的**安全条件**状态机（谁持有缓冲、谁可释放）。  
3. 将 S9 与 SKU 验收协议（T1–T4）显式绑成同一份「加速臂是否仍在确定性合同」矩阵。  
4. 外部：继续读 DeepInfer overlap / graph / deployment QA，只抽取策略句，不复制核 API。

---

## 5. 相关文档

| 路径 | 关系 |
|---|---|
| [`2026-10-26-optimization-roadmap.md`](2026-10-26-optimization-roadmap.md) | 优化方案落点 |
| [`2026-10-26-deepinfer-to-h3c-takeaways.md`](2026-10-26-deepinfer-to-h3c-takeaways.md) | 取舍清单（应链到本文） |
| [`2026-10-26-deepinfer-deeper-pass.md`](2026-10-26-deepinfer-deeper-pass.md) | 模块与核质量；§6 需与本文策略层分工 |
| [`2026-10-07-llm-to-dit-transfer.md`](2026-10-07-llm-to-dit-transfer.md) | 算子同构 N-A / 部分迁移 |
| [`2026-10-05-acceleration-mechanisms.md`](2026-10-05-acceleration-mechanisms.md) | 视频扩散速度机理族 |

---

## 6. 修订记录

| 日期 | 说明 |
|---|---|
| 2026-10-26 | 初稿：策略层 S1–S12；纠正「LLM ⇒ 无价值」；Top 可迁移策略排序。 |
