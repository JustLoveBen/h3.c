# DeepInfer 优化思路 → h3.c（思维模式深挖 · 细节映射）

> **日期：** 2026-10-27（Asia/Singapore）。  
> **动机：** 用户要求**不要**再写浅层「LLM 所以搬不动」拒绝清单；要深入拆 DeepInfer 的**优化思维**，再细到文件/开关地映射到 h3.c。  
> **方法：** 本机 sparse clone [`deepseek-ai/DeepInfer`](https://github.com/deepseek-ai/DeepInfer)（`/Volumes/Ben/Digen/code/_scratch/DeepInfer`，`main` @ 拉取日）+ 官方 [`doc/inference_deployment_qa.md`](https://github.com/deepseek-ai/DeepInfer/blob/main/doc/inference_deployment_qa.md) + 开源代码注释。每条用固定六栏：`DeepInfer 具体做法 → 底层思路 → h3.c 细化改编 → 坑 → 优先级 → Catalog`。  
> **上游：** [`speed-strategies` S1–S12](2026-10-26-deepinfer-speed-strategies.md)、[`deeper-pass`](2026-10-26-deepinfer-deeper-pass.md)、[`takeaways`](2026-10-26-deepinfer-to-h3c-takeaways.md)、[`full-optimization-catalog`](2026-10-27-full-optimization-catalog.md)、[`llm-to-dit-transfer`](2026-10-07-llm-to-dit-transfer.md)。  
> **不做：** 238 GPU 跑数；不碰 NVFP4 链式 WIP；不把 FlashMLA/DeepEP 当主杠杆。  
> **立场：** 「核不同构」只排除**错误主线**；优秀工程的**决策方式、测量方式、合同方式、生命周期方式**才是本文对象。

---

## 0. 怎么读 DeepInfer 的「思路」而不是「核名单」

DeepInfer 部署 QA 开篇把目标钉死为：

1. **多加速器**（NVIDIA SM100 + Ascend）  
2. **按阶段组织硬件**（Prefill 用普通 8-GPU；Decode 用 64–128 Wide-EP）  
3. **长寿命缓存**（KV >72h）换掉重复算力  

然后声明优化对象是：**更高吞吐 + 更低延迟**，手段是「计算 / 通信 / 调度 / 存储」一起抬。

对 h3.c，同构改写是：

| DeepInfer 问句 | h3.c 问句 |
|---|---|
| 这一阶段的瓶颈是算力还是带宽还是调度气泡？ | TE / DiT 步 / VAE / mux 各自瓶颈是什么？ |
| 能不能换精度档 × 核共设计？ | INT8 已近峰 → 下一档是不是 NVFP4？ |
| 能不能少跑昂贵 forward？ | 少步合同（PDMD/DMAD）而不是投机核 |
| 能不能重叠准备与执行？ | TE↔DiT H2D、VAE prewarm↔首步 |
| 能不能让配置/测量成为合同？ | env 合同、P0 硬门、waffle 同配置 |

**本文 14 条思维模式（T01–T14）** 覆盖调度、图、重叠、量化、显存、确定性、配置、测量、融合、失败分类等；每条都给可执行改编，并挂 Catalog `Cxxx`。

与 S1–S12 对照：T 是「他们怎么想」；S 是「抽象策略」；C 是「本仓机会 ID」。

---

## T01 — 阶段特化（E/P/D 分治）再优化

### DeepInfer 具体做法
- QA：Vision Encoder / Prefill / Decode **拆部署**；P 用小 EP（2–8），D 用 Wide-EP（64–128）；VE 单卡即可。  
- 代码：`EngineType.PROMPT` vs decode → `model_runner.py` 里 **PROMPT 默认不开 CUDA graph**（`use_cuda_graph = not ENFORCE_EAGER and engine_type != PROMPT`）。  
- Prefill 内还有 **时间预算 + 二次成本模型** 决定 chunk 大小（`compute_quota_limit`，`cost(q,c)`），不是固定 chunk。

### 底层思路
**不同阶段瓶颈不同 ⇒ 不要一套通吃的优化默认值。** Prefill 吃算力与 KV 写入；Decode 吃带宽与步间气泡；VE 吃另一套形状。把「政策」绑在阶段上，比绑在「模型名字」上更稳。

### h3.c 细化改编
| 阶段 | 瓶颈假设 | 政策（已有 / 待做） | 文件 / 开关 |
|---|---|---|---|
| TE | 主机侧/磁盘/首次编译摆动 | `te_cache` HIT 硬门；合同打印 TE 路径 | `h3_te_cache*`；`H3_TEXT_H_LOAD`；**C062/C060/C091** |
| DiT | INT8 GEMM 近峰；中转与精度档 | graph **默认关**；NVFP4 实验臂；E2 fuse opt-in | `H3_SDPA_CUDNN_CUDA_GRAPH`；`H3_DIT_NVFP4`；`H3_DIT_EPI_FUSE`；**C001/C003/C006** |
| VAE | cast/prescale、与 DiT 峰值叠 | skip-audio 对照；prewarm 条件化 | `--skip-audio`；`H3_GEN_VAE_PREWARM` / `H3_VAE_PREWARM_DEV`；**C092/C047** |
| mux | 已优化尾料 | 勿再开 overlap 叙事 | **C093 / Z51** |

实验：同一 prompt 打四个 span 墙钟（TE / dit / vae / mux），**禁止**用整段 wall 当 DiT 加速比分母。

### 坑
- 把 Decode 的 graph-on 习惯搬到 DiT 对照 → 已踩过 sm_120 不确定；政策是 **对照关 graph**。  
- 跨阶段混 waffle（TE live vs load）→ 假失败。

### 优先级
**P0 纪律（持续）**；阶段账本插桩属 **P1/P2（C040 类）**。

### Catalog
C006, C060, C062, C091, C092；S6。

---

## T02 — 「先 profile 峰值，再按利用率切预算」的显存思维

### DeepInfer 具体做法
`worker/base.py::profile_num_available_blocks`：
1. `empty_cache` + `reset_peak_memory_stats`  
2. `profile_run`（可选 `DEEPINFER_ENABLE_MEMORY_TRACING` dump snapshot）  
3. `peak = peak_torch + max(non_torch, 0)`  
4. `available = total * gpu_memory_utilization - peak - state_size*2`  
5. **打一条结构化 Memory profiling results 日志**（duration、initial、peak、kv_cache_size、utilization）  
6. OOM 文案指向调 `gpu_memory_utilization` / `max_model_len`（`llm_engine.py`）

### 底层思路
**显存不是「猜还剩多少」；是「测完峰值再留 utilization 安全垫」。** 追踪（tracing）与 KPI 墙钟分离：`ENABLE_MEMORY_TRACING` 只在 profile 用。

### h3.c 细化改编
1. **C040**：在 DiT 关键窗 / 激活峰值 / VAE 进入前插 `cudaMemGetInfo` 分桶，写到 generate 日志（字段建议：`vram_free_before_dit_giB`、`vram_peak_dit_giB`、`vram_before_vae_giB`）。  
2. 复用 `H3_GEN_MEM_TRACE` **仅诊断**；KPI 跑必须 unset（已有纪律 **C065**）。  
3. 768p 决策顺序：**账本 → FP16 中转（C015/C042）→ 减窗探针（C041）→ streaming（C043）**，禁止无数字谈 streaming。  
4. OOM 退出码分类（见 T12）指向「哪一阶段顶满」。

实验（授权 238 后）：480p INT8 与 768p（若能起）各跑 1 clip，只收集分桶，不改默认。

### 坑
- 把 tracing 开着测 dit ms → 虚高。  
- 用 `cudaMemGetInfo` 瞬时值当峰值（需 peak tracker 或分阶段采样）。

### 优先级
**P2 能力轴第一刀（C040）**；不挡 P0 视觉门。

### Catalog
C040–C044, C047, C065；S2。

---

## T03 — 配置即合同（集中 EnvVar + 启动打印 + 类型失败）

### DeepInfer 具体做法
- `environ.py`：`EnvVar` 描述符，**首次访问解析、缓存、非法值带变量名抛错**；`choice` 枚举；`rearm` / `rearm_all` 测后可重读。  
- `llm_engine` 启动日志一行打出 revision / dtype / max_seq / EP/DP / chunked_prefill / **zero_overhead_scheduling** / seed …  
- README：确定性路径写死组合：`BATCH_INVARIANT=1` + `ENABLE_MTP_VARLEN=0` + 固定 `seq_seed`。

### 底层思路
**生效配置必须可打印、可断言、失败可分类。** 散落 `getenv` 且静默默认 = 测错自己不知道。

### h3.c 细化改编
| 动作 | 落点 | Catalog |
|---|---|---|
| 已有离线打印机 | `tools/print_h3_env_contract.py --expect-p0-visual` | C061 partial |
| **立即开发：** generate 启动内联同一合同块 | `tools/h3_cuda_generate.c`（或公共 `h3_env_contract_fprint`） | **C061** |
| fewstep 错合同 FAIL | `--fewstep-contract` / `--fewstep-preset`（已落地） | C032 |
| 远期集中注册表 | 对标 `EnvVar` 的 `h3_env`（typed + rearm） | C072 |
| 失败标签 | `FAIL_CONTRACT` / `FAIL_OOM_DIT` / `FAIL_OOM_VAE` / `FAIL_ENV` | C074 |

实验：无 GPU 也可做——启动 generate `--help` 路径或 dry-run 打印合同；单测比对打印机输出字段集合。

### 坑
- 只打印、不 assert → 仍会用错合同出片。  
- 大重构 `h3_env` 插队挡视觉门。

### 优先级
**C061 = 无 GPU 最高价值下一刀（P0-dev）**；C072 远期。

### Catalog
C061, C032, C072, C074；S9/S11。

---

## T04 — 确定性 / batch-invariant 是**产品开关**，不是优化副作用

### DeepInfer 具体做法
- `DEEPINFER_BATCH_INVARIANT=1`：注意力路径传 `batch_invariant=`；`runtime_config.configure_worker_kernels` 对 DeepGEMM/TileKernels `use_deterministic_algorithms(True)`；与 EPLB 重排互斥时关掉 multiple reduction（`userbuffer.py`）。  
- 确定性采样：**显式要求关 MTP varlen**，避免投机路径破坏轨迹。  
- NPU 上暂时 assert 不许开 batch invariant（数值不稳则诚实关闭）。

### 底层思路
**速度臂与确定性臂可以共存于一仓，但必须声明「当前是否仍在确定性合同内」。** 优化默认可以非确定；对照门必须确定。

### h3.c 细化改编
1. P0 视觉/waffle：**graph 关、同 seed、同 TE 路径、同 skip-audio**（runbook 已有）。  
2. 任何加速开关（`H3_DIT_EPI_FUSE`、`H3_DIT_NVFP4`、临时开 graph）在日志打 `determinism_contract=relaxed|strict`。  
3. waffle **同配置门**已落地（C067）——跨臂直接 FAIL。  
4. 1-step velocity md5 只在 strict 合同下宣称「可复现」。

实验：INT8 strict 两跑 md5；再开 `H3_DIT_EPI_FUSE=1` 只比 waffle 区间，不比 bit-identical（E2 DESIGN 已写明）。

### 坑
- 用非确定路径的速度数字当默认 SKU 金线。  
- 关 deterministic 却仍报 md5 回归绿。

### 优先级
**P0 持续**；与 C001 视觉门绑定。

### Catalog
C060, C066, C067, C001；S9。

---

## T05 — 热路径禁 sync：守卫 + 白名单，而不是口头约定

### DeepInfer 具体做法
`utils.py::forbid_cuda_sync` / `allow_cuda_sync`：
- 基于 `torch.cuda.set_sync_debug_mode("error")` 的仲裁器（线程栈 top）。  
- `execute_model` / `execute_sample` 文档写死 **IMPORTANT: must NOT contain any cuda synchronize**，并用 `with forbid_cuda_sync()` 包住。  
- `DEEPINFER_DISABLE_SYNC_GUARD=1` 仅 debug（打印/dump）时关。  
- Event.synchronize **故意不在** torch sync debug 覆盖内 → ZOS 合法等 event。

### 底层思路
**同步是隐形串行化；要用机制强制，而不是 code review 盼自觉。** 生命周期合法 sync 走 allow 白名单。

### h3.c 细化改编
1. **C070**：审计 `h3_gpu_cuda.cu` / VAE / generate 中 `cudaDeviceSynchronize` 与阻塞 D2H；分类：`lifecycle` / `kpi_boundary` / `debug` / **`hot_suspect`**。  
2. 可选 debug 宏或 env：`H3_FORBID_HOT_SYNC=1` 在 DiT block 循环内 assert（CUDA 无 torch 级 sync debug，可用「禁止调用包装」或 nsys 标注）。  
3. 白名单写进 [`te-lifecycle-audit`](2026-10-26-te-lifecycle-audit.md)（已有草案）：TE join、VAE wait、nsys 边界、进程退出。  
4. KPI：`cuda_timer.flush_out(non_blocking=...)` 思维 → 墙钟测量尽量 event 差分，少 device sync。

实验：`grep -n cudaDeviceSynchronize` 出表 → 标 hot_suspect → 能改成 event 的改。

### 坑
- 在 forward 里加校验打印引入 sync，却怪「优化退步」。  
- 过度禁 sync 导致错误的异步 UAF。

### 优先级
**P3 工程卫生（C070）**；不挡主线。

### Catalog
C070, C064；S8。

---

## T06 — DeferredRelease：生命周期「晚一轮」换重叠

### DeepInfer 具体做法
`llm_engine.py::DeferredRelease`：
- `collect_*` 把本步产出投入待释池；  
- 在下一步的 `pre_sampler_sync` 才 `release`，注释写明 **与 GPU 计算 overlap**；  
- clone unpin 先 `Event.record`，release 时 `synchronize` 再 unpin。

### 底层思路
**「读完才能 free」与「尽早腾显存」冲突时，用明确的延迟释放对象 + event，而不是到处 `free`。**

### h3.c 细化改编
已有笔记 [`2026-10-26-te-lifecycle-audit.md`](2026-10-26-te-lifecycle-audit.md)：
- TE `text_emb` 何时 free、NVFP4 store free、host_stage join（NVFP4 **禁止**与 DiT overlap）。  
- VAE prewarm：`resident=0 && window>0` → host-only，除非 `H3_VAE_PREWARM_DEV=1`。  
- **C064 / C047**：把「DeferredRelease 白名单」落成代码注释表 + 可选 struct（对象名、产生阶段、释放阶段、依赖 event）。

安全重叠候选（仅审计通过后）：
1. INT8 disk→host stage 与 TE encode（已有 `H3_DIT_HOST_STAGE_TE`）  
2. VAE host prewarm 与 DiT 尾步（已有 kick；device prefill 有空闲门槛）  
3. 禁止：NVFP4 host_stage 与 DiT；TE packed 与 DiT GPU prefetch 在 `latent_t≥32` 默认跳过

### 坑
- PDMD 预热 OOM 已是先例：重叠前必须看 free GiB。  
- 延迟释放过久 → resident 顶满 768p。

### 优先级
**P2（C064）**；真机峰值后再动代码。

### Catalog
C064, C047, C075；S1。

---

## T07 — 零开销调度：CPU 准备与 GPU 执行错开

### DeepInfer 具体做法
QA「Asynchronous scheduling」：
- 朴素：`schedule → forward → sample → postprocess`，sample 后 CPU-GPU sync → GPU 空等 postprocess+下一次 schedule。  
- ZOS：GPU 跑 `forward(N)/sample(N)` 时，CPU 跑 `schedule(N+1)/launch_forward(N+1)/postprocess(N-1)`。  
- 代码：`DEEPINFER_ZERO_OVERHEAD_SCHEDULING`；`MetadataZOS` / `WorkerStepState` 用 pinned remap + event；`cuda_timer.flush_out(non_blocking=True)`；graph runner `input_on_device=True` 时零拷贝视图。

### 底层思路
**气泡税来自「必须等结果才能想下一步」。** 能预测或能缓冲的元数据提前准备；结果用 event 延迟回收。

### h3.c 细化改编
单请求 CLI **不做** ZOS 调度器（C081 defer）。同构落点：
1. 下一 DiT block 的权重窗准备 vs 当前 block GEMM（窗侧重叠 **Z57 已做**；L2 pin 仍开 **C017**）。  
2. TE 完成后 DiT 首步前的 host staging（`H3_DIT_HOST_STAGE_*`）。  
3. ffmpeg/mux 与尾段无关（**C093 勿再开**）。  
4. 少步路径：预计算 `student_grid` σ（金表已有 dump 工具）避免步间主机决策。

### 坑
- 把 continuous batch / ZOS 当 5090 单卡出片的主优化 → 浪费月份。  
- 重叠引入不确定性却仍宣称 md5 门。

### 优先级
**已有开关调参 + C017 小刀；整机 ZOS defer。**

### Catalog
C064, C017, C075, C081；S1/S3。

---

## T08 — 量化与硬件共设计（按**张量角色**选格式）

### DeepInfer 具体做法
QA Low-precision：
- Main KV：per-16 e4m3（类 NVFP4 **无** global scale）  
- Indexer KV：per-32 ue8m0 MXFP4  
- MoE routed experts：per-32 ue8m0 MXFP4  

同一系统里 **不同张量不同 scheme**，不是「全局 FP4」。

内核侧：`configure_worker_kernels` 默认 `deep_gemm.set_pdl(True)` / `tile_kernels.set_pdl(True)`——**PDL 是默认政策**，确定性另开。

### 底层思路
**当同精度核已贴硬件峰，下一档墙钟来自数值格式 × tensor core 路径，并按张量敏感度分档；质量用合同门，不靠「感觉能跑」。**

### h3.c 细化改编
1. **主杠杆仍是 C001 NVFP4 视觉门**（INT8 GEMM ~98–99% sm_120 → 别死磕更快 INT8）。  
2. 链式 FC1→FC2（C002）= 「消激活中转」的共设计第二步——**视觉门 + 用户确认前不动 WIP**。  
3. Hadamard/ConvRot（C005）绑在 FP4 outlier，不单独立项。  
4. E2 `H3_DIT_EPI_FUSE`（C003）= **同精度**消 quant↔GEMM 往返（S5），与换档正交。  
5. 注意力 FP4（SageAttn3，C009）仅独立探针。

实验（授权后）：P0 runbook INT8 金线 → NVFP4 `H3_DIT_NVFP4=1` chain 关 → 目检 + 同配置 waffle。

### 坑
- 把 SM100 DeepGEMM 宣传数字当 5090 预期。  
- 视觉未过升默认 / 强推链式。  
- 用 cosine 当通过尺（followup 已否）。

### 优先级
**P0（C001）**；C003 可并行真机 A/B；C002 gated。

### Catalog
C001–C005, C009, C044；S4/S5。

---

## T09 — 融合与「中转税」：大核旁的小拷贝也是主敌

### DeepInfer 具体做法
- Mega MoE：「无论 EP/batch，融合显著快于 unfused」。  
- TileLang `fused_transform` / `fused_sample` / quant 旁路。  
- `try_concat`：全 pinned → 多路 non_blocking H2D 再 cat；否则先拼一个 pinned 再一次 H2D——**减少 pinned 分配次数与往返**。

### 底层思路
**峰值核之外，cast/quant/bias/激活往返与零碎 H2D 才是常被低估的税。** 先 nsys 归因，再融；融之前可用「少分配、pin、non_blocking」降税。

### h3.c 细化改编
1. **C003** E2：`out/review-2026-10-07/e2-epi-fuse/` + `H3_DIT_EPI_FUSE=1`；`ab.sh` hot clip2。  
2. **C004** 跨 phase 融：C003 数值门后再开。  
3. **C011** VAE cast/prescale 余量：nsys 后。  
4. **C016** PDL：排在融合后、自写短链。  
5. Host：检查 TE/权重上传是否重复 pin/alloc（对标 `try_concat`）。

### 坑
- 无 profile 融错位置。  
- 追求 bit-exact 把 requant 挪回失去收益（E2 DESIGN §3）。

### 优先级
**P1 真机：C003**；其余 P2–P3。

### Catalog
C003, C004, C011, C016, C018, C019；S5。

---

## T10 — 「少做昂贵前向」的经济学（投机 → 合同变体）

### DeepInfer 具体做法
DSpark **变长验证**：Confidence Head 估每个 draft 接受率；按硬件吞吐曲线选边际收益最大的验证前缀。  
表：固定-5 → 动态验证长度 5→3.2–4.4，接受长度几乎不变，生产点约 **+4–7% tok/s**。  
与异步调度交互：CPU 只定总预算，**GPU TileLang 核**做 cumprod/sort/top-k。

### 底层思路
**贵的是完整 forward；用便宜预测 + 合同化验证预算，少付完整代价。** 关键不是「有投机核」，而是「验证预算也是优化变量」。

### h3.c 细化改编
1. **不搬** MTP/DSpark 到 DiT。  
2. **同族：** PDMD-4 / DMAD-4 **少步合同**（C030/C031）——用蒸馏权重换 NFE；`--fewstep-preset` 锁死 steps/shift/sampler。  
3. 验收经济学：加速比 = 同画质合同下的 wall；错合同加载 FAIL（C032）。  
4. C037 扩散投机仅观察，不进主序列。

实验（授权后）：`pdmd4` 480p 同 seed；`dmad4` 必须 `lcm_renoise`（Euler ≠ 通过）。

### 坑
- 2NFE 冲观感；turbo4 错合同计加速比。  
- 把投机解码当默认路径。

### 优先级
**P1 C030 → P2 C031**（不插队 C001）。

### Catalog
C030–C037；S7。

---

## T11 — 图捕获：离散尺寸、padding、池、销毁、测量特化

### DeepInfer 具体做法
`cuda_graph_runner.py`：
- `build_graph_batch_sizes`：先幂次再固定 interval，末档 max。  
- 大→小 capture；共享 `graph_memory_pool`；padding metadata。  
- `select`：`bisect_left` 选 ≥ total_q 的图。  
- `destroy` 解绑 timer、清 handle。  
- `CUDAGraphTimer`：**graph 内 event ping-pong**，避免破坏捕获；`keep_graph=True` 为了挂 probe。  
- Prefill / `ENFORCE_EAGER` 可关图；OOM 提示降 `gpu_memory_utilization`。

### 底层思路
**图是「固定 shape 热路径」的特化工具；要有尺寸策略、内存池、销毁纪律，以及「图内怎么测」的特化计时。** 不是「能 capture 就默认开」。

### h3.c 细化改编
1. 现状政策正确：**cuDNN SDPA graph 默认关**（sm_120 不确定）——**C006/C077**。  
2. 若将来再开：必须有 pool/clean/destroy 清单；对照臂标题标明 graph-on。  
3. DiT 固定 shape 已利于图，但 **确定性合同优先于图加速**。  
4. 学习点：KPI 计时与 graph 共存时用 **event buffer flip**，不要在 replay 前后乱 sync。

### 坑
- 用旧 graph-on 的「12%」数字与 graph-off 比。  
- capture 中 realloc（`h3_gpu_cuda.cu` 已有 forbid realloc 注释，保持）。

### 优先级
**政策保持（landed）**；仅文档/C077 备忘。

### Catalog
C006, C077；S6。

---

## T12 — 测量卫生：相位 timer、非阻塞 flush、生成/评分分离

### DeepInfer 具体做法
- `timer.py`：`cpu_timer` / `cuda_timer` / `cuda_graph_timer` 三分；`flush_out(non_blocking=)`。  
- `examples/benchmark.py` + `examples/sweep` 复现 Pareto。  
- eval 与 serving 分离（`eval/`）。  
- 启动后 `set_random_seed_with_rank`：**profile/capture 不得污染正式 seed**（`warm_up_model` 末尾重置）。

### 底层思路
**测不准就优化不对；加速臂要声明测量合同；warmup/profile 副作用要清掉。**

### h3.c 细化改编
| 项 | 动作 | ID |
|---|---|---|
| env 合同 | 启动打印 | C061 |
| P0 日志硬门 | `p0_protocol_check_log.py` | C066 |
| waffle 同配置 | `waffle_same_config_gate.py` | C067 |
| 墙钟 | 禁 `H3_GEN_MEM_TRACE` | C065 |
| dump-once / score-many | `H3_DUMP_LATENT` / replay | C073 |
| timing 文档 | 只追加 | followup |
| seed | 任何 probe/autotune 后恢复用户 seed | C007 相关 |

### 坑
- 边改配置边改金线。  
- TE 7–58s 摆动当 dit 加速比（→ C062）。

### 优先级
**P0-dev 持续；C062 P1。**

### Catalog
C060–C067, C073；S9/S12。

---

## T13 — 近似重算换全量重放（有界窗口）

### DeepInfer 具体做法
SWA bounded replay：只重算最近 128 token，注意力可见域截在重放段内 → **重算代价与上下文长度解耦**。  
Hit 政策：block KV 定命中长度；SWA miss 则退一 block。  
YOCO `compact` 模式真正丢掉截断区 query。

### 底层思路
**缓存 miss 时不要默认「全量重做」；用有界近似 + 明确质量合同换延迟。**

### h3.c 细化改编
DiT **无**跨步 KV，不能盲搬 SWA。同构场景：
1. **TE cache miss** → 是否允许「近似 TE」（一般 **否**，TE 是条件，质量敏感）→ 更应强化 **精确 cache**（C062）。  
2. **step_cache / token_reduction**（C096/C095）→ 必须独立质量合同，默认关。  
3. **VDN / SelfLift / TAE**（C080/C083）→ 产品分档「降合同换速度」，对标「有界近似」。  
4. 768p 减窗探针（C041）= 显存版有界：一次性实验，不解默认。

### 坑
- 把有损 step_cache 当默认金线。  
- 用 SWA 叙事开 DiT 稀疏默认。

### 优先级
**观察 / 产品分档；C062 精确缓存优先。**

### Catalog
C062, C041, C080, C083, C095, C096；S2/S7。

---

## T14 — 编译与运行解耦 + JIT/warmup 缓存层级

### DeepInfer 具体做法
- Structured Output：专用机编译 grammar → 请求进引擎前取结果。  
- `DEEPINFER_JIT_CACHE_DIR`：tilelang / deep_gemm / deep_ep 分目录；**warmup 目录挂在 miss 查找链**（`history_warmup`）。  
- `DECODE_PERF_CACHE_DIR` + TTL：投机调度用的硬件吞吐曲线可缓存。

### 底层思路
**冷启动税与稳态热路径分开治；编译产物与曲线要可缓存、可失效。**

### h3.c 细化改编
1. **C007**：sol autotune 是否每 clip 付 ~1s → 持久化 `(T,shape,GPU)` 或证实 cold-only 后关项。  
2. **C062**：`te_cache` 产品化默认策略（key 含 layers/dim/ids）。  
3. CUTLASS/sol 的本地 cache 目录文档化（对标 JIT_CACHE_DIR）。  
4. E2/NVFP4 的 bench 与 generate 二进制分离（已有 `bench/`）。

### 坑
- cache key 漏字段 → 静默错结果。  
- warmup 目录污染确定性对照。

### 优先级
**C062 P1；C007 P2。**

### Catalog
C007, C062；S6/S9。

---

## 1. 思维 → 策略 → 机会 总表

| Thinking | 核心一句话 | S | 最热 C | 优先级 |
|---|---|---|---|---|
| T01 阶段特化 | 政策绑阶段 | S6 | C006/C062 | P0 纪律 |
| T02 profile 后切预算 | 先测峰值 | S2 | **C040** | P2 |
| T03 配置即合同 | 可打印可断言 | S11 | **C061** | **P0-dev** |
| T04 确定性开关 | 速度臂声明合同 | S9 | C060/C067 | P0 |
| T05 禁热路径 sync | 机制强制 | S8 | C070 | P3 |
| T06 DeferredRelease | 晚一轮释放 | S1 | C064 | P2 |
| T07 CPU/GPU 错开 | 消气泡 | S1 | C064/C017 | 调参/P3 |
| T08 量化共设计 | 换档×核 | S4 | **C001** | **P0** |
| T09 融合消中转 | 大核旁小税 | S5 | **C003** | P1 |
| T10 少做前向 | 合同换 NFE | S7 | C030/C031 | P1–P2 |
| T11 图纪律 | 特化非默认 | S6 | C006/C077 | 政策 |
| T12 测量卫生 | 测准再优化 | S9/S12 | C061–C067 | P0-dev |
| T13 有界近似 | miss 不全量重做 | S2/S7 | C062/C080 | 观察 |
| T14 编译解耦 | 冷热分离 | S6 | C007/C062 | P1–P2 |

---

## 2. 仍应诚实 defer 的（不是「没价值」，是分母为 0）

| DeepInfer 能力 | 为何 defer | 何时重开 |
|---|---|---|
| Wide-EP / DeepEP / EPLB | 无 MoE | 多卡 MoE DiT（无计划） |
| FlashMLA / paged KV | 无跨步 KV | — |
| Continuous batch / ZOS 整机 | 单请求 CLI | 产品变多并发出片（C081） |
| DSpark 核 | 无 draft token 轨迹 | 仅策略已迁到少步合同 |
| SM100 专核数字 | 目标是 sm_120 | 换硬件世代 |

这些 **不否定** T01–T14；它们证明 DeepInfer 强在「按分母下药」。

---

## 3. 对 h3.c 最锋利的 5 个可执行映射（给执行层）

1. **T08→C001**：INT8 近峰 ⇒ 下一刀是 NVFP4 **视觉门**（要 238），不是再抠 INT8 GEMM。  
2. **T03+T12→C061**：generate 启动内联 env 合同（**无 GPU 可做**），把「测错」从根源掐掉。  
3. **T09→C003**：`H3_DIT_EPI_FUSE` 真机 A/B——同精度里最具体的秒级刀（代码已齐）。  
4. **T10→C030**：PDMD-4NFE 合同验收——少做前向的视频版（与量化正交）。  
5. **T02→C040**：768p 显存账本插桩——一切 streaming/减窗/FP16 中转的前提。

---

## 4. 与旧文关系

| 文档 | 本文补什么 |
|---|---|
| llm-to-dit-transfer / takeaways | 同构裁决与「可留/扔掉」→ 本文不重复拒绝清单 |
| speed-strategies S1–S12 | 策略抽象 → 本文给 **源码级思维 + 文件开关** |
| deeper-pass N1–N8 / §6 | 工程卫生与核质量 → 本文并入 T 并挂 C |
| full-optimization-catalog | 机会表 → 本文是 **思维索引**；执行仍看 catalog/roadmap/followup |

---

## 5. 源码锚点（DeepInfer clone）

| 主题 | 路径 |
|---|---|
| 部署思维 | `doc/inference_deployment_qa.md` |
| Env 合同 | `deepinfer/environ.py` |
| 启动打印 | `deepinfer/engine/llm_engine.py`（engine 构造日志） |
| DeferredRelease | `deepinfer/engine/llm_engine.py::DeferredRelease` |
| 禁 sync | `deepinfer/utils.py::forbid_cuda_sync` |
| 显存 profile | `deepinfer/worker/base.py::profile_num_available_blocks` |
| Graph | `deepinfer/worker/cuda_graph_runner.py` |
| ZOS 元数据 | `deepinfer/worker/zos_status.py` |
| Forward 无 sync | `deepinfer/worker/model_runner.py::execute_model` |
| 确定性核 | `deepinfer/worker/runtime_config.py::configure_worker_kernels` |
| Timer | `deepinfer/timer.py` |

本机副本：`/Volumes/Ben/Digen/code/_scratch/DeepInfer`（分析用，**不进** h3.c git）。

---

## 6. 修订记录

| 日期 | 说明 |
|---|---|
| 2026-10-27 | 初版：14 条思维模式深挖；挂 S 与 C；五锋利动作；源码锚点。不跑 GPU；不碰链式 WIP。 |
