# DeepSeek 借鉴落地清单（h3.c · 2026-10-28）

> **Asia/Shanghai.** Inventory-only：对齐 DeepSeek-V3/R1 公开课 + DeepInfer 策略层 → 本仓 C0xx。  
> **产品两轨：** Track **A** 近无损；Track **B** 有损可接受。  
> **不抢 GPU：** G1 sol+护栏（Track B）另机；本清单不排长跑。  
> **源：** [`h3c-deepseek-opt-lessons`](../superpowers/plans/2026-09-13-h3c-deepseek-opt-lessons.md)、[`deepinfer-thinking-to-h3c`](2026-10-27-deepinfer-thinking-to-h3c.md)、[`full-optimization-catalog`](2026-10-27-full-optimization-catalog.md)、[`accel-quality-theory-and-industry`](2026-10-28-accel-quality-theory-and-industry.md)。

## 一句话

DeepSeek / DeepInfer 能搬的是 **「少字节、少同步、重叠独立阶段、配置即合同、融合消中转、精度×核共设计」**；不能搬的是 MoE/MLA/DualPipe/FlashMLA/DeepEP 二进制。对 4090 单卡产品，**C015 工作区 FP16 中转已 default ON**（2026-10-28 decision dig）；C018/C019 已被 Exp-G3 **stale closed**。不是再开 Ulysses/BF16 A2A / FlashMLA。

## 对照表（可执行）

| item | origin claim | our id | status | Track | next action |
| --- | --- | --- | --- | --- | --- |
| 低精度通信砍 A2A 字节 | DeepSeek FP8 dispatch / BF16 combine（P0.1） | Ulysses `dtype_bytes=2` BF16 pack | **landed(code)**；非 4090 单卡默认路径 | A(多卡实验) | 4090 产品 **不排**；U=2 同 NUMA KPI 另档 |
| 禁热路径 DeviceSync / event 完成 | EventOverlap；forbid_cuda_sync | **C070** `H3_FORBID_HOT_SYNC`；`uly_sync_rank` 已改 event | **landed(opt-in)** | A | KPI 可开；勿当墙钟主杠杆 |
| 可缓存中间量勿重算 | Day6 on-disk KV；TE 同构 | **C062** `H3_TE_CACHE_AUTO` | **partial+ / KPI-closed@480p**；默认仍 off | A | 多 prompt 后再议默认 |
| 8 卡 = DP 吞吐非 SP | Day6 DP attention | 调度层（非 Ulysses） | **policy landed** | — | 产品 KPI 口径保持 |
| 重叠独立阶段（≠ DualPipe） | 推理双 micro-batch | **C064** TE_OVERLAP / DeferredRelease | **partial+；NO_WC_WIN@4090 ov8**；keep opt-in | A | **勿 default-on**；HS+pref 另切片 |
| 细粒度量化 + 原生低精 TC | DeepGEMM FP8/FP4 不解包 | **C001** NVFP4 W4A4 | **terminal BLOCK** | B(曾试) | **勿升默认 / 勿交 chain**；需 QAT 才重开 |
| 链式消激活中转 | Mega/链式 GEMM 思想 | **C002** FC1→FC2 chain | **blocked**(绑 C001) | — | 停 |
| Attention 保持高精度 | MLA/attn = BF16 | **C006** cuDNN SDPA Tier-0 | **landed** | A | 保持；勿 FlashMLA 替 |
| 融合消中转 / epilogue | 小核融；GEMM+epi | **C003** EPI_FUSE / SL / INK | **boundary；keep OFF**（SL +46% dit） | A | 勿 default；tiled/TC 前不谈 WC |
| E2-P3 跨 phase | 融合延伸 | **C004** | **idea** | A | C003 数值门后 |
| Hadamard / ConvRot 共设计 | outlier 抑 | **C005** + `H3_DIT_ACT_F16` ConvRot | **partial**（ConvRot F16 **default ON**） | A | 不单独立项；≠ C015 |
| 激活带宽中转降半 | 显存/带宽思维（DeepInfer T02） | **C015/C042** `H3_DIT_ACT_FP16_STAGE` | **default ON**（decision dig PASS） | A | opt-out `=0`；见 [`c015-default-on`](2026-10-28-c015-default-on-decision.md) |
| 权重窗 / streaming | 显存预算 | **C041** window24；**C043** streaming | C041 **probed**；C043 **stale closed (covered by window+prefetch)** | A | 见 [`c043-close`](2026-10-28-c043-layerwise-streaming-dig-close.md)；勿新建 stream 运行时 |
| L2 pin / RoPE memo | cache hint / 不变量 | **C017**；**C068/C069** | C017 opt-in 无 WC；C068 default-on | A | 保持 |
| PDL 短链 | launch 气泡 | **C016** | **stale closed (NO_WC_WIN / HW_UNAVAILABLE@4090)** | A | CC 8.9 无 PDL；cublasLt 无挂点；见 [`c016-close`](2026-10-28-c016-pdl-dig-close.md) |
| QKV 3→1 宽 GEMM | Sol Pillar-1 / DeepSeek Absorb 同构 | **C018** `H3_DIT_QKV_MERGE` | **stale closed**（Exp-G3 静态：INT8 已是单次宽 GEMM） | A | 停 merge 叙事；catalog 标 closed |
| QKNorm+RoPE 融合 | Pillar-1 / Absorb | **C019** `H3_DIT_QKNORM_ROPE_FUSE` | **stale closed**（Exp-G3：`qkv_rope_f16_kernel` 已融） | A | 停；残留 F32 `d_qkv` → **C015** |
| VAE cast 融合余量 | 中转税 | **C011** | **stale closed (NO_WC_WIN)**；C012 sgemm 仍 idea | A | nsys@4090：VAE roundtrip ≈29 ms；见 close doc |
| 配置即合同 / 测量卫生 | DeepInfer EnvVar / T03/T12 | **C061/C065/C066/C067/C073/C074** | **landed**（C073 waffle Δ=0） | A | 持续强制 |
| VRAM 账本 | profile 后切预算 T02 | **C040** | **landed** 480p+768p@4090 | A | 指导 C015 余量 |
| 少做前向 | 投机→合同少步 | **C030/C031** PDMD/DMAD | **landed(visual-OR)** | B | 产品主墙钟轴仍是 NFE |
| DualPipe / DualPipeV | 训练双向 PP | — | **BLOCK/N-A** | — | 不装 |
| FlashMLA / MLA absorb / MTP | decode KV | — | **BLOCK/N-A** | — | 不装 |
| DeepEP / 20 SM / IB↔NVLink | MoE+NVLink | — | **BLOCK/N-A** | — | 不装 |
| DeepGEMM 二进制 | SM90/100 | — | **BLOCK**（思想→sm_120 FP4 已 BLOCK） | — | 不 submodule |
| Continuous batch / ZOS | serving | **C081** | **defer** | — | CLI 单请求前不做 |
| Sage3 / sol 全稀疏 | 近似注意力 | **C007/C009/C010**；G1 | G1 Track-B 另机 | B | **本执行器不抢卡** |

## WIP 注意（勿踩）

C015 packing **已落地且 default ON**（decision dig 2026-10-28）。勿重开 stub 叙事；opt-out `H3_DIT_ACT_FP16_STAGE=0`。

## Track-A 下一刀（最高 ship ROI · 给执行器）

1. ~~**C015/C042**~~ — **default ON**（[`c015-default-on`](2026-10-28-c015-default-on-decision.md)；multi-prompt waffle PASS；Euler −1.41 GiB；warm WC OK）。  
2. ~~**C011**~~ — **stale closed NO_WC_WIN**（[`c011-nsys-close`](2026-10-28-c011-vae-cast-prescale-nsys-close.md)；VAE meat ≈29 ms）。  
3. ~~**C016 PDL**~~ — **stale closed**（[`c016-close`](2026-10-28-c016-pdl-dig-close.md)：4090 CC 8.9 无 PDL HW；cublasLt 无挂点；L2/P2b 上界 ≪0.3 s）。  
4. ~~**C043 streaming**~~ — **stale closed**（[`c043-close`](2026-10-28-c043-layerwise-streaming-dig-close.md)：INT8 窗+prefetch 已覆盖；无超出 C041 的清晰 VRAM/WC）。  
5. ~~**C015 default-on decision**~~ — **CLOSED YES → flipped ON**（本轮）。  
   - C012 VAE SIMT sgemm ≈214 ms 仍 idea（TF32 质量风险），非 DeepSeek 下一刀。  
   - Track-A DeepSeek 近无损主刀已收；下一刀回 catalog / 产品墙钟（NFE）或 hygiene，勿重开 C018/C019/C003-SL。


**先别排：** DualPipe、FlashMLA、DeepEP、NVFP4 重开、U=2 BF16 当 4090 主刀、EPI_FUSE 默认开、重开 C018/C019、与 G1 抢同一张卡。

## 修订

| 日期 | 说明 |
| --- | --- |
| 2026-10-28 | 初版：DeepSeek+DeepInfer → C0xx 状态表。 |
| 2026-10-28 | 对齐 Exp-G3：C018/C019 → stale closed；Track-A #1=C015，#2=C011（勿重开 merge）。 |
| 2026-10-28 | C015 PASS；C011 nsys **NO_WC_WIN closed**；下一刀 **C016 PDL dig**。 |
| 2026-10-28 | **C016 CLOSED** NO_WC_WIN/HW_UNAVAILABLE@4090（CC 8.9）；下一刀 **C043 streaming dig**。 |
| 2026-10-28 | **C043 CLOSED** covered-by-window+C041；下一刀 **C015 default-on decision dig**（不静默翻转）。 |
| 2026-10-28 | **C015 default ON**（decision dig YES；multi-prompt waffle+warm WC+Euler −1.41 GiB；DMAD LoRA soft-note）；Track-A DeepSeek 主刀收。 |
