# h3.c 全量优化机会目录（长周期 · 2026-10-27）

> **日期：** 2026-10-27（Asia/Singapore）。  
> **对象：** MiniMax-H3 原生推理仓 `h3.c`（Metal + CUDA/sm_120 旁路；默认 INT8 DiT、NVFP4 实验臂、T3 少步 LoRA）。  
> **性质：** **一份可指导数月工作的穷尽式优化目录**——把 `docs/research/` 既有研究、死路审计、DeepInfer 策略、管线源码扫掠去重收成单一 taxonomy；**不是**本周执行 checklist（执行仍以 [`2026-10-26-followup-dev-plan.md`](2026-10-26-followup-dev-plan.md) / [`2026-10-26-optimization-roadmap.md`](2026-10-26-optimization-roadmap.md) 为准）。  
> **约束（本文撰写时）：** 不跑 GPU/238；不碰 NVFP4 链式 WIP；不发明新的墙钟/质量数字——凡涉及量级均引用既有文档。  
> **用法：** 新想法先在本目录找 ID；能挂上轴 + 证据 + 状态才开工；「已关闭」节禁止再诉讼。

---

## 0. 一句话与读法

优化只能消四类税：**每步算力、步数(NFE)、显存能力、工程浪费（含测不准）**。另有第五轴是产品/serving 形态变更。  
DeepInfer 贡献的是**速度策略**（S1–S12），不是 FlashMLA 本体；视频侧机理见族 1–8（[`2026-10-05-acceleration-mechanisms.md`](2026-10-05-acceleration-mechanisms.md)）。

| 章节 | 用途 |
|---|---|
| §1 五轴 taxonomy | 禁止混 SKU 验收 |
| §2 管线地图 | 按阶段找责任文件 |
| §3 已关闭 / 勿重开 | 止损 |
| §4 全量机会表（C001+） | 主目录 |
| §5 旧文档交叉索引 | A*/S*/E*/M*/L* 映射 |
| §6 近 2–8 周 backlog | 可开工排序 |
| §7 更长地平线附录 | 观察 / 产品 / 研究 |
| §8 CSV 附录 | 机读摘要 |

---

## 1. 五轴 taxonomy（正交，禁止合成一个验收）

| 轴 | 代码 | 消什么 | 本仓主杠杆 | 质量尺 |
|---|---|---|---|---|
| **A 每步算力** | `A` | 单次 DiT/VAE forward 的算子+中转税 | INT8 近峰后 → **NVFP4**；同精度 → **E2 融合** | 同配置 waffle + 目检；速度场 md5 仅 INT8 稳臂 |
| **B 步数/合同** | `B` | NFE / 采样合同 | **PDMD-4 → DMAD `lcm_renoise`** | 合同自检 PASS + 视觉；禁止错合同计加速 |
| **C 显存/能力** | `C` | 能否跑 768p 全序列 / 长片 | 账本 → FP16 中转 / 减窗探针 / layerwise streaming / NVFP4 权重体积 | 先「能跑完」再谈加速比 |
| **D 工程/测量** | `D` | 测不准、气泡、生命周期、合同漂移 | env 合同、P0 硬门、TE cache、重叠审计、sync 白名单 | 日志可审计；profile ≠ KPI |
| **E 产品/serving 未来** | `E` | 多请求、SKU 分档、换底座 | VDN/SelfLift/TAE 分档；batching defer | 独立 SKU，不污染金线 |

**叠乘规则（机理文）：** `{B} × {A-attn或token取一} × {A-quant} × {D}` 可乘；`B↔有损cache`、`稀疏↔token缩减` 重叠递减；族 6 + 精确不变量缓存永远先做零风险。

**DeepInfer S → 轴：** S4→A；S7→B；S2/部分S1→C；S1/S5/S6/S8/S9/S11/S12→D；S3→E。

---

## 2. 管线地图（阶段 → 责任面）

```
load/weights → TE(+vision) → refine text_h → DiT schedule/blocks
  (attn SDPA + MLP quant paths) → sampler/fewstep → VAE video/audio
  → mux/ffmpeg → waffle/report
旁路：host staging · CUDA streams/graph · LoRA · te_cache · w4a8 window · NVFP4
```

| 阶段 | 关键文件 / 开关 | 主要轴 |
|---|---|---|
| Load / 权重窗 | `h3_weights.*`、`h3_w4a8_cache.*`、`h3_safetensors.*`、`--ssd-streaming` | C,D |
| TE / Vision | `h3_text_encoder.*`、`h3_vision_*`、`h3_te_cache.*`、`H3_TEXT_H_LOAD`、`--te-cache` | D,A(次) |
| DiT host / Metal | `h3_dit.*`、`h3_gpu.m`、`h3_shaders.metal` | A（Mac 臂） |
| DiT CUDA INT8 | `h3_dit_w4a8.*`、`h3_dit_int8*`、`h3_gpu_cuda.*`、`H3_DIT_EPI_FUSE` | A,D |
| DiT NVFP4 | `h3_dit_nvfp4.*`、`h3_nvfp4*`、`H3_DIT_NVFP4`（链式 WIP **勿动**） | A,C |
| Attn / SDPA | `h3_sdpa*`、`h3_gpu_cuda_sdpa_cudnn.cpp`、sol / cuDNN graph 政策 | A,D |
| Schedule / sampler | `h3_dit_schedule.*`、`h3_sampler.*`、`h3_fewstep_contract.*`、`--fewstep-preset` | B,D |
| LoRA | `h3_lora.*` | B |
| VAE | `h3_video_vae*`、`h3_audio_vae*`、`h3_vae_int8*`、`h3_vae_prewarm.*` | A,C,D |
| Mux | `h3_ffmpeg.*`（veryfast 已落地） | D（尾料） |
| 测量 | `h3_waffle.*`、`h3_run_report.*`、`tools/print_h3_env_contract.py`、`tools/p0_protocol_check_log.py`、timing 文档 | D |
| 多卡/Ulysses | `h3_ulysses*` | E（非默认） |

---

## 3. 已关闭 / 勿重开（Closed）

下列除非**新证据**（新硬件 ISA、新测量推翻噪声带、新产品合同），禁止再开主线诉讼。

### 3.1 算子 / 工程（[`deadend-audit-operators`](2026-10-07-deadend-audit-operators.md)）

| ID | 条目 | 裁决 | 备注 |
|---|---|---|---|
| Z01 | INT8 cublasLt autotune 无候选 | **维持关闭** | 256MiB WS、16 heuristic→1；可选 1GiB 封档实验不阻塞 |
| Z02 | 手写更快 sm_120 INT8 GEMM | **维持关闭** | 8 形状 730–742 TFLOP/s = 峰 98–99%；无 15% 余量 |
| Z03 | FFN epilogue **vec4** 路径 | **关闭该实现** | 「epilogue 减往返」整族曾被误禁；**E2 对撞融合仍开放**（见 C020） |
| Z04 | sol KV fuse | **维持关闭** | 访存饱和；窗口内 ≤0.09s |
| Z05 | 整步 CUDA Graph 翻默认 | **不翻默认** | 收益 <±0.5s 噪声；理论上限 ~0.3s；graph **对照默认关** |
| Z06 | VAE tile batch 4→8 | **维持 batch=4** | 稳定 −0.36s 回归 |
| Z07 | mux overlap | **关闭并降级档案** | 管道已流式；ultrafast 后 mux ~35ms |
| Z08 | E1 CUTLASS EVT @ sm_120 | **已证伪 / 关闭** | 见增量机会文 defer 条；勿再当 T1 主刀 |
| Z09 | E3 cuBLASLt native custom epilogue | **关闭** | 库不支持 swiglu/gate custom epi |
| Z10 | L3 Mirage megakernel | **N-A** | 负载形态相反 |
| Z11 | PDL 作主加速 | **低优先尾料** | 可挂自写短链，排融合后；非主线 |
| Z12 | DeepGEMM JIT / 多 shape plan-run | **已闭环（固定 shape）** | 热路径零决策是优势 |

### 3.2 注意力 / 质量（[`sol-attn-negative-audit`](2026-10-07-sol-attn-negative-audit.md)、[`deadend-audit-quality`](2026-10-07-deadend-audit-quality.md)、机理附录）

| ID | 条目 | 裁决 |
|---|---|---|
| Z20 | 「sol 全面死刑」过度泛化 | **纠正边界**：负面多来自合同不对等/噪声；sol 不是默认金线，也不是永久禁谈 |
| Z21 | MagCache / 有损特征 reuse @ 少步默认 | **主题禁** | turbo4 上跨步冗余已榨干（多方旁证） |
| Z22 | 推理期稀疏当 4-step 无损默认 | **高风险** | Hao/FastWan：<10 步蒸馏上训练无关稀疏易退化；需独立合同验证 |
| Z23 | Sage MMA / 多卡进默认 | **禁默认** | 观察臂另开 |
| Z24 | §12 cosine>0.99 / rel<0.05 当产品门 | **禁止** | 调试尺 ≠ 投产门 |
| Z25 | 2NFE 冲观感当 T3 通过 | **禁止** | PDMD-2 发糊是 σ 大跳固有 |

### 3.3 显存（[`deadend-audit-memory`](2026-10-07-deadend-audit-memory.md)）

| ID | 条目 | 裁决 |
|---|---|---|
| Z30 | 「768p 全 sol 在不减窗红线下永远不可行」 | **过度泛化 → 重开调查权** | 28.8GiB **无逐项实测**；减窗/FP16/streaming **从未评估**——见 C040+ |
| Z31 | torch 侧 drop+trim 两版释放救全 sol | **维持失败结论** | 但不得据此禁止一切显存手段 |
| Z32 | VAE restore「不可闭合」 | **收窄** | 在当前 INT8+FP16 权重策略下余尾 ~0.3s；手法可再探（C048） |

### 3.4 LLM 核盲搬（[`llm-to-dit-transfer`](2026-10-07-llm-to-dit-transfer.md)、DeepInfer 文）

| ID | 条目 | 裁决 |
|---|---|---|
| Z40 | FlashMLA / DeepEP / Wide EP / SWA replay | **N-A** |
| Z41 | MLA/GQA/MQA、PagedAttention、radix cache、Medusa | **N-A** |
| Z42 | FA4 本体（需 tcgen05/TMEM） | **N-A @ sm_120** |
| Z43 | 投机解码进默认路径 | **观察项，不进施工主序列** |
| Z44 | SM100 峰值数字当 5090 路线图 | **禁止** |
| Z45 | 「LLM ⇒ 策略也无价值」 | **错误框架** | 策略可学，核勿盲搬 |

### 3.5 已落地、勿当「缺口」重开

| ID | 条目 | 状态 |
|---|---|---|
| Z50 | VAE unpatch D2H pinned+async | **默认开，已收割**（增量文明确） |
| Z51 | x264 veryfast（T2）/ ultrafast（T1 预览） | **已落地** |
| Z52 | AdaLN conditioning 预计算 | **双路径已收割** |
| Z53 | TE 运行时/跨 clip 缓存；磁盘 te_cache opt-in | **已有**；缺的是产品化默认策略（C012） |
| Z54 | SDPA cuDNN graph 默认关（sm_120 可复现） | **已落地 `321538a`** |
| Z55 | 少步合同自检 / preset / `lcm_renoise` / host 测 / σ 金表 | **开发已齐**；差 GPU 视觉 |
| Z56 | P0 runbook、日志硬门、env 合同打印工具、waffle 同配置门 | **离线工具已齐** |
| Z57 | 权重 H2D 与计算 overlap（窗侧） | **已做**；L2 pin 未做（C046） |
| Z58 | INT8 作为默认生产路径近峰 | **事实**；勿再开「更快 INT8 GEMM」主线 |

---

## 4. 全量机会目录

**字段说明**

- **杠杆：** `WC` 墙钟 · `VRAM` 显存/能力 · `Q` 质量/可复现 · `R` 风险（L/M/H）  
- **状态：** `idea` / `partial`（代码或设计在） / `landed` / `blocked`（等人/盘/视觉门） / `closed`（见 §3）  
- **量级：** 凡写秒数/倍数均来自既有文档；**未在 238 按现基线重测前不得进 KPI 宣传**。

### 4.1 轴 A — 每步算力

| ID | 阶段 | 问题 | 机制 | 证据 | 杠杆 | 前置 | 状态 | S链 | 下一步实验 |
|---|---|---|---|---|---|---|---|---|---|
| **C001** | DiT quant | 同精度 GEMM 已近峰，算力墙在精度档 | NVFP4 W4A4 实验臂；视觉门投产 | `nvfp4-w4a4-path`；roadmap P0；timing graph-关 ~1.14× dit 中位；[`gpu-unified-test-results`](2026-10-27-gpu-unified-test-results.md) | WC高 VRAM中 Q高 R高 | INT8 金线；graph关；chain关 | **partial**（238 PROTO_OK；INT8 waffle PASS；NVFP4 waffle FAIL；**目检未做**） | S4 | 人眼目检 `out/p0_visual_20261027_from238/` |
| **C002** | DiT NVFP4 | 链式消除 FC1→FC2 激活中转 | CUTLASS 79b FP4-out+SFD | `nvfp4-chained-*`；WIP 未提交 | WC中 Q高 R高 | **C001 视觉结论 + 用户确认** | **blocked**（WIP 不动） | S4/S5 | 视觉门后解释 latent rel~0.33 再议提交 |
| **C003** | DiT INT8 链路 | quant/dequant 中转 ~1.465s/clip | E2 dequant↔requant 对撞融合 | `e2-epi-fuse/DESIGN.md`；`H3_DIT_EPI_FUSE`；[`c003-epi-fuse-nan-fix`](2026-10-27-c003-epi-fuse-nan-fix.md) @ `1b37f28` | WC中（实测 hot ~3% dit） Q中 R中 | 授权 238；act_f16+convrot | **partial**（NaN 已修；fuse waffle FAIL；**默认仍 OFF**） | S5 | 残差数值/目检；勿 default-on；C004 仍等质量门 |
| **C004** | DiT INT8 | E2-P3 跨 phase FC2→QKV | mode 3 requant 接线 | DESIGN.md §5 | WC中 R中高 | C003 数值门 | **idea** | S5 | A1 通过后再开 |
| **C005** | DiT | Hadamard/ConvRot 与 FP4 共设计 | 激活旋转抑 outlier | transfer 2.2；nvfp4 文 | Q中（绑 C001） | 随 NVFP4 | **partial** | S4 | 不单独开项 |
| **C006** | Attn | cuDNN SDPA / sol 后端选择与稳态 | registry；graph 政策 | `h3_sdpa*`；`321538a` | WC低~中 Q高 | 对照纪律 | **landed/政策** | S6 | 保持 graph 默认关；临时开写标题 |
| **C007** | Attn | sol warmup/autotune 是否 per-clip ~1.05s | 持久化 (T,shape,GPU) | sm120 L1；sol-attn R3 | WC中（cold） | 先证实付费频率 | **idea** | S6 | 冷热日志对照；已是 cold-only 则关闭项 |
| **C008** | Attn | SLA 式稀疏+线性补偿叠 sol | 训/推混合；保运动 | video-dit-attention-frontier §7 | WC中~高 Q高 R高 | 独立 SKU；非默认金线 | **idea** | — | 仅观察/研究臂；勿与 C001 交叉 |
| **C009** | Attn | SageAttention3 FP4 opt-in | 注意力段低精度 | frontier §5.1 | WC中 Q高 | sm_120 验证 | **idea** | S4 | 1–2 天探针，独立臂 |
| **C010** | Attn | sol MMDiT dense 护栏（文本行/前层 dense） | 降稀疏质量风险 | accel 附录 | Q中 | sol 臂 | **idea** | — | A/B 质量 |
| **C011** | VAE | cast/prescale ~2.29s 中可融余量 | 融合/消除往返 | quality-preserving；sm120 | WC中 | nsys 授权 | **idea** | S5 | 剖析后再改 |
| **C012** | VAE | VAE 段 SIMT sgemm / TF32 衍生 | 算力路径升级 | sm120「VAE 族未施工」 | WC中 Q中（TF32） | nsys | **idea** | S5 | T1 允许 TF32；T2 慎 |
| **C013** | VAE | VAE INT8 路径深化 | 与 DiT INT8 同族 | `h3_vae_int8*` | WC低~中 | 质量门 | **partial** | S4 | 不挡 DiT 主线 |
| **C014** | DiT Metal | Mac Metal 核/融合余量 | shaders / 调度 | `h3_shaders.metal`、`h3_gpu.m` | WC（Mac） | 与 CUDA 臂隔离 | **idea** | — | 单独 Mac 账本；本文默认 CUDA 5090 |
| **C015** | DiT | 激活 FP16 中转降带宽 | d_wide/d_qkv 存 FP16 | sm120 M2 | WC低 VRAM高 Q中 | C040 账本 | **idea** | S2 | 768p 备胎 |
| **C016** | DiT | PDL 挂 GEMM→dequant 短链 | 消 launch 气泡 | transfer 3.2；sm120 L2 | WC低（−0.1~−0.3） | 融合后；自写核 | **idea** | S5/S8 | 排 C003 后 |
| **C017** | DiT | L2 accessPolicyWindow pin 小热表 | ConvRot/RoPE/mod/s_channel | sm120 M3 | WC低 bit同 | 授权小改 | **idea**（本片跳过） | S2 | 与 C003 可并行；需 profile |
| **C018** | DiT | QKV merge（3→1 宽 GEMM） | 历史 gap list | quality-preserving gap | WC中 | 验证是否已被现路径覆盖 | **idea/可能过时** | S5 | 先 diff 现状再决定；勿假设仍缺 |
| **C019** | DiT | QKNorm+RoPE 融合 | gap list 附属 | quality-preserving | WC低 | — | **idea** | S5 | profile 后 |
| **C020** | — | （占位）E2 即 C003 | — | — | — | — | — | — | 与 A1 同 |

### 4.2 轴 B — 步数 / 合同

| ID | 阶段 | 问题 | 机制 | 证据 | 杠杆 | 前置 | 状态 | S链 | 下一步 |
|---|---|---|---|---|---|---|---|---|---|
| **C030** | fewstep | T3 最短路径未 GPU 验 | PDMD-4NFE 原件+Euler+12/3 | pdmd-dmad-loras；`--fewstep-preset pdmd4`；gpu-unified-results | WC高（少步） Q高 | INT8；合同开 | **partial**（238 PROTO+waffle PASS @ rank128 原件；**目检未做**） | S7 | 人眼目检 `out/c030_pdmd4_20261027_from238/` |
| **C031** | fewstep | DMAD 官方采样器需诚实验 | `lcm_renoise`+full_critic+12/2 | sampler `e037d42`；gpu-unified addendum；LoRA `dmad_minimax_h3_4step_full_critic.h3repo.safetensors` | WC高 Q高 | C030 纪律 | **partial**（238 PROTO+waffle PASS；`lcm_renoise` 已确认；**目检未做**） | S7 | 人眼目检 `out/c031_dmad4_20261027_from238/` |
| **C032** | fewstep | 错合同污染加速叙事 | contract FAIL / preset / σ 金表 | fewstep_contract；fixtures | Q高 R低 | — | **landed**（host） | S11 | GPU 跑时强制 `--fewstep-contract` |
| **C033** | fewstep | PDMD-2 仅作合同回归 | 大 σ 跳固有发糊 | pdmd 文 r2–r4 | — | — | **landed/边界** | — | 不作 T3 通过尺 |
| **C034** | fewstep | Metal CLI 无 LoRA/sampler | 边界说明 | fewstep-metal-boundary | — | — | **landed（文档）** | — | 验收走 `h3_cuda_generate` |
| **C035** | 蒸馏质量 | turbo4 过饱和/糊 | taomate 0.7 refiner / 细节 LoRA | accel 附录 | Q中 | 独立臂 | **idea** | S7/族8 | 低成本 A/B；勿进默认 |
| **C036** | 求解器 | 高阶求解器少 NFE（非蒸馏） | 族1 solver | accel 机理 | WC中 Q中 | 金线步数政策 | **idea** | — | 观察；与蒸馏正交 |
| **C037** | 投机扩散 | ASDSV 类 | 少跑完整 forward | transfer 4.4 | WC? R高 | draft 权重无 | **idea/观察** | S7 | **不进主序列** |

### 4.3 轴 C — 显存 / 能力

| ID | 阶段 | 问题 | 机制 | 证据 | 杠杆 | 前置 | 状态 | S链 | 下一步 |
|---|---|---|---|---|---|---|---|---|---|
| **C040** | DiT@768p | 无逐项显存账本 | 关键节点 `cudaMemGetInfo` 分桶 | deadend-memory 条目1；A2；`H3_VRAM_LEDGER` | VRAM决策前提 | 授权一跑；勿 nsys 若盘满 | **partial**（插桩 landed；238 未跑） | S2 | 238：开 ledger 跑 768p 分桶 |
| **C041** | DiT | 窗 30→24 探针 | 换 ~GiB 看全 sol | memory 审计 ② | VRAM中 WC可能负 | C040 | **idea** | S2 | 一次性实验，不解默认 |
| **C042** | DiT | 激活 FP16 中转 | 同 C015 | M2 | VRAM高 | C040 | **idea** | S2 | waffle 非 bit-identical |
| **C043** | DiT | Layerwise weight streaming | 权重按层驻留+prefetch | transfer 4.3；SGLang −30GB 指针 | VRAM高 WC中性~负 | C040；视觉后 | **idea** | S1/S2 | 768p 全序列能力优先 |
| **C044** | DiT | NVFP4 权重体积治本 | 与 C001 同刀显存维 | sm120 M1 | VRAM高 | C001 | **blocked** | S4 | 视觉门绑定 |
| **C045** | DiT | SSD BF16 streaming（已有开关） | 两 slot overlap 读盘 | `ssd_streaming`；`h3_dit.c` | VRAM（Mac/BF16） | 与 INT8 互斥 | **landed** | S2 | CUDA INT8 窗是另一故事 |
| **C046** | DiT | L2 pin（同 C017） | — | — | — | — | — | — | — |
| **C047** | VAE | VAE 与 DiT 峰值错峰 | DeferredRelease / 预热策略 | te-lifecycle-audit | VRAM中 | 审计→真机 | **partial** | S1 | PDMD 预热 OOM 清单 |
| **C048** | VAE | restore 指针重绑 / FP8 缓存 | 消 ~0.3s 余尾 | memory 条目3；PR#36727 | WC低 VRAM敏感 | 不与 INT8 窗恶斗 | **idea** | S1/S2 | P3 |
| **C049** | 产品能力 | VDN 恒定内存 @768p | 换注意力合同 | vdn-direct-use | VRAM/能力 | 独立 SKU | **idea** | E/S7 | 长片子产品档 |

### 4.4 轴 D — 工程 / 测量 / 生命周期

| ID | 阶段 | 问题 | 机制 | 证据 | 杠杆 | 前置 | 状态 | S链 | 下一步 |
|---|---|---|---|---|---|---|---|---|---|
| **C060** | 对照 | TE 路径漂移污染 md5/waffle | 强制标明 live vs `H3_TEXT_H_LOAD` | P0 runbook；timing | Q高 | — | **partial** | S9 | 报告硬字段 |
| **C061** | 对照 | env 合同只在离线工具 | generate 启动内联打印 | `h3_env_contract_*`；print_h3_env_contract；A10 | Q（防测错） | — | **partial**（内联 landed；GPU 日志未验） | S9/S11 | GPU 跑强制过 `# H3_ENV_CONTRACT` |
| **C062** | TE | TE 7–58s 摆动污染加速比 | te_cache 产品化 + 常驻 + HIT 硬门 | timing F1/F2；h3_te_cache；productization；gpu-unified addendum HIT smoke | WC测准 / 次要墙钟 | key 含 layers/dim/ids | **partial**（opt-in landed；238 HIT+REQUIRE_HIT **PROTO_OK**；默认仍 off） | S1/S9/S12 | 默认策略保持 off；多 prompt 稳健后再议默认 |
| **C063** | TE | `c_te_raw` 偶发不确定 | 根治债 | followup；dev-plan | Q | 不插队 | **idea** | S9 | 单独 ticket |
| **C064** | 生命周期 | TE↔DiT↔VAE 峰值与释放 | 状态机 + DeferredRelease | te-lifecycle-audit；[whitelist](2026-10-27-lifecycle-deferred-release.md)；`h3_lifecycle_whitelist.h` | VRAM/WC | 真机峰值 | **partial**（白名单+trace landed；真重叠仍门控） | S1 | 安全条件满足再重叠 H2D |
| **C065** | 测量 | 墙钟卫生 | 关 `H3_GEN_MEM_TRACE`；只追加 timing | generation-timing | Q | — | **landed（纪律）** | S9 | 持续执行 |
| **C066** | 测量 | P0 日志协议 | `p0_protocol_check_log.py` | tools | Q | — | **landed** | S9 | GPU 日志强制过检 |
| **C067** | 测量 | waffle 同配置门 | 禁跨臂 | waffle-same-config-gate | Q | — | **landed** | S9 | — |
| **C068** | 不变量 | INT8 路径 RoPE 表每 forward 现算 | per-shape memo | invariant-cache-audit；[rope note](2026-10-27-c068-c069-rope-memo.md) | WC低（估0.1–0.3） bit同 | profile 确认调用点 | **landed**（host memo；GPU WC 未验） | 族4 | GPU 测墙钟 |
| **C069** | 不变量 | VAE RoPE 每 chunk 现算+H2D | host memo + same-key H2D skip | 同上；[rope note](2026-10-27-c068-c069-rope-memo.md) | WC低 bit同 | — | **partial**（host memo+H2D skip；clip 级常驻 defer） | 族4 | GPU 测；全 geometry 常驻 defer |
| **C070** | sync | 热路径隐蔽 sync | 白名单 / `H3_FORBID_HOT_SYNC` | [`cuda-sync-audit`](2026-10-27-cuda-sync-audit.md)；S8 | WC低 | forbid 开关未做 | **partial**（审计页 landed） | S8 | 不挡主线；forbid 另开 |
| **C071** | cold | cold-clip 间歇高噪 | 方向 B 根治 | roadmap T3-c；A11 | Q | 高不确定 | **idea** | S9 | 不插队 |
| **C072** | 配置 | 散落 getenv | 集中 `h3_env` 注册表 | S11；A15 | Q | 大重构 | **idea** | S11 | 远期观察 |
| **C073** | 生成/评分 | 边改配置边改金线 | dump-once / score-many | S12；META+log；[conventions](2026-10-27-dump-once-score-many.md) | Q | — | **partial**（约定+日志标记+checker landed） | S12 | GPU 上跑 dump→replay 验收 |
| **C074** | 失败分类 | OOM/合同/数值/环境混报 | 退出码/日志 `tag=FAIL_*` | DeepInfer environ 族；与 C061 同批 | Q | — | **partial**（generate 站点 tagged；VAE 专用 tag 无独立站点） | S11 | GPU 日志抽检 |
| **C075** | host | Qwen prefetch 深度 | `H3_QWEN_PREFETCH*` | text_encoder | WC低 | — | **partial** | S1 | 调参级 |
| **C076** | DiT | 大量 `H3_DISABLE_FUSED_*` 回退开关 | 回归与 A/B | h3_dit.c env 列表 | D | — | **landed** | S6 | 保持；勿默认关融合 |
| **C077** | SDPA | graph pool/clean/destroy 纪律 | 防挂 | deeper-pass；cudnn sdpa | Q | 若再开 graph | **partial** | S6 | 政策备忘 |

### 4.5 轴 E — 产品 / serving 未来

| ID | 阶段 | 问题 | 机制 | 证据 | 杠杆 | 前置 | 状态 | S链 | 下一步 |
|---|---|---|---|---|---|---|---|---|---|
| **C080** | SKU | 同合同加速 vs 产品分档混淆 | VDN / SelfLift / TAE 独立 SKU | A12；vdn 文；selflift/tae | 产品 | 金线隔离 | **idea** | S7 | 观察；合同表 |
| **C081** | serving | 多请求 continuous batch | 打包喂满 GPU | DeepInfer S3；transfer 4.1 N-A 现状 | 吞吐 | 产品形态变更 | **defer** | S3 | CLI 单请求前不做 |
| **C082** | 多卡 | Ulysses / 多卡 | 能力轴 | `h3_ulysses*` | 能力 | 非默认 | **partial/非默认** | 族7 | 禁默认 |
| **C083** | 预览 | SelfLift/LBH/TAE 快速预览 | 降合同换速度 | h3_selflift；h3_tae | WC预览 | 非金线 opt-in | **partial** | 族8 | 文档标非金线 |
| **C084** | W4A8 主题 | 以 W4A8 为默认叙事 | — | followup 否定 | — | — | **closed主题** | — | 代码路径可留，不作主题默认 |

### 4.6 管线序补充（易漏阶段）

| ID | 阶段 | 问题 | 机制 | 证据 | 杠杆 | 状态 | 下一步 |
|---|---|---|---|---|---|---|---|
| **C090** | Load | 权重窗与上传可观测性 | upload_ms / HIT 日志 | te-lifecycle；timing | D | partial | 与 C062 合并产品化 |
| **C091** | Vision | VISION span 跳过 te_cache | 正确但需合同打印 | te_cache 注释 | Q | landed边界 | env 合同写明 |
| **C092** | Audio VAE | 与 video 峰值叠加 | 错峰 / skip-audio 对照 | P0 `--skip-audio` | VRAM/对照 | partial | 视觉门默认 skip-audio |
| **C093** | Mux | 已优化尾料 | veryfast/ultrafast | Z51 | — | landed | 勿再开 overlap |
| **C094** | Report | run_report / KPI guard | 门禁自动化 | h3_run_report；kpi_guard | D | partial | 接 P0 字段 |
| **C095** | Token reduction | `H3_TOKEN_REDUCTION*` | 族3 | h3_dit.c | WC/Q | partial/实验 | 独立质量合同 |
| **C096** | Step cache | `h3_step_cache` | 有损风险 | 禁默认 | — | 存在即观察 | 默认关 |
| **C097** | Latent upscaler | 后处理能力 | h3_latent_upscaler* | 产品 | idea/partial | 非 DiT 主墙 |
| **C098** | AV sync | 音画同步 | h3_av_sync* | Q | partial | 回归集 |
| **C099** | 多模态条件 | vision condition | multimodal* | 产品 | partial | 与 TE 合同一起 |

---

## 5. 旧文档交叉索引（去重）

| 旧 ID / 主题 | 本目录 |
|---|---|
| Roadmap P0 NVFP4 视觉门 | C001 |
| Roadmap P1 PDMD-4 / 耗时 | C030 / C065 |
| Roadmap P2 DMAD / 重叠 / 融合 | C031 / C064 / C003 |
| Roadmap P3 链式 / streaming | C002 / C043 |
| A1 E2 epi-fuse | C003 |
| A2 显存账本 | C040 |
| A3 TE cache 产品化 | C062 |
| A4 激活 FP16 | C015/C042 |
| A5 sol autotune | C007 |
| A6 L2 pin | C017 |
| A7 VAE cast/sgemm | C011/C012 |
| A8 E2-P3 | C004 |
| A9 窗 30→24 | C041 |
| A10 env 内联 | C061 |
| A11 cold 高噪 | C071 |
| A12 VDN/SelfLift/TAE | C080/C049/C083 |
| A13 VAE restore | C048 |
| A14 PDL | C016 |
| A15 h3_env | C072 |
| sm120 E1 | Z08 closed |
| sm120 E2/L1/M2/M3 | C003/C007/C015/C017 |
| DeepInfer S1–S12 | §1 映射；条目 S链列 |
| invariant RoPE gaps | C068/C069 |
| transfer NVFP4/streaming/PDL/Hadamard | C001/C043/C016/C005 |
| quality gap VAE unpatch / QKV merge | Z50 / C018 |
| fewstep 开发落地 | C032–C034 landed |

---

## 6. 近 2–8 周 backlog（P0–P3）

服从 followup：**不打断 INT8→NVFP4 视觉门主线**；下列「∥」可并行。

### P0（质量地基 + 主算力门）

1. **C001** NVFP4 P2.2 视觉门（238 授权）— 主杠杆  
2. **C060/C066/C065** 对照纪律执行（每次跑）  
3. **C061** generate 内联 env 合同 — 无 GPU 可做  
4. **C092** 视觉门默认 `--skip-audio` 写进报告  

### P1（账本 + 少步 + 同精度秒级）

5. **C030** ∥ PDMD-4NFE GPU  
6. **耗时追加** timing 文档（不挡视觉）— 绑 C065  
7. **C003** ∥ E2 真机 A/B（授权）  
8. **C062** ∥ TE cache 产品化设计/落地  

### P2

9. **C031** DMAD full_critic GPU  
10. **C040** 768p 显存账本插桩  
11. **C064** 重叠安全实现（审计已有）  
12. **C007** sol autotune 付费频率证实  
13. **C011/C012** nsys 后 VAE/cast（盘够+授权）  

### P3（门控）

14. **C002** 链式 — 仅 C001 结论 + 用户确认  
15. **C041→C042→C043** 减窗探针 → FP16 → streaming  
16. **C004** E2-P3  
17. **C017/C068/C069/C016/C070** 小刀与尾料  
18. **C063/C071** 生产债不插队  

### 明确本窗口不做

- Z* 关闭项诉讼；FlashMLA 等盲搬；链式 WIP 提交；清盘/nsys 未授权；W4A8 主题默认；2NFE 当通过。

---

## 7. 更长地平线附录（月级+）

| 主题 | ID | 说明 |
|---|---|---|
| NVFP4 升默认 / 并入 golden | C001 后产品决策 | 需视觉+多 prompt 稳健 |
| 链式拓宽（非仅 FC1→FC2） | C002 Phase2 | 基线稳后 |
| SLA/VDN 级注意力合同 | C008/C049 | 换模型或稀疏感知微调；独立 SKU |
| 多请求 serving | C081 | 产品形态变了再搬 S3 |
| 集中 env 注册表 | C072 | 卫生大重构 |
| Metal 性能账本 | C014 | CUDA 主线外 |
| 扩散投机 / 高阶求解器 | C037/C036 | 研究观察 |
| 细节恢复 LoRA / taomate refiner | C035 | 质量补丁臂 |
| FA3 式 warp spec @768p dense | transfer 3.3 | 随 sol 演进，非新仓 |
| Ulysses 多卡产品化 | C082 | 能力，非默认速度故事 |

---

## 8. CSV 附录（机读）

```csv
id,axis,stage,title,status,s_link,near_term_pri
C001,A,DiT-NVFP4,NVFP4 visual gate P2.2,partial,S4,P0
C002,A,DiT-NVFP4,NVFP4 FC1-FC2 chain,blocked,S4/S5,P3-gated
C003,A,DiT-INT8,E2 epi-fuse A/B,partial-nan-fixed-waffle-fail,S5,P1
C004,A,DiT-INT8,E2-P3 FC2-to-QKV,idea,S5,P3
C007,A,Attn,sol autotune persistence,idea,S6,P2
C011,A,VAE,cast/prescale fusion remainder,idea,S5,P2
C015,A/C,DiT,act FP16 staging,idea,S2,P3
C017,A,DiT,L2 pin hot tables,idea,S2,P3
C030,B,fewstep,PDMD-4NFE GPU accept,partial,S7,P1
C031,B,fewstep,DMAD lcm_renoise GPU,partial-proto-pass,S7,P2
C032,B,fewstep,contract selfcheck,landed,S11,done-dev
C040,C,768p,VRAM ledger instrumentation,partial,S2,P2
C041,C,DiT,window 30to24 probe,idea,S2,P3
C043,C,DiT,layerwise streaming,idea,S1/S2,P3
C061,D,tooling,inline env contract at generate,partial,S9/S11,P0-dev
C062,D,TE,te_cache productization,partial-hit-verified,S1/S9,P1
C064,D,lifecycle,TE-DiT-VAE overlap safe impl,partial,S1,P2
C073,D,eval,dump-once score-many conventions,partial,S12,P2
C068,D,DiT,INT8 RoPE table memo,landed,fam4,P3
C069,D,VAE,VAE RoPE memo+H2D skip,partial,fam4,P3
C080,E,SKU,VDN/SelfLift/TAE tiering,idea,S7,observe
C081,E,serving,continuous batching,defer,S3,defer
```

（完整字段以 §4 表为准；CSV 为排程摘要。）

---

## 9. 相关文档（权威链）

| 文档 | 角色 |
|---|---|
| [`2026-10-26-followup-dev-plan.md`](2026-10-26-followup-dev-plan.md) | **执行 checklist 权威** |
| [`2026-10-26-optimization-roadmap.md`](2026-10-26-optimization-roadmap.md) | 四周优化方案；应链到本文 |
| [`2026-10-27-additional-optimization-opportunities.md`](2026-10-27-additional-optimization-opportunities.md) | A1–A15 增量扫掠 |
| [`2026-10-26-deepinfer-speed-strategies.md`](2026-10-26-deepinfer-speed-strategies.md) | S1–S12 |
| [`2026-10-27-deepinfer-thinking-to-h3c.md`](2026-10-27-deepinfer-thinking-to-h3c.md) | DeepInfer **思维模式** T01–T14 → 文件/开关/Cxxx（非拒绝清单） |
| [`2026-10-27-c061-inline-env-contract.md`](2026-10-27-c061-inline-env-contract.md) | C061 内联 env 合同落地说明 |
| [`2026-10-27-cuda-sync-audit.md`](2026-10-27-cuda-sync-audit.md) | C070 sync 审计 |
| [`2026-10-05-acceleration-mechanisms.md`](2026-10-05-acceleration-mechanisms.md) | 族 1–8 |
| [`2026-10-07-sm120-operator-frontier.md`](2026-10-07-sm120-operator-frontier.md) | E/L/M 原始方案 |
| [`2026-10-07-deadend-audit-*.md`](2026-10-07-deadend-audit-operators.md) | 关闭项证据 |
| [`2026-10-07-llm-to-dit-transfer.md`](2026-10-07-llm-to-dit-transfer.md) | 同构裁决 |
| [`2026-10-09-dev-plan-and-progress.md`](2026-10-09-dev-plan-and-progress.md) | 总目标上游 |
| [`out/review-2026-10-07/e2-epi-fuse/DESIGN.md`](../../out/review-2026-10-07/e2-epi-fuse/DESIGN.md) | E2 施工 |

---

## 10. 修订记录

| 日期 | 说明 |
|---|---|
| 2026-10-27 | 238 GPU：C003 NaN 修 @ `1b37f28`（waffle 仍 FAIL，默认 OFF）；C001 PROTO；C030/C031 PROTO+waffle PASS；C062 HIT 硬门 PROTO_OK；均待目检（C062 除外）。见 [`gpu-unified-test-results`](2026-10-27-gpu-unified-test-results.md)。 |
| 2026-10-27 | 离线落地 C061/C074/C040 插桩 + C070 审计页；状态 idea→partial（无 GPU）。 |
| 2026-10-27 | 链到 [`deepinfer-thinking-to-h3c`](2026-10-27-deepinfer-thinking-to-h3c.md)：T01–T14 思维深挖；不改 C 表优先级。 |
| 2026-10-27 | 初版：合并 research 目录 + 管线扫掠；五轴 taxonomy；Closed Z*；机会 C001–C099（含占位/交叉）；2–8 周 backlog + 长地平线；CSV 摘要。不跑 GPU；不碰链式 WIP。 |

---

## 11. 统计（本稿）

| 集合 | 约数 |
|---|---|
| 开放/进行中机会（非 closed、非纯 landed 尾） | **~55** 主动条目（C001–C099 中扣除关闭与纯文档边界） |
| 明确关闭 Z* | **~30** |
| 已落地工具/政策 landed | **~15** |
| 近 2–8 周排序项（§6） | **18** 步（含并行） |

*「主动条目」含 blocked/partial/idea；landed 纪律项仍列以便审计。*
