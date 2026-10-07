# h3.c vs SGLang Diffusion vs vLLM-Omni：MiniMax-H3 位置对照

> **后续开发方案见** [2026-08-26-h3c-next-tech-plan.md](./2026-08-26-h3c-next-tech-plan.md)

日期：2026-08-25  
仓库：`h3.c`（纯 C/CUDA MiniMax-**H3 视频+音频**推理；**不是** MiniMax-M1/M2 LLM）  
本文只做对比分析与建议；不改 generate、不 commit、不写密码。

> **深度技术版（机制 / 误差预算 / Amdahl / 判据函数）：**  
> [2026-08-25-h3c-vs-sglang-vllm-tech-deepdive.md](./2026-08-25-h3c-vs-sglang-vllm-tech-deepdive.md)  
> 若反馈「太粗浅」，请读深度版：运行时栈、注意力近似来源、Tea/Mag/FB/Cache-DiT 探测信号差异、8-step 可跳天花板、INT8 与 cache 误差叠加、并行切分点、质量门口径。本文保留产品定位与公开秒数禁横比表。

**对照底稿（冲突以实测报告 + 公开 cookbook 原文为准）：**

- [2026-08-25-h3c-accel-backends-cache-matrix.md](./2026-08-25-h3c-accel-backends-cache-matrix.md)
- [2026-08-25-h3c-4090-accel-plan.md](./2026-08-25-h3c-4090-accel-plan.md)
- [2026-08-25-sla-vs-h3c.md](./2026-08-25-sla-vs-h3c.md)
- [2026-08-25-h3c-multi-gpu-arch-speed.md](./2026-08-25-h3c-multi-gpu-arch-speed.md)
- [out/verify_5090_bench/REPORT.md](../../../out/verify_5090_bench/REPORT.md)
- [out/verify_sol_vis_gate/REPORT.md](../../../out/verify_sol_vis_gate/REPORT.md)
- [docs/nvidia_build.md](../../nvidia_build.md)

**公开来源（WebFetch / cookbook，禁止编造）：**

- SGLang Diffusion cookbook：[MiniMax-H3](https://lmsysorg.mintlify.app/cookbook/diffusion/MiniMax/MiniMax-H3)
- vLLM-Omni recipe：[recipes/MiniMaxAI/MiniMax-H3.md](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md)
- 官方 Sol-Engine 4.44× / 5090 整栈：[NVlabs/Sana sol-engine](https://github.com/NVlabs/Sana/blob/sol-engine/models/minimax_h3/README.md)、[H3-OnDevice](https://nvlabs.github.io/Sana/Sol-Engine/H3-OnDevice/)
- HF 模型卡推荐框架：[MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)

---

## 0. 结论先行

1. **产品位不同，不能当「谁快谁慢」的同赛道排名。**  
   - **h3.c**：原生 C/CUDA **单机推理引擎**，目标是消费级单卡（4090/5090）上 **8-step INT8 + cuDNN** 的**质量金线**成片。  
   - **SGLang Diffusion / vLLM-Omni**：Python **serving 框架**，目标是 HTTP 多任务、多卡拓扑、BF16/FP8 官方 50-step 768p 配方 + 可插拔 cache / 并行。

2. **本仓已钉死的质量地板：** cafe 有声 T2VA，864×480，真 5s / 120@24fps，cfg=1，**8-step INT8 ConvRot + fp16 VAE tile+deblock**，SDPA=`cudnn`：  
   - **4090 ~75.4s**（q1）；gate warm ~78.6s  
   - **5090 warm ~66s**（TE 稳档）  
   Tier-1 `sol_cute` / `sage3` **DiT 真快 14–15%，视觉 MAE 不过门** → 不当默认。

3. **框架公开 e2e 几乎全是另一把尺子：** 多数是 **50-step、1344×768、BF16/FP8、多卡 Ulysses**；Cache-DiT / TeaCache 的 × 也是相对**同配方 lossless 50-step**。  
   **禁止**把「本仓 75s」和「SGLang 4×H200 ~74s / vLLM high 63s / Sol-Engine 4.44×」说成「我们慢/快 X 倍」。

4. **本仓相对框架的优势：** 依赖薄、单卡可跑真 5s 有声、质量门（PPM MAE）严、INT8 路径可控、无 Python serving 栈。  
   **短板：** 无 HTTP serving、无 Ulysses/多卡、Cache-DiT / 独立 TeaCache / FBCache 未齐、Ref2VA 全组合与官方 cookbook 深度不对齐、LoRA Turbo 未接。

5. **若要对齐「serving 级 cache 体验」：优先缺口是独立 TeaCache（+ AV veto）与 Cache-DiT 类块级 skip 的可测实现**——且必须在 **8-step 金线上**重标定；官方 50-step thresh **不可照抄**。注意力臂继续抠质量，过门前不改默认。

---

## 1. 产品定位

| 维 | **h3.c** | **SGLang Diffusion** | **vLLM-Omni** |
| --- | --- | --- | --- |
| 形态 | 原生 **C/CUDA** CLI / 库（`h3_cuda_generate`） | Python **`sglang serve`** multimodal_gen | Python **`vllm serve --omni`** / OpenAI-compat `/v1/videos` |
| 目标场景 | **边缘 / 单卡** 可控成片；质量默认金线 | **集群 / 多卡 serving**；cookbook 覆盖消费卡 offload 到 B300 | **同左**；强调 combined FL2VA+Ref2VA 一服务、请求级 `quality` |
| 权重默认故事 | **INT8 ConvRot** 蒸馏 8-step | **BF16/FP32** 官方 50-step；Online FP8 / kitchen_int8 / GGUF 等可选 | **BF16** 50-step 参考；Turbo LoRA 另开；Cache-DiT `high` |
| 部署复杂度 | 构建 CUDA 旁路 + 权重路径；Sage3/Sol 插件 opt-in | Docker/venv、Triton、并行拓扑、layerwise offload、大主机 RAM | 同左 + Omni 管道、可选 FA4 / TRTLLM_ATTN |
| 架构红线 | **禁止**把 Python/Comfy 当质量默认生成栈 | Python 即主路径 | Python 即主路径 |

**一句话定位：** h3.c 是「把 H3 压进单卡、守住无损金线」的引擎；SGLang/vLLM 是「把 H3 做成可服务 API」的框架。HF 卡把后两者列为推荐部署框架；本仓是另一条工程线。

---

## 2. 模型能力覆盖（MiniMax-H3 视频，非 LLM）

官方能力族（HF / cookbook）：**T2VA / FL2VA（含首末帧）/ Ref2VA**；时长 4–15s @24fps；有声同步输出。权重分 **FL2VA** 与 **Ref2VA** 两分区。

| 能力 | h3.c（代码/计划事实） | SGLang cookbook | vLLM-Omni recipe |
| --- | --- | --- | --- |
| **T2VA 有声** | **主金线**；默认 audio pack | ✅ `task=t2va`，FL2VA 分区 | ✅ `task=t2va`，FL2VA |
| **F2VA / 首帧** | ✅ `--first-frame` 等条件图路径；vision 仍有 host SDPA 成本 | ✅ FL2VA | ✅ FL2VA |
| **L2VA / 末帧** | ✅ `--last-frame`；质量门另有 FL 末帧 MAE 叙事 | ✅ 尾关键帧形式 | ✅ `frame_indices` 含 `[-1]` / `[0,-1]` |
| **FL2VA 双帧** | 有首+末接线能力（lite / multimodal） | ✅ | ✅ |
| **Ref2VA** | 有 ref-video / picture 条件行路径；**深度与官方 9 图 3 视频 3 音频组合不完全对齐 serving 合同** | ✅ 独立 `model-variant ref2va` | ✅ combined 或 Ref2VA-only |
| **有声** | 默认；EasyCache 有声高风险（Comfy #15326） | 同步 AV；Sage 需检视音画 | 同步 AAC；TeaCache 仅标定 FL2VA |
| **时长** | 金线 `--seconds 5`（120 帧）；禁止 `--frames` 假时长 | 4–15s，默认验证档 5s @768 short edge | `duration` / 帧数网格；参考 workload 常 50-step 124/209 帧 |

---

## 3. 默认质量路径 vs 加速路径

### 3.1 h3.c

| 层 | 内容 |
| --- | --- |
| **Tier-0 质量默认** | INT8 ConvRot DiT + fp16 VAE tile+deblock 0.28；cfg=1；**steps=8**；864×480；有声；`--sdpa-backend cudnn`；reuse/magcache/sage/sol **全关** |
| **Tier-1 加速（opt-in）** | `sol_cute`（4090 主战场）、`sage3`（5090 主战场）；视觉不过门 → **不进默认** |
| **缓存类** | MagCache lite / calibrated（Tea 风格）已接线默认关；独立 Tea / Easy / FB **未齐或未实现** |
| **明确禁止当默认** | W4A8 主题、关 deblock、假 5s、未蒸馏 4 步、nested-loop Sol、假 sage2、SLA 插 8-step INT8 |

### 3.2 SGLang Diffusion（cookbook）

| 层 | 内容 |
| --- | --- |
| **参考质量** | `quality=lossless`：无 Cache-DiT 近似；BF16/FP32 混合；released recipe **768 short edge、常 50-step** |
| **加速档** | `quality=high` → 审计过的 **Cache-DiT** preset（**1.40× / SSIM 0.931 / PSNR 28.16**，**仅** audited：resident eager **4×H200**；别处 unaudited） |
| **其它臂** | Online FP8（B200/B300 resident 验证优先）；SageAttention（AMD 不暴露）；Ulysses / Ring / TP；layerwise offload（5090）；BCG；AdaLN cache；Turbo/少步适配器叙事 |
| **消费卡** | 单卡 layerwise offload；4090 可叠 `kitchen_int8`；**2×5090** 优先容量非延迟 |

### 3.3 vLLM-Omni

| 层 | 内容 |
| --- | --- |
| **参考** | `quality=lossless` / 省略；`num_inference_steps=50` 对齐参考精度 |
| **加速** | 请求级 `quality=high` → 保守 **Cache-DiT**；或服务器 `--cache-backend tea_cache` / `cache_dit`（**互斥**） |
| **TeaCache** | **仅标定 FL2VA**；默认例 `rel_l1_thresh=0.17`（相对「validated 107-frame T2VA」）；Ref2VA 不 cache |
| **注意力** | 消费 2-GPU：显式 **cuDNN**；数据中心 Blackwell 默认 **TRTLLM_ATTN**（与 FA4 测得相差 &lt;2%） |
| **Turbo** | 支持官方 Diffusers **4-step** LoRA 产物（`num_inference_steps=5` 等）；与 Cache-DiT `quality` **独立** |

---

## 4. 加速手段对照表

图例：**✅ 有且可开** · **◐ 半成品 / 间接** · **❌ 无** · **禁** 产品禁用 · **测** 本仓或公开有数字。

| 手段 | h3.c | SGLang Diff. | vLLM-Omni | 备注 |
| --- | --- | --- | --- | --- |
| **cuDNN SDPA** | ✅ 默认 · **测** | ✅（消费卡路径常见） | ✅ 2×5090/4090 配方显式 cuDNN | 本仓金线注意力基线 |
| **sol_cute（官方 CuTe Sol）** | ✅ Tier-1 · **测** DiT−15%、MAE~35 不过门 | ❌（Sol 在 Sol-Engine / 社区栈，非本 cookbook 主开关） | ❌ | 官方 4.44× 里 Sol 只是 **1.22× 增量臂** |
| **sage3（thu-ml FP4）** | ✅ Tier-1 5090 · **测** DiT−14%、MAE~27 | ✅ SageAttention 选项（非 AMD） | ◐ TRTLLM SAGE / 量化路径（损质 opt） | 本仓假 sage2 **禁** |
| **MagCache** | ✅ lite · **测** FAIL 默认 | ❌（名未作主推） | ❌ | 论文×相对长步数；8-step 易炸 |
| **TeaCache** | ◐ `magcache-calibrated` 风格 · 非独立模块 | ◐ 整栈常合入 Sol-Engine；SGLang 主推 **Cache-DiT** | ✅ `tea_cache` · FL2VA 标定 0.17 | 官方 50-step Tea 是 4.44× **主力** |
| **EasyCache** | ❌ | ❌（Comfy/Sol 侧有；#15326 伤 H3 音频） | ❌ | 有声高风险 |
| **FBCache / First Block** | ❌ | ◐ 社区/issue 讨论；非 cookbook 主表 | ◐ Cache-DiT 含 DBCache 类块策略 | 勿与本仓 `h3_w4a8_cache` first_block 权重窗混淆 |
| **Cache-DiT** | ❌ | ✅ `quality=high` · **测** 1.40×@4×H200 | ✅ `high` · **测** 1.35×@4×H200 | serving 级「可接受损质」口径 = SSIM，非本仓 MAE |
| **Online FP8** | ❌（另有 NVFP4 探测，非金线） | ✅ `--quantization fp8` | ◐ 权重/注意力量化路径 | 本仓金线是 INT8 ConvRot |
| **Ulysses / Ring / TP** | ❌（架构文写多卡路线，产品未成 serving） | ✅ cookbook 核心 | ✅ SP/Ulysses/VAE PP | 本仓单卡 |
| **SubBlock / BCG** | ❌ | ✅ Breakable CUDA Graph opt-in | ❌/其它图策略 | — |
| **TRTLLM_ATTN** | ❌ | ❌（SGLang 自有注意力栈） | ✅ 数据中心默认 | — |
| **reuse 固定 skip** | ✅ 默认关 | — | — | 与 mag 互斥 |
| **SLA / Turbo-SLA** | **禁**接 8-step INT8 金线（见 sla 文） | Turbo/少步适配器另档 | ✅ 4-step Turbo LoRA | 换权换配方 |

**GitHub 诚实注脚（SGLang）：** issue [#33720](https://github.com/sgl-project/sglang/issues/33720) 指出：若 DiT block **in-place**，Cache-DiT Pattern_3 residual 可能恒为 0 → `can_cache` 永不触发；审计 1.40× **可能无法在当前 main 复现**。引用公开数字时标明「cookbook 审计值 / 实现可能漂移」。

---

## 5. 性能数字（配方写清；不可横比声明）

### 5.1 本仓金线（唯一允许当「我们如何」的尺子）

| 项 | 值 | 来源 |
| --- | --- | --- |
| Prompt | cafe 574 ids，seed `0x11` | 矩阵计划 / q1 |
| 分辨率 / 时长 | **864×480**，`--seconds 5`（120@24fps），有声 | 同上 |
| 配方 | INT8 ConvRot，cfg=1，**8-step**，fp16 VAE tile+deblock，`sdpa=cudnn` | 同上 |
| **4090 e2e** | **~75.4s**（te≈11.6 / dit≈47.5 / vae≈9.3） | q1 / 4090 计划 |
| **5090 e2e warm** | **~66s**（dit≈30.7） | `verify_5090_bench` † |
| sol_cute @4090 | DiT **−15.3%**；e2e≈0；**MAE 35.2 不过门** | `verify_sol_vis_gate` |
| sage3 @5090 | DiT **−14.1%**（8-step）；**MAE ~26.6 不过门** | `verify_5090_bench` |
| MagCache 0.08 | ~79s，skips=3，**MAE 52–68** | 4090 计划 / 矩阵 |

### 5.2 SGLang Diffusion（公开）

| 场景 | 数字 | 配方要点 | 能否与本仓横比 |
| --- | --- | --- | --- |
| Cache-DiT `high` | **53.70s**，**1.40×** vs lossless；SSIM **0.931**，PSNR **28.16** | **4×H200**，eager resident，1344×768，124 帧，**50-step** T2VA | **否** — GPU 数/步数/分辨率/精度全不同 |
| H200 Ulysses4 | E2E **74.38s**（denoise 71.73 + decode 1.32） | 同上 50-step 768p，warmup 覆盖分辨率 | **否** — 4 卡 BF16 vs 单卡 INT8 8-step |
| B300 8× FL2VA BF16 | Latency **19.04s** | 8×Ulysses8，50-step 768p 5s | **否** |
| B300 8× FL2VA FP8 | **18.03–18.04s** | Online FP8 | **否** |
| **2×RTX 5090** | **559.67s**（denoise 525 + decode 33.6） | TP2 + layerwise，**50-step 1344×768 5s**；5-step 子测 inference **78.11s** | **否** — 50-step 768p offload vs 本仓 8-step 480p 单卡 |
| 消费单卡 offload | ~230–250s/请求量级（hint：~10.6s/step + decode；**20 NFE** 864×480 档另有 6–18 s/it） | layerwise；host RAM 敏感 | **弱相关** — 步数/精度/offload 不同；**非**本仓 INT8 金线 |

### 5.3 vLLM-Omni（公开）

| 场景 | 数字 | 配方要点 | 能否横比 |
| --- | --- | --- | --- |
| `quality` Cache-DiT | lossless **85.49s** → high **63.36s**（**1.35×**）；SSIM **0.9709**，PSNR **34.98** | **4×H200**，SP4 + TE TP4，1344×768，124 帧，**50-step** | **否** |
| 2×5090 验证 | 一次跑 **8 min 38 s** client E2E | 1344×768，124 帧，**50-step**；**非** warm 多跑 latency 表 | **否** — 且是能力验证非金线 |
| TeaCache | 给 thresh=0.17 与 FL2VA 限定；**未在 recipe 表给与本仓同形 e2e** | 与 Cache-DiT 互斥 | **不可横比**（无同配方公开秒数） |
| ROCm 例 | MI300X 4× T2VA E2E **267s**；MI350 1× 832×480 ~4s **40-step** ~**55s** 含 warmup | 非 NVIDIA 4090/5090 金线 | **否** |
| Turbo LoRA | 支持 4-step 产物；**recipe 未钉死与 75s 对照的秒数** | 换权 | **另档** |

### 5.4 官方 Sol-Engine（常被误读进「框架对比」）

| 声明 | 真相（本仓已钉） |
| --- | --- |
| 4090 **4.44×**（2239→504s） | **50-step BF16 768p** 整栈：TeaCache **~3.18×** + Sol **再 ~1.22×**；对照是未优化 Diffusers/SGLang BF16，**不是**本仓 8-step INT8+cuDNN |
| 5090 **4.52×**（OnDevice 页：1045→231s 量级梯子） | 同样是 **50-step 768p** + cache 主力；与本仓 cafe 8-step **不可横比** |
| 「Sol 给了 4×」 | **✗ 错误**；Sol 是注意力臂，cache 才是主力 |

### 5.5 为什么不能说「我们慢/快 X 倍」（Amdahl / 基线）

1. **步数：** 8 vs 50（或 4 Turbo）——墙钟差一个数量级的主因常在步数，不在「引擎谁优」。  
2. **精度与算子：** INT8 ConvRot + fused cuDNN vs BF16/FP8 + FA/TRTLLM/Sage——赢 FA ≠ 赢本仓 cuDNN。  
3. **分辨率 / 序列：** 864×480 cafe packed T≈15973 vs 1344×768 官方短边——算力与 TE 体积不同。  
4. **并行：** 单卡 vs 4×/8× Ulysses——延迟账本完全不同。  
5. **Amdahl：** 本仓 e2e 里 TE+VAE+开销已占 ~25–35s；即使 DiT 再砍 15%，e2e 常被 I/O 吃光（sol_cute 已实证）。  
6. **质量口径：** 本仓「不可辨 MAE」vs 框架 SSIM 0.93 / quality=high——过门标准不同。

---

## 6. 质量门口径差异

| | h3.c | SGLang / vLLM-Omni |
| --- | --- | --- |
| 默认目标 | **成片视觉与金线不可辨**（允许 checksum 不同） | **lossless** = 参考轨迹；**high** = 审计 SSIM/PSNR 带可接受偏差 |
| 主指标 | 同机 cudnn 对照 **PPM RGB MAE** + 口型/道具目视 + 有声波形 | **SSIM / PSNR** vs lossless；cookbook 明示 trade-off |
| 失败例 | sol MAE~35、sage MAE~27、Mag MAE 52–68 → **不过门** | high：SGLang SSIM 0.931；vLLM SSIM 0.9709（不同拓扑审计） |
| 音频 | EasyCache / 视频 skip 套音频 → 高风险；要求 AV 分账 | 同步输出；Sage/cache 需人工检视 |

**结论：** 框架的 `quality=high` **不等于**本仓 Tier-0；把 SSIM 0.93 当成「可以改默认」会直接破坏本仓产品承诺。

---

## 7. 工程代价

| | h3.c | SGLang | vLLM-Omni |
| --- | --- | --- | --- |
| 运行时 | C/CUDA；Tier-1 可选 Python embed 插件 | Python + Triton/插件 + 大依赖 | Python + vLLM-Omni + 可选 FA4/TRT |
| 单卡可跑性 | **强**（INT8 金线专为 24GB 设计） | 需 layerwise + **大主机 RAM**（2×5090 写明 ~384 GiB 级） | 单卡 CPU offload 可行但慢；2×5090 验证过 50-step |
| 多卡 | 架构讨论有；**非**产品 serving | **一等公民**（Ulysses/Ring/TP/FSDP） | **一等公民**（SP/VAE PP/TE TP） |
| 可控性 | CLI/env 明确；缺插件 **硬失败**；默认永不自动 sage/sol | cookbook picker 复杂；拓扑锁（high 审计绑 4×H200） | 请求级 quality；Tea/Cache-DiT 互斥 |
| 运维 | 无 HTTP 队列；适合嵌入/工具链 | 为并发、副本、warmup 设计 | OpenAI-compat API；Comfy 前端集成在推进 |

---

## 8. 差距与机会

### 8.1 本仓优势（相对两者）

- 单卡 **8-step INT8** 真 5s 有声路径短、依赖可控。  
- **质量门严**且有可复现 cafe 金线与 phase_ms 拆账。  
- 注意力 Tier-0/1 registry 已产品化（cudnn / sol_cute / sage3）。  
- 不被 serving 拓扑 / Python 栈绑架。

### 8.2 本仓短板

- **无 serving API**、无官方 Ulysses 多卡延迟故事。  
- **Cache-DiT / 独立 TeaCache / FBCache** 相对框架缺失或半成品；8-step 上 skip 空间可能极小。  
- Ref2VA / 全模态组合深度弱于 cookbook。  
- LoRA Turbo / 官方 4-step 产品档未接（且故意与金线隔离）。  
- Tier-1 注意力 **质量未过门**（长期线）。

### 8.3 优先缺口（若要对齐「Cache-DiT / TeaCache serving 体验」）

按 ROI 与本仓约束排序（与矩阵计划 §8 一致，略作框架对齐修订）：

| 优先级 | 动作 | 理由 |
| ---: | --- | --- |
| **P0** | 守住 Tier-0；矩阵填满已有注意力数字；**禁止**拿框架 50-step 秒数改金线承诺 | 避免错误产品决策 |
| **P1** | **独立 TeaCache**（默认关）+ 有声 AV veto；在 cafe 8-step **从严**扫 thresh | 官方 4.44× 主力臂；serving 也标 Tea；本仓需诚实测「还有没有 skip 空间」 |
| **P2** | MagCache / calibrated 统一 harness 复现 FAIL 证据 | 成本低，防回潮默认 |
| **P3** | 保守 **FBCache** 或 Cache-DiT **思想**的块级 residual skip（默认关） | 对齐 vLLM/SGLang「high」体验；须自建 **MAE 门**，勿抄 SSIM 过门 |
| **P4** | EasyCache（最后；强制 AV 分账） | #15326 |
| **P5** | sol_cute / sage3 **质量**（dense 前缀、sink 等）— 过门前不改默认 | DiT 有点，e2e/视觉未闭环 |
| **P6** | 条件图 vision host SDPA→cuDNN | F2VA/L2VA 大头；T2VA 金线不受益 |
| **非目标（现阶段）** | 复现 4.44×；接 SLA 进 8-step INT8；把框架 4×H200 74s 当对标 KPI | 基线错误 |

---

## 9. 建议行动（给产品 / 工程）

1. **对外叙事：** 「单卡质量金线 ~75s@4090 / ~66s@5090（8-step INT8）」与「多卡 serving 50-step 768p」**分列**；永不写成同一倍速。  
2. **对内投资：** TeaCache 独立化 + 矩阵验收；Cache-DiT 类仅作 opt-in 研究档。  
3. **Serving：** 若需要 API，优先评估 **调用** SGLang/vLLM 做边缘网关，本仓继续守金线引擎——或明确另开 multi-GPU 里程碑，勿混进 75s KPI。  
4. **质量默认零妥协：** 任何 cache/稀疏/量化进默认前，必须过 cafe MAE + 有声门。

---

## 10. 附录：来源锚点

| 声明 | 锚点 |
| --- | --- |
| 4090 cafe ~75.4s | `out/verify_480p_quality_full/t2va_complex_5s_cafe_q1/`；4090 计划 §2.1 |
| 5090 ~66s / sage3 | `out/verify_5090_bench/REPORT.md` |
| sol_cute 视觉不过 | `out/verify_sol_vis_gate/REPORT.md` |
| 4.44× 读法 | 4090 计划 §4.1；Sol-Engine README |
| SGLang Cache-DiT 1.40× | cookbook § quality / Benchmarks |
| SGLang 2×5090 559.67s | cookbook § RTX 5090 capacity run |
| vLLM high 1.35× | recipe § Request-scoped quality |
| vLLM TeaCache 0.17 FL2VA-only | recipe § TeaCache acceleration |
| SLA 不接金线 | `2026-08-25-sla-vs-h3c.md` |

**未公开 / 不可横比（诚实留白）：**

- 本仓 cafe 8-step INT8 与 SGLang/vLLM **同卡同分辨率同步数**的并排 e2e：**无**。  
- vLLM TeaCache 在 H3 上的公开 median 秒数表：**无**（仅有 thresh 与适用范围）。  
- SGLang 消费卡 INT8 8-step 与本仓金线对齐数字：**无**。  
- 官方 sparse / SLA 训练自由核：**未开源可链入本仓 C 路径**（见 sla 文）。
