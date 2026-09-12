# TaoMate-H3 深度分析：对本仓 5090 turbo 方案的可借鉴性评估

> **For agentic workers:** 纯分析文档，不是实现计划。禁止改代码 / 改金线默认 / commit / 碰 `out/` 实测报告。  
> **对照基线：** 单卡 5090 金线（INT8 ConvRot 8-step + cudnn，864×480 5s warm e2e 78.7s：DiT 31.7 / TE 14.9 / VAE 10.6 / mux 1.4）；turbo4（4-step larryvrh v4 EMA r=64 LoRA + Sage3 + cache 全关，预期 e2e ~40s 档，8×5090 实测中）。  
> **关联文档：** [PCIe 多卡教训](./2026-09-11-h3c-pcie-multigpu-lessons.md) · [8x5090 768p 估算](./2026-09-11-h3c-8x5090-768p-estimate.md) · [turbo SKU 选型](./2026-09-11-h3c-5090-turbo-sku.md) · [turbo 实现计划](./2026-09-11-h3c-5090-turbo-implementation.md) · [少步质量](./2026-09-11-h3-few-step-quality.md)  
> **一手来源：** [GitHub TaoLiveAIGC/TaoMate-H3](https://github.com/TaoLiveAIGC/TaoMate-H3)（源码已逐文件读过，浅克隆自 main）· [HF TaoLiveAIGC/TaoMate-H3](https://huggingface.co/TaoLiveAIGC/TaoMate-H3)（README + adapter_config.json）

**一句话：** TaoMate-H3 的核心是「**3 步 LoRA × 4 小块流式 × clean-KV 滑窗**」，赢的是**首块可播延迟**与**长片恒定成本**，不是总吞吐——它的每请求 GEMM 总量与 4-step 全片**完全相同**（下 §2.3 有账）。对本仓：clean-KV 长片连续性、TE 全 prompt 预计算、音频与视频步数解耦这三条值得抄思路；它的 LoRA 权重、TP2、FA3 都抄不了。「5s 全片 2s」在 8×5090 PCIe 上不可达（公开 SOTA 唯一 sub-2s 是 NVIDIA Sol-H3 的 8×B300 + NVLink，还不含 MP4 封装）；现实最优是「全片 ~5–8s + 首块可播 ≤3s」两档。

---

## 1. TaoMate-H3 是什么（源码级拆解）

发布物 = **一个 rank128/alpha128 step-3000 generator-EMA LoRA**（HF，FP32 存储、BF16 推理）+ **一个 PyTorch 流式 runtime**（GitHub，vLLM 0.11.1 只用其 CUTLASS W8A8 算子，FA3 hopper 分支做注意力）。验证配置 8×H20 96GB（SM90，NVLink），TP2×Ulysses4；4 卡档 TP2×Ulysses2。分辨率 480p/768p/1080p（1088 生成后裁 1080）。

### 1.1 三步去噪区间怎么切（`denoise_schedule.py` + `model/pipeline.py`）

不是独立设计的 3 步表，而是**从官方 50 步 shifted sigma 表里取 4 个锚点**：

```
DISTILLED_STATE_INDICES = (0, 16, 33, 49)     # 50 步表的下标
video: time_shift_sigmas(50, shift=12.0)[{0,16,33,49}]
audio: time_shift_sigmas(50, shift=3.0)[{0,16,33,49}]
→ 4 个 sigma → 3 次 Euler-eta0 步进
```

LoRA 的蒸馏目标就是「一步跳完 50 步表的一个区间」（[0→16]、[16→33]、[33→49]，即 config 里 `stage3_target_state_indices=(16,33,49)`）。这与 FastH3/larryvrh 的均匀 4-step sigma 完全不同轨。视频/音频保持官方双 shift（12.0 / 3.0）双时钟，与本仓金线一致。

### 1.2 分块调度（`streaming/geometry.py`）

一个 5s 请求（124 native 帧 = 官方 17n+5 桶）切成 **4 个 phase**，按 H3 native codec 组（17 帧/5 latent，另有 5 帧/2 latent 前缀）分组：

| phase | codec 组 | 帧 | video latent | 3-step forward | clean commit |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | 2 组 + 前缀 | 39 | 12 | 3 | 1 |
| 1 | 2 组 | 34 | 10 | 3 | 1 |
| 2 | 2 组 | 34 | 10 | 3 | 1 |
| 3 | 1 组 | 17 | 5 | 3 | 1 |
| 合计 | | 124 | **37** | **12** | **4** |

音频按 40Hz latent 时钟在全局帧时间轴上取整对齐（`_audio_latent_boundary`，banker's rounding），5s 共 207 latent/声道。10s = 2 个请求；请求 1 走 `canonical_continuation_plan`（不再生成 5 帧前缀，7 组 = 119 帧 / 35 latent，音频 198–199）。所以 **10s = 24 次去噪 forward + 8 次 clean-KV forward**，与 README 数字精确吻合。

每个 phase 是一个独立的 packed 序列（text + 本块 video + 本块 audio），RoPE 时间坐标放在**全局请求时间轴**上（`video_temporal_position`：前缀 latent 权重 1、其余 4，步长 5/3；prompt 右对齐到当前媒体原点）。跨 5s 请求边界时 prompt 可以换（`--prompt-json` 每 5s 一条）。

### 1.3 clean KV cache（`streaming/cache.py` + `attention_hook.py`）

**结构：** 每层（50 层 attn）持久保存 BF16 的 K/V，形状 `[tokens, local_heads, 128]`（local_heads = 56/(TP×U)，8 卡 = 7）。只存**干净（全去噪后）**的 media token KV——每个 phase 3 步去噪完，把结果行以 `timestep=1.0` 再跑一次 forward（`streaming_cache_only=True`），把这次 forward 的 K/V 原子提交进 cache。**这就是「clean KV」：cache 里从来没有含噪声的 KV。**

**保留策略（`retain_sink_and_recent_commits`）：** 首块的 video 行永久保留为 **sink**（音频行老化后丢弃），加**最近 2 个完整 AV 块**。即注意力窗口 = sink + 2 recent，长片成本有界。另有 `audio_kv_reset_window_requests=12`：每 12 个 5s 请求把音频历史整体清掉一次（防漂移）。

**注意力（FA3 dense hook）：** 当前块的 media query 对 `[text 条件 + 持久 clean KV + 本块 media KV]` 拼接后做一次 dense FA3（非因果、`num_splits=1`）；text 条件行只在本块内部自注意。没有稀疏核，靠「窗口有界」控制成本。

**连续性辅助：** `_renorm_clean_video_rows` 把每个新块的 latent 均值/方差仿射对齐到首块统计（`prefix_normalization=affine_to_first_stream_chunk`）；跨请求传递最后 2 latent 视频 + 9 latent 音频作为 transport 前缀。

### 1.4 音频：Base10 teacher（`teacher.py` + `base10_teacher.py`）——最容易被忽略的机关

**流式 3-step LoRA 并不真正生成音频。** 正式推理前有一个**独立的 torchrun 预备阶段**：用 **BF16 基座（不挂 LoRA）** 跑 audio-only packed 序列（**0 个 video 行**，只 text + 音频 207×2 行），10 状态表（9 次 forward，shift 12/3），在 step 3/6/9 捕获三个 milestone。正式流式循环里每步结束用 `on_step` 回调把音频行**整体覆盖**为对应 milestone；最终发布的音频与 teacher clean latent **逐位相等**（runtime 里有 `torch.equal` 硬断言）。跨请求靠尾部 40 latent（1s）rollover 作为 frozen 前缀保证声音连续（`fl2va_tail40_rollover`）。

意义：3 步蒸馏权只需要对视频质量负责，音频保持基座 9 步质量；audio-only 序列只有 ~1000 行（vs 全 AV ~15k+ 行 480p），9 次 forward 很便宜。**但公开性能表把这段完全排除**（"excludes the command's internal audio preparation"）——而它是独立进程、含一次完整模型加载和自己的 TE。

### 1.5 LoRA adapter（HF + `inference/lora_checkpoint.py`）

- **52 个模块**：`token_refiner.blocks.{0,1}` + `blocks.{0..49}`，每块 4 投影（`attn.qkv_proj / attn.out_proj / mlp.fc1 / mlp.fc2`），共 208 对 A/B。**含 token_refiner，无 adaln。**
- 格式是**自有扁平 safetensors**（`blocks.N.attn.qkv_proj.lora_a/lora_b`，FP32），**不是 PEFT 目录命名**（无 `base_model.model.` 前缀）；`adapter_config.json` 也只有 `{rank, alpha, weight_source, optimizer_step}` 四个键，非 PEFT schema。
- 推理时 LoRA **不 merge**：包一层 wrapper，BF16 buffer，`base(x) + B(A(x))·(α/r)`。TP 下 A/B 按行/列切；row-parallel 模块开 `fused_row_reduce`——**LoRA partial 在 TP all-reduce 之前加进 base partial，省掉每模块一次额外 AR**。这与本仓「LoRA 挂在 INT8 GEMM 后做残差」同构。

### 1.6 精度与内核（`inference/w8a8.py` + `fused_kernels.py`）

- 精度策略 `bf16_with_interior_fc1_qkv_w8a8`：**只**量化内层 block 的 `qkv_proj` 和 `fc1`（per-output-channel INT8，vLLM CUTLASS `cutlass_scaled_mm`），**保护首 2 块 + 末 3 块整块 BF16，且 out_proj/fc2 全程 BF16**。比本仓全 50 块 INT8 ConvRot 保守得多。加载时直接流式量化进 INT8 存储（不落 BF16 全量）。
- Triton 融合核一套：fused modulate / gate-add / rope / swiglu / rmsnorm / qk-norm-rope / 合并 QKV 预处理；函数名就叫 `enable_h3_sol_engine_fusions`——**明显借了 NVIDIA Sol-Engine 的融合清单**。另有 video VAE 的 exact fusion（swiglu / scaled residual / rgb24）。
- 静态 timestep AdaLN cache（`enable_inference_static_timestep_cache`）：3 步的 timestep 集合固定，AdaLN 投影可缓存。

### 1.7 并行（TP2×Ulysses4，`distributed/`）

TP2 硬编码（`config.py` 断言 `tensor_parallel_size==2`），8 卡 U4 / 4 卡 U2。Ulysses 用 `all_to_all_single`，Q/K/V stack 成**一次集体**（`packed_qkv_sequence_to_heads`），输出再一次 A2A。TP 是 Megatron 式 column/row 切 + row 后 AR。头数账：56 头 / TP2 = 28，28/U4 = 7 个本地头（合法）。**这套拓扑的前提是 H20 有 NVLink（900GB/s 级）**——TP2 意味着 50 块 × 每 forward 2 次（out_proj+fc2）hidden 级 all-reduce，PCIe 上是灾难（§5）。

### 1.8 计时口径的诚实拆解（重要）

公开表（8×H20，480x864，10s，seed 8301）：纯 DiT 14.810s（11.45×）、首块 final latent 6.148s（27.66×）、首个可播视频 17.287s（10.60×）、峰值显存 31.37GiB。逐条抠：

1. **纯 DiT 14.81s** = 24 noisy + 8 clean 共 32 次 forward 的 CUDA event 时间（跨 rank 取 max），**不含** TE、VAE、加载、音频准备。真实分母：官方 169.57s 是同机 50-step。
2. **发布的命令并不流式出片。** `runner.py` 的 `publication=one_shot_audio_video_vae`：全部请求 latent 算完 → 一次 video VAE decode → 一次 audio VAE → 一次 ffmpeg mux。**「首个可播 17.29s」来自另一个 matched first-chunk publication benchmark，不是开源命令的行为。**
3. **音频 Base10 teacher 整段被排除**：独立 torchrun、二次模型加载、每请求 9 次 audio-only forward + TE。用户真实端到端墙钟远大于表内数字。
4. 首块 6.148s 与纯 DiT 内部账对不上：phase0 占 48/288 latent-unit（10s = 148+140，续段 35 latent）≈ 2.5s DiT，说明 6.148s 里还含 TE/噪声准备/首次调度开销等（口径未公开细节，存疑标注）。
5. H20 是弱算力 Hopper（~148 TFLOPS FP16 dense），8 卡合计 ~1.2 PFLOPS；8×5090 dense BF16 合计 ~1.7 PFLOPS。**同等效率下这套 3-step 方案在 8×5090 上 DiT 会更快**——瓶颈换成 PCIe 通信而非算力。

---

## 2. 与本仓 turbo4 的逐项对比

| 维度 | TaoMate-H3 | 本仓金线（5090×1） | 本仓 turbo4（8×5090 实测中） |
| --- | --- | --- | --- |
| 步数/轨迹 | 3 步 = 50 步表锚点 {0,16,33,49} 区间蒸馏 | 8 步（官方轨迹） | 4 步均匀（larryvrh v4 蒸馏轨迹） |
| 生成粒度 | 4 小块流式（12/10/10/5 latent）+ clean-KV | 全片一次 | 全片一次 |
| LoRA | r128 α128 step3000 EMA，52 模块（含 token_refiner），自有格式，不 merge | 无 | r64 v4 EMA，50 块 qkv/out/fc1/fc2，INT8 GEMM 后残差 |
| 骨干精度 | BF16 + 内层 qkv/fc1 W8A8（护首2末3块） | 全 50 块 INT8 ConvRot | 同金线 INT8 |
| 注意力 | FA3 dense（SM90 专属）+ KV 窗口有界 | cuDNN SDPA | SageAttention3 FP4 |
| 音频 | 基座 9-forward audio-only teacher 预计算，流式步覆盖注入 | 联合 AV 8 步 | 联合 AV 4 步 |
| TE | Qwen3-VL layer-50，TP2，**全 prompt 预计算一次** | NVFP4 每请求 ~14.9s，无 cache | 同金线 |
| VAE | 官方 VAE 一次性 decode（+Triton exact fusion）；无 TAE | 官方 fp16 VAE 10.6s | 同金线（TAE 是 stub） |
| 多卡 | TP2×U4，8×H20 NVLink | 无 | Ulysses U∈{1,2,4,8} 验证器已写，NCCL 未装未实测，无 TP |
| 长片 | clean-KV sink+recent2，恒定成本，5s 粒度换 prompt | 不支持 >5s 产品化 | 同左 |
| 公开数字 | 10s 480p 纯 DiT 14.81s（8×H20）；首块 latent 6.15s；可播 17.29s（专项 bench） | 5s 480p e2e 78.7s 全含 | 预期 e2e ~40s 档（单卡） |

### 2.3 关键的计算量账：3-step 流式 ≠ 更少计算

每 5s 请求的 DiT latent-unit（video latent × forward 次数）：

- TaoMate：4 块 × 3 步 = 3×37 noisy + 1×37 clean = **148 latent-unit**
- 本仓 turbo4：4 步全片 = 4×37 = **148 latent-unit**

**GEMM 总量完全一样**（clean commit 就是它的「第 4 步」）。差别在注意力：TaoMate 每次 forward 的 Q 只有本块（≤12/37），K/V 是「本块 + 有界窗口」，代价 ~S_chunk×S_window；全片 dense 是 S²。480p 5s（S≈15k）时这点差别不大且被 4 次额外的 packed 布局/调度开销部分抵消；**768p（S≈37k）和 10s+ 长片才是流式分块的主场**——全片 S² 翻 4 倍时，流式每块成本不变。

**所以对本仓「一次出 5s 480p 成片」的产品形态：3-step 流式在总吞吐上对 turbo4 没有优势，唯一收益是首块可播延迟。**

---

## 3. 可借鉴清单（按价值排序）

| # | 思路 | 理由 | 落地建议 |
| --- | --- | --- | --- |
| 1 | **TE 全 prompt 预计算 / prompt cache** | 他们 `precompute_text_prompts` 一次算完全部 5s 段的 prompt embedding 再进 DiT 循环，TE 从关键路径消失。本仓 TE 14.9s 是 turbo4 之后的**最大单项**，且纯文本 embedding 与 seed/分辨率无关，可持久 cache | P0。`h3.c` 加 TE embedding 磁盘/内存 cache（key = prompt+TE 权指纹）；重复 prompt 请求 warm 命中 → e2e 直接 −14.9s |
| 2 | **clean-KV 滑窗长片连续性** | 「10s 视频」的正路：不是把 74 latent 塞进一个 dense forward（S² 翻 4 倍），而是两个 5s 段 + sink/recent KV 窗口 + 音频尾部 rollover + latent 统计 renorm。结构简单（每层 append-only KV + 3 段拼接注意力），本仓 cuDNN/sage 都能拼 KV | P2（10s 产品立项时 P0）。但**必须配匹配训练的权重**（见不可借鉴 #1）；短期可先试「基座 8-step + clean-KV 拼接」验证连续性机制本身 |
| 3 | **音频与视频步数解耦（audio-only teacher）** | audio-only packed 序列（0 video 行）forward 极便宜（~1k 行 vs 15k 行），9 步音频 + 4 步视频可以让 turbo4 的音频质量兜底到基座水平。本仓 turbo4 若人声/音效不过关，这是最便宜的修复刀 | P1。本仓 packed 布局支持去掉 video 行即可复用现有 INT8 DiT 跑 audio-only pass；成本估 <2s（480p 口径） |
| 4 | **首块可播（流式分块）产品形态** | 「2s」目标唯一现实的解释是 TTFF（time-to-first-frame）。TaoMate 用 27.66× 的首块改善证明了用户体感价值。分块按 H3 native codec 组（17f/5lat + 5f/2lat 前缀）切齐，VAE/mux 也能按块出 | P3（工程量大：分块调度 + 分块 VAE + 分段 mux + 流式 LoRA）。先把 #1/#2 收益吃完 |
| 5 | **边界块保护式量化** | 他们只量化内层块的 qkv/fc1、保护首 2 末 3 块。本仓 INT8 全量在 8-step 金线过门，但 turbo4 4-step 下误差预算更紧——若 turbo4 质量门不过，「首末块回 BF16/FP16」是现成的手术刀 | P2 备用。有 `h3_dit_w4a8` / cache 基建，按 block 白名单降级不难 |
| 6 | **静态 timestep AdaLN cache** | 4-step 的 timestep 集合固定，AdaLN 投影可预计算。本仓 `h3_step_cache` 已有类似基建，确认 turbo4 路径开没开 | P2 顺手项 |
| 7 | **锚点式少步 sigma（从 50 步表取锚点）** | 与均匀 4-step 是不同蒸馏轨迹设计；若未来自蒸馏 3–4 step 学生，锚点式（前密后疏，shift 后 [0,16,33,49] 前段步长大）值得进对照组 | 仅记录，蒸馏立项才用 |

## 4. 不可借鉴清单

| # | 项 | 为什么 |
| --- | --- | --- |
| 1 | **TaoMate LoRA 权重直接挂本仓 dense 4-step 路径** | 训练目标是「3 步跳 50 步表区间 × 小块 packed 布局 × clean-KV 历史条件 × 块级 renorm」。脱离其调度：(a) sigma 锚点不同（{0,16,33,49} vs 均匀 4-step），(b) 训练时每个 forward 都看到 clean-KV 历史上下文，dense 全片没有；(c) packed 布局是 12-latent 小块不是 37-latent 全片。任何一条都足以掉质量，三条叠加大概率崩。另有硬工程障碍：52 模块含 token_refiner（本仓 `h3_lora` 只映射 blocks.{0..49}×4），自有命名非 PEFT 需写转换。**诚实说明：没实测，不能 100% 断言，但训练-推理匹配原则下不值得花加载器工程量去赌** |
| 2 | **TP2（任何形式的 TP）上 8×5090** | 见 §5 带宽账。他们的 TP2 前提是 NVLink + 96GB BF16 常驻需求；本仓 INT8 单卡已常驻，TP 是纯亏 |
| 3 | **FlashAttention-3** | hopper 分支 SM90 专属，5090 是 sm_120 无 FA3。本仓 cudnn/sage3 已覆盖 |
| 4 | **vLLM CUTLASS W8A8 / PyTorch 栈** | 本仓 C/CUDA 独立实现；且他们的 W8A8 比本仓 INT8 ConvRot 覆盖面窄、无 ConvRot，量化技术上是倒退 |
| 5 | **把他们的表读成「流式更快」** | §2.3：GEMM 总量与 4-step 全片持平。11.45× 的分母是 50-step；首块 27.66× 是延迟指标不是吞吐。任何「TaoMate 比 turbo4 快」的说法都是口径错误 |
| 6 | **一次性 publication 当流式宣传** | 他们开源命令本身就是全片一次出（`one_shot_audio_video_vae`）；如果本仓做流式 SKU，分块 VAE/mux 得自己写，抄不到 |

---

## 5. PCIe 多卡账：TP2×Ulysses4 在 8×5090 上不可行

（口径与 [pcie-multigpu-lessons](./2026-09-11-h3c-pcie-multigpu-lessons.md) 一致，此处补 TaoMate 视角的数字。）

**TP2 的 AR 税：** 50 块 × 每 forward 2 次 row-parallel all-reduce（out_proj、fc2），每次体积 = S×hidden×2B。480p 5s 全片 S≈16k（cafe 实测 packed 15,973）、hidden 5376 → 每次 AR ~170MB（流式小块按比例小，但次数同）。TP2 ring-AR 有效传输 ≈ 1× 数据量：

- H20 NVLink ~900GB/s：170MB → ~0.2ms/次 → 100 次/forward ≈ 20ms，可忽略。
- 5090 PCIe 5.0 x16 ~64GB/s（实测常 ~50）：170MB → ~2.7–3.4ms/次 → **~0.25–0.3s/forward**；16 次 forward/请求 ≈ **4–5s 纯 AR 税**，与 8 卡分掉后的 DiT 计算本身同量级。另外每块 2 次同步 AR 的延迟叠加在 kernel 间隙里，实际更糟。

**纯 Ulysses8 的 A2A 税（对照）：** 每层 2 次 A2A，QKV 打包后每卡收发 ~S/8×hidden×3×2B×(7/8)≈33MB/次，PCIe 上 ~0.6ms×100 次/forward ≈ 60ms/forward，16 forward ≈ **~1s**——形状友好一个量级。且 A2A 是置换不是归约，跨 NUMA 只是变慢不变形。

**结论不变：** 8×5090 先测 **TP=1 + Ulysses8**（56%8=0、64%8=0 合法）；TaoMate 的 TP2×U4 是 NVLink 机器的解，照抄进 PCIe 机器等于把最贵的通信轴放到最差的互连上。TaoMate 唯一值得抄的通信细节是 **QKV stack 成一次 A2A 集体**——本仓 `h3_ulysses_comm.cu` 已按此设计（destination-major 一次集体），方向正确。

---

## 6. 「2s 生成」诚实结论

### 6.1 公开 SOTA 坐标（2026-09）

| 方案 | 硬件 | 5s 结果 | 口径 |
| --- | --- | --- | --- |
| **NVIDIA Sol-H3**（[项目页](https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3/)） | 8×B300 NVLink | **1.653s**（768p 1344×768！） | 4-step FastH3 + SOL 稀疏注意力 + INT8 QKV/FP8 传输；warm；含 TE+DiT+VAE，**不含 MP4 封装**；单卡 B300 同配方 13.7s |
| TaoMate-H3 | 8×H20 NVLink | 10s 480p 纯 DiT 14.81s；首块 latent 6.15s；可播 17.29s | 不含 TE/VAE/加载/音频 teacher（除「可播」含 VAE+H264 专项 bench） |
| Sol Super-Accel（[页](https://nvlabs.github.io/Sana/Sol-Engine/H3-Super-Acceleration/)） | 2×GB200 | 6.85s（768p，两段式 H3 草稿+LTX 精修） | 见本仓 sol-engine-excerpts |
| RH Lightning | 8×RTX6000D PCIe | 28.7s（768p 4-step 全栈） | 本仓已拆账 |
| RAVEN streaming LoRA（[HF](https://huggingface.co/mvp-lab/MiniMax-H3-RAVEN-Streaming-LoRA)） | 24GB 单卡可跑 | 192f 768p 301s、首段 51s（ComfyUI 验证） | 公开因果流式 LoRA（r128、4-NFE、sink2+window2），**未训完的 preview** |

**全片 2s 在公开 SOTA 中无先例**（Sol-H3 的 1.653s 最接近，但那是 8×B300——单卡 dense 算力数倍于 5090、FP4 TensorCore 更宽、NVLink 全互连，且不含封装）。

### 6.2 8×5090 的理论账（864×480 5s，全部标外推）

| 段 | 现状（单卡） | 各刀之后（8 卡） | 依据 |
| --- | ---: | ---: | --- |
| TE | 14.9s | **~0（cache 命中）** / 14.9s（未命中，可与首块 DiT 重叠） | 可借鉴 #1；纯文本 embedding 可持久化 |
| DiT | turbo4 预期 ~16s | U8 理想 2s + PCIe A2A 税 ~0.5–1.5s → **~2.5–3.5s** | §5；效率打 7 折则 ~4s |
| VAE | 10.6s | TAE ~1–3s（本仓 stub 需补）或 8 卡 tile 并行 ~1.5–3s | Sol 用 TAE 成片先例 |
| mux | 1.4s | ~1s | 优化空间小 |
| **全片合计** | ~40s（turbo4 单卡） | **~5–8s** | — |

**结论：**

1. **「5s 全片 2s」不可达。** 即使 DiT/VAE 全部理想化，mux + 采样杂项 + 集体通信启动已 ~2s。物理上需要 Sol-H3 级硬件（8×B300）+ 不计封装的口径才摸到 1.65s。
2. **「首块可播 ≤2–3s」是可立项的现实变体**：TE cache 命中 + 流式首块（12/37 latent × 3-step，8 卡 DiT ~0.6–1s）+ 首块 TAE decode（39 帧，<0.5s）+ 分段 mux。前提是拿到/训出流式 3-step 权重（TaoMate 权不通用，RAVEN 未训完，大概率要自蒸馏），工程量 L。
3. **「10s 视频 2s」更不可达**；10s 的正路是 clean-KV 两段拼接（可借鉴 #2），预期 10s ≈ 1.5–1.8× 5s 档墙钟（TE/加载摊销），即 8 卡 ~8–13s。
4. **最优方案排序（性价比）**：① TE prompt cache（−14.9s，工程量 S–M）→ ② Ulysses8 实测落地（DiT −85%，L 但已有代码）→ ③ TAE 成片（VAE −8s，M）→ ④ 音频 teacher 解耦兜底 turbo4 质量（M）→ ⑤ clean-KV 10s 长片（L）→ ⑥ 流式首块 SKU(XL)。①+②+③ 落地后全片 ~5–8s，已把「不惜成本」的硬件红利吃满；再往下只剩换硬件（B300/GB200）或降口径（首块延迟代替全片）。

---

## 7. 其他公开方案雷达（确保不漏）

- **alibaba-pai PDD LoRA**（[HF](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs)）：官方 8-step r64 PDD（并行解码蒸馏，arXiv 2607.26004），强度必须 1.0、步数限 4/8/16。8-step 对 turbo4 墙钟无收益，但社区对比里质量口碑好于 turbo 4-step——**turbo8 中档 SKU 的候选权重**，记入 turbo-sku 文档的备选。
- **ModelTC Minimax-H3-Turbo**（[GitHub](https://github.com/ModelTC/Minimax-H3-Turbo)）：lightx2v 系 4-step 蒸馏 v0.1 preview，质量自述待改进；与 larryvrh v4 同生态位，无需换。
- **RAVEN streaming**：见 §6.1，唯一公开的因果流式 LoRA，undertrained；若做流式 SKU 值得跟踪其后续版本，省自蒸馏成本。
- 未发现比 Sol-H3 更快的公开结果；未发现任何 5090/消费卡上的 sub-10s 全片先例。

## 8. 来源

- https://github.com/TaoLiveAIGC/TaoMate-H3 （源码：`denoise_schedule.py`、`streaming/{geometry,cache,attention_hook,runtime}.py`、`teacher.py`、`inference/{base10_teacher,lora_checkpoint,w8a8,acceleration,fused_kernels}.py`、`distributed/ulysses.py`、`config.py`、`runner.py`、`cli.py`）
- https://huggingface.co/TaoLiveAIGC/TaoMate-H3 （README 性能表、adapter_config.json：`{rank:128, alpha:128, weight_source:generator_ema, optimizer_step:3000}`）
- https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3/ · https://nvlabs.github.io/Sana/Sol-Engine/H3-Super-Acceleration/
- https://huggingface.co/mvp-lab/MiniMax-H3-RAVEN-Streaming-LoRA · https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs · https://github.com/ModelTC/Minimax-H3-Turbo · https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora
- 本仓：[pcie-multigpu-lessons](./2026-09-11-h3c-pcie-multigpu-lessons.md) · [8x5090-768p-estimate](./2026-09-11-h3c-8x5090-768p-estimate.md) · [5090-turbo-sku](./2026-09-11-h3c-5090-turbo-sku.md) · `third_party/accel_survey/sol-engine-excerpts/`

未改代码。未 commit。未触碰 `out/`。
