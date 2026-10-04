# H3.c 优化路线图：分场景目标书（四档 SKU + 逐点归因）

> **日期：** 2026-10-05。**机器基准：** 192.168.0.238，1×RTX 5090 32GB，sm_120，CUDA 13。
> **定位：** 这是本项目的**工程目标书**——每一档（SKU）有一个**可测量的具体目标**、一条**逐点归因的达标路径**、一组**验收门**。没有目标的优化不做；达不到目标的承诺不写。
> **数据纪律：** 文中所有「实测」数字来自 238 真机日志（`out/review-2026-10-04/`、`docs/superpowers/plans/2026-10-04-h3c-480p-10s-plugin-plan.md` §5 修订记录、`2026-10-01-h3c-nvidia-rtx5090-sol-h3-analysis.md` §8）；所有「预估」都标注依据。不假数字、不承诺未证明的秒数。
> **方法论审计依赖：** 另有一个方法论审计代理可能重开部分已验证死路（dead-end）。凡本计划依赖其结论处，一律标注 **[审计依赖]**，按**风险/依赖**处理，**不作为承诺**。

---

## 0. 全局地基（所有档位共享的实测账本）

### 0.1 480P 基线（864×480 · 5s · 24fps · turbo4 · 官方 VAE · sol_cute · 热 clip2）

| 阶段 | 耗时 | 占比 | 状态判定 |
| --- | ---: | ---: | --- |
| DiT（4-step INT8 ConvRot + sol_cute） | **10.37s** | 63% | INT8 GEMM 已达硬件峰值 98-99%（C5b 三层验证），**无 raw-FLOPS 余量** |
| VAE（官方 fp16，含 stitch pipeline） | **4.96s** | 30% | 族#6 零成本浪费子集 ~2.3s（unpatch_d2h 串行尾 ~1.6s + FillFunctor ~0.7s，bit-identical 可达，**P0 靶子**）；vae_blocks 主体 ~2.6s 为 fp16 未量化真实计算（族#5 未收割，有质量杠杆）——归因依据 `docs/research/2026-10-05-acceleration-mechanisms.md` |
| audio VAE | 0.25s | 1.5% | 已 overlap，极致 |
| ppm + mux | 0.57s | 3.5% | x264 veryfast 已压到 ~0.23s（mux 单项） |
| **热 e2e 合计** | **~15.9s** | 100% | = 16.4s（C6/C7 stitch 基线）− 0.26s（x264 veryfast）− 0.25s（DiT TF32） |

**验收金线（480P 默认路径）：** latent checksum clip1=`0x28e8a880d52e9be1` / clip2=`0xce0b5df2bcdb4ad3`（bit-identical 档）；waffle b16 ∈ 1.0–1.7（clip2 金线区间）；av_sync PASS；sol 400/400 fallback=0。

### 0.2 768P 基线（1376×768 · 5s · TaoMate 2-stage · 官方 VAE · 混合臂：低分 sol + 高分 cudnn）

| 阶段 | 耗时（热 clip2） | 判定 |
| --- | ---: | --- |
| DiT（低分 4-step sol + 高分 4-step cudnn） | **~37-41s** | 主墙；高分 cudnn SDPA 是最大单段 |
| VAE（vae_blocks + unpatch_d2h + load） | **~14.7s** | vae_blocks ~9.9s + unpatch_d2h ~9.5s（部分与 blocks 并行记账，wall ~14.7s） |
| audio + ppm + mux | ~1.1s | 近极致 |
| **热 e2e 合计** | **~54-57s**（P50 56.47s） | 32GB VRAM 硬墙下全 sol OOM，混合臂是唯一稳定出片路径 |

### 0.3 已验证死路清单（防止重复踩坑，详见 §5 不做清单）

> 口径：下列「死」指**特定实现路径**已闭环，不等于**方向**关闭——每条注明根因与未关闭的相邻方向。

sol KV fuse（实现死：融合 bit-identical 但零收益，reduce 已访存饱和；相邻方向：窗口外 aux 核/launch 开销未关闭，属 [审计依赖] T1-e 范围）· INT8 autotune（cublasLt 无备选核）· FFN epilogue vec4（实现死：vec4 向量化撞生产指针非对齐坑致整管发散，根因是对齐假设；**未关闭方向：GEMM+epilogue 融合本身**——非 vec4 的实现路径待审计，quant/dequant 中转 1.465s 仍开放）· 手写 sm_120 INT8 GEMM（实现死：cublasLt 已在硬件峰值 98-99%，「更快的 INT8 kernel」数学上无余量；**未关闭方向：消除 GEMM↔dequant/swiglu 内存往返**——这是链路网关问题不是 GEMM 本征问题）· mux overlap（管道本已流式）· VAE tile batch 4→8（稳定 −0.36s 回归）· 768P 全 sol（32GB OOM 墙，native INT8 窗+激活 28.8GiB）· VAE 常驻 keep（与 INT8 cache 冲突 OOM）。

---

## 1. 目标体系（四档 SKU）

四档对应**不同业务场景**，不是一档通吃。档位之间通过 **opt-in 插件开关**组合实现，默认路径零行为变化。

| 档 | 名称 | 业务场景 | 具体目标（可测量） | 验收门 |
| --- | --- | --- | --- | --- |
| **T1** | 极速档 | 预览/草稿/批量抽卡、交互式调 prompt，速度优先、画质可让位 | 480P 热 e2e **≤13.5s**（预估可达区间 13.0–13.8s） | waffle ≤2.5（放宽档）+ av_sync PASS + 抽帧人眼可接受；**豁免 bit-identical**，记录 latent 漂移量 |
| **T2** | 均衡档（**默认产品**） | 正式出片主力，画质与金线逐位一致前提下的最快 | 480P 热 e2e **≤15.0s**（bit-identical-only，预估可达区间 14.6–15.2s） | latent checksum **逐位一致**（clip1/clip2 金线值）+ waffle 1.0–1.7 + av_sync PASS + sol 400/400 |
| **T3** | 质量档 | 精品/交付级内容，质量优先、不计秒数 | **目标不是秒**——waffle 稳定 ≤1.35（DMAD full_critic 实测区间 0.79–1.34，优于 turbo4 的 1.04–1.71）且 composition 人审通过；速度容忍 ≤ +15%（实测 dit +11%） | waffle 区间 + av_sync PASS + **人工 review 签字**；SKU 隔离，**永不并入 golden** |
| **T4** | 分辨率档（768P） | 高清正式内容 | 768P 热 e2e **≤50s**（预估可达区间 47–50s；从 P50 56.47s 起） | latent checksum 逐位一致（混合臂基线值）+ waffle 0.75–1.05 + av_sync PASS + 不出 OOM、不减窗（≥24） |

**目标定档论证（为什么是这个数，而不是更激进的数）：**

- **T1 不定 ≤12s：** 见 §3.1 逐点求和——全部已知+外部已证手段叠加后落在 12.9–13.8s，其中 QKV merge 和 VAE 流水两项外部收益尚未在本仓验证。承诺 12s 需要这两项同时打到收益区间上限，且还要再从已证 98-99% 硬件峰值的 INT8 GEMM 里挤出 1s——**数学不闭合，故不定 12s**。**严禁承诺 10s**（bit-identical 已证不可达；精度档全叠加也闭不合，见 §3.1）。
- **T2 不定更低：** bit-identical 约束下，已落地 15.9s 里只剩 VAE 流水（1.3–1.6s 预估，属 VAE 族#6 零成本浪费子集 d2h+fill ~2.3s 之内）一把已证原理的刀；其余死路全部经实测闭环。15.0s 已要求 VAE 流水打到收益区间上限附近，**再低必须依赖 [审计依赖] 重开死路**，故定 ≤15.0s。
- **T4 不定 ≤45s：** 45s 要求 768P DiT 段再省 ~5s，而高分全 sol 已撞 32GB OOM 墙（§0.3）、INT8 GEMM 无峰值余量；已知手段合计只能省 ~6.5-9.5s（§3.4），落在 47–50s。**KPI-B ≤27s 维持「当前不可达」判定**，不写入任何档位目标。

---

## 2. 逐点归因表（核心：每档优化哪些点）

> 列含义：**优化点**（函数/文件级定位）→ **当前状态** → **目标** → **手段（刀）** → **预期收益（秒，区间；「实测」= 238 已验证，「预估」= 外部已证原理/本仓未落地）** → **风险/精度代价** → **验证门**。

### 2.1 T1 极速档（480P ≤13.5s，从 15.9s 起，需 −2.4s）

| # | 优化点 | 当前状态 | 目标 | 手段（刀） | 预期收益 | 风险/精度代价 | 验证门 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1-a | DiT QKV 三次瘦 GEMM → 一次宽 GEMM + QKNorm+RoPE 融合（`h3_dit_w4a8.c` attn 前段 / `h3_gpu_cuda.cu` qkv_rope 0.25s+adaln 0.17s） | 3 次独立 QKV-proj INT8 GEMM/层（每层独立 d_q/d_k/d_v，§1.6.1 C1c 已证）；QKNorm+RoPE 为独立核 | 合并后每步层 GEMM 数 16→14；RoPE 零独立核 | **QKV merge**（K 维拼宽，tensor-core 利用率升）+ **QKNorm+RoPE 融合核**（NVIDIA RTX5090 pillar1 同款，外部已证） | **−0.6 ~ −1.2s（预估）**，依据：nsys DiT 窗口内 GEMM launch/访存间隙 + rope 0.42s + 合并后 K 维增大的 IMMA 效率提升；外部 5090 kernel 阶梯 1.08× | 中：INT8 合并 GEMM 若换累加顺序则**非 bit-identical**（T1 允许，记漂移）；融合核需逐层对 MAE | waffle ≤2.5 + latent 漂移量记录 + av_sync PASS；无需 checksum 一致 |
| T1-b | VAE unpatch_d2h 串行尾 ~1.6s（`h3_vae_int8_cuda.c` decode_loop 逐 tile 串行 D2H + 主机 stitch） | stitch pipeline 已省 1.2s 后仍剩 ~1.6s 串行尾；pinned 探索曾因 `fr->rgb` 生命周期 segfault 被否 | 串行尾 ≤0.3s | **pinned + async D2H + 与下一 tile GPU decode 的 stream 并行**（外部 VDN 清单第 2 条已证方向；须把 pinned 标志穿进 `h3_vae_int8_frames` 生命周期——§8.6 遗留改造） | **−1.3 ~ −1.6s（预估）**，依据：串行尾全长 1.6s 中 staging/等待占绝大部分，pinned 只省 staging 上限已估 2-3s@768P（480P 等比 ~1.3s） | 中高：frames 生命周期改造（上次 segfault 点）；async 竞态（`H3_VAE_DECODE_PIPELINE` 已证时序敏感）→ 用 per-slot done-event 替换全局 quiesce | latent checksum 逐位一致（此刀 bit-identical 可达）+ waffle 1.0–1.7 |
| T1-c | x264 ultrafast 档（`tools/h3_cuda_generate.c` mux，`H3_X264_PRESET`） | veryfast 已落地 −260ms（实测） | mux ≤0.15s | veryfast→**ultrafast**（再省 ~100ms 实测） | **−0.1s（实测）** | 画质损失明显（实测判定不推荐用于正式档）——**仅 T1 预览场景可接受** | 抽帧人眼可接受即可 |
| T1-d | DiT TF32 精度档（`H3_DIT_GEMM_TF32`） | 已落地 −250ms（实测），已含在 15.9s 基线内 | 保持开 | 无新增 | 0（已在基线） | 非 bit-identical（T1 允许） | waffle ≤2.5 |
| T1-e | **[审计依赖]** sol aux / epilogue / INT8 死路重开 | 全部 dead-end-verified（§0.3） | 若审计推翻某条，按新结论补 | 待审计结论 | **未知（下限 0，上限待审计；reference：quant/dequant 中转 1.465s，回收一半即超 −0.5s）** | 未知 | 以审计报告为准 |

**T1 求和（§3.1 详算）：** −0.6~−1.2（a）−1.3~−1.6（b）−0.1（c）= **−2.0 ~ −2.9s** → 15.9 −(2.0~2.9) = **13.0–13.9s**。目标定 **≤13.5s**（要求两刀都落在区间中位以上）。

### 2.2 T2 均衡档（默认产品，480P ≤15.0s bit-identical，从 16.4s bit-identical 基线起）

> 注意口径：T2 的基线是 **16.4s**（stitch pipeline，bit-identical），不是 15.9s——15.9s 里含的 DiT TF32（−0.25s）非 bit-identical，不属于 T2。x264 veryfast −0.26s 实测 checksum 逐位一致，属 T2。

| # | 优化点 | 当前状态 | 目标 | 手段（刀） | 预期收益 | 风险/精度代价 | 验证门 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T2-a | VAE unpatch_d2h 串行尾 ~1.6s | 同 T1-b | 串行尾 ≤0.3s | **同 T1-b：pinned+async+stream 并行**（此刀本身 bit-identical，T1/T2 共享同一次施工） | **−1.3 ~ −1.6s（预估）** | 同 T1-b（生命周期改造+竞态），**T2 门槛更高：必须 checksum 逐位一致才算成** | clip1/clip2 金线 checksum 逐位一致 + waffle 1.0–1.7 + av_sync PASS |
| T2-b | x264 veryfast（`H3_X264_PRESET=veryfast`） | 已落地（实测 mux −260ms，checksum 双精确） | 保持开并**考虑进默认** | 无新增施工 | **−0.26s（实测）** | 画质在质量地板内（实测判定）；注意它改的是封装非 latent，故 latent checksum 不变 | latent checksum 逐位一致（已验） |
| T2-c | **[审计依赖]** C1 sol aux（0.09s/clip 真实权重）等死路重开 | dead-end-verified | 待审计 | 待审计 | **0 ~ −0.1s（未知，量级小）** | 未知 | 审计报告 + checksum 逐位一致 |

**T2 求和：** 16.4 −0.26（b，实测）−1.3~−1.6（a，预估）= **14.5–14.8s**。目标定 **≤15.0s**——即使 VAE 流水只打到下限 −1.3s 也达标，留 ~0.3s 噪声余量。**注意：若 VAE 流水因生命周期/竞态问题最终只能部分落地（如只做到 −0.8s），T2 目标需下修至 ~15.3s，届时改目标不改口径。**

### 2.3 T3 质量档（目标 = 质量尺，不是秒）

| # | 优化点 | 当前状态 | 目标 | 手段（刀） | 预期收益 | 风险/精度代价 | 验证门 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T3-a | DiT 后端换 DMAD full_critic（iter1600，ZhengmingYu/DMAD，PEFT 键转换 max\|err\|=1.1e-08） | 实测 waffle 0.79–1.34（优于 turbo4 1.04–1.71），composition 更干净；**无 speedup（dit +11%）**；480P 双臂 av_sync/sol 全 PASS；成片 `kpib-dmad-kj/C_full_critic/` 待人审 | waffle 稳定 ≤1.35 且人审签字 | **SKU 隔离后端**：`--lora <DMAD full_critic>` 可选臂，永不 merge 进 golden | 质量：waffle 区间下移 ~0.3（实测）；速度：**−11%（负收益，实测）** | 换学生权 = 换轨迹（合同内 SKU 隔离）；人工 review 未通过则整档不发布 | ① 人工 review 签字（成片已在 review 目录）② waffle ≤1.35 全 clip ③ av_sync PASS ④ sol 400/400 |
| T3-b | waffle 稳定性护栏（`H3_WAFFLE_GUARD` + `H3_WAFFLE_GUARD_RETRY`，GUARD stage 已落地） | 已验证：armed=3.0 时 clip1=3.741 触发 WARN（cold-clip 间歇高噪仍活跃实证） | cold-clip 高噪（b16>3）自动告警+重试消除 | 全帧 max_b16 聚合 + 阈值告警 + 可选重试（bit-identical，已注册进插件表） | 质量稳定性：消除间歇性高噪片（实测触发过）；速度：重试时 +1 次 VAE decode | 重试掩盖根因——仅作护栏，cold-clip 高噪根因（VAE decode）另行立项 | WARN 触发率统计 + 重试后 waffle 回落验证 |
| T3-c | **[阻塞] cold-clip VAE 间歇高噪根因** | clip1 waffle 2.7–8.8 逐次漂移（基线固有，T3 人审的最大风险） | 定位并消除 | 待立项：VAE decode 数值路径审计（怀疑 host dequant/unpatch 边界） | 未知 | 未知 | waffle clip1 ≤2.0 稳定 |

**T3 达标判定：** T3-a 人审通过 + T3-b 护栏上线 = 质量档可用；T3-c 是质量档的已知残余风险，不阻塞发布但必须在交付文档标注。

### 2.4 T4 分辨率档（768P ≤50s，从 P50 56.47s 起，需 −6.5s）

| # | 优化点 | 当前状态 | 目标 | 手段（刀） | 预期收益 | 风险/精度代价 | 验证门 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T4-a | VAE unpatch_d2h 串行尾 ~9.5s（768P，`h3_vae_int8_cuda.c`） | 占 VAE wall ~2/3；pinned 探索 segfault 被否（同 480P 生命周期问题） | 串行尾 ≤4s | **同 T1-b 的 pinned+async+stream 流水，768P 放大版**（6 批 × ~0.35GiB/批 D2H） | **−3.5 ~ −5.5s（预估）**，依据：§8.6 已估 pinned 上限 2-3s + stream 并行消等待；串行尾全长 9.5s | 高：同生命周期改造 + 768P 单批 0.35GiB 的 pinned 内存压力（host 侧） | latent checksum 逐位一致（混合臂基线 `0x63987d3e8083e8c1`）+ 不 OOM |
| T4-b | VAE blocks GEMM ~9.9s（36 transformer block 逐 chunk） | tile batch 4→8 已证无收益（−0.36s 回归，死路） | −1.5s | **tile 批间 CUDA stream 并行（2 流）**——§8.6 结论明确的下一刀候选；VRAM 热 clip free≈22GiB 充足 | **−1.5 ~ −2.5s（预估）**，依据：blocks 9.9s 中 GEMM/SDPA 串行 6 批，2 流理论近半；打七折估 | 中：激活 workspace ×2，须 fit 检查；SDPA per-tile 形状在双流下的 cuDNN 句柄竞争 | checksum 逐位一致 + 不 OOM + waffle 0.75–1.05 |
| T4-c | vae_load ~0.3s（prewarm 余尾，`h3_cuda_vae_weights_load` dequant 249ms + h2d 24ms） | §8.7 已登记：host 暖载全隐藏，device 尾仍串行 | 0 | prewarm 线程预填 w_dev 或 host_pack restore 修复 | **−0.3s（预估）** | 中：多占一份设备显存——768P 高分段 32GB **无 headroom**，可能不可行；**降级选项：接受余项** | 不 OOM 前提下 checksum 一致 |
| T4-d | 低分 sol 红利（混合臂已含，低分 sol + 高分 cudnn） | 已落地（混合臂 vs 纯 cudnn 快 ~2s 量级，但机器噪声 ±3s 内） | 保持 | 无新增 | 0（已在基线） | — | sol hits=300/300 |
| T4-e | **[审计依赖]** 高分全 sol | 32GB OOM 墙已证（native INT8 窗+激活 28.8GiB，sol fwd 连 530MiB 挤不出） | 若审计给出新显存账本 | 待审计 | **0 ~ −4s（未知，高不确定）** | 高：上次三版释放策略全 OOM | 出片 + 热 total 实测 |

**T4 求和（§3.4 详算）：** −3.5~−5.5（a）−1.5~−2.5（b）−0.3（c）= **−5.3 ~ −8.3s** → 56.5 −(5.3~8.3) = **48.2–51.2s**。目标定 **≤50s**（要求 a、b 两刀都落地且至少打到区间中位）。**若 T4-c 因显存不可行，目标仍可达（a+b 下限 −5.0s → 51.5s 触边，故 T4-c 列为必须尝试项）。**

---

## 3. 可达性论证（诚实数学）

### 3.1 T1 ≤13.5s 的闭合账

```
基线 15.9s（实测：16.4 stitch − 0.26 x264veryfast − 0.25 DiT TF32）
− T1-a QKV merge + QKNorm/RoPE 融合   −0.6 ~ −1.2s  [预估，外部已证原理]
− T1-b VAE unpatch_d2h 流水           −1.3 ~ −1.6s  [预估，外部已证原理]
− T1-c x264 ultrafast                 −0.1s          [实测]
─────────────────────────────────────
合计节省                               −2.0 ~ −2.9s
落点                                   13.0 ~ 13.9s
```

- **达标条件：** T1-a、T1-b 均落地且合计 ≥ −2.4s（即两刀都在区间中位以上）。**两刀都是 [外部已证、本仓未落地]**，这是 T1 目标的最大风险点——若任何一刀落地后收益不足区间下限的 2/3，T1 目标下修。
- **为什么不定 12s：** 12s 要求 −3.9s，超出全部已知手段合计上限（−2.9s）1 秒整；这 1 秒只能来自 INT8 GEMM 本征（已证 98-99% 硬件峰值，无余量）或减步/换模型（违反质量地板、且属改合同）。**数学不闭合，不定。**
- **为什么严禁 10s：** bit-identical 下已证不可达（§0.3 死路链）；精度档全叠加后距 10s 仍差 ~3s，缺口在 INT8 GEMM 本征 3.59s——C5b 三层独立验证（ISA/硬件峰值/cublasLt 实测）已关闭。**边界条件（假设显式化）：此判定在 T1-e 上限=−0.5s 的假设下成立；若方法论审计重开死路的真实收益超过 −0.5s（quant/dequant 中转 1.465s 提示可能，见 T1-e），则 10–12s 的判定需重估。** 在审计结论出来前，10s 维持不可达判定，除非换 NFE 结构或换精修模型（= 改质量尺 = 另一份合同）。

### 3.2 T2 ≤15.0s 的闭合账

```
基线 16.4s（bit-identical，stitch pipeline 实测）
− T2-b x264 veryfast    −0.26s  [实测，checksum 双精确]
− T2-a VAE d2h 流水     −1.3 ~ −1.6s  [预估；bit-identical 可达，但工程风险中高]
─────────────────────────────────────
落点                    14.5 ~ 14.8s
```

- **达标条件：** 实质只有一刀——VAE 流水，且必须做到 **checksum 逐位一致**（比 T1 更严）。`H3_VAE_DECODE_PIPELINE` 已证全异步有竞态，故施工方案定为 **per-slot done-event 就绪追踪**（修订记录已指明的后续方向）。
- **诚实声明：** 这是**单点依赖目标**。若 VAE 流水最终证明无法在 bit-identical 下超过 −0.8s，T2 目标下修至 15.3s 并改目标书，**不硬撑**。

### 3.3 T3 的闭合账（质量尺）

- waffle：turbo4 1.04–1.71 → DMAD full_critic 0.79–1.34（**实测已达标**，差距 −0.25~−0.37）。
- 未闭合项：**人工 review 未签字**（成片在 `out/review-2026-10-04/kpib-dmad-kj/`）。T3 发布 = 人审签字，这是流程门不是技术门。
- 速度代价 dit +11%（实测）已写入 SKU 合同，购买方知情。

### 3.4 T4 ≤50s 的闭合账

```
基线 56.47s（30-repeat P50，混合臂）
− T4-a VAE d2h 流水(768P)   −3.5 ~ −5.5s  [预估]
− T4-b VAE blocks 2-stream  −1.5 ~ −2.5s  [预估]
− T4-c vae_load 余尾        −0.3s          [预估，可能因显存不可行]
─────────────────────────────────────
合计                        −5.3 ~ −8.3s
落点                        48.2 ~ 51.2s
```

- **达标条件：** T4-a + T4-b 合计 ≥ −6.5s，即两刀都要打到区间中位。**两刀同落在一个文件（`h3_vae_int8_cuda.c`）的相邻路径，可一次施工拍完成，但必须分臂独立验证**（纪律：每拍只动一个变量）。
- **不定 45s 的理由：** 45s 需 −11.5s，超出已知手段上限 3.2s；缺口只能在高分 DiT（37-41s 主墙），而高分全 sol 已证 32GB OOM。**[审计依赖]：若方法论审计推翻全-sol OOM 结论或 INT8 窗账本，T4 可重估；在审计结论出来前，45s 不写入目标。**
- **27s（KPI-B 原目标）维持不可达判定：** 27s 需 −29.5s（−52%），等于把 DiT+VAE 双双砍半再砍半；所有已验证手段合计 −8.3s 上限。**不重新诉讼。**

---

## 4. 优先级与排序

**排序原则：** 置信度 × 收益 ÷ 风险；先共享刀（一刀喂多档），后独享刀；阻塞项显式列出。

### 4.1 施工顺序（P0 → P3）

| 序 | 刀 | 服务档位 | 置信度×收益 | 风险 | 前置条件 |
| --- | --- | --- | --- | --- | --- |
| **P0** | **VAE unpatch_d2h pinned+async+stream 流水**（一刀同喂 T1/T2/T4：480P −1.3~1.6s + 768P −3.5~5.5s）——**作用域限定：收益落在 VAE 4.96s 的族#6 零成本浪费子集内（d2h 串行尾 + FillFunctor ≈ 2.3s，bit-identical 可达）；不触碰 vae_blocks 主体 ~2.6s 的 fp16 真实计算**（归因依据 `docs/research/2026-10-05-acceleration-mechanisms.md`） | T1、T2、T4 | **最高**（原理外部已证 + 本仓串行尾已精确归因；收益占三档目标缺口的大头） | 中高（frames 生命周期 segfault 前科 + 异步竞态前科）→ 施工方案已定：pinned 标志穿进 `h3_vae_int8_frames` + per-slot done-event | **需要独占 GPU**（竞态调试对时序敏感，拥挤 GPU 下位精确验证已失败过一次） |
| **P1** | **QKV merge + QKNorm+RoPE 融合**（T1-a） | T1（主）、T2（若做到 bit-identical 则也喂） | 高（NVIDIA pillar1 外部已证 + 本仓 nsys 已定位 rope/adaln 0.42s + QKV 三瘦 GEMM 结构已确认） | 中（INT8 合并 GEMM 的累加顺序 → 先做 FP 路径验证 MAE，再定 INT8 是否进 T2） | 无硬前置；与 P0 并行（不同文件） |
| **P2** | **VAE blocks 2-stream**（T4-b，768P 独享） | T4 | 中高（§8.6 已明确指向此刀；VRAM 充足已实测） | 中（激活 workspace ×2 的 fit 检查） | P0 落地后（同一文件，避免在飞冲突） |
| **P3** | x264 ultrafast（T1-c）/ vae_load 余尾（T4-c） | T1 / T4 | 高收益小 | 低 / 中（T4-c 可能撞显存墙，允许失败降级） | 无 / P0 后 |
| **P2b** | **CUDA Graph 整步捕获**（新增，双源背书） | T1、T2 | 中高（sol-h3 官方 Spark 1.39× 明确含 graph capture + 12-方法深拆确认本仓 sol aux 12232 小核实例 launch-bound；预估 −0.5~−1.5s——**口径注意：官方重分析给的是「DiT 侧 3–8%」≈0.3~0.8s，−1.5s 上限偏乐观，取双源区间**） | 中（静态形状约束→每 shape 一图；与 INT8 窗重载交互需验证）**纯无损 bit-identical 可达** | 无硬前置；与 P1 并行 |

### 4.1b 质量验证与机制演进层（2026-10-05 交叉验证新增，与 P 序列并行）

> 这一层不是速度刀，是**主栈成立性验证 + 质量补齐 + 演进方向**。来源：4 个官方/社区一手调研代理交叉核对（`docs/research/2026-10-05-acceleration-mechanisms.md` 附录）。

| 序 | 项 | 类型 | 服务档位 | 依据 | 风险/前置 |
| --- | --- | --- | --- | --- | --- |
| **V0** | **sol_cute @ turbo4 稀疏有效性复核** | ⚠️ **主栈成立性** | T1、T2（默认产品） | **实质冲突**：FastWan/Hao AI Lab 称「训练无关稀疏在 <10 步蒸馏下急剧退化」——我们 turbo4(4-step)+sol_cute（推理期稀疏）正中此条。480p hits=400/400 只证路由在跑**未证质量无损**（sol 官方甜区是 50-step/768p 长序列） | **最高优先级验证项**：sol_cute vs 全 dense 的 waffle/人眼 A/B（同 seed 同机）。**退化判定线（dev-plan 补充口径）：waffle 均值超对照 >0.15，或漂出金线区间且人眼可见**——执行时可再校准。若退化 → sol_cute 只能留 8-step 金线档或需 SLA2 式稀疏感知微调 |
| **V1** | **sol MMDiT dense 护栏 A/B** | 质量护栏 | T2 | sol-h3 官方近无损背书来源=文本 query 行 dense + 前 2 层 dense + 前 10 步 dense；我们接了 sink 但 `FIRST_DENSE_*=0` 全稀疏。**这是 sol_cute 臂能否从 opt-in 挪到产品默认的关键** | 与 V0 同批做（都是 sol 质量门）；低成本 |
| **V2** | **taomate 当 0.7 强度 refiner 收尾 A/B** | 质量补齐 | T1、T3 | 社区高票用法（WhichH3/Reddit）：高步主体 + taomate 3-step @0.7 强度收尾，**治的正是 turbo4 的「overcooked」过饱和/糊**。整段 3-step 盲测垫底，但 refiner 用法口碑极好 | 坑：①基座版本强耦合（stock pruned 保色，Singularity 掉色）②`h3_lora_parse_name` 拒绝 `token_refiner` 模块→用 Kijai rank-19 版 |
| **V3** | **细节恢复 LoRA 调研**（Z-Image DistillPatch 类） | 演进方向 | T3 | 图像侧已验证「蒸馏轨迹对齐的细节恢复 LoRA」、视频/H3 侧空白；与 V2 同向（都是治蒸馏过饱和） | 调研项非施工项；待 V2 结论定是否立项 |
| **V4** | **SLA 式稀疏+线性混合叠 sol_cute** | 演进方向 | T4（768P） | VDN 保运动>FastH3 被社区独立复现 + 线性注意力 O(N) 恒定内存命中 768P OOM 墙；marginal 块加线性分支（成本<0.5%，稀疏度 70%→95%） | 需少量微调；768P 显存墙突破后才成立，低优先 |

### 4.2 阻塞项（不施工，等外部输入）

| 项 | 等谁 | 解锁后动作 |
| --- | --- | --- |
| T3-a DMAD full_critic 发布 | **人工 review 签字**（成片 `out/review-2026-10-04/kpib-dmad-kj/C_full_critic/`） | 签字 → SKU 隔离臂上线；不签 → T3 整档冻结 |
| T1-e / T2-c / T4-e 死路重开 | **方法论审计报告** | 逐条对照 §0.3 死路清单：推翻哪条就重估哪条所属档位的目标；未推翻则目标书不变 |
| T3-c cold-clip VAE 高噪根因 | 新立项（VAE 数值路径审计） | 不阻塞任何档位发布；T3 交付文档必须标注此残余风险 |
| T4-e 高分全 sol | 审计报告 + （若重开）32GB 显存新账本 | 推翻 OOM 结论 → T4 目标重估至 ≤45s 档；未推翻 → 维持 ≤50s |
| **V0 sol_cute@turbo4 质量复核** | **独占 GPU + 同 seed A/B 窗口**（sol_cute vs 全 dense 的 waffle/人眼） | **若退化** → sol_cute 限 8-step 金线档 / 启动 SLA2 式稀疏感知微调调研；**若无损** → 主栈成立性确认，V1 dense 护栏 A/B 顺推 |

---

## 5. 不做清单（anti-goals，防止重新诉讼）

| 不做什么 | 为什么（实测证据） | 文档指针 |
| --- | --- | --- |
| **不承诺 480P ≤10s** | bit-identical 下死路链闭环；精度档全叠加仍差 ~3s，缺口在 INT8 GEMM 本征（C5b 三层验证关闭：sm_120 无新 INT8 指令、硬件峰值 745.7 TOPS 实测、cublasLt 已达 98-99%） | 480p-10s-plugin-plan §1.8/§5 C5b 条 |
| **不承诺 768P ≤27s（KPI-B）** | 需 −52%；已知手段上限 −8.3s；高分全 sol 撞 32GB OOM（native INT8 窗+激活 28.8GiB，三版释放策略全败） | sol-h3-analysis §8.4/§8.5 |
| **不再做 INT8 GEMM raw-FLOPS 优化**（autotune/手写核/epilogue 减往返） | cublasLt 对全部 8 形状只回 algos=1（C5a 实测 Δ=0.000ms）；手写 sm_120 核数学上无 15% 余量（C5b）；FFN epilogue vec4 撞生产指针非对齐致整管发散（C2b） | 同上 §1.8/§1.9/§5 |
| **不做 sol KV fuse / sol aux 压缩（DiT 窗口内实现路径）** | 本条关闭的是**窗口内**融合/压缩实现：DiT 窗口内真实权重仅 0.09s/clip（nsys 按 stream 分桶修正）；KV fuse 融合 bit-identical 但零收益（reduce 已访存饱和）。**口径注意：T1-e [审计依赖] 的「sol aux 死路重开」范围是窗口外 aux 核、launch 开销与 quant/dequant 中转（1.465s 量级），与窗口内 0.09s 是两个不同口径，不矛盾** | §1.6.1 |
| **不做 mux overlap** | 编码管道本已流式（音/视双线程喂 ffmpeg），实测无净增益 | §5 C8 条 |
| **不做 VAE tile batch 4→8** | 实测稳定 −0.36s 回归（L2 命中下降抵消行合并收益） | sol-h3-analysis §8.6 |
| **不做 VAE 常驻 keep** | 与 INT8 cache 冲突，clip2 adaln OOM（window 2/3/8 全崩） | §5 C6/C7 诊断条 |
| **不减窗（<24）、不假 sage3、不 TAE/LTX、不动质量尺** | 合同红线 | 各计划文档头部 |
| **不把 DMAD 并入 golden、不宣称 DMAD 有加速** | 换学生权=换轨迹；实测 dit +11% 负收益；SKU 隔离专属 | §5 DMAD 条 |
| **[审计依赖] 例外条款** | 若方法论审计以新证据推翻上表任一条，该条移出不做清单并进入对应档位重估——**仅以审计报告为准，不接受口头重开** | 本文件 §4.2 |

---

## 6. 验收协议（每档达标的判定流程）

1. **每拍只动一个变量**，对照臂与实验臂同 seed、同机器、同 binary（仅开关差异）。
2. **速度门：** 热 clip2 `phase_ms total=` × ≥2 次复现，run-to-run 噪声 ±0.5s（480P）/ ±3s（768P）内取优臂。
3. **质量门：** T1=waffle ≤2.5+抽帧人眼；T2/T4=latent checksum 逐位一致+waffle 金线区间；T3=waffle 区间+人审签字。
4. **稳定性门：** av_sync PASS、sol hits 满额 fallback=0、无 OOM、repeat=2 连跑 exit=0。
5. **成片归档：** `out/review-<date>/` 待人工确认；日志落 `docs/superpowers/plans/` 修订记录。
6. **达标判定：** 档位目标 = 速度门 ∧ 质量门 ∧ 稳定性门同时通过。**任一门不过，目标自动下修并修订本文件，不倒改口径。**

## 7. 修订记录

| 日期 | 变更 |
| --- | --- |
| 2026-10-05 | 初稿：四档目标体系（T1 ≤13.5s / T2 ≤15.0s bit-identical / T3 waffle ≤1.35+人审 / T4 ≤50s）+ 逐点归因表 + 可达性数学 + 优先级排序 + 不做清单。全部数字锚定 238 实测账本（§0）；预估项全部标注依据与外部出处。方法论审计结论为 pending 输入，相关条目标 [审计依赖]。 |
| 2026-10-05 | **交叉验证修订**（4 个官方/社区一手调研代理返回后）：① §4.1 新增 **P2b CUDA Graph 整步捕获**（sol-h3 官方 Spark 1.39× 含 graph capture + 本仓 sol aux 12232 小核 launch-bound 双源背书，预估 −0.5~−1.5s，纯无损 bit-identical 可达）。② 新增 **§4.1b 质量验证与机制演进层**（V0–V4）：V0=sol_cute@turbo4 稀疏有效性复核（⚠️ 主栈成立性，FastWan「训练无关稀疏<10 步蒸馏退化」实质冲突）；V1=sol MMDiT dense 护栏 A/B（官方近无损背书来源）；V2=taomate 0.7 强度 refiner 收尾 A/B（治 turbo4 过饱和）；V3=细节恢复 LoRA 调研；V4=SLA 式稀疏+线性混合。③ §4.2 阻塞项登记 V0。**速度档位目标（T1–T4）未变**——新增项全是质量验证/演进方向，不改任何秒数承诺。依据：`docs/research/2026-10-05-acceleration-mechanisms.md` 附录 + 4 份一手调研报告。 |
| 2026-10-05 | **开发方案对齐修订**（dev-plan 落盘后）：① P2b 收益区间补口径——官方重分析给「DiT 侧 3–8%」≈0.3~0.8s，本文件 −1.5s 上限偏乐观，改为标注双源区间。② V0 退化判定线补量化阈值（dev-plan 建议：waffle 均值超对照 >0.15 或漂出金线区间且人眼可见，执行时可校准）。③ 施工级落地文档 `2026-10-05-h3-optimization-dev-plan.md` 已落盘（8 节，含代码级定位/施工步骤/验证门/风险回退/依赖图/里程碑），本文件为目标书、dev-plan 为施工手册，两者配合。 |
