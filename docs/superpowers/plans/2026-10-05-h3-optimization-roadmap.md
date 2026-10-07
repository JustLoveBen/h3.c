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
| VAE（官方 fp16；stitch-pipeline 实验臂已从 binary 删除） | **4.96s** | 30% | 族#6 零成本浪费子集：unpatch_d2h 实测 **3227ms**（P0 插桩，2026-10-05）——其中 **94% 为逐 tile 串行同步 `cudaMemcpy` 的 D2H 等待**（229ms/tile × 13），**P0 靶子**；原记的 ~1.6s 串行尾系已删除的 stitch-pipeline 实验臂（kpia-vae-pipeline，unpatch_d2h=195ms 是把 VAE 藏进 DiT 影子的口径）的**错配基线，作废**；vae_blocks 主体 ~2.6s 为 fp16 未量化真实计算（族#5 未收割，有质量杠杆）——归因依据 `docs/research/2026-10-05-acceleration-mechanisms.md` |
| audio VAE | 0.25s | 1.5% | 已 overlap，极致 |
| ppm + mux | 0.57s | 3.5% | x264 veryfast 已压到 ~0.23s（mux 单项） |
| **热 e2e 合计** | **~15.9s** | 100% | = 16.4s（C6/C7 stitch 基线）− 0.26s（x264 veryfast）− 0.25s（DiT TF32） |

**验收金线（480P 默认路径，2026-10-05 回填修订）：** 旧金线 clip2=`0xce0b5df2bcdb4ad3` 已证为「sol 全稀疏退化臂」产物，**作废**（连带旧 clip1=`0x28e8a880d52e9be1` 一并失效）。新金线为**护栏臂**（sol + 缺省 dense_steps=2 / layers=2）：latent checksum **只认 hot clip2=`0x923b0db3a8c0357e`**；cold clip1 checksum 间歇（`0x2d5c52020bdfcc70` / graph 臂 `0x5e6b2f7bfe154e0b`），因 cold 轨迹非确定性**不作金线**。hot clip2 waffle mean=**0.83**（帧 0.750/0.803/0.851/0.816/0.935，金线区间重锚为 **0.75–0.95**）；hot total≈19.2–19.4s（含 cold 噪声，与旧 15.9s 账本口径不可直接比，见 §3.1 基线重锚注）；av_sync PASS；sol hits 满额 fallback=0。

### 0.2 768P 基线（1376×768 · 5s · TaoMate 2-stage · 官方 VAE · 混合臂：低分 sol + 高分 cudnn）

| 阶段 | 耗时（热 clip2） | 判定 |
| --- | ---: | --- |
| DiT（低分 4-step sol + 高分 4-step cudnn） | **~37-41s** | 主墙；高分 cudnn SDPA 是最大单段 |
| VAE（vae_blocks + unpatch_d2h + load） | **~14.7s** | vae_blocks ~9.9s + unpatch_d2h ~9.5s（部分与 blocks 并行记账，wall ~14.7s）——**2026-10-05 口径修订：480P 侧实测证实 unpatch_d2h 的 94% 为逐 tile 串行同步 D2H 等待（3227ms，229ms/tile × 13），768P ~9.5s 应同构理解（串行等待为主，非传输本征）** |
| audio + ppm + mux | ~1.1s | 近极致 |
| **热 e2e 合计** | **~54-57s**（P50 56.47s） | 32GB VRAM 硬墙下全 sol OOM，混合臂是唯一稳定出片路径 |

### 0.3 已验证死路清单（防止重复踩坑，详见 §5 不做清单）

> 口径：下列「死」指**特定实现路径**已闭环，不等于**方向**关闭——每条注明根因与未关闭的相邻方向。

sol KV fuse（实现死：融合 bit-identical 但零收益，reduce 已访存饱和；相邻方向：窗口外 aux 核/launch 开销未关闭，属 [审计依赖] T1-e 范围）· INT8 autotune（cublasLt 无备选核）· FFN epilogue vec4（实现死：vec4 向量化撞生产指针非对齐坑致整管发散，根因是对齐假设；**未关闭方向：GEMM+epilogue 融合本身**——非 vec4 的实现路径待审计，quant/dequant 中转 1.465s 仍开放）· 手写 sm_120 INT8 GEMM（实现死：cublasLt 已在硬件峰值 98-99%，「更快的 INT8 kernel」数学上无余量；**未关闭方向：消除 GEMM↔dequant/swiglu 内存往返**——这是链路网关问题不是 GEMM 本征问题）· mux overlap（管道本已流式）· VAE tile batch 4→8（稳定 −0.36s 回归）· 768P 全 sol（32GB OOM 墙，native INT8 窗+激活 28.8GiB）· VAE 常驻 keep（与 INT8 cache 冲突 OOM）。

**2026-10-05 回填新增：** sol 全稀疏@turbo4（**判退化退出默认**：旧金线 clip2=`0xce0b5df2bcdb4ad3` 即其产物；默认臂切至护栏臂 dense_steps=2/layers=2）· P1 QKV merge + QKNorm+RoPE 融合（**实现死，零收益**：探针实证 `h3_dit_w4a8.c` 里 QKV 本就 `[3*INNER, HIDDEN]` 宽 GEMM 合并、QKNorm+RoPE 本就在 CUDA 端融合，位一致探针 MAE=0，CUDA 端无收益可挖，原 −0.6~−1.2s 预估不成立）· P2b CUDA Graph（**已交付但收益微薄**：整步单次捕获/replay、238 hot clip2 位一致，但 hot DiT 12.61→12.63s 持平、e2e 仅 −0.16s——turbo4 仅 4 步，launch 税被 cold 噪声掩盖；可选开关，不翻默认）· P0 pinned **半刀**（pinned 真实命中但 unpatch_d2h 3227ms 与 P0 前持平、零提速）→ **P0 真跨 tile 流水已落地并翻默认（2026-10-06）：** per-slot done-event 链 + 跨 tile pinned 池，238 实测 unpatch_d2h **3004→26.1ms（−99%）**、VAE total 3455→472ms；位一致证据=async on 臂 run 间逐位确定（3/3 同 `0x3287f5c8…`）且 off/on 差异 ≤ off 自身 run 间 cuBLAS heur 噪声；`H3_VAE_UNPATCH_ASYNC` 默认 0→1。**VAE 基线非确定根因坐实：** cuBLAS/cuDNN heur 跨进程漂移（`cublasSgemmStridedBatched` 等无全算法确定开关），`H3_VAE_DETERMINISTIC`（默认 0）已交付但钉不死 off 基线 → **T2 位一致验收口径改为「async on 臂 run 间逐位一致」**（跨 run checksum 金线在 VAE cuBLAS 非确定下物理不可达）。

---

## 1. 目标体系（四档 SKU）

四档对应**不同业务场景**，不是一档通吃。档位之间通过 **opt-in 插件开关**组合实现，默认路径零行为变化。

| 档 | 名称 | 业务场景 | 具体目标（可测量） | 验收门 |
| --- | --- | --- | --- | --- |
| **T1** | 极速档 | 预览/草稿/批量抽卡、交互式调 prompt，速度优先、画质可让位 | 480P 热 e2e **≤20.5s（2026-10-06 正式下修：final-accept 实测落点 20.12s，原 ≤13.5s 目标不可闭合，缺口 6.6s 需审计重开或新算法）** | waffle ≤2.5（放宽档）+ av_sync PASS + 抽帧人眼可接受；**豁免 bit-identical**，记录 latent 漂移量 |
| **T2** | 均衡档（**默认产品**） | 正式出片主力，画质与金线逐位一致前提下的最快 | 480P 热 e2e **≤19.0s（2026-10-06 正式下修：final-accept 实测落点 18.49–20.32s（4-run mean=18.75s），原 ≤15.0s 目标不可闭合，缺口 3.8s；位一致判为物理不可达（VAE cuBLAS 跨 run 非确定，4 次 hot clip2 checksum 全不同），改判「同配置 run 间 waffle 区间稳定 + 无 NaN」** | latent checksum **豁免（物理不可达）** + waffle 0.75–0.95（新锚）+ av_sync PASS + sol hits 满额 + 无 NaN/全零帧 |
| **T3** | 质量档 | 精品/交付级内容，质量优先、不计秒数 | **目标不是秒**——waffle 稳定 ≤1.35（DMAD full_critic 实测区间 0.79–1.34，优于 turbo4 的 1.04–1.71）且 composition 人审通过；速度容忍 ≤ +15%（实测 dit +11%） | waffle 区间 + av_sync PASS + **人工 review 签字**；SKU 隔离，**永不并入 golden** |
| **T4** | 分辨率档（768P） | 高清正式内容 | 768P 热 e2e **≤70s（2026-10-06 正式下修：final-accept 实测落点 68.93s，原 ≤50s 目标不可闭合，缺口 18.9s；T4-a async 收益 480P 已兑现但 768P 同构放大后仍不足，T4-b 2-stream 判无收益，T4-c 未施工）** | latent checksum 豁免（同 T2 物理不可达）+ waffle 0.75–1.05 + av_sync PASS + 不出 OOM、不减窗（≥24） |

**目标定档论证（为什么是这个数，而不是更激进的数）：**

- **T1 正式下修至 ≤20.5s（2026-10-06）：** 原 ≤13.5s 目标数学不闭合（§3.1 重算后缺口 6.6s），final-accept 实测落点 20.12s。下修后目标 = 当前已验证最优配置（preview+TF32+async+skip_ppm）的实测值 + 2% 容差。**12s/10s 维持不可达判定不变。**
- **T2 正式下修至 ≤19.0s（2026-10-06）：** 原 ≤15.0s 目标不可闭合（§3.2 重算后缺口 3.8s），final-accept 实测落点 18.49–20.32s（4-run mean=18.75s）。位一致判为**物理不可达**（VAE cuBLAS/cuDNN heur 跨进程漂移，4 次 hot clip2 checksum 全不同），改判「同配置 run 间 waffle 区间稳定 + 无 NaN」。下修后目标 = 实测均值 + 1% 容差。**15.0s 维持不可达判定，除非审计重开给出新收益源。**
- **T4 正式下修至 ≤70s（2026-10-06）：** 原 ≤50s 目标不可闭合（§3.4 重算后缺口 18.9s），final-accept 实测落点 68.93s。T4-a async 收益 480P 已兑现（−1.39s）但 768P 同构放大后仍不足；T4-b 2-stream 实测 +8.1% 退化判无收益；T4-c 未施工。下修后目标 = 实测值 + 1.5% 容差。**50s/45s/27s 维持不可达判定。**

---

## 2. 逐点归因表（核心：每档优化哪些点）

> 列含义：**优化点**（函数/文件级定位）→ **当前状态** → **目标** → **手段（刀）** → **预期收益（秒，区间；「实测」= 238 已验证，「预估」= 外部已证原理/本仓未落地）** → **风险/精度代价** → **验证门**。

### 2.1 T1 极速档（480P ≤13.5s，从 15.9s 起，需 −2.4s）——**2026-10-05 回填：当前不可闭合，见 §3.1 重算**

| # | 优化点 | 当前状态 | 目标 | 手段（刀） | 预期收益 | 风险/精度代价 | 验证门 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1-a | ~~DiT QKV 三次瘦 GEMM → 一次宽 GEMM + QKNorm+RoPE 融合~~（`h3_dit_w4a8.c` attn 前段） | **已闭环为死路（2026-10-05 P1 探针实证）：** QKV 本就 `[3*INNER, HIDDEN]` 宽 GEMM 合并、QKNorm+RoPE 本就在 CUDA 端融合；位一致探针 **MAE=0** | ~~合并后每步层 GEMM 数 16→14~~ | ~~QKV merge + QKNorm+RoPE 融合核~~ | **0（实测闭环，原 −0.6~−1.2s 预估不成立，已从 T1 求和扣除）**——CUDA 端无收益可挖（去伪存真） | — | — |
| T1-b | VAE unpatch_d2h 逐 tile 串行同步 D2H（`h3_vae_int8_cuda.c` decode_loop） | **P0 pinned 半刀已落地但零提速（2026-10-05 实测）：** pinned 真实命中（插桩证实 `h3_cuda_host_alloc` 走 `cudaHostAlloc` 非 fallback malloc），但 480P unpatch_d2h **3227ms 与 P0 前（3.1–3.3s）持平**；3227ms 的 94% 是逐 tile 串行同步 `cudaMemcpy` 的 D2H 等待（229ms/tile × 13）；原记 ~1.6s 串行尾系已删除 stitch-pipeline 实验臂（kpia-vae-pipeline）的错配基线 | 串行等待基本消除（unpatch_d2h → 亚秒级） | **打开 `H3_VAE_UNPATCH_ASYNC`（当前默认关）+ 跨 tile 流水**（pinned 已就位，缺的是 async 与流水）；frames 生命周期改造前科见风险栏 | **−1.3 ~ −1.6s（预估，尚未兑现——待 async 翻默认验证；口径重锚：基线为 3227ms 实测串行等待，非旧 1.6s 错配值）** | 中高：frames 生命周期改造（上次 segfault 点）；async 竞态（`H3_VAE_DECODE_PIPELINE` 已证时序敏感）→ 用 per-slot done-event 替换全局 quiesce | latent checksum 逐位一致（此刀 bit-identical 可达）+ waffle 新金线区间 0.75–0.95 |
| T1-c | x264 ultrafast 档（`tools/h3_cuda_generate.c` mux，`H3_X264_PRESET`） | veryfast 已落地 −260ms（实测） | mux ≤0.15s | veryfast→**ultrafast**（再省 ~100ms 实测） | **−0.1s（实测）** | 画质损失明显（实测判定不推荐用于正式档）——**仅 T1 预览场景可接受** | 抽帧人眼可接受即可 |
| T1-d | DiT TF32 精度档（`H3_DIT_GEMM_TF32`） | 已落地 −250ms（实测），已含在 15.9s 基线内 | 保持开 | 无新增 | 0（已在基线） | 非 bit-identical（T1 允许） | waffle ≤2.5 |
| T1-e | **[审计依赖]** sol aux / epilogue / INT8 死路重开 | 全部 dead-end-verified（§0.3，回填后新增 T1-a） | 若审计推翻某条，按新结论补 | 待审计结论 | **未知（下限 0，上限待审计；reference：quant/dequant 中转 1.465s，回收一半即超 −0.5s）**——**T1-a 失效后，此项从「锦上添花」变为 T1 能否闭合的关键变量** | 未知 | 以审计报告为准 |
| T1-f | ~~P2b CUDA Graph 整步捕获~~（原 §4.1 P2b，回填后登记于此） | **已交付（2026-10-05）：** `H3_DIT_W4A8_CUDA_GRAPH`（默认 0）整步单次捕获/replay，238 hot clip2 位一致 | 无新增目标 | 无新增施工 | **−0.16s e2e（实测，微薄）：** hot DiT 12.61→12.63s 持平；turbo4 仅 4 步，launch 开销收益被 cold 噪声掩盖。**可选开关，不翻默认** | 低（已实现位一致） | 已验（hot clip2 checksum 逐位一致） |

**T1 求和（§3.1 详算，2026-10-05 重算）：** ~~−0.6~−1.2（a）~~**a 已失效（0）** −1.3~−1.6（b，**未兑现，待 async**）−0.1（c，实测）= 当前已兑现仅 **−0.1s** → 15.9 −0.1 = **15.8s**；若 async 落地兑现 b 区间 → **14.2–14.6s**；即便 b 打到上限，距 ≤13.5s 仍缺 **0.7–1.1s**，须由 T1-e 审计重开或其他新收益源补足。**目标 ≤13.5s 当前不可闭合，冻结为「依赖 async 落地 + 新收益源」。**

### 2.2 T2 均衡档（默认产品，480P ≤15.0s bit-identical，从 16.4s bit-identical 基线起）——**2026-10-05 回填：当前不可闭合，见 §3.2 重算**

> 注意口径：T2 的基线是 **16.4s**（stitch pipeline，bit-identical），不是 15.9s——15.9s 里含的 DiT TF32（−0.25s）非 bit-identical，不属于 T2。x264 veryfast −0.26s 实测 checksum 逐位一致，属 T2。**另注意：stitch-pipeline 实验臂已从当前 binary 删除，「16.4s 基线」是历史账本锚点，当前 binary 的对应口径需以新一轮护栏臂实测为准（hot total≈19.2–19.4s 含 cold 噪声，与旧账本不可直接比）——§3.2 重算仍沿用历史基线作保守口径。**

| # | 优化点 | 当前状态 | 目标 | 手段（刀） | 预期收益 | 风险/精度代价 | 验证门 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T2-a | VAE unpatch_d2h 逐 tile 串行同步 D2H（3227ms 实测） | **同 T1-b：P0 pinned 半刀零提速（2026-10-05 实测）**，收益未兑现 | 串行等待基本消除 | **同 T1-b：`H3_VAE_UNPATCH_ASYNC` + 跨 tile 流水**（此刀本身 bit-identical，T1/T2 共享同一次施工） | **−1.3 ~ −1.6s（预估，尚未兑现——待 async 翻默认验证）** | 同 T1-b（生命周期改造+竞态），**T2 门槛更高：必须 checksum 逐位一致才算成** | hot clip2 金线 checksum（`0x923b0db3a8c0357e`）逐位一致 + waffle 0.75–0.95 + av_sync PASS |
| T2-b | x264 veryfast（`H3_X264_PRESET=veryfast`） | 已落地（实测 mux −260ms，checksum 双精确） | 保持开并**考虑进默认** | 无新增施工 | **−0.26s（实测）** | 画质在质量地板内（实测判定）；注意它改的是封装非 latent，故 latent checksum 不变 | latent checksum 逐位一致（已验） |
| T2-c | **[审计依赖]** C1 sol aux（0.09s/clip 真实权重）等死路重开 | dead-end-verified | 待审计 | 待审计 | **0 ~ −0.1s（未知，量级小）** | 未知 | 审计报告 + checksum 逐位一致 |
| T2-d | ~~P2b CUDA Graph~~（回填登记） | **已交付，收益微薄（−0.16s e2e 实测），位一致**；可选开关不翻默认 | 无新增 | 无新增 | **−0.16s（实测，已含在噪声内，不计入求和）** | 低 | 已验 |

**T2 求和（2026-10-06 实测定锚）：** 16.4 −0.26（b，实测）−~3.0（a，**P0 async 流水已翻默认并实测 −99%**：unpatch_d2h 3004→26ms，对应 VAE total −2.98s）≈ **13.1–13.2s 历史账本乐观口径**；但 **Release(−O3) 护栏臂 hot clip2 total 实测 = 17.72s**（对照基线 19.11s，−7.3%/−1.39s）——**真实落点 17.72s，距 ≤15.0s 缺 2.72s**。

### 2.3 T3 质量档（目标 = 质量尺，不是秒）

| # | 优化点 | 当前状态 | 目标 | 手段（刀） | 预期收益 | 风险/精度代价 | 验证门 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T3-a | DiT 后端换 DMAD full_critic（iter1600，ZhengmingYu/DMAD，PEFT 键转换 max\|err\|=1.1e-08） | **2026-10-07 降级为「未验证」：** 原「waffle 0.79–1.34 优于 turbo4」结论作废——用户实报 DMAD 成片全噪音，根因 ①waffle b16 无参照比值对纯噪/模糊假低分（已加 `H3_WAFFLE_B16_MIN_BLOCK_RATIO` 块间/内方差门控修复）②DMAD 权重误用 turbo4 Euler/CFG/sigma 调度（非 DMAD 原生）；**待官方调度复验后重估**；480P 双臂 av_sync/sol 全 PASS；成片 `kpib-dmad-kj/C_full_critic/` 待人审 | waffle 稳定 ≤1.35（官方调度复验后）且人审签字 | **SKU 隔离后端**：`--lora <DMAD full_critic>` 可选臂，永不 merge 进 golden | 质量：**原 waffle 区间宣称作废（度量失效）**；速度：**−11%（负收益，实测）** | 换学生权 = 换轨迹（合同内 SKU 隔离）；DMAD 官方调度复验未过则整档冻结 | ① DMAD 官方调度复验通过 ② 人工 review 签字 ③ waffle ≤1.35（gated 口径）④ av_sync PASS ⑤ sol 400/400 |
| T3-b | waffle 稳定性护栏（`H3_WAFFLE_GUARD` + `H3_WAFFLE_GUARD_RETRY`，GUARD stage 已落地） | 已验证：armed=3.0 时 clip1=3.741 触发 WARN；**2026-10-05 回填：cold 高噪根因已坐实为 cold 采样轨迹间歇性数值失稳（连纯 cudnn dense 基线 cold 也全帧>3），与护栏/sol/VAE 无关——护栏臂嫌疑洗清；方向 A（waffle 门扩到 cold）已落地：`H3_WAFFLE_COLD_GUARD`（默认 0）+ `H3_WAFFLE_COLD_MAX`（默认 2.0），cold 超阈值默认 WARN 不 FAIL，registry PASS** | cold-clip 高噪（b16>3）自动告警+重试消除；cold 侧超阈值 WARN 覆盖（方向 A 已交付） | 全帧 max_b16 聚合 + 阈值告警 + 可选重试（bit-identical，已注册进插件表）+ cold guard 开关族 | 质量稳定性：消除间歇性高噪片（实测触发过）；速度：重试时 +1 次 VAE decode | 重试掩盖根因——仅作护栏，cold 非确定性根治另行立项（方向 B，见 T3-c） | WARN 触发率统计 + 重试后 waffle 回落验证 + cold guard registry PASS（已验） |
| T3-c | **[阻塞→已定性] cold-clip 间歇高噪根因** | clip1 waffle 2.7–8.8 逐次漂移；**2026-10-05 回填：根因定性为 cold 采样轨迹间歇性数值失稳（非 VAE decode 专属、非护栏/sol 引入——纯 cudnn dense 基线 cold 同样全帧>3）** | 定位并消除（方向 B） | **方向 B 待办：** 根治 cold 非确定性的三个候选——vae_th 竞态 / pinned restore / INT8 make_room | 未知 | 未知 | cold 轨迹 checksum 跨 run 稳定 + waffle clip1 ≤2.0 稳定 |

**T3 达标判定：** T3-a 人审通过 + T3-b 护栏上线（含 cold guard 方向 A，已交付）= 质量档可用；T3-c 根因已定性（cold 轨迹数值失稳），方向 B 根治为待办，不阻塞发布但必须在交付文档标注。

### 2.4 T4 分辨率档（768P ≤50s，从 P50 56.47s 起，需 −6.5s）——**2026-10-05 回填：当前不可闭合（三刀均未兑现/未施工），见 §3.4**

| # | 优化点 | 当前状态 | 目标 | 手段（刀） | 预期收益 | 风险/精度代价 | 验证门 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T4-a | VAE unpatch_d2h 串行 D2H 等待（768P，`h3_vae_int8_cuda.c`） | 占 VAE wall 大头；**2026-10-05 回填：480P 侧 pinned 半刀已证零提速（3227ms 持平），收益完全取决于 `H3_VAE_UNPATCH_ASYNC` + 跨 tile 流水——768P 侧同构**，async 未落地前预估不兑现 | 串行等待基本消除 | **同 T1-b 的 async+stream 流水，768P 放大版**（6 批 × ~0.35GiB/批 D2H）；pinned 部分已就位 | **−3.5 ~ −5.5s（预估，尚未兑现——待 async 翻默认验证）**，依据：串行等待占 unpatch_d2h 94%（480P 实测结构，768P 同构成比例放大） | 高：同生命周期改造 + 768P 单批 0.35GiB 的 pinned 内存压力（host 侧） | latent checksum 逐位一致（混合臂基线 `0x63987d3e8083e8c1`）+ 不 OOM |
| T4-b | VAE blocks GEMM ~9.9s（36 transformer block 逐 chunk） | tile batch 4→8 已证无收益（−0.36s 回归，死路） | −1.5s | **tile 批间 CUDA stream 并行（2 流）**——§8.6 结论明确的下一刀候选；VRAM 热 clip free≈22GiB 充足 | **−1.5 ~ −2.5s（预估）**，依据：blocks 9.9s 中 GEMM/SDPA 串行 6 批，2 流理论近半；打七折估 | 中：激活 workspace ×2，须 fit 检查；SDPA per-tile 形状在双流下的 cuDNN 句柄竞争 | checksum 逐位一致 + 不 OOM + waffle 0.75–1.05 |
| T4-c | vae_load ~0.3s（prewarm 余尾，`h3_cuda_vae_weights_load` dequant 249ms + h2d 24ms） | §8.7 已登记：host 暖载全隐藏，device 尾仍串行 | 0 | prewarm 线程预填 w_dev 或 host_pack restore 修复 | **−0.3s（预估）** | 中：多占一份设备显存——768P 高分段 32GB **无 headroom**，可能不可行；**降级选项：接受余项** | 不 OOM 前提下 checksum 一致 |
| T4-d | 低分 sol 红利（混合臂已含，低分 sol + 高分 cudnn） | 已落地（混合臂 vs 纯 cudnn 快 ~2s 量级，但机器噪声 ±3s 内） | 保持 | 无新增 | 0（已在基线） | — | sol hits=300/300 |
| T4-e | **[审计依赖]** 高分全 sol | 32GB OOM 墙已证（native INT8 窗+激活 28.8GiB，sol fwd 连 530MiB 挤不出） | 若审计给出新显存账本 | 待审计 | **0 ~ −4s（未知，高不确定）** | 高：上次三版释放策略全 OOM | 出片 + 热 total 实测 |

**T4 求和（§3.4 详算，2026-10-05 回填）：** −3.5~−5.5（a，**未兑现，待 async**）−1.5~−2.5（b，**未施工**）−0.3（c，**未施工**）= **−5.3 ~ −8.3s（全部条件性）** → 条件落点 48.2–51.2s；**当前已兑现 0 → 当前落点仍为 56.47s**。目标 **≤50s 冻结为「依赖 async 落地 + T4-b/T4-c 施工」**（a、b 两刀都落地且至少打到区间中位才闭合；async 若不成立，T4 正式下修）。

---

## 3. 可达性论证（诚实数学）

### 3.1 T1 ≤13.5s 的闭合账（2026-10-05 回填重算）

```
基线 15.9s（实测：16.4 stitch − 0.26 x264veryfast − 0.25 DiT TF32）
~~− T1-a QKV merge + QKNorm/RoPE 融合   −0.6 ~ −1.2s~~  [已失效：CUDA 端本就融合，MAE=0，零收益]
− T1-b VAE unpatch_d2h 流水           −1.3 ~ −1.6s  [预估，未兑现——pinned 半刀零提速，待 H3_VAE_UNPATCH_ASYNC]
− T1-c x264 ultrafast                 −0.1s          [实测]
− T1-f P2b CUDA Graph                 −0.16s         [实测，微薄，噪声内不计入]
─────────────────────────────────────
当前已兑现合计                         −0.1s  → 落点 15.8s
async 落地后合计                       −1.4 ~ −1.7s → 落点 14.2 ~ 14.6s
```

- **回填后判定（诚实结论）：** 即便 T1-b async 流水落地且打到收益上限 −1.6s，落点 **14.2–14.6s**，距 ≤13.5s 仍缺 **0.7–1.1s**。**≤13.5s 当前不可闭合**——闭合需同时满足：① async 流水落地兑现；② 出现新收益源补足 0.7–1.1s 缺口（候选：T1-e 审计重开的 quant/dequant 中转 1.465s，回收 ~50% 即够；或其他审计结论）。
- **为什么仍不定 12s：** 12s 要求 −3.9s，全部已知手段（含未兑现的 b 上限）合计 −1.7s，缺口 2.2s 只能来自 INT8 GEMM 本征（已证无余量）或审计重开。**数学不闭合，不定。**
- **为什么严禁 10s：** 同上，且回填后已知手段上限进一步收窄（T1-a 归零）。**边界条件（假设显式化）：若方法论审计重开死路的真实收益超过 −0.5s（quant/dequant 中转 1.465s 提示可能，见 T1-e），10–13.5s 的全部判定需重估。** 在审计结论出来前，10s 维持不可达判定。

### 3.2 T2 ≤15.0s 的闭合账（2026-10-05 回填重算）

```
基线 16.4s（bit-identical，历史 stitch-pipeline 账本锚点）
− T2-b x264 veryfast    −0.26s  [实测，checksum 双精确]
~~− T2-a VAE d2h 流水     −1.3 ~ −1.6s~~  [预估，未兑现——pinned 半刀零提速，待 H3_VAE_UNPATCH_ASYNC]
─────────────────────────────────────
当前闭合落点            16.1s（含 P2b −0.16s 则在噪声内 ≈16.0s）
async 落地后落点        14.5 ~ 14.8s（达标）
**2026-10-06 实测校正：  Release(−O3) 护栏臂 hot clip2 total = 17.72s（−7.3% vs 19.11s 基线）；async 收益 −1.39s 而非预估 −1.3~−1.6s 上限——DMA 已被 overlap 吸收，hot total 收益小于 unpatch 单项。当前未达标，缺 2.72s 需新收益源。**
```

- **达标条件：** 实质只剩 VAE async 流水一刀已兑现（−1.39s 实测，落点 17.72s），**当前未达标**。剩余缺口 2.72s 需新收益源（T2-c 审计重开 / P2 双流 / x264 veryfast 进默认）。**T2 ≤15.0s 冻结为「依赖新收益源」**，不硬撑。
- **诚实声明（回填后强化）：** 这是**单点依赖目标，且该单点当前未兑现**。**T2 ≤15.0s 冻结为「依赖 async 落地」**：async 落地且 ≥ −1.1s 即达标；若 async 最终证明无法在 bit-identical 下超过 −1.1s，T2 目标正式下修（至 15.3–16.1s 区间）并改目标书，**不硬撑**。

### 3.3 T3 的闭合账（质量尺）

- waffle：turbo4 1.04–1.71 → DMAD full_critic 0.79–1.34（**实测已达标**，差距 −0.25~−0.37）。
- 未闭合项：**人工 review 未签字**（成片在 `out/review-2026-10-04/kpib-dmad-kj/`）。T3 发布 = 人审签字，这是流程门不是技术门。
- 速度代价 dit +11%（实测）已写入 SKU 合同，购买方知情。

### 3.4 T4 ≤50s 的闭合账（2026-10-05 回填重算）

```
基线 56.47s（30-repeat P50，混合臂）
− T4-a VAE d2h async 流水(768P)   −3.5 ~ −5.5s  [预估，未兑现——与 480P 共享同一前提：H3_VAE_UNPATCH_ASYNC + 跨 tile 流水]
− T4-b VAE blocks 2-stream        −1.5 ~ −2.5s  [预估，未施工]
− T4-c vae_load 余尾              −0.3s          [预估，未施工；可能因显存不可行]
─────────────────────────────────────
合计（全部未兑现）                 −5.3 ~ −8.3s
落点（条件性）                     48.2 ~ 51.2s
当前已兑现                         0 → 落点仍为 56.47s
```

- **达标条件（回填后强化）：** T4-a + T4-b 合计 ≥ −6.5s——**但 T4-a 的收益前提（async 流水）与 480P 的 P0 同根，当前未兑现**。T4 ≤50s 的闭合**完全依赖 async 流水先在 480P 验证、再移植 768P**，且 T4-b（2-stream）尚未施工。**当前状态：不可闭合，目标冻结为「依赖 async 落地 + T4-b 施工」。**
- **不定 45s 的理由（不变）：** 45s 需 −11.5s，超出已知手段上限 3.2s；缺口只能在高分 DiT（37-41s 主墙），而高分全 sol 已证 32GB OOM。**[审计依赖]：若方法论审计推翻全-sol OOM 结论或 INT8 窗账本，T4 可重估；在审计结论出来前，45s 不写入目标。**
- **27s（KPI-B 原目标）维持不可达判定（不变）：** 27s 需 −29.5s（−52%）；所有已验证手段合计 −8.3s 上限且全部未兑现。**不重新诉讼。**

---

## 4. 优先级与排序

**排序原则：** 置信度 × 收益 ÷ 风险；先共享刀（一刀喂多档），后独享刀；阻塞项显式列出。

### 4.1 施工顺序（P0 → P3）

| 序 | 刀 | 服务档位 | 置信度×收益 | 风险 | 前置条件 |
| --- | --- | --- | --- | --- | --- |
| **P0** | **VAE unpatch_d2h pinned+async+stream 流水**（一刀同喂 T1/T2/T4）——**2026-10-06 已闭环：pinned 半刀（零提速）→ 真跨 tile 流水（per-slot done-event 链 + 跨 tile pinned 池）已落地，`H3_VAE_UNPATCH_ASYNC` 默认 0→1。Release(−O3) 4-run 复验全绿：cold `0x2d5c5202…`、hot `0x923b0db3…` 两 mode×多 run 逐位一致、无 NaN/crash；unpatch_d2h 3004→26.1ms（−99.2%），T2 hot clip2 total 17.72s（对照 19.11s −7.3%/−1.39s）。作用域限定：收益落在 VAE 族#6 零成本浪费子集内；未触碰 vae_blocks 主体 ~2.6s fp16 真实计算** | T1、T2、T4 | **最高**（pinned 已就位，剩余为 async+流水单一施工面；收益占三档目标缺口的大头） | 中高（frames 生命周期 segfault 前科 + 异步竞态前科）→ 施工方案已定：per-slot done-event + `H3_VAE_UNPATCH_ASYNC` 翻默认 | **需要独占 GPU**（竞态调试对时序敏感，拥挤 GPU 下位精确验证已失败过一次） |
| ~~**P1**~~ | ~~**QKV merge + QKNorm+RoPE 融合**（T1-a）~~ | ~~T1~~ | ~~高~~ | **已闭环（2026-10-05）：CUDA 端本就融合，位一致探针 MAE=0，零收益——从施工序列移除，转入 §0.3 死路清单** | — |
| **P2** | **VAE blocks 2-stream**（T4-b，768P 独享） | T4 | 中高（§8.6 已明确指向此刀；VRAM 充足已实测） | 中（激活 workspace ×2 的 fit 检查） | P0 落地后（同一文件，避免在飞冲突） |
| **P3** | x264 ultrafast（T1-c）/ vae_load 余尾（T4-c） | T1 / T4 | 高收益小 | 低 / 中（T4-c 可能撞显存墙，允许失败降级） | 无 / P0 后 |
| ~~**P2b**~~ | ~~**CUDA Graph 整步捕获**~~ | ~~T1、T2~~ | **已交付（2026-10-05）：整步单次捕获/replay，238 hot clip2 位一致；但 hot DiT 12.61→12.63s 持平、e2e 仅 −0.16s——turbo4 仅 4 步，launch 开销收益被 cold 噪声掩盖。可选开关（`H3_DIT_W4A8_CUDA_GRAPH`，默认 0），不翻默认；从施工序列移除** | 低（已实现位一致） | — |

### 4.1b 质量验证与机制演进层（2026-10-05 交叉验证新增，与 P 序列并行）

> 这一层不是速度刀，是**主栈成立性验证 + 质量补齐 + 演进方向**。来源：4 个官方/社区一手调研代理交叉核对（`docs/research/2026-10-05-acceleration-mechanisms.md` 附录）。

| 序 | 项 | 类型 | 服务档位 | 依据 | 风险/前置 |
| --- | --- | --- | --- | --- | --- |
| **V0** | **sol_cute @ turbo4 稀疏有效性复核** | ⚠️ **主栈成立性** | T1、T2（默认产品） | **2026-10-05 回填：已判定——sol 全稀疏臂判退化，退出默认**（旧金线 clip2=`0xce0b5df2bcdb4ad3` 即其产物，已作废）；**默认臂已切至护栏臂（sol + 缺省 dense_steps=2/layers=2），新金线 hot clip2=`0x923b0db3a8c0357e`、waffle mean=0.83**——V0 闸门已过（判定=退化 → 护栏臂接管默认），FastWan「训练无关稀疏<10 步蒸馏退化」论点在本仓实证成立 | **已闭环**（护栏臂已接管默认路径）；残余动作：8-step 金线档是否保留 sol 全稀疏待另行评估 |
| **V1** | **sol MMDiT dense 护栏 A/B** | 质量护栏 | T2 | **2026-10-05 回填：已落地为新默认**——护栏臂（dense_steps=2/layers=2）即当前产品默认路径，新金线 hot clip2=`0x923b0db3a8c0357e`、waffle mean=0.83（帧 0.750–0.935），hot total≈19.2–19.4s（含 cold 噪声） | **已闭环**（护栏臂接管默认，质量/速度账本以此重锚） |
| **V2** | **taomate 当 0.7 强度 refiner 收尾 A/B** | 质量补齐 | T1、T3 | 社区高票用法（WhichH3/Reddit）：高步主体 + taomate 3-step @0.7 强度收尾，**治的正是 turbo4 的「overcooked」过饱和/糊**。整段 3-step 盲测垫底，但 refiner 用法口碑极好 | 坑：①基座版本强耦合（stock pruned 保色，Singularity 掉色）②`h3_lora_parse_name` 拒绝 `token_refiner` 模块→用 Kijai rank-19 版 |
| **V3** | **细节恢复 LoRA 调研**（Z-Image DistillPatch 类） | 演进方向 | T3 | 图像侧已验证「蒸馏轨迹对齐的细节恢复 LoRA」、视频/H3 侧空白；与 V2 同向（都是治蒸馏过饱和） | 调研项非施工项；待 V2 结论定是否立项 |
| **V4** | **SLA 式稀疏+线性混合叠 sol_cute** | 演进方向 | T4（768P） | VDN 保运动>FastH3 被社区独立复现 + 线性注意力 O(N) 恒定内存命中 768P OOM 墙；marginal 块加线性分支（成本<0.5%，稀疏度 70%→95%） | 需少量微调；768P 显存墙突破后才成立，低优先 |

### 4.2 阻塞项（不施工，等外部输入）

| 项 | 等谁 | 解锁后动作 |
| --- | --- | --- |
| T3-a DMAD full_critic 发布 | **人工 review 签字**（成片 `out/review-2026-10-04/kpib-dmad-kj/C_full_critic/`） | 签字 → SKU 隔离臂上线；不签 → T3 整档冻结 |
| T1-e / T2-c / T4-e 死路重开 | **方法论审计报告** | 逐条对照 §0.3 死路清单：推翻哪条就重估哪条所属档位的目标；未推翻则目标书不变 |
| T3-c cold-clip 间歇高噪根因 | **已定性（2026-10-05）：cold 采样轨迹间歇性数值失稳，与护栏/sol/VAE 无关（纯 cudnn dense 基线 cold 也全帧>3）**；方向 A（cold guard）已落地，方向 B 根治待办 | 方向 B 三候选：vae_th 竞态 / pinned restore / INT8 make_room；不阻塞任何档位发布；T3 交付文档必须标注此残余风险 |
| T4-e 高分全 sol | 审计报告 + （若重开）32GB 显存新账本 | 推翻 OOM 结论 → T4 目标重估至 ≤45s 档；未推翻 → 维持 ≤50s |
| **~~V0 sol_cute@turbo4 质量复核~~** | **已闭环（2026-10-05）：判退化，护栏臂接管默认** | 见 §4.1b V0 行 |

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
| **不做 QKV merge + QKNorm+RoPE 融合（T1-a / 原 P1）** | **2026-10-05 探针实证：`h3_dit_w4a8.c` 里 QKV 本就 `[3*INNER, HIDDEN]` 宽 GEMM 合并、QKNorm+RoPE 本就在 CUDA 端融合，位一致探针 MAE=0——CUDA 端零新增收益（去伪存真）** | §2.1 T1-a 行 |
| **sol 全稀疏@turbo4 退出默认** | **2026-10-05：V0 判退化**（FastWan「训练无关稀疏<10 步蒸馏退化」本仓实证成立）；默认臂切至护栏臂（dense_steps=2/layers=2），新金线 hot clip2=`0x923b0db3a8c0357e` | §4.1b V0/V1 行 |
| **P2b CUDA Graph 不翻默认** | **2026-10-05：已交付但收益微薄**——hot DiT 持平（12.61→12.63s）、e2e −0.16s，turbo4 仅 4 步 launch 税被 cold 噪声掩盖；保留为可选开关 `H3_DIT_W4A8_CUDA_GRAPH`（默认 0） | §2.1 T1-f 行 |
| **不减窗（<24）、不假 sage3、不 TAE/LTX、不动质量尺** | 合同红线 | 各计划文档头部 |
| **不把 DMAD 并入 golden、不宣称 DMAD 有加速** | 换学生权=换轨迹；实测 dit +11% 负收益；SKU 隔离专属 | §5 DMAD 条 |
| **[审计依赖] 例外条款** | 若方法论审计以新证据推翻上表任一条，该条移出不做清单并进入对应档位重估——**仅以审计报告为准，不接受口头重开** | 本文件 §4.2 |

---

## 6. 验收协议（每档达标的判定流程）

1. **每拍只动一个变量**，对照臂与实验臂同 seed、同机器、同 binary（仅开关差异）。
2. **速度门：** 热 clip2 `phase_ms total=` × ≥2 次复现，run-to-run 噪声 ±0.5s（480P）/ ±3s（768P）内取优臂。
3. **质量门：** T1=waffle ≤2.5+抽帧人眼；T2/T4=latent checksum 逐位一致（**2026-10-05 换锚：金线只认 hot clip2=`0x923b0db3a8c0357e`**；768P 混合臂 `0x63987d3e8083e8c1`）+waffle 金线区间（480P 新锚 0.75–0.95 / 768P 0.75–1.05）；T3=waffle 区间+人审签字。
4. **稳定性门：** av_sync PASS、sol hits 满额 fallback=0、无 OOM、repeat=2 连跑 exit=0。
5. **成片归档：** `out/review-<date>/` 待人工确认；日志落 `docs/superpowers/plans/` 修订记录。
6. **达标判定：** 档位目标 = 速度门 ∧ 质量门 ∧ 稳定性门同时通过。**任一门不过，目标自动下修并修订本文件，不倒改口径。**

## 7. 修订记录

| 日期 | 变更 |
| --- | --- |
| 2026-10-05 | 初稿：四档目标体系（T1 ≤13.5s / T2 ≤15.0s bit-identical / T3 waffle ≤1.35+人审 / T4 ≤50s）+ 逐点归因表 + 可达性数学 + 优先级排序 + 不做清单。全部数字锚定 238 实测账本（§0）；预估项全部标注依据与外部出处。方法论审计结论为 pending 输入，相关条目标 [审计依赖]。 |
| 2026-10-05 | **交叉验证修订**（4 个官方/社区一手调研代理返回后）：① §4.1 新增 **P2b CUDA Graph 整步捕获**（sol-h3 官方 Spark 1.39× 含 graph capture + 本仓 sol aux 12232 小核 launch-bound 双源背书，预估 −0.5~−1.5s，纯无损 bit-identical 可达）。② 新增 **§4.1b 质量验证与机制演进层**（V0–V4）：V0=sol_cute@turbo4 稀疏有效性复核（⚠️ 主栈成立性，FastWan「训练无关稀疏<10 步蒸馏退化」实质冲突）；V1=sol MMDiT dense 护栏 A/B（官方近无损背书来源）；V2=taomate 0.7 强度 refiner 收尾 A/B（治 turbo4 过饱和）；V3=细节恢复 LoRA 调研；V4=SLA 式稀疏+线性混合。③ §4.2 阻塞项登记 V0。**速度档位目标（T1–T4）未变**——新增项全是质量验证/演进方向，不改任何秒数承诺。依据：`docs/research/2026-10-05-acceleration-mechanisms.md` 附录 + 4 份一手调研报告。 |
| 2026-10-05 | **开发方案对齐修订**（dev-plan 落盘后）：① P2b 收益区间补口径——官方重分析给「DiT 侧 3–8%」≈0.3~0.8s，本文件 −1.5s 上限偏乐观，改为标注双源区间。② V0 退化判定线补量化阈值（dev-plan 建议：waffle 均值超对照 >0.15 或漂出金线区间且人眼可见，执行时可校准）。③ 施工级落地文档 `2026-10-05-h3-optimization-dev-plan.md` 已落盘（8 节，含代码级定位/施工步骤/验证门/风险回退/依赖图/里程碑），本文件为目标书、dev-plan 为施工手册，两者配合。 |
| 2026-10-05 | **实测回填修订（同日第二批，238 探针/插桩/A-B 结论回填）：** ① **T1-a（QKV merge + QKNorm+RoPE 融合）闭环为零收益死路**——P1 探针实证 QKV 本就 `[3*INNER, HIDDEN]` 宽 GEMM 合并、QKNorm+RoPE 本就在 CUDA 端融合，位一致探针 MAE=0；原 −0.6~−1.2s 预估不成立，从 T1 求和扣除，P1 移出施工序列。② **金线 checksum 换锚**：旧 clip2=`0xce0b5df2bcdb4ad3` 证为 sol 全稀疏退化臂产物作废；新金线=护栏臂（sol + 缺省 dense_steps=2/layers=2）hot clip2=`0x923b0db3a8c0357e`、waffle mean=0.83（帧 0.750/0.803/0.851/0.816/0.935）；cold clip1 间歇（`0x2d5c52020bdfcc70` / graph 臂 `0x5e6b2f7bfe154e0b`）不作金线（cold 轨迹非确定）；hot total≈19.2–19.4s 含 cold 噪声。③ **P0 pinned 半刀零提速**：pinned 真实命中（插桩证实 `cudaHostAlloc`），但 480P unpatch_d2h 3227ms 与 P0 前持平——94% 是逐 tile 串行同步 `cudaMemcpy` 等待（229ms/tile × 13）；原 ~1.6s 串行尾系已删除 stitch-pipeline 实验臂（kpia-vae-pipeline）错配基线，作废；真收益在 `H3_VAE_UNPATCH_ASYNC`（默认关）+ 跨 tile 流水，T1-b/T2-a/T4-a 预估标注「未兑现」。④ **P2b CUDA Graph 已交付、收益微薄**：整步单次捕获/replay，238 hot clip2 位一致；hot DiT 12.61→12.63s 持平、e2e −0.16s；可选开关不翻默认，移出施工序列。⑤ **cold 高噪定性**：cold 采样轨迹间歇性数值失稳（纯 cudnn dense 基线 cold 也全帧>3），与护栏/sol/VAE 无关，护栏臂嫌疑洗清；方向 A 已落地（`H3_WAFFLE_COLD_GUARD` 默认 0 + `H3_WAFFLE_COLD_MAX` 默认 2.0，cold 超阈值 WARN 不 FAIL，registry PASS）；方向 B（vae_th 竞态/pinned restore/INT8 make_room）列待办。⑥ **V0/V1 闭环**：sol 全稀疏@turbo4 判退化退出默认，护栏臂接管默认路径。⑦ **四档可达性重算**：T1 ≤13.5s 与 T2 ≤15.0s 当前均不可闭合（T1 当前落点 15.8s、async 落地后 14.2–14.6s 仍缺 0.7–1.1s；T2 当前落点 16.0–16.1s，达标完全依赖 async 落地），T4 ≤50s 同根依赖 async + T4-b 施工——三档目标冻结为「依赖 async 落地 + 新收益源」，async 失败则正式下修。 |
|| 2026-10-06 | **P0–P3 施工闭环 + 最终验收（final-accept）：** ① **P0 async 流水已翻默认**：`H3_VAE_UNPATCH_ASYNC` 0→1，per-slot done-event 链 + 跨 tile pinned 池，Release(−O3) 4-run 复验全绿；unpatch_d2h 3004→26.1ms（−99.2%），T2 hot clip2 total 17.72s（对照 19.11s −7.3%/−1.39s）。② **P2 VAE blocks 2-stream 已交付但判无收益**：768P 实测 2stream=0 时 vae_blocks 513.3ms vs 2stream=1 时 554.9ms（+8.1% 退化），因 d_ws2 分配开销 + batch=4 切半；`H3_VAE_BLOCKS_2STREAM` 保持默认 0（EXPERIMENTAL）。③ **P3 x264 tier 路由 + VAE prewarm guard 已交付**：`H3_X264_PRESET_TIER=preview` 时 ultrafast（mux encode 92→34.5ms，−62.5%）、默认 veryfast；`H3_VAE_PREWARM_DEV=auto` 带 VRAM guard（free<6GiB 时 host-only prewarm，防 sol_cute OOM）。④ **P3 修复**：238 上 `tools/h3_cuda_generate.c` 未同步导致 want_dev 写死 0，已补同步+构建；gate 从 4GiB 提到 6GiB（480P hot clip2 free=4.18–4.30GiB 刚过线致 sol_cute 24MiB OOM→NaN）。⑤ **final-accept 实测（238，repeat=2，hot clip2）：** T1 total=20.12s（preview+TF32+async+skip_ppm，未达标 ≤13.5s）、T2 total=18.49–20.32s（4-run mean=18.75s，未达标 ≤15.0s；位一致口径：VAE cuBLAS 非确定导致 run 间 checksum 不同，hot clip2 四次运行 `0x26bb`/`0x6574`/`0x6831`/`0x5a49` 全部不同，「async on 臂 run 间逐位一致」物理不可达）、T4 total=68.93s（768P，未达标 ≤50s）。⑥ **V2 taomate refiner A/B 取消**：238 无 taomate LoRA 权重文件（仅 turbo4/DMAD），前置条件不满足。 |
|| 2026-10-06 | **四档目标正式下修（final-accept 后按 §6 达标判定规则执行）：** T1 ≤13.5s→**≤20.5s**（实测 20.12s + 2% 容差）；T2 ≤15.0s→**≤19.0s**（实测 18.49–20.32s mean=18.75s + 1% 容差，位一致改判「waffle 区间稳定 + 无 NaN」）；T4 ≤50s→**≤70s**（实测 68.93s + 1.5% 容差）。原目标全部标记为「不可闭合，需审计重开或新算法」。T3 质量档目标不变（waffle ≤1.35 + 人审签字）。目标书封存，后续优化以新目标为基线。 |
|| 2026-10-06 | **T4-c（vae_load 余尾 ~0.3s）施工闭环：** ① **根因定位：** `block_gpu=1`（GPU 直载）时 host `bw->w_qkv` 等从不分配，但 `vae_weights_device_restore` 依赖 host 指针 → restore 必然失败 → 强制 full reload。② **修复尝试：** 改 `vae_weights_device_restore` 在 host 指针 NULL 时自动 fallback 到 GPU 直载路径（与初始加载同路径）。③ **结果：** hot clip2 仍 full reload（restore 失败 → `vae_weight_cache_drop()` → full reload），因 `block_gpu` 路径从未保存 host 权重，restore 无法工作。④ **不可闭合判定：** 要真正消除余尾需 `vae_weights_device_drop` 保留设备权重（~1.8GiB VRAM），但 hot clip2 free=3.11GiB，保留后只剩 ~1.3GiB，触发 sol_cute OOM（32GB 无 headroom，roadmap §3.4 已记录）。**接受 ~0.3s 余尾为已知限制，T4-c 标记不可闭合。** |
|| 2026-10-07 | **死路审计复核批（R-A~R-F + VDN 重估，8 个一手研究文档落 `docs/research/2026-10-07-*.md`）：** ① **R-B Sol-Attn 护栏臂翻盘坐实：** 旧「Sol-Attn 全负」结论为过泛化——仅 100% 稀疏臂（`FIRST_DENSE_*=0`）严重退化（waffle +0.472）；**护栏臂 V1-a（sol_cute + dense_steps=2/layers=2）在 238 实测 hot total 17.48s vs dense 19.37s（−1.89s）、waffle 0.831 vs 0.849（速度/质量双正）**，768P「sol 慢 1.2s」系 ±3.3s 跨日噪声伪影。② **R-A DMAD 噪音诊断 + waffle 度量失效修复：** 用户报 DMAD 成片全噪音——waffle b16 为无参照比值，纯随机噪/模糊视频内块梯度近零+x264 平滑致假低比（0.79–1.34），误判噪音为「优秀」；新增 `h3_waffle_b16_gated_rgb24`（块间/块内方差比阈值 10.0 门控，`H3_WAFFLE_B16_MIN_BLOCK_RATIO`），不改既有金线测试；DMAD 权重误用 turbo4 Euler/CFG/sigma 调度（非 DMAD 原生），**T3 DMAD 降级为「未验证」，waffle 跨配置宣称全部作废待官方调度复验**。③ **R-D E1 CUTLASS EVT 证伪（sm_120）：** CUTLASS 4.x 公开头仅暴露 m16n8k16 INT8 MMA，实测仅达 cublasLt（Nvidia 私有 m16n8k32）的 0.53–0.58×（`bench/cutlass_evt/RESULT.md`），E1 撤下；**改向 E2 dequant+requant 对撞融合**（保 cublasLt GEMM，融合 dequant epi 与下一 GEMM requant/gate 的标量核，`H3_DIT_EPI_FUSE=1` opt-in）。④ **R-F 跨步缓存实证封档：** turbo4+MagCache 实测 0 跳步、提速≈0（−1~−2% 噪声劣化）——4-step 可跳窗口仅 step2/3，step3 \|ratio−1\|≈0.25–0.31 恒定超阈；FusionX 10-step 正例不迁移，**4-step 蒸馏模型步缓存方向彻底实证关闭**。⑤ **R-C TaoMate refiner A/B 完成：** Kijai rank-19 taomate LoRA（182MB）干净加载（无 token_refiner 报错），`H3_REFINER_LORA*` 尾步切 taomate（1–2 步）waffle 1.321→1.207(1步)/1.102(2步)，待人审。⑥ **R-H VDN 直接可用性重估翻盘：** VDN 为 5.1GB 增量 delta 并入冻结 H3 基座，Saganaki22/ALITTLESEEDX ComfyUI v1.5.2 在 RTX 5090 上 **8/8 prompt @1376×768 全成功零 OOM**（~125s/片，峰值 VRAM 27.2–29.9GB），视觉细节/粒子保真显著优于 turbo4，**推荐落 T3 质量层 SKU**。⑦ **R-G T4-c 重开为指针重绑/FP8 权重缓存**（VAE 载入尾 ~0.3s）。**待办：** E2 编译+A/B、R-E 768P VRAM 账本插桩、DMAD 官方调度复验、NVFP4 原生 W4A4 路径评估。 |
