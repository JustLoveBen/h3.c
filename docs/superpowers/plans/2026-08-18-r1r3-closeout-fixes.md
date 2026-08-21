# R1/R3 收尾修复 Backlog 与执行顺序

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 仅围绕当前阻碍收口的 4 个残余问题重排执行顺序，先恢复默认质量路径的正确性与可解释性，再压降总时长，最终形成可复验的收尾版本。

**Architecture:** 本计划不再按旧的 R1/T0-T8 顺序推进，而是改为“先清默认路径与主题正确性，再压关键暴露项，再做端到端收口”的收尾修复链路。DiT 优化有效、TE 契约未回退、VAE 热缓存证据已补齐，均视为既有成立前提，不再作为主修复项；除非新修复触发回归，否则不重开这些方向。

**Tech Stack:** 纯 C11 + CUDA（RTX 4090 / cuBLAS / cuBLASLt / cuDNN SDPA），主路径为 INT8 ConvRot DiT + NVFP4 TE + VAE CUDA；文档与验证口径沿用 `docs/nvidia_optimization_plan.md`、`docs/nvidia_quant_contract.md` 与既有 `out/verify_*` 产物。

**Spec:** `docs/nvidia_optimization_plan.md` §22，以及 2026-08-18 当前真实收尾状态

## Global Constraints

- 本轮**不能按已完成收口**；新计划必须以“仍未达标”为前提。
- 已成立结论保留，不作为主线返工项：**DiT 优化有效**、**TE 契约未回退**、**VAE 热缓存证据已补齐**。
- 当前未达标的 5 个现状必须显式反映到优先级判断中：
  - 默认路径总时长仍约 **50.7s-52.8s**，未到 **<=35s**
  - 双主题中**橙猫 PASS**，但**水下 FAIL**（未见海龟）
  - 默认日志仍出现 **`reuse=1`**，默认配置不干净
  - **TE 暴露时间**仍过高
  - **VAE 冷启动 load** 仍未压到目标区间
- 默认质量路径仍必须保持：**INT8 ConvRot DiT + 已修复 NVFP4 TE + 当前默认 VAE 路径 + cuDNN SDPA**。
- **不允许**通过切到 W4A8 默认、打开 magcache/reuse、改默认注意力后端、减少 steps/layers 来“伪收口”。
- **F2VA 首帧素材不是主修复链路优先项**；只能放在收尾完成后的后验验证。
- 不 commit；本文件仅定义 backlog、顺序、依赖、门禁与风险。

---

## 0. 当前收尾判断

### 已可视为完成的前提

- **DiT 优化有效**：不再把 DiT 主体性能当作第一优先级问题；仅保留“是否被其他改动拖回去”的回归检查。
- **TE 契约未回退**：`hi_first`、`unswizzle`、`pre_quant` 仍视为冻结契约，不允许为压时延破坏正确性。
- **VAE 热缓存证据已补齐**：热路径可命中已成立；当前问题集中在**冷启动 load**，不是“有没有 cache hit 证据”。

### 当前真正阻塞收口的 4 个残余问题

1. **默认路径不干净**：日志仍出现 `reuse=1`，说明默认配置存在隐式开关、参数继承或 preset 污染，当前基线不可信。
2. **主题正确性不完整**：橙猫 PASS 但水下 FAIL，说明主题 follow 仍未稳定，不能宣布质量路径收口。
3. **TE 暴露时间过高**：即使契约正确，TE 仍然在默认 e2e 中占据过高可见时延。
4. **VAE 冷启动 load 过高**：热缓存证据成立，但第一次 load 仍拖累默认端到端时间。

### 这 4 项与最终收口的关系

| 项目 | 是否阻塞最终收口 | 阻塞原因 |
| --- | --- | --- |
| 默认 `reuse=1` 来源与默认配置清洁 | **是** | 基线不可信，后续所有 timing 结论都可能失真 |
| 水下主题恢复（海龟出现） | **是** | 双主题未同时 PASS，质量路径不能收口 |
| TE 暴露时间压降 | **是** | 总时长仍在 50s+，TE 是明确剩余大头之一 |
| VAE 冷启动 load 压降 | **是** | 默认路径是“首轮可复现”体验，冷启动 load 仍在拖总时长 |

### 第 5 步之后（2026-08-18 统一复验，未重跑 generate）

P0–P3 审查通过后，采用 `out/verify_r1r3_p3_vae/` 作为收口前基线（详见 §3）。

| 项目 | 复验结果 |
| --- | --- |
| 默认配置清洁（cfg=5、reuse=off、INT8 DiT、无 magcache） | **PASS** |
| 双主题（橙猫单猫窗台；水下海龟+礁石无人） | **PASS** |
| VAE 冷启 load ~1.9s | **PASS**（P3） |
| 默认总时长 `<=35s` | **FAIL**（橙猫 63.75s / 水下 68.03s） |
| TE 暴露 | **仍阻塞时长**：本跑 23.77s / 22.31s（历史口径 ~16.6–23.8s） |

**本轮不能按已完成收口对外表述。** 不为碰运气重跑 generate。

---

## 1. 建议执行顺序

### 顺序总览

1. **先清默认路径**：查清 `reuse=1` 来源，拿到可信基线。
2. **再修主题正确性**：先把水下主题恢复到“海龟可见”。
3. **再压 TE 暴露时间**：在可信且正确的默认链路上压 TE。
4. **最后压 VAE 冷启动 load**：完成默认体验层面的最后一块时间回收。
5. **收口复验**：判断默认双主题与总时长是否达到 `<=35s`。**实际做法（Parent ruling）：采用 P3 产物 `out/verify_r1r3_p3_vae/`，跳过重复 generate，不为碰运气重跑。**

### 为什么不是先盯总时长

- 当前总时长 50s+ 的测量前提还不干净，因为日志里仍有 `reuse=1`。
- 水下主题仍 FAIL，说明即使先压出更快数字，也不能用于宣布收口。
- 只有在“默认配置干净 + 双主题正确”之后，TE/VAE 的优化结果才有收口意义。

---

## 2. 收尾修复 Backlog

### P0. 默认 `reuse=1` 来源与默认配置清洁

**目标**

确认默认路径中 `reuse=1` 的真实来源，去掉一切隐式继承、preset 残留、环境变量污染或默认值串线，建立“未显式开启就绝不会出现 `reuse=1`”的干净基线。

**验收标准**

- 默认命令、默认 preset、默认环境下，日志中**不再出现** `reuse=1`
- 若显式开启 reuse，日志仍能正确显示 `reuse=1`
- 默认路径的配置来源可解释：能明确说清是 CLI 默认、env 默认、preset 默认，还是运行时派生值
- 清洁基线落盘后，后续 timing 复测统一基于这条干净默认路径

**建议顺序**

排第 1，先做。

**依赖**

- 无前置依赖
- 需要对默认 CLI 参数、preset、环境变量读取顺序、日志打印来源做一轮对账

**建议排查范围**

- `tools/h3_cuda_generate.c`
- 与 preset / 默认参数 / env 读取相关的路径
- `reuse`、`magcache`、`window`、`token_reduce` 等历史产品档旋钮的默认注入点

**主要风险**

- 不是单点 bug，而是“日志打印值”和“真实生效值”不一致
- 可能存在 preset 与手工参数的优先级串线
- 清理默认值后，当前 50s+ 基线可能进一步上升或下降，需要重新接受真实数字

**是否阻塞最终收口**

**阻塞。** 若默认链路不干净，则所有后续“压到 <=35s”的结果都不具备说服力。

**执行记录（2026-08-18）**

- [x] 已确认根因：`tools/h3_cuda_generate.c` 将 CLI 局部变量 `reuse` 默认初始化为 `1`，并直接用于 shape 日志与 `reuse_schedule` 打印；因此即使**未传** `--reuse`，日志仍会出现 `reuse=1` 与 `selected=8/8`。这不是 env / preset 注入，也不是 schedule 函数 fallback，而是 CLI 默认值与“默认无 reuse”口径不一致。
- [x] 已做最小修复：把“默认全量评估”与“显式 `--reuse 1`”拆开。默认路径保持 full-eval 数学行为，但日志改为 `reuse=off`，`reuse_schedule` 改为 `disabled default_full_eval=1`；只有显式 `--reuse`（或显式 `magcache` 需要 schedule 口径）时才打印 interval/selected。
- [x] 已做最小复验：远程 4090 `/home/haiyang/h3_c/src_sync/build-cuda/h3_cuda_generate` 上，默认命令日志见 `out/verify_r1r3_p0_reuse_clean/default.log`，显式 `--reuse 1` 对照日志见 `out/verify_r1r3_p0_reuse_clean/reuse1.log`。
- [x] P0 DoD：通过。默认不再报 `reuse=1`，不再出现 `reuse_schedule: interval=1 selected=8/8`；显式 `--reuse 1` 仍正确显示；默认完整生成仍成功结束并写出帧。

---

### P1. 水下主题恢复（海龟出现）

**目标**

在干净默认路径上恢复水下主题 follow，使结果重新回到“珊瑚礁 + 海龟、无人”的既有 PASS 语义。

**验收标准**

- 水下主题默认链路首帧/关键帧中可稳定看到**海龟**
- 结果不再退回旧的“人物/错误主题”语义
- 橙猫主题继续保持 PASS，不因修水下而回退
- 与 `out/verify_theme_ok/underwater_turtle.mp4` 的语义锚点一致

**建议顺序**

排第 2，在默认路径清洁之后立即做。

**依赖**

- 依赖 P0：先确认不是 `reuse`/preset 污染导致的假性 theme fail
- 依赖既有 TE 契约冻结：修水下时不得回退 `hi_first`、`unswizzle`、`pre_quant`

**建议排查范围**

- 默认 prompt / token ids / 主题 follow 相关输入链路
- TE 输出到 DiT 条件输入之间的默认路径
- 与“官方 prompt FAIL 片”对比已有“theme OK”产物的差异来源
- 任何会让条件语义被复用、截断、覆盖的 reuse / cache / warm-start 行为

**主要风险**

- 水下 FAIL 可能不是单一 prompt 问题，而是默认链路里某个条件复用状态污染
- 若只看橙猫 PASS 可能掩盖“双主题不对称”的条件偏置
- 修水下时容易误触 TE/DiT 条件接口，需防止契约回退

**是否阻塞最终收口**

**阻塞。** 只要双主题未同时 PASS，就不能按质量路径收口。

**执行记录（2026-08-18）**

- [x] 已确认主因不是 TE/DiT 契约回退：当前 FAIL 片仍稳定产出“珊瑚礁/海底氛围、无人”，但默认复验把水下样例切到了 `v6_underwater.txt` 这组 **19-token 压缩 ids**，同时 `h3_cuda_generate` 的当前默认 CFG 也漂到了 `1`；两者叠加后，条件强度不足以把“sea turtle”稳定拉进关键帧。
- [x] 已做最小修复：`tools/h3_cuda_generate.c` 恢复主题路径默认 `cfg=5`（显式 `--cfg 1` 仍可关掉 CFG）；默认水下 gate 口径改回 `out/verify_official_prompt/b_underwater.ids`，保留 `v6_underwater.txt` 仅作探索/历史对照，不再作为默认收口样例。
- [x] 已做远程 4090 复验：`v6_underwater.txt + seed=91 + cfg=1/5` 与 `seed=42 + cfg=5` 均未稳定给出海龟；改用 `b_underwater.ids + seed=91 + cfg=5` 后，关键帧恢复到“珊瑚礁 + 海龟、无人”。对照产物见 `out/verify_r1r3_p1_officialids_uw/`；失败对照见 `out/verify_r1r3_final/`、`out/verify_r1r3_p1_cfg5_uw/`、`out/verify_r1r3_p1_seed42_uw/`。
- [x] 已顺手回归橙猫：`v4_orange_cat.txt + seed=17 + cfg=5` 仍为单只橙猫睡窗台，预览见 `out/verify_r1r3_p1_cfg5_cat/`。
- [x] P1 DoD：通过。默认主题路径下，水下样例已恢复“可见海龟且无人”，且橙猫未回退。

---

### P2. TE 暴露时间压降

**目标**

在不回退 TE 契约、不改变默认质量路径数学语义的前提下，继续压低 TE 在默认 e2e 中的可见时间占比，使其不再成为 50s+ 总时长中的主要暴露项。

**验收标准**

- TE 契约回归继续成立
- TE wall / expose 时间较当前默认基线有明确下降
- 压降后默认总时长出现可测改善，而不是只把时间转移到其他阶段
- 默认双主题结果不回退

**建议顺序**

排第 3，在默认路径干净且水下恢复后执行。

**依赖**

- 依赖 P0：必须基于干净默认基线量 TE
- 依赖 P1：先确认条件语义正确，再谈压 TE
- 依赖已成立前提：TE 契约冻结，不允许为追时延动 `hi_first` / `unswizzle` / `pre_quant`

**建议排查范围**

- `h3_nvfp4.c` / `h3_nvfp4_cuda.cu`
- TE encode 与 generate 调度重叠点
- host/device cache 生命周期
- 是否还有可移除的同步点、重复 pack/H2D、重复 dequant 或不必要暴露

**主要风险**

- 容易把“暴露时间压降”误做成“契约简化”或“结果缓存走样”
- overlap 类优化可能让计时看起来变好，但真实总时长未改善
- 若与默认配置清理互相耦合，需要先固定日志与阶段口径

**是否阻塞最终收口**

**阻塞。** 当前总时长距 `<=35s` 仍差很大，TE 是已知必须回收的剩余时间块。

**执行记录（2026-08-18）**

- [x] 已确认 T4/T5 的 8 槽 device `d_w` 缓存对 50L×7 独特 linear **几乎不命中**，热路径仍是每次 `cudaMalloc` + `h3_cuda_gemm_f32_dw` 的 x H2D / EventSynchronize / y D2H；CFG=5 还会跑第二次 pad-token encode。
- [x] 已做无损修复（契约 `hi_first` / `unswizzle` / `pre_quant` 未改）：
  - TE 常驻 workspace（`d_x`/`d_y`/`d_w`/packed/scales），去掉每 linear 大块 `cudaMalloc/Free`
  - 同 host 指针跳过 x H2D（q/k/v、gate/up 共享激活）
  - GEMM 改为 `h3_cuda_gemm_f32_dd`，`gemm_ms==NULL` 时不再 EventSynchronize
  - AWQ `pre_quant` 改到 device kernel（CUDA 路径不再 host 扩一张 scaled x）
  - growable **host** unswizzle/pre_quant 缓存，覆盖 CFG 第二遍
  - packed 经 pinned staging 再 H2D（避免 raw mmap→GPU 缺页）
- [x] 远程 4090 A/B（cfg=5、reuse=off、INT8 DiT、cuDNN）：橙猫 seed=17；水下官方 `b_underwater.ids` seed=91。产物 `out/verify_r1r3_p2_te/`。
- [x] 主题未回退：橙猫仍单猫窗台；水下仍可见海龟且无人。两边 `latent_stats` 与 P1 逐字段相同。
- [x] P2 DoD：TE 暴露有明确下降，但 **未达 8–10s**。橙猫 te 18.3s→16.6s（layers 13.5s→10.7s），水下 te 23.4s→20.0s（layers 16.4s→13.4s）；uncond 第二遍约 5.6–6.6s。总时长橙猫 66.5s→64.6s，水下 76.3s→68.3s。剩余瓶颈仍是 350 次 packed H2D+dequant+skinny GEMM 与首遍 mmap/unswizzle，以及层间 host RMS/RoPE/GQA 迫使每 linear D2H。

---


### P3. VAE 冷启动 load 压降

**目标**

把 VAE 的冷启动 load 从“已有热缓存命中证据”进一步推进到“默认首次运行也可接受”，避免默认端到端体验被第一次 load 拖住。

**验收标准**

- 明确区分冷启动与热启动口径
- 冷启动 `load_ms` 较当前默认基线明确下降
- 热缓存命中证据继续成立，不因冷启动优化丢失
- 默认总时长得到实质改善，而不是仅把 load 挪到别的阶段

**建议顺序**

排第 4，放在 TE 之后。

**依赖**

- 依赖 P0：要先有干净默认基线
- 不依赖 P1/P2 的代码完成才能开始分析，但建议在主题恢复后再正式收口，以免重复复验
- 必须保留既有热缓存证据链

**建议排查范围**

- `h3_vae_int8.c`
- `h3_vae_int8_cuda.c`
- VAE 权重打开、首次 load、mmap/host cache/device cache、borrow/reuse 生命周期
- 生成路径中何时触发 VAE open/load，是否存在不必要的重复初始化

**主要风险**

- 可能只优化了“同进程第二次”而没有真正改善首次 load
- 可能通过隐式常驻显存换数值，需要明确这是不是默认允许的行为
- 若日志口径没有把 open/load/dequant/H2D 分开，容易误判收益来源

**是否阻塞最终收口**

**阻塞。** 当前目标是默认路径 `<=35s`，冷启动 load 仍明显占时就不能算完成。

**执行记录（2026-08-18）**

- [x] 已确认 P2 默认冷启瓶颈：`vae_weights_load` 日志 `dequant_ms≈3.8–5.5s`（host I8×scale 循环写出 ~7GB F32）+ `h2d_ms≈1.1s`（F32 H2D）。T7 `H3_VAE_WEIGHT_CACHE=1` 同进程第二次 load=0 仍成立，但不算默认冷启成功。
- [x] 已做无损默认路径修复（不依赖 `H3_VAE_WEIGHT_CACHE`）：
  - 新增 `h3_cuda_dequant_i8_rowscale_f32`（device I8×per-row scale → F32）
  - `h3_vae_int8_cuda_weights_load` 默认 mmap I8 H2D + GPU dequant 到原 resident F32 W；不再在 host 展开 7GB
  - `madvise(WILLNEED)` 预读 mmap；`H3_VAE_HOST_DEQUANT=1` 回退旧 host 循环；`H3_VAE_HOSTREG=1` 可选整文件 `cudaHostRegister`（默认关，避免把 encoder 页也 pin 进来）
- [x] 远程 4090 A/B（cfg=5、reuse=off、INT8 DiT、cuDNN、**未** export cache）：橙猫 seed=17；水下官方 `b_underwater.ids` seed=91。产物 `out/verify_r1r3_p3_vae/`。
- [x] 主题未回退：橙猫仍单猫窗台；水下仍可见海龟且无人。两边 `latent_stats` 与 P1/P2 逐字段相同。
- [x] T7 热缓存未丢：`--probe-vae-load --vae-repeats 2` + `H3_VAE_WEIGHT_CACHE=1` 第二次 `load_ms=0.0`。
- [x] P3 DoD：默认冷启 `vae_load` 进入 ~1–3s。橙猫 4.88s→**1.90s**，水下 6.62s→**1.96s**；VAE wall 6.59s→3.65s / 8.36s→3.72s；`blocks_ms` 仍约 0.96s。独立 probe（无 DiT 压力）~0.5s。总时长仅略降（64.6s→63.8s / 68.3s→68.0s），因为本跑 TE wall 上升（P2 已记录 TE 残量，P3 未再改 TE）。剩余冷启成本主要是 DiT 释后 ~1.8GB I8 mmap 缺页 + resident F32 `cudaMalloc`。

---

## 3. 收口复验门禁

只有在 P0-P3 全部通过后，才进入最终收口判断。

### 最终验收必须同时满足

- 默认路径日志干净：**无隐式 `reuse=1`**
- 双主题同时 PASS：
  - 橙猫：继续 PASS
  - 水下：海龟出现，恢复 PASS
- 默认总时长达到 **<=35s**
- TE 契约不回退
- VAE 热缓存证据不丢失

### 建议记录的最终结果

- 默认命令与默认环境
- 每阶段 wall-clock：TE / DiT / VAE load / total
- 默认双主题关键帧或成片锚点
- 负向确认：
  - 未默认打开 reuse
  - 未默认打开 magcache
  - 未把默认后端改成非既定路径
  - 未通过减 steps/layers 伪造达标

### 第 5 步统一复验结论（2026-08-18）

Parent ruling：P0–P3 已全部审查通过（P3 为 **APPROVE_WITH_NOTES**）。**不再重复 generate**；不为碰运气重跑 `<=35s`。收口前基线采用既有默认双主题 e2e：`out/verify_r1r3_p3_vae/`。短索引见 `out/verify_r1r3_closeout/README.txt`（只指向 P3，不复制大视频）。

- [x] 采用 P3 产物，跳过重复 generate
- [x] 默认配置干净：cfg=5、reuse=off、INT8 ConvRot DiT、无 magcache、sdpa=cudnn；日志无隐式 `reuse=1`
- [x] 双主题 PASS：橙猫单猫窗台；水下海龟+礁石无人
- [x] VAE 冷启 load 达标：橙猫 **1.90s**，水下 **1.96s**（P3 DoD）
- [x] 默认总时长 **`<=35s`：FAIL**（橙猫 **63.75s**，水下 **68.03s**）
- [x] 负向确认：未默认打开 reuse/magcache；未改默认注意力后端；未减 steps/layers 伪造达标

**收口判定：本轮不能按「已完成收口」对外表述。** 质量路径（默认干净 + 双主题 PASS + VAE 冷启达标）已可复验；时长门禁未过。阻塞 `<=35s` 的剩余大头是 TE wall（本跑橙猫 **23.77s** / 水下 **22.31s**；P2 曾记到 **16.6s / 20.0s**，口径仍是 ~16.6–23.8s）。

记录数字（`phase_ms`，来自 P3 日志，未重跑）

| | 橙猫 seed=17 `v4_orange_cat.txt` | 水下 seed=91 官方 `b_underwater.ids` |
| --- | --- | --- |
| te wall | 23.77s | 22.31s |
| dit | 18.23s | 18.10s |
| vae wall | 3.65s | 3.72s |
| vae_load | 1.90s | 1.96s |
| **total** | **63.75s** | **68.03s** |
| 主题 | PASS 单猫窗台 | PASS 海龟+礁石无人 |
| 日志 | `out/verify_r1r3_p3_vae/cat/generate.log` | `out/verify_r1r3_p3_vae/uw/generate.log` |
| 锚点 | `cat/cat_f0.png` `cat/cat_f3.png` | `uw/uw_f0.png` `uw/uw_f3.png` `uw/uw_f7.png` |

默认日志口径（两边相同）：`reuse=off magcache=0 cfg=5 sdpa=cudnn dit_format=INT8 ConvRot`；`reuse_schedule: disabled default_full_eval=1`；`dit_evals=8 dit_skips=0`。

---

## 4. 不再放进主修复链路的事项

以下事项本轮**不应**排在主链路前面：

- 再证明 DiT 优化有效
- 再证明 TE 契约是否成立
- 再补 VAE 热缓存命中证据
- 新的产品档、探索档、质量换速旋钮实验
- F2VA 首帧能力验证

原因很简单：这些都不是当前阻碍收尾的第一优先问题，提前做只会稀释修复带宽。

---

## 5. 完成收尾后再做的验证

这部分是**后验验证**，不是主修复链路前置条件。第 5 步已判定默认总时长未达 `<=35s`，**本轮不启动 F2VA**。

### F2VA 验证目标

在 P0-P3 完成、默认质量路径达到收口门槛后，再使用用户提供的 F2VA 素材验证“首帧条件 + 文本条件”的后续链路是否稳定。

### 验证素材

- 首帧图：`input/test_1_first_frame.jpeg`
- 文本提示：`input/test_1_prompt.txt`

### 放在后面的原因

- 当前默认路径尚未干净，双主题尚未完全恢复，先做 F2VA 只会把问题空间扩大
- F2VA 是扩展验证，不是这轮“收尾修复能否宣布完成”的核心门禁
- 如果主链路都还没收住，F2VA 结果也无法解释到底是在暴露新问题，还是旧问题未清干净

### 后验验收建议

- 在已收口默认链路上跑 F2VA，确认不会重新引入主题/条件污染
- 记录首帧跟随是否稳定、是否出现与文本冲突的主体偏移
- 若 F2VA 暴露问题，另开后续计划，不回插到本次收尾主链路

---

## 6. 一页式执行摘要

本轮收尾不再按“已经快做完”的假设推进，而是回到 4 个真正阻塞项：**默认 `reuse=1` 污染、水下主题 FAIL、TE 暴露时间过高、VAE 冷启动 load 过高**。建议顺序是：

1. [x] **先清默认配置**，拿到可信基线（P0 PASS）；
2. [x] **再修水下主题**，恢复双主题 PASS（P1 PASS）；
3. [x] **再压 TE 暴露时间**（P2 PASS：有下降，未到 8–10s）；
4. [x] **最后压 VAE 冷启动 load**（P3 PASS / APPROVE_WITH_NOTES：冷启 ~1.9s）；
5. [x] **第 5 步统一复验**：采用 P3 产物、跳过重复 generate。双主题 PASS、默认配置干净；**`<=35s` FAIL**（橙猫 63.75s / 水下 68.03s）。TE 仍 ~16.6–23.8s，阻塞时长门禁。

质量路径已可复验；时长未达标。本轮**不能**按“已完成收口”对外表述。
