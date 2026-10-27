# 4090 INT8 产品路径 · 优化菜单（Beyond-DMAD + 无损 · 2026-10-28）

> **Asia/Singapore.** Analysis-only（本轮不抢 GPU；C041 缩窗 / C015–C042 act-FP16 由另一执行器施工）。  
> **产品硬约束：** INT8 ConvRot；注意力 = cuDNN SDPA（Sage3/SOL/Kitchen **不在** 4090 产品路径；`sol_cute` lib=0）；主臂 **DMAD-4 或 Euler-8**；PDMD **仅对照**；FaceRefine **STOPPED**；NVFP4 W4A4 **terminal BLOCK**；禁无意义 convert/fallback。  
> **门：** 480p×5s 复杂 prompt；burn-in scheme+warm；优先 **优于 DMAD fewstep** 与 **waffle-passable 无损加速**。  
> **证据源：** [`full-optimization-catalog`](2026-10-27-full-optimization-catalog.md)、[`beyond-dmad-and-lossless`](2026-10-27-beyond-dmad-and-lossless.md)、[`ghosting-and-small-face`](2026-10-27-ghosting-and-small-face.md)、[`facerefine-p10`](2026-10-27-facerefine-p10.md)、C040/C062/C073 KPI notes @ `968b99f`/`5e6b8e2`/`5d8c9cd`。

---

## 0. 一句话

4090 产品栈上：**每步 INT8 GEMM 已近峰 → 无损墙钟只剩链路/VAE/测量卫生的薄肉**；**大墙钟差仍来自 NFE（DMAD-4 ≈ Euler-8 暖 DiT 1.61× @480p）**；**远脸崩是画布像素预算，不是 FaceRefine 能缝合的、也不是「再加一张通用 LoRA」就能发明的 34px 五官**。下一步并行两刀：**(1) 让 C041/C015 交出 ~GiB 余量**；(2) **质量轴走「抬画布 / 合同内 detail LoRA / 正式二采 SKU」**，不重开 crop→F2VA→stitch。

---

## 1. 轴分离（禁止合成一个验收）

| 层 | 消什么 | 本仓代表 | 质量尺 |
|---|---|---|---|
| **Kernel/engine** | 每步算子+中转+launch | INT8 链路、EPI fuse、VAE cast、QKV merge、RoPE memo、L2 pin、sync forbid | 同配置 waffle / BIT_ID |
| **Sampler/schedule** | NFE、σ、renoise | Euler-8 金线；DMAD `lcm_renoise` 12/2；PDMD 对照 | `--fewstep-contract` + 目检 |
| **Architecture** | 注意力/token/窗/streaming | cuDNN dense；窗大小；layerwise stream；VDN/SLA 观察 | 独立 SKU 或能力门 |
| **Training** | 学生/补丁权重 | PDMD/DMAD LoRA；C035 detail/refiner；人脸/身份 LoRA；SGMD 类 | 合同绑定；非 waffle 跨臂 |

---

## 2. Top 菜单（12 项 · 按「现在该不该做」排）

字段：类别 · 期望收益 · 风险 · waffle · 4090 可行 · 工作量 · **why now / defer**

| # | ID / 主题 | 类别 | 期望收益 | 风险 | waffle | 4090 | 工作量 | Why now / defer |
|---:|---|---|---|---|---|---|---|---|
| **1** | **C041 窗 30→24** | VRAM | ~GiB 级峰值余量；可能 −WC（H2D 更频） | L | 同配置可过（能力探针） | ✅ | 低 | **NOW（他机在跑）** — C040 768p free≈0.8 GiB，缩窗是最便宜的余量探针；**不解默认** |
| **2** | **C015/C042 act-FP16 中转** | VRAM / 近无损 WC | 激活带宽↓；768p+ 余量；WC 可能小正/零 | M | **非 BIT_ID**；需同配置 waffle | ✅ | 中 | **NOW（他机在跑）** — 与 C041 互补；成功标准见 §4 |
| **3** | **抬画布质量 SKU**（960×544 / 1152×640 PDMD上限；1344 Euler） | quality@sameNFE | 小脸五官可辨（主杠杆） | L | 跨分辨率非 waffle 金线 | ✅ 已证 | 低（产品档） | **NOW（质量）** — [`ghosting`](2026-10-27-ghosting-and-small-face.md)：远脸=像素预算；**加步不是免费清晰度** |
| **4** | **C035 detail / taomate-0.7 refiner LoRA** | quality@sameNFE / better-than-DMAD polish | 少步过饱和/糊边恢复；非发明远景像素 | M | 有损臂；独立合同 | ✅ | 中 | **BLOCK 2026-10-28** — 无本地权（[`exp-g2-block`](2026-10-28-exp-g2-c035-refiner-block.md)）；权重到位再扫 |
| **5** | **人脸/身份 LoRA（训练臂 · 非 FaceRefine）** | training / quality | 近景身份/口型更稳；**不能**单独救 ~34px 远脸 | H | 新合同 | ✅ 推理可挂 | **高（训）** | **有条件 NOW 设计、DEFER 重训** — 见 §2.1；需 face-heavy 数据 + 与 DMAD/Euler 合同正交 |
| **6** | ~~**C011 VAE cast/prescale**~~ | — | ≈29 ms VAE meat | — | — | — | — | **CLOSED NO_WC_WIN** [`nsys-close`](2026-10-28-c011-vae-cast-prescale-nsys-close.md)；next DeepSeek **C016 PDL dig** |
| **7** | **C018 QKV merge（先 diff 现状）** | lossless WC? | 历史估 −0.6~−1.2s；可能已覆盖 | M | INT8 序可能非 bit | ✅ | 低探+中改 | **NEXT probe** — catalog 标 *maybe stale*；先证明缺口再写核 |
| **8** | **C064 TE↔DiT↔VAE 安全重叠** | VRAM/WC hygiene | 峰值错峰；冷启动/预热 OOM 少 | M | 应 bit | ✅ | 中 | **AFTER C041/C015** — 白名单已 landed；真重叠仍门控 |
| **9** | **C043 layerwise weight streaming** | VRAM 能力 | 768p+ / 更大窗「能跑完」 | M | WC 中性~负 | ✅ | 高 | **DEFER** — 余量决策前提是 C041/C015 数字 |
| **10** | **正式二采 / latent upscaler SKU（C097）** | architecture/product | 480 草稿→高分比直出便宜 | M | 有损 | ✅ 部分代码 | 高 | **DEFER 产品档** — 非 DiT 主墙；二采少步仍伤小脸（需 face-aware 二采合同） |
| **11** | **SGMD / 下一代 fewstep 蒸馏（外研）** | better-than-DMAD | 运动/动态优于 DMD2 类（论文人评）；需训 | H | 新学生合同 | 推理后可 | **很高** | **OBSERVE** — PDMD 已优于同 NFE DMAD（论文 Table2）；SGMD 是「再训一条学生」不是推理开关 |
| **12** | **C036 UniPC/DPM++（无训）** | sampler | ≤8 NFE 训练-free；难打蒸馏 4NFE | M | 轨迹变 | ✅ | 低 | **LOW** — 与 LoRA 正交；不作击败 DMAD 主叙事 |

### 2.1 LoRA vs FaceRefine（用户问）

| | FaceRefine（已停） | LoRA 补丁 |
|---|---|---|
| 机制 | crop → H3 F2VA **全量 regen** → 椭圆缝合 | 改 DiT 轨迹/高频先验（少步学生或 detail/face 适配器） |
| 远脸 RCA | ~34px → 极少 latent patch；**物理不可逆** | **同物理上限**：LoRA 不能在 34px 槽里「生成」可辨五官 |
| 已证 | Euler-8/16 face-CU **不比** PDMD-4 更清；多步放大 framing drift（[`facerefine-p10`](2026-10-27-facerefine-p10.md)） | PDMD/DMAD 少步 LoRA **已**是主质量/速度杠杆；detail LoRA = C035 idea |
| 结论 | **勿重开** stitch 主线 | **可以加 LoRA**，但定位是：**(a)** 合同内少步质量（已有）；**(b)** detail/refiner 抛光；**(c)** 可选 face-heavy 微调。**主修复仍是抬画布 / 二采 SKU** |

**推荐 LoRA 实验顺序（不训则只 A/B 已有权重）：**  
1. DMAD-4 金合同 +（若有）社区 detail/refiner @ scale sweep 0.3–0.7；  
2. 同 seed 抬到 960×544 / 1152×640 看远脸；  
3. 仅当 (1)(2) 不够再开 face-identity 训练票（数据/算力另批）。

---

## 3. 已试且关闭（勿再诉讼）

| ID | 裁决 | 关键 SHA / 文档 |
|---|---|---|
| **C001** NVFP4 W4A4 | terminal BLOCK（多层能量塌缩；非 scale2/SF 单修） | `ebd77a5`；`nvfp4-energy-collapse-block` |
| NVFP4→host dequant→INT8 | **disarm** 无意义旁路 | `a7d7375` |
| KEEP=attn | 实验臂；mosaic 质量；非产品默认 | `0d1848f` |
| **C003** EPI_FUSE_SL | BIT_ID OK 但 hot dit **+46%** → **keep OFF** | `ed1ce72` / `cc427ef` |
| EPI_FUSE_INK / P2 skip-ConvRot | ~0% WC / DISARMED | beyond-dmad §6 |
| **C017** L2 pin | BIT_ID；fair hot **无墙钟**；opt-in | `95ab073` |
| **C068/C069** RoPE memo | default-on；dit Δ noise | beyond-dmad §6.2 |
| **Z02** 更快 INT8 GEMM | 峰 98–99%；关闭 | deadend-operators |
| **Z05** CUDA Graph 默认开 | <噪声；对照关 | deadend-operators |
| **Z21** MagCache/有损 reuse @ 少步默认 | 主题禁 | deadend-quality |
| **C007** sol autotune | **N/A @4090 cudnn** | `3cee9ff` KPI |
| **FaceRefine** | STOPPED | `fbde594`/`0881ecc`；facerefine-p10 |
| **C040** 账本 | 480p+768p **landed**；768 peak_dit≈22.6–22.7 GiB | `968b99f` |
| **C062** TE_AUTO | clip2 te=0 KPI-closed；默认仍 off | `5e6b8e2` |
| **C073** dump→replay | waffle Δ=0 + rgb bit-id | `5d8c9cd` |

**KPI 锚（4090，Asia/Singapore 2026-10-28）：**  
- DMAD-4 vs Euler-8 @480p：暖 dit **1.61×** / 暖 e2e **1.43×**（`3cee9ff` 笔记）。  
- 768p：Euler/DMAD 均可跑满；free@peak **~0.8 GiB**。

---

## 4. 下一刀 2–3 个实验（成功标准写死）

> 不与 C041/C015 GPU 抢卡；下列标准供他机收口或紧随其后。

### E1 — C041 缩窗探针（VRAM）

- **配置：** 1344×768×5s · Euler-8 INT8 · cudnn · `--skip-audio` · window **30 vs 24**（仅探针）。  
- **PASS：** peak_dit 降 ≥0.5 GiB **或** free@peak ≥1.5 GiB；无 OOM；成片无「window≈6 噪声」级崩。  
- **FAIL / 不解默认：** 暖 WALL 回归 >+8% 且余量增益 <0.3 GiB。

### E2 — C015/C042 act-FP16（VRAM + 近无损）

- **配置：** 同 768p Euler-8；act FP16 on vs FP32/基线中转。  
- **PASS：** peak_dit 明显降（目标再挤出 ≥0.5 GiB）；**同配置 waffle PASS**（允许非 BIT_ID）；无马赛克/能量塌。  
- **FAIL：** waffle FAIL 或视觉马赛克 → 保持 off。

### E3 — 质量：画布 ×（可选）detail LoRA（不重开 FaceRefine）

- **配置：** 同一复杂 p10；臂 A Euler-8 @480p；臂 B Euler-8 @960×544；臂 C DMAD-4 @480p ± detail/refiner scale∈{0,0.5}（有权重才跑 C）。  
- **PASS（脸）：** 远景脸 YuNet 侧长或人眼五官可辨：**B ≫ A**；C 相对 A 减少 smear/过饱和且无新伪影。  
- **PASS（速度叙事）：** 不把 B 的更慢墙钟记成「加速失败」——这是 **质量 SKU**。  
- **禁：** crop-stitch FaceRefine；跨 sampler 混 waffle。

---

## 5. 明确不做（本窗口）

- 重开 NVFP4 默认 / chain / host-dequant 旁路  
- 默认打开 EPI_FUSE_SL / CUDA Graph / MagCache  
- 4090 上武装 sol_cute/Sage3 当产品加速（lib=0）  
- FaceRefine 主线；2NFE 冲观感当通过  
- 与 C041/C015 抢同一张 4090 长跑  

---

## 6. 修订

| 日期 | 说明 |
|---|---|
| 2026-10-28 | 初稿：4090 INT8 约束下重排 Top12；关闭项对齐 catalog；LoRA vs FaceRefine；E1–E3 成功标准。分析-only，无 GPU。 |
