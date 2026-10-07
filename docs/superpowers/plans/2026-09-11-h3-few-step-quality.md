# 他们如何用 4 / 8 步做出「很好」的视频（机制，非营销）

> **后续优化方案见** [2026-09-11-h3c-next-opt-plan.md](./2026-09-11-h3c-next-opt-plan.md)  
> **极速 SKU 实现：** [2026-09-11-h3c-5090-turbo-implementation.md](./2026-09-11-h3c-5090-turbo-implementation.md)  
> **For agentic workers:** 专章回答「少步为何好看」。禁止改 `h3_cuda_generate` 默认 / 单卡金线 KPI / 默认 `sdpa=cudnn`。不 commit。不编造未公开训练超参；推断打标。  
> **浅层普查：** [2026-09-11-h3c-accel-survey-multigpu-5090.md](./2026-09-11-h3c-accel-survey-multigpu-5090.md)  
> **机制深潜：** [2026-09-11-h3c-accel-survey-tech-deepdive.md](./2026-09-11-h3c-accel-survey-tech-deepdive.md)  
> **4-step LoRA 禁接金线：** [2026-08-25-sla-vs-h3c.md](./2026-08-25-sla-vs-h3c.md) · [2026-08-21-lightwan-h3-speed.md](./2026-08-21-lightwan-h3-speed.md)

**Goal:** 拆清「8 步好看」和「4 步也好看」分别来自哪一层训练，而不是哪一个推理开关。

**本仓金线（已钉）：** INT8 ConvRot + cuDNN SDPA + **8-step + cfg=1** + 864×480 有声。4090 cafe ~75s / 5090 cafe warm ~66s（DiT ~31.4s）。Sage / Tea / Mag **不过 MAE 门**。CFG 已在官方权里，推理无负分支。

---

## 0. 结论先行

1. **官方 H3 的 8 步不是「把 50 步 Euler 截断」。** 发布的 transformer 是 **CFG-distilled**（SGLang denoise loop 原话：*MiniMax H3 only supports cfg-distilled checkpoints*；每步只 forward 正分支一次）。本仓 / FastH3 / SGLang 质量合同都是 **cfg=1、无 negative**。50-step BF16 是框架用来报倍数的**另一条离散化**，像素轨迹与 8-step 金线不是同一条 ODE。

2. **社区「4 步也很好看」是第二次蒸馏，换了学生权。** FastH3 DMD2、LightX2V / larryvrh Turbo LoRA、RH 自训加速权，都是在已经 CFG 蒸馏的 H3 上再训一条 **少 NFE 学生**（几乎都是 LoRA，不是把 INT8 金线 `--steps` 改成 4）。推理时还剩：merge 后的学生权 + 学生自己的 sigma 格点 + 仍 cfg=1。教师 50/20/30 步轨迹**不再被执行**。

3. **Sol Super Accel 的「4 步」不是 H3 四步成片。** 它是 **H3 4-step 低分草稿 + LTX-2.5 3-step 精修**。成片质量来自第二套模型；音频来自草稿 PCM。步数少但**多了一个模型**。页面写明非无损。

4. **VDN 停在 8-NFE 是有意的。** hybrid 注意力加速的是 **dense 8 步 / 50 步轨迹上的每步 FLOP**，不是把语义步砍到 4。另有一条 DMD2 turbo LoRA（250 step、无 GAN）把 VDN 自己的 50-NFE 对齐到 8-NFE；README 写明**故意不做 4-step**。

5. **Cache / Sage / compile 不减少语义步。** 它们改的是「这一步里算多少 / 用什么核」。8-step 蒸馏后再 skip，本仓 Mag/Tea 已测视觉不过门；LightX2V #579 同方向。勿与蒸馏混谈。

6. **「看起来很好」用的尺子不是本仓 PPM MAE。** 宣传片对人眼、对 50-step BF16 / 30-step base 并排；本仓金线已经是 **8-step INT8 + cuDNN**。把已经蒸馏的 8 步再砍到 4 **而不换匹配 LoRA**，是未蒸馏 4 步，主题/运动崩。4-step 对学生自己的教师可以很好，对金线再砍通常崩。

**一句话：** 本仓已经在官方 8-step CFG 蒸馏轨迹上。再 4-step 必须 **另训 LoRA / 快档 SKU**，不是推理开关。

---

## 1. 官方 8-step：CFG 已经进权

### 做什么

H3 是 **rectified-flow / flow-matching** DiT：模型预测 data-ward velocity，Euler 把 latent 从噪声推到数据。未蒸馏的 CFG 每步要跑 **cond + uncond** 两路再线性合成。发布权把这条合成 **写进网络**：正分支 velocity 直接当更新信号。

SGLang `minimax_h3/denoise_loop.py`：

> Per step, the positive presentation is forwarded exactly once.  
> The model's raw positive velocity is the update signal; MiniMax H3 only supports cfg-distilled checkpoints.

FastH3 `basic_fasth3.py`：`guidance_scale=1.0`，注释 *MiniMax-H3 is guidance-distilled; FastH3 inherits that contract.*

本仓：`h3_dit_w4a8.h`「H3 is CFG-distilled — keep scale=1」；质量档 `--cfg 1`。`cfg=5` 在 8-step INT8 上后期塌成亮斑（OOD，不是「更强引导」）。

### 训练发生在哪

MiniMax **未公开** CFG 蒸馏的教师 NFE、loss、数据、是否同时做了 50→8 的 step distillation。能钉死的只有推理合同：发布 transformer **只支持 cfg-distilled 用法**。

**推断（非事实）：** 发布权至少做了 guidance distillation；8-step 是 MiniMax / 本仓 / FastH3 / vLLM 质量栈的 **产品采样格点**。Comfy 官方模板常走 **20-step `res_multistep`**——那是另一条调度，不是把金线 8 改成 20。本仓 av20（20 步）墙钟 ~140s，**同片不更清**，整段加步会拉歪 8-step 轨迹。

### 推理时还剩什么

| 项 | 官方 8-step 金线 |
| --- | --- |
| 权 | 发布 DiT（本仓再量化成 INT8 ConvRot） |
| CFG | **无**。cfg=1，无负分支，无 CFG 并行可切 |
| 步 | 8 次 DiT forward（sigma 格点 9 个含 0；各家 `num_inference_steps` 计数口径略有出入，本仓 CLI `--steps 8` = 8 次评估） |
| 调度 | Euler；视频 `flow_shift` 社区常 12、音频 3（RH / VDN 公开请求） |
| 注意力 | dense（本仓 cuDNN） |

### 质量代价

相对「同一套权跑 50-step BF16」：**不是同一条采样轨迹**，不能当 GT。RH / Sol-Engine 用 50-step 当慢基线报倍数；那是 **更长的 Euler 离散化 + 未量化 BF16 + 常是另一套注意力**，不是「官方更真的片」。相对未蒸馏、每步双路 CFG 的老扩散模型：CFG 蒸馏换来 **每步 NFE 减半** 和 8 步可用，代价是 cfg>1 变成 OOD。

**禁止：** 把 50-step 参考片的像素当 8-step 金线的 MAE 分母；把 `--steps 8` 改成 4 冒充官方少步。

---

## 2. 4-step 各家配方（第二次蒸馏）

统一四列：**做什么 / 训练在哪 / 推理还剩什么 / 质量代价**。未公开超参标推断。

### 2.1 FastH3（DMD2 + 可训练 VSA）

| | |
| --- | --- |
| **做什么** | 把基座 H3 的 ~49 次 DiT forward 蒸成 **5 个 sigma 点 = 4 次 Transformer forward**。日程写死在 `basic_fasth3.py`：`t=1000,750,500,250 → 0`。video-video 保留约 10% tile（训练 sparsity 0.9，tile=64）；**text/audio 查询永远 dense**。 |
| **训练** | **DMD2**：frozen teacher + critic，score 差作学生信号（FastVideo `docs/distillation/dmd.md`；Wan 配方公开，H3 训练「即将」）。学生是 **LoRA**（HF `FastVideo-FastH3-4-step-Preview-v1-LoRA`），不是全量换骨干。推荐 ckpt **VSA / Data-Free**（只吃 prompt、无目标视频，公开 step 1300）。`to_gate_compress` 基座不存在，loader 零初始化，学生才带训练门。VSA LoRA **不能**丢到 dense 路径（launcher 硬拒绝 `--no-vsa`）。Teacher/critic 走 dense，给学生一个全注意力靶。 |
| **推理还剩** | merge 后的 4-call 学生 + 训练过的 VSA 门 + `guidance_scale=1.0`（继承 CFG 蒸馏，**不再跑 CFG**）。可选 compile / 并行 VAE，改 FP 序，`profile=all` 自称 report-only。 |
| **质量代价** | **非无损**。Preview **T2VA only**。运动/细节/部分音频可低于 Base。公开材料写下一步才考虑 **8-step 质量档**——等于承认 4-call 是快档不是金线。对照分母是 ~49-step BF16，不是本仓 8-step INT8。 |

### 2.2 LightX2V Turbo / larryvrh Turbo LoRA

| | |
| --- | --- |
| **做什么** | 社区 **few-step LoRA** 打在发布 H3 上。LightX2V 同时提供 **4-step 与 8-step** LoRA；T2VA 图默认是 **8-step v1.0**，不是默认 4。larryvrh 卡原文：as few as **4** sampling steps instead of the usual **~20**（~5×）；作者写 **4 是下限、6–8 更好**，4 步大运动会 smear。Turbo-SLA 变体另加 85% 动态稀疏 + Sage2，任务是 **4-step FL2V**，不是 T2VA 金线。 |
| **训练** | **LoRA，冻骨干。** larryvrh / LightX2V **未公开**完整 DMD vs consistency vs rectified-flow 损失与教师步数。能钉死：对照宣传是 **~20 / 30-step base**，不是本仓 8-step。VDN 把 `minimax_h3_turbo_v4_step600_ema.safetensors` 当 **外部 few-step 初始化**再在自己流形上 DMD2（见 §2.4）——说明该 LoRA 已被下游当 8-NFE 教师种子，不说明它在 dense H3 上等价于官方 8-step。 |
| **推理还剩** | `--lora-path` + 学生推荐步数（4 或 8）+ cfg=1。RH 公开复现：larryvrh v4 EMA、`num_inference_steps=4`、`flow_shift=12`、`audio_flow_shift=3`。LightX2V ref2v 公开档是 **8-step** LoRA。 |
| **质量代价** | 对人眼 / 对 20–30 步 base 可以「很好」。相对已蒸馏的 **8-step INT8+cudnn 金线**：换轨迹 + 换（BF16）LoRA，本仓无产品 LoRA 加载器。未蒸馏把 8 改 4 = 红线崩主题。高动态 RH 自己建议回到 **8**。 |

### 2.3 RunningHub Lightning（自训 4-step + 执行三件套）

| | |
| --- | --- |
| **做什么** | README 三层显式相乘：① 后训练步数蒸馏 → 请求 **4 步**；② SageAttention2 + Cache-DiT + `torch.compile`；③ TP2+Ulysses4。同一 8×6000D：50-step 348.8s → **仅换权 4-step 43.0s（8.1×，占节省的 95.5%）** → 再加三件套 28.7s（再 ×1.50，占 4.5%）。 |
| **训练** | **RH 自训加速权未公开。** 公开复现用 larryvrh 4-step 或 LightX2V ref2v **8-step**，**不是** 12.2× 那次实验的权。蒸馏目标 / 教师步数 / 是否全量 **未公开**（推断：与社区 Turbo LoRA 同类的 few-step 后训练，打在 BF16 骨干上）。 |
| **推理还剩** | 4-step 学生（或公开替代 LoRA）+ Sage2 量化注意力 + Cache-DiT（少步上 TaylorSeer 默认关；`max_warmup_steps=4` 与 4-step 等长，跳步空间极窄）+ compile。**仍 cfg=1。** 高动态档请求 **8 步**。 |
| **质量代价** | 不宣称无损。无公开 MAE。Sage 内部量化 + 蒸馏 + Cache 都改计算。12.2× 的分母是 **同机 50-step BF16**，禁止写成相对本仓 66s。 |

### 2.4 VDN 的 8-NFE turbo（故意停在 8，不是 4）

| | |
| --- | --- |
| **做什么** | 两条独立的刀。**(A) Hybrid attn**：chunk 窗 softmax（精确局部）+ Video Delta 线性（远端）+ 边界锚；冻骨干，加线性支 + 小 LoRA。**(B) DMD2 8-NFE turbo**：另训 `turbo` 适配器，把 **VDN 自己的 50-NFE** 对齐到 8 次评估。`num_inference_steps` 计 sigma 格点：9 点 = 8 次模型评估。 |
| **训练** | 公开、可复现。A1 逐层对齐线性支（200）→ A2 端到端（500）→ B QKVO LoRA+线性（2000）→ **DMD2 无 GAN、250 step**。yaml 原文：generator = 冻 Stage-B VDN + 可训 `turbo`（初始化自 larryvrh v4 EMA）；**real score = VDN 自身（generator 减 turbo）**，因此 turbo **只学 50→8，且留在 VDN 流形内**；fake score = VDN + `fake`（rank 128）。data-free，只吃 prompt 行。损失：`distribution_matching_loss` = `(x0_fake - x0_real) / mean|x0_g - x0_real|` 包装成对 `x0_g` 的 MSE；fake 再回归 generator 样本上的 velocity。 |
| **推理还剩** | merge 后的 hybrid +（快档）turbo LoRA；**仍 8 Euler 步**；cfg 沿用 H3 蒸馏。50-NFE 档不挂 turbo。无 4-step 产品路径。 |
| **质量代价** | 50-NFE hybrid 口头 near-lossless vs **dense MiniMax BF16**（不是本仓 MAE）。8-NFE 是明确折中。计时排除 VAE。停在 8 是质量判断，不是算力不够。 |

### 2.5 Fal H3 Max（闭源；推断）

能核实：fal 后训练开源 H3；5s 768p `timings.inference ≈ 2.5s`；相对官方托管 ~35× 吞吐。**无步数、无核、无权。**

**推断：** 要打到 ~2.5s，信息论上几乎必然是 **极少 NFE（约 4 或更少）+ 常驻/融合**，不是在官方 8-step INT8 上拧开关。未证明与本仓 8-step 同分布。秒数不可复盘。

---

## 3. Draft + refine：步数少，模型多

Sol Super Acceleration（仓内 v2，`sol-engine-excerpts/super_acceleration/`）：

```
GPU A  Stage1 H3 FL2VA   896×512 / 124f / 4 DiT forwards
         LightX2V 4step_v0.1_544p LoRA  scale=0.0625
         cache 关 · TAEH3 出草稿像素 + H3 PCM
         ── TCP 直传张量（禁止中间 MP4）──
GPU B  Stage2 LTX-2.5    原版 LTX VAE encode + x2 latent upsample → 1344×768
         3 updates，sigma 0.909375, 0.725, 0.421875, 0.0
         layer0 dense，1–47 Sol-Attn cute_sm100 · TAEHV 成片
         扔掉 Stage2 音频，mux 回 Stage1 PCM
```

| | |
| --- | --- |
| **做什么** | 低分 4-step **草稿**（仍是社区 4-step LoRA，scale 很小）+ **另一个**视频模型做 3-step 精修 + 轻量 VAE。4 GPU = 两对独立流水，**零 TP/SP**。 |
| **训练** | Stage1 LoRA 来自 LightX2V（H3 侧）。Stage2 refiner 训在 **原版 LTX VAE latent** 上，故 encode **不能**换 TAE。H3 官方 8-step 权 **不负责** 768p 成片。 |
| **推理还剩** | 两套常驻运行时、两套 VAE、一次张量 handoff。算法 cache 两阶段都禁用（少步上 cache 不是他们的刀）。 |
| **质量代价** | 页面 **明确非无损**。仓内 `perceptual_quality_gate_run: false`，`speedup_claim: null`。成片观感来自 **LTX refine + TAEHV**，不是「H3 四步已经够好」。口型合同在 **草稿 PCM**。博客 6.85s / 22.2× 的分母是已发表 SGLang 152.3s @768p GB200，不是本仓 66s。 |

同类结构（不是 H3，只说明机制可抄）：FastVideo LongCat 16-step 蒸馏 480p → refinement LoRA + BSA 升到 768p。质量来自第二段，墙钟两段相加。

---

## 4. Cache / Sage / compile：不是少步

| 刀 | 改什么 | 语义步？ | 8-step 金线上 |
| --- | --- | --- | --- |
| Tea / Mag / FB / Cache-DiT | 步间或块级 residual 相似则 **复用** 上一输出 | **不减**请求步数；减的是实际 DiT eval | 蒸馏把中段冗余榨干。本仓 Mag 0.08：skip 但历史 MAE 52–68 FAIL。Tea 0.04：skip 1，不过门不进默认。FB 0.04：8/0 零 skip。LightX2V #579：4 步蒸馏后再开 Tea **几乎不跳**。RH 在 **已经 4-step** 上开 Cache-DiT，增量只占 12.2× 里的 4.5% 之一 |
| Sage2 / sage3 | 量化注意力 | 仍 8 或 4 次 forward | sage3 5090 DiT −17%，MAE~27 FAIL |
| Sol / VSA | 稀疏/近似 Softmax | 同上 | sol_cute MAE~35 FAIL；VSA 必须跟学生 LoRA 一起训 |
| `torch.compile` | 改运算序、减 launch | 不减步 | RH/FastH3 自己承认改数值；本仓静态 C 无对等物 |

**机制：** 50-step 长轨迹中段有「无聊细化」可 skip；**CFG 蒸馏 8-step 的每一步都是大更新**，单步误差没有后续步可抹平（推断，与 Mag FAIL / #579 同方向）。

---

## 5. 为何「看起来很好」，以及本仓再少步意味着什么

### 5.1 三把尺子

| 尺子 | 谁在用 | 4-step 对学生教师 | 未蒸馏 8→4 对金线 |
| --- | --- | --- | --- |
| 人眼 / 宣传片 / Elo | Fal、RH 验收清单、HF demo | 可以「很好」 | 主题 smear、口型乱（本仓红线经验） |
| 50 / 30 / 20-step BF16 并排 | FastH3 14×、RH 12.2×、Turbo-SLA 2.5×、Comfy 20-step 模板 | 设计目标就是赢这个分母 | **类别错误**：金线已经不是这条轨迹 |
| 本仓 PPM MAE ∩ 有声 vs **5090/4090 cudnn 8-step** | h3.c 金线 | **未测过门；预期 FAIL**（换权+换步） | **必崩**（无匹配学生） |

HF Turbo-SLA demo 是 **30-step base vs 4-step turbo** 并排。larryvrh 对照是 **~20 步**。没有任何公开材料证明：同一 INT8+cudnn 权、同一 cafe seed，4 步能过本仓 MAE。

### 5.2 对本仓：已经 8-step，再 4 必须另训

Amdahl（5090 cafe，已测量级）：e2e ~66s ≈ TE 12–14 + **DiT 31.4** + VAE ~10.5。

| 动作 | DiT 次数 | 权 | 质量门 | e2e 含义 |
| --- | ---: | --- | --- | --- |
| 金线（现状） | 8 | INT8 + cudnn | 自身 | ~66s |
| `--steps 4` 不换权 | 4 | **同一 INT8** | **崩**（未蒸馏 4 步） | 数字无意义 |
| 公开 4-step LoRA merge 进 INT8 | 4 | **换轨迹** | 另册；预期过不了金线 MAE | 快档 SKU，须独立 CLI |
| 公开 **8-step** turbo LoRA | 8 | 换权、同 NFE | 仍非金线，但比 4-step 更接近「保效果」 | SLA 文：另开产品档 |
| Super Accel 式 draft+refine | 4+3、换 LTX | 两套模型 | 非无损；32GB 不友好 | 快档，永不默认 |
| Sage / Tea / Mag | 仍 8（或 skip 1） | 同权近似 | **已测不过门** | 金线有效加速 ≈0 |

本仓 **没有** 产品级 LoRA 加载器进默认路径；BF16 LoRA 打在 FP8/BF16 基座上，接到 INT8 ConvRot 要么反量化（丢掉 INT8 金线），要么需要一起蒸馏的 INT8 学生（没有）。

### 5.3 硬停止

1. 不要把 `--steps` 改成 4 当加速。  
2. 不要把 4-step LoRA merge 进 INT8 金线。  
3. 不要把 RH 12.2× / FastH3 14× / Super Accel 22.2× 写成 66s 的倍数。  
4. 不要把 Cache/Sage 说成「也是少步」。  
5. 若产品要快档：独立 preset（如 `--preset fast8` / `fast4`），help 写「非金线」，MAE 阈值不得冒充金线。优先考虑 **公开 8-step turbo LoRA 侧车**（同 NFE、换权），再考虑 4-step 或 draft+refine。

---

## 6. 步数来源表

口径：NFE = DiT（或 refine 模型）前向次数。CFG 已蒸馏则每步 1 次 forward。

| 配方 | 语义 NFE | 步数从哪来 | 训练（公开部分） | 推理还要不要 CFG | 质量对照分母 | 对本仓 |
| --- | ---: | --- | --- | --- | --- | --- |
| **本仓金线** | **8** | 官方 CFG-distilled 采样合同 | MiniMax 发布权（CFG 蒸馏配方未公开）+ 本仓 INT8 | **否**（cfg=1） | 自身 cudnn PPM MAE ∩ 有声 | **默认** |
| MiniMax / SGLang / FastH3 基座「长轨迹」 | ~49–50 | 同一套 CFG-distilled 权的更密 Euler | 无额外学生；只是改 `num_inference_steps` | 否 | 常被当成「慢质量基线」 | **不是** 8-step GT；本仓 20 步已证不更清 |
| Comfy 官方模板 | 20 | `res_multistep` 另一条调度 | — | 否 | 模板观感 | 勿把金线改 20 |
| **FastH3 Preview** | **4** | DMD2 学生日程 `1000,750,500,250→0` | LoRA + VSA；Data-Free / synthetic；teacher+critic dense | 否（继承） | ~49-step Base BF16 | 快档隔离跑；禁接 INT8 |
| **larryvrh Turbo v4** | **4**（作者建议 6–8） | 社区 few-step LoRA | LoRA；损失/教师步 **未公开**；宣传 vs ~20 | 否 | ~20-step | RH 公开复现用；禁默认 |
| **LightX2V T2VA turbo** | **8**（默认图）或 4 | 分 SKU 的 LoRA | LoRA；4-step 另有 SLA 稀疏变体 | 否 | 30-step base（SLA 卡） | 8-step turbo = 较近的快档候选 |
| **LightX2V ref2v** | **8** | 专用 8-step LoRA | LoRA | 否 | 未对金线 | RH `MODEL_VARIANT=ref2va` |
| **RH Lightning 宣传** | **4**（高动态 **8**） | RH 自训权（未公开） | 后训练蒸馏；细节未公开 | 否 | 同机 50-step BF16 | 抄拓扑不抄 4-step |
| **VDN 50-NFE** | 50 | hybrid 换 attn，**不减步** | A1/A2/B 线性支+LoRA | 否 | dense MiniMax BF16 | 研究 attn；显存不够 5090 |
| **VDN 8-NFE turbo** | **8** | DMD2 turbo，教师=VDN 50-step | 250 step、无 GAN；init larryvrh | 否 | VDN 50-step | 故意不做 4；同 NFE 换 hybrid+turbo |
| **Sol Super Accel** | **4 + 3** | H3 草稿 LoRA + **LTX** 3-update | LightX2V 4-step LoRA；LTX refiner 另训 | H3 侧否 | 非无损；无 MAE 门 | 快档模板；禁写 66s |
| **Fal H3 Max** | **未公开** | 闭源后训练 + 自研引擎 | 未公开 | 未知 | 托管 Elo / 吞吐 | **推断**极少 NFE；不可复盘 |

未 git commit。默认路径未改。
