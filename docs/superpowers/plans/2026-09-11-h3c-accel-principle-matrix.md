# h3.c 加速原理对照（机制轴）

> **不是实现计划。** 禁止改 `h3_cuda_generate` 默认 backend / steps / mag / tea / fb / sage / sol / ulysses。不 commit。不编造秒数。  
> 综合：[next-opt-plan](./2026-09-11-h3c-next-opt-plan.md) · [few-step-quality](./2026-09-11-h3-few-step-quality.md) · [accel-survey-tech-deepdive](./2026-09-11-h3c-accel-survey-tech-deepdive.md) · [pcie-multigpu-lessons](./2026-09-11-h3c-pcie-multigpu-lessons.md) · [vs-sglang-vllm-tech-deepdive](./2026-08-25-h3c-vs-sglang-vllm-tech-deepdive.md) · [4090-accel-plan §4](./2026-08-25-h3c-4090-accel-plan.md)（4.44× 梯子）  
> 秒数只在**各自公开分母**下引用；禁止写成相对本仓 8-step INT8 金线（5090 cafe warm ~66s / DiT ~31.4s）的倍数。  
> **实现计划：** [2026-09-11-h3c-5090-turbo-implementation.md](./2026-09-11-h3c-5090-turbo-implementation.md)

**Goal:** 回答「各个方案加速的原理是什么」——按**五类杠杆**对照，不是再堆公开秒数。

---

## 0. 结论先行：五类杠杆

墙钟下降只有五把刀。宣传倍数几乎都是**几把刀叠在一起、再换分母**。本仓金线已经用掉其中两把（官方 8-step CFG 蒸馏 + INT8/cuDNN 后融合），剩下能保 MAE∩有声的只有「同轨迹切算力墙」。

| # | 杠杆 | 改什么 | 不改什么 | 8-step INT8 金线上的命运 |
| --- | --- | --- | --- | --- |
| **1** | **语义 NFE / 采样轨迹** | DiT 前向次数、sigma 格点、是否换学生权 | 单步核、卡数 | **已经 8 步。** 再砍到 4 **必须换 LoRA**；未蒸馏 `--steps 4` 红线崩。数量级倍数的主因几乎都在这里。 |
| **2** | **每步 FLOP** | 权量化、注意力稀疏/量化、核融合 | 请求步数 | 金线已花预算在 **INT8 + dense cuDNN**。再换 Sage/Sol/VSA = 在 INT8 上叠近似 → 本仓 MAE 已 FAIL。 |
| **3** | **被执行的层/步（cache skip）** | 步间/块级 residual 相似则复用，少算 DiT | 请求 `num_inference_steps` | 50-step 中段有「无聊细化」可跳。CFG 蒸馏 8-step **每步都是大更新**。本仓 Tea/Mag skip 1 不过门；FB 0.04 **零 skip**。 |
| **4** | **单卡算力墙（Ulysses/TP）** | 把 packed 序列/头/权摊到多卡 | 轨迹、核、步数 | **唯一合法的同轨迹刀。** 只砍 DiT；TE+VAE 仍是 e2e 地板。本仓 **尚未实现**。 |
| **5** | **解码 / 分辨率** | TAE/TinyVAE、低分草稿、二采换模型 | DiT 配方本身 | 成片观感来自第二套模型或轻量 VAE。金线禁止 TAE 成片。 |

**一句话：** 12.2× / 22.2× / 4.44× **都不是「无损加速本仓 66s」**。12.2× 的 95.5% 是杠杆 1；4.44× 的主力是杠杆 3（相对 50-step）；22.2× 是杠杆 1+5 换栈。对本仓：保效果只做 4；要数量级走独立快档（1 或 5），不过门不开默认。

---

## 1. 总表（方案 × 杠杆 × 质量）

符号：**—** = 不用该杠杆；**✓** = 主刀；**◐** = 次刀 / 已落地但默认关。  
「对金线」：**默认** / **opt-in 禁默认** / **可叠加（未实现）** / **快档 SKU 互斥** / **禁止接默认**。

| 方案 | 1 NFE/轨迹 | 2 每步 FLOP | 3 cache skip | 4 并行 | 5 解码/分辨率 | 质量合同 | 对金线 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **本仓金线** | 已是 **8**（CFG 蒸馏，cfg=1） | **INT8 ConvRot + fused cuDNN** | — | — 单卡 | 官方 fp16 VAE tile | **金线自身**（PPM MAE∩有声） | **默认** |
| **sage3** | — 仍 8 | ✓ QKV **FP4 量化 attn** | — | — | — | **非无损**；MAE~27 FAIL | **opt-in 禁默认**；与 cudnn/sol **互斥** |
| **sol_cute** | — 仍 8 | ✓ **块稀疏 Softmax**（Sol-Attn） | — | — | — | **非无损**；MAE~35 FAIL | **opt-in 禁默认**；与 cudnn/sage **互斥** |
| **Tea 0.04** | — 请求仍 8 | — | ✓ 整步 skip（本仓 **7/1**） | — | — | 未对本轮成片重测 MAE；skip=外推 | **opt-in 禁默认**；与 Mag/FB **互斥** |
| **Mag 0.08** | — 请求仍 8 | — | ✓ 整步 skip（**7/1**；历史 MAE 52–68） | — | — | **非无损**；历史 FAIL | **opt-in 禁默认**；永不开默认 |
| **FB 0.04** | — 请求仍 8 | — | ◐ 首块探测；本档 **8/0 零 skip** | — | — | 无加速，无过门 | **opt-in 禁默认** |
| **FastH3** | ✓ DMD2 **4-call**（换学生 LoRA） | ✓ **训练 VSA**（video 90% 稀疏；text/audio dense） | —（不靠 Tea） | SP 可选；宣传数字是 **1×B200** | 官方 H3 VAE | **明确非无损**；T2VA Preview | **快档 SKU 互斥**；禁 merge INT8 |
| **Sol Super Accel** | ✓ H3 **4** + LTX **3**（换模型） | ◐ Stage2 Sol `cute_sm100` | **关**（少步上不是他们的刀） | **零 TP/SP**；2 GPU 流水 | ✓ **低分草稿 + TAEH3/TAEHV** | 页面 **“not lossless”**；仓内不声称倍数 | **快档 SKU 互斥**；禁写 66s |
| **Sol-Engine** | — **仍 50-step BF16** | ✓ 核融合 + Sol-Attn | ✓ **主力**（Tea 或 FBC） | — 单卡 / layerwise | 官方 VAE | 相对 eager 50-step 的 near-lossless **pretence**；无本仓 MAE | **禁当 8-step 倍数**；融合对本仓已 DROP |
| **VDN hybrid** | — 50-NFE 档不减步 | ✓ 窗 softmax + Delta 线性 | 非主刀 | ✓ Ulysses；B200 另 5+3 分支 | 官方 VAE；计时排除 VAE | 口头 near-lossless vs **dense MiniMax BF16** | 研究 attn；未过 MAE 禁默认；5090 装不下 |
| **VDN 8-NFE turbo** | ✓ DMD2 **50→8**（VDN 流形内） | 继承 hybrid | — | 同上 | 同上 | **明确折中**；故意不做 4 | 同 NFE 换权+attn；**快档另册** |
| **RH Lightning** | ✓ 后训练 **50→4**（高动态回 8） | ◐ Sage2 量化 attn + compile | ◐ Cache-DiT（4-step 上空间极窄） | ✓ **TP2+Ulysses4** | 官方 VAE | **不宣称无损**；无 MAE 表 | **只抄 Ulysses 思想**；整栈禁默认 |
| **SGLang Cache-DiT / TeaCache** | — | — | ✓ 块级 / 整步（服务 50-step 主战场） | 与 serving 拓扑正交 | — | 框架 `high`=SSIM；**≠** 本仓 MAE | **一行：** 杠杆 3，金线上等同 Tea/Mag 族；禁默认 |
| **本仓 Ulysses（规划）** | — 同 8-step 轨迹 | — 仍 cuDNN | — | ✓ packed 序列 A2A；**TP=1** | TE/VAE 仍单卡 | **置换 pretence**：过 MAE 才算实现正确 | **可叠加金线**；默认 U=1；**尚未实现** |

---

## 2. 每方案一段原理

**本仓金线（INT8 8-step + fused cuDNN）。** 发布 H3 已是 CFG 蒸馏：每步只 forward 正分支，`cfg=1`。本仓再把 DiT 权打成 INT8 ConvRot、SDPA 走 fused cuDNN。加速已经发生在「相对 50-step eager BF16」的杠杆 1（官方采样合同）和杠杆 2（权+核）。TE / DiT / VAE 串行；5090 cafe 分解约 TE 12–14 + DiT 31.4 + VAE ~10.5。再抠融合相对 eager 的那一截，对本仓是 DROP-HERE。

**sage3 / sol_cute。** 只动杠杆 2：仍 8 次 dense 轨迹上的 SDPA，把 Softmax 换成 FP4 量化注意力或块稀疏近似。请求步数、层数、VAE 不变。本仓尺子上核可以快（sage3 DiT −17%；sol_cute 4090 DiT −15%），片不过门（MAE~27 / ~35），e2e 常被 TE 吃光。8-step 每步都是大更新，注意力误差没有后续步可抹平。`auto` 永不选；缺插件硬失败。

**Tea / Mag / FB。** 只动杠杆 3：请求仍 8 步，探测 residual / velocity / 首块差，决定少算几次 DiT。50-step 上这是 Sol-Engine 4.44× 的主力；蒸馏 8-step 把中段冗余提前榨干。实测：Tea 0.04 与 Mag 0.08 各 skip 1（5090 DiT ~27.6s）；Mag 历史 MAE 52–68 FAIL；FB 0.04 零 skip。三者互斥。LightX2V #579：4-step 蒸馏后再开 Tea 几乎不跳——同方向。

**FastH3。** 杠杆 1 换学生：DMD2 把 ~49 次 teacher forward 蒸成 4 次（sigma `1000,750,500,250→0`），权是 LoRA 不是 INT8 骨干。杠杆 2 的 VSA 跟学生一起训（video-video ~10% tile；text/audio 永远 dense；`--no-vsa` 硬拒）。不靠 cache。对照分母是 ~49-step Base BF16（1×B200 15s 768p 47.2s / 14.38×），不是 8-step INT8。明确非无损。

**Sol Super Acceleration。** 杠杆 1+5 换栈，不是「H3 四步成片」：GPU A 上 H3 4-step **低分草稿**（896×512）+ TAEH3 + 社区 4-step LoRA；GPU B 上 **另一套** LTX-2.5 做 3-step 精修 + TAEHV 成片；音频 mux 回草稿 PCM。零 TP/SP，两卡流水。Cache 两阶段都关。页面写明非无损。成片观感来自 LTX+TAE，不是 H3 四步已经够好。

**NVIDIA Sol-Engine（相对 50-step BF16）。** **不减 NFE。** 三柱：① 核融合（相对 eager BF16，官方 4090 ~1.15×）② **跨步 cache（主力）** ③ Sol-Attn。4090 受控梯子：lossless opt 1951s → Tea 613.71s（**3.18×**，14 full + 35 reuse / 49）→ 再 Sol **1.22×** → 504.33s；相对未优化 2239.22s = **4.44×**。5090 OnDevice 4.52×（1045→231s @768p）同一三柱、梯子顺序不同（Sol 先、cache 后）。对本仓：柱 1 已融合；柱 3 单独接 generate = MAE FAIL；柱 2 在 8-step 上已测不过门。

**VDN。** 两条独立的刀。**(A) Hybrid attn（杠杆 2，不减步）：** chunk 对齐的窗 softmax（精确局部）+ Video Delta 线性（远端联合求解）+ 边界锚；冻骨干加支路。**(B) 8-NFE turbo（杠杆 1）：** DMD2 把 VDN **自己的** 50-NFE 对齐到 8，故意不做 4。多卡是 Ulysses +（B200）5+3 分支并行。质量口头 near-lossless 的分母是 dense MiniMax BF16，不是本仓 MAE。计时排除 VAE。5090 32GB 装不下其 ~65GB 工作集。

**RH Lightning。** 三层显式相乘，**不要合成一个「多卡 12×」**：① 杠杆 1：后训练 50→4（同 8 卡 348.8s→43.0s = **8.1×，占节省 95.5%**）② 杠杆 2+3：Sage2 + Cache-DiT + compile（43.0→28.7s，再 ×1.50，**4.5%**）③ 杠杆 4：TP2+Ulysses4 切 DiT（**12.2× 表里卡数是常量**；同配方拓扑差另表 54.0→48.2s ≈12%）。无 NVLink 上少 TP、多 Ulysses，因为 AR 形状劣于 A2A。不宣称无损。对本仓：抄 ③ 的切分，不抄 ①② 进默认。

**SGLang Cache-DiT / TeaCache。** 纯杠杆 3。Cache-DiT 是块级 residual（Fn/Bn + warmup + continuous cap）；Tea 是整步。服务侧在 **50-step BF16** 上有大可跳空间（SGLang high ~1.40× @4×H200）。与 vLLM Tea **互斥**。8-step INT8 上把框架 high 直接塞进来 = 在已占用的 INT8 预算上再花 skip 预算。本仓已有同族 CLI，默认关。

**本仓规划 Ulysses。** 纯杠杆 4：`tp=1`，切 packed 序列行，attn 内 A2A 变头切，INT8 权每卡副本。合法 U∈{1,2,4,8}（56 头、对齐 64；**U=7 非法**）。N=1 零行为差。过门标准是相对单卡 cudnn 的 MAE∩有声——置换不是新核，MAE 落到 sage/sol 量级当实现 bug。Amdahl：只砍 DiT，2 卡也到不了 e2e 33s。尚未实现，默认不得变。

---

## 3. 哪些倍数来自哪根杠杆

禁止把下表任何一行除进 66s。

### 3.1 RH 12.2× = 几乎全是杠杆 1

同一 **8× RTX 6000D**、5s 1344×768 t2va（[tech-deepdive §1.4](./2026-09-11-h3c-accel-survey-tech-deepdive.md)）：

| 增量 | 秒 | 倍数 | 杠杆 | 占 348.8→28.7 节省 |
| --- | ---: | ---: | --- | ---: |
| 50-step BF16 基线 | 348.8 | 1.0× | — | — |
| + 蒸馏权 **50→4** | 43.0 | **8.1×** | **1 NFE** | **95.5%** |
| + Sage2 + Cache-DiT + compile | 28.7 | 再 ×1.50 → **12.2×** | **2+3** | **4.5%** |
| 卡数 | 两行都是 8 卡 | **1.0×** | 4 **未进入此表** | **0%** |

另表（同 4-step、同 8 卡）：TP4+U2 **54.0s** → TP2+U4 **48.2s**（约 **12%**）才是杠杆 4。禁止与 12.2× 相乘。理论 NFE 50/4=12.5，实测蒸馏臂只有 8.1×：TE/VAE/通信不随步数按比例消失（Amdahl）。

### 3.2 Sol-Engine 4.44× = 杠杆 3 主力，不是 Sol 单核

官方 4090 **50-step BF16 768p**（[4090-accel-plan §4.1](./2026-08-25-h3c-4090-accel-plan.md)）：

| 臂 | E2E | 相对上一档 | 杠杆 |
| --- | ---: | ---: | --- |
| 未优化 Diffusers/SGLang BF16 | 2239.22 s | — | 分母 |
| lossless opt（融合 + compile，关 Tea 关 Sol） | 1951.27 s | ~**1.15×** | **2** 核融合（相对 eager） |
| + TeaCache（14 full + 35 reuse / 49） | 613.71 s | **3.18×** | **3 cache** ← 4× 的主力 |
| + Sol-Attn | 504.33 s | 再 **1.22×** | **2** 稀疏 attn |
| **整栈** | | **4.44×** vs 2239s | **1 NFE = 未减（仍 50）**；**4/5 = 未用** |

5090 OnDevice **4.52×**（1045.4→231.2s @768p）同一三柱；公开页梯子是 Sol 先、**FBC** 后，cache 仍是最大单步。Spark 3.92× 同结构、分辨率不同，勿与 4090 横比秒数。

### 3.3 Super Accel 22.2× = 杠杆 1+5 换栈，仓内不声称倍数

博客 **6.85s / 22.2×** 的分母是已发表 **SGLang 152.3s @5s 768p GB200**，不是 50-step 2239s，也不是本仓 66s。正式 job 中位 **6.761s**（Stage1 **4.292s** + Stage2 **2.447s**），JSON **`speedup_claim: null`**。

| 段 | 中位 | 杠杆 |
| --- | ---: | --- |
| Stage1 H3 4-step @896×512 + TAEH3 | 4.292 s | **1** 四步 LoRA + **5** 低分/轻量 VAE |
| Stage2 LTX 3-step + TAEHV @1344×768 | 2.447 s | **1** 换模型三步 + **2** Sol sm100 + **5** TAE 成片 |
| 并行形态 | 2 GPU 流水 | **不是** 杠杆 4 的 Ulysses/TP |
| cache | 关 | 杠杆 3 = 0 |

没有官方 ablation 能把 22.2× 再拆成「几倍来自少步、几倍来自 TAE」。能钉死的只有：这是**另一条产品配方**，原话非无损。

### 3.4 FastH3 14.38×（附：同样主要是杠杆 1）

1×B200、15s 768p：Base ~49 DiT **678.7s** → FastH3 4-call+VSA **47.2s** = **14.38×**（[survey 秒数表](./2026-09-11-h3c-accel-survey-multigpu-5090.md)）。理论 NFE 49/4≈12.25；余量来自 VSA（杠杆 2）与 compile 等，**公开材料无像 RH 那样的分账表**，此处不编。分母不是 8-step INT8。

---

## 4. 叠加规则（对金线）

| 组合 | 判决 |
| --- | --- |
| 金线 ⊗ Ulysses（规划） | **可叠加**：同轨迹、置换。默认 U=1。过 MAE 才留。 |
| 金线 ⊗ sage3 或 sol_cute | 机制正交于步数，**质量互斥于默认**。二者彼此也互斥（一个 SDPA 槽）。 |
| 金线 ⊗ Tea/Mag/FB | 机制正交，**质量禁默认**；三者互斥。 |
| sage/sol ⊗ Tea/Mag ⊗ 少步 | **禁止静默叠刀**。RH 是在 **4-step BF16** 上叠的，不能论证 INT8 8-step 也能叠。 |
| 金线 ⊗ 4-step LoRA / VSA / TAE / LTX | **互斥配方**。独立 `--preset`，help 写非金线。 |
| 金线 ⊗ 把 `--steps` 改 4 不换权 | **禁止**（未蒸馏 4 步）。 |
| 框架 SSIM 0.93 代替本仓 MAE | **禁止改口**。 |

---

## 5. 交叉链接

| 要深读 | 去哪 |
| --- | --- |
| 少步为何好看（CFG vs 二次蒸馏 vs draft+refine） | [few-step-quality](./2026-09-11-h3-few-step-quality.md) |
| RH 12.2× 拆账、Ulysses 通信、五方案计算图 | [accel-survey-tech-deepdive](./2026-09-11-h3c-accel-survey-tech-deepdive.md) |
| PCIe 上为何少 TP 多 Ulysses | [pcie-multigpu-lessons](./2026-09-11-h3c-pcie-multigpu-lessons.md) |
| 本仓 opt-in 已测 DiT/MAE | [out/verify_accel_matrix/REPORT.md](../../../out/verify_accel_matrix/REPORT.md) |
| 4.44× 官方梯子原文读法 | [4090-accel-plan §4](./2026-08-25-h3c-4090-accel-plan.md) |
| Tea/Mag/FB vs Cache-DiT 判据 | [vs-sglang-vllm-tech-deepdive §3](./2026-08-25-h3c-vs-sglang-vllm-tech-deepdive.md) |
| 金线可抄 / 快档 / 永不碰 | [next-opt-plan §3](./2026-09-11-h3c-next-opt-plan.md) |

未 git commit。默认路径未改。
