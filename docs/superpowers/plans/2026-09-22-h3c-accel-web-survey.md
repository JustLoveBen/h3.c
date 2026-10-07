# H3.c 加速方案联网复核（2026-09-22）

> 研究备忘，**不改 C/CUDA 生产代码**，不 commit。检索日：2026-09-22。  
> 必读对照：[社区极速方向 09-16](./2026-09-16-h3c-community-speed-direction.md) · [方案选型 09-16](./2026-09-16-h3c-scheme-selection.md) · [分模块枢纽 09-15](./2026-09-15-h3c-module-opt-plan.md)。  
> **kitchen = [comfy-org/comfy-kitchen](https://github.com/comfy-org/comfy-kitchen)**，是 Comfy 社区加速核库，不是 MiniMax 官方源码。

**尺子（分列，禁止混报）**

| 列 | 480P 档 | 方向档 768P |
| --- | --- | --- |
| 设备 | 1× RTX 5090 热跑（模型已在显存） | 同左 |
| 成片 | **5s · 864×480 · 124 帧 · 有声 MP4** | **5s · 1376×768 · 124 帧 · 有声 MP4** |
| 用户关心的墙钟 | **~30s** | **~27s** |
| 质量地板 | Comfy 官方 H3 VisualVAE 片，无 16px 格子 | 用户认过的 Comfy 2-stage 官方 VAE 片，无格子 |
| **不当尺子** | LTX 成片、TAE、本仓金线 MAE、480p cafe 冒充 768P | 同左；1344 只作对照臂 |

公开数字几乎都不是这两列。下面凡缺「画布 / 步数 / 冷热 / 是否含 load·VAE·mux / 硬件」任一列的，**禁止写成打到 30s 或 27s**。编造数字比漏写更糟：搜不到就写搜不到。

---

## 0. 相对 09-16 文档要纠正的叙事

09-16 社区方向文**废金线默认、把 27s 当社区地板、kitchen 核迁 C + 加载常驻当主线**——这三句仍成立。下面几句**过时或过硬**，本次联网改口径。

| 09-16 / 09-15 旧句 | 2026-09-22 一手 | 改怎么说 |
| --- | --- | --- |
| 「Comfy kitchen 同画布 decode **~4s**」当已测墙钟 | kitchen PR [#167](https://github.com/comfy-org/comfy-kitchen/pull/167) / Comfy [#16187](https://github.com/Comfy-Org/ComfyUI/pull/16187)（2026-09-13/15 合入，随 ComfyUI **0.36.0**）：**5090 · 1344×768 · 362 帧（~15s）** int8 decode **18.7→12.9s**。Kijai 讨论区另有 **5s@1344×768** int8 decode **10.0s**（load 另列 3.1s）[HF#21](https://huggingface.co/Kijai/MiniMax-H3-experimental/discussions/21)。Chaosyn **0.4MP 5s** VAE **4.2s**；**0.7MP/1.0MP** 仍 **11.6–11.8s** | **~4s 不是 5s 768P 的已测厨房墙钟。** 12.9×124/362≈4.4s 只是按帧线性外推。5s 768P 公开 decode 仍是 **10s 档**（可能早于 PR 167）。用户 Comfy 2-stage 里 VAE 份额以用户图为准，不要再把「4s」当 PR 头牌 |
| 「18s 是碎核不是下限」 | 仍对。12.9s（15s 片）/ 10.0s（5s 片）证明同合同还能再快 | 继续打厨房核；验收改成 **对齐 Comfy 0.36 int8 路径**，不要写「必须 4s 才算厨房」 |
| 选型文「金线仍是默认栈」「画布锁 1344」「1376 只作备注」 | 用户方向档 JSON 成片 **1376×768**；09-16 社区文已否决 | 以社区方向文为准。选型文 §0 金线默认 **作废** |
| FastH3 只当 **4-step Preview** | 2026-09 FastVideo 发 **8-Step V2**，Comfy-Org 已 repack：[FastVideo-FastH3-8-Step-V2](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2) · [Comfy-Org/FastVideo-FastH3](https://huggingface.co/Comfy-Org/FastVideo-FastH3) | 漏网。V2 是 **8 NFE + 80% VSA**，FL2VA/Ref2VA **未蒸**。5090 消费卡 V2 全含秒数 **公开搜不到** |
| 少步只提 LightX2V / turbo4 / TaoMate | Alibaba PAI **官方 PDD Acc 8-step** [MiniMax-H3-Acc-LoRAs](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs)（2026-08-26）。普通 LoRA loader **丢 head bank** | 漏网。研究档；**没有** 1×5090 5s 全含表 |
| kitchen 只当 VAE 核 | kitchen README 已列 **`sol_attn` / `int8_attention` / `rms_rope_split_half` / `scaled_mm_nvfp4`**（cuda） | kitchen 同时是 **VAE 融合核 + DiT attn/量化后端**。Comfy `--use-sage-attention` **不会**自动选 sage3 |
| NVFP4 DiT「约 1.14×/step」可当加速柱 | smeltcore 5090：社区 pruned NVFP4 **175s vs INT8 185s（+5.4%）**，作者写 *Weight-only quantization buys power budget, not arithmetic*。[smeltcore](https://smeltcore.com/recipes/minimax-h3-on-rtx-5090-32-gb-video-audio-and-what-nvfp4-does-not-buy-you/) Chaosyn：测试 NVFP4 8-step **89.9s vs INT8 85.6s（更慢，因 offload）** | 5090 上 NVFP4 **几乎不买墙钟**；买的是盘/显存。禁止写成 27s 主刀 |
| Sage3 是 768P 注意力银弹 | 官方 Sage3 README **点名 CogVideoX / Hunyuan / Mochi**，**未点名 MiniMax H3**。[sageattention3_blackwell](https://github.com/thu-ml/SageAttention/blob/main/sageattention3_blackwell/README.md) Comfy [#15263](https://github.com/Comfy-Org/ComfyUI/issues/15263)：H3 不传 `low_precision_attention=False` 时 **纯噪声**。muninn 15s 768P 14-step 栈上 Sage **e2e 仅 1.02×** | sage3 假 hit 硬退纪律保留。5090 长序列上 **sage2 也常常只剩几个百分点**；真正吃 S² 的是 2-stage 把高分 NFE 砍掉 |

**09-16 仍正确、本次确认：**

- 金线 MAE 不是质量门；用户认 Comfy 官方 VAE 无格子片。
- 18s C 碎核不是合同下限；几何红线是 256/64 tile-local RoPE。
- Sol-H3-Spark **56s** 与 muninn 5090 热 **30.6s** 是 **H3 草稿 + LTX 精修**，不成片质量尺。
- VDN **不是 VAE**；Flash-VAED / KJ Triton VAE **对不上 H3 ViT**。
- 8 卡 Ulysses 冻结；`peer=0` 再切请求只会更慢。
- 5s/10s、1344/1376、冷/热、是否含 load **禁止同一列**。

---

## 1. 总表（一层一眼）

栏位：**层** = TE / DiT attn / DiT 步数 / VAE / 加载 / 成片模型。**VAE 合同** = 官方 H3-VisualVAE（36 层 ViT3D，tile 256/64）还是换解码器。  
**480P / 768P** 分开写对本仓两档的搬法。优先级只对 **1×5090 热跑 + 官方 VAE 质量地板**。

| # | 方案族 | 改哪一层 | 官方 VAE？ | 公开数字（硬件 / 画布 / 步 / 冷热 / 含什么） | 480P ~30s | 768P ~27s | 风险 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | **comfy-kitchen** | VAE 核 + DiT attn/量化后端 | **是**（换核不换权） | 见 §2.1。头牌是 **15s 768P decode 12.9s**，不是 5s 4s | **能搬核**；480P VAE 已不是主墙（Chaosyn 0.4MP **4.2s**） | **P0 能搬**：本仓 196 窗 ~18s vs 社区 int8 路径。目标对齐 0.36，不要死钉 4s | Python 税；kitchen CUDA 要 cu130；INT8 权才走 `int8_attention`；假 kitchen fallback |
| 2 | **Sage / sage3 / sm_120** | DiT attn | 是 | 见 §2.2。e2e 从 **−31%（480p 竖）到 +2%（15s 14-step 768P）** 不等 | **P1 能搬 sage2**（采样段 −21–26%，Chaosyn 0.7MP） | **谨慎**。768P seq~37k：本仓 sage3 hits=0；社区 5090 长序列 e2e 几乎吃不满。缺核硬退 | 假 hit；H3+sage 噪声 [#15263](https://github.com/Comfy-Org/ComfyUI/issues/15263)；池子挤 INT8 `window=1` |
| 3 | **Sol-H3 / Spark / Sol-Attn / CuTe** | 配方：少步+异模型；核：稀疏 attn | Spark/30.6s 路径 **否**（LTX VAE）。On-Device / Sol-Attn **是** | Spark 官方 5s 1344 **热全含 56.17s**（[Sol-H3-Spark](https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3-Spark/)）。muninn 同图 5090 热 **30.6s**（[对照表](https://ai-muninn.com/en/blog/h3-ltx-two-stage-dgx-spark)）。On-Device 5090 **50-step 768P 231.2s** 近无损 4.52×（[H3-OnDevice](https://nvlabs.github.io/Sana/Sol-Engine/H3-OnDevice/)） | 不搬 LTX。Sol-Attn 对短序列收益小 | **核 P1 能搬**（sm_120 CuTe / kitchen `sol_attn`）。**配方不能当默认质量尺**。muninn 同卡 15s 栈 Sol-Attn **e2e 1.03×** | 换质量尺；32GB 双栈每轮 reload Stage1（muninn：Stage1 **33.5GiB** 5090 装不下）；`dense_fallback` 静默 |
| 4 | **TaoMate / SelfLift / LBH 2-stage** | DiT 步数 + 日程（低分→lift→高分） | **是**（成片仍官方 VAE） | TaoMate 官方：**8×H20、480×864、10s、TP2×U4、Pure DiT 14.810s，不含 load/TE/VAE/mux**（[HF](https://huggingface.co/TaoLiveAIGC/TaoMate-H3)）。用户 5090 2-stage **5s 1376 ~27s** 是私有 JSON，检索无公开同名 gist | **P0 配方**：少步 3 NFE 在 864×480 上 Amdahl 最大。开源 runtime **不能**（只要 Hopper 4/8 卡） | **P0 配方已是方向档**：864×480×3 → LBH → 1376×2。本仓热对照仍 **52s** 级，差在 VAE+高分 attn/权流 | 开源 FA3/Hopper；Comfy 只吃到 3-step LoRA 不是 chunk+KV；LBH ≠ 论文 SelfLift-rich；糊/过饱和是少高分步上限 |
| 5 | **VDN / OpenVDN** | DiT hybrid attn + 8-NFE LoRA | **是**（VAE 原样留下） | **8×B200、14.4s 768p、denoise 6.9s / e2e ~9.0s，排除 load/warmup/VAE/mux**（[README](https://github.com/OpenVDN/vdn-minimax-h3) 2026-09-17）。单卡 B200 8-NFE denoise **51s**。Comfy 5090 ~**17s/it** @ 1280×736/145f/8-step（[ComfyUI-VDN-H3](https://github.com/Saganaki22/ComfyUI-VDN-H3)） | 不成片主路径。32GB 要 block offload | **研究档，不能搬成 27s 主刀**。FA4 不服务 sm_120 | 头牌不含 VAE；改 attn 合同；与 kitchen 争显存 |
| 6 | **TAE / TAESD / taeh3** | 换解码器 | **否** | 作者定位 preview。Kijai「not the greatest」[HF](https://huggingface.co/Kijai/MiniMax-H3-TAE)。holdout PSNR ≈ **25.5 dB**（[PreviewOverride](https://github.com/simsim9-stack/ComfyUI-MiniMaxH3-PreviewOverride)）。decode「milliseconds」是预览瓦片不是成片 | **禁止当 30s 质量尺** | **禁止当 27s 质量尺**。预览档可留 | 闪/抹/色漂；静默 TAE 装快 |
| 7 | **LTX / 异模型精修** | 成片模型 | **否** | muninn 5090：H3 4-step 384p 草稿 + LTX 3-step，5s 1344 冷 40.3 / **热 30.6s 全含**。Spark 官方 56.17s。decode 是 **LTX 卷积 VAE 12.5%** | 不做产品默认 | **速度参照，不是质量尺** | 5090 每轮 reload；Comfy&lt;0.34 精修静默垃圾帧（muninn） |
| 8 | **Flash-VAED / KJ Triton / PatchTritonVAE / H3VAE_TRT** | VAE（conv 系）或 TRT 另编 | Flash/Triton：**对不上 H3 ViT**。TRT：试图编官方 VAE | PatchTritonVAE 只 Wan/KL/LTX conv（[triton_vae.py](https://github.com/kijai/ComfyUI-KJNodes/blob/main/nodes/triton_vae.py)）。Flash-VAED 剪 CausalConv3D（[arXiv:2602.19161](https://arxiv.org/html/2602.19161v1)）。社区 ONNX/TRT 空 latent 5s 1344：FP16 **17.87→11.84s**（二手转载 [Seedance](https://www.seedance.tv/blog/minimax-h3-vae-trt-speedup)）。muninn 5090 15s 栈 TRT-VAE **e2e 1.02×** | 不搬 Triton/Flash | TRT **P2 可选对照**，不是厨房替代。1.7× 宣传 vs 实测 1–2% e2e | 换数值；编译冷启动；空 latent ≠ 人脸运动片 |
| 9 | **NVFP4 / INT8 / 蒸馏 LoRA（turbo4 / LightX2V / PDD）** | 量化 + DiT 步数 | 是（VAE 独立） | INT8 ConvRot 是 Comfy 默认骨干。NVFP4 5090 **~1.05× 或更慢**（上表）。Chaosyn 0.7MP 5s：**4-step dual-clock 46s / 8-step 58s / 20-step 131s**（含 VAE+存盘，[Chaosyn](https://blog.chaosyn.com/en/posts/minimax-h3-rtx-5090-speed-cost/)）。LightX2V 4-step v1.2 偏音频修复，无 5090 全含表。PDD 官方无 5090 表 | **P0 少步**：20→4/8 才从 76–131s 看见 30s。4-step 音频曾坏，需 dual-clock | **P0 少步已在 2-stage 里**（3+2）。单段 8-step 768P 公开仍是 **85s**（ChrisJohnson 10-step 1344）到 **142s**（Chaosyn 1.0MP 8-step） | 蒸馏不叠蒸馏；PDD 丢 head = 假加速；turbo4 画质/声失败（zenn） |
| 10 | **TeaCache / te-cache / 步间 cache** | DiT 步间跳块 | 是 | Icyoung：CMP 170HX、20-step、1024×576、5.2s：**306→102s（3.0×）**，**不含**声明是否 VAE？表格含 Total wall-clock（[README](https://github.com/Icyoung/ComfyUI-MiniMaxH3-TeaCache)）。RTX 3060 20-step 量级 **1.94×**（[minimaxh3tutorial](https://www.minimaxh3tutorial.com/nodes/cache)）。节点文：8-step + warmup/cooldown **几乎没东西可跳** | 20-step 预览可试。冲 30s 的配方已是 3–8 NFE → **几乎 0** | **不做产品默认**。蒸馏档叠 cache 是已知坑 | 改轨迹；8-step 无效；TE-Speed 节点 3060 上 **0.999× 且改输出** |
| 11 | **多卡 Ulysses / CFG parallel** | 通信 | 是 | RH Lightning：**8×RTX 6000D、5s 1344×768、warm 生成 28.7s，不含 load/compile**（[选型已引](./2026-09-16-h3c-scheme-selection.md)，README 仍此口径）。VDN 8×B200 见上。本仓 8×5090 `peer=0` U=8 更慢 | 对照冻结 | **方向档冻结 8 卡**。27s 是单卡配方问题 | 消费卡无 GPU P2P；把 12.2× 除进单卡 |
| 12 | **2026-09 漏网** | 见 §3 | — | FastH3 8-step V2；PDD Acc；Comfy 0.36 kitchen VAE PR；kitchen 原生 `sol_attn`；Comfy core BlockSparseAttention；KingGore SM120 Sol-Attn 插件 | 见 §3 | 见 §3 | 5090 秒数大多未发表 |

---

## 2. 分方案（一手、机制、能不能搬）

### 2.1 comfy-kitchen（VAE + attn 后端）

**是什么。** PyPI `comfy-kitchen`，GitHub [comfy-org/comfy-kitchen](https://github.com/comfy-org/comfy-kitchen)。多后端（eager / cuda / triton / hip）核库：量化、RoPE、AdaLN、INT8 GEMM、**`int8_attention`、`sol_attn`**。ComfyUI 0.36.0 把 MiniMax H3 **视频 VAE** 接到这些核上（kijai，PR 16187 + kitchen PR 167，2026-09-15 合入）。**不是** MiniMax-AI 官方仓。

**改哪一层。**  
- VAE：encoder 侧 `group_norm_silu_pad3d` + 可选 `--fast` `fp16_conv3d`；decoder 36 层 ViT：`linear_input_act`（rms/swiglu + residual epilogue）、`int8_attention`、tile **最多 4 个一批**。  
- DiT：同一库的 `int8_linear` / `sol_attn` / `scaled_mm_nvfp4` 给骨干用。Comfy 里 kitchen attn 与 Sage 是不同选择器。

**VAE 合同。** 官方 VisualVAE 权（现推荐 `minimax_h3_video_vae_int8_convrot.safetensors` 2.81 GB，encoder 改存 fp16；Kijai：与旧 3.17 GB 实验文件 **decode 相同**）。几何仍内部 tile 256/64。**不换 TAE**。

**公开数字。**

| 来源 | 硬件 | 形状 | 冷热 | 含什么 | 秒 |
| --- | --- | --- | --- | ---: | ---: |
| kitchen PR 167 原文 | 5090 | 1344×768 **362 帧** | 未写冷热；`comfy.sd.VAE` 调用 | **只 VAE encode/decode** | decode fp16 34.6→31.1；`--fast` 27.7→23.2；**int8 18.7→12.9** |
| 同 PR 质量 | 同 | 同 | — | vs exact path | int8 decode **67.7 dB**（VAE 自身重建地板 ~38 dB） |
| Kijai HF discussion 21 | 5090，torch 2.12+cu130，Win11 | **5s @1344×768** | 脚本 decode，load 另列 | decode / load | fp16 decode **19.5s**；int8 **10.0s**；load 4.0 / 3.1s |
| Chaosyn 2026-08-21 | 5090 460W | 5s **0.4 / 0.7 / 1.0 MP**，8-step I2V | 热（resident 另测） | VAE decode 段 | **4.2 / 11.6 / 11.8s** |
| Comfy PR 15446（旧） | 未钉 5090 | 175×1344×768 | — | chunked I/O | 14.6→13.9s（显存为主）——**不要**当厨房 4s |

**480P ~30s：** 能搬核，但 Amdahl 小。0.4MP VAE 已经 4.2s；30s 预算里主墙是 **NFE 与加载**。  
**768P ~27s：** **P0。** 本仓 5s 1376 热 VAE **17.6–18.2s**（09-16 引用 `out/verify_taomate/REPORT.md`）。社区 5s 768P decode 公开是 **10s 档**；15s 片厨房后 **12.9s**。把 18s 写成下限仍然错；把目标写成「必须 4s」则 **缺乏 5s 768P 一手**。验收：同 latent、无 waffle、热 `vae_blocks+load` 对齐 **Comfy 0.36 int8 + kitchen 0.2.34+**。

**风险。** 嵌 Python 进 C 生产 = 税。kitchen CUDA 约束失败会回 eager（PR 评论：fp16 linear 无核时的 fallback）。Windows pinned memory + kitchen CUDA 曾在 5070 Ti 上 AV（[Comfy#15337](https://github.com/Comfy-Org/ComfyUI/issues/15337)）。INT8 慢于 FP16 的报告几乎都是 **backend 没亮**（同一 HF 帖：Linux 后「INT8 变得飞快」）。

---

### 2.2 SageAttention / sage3 / SageAttention3 sm_120

**改哪一层。** DiT（及可选图像模型）注意力核。VAE D=64，Sage3 合同对不上。

**官方 VAE？** 是。

**公开数字。**

| 来源 | 硬件 | 画布 / 步 | 含什么 | 秒 / 倍数 |
| --- | --- | --- | --- | --- |
| zenn toki_mwc | 5090 Win11 Comfy 0.30 | **竖 480×832 · 124f · 20-step** | 文称推理时间 | Sage **83.93→57.53s（−31.48%）**；过 ASR。**480p，禁止当 768P** |
| Chaosyn | 5090 Ubuntu | ~0.7MP 5s INT8 | 采样段 / 全含 | 4-step 采样 36.7→28.9（−21%），全含 60.6→53.5；8-step 采样 −26%，全含 85.6→67.0。VAE **11–14s 不受益** |
| muninn config-stack | 5090 Win Comfy 0.34 | **1344×768 · 362f · 14-step** NVFP4 | 全图 e2e | Sage **52.18→51.11s = 1.02×**（核 millibench 仍 4.1×——**e2e 被别的段淹没**） |
| muninn VSR 文 | 5090 | **15s 1080p · 14-step** | 全图 | Sage 2.2.0 **507.0→411.8s（−18.8%）** ——1080p 15s，不是 5s 768P |
| ChrisJohnson | 5090 Sage 2.2.0 sm_120 | 1344×768 5s **10-step** | **全节点含 TE/VAE/音频/MP4** | 热 **85.23s**（有 Sage，不是「开了 Sage 就 27s」） |
| Sage3 README | 5090 核图 | 未给 H3 e2e | 仅 kernel | [thu-ml/SageAttention sageattention3_blackwell](https://github.com/thu-ml/SageAttention/blob/main/sageattention3_blackwell/README.md)。支持表 **无 MiniMax H3** |
| Comfy #15263 | PRO 6000 Blackwell | 1920×1088 362f 20-step | 开 `--use-sage-attention` | **视频+音频纯噪声**；缺 `low_precision_attention=False` |

**机制要点。** Sage3 = MXFP4 attention，源码编译，setup.py 只认 **sm_100 / sm_120**，CUDA ≥12.8。[setup.py](https://github.com/thu-ml/SageAttention/blob/main/sageattention3_blackwell/setup.py)  
Comfy `--use-sage-attention` 找包名 **`sageattention`**。`sageattn3` **不会被自动选中**（muninn VSR 文写明；Reddit 安装帖要手补 `attention3_sage`）。这是本仓「日志写 sage3、实际 dense」的社区同源问题。

**480P ~30s：** sage2 **能搬**，对 20-step 采样段有两成；对已经 4/8-step 的配方，VAE+mux 会吃掉收益。不要指望 sage3 把 76s 砍到 30s。  
**768P ~27s：** **不能当 P0。** 本仓 768P `seq≈37788` sage3 hits=0（09-16）。社区在「已经 chunk + cu130 + NVFP4」的 15s 栈上 e2e 只剩 2%。优先级：先 2-stage 把高分序列变短，再要 **真能进核** 的 kitchen/sol，失败硬退。

**风险。** 假 hit；噪声；workspace 挤掉 INT8 常驻（本仓 window=1）；Windows TMA 对齐 bug（[Sage#357](https://github.com/thu-ml/SageAttention/issues/357)）。

---

### 2.3 Sol-H3 / Sol-H3-Spark / Sol-Attn / CuTe

**名字不要混**（09-16 §2.1 仍对，本次复核链接仍活）：

| 名字 | 一手 | 实际 |
| --- | --- | --- |
| Sol-Engine | [Sana/Sol-Engine](https://nvlabs.github.io/Sana/Sol-Engine/) | 推理框架（cache / 稀疏 / 量化 / 融合） |
| Sol-Attn | arXiv:2607.24027；kitchen 亦有 `sol_attn` | 在线 softmax 里做 block 选择。官方 backend 表含 **5090 SM120 = CuTe** |
| H3 On-Device | [H3-OnDevice](https://nvlabs.github.io/Sana/Sol-Engine/H3-OnDevice/) | **50-step、无 LoRA**。Spark 测 480p，5090 测 **768p 231.2s / 4.52× 近无损** |
| Sol-H3-Spark | [Sol-H3-Spark](https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3-Spark/) | **用户口中的 56s**：5s 1344、H3 4-step 384p 草稿 + **LTX-2.5 3-step**，热全含（含 Qwen 与 MP4） |
| Sol-H3（B300） | [Sol-H3](https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3/) | 8×B300 4-step serving 5s **1.653s** ——另一台机器 |

**Spark 56.17s 分解（官方页，本次再抓）：** H3 4-step 35.2% · LTX 3-step 44.0% · **LTX VAE 12.5%** · upscaler+adapter 3.8% · Qwen 2.4% · mux 1.3% · overhead 0.9%。**成片 decode 不是 H3 ViT。**

**muninn 独立复现（同一 5s 1344 图，全含 submission→文件）：**

| | DGX Spark | RTX 5090 |
| --- | ---: | ---: |
| Native H3 4-step 冷 / 热 | 237.7 / 230.1 | **51.5 / 41.5** |
| 两阶段 H3+LTX 冷 / 热 | 118.8 / 73.3 | **40.3 / 30.6** |

Spark 热无 Sol-Attn 73.3；装 Sol-Attn + 解时钟帽 **66.1s**，仍慢于官方 56.17 约 1.18×。[ai-muninn Spark](https://ai-muninn.com/en/blog/h3-ltx-two-stage-dgx-spark)

**5090 装不下 Stage1 33.5GiB，每轮 reload** ——所以 30.6s 已经含换权税。Spark 靠 128GB UMA 常驻才把加载从口径里拿掉。

**Sol-Attn 插件生态（相对 09-16 新）：**

- [kijai/ComfyUI-SolAttn_triton](https://github.com/kijai/ComfyUI-SolAttn_triton)（muninn 钉 commit `26d816e`；未装则 `dense_fallback_sol_attn_not_loaded` **不报错**）。
- [KingGore/ComfyUI_sol-attn_Blackwell](https://github.com/KingGore/ComfyUI_sol-attn_Blackwell)：SM120 `flex_attention`，**核 millibench** seq 32768 相对 SDPA ~9×；**没有** H3 5s 全含表。失败自动回 SDPA。
- kitchen README 已有 **`sol_attn` cuda 后端** ——本仓旧文没把「厨房核」和 Sol-Attn 写成同一库。

muninn 15s 768P 生产栈：**Sol-Attn e2e 1.03×**（改数值）。[config-stack](https://ai-muninn.com/en/blog/minimax-h3-rtx5090-config-stack)

**480P：** 不搬 LTX。短序列 Sol-Attn 收益更小。  
**768P：** **核 P1 能搬**（真 sm_120，shape 不支持硬退）。**配方（LTX 成片）不能当 27s 质量尺。** 30.6s 只证明「热全含量级社区已经有人碰到过」——那条片不是官方 VAE。

---

### 2.4 TaoMate / SelfLift / 2-stage LBH

**改哪一层。** DiT NFE 与分辨率日程。成片仍走官方 VAE。LBH 是 **latent 空间升采样**，跳过 decode→像素放大→encode。

**TaoMate 一手。** [TaoLiveAIGC/TaoMate-H3](https://huggingface.co/TaoLiveAIGC/TaoMate-H3) · 项目页 [taoliveaigc.github.io/TaoMate](https://taoliveaigc.github.io/TaoMate/)  
- 3-step LoRA r=128；chunk 生成 + clean-KV。  
- **验证硬件：8×H20 96GB，Hopper/SM90，CUDA 12.8，4 或 8 GPU，TP2×Ulysses4。**  
- 表：**480×864 · 10s · seed 8301 · Pure DiT 14.810s vs MiniMax 169.572s（11.45×）**；first playable **17.287s**。显式 **DiT FPS 不含 loading / VAE**。  
- 分辨率合同含 **864×480 与 1376×768**（与用户 JSON 成片同桶）。  
- Comfy 生态只有 **LoRA 转存**（[CZMartin22](https://huggingface.co/CZMartin22/TaoMate-H3-3step-ComfyUI)、[Asirus](https://huggingface.co/Asirus/TaoMate_H3_3_Step_LoRA)）：3 step、CFG=1、Euler。Asirus 表（**未写 GPU**）INT8+Turbo4 **474s** vs TaoMate **373s** ——不能当 5090。

**SelfLift / LBH 一手。** 社区落地是 [LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) + [HF 权](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)，**不是**论文 SelfLift-rich 全套。本仓 `h3_latent_upscaler.*` 已接 3D resizer。README：跳过 ~5B VAE 往返；1–4×；3D 可 target 像素。

**用户 2-stage（质量+速度地板，私有）：** `input/minimax-h3-r2v-taomate-2stage.json` — 864×480 ×3 NFE → LBH → **1376×768** ×2，LoRA 0.75，attn=kitchen，VAE tile 256/64。用户报单卡 5090 热 **~27s**。检索 **没有** 公开同名 gist。本仓同配方热 **52–53s**（VAE 18 + 高分 ~19）。

**480P ~30s：** 若成片就停在 864×480，**不必 LBH**。3-step TaoMate LoRA + 官方 VAE + 常驻，是最像 30s 的公开机制。公开 5090 **单段** 5s 480p：zenn 20-step 热 **76.5s**；Chaosyn 4-step dual-clock **46s（~0.7MP）** / 0.4MP 8-step **57.2s**。**没有**「864×480 5s ~30s」的第三方全含表。用户 30s 按 **私有配方目标** 记账，不要写成社区 PR 头牌。  
**768P ~27s：** **P0 配方已经对。** 缺的是厨房核与常驻，不是再发明一种 lift。

---

### 2.5 VDN / OpenVDN / Video DeltaNet（再次确认不是 VAE）

**2026-09-17 更新仍写：改 video-video 注意力**（窗 softmax + Video Delta 线性支 + 小 LoRA），**官方视频/音频 VAE 原样在 `h3-base/`**。[README](https://github.com/OpenVDN/vdn-minimax-h3) · [博客](https://openvdn.github.io/h3-vda-blog.html) · 论文 [arXiv:2609.20744](https://arxiv.org/abs/2609.20744)

README 原文：*We exclude model loading, warm-up, VAE decoding, and MP4 encoding. For a live setup, we recommend running … VAE decoding … on separate machines.*

**数字（768p，时长与本仓 5s 不同）：**

| 配置 | 秒 | 含 VAE？ |
| --- | ---: | --- |
| SGLang 8×B200 8-NFE，**14.4s** 片 | denoise **6.9s** / e2e **~9.0s** | **否**（e2e 仍建议 VAE 另机；9.0s 是 serving 口径） |
| 1×B200 8-NFE FP8 | denoise **51s** | 否 |
| Comfy 端口 1×5090，1280×736 145f 8-step | ~**17 s/it**，采样 ~2:15 | 作者写含音频；**headline 拿不到**；FA4 不服务 sm_120 |

INT8 社区支 [drbaph](https://huggingface.co/drbaph/vdn-minimax-h3-int8-convrot-comfyui)：同卡 1280×736 61f，bf16 ~111s → int8 ~95s（**未钉 5090**）。

**480P / 768P：** 都不能当官方 VAE 成片主路径。32GB 常驻不了 B200 FP8 配方。与 09-15 §2.5 结论一致；本次只是把日期和 6.9s 更新钉死。

---

### 2.6 TAE / TAESD / 轻量 decode

**换解码器。** [Kijai/MiniMax-H3-TAE](https://huggingface.co/Kijai/MiniMax-H3-TAE) · madebyollin taeh3 · [PreviewOverride](https://github.com/simsim9-stack/ComfyUI-MiniMaxH3-PreviewOverride)（decoder-only ~40MB，holdout **25.5 dB**）。Comfy issue [#15592](https://github.com/Comfy-Org/ComfyUI/issues/15592)：官方 VAE 多秒预览 vs latent2rgb；TAE 是中间档，**16GB 上 preview_frames 过大仍 OOM**。

本仓 TAE **7.03s 彩色闪**（09-15）。**两档都禁止当质量尺。** 失败可见：禁止静默 TAE。

---

### 2.7 LTX / 异模型精修

见 §2.3。额外：muninn 写 Comfy **&lt;0.34** 时 refine **exit 0 + 黑帧/彩噪**，latent 统计还「健康」。质量事故会装成成功。

**480P / 768P：** 速度参照（5090 热 30.6s @ **1344** 不是 1376）。**默认质量尺保持官方 H3 VAE。**

---

### 2.8 Flash-VAED / KJ Triton VAE / PatchTritonVAE（+ TRT 旁路）

**对不上 H3 ViT decoder。** PatchTritonVAE 自述只 Wan / KL / LTX conv，其它 arch **pass-through**。[wiki](https://comfyui-wiki.com/en/custom-nodes/ComfyUI-KJNodes/nodes/patch-triton-vae) · [源码](https://github.com/kijai/ComfyUI-KJNodes/blob/main/nodes/triton_vae.py)  
Flash-VAED：通道剪枝 + 换 CausalConv3D，实验模型 Wan/LTX。[arXiv HTML](https://arxiv.org/html/2602.19161v1)

**H3VAE_TRT / ONNX：** 试图编 **官方** H3 VAE，不是剪 conv。社区空 latent 5s 1344 FP16 **17.87→11.84s / 65.84 dB**（二手博客转 model card，**不是**本次能打开的 GitHub README 原文；当弱证据）。muninn 在 **14-step 15s 768P** 上测 **−1.7% e2e**，并拆穿「别人测 −9.8%」是因为别人 8-step、VAE 占比高——Amdahl，不是 TRT 撒谎。

**480P：** 不值得。  
**768P：** P2 对照。P0 仍是厨房融合核（同合同、已合入 Comfy 核心）。

---

### 2.9 NVFP4 / INT8 / 蒸馏 LoRA（turbo4）

**INT8 ConvRot** 是 Comfy-Org 默认 DiT，**官方 VAE 另文件**。0.36 起官方仓带 **VAE int8 convrot**（见 §2.1）。

**NVFP4：** `supports_nvfp4_compute` 在 CC≥10 为 native。smeltcore 5090：社区 `coolthor` pruned NVFP4 **11.7GiB vs INT8 19.5GiB**，同工作流 **175 vs 185s（5.4%）**。Chaosyn：NVFP4 文件 ~33GB **offload 后 8-step 比 INT8 更慢**。Issue [#14157](https://github.com/Comfy-Org/ComfyUI/issues/14157)：5090 nvfp4 绕过 PyTorch allocator → VAE decode 150–300s / OOM。

**蒸馏族：**

| 权 | 步 | 5090 5s 公开全含 | 备注 |
| --- | --- | --- | --- |
| LightX2V FL2V turbo 4-step v1.2 768p | 4 | **搜不到 5090 全含** | 音频修复；有人报画面不如 v1.1。[wiki 2026-09-06](https://comfyui-wiki.com/en/news/2026-09-06-minimax-h3-fl2v-turbo-v1-2) |
| LightX2V Ref2VA turbo 8-step | 8 | 无 | [wiki 2026-09-04](https://comfyui-wiki.com/en/news/2026-09-04-minimax-h3-turbo-ref2v-8step) |
| Chaosyn INT8+Turbo dual-clock | 4 / 8 | **~0.7MP 46s / 58s**（含 VAE+存盘） | **不是 864×480 也不是 1376**。4-step 旧 sampler 音频坏 |
| zenn turbo 4-step R2V+int8 | 4 | 失败（视+声） | 480p 档 QC 不过 |
| ChrisJohnson 无 turbo | 10 | **1344×768 5s 热 85.23s 全节点** | 普通 Comfy 768P 仍是分钟下沿 |
| **PDD Acc**（§3） | 8 或 4 | **无 5090 表** | 官方 head bank |

**480P ~30s：** 少步是 **P0**。公开最接近的是 Chaosyn 4-step **46s @ ~0.7MP**，仍高于 30s，且画布更大。要 30s 必须叠 **热常驻 + 厨房 VAE（小）+ 3/4 NFE**，并人眼过官方 VAE 片。  
**768P ~27s：** 单段少步公开到不了 27s（85s / 41.5s native 4-step 是另一质量）。27s 来自 **2-stage 日程**，不是 NVFP4。

---

### 2.10 TeaCache / te-cache / 步间 cache

**改 DiT 步间**，官方 VAE 不动。

Icyoung 自称 3×：20-step、1024×576、124f、CMP 170HX（**不是 5090**）。[README](https://github.com/Icyoung/ComfyUI-MiniMaxH3-TeaCache)  
3060 横评：TeaCache **1.94×**，FirstBlockCache 1.64×，TE-Speed **0.999× 且改输出**。[cache 横评](https://www.minimaxh3tutorial.com/nodes/cache)  
Comfy.icu 节点说明：**8-step + 默认 warmup/cooldown 几乎跳不了**；甜点 20–40 step。PDD README：**不要叠** blockcache（8-step 蒸馏无缓存空间）。

本仓 te-cache 是 **TE prompt cache**（热 HIT~0），不是 TeaCache。不要混名。

**两档冲 KPI 的配方都是 3–8 NFE → cache 列「几乎不能搬」。**

---

### 2.11 多卡 Ulysses / CFG parallel（只对照）

方向档 **冻结 8 卡**。公开对照：

- RH Lightning：8×6000D sm_120 85GB，5s 1344 **warm 生成 28.7s**，相对同机 50-step 348.8s = 12.2×；**不含 load / 首次 compile**；加速权未公开。[GitHub](https://github.com/RH-RunningHub/MiniMax-H3-MultiGPU-Lightning)  
- VDN 8×B200：见 §2.5。  
- TaoMate 8×H20：见 §2.4。  
- 本仓：`canAccessPeer=0`，U=8 更慢（09-16）。

**禁止**把 28.7s / 9.0s / 14.8s DiT 画进单卡 27s 列。

---

## 3. 漏网（09-15/09-16 没写成主线、2026-08 末–09 出现）

检索词覆盖 MiniMax H3 / ComfyUI / 5090 / 2026-09。下列是本仓枢纽/选型/社区方向 **未当主线** 或 **版本过旧** 的。

### 3.1 FastH3 8-Step V2 + Comfy-Org repack（高优先级漏网）

旧文停在 FastH3 **4-step Preview + sm100a VSA**。  
现：[FastVideo-FastH3-8-Step-V2](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2) — DMD2 data-free、**VSA 80% 稀疏**、shift **10 不是 12**、**只蒸 T2VA**（FL2VA/Ref2VA 未蒸）。测试默认 **4×B200**。消费卡要 `--vsa-kernel triton --no-fa4`。  
Comfy-Org 文件布局含 **官方 video VAE fp16 与 int8_convrot**：[Comfy-Org/FastVideo-FastH3](https://huggingface.co/Comfy-Org/FastVideo-FastH3)（HF 显示 15 万+ downloads）。  
社区警告：无 VSA 核会丢掉 `to_gate_compress` → **均匀噪声**（[realrebelai GGUF 卡](https://huggingface.co/realrebelai/FastH3_GGUFs)）。4-step 旧权 keep_percent **10**；V2 应 **20**。

**5090 5s 480P/768P 全含：搜不到。** 对方向档（用户 R2V JSON）**对不上**（Ref2VA 未蒸）。480P T2V 研究档可跟；不要静默 dense。

### 3.2 Alibaba PAI PDD Acc LoRA（官方 8-step，特殊 loader）

[alibaba-pai/MiniMax-H3-Acc-LoRAs](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs) · 方法 [arXiv:2607.26004](https://arxiv.org/abs/2607.26004) · Comfy [Jalen-Brunson](https://github.com/Jalen-Brunson/ComfyUI-MiniMax-H3-PDD-Acc) / T8 节点。  
rank-64 主干 + **32 个 interval 的 final-layer video/audio head**。普通 `LoraLoader` **丢掉蒸馏**。CFG=1，Euler，shift 12/3。官方 demo **4 与 8 NFE**。**禁止与其它蒸馏 / TeaCache 叠。**  
**无 1×5090 全含秒数。** 与 TaoMate 3-step 互斥选一条做人眼，不要两条加载器。

### 3.3 ComfyUI 0.36.0 / kitchen 0.2.34 VAE PR（主线升级，不是新族）

09-16 写厨房 ~4s 时，**这组 PR 尚未合入**（合入 09-13/15）。执行厨房迁 C 应以 **PR 167 核清单 + 0.36 `vae.py`** 为准，而不是更早的「一行 fused」口头描述。int8 路径从 18.7→12.9s（15s 片）是 **当前** 社区核，不是 09-15 的 4s 口头。

### 3.4 kitchen 原生 `sol_attn` + Comfy `BlockSparseAttention`

厨房 README 矩阵有 `sol_attn`（cuda/eager/hip）。Comfy 核心 2026-09 增加 BlockSparseAttention 节点，FastH3 模板用 VSA backend。这是 **DiT attn 插件**，不是 VAE。768P 长序列比 sage3 假 hit 更值得跟；仍要硬失败。

### 3.5 KingGore SM120 Sol-Attn（无 CuTe 依赖）

纯 PyTorch `flex_attention`，号称零 CUTLASS。核表 seq 8k–32k 有倍数。**自动 fallback SDPA** ——与本仓「禁止假 hit」相反。可当对照，不能当加速档成功条件。

### 3.6 Spectrum / LongMedia / TE-Speed

muninn 15s 栈 Spectrum ON。LongMedia 嵌 Sol 做长序列切块（[vizart-vj](https://github.com/vizart-vj/ComfyUI-MiniMax-H3-LongMedia)）。TE-Speed 3060 横评无效。**均无** 5s 864×480~30s 或 1376~27s 一手。不进主线。

### 3.7 明确搜不到 / 对不上的

- 公开「1×5090 · 5s · **864×480** · 热全含 **~30s**」——**搜不到**。最接近：Chaosyn 4-step ~0.7MP **46s**；zenn 20-step 864×480 **76.5s**。  
- 公开「1×5090 · 5s · **1376×768** · 官方 VAE · 热全含 **~27s**」——**只有用户私有 Comfy 图**。公开 768P 5s：ChrisJohnson 10-step **85.23s**；muninn native 4-step 1344 **41.5s**（少步质量另说）；hybrid LTX **30.6s**（换解码器）。  
- 「4090 768P ≤50s 全含」——09-16 已判搜不到，本次未出现新一手。  
- Flash-VAED / PatchTritonVAE **支持 H3 ViT**——搜不到，文档明确排除。  
- Sage3 官方 H3 e2e 表——**无**。

---

## 4. 我们应该怎么排

前提：1×5090 热跑；质量地板 = Comfy 官方 H3 VAE 无格子；**不**开 8 卡冲 KPI；**不**把 LTX/TAE/金线 MAE 当默认尺。

### 4.1 两档分列（不要合成一列 27s）

**A. 5s 480P（864×480）~30s**

公开单段 20-step 是 **76–131s**。30s 只能来自 **少步 + 热常驻**，VAE 在 0.4MP 上已经 4s 量级。

| 序 | 做 | 不做 | 验收口径 |
| ---: | --- | --- | --- |
| P0 | 3/4-step 蒸馏（TaoMate LoRA 或 dual-clock turbo；人眼+听音过官方 VAE 片） | 未蒸馏 `--steps 4`；zenn 已失败的 turbo+R2V 当默认 | 热 e2e、含 VAE+mux、画布写死 864×480 |
| P0 | TE/DiT/VAE 加载常驻（Chaosyn 冷额外 **14.3s**） | 把冷 load 折进 30s | 第二次 `load≈0` |
| P1 | sage2 或 kitchen attn（采样 −20% 级） | sage3 假 hit | 日志核名真实 |
| P2 | 厨房 VAE 核 | 为 480P 单独减 tile | waffle 红线 |
| 不做 | LTX、TAE 成片、Ulysses、TeaCache 叠 3-step | — | — |

**B. 5s 1376×768 ~27s（方向档）**

用户 27s = **2-stage 配方**，不是单段 8-step。本仓同配方 ~52s。09-16 回代仍合理，但 **VAE 目标从「口头 4s」改成「对齐 Comfy 0.36 int8」**：

| 段（09-16 热 lift） | 现状 | 社区对照 | 刀 |
| --- | ---: | --- | --- |
| VAE | 17.6–18.2s | 5s 768P 公开 decode **10s**；15s 片厨房 **12.9s**；用户栈可能更快 | **P0 厨房核 + 常驻**。不要写「4s 已是 PR 事实」 |
| 高分 2 NFE | ~19s（sage 池 + window=1 权流） | Comfy 同配方份额更小 | **P1 真 attn + 卸池让 INT8 常驻** |
| 低分 3 NFE | ~10s | 480P 少步 | 已在配方里 |
| LBH | ~2.5–4.6s | 学到的 3D | 不是第一刀 |
| mux | ~2s | — | 次要 |
| TE 热 | ~0（有 te-cache） | 第二次 Queue 跳过 CLIP | 热评测必须常驻 |

算术：VAE 18→~10 已能看见 **~44s**；再把高分 19→社区 ~8–10，才 **看见 27s**。继续说「C 碎核已是下限」仍然错；说「厨房一定 4s」则 **2026-09-22 没有 5s 768P 一手**。

| 序 | 768P | 说明 |
| ---: | --- | --- |
| **P0** | kitchen VAE 核迁 C（256/64 不动） | 对齐 0.36 int8，失败硬退，禁止 TAE |
| **P0** | 加载常驻 | WEIGHT_CACHE 默认开；pin 活过 clip；热 vae_load≈0 |
| **P0** | 保持 2-stage 日程为极速默认 | 864×480×3 → LBH → 1376×2；gold 对照臂 |
| **P1** | 真 768P attn：kitchen `sol_attn` / Sol-Attn sm_120 / 修 sage 形状 | 缺核或 hits=0 **硬失败** |
| **P1** | 人眼二选一：TaoMate 3-step vs PDD 8-step vs FastH3 V2（仅 T2V） | 不叠蒸馏；R2V 不要 FastH3 V2 |
| **不做默认** | LTX 成片、TAE、NVFP4 当加速银弹、TeaCache、8 卡 | 30.6s 只写在日志当速度参照 |
| **冻结** | Ulysses / aikitoria P2P | 解冻条件不是 27s |

### 4.2 一句话给执行者

厨房核 + 常驻 + 用户已经认的 2-stage，仍是唯一能同时守 **官方 VAE 质量地板** 并看见 **768P 27s** 的路径。480P 30s 更靠 **少步与常驻**，不要用 768P 的 VAE 故事去解释 480P。Sol/LTX/VDN/TAE 的漂亮秒数全部换过尺子或换过机器；**把它们写进 27s 列才是这次最容易再犯的错。**

---

## 5. 一手链接（本次实际打开或检索到正文）

- comfy-kitchen README：https://github.com/comfy-org/comfy-kitchen  
- kitchen PR 167：https://github.com/comfy-org/comfy-kitchen/pull/167  
- Comfy PR 16187：https://github.com/Comfy-Org/ComfyUI/pull/16187  
- Comfy-Org MiniMax-H3：https://huggingface.co/Comfy-Org/MiniMax-H3  
- 0.36 新闻转述：https://comfyui-wiki.com/en/news/2026-09-19-h3-video-vae-optimization  
- Kijai int8 VAE 5s 10.0s：https://huggingface.co/Kijai/MiniMax-H3-experimental/discussions/21  
- Sage3 README：https://github.com/thu-ml/SageAttention/blob/main/sageattention3_blackwell/README.md  
- Comfy sage 噪声：https://github.com/Comfy-Org/ComfyUI/issues/15263  
- Sol-H3-Spark：https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3-Spark/  
- H3 On-Device：https://nvlabs.github.io/Sana/Sol-Engine/H3-OnDevice/  
- muninn Spark/5090 30.6s：https://ai-muninn.com/en/blog/h3-ltx-two-stage-dgx-spark  
- muninn 5090 hybrid：https://ai-muninn.com/en/blog/minimax-h3-ltx-hybrid-refine-rtx5090  
- muninn 15s 栈：https://ai-muninn.com/en/blog/minimax-h3-rtx5090-config-stack  
- ChrisJohnson 85.23s：https://github.com/ChrisJohnson89/minimax-h3-comfyui  
- Chaosyn 46–60s：https://blog.chaosyn.com/en/posts/minimax-h3-rtx-5090-speed-cost/  
- zenn 5090 480p：https://zenn.dev/toki_mwc/articles/minimax-h3-rtx5090-benchmark  
- smeltcore NVFP4：https://smeltcore.com/recipes/minimax-h3-on-rtx-5090-32-gb-video-audio-and-what-nvfp4-does-not-buy-you/  
- TaoMate HF：https://huggingface.co/TaoLiveAIGC/TaoMate-H3  
- LBH upscaler：https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler  
- OpenVDN README：https://github.com/OpenVDN/vdn-minimax-h3  
- VDN 论文：https://arxiv.org/abs/2609.20744  
- Comfy VDN：https://github.com/Saganaki22/ComfyUI-VDN-H3  
- Kijai TAE：https://huggingface.co/Kijai/MiniMax-H3-TAE  
- FastH3 V2：https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2  
- Comfy-Org FastH3：https://huggingface.co/Comfy-Org/FastVideo-FastH3  
- PDD Acc：https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs  
- PatchTritonVAE：https://github.com/kijai/ComfyUI-KJNodes/blob/main/nodes/triton_vae.py  
- Flash-VAED：https://arxiv.org/html/2602.19161v1  
- TeaCache：https://github.com/Icyoung/ComfyUI-MiniMaxH3-TeaCache  
- RH Lightning：https://github.com/RH-RunningHub/MiniMax-H3-MultiGPU-Lightning  
- KingGore Sol-Attn SM120：https://github.com/KingGore/ComfyUI_sol-attn_Blackwell  
- MiniMax-H3#46 tile：https://github.com/MiniMax-AI/MiniMax-H3/issues/46  

本仓对照（口径以社区方向文为准，选型/枢纽事实可引用）：`docs/superpowers/plans/2026-09-16-h3c-community-speed-direction.md` · `2026-09-16-h3c-scheme-selection.md` · `2026-09-15-h3c-module-opt-plan.md` · `out/verify_taomate/REPORT.md`
