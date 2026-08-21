# MiniMax H3 加速方案普查 + Sol-Attn 专章

日期：2026-08-21  
仓库：`h3.c`（纯 C/CUDA MiniMax-H3 T2VA，RTX 4090 24GB）  
本文不改推理代码、不跑 generate。数字来自联网源码/论文/本仓已存计划。

对照：

- 质量金线：[2026-08-21-remaining-quality-speed.md](./2026-08-21-remaining-quality-speed.md)
- LightWan 落地账：[2026-08-21-lightwan-h3-speed.md](./2026-08-21-lightwan-h3-speed.md)
- 更早 NVIDIA 加速菜单：`docs/nvidia_optimization_plan.md` §9

**结论先说：**

1. **Sol-Attn 是真东西**：training-free、on-the-fly 阈值路由、未选块用 pooled-KV 近似校正，不是 hard-drop 稀疏。论文 arXiv:2607.24027；H3 不在论文评测里。
2. **社区已在 4090/5090 + MiniMax H3 跑通**（kijai Triton）。但 Comfy 对照基线是 **SageAttention**，本仓对照基线是 **cuDNN fused SDPA**。两者不是同一把尺子。
3. **不要指望把 cafe DiT 50.5s 砍掉一半。** 合理预期是稳态注意力段再压 **数秒到约 8s**；step1 的 11s unpack/建图 Sol 帮不上。叠 MagCache/EasyCache 才是 Sol-Engine 宣传的 4–5×，本仓 8-step 上 MagCache 0.08 已质量 FAIL。
4. **现在不要接到 h3.c 质量默认。** 2026-08-21 孤立 CUDA Sol（tau=1.3 / sink=988 / block=64）在 cafe packed `T=15973,H=56,D=128` 上 **输给本仓 cuDNN：351ms vs 47ms（约 7.5× 慢）**。未跑 q2 generate。禁止 Triton 侧车。详见 §1.10。
5. A1 TE packed 常驻 + 有声 overlap 已合入：有声 q1 **75.4s**（te 11.6 / dit 47.5 / vae 9.3），checksum 与 p3 相同。到 40–50s 仍被 DiT 47.5s 挡住，必须跨红线。

禁止当质量：W4A8、关 deblock、`--frames 17` 冒充 5s、默认 magcache、TinyVAE 成片、未蒸馏 4 步。

---

## 0. 本仓对照账（必须算进）

质量金线：`t2va_complex_5s_cafe_p3`，**90.5s**，真 5.000s / 120 帧 @24fps / 864×480，无 MagCache。

| 刀 | 墙钟 | 备注 |
| --- | ---: | --- |
| **p3 质量默认** | **90.5s** | TE 16.4 + **DiT 50.5** + VAE 15.6；`dit_evals=8 dit_skips=0` |
| 8-step 4090 **无损下限实测** | **82–90s** | TE~16 + DiT~50 + VAE~16。p3 已贴这条带 |
| MagCache A/B thresh=0.08 | **79s** | skips=3；帧 RGB MAE **52–68**；**已回退，不当默认** |
| 有声 av8 | **109s** | 现已默认 audio；相对无声金线约 +18s |
| 有声 av20 | **140s** | **20 步不是同片更清** |

产品形状：

- packed `seq=15973`（text 574 + audio 414 + video 14985）
- 8 step、cfg=1（CFG 蒸馏）、INT8 ConvRot DiT、fp16 VAE tile 256 + deblock 0.28
- 本仓已做：NVRTC / seq pad 64 / TE resident act（每 linear 仍 H2D packed）/ INT8 权窗 37 freeze / 默认 audio mux / cuDNN SDPA

DiT 内部粗拆（用于后面估 Sol 能砍多少）：

- step1 ≈ **11s**（INT8 窗 unpack 37 块 + cuDNN 建图，几乎不随分辨率线性掉）
- 稳态 ≈ **5.33s × 7 = 37.3s**
- 每层 ≈ 107ms（50 层）。seq=16K 时注意力 FLOP 与 MLP+QKV GEMM 同量级；注意力是可砍对象，但不是 DiT 的全部。

---

## 1. Sol-Attn 专章

### 1.1 论文原理（arXiv:2607.24027 + 项目页）

来源：

- 项目页：https://nvlabs.github.io/Sana/Sol-Attn/
- 论文：https://arxiv.org/abs/2607.24027 （HTML：https://arxiv.org/html/2607.24027）
- 引擎总成：https://arxiv.org/abs/2606.23743 ；H3 on-device：https://nvlabs.github.io/Sana/Sol-Engine/H3-OnDevice/

**Training-free。** 不改权重、不做离线校准数据集。论文原句：on-the-fly attention sparsification。

现有 training-free 块稀疏的两个病：

1. **离线路由贵**：top-k / top-p 先物化整张 proxy score map 再选块。
2. **keep-or-drop 损质**：没选中的 KV 块直接丢掉。

Sol-Attn 三步合成 **一次 online-softmax**：

1. **Query-dependent threshold。** 块代理分数近似高斯。对 query 块 \(i\)：\(\tau_i = \mu_i + \beta \sigma_i\)，选 \(\widehat{s}_{ij} > \tau_i\) 的 KV 块。\(\beta\) 就是社区说的 **tau**。\(\mu,\sigma\) 从 pooled-K 的一二阶矩算出，不必物化 \(N\times N\) proxy。
2. **On-the-fly routing。** 外环扫 pooled-K 的 chunk，当场比阈值；超阈值的块立刻进内环做精确注意力。不写 routing index 到 HBM。
3. **Proxy-score reuse（近似校正）。** 低于阈值的块不用 hard drop，而用块均值 \(\bar K_j,\widehat V_j\) 做零阶 Taylor，把长尾质量加回 softmax 的分子分母。这是相对 Radial / SVG2 / KingGore hard-mask 的核心差别。

论文实现细节（附录 A，评测模型 **不是 H3**）：

- 物理块 64×64。
- Wan / Hunyuan / Bernini / Ideogram：**前 20% 去噪步 + 第一层** 保持 dense。
- Hunyuan / Ideogram：把 text KV 当 **sink**，精确分支保留。
- 论文数字：视频生成相对 FA3 最高约 **2.1×** e2e；剪辑 2.3×；塞进 Sol-Engine 后 Hunyuan 可达 **5.08×**（那是「融合 + 步缓存 + Sol」叠乘，不是单核）。
- H100 上相对 FA3，128K / 90% sparsity kernel 可达 5.4×。**这些是 50 步、720p/1080p、FA3 基线。**

论文 **没有** 评 MiniMax H3。H3 的 packed `[text|audio|video]` 是社区后接的。

### 1.2 三处实现对照（都读过源码/README）

| | NVIDIA 论文 / Sol-Engine | kijai Triton | Saganaki22 Comfy |
| --- | --- | --- | --- |
| URL | [Sol-Attn](https://nvlabs.github.io/Sana/Sol-Attn/) · [H3-OnDevice](https://nvlabs.github.io/Sana/Sol-Engine/H3-OnDevice/) · [Sana sol-engine](https://github.com/NVlabs/Sana/tree/sol-engine) | https://github.com/kijai/ComfyUI-SolAttn_triton | https://github.com/Saganaki22/ComfyUI-sol-attn |
| 核 | 论文 CuTe / Sol-Engine；NVIDIA 现有 Linux SM120 CuTe 可选 | 独立 Triton：`_tri_fwd.py` bf16、`_int8_fwd.py` INT8 | vendored NVIDIA Triton + 本地改：`sol_kernel/fwd.py`、`quant.py` |
| 测过 H3？ | Sol-Engine **50 步** BF16：Spark 480p 3.92×、5090 768p 4.52×（含 cache） | README 原文：**RTX 4090 和 5090 + MiniMax H3** | 5090 本地矩阵；SM86 3090Ti / SM89 4080 社区冒烟；H3 节点完整 |
| 默认 tau | 论文 \(\beta=1.0\)；H3-OnDevice 写 \(\tau=1\) | **1.3**；`start_percent=0.2` `end_percent=0.9` | **1.3**；scheduled：`tau_start=1.3` → `tau_end=0.8` |
| INT8 | 论文核是 bf16 稀疏；Sol-Engine 另有 NVFP4 等 | **默认 `int8_qk=True` `int8_pv=True`**。Sage 风格 **smoothed full-key INT8** | **默认关。** residual-int8：只量化 K 相对块均值的残差，均值走 bf16 |
| H3 Morton | 论文未写 H3 | **`_morton_h3.py` 专门做 packed** | 未做 Morton；靠 sink 保 conditioning |
| sink | Hunyuan 把 text 当 exact KV | 默认 **`exact_kv_and_rows`**（音频 query 也 dense，~20% 成本） | 默认 **`exact_kv`**（~3%）；可选 `exact_kv_and_rows` |
| 相对 Sage | 论文对照 FA3，不是 Sage | 社区图：Sage vs Sol；BENCHMARKS 历史交叉 | 16K：Sage 14.68ms → Sol bf16 9.57ms（**1.53×**，5090 随机张量） |
| 架构 | SM90+ CuTe 主线；Triton 参考 | SM89 **pointer**（4090）；TMA 要 SM90+，默认 `use_tma=False` | SM86/89/120 pointer；SM90/100/121 TMA |
| 许可 | Apache-2.0（Sana） | 未标 license 文件 | Apache-2.0，kernel 来自 NVIDIA |

**kijai 仓库没有大 README**，只有 2.3KB `readme.md`。关键旋钮和 Morton/H3 sink 都在 `__init__.py`、`_morton_h3.py`、`_tri_fwd.py`、`_int8_fwd.py`。

用户已存文档：`/Users/digenman/.cursor/projects/Volumes-Ben-Digen-code-h3-c/uploads/ComfyUI-sol-attn-2.md`（Saganaki22 README 镜像，v0.6.2）。

第三家（不建议当主线）：

- https://github.com/sumeetprashant/ComfyUI-SolAttn — 通用 hook，15s H3 上 Triton 路径一度 **慢于 Sage**（+3.5%），flex 才快。
- https://github.com/KingGore/ComfyUI_sol-attn_Blackwell — `flex_attention` **hard mask**，无近似校正，密度 8–11%。BENCHMARKS：8K rel L2 vs SDPA **2.06**（Sol bf16 是 0.00097）。**不同方法。**

### 1.3 Kernel 在干什么（读过 `_tri_fwd.py` / `_int8_fwd.py` / `_morton_h3.py`）

公共数学（kijai `_tri_fwd.py`，块=64，group=32）：

```
for each query block:
  for each KV group:
    scores = Q @ KC^T          # pooled-K，routing + 近似校正
    routed = (mean(scores) > tau) OR |q_block - kv_block|<=1 OR sink_kv
    未选中 → 用 scores × VC 做近似 softmax 更新
    选中   → 再 Q @ K_full、P @ V_full 精确更新
```

强制 **邻块 exact**（`|Δblock|≤1`），避免把局部时空邻域全部近似掉。

INT8 两条路：

| | kijai `_int8_fwd.py` | Saganaki22 `sol_kernel/quant.py` |
| --- | --- | --- |
| K | Sage 风格 **整 key 平滑后量化**；平滑是行公共常数，softmax/routing 与 bf16 bit-identical | **只量化块均值残差**；均值已在 bf16 routing 分数里，精确加回 |
| 近似分支 | 仍 bf16（pooled 是 O(N·d)，量化不划算） | 同 |
| vs 己方 bf16 Sol，8K rel L2 | **0.029**（full-key） | **0.008**（residual）；再开 `int8_pv` → 0.014 |
| 默认 | Comfy 节点 **开** | Comfy 节点 **关** |

结论：INT8 **不是** Sol-Attn 方法本身，是 exact 分支的可选量化。论文核是 bf16。接到 h3.c 若要做，应先 bf16 稀疏，不要一上来叠 INT8 attn——本仓 DiT 权已经是 INT8 ConvRot。

### 1.4 H3 packed `[text|audio|video]` 与 Morton

H3 是 **一条自注意力**，不是独立 cross-attn。cafe：`[text 574 | audio 414 | video 14985]`。

kijai `_morton_h3.py` 原文要点：

> H3 packs one self-attention sequence: `[text][cond][ref_img][ref_audio][audio][video]`.  
> **Only the target video segment is reordered** — text, audio and reference rows keep their positions.

视频段始终在最后、t-major 再 h 再 w。Morton 只打视频 span，让每个 64-token 块变成紧凑 3D 邻域，而不是「一帧里一条 2×32 细带」（细带会把同一局部拆进很多块，路由变差）。

**H3 时间轴不均匀**：`FRAME_PER_TOKEN = (1, 4, 4, 4, 4)`。kijai 因此默认 `morton_curve=2d_frame`（帧内 Z-order，帧序不变），注释写明 3D 曲线会把时间上远离的 token 捆进同一块。

Morton 对 **dense** 注意力数学上是恒等（token 与 RoPE 一起 permute，最后一层 inverse）。对 **Sol** 它改变块内相关性，从而改变哪些块被选 exact。

Saganaki22 **没有** Morton，用 `sink_conditioning` 保 text/audio：

- `exact_kv`：所有 query 看 conditioning KV 都走精确分支（~3%）
- `exact_kv_and_rows`：conditioning 的 query 行也对全部 KV dense（~20%，护音频）
- `off`：不保护。社区配方写明 **音频会先坏**

kijai 还把 video span 写进 `transformer_options["sol_h3_video_span"]`，供 sink 用。Morton 默认 **关**；sink 即使 Morton 关也会装 H3 hooks。

本仓 pack 在 `h3_dit_w4a8.c` / `tools/h3_cuda_generate.c`：T2VA 默认 `[text|audio|video]`，F2VA 才有 cond/ref。接 Sol 必须把 **text+audio 前缀块标成 sink**，否则对白/口型是第一张牌。

### 1.5 旋钮：tau / start_percent / end_percent

| 旋钮 | 论文 | kijai 默认 | Saganaki22 默认 | 含义 |
| --- | --- | --- | --- | --- |
| tau / β | 1.0 | **1.3** | **1.3** | 高出均值多少个 σ 才 exact。kijai tooltip：1.0 ≈ 16% exact，1.5 ≈ 7%，2.0 ≈ 2.7% |
| start_percent | 前 20% dense | **0.2** | scheduled 用 `dense_percent`，默认 **0.0**（论文配方是 0.2） | 高噪构图期保持 dense |
| end_percent | 论文主要保头不保尾 | **0.9** | `tau_end=0.8` 把后期变 dense | 低噪细节期 |
| min_tokens | — | 4096 | 4096 | 短序列 Sage/cuDNN 更快 |
| dense_blocks | 第一层 dense | 空（可填 `0-2,-1`） | 空 | 首尾层最敏感 |
| sink | text sink（部分模型） | `exact_kv_and_rows` | `exact_kv` | H3 必开至少 exact_kv |

本仓只有 **8 步**。`start_percent=0.2` ≈ 前 1.6 步 dense，`end_percent=0.9` ≈ 最后 0.8 步 dense → 大约 **2–3/8 步根本不走稀疏**。加速被稀释，但这是论文保质量的配方。8-step 蒸馏后再把 8 步全稀疏，风险比 50 步大（MagCache 已经证明）。

### 1.6 与 SageAttention 对比

Sage = **稠密**注意力的量化核（QK INT8 / PV FP8），数学仍是满 softmax。  
Sol = **稀疏**注意力（多数 KV 块只走 pooled 近似）。

可以叠：Comfy 推荐 `KJ Sage → H3 mem-efficient Sage → Sol`。Sol 不接的短序列/dense 步回落到 Sage。

Saganaki22 BENCHMARKS（RTX 5090，H3 形状 `B=1,H=56,D=128`，tau=1.0，**随机高斯 = routing 最坏情况**）：

| tokens | PyTorch SDPA | Sage | Sol bf16 | residual int8_qk | int8_qk+pv |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 23.97 | 3.93 | 2.84 | 2.27 | 1.98 |
| **16,384** | 94.85 | **14.68** | **9.57** | 8.10 | 7.05 |
| 32,768 | 378.41 | 58.09 | 35.72 | 30.06 | 25.91 |

cafe packed 15973 ≈ 16K 行。**相对 Sage** Sol bf16 ≈ **1.5×**。

**相对本仓 cuDNN 不是这个比。** 本仓 2026-08-16：满 seq 1L **手写 sage2 ~9.0s vs cuDNN ~150–177ms**（缺真 Sage2++ MMA）。cuDNN 已经是 Ada 上的稠密上限。Comfy 的 Sage 赢的是 PyTorch SDPA/FA2，不是我们的 cuDNN。

官方 Comfy 教程仍写 Sage 能把 H3「大约快一倍」：https://docs.comfy.org/tutorials/video/minimax/minimax-h3 ——那是对 **Comfy 默认注意力**，不是对 h3.c。

独立测评（https://xhinker.medium.com/minimax-h3-everything-i-know-after-days-of-testing-what-works-what-not-648e4df78128）：

> Sage + EasyCache 是他测过最好的组合（>100%）。**Sol-Attn 可以更快，但会掉细部，人脸尤其明显，终稿他停用了。**

### 1.7 质量风险：小脸 / 对白

**小脸（cafe 金线的死穴）**

- 480p 上 20–30px 脸只占极少 video token。阈值路由按 **块均值** 决定 exact；背景大块分数高时，脸所在的 64-token 块可能被近似。
- 近似校正是 zeroth-order pooled V，**补纹理补不出身份**。这和 Distance Attention「中前期背景吞五官」是同一层问题，方向相反：Distance 是加 mask 保局部，Sol 是减计算碰局部。
- Morton `2d_frame` 能让脸的空间邻域进同一块，**可能**降低误伤；3D Morton 在 H3 非均匀时间轴上可能更糟。
- 论文保质量靠 **前 20% dense + 第一层 dense**。Saganaki22 自己写：H3 节点 **没实现** dense first-layer 和论文其余保守配方。
- 8-step 蒸馏：每步都更「值钱」。MagCache 跳 3 步 → RGB MAE 52–68。Sol 不跳步，但每步注意力都是近似。

**对白 / 音频**

- 不设 sink：音频 token 被当普通 KV 稀疏，社区明确「音频先坏」。
- `exact_kv` 保 prompt 遵循和音画同步，成本约 3%。
- `exact_kv_and_rows` 让音频 query 也 dense，~20%，护生成音轨。本仓 **默认有声**，若接 Sol 应至少 `exact_kv`，产品有声档倾向 `exact_kv_and_rows`（吃掉一部分加速）。
- EasyCache 已有 H3 音频回归：[ComfyUI#15326](https://github.com/Comfy-Org/ComfyUI/issues/15326)（用视频变化决定 skip，却把残差套到音频；H3 音频 σ 调度与视频不同）。Sol 与 cache 叠乘更危险。

**Sol-Engine 自己的质量声明**是 50 步 BF16 + First Block Cache，LPIPS 图上「near-lossless」。那不是 8-step INT8 cafe 小脸。

### 1.8 预期能砍 DiT 50.5s 多少

先拆别人的梯子，再外推。

NVIDIA Sol-Engine H3-OnDevice（**50 步、BF16 参考实现**，不是 8-step INT8）：

| | Spark 480p | 5090 768p |
| --- | ---: | ---: |
| Baseline | 710.6s | 1045.4s |
| + kernel fusion | 511.8s（1.39×） | 971.6s（1.08×） |
| **+ Sol-Attn τ=1** | **451.2s（再砍 12%）** | **708.3s（再砍 27%）** |
| + First Block Cache | 181.3s（3.92×） | 231.2s（4.52×） |

注意：**cache 才是 4× 的主力**；Sol-Attn 在已经融合的 480p 上只再砍约 **12%** 墙钟。768p 序列更长，注意力更占，Sol 才到 27%。

本仓已经做完「kernel fusion」那一档（INT8 GEMM + cuDNN SDPA + quiet + 权窗）。对应梯子的第二级。

粗算（4090，seq≈16K，8 step）：

| 段 | 时间 | Sol 能否动 |
| --- | ---: | --- |
| step1 unpack + cuDNN 建图 | ~11s | **几乎不能** |
| 稳态 7 步 × 注意力 | 估 12–20s | 能。若 Triton 相对 cuDNN 1.2–1.5×，砍 **3–7s** |
| 稳态 GEMM / AdaLN / RoPE | 其余 ~17–25s | 不能（那是 Sol-Engine 的 fused RMS-AdaLN；Saganaki22 另有 fusion 节点，bit-exact） |
| 前 20%+ 尾部 dense 门 | 2–3/8 步 | 稀释稀疏收益 |

**中位预期：DiT 50.5s → 44–48s（砍 2–6s）。**  
乐观（Triton 在 4090 上明显快于 cuDNN、tau=1.3、sink 只用 exact_kv、不开 dense 头）：**约 8–10s**。  
悲观（Triton 慢于 cuDNN，或 sink=`exact_kv_and_rows` 吃掉 20%）：**0 或负**。

**不可能把 50s 砍成 25s。** 论文 2.1× 是 vs FA3、长序列、50 步。Spark 480p 融合后再加 Sol 只有 1.13×。

Saganaki22 自己的整模数字更诚实：H3 15s 480×864 20 步，hook 版 Sol 相对 Sage **9.91 → 8.92 s/it（−10%）**。核级 1.5× 进整模只剩一成，因为还有 MLP/AdaLN。

### 1.9 是否接到 h3.c：C/CUDA 重写 vs 调 Triton

本仓 SDPA 入口：`h3_gpu_cuda.cu` / `h3_gpu_cuda_sdpa_cudnn.cpp`，`H3_CUDA_SDPA_BACKEND=cudnn` 默认。Packed 布局已在 `h3_dit_w4a8_forward_*`。

| 接法 | 判定 |
| --- | --- |
| **进程内调 Triton/Python** | **禁止。** 架构红线：纯 C/CUDA，不把生成栈改成 Comfy 节点。 |
| **把 kijai/Saganaki kernel 当参考，CUDA 重写 nested-loop** | 数学可复现：preprocess（block mean K/V + 对角阈值）+ 外环 pooled GEMM + 内环精确 SDPA + sink mask。工程：**数周**。autotune、ragged T、56 head、D=128、INT8 权路径的 QKV 布局都要自己做。 |
| **先孤立 kernel bench** | **唯一合理的下一步**（仍不改 generate）。同形状 `T=15973,H=56,D=128,bf16` 对拍 cuDNN。输了就停。 |
| **Morton** | 可选第二刀。只需 permute video span + RoPE，dense 恒等，可先关。H3 用 `2d_frame`。 |
| **AdaLN fusion / FFN chunk** | Saganaki22 的 fusion 是 bit-exact 显存/launch 优化，**不是 Sol**。本仓 p5 已试过 FFN 包装，更慢已回退。4090 上 AdaLN fusion 可能还有一点（Sol-Engine 说 H3 13B 在 AdaLN），属 A 档工程，与稀疏正交。 |

**建议：B 档可测，不进质量默认。不先写 CUDA 核。** 若将来 kernel bench 证明 Triton ≪ cuDNN，再开独立计划做 C 核，配方锁定：

- tau=1.3，`start_percent=0.2`，末步偏 dense
- `sink_conditioning=exact_kv`（有声产品再 A/B `exact_kv_and_rows`）
- 先 bf16 Sol，不开 `int8_pv`
- `dense_blocks=0-2,-1` 可作质量保险
- cafe_p3 帧 MAE / 口型 / 音频波形三门，过不了就关

### 1.10 实测（2026-08-21）：孤立 CUDA Sol vs 本仓 cuDNN — **输，停**

工具：`tools/h3_sol_attn_bench.c`（`--target h3_sol_attn_bench`）。4090，device-resident F32→FP16 SDPA，不含 H2D。未跑 Triton、未跑 magcache、**未跑** `t2va_complex_5s_cafe_q2_sol/`（kernel 未赢，按计划停）。

核：本仓 nested-loop CUDA（block=64，pooled-K 路由 + 精确 tile + pooled-V 校正）。`H3_SOL_TAU=1.3`，`H3_SOL_SINK_TOKENS=988`（cafe text+audio）。generate 默认仍是 `--sdpa-backend cudnn`；`sol` 仅 opt-in，质量路径不要开。

| 形状 | fill | backend | ms/iter | 备注 |
| --- | --- | --- | ---: | --- |
| T=512,H=4,D=128（冒烟） | local-corr | cuDNN | 0.05 | |
| 同上 | local-corr | **sol** | **0.09** | exact_frac=0.44，rel_l2=0.055 |
| **T=15973,H=56,D=128（cafe packed）** | local-corr（邻域相关，对路由偏乐观） | **cuDNN fused** | **46.74** | plan tensor_seq=16000 mask=1 |
| 同上 | local-corr | **sol CUDA** | **351.21** | exact_frac=**0.193**，rel_l2_vs_cudnn=0.15 |
| 同上 | local-corr | 本仓 flash f16 | 2758 | 1 iter，约 59× 慢于 cuDNN |

判定：**sol 没有稳定加速（351ms ≮ 0.92×47ms）。STOP。** 即使 exact 只占 ~19%，未融合的 nested-loop 仍远慢于 cuDNN Flash。手写 flash 更慢，和历史 sage2 满 seq 结论一致。

未跑 q2 有声 5s：接上去只会把 DiT 47.5s 拉长，墙钟不可能进 40–50s。目标仍 40–50s，本轮不承诺。

---

## 2. 市面 H3 加速方案普查

每条：**原理 → 来源 → 对 h3.c → 档**。档定义见第 3 节。

### 2.1 注意力核

#### Sol-Attn Triton（kijai）— **B，专章见上**

- URL：https://github.com/kijai/ComfyUI-SolAttn_triton
- 4090/5090 + H3 已测。pointer kernel 适合 4090（SM89 无 TMA）。
- 对 h3.c：可能砍 DiT 数秒；质量风险在小脸；需 CUDA 重写。

#### Sol-Attn Comfy 增强（Saganaki22）— **B（核）/ C（Python 栈）**

- URL：https://github.com/Saganaki22/ComfyUI-sol-attn
- 额外：zero-copy fused-qkv、scheduled tau、H3 modulation fusion、FFN chunk −37% MLP 峰值（INT8 ConvRot）。
- fusion/chunk 是显存刀，本仓 4090 24GB 不是主瓶颈；p5 已证乱融 FFN 会更慢。

#### SageAttention 2 / 2++ — **C 当默认 / 已证伪加速（本仓核）**

- URL：https://github.com/thu-ml/SageAttention ；Windows 轮子：https://github.com/woct0rdho/SageAttention
- Comfy：https://github.com/kijai/ComfyUI-KJNodes `Patch Sage Attention KJ` + `MiniMax H3 Memory Efficient Sage Attention Patch`
- 官方教程：https://docs.comfy.org/tutorials/video/minimax/minimax-h3 （相对 Comfy 默认约 2×）
- 本仓：`H3_CUDA_SDPA_BACKEND=sage2` 已接线；满 seq **9s/层 vs cuDNN 150ms**。真 Sage2++ MMA 未链入。**禁止改默认。**
- Sage3 = Blackwell FP4，非 4090。

#### W4A8 / INT8 ConvRot（Comfy）— **INT8 已做；W4A8 = C 主题**

- INT8 ConvRot：`minimax_h3_fl2va_pruned_int8_convrot.safetensors`，本仓质量默认。
- W4A8：Kijai mixed 4bit。本仓主题崩（文本门控过弱，对话先验）。**禁止当质量。**
- NVFP4 DiT：Blackwell 故事；4090=Ada。Optimization Suite 的 NVFP4 Fused MLP 明确 **Ada unsupported**。

#### Flash / cuDNN SDPA / 本仓 flash — **A 已做（cuDNN）**

- 默认 cuDNN fused。NVRTC + seq pad 64 已消灭 `fallback seq=15973`。
- FA3 = Hopper，4090 不可用。
- 手写 flash 相对 cuDNN 已无边际。

#### TensorRT / CUDA Graph — **C / 已证更慢**

- TRT：与自有 INT8 契约+纯 C 旁路冲突（`docs/nvidia_optimization_plan.md`）。
- CUDA Graph：`H3_DIT_W4A8_CUDA_GRAPH` 已接线；**cuDNN SDPA 不可捕获**。p5 pthread prebuild 90.5→98.8s，已回退。

#### Window attention / 稀疏 90% / Radial / SLA — **C 默认**

- LightWan extreme：`sparsity_ratio=0.9` + sage2。要微调，伤细部。
- RadialAttention：https://github.com/mit-han-lab/radial-attention ；Comfy：https://github.com/woct0rdho/ComfyUI-RadialAttn
- 本仓 `--attn-window` 是产品档，不进质量默认。
- Canvas/LightWan 文档已禁「稀疏 90% 当终稿」。

#### Token reduction — **C 质量默认**

- CLI `--token-reduction` 已有（空间 pool 再 expand）。与 float RoPE 质量档不兼容（旧日志 `tr_infer_spatial` 失败）。

### 2.2 步缓存 / 跳步

#### TeaCache — **C 首选**

- 论文：https://arxiv.org/abs/2411.19108
- 维护差，Comfy 核心已用 EasyCache 取代。H3 不在其 `model_type` 白名单。

#### EasyCache / LazyCache（Comfy 核心）— **C 质量默认（音频）**

- `comfy_extras/nodes_easycache.py`
- 处理 H3 双输出，但 [#15326](https://github.com/Comfy-Org/ComfyUI/issues/15326)：用视频变化 skip，残差套到音频 → 低音变闷、幅度腰斩。本仓默认有声。
- 8-step 上可跳步数更少（LightX2V [#579](https://github.com/ModelTC/LightX2V/issues/579)）。

#### MagCache — **已测 FAIL（质量）= C 默认；代码保留 opt-in**

- https://github.com/Zehong-Ma/MagCache ；论文 https://arxiv.org/abs/2506.09045
- 本仓 `--magcache-thresh 0.08`：79s，latent MAE 0.85，帧 MAE 52–68。

#### FBCache / First Block Cache — **B 可测，8-step 不看好**

- Sol-Engine H3 用 First Block Cache 做 4× 主力（50 步）。
- DGX 配方：thresh=0.08，`start_step=3`，`end_dense_steps=2`，`max_consecutive_skips=2`（https://github.com/drowzeys/keys-SM121-Optimized-MiniMax-H3-Nvidia-Sol-Engine-Kijai-SolAttn_Triton-Single-DGX-Spark/blob/main/RECIPE.md）
- 本仓 MagCache 已是同类 skip；8 步再 skip 几乎必漂。

#### cache-dit（DBCache ± TaylorSeer）/ SenCache — **B 研究 / C 默认**

- https://github.com/vipshop/cache-dit
- SenCache：https://github.com/vita-epfl/SenCache
- 实现成本高于 MagCache；8-step 蒸馏空间同样小。

#### `--reuse N` — **B 已有，默认关**

- 隔步复用 velocity。旧测 DiT −30–40% 但主题 PSNR 差。

### 2.3 蒸馏 / 少步 / 求解器

#### LightWan22-A14B 类 — **C 直接搬**

- https://huggingface.co/lightx2v/LightWan2.2-A14B
- 80× = **40→4 步蒸馏 + NVFP4 QAT + 稀疏**，绑 5090。详见 [lightwan 文档](./2026-08-21-lightwan-h3-speed.md)。
- H3 已是 8-step CFG-distilled，没有官方 4-step 权。

#### 步数 4 / 8 / 20 — **8=默认；4=C；20=C「更清」**

- 4 步无蒸馏权会崩。
- av20=140s，同片不更清，整段加步拉歪 8-step 轨迹。
- Comfy 官方模板常 20 步 `res_multistep`；那是另一条调度，不是把本仓 8 改成 20。

#### CFG skip — **A 已做**

- H3 已蒸馏 cfg=1。`cfg=5` 过曝。无第二路可并行。

#### YCNodes Dynamic CFG — **C 加速；B 质量微调**

- https://github.com/yichengup/ComfyUI-YCNodes-MiniMax-H3
- `cfg_low=0.9` → `cfg_high=1.1`。H3 flow matching 动态范围很小。可能略增步成本（若变成真双路），不加速。

#### CAB Sampler（Adams-Bashforth 校正）— **B 可测，不进默认**

- https://github.com/ByronLeeeee/ComfyUI-MiniMax-H3-Optimization-Suite
- 论文：https://arxiv.org/abs/2605.16736
- 用历史 velocity 做多步求解，**减少 NFE**（20→14/12/10）。不是 skip cache。
- 在 20 步官方轨迹上有意义；本仓已是 8 步，再减 NFE ≈ 未蒸馏 4 步，质量风险高。
- NVFP4 Fused MLP 部分 **Ada unsupported**。

#### Spectrum / Turbo 4-step LoRA — **C 质量默认**

- 社区「step forecasting」；有人口碑毁画。
- Turbo LoRA：换权重家族，本仓不接。

### 2.4 分辨率 / 二采 / 显存

#### YCNodes 六件套 — 质量刀为主，不是 480p 加速

源：https://github.com/yichengup/ComfyUI-YCNodes-MiniMax-H3

| 节点 | 原理 | 加速？ | 档 |
| --- | --- | --- | --- |
| **Tail 二采** | 透传一采 latent，重算锚点 | 接线，不加速 | A 钩子已有 `--init-latent` |
| **Prompt Relay** | packed 自注意力上对 video→text 加时间惩罚 mask | 不加速，略增 mask | B 叙事控制，非速度 |
| **Distance Attention** | 时空高斯感受野，挡背景吞五官 | 不加速 | **B 质量刀**（cafe 小脸对症） |
| **Dynamic CFG** | 见上 | 不加速 | C 加速 |
| **Sigma Refiner** | 低 σ 尾巴 **加步**（默认 +1） | **更慢** | C 加速；B 运动边缘颗粒 |
| **Tiled Sampler** | 空间切块 cosine 融合；音频 passthrough | 480p **更慢**；720p/2K 显存刚需 | C 对 864×480 |

细节已写在 remaining-quality-speed.md §2。

#### Latent Upscaler — **C 质量默认（480p 金线）**

- https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler
- 低分一采 + 345M 3D CNN + 高分短二采。**不跳过目标分辨率 DiT。**
- 对「原生 720p 太贵」有意义；对已经 480p 8 步的金线会牺牲小脸。专章见 remaining-quality-speed.md §3。

#### 分辨率一采二采（通用）— **B 换产品定义**

- 一采 608×352×4 + 二采 480p×3 算术上可能略快，但 step1 建图付两次，小脸在一采就没了。

### 2.5 融合 / 显存 / 其它 H3 节点

#### Sol-Engine 无损核融合 — **A 部分已做；AdaLN fusion 未做**

- fused RMS-AdaLN、QKNorm+RoPE、QKV merge、GEMM+GELU、compile。
- 本仓已有 fused QKV 路径、INT8 GEMM、cuDNN SDPA、quiet。
- H3 AdaLN 占 13B/33B，Sol-Engine 称这是最大无损刀之一。Saganaki22 fusion：38k×5376 上 scale/shift **1.91×**、gate/add **1.22×**，bit-exact。本仓 p5 乱融更慢，但 **分段 AdaLN Triton/CUDA 融合仍可能是 A 档**（需单独设计，不要 pthread SDPA prebuild）。

#### Optimization Suite 其它 — **多数 Blackwell / 显存**

- https://github.com/ByronLeeeee/ComfyUI-MiniMax-H3-Optimization-Suite
- NVFP4 Fused MLP：Ada unsupported。
- Low-Memory Sage2：显存 −566MiB，速度 −2%。本仓不缺这 0.5GB 到要换慢核。
- Long-Sequence VRAM Optimizer：16GB 保活，不是速度。

#### KJNodes H3 Low VRAM Attention / Chunk FF — **C 速度**

- 显存刀。Saganaki22 chunk FF 在 INT8 ConvRot 上 −37% MLP 峰值，吞吐中性。

### 2.6 本仓已做（不要空跑）

| 项 | 状态 |
| --- | --- |
| NVRTC / seq pad 64 / cuDNN 失败缓存 | 合入 |
| TE device-resident **act** | 合入；packed 权仍每 linear H2D |
| INT8 权窗 37 freeze / pregrown | 合入，禁止回退 |
| VAE fp16 GPU upcast | 合入 |
| 默认 audio pack | 合入 |
| quiet / 少刷屏 | 合入 |
| MagCache / reuse / token-reduction / sage2 CLI | 接线，默认关 |
| `--init-latent` | 二采钩子已有 |

**不要再空跑：** NVRTC/seq pad、cuDNN 失败缓存、TE device-resident act、VAE fp16 GPU upcast、window freeze、pthread SDPA prebuild。

---

## 3. 分档总表

### A. 无损或几乎无损，尚未做或未做完

| ID | 方案 | 预期 | 来源 |
| --- | --- | --- | --- |
| A1 | TE NVFP4 packed 常驻 GPU | TE 16s → 8–12s | remaining / lightwan |
| A2 | VAE decode 核融合 | VAE 再砍 1–3s | remaining |
| A3 | 有声 encode/decode 与视频 overlap | av8 109s 向 90s 靠 | remaining |
| A4 | H3 分段 AdaLN / gate 融合（bit-exact） | DiT 可能 1–3s；p5 反例说明要极小心 | Sol-Engine pillar 1；Saganaki22 fusion |
| A5 | 守住 INT8 窗 37 | 守住 50.5s | 已做 |

合计乐观仍是 **数秒到十几秒**，到不了 40s。

### B. 可测的质量换速度（必须 cafe A/B，默认关）

| ID | 方案 | 质量风险 | 对 DiT 50s |
| --- | --- | --- | --- |
| **B-Sol** | Sol-Attn（bf16，sink=exact_kv，start=0.2） | 小脸、细部；人脸终稿有人停用 | **已测 CUDA Sol 351ms vs cuDNN 47ms（7.5× 慢）**。停止；不进 generate 默认 |
| B1 | 更保守 MagCache / FBCache | 8-step 几乎必漂 | 已测 0.08 FAIL |
| B2 | `--reuse N` | 色块/轨迹 | CLI 已有 |
| B3 | Distance Attention | 不加速；可能改善小脸 | 质量刀 |
| B4 | 低分一采 + Upscaler + 短二采 | 小脸/口型 | 可能墙钟略降，质量先碎 |
| B5 | CAB 少 NFE | 8 步再减 ≈ 假 4 步 | 不看好 |
| B6 | Prompt Relay | 叙事，非速度 | — |
| B7 | cache-dit / SenCache | 同 skip 家族 | 实现重 |

### C. 已证伪或不适用

| 项 | 为什么死 |
| --- | --- |
| MagCache 0.08 当质量 | 79s，MAE 52–68 |
| 未蒸馏 4 步 / LightWan 80× 直接搬 | 没有 H3 4-step 权；80× 主力在蒸馏 |
| 20 步当更清 | av20 140s |
| W4A8 DiT 当主题 | 主题崩 |
| NVFP4 DiT / Sage3 / FA3 | 4090=Ada |
| 默认 Sage2 | 本仓核比 cuDNN 慢数量级 |
| **Sol-Attn 赢 cuDNN / 进质量默认** | **§1.10：CUDA Sol 351ms vs cuDNN 47ms**；Triton 侧车禁止 |
| 稀疏 90% / KingGore hard-mask | 伤脸；hard-mask L2 爆炸 |
| TinyVAE / 关 deblock / `--frames 17` | waffle / 假 5s |
| cfg=5 / 多卡 CFG 并行 | 过曝；单卡 cfg=1 无第二路 |
| EasyCache 当有声默认 | #15326 音频 |
| Tiled Sampler 加速 480p | 乘 DiT 次数 |
| Sigma Refiner / Dynamic CFG 当加速 | 加步或加 CFG |
| TensorRT / 默认 CUDA graph | 契约冲突 / 已更慢 |
| Python/Comfy 生成栈 | 架构禁止 |
| p5 cuDNN pthread prebuild | 90.5→98.8s |

---

## 4. 建议顺序（仍不改代码）

1. **A1 TE packed 常驻** — 唯一还没做、几乎无损、能量级可见。
2. **A3 有声路径** — 产品墙钟是 109s 不是 90.5s。
3. **Sol-Attn 孤立 kernel 已测完并输掉**（§1.10：351ms vs cuDNN 47ms）。默认保持 cuDNN；不要为 Sol 开质量路径。Triton 侧车仍禁止。
4. **小脸**走 Distance Attention 类感受野（B3），不要用 Sol/超分幻想补 20px 脸。
5. **不要**把 MagCache、EasyCache、Latent Upscaler、Tiled、Refiner、20 步、W4A8 接进质量默认。
6. 40–50s 仍然只有跨红线：官方 4-step 蒸馏权 / 激进 skip / 激进稀疏 / TinyVAE / 假时长。本轮继续不承诺。

---

## 5. URL 清单

**Sol-Attn / Sol-Engine**

- https://github.com/kijai/ComfyUI-SolAttn_triton
- https://github.com/Saganaki22/ComfyUI-sol-attn
- https://nvlabs.github.io/Sana/Sol-Attn/
- https://arxiv.org/abs/2607.24027
- https://arxiv.org/abs/2606.23743
- https://nvlabs.github.io/Sana/Sol-Engine/H3-OnDevice/
- https://github.com/NVlabs/Sana/tree/sol-engine
- https://github.com/sumeetprashant/ComfyUI-SolAttn
- https://github.com/KingGore/ComfyUI_sol-attn_Blackwell

**Sage / Comfy H3**

- https://github.com/thu-ml/SageAttention
- https://github.com/kijai/ComfyUI-KJNodes
- https://docs.comfy.org/tutorials/video/minimax/minimax-h3
- https://github.com/ByronLeeeee/ComfyUI-MiniMax-H3-Optimization-Suite
- https://xhinker.medium.com/minimax-h3-everything-i-know-after-days-of-testing-what-works-what-not-648e4df78128

**Cache**

- https://arxiv.org/abs/2411.19108 （TeaCache）
- https://arxiv.org/abs/2506.09045 （MagCache）
- https://github.com/Comfy-Org/ComfyUI/issues/15326 （EasyCache 伤 H3 音频）
- https://github.com/vipshop/cache-dit
- https://github.com/vita-epfl/SenCache
- https://github.com/ModelTC/LightX2V/issues/579

**YCNodes / Upscaler / LightWan**

- https://github.com/yichengup/ComfyUI-YCNodes-MiniMax-H3
- https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler
- https://huggingface.co/lightx2v/LightWan2.2-A14B
- https://github.com/ModelTC/lightx2v

**其它**

- https://github.com/mit-han-lab/radial-attention
- https://arxiv.org/abs/2605.16736 （CAB）
- 本仓金线：`out/verify_480p_quality_full/t2va_complex_5s_cafe_p3/`
- 用户已存 Sol README：`/Users/digenman/.cursor/projects/Volumes-Ben-Digen-code-h3-c/uploads/ComfyUI-sol-attn-2.md`

---

## 6. 与其它计划的关系

| 文件 | 关系 |
| --- | --- |
| [2026-08-21-remaining-quality-speed.md](./2026-08-21-remaining-quality-speed.md) | 质量菜单；本文把 Sol-Attn 从笼统「稀疏 attn」里拆出来，并交叉链接 |
| [2026-08-21-lightwan-h3-speed.md](./2026-08-21-lightwan-h3-speed.md) | LightWan 80× 为何搬不来；p0–p5 墙钟 |
| `docs/nvidia_optimization_plan.md` §9 | 2026-08 的 Sage/MagCache/TRT 菜单；本文用 cafe_p3 数字刷新 |
