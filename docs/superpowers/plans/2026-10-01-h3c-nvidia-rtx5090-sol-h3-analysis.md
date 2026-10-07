# NVIDIA Sol-Engine 两条 5090 路径：RTX5090 vs Sol-H3-RTX5090

> **日期：** 2026-10-01。调研 + §8 施工（238 真 `cute_sm120`）。未 commit。产品默认仍 cudnn。  
> **一手树：** [NVlabs/Sana `sol-engine`](https://github.com/NVlabs/Sana/tree/sol-engine)  
> - [models/minimax_h3/RTX5090](https://github.com/NVlabs/Sana/tree/sol-engine/models/minimax_h3/RTX5090)  
> - [models/minimax_h3/Sol-H3-RTX5090](https://github.com/NVlabs/Sana/tree/sol-engine/models/minimax_h3/Sol-H3-RTX5090)  
> **产品页：** [H3-OnDevice（4.52×）](https://nvlabs.github.io/Sana/Sol-Engine/H3-OnDevice/) · [Sol-H3 主页（8×B300）](https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3/)  
> **论文：** [Sol Video Inference Engine arXiv:2606.23743](https://arxiv.org/abs/2606.23743) · [Sol-Attn arXiv:2607.24027](https://arxiv.org/abs/2607.24027)  
> **本仓对照：** [5090 fine-opt](./2026-09-30-h3c-5090-fine-opt.md) · [5090 opt-dev §19](./2026-09-30-h3c-5090-opt-dev.md) · [turbo-explainer](./2026-09-13-h3c-turbo-explainer.md) · [TaoMate](./2026-09-13-h3c-taomate-h3-analysis.md) · [few-step quality](./2026-09-11-h3-few-step-quality.md) · [accel survey deepdive](./2026-09-11-h3c-accel-survey-tech-deepdive.md)  
> **合同纪律：** 外部倍数 / 秒数 **禁止**换算到本仓金线 66s/79s、KPI-A ≤30s、KPI-B ≤27s。两条 NVIDIA 路径彼此也 **不可横比墙钟**（合同不同）。

**一句话：** 同一张 1×RTX 5090 上，NVIDIA 放了两条互不替代的产品线——**RTX5090** 保 50-step BF16 原 H3 轨迹做近无损加速（231s / 4.52×）；**Sol-H3-RTX5090** 换配方（FastH3 4-step 草稿 + LTX 精修）压产品延迟（正式请求 39.8s）。对本仓：前者对标「金线轨迹语义下的融合核 / Sol-Attn / 缓存边界」；后者对标「KPI-B 两段式工程」，但成片走 LTX 而非官方 H3-VisualVAE。

---

## 0. 尺子（先钉死再谈秒）

| 谁 | 硬件 | 画布 / 时长 | 采样合同 | 计时口径 | 公布墙钟 |
| --- | --- | --- | --- | --- | ---: |
| **RTX5090 fullopt** | 1×5090 32GB | **1344×768** · 5s · 124f · 24fps | **50-step BF16 stock H3** · H3 VAE | warm · vs matching dense baseline | **231.2 s**（基线 1045.4 → **4.52×**） |
| **Sol-H3-RTX5090** | 1×5090 32GB | **1344×768** · 121f · 24fps | **4× H3 draft @672×384** → ×2+adapter → **LTX×3** · Conv VAE · H3 音频 | 正式请求 entry→MP4；**排除 warmup** | **39.80 s**（单次，2026-09-23） |
| 本仓 KPI-A | 1×5090 | **864×480** · 5s | **turbo4** · 官方 VAE | 热 e2e 全含 | 目标 ≤30s |
| 本仓 KPI-B | 1×5090 | **1376×768** · 5s | TaoMate/Comfy **2-stage H3→H3** · 官方 VAE | 热 e2e 全含 | 目标 ≤27s；现状 ~52–53s |
| Sol-H3 8×B300（旁注） | 8×B300 NVLink | 768p · 5s | FastH3 等 | warm；**不含 MP4 封装** | 全片 **1.653 s**（硬件代差，不可对 5090） |

**禁止混报：**

1. 把 4.52× 写成相对本仓 INT8 8-step 的倍数。  
2. 把 Sol-H3 39.8s 与 RTX5090 231s 并排放「谁更快」而不注明合同。  
3. 把 8×B300 的 1.653s 写成 5090 可达。  
4. 把 Sol-H3 的 warmup ~375s 折进正式 39.8s。

---

## 1. 总览对照

| 维度 | RTX5090 | Sol-H3-RTX5090 |
| --- | --- | --- |
| **产品合同** | 保 50-step BF16 H3 轨迹，近无损 | 换配方：H3 草稿 + LTX 精修 |
| **输出** | 1344×768 · ~124f · H3 video+audio VAE | 1344×768 · 121f · Conv VAE + 原 H3 音频 AAC |
| **热墙钟** | 231.2 s（4.52× vs dense 1045s） | 39.8 s（单次正式；warmup 另计） |
| **冷 / 启动税** | layerwise offload + 大 host RAM | warmup ~375 s；正式才计 39.8 s |
| **主干技术** | Sol-Attn + TeaCache + 融合核 | FastH3 VSA LoRA + W8A8 + LTX Sol |
| **运行时底座** | SGLang MiniMax-H3 + 薄 adapter | FastVideo stage1 + LTX stage2 多进程 |
| **显存策略** | component layerwise；VAE 解码时常驻 | CPU offload；H3 一次一块 body block |
| **质量旋钮** | 推理时近似，原 checkpoint 不动 | 蒸馏学生 + 跨模型 latent 迁移 |
| **对 h3.c** | 50-step 缓存红利 ≠ 8/4-step 可搬 | 2-stage 思路贴近 KPI-B；LTX 非本仓合同 |

---

## 2. RTX5090：保轨迹加速栈

### 2.1 它是什么

[`models/minimax_h3/RTX5090`](https://github.com/NVlabs/Sana/tree/sol-engine/models/minimax_h3/RTX5090) 跑 **released BF16 FL2VA** checkpoint：33B DiT + Qwen3-VL + VAEs，靠 **SGLang layerwise component offload** 塞进 32 GiB——任意时刻只有活跃组件在 GPU。

入口配置钉死在 [`config/minimax_h3/rtx5090_fullopt.toml`](https://github.com/NVlabs/Sana/blob/sol-engine/config/minimax_h3/rtx5090_fullopt.toml)：

```text
SOL_ATTN_TAU=1.0 · THRESH=diag · FIRST_DENSE_STEPS=10 · FIRST_DENSE_LAYERS=2
H3_TEACACHE_ENABLED=1 · THRESHOLD=0.10 · RETAIN=5 · COOLDOWN=1 · NUM_FORWARDS=49
H3_FULL_VAE_AFTER_DENOISE=1 · DTYPE=bfloat16
steps=50 · 1344×768 · duration=5 · seed=0
```

对照臂：`rtx5090_dense.toml`（同合同、关优化）。运行环境：PyTorch 2.11 + CUDA 13.0 + Triton 3.6 + pinned SGLang。

### 2.2 加速阶梯（官方 H3-OnDevice）

| 阶段 | 墙钟 (s) | 累计倍数 |
| --- | ---: | ---: |
| Baseline unmodified | 1045.4 | 1.00× |
| + Kernel / resident BF16 VAE | 971.6 | 1.08× |
| + Sol-Attn | 708.3 | 1.48× |
| + Cross-step cache（TeaCache） | **231.2** | **4.52×** |

**最大单步增益来自跨步缓存，不是 Sol-Attn。** Spark 480p 同族阶梯是 710→512→451→181（3.92×）；5090 像素多 2.6×，两机墙钟 **不可画在同一横轴**。

### 2.3 三根柱子（技术）

#### Pillar 1 — Lossless kernels

H3 约 13B 在 AdaLN 分支，eager PyTorch 大量 HBM 往返。融合表（官方页）：

| 融合 | 消掉什么 |
| --- | --- |
| fused RMS-AdaLN | modulation 分支 HBM 往返（H3 最大单点） |
| fused QKNorm + RoPE | Q/K 两趟读写 + 3D MM-RoPE 索引 |
| QKV merge | 三次瘦 GEMM → 一次宽 GEMM |
| GEMM + GELU | MLP 激活级 HBM 往返 |
| regional `torch.compile` | 残余 launch / elementwise |

5090 上 kernel 仅 **1.08×**——长序列注意力更吃紧，kernel 红利小于 Spark（1.39×）。

#### Pillar 2 — TeaCache（源码钉死；主页文案有偏差）

README / toml / [`teacache.py`](https://github.com/NVlabs/Sana/blob/sol-engine/models/minimax_h3/RTX5090/teacache.py) 全是 **TeaCache**：

- 探 `modulated_input` 相对 L1，多项式 rescale 后累加；≥ threshold 才重算整栈。  
- 前 `retain_steps=5` 强制 compute；末 `cooldown_steps=1` 强制 compute。  
- reuse 时 `hidden += residual`，跳过中间 blocks。

产品页写成 **First Block Cache**。**以仓库 toml/源码为准：TeaCache。** 两者同属「跨步 residual 复用」族，但探针与门控实现不同——迁移时勿混名。

#### Pillar 3 — Sol-Attn

训练无关稀疏注意力（[arXiv:2607.24027](https://arxiv.org/abs/2607.24027)）：

- Query 自适应阈值：τᵢ = μᵢ + βσᵢ（配置标量 τ/β）。  
- 在线 softmax 路由，不物化完整 proxy map。  
- 丢弃块用 Taylor 补偿，高稀疏时比硬 top-k 更缓降。  
- 配方：前 10 step + 前 2 layer **dense**；其余 diag thresholding + prefix KV sink。

### 2.4 流水线与代码落点

`pipeline.py` 是 **SGLang `MiniMaxH3Pipeline` 子类**，几乎只改内存阶段：

1. Denoise：compile warmup 时若显式要求 dit/transformer layerwise，则不要被父类卸掉。  
2. Decode：`H3_FULL_VAE_AFTER_DENOISE` → 整模 VAE `disable_offload()` 常驻一次 decode，再 `enable_offload()`；可选 tile batch 上限。

阶段序仍是：Text / Visual / Audio encode → Latent+Timestep → DiT denoise → Video+Audio VAE decode。

### 2.5 环境与限制

- 权重：`H3_MODEL_PATH` 指 released BF16 FL2VA。  
- Placement：1×5090 + **足够 host 内存**做 layerwise。  
- Agent-Native：Sol Engine 按 technique family 搜索再组合（kernel / sparse / cache），人眼签成片。  
- 质量：页上 LPIPS 示意近无损；**不是** bit-exact。

---

## 3. Sol-H3-RTX5090：换配方系统工程

### 3.1 它是什么

[`Sol-H3-RTX5090`](https://github.com/NVlabs/Sana/tree/sol-engine/models/minimax_h3/Sol-H3-RTX5090) 是 **Sol-H3 Spark** 的 Linux x86_64 / 5090 专用入口。CLI/API **默认 CPU offload**。目标不是加速 50-step H3，而是用两段式把 **热正式请求**压到几十秒。

### 3.2 T2VA 配方（README 钉死）

| 组件 | 固定配置 |
| --- | --- |
| Prompt | **NVFP4 AWQ Qwen**；每请求加载，H3 前关闭 |
| Draft | **W8A8 MiniMax-H3 + FastH3 VSA DataFree LoRA**；**4 updates** @ **672×384×124** |
| Draft attn | VSA **90%** video sparsity + **cuDNN BSA** |
| Latent transfer | 学习 **H3 ×2 upscaler** → **H3-to-LTX adapter** |
| Refine | **LTX BF16 ×3** + Triton Sol attention + 固定 generic AV context |
| Output | **Original Conv VAE**；1344×768×121 @24fps；**原 H3 音频** stereo AAC |

关键语义：

- H3 body blocks（含非持久 FP8 buffer）默认在 CPU；**一次搬一块**上 GPU。  
- **旁路 H3 视频解码**；归一化 H3 latent + PCM 经本地 CPU tensor 文件过界。  
- 视频阶段 **串行**（`overlap_stages=False`）。  
- FL2VA / Ref2VA：有 CPU 合同覆盖，**未在本 offload profile 上 GPU 验证**。

### 3.3 计时与 validation（一手）

[`docs/validation.md`](https://github.com/NVlabs/Sana/blob/sol-engine/models/minimax_h3/Sol-H3-RTX5090/docs/validation.md)（2026-09-23，mountain-lake，seed 42）：

| 测量 | 结果 |
| --- | ---: |
| 正式请求 entry → 完成 MP4 | **39.796751168 s** |
| 初始化 + 全量 warmup | **375.143728772 s** |
| H3 formal generation/capture | **11.224092651 s** |
| LTX formal stage | **12.000252239 s** |
| GPU 峰值（每 2s 采样） | **21583 MiB** |
| 输出 | H.264 1344×768 · 121f · 24fps · AAC 32kHz stereo |

配方演进表（README，相对自家，非相对 50-step）：

| Method | E2E |
| --- | ---: |
| 4-step LoRA | 97 s |
| 4-step LoRA + quant | 51 s |
| Sol-H3 | **39.8 s** |

限制（官方自述）：单次正式、非多跑统计；采样可能漏峰值；188 GiB host 是观测值非最小；fresh install / FL2VA·Ref2VA GPU / 与 Spark 感知对齐 **未验证**。

### 3.4 运行时架构

`runtime/pipeline.py`：**三 worker 进程组**（stage1 / qwen / stage2），JSON 通道：

```text
start():
  stage1 起活 → 临时 Qwen warmup → stage1 warmup → stage2 prepare+run → release_idle_cache
  （warmup 墙钟写入 results，不进正式 e2e）

generate()（CPU offload 默认）:
  每请求新建 Qwen → conditioning → 关 Qwen
  → stage1.run（H3 draft）
  → stage2.prepare → stage2.run（upscale/adapter/LTX/VAE/mux）
  父时钟：request_start → final_mp4_complete
```

环境（[`docs/setup.md`](https://github.com/NVlabs/Sana/blob/sol-engine/models/minimax_h3/Sol-H3-RTX5090/docs/setup.md)）：

| Env | 观测栈 |
| --- | --- |
| H3 / Qwen | Py 3.12 · Torch 2.12+cu130 · Triton 3.7 · patched FastVideo · cuDNN frontend · ComfyUI 0.30 |
| LTX (stage2) | Py 3.13 · Torch 2.13+cu132 · Triton 3.7.1 · ltx-core/pipelines/kernels |
| FA4 | 独立 namespace + CuTe 路径覆盖 |

第三方钉子见 [`THIRD_PARTY_NOTICES.md`](https://github.com/NVlabs/Sana/blob/sol-engine/models/minimax_h3/Sol-H3-RTX5090/THIRD_PARTY_NOTICES.md)：FastVideo、FA4、ComfyUI(GPL)、LTX Community License、MiniMax/VSA LoRA、LightX2V Ref2VA LoRA、**H3 latent upscaler 许可证未声明**、H3-to-LTX adapter 权重另下。

### 3.5 与 8×B300 Sol-H3 的关系

[Sol-H3 主页](https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3/) 是 **8×B300 + NVLink** 的 speed-of-light 叙事（5s 全片 1.653s，口径不含封装）。5090 包是 **同品牌两段式思想的桌面 offload 移植**，不是同一硬件合同。本仓 [turbo-explainer](./2026-09-13-h3c-turbo-explainer.md) 已写清：B300 数字相对 5090 **硬件代差 + 口径差**，算法复刻填不满。

---

## 4. 加速来源本质差异

| | RTX5090 | Sol-H3-RTX5090 |
| --- | --- | --- |
| **少什么** | 少 FLOPs（稀疏 attn + 跳步） | 少 NFE + 换更小精修模型 |
| **轨迹** | 仍在 stock 50-step ODE 附近 | 已离开 H3 50/8-step 金线；LTX 另一条 |
| **倍数分母** | 匹配的 **dense 50-step BF16** | 自家 **4-step LoRA**（97s） |
| **哲学** | inference-time only，不动权 | 蒸馏 LoRA + 学习 upscaler/adapter |

这与本仓已有纪律同构（few-step / turbo-explainer）：

> 缓存吃的是「步数冗余」；蒸馏把冗余先吃了。  
> 公开 × 的分母几乎总是「50-step 全精度慢基线」。

---

## 5. 对 h3.c 的可迁移 / 不可迁移

| 技术点 | 可搬？ | 理由 |
| --- | --- | --- |
| TeaCache / FBCache @ 50-step | **概念** | 本仓 8/4-step 冗余已被蒸馏榨干；turbo4 已拒绝 cache 叠加（见 turbo-explainer） |
| Sol-Attn sm_120 | **可对照** | 本仓已有 sol 路径；须 MAE 门；勿把 50-step 近无损当 4-step 无损 |
| AdaLN / QKV / RoPE 融合 | **高优先** | 与 RTX5090 pillar1 同族；[5090 fine-opt](./2026-09-30-h3c-5090-fine-opt.md) 已列 |
| VAE decode 后整模常驻 | **可借鉴** | DiT→VAE 大切换税；语义近 `H3_FULL_VAE_AFTER_DENOISE` |
| FastH3 4-step VSA LoRA | **SKU 隔离** | 换学生权；禁 merge 进 INT8 金线；对标 turbo4 |
| H3→LTX 两段精修 | **合同外** | KPI-B 是 H3→H3（TaoMate/LBH），不是 LTX 成片 |
| layerwise / block CPU offload | **低优先** | 本仓 INT8 权窗已解决 32GB 装载；offload「从不能跑到能跑」不是主矛盾 |
| 4.52× / 39.8s / 1.653s | **禁换算** | 分母与合同不同 |

### 5.1 选型一句话

- 要「**原 H3 50-step 画质、桌面 5090 能跑**」→ 看 **RTX5090**（~4 分钟级）。  
- 要「**产品级 ~40s 热延迟、接受 LTX 精修轨迹**」→ 看 **Sol-H3-RTX5090**。  
- 本仓若守 **官方 H3-VisualVAE** 金线 / KPI-A·B：吸收前者的融合核与稀疏对照、后者的两段式调度与计时诚实（warmup 单列），**不要照搬任一条的秒数叙事**。

### 5.2 与本仓 KPI 的位置感（禁止当倍率）

| 本仓尺子 | 更近的 NVIDIA 参照 | 差距性质 |
| --- | --- | --- |
| KPI-A turbo4 @480p ≤30s | Sol-H3 的「少 NFE + quant」思想；非其 768p 秒数 | 分辨率/栈不同；可对标工程，不对标墙钟 |
| KPI-B 2-stage @768p ≤27s | Sol-H3 两段式 + TaoMate 用户 27s 叙事 | 成片模型不同（H3 vs LTX）；本仓现状仍 ~52–53s |
| 金线 8-step INT8 | RTX5090 **不是**对手——对方是 50-step BF16 近无损 | 轨迹类别错误 |

---

## 6. 一手文件地图（便于复查）

### RTX5090

| 路径 | 角色 |
| --- | --- |
| `models/minimax_h3/RTX5090/README.md` | 合同与 4.52× 表 |
| `models/minimax_h3/RTX5090/pipeline.py` | VAE 常驻 / denoise offload 覆盖 |
| `models/minimax_h3/RTX5090/teacache.py` | TeaCache 控制器 |
| `models/minimax_h3/RTX5090/adapter.py` | SGLang 接线 / 事件 |
| `config/minimax_h3/rtx5090_fullopt.toml` | 环境变量钉子 |
| `config/minimax_h3/rtx5090_dense.toml` | 对照臂 |

### Sol-H3-RTX5090

| 路径 | 角色 |
| --- | --- |
| `.../Sol-H3-RTX5090/README.md` | 配方与 39.8s |
| `.../docs/validation.md` | 正式拆账与限制 |
| `.../docs/setup.md` | 双环境 / paths-5090 |
| `.../runtime/pipeline.py` | 三 worker 生命周期与时钟 |
| `.../runtime/stage1.py` / `stage2.py` | H3 draft / LTX refine |
| `.../runtime/offload.py` | CPU offload |
| `.../THIRD_PARTY_NOTICES.md` | 许可证与权重钉 |
| `.../validation/rtx5090-t2va.json` | 消毒后的测量收据 |

---

## 8. 施工（2026-10-01 · 238 · 真 Sol-Attn `cute_sm120`）

> 本拍把 §5「Sol-Attn sm_120 可对照」落成接线 + 探测 + 真核。**禁止**声称 KPI-B ≤27。产品默认仍 `--sdpa-backend cudnn`。未换 TAE/LTX。未装假核。未 commit。未拆 `generate.c`（仅 lift/VAE 前调用 `h3_cuda_sol_cute_drop_pool`，对标已有 sage3 drop）。

### 8.1 本拍改了什么

| 项 | 落点 |
| --- | --- |
| 官方 CuTe 名表 | `h3_sdpa_sol_cute_is_official`：`cute_sm89/90/100/120`；`triton`/`nested_loop` 拒绝 |
| 去掉 4090 机器默认路径 | `h3_sol_cute_embed.{c,py}` / bench：无 `/home/haiyang`；`H3_SOL_CUTE_TOOLS` |
| 5090 真核路径 | embed 走官方 `sol_attn()` → `sm120.make_kernel`，不再硬钉 `cute_sm89` |
| 库存盘 | `tools/h3_sol_sage3_inventory.py`；`h3_cuda_info` 打 ROOT env |
| 假 ROOT | `h3_768p_eval.sh` / `h3_dp_generate.sh` 不再把 `H3_SAGE3_ROOT` 指到 `build-cuda-sm120` |
| 插件 ABI | 238 上 `libh3_sol_cute.so` 链 conda **Python 3.11**（系统 3.10 看不到 torch/cutlass） |
| 真核树 | `/mnt/nvme1n1/workspace/Sana-sol-engine` = `NVlabs/Sana` `sol-engine` sparse checkout `techniques/sparse_backends` |
| 32GB lift | `h3_cuda_sol_cute_drop_pool`：卸低分 QKVO 再进高分 |

### 8.2 238 有没有真核

| 后端 | 结论 |
| --- | --- |
| **sol / cute_sm120** | **有真核。** `cutlass.cute` + `cuda.bindings` + Triton 3.7.1 `TensorDescriptor` 已在 conda；官方 `get_sol_attn_backend("cuda")` → **`cute_sm120`**；`sol_attn.sm120.make_kernel` 存在；preprocess **无** `_prepare_torch`。 |
| **sage3** | **无核。** 无 `sageattn3` / `fp4attn_cuda`。只有 `sageattention` 1.x（**不是** Sage3）。不要报 sage3 hit。拿核：`thu-ml/SageAttention` 的 `sageattention3_blackwell`（setup.py 认 sm_120）。 |

### 8.3 测了什么（KPI 只取 `phase_ms total=`）

**KPI-A 产品 cudnn：** 本拍未重跑 `--repeat 30`。仍用 opt-dev §8.9：热 P50 **21.704s**（≤30 PASS）。

**KPI-A opt-in sol_cute**（`--sdpa-backend sol_cute`，`H3_SOL_FIRST_DENSE_*=0`，官方 VAE，cafe ids，`--repeat 2`）`out/sol_sm120_kpia`：

| clip | `dit` | **`total`** | `vae_blocks` | hits | waffle b16 | checksum |
| --- | ---: | ---: | ---: | --- | --- | --- |
| 冷 1 | 16.610s | 27.820s | 3.601s | 200/200 | 0.59–2.66 | `0xa2c155a8da0970a7` |
| **热 2** | **10.922s** | **17.244s** | 3.401s | **400/400** | 0.609–0.698 | `0xf349eaad40a81da5` |

日志：`backend=cute_sm120` `kernel=sol_attn.sm120.make_kernel` `shape=(1, 15973, 56, 128)` `nested_loop=never` `fallback=0`。`av_sync: PASS`。成片 `out/review-2026-10-01/kpi-a-sol/`。checksum **不同于** cudnn（稀疏 attn，预期）。**不当产品默认**（分析 §5：勿把 50-step 近无损当 4-step 无损）。热 total 17.2s 相对 cudnn P50 21.7s 是 **opt-in 对照**，不是改口径。

**KPI-B 产品 cudnn：** 未重跑。仍 §8.8/§8.9：干净热 **56.143s** / 30-repeat P50 **56.470s**。**不能**声称 ≤27。§8.8 算术硬墙对 **cudnn 产品路径** 仍成立。

**KPI-B opt-in sol_cute（2026-10-04 续拍，WS-first + drop QKVO + 混合臂已编入二进制）：**

- **全 sol（高分也 sol，`H3_SOL_MAX_SEQ=0`）：** 低分 `seq=9768` 与高分 `seq=38676` 均真进核（`cute_sm120`，`hits=178/178`、`fallback=0`、`dense_skip=0`），`w_i8` OOM **已修**（WS-first：先卸 INT8→预生长高分 WS→sol warmup 后 drop QKVO→trim→refit 窗 34）。但高分 step 3 在 `adaln_dev kernel` 报 `out of memory`（前序 malloc 失败冒泡到 kernel launch），高分 INT8 窗 34 + sol 高分 QKVO ~5.5GiB + adaln/GEMM 激活在 32GB 上超载。进程 **exit 1**，无高分成片、无热 `total`。window 仍 34（**未减窗**）。这是 **32GB 显存墙**，不是加载 bug。
- **混合臂（低分 sol + 高分 cudnn，`H3_SOL_MAX_SEQ=16000`）：** 稳定出片。`sol_cute skip seq=38676 max_seq=16000 (hires cudnn mixed arm)`；低分 sol hits=300/300、高分走 cudnn（`cudnn want=200 hits=200`）。`av_sync: PASS` 两 clip。成片 `out/review-2026-10-01/kpi-b-mixed/`。

| clip | `dit` | **`total`** | `vae_blocks` | waffle b16 | latent checksum |
| --- | ---: | ---: | ---: | --- | --- |
| 冷 1 | 41.335s | 72.794s | 10.534s | 1.334–2.980 | `0x21db2f9196afaf24` |
| **热 2** | **37.211s** | **54.069s** | 9.862s | **0.753–0.874** | `0x856a83558551ef38` |

混合臂热 total **54.069s** 相对产品 cudnn 56.143s 快 ~2.1s（低分 sol 红利），但高分仍 cudnn，**不是 27s**。checksum 与产品 cudnn（`0xebdaf3bfc36beb75`）不同（稀疏 attn + 混合 backend，预期）。**不当产品默认**（高分仍 cudnn，且 sol 只覆盖低分）。

**KPI-B opt-in sol_cute（2026-10-01 首拍，WS-first 之前）：** 低分 `seq=10756` 真进核；高分 `seq=39172` embed 成功后 `cudaMalloc w_i8: out of memory`。进程 exit 1，无热 total。window 仍 32。

### 8.4 是否仍硬墙

- **产品 KPI-B ≤27：** 仍硬墙（cudnn 高分 SDPA + 官方 VAE 已 >27）。本拍 **没有** 把 sol 接到 KPI-B 高分成片。
- **sol 在 480p：** 真 hit，热 e2e 可跑。
- **sol 在 768p 2-stage 全 sol：** 真核已 init（高分 174 进核 fallback=0），但 **32GB 显存墙没过**（见 §8.5：native INT8 窗 + 激活占 28.8GiB，sol fwd 连 530MiB 都挤不出）。QKVO「用完即卸」只释放 torch 侧 2GiB，救不了。**在不减窗红线下判定不可行/不划算，退回混合臂。**
- **sol 在 768p 2-stage 混合臂（低分 sol + 高分 cudnn）：** 稳定出片，热 **54.069s**（上拍）。本拍用新二进制复测对照（见 §8.5）。
- **下一刀：** 全 sol 过 32GB 只剩「压 INT8 窗到 24 以下（**减窗，禁**）」或「高分 QKVO/INT8 挪 host（offload，会让高分 sol 比 cudnn 慢，改动面大）」。都不是打 GEMM、overlap、假 sage3、TAE/LTX。**现实路径是混合臂**（低分 sol 红利 ~2s）+ 在调度层找 VAE/激活的省显存或提速（对应 §10.1 第 2 条并行/批量 VAE 解码方向）。

### 8.5 本拍（2026-10-04 续拍 2）：高分 sol QKVO「用完即卸 + 按需重分配」

> 目标：让 KPI-B 全 sol（高分也走 `cute_sm120`）在 32GB 上出片。约束：**INT8 窗不得压到 24 以下**（不减窗）、不 overlap、不假 sage3、不 TAE/LTX、不打 GEMM tile、不拆 `generate.c`（仅 lift 前已存在的 `h3_cuda_sol_cute_drop_pool` 调用保留）。产品默认仍 `--sdpa-backend cudnn`。

**OOM 归因（本拍定位，非加载 bug）：** 高分 sol QKVO ~5.5GiB 由 `h3_sol_cute_init` 一次性经 torch caching allocator 分配，且 `sol_attn()` 每次 fwd 还会临时物化 `output/lse/kc/vc/threshold` 等张量留在 torch pool。叠加高分 INT8 窗 34 + adaln/GEMM 激活后在 32GB 超载，在高分 step 3 `adaln_dev` kernel 冒泡 OOM。低分 QKVO（不同 seq）虽在 lift 前 `drop_pool`，但 torch caching allocator 缓存段不还驱动，须 `trim_cache` 才释放。

**方案（env 门控 `H3_SOL_HIRES_QKVO`，默认 off，零行为变化保护混合臂/480p）：**

| 落点 | 改动 |
| --- | --- |
| `h3_gpu_cuda.cu` | `h3_sol_cute_hires_qkvo_on()` 读 `H3_SOL_HIRES_QKVO`；`h3_sol_cute_is_hires_seq(seq)` 以 seq>20000 判定高分阶段（480p ~15973 / 768p packed ~38676） |
| `h3_gpu_cuda.cu` fwd 成功尾 | 高分 sol 且已冻结每步层数后，**末层 hit 后 `h3_cuda_sol_cute_drop_pool()` + `h3_cuda_sol_cute_trim_cache()`**，把 QKVO + torch pool 还驱动，让同 step 后续 adaln/GEMM 激活有 headroom；下一 step 首 hit 经 caching allocator 重分配 |
| `h3_gpu_cuda.cu` `set_step` | 步边界 `h3_sol_cute_freeze_layers_per_step()`：首 hires step 只学习不释放，步末冻结 `g_sol_cute_layers_per_step`；低分 step 不置 `hires_seen`，不污染层数 |
| `tools/h3_238_kpi_b_allsol.sh` | 238 KPI-B 全 sol 驱动：`H3_SOL_HIRES_QKVO=1`、`H3_SOL_MAX_SEQ=0`、window=34（不减窗）、`H3_SOL_WARMUP_SEQ=38676`、taomate-lift 2-stage、官方 VAE |

**实测（238，本拍）：全 sol 仍撞 32GB 显存墙，未出片，退回混合臂。**

两版释放策略都试了，均 OOM：

| 版本 | 策略 | 结果 |
| --- | --- | --- |
| v1 末层释放 | 首 hires step 学习层数、步末冻结、末层 hit 后 drop+trim | `hires chunk 0 step 3 adaln_dev kernel: out of memory`（首 hires step 内就来不及释放，进程 exit 1） |
| v2 每层释放 | 每个 hires sol hit 后 drop+trim（per-layer realloc） | 释放触发（`hires_qkvo: released`×5），但 `sol_cute fwd OOM seq=38676: embed.fwd(): CUDA out of memory. Tried to allocate 530MiB. 进程已占 30.89GiB，仅剩 432MiB`，fwd OOM 后 `skip_from`→该层回落 cudnn，未出片 |

**OOM 根因（钉死，非 QKVO 能解）：** OOM 时进程占 **30.89 GiB**，其中 **torch（PyTorch）仅 2.08 GiB**，**非 torch 的 native 内存 ~28.8 GiB**——即高分 **INT8 窗 30 的 w_i8（native cudaMalloc，~11GiB）+ DiT 激活 workspace + cudnn workspace**。`trim_cache` 只 trim torch caching allocator（2 GiB 级），动不了 native 的 28.8 GiB。释放 sol QKVO（torch 侧）杯水车薪，sol fwd 连 530 MiB 都挤不出。

**结论：全 sol 在 32GB 不可行/不划算（在「不减窗」红线内）。** 过墙只剩两条，都在红线外或收益存疑：
1. **压 INT8 窗到 24 以下**——**涉减窗，禁**（本拍约束）。
2. **高分 QKVO / INT8 窗挪 host（CPU offload）**——host 往返会让高分 sol 反而比 cudnn 慢，且改动面大，非本拍范围。

**每步层数钉死：** 高分 sol 每步 50 hits（`dit_layers=50`，`step_done` step_hits=50），与低分一致；真核 `cute_sm120` 在高分 seq=38676 init 成功（`hits=174 fallback=0` 于 v1 进核段），核是真的，墙在显存不在核。

**环境修复（顺带）：** 238 嵌入式插件 dlopen `libpython3.11` 缺 `PYTHONHOME` 时 `import h3_sol_cute_embed` / `subprocess` 报 `No module named '_posixsubprocess'`（stdlib 的 lib-dynload 找不到）。**必须 `export PYTHONHOME=/home/yongbin/miniconda3/envs/yongbin`** 让嵌入式解释器定位 stdlib，否则 sol 全 fallback。本拍全 sol 真进核即靠此修复。

**混合臂复测（本拍，新二进制，`H3_SOL_MAX_SEQ=16000`，`H3_SOL_HIRES_QKVO=0`，window=34→refit 34）：** 稳定出片，低分 sol 全进核（`hits=300 fallback=0`），高分 `skip seq=38676 → cudnn`。热 clip 2 `phase_ms: dit=40886.5 total=57371.8`（**57.372s**），`vae_blocks=9869.8`，waffle b16 **0.826–1.037**，`av_sync PASS`，latent checksum `0x63987d3e8083e8c1`。本拍 57.372s 与上拍 54.069s 同量级（差 ~3.3s，机器/二进制状态差异），均远非 27s。成片 `out/review-2026-10-01/kpi-b-mixed/kpi-b-mixed_5s_r{1,2}.mp4`。

**三臂对照（热 clip `phase_ms total=`）：**

| 臂 | 高分 SDPA | 热 `dit` | 热 `total` | 出片 | 备注 |
| --- | --- | ---: | ---: | --- | --- |
| 产品 cudnn | cudnn | 39.470s | **56.143s** | 是 | 上拍 §8.9；本拍未重跑 |
| 混合臂（低分 sol） | cudnn | 40.887s | **57.372s** | 是 | 本拍复测；上拍 54.069s |
| 全 sol（高分 sol） | cute_sm120 | — | — | **否（OOM）** | 本拍 §8.5；32GB 显存墙 |

**未 commit。** 高分 QKVO「用完即卸」代码（`H3_SOL_HIRES_QKVO`，默认 off）保留为可选臂，不作为默认；混合臂/480p 不受影响。

### 8.6 本拍（2026-10-04 续拍 3）：VAE tile 批并行（§10.1 第 2 条）实测——无真收益

> 目标：在「不换 VAE、不减窗、不 overlap DiT、不动质量尺」红线下，把 VAE decode 的 tile 批并行度从 4 提高（Comfy `_decode_tile_row` 批更多 tile），对照 §10.1 第 2 条外部参考（7.55s→0.602s，非本仓数字）。产品默认保持 batch=4。

**现状审计（串行度）：** 768p 1376×768 画布 → `6×3=18` 个 304px 空间 tile（overlap≥64）。每行 6 tile 按 `bmax=4` 分 2 批（4+2），3 行共 **6 批串行**；热 clip VAE 分解：`vae=14.75s` 内 `vae_load≈0.30s`（W resident 热载）+ `vae_blocks≈9.87s`（36 个 transformer block 逐 chunk GEMM+SDPA）+ `unpatch_d2h≈9.47s`（GPU unpatch + 每批 `cudaMemcpy` D2H ~0.35GiB malloc 内存）。tile 批间无 stream 并行、decode 与 mux/音频无 CPU-GPU 流水重叠（音频 decode 已在 DiT 期完成，热 audio≈0.25s）。

**改动（默认零行为变化）：**
- `H3_VAE_MAX_TILE_BATCH` 4→8；`h3_vae_int8_tile_batch_want` clamp 4→8（`H3_VAE_TILE_BATCH=8` 才启用，默认仍 4）。
- `vae_tile_batch_fit`：per-tile 常数 512→**384** MiB（实测 tile scratch ~265MiB/tile，旧 512 会把 22GiB 空闲错算成 cap 6）；删除「free<12GiB 强制 want=1」硬闸；slack/per 可用 env 覆盖；log 加 `want=`。

**238 实测（混合臂，热 clip 2，各跑 repeat=2 取 hot=1；同 seed 0x76800001）：**

| 臂 | cap_batch | tile_row | `vae_blocks` | `unpatch_d2h` | `vae` 总 | 热 `total` | waffle b16 | 结论 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| batch=4（基线） | 4 | 4+2 ×3 | 9.893s | 9.469s | 14.755s | **57.317s** | 0.826–1.034 | 现状 |
| batch=4（复测） | 4 | 4+2 ×3 | 9.883s | — | 14.749s | **57.430s** | ≤0.943 | 噪声 ±0.1s |
| batch=8 | 8 | **6 ×3** | 10.158s | 9.639s | 15.080s | **57.675s** | 0.822–1.030 | **无收益，−0.36s** |
| batch=8（复测） | 8 | 6 ×3 | 10.155s | — | 15.058s | **57.758s** | ≤1.030 | 稳定复现 |

- **tile 批 4→8 未提速，反而稳定慢 ~0.3–0.4s**：6 tile 合批后 SDPA 仍是 per-tile 形状（`n_batch=6` 一次 cuDNN SDPA），GEMM 行合并的收益被更大 batch 的激活 workspace 压力与 L2 命中下降抵消；`unpatch_d2h`（~9.5s，占 VAE 2/3）与 tile batch 无关，是真正的串行尾。
- **32GB 上限：** batch=8 在冷 clip（DiT/sage3 驻留，free≈17GiB）fit cap=8 实跑 6/批 **未 OOM**；热 clip free≈22GiB 更宽。上限高于 8，但 **无提速动机** 不再上调。
- **未 OOM、未减窗（34→refit 27/29 是既有 INT8 窗策略非 VAE 窗）、未 overlap DiT、未假 sage3。**

**质量对照：** 同 seed 下两臂 latent checksum 均 `0x5fc00e592bac60a4`（DiT 输出一致）；waffle 热片 b16 batch8=0.822–1.030 vs batch4=0.826–1.034（同区间）；成片像素级 |d| mean=1.48、p99.9=12（解码 mp4 噪声级），逐帧亮度相关 0.9973——**无结构差异**。

**探索过但否决：** unpatch D2H 改 pinned（`h3_cuda_host_alloc`）在 `fr->rgb` 生命周期（frames_free 用 `free()`）下 segfault，须把 pinned 标志穿进 `h3_vae_int8_frames` 结构/释放路径，改动面出本拍「不动 VAE 实现语义」范围；且 D2H 尾 ~9.5s 是跨 6 批累加，pinned 只省每批 memcpy 的 staging，理论上限 ~2–3s，留作下一刀候选（需带生命周期改造）。

**结论：§10.1 第 2 条「并行/批量 VAE 解码」在本仓 tile 批维度无真收益（−0.36s 为噪声外稳定回归），外部 7.55s→0.602s 的前提是整帧 batch/流水化实现，与本仓逐 tile decode 形态不同。产品默认保持 batch=4 不变。** 真串行尾在 `unpatch_d2h`（~9.5s）与 `vae_blocks`（~9.9s），下一刀候选：① unpatch D2H pinned+async 流水（须改 frames 生命周期）；② tile 批间 CUDA stream 并行（多 stream decode，VRAM 充足时 2 流）。

### 8.7 本拍（2026-10-04 续拍 4）：TaoMate 2-stage VAE prewarm 接线——host 暖载全隐藏，device 尾 ~0.3s 仍串行

> 目标：flat 去噪循环在 `step==steps-2` 有 `kick_vae_prewarm`（pthread 暖 VAE mmap pin/hostreg），但 TaoMate 2-stage 预设分支（`hi_st` 高分尾循环）绕过了该循环，KPI-B 的 `vae_load` 在高分 DiT 后同步跑。补上同语义 prewarm。

**改动（`tools/h3_cuda_generate.c`，一处插入）：** TaoMate 高分尾 `for (hi_st = lift_k; …)` 内 `hi_st == steps - 2` 处 `kick_vae_prewarm(&vae_th, &vae_prewarm_job, vae_path, &vae_store, &vae_opened, device, hi_st+1, 0, vae_blocks)`，守卫 `!vae_th_ok && !skip_vae && vae_path`——与 flat 路径完全同签名同语义。

**238 实测（混合臂，`H3_SOL_MAX_SEQ=16000`，窗 34，官方 VAE，seed 0x76800001，repeat=2，seed 同 §8.6 基线）：** `out/review-2026-10-04/kpib-vae-prewarm-mixed2/`
- 两臂 `vae_th: created at=dit_step4`、`joined ok=1 hostreg=1`：冷 clip `work_ms=166.1`、热 clip `work_ms=0.0`，**host 暖载全部藏在高分 DiT ~30s 之后**（joined_ms≈30s 是等 DiT 跑完）。
- 热 clip 2：`phase_ms: dit=41001.2 vae=14694.8 vae_load=295.6 vae_blocks=9882.1 audio=252.9 mux=867.8 total=57377.0`（**57.377s**，对照 §8.6 batch=4 基线 57.317s，噪声内**无回归**）。冷 clip 1 `total=77403.1`（te=7.47s 冷 TE 为主，与 prewarm 无关）。
- sol 低分 hits=300/300 fallback=0、高分 cudnn want=200 hits=200、`sdpa_strict` 干净；waffle b16 0.822–1.030（同 §8.6 区间）；`av_sync PASS`（drift 8ms）两 clip。成片 `kpib-vae-prewarm-mixed2_5s_r{1,2}.mp4` 已在 review 目录。

**余尾（判定：本拍不收）：** 热 clip `vae_load≈296ms` 仍在关键路径——32GB 下 clip 间 `vae_release_device` 卸 device W（`host_pack=kept mmap_pin=1`），prewarm（`want_dev=0`）只暖 host 侧，`h3_cuda_vae_weights_load` 的 dequant 249ms + h2d 24ms 仍同步。要消这 ~0.3s 须 prewarm 线程预填 `w_dev`（多占一份设备显存，32GB 高分段无 headroom）或接受 `host_pack_restore_fail` 路径改 restore——均出本拍「调度层小改」范围。**对 27s 目标无实质影响**（占比 0.5%），登记为已知余项。

---

## 10. 外部分析参考：Kimi 分享《VDN-H3 加速原理对比》对照清单

> **性质声明：** 本节是 **第三方外部分析**（Kimi 分享文档）的可迁移结论摘录，**不是本仓实测**，其秒数/倍数一律 **禁止** 换算到本仓金线、KPI-A、KPI-B。只作下一跳调研对照。红线：不装假 sage3、不 TAE/LTX、不减窗、不动质量尺；任何改变步数或质量尺的方向，明确标为 **不可用于默认产品路**。

### 10.1 逐条对照

| # | VDN 分享中的结论 | 与本仓的对照 | 处置 |
| --- | --- | --- | --- |
| 1 | **跨步缓存（cross-step caching）**：Sol 单项 17.7s → 6.88s | 与 §2.2 官方阶梯一致（TeaCache 是该阶梯最大单步增益），**前提是把推理臂改成可复用 residual 的 NFE 结构**。本仓 turbo4 5-step 质量地板上蒸馏已把步数冗余榨干，缓存红利不可叠加（turbo-explainer 已拒）。 | **不可用于默认产品路**（改 NFE 结构 = 改质量尺）。若引用其数字，必须写明前提；**KPI-B 不因此宣称可 ≤27s**。 |
| 2 | **并行/批量 VAE 解码**：7.55s → 0.602s | 对照本仓硬墙里的 **官方 H3 VAE 解码 9.86s**：方向是解码的批量/流水化（分块并行、与 DiT 阶段流水重叠），**不换 VAE 实现、不动质量尺**。 | **已实测（§8.6，2026-10-04 续拍 3）**：tile 批 4→8 无真收益（−0.36s 稳定回归）；串行尾在 `unpatch_d2h` ~9.5s 与 `vae_blocks` ~9.9s。外部倍数的前提是整帧 batch/流水化实现，与本仓逐 tile decode 形态不同。下一刀候选：pinned+async D2H（须改 frames 生命周期）、跨 tile stream 并行。 |
| 3 | **latent-to-latent 直连**：潜空间翻译器 + H3 ×2 上采样，跳过像素中转 | 对照本仓 2-stage LBH：需先核实 lift 后是否仍在 **pixel 中转**（latent→decode→pixel→re-encode）。若是，latent 直连可省两趟 VAE 税。 | **可调研方向**：先审计 LBH 中转形态；任何换精修模型（LTX 类）仍属合同外。 |
| 4 | **算子融合清单**：RMS-AdaLN、QKNorm+RoPE、QKV 合并 | 对照本仓现有 fused gate+AdaLN（`h3_dit_fused_epi.h`）与 gemm_epi 系列：**RMS-AdaLN 已覆盖**；**QKNorm+RoPE 融合、QKV 三次瘦 GEMM 合并为宽 GEMM 是查漏项**，与 §2.3 Pillar 1 官方融合表同源。 | **查漏项**：列入 5090 fine-opt 的融合核对表，逐项确认是否已有等价核。 |
| 5 | **VDN 工程技巧**：chunk 级扫描；寄存器内分块 Gauss-Jordan；VDA 旁路 CUDA stream 与窗口 Softmax 并行 | 与本仓 sol/waffle 注意力路径的 **调度层** 对照：重点是「旁路分支不占主 stream、与窗口 softmax 并行」的调度思想，不是核本身。 | **注意力调度参考**：供 sol_cute/waffle 后续调度优化比对，不涉及换核合同。 |
| 6 | **DGX Spark 56.2s @768p 不 OOM** | 作为 **32GB 级受限显存可行性**的外部参照：768p 全程不 OOM 在工程上成立，佐证 §8.4「32GB 装载是下一刀」方向正确。 | 仅作可行性参照，不作秒数对照。 |

### 10.2 本节红线复述

1. 以上所有数字均为 **外部分析的转述**，非本仓实测，不进 KPI 账目。  
2. KPI-B ≤27s **维持现状表述**（~52–53s，硬墙未破），不因第 1 条改口。  
3. 第 1 条（跨步缓存/任何改变 NFE 步数或蒸馏配方的方向）**不可用于默认产品路**，只可在 SKU 隔离臂讨论。  
4. 不因此装假 sage3、不引入 TAE/LTX、不减窗（window 仍 32）。

### 10.3 外部参考：DMAD 4步 LoRA（arXiv:2610.02188）

> **性质声明：** 本节是 **上一拍已完成的调研结论** 的归档，来源是公开论文/权重信息，**非本仓实测**。所有秒数/质量结论 **禁止** 换算到本仓金线、KPI-A、KPI-B；不承诺任何秒数。

- **是什么：** DMAD（字节官方 H3 4步 LoRA，[arXiv:2610.02188](https://arxiv.org/abs/2610.02188)）把 33B H3 DiT 蒸馏到 **4 NFE**。与本仓 turbo4 同属「少 NFE 蒸馏学生」族，是 **turbo4 的竞品/对照**，不是叠加项。
- **与本仓的关系：** 本仓默认产品路是 turbo4 INT8 金线。DMAD 若要引入，**只能作可选后端 / SKU 隔离臂**，**禁 merge 进 INT8 金线**（换学生权 = 换轨迹 = 改质量尺）。
- **采用门槛：** 必须先 **A/B 过本仓质量地板**（与 turbo4 同尺对比 waffle / av_sync / 人眼签成片），证明其 4-step 轨迹在本仓尺子下不差于 turbo4，才谈得上作可选后端。未过 A/B 前，**不进入任何默认路径，不宣称加速**。
- **红线：** 不因此减窗、不 TAE/LTX、不假 sage3；任何改变步数/蒸馏配方的方向仅 SKU 隔离讨论。

---

## 9. 修订记录

| 日期 | 变更 |
| --- | --- |
| 2026-10-01 | 初稿：基于 sol-engine 公开 README/toml/源码/validation 与 H3-OnDevice 页；对齐本仓 KPI-A/B 与既有加速纪律。 |
| 2026-10-01 | §8 施工：238 接真 `cute_sm120`；KPI-A opt-in 热 17.244s hits=400/400；KPI-B 高分 OOM；sage3 仍无核。不声称 27s。 |
| 2026-10-04 | §10 追加：Kimi 分享《VDN-H3 加速原理对比》摘录为外部分析参考对照清单；不换算、不宣称 KPI-B 可达 ≤27s。 |
| 2026-10-04 | §8.5 施工：高分 sol QKVO「用完即卸+按需重分配」（`H3_SOL_HIRES_QKVO`，默认 off）。OOM 归因为 QKVO+sol 临时张量+INT8 窗 34+激活叠加超载（非加载 bug）。本拍会话无 238 凭据，238 实测待补；不减窗、不 TAE/LTX、不假 sage3、未 commit。§10.3 归档 DMAD 4步 LoRA 调研（arXiv:2610.02188，turbo4 竞品，须 A/B 过质量地板才作可选后端）。 |
| 2026-10-04 | §8.5 实测（238）：全 sol 两版释放（末层/每层）均 OOM——根因是 native INT8 窗+激活占 28.8GiB、sol fwd 连 530MiB 挤不出，QKVO 释放（torch 侧 2GiB）救不了；**不减窗红线下判定全 sol 不可行/不划算，退回混合臂**。混合臂复测热 57.372s（低分 sol hits=300 fallback=0，waffle 0.826–1.037，av_sync PASS）。环境修复：嵌入式插件须 `PYTHONHOME=conda-env` 否则 `_posixsubprocess` 缺失致 sol 全 fallback。窗 30≥24 未减、未 TAE/LTX、未假 sage3、未 commit。 |
| 2026-10-04 | §8.6 施工+实测（238）：VAE tile 批并行（§10.1 第 2 条）。`H3_VAE_MAX_TILE_BATCH` 4→8、`tile_batch_want` clamp→8（默认仍 4）、`vae_tile_batch_fit` per-tile 512→384MiB + 去 12GiB 硬闸 + log 加 want。实测 tile 批 4→8 **无真收益**（热 57.32/57.43s vs 57.68/57.76s，稳定 −0.36s；`vae_blocks` 9.88→10.16s 反升）；32GB 未 OOM（上限高于 8，无提速动机不再上调）。质量对照：同 seed latent checksum 一致、waffle 同区间、逐帧亮度相关 0.9973。pinned D2H 探索因 `fr->rgb` 生命周期 segfault 否决。§10.1 第 2 条改判「已实测，外部倍数不适用本仓逐 tile 形态」。产品默认 batch=4 不变、未减窗、未 overlap DiT、未假 sage3、未 commit。 |
| 2026-10-04 | §8.7 施工+实测（238）：TaoMate 2-stage 高分尾接 `kick_vae_prewarm`（`hi_st==steps-2`，`want_dev=0`，同 flat 路径签名）。混合臂（`H3_SOL_MAX_SEQ=16000`，窗 34，官方 VAE，seed 0x76800001，repeat=2）`out/review-2026-10-04/kpib-vae-prewarm-mixed2/`：两臂均 `vae_th: created at=dit_step4` 且 `ok=1`（hostreg/mmap pin 166ms 冷 / 0ms 热**全藏在高分 DiT ~30s 后**）。热 clip 2 `phase_ms: dit=41001.2 vae_load=295.6 vae_blocks=9882.1 total=57377.0`（57.377s vs 基线 57.317s，**无回归**）；sol hits=300/300、高分 cudnn 200/200、waffle 0.822–1.030、`av_sync PASS`。**余尾：** `vae_load≈296ms` 仍在关键路径——32GB 下 clip 间 `vae_release_device` 卸 device W，prewarm（want_dev=0）只暖 host，`h3_cuda_vae_weights_load` 的 dequant+h2d ~296ms 仍串行；预填 `w_dev` 须加分设备显存，出本拍范围。成片已入 review 目录待人工确认。未减窗、未 TAE/LTX、未假 sage3、未 commit。 |
