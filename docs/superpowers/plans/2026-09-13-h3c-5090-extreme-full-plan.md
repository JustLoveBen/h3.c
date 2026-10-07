# 8×RTX 5090 · H3 768P 完整极致施工图

> **For agentic workers:** 本文是施工图。按 Phase 落地；每相有动作 / 文件 / 验证门 / 依赖 / 风险。不要嵌 Python Comfy。不要把社区内核写进生产路径。不 commit。密码只走 `H3_SSH_PASS_5090x8`，不入库。
>
> **日期：** 2026-09-13。取代 [旧「极致」](./2026-09-13-h3c-5090-multigpu-extreme.md)（偏文献、偏保守、把 aikitoria 写成永不测）。主尺子：[768P 评测协议](./2026-09-13-h3c-768p-eval-protocol.md)（**5s · 1344×768**）。SelfLift 以 [facok/comfyui-SelfLift](https://github.com/facok/comfyui-SelfLift) **master 源码**为准，旧结论「接不上 / 只是图模论文」作废。
>
> **Goal:** 在本仓 MiniMax-H3 + 8×5090 上把已证明的刀叠满，用第一方数据决定 P2P，并把 SelfLift 做成加速档正式候选。
>
> **Architecture:** 金线保持 U=1 / 8-step INT8 / cuDNN / 官方 VAE。延迟 KPI 走单卡叠刀 + 可选 U=2 + 可选 SelfLift。吞吐 KPI 走 8 路请求级 DP。公路仍是官方 `canAccessPeer`；P2P 只认本机探针。
>
> **Tech Stack:** 本仓 C/CUDA。对照不进生产：facok SelfLift、LBH-123-AI H3 latent upscaler、aikitoria `595.45.04-p2p`。
>
> **Spec:** [lightning-survey](./2026-09-13-h3c-minimax-lightning-survey.md) · [ulysses-driver-options](./2026-09-13-h3c-5090-ulysses-driver-options.md) · [deepseek-opt-lessons](./2026-09-13-h3c-deepseek-opt-lessons.md) · [768p-eval](./2026-09-13-h3c-768p-eval-protocol.md)。秒数来源：`out/verify_host_stage/REPORT.md`、`out/verify_tae/REPORT.md`、`out/verify_nvfp4/REPORT.md`、`out/verify_turbo_impl/PROGRESS_PHASE2.md`。

**一句话：** 极致不是 FastWan 1.78 s，也不是 U=8。极致 = 本仓模型 + 8×5090 上把已测刀叠到 5s 1344×768，并用第一方 P2P 数据决定要不要开卡直写。SelfLift 是加速档正式候选，不是「以后再说」。

---

## Global Constraints

- 主 KPI：**一条 5s · 1344×768 · 有声 mp4 墙钟**（`phase_ms total`）。480p 数字只作缩放参考，必须标明「未在 768P 复测」。
- 金线默认：U=1、8-step、`cfg=1`、`sdpa=cudnn`、官方 VAE、无 LoRA、`--te-cache` 关。
- 8× GeForce RTX 5090，驱动 **595.45.04** Open Kernel Module，无 NVLink，`topo -p2p r` 全 CNS（2026-09-13 07:47 UTC 只读）。
- 不 commit。不改推理代码除非本 Phase 点名的文件。P2P 探针是独立工具 `tools/h3_p2p_bw_probe.cu`。
- 密码不写文档。SSH：`H3_SSH_5090x8` / `H3_SSH_PASS_5090x8`，`ssh -b 192.168.0.92`。
- 凡回代必须写「回代」。禁止 `480p × 2.49` 冒充已测 768P。

---

## 1. 旧「极致」哪里不够深

旧文 `2026-09-13-h3c-5090-multigpu-extreme.md` 是战略综述，不是施工单。它自己文首写了「不要装内核、不要改代码」。结合本仓之后，不够深的地方如下。

### 1.1 文献综述当结论

| 段落 | 做了什么 | 没做什么 |
| --- | --- | --- |
| §1 社区地图 | xDiT / LightX2V / Cache-DiT / FastWan / vLLM 14.1 GB/s 排了表 | 没有一条接到 `h3_ulysses_comm.cu` / `h3_dit_w4a8.c` 的补丁级步骤 |
| §1.2 P2P | 把 aikitoria README 55 GB/s 和 vLLM 14.1 GB/s 并排 | **用别人的 14.1 当本机终局**。用户已否决 |
| DeepSeek | driver-options §8 + deepseek-opt-lessons 拆清了「没改 ko」 | 正确，但旧极致把它收成「不要学改驱动」，没有把「少字节 A2A」写成 Phase |

### 1.2 没有和 C 栈叠刀

旧文写的时候，下面这些 **已经或即将落地**，但天花板仍按「金线 e2e 78.74 / turbo+TE 50」在算：

| 刀 | 文件 | 已测（480p cafe，未在 768P 复测） |
| --- | --- | --- |
| host_stage mmap 别名 | `h3_w4a8_cache.c`、`h3_gpu_cuda.cu`、`tools/h3_cuda_generate.c` | host **13.2 → 1.26 s**；金线 e2e GPU1 warm **62.13 s**（册 GPU0 仍是 78.74） |
| TE cache | generate `--te-cache` | HIT 金线 e2e **61.41 s**（旧 host 路径）；turbo4+TE HIT **~49.7 s** |
| turbo4 | `h3_generate_preset.c` | DiT **16.47 s** 档 |
| TAE | `h3_tae.c` / `h3_tae_cuda.cu` | VAE **10.8 → 2.35 s**；turbo4+TAE e2e **~46.4 s**（有损） |
| Ulysses | `h3_ulysses_comm.cu`、`h3_dit_w4a8.c` | 正确（U=2≡U=8 MAE 11.159）但 U=2 DiT **34.66**、U=8 **137** |
| NVFP4 | `h3_dit_nvfp4.c`、`h3_w4a8_cache.c` | 出片；CPU dequant **~424 s**，墙钟 ~479 s，**不能宣传** |

旧文把「今天就能宣称」钉在 50 s，没把 host_stage + TAE 叠进去，也没把尺子切到 768P。

### 1.3 把风险写成永远不测

driver-options §7 的门槛（有人 generate / 缺 `iommu=pt` / 默认分支 610）**当时成立**，结论「这次不试」也对。旧极致把它升级成「生产箱连 `595.45.04-p2p` 都不装」的战略句，并用 vLLM 14.1 GB/s 暗示「即使门开了也没用」。

**作废：** 用他人带宽当本机终局。P2P 必须是 Phase 硬门：本机 `canAccessPeer` + `tools/h3_p2p_bw_probe.cu`。缺窗口就写 runbook，不装；有窗口就测。

### 1.4 SelfLift 旧结论作废

768P 协议 §7 和旧极致把 self-lift 写成「未接上 / 图模论文 / 480+FL2VA 更慢」。那是在没读 [facok/comfyui-SelfLift](https://github.com/facok/comfyui-SelfLift) 之前。该仓是 Comfy 插件，**已经有 H3 音视频采样器**（论文未验证，但是可 port 的算法，不是「接不上」）。见本文 §3。

### 1.5 480p 尺子低估了 U=2

旧文用 cafe 480p 断定「U=2 赢不了 turbo4」。768P 计算大约 3.5–5×（回代），A2A 体积只按 packed S ≈2.40×，**计算/通信比变好**。U=2 在 768P 上比在 480p 上更值得做——仍须 Phase 0 金线数字后回填，不能用 480p 封死。

---

## 2. 「极致」的定义与两条 KPI

**做不到的：** FastWan-QAD 1.78 s（1.3B Wan + 3-step + 原生 NVFP4 + TAEHV，异模型）。RunningHub 28.7 s（8×6000D、50→4 步、能 P2P）。explainer 5–8 s。

**本仓极致：** 在 MiniMax-H3 + 8×5090 上把已证明的刀叠满，用第一方数据决定 P2P，SelfLift 只进加速档。

| KPI | 成功长什么样 | 错误手段 |
| --- | --- | --- |
| **延迟**（一条 5s 768P 墙钟） | 单卡叠刀天花板；U=2 仅当软件刀或自测 P2P 后 DiT 快于单卡 | U=8 当主路径；把 480p 50 s 改写成 768P |
| **吞吐**（8 卡条/小时） | 8 路完整 generate，卡间零 A2A | 一条请求切 8 卡还说「8 卡都用上了」 |

---

## 3. SelfLift（facok/comfyui-SelfLift）

源：[facok/comfyui-SelfLift](https://github.com/facok/comfyui-SelfLift) `master`（2026-09-12 推）。已读 `README.md` / `README_CN.md` / `selflift.py` / `h3_upscaler.py` / `h3_tst.py` / `h3_tiling.py` / `nodes.py`。可选权：[LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)。**不要 git submodule 进生产。**

### 3.1 算法：和「480 草稿再跑一遍 768」不是一回事

`nodes.py:progressive_sample` 把 **同一条 N 步 Euler 日程**切成三段：

```text
低分前缀     步 1..k          在 (h,w) 上跑 k 次 denoiser
transition   第 k 次预测 x0   不增加 NFE：丢掉这次 Euler，lift 后再补
高分后缀     步 k+1..N        在目标 (H,W) 上跑完剩余步
```

源码钉死：

- `transition_step` 默认 **6**（8 步里 6 次低分 NFE）。校验：`1 ≤ k ≤ len(sigmas)-2`（`nodes.py` `_validate_schedule`）。
- `lowres_scale` 默认 **0.5**，范围 0.25–1.0。低分格子：`h = max(2, round(H * scale / 2) * 2)`（对 **latent** 空间，不是像素）。
- 只接受 **CONST + 标准 Euler + `s_churn=0`**（`_validate_sampling`）。本仓 generate 已是 velocity euler（`h3_dit_w4a8_euler_step_f32`，`X += (σ-σ')·v`），合同对齐。
- **N 步仍是 N 次 NFE。** 第 k 次低分前向当作论文 Eq.3 的 clean 预测；Euler 更新丢弃，lift 后按 `sigma_k → sigma_{k+1}` 重做（`nodes.py` 356–357、410–417 行）。
- `rho=0`：**不走** VAE decode→upscale→re-encode。`rho>0`：多一次像素锚点往返（`selflift.paired_lifts`）。
- 音频：H3 nested latent 里音频 **无空间维**，transition 用同一边界 Euler 接着走（`auxiliary_next`），不是重采样一条音轨。

`selflift.py` 的 zero 路径（论文 Algorithm 1 mode=zero）：

1. 低分 x0（VAE 空间）。
2. 双路 lift：直接 latent 上采样（nearest / 外部 lifter）vs 像素 VAE 锚（decode → 上采样 → encode）。
3. `delta = z_pix - z_lat`，按位置不一致度取 top-`rho`，强度 `w_min..w_max` 修正。
4. 在 transition sigma 上重新加噪，续跑。

**和「480 草稿 + 768 FL2VA 再 8 步」的差别：**

| | SelfLift（本仓要 port 的） | 480 + FL2VA 8+8（旧协议禁止叫 SelfLift） |
| --- | --- | --- |
| NFE | **仍是 N** | 2N |
| 日程 | 中途换分辨率，从 transition σ **续跑** | 第二段从高分噪声/条件 **重开** |
| 音频 | 同一条 Euler 边界带过去 | 两段各出 PCM，跟谁是产品问题 |
| VAE | zero 才多一次往返；外部 upscaler 可 0 | 低分解码 + 高分整段 |
| 墙钟 | 省的是前缀 DiT 面积 | 高分 8-step DiT 几乎全付 → **更慢** |

SelfLift-rich（蒸馏 lifter + On-Policy Self Recovery）**不在该仓**。

### 3.2 H3 适配和论文差在哪

论文：rectified-flow **图像**，SelfLift-zero。插件 README 写明 H3 是 **实验性音视频适配，论文未验证**。

| | 论文 / Image 节点 | H3 节点（源码） |
| --- | --- | --- |
| 默认 `rho` | 0.3（图） | **0.0**（`SelfLiftH3Sampler`） |
| 默认 lift | nearest / bilinear | 有 H3 权则 **外部 3D upscaler**；否则 none |
| 硬拒 | — | `rho=0` 且 `upscaler_model=none` 直接 ValueError（`nodes.py` 485–489） |
| 建议 zero | 图参 `rho=0.3, w=0.5/1` | **`upscaler_model=none` + `rho=0.6, w_min=w_max=1`** |
| 像素锚 | 图像 VAE | 必须是 **H3 官方 VAE**（同一 latent 空间）。TAE 禁止当锚 |
| 外部权 | 无 | LBH-123-AI `LatentResizer3D`（24ch、512、12+12 block）。**不是** SelfLift-rich |

两种 H3 模式（README 与 `nodes.py` 一致）：

1. **外部 upscaler（实用默认）：** 装 LBH 权，`rho=0` → 学习式纯 latent lift，**跳过 VAE 往返**。不是 SelfLift-zero。
2. **SelfLift-zero：** `upscaler_model=none` 且 `rho>0`。直接 nearest 在 H3 上伪影大；单 seed 对照：纯像素锚干净，`rho=0.3` 仍糊，`rho=0.6`+`w=1` 去掉主伪影。**单 seed，不是统计。**

768 短边 ×0.5 = **384**，README 警告可能出训练分布。`highres_tiling` 省显存，只留首块音频、无跨块注意力；**与 TST、`noise_mask` 不兼容**。

### 3.3 对我们 C 栈：几何与最小改动

官方 768P：**1344×768**（`h3_adapt_canvas(1920,1080)`）。`H3_CANVAS_MULTIPLE=32`，`H3_VAE_SPATIAL_RATIO=16`，DiT patch 2×2（latent 必须偶数）。

| | 像素 | %32 | latent | patch 格（H/2×W/2） | video_seq（t=37） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 目标 | **1344×768** | 42×32 / 24×32 | **84×48** | 42×24 = 1008 | **37,296** |
| `lowres_scale=0.5`（插件公式打在 latent 上） | **672×384** | 21×32 / 12×32 | **42×24** | 21×12 = 252 | **9,324** |

`round(48*0.5/2)*2 = 24`，`round(84*0.5/2)*2 = 42`。672×384 已 32/16 对齐，**不要**再 `adapt_canvas` 成 864×480。低分不是 cafe 480p。

`--init-latent`（`tools/h3_cuda_generate.c`）是 **开跑前**灌整段 latent / G3-lite t=0 混合，**不能**当 transition。SelfLift 要的是：步 k 之后改 packed 形状、lift、按 σ 续噪、重建 cuDNN plan。

**最小 port（不嵌 Comfy）：**

| 动作 | 文件 |
| --- | --- |
| Euler 循环在 `transition_step` 插入 lift，丢弃该步 Euler，lift 后补一步 | `tools/h3_cuda_generate.c`（`for (step = 0; step < steps)` ~3798） |
| `paired_lifts` + `artifact_aware_consistency_lift` | **新建** `h3_selflift.c` / `h3_selflift.h`（从 `selflift.py` 搬公式，C） |
| 低分 x0 → 官方 VAE decode → 像素放大 → `h3_vae_encode_f32` | 已有 `h3_vae_encode_f32.c`（现用于首帧/REF）；decode 用官方路径，**禁止 TAE 当锚** |
| 外部 3D lifter（rho=0 默认候选） | **新建** `h3_h3_upscaler.c`（对照 `h3_upscaler.py` `LatentResizer3D`），权不进 git |
| packed ↔ `[24,T,H,W]` 在 transition 重切；音频行不改空间 | `h3_dit_w4a8.c` pack/unpack；audio 走现有 `h3_dit_w4a8_euler_step_f32` |
| 切换后重建 SDPA plan | `h3_gpu_cuda_sdpa_cudnn.cpp`（已按 device 分槽；还要按新 `tensor_seq`） |
| 加速档入口 | `h3_generate_preset.c`：开 `--preset selflift`（非金线）。现有 `lift` 硬拒改成指向本 preset，金线永不走 |
| 评测臂 | `tools/h3_768p_eval.sh` 加 `selflift`；成片 `out/review_videos_768p/` |

`--preset gold` 永不走 SelfLift。`cfg=1` 保持。

### 3.4 5s 1344×768 墙钟回代

分母：单卡 768P 金线 DiT **回代 105–150 s**（中位 ~120；480p host_stage 后 DiT 30.06 × 3.5–5.0）。**未测。** TE ~13 s（HIT 0）。官方 VAE decode **回代 20–35 s**（480p 10.5 × 像素 2.49）。host_stage 已在默认路径 ~1.3 s。

假设 `k=6 / N=8`、低分 672×384（面积 1/4，packed S 回代 ≈0.27× 因为文本+音频不缩）。

| 段 | 原生 768P 8-step（回代） | SelfLift 外部 upscaler `rho=0`（回代） | SelfLift-zero `rho=0.6`（回代） |
| --- | ---: | ---: | ---: |
| 低分 6-step DiT | （含在 120） | **14–20 s**（按 480p 步时 × token 比） | 同左 |
| 高分 2-step DiT | （含在 120） | **26–38 s** | 同左 |
| **DiT 合计** | **105–150** | **40–58** | **40–58** |
| transition | 0 | 学到的 3D lift **2–8 s**（未测） | 低分解码 **~6.5 s** + 768P **encode 20–35 s** |
| 成片 decode | 20–35 / TAE ~6 | 同左 | 同左（再付一次） |
| TE + 其它 | 13+5 | 13+5 | 13+5 |
| **e2e 金线配方** | **145–200** | **~85–120** | **~110–160**（往返可能吃掉 DiT 节省） |
| **e2e turbo4+TE HIT+TAE** | **~64–95** | **~40–60** | 仍可能被 encode 托住 |

turbo4（4-step）应对齐论文 FLUX「3 of 4」：`transition_step=3`，不要套默认 6。

**验证门（必须自测，回代不作承诺）：**

- 外部 `rho=0`：e2e 相对同 seed 原生 768P **回代目标 0.55–0.75×**，且人眼可接受（协议 §5）。
- zero：若 768P encode 墙钟 ≥ 20 s，**默认不采用** zero 当延迟主刀；只作质量对照。
- 任一臂花屏 / 身份崩 / 音频丢 = FAIL，关掉 preset。

### 3.5 质量风险

- **384 短边 OOD：** 插件自己写。Phase 1B 必须用 `p01_sprint` + `p02_talking_head` 人眼，不能只看 MAE。
- **单 seed：** README 诊断表不是统计。我们的门是协议 8 条里至少冒烟 2 条同 seed 对原生。
- **金线仍是原生 768P 8-step。** SelfLift 只进加速档。
- nearest-only（rho=0 无外部权）在 H3 上大范围伪影——C 栈不要做这条当默认。
- 像素锚必须官方 VAE；TAE 只用于 **最后成片**。
- 音频必须跟视频同一 transition；禁止「视频 lift、音频重开」。

### 3.6 TST 要不要进 C 栈

`h3_tst.py`：无训练，~1–2% 开销，改 packed 视频段 post-RoPE Q 的温度（谱张力）。与 `highres_tiling` 不兼容。

本仓金线 SDPA 是 **cuDNN fused**（`h3_gpu_cuda_sdpa_cudnn.cpp`），拿不到逐头 Q/K 做 `A = softmax(QK)` 的帧级算子，除非改走 flash/手写核。

**结论：** TST **不进 Phase 0–2 关键路径**。等 SelfLift 成片后，若闪烁/身份漂是主诉，再开实验臂（需能导出 Q/K）。不要为 TST 重写 cuDNN。

### 3.7 明确不嵌 Python

禁止把 Comfy / facok 插件当 generate 后端。要的是：**公式 + 可选 LBH 权** port 进 C。SelfLift-rich 不训。

---

## 4. 施工图（Phase 0–4）

可并行的标「∥」。768P 数字在 Phase 0 之前全是回代。

```text
Phase 0  768P 金线第一枪          ─┐
Phase 1A turbo4+TE+TAE @768P     ├─ 延迟叠刀（单卡）
Phase 1B SelfLift 加速档候选     │
Phase 1C NVFP4 去掉 424 s        ┘  ∥ 1A/1B
Phase 2  U=2 软件刀（BF16+去 sync）  ∥ 可与 1C 并行；768P 对比要等 Phase 0
Phase 3  aikitoria 第一方 P2P     硬门（有窗口才装）
Phase 4  若 P2P≥20 GB/s → U=2 短测；8-way DP 启动器（∥ 随时可开）
```

self-lift 不再是「未接上以后再说」；它是 **Phase 1B**。

---

### Phase 0 — 768P 金线第一枪

**动作：** `p01_sprint`，5s，1344×768，金线配方，warm 至少 1 次（冒烟可 n=1，标题写冒烟）。脚本：`tools/h3_768p_eval.sh --arm gold --prompt p01_sprint --remote`。

**文件：** 不改推理。`tools/h3_768p_eval.sh`、`tools/h3_tokenize_prompt.py`、`evals/768p/prompts/p01_sprint.txt`。进度：`out/verify_768p/PROGRESS.md`。成片：`out/review_videos_768p/`（不要覆盖 `out/review_videos_2026-09-13/`）。

**验证：** ffprobe 5.000s±0.05 / 120 帧 / 有声；记下 `phase_ms` 全部分段与峰值显存。OOM 也是结果。

**依赖：** 远端 INT8 DiT / VAE / TE / audio / ids。GPU 上不要杀别人的 generate。

**风险：** 32 GB 上 768P INT8 峰值未测，可能贴边。

**门：** 没有这条原生墙钟，后面所有 768P「省了多少」都不许宣称。

---

### Phase 1A — turbo4 + TE + TAE @768P

**动作：** 同 prompt/seed：`turbo4`、`turbo4_te`（HIT 才报）、`turbo4_tae`。host_stage 已在默认路径，不要再当新刀。

**文件：** 已有 `h3_generate_preset.c`、`--te-cache`、`h3_tae.c` / `h3_tae_cuda.cu`。TAE 几何未写死 54×30，1344×768 → latent 84×48 应对上；**仍须 768P 实测**。

**验证：** 人眼（有损预期）+ e2e。480p 对照只许当缩放参考。

**依赖：** Phase 0 成片当质量对照。

**风险：** 768P TAE 时间对齐 / 糊。金线开 tae 必须仍 RC=1。

**门：** 加速档 768P e2e 有第一方数字。回代叠刀（480p HIT+TAE+host ≈ 30–38 s）**未在 768P 复测**。

---

### Phase 1B — SelfLift（加速档正式候选）

**动作：** 按 §3.3 port。先落地 **外部 upscaler + rho=0**（跳过 768P encode）。再对照 zero（`none` + rho=0.6）。`transition_step=6` 绑 8-step；turbo4 用 3。

**文件：** §3.3 表。权：远端 `latent_upscale_models/`，不进 git。

**验证：**

1. U=1 几何：低分 672×384、高分 1344×768、音频连续。
2. NFE：日志低分 6 + 高分 2 = 8（turbo4：3+1=4）。
3. 同 seed vs Phase 0 原生：人眼 + 分段墙钟。
4. `rho=0` 且无权 → 硬拒（对齐插件）。

**依赖：** Phase 0。官方 VAE encode 仅 zero 臂需要。

**风险：** 384 OOD；单 seed 不可外推；zero 的 encode 税；不要把 TAE 当锚。

**门：** 外部臂 e2e 低于原生且人眼过协议 §5 → 开 `--preset selflift`。不过门关预设。零臂若 encode ≥ 20 s → 标「质量对照，非延迟主刀」。

---

### Phase 1C — NVFP4 去掉 ~424 s（∥ 1A/1B）

**动作：** 每次 generate CPU `h3_dit_nvfp4_dequant_to_i8` 是墙。先做 **INT8 sidecar 落盘** 或 **GPU dequant 驻留**。不要先赌 sm_120 原生 FP4。

**文件：** `h3_dit_nvfp4.c` / `h3_dit_nvfp4.h`、`h3_w4a8_cache.c`（`stage_one_host_nvfp4` / `load_one_linear_nvfp4`）、`h3_w4a8_dequant.c`。可新建 sidecar 写入工具（独立，不改金线 `--dit`）。

**验证：** cafe 或 768P 墙钟不再含 ~424 s；DiT 仍约 28 s 量级（480p 已测）；探测失败仍硬退出。

**依赖：** 无。与 1A/1B 并行。

**风险：** sidecar 脏 key。禁止静默回退 INT8 ConvRot 金线权。

**门：** e2e 有资格和金线比之后，才讨论原生 FP4。

---

### Phase 2 — U=2 软件刀

**动作：** 只做 U=2，同 NUMA GPU0,1。禁止 U=8。

1. `h3_ulysses_comm.cu`：去掉 `dtype_bytes != 4` 硬拒（约 468、524 行）；pack 前 F32→BF16，unpack 升回。头文件已写 `dtype_bytes=2`。
2. `h3_dit_w4a8.c`：`uly_sync_rank`（3884）去掉每相 `cudaDeviceSynchronize`，A2A 前 per-rank event；`uly_h2d_aux`（3743）改 async。

**验证：** 先 480p cafe warm：DiT ≤ 28 s + BF16 另阈 MAE。过门再 768P（或 1s/2step 672 几何短测）。**不要写 overlap 后 U=2≈16 s。**

**依赖：** 软件刀不依赖 P2P。768P 对比依赖 Phase 0。

**风险：** 非位级；曾 IMA（per-device cuDNN plan 必须保持）。

**门：** 480p DiT < 31.67 才开加速开关。768P：U=2 DiT < Phase 0 单卡 DiT。

**回代（768P，F32 体积 ×2.40，host ~10–12 GB/s）：** U=2 DiT **回代 ~80–100 s** vs 单卡 **105–150**，有机会首次赢单卡。BF16 再半： **回代 ~70–85 s**。turbo4 4-step 通信减半，U=2 **可能与 turbo 叠**（480p 上旧文认为会亏——768P 要重测）。

**不要 U=8：** 480p 已测 137 s。768P 总线 **回代 ~1540 GB** F32；host 下 DiT **回代 ~190 s** 仍大亏。即便将来 P2P 40 GB/s，通信仍可能 ≥ 计算（~38 vs ~15 s），跨 NUMA `SYS`，8× 宿主税。8 卡赢面在 Phase 4 DP，不在 U=8。

---

### Phase 3 — aikitoria 第一方 P2P（硬门）

见 §5 runbook。本 Phase **不是**「读 vLLM 14.1 GB/s 后放弃」。

**动作：** 只读条件 → 缺则停（不装）→ 有窗口才装 `595.45.04-p2p` → `h3_p2p_bw_probe` → 失败立刻回滚。

**文件：** 不改推理。探针：`tools/h3_p2p_bw_probe.cu`（独立 nvcc）。

**验证：** `canAccessPeer` 0→1？单向/双向 GB/s。host_bounce 基线也要打。

**依赖：** 箱空、官方 ko 备份、IPMI/console、`iommu=pt`（缺则先 GRUB+reboot，不要裸装）。

**风险：** hang / GSP / 误写 / 覆盖 `nvidia.ko` / GRUB 残留。见 §5。

**门：**

| 结果 | 下一步 |
| --- | --- |
| 缺条件 / 有人 generate | **不装**。本 Phase 交付 = 条件清单 + 窗口预约 |
| `can=0` 或装失败 | 立刻回滚。P2P 记失败。延迟走 1A/1B/2 |
| `can=1` 且 uni **< ~20 GB/s** | 记第一方数。U=2 仍以软件刀为主（14 GB/s 级对 440 GB 回代通信 ~31 s） |
| `can=1` 且 uni **≥ ~20 GB/s** | 进入 Phase 4 的 U=2 短测（先 1s/2step，再 cafe/768P） |

**本轮执行（2026-09-13）：** 本会话 **没有** `H3_SSH_PASS_5090x8`，SSH 仅确认主机要密码（`Permission denied (publickey,password)`）。**未装、未 rmmod、未测带宽。** 上一份只读（07:47 UTC）：驱动 595.45.04、**无 `iommu=pt`**、当时 GPU0 有 generate。按门槛 **不要装**。探针已写好待窗口。

---

### Phase 4 — P2P 成立后的 U=2 + 8-way DP

**动作 A（仅 Phase 3 过 ≥20 GB/s 门）：** U=2、同 NUMA、1s/2step（480 或 768 几何，写清）。再 5s。不要一上来 U=8 cafe。

**动作 B（∥ 随时可做，不靠 P2P）：** 请求级 DP。新建 `tools/h3_dp_generate.sh`：8× `CUDA_VISIBLE_DEVICES=i` 调现有 `h3_cuda_generate`，`--ulysses 1`。TE cache 目录加 `gpu$i`（前科脏 key `63ba422170cf015a`）。NUMA：0–3 node0，4–7 node1。**不要改** `h3_dit_w4a8.c`。

**验证 A：** DiT < Phase 0 单卡；成片不花。  
**验证 B：** 8 条并发，单条 e2e 不差于单卡 10%；聚合 ≥ 6.5×（热/电）。

**风险：** P2P `can=1` 但拷贝坏（llama.cpp opt-in 教训）→ 位级/人眼不过就回滚驱动。DP 不要 MPS、不要 MIG（5090 无 MIG）。

---

## 5. aikitoria 自测 runbook

密码只用环境变量。**先写回滚，再动。** 有人 generate **禁止** `rmmod` / `install.sh`。

### 5.1 回滚（先写后动）

本机应对齐：驱动 **595.45.04** Open Kernel Module；`nvidia.ko` 在 `/lib/modules/$(uname -r)/kernel/drivers/video/nvidia.ko`（2026-09-13：`5.15.0-25-generic`）。社区模块 `modules_install` **覆盖** ko，不是 DKMS 一键卸。

```sh
# 装之前（官方栈仍健康时）
sudo mkdir -p /root/h3-nvidia-rollback
sudo cp -a /lib/modules/$(uname -r)/kernel/drivers/video/nvidia*.ko \
           /root/h3-nvidia-rollback/
uname -r | sudo tee /root/h3-nvidia-rollback/uname
nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 \
  | sudo tee /root/h3-nvidia-rollback/driver
cat /proc/cmdline | sudo tee /root/h3-nvidia-rollback/cmdline
# 记录 GRUB 是否已有 iommu=pt。没有就不要在装补丁的同时改 cmdline。

# 失败时（能进系统）
sudo cp -a /root/h3-nvidia-rollback/nvidia*.ko \
           /lib/modules/$(uname -r)/kernel/drivers/video/
sudo depmod -a
# 若本窗口改过 GRUB：还原 /etc/default/grub 后 sudo update-grub
sudo reboot
# 起来后必须：
nvidia-smi
nvidia-smi topo -p2p r
```

进不了系统：IPMI / 救援盘 + 官方 595.45.04 `.run` open 模块。做不到「只 rmmod、不重启」。没有备份 **不要装**。

### 5.2 只读门槛（不装）

```sh
export H3_SSH_5090x8=user@89.185.31.140
# export H3_SSH_PASS_5090x8=...   # 仅会话
sshpass -e ssh -p 22 -o StrictHostKeyChecking=accept-new -b 192.168.0.92 \
  "$H3_SSH_5090x8" 'bash -s' <<'EOS'
set -e
echo "=== nvidia-smi ==="
nvidia-smi
echo "=== generate? ==="
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
echo "=== driver / ko ==="
nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1
ls -l /lib/modules/$(uname -r)/kernel/drivers/video/nvidia.ko
echo "=== cmdline / iommu ==="
cat /proc/cmdline
ls /sys/class/iommu 2>/dev/null | head
echo "=== p2p topo ==="
nvidia-smi topo -p2p r
EOS
```

| 门槛 | 缺了怎么办 |
| --- | --- |
| `nvidia-smi` 上有别人的 `h3_cuda_generate` / cafe | **停。** 预约空窗 |
| 驱动不是 **595.45.04** | **不要**装 `595.45.04-p2p`，也不要为 P2P 升官方 610 |
| 没有 `iommu=pt`（翻译模式 dmar0–19） | **不要装补丁。** 先单独窗口：GRUB `intel_iommu=on iommu=pt` → reboot → `nvidia-smi` 仍健康 → 再开补丁窗 |
| `/root/h3-nvidia-rollback` 不存在 | **不要装** |
| 无 IPMI/人在现场 | 生产 8 卡箱 **不要**当第一次实验机 |
| aikitoria 源不是分支 `595.45.04-p2p` | 不要用默认 `610.57.04-p2p-v2` |

**缺条件 = 不装。** 窗口：空箱 2–4 h + 可 reboot。

### 5.3 有窗口才装（一对 GPU，不要 U=8 cafe）

1. 再跑 §5.2，确认仍空。
2. 克隆 aikitoria，**checkout `595.45.04-p2p`**，按他们 `install.sh`（rmmod → make → modules_install → depmod）。userspace/GSP 锁 595.45.04。
3. 需要的 reboot 做完后立刻 `nvidia-smi`。
4. 探针（不要 generate）：

```sh
nvcc -O2 -std=c++17 -o /tmp/h3_p2p_bw_probe tools/h3_p2p_bw_probe.cu
# 只用空闲的一对，例如 2 3；不要抢正在跑的卡
/tmp/h3_p2p_bw_probe 2 3 64
```

看：`canAccessPeer` 是否 1；`p2p_uni_*_GBps`；`host_bounce_roundtrip_GBps`。`peer_gate=0` 或拷贝错 → §5.1 回滚。

5. **仅当** `can=1` 且 uni ≥ ~20 GB/s：计划 U=2 **1s/2step**（写清 480 或 768）。仍不要 U=8 cafe。
6. 任何 hang / `RmInitAdapter` / `nvidia-smi` 失败 → 立刻回滚 → 再 `nvidia-smi`。

排除：tinygrad / QuixiAI 610 / Duanyll / **rtx-p2p（GSP 逆向）**。

---

## 6. 数字板

**480p cafe 已测**（不是 768P）：

| 项 | 秒 | 源 |
| --- | ---: | --- |
| 金线 GPU0 e2e / DiT | 78.74 / 31.67 | 金线册 |
| host_stage 后金线 GPU1 e2e / DiT / host | 62.13 / 30.06 / 1.26 | `verify_host_stage` |
| TE HIT 金线 e2e | 61.41 | 旧 host 路径 |
| turbo4+TE HIT e2e | ~49.7 | `PROGRESS_PHASE2`（旧 host） |
| turbo4+TAE e2e / VAE | ~46.4 / 2.35 | `verify_tae` |
| U=2 / U=8 DiT | 34.66 / 137.43 | 同 binary，正确更慢 |
| NVFP4 DiT / 墙钟 / dequant | 28.27 / ~479 / ~424 | `verify_nvfp4` |

**768P 回代（Phase 0 前禁止当已测）：**

| 项 | 回代 | 假设 |
| --- | --- | --- |
| 金线 U=1 e2e | **145–200 s**（中位 ~165） | DiT 30×3.5–5.0；VAE ×2.49；host 已 1.3 |
| turbo4+TE HIT+TAE U=1 e2e | **64–95 s**（中位 ~79） | DiT 16.5×3.5–5.0；TAE ×2.49；TE=0 |
| SelfLift 外部 rho=0 e2e | **85–120 s**（金线配方） | §3.4；lift 2–8 s |
| SelfLift-zero e2e | **110–160 s** | 多 768P encode |
| U=2 BF16 DiT / e2e | DiT **70–85 s** / e2e **~115–130** | 184×2.40 GB / 2 / 12 GB/s |
| U=2 + 自测 P2P 40 GB/s DiT | **~70–80 s** | 本机带宽必须自测，禁止抄 55 |
| U=8 F32 host DiT | **~190 s** | 总线 ×2.40；不要做 |
| 8× DP 金线 条/小时 | **8 × 3600/165 ≈ 175**（热/电 0.8× → ~140） | 单条仍 ~165 |
| 8× DP turbo 叠刀 | **8 × 3600/79 ≈ 365**（0.8× → ~292） | 单条仍 ~79 |

**必须自测才能填：** Phase 0 全部分段；768P turbo/TAE/TE HIT；SelfLift 两臂墙钟与人眼；U=2 BF16 480 与 768；`canAccessPeer` + uni/bi GB/s；NVFP4 去 424 s 后的 e2e。

---

## 7. 明确不做

- **U=8 当延迟主路径**（480p 已证伪；768P 回代仍亏）。
- **生产箱无回滚备份 / 无人 generate 检查 / 无 `iommu=pt` 就刷 ko。**
- **嵌 Python**（Comfy、xDiT、LightX2V、facok 插件当后端）。
- 自研 `nvidia.ko`；rtx-p2p；为 P2P 升官方驱动 / 换 CUDA / `NCCL_P2P_DISABLE`。
- FastWan 1.78 s、RH 12.2×、explainer 5–8 s 写成 H3 目标。
- NVFP4 CPU dequant 424 s 当加速档；金线 TAE；CFG 并行（`cfg=1`）；HOST_FWD=1；20 SM 划给 NCCL。
- 用 `--init-latent` / 480+FL2VA 8+8 冒充 SelfLift。
- TST 为了「完整」先重写 cuDNN。
- 把 480p 秒数横比成 768P 倍率。

---

## 8. 给执行者

1. **Phase 0** 打 768P 金线。没有它后面都是回代。  
2. **1A / 1C** 可并行；**1B SelfLift** 在 0 之后做外部 upscaler 先行。  
3. **Phase 2** 软件 U=2，先 480 后 768。  
4. **Phase 3** 按 §5：缺条件不装；有窗口只测一对卡。  
5. **Phase 4B DP** 随时可开，这是 8 卡吞吐极致。  

未改推理热路径。新增独立探针 `tools/h3_p2p_bw_probe.cu`。未 commit。密码未入库。

---

## 9. FreeToken 指针（2026-09-13，不新开 Phase）

> 调研备忘：[2026-09-13-h3c-freetoken-analysis.md](./2026-09-13-h3c-freetoken-analysis.md)。一手：[arXiv:2608.16157](https://arxiv.org/abs/2608.16157) · [FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken) `@9535656`（本地 `third_party/accel_survey/FreeToken/`，不进 git）。

**结论：** FreeToken 是消费卡上的 **MoE LLM serving**（专家 LRU + PCIe/CPU 带宽自适应 + NVFP4/MXFP4），不是视频 DiT，也不是减 token。公开数字是 tok/s，没有 5s · 1344×768。`docs/models.md` 的 MiniMax 是 **M2.5 LLM**，不是 H3。

**不要新开 Phase。** 唯一可借的是「NVFP4 保持 packed、GPU 上算，禁止 CPU requant」——已经是 **Phase 1C**。别碰 expert LRU / `q*` CPU 专家 / 语义 KV / 嵌他们的 Python。Ulysses / P2P / SelfLift / turbo4 路线不变。

---

## 10. TaoMate-H3 指针（2026-09-14，不新开 Phase）

> 调研备忘：[2026-09-13-h3c-taomate-h3-analysis.md](./2026-09-13-h3c-taomate-h3-analysis.md)。一手：[TaoLiveAIGC/TaoMate-H3](https://github.com/TaoLiveAIGC/TaoMate-H3) `@ccc1a70`（本地 `third_party/accel_survey/TaoMate-H3/`，gitignore，不 commit）· [HF LoRA](https://huggingface.co/TaoLiveAIGC/TaoMate-H3)。

**结论：** 公开表是 **8×H20 · 480×864 · 10s · TP2×U4** 的 Pure DiT，不是 5090、不是 1344×768。开源命令不能单卡、要 FA3/Hopper、默认一次性 VAE。用户 **27s/10s/768P/单卡5090+双采样** 是另一条栈；**双采样不在他们仓里**（SelfLift 同类）。

**不要新开 Phase。** 不要抄 TP2×U4。P0 仍是施工图已有刀：768P 注意力真正变便宜、SelfLift 质量、10s 再谈 chunk+KV。3-step LoRA 只许加速档 opt-in，金线不动；权不入库。
