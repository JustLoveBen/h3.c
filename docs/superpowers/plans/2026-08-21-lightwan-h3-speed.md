# LightWan22-A14B → h3.c CUDA MiniMax-H3 加速开发文档

日期：2026-08-21  
仓库：`h3.c`（纯 C/CUDA MiniMax-H3，不是 Python Diffusers / LightX2V）  
硬件：RTX 4090（Ada sm_89），单卡  
对照刀：`t2va_complex_5s_cafe` 官方 ids，**真 5.000s / 120 帧 @24fps**，`--seconds 5`（禁止 `--frames 17`）

p3 之后还剩什么、以及 Comfy [Latent Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) / [YCNodes](https://github.com/yichengup/ComfyUI-YCNodes-MiniMax-H3) 是否可迁，见 **[2026-08-21-remaining-quality-speed.md](./2026-08-21-remaining-quality-speed.md)**。结论：Upscaler 不跳过 480P DiT，不进质量默认。

Sol-Attn（kijai / 论文 / Saganaki22）与其余市面加速的分档，见 **[2026-08-21-h3-accel-survey.md](./2026-08-21-h3-accel-survey.md)**。Sol 预期只砍 DiT 数秒，不是 LightWan 式 80×。

## 0. 本仓现状（不要回滚）

已合入（不要回滚、不要空跑 p1/p2 同一配方）：

| 项 | 状态 | 证据 |
| --- | --- | --- |
| NVRTC / seq pad 64 / cuDNN 失败缓存 | 合入 | 无 `cudnn sdpa fallback seq=15973` |
| TE device-resident act | 合入 | `te_act: device_resident=1` |
| INT8 开窗默认 reserve 8192 | 合入 | 不设 env 也不 OOM |
| OpenMP deblock 共享 y,x,c 越界 | 已修 | p2 成片 |
| GEMM pregrow + M pad 16 | p2 | `int8_gemm: pad M 15973 → 15984` |
| P3 INT8 refit 1536 + freeze | **p3** | `pregrown=1 window=37 host_hits=0`；无 `make_room 29→16` |
| P3 VAE GPU F16 upcast | **p3** | `native_f16=1 gpu_dequant=1` load **1.0s**（p2 9.1s） |
| P4 MagCache A/B thresh=0.08 | **质量失败，默认仍关** | skips=3；latent MAE=0.85 vs p3；帧 RGB MAE 52–68 |

**墙钟对照**（同一 cafe ids，5.000s/120，cfg=1）：

| 刀 | te | dit | vae | total | DiT 窗 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 悬崖 cafe | 29.8 | 1148 | 260 | 24.1 min | flash 慢路径 | 已修 |
| p0 | 16.4 | 64.8 | 29.2 | **118.9s** | refit 29 | host dequant VAE |
| p2（对照基线） | 20.3 | 60.6 | 23.7 | **112.1s** | refit 19 | step2–8 ≈ **6.61s×8**；VAE `native_f16=0` dequant 8.0s；`host_hits=292` |
| **p3（质量默认）** | **16.4** | **50.5** | **15.6** | **90.5s** | **refit 37 pregrown=1** | 稳态 **5.33s×7**；VAE load **1.0s**；checksum=`0x3b3464adf6c39df2`；ffprobe **5.000s / 120**；**无 magcache** |
| p4 MagCache ON | 19.9 | 33.8 | 17.9 | **79.2s** | 37 | `--magcache-thresh 0.08` evals=5 skips=3；checksum=`0xc6b169de5325f6b8`；**帧级明显变差，不当默认** |
| p5 无损尝试（失败） | 18.1 | **54.9** | 15.3 | **98.8s** | 37 | pthread cuDNN prebuild；step1 11.0→12.4；checksum=p3；**比 p3 慢，已回退** |
| p5 关 prebuild | 19.2 | **51.2** | 20.3 | 100.2s | 37 | DiT 回到 51.2（≈p3 50.5）；VAE dequant 5.7s 是连跑页缓存被挤掉 |

p3 相对 p2：**−21.6s（112.1→90.5）**。p4 再砍 DiT 50.5→33.8s 但 **latent MAE=0.85、RGB MAE≈52–68/255**，主题/运动不可接受。质量路径仍是 p3；MagCache **默认关**。p5 无损刀未低于 90.5s。

循环目标仍是质量不降、**40–50s**。质量默认 **90.5s**，离 40–50s 还差 **~40–50s**。p4 的 79s **不计入达标**。无损 C/CUDA 已接近算力墙（稳态 5.33s×8）；再到 40s 必须跨红线（激进 skip / 官方 4-step 蒸馏 / 稀疏 attn / TinyVAE）——不冒充。

质量红线（全程不可破）：

- `cfg=1`（H3 已 CFG 蒸馏）
- INT8 ConvRot DiT（禁止 W4A8 当主题）
- fp16 VAE tile 256 + deblock 0.28（禁止关 deblock / TinyVAE 成片）
- 禁止 MagCache/reuse 冒充质量档、禁止短帧冒充 5s

---

## 1. LightWan22-A14B / LightX2V 检索摘要（2026-08-21 联网）

主项目：

- HF 模型卡：[lightx2v/LightWan2.2-A14B](https://huggingface.co/lightx2v/LightWan2.2-A14B)（与 [Wan2.2-NVFP4-Sparse](https://huggingface.co/lightx2v/Wan2.2-NVFP4-Sparse) 同族）
- 运行时：[ModelTC/LightX2V](https://github.com/ModelTC/lightx2v)
- 4-step LoRA 蒸馏：[lightx2v/Wan2.2-Lightning](https://huggingface.co/lightx2v/Wan2.2-Lightning) / [ModelTC/LightX2V-Wan2.2-Lightning](https://github.com/ModelTC/LightX2V-Wan2.2-Lightning)
- extreme 配置：`configs/wan22/extreme/wan_moe_t2v_distill_nvfp4_sparse_attn.json`
- 脚本：`scripts/wan22/extreme/run_wan22_moe_t2v_extreme.sh`（含 `--warmup`）

公开数字（**RTX 5090 / Blackwell**，Wan2.2-T2V-14B，**不要当成 H3 4090 同款**）：

| 方法 | 分辨率 | NFE | E2E | 相对基线 |
| --- | --- | --- | --- | --- |
| Wan2.2-T2V-14B | 480p | 40 | 734s | 1.0× |
| LightWan2.2-A14B | 480p | 4 | **9.1s** | 80.7× |
| 同上 | 720p | 4 | 22.5s | 118.7× |

80× 来自三件事叠乘：**40→4 步蒸馏 + NVFP4 QAT + 稀疏注意力**，而且绑死 Blackwell。extreme JSON 原文要点：

```json
"infer_steps": 4,
"self_attn_1_type": "dynamic_sparse_attn",
"dynamic_sparse_attn_setting": { "sparsity_ratio": 0.9, "operator": "sage2" },
"cross_attn_1_type": "sage_attn2",
"enable_cfg": false,
"cpu_offload": false,
"t5_cpu_offload": true,
"vae_cpu_offload": false,
"dit_quant_scheme": "nvfp4",
"denoising_step_list": [1000, 750, 500, 250]
```

LightX2V 工程菜单（文档 / examples / issues）：

| 方向 | 来源 |
| --- | --- |
| 量化 FP8/INT8/NVFP4/INT4-Marlin | [quantization.html](https://lightx2v-en.readthedocs.io/en/latest/method_tutorials/quantization.html) |
| TeaCache / TaylorSeer / AdaCache / CustomCache | [cache_source.md](https://github.com/ModelTC/LightX2V/blob/main/docs/EN/source/method_tutorials/cache_source.md)；论文 [TeaCache](https://arxiv.org/abs/2411.19108) |
| MagCache（Wan2.2 1.5–2×） | [MagCache4Wan2.2](https://github.com/Zehong-Ma/MagCache/blob/main/MagCache4Wan2.2/README.md)；[arxiv 2506.09045](https://arxiv.org/html/2506.09045v1) |
| SageAttention 2 / 2++ | [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention)；LightX2V 默认推荐 `sage_attn2` |
| 动态稀疏 / SLA / Radial | extreme `sparsity_ratio=0.9`；[SLA](https://arxiv.org/html/2509.24006v1)；[Radial Attention](https://github.com/mit-han-lab/radial-attention) |
| CFG skip / `enable_cfg=false` | extreme JSON；Hunyuan/Wan 蒸馏档必须 cfg=1 |
| LightTAE / 轻量 VAE | Hunyuan LightTAE；官方称细细节会糊，适合预览 |
| `torch.compile` / MagiCompiler | [LightX2V-MagiCompiler](https://github.com/SandAI-org/LightX2V-MagiCompiler) |
| 序列并行 Ulysses/Ring、CFG 并行 | [parallel.html](https://lightx2v-en.readthedocs.io/en/latest/method_tutorials/parallel.html) |
| CPU/磁盘 offload | [offload.html](https://lightx2v-en.readthedocs.io/en/latest/method_tutorials/offload.html)；**速度路径应关 DiT offload** |
| 蒸馏+cache 冲突 | [LightX2V#579](https://github.com/ModelTC/LightX2V/issues/579)：4 步蒸馏后再开 TeaCache **不会再跳步** |

维护者在 #579 对「蒸馏后再加速」的结论：单卡只剩量化、降分辨率、Flash/Sage/稀疏注意力、算子融合；多卡才是 Ulysses + CFG 并行。

---

## 2. 方案表：原理 / 对 h3.c 是否适用 / 质量风险 / 优先级

| ID | LightWan / LightX2V 方案 | 原理 | 对 h3.c（C CUDA + INT8 DiT + fp16 VAE） | 质量风险 | 优先级 |
| --- | --- | --- | --- | --- | --- |
| L1 | 4-step 蒸馏（high/low noise expert） | 把 40 NFE 蒸馏成 4；CFG 也蒸馏掉 | **不适用**。H3 权重已是 8-step CFG-distilled，没有 Wan2.2 MoE 的 high/low ckpt，不能把 `--steps 8` 改成 4 冒充 | 未蒸馏强行 4 步会崩主题/运动 | **禁用** |
| L2 | NVFP4 QAT | Blackwell FP4 Tensor Core + 量化感知蒸馏 | **不适用**。4090=Ada；红线锁 INT8 ConvRot；NVFP4 是 TE 路径已有，不能换 DiT | 换量化格式 = 主题回归 | **禁用** |
| L3 | 动态稀疏 attn 90% + Sage2 | 跳过 O(N²) 中「不重要」块 | **不可默认**。本仓 packed 是 text+audio+video 统一自注意力；Canvas 已禁 Sol-Attn/稀疏当终稿；本仓 sage2 满 seq 比 cuDNN 慢 ~60× | 人脸/对白细部丢失；SLA 论文也要微调 | **P2 研究，默认关** |
| L4 | SageAttention 替换 FA2 | QK INT8 / PV FP8 MMA | **不可默认**。Ada 上社区相对 FA2 可 ~2×，但本仓上限是 cuDNN fused，sage2 核 9s/层 ≫ cuDNN 150ms | INT8/FP8 attn 微损 | **保持 opt-in，禁止改默认** |
| L5 | `enable_cfg=false` | 蒸馏模型单路 | **已做**。质量档 cfg=1 | cfg=5 + 短 ids 过曝 | **保持** |
| L6 | TeaCache / MagCache / FBCache | 步间 residual 相似则 skip DiT | **可接线，不可当质量默认**。H3 已 8-step 蒸馏，#579 说蒸馏后再 cache 几乎不跳；本仓 `--magcache-thresh` 已存在但主题默认关；EasyCache 伤 H3 音频（Comfy #15326） | 轨迹漂移、对白节拍乱 | **P1 A/B，默认关** |
| L7 | 序列并行 Ulysses/Ring | 多卡切 seq | **不适用**。单卡 4090 | — | **禁用** |
| L8 | CFG 并行 | 双路 CFG 分到两卡 | **不适用**。cfg=1 没有第二路 | — | **禁用** |
| L9 | torch.compile / MagiCompiler | 图级融合、减少 launch | **可迁移思路，不能搬 Python**。本仓已有 `H3_DIT_W4A8_CUDA_GRAPH`，但 cuDNN SDPA 不可捕获所以默认关。等价物是 **持久 workspace + 少 sync + 少 host printf** | 捕获失败会算错 | **P0 工程等价** |
| L10 | 轻量 VAE / LightTAE | 蒸馏 decoder，亚秒预览 | **禁止成片**。红线 fp16 tile+deblock | waffle / 糊 | **仅草稿档** |
| L11 | VAE/DiT CPU offload | 显存不够时换速度 | **反面教材**。extreme 明确 `cpu_offload=false`（要吞吐就别 offload）。本仓 INT8 窗 29→16 就是一种 offload，会把 DiT 拉慢 | 无损但更慢 | **P0：禁止无谓换页** |
| L12 | T5 CPU offload | 把文本编码器丢 CPU | **不要学**。本仓要的是 TE 更快，不是给 DiT 腾显存；TE 已 device-resident act | TE 更慢 | **禁用** |
| L13 | 量化 INT8/FP8 DiT | 带宽↓、Tensor Core | **已做 INT8 ConvRot**。不要换 W4A8 / NVFP4 DiT | W4A8 主题崩 | **保持** |
| L14 | 算子融合（MXFP8 modulate+FFN） | 把 AdaLN/quant/GEMM 合成一核 | **可借鉴，但 SM120 核不能用**。4090 可做 fp16/INT8 侧小融合；大改架构不值 | 数值需对照 | **P1** |
| L15 | `--warmup` | 先跑一步把 kernel/图建好 | **可迁移**。cafe step1=13.0s、step2–8≈6.67s，差额主要是 cuDNN 建图 + 首窗 unpack。不要为此 shrink 权窗 | 无损 | **P0：保住 window，别为 warmup 换页** |
| L16 | 分辨率 / 帧数技巧 | 480p、81 帧、text_len=512 | **不要截断 574 tok、不要改 864×480、不要短帧** | 假 5s / 丢 prompt | **禁用** |
| L17 | 480p 而非 720p | 维护者 #579：单卡继续加速就别跑 720 | **已做** 864×480 | — | **保持** |

---

## 3. 可迁移项（真能接到本仓架构）

只列「C CUDA + 现有 INT8 DiT + fp16 VAE」能落地、且不破红线的：

### P0（立刻做，不大改架构）

对应 LightX2V：**关 offload 保吞吐（L11）+ warmup 不要破坏驻留（L15）+ VAE 留 GPU（L10 的反面：全尺寸 VAE 但不要 host 反量化）+ 减 host 间隙（L9）**。

1. **INT8 权窗禁止 step1 后 29→16**  
   p0：`refit window=29` → step1 13.0s → `make_room dropped=13 window=16` → step2–8 各 ~6.67s。  
   根因：`ensure_block_ws` 先 `cudaFree` 再 `cudaMalloc`；cuDNN 建图后碎片化导致第二次 malloc 失败，于是 `make_room(need+512MiB≈8.65GiB)` 主动丢掉 13 层。  
   动作：malloc-新块-成功后再 free 旧块；首次分配加 64MiB slack；**已有 WS 时禁止为「再挤 8GiB 空闲」而 evict**；第一次完整 step 成功后 freeze window（可用 `H3_DIT_INT8_ALLOW_SHRINK=1` 回退）。  
   预期：DiT 64.8s → **~50–58s**（step1 仍含 cuDNN 建图；稳态不再付 16 层换页）。无损。

2. **VAE fp16 权重走 GPU upcast，不要 8s host dequant**  
   p0：`vae_ckpt: F16 native` 但 `gpu_dequant=0 native_f16=0 dequant_ms=8001`。  
   根因：`cudaHostRegister` 失败（`hostreg=0`）后仍在 **non-blocking stream** 上对未 pin 的 mmap 做 `cudaMemcpyAsync`（不合法），`load_linear_f16_gpu` 失败退回 host F16→F32。  
   动作：hostreg 失败则改用 default stream / 同步 memcpy；失败打一条原因。  
   预期：VAE load 9.1s → **~1–3s**；VAE 墙钟 29.2s → **~21–23s**。数学仍是 fp16 W → F32 GEMM + tile/deblock。

3. **热路径少刷屏**  
   TE 每层 `fprintf+fflush`×2、VAE 每 tile×chunk 打 `vae_temporal_chunk`。LightX2V warmup/compile 的工程等价：减少 host 间隙。默认 quiet，`H3_NVFP4_VERBOSE=1` / `H3_VAE_VERBOSE=1` 恢复。  
   预期：TE 16.4s 里可能有 0.5–1s 日志气泡；主要是日志可读性。

### P1（P0 达标后再 A/B，有回退）

- **保守 FirstBlockCache / MagCache**（H3 Safe thresh≈0.08，最多连续 2 hit，音视频分账）。本仓已有 `--magcache-thresh`。8-step 蒸馏空间小（#579）。失败即关。
- TE 权常驻：现在每 linear 仍 H2D packed + device dequant；把 NVFP4 packed 常驻 GPU 可再砍 TE（目标 8–12s）。契约 cos 必须 ≈1。
- VAE decode 核：tile 内 SDPA 已 cuDNN；load 摊销后看 blocks 9.3s 能否再融。
- 10s INT8 窗不要掉到 1（Canvas A6；本刀是 5s）。

### P2（需要新核 / 新权重，不在本轮）

- 真 MMA Sage2++ 且 **核速度 ≤ cuDNN** 才允许 opt-in。
- Ada 上的稀疏 attn 研究档（非成片）。
- 若未来有 H3 官方 4-step 蒸馏权重，再单独立档（不是把现在 8 步砍成 4）。

---

## 4. 不可用项（明确不要做）

| 项 | 原因 |
| --- | --- |
| 把 `--steps` 改成 4 | 没有 H3 4-step 蒸馏权重；LightWan 的 80× 主力在这里 |
| DiT 改 NVFP4 / W4A8 / FP8 当质量路径 | 4090 无 NVFP4 吞吐故事；W4A8 本仓主题崩 |
| 默认 Sage2 / 稀疏 90% | 本仓 sage 更慢；稀疏伤细部 |
| MagCache/reuse 开在质量默认 | 红线；蒸馏后再 skip 收益小且伤音频 |
| TinyVAE / 关 deblock / 无 tile | waffle |
| `--frames 17` 冒充 5s | mux 22 帧 ≈0.92s |
| cfg=5 | 过曝指纹 |
| 截断 cafe 574 tok | 复杂 prompt 质量 |
| 多卡 Ulysses / CFG parallel | 单卡 |
| T5 CPU offload | 让 TE 更慢 |
| 为仿 LightX2V 把生成栈改成 Python Diffusers | 架构禁止 |

**关于 40–50s：** LightWan 5090 上 480p 9.1s 不能外推到 H3 4090。H3 已是 8-step + INT8 + cuDNN。p2=112s = TE 20 + DiT 61（6.6s×8）+ VAE 24。无损 C/CUDA 乐观 **85–95s**。40–50s 必须跨红线（MagCache 激进 skip / 官方 4-step 蒸馏权重 / 稀疏 attn / TinyVAE / 假时长）——本轮不做。

---

## 5. 本仓分步实施

### 步 0 — 对照（已完成，勿重跑）

- p0：118.9s；**p2 新基线 112.1s**（`t2va_complex_5s_cafe_p2/`）
- 不要再跑 p1/p2 同一配方

### 步 1 — P3 代码（相对 p2 的新项）— **已落地，见 p3 墙钟**

对应 LightX2V：**关无谓 offload（L11）** + **VAE 留 GPU、不要 host 反量化** + **减 host 间隙（L9）**。

1. **`h3_w4a8_cache_refit_after_workspace`**：**已做。** pregrown 时 reserve **1536MiB**；`open_window_cap` 锁开窗上限 41。p3：`window=37 pregrown=1 host_hits=0`，稳态 5.33s/步。`H3_W4A8_CACHE_REFIT_RESERVE_MIB` 仍可覆盖。
2. **VAE fp16 GPU upcast**：**已做。** hostreg 失败清 CUDA last error + pinned staging + default-stream H2D；kernel 后 `cudaDeviceSynchronize`。p3：`native_f16=1 gpu_dequant=1 dequant_ms=970`（p2 ≈8000）。decode 仍走 F16 TC（与 p2 同路径；`native_float_w` 不再误关 TC）。`H3_VAE_F32_GEMM=1` 可回退纯 F32 GEMM。
3. **WS grow-safe + freeze window**：**已做。** p3：`dit_int8_cache_freeze: window=37`；无 `make_room ... window=16`。
4. **TE/VAE 热路径少刷屏**：**已做。** p3 无 `te_layer_begin` 刷屏；`H3_NVFP4_VERBOSE=1` / `H3_VAE_VERBOSE=1` 恢复。

不改：量化格式、cfg、步数、VAE deblock、SDPA 默认 cuDNN、magcache 默认。

### 步 3 — 测 cafe_p3 — **已完成 2026-08-21 06:06–06:08**

目录：`out/verify_480p_quality_full/t2va_complex_5s_cafe_p3/`（远程 `/tmp/h3_verify_480p_quality_full/t2va_complex_5s_cafe_p3/`）。

- binary `2026-08-21 06:06:57`；ids 与 p2 同一份 574 tok；`--seconds 5 --cfg 1`；无 magcache / `--frames 17`
- `phase_ms: te=16449.2 dit=50536.6 vae=15648.4 total=90515.6 dit_evals=8 dit_skips=0`
- ffprobe：`duration=5.000000` `nb_frames=120` `avg_frame_rate=24/1`
- DiT latent checksum 与 p2 相同：`0x3b3464adf6c39df2`（DiT 数值未漂）
- 验收：window=37≥30、VAE load 1.0s、total 90.5s < 112.1s、DiT 50.5s ≤52s。**未到 40–50s（本轮不承诺）**

### 步 2 — 编译到 4090 `src_sync` — **已做**

`haiyang@192.168.0.38:/home/haiyang/h3_c/src_sync`，binary `2026-08-21 06:06:57 +0800`。

禁止：`--frames 17`、`--magcache-thresh`、`--reuse`、sage2、`H3_VAE_NO_SPATIAL_TILE`、设 `H3_W4A8_CACHE_REFIT_RESERVE_MIB=8192`（会退回 p2 窗）。

### 步 4 — P1 MagCache A/B — **已测，质量失败，默认仍关**

`t2va_complex_5s_cafe_p4/`（`--magcache-thresh 0.08`，max_consec=2，音视频分账；cfg=1 / INT8 / tile+deblock 未改）。对照 p3（关，不重跑）。

- p4：`phase_ms te=19917 dit=33788 vae=17891 total=79205 dit_evals=5 dit_skips=3`；ffprobe 5.000s/120
- vs p3 latent MAE=**0.85** max_abs=6.06；帧 RGB MAE t=0/2.0/4.9s = **68.5 / 66.2 / 52.2**
- 结论：8-step 蒸馏后再 skip 伤轨迹（LightX2V #579）。**不把 MagCache 开进质量默认。** 代码侧保留 opt-in + `mag_fresh`（大 delta 后禁止立刻 skip）+ 连续最多 2 hit。

下一步仍是无损：TE packed 常驻、VAE blocks 融合。不要把 p4 79s 当质量墙钟。

### 步 5 — P5 无损（融合 FFN / cuDNN 多槽 / TE overlap）— **未快于 p3，已回退变慢项**

目录：`out/verify_480p_quality_full/t2va_complex_5s_cafe_p5/`（远程 `/tmp/h3_verify_480p_quality_full/t2va_complex_5s_cafe_p5/`）。同一 cafe ids，`--seconds 5`，magcache=0。

| 变体 | te | dit | vae | total | 说明 |
| --- | --- | --- | --- | --- | --- |
| p3 质量基线 | 16.4 | **50.5** | 15.6 | **90.5s** | step1 **11.0s**；稳态 5.33s；VAE load 1.0s |
| p5 pthread prebuild | 18.1 | **54.9** | 15.3 | **98.8s** | `dit_sdpa_prebuild ok=1`；step1 **12.4s**；checksum=`0x3b3464adf6c39df2` |
| p5 关 prebuild | 19.2 | **51.2** | 20.3 | 100.2s | step1 11.7s；VAE `dequant_ms=5667`（连跑挤掉页缓存） |

**DiT 50.5→54.9 的原因：** `te_open` 期间 pthread 预编译 DiT cuDNN 图。图一度是 `thread_local`（主线程看不见）；改成进程级 static 后主线程能命中，但绑在 worker 的 cudnn handle 上，step1 仍从 11.0s 升到 12.4s。8-block TE-overlap GPU prefetch 会抢 TE。15GB `cudaHostRegister` 失败后每个 TE linear 重试，多约 2s。

**已回退：** pthread SDPA prebuild、TE 期间 8-block prefetch、VAE `posix_fadvise` 与 DiT 重叠、FFN `h3_cuda_ffn_swiglu_i8_dev` 包装（改回 `swiglu_dev` + `gemm_i8`）。**保留：** p3 权窗 37/pregrown、VAE fp16 GPU upcast、cuDNN 多槽（TE/DiT/VAE 共存）、超大 mmap 不重试 hostreg、`release_heavy_workspace` 不 reset SDPA 图。

拉回的 p5 mp4：ffprobe **5.000s / 120**；latent checksum 与 p3 相同。

**8-step 4090 无损下限（实测）：** TE ≈16.4s（15.7GB 无法 pin）；DiT ≈50.5s（step1 unpack 37 块 INT8 不能与 TE overlap，否则 24GB OOM；稳态 5.33s×7；cuDNN SDPA 上 CUDA graph 默认关）；VAE ≈15.6s（tile+deblock）。**合计地板 ≈ 82–90s。** p3 90.5s 已贴这条带。到 40–50s 必须跨红线（MagCache / 4-step 蒸馏 / 稀疏 attn / TinyVAE）。

---

## 6. 验收

| 项 | 标准 |
| --- | --- |
| ids | 与 cafe p2 同一份 `prompt.ids`（574 tok，不截断） |
| 时长 | `ffprobe` **5.000s / 120 帧** @24fps |
| 质量 | cfg=1、INT8 ConvRot、fp16 VAE tile+deblock 0.28；目视主题/对白节拍不劣于 p2 |
| 日志 | 无 `cudnn sdpa fallback seq=15973`；无 magcache skip；`dit_evals=8 dit_skips=0` |
| DiT 窗 | `pregrown=1` 且 window≥30（p2 是 19）；不应再 `make_room ... window=16` |
| VAE load | `native_f16=1` 或 `gpu_dequant=1`；`dequant_ms` 不再是 ~8000 |
| 墙钟 | **p3=90.5s** < p2 112.1s；DiT **50.5s**（稳态 5.33s）；VAE load **1.0s**。**40–50s 不是本轮承诺** |
| 对照短 prompt | 不回退橙猫质量档路径 |

阶段数字从 `phase_ms: te=... dit=... vae=... total=...` 读取。

---

## 7. 检索 URL 清单

- https://huggingface.co/lightx2v/LightWan2.2-A14B
- https://huggingface.co/lightx2v/Wan2.2-NVFP4-Sparse
- https://github.com/ModelTC/lightx2v
- https://github.com/ModelTC/LightX2V/blob/main/configs/wan22/extreme/wan_moe_t2v_distill_nvfp4_sparse_attn.json
- https://github.com/ModelTC/LightX2V/issues/579
- https://lightx2v-en.readthedocs.io/en/latest/method_tutorials/quantization.html
- https://lightx2v-en.readthedocs.io/en/latest/method_tutorials/parallel.html
- https://lightx2v-en.readthedocs.io/en/latest/method_tutorials/offload.html
- https://github.com/ModelTC/LightX2V/blob/main/docs/EN/source/method_tutorials/cache_source.md
- https://github.com/Zehong-Ma/MagCache/blob/main/MagCache4Wan2.2/README.md
- https://arxiv.org/abs/2411.19108 （TeaCache）
- https://arxiv.org/html/2506.09045v1 （MagCache）
- https://github.com/thu-ml/SageAttention
- https://arxiv.org/html/2509.24006v1 （SLA）
- https://github.com/mit-han-lab/radial-attention
- https://huggingface.co/lightx2v/Wan2.2-Lightning
- 本仓 Canvas：`h3c-480p-quality-speed-plan.canvas.tsx`（A1–A5 已在 p0 落地，本文件不重复做 pad/nvrtc）
