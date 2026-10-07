# H3.c 分模块能力/性能优化方案（VAE 优先）

> **For agentic workers:** 本文件是 **枢纽方案**，不是一次把所有模块做完的施工单。REQUIRED：执行某一模块时用 `superpowers:subagent-driven-development` 或 `executing-plans`，且 **先做 VAE**。禁止新开 5s/10s cafe 去冲 27s。禁止改金线默认。禁止 commit。密码不入库。未授权不要改推理；读接口可以。

**Goal:** 把加载 / TE / DiT / SelfLift / VAE / mux 各自做到「能力上限 + 性能上限」可独立验收；在 VAE 未追上 Comfy 质量合同与秒数之前，冻结整体 e2e 对照。

**Architecture:** 一次生成 = 权进内存 → TE 文本 → DiT 去噪（± LoRA / lift）→ 视频 VAE decode → mux。Amdahl 定律：任一模块比对照栈慢 4×，整体对比没有意义。H3 视频 VAE 是 **36 层 ViT3D**，不是 Wan 那种 CausalConv3D；网上「给 Wan 剪通道 / Triton 融 conv」不能直接套。

**Tech Stack:** 本仓 C/CUDA（`h3_vae_int8_cuda.c`、`h3_gpu_cuda.cu`、`h3_w4a8_cache.c`、`h3_te_cache.c`、`h3_ffmpeg.c`）。对照：Comfy `comfy/ldm/minimax/vae.py` + comfy-kitchen；官方 tile `256/64`；TAE 仅预览。

**Spec / 尺子：** [768P 评测协议](./2026-09-13-h3c-768p-eval-protocol.md)（5s · **1344×768**）· [TaoMate 分析](./2026-09-13-h3c-taomate-h3-analysis.md) · [SelfLift](./2026-09-13-h3c-selflift-comfy-analysis.md) · [模型格式](./2026-09-13-h3c-model-format-5090.md) · [极致施工图](./2026-09-13-h3c-5090-extreme-full-plan.md)

## Global Constraints

- 金线默认：U=1、8-step、`sdpa=cudnn`、官方 VAE、无 LoRA、`--te-cache` 关。
- 本仓官方 768P 横屏是 **1344×768**。`H3_MAX_PIXELS` **拒收 1376×768**。下文「热 lift」秒数是 **1344×768**；1376 只作产品画布备注，不是已测尺子。
- 不要新开 5s/10s cafe 冲用户 27s。本文件只整理方案。
- 有损刀（TAE / sage3 / 少步 / NVFP4 DiT / lift）进加速档，不进 `gold`。
- 密码、HF token、远端登录信息不入库。

---

## 0. 为什么整体对比暂缓

用户对照是「单卡 5090 + 双采样 · 10s 768P ≈ 27s」。本仓热 lift **5s · 1344×768** 已是 **47–50s 全含**。把 5s 乘 2 再和 10s 比、或拿 480p cafe 78s 去解释 27s，都会撒谎。

更关键的是 **模块没有各自到顶**：

| 段 | 本仓 5s 768P 已测 | 对照栈大致位置 | 若先比整体会怎样 |
| --- | ---: | --- | --- |
| 官方 VAE | **23.57 s**（金线）/ **~18 s**（196 窗 C+cuDNN） | Comfy 同画布量级 **~4 s** | VAE 一项就吃掉 27s 预算的大半 |
| DiT（lift） | 低 3 + 高 1 ≈ **22 s**；768P sage3 `hits=0 fallback=200` @ `seq≈37788` | 分块 Q + 窗口 K / FA3（Hopper） | 注意力仍付满序列，少步也追不上 |
| TAE 快档 | **7.03 s** 但彩色闪 | 官方仍用完整 VAE | 用 TAE 去「赢秒数」等于换一条质量尺 |
| TE | MISS **~13 s**；HIT **~0**（host_stage 会记进 DiT） | 常驻 / cache | 热路径已接近上限，不是 27s 主因 |
| host_stage | mmap+pin **1.26 s**（旧 13.2 s） | 常驻 | 已不是主墙 |
| 多卡 Ulysses | U=2/8 **更慢** | 他们的 8 卡表是另一条公路 | 再叠多卡只会把脏数字乘开 |

**结论：** 先把 **VAE 做到与 Comfy 同合同、同量级秒数**，再让 DiT 注意力在 768P 上真正变便宜，最后才谈 10s / 27s / 多卡。否则「整体慢 2×」分不清是 decode 慢、attn 死、还是口径不同。

---

## 1. 已踩坑（本仓事实，禁止再踩）

| 坑 | 现象 | 合同 |
| --- | --- | --- |
| **整幅低频 RoPE** | 用全画布 `[-1,1]` 当 denom，或 `H3_VAE_NO_SPATIAL_TILE` 整幅 decode | **16px ViT patch waffle**（`b16≈2.6` vs tile+deblock `≈1.04`）。`h3_vae_int8.h`：Comfy `create_token_ids` 用 **当前 tensor H/W（tile-local）**；canvas denom 是 OOD。 |
| **F16 cuDNN 整幅** | `seq~29k` 走 F16 fused SDPA | 同样 16px waffle。`h3_gpu_cuda_sdpa_cudnn.cpp`：必须 **真 F32 且 TF32 off**。 |
| **tile > 320 / 512px** | `H3_VAE_TILE_PIXELS=512` | clip2 **512 waffle**。默认锁 **256–320**，overlap **64** 且 `%16==0`。官方/Comfy 默认 **256 / 64**。[MiniMax-H3#46](https://github.com/MiniMax-AI/MiniMax-H3/issues/46)：乱改 tile/overlap → 阴影和方块。 |
| **TAE 成片** | `h3_tae*` / taeh3 | **彩色闪 / 纹理抹 / 色漂**。Kijai：[MiniMax-H3-TAE](https://huggingface.co/Kijai/MiniMax-H3-TAE)「not the greatest」；madebyollin taeh3 也是 **preview**。金线禁止。 |
| **naive flash** | `H3_VAE_FLASH_SDPA=1` 或无 F32 engine 退手写 flash | **~305 s** 级墙钟。禁止当产品路径。 |
| **196 窗 C+cuDNN** | 本仓 tile 窗 + C GEMM + cuDNN | **~18 s** vs **Comfy ~4 s**（同 5s 768P 量级）。差在核，不在「再减窗」。减窗若不守 RoPE 合同会回到 waffle。 |

代码锚点：

- RoPE 合同：`h3_vae_int8.h` `prepare_rope` / `prepare_rope_ex`；`tests/test_h3_vae_rope.c`
- 整幅 F16 waffle：`h3_gpu_cuda.cu`（`VAE full-frame: F16 cuDNN waffles`）、`h3_vae_int8_cuda.c` 同行注释
- 默认回 tile：`tools/h3_cuda_generate.c`（「Full-frame decode is faster but leaves a 16px VAE-patch waffle」）
- 权复用：`h3_vae_int8_cuda.c`（每 tile 重读盘曾 **~6s × N → ~90s**）

---

## 2. 网上方案（2026-09-15 检索）与对本仓的含义

H3 VisualVAE = **因果 3D CNN encoder + 36 层 ViT3D decoder**，f16t4d24，官方 `clip_length=17`、`tile_size=256`、`tile_overlap_min=64`。[MiniMax-H3 README](https://github.com/MiniMax-AI/MiniMax-H3/) · Comfy 实现 [comfy/ldm/minimax/vae.py](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/minimax/vae.py)

### 2.1 排行：最快且仍守质量合同（decode 主路径）

| 序 | 候选 | 来源 | 对本仓 | 质量 |
| ---: | --- | --- | --- | --- |
| **1** | **comfy-kitchen 解码器核**（`rms_rope_split_half` + `int8_attention` 或 `optimized_attention` + `linear_input_act` RMS/SwiGLU）+ **256/64 tile-local RoPE** + 每行最多 4 tile 一 batch | [comfy-kitchen](https://github.com/comfy-org/comfy-kitchen) · vae.py Attention/FFN | **P0。** 这就是 Comfy ~4s 的核，不是另训一个 VAE。 | 与官方 decode 同合同；INT8 attn 仅当权已是 QuantizedTensor |
| **2** | **融合 36 层 transformer**（RMS+QKV+RoPE+SDPA+gated FFN 少 launch）打在 **tile seq**（256px → latent 16×16×Tchunk ≈ 1.8k token，不是 29k） | 同上 + vLLM-omni [RFC #5948](https://github.com/vllm-project/vllm-omni/issues/5948)（stack tiles / compile 固定 tile 形状） | **P0/P1。** 本仓已有 `vae_kitchen_sdpa` batched cuDNN 和层 graph 雏形，但 GEMM/RoPE 仍碎。 | 数值对齐 tile 参考 |
| **3** | **固定 tile 形状 TensorRT / ONNX** | 通用 TRT；**不是** TRT-LLM 的 Wan 平行 VAE | **P1。** 形状桶必须钉死 256px×clip。H3 是 ViT 注意力，不是 Wan conv。 | 需 vs Comfy 同 latent |
| **4** | **减窗但不伤 RoPE** | [H3#46](https://github.com/MiniMax-AI/MiniMax-H3/issues/46)：乱改会阴影 | **仅当** denom 仍是 **当前 tile H/W**、边界 `%16`、overlap 64 交叉淡化。禁止靠「更小窗」补核慢。 | waffle 红线 |
| **5** | 时间维 chunked I/O（17 帧 clip，峰值与时长无关） | Comfy [PR #15446](https://github.com/Comfy-Org/ComfyUI/pull/15446)（175×1344×768 decode **14.6→13.9s**，主收益是显存不是秒）· [PR #15456](https://github.com/Comfy-Org/ComfyUI/pull/15456) decode 前腾 DiT | **P0 加载/常驻。** 本仓已「DiT 释放再 VAE」；还要避免全片 RGB 双份 F32。 | 位级/像素级对齐 |
| **6** | 多卡 parallel / tile-parallel VAE | [TRT-LLM Wan parallel VAE](https://github.com/NVIDIA/TensorRT-LLM/pull/11875) · [LTX-2 tile-parallel](https://github.com/NVIDIA/TensorRT-LLM/pull/15753) · [NVL72 文](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog25_Scaling_Video_Generation_Across_NVL72_Rack_with_TensorRT-LLM.html) | **附录冻结。** 那些是 conv UNet 切空间 + halo；H3 是 ViT 全局 attn。单卡核没到 4s 之前切卡没有意义。 | — |

### 2.2 明确不是「最快最好成片 VAE」

| 方案 | 链接 | 为什么不做金线 |
| --- | --- | --- |
| TAEHV / taeh3 / MiniMax-H3-TAE | [taehv](https://github.com/madebyollin/taehv) · [Kijai TAE](https://huggingface.co/Kijai/MiniMax-H3-TAE) · [PreviewOverride](https://github.com/simsim9-stack/ComfyUI-MiniMaxH3-PreviewOverride)（holdout PSNR ≈ **25.5 dB**） | 作者定位 preview；TAESD 系会 **逐帧造细节 → 闪**。本仓已见彩色闪。 |
| KJNodes PatchTritonVAE | [triton_vae.py](https://github.com/kijai/ComfyUI-KJNodes/blob/main/nodes/triton_vae.py) | **不支持 MiniMax H3**（只 Wan/KL/LTX conv）。 |
| Flash-VAED / LightVAE / Turbo-VAED | [Flash-VAED](https://arxiv.org/html/2602.19161v1) · [Turbo-VAED](https://arxiv.org/abs/2508.09136v1) | 剪的是 **Wan/LTX CausalConv3D**。H3 解码器是 ViT，剪 conv 通道对不上。 |
| 通用 Tiled VAE（空间+时间一起切） | Comfy `VAEDecodeTiled` | H3 已内部空间 tile；再切时间会闪。[Wan 空间-only tile](https://github.com/crmbz0r/ComfyUI_Wan22Blockswap/blob/main/vae_decode.py) 的教训：时间维必须整段。 |
| 第三方「full-overlap 256/64」节点 | [comfy.icu H3 VAE Decode](https://comfy.icu/node/TuringUtilsMiniMaxH3VideoVAEDecode) | 合同与官方一致时可用作对照；W8A8 是 **QK attn 不是 VAE 权**。不要嵌 Python 进 C 生产。 |

### 2.3 F16 vs F32 注意力（VAE）

- **Tile 内（seq ~1–2k）:** Comfy 注释「fp16 keeps exact flash」；INT8 权才走 `ck.int8_attention`。本仓 tile 路径已用 F16 RoPE+cuDNN。
- **整幅（seq ~29k）:** 本仓实测 F16 cuDNN → waffle；F32 TF32-off 才稳，且极慢。
- 图像 SDXL 的「F16 溢出变黑」与 H3 waffle **不是同一 bug**（[Photoroom 文](https://dev.to/elise_moreau/the-sdxl-vae-overflow-that-decoded-black-images-in-fp16-46g6)）。
- vLLM-omni [RFC #5948](https://github.com/vllm-project/vllm-omni/issues/5948) 把 **输出累加器** 收到 fp16：PSNR 很高但 **墙钟几乎不动**（chunk 仍 fp32）。不能当加速银弹。

### 2.4 加载 / 常驻 / pin

- SGLang：[VAE 权直接流上 GPU](https://github.com/sgl-project/sglang/pull/37004)（meta 建模块，safetensors tensor 直赋 CUDA，避免完整 CPU state dict）。
- vLLM：[mmap + `cudaHostRegister`](https://github.com/vllm-project/vllm/issues/42361)（零拷 pin；注册页 unevictable）。
- PyTorch：`PYTORCH_CUDA_ALLOC_CONF=pinned_reserve_segment_size_mb:…` 预留 pin 段，避免反复 `cudaHostAlloc`。
- Comfy：decode 前 `free_memory` 赶走 DiT，VAE patcher 留下（PR 15456）。

本仓 DiT 已做 mmap 别名 + pin（host_stage **13.2 → 1.26 s**）。VAE 有 mmap store 与 `H3_VAE_WEIGHT_CACHE` / `H3_VAE_CUDA_W_RESIDENT`，但 **F16 W→F32 GEMM** 与 **未融合 36 层** 仍是算力墙，不是只读盘。

### 2.5 VDN（Video DeltaNet）——不是 VAE，5090 不成片 decode 主路径

检索「VDN VAE / Video Decoder Network / kitchen VAE」后：**VDN = Video DeltaNet（OpenVDN）**，不是 Video Decoder Network，也不是厨房核。它改的是 **DiT video-video 注意力**（窗 softmax + Video Delta 线性支 + 小 LoRA），**官方视频 VAE 原样留下**。Comfy 节点原文：conditioning / LoRA / sampler / **VAE decode 节点不变**。

一手链接：

- 博客 / 架构：https://openvdn.github.io/ · https://openvdn.github.io/h3-vda-blog.html
- 代码：https://github.com/OpenVDN/vdn-minimax-h3
- 权重：https://huggingface.co/OpenVDN/vdn-minimax-h3（`h3-base/` 含官方 video+audio VAE；`stage-dmd-step-250` 是 8-NFE 头牌）
- Comfy 端口：https://github.com/Saganaki22/ComfyUI-VDN-H3
- 社区 INT8 支：https://huggingface.co/drbaph/vdn-minimax-h3-int8-convrot-comfyui

头牌数字：**8×B200、14.4s 768p、8-NFE denoise 11.23 s**（博客）；SGLang 更新为 denoise **6.9 s / e2e ~9.0 s**。README 写明 **排除 load / warmup / VAE decode / mux**，并建议 VAE 放到别的机器。单卡 B200 8-NFE 仍 **51 s** 只 denoise。Comfy 端口在 **RTX 5090** 测到 ~**17 s/it**（1280×736 / 145f / 8-step，采样 ~2:15），作者自己写 **headline 数字拿不到**、FA4 不服务 **sm_120**、VRAM 紧时 **不推荐**；同卡 kitchen / Sage / SOL 能推更高分辨率。

| 路径 | 改什么 | 5090 成片 decode（36 层 ViT，256/64 tile-local RoPE） |
| --- | --- | --- |
| **comfy-kitchen** | 官方 VAE 的 RMS/RoPE/attn/FFN 融合核 | **P0。** 这就是 Comfy ~4s。 |
| **本仓 C+cuDNN（196 窗）** | 同合同、核未融合 | ~18 s。差在核，不在再减窗。 |
| **TRT 固定 256 tile 桶** | 官方 VAE 形状钉死后再编 | P1，kitchen 无 waffle 之后。 |
| **TAE / taeh3** | 另训 Tiny 解码器 | 预览；彩色闪。金线禁。 |
| **VDN** | DiT hybrid attn + 8-NFE turbo | **不改 VAE。** 32GB 常驻不了 B200 FP8 配方（需 block offload）；计时还不含 decode。 |

**5090 硬话：成片 decode 选 kitchen 官方 VAE，不要选 VDN。** VDN 最多是以后 DiT 768P 注意力的研究档（线性长程 vs sage3 fallback），且须另过 MAE；它解决不了 ~18s → Comfy ~4s。旧结论见 [turbo-sku §3.4](./2026-09-11-h3c-5090-turbo-sku.md)、[accel-survey §2.3](./2026-09-11-h3c-accel-survey-multigpu-5090.md)。

---

## 3. 模块方案（各自做到最好）

统一栏位：**现状秒数**（5s · 1344×768 · 单卡 5090 · 热路径；标注来源）· **能力上限** · **瓶颈** · **P0 / P1** · **质量红线** · **依赖**。

秒数来源：施工图工人 2026-09-14，`out/review_videos_768p/`，写入 [TaoMate 分析 §3.1](./2026-09-13-h3c-taomate-h3-analysis.md)。**1376×768 热 lift：无**（拒收）。480p 只作脚注。

### 3.1 加载 / mmap / pin / 常驻

| 项 | 内容 |
| --- | --- |
| **现状** | DiT INT8：`h3_w4a8_cache_stage_host_int8` mmap+`cudaHostRegister`，热 **~1.26 s**。VAE：`h3_vae_int8_store` 文件 mmap；可选 `H3_VAE_WEIGHT_CACHE=1` 跨 decode 留 pack；`H3_VAE_CUDA_W_RESIDENT` 设备 F32 W。历史每 tile 重 dequant **~90 s** 已修。主题档一次性 VAE load 曾 **~6 s**（[nvidia_optimization_plan §22](../../nvidia_optimization_plan.md)）。金线 768P `phase_ms` 未单列 load（含在 `vae=`）。 |
| **能力上限** | 热请求：DiT/TE/VAE **权已在 pin 或 VRAM**，load 墙钟 **<0.5 s**（只 H2D 缺页）。冷启动：直 GPU 流式（SGLang 模式）避免 7GB host F32。 |
| **瓶颈** | VAE 仍可能 host dequant（`H3_VAE_HOST_DEQUANT`）；F16 ckpt → F32 GEMM 材料化；TE HIT 后 `dit_host_stage` 记进 DiT 造成分段错觉。 |
| **P0** | VAE 与 DiT 同一套：**mmap 别名 + pin + 设备常驻 W**；默认 `WEIGHT_CACHE` 在 generate 热路径打开（金线可开，无损）。decode 前 `release_heavy_workspace` 已有，保持。日志把 `load_ms` 打进 `phase_ms`。 |
| **P1** | F16 W 常驻、GEMM 走 F16/TF32-off 或 INT8（仅当 ckpt 是 I8）；禁止每 clip 重新解 36 层。 |
| **质量红线** | 不改数值。pin 失败必须显式 fallback，禁止静默走 pageable bounce。 |
| **依赖** | `h3_w4a8_cache.c`、`h3_vae_int8_cuda.c` `weights_load`、`h3_vae_int8.c` `store_copy`。不依赖多卡。 |
| **文件** | Modify: `h3_vae_int8_cuda.c`（resident/pin）· `h3_w4a8_cache.c`（已有模式可抄）· `tools/h3_cuda_generate.c`（env 默认与 `phase_ms`） |

### 3.2 TE + prompt cache

| 项 | 内容 |
| --- | --- |
| **现状** | Qwen3-VL **NVFP4-AWQ**（`h3_nvfp4*`）。MISS **~13.14 s**（768P 金线）。`--te-cache` HIT 计算 **~0**；480p HIT 金线 e2e **61.41 s**（host_stage 暴露进 DiT）。视觉/多模 embedding **永不缓存**（`h3_te_cache.h`）。 |
| **能力上限** | 热同 prompt：**TE=0**。冷：**原生 FP4 GEMM**（不是 dequant→F32）把 MISS 再砍一截；与 cache 正交。 |
| **瓶颈** | 热路径已经是 cache。剩余是脏 key 前科、HIT 后分段口径、冷启动仍 dequant。 |
| **P0** | 768P 热评测默认打开 `--te-cache`（金线命令仍默认关）。cache 目录与评片隔离；key = ids + layers + 权 mtime/size。 |
| **P1** | TE 层 NVFP4 原生 MMA；VISION 路径不走 1D RoPE 冒充 mRoPE。 |
| **质量红线** | 同 ids HIT vs MISS **MAE=0**。脏 key 当事故。 |
| **依赖** | `h3_te_cache.c`、`h3_nvfp4.c`。不挡 VAE。 |
| **文件** | 已有模块。评测脚本：`tools/h3_768p_eval.sh` |

### 3.3 DiT（INT8 / sage3 / LoRA / 权窗）

| 项 | 内容 |
| --- | --- |
| **现状** | 金线：AdaLN-pruned **INT8 ConvRot** + cuDNN，8-step。768P 金线 DiT **112.23 s**。turbo4+TE+TAE e2e **80.48 s**（sage3 **fallback**）。lift n=2：低 **~6.8–7.0 s** + 高 **~15.2 s** ≈ **22 s**。768P `seq≈37788` sage3 **hits=0 fallback=200**。权窗：`h3_w4a8_cache` RESIDENT_ALL / WINDOW；VAE 前 `drop_device_weights` 留 host pin。 |
| **能力上限** | 单卡 5s 768P：注意力必须 **不再付 S=37k dense**（chunk Q / 窗口 K，或 5090 上能跑的快核）。少步（4/3）只砍 NFE，砍不掉 S²。NVFP4 社区权约 **1.14×/step**，远小于少步。 |
| **瓶颈** | sage3 在 768P 形状死掉 → 退 dense。LoRA 挂 INT8 GEMM 后。WINDOW 过小会在 lift 高分步 LRU 错层（已有 mosaic 注释）。 |
| **P0** | **修好 768P sage3 或等价快核**（缺插件硬失败，禁止假 sage）。在 VAE 到顶之前，**不要**用新 cafe 证明 27s。 |
| **P1** | NVFP4 DiT 加速档（`h3_dit_nvfp4.c` sm_120 MMA，禁止 CPU dequant 进 generate）；TaoMate 3-step 只许 **chunk+KV 对齐** 的 opt-in。 |
| **质量红线** | 金线：cudnn 8-step 官方 VAE。sage3 MAE~27 禁默认。Ulysses MAE 落到 sage 量级当 **bug**。Kijai W4A8 禁主题。 |
| **依赖** | VAE 方案不改 DiT 数学。权窗必须在 VAE 前腾 VRAM（已接线）。 |
| **文件** | `h3_dit_w4a8.c`、`h3_gpu_cuda.cu` sage3、`h3_lora.c`、`h3_w4a8_cache.c` |

### 3.4 SelfLift / LBH

| 项 | 内容 |
| --- | --- |
| **现状** | `--preset lift` 已接线为非金线。5s 768P 热 **47.09 / 50.39 s**；LBH **4.63 s**；能认人，仍糊/过饱和。`h3_latent_upscaler.*` = LBH-123-AI 3D resizer，**不是**论文 SelfLift-rich。`h3_selflift.h` 只做几何/NFE。rho=0 跳过 VAE 往返。 |
| **能力上限** | 延迟：多数 NFE 在 ~1/4 token + **学到的** lift 秒级。质量：过饱和/糊是 LBH+少高分步的上限，不是再砍 VAE 能修好的。 |
| **瓶颈** | 高分仍 1 步 ~15 s dense attn；成片仍付官方 VAE ~18–24 s；TAE 不能当锚。 |
| **P0** | **先把成片 VAE 做到 4s 量级**，lift e2e 才有资格和 27s 比。LBH 4.63 s 不是第一刀。 |
| **P1** | 高分步 chunk attn；禁止 TAE 锚；rho>0 像素锚必须官方 VAE。 |
| **质量红线** | `gold` 永不 lift。像素锚 ≠ TAE。画布仍 1344 直到产品明确要 1376。 |
| **依赖** | 官方 VAE decode（成片）+ 可选 encode（rho>0）。权不进 git。 |
| **文件** | `h3_selflift.c`、`h3_latent_upscaler.c`、`h3_generate_preset.c` |

### 3.5 VAE（专章，见 §4）

摘要：现状官方 **23.57 s** / 196 窗 **~18 s** / TAE **7.03 s（闪）**；Comfy **~4 s**。能力上限 = **厨房核 + tile-local RoPE + 常驻 W**，不是 TinyVAE。

### 3.6 mux / 编码

| 项 | 内容 |
| --- | --- |
| **现状** | `h3_ffmpeg_write_av_rgb24_f32`：双 pipe，`libx264 -preset fast -crf 18 -pix_fmt yuv420p`，音频 AAC 192k。768P 金线未单列 mux；480p 历史 ppm+mux **<1.5 s**。`phase_ms` 有 `ppm` / `mux`。 |
| **能力上限** | 120×1376×768 rgb24 在 5090 上 x264 fast **应 <2 s**；可与下一 clip DiT 重叠（流式才有 TTFF 意义）。 |
| **瓶颈** | 全帧 F32 RGB → u8 再 pipe；不是 27s 主墙。U=8 曾出现 ~6 kb/s 假片（审计文）。 |
| **P0** | 验收：ffprobe **5.000s±0.05 / 120 帧 / 有声**。mux 失败当臂失败，不当「VAE 很快」。 |
| **P1** | NVENC（质量门：块效应 vs crf18）；与 decode 重叠写盘。 |
| **质量红线** | 不许 `--frames 17` 假 5s。静音 AAC pad = 失败。 |
| **依赖** | `h3_ffmpeg.c`。系统 ffmpeg。 |
| **文件** | `h3_ffmpeg.c`、`tools/h3_cuda_generate.c` mux 段 |

### 3.7 附录：多卡 — 先冻结

| 项 | 内容 |
| --- | --- |
| **现状** | 8×5090 PCIe、`canAccessPeer=0`。U=2 cafe DiT **更慢**；U=8 **4.3× 更慢**。VAE **不并行**。 |
| **为何冻结** | 公路是 host bounce。VAE 还是单卡 18s 时，切 VAE 到 8 卡只会复制 waffle 合同。TRT-LLM parallel VAE 是 Wan/LTX conv + halo，H3 ViT 要另做注意力通信。 |
| **解冻条件** | (1) 单卡 VAE ≈ Comfy 且无 waffle；(2) 单卡 768P attn 不再 fallback dense；(3) 本机 `canAccessPeer=1` 或放弃 Ulysses 改请求级 DP。 |
| **P0** | **无。** 不要为对照 27s 开 U=8 cafe。 |
| **质量红线** | 多卡 vs 单卡同配方 MAE 若到 sage 量级 = bug。 |

---

## 4. VAE 专章：最快路径与验收

### 4.1 架构（对照 Comfy，禁止猜）

解码器 `ViT3DDecoder`：patch 16 / t 4，24ch→2048，**36 block**，32×64 heads，RoPE theta 100、`rope_dim_ratio=0.75`（本仓 `H3_VAE_INT8_ROPE_HALF=24`），4 register + 1 zero suffix。

`create_token_ids`：`arange(0.5, dim)/dim` 再 `2x-1`，**dim = 当前 tile 的 latent T,H,W**。重叠 tile 靠像素 **blend**，不靠「全画布低频 RoPE」。

`_decode_tile_row`：按显存最多 **4** 个空间 tile `cat` 一次 `_decode_pixels`（本仓 `H3_VAE_MAX_TILE_BATCH` 同思路）。

时间：`clip_length=17`，`token_drop=3`。Comfy `comfy_has_chunked_io = True`。

本仓接口：

```c
int h3_vae_int8_decode_cuda(h3_cuda *cuda, const h3_vae_int8_store *store,
                            const float *latent_chw, int latent_t, int latent_h,
                            int latent_w, unsigned n_blocks,
                            h3_vae_int8_frames *frames,
                            h3_vae_int8_decode_stats *stats, char *error,
                            size_t error_size);
/* stats: load_ms, canvas_h2d_ms, blocks_ms, unpatch_d2h_ms, total_ms */
int h3_vae_int8_prepare_rope(int t_len, int latent_h, int latent_w, ...); /* tile-local */
int h3_vae_int8_prepare_rope_ex(..., int full_h, int full_w, ...); /* canvas；OOD 风险 */
```

### 4.2 最快路径候选排序（执行顺序）

1. **对齐 Comfy 核，不改几何**  
   Tile **256 / overlap 64**、tile-local RoPE、temporal 17。把 36 层打到 kitchen 同类：融合 RMS+RoPE、F16 flash/cuDNN（**仅 tile seq**）、INT8/F16 GEMM 常驻 W。目标：**5s 1344×768 decode ~4 s 热**（load 另计）。  
   这是对「196 窗 ~18s」的直接回答：窗已经够小，缺的是 **Python kitchen 的融合核**，再减窗会踩 #46。

2. **融合 transformer 整块**  
   一层一次 launch（或 CUDA graph 36 次成 1）。本仓 `g_vae_layer_graph` 已有开关，默认与 flash 绑在一起——**不要**再用 naive flash 换 graph。

3. **TRT 固定 bucket**  
   输入形状钉死 `(1,24,Tchunk,16,16)`。只有 1 稳定且无 waffle 之后才做。不要上 Wan 的 parallel_vae。

4. **减窗**  
   仅实验：仍 tile-local、仍 `%16`、仍 overlap blend。验收失败立刻回 256。禁止「整幅 + F32」当快路径（waffle）。

5. **TAE / Flash-VAED**  
   预览档。金线永不。

### 4.3 加载优化（VAE 专有）

| 动作 | 预期 | 风险 |
| --- | --- | --- |
| 热路径默认 `H3_VAE_WEIGHT_CACHE` + device W | 去掉 ~6 s 冷 load / 跨 tile | VRAM；DiT 必须先释放（已做） |
| mmap I8/F16 + pin，禁止 host 7GB F32 | 对齐 DiT host_stage | pin 失败要打日志 |
| decode 直写预分配 RGB，不要全片再 cast F32 | 对齐 PR 15446；时长无关峰值 | 输出必须 [0,1] 与 mux 一致 |
| 最后 clip 才 `W_RESIDENT`（taomate 路径已 setenv） | 与多 clip LRU 共存 | clip2 不要 512 tile |

### 4.4 如何验收「无 waffle」

**禁止用 e2e 秒数代替。** 用 **同一 DiT latent**（`--skip-dit` 或 dump chw）：

1. **目视：** p01/p05/p06 抽 f0/24/48/90/119。16px 棋盘、tile 接缝阴影、整幅「quilt」= FAIL。对照 Comfy 同 latent。  
2. **网格能量：** 已有经验 `b16≈1.04`（256+deblock）vs `≈2.6`（整幅）。新核必须落在 tile 侧。  
3. **RoPE 单测：** `tests/test_h3_vae_rope.c` 必须继续证明 overlap 处与 **tile-local Comfy 公式** 一致；canvas `prepare_rope_ex` 不得 silently 成为默认。  
4. **几何：** `tile_size=256`、`overlap=64`、像素 `%16==0`。改这些必须重跑 1–2。  
5. **精度档：** tile 内允许 F16 attn；禁止 F16 整幅。TF32 在 VAE F32 SDPA **关**。  
6. **TAE：** 彩色闪 = 快档已知病，**不能**用来宣称「VAE 已做到最好」。  
7. **墙钟：** 热、n≥3、只报 `stats.blocks_ms`+`load_ms`。目标对齐 Comfy ~4s，而不是「比 23s 好一点」。

---

## 5. 建议执行顺序（仍不在本会话改推理）

```text
P0-VAE 核与 RoPE 合同 ──► P0-VAE 常驻/pin/chunked 输出
        │
        ├─► 无 waffle 验收通过 ──► 才允许 768P 热 lift 重测 VAE 段
        │
        ├─► P0-DiT 768P sage3/chunk attn（可并行写，但不要开 cafe 冲 27s）
        │
        └─► TE cache 评测默认 / mux 门 ── 已接近上限，不挡 VAE
多卡 ──────────────────────────────────── 冻结
```

拆分子方案（本枢纽不够细时另开，不要混进一次 PR）：

1. `2026-09-15-h3c-vae-kitchen-decode.md`（核 + RoPE + 验收）
2. `2026-09-15-h3c-dit-768p-attn.md`（sage3/chunk）
3. 加载/TE/mux 可附在 VAE PR 里，因为文件冲突面小。

---

## 6. Task 0 — 文档与接口冻结（已完成于本文件）

**Files:** `docs/superpowers/plans/2026-09-15-h3c-module-opt-plan.md`（本文件）

- [x] **Step 1:** 检索 Comfy H3 VAE / kitchen / TAE / TRT / pin，对照本仓 waffle/TAE/flash/196 窗
- [x] **Step 2:** 按模块写下现状秒数、P0/P1、红线
- [ ] **Step 3:** （执行期）VAE 子方案按 `writing-plans` 拆测试：同 latent vs Comfy、`test_h3_vae_rope`、热 `decode_stats`

**不要 commit。不要开 cafe。**

---

## 7. 交叉链接

| 主题 | 文档 |
| --- | --- |
| 768P 尺子 | [2026-09-13-h3c-768p-eval-protocol.md](./2026-09-13-h3c-768p-eval-protocol.md) |
| 27s / TaoMate 口径 | [2026-09-13-h3c-taomate-h3-analysis.md](./2026-09-13-h3c-taomate-h3-analysis.md) |
| LBH / SelfLift | [2026-09-13-h3c-selflift-comfy-analysis.md](./2026-09-13-h3c-selflift-comfy-analysis.md) |
| INT8/NVFP4 | [2026-09-13-h3c-model-format-5090.md](./2026-09-13-h3c-model-format-5090.md) |
| 多卡为何慢 | [2026-09-13-h3c-minimax-lightning-survey.md](./2026-09-13-h3c-minimax-lightning-survey.md) |
| 旧 VAE load 叙事 | [nvidia_optimization_plan.md](../../nvidia_optimization_plan.md) §22 |
| VDN hybrid（非 VAE） | [2026-09-11-h3c-5090-turbo-sku.md](./2026-09-11-h3c-5090-turbo-sku.md) §3.4 · [accel-survey §2.3](./2026-09-11-h3c-accel-survey-multigpu-5090.md) |

---

## 8. 来源（网上，2026-09-15）

- Comfy MiniMax VAE：https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/minimax/vae.py
- Optimize MiniMax-H3 VAE（chunked I/O）：https://github.com/Comfy-Org/ComfyUI/pull/15446
- handles_tiling OOM retry：https://github.com/Comfy-Org/ComfyUI/pull/15456 · https://github.com/Comfy-Org/ComfyUI/issues/15453
- 官方 tile 可否改：https://github.com/MiniMax-AI/MiniMax-H3/issues/46
- MiniMax-H3 模型卡：https://github.com/MiniMax-AI/MiniMax-H3/
- comfy-kitchen：https://github.com/comfy-org/comfy-kitchen
- KJNodes Triton VAE（不含 H3）：https://github.com/kijai/ComfyUI-KJNodes/blob/main/nodes/triton_vae.py
- TAEHV / taeh3：https://github.com/madebyollin/taehv · https://huggingface.co/Kijai/MiniMax-H3-TAE
- Flash-VAED：https://arxiv.org/html/2602.19161v1
- Turbo-VAED：https://arxiv.org/abs/2508.09136v1
- TRT-LLM parallel VAE：https://github.com/NVIDIA/TensorRT-LLM/pull/11875 · https://github.com/NVIDIA/TensorRT-LLM/pull/15753
- vLLM-omni VAE RFC：https://github.com/vllm-project/vllm-omni/issues/5948
- SGLang VAE GPU 直载：https://github.com/sgl-project/sglang/pull/37004
- mmap+pin RFC：https://github.com/vllm-project/vllm/issues/42361
- rlx-models H3 VAE tiling 说明：https://github.com/MIT-RLX/rlx-models/blob/main/crates/rlx-minimax-h3/README.md
- VDN（Video DeltaNet，**不是** VAE）：https://openvdn.github.io/ · https://github.com/OpenVDN/vdn-minimax-h3 · https://huggingface.co/OpenVDN/vdn-minimax-h3 · Comfy 端口 https://github.com/Saganaki22/ComfyUI-VDN-H3
