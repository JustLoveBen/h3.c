# NVIDIA 量化契约（冻结稿 · P0）

本文件冻结本仓 NVIDIA 路径的**第一目标量化格式**与命名边界。实现尚未开始；以仓库代码为准，社区 checkpoint 仅作目标参考。

## 1. 命名陷阱：本仓 int8 ≠ 社区 W4A8 / INT8 ConvRot

| 名称 | 本仓现状 | 社区 Comfy / comfy-kitchen |
| --- | --- | --- |
| **M5 运行时 int8** | Apple Metal 4 路径：从官方 **BF16** 权重在加载期量化为 INT8（每输出通道一个 F32 scale）；激活动态量化。见 `h3_gpu_quantize_weight_int8` / `h3_gpu_mlp_int8_bf16` / `h3_gpu_linear_int8_*`（`h3_gpu.h` + `h3_dit.c`）。 | 无关 |
| **W4A8 / asym_w4a8_int8** | **未实现**。无 ConvRot、无 4-bit pack、无 codebook、无 FP8 group scale loader。 | 权重为 ConvRot 旋转后的 packed INT4 + Lloyd-Max codebook + FP8 scales；推理时激活 INT8 + INT8 GEMM |
| **INT8 ConvRot** | **Host loader 已启动**（`h3_dit_int8` / `h3_dit_int8_probe`）；CUDA 前向未接。 | 次选主题跟随：整权重量化为 INT8+ConvRot（~21 GB pruned DiT）；Comfy A/B 已证主题 PASS；质量通常高于 W4A8 |

**硬规则：** 不要把 CLI/README 里的「int8」（M5 Metal）理解成可直接加载 Hugging Face 上的 `*_w4a8_*` / `*_int8_convrot_*` 文件。那些是另一套存储与内核契约。

## 2. 第一目标 checkpoint

**首选（4090 主线）：** Kijai W4A8 pruned mixed —  
[`Kijai/MiniMax-H3-experimental`](https://huggingface.co/Kijai/MiniMax-H3-experimental)

| 文件 | 变体 | 约大小（hub） |
| --- | --- | --- |
| `minimax_h3_fl2va_pruned_w4a8_mixed.safetensors` | FL2VA（文生 / 首末帧） | ~11.7 GB |
| `minimax_h3_ref2va_pruned_w4a8_mixed.safetensors` | Ref2VA | ~11.0 GB |
| `minimax_h3_video_vae_int8_convrot.safetensors` | Video VAE INT8 ConvRot | ~3.0 GB |
| `loras/minimax_h3_ref_lora_rank_256_bf16.safetensors` | Ref LoRA | ~2.4 GB |

**官方 / Comfy 全集（勿整仓下）：**  
[`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3)（hub 合计约 ~433 GB：`diffusion_models/` + `text_encoders/` + `vae/`）。端到端最小补齐见文末清单；本仓**不**自动拉取该全集。

其他同格式族社区卡（loader/元数据可能不同，加载前必须 header inventory）：  
`Winnougan/MiniMax-H3-INT4_Convrot_ComfyUI`、`AX1Y2JP/MiniMax-H3-W4A8-ConvRot`、历史猜测名 `starsfriday/MiniMax-H3-w4a8`（以实际可解析 URL 为准）。

**次选 DiT：** Comfy-Org 完整 INT8 ConvRot pruned（`diffusion_models/*_pruned_int8_convrot.safetensors`，各 ~19.5 GB）。仅在 W4A8 内核未就绪或质量不够时考虑。

本仓默认**不下载**大模型。远程/本地路径由用户或落地流程提供后，用 safetensors **只读 header** inventory 核对 dtype / 附属 tensor 名。

### 布局假设

| 布局 | 说明 | 本仓今天支持 |
| --- | --- | --- |
| **官方树** | `MiniMax-H3/FL2VA/transformer/`、`Ref2VA/transformer/` 多分片 BF16 safetensors | **是**（`h3_weight_store_open` + `h3_dit`） |
| **Comfy 单文件** | 一个 pruned / W4A8 `.safetensors`（常放 `diffusion_models/`） | **否** — P3 loader 目标 |

W4A8 第一目标假设：**Comfy / Kijai 单文件 mixed checkpoint**，不是官方多分片 BF16 树。官方名可作逻辑映射基准；物理存储是 packed 4-bit + 附属 scale/codebook。

对每个 DiT `blocks.{i}`（i=0..49）量化这四个权重：

- `attn.qkv_proj.weight`
- `attn.out_proj.weight`
- `mlp.fc1.weight`
- `mlp.fc2.weight`

**已用 Kijai FL2VA W4A8 header 核实的参数族：** format=`asym_w4a8_int8`；`group_size=16`；`convrot=true`；`convrot_groupsize=256`；`weight_codebook` 为 F32[16]；`weight_s_rel` 为 F8_E4M3；`weight_s_channel` 为 F32；packed 权重存为 **I8**（K 维约为未量化的一半，两 nibble/字节）；其余张量 mixed（BF16/F16/F32）。

## 3. 与本仓官方 tensor 名的映射

本仓 DiT 从官方式名字加载（见 `h3_dit.c` `load_block` / `load_core`），前缀 `blocks.{i}.`（i = 0..49）：

| 逻辑角色 | 本仓 / 官方后缀 | W4A8 目标（Kijai FL2VA inventory） | 备注 |
| --- | --- | --- | --- |
| RMS / norms | `norm1.weight`, `norm2.weight` | BF16 | 非四目标权 |
| QKV | `attn.qkv_proj.weight` | I8 packed + `weight_codebook` / `weight_s_channel` / `weight_s_rel` | **P3/P4 主路径** |
| Q/K norm | `attn.q_norm.weight`, `attn.k_norm.weight` | BF16 | |
| Attn out | `attn.out_proj.weight` | 同上附属 | |
| MLP | `mlp.fc1.weight`, `mlp.fc2.weight` | 同上附属 | |
| AdaLN | `blocks.{i}.adaln_proj.linear.{weight,bias}` | **保留**；dtype=**F16**；shape **`[96768, 8]` / `[96768]`**（= `3×6×5376` × time_feat）；顶层 `adaln_t_table` F32[1025,8] | **异于**官方 BF16 `TIME_DIM=2688` 路径；W4A8 用 8-d table lookup |
| Patch / final | `video_patch_proj.*`, `audio_patch_proj.*`, `final_layer.*` | **保留**；**无** `diffusion_model.` 前缀 | |
| Token refiner | `token_refiner.blocks.{0,1}.*` | **保留**；其 attn/mlp **未** W4A8（BF16） | |
| RoPE | `rope.inv_freq` | **保留** F32[16] | |
| Condition | `condition_proj.{weight,bias}` | **保留**（BF16/F32） | 单文件内置 |

**映射策略（P3）：**

1. 打开用户路径下的 safetensors **只读 header**（不映射整文件到 GPU）。C 工具：`make h3_st_inventory && ./h3_st_inventory FILE.safetensors`（可选 `--w4a8-block N`）。
2. 枚举键：Kijai FL2VA **无** `diffusion_model.` / `model.` 前缀；四目标权拆成 `*.weight`(I8) + `*.weight_s_rel`(F8_E4M3) + `*.weight_codebook`(F32[16]) + `*.weight_s_channel`(F32)。
3. 将逻辑后缀对齐到本仓 `blocks.{i}.…`；无法对齐的键记入 TODO，禁止 silently skip 关键投影。
4. 本仓 M5 的 `*_int8` / `*_scales` 是**运行时分配的 GPU 张量**，不是 checkpoint 字段；W4A8 loader 不得复用这套语义。

**已知（Kijai FL2VA `minimax_h3_fl2va_pruned_w4a8_mixed.safetensors`，2026-08-13 inventory）：**

- [x] 键前缀：直接 `blocks.` / `token_refiner.` / `final_layer.` / `*_patch_proj.` / `condition_proj.` / `rope.` / `adaln_t_table`（无 Comfy `diffusion_model.` 包装）
- [x] blocks 仍为 0..49（50 层）；「pruned」未删整层，主要是 W4A8 mixed 打包
- [x] 附属量化张量名与 dtype 如上；`__metadata__._quantization_metadata` 标明 `asym_w4a8_int8`
- [x] token_refiner / patch / final / rope / adaln / condition **内置于同一单文件**；**不含** text encoder / VAE（需另下）

**仍 TODO：**

- [x] Ref2VA 粗查：同为 `blocks.0..49` + codebook/s_rel/s_channel；**额外**有大量 `*.comfy_quant`（U8，例 shape [89]），张量总数 1348 > FL2VA 的 1132 — loader 需兼容
- [x] Video VAE `minimax_h3_video_vae_int8_convrot`：前缀 `encoder.` / `decoder.`；量化权为 I8 + `weight_scale`(F32) + `*.comfy_quant`(U8[72])；**不是** DiT 的 codebook/s_rel 布局
- [x] INT8 ConvRot DiT 次选文件的键差异：`minimax_h3_fl2va_pruned_int8_convrot` → **932** tensors；线性为 **I8 `*.weight` + F32 `*.weight_scale` + U8 `*.comfy_quant`**（与 VAE 同族 `int8_scale_convrot_like`，**无** codebook/s_rel）；`token_refiner` 仍为 BF16；host loader `h3_dit_int8` / `make h3_dit_int8_probe`（2026-08-18）

- [x] packed I8 的精确 pack 顺序（nibbles）与 dequant 公式（对照 comfy-kitchen eager）：见 §10
- [ ] FL2VA 是否也应出现 `comfy_quant`（本次 inventory 未见；与 Ref2VA 差异待确认是否导出差异）
- [x] **激活侧 ConvRot**（Hadamard group=256；`Y = X@W_undo.T == (X@H)@W_rot.T`）— 见 §10 / `h3_w4a8_convrot_act_f32`
- [x] CUDA / cuBLAS FP32 加速路径（dequant 后 `cublasSgemm`）— 见 §10 / `h3_w4a8_cuda_probe`
- [x] 多 block 冒烟 + CPU INT8 激活量化 / INT8×INT8→INT32→FP32 参考 — 见 §10.3 / `h3_w4a8_blocks_probe`
- [x] CUDA INT8 / cuBLASLt INT8×INT8→INT32 + scale 反量化 — 见 §10.4 / `h3_w4a8_int8_cuda_probe`
- [x] on-device W4A8 unpack / 权重常驻显存 — 见 §10.5 / `h3_w4a8_device_probe`
- [x] 50-block INT8 常驻 / 换入管理器 — 见 §10.6 / `h3_w4a8_cache_probe`
- [x] 接 DiT 四线性（旁路 `h3_dit_w4a8`）— 见 §10.7
- [x] 单 block 近完整前向（AdaLN / QK-RMS / RoPE / SDPA / gate，CPU F32 + CUDA linears）— 见 §10.8
- [x] 多 block 串联（可配置 `--layers`，含 0..49 `resident_all`）— 见 §10.9
- [x] patch_embed / final_layer / `t_table` 时间条件（video-only）— 见 §10.9
- [x] 最小 Euler 采样环（纯 C，可复现 checksum）— 见 §10.9
- [x] Video VAE INT8+scale / TE NVFP4 最小 dequant 探测 — 见 §10.10
- [x] 端到端 generate 旁路冒烟（§10.11 / `h3_w4a8_generate_probe`）
- [x] CUDA Flash SDPA + 激活 ConvRot/INT8 上 GPU（§10.12）
- [x] DiT block 激活 GPU 常驻 + SDPA tile 加强（§10.14）
- [ ] 接 `h3_dit.c` 运行时 / 完整 token_refiner / 全量 TE+VAE / VAE GPU

## 4. 4090 / 5090 推荐档位（一句话）

**RTX 4090（24 GB）与 5090：默认上 W4A8 pruned mixed DiT（~11.7 GB 文件级）；INT8 ConvRot 作质量次选，4090 上更吃紧。**

（峰值 VRAM 还含 TE / VAE / 激活；文件更小 ≠ 峰值同等下降。）

## 5. Phase 顺序

| Phase | 内容 | 状态 |
| --- | --- | --- |
| **P0** | 本文：冻结契约与命名边界 | 本文件 |
| **P1** | CUDA 探测空壳 + CMake 切换；不替换 Metal `h3_gpu` | 见 `docs/nvidia_build.md` |
| **P2** | （预留）设备抽象 / 错误模型，仍不实现 GEMM | 未开始 |
| **P3** | W4A8 / 单文件 safetensors loader + header inventory | **完成（最小接入）**：见 §8 / §9；无 ConvRot / GEMM / generate |
| **P4** | INT8 GEMM / W4A8 dequant+GEMM 内核 | **旁路冒烟完成（§10.11）**：条件采样 + VAE PPM + `generate_probe`；**未**接 Metal `h3_dit.c` |

**不做：** 用本仓 M5 int8 内核“假装”跑 W4A8；不重写 DiT 图；不在 P1 实现完整推理。

## 8. P3：C 只读 header / W4A8 探测（2026-08-13）

**硬规则：验收与探测路径全部为 C；不使用 Python 脚本。**

**构建（不碰 Metal 全量链路）：**

```bash
# Darwin
make h3_st_inventory h3_w4a8_probe

# Linux（可与 CUDA stub 同树，或直接 cc）
cc -std=c11 -O2 -I. -o h3_st_inventory \
  tools/h3_st_inventory.c h3_safetensors.c h3_w4a8.c
cc -std=c11 -O2 -I. -o h3_w4a8_probe \
  tools/h3_w4a8_probe.c h3_safetensors.c h3_w4a8.c
# 或: cmake --build build-cuda --target h3_st_inventory h3_w4a8_probe
```

**运行：**

```bash
./h3_st_inventory FILE.safetensors
./h3_st_inventory --w4a8-block 0 FILE.safetensors
./h3_w4a8_probe --block 0 FILE.safetensors
```

能力（均经 `h3_safetensors` / `h3_w4a8` C API）：

- 读取 safetensors **header + `__metadata__`**（`metadata_json`）；识别 `F8_E4M3` / `U8`
- 解析 format / `group_size` / `convrot` / `convrot_groupsize`
- 统计附属键（`weight_codebook` / `weight_s_rel` / `weight_s_channel` / `comfy_quant` / `weight_scale`…）
- `--w4a8-block N`：按 `load_block` 方向解析 `blocks.{N}.{qkv,out,fc1,fc2}` 四元组，并 host 读 codebook 冒烟

**C API 摘要：**

| API | 作用 |
| --- | --- |
| `H3_DTYPE_F8_E4M3` | FP8 只读登记（1 byte） |
| `h3_st_header.metadata_json` | 保留 `__metadata__` 原始 JSON |
| `h3_st_quant_info_from_header` | format / group_size / convrot… |
| `h3_w4a8_store_open` | 单文件 DiT W4A8 打开 |
| `h3_w4a8_find_block_linear` | `blocks.{i}.{suffix}` 四元组 |
| `h3_w4a8_read_tensor` / `dup` | host buffer 拷贝 |

**P4 旁路核心已完成（§10.1–§10.11）：** 50-block 串联、patch/final、Euler 少步、条件接入、VAE 最小 decode、`h3_w4a8_generate_probe` 端到端冒烟。仍未做：接 Metal `h3_dit.c`、完整 TE/VAE 运行时 / 观感质量。

**远程 Kijai 探测结果（`/home/haiyang/h3_c/models/kijai/`，历史 inventory；以 C 工具复验为准）：**

| 文件 | kind | 要点 |
| --- | --- | --- |
| `minimax_h3_fl2va_pruned_w4a8_mixed.safetensors` | `asym_w4a8_int8` | 1132 tensors；metadata 200 层全 `asym_w4a8_int8`；`group_size=16`，`convrot_groupsize=256`；附属 200×codebook/s_rel/s_channel；**无** `comfy_quant` |
| `minimax_h3_ref2va_pruned_w4a8_mixed.safetensors` | `asym_w4a8_int8` | 同族 + **有** `comfy_quant`（U8）；张量更多 |
| `minimax_h3_video_vae_int8_convrot.safetensors` | `int8_scale_convrot_like` | I8 + `weight_scale` + `comfy_quant`；**不是** DiT codebook 布局 |

**Comfy-Org TE / audio VAE（端到端补齐，勿下全集）：**

| 路径（远程） | hub 期望大小 | 说明 |
| --- | --- | --- |
| `models/comfy-org/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 15687142551 B (~14.61 GB) | 已核对 hub `text_encoders/` 真实文件名 |
| `models/comfy-org/vae/minimax_h3_audio_vae_fp32.safetensors` | 605254808 B (~0.56 GB) | 空间够则下 |

**TE inventory（已完整，2026-08-13）：**  
`models/comfy-org/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` = **15687142551 B**（与 hub 一致）。

| 项 | 实测 |
| --- | --- |
| tensors | 2054；前缀 `model.`(1703) + `visual.`(351) |
| `__metadata__` | **无** |
| 量化形态（LLM 线性） | `*.weight` **U8**（K 约为 BF16 一半 → nibble/FP4 pack）+ `*.weight_scale` **F8_E4M3** + `*.weight_scale_2` **F32 标量** + `*.comfy_quant` U8；部分有 `pre_quant_scale` BF16 |
| 例 | `model.layers.0.self_attn.q_proj.weight` U8[8192,2560] / `weight_scale` F8_E4M3[8192,320] / `weight_scale_2` F32[] |
| visual | 大体 **BF16**（未走 U8 pack） |
| embed | `model.embed_tokens.weight` I8 + `weight_scale` F32 |
| probe kind | `nvfp4_awq_like`（**不是** DiT 的 `asym_w4a8_int8`，也无经典 AWQ `qweight`） |

**Audio VAE：** `vae/minimax_h3_audio_vae_fp32.safetensors` = **605254808 B**（完整）。

## 9. P3 完成边界 / P4 切入口

**P3 已完成：** F8_E4M3 + metadata 只读；单文件 `h3_w4a8_store`；`load_block` 方向四元组查找与 host 读；C inventory/probe 验收。

**P3 明确不做：** 不改 `h3_dit.c` 的 BF16 `load_block` 运行时路径；不实现 dequant/GEMM。

**P4 建议切入口：** 在 `h3_w4a8_find_block_linear` 拿到的四元组上实现 packed I8 nibble unpack + codebook + `s_channel`/`s_rel` dequant，再接 INT8 GEMM；与现有 Metal M5 int8 运行时量化语义隔离。

## 10. P4：CPU dequant / 激活 ConvRot / cuBLAS（2026-08-13）

### 10.1 第一步：CPU weight dequant + FP32 GEMM

**已完成（纯 C，无 Python）：**

| 项 | 状态 |
| --- | --- |
| nibble unpack（低 nibble=偶 K，高 nibble=奇 K） | 完成 |
| codebook[idx] × F8_E4M3 `s_rel` → round/clamp INT8 | 完成 |
| INT8 × F32 `s_channel` → FP32（ConvRot 旋转基） | 完成 |
| 权重量侧 Hadamard undo（`convrot_groupsize=256`） | 完成（可选 `--no-convrot`） |
| 单 linear dequant-then-FP32-GEMM + 自洽检查 | 完成（`h3_w4a8_matmul_probe`） |

**公式 / 布局（对齐 comfy-kitchen eager `AsymW4A8Int8Layout`）：**

1. `packed[N, K/2]` I8：`byte = (odd<<4) | even`；展开索引 ∈ [0,15]
2. `int8 = clamp(round(codebook[idx] * f8e4m3(s_rel[n, g])), -127, 127)`，`g = k // group_size`，`group_size=16`
3. `W_rot[n,k] = int8[n,k] * s_channel[n]`
4. （可选）`W = W_rot @ H` 按 `convrot_groupsize` 分组；归一化正则 Hadamard 为对合，再旋一次即还原
5. Probe GEMM：`Y = X @ W.T`（FP32），默认 undo ConvRot 后可对**未旋转**激活做 float 参考

**构建 / 运行：**

```bash
make h3_w4a8_matmul_probe
./h3_w4a8_matmul_probe --block 0 --suffix out --batch 4 FILE.safetensors
./h3_w4a8_matmul_probe --check-act-convrot --block 0 --suffix out FILE.safetensors
# 或: cmake --build build-cuda --target h3_w4a8_matmul_probe
```

**远程验收（第一步，`haiyang@192.168.0.38`，FL2VA W4A8，2026-08-13）：**

```text
linear: blocks.0.attn.out_proj
packed_shape: [5376,3584] (logical K=7168)
dequant: N=5376 K=7168 time_ms≈6791  (weight undo ConvRot)
weight_stats: finite, absmax≈1.88, checksum=7d0d37f29a09355b
gemm batch=2: finite, absmax≈20.81, checksum=6931d38d397bd4d9
self_check_bit_identical / dequant_repeat / finite: PASS
```

### 10.2 第二步：激活 ConvRot + cuBLAS FP32

**数学等价（H 对称且对合）：**

```text
W = W_rot @ H
Y_ref  = X @ W.T           = X @ H @ W_rot.T
Y_act  = (X @ H) @ W_rot.T
⇒ Y_ref ≡ Y_act   （FP32 累加序差异 → 用 atol/rtol，非 bit-identical）
```

| 项 | 状态 |
| --- | --- |
| `h3_w4a8_convrot_act_f32`（Hadamard group=256） | 完成 |
| CPU 等价路径自洽（`--check-act-convrot` / cuda probe） | 完成 |
| `h3_cuda_gemm_f32` → cuBLAS `cublasSgemm`（FP32，row-major Y=X@Wᵀ） | 完成 |
| `tools/h3_w4a8_cuda_probe.c` | 完成 |
| INT8 激活量化 + INT8 Tensor Core GEMM | **完成（CPU 参考 §10.3；CUDA cuBLASLt §10.4）** |
| 全 50 block / `h3_dit` 改接 / generate | **TODO** |

**容差策略（默认）：** `max_abs ≤ atol + rtol * absmax(Y_ref)`，`atol=1e-3`，`rtol=1e-4`。
另打印 checksum / RMSE 便于跨机对照；cuBLAS 与 naive CPU 累加序不同，**不以 bit-identical 为 PASS 条件**。

**构建 / 运行（Linux + CUDA，`conda activate comfyui`）：**

```bash
cmake -S . -B build-cuda -DH3_GPU_BACKEND=cuda
cmake --build build-cuda -j --target h3_w4a8_cuda_probe h3_w4a8_matmul_probe
./build-cuda/h3_w4a8_cuda_probe --block 0 --suffix out --batch 4 \
  /path/to/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors
```

**C API：**
`h3_w4a8_dequant.h` — 既有 dequant + `h3_w4a8_convrot_act_f32` / `h3_w4a8_compare_f32`
`h3_gpu_cuda.h` — `h3_cuda_gemm_f32`（链接 `cudart` + `cublas`）

**不破坏：** Darwin Metal / BF16 `load_block`；CUDA 目标仅经 CMake；dequant 不进 Metal `libh3` 默认依赖。

**远程验收（第二步，`haiyang@192.168.0.38` / RTX 4090 / nvcc 12.9，FL2VA W4A8，2026-08-13）：**

```text
linear: blocks.0.attn.out_proj  N=5376 K=7168  batch=4
dequant_undo≈6719 ms | dequant_rot≈155 ms
cpu_ref_gemm≈118 ms  absmax≈24.18  checksum=6d746b7659612c74
act_convrot_vs_weight_undo: max_abs≈8.2e-5 rmse≈8.8e-6  PASS
cublas_vs_cpu_ref: max_abs≈5.5e-5 rmse≈6.3e-6  gemm_ms≈0.159  PASS
summary: PASS
```

conda `comfyui` 下链接 cuBLAS 时建议系统 host 编译器：
`cmake -S . -B build-cuda -DH3_GPU_BACKEND=cuda -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++`

### 10.3 第三步：多 block 冒烟 + CPU INT8 参考（2026-08-13）

**目标：** 在不接 `h3_dit` / generate 的前提下，对若干 DiT block 的四线性做 dequant 自洽 + 短 GEMM，并落地最小可用 A8 路径。

#### 激活 INT8 假设（对齐 comfy-kitchen / 契约）

| 项 | 约定 |
| --- | --- |
| 时机 | **激活先 ConvRot**（`X_rot = X @ H`，`group=256`），再动态量化 |
| 默认粒度 | **per-row**（每 token 一个 F32 scale）；可选 `--act-quant per-tensor` |
| 公式 | `scale = absmax/127`（全零 → scale=1）；`x_q = clamp(round(x/scale), -127, 127)` |
| 权重 | 保留 dequant 中间态 `W_i8[N,K]`（ConvRot 旋转基）+ `s_channel[N]`；**不做** weight Hadamard undo（与 act ConvRot 配对） |
| GEMM | `Y[m,n] = act_scale[m] * s_channel[n] * Σ_k x_q[m,k] * W_i8[n,k]`（INT32 累加 → FP32） |

与 FP32 参考 `Y_fp32 = X_rot @ (W_i8 * s_channel).T` 的差异**仅来自激活量化**；另用重构激活 `X_hat = x_q * act_scale` 的 FP32 GEMM 做紧容差自洽（验证 INT8 累加/反量化代数正确）。

**容差：**

| 对比 | 默认 |
| --- | --- |
| `Y_i8` vs `Y_qi`（量化激活重构） | `atol=1e-3`，`rtol=1e-4`（应极紧） |
| `Y_i8` vs `Y_fp32`（全精度激活） | `atol=0.5`，`rtol=0.05`（A8 量化误差） |

#### API（预留，未接 DiT）

| API | 作用 |
| --- | --- |
| `h3_w4a8_load_linear_int8` | packed → host `W_i8` + `s_channel` |
| `h3_w4a8_quantize_act_int8` | 动态 per-row / per-tensor 激活 INT8 |
| `h3_w4a8_gemm_i8_dequant_f32` | INT8×INT8→INT32→FP32 反量化 GEMM（CPU 参考） |

#### 构建 / 运行

```bash
# Darwin / 任意 host（无 CUDA）
make h3_w4a8_blocks_probe
./h3_w4a8_blocks_probe --blocks 0,1,24,49 --batch 2 FILE.safetensors
./h3_w4a8_blocks_probe --blocks 0-3 --act-quant per-tensor FILE.safetensors

# Linux + CMake（与 CUDA 树同编）
cmake --build build-cuda -j --target h3_w4a8_blocks_probe
./build-cuda/h3_w4a8_blocks_probe --blocks 0,1,24,49 FILE.safetensors
```

**下一步（未做）：** 接 `h3_dit` / generate。CUDA INT8 见 §10.4；device unpack 见 §10.5。

**不破坏：** Metal / `libh3` 默认依赖；不改 `h3_dit` 运行时接线。

**远程验收（第三步，`haiyang@192.168.0.38` / RTX 4090 / nvcc 12.9 / conda `comfyui`，FL2VA W4A8，2026-08-13）：**

```bash
cmake -S . -B build-cuda -DH3_GPU_BACKEND=cuda \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++
cmake --build build-cuda -j --target h3_w4a8_blocks_probe
./build-cuda/h3_w4a8_blocks_probe --blocks 0,1,24,49 --batch 2 \
  /home/haiyang/h3_c/models/kijai/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors
```

```text
act_quant: per-row
tol_qi: atol=1e-3 rtol=1e-4 | tol_int8_vs_fp32: atol=0.5 rtol=0.05
blocks 0/1/24/49 × {qkv,out,fc1,fc2} = 16 linears
例 blocks.0.out:  qi_max≈5.0e-5  i8_max≈0.15  PASS
例 blocks.24.fc2: qi_max≈2.2e-4  i8_max≈0.44  PASS
例 blocks.49.out: qi_max≈1.2e-4  i8_max≈0.41  PASS
summary: pass=16 fail=0 skip=0 → PASS
```

### 10.4 第四步：CUDA INT8 GEMM（cuBLASLt）（2026-08-13）

**目标：** 在 GPU 上真实执行 INT8×INT8→INT32，再按 `act_scale × s_channel` 反量化到 FP32，与 CPU `h3_w4a8_gemm_i8_dequant_f32` **同语义**对齐。

| 项 | 状态 |
| --- | --- |
| `h3_cuda_gemm_i8_dequant_f32` | 完成（优先 `cublasLt` / `CUBLAS_COMPUTE_32I`；回退 `cublasGemmEx`；再回退自定义 device kernel） |
| 设备侧 scale 反量化 kernel | 完成 |
| `tools/h3_w4a8_int8_cuda_probe.c` | 完成（默认 `out,fc2`） |
| on-device W4A8 unpack / 接 DiT | **device unpack 完成 §10.5；接 DiT TODO** |

**布局：** 与 FP32 cuBLAS 相同的 row-major `Y = X @ Wᵀ` ↔ `cublas(OP_T, OP_N)`；输入 `CUDA_R_8I`，累加 `CUDA_R_32I`。

**容差（cuda_i8 vs cpu_i8）：** `atol=1e-3`，`rtol=1e-4`。INT32 累加应对齐到 bit-identical（实测 max_abs=0）；容差预留 FP32 反量化舍入。

**构建 / 运行：**

```bash
cmake -S . -B build-cuda -DH3_GPU_BACKEND=cuda \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++
cmake --build build-cuda -j --target h3_w4a8_int8_cuda_probe
./build-cuda/h3_w4a8_int8_cuda_probe --block 0 --suffixes out,fc2 --batch 4 \
  /path/to/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors
```

**C API：** `h3_gpu_cuda.h` — `h3_cuda_gemm_i8_dequant_f32`（链接 `cudart` + `cublas` + `cublasLt`）

**远程验收（第四步，`haiyang@192.168.0.38` / RTX 4090 / nvcc 12.9 / conda `comfyui`，FL2VA W4A8，2026-08-13）：**

```text
cuda_device: [0] NVIDIA GeForce RTX 4090 sm_89
tolerance (cuda_i8 vs cpu_i8): atol=1e-3 rtol=1e-4
blocks.0.out: N=5376 K=7168 batch=4 backend=cublasLt
  cpu_i8_ms≈13.92  cuda_i8_ms≈0.039  cublas_f32_ms≈0.154
  max_abs=0 rmse=0  PASS
blocks.0.fc2: N=5376 K=14336 batch=4 backend=cublasLt
  cpu_i8_ms≈27.90  cuda_i8_ms≈0.080  cublas_f32_ms≈0.349
  max_abs=0 rmse=0  PASS
summary: pass=2 fail=0 → PASS
```

相对 CPU INT8：out ≈ **357×**，fc2 ≈ **349×**；相对同形状 FP32 cuBLAS：out ≈ **3.9×**，fc2 ≈ **4.4×**（仅 GEMM+dequant 设备侧计时，不含 H2D/D2H）。

### 10.5 第五步：device W4A8 unpack + 权重常驻（2026-08-13）

**目标：** 将 packed I8 + codebook + `s_rel` + `s_channel` 上传到 GPU，在 device 上 nibble unpack → INT8[N,K]，**INT8 与 `s_channel` 常驻显存**；再接已有 `h3_cuda_gemm_i8_dequant_f32` 路径（resident 变体）跑 `out`/`fc2`。

| 项 | 状态 |
| --- | --- |
| `h3_cuda_w4a8_unpack_kernel`（nibble + F8 `s_rel` + codebook → INT8） | 完成 |
| `h3_cuda_w4a8_weight_create` / `_free` / `_download_int8` | 完成（H2D packed/scales + unpack；INT8+`s_channel` 常驻） |
| `h3_cuda_gemm_i8_dequant_f32_w` | 完成（不重复上传 W） |
| `tools/h3_w4a8_device_probe.c` | 完成（默认 `out,fc2`；对比 host `load_linear_int8`） |
| 接 `h3_dit` / generate | **TODO** |

**数值：** device unpack vs host `h3_w4a8_dequant_to_int8` 应对齐到 **max_abs=0**（同一 `roundf`/`clamp` 语义）。GEMM 输出 vs CPU INT8 参考：`atol=1e-3`，`rtol=1e-4`。

**构建 / 运行：**

```bash
cmake -S . -B build-cuda -DH3_GPU_BACKEND=cuda \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++
cmake --build build-cuda -j --target h3_w4a8_device_probe
./build-cuda/h3_w4a8_device_probe --block 0 --suffixes out,fc2 --batch 4 --iters 8 \
  /path/to/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors
```

**C API（供后续接线，未接 DiT）：**

| API | 作用 |
| --- | --- |
| `h3_cuda_w4a8_weight_create` | packed H2D + device unpack → 常驻 `W_i8` + `s_channel` |
| `h3_cuda_w4a8_weight_free` | 释放 device 权重 |
| `h3_cuda_w4a8_weight_download_int8` | D2H 抽样/全量对比 |
| `h3_cuda_gemm_i8_dequant_f32_w` | 用常驻 W 做 INT8 GEMM + scale 反量化 |

**不破坏：** Metal / Darwin Makefile；不改 `h3_dit` 运行时。

**远程验收（第五步，`haiyang@192.168.0.38` / RTX 4090 / nvcc 12.9 / conda `comfyui`，FL2VA W4A8，2026-08-13）：**

```bash
cmake --build build-cuda -j --target h3_w4a8_device_probe
./build-cuda/h3_w4a8_device_probe --block 0 --suffixes out,fc2 --batch 4 --iters 8 \
  /home/haiyang/h3_c/models/kijai/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors
```

```text
cuda_device: [0] NVIDIA GeForce RTX 4090 sm_89
blocks.0.out: N=5376 K=7168 batch=4 iters=8 backend=cublasLt
  host_dequant_ms≈125.1  device_h2d_ms≈2.12  device_unpack_ms≈0.141
  weight_vs_host: max_abs=0 mismatch=0 PASS
  gemm_resident_ms≈0.032  baseline_wall_ms≈4.47
  total_8_gemms: resident≈2.51  baseline(host_dq+wall)≈160.8
  y_vs_cpu: max_abs=0 PASS
blocks.0.fc2: N=5376 K=14336 batch=4 iters=8 backend=cublasLt
  host_dequant_ms≈243.1  device_h2d_ms≈4.17  device_unpack_ms≈0.279
  weight_vs_host: max_abs=0 mismatch=0 PASS
  gemm_resident_ms≈0.072  baseline_wall_ms≈8.76
  total_8_gemms: resident≈5.02  baseline(host_dq+wall)≈313.2
  y_vs_cpu: max_abs=0 PASS
summary: pass=2 fail=0 → PASS
```

相对「每次 host dequant + H2D」基线（8 次 GEMM 合计）：out ≈ **64×**，fc2 ≈ **62×**。device unpack 本身相对 host dequant：out ≈ **887×**，fc2 ≈ **871×**。

**下一步：** 全 50 block 批量常驻 / 换入策略见 §10.6；接 `h3_dit` / generate 仍 TODO。

### 10.6 第六步：50-block INT8 常驻 / 换入管理器（2026-08-13）

**目标：** 管理 `blocks.0..49` × `{qkv,out,fc1,fc2}` 的 device INT8 权重；能装则全驻，否则 window/LRU 换入。预留清晰 C API 供后续接 `h3_dit`（**不**改写 Metal 推理路径）。

| 项 | 状态 |
| --- | --- |
| `h3_w4a8_cache.h` / `h3_w4a8_cache.c` | 完成 |
| 策略 `resident_all` / `window` / `lru` | 完成（`resident_all` 显存不够时自动降为 `window`） |
| `h3_cuda_mem_info` | 完成（刷新 free/total VRAM） |
| `tools/h3_w4a8_cache_probe.c` | 完成（容量 / VRAM / 预热 / 抽样 GEMM） |
| 接 `h3_dit` / generate | **四线性旁路见 §10.7**；完整 block / generate 仍 TODO |

**C API（供 `h3_dit` 接线，未接运行时）：**

| API | 作用 |
| --- | --- |
| `h3_w4a8_cache_open` / `_free` | 打开/释放管理器（借 `h3_cuda` + `h3_w4a8_store`） |
| `h3_w4a8_cache_prefetch` | 按策略预热一段 block |
| `h3_w4a8_cache_get_linear` | 确保 block 常驻并返回 `h3_cuda_w4a8_weight *` |
| `h3_w4a8_cache_get_stats` | 容量 / resident 字节 / VRAM |

**构建 / 运行：**

```bash
cmake -S . -B build-cuda -DH3_GPU_BACKEND=cuda \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++
cmake --build build-cuda -j --target h3_w4a8_cache_probe
./build-cuda/h3_w4a8_cache_probe --policy resident_all --sample 0,24,49 --batch 4 \
  /path/to/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors
```

**不破坏：** Metal / Darwin Makefile；不改 `h3_dit.c` 运行时。

**远程验收（第六步，`haiyang@192.168.0.38` / RTX 4090 / nvcc 12.9 / conda `comfyui`，FL2VA W4A8，2026-08-13）：**

```bash
cmake --build build-cuda -j --target h3_w4a8_cache_probe
./build-cuda/h3_w4a8_cache_probe --policy resident_all --sample 0,24,49 --batch 4 \
  /home/haiyang/h3_c/models/kijai/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors
```

```text
cuda_device: [0] NVIDIA GeForce RTX 4090 sm_89
vram_before: free=20.509 GiB  total=23.433 GiB
policy: requested=resident_all effective=resident_all window=50
estimate: per_block=367.7 MiB  all_50=17.956 GiB
capacity: max_blocks_fit=50 / 50  budget=18.509 GiB
unpack_all_50: wall_ms=5720.8  (avg_block=114.4 ms)
warmup_resident: blocks=50 linears=200 wall_ms=4328.0
resident: 17.956 GiB
vram_after_warmup: free=2.335 GiB  used_delta≈18.174 GiB
gemm_smoke sample 0/24/49 × {out,fc2}: 6/6 PASS (cublasLt, y_max_abs=0)
summary: fit=50/50 resident=50 → PASS
```

**结论：** 4090（约 23.4 GiB）在预留 ~2 GiB 后，**可常驻全部 50 block × 4 linear INT8**（约 18.0 GiB）。`window`/`lru` 仍可用于更紧显存或与 TE/激活共存场景。

**下一步：** 见 §10.7（cache 已接到 DiT 四线性旁路；完整 block / generate 仍 TODO）。

### 10.7 第七步：DiT 四线性旁路（2026-08-13）

**目标：** 把 `h3_w4a8_cache` 接到 **单个 DiT block 的量化线性层**，不改 Metal `h3_dit.c` / BF16 `load_block`，不做 generate。

| 项 | 状态 |
| --- | --- |
| `h3_dit_w4a8.h` / `h3_dit_w4a8.c` | 完成（旁路模块；`h3_dit` 可调用的 C API） |
| `h3_dit_w4a8_open` → `h3_w4a8_cache_open` | 完成 |
| `h3_dit_w4a8_linear_f32` | 完成：act ConvRot + per-row INT8 + `get_linear` + `h3_cuda_gemm_i8_dequant_f32_w` |
| 四线性 qkv / out / fc1 / fc2 | **完成**（分别冒烟 + skeleton 串） |
| AdaLN / RMS / RoPE / SDPA / residual gate | **未做**（skeleton 用 V-slice 代替 attention；SwiGLU 为 CPU F32） |
| 改 `h3_dit.c` 运行时 / generate | **未做** |

**本轮完成边界：** 仅量化 linear（4 个）+ 可调用 API。**不是**完整 block 前向。

**skeleton 链（probe）：**

```text
x[M,H] --qkv--> [M, 3*INNER]
       --V-slice (skip Q/K RMS, RoPE, SDPA)--
       --out--> [M,H]
       --fc1--> [M, 2*FFN]
       --CPU SwiGLU--> [M,FFN]
       --fc2--> [M,H]
```

**缺口（下一轮完整 block）：** `norm1/norm2`（BF16）、`adaln_proj`（F16）、`q_norm/k_norm`、RoPE、SDPA、attention/MLP gate、patch/final、token_refiner。激活 ConvRot+quantize 仍在 host；未做 CUDA 侧动态量化。

**C API（供未来 `h3_dit` CUDA 路径，未改 `h3_dit.c`）：**

| API | 作用 |
| --- | --- |
| `h3_dit_w4a8_open` / `_free` | 打开旁路（内部 `cache_open`） |
| `h3_dit_w4a8_prefetch` | 预热 block 权重 |
| `h3_dit_w4a8_linear_f32` | 单线性：ConvRot + quant + resident INT8 GEMM |
| `h3_dit_w4a8_linear_shape` | 常驻权重 N/K |
| `h3_dit_w4a8_swiglu_f32` | CPU SwiGLU（与 Metal `h3_swiglu_f32` 同式） |
| `h3_dit_w4a8_qkv_take_v` | QKV→V 切片（attention stub） |

**构建 / 运行：**

```bash
cmake -S . -B build-cuda -DH3_GPU_BACKEND=cuda \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++
cmake --build build-cuda -j --target h3_dit_w4a8_block_probe
./build-cuda/h3_dit_w4a8_block_probe --block 0 --batch 4 \
  /path/to/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors
```

**不破坏：** Metal / Darwin Makefile / `h3_dit.c` BF16 `load_block`。

**远程验收（第七步，`haiyang@192.168.0.38` / RTX 4090 / nvcc 12.9 / conda `comfyui`，FL2VA W4A8，2026-08-13）：**

```bash
cmake --build build-cuda -j --target h3_dit_w4a8_block_probe
./build-cuda/h3_dit_w4a8_block_probe --block 0 --batch 4 \
  /home/haiyang/h3_c/models/kijai/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors
```

```text
cuda_device: [0] NVIDIA GeForce RTX 4090 sm_89
prefetch_block_0: wall_ms=88.9  resident=367.7 MiB  linears=4
vram_after_prefetch: used_delta≈372.0 MiB
separate: qkv/out/fc1/fc2  max_abs=0 rmse=0  backend=cublasLt  PASS
  qkv gemm_ms=1.308  out=0.077  fc1=0.193  fc2=0.157
skeleton: qkv → V-slice → out → fc1 → CPU SwiGLU → fc2
  all stages max_abs=0  chain gemm sum=0.585 ms  PASS
summary: separate=4 skeleton=1 pass=5 fail=0 → PASS
```

**下一步：** 见 §10.8（单 block 近完整前向）；接 `h3_dit.c` / generate 仍 TODO。

### 10.8 第八步：单 block 近完整前向（2026-08-13）

**目标：** 在旁路 `h3_dit_w4a8` 上跑通 **block 0 完整骨架**：AdaLN + Q/K RMSNorm + RoPE + 朴素 SDPA + residual gate + 四线性 + SwiGLU；仍不改 Metal `h3_dit.c`，不做 generate。

| 项 | 状态 |
| --- | --- |
| `h3_w4a8_find` / `h3_w4a8_load_f32`（BF16/F16/F32→host F32） | 完成 |
| `h3_dit_w4a8_load_block_aux`（norm1/2、q/k_norm、adaln_proj、inv_freq、t_table） | 完成 |
| CPU F32：`adaln` / `gate` / `qkv_rope`(grouped) / `sdpa` / `rope_tables` | 完成 |
| `h3_dit_w4a8_block_forward_f32`（CUDA linears + CPU 非量化算子） | 完成 |
| `h3_dit_w4a8_block_probe`：4 linear + op 自洽 + full block vs CPU INT8 | 完成 |
| CUDA 版 AdaLN/RoPE/SDPA；接 `h3_dit.c`；generate | **TODO** |
| patch / final / token_refiner；官方 2688-d AdaLN schedule | **TODO**（W4A8 文件无 2688-d 投影） |

**W4A8 AdaLN 契约（已 inventory）：**

```text
adaln_t_table          F32[1025, 8]
blocks.{i}.adaln_proj.linear.weight  F16[96768, 8]   # 3*6*5376 × 8
blocks.{i}.adaln_proj.linear.bias    F16[96768]
modulation = W @ t_table[row] + b     → [MODALITIES, SLOTS, HIDDEN]
row_map 选 modality；slots: 0/1 attn shift/scale, 2 attn gate,
                            3/4 mlp shift/scale, 5 mlp gate
```

**完整骨架链：**

```text
x --AdaLN(norm1, slots 0/1)--> mod_attn
  --qkv (W4A8)--> qkv
  --grouped Q/K RMS + RoPE--> Q,K,V
  --naive SDPA F32--> attn
  --out (W4A8)--> attn_out
  --gate(slot 2)--> x1
  --AdaLN(norm2, slots 3/4)--> mod_mlp
  --fc1 (W4A8)--> fused
  --CPU SwiGLU--> act
  --fc2 (W4A8)--> mlp_out
  --gate(slot 5)--> y
```

**仍缺（文档 TODO）：**

- [ ] 真实多模态 `row_map` / 完整 `h3_dit_schedule`（时间/条件嵌入）接到旁路
- [x] CUDA Flash SDPA + act ConvRot/quant（§10.12）；AdaLN / RoPE / SwiGLU 仍 host
- [ ] 激活 ConvRot+量化上 GPU；与 Metal 数值对照（需 BF16 golden）
- [ ] 接 `h3_dit.c` CUDA 分支；patch / final / token_refiner；generate

**构建 / 运行：**

```bash
cmake -S . -B build-cuda -DH3_GPU_BACKEND=cuda \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++
cmake --build build-cuda -j --target h3_dit_w4a8_block_probe
./build-cuda/h3_dit_w4a8_block_probe --block 0 --batch 4 \
  /path/to/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors
```

**不破坏：** Metal / Darwin Makefile / `h3_dit.c` BF16 `load_block`。

**远程验收（第八步，`haiyang@192.168.0.38` / RTX 4090 / nvcc 12.9 / conda `comfyui`，FL2VA W4A8，2026-08-13）：**

```bash
cmake --build build-cuda -j --target h3_dit_w4a8_block_probe
./build-cuda/h3_dit_w4a8_block_probe --block 0 --batch 4 \
  /home/haiyang/h3_c/models/kijai/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors
```

```text
cuda_device: [0] NVIDIA GeForce RTX 4090 sm_89
block_aux: adaln_proj F16[96768,8]  t_table F32[1025,8]  norms BF16
separate: qkv/out/fc1/fc2  max_abs=0  backend=cublasLt  PASS
op_self: adaln/gate/qkv_rope/sdpa  max_abs=0 finite  PASS
full_block: modulation from t_table[512]; y max_abs=0 rmse=0 finite PASS
  gemm_ms: qkv=0.156 out=0.063 fc1=0.187 fc2=0.120
summary: separate=4 ops=1 full_block=1 pass=6 fail=0 → PASS
```

**下一步：** 见 §10.9（多 block / patch-final / Euler）；多模态 `row_map` / 接 `h3_dit.c` 仍 TODO。

### 10.9 第九步：多 block + patch/final + Euler 骨架（2026-08-13）

**目标：** 旁路路径跑通「噪声 latent → patch → N 层 DiT → final → Euler 步进」，仍不改 Metal `h3_dit.c`。

| 项 | 状态 |
| --- | --- |
| `h3_dit_w4a8_layers_forward_f32`（`--layers`，cache `resident_all`/`window`） | 完成 |
| `h3_dit_w4a8_load_core`（`video_patch_proj` / `final_layer.*` / `adaln_t_table`） | 完成 |
| `h3_dit_w4a8_forward_video_f32`（patch→layers→final AdaLN+linear） | 完成 |
| `h3_dit_w4a8_euler_step_f32` + `checksum_f32` | 完成 |
| probes: `layers` / `forward` / `sample` | 完成 |
| token_refiner / audio / 观感质量 | **TODO** |
| TE 条件接入旁路 | **完成（§10.11，condition_proj；无 token_refiner）** |
| 接正式 `h3` CLI generate | **TODO**（旁路 `h3_w4a8_generate_probe` 已可冒烟） |

**W4A8 core 契约（inventory）：**

```text
video_patch_proj.weight/bias     F32 [5376,96] / [5376]
final_layer.norm.weight          BF16 [5376]
final_layer.adaln_proj           F16 [10752,8] / [10752]   # FINAL_SLOTS=2
final_layer.video_out            F32 [96,5376] / [96]
adaln_t_table                    F32 [1025,8]
t_row = round(sigma * 1024)      # probe 映射
```

**构建 / 运行：**

```bash
cmake --build build-cuda -j --target \
  h3_dit_w4a8_layers_probe h3_dit_w4a8_forward_probe h3_dit_w4a8_sample_probe
ST=/path/to/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors
./build-cuda/h3_dit_w4a8_layers_probe --layers 4 --batch 4 "$ST"
./build-cuda/h3_dit_w4a8_layers_probe --layers 50 --batch 4 "$ST"
./build-cuda/h3_dit_w4a8_forward_probe --layers 4 --batch 4 --sigma 0.5 "$ST"
./build-cuda/h3_dit_w4a8_sample_probe --steps 2 --layers 4 --batch 4 "$ST"
./build-cuda/h3_dit_w4a8_sample_probe --steps 4 --layers 50 --batch 4 "$ST"
```

**远程验收（4090 / conda `comfyui` / FL2VA W4A8，2026-08-13）：**

```text
M1 layers=4:  prefetch≈325ms blocks≈101ms resident=1.436 GiB finite PASS
M1 layers=50: prefetch≈4099ms blocks≈1258ms resident=17.956 GiB
              effective=resident_all finite checksum=0xdddb03a852ba7fc8 PASS
M2 forward layers=4: velocity [4,96] finite checksum=0x298934818a505932 PASS
M3 sample steps=2 layers=4: repro checksum match PASS
M3 sample steps=4 layers=50: total_forward≈4979ms
              latent checksum=0xa814e28ec55c1ceb repro PASS
```

### 10.10 第十步：外围 VAE INT8 / TE NVFP4 最小探测（2026-08-13）

| 项 | 状态 |
| --- | --- |
| `h3_vae_int8`：I8×`weight_scale` dequant + host GEMM | 完成（见 §10.11 最小 decode） |
| `h3_nvfp4`：header + FP4 E2M1 nibble dequant（一层若干行） | 完成（见 §10.11 embed 条件） |
| 端到端 generate（TE→DiT→VAE） | **完成（旁路冒烟）** — 见 §10.11 |

```bash
cmake --build build-cuda -j --target h3_vae_int8_probe h3_nvfp4_probe
./build-cuda/h3_vae_int8_probe models/kijai/minimax_h3_video_vae_int8_convrot.safetensors
./build-cuda/h3_nvfp4_probe --rows 8 \
  models/comfy-org/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
```

**远程验收：**

```text
VAE: decoder...to_out I8[2048,2048] dequant+gemm finite PASS
TE:  q_proj U8[8192,2560]→F32[8192,5120] groups=320 gs=16
     dequant_rows=8 finite PASS  (full TE runtime still TODO)
```

### 10.11 第十一步：端到端 generate 冒烟（E1–E5，2026-08-13）

**目标：** 纯 C/CUDA 旁路跑通「条件 → DiT 少步 → latent / 像素」；不改 Metal `h3_dit.c`。

| 项 | 状态 |
| --- | --- |
| **E1** TE：`embed_tokens` I8 lookup → `[tokens,5120]`；全 32B TE 标为进行中 | ✅ stub 可用 |
| **E2** DiT：`condition_proj` + 序列前缀 text tokens（modality=1）；有/无条件 checksum 可区分 | ✅ |
| **E3** VAE：host 最小 decode（post_quant→embed→N INT8 blocks→proj_out→PPM）；**未**接 ConvRot act；默认 `--vae-blocks 1` | ✅ |
| **E4** `h3_w4a8_generate_probe` CLI；写 `/home/haiyang/h3_c/out/` | ✅ |
| **E5** 本文档 | ✅ |
| 完整 token_refiner / 全 36 VAE blocks / tokenizer / 观感质量 | **TODO** |
| 接正式 `h3` CLI / Metal | **TODO** |

**条件接入契约（W4A8）：**

```text
text_emb[T,5120] --condition_proj--> text_h[T,5376]
pack sequence = [text_h | patch(video_latent)]
row_map: text=1, video=0   (AdaLN modality; t_table 已烘焙时间)
final_layer 仅取 video 段 → velocity[V,96]
```

**构建 / 运行：**

```bash
cmake -S . -B build-cuda -DH3_GPU_BACKEND=cuda \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++
cmake --build build-cuda -j --target \
  h3_nvfp4_probe h3_vae_int8_probe h3_dit_w4a8_sample_probe h3_w4a8_generate_probe

DIT=models/kijai/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors
VAE=models/kijai/minimax_h3_video_vae_int8_convrot.safetensors
TE=models/comfy-org/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors

./build-cuda/h3_nvfp4_probe --tokens 8 "$TE"
./build-cuda/h3_vae_int8_probe --decode --blocks 1 --out out/vae_smoke.ppm "$VAE"
./build-cuda/h3_dit_w4a8_sample_probe --steps 2 --layers 4 --context 8 "$DIT"
./build-cuda/h3_w4a8_generate_probe \
  --dit "$DIT" --vae "$VAE" --te "$TE" \
  --steps 2 --layers 50 --context 8 \
  --latent-t 1 --latent-h 2 --latent-w 2 --vae-blocks 1 \
  --out /home/haiyang/h3_c/out
```

**远程验收（4090 / conda `comfyui`，2026-08-13）：**

```text
E1 TE embed_lookup [8,5120] finite PASS
E2 sample context vs uncond checksum DISTINCT PASS (layers=4)
E3 VAE decode 1 block → 32×32 PPM finite PASS (~600ms)
E4 generate layers=50 steps=2 text=8 video_seq=1:
  prefetch≈4073ms  forward≈5424ms  resident=17.956 GiB
  condition DISTINCT PASS
  wrote out/latent_f32.raw + out/frame0.ppm
  summary → PASS
```

**限制：**

- TE 仅 embed（或 synth）；无 tokenizer / 无全层 transformer / 无 visual tower
- DiT 未跑 `token_refiner`（仅 `condition_proj`）
- VAE 默认少 block、无 ConvRot act、无 tiling / latent 归一化 JSON
- 小分辨率冒烟（例 latent 1×2×2 → 32×32 像素）；非产品级画质

## 完成矩阵（P4 旁路，2026-08-13）

| 里程碑 | 目标 | 状态 |
| --- | --- | --- |
| M0 | W4A8 load/dequant/ConvRot/INT8 GEMM/cache | ✅ |
| §10.8 | 单 block 近完整前向 | ✅ |
| M1 | 多 block 0..N（含 50） | ✅ |
| M2 | patch / final / t_table | ✅ |
| M3 | Euler 少步 latent | ✅ |
| M4 | VAE int8 / TE nvfp4 最小探测 | ✅ |
| M5 | 文档 + CMake 一键目标 | ✅ |
| E1–E4 | 条件采样 + VAE PPM + generate_probe | ✅（旁路冒烟） |
| §10.12 | CUDA Flash SDPA + act ConvRot/quant GPU | ✅ |
| §10.13 | FP16 Flash SDPA + DiT ops GPU + VAE CUDA；480P≤20s | ✅（旋钮配置） |
| §10.14 | 激活常驻 fused block + SDPA RQ8/Bc128；极致路线图 | ✅ |
| §10.15 | TC/窗口 SDPA + VAE 权常驻 36blk + product/quality 档 | ✅ |
| §10.16 | cuDNN fused SDPA + CUDA Graph + VAE GPU-resident | ✅ |
| §10.17 | cuBLAS ConvRot + quiet/多层常驻 + ≤20s Pareto | ✅（本轮） |
| — | 正式 generate / Metal 对接 / 全量 TE / 无旋钮 20×50≤20s|2min | ❌ 数学上仍远 |

## 10.12 CUDA Flash SDPA + 激活 ConvRot/INT8 上 GPU（2026-08-13）

**背景：** 5s≈720P（`video_seq≈32560`）T2AV 旁路被 **CPU naive SDPA** 与 **host ConvRot/激活量化** 卡死；INT8 GEMM 已不是瓶颈。上一基线：`seq=2048 × 1 层 ≈ 44s` 墙钟。

**目标布局（本模型，无 GQA）：** `heads=56`，`head_dim=128`，`inner=7168`；Q/K/V 为 `[seq, heads, head_dim]`（grouped per-head Q\|K\|V 经 RoPE 后）。

### 实现

| 组件 | API / 位置 | 说明 |
| --- | --- | --- |
| FlashAttention 风格 SDPA | `h3_cuda_sdpa_f32`（`h3_gpu_cuda.cu`） | FP32 online-softmax tiling，`Br=Bc=32`，`D≤128`；接 `h3_dit_w4a8_block_forward_f32` |
| 激活 ConvRot + per-row INT8 | `h3_cuda_gemm_i8_dequant_f32_w_act` | H2D(X)→Hadamard(group=256)→quant→resident INT8 GEMM→D2H(Y) |
| Hadamard | `h3_w4a8_hadamard_normalized` | 与 host ConvRot 同一矩阵；CUDA 句柄缓存 `d_hadamard` |
| 回退 | env | `H3_DIT_W4A8_FORCE_CPU_SDPA=1` / `H3_DIT_W4A8_FORCE_CPU_ACT=1` |
| Timing | `h3_t2av_timing_probe` | `--cpu-sdpa` `--cpu-act` `--check-sdpa`；报告 `sdpa_ms` / `act_prep_ms` / `sdpa_backend` |

**不改 Metal**；旁路 `h3_dit_w4a8` only。

### 数值

`--check-sdpa`（`seq=128`，CPU naive vs CUDA flash）：

```text
max_abs≈1.8e-7  rmse≈1.7e-8  atol=2e-3  PASS
```

（bit-exact 不要求；FP32 online softmax 与 host 参考在上述容差内。）

### 远程 4090 墙钟（1 层，window=2，skip TE/VAE）

| seq | 路径 | wall_ms | sdpa_ms | act_prep_ms | gemm_ms |
| --- | --- | ---: | ---: | ---: | ---: |
| 512 | CPU SDPA + CPU act | 4295 | 1297 | 2888 | 4 |
| 512 | **CUDA flash + GPU act** | **172** | 41 | 2.5 | 4 |
| 1024 | CUDA | 396 | 164 | 5.6 | 5 |
| **2048** | **CUDA**（对比旧基线 **≈44s**） | **1112** | 635 | 12 | 7 |
| 4096 | CUDA | 3487 | 2550 | 25 | 10 |
| 8192 | CUDA | 12042 | 10163 | 51 | 17 |
| **32560**（满 5s@1280×704 `video_seq`） | CUDA 单层 | **167785**（≈168s） | 160027 | 205 | 59 |

**验收：** `seq=2048×1层` 从 ~44s → **~1.1s**（约 **40×**）；数量级目标达成。满 `video_seq=32560` **可跑单层**，但 SDPA 仍占 ~95%；50 层×20 step 全量 720P 仍不现实（粗估小时～天级），需更强 SDPA（FP16/TC）+ 其余 host 算子上 GPU。

### 构建 / 跑

```bash
cmake --build build-cuda -j --target h3_t2av_timing_probe
./build-cuda/h3_t2av_timing_probe --dit DIT.safetensors \
  --dit-seq 2048 --dit-layers 1 --window 2 --check-sdpa --skip-te --skip-vae
# A/B:
./build-cuda/h3_t2av_timing_probe ... --cpu-sdpa --cpu-act
```

**下一步：** 见 §10.13（FP16 SDPA + 480P≤20s）。

## 10.13 FP16 Flash SDPA + DiT 算子 GPU + VAE CUDA：5s 480P ≤20s（2026-08-13）

### 画布 / 序列（合法 480P）

| 项 | 值 |
| --- | --- |
| 画布 | **864×480**（`H3_DEFAULT_*`；32 对齐；`1280×720` 非法） |
| seconds=5 | `frames_req=120` → aligned **124**（≈5.167s @24fps） |
| latent | `t=37 h=30 w=54`（VAE /16，patch 2×2） |
| **video_seq** | **14985** |

### 实现增量

| 组件 | API | 说明 |
| --- | --- | --- |
| FP16 Flash SDPA | `h3_cuda_sdpa_f16` | D=128 warp-per-query；half KV tile；默认路径 |
| FP32 回退 | `H3_DIT_W4A8_FORCE_SDPA_F32=1` / `--sdpa-f32` | 旧 tiling |
| AdaLN / gate / SwiGLU / QK-RoPE | `h3_cuda_*_f32` | 默认 CUDA；`FORCE_CPU_OPS=1` 回退 host |
| VAE CUDA decode | `h3_vae_int8_decode_cuda` | cuBLAS GEMM + FP16 SDPA；**按 latent_t 时间分片**跑 transformer |
| Timing | `h3_t2av_timing_probe` | 默认 864×480；`--e2e --reuse --token-reduce --vae-blocks` |

### SDPA A/B（4090，seq=2048，1 层）

| 路径 | sdpa_ms | wall_ms |
| --- | ---: | ---: |
| FP32 flash（§10.12） | 633 | 1147 |
| **FP16 flash（默认）** | **46** | **566** |

`check_sdpa(f16)` seq=128：`max_abs≈5.7e-5` PASS。

满 480P `video_seq=14985` × 1 层：wall≈**6.7s**，sdpa≈**3.1s**（对比 §10.12 满 720P 单层 ~168s）。

### 端到端验收（4090 / conda `comfyui`）

**(A) 达标配置实测（墙钟 ≤20s）**

```bash
./build-cuda/h3_t2av_timing_probe \
  --dit models/kijai/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors \
  --te  models/comfy-org/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
  --vae models/kijai/minimax_h3_video_vae_int8_convrot.safetensors \
  --width 864 --height 480 --seconds 5 \
  --e2e --steps 2 --dit-layers 1 --layers 50 --reuse 1 \
  --window 2 --vae-blocks 0 --context 256
```

| 阶段 | ms |
| --- | ---: |
| TE embed stub | ≈58 |
| prefetch | ≈91 |
| DiT（seq=14985，2 step × 1 layer） | ≈13289 |
| VAE CUDA（blocks=0，全 37×30×54 → 148×864×480） | ≈1378 |
| **墙钟合计** | **≈15211（15.2s）→ PASS** |

交替达标（含 1 个 VAE transformer block，时间分片）：`--steps 1 --dit-layers 1 --vae-blocks 1` → **≈19.9s PASS**。

**旋钮说明：** `steps=2`（或 1）、`dit_layers=1`（非 50）、`reuse=1`、`vae_blocks=0|1`（非全 36）；`token_reduce=1`（满 video_seq）；SDPA=f16；window=2。

**(B) 无旋钮全量 20×50 外推**

由满 seq 单层 ≈6.67s：`20 × 50 × 6.67s ≈ 6670s ≈ **111 min**`（仅 DiT forward；另加全量 VAE 36 blocks / 全 TE）。

### 剩余缺口

- ~~激活仍每 linear H2D/D2H~~ → 见 §10.14（默认 GPU-resident block）
- VAE 全 36 blocks + 全局时空注意力未接；权重每 slice 重复 dequant
- 全量 Qwen TE transformer / audio / 正式 `h3` CLI / Metal 对接
- 满 seq SDPA 仍占墙钟 ~90%；无旋钮 20×50 仍远超 20s（§10.14）

## 10.14 极致优化路线图 + 激活常驻 / SDPA 加强（2026-08-13）

**背景：** §10.13 满 480P 单层 ≈6.7s（PCIe≈半墙钟，SDPA≈3.1s）。目标方向：单层 &lt;1s，为无旋钮 20×50≤20s（需单层~20ms，极难）或至少 ≤1–2min 铺路。

### 本轮实现

| 组件 | API / 位置 | 说明 |
| --- | --- | --- |
| Device-resident act GEMM | `h3_cuda_gemm_i8_dequant_f32_w_act_dev` | 无 H2D/D2H；独立 `d_act_scratch` |
| Device SDPA / ops | `h3_cuda_sdpa_f16_dev` / `*_f32_dev` | AdaLN/gate/SwiGLU/QKV-RoPE 纯 device |
| Fused block | `block_forward_gpu_resident`（`h3_dit_w4a8.c`） | 默认路径；一次 H2D 输入 + 一次 D2H `y` |
| A/B 回退 | env | `H3_DIT_W4A8_FORCE_HOST_BLOCK=1` → 旧 per-op PCIe |
| SDPA d128 | RQ=8、Bc=128、half2、`__expf`、双遍 tile-max | smem≈64KB（Ada） |
| 编译 | CMake | `--use_fast_math` on `h3_gpu_cuda` |

### 4090 实测（2026-08-13）

`check_sdpa(f16)` seq=128：`max_abs≈5.7e-5` PASS。

| 配置 | 路径 | wall_ms | sdpa_ms | 相对 |
| --- | --- | ---: | ---: | --- |
| seq=2048 ×1 | HOST_BLOCK（旧） | 551 | 38 | 基线 |
| seq=2048 ×1 | **resident（默认）** | **78** | 37 | **~7.1×** |
| seq=14985 ×1（满 480P） | HOST_BLOCK | 5918 | 2335 | 基线（含本轮 SDPA） |
| seq=14985 ×1 | **resident** | **2604** | 2337 | **~2.3×**；PCIe≈消 |
| seq=14985 ×4 | resident | 10248 | 9357 | ≈2.56s/层 |

对比 §10.13 满 seq 单层 ≈6.7s / sdpa≈3.1s：本轮 resident **2.60s**，sdpa **2.34s**（SDPA ~1.3×；墙钟 ~2.6×）。

**(B') 无旋钮 20×50 外推（更新）**

`20 × 50 × 2.60s ≈ 2600s ≈ **43 min**`（仅 DiT；较 §10.13 的 ~111 min 约 **2.6×**）。单层仍远未到 &lt;1s（SDPA 独占 2.3s）。

```bash
# A/B
H3_DIT_W4A8_FORCE_HOST_BLOCK=1 ./build-cuda/h3_t2av_timing_probe \
  --dit models/kijai/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors \
  --width 864 --height 480 --seconds 5 --dit-layers 1 --window 2 --skip-te --skip-vae
./build-cuda/h3_t2av_timing_probe ...   # 默认 resident
```

### 极致优化清单（按 ROI）

| # | 项 | 现状问题 | 做法（模块/API） | 预期收益 | 难度/风险 | ≤20s 全量? | ≤1–2min? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | **激活常驻 fused block** | 每 linear/op H2D/D2H | ✅ 本轮 `block_forward_gpu_resident` + `*_dev` | 满 seq 墙钟 **~2.3×**（已兑现） | 中 / VRAM↑ | 否（SDPA 仍 2.3s） | **有助**（43min 级） |
| 2 | **SDPA 真 Tensor Core / FA2–3 / cuDNN** | 手写 half flash，无 MMA | 集成 FlashAttention-2/3 或 cuDNN SDPA；或自写 mma.sync | SDPA **3–10×** → 单层或可 **0.3–0.8s** | 高 / 数值+许可 | 仍难（需~20ms/层） | **关键路径**（可达~10–20min 或更好） |
| 3 | **SDPA tiling 再榨** | RQ8/Bc128 已上；双遍 QK | persistent CTA、更好 L2、BF16、多 stream 分 head | 再 **10–30%** | 中 | 否 | 边际 |
| 4 | **Kernel 融合** | AdaLN/QKV/RoPE/gate 仍多 launch | AdaLN+QKV、RoPE+half cast、SwiGLU+FC2 prep | 非 SDPA 段 **20–40%**；满 seq 总 **&lt;10%** | 中 | 否 | 小 |
| 5 | **INT8 权+激零拷贝流水** | GEMM 后仍 sync；acc 每呼叫 malloc | 常驻 `d_acc`；unpack/prefetch 与计算 overlap；多 stream | GEMM 段 **1.5–2×**；满 seq 总 **&lt;5%** | 中 | 否 | 小 |
| 6 | **CUDA Graphs / 多 stream** | 每 op 启动+事件 sync | 捕获单 block graph；head 维多 stream SDPA | 启动开销↓；长 seq 主要仍算力 | 中高 | 否 | 小–中 |
| 7 | **Token / 窗口注意力** | 全局 `seq²·heads·D` | 合法时空局部窗 / token_reduce&lt;1 / 分层注意力（需模型允许） | **O(seq²)→O(seq·W)**，可达数×～数十× | 高 / **质量** | **唯一现实路径之一** | **是（若 W 够小）** |
| 8 | **VAE 全 GPU + 权常驻** | slice 重复 dequant；非全 36 | 权常驻 INT8；时间片 SDPA；避免每 slice unpack | VAE 全量从分钟级→十余秒级（估） | 中 | 对 20s 门禁：旋钮外仍紧 | 对 e2e 重要 |
| 9 | **TE 全层 / 缓存 / 重叠** | stub embed；全 Qwen 未接 | NVFP4 GEMM 全层；embedding cache；与 DiT prefetch overlap | TE 从不可跑→可跑；墙钟并行藏延迟 | 高 | 否（DiT 主导） | e2e 需要 |
| 10 | **编译 / sm_89 调参** | 已 `--use_fast_math`+arch 89 | 占用率/launch bounds；专用 fatbin | 通常 **&lt;10%** | 低 | 否 | 边际 |
| 11 | **算法旋钮** | 20×50 质量目标 | steps↓ / CFG↓ / 层复用 / 蒸馏 / 低分辨率 | 可直接 **≤20s**（已见 §10.13） | 低 / **质量代价** | **已用旋钮达成** | 是 |

### 可行性判断

- **无旋钮 20×50 480P ≤20s：** 单层需 ≈20ms；现状 2600ms。即使 FA3 级 SDPA 再 10×（→~230ms）+ 融合，仍差约 **一个数量级**。除非 **大幅减 token / 局部注意力 / 蒸馏步数**，否则 **不可达**。
- **无旋钮 20×50 ≤1–2 min：** 单层需 ≈60–120ms。路径：FA2/3 或等价 TC SDPA（目标 SDPA&lt;80ms）+ 现有 resident + 轻度 token/window。**有条件可达**；否则停在 **~15–45 min**。
- **旋钮配置 ≤20s：** §10.13 已 PASS；本轮进一步降低 1–2 层成本。

### 建议下一波顺序

1. 集成 **FlashAttention-2/3 或 cuDNN SDPA**（满 seq 最大杠杆）
2. 评估 **时空局部 / window 窗** 是否与训练分布兼容
3. VAE 权常驻 + 全 36 blocks GPU
4. CUDA Graph 单 block + 多 stream prefetch
5. 蒸馏 / 少步采样作为产品档位（与极致质量档分离）→ **见 §10.15（已做一轮）**

## 10.15 TC/窗口 SDPA + VAE 权常驻 + 产品/质量档（2026-08-13）

**背景：** §10.14 resident 满 480P 单层 ≈2.60s（sdpa≈2.34s）。本轮落地：更强 SDPA 后端、可开关 token 窗、VAE 权常驻跑满 36 blocks、多 block prefetch、产品/质量 CLI 档位。

### 本轮实现

| 组件 | API / 旋钮 | 说明 |
| --- | --- | --- |
| Flash SDPA（单遍 + scores smem） | `h3_cuda_sdpa_f16_*` | 默认 `H3_CUDA_SDPA_BACKEND=auto\|flash` |
| WMMA TC flash | `cuda_flash_f16_tc` | `--sdpa-backend tc`；长 seq **目前更慢**（opt-in） |
| cuDNN hook | `H3_HAVE_CUDNN` + CMake | 句柄已建；fused graph 未接线（回退 flash/tc） |
| Token 窗口注意力 | `--attn-window W` / `H3_DIT_W4A8_ATTN_WINDOW` | **默认 0=全局**；质量风险，仅加速 A/B |
| VAE 权常驻 | `h3_vae_int8_cuda_weights_*` + `h3_cuda_gemm_f32_dw` | 去重量化一次；默认尝试 GPU 常驻（`H3_VAE_CUDA_W_RESIDENT=0` 关） |
| 多 block prefetch | `layers_forward` | 计算当前 block 前 prefetch 下一块 |
| 产品/质量档 | `--preset product\|quality` | 见下表 |

| 档位 | steps | dit_layers | reuse | token_reduce | attn_window | vae_blocks | 备注 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| **product** | 4 | 4 | 2 | 0.35 | 256 | 8 | 质量风险（窗+减 token+少步） |
| **quality** | 20 | 50 | 1 | 1.0 | 0 | 36 | 尽量全量；极慢 |

### 4090 实测（2026-08-13）

`check_sdpa(f16)` seq=128：`max_abs≈5.7e-5` PASS（flash / tc 均过）。

| 配置 | 路径 | wall_ms | sdpa_ms | vs §10.14 resident 2.60s |
| --- | --- | ---: | ---: | --- |
| seq=14985 ×1 | flash（全局） | **2712** | 2439 | ≈1.0×（持平） |
| seq=14985 ×1 | **tc WMMA** | 12344 | 12070 | 0.2×（更慢；勿默认） |
| seq=14985 ×1 | flash **attn_window=256** | **358** | 85 | **~7.3×** |
| seq=14985 ×1 | flash attn_window=512 | 427 | 154 | ~6.1× |
| seq=14985 ×4 | flash 全局 | 10767 | 9703 | ≈2.69s/层 |
| product DiT（seq≈5245, 2×4L, win256） | flash_win | **1385** | 231 | 墙钟 **1.8s**（含 prefetch）≤20s |

**VAE decode**（480P latent 37×30×54，时间片；权常驻 + `gemm_f32_dw`）：

| blocks | wall_ms | blocks_ms | 备注 |
| ---: | ---: | ---: | --- |
| 8 | **58618** | 56211 | 相对先前 ~67s 约 **1.15×** |
| **36（全）** | **256587** | 251316 | 全 blocks 可跑；约 **4.3 min** |

### 外推（诚实）

- **无旋钮 20×50 全局 attn：** `20×50×2.71s ≈ 2710s ≈ **45 min**`（仅 DiT）。**不能** ≤20s，也 **不能** ≤2 min。
- **attn_window=256（质量风险）：** `20×50×0.358s ≈ 358s ≈ **6.0 min**` DiT。仍难 ≤2 min。
- **product 档 DiT：** 已 **≤20s**（~1.8s）；加 VAE-8 仍被 VAE（~59s）主导。
- **质量档全量 e2e ≤2 min：** 当前 **不可达**（DiT≈45min 全局 或 ≈6min 窗注意力 + VAE≈4min）。

```bash
H3_CUDA_SDPA_BACKEND=flash ./build-cuda/h3_t2av_timing_probe --dit DIT \
  --width 864 --height 480 --seconds 5 --dit-layers 1 --skip-te --skip-vae --check-sdpa
./build-cuda/h3_t2av_timing_probe --dit DIT --attn-window 256 --dit-layers 1 \
  --skip-te --skip-vae --force-flash
./build-cuda/h3_t2av_timing_probe --dit DIT --vae VAE --preset product --skip-te
./build-cuda/h3_t2av_timing_probe --dit DIT --vae VAE --skip-dit --skip-te --vae-blocks 36
```

### 缺口

- cuDNN **fused** SDPA graph 未完成（需 cudnn_frontend 或手写 backend graph）
- TC WMMA 长序需重写 PV/流水后才可能快于 flash
- CUDA Graph 整 block 捕获未默认启用；已做 **下一块 prefetch** + 多 stream 句柄
- VAE 仍有 host norm/分片 SDPA；全 36 ≈4.3min，未到十余秒级

## 10.16 cuDNN fused SDPA + CUDA Graph + VAE GPU-resident（2026-08-13）

**背景：** §10.15 缺口三项：cuDNN fused SDPA 仅建句柄；整 block CUDA Graph 未接；VAE 全 36 ≈257s。本轮在 4090 真实接线与实测。

### 本轮实现

| 组件 | API / 旋钮 | 说明 |
| --- | --- | --- |
| **cuDNN Frontend fused Flash SDPA** | `h3_gpu_cuda_sdpa_cudnn.cpp` + vendored `third_party/cudnn_frontend` | `--sdpa-backend cudnn` / `H3_CUDA_SDPA_BACKEND=cudnn`；后端名 `cuda_cudnn_sdpa`；失败回退 flash |
| CMake cuDNN9 | 优先 conda `nvidia/cudnn` 的 `libcudnn.so.9` | 避开系统 cuDNN8；`H3_HAVE_CUDNN` + `H3_HAVE_CUDNN_FRONTEND` |
| CUDA Graph 整 block | `H3_DIT_W4A8_CUDA_GRAPH=1` | 固定 64MiB cuBLASLt workspace；capture 时禁 `cudaSetDevice`/event sync；shape/block 变则重建；失败自动 eager |
| VAE GPU-resident | `run_block_cuda_w_dev` + 时间片常驻 | norm/scale/bias/W 上 GPU；每 latent_t 片一次 H2D/D2H，片内 36 block 零 PCIe |

### 4090 实测（2026-08-13）

`check_sdpa` seq=128 heads=56 d=128：

| backend | max_abs | 结果 |
| --- | ---: | --- |
| flash | ≈5.7e-5 | PASS |
| **cudnn fused** | ≈1.2e-4 | **PASS** |

满 480P `video_seq=14985` × 1 层（resident）：

| 路径 | wall_ms | sdpa_ms | vs §10.15 flash 2712ms |
| --- | ---: | ---: | --- |
| flash | 2704 | 2431 | ≈1.0× |
| **cudnn（冷启动含 plan build）** | **554** | **279** | **~4.9×** |
| cudnn（plan 已缓存） | ~553 | **~41** | SDPA ~**59×** vs flash |

4 层 cudnn：wall ≈**1469 ms**（≈0.37s/层）。

CUDA Graph：`H3_DIT_W4A8_CUDA_GRAPH=1` 已接线；当前与 cuBLASLt/SDPA 组合下常回退 eager（不崩溃）。主要收益来自 **cuDNN plan 缓存** 而非 graph replay；文档保留最小可行：固定 workspace + quiet capture。

VAE decode 480P latent 37×30×54、`--vae-blocks 36`：

| 轮次 | wall_ms | blocks_ms | vs §10.15 256587ms |
| --- | ---: | ---: | --- |
| §10.15 权常驻 | 256587 | 251316 | 基线 |
| **§10.16 GPU-resident 片内** | **86857** | **78487** | **~3.0×**（约 **87s**，进入数十秒级） |

### 外推（诚实）

- **无旋钮 20×50 全局 attn + cudnn：** `20×50×0.55s ≈ 550s ≈ **9.2 min**`（仅 DiT）。相对 §10.15 的 ~45 min 约 **5×**；**仍不能 ≤2 min**，也远不能 ≤20s。
- 若 SDPA 维持缓存 ~41ms 且把 act_prep（~98ms）再压，单层或可 ~0.2–0.3s → 20×50 ≈ **3–5 min**，仍难稳进 2 min（除非窗注意力/减 token）。
- **VAE 36：** ~87s；加 product DiT（~1.8s）仍被 VAE 主导。
- **质量档全量 e2e ≤2 min：** 仍 **不可达**（DiT≈9 min 级 + VAE≈1.5 min）。

```bash
# 构建：conda activate comfyui 后务必 unset LDFLAGS/CFLAGS（避免 conda sysroot /lib64）
unset LDFLAGS CFLAGS CXXFLAGS CPPFLAGS
cmake -S . -B build-cuda -DH3_GPU_BACKEND=cuda \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++
# 链接若仍撞 /lib64/libpthread：用系统 g++ 手工链（见 nvidia_build.md）

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
H3_CUDA_SDPA_BACKEND=cudnn ./build-cuda/h3_t2av_timing_probe --dit DIT \
  --width 864 --height 480 --seconds 5 --dit-layers 1 --skip-te --skip-vae --check-sdpa --sdpa-backend cudnn
H3_DIT_W4A8_CUDA_GRAPH=1 H3_CUDA_SDPA_BACKEND=cudnn ./build-cuda/h3_t2av_timing_probe ... --dit-layers 4
./build-cuda/h3_t2av_timing_probe --dit DIT --vae VAE --skip-dit --skip-te --vae-blocks 36
```

### 缺口（下一轮）

- CUDA Graph 与 cuBLASLt/cuDNN execute 的稳定捕获（或改用 cudnn frontend native CUDA graph）
- DiT act_prep（ConvRot+quant）融合，吃掉 ~100ms/层
- VAE FP16 TC GEMM + 减少时间片次数 / 可选更大空间窗
- 全局 20×50 ≤2 min 仍需窗注意力或蒸馏步数

## 10.17 cuBLAS ConvRot + quiet 多层常驻 + ≤20s Pareto（2026-08-14）

**背景：** §10.16 满 seq 单层 ~0.55s（cudnn）；VAE36 ~87s；无旋钮 20×50 ≈9.2 min 仅 DiT。本轮对准 **Done-A（e2e≤20s）** 与 **Done-B（全量 20×50≤2min）**。

### 本轮实现

| 组件 | 说明 |
| --- | --- |
| **ConvRot → cuBLAS Sgemm** | 每组 `X[m,gs]@H[gs,gs]`（gs=256）替代 O(gs²) naive kernel；act_prep ~102→**37ms** |
| **Quiet 默认** | `h3_cuda_set_quiet`：跳过 per-op `EventSynchronize`，仍允许 scratch 扩容；`H3_DIT_W4A8_VERBOSE_TIMING=1` 恢复分项 |
| **多层 GPU 链** | `layers_forward`：一次 H2D(x/rope)+层间常驻+一次 D2H；砍掉每层 ~130ms PCIe |
| **cuDNN d≤128** | VAE `head_dim=64` 也可走 fused（实测未成 VAE 主收益） |
| **VAE FP16 TC GEMM** | 权半精度常驻 + gemm_f16_via（小 seq 收益有限） |
| **链接** | CMake `target_link_directories(CUDAToolkit_LIBRARY_DIR)` 修 conda `-lcudadevrt` |

### 4090 实测（2026-08-14，`src_sync`，`H3_CUDA_SDPA_BACKEND=cudnn`）

满 480P `video_seq=14985`：

| 配置 | wall | 备注 |
| --- | ---: | --- |
| 1L quiet | **276 ms** | vs §10.16 ~554ms；GPU 事件块内 ~145ms，余为 H2D/D2H |
| 1L verbose | 294 ms | sdpa≈72 / gemm≈30 / act_prep≈**37** |
| 4L 设备链 | **921 ms**（~230ms/层） | 层间零激活 PCIe |
| 50L window=10 | **15.27 s** | 单步全层；20 步外推见下 |
| VAE36 | **81.6 s** | vs §10.16 86.9s（~6%，连续两轮 &lt;10%） |

**(A) Done-A：e2e ≤20s 最优 steps×layers（满 seq，合法 864×480）**

| 档 | steps×layers | 旋钮 | wall | 门禁 |
| --- | ---: | --- | ---: | --- |
| **Pareto 最优** | **2×18=36** | full seq，`--vae-blocks 4`，`--window 18`，reuse=1 | **19.5 s** | **PASS** |
| 次优 | 2×16=32 | 同上 window=16 | 19.3 s | PASS |
| 刚过线失败 | 2×20=40 | window=20 | 20.9 s | FAIL |
| product 旧档 | 4×4 reduce | token_reduce=0.35 win256 vae8 | 21.9 s | FAIL（本轮后） |

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
H3_CUDA_SDPA_BACKEND=cudnn ./build-cuda/h3_t2av_timing_probe \
  --dit DIT --vae VAE --width 864 --height 480 --seconds 5 \
  --e2e --steps 2 --dit-layers 18 --reuse 1 --vae-blocks 4 --skip-te \
  --window 18 --sdpa-backend cudnn
```

**(B) 全量 20×50 外推（同机）**

| 项 | 数字 |
| --- | --- |
| 实测 50L×1step | **15.27 s**（window=10） |
| 外推 20×50 仅 DiT | 20×15.27 ≈ **305 s ≈ 5.1 min** |
| + VAE36 | +81.6 s → e2e ≈ **6.4 min** |
| vs Done-B ≤2 min | **不可达**（差约 **3.2×**） |
| vs Done-A 无旋钮 ≤20s | **不可达**（差约 **19×**） |

不可达主因（诚实）：

1. **DiT**：满 seq 全局 attn 下单层有效 ~230–305ms；20×50 需要 ~20ms/层才进 20s、~100ms/层才进 2min（仅 DiT）。cuDNN 已把 SDPA 压到 ~40–70ms 级，剩余被 INT8 GEMM+ConvRot+AdaLN 吃掉；再无算法减 token/蒸馏，硬件上再挖不动一个数量级。
2. **VAE**：36blk 时间片仍 ~82s，已是 e2e 地板；连续两轮优化收益 &lt;10%（87→82→81.6）。
3. **window=50 全权常驻**：与 block workspace 叠加 **OOM**，无法用「全驻留」再砍 cache thrash。

**最接近全量档：** 单步 50L DiT **15.3s** + VAE36 **81.6s**；多步只能靠减 steps/layers/vae_blocks（见 Pareto 2×18@19.5s）。

### 缺口

- VAE：时间片合并 / 更大空间窗 / 跳过非关键 blk（质量风险）或换更粗 decode
- DiT：蒸馏少步、局部 attn、或 FA3 级 kernel；CUDA Graph 稳定 replay 仍次要
- TE stub 与 DiT/VAE 流水重叠未测

### 10.18 优化开发计划（指向）

质量优先的下一轮工程任务（opt3 后：F1 Sage2/2++、F2 TE overlap、F5 MagCache…）见独立文档：

→ [`nvidia_optimization_plan.md`](nvidia_optimization_plan.md)（§9 任务表；§10–§14 监控/对照/完整度；**§15 完整框架定义与 G\* 补齐**；**§16 T2V 实测**；**§17 缺口/耗时/CUDA Audio/Phase 计划**；**§18 业界/官方最新 H3 优化补充（2026）**）

「质量换速」（window attn / 少步 / TeaCache）不进该文档的 P0 默认路径。

### 10.19 opt2 实测摘要（2026-08-15）

在 §10.18 P0 之上继续无损压墙钟（详见 optimization_plan）：

| 指标 | P0 | opt2 |
| --- | ---: | ---: |
| 1L quiet | ~270 ms | **~175 ms**（进 150–200 带） |
| 4L | ~731 ms | **~420 ms** |
| 50L×1 | 9.22 s | **~5.62 s** |
| VAE36 | 12.2 s | **~12.1 s**（收益尽） |
| 外推 20×50+VAE | ~3.3 min | **~2.1 min** |
| check_sdpa | PASS | **PASS** |

要点：probe≥2L 走 `layers_forward`；RoPE 直出 FP16 进 SDPA；unpack 侧流；aux 双缓冲；`all_finite` 采样。

## 6. 构建指针

Darwin / Metal 继续用根目录 `Makefile`。Linux / CUDA stub 见 [`nvidia_build.md`](nvidia_build.md)。

## 7. 端到端最小补齐（Comfy-Org，勿下 433GB 全集）

Kijai experimental **只有** DiT W4A8 + video VAE int8 + ref LoRA。要跑通本仓/Comfy 管线，通常还需从 [`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3) **按需**补：

| 优先级 | 文件 | 约大小 | 说明 |
| --- | --- | --- | --- |
| P0 | `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | ~14.6 GB | 4090 友好 TE；或改用 bf16(~48GB)/int8(~25GB) |
| P1 | `vae/minimax_h3_audio_vae_fp32.safetensors` | ~0.56 GB | 若走音视频联合；纯视频可暂缓 |
| 可选 | `vae/minimax_h3_video_vae_fp16.safetensors` | ~4.85 GB | 已有 Kijai int8 VAE 时可先不重复下 |
| 可选 | `loras/...`（Kijai） | ~2.4 GB | Ref2VA 工作流 |

**建议最小盘占用（4090）：** Kijai FL2VA W4A8(~11.7) + Kijai video VAE int8(~3.0) + Comfy TE nvfp4(~14.6) ≈ **29 GB**；加 audio VAE ≈ **30 GB**。不要拉 bf16 DiT 双份（各 ~62 GB）。

### 10.20 优化计划扩展指针（2026-08-15）

监控覆盖审计、代码细节可优化点、业界方案对照（已添加/未添加/不做）、单卡推荐顺序与框架完整度评估，见 [`nvidia_optimization_plan.md`](nvidia_optimization_plan.md) **§10–§14**（§9 任务表 F1–F9 仍为加速执行主线）。

**完整框架（产品 e2e）蓝图 / 清单 / 验收 / G\* 补齐任务：** 同文件 **§15**；**opt4 T2V 子集实测与说明见 §16**；**缺口/耗时/CUDA Audio/后续 Phase 见 §17**；**业界/官方最新 H3 优化补充见 §18**。硬区分：**算力旁路完整 ≠ 产品框架完整**。T2V / T2AV 子集已封口（§16/§17）；Sage2 默认仍 cudnn。

---

## opt4 旁路框架封口（2026-08-15）

T2V 子集（G1/G4/G5/G6/F1接线/F2）+ 4090 实测见 [`nvidia_optimization_plan.md`](nvidia_optimization_plan.md) **§16**。  
**G2 Audio / T2AV：✅（§17 Au1–Au7，2026-08-16）** — CUDA Audio VAE + DiT audio token + `h3_cuda_generate --audio-vae`；不破坏 Metal；未 git commit。

### 10.21 现状缺口 · 耗时 · CUDA Audio 计划（指向 §17）

缺失清单、实测耗时表（含 §16 **68.9s** T2V 与 opt3 **~2.06 min** 全量外推）、业界对比、**CUDA Audio（Au\*）**、分 Phase 后续计划，见：

→ [`nvidia_optimization_plan.md`](nvidia_optimization_plan.md) **§17**

### 10.22 业界/官方最新优化补充（指向 §18）

Comfy/Kijai、vLLM-Omni TeaCache/Cache-DiT、SGLang Diffusion、官方 AdaLN/稀疏注意力、Metal reuse/token-reduction 对照与本轮 F5-lite/F7a 落地，见：

→ [`nvidia_optimization_plan.md`](nvidia_optimization_plan.md) **§18**

硬约束不变：生产 SDPA 默认 **cudnn**（sage2 仅 opt-in）；质量换速旋钮默认关。