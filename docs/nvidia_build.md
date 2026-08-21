# Linux / CUDA 构建（P1–P4 旁路）

Apple Silicon 上的完整推理路径不变：继续使用仓库根目录的 `Makefile`（Metal）。

NVIDIA 侧通过本 CMake 文件构建 **CUDA 旁路 / probe**；不链接、不替换 `h3_gpu.m`，不破坏 Darwin Metal / BF16 主路径。

## 依赖

- CMake ≥ 3.18
- CUDA Toolkit（建议 12.8+，以便 `sm_120`；更旧工具链会自动省略 120，仅保留 `sm_89`）
- 支持的主机编译器（Linux: gcc/clang）

本机若无 `nvcc` / CUDA，配置应失败并提示安装 Toolkit，而不是静默生成空目标。

## 配置与构建

```sh
cmake -S . -B build-cuda -DH3_GPU_BACKEND=cuda \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++
cmake --build build-cuda -j
```

若在 conda（如 `comfyui`）里用 nvcc 12.x，链接 `cublas` 时可能撞到 conda sysroot 的 glibc 符号问题；**务必**用系统 host 编译器（如上）。

可选指定架构（默认在工具链允许时包含 89 与 120）：

```sh
cmake -S . -B build-cuda -DH3_GPU_BACKEND=cuda \
  -DCMAKE_CUDA_ARCHITECTURES="89;120"
```

## 一键目标（常用）

```sh
# 设备探测
cmake --build build-cuda -j --target h3_cuda_info

# W4A8 基础栈
cmake --build build-cuda -j --target \
  h3_st_inventory h3_w4a8_probe h3_w4a8_matmul_probe \
  h3_w4a8_cuda_probe h3_w4a8_blocks_probe \
  h3_w4a8_int8_cuda_probe h3_w4a8_device_probe h3_w4a8_cache_probe

# DiT W4A8 旁路（block / 多 block / forward / sample）
cmake --build build-cuda -j --target \
  h3_dit_w4a8_block_probe h3_dit_w4a8_layers_probe \
  h3_dit_w4a8_forward_probe h3_dit_w4a8_sample_probe

# 外围最小探测
cmake --build build-cuda -j --target h3_vae_int8_probe h3_nvfp4_probe

# 正式 T2V CLI（CUDA 旁路；非 Metal）
cmake --build build-cuda -j --target h3_cuda_generate h3_t2av_timing_probe
```

## 产物一览

| 目标 | 说明 |
| --- | --- |
| `h3_cuda_info` | GPU 名 / VRAM / sm |
| `h3_st_inventory` / `h3_w4a8_probe` | safetensors / W4A8 header |
| `h3_w4a8_matmul_probe` | CPU dequant + FP32 GEMM |
| `h3_w4a8_cuda_probe` | act ConvRot + cuBLAS FP32 |
| `h3_w4a8_blocks_probe` | 多 block CPU INT8 参考 |
| `h3_w4a8_int8_cuda_probe` | cuBLASLt INT8 GEMM |
| `h3_w4a8_device_probe` | device unpack 常驻 |
| `h3_w4a8_cache_probe` | 50-block cache 策略 |
| `h3_dit_w4a8_block_probe` | 单 block 近完整前向 |
| `h3_dit_w4a8_layers_probe` | 多 block 串联（`--layers`） |
| `h3_dit_w4a8_forward_probe` | patch → layers → final |
| `h3_dit_w4a8_sample_probe` | Euler 少步 latent |
| `h3_vae_int8_probe` | Video VAE I8×scale dequant |
| `h3_nvfp4_probe` | TE NVFP4 dequant + **可配置层 encode（G1）**；默认 **device dequant+cuBLAS**（`H3_NVFP4_HOST_GEMM=1` 回退） |
| `h3_t2av_timing_probe` | 5s480P e2e / DiT / VAE 墙钟 + check_sdpa |
| **`h3_cuda_generate`** | **正式 CUDA 旁路 T2V CLI（G4）**：出 PPM/latent |

静态库：`h3_gpu_cuda`（`h3_cuda_*` / resident INT8 GEMM / device unpack）。

## 远程 4090 验收命令（示例）

主机：`haiyang@192.168.0.38`，环境：`conda activate comfyui`，权重在 `/home/haiyang/h3_c/models/`。源码常在 `src_sync/`。

```sh
cd /home/haiyang/h3_c/src_sync   # 或 /home/haiyang/h3_c
conda activate comfyui
unset LDFLAGS CFLAGS CXXFLAGS CPPFLAGS
export LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib:$LIBRARY_PATH
cmake -S . -B build-cuda -DH3_GPU_BACKEND=cuda \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++
cmake --build build-cuda -j --target h3_t2av_timing_probe
```

## §10.15–§10.17 旋钮（4090）

```sh
unset LDFLAGS CFLAGS CXXFLAGS CPPFLAGS
export LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH

# SDPA cudnn
H3_CUDA_SDPA_BACKEND=cudnn ./build-cuda/h3_t2av_timing_probe --dit DIT \
  --sdpa-backend cudnn --dit-layers 1 --skip-te --skip-vae

# 分项计时（默认 quiet）
H3_DIT_W4A8_VERBOSE_TIMING=1 H3_CUDA_SDPA_BACKEND=cudnn ./build-cuda/h3_t2av_timing_probe \
  --dit DIT --dit-layers 1 --skip-te --skip-vae --sdpa-backend cudnn

# Done-A Pareto：满 seq 2×18 + VAE4 ≤20s
H3_CUDA_SDPA_BACKEND=cudnn ./build-cuda/h3_t2av_timing_probe --dit DIT --vae VAE \
  --width 864 --height 480 --seconds 5 \
  --e2e --steps 2 --dit-layers 18 --reuse 1 --vae-blocks 4 --skip-te \
  --window 18 --sdpa-backend cudnn

# 50L 单步（外推 20×50）
H3_CUDA_SDPA_BACKEND=cudnn ./build-cuda/h3_t2av_timing_probe --dit DIT \
  --width 864 --height 480 --seconds 5 --dit-layers 50 --skip-te --skip-vae \
  --window 10 --sdpa-backend cudnn

# VAE 36（建议 cudnn：canvas 常驻下连续 quiet launch，SDPA 才能吃满）
H3_CUDA_SDPA_BACKEND=cudnn ./build-cuda/h3_t2av_timing_probe --dit DIT --vae VAE \
  --skip-dit --skip-te --vae-blocks 36

# cuDNN frontend JIT 需要 libnvrtc.so.12（h3_cuda_create 也会从 CONDA_PREFIX 预加载）。
# 上面默认 LD_LIBRARY_PATH 已含 nvidia/cuda_nvrtc/lib。
```

cuDNN：`-DH3_ENABLE_CUDNN=ON` + vendored `third_party/cudnn_frontend`。

实测数字见 [`nvidia_quant_contract.md`](nvidia_quant_contract.md) §10.15–§10.17。

## 已知限制

- 旁路模块**不**替换 Metal `h3_dit` / BF16 `load_block` / `h3_generate`
- DiT：cuDNN fused SDPA + quiet 多层常驻；无旋钮 20×50 仍约数分钟级
- **Sage2**（`H3_CUDA_SDPA_BACKEND=sage2`）：数值门禁可过；当前 INT8-QK（+dp4a）满 seq **远慢于** cudnn，**勿改默认**
- CUDA Graph：已接线；生产默认靠 quiet 去 sync（F3 审计：bank sync 保留）
- VAE：canvas 路径；粗分桶见 decode_stats
- `window=50` 全权常驻易与大 workspace OOM
- TE NVFP4：**可配置层** + **CUDA dequant/GEMM 默认**；**token-ids/canary 产品路径 ✅**；HF `tokenizer.json` BPE / visual / 默认全 50L **⛔**
- 满档 T2AV（5s480P 2×18+VAE4+TE1+audio）：**≈8.7–9.7 s**（TE ~0.17 s）；`te-layers=4` ≈9.0 s；TR opt-in ≈**7.7 s**
- **G2 Audio / T2AV：✅** — decode+encode+DiT AV+满档 CLI+**Au8 mux**；Metal 未改
- G3：官方 `--first-frame` F2VA / `--last-frame` L2VA / 两者同时 FL2VA = VAE encode → COND → AdaLN 0.999 + vision TE。F2VA 像素 STRETCH、keyframes=`[0]`、condition_time=`text_seq`。L2VA 像素 COVER、keyframes=`[frame_count-1]`、condition_time=`text_seq+video_span_sum-5/3`（公式冻结，FL2VA 尾段沿用）。FL2VA pack `[text | cond0 | cond1 | audio | video]`，`cond_seq = 2 * frame_rows`，首段 `condition_time=text_seq`，尾段用 L2VA 公式；日志 `keyframes=0,last`。**`--ref-image` Ref2VA（阶段 H1）：** 切 **另一棵** DiT（`--dit` 路径须含 `ref2va`；FL2VA 文件名拒绝，禁止静默回退）。Canvas 为 Metal `h3_reference_image_canvas`（只缩小、保比例）。Pack `[text | H3_SEG_REF_IMAGE | audio | video]`；RoPE cursor = `text_seq` 起每次 +1（不是 keyframe 0/last）；AdaLN 仍 visual_condition。与 `--first-frame`/`--last-frame` 互斥。默认 cfg=1。`--ref-audio`/`--ref-video` 仍为 H2/H3 占位拒绝。**CFG：** T2VA 主题默认 cfg=5；官方 F2VA/L2VA/FL2VA/Ref2VA 默认 cfg=1。CUDA INT8 默认 **不** 把 Picture+vision span 打成 VIDEO@σ（`--dit-text-tags` 才恢复 Metal-like tagging，非 bit-match）。`--init-latent` / `--lock-first-frame` 为 **debug only**（non-official G3-lite），不得冒充默认。质量 decode 可走 fp16 VAE；INT8 VAE 仍是速度档。`--skip-dit --first-frame/--last-frame` 测 VAE encode→decode 天花板（非 lite；末帧门用 still 的 **f0 vs COVER**，不要用时域发糊的 still-plant 末帧）。
- G7：`h3_quality_gate_probe` 支持 `--ppm-a/--ppm-b --min-psnr`
- Audio 1D：`H3_AUDIO_CONV_BACKEND=auto|naive|gemm`（Au7 im2col+cuBLAS）
- 本文件不存放远程密码或密钥

## 明确不做

- 不实现完整 `h3_gpu.h` Metal 等价 API
- 不修改 Darwin `make` / `make test` 默认路径

## T2V / T2AV e2e（4090）

- **T2V**：见 [`nvidia_optimization_plan.md`](nvidia_optimization_plan.md) **§16**（≈68.9s）。
- **T2AV 满档**：同 §16 视频旋钮 + audio（≈8.7–9.7s）。详见 §17/§18.9。

```sh
cmake --build build-cuda -j --target \
  h3_audio_ops_probe h3_audio_vae_cuda_probe \
  h3_audio_vae_cuda_encode_probe h3_quality_gate_probe \
  h3_dit_w4a8_audio_probe h3_cuda_generate

AUDIO=.../minimax_h3_audio_vae_fp32.safetensors
./build-cuda/h3_audio_ops_probe
./build-cuda/h3_audio_vae_cuda_probe --weights "$AUDIO" --latent-t 2
H3_AUDIO_CONV_BACKEND=naive ./build-cuda/h3_audio_vae_cuda_probe --weights "$AUDIO" --latent-t 16
H3_AUDIO_CONV_BACKEND=gemm  ./build-cuda/h3_audio_vae_cuda_probe --weights "$AUDIO" --latent-t 16
./build-cuda/h3_audio_vae_cuda_encode_probe --weights "$AUDIO" --samples 1600
./build-cuda/h3_quality_gate_probe --audio-vae "$AUDIO"

# 满档 T2AV + token canary + Au8 mux
H3_CUDA_SDPA_BACKEND=cudnn H3_AUDIO_CONV_BACKEND=gemm ./build-cuda/h3_cuda_generate \
  --dit "$DIT" --vae "$VAE" --te "$TE" --audio-vae "$AUDIO" --out /tmp/h3_t2av_full \
  --layers 18 --te-layers 1 --vae-blocks 4 --steps 2 --context 16 \
  --width 864 --height 480 --seconds 5 --audio-t 200 --window 18 --sdpa-backend cudnn \
  --token-canary fox --mux /tmp/h3_t2av_full/out.mp4
```
