# Linux / CUDA 构建（P1–P4 旁路）

Apple Silicon 上的完整推理路径不变：继续使用仓库根目录的 `Makefile`（Metal）。

NVIDIA 侧通过本 CMake 文件构建 **CUDA 旁路 / probe**；不链接、不替换 `h3_gpu.m`，不破坏 Darwin Metal / BF16 主路径。

## 依赖

- CMake ≥ 3.18
- CUDA Toolkit：**RTX 5090 必须 13+**（`H3_REQUIRE_CUDA13=ON`，产物目录建议 `build-cuda-sm120`）。4090 继续用 conda 12.x + `build-cuda`（sm_89；工具链 ≥12.8 时 fatbin 可另含 120）。
- 支持的主机编译器（Linux: gcc/clang）

本机若无 `nvcc` / CUDA，配置应失败并提示安装 Toolkit，而不是静默生成空目标。

Ubuntu 软件源常把 CUDA 11.x 的 `nvcc` 放在 PATH 上。CMake 在 PATH nvcc 过旧、且未指定 `CMAKE_CUDA_COMPILER` 时，会优先选 CUDA 13（conda env 或 `/usr/local/cuda-13*`），找不到 13 才回退 12.8。5090 请打开 `H3_REQUIRE_CUDA13`，禁止 12.x 凑合；NVRTC/cuDNN 随 13 链 `libnvrtc.so.13`，不要链到 CUDA 12。

```sh
# RTX 5090 / CUDA 13+
export CONDA_PREFIX=$HOME/miniconda3/envs/yongbin   # 若 nvcc 13 在该 env
export PATH=$CONDA_PREFIX/bin:$PATH
unset LDFLAGS CFLAGS CXXFLAGS CPPFLAGS
export LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib:$LIBRARY_PATH
cmake -S . -B build-cuda-sm120 -DH3_GPU_BACKEND=cuda -DH3_REQUIRE_CUDA13=ON \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++
which nvcc && nvcc --version   # 必须是 13.x

# RTX 4090 / CUDA 12.x（不要开 H3_REQUIRE_CUDA13）
cmake -S . -B build-cuda -DH3_GPU_BACKEND=cuda \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++
```

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

## 远程主机（SSH）

| 机 | 环境变量（优先） | 默认 `user@host` | 备注 |
| --- | --- | --- | --- |
| RTX 4090 | `H3_SSH_4090` | `haiyang@192.168.31.34` | 局域网；4090 旧址 `192.168.0.38` 已迁移，勿再用 |
| RTX 5090（单卡） | `H3_SSH_5090` | `yongbin@192.168.31.238` | **保留**。局域网常通 `yongbin@192.168.0.238`（用 env 覆盖即可） |
| **8× RTX 5090** | `H3_SSH_5090x8` | `user@89.185.31.140` | 端口 22；机架 **2FB-12-14 · U 16–22U**。后续多卡测；本轮未同步源码/权重 |

密码仅会话注入（`H3_SSH_PASS_4090` / `H3_SSH_PASS_5090` / `H3_SSH_PASS_5090x8` 或 `sshpass -e`），**永不写入 repo / docs / scripts / git**。只写变量名，不写值。

编排示例见 `scripts/h3_sdpa_backend_smoke_matrix.sh`（默认值即上表；可用 env 覆盖）。`--host 5090x8` 应对齐 `H3_SSH_5090x8`；脚本默认值不得含密码。

### 8×5090 远程机（探测 2026-09-11）

极速 SKU / Ulysses 实现计划：[2026-09-11-h3c-5090-turbo-implementation.md](superpowers/plans/2026-09-11-h3c-5090-turbo-implementation.md)。

Ubuntu **22.04.5**；主机名 `user`；登录用户 `user`（sudo）。**8× GeForce RTX 5090 属实**（各约 **32607 MiB / 32 GB**，TGP 575 W）。驱动 **595.45.04**（Open Kernel Module）；`nvidia-smi` 报 CUDA **13.2**。Toolkit：`/usr/local/cuda` → `cuda-13.2`，`nvcc` **13.2.51**（默认 PATH **无** nvcc，用 `/usr/local/cuda/bin`）。系统 `python3` 为 3.10.12；**Python 3.12.13** 在 `/usr/bin/python3.12`。CPU：2× Xeon Gold 6530（128 线程）· 主机内存 ~314 GiB。磁盘 `/` ext4 **879 G**（约 26 G 已用）。home：`/home/user`。**尚无 h3_c 工作区**（不要擅自大拷模型）。

**无 NVLink**（`nvidia-smi nvlink -s`：Device does not have or support Nvlink）。拓扑为 PCIe + 双 NUMA：GPU0–3 / NIC `mlx5_*` 在 NUMA 0，GPU4–7 在 NUMA 1；同 NUMA 为 `NODE`，跨 NUMA 为 `SYS`。多卡请按无 NVLink（Ulysses / 跨 NUMA 通信税）设计，勿假设 NV#。

SSH 示例（密码只进当前 shell，勿写入文件）：

```sh
export H3_SSH_5090x8=user@89.185.31.140
# export H3_SSH_PASS_5090x8=...   # 仅会话；sshpass -e
# 若本机 TUN/Clash 出现「TCP 通但无 SSH banner」，对该 IP 走直连，或 ssh -b <物理网卡 IPv4>
ssh -p 22 "$H3_SSH_5090x8"
```

## 远程 4090 验收命令（示例）

主机：`${H3_SSH_4090:-haiyang@192.168.31.34}`，环境：`conda activate comfyui`，权重在 `/home/haiyang/h3_c/models/`。源码常在 `src_sync/`。

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

## SDPA 三后端切换（4090 / 5090）

统一 registry（`h3_sdpa.h`）：**Tier-0** `cudnn`（质量默认，`auto`→cudnn）；**Tier-1** `sage3` / `sol_cute`（一等加速，显式开启）。假 `sol` nested-loop / 假 `sage2` 禁止当产品后端。详见 [2026-08-25-h3c-multi-gpu-arch-speed.md](superpowers/plans/2026-08-25-h3c-multi-gpu-arch-speed.md)。

```sh
# Tier-0 — 双卡质量默认
./build-cuda*/h3_cuda_generate ... --sdpa-backend cudnn

# Tier-1 Sage3 — 5090 / build-cuda-sm120（需插件 + ROOT）
export H3_SAGE3_LIB=$PWD/build-cuda-sm120/libh3_sage3.so
export H3_SAGE3_ROOT=/path/to/sageattention3_blackwell
export H3_SAGE3_TOOLS=$PWD/tools
export PYTHONPATH=$H3_SAGE3_TOOLS:$H3_SAGE3_ROOT/build/lib.linux-x86_64-cpython-310:$H3_SAGE3_ROOT
./build-cuda-sm120/h3_cuda_generate ... --sdpa-backend sage3
# 日志应见：layout=F16_BHTD prep=cuda_thd_bhtd；hits 满、fallback=0

# Tier-1 Sol CuTe — 4090 / build-cuda（需插件 + 官方 ATTN ROOT + Triton≥3.3）
export H3_SOL_CUTE_LIB=$PWD/build-cuda/libh3_sol_cute.so
export H3_SOL_ATTN_ROOT=/path/to/techniques/sparse_backends   # 官方：无 _prepare_torch
export H3_TRITON_OVERLAY=/path/to/triton34_overlay            # TensorDescriptor
export PYTHONPATH=$H3_TRITON_OVERLAY:$H3_SOL_ATTN_ROOT:$PWD/tools
# 测满核命中时可：
export H3_SOL_FIRST_DENSE_STEPS=0 H3_SOL_FIRST_DENSE_LAYERS=0
./build-cuda/h3_cuda_generate ... --sdpa-backend sol_cute
# 日志应见：backend=cute_sm89 preprocess=triton_prepare；hits>0 fallback=0

# 能力探测
./build-cuda*/h3_cuda_info
# 打印 caps: sm / fp4 / sage3_lib / sol_cute_lib 与 sdpa_select reason
```

### 缺什么会硬错

| 后端 | 缺项 | 行为 |
| --- | --- | --- |
| `sage3` | `H3_SAGE3_LIB` / 插件 | generate resolve / dlopen **硬错** |
| `sage3` | `H3_SAGE3_ROOT` | init **硬错** |
| `sol_cute` | `H3_SOL_CUTE_LIB` | **硬错** |
| `sol_cute` | `H3_SOL_ATTN_ROOT` | **硬错**（无机器默认；warmup 亦不回退 sparse） |
| `sol_cute` | ROOT 含 `_prepare_torch` / Triton&lt;3.3 | embed **硬错**（拒绝假 preprocess） |
| `sol_cute` | backend≠`cute_sm89` | embed **硬错** |

不会静默落到 nested-loop / sage2。

### 已测数字（诚实，2026-08-25）

| 卡 | 后端 | 配方摘录 | 结果 |
| --- | --- | --- | --- |
| 5090 | cudnn | cafe 8-step warm | DiT **~31.3s**；TE 上传稳档后 e2e warm mean **~66s** |
| 5090 | sage3 | 同配方；CUDA prep | DiT **~26.4–27.4s**；hits=400 fallback=0；MAE~27 |
| 4090 | sol_cute | cafe 5s×8-step 有声，dense=0，官方 ROOT | **hits=400**；DiT **40.9 vs cudnn 48.3（−15%）**；e2e≈79s≈cudnn；**MAE~35 视觉不过** |

报告路径：

- Sol 视觉门：[`out/verify_sol_vis_gate/REPORT.md`](../out/verify_sol_vis_gate/REPORT.md)（远程 `/tmp/h3_sol_vis_gate/`）
- 5090 TE 稳档：[`out/verify_5090_te_stable/REPORT.md`](../out/verify_5090_te_stable/REPORT.md)（远程 `…/out_5090_te_stable/`）
- 5090 后端 bench：[`out/verify_5090_bench/REPORT.md`](../out/verify_5090_bench/REPORT.md)

**Sol ROOT：** 用官方 NVlabs `sol-engine` 的 `techniques/sparse_backends`（preprocess 无条件 `TensorDescriptor`）。带 `try/except` + `_prepare_torch` 的 sparse 拷贝**不能**当 `H3_SOL_ATTN_ROOT`。

**Sage3 prep：** 插件暴露 BHTD；C 做 `THD→BHTD` / `BHTD→THD F32` CUDA kernel，不再走 Python permute。质量门不因加速改默认。

**TE 稳档（5090）：** 偶发 `upload_ms~50s` 来自 12.8GiB `mmap_direct` 页错误。默认 `MAP_POPULATE` + sync prefault（`H3_TE_MMAP_POPULATE` / `H3_TE_PREFAULT`，`=0` 关）。稳档后 `upload_ms≈6.2s`（4 连续跑无尖峰）；warm e2e mean **~66s**。

## 双卡 SDPA 后端冒烟矩阵

可重复轻量检查（真 backend；默认 `--seconds 1 --steps 2`；**不开 MagCache**）。脚本在 GPU 机上跑；编排器可选从笔记本 SSH（密码只走环境变量，勿入库）。

```sh
# 在 5090 / 4090 仓库根：
bash scripts/h3_sdpa_backend_smoke.sh --host 5090 --backend cudnn
bash scripts/h3_sdpa_backend_smoke.sh --host 5090 --backend sage3   # 需 H3_SAGE3_LIB+ROOT
bash scripts/h3_sdpa_backend_smoke.sh --host 5090x8 --backend cudnn  # 8×5090；H3_SSH_5090x8；后续测，本轮未同步仓库
bash scripts/h3_sdpa_backend_smoke.sh --host 4090 --backend cudnn
bash scripts/h3_sdpa_backend_smoke.sh --host 4090 --backend sol_cute # 官方 SOL ATTN ROOT

# 可选编排（主机：H3_SSH_5090 / H3_SSH_4090 / H3_SSH_5090x8；
# 密码：H3_SSH_PASS_5090 / H3_SSH_PASS_4090 / H3_SSH_PASS_5090x8，由操作者注入，勿入库）
# 8×5090 默认不 rsync；需 H3_SMOKE_ENABLE_5090x8=1。TUN 拦 SSH 时可设 H3_SSH_BIND。
bash scripts/h3_sdpa_backend_smoke_matrix.sh
```


| 卡 | 后端 | 门禁 |
| --- | --- | --- |
| 5090 | cudnn | `sdpa_select=cudnn` + 出片 |
| 5090 | sage3 | hits>0、fallback=0 |
| 4090 | cudnn | 同上 + 出片 |
| 4090 | sol_cute | cute_sm89 + hits>0、fallback=0 |

失败非零退出。本地汇总可放 `out/verify_sdpa_smoke/`。

### 加速矩阵（cafe 金线臂）

统一入口（真 `--seconds`；reuse/mag/tea/fb/sage/sol **不**改默认）：

```sh
bash scripts/h3_accel_matrix_smoke.sh --gpu 4090 --arm cudnn
bash scripts/h3_accel_matrix_smoke.sh --gpu 4090 --arm sol_cute
bash scripts/h3_accel_matrix_smoke.sh --gpu 4090 --arm magcache --thresh 0.08
bash scripts/h3_accel_matrix_smoke.sh --gpu 4090 --arm teacache --thresh 0.04
bash scripts/h3_accel_matrix_smoke.sh --gpu 4090 --arm fbcache --thresh 0.04
bash scripts/h3_accel_matrix_smoke.sh --gpu 5090 --arm cudnn
bash scripts/h3_accel_matrix_smoke.sh --gpu 5090 --arm sage3
```

总表：[`out/verify_accel_matrix/REPORT.md`](../out/verify_accel_matrix/REPORT.md)。  
步/块 cache 独立模块：`h3_step_cache.h` / `.c`（`--teacache-*` / `--fbcache-*`；`--easycache-thresh` → **NOT IMPLEMENTED** 硬错误）。与 Mag/reuse 互斥。

### Vision 条件图 SDPA（F2VA/L2VA）

默认：绑 `h3_cuda` 后 vision **cuBLAS GEMM + Tier-0 cuDNN SDPA**（强制 cudnn，不受 DiT sage3/sol 影响）。  
`H3_VISION_HOST=1` → 全 host；`H3_VISION_SDPA=host` → 仅 SDPA 回退 host（便于 A/B）。  
实测见 [`out/verify_vision_cudnn_sdpa/REPORT.md`](../out/verify_vision_cudnn_sdpa/REPORT.md)。

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
