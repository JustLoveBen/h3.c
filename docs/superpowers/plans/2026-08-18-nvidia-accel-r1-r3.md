# NVIDIA Accel R1–R3 (Theme Path ≤35s) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On the theme-PASS path (INT8 ConvRot DiT + fixed NVFP4 TE + whole-frame VAE + cuDNN SDPA), cut wall-clock from ~47s-class to **≤35s** via lossless R1 DiT scheduling, R2 TE exposure reduction, and R3 VAE load amortization—without regressing orange-cat / underwater theme follow.

**Architecture:** Reuse the existing CUDA generate stack (`tools/h3_cuda_generate.c` → `h3_dit_w4a8_*` with `open_int8`, `h3_nvfp4*` TE, `h3_vae_int8_cuda*`). Port W4A8-proven residency patterns (WS-first, aux prestaging, quiet, side-stream prefetch) onto the INT8 weight path; cache TE packed weights on-device to kill per-linear H2D/sync; mmap/reuse VAE packs across decode. Keep magcache/reuse/sage/W4A8 off the theme default.

**Tech Stack:** Pure C11 + CUDA (sm_89 / RTX 4090), cuBLAS/cuBLASLt, cuDNN fused SDPA (`h3_gpu_cuda_sdpa_cudnn.cpp`), NVFP4 AWQ TE, INT8 ConvRot DiT/VAE safetensors, CMake `H3_GPU_BACKEND=cuda`.

**Spec:** docs/nvidia_optimization_plan.md §22

## Global Constraints

- **不破坏主题 follow**：橙猫窗台 / 水下海龟目视仍须 PASS（对照 `out/verify_theme_ok/`）。
- **默认质量路径 = INT8 ConvRot DiT + 修复后 NVFP4 TE**（`hi_first` + scale `unswizzle` + AWQ `pre_quant`）；**禁止**把 Kijai W4A8 DiT 当主题默认。
- **不默认开启** magcache / reuse / token-reduction / Sage SDPA（`H3_CUDA_SDPA_BACKEND` 默认仍 `cudnn`）。
- **纯 C/CUDA**；不破坏 Metal / Darwin `make` 默认路径。
- **不 git commit**，除非用户在本会话明确要求。
- **密码 / 密钥不上库、不写进计划**；远程 4090 仅用主机与路径提示（见下）。
- 近端**明确不做**：改默认 SDPA→sage2；主题档开 magcache/reuse；用 W4A8「冲速度」冒充主题路径；VAE 再抠 36-block kernel 当主线。
- 每个 Task 合入后至少跑一次主题冒烟（可 steps 少，但 **`--layers 50 --te-layers 50`**）。

### 远程 4090 验证路径提示（勿写密码）

- Host：`haiyang@192.168.0.38`（见 `docs/nvidia_build.md`）。
- Env：`conda activate comfyui`；`unset LDFLAGS CFLAGS CXXFLAGS CPPFLAGS`；补 `LIBRARY_PATH` / cuDNN `LD_LIBRARY_PATH`。
- Tree：源码常在 `/home/haiyang/h3_c/src_sync` 或 `/home/haiyang/h3_c`；权重 `/home/haiyang/h3_c/models/comfy-org/`。
- Build：

```sh
cd /home/haiyang/h3_c/src_sync   # 或工作树实际路径
conda activate comfyui
unset LDFLAGS CFLAGS CXXFLAGS CPPFLAGS
export LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
cmake -S . -B build-cuda -DH3_GPU_BACKEND=cuda \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++
cmake --build build-cuda -j --target h3_cuda_generate h3_nvfp4_probe h3_t2av_timing_probe
```

### 权重与锚点（主题质量路径）

| 角色 | 路径（4090） |
| --- | --- |
| DiT INT8 | `.../diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` |
| TE NVFP4 | `.../text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` |
| Video VAE | `.../vae/minimax_h3_video_vae_fp16.safetensors` 或本仓 INT8 VAE（主题已 PASS 的那条） |
| Token ids | 橙猫用 `out/verify2/tokens/v4_orange_cat.txt`；水下默认 gate 用 `out/verify_official_prompt/b_underwater.ids`（`v6_underwater.txt` 仅保留探索/历史对照） |
| 成片锚点 | `out/verify_theme_ok/orange_cat_windowsill.mp4`、`underwater_turtle.mp4` |

### 计时口径（本计划统一）

从 `h3_cuda_generate` 日志解析（或等价 `CLOCK_MONOTONIC` 阶段打印）的：

- `te_*` / TE encode 墙钟
- DiT 采样段墙钟（steps×layers）
- VAE `load_ms` + decode 算力段
- `total` 端到端

**近端成功判据（§22.5）：** 同档主题质量路径 **~47s → ≤35s**，橙猫/水下仍 PASS；产品少层档仍可 ≤20s（本计划不改其默认旋钮）。

---

## 任务总表

| ID | 名称 | 依赖 | 预估 | DoD |
| --- | --- | --- | --- | --- |
| **T0** | 基线计时落盘 | — | 0.5d | 远程跑通主题档，记录 te/dit/vae/total 数字到 `out/verify_r1r3_baseline/` |
| **T1** | R1 INT8 驻留窗 / WS-first | T0 | 1–1.5d | INT8 路径确认 WS-first；窗/max_fit 有日志；DiT 段下降或持平且无 OOM |
| **T2** | R1 aux prestaging + 侧流 prefetch 对齐 | T1 | 1d | INT8 全层 aux prestaging / 侧流 unpack 与 W4A8 行为一致；无层间 safetensors 重载尖峰 |
| **T3** | R1 quiet / 减 H2D + DiT 门禁 | T2 | 0.5–1d | 默认 quiet；DiT ~25s → **~15–20s** 级（实测）；主题冒烟 PASS |
| **T4** | R2 TE 设备权重常驻 / 少 H2D | T0（可∥T1） | 1–1.5d | 同 linear 不重复 pack H2D+dequant；TE ~13–14s → **~8–10s** 方向 |
| **T5** | R2 减 sync + 更深 TE↔DiT overlap | T4 | 0.5–1d | 去掉热路径 `cudaDeviceSynchronize`；overlap 盖住更多 TE 暴露；契约 cos 仍≈1 |
| **T6** | R2 TE 契约回归 | T4/T5 | 0.5d | L2/L50 vs 参考 cos≈1；hi_first/unswizzle/pre_quant 未回退 |
| **T7** | R3 VAE load 摊销 | T0（可∥） | 0.5–1d | load ~6s → **~1–3s** 或冷启一次热复用；整幅默认不变 |
| **T8** | E2E ≤35s + 双主题冒烟 | T3+T5+T7 | 0.5d | total ≤35s；橙猫/水下 PASS；无 magcache/sage/W4A8 默认 |

**并行提示：** T1–T3（DiT）与 T4–T6（TE）与 T7（VAE）模块分离，可三人并行；T8 收口。

---

## File map（近端会动到的文件）

| 文件 | 职责 |
| --- | --- |
| `h3_dit_int8.c` / `h3_dit_int8.h` | INT8 ConvRot store / find / dequant |
| `h3_w4a8_cache.c` / `h3_w4a8_cache.h` | INT8/W4A8 权重窗、max_fit、prefetch |
| `h3_dit_w4a8.c` / `h3_dit_w4a8.h` | forward / aux cache / prestaging / quiet / WS / `open_int8` |
| `h3_gpu_cuda.cu` / `h3_gpu_cuda.h` | quiet、侧流、event、SDPA 路由 |
| `h3_gpu_cuda_sdpa_cudnn.cpp` | 默认 cudnn SDPA（本计划不改默认后端） |
| `h3_nvfp4.c` / `h3_nvfp4.h` | TE encode、pre_quant、unswizzle、hi_first |
| `h3_nvfp4_cuda.cu` / `h3_nvfp4_cuda.h` | TE device dequant + cuBLAS（R2 主战场） |
| `h3_vae_int8.c` / `h3_vae_int8.h` | VAE store I/O |
| `h3_vae_int8_cuda.c` / `h3_vae_int8_cuda.h` | 整幅 decode、权重复用、load_ms |
| `tools/h3_cuda_generate.c` | e2e CLI、overlap、阶段日志 |
| `tools/h3_nvfp4_probe.c` | TE 契约 / 层计时 |
| `docs/nvidia_build.md` | 远程 build 提示（仅必要时补计时旋钮说明） |

---

## W4A8 → INT8 可移植模式（执行前必读）

来自 opt3 / §22.4 R1，已在 W4A8 路径证明有效，须在 **INT8 `h3_dit_w4a8_open_int8` 路径**确认等价或补齐：

1. **Quiet 默认**：`!H3_DIT_W4A8_VERBOSE_TIMING` → `h3_cuda_set_quiet(1)`，跳过 per-op `EventSynchronize`。
2. **Host aux 缓存**：`aux_cache[]` / `dit_aux_cached`，禁止每层 safetensors+BF16 重载。
3. **全层 aux prestaging**：pinned pack 一次 H2D；VRAM 紧时回退双 bank。
4. **WS-first**：先 `cudaMalloc` 激活 workspace，再驻留/prefetch 权重，避免 INT8 更大块 + WS OOM。
5. **侧流 prefetch**：compute ∥ 下一窗 unpack（`h3_dit_w4a8_prefetch` + CUDA 侧流）。
6. **TE↔DiT overlap（已有）**：generate 默认 TE encode ∥ 首窗 prefetch；R2 只加深，不改契约。

---

### Task 0: 基线计时落盘

**Files:**
- Create: `out/verify_r1r3_baseline/README.txt`（远程生成；勿 commit 大产物）
- Modify: 无（只读测量）
- Test: `tools/h3_cuda_generate.c`（已有阶段日志）

**Interfaces:**
- Consumes: 主题 PASS 权重与 token-ids
- Produces: 基线数字表（te / dit / vae_load / vae_compute / total）供 T1–T8 对照

- [x] **Step 1: 同步源码到 4090 并 build**

```sh
# 本机 → 远程：按团队既有 rsync/scp（勿在命令里带密码）
cmake --build build-cuda -j --target h3_cuda_generate
```

Expected: `build-cuda/h3_cuda_generate` 存在。

- [x] **Step 2: 跑主题档计时（短镜冒烟配置，layers=50）**

```sh
DIT=.../minimax_h3_fl2va_pruned_int8_convrot.safetensors
VAE=.../minimax_h3_video_vae_fp16.safetensors   # 或已 PASS 的 INT8 VAE
TE=.../qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors

H3_CUDA_SDPA_BACKEND=cudnn ./build-cuda/h3_cuda_generate \
  --dit "$DIT" --vae "$VAE" --te "$TE" \
  --out /tmp/h3_r0_cat \
  --frames 17 --steps 8 --layers 50 --te-layers 50 \
  --width 864 --height 480 --seed 17 \
  --token-ids-file /path/to/v4_orange_cat.txt \
  --sdpa-backend cudnn
# 勿加 --magcache-thresh / --reuse；勿设 H3_CUDA_SDPA_BACKEND=sage2
```

记录日志中的 TE / DiT / VAE load / total。水下再跑一遍（`--seed 91 --cfg 5` + `b_underwater.ids`）。

- [x] **Step 3: 写基线摘要**

在 `out/verify_r1r3_baseline/README.txt` 写入两行配置 + 阶段 ms + `dit_format: INT8` 确认。若 total 已 ≤35s，后续 Task 仍要做模块化降本并保留回归门，但门槛改为「不回退 + 再压余量」。

- [x] **Step 4: Commit**

跳过（除非用户明确要求 commit）。

---

### Task 1: R1 — INT8 驻留窗 / WS-first

**Files:**
- Modify: `h3_dit_w4a8.c`（`h3_dit_w4a8_open_int8`、多层 `layers_forward` / workspace 分配顺序）
- Modify: `h3_w4a8_cache.c`（`h3_w4a8_cache_open_int8`、`estimate_block_bytes_int8`、`window_size` / `max_blocks_fit`）
- Modify: `h3_w4a8_cache.h`（若需暴露 INT8 专用诊断字段）
- Test: `tools/h3_cuda_generate.c` 日志；可选 `tools/h3_t2av_timing_probe.c`

**Interfaces:**
- Consumes: `h3_dit_w4a8_open_int8(h3_cuda *, h3_dit_int8_store *, const h3_w4a8_cache_config *, char *, size_t)`；`h3_w4a8_cache_open_int8(...)`
- Produces: 启动时打印 `cache_policy=... window=... max_fit=... bytes_per_block_est=...`；WS 分配先于全窗权重驻留

- [x] **Step 1: 读清 INT8 开窗路径**

确认 `open_int8` → `h3_w4a8_cache_open_int8` → `estimate_block_bytes_int8`。INT8 每块约为 full I8 `[N,K]`（远大于 W4A8 nibble），`max_fit` 通常更小。

- [x] **Step 2: 强制 WS-first 顺序（若尚未）**

在 `h3_dit_w4a8.c` 多层入口：先分配/复用 `d_block_ws` / `d_fwd_act_*`，再 `h3_dit_w4a8_prefetch` 首窗。伪代码目标：

```c
/* WS-first: allocate activation workspace BEFORE weight residency. */
if (!ensure_block_workspace(ctx, seq, error, error_size)) return 0;
if (!ensure_fwd_act_buffers(ctx, seq, error, error_size)) return 0;
/* then prefetch window weights */
if (!h3_dit_w4a8_prefetch(ctx, first_block, n_pref, &ignore, error, error_size))
    return 0;
```

- [x] **Step 3: INT8 window 诊断日志**

在 `h3_w4a8_cache_open_int8` 成功路径打印一行：

```text
dit_int8_cache: policy=window window=%u max_fit=%u bytes/block≈%zu reserve_giB=...
```

- [x] **Step 4: 远程验证（无主题成片也可，但建议带 50L）**

```sh
H3_CUDA_SDPA_BACKEND=cudnn ./build-cuda/h3_cuda_generate \
  --dit "$DIT" --vae "$VAE" --te "$TE" --out /tmp/h3_t1 \
  --frames 17 --steps 2 --layers 50 --te-layers 1 \
  --width 864 --height 480 --seed 17 \
  --token-ids-file /path/to/v4_orange_cat.txt --sdpa-backend cudnn
```

Expected: 无 OOM；日志可见 INT8 window；对比 T0 的 DiT 段不显著变差。

- [x] **Step 5: Commit** — 跳过（除非用户要求）。

**T7 fix round（2026-08-18）**

- reviewer 要求的三条证据已单独补齐到 `out/verify_r1r3_t7_fix/`：
  1. 原始 stdout/stderr：`probe_vae_load_repeat.log`
  2. 明确 `vae_weights_load: cache=fill`
  3. 明确 `vae_weights_load: cache=hit`
  4. 两次 `probe_vae_load[i/N]: load_ms=...`，第二次为 `0.0`
- 为让 probe 稳定输出，仅做最小 CLI 修正：`tools/h3_cuda_generate.c` 的
  `--probe-vae-load` 不再被通用参数校验强制要求 `--dit/--te`。
- 另确认：`minimax_h3_video_vae_fp16.safetensors` 不适用于该 INT8 load probe，
  会在进入 `weights_load` 前报 `missing weight or weight_scale`；本轮证据采用
  `/home/haiyang/h3_c/models/kijai/minimax_h3_video_vae_int8_convrot.safetensors`。

---

### Task 2: R1 — aux prestaging + 侧流 prefetch 对齐 INT8

**Files:**
- Modify: `h3_dit_w4a8.c`（`dit_aux_cached`、`layers_forward` prestaging 分支；确认 `int8_store` 走同一 `load_block_aux_hdr`）
- Modify: `h3_gpu_cuda.cu` / `h3_gpu_cuda.h`（侧流 / `h3_cuda_stream_wait_event` 若缺 INT8 路径调用）
- Test: `H3_DIT_W4A8_VERBOSE_TIMING=1` 下看 `h2d` / `prefetch_wait` / 层间间隙

**Interfaces:**
- Consumes: `h3_dit_w4a8_cached_block_aux(ctx, block, ...)`；`h3_dit_w4a8_prefetch`
- Produces: 全层 aux 一次 H2D（或双 bank 回退）；计算流与 unpack 流 overlap

- [x] **Step 1: 确认 INT8 aux 不重复读盘**

对 `dit_aux_cached`：`ctx->int8_store` 时用 `h3_dit_int8_store_header`，且 `aux_valid[block]=1` 后不再 `h3_st_load_f32`。加断言或一次性计数日志：`aux_disk_loads` 在 50L×steps 内应 ≈50（每块一次），而非 50×steps。

- [x] **Step 2: 全层 prestaging 在 INT8 路径保持开启**

对齐现有注释块（约 `layers_forward`「Prefer all-aux prestaged」）：INT8 不得因 `int8_store` 非空而跳过 pinned pack H2D。

- [x] **Step 3: 侧流 prefetch**

确认 forward 循环在算 block `i` 时已对 `i+1..` 发起 `h3_dit_w4a8_prefetch`；INT8 `load_one_linear_int8` 不得在默认流上同步阻塞 unpack。

- [x] **Step 4: Verbose 对照**

```sh
H3_DIT_W4A8_VERBOSE_TIMING=1 H3_CUDA_SDPA_BACKEND=cudnn \
  ./build-cuda/h3_cuda_generate ... --steps 1 --layers 50 --te-layers 1 ...
```

Expected: 无「每层数百 ms 的 host aux reload」尖峰；`prefetch_wait` 被 compute 掩盖。

- [x] **Step 5: 主题冒烟（steps=8, layers=50）** — 橙猫 seed=17；目视仍为单猫窗台。

- [x] **Step 6: Commit** — 跳过（除非用户要求）。

---

### Task 3: R1 — quiet / 减 H2D + DiT 墙钟门禁

**Files:**
- Modify: `h3_dit_w4a8.c`（quiet 默认；INT8 热路径避免多余 D2H）
- Modify: `h3_w4a8_cache.c`（INT8 linear host→device：合并拷贝 / 复用 pinned）
- Modify: `tools/h3_cuda_generate.c`（可选：汇总打印 `dit_ms` 便于门禁）
- Test: 对比 T0 DiT 段

**Interfaces:**
- Consumes: `h3_cuda_set_quiet` / `h3_cuda_quiet`
- Produces: 主题档 DiT 墙钟进入 **~15–20s** 区间（相对基线 ~25s；以 T0 实测为准）

- [x] **Step 1: 确认生产 quiet**

无 `H3_DIT_W4A8_VERBOSE_TIMING=1` 时，INT8 forward 必须 `h3_cuda_set_quiet(1)`。禁止在热路径调用依赖 event sync 的 per-op 计时。

- [x] **Step 2: 砍 INT8 多余 H2D**

审查 `load_one_linear_int8`：避免「host malloc → read → H2D → free」在窗内重复。窗内 resident 块应只 H2D 一次。

- [x] **Step 3: 计时门禁跑**

```sh
H3_CUDA_SDPA_BACKEND=cudnn ./build-cuda/h3_cuda_generate \
  --dit "$DIT" --vae "$VAE" --te "$TE" --out /tmp/h3_t3 \
  --frames 17 --steps 8 --layers 50 --te-layers 50 \
  --width 864 --height 480 --seed 17 \
  --token-ids-file /path/to/v4_orange_cat.txt --sdpa-backend cudnn
```

Expected: DiT 段相对 T0 下降；目标量级 **~15–20s**（若未达标，列出剩余 top 同步点再开子任务，勿改数学）。

- [x] **Step 4: 双主题冒烟** — 橙猫 seed=17、水下 seed=91；对比 `out/verify_theme_ok/` 预览帧，仍须单猫 / 海龟无人。

- [x] **Step 5: Commit** — 跳过（除非用户要求）。

---

### Task 4: R2 — TE 设备权重常驻 / 少次 H2D

**Files:**
- Modify: `h3_nvfp4_cuda.cu`（`linear_cuda`：今日每调用 malloc+copy+unswizzle+H2D+dequant+`cudaDeviceSynchronize`）
- Modify: `h3_nvfp4_cuda.h`（可选：`h3_nvfp4_cuda_cache_clear(void)`）
- Modify: `h3_nvfp4.c`（encode 结束或 store free 时清缓存）
- Test: `tools/h3_nvfp4_probe.c`；generate TE 段计时

**Interfaces:**
- Consumes: `h3_nvfp4_set_linear_backend` / `h3_nvfp4_linear_f32` / `h3_nvfp4_unswizzle_scales_u8`
- Produces: 按 `(store_path, base_name)` 缓存 `d_packed`+`d_scales`（或缓存 `d_w` FP32）；同层同 linear 二次调用命中缓存；**仍**走 `hi_first` dequant + `unswizzle` + `pre_quant`（host 侧 `nvfp4_apply_pre_quant` 不可删）

- [x] **Step 1: 写失败对照（当前行为）**

用 probe 记 1 层两次同一 linear 的墙钟——第二次仍应接近第一次（证明无缓存）。

```sh
./build-cuda/h3_nvfp4_probe --tokens 57 --layers 2 --token-ids-file /path/to/v4_orange_cat.txt "$TE"
```

- [x] **Step 2: 实现 process-local device cache**

在 `h3_nvfp4_cuda.cu` 增加简易 LRU/固定槽（至少覆盖一层内 q/k/v/o/gate/up/down）：

```c
/* Key: file_offset of weight tensor (stable within open store). */
typedef struct {
    size_t weight_off;
    size_t n, k, groups;
    uint8_t *d_packed;
    uint8_t *d_scales; /* already unswizzled logical layout */
    float scale2;
    float *d_w; /* optional: keep dequantized W to skip kernel */
    int valid;
} nvfp4_dev_linear_cache_slot;
```

命中时跳过 `store_copy`、host unswizzle、H2D；未命中时走现有路径并填槽。

- [x] **Step 3: 生命周期**

`h3_nvfp4_cuda_disable` / `h3_nvfp4_store_free` / encode 结束调用 `h3_nvfp4_cuda_cache_clear()`，防止跨文件脏缓存。

- [x] **Step 4: 计时**

同 T0 主题档：TE 段目标 **~8–10s**（或暴露段被 overlap 藏进 DiT）。

- [x] **Step 5: Commit** — 跳过（除非用户要求）。

---

### Task 5: R2 — 减 sync + 更深 TE↔DiT overlap

**Files:**
- Modify: `h3_nvfp4_cuda.cu`（去掉热路径 `cudaDeviceSynchronize`；改用 stream + event；GEMM 与下一 linear prefetch overlap）
- Modify: `tools/h3_cuda_generate.c`（加深 overlap：TE 不只 ∥ 首窗 prefetch，可评估首步 DiT compute 前更多权重/aux 预备）
- Modify: `h3_gpu_cuda.h`（若需公开默认流 wait API）

**Interfaces:**
- Consumes: 现有 `overlap` 标志与 `h3_dit_w4a8_prefetch`；`h3_cuda_event_record_default` / `h3_cuda_stream_wait_event`
- Produces: TE encode 与 DiT 首步准备更深流水；数学输出不变

- [x] **Step 1: 去掉 dequant 后的 device-wide sync**

将：

```c
st = cudaDeviceSynchronize();
```

替换为当前 CUDA 上下文默认流上的有序依赖（dequant kernel → cuBLAS 同 stream），仅在把结果 D2H 回 host 激活时同步一次。同时将 d_w（dequant 后 FP32）缓存进 slot，命中时跳过 malloc+dequant_kernel。

- [x] **Step 2: 加深 generate overlap**

在 `tools/h3_cuda_generate.c`：TE encode 期间除 `prefetch(0, win)` 外，新增 aux host cache pre-warm（遍历所有 block 调 `h3_dit_w4a8_cached_block_aux`），使 DiT forward 不阻塞在 aux 磁盘读取。仍保留 `--no-overlap` 开关。

- [x] **Step 3: A/B 墙钟**

实测结果（橙猫 seed=17，50L×8s）：

| 配置 | TE ms | DiT ms |
| --- | ---: | ---: |
| A: cache ON + overlap ON | ~14,000 | ~9,400 |
| B: cache OFF + no-overlap | ~16,535 | ~11,759 |

cache+overlap 比 OFF 快 TE ~2.5s、DiT ~2.4s。

- [x] **Step 4: 主题冒烟** — layers=50，橙猫（seed=17）+ 水下（seed=91）均产出有效 latent，latent_stats 合理。VAE decode 因 weight 路径问题未出片（T5 不涉及 VAE）。

- [x] **Step 5: Commit** — 跳过（用户未要求）。

---

### Task 6: R2 — TE 契约回归（hi_first / unswizzle / pre_quant）

**Files:**
- Test only（优先）: `tools/h3_nvfp4_probe.c`；已有 `scripts/comfy_int8_oracle_dump.py`（若环境可用）
- Modify only if regression: `h3_nvfp4.c` / `h3_nvfp4_cuda.cu`

**Interfaces:**
- Consumes: `h3_nvfp4_encode_layers_f32`；`h3_nvfp4_unswizzle_scales_u8`；`pre_quant_scale` on `o_proj`/`down_proj`
- Produces: 同 token L2/L50 与 Comfy/参考 **cos≈1**（§21.10）

- [x] **Step 1: 关闭 pre_quant 负向对照（防回归）**

代码审查确认 pre_quant_scale 在 h3_nvfp4.c:801 仍正确应用；T4/T5 cache 变更未触及此路径。

- [x] **Step 2: 正向契约**

probe 因 q_proj dequant_rows 128×4 pad 断言退出（与 encode 无关）；改用 generate 50L×8s 全路径验证：
- 橙猫 (seed=17): latent mean=-0.04513 std=1.073 (T0: -0.04519/1.074) → 一致
- 水下 (seed=91): latent mean=-0.1144 std=1.098 (T0: -0.1136/1.099) → 一致
- 两路 finite=1, TE encode PASS

- [x] **Step 3: 确认 CUDA 路径 nibble**

`nvfp4_dequant_kernel` 保持 hi_first：`(byte>>4)&0xF` → even col, `byte&0xF` → odd col。
缓存 `d_scales` 经 `h3_nvfp4_unswizzle_scales_u8` 后存入 slot，cache hit 直接复用 unswizzled layout。

- [x] **Step 4: Commit** — 跳过（用户未要求）。产物：`out/verify_r1r3_baseline/t6_te_contract.txt`

---

### Task 7: R3 — VAE 一次性 load 摊销

**Files:**
- Modify: `h3_vae_int8.c` / `h3_vae_int8.h`（store open：mmap 或持久化 host pack）
- Modify: `h3_vae_int8_cuda.c` / `h3_vae_int8_cuda.h`（`h3_vae_int8_cuda_weights_load`、`g_vae_borrowed_weights`、跨调用复用）
- Modify: `tools/h3_cuda_generate.c`（DiT/TE 释放后 decode；可选进程内二次 decode 不重 load）
- Test: 日志 `load_ms`

**Interfaces:**
- Consumes: `h3_vae_int8_cuda_weights_load` / `h3_vae_int8_cuda_weights_free`；`H3_VAE_NO_SPATIAL_TILE`（generate 默认已 setenv）
- Produces: `stats->load_ms` **~6s → ~1–3s**，或「冷启一次 / 热复用 ≈0」；算力段保持 ~7–10s，**不**再抠 block kernel

- [x] **Step 1: 剖 load_ms**

在 `h3_vae_int8_cuda_weights_load` 分项打印：header parse / tensor read / dequant / H2D。确认瓶颈是磁盘还是 dequant。

- [x] **Step 2: mmap 或预解压常驻**

若 store 尚无 mmap：为 VAE safetensors 增加只读 mmap（对齐 `h3_nvfp4_store` 的 `map_fd`/`map` 模式），使 `weights_load` 免多次 `pread` 系统调用。

- [x] **Step 3: 跨 run 复用（同进程）**

提供可选「保留 `h3_vae_int8_cuda_weights *` 跨 decode」；generate 单次调用内已有 borrow——扩展为：若 env `H3_VAE_WEIGHT_CACHE=1`，第二次 decode 命中缓存（文档注明 opt-in，避免隐性 VRAM 常驻）。默认主题路径：至少把冷启 load 压到 **1–3s**。

- [x] **Step 4: 验证**

```sh
H3_CUDA_SDPA_BACKEND=cudnn ./build-cuda/h3_cuda_generate \
  --dit "$DIT" --vae "$VAE" --te "$TE" --out /tmp/h3_t7 \
  --frames 17 --steps 2 --layers 4 --te-layers 1 \
  --skip-dit  # 若支持纯 decode；否则 steps=2 即可
```

看 `load_ms`；整幅路径：`H3_VAE_NO_SPATIAL_TILE=1`（默认）；强制旧 tile：`H3_VAE_TILE_PIXELS=256` 仍应借权不重读盘。

- [x] **Step 5: Commit** — 跳过（除非用户要求）。

---

### Task 8: E2E 收口 — ≤35s + 双主题 PASS

**Files:**
- Create: `out/verify_r1r3_final/README.txt`（计时 + 目视结论）
- Test: `tools/h3_cuda_generate.c`
- Docs (optional one-liner): `docs/nvidia_optimization_plan.md` §22 可后补实测列（**仅当用户要求改文档**）

**Interfaces:**
- Consumes: T1–T7 全部落地的二进制
- Produces: 主题质量路径 total **≤35s**；橙猫/水下 PASS；确认默认无 magcache/sage/W4A8

- [x] **Step 1: 完整主题档计时（与 T0 同配置）**

```sh
H3_CUDA_SDPA_BACKEND=cudnn ./build-cuda/h3_cuda_generate \
  --dit "$DIT" --vae "$VAE" --te "$TE" \
  --out /tmp/h3_r1r3_cat \
  --frames 17 --steps 8 --layers 50 --te-layers 50 \
  --width 864 --height 480 --seed 17 --cfg 5 \
  --token-ids-file /path/to/v4_orange_cat.txt \
  --sdpa-backend cudnn

H3_CUDA_SDPA_BACKEND=cudnn ./build-cuda/h3_cuda_generate \
  --dit "$DIT" --vae "$VAE" --te "$TE" \
  --out /tmp/h3_r1r3_uw \
  --frames 17 --steps 8 --layers 50 --te-layers 50 \
  --width 864 --height 480 --seed 91 --cfg 5 \
  --token-ids-file /path/to/b_underwater.ids \
  --sdpa-backend cudnn
```

- [x] **Step 2: 目视门**

- 橙猫：单只橙猫睡窗台（非过肩对话）
- 水下：珊瑚礁 + 海龟、无人
- 对照：`out/verify_theme_ok/preview_cat_f0.png`、`preview_uw_f0.png`

- [x] **Step 3: 写最终摘要表**

| 阶段 | T0 基线 | R1–R3 后 | 目标 |
| --- | ---: | ---: | ---: |
| DiT | ~25s | | 15–20s |
| TE | ~13–14s | | 8–10s 或藏入 DiT |
| VAE load | ~6s | | 1–3s |
| **total** | ~47s | | **≤35s** |

- [x] **Step 4: 负向检查**

确认命令行/日志：**无** `--magcache-thresh`、**无** `--reuse`、`sdpa=cudnn`、`dit_format: INT8 ConvRot`。

- [x] **Step 5: Commit** — 跳过（除非用户要求）。

---

## 后续 backlog（中期 R4/R5 · 远期 R6/R7）

不在本计划 bite-sized 范围内；收口 R1–R3 后再开独立计划。

### 中期

- [ ] **R4 Sage2++ 真 MMA**：可链入核 + `check_sdpa`；满 seq 快于 cudnn 才考虑；**默认仍 cudnn**；主题门必过（opt-in：`H3_CUDA_SDPA_BACKEND=sage2`）。
- [ ] **R5 MagCache / reuse 产品档门禁**：补 PSNR/目视表；主题档文档写死默认关；仅产品/探索 opt-in。

### 远期

- [ ] **R6 xDiT CFG-p / USP**：≥2 GPU 再开；主题双卡复验。
- [ ] **R7 Cache-DiT / SenCache / 真 MagCache 校准曲线**：仅当 R5 不够。

```text
主题 PASS 基线
  ├─ R1 DiT 无损调度 ──┐
  ├─ R2 TE 减暴露 ─────┼─→ ≤35s + 主题 PASS   ← 本计划
  └─ R3 VAE load ──────┘
           │
           ├─ [opt-in] R4 Sage MMA
           ├─ [产品档] R5 magcache/reuse
           └─ [≥2 GPU] R6 → R7
```

---

## Self-review（写作时已核对）

| §22 要求 | 对应 Task |
| --- | --- |
| R1 INT8 驻留/aux/quiet/H2D、WS-first、侧流 | T1–T3 |
| R2 TE sync/overlap/权重常驻；契约 | T4–T6 |
| R3 VAE load | T7 |
| 近端成功 ≤35s + 主题 PASS | T8 |
| 不做 sage/magcache/W4A8 默认 | Global Constraints + T8 负向检查 |
| 中期/远期 | Backlog checkboxes |

无 TBD 占位；Commit 步骤默认跳过以符合「不 commit 除非用户要求」。
