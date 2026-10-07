# h3.c 多 GPU 架构与加速后端（2026-08-25，Tier 修订）

> **后续开发方案见** [2026-08-26-h3c-next-tech-plan.md](./2026-08-26-h3c-next-tech-plan.md)  
> **2026-09-11 五方案普查（多卡 5090，不改金线）：** [2026-09-11-h3c-accel-survey-multigpu-5090.md](./2026-09-11-h3c-accel-survey-multigpu-5090.md)

日期：2026-08-25（修订：Sage3 / Sol-Attn **升为一等公民加速后端**，不再写成「诊断 / 可有可无」）  
仓库：`h3.c`（纯 C/CUDA MiniMax-H3）  
本文写架构与路线；密码不入库。不 commit。

对照（冲突以实测报告 + 本文分层为准）：

- [2026-08-25-h3c-4090-accel-plan.md](./2026-08-25-h3c-4090-accel-plan.md) — 4090 金线 ~75s；**Sol CuTe = Tier-1 重点（sm_89）**
- [out/verify_5090_bench/REPORT.md](../../../out/verify_5090_bench/REPORT.md) — 5090 cafe：cudnn ~66s；sage3 DiT −14%~−22%、MAE~27
- [docs/nvidia_build.md](../../nvidia_build.md) — `build-cuda` / `build-cuda-sm120`、三后端切换

**一句话：** 质量默认永远是 **cuDNN**；但 **Sage3** 与 **Sol-Attn CuTe（`sol_cute`）** 是产品必须一等支持的加速档——统一 registry、可探测、可测、可回退、CLI 同一套名字。已测诚实：Sage3 更快但 MAE~27；**Sol cafe 8-step 真核满 hits、DiT −15%、e2e≈cudnn、MAE~35 视觉不过**——仍要投资质量。假 nested-loop Sol / 假 Sage2 参考核 **禁止**当产品后端。MagCache/SLA 不是本阶段重点。

对照报告：

- [out/verify_sol_vis_gate/REPORT.md](../../../out/verify_sol_vis_gate/REPORT.md) — 4090 Sol 视觉门
- [out/verify_5090_te_stable/REPORT.md](../../../out/verify_5090_te_stable/REPORT.md) — 5090 TE 上传稳档
- [out/verify_5090_bench/REPORT.md](../../../out/verify_5090_bench/REPORT.md) — 5090 cudnn/sage3 bench

---

## 1. 目标分层（新口径）

### 1.1 Tier-0 — 质量默认（视觉无损）

| 项 | 约定 |
| --- | --- |
| 配方 | 8-step INT8 ConvRot + fp16 VAE tile+deblock 0.28 + 有声 pack + cafe 真 5s |
| SDPA | **`--sdpa-backend cudnn`**；`auto` / 未指定 → **cudnn**（永不自动选损质后端） |
| 视觉门 | 与 cafe 同 prompt/seed 不可辨；允许 checksum 不同 |
| 设备无关 | 4090 / 5090 / future **同一质量语义** |

实测地板（不要编）：

| 设备 | e2e（cafe warm） | DiT | 备注 |
| --- | ---: | ---: | --- |
| 4090 | **~75–79s** | **47.5–48.3s** | q1 金线 ~75s；gate cudnn warm 78.6s |
| 5090 | **~66s**（TE 上传稳档 mean） | **~31.3s** | `verify_5090_te_stable`；upload≈6.2s |

相对 4090：**e2e ≈1.14×，DiT ≈1.55×**。不要承诺同路径 2× e2e。

### 1.2 Tier-1 — 重点加速（必须一等支持）

| 后端 | CLI 名 | 主战场 | 已测（诚实） | 默认？ | 产品要求 |
| --- | --- | --- | --- | --- | --- |
| **Sage3** | `sage3` | **5090 / sm_120+ FP4** | DiT −14%~−22%；帧 MAE **~27**；prep 已 CUDA 化 | **否**（质量） | 插件可切换、可测、可回退；质量门勿因快改默认 |
| **Sol-Attn CuTe** | `sol_cute`（`sol` 别名勿指 nested） | **4090 / sm_89** | cafe 8-step：**hits=400**；DiT **−15%**；e2e≈cudnn；**MAE~35 不过门** | **否** | 官方 `H3_SOL_ATTN_ROOT`；禁止 sparse/`_prepare_torch` 冒充 |

原则：

1. **Tier-1 ≠「诊断玩具」**。架构、构建、文档、CI 矩阵、help 文案都以这两条为优先投资。
2. **Tier-1 ≠ 质量默认**。默认仍 cudnn；显式 `--sdpa-backend sage3|sol_cute` 才开。
3. 缺库 / 缺 ROOT → **清晰错误**，禁止静默落到 nested-loop / sage2 假核。
4. 两边 **同一套 registry / CLI 名字**；设备矩阵只影响「优先打磨哪条」，不分裂 API。

### 1.3 禁止当产品后端

| 名 | 实质 | 处置 |
| --- | --- | --- |
| `sol`（nested-loop） | 本仓假核，远慢于 cuDNN | **DROP-FAKE**；help 标 forbidden |
| `sage2`（in-repo） | 参考 INT8 循环，不是 thu-ml Sage3 | **禁止冒充**；仅参考 |

### 1.4 设备矩阵（能力档，不写死主机名）

| 档位 | 典型卡 | CUDA | Arch | Tier-0 | Tier-1 优先打磨 |
| --- | --- | --- | --- | --- | --- |
| **ada** | RTX 4090（sm_89） | 12.x | 89 | cudnn | **sol_cute** |
| **blackwell** | RTX 5090（sm_120） | **13+** | 120 | cudnn | **sage3** |
| **future** | 下一代 | 按 caps | `H3_CUDA_ARCHS` | cudnn | 新插件注册，主循环不动 |

硬规则：

1. **业务 generate / DiT 禁止** `if (sm == 120)` 散落选后端。
2. 只问 `h3_cuda_device_caps` + `h3_sdpa_resolve`。
3. 探测到 FP4 / sage3.so **绝不**让 `auto` 选 sage3。

---

## 2. 5090 上如何更快（ROI，诚实）

基线：cafe 8-step cudnn warm ≈ **te 12–14 / dit 30.7 / vae ~10.5 / total ~66s**。

| 刀 | 档 | 预期 | 说明 |
| --- | --- | --- | --- |
| TE mmap / 减抖动 | Tier-0 **✅** | upload≈6.2s 稳；e2e warm mean ~66s | `MAP_POPULATE`+prefault；见 `verify_5090_te_stable` |
| Vision host→cudnn | Tier-0 KEEP | 条件图 −20~30s 级 | T2VA 不受益 |
| **Sage3** | **Tier-1 重点** | DiT −4.3s（8-step）；20-step −21.5% | MAE~27；商用可切换 |
| Sage3 prep CUDA 化 | **Tier-1 ✅** | Python permute 已移除；BHTD 常驻 + CUDA THD↔BHTD | warm DiT ~27s 级（相对 cudnn ~31s） |
| sol_cute on 5090 | Tier-1 可跑 | 次优先于 sage3 | 4090 视觉门见 `verify_sol_vis_gate` |
| NVFP4 DiT / 4-step | 另开产品档 | 换权 / 换配方 | 非本金线 |

不要承诺同路径 2× e2e；不要把 canary 当 cafe 金线；不要再推 MagCache/SLA 当主线。

---

## 3. 架构——统一 registry（P0 已落骨架）

### 3.1 模块

```
tools/h3_cuda_generate  --sdpa-backend / setenv / 打印 caps+resolve
         │
h3_cuda_device_caps     SM / CUDA / FP4 / sage3.so / sol_cute.so
         │
h3_sdpa_registry        resolve(auto|cudnn|sage3|sol_cute|…)
         │
    ┌────┼────────────┐
 cudnn   sage3 plugin  sol_cute plugin
 (in-tree) libh3_sage3.so  libh3_sol_cute.so
         │
   DiT 只调 sdpa_run；不知 sm==120
```

| 文件 | 职责 |
| --- | --- |
| `h3_cuda_device_caps.h/.c` | `h3_cuda_query_caps` / `h3_cuda_caps_format` |
| `h3_sdpa.h` / `h3_sdpa_registry.c` | 表 + `h3_sdpa_resolve` / `h3_sdpa_log_selection` |
| `h3_gpu_cuda.cu` | want_* → registry id；插件 dlopen **无机器路径默认** |
| `tools/h3_cuda_info.c` | 打印 caps + resolve |
| `tools/h3_cuda_generate.c` | Tier-1 缺库 **直接 error 退出**（不静默假核） |

### 3.2 resolve 策略

| 请求 | 行为 |
| --- | --- |
| 未设 / `auto` | **cudnn**；reason 写明永不自动 sage3/sol |
| `cudnn` | Tier-0 |
| `sage3` | 需插件；缺则清晰错误。caps 无 FP4 时 reason 警告 |
| `sol_cute` | 需插件 + `H3_SOL_ATTN_ROOT`；缺则清晰错误 |
| `sage2` / `sol` | 可跑参考，但日志 **FORBIDDEN as product** |

环境变量（正式）：

- `H3_CUDA_SDPA_BACKEND`
- `H3_SAGE3_LIB` / `H3_SAGE3_ROOT` / `H3_SAGE3_TOOLS`（5090）
- `H3_SOL_CUTE_LIB` / `H3_SOL_ATTN_ROOT` / `H3_TRITON_OVERLAY`（4090）
- Sol 可选：`H3_SOL_FIRST_DENSE_STEPS=0` `H3_SOL_FIRST_DENSE_LAYERS=0`（测满 hits；默认仍 2/2 对齐官方 dense 比例）

**禁止**把个人机绝对路径写回源码默认值。

### 3.2.1 如何开 / 缺什么硬错 / 已测数字（2026-08-25 本轮）

#### Sage3（5090）

```sh
export H3_SAGE3_LIB=$PWD/build-cuda-sm120/libh3_sage3.so
export H3_SAGE3_ROOT=/path/to/sageattention3_blackwell   # 必含 fp4attn 构建产物
export H3_SAGE3_TOOLS=$PWD/tools
export PYTHONPATH=$H3_SAGE3_TOOLS:$H3_SAGE3_ROOT/build/lib.linux-x86_64-cpython-310:$H3_SAGE3_ROOT
./build-cuda-sm120/h3_cuda_generate ... --sdpa-backend sage3
```

| 缺什么 | 行为 |
| --- | --- |
| `H3_SAGE3_LIB` / dlopen 失败 | generate **硬错**（registry / dlopen） |
| `H3_SAGE3_ROOT` | init **硬错** |
| 非 FP4 卡仍可请求 | resolve 警告；缺插件则硬错 |

已测（cafe 864×480 5s 8-step，T=15973 H=56 D=128）：

| 项 | 数字 |
| --- | --- |
| hits / fallback | **400/400，fallback=0**（warm） |
| layout | `F16_BHTD` + `prep=cuda_thd_bhtd`（Python permute 已删） |
| DiT warm（本轮） | sage3 **27.4s** vs cudnn **31.4s** |
| 质量 | MAE~27 → **勿改默认** |

#### Sol CuTe（4090）

```sh
export H3_SOL_CUTE_LIB=$PWD/build-cuda/libh3_sol_cute.so
# 必须是官方 preprocess：无条件 TensorDescriptor，无 _prepare_torch
export H3_SOL_ATTN_ROOT=/path/to/techniques/sparse_backends
export H3_TRITON_OVERLAY=/path/to/triton>=3.3   # 本机常用 triton 3.4 overlay
export PYTHONPATH=$H3_TRITON_OVERLAY:$H3_SOL_ATTN_ROOT:$PWD/tools
export H3_SOL_FIRST_DENSE_STEPS=0 H3_SOL_FIRST_DENSE_LAYERS=0   # 测满核命中
./build-cuda/h3_cuda_generate ... --sdpa-backend sol_cute
```

| 缺什么 | 行为 |
| --- | --- |
| `H3_SOL_CUTE_LIB` | **硬错** |
| `H3_SOL_ATTN_ROOT` | **硬错**（warmup 也不再回退到 sparse 树） |
| ROOT 仍是 `_prepare_torch` / Triton&lt;3.3 | embed **硬错**（拒绝 pytorch_fallback） |
| backend ≠ `cute_sm89` | embed **硬错** |

已测（4090，官方 `sol-engine-official` @ `d0c0a468`，cafe ids，1s×2-step）：

| 项 | 数字 |
| --- | --- |
| backend / preprocess | **`cute_sm89` / `triton_prepare`** |
| kernel | `SolAttnForwardSm89`，compile key MATCH |
| hits / fallback | **100/100，fallback=0，dense_skip=0** |
| 视觉门 | 仍不过（既有结论）；非质量默认 |

**ROOT 判别：** `grep _prepare_torch $H3_SOL_ATTN_ROOT/sol_attn/preprocess.py` 必须为空；官方树在 4090 为 `/home/haiyang/h3_c/sol-engine-official/techniques/sparse_backends`（`sol-engine-sparse` 含 `_prepare_torch`，**禁止**当产品 ROOT）。

### 3.3 Build

| 机器 | 目录 | CMake |
| --- | --- | --- |
| 4090 | `build-cuda` | 勿 `H3_REQUIRE_CUDA13`；arch 89 |
| 5090 | `build-cuda-sm120` | `H3_REQUIRE_CUDA13=ON`；arch 120 |
| 插件 | 同 build 出 `libh3_sage3.so` / `libh3_sol_cute.so` | 需 Python3 Development |

---

## 4. 迁移步骤

### P0 — caps + registry + auto→cudnn（骨架 ✅）

- [x] 本文档 Tier 修订  
- [x] `h3_cuda_query_caps` + `h3_sdpa_resolve`  
- [x] generate / info 打印 caps；Tier-1 缺库硬错  
- [x] 去掉 sage3/sol_cute 机器硬编码默认路径  
- [x] `auto` → cudnn（修旧注释 auto→flash）  
- [x] 4090 / 5090 冒烟确认接口（1s canary；见下文状态）

### P1 — Tier-1 商用打磨（优先于 MagCache/SLA）

- [x] Sage3：prep THD↔BHTD CUDA 化；hits 满、fallback=0（本轮 warm 400/0）  
- [x] Sol CuTe：官方 ROOT 真核命中 `cute_sm89`（本轮 hits=100 fallback=0）；视觉门 / 墙钟 ROI 仍待迭代  
- [ ] CI 矩阵：cudnn（双卡）+ sage3（5090）+ sol_cute（4090）  
- [x] 文档与 help 保持 Tier-0 / Tier-1 措辞一致（本轮更新）  

### P2 — Tier-0 墙钟稳（次于 Tier-1 投资，但仍做）

- [ ] 5090 TE 抖动 † 档稳定 ~66s  
- [ ] 条件图 vision→cudnn  

### 明确不做（本阶段）

- nested-loop Sol 再 bench  
- auto→sage3  
- MagCache/SLA 抬成重点  
- 同路径 2× e2e 营销数字  

---

## 5. 给用户的摘要

1. **质量默认：** `--sdpa-backend cudnn`（`auto` 同）。  
2. **重点加速（一等支持）：** `sage3`（5090）、`sol_cute`（4090）；同一 registry。  
3. **本轮：** Sol 官方 ROOT **真核命中**；Sage3 **CUDA prep** 落地，hits 满；默认仍 cudnn。  
4. **已测诚实：** Sage3 更快但 MAE~27；Sol 视觉不过——可商用切换，不可当质量默认。  
5. **假核禁止：** nested `sol` / 参考 `sage2` / sparse `_prepare_torch` ROOT。  
6. **还差：** Sol 视觉门、双卡 CI 矩阵、5090 TE 稳档。

---

## 6. 索引

| 项 | 出处 |
| --- | --- |
| 4090 cafe ~75s | 4090 accel plan |
| 5090 cudnn † ~66s / sage3 MAE~27 | `out/verify_5090_bench/REPORT.md` |
| registry / caps | `h3_sdpa.h`、`h3_cuda_device_caps.h` |
| CLI | `tools/h3_cuda_generate.c` `--sdpa-backend` |
| 构建与切换 | `docs/nvidia_build.md` |

密码未写入本文。未 git commit。
