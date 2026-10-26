# CUDA sync 审计（C070 · 2026-10-27）

> **范围：** 本仓生产路径 + 本仓 probe 中的 `cudaDeviceSynchronize` / 明显阻塞 D2H。  
> **不做：** 不改热路径；不跑 GPU；不碰 NVFP4 链式 WIP；第三方 `third_party/` / `bench/` / `experiments/` 仅作对照备注。  
> **目标：** 给后续 `H3_FORBID_HOT_SYNC` / 白名单提供分类清单（DeepInfer S8 思路）。

---

## 1. 分类标签

| 类 | 含义 | 处置 |
|---|---|---|
| **lifecycle** | 阶段边界（TE→DiT→VAE 释放、设备切换、池销毁） | 保留；计入墙钟时单独标 |
| **kpi_boundary** | 仅在 profile/计时段首尾 | 保留；禁止算进「算子加速」 |
| **debug** | checksum / diag / probe / 单测 | 默认关或仅工具；KPI 禁开 |
| **hot_suspect** | 可能落在 per-step / per-block 热路径 | **优先审计**；需 stream 事件替代或证明已在边界 |

阻塞 D2H（`cudaMemcpy(..., DeviceToHost)` 默认同步）与 `cudaDeviceSynchronize` 同等对待：若在步内循环，标 **hot_suspect**。

---

## 2. `cudaDeviceSynchronize` 清单（生产 / 旁路）

| 文件 | 行（约） | 类 | 备注 |
|---|---|---|---|
| `h3_dit_w4a8.c` | ~69 | debug / lifecycle | 早期 init/open 路径 |
| `h3_dit_w4a8.c` | ~726, ~738 | debug | 条件 diag 分支 |
| `h3_dit_w4a8.c` | ~3098, ~3188 | debug / hot_suspect | 与 D2H 对照核相邻；确认是否仅校验臂 |
| `h3_dit_w4a8.c` | ~6954（注释） | — | 明确**避免**全设备 sync，改 stream wait |
| `h3_gpu_cuda.cu` | ~3670, ~3679 | debug | weight checksum / diag copy 前后 |
| `h3_gpu_cuda.cu` | ~3751, ~3759 | debug | diag scalar D2H |
| `h3_gpu_cuda.cu` | ~9662 | lifecycle / hot_suspect | 需对照调用栈是否 per-forward |
| `h3_nvfp4_cuda.cu` | ~1182 | lifecycle / debug | NVFP4 路径；链式 WIP 勿借机改 |
| `h3_tae_cuda.cu` | ~731 | lifecycle | TAE decode 收尾 |
| `h3_audio_vae_cuda.c` | ~1128, ~1204 | lifecycle | audio VAE；`stream ? streamSync : deviceSync` |
| `h3_vae_int8_cuda.c` | ~4156, ~4166 | debug / lifecycle | VAE 段；确认非每 tile |
| `tools/h3_audio_ops_probe.c` | 多处 | debug | probe only |
| `tools/h3_p2p_bw_probe.cu` | warmup | kpi_boundary | probe |
| `bench/*` / `experiments/*` | 多处 | debug | 非产品路径 |

> 行号会随提交漂移；以符号/注释定位为准。本表由 2026-10-27 静态 grep 生成。

---

## 3. 阻塞 D2H 热点（摘录）

| 文件 | 模式 | 类 | 备注 |
|---|---|---|---|
| `h3_dit_w4a8.c` | `cudaMemcpy(... DeviceToHost)` 邻近 sync | debug / hot_suspect | ~3107–3193 对照核；~4493+ 条件路径 |
| `h3_gpu_cuda.cu` | 默认同步 D2H GEMM/SDPA out | lifecycle / hot_suspect | 多数在 API 边界把结果带回 host；Ulysses 已有 async+event 注释 |
| `h3_gpu_cuda.cu` | diag 多缓冲 D2H | debug | ~3687+ |
| `tools/h3_cuda_generate.c` | 几乎不直接 sync | — | 依赖库边界；C040 账本用 `cudaMemGetInfo`（非 sync） |

原则：步内「为了 printf 一个 scalar」的 D2H+sync → 迁到 `H3_DIAG_*` 或删；KPI 跑禁开。

---

## 4. 建议白名单（草案，未接线）

允许（lifecycle / kpi_boundary）：

- generate 阶段切换前后（DiT 释放→VAE、clip 结束）
- probe / waffle / checksum 工具入口
- nsys 区间边界（已有 `H3_NSYS_HIRES`）

拒绝进默认热路径（未来 `H3_FORBID_HOT_SYNC=1` 可 assert）：

- per DiT block / per denoise step 的无条件 `cudaDeviceSynchronize`
- 非 diag 的标量 D2H

**Forbid 开关已落地（2026-10-27 晚）：** `H3_FORBID_HOT_SYNC=1` + `h3_cuda_hot_sync.h`；生产路径已知 `cudaDeviceSynchronize` 已按类包裹（uly=DEBUG，lora f16=LIFECYCLE，sol sdpa_ms fallback=HOT）。默认 unset=不改行为。

---

## 5. 与目录映射

| ID | 状态（本提交后） | 说明 |
|---|---|---|
| **C070** | **partial→landed(forbid)**（审计页 + `H3_FORBID_HOT_SYNC`） | 本文；238 smoke 见 gpu-unified |
| C040 | partial（插桩 landed；238 未跑） | `H3_VRAM_LEDGER` |
| C065 | landed 纪律 | KPI 禁 mem_trace / 乱 sync |

---

## 6. 修订

| 日期 | 说明 |
|---|---|
| 2026-10-27 | 初版：静态 grep 分类；不改代码热路径；不跑 GPU。 |
| 2026-10-27 | forbid 开关落地：`h3_cuda_hot_sync.h` + 已知 sync 点分类包裹；默认 off。 |
