# h3.c 后续优化方案（金线多卡 Ulysses + 快档 SKU）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.  
> **禁止**改 `h3_cuda_generate` 默认 backend / steps / mag / tea / fb / sage / sol；**禁止**改写单卡金线 75s/66s。不把密码写入 repo。本文件只规划；执行时另开会话。  
> **冲突裁决：** 金线 KPI 仍以 [2026-08-26-h3c-next-tech-plan.md](./2026-08-26-h3c-next-tech-plan.md) + `out/verify_accel_matrix/REPORT.md` §F 为准。机制 / 拓扑以 [accel-survey-tech-deepdive](./2026-09-11-h3c-accel-survey-tech-deepdive.md) + [pcie-multigpu-lessons](./2026-09-11-h3c-pcie-multigpu-lessons.md) 为准。少步质量以 [few-step-quality](./2026-09-11-h3-few-step-quality.md) 为准。768p / 二采秒数以 [8x5090-768p-estimate](./2026-09-11-h3c-8x5090-768p-estimate.md) + [two-stage-768p](./2026-09-11-h3c-two-stage-768p.md) 为准（**未测**）。本文不覆盖、不改写那些数字。  
> **极速 SKU 实现：** [2026-09-11-h3c-5090-turbo-implementation.md](./2026-09-11-h3c-5090-turbo-implementation.md)（`--preset turbo4`；本文 P3 `fast4` 并入该档，不另开第二套 preset）

**Goal:** 在守住 INT8 8-step + cuDNN 金线（MAE∩有声）的前提下，于 8×5090（无 NVLink）落地 **Ulysses 序列并行**（同轨迹砍 DiT 墙钟），并把少步 / 二采做成 **独立快档 SKU**（不过门不开默认）。

**Architecture:** 金线永远是单卡 dense cuDNN、cfg=1、steps=8。多卡只切 packed 序列上的 SDPA（TP=1 复制 INT8 权；attn 内 A2A 变头切）。TE / VAE 默认仍单卡。快档（4-step LoRA / draft+refine）独立 CLI 与独立 KPI。本仓当前 **无多卡实现**。

**Tech Stack:** 本仓 C/CUDA `h3_cuda_generate`；NCCL（U≥4）+ 2 卡可选 CUDA-IPC A2A；`sdpa=cudnn`；8×5090 `build-cuda-sm120` + CUDA 13.2 / sm_120。对照仓仅作机制参考，不进默认路径。

**Spec:** 本文综合下列已写文档，不另起炉灶：

| 文档 | 本文怎么用 |
| --- | --- |
| [accel-survey-multigpu-5090](./2026-09-11-h3c-accel-survey-multigpu-5090.md) | 可抄 / 永不碰总判 |
| [accel-survey-tech-deepdive](./2026-09-11-h3c-accel-survey-tech-deepdive.md) | Ulysses 通信、头数约束、Amdahl |
| [pcie-multigpu-lessons](./2026-09-11-h3c-pcie-multigpu-lessons.md) | 无 NVLink：少 TP 多 Ulysses；2 卡 IPC |
| [few-step-quality](./2026-09-11-h3-few-step-quality.md) | 4-step 必须换学生权；金线已是 8-step CFG 蒸馏 |
| [8x5090-768p-estimate](./2026-09-11-h3c-8x5090-768p-estimate.md) | 5s 768p 外推（标未测） |
| [two-stage-768p](./2026-09-11-h3c-two-stage-768p.md) | 二采快档成立条件 |
| [2026-08-26-h3c-next-tech-plan](./2026-08-26-h3c-next-tech-plan.md) | 旧 P0–P3 已大多落地；本文接管其 **M-GPU** |

## Global Constraints（全文隐含）

- 质量默认：**INT8 ConvRot DiT + fp16 VAE tile+deblock，cfg=1，steps=8，864×480，有声**；`sdpa=cudnn`；reuse / mag / tea / fb / sage / sol **默认关**
- 单卡金线：**4090 cafe ~75s**；**5090 cafe cudnn warm ~66–76s e2e、DiT ~31–32s**（矩阵 n=3：e2e **66.06±5.80**，DiT **31.37±0.05**）
- Prompt：cafe 574 ids，seed `0x11`；真 `--seconds 5`；**禁止** `--frames` 假时长
- 多卡 KPI **另册**（`out/verify_5090_sp2/` 等），**不改 66s 行**
- `num_gpus` / Ulysses ∈ {2,4,8}；`H3_DIT_W4A8_HEADS=56`；packed 对齐 **64**（**U=7 非法**，64%7≠0）
- 8×5090：`H3_SSH_5090x8` 默认 `user@89.185.31.140`；无 NVLink；32GB/卡；CUDA 13.2；密码仅 `H3_SSH_PASS_5090x8`
- 旧单卡 5090：`H3_SSH_5090`，局域网常通 `yongbin@192.168.0.238`
- 根目录 `/scripts` **被 .gitignore**；新 harness 放 **`tools/`**，勿写入被忽略的 `scripts/`
- **密码不入库**；本轮规划 **不 commit**

---

## 1. 目标分列

### A — 金线保效果加速（多卡 Ulysses）

| 项 | 定义 |
| --- | --- |
| 配方 | 与单卡金线 **同一条轨迹**：INT8 8-step、`sdpa=cudnn`、cfg=1、864×480 有声 |
| 刀 | **只切 DiT packed 序列**（Ulysses；TP=1）；权 INT8 **每卡副本** |
| 门 | 相对 **同机单卡 5090 cudnn** 同 prompt/seed 的 PPM MAE ∩ 口型/有声 |
| 成功 | 2 卡：DiT 明显下降且过门；数字写 `out/verify_5090_sp2/`。默认 CLI 仍是单卡 cudnn |
| 非成功 | e2e 到不了 33s（TE+VAE 地板）；把 12.2× 写进 66s |

### B — 快档 SKU（少步 / 二采）

| 项 | 定义 |
| --- | --- |
| 配方 | 独立 preset（如 `--preset fast4` 或 `draft480_refine768`）；help 写 **非金线** |
| 候选 | FastH3 / larryvrh **DMD2 4-step LoRA**；或 Sol 式低分草稿 + 轻量 refine（H3→H3 少步，**不是** GB200 Super Accel） |
| 门 | **自建 KPI**，阈值不得冒充金线 MAE；不过门不开默认、不进 A 的对照表 |
| 成功 | 产品能切档；默认仍 A |

### C — 不做

见 §7。摘要：4-step 当无损；训练 VSA 进默认；TAE 金线成片；6.85s/GB200 写 5090；Sage/Cache 进默认；12.2× 当多卡比；本仓无多卡却对外承诺 8 卡秒数。

---

## 2. 现状一页

| 轴 | 状态 | 证据 |
| --- | --- | --- |
| 单卡金线 | ✅ 产品默认 | 4090 ~75s；5090 cafe cudnn warm **66.06±5.80s** / DiT **31.37s**（`out/verify_accel_matrix/REPORT.md` §F） |
| `sdpa=cudnn` | ✅ Tier-0 | `auto`→cudnn；`h3_sdpa.h` |
| Tea / Mag / FB | ✅ CLI 已落；**默认关** | Tea 0.04 skip 1；Mag 0.08 skip 1 MAE 历史 FAIL；FB 0.04 **8/0** |
| sage3 / sol_cute | ✅ Tier-1；MAE FAIL | sage3 DiT −17% MAE~27；sol_cute MAE~35 |
| Vision cuDNN | ✅ KEEP | L2VA layers 32.4s→2.8s；T2VA 0 收益 |
| **多卡 SP/TP** | **❌ 产品未成** | 无 `ulysses` 符号；generate 仅 `--device` 单卡 |
| 本仓 768p generate | ❌ 从未跑 | 768p 秒数全是公式外推 |
| 8×5090 机 | ◐ 已探测，**无工作区** | `docs/nvidia_build.md`：8×5090、32GB、无 NVLink、CUDA 13.2；尚未 sync 源码/权重 |
| 旧 8-26 P0–P3 | ✅ 大多落地 | 矩阵 / Tea / FB / Vision；**M-GPU 未做 → 本文接管** |

Amdahl（5090 cafe，已测）：e2e ~66s ≈ TE 12–14 + **DiT 31.4** + VAE ~10.5。Ulysses **只砍 DiT**。2 卡理想也到不了 33s e2e。

---

## 3. 借鉴结论表（抄机制 / 快档 SKU / 永不碰）

| 来源 | 金线可抄（进 A） | 快档另册（进 B） | 永不碰 |
| --- | --- | --- | --- |
| **FastH3** | — | DMD2 **4-step LoRA** 作独立 SKU；隔离跑 Triton VSA 对照 | 训练 VSA 进默认；4-step 当无损；sm100a 核当 5090 默认 |
| **Sol-H3 Super** | — | **低分草稿 + 轻量 refine** 二采 SKU（H3→H3 少步；音频跟草稿 PCM） | **6.85s / 2×GB200** 写 5090；TAE 当金线成片；LTX / `cute_sm100` 进默认 |
| **VDN** | **8-NFE 哲学**（金线停在 8）；**Ulysses/SP 思想**（切 packed、attn 内 A2A） | hybrid attn（窗 softmax + 线性支）**研究档**；过 MAE 才谈 | 65GB BF16/FP8 常驻假设；未过 MAE 当默认；11.23s 当无损金线 |
| **RH Lightning** | **PCIe 少 TP 多 Ulysses**；头数/`64` 对齐硬校验；**A2A（置换）优于 AR（归约）**；2 卡优先 IPC | 其 4-step + Sage2 + Cache-DiT + compile **整栈**另册 | **12.2× 当多卡比**（含步数变化）；Sage/Cache/compile **进默认**；从 TP2 起步 |

**一句话：** A 只抄 Ulysses/SP。墙钟数量级下降必须换配方，走 B。没有「无损加速本仓 66s」的现成补丁。

---

## 4. 文件落点（执行前锁定）

| 路径 | 职责 |
| --- | --- |
| `h3_ulysses.h` / `h3_ulysses.c` | 头数与 64 对齐校验；rank 配置；pad 到 `S_pad` |
| `h3_ulysses_comm.cu` | NCCL all-to-all；U=2 可选 CUDA-IPC；packed QKV 一次 destination-major 集体 |
| `h3_dit_w4a8.c` | `block_device_compute`：U>1 时 QKV 后 A2A → 现有 `h3_cuda_qkv_rope_sdpa_f16_dev`（`heads=H/U`）→ 输出 A2A；MLP 仍在 `S/U` 行 |
| `tools/h3_cuda_generate.c` | **opt-in** `--ulysses N`（默认 **1** = 今日单卡）；help 写「实验、非默认」；N=1 零行为差 |
| `CMakeLists.txt` | `libnccl`；`h3_ulysses_test`；`h3_cuda_generate` 链接 comm；nvcc 候选含 **13.2** |
| `tests/test_h3_ulysses.c` | 无 GPU：合法/非法 (T,U,R)；pad 公式 |
| `tests/test_h3_ulysses_a2a.cu` | 有 GPU：2 卡置换往返 vs 单卡参考（位级或 max-abs 门） |
| `tools/h3_5090x8_sync.sh` | rsync **无 `--delete`**；密码只走 env |
| `out/verify_5090x8_gold/` | P0 单卡金线复现 |
| `out/verify_5090_sp2/` | P1 Ulysses2 另册 |
| `out/verify_5090_sp4/` · `out/verify_5090_sp8/` | P2 |
| `out/verify_fast_sku/` | P3 快档，禁止写入金线表 |

不改：`h3_sdpa` 默认 resolve；`--sdpa-backend` 缺省；`--steps` 缺省；cache 缺省。

---

## 5. 分阶段（每阶段 1–2 周可收口）

并行标注：`‖` = 可与同阶段其它任务并行。阶段门：P0 金线复现齐 → 开 P1；P1 MAE∩有声过门 → 开 P2；P3 **不依赖** P2 过门，但 **不得** 与 A 的 KPI 混表。

### P0 — 8×5090 环境（约 3–5 人天）

**动机：** 新机尚无工作区。先把尺子搬过去，再谈多卡。单卡金线复现失败则 **停止**，不进入 Ulysses。

#### Task P0.1: 连通与探测（不拷模型）

**Files:** 只读 `docs/nvidia_build.md`；写 `out/verify_5090x8_gold/HOST.md`（无密码）。

- [ ] **Step 1: SSH（密码仅会话）**

```sh
export H3_SSH_5090x8=user@89.185.31.140
# export H3_SSH_PASS_5090x8=...    # 仅当前 shell；禁止 echo 进文件
ssh -p 22 "$H3_SSH_5090x8" 'nvidia-smi -L; nvidia-smi nvlink -s; echo PATH_NVCC=$(command -v nvcc); ls /usr/local/cuda/bin/nvcc; nvidia-smi topo -m | head'
```

Expected: 8× GeForce RTX 5090；`nvlink` 报不支持；nvcc 在 `/usr/local/cuda/bin`（默认 PATH 常无 nvcc）。

- [ ] **Step 2: 把探测写进 HOST.md**（卡数、驱动、CUDA 13.2、无 NVLink、NUMA 0=GPU0–3 / NUMA 1=GPU4–7、磁盘余量）。禁止写密码。

**通过：** 8 卡属实、无 NVLink 属实。 **失败：** 连不上就修 SSH（`H3_SSH_BIND`），不编造拓扑。

#### Task P0.2: sync 源码（无 `--delete`）

**Files:** Create `tools/h3_5090x8_sync.sh`。

仓库根 `/scripts` 被 gitignore，**不要**往那里加。rsync **禁止 `--delete`**（矩阵报告曾误删远程 models 空壳）。

- [ ] **Step 1: 写 sync 脚本**（可执行、无密码字面量）

```sh
#!/usr/bin/env bash
# tools/h3_5090x8_sync.sh — source tree only. Never rsync --delete. Never print passwords.
set -euo pipefail
HOST="${H3_SSH_5090x8:-user@89.185.31.140}"
REMOTE_ROOT="${H3_REMOTE_ROOT:-/home/user/h3_c}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ssh -p 22 "$HOST" "mkdir -p '$REMOTE_ROOT'"
rsync -az --exclude '.git/' --exclude 'out/' --exclude 'models/' --exclude 'third_party/accel_survey/FastVideo/' \
  --exclude 'third_party/accel_survey/vdn-minimax-h3/' \
  --exclude 'third_party/accel_survey/MiniMax-H3-MultiGPU-Lightning/' \
  -e "ssh -p 22" "$ROOT/" "$HOST:$REMOTE_ROOT/"
echo "synced source -> $HOST:$REMOTE_ROOT (no --delete, no models)"
```

- [ ] **Step 2: 跑 sync；远端 `ls` 见到 `CMakeLists.txt` `h3_dit_w4a8.c` `tools/h3_cuda_generate.c`**

**通过：** 远端有源码。 **失败：** 把密码写进脚本 / 使用 `--delete`。

#### Task P0.3: sm_120 构建

**Files:** Modify `CMakeLists.txt` nvcc 候选（现有列表含 13.1/13.0/`/usr/local/cuda`，**补 `cuda-13.2`**）；不改 generate 默认。

远端：

```sh
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
cd /home/user/h3_c
cmake -S . -B build-cuda-sm120 -DH3_GPU_BACKEND=cuda -DH3_REQUIRE_CUDA13=ON \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build-cuda-sm120 -j --target h3_cuda_info h3_cuda_generate h3_quality_gate_probe h3_step_cache_test
./build-cuda-sm120/h3_cuda_info
./build-cuda-sm120/h3_step_cache_test
```

Expected: nvcc 13.2.x；arch 含 **120**；`h3_cuda_info` 见 RTX 5090 sm_120；step_cache PASS。

- [ ] **Step 1: 补 CMake 候选 `/usr/local/cuda-13.2/bin/nvcc`（若 configure 已能命中 `/usr/local/cuda` 则只记一笔，不必强改）**
- [ ] **Step 2: 远端 configure + build 上列 target**
- [ ] **Step 3: `h3_cuda_info` 日志存 `out/verify_5090x8_gold/cuda_info.txt`**

**通过：** 二进制在 5090 上跑起来。 **失败：** 用 CUDA 12 凑合、或改 generate 默认 backend。

#### Task P0.4: 权重与 cafe ids（有意拷贝，不大扫盘）

**Files:** `out/verify_5090x8_gold/MODELS.md`（只写远端路径，不写局域网密码）。

权重建议从 **旧单卡 5090**（`H3_SSH_5090`，常通 `yongbin@192.168.0.238`）拉已经验证过的 INT8 DiT / fp16 VAE / TE / audio VAE / cafe `prompt.ids`。新机磁盘 ~850G 空闲，但仍 **禁止** 擅自整树拷 MiniMax BF16。

- [ ] **Step 1: 列出金线所需 4–5 个 safetensors + cafe 574 ids 的远端路径**
- [ ] **Step 2: scp/rsync **点名文件****（仍无 `--delete`）
- [ ] **Step 3: 写 `MODELS.md`：`H3_MODELS=...` 一节，供后续命令引用**

**通过：** 文件齐全、可 `ls -lh`。 **失败：** 把 BF16 全量 65GB+ 当「先拷着」。

#### Task P0.5: 单卡金线复现 warm×3

**Files:** `out/verify_5090x8_gold/REPORT.md`。对照：`out/verify_accel_matrix/REPORT.md` §F。

金线命令（`--cfg 1` 必须显式；T2VA help 主题默认 cfg=5 **不是** 金线）：

```sh
# GPU0 only. Cache/sage/sol 全部不要传。
export CUDA_VISIBLE_DEVICES=0
OUT=/home/user/h3_c/out/verify_5090x8_gold/run_$i
./build-cuda-sm120/h3_cuda_generate \
  --dit "$DIT" --vae "$VAE" --te "$TE" --audio-vae "$AUDIO" \
  --out "$OUT" --mux "$OUT/out.mp4" \
  --width 864 --height 480 --seconds 5 --steps 8 --cfg 1 \
  --sdpa-backend cudnn --seed 0x11 \
  --token-ids-file "$CAFE_IDS"
# 冷 1 + warm 3（i=1..4）；报表只用 run_2..4
```

- [ ] **Step 1: 冷跑 1 次确认出片 + `ffprobe` 时长 5.0s + 有音频流**
- [ ] **Step 2: 再 warm×3；记录 e2e / DiT / TE / VAE；日志须含 `sdpa=cudnn`、无 mag/tea/sage**
- [ ] **Step 3: 写 REPORT：mean±std；与旧单卡 66s/DiT 31s **同量级** 即可（新机允许 ±15% 墙钟，因 PCIe/NUMA/CPU 不同）；DiT 应仍在 ~31s 带**

**通过：** 有声 5s；`sdpa=cudnn`；warm DiT ~31–32s 量级。 **失败：** 默认被改成 sage/tea；或用 `--frames` 假 5s。

**P0 回滚：** 删远端工作区或不用即可；本仓默认路径无改。P0 失败 **禁止** 开 P1。

**阶段门：** REPORT 有 warm×3 → P1。

---

### P1 — 本仓 Ulysses2（2 卡）同轨迹 8-step（约 1–2 周）

**动机：** 唯一不换采样、有机会过 MAE∩有声的大延迟杠杆。抄 RH「2 卡 TP=1、U=2、A2A 置换」；不抄 4-step。

**设计锁死（实现前不许改）：**

1. `tp_size=1`，`ulysses=2`，`ring=1`。INT8 权 **每卡一份**。无 per-block all-reduce。
2. 切 **packed 序列行**；attn 前 A2A 成全序列、`H/U=28` 头；attn 后 A2A 还原。MLP/INT8 GEMM 留在 `S_pad/U` 行。
3. 输入 QKV **一次** destination-major 集体，输出再一次 A2A（对齐 RH `_usp_input_all_to_all_packed_qkv`）。
4. 合法度：`heads % tp == 0`；`(heads/tp) % U == 0`；`64 % (U*R) == 0`。非法 **fail-fast**，禁止 silent pad 把头切碎。
5. **序列填充：** cafe packed S≈15973，**15973%2≠0**。必须 `S_pad = ceil(S/64)*64`（16000），再按行切。RoPE/模态 tag 对 pad 行给 dummy；SDPA 后切回逻辑 S。这是功能正确性，不是优化。
6. TE / VAE / mux 仍在 **device 0**。Rank 1 在这些阶段空闲（Amdahl 已承认）。
7. 2 卡放 **同一 NUMA**（GPU0–1 或 GPU0–3 内），`CUDA_VISIBLE_DEVICES=0,1`。不要一上来跨 SYS（GPU0+GPU4）。
8. 通信：P1 先 NCCL `ncclCommInitAll`（**单进程双设备**，避免 MPI 硬依赖）；U=2 再加 CUDA-IPC 快路径（`can_device_access_peer` 失败则退 NCCL，数值位级相同）。**不要**设 `NCCL_P2P_DISABLE`。
9. `--ulysses` 默认 **1**；N=1 不初始化 NCCL，行为与今日金线一致。

#### Task P1.1: 校验器 + pad 公式（无 GPU）

**Files:**
- Create: `h3_ulysses.h`, `h3_ulysses.c`, `tests/test_h3_ulysses.c`
- Modify: `CMakeLists.txt`（`h3_ulysses_test`，不链 CUDA）

**Interfaces:**

```c
/* h3_ulysses.h */
enum { H3_ULYSSES_PACKED_ALIGN = 64 };

typedef struct h3_ulysses_cfg {
    int tp_size;       /* P1: 1 */
    int ulysses;       /* 1,2,4,8 */
    int ring;          /* P1: 1 */
    int rank;          /* 0 .. ulysses-1 */
    int world_size;    /* = ulysses when tp=1 */
} h3_ulysses_cfg;

/* 0 = illegal. fills err. */
int h3_ulysses_validate(int heads, int packed_align, int tp, int ulysses,
                        int ring, char *err, size_t err_size);

/* S_pad = ceil(seq / align) * align; local_rows = S_pad / ulysses */
size_t h3_ulysses_pad_seq(size_t seq, int align);
size_t h3_ulysses_local_rows(size_t seq_pad, int ulysses, int rank);
```

- [ ] **Step 1: 写失败测试**

```c
/* tests/test_h3_ulysses.c */
#include "h3_ulysses.h"
#include <stdio.h>
#include <string.h>
static int fails;
static void expect(int c, const char *m) { if (!c) { fprintf(stderr, "FAIL %s\n", m); fails++; } }
int main(void) {
    char err[128];
    expect(h3_ulysses_validate(56, 64, 1, 1, 1, err, sizeof err) == 1, "U1");
    expect(h3_ulysses_validate(56, 64, 1, 2, 1, err, sizeof err) == 1, "U2");
    expect(h3_ulysses_validate(56, 64, 1, 4, 1, err, sizeof err) == 1, "U4");
    expect(h3_ulysses_validate(56, 64, 1, 8, 1, err, sizeof err) == 1, "U8");
    expect(h3_ulysses_validate(56, 64, 1, 7, 1, err, sizeof err) == 0, "U7 illegal align");
    expect(h3_ulysses_validate(56, 64, 1, 3, 1, err, sizeof err) == 0, "U3 illegal heads");
    expect(h3_ulysses_validate(56, 64, 2, 4, 1, err, sizeof err) == 1, "TP2+U4 legal");
    expect(h3_ulysses_validate(56, 64, 4, 4, 1, err, sizeof err) == 0, "TP4+U4 illegal");
    expect(h3_ulysses_pad_seq(15973, 64) == 16000, "cafe pad");
    expect(h3_ulysses_local_rows(16000, 2, 0) == 8000, "rank0 rows");
    expect(h3_ulysses_local_rows(16000, 2, 1) == 8000, "rank1 rows");
    return fails ? 1 : (printf("test_h3_ulysses: PASS\n"), 0);
}
```

- [ ] **Step 2: 编译跑，确认 FAIL（符号不存在）**

```sh
cmake --build build-cuda-sm120 -j --target h3_ulysses_test
./build-cuda-sm120/h3_ulysses_test
```

Expected: 链接失败或 FAIL。

- [ ] **Step 3: 实现 `h3_ulysses_validate` / `pad_seq` / `local_rows`（纯 C）**

规则（与 RH `_validate_sequence_parallel_config` 同构）：

```
heads % tp == 0
(heads / tp) % ulysses == 0
packed_align % (ulysses * ring) == 0
tp>=1, ulysses>=1, ring>=1
```

- [ ] **Step 4: 再跑测试，Expected: `test_h3_ulysses: PASS`**

**通过：** U=7 / U=3 / TP4+U4 被拒；cafe pad=16000。 **失败：** 抄 FastVideo 只查 `heads%sp` 而放行 U=7。

#### Task P1.2: NCCL A2A 原语（合成张量，2 GPU）

**Files:**
- Create: `h3_ulysses_comm.cu`, `h3_ulysses_comm.h`, `tests/test_h3_ulysses_a2a.cu`
- Modify: `CMakeLists.txt`（`find_library(NCCL nccl)`；链 `h3_ulysses_test_a2a`）

**Interfaces:**

```c
/* h3_ulysses_comm.h — device pointers. dtype_bytes = 2 (fp16 QKV) or 4. */
int h3_ulysses_comm_init(h3_ulysses_cfg *cfg, const int *device_ids, int n_dev,
                         char *err, size_t err_size);
void h3_ulysses_comm_destroy(h3_ulysses_cfg *cfg);

/* packed QKV: each rank holds [S_local, 3, H, D] → after A2A [S_pad, 3, H/U, D]
 * One collective (destination-major), not three. */
int h3_ulysses_a2a_qkv(h3_ulysses_cfg *cfg, const void *d_in, void *d_out,
                       size_t seq_local, size_t heads, size_t head_dim,
                       int dtype_bytes, char *err, size_t err_size);
int h3_ulysses_a2a_attn_out(h3_ulysses_cfg *cfg, const void *d_in, void *d_out,
                            size_t seq_local, size_t heads, size_t head_dim,
                            int dtype_bytes, char *err, size_t err_size);
```

- [ ] **Step 1: 合成测试：2 卡、S_pad=64、H=56、D=8 小张量；A2A 往返等于身份**
- [ ] **Step 2: 远端 `CUDA_VISIBLE_DEVICES=0,1` 跑 `h3_ulysses_test_a2a`；Expected: PASS**
- [ ] **Step 3: 日志打印 `peer_access=1/0`；P2P 关着则记 WARN，仍应数值正确**

**通过：** 往返 max-abs==0（整数填充）或 fp16 往返位级相同。 **失败：** 默默改 `NCCL_P2P_DISABLE=1` 当「最佳实践」。

#### Task P1.3: 接入 DiT block（opt-in）

**Files:**
- Modify: `h3_dit_w4a8.c` `block_device_compute`（U=1 走现路径，**零额外 A2A**）
- Modify: `tools/h3_cuda_generate.c`：`--ulysses N` 默认 1；与 `--device` 组合：N>1 时忽略单 `--device`，用 `0..N-1`
- Modify: `CMakeLists.txt`：`h3_cuda_generate` 链 `h3_ulysses.c` + `h3_ulysses_comm.cu` + nccl

插点（现路径，约 `h3_dit_w4a8.c` `h3_cuda_qkv_rope_sdpa_f16_dev` 前后）：

```
QKV GEMM on local rows [S_local, 3*H*D]
→ (U>1) A2A QKV → [S_pad, 3*(H/U)*D]
→ rope+sdpa with heads=H/U, seq=S_pad   /* 现有 cudnn 核，不要换 sage */
→ (U>1) A2A out → [S_local, H*D]
→ out-proj + MLP on local rows
```

- [ ] **Step 1: `--ulysses 1` cafe 短冒烟（`--seconds 1 --steps 2`）与改前 checksum/日志一致（允许非 bit，但 sdpa=cudnn、无 NCCL init）**
- [ ] **Step 2: `--ulysses 2` 短冒烟出片；非法 `--ulysses 7` 非零退出且 stderr 含 `packed align`**
- [ ] **Step 3: help 增加 `--ulysses N`：** 实验性序列并行，默认 1；N∈{1,2,4,8}；**不改** `--sdpa-backend` 缺省

**通过：** 默认路径无 NCCL；U=2 能出短片。 **失败：** 默认 U=2；或 U=2 时自动开 sage3。

#### Task P1.4: cafe 同轨迹 8-step 质量门 + 另册计时

**Files:** `out/verify_5090_sp2/REPORT.md`

对照：P0 的 `out/verify_5090x8_gold/` 同 seed 成片。

```sh
export CUDA_VISIBLE_DEVICES=0,1
./build-cuda-sm120/h3_cuda_generate ... --ulysses 2 --sdpa-backend cudnn \
  --steps 8 --cfg 1 --seconds 5 --seed 0x11 --token-ids-file "$CAFE_IDS" \
  --out /home/user/h3_c/out/verify_5090_sp2/run_$i
```

门（必须同时）：

| 门 | 怎么测 | 过 |
| --- | --- | --- |
| 视觉 | `h3_quality_gate_probe --ppm-a gold.ppm --ppm-b sp2.ppm` 抽 f0/24/48/90/119；目视口型/道具 | **不可辨**。Ulysses 是置换不是新核：若 MAE 落到 sage3（~27）/sol（~35）量级，当 **实现 bug**，不要改阈值放行 |
| 有声 | ffprobe 双流；时长 5.0s；波形非静音；口型目视 | 缺音频 = FAIL |
| 墙钟 | warm×3；报 e2e **和** DiT | DiT 相对单卡 ~31.4s **明显下降**（机制预期 ~20s 量级已算拓扑赢，**非承诺**）。e2e 仍被 TE+VAE 托底，**禁止**写「2 卡 → 33s」 |

- [ ] **Step 1: 同机单卡 cudnn 金线片子作 GT（可复用 P0）**
- [ ] **Step 2: U=2 warm×3；填 REPORT 表：e2e/DiT/TE/VAE、MAE 五帧、有声 Y/N、`sdpa=cudnn`、`ulysses=2`**
- [ ] **Step 3: 不过门 → 留实验分支，`--ulysses` 保持默认 1；矩阵写 FAIL。禁止叠 sage/tea「先把秒数做漂亮」**

**通过：** MAE∩有声过 + DiT 下降。 **失败即关。** 数字 **禁止** 除 66s。

**P1 回滚：** `--ulysses` 默认 1；`h3_ulysses_comm_init` 在 N=1 不调用；可 `#if 0` 掉 A2A 插点。金线二进制行为与 P0 一致。

**阶段门：** 过门 → P2。不过门 → 修 A2A/pad，**不要**开 U=4。

---

### P2 — Ulysses4/8；TE/VAE 是否另刀（约 1–2 周）

**动机：** 8 卡机器的合法拓扑是 **U=8 TP=1**（56%8=0，64%8=0），**不要默认抄 RH 的 TP2+U4**（那是 6000D BF16 768p 工作集折中）。TE/VAE 不随 Ulysses 变便宜，e2e 要再砍才单独决策。

#### Task P2.1: U=4 同 NUMA（GPU0–3）

- [ ] **Step 1: `--ulysses 4`，`CUDA_VISIBLE_DEVICES=0,1,2,3`（NUMA 0）。U=4 无 IPC 快路径，走 NCCL all-to-all**
- [ ] **Step 2: cafe 8-step 同 P1 门；REPORT `out/verify_5090_sp4/`**
- [ ] **Step 3: 填 k_eff = DiT_1 / DiT_4（未测不算）。通信税预期大于 U=2**

**通过：** 过 MAE∩有声。DiT 再降则记赢；e2e 若几乎不动（TE+VAE 主导）也要诚实写。

#### Task P2.2: U=8（跨 NUMA，必测税）

- [ ] **Step 1: `--ulysses 8`，GPU0–7。跨 SYS。先量 **同配方** U=8 vs U=4 的 DiT，不要先加 TP**
- [ ] **Step 2: `out/verify_5090_sp8/REPORT.md`。若跨 NUMA 把 U=8 打成比 U=4 更慢，记「U=4 为 480p 甜点」，U=8 留 768p**
- [ ] **Step 3: 仅当 U=8 显存/实现爆了才允许试 TP2+U4。864×480 INT8 **预期用不上**。试之前必须 P2.1 已过门**

**禁止：** 为了「对齐 RH start.sh」一上来 TP2。

#### Task P2.3: TE / VAE 另刀决策（可 FAIL 收口）

Amdahl：单卡 TE 12–14s + VAE ~10s 在 U=8 后会变成 e2e **主项**。

| 刀 | 做什么 | 何时做 | 不做 |
| --- | --- | --- | --- |
| VAE tile/chunk 并行 | FastVideo bit-exact 时间 chunk 可参考；本仓已有 tile decode | **仅当** P1 过门且 e2e 仍被 VAE 托住，产品要明显低于 66s 量级 | 未过 MAE 就切 VAE |
| TE | 单请求 batch=1，encoder DP **无收益** | **本阶段不做** | 为「看起来对称」切 TE |
| 768p 单卡/U=8 探路 | 先 **单卡** 864×480 以外的 1344×768 能否常驻 32GB | 独立 `out/verify_5090_768p_single/`；OOM 记缺口，不编秒数 | 把估时文 70–110s 当实测 |

- [ ] **Step 1: 写决策段进 `out/verify_5090_sp8/REPORT.md`：VAE 刀做 / 延后 / 不做，各一行理由**
- [ ] **Step 2: 若做 VAE 并行：独立 CLI（如 `--vae-parallel N`），默认 0；质量门相对单卡 decode 近 bit-exact**

**P2 回滚：** `--ulysses` 默认仍 1；U=4/8 仅显式打开。失败拓扑不进文档「已支持」列表。

**阶段门：** 至少 U=2 过门且 U=4 有诚实数字（可 FAIL）→ 收口。P3 可 ‖ 调研，但实现不得改 A 默认。

---

### P3 — 快档 SKU（约 1–2 周；独立 KPI）

**动机：** 要墙钟数量级，必须换配方。VDN 停在 8-NFE；4-step 是第二次蒸馏。**不过门不开默认。**

优先级（保效果递减、墙钟递增）：

| 序 | SKU | 来源 | 本仓缺口 |
| --- | --- | --- | --- |
| B1 | 公开 **8-step** turbo LoRA 侧车 | LightX2V T2VA 8-step / VDN turbo 哲学 | **无**产品 LoRA 加载器；BF16 LoRA 打 INT8 要么反量化要么另蒸 |
| B2 | 公开 **4-step** LoRA | FastH3 DMD2 / larryvrh v4 | 同上；未蒸馏 `--steps 4` **红线崩** |
| B3 | H3→H3 二采 | Sol 机制模板；本仓可接 FL2VA | 8+8 几乎不省；少步精修须匹配权；8 卡上多一次 VAE 可能更慢 |
| B4 | H3→LTX Super Accel | Sol v2 | **本仓无 LTX / TAE / cute_sm100**；禁止写 5090 秒数 |

#### Task P3.1: CLI 合同（先门，再加载器）

**Files:** Modify `tools/h3_cuda_generate.c` **help 与 preset 解析 only**（P3.1 可以只拒绝、不加载）。

```
--preset gold     → 今日默认（可省略）
--preset fast8    → 预留 8-step turbo；未接加载器时硬错：「非金线；loader not built」
--preset fast4    → 预留 4-step；同上
--preset draft480_refine768 → 预留二采；同上
```

- [ ] **Step 1: `--preset fast4` 无权重 → 非零退出，stderr 含 `非金线` 与 `not implemented`**
- [ ] **Step 2: 不传 preset 时行为与金线完全一致（回归 P0 命令）**
- [ ] **Step 3: help 明确：fast* 不是 MAE 金线；禁止 `--steps 4` 不换权**

**通过：** 金线路径零变化。 **失败：** 把 `--steps` 缺省改成 4。

#### Task P3.2: 选一条快档做侧车（二选一，不要并行两条）

执行时产品选 **一条**：

**选项 α — 4-step LoRA SKU（FastH3 机制）**

- 隔离目录跑公开 LoRA（SGLang / FastVideo），**不要** merge 进 INT8 金线权。
- 本仓若做加载器：独立代码路径 + `--preset fast4`；INT8 骨干上挂 BF16 LoRA 的数值合同必须先写（反量化 vs 拒接）。
- KPI：`out/verify_fast_sku/fast4/REPORT.md` 自建门（人眼 + 有声）。**预期 FAIL 金线 MAE** → 停，不要拧阈值。

**选项 β — 二采 SKU（Sol 机制，H3→H3）**

- 负对照先跑：**480p 8-step + 768p FL2VA 8-step**（预期不赢直出 768p）。
- 少步精修 **必须** 有匹配学生；否则不做 8+2/8+4。
- 音频合同写死：跟草稿 PCM 或跟精修，二选一；Sol 是扔掉 refine 音。
- 禁止把 6.85s 写进报告分子。

- [ ] **Step 1: 选定 α 或 β，写入 `out/verify_fast_sku/CHOICE.md` 三行理由**
- [ ] **Step 2: 按选项跑对照；填独立 REPORT**
- [ ] **Step 3: 金线表（accel_matrix / 66s）一行都不改**

**P3 回滚：** 删 preset 或使其恒硬错；默认 generate 与 P0 相同。

**阶段门：** 有独立 REPORT 且默认未开。质量不过 → SKU 不上产品开关。

---

## 6. 验收与回滚（总表）

| 阶段 | 验收 | 回滚 | 默认变？ |
| --- | --- | --- | --- |
| **P0** | 8×5090 sm_120 构建；单卡 cafe warm×3；`sdpa=cudnn`；有声 5s；DiT ~31s 量级 | 不用远端工作区 | 否 |
| **P1** | `--ulysses 2` 相对单卡 GT：MAE∩有声；DiT 下降；默认 U=1 | 默认 U=1；N=1 不碰 NCCL | 否 |
| **P2** | U=4 有数字（可 FAIL）；U=8 vs U=4 有跨 NUMA 税；TE 不做；VAE 有书面决策 | 失败拓扑不宣称支持 | 否 |
| **P3** | 独立快档 REPORT；help「非金线」；金线 66s 行未改 | preset 硬错或删除 | 否 |

**全局回滚测试（每阶段结束必跑）：**

```sh
CUDA_VISIBLE_DEVICES=0 ./build-cuda-sm120/h3_cuda_generate ... --sdpa-backend cudnn \
  --steps 8 --cfg 1 --seconds 5 --seed 0x11
# 日志：sdpa=cudnn，无 mag/tea/sage/sol，无 ulysses>1
```

**硬停止（执行者）：**

1. 不要改 generate 默认 backend / steps / mag / sage / sol / ulysses。
2. 不要把 4-step LoRA merge 进 INT8 金线。
3. 不要把 22.2× / 4.44× / 12.2× / 14× / 35× / 6.85s 写成 8-step 加速比。
4. 多卡数字只进 `out/verify_5090_sp*`，不改 next-tech-plan / 矩阵的 66s 行。
5. 不要设 `NCCL_P2P_DISABLE`；不要从 TP 起步；不要 U=7。
6. 密码、本机绝对路径默认值不入库。
7. 需要 serving 时调 SGLang 当网关，本仓守 C 引擎。

---

## 7. 明确不做

1. 4-step / Turbo / FastH3 VSA / RH 蒸馏权 **接 INT8 金线**或当无损。
2. 默认开 sage3 / sol_cute / VSA / Cache-DiT / Mag / Tea / FB。
3. TAEH3 / TAEHV / TinyVAE 当金线成片解码。
4. Super Accel 整栈当 5090 默认；把 **6.85s GB200** 写进 5090 承诺。
5. VDN 65GB BF16/FP8 常驻假设；hybrid 未过 MAE 当默认。
6. 把 RH **12.2×** 解释成多卡效率或相对 66s（12.2× 含 50→4 步；多卡效率只引用同配方 54.0→48.2s 的 12%）。
7. 未蒸馏把 `--steps 8` 拧成 4。
8. CFG 并行 / Ring / PP 进金线（cfg=1 无第二分支；Ring 要 FA，与 cuDNN 冲突）。
9. 关 P2P 当无 NVLink 最佳实践。
10. HTTP serving / Python `torch.compile` 替换 C 质量栈。
11. 静默叠刀（少步×稀疏×cache×量化）。
12. 把估时文的 70–110s / 45–75s 标成 **已测**。

---

## 8. 8×5090 上 5s 480p / 768p 预期（引用估时，**全部标未测** 除非注明）

口径：warm generation，不含 load。本仓 **无多卡实现、从未 768p generate** → 下表除 480p 单卡外均为外推。

### 5s · 864×480 · 8-step INT8+cudnn · 有声

| 配置 | e2e | DiT | 性质 |
| --- | ---: | ---: | --- |
| 1×5090（旧机已测） | **~66s**（矩阵 66.06±5.80；剔尖峰 ~62.7） | **31.37±0.05s** | **已测** |
| 1×5090（新 8 卡机单卡） | 同量级，允许 ±15% | ~31–32s | **P0 必测** |
| 2× Ulysses2 | e2e 机制预期 **~45–55s 带**（TE+VAE 几乎不动） | 机制预期 ~20s 量级算赢 | **未测** |
| 4× Ulysses4 | e2e 更接近 TE+VAE 地板 **~35–50s** | 再降，通信税↑ | **未测** |
| 8× Ulysses8 | 地板 **~25–40s** 量级（TE 12–14 + VAE ~10 + 残余 DiT + 跨 NUMA） | 不确定是否优于 U=4 | **未测** |
| 理想 k_eff=8 | e2e **达不到** ~33s 以下很多 | DiT~4s | 下沿，**非承诺** |

### 5s · 1344×768 · 有声（摘 [8x5090-768p-estimate](./2026-09-11-h3c-8x5090-768p-estimate.md)，**未测**）

| 场景 | 理论 e2e | 质量 |
| --- | ---: | --- |
| A 金线外推：8-step INT8+cudnn + Ulysses（不接 4-step/Sage/Cache） | **65–125s**（最可能 **70–110**，中位粗看 ~85） | 保效果；768p MAE **未建** |
| A 理想 k_eff=8 下沿 | ~60–75s | 达不到的下沿 |
| B 快档：4-step LoRA+Sage+Cache+compile | **45–90s**（最可能 **45–75**） | **非金线**；5090 32GB **装不下** RH 60–66 GiB/卡 BF16 |
| RH 28.7s / Super Accel 6.85s | **禁止**写成 8×5090 | 6000D 8 卡 4-step / 2×GB200 |

二采（摘 [two-stage-768p](./2026-09-11-h3c-two-stage-768p.md)，**未测**）：8+8 同权 **不赢**直出；8+2 且匹配权才可能 55–90s；H3→LTX **不能跑**。

---

## 9. 阶段总表（执行摘要）

| 阶段 | 主题 | 约周期 | 关键验收 | 默认变？ |
| --- | --- | --- | --- | --- |
| **P0** | 8×5090 sync、sm_120、单卡金线 warm×3 | 3–5 人天 | DiT ~31s 量级；有声；`sdpa=cudnn` | 否 |
| **P1** | Ulysses2 同轨迹 8-step | 1–2 周 | MAE∩有声 vs 单卡；DiT 下降；默认 U=1 | 否 |
| **P2** | Ulysses4/8；TE 不做；VAE 书面决策 | 1–2 周 | U=4 数字；跨 NUMA 税诚实 | 否 |
| **P3** | 快档 4-step LoRA **或** 二采，独立 KPI | 1–2 周 | 不过门不开默认；66s 行不动 | 否 |

建议顺序：**P0 → P1 → P2**；P3 与 P2 后期可 ‖ 调研，实现不得污染 A。

---

## 10. 执行交接

方案已写入 `docs/superpowers/plans/2026-09-11-h3c-next-opt-plan.md`。

本轮 **只产出方案与交叉链接**；不改 generate 默认、不跑 GPU、不 commit。

**执行选项：**

1. **Subagent-Driven（推荐）** — 每 Task 新 subagent + 两段 review  
2. **Inline Execution** — 本会话按阶段 checkpoint

Which approach?
