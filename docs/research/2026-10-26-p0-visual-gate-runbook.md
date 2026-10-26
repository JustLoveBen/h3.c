# P0 视觉门 Runbook（INT8 金线纪律 + NVFP4 P2.2 重验）

> **日期：** 2026-10-26。  
> **上游：** [`2026-10-26-followup-dev-plan.md`](2026-10-26-followup-dev-plan.md) P0；[`2026-10-09-dev-plan-and-progress.md`](2026-10-09-dev-plan-and-progress.md)；P2.2 协议见 [`2026-10-07-nvfp4-w4a4-path.md`](2026-10-07-nvfp4-w4a4-path.md) §P2.2（旧片子作废）。  
> **性质：** 可执行对照模板 + 命令骨架。投产门 = **目检 + 同配置 waffle**，不是 cosine / SSIM / bit-identical。  
> **授权边界：** 本 runbook **不** push origin、**不**清盘、**不**开 nsys/EVT、**不**提交链式 WIP。

---

## 0. 哪台机器干什么

| 角色 | 机器 | 路径 | 能做 / 不能做 |
| --- | --- | --- | --- |
| 文档与编排 | Mac mini `DigenMans-Mac-mini.local` | `/Volumes/Ben/Digen/code/h3.c` | 写 docs、改源码、`git commit`；**无 sm_120，不能跑 `h3_cuda_generate`** |
| CUDA 生成（唯一执行机） | `yongbin@192.168.0.238`，RTX 5090 sm_120 | 仓库 `/mnt/nvme1n1/workspace/h3_c_p4b`；模型 `/mnt/nvme1n1/workspace/models/`；输出建议 `.../h3_c_p4b/out/p0_visual_20261026/` | 跑 INT8 / NVFP4 对照；盘紧（nvme1n1 ~99%），**禁止**未授权清理或写大 trace |
| Cursor 已注册机 | Mac mini、MacBook Air；**238 未注册为 Cursor machine** | — | 在 238 上跑需 SSH / 人工终端；本 runbook 命令假定已 `ssh yongbin@192.168.0.238` |

同步提示（人工）：把 Mac mini `main`（含 `321538a` graph 默认关 + 之后 docs）同步到 `h3_c_p4b` 后再编或确认二进制已含 graph-off 默认。未设 `H3_SDPA_CUDNN_CUDA_GRAPH` 时日志须有 `sdpa_graph: wanted=0`。

---

## 1. 对照纪律模板（每次跑都填）

复制到该次 `OUT/META.txt`：

```
date_local: 2026-10-26 Asia/Singapore
host: yongbin@192.168.0.238
repo: /mnt/nvme1n1/workspace/h3_c_p4b
binary: ./build-cuda-sm120/h3_cuda_generate
git_sha: <fill>
arm: INT8 | NVFP4   # 二选一；禁止同进程混臂
H3_DIT_NVFP4: unset | 1
H3_DIT_NVFP4_FILE: <path or n/a>
H3_DIT_NVFP4_CHAIN: MUST unset
H3_SDPA_CUDNN_CUDA_GRAPH: MUST unset (wanted=0)
H3_TEXT_H_LOAD: unset (live TE) | <path>   # 视觉门默认 live TE；速度场诊断才用 refine dump
H3_SKIP_PPM: 1
H3_DIAG_*: unset
seed: 0x76800002
token_ids: evals/768p/ids/p01_sprint.ids
prompt_id: p01_sprint
width_height: 864x480   # P0 首轮；768p 另开目录勿混
seconds: 5
steps: 8
cfg: 1
sdpa: cudnn
skip_audio: yes
cold_start: yes|no   # 本批该臂第几次新进程
out_dir: ...
```

臂隔离硬规则：

1. INT8 与 NVFP4 **各一次独立进程**；不要在同一 shell 里残留 `H3_DIT_NVFP4*`。  
2. **禁止** `H3_DIT_NVFP4_CHAIN`（链式是 P3，视觉门前不开）。  
3. **禁止**打开 SDPA CUDA graph 当默认；临时 `H3_SDPA_CUDNN_CUDA_GRAPH=1` 必须写进 META 标题，且该次不得进金线 waffle。  
4. 墙钟只追加 [`2026-10-09-generation-timing.md`](2026-10-09-generation-timing.md)；**不得**用墙钟替代视觉门。  
5. 旧 P2.2 片子（graph-on、`seed 0x76800001`、2026-10-07）**作废**，不得当本轮通过证据。

---

## 2. 环境与路径（238）

```bash
cd /mnt/nvme1n1/workspace/h3_c_p4b
BIN=./build-cuda-sm120/h3_cuda_generate
MODELS=/mnt/nvme1n1/workspace/models
DIT=$MODELS/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
VAE=$MODELS/vae/minimax_h3_video_vae_fp16.safetensors
TE=$MODELS/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
# 视觉门 --skip-audio：可不挂 audio VAE
NVFP4_SIDE=$MODELS/diffusion_models/minimax_h3_fl2va_nvfp4_w4a4.safetensors
IDS=evals/768p/ids/p01_sprint.ids
OUT_ROOT=out/p0_visual_20261026
mkdir -p "$OUT_ROOT"
```

公共 unset（每臂开跑前执行）：

```bash
unset H3_DIT_NVFP4 H3_DIT_NVFP4_FILE H3_DIT_NVFP4_CHAIN \
      H3_DIT_NVFP4_CHAIN_FC1_F32 H3_DIT_NVFP4_CHAIN_DIAG \
      H3_SDPA_CUDNN_CUDA_GRAPH H3_TEXT_H_LOAD H3_DIAG_W_CKSUM H3_DIAG_BLK0
export H3_SKIP_PPM=1
```

---

## 3. 命令骨架

### 3.1 Phase A — INT8 金线（864×480，8-step，live TE）

```bash
ARM=int8_1   # 同臂至少再跑 int8_2；第一次常算冷启动
OUT=$OUT_ROOT/$ARM
mkdir -p "$OUT"
# 填 META.txt 后：
"$BIN" \
  --dit "$DIT" --vae "$VAE" --te "$TE" \
  --width 864 --height 480 --seconds 5 --steps 8 --cfg 1 \
  --seed 0x76800002 --sdpa-backend cudnn --skip-audio \
  --token-ids-file "$IDS" \
  --out "$OUT" --mux "$OUT/p01_sprint.mp4" \
  2>&1 | tee "$OUT/generate.log"
echo RUN_RC=${PIPESTATUS[0]} | tee -a "$OUT/generate.log"
```

日志必须出现：

- `dit_format: INT8 ConvRot`（无 `dit_nvfp4: H3_DIT_NVFP4=1`）  
- `sdpa_graph: wanted=0`  
- `sigma[0..8] video:` …（默认 video shift 12）  
- `adaln_dual_sigma: video_shift=12` …  
- `phase_ms:` …  
- `RUN_RC=0`（或进程退出码 0）  
- 写出 mux mp4

离线核验（可在 Mac mini 上对拉回的 log）：

```bash
python3 scripts/p0_protocol_check_log.py --arm int8 "$OUT/generate.log"
```

### 3.2 Phase B — NVFP4 实验臂（同协议，chain 关）

```bash
ARM=nvfp4_1
OUT=$OUT_ROOT/$ARM
mkdir -p "$OUT"
unset H3_DIT_NVFP4_CHAIN
export H3_DIT_NVFP4=1
export H3_DIT_NVFP4_FILE="$NVFP4_SIDE"
"$BIN" \
  --dit "$DIT" --vae "$VAE" --te "$TE" \
  --width 864 --height 480 --seconds 5 --steps 8 --cfg 1 \
  --seed 0x76800002 --sdpa-backend cudnn --skip-audio \
  --token-ids-file "$IDS" \
  --out "$OUT" --mux "$OUT/p01_sprint.mp4" \
  2>&1 | tee "$OUT/generate.log"
echo RUN_RC=${PIPESTATUS[0]} | tee -a "$OUT/generate.log"
unset H3_DIT_NVFP4 H3_DIT_NVFP4_FILE
python3 scripts/p0_protocol_check_log.py --arm nvfp4 "$OUT/generate.log"
```

日志必须出现：`dit_nvfp4: H3_DIT_NVFP4=1`、`dit_nvfp4: sidecar=...`、**无** `H3_DIT_NVFP4_CHAIN`、`sdpa_graph: wanted=0`。  
注意：基座仍是 INT8 checkpoint 时，可能仍打印 `dit_format: INT8 ConvRot`；**臂身份以 `dit_nvfp4:` 行为准**（与 timing §6 一致）。

### 3.3（可选）1344×768 同形

仅在 864×480 目检有结论、或需要与历史 P2.2 **形状**对齐时另开目录 `out/p0_visual_20261026_768/`。把 `--width 1344 --height 768` 换上，其余协议不变。仍用 `seed 0x76800002`（不要用旧 G1/G2 的 `0x76800001` 混进本轮金线）。768P 上 `waffle_guard INVALID_LOWENERGY` 已知可能误报（见 P3.1 笔记）——**不得**单靠该 WARN 判 NVFP4 失败；以人审 + 480p waffle 区间为主。

### 3.4 不在本 runbook 内的命令

- `H3_TEXT_H_LOAD=.../c_te_refined.f32`：仅速度场 / 1-step md5 诊断，**不是**本轮视觉门默认。  
- `H3_DIT_NVFP4_CHAIN=1`：链式 P3。  
- `--preset turbo4` / 任意 LoRA：少步臂，正交验收。  
- 1-step only：可作冒烟，**不能**单独关闭视觉门。

---

## 4. 通过 / 失败标准

### 4.1 协议门（机器可检）

对每个 `generate.log` 跑 `scripts/p0_protocol_check_log.py`；失败则整次作废重跑，不进入目检。

### 4.2 视觉门（人审，决定性）

并排看 INT8 vs NVFP4 的 mux（建议抽 frame 0 / mid / last）：

| 观察 | 通过倾向 | 失败倾向 |
| --- | --- | --- |
| 结构、主体、透视、无整帧噪声/色块/崩坏 | 可用 | 结构坍塌、纯噪、严重伪影 |
| 与 INT8 **构图不同** | **允许**（混沌轨迹；P2.2 原结论） | — |
| 时域闪烁 / 帧间炸裂 | — | 失败 |
| latent 非有限 / VAE 未写出帧 | — | 失败 |

**不要求：** bit-identical、cosine>0.99、SSIM 高、与 2026-10-07 旧片一致。

结论三选一，写入进展文档（勿只写在聊天里）：

1. **通过（隔离臂可用）** — 仍不自动升默认  
2. **不可用** — 记失败模式，停在核修复前  
3. **需改核** — 列出怀疑层 / 下一步实验，不提交链式

### 4.3 Waffle（同配置）

- **864×480：** 参考金线区间约 **0.75–0.95**（roadmap / 既有 480p 标定）。同臂冷热、INT8 与 NVFP4 **分别**记；禁止跨臂用 waffle 差值当唯一质量门。  
- **768×1344：** block_ratio 门已知对 AI 768p 误报；人审优先，waffle 只作同配置回归备忘。  
- 若 INT8 金线自己也全面 `INVALID_LOWENERGY`，记「门失效」而非「NVFP4 独有失败」。

---

## 5. 明确不要做

- 打开 graph / chain 当本轮默认。  
- 用旧 P2.2 graph-on 片子或 SSIM 0.1055 当通过。  
- 把链式 §12 cosine/rel 当投产完成标准。  
- 视觉未过就把 NVFP4 升默认或并入 golden。  
- 首轮把 PDMD/DMAD LoRA 叠进 NVFP4。  
- 未授权清盘、nsys、push、提交 `h3_dit_nvfp4_chain.cu` 等脏链式源码。  
- 在 Mac mini 上幻想能跑通 CUDA generate。

---

## 6. 跑完后写回哪里

1. 目检结论 → 追加 [`2026-10-09-dev-plan-and-progress.md`](2026-10-09-dev-plan-and-progress.md)「进展」或新开 `2026-10-26-p0-visual-gate-results.md`。  
2. 墙钟 → 只追加 [`2026-10-09-generation-timing.md`](2026-10-09-generation-timing.md)。  
3. 勾 [`2026-10-26-followup-dev-plan.md`](2026-10-26-followup-dev-plan.md) §6 第 1–2 项。  
4. 再请示：是否 push、是否动链式、是否清盘。

---

## 7. 建议执行顺序（238 上）

1. 确认 `git log -1` / 二进制与 graph-off 一致；`df -h` 确认还能写 mp4（盘满则停，请示）。  
2. INT8 `int8_1` → 协议检查 → 保留 mp4。  
3. INT8 `int8_2`（非首进程冷 TE 对照；目检可用二次或两次都看）。  
4. NVFP4 `nvfp4_1` → 协议检查 → 与 INT8 并排目检。  
5. 需要稳定性再跑 `nvfp4_2`。  
6. 写结论；可选再开 768 目录。

**下一条在 238 上应敲的第一句（在 §2 变量设好之后）：** Phase A 的 INT8 命令（`ARM=int8_1`）。
