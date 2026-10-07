# V0 + V1 实测：sol_cute @ turbo4 稀疏有效性复核 + MMDiT dense 护栏 A/B

> **日期：** 2026-10-05。**机器：** 192.168.0.238（独享），1×RTX 5090 32GB，sm_120，CUDA 13。
> **性质：** 只读实测（仅 env 开关差异），未改核心代码。实验设计依据 `docs/superpowers/plans/2026-10-05-h3-optimization-dev-plan.md` §3.1/§3.2。
> **归档：** `238:/mnt/nvme1n1/workspace/h3_c/out/review-2026-10-05/{v0-solcute,v1-dense-guard}/`（成片 mp4 + generate.log 全文）。

---

## 1. 实验设置

| 维度 | 设定 |
| --- | --- |
| 画布/时长 | 864×480 · 5s · 24fps · turbo4（4-step 蒸馏） |
| seed / ids | `--seed 0x48000001` · `evals/768p/ids/p01_sprint.ids`（te-cache key=`ea5f4c51c57276b2`，clip2 te=0 热命中） |
| binary | `238:/mnt/nvme1n1/workspace/h3_c/build-cuda-sm120/h3_cuda_generate`（同 binary 全臂） |
| 权重 | DiT `minimax_h3_fl2va_pruned_int8_convrot.safetensors`；VAE `minimax_h3_video_vae_fp16.safetensors`；TE `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`；audio `minimax_h3_audio_vae_fp32.safetensors`；LoRA `minimax_h3_turbo_v4_step600_ema.safetensors` |
| 复现环境 | `H3_VAE_PINNED_D2H=1`（金线复现必需，OFF 路径有既有 `missing rgb` bug）；sol 臂需 conda py3.11 `PATH/LD_LIBRARY_PATH/PYTHONHOME` + `PYTHONPATH=<repo>/tools`（否则 `import h3_sol_cute_embed` 失败全 fallback——本次踩到并修复，失败现场归档 `v0-solcute/sol-failed-embed/`） |
| 每臂 | `--repeat 2`（clip1 冷 / clip2 热），双臂 waffle 读 hot clip2 五帧 |

### 臂定义（每拍只动一个变量）

| 臂 | sdpa | `H3_SOL_FIRST_DENSE_STEPS` | `H3_SOL_FIRST_DENSE_LAYERS` | 备注 |
| --- | --- | --- | --- | --- |
| V0-对照 dense | cudnn | — | — | 全精确注意力 |
| V0-实验 sol | sol_cute | 0 | 0 | 现产品默认接线（全稀疏+sink） |
| V1-a | sol_cute | 0 | **2** | 官方「前 2 层 dense」护栏 |
| V1-c | sol_cute | **1** | 0 | turbo4 等效「首步 dense」（官方前 10 步 dense 在 4-step 下的等效项） |
| V1-ac | sol_cute | 1 | 2 | 两护栏叠加 |

V1 各臂需 `H3_SDPA_STRICT=0`：dense 护栏的 layer/step 走 dense 属**设计行为**（`dense_skip` 计数），默认 strict 把 `mixed` 判 FAIL 中止（首跑 V1-a rc=1 即此原因；`h3_kpi_guard.c` 的 verdict 只把 `hits==want` 当 PASS）。这本身是**默认 strict 与官方 dense 护栏语义不兼容**的既有冲突，记入阻塞项。

### 执行命令（模板，各臂仅末行开关不同）

```bash
cd /mnt/nvme1n1/workspace/h3_c
H3_VAE_PINNED_D2H=1 H3_SOL_ATTN_ROOT=/mnt/nvme1n1/workspace/Sana-sol-engine/techniques/sparse_backends \
H3_SOL_CUTE_LIB=$PWD/build-cuda-sm120/libh3_sol_cute.so \
H3_SOL_FIRST_DENSE_STEPS=<0|1> H3_SOL_FIRST_DENSE_LAYERS=<0|2> \
./build-cuda-sm120/h3_cuda_generate --preset turbo4 \
  --dit <fl2va_int8_convrot> --vae <official_fp16> --te <qwen3vl_nvfp4_awq> \
  --audio-vae <audio_fp32> --lora <turbo_v4_step600_ema> \
  --width 864 --height 480 --seconds 5 --seed 0x48000001 --repeat 2 \
  --te-cache out/te_cache_kpi_a --token-ids-file evals/768p/ids/p01_sprint.ids \
  --vae-decode official --sdpa-backend <cudnn|sol_cute>
```

运行器脚本：`238:/tmp/h3_v0v1_run.sh`（臂参数 v0-dense / v0-sol / v1-a / v1-c / v1-ac）。

---

## 2. 原始数字（hot clip2，五帧 waffle b16 全列）

| 臂 | waffle f0/24/48/90/119 | **waffle 均值** | latent checksum clip2 | phase_ms total (hot) | dit (hot) | sol hits | av_sync |
| --- | --- | ---: | --- | ---: | ---: | --- | --- |
| **V0-dense** | 0.774 / 0.819 / 0.851 / 0.855 / 0.944 | **0.849** | `0x22d534015b468279` | 22556.0 | 14483.7 | cudnn 400/400 fallback=0 | PASS 8ms |
| **V0-sol** | 1.672 / 1.096 / 1.087 / 1.042 / 1.710 | **1.321** | `0xce0b5df2bcdb4ad3` ✅金线 | 16768.6 | 10339.4 | 400/400 fallback=0（cute_sm120） | PASS 8ms |
| **V1-a** (layers=2) | 0.817 / 0.797 / 0.776 / 0.862 / 0.814 | **0.813** | `0x730a430a2d33cabe` | 17013.7 | 10624.3 | 384/400 dense_skip=16 fallback=0 | PASS 8ms |
| **V1-c** (steps=1) | 0.801 / 0.804 / 0.817 / 0.825 / 0.874 | **0.824** | `0xb5ce0777764d4d30` | 17877.9 | 11494.9 | 300/400 dense_skip=100 fallback=0 | PASS 8ms |
| **V1-ac** (1+2) | 0.736 / 0.821 / 0.840 / 0.783 / 0.912 | **0.818** | `0xb2a11c888bf399cb` | 17989.1 | 11619.6 | 288/400 dense_skip=112 fallback=0 | PASS 8ms |

冷 clip1 数据（全臂完整）：V0-dense 25965.3ms / waffle 均值 1.404 / checksum `0x9375bd6a5482fd4e`；V0-sol 25935.4ms / 1.893 / `0x28e8a880d52e9be1` ✅金线；V1-a 26082.2ms / 2.669 / `0x06584abf8f72c8d0`；V1-c 27072.6ms / 0.758 / `0x2c0edcba987cd58c`；V1-ac 25038.8ms / **latent_stats 全 NaN**（waffle 0.000，成片为废片；hot clip2 恢复正常 0.818）。

### 复现闭环验证

V0-sol 臂逐位复现 2026-10-04 金线：clip1=`0x28e8a880d52e9be1`、clip2=`0xce0b5df2bcdb4ad3`，且 waffle 逐帧与 `out/review-2026-10-04/kpia-sol-480p` 完全一致（1.672/1.096/1.087/1.042/1.710）——环境/命令/权重复现闭环 ✅，A/B 对比可信。

---

## 3. V0 判定：**退化**（明确）

| 判据（dev-plan §3.1） | 结果 | 判定 |
| --- | --- | --- |
| 实验臂 waffle 均值超对照臂 >0.15 | 1.321 − 0.849 = **+0.472** | **触发退化线** |
| 任一 clip 出金线区间（1.0–1.7）且人眼可见 | sol 臂 f0=1.672 贴边、f119=1.710 出区间上沿；dense 臂全帧 ≤0.944 | 触发（边缘帧明显劣化） |
| av_sync 劣化 | 双臂均 PASS 8ms | 未触发 |

**结论：sol_cute @ turbo4 全稀疏（现产品默认接线）相对全 dense 判退化。** waffle 均值超对照 0.47，超判定线 3 倍；劣化集中在首尾帧（f0 +0.90、f119 +0.77）。FastWan「训练无关稀疏在 <10 步蒸馏下急剧退化」的社区论点在本仓 480P turbo4 栈上**实测成立**。

**补充口径说明：** dense 对照臂 waffle（0.849）与 10-04 历史 cudnn 产品臂（0.750–0.947）一致；sol 臂 waffle（1.321 均值）与历史 sol 臂逐帧一致——退化对比不是本次环境噪声。sol 臂 checksum「逐位复现金线」恰恰说明**现行金线本身是 sol 臂产物**——金线 checksum 一致只证明 sol 可复现自己，不证明它相对 dense 无损；本 A/B 首次补上 dense 对照，坐实退化。

---

## 4. V1 判定：**护栏有效，且可完全救回退化**

| 臂 | waffle 均值 | vs V0-sol (1.321) | vs V0-dense (0.849) | hot total 代价 |
| --- | ---: | ---: | ---: | ---: |
| V1-a（前 2 层 dense） | **0.813** | **−0.508** | **−0.036（略优于 dense）** | +245ms（16.77→17.01s） |
| V1-c（首步 dense） | **0.824** | **−0.497** | **−0.025** | +1109ms（→17.88s） |
| V1-ac（叠加） | **0.818** | **−0.503** | **−0.031** | +1221ms（→17.99s） |

- **三护栏臂全部把 waffle 拉回 dense 对照水平甚至略优**（0.81–0.82 vs 0.849），官方「近无损」背书所需的 dense 护栏（前 2 层 dense / 首步 dense）在本仓实测有效——之前 `FIRST_DENSE_*=0` 全稀疏只用了官方接线的一半，这正是 V0 退化的直接根因。
- **代价：** V1-a 几乎零速度代价（+245ms，hot total 17.01s 仍远快于 dense 22.56s）；V1-c 代价 +1.1s（首步全 dense 时该步 50 层全走 cudnn）。
- **V1-ac 发现缺陷：** cold clip1 latent 全 NaN（waffle=0 废片），hot clip2 正常。叠加臂在冷启动路径有数值 bug（dense_steps=1 + dense_layers=2 组合下首 clip 某处产出 NaN），**V1-ac 配置在修复前不可用**；单独 V1-a、V1-c 无此问题。
- 文本 query 行 dense 开关：**不存在**（`h3_gpu_cuda.cu` 只有 sink=988 K/V 侧 + `FIRST_DENSE_STEPS/LAYERS` 两个读取点；官方 `make_mmdit_sol_attn_dispatch` 的 query 行 dense 未接线）——记为「需编码补齐」，但 V1-a 已证无它亦可救回质量，补齐优先级降低。

---

## 5. 分支建议（按 dev-plan §3 预案）

1. **V0 判退化 → 立即动作：** sol_cute 以 `FIRST_DENSE_*=0` 全稀疏形态**不得留在 480P turbo4 默认产品路径**。默认路径应回退 `--sdpa-backend cudnn`，或——
2. **采纳 V1-a 为 sol 新默认候选：** `H3_SOL_FIRST_DENSE_LAYERS=2`（+0.245s 代价）在 480P 实测 waffle 0.813 优于 dense 对照 0.849，hot total 17.01s 仍比 dense 快 5.5s。建议：默认栈改为 turbo4 + INT8 + sol_cute + `FIRST_DENSE_LAYERS=2`，保留 480P 的速度红利同时消除退化。落地前需按惯例跑 repeat≥2 稳定性门 + 人眼抽帧确认首尾帧。
3. **V1-ac 禁用直至 NaN 修复：** 叠加臂 cold-clip NaN 立案（疑似 dense_steps×dense_layers 组合在首 clip 的数值路径 bug）。
4. **V4 提级：** 按 dev-plan §3.4 条件分支，V0 判退化 → SLA 式稀疏感知微调调研从低优先升为正式立项。
5. **sdpa_strict 与 dense 护栏的语义冲突**需编码解决：护栏臂 hits<want 是设计行为，默认 strict=1 会把任何带护栏的 sol 配置判 FAIL 中止——这决定 `FIRST_DENSE_LAYERS=2` 能否进默认路径（建议 verdict 增加「dense_skip 计入预期」口径，而非依赖 `H3_SDPA_STRICT=0` 全局放宽）。
6. **金线口径修订：** 现行 480P 金线 checksum 是 sol 全稀疏臂产物（V0 判退化的臂）。T2 档「bit-identical 金线」口径需重定——以 dense 臂（checksum clip2=`0x22d534015b468279`）还是 V1-a 护栏臂为新金线，需目标书层面决策（本代理不替决策，只供数）。

## 6. 阻塞/环境备忘

- **sol 开关准确名称与用法：** `--sdpa-backend sol_cute`（CLI）/ `H3_CUDA_SDPA_BACKEND=sol_cute`（env）；护栏 `H3_SOL_FIRST_DENSE_STEPS`、`H3_SOL_FIRST_DENSE_LAYERS`（注意：**缺省值=2**，全稀疏必须显式设 0）；`H3_SOL_SINK_TOKENS`（默认 988=文本 K/V sink）、`H3_SOL_TAU`。护栏臂需 `H3_SDPA_STRICT=0`。
- **sol 插件环境：** 需 conda py3.11 完整环境（PATH+LD_LIBRARY_PATH+PYTHONHOME）+ `PYTHONPATH` 含 `<repo>/tools`；缺则 `import h3_sol_cute_embed` 失败、全 fallback 且 strict FAIL。本次第一轮 sol 臂即因此作废（归档 `sol-failed-embed/`）。
- **复现基线必须 `H3_VAE_PINNED_D2H=1`**（OFF 路径 `missing rgb` bug 仍未修，10-04 已登记）。
- 238 ssh：`yongbin@192.168.0.238`（密码会话注入 `H3_SSH_PASS_5090`，不入库）。
