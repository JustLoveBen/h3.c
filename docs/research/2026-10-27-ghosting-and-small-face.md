# 虚影（ghosting）与小脸崩（2026-10-27）

> Asia/Singapore。触发：4090 INT8 多场景门禁（`bd330d2`，864×480 · 8-step · cfg1）人眼确认 **马赛克消失，虚影仍在**；并问如何修 **小脸崩**（细节太少导致的异常显示）。
> 成片：`~/grokbot-share/keep_int8_4090_20261027/`；主机 `lan-4090-31`（Ada sm_89）。

## 1. 现象拆分（不要混锅）

| 症状 | 典型画面 | 本轮证据 | 主因假设（可证伪） |
| --- | --- | --- | --- |
| **马赛克** | 块状量化塌缩 | 4090 INT8×5 人眼 OK；NVFP4 KEEP=attn 仍有 | INT8 金线 OK；naive W4A4 权质量（另档） |
| **虚影 / 运动拖影** | 下颌/发丝/快速运动边缘「拖一层」 | `p02_talking_head` t=2.5s 主脸边缘晕；sprint/market 运动更明显 | **少步 Euler 时间离散 + 480p 高频不够**；非 INT8 pack 写坏 |
| **小脸崩** | 远景人脸糊成皮色团、五官糊掉 | 同帧背景人物无五官；主脸纹理偏蜡 | **空间预算**：864×480 下小脸只有极少 latent patch |

## 2. 已排除 / 降权的假因

### 2.1 「4090 naive F32 flash → 虚影」——**主因否决（DiT）**

日志同时出现：

```
[h3] sdpa_f32 cudnn unavailable (...); naive F32 flash (slow)
sdpa_backend=cuda_cudnn_sdpa want=400 hits=400 fallback=0
sdpa_strict: PASS hits==want backend=cuda_cudnn_sdpa
```

- DiT 主路径 **`cuda_cudnn_sdpa` 400/400**，不是 naive flash。
- `sdpa_f32 cudnn unavailable` 是 **F32 计划建图失败**的旁路日志（sm_89 上 cuDNN F32 graph 无 execution plan）；**未**把 DiT 注意力打到 naive F32。
- 因此：**不要**把虚影归因于「4090 没有 cudnn、整段 flash 慢路径」。修 F32 plan 是工程债，不是本轮虚影的第一刀。

### 2.2 「再加步到 16/20」——**质量默认否决**

[`2026-08-21-remaining-quality-speed.md`](../superpowers/plans/2026-08-21-remaining-quality-speed.md)、[`2026-09-11-h3-few-step-quality.md`](../superpowers/plans/2026-09-11-h3-few-step-quality.md)：

- 官方发布权是 **CFG-distilled 8-step**；`--cfg 1` 是合同。
- 有声 **av20**：墙钟↑，**同片不更清**；整段加步会 **拉歪 8-step 蒸馏轨迹**。
- 未匹配学生 LoRA 时把 8→4 = 红线崩主题；反过来 8→16/20 **也不是**「同一条 ODE 更密、必然更清」。

→ A/B 可以跑 steps=16 **作对照**，但 **不得**写成产品修脸/去虚影默认。

### 2.3 MagCache / Tea / 默认 EPI / NVFP4 KEEP

- Mag/Tea 在 8-step 金线上帧 MAE 炸过；不进质量默认。
- EPI_FUSE / NVFP4 KEEP：另档；本轮虚影/小脸与之无关。

## 3. 分因 RCA（带证据）

### 3.1 虚影 ← 少步时间 + 中低分辨率

- 协议固定 **8 Euler + cfg1 + velocity_euler**（金线）。每步是大更新；运动边缘没有长轨迹可抹平（与 Mag FAIL / LightX2V #579 同方向机理）。
- larryvrh / 社区 Turbo：**4 步大运动会 smear**；本仓 8 步仍可能在快动边缘留拖影。
- PDMD 复验笔记：错误 FC1 gate/up 会出 **纵向双影**（实现 bug）；修好后双影消失、**全局发糊**仍在——说明「影」与「糊」可分离。本轮 INT8 无 LoRA，更接近 **时间 smear + 空间糊**，不是 FC1 双影。

**可证伪实验：**

| 臂 | 期望 |
| --- | --- |
| 同 seed **960×544 / 1344×768** · 仍 8-step | 边缘拖影减轻、主脸锐度↑ → 支持「空间预算」 |
| **PDMD-4**（`--fewstep-preset pdmd4` + 原件 LoRA） | 合同正确时 OR/人眼可 PASS；**不**指望单独消灭全部 smear |
| 盲目 `--steps 16` 无新权 | 预期墙钟≈2×、观感未必更好（轨迹拉歪） |

### 3.2 小脸崩 ← 480p 细节饥饿（主）+ 少步高频不足（次）

- 空间：VAE /16 → 864×480 → latent ≈30×54；DiT 2×2 patch → 每帧 ~405 token。远景脸可能只占 **几个 patch** → 五官不可恢复。
- [`2026-08-21-remaining-quality-speed.md`](../superpowers/plans/2026-08-21-remaining-quality-speed.md) Latent Upscaler：二采若只跑少步，**小脸/身份/口型会先碎**；社区要的是 480→720/2K 比原生高分便宜，不是把已 480p 金线再砍。
- [`2026-09-11-h3c-two-stage-768p.md`](../superpowers/plans/2026-09-11-h3c-two-stage-768p.md)：二采可作快档 SKU，**不能**当金线无损开关；同权 8+8 几乎不省。

**可证伪实验：** 抬分辨率（960×544 起）或正式 768p；小脸应先于「加步」恢复可辨五官。

## 4. 排序修复菜单（4090 现在可跑）

约束：保质量优先；**PDMD-4 > DMAD**；无 NVFP4 KEEP、无 EPI default-on、无 convert bypass。

| Rank | 动作 | 目的 | 代价 | 状态 |
| ---: | --- | --- | --- | --- |
| **1** | **抬画布** `960×544` 或金线评测 `1344×768`，仍 `--steps 8 --cfg 1` INT8 | 小脸 + 边缘锐度 | DiT ∝ token；4090 暖 DiT 从 ~27s 升到 ~1.3–4× | **本轮跑 960×544 talking_head** |
| **2** | **PDMD-4** `--fewstep-preset pdmd4 --lora <pdmd_4NFE h3repo>` | 同 NFE=4 质量优于 DMAD（beyond-dmad 表） | 需下载/转换 LoRA；非 8-step 金线 | LoRA 未在 4090；拿到权再跑 |
| **3** | 产品档：**低分草稿 + latent 超分/二采**（Comfy Upscaler / FL2VA 条件图） | 480→高分比直出便宜 | 非无损；二采少步伤小脸 | 本仓无产品级 ×2 latent upsample |
| **4** | `--steps 16` 同 480p **仅作负对照** | 证伪「加步去虚影」 | 墙钟↑、可能拉歪轨迹 | 可选；勿默认 |
| — | 修 sm_89 cuDNN F32 plan | 消旁路日志 / 真 F32 路径 | 与本轮虚影弱相关 | 工程债，另开 |
| — | cfg>1 / MagCache / 未匹配 4-step | — | 过曝 / MAE 炸 / 主题崩 | **禁止**当修脸 |

### 4.1 CLI 模板（4090，`lan-4090-31`）

环境与 [`tools/run_int8_multiscene_4090.sh`](../../tools/run_int8_multiscene_4090.sh) 相同（conda `comfyui`、unset NVFP4/EPI、`H3_SKIP_PPM=1`）。

**A. 抬分辨率（推荐先跑）**

```bash
ROOT=/home/haiyang/workspace/h3_c_main
MODELS=/home/haiyang/workspace/models
BIN=$ROOT/build-cuda/h3_cuda_generate
OUT=$ROOT/out/ghost_fix_ab_20261027/talking_head_960x544_s8
$BIN --dit $MODELS/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
  --vae $MODELS/minimax_h3_video_vae_fp16.safetensors \
  --te $MODELS/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
  --width 960 --height 544 --seconds 5 --steps 8 --cfg 1 \
  --seed 0x76800002 --sdpa-backend cudnn --skip-audio \
  --token-ids-file $ROOT/evals/768p/ids/p02_talking_head.ids \
  --out $OUT --mux $OUT/talking_head_960x544_s8.mp4
```

**B. PDMD-4（有 h3repo LoRA 后）**

```bash
# 仍须原件转换后的 h3repo；preset 绑 steps/cfg/shift/sampler/contract
$BIN ... --fewstep-preset pdmd4 \
  --lora /path/pdmd_4NFE_lora_model_0.h3repo.safetensors --lora-scale 1 \
  --width 864 --height 480 --seconds 5 --seed 0x76800002 \
  --sdpa-backend cudnn --skip-audio \
  --token-ids-file $ROOT/evals/768p/ids/p02_talking_head.ids \
  --out ... --mux ...
# 期望日志：fewstep_preset → fewstep_contract early_ok → sampler=velocity_euler
# 禁与 --preset turbo4|taomate 同开
```

**C. 负对照 steps=16（可选）**

```bash
# 同 864×480，仅 --steps 16；预期未必更清
$BIN ... --width 864 --height 480 --steps 16 --cfg 1 ...
```

## 5. 本轮执行记录

| 项 | 值 |
| --- | --- |
| 基线 | 4090 INT8 864×480×8 · seed `0x76800002` · `bd330d2` · warm dit ≈27.2s · WALL≈52.7s |
| DiT SDPA | 基线与 A/B 均为 `cuda_cudnn_sdpa` hits=400（虚影≠naive flash） |
| A/B | talking_head **960×544 · 8-step** · dit=37.46s · WALL=65.82s · latent std=1.082 · RUN_RC=0 · SHA doc `cd6b1c6` |
| 产物 | 4090 `out/ghost_fix_ab_20261027/`；Mac `~/grokbot-share/ghost_fix_ab_20261027/{talking_head_960x544_s8,baseline_864x480_s8_talking_head}.mp4` |
| 未跑 | PDMD-4（4090 无 LoRA）；1344×768；steps=16 负对照 |
| 人眼 | 待消息列表贴片：预期小脸/背景五官相对 480p 更可辨；运动虚影可能仍在（时间预算未改） |

## 6. 给产品的一句话

- **虚影：** 先当 **8-step + 480p 的时间/空间预算问题**，不要当成 INT8 坏权或 4090 flash 回退。
- **小脸崩：** **加分辨率（或正式二采超分）**，不要先加步；加步在蒸馏金线上不是免费清晰度。
- **要快又要脸：** 优先 **PDMD-4 合同 + 原件 LoRA**（同 4NFE 优于 DMAD），高分场景再叠抬画布 / 二采 SKU。

