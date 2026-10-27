# Face LoRA vs FaceRefine for distant-face mush（2026-10-28）

> Asia/Singapore. Trigger: FaceRefine（crop→F2VA→stitch）已判弱并 STOP；用户问能否加/训 LoRA 修远脸崩。
> 产品臂：DMAD / Euler。本注**分析优先**，未占 4090 长跑。

## 0. 一句话

**LoRA 不是远脸崩的第一杠杆。** 主因是 **头在画面里的像素预算**（~34px → VAE/16 → ~1 个 DiT patch），蒸馏 LoRA（PDMD/DMAD）只改轨迹与 NFE，**不发明**不存在的空间 token。优先抬画布 / 二采超分 SKU；身份类 LoRA 与「低 face-px 细节恢复」是两件不同的事。FaceRefine 停产正确。训脸细节 LoRA：**暂 no-go 全训**；可做**零训练冒烟**（已有 refiner 槽 / 抬画布对照）。

## 1. 硬证据（仓内，禁止编造指标）

### 1.1 RCA（FaceRefine / ghost 笔记一致）

| 来源 | 结论 |
| --- | --- |
| [`2026-10-27-facerefine-p10.md`](2026-10-27-facerefine-p10.md) | p10 f000 YuNet face ≈34px；f119 MCU ≈173px。崩 = 空间预算，非 RefMod |
| 同文 Steps A/B | face-CU 544 F2VA：PDMD-4 / Euler-8 / Euler-16；**NFE 非主限**；宽景下采样回 ~34px 吃掉 HF；Euler 更严重 framing drift（bobblehead） |
| [`2026-10-27-ghosting-and-small-face.md`](2026-10-27-ghosting-and-small-face.md) | 小脸：抬画布优先于加步；1344 PDMD window≈6 **FAIL 噪声**；1344 Euler-8 PASS 脸最好；PDMD 可用高分上限约 **1152×640** |
| FaceRefine META | H3 路径 warm ≈76.5s；F2VA regen ≠ pixel-locked img2img；classical Lanczos 弱 |

### 1.2 同构图相对脸高 → 绝对 face-px（估算，非实测）

相对脸高取 p10 宽景 `34/480`：

| 画布 | 估算 face 边长 | 估算 face DiT patches | 帧 token |
| --- | ---: | ---: | ---: |
| 864×480 | ~34 | ~1.1 | 405 |
| 960×544 | ~39 | ~1.5 | 510 |
| 1152×640 | ~45 | ~2.0 | 720 |
| 1344×768 | ~54 | ~2.9 | 1008 |

抬画布在**同一构图**下增加 face patches；若构图故意把脸钉死在 34px（推远），抬画布无效——那是提示/分镜问题。

### 1.3 仓内已有 LoRA = 少步蒸馏，不是脸细节

| 资产 | 用途 |
| --- | --- |
| PDMD-4 / DMAD-4 学生 LoRA + convert 工具 | 4NFE 合同；[`2026-10-09-pdmd-dmad-loras.md`](2026-10-09-pdmd-dmad-loras.md) |
| TaoMate-H3 3-step LoRA | 流式少步 SKU，非远脸 |
| VDN 训练栈 `third_party/.../vdn-minimax-h3` | 混合注意力 + DMD turbo，非 face-SR |
| `H3_REFINER_LORA` + `STEPS` + `SCALE` | **末 N 步切换**第二套 LoRA（C035 细节/refiner 想法已接线） |
| CLI | 单路 `--lora`；无「仅 face token 上 LoRA」 |

社区 musubi-tuner **角色 LoRA**（静止画面部追训）修的是 **identity 像不像**，证据域 ≠「远景 34px 五官可辨」。

## 2. 问题 1：LoRA 是不是对杠杆？

相对候选排序（远脸保真，480p 产品门禁语境）：

| Rank | 杠杆 | 与证据关系 | 相对 LoRA |
| ---: | --- | --- | --- |
| **1** | **抬画布**（PDMD≤1152；Euler@1344） | 仓内 A/B 已证脸预算↑ | LoRA 不能替代 |
| **2** | **低分草稿 + latent 超分/二采**（SelfLift / Comfy Minimax latent Upscaler / FL2VA 条件） | 计划文档：快档 SKU，非金线无损 | 比脸 LoRA 更贴近「补空间」 |
| **3** | 更好少步合同（PDMD≥DMAD） | beyond-dmad 表；不解决 patch 饥饿 | 正交（速度） |
| **4** | 更长 NFE / 非蒸馏轨迹 | FaceRefine 与 ghost 笔记：**加步不是免费清晰度** | 弱 |
| **5** | Identity / Ref2VA 适配 | 修「是谁」，不修「多小」 | 不同问题 |
| **6** | **脸细节 / 少步脸恢复 LoRA** | 无仓内权重；理论可幻觉 HF，数据贵、易糊脸/风格渗漏 | **次级、投机** |
| **STOP** | FaceRefine crop→F2VA→stitch | 76s+、halo、身份漂、步数无增益 | 已停 |

**结论：** LoRA **可以是辅助**（身份稳定、或末步细节恢复），**不是**远脸崩的主修。主修仍是 **空间预算**（抬画布或二采）。

## 3. 问题 2：LoRA 目标选哪个？

| 目标 | 含义 | 对本问题 | 建议 |
| --- | --- | --- | --- |
| (a) 低 face-px 高细节 | 从极少 patch 幻觉五官 | 直接对题，但最难；需成对高低分/区域加权 | 仅研究臂 |
| (b) 身份一致性 | 角色 LoRA / Ref2VA | 修漂移，不修 34px mush | 产品另开（长镜头保险） |
| (c) 反虚影时间 | 运动 smear | ghost 笔记：少步+分辨率；非 LoRA 主因 | 否 |
| (d) 少步脸质量恢复 | turbo/DMAD 过饱和后末步细节 | 对齐 **C035 + `H3_REFINER_LORA`**；仓内已有接线 | **若做 LoRA，优先 (d)** |

推荐：**不训 (a) 直到抬画布 ROI 不够**；若做适配器，先试 **(d) 末步 refiner**，再考虑角色 **(b)**。

## 4. 问题 3：数据配方草图（仅在 go 训时）

**不要**用「好看静物棚拍」当唯一数据（用户已知 stock ≠ H3 视频）。

1. **成对（首选 (a)/(d)）：** 同 prompt/seed 的 **教师**（Euler-8 @1152/1344 或 50-step 若可得）vs **学生**（DMAD/PDMD @864×480）；蒸馏或 MSE on latent；可选 face-bbox 区域加权（YuNet → latent 坐标）。
2. **多尺度桶：** 宽景小脸 + MCU 大脸 + push-in（p10 类 dolly）；避免只训大脸导致远景无效。
3. **运动：** 说话/挥手/走步，防静帧过拟合。
4. **课程：** 先大脸（学纹理）→ 混入小脸（学在少 patch 下保持结构）；学习率递减（类 LTX 分辨率链）。
5. **秩：** rank 32–64 若要与蒸馏 LoRA 共存；全 128 易与 PDMD/DMAD 冲突。
6. **栈：** 现有 convert（`convert_*_lora_h3repo.py`）+ `h3_lora` 合同；训练壳可参考 musubi-tuner / VDN 脚本，但**目标损失与数据不同**。
7. **代价量级：** PDMD 论文配方 ~2×8 H100、千级 iter；单卡 4090 musubi 角色 LoRA ~3h/1k step 是 **identity 静帧**，不是视频成对超分。**(a) 全训不便宜。**

## 5. 问题 4：推理期「face_px < 阈值才挂 LoRA」？

| 策略 | 缝合伪影 | 可行性（本仓） |
| --- | --- | --- |
| 整段挂脸 LoRA（预检 face_px） | **无** stitch | 可行；但改**整段**风格/身份，非仅脸 |
| 仅 face token 上 LoRA | 无 | **无**区域 LoRA；需改 DiT 掩码 |
| crop→独立生成→stitch | **有**（FaceRefine 已见） | 已 STOP |
| `H3_REFINER_LORA` 末 N 步 | 无 stitch | **已实现**；全局末步，适合 (d) |

**不要**为了条件挂载重新引入 stitch。阈值策略最多做 **整段开关** 或 **末步 refiner**。

## 6. 问题 5：ROI 对照

| 方案 | 证据强度 | 额外墙钟（量级） | 远脸收益 | 伪影风险 |
| --- | --- | ---: | --- | --- |
| FaceRefine（已停） | 实测弱 | ~+76s F2VA | 宽景部分↑ | halo / bobblehead / 身份漂 |
| Euler @1344 | 实测强 | ~143s e2e（C040 量级） | **最好** | 低（原生） |
| DMAD/PDMD @1152 | 实测可用 | ~74s 级（ghost 表） | 明显优于 480 | 低；1344+PDMD 禁 |
| 训 (a) 脸细节 LoRA | **无** | 训：天–周级；推理≈0（若 fuse） | 投机 | 糊脸/渗漏/与蒸馏冲突 |
| 冒烟 (d) refiner LoRA | 接线有、权重或缺 | 切换开销小 | 未知 | 低于 stitch |

**相对 FaceRefine：** 抬画布 ROI ≫ stitch；训 LoRA ROI 未知且前置成本高。  
**相对 Euler@高分：** 高分已是已知最强远脸；LoRA 只有在 **必须锁 480p 墙钟** 时才值得赌。

## 7. Go / No-Go

| 决策 | 内容 |
| --- | --- |
| **No-Go（主路径）** | 用「训一个脸 LoRA」替代 FaceRefine / 替代抬画布 |
| **No-Go（近期）** | 全量 (a) 成对训练（无现成数据配方、占 4090、与 C041 争卡） |
| **Keep STOP** | FaceRefine crop→F2VA→stitch |
| **Go（产品）** | 远脸场景默认抬画布：DMAD/PDMD ≤1152；要最大脸预算用 Euler@1344；480p 门禁只作速度金线 |
| **Go（最小验证，零/近零训）** | 见 §8 |
| **Maybe later** | 若产品硬锁 480p 且抬画布不可接受 → 再开 (d) 末步细节 LoRA 数据臂 |

## 8. 最小验证实验（先于任何全训）

**不占长 GPU；可与 C041 错峰。**

1. **复用已有片（零算力）：** 对照 `ghost_fix_ab` / `facerefine_p10` 成片：同 p10 构图下 480 PDMD vs 1152 PDMD vs 1344 Euler；记宽景 face-px（YuNet）与主观五官可辨——确认「构图缩放」假设。
2. **零训接线冒烟（若权重在 models/）：** 480p p10 + 产品臂 LoRA，设 `H3_REFINER_LORA`（TaoMate 或任一细节适配器）`STEPS=1~2`；人眼看远脸是否优于裸 DMAD。**无权重则跳过，不下载长跑。**
3. **禁止本阶段：** 新开 musubi/VDN 全训、GFPGAN 缝合复活、把角色 LoRA 吹成远脸修复。

通过标准（冒烟）：宽景五官可辨 **明显**优于同画布无 refiner，且无全局糊/身份塌；否则 **维持 no-go 训 (a)**。

## 9. 给用户的中文摘要

FaceRefine 停得对：它是二次生成再贴回去，贵、有缝、加步也救不了「缩回 34px」。  
远脸糊主要是 **脸太小、latent 里没几个 patch**，不是少挂一个 LoRA。  
现有 PDMD/DMAD LoRA 是 **加速蒸馏**，不是补脸分辨率。  
角色 LoRA 能让「像谁」更稳，但 **不能 magically 把 34px 变成清晰五官**。  
若要立刻变好：同一构图 **抬到 1152（DMAD/PDMD）或 1344（Euler）**；真要死守 480p 再谈「末步细节 LoRA」小实验，而不是先开全量训练。
