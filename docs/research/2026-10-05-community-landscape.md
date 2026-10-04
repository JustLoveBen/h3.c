# H3/视频-DiT 加速社区全景与横向基准

> 来源：子代理 `90eeebd6-7406-4163-ae18-7021bc2e1419`（只读）。

## 1. H3 生态加速件社区口碑

| 加速件 | 口碑 |
|---|---|
| 官方 turbo4 LoRA（MiniMax）+ larryvrh 节点 | 默认主流 |
| Kijai int8_convrot / w4a8（Comfy Kitchen） | **质量最稳但慢** |
| Sage Attention（KJ） | 最受欢迎的即插即用无损加速 |
| EasyCache | **口碑崩坏**（毁脸/毁肢体/高分辨率崩溃） |

## 2. 机制横向对比（DisCa CVPR 2026 表 = 最干净横向证据）

质量-加速比排序：**可训练蒸馏+稀疏 > 可训练稀疏(SLA2/VSA) > 自适应缓存(≤2×) > 朴素缓存(>2× 悬崖）**。DisCa 表显示缓存类在 6–9× 加速下全线掉 13–42% 语义分——印证「缓存 >2× 是悬崖」。

## 3. 社区共识与我们结论的关系

- **turbo 4 步运动软化、8 步回归** ✅ 印证 golden=8
- **量化+蒸馏叠加变软** ✅ 印证需逐组合验证
- **蒸馏非免费午餐**（大运动软化/纹理丢失），前沿配方=「蒸馏+细节恢复 LoRA+可训练稀疏」三件套
- **8 步是甜点** —— 与内部 golden=8 一致

## 4. ⚠️ 唯一实质冲突（需复核）

**Hao AI Lab / FastWan 论点：训练无关稀疏在 <10 步蒸馏下急剧退化。** 这与我们「turbo4 + sol_cute（推理期训练无关稀疏）」组合直接冲突——**建议复核 sol_cute 在 turbo4 下的实际稀疏有效性**（480p hits=400/400 说明路由在跑，但需确认在 4-step 蒸馏轨迹上稀疏是否仍质量无损）。**SLA2（可学习路由+QAT）** 是更稳的演进方向（稀疏训进权里，95–97% 稀疏质量不输 full attention）。

## 5. 需自查的新 SOTA（此前未分析）

SLA2、DisCa、Video-BLADE、FPSAttention/PAROAttention。

## 6. 新机会（图像侧已验证、视频/H3 侧空白）

**蒸馏轨迹对齐的细节恢复 LoRA**（Z-Image DistillPatch、Krea2 4-step）——在视频/H3 侧是空白，可作为 **4 步质量补齐候选**（治 turbo4 过饱和/糊，与 taomate refiner 用法同向）。

## 7. 实践痛点 = 我们的差异化

VRAM 墙 / kernel 兼容地狱 / marketing vs real —— 我们的**独立 C 引擎 + eval 协议**恰好是差异化（不依赖 PyTorch 生态脆性、有 checksum/waffle/av_sync 门）。
