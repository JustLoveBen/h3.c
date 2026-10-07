# DMAD 成片全噪音 × waffle 低分矛盾：根因诊断

日期：2026-10-07 · 诊断对象：`out/review-2026-10-04/kpib-dmad-kj/{C_full_critic,D_kijai_rank39}/ab_5s_r{1,2}.mp4`

## 1. 噪音形态确认（抽帧证据）

ffmpeg 抽帧（第 0 / 60 / 119 帧）到 `out/diag-dmad/`，Read 直看结论：

| 臂 | 帧 | 形态 |
|---|---|---|
| C_full_critic r1 | f0/f60/f119 | **均匀灰色细颗粒噪声，零语义结构**，三帧噪声图样几乎不随时间变化 |
| C_full_critic r2 | f60 | 同上，纯噪 |
| D_kijai_rank39 r1 | f60 | **严重退化的网格状马赛克**（隐约有人形轮廓残余，周期性格状伪影） |
| D_kijai_rank39 r2 | f60 | 接近纯噪 |
| 对照 kpia-baseline-480p r1 | f60 | 正常画面（人物、清晰边缘），全局像素 std=23.4 |

证据文件：`out/diag-dmad/kpib-dmad-kj_C_full_critic_ab_5s_r1_f{1,2,3}.png`、`..._r2_f2.png`、`kpib-dmad-kj_D_kijai_rank39_ab_5s_r{1,2}_f2.png`、`kpia_baseline_f60.png`。

**量化指纹**（f60，16px 块统计，纯 Python 实测）：

- DMAD r2：块内方差 13.1，**块均值间方差/块内方差 = 2.95**，全局 std=7.2
- 正常对照：块内方差 242.7，块间/块内 = 1.28，全局 std=23.4

> **2026-10-07 修正**：上行"正常对照块间/块内 = 1.28"系早期 64×64 合成平滑帧记录，不是 480p 实拍。480p 全量复测（kpia r1/r2 × f0/f60/f119，18 帧）：正常片块间/块内 = **17.5–146.8**，DMAD 噪音双臂 = **1.20–7.57**——两类分布有 ~2.3 倍间隙，该比值是已验证可分统计量（见末节）。

DMAD 输出的能量集中在 **16px 块均值之间的跳变**，块内部几乎平坦——即"VAE patch（16px）级别的低频网格噪声"，不是传感器式白噪声。这解释了为什么肉眼看到"灰色颗粒+格子感"。

## 2. waffle 矛盾解释：度量失效，不是记录错误

waffle b16 的实现（`h3_waffle.c:10-35`）是**无参考**启发式：沿 16px 网格边界线的亮度梯度能量 ÷ 块内部梯度能量。设计意图是检 VAE tile/patch 接缝（好画面块内有纹理→inner 大→比值低；有接缝→edge 大→比值高）。

纯噪音恰好欺骗了该度量的两个前提：

1. VAE patch 级噪声使块内梯度（inner）趋近于 0，块间均值跳变（edge）保留 → 但 x264 编码把边界跳变 smoothing 后，比值落入 0.8–1.3 的"低 waffle"区间；
2. 度量完全没有"画面必须有语义内容/足够梯度能量"的 sanity gate——`inner < 1` 才返回 0，纯噪的 inner≈13（>1）轻松通过。

原始记录溯源：`docs/superpowers/plans/2026-10-04-h3c-480p-10s-plugin-plan.md:267`（2026-10-05 修订）明确写"hot waffle 0.79-1.34 优于 turbo4 1.04-1.71…建议作 SKU 隔离可选后端候选（**待人审**）"。日志复核：C_full_critic r2 实测 0.790–1.339、D_kijai_rank39 r2 0.859–1.233（`generate.log:206-210`）——数字记录无误，**是度量对纯噪输出给出了假低分**。

旁证：同目录下 r1（cold clip）waffle 为 3.6–3.8（C）/ 5.3–8.8（D）——噪音更重/编码不同段时 waffle 也会升高，说明该指标对噪音的响应方向都不稳定，完全不可用于跨配置质量比较。

## 3. 生成配置复核

两臂与 turbo4 对照共享同一调度：`steps=4 cfg=1 sigma=[1, 0.973, 0.9231, 0.8, 0]`、`preset=turbo4`、sol_cute 400/400 hits、av_sync PASS。可疑点：

- `adaln_cond: step0 sigma=1 active=0 mod_row=0 t_row=0 (want active=1 mod_row=3 for F2VA σ=1)`——**DMAD 两臂与 turbo4 金线均打印同样的 want/active 不符**，说明这不是 DMAD 独有问题（turbo4 在此配置下出片正常），不足以单独定罪，但提示 turbo4  preset 的 F2VA 条件注入本身处于"非合同"状态。
- DMAD（arXiv:2610.02188）是字节官方 4 步蒸馏 LoRA，论文配套调度未必等于本仓 turbo4 的 sigma 表与 `velocity_euler` 采样器；PEFT 键转换 err=1.1e-08 只保证权重数值加载正确，**不保证推理配置匹配**。用 turbo4 的 schedule 跑 DMAD 权重出噪音是最可能的生成侧根因。
- D_kijai_rank39 当时的修订记录已写"cold clip 噪声 b16 5.3-8.8，不推荐"——D 臂部分被 waffle 抓到过，C 臂（full_critic）则完全漏检。

## 4. 根因裁决

**裁决：C（两者皆有），B 为主责，A 为被掩盖的真因。**

- **B. waffle 度量失效（主责）**：无参考 16px 边界梯度比对纯噪输出给出 0.79–1.34 假低分，且响应方向不稳定（同片 r1 3.6–3.8）。所有"waffle 优于 turbo4"的跨配置结论作废。
- **A. 生成配置错（真因，被 B 掩盖）**：DMAD 权重在 turbo4 调度（sigma 表 / velocity_euler / adaln_cond active=0）下实际产出的是 patch 级网格噪音。需按 DMAD 官方推理配置（调度器、σ 序列、CFG、条件注入）重跑验证，才能判定是配置不匹配还是权重/加载问题。

## 5. 对 T3 / roadmap 的修正建议

1. **立即作废**：`2026-10-05-h3-optimization-roadmap.md:51,96,152`、`2026-10-05-h3-optimization-dev-plan.md:41`、`2026-10-05-quality-preserving-acceleration.md:60-69`、`2026-10-07-deadend-audit-quality.md:16,64` 中所有"DMAD waffle 0.79–1.34 优于 turbo4"结论及 T3 达标表述。
2. **T3 档状态降级**：DMAD 后端从"待人审签字"降级为"**质量未验证**"——人审已判否（全噪音）。重跑前不得进入 T3。
3. **waffle 度量加 gate**：在任何质量档验收中，waffle 必须搭配最低语义能量门槛（如块内方差下限、或与参考帧的相关性检查），纯噪/平坦输出直接判 FAIL；waffle 仅限用于同配置 tile 接缝回归，禁止跨权重/跨调度比较。
4. **DMAD 复验步骤**：查 DMAD 官方推理配置（scheduler/σ/CFG/步数语义）→ 修正 `h3_generate_preset` 的 DMAD 臂调度 → 重跑 480p 双臂 → 人审 + 有参考指标（与 turbo4 同 prompt 的 VBench/人审对比）双重确认。

## 6. Gate 实现与标定（2026-10-07，同日完成）

`h3_waffle.c` 已加最低语义能量 gate（`h3_waffle_b16_gated_rgb24`，opt-in、WARN-only，不改退出码）：

- **判别统计量**：16px 块间方差/块内方差比。**原始 inner 梯度能量不可用作 gate**——实测正常 kpia r2（人审通过）inner_e 低至 0.45M，而 D 臂马赛克噪音 r1 高达 2.5M，方向与画质无关。
- **标定**（864×480，18 帧 = 正常 kpia r1/r2 × f0/f60/f119 + 噪音 C/D 双臂 × f0/f60/f119）：正常 17.5–146.8，噪音 1.20–7.57。
- **阈值 10.0**（贴噪音簇上方，不取几何中点 24）：kpia r1 f0 是真实人像特写帧，大面积平坦肤色使 16px 块低频自相似、ratio=17.5 落在原始分布间隙内，中点阈值会假阳性真实内容。
- **离线验证**：18/18 判别正确（正常 6 帧零触发、DMAD 噪音 12 帧全触发）；`h3_waffle_test` PASS。
- 待 238 验证：整仓 CUDA 编译 + turbo4 golden 流程零触发回归。
