# taomate / FastH3 / VDN 官方+社区重分析

> 来源：子代理 `ab30aadd-476f-4c8b-9311-0e828676ea65`（只读）。一手源：TaoLiveAIGC/TaoMate-H3 GitHub+HF、FastVideo hao-ai-lab 官方博客+HF、OpenVDN GitHub+项目页+arXiv:2609.20744；社区：WhichH3 盲测 arena（4000+ clip）、Reddit r/StableDiffusion、ComfyUI Wiki、Kijai 转换、Banodoco 实测。

## 一、taomate —— 最大认知缺口已补，藏了一个我们 miss 的用法

**官方机制 = 三件套非单一技术**：①3-step 区间蒸馏 LoRA（锚点 {0,16,33,49}，与我们 `H3_TAOMATE_STATE_INDICES` 完全对应）②流式分块（5s=124帧切4 phase，对应我们 plan 表）③clean-KV（每块去噪后 timestep=1.0 再 forward 提交干净 K/V，对应 `H3_TAOMATE_CLEAN_KV`）。**`H3_TAOMATE_STAGE1_H/W` 非官方概念**——是我们自创的 taomate-lift preset（taomate LoRA 叠 SelfLift 双采样）。官方数字 8×H20 11.45×（分母 50-step，单卡 5090 官方树跑不起来，FA3/Hopper 硬合同）。

**⚠️ 社区发现的我们完全错过的新用法：taomate 当 0.7 强度 refiner 收尾而非整段 3-step 生成。**
- 整段 3-step 官方意图：WhichH3 盲测 arena（4000+ clip）**垫底**（i2v Elo 903.5 #19）；Reddit「3 steps nothing to write home about…smearing/broken output」。
- **refiner 用法（高票）**：正常高步主体 + taomate 3-step @ 0.7 强度收尾——「close the low step generation without smearing/flickering…preserving physics/material/prompt adherence」，且**治的正是 turbo LoRA 的「overcooked」过饱和**（「VERY good visual quality without obvious overcooking plaguing the turbo loras」）。已固化配方：2 步 turbo4+3 步 taomate 收尾，或 8/9 步主体+taomate 收尾；Kijai 出 rank-19 平均版专当 refiner。
- 坑：①基座版本强耦合（stock pruned 保色 42.8 vs Singularity v1.3 掉色 28.6）②我们 `h3_lora_parse_name` 显式拒绝 `token_refiner` 模块，挂官方全量权会失败，Kijai rank-19 版反而更适配。

**裁决**：官方 streaming/clean-KV 机制我们分析到位不构成新发现；**真正盲区=refiner 用法**——我们把 taomate 框死在「整段 3-step 生成器」，从没测过「高步主体+taomate 低强度收尾」这条社区主推路径。考虑到 turbo4 痛点正是比金线糊/过饱和，而社区报告 taomate refiner 恰好不过饱和保物理/材质/运动——**值得低成本 A/B**。

## 二、FastH3 —— 官方自认 4-step=最低延迟档，我们没误评

**机制两条刀**：DMD2 步数蒸馏（49→4 NFE）+ **VSA 可训练稀疏**（video-video 只留 ~10% tile，**稀疏训进权里**，VSA LoRA 不能丢进 dense 路径，需 B200 sm100a tile-64 kernel）。**官方自己框定质量合同**：「The four-step model targets minimum latency; **eight steps may be a better quality setting**」+ 承认「difficult motion, fine detail, some audio can fall below base」。

**社区独立裁决**：WhichH3 arena 中段偏下（i2v Elo 987.1 #12，低于 larryvrh turbo v4 8-step #2）；**VDN 官方实测 FastH3 4-step RAFT 运动量塌到 9.19（dense 11.55）——快运动塌缩被独立复现**；ComfyUI 转换版**必须 6 步而非 4 步**（4 步 jitter/flicker/color bloom unusable，因 repacked 模型丢 AdaLN 时间步校准）——证明真实质量点在 6 步非宣传的 4 步。

**裁决**：我们实测「worse quality」**无误评**——官方立场+arena 盲测+VDN 实测三方独立确认。真正值得记下的 refinement：① FastH3 的价值不在权而在「**蒸馏时把稀疏训进去**」配方（暗示 sol_cute 若做稀疏感知微调上限更高，SLA 同向）② 若未来出官方 8-step matched 版才值得与 turbo4/金线重测。当前维持既有评估。

## 三、VDN —— 社区早期但真实采纳，「模型否决机制重开」获独立佐证

**发布形态比我们此前记录更具体**：非全量换骨干——线性分支+两个小 LoRA adapter **推理时 merge 进冻结 H3 backbone**（社区形容「plug-in architectural patch」）。训练四阶段 A1 线性对齐→A2 端到端→B(QKV/O LoRA+线性）→DMD 8-step turbo LoRA（**初始化自 larryvrh/MiniMax-H3-Turbo-Lora——就是我们 turbo4 同族**）。

**社区采纳**：GitHub ~491 star；**已有 2 个 ComfyUI 移植**（Saganaki22/ComfyUI-VDN-H3、FL_MiniMaxH3VDN），用可移植 PyTorch 等价物替换 FP8/Triton 核并单测。**独立实测与 14.5× 的距离**（Banodoco）：5090 单卡 ~17s/it；8-step VDN 2:04 vs LightXv2 4-step turbo 1:24 同硬件——**消费卡仅 ~2.6× 非 headline 加速，但「holds fast motion better than FastH3」**（运动保持优于 FastH3，与官方 RAFT 互证）。批评：依赖「undocumented fragile web of patched Diffusers+FlexAttention Triton kernels」部署脆；sm_86/3090 fallback 黑图/噪声复现。

**裁决**：官方+社区记录**强化而非推翻**我们既有结论——VDN-the-model 维持否决（多卡数据中心合同、依赖脆弱、5090 无官方支持、消费卡无 headline 加速）；线性/混合机制重开不变。两个 refinement：① 社区移植证明「线性分支+LoRA 挂载冻结 backbone」工程可插拔（降低「必须换底座」估计——但质量靠四阶段训练，插拔的是推理形态非训练成本）② 「VDN 保运动>FastH3」被社区独立复现，给「SLA 式稀疏+线性混合叠 sol_cute」再添一票——治的正是稀疏/少步栈的快运动弱点。

## 四、跨方案一句话

| 方案 | 官方合同 | 社区 verdict | 对我们的新信息 |
|---|---|---|---|
| **taomate** | 3-step LoRA×流式×clean-KV | 整段垫底；**refiner 收尾口碑极好** | ⚠️ refiner 用法没测过，治 turbo4 过饱和，值得低成本 A/B |
| **FastH3** | 4-step=最低延迟档（官方自认） | 中段偏下；快运动塌缩被复现；需 6 步 | 我们评测无误；有价值的是「稀疏训进权」配方非权本身 |
| **VDN** | 混合线性，单卡仅 2.6× | 2 个 ComfyUI 移植；消费卡无 headline；保运动>FastH3 | 「模型否决机制重开」获佐证；可插拔但质量靠训练 |
