# VDN 现成模型直接用于本仓的可行性深拆（重开裁决）

> 调研日期：2026-10-07
> 触发：用户明确「VDN 训练成本可接受、已有现成模型、不能因成本否决，一切以目标为主」——此前 `2026-10-05-vdn-linear-attention.md` 第六节「VDN-the-model 维持否决」裁决**重开**。
> 一手源：OpenVDN/vdn-minimax-h3 GitHub+HF README、arXiv:2609.20744、openvdn.github.io、Saganaki22/ComfyUI-VDN-H3（v1.5.2）、ComfyUI Wiki、Banodoco 社区实测、ai-muninn 5090 配置栈实测、SGLang PR #37903。
> 前置：`2026-10-05-vdn-linear-attention.md`（机理）、`2026-10-05-taomate-fasth3-vdn-reanalysis.md`（社区）、`2026-10-07-video-dit-attention-frontier.md`（全景）。
> 本仓锚点：T2（480P 864×480 5s）热 e2e 18.75s；T4（768P 1376×768 5s 混合臂）68.93s；768P 高分全 sol 撞 32GB OOM。

---

## 一、问题 1 钉死：VDN 现成模型的实际形态

### 1.1 发布的权重是什么

**不是完整 checkpoint，是「分支 + LoRA 增量包」，挂在冻结的 H3 backbone 上。** HF 仓库 `OpenVDN/vdn-minimax-h3` 目录结构（2026-10-07 现状）：

```text
ckpts/
  h3-base/             官方 MiniMax H3 全量：transformer + video/audio VAE + schedulers · 72 GB（仅为方便，非必需）
  stage-b-step-2000/   VDN-H3-50-step：linear_branch/ + adapters/default/ LoRA · 4.3 GB
  stage-dmd-step-250/  VDN-H3-8-step：上述 + adapters/turbo/ · 5.1 GB
```

- **每个 stage 只含**：`linear_branch/model.safetensors`（新增逐帧 VDA 线性分支权重）+ `adapters/{default,turbo}/adapter_model.safetensors`（两个小 LoRA，挂 QKV/O 投影）+ `model_spec.json`（混合架构规格）+ `metadata.json`（训练配方）。
- **推理时 merge 进 backbone，不改 backbone 权重**（官方原话 "Plug-and-Play… merged into the backbone during inference without touching the backbone weights"）。
- 训练配方（全部基于冻结 dense H3）：A1 线性分支逐层对齐 200 步 → A2 端到端 500 步 → B（QKV/O LoRA+线性分支）2000 步 → DMD2 8-step turbo LoRA 250 步（**初始化自 larryvrh/MiniMax-H3-Turbo-Lora——正是本仓 turbo4 同族**）。

### 1.2 许可证

- **代码（GitHub 仓库）**：Apache-2.0。
- **权重**：MiniMax H3 Community License（VDN 是 H3 衍生，沿用上游协议；ComfyUI 移植方特别提示「excludes some territories，用前先读」）。
- **与本仓现有 H3 权重同一许可证族**——本仓已在使用 MiniMax H3 基座，无新增许可证障碍。

### 1.3 对 H3 基座版本的依赖

- 基于**官方发布的 MiniMax H3 transformer 权重**（非 stock pruned 特化——但社区 drbaph 已为 pruned fl2va/ref2va 各出专用 turbo LoRA 变体，说明基座变体间 LoRA 不通用）。
- ComfyUI 移植实测兼容 `fl2va` 和 `ref2va` 两种基座（pruned 和非 pruned 均可，用对应 LoRA 变体）。
- **注意**：块数必须匹配（"checkpoint has N blocks but the loaded model has M" 硬校验），VDN stage 与基座深度强绑定。

### 1.4 推理代码依赖栈（官方栈 vs 社区栈）

| 层 | 官方栈（OpenVDN/SGLang） | 社区栈（Saganaki22 ComfyUI 移植 v1.5.2） |
|---|---|---|
| Python | 3.12 | ComfyUI 自带（3.13 实测 OK） |
| PyTorch | 2.13 + cu129 | 2.10/2.13 + cu130（Windows 实测） |
| 注意力核 | **FlashAttention-4**（仅 Hopper + 数据中心 Blackwell sm_100，**无 sm_120、无 Windows**） | 纯 PyTorch eager：grouped SDPA（默认）/ FlexAttention（可选），**零新依赖、无 Triton、无 CUDA 编译** |
| 线性层 | FP8 linears + 融合 Triton kernel | 可换 INT8 ConvRot（drbaph 预量化版，分支 matmul 2.7×、e2e 1.2×） |
| 分布式 | Ulysses 8 卡序列并行（SGLang PR #37903 已合） | 单卡 only |
| 部署评价 | 「undocumented fragile web of patched Diffusers + FlexAttention Triton kernels」（社区原话） | ComfyUI 运行时 patch，不改 core；51 项 CUDA 回归测试全过 |

---

## 二、问题 2 钉死：5090 单卡可运行性

### 2.1 两个 ComfyUI 移植最新状态

| | Saganaki22/ComfyUI-VDN-H3 | FL_MiniMaxH3VDN |
|---|---|---|
| 最新版 | **v1.5.2**（~3 周前更新，239 star，活跃维护） | 活跃（ComfyUI Cloud 上架） |
| 形态 | 运行时 MODEL patch（节点串在 loader 和 sampler 之间），不改 ComfyUI core | 同类运行时 patch，clone 基座模型便于 A/B |
| 5090 实测 | **1280×736 / 145 帧（~6s）/ 8 step：~17 s/it ≈ 2:15 采样**（INT8 ConvRot 基座 + stream 模式 + sage2 patch）；bf16 stage 111s → INT8 stage 95s（61 帧 A/B） | 无独立 benchmark，定位长 multi-shot 渲染 |
| 质量验证 | 官方数学逐层单测对齐；bf16 vs INT8 ConvRot 同 seed 输出**逐位一致**；`merge` 模式 8-step 要求严格（`bypass` 模式在深块放大 bf16 舍入噪声 → 颗粒化，必须用 merge） | 无独立质量报告 |
| fast_kernels | torch.compile 热点（RMSNorm+gate epilogue、state gather、双向扫描）→ **8-step 1280×736 从 ~2:04 → ~1:13**；但**在 8-step DMD stage 上已知漂移**（ulp 级 bf16 舍入被蒸馏采样器放大）——正式渲染必须关掉 | — |

### 2.2 PyTorch 等价物精度损失

- **grouped SDPA vs FlexAttention**：34.5k token 上实测 parity（grouped 每 block 每 step 只发 ~6 次 dense SDPA，flex 融合无收益）。
- **INT8 ConvRot 分支量化**：同 seed A/B 输出**逐位一致**，分支 matmul 2.7×、e2e ~1.2×。
- **真正的精度坑**：① `bypass` LoRA 模式在 8-step DMD 上颗粒化（必须 `merge`）；② `fast_kernels` 在 8-step DMD 上漂移（正式渲染必须关，即 1:13 那个数字**不可用于交付质量**）；③ 窗口 softmax 必须走精确 SDPA，**不能过 sage/kitchen-int8 后端**（实测软化输出）。

### 2.3 sm_120 兼容性

- **官方 FA4 kernel：不支持 sm_120**（仅 Hopper + 数据中心 Blackwell sm_100），无 Windows build。
- **社区移植：sm_120 是它的主测试平台**（RTX 5090 实测，torch 2.10+cu130，纯 PyTorch 无 arch 特定 kernel）——此前 sm_86/3090 的黑图/噪声问题在 5090 上**无报告**，v1.5.x 的 51 项 CUDA 回归测试在 5090 上全过。
- **结论：sm_120 可运行性 = 社区移植已实证，官方栈不可。**

### 2.4 768P 单卡 32GB 能否不 OOM

**实证结论：采样阶段能，VAE decode 阶段需要 tiled decode。**

- 实测：1280×736 / 145 帧 8-step，bf16 stage 峰值后**剩 3.6 GB free**，INT8 stage **剩 8.3 GB free**——VDN 分支只加 ~4.3GB（stream 模式工作集增量仅 ~86MB/block）。
- **1280×736 1344×768 差别不大**（token 数 ~34.5k vs ~39k，+13%），INT8 栈下 32GB 可承载。
- **VDN 的线性分支注意力成本随帧数线性涨、恒定 d×d 状态**——145 帧（6s）已实证不 OOM，理论上 5s/124 帧更轻松。
- **真正的 OOM 点在 VAE decode**：untiled VAEDecode 在 768p+ 一次解码全帧 → OOM，必须用 `VAEDecodeTiled`（与本仓自己的 tiled VAE 同构，非新问题）。
- 旁证：ai-muninn 实测 dense H3 1344×768 362 帧裸跑要 47.11 GiB（OOM），VDN 的恒定状态正是消掉这个二次项的手段。

---

## 三、问题 3 钉死：VDN vs 本仓当前臂的端到端对比预估

### 3.1 Banodoco 2:04 数字的口径拆解

- **实测口径**：1280×736、8-step VDN-H3 Turbo、5090 单卡、ComfyUI 栈，**2:04**；同硬件 LightXv2 4-step turbo **1:24**。帧数未在引用中明示，但 Saganaki22 的 145 帧 ~2:15 / 61 帧 ~95-111s 给出刻度。
- **官方单卡刻度**（arXiv 表，1344×768 / 345 帧 / 14.4s / 768p，DiT 去噪 only）：dense H3 单卡 15.99 s/NFE → VDN-H3 单卡 6.16 s/NFE（2.6×）→ 8 NFE = 49.3s（单 B200，FP8+FA4 栈）。

### 3.2 换算到本仓目标口径（1344×768 或 1376×768，5s = 124 帧）

以社区实测为锚（5090、ComfyUI 纯 PyTorch 栈、INT8）：

| 口径 | 数字 | 依据 |
|---|---|---|
| VDN 8-step @1280×736 / 61 帧 / INT8 | ~95s（e2e 含 VAE+audio） | Saganaki22 A/B 实测 |
| VDN 8-step @1280×736 / 145 帧 / INT8 stream | ~135s（2:15） | Saganaki22 实测 |
| **插值到 124 帧（5s）@ 1280×736** | **~120–125s**（~17 s/it × 8 ≈ 136s 上限，INT8 略低） | 线性插值 |
| **换算到 1344×768（token +13%）** | **~135–150s** | 注意力近线性区（VDN 特性），FFN 严格线性 |
| fast_kernels 乐观值（质量不可交付） | ~1:13 @61帧 → 124帧 ~150s？不适用，8-step DMD 漂移 | 仅列示，否决 |

### 3.3 与本仓 T4 的诚实对比

| | 本仓 T4 混合臂（现状） | VDN 8-step 现成模型（社区栈 5090） |
|---|---|---|
| 分辨率/时长 | 1376×768 · 5s | ~1344×768 · 5s（插值） |
| NFE | 4（turbo4） | 8（VDN turbo） |
| **热 e2e** | **68.93s** | **~135–150s（预估）** |
| 显存 | 高分全 sol OOM，混合臂兜底 | INT8 栈剩 ~8GB，不 OOM |
| 质量 | waffle 0.75–1.05（turbo4 轨迹） | 官方：8-step 五指标追平/略超 50-step dense H3，RAFT 11.71 vs 11.55；社区：「保运动 > FastH3」 |

**诚实结论：在 5090 单卡、纯 PyTorch 社区栈下，VDN 现成模型跑 5s 768P 比本仓 T4 慢 ~2×（~140s vs 69s）。** 原因很简单：① 8 NFE vs 4 NFE（2×）；② 官方 2.6× 架构收益在消费卡纯 PyTorch 等价物上「lands somewhat under that」（移植方原话）；③ 本仓 T4 已是 INT8 W4A8 + sol 混合臂深度优化产物，而社区 VDN 栈是 eager PyTorch。

**VDN 的结构性优势在「长视频 + 高分辨率」**：注意力成本随帧数线性而非二次——145 帧 vs 61 帧只慢 1.42×（帧数 2.38×），dense 栈同比例会慢 ~5.7×。**对本仓 5s 目标，VDN 的线性优势还没进入甜区；对 10s+ 长片，优势会反转。**

### 3.4 质量上 VDN vs 本仓金线

- VDN 8-step 官方口径「near-lossless vs 50-step dense H3」——即 **VDN 的质量锚是 dense H3 金线**，不是 turbo4。
- 本仓 T2/T4 走 turbo4 4-step，已知痛点=比金线糊/过饱和（taomate 调研已证）。
- **若 VDN 官方质量声明成立，VDN 8-step 质量应 ≥ 本仓 turbo4 轨迹**（VDN 追平 dense，turbo4 已知低于 dense）。这恰好命中 T3 质量档的 SKU 定位：**换权换轨迹但人审通过即可**。
- 验证方式：用本仓 `evals/768p` 的 8 prompt 在 ComfyUI VDN 栈跑一遍，waffle + 人审对比 T4 输出。

---

## 四、问题 4 钉死：三条集成路径

| | 路径 A：ComfyUI 外部对照（零集成） | 路径 B：VDN Python 栈出 latent → 本仓 VAE/mux 后半段 | 路径 C：VDN 权重+线性分支移植进本仓 C CUDA 栈 |
|---|---|---|---|
| **做什么** | 装 Saganaki22 节点，跑 evals/768p 8 prompt，产出 mp4 与本仓 T4 对比 | ComfyUI/官方 Diffusers 栈只跑到 latent，导出 `.safetensors`/raw，本仓 `h3_tae`/VAE/mux 接管 | 把 linear_branch + 2 LoRA 的数学（VDA 逐帧联合最小二乘、双向扫描、窗口 softmax、门控）在 `h3_gpu_cuda.cu` 体系重写 |
| **工程量** | **~1 天**（装节点 + 下 5.1GB stage + 跑 8 prompt + waffle/人审） | **~2–3 天**（latent 导出格式对齐 + 本仓 VAE 输入适配 + e2e 脚本） | **~4–8 周**（VDA delta-rule Cholesky solve、双向状态扫描、窗口分组注意力、c=5 r=1 anchor 几何，全部要写 sm_120 kernel 或 cuBLAS 组合；LoRA merge 进 W4A8 权重需重新量化校准） |
| **成本** | 零开发；~5GB 下载；5090 跑 ~2.5h（8 prompt × ~150s） | 低开发；要处理 latent layout/audio latent 对齐 | 高开发；且 W4A8 重量化可能引入新误差（VDN 的 LoRA 是在 bf16 权上训的，merge 进 INT8 权需重校准） |
| **风险** | 仅环境风险（torch 版本、ComfyUI 安装） | latent 格式 mismatch（packed layout、audio latent shape）；两段栈的 RNG/sampler 语义差异 | ① VDA 的 `(I+A_t)⁻¹` 逐帧求解在 INT8 管线里的数值稳定性未验证；② 8-step 轨迹与本仓 turbo4 4-step 调度不兼容（要么换 8-step 要么重蒸馏）；③ 移植后速度未必赢——官方 2.6× 架构收益里 FA4/FP8 贡献不明，纯 sm_120 手写栈可能只剩 ~2× 注意力收益，摊到 e2e 更少 |
| **验证方式** | waffle + 人审 vs T4 输出 | e2e waffle + av_sync PASS + 人审 | 先单 block 位对齐（vs PyTorch 参考），再 e2e |
| **产出性质** | 质量/速度**参照系**（回答「VDN 质量是否真追平 dense」） | **可用的 T3-VDN SKU 档**（若质量过关） | 本仓原生的 VDN 档 |
| **对 T2/T4 目标的贡献** | 零（外部工具） | 零速度贡献（~140s vs T4 69s）；**质量档贡献**（若追平 dense 则优于 turbo4 轨迹） | 理论上有，但 8-step × 2.6× 架构收益 ≈ dense 3.1×，对 5s 短片未必赢当前混合臂；**长视频才反转** |

### 推荐顺序

**A → （若质量惊艳）→ B；C 仅在「本仓决定做 10s+ 长视频 SKU」或「768P 全 sol OOM 墙被判不可解」时才值得。** 对当前 T2（480P 18.75s）/T4（768P 68.93s）目标，三条路径都给不出速度收益——VDN 的甜区（长序列线性优势）与本仓的 5s 目标错位。

---

## 五、问题 5 钉死：VDN 现成模型 vs 本仓 SLA 式方案——替代还是互补

**结论：互补，且 SLA 式方案的优先级不因 VDN 现成模型可用而下降。**

| 维度 | VDN 现成模型 | SLA 式稀疏+线性叠加（本仓计划） |
|---|---|---|
| 动什么 | 换权、换轨迹（4→8 step）、换注意力架构 | 在现有 sol_cute 上加 marginal 块线性分支，不换 backbone |
| 训练 | **零**（现成） | 少量微调（10–100 GPU 时，用户已批可接受） |
| 治 768P OOM | 治（恒定状态）——但 e2e 更慢 | 治（线性分支恒定内存）——且保住 4-step 速度 |
| 对 T2/T4 速度 | **负贡献**（5090 上 ~2× 慢） | 正贡献（稀疏度 70→95%，注意力 13.7×） |
| 质量 | 追平 dense（官方声明，待 A 路径验证） | 论文 95% 稀疏不掉质量；turbo4 上需 Phase 0 验证门 |
| 定位 | **质量参照 + 可能的质量档 SKU** | **T2/T4 速度目标的主攻方向** |

**具体关系**：
1. **VDN 给 SLA 提供了「混合注意力在 H3 上可行」的实证**——VDN 的窗口（c=5 r=1）+线性分工在 H3 上训练收敛且 near-lossless，直接降低 SLA 式叠加的架构风险。
2. **VDN 现成权重可当 SLA 的「教师信号」**——VDN 的线性分支输出可作为本仓线性分支对齐的参考（类似 VDN A1 阶段的逐层对齐）。
3. **若路径 A 验证 VDN 质量真追平 dense，T3 质量档可直接用 VDN SKU（路径 B），SLA 专攻 T2/T4 速度**——两者分管质量档和速度档，不冲突。
4. **唯一替代关系**：若 VDN 质量惊艳且本仓接受 8-step 轨迹，理论上可放弃 turbo4 改 VDN 全线——但这意味着 T2/T4 速度目标全面倒退（5090 上 2× 慢），与「一切以目标为主」冲突，**不推荐**。

---

## 六、总裁决

**现成模型可用性：条件能。**
- ✅ 权重现成（5.1GB 增量包）、许可证无新增障碍、sm_120 经社区移植实证可跑、768P 32GB 不 OOM（INT8 栈 + tiled VAE decode）。
- ⚠️ 条件：必须用社区 PyTorch 移植栈（官方 FA4/FP8 栈 sm_120 不可用）；必须 `merge` 模式 + 关 `fast_kernels`（否则 8-step DMD 漂移）；e2e 速度在 5090 上 ~135–150s/5s 768P，**比本仓 T4 慢 ~2×**。
- **定位：质量参照系 + 候选 T3-VDN 质量档 SKU，不是 T2/T4 速度目标的解法。**

**与本仓目标的诚实差距**：
- T2（480P ≤19s）：VDN 无贡献（VDN 甜区在长序列，480P 16k token 且 8-step vs 4-step 必慢）。
- T4（768P ≤70s）：VDN 现成模型 ~135–150s，**差 ~2×**；VDN 的价值是把「768P 不 OOM + 追平 dense 质量」变成现货，但速度档目标仍需 SLA 式方案。
- **下一步行动：路径 A（1 天，ComfyUI 外部对照跑 evals/768p 8 prompt）→ 若质量确认追平 dense，启动路径 B 作为 T3-VDN SKU；SLA 式叠加维持 T2/T4 主攻方向不变。**
