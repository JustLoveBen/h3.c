# 质量/模型/缓存类「封档结论」重审：dead-end 审计（质量线）

> **日期：** 2026-10-07。**性质：** 专项方法论审计（只读，不动代码、不动 238）。
> **背景：** Sol-Attn 审计（`2026-10-07-sol-attn-negative-audit.md`）已坐实本仓存在「负面结论过度泛化」问题；VDN 重分析（`2026-10-07-vdn-direct-use-reanalysis.md`）在用户「训练成本可接受、一切以目标为主」口径下推翻了「模型否决」裁决。本审计按同一标尺重审**质量/模型/缓存**线全部封档结论。
> **必读已读：** roadmap（`2026-10-05-h3-optimization-roadmap.md`）、机理文档（`2026-10-05-acceleration-mechanisms.md`）、taomate/FastH3/VDN 重分析（`2026-10-05-taomate-fasth3-vdn-reanalysis.md`）、V0/V1 原始 A/B（`2026-10-05-v0-v1-sol-ab-results.md`）、Sol-Attn 审计、VDN 重分析、external-h3-benchmark 拒绝清单勘误、480p-10s-plugin-plan §5、lightwan-h3-speed（MagCache p4 实测）、2026-10-07-video-dit-attention-frontier.md。
> **联网补充（2026-10-07）：** MagCache 论文 Table 5（FusionX 10-step 蒸馏实测）、Kijai HF 仓库（taomate rank-19 权重路径）。

---

## 0. 裁决总表（先给结论）

| # | 条目 | 原结论 | 裁决 | 一句话理由 |
|---|---|---|---|---|
| 1 | 跨步缓存 @ turbo4「冗余已榨干」封档 | 三方印证、以实验封档 | **修正（措辞）+ 建议低成本实测封档** | 三方证据全部成立但均非 turbo4 本体实测；MagCache 论文有蒸馏模型正例（FusionX 10-step 1.66×），「部分重叠」机理意味着残余空间未被排除——封档方向不变，封档方式应从「推理封档」升级为「一次 2–3h SKU 隔离臂实测封档」 |
| 2 | sol 全稀疏 @ turbo4 判退化 | 判退化退出默认 | **维持 + roadmap 需回填 4 处差异** | 与 Sol-Attn 审计一致；roadmap 未回填「全稀疏配置退化≠sol 退化」的措辞收窄、V1-a 双正、768P 反慢 1.2s 证伪、R1–R4 重测项 |
| 3 | DMAD 评测（无 speedup、dit +11%、waffle 0.79–1.34） | T3 质量档候选，待人审 | **维持 + 流程缺口的唯一动作是人审签字** | +11% 是在当时的全优化默认栈上测的（rank384 vs 64 是机理充分解释）；成片已在 review 目录但无人审记录 |
| 4 | FastH3「worse quality 无误评」 | 评测无误 | **修正：不是「本仓实测」，是三方旁证裁决；触发条款已有现货** | 全仓搜不到本仓 FastH3 实测记录，「我们实测」表述失实；ComfyUI 6-step 发现针对 repack 转换版，不直接证本仓测的是缺陷版；官方 8-Step V2 已发布——重分析自己写的「若出 8-step matched 版才值得重测」条款已被触发 |
| 5 | taomate 整段 3-step 垫底 vs refiner 用法 | V2 因缺权重取消 | **重开（障碍已消失）** | Kijai rank-19 版权重已在 HF 公开（`Kijai/MiniMax-H3_comfy`，182MB，2026-09 上传），「238 无权重」前置条件不复存在 |
| 6 | T3-c cold-clip 间歇高噪根因定性 | cold 采样轨迹数值失稳 | **修正：定性收窄 + 给出方向 B 成本排序** | 「纯 cudnn dense cold 也全帧>3」充分排除了 sol/护栏，但**不充分排除 VAE/INT8 权窗**（cold 臂同样经过 VAE decode 与 INT8 DiT 栈）；方向 B 三候选成本：pinned restore ≈0（已交付）、vae_th 竞态 0.5–1 天、INT8 make_room 2–3 天 |
| 7 | 金线换锚可靠性 | 新金线 `0x923b0db3a8c0357e` | **维持（checksum 口径）+ 修正（速度口径建议改 hot-only）** | checksum 取自 hot clip2，天然免疫 cold 噪声；但「hot total≈19.2–19.4s」含 cold 段，与 P0 后 17.72s 口径并存易误读——建议速度金线统一改为 hot-only（repeat≥3 取 clip2 P50） |

---

## 1. 跨步缓存（TeaCache/MagCache/EasyCache）@ turbo4「冗余已榨干」封档

**原结论：** 三方证据升级为「以实验封档」——① Sol-H3 官方 4-step 栈也不用 cache 而用蒸馏（FastWan 论点）；② IonDen 实测 8-step 蒸馏相邻 rel-L1 全 ≥25%、阈值 0.20 零跳步；③ SGLang SCM 对 <8 步自动禁用。

### 审计点 ①：MagCache 在 4-step 蒸馏模型上有没有直接实测？

**查证结果：没有 turbo4/H3 4-step 的直接实测，但有相邻证据，且其中一个方向相反：**

- **本仓实测只有 8-step**：`2026-08-21-lightwan-h3-speed.md` p4 臂（`--magcache-thresh 0.08`）在 8-step 蒸馏栈上 skips=3、latent MAE=0.85、帧 RGB MAE 52–68 判质量失败。**turbo4 4-step 上本仓零实验**——external-h3-benchmark 勘误（`2026-10-05-external-h3-benchmark.md` §拒绝清单勘误）自己已承认这一点，并登记了桶 B「2–3 天 opt-in 以实验封档」待办，但该待办从未执行。**现状是「推理封档」，不是文档宣称的「以实验封档」。**
- **MagCache 论文 Table 5 直接做了蒸馏兼容性实验（NeurIPS'25，本审计联网补查）**：在 FusionX（Wan2.1-14B 的少步蒸馏变体，T=10）上 MagCache 达 **1.66× 加速且质量无显著下降**（LPIPS 0.1812/SSIM 0.7868/PSNR 24.23，远优于把 FusionX 直接砍到 T=6 的 0.2982/0.6471/20.35）。**这证明「蒸馏后再缓存」并非在所有蒸馏模型上都零空间**——10-step 蒸馏与 4-step 蒸馏的冗余量不同，但方向性证据存在，且与 IonDen 的 8-step 零跳步结论形成张力（两者基线模型/调度不同，不矛盾，但说明「蒸馏榨干」的程度是模型相关的，不能从 8-step 外推到 4-step 判死刑，同样也不能从 10-step 外推说有肉）。
- **SGLang SCM <8 步禁用**：属实（LMSYS blog + Cache-DiT 文档），是工程保守默认值，不是「4-step 上测过且失败」的证据。
- **EasyCache**：`2026-10-05-industry-coverage-gaps.md` 盲区 5 已登记「疑似同族、自身质量数据未查、不宜直接封档」——与缓存封档结论存在未消化的内部矛盾。

### 审计点 ②：缓存的「冗余」与蒸馏的「冗余」是否同一块？

**机理文档自己的口径是「部分重叠而非全等」**（`acceleration-mechanisms.md` 族 1(e)：「与族 4 缓存**部分重叠**（都吃相邻步冗余；蒸馏后缓存空间塌缩）」；族 4(d)：「低步数/蒸馏模型上空间塌缩」）。**「塌缩」≠「归零」**——文档的机理表述从未说过全等。但 roadmap/不做清单的实际执行口径（turbo4 preset 在代码层直接拒绝 cache 参数叠加）是按「全等/归零」处理的。**表述与执行之间存在口径滑移。** 另外注意 IonDen 的 rel-L1≥25% 是「相邻步 latent 变化量」证据，它证明的是 TeaCache 式阈值法难命中；MagCache 的幅度比衰减律是另一类代理量，IonDen 证据不直接覆盖。

### 审计点 ③：SKU 隔离臂低成本实测封档是否值得？

**值得，且成本已知。** 本仓已有 `--magcache-thresh` 完整实现（opt-in、max_consec=2、音视频分账、mag_fresh 护栏），只需在 turbo4 preset 上解锁代码层拒绝 + 保守阈值（0.02–0.04，retain 首末步）跑 480P A/B：量 skip 命中率 + waffle + 人眼。预期结果「跳 0–1 步、收益 <0.5s」，从而把三方推理封档升级为四方实测封档；小概率（FusionX 先例）白捡 T1 档收益。**预估 238 机时 2–3h（含 build），零代码风险（默认路径不动）。**

**裁决：【修正】**——封档方向维持（不翻默认、不进质量档），但：① 「以实验封档」表述修正为「三方旁证 + 8-step 实测外推，turbo4 本体未实测」；② 登记一次 2–3h SKU 隔离臂实测作为正式封档动作（这也是 external-h3-benchmark 桶 B 早已承诺但未兑现的待办）；③ EasyCache/新缓存族工作的「疑似同族封档」在实测前一律降级为「待查一手数据」。

---

## 2. sol 全稀疏 @ turbo4 判退化：V0/V1 与新审计的一致性核对

Sol-Attn 审计（`2026-10-07-sol-attn-negative-audit.md`）已覆盖本条，此处只做一致性核对。**V0/V1 文档（`2026-10-05-v0-v1-sol-ab-results.md`）与审计结论无冲突**——V0/V1 文档本身的数据（V1-a waffle 0.813 优于 dense 0.849、hot 17.01s 比 dense 快 5.5s）就是审计「双正」结论的来源，V0/V1 §5 分支建议第 2 条甚至已建议「默认栈改为 V1-a」。

**但 roadmap 尚未回填以下差异点（裁决：【维持判定 + 修正 roadmap 文本】）：**

1. **措辞收窄未回填**：roadmap §0.3/§5 的「sol 全稀疏@turbo4（判退化退出默认）」本身措辞是对的（限定了「全稀疏」），但 §4.1b V0 行「FastWan『训练无关稀疏<10 步蒸馏退化』论点在本仓实证成立」**过度泛化**——实证成立的只是「全稀疏配置退化」，带护栏的 sol（V1-a）在 turbo4 上双正，FastWan 论点不能写成无修饰成立。frontier 文档 §6.2/§9 同款表述精度不足（审计 §2 已点名）。
2. **「8-step 金线档是否保留 sol 全稀疏待另行评估」残余动作悬空**——V1-a 数据实际上已暗示答案（护栏几乎零代价 +0.245s），该评估可直接闭合。
3. **768P「sol 臂反慢 ~1.2s」**（`2026-10-01-sol-h3-analysis` §8.5 叙事）被审计证伪（同臂跨拍漂移 3.3s，±3s 噪声内判不了正负）——roadmap T4-d 行「快 ~2s 量级，但机器噪声 ±3s 内」口径正确，但 10-01 文档的负面叙事未勘误。
4. **R1–R4 重测项未登记进 roadmap 阻塞表**：R1（480P V1-a 公平速度对照，~1.5h）、R2（768P 同日三臂，~2h）、R3（sol warmup per-clip 归属，~0.5h）、R4（INT8 vs FP16 稀疏率 Jaccard，~1h）。

---

## 3. DMAD 评测结论

**原结论：**「无 speedup、dit +11%、waffle 0.79–1.34 优于 turbo4」（480p-10s-plugin-plan 2026-10-05 修订记录）。

### 审计点 ①：+11% 是在什么配置上测的？

从修订记录与成片归档（`out/review-2026-10-04/kpib-dmad-kj/`，本机已核实目录存在，含 C_full_critic 与 D_kijai_rank39 两臂 ab_5s_r1/r2.mp4 + generate.log）交叉确认：**DMAD A/B 跑在当时的默认产品栈上**（turbo4 preset = INT8 ConvRot DiT + sol_cute + 官方 VAE + stitch pipeline，480P 双臂 av_sync/sol 全 PASS），即「本仓全部已落地优化」配置。**但两个口径注记**：① 测量日（10-05）早于 P0 async 流水翻默认（10-06）与 DiT TF32/x264 tier 路由，故 +11% 是「dit 段」口径，与今日 binary 的 e2e 占比会略有不同（VAE 变快后 dit 占比上升，+11% dit 对 e2e 的拖累会略放大）；② +11% 有充分机理解释（DMAD full_critic LoRA rank=384 vs turbo4 rank=64，同 NFE=4，LoRA merge 后无额外 kernel 但权重矩阵 rank 差异改变计算量）——**这不是「测试配置缺陷」产物，是 LoRA 固有成本，结论稳健**。

### 审计点 ②：T3 人审是否签字？成片 review 状态？

**未签字。** roadmap §4.2 阻塞表至今挂「T3-a DMAD full_critic 发布｜等人工 review 签字」，修订记录无签字条目；成片在 review 目录已超 3 天。**T3 整档的唯一未闭合项就是这个流程门，不是技术门。**

**裁决：【维持】**——评测结论（无 speedup、waffle 区间、Kijai rank39 不推荐）证据链完整，无泛化问题。唯一动作项：**人审签字**（成本 = 人看 2×2 个 5s mp4，~15 分钟）。建议附带一次「今日 binary 复跑一次 C_full_critic」以刷新 P0 后口径（~0.5h 238 机时，可选）。

---

## 4. FastH3 评测结论

**原结论：**「worse quality 无误评」（taomate-fasth3-vdn-reanalysis §二）。

### 审计点 ①：测试用的是 4-step 还是 6-step？

**关键发现：全仓文档搜索找不到任何本仓 FastH3 实测记录。**「我们实测『worse quality』」的表述失实——重分析文档自己列的证据全部是**外部三方旁证**：① 官方立场（「4-step targets minimum latency; 8 steps may be a better quality setting」+ 承认 motion/detail/audio 可低于 base）；② WhichH3 arena 盲测（Elo 987.1 #12）；③ VDN 官方 RAFT 实测（9.19 vs dense 11.55）。本仓既没有跑过 FastH3 4-step，也没有跑过 6-step——**FastH3 从未在本仓落地过（无 LoRA 加载器适配 VSA 门，09-11 计划文档即已登记「本仓无 LoRA 加载器」）。**

### 审计点 ②：ComfyUI 版必须 6 步是否意味着本仓测的是有缺陷的转换版？

**不适用（因为本仓根本没测）**，但该发现的归属需要澄清：ComfyUI「必须 6 步」的原因是 **repacked 转换版丢 AdaLN 时间步校准**（4 步 jitter/flicker/color bloom）——这是**转换 artifact**，不是 FastH3 官方权本身在 4-step 不可用。官方 FastVideo 原生栈的 4-step 就是按 4-step 设计的（最低延迟档）。所以：若未来本仓要测 FastH3，**用 Comfy-Org repack 版必须 ≥6 步，用官方栈才是 4-step**——这一口径必须写进任何未来 A/B 的实验设计，否则会犯「测转换缺陷版再判模型死刑」的错误（与本次 Sol-Attn 审计发现的「用半接线测全稀疏再判 sol 退化」是同一类方法论风险）。

### 审计点 ③（新发现）：触发条款已有现货

重分析自己写了「若未来出官方 8-step matched 版才值得与 turbo4/金线重测」。**该版本已存在**：2026-09 发布的 **FastVideo-FastH3-8-Step-V2**（DMD2 data-free、VSA 80% 稀疏、只蒸 T2VA，Comfy-Org 已 repack，`2026-09-22-h3c-accel-web-survey.md` §3.1 已登记但标注「5090 全含秒数公开搜不到、对方向档 Ref2VA 未蒸」）。**8-Step V2 与金线 8-step 同 NFE——「matched 版」条件已满足**，重测条款被触发（限定 T2VA/480P 场景）。

**裁决：【修正】**——① 「本仓实测 worse quality」修正为「本仓未实测；结论系官方立场+arena+VDN RAFT 三方旁证裁决，旁证链本身成立、『无误评』的下半句维持」；② 实验设计纪律新增：任何 FastH3 A/B 必须声明权重来源（官方栈 4-step vs Comfy repack ≥6-step），禁止混用；③ **登记 FastH3-8-Step-V2 重测项**（T2VA 限定）：因 VSA 快核是 sm_100a 专属，5090 上需 `--vsa-kernel triton --no-fa4` 或 dense 跑——dense 跑 8-step 相对 turbo4 无速度意义，**故重测价值限于「8-step VSA 质量 vs 金线 8-step」的质量参照**（外部 ComfyUI 对照即可，~1 天，与 VDN 路径 A 同形态），不进本仓速度栈。

---

## 5. taomate 整段 3-step 垫底 vs refiner 用法（V2 取消）

**原结论：** V2（taomate 当 0.7 强度 refiner 收尾 A/B）因「238 无 taomate LoRA 权重文件（仅 turbo4/DMAD），前置条件不满足」取消（roadmap 2026-10-06 修订记录 ⑤）。

**审计发现：前置条件的消失路径是已知的，且权重极易获取。**

- **Kijai rank-19 版权重已在 HF 公开**（本审计联网核实）：`Kijai/MiniMax-H3_comfy` 仓库 `loras/minimax_h3_taomate_3step_lora_avg_rank_19_bf16.safetensors`，**182MB，BF16**，2026-09 底上传（~9 天前 verified）。这正是重分析点名「更适配本仓」的版本——本仓 `h3_lora_parse_name` 显式拒绝官方全量权的 `token_refiner` 模块，Kijai 转换版规避了该解析坑。
- 上游权重（TaoLiveAIGC/TaoMate-H3，rank 128 FP32）也公开，但会被本仓 parser 拒绝，**不要走这条**。
- 旁证：Kijai 同仓库还有 DMAD rank39 版（367MB，2 天前上传），说明该 LoRA 分发渠道活跃维护。
- 获取成本：`hf download` 单文件 182MB → scp 到 238，**<0.5h**。
- 实验成本：V2 设计早已就绪（高步主体 + taomate 3-step @0.7 强度收尾；注意坑①基座版本强耦合——stock pruned 保色 42.8 vs Singularity 掉色 28.6，本仓用 stock pruned fl2va 正好是对的那个）。238 机时 ~1–2h（2–3 臂 × repeat 2 + waffle）。

**裁决：【重开】**——「缺权重」取消理由自 2026-09 底起不成立（权重上传时间早于 V2 取消日 10-06，属调研漏网）。V2 恢复为可执行项，总成本 ~2h（含下载）。它治的正是 turbo4 已知痛点（比金线糊/过饱和），且与 T3 质量档同向。

---

## 6. T3-c cold-clip 间歇高噪根因定性

**原结论：**「cold 采样轨迹间歇性数值失稳，非 VAE decode 专属、非护栏/sol 引入——纯 cudnn dense 基线 cold 也全帧>3」；方向 B 三候选（vae_th 竞态 / pinned restore / INT8 make_room）未施工。

### 审计点 ①：定性证据是否充分排除 VAE/护栏嫌疑？

**排除 sol/护栏：充分。** 纯 cudnn dense 臂（无 sol、无护栏）cold 也全帧>3，这两条嫌疑洗清是硬证据。

**排除 VAE/INT8：不充分。** 注意「纯 cudnn dense」换的是 **SDPA 后端**，而 cold 臂仍然完整经过：① INT8 权窗 DiT 栈（make_room 驱逐逻辑在 cold 首 clip 的显存布局与 hot 不同——lightwan-h3-speed 有前科：`make_room dropped=13 window=16` 就是 step1 后碎片化触发）；② VAE decode（cold 走首次权重加载 + 可能不同的 tile/pinned 路径，`vae_th` prewarm 线程是在 dit_step4 创建的，cold clip 的时序与 hot 本质不同）；③ pinned restore（10-04 §8.7 记录 restore 失败→full reload 的行为差异）。**「全帧>3」证明的是「不是 sol/护栏引入」，但「cold 采样轨迹数值失稳」这个定性把锅默认给了采样轨迹本身，却没有一个「cold 轨迹 checksum 跨 run 稳定性」实验把 VAE/权窗变量钉死**——方向 B 三候选恰好就是这三个未排除项，说明原文其实也承认定性未闭环（所以才有方向 B）。**结论：定性方向正确但措辞越界**——应表述为「已排除 sol/护栏；cold 间歇高噪的剩余嫌疑集中在 cold-only 路径差异（vae_th 时序 / pinned restore / INT8 make_room），采样轨迹本征失稳 vs 这三者尚未分离」。

### 审计点 ②：方向 B 三候选施工成本

| 候选 | 机理 | 成本预估 | 依据 |
|---|---|---|---|
| **pinned restore** | block_gpu=1 时 host 权重指针 NULL→restore 必失败→full reload，cold 行为差异 | **≈0（已闭环/已交付）** | 2026-10-06 T4-c 施工已定位根因并判定「不可闭合但已接受」——restore 路径已 fallback 到 GPU 直载，cold 侧行为已确定化；此项实际已完成，从方向 B 移除 |
| **vae_th 竞态** | prewarm 线程 at=dit_step4 创建，cold clip 与 DiT 尾部/clip 间消费存在时序敏感（`H3_VAE_DECODE_PIPELINE` 已证时序敏感前科） | **0.5–1 天**（加 per-slot done-event 已在 P0 有成熟模式可抄；主要是复现 cold 非确定 + 插桩定位） | P0 施工档案 |
| **INT8 make_room** | cold 首 clip 显存碎片化触发 `make_room` 驱逐窗（前科 window 29→16），窗变化改变 cold 轨迹的数值路径 | **2–3 天**（freeze window 已有先例 p3 `dit_int8_cache_freeze`，但 cold 路径的 freeze 时机 + 验证 cold checksum 跨 run 稳定需要 repeat≥5 的 cold 专项） | lightwan-h3-speed §1/§3 |

**建议顺序**：vae_th（便宜且 P0 模式成熟）→ INT8 make_room（收益最大但最贵）。**最小诊断实验先行**：cold clip 跑 repeat≥5，只记 cold checksum 逐 run 值 + `make_room`/`vae_th` 日志行——若 checksum 跨 run 稳定但 waffle 仍间歇>3，则锅在 VAE decode 数值而非 DiT 轨迹，方向 B 直接缩到一候选（~1h 238 机时）。

**裁决：【修正】**——定性措辞收窄为「已排除 sol/护栏，剩余三候选未分离」；方向 B 更新为两候选（pinned restore 已闭环移除）+ 前置 1h 诊断实验。

---

## 7. 金线换锚的可靠性

**原结论：** 旧金线作废（sol 全稀疏臂产物），新金线只认 hot clip2=`0x923b0db3a8c0357e`；hot total≈19.2–19.4s（含 cold 噪声）。

**审计：**

- **checksum 口径：无跨日噪声问题。** `0x923b0db3a8c0357e` 取自 **hot clip2**，cold 噪声物理上不进入该值；且 P0 验收（10-06）Release(−O3) 4-run 复验 hot `0x923b0db3…` 多 run 逐位一致（护栏臂 async-on 口径内）。「hot total 含 cold 噪声」的问题**不影响 checksum 金线本身**。
- **但随后的反例要登记**：10-06 final-accept 发现 VAE cuBLAS/cuDNN heur 跨进程漂移，hot clip2 checksum 4 次运行全不同（`0x26bb`/`0x6574`/`0x6831`/`0x5a49`）——即**「跨 run 逐位一致」在当前 binary 上已物理不可达**，T2 验收口径已改判「waffle 区间稳定 + 无 NaN」。这意味着 `0x923b0db3a8c0357e` 作为「逐位金线」的有效期止于 P0 翻默认时的 binary；**它现在是「waffle 区间锚（0.75–0.95）」而非「checksum 锚」**——roadmap §0.1 与 §6 的表述仍把它写成 checksum 金线，措辞滞后于 final-accept 的自己。
- **速度口径：确有口径污染。**「hot total≈19.2–19.4s（含 cold 噪声）」这个量把 cold 段的间歇失稳方差算进了「hot」数里，与 P0 后的 17.72s（hot clip2 total 口径）并存，正是 Sol-Attn 审计点名的「多口径并置易误读」实例。**建议速度金线统一改为 hot-only：`phase_ms total=` 的 hot clip2 值，repeat≥3 取 P50**（cold 只进 cold guard 护栏口径，不进速度账本）。

**裁决：【维持 checksum 锚 + 修正两处口径】**——① 新金线选取过程无跨日噪声问题（hot clip2 天然免疫）；② 措辞更新：`0x923b0db3a8c0357e` 降级为「waffle 区间锚」，checksum 逐位金线口径按 final-accept 判物理不可达处理；③ 速度金线改 hot-only P50 口径。

---

## 8. 行动项汇总（按最小成本排序）

| 序 | 动作 | 服务条目 | 成本 | 预期产出 |
|---|---|---|---|---|
| A1 | **人审签字 DMAD 成片**（`out/review-2026-10-04/kpib-dmad-kj/C_full_critic/`） | #3 | 人审 ~15min | T3 整档解锁（唯一剩余门） |
| A2 | **cold 诊断实验**：cold clip ×5 记 checksum + make_room/vae_th 日志 | #6 | ~1h 238 | 方向 B 缩到单候选 |
| A3 | **V2 重开**：下载 Kijai rank-19 taomate（182MB）+ refiner A/B | #5 | ~2h（含下载） | 治 turbo4 过饱和的实测答案 |
| A4 | **跨步缓存 @ turbo4 实测封档**：解锁 turbo4 preset 的 magcache 拒绝，保守阈值 A/B | #1 | 2–3h 238 | 推理封档→实测封档（或小概率白捡 T1 收益） |
| A5 | roadmap 文本回填：#2 的 4 处差异 + #7 的两处口径 | #2/#7 | 纯文档 ~0.5h | 消除多口径并置 |
| A6 | （可选）FastH3-8-Step-V2 外部质量参照（ComfyUI 对照金线 8-step） | #4 | ~1 天 | 8-step matched 重测条款闭环 |
| A7 | （可选）DMAD 今日 binary 复跑刷新 P0 后口径 | #3 | ~0.5h 238 | +11% 口径刷新 |

---

## 9. 修订记录

| 日期 | 变更 |
|---|---|
| 2026-10-07 | 初稿：7 条质量/模型/缓存封档结论逐条重审。裁决：维持 2（DMAD、金线 checksum）、修正 4（跨步缓存、sol 措辞回填、FastH3 表述、T3-c 定性收窄）、重开 1（taomate V2）。联网补查：MagCache 论文 Table 5（FusionX 蒸馏兼容性正例）、Kijai HF（taomate rank-19 / DMAD rank39 权重现货）。 |
