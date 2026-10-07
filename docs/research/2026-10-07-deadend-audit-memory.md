# 显存/系统类死路重审：768P 32GB 墙系列（2026-10-07）

> **触发：** Sol-Attn 审计发现「负面结论过度泛化」问题；VDN 重分析（`2026-10-07-vdn-direct-use-reanalysis.md`）坐实 **VDN 恒定内存特性可在 768P 32GB 不 OOM（INT8 栈剩 8.3GB）**——直接挑战本仓「768P 32GB 显存墙」系列结论。本档对 5 条显存/系统类死路逐条重审：原结论 → 证据 → 裁决【维持/修正/重开】→ 理由。
> **必读源：** `2026-10-01-h3c-nvidia-rtx5090-sol-h3-analysis.md` §8.4/§8.5（全 sol OOM 三版释放全败、native 28.8GiB 账本）、`2026-10-05-h3-optimization-roadmap.md` §0.2/§3.4/§5、`2026-10-07-vdn-direct-use-reanalysis.md`（VDN 768P 不 OOM 实证）、`2026-10-07-llm-to-dit-transfer.md` §4.3（SGLang 视频模型 layerwise streaming VRAM −30GB + PR#36727 同模型同卡指针重绑定）。
> **裁决纪律：** 只判死路条目本身的成立性，不改 KPI 目标数字；推翻某条只意味着「该条不再作为不做清单的挡箭牌」，落不落地另立施工项。

---

## 条目 1：768P 全 sol 32GB OOM 墙

**原结论（§8.4/§8.5）：** native INT8 窗 + 激活占 28.8GiB（其中 torch 仅 2.08GiB），sol fwd 连 530MiB 挤不出；两版释放策略（末层/每层 drop+trim）全 OOM；**不减窗红线下全 sol 不可行/不划算，退回混合臂**。

**逐审计点核对：**

| 审计点 | 核对结果 |
|---|---|
| ① 28.8GiB 逐项实测分解？ | **没有。** 现有只有「torch 2.08 / 非 torch ~28.8」二分 + w_i8 ~11GiB 单项。DiT 激活 workspace、cudnn workspace、sol QKVO ~5.5GiB 与 per-fwd 临时张量、allocator 碎片**从未各自实测**。「激活占满」是余额推定，不是测量。 |
| ② 窗 30/34「不减窗红线」语义？ | **窗 = INT8 权重缓存的滑动窗口大小（同时驻留显存的 DiT 层数，`h3_w4a8_cache.window_size`），不是注意力窗。** 「减窗禁」= 禁止把驻留层数压到 24 以下——因为窗小了 H2D 权重换入换出频繁，DiT 主墙变慢。但注意：**窗 30→24 释放的 ~6 层 × ~360MiB ≈ 2.2GiB，并未被作为「换 sol 高分出片能力」的等价交换定量评估过**——只被红线一刀切。 |
| ③ layerwise weight streaming 评估过？ | **从未。** LLM-to-DiT 迁移文档 §4.3 首次把它列为候选（SGLang Diffusion 默认开启、视频模型 VRAM −30GB、PR#36727 同模型同卡实证指针重绑定）；§8.5 的「offload 会让高分 sol 比 cudnn 慢」是**无实测的先验假设**，与 SGLang「性能反升 58%（prefetch 重叠良好）」直接冲突。 |
| ④ 激活 FP16 中转（sm120-frontier M2，腾 2–4GiB）试过？ | **没有。** 激活当前按 INT8 路径物化；FP16 中转省显存（把 quant/dequant 的 INT8 中转 buffer 直接以 FP16 复用）从未施工。2–4GiB 量级恰好覆盖 sol fwd 缺口（530MiB）+ QKVO 常驻。 |
| ⑤ 28.8GiB 里「结构性必须」vs「实现选择」？ | VDN 同卡同分辨率 INT8 栈剩 8.3GB（峰值 ~23.7GB）——本仓 28.8GiB 比 VDN 栈**多 ~5GiB**。差值来源候选：INT8 权重窗常驻 30 层（VDN 也是全量 INT8 权重但无 sol QKVO 高分 ~5.5GiB 一次性 torch 分配）、cudnn workspace、以及本仓 WS-first 预生长的高分 workspace。**未分解前无法判多少是必须。** |

**裁决：【重开】。**

**理由：** 原结论的证据链是「两版释放策略（都只动 torch 侧 2GiB）失败 → 判全 sol 不可行」。这是**对释放策略族的否定，被过度泛化为对全 sol 路径的否定**——正是本次审计要纠的「负面结论过度泛化」同型错误。三个从未评估的候选（layerwise streaming、激活 FP16 中转、窗 30→24 等价交换）任意一个都可能解开 530MiB 缺口；VDN 实证同卡 768P INT8 栈可以剩 8.3GB，说明 28.8GiB 账本里存在可腾挪的实现选择。「不减窗」红线的代价（放弃全 sol）从未与收益（保住 H2D 换入开销）放在同一杆秤上。

**重测/验证成本：** ① 显存账本逐项插桩（w_i8 / DiT WS / cudnn WS / QKVO / fwd 临时分桶，`cudaMemGetInfo` 在关键节点打点）：**~0.5 天**，纯插桩无风险。② 窗 24 + 全 sol 对照臂（解红线做一次性实验，不翻默认）：**~0.5 天**（改 env + 跑一次）。③ 激活 FP16 中转探针（sm120-frontier M2 项）：**~2–3 天**施工 + 位一致/质量门。④ layerwise streaming 原型（逐层 H2D prefetch 与计算重叠）：**~1–2 周**，是四项中最贵的，建议排在 ①② 给出新账本后再定。

---

## 条目 2：VAE 常驻 keep 死路（与 INT8 cache 冲突 OOM）

**原结论（§0.3/§5）：** `H3_VAE_REPEAT_RELEASE=keep`（VAE 常驻）在 clip2 `adaln_dev out of memory` 崩溃（INT8 window 2/3/8 全崩），与 INT8 cache 冲突；**不做 VAE 常驻 keep**。

**证据核对：** 该结论的实验语境是 **480P clip2、P0 async 流水落地之前**——当时 clip 间 `vae_release_device` 卸 device W（host_pack=kept mmap_pin），keep 档把 ~1.8GiB 设备权重常驻后，clip2 高分 DiT 激活（adaln_dev）OOM。P0 落地后 VAE 权重生命周期已改造（prewarm hostreg + device drop/restore 分离），但**判 keep 死路的那次实验从未在新生命周期下重跑**。另外注意原实验把 INT8 window 压到 2/3/8 全崩——这说明 OOM 的边际贡献者是 keep 的 1.8GiB，而窗 2 时 w_i8 只占 ~1.4GiB、整机 native 占用远小于 768P 高分态，**keep 死在 480P 的账本与 768P 的账本不是同一本**。

**裁决：【维持，但限定适用范围】。**

**理由：** ① 对 **480P 默认路径**，keep 的收益（省 clip 间 vae_load ~0.3s）在 P0 async + prewarm 落地后已被 host 侧暖载基本隐藏（热 work_ms=0.0），重开动机弱；② 对 **768P**，keep 的 1.8GiB 在 hot clip2 free=3.11GiB 下仍挤 sol_cute（见条目 3），维持不 OOM 判定；③ 但「与 INT8 cache 冲突 OOM」的表述应限定为「在当前 INT8 窗策略 + 32GB 下」，不能泛化为「VAE 常驻这个方向死」——若条目 1 的显存账本重测腾出 headroom，keep 应连带重估。**维持死路标签，但加限定语 + 与条目 1 联动重估。**

**重测/验证成本：** 若条目 1 ①的账本插桩落地，keep 重估只需在 768P 混合臂跑一次 keep 对照：**~0.5 天**。不单独施工。

---

## 条目 3：T4-c vae_load 0.3s 余尾「不可闭合」

**原结论（§7 修订记录 2026-10-06）：** 消除余尾需 `vae_weights_device_drop` 保留设备权重 ~1.8GiB，但 hot clip2 free=3.11GiB 保留后只剩 ~1.3GiB，触发 sol_cute OOM；**接受 ~0.3s 余尾为已知限制，不可闭合**。

**证据核对：** ① 账本时刻是「hot clip2 free=3.11GiB」——这是 480P 混合臂热态的瞬时值，不是 768P 的（T4-c 属 768P 档位，768P 热态 free 另有其数）；② 两个从未评估的减压阀：**INT8 窗策略调整**（同条目 1②，窗 −6 层腾 ~2.2GiB）与 **VAE 设备权重 FP8 存储**（1.8GiB→0.9GiB——VAE 当前 fp16 未量化，FP8 存储 + 解码时 dequant 是单向往返，质量门可测）；③ SGLang PR#36727 的指针重绑定手法（host 保 mmap 视图、swap 只重绑指针、消重复 D2H 4.85s→~0）对**同模型同卡**已实证，本仓 restore 路径（`host_pack_restore_fail` → full reload）正是被该手法淘汰的形态。

**裁决：【修正】。**

**理由：** 「不可闭合」的成立前提是「保留 1.8GiB 设备权重是唯一消余尾手段」——这是实现选择，不是物理约束。FP8 存储（0.9GiB，free 3.11GiB 下剩 2.2GiB，sol_cute 单次 fwd 需求 530MiB 级，可行概率高）与 PR#36727 指针重绑定（不增常驻显存、消的是 restore 失败的 full reload）都未被评估即判死，属过度泛化。修正为：**「在当前 INT8 窗策略 + FP16 设备权重下不可闭合」**，两个候选手段列入重开观察。注意 0.3s 本身量级小（占 T4 ~0.4%），重开价值主要不在秒数而在**restore 路径与 PR#36727 手法的对齐**（连带影响 keep 死路与 prewarm 语义）。

**重测/验证成本：** ① VAE 设备权重 FP8 存储 + 解码 dequant：**~1–2 天**（改存储格式 + dequant kernel + waffle 质量门）。② 指针重绑定式 restore（host mmap 视图 + 指针重绑）：**~1 天**（改 `vae_weights_device_restore` 失败路径）。均可独立施工。

---

## 条目 4：768P 混合臂为唯一稳定路径

**原结论（§0.2/§8.5）：** 32GB VRAM 硬墙下全 sol OOM，**混合臂（低分 sol + 高分 cudnn）是唯一稳定出片路径**。

**证据核对：** 该结论的「唯一」建立在两个现已动摇的前提上：① 全 sol OOM 墙（条目 1 已重开）；② 没有第三条路——但 VDN 实证给出了另一条：**VDN 恒定内存栈在同卡 768P 不 OOM（剩 8.3GB）**，虽然 e2e ~135–150s 比混合臂 68.93s 慢 ~2×（8 NFE vs 4 NFE + 纯 PyTorch eager 栈），它证明了「768P 高分注意力全量走非 cudnn 后端且不 OOM」物理可行。混合臂的「唯一稳定」应改为「当前最快稳定」。

**裁决：【修正】。**

**理由：** 「唯一稳定路径」是能力命题，VDN 证据直接证伪其唯一性（VDN 栈稳定出片、不 OOM）。修正为：**「混合臂是当前 4-step 合同下唯一满足速度目标的稳定路径」**——加上速度约束后原结论成立，去掉则过度泛化。联动条目 1：若全 sol OOM 墙重开后解开（layerwise streaming / 激活 FP16 / 窗等价交换任一兑现），混合臂的「唯一」连速度约束下都不保。

**重测/验证成本：** 零新增施工——本条目是表述修正，落实在条目 1 的重测项里。

---

## 条目 5：layerwise/CPU offload「低优先」判定

**原结论（sol-h3 分析 §5 迁移表）：** 「本仓 INT8 权窗已解决 32GB 装载，offload『从不能跑到能跑』不是主矛盾」→ **低优先**。

**证据核对：** ① 原判定的语境是 **480P 装载问题**——INT8 权窗确实已解；但在 **768P 全 sol 解 OOM** 语境下，offload 的对象不是「装载」而是「高分态的常驻峰值」（QKVO ~5.5GiB + INT8 窗 30 层 ~11GiB + 激活），这正是 SGLang Diffusion layerwise offload 的主战场（视频模型 VRAM −30GB 实证）；② §8.5 把 offload 贬为「会让高分 sol 比 cudnn 慢」是无实测先验，SGLang 报 prefetch 重叠良好时**性能反升 58%**——5090 PCIe 5.0 x16 ~64GB/s，逐层流式 33B W4 权重 ~18GiB / 4 步的理论下限 ~4.5s 的前提是完全不重叠，而 DiT 每层计算 ~200ms 级，prefetch 下一层（~360MiB，PCIe ~6ms）**重叠空间 30 倍**，「必慢」不成立；③ offload 的价值轴被判错：它不是速度杠杆，是**能力杠杆**（解锁全 sol@768P）——用速度轴判低优先是范畴错误。

**裁决：【重开】。**

**理由：** 原判定犯了两个错：① 范畴错置——把能力杠杆当速度杠杆判优先级；② 无实测先验——「offload 必慢」与 SGLang 实证（−30GB VRAM 且 +58% 性能）冲突。重开后的正确定位：**offload 是条目 1（768P 全 sol OOM 墙）的候选解法之一**，优先级随条目 1 重开而解除冻结。是否落地取决于条目 1 ①的显存账本重测——若窗等价交换或激活 FP16 已能解 530MiB 缺口，offload 不必上（它改动面最大）；若不能，offload 是保底解。

**重测/验证成本：** 原型施工 **~1–2 周**（逐层 H2D prefetch + 与计算重叠 + 窗策略联动）；但**前置只需条目 1 ①的 0.5 天账本插桩**即可定是否值得动工。建议顺序：账本插桩 → 窗 24 对照 → 激活 FP16 → 都不够才上 offload 原型。

---

## 汇总裁决表

| # | 条目 | 裁决 | 一句话理由 |
|---|---|---|---|
| 1 | 768P 全 sol 32GB OOM 墙 | **重开** | 两版释放失败只否定 torch 侧释放策略族，被过度泛化为全 sol 否定；layerwise streaming/激活 FP16/窗等价交换三候选从未评估；VDN 实证同卡 768P INT8 栈剩 8.3GB |
| 2 | VAE 常驻 keep 死路 | **维持（限定范围）** | P0 后重开动机弱、768P 仍挤 sol_cute；但限定为「当前 INT8 窗策略 + 32GB 下」，与条目 1 联动重估 |
| 3 | T4-c vae_load 0.3s 余尾不可闭合 | **修正** | 「保留 1.8GiB 设备权重是唯一手段」是实现选择非物理约束；FP8 存储（0.9GiB）与 PR#36727 指针重绑定均未评估 |
| 4 | 768P 混合臂为唯一稳定路径 | **修正** | VDN 栈稳定出片证伪「唯一」；修正为「当前 4-step 合同下唯一满足速度目标的稳定路径」 |
| 5 | offload 低优先判定 | **重开** | 范畴错置（能力杠杆当速度杠杆判）+ 无实测先验（SGLang −30GB/+58% 实证冲突）；随条目 1 解除冻结 |

**联动动作（按成本升序）：** ① 768P 显存账本逐项插桩（0.5 天，解锁全部后续决策）→ ② 窗 24 + 全 sol 一次性对照（0.5 天，红线等价交换定量）→ ③ T4-c 指针重绑定 restore（1 天）→ ④ 激活 FP16 中转探针（2–3 天）→ ⑤ VAE 设备权重 FP8 存储（1–2 天）→ ⑥ offload 原型（1–2 周，保底解，前五项不够才动工）。
