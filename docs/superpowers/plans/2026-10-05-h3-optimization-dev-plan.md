# H3.c 优化开发方案（施工级）：速度轨 P0–P3 + 质量轨 V0–V4

> **日期：** 2026-10-05。**机器基准：** 192.168.0.238，1×RTX 5090 32GB，sm_120，CUDA 13。
> **定位：** 本文是 `2026-10-05-h3-optimization-roadmap.md`（目标书）的**施工级落地文档**——读者是接下来动手写代码的工程师（可能是另一个代理）。roadmap 回答「做什么、做到多少、为什么」；本文回答「**在哪个文件哪个函数动手、按什么步骤动、动完怎么验证、出问题怎么回退**」。
> **数据纪律继承 roadmap：** 所有「实测」数字锚定 238 真机日志；所有「预估」标注依据；不假数字、不承诺 10s/27s；每拍只动一个变量。
> **代码定位纪律：** VAE 源码可能在 238 远程机器上，本地副本若缺对应函数，一律以 roadmap/审计文档中**真实出现过的函数级归因**（`decode_loop`、`unpatch_d2h`、`h3_vae_int8_frames`、`qkv_rope`、`adaln`、`rope_tables_f32` 等）为准，**不编造函数名**。本地已核实的行号会标注「本地核实」。

## 0. 施工状态快照（2026-10-05 实测回填，先于一切施工阅读）

> 本节为同日第二批回填——下列项的实测结论已改变原施工序列，**动手前先读本节**。

| 项 | 回填结论 | 对施工的影响 |
| --- | --- | --- |
| **P0 VAE pinned** | **半刀已落地、零提速**：pinned 真实命中（插桩证实 `h3_cuda_host_alloc` 走 `cudaHostAlloc` 非 fallback malloc），但 480P unpatch_d2h **3227ms 与 P0 前（3.1–3.3s）持平**。根因：3227ms 的 94% 是逐 tile 串行同步 `cudaMemcpy` 的 D2H 等待（229ms/tile × 13）；roadmap 原记的 ~1.6s 串行尾是**已从 binary 删除的 stitch-pipeline 实验臂**（kpia-vae-pipeline，unpatch_d2h=195ms 为 VAE 藏进 DiT 影子口径）的错配基线 | §2.1 施工范围收窄为「打开 `H3_VAE_UNPATCH_ASYNC`（当前默认关）+ 跨 tile 流水」；pinned 生命周期改造已完成，**不得回退** |
| **P1 QKV 融合** | **闭环零收益**：`h3_dit_w4a8.c` 里 QKV 本就 `[3*INNER, HIDDEN]` 宽 GEMM 合并、QKNorm+RoPE 本就在 CUDA 端融合，位一致探针 **MAE=0** | **§2.2 整节作废（保留作死路档案），不再施工** |
| **P2b CUDA Graph** | **已交付**：`H3_DIT_W4A8_CUDA_GRAPH`（默认 0）整步单次捕获/replay，238 hot clip2 **位一致**；但 hot DiT 12.61→12.63s 持平、e2e 仅 −0.16s（turbo4 仅 4 步，launch 税被 cold 噪声掩盖） | **§2.3 施工完成，不翻默认**；保留为可选开关 |
| **金线 checksum** | 旧 clip2=`0xce0b5df2bcdb4ad3` 系 sol 全稀疏退化臂产物，**作废**；新金线=护栏臂（sol + 缺省 dense_steps=2/layers=2）：**hot clip2=`0x923b0db3a8c0357e`**，waffle mean=0.83（帧 0.750/0.803/0.851/0.816/0.935）；cold clip1 间歇（`0x2d5c52020bdfcc70` / graph 臂 `0x5e6b2f7bfe154e0b`）不作金线 | 本文所有「checksum 逐位一致」验证门的对照值以新金线为准 |
| **cold 高噪** | 根因定性：cold 采样轨迹间歇性数值失稳（纯 cudnn dense 基线 cold 也全帧>3），**与护栏/sol/VAE 无关**；方向 A 已落地：`H3_WAFFLE_COLD_GUARD`（默认 0）+ `H3_WAFFLE_COLD_MAX`（默认 2.0），cold 超阈值 WARN 不 FAIL，registry PASS；方向 B（vae_th 竞态 / pinned restore / INT8 make_room）待办 | V0/V1 的 A/B 设计需用护栏臂作产品默认对照（§3.1/§3.2 已更新） |
| **V0/V1** | **已闭环**：sol 全稀疏@turbo4 判退化退出默认；护栏臂接管默认路径 | §3.1/§3.2 标记已闭环，残余动作仅为 8-step 档评估 |

---

## 1. 总览

### 1.1 双轨结构

本方案分两条**并行施工轨**，互不阻塞、共享验收纪律：

| 轨 | 序列 | 性质 | 依据 |
| --- | --- | --- | --- |
| **速度轨** | ~~P0 / P1 / P2b~~ / P2 / P3（**2026-10-05 回填：P1 闭环零收益、P2b 已交付收益微薄，双双移出施工序列；有效序列 = P0-async / P2 / P3**） | 纯工程刀（机理族#6 + 族#4 精确不变量缓存），外部已证原理，无质量杠杆 | sol-h3 官方重分析、12 方法机制深拆、不变量缓存审计 |
| **质量轨** | ~~V0 / V1~~ / V2 / V3 / V4（**回填：V0/V1 已闭环**——V0 判退化、护栏臂接管默认） | 主栈成立性验证（V0/V1，已闭环）+ 质量补齐（V2/V3）+ 机制演进（V4） | 社区全景（FastWan 冲突）、taomate/VDN 重分析 |

**两轨的交互点只有一个：** V0（sol_cute@turbo4 有效性复核）的判定结果决定 sol_cute 能否留在默认产品路径——**2026-10-05 回填：V0 已判退化，护栏臂（sol + 缺省 dense_steps=2/layers=2）已接管默认路径，此交互点已闭环**。

### 1.2 全局约束（不可逾越）

1. **四档 SKU 目标**（roadmap §1，本文一切施工以达此为目的；**2026-10-05 回填可达性注记见各项末尾**）：
   - **T1 极速档**：480P 热 e2e ≤13.5s，waffle ≤2.5 + av_sync PASS，豁免 bit-identical（记漂移量）。**回填：当前不可闭合（已兑现落点 15.8s；async 落地后 14.2–14.6s 仍缺 0.7–1.1s，需新收益源）——冻结为依赖项。**
   - **T2 均衡档（默认产品）**：480P 热 e2e ≤15.0s，latent checksum **逐位一致**（**新金线只认 hot clip2=`0x923b0db3a8c0357e`，护栏臂；旧 clip1/clip2 值作废**）+ waffle 0.75–0.95（新锚）。**回填：当前闭合落点 16.0–16.2s，达标完全依赖 P0-async 落地且 bit-identical。**
   - **T3 质量档**：waffle ≤1.35 + 人工 review 签字（DMAD full_critic 成片已在 `out/review-2026-10-04/kpib-dmad-kj/`），**等人工，不在本方案施工范围**。
   - **T4 分辨率档**：768P 热 e2e ≤50s（从 P50 56.47s 起），checksum 逐位一致（混合臂基线 `0x63987d3e8083e8c1`）+ waffle 0.75–1.05 + 不 OOM、不减窗（≥24）。**回填：条件性目标——T4-a 收益与 480P 共享 P0-async 前提（未兑现），T4-b/T4-c 未施工；当前落点 56.47s。**
2. **金线 checksum 不可动（2026-10-05 换锚）：** **hot clip2=`0x923b0db3a8c0357e`**（护栏臂，唯一金线 checksum）；cold clip1 间歇（`0x2d5c52020bdfcc70` / graph 臂 `0x5e6b2f7bfe154e0b`）因 cold 轨迹非确定性**不作金线**；旧值 clip1=`0x28e8a880d52e9be1` / clip2=`0xce0b5df2bcdb4ad3`（sol 全稀疏退化臂产物）**作废**。任何声称 bit-identical 的刀必须逐位复现 hot clip2 新金线值。
3. **不动 INT8 GEMM 本征**：cublasLt 已达硬件峰值 98–99%（C5b 三层验证闭环），不 autotune、不手写 sm_120 核、不做 FFN epilogue vec4（对齐坑前科）。
4. **不承诺 10s / 27s**：数学不闭合（roadmap §3.1/§3.4），写入任何文档即违约。
5. **每拍只动一个变量**：一次施工拍只改一把刀；对照臂与实验臂同 seed、同机、同 binary（仅开关差异）。
6. **新开关默认 OFF**：所有新刀按 `H3_*` 惯例注册进 `tools/h3_plugin.h` 的 `H3_PLUGIN_BUILTIN_DEFS`，默认 OFF 零行为变化，`tools/check_plugin_registry.sh` 必须 PASS。

### 1.3 执行依赖（资源约束）

| 依赖 | 影响范围 | 说明 |
| --- | --- | --- |
| **独占 GPU 窗口** | P0-async（必需） | P0 的竞态调试对时序敏感——拥挤 GPU 下位精确验证已失败过一次（`H3_VAE_DECODE_PIPELINE` 实验）。~~V0/V1（必需）~~ **V0/V1 已闭环（2026-10-05），不再需要窗口**。 |
| **T3 人工 review** | T3 整档 | DMAD 成片待人审签字，本方案不覆盖。 |
| **方法论审计报告** | T1-e / T2-c / T4-e | [审计依赖] 项不在本方案主动施工，仅登记触发条款。 |
| **VAE 源码位置** | P0 / P2 | `h3_vae_int8_cuda.c` 本地副本存在（已核实关键符号），但完整施工与 profile 需在 238 上进行；本地行号仅供预读，施工前以 238 工作副本为准。 |

---

## 2. 速度轨施工方案

> 每把刀一节，统一结构：**目标与收益 → 代码级定位 → 施工步骤 → 验证门 → 风险与回退 → 插件接线**。
> 施工顺序与文件冲突详见 §5；本节按原优先级 P0→P1→P2b→P2→P3 排列（**2026-10-05 回填后：P1 已闭环存档、P2b 已交付，有效施工节仅剩 P0-async / P2 / P3**）。

### 2.1 P0：VAE unpatch_d2h async + 跨 tile 流水（最高优先；pinned 半刀已落地）

> **2026-10-05 回填状态：** pinned 部分**已施工完成且真实命中**（插桩证实 `h3_cuda_host_alloc` 走 `cudaHostAlloc` 非 fallback malloc），但实测 **480P unpatch_d2h 3227ms 与 P0 前（3.1–3.3s）持平、零提速**。根因已查明：3227ms 的 **94% 是逐 tile 串行同步 `cudaMemcpy` 的 D2H 等待**（229ms/tile × 13）——pinned 只省 staging，省不掉同步等待。**本刀剩余施工 = 打开 `H3_VAE_UNPATCH_ASYNC`（当前默认关）+ 跨 tile 流水**，收益预估（−1.3~−1.6s@480P / −3.5~−5.5s@768P）**尚未兑现，待 async 翻默认验证**。另：roadmap 原记的 ~1.6s 串行尾是**已从 binary 删除的 stitch-pipeline 实验臂**（kpia-vae-pipeline，其 unpatch_d2h=195ms 是把 VAE 藏进 DiT 影子的口径）的错配基线，**作废**；当前真实基线 = 3227ms。

#### 目标与收益

| 档位 | 收益（预估，**未兑现**——待 async 翻默认验证） | 达标贡献 |
| --- | --- | --- |
| T1（480P） | **−1.3 ~ −1.6s** | 15.9s → 14.2–14.6s 的最大单刀（**仍不足闭合 T1 ≤13.5s**，roadmap §3.1 重算） |
| T2（480P bit-identical） | **−1.3 ~ −1.6s** | 16.4s → 14.5–14.8s 的**唯一**刀（T2 实质单点依赖；当前闭合落点 16.1s） |
| T4（768P） | **−3.5 ~ −5.5s** | 56.47s → 48.2–51.2s 的双刀之一（条件性） |

**作用域限定（roadmap §4.1）：** 收益落在 VAE unpatch_d2h 的串行 D2H 等待上（480P 实测 3227ms，其中同步等待占 94%）；**不触碰 vae_blocks 主体 ~2.6s 的 fp16 真实计算**。bit-identical 可达（async 只改传输时序，不改数值）。

#### 代码级定位

- **文件：** `h3_vae_int8_cuda.c`。
- **核心结构：** `h3_vae_int8_frames`（本地核实：`h3_vae_int8_cuda.c:1542, 1693, 1785` 等处出现；批量槽位 `g_vae_batch_frames[H3_VAE_MAX_TILE_BATCH]` 在 `:165`）。
- **瓶颈路径（文档归因）：** `decode_loop` 逐 tile 串行 D2H + 主机 stitch——`st_acc.unpatch_d2h_ms` 累加点在 `:1955` 与 `:2220`（本地核实存在该统计字段），说明 unpatch_d2h 在 decode 主循环内被逐段计时、串行等待。
- **前科定位：** pinned 探索曾因 `fr->rgb` 生命周期 segfault 被否（roadmap §2.1 T1-b）；`H3_VAE_DECODE_PIPELINE` 实验已证全异步有竞态（copy worker 异步 pinned→final memcpy 与 stitch 跨 clip 消费），且已顺带固化 2 个真实 bug（GPU unpatch batch stride 误用被覆写的 `g_vae_attn_seq` → 局部 `batch_stride`；`d_rgb` 双缓冲空指针）——**这两个修复已落地，施工时不得回退**。
- **RoPE 顺带位：** chunk 循环内 `h3_vae_int8_prepare_rope_ex`（本地核实 `:2782`）每 chunk host 现算 + H2D——见 §5.1 零风险尾料，可同拍顺带。

#### 施工步骤（2026-10-05 回填修订：步骤 1 已完成，剩余 2–4）

1. ~~**pinned 标志穿进 `h3_vae_int8_frames` 生命周期**~~ **【已完成，不得回退】**：
   - pinned 分配已落地且插桩证实真实命中 `cudaHostAlloc`（非 fallback malloc）；`fr->rgb` 生命周期问题已解决。
   - **但实测证明 pinned 单独零收益**（unpatch_d2h 3227ms 持平）——它只是 async 的前置，不是收益本身。
2. **打开 `H3_VAE_UNPATCH_ASYNC`（当前默认关）+ per-slot done-event 替换全局 quiesce**：
   - 为 `g_vae_batch_frames[]` 每个槽位配一个 `cudaEvent_t done_event`；D2H async copy 提交后立即 `cudaEventRecord` 到 copy stream。
   - stitch/消费端不再全局 `cudaDeviceSynchronize`，而是 `cudaEventSynchronize(slot->done_event)` 按槽就绪。
   - 淘汰 `H3_VAE_DECODE_PIPELINE` 时代的「全局 quiesce」模式（竞态根源）。
3. **跨 tile 流水：D2H 与下一 tile GPU decode 重叠**：
   - decode kernel 跑在主 stream；D2H copy 跑在独立 copy stream，用 event 建立「tile N decode 完成 → tile N D2H 开始」与「tile N D2H 与 tile N+1 decode 并行」的依赖边。
   - 主机 stitch 线程用 per-slot event 就绪后即消费，不阻塞 GPU 侧下一 tile。
   - **收益判据：消除 229ms/tile × 13 的串行同步等待**——这是 3227ms 的 94%。
4. **480P 验证通过后移植 768P 放大版**：
   - 768P 为 6 批 × ~0.35GiB/批 D2H；注意 host 侧 pinned 内存压力（见风险）， pinned 总量超阈值时降级为「前 K 批 pinned + 其余 pageable」混合策略。

#### 验证门

- **速度门：** 热 clip2 `phase_ms total=` × ≥2 次复现；480P 目标 unpatch_d2h 从实测 **3227ms** 压到亚秒级（消除 229ms/tile × 13 串行同步等待）；768P 等比消除串行等待。run-to-run 噪声 ±0.5s（480P）/ ±3s（768P）。
- **质量门：** latent checksum **逐位一致**（**新金线：hot clip2=`0x923b0db3a8c0357e`**；768P 混合臂基线 `0x63987d3e8083e8c1`）+ waffle 0.75–0.95（480P 新锚）/ 0.75–1.05（768P）。
- **稳定性门：** av_sync PASS、sol 400/400 fallback=0、无 OOM、repeat=2 连跑 exit=0、host pinned 占用可观测无泄漏。

#### 风险与回退

| 风险 | 前科/依据 | 回退方案 |
| --- | --- | --- |
| ~~`h3_vae_int8_frames` 生命周期 segfault~~ | **已解决（pinned 半刀落地，插桩证实命中 `cudaHostAlloc`）** | pinned 部分已在默认路径，不得回退 |
| 异步竞态（stitch 跨 clip 消费 vs copy worker） | `H3_VAE_DECODE_PIPELINE` 实测 ON 未达标、拥挤 GPU 位精确验证失败；`H3_VAE_UNPATCH_ASYNC` 当前默认关即因此 | per-slot done-event 即回退方案本身；若仍不稳，降级为「tile 内 pinned + tile 间同步」保守档（= 现状，零收益但位精确） |
| 768P pinned 内存压力（6×0.35GiB host） | 预估 | 混合策略（前 K 批 pinned）；最差接受 480P 收益、768P 仅部分落地，T4 目标相应下修（roadmap §3.4 允许 a+b 下限 −5.0s 触边） |

#### 插件接线

- 开关：`H3_VAE_UNPATCH_ASYNC`（已存在，**当前默认 0**）——本刀的核心动作就是让它在 per-slot done-event 就位后翻默认。
- pinned 分配已落地（`h3_cuda_host_alloc` 真实命中 `cudaHostAlloc`），无需新增开关。
- 注册进 `tools/h3_plugin.h` 的 `H3_PLUGIN_BUILTIN_DEFS`，默认 OFF；`tools/check_plugin_registry.sh` PASS。
- 保留 `H3_VAE_D2H_ASYNC_COPY` / `H3_VAE_DECODE_PIPELINE` 实验项语义不动（历史开关，不删）。

---

### 2.2 ~~P1：QKV merge + QKNorm+RoPE 融合~~（**已闭环：零收益死路，不再施工**）

> **2026-10-05 回填结论（本节保留作死路档案）：** P1 探针实证——`h3_dit_w4a8.c` 里 QKV **本就** `[3*INNER, HIDDEN]` 宽 GEMM 合并、QKNorm+RoPE **本就**在 CUDA 端融合；位一致探针 **MAE=0**。本节假设的「3 次独立 QKV 瘦 GEMM + 独立 RoPE 核」结构前提**不成立**（去伪存真：该结构描述的是逻辑视图，CUDA 端物理实现已合并）。**原 −0.6~−1.2s 预估作废，CUDA 端无收益可挖，本节不再施工。** roadmap §0.3 死路清单已收录。
>
> **教训（防再诉讼）：** 凡以「外部引擎有 QKV merge 收益」为由立项前，必须先核实本仓 CUDA 端物理实现是否已合并——外部收益前提（对方是 3 瘦 GEMM）在本仓不成立。

#### 原目标与收益（作废存档）

- ~~**收益（预估）：−0.6 ~ −1.2s**（480P DiT 段）~~——**实测 0**。原依据（nsys `qkv_rope` 0.25s + `adaln` 0.17s + 3 次瘦 GEMM 间隙）系对逻辑视图的误读：物理上 QKV 已是单次 `[3*INNER, HIDDEN]` 宽 GEMM，QKNorm+RoPE 已在 CUDA 端融合，无独立核可消。
- ~~**服务档位：** T1 主~~——无收益，不服务任何档位。

#### 原施工设想（作废存档，勿执行）

> 以下小节保留仅作历史档案。**注意：原「代码级定位」中「每层 3 次独立 QKV-proj INT8 GEMM」的描述已被探针证伪**——物理实现为单次 `[3*INNER, HIDDEN]` 宽 GEMM。

- **代码级定位（原始记录）：** `h3_dit_w4a8.c` attn 前段；`h3_dit_w4a8_qkv_rope_f32`（`h3_dit_w4a8.c:2279` 定义，`:3224/:3232/:5894` 调用）+ CUDA 侧 `h3_cuda_qkv_rope_f32` / `h3_cuda_qkv_rope_sdpa_f16_dev`（`h3_gpu_cuda.cu`）；RoPE 表 `h3_dit_w4a8_rope_tables_f32`（`:1921`，缓存化仍有效，见 §4.1）。
- **原施工步骤（QKV merge FP 验证 MAE → INT8 评估；QKNorm+RoPE 融合核逐层 MAE）：** 全部作废——探针 MAE=0 已证明无需施工。
- **原风险（INT8 累加序、融合核 MAE）：** 随施工作废而消解。
- **原插件接线（`H3_DIT_QKV_MERGE` / `H3_DIT_QKNORM_ROPE_FUSE`）：** 不再创建。

---

### 2.3 P2b：CUDA Graph 整步捕获（**已交付，收益微薄，不翻默认**）

> **2026-10-05 回填结论：** 已实现**整步单次捕获/replay**（`H3_DIT_W4A8_CUDA_GRAPH`，默认 0），238 验证 hot clip2 **位一致**。但实测收益微薄：hot DiT 12.61→12.63s（**持平**）、e2e 仅 **−0.16s**——turbo4 仅 4 步，总 kernel 数有限，launch 开销收益被 cold 噪声掩盖。**判定：可选开关，不翻默认，移出施工序列。** 原预估 −0.5~−1.5s（双源背书）**未兑现**——官方 Spark 1.39× 的 graph 收益前提是 50-step 长序列，4-step 下杠杆不成立（回填实证）。

#### 交付物存档

- **实现：** `h3_dit_w4a8.c` 的 `H3_DIT_W4A8_CUDA_GRAPH`（默认 0）——整步单次捕获/replay（原「接线点已预留」的判断已落地为完整实现）。
- **验证：** 238 hot clip2 checksum 逐位一致（bit-identical 立身之本已验）。
- **实测收益：** hot DiT 12.61→12.63s（持平）；e2e −0.16s（在 ±0.5s 噪声内，不计入任何档位求和）。
- **保留价值：** 若未来步数上升（8-step 金线档）或 cold 噪声消除，launch 税占比回升时可重新评估翻默认；当前维持默认 0。

---

### 2.4 P2：VAE blocks 2-stream（768P 独享，P0 之后）

#### 目标与收益

- **收益（预估）：768P −1.5 ~ −2.5s**。依据：vae_blocks 9.9s 中 GEMM/SDPA 串行 6 批，2 流理论近半、打七折估（roadmap §2.4 T4-b）；VRAM 热 clip free≈22GiB 充足（实测）。
- **服务档位：** T4 独享（480P 收益不成比例，不做）。
- **前置：** P0 落地后施工（同一文件 `h3_vae_int8_cuda.c`，避免在飞冲突）。

#### 代码级定位

- **文件：** `h3_vae_int8_cuda.c`；36 transformer block 逐 chunk 的 vae_blocks 主循环（`decode_loop` 同文件的批循环，与 P0 施工区域相邻但路径独立）。
- **死路警示：** tile batch 4→8 已证 −0.36s 稳定回归（L2 命中下降），**不要再碰 batch 维度**，只碰 stream 维度。

#### 施工步骤

1. 批间切 2 条 CUDA stream：批 i 在 stream A、批 i+1 在 stream B，event 串依赖。
2. 激活 workspace ×2 的 fit 检查：施工前先实测单批激活峰值，×2 后须 fit 22GiB headroom；不 fit 则降级「2 流但共享 workspace + 轻量同步」保守档。
3. cuDNN 句柄竞争排查：SDPA per-tile 形状在双流下的句柄/workspace 复用需 per-stream 独立。
4. 分臂独立验证（与 P0 虽同文件可同窗口施工，但**必须分拍独立验证**——每拍只动一个变量）。

#### 验证门

- **速度门：** 768P 热 clip2 × ≥2；目标 −1.5~−2.5s。
- **质量门：** checksum 逐位一致（混合臂基线 `0x63987d3e8083e8c1`）+ waffle 0.75–1.05。
- **稳定性门：** 不 OOM、不减窗（≥24）、repeat=2 exit=0。

#### 风险与回退

| 风险 | 依据 | 回退方案 |
| --- | --- | --- |
| 激活 workspace ×2 超显存 | roadmap T4-b 风险栏 | 共享 workspace 保守档；仍不行则整刀放弃，T4 目标靠 P0+T4-c 触边（51.5s 边界，roadmap §3.4 允许下修） |
| cuDNN 双流句柄竞争 | roadmap T4-b | per-stream 句柄实例 |

#### 插件接线

- 建议开关：`H3_VAE_BLOCKS_STREAMS`（取值 1/2，默认 1=零行为变化）；注册表 + lint。

---

### 2.5 P3：x264 ultrafast / vae_load 余尾（小尾料）

#### P3-a：x264 ultrafast（T1）

- **定位：** `tools/h3_cuda_generate.c` mux 段，`H3_X264_PRESET` 已存在（veryfast 已落地 −260ms 实测）。
- **施工：** `H3_X264_PRESET=ultrafast` 再省 ~100ms（实测）；**仅 T1 预览档可接受**（画质损失明显，正式档不推荐——实测判定）。
- **验证门：** 抽帧人眼可接受（T1 标准）；mux ≤0.15s。
- **风险：** 无工程风险；产品风险=勿让 ultrafast 混进 T2/T4 默认。

#### P3-b：vae_load 余尾（T4）

- **定位：** `h3_cuda_vae_weights_load` dequant 249ms + h2d 24ms（roadmap §2.4 T4-c，§8.7 登记：host 暖载全隐藏，device 尾仍串行）。
- **施工：** prewarm 线程预填 `w_dev` 或 host_pack restore 修复，目标 −0.3s。
- **风险：** 多占一份设备显存——768P 高分段 32GB **无 headroom**，可能不可行；**降级选项：接受余项**（roadmap §3.4 明确允许，T4-c 失败 T4 仍可达）。
- **验证门：** 不 OOM 前提下 checksum 一致。

---

## 3. 质量轨实施方案

> V0/V1 ~~是~~ **曾是主栈成立性闸门——2026-10-05 回填：已闭环**（V0 判退化、护栏臂接管默认，见 §3.1）；V2 是质量补齐 A/B；V3/V4 是调研/演进项。质量轨不动任何速度秒数承诺。

### 3.1 V0：sol_cute @ turbo4 稀疏有效性复核（**已闭环：判退化**）

#### 回填结论（2026-10-05）

- **判定：退化。** sol 全稀疏臂（`FIRST_DENSE_*=0`）在 turbo4 下判退化，**退出默认路径**；FastWan/Hao AI Lab「训练无关稀疏在 <10 步蒸馏下急剧退化」论点在本仓实证成立。
- **证据链：** 旧金线 clip2=`0xce0b5df2bcdb4ad3` 证为该退化臂产物，作废；对照之下护栏臂（sol + 缺省 dense_steps=2/layers=2）hot clip2 waffle mean=0.83（帧 0.750/0.803/0.851/0.816/0.935），成为新默认与新金线（`0x923b0db3a8c0357e`）。
- **分支动作：** 按原「退化」预案执行——sol_cute 全稀疏退出默认；**残余待办**：8-step 金线档是否保留 sol 全稀疏臂待另行评估（不阻塞）；V4（SLA 式稀疏感知微调）维持低优先演进方向（护栏臂已解决默认路径问题，微调提级条件未触发）。

#### 原实验设计（存档）

- 对照臂全 dense（`--sdpa-backend cudnn`）vs 实验臂 sol_cute 全稀疏；判定线 waffle 均值超对照 >0.15 或漂出金线区间且人眼可见。**注：cold 侧 waffle 高噪已定性为 cold 轨迹固有数值失稳（纯 cudnn dense 基线 cold 也全帧>3），与本判定无关**——本判定基于 hot clip2 与金线 checksum 证据链。

### 3.2 V1：sol MMDiT dense 护栏 A/B（**已闭环：护栏臂已接管默认**）

#### 回填结论（2026-10-05）

- **已落地为新默认路径**：护栏配置（sol + 缺省 `dense_steps=2` / `dense_layers=2`）即当前产品默认臂，对应原设计的 V1-a 臂。
- **新金线账本**：hot clip2 checksum=`0x923b0db3a8c0357e`，waffle mean=0.83（帧 0.750–0.935），hot total≈19.2–19.4s（含 cold 噪声）。
- **与原设计的差异注记：** 原设计的 V1-b（文本 query 行 dense）与 V1-c（前 1 步 dense）**未单独 A/B**——护栏臂直接以 dense_steps=2/layers=2 落地并通过质量门，更激进的护栏组合暂无必要；若未来质量回归再启用原矩阵。

#### 原背景（存档）

- sol-h3 官方「近无损」背书来源=完整接线（sink + 文本 query 行 dense + 前 2 层 dense + 前 10 步 dense）；本仓原接 sink 但 `H3_SOL_FIRST_DENSE_STEPS=0` / `H3_SOL_FIRST_DENSE_LAYERS=0` 全稀疏——回填后默认已改为护栏配置，此缺口已闭合。

### 3.3 V2：taomate 0.7 强度 refiner 收尾 A/B（质量补齐）

#### 背景

- 社区高票用法（taomate-fasth3-vdn 重分析 §一）：**高步主体 + taomate 3-step @ 0.7 强度收尾**——治的正是 turbo4 的「overcooked」过饱和/糊；整段 3-step 盲测垫底（WhichH3 i2v Elo 903.5 #19），但 refiner 用法口碑极好（保物理/材质/prompt adherence）。
- 已固化社区配方：2 步 turbo4 + 3 步 taomate 收尾，或 8/9 步主体 + taomate 收尾；Kijai 出 rank-19 平均版专当 refiner。

#### 施工与实验设计

- **坑（必须先排）：**
  ① **基座版本强耦合**：stock pruned 保色（42.8）vs Singularity v1.3 掉色（28.6）——锁定 stock pruned 基座。
  ② **`h3_lora_parse_name` 显式拒绝 `token_refiner` 模块**——挂官方全量权会失败，**用 Kijai rank-19 版**（天然规避）。
- **实验矩阵：** 主体 = turbo4 4-step（现状）vs turbo4 2-step + taomate 3-step @0.7 收尾 vs 金线 8-step + taomate 收尾；同 seed 同机；读 waffle 区间 + 抽帧人眼（重点看过饱和/糊）+ av_sync。
- **接线：** 利用现有 taomate 接线（`H3_TAOMATE_*` 族已在注册表）；强度参数若无开关，加 `H3_TAOMATE_REFINER_STRENGTH`（默认 OFF/1.0 零行为变化）。

#### 判定

- waffle 区间下移且人眼过饱和改善 → 作为 T1/T3 的 opt-in 质量臂；不并 golden。
- 与 V3 的关系：V2 若有效，V3（细节恢复 LoRA 调研）降级为「同向备选」；V2 若无效，V3 提级。

### 3.4 V3 + V4：调研/演进项

#### V3：细节恢复 LoRA 调研

- **性质：** 调研项非施工项。图像侧已验证「蒸馏轨迹对齐的细节恢复 LoRA」（Z-Image DistillPatch、Krea2 4-step），视频/H3 侧空白（community-landscape §6）。
- **动作：** 监控 Z-Image/Krea2 类方法向视频侧迁移的动态；待 V2 结论定是否立项（§3.3 判定分支）。

#### V4：SLA 式稀疏+线性混合叠 sol_cute

- **与 V0 的条件分支关系（roadmap §4.1b/§4.2）：** V4 是**条件触发项**——V0 判退化 → V4 从低优先提级为正式立项；V0 判无损 → V4 维持低优先演进方向。
- **机制依据（vdn-linear-attention §四/§六）：** SLA 式按注意力权重分——critical 块走精确 FlashAttention、marginal 块走线性（成本 <0.5%）、negligible 跳过，稀疏度 70%→95% 不掉质量；已有 sol 稀疏栈（480P 400/400），给 marginal 块加线性分支是**增量**非换底座。
- **真实战场定位（必须写清防误读）：** 线性注意力的价值是 **768P 显存墙**（O(N²) 矩阵无处安放、全 sol 32GB OOM——线性恒定 d×d 状态是稀疏 softmax 原理上给不出的性质），**不是 480P 速度**。480P(16k) 已被 sol 解决，不动。
- **能力边界声明：** SLA 是 fine-tunable——**需少量微调，跨过本仓「不动训练」的默认边界**；立项即意味着接受一个训练子项目（远轻于 VDN 换底座，但非零）。
- **不做全模型替换：** VDN-the-model 维持否决（合同外：训练+换权重+B200 核），只借鉴机制。

---

## 4. 零风险尾料（可随手顺带）

> 来源：invariant-cache-audit。两项均 **bit-identical 可达**、施工简单、不改变 P0/P1 排序——随对应主刀同文件顺带，不单独占拍。

### 4.1 RoPE 表缓存补齐

| 项 | 定位（本地核实） | 施工 | 预估收益 |
| --- | --- | --- | --- |
| **W4A8 主路径 RoPE 表 per-shape memo** | `h3_dit_w4a8.c:1921`（`h3_dit_w4a8_rope_tables_f32` 三重循环现算）；每 forward 调用点 `:6232/:6628/:6916`；token-reduction 分支 `tr_build_reduced_rope` 同样每调用现算 | per-(seq, 网格形状, spatial_scale) memo：表只依赖形状，clip 形状固定时跨 4-step、跨 clip 均为真常量。同一份浮点结果只算一次 → bit-identical | **0.1–0.3s/run**（量级估计未 profile，以实测为准） || **VAE RoPE 表缓存 + H2D 移出循环** | `h3_vae_int8_cuda.c:2782`（chunk 循环内 `h3_vae_int8_prepare_rope_ex` host 现算 + `cudaMemcpy H2D` 逐 chunk）；host buffer `:2742-2752` 复用但内容每 chunk 重填 | 表只依赖 (latent_h, latent_w, chunk, origin) → 按 key 缓存；H2D 移出 chunk 循环顺带削 launch/同步气泡（族#6 叠加） | **0.05–0.2s/run**（量级估计未 profile） |

- **审计警示（invariant-cache-audit §4）：** W4A8 侧需先 profile 确认 `rope_tables_f32` 确实在 per-step 循环内被调（当前证据显示在 `forward_video` 内）；若实际在外层则收益更低——**以 profile 为准，不假数字**。
- **顺手关系：** VAE 侧随 P0-async 同文件顺带；DiT 侧可独立小拍（原「随 P1 顺带」失效——P1 已闭环，不再存在该拍）。
- **插件接线：** 纯内部优化、无行为变化，**不需要开关**（bit-identical 的不变量缓存属「永远该做」）；若保守起见要开关，用 `H3_DIT_ROPE_MEMO` / `H3_VAE_ROPE_CACHE` 默认 ON——但首选无开关直接上，验证门=checksum 逐位一致。

### 4.2 GEMM+GELU epilogue 融合（原「P1 顺带」，P1 闭环后改独立小拍）

- **定位：** quant/dequant 中转 1.465s 的一部分（roadmap §0.3 FFN epilogue 条目：**vec4 实现路径已死**，但「GEMM+epilogue 融合本身」未关闭——死的是 vec4 对齐假设，不是融合方向）。
- **施工：** 非 vec4 路径的 GEMM+GELU epilogue（cublasLt epilogue 或手写融合核，避免生产指针非对齐坑）；独立小拍（原挂靠的 P1 已闭环）。
- **验证门：** checksum 逐位一致（若 FP 结合序变化则退为 T1-only）。
- **诚实声明：** 收益未知（T1-e [审计依赖] 范围与之重叠——若方法论审计重开 quant/dequant 中转，本项并入审计结论；审计前按小项对待）。

---

## 5. 施工顺序与依赖图

### 5.1 依赖与并行关系（2026-10-05 回填后重排）

```
   [已闭环/已交付，不再占拍]
   ✓ 拍1: V0 + V1（sol 质量闸）──── 判定=退化 → 护栏臂接管默认（V4 维持低优先）
   ✓ 拍3: P1 QKV merge 探针 ──────── 闭环零收益（MAE=0，CUDA 端本就融合）
   ✓ 拍3': P2b CUDA Graph ────────── 已交付位一致，收益微薄（−0.16s），不翻默认

                    ┌──────────────────────────────────┐
   独占 GPU 窗口 ──►│ 拍2: P0-async VAE d2h 流水          │──► 480P 验证 ──► 768P 移植（T4-a）
                    │   （打开 H3_VAE_UNPATCH_ASYNC +      │     【当前唯一主刀：T1/T2/T4
                    │     per-slot done-event + 跨 tile 流水│      三档达标全部系于此拍】
                    │     顺带: VAE RoPE 缓存 §4.1）        │
                    └──────────────────────────────────┘
                              │（同文件 h3_vae_int8_cuda.c，必须串行）
                              ▼
                    ┌─────────────────────────────┐
                    │ 拍4: P2 VAE blocks 2-stream   │（768P，T4-b）
                    └─────────────────────────────┘

   无需独占（随时可插）：
   - DiT RoPE memo（§4.1，h3_dit_w4a8.c，bit-identical 不变量缓存）
   - GEMM+GELU epilogue（§4.2，[审计依赖] 重叠，审计前按小项）
   - P3-a x264 ultrafast（无依赖）；P3-b vae_load 余尾（P0 后，768P 窗口）

   等人工：T3 DMAD review。等审计：T1-e/T2-c/T4-e（T1-e 现为 T1 能否闭合的关键变量）。
   等 V2 结论：V3 立项。cold 根治（方向 B：vae_th 竞态/pinned restore/INT8 make_room）待立项。
```

**排序逻辑（回填后）：**
1. ~~V0/V1 与 P0 同窗口~~——**V0/V1 已闭环**；独占 GPU 窗口全部留给 P0-async。
2. **P0-async 是唯一关键路径**——T1/T2/T4 三档的达标数学全部建立在它兑现之上（roadmap §3.1/§3.2/§3.4 重算）；其失败 = 三档集体下修。
3. **P2 必须在 P0 后**——同一文件同一 decode 区域，在飞冲突风险高。
4. ~~P1/P2b 与 P0 并行~~——**P1 闭环、P2b 已交付，并行轨消失**；`h3_dit_w4a8.c` 侧只剩零风险尾料（RoPE memo / epilogue），不挡 P0。
5. **P3 随时插缝**。

### 5.2 里程碑（2026-10-05 回填后重排）

| 里程碑 | 定义 | 依赖 | 验收 | 状态 |
| --- | --- | --- | --- | --- |
| ~~**M1**~~ | V0 主栈定性（sol_cute@turbo4 无损/退化判定） | 独占 GPU | A/B 报告 + 判定结论 + 分支动作启动 | **✓ 已达成（判退化，护栏臂接管默认）** |
| **M2** | P0-async 落地 → **T2 达标**（480P ≤15.0s bit-identical） | 独占 GPU | **hot clip2=`0x923b0db3a8c0357e` 逐位一致** + 速度门 + 稳定性门 | 待施工（pinned 已就位，剩 async+流水） |
| ~~**M3**~~ | ~~P1 + P2b 落地 → T1 达标~~ | — | — | **已重定义见 M3'**（P1 闭环零收益、P2b 微薄，原路径无法达标） |
| **M3'** | **T1 达标（480P ≤13.5s）** = P0-async 兑现 + **新收益源补足 0.7–1.1s 缺口**（候选：T1-e 审计重开 quant/dequant 中转） | M2 + 审计报告 | waffle ≤2.5 + 漂移记录 + 速度门 | **阻塞：async + 审计双依赖** |
| **M4** | T4 双刀（P0-async 768P 移植 + P2）→ **768P ≤50s** | M2 | 混合臂 checksum（`0x63987d3e8083e8c1`）+ waffle 0.75–1.05 + 不 OOM | 待施工（条件性目标） |
| **M5** | V2 质量补齐（taomate refiner A/B 结论） | 无硬依赖，可插缝 | A/B 报告 + 人眼评审 | 待施工 |

---

## 6. 验收协议（每拍通用，继承 roadmap §6）

1. **每拍只动一个变量**：一次施工拍只改一把刀（同文件多刀也必须分拍验证，如 P0/P2）。
2. **对照纪律**：对照臂与实验臂同 seed、同机器、同 binary（仅开关差异）。
3. **速度门**：热 clip2 `phase_ms total=` × ≥2 次复现；run-to-run 噪声 ±0.5s（480P）/ ±3s（768P）内取优臂。
4. **质量门**：T1=waffle ≤2.5+抽帧人眼；T2/T4=latent checksum 逐位一致（**新金线：hot clip2=`0x923b0db3a8c0357e`**；768P 混合臂 `0x63987d3e8083e8c1`）+waffle 金线区间（480P 新锚 0.75–0.95 / 768P 0.75–1.05）；T3=waffle 区间+人审签字。**cold clip1 间歇 checksum 不作金线**（cold 轨迹非确定性已定性）。
5. **稳定性门**：av_sync PASS、sol hits 满额 fallback=0、无 OOM、repeat=2 连跑 exit=0。
6. **成片归档**：`out/review-<date>/` 待人工确认；日志落 `docs/superpowers/plans/` 修订记录。
7. **达标判定**：档位目标 = 速度门 ∧ 质量门 ∧ 稳定性门同时通过。**任一门不过，目标自动下修并修订 roadmap 目标书，不倒改口径。**
8. **插件注册 lint**：每拍结尾跑 `tools/check_plugin_registry.sh`，PASS 才算拍完成。

---

## 7. 不做清单（anti-goals，继承 roadmap §5 并补充）

| 不做什么 | 为什么 | 文档指针 |
| --- | --- | --- |
| 不承诺 480P ≤10s / 768P ≤27s（KPI-B） | bit-identical 死路链闭环；768P 需 −52% 超已知手段上限 | roadmap §3.1/§3.4 |
| 不再做 INT8 GEMM raw-FLOPS 优化（autotune/手写核/FFN epilogue **vec4**） | cublasLt 98–99% 峰值闭环；vec4 撞对齐坑整管发散 | roadmap §0.3 |
| 不做 sol KV fuse / mux overlap / VAE tile batch 4→8 / VAE 常驻 keep / 768P 全 sol | 全部实测闭环（零收益/负收益/OOM） | roadmap §0.3 |
| **不做 QKV merge + QKNorm+RoPE 融合（原 P1）** | **2026-10-05 探针实证：CUDA 端本就融合（QKV=`[3*INNER, HIDDEN]` 宽 GEMM），MAE=0 零收益** | §2.2 死路档案 |
| **P2b CUDA Graph 不翻默认** | **2026-10-05 已交付位一致但收益微薄**（hot DiT 持平、e2e −0.16s，turbo4 4-step launch 税被 cold 噪声掩盖）；保留可选开关 | §2.3 |
| **sol 全稀疏@turbo4 退出默认** | **2026-10-05 V0 判退化**；默认=护栏臂（dense_steps=2/layers=2），新金线 hot clip2=`0x923b0db3a8c0357e` | §3.1/§3.2 |
| 不减窗（<24）、不假 sage3、不 TAE/LTX、不动质量尺 | 合同红线 | 各计划文档头部 |
| **不做 VDN 全模型替换** | 合同外（训练+换权重+B200 核）；只借鉴 SLA 式机制（V4） | vdn-linear-attention §六 |
| **不做 FastH3 4-step 权** | 官方自认质量档未到（「8 steps may be a better quality setting」）；社区实测 4-step 快运动塌缩、需 6 步 | taomate-fasth3-vdn §二 |
| **不做 taomate 整段 3-step 生成** | WhichH3 盲测垫底（Elo 903.5 #19）；只做 0.7 强度 refiner 收尾（V2） | taomate-fasth3-vdn §一 |
| **不指望 turbo4 加步救质量** | DMD mode-seeking 机理：学生只在校准步数网格训练，加步=外推区误差放大；8-step 学生需重蒸馏 | 12 方法深拆 §13 |
| 不把 DMAD 并入 golden、不宣称 DMAD 有加速 | 换学生权=换轨迹；实测 dit +11% 负收益；SKU 隔离 | roadmap §5 |
| [审计依赖] 例外条款 | 方法论审计以新证据推翻任一条 → 该条移出清单并重估；仅以审计报告为准 | roadmap §4.2 |

---

## 8. 修订记录

| 日期 | 变更 |
| --- | --- |
| 2026-10-05 | 初稿：基于 roadmap（四档目标书）+ 7 份调研文档（acceleration-mechanisms / sol-h3-official-reanalysis / methods-mechanism-deepdive / taomate-fasth3-vdn-reanalysis / community-landscape / invariant-cache-audit / vdn-linear-attention）展开施工级方案。速度轨 P0–P3 五节 + 质量轨 V0–V4 四节 + 零风险尾料 + 依赖图 + 验收协议 + 不做清单。代码定位经本地源码核实处已标注（`h3_vae_int8_cuda.c` frames/unpatch_d2h 统计点、`h3_dit_w4a8.c` rope_tables/qkv_rope/H3_DIT_W4A8_CUDA_GRAPH 读取点、`h3_gpu_cuda.cu` H3_SOL_FIRST_DENSE_* 读取点）；VAE 详细行号以 238 工作副本为准。 |
| 2026-10-05 | **实测回填修订（同日第二批，与 roadmap 回填同步）：** ① 新增 **§0 施工状态快照**——六项回填结论一览（P0 pinned 半刀零提速/P1 闭环/P2b 已交付微薄/金线换锚/cold 定性/V0V1 闭环），动手前必读。② **§2.1 P0 重写**：pinned 部分标记已完成（插桩证实 `cudaHostAlloc` 命中），施工范围收窄为「打开 `H3_VAE_UNPATCH_ASYNC`（默认关）+ per-slot done-event + 跨 tile 流水」；速度门基线从错配的 ~1.6s 改为实测 3227ms（94% = 229ms/tile × 13 串行同步等待）；收益预估标注「未兑现」。③ **§2.2 P1 整节作废存档**：探针实证 QKV 本就 `[3*INNER, HIDDEN]` 宽 GEMM 合并、QKNorm+RoPE 本就在 CUDA 端融合，MAE=0 零收益；补「防再诉讼」教训（外部引擎收益前提须先核实本仓物理实现）。④ **§2.3 P2b 标记已交付**：整步捕获/replay 位一致，但 hot DiT 持平、e2e −0.16s 微薄，不翻默认；原 −0.5~−1.5s 预估未兑现（官方 graph 收益前提是 50-step，4-step 杠杆不成立）。⑤ **§3.1/§3.2 V0/V1 标记已闭环**：sol 全稀疏判退化退出默认，护栏臂（dense_steps=2/layers=2）接管默认；新金线 hot clip2=`0x923b0db3a8c0357e`。⑥ **§1.2 全局约束换锚**：金线 checksum 只认 hot clip2 新值，cold clip1 间歇不作金线；四档目标补可达性注记（T1/T2 当前不可闭合、T4 条件性）。⑦ **§5 依赖图与里程碑重排**：有效施工序列收窄为 P0-async（唯一关键路径）→ P2 → P3；M1 已达成、M3 重定义为 M3'（T1 = async + 新收益源双依赖）。⑧ §7 不做清单新增三条（QKV 融合/P2b 翻默认/sol 全稀疏默认）。 |
