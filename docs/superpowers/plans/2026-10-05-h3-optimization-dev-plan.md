# H3.c 优化开发方案（施工级）：速度轨 P0–P3 + 质量轨 V0–V4

> **日期：** 2026-10-05。**机器基准：** 192.168.0.238，1×RTX 5090 32GB，sm_120，CUDA 13。
> **定位：** 本文是 `2026-10-05-h3-optimization-roadmap.md`（目标书）的**施工级落地文档**——读者是接下来动手写代码的工程师（可能是另一个代理）。roadmap 回答「做什么、做到多少、为什么」；本文回答「**在哪个文件哪个函数动手、按什么步骤动、动完怎么验证、出问题怎么回退**」。
> **数据纪律继承 roadmap：** 所有「实测」数字锚定 238 真机日志；所有「预估」标注依据；不假数字、不承诺 10s/27s；每拍只动一个变量。
> **代码定位纪律：** VAE 源码可能在 238 远程机器上，本地副本若缺对应函数，一律以 roadmap/审计文档中**真实出现过的函数级归因**（`decode_loop`、`unpatch_d2h`、`h3_vae_int8_frames`、`qkv_rope`、`adaln`、`rope_tables_f32` 等）为准，**不编造函数名**。本地已核实的行号会标注「本地核实」。

---

## 1. 总览

### 1.1 双轨结构

本方案分两条**并行施工轨**，互不阻塞、共享验收纪律：

| 轨 | 序列 | 性质 | 依据 |
| --- | --- | --- | --- |
| **速度轨** | P0 / P1 / P2b / P2 / P3 | 纯工程刀（机理族#6 + 族#4 精确不变量缓存），外部已证原理，无质量杠杆（P1 有 INT8 累加序例外，见 §3.2） | sol-h3 官方重分析、12 方法机制深拆、不变量缓存审计 |
| **质量轨** | V0 / V1 / V2 / V3 / V4 | 主栈成立性验证（V0/V1）+ 质量补齐（V2/V3）+ 机制演进（V4） | 社区全景（FastWan 冲突）、taomate/VDN 重分析 |

**两轨的交互点只有一个：** V0（sol_cute@turbo4 有效性复核）的判定结果决定 sol_cute 能否留在默认产品路径——这不改变任何速度刀的施工内容，只改变「默认 binary 里 `--sdpa-backend sol_cute` 是否仍是产品默认」。

### 1.2 全局约束（不可逾越）

1. **四档 SKU 目标**（roadmap §1，本文一切施工以达此为目的）：
   - **T1 极速档**：480P 热 e2e ≤13.5s，waffle ≤2.5 + av_sync PASS，豁免 bit-identical（记漂移量）。
   - **T2 均衡档（默认产品）**：480P 热 e2e ≤15.0s，latent checksum **逐位一致**（clip1=`0x28e8a880d52e9be1` / clip2=`0xce0b5df2bcdb4ad3`）+ waffle 1.0–1.7。
   - **T3 质量档**：waffle ≤1.35 + 人工 review 签字（DMAD full_critic 成片已在 `out/review-2026-10-04/kpib-dmad-kj/`），**等人工，不在本方案施工范围**。
   - **T4 分辨率档**：768P 热 e2e ≤50s（从 P50 56.47s 起），checksum 逐位一致（混合臂基线 `0x63987d3e8083e8c1`）+ waffle 0.75–1.05 + 不 OOM、不减窗（≥24）。
2. **金线 checksum 不可动**：clip1=`0x28e8a880d52e9be1`、clip2=`0xce0b5df2bcdb4ad3`。任何声称 bit-identical 的刀必须逐位复现这两个值。
3. **不动 INT8 GEMM 本征**：cublasLt 已达硬件峰值 98–99%（C5b 三层验证闭环），不 autotune、不手写 sm_120 核、不做 FFN epilogue vec4（对齐坑前科）。
4. **不承诺 10s / 27s**：数学不闭合（roadmap §3.1/§3.4），写入任何文档即违约。
5. **每拍只动一个变量**：一次施工拍只改一把刀；对照臂与实验臂同 seed、同机、同 binary（仅开关差异）。
6. **新开关默认 OFF**：所有新刀按 `H3_*` 惯例注册进 `tools/h3_plugin.h` 的 `H3_PLUGIN_BUILTIN_DEFS`，默认 OFF 零行为变化，`tools/check_plugin_registry.sh` 必须 PASS。

### 1.3 执行依赖（资源约束）

| 依赖 | 影响范围 | 说明 |
| --- | --- | --- |
| **独占 GPU 窗口** | P0（必需）、V0/V1（必需） | P0 的竞态调试对时序敏感——拥挤 GPU 下位精确验证已失败过一次（`H3_VAE_DECODE_PIPELINE` 实验）；V0/V1 的 A/B 要求同机无噪声。 |
| **T3 人工 review** | T3 整档 | DMAD 成片待人审签字，本方案不覆盖。 |
| **方法论审计报告** | T1-e / T2-c / T4-e | [审计依赖] 项不在本方案主动施工，仅登记触发条款。 |
| **VAE 源码位置** | P0 / P2 | `h3_vae_int8_cuda.c` 本地副本存在（已核实关键符号），但完整施工与 profile 需在 238 上进行；本地行号仅供预读，施工前以 238 工作副本为准。 |

---

## 2. 速度轨施工方案

> 每把刀一节，统一结构：**目标与收益 → 代码级定位 → 施工步骤 → 验证门 → 风险与回退 → 插件接线**。
> 施工顺序与文件冲突详见 §6；本节按优先级 P0→P1→P2b→P2→P3 排列。

### 2.1 P0：VAE unpatch_d2h pinned + async + stream 流水（最高优先）

#### 目标与收益

| 档位 | 收益（预估） | 达标贡献 |
| --- | --- | --- |
| T1（480P） | **−1.3 ~ −1.6s** | 15.9s → 13.0–13.9s 的最大单刀 |
| T2（480P bit-identical） | **−1.3 ~ −1.6s** | 16.4s → 14.5–14.8s 的**唯一**刀（T2 实质单点依赖） |
| T4（768P） | **−3.5 ~ −5.5s** | 56.47s → 48.2–51.2s 的双刀之一 |

**作用域限定（roadmap §4.1）：** 收益只落在 VAE 4.96s 中的族#6 零成本浪费子集（d2h 串行尾 ~1.6s@480P / ~9.5s@768P + FillFunctor ~0.7s ≈ 2.3s@480P）；**不触碰 vae_blocks 主体 ~2.6s 的 fp16 真实计算**。bit-identical 可达。

#### 代码级定位

- **文件：** `h3_vae_int8_cuda.c`。
- **核心结构：** `h3_vae_int8_frames`（本地核实：`h3_vae_int8_cuda.c:1542, 1693, 1785` 等处出现；批量槽位 `g_vae_batch_frames[H3_VAE_MAX_TILE_BATCH]` 在 `:165`）。
- **瓶颈路径（文档归因）：** `decode_loop` 逐 tile 串行 D2H + 主机 stitch——`st_acc.unpatch_d2h_ms` 累加点在 `:1955` 与 `:2220`（本地核实存在该统计字段），说明 unpatch_d2h 在 decode 主循环内被逐段计时、串行等待。
- **前科定位：** pinned 探索曾因 `fr->rgb` 生命周期 segfault 被否（roadmap §2.1 T1-b）；`H3_VAE_DECODE_PIPELINE` 实验已证全异步有竞态（copy worker 异步 pinned→final memcpy 与 stitch 跨 clip 消费），且已顺带固化 2 个真实 bug（GPU unpatch batch stride 误用被覆写的 `g_vae_attn_seq` → 局部 `batch_stride`；`d_rgb` 双缓冲空指针）——**这两个修复已落地，施工时不得回退**。
- **RoPE 顺带位：** chunk 循环内 `h3_vae_int8_prepare_rope_ex`（本地核实 `:2782`）每 chunk host 现算 + H2D——见 §5.1 零风险尾料，可同拍顺带。

#### 施工步骤

1. **pinned 标志穿进 `h3_vae_int8_frames` 生命周期**（§8.6 遗留改造）：
   - 在 `h3_vae_int8_frames` 结构体增加 `int rgb_pinned;` 标志字段（或等价机制），`h3_vae_int8_frames_free` 按标志选择 `cudaFreeHost` vs `free`。
   - 分配点改为 `cudaHostAlloc`（pinned），只在新开关 ON 时启用；OFF 路径保持 `malloc` 行为逐字节不变。
   - **这就是上次 segfault 的点**：`fr->rgb` 被跨作用域消费（stitch 阶段跨 clip 读取），上次改造在消费前释放了 pinned 缓冲。本次必须画出「分配 → decode 写入 → D2H → stitch 消费 → free」的完整生命周期图，free 只能在 stitch 消费完之后。
2. **per-slot done-event 替换全局 quiesce**：
   - 为 `g_vae_batch_frames[]` 每个槽位配一个 `cudaEvent_t done_event`；D2H async copy 提交后立即 `cudaEventRecord` 到 copy stream。
   - stitch/消费端不再全局 `cudaDeviceSynchronize`，而是 `cudaEventSynchronize(slot->done_event)` 按槽就绪。
   - 淘汰 `H3_VAE_DECODE_PIPELINE` 时代的「全局 quiesce」模式（竞态根源）。
3. **stream 并行：D2H 与下一 tile GPU decode 重叠**：
   - decode kernel 跑在主 stream；D2H copy 跑在独立 copy stream，用 event 建立「tile N decode 完成 → tile N D2H 开始」与「tile N D2H 与 tile N+1 decode 并行」的依赖边。
   - 主机 stitch 线程用 per-slot event 就绪后即消费，不阻塞 GPU 侧下一 tile。
4. **480P 验证通过后移植 768P 放大版**：
   - 768P 为 6 批 × ~0.35GiB/批 D2H；注意 host 侧 pinned 内存压力（见风险）， pinned 总量超阈值时降级为「前 K 批 pinned + 其余 pageable」混合策略。

#### 验证门

- **速度门：** 热 clip2 `phase_ms total=` × ≥2 次复现；480P 目标 unpatch_d2h 串行尾 ≤0.3s（从 ~1.6s）；768P ≤4s（从 ~9.5s）。run-to-run 噪声 ±0.5s（480P）/ ±3s（768P）。
- **质量门：** latent checksum **逐位一致**（clip1/clip2 金线值；768P 混合臂基线 `0x63987d3e8083e8c1`）+ waffle 1.0–1.7（480P）/ 0.75–1.05（768P）。
- **稳定性门：** av_sync PASS、sol 400/400 fallback=0、无 OOM、repeat=2 连跑 exit=0、host pinned 占用可观测无泄漏。

#### 风险与回退

| 风险 | 前科/依据 | 回退方案 |
| --- | --- | --- |
| `h3_vae_int8_frames` 生命周期 segfault | pinned 探索已崩过一次（`fr->rgb`） | 开关 OFF 回退到 malloc+同步 copy 旧路径（零行为变化）；分阶段落地：先做「pinned+同步」验证位精确，再开 async |
| 异步竞态（stitch 跨 clip 消费 vs copy worker） | `H3_VAE_DECODE_PIPELINE` 实测 ON 未达标、拥挤 GPU 位精确验证失败 | per-slot done-event 即回退方案本身；若仍不稳，降级为「tile 内 pinned + tile 间同步」保守档，收益打折但位精确 |
| 768P pinned 内存压力（6×0.35GiB host） | 预估 | 混合策略（前 K 批 pinned）；最差接受 480P 收益、768P 仅部分落地，T4 目标相应下修（roadmap §3.4 允许 a+b 下限 −5.0s 触边） |

#### 插件接线

- 新开关建议名：`H3_VAE_D2H_PIPELINE`（或沿用 `H3_VAE_DECODE_PIPELINE` 语义重写后复用——若复用，需在修订记录注明「语义已换实现」）。
- 注册进 `tools/h3_plugin.h` 的 `H3_PLUGIN_BUILTIN_DEFS`，默认 OFF；`tools/check_plugin_registry.sh` PASS。
- 保留 `H3_VAE_D2H_ASYNC_COPY` 实验项语义不动（历史开关，不删）。

---

### 2.2 P1：QKV merge + QKNorm+RoPE 融合（T1 主刀）

#### 目标与收益

- **收益（预估）：−0.6 ~ −1.2s**（480P DiT 段）。依据：nsys DiT 窗口内 `qkv_rope` 0.25s + `adaln` 0.17s + 3 次瘦 GEMM 的 launch/访存间隙；合并后 K 维增大的 IMMA 效率提升；外部 NVIDIA RTX5090 Pillar-1 kernel 阶梯 1.08×。
- **服务档位：** T1 主；**若做到 bit-identical 则同时喂 T2**（见风险节的 INT8 累加序分析）。
- **bit-identical 可达性：** QKNorm+RoPE 融合核若保持同一浮点结合序则 bit-identical；QKV merge 的 INT8 合并 GEMM 若换累加分块顺序则**非 bit-identical**（T1 允许，记漂移；T2 需逐位才可收编）。

#### 代码级定位

- **DiT attn 前段：** `h3_dit_w4a8.c`——每层 3 次独立 QKV-proj INT8 GEMM（d_q/d_k/d_v 独立，roadmap §1.6.1 C1c 已证结构）。
- **qkv_rope 核：** `h3_dit_w4a8_qkv_rope_f32`（本地核实 `h3_dit_w4a8.c:2279` 定义，`:3224/:3232/:5894` 三处调用）+ CUDA 侧 `h3_cuda_qkv_rope_f32` / `h3_cuda_qkv_rope_sdpa_f16_dev`（`h3_gpu_cuda.cu`）。
- **RoPE 表：** `h3_dit_w4a8_rope_tables_f32`（本地核实 `:1921` 定义，`:6232/:6628/:6916` 每 forward 现算）——缓存化见 §5.1。
- **adaln：** adaln 相关核在 `h3_gpu_cuda.cu`（nsys 归因 0.17s）。

#### 施工步骤

1. **QKV merge（先做 FP 路径验证 MAE，再定 INT8）：**
   - 权重加载时把 d_q/d_k/d_v 三个 INT8 权重沿输出维拼成单个宽 GEMM 权重（K 维拼宽）；scale 表同步拼接。
   - 前向改为一次宽 GEMM 产出 `qkv` 连续缓冲，再按偏移切视图给下游。
   - **顺序：先在 FP/对照路径（或 CPU 参考）验证 merged GEMM 输出对三次独立 GEMM 的 MAE**；INT8 路径的 dequant 浮点结合序差异单独评估（12 方法深拆 §8：风险不在 INT32 累加——精确，在 dequant scale 应用的浮点序）。
2. **QKNorm+RoPE 融合核：**
   - 现状：q_norm/k_norm RMS 与 RoPE 为独立核，中间张量走 HBM。
   - 目标：单核内完成 q/k 的 RMS normalize + RoPE 旋转，中间结果留 SMEM（NVIDIA Pillar-1 同款，sol-h3 官方重分析 §五.3「缺失，需 checksum 门」）。
   - 逐层对 MAE：融合核输出 vs 旧两核串联，per-layer MAE 阈值内（建议 ≤1e-6 相对）才算等价。
3. **与 sol_cute 路径的兼容性确认**：融合核输出布局必须喂得进 sol_cute（sink/τ 路由读取 q/k 的布局契约不变，`H3_SOL_SINK_TOKENS` 等不受影响）。

#### 验证门

- **速度门：** 热 clip2 `phase_ms total=` × ≥2；DiT 段目标 −0.6~−1.2s。
- **质量门：** T1=waffle ≤2.5 + latent 漂移量记录；**若申报 T2 收编：clip1/clip2 checksum 逐位一致**。
- **稳定性门：** av_sync PASS、sol 400/400 fallback=0、repeat=2 exit=0。

#### 风险与回退

| 风险 | 依据 | 回退方案 |
| --- | --- | --- |
| INT8 合并 GEMM 累加顺序 → dequant 浮点结合序差异 → 非 bit-identical | 12 方法深拆 §8（INT8 caveat 源头=dequant 浮点序非 INT32 累加） | T1 内接受（记漂移）；T2 收编失败则 QKV merge 仅限 T1 开关，RoPE 融合若位精确仍单独喂 T2 |
| 融合核逐层 MAE 超标 | sol-h3 官方重分析 §五.3 标注「需 checksum 门」 | 回退独立核；融合核限 T1 |
| 与 P0 并行施工的文件冲突 | P1 动 `h3_dit_w4a8.c`/`h3_gpu_cuda.cu`，P0 动 `h3_vae_int8_cuda.c` | 天然隔离，可并行（§6） |

#### 插件接线

- 建议开关：`H3_DIT_QKV_MERGE` + `H3_DIT_QKNORM_ROPE_FUSE`（两个独立开关，便于每拍只动一个变量）。
- 注册 `H3_PLUGIN_BUILTIN_DEFS`，默认 OFF；lint PASS。
- 历史开关 `H3_DIT_QKV_GROUPED`（`h3_dit_w4a8.c:141`，Metal-layout 导出用）语义不同，**勿复用勿混淆**。

---

### 2.3 P2b：CUDA Graph 整步捕获（纯无损，双源背书）

#### 目标与收益

- **收益（预估）：−0.5 ~ −1.5s**（480P DiT 段）。依据双源：① sol-h3 官方 Spark 1.39× 明确含 graph capture（官方重分析 §一/§二.1「我们完全没有，确认缺失」）；② 本仓 sol aux 12232 小核实例 launch-bound（12 方法深拆 §7：CUDA Graph bit-identical——重放相同 kernel/参数/地址/顺序，连浮点结合序都不变；**对 4-step 杠杆比 50-step 更大**——总 kernel 减半但 launch 税不变，占比反升）。
- **服务档位：** T1、T2（纯无损 bit-identical 可达，T2 可收编）。
- **优先级定位：** 与 P1 并行（不同子系统），无硬前置。

#### 代码级定位

- **捕获对象：** DiT 一个完整 denoise step 的 kernel 序列（`h3_dit_w4a8.c` 的 per-step forward：QKV GEMM → qkv_rope → SDPA（sol_cute/cudnn）→ FFN GEMM → adaln）。
- **既有钩子：** `h3_dit_w4a8.c:2985` 已存在 `H3_DIT_W4A8_CUDA_GRAPH` 读取（本地核实 `env_flag_is_one("H3_DIT_W4A8_CUDA_GRAPH")`）——说明 CUDA Graph 的**接线点已预留**，施工前先核实该开关当前实际控制范围（可能只是局部或未完工），在其基础上补全整步捕获，不要另起炉灶。

#### 施工步骤

1. **形状盘点：** 枚举 480P/768P 各档的静态形状组合（seq、batch、window），确认每 shape 一图（静态形状约束，12 方法深拆 §7）。
2. **捕获边界划定：** 以「一个 denoise step」为捕获单元；step 间变化的部分（sigma、adaln 行索引）必须通过**静态设备内存 + graph 外更新**喂入（graph 重放只认固定地址）。
3. **INT8 窗重载交互验证：** INT8 权重窗（`H3_SOL_MAX_SEQ`/window 机制）若在 step 间换窗，换窗写权重必须发生在 graph 外且地址固定；验证「窗重载 + graph 重放」组合下 checksum 逐位一致。
4. **回退路径：** 捕获失败（动态形状/非法地址）自动降级为 eager 执行并打一次性 WARN，不影响正确性。

#### 验证门

- **速度门：** 热 clip2 × ≥2；目标 −0.5~−1.5s。
- **质量门：** **checksum 逐位一致**（这是此刀的立身之本，必须过 T2 门）。
- **稳定性门：** av_sync PASS、sol 400/400、无 OOM、repeat=2 exit=0；graph 内存驻留量可观测。

#### 风险与回退

| 风险 | 依据 | 回退方案 |
| --- | --- | --- |
| 静态形状约束 → 每 shape 一图，多 shape 内存膨胀 | 12 方法深拆 §7 | 图缓存按 shape LRU，超限回退 eager |
| 与 INT8 窗重载交互 | roadmap §4.1 P2b 风险栏 | 窗重载移出捕获区；若无法共存，graph 仅覆盖无换窗的 shape |
| sol_cute 核是否可捕获 | 官方核 launch 语义未审 | 先对 cudnn SDPA 臂捕获验证，sol_cute 臂单独一拍 |

#### 插件接线

- 沿用/补全 `H3_DIT_W4A8_CUDA_GRAPH`（已存在的读取点）；默认 OFF；注册表确认条目存在，lint PASS。

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
- **质量门：** checksum 逐位一致（混合臂基线）+ waffle 0.75–1.05。
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

> V0/V1 是**主栈成立性闸门**（判定 sol_cute 能否留默认路径）；V2 是质量补齐 A/B；V3/V4 是调研/演进项。质量轨不动任何速度秒数承诺。

### 3.1 V0：sol_cute @ turbo4 稀疏有效性复核（最高优先，主栈成立性闸门）

#### 背景与冲突

- **实质冲突（community-landscape §4 + acceleration-mechanisms 附录）：** Hao AI Lab / FastWan 论点——**训练无关稀疏在 <10 步蒸馏下急剧退化**。我们的组合 turbo4（4-step 蒸馏）+ sol_cute（推理期训练无关稀疏）正中此条。
- **现状证据缺口：** 480P hits=400/400 只证**路由在跑**，未证**质量无损**（sol 官方甜区是 50-step/768P 长序列；社区实测 480P 收益已近下限，RTX PRO 6000 仅 2.5%）。

#### 实验设计（完整 A/B）

| 维度 | 设定 |
| --- | --- |
| **对照臂** | 全 dense：`--sdpa-backend cudnn`（关 sol） |
| **实验臂** | sol_cute：`--sdpa-backend sol_cute`（现产品默认接线，`H3_SOL_FIRST_DENSE_STEPS=0`、`H3_SOL_FIRST_DENSE_LAYERS=0` 全稀疏） |
| **控制变量** | 同 seed、同机（238 独占窗口）、同 binary，仅开关差异；clip1/clip2 全测，repeat ≥2 |
| **读取指标** | ① waffle b16 区间（金线 1.0–1.7）② latent checksum 漂移量（双臂不必相同——sol 官方自认近似算法非 bit-exact，关键是漂移是否落在质量可接受区）③ 抽帧人眼（快运动/纹理/人脸场景重点）④ av_sync |
| **判定标准（退化定义）** | 实验臂 waffle 均值超对照臂 **>0.15**，或任一 clip 出金线区间且人眼可见差异，或 av_sync 劣化 → 判退化 |

#### 分支预案

| 判定 | 动作 |
| --- | --- |
| **无损**（漂移在区间内） | 主栈成立性确认 → V1 dense 护栏 A/B 顺推（§3.2），sol_cute 维持默认候选 |
| **退化** | ① sol_cute 限 8-step 金线档（turbo4 4-step 默认路径退回 cudnn 或启动 V1 抢救）② **触发 V4 提级**（SLA 式稀疏感知微调调研从低优先升为正式立项，见 §3.4 条件分支）③ 修订 roadmap 目标书相关档位口径 |

#### 验证门

- 速度记录（双臂 total，供后续决策参考）但**本项不设速度门**——这是质量闸。
- 稳定性门：双臂 av_sync PASS、sol 臂 400/400 fallback=0、repeat=2 exit=0。

### 3.2 V1：sol MMDiT dense 护栏 A/B（与 V0 同批）

#### 背景

- sol-h3 官方「近无损」背书的来源是**完整接线**：sink 保文本 K/V + **文本 query 行整个走 dense + 前 2 层 dense + 前 10 步 dense**（官方 `make_mmdit_sol_attn_dispatch`）。
- 我们接了 sink（K/V 侧），但 `H3_SOL_FIRST_DENSE_STEPS=0` / `H3_SOL_FIRST_DENSE_LAYERS=0` **全稀疏**（本地核实读取点 `h3_gpu_cuda.cu:5243, 5251`）——只用了官方接线的一半。**这是 sol_cute 臂能否从 opt-in 挪到产品默认的关键**（官方重分析 §二.4）。

#### 实验设计

- 与 V0 同窗口同批做（都是 sol 质量门，摊薄独占 GPU 成本）。
- 实验矩阵（在 V0 的 sol_cute 臂基础上）：
  - 臂 V1-a：`H3_SOL_FIRST_DENSE_LAYERS=2`（前 2 层 dense，对齐官方）。
  - 臂 V1-b：V1-a + 文本 query 行 dense（若需代码补齐接线，本拍含小施工）。
  - turbo4 只有 4 step，「前 10 步 dense」无意义——**等效项是「前 1 步 dense」**，可作臂 V1-c（`H3_SOL_FIRST_DENSE_STEPS=1`）。
- 指标同 V0；额外记 sol hits 变化（dense 层/步会降 hits，如 400/400 → 更低，属预期，记实际值）。

#### 判定与动作

- 若护栏臂质量显著优于全稀疏臂且速度代价可接受（hits 下降带来的 DiT 回退 < V0 臂相对 dense 臂的收益）→ 护栏配置成为 sol_cute 新默认候选。
- 若护栏也救不回（V0 判退化且 V1 臂仍退化）→ 强化「sol_cute 限 8-step 金线档」结论。

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
| **W4A8 主路径 RoPE 表 per-shape memo** | `h3_dit_w4a8.c:1921`（`h3_dit_w4a8_rope_tables_f32` 三重循环现算）；每 forward 调用点 `:6232/:6628/:6916`；token-reduction 分支 `tr_build_reduced_rope` 同样每调用现算 | per-(seq, 网格形状, spatial_scale) memo：表只依赖形状，clip 形状固定时跨 4-step、跨 clip 均为真常量。同一份浮点结果只算一次 → bit-identical | **0.1–0.3s/run**（量级估计未 profile，以实测为准） |
| **VAE RoPE 表缓存 + H2D 移出循环** | `h3_vae_int8_cuda.c:2782`（chunk 循环内 `h3_vae_int8_prepare_rope_ex` host 现算 + `cudaMemcpy H2D` 逐 chunk）；host buffer `:2742-2752` 复用但内容每 chunk 重填 | 表只依赖 (latent_h, latent_w, chunk, origin) → 按 key 缓存；H2D 移出 chunk 循环顺带削 launch/同步气泡（族#6 叠加） | **0.05–0.2s/run**（量级估计未 profile） |

- **审计警示（invariant-cache-audit §4）：** W4A8 侧需先 profile 确认 `rope_tables_f32` 确实在 per-step 循环内被调（当前证据显示在 `forward_video` 内）；若实际在外层则收益更低——**以 profile 为准，不假数字**。
- **顺手关系：** VAE 侧随 P0 同文件顺带；DiT 侧可随 P1 顺带（同文件 `h3_dit_w4a8.c`），也可独立小拍。
- **插件接线：** 纯内部优化、无行为变化，**不需要开关**（bit-identical 的不变量缓存属「永远该做」）；若保守起见要开关，用 `H3_DIT_ROPE_MEMO` / `H3_VAE_ROPE_CACHE` 默认 ON——但首选无开关直接上，验证门=checksum 逐位一致。

### 4.2 GEMM+GELU epilogue 融合（P1 顺带）

- **定位：** quant/dequant 中转 1.465s 的一部分（roadmap §0.3 FFN epilogue 条目：**vec4 实现路径已死**，但「GEMM+epilogue 融合本身」未关闭——死的是 vec4 对齐假设，不是融合方向）。
- **施工：** 非 vec4 路径的 GEMM+GELU epilogue（cublasLt epilogue 或手写融合核，避免生产指针非对齐坑）；随 P1 同区域顺带。
- **验证门：** checksum 逐位一致（若 FP 结合序变化则退为 T1-only）。
- **诚实声明：** 收益未知（T1-e [审计依赖] 范围与之重叠——若方法论审计重开 quant/dequant 中转，本项并入审计结论；审计前按小项对待）。

---

## 5. 施工顺序与依赖图

### 5.1 依赖与并行关系

```
                    ┌─────────────────────────────┐
   独占 GPU 窗口 ──►│ 拍1: V0 + V1（sol 质量闸，同批）│──► 判定 ──► 无损: V1 顺推 / 退化: V4 提级
                    └─────────────────────────────┘
                    ┌─────────────────────────────┐
   独占 GPU 窗口 ──►│ 拍2: P0 VAE d2h 流水          │──► 480P 验证 ──► 768P 移植（T4-a）
                    │   （顺带: VAE RoPE 缓存 §4.1）  │
                    └─────────────────────────────┘
                              │（同文件 h3_vae_int8_cuda.c，必须串行）
                              ▼
                    ┌─────────────────────────────┐
                    │ 拍4: P2 VAE blocks 2-stream   │（768P，T4-b）
                    └─────────────────────────────┘

   无需独占（与 P0 并行，不同文件）：
                    ┌─────────────────────────────┐
                    │ 拍3: P1 QKV merge + QKNorm/RoPE│（h3_dit_w4a8.c / h3_gpu_cuda.cu）
                    │   （顺带: DiT RoPE memo +     │
                    │     GEMM+GELU epilogue §4.2）  │
                    └─────────────────────────────┘
                    ┌─────────────────────────────┐
                    │ 拍3': P2b CUDA Graph 整步捕获  │（与 P1 同文件区域——建议同工程
                    └─────────────────────────────┘  师顺序做但分拍验证，或与 P1 交错）

   随时可插：P3-a x264 ultrafast（无依赖）；P3-b vae_load 余尾（P0 后，768P 窗口）。
   等人工：T3 DMAD review。等审计：T1-e/T2-c/T4-e。等 V2 结论：V3 立项。
```

**排序逻辑：**
1. **V0/V1 与 P0 同窗口或紧邻窗口**——两者都需要独占 GPU，且 V0 的判定可能影响产品默认路径的叙事，越早越好；P0 是收益最大单刀，也是 T2 唯一依赖。
2. **P1 与 P0 并行**——不同文件（`h3_dit_w4a8.c`/`h3_gpu_cuda.cu` vs `h3_vae_int8_cuda.c`），无冲突，可由不同工程师/不同拍推进。
3. **P2 必须在 P0 后**——同一文件同一 decode 区域，在飞冲突风险高。
4. **P2b 独立**——逻辑独立，但与 P1 同文件区域，若同人施工建议交错（P1 FP 验证等待期做 P2b 形状盘点）。
5. **P3 随时插缝**。

### 5.2 里程碑

| 里程碑 | 定义 | 依赖 | 验收 |
| --- | --- | --- | --- |
| **M1** | V0 主栈定性（sol_cute@turbo4 无损/退化判定） | 独占 GPU | A/B 报告 + 判定结论 + 分支动作启动 |
| **M2** | P0 落地 → **T2 达标**（480P ≤15.0s bit-identical） | 独占 GPU | checksum 逐位一致 + 速度门 + 稳定性门 |
| **M3** | P1 + P2b 落地 → **T1 达标**（480P ≤13.5s） | M2 不必前置（并行轨） | waffle ≤2.5 + 漂移记录 + 速度门 |
| **M4** | T4 双刀（P0 768P 移植 + P2）→ **768P ≤50s** | M2 | 混合臂 checksum + waffle 0.75–1.05 + 不 OOM |
| **M5** | V2 质量补齐（taomate refiner A/B 结论） | 无硬依赖，可插缝 | A/B 报告 + 人眼评审 |

---

## 6. 验收协议（每拍通用，继承 roadmap §6）

1. **每拍只动一个变量**：一次施工拍只改一把刀（同文件多刀也必须分拍验证，如 P0/P2）。
2. **对照纪律**：对照臂与实验臂同 seed、同机器、同 binary（仅开关差异）。
3. **速度门**：热 clip2 `phase_ms total=` × ≥2 次复现；run-to-run 噪声 ±0.5s（480P）/ ±3s（768P）内取优臂。
4. **质量门**：T1=waffle ≤2.5+抽帧人眼；T2/T4=latent checksum 逐位一致+waffle 金线区间；T3=waffle 区间+人审签字。
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
