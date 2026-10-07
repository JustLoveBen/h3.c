# H3.c 算子/工程类死路清单审计：原结论逐条重判

> **日期：** 2026-10-07。**性质：** 专项方法论审计（只读，不动代码、不动 238）。
> **审计范围：** roadmap §0.3 死路清单 + §5 不做清单中的**算子/工程类**条目（不含 sol 稀疏质量类，已由 `2026-10-07-sol-attn-negative-audit.md` 审理）。
> **审计标准（与 sol-attn 审计同源）：** ① 原始证据是否充分（测量口径/重复次数/变量隔离）；② 结论是否被过度泛化（实现路径之死被泛化成方向之死）；③ 「零收益」判定是否落入 run-to-run 噪声带。
> **必读已读：** `2026-10-05-h3-optimization-roadmap.md` §0.3/§5、`2026-10-04-h3c-480p-10s-plugin-plan.md`（C1–C8 全修订记录）、`2026-10-01-h3c-nvidia-rtx5090-sol-h3-analysis.md` §8.6、`2026-10-07-sm120-operator-frontier.md`、源码 `h3_gpu_cuda.cu`（INT8 GEMM/autotune/workspace）、`h3_dit_int8_opt.h`（workspace 默认值）、`h3_dit_w4a8.c`（fused_epi/CUDA Graph）。

---

## 0. 裁决总览

| # | 死路条目 | 裁决 | 一句话理由 |
| --- | --- | --- | --- |
| 1 | INT8 autotune（cublasLt 无备选核） | **维持** | 实测时 workspace 已是 256MiB 充裕值、每形状请求 16 个 heuristic 只回 1 个；理论上仅剩「更大 workspace 换更多候选」的低价值开放问题（0.5h 可关） |
| 2 | 手写 sm_120 INT8 GEMM（98–99% 峰值） | **维持** | 8 个生产形状全部 730–742 TFLOP/s（含全部 adaln 小 M 形状），三层独立验证链硬，数学上无 15% 余量 |
| 3 | FFN epilogue vec4 | **修正** | 原档案本身只关闭了 vec4 路径（诚实），但 §5 不做清单把「epilogue 减往返」整族误并入禁做；CUTLASS EVT 式非 vec4 融合从未试过，须拆出单列 |
| 4 | sol KV fuse（访存饱和零收益） | **维持** | 微基准+nsys 双源证实每核读 227MB 已打满 ~1.6TB/s；窗口内 0.09s/clip 的理论上限无论如何不值得重开 |
| 5 | CUDA Graph（−0.16s 微薄） | **修正** | hot DiT 12.61→12.63s 只有单次数值对、且被 ±0.5s 噪声淹没；「无收益」与「测不出」未被区分——结论方向大概率对，但证据强度低于档案宣称 |
| 6 | VAE tile batch 4→8（−0.36s 回归） | **维持** | 双臂×复测×2 稳定复现于 ±0.1s 噪声外，DiT checksum 相同保证同输入；唯缺 nsys 证实 L2 解释，但「维持 batch=4」不依赖该解释 |
| 7 | mux overlap（管道本已流式） | **维持** | 双臂×2 无净增益证据充分；T1 档 ultrafast 后 mux 仅 ~35ms，此条量级已不值得保留在路线图死路清单（建议降级归档） |
| 8 | x264 veryfast/ultrafast（正收益） | **维持** | 多次复现、latent checksum 双精确（改的是封装非 latent），口径可靠；ultrafast 画质判定带主观成分但只影响 T1 预览档 |

**与 sol-attn 审计的对照结论：** 本次 8 条中**无一条出现 sol 审计那种量级的过度泛化**（全稀疏→sol 死刑）或噪声伪影（±3s 当 1.2s 反慢）。算子类死路的档案纪律明显更好——每条都有开关级隔离 A/B 和 checksum 双验。发现的问题是次一级的：1 处禁做清单的误并（#3）、1 处证据强度不足的单次数值对（#5）、2 处可低成本关闭的理论开放问题（#1 的 workspace、#6 的 L2 解释）。

---

## 1. INT8 autotune 死路（C5a）——【维持】

**原结论：** cublasLt 对全部 8 个 INT8 形状只回 `algos=1`，autotune 无候选可挑，零增益。

**原始证据：** 2026-10-04 `kpia-int8-autotune`：`H3_DIT_INT8_ALGO_BEST=1` 落地后，8 形状（m∈{3200,4096} × n∈{21504,5376,28672} × k∈{5376,7168,14336}）全部 `best=#0 t == first-fit`，Δ=0.000ms；hot clip2 两臂在噪声内；latent checksum 逐位一致。

**审计点逐条核实：**

| 审计问题 | 核实结果 |
| --- | --- |
| 测的是哪个 cuBLAS 版本？ | 238 机器 CUDA 13 工具链自带的 cuBLASLt（sm_120 目标构建）。文档未钉死精确小版本号——**次要缺口**，但 CUDA 大版本已定 |
| heuristic 是否随 workspace 大小变化？ | **理论上会**——cublasLt 的候选集按 `CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES` 过滤，split-k/大 tile 变体需要更大 workspace |
| 当时请求的 workspace 多大？ | **不是小 workspace。** 源码 `h3_dit_int8_opt.h` 注释明确：「cuBLASLt INT8 workspace. **Default 256 MiB (was 64)**」——即 C5a 落地时默认值已从 64MiB 提升到 256MiB（且有 `H3_INT8_LT_WORKSPACE_MIB` 可调，clamp 32–1024）。源码同时证实每形状请求 `heurs[16]`（requestedAlgoCount=16），**只回 1 个** |

**证据充分性判定：** 充分。256MiB 对 INT8 GEMM（m×n×4B acc 已另有 d_gemm_acc，workspace 仅用于 split-k/内部 tile）是充裕值；IMMA 核族本身不需要大 workspace，heuristic 返回集在 256MiB 下通常已完整。请求 16 回 1 说明 cublasLt 在该版本/架构/形状组合下确实只注册了一个 INT8 IMMA 实现（即 nsys 看到的 `cutlass_80_i16832`）。

**残余开放问题（低价值）：** ① 未试 512MiB/1GiB workspace 是否解锁 split-k 变体（CUDA 13 的 cublasLt 对 sm_120 INT8 理论上可能有 split-k 候选）；② 未在更大 M（768P tiled-M remainder 形状）上重复。两者都不动摇 480P 主结论。

**裁决：维持。** 重开价值低——即便出现 split-k 候选，对已达 98–99% 峰值的形状不可能有 >1–2% 收益（见 #2 的峰值账本）。
**可选关闭实验（0.5h）：** `H3_INT8_LT_WORKSPACE_MIB=1024 H3_DIT_INT8_ALGO_BEST=1` 单跑一次，看 `algos=` 日志是否仍 =1。若仍 1，彻底封档；若 >1，才有后续。

---

## 2. 手写 sm_120 INT8 GEMM 死路（C5b）——【维持】

**原结论：** cublasLt 已在硬件峰值 98–99%，「更快的 INT8 kernel」数学上无余量。

**原始证据（三层独立验证链）：**
1. **ISA 层：** sm_120 无 tcgen05/TMEM/WGMMA，唯一 INT8 tensor 指令 = `mma.sync.m16n8k32.s8`（与 Ampere 相同，正是 cublasLt 已用的 cutlass_80_i16832 核族）；CUTLASS issue #2717 确认 SM120 CollectiveBuilder 只支持 F8F6F4、不支持 INT8。
2. **硬件峰值层：** 实测 5090 INT8 mma.sync 峰值 = 745.7 TOPS（标称 838 的 89%，同步发射开销）。
3. **cublasLt 实测层：** **8 个 DiT 形状全部 730–742 TFLOP/s**（= 硬件上限 98–99%），代表形状 4096×28672×5376 = 1.708ms = 739.5 TFLOP/s = 99.1%。

**审计点逐条核实：**

| 审计问题 | 核实结果 |
| --- | --- |
| 745.7 TOPS 的实测口径是什么形状？ | 峰值 spike 用微基准 GEMM 形状测得；关键问题是它是否代表生产形状——见下行 |
| 生产 GEMM 形状是否全部达到 98–99%？ | **是。C5b 记录原文：「实测 8 个 DiT shape 全部 730-742 TFLOP/s」**——8 个形状 = 480P/768P INT8 GEMM 的全部 (m,n,k) 组合（`h3_gpu_cuda.cu` 的 `lt_i8[8]` slot 注释：「50L×4 GEMMs … Tiled-M adds a remainder shape, so 8 slots」）。不是只挑了大形状 |
| 小 M 形状（adaln 分支 M=15973 但 N/K 较小）是否也在峰值？ | 480P 的 m∈{3200,4096} 正是 tiled-M 切分后的实际 GEMM M 值，**adaln/modulation 等分支的 GEMM 都包含在这 8 个 slot 内**（qkv/proj/fc1/fc2 四种类型 × m 两档 × n/k 组合）。adaln 的 modulation GEMM 若为更小形状（如 n=1536 级别）不在 8 slot 内则属于未测——但该类 GEMM 在 nsys 中占比微小（adaln+rope 合计 0.424s/clip 含非 GEMM 部分），即便未达峰值也无量级意义 |

**证据充分性判定：** 充分。三层链互相独立（ISA 事实、微基准峰值、生产形状实测），且结论被 C5a 的 `algos=1` 交叉佐证（库内确实无备选核）。「98–99%」不是单一形状的过度泛化，是全部生产形状的一致性观测。

**裁决：维持。** 唯一正确的残余开放方向已被原档案自己标注（「消除 GEMM↔dequant/swiglu 内存往返」），并由 sm120-operator-frontier 的 E1/E2 方案接管——那不是 GEMM 本征优化，不在本条范围。

---

## 3. FFN epilogue vec4 死路（C2b）——【修正】

**原结论：** epilogue vec4 向量化撞生产指针非对齐坑致整管发散。

**原始证据：** 2026-10-04 C2b 记录（三层）：(1) **架构否定**——swiglu 与 gate 分别是 fc1 与 fc2/out 三个不同 GEMM 的 i32 累加器 epilogue，cublasLt INT8 不支持自定义 epilogue；(2) vec4 版独立 kernel（`H3_DIT_FFN_EPI_FUSE`）在隔离 harness 逐位一致，但整管确定性发散——根因=生产指针为子分配、非规范对齐（seq=15477 非 32 对齐），float4 与 scalar 边界不等价；(3) 热步无 dit 加速（OFF 10404 / ON 10305ms，~1%）。

**审计点：是否只关闭了 vec4 这一实现路径？**

**是，原档案本身是诚实的，但下游引用发生了漂移。** 分层核实：

| 层级 | 表述 | 是否准确 |
| --- | --- | --- |
| C2b 原始记录（plugin-plan §5） | 「真正折进 GEMM epilogue 只能手写 GEMM（C5b 领域），cublasLt INT8 不支持自定义 swiglu-dequant epilogue」 | **当时准确，但有一个盲区：「只能手写 GEMM」排除了 CUTLASS 库的 EVT 路线**——EVT 不需要「手写」GEMM（库提供 collective），也不假设生产指针对齐（epilogue 在 GEMM tile 内操作，tile 天然对齐）。2026-10-04 时未调研 CUTLASS 4.x EVT 的 sm_120 支持 |
| roadmap §0.3 死路清单 | 「FFN epilogue vec4（实现死：vec4 向量化撞生产指针非对齐坑…；**未关闭方向：GEMM+epilogue 融合本身**——非 vec4 的实现路径待审计，quant/dequant 中转 1.465s 仍开放）」 | **准确**——已正确限定「死的是 vec4 路径」，并显式留出未关闭方向 |
| roadmap §5 不做清单 | 「不再做 INT8 GEMM raw-FLOPS 优化（autotune/**手写核**/**epilogue 减往返**）」 | **过宽**——把「epilogue 减往返」整族（含 CUTLASS EVT 这种从未试过的路径）并入了禁做清单，与 §0.3 自己的「未关闭方向」注记自相矛盾 |

**证据充分性判定：** 对「vec4 路径死」充分（隔离 harness + 整管 A/B + 根因定位三层）。对「GEMM+epilogue 融合方向死」**无证据**——CUTLASS EVT、scalar 融合、dequant+requant 对撞融合（E2）都从未施工。

**裁决：修正。** 修正内容：
1. **死路维持部分：** vec4 向量化实现路径 + cublasLt 原生 epilogue 表达能力（sm120-operator-frontier §2.3 已钉死：cuBLASLt INT8 epilogue 无 dequant/无 SwiGLU，物理不可表达）。
2. **须从 §5 不做清单拆出的部分：** 「epilogue 减往返」改为「epilogue 减往返的 **vec4/cublasLt-native** 实现路径」；CUTLASS EVT 融合（frontier E1，预估 −0.6~−1.0s）与 dequant+requant 对撞融合（E2，−0.2~−0.4s）登记为**待施工新方案**，与 C2b 的区别已由 frontier §2.1/§2.2 论证（不写独立 kernel、不假设指针对齐）。
3. **重测方案：** 采纳 frontier 的建议——第 0 步只做 8 形状 CUTLASS-vs-cublasLt 离线 bench（**机时 ~0.5 天**，含环境搭建），GEMM 本征不慢于 cublasLt 3% 才进入 EVT 施工；否则 E1 降级、E2 备胎顶上。

---

## 4. sol KV fuse 死路（C1b）——【维持】

**原结论：** kc+vc 融单核 bit-identical 但零收益，reduce 已访存饱和。

**原始证据：** 2026-10-04 `kpia-sol-c1aux`：① **微基准实测**（非推算）：T=15477 下 orig 两核 0.278ms vs 融合单核 0.278ms（GPU 时间 0.275→0.275ms）；② **访存饱和论证**：reduce 每核读 227MB，实测带宽 ~1.6TB/s（5090 显存带宽 ~1.8TB/s 标称，89% 利用），融合省不掉数据量；③ **nsys stream 分桶**：DiT 窗口内 kc/vc/diag/stats 各 199×，合计 ~0.086s/clip；④ 整管 A/B：ctrl-a vs ctrl-b dit 差 +17.5ms、total 差 +470ms（后者为 VAE/ppm 段噪声）。

**审计点逐条核实：**

| 审计问题 | 核实结果 |
| --- | --- |
| 访存饱和的测量方法？ | **微基准实测 + 带宽推算双源**，不是纯推算。0.278ms 读 227MB → 816GB/s/核；两核并行/流水下合计贴近 1.6TB/s 实测带宽。方法可靠 |
| 「零收益」的测量噪声？ | 窗口内理论上限 = 省 200 次 launch ≈ 几 ms（0.086s 中 launch 占小部分），而 run-to-run 噪声 ±0.5s（480P e2e）/ ±17ms（dit 段实测）。**「零收益」= 真实收益（<10ms）远小于噪声下限（17ms），物理上不可分辨**——但这不是「测不准所以可能有益」，而是「即使有益也无量级意义」 |

**证据充分性判定：** 充分。且有两个独立的封死论证：① 数据量不变 → 访存饱和下不可能快；② 窗口内总权重 0.09s/clip 的 100% 回收也不到噪声的 1/5。**即使访存饱和论证有误，量级论证仍然封死。**

**裁决：维持。** 不需要重测。窗口外 aux（warmup/autotune 1.05s 是否 per-clip）是另一个问题，已由 sol-attn 审计 R3 登记（0.5h profile），不属于本条。

---

## 5. CUDA Graph（P2b，−0.16s 微薄）——【修正】

**原结论：** 已交付但收益微薄——hot DiT 12.61→12.63s 持平、e2e −0.16s，turbo4 仅 4 步 launch 税被 cold 噪声掩盖。

**原始证据：** 2026-10-05 交付记录：整步单次捕获/replay，238 hot clip2 checksum 位一致；hot DiT 12.61→12.63s；e2e −0.16s。

**审计点逐条核实：**

| 审计问题 | 核实结果 |
| --- | --- |
| hot DiT 12.61→12.63s 是单次还是多次均值？ | **档案中只出现一对数值，未记录 run 数与统计方法**——这是全清单中证据强度最弱的一处。对照同文档其它条目（C5a 有 run1/run2 + ctrl-off、C8 有 ON/OFF 各 2 run），P2b 的 A/B 只引用了单次（或至少未报告重复次数） |
| 热 clip 单独测量时 Graph on/off 的 DiT 段差异？ | **未做。** 记录的是含 cold 的 e2e（−0.16s）+ 单次 hot DiT 对。480P run-to-run 噪声 ±0.5s，−0.16s 与「DiT 持平」**都在噪声带内**——严格说，现有证据只能支持「收益 < 噪声（±0.5s）」，不能支持「收益 = −0.16s」或「DiT 段收益 = 0」 |

**证据充分性判定：** 结论方向大概率正确（launch 税的理论上限可估：DiT 稳态 ~2000 核 × 省 ~3μs/launch ≈ 6ms，加上 graph 消除的 host 间隙，合理预期就是 0.1–0.3s 量级，确实微薄），但**证据强度低于档案宣称的确定度**——「12.61→12.63s 持平」和「−0.16s」都是噪声带内的单次观测，本质上是「测不出差异」而非「证明无差异」。

**裁决：修正（措辞级，非方向级）。**
1. 结论从「收益微薄 −0.16s」修正为「**收益 < ±0.5s 噪声带，实测不可分辨；理论上限 ~0.3s（2000 核 × 微秒级 launch 税），与实测不矛盾**」。不翻默认的决策**维持**（无论收益是 0 还是 0.16s 还是 0.3s，都不够翻默认的门槛）。
2. **重测方案（低优先，0.5h）：** 同 binary 同 seed，Graph on/off 各 repeat=4，只取 hot clip 的 `phase_ms dit=` 报 P50。预期结果：差异 <0.3s 且不稳定——则封档；若意外出现稳定 >0.5s 差异，说明 launch 税模型有漏，再评估。**不重测也不阻塞任何档位**（T1/T2 缺口是 2.7–6.6s，0.3s 上限无战略意义）。

---

## 6. VAE tile batch 4→8 回归（−0.36s）——【维持】

**原结论：** 实测稳定 −0.36s 回归（L2 命中下降抵消行合并收益），产品默认保持 batch=4。

**原始证据：** 2026-10-01 文档 §8.6（2026-10-04 续拍 3）：双臂 × 各 repeat=2 复测——batch=4：57.317s/57.430s；batch=8：57.675s/57.758s。回归 0.36s/0.33s，**两倍于同臂复测噪声（±0.11s）**；`vae_blocks` 9.88→10.16s 反升；质量对照：同 seed 两臂 latent checksum 均 `0x5fc00e592bac60a4`（DiT 输出一致，VAE 输入相同），waffle 同区间，逐帧亮度相关 0.9973。

**审计点逐条核实：**

| 审计问题 | 核实结果 |
| --- | --- |
| 回归是否稳定复现？ | **是。** 4 个数据点（2 臂 × 2 run），回归方向一致、量级一致（0.33–0.36s），噪声带 ±0.11s 之外。对照实验设计干净：同 seed、同 binary、仅 `H3_VAE_TILE_BATCH` 差异、DiT checksum 相同排除了上游变量 |
| L2 命中下降的解释有没有 nsys 证据？ | **没有。** 原文给出的解释是「GEMM 行合并的收益被更大 batch 的激活 workspace 压力与 L2 命中下降抵消」——这是合理推断（batch 翻倍 → 激活 footprint 翻倍 → L2 工作集溢出），但**未做 nsys L2 hit-rate profile 证实**。替代解释（如 cuDNN SDPA 在 n_batch=6 vs 4 下选了更慢的 plan）未排除 |

**证据充分性判定：** 对决策「维持 batch=4」**充分**——回归稳定复现于噪声外，无论根因是 L2 还是 SDPA plan 还是 workspace 压力，结论不变。L2 解释只是叙事层装饰，缺 nsys 证据不构成决策风险。

**裁决：维持。** 可选补强（不阻塞）：一次 nsys `l2_tex_hit_rate` 对照 profile（0.5h），把 L2 解释从推断升级为实证——**仅当未来有人想以「调 tile 形状/overlap 救回 batch=8」为由重开时才有必要**。当前无此需求，不建议花机时。

---

## 7. mux overlap 死路（C8）——【维持，建议降级归档】

**原结论：** 编码管道本已流式（音/视双线程喂 ffmpeg），提前喂帧无净增益。

**原始证据：** 2026-10-04 C8 记录：`H3_MUX_OVERLAP=1` 对照（hot clip2）——OFF total=16382.2/16325.7、mux=449.2/442.5；ON total=16426.7/16439.1、mux=471.7/433.2。**无净增益（噪声内）**。根因分析：`h3_ffmpeg_write_av_rgb24_f32_ex` 已是音频线程+视频线程双管道喂 ffmpeg，x264 ~200ms 已在管道内流水。

**审计点核实：**
- 双臂各 2 run，total 差 +44ms/+114ms（方向还偏负），mux 单项两臂重叠——「无净增益」判定可靠。
- 「管道本已流式」的根因有源码级佐证（双线程管道是既有架构，非推测）。

**量级复核（审计问题 7）：** T1 档 x264 ultrafast 后 mux encode 已压到 ~34.5ms（P3 记录：92→34.5ms），mux 单项占 e2e 的 0.17%。即便 overlap 完美消除全部 mux 串行暴露，上限 ~35ms——**该条死路的战略价值已归零**（不是结论错了，是结论对应的靶子消失了）。

**裁决：维持结论，建议处理动作降级**——从 roadmap §0.3 死路清单（与 INT8 GEMM 等并列）移到纯档案位（plugin registry 的 DEAD_END_VERIFIED 已足够），避免在路线图上占用读者注意力。不重测。

---

## 8. x264 veryfast/ultrafast 正收益（−260ms/−360ms）——【维持】

**原结论（正向）：** veryfast mux −260ms（2×快）画质在地板内、checksum 双精确；ultrafast 再省 ~100ms 但画质损失明显，仅 T1 预览可接受。

**原始证据：** 2026-10-05 记录：veryfast mux 483-507ms→227-244ms（多次复现），hot e2e 16.40→16.15-16.19s；默认路径 argv 逐字节一致、latent checksum 双精确（clip1/clip2 与金线一致）。P3（2026-10-06）复验：ultrafast encode 92→34.5ms（−62.5%）。

**审计点：测量口径是否可靠？**

- **速度口径：可靠。** mux 单项 483→227ms 是 ~250ms 量级、远超 mux 段自身噪声（±10ms）；多次复现方向一致；latent checksum 双精确证明了变量隔离（x264 preset 只改封装层，不动 latent——这个因果链本身就是口径正确的证据）。
- **画质判定口径：带主观成分但量级安全。** 「veryfast 画质在质量地板内」的判定依据是实测判定（waffle 不受 x264 影响——waffle 测的是 latent 解码前质量，x264 在其下游；真正的画质风险是编码压缩伪影，靠抽帧人眼）。waffle 对 x264 不敏感是**结构性正确**的（度量对象在编码器上游），残余风险只有「人眼对 veryfast 伪影的容忍度」这一主观判断——但 T2 默认仍 veryfast 已上线、T1 ultrafast 是预览档，风险敞口与判定精度匹配。

**裁决：维持。** 无重测需求。唯一建议：若 T2 档对终端用户开放 veryfast 出片，补一次双人盲测确认压缩伪影可接受（流程门，非技术门）。

---

## 9. 汇总：修正/重开条目对路线图的影响

| 条目 | 对 roadmap 的修改建议 | 机时 |
| --- | --- | --- |
| #3 FFN epilogue | §5 不做清单「不再做 INT8 GEMM raw-FLOPS 优化（…epilogue 减往返）」**收窄为「…epilogue 减往返的 vec4/cublasLt-native 路径」**；CUTLASS EVT（E1）与对撞融合（E2）从「被误并禁做」恢复为 T1-e 候选刀，第 0 步 = 8 形状 CUTLASS bench | **~0.5 天**（bench 证伪/证实） |
| #5 CUDA Graph | §0.3/T1-f 的「−0.16s 实测」**改注「噪声带内不可分辨，理论上限 ~0.3s」**；不翻默认的决策不变 | 可选 0.5h（不阻塞） |
| #1 INT8 autotune | 可选补一刀 1GiB workspace 实验彻底封档 | 可选 0.5h（不阻塞） |
| #7 mux overlap | 从 §0.3 死路清单降级到档案（靶子 ~35ms 已消失） | 0（文档动作） |

**重开条目合计预期收益：** E1（−0.6~−1.0s）+ E2（备胎 −0.2~−0.4s）——**这正是 T1/T2 目标重估所依赖的 [审计依赖] 变量**：T2 缺口 2.72s 中，E1 兑现中位可覆盖 ~30%；T1 缺口 6.6s 中覆盖 ~12%。**不足以单独闭合任何档位目标，但它是目前唯一证据链干净、未施工、量级过 0.5s 的算子层收益源。**

**本次审计未发现需要推翻的硬结论**——8 条中 6 条维持、2 条修正（均为证据措辞/禁做范围收窄，非方向翻转）。与 sol-attn 审计（发现 1 处过度泛化 + 1 处噪声伪影）相比，算子类档案的质量更高，主要遗留是 **CUTLASS EVT 这个 2026-10-04 时未调研到的盲区**被 §5 误并——这是本次审计唯一有实际工程价值的发现。
