# 额外优化机会（相对 10-26 路线图的增量扫掠）

> **日期：** 2026-10-27（Asia/Singapore）。  
> **动机：** 在 [`2026-10-26-optimization-roadmap.md`](2026-10-26-optimization-roadmap.md) / DeepInfer 速度策略 / followup-dev-plan 已定主线之后，再扫热路径与 10-05～10-09 研究，列出**尚未进 10-26 执行表、或被低估**的机会。  
> **约束（本轮）：** 不跑 GPU/238；不碰 NVFP4 链式 WIP；只写文档。  
> **NVFP4 P0 视觉门：** 仍是主算力杠杆，本文**不当新发现**，仅作上下文。

---

## 0. 扫掠范围与方法

| 已读 | 热路径 / 代码 |
|---|---|
| `2026-10-26-optimization-roadmap.md`、`deepinfer-speed-strategies.md`、`followup-dev-plan.md` | `h3_dit_w4a8.c` / `h3_dit_fused_epi.h` / `h3_gpu_cuda.{h,cu}`（E2） |
| `2026-10-05-acceleration-mechanisms.md`、`quality-preserving-acceleration.md` | `h3_vae_int8_cuda.c`（unpatch async 已默认）、`h3_vae_prewarm.c` |
| `2026-10-07-sm120-operator-frontier.md`、deadend-audit-{operators,memory}.md | `h3_te_cache.*`、`tools/h3_cuda_generate.c` TE 段 |
| `2026-10-09-generation-timing.md`、`2026-10-26-te-lifecycle-audit.md` | `out/review-2026-10-07/e2-epi-fuse/`（代码齐、238 未验） |
| `docs/superpowers/plans/2026-10-05-h3-optimization-roadmap.md`（SKU 账本） | SDPA registry / sol warmup 叙事 |

**刻意不重复当「新项」：** P0 NVFP4 视觉门、PDMD/DMAD 少步合同验收、链式提交、DeepInfer 算子盲搬、已闭环的 VAE unpatch async（默认开）、已判死的 QKV merge / vec4 epi / mux overlap / MagCache@4step。

---

## 1. 排序表（杠杆 × 可行性）

分数为相对序（1=最高），非绝对秒数承诺。墙钟数字来自既有 nsys/插桩文档，**须授权后重测才可进 KPI**。

| 序 | ID | 机会 | 杠杆 | 可行性 | 档位建议 |
|---|---|---|---|---|---|
| 1 | **A1** | **E2 `H3_DIT_EPI_FUSE` 真机 A/B + 可选翻默认** | 高（同精度链路秒级） | 高（代码已齐） | P1∥P2（授权 238） |
| 2 | **A2** | **768p 显存账本插桩（R-E）→ 再定 streaming/窗/FP16** | 高（能力轴决策前提） | 高（纯插桩） | P2 前置 / 能力 |
| 3 | **A3** | **TE 常驻与 `te_cache` 产品化 + `c_te_raw` 债** | 中高（TE 7–58s 摆动） | 中 | P1 工程 |
| 4 | **A4** | **激活 FP16 中转（M2）作 768p 备胎** | 中高（腾 2–4GiB） | 中 | P3 备胎 |
| 5 | **A5** | **sol / Triton autotune 持久化（L1）** | 中（cold ~1s） | 中（先证实 per-clip） | P2 cold |
| 6 | **A6** | **L2 `accessPolicyWindow` pin 小热表** | 低中（−0.05~−0.15s） | 高 | P3 小刀 |
| 7 | **A7** | **cast/prescale 与 VAE SIMT sgemm/TF32 衍生** | 中（VAE 段） | 中 | P2 VAE |
| 8 | **A8** | **E2-P3 跨 phase（FC2→QKV）requant 接线** | 中（E2 余量） | 中 | A1 通过后 |
| 9 | **A9** | **窗 30→24 一次性实验（不解默认）** | 中（可能解 530MiB） | 高 | A2 后探针 |
| 10 | **A10** | **generate 启动自动打 env 合同（N1 内联）** | 低（防测错） | 高 | P0-dev |
| 11 | **A11** | **cold-clip 间歇高噪方向 B** | 中（质量/可复现） | 低中 | 不插队债 |
| 12 | **A12** | **VDN / SelfLift / TAE 作 SKU 分档（非默认金线）** | 产品轴 | 中 | 观察 / T1·T3 |
| 13 | **A13** | **VAE restore 指针重绑 / FP8 权重缓存（T4-c 重开）** | 低（~0.3s 余尾） | 中 | P3 |
| 14 | **A14** | ~~**PDL 挂 GEMM→dequant 短链**~~ | — | — | **CLOSED @4090**（CC 8.9）[`c016-close`](2026-10-28-c016-pdl-dig-close.md) |
| 15 | **A15** | **集中 `h3_env` 注册表（S11 远期）** | 低（工程卫生） | 中高工 | 观察 |

---

## 2. 条目详表

### A1 — E2 `H3_DIT_EPI_FUSE` 真机验收（最被低估）

| 栏 | 内容 |
|---|---|
| **是什么** | 把「上一个 GEMM dequant epi」与「下一个 GEMM act requant」对撞进同一 kernel，省 FP32 HBM 往返 + 下游 ConvRot/quant prep。开关 `H3_DIT_EPI_FUSE=1`（默认 0）。 |
| **为什么** | INT8 **GEMM 本征**已近峰；真正开放的是 **链路中转**（nsys：quant/dequant ~1.465s/clip）。E1 CUTLASS EVT 已在 sm_120 证伪；**E2 是留下的可施工主刀**，却几乎没出现在 10-26 路线图（只笼统写「大核旁消中转」）。 |
| **仓内证据** | `h3_dit_fused_epi.h`、`h3_gpu_cuda.cu` fused kernels、`h3_dit_w4a8.c` P1/P2 接线；`out/review-2026-10-07/e2-epi-fuse/DESIGN.md` + `ab.sh`：**代码齐，238 A/B 从未跑**。预估 −0.2~−0.4s/clip（保守）；非保证 bit-identical（f16 ConvRot 舍入路径变）。 |
| **工作量** | 授权后按 `ab.sh` 跑 hot clip2 ≥2 轮；日记入 timing；waffle 同配置门。 |
| **风险** | 数值噪声超 waffle；P3（FC2→QKV）未接线，勿一次翻默认。 |
| **关系** | **S5** 实体化；roadmap P2「融合」的可执行第一步；**正交于 NVFP4**。 |

### A2 — 768p 显存账本插桩（决策前提）

| 栏 | 内容 |
|---|---|
| **是什么** | 在权重窗 / DiT WS / cuDNN WS / QKVO / fwd 临时等关键节点 `cudaMemGetInfo` 分桶，重画 28.8GiB 叙事。 |
| **为什么** | deadend-audit-memory **重开** 768p 全 sol OOM：旧结论只否定了 torch 侧释放策略，却被泛化为「全 sol 不可行」。layerwise streaming / 激活 FP16 / 窗交换都从未评估。没有新账本，P3 streaming 容易拍脑袋。 |
| **仓内证据** | `2026-10-07-deadend-audit-memory.md` 条目 1；VDN 同卡 768p INT8 剩 ~8.3GB；SGLang layerwise −30GB 指针。 |
| **工作量** | ~0.5 天插桩 + 一次 768p 跑（授权）。 |
| **风险** | 低（只读探针）；盘满时勿开 nsys。 |
| **关系** | **S2** 落地前提；roadmap P3 能力轴；先于 A4/A9/streaming 大改。 |

### A3 — TE 常驻 / `te_cache` 产品化 + `c_te_raw` 债

| 栏 | 内容 |
|---|---|
| **是什么** | (1) 多 clip / 同 prompt 复验时强制或默认 `te_cache` HIT；(2) 进程内 TE 权重常驻与 `upload_ms` 可观测；(3) 单独修 `c_te_raw` 偶发不确定。 |
| **为什么** | timing 账本：TE phase 从 ~7s 到 ~58s 随上传/冷热剧变；全程 total 常被 TE 污染。DeepInfer **S1 重叠**在本仓最肥的「准备活」就是 TE。10-26 只记了「生产债不插队」，未给**可执行产品化步骤**。 |
| **仓内证据** | `h3_te_cache.c` 已有 load/save；`h3_cuda_generate.c` 支持 `--te-cache`；`2026-10-09-generation-timing.md` F1/F2 TE 摆动；`te-lifecycle-audit.md`。 |
| **工作量** | 中：默认策略、HIT 日志硬门、与 P0 env 合同对齐；`c_te_raw` 根治另开。 |
| **风险** | 错缓存污染金线 → 必须 key 含 layers/dim/token-ids；VISION span 路径已跳过 cache。 |
| **关系** | **S1 + S9 + S12**；不替代 NVFP4，但让加速比可信。 |

### A4 — 激活 FP16 中转（M2，768p 备胎）

| 栏 | 内容 |
|---|---|
| **是什么** | `d_wide` / `d_qkv` 等宽中转槽改 FP16/BF16 存储，降低激活峰值。 |
| **为什么** | 预估腾 **2–4GiB**，量级盖住 sol fwd ~530MiB 缺口；NVFP4 权重治本之前的**能力备胎**。 |
| **仓内证据** | sm120-frontier §M2；deadend-memory 条目 1 ④「从未施工」。 |
| **工作量** | 2–3 天 + waffle（**非 bit-identical**）。 |
| **风险** | 中：精度；与 INT8 quant 链交互。 |
| **关系** | **S2**；排在 A2 账本之后；失败则上 layerwise streaming。 |

### A5 — sol / Triton autotune 持久化（L1）

| 栏 | 内容 |
|---|---|
| **是什么** | 确认 ~1.05s warmup/autotune 是 per-process 还是 per-clip；若每 clip 付，则按 (T, shape, GPU) 落盘缓存。 |
| **为什么** | sm120-frontier：**DiT 稳态小核只剩 ~0.09s**；唯一有肉的 launch 税是这 1.05s 是否重复付。10-26 完全没提。 |
| **仓内证据** | `2026-10-07-sm120-operator-frontier.md` §4 / L1。 |
| **工作量** | 先 0.5 天对照冷热日志；再接 sol-engine cache。 |
| **风险** | 低；若已是 cold-only 则 **关闭该项**（诚实）。 |
| **关系** | **S6** phase 税；只影响 cold / 首 clip KPI。 |

### A6 — L2 residency pin 小热表

| 栏 | 内容 |
|---|---|
| **是什么** | `cudaStreamSetAttribute` accessPolicyWindow，把 ConvRot 旋转矩阵、RoPE、modulation、`s_channel` 等每层复读小表钉 L2。 |
| **为什么** | 5090 L2=96MB；层权重超 L2，但小热表可 pin；预估 −0.05~−0.15s/clip，**bit-identical**。 |
| **仓内证据** | sm120-frontier §5.3 M3；权重侧 stream overlap **已做**，L2 pin **未做**。 |
| **工作量** | 低（host 属性）。 |
| **风险** | 低；收益小，勿当主叙事。 |
| **关系** | **S2** 微优化；可与 A1 并行（授权后）。 |

### A7 — cast/prescale 与 VAE 段 SIMT sgemm

| 栏 | 内容 |
|---|---|
| **是什么** | 拆开 nsys 里 cast/prescale **2.29s**（其中 ~0.9–1.2s 可融余量）与 VAE 段 SIMT sgemm ~1.15s；VAE f16 的 TF32/融合衍生。 |
| **为什么** | unpatch D2H 已 −99%；**vae_blocks 主体仍是 fp16 真算力**。T1/T2 缺口（final-accept 后仍欠数秒）需要 VAE/cast 侧新源，不能只盯 DiT GEMM。 |
| **仓内证据** | quality-preserving §2；sm120-frontier 登记「VAE 族指针未施工」；roadmap T1 缺口叙事。 |
| **工作量** | 中高；需 nsys 授权。 |
| **风险** | TF32 非 bit-identical（T1 允许、T2 慎）。 |
| **关系** | **S5** 扩到 VAE；与 A1 互补（DiT 链 vs VAE 链）。 |

### A8 — E2-P3 跨 phase requant（FC2→QKV）

| 栏 | 内容 |
|---|---|
| **是什么** | DESIGN.md 已留 mode 3；打通 `block_phase_out_mlp` → `block_phase_qkv`。 |
| **为什么** | P1/P2 只吃函数内闭环；跨 phase 还有一次 act 往返。 |
| **仓内证据** | `e2-epi-fuse/DESIGN.md` §5。 |
| **工作量** | 中；依赖 A1 数值门。 |
| **风险** | 生命周期 / stream 事件更难。 |
| **关系** | A1 扩展；**S5**。 |

### A9 — INT8 窗 30→24 探针（不解默认）

| 栏 | 内容 |
|---|---|
| **是什么** | 一次性实验缩小权重窗，换 ~GiB 级显存，看全 sol@768p 是否过线。 |
| **为什么** | deadend-memory：从未与「不减窗」红线做显式权衡。 |
| **仓内证据** | 同上审计条目 1 ②。 |
| **工作量** | 低（env + 一跑）。 |
| **风险** | H2D 更频 → 可能变慢；只作能力探针。 |
| **关系** | A2 之后；**S2** 廉价试探。 |

### A10 — generate 启动内联 env 合同

| 栏 | 内容 |
|---|---|
| **是什么** | `h3_cuda_generate` 启动时打印与 `print_h3_env_contract.py` 同构的生效键（或 `--print-env-contract`）。 |
| **为什么** | 离线工具已有；日志里仍可能漏打 → 加速臂对照翻车（DeepInfer S9/S11）。 |
| **仓内证据** | `tools/print_h3_env_contract.py`；deeper-pass N1。 |
| **工作量** | 低。 |
| **风险** | 极低。 |
| **关系** | **S9/S11**；P0-dev 可立即做（无 GPU）。 |

### A11 — cold-clip 间歇高噪方向 B

| 栏 | 内容 |
|---|---|
| **是什么** | 根治候选：vae_th 竞态 / pinned restore / INT8 make_room（方向 A cold guard 已落地）。 |
| **为什么** | 不挡发布，但污染 cold waffle 与「可复现产品」叙事。 |
| **仓内证据** | 10-05 roadmap T3-c。 |
| **工作量** | 高不确定性。 |
| **风险** | 时序敏感。 |
| **关系** | **S9**；不插队。 |

### A12 — VDN / SelfLift / TAE 分档 SKU

| 栏 | 内容 |
|---|---|
| **是什么** | VDN 作长视频/高分质量层；SelfLift/LBH 作预览；TAE 仅非金线 opt-in。 |
| **为什么** | 10-26 优化轴偏「同合同加速」；**产品分档**可另开墙钟曲线（VDN 在长片线性优势）。 |
| **仓内证据** | `2026-10-07-vdn-direct-use-reanalysis.md`；`h3_selflift.*`；`h3_tae.*`。 |
| **工作量** | 高（集成/合同）。 |
| **风险** | 与默认金线混淆 → 必须独立 SKU。 |
| **关系** | **S7** 合同变体的产品面；观察项。 |

### A13 — VAE restore 指针重绑 / FP8 缓存（T4-c）

| 栏 | 内容 |
|---|---|
| **是什么** | 消 clip 间 ~0.3s VAE restore 余尾；PR#36727 风格指针重绑或不增常驻的 FP8 设备权重。 |
| **为什么** | 「不可闭合」被审计改为「当前 INT8+FP16 权重策略下」；量级小但手法可复用。 |
| **仓内证据** | deadend-memory 条目 3；`h3_vae_prewarm.c`。 |
| **工作量** | 中。 |
| **风险** | 与 INT8 窗争显存。 |
| **关系** | **S1/S2**；P3。 |

### A14 — PDL 短链

| 栏 | 内容 |
|---|---|
| **是什么** | Programmatic Dependent Launch 消 GEMM→dequant→quant 气泡。 |
| **为什么** | transfer【部分迁移】；预估 −0.1~−0.3s，排融合后。 |
| **仓内证据** | llm-to-dit-transfer；sm120 L2。 |
| **工作量** | 中；需 sm_90+ 路径验证。 |
| **风险** | 调试难；收益小。 |
| **关系** | **S5/S8** 尾部。 |

### A15 — 集中 `h3_env` 注册表

| 栏 | 内容 |
|---|---|
| **是什么** | 远期把散落 `getenv` 收成可枚举合同表（DeepInfer environ.py 同族）。 |
| **为什么** | 防 `off`≠false；失败分类。 |
| **仓内证据** | speed-strategies S11；deeper-pass。 |
| **工作量** | 高（重构面）。 |
| **风险** | 干扰热路径；勿挡主线。 |
| **关系** | **S11** 观察。 |

---

## 3. 与 10-26 路线图的衔接（不改 P0 优先级）

```
P0 主线（不变）     INT8 纪律 → NVFP4 视觉门（上下文，非本文新项）
       │
       ├─∥ A10 env 内联          （随时可开发）
       ├─∥ A1 E2 真机            （授权 238；S5 实体）
       ├─∥ A3 TE cache 产品化    （测量可信）
       │
P1/P2 并行           PDMD/DMAD GPU 验收（已有）
       │
       ├─ A2 显存账本 → A9 窗探针 → A4 FP16 中转 → streaming
       ├─ A5 autotune 证实 → 可选持久化
       ├─ A7 VAE/cast 剖析（nsys 授权）
       └─ A6 L2 pin（小刀）
```

**明确仍 defer：** FlashMLA/DeepEP/投机核；W4A8 主题默认；有损 MagCache 默认；多卡；NVFP4 链式（WIP 不动）；E1 CUTLASS EVT（已证伪）。

---

## 4. Top 5（给执行层 / 用户转述）

1. **E2 epi-fuse 真机 A/B** — 代码已在仓里躺着，是同精度下最具体的秒级刀。  
2. **768p 显存账本插桩** — 没有它，streaming / 减窗 / FP16 中转都是猜。  
3. **TE cache / 常驻产品化** — 让 TE 不再把加速比测歪。  
4. **激活 FP16 中转（账本后）** — 768p 能力备胎。  
5. **sol autotune 是否每 clip 付** — 可能白捡 cold ~1s，或诚实关闭。

---

## 5. 相关文档

| 路径 | 关系 |
|---|---|
| [`2026-10-26-optimization-roadmap.md`](2026-10-26-optimization-roadmap.md) | 主路线图；应链到本文 |
| [`2026-10-26-deepinfer-speed-strategies.md`](2026-10-26-deepinfer-speed-strategies.md) | S1–S12 策略层 |
| [`2026-10-07-sm120-operator-frontier.md`](2026-10-07-sm120-operator-frontier.md) | E2/L1/M2/L2 原始方案 |
| [`2026-10-07-deadend-audit-memory.md`](2026-10-07-deadend-audit-memory.md) | 768p 重开 |
| [`out/review-2026-10-07/e2-epi-fuse/DESIGN.md`](../../out/review-2026-10-07/e2-epi-fuse/DESIGN.md) | E2 施工记录 |

---

## 6. 修订记录

| 日期 | 说明 |
|---|---|
| 2026-10-27 | 初稿：增量机会 A1–A15；强调 E2 已施工未验收与显存账本前置。 |
