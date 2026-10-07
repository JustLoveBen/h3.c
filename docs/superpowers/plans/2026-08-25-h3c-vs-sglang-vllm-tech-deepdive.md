# h3.c vs SGLang Diffusion vs vLLM-Omni：工程/架构级深度技术对照

> **后续开发方案见** [2026-08-26-h3c-next-tech-plan.md](./2026-08-26-h3c-next-tech-plan.md)  
> **2026-09-11 五方案普查（多卡 5090，不改金线）：** [2026-09-11-h3c-accel-survey-multigpu-5090.md](./2026-09-11-h3c-accel-survey-multigpu-5090.md)

日期：2026-08-25  
仓库：`h3.c`（纯 C/CUDA MiniMax-**H3** 视频+音频推理引擎）  
本文是 [浅层产品对照](./2026-08-25-h3c-vs-sglang-vllm-omni.md) 的**技术加深版**：回答机制上谁赢谁输、对本仓 **INT8 8-step 金线**意味着什么。  
**不编造未测秒数**；公开材料不足处标明；不改 generate、不 commit、不写密码。

**必读底稿：** [accel-backends-cache-matrix](./2026-08-25-h3c-accel-backends-cache-matrix.md) · [4090-accel-plan](./2026-08-25-h3c-4090-accel-plan.md) · [sla-vs-h3c](./2026-08-25-sla-vs-h3c.md) · [multi-gpu-arch](./2026-08-25-h3c-multi-gpu-arch-speed.md)

**外部原文（本轮 WebFetch）：**  
SGLang cookbook [MiniMax-H3](https://lmsysorg.mintlify.app/cookbook/diffusion/MiniMax/MiniMax-H3) · vLLM-Omni [recipes/MiniMaxAI/MiniMax-H3.md](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md) · [Cache-DiT design](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/cache_dit/) · [Sol-Engine H3-OnDevice](https://nvlabs.github.io/Sana/Sol-Engine/H3-OnDevice/) · MagCache [arXiv:2506.09045](https://arxiv.org/html/2506.09045v1) · TeaCache [README](https://github.com/ali-vilab/TeaCache) · FBCache [Comfy-WaveSpeed](https://github.com/chengzeyi/Comfy-WaveSpeed)

---

## 0. 结论先行（技术定位，非秒数口号）

1. **三者不在同一运行时哲学上竞争。**  
   - **h3.c**：静态编译的 **单卡、常驻权、fused 算子管道**；质量门是「相对同机 cuDNN 金线 PPM MAE 不可辨」。  
   - **SGLang / vLLM-Omni**：Python **serving 运行时**（调度、拓扑、请求级 `quality`、HTTP）；质量门是「相对同配方 lossless 的 SSIM/PSNR 轨迹偏差」+ 审计拓扑锁。

2. **机制上框架真正赢 h3.c 的地方：**  
   - **步间块级 cache（Cache-DiT / TeaCache / FBC）** 作为一等公民，且在 **50-step BF16** 轨迹上有大可跳空间；  
   - **序列并行（Ulysses/SP）+ 请求调度** 把延迟账本从「单卡墙钟」改成「多卡通信/offload」；  
   - **请求级 quality 切换**（同一进程挂/卸 Cache-DiT hooks）与 Ref2VA/FL2VA 合同深度。

3. **机制上 h3.c 真正赢框架的地方：**  
   - **无 Python 调度税**的稠密路径：INT8 ConvRot GEMM + fused cuDNN SDPA 已是 Ada/Blackwell 消费卡上「相对 eager BF16」的**后融合态**；  
   - **误差预算单点可控**：金线只有 INT8 权 + dense SDPA，不叠 Online FP8 / sparse / step-skip；  
   - **24GB 真 5s 有声可跑**：框架消费卡路径依赖 layerwise + 数百 GiB 主机 RAM，延迟故事是另一把尺子。

4. **对本仓 8-step INT8 金线，理论收益天花板（Amdahl，用已测分解）：**  
   - 4090 cafe：`te≈11.6 / dit≈47.5 / vae≈9.3 / 其它≈7 → total≈75s`。  
   - **注意力替换**（实测）：DiT −14–15%，e2e 常被 TE 吃光；**MAE 不过门** → 现阶段 **e2e 有效天花板 ≈0**（质量路径）。  
   - **整步 skip**（跳 1–2 步）：稳态 ~5.3s/step → **最多 −5.3～−10.6s**；MagCache 已证视觉 FAIL；Tea/FBC 未证。  
   - **块级 Cache-DiT 思想**：在 8-step 上若只跳中间层而非整步，收益可能介于「0」与「整步 skip」之间，但**误差耦合更密**（见 §3）；本仓未实现。  
   - **条件图 vision host SDPA→cuDNN**：T2VA 金线 **0 收益**；F2VA/L2VA 才是 KEEP 刀。

5. **先前浅层分析粗在哪：** 主要停在产品表、公开秒数禁横比、缺口清单；**没有**把「探测信号 / 跳过粒度 / 误差叠加 / 运行时同步税 / 质量门口径」拆到机制层。本文补的是这些。

---

## 1. 架构栈对照（一层层）

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 产品合同：T2VA / FL2VA / Ref2VA · 有声 · 4–15s · 质量档                 │
├───────────────┬─────────────────────────┬───────────────────────────────┤
│ h3.c          │ SGLang Diffusion        │ vLLM-Omni                     │
├───────────────┼─────────────────────────┼───────────────────────────────┤
│ CLI/库        │ sglang serve /generate  │ vllm serve --omni /v1/videos  │
│ 无 HTTP 队列  │ 多请求调度 + warmup     │ 同左 + Omni 管道              │
├───────────────┼─────────────────────────┼───────────────────────────────┤
│ 静态 C 管道   │ PyTorch eager（H3 speed │ Diffusers/Omni + enforce-eager│
│ 无 Dynamo 主路径│ 故意关 compile：改数值）│ consumer 显式 eager           │
├───────────────┼─────────────────────────┼───────────────────────────────┤
│ INT8 ConvRot  │ BF16 默认；Online FP8 / │ BF16；可选 fp8 quant；        │
│ 蒸馏 8-step   │ kitchen_int8 / GGUF /   │ Turbo LoRA 另档               │
│               │ pruned INT8 ConvRot 可选│                               │
├───────────────┼─────────────────────────┼───────────────────────────────┤
│ SDPA registry │ fa / sage_attn /        │ CUDNN / FLASH / TRTLLM_ATTN / │
│ cudnn|sage3|  │ sol_attn（opt-in）      │ SAGE（损质 opt）               │
│ sol_cute      │                         │                               │
├───────────────┼─────────────────────────┼───────────────────────────────┤
│ MagCache lite │ Cache-DiT quality=high  │ cache_dit ⊕ tea_cache（互斥） │
│ / calibrated  │ （审计 4×H200）         │ Tea 仅 FL2VA 标定             │
│ Tea/Easy/FB ❌│ 手动 SGLANG_CACHE_DIT_* │                               │
├───────────────┼─────────────────────────┼───────────────────────────────┤
│ 单卡常驻      │ Ulysses/Ring/TP/FSDP /  │ SP/Ulysses/VAE PP/TE TP / DLO │
│               │ layerwise offload       │                               │
└───────────────┴─────────────────────────┴───────────────────────────────┘
```

### 1.1 计算图与运行时

| 机制轴 | h3.c | SGLang | vLLM-Omni | 对本仓适用性 | 风险 |
| --- | --- | --- | --- | --- | --- |
| **图形态** | 手写前向：TE → DiT 环 → VAE tile；形状在 CLI 钉死 | Eager DiT；`performance-mode speed` **故意不**隐式 `torch.compile`（cookbook：compile **改数值**，不作 consistency GT） | Consumer 配方 `--enforce-eager`；数据中心可 regional compile | 本仓已「后融合」；抄 compile **无意义且架构禁 Python** | 框架 compile 与「lossless」冲突——他们自己也承认 |
| **Host/device 同步** | 权窗 freeze、TE packed H2D 一次、DiT device-resident act | Layerwise / FSDP 引入 per-block H2D/all-gather | DLO 双缓冲流式尾层；单卡 CPU offload | 本仓金线已消大部分同步；再抠 I/O 在噪声里（q1 vs q1_spd） | 框架 offload 把「延迟」卖成「可跑」——别当加速 |
| **常驻权重** | INT8 整网 + window window；24GB 设计点 | 数据中心 resident；消费卡 20 层 resident + 流式 | 2×5090：20 层 resident；单卡离场式 offload | 本仓赢在「真常驻」 | 框架 2×5090 写明 ~384 GiB **主机 RAM**——不是 GPU 故事 |
| **Python/GIL/预处理税** | Tier-0 **零**；Tier-1 `dlopen` + embed：sage3 曾 **~40ms/层 Python permute**（`verify_5090_bench`），后改为 **CUDA THD↔BHTD**（`nvidia_build.md`） | 整栈 Python；Triton prepare（Sol）同类税在框架内摊销 | 同左 | 插件路径必须把 prep 留在 device；否则「核快、层慢」 | 再引入 Python permute / 每层同步会抵消 DiT −15% |
| **Serving scheduler** | 无 | 多请求、warmup 分辨率、BCG 签名桶 | 请求级 `quality` 挂/卸 cache；co-batch packed varlen（部分后端） | 产品若要 API：**调用框架当网关**，勿把调度写进金线引擎 | 混 KPI：serving p50 ≠ cafe 金线 |

**机制判决：** 框架在「多租户 + 多拓扑 + 请求档」赢；h3.c 在「单请求、无调度、算子已融合」赢。把框架的 offload/compile 数字拿来压本仓 75s，是**运行时类别错误**。

---

## 2. 注意力后端（机制 → 适用 → 风险）

### 2.1 各后端在算什么

| 后端 | 近似点 | 典型对照基线 | 本仓状态 |
| --- | --- | --- | --- |
| **cuDNN fused SDPA** | dense Softmax Attention（数值上接近「精确」稠密） | 自身 | **Tier-0 默认**；cafe `T≈15973,H=56,D=128` ~47ms/层 |
| **Sage3 FP4** | Q/K/V 微缩放 FP4 + 量化注意力 | FA2 / 卡上最快 Flash（5090） | Tier-1；DiT **−14.1%**；MAE **~26.6** FAIL |
| **Sol CuTe / Sol-Attn** | 在线块稀疏：τᵢ=μᵢ+βσᵢ；拒块用 Taylor 代理分数近似 | FA3 / Sage / eager | Tier-1 `sol_cute`；DiT **−15.3%**；MAE **35.2** FAIL；nested-loop **禁** |
| **TRTLLM_ATTN** | 数据中心稠密 BF16；可选 SAGE QK 量化 + Skip-Softmax 稀疏（损质） | FA4（vLLM：差 &lt;2%） | 本仓无 |
| **SGLang fa / sage_attn** | fa=exact；sage=量化注意力；H3 packed varlen 有专用路径 | FA3 上 Sage 仅 ~5–7%（社区 PR） | 说明：**赢 FA ≠ 赢本仓 cuDNN** |

### 2.2 数值误差来源（为何本仓 MAE 更严）

1. **量化误差（Sage / TRTLLM SAGE）：** 注意力 logits/累加进入低精度；官方自称 plug-and-play，但 H3 packed `[text|audio|video]` 对音频前缀敏感。  
2. **稀疏/近似 Softmax（Sol）：** 未选中块不做 exact；代理校正是**有偏**近似。官方 Sol-Engine 用 dense 前缀步/层缓解；8-step 若抄「前 10 步 dense」则 Sol **几乎不跑**。  
3. **轨迹叠加：** 8 步每步都是 CFG 蒸馏轨迹上的大更新；单步注意力误差在少步上**不可被后续步抹平**（50 步反而有更多「平均」空间——这是机制推断，非本仓实测）。  
4. **门口径：** 框架 `quality=high` 用 **SSIM 0.93 / PSNR 28** 相对 lossless；本仓用 **PPM RGB MAE + 口型/道具目视**。MAE~27–35 在本仓是明确失败带；SSIM 0.93 在框架是「可接受加速档」。

### 2.3 Sol-Engine 融合边界 vs「纯注意力替换」

Sol-Engine H3-OnDevice **明文三柱**：

1. **Lossless kernel fusion**（RMS-AdaLN、QKNorm+RoPE、QKV merge、GEMM+GELU、compile）— 相对 **eager BF16**；  
2. **Cross-step cache**（Spark/5090 页写 **First Block Cache**；4090 fullopt 用 **TeaCache**）— **整栈最大单步**；  
3. **Sol-Attn** — 在 cache 之后再砍一截（4090 受控表：cache 3.18× 后再 **1.22×**）。

**本仓含义：** 柱 1 对本仓大多是 **DROP-HERE**（已融合 INT8+cuDNN；AdaLN 8-step 上限 ~1.5s）。柱 3 单独接 generate = **纯注意力替换** → 实测核赢、片不过、e2e≈0。柱 2 才是框架/官方的主力——本仓未齐。

### 2.4 风险（注意力）

- 默认污染：auto→cudnn 已焊死；缺 `LIB/ROOT` 硬失败（正确）。  
- 误差叠加：Sol/Sage + MagCache + INT8 **禁止**在未测情况下叠默认。  
- 假核：nested-loop Sol、参考 sage2 已证伪相对 cuDNN。

---

## 3. 步间缓存族（判据函数差异）

### 3.1 判据对照（论文/官方 README）

| 方法 | 探测信号 | 跳过粒度 | 校准 | CFG/有声 | 公开 H3 用法 |
| --- | --- | --- | --- | --- | --- |
| **TeaCache** | timestep embedding / modulated input 差异 → **多项式拟合**估计输出残差变化；累积 rel-L1 超阈则算 | **整步**（复用上一 DiT 输出/残差） | 常需多 prompt 拟合系数 | 视频轨为主；音频需另策 | Sol-Engine 4090 主力；vLLM `rel_l1_thresh=0.17` **仅 FL2VA** |
| **MagCache** | 更新残差 \(\mathbf{r}_t=\mathbf{v}_\theta-\mathbf{x}_t\) 的 **magnitude ratio** \(\gamma_t\)；跳步误差 \(1-\prod\gamma\) 累积 | **整步**；有 max consecutive \(K\) | 单样本校准曲线 | 论文未专治 AV | 本仓 lite：**velocity mean-abs ratio**；p4 FAIL |
| **FBCache** | **第一 transformer block 的 residual** 与上步之差；小则复用**整网最终 residual**，跳过后续块 | **步内块级**（跑首块探测 + 可能跳过其余块） | 阈值经验表（Flux 0.12 等） | 与 Tea 同家族 | Sol-Engine Spark/5090 页用 FBC；本仓 **未实现** |
| **Cache-DiT (DBCache)** | **块级 residual L1/diff threshold**（`residual_diff_threshold`）；Fn/Bn 强制算前/后若干块；warmup；max continuous cached | **块级**（非必须整步跳）+ 可选 TaylorSeer / SCM | 服务端 profile；SGLang `quality=high` 锁审计拓扑 | cookbook：**改 AV 联合轨迹**，指标却常只报视频 SSIM | SGLang 1.40×@4×H200；vLLM 1.35×；与 Tea **互斥** |
| **EasyCache** | **公开材料不足**（无与 Tea/Mag 同级的 H3 官方判据页可 WebFetch 到稳定定义） | 社区多为整步/残差类 | — | Comfy **#15326** 伤 H3 **音频** | 本仓 **未实现**；有声高风险 |
| **reuse（本仓）** | **无探测**；固定间隔 schedule | 整步外推 velocity | 无 | AV 同步外推 | 已实现，默认关；与 mag 互斥 |

**本仓 MagCache 代码真相（`h3_cuda_generate.c`）：**

- 信号：相邻 **EVAL** 步的 `velocity`（及音频 velocity）**mean abs ratio − 1**；calibrated 时再套 `poly≈0.1x³+0.2x²+0.7x`。  
- 决策：首末步强制 EVAL；中间在 `accum < thresh` 且 `consec < max` 时 SKIP → `extrapolate_velocity`。  
- **不是**论文完整 MagCache（无 \(\prod\gamma\) 形式误差界、无单样本 residual 校准曲线入库）；calibrated ≈ Tea **风格**，非独立 TeaCache 模块。

### 3.2 8-step vs 50-step：可跳空间（Amdahl）

设稳态 DiT 步耗 \(c\)（4090 cafe ≈5.3s），总步 \(N\)，强制算首末与 warmup \(W\) 步，最多可跳 \(N-W\)。

| 配方 | \(N\) | 典型官方 reuse 叙事 | 跳 \(k\) 步的 DiT 上限 | 相对 e2e |
| --- | ---: | --- | --- | --- |
| 本仓金线 | **8** | Mag 曾 skips=3 但 **MAE 炸**；安全预期 **0–1** | \(k\cdot 5.3\)s | 跳 1 → ~70s；跳 2 → ~65s（**未证质量**） |
| 官方 50-step | **50** | Tea：14 full + 35 reuse（4090 fullopt） | 量级 **数十秒到数分钟** | 才支撑 3× 级 cache 臂 |

**机制结论：** 缓存论文/框架的 × 来自「长轨迹中段冗余」。**CFG 蒸馏 8-step 把冗余提前榨干**；再 skip 是在吃「结构/细节」预算，不是吃「无聊细化」预算。LightX2V #579（蒸馏后几乎不跳）与本仓 Mag FAIL **同方向**。

### 3.3 Cache-DiT 块级 vs 整步：对本仓的含义

- **整步 skip：** 实现简单（本仓 Mag/reuse 插点已在）；误差直接进 Euler。  
- **块级 skip（DBCache/FBC）：** 可能在「不能不算这一步」时仍省中间块 FLOP；但 H3 **50 层** + packed AV 注意力，首块 residual 是否仍是好代理 = **待证**。  
- SGLang issue 叙事（浅层文已引）：DiT block **in-place** 时 Pattern_3 residual 可能恒 0 → `can_cache` 永不触发——**实现脆弱性**是真实工程风险，不是营销。

### 3.4 CFG / 有声耦合

- 本仓金线 **cfg=1**（已蒸馏）→ FasterCache 类「CFG 双分支冗余」**无空间**。  
- 有声：一条自注意力；视频 skip 套音频 = EasyCache 类事故面。本仓 Mag 用 **audio |ratio−1| 抬高 delta** 做弱 veto；完整 AV 分账仍不足。  
- vLLM：**Tea 仅 FL2VA 标定**；Ref2VA 不 cache——技术含义：条件模态改变残差统计，**阈值不可跨分区搬运**。

---

## 4. 量化栈与误差预算叠加

| 层 | h3.c 金线 | 框架常见 | 叠加规则（工程） |
| --- | --- | --- | --- |
| 权 | **INT8 ConvRot**（蒸馏配方） | BF16；Online FP8；kitchen_int8；社区 pruned INT8/NVFP4 | 权误差已占用预算 |
| 激活/注意力 | dense cuDNN（近精确） | fa 精确；sage/FP8/SAGE 再近似 | 注意力近似 **叠在 INT8 上**更窄 |
| 步/块 | 全算 8 步 | Cache-DiT / Tea | skip 误差 **乘性**进入积分器 |
| VAE | fp16 tile+deblock | BF16 resident / tile PP | 相对次要 |

**误差预算直觉（推断，标明）：**  
金线把「可花的近似预算」几乎全花在 **INT8 权**上，用 **dense SDPA + 全步** 换回视觉。框架 `lossless` 把预算留给 **BF16+dense**，再在 `high` 档显式花掉 cache 预算并报告 SSIM。  
因此：**把框架 high 的 Cache-DiT 直接塞进本仓 INT8 8-step，是双倍花预算**——不是「他们能开我们也能开」。

NVFP4：本仓 TE 路径有 device dequant；DiT 金线不是 NVFP4 TC。SGLang Online FP8：**近似**，可与 Cache-DiT 叠，cookbook 要求自测视听。

---

## 5. 并行：Ulysses / TP / SP vs 单卡

| | h3.c | SGLang | vLLM-Omni |
| --- | --- | --- | --- |
| 切分点 | 无产品 SP | **Ulysses=头维 all-to-all**；Ring=跨节点序列；TP=层内 | USP/Ulysses + VAE patch PP + TE TP |
| 通信 vs 算力 | N/A | H200 上纯 Ulysses4 快于 TP2+Ulysses2（cookbook 实测叙事） | 消费卡 DLO 用带宽换容量 |
| 4090/5090 现实 | **单卡 INT8 金线成立** | 1×4090：kitchen_int8 + 全层 stream；2×5090：TP2+layerwise，**50-step 768p ~560s** 级 | 2×5090 cuDNN+DLO 验证；单卡 offload 慢 |

**机制判决：** 多卡赢的是 **大分辨率长步** 的延迟；不自动转化为「480p 8-step 单卡再砍一半」。本仓多卡架构文是路线图，**不是**金线 KPI。

---

## 6. 条件多模态路径

| 路径 | h3.c | 框架 | 技术含义 |
| --- | --- | --- | --- |
| T2VA | **金线**；无 vision encode 大头 | FL2VA 分区 `task=t2va` | 注意力/cache 实验应优先 T2VA cafe |
| F2VA/L2VA | `--first/last-frame`；vision **cuBLAS GEMM + 仍可能 host SDPA**（历史 layers~31.5s） | FL2VA 合同完整 | **KEEP：vision SDPA→cuDNN**；与 DiT 后端正交 |
| Ref2VA | 有条件行路径；组合深度弱于 cookbook | 独立 variant / combined 一服务双 DiT | Tea **不标定** Ref2VA → cache 统计换区即失效 |
| 有声 AV | 默认 pack；cache 需 veto | 同步 AAC；SSIM 常**不含音频** | 框架 high 的 SSIM **不能**代替本仓有声门 |

**vLLM「Tea 仅 FL2VA」：** 不是产品吝啬，是 **残差/embedding 统计与条件分布绑定**——换 Ref2VA 或未标定步数/分辨率，阈值外推无理论担保。本仓若做 Tea，必须在 **cafe 8-step INT8 有声**上重标定，禁止抄 0.17/0.10。

---

## 7. 可维护性与产品边界

| 维 | h3.c | 框架 |
| --- | --- | --- |
| 扩展 | `h3_sdpa` registry + `dlopen` 插件；缺依赖 **硬失败** | 框架内置后端 + pip 依赖；版本漂移（Sage SM90 pin、Cache-DiT in-place bug） |
| 质量门 | PPM MAE + 目视 + 音频；**不可辨才候选默认** | lossless bit/CI；high = 审计 SSIM |
| 默认污染 | auto→cudnn；mag/reuse/sage/sol **关** | `quality` 默认 lossless；high fail-closed 到审计拓扑（SGLang） |
| 产品边界 | 边缘引擎；非 HTTP | Serving 合同；HF 推荐部署栈 |

**风险：** 用框架 SSIM 过门标准「放宽」本仓 MAE，会在机制上把产品从「金线引擎」改成「可接受损质 serving」——必须产品改口，不能工程静默。

---

## 8. 对本仓 8-step INT8 金线：各技术杠杆的理论收益天花板

> 基线：4090 cafe ~75.4s（dit 47.5）；5090 warm ~66s（dit 30.7）。下表 **DiT/e2e 为算术或已测**；质量列为已测或「未证」。

| 杠杆 | DiT 上限 | e2e 上限 | 质量（本仓尺子） | 判决 |
| --- | ---: | ---: | --- | --- |
| 再抠 TE/VAE/mmap | ~0 | ~0（噪声） | 无损 | **DROP-HERE** |
| AdaLN/gate 再融合 | ≲1.5s | ≲1.5s | 应 bit-exact | **DROP-HERE** |
| `sol_cute` / `sage3` | **−14–15% 已测** | **≈0 已测**（TE 吃掉） | **MAE FAIL 已测** | Tier-1 opt-in；不进默认 |
| MagCache 0.08 | 跳 3 步量级 | 曾 ~79s（未加速） | **MAE 52–68 FAIL** | 默认永关 |
| 独立 Tea / 保守 FBC | 跳 1–2 → **−5.3～−10.6s** | 同左减 TE 噪声 | **未证**；预期悲观 | **OPT-IN 实验** |
| Cache-DiT 块级思想 | 不确定（可能 &lt; 整步） | 不确定 | 未实现；须自建 MAE 门 | 研究档 |
| EasyCache | 同整步族 | — | 音频高风险 | **最后 / 有声慎** |
| Online FP8 / 换 BF16 | 可能更慢或主题变 | — | 离开金线 | 非金线 |
| Ulysses 多卡 | N/A（换产品） | 延迟账本重写 | 另 KPI | 非金线 |
| Vision SDPA→cuDNN | T2VA **0** | F2VA/L2VA **−?（E3 profile）** | 应近 exact | **KEEP（条件图）** |
| SLA / Turbo 4-step | 换权换步 | 营销 × 不可平移 | 另一产品 | **禁接金线** |

---

## 9. 优劣矩阵（技术维）

| 技术维 | h3.c 相对框架 | 框架相对 h3.c |
| --- | --- | --- |
| 单卡稠密算子效率（相对自身基线） | **已后融合**；再换 attn 难赢 cuDNN+过门 | 相对 eager/FA 仍有 attn/quant 空间 |
| 步间冗余榨取 | 8-step **空间极小** | 50-step **主战场** |
| 误差可控性 | **单点 INT8**；默认无叠近似 | 多档显式近似 + SSIM 合同 |
| 多卡延迟 | 弱 | **强** |
| 消费卡可跑性（真常驻） | **强（INT8）** | 弱（offload+主机 RAM） |
| 条件模态合同 | 中（F/L 有，Ref 浅） | **强** |
| 可维护/依赖 | 薄；插件硬失败 | 厚；版本与钩子脆弱 |
| 质量门与产品承诺对齐 | MAE 严 ↔ 「不可辨」 | SSIM 松 ↔ 「high 可服务」 |

---

## 10. 推荐技术路线（按 ROI）

### 该抄机制（思想级，不是抄 Python）

1. **独立 TeaCache / 保守 FBCache 探测语义**（阈值、首末强制、max consec、**AV veto**）——对齐官方 4.44× **主力臂**与 vLLM Tea 合同；在 cafe 8-step **从严**扫；不过门就关。  
2. **Cache-DiT 的块级 residual 思想**（Fn/Bn + warmup + continuous cap）——作研究档；自建 **MAE+音频** 门，**禁止**用 SSIM 0.93 过门。  
3. **请求/CLI 档位分离**（lossless vs high 的产品哲学）——本仓已有 Tier-0/1；保持「缺插件硬失败、默认不污染」。  
4. **条件图 vision → cuDNN**——框架不教这个，但是本仓唯一 KEEP 大头（非 T2VA）。

### 该自研（不能靠搬框架）

1. **C 侧 cache 决策与 velocity 外推**（已有 Mag/reuse 插点）——框架钩子是 PyTorch module forward，ABI 不通。  
2. **有声分账 veto**——框架 SSIM 不管音频。  
3. **INT8 金线误差预算下的阈值重标定**——0.17/0.10/RDT=0.12 **不可照抄**。  
4. **注意力质量**（dense 前缀、sink、布局）——过门前不改默认；prep 必须保持 CUDA。

### 该永远不碰（对这条金线）

1. 用框架 **50-step / 4×H200 / SSIM** 数字改 75s 承诺。  
2. **SLA / Turbo-SLA / 未蒸馏 4-step** 插 INT8 金线（见 sla 文）。  
3. **nested-loop Sol / 假 sage2** 当产品。  
4. **EasyCache 无 AV 分账**进有声默认。  
5. **Python/Comfy 生成栈**替换 C 管道。  
6. MagCache 已 FAIL 档 **回潮默认**。

---

## 11. 附录：本仓代码锚点

| 主题 | 位置 |
| --- | --- |
| SDPA Tier | `h3_sdpa.h` / `h3_sdpa_registry.c` |
| Mag/reuse 环 | `tools/h3_cuda_generate.c`（`magcache_thresh` / `extrapolate_velocity`） |
| sage3 embed | `tools/h3_sage3_embed.{c,py}`；prep 曾 ~40ms/层，现 CUDA |
| sol_cute | `tools/h3_sol_cute_embed.*` + `h3_gpu_cuda.cu` |
| 金线分解 | `out/verify_480p_quality_full/t2va_complex_5s_cafe_q1/` |
| sol 视觉 FAIL | `out/verify_sol_vis_gate/REPORT.md` |
| sage3 FAIL | `out/verify_5090_bench/REPORT.md` |

---

## 12. 公开材料不足 / 推断清单

| 项 | 状态 |
| --- | --- |
| EasyCache 对 H3 的权威判据函数 | **公开材料不足**；仅有社区/Comfy 音频回归信号 |
| vLLM TeaCache 与本仓同形 e2e 秒数 | **无** |
| SGLang Cache-DiT 在 8-step INT8 单卡上的行为 | **无公开**；推断：可跳空间≪50-step，且可能触发 in-place residual 坑 |
| 8-step 上注意力误差比 50-step「更难抹平」 | **机制推断**（少步大更新）；非 A/B 实测 |
| 块级 cache 在 H3 50L packed AV 上的首块代理质量 | **未证** |

---

**一句话收束：** 框架赢在 **长轨迹块级 cache + 多卡 serving 合同**；h3.c 赢在 **短轨迹后融合 INT8 单卡金线**。对本仓，下一单位 ROI 不是再换注意力核，而是 **可测的步/块 cache（默认关）+ 条件图 vision SDPA**；任何进默认的近似必须过 **MAE∩有声**，不是过 SSIM。
