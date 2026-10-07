# FreeToken 技术备忘（供完整极致方案引用）

Date: 2026-09-13  
一手论文：[arXiv:2608.16157](https://arxiv.org/abs/2608.16157) *FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution*（2026-08-17，Yang / Fan / Keutzer / Han / Zaharia / Xu / Stoica 等）  
代码：[FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken) `main` `@953565667f3141c90d0f0eb469bb2655d2407140`（`python/freetoken/version.py` = **0.1.2**，Apache-2.0）  
站点：[flashml.ai](https://flashml.ai) · 安装/模型/CLI：[docs/install.md](https://github.com/FlashML-org/FreeToken/blob/main/docs/install.md) · [docs/models.md](https://github.com/FlashML-org/FreeToken/blob/main/docs/models.md) · [docs/cli.md](https://github.com/FlashML-org/FreeToken/blob/main/docs/cli.md)  
本地只读浅克隆：`third_party/accel_survey/FreeToken/`（**非 submodule**，已进 `.gitignore`，不 commit）  
对照：[完整极致施工图](./2026-09-13-h3c-5090-extreme-full-plan.md) · [SelfLift 备忘](./2026-09-13-h3c-selflift-comfy-analysis.md) · [DeepSeek 课](./2026-09-13-h3c-deepseek-opt-lessons.md)

> **For agentic workers：** 独立机制备忘，不是施工单。禁止改推理代码、禁止嵌 Python、禁止 commit、禁止把 `ft serve` 当 generate 后端。施工图 **不新开 Phase**；唯一可借的刀已经是 Phase 1C。

**一句话：** FreeToken 是消费卡上的 **MoE 语言模型 serving**（专家池放 host、VRAM 只缓存热专家、miss 按本机 PCIe/CPU 带宽拆给 GPU 或 CPU）。它解决的是「284B/753B 装不进 8–32 GB」，不是视频 DiT 的 token / 步数 / A2A。本仓 MiniMax-H3 是稠密视频 DiT，**没有可路由的专家**；公开数字是 tok/s，没有 5s · 1344×768。能借的只有「NVFP4 保持 packed、GPU 上算」——已经写在 Phase 1C。

---

## 0. 同名项目：哪个才是「消费级显卡」那个

联网检索（GitHub / arXiv 2026 / 消费卡 / DiT / video / MiniMax / H3 / 5090）后，叫 FreeToken 的有好几家。**针对消费级显卡、2026 新出、用户语境能对上的只有 FlashML 这一家。**

| 项目 | 是什么 | 消费卡？ | 视频 DiT？ | 判词 |
| --- | --- | --- | --- | --- |
| **[FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken)** + [arXiv:2608.16157](https://arxiv.org/abs/2608.16157) | 边缘 MoE LLM serving；`ft serve`；RTX 30/40/50 | **是。** 论文 Table 1：4060 Laptop 8 GB / 3090 / 4090 / **5090** / 5090 desktop / PRO 6000 | **否。** `docs/models.md` 全是 LLM；「Multimodal checkpoints are served **text-only**」 | **就是这个。** |
| [FreeTokenAI/FreeTokenSwift](https://github.com/FreeTokenAI/FreeTokenSwift) | iOS/macOS 混合云+端 SDK，要 `console.freetoken.ai` 的 app token | 手机/Mac，不是 5090 施工 | 否 | 名字撞车，SDK/计费 |
| [freetokennet/$FREET](https://github.com/freetokennet/freetoken_net) | ERC-20「免费发币」 | 否 | 否 | 加密货币 |
| 各类「free API tokens」博文 | 云 API 额度 | 否 | 否 | 计费，不是系统 |

没有找到名为 FreeToken 的视频 / DiT / token-merge / MiniMax-H3 加速仓。相近但**不是** FreeToken 的消费卡视频方案：TurboDiffusion、FastWan-QAD、LightX2V（已在别的 survey 里）。`docs/models.md` 的 **MiniMax-M2.5** 是 LLM（`nvidia/MiniMax-M2.5-NVFP4`），**不是** MiniMax-H3 视频 DiT。

---

## 1. FreeToken 到底做什么 / 堵哪堵墙

**不是：** 减 DiT token、ToMe、TeaCache、量化蒸馏、多卡通信编译、视频步数压缩。

**是：** 把一台「个人机」当成 GPU + CPU + host DRAM + PCIe 的统一 serving 平台，专门跑 **专家池远大于 VRAM** 的 MoE LLM。

论文 §1–§3 和 README 钉死的三块：

| 机制 | 做什么 | 堵墙 |
| --- | --- | --- |
| **全局 LRU 专家缓存** | 非专家权常驻 GPU；全部 routed expert 放 host（source of truth）；VRAM 剩的做成 `(layer, expert)` slot 缓存。decode 跟路由走，不在 load 时静态钉专家 | **显存。** 5090 32 GB 装得下 DSV4-Flash 的 **活跃** 13B，装不下 **全池** 284B |
| **带宽自适应 `q*` / hybrid** | miss 的 *m* 个专家拆成：*q* 个走 PCIe 填缓存再 GPU 算，其余 CPU 原地算，再精确 merge。`q* ≈ m · B_PCIe / B_Host`，本机 `ft bench bw` 测两个带宽 | **PCIe + 无 NVLink。** 消费卡专家 miss 只能走 PCIe 或干脆在 CPU 算；论文测 5090 desktop PCIe **49.0 GB/s**、host 专家核 **53.8 GB/s** |
| **Prefill 整层双缓冲** | GPU 算 layer *l* 时，transfer stream 预取 *l+1* 整层专家（prefill 几乎激活全池） | **带宽。** 论文：DSV4 FP4 一次 prefill 约 140 GB 专家；5090 PCIe5 x16 ~2 s，4090/3090 ~5 s |
| **语义锚点状态缓存** | 在 `</tool_call>` / thinking / turn 边界存 recurrent/KV checkpoint，agent 改上下文只重算 suffix | **LLM agent 重 prefill。** 视频 generate 没有 tool-call 编辑 |
| **弹性 VRAM** | 运行中在 expert cache 与 KV 之间改预算，不重启、不重载 host 池 | 桌面 GPU 和浏览器/游戏抢显存 |
| **FTW + 量化核** | HF → 运行时 bank 布局；GPU 上 NVFP4/MXFP4/FP8 **inline dequant / Marlin / b12x**，CPU 专家核 SIMD + in-kernel dequant | 启动 I/O + **消费卡没有数据中心 FP8/FP4 tensor-core 档位时仍要能算 4-bit** |

消费卡墙对照本仓：

| 墙 | FreeToken 打的 | 本仓 8×5090 H3 |
| --- | --- | --- |
| 显存装不下全模型 | **主问题。** 8 GB 跑 35B，32 GB 跑 284B | INT8 DiT **已经常驻**；墙不是「装不下」 |
| PCIe / 无 NVLink | 专家权 H2D；不解决 GPU–GPU P2P | Ulysses A2A 是 **激活** host bounce，`peer_access=0` |
| 无 FP8 tensor-core 档 | 用 NVFP4/MXFP4 + Triton/Marlin/CPU SIMD，不要求 Hopper | 金线 INT8 ConvRot；NVFP4 社区权 **CPU dequant ~424 s** |
| 编译 / CUDA Graph | 动态 LRU 做成 graph 内 data（固定 shape + device 计数） | 与 DiT 8-step 无关 |

源：论文 §3.1–3.3、§4；`python/freetoken/moe/offload_cache.py`（`decode_target` = gpu/cpu/hybrid，`hybrid_fetch_fraction`）；`python/freetoken/moe/benchbw.py`；`python/freetoken/layers/quantization/{linear,moe}/nvfp4.py`。

---

## 2. 依赖

| 项 | 事实 |
| --- | --- |
| 模型族 | MoE LLM：DeepSeek-V4-Flash、Qwen3.6-35B-A3B、GLM-5.2/4.7、gpt-oss、Gemma-4、**MiniMax-M2.5**、Muse-Glimmer。dense Qwen 走 `fused`（权全在 GPU）。**无 Wan / Hunyuan / H3 / 任何视频 DiT** |
| 框架 | Python 3.10+，PyTorch 2.11，Triton 3.6，可选 FlashInfer / sglang-kernel。灵感来自 mini-sglang / SGLang / vLLM / FlashInfer / llama.cpp。**不是 C 生产栈，不是 Comfy** |
| 精度 | BF16 / FP8 / MXFP4 / NVFP4（ModelOpt：packed e2m1 + e4m3 per-16 + per-tensor global）。CPU 专家核只吃 bf16 / nvfp4 / mxfp4；block-fp8 **不能** CPU，只能 offload |
| 训练 | **不训练。** 论文强调 routed 计算 **exact**，不改模型、不 skip 专家 |
| 硬件/OS | Linux x86_64，Ampere+，driver r580+，CUDA 13。FAQ：[issue #84](https://github.com/FlashML-org/FreeToken/issues/84)。单卡为主（`--gpu` 选一张） |
| 只服务 LLM？ | **是。** OpenAI/Anthropic HTTP，给 Codex / Claude Code / OpenCode 用。多模态 checkpoint **只跑文本** |

本仓对不上：H3 视频 DiT 每步每层 **全部** linear 都要算，没有 top-k 专家。把 expert LRU / `q*` 抄进来没有路由信号。

---

## 3. 公开数字与公平性

论文 Table 1 + §5（作者自己的数，不是第三方实验室）：

| 机器 | GPU | 测到的 PCIe *B_P* | host *B_H* | 代表结果 |
| --- | --- | ---: | ---: | --- |
| 5090 server | RTX **5090** 32 GB，PCIe 5.0×16 | 52.7 GB/s | 77.3 | Qwen3.6 BF16 **77–83 tok/s**；DSV4 MXFP4 **22–25 tok/s**（1.5–2.3× 最强基线） |
| 5090 desktop | RTX **5090** 32 GB，9950X3D / DDR5 192 GiB | 49.0 | 53.8 | 同上量级；相对 server 只掉 4%（llama.cpp 只剩 80%） |
| 4090 | RTX **4090** 24 GB | 25.1 | 63.2 | 跨硬件 W2：领先最强基线 **1.3×**（与 3090 同档） |
| 3090 | RTX 3090 24 GB | 25.3 | 56.7 | 1.3× |
| 4060 laptop | 8 GB，PCIe 4.0×8 | 11.8 | 47.5 | Qwen3.6 **NVFP4 39.3 tok/s**（论文写 ≈4090 的 92%） |
| PRO 6000 | 96 GB | 51.5 | 178 | GLM-5.2 753B NVFP4 **14.9 tok/s** vs llama.cpp 7.3 |

基线：llama.cpp、Ollama、KTransformers、MoE-Infinity。论文写 **权格式 bit-exact**（Qwen BF16 对齐；DSV4 原生 MXFP4 对齐）。3090/4090/5090 **租的双路服务器** CPU 远超桌面，测量 **卡到 6 线程 + NUMA pin** 来仿消费主机。

**没有：** 分辨率、时长、fps、VAE、DiT step。单位全是 **tok/s / TTFT**。

独立评测（不是论文）：

- [Zenn, 2026-08-22](https://zenn.dev/holy_fox/articles/53b82eed45f956?locale=en)：自家 5090 32 GB + 128 GB RAM。Qwen 35B NVFP4 **装得进 VRAM**，FreeToken 不是甜点；**gpt-oss-120b**（专家 ~58 GB）才体现价值，单流 **127.1 tok/s**。结论：装得下就不如 llama.cpp/vLLM；价值是抬模型上限。
- [Wavect 评述](https://wavect.io/blog/freetoken-ai-inference-engine-review/)：复述论文数字，标明 **不是他们复测**。
- [issue #151](https://github.com/FlashML-org/FreeToken/issues/151)：2×3090 上 `ft bench bw` 选 hybrid → **0.67 tok/s**，强制 offload → **5.58 tok/s**（8.3×）。自动策略会选错。

公平性：同权对比做得认真；数字仍是作者 + 个别用户。**不能**把 77 tok/s 换算成 768P 秒数。

---

## 4. 和本仓已有刀的重叠

| 本仓刀 | 重叠？ | 说明 |
| --- | --- | --- |
| **ToMe / token merge** | **无。** | FreeToken 的「token」是 LLM decode token，不减 DiT 序列 |
| **TeaCache / TE cache** | **无 / 弱类比。** | 他们缓存的是 **专家权 + agent KV/recurrent**，不是步间 DiT 特征，也不是 Qwen-VL TE 输出 |
| **SageAttention** | **无。** | 他们用 FlashInfer / TRT-LLM / Triton attn 后端服务 LLM；不碰视频 SDPA |
| **Ulysses** | **无。** | 单卡 serving。不解决 `peer_access=0` 的 A2A。2×3090 工单还暴露 hybrid 选错 |
| **NVFP4** | **有，合同不同。** | 他们：packed e2m1 **留在 GPU/host bank**，forward 走 Triton/Marlin/b12x（W4A16）或 CPU in-kernel dequant。我们：社区 pruned_nvfp4 → `h3_dit_nvfp4_dequant_to_i8` **CPU requant 成 INT8**，480p 墙钟 ~479 s / dequant ~424 s（`out/verify_nvfp4`） |
| **host_stage mmap** | **弱类比。** | 都是「host 是 source of truth + 少拷」。我们 INT8 已常驻，host **13.2 → 1.26 s**。他们 mmap/pin 的是 **装不下的专家池** |
| **SelfLift** | **无。** | 分辨率日程 / lift，和 MoE offload 无关 |
| **turbo4 / TAE** | **无。** | 步数与 VAE |

DeepSeek 课（[deepseek-opt-lessons](./2026-09-13-h3c-deepseek-opt-lessons.md)）打的是 **NVLink/IB 上的 MoE 训练/推理**；FreeToken 打的是 **消费卡单机、无 NVLink 的 MoE serving**。两者都是 MoE，都搬不到稠密视频 DiT 的 Ulysses。

---

## 5. 可迁移性（P0 / P1 / P2）

尺子：5s · 1344×768 墙钟分段。改 C 文件。做不到直说。

### P0 — 没有

没有一条「抄了立刻砍 768P 主墙、施工图还没排」的刀。H3 不是 MoE；INT8 已常驻；Ulysses 亏的是激活 A2A 不是专家 H2D。**不要为 FreeToken 新开 Phase。**

### P1 — NVFP4 保持 packed，GPU 上算（已是 Phase 1C）

| | |
| --- | --- |
| 借什么 | FreeToken `nvfp4.py` 的硬规则：kernel 拥有 `{load-time pack, forward}`；**禁止**先 CPU dequant 再另存一份。论文/代码用 ModelOpt e2m1 + e4m3/16 + global scale |
| 改哪个 C 文件 | 已点名：`h3_dit_nvfp4.c` / `.h`、`h3_w4a8_cache.c`（`stage_one_host_nvfp4` / `load_one_linear_nvfp4`）、`h3_w4a8_dequant.c`。sidecar 或 GPU dequant 驻留。**不要** `import freetoken`，不要嵌 Triton/vLLM Marlin |
| 打 768P 哪一段 | **host / dequant**，不是 DiT GEMM。480p 已测 dequant ~424 s、DiT 28.27 s。768P 去掉 424 s 之后，e2e 才有资格和金线比 |
| 风险 | 社区 pruned_nvfp4 布局未必等于他们的 ModelOpt bank；sidecar 脏 key；禁止静默回退金线 INT8 ConvRot。他们的核是 **W4A16**，不是我们的 INT8 ConvRot |
| 做不到 | 把 Marlin/b12x/FlashInfer 当生产路径；赌 sm_120 原生 FP4 当第一刀（施工图已否） |

### P2 — 「先测带宽再选路径」的思想（不要抄 `q*` 公式）

| | |
| --- | --- |
| 借什么 | `ft bench bw`：用 **真实 kernel** 测 PCIe gather vs CPU 核，而不是规格书。issue #151 证明自动选 hybrid 会惨败 |
| 改哪个 C 文件 | **不必为 MoE 写新执行器。** 若要，只在 NVFP4 工具里加「packed H2D+GPU dequant vs CPU requant」微基准。Ulysses 继续用已有 `tools/h3_p2p_bw_probe.cu`（测的是卡间，不是专家 gather） |
| 打 768P 哪一段 | 仍是 host/dequant；**打不动** DiT 105–150 s 回代，也打不动 U=2 A2A |
| 风险 | 测错对象（专家 slot copy ≠ DiT 线性 ≠ Ulysses packed S）。CPU 专家路径对稠密 DiT 是负优化（论文自己写消费 CPU ~50–80 GB/s vs 5090 ~1.8 TB/s） |
| 做不到 | 把 `q* = m · B_P / B_H` 接到 `h3_ulysses_comm.cu`；Ulysses 没有「可丢给 CPU 的专家」 |

### P2 — 下一层权与当前层计算重叠

| | |
| --- | --- |
| 借什么 | Prefill 双缓冲：算 *l* 时 DMA *l+1* |
| 改哪个 C 文件 | `h3_w4a8_cache.c` / `h3_gpu_cuda.cu` 预取下一 block（若还剩 H2D） |
| 打 768P 哪一段 | host。host_stage **已经** 1.26 s，768P 回代里不是主墙 |
| 风险 | 和 INT8 常驻重复劳动；预期亚秒，不值得插队 |

### 明确做不到 / 别碰

- Expert LRU、CPU 专家 GEMV、语义 KV、FTW 当 H3 权格式、`ft serve` 嵌进 generate。
- 把 MiniMax-M2.5 支持写成「他们已经接 H3」。
- 用 77 tok/s 或 39.3 tok/s 宣传 768P。
- 多卡：他们几乎是单卡；不解决 8×5090 `peer_access=0`。

---

## 6. 要不要写进 extreme-full-plan 的 Phase

**不要新开 Phase。** 施工图文末已加 **§9 指针**：唯一可借的 = 已有 **Phase 1C**。Phase 0 / 1A / 1B / 2 / 3 / 4 不动。

---

## 7. 源链接（一手优先）

- 论文：https://arxiv.org/abs/2608.16157 · PDF https://arxiv.org/pdf/2608.16157
- 代码：https://github.com/FlashML-org/FreeToken `@9535656`
- 模型表：https://github.com/FlashML-org/FreeToken/blob/main/docs/models.md
- CLI / `ft bench bw`：https://github.com/FlashML-org/FreeToken/blob/main/docs/cli.md
- FAQ：https://github.com/FlashML-org/FreeToken/issues/84
- hybrid 选错：https://github.com/FlashML-org/FreeToken/issues/151
- 第三方 5090：https://zenn.dev/holy_fox/articles/53b82eed45f956
- 评述（复述论文，非复测）：https://wavect.io/blog/freetoken-ai-inference-engine-review/

未改推理代码。未 commit。密码未入库。未 SSH。未嵌 Python。
