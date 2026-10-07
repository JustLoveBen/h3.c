# H3.c 社区极速方向（5090 · 废金线叙事）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development（推荐）或 executing-plans 按任务施工。步骤用 checkbox（`- [ ]`）跟踪。禁止 commit。密码 / SSH / HF token 不入库。禁止开 8 卡 / Ulysses 去冲 27s。256/64 tile-local RoPE **不改**。`--preset gold` 只是保真对照臂，**不是**发布默认，也**不是**质量上限。

**Goal:** 在用户认可的 Comfy 2-stage 质量地板上，把单卡 RTX 5090 热跑 5s 768P 做到社区已测速度地板（Comfy 双采样 ~27s；能打到 Sol-H3 5090 热 ~30.6s 全含更好），主线是 kitchen VAE 核迁 C + 加载常驻。

**Architecture:** 一个 `h3_cuda_generate` 内核 + `--preset` 配方表。配方为极速服务：默认加速档对齐 TaoMate/Comfy 2-stage；`gold` 可选对照。VAE / attn / 量化 backend 失败可见，禁止静默 TAE、禁止 sage3 假 hit。

**Tech Stack:** 本仓 C/CUDA（`h3_vae_int8_cuda.c`、`h3_gpu_cuda.cu`、`h3_w4a8_cache.c`、`h3_generate_preset.c`）。对照只读：comfy-kitchen、Comfy `comfy/ldm/minimax/vae.py`、用户 JSON `input/minimax-h3-r2v-taomate-2stage.json`、Sol-Attn sm_120。

**Spec:** 本文替换 [方案选型](./2026-09-16-h3c-scheme-selection.md) 里「金线神圣 / Sol-H3 不适合我们」的结论；枢纽 [分模块方案](./2026-09-15-h3c-module-opt-plan.md) 的模块事实仍可用，**口径与优先级以本文为准**。实测账：[verify_taomate/REPORT.md](../../../out/verify_taomate/REPORT.md)。

## 旧金线叙事作废

本仓自建金线没有社区验证，不能当「更好」的挡箭牌，也不能挡社区已经跑通的 27s / kitchen ~4s。金线最多是 `--preset gold` 保真对照臂，不是发布默认，也不是质量上限。下一步只对社区已测配方做极致实现，不再写「金线不动所以不搬」。

---

## 给执行者 / 用户的一段话（25–35 行）

1. 旧分析错了：拿未验证的本仓金线 MAE，压过用户已经认过的 Comfy 2-stage。
2. kitchen ~4s 被说成「合同外」；196 窗 ~18s 被写成「合同下限」——同几何社区已经 4s，是我们没做到。
3. 27s 被 5s/10s、1344/1376、冷/热、是否含 load 口头混用，不能再当同一列数字。
4. 金线没有社区验证，不能当更好，也不能挡优化；最多 `--preset gold` 对照。
5. 新尺子：RTX 5090 单卡热跑，模型已在显存。
6. 5s 768P，画布写死 **1376×768**（用户 Comfy 2-stage JSON 成片）。
7. 1344 只出现在 gold / Sol-H3 对照，禁止再和 27s 画同一列。
8. 质量地板 = 那条 Comfy 片：无 16px 格子、不切块换馆；不是金线 MAE。
9. 速度地板 = 社区已测 Comfy 双采样热 ~27s。
10. 更好：独立复现 Sol-H3 5090 热 ~30.6s 全含（那是 LTX 成片，质量尺另说）。
11. 加载单独记账，目标对齐 Comfy 第二次 Queue / SGLang `vae=resident` 的几乎不耗时。
12. 我们差在 VAE 核：36 层 × 196 窗 C GEMM + 串行 cuDNN ≈18s，不是几何到顶。
13. 我们差在加载：WEIGHT_CACHE 默认关、pin 用完就卸、repeat 丢掉 GPU 包。
14. 我们差在 sage3：768P seq≈37788 是 hits=0 fallback=200 仍继续。
15. sage3 池子把 INT8 挤成 window=1，核快换来权流，Comfy 是快注意力 + 权常驻同时有。
16. TE 冷 ~13s；热靠 te-cache 不是进程常驻。mux ~2s 次要，失败不能冒充 VAE 快。
17. 不要再用「合同下限」把 18s 说成已经极致。
18. 社区还能搬：kitchen 核迁 C，256/64 RoPE 不改。
19. 社区还能搬：VAE/DiT/TE 加载常驻。
20. 社区还能搬：Comfy 2-stage 日程 864×480×3 → LBH → 1376×768×2。
21. 社区还能搬：Sol-Attn sm_120 打 768P 长序列。
22. 不要一上来换 LTX 成片当默认质量尺。
23. 下一步唯一实现主线：**kitchen VAE + 加载常驻**。
24. 金线代码路径留 `--preset gold`。不开 8 卡 / Ulysses。
25. 失败可见：kitchen 没有就硬退，sage3 假 hit 硬退，禁止静默 TAE。
26. SSH 可测 5090，不为冲 27s 乱切块。不 commit。密码不入库。

---

## Global Constraints

- 设备：RTX 5090 单卡。热跑 = 模型已在显存 / 进程已活过至少一次 decode。冷 load 另列，禁止折进 27s。
- 时长：5s · 24fps · 124 帧。禁止拿 10s 27s 或 480p 横比。
- 画布：**极速配方锁 1376×768**。`H3_MAX_PIXELS` 拒 1376 是旧金线桶，加速档 generate 已经在跑 1376，不要退回 1344 装对齐协议。
- 质量地板：用户认可的 Comfy 2-stage 成片（无格子、不切块换馆）。金线 MAE **不是**门。
- 速度地板：Comfy 双采样热 ~27s（用户单卡 5090 验证图）。更好：Sol-H3 两阶段 5090 热 30.6s 全含量级。
- 加载：`phase_ms` 必须单列 `vae_load` / `dit_host_stage` / `te`。目标热路径 load ≈ 0。
- `--preset gold`：8-step cudnn 官方 VAE 1344，对照臂。优化 **不必**先过金线。
- 256/64 tile-local RoPE、Comfy 线性 blend：质量红线（waffle）。减窗 / 整幅低频 RoPE / TAE 成片 **不是**本主线。
- 禁止 8 卡 / Ulysses 冲 KPI。禁止 commit。密码不入库。

---

## 1. 旧分析错在哪

### 1.1 `2026-09-16-h3c-scheme-selection.md`

| 旧句 | 错在哪 |
| --- | --- |
| 开篇「Sol-H3 不是更适合我们 5090 的通吃方案」 | 用户否决的就是这种叙事。Sol-H3 5090 **热 30.6s 全含**是社区一手（muninn，同一 5s 1344 图）。不适合当 **H3 官方 VAE 默认栈** ≠ 不适合当 **速度标尺**。 |
| 「该采用的默认栈：金线仍是本仓 INT8…官方 VAE」 | 把未验证金线（本仓 5s 1344 热 e2e **159s**）设成产品默认。社区已验证的是 Comfy 2-stage ~27s。 |
| 金线 MAE 当质量红线，有损刀「不进 gold」当不搬社区的理由 | 金线没有社区验证。用户认的是 Comfy 片。少步 / lift / kitchen attn 是那条 27s 配方的一部分。 |
| 用 Spark 56s「口径陷阱」（跳过 H3 VAE、换 LTX）否定整条 Sol 线 | 口径拆穿是对的；错在用它挡住 **5090 30.6s** 和 **Sol-Attn sm_120**。56s 不是我们的 KPI；30.6s 是。 |
| 5s 1344 金线、5s 1376 lift、用户 10s≈27s 口头混用 | 同一份文档里 27s 分母没钉死。REPORT 后改口「用户这次是 5s≈27s」，选型文仍写「10s 768P ≈27s」当对照。 |
| 「A 金线 / B 加速 / C 8 卡」三条公路并列，金线排第一 | 初心是一定质量上的极速。公路顺序应是 **极速配方 → 对照金线**，8 卡冻结。 |

### 1.2 `2026-09-15-h3c-module-opt-plan.md`

| 旧句 | 错在哪 |
| --- | --- |
| 「禁止改金线默认。有损刀进加速档，不进 gold」 | 把金线默认写成不可碰的产品合同。kitchen 核是官方 VAE 的实现，不是有损刀；加载常驻是无损。 |
| 「VAE 未追上 Comfy 质量合同与秒数之前，冻结整体 e2e」 | 模块拆账有用；「冻结 e2e」被拿来推迟 27s 对照。社区尺子就是 e2e。 |
| 画布写死 1344，1376「只作产品画布备注」 | 用户认的 2-stage JSON **就是 1376×768**。用 1344 协议否定 1376 实测，是口径武器。 |
| 「不要新开 cafe 冲用户 27s」 | 27s 是社区地板，不是禁止靠近的数字。禁止的是 **切块 / TAE / 假 sage** 装快，不是禁止冲速度。 |
| 196 窗 ~18s vs Comfy ~4s 写清楚了，但又把「再减窗」和「核」绑在合同里 | 差在核。把 18s 留在「合同内下限」叙事里，执行者会停手。枢纽 §4.2 其实已经说要 kitchen——**优先级被金线门压住了**。 |

### 1.3 `out/verify_taomate/REPORT.md`

| 旧句 | 错在哪 |
| --- | --- |
| 每轮徽章「金线未动」 | 当成绩。用户要的是极速，不是金线静止。 |
| 「官方 VAE 合同下限 ~18s」「已经贴计算下限」「这份 C 路径做不到 Comfy 几秒」 | **这是本文要钉死的错话。** 同 256/64、同 36 层 ViT，Comfy kitchen ~4s。18s 是 **C 碎核的现状**，不是合同，不是算力下限。Batch-4 GEMM −0.5s 只证明「还不是 kitchen」。 |
| 用「合同禁止减 tile / 整幅 RoPE / TAE」推出「再抠只能换 SDPA 才能 graph」然后停 | 换融合核 **不改几何合同**。停手理由不成立。 |
| 27s 拆账有用（VAE +13s、高分 DiT +11s、低分 +7s），但又说 INT8 recycle「不是能打进 27s 的刀」 | 单刀不够 ≠ 主线不该做 VAE。算术「VAE=18 则 e2e≥34」成立，结论应是 **必须把 VAE 做到 4s**，不是接受 18s。 |
| 5s/10s、1344/1376 在同一 REPORT 里换尺子 | 旧 P0 10s Z0+TAE 26.9s 花屏曾当过 KPI；后来 5s 1376 lift 52s vs 27s。必须废旧分母。 |

**一句话：** 旧文用未验证金线当质量上限，用 18s 碎核当速度下限，用口径差当「社区方案不适合我们」。三件事都否。

---

## 2. 社区标尺（2026-09-16 检索，一手）

| 来源 | 硬件 | 画布 / 时长 | 热墙钟 | 含什么 | 对我们的用法 |
| --- | --- | ---: | ---: | --- | --- |
| **用户 Comfy 2-stage JSON** | 1×5090 | **1376×768 · 5s**（864×480 → LBH → 1376；Duration 控件曾是 10，对账按用户 5s） | **~27s** 双采样热（CLIP 已缓存） | TE 热跳过；官方 VAE tile 256/64；kitchen attn；INT8 常驻 | **速度地板 + 质量地板** |
| **muninn Sol-H3 两阶段** [Spark 对照表](https://ai-muninn.com/en/blog/h3-ltx-two-stage-dgx-spark) | 1×5090 | 1344×768 · 5s · H3 4-step 384p 草稿 + LTX 3-step | 冷 40.3 / **热 30.6s** | 全含 MP4；热 = 模型仍在。Stage1 33.5GiB **5090 装不下、每轮重载** | **全含热跑量级**。质量尺是 LTX decode，**不**当 H3 VAE 金线 |
| muninn 同页 native H3 4-step | 1×5090 | 1344×768 · 5s | 冷 51.5 / 热 **41.5s** | 全含 | 单段 768P 少步的社区数 |
| muninn [5090 两阶段详解](https://ai-muninn.com/en/blog/minimax-h3-ltx-hybrid-refine-rtx5090) | 1×5090 | 同上 | 冷 hybrid 40.3 vs native 51.5；warm hybrid 32.2（含 Gemma）vs 30.6（Spark 表） | 加载单独列：hybrid 加载 15.9s，native 10.0s | **加载必须单列**。常驻才是 30s 档 |
| [ChrisJohnson Comfy 5090](https://github.com/ChrisJohnson89/minimax-h3-comfyui) | 1×5090 | 1344×768 · 5s · 10-step | 热 **85.23s** 全节点 | 含 TE、VAE、音频、MP4；模型 warm | 10-step 单段，不是 2-stage。证明 5090 上「普通 Comfy 768P」仍是分钟下沿，**27s 来自配方不是来自卡** |
| Comfy PR [#15446](https://github.com/Comfy-Org/ComfyUI/pull/15446) | 未钉 5090 | 175×1344×768 decode | **14.6→13.9s**（显存为主） | chunked I/O；bitwise 同 | 不是 kitchen 4s；是峰值。本仓不要拿 14s 当 kitchen |
| 枢纽已引 Comfy 同画布 decode | 对照栈 | 5s 768P 量级 | **~4s** | kitchen 核 + 256/64 | **VAE 段目标** |
| SGLang cookbook | 24GB 级 | tiled decode | — | `--component-residency vae=resident`；VAE 放进 layerwise offload 会 **每 tile 再流 ~9GiB** | **加载常驻的反例与正例** |
| Sol-Attn | 5090 sm_120 CuTe | 长序列 | 官方 On-Device 50-step 768P **231.2s**（近无损 4.52×，不是 27s 配方） | 核存在 | **DiT 768P 注意力插件**，不是换 LTX |
| Sol-Engine H3-OnDevice | 5090 | 1344×768 · 50-step | 1045→231s | 无 LoRA | 证明 sm_120 核有；**分母是 50-step**，禁止除进 27s |
| zenn 5090 本地 | 5090 | 多为 480p | Sage 83.9→57.5s（竖 480） | 4-step turbo 画质/声失败 | 480p 禁止当 768P 地板 |

TaoMate JSON 锚（用户图，源 `input/minimax-h3-r2v-taomate-2stage.json`）：

- 160/161 阶段1 **864×480**；162/163 成片 **1376×768**（分辨率选择器 150/153 **没接线**）。
- SplitSigmas step=3 → 3 NFE 低 + 2 NFE 高；LoRA `taomate_h3_3step_comfy` 0.75；attn = **comfy kitchen**；VAEDecode tile=256 overlap=64。
- 本仓热对照 comfy27/comfy5s6_lift：**e2e 52–53s**，VAE **17.6–18.2s**，高分 2 NFE **~19s** 且 INT8 `window=1`。

---

## 3. 新尺子（替换 27s 口头混用）

对外只准用这一张表。缺任何一列 = 禁止报「打到 27s」。

| 列 | 值 |
| --- | --- |
| 设备 | 1× RTX 5090（sm_120，32GB） |
| 冷 / 热 | **热**：进程已加载，第二次生成只换 seed。冷另报 |
| 画布 | **1376×768**（极速配方）。1344 对照臂单独标注 |
| 时长 | **5s / 124 帧 / 有声 MP4** |
| 质量地板 | 用户认可的 Comfy 2-stage 片：无 16px 格子、无切块换馆。抽 f0/24/48/90/119 |
| 速度地板 | **~27s** 热 e2e（Comfy 双采样，TE 已缓存） |
| 更好 | 全含量级贴近 **Sol-H3 5090 热 30.6s**（须在日志写明：那是 H3 草稿+LTX 精修，不是本配方质量尺） |
| 加载 | `vae_load` / `dit_host_stage` / `te` 单列。热目标：**几乎 0** |
| 分段 | 至少：te, dit_low, lbh, dit_high, vae_load, vae_blocks, mux, total |

禁止：5s×2 冒充 10s；480p cafe；Z0+TAE 26.9s 花屏；只报 DiT；把 18s VAE 说成已经极致。

---

## 4. 我们方案哪里不够好（机制，不讲情）

### 4.1 VAE：196 窗 C 碎核 vs kitchen

现状（REPORT comfy27 热 clip2）：VAE **18.2s**。几何已经是 Comfy：`tile=256 overlap=64`、tile-local RoPE、7×4 空间 × 7 时间 = **196 窗**、seq≈1797、36 block、F16 GEMM + cuDNN SDPA。`TILE_BATCH=4` 只把 18.7→18.2（−0.5s）。naive `H3_VAE_FLASH_SDPA` 走 D=64 generic，clip1 **304.8s**。

Comfy ~4s 不是更少窗，是 **每层融合**：`rms_rope_split_half` + `int8_attention` / `optimized_attention` + `linear_input_act`（RMS/SwiGLU），一行最多 4 tile 一次 `_decode_pixels`。本仓：层内多次 launch、GEMM 经 F32↔F16 cast（`gemm_f16_via`）、4 tile 的 D=64 注意力仍 **串行 cuDNN**（`vae_kitchen_sdpa` 有 batched 入口，热账证明没吃满）。内部 `blocks≈10.6s` + 墙钟 stitch/D2H → 17–18s。

**18s 不是合同下限。** 合同是 256/64 RoPE 与 blend。核可以换。把 18s 说成极致，是用几何红线给碎核开脱。

### 4.2 加载 / pin：社区几乎 0，我们当主路径税

社区：Comfy VAE patcher 留下、第二次 Queue 跳过 CLIPLoader；SGLang `vae=resident`（否则 tiled decode **每 tile 回流 ~9GiB**）；vLLM mmap+`cudaHostRegister`；PR 15456 decode 前赶走 DiT、VAE 留下。

本仓：

- `H3_VAE_WEIGHT_CACHE` **默认关**（`=1` 才跨 decode 留 pack）。generate 只在 **taomate official** 才 `setenv`。
- `H3_VAE_CUDA_W_RESIDENT` 逻辑默认尝试设备 W，但 generate **只在 taomate 末 clip** 才强调；中间 clip 还要给 INT8 让路。
- `weights_load` 成功后立刻 `cudaHostUnregister(mmap)`。下一 clip 再 register，pin 寿命 = 一次 load。
- `--repeat` 下一 clip 前 `h3_vae_int8_cuda_release_cached()`：**丢掉 GPU 包**，clip2 再走 dequant/H2D。社区热跑不会每条请求拆 VAE。
- `phase_ms` 把 load 埋进 `vae=`。主题档冷 load 曾 ~6s。

DiT host_stage 13.2→1.26s 是做过的。VAE 加载几乎没按社区常驻做完。kitchen 再快，每请求 6s 读权也会把 27s 吃穿。

### 4.3 sage3 假 hit

`h3_gpu_cuda.cu`：`g_sage3_want++`，失败则 `g_sage3_fallback++`，清 error，**继续 dense cuDNN**。768P 金线/turbo4 `seq≈37788`：**hits=0 fallback=200** 仍报 `sdpa=sage3`。加速档 taomate 5s 上 hits=500 是另一形状（能进核），但池子占 2–4GiB。

假 hit 的伤害：日志看起来在用快注意力；实际付满 S²；VRAM 被池子占住。加速档要求插件缺失或 shape 不支持 = **硬失败**，不要 fallback 装成功。

### 4.4 INT8 `window=1`

高分 2 NFE @ 1376：sage3 之后 `free≈1.1–1.7GiB` + reserve → **window=1**，clip2 `host_hits` 上千。sage3 核 ~11ms/层，墙钟 19s 里大量是 **权流**。cudnn 对照 window=30 仍 ~31s（注意力算力）。Comfy：**kitchen 小 workspace + 50 层 INT8 常驻**，两条同时有。我们用快核换来不能常驻。这不是「32GB 不够」，是 **池子与权抢同一张卡**，没有做 Comfy 那种 decode 前卸注意力池、采样时权常驻。

### 4.5 TE 冷

MISS ~13s（32B Qwen NVFP4）。HIT ~0，但 cache 不是 daemon；冷两边都不是 27s。热评测必须 `--te-cache` 且进程常驻。把冷 TE 折进 27s 是撒谎；热路径 TE 不是主墙，但 **没有常驻 generate 就没有社区那种第二次 Queue**。

### 4.6 mux

`libx264 -preset fast -crf 18`，768P ~1.8–2s。能力应 <2s，可与 decode 重叠。不是主墙。U=8 假片 ~6kb/s 已有审计。mux 失败 = 臂失败。

### 4.7 回代（不许再把 52s 说成「VAE 合同内已尽力」）

热 lift 现状 ~52s ≈ DiT 低 10 + LBH 2.5 + 高 19 + VAE 18 + mux 2。厨房 VAE ~4s（−14）+ INT8 常驻/小 WS attn 把高分 19→~8–10（Comfy 份额）+ 低分 10→~4：算术上 **可以看见 27s**。继续说「C 再压打不进」是拒绝换核。

---

## 5. 社区方案我们还能改进什么

不做「换一个魔改 binary」。做社区已验证的机制，迁进 C。

| 序 | 搬什么 | 不搬什么 | 验收 |
| ---: | --- | --- | --- |
| **P0** | **kitchen 解码器核** → C/CUDA：融合 RMS+RoPE、tile 内 F16/INT8 attn 一次 launch、FFN 少 roundtrip。几何 **仍 256/64 tile-local** | 不减窗、不整幅 RoPE、不上 TAE、不嵌 Python | 同 latent vs Comfy：无 waffle；热 `vae_blocks+load` → **~4s** |
| **P0** | **加载常驻**：mmap pin 活过多次 decode；GPU W 在 DiT 释放后常驻；WEIGHT_CACHE **默认开**（`=0` 才关）；clip 间可卸 GPU W 但 **禁止卸 pin/mmap**；热第二次 `vae_load≈0` | 不把 7GB host F32 当热路径 | `--probe-vae-load --vae-repeats 2` 第二次 0；`phase_ms` 有 `vae_load` |
| **P0** | **Comfy 2-stage 日程** 当默认极速配方：864×480 ×3 → LBH → 1376×768 ×2，LoRA 0.75，官方 VAE | 不把 R2V 9 图缺省假装已对齐；T2V 写明 | 人眼过质量地板；e2e 冲 27s |
| **P1** | **sage3 / kitchen attn / Sol-Attn sm_120**：缺核或 768P shape 不支持 = 硬失败。目标取消 sage3 大池，让 INT8 常驻 | 禁止 hits=0 继续 | 日志 `hits==want` 或直接 abort |
| **P1** | Sol-Attn sm_120 CuTe 打 seq~37k | 不把 On-Device 231s/50-step 当 27s | 加速档对照 |
| **不做产品默认** | LTX 成片 / Sol-H3-Spark 56s 配方 | 32GB 双栈会变成换权税（muninn：5090 每轮 reload Stage1） | 30.6s 只当速度参照 |
| **冻结** | 8 卡 Ulysses / aikitoria P2P | `peer=0` 已经更慢 | 解冻条件不是 27s |

VDN 仍不是 VAE。Flash-VAED / KJ Triton VAE 不支持 H3 ViT。这些旧排除 **保留**，因为是机制对不上，不是金线门。

---

## 6. 架构：preset 为极速配方服务，失败可见

```text
h3_cuda_generate
  ├── kernel: TE | DiT(quant, attn) | VAE | mux | lift
  ├── backends: 失败硬退、可测、可回滚
  └── recipes (--preset):
        默认极速 / taomate-lift  → 2-stage + kitchen VAE + 常驻 + 真快 attn
        turbo4 / nvfp4           → 实验支路
        gold                     → 8-step cudnn 1344 对照，不挡优化
```

今天 `--preset` 只能切步数 / LoRA / lift，不能切 VAE 核、不能让 sage3 假 hit 死掉。

要补：

| 缺口 | 需要 | 失败行为 |
| --- | --- | --- |
| `vae=` | `official_tile`（现碎核）→ `kitchen`（目标）→ `tae`（预览） | kitchen 不可用硬退。禁止静默 TAE |
| `attn=` | `cudnn` / `sage3` / `kitchen` / `sol` | shape 不支持硬退。禁止 fallback 还报原名 |
| `resident=` | TE/DiT/VAE 热常驻 | pin 失败打日志并记入 `load_ms`，禁止静默 pageable |
| 日志 | 冷热、画布、步数、是否含 load/TE/VAE、backend 真名 | 对外禁止只报 DiT |

通吃 binary 不存在。正确形态是配方表。**极速配方是默认要打的那条**；gold 是对照。

---

## 文件地图（施工前锁边界）

| 文件 | 职责 |
| --- | --- |
| `h3_vae_int8.h` / `h3_vae_int8.c` | RoPE 合同、store mmap、**默认 cache/batch 开关**（CPU 可测） |
| `h3_vae_int8_cuda.c` | decode、weights_load、tile batch、hostreg 寿命 |
| `h3_vae_int8_cuda.h` | load/release/unpin API |
| `h3_gpu_cuda.cu` | `vae_kitchen_sdpa` batched、未来融合 RMS+RoPE；sage3 硬失败 |
| `tools/h3_cuda_generate.c` | 热路径默认常驻；`phase_ms` 单列 load；末 clip 才卸 DiT |
| `h3_generate_preset.c` | 极速配方；gold 不挡 VAE 核 |
| `tests/test_h3_vae_rope.c` | 256 tile-local 公式（**不改断言方向**） |
| `tests/test_h3_vae_resident.c` | 默认 cache=on、batch=4、`=0` 能关 |
| `tests/test_h3_generate_preset.c` | 配方失败可见 |

---

### Task 1: VAE 加载常驻默认开（无损）

**Files:**
- Create: `tests/test_h3_vae_resident.c`
- Modify: `h3_vae_int8.h`、`h3_vae_int8.c`、`h3_vae_int8_cuda.c`、`h3_vae_int8_cuda.h`、`tools/h3_cuda_generate.c`、`CMakeLists.txt`

**Interfaces:**
- Produces: `int h3_vae_int8_weight_cache_wanted(void);` — 默认 1；仅 `H3_VAE_WEIGHT_CACHE=0` 为 0
- Produces: `int h3_vae_int8_tile_batch_want(void);` — 默认 4；env `H3_VAE_TILE_BATCH` 覆盖；clamp 1..4
- Produces: `void h3_vae_int8_cuda_hostreg_release(void);` — munmap 前卸 pin

- [x] **Step 1: 写失败测试**

```c
/* tests/test_h3_vae_resident.c */
#include "h3_vae_int8.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int fails;
static void expect(int c, const char *m) {
    if (!c) { fprintf(stderr, "FAIL %s\n", m); fails++; }
}
int main(void) {
    unsetenv("H3_VAE_WEIGHT_CACHE");
    expect(h3_vae_int8_weight_cache_wanted() == 1, "cache default on");
    setenv("H3_VAE_WEIGHT_CACHE", "0", 1);
    expect(h3_vae_int8_weight_cache_wanted() == 0, "cache 0 disables");
    setenv("H3_VAE_WEIGHT_CACHE", "1", 1);
    expect(h3_vae_int8_weight_cache_wanted() == 1, "cache 1 on");
    unsetenv("H3_VAE_TILE_BATCH");
    unsetenv("H3_VAE_FLASH_SDPA");
    expect(h3_vae_int8_tile_batch_want() == 4, "tile batch default 4");
    setenv("H3_VAE_TILE_BATCH", "1", 1);
    expect(h3_vae_int8_tile_batch_want() == 1, "tile batch 1");
    setenv("H3_VAE_TILE_BATCH", "9", 1);
    expect(h3_vae_int8_tile_batch_want() == 4, "tile batch clamp 4");
    return fails ? 1 : (printf("test_h3_vae_resident: PASS\n"), 0);
}
```

- [x] **Step 2: 跑测试，确认失败**（符号不存在）

Run: 把 `h3_vae_resident_test` 加进 `CMakeLists.txt`（只链 `h3_vae_int8.c` + `h3_safetensors.c` + `h3_w4a8_dequant.c` + `h3_w4a8.c`，同 `h3_vae_rope_test`），编译运行。

Expected: FAIL link / 未定义 `h3_vae_int8_weight_cache_wanted`

- [x] **Step 3: 最小实现**

`h3_vae_int8.h` 在 `store_free` 后声明两个函数。`h3_vae_int8.c`：

```c
int h3_vae_int8_weight_cache_wanted(void) {
    const char *e = getenv("H3_VAE_WEIGHT_CACHE");
    if (e && e[0] == '0' && e[1] == '\0') return 0;
    return 1;
}
int h3_vae_int8_tile_batch_want(void) {
    const char *e = getenv("H3_VAE_TILE_BATCH");
    int n = 4;
    if (e && e[0]) n = (int)strtol(e, NULL, 10);
    if (n < 1) n = 1;
    if (n > 4) n = 4;
    return n;
}
```

`h3_vae_int8_cuda.c`：`vae_weight_cache_enabled()` 改调 `h3_vae_int8_weight_cache_wanted()`。tile `bmax` 默认改 `h3_vae_int8_tile_batch_want()`（不再绑 `FLASH_SDPA`）。`weights_load` **成功路径不要 `cudaHostUnregister`**；静态记下 map，提供 `h3_vae_int8_cuda_hostreg_release`。同一 map 二次 load 跳过重复 register。

`tools/h3_cuda_generate.c`：`phase_ms` 增加 `vae_load=%.1f`（`vst.load_ms`）。cleanup 在 `h3_vae_int8_store_free` 之前调 `hostreg_release`。`--repeat` 仍可 `release_cached` 卸 GPU W，**不得**卸 mmap pin。

- [x] **Step 4: 跑测试，确认通过**

Run: `h3_vae_resident_test` 与已有 `h3_vae_rope_test`。

Expected: PASS。RoPE 测试不得改断言。

- [x] **Step 5: 不 commit**（用户禁令）

---

### Task 2: kitchen 对齐入口（几何不变，核开始并）

**Files:**
- Modify: `h3_gpu_cuda.cu`（`h3_cuda_qkv_rope_sdpa_f16_dev_nb` / batched cuDNN）
- Modify: `h3_vae_int8_cuda.c`（一层内 RMS+RoPE+SDPA+FFN 少 HBM roundtrip；默认 `TILE_BATCH=4` 且 batched SDPA 必须走 `one_cudnn`，失败打 `vae_kitchen_sdpa: batched failed` **且** 热路径计为未对齐）

**Interfaces:**
- Consumes: Task 1 的 `tile_batch_want()`、常驻 W
- Produces: 热 decode 日志必须出现一次 `vae_kitchen_sdpa: one_cudnn batch>=2`；禁止再默认 naive flash

- [ ] **Step 1: 用现有同 latent 抽帧当质量门**（不改 RoPE 测试）

p01 中帧 / 4× 近邻无 16px 棋盘。`tests/test_h3_vae_rope.c` 继续证明 overlap 处 tile-local ≠ canvas。

- [ ] **Step 2: batched SDPA 失败视为未完成，不是「fallback 也算 kitchen」**

`h3_gpu_cuda.cu` 已有 `h3_cuda_sdpa_cudnn_f16_run_batched`。fallback_loop 必须在 stderr 打明确失败；加速档可后续改硬退。本任务先让默认 batch=4 走通 batched，热 VAE 再测。

- [ ] **Step 3: 禁止把 18s 写回「下限」**

任何新 REPORT 只准写：现状 vs Comfy ~4s。

- [ ] **Step 4: 不 commit**

Kitchen 完整 36 层融合（RMS+QKV+RoPE+SDPA+gated FFN 一层一 launch）若本任务只打通 batched SDPA，在日志写 `kitchen_align=partial`。完整融合是 Task 3，不要假装 4s 已到。

---

### Task 3: kitchen 36 层融合核（主墙）

**Files:**
- Modify: `h3_vae_int8_cuda.c` 36-block 循环
- Modify: `h3_gpu_cuda.cu` 新 fused kernel 或 graph（**禁止**默认 `H3_VAE_FLASH_SDPA` naive）
- 对照只读：https://github.com/comfy-org/comfy-kitchen `rms_rope_split_half` / `int8_attention`

**Interfaces:**
- Produces: `h3_cuda_vae_block_fused(...)`（一层：RMS → QKV GEMM → split_half RoPE → SDPA → out → FFN）。权重已在设备（Task 1）。

- [ ] **Step 1: 单 block vs 现碎核数值**（同 tile latent，MAE 门自定像素级，waffle 红线优先）
- [ ] **Step 2: 36 层接上，热 5s 1376 `vae_blocks` 冲 ~4s**
- [ ] **Step 3: 不 commit**

---

### Task 4: 极速 preset 失败可见（不挡 Task 1–3）

**Files:** `h3_generate_preset.c`、`tests/test_h3_generate_preset.c`、`h3_gpu_cuda.cu` sage3 路径

- [ ] sage3：`want>0 && hits==0` 在加速档 **硬失败**（gold 可继续 cudnn，不必走 sage3）
- [ ] `--preset gold` 保持 8-step cudnn；**不要**在文档或代码注释写「必须先过金线才能优化」
- [ ] 不 commit

---

## 7. 执行顺序

```text
Task 1 加载常驻默认开 ──► Task 2 batched kitchen SDPA（几何不动）
        │
        └─► Task 3 36 层融合核 ──► 热 5s 1376 重测 vae= 段 vs ~4s
Task 4 sage3 硬失败 / preset    ── 可并行写，不开 8 卡
Ulysses / LTX 成片默认          ── 不做
```

SSH 可用，只为在 5090 上测 Task 1–2 的 `load_ms` 与 waffle，**不为**乱切块冲 27s。

---

## 8. 不要做的事

- 不要把金线 MAE 当发布门。不要写「金线不动所以不搬」。
- 不要把 18s 写成合同下限 / 算力下限。
- 不要减 256/64、不要整幅低频 RoPE、不要 TAE 成片冒充 VAE 赢了。
- 不要 naive flash。不要 8 卡 Ulysses。不要 commit。不要把密码写入仓库。
- 不要一上来换 LTX 成片当默认质量尺。30.6s 是速度参照。
- 不要 5s/10s、1344/1376、冷/热混报 27s。

---

## 9. 一手链接

- 用户 2-stage JSON：`input/minimax-h3-r2v-taomate-2stage.json`（本仓；检索未找到公开同名 gist）
- Comfy MiniMax VAE：https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/minimax/vae.py
- comfy-kitchen：https://github.com/comfy-org/comfy-kitchen
- Comfy VAE chunked I/O：https://github.com/Comfy-Org/ComfyUI/pull/15446
- SGLang H3 residency：https://lmsysorg.mintlify.app/cookbook/diffusion/MiniMax/MiniMax-H3
- Sol-H3-Spark：https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3-Spark/
- Sol-Attn / On-Device 5090：https://nvlabs.github.io/Sana/Sol-Engine/H3-OnDevice/
- muninn 5090 vs Spark 30.6s：https://ai-muninn.com/en/blog/h3-ltx-two-stage-dgx-spark
- muninn 5090 两阶段：https://ai-muninn.com/en/blog/minimax-h3-ltx-hybrid-refine-rtx5090
- ChrisJohnson 5090 Comfy 85s：https://github.com/ChrisJohnson89/minimax-h3-comfyui
- MiniMax-H3 tile 警告：https://github.com/MiniMax-AI/MiniMax-H3/issues/46
- 本仓旧文（口径作废，事实可引用）：[选型](./2026-09-16-h3c-scheme-selection.md) · [枢纽](./2026-09-15-h3c-module-opt-plan.md) · [REPORT](../../../out/verify_taomate/REPORT.md)
