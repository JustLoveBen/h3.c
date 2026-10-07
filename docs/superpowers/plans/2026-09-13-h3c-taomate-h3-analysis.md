# TaoMate-H3 分析（768P / 10s / 单卡 5090 对照）

> **For agentic workers:** 纯分析。禁止改推理代码 / 改金线默认 / commit / 碰 `out/` 实测报告。密码不入库。  
> **本仓现行尺子：** 单卡 RTX 5090 · **5s · 1344×768 · p01_sprint**（[768P 评测协议](./2026-09-13-h3c-768p-eval-protocol.md)）。**禁止**把旧 864×480 cafe 金线（78.7s 等）和本文横比倍率。  
> **TaoMate 公开尺子：** **8×H20 96GB Hopper（SM90）** · **480×864** · **10s** · seed 8301 · **TP2×Ulysses4**。**不是 5090，不是 768P。**  
> **一手：** [GitHub TaoLiveAIGC/TaoMate-H3](https://github.com/TaoLiveAIGC/TaoMate-H3) `main` `@ccc1a70`（2026-09-09；本地 `third_party/accel_survey/TaoMate-H3/`，gitignore，不 commit）· [HF TaoLiveAIGC/TaoMate-H3](https://huggingface.co/TaoLiveAIGC/TaoMate-H3)  
> **对照：** [完整极致施工图](./2026-09-13-h3c-5090-extreme-full-plan.md) · [SelfLift](./2026-09-13-h3c-selflift-comfy-analysis.md) · [PCIe 多卡](./2026-09-11-h3c-pcie-multigpu-lessons.md)

**一句话：** TaoMate 赢在「**3-step 区间蒸馏 LoRA × 4 小块 × clean-KV**」——公开 11.45× 的分母是同机 **50-step MiniMax H3**，并行轴两边都有，**不是**相对本仓 4-step/双采样的倍数。开源命令 **不能单卡、不能 5090（FA3/Hopper）**、默认出片 **一次性 VAE**。用户实测 **单卡 5090 + 双采样 · 10s 768P ≈ 27s** 与开源 CLI 不是同一条栈；**双采样不在他们仓库里**，是 SelfLift 同类叠刀。27s 物理上说得通（尤其流式重叠 + 少步 + 低分前缀），但不能当成「官方 TaoMate 单卡 5090 e2e」。对本仓：**不要抄 TP2×U4**；P0 是 768P 上让注意力真正变便宜（chunk Q / 修好 sage3 fallback）和 **10s 的 chunk+KV**；3-step LoRA 只许 opt-in，且必须配他们的日程，金线不动。

---

## 0. 尺子（先钉死再谈秒）

| 谁 | 硬件 | 画布 | 时长 | 并行 | 口径 |
| --- | --- | --- | ---: | --- | --- |
| **TaoMate README 表** | **8×H20 96GB** Hopper | **480×864** | **10s** | TP2×U4 | Pure DiT **不含** load / TE / VAE / mux；另排除内部 **Base10 音频预备** |
| TaoMate 768p 产品画布 | 同上（表未给 768p 秒） | **1376×768** 或 **768×1376**（CLI 默认竖屏 `768x1376`） | 5 的倍数 | 仅 `--gpus 4\|8` | 短边 480/768/1088，两边 %32 |
| **本仓现行** | **1×RTX 5090** | **1344×768** | **5s** · p01 | U=1 | `phase_ms total` **全含** TE+DiT+VAE/TAE+mux |
| 本仓 lift 双采样 | 同上 | 低分前缀 → LBH → 高分后缀 | 5s | U=1 | 加速档；金线仍原生 8-step |

**1376 vs 1344：** 他们 768p 横屏是 **1376×768**（VAE `/16` → 86×48，每帧 token 43×24=**1032**，5s `video_seq`≈**38,184**）。我们官方 adapt 是 **1344×768**（84×48 → 1008 token/帧，`video_seq`=**37,296**），约 **+2.4%** token。`H3_MAX_PIXELS = 768*1344` **拒收 1376×768**。对齐 1376 是产品决定，不是免费加速。

**开源命令硬约束（`config.py` + `cli.py` + `attention_hook.py`）：**

- `--gpus` 只能 **4 或 8**；`tensor_parallel_size==2` 写死；8 卡 U4 / 4 卡 U2。
- 注意力后端 **只认 FA3**（`flash_attn_interface`，README 要求 Hopper/SM90）。**5090 = sm_120，官方树跑不起来。**
- `publication=one_shot_audio_video_vae`：全部 5s 请求的 latent 算完 → **一次** video VAE → 一次 audio VAE → 一次 mux。
- DirectPolicy **不是 CLI 开关**（chunk 切法、3 锚点、KV 保留、W8A8 范围都冻死）。

旧文用 480p 金线 78.7s / turbo4~40s 档当本仓对照，**作废**。480p 数字只允许出现在「他们的公开表」一节。

---

## 1. TaoMate 加速从哪来（源码，能拆就拆）

发布物 = **rank 128 / alpha 128 / step-3000 generator-EMA LoRA**（HF，非 PEFT 目录名）+ **PyTorch 流式 runtime**。验证机 8×H20。

### 1.1 3-step LoRA（主因，相对 50-step）

`denoise_schedule.py`：`DISTILLED_STATE_INDICES = (0, 16, 33, 49)`，从官方 **50 步 shifted sigma** 抽 4 个锚点 → **3 次** Euler-η0。视频 shift=12、音频 shift=3（与金线双时钟同构）。蒸馏目标是跳完 `[0→16]` / `[16→33]` / `[33→49]`（`stage3_target_state_indices`）。

LoRA：`token_refiner.blocks.{0,1}` + `blocks.{0..49}` × `{qkv,out,fc1,fc2}` = **52 模块 / 208 对 A/B**。扁平 `*.lora_a/lora_b`，推理 **不 merge**。本仓 `h3_lora_parse_name` **显式拒绝 `token_refiner`**（防和 backbone `blocks.N` 撞名）。

**相对他们表里的 50-step：** 10s = 两个 5s 请求的话，基座约 **100 次全片 forward**，TaoMate **24 noisy + 8 clean = 32 次小块 forward**。小块 Q ≈ 全片的 ~1/4，GEMM 量级约 `100×37` vs `32×9.25` ≈ **12×**，和公开 **11.45×** 同量级。  
**拆账（同一 8×H20，480p 10s Pure DiT）：**

| 刀 | 占 11.45× 的角色 |
| --- | --- |
| **50→3 步蒸馏** | **主因。** 没有它，chunk 只是把同一步数切碎，总 GEMM 不降 |
| **4 小块（更小 Q）** | 把 3+clean 的实际 GEMM 压到「约等于 4-step 全片」，并让注意力从 S² 变成 S_chunk×S_window |
| **8 次 clean-KV** | **额外成本**（第 4 次 forward/块），换的是长片连续，不是表上的倍数 |
| **TP2×U4 / FA3 / 内层 W8A8** | 对照臂 MiniMax H3 **同一拓扑**，**不解释 11.45×** |

**相对本仓（已经 8-step 金线 / 4-step turbo / 3+1 lift）：** 3 vs 4 全片大约 **25% DiT**；真正拉开 768P 的是 **Q 变短 + 窗口 K**（我们 768P sage3 `hits=0 fallback=200` @ `seq=37788`，等于每步仍付全序列注意力）。

### 1.2 Chunk 几何（`streaming/geometry.py`）

5s = native 124 帧 = 4 phase，`chunk_group_counts=(2,2,2,1)`：

| phase | 组 | 帧 | video latent | 3-step | clean |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 2 + 5 帧前缀 | 39 | **12** | 3 | 1 |
| 1 | 2 | 34 | **10** | 3 | 1 |
| 2 | 2 | 34 | **10** | 3 | 1 |
| 3 | 1 | 17 | **5** | 3 | 1 |
| **5s 合计** | | 124 | **37** | **12** | **4** |

续段（请求 1+）走 `canonical_continuation_plan`：不再造 5 帧前缀，7 组 = 119 帧 / **35** latent。故 **10s = 24 generation + 8 clean-KV**，与 README 一字不差。

### 1.3 Clean KV（`streaming/cache.py` + `attention_hook.py`）

每层 BF16 K/V `[tokens, local_heads, 128]`。只提交 **去噪完、`timestep=1.0` 再 forward 一次** 的 KV（`streaming_cache_only=True`）。保留：**首块 video sink + 最近 2 个完整 AV 块**。当前 Q 看 `[text + 持久 clean KV + 本块 KV]` 一次 dense FA3。另有块级 `affine_to_first_stream_chunk` renorm、跨请求 2 video / 9 audio latent transport。音频 KV 每 12 个 5s 请求清一次。

**768P/10s 才是这块的主场：** 全片 10s 的 S 约 2×，S² 约 4×；流式每块 Q 仍是 5–12 latent。480p 5s 上 GEMM 与 4-step 全片几乎打平（旧结论仍对）；**不要**再用 480p 说「流式没有吞吐优势」来封死 768P 10s。

### 1.4 音频 Base10 teacher（表外）

流式 3-step **不真正生成音频**。独立 torchrun、**不挂 LoRA 的 BF16 基座**、audio-only 10 状态（9 forward），在 3/6/9 抓 milestone，流式步里整段覆盖。公开表 **excludes the command's internal audio preparation**。用户墙钟若含这一段，会明显大于 Pure DiT。

### 1.5 精度 / 内核 / 并行

- `bf16_with_interior_fc1_qkv_w8a8`：只量化内层 `qkv`/`fc1`，护首 2 + 末 3 块，`out_proj`/`fc2` 全程 BF16。比本仓全 50 块 INT8 **更保守**。
- Triton 融合名 `enable_h3_sol_engine_fusions`；静态 timestep AdaLN cache。
- **TP2×U4 前提是 NVLink H20。** 5090 PCIe 上 TP2 的 hidden 级 AR 是灾难（施工图 / pcie-lessons）。**禁止把他们的拓扑当 5090 无 P2P 的答案。**

### 1.6 公开数字（480×864 · 10s · 8×H20 · 必须带尺子）

| 指标 | TaoMate | MiniMax H3 | 倍 |
| --- | ---: | ---: | ---: |
| Pure DiT | **14.810 s** | 169.572 s | 11.45× |
| 首块 final latent | **6.148 s** | 170.052 s | 27.66× |
| 专项 bench 首可播 | **17.287 s** | 183.313 s | 10.60× |
| 峰值 DiT 显存 | 31.37 GiB | 32.03 GiB | — |

口径：DiT = 32 次 forward 的 CUDA event（跨 rank max）。**首可播含 Video VAE + H.264，来自 matched first-chunk publication benchmark，不是开源 `one_shot` 命令。** 768p 秒数 **他们没公布**。

---

## 2. 「双采样」是不是 TaoMate 里的 SelfLift？

**不是。** `src/taomate_h3/` 无 SelfLift / lift / latent upsample / 两档分辨率日程。每个 chunk 都在 **用户请求的原生画布**（480/768/1088 短边）上做 3 个 Stage3 interval。

用户说的 **「单卡 5090 + 双采样」** = 在 5090 上 **另外叠了** 与本仓 `--preset lift` 同类的东西（低分前缀 + 过渡 + 高分后缀），或 Comfy/改版 runtime。官方树：`--gpus`∈{4,8} + FA3/SM90，**不能**解释成开箱单卡 5090。

不要把 27s 记成「TaoMate 官方单卡成绩」。记成：**用户栈 = TaoMate 3-step/chunk 思想（或权）+ 双采样 + 5090 能跑的注意力**，墙钟 27s/10s/768P。

---

## 3. 用户 27s / 10s / 768P / 单卡 5090 是否说得通

**不否定实测。** 标清未知项：是否含 TE encode、是否含 VAE/TAE/mux、是否含音频 teacher、是整段 e2e 还是首可播、画布是 1376 还是 1344/1280。

### 3.1 回代（标「回代」）

本仓已测 **5s · 1344×768 · p01 · 1×5090**：

| 臂 | e2e | DiT 相关 | 备注 |
| --- | ---: | --- | --- |
| 金线 8-step cudnn 官方 VAE | **159.38 s** | DiT **112.23** | TE 13.14 / VAE 23.57 |
| turbo4+TE+TAE | **80.48 s** | 全分辨率 4-step；sage3 **fallback** | TAE 7.03 |
| **lift 双采样 n=2** | **47.09 / 50.39 s** | 低 3 步 ~6.8–7.0 + 高 1 步 ~15.2；合计 **~22 s**；LBH **4.63 s** | 能认人，仍糊/过饱和 |

金线 ~14 s/步 @ 全 768P packed。10s 若 **两次 5s 全片 3 NFE**：粗算 2×3×14 ≈ **84 s 仅 DiT**，再加 VAE，**远大于 27s** → 用户不可能在跑「两次全分辨率 dense 3-step」。

若 **5s 块 ×2，每块 3 NFE，Q 只有本块（12/10/10/5 latent）**：

- GEMM ~ 全片 3-step 的 1×（每 token 仍走 3 次）+ 8 次 clean ≈ **4-step 全片 GEMM**。本仓 4-step 768P DiT 仍 ~60 s/5s（sage3 死了）→ 10s ~120 s。还是太慢。
- 所以 27s **还必须**有：(a) **注意力不再付 S=37k**（chunk Q + 窗口 K，或 5090 上能用的快核）；(b) **双采样让多数 NFE 落在 ~1/4 token**；(c) 很可能 **TAE/分块 decode 与下一 chunk DiT 重叠**；(d) TE cache / 不计 teacher。

本仓 lift 已经证明：低+高 DiT ~22 s / 5s。朴素 ×2 ≈ 44 s DiT / 10s。要落到 **27s 全含**，需要过渡+decode 被重叠吃掉、且高分步也按块跑（Q 短），或 27s **不是**全含 e2e。

**首可播口径：** 官方专项 480p 8 卡是 17.29s。单卡 5090 上「第一块 3-step + 一刀 lift + TAE」落到 20–30s 量级 **合理**。若用户报的是 **开始能播** 而不是 10s mp4 落盘，27s 更容易。

**结论：** 27s 作为「少步 + 分块 + 双采样 + 快 decode、单卡 5090、10s 768P」**物理上可成立**。作为「官方 `python -m taomate_h3`、FA3、8×H20 表的 5090 换算」**不成立**。差我们 **47s/5s** 的主因不是 1376 vs 1344（2.4%），而是：**他们（用户栈）在做 10s 流式少步；我们在做 5s 一次出满，且 768P 注意力没吃到 sage3，高分还留 1 步 ~15 s。**

### 3.2 和 47s/5s 怎么比（归一）

| | 秒 / 秒视频（e2e） | 备注 |
| --- | ---: | --- |
| 用户 27s / 10s | **2.7** | 口径未完全钉死 |
| 本仓 lift 47s / 5s | **9.4** | 全含、一次出满、画糊/过饱和 |
| 本仓 turbo4+TE+TAE 80s / 5s | **16.1** | 全分辨率 4-step |
| 本仓金线 159s / 5s | **31.9** | 8-step |

横比「27 vs 47」会低估差距：时长 2×、我们还没做 10s。按每秒视频大约 **3×+**。不要用 480p 78s 去解释这个缺口。

---

## 4. 本仓 C 栈能借鉴的 P0 / P1

| 优先级 | 项 | 做什么 | 不要做什么 |
| --- | --- | --- | --- |
| **P0** | **768P 注意力变便宜** | 修 sage3 在 `seq≈37788` 的 fallback；或 port **按 codec 组切块、Q 只含本块** | 以为再砍 1 步（4→3）就能追上 27s |
| **P0** | **10s = chunk + clean-KV** | 5s×2 + sink/recent2 + 音频 tail rollover + 块 renorm；长片成本有界 | 把 74 latent 塞进一次 dense（S²×4） |
| **P1** | **3-step 锚点日程 opt-in** | `{0,16,33,49}` + 匹配学生权；**金线 8-step 零污染** | 把 TaoMate 权挂到现行 **全片 4-step turbo4** 上赌质量 |
| **P1** | 流式首块 / 分块 TAE | TTFF；chunk i+1 DiT ∥ chunk i decode | 把 README「首可播 17s」当成开源命令行为 |
| **P2** | 音频 teacher 解耦 | audio-only 9-forward 保音质 | 3-step 联合 AV 硬扛人声 |
| **P2** | 画布 1376 | 仅当产品要咬他们像素；需抬 `H3_MAX_PIXELS` | 当加速项 |
| **不做** | **TP2×U4 上 8×5090** | — | NVLink 解；PCIe 上 AR 税见 pcie-lessons |
| **不做** | FA3 / vLLM W8A8 / 嵌他们的 Python | — | 5090 无 FA3 hopper；量化覆盖面比本仓 INT8 窄 |

QKV packed 一次 A2A：本仓 `h3_ulysses_comm.cu` 已按此方向，**不是新 P0**。U=2/8 在 768P 已测且未赢 turbo/lift，施工图路线不变。

---

## 5. 质量：流式 + 3-step 的代价

- **相对 50-step / 本仓金线：** 另一条轨迹，MAE 无意义。身份靠 sink KV + renorm，不是逐像素对齐。
- **相对本仓 turbo4：** 他们的学生是 **小块+clean 历史条件** 训的；全片 4-step larryvrh 是另一条。谁好看取决于场景，不能从秒数推。
- **相对本仓 lift：** 官方 TaoMate **不 lift**，原生 768p 小块 3-step，没有 LBH 过饱和路径。用户若叠了双采样，糊/过饱和会 **叠加上去**——「平衡」更可能来自 **3-step LoRA 本身比 4-step+lift 更像老师**，而不是 lift。
- **音频：** 3-step 视频学生 + 9-step 基座音频，是他们敢少步的质量机关；公开秒数把它藏了。
- **开源一次性 VAE：** 流式的产品价值（边出边播）要自己写分块 decode；抄仓库命令抄不到。

---

## 6. 要不要把 TaoMate LoRA / 3-step 接到 `h3_cuda_generate`？

**可以做 opt-in 实验臂，不要当默认，金线绝对不动。**

| 接法 | 建议 |
| --- | --- |
| 权直接挂现行 turbo4（均匀 4-step、全片、无 KV） | **不要。** 训推失配：锚点不同、未见 clean 历史、块大小 12≠37、含 token_refiner。本仓解析还会因 `token_refiner` 失败 |
| 新 preset：3-step `{0,16,33,49}` + 转换加载器（非 PEFT 名、r=128、52 模块）仍全片 | **弱实验。** 可 A/B 一眼，预期不如他们的流式质量；秒数大约 0.75× 现 4-step DiT，**追不上 27s/10s** |
| 3-step **且** chunk+KV（真对齐） | **这才是「接 TaoMate」。** 工程量 L；先 P0 注意力/分块，再挂权 |

HF：`TaoLiveAIGC/TaoMate-H3`（`adapter_config.json` = `{rank:128, alpha:128, weight_source:generator_ema, optimizer_step:3000}`）。权不入库。

---

## 7. 来源

- https://github.com/TaoLiveAIGC/TaoMate-H3 `@ccc1a70`（README；`config.py`；`denoise_schedule.py`；`streaming/{geometry,cache,attention_hook,runtime,session}.py`；`model/{pipeline,denoise,geometry}.py`；`inference/{lora_checkpoint,acceleration,w8a8,base10_teacher}.py`；`distributed/parallel.py`；`runner.py`；`cli.py`；`teacher.py`）
- https://huggingface.co/TaoLiveAIGC/TaoMate-H3
- 本仓 768P 秒数：金线 159.38 / turbo4+TE+TAE 80.48 / lift n=2 47.09–50.39（施工图工人 2026-09-14，`out/review_videos_768p/`）

未改推理代码。未 commit。未触碰 `out/`。浅克隆仅 `third_party/accel_survey/TaoMate-H3/`（gitignore）。
