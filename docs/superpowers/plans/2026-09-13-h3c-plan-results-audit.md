# h3.c 方案/实测审计（2026-09-13）

## 第二轮摘要（成片复算，2026-09-13 下午）

第一轮成立的发布/文档问题（explainer 5–8s、`sp2/sp8` 废片仍像正式报告、三套时间线）不变。本轮用本机 mp4 独立抽帧 + 远端 inode/md5 把 **11.159** 钉死，并修正第一轮对「U=8 可能是拷贝」的存疑。

**成片（本机新抽 RGB24，f0/24/48/90/119，MAE 0–255）：**

| 对照 | f0 | f24 | f48 | f90 | f119 | mean | max\|Δ\| |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 01 vs 02 | 10.060 | 10.184 | 10.777 | 14.232 | 10.506 | **11.152** | 224 |
| 01 vs 03 | 同上（03≡02） | | | | | **11.152** | 224 |
| 02 vs 03 | 0 | 0 | 0 | 0 | 0 | **0** | 0 |
| 01 vs 04 | 43.004 | 44.539 | 42.283 | 30.180 | 48.695 | **41.740** | 255 |
| 01 vs 05 | 45.567 | 46.740 | 40.237 | 35.544 | 50.722 | **43.762** | 255 |
| 04 vs 05 | 23.766 | 24.494 | 32.521 | 28.814 | 19.839 | **25.887** | 254 |

- **02≡03 字节相同**（md5 `607ce93d…`，1 726 934 B）。**06≡07 也相同**（`e881179c…`）。
- 不是本机 scp 拷错：远端不同 inode / 不同 mtime / 不同 generate.log，mp4+`frame_0000.ppm`+`audio.wav` 仍全等。U=8 **真跑了**（`ulysses=8`、`heads=7`×8 卡、e2e 212.29s）。
- 同一 md5 还等于远端 **当前 U=1** `verify_te_cache/ab_nocache`（及 HIT）。同 binary 质量门 **U=1≡U=2≡U=8，MAE=0**——第一轮写「未测」是报告没写，文件上已经是 0。
- **11.152 ≈ 报告 11.159**：锚的就是本机 `01_gold` = `verify_5090x8_gold/run_4`（2026-09-12T17:30Z，只 load `libnvrtc.so.13`）。当前栈 preload `so.12`。差 0.007 是 H.264 相对远端 PPM。
- 音频：01 peak **0.1471** 未削波；02=03 **0.1416**（与金线不是一条轨）；04 解码 **0.961**（报表 PCM=1.000）；05 **0.763**。04/05/金线/Ulysses **四条音轨都不同**（仅 02=03）。
- 目视：01/02 同机位同叙事、无花屏/色块/错轴，细节漂（手势、杯位）对得上 MAE~11；04/05 轨迹已换但仍是咖啡馆、时间轴对齐。抽帧在 `out/review_videos_2026-09-13/audit2_frames/`（不 commit）。

**方案还能当什么：** 两档拆分 + 单卡 turbo4/TE cache 仍是工作底稿。**Ulysses 不能当 8×5090 加速主路径**（正确、更慢）。**5–8s 算术物理上到不了**（漏了 host_stage ~13s；U=8 还把 DiT 做慢）。对外仍不能当发布稿。

**最严重新发现（第一轮成立，新证据是…）**

1. P0-3 修正：11.159 不是「Ulysses 糊了」，是 **旧金线片 vs 当前 binary**；同 binary 置换是位级对齐。报告仍把 11 写成过门，错在锚，不在切分。
2. P0「U=8 可能是拷贝」不成立——八卡 cafe 独立产出与 U=2/U=1 **同一份像素**。秒数不可比仍成立（U=8 冷 n=1 vs U=2 warm）。
3. explainer 5–8s：四刀相乘漏掉金线 host_stage / HIT 记进 DiT 的 ~9–13s；再叠加 U=8 已测 137s DiT。地板在 **~40s（再加未测 TAE）或 ~50s（现状 HIT）**，不是 5–8。
4. 04 与 05 MAE **25.89**：TE HIT 不是 A3 turbo4 那条片（跨 binary）。HIT vs 同 binary MISS=0 仍只能信 PHASE2，不能拿 04 当 05 的无损对照。
5. 金线污染面：CLI 默认仍 opt-in（`ulysses=1`、`--te-cache` 关、help 写了 `H3_TE_PROMPT_CACHE`）。漂移来自 **运行时 nvrtc preload / 覆盖后的 generate**，不是默认 argv。

下文第一轮原文未改。复算细节见文末「第二轮深入」。

---

只读。不改推理代码。不 commit。不开 GPU 长跑。  
对照尺子（文件口径）：864×480 · 真 5s · cafe · seed `0x11`。  
现行秒数以 `out/verify_5090x8_gold/`、`out/verify_fast_sku/`、`out/verify_te_cache/`、`out/verify_turbo_impl/PROGRESS_PHASE2.md` 为准。  
`PROGRESS.md`、explainer 附录 B.4、`out/verify_5090_sp2/`、`out/verify_5090_sp8/` **不是现行**。

**总判：** 单卡金线 / turbo4 / TE cache 的秒数大体能对上，默认路径也仍是 opt-in。问题在于：**对外说明文档还在卖已被实测打穿的 8 卡故事；官方 Ulysses 报告目录仍是 IMA 修前废片；质量门把「当前 binary 相对旧金线片漂了 11」当成 Ulysses 过门。** 方案还能当工作底稿，不能当发布稿。

---

## P0 — 会误导发布 / 质量门假绿

### P0-1 对外 explainer 仍把 8 卡写成「5–8s」，并把 Ulysses / TE cache 写成没测

**证据**

- `docs/superpowers/plans/2026-09-13-h3c-turbo-explainer.md` 开篇：「已实测做到 **64 秒**（单卡），全部手段落地后的现实目标是 **约 5–8 秒**（8 卡，推断）」；主线② TE 缓存「**规划中**」；主线④「代码写完，还没在真机上验证」；§2.4「C3/C4 **未跑**」；§2.6 / §4.1 / §4.4 反复写 8 卡 e2e **5–8s**、U=8 DiT **2.5–4s**。
- 同一文写「项目状态以 `PROGRESS.md` 为准」。`PROGRESS.md` 停在 `U2_SHORT_RC0` / C4 incomplete。
- 现行 `PROGRESS_PHASE2.md`：U=2 cafe warm e2e **83.40s / DiT 34.66s**（比金线 **78.74 / 31.67 更慢**）；U=8 cafe n=1 e2e **212.29s / DiT 137.43s**；TE HIT e2e **61.41s**、turbo4+TE HIT **~49.85s**。文内写「**不宣称 U=2/U=8 加速**」。

**为什么是问题**

5–8s 的算术前提是「Ulysses 把 turbo4 DiT 16.47s 砍到 2.5–4s」。同机已经跑过 5s×8step：U=2 没加速，U=8 把 DiT 拖到 4.3×。再叠加「TE 还没做 / Ulysses 还没上 GPU」——两句都假。这是发布叙事，不是内部草稿笔误。

**建议**

explainer / 任何对外页立刻改口：8 卡 Ulysses **已测且更慢**；TE cache **已测 opt-in**；5–8s 整行删或改成「当前实现下不成立」。状态源改 `PROGRESS_PHASE2.md`。

---

### P0-2 计划指定的 Ulysses 正式报告仍是 IMA 修前废片，且未标过时

**证据**

- 实现计划 C3/C4 落点：`out/verify_5090_sp2/REPORT.md`、`out/verify_5090_sp8/REPORT.md`。
- 这两份 **没有**「IMA 修前 / 勿作现行」条。数字：U=2 DiT **496.56s** MAE **48.536** peak=**1.0000** VAE ~118s；U=8 DiT **236.38s** MAE **45.130** peak=**1.0000**，run_4 mux「~6 kb/s……成片高度可疑」。
- 修后数字只写在 `PROGRESS_PHASE2.md`（U=2 34.66s / U=8 137.43s / MAE 11.159 / peak 0.1417）。本机 **没有** `verify_5090_sp2_concurrent/` 成片目录。

**为什么是问题**

按计划文件图打开「正式报告」，读到的是花屏/饱和/500s 那一版。PHASE2 再诚实，也盖不住官方路径上的假现行。混用旧 496s 与新 35s 会直接写错结论。

**建议**

这两份 REPORT 顶栏加「IMA 修前，作废」。修后 cafe 另册（或覆盖）并链到 PHASE2。旧 mp4 不要放进评片目录。

---

### P0-3 Ulysses 质量门锚在「已漂的旧金线片」，不是同 binary 单卡；11.159 被当成过门

**证据**

- 金线 `out/verify_5090x8_gold/REPORT.md` **自身无 MAE**。run_4 日志 `2026-09-12T17:30Z`，`dit_format: INT8 ConvRot`，peak=0.1472。
- TE cache REPORT：「相对金线 run_4 的 ~11.16 是**当前 binary vs 金线片**（三臂相同）……金线 binary 已被覆盖。」旧 nocache vs 金线也是 **11.159**。
- PHASE2 U=2 / U=8 vs 同一 `gold/run_4`，五帧 **逐格相同**：10.084 / 10.194 / 10.783 / 14.310 / 10.423，mean **11.159**。文称「与当前 binary 无 cache 对金线 11.159 完全相同……不判质量 bug」。上一版正确性门 **9.713**，「阈值不改」。
- 合同（implementation C3、next-opt P1）：对照是 **同机单卡 cudnn 同 prompt/seed**；落到 sage3~27 才当 bug。**没有**「对旧金线片 11 就算过」这一条。
- 本机金线 run_4 **无 PPM**（只有 mp4/log），U=2/U=8 修后帧也不在本机。11.159 无法就地复算。也 **没有**「U=2 vs 当前 U=1」MAE。

**为什么是问题**

Ulysses 是置换，正确性门必须是 **同 binary U=1**。现在的 11.159 只证明「修后多卡和当前单卡一样漂离旧金线片」——若表是复制的，连这个都没有。金线 REPORT 无自 MAE，11 的分母不明。9.713→11.16 不改阈值、直接「不判 bug」，是放水。U=2 与 U=8 五帧小数全同，在缺原始帧时不能排除粘贴。

**建议**

冻结当前 U=1 cafe 为新锚（另册，写 binary hash）。重报 U=2/U=8 **vs 该锚**（期望 ~0 / FP 噪声）。旧金线 11.16 只当「binary 漂了」记一笔。在复算前不要写「质量对齐」。

---

## P1 — 数字或方法不可比

### P1-1 尺子和 n 不齐：U=8 冷 n=1、U=2 只 1 次 warm、HIT 的 DiT 含 HOST 记账

**证据**

- 金线 / turbo4：冷 1 + warm×3。
- U=2 cafe：冷 1 + warm **1**。U=8 cafe：**n=1 冷**。
- TE HIT DiT 40.35s、turbo4+TE HIT DiT 25.55s vs 册内 turbo4 DiT 16.47s：PHASE2 写明 TE 跳过后 `dit_host_stage` 记进 DiT。金线 run_4 日志里 host_stage 在 TE 段就 `done`（wall 13.2s），不进 DiT 31.44s。
- turbo4+TE MISS DiT **20.55s** vs 册内 turbo4 **16.47s**（同机同配方，差 ~4s）。PHASE2 写同一 inode `7451464`，但册内 turbo4 是更早的 A3 四跑。

**为什么是问题**

U=8 212s 对金线 78.74s 方向对（就是慢），不能当同口径倍率。HIT 的 DiT 不能跟金线/turbo4 DiT 横比。turbo4 16.47 与 turbo4+TE 49.85 可以比 e2e，比 DiT 会撒谎。

**建议**

表上标 n / 冷热 / host_stage 是否进 DiT。U=8 至少再补 1 次 warm 再写「约 2.7×」。不要用 HIT DiT 讲「DiT 变慢」。

---

### P1-2 turbo4 MAE 41.97 被写成极速过门；合同里这是 Mag 档边缘，且音频饱和

**证据**

- `out/verify_fast_sku/REPORT.md`：MAE **41.97**（30.3–48.9），「目视过极速尺子」；`audio_pcm` **peak=1.000**（四跑相同）。
- 实现计划 A3 / SKU §1：sage ~27 预期可放行；**Mag 52–68 当 FAIL**（花/糊）。41.97 在中间，更靠 Mag。
- explainer §2.1：「目视通过」；不提 peak=1.0。金线 peak=0.1472。turbo4+TE HIT peak=0.7715。
- SKU「能看」没有第二人、没有禁饱和、没有动作用例。上游 #14 已警告 4 步拖影。

**为什么是问题**

极速门被收成「有人看过 cafe 抽帧」。MAE 42 + 削波，按自己写的 Mag 尺子至少是黄灯。explainer 把它讲成方法学示范，发布会当成「质量 OK」。

**建议**

SKU 补一条硬指标：peak≤某值，或显式「允许削波」。41.97 保持「只记录」，过门改口为「单次目视，未过第二人/动作集」。不要把 42 说成可接受标准。

---

### P1-3 Ulysses 作为 8×5090 主路径，原理上就已经很难加速；文档还在用线性摊薄

**证据**

- PHASE2 U=8：`peer_access ok=0 fail=56 peer_a2a=0`。`h3_ulysses_comm.cu` 写明 GeForce 无 NVLink 时 `canAccessPeer` 常为 0，NCCL 走 SHM/host，**不是 P2P**。
- A2A 仍 F32（`dtype_bytes != 4` 直接拒）。U=8 → 每卡 **7 头**。U=2 已比单卡慢（34.66 vs 31.67）；U=8 再慢一个数量级。
- lightning-survey §4.2 估 U=2 DiT 20–21s、U=8 DiT 8–10s / e2e ~56s。model-format §5：Ulysses「DiT 推断 2.5–4s（未测）」。实测打穿。
- 实现侧：pad 用 `seq_real`、cudnn plan 按 device 分槽、generate 打印 `DiT SP attached`——插点在。慢的是 PCIe A2A + 无 peer + 头数被切碎，不是「没接线」。

**为什么是问题**

金线「唯一同轨迹刀 = Ulysses」在这台机器上目前是 **越切越慢**。再把 16.47÷8 写进 5–8s，是方法学错误，不是估时误差。

**建议**

8×5090 主叙事改成：单卡 turbo4 + TE cache（已测 ~50s HIT）；Ulysses 降级为「正确性已冒烟、性能未赢、默认保持 1」。U=4 未测，不要猜甜点。有 peer / BF16 payload / 去 per-block 屏障之前，禁止再写「8 卡再摊薄」。

---

### P1-4 金线污染面：CLI 默认仍干净，但 `H3_TE_PROMPT_CACHE` 可无开关脏 HIT

**证据**

- generate 默认：`ulysses=1`、`preset=NULL`、`te_cache_dir=NULL`、`vae_decode=official`。TAE 在非 turbo4 硬失败。无 DiT NVFP4 打开路径（TE 才是 NVFP4）。
- `tools/h3_cuda_generate.c`：未传 `--te-cache` 时读 **`H3_TE_PROMPT_CACHE`**。TE cache REPORT 已有「脏 HIT key `63ba422170cf015a`」前科（无 engine 的 MISS 写入）。
- 金线命令本身没带 cache；金线 log 也无 HIT。风险在环境变量，不在缺省 argv。

**建议**

金线/对照跑清空该环境变量；脏 key 目录与评片目录隔离。help 写明 env 与 `--te-cache` 等价。

---

### P1-5 短测 1s/2step 仍占据 `PROGRESS.md` 的 C3 状态位

**证据**

- `PROGRESS.md`：`status=U2_SHORT_RC0`，C3 = 「1s/2step … DiT **12.66s**」，C4 = incomplete。
- PHASE2 把 12.66s 标成短测，5s cafe 另表。explainer 若跟 PROGRESS，会把短测当进度顶点。

**建议**

`PROGRESS.md` 顶栏改「过时，见 PHASE2」。短测秒数禁止进 5s 表。

---

## P2 — 文档过时或表述夸大

### P2-1 多份方案仍写「Ulysses 尚未实现 / 金线 66s / turbo 32–42s」

- SKU Architecture：「本仓当前 **无多卡、无产品 LoRA**」；金线锚仍是旧机 **66.06s**；推荐单卡「**~32–42s（推断）**」。实测 turbo4 **63.69s**。
- principle-matrix：「本仓 **尚未实现**」Ulysses；金线 ~66s。
- `out/verify_fast_sku/REPORT.md` 正文仍写「U=2 cafe / U=8 **未跑**」（后半才补 TE HIT）。
- model-format §5 P0：Ulysses「DiT 推断 2.5–4s（**未测**）」——已测且反号。

这些当历史选型可以留，当现行状态不行。

### P2-2 lightning-survey 与磁盘代码打架（以磁盘为准）

磁盘 `2026-09-13-h3c-minimax-lightning-survey.md` 仍写：C2 插点未接、`h3_dit_w4a8.c` 零 ulysses、generate 打印 `DiT SP … not driving this denoise yet`、每次 A2A `cudaStreamSynchronize`。

抽查（2026-09-13 工作区）：`layers_forward_ulysses` 存在；SDPA 用 `seq_real` + pad 置零；generate 打印 `DiT SP attached`；comm 已改 event，**没有** `cudaStreamSynchronize`。  
调研里 pad 污染 / 少 TP 的方向仍有用。**进度事实过时。** 本文不改那份文件（可能另有人在写）。

### P2-3 NVFP4 DiT / TAE：**没有**被写成 cafe 已测；但仍有「权已在 = 能宣传」的缝

- PHASE2：TAE stub + 缺权，CLI 合同在，不成片。`h3_tae_open` / `decode` 一律 deferred。金线开 tae 硬失败。
- NVFP4 DiT：先写远端无文件，后写 `minimax_h3_fl2va_pruned_nvfp4.safetensors` 12G **已在远端**；头探测有，**未跑 cafe**。generate 无 `open_nvfp4`。
- model-format / explainer 对这两项大体诚实。不要把「权已 ls 到」写进已测表。

### P2-4 本机无 `out/review_videos_2026-09-13/`

`out/from_5090/` 有旧的 `02_ulysses_u2_5s_cafe.mp4`（9 月 11 日树）。不要和修后片混看。拷片 agent 未完成，本审计未等人眼新片。

### P2-5 PHASE2 自己的页眉也过时

`PROGRESS_PHASE2.md` 头仍是 `phase=B1` / `B1_TAE_BLOCKED`，正文已含 C3/C4 cafe、TE、NVFP4。比 `PROGRESS.md` 新，但头不能当状态机。

---

## 尺子对照（已知宣称 vs 文件）

| 宣称（2026-09-13 汇总） | 文件 | 判 |
| --- | --- | --- |
| 金线 78.74±0.86 / DiT 31.67 | gold REPORT warm 2–4 | 合。无 MAE。 |
| turbo4 63.69 / DiT 16.47 / MAE 41.97 | fast_sku REPORT | 合。peak=1.0 未进「过门」叙事。 |
| TE HIT 61.41 / TE~0 / MAE 0 vs MISS；vs 金线 ~11.16 | te_cache REPORT | 合。n=1。脏 HIT 已排除。 |
| turbo4+TE HIT ~49.85 / MAE 0 vs MISS；vs 金线 44 | PHASE2 + fast_sku 后半 | e2e 合。DiT 25.55 含 host_stage。n=warm×2。 |
| U=2 cafe 83.40 / 34.66 / MAE 11.159 | **仅 PHASE2** | 秒数合。MAE 锚有病（P0-3）。n=1 warm。 |
| U=8 cafe 212.29 / 137.43 / MAE 11.159 | **仅 PHASE2** | 秒数合。n=1 冷。MAE 与 U=2 逐格相同。 |
| 旧 sp2/sp8：U=2 DiT~496 MAE~48 | sp2/sp8 REPORT | 合，但是 **未标作废的现行文件**（P0-2）。 |
| NVFP4 权远端未测；TAE stub | PHASE2 后半 | 权后来出现，仍未 cafe。TAE 合。 |
| 8×5090 peer=0，不宣称 Ulysses 加速 | PHASE2 | 合。explainer 仍在宣称 5–8s（P0-1）。 |

实现抽查未发现「报告说已接线、代码没有」：Ulysses / pad / per-device plan / TE cache opt-in / TAE stub 与 PHASE2 一致。lightning-survey 相反，是文档落后代码。

---

## 方案还能信什么

**还能信**

- 单卡金线 ~79s、turbo4 ~64s、重复 prompt 上 TE cache 能把 TE 打到 ~0（同 binary 无损）。
- 默认仍 U=1、无 preset、官方 VAE、cache 默认关（除 env）。
- 未蒸馏 `--steps 4`、Kijai W4A8 主题、TAE 进金线、NVFP4 替换金线——这些禁令还在，也未被报成已测。
- IMA 根因（cudnn plan 跨卡）与修法（按 device 分槽）对得上代码。

**不能再信**

- 「四刀落地 → 8 卡 5–8s」。
- 「Ulysses 是这台 8×5090 的主加速路径」。
- 「U=2/U=8 已对金线过质量门」。
- `PROGRESS.md` / explainer 附录 B.4 / `verify_5090_sp{2,8}/REPORT.md` 作为现行。

**方案本身**

SKU 拆两档、少步必须换 LoRA、公开倍数不准除进 66/79——这几条没烂。烂的是杠杆 4 被写成「唯一无损快刀」却没把「无 P2P 的 PCIe A2A + heads/U」算进可实现区间。turbo4 作为独立快档成立（墙钟真降了），质量故事没写完。

---

## 第二轮深入（成片复算）

只读。不改推理代码。不 commit。不开新的 5s GPU cafe。抽帧与 `SUMMARY.json` 在 `out/review_videos_2026-09-13/audit2_frames/`（gitignore，不入库）。密码未写文件。

### A. 成片身份与独立 MAE

本机 `cmp`/`md5`：

| 文件 | 大小 | md5 | inode | mtime（本机） |
| --- | ---: | --- | ---: | --- |
| 01 金线 | 1 756 448 | `b0fd20f2edd7f7a6` | 8194991 | 01:31 |
| 02 U=2 | 1 726 934 | `607ce93db4ff896f` | 8195004 | 09:16 |
| 03 U=8 | 1 726 934 | **同 02** | 8195005 | 09:23 |
| 04 turbo4 | 2 047 593 | `0c2822336f8ef368` | 8194992 | 02:02 |
| 05 TE HIT | 1 998 773 | `4467c61fe6b561be` | 8195006 | 09:31 |
| 06 短测 U=2 | 356 298 | `e881179c9b392cdb` | 8195009 | 09:11 |
| 07 短测 U=8 | 356 298 | **同 06** | 8195010 | 09:19 |

01 ≡ 本机 `verify_5090x8_gold/run_4/out.mp4`。04 ≡ `verify_fast_sku/cafe_turbo4/run_4/out.mp4`。

**02≡03、06≡07 是字节级相同，不是「五帧表抄同一行」。** 第一轮怀疑本机拷片剂拉错源。远端只读核对（不同 inode，生成日志 mtime 对齐各自 mp4）：

| 远端路径 | inode | mtime UTC | mp4 md5 | f0.ppm | wav |
| --- | ---: | --- | --- | --- | --- |
| `sp2_concurrent/cafe_warm` | 53346981 | 01:16:41 | `607ce93d` | `13c04d48` | `9188012d` |
| `sp2_concurrent/cafe_cold` | 53346855 | 01:15:16 | 同 | （同组） | 同 |
| `sp8_concurrent/cafe_1` | 53477549 | 01:23:57 | 同 | `13c04d48` | `9188012d` |
| `te_cache/ab_nocache`（U=1） | 53346679 | 09-12 23:52 | 同 | `13c04d48` | `9188012d` |
| `te_cache/ab_hit` | 53346554 | 09-12 23:51 | 同 | | |
| `gold/run_4` | 52823262 | 09-12 17:31 | `b0fd20f2` | `ef66a3f7` | `845b0bbf` |
| `sp2_overlap/short` | 53346727 | 01:11:06 | `e881179c` | | |
| `sp2_concurrent/u8_short` | 53477420 | 01:19:40 | `e881179c` | | |

U=8 cafe **不是拷贝**：日志 `ulysses=8`、`peer_access ok=0 fail=56 peer_a2a=0`、`tensor_seq=16000 heads=7` device=0..7 各一条、`phase_ms … dit=137434.5 … total=212288.1`、`DiT SP attached`。U=2 warm：`ulysses=2`、`heads=28`×2、`dit=34656.4 total=83396.9`。短测两边日志也各自成立（U=2 DiT 12.66s / U=8 106.05s），成片却位级相同。

结论：八卡 **真跑了**；产出像素/PCM 与当前 U=1、U=2 **位级对齐**。这是置换正确，不是「没跑 U=8」。

独立复算（ffmpeg `select=n` → RGB24，不用报告里的 11.159）：

| 对照 | f0 | f24 | f48 | f90 | f119 | mean | 每帧 max\|Δ\| |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 01–02 | 10.0601 | 10.1843 | 10.7765 | 14.2319 | 10.5062 | **11.1518** | 224/222/192/206/201 |
| 01–03 | 同 01–02 | | | | | **11.1518** | 同 |
| 02–03 | 0 | 0 | 0 | 0 | 0 | **0** | 0 |
| 01–04 | 43.0036 | 44.5389 | 42.2830 | 30.1795 | 48.6951 | **41.7400** | 255/255/232/212/255 |
| 01–05 | 45.5670 | 46.7400 | 40.2372 | 35.5438 | 50.7217 | **43.7620** | 255/254/254/228/255 |
| 04–05 | 23.7663 | 24.4938 | 32.5211 | 28.8135 | 19.8394 | **25.8868** | 253/245/238/230/254 |

对照 PHASE2/fast_sku：11.159（PPM vs gold）、41.971（turbo vs gold）、44.03 / 26.12（HIT vs gold / 旧 turbo）。本轮是解码后 mp4，差 0.007–0.23，方向一致。现成 `frames/` 里 02/03、06/07 png 也是 `cmp` 全等；以本轮新抽为准。

**音频（ffmpeg → s16le 32 kHz stereo）：**

| 片 | peak (i16 / 归一) | 削波样本 | 与谁同轨 |
| --- | ---: | ---: | --- |
| 01 | 4821 / **0.1471** | 0 | 金线独轨（报表 0.1472） |
| 02=03 | 4640 / **0.1416** | 0 | 彼此相同；= 远端 U=1 nocache wav（报表 0.1417） |
| 04 | 31500 / **0.9613** | 0（解码后） | 独轨。报表 PCM peak=**1.000**，AAC 把削波抹到 0.96 |
| 05 | 24989 / **0.7626** | 0 | 独轨（报表 0.7715） |

金线 / Ulysses / turbo4 / TE HIT **不是同一条音轨**。只有 02 与 03 是。

**目视（audit2_frames）：** 01 与 02 同一咖啡馆双人+雨窗+杯特写时间轴，无花屏/色块/错帧；漂在手势、杯位、窗光（对得上 MAE~11）。03 即 02。04/05 轨迹更亮、路景更实、面孔不同，仍是同一套镜头语法；05 f0 女发髻、男对视，04 更侧分；f90 都是杯特写但托碟/握杯不同。无人眼花屏。

### B. 金线锚点与 binary

- 金线 REPORT **仍无 MAE**。11.159 锚的就是这份 `01_gold` / `run_4`，不是更早的 66s 旧机片。本轮解码 MAE **11.152**。
- 金线 log：`2026-09-12T17:30Z`，`ulysses=1`，`dit_format: INT8 ConvRot`，**只** `loaded libnvrtc.so.13`（无 preload so.12），`sage3_lib=0`，peak=0.1472。
- 当前远端 generate：inode/size **7451464**，mtime **2026-09-13 01:10 UTC**（金线跑完后覆盖）。TE 三臂 / U=2 / U=8 日志都有 `preloaded libnvrtc.so.12` + 再 load so.13。
- TE 三臂「相对金线也是 ~11」= **当前栈 vs 旧金线片**，不是 cache，也不是 Ulysses。generate binary（及 nvrtc preload）≠ 金线 cafe 当时的运行时。
- **同 binary U=1 vs U=2 MAE：存在，且为 0**（远端 mp4/ppm/wav 全等）。第一轮「质量门未测」改为：**文件上已过置换门；报告没把 U=1 当分子，误把 11 当过门。**

### C. 方案是否自洽

1. **5–8s 到不了。** explainer §2.6：TE~0 + DiT 16.47÷8≈2.5–4 + TAE 1–3 + mux 1–2 = 5–8。漏项：金线 `dit_host_stage` **13.2s**（HIT 后记进 DiT，TE HIT DiT 25.55 vs 册内 16.47）；Amdahl 地板在 TE+VAE+host，不在「再除 8」。已测：TE HIT 金线 **61.41**、turbo4+TE **~49.85**、TAE stub 未成片、U=8 DiT **137**。即便 TAE 砍 VAE ~8s，现状 HIT 也只到 **~40s**。host_stage 不消失时，线性 Ulysses + TAE 的诚实下限大约 **15–22s**，不是 5–8。
2. **Ulysses 在方案里是主路径，实现里是可选。** SKU：「金线只加 Ulysses」「8 卡极速必做」。implementation Part C 与 explainer 主线④。CLI 默认 U=1（opt-in）。与 8×5090 `peer_access=0`、A2A F32 host bounce **矛盾**——主路径假设了线性摊薄，机器给的是 SHM。
3. **turbo4 MAE 42 在 SKU 里是黄灯。** 合同：sage~27 可放行，Mag 52–68 FAIL。41.74/41.97 夹在中间、更靠 Mag。人眼：本轮抽帧「能看、无花」，但是单人、无动作集、无禁饱和。PCM peak=1.0 未写入过门。
4. **三套时间线：** 对外 explainer + `PROGRESS.md`（C3 短测 / C4 incomplete / TE「规划中」/ Ulysses「未上 GPU」）**不是现行**。现行秒数与状态是 `PROGRESS_PHASE2.md` + gold/fast_sku/te_cache REPORT。fast_sku 正文仍写「U=2/U=8 未跑」。
5. **金线默认仍 opt-in。** `h3_cuda_generate.c`：`ulysses=1`、`preset=NULL`、`te_cache_dir=NULL`；help 写 `--te-cache` **或** `H3_TE_PROMPT_CACHE`、Default official VAE。远端 U=1/U=2/U=8 日志 `preset=none`。污染面是 env + 覆盖后的 binary/nvrtc，不是缺省 argv。

### D. 会不会让测试撒谎的实现

- `uly_run_ranks`：默认 **先全 rank enqueue 再全 rank `uly_sync_rank` → `cudaDeviceSynchronize`**。是并发发射，不是 SERIAL；每相仍整机排干。`jobs.compute_stream = NULL`（PHASE2/survey 已写）。
- `h3_ulysses_comm.cu`：`canAccessPeer` 为 0 则 `peer_a2a=0`，注释写明 NCCL **SHM/host bounce**；`dtype_bytes != 4` 拒。A2A 用 event，不是函数末 `cudaStreamSynchronize`。
- cafe U=2/U=8 **建成了** `heads=28` / `heads=7` cuDNN plan，sage/sol `want=0 fallback=0`。第一轮担心的「heads=H/U 退 flash」**没有发生在这两条 5s cafe**（短测 cafe 几何以外的 shape 另说）。
- TAE：`h3_tae_open`/`decode` 仍 deferred。PHASE2 写成不成片。没有报告把它标成 cafe 已测。
- NVFP4 DiT：generate 无 `open_nvfp4`。PHASE2 写成权已 ls、未 cafe。没有把「权在」写成已测秒数。
- `out/verify_5090_sp2/REPORT.md` / `sp8/REPORT.md`：**仍像正式报告**（U=2 DiT 496s MAE 48.5 peak=1.0；未标 IMA 作废）。修后数字只在 PHASE2。

### E. 秒数是否可比

| 对比 | 冷热 / n | 同一 cafe/seed | HOST_FWD | 能不能横比 |
| --- | --- | --- | --- | --- |
| 金线 78.74 vs turbo4 63.69 | 都是冷 1+warm×3 | 是 / `0x11` | 关 | **能**（e2e、分段） |
| 金线 vs U=2 83.40 | U=2 只 1 次 warm | 是 | 关（日志 device-resident） | 方向能比；n 不齐 |
| 金线 / U=2 warm vs U=8 212.29 | **U=8 仅 n=1 冷** | 是 | 关 | **不能当同口径倍率**（212/78.7 的 2.7× 偏冷） |
| TE HIT 61.41 / turbo+TE 49.85 的 DiT | n=1 或 warm×2；DiT 含 `dit_host_stage` | 是 | 关 | e2e 能比；**DiT 不能跟 31.67 / 16.47 横比** |
| U=2 冷 84.41 vs warm 83.40 | 同臂 | 是 | 关 | 能。成片冷≡热（位级） |
| 短测 1s/2step vs 5s cafe | 不同几何 | seed 同、时长不同 | 关 | **不能进 5s 表**（`PROGRESS.md` 仍占 C3 位） |

同一 cafe token-ids、同一 seed `0x11`：金线 / TE / U=2 / U=8 日志都成立。

### 第二轮对「方案还能否当主路径」

- **能当工作底稿：** 两档 SKU、turbo4 少步必须 LoRA、默认不改、TE cache opt-in 无损（同 binary）、Ulysses 置换在当前 binary 上是对的。
- **不能当 8 卡加速主路径：** `peer_access=0` + F32 A2A + 每相 DeviceSync；U=2 略亏、U=8 大亏。8 卡正当用法仍是多请求 DP，或等有 P2P 的箱子。
- **不能当发布目标：** 5–8s 与已测分段+host_stage 不相容；explainer / `PROGRESS.md` / `sp2/sp8` 仍在对外说话。
