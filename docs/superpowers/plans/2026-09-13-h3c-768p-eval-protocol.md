# H3.c 768P 专业评测协议

Date: 2026-09-13  
**现行主尺子：5 秒 · 1344×768 · 生成墙钟 e2e。**  
864×480 cafe seed `0x11` 的数字只作历史，禁止与本协议横比倍率。  
金线默认配方不改：8-step / `sdpa=cudnn` / 官方 VAE / 无 LoRA / `--ulysses 1`。  
本文件是评测/加速档合同，不是改默认路径。不 commit。密码不入库。

对照：[two-stage-768p](./2026-09-11-h3c-two-stage-768p.md) · [8x5090-768p-estimate](./2026-09-11-h3c-8x5090-768p-estimate.md) · [turbo-implementation](./2026-09-11-h3c-5090-turbo-implementation.md)

驱动：`tools/h3_768p_eval.sh` · 进度：`out/verify_768p/PROGRESS.md` · 成片：`out/review_videos_768p/`  
不要覆盖 `out/review_videos_2026-09-13/` 的 480p 片。

---

## 0. 一句话

以后只报 **5s 768P 墙钟**。768P 在本仓官方档是 **1344×768**（横屏），不是 RunningHub 宣传图上的 1280×768，也不是旧尺子 864×480。

---

## 1. 768P 在本仓库是什么（代码钉死，不猜）

| 来源 | 宽×高 | 性质 |
| --- | ---: | --- |
| `h3_adapt_canvas(1920, 1080)`（`tests/test_h3.c`） | **1344×768** | **官方 16:9 768p 横屏桶** |
| `h3_adapt_canvas(1080, 1920)` | **768×1344** | 官方竖屏 768p |
| `h3_adapt_canvas(32, 32)` | 768×768 | 官方方屏 |
| `H3_MAX_PIXELS` / `h3.c` 拒超限文案 | `768*1344` | 发布像素上限 |
| RH Lightning `short_edge=768` + `aspect_ratio=16:9` | **1344×768** | 产品 768p 与官方 adapt 一致 |
| CLI 直接 `--width 1280 --height 768` | 1280×768 | **能跑**（32 对齐且未超像素上限），**不是** 16:9 官方桶 |
| generate 默认（未改） | 864×480 | 历史 480p 默认；评测脚本必须显式传 1344×768 |

16:9 的 768 高本应是 1365.3 宽，被 `768*1344` 像素上限缩放后再按 `H3_CANVAS_MULTIPLE=32` 取整，才落到 **1344×768**。`1280×768` 是常见显示器/运营口播尺寸，**禁止当本仓 768P 尺子**。

**本协议横屏金线画布：`--width 1344 --height 768`。**

其它几何（公式，非推断）：

| 量 | 5s · 1344×768 | 旧 864×480（历史） |
| --- | ---: | ---: |
| fps | **24**（`H3_FPS`） | 24 |
| `--seconds` | **5**（禁止 `--frames` 假时长） | 5 |
| mux 帧 | 120 | 120 |
| 内部对齐帧 `17n+5` | 124 | 124 |
| `latent_t` | 37 | 37 |
| VAE 空间 /16 | 84×48 | 54×30 |
| 空间 token / latent 帧 | 42×24 = 1008 | 27×15 = 405 |
| `video_seq` | **37,296** | 14,985 |

---

## 2. KPI

主 KPI：**一条 5s · 1344×768 · 有声 mp4 的生成墙钟 e2e**（`phase_ms` 的 `total`，秒）。

必须同时记下分段（同一条 `phase_ms` / `dit_host_stage` 日志，单位 ms→s）：

| 段 | 日志键 | 备注 |
| --- | --- | --- |
| TE | `phase_ms te=` | Qwen 文本；与分辨率弱相关 |
| DiT | `phase_ms dit=` | 含部分 host 重叠；768p 才是新分母 |
| VAE | `phase_ms vae=` | 官方 decode；768p 约按像素 2.49× 外推 |
| host_stage | `dit_host_stage: done … wall_ms=` | 权从盘进 host；与 TE overlap |
| audio / ppm / mux | `phase_ms` 其余字段 | 通常不是主墙 |
| 有声 | ffprobe + `audio_pcm` peak | 见 §5 |

冷热：产品对照用 **warm**（同进程或同机热权后再跑）。报表写清 n 与冷/热。单条冒烟可以 n=1，标题必须写「冒烟 / 未作 warm 均值」。

**禁止：** 把旧 864×480 cafe 秒数改写成 768P；禁止 `480p_e2e × 2.49` 冒充已测 768P；禁止和 480p 表横比「×N」。

---

## 3. Seed 策略

| 规则 | 值 |
| --- | --- |
| 公式 | `seed = 0x76800000 + prompt_index`（`p01`→`0x76800001`） |
| 同 prompt 跨臂 | **同一 seed**（金线 / turbo / TE / TAE / lift 比质量） |
| 跨 prompt | **不同 seed**（避免一套噪声吃全部场景） |
| 旧 cafe `0x11` | **不再用于新尺子**；只留在 480p 历史册 |

---

## 4. 提示词集（专业评测，不再用 cafe 当唯一条）

原文在 `evals/768p/prompts/*.txt`。CUDA generate 无 `--prompt` 文本入口，必须先变成 Qwen ids（`tools/h3_tokenize_prompt.py` → `evals/768p/ids/<id>.ids` 或远端现算）。**禁止**把 cafe 574 ids 或 `--token-canary` 当成专业评测条。

| id | 中文一句 | 覆盖 | 为何收入 |
| --- | --- | --- | --- |
| `p01_sprint` | 雨后跑道冲镜短跑，测主体高速运动 | 运动 | 旧 cafe 几乎是坐姿闲聊；要看运动模糊、肢体、水花是否碎 |
| `p02_talking_head` | 演播室女主播特写说话，测脸/口型 | 人体/脸 | 768p 才谈得上五官；口型跟有声轨 |
| `p03_corridor` | 博物馆长廊手持前进，测空间纵深 | 空间 | 透视、遮挡、远近尺度；480p 小脸评不了这个 |
| `p04_volumetric` | 木屋金色体积光，尘埃与灯亮起 | 光照 | 体积光/阴影时间变化，不靠人脸凑分数 |
| `p05_neon_sign` | 东京夜店招牌「OPEN 24 / 深夜食堂」+ 字幕 | 文字/字幕 | H3 弱项；可读性单独记账，不并进运动分 |
| `p06_orbit` | 铜马雕像 180° 环绕，测镜头运动 | 镜头 | 视差、主体不变形、背景不拖瓦 |
| `p07_market_track` | 湿货市场 steadicam 穿过人群 | 运动+脸+空间 | 多身份、短暂入画出画，测身份稳定 |
| `p08_ballroom_dolly` | 双人舞由近拉到大厅全景 | 镜头+人体+光照 | 焦距/构图大变时脸是否换人 |

冒烟最低集：先跑 **`p01_sprint` 的 5s 金线**（有时间再加 turbo4）。全量 8 条是专业评测，不是每次优化都要 8×全臂。

**不要再写「cafe 一条过了所以 768P 也行」。**

---

## 5. 质量门（人眼主、数字辅）

专业评测 **以人眼 + 成片为主**。抽帧 MAE 只作回归警报，**不是唯一过门**。

### 5.1 人眼清单（每条成片必勾）

1. 无花屏 / 色块 / 错轴 / 棋盘。
2. 主体可辨；`p02`/`p07`/`p08` 五官不融化。
3. 运动方向与提示一致；无瞬移、无多余肢体。
4. `p05`：招牌/字幕有没有可辨字形（记「可读 / 糊 / 乱码」）。
5. 镜头运动（`p06`/`p08`）视差是否合理、主体是否橡皮。
6. 时间轴：ffprobe **5.000s ±0.05** / **120** 视频帧 @24fps。
7. 有声：不是静音 AAC pad；对白或环境能听出来。

### 5.2 抽帧 MAE（回归）

抽 f0 / 24 / 48 / 90 / 119，对 **同一 prompt+seed 的金线 768P 片**。记录五帧均值与区间。  
**不设「MAE&lt;X 即过」的 768P 新阈值**（768P 金线片子本仓此前没有）。turbo/TAE/lift **预期大幅偏离**，只记账。

### 5.3 音频 peak

读 generate 日志 `audio_pcm` peak。记饱和（=1.0）或过小（&lt;0.01）。饱和不是自动 FAIL，但要写进报告。无 PCM / 无音轨 = 该臂失败。

---

## 6. 对照臂

全部 **1344×768 · 5s · 新 prompt · 同 seed**。Ulysses 不是主臂。

| 臂 | 配方 | 金线？ | 脚本 `--arm` |
| --- | --- | --- | --- |
| **金线 768P 8step** | 无 `--preset`，`--steps 8 --cfg 1 --sdpa-backend cudnn`，官方 VAE，无 LoRA，`ulysses=1` | **是** | `gold` |
| turbo4 768P | `--preset turbo4 --lora …`（larryvrh v4 EMA） | 否 | `turbo4` |
| turbo4 + TE cache | 同上 + `--te-cache DIR`（HIT 才报加速） | 否 | `turbo4_te` |
| turbo4 + TAE | `--vae-decode tae --tae FILE` | 否 | `turbo4_tae` |
| self-lift | 见 §7；**当前默认 BLOCKED** | 否 | `self-lift` |
| U=2（可选） | 金线几何 + `--ulysses 2` | 否 | `ulysses2` |

**TAE 与 768P latent：** `h3_tae` 合同是官方 unpack `[24,T,H,W]`、空间 16×，**没有写死 54×30**。1344×768 → latent 84×48，几何对得上。仍须 `--tae` 权在、且仅 turbo/nvfp4。未跑过 768P TAE = 该臂写「未测」，不要写成已接好。

成片命名：`out/review_videos_768p/<arm>_<prompt_id>_5s.mp4`。原始日志：`out/verify_768p/<arm>/<prompt_id>/`。

---

## 7. Self-lift / 双采样（联网对照 + 本仓缺口）

社区里至少三件被混叫「self-lift / 二采」的事，**必须拆开**。

### 7.1 SelfLift 论文（arXiv:2609.02036，2026-08）

[SelfLift: Accelerating Few-Step Diffusion via Self-Recovering Resolution Transition](https://arxiv.org/pdf/2609.02036v1)（清华 / 字节）。

它**不是**「先出一条 480p 成片再 img2img 出 768p」，也**不是**蒸馏 teacher 在线引导整段轨迹。

- 同一套少步 denoiser 上做 **渐进分辨率**：前缀低分 denoising，在过渡步 \(t_r\) **抬到目标分辨率**，后缀继续高分。
- **SelfLift-zero**（免训）：Artifact-Aware Consistency Lift。一边把中间 latent **直接上采样**，一边 **像素域 VAE 编解码锚**；两者差当作伪影风险与修正方向。不引入外部超分、不加额外 denoiser 步、不改采样日程。
- **SelfLift-rich**：把 zero 过渡蒸成轻量 latent lifter \(G_\phi\)，再用 LoRA 学生做 On-Policy Self Recovery；监督来自 **同一学生轨迹上的内部 self-teacher**，不是另训的 H3 teacher。
- 公开数字在 **FLUX.2-Klein / Z-Image-Turbo（图）**：零方案约 −41.5% / −44.1% e2e；叠加步蒸馏才有 29×/19× 叙事。**不是 MiniMax-H3 视频、不是 5090、不是本仓 8-step INT8。**

### 7.2 社区 H3 / LTX「低分草稿再 refine」

- Comfy **MiniMax H3 Latent Upscaler 3D**：低分出运动/声音 → **学到的 3D latent ×2** → 短程低 σ refine（常见 turbo 切成 3+5）。跳过 VAE 像素往返。本仓 **没有** 这份 upscaler 权，也没有 refine 节点。
- LTX-2 / Sol Super Accel：Stage1 低分 H3 + Stage2 **LTX latent upsample + 3 update**。本仓 **无 LTX**，`cute_sm100` 不能落 5090。见 [two-stage-768p](./2026-09-11-h3c-two-stage-768p.md)。

### 7.3 本仓 C 栈怎么接（诚实）

| 已有 | 文件 | 能不能当 SelfLift |
| --- | --- | --- |
| 任意画布 T2VA | `tools/h3_cuda_generate.c` `--width/--height` | 能跑 **原生 768P**（金线臂） |
| 结束时 `latent.raw` | 同上，`--out/latent.raw` | 是 **收尾 clean latent**，不是过渡步的 noisy 态 |
| `--init-latent` / `--init-first-frame` / `--lock-first-frame` | 同上 | **Debug G3-lite**，官方禁止当产品二采 |
| `--first-frame` / `--last-frame` FL2VA | 同上 | 只锁端点；中间仍是 **完整高分 Euler** |
| TAE 16× decode | `h3_tae.c` / `h3_tae_cuda.cu` | 成片刀，不是 lift |
| `--preset gold\|turbo4\|nvfp4` | `h3_generate_preset.c` | 金线/加速档组合器 |

**`--preset lift`：已登记为硬拒绝。** 不要假装有二采。错误文案指向本文件。没有学到的 ×2 upsample，Euler 循环也 **没有** mid-sigma 换分辨率续采。

要真接 SelfLift-zero，至少还要：

1. 在 `h3_cuda_generate.c` 的 step 循环插入 \(t_r\)：停低分、构造高分态、改 packed 形状、重建 cuDNN plan。
2. 低分 latent → 目标格子的 **直接 lift** + **官方 VAE decode→像素放大→encode** 残差（SelfLift-zero）。像素锚要官方 VAE，不能偷偷换 TAE。
3. 从过渡 σ **续跑** 剩余步（不是从 σ=1 再噪一遍）。
4. 音频：跟论文（图模）不同，H3 packed 有 audio token。默认 **草稿 PCM mux**（Sol 同风险：口型在低分脸）。
5. 只进加速档；`h3_generate_preset_apply("gold")` 永不走 lift。

### 7.4 预期：打不打得过原生 768P 8step

对照分母 = **原生 1344×768 · 8-step INT8+cudnn**（本仓此前 **未测**，1×5090 外推 150–220s，见估时文）。

| 方案 | 预期墙钟 vs 原生 768P | 质量 | 本仓 |
| --- | --- | --- | --- |
| 原生 8step | 1×（新尺子） | 金线轨迹 | **能跑，待测** |
| SelfLift-zero（若接上） | 论文图模 −40% 量级 **不能抄到 H3**；视频还要付 768p VAE。乐观：多数步在 480p（DiT ~¼）+ 2–3 步 768p。e2e 仍被 TE+768p VAE 托住 | 少步下 lift 伪影是论文要修的东西；H3 8-step 预算更薄 | **未接** |
| 480 草稿 N + 768 FL2VA M=8 同权 | **更慢**（768p 8step DiT 几乎全付 + 多一次 VAE） | 端点可锁，中间不是超分 | 能打两枪，**禁止叫 SelfLift** |
| 480 N + 768 M≤4 且匹配学生/LoRA | 才可能快一截（估时文 8+2 约 0.55–0.8×，推断） | 快档 | 无高分少步学生则 **质量红线** |
| H3→LTX 4+3 | 禁止写 5090 秒数 | 非无损 | **不能跑** |

**结论：** 现在这条 C 栈，self-lift **对不上，不要假装做了。** 金线 768P 走原生 8step。加速档若有人要「低分预览」，用独立脚本打 FL2VA 负对照，且必须在报告写「proxy，不是 SelfLift」。预期 **打不过** 原生 768P，除非精修少步且权匹配。

---

## 8. 命令模板（评测脚本默认；不改 generate 缺省）

```sh
# 金线 768P（主尺子）。seed 按 §3。
export CUDA_VISIBLE_DEVICES=0
./build-cuda-sm120/h3_cuda_generate \
  --dit "$DIT" --vae "$VAE" --te "$TE" --audio-vae "$AUDIO" \
  --out "$OUT" --mux "$MUX" \
  --width 1344 --height 768 --seconds 5 --steps 8 --cfg 1 \
  --sdpa-backend cudnn --seed "$SEED" \
  --token-ids-file "$IDS"
```

封装：`tools/h3_768p_eval.sh --arm gold --prompt p01_sprint`。

---

## 9. 进度与历史册

- 新进度：`out/verify_768p/PROGRESS.md`（尺子已切）。
- 旧 480p REPORT 顶栏已标明「主尺子已切 768P，本报告历史」。
- 第一枪 5s 768P 秒数：见 PROGRESS；跑不完写 **未测**。
