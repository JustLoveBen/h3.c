# comfyui-SelfLift 技术备忘（供完整极致方案引用）

Date: 2026-09-13  
源码：`third_party/accel_survey/comfyui-SelfLift/`（只读浅克隆，**非 submodule**）  
钉死：[`facok/comfyui-SelfLift`](https://github.com/facok/comfyui-SelfLift) `master` `@19ec540505dcbc261aaecf450d83c7536be2826d`（2026-09-12，`Revert "Drop guide color mean correction"`）  
论文：[SelfLift](https://arxiv.org/abs/2609.02036)（图像 RF，2026-09-02）· [TST](https://arxiv.org/abs/2609.08505)  
外部权：[LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)

> **For agentic workers：** 独立机制备忘，不是施工单。禁止改推理代码、禁止改金线默认、禁止 commit、禁止把 Comfy 节点/图嵌进生产路径。`--preset lift` 保持硬拒绝，直到 C 栈自己接线。  
> **对照：** [768P 评测协议](./2026-09-13-h3c-768p-eval-protocol.md) §7 · [二采估时](./2026-09-11-h3c-two-stage-768p.md) · [少步质量](./2026-09-11-h3-few-step-quality.md)

**一句话：** SelfLift 是「同一条 N 步 Euler 日程、前缀低分、过渡步 lift、后缀高分续跑」；**去噪 NFE 在非 tiling 路径上确实不变**，但 H3 还要付一次 VAE 往返或外部 3D upscaler。本仓缺中途改 packed seq、空间 upsample、VAE 锚三条原语；`--init-latent` 不是 mid-σ 续采。5s · 1344×768 的 6+2 **打不过「稳赢原生 8 步」的叙事**——DiT 能省，768P VAE 往返会把账吃回去。

---

## 0. 插件是什么 / 不是什么

`comfyui-SelfLift` 是 ComfyUI 自定义节点包（`selflift` 分类），不是 H3 官方 serving，也不是本仓 C 栈。三个节点：

| 节点 | 文件 | 作用 |
| --- | --- | --- |
| `SelfLiftH3Sampler` | `nodes.py` | H3 音视频渐进分辨率采样（论文未验证） |
| `SelfLiftImageSampler` | `nodes.py` | 论文 SelfLift-zero（FLUX / Z-Image 一类 4D 图像 latent） |
| `SelfLiftH3TST` | `h3_tst.py` | 时间注意力校正补丁，可挂任意标准采样器 |

共享引擎是 `nodes.py:progressive_sample`。lift 数学在 `selflift.py`。学到的 3D latent 提升在 `h3_upscaler.py`（抄 LBH 架构，**不是**论文 SelfLift-rich）。`h3_tiling.py` 是高分空间切块，实验性。`diagnostics.py` 只打内存日志。

**硬限制（README + 代码一致）：**

- 只要标准 Euler，`s_churn=0`；其它采样器直接 `ValueError`。
- 模型必须是 Comfy `CONST`（rectified flow）。
- SelfLift-rich（蒸馏 lifter + On-Policy Self Recovery）**不在插件里**。
- H3 路径是「论文图像算法的实验性外推」，README 自己写 *not paper-validated*。

**生产口径：** 只读算法与权来源。不要 `import` 这个包，不要把 workflow JSON 当 SKU。

---

## 1. 逐步算法：低分 / transition / 高分；NFE 是否真不变

### 1.1 日程怎么切（8 步、`transition_step=6`）

Comfy `sigmas` 长度 = `steps+1`，末项为 0。`transition_step=k` 表示 **前 k 次去噪评估在低分**。

以 8 步为例（`sigmas[0..8]`）：

| 阶段 | 调用 | sigma 切片 | 去噪评估 | 分辨率 |
| --- | --- | --- | ---: | --- |
| 低分 | `sample(sigmas[:7])` | `σ0…σ6`（6 个区间） | **6** | latent `(T,h,w)`，默认 `lowres_scale=0.5` |
| 过渡 | **不用 denoiser** | 复用第 6 次评估的 `x0` | **0** | lift 到 `(T,H,W)`，再按 `σ5→σ6` 补完 Euler |
| 高分 | `sample(sigmas[6:])` | `σ6…σ8`（2 个区间） | **2** | 目标格子 |
| 合计 | | 原 8 步表 | **8** | |

`nodes.py` 写得很死：低分 callback 必须恰好 `transition_step` 次，高分必须恰好 `total_steps-transition_step` 次，否则 `RuntimeError`。最后一次低分评估被当成论文 Eq.3 的 clean endpoint；**它的 Euler 更新被丢掉**，过渡后再用同一对 `(σ_k, σ_{k+1})` 重做。

本仓对照：`tools/h3_cuda_generate.c` 的 `build_serving_sigmas` 同样是 `steps+1` 个 sigma、末项 0，视频 `shift=12`、音频 `shift=3`。Euler 合同是 `X += (σ-σ')·v`（`h3_dit_w4a8_euler_step_f32`），与 Comfy CONST「`denoised=X+σ·v` → `X += (σ-σ')·v`」同构。插件 `_euler_step` 用的是 x0 形式：

```
step = (σ_next - σ) / σ
X'   = X + (X - x0) * step
```

代数上等价于本仓 velocity Euler。差在本仓循环 **从第 0 步到第 N-1 步形状不变**，没有「丢掉最后一次低分 Euler、换格子再补」。

### 1.2 低分阶段

1. 目标视频 latent 下采样到偶数格子：`h = max(2, round(H·scale/2)·2)`（`nodes.py`）。  
   768P latent `84×48`、`scale=0.5` → **`42×24`** → 像素 **672×384**。
2. 噪声按 **低分形状** + 原 seed 采样。
3. H3 keyframe COND 用 bilinear 缩到低分格子，再做 **逐通道逐帧均值对齐**（防色漂；`_resize_keyframes`）。这就是 `@19ec540` 撤回「丢掉均值修正」要保住的东西。
4. 音频流 **没有空间维**，原样穿过低分采样。
5. 走完整 Euler k 步。最后一次 callback 抓住 `(state, x0)`。

### 1.3 过渡（Artifact-Aware Consistency Lift，`selflift.py`）

对低分 `x0`：

1. `latent_format.process_out` → VAE 原生空间（H3：24ch，含 mean/std）。
2. **直接 lift** `z_lat`（Eq.4）：
   - 图像：nearest / bilinear；
   - 视频默认 nearest；可选 bilinear→trilinear；
   - 或 `latent_lifter`（外部 3D upscaler）。
3. **像素锚** `z_pix`（Eq.5，`rho>0` 才做）：VAE decode 低分 → 像素上采样 → VAE encode 高分。  
   图像用 lanczos；视频用 **逐 32 帧 bicubic**（避免整段 fp32 像素在 CPU 爆内存）。
4. **选择性修正**（Eq.6–9）：`δ = z_pix - z_lat`，位置风险 `s = mean(|δ|)`，取 top-`ρ` 分位，强度从 `w_min` 线性插到 `w_max`：  
   `z0_high = z_lat + w·δ`。  
   短路：`ρ=0` → 只要 `z_lat`；`ρ=w_min=w_max=1` → 只要 `z_pix`。
5. `process_in` 回模型空间。
6. **用新噪声（seed+1）在 `σ_k` 上 re-noise**，再无 NFE 地走完 `σ_k→σ_{k+1}`。音频用低分丢掉的那次 Euler 结果（`auxiliary_next`），不重新评估。
7. `inverse_noise_scaling(σ_{k+1})` 得到高分 resume latent；高分 `sample` 的 noise 是 **全零**（态已经带对 σ）。

这不是「先出一条低分成片再 img2img」，也不是从 `σ=1` 重新加噪。

### 1.4 高分阶段

从 `σ_{k+1}` 跑完剩余步，形状锁在目标格子。可选 `model_hires`（另一份同架构 checkpoint / LoRA），低分始终用 `model`。

### 1.5 NFE：去噪次数不变；墙钟不是

**去噪 NFE（非 tiling、按 sampler step 计）：真的不变。** 过渡不调用 denoiser。日志里的 `low_nfe` / `high_nfe` 加起来等于 `len(sigmas)-1`。

必须同时写清的例外：

| 项 | 是否额外 NFE | 是否额外墙钟 |
| --- | --- | --- |
| VAE decode→upsample→encode | 否 | **是**（`ρ>0`）。H3 视频 VAE 很重 |
| 外部 3D upscaler | 否（不是 DiT） | 是（一次 3D-conv forward） |
| `ρ=0` 且无 upscaler | H3 节点直接报错 | — |
| CFG>1 | 每步仍 2 次 cond/uncond，总数仍 `2N` | 与原生相同 |
| `highres_tiling` | **高分每步 × tile 数** | 显存换 NFE；NFE 不变被打破 |
| TST | 否 | 约 1–2%（注意力后处理） |
| `SELFLIFT_DEBUG` 中途 decode PNG | 否 | 极大 |

插件 H3 节点默认 `cfg=5.0`，本仓金线 768P 是 `--cfg 1`。比 NFE 时必须先对齐 CFG，否则 2× 会被误读成 SelfLift 开销。

**结论：** 「N 步仍是 N 次 DiT」在 **单段、无 tiling** 成立。把这句话抄成「e2e 也免费」是错的。

---

## 2. H3 路径：外部 upscaler vs SelfLift-zero，权从哪下

两条互斥主路（可杂交，插件会警告）：

| 模式 | 开关 | 是不是论文 SelfLift-zero | 权 |
| --- | --- | --- | --- |
| **外部 upscaler（H3 节点默认）** | `upscaler_model≠none` 且 `ρ=0` | **不是**。学习式纯 latent 提升 | HF `LBH-123-AI/Minimax_h3_latent_Upscaler` → `ComfyUI/models/latent_upscale_models/`；节点选文件名含 `h3` 的第一份 |
| **SelfLift-zero** | `upscaler_model=none` 且 `ρ>0` | 是（算法），但 **论文没在 H3 上验证** | 无新权；像素锚必须是 **当前采样模型所属官方 VAE** |
| 杂交 | upscaler + `ρ>0` | 实验；README 不推荐当主对照 | 两套都要 |
| 非法 | `none` + `ρ=0` | — | `SelfLiftH3Sampler` 直接拒绝 |

H3 节点默认 `ρ=0.0`：装了 upscaler 就走学习式 lift，**跳过 VAE 往返**。这是实用默认，也是和论文叙事最容易混的一点。

SelfLift-zero 建议起点（README「诊断 H3」）：`ρ=0.6`、`w_min=w_max=1`。图像论文默认是 `ρ=0.3`、`w=0.5/1.0`。插件认为 H3 nearest 误差比图像骨干更广，要更狠地往像素锚扳。

### 2.1 外部权长什么样

`h3_upscaler.py:LatentResizer3D`：

- 输入 `[B,24,T,h,w]`，输出同 T、目标 `H×W`。
- 24→512 Conv3d，12+12 `ResBlockEmb3D`，穿插 `TemporalConv`（默认 every 2，k=5）；中间 **trilinear** 拉空间。
- 推理前按 Comfy `LATENTS_MEAN/STD` 标准化，出来再还原——与本仓 `latents_mean/std`（`h3_vae_int8_load_latent_stats`）是同一套官方统计。
- 架构声明借鉴 LTX 2.3 spatial upscaler；**明确不是**论文 SelfLift-rich 的 \(G_φ\)。
- HF 三份同结构：`*_bf16.safetensors` / `*_fp16.safetensors` ~691 MB，`*_fp32.pth` ~1.38 GB。训练约 8 万对（约 7 万视频 + 8k 2K 图），2× 占 40%。
- `T>32` 时按 overlap 切时间窗再加权拼。

本仓 **没有** 这份权，也没有 3D resizer 实现。不要把 TAE 的 `k_upsample2`（解码器内部 2×）当成这份 lifter。

### 2.2 和「本仓二采」必须拆开

社区至少三件被混叫 self-lift，768P 协议 §7 已拆。对照本插件再钉一次：

| 名字 | 实际在做什么 | 本仓 |
| --- | --- | --- |
| SelfLift-zero | 同模型、同 N 步、mid-σ 换分辨率 + VAE 锚 | **未接** |
| LBH 3D upscaler | 学到的 latent ×2，常配「低分 N + 高分 M」再采样 | **无权** |
| 480p 成片 + FL2VA 8 步 | 端点条件再生成，中间仍是完整高分 Euler | 能打两枪，**禁止叫 SelfLift** |
| Sol H3→LTX | 换模型 3 update | 无 LTX，5090 落不了 `cute_sm100` |

`--preset lift` / `self-lift` / `self_lift`：`h3_generate_preset.c` 硬拒绝，测试在 `tests/test_h3_generate_preset.c`。文案指向 768P 协议。保持如此。

---

## 3. 和本仓 C 栈差哪些原语

本仓 Euler 是一条形状锁死的循环：`video_seq = T·(H/2)·(W/2)` 在 `h3_cuda_generate.c` 里算一次，RoPE / cuDNN SDPA plan / Ulysses grid 都按这个 seq 建。中间没有换格子。

| 原语 | SelfLift（Comfy） | 本仓现状 | 缺口 |
| --- | --- | --- | --- |
| **中途改 packed seq** | 低分 `sample` 与高分 `sample` 两次；第二次 `video_seq` 变成 4× | 一次 `h3_cuda_sdpa_cudnn_prebuild`，`pt/ph/pw`、`n_lat`、Ulysses `grid` 绑死 | 要在 `σ_k` 停、重算 `video_seq`、重建 RoPE、重建 cuDNN plan、重绑 Ulysses。金线 U=1 也要重建 plan |
| **空间 upsample** | `F.interpolate` nearest/trilinear，或 3D lifter | generate **没有** latent 空间插值。TAE `k_upsample2` 只服务 TAE 解码器 | 要在 unpack `[24,T,H,W]` 上做 2×（再 `h3_dit_patchify_video` 打回 `[tokens,96]`） |
| **VAE 锚** | 官方 VAE decode 低分 → 像素放大 → encode 高分 | decode：`h3_vae_int8_*` / 官方 fp16 tile；encode：`h3_vae_encode_f32`（条件图 / skip-dit） | **零件在，没焊进 Euler 中途**。金线成片 decode 在循环 **之后**。TAE 16× 是快档 decode，**不能当锚**（合同：仅 turbo/nvfp4，有损） |
| mid-σ 续采 | `noise_scaling(σ_k)` + 无 NFE Euler + `inverse_noise_scaling` | `--init-latent`：整段 f32 `[T,H/2,W/2,96]` **替换初值**，日程仍从 `σ0` 走满 N 步 | **不是**过渡续跑。help 已标 Debug G3-lite。`--init-first-frame` 是 t=0 img2img mix，更远 |
| 双时钟音频 | 低分 x0 对音频做一次被丢弃的 Euler，高分带着走 | `audio_sigmas` shift=3 与视频并行，形状从不换 | 过渡时音频 **不要** 当空间 lift；按插件 `auxiliary_next` 做即可 |
| x0 提取 | Comfy callback 直接给 denoised | 本仓存的是 `velocity`；`x0 = X + σ·v` | 一行公式，但循环里现在没这个张量 |
| keyframe 缩格子 | bilinear + 均值对齐 | FL2VA/L2VA 按 **目标** 画布 encode | 低分阶段若仍喂 768P COND，格子不对 |

768P 几何（代码钉死，不猜）：

| 量 | 原生 768P | SelfLift 低分 `scale=0.5` |
| --- | ---: | ---: |
| 像素 | **1344×768**（`h3_adapt_canvas(1920,1080)`） | **672×384** |
| VAE 16× latent | 84×48 | 42×24 |
| 空间 token / 帧 | 42×24 = 1008 | 21×12 = 252 |
| `video_seq`（5s，`latent_t=37`） | **37 296** | **9 324**（恰好 1/4） |
| 对比旧 480p | 14 985 | 低分比 480p 还小（9 324 / 14 985 ≈ 0.62） |

`h3_adapt_canvas` **不会**产出 672×384 作为 T2VA 桶——它把请求都抬到短边≈768。672×384 只出现在 `h3_reference_image_canvas(1920,1080 → 512×512 面积)` 测试金线。CLI `--width 672 --height 384` **能跑**（32 对齐），但是手工画布，不是官方桶。

`--preset lift` 现况：硬拒绝。TAE：`h3_tae.h` 合同官方 unpack `[24,T,H,W]`、空间 16×，1344×768 → 84×48 几何对；仍只许 turbo/nvfp4 成片，且 **未测 768P TAE**。

---

## 4. 5s · 1344×768 回代：6@672×384 + 2@1344×768 + VAE 往返 vs 8 步原生

分母：原生 **1344×768 · 8-step · INT8+cudnn · cfg=1 · 官方 VAE**（本仓此前未测 768P e2e；1×5090 外推 150–220 s，见二采文 / 768P 协议）。  
480p 锚（历史 cafe，只用来外推，禁止当 768P 尺子）：DiT ~31.4–31.7 s / VAE decode ~7.9–10.6 s / TE ~12–15 s。

768p/480p DiT 估时文放在 **~4×（3.5–5.0）**。672×384 相对 768P token **恰好 1/4**，相对 480p ≈0.62。

### 4.1 DiT（单卡，按步线性切，混二次项）

| 段 | 估时 |
| --- | ---: |
| 6 步 @672×384 | ≈ `0.62 × 31.6 × 6/8` ≈ **15 s**；若更偏 SDPA 二次，可落到 10–14 s |
| 2 步 @1344×768 | ≈ `(3.5…5.0)×31.6 × 2/8` ≈ **28–40 s** |
| SelfLift DiT 合计 | **约 43–55 s** |
| 原生 8 步 768P DiT | **约 110–158 s** |
| DiT 比 | **约 0.35–0.45×** |

这是 SelfLift 唯一真正便宜的地方：6/8 的评估落在 1/4 token 的格子上。

### 4.2 VAE / TE / 其它

| 项 | 原生 8 步 | SelfLift-zero（`ρ>0`） | 外部 upscaler（`ρ=0`） |
| --- | --- | --- | --- |
| TE | 1×（12–15 s） | 1×（同进程） | 1× |
| 过渡 | 无 | **decode 672×384 ~6–8 s + encode 1344×768 ~20–35 s** | 3D resizer，量级秒级（未在本仓测；HF 卖点就是躲 VAE 往返） |
| 成片 decode | 768P **20–35 s** | 同样 768P **20–35 s** | 同样 |
| 额外相对原生 | 0 | **+27–43 s** | **+数秒** |

H3 视频 VAE（公开口径约 5B）和论文图像 VAE 不是一个量级。论文图模 −41% / −44% e2e **不能抄到本仓**。

### 4.3 e2e 粗区间（1×5090，外推，warm）

```
原生 8 步     ≈ TE 14 + DiT 134 + VAE 27 + mux 2     → 150–220 s（已有带宽）
6+2 + VAE锚  ≈ TE 14 + DiT 49 + 往返 35 + VAE 27 + 2 → 110–180 s（与原生带宽重叠）
6+2 + 学到的lift ≈ TE 14 + DiT 49 + lift ~5 + VAE 27 + 2 →  90–140 s（更像能赢，但是另一条算法）
```

**结论：**

1. **6+2 不是「稳赢原生 8 步」。** DiT 省下的 80 s 级，SelfLift-zero 会被 768P encode 吃掉一大半；单卡乐观也只是落到原生下沿。
2. 外部 upscaler 路径才比较像「e2e 真快」，但它 **不是** SelfLift-zero，要另下 ~691 MB 权，质量跟 LBH 学生走，不跟论文走。
3. 8 卡 Ulysses 只砍 DiT。往返 VAE 打在 Amdahl 地板上（二采文：TE+768P VAE 已 45–55 s）。多卡上 SelfLift-zero **更不香**。
4. 不要用「480p 8 步 + 768P FL2VA 8 步」冒充本组数字——那是再付一遍 768P DiT，更慢。
5. 成片仍要官方 VAE（金线）或 TAE（仅快档）。TAE 不能替代过渡锚。

---

## 5. 质量：README 单 seed 能不能信；384 短边 OOD

### 5.1 单 seed 表

README「诊断 H3」：同一 prompt/seed，`transition_step=6`、`lowres_scale=0.5`。

| 测试 | 插件观察 |
| --- | --- |
| 原生基线 | 干净 |
| 纯像素锚（`none`，`ρ=1`，`w=1/1`） | 干净 |
| nearest 直接 lift（`ρ=0`） | **大范围伪影** |
| 论文图像参数（`ρ=0.3`，`w=0.5/1`） | 伪影大多还在 |
| H3 强修正（`ρ=0.6`，`w=1/1`） | 主伪影去掉 |

**不能当产品证据。** 理由：

1. 单 seed、未公开 prompt、无多场景、无和本仓 768P 八条评测集对齐。
2. 作者自己写 *single-seed evidence only*。
3. 「干净」没有抽帧 MAE、没有口型/字幕/身份清单。
4. 强修正 `w=1` 等于大面积用 VAE 锚替换直接 lift——过度平滑是 README 自己标的反面风险。
5. 论文数字在 FLUX.2-Klein / Z-Image-Turbo **图模**；H3 是 5s 音视频、8 步预算更薄（少步质量文红线：不换权减步会崩）。

本仓若做对照，必须走 [768P 协议](./2026-09-13-h3c-768p-eval-protocol.md)：同 prompt、同 seed、人眼清单 + 抽帧记账。`self-lift` 臂当前 **BLOCKED**。

### 5.2 384 短边 OOD — 是的，而且比「可能」更硬

- 官方 16:9 768P 桶是 **1344×768**（短边 768）。产品 480p 是 **864×480**（短边 480）。
- `lowres_scale=0.5` 把 768P 打到 **672×384**（短边 **384**），比已测金线 480p 还矮 20%。
- `h3_adapt_canvas` 的 T2VA 桶从不落在 384：`adapt(864,480)` 反而升到 1376×768。384 只作为 **Ref2VA 参考图缩小** 出现。
- README 原文：*H3's 768-pixel short edge becomes 384 px at lowres_scale=0.5, which may sit outside the backbone's training distribution — validate per model.*
- 插件低分 token 是 252/帧，480p 是 405/帧。运动/脸/文字在 384 上先损一轮，再靠 2 步高分救——8 步预算里只剩 25% 高分步。

可选缓解（若以后做实验，不是现在接线）：低分改走 **864×480**（in-dist，但不再是 0.5×，也不是插件默认）；或 `lowres_scale≈0.625`。两者都要重测，不能偷论文 6/8 切分。

音频：口型在低分脸上生成，高分只续 2 步。Sol 同风险：草稿 PCM mux。插件把音频 Euler 在过渡处「补完」但不重新空间 lift。

---

## 6. TST 值不值得 port

`h3_tst.py` 把 [TST / Spectral Tension](https://arxiv.org/abs/2609.08505) 接到 H3 **单流 packed attention**（没有独立 temporal attn）：

- 视频段永远是 packed 最后一段（锚定 Comfy `PackedLayout`）。
- 对每头：把视频 Q/K 按帧空间均值池化 → `F×F` 传输算子 → `T = H_row - H_vN`。
- `γ = exp(τ_eff · T)` 只缩放 **视频行 query**。深层、早步权重大（余弦日程）。
- 注入点：Comfy `optimized_attention_override` + `WrappersMP.DIFFUSION_MODEL`。
- 默认 `τ=0.2`；`0.5` 过强。声称 ~1–2% 运行时。
- 与 `highres_tiling` **不兼容**（wrapper 顺序导致看不到 tile 的 video 行）。
- `SELFLIFT_TST_EXACT=1` 才算精确算子；README 称真实 H3 上逐头符号一致率 89–100%（插件自报，未独立复现）。

**不值得进 P0 / 不值得为极致墙钟 port。**

| 维度 | 判断 |
| --- | --- |
| 加速 | 不省 NFE，还加 1–2% + 诊断同步。极致方案不靠它 |
| 质量 | 针对闪烁、字幕、身份漂移，和 lift 伪影正交；本仓 768P 人眼清单里有用，但是 **第二优先级** |
| 工程 | 本仓主力是 cuDNN fused SDPA，没有 Comfy `wrap_attn`。要在 RoPE 后、SDPA 前改 per-head Q scale，并自己算 `F×F` 张力。和 tiling / Ulysses 切头会打架 |
| 证据 | 论文主实验不是 MiniMax-H3 serving；插件是实验移植 |

若以后做质量侧车：独立补丁、默关、禁止进 gold。不要和 SelfLift sampler 绑死。

---

## 7. 最小 port 清单（P0，文件级）

目标若是 **C 栈 SelfLift-zero 实验臂**（非金线、非 Comfy）：只搬算法，不搬节点。`highres_tiling` / TST / `nodes.py` / `diagnostics.py` **不要 port**。

### P0（没有这些就接不上 mid-σ 换分辨率）

| 文件 | 做什么 |
| --- | --- |
| `tools/h3_cuda_generate.c` | 把 Euler 循环切成低分 k 步 / 过渡 / 高分剩余步。过渡处提 `x0=X+σv`、换 `latent_h/w`、`video_seq`、重分配 `latent/velocity/RoPE`。高分从 `σ_{k+1}` 续，**禁止**走 `--init-latent` 从 `σ0` 重开 |
| **新建** `h3_selflift.c` + `h3_selflift.h` | 从 `selflift.py` 搬：`paired_lifts`（nearest/trilinear）+ `artifact_aware_consistency_lift`（分位 + `w`）。纯 C/CUDA，零 Comfy |
| **新建** 空间 upsample 核（可放 `h3_selflift.cu` 或挂现有 cuda 文件） | unpack `[24,T,h,w]` → 2× → pack。复用 `h3_vae_int8_unpack_dit_latent_f32` + `h3_dit_patchify_video`，不要借用 TAE `k_upsample2` |
| `h3_vae_int8*.c` / `h3_vae_encode_f32.c`（只接线，不改金线默认） | 过渡：官方 decode 低分 + encode 高分。`latents_mean/std` 与 DiT 空间对齐。**禁止 TAE 当锚** |
| `h3_dit_w4a8.c` + `h3_gpu_cuda.cu` | 换 seq 后重建 `h3_dit_w4a8_fill_metal_video_pos`、cuDNN SDPA plan（`h3_cuda_sdpa_cudnn_prebuild`）、Ulysses grid。低分/高分两套 plan |
| 音频路径（仍在 `h3_cuda_generate.c`） | 过渡对 `audio_rows` 做一次无 NFE 的 `h3_dit_w4a8_euler_step_f32`（对齐 `auxiliary_next`）。不要空间 lift 音频 |
| `tests/test_h3_selflift.c`（新建） | lift 数学（ρ=0 / ρ=1 / 分位）+ 672×384→84×48 形状 + pack/unpack 往返 |

### 明确保持不动（P0 不要碰）

| 文件 | 原因 |
| --- | --- |
| `h3_generate_preset.c` / `.h` | `lift` **继续硬拒绝**，直到上面接上并有 768P 人眼对照 |
| `h3_tae.c` / `h3_tae_cuda.cu` | 成片刀，不是 lift |
| 金线默认 steps / sdpa / VAE | 768P 协议：原生 8 步仍是尺子 |

### 外部 upscaler 是另一条 P0（不要和 zero 焊死）

| 文件 | 做什么 |
| --- | --- |
| **新建** `h3_latent_upscaler.c/.h` + `.cu` | 搬 `LatentResizer3D` 推理（`h3_upscaler.py` 约 350 行有效）。权从 HF 下载到 `--latent-upscaler FILE`，**不入库** |
| 加载器 | 认 `upscaler.` 前缀、bf16/fp16/fp32；推理用官方 `LATENTS_MEAN/STD` |

LBH 原 Comfy 节点仓（`LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler`）同样只读对照，不进生产。

### 不要 port

- `h3_tiling.py`：无跨块注意力、只留首块音频、打破 NFE 不变、和 TST/Ulysses 打架。
- `h3_tst.py` / `SelfLiftH3TST`：见 §6。
- `nodes.py` / `__init__.py` / Comfy `SAMPLER`/`SIGMAS` 包装。
- SelfLift-rich 训练。

### 验收（若将来开工，不是现在）

1. 去噪评估次数 = `--steps`（cfg=1 时；对拍日志 `dit_schedule: evals=`）。
2. 同 seed 原生 768P vs 6+2 zero vs 6+2 外部 lift，走 768P 协议人眼清单。
3. `gold` preset 路径零分叉。
4. 384 OOD 必须在报告里单独成节，不能藏进「论文 −40%」。

---

## 8. 给完整极致方案的引用句

可直接粘：

- SelfLift **不是** 480p 成片再 FL2VA，也不是 Sol H3→LTX。
- 去噪 NFE 在非 tiling 路径上不变；H3 e2e 仍可能被 **768P VAE encode** 拖回原生带宽。
- 本仓缺三条原语：中途改 seq、空间 upsample、VAE 锚。`--init-latent` 顶不了。
- `--preset lift` 继续拒绝。不要把 Comfy 图嵌进 5090 施工路径。
- 若要赌墙钟，优先评估 **外部 3D upscaler + 6+2**（另权）；SelfLift-zero 更像质量实验。
- TST 不进加速主臂。

相关已有文：768P 协议 §7（本仓缺口清单，本文把它补成源码级）· 二采文（8+8 / 8+2 为何不赢）· `third_party/accel_survey/README.md` 已登记本克隆。

---

## 9. 指针：完整极致施工图

施工图已存在：[2026-09-13-h3c-5090-extreme-full-plan.md](./2026-09-13-h3c-5090-extreme-full-plan.md)（另一路在写，**本文不改那篇**）。

引用关系：施工图把 SelfLift 列为 Phase 1B 加速档候选；**算法、NFE、原语缺口、384 OOD、VAE 往返账、P0 文件表以本文为准**。那篇若与本文冲突，以 `@19ec540` 源码 + 本文 §1–7 为准，不要把 Comfy 图写进施工路径。`--preset lift` 在 C 栈未接线前保持硬拒绝；施工图若开 `--preset selflift`，必须先满足本文 §7 验收，且 `gold` 永不走 lift。
