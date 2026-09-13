# 上游 antirez/h3.c 全部 open PR 评审（2026-09-13）

## 背景与方法

- 上游 https://github.com/antirez/h3.c 已停更，社区仍在提 PR。本次评审全部 **31 个 open PR**。
- 本地 `upstream` remote 已存在，通过 `git fetch upstream '+refs/pull/*/head:refs/upstream-pr/*'` 拉取全部 PR head，逐个看 diff；讨论区用 `gh pr view` 补充。
- 评估基准：我们的主战场是 **Linux/CUDA 8×RTX 5090（PCIe）** 推理加速（W4A8 DiT、sage3 attention、turbo4 四步 LoRA preset、Ulysses SP）；上游是 macOS/Metal 原版。
- **金线零污染纪律**：任何 cherry-pick 不动金线数值路径；正确性 bug 修复单独标注验证方法。
- 评级：P0 必须看/建议采纳；P1 值得（采纳或列为审计/验证项）；P2 可选参考；N/A 不适用。

## 总表（31 个 open PR）

| PR | 标题 | 一句话 | 评级 |
|---|---|---|---|
| [#64](https://github.com/antirez/h3.c/pull/64) | Split long non-causal SDPA queries | MPSGraph 在 M2/M3 Ultra 上 >~15k query 行静默损坏，按 2048 行分块规避 | P1（教训/测试项） |
| [#63](https://github.com/antirez/h3.c/pull/63) | Barrier before reusing GQA reduction scratch | Metal shader softmax max→sum 复用共享内存缺 barrier（WAR 竞态） | P1（审计项） |
| [#62](https://github.com/antirez/h3.c/pull/62) | Flush Done lines when stdout is a pipe | `setvbuf(stdout, _IOLBF)`，管道日志即时可见 | P2 |
| [#59](https://github.com/antirez/h3.c/pull/59) | Build against SDKs older than macOS 26 | macOS 构建兼容 | N/A |
| [#56](https://github.com/antirez/h3.c/pull/56) | Register the seven H3 control tokens | tokenizer.json 漏了 `<d>` 等 7 个控制 token（id 151669 起），BPE 会吞掉标记 | P1（验证项） |
| [#55](https://github.com/antirez/h3.c/pull/55) | H3_ATTENTION_CACHE streaming | 磁盘预量化 int8 权重缓存＋按块流式，24GB Mac 跑满 50 层 | P2（同构已有） |
| [#53](https://github.com/antirez/h3.c/pull/53) | docs: Community integrations | 文档 | N/A |
| [#48](https://github.com/antirez/h3.c/pull/48) | Reject overlapping safetensors data offsets | 加载器校验 tensor offsets 连续且覆盖全文件 | **P0** |
| [#45](https://github.com/antirez/h3.c/pull/45) | H3 Studio macOS GUI | Python/macOS GUI | N/A |
| [#44](https://github.com/antirez/h3.c/pull/44) | Align causal GQA threadgroup memory to 16 bytes | Metal API 校验层要求，CUDA 无此约束 | N/A |
| [#43](https://github.com/antirez/h3.c/pull/43) | CUDA backend (GB10) + web UI | 3.5k 行 CUDA kernels 远落后我们的栈；但附带 938 行纯 C tokenizer | P2（仅 tokenizer 部分） |
| [#40](https://github.com/antirez/h3.c/pull/40) | docs: fix grammar | 文档 | N/A |
| [#35](https://github.com/antirez/h3.c/pull/35) | Report progress while VAE decodes | 解码进度回调（UX） | P2 |
| [#34](https://github.com/antirez/h3.c/pull/34) | Preview the denoised estimate | 预览应解码 x̂0 = x + σ·v 而非含噪 x_t | P1（设计参考） |
| [#33](https://github.com/antirez/h3.c/pull/33) | Add H3_MODEL_DIR env | CLI 便利性＋日志模块 | N/A |
| [#32](https://github.com/antirez/h3.c/pull/32) | Embed Metal shaders | 单文件二进制（Metal） | N/A |
| [#31](https://github.com/antirez/h3.c/pull/31) | ffmpeg: tolerate trailing CSV separator | FFprobe CSV 输出对 iPhone .mov（side_data）多带一个 `x` 分隔符导致误判 | P1 |
| [#24](https://github.com/antirez/h3.c/pull/24) | feat/cuda I4 cuBLAS linear | 早期 CUDA 系列，被我们的栈完全超越 | N/A |
| [#23](https://github.com/antirez/h3.c/pull/23) | feat/cuda I3 elementwise/norm kernels | 同上 | N/A |
| [#22](https://github.com/antirez/h3.c/pull/22) | docs: model download | 文档 | N/A |
| [#21](https://github.com/antirez/h3.c/pull/21) | feat/cuda I2 tensor layer | 同上 CUDA 系列 | N/A |
| [#20](https://github.com/antirez/h3.c/pull/20) | feat/cuda I1 skeleton | 同上 CUDA 系列 | N/A |
| [#14](https://github.com/antirez/h3.c/pull/14) | Distilled 5-6 step folded Turbo LoRA + three fixes | 与我们 turbo4 直接同域：折叠式 LoRA 工具＋3 个独立修复＋关键质量情报 | **P0** |
| [#11](https://github.com/antirez/h3.c/pull/11) | Avoid overflow when aligning large frame counts | `h3_align_frame_count` int 溢出（UB），我们同款代码 | P1 |
| [#10](https://github.com/antirez/h3.c/pull/10) | Clarify presets and callback contracts | 文档/注释 | N/A |
| [#9](https://github.com/antirez/h3.c/pull/9) | Keep Video VAE RGB output finite | `if(v<0)/if(v>1)` 放行 NaN，改 `fminf(fmaxf())`；我们 3 处同款拷贝 | P1 |
| [#8](https://github.com/antirez/h3.c/pull/8) | docs: clarify M4 support | 文档 | N/A |
| [#7](https://github.com/antirez/h3.c/pull/7) | RES audio updates on the audio sigma schedule | CPU RES 采样器音频改用原生 audio sigma 网格（数值变化） | P2 |
| [#4](https://github.com/antirez/h3.c/pull/4) | Keep scaled queries in F32 for causal GQA | Metal shader 里 Q·scale 不再中途舍入回 BF16 | P2（审计项） |
| [#2](https://github.com/antirez/h3.c/pull/2) | Add resumable generation checkpoints | 断点续跑（CPU/Metal 侧，1.1k 行） | P2 |
| [#1](https://github.com/antirez/h3.c/pull/1) | Use 256-pixel VideoVAE tiles by default | 撤销自动 tile 搜索，回到 checkpoint 默认 256 | P2 |

## P0 详细分析

### #48 Reject overlapping safetensors data offsets —— 建议采纳

**改了什么**：`h3_st_read_header` 解析完 JSON 后，把所有 tensor 按 `data_begin` 排序，要求 offsets 严格连续（无重叠、无空洞）且恰好覆盖数据区；否则拒绝加载。附带单测（乱序合法文件＋重叠非法文件）。

**对我们的适用性**：我们的 `h3_safetensors.c` 与上游同源，只有 `"tensor data offsets exceed file"` 的越界检查（第 333 行附近），**没有重叠/空洞校验**。我们的暴露面比上游更大：`h3_lora.c` 会加载第三方（Civitai 等）LoRA safetensors，`h3_tae.c` 加载 TAE 权重，恶意或损坏文件通过重叠 offsets 可以让两个 tensor 别名同一段数据，静默产出错误权重。

**建议**：直接 cherry-pick（改动完全在解析/校验层，合法文件行为不变）。同时把 #14 里的另一个加载器修复一并带上（见下）。

**金线风险**：无。只拒绝非法文件，不改任何数值路径。
**验证**：单测构造重叠/乱序文件；跑一次金线 preset 确认输出帧 hash 不变、全部权重文件正常加载。

### #14 Distilled 5-6 step folded Turbo LoRA + three fixes —— 必须看（情报＋部分采纳）

与我们 turbo4 直接同域。拆成四部分评估：

**A. `tools/fold_turbo_lora.py`（folded LoRA，主体）**
把 larryvrh Turbo LoRA 离线折叠进 bf16 shard（逐字节 patch，header/顺序/对齐不变，folded vs composed 随机探针 parity check）。上游没有 LoRA runtime 所以只能折叠；**我们有 residual 挂载 runtime（`h3_lora.c`），不需要折叠**。取舍对照：

| | folded（#14） | residual 挂载（我们） |
|---|---|---|
| 运行时开销 | 零 | 每层一次低秩加算（可忽略/已在预算内） |
| 磁盘 | 多一份 ~18GB+ 折叠权重 | 无 |
| 灵活性 | scale 固定、切换要重新折叠 | 运行时开关/换 LoRA/调 scale |
| 与 W4A8 交互 | 折叠后需重新量化整个 checkpoint | residual 保持 f32/bf16 旁路，不动量化权重 |

结论：维持 residual 方案。可借鉴的点：它的 **parity check 思路**（folded @ x vs W@x + scale·B(Ax)，相对误差 <1e-4）可以搬到我们 `tests/test_h3_lora.c` 作为 LoRA 正确性回归。

**B. 关键质量情报（对 turbo4 直接相关）**
作者实测：**4 步有 motion smear（动作拖影），5 步是验证过的质量下限**；且蒸馏 schedule 消除了 `--reuse`/`--core-reuse` 依赖的冗余，不能与这些复用选项组合。这与我们 turbo4（4 步，63.69s）的质量评估直接相关——建议在我们的 turbo 质量验收里**专门加"快速动作"类 prompt 检查拖影**，并把 turbo5（5 步）作为质量兜底档位纳入对比（对应 2026-09-11-h3c-5090-turbo-implementation.md 的质量矩阵）。

**C. safetensors 数据区 8 字节对齐检查（three fixes 之一）—— 建议随 #48 一并采纳**
MLX 的 `save_safetensors` 等写出的 header 不做 8 字节 padding，数据区错位后 mmap 路径静默解码成垃圾（黑屏视频）。修复是加载时拒绝 `(8+header_size)%8 != 0` 的文件并给出可操作报错。我们的 CUDA 加载路径同样按 offset mmap/读取，且会吃第三方 LoRA/TAE 文件，同样的静默失败模式存在。纯防御性，采纳。

**D. 其余两个修复**：Metal shader 路径回退到可执行文件目录（N/A）；`H3_VAE_TILE_PIXELS` 上限 512→320（作者实测 >320 的 tile 在整帧出现网格/quilt 伪影）。我们的 `h3_video_vae.c` `configured_tile_pixels` 同款代码、env 上限也是 512——**如果有人手动设 >320 会踩同一伪影**。低成本改掉或至少记录。

**金线风险**：B 是情报不动代码；C/D 只影响非法文件/非默认 env 覆盖，不碰金线数值。

## P1 详细分析

### #9 VAE RGB 输出 NaN 放行 —— 建议采纳（正确性修复，单独标注）

上游 `unpack_frame_range` 的 clamp 写法 `if (value < 0) ... if (value > 1) ...` 对 NaN 两个条件都为假，NaN 直接进 RGB 输出；改为 `fminf(fmaxf(value, 0.0f), 1.0f)`（fmax/fmin 遇 NaN 返回非 NaN 操作数，落到 0）。

**我们有 3 处同款拷贝**：
- `h3_video_vae.c:626`（Metal 路径）
- `h3_vae_int8.c:1012`
- `h3_vae_int8_cuda.c:2174`（**我们 CUDA 金线的 VAE unpack**）

**金线风险标注**：此修复碰金线文件，但对一切有限值 `fminf(fmaxf())` 与原写法逐位等价，只有 latent 出现 NaN/Inf 时行为才不同（原来输出 NaN，现在输出 0）。属于"正确性 bug 修复"豁免类。
**验证**：金线 seed 跑一次，输出帧逐字节比对（有限值等价 ⇒ 必须完全一致）；另加一个非有限 latent 的单测（上游 PR 自带测试可参考改写到 `tests/test_h3_vae_encode_f32.c` 同级）。

### #11 帧数对齐 int 溢出 —— 建议采纳

`h3_align_frame_count` 对超大 `requested`（如 INT32_MAX）做 `value += 17 - remainder` 是有符号溢出 UB，且上游 `h3_valid_params` 的范围检查依赖该返回值。修复改用 int64 中间量，溢出返回 0，调用方拒绝。**我们 `h3_host.c:15-21` 是同款代码**。纯边界防御，正常帧数（≤362）路径数值不变。
**验证**：上游自带 `CHECK(h3_align_frame_count(INT32_MAX) == 0)` 单测；金线不受影响。

### #31 FFprobe 尾部分隔符容错 —— 建议采纳

FFmpeg ≥5.1 的 CSV writer 对带 display matrix side_data 的流（典型：iPhone .mov）会多输出一个分隔符，得到 `3840x2160x`，上游（和我们 `h3_ffmpeg.c:146-153` 同款的）严格校验直接报 "invalid visual size"。修复只放行恰好一个尾随 `x`。我们的参考视频/图像 ingest 走同一函数，用户喂 iPhone 素材即触发。不碰任何数值路径。
**验证**：拿一个 iPhone .mov 跑 `--ref-video`；或 mock ffprobe 输出单测。

### #63 GQA 归约 scratch 复用竞态 —— 不直接 cherry-pick，转为 CUDA 审计项

上游 Metal `h3_gqa_causal_bf16` 里 softmax 的 max 归约结果 `reductions[0]` 被读取后，同一块共享内存立刻被 sum 归约复用，缺一个 barrier（WAR hazard）。修复是加 `threadgroup_barrier`。

我们不用这个 shader，但**同类隐患模式**（共享内存 max→sum 两段归约之间：读 `red[0]` 之后、复用 `red[]` 之前少一个 `__syncthreads()`）在 CUDA 里同样成立。抽查了 `h3_gpu_cuda.cu` 的 `h3_cuda_quantize_act_per_row_kernel`（该处 pattern 安全：读 absmax 后有 `__syncthreads()` 且不复用 scratch），但文件里有约 40 处 `__syncthreads()`、多个 attention/softmax kernel（4388、5257、6406 行附近）没有逐一人工核。
**建议验证方法**：`compute-sanitizer --tool racecheck` 跑一遍我们的 attention/softmax kernel 短测（`tests/test_h3_multimodal_cuda.c`、sage3 plugin 的 parity 测试），一次性把这一类问题清掉。零代码改动，属于纯审计。

### #64 长序列 SDPA 静默损坏 —— bug 本体 N/A，教训转为测试项

Bug 是 MPSGraph `scaledDotProductAttention` 在 M2/M3 Ultra 上 >~15k query 行静默产出垃圾（Apple FB24605554），修复是非因果 query 按 2048 行分块。我们是 sage3/cuDNN，**不共享这个 bug**。

但要点在于：**厂商 attention 黑盒在 ~15k 行档位静默损坏，而我们的主力序列长度恰好是 15973 token**。上游用户的症状是"长视频输出静态/噪声"，没有任何报错。建议：给 sage3 与 cuDNN SDPA 各加一个 **恰好 15973 token（及我们 SDPA pad 截断后的实际长度）的 CPU 参考 parity 测试**，纳入常规回归——尤其我们正在做 SDPA pad 截断，改动前后各跑一次。另外它"分块阈值/块大小可用 env 调"的做法与我们 sage3 插件的可配置风格一致，无需照搬。

### #56 七个 H3 控制 token —— 转为验证项

MiniMax 把 `<d>` `</d>` `<|cutoff|>` `<|lyrics_start|>` `<|lyrics_end|>` `<|caption_start|>` `<|caption_end|>` 只声明在 `tokenizer_config.json` 的 `additional_special_tokens`，不在 `tokenizer.json`；参考 tokenizer 把它们追加在末尾（`<d>`=151669）。缺失时 BPE 会把标记吞成普通 pieces，对白/歌词/字幕标记永远到不了 DiT。

我们的 CUDA 管线用 `--token-ids-file/--token-ids` 外部注入 token（`tools/h3_cuda_generate.c`），tokenization 在管线外完成：
- 若外部用 HF `AutoTokenizer`（读 tokenizer_config.json）→ 天然正确，无需动作；
- 若外部工具只读 `tokenizer.json`，或金线 token-ids 曾用上游 Metal `h3_tokenizer.m` 生成 → **同 bug**，凡 prompt 含对白/歌词标记的用例 token 序列是错的。

**验证方法**：拿 `"<d>[English] Hello there.</d>"` 过一遍我们的 token-ids 生成链，比对 PR 单测里的期望序列 `{151669, 58, 22574, 60, 21927, 1052, 13, 151670}`。若一致则关闭此项；不一致则按 #56 的方式补注册（这会改变含标记 prompt 的 token 序列 → 数值会变，属正确性修复，需单独跑质量对比）。

### #34 预览应解码 x̂0 而非 x_t —— 设计参考（TAE 预览）

上游 CPU 采样器预览改为解码 `x̂0 = x_t + σ·v`（模型当前对成片的估计），而不是含噪样本 x_t——后者直到 σ 收敛前全是噪声，无法用于"是否取消"的判断。我们正在做 **TAE 轻量解码预览**，这条直接适用：TAE 喂的应当是 x̂0 估计而不是中间 latent。好消息是我们的 velocity-euler 采样器注释里已有 `denoised = X + σ·v` 的约定（`tools/h3_cuda_generate.c:3766`），拿 velocity buffer 组一份 x̂0 给 TAE 即可，无需读回额外张量。GPU 侧成本：一个 axpy。不碰金线（预览旁路）。

## P2 简评

- **#55 H3_ATTENTION_CACHE**：磁盘预量化 int8 权重缓存＋双缓冲按块流式。与我们的 `h3_w4a8_cache` 同构（我们已做且更激进），带宽受限 Mac 的方案对 8×5090 无增益。**值得借鉴的教训**：讨论区里 FL2VA 的 cache 被 Ref2VA 静默接受（两个 transformer 张量形状相同），导致质量劣化，作者后来在 cache 格式里加了 transformer 身份标识。建议检查我们 w4a8 cache 的键控是否包含模型/transformer 身份与量化参数版本，防止同类静默混用。
- **#7 RES 音频 sigma 网格**：CPU RES 采样器把音频从"video sigma 网格＋slope 补偿"改为原生 audio 网格（数值变化）。我们金线用 velocity-euler 且已正确使用 `sigmas.audio`（`h3_dit.c:2727`），不受影响；只有跑上游 CPU RES 路径做对照时才相关。**不得顺手 cherry-pick**——它改 CPU 路径数值。
- **#4 Q scale 保持 F32**：Metal causal GQA 不再把 Q·scale 中途舍回 BF16。我们的 CUDA GQA（文本编码器路径)是独立实现，建议顺手确认我们没有类似的中途 BF16 舍入；它附带的 GQA CPU 参考单测（`tests/test_gqa.c`, 194 行）写得不错，可作为我们 GQA 测试的参考模板。
- **#43 CUDA backend**：kernels（bf16 直算、cuBLAS linear、GB10 单卡）远落后我们的 W4A8+sage3+Ulysses 栈，全部 N/A。唯一有价值的是 938 行**纯 C tokenizer**（`h3_tokenizer.c`，依赖 ICU），若将来想去掉外部 token-ids 依赖可参考；注意它**不含 #56 的 7 个控制 token 修复**，采用时要叠加。#20/#21/#23/#24 是同一作者更早的增量系列，被 #43 与我们自己的栈双重超越，无可取细节。
- **#62 stdout 行缓冲**：一行 `setvbuf`。我们 `tools/h3_5090x8_sync.sh` 远程跑批时管道日志能即时刷出，低价值但零风险，可顺手。
- **#2 可恢复检查点**：denoise 中途落盘续跑（CPU 侧 1.1k 行）。对我们两阶段 768p 长任务概念上有意义，但实现绑定上游 CPU 采样器，且我们有 step cache 方向，不采纳，仅记录。
- **#35 VAE 解码进度回调 / #33 H3_MODEL_DIR / #1 VAE tile 默认值**：UX/便利性/与 #14-D 相关的 tile 讨论，均不影响我们。
- **#10/#8/#22/#40/#53 文档、#32 shader 内嵌、#59 旧 SDK、#45 GUI**：N/A。

## 建议的 cherry-pick 顺序与验证

原则：先纯防御（不可能动数值），再"有限值等价"的正确性修复，审计/验证项并行推进。全程不动 turbo4/金线 preset 定义。

1. **#48 ＋ #14-C**（safetensors overlap 校验＋8 字节对齐检查）：一个提交。验证＝上游两个单测改写＋金线 preset 跑通、输出帧 hash 与基线一致。
2. **#11**（帧数对齐溢出）：一个提交，带 `INT32_MAX→0` 单测。金线 frames 远小于阈值，hash 必然不变。
3. **#9**（NaN clamp，3 处：`h3_video_vae.c` / `h3_vae_int8.c` / `h3_vae_int8_cuda.c`）：**正确性修复标注**——有限值逐位等价；验证＝金线 seed 输出帧逐字节比对（必须完全一致）＋非有限 latent 单测。
4. **#31**（ffprobe 尾分隔符）＋ **#62**（setvbuf）：小修一批。验证＝iPhone .mov ref 冒烟。
5. **审计项（不改代码，先出结论）**：
   - #63 类竞态：`compute-sanitizer --tool racecheck` 过我们 attention/softmax/归约 kernel 测试集；
   - #64 类静默损坏：sage3 与 cuDNN SDPA 在 15973 token（及 pad 截断后长度）加 CPU parity 回归；
   - #56：验证 token-ids 生成链对 7 个控制 token 的输出（期望序列见上）；若失配则修复并单独做质量对比；
   - #4 类精度：确认我们 CUDA GQA 无 Q·scale 中途 BF16 舍入。
6. **情报落地（无 cherry-pick）**：
   - #14-B：turbo 质量验收加"快速动作拖影"用例；评估 turbo5 兜底档；
   - #34：TAE 预览喂 x̂0 = x + σ·v；
   - #55：检查 w4a8 cache 键控含 transformer 身份；
   - #14-A：LoRA parity check（folded vs composed 探针）加进 `tests/test_h3_lora.c`。

## 结论

31 个 open PR 中：**2 个 P0**（#48 采纳、#14 情报＋部分采纳）、**6 个 P1**（#9/#11/#31 采纳，#63/#64/#56 转审计/验证项，#34 设计参考）、8 个 P2、其余 N/A。上游 CUDA 系列（#20/21/23/24/43）对我们的 kernel 栈无增量价值，唯一例外是 #43 的纯 C tokenizer 可留作参考。所有建议采纳项均不触碰金线数值路径；唯一碰金线文件的 #9 属"有限值逐位等价"的 NaN 防御，验证方法为金线输出逐字节比对。
