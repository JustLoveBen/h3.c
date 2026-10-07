# h3.c 代码结构整洁度审计与整理方案（2026-09-30）

> 只读审计。数据来源：`git status --short -uall`、`git diff --stat`、`ls`、`wc -l`、`rg`、`CMakeLists.txt` / `Makefile` 目标清单，采集于 2026-09-30 04:10（UTC+8）。**本文不移动/删除/重命名任何文件，不改 C/CUDA/CMake，不 commit**；所有「方案」均为建议，结构变动一律等用户拍板。
>
> 快照漂移：审计进行中（04:21）工作区另有并行改动落地——`h3_sdpa.h` 新增 `h3_sdpa_sage3_require_hit()` 与 env `H3_SAGE3_REQUIRE_HIT`（默认硬失败，`=0` 回旧行为）。因此截稿时实际为 **34 个 `M`**、**235 个 env**；下文的 33 / 234 是 04:10 快照数字，未逐处改写。该 env 尚无文档，归入 §6「sage3 实验臂」。
>
> 与施工册的关系：[2026-09-30-h3c-5090-fine-opt.md](./2026-09-30-h3c-5090-fine-opt.md) 是当前主施工册（R1–R61），本文所有会碰 `tools/h3_cuda_generate.c` / `h3_gpu_cuda.cu` / `h3_vae_int8_cuda.c` / `h3_w4a8_cache.c` 的整理动作，都要按 §9 的 A/B/C 分档避开施工窗口。

## 总体评价（一句话）

仓库是「一条 CUDA 产品路径 + 一条 Metal 上游路径 + 十几条实验臂」全部平铺在根目录的单层结构：**100 个 `h3_*` 源/头文件、60.7k 行**在根目录；一个 5117 行的 `main()`（`tools/h3_cuda_generate.c`）承担了 CLI / 条件编排 / 去噪环 / 2-stage / 解码 / mux / 日志的全部职责；**234 个 `H3_*` env 开关中 163 个无任何文档**；33 个已修改 + 76 个未跟踪文件（8676 行新增）横跨至少 7 件独立工作却尚未分批提交。代码本身有清晰的模块前缀纪律（`h3_<模块>_<后端>`），所以问题主要是「没有目录层级 + 巨型文件 + 提交积压 + 文档无索引」，而不是命名混乱。

---

## 1. 顶层平铺：根目录源文件盘点

### 1.1 总量

| 项 | 数 |
| --- | --- |
| 根目录 `.c/.h/.cu/.cpp` | **103 个**（含 `main.c`、`linenoise.c/h`）→ `h3_*` 为 100 个 |
| 根目录 `.m/.metal` | 4 个（`h3_gpu.m` 4711、`h3_shaders.metal` 4332、`h3_tokenizer.m` 521、`h3_metal.m` 47 = 9611 行） |
| 根目录 C/CUDA 总行数 | **60,742**（不含 .m/.metal） |
| `tools/` | 40 个（C/CU/py/sh），其中 `h3_cuda_generate.c` 6553 行 |
| `tests/` | 37 个源（含 2 个残留 `test_h3.o/.d`） |
| 根目录残留产物（已 gitignore） | `*.o/*.d` 各 ~20 个、5 个 probe 二进制、`h3_real_video_encoder_test`、8 个 `tmp_*` 脚本 |
| 大目录（均 gitignore） | `models/` 39G、`out/` 17G（84 项）、`third_party/accel_survey/` 386M |

### 1.2 按模块分类（行数 = .c+.h(+.cu)）

**路径标记**：`P` 产品路径（09-16 方向档 / 09-30 施工册默认栈）· `E` 实验臂（opt-in、非默认）· `M` Metal 上游路径（Darwin 产品，本仓只读对照）· `?` 可能废弃/待定。

| 模块 | 文件 | 行 | 路径 | 备注 |
| --- | --- | --- | --- | --- |
| **公共/宿主** | `h3.c/h`、`h3_host.c/h`、`h3_internal.h`、`h3_safetensors.c/h`、`h3_weights.c/h`、`h3_w4a8.c/h`（safetensors W4A8 读头）、`h3_token_ids.c/h`、`h3_dit_schedule.c/h`、`h3_ffmpeg.c/h`、`h3_terminal.c/h`、`h3_cli.c/h`、`main.c`、`linenoise.c/h` | ~9,530 | P/M 共用 | `h3.c`/`h3_host.c`/`h3_cli.c`/`main.c`/`linenoise` 只进 Makefile（Metal CLI） |
| **TE（CUDA）** | `h3_nvfp4.c/h`（1714）、`h3_nvfp4_cuda.cu/h`（1863）、`h3_te_cache.c/h`、`h3_vision_cuda.c/h`、`h3_multimodal_cuda.c/h` | ~4,940 | P（NVFP4 TE + 提示词缓存）；vision/multimodal 为 Ref2VA 条件通道 | `H3_NVFP4_HOST_*` 系列为调试臂 |
| **DiT（CUDA）** | `h3_dit_w4a8.c/h`（**7396**）、`h3_dit_int8.c/h`、`h3_dit_nvfp4.c/h`（未跟踪）、`h3_w4a8_cache.c/h`（2151）、`h3_w4a8_dequant.c/h`（840）、`h3_step_cache.c/h`、`h3_lora.c/h` | ~12,200 | P：INT8 ConvRot 主线 + cache + LoRA；E：NVFP4 DiT（`--preset nvfp4` 非金线）、`h3_step_cache`（Tea/FB/Easy，未过 MAE 门）、token reduction（`dit_w4a8.c` 5000–6000 段，Metal 对齐臂） | 名字 `w4a8` 已成历史包袱：`h3_w4a8_cache` 实际承载 W4A8/INT8/NVFP4 三格式 |
| **attn / GPU 核** | `h3_gpu_cuda.cu/h`（**10,288**）、`h3_gpu_cuda_sdpa_cudnn.cpp`（1035）、`h3_gpu_cuda_audio.cu/h`、`h3_cuda_device_caps.c/h`、`h3_sdpa.h`、`h3_sdpa_registry.c` | ~12,560 | P：cuDNN SDPA、GEMM、DiT 逐元素核；E：Sol-Attn 嵌套环（6294–）、sol_cute / sage3 dlopen 插件（4500–5050）、dist_attn | 单文件 46 个 `__global__`、119 个 extern "C" |
| **VAE（CUDA）** | `h3_vae_int8.c/h`（1359）、`h3_vae_int8_cuda.c/h`（**3974**）、`h3_vae_encode_f32.c/h`（1138）、`h3_audio_vae_cuda.c/h`（1287） | ~7,760 | P：INT8 视频 VAE decode（官方 256/64 tile）、音频 VAE；`vae_encode_f32` 为 F2VA/L2VA 条件编码（host F32） | 09-16 Task1「VAE 常驻」正在此处施工（未提交 diff 1811 行） |
| **TAE** | `h3_tae.c/h`、`h3_tae_cuda.cu/h`、`h3_tae_internal.h` | ~1,495 | E（`--vae-decode tae` 仅 turbo opt-in 预览） | `h3_tae_decode_f32` CPU 路径无调用者 |
| **lift / 2-stage** | `h3_selflift.c/h`、`h3_latent_upscaler.c/.cu/.h/_internal.h`、`h3_taomate.c/h` | ~2,150 | P（09-16：默认加速档对齐 TaoMate/Comfy 2-stage）→ `--preset lift / taomate / taomate-lift` | 全部未跟踪 |
| **Ulysses** | `h3_ulysses.c/h`、`h3_ulysses_comm.cu/h`、`h3_ulysses_comm_stub.c` | ~810 | E（`H3_LINK_NCCL_IN_GENERATE` 默认 OFF，U=1 走 stub）；09-13 驱动档结论 U=8 救不回 | 保留为 8 卡吞吐实验 |
| **编排** | `h3_generate_preset.c/h`（229）、`tools/h3_cuda_generate.c`（**6553**） | ~6,780 | P | 见 §2 |
| **Metal 路径** | `h3_gpu.m`、`h3_metal.m/h`、`h3_shaders.metal`、`h3_tokenizer.m/h`、`h3_gpu.h`、`h3_dit.c/h`（3293）、`h3_text_encoder.c/h`、`h3_video_vae.c/h`、`h3_video_encoder.c/h`、`h3_audio_vae.c/h`、`h3_audio_vae_layout.h`、`h3_vision_encoder.c/h`、`h3_multimodal.c/h` | ~9,500 + 9,611（.m/.metal） | M | 全部 Makefile 构建，CMake 不碰；上游 antirez/h3.c 同步面，**不建议动** |

**可能废弃 / 待定（`?`）**：无整文件级废弃。文件级最接近的是 `h3_ulysses_comm_stub.c`（仅为 NCCL OFF 占位，合理）和 `tools/h3_p2p_bw_probe.cu`（未接 CMake，头注释指示手工 `nvcc`，属一次性带宽探针）。函数级死代码见 §5。

### 1.3 `tools/` 分类

| 类 | 文件 | 备注 |
| --- | --- | --- |
| 产品 CLI | `h3_cuda_generate.c` | 6553 行 |
| 评测/运维脚本 | `h3_768p_eval.sh`（602）、`h3_dp_generate.sh`、`h3_5090x8_sync.sh`、`h3_tokenize_prompt.py` | 全部未跟踪；配 `evals/768p/` |
| W4A8 里程碑探针（P4 step-1…M3，8 月） | `h3_w4a8_probe/_matmul/_blocks/_cuda/_int8_cuda/_device/_cache/_generate_probe.c`（8 个）、`h3_dit_w4a8_block/_layers/_forward/_sample/_audio_probe.c`（5 个） | 历史里程碑，仍在 CMake；INT8 主线后多数不再运行 |
| 格式/权重工具 | `h3_st_inventory.c`、`h3_dit_int8_probe.c`、`h3_dit_nvfp4_probe.c`、`h3_nvfp4_probe.c`、`h3_nvfp4_int8_sidecar.c`、`h3_vae_int8_probe.c` | `nvfp4*` 三个为 NVFP4 臂 |
| 音频探针 | `h3_audio_ops_probe.c`、`h3_audio_vae_cuda_probe.c`、`h3_audio_vae_cuda_encode_probe.c`、`h3_quality_gate_probe.c` | Au1–Au6 里程碑 |
| 诊断 | `h3_condition_probe.c`、`h3_t2av_timing_probe.c`（891）、`h3_cuda_info.c`、`h3_lbh_oracle.c` + `h3_lbh_oracle_ref.py`、`h3_p2p_bw_probe.cu` | `lbh_oracle` 实际是 latent_upscaler 的数值对照测试，却挂在 tools |
| SDPA 插件（Python embed） | `h3_sol_cute_embed.c/.py`、`h3_sol_cute_bench.py`、`h3_sage3_embed.c/.py`、`h3_sol_attn_bench.c` | 实验臂；`libh3_sol_cute.so` / `libh3_sage3.so` |

---

## 2. 巨型文件职责混合度

### 2.1 `tools/h3_cuda_generate.c`（6553 行）

硬数据：`main()` 从 1437 行到 6553 行 = **5117 行**；`goto cleanup` **191 处**；CLI 旗标 **68 个**；文件内 `getenv` **44 处**；`printf/fprintf` **463 行**；26 个 `#include "h3_*.h"`；`main()` 开头 ~240 行是 ~200 个局部变量声明 + memset。

main 之前的 56 个 static 函数已经按功能自然成簇，main 内部则按行号可以切出以下内聚块：

| 内聚块 | 位置（行） | 约行数 | 建议目标文件 | 依赖 |
| --- | --- | --- | --- | --- |
| 计时/dump/PPM/PGM/WAV/raw IO 助手 | 71–178、364–486、695–772、1326–1384 | ~400 | `tools/generate/h3_gen_io.c/h` | 无（纯 host） |
| 线程作业（sdpa 预建 / DiT host stage / VAE willneed / 音频解码） | 252–354 + join 逻辑 6130、6463 | ~150 | `tools/generate/h3_gen_jobs.c/h` | pthread、`h3_w4a8_cache` |
| 官方 keyframe / ref-image / ref-video / vision 条件行构建 | 487–647；main 2981–4270 | **~1,500** | `tools/generate/h3_gen_cond.c/h` | `h3_vae_encode_f32`、`h3_vision_cuda`、`h3_multimodal_cuda` |
| TaoMate 分块/历史 KV/重叠融合 | 789–1123；main 2409–2560、4910–5440 | ~900 | `tools/generate/h3_gen_taomate.c/h` | `h3_taomate`、`h3_dit_w4a8_clean_kv_*` |
| SelfLift / LBH 过渡 / hires 窗口显存 | 828–947、1124–1294；main 5268–5440、5770–5900 | ~600 | `tools/generate/h3_gen_lift.c/h` | `h3_selflift`、`h3_latent_upscaler` |
| Ulysses 胶水（bind/a2a 回调/多 rank open/free） | 1385–1436；main 2845–2965 | ~200 | `tools/generate/h3_gen_ulysses.c/h` | `h3_ulysses*` |
| CLI 解析 + `--help` 文本 + preset 应用 | main 1678–2200 | ~520 | `tools/generate/h3_gen_cli.c/h`（产出 `h3_gen_opts` 结构） | `h3_generate_preset` |
| 形状/日程/sigma/audio_t 推导 | 773–827；main 2384–2410、4279–4340 | ~150 | `h3_gen_shape.c` 或并入 cli | 无 |
| 权重打开（DiT INT8/W4A8/NVFP4、TE、VAE、TAE、LoRA、upscaler） | main 2680–2960 | ~300 | `h3_gen_open.c` | 各模块 open |
| TE 前向 + 提示词缓存 + pad-uncond | main 4340–4700 | ~360 | `h3_gen_te.c` | `h3_nvfp4`、`h3_te_cache` |
| `clip_denoise:` Euler 环（reuse / MagCache / step_cache / clean-KV / lift 切换） | 4831–5960 | **~1,130** | `h3_gen_denoise.c` | `h3_dit_w4a8_forward_*`、`h3_step_cache` |
| VAE / TAE / 音频 decode + WAV + PPM + mux + `phase_ms` | 5960–6455 | ~500 | `h3_gen_decode.c`（含 `phase_ms` 日志格式单点） | `h3_vae_int8_cuda`、`h3_tae`、`h3_audio_vae_cuda`、`h3_ffmpeg` |
| `--repeat` 在进程内多 clip 与资源保留策略 | 6036–6110、6455 | ~100 | 并入 decode 或独立 `h3_gen_repeat.c` | `h3_w4a8_cache_*`、`h3_vae_int8_cuda_release_cached` |
| `cleanup:` | 6460–6553 | ~90 | 变成 `h3_gen_ctx_free()` | 全部 |

拆分前提是把 ~200 个局部变量收成一个 `h3_gen_ctx` 结构体，191 个 `goto cleanup` 改为 `return gen_fail(ctx, ...)`。这是**行为零变化**的机械重构，但 diff 极大，且与 09-30 施工册 R 项（日志字段、repeat_core、phase_ms、VAE 常驻）都改同一文件——**必须排在施工窗口之间**（见 §9.B1）。

已知死代码：`vae_th` / `vae_th_ok`（1635、1639、6130–6132、6463）只有 `pthread_join`，全文无 `pthread_create(&vae_th…)`（rg 6 处命中均为声明/join）。

### 2.2 `h3_gpu_cuda.cu`（9707 行）

132 个 static + 119 个 extern "C"，46 个 `__global__`。文件内已有 10 个 `/* ---- */` 分节注释，可直接沿分节切：

| 段 | 行 | 内容 | 建议文件 |
| --- | --- | --- | --- |
| 上下文/流/scratch/nvrtc 预载/设备重绑 | 1–1500；5000–5500（stream/event/graph/quiet） | `h3_cuda_bind/free/ensure_*/pregrow_*/recreate_blas/mem_info` | `h3_cuda_ctx.cu` |
| GEMM（F32/F16/I8 cuBLASLt/Ex/kernel）+ conv3d + ncdhw 算法缓存 | 1500–2500 | `h3_cuda_gemm_*`、`h3_cuda_conv3d_*` | `h3_cuda_gemm.cu` |
| W4A8/INT8 权重对象、host register、Hadamard/ConvRot/FWHT | 2500–3600（3214 分节） | `h3_cuda_w4a8_weight_*`、`h3_cuda_convrot_act_f32_dd`、`h3_cuda_fwht_*` | `h3_cuda_weight.cu` |
| FP32 flash SDPA；FP16 flash/TC/window；GQA causal | 3937–4290；6700–7529 | `h3_cuda_sdpa_f32*`、`h3_cuda_sdpa_f16*`、`gqa_causal` | `h3_cuda_sdpa_core.cu` |
| 后端选择 `want_*` + sol_cute / sage3 dlopen 插件 + Sol profile | 4290–5050、6000–6300 | `h3_cuda_sdpa_want_*`、`h3_sol_cute_*`、`h3_sage3_*`、`h3_sp_*` | `h3_cuda_sdpa_backends.cu` |
| Sol-Attn 嵌套环 | 6294–6700 | `h3_cuda_sdpa_sol_dev` | `h3_cuda_sdpa_sol.cu`（实验臂） |
| DiT 逐元素核（AdaLN/gate/SwiGLU/QKV-RoPE）+ 设备指针版 | 7529–8535 | `h3_cuda_adaln/gate/swiglu/qkv_rope*` | `h3_cuda_dit_ops.cu` |
| Layer-wise KV（TaoMate clean-KV） | 8535–9013 | `h3_cuda_sdpa_bind_kv*`、`qkv_rope_sdpa_f16_dev_layout*` | `h3_cuda_layer_kv.cu` |
| 通用设备助手（rms/ln/silu/residual/cast/dequant）+ `vae_unpatch_rgb` | 9013–9336、9500+ | | `h3_cuda_elem_ops.cu` |
| Token reduction pool/expand | 9336–9500 | | `h3_cuda_token_reduction.cu`（实验臂） |

先例已有：`h3_gpu_cuda_audio.cu`、`h3_nvfp4_cuda.cu`、`h3_gpu_cuda_sdpa_cudnn.cpp` 都是从这里分出去的，编译单元并入 `h3_gpu_cuda` STATIC 库即可，不需要改任何调用方。

### 2.3 `h3_vae_int8_cuda.c`（3915 行）

| 段 | 行 | 内容 | 备注 |
| --- | --- | --- | --- |
| 权重加载/释放/hostreg/H2D/dequant/f16 cast | 65–117（前置声明）、630–667、3080–3915 | `h3_vae_int8_cuda_weights_load/free`、`release_cached`、`hostreg_release`、`dequant_linear_gpu`、`load_linear_f16_gpu`、`gemm_f16_via` | 前置声明与定义相隔 3000 行，说明文件需要重排 |
| 单 block 前向（三种权重来源） | 328–1196 | `run_block_cuda_w_dev`（668–1005，**338 行**）、`run_block_cuda_w`、`run_block_cuda`、`sdpa_cuda_rope`；含 host `rms_norm`/`silu_mul` | `rms_norm`/`silu_mul` 与 `h3_vae_int8.c:150/182` **完全重复** |
| Metal 风格空间 tile：轴/羽化/Comfy blend/stitch + `h3_vae_decode_cuda_spatial` | 1197–2207 | `h3_vae_tile_*`、`h3_vae_stitch_rgb_tiles(_comfy)`、`decode_cuda_spatial`（1719–2207，**488 行**） | 官方 256/64 几何 |
| 36 层设备常驻链 + CUDA graph 捕获 + scratch | 2208–3079 | `vae_run_layers_dev` **单函数 871 行** | Task1 施工现场 |
| env 助手 | 118–130、283 | `env_flag_is_one`、`env_int_default`、`env_is_one`（**同文件三份**） | |

建议目标：`h3_vae_cuda_weights.c` / `h3_vae_cuda_block.c` / `h3_vae_cuda_tiling.c` / `h3_vae_cuda_layers.c`，共享 `h3_vae_int8_cuda_internal.h`。**现在不做**（§9.C2）。

### 2.4 `h3_w4a8_cache.c`（1994 行）

| 段 | 行 | 内容 |
| --- | --- | --- |
| slot/LRU/recycle/load_block | 88–710 | 三格式 `load_one_linear_nvfp4/_int8/_(w4a8)`（326–628）|
| `ensure_capacity_for` | 711–1082 | **单函数 372 行**（窗口 / 预留 / refit 判定） |
| 窗口与显存治理 API | 1083–1312 | `refit_after_workspace`、`freeze/unfreeze_window`、`drop_device_weights`、`make_room`、`set_window_size` |
| host 分级流水线（mmap+pread 线程 / hostreg / async H2D） | 1356–1933 | `stage_one_host_*`、`stage_read_th`、`h3_w4a8_cache_stage_host_int8`（1605–1933，**329 行**） |
| 统计 | 1934–1994 | |

建议：`h3_w4a8_cache.c`（核心 LRU）+ `h3_w4a8_cache_formats.c`（三格式 load/estimate）+ `h3_w4a8_host_stage.c`（分级流水线）。名字问题：文件叫 `w4a8` 但主线是 INT8，建议长期改名 `h3_dit_weight_cache*`（C 档，因所有 probe/CMake 都引用）。

### 2.5 顺带：`h3_dit_w4a8.c`（6742 行）

用户未点名，但它是第二大文件：0–1000 配置/LoRA/clean-KV/Ulysses 设置；1000–2000 workspace/aux/AdaLN；2000–3000 block 前向；3000–4000 core/patch/final；4000–5000 Ulysses layer chain；5000–6000 refiner + token reduction；6000+ forward/Euler/reuse/extrapolate。至少 Ulysses chain、token reduction、refiner 三块可独立成文件。同样属 C 档。

---

## 3. 未跟踪文件分类（`git status --short -uall` 共 76 个）

| 组 | 文件 | 模块 | 性质 | 处置建议 |
| --- | --- | --- | --- | --- |
| 文档 | `docs/superpowers/plans/2026-08-25-*`（6）、`2026-08-26-*`（1）、`2026-09-11-*`（9）、`2026-09-13-*`（8）、`2026-09-15/16/22/30-*`（5）= **29 份** | docs | 产品文档 | 提交；但先按 §8 标 superseded |
| 评测集 | `evals/768p/MANIFEST.tsv`、`ids/.gitkeep`、`ids/p01_sprint.ids`、`prompts/p01..p08_*.txt`（11） | 评测 | 产品（768P 协议输入，44K） | **应跟踪**，不进 .gitignore；若未来放输出请写到 `out/` |
| 2-stage 模块 | `h3_selflift.c/h`、`h3_taomate.c/h`、`h3_latent_upscaler.c/.cu/.h/_internal.h`（8） | lift/taomate | 产品（preset lift/taomate） | 提交，与其 test 同批 |
| NVFP4 DiT 臂 | `h3_dit_nvfp4.c/h`、`tools/h3_dit_nvfp4_probe.c`、`tools/h3_nvfp4_int8_sidecar.c`（4） | DiT/quant | 实验 | 提交（CMake 已引用，不提交则构建断） |
| TAE CUDA | `h3_tae_cuda.cu/h`、`h3_tae_internal.h`（3） | TAE | 实验（opt-in） | 提交 |
| 测试 | `tests/test_h3_selflift.c`、`test_h3_taomate.c`、`test_h3_tae.c`、`test_h3_vae_resident.c`、`test_h3_vae_rope.c`（5） | tests | 产品 | 与对应模块同批 |
| 工具/脚本 | `tools/h3_768p_eval.sh`、`h3_dp_generate.sh`、`h3_5090x8_sync.sh`、`h3_tokenize_prompt.py`、`h3_lbh_oracle.c`、`h3_lbh_oracle_ref.py`、`h3_p2p_bw_probe.cu`（7） | 运维/诊断 | 产品（eval）/ 诊断 | 提交；`h3_5090x8_sync.sh` 含主机名请先核查无凭据 |
| SDPA 插件 | `tools/h3_sage3_embed.c/.py`、`h3_sol_cute_embed.c/.py`、`h3_sol_cute_bench.py`（5） | attn | 实验 | 提交（CMake 已引用 `h3_sol_cute_embed.c`/`h3_sage3_embed.c`） |
| 第三方摘录 | `third_party/accel_survey/README.md`、`sol-engine-excerpts/README.md`、`super_acceleration/README.md`、`STAGE2_CONTRACT.md`（4，12K） | survey | 文档 | 提交（6 个整仓 clone 已在修改后的 .gitignore 里；`FastVideo-excerpts/` 是空目录，git 自动忽略） |

**不需要新增 .gitignore 项**的：`out/`（已 `/out`）、`models/`、`input/`、`tmp_*`、`*.o/*.d`、probe 二进制、accel_survey 六个 clone（本次 .gitignore diff 已加）。

**建议补的 .gitignore 项**：`build*/`（Linux 侧 `build-cuda` / `build-cuda-sm120` 若在仓内构建会全量泄漏）、`h3_ulysses_test_a2a`（`h3_*test` 模式匹配不到它）、`h3_cuda_generate` / `h3_cuda_info` / `h3_lbh_oracle` / `h3_nvfp4_int8_sidecar`（若用 Makefile 风格在根目录产出）。

### 3.1 建议分批提交（仅建议）

`tools/h3_cuda_generate.c`（+2332 行）与 `h3_gpu_cuda.cu`（+1654 行）的改动横跨所有功能，严格按功能拆需要 `git add -p`；给出两档：

**细分 10 批（需要 `add -p`）**

1. 构建/忽略：`.gitignore`、`CMakeLists.txt`
2. GPU 核（无行为切换的新 API）：`h3_gpu_cuda.cu/h`、`h3_gpu_cuda_sdpa_cudnn.cpp`、`h3_ulysses_comm.cu`
3. DiT 核：`h3_dit_w4a8.c/h`（clean-KV、NVFP4 open、pregrow）、`h3_w4a8_cache.c/h`（host stage、窗口 API）、`h3_w4a8_dequant.c`
4. NVFP4 DiT 臂：`h3_dit_nvfp4.c/h`、`tools/h3_dit_nvfp4_probe.c`、`tools/h3_nvfp4_int8_sidecar.c`
5. VAE 常驻 / tile / RoPE（09-16 Task1）：`h3_vae_int8.c/h`、`h3_vae_int8_cuda.c/h`、`tests/test_h3_vae_resident.c`、`tests/test_h3_vae_rope.c`
6. TAE CUDA：`h3_tae.c/h`、`h3_tae_cuda.cu/h`、`h3_tae_internal.h`、`tests/test_h3_tae.c`
7. LoRA refiner scope + preset 表：`h3_lora.c/h`、`tests/test_h3_lora.c`、`h3_generate_preset.c/h`、`tests/test_h3_generate_preset.c`
8. 2-stage：`h3_selflift*`、`h3_taomate*`、`h3_latent_upscaler*`、`tests/test_h3_selflift.c`、`tests/test_h3_taomate.c`、`tools/h3_lbh_oracle.c`、`tools/h3_lbh_oracle_ref.py`
9. generate CLI + 评测工具：`tools/h3_cuda_generate.c`、`tools/h3_sol_attn_bench.c`、`tools/h3_768p_eval.sh`、`h3_dp_generate.sh`、`h3_5090x8_sync.sh`、`h3_tokenize_prompt.py`、`h3_p2p_bw_probe.cu`、`evals/768p/`、`tools/h3_sage3_embed.*`、`h3_sol_cute_*`
10. 文档：29 份未跟踪计划 + 8 份已修改文档 + `third_party/accel_survey/**/README.md`、`STAGE2_CONTRACT.md`

**粗分 3 批（不需要 `add -p`，每批都能独立编译）**

- A：批 1+2+3+4（构建 + 核 + DiT + NVFP4 臂）
- B：批 5+6+7+8（VAE/TAE/LoRA/preset/2-stage 模块与单测）
- C：批 9+10（generate CLI、工具、评测集、文档）

每批验证：Linux 侧 `cmake --build build-cuda-sm120 -j` 全目标 + 12 个 `h3_*_test` 跑通；批 5 另跑 `h3_vae_resident_test`/`h3_vae_rope_test`。

---

## 4. 已修改文件（33 个 `M`，+8676 / −1207）

| 事项 | 文件（diff 行） | 说明 |
| --- | --- | --- |
| 构建/忽略 | `.gitignore`（+8）、`CMakeLists.txt`（135） | 新增 6 个可执行目标、`h3_cuda_generate` 挂 9 个新源 |
| **Task1 VAE 常驻 / tile / RoPE** | `h3_vae_int8_cuda.c`（**1811**）、`h3_vae_int8_cuda.h`（9）、`h3_vae_int8.c`（46）、`h3_vae_int8.h`（16） | 新 API：`weight_cache_wanted`、`tile_batch_want`、`prepare_rope_ex`、`hostreg_release`、`release_cached` |
| DiT 核：clean-KV / host stage / NVFP4 open | `h3_dit_w4a8.c`（529）、`.h`（33）、`h3_w4a8_cache.c`（**971**）、`.h`（24）、`h3_w4a8_dequant.c`（6） | 新 API：`open_nvfp4`、`clean_kv_*`（8 个）、`pregrow_layers_ws`、`host_ready_blocks`、`set_async_h2d`、`unfreeze_window`、`drop_device_weights` |
| GPU 核 | `h3_gpu_cuda.cu`（**1654**）、`.h`（75）、`h3_gpu_cuda_sdpa_cudnn.cpp`（517）、`h3_ulysses_comm.cu`（100） | `sage3_warmup/drop_pool`、`conv3d_ncdhw_bias_dd`、`host_register`、`sdpa_f32_dev`、`qkv_rope_*_layout(_nb)`、`sdpa_bind_kv*`、`vae_unpatch_rgb`；cuDNN 517 行为 rect/batched 计划 |
| preset 表 | `h3_generate_preset.c`（88）、`.h`（6）、`tests/test_h3_generate_preset.c`（+81） | 新 preset：`nvfp4`、`lift`、`taomate`；`H3_TAOMATE_LORA` / `H3_TURBO_LORA` |
| LoRA | `h3_lora.c`（119）、`.h`（18）、`tests/test_h3_lora.c`（+18） | refiner scope、`parse_tensor`、`load_flags` |
| TAE | `h3_tae.c`（**619**）、`.h`（20） | `allow_opt_in`、`bind_cuda`、`decode_frames`（配未跟踪 `h3_tae_cuda.cu`） |
| generate CLI | `tools/h3_cuda_generate.c`（**2332**）、`tools/h3_sol_attn_bench.c`（27） | 承接以上全部 |
| 文档 | `docs/nvidia_optimization_plan.md`（4）、`nvidia_quant_contract.md`（2）、6 份 plans（其中 `09-13-minimax-lightning-survey` 322、`09-13-taomate-h3-analysis` 269） | 8 文件 +374/−247 |

结论：至少 **7 件独立工作**（Task1 VAE、DiT clean-KV/host stage、NVFP4 臂、preset、LoRA、TAE、GPU 核 API）混在一个工作区，且已持续到 09-30 未提交。

---

## 5. 死代码 / 半成品 / 重复实现

### 5.1 无调用者的对外函数（头文件声明 + 定义，全仓 `.c/.cu/.cpp/.m` 仅出现 1 次）

| 函数 | 定义处 | 判断 |
| --- | --- | --- |
| `vae_th` / `vae_th_ok` | `tools/h3_cuda_generate.c:1635/1639/6130–6132/6463` | 只 join 无 create，**确认死** |
| `h3_cuda_sdpa_set_kvpad` | `h3_gpu_cuda.cu:8569`（`(void)seq;` 空实现） | 死桩 |
| `h3_cuda_qkv_rope_f32_dev`、`h3_cuda_qkv_rope_split_half_f32_dev` | `h3_gpu_cuda.cu:8400/8493` | 已被 `_dev_layout` 版取代 |
| `h3_cuda_ffn_swiglu_i8_dev` | `h3_gpu_cuda.cu:8343` | 无调用 |
| `h3_cuda_fwht_act_f32_dd`（+ `h3_cuda_ensure_hadamard_fwht`） | `h3_gpu_cuda.cu:3365` | FWHT 替代 ConvRot 的实验，无调用 |
| `h3_cuda_event_destroy`、`h3_cuda_op_stream`、`h3_cuda_stream_count`、`h3_cuda_sol_profile_enabled` | `h3_gpu_cuda.cu` | 无外部调用 |
| `h3_tae_decode_f32` | `h3_tae.c:599` | CPU TAE 解码，被 CUDA 版取代（`test_h3_tae.c` 未用） |
| `h3_w4a8_cache_max_blocks_fit` | `h3_w4a8_cache.c:185` | 无调用 |
| `h3_dit_w4a8_clean_kv_max_tokens`、`h3_dit_w4a8_velocity_rel_l1_f32`、`h3_dit_w4a8_qkv_take_v`、`h3_dit_w4a8_token_reduction_enabled`、`h3_dit_w4a8_cuda`、`h3_dit_w4a8_store` | `h3_dit_w4a8.c` | 无调用（后两个是 getter，可能为将来预留） |
| `h3_dit_int8_find`、`h3_sdpa_backend_table`、`h3_dit_schedule_audio_condition_row`、`h3_segment_name`、`h3_weight_store_shards` | 各自 .c | 无调用 |
| `h3_gpu_tensor_dtype/elements/write_bf16/write_f32`、`h3_gpu_text_qk_rope_bf16` | `h3_gpu.m` | **Metal 公开 API，不建议删** |

（检测方法：对 637 个 `h3_*.h` 声明的函数做全词计数；≤1 即无调用。粗筛，删除前逐个 `rg -w` 复核。）

### 5.2 默认关且无测试的 env 分支

- `H3_DIT_DIST_ATTN`（generate.c:2588）→ `h3_cuda_dist_attn_*`：无单测、无文档。
- `H3_LIFT_Z0_ONLY` / `H3_LIFT_ZLOW_ONLY` / `H3_LIFT_HIRES_WINDOW`：调试臂，无测试。
- `H3_TAOMATE_CLEAN_KV`（Packed-Q，注释自述「OOM at 768P」）vs `H3_TAOMATE_LAYER_KV`（真 TaoMate hook）：两条互斥实现并存，前者已被后者取代但仍在。
- `H3_NVFP4_HOST_ACT` / `H3_NVFP4_DEV_ACT` / `H3_NVFP4_HOST_GQA` / `H3_NVFP4_HOST_GEMM`：TE 的 host 回退臂，各在 `h3_nvfp4.c` 与 `h3_nvfp4_cuda.cu` 双读。
- `H3_VAE_QKV_LAYOUT`、`H3_VAE_SDPA_F32`、`H3_VAE_NO_CONVROT`、`H3_VAE_ENCODE_HOST`：VAE 调试臂。
- `H3_ULYSSES_A2A_F32`：U>1 才生效，而 U>1 默认 stub。
- `H3_CONVROT_SYLVESTER`：在 `h3_gpu_cuda.cu:3235` 与 `h3_w4a8_dequant.c:372` 各读一次，host/device Hadamard 必须一致却无单点。
- 4 个 `pthread` 作业里 `vae_willneed_th`（318）目前是否仍被 create 需复核（本次只确认 `vae_th` 死）。

### 5.3 重复实现

| 助手 | 份数 | 位置 |
| --- | --- | --- |
| `static fail(error, size, msg/fmt…)` | **28 份**（两种签名：`msg` 与 `fmt, ...`） | 几乎每个 .c/.cu |
| `env_flag_is_one` / `env_is_one` / `env_int_default` / `env_int` | 5 份 | `h3_dit_w4a8.c:26/65`、`h3_vae_int8_cuda.c:118/123/283`、`h3_step_cache.c:16` |
| `now_ms` / `elapsed_ms` | 4 + 4 份 | `generate.c`、`h3_vae_int8_cuda.c`、`h3_audio_vae_cuda.c`、`h3_t2av_timing_probe.c`；`h3_dit_w4a8.c`、`h3_vae_int8.c`、`h3_vae_int8_cuda.c`、`h3_nvfp4_probe.c` |
| host `rms_norm` / `silu_mul` | 2 份各 | `h3_vae_int8.c:150/182` 与 `h3_vae_int8_cuda.c:387/400` **逐行相同** |
| RoPE `HALF_OFFSET/SPLIT_HALF` env 解析 | 4 文件 5 处 | `h3_dit_w4a8.c:143`、`h3_gpu_cuda.cu:48`、`h3_vae_int8_cuda.c:800/871`、`generate.c:4564` —— 名字带 `DIT_` 却被 VAE 读 |
| RoPE 表/旋转 | 多份合理（Metal `h3_dit.c`、`h3_video_vae.c`、`h3_vision_encoder.c` 各一；CUDA `h3_dit_w4a8_rope_tables_f32`、`h3_vae_int8_prepare_rope(_ex)`、`h3_nvfp4_rope(_mrope)`、`h3_vision_cuda.c:456`）| 语义不同（DiT 3D / VAE 3D / TE 1D+mRoPE / vision 2D），不算重复；但 CUDA 侧 `h3_cuda_qkv_rope_*` 有 6 个变体（f32 / f16 / split_half / dev / layout / layout_nb），其中 2 个无调用（§5.1） |
| bf16↔f32 cast | `h3_vision_encoder.c:51/58`（Metal）+ CUDA 核 `f32_to_f16/f16_to_f32_dev` | 跨后端不算重复 |
| `load_f32` 前置声明+定义、`ensure_vae_head_dev` 等 4 个前置声明 | `h3_vae_int8_cuda.c:65–76/314 ↔ 552/3179–3444` | 文件排序问题 |

### 5.4 TODO/FIXME/XXX

代码内仅 2 处真 TODO：`h3_vae_int8.h:16`「Full tiling still TODO」（实际 tile 已实现，注释过期）、`tools/h3_w4a8_generate_probe.c:244`「full TE forward TODO」（历史探针）。其余 `XXXXXX` 均为 `mkstemp` 模板。**没有 FIXME/HACK**。

### 5.5 构建接线

- `tools/h3_p2p_bw_probe.cu`：唯一未接 CMake 的源（头注释：手工 `nvcc -O2 -std=c++17`）。
- Makefile `h3_dit_bench_864` 看似引用不存在的 `tests/bench_dit_864.c`，实为用 `-D` 从 `bench_dit.c` 另编一份，**不是死目标**。
- 20 个 Metal 测试（`tests/test_*.c`）只在 Makefile；不算未接线。

---

## 6. `H3_*` env 开关总表

硬数据：**234 个不同名字**（`rg 'getenv\("H3_'`，含 .m），其中 **163 个在 `docs/` 与 `README.md` 中零出现**。分布：

| 前缀族 | 数 | 读取位置 | 文档覆盖 |
| --- | --- | --- | --- |
| `H3_BENCH_*` | ~70 | `tests/bench_dit.c`、`tests/test_bf16.c`、`h3_dit.c`（Metal 基准 A/B） | 几乎无；Metal 上游范畴 |
| `H3_DISABLE_*` | ~60 | `h3_gpu.m`、`h3_dit.c`（Metal 核开关）；CUDA 仅 `H3_DISABLE_FUSED_FINAL_*`（也在 `h3_dit.c`） | 无；Metal 上游范畴 |
| `H3_INT8_*` / `H3_NAX*` / `H3_MPS_*` | ~20 | `h3_gpu.m`（Metal INT8/NAX） | 无；Metal |
| **CUDA 产品路径相关（不含上三族）** | **~97** | 见下表 | 约 1/4 有文档 |

下表只列 CUDA 路径（Metal 族略）。默认值按代码读法：`=1` 表示 `getenv && [0]=='1'` 才开；`int` 表示 `strtol` 带默认。文档列：`✓` 出现在 docs/README，`—` 未出现。

| env | 默认 | 读取处 | 文档 | 备注 |
| --- | --- | --- | --- | --- |
| `H3_CUDA_SDPA_BACKEND` | 未设→CLI `--sdpa` | `h3_sdpa_registry.c:187/199`、`h3_gpu_cuda.cu:4295/4701`、`h3_vision_cuda.c:101`、`tools/h3_cuda_info.c`、`h3_t2av_timing_probe.c` | ✓ | **9 处读**，应集中到 registry |
| `H3_CUDNN_SDPA_ALIGN`、`H3_CUDNN_PLAN_BUILD_MS` | — / int | `h3_gpu_cuda_sdpa_cudnn.cpp:129/407` | — | |
| `H3_SOL_*`（`ATTN_ROOT`、`CUTE_LIB`、`TAU`、`SINK_TOKENS`、`FIRST_DENSE_STEPS/LAYERS`、`WARMUP_SEQ/HEADS/DIM`、`PROFILE`） | 各异 | `h3_gpu_cuda.cu` 4512–6317 | `ROOT/TAU` ✓，其余 — | 实验臂；`SOL_TAU`、`SINK_TOKENS` 各读 2 处 |
| `H3_SAGE3_LIB` / `H3_SAGE3_ROOT` / `H3_SAGE3_TOOLS` | — | `h3_gpu_cuda.cu:4847/4898/5049`、`tools/h3_sage3_embed.c:45` | ✓ | 实验臂 |
| `H3_INT8_GEMM_TILE_M` | int | `h3_gpu_cuda.cu:882` | — | |
| `H3_LAYER_KV_ALLOW_TC` | =1 | `h3_gpu_cuda.cu:8697` | — | TaoMate clean-KV |
| `H3_CONVROT_SYLVESTER` | =1 | `h3_gpu_cuda.cu:3235`、`h3_w4a8_dequant.c:372` | — | **双读需一致** |
| `H3_DIT_QKV_GROUPED` | =1 | `h3_gpu_cuda.cu:42`、`generate.c:4563` | — | 双读 |
| `H3_DIT_ROPE_HALF_OFFSET` / `H3_DIT_ROPE_SPLIT_HALF` | =1 | `h3_dit_w4a8.c:144/145`、`h3_gpu_cuda.cu:49/50`、`h3_vae_int8_cuda.c:800/871`、`generate.c:4564/4565` | `HALF_OFFSET` ✓ | **4 文件 5 处**；VAE 读 `DIT_` 前缀 |
| `H3_DIT_ADALN_SILU`、`H3_DIT_TEXT_GAIN`、`H3_DIT_TEXT_MOD_ROW`、`H3_DIT_NO_REFINER`、`H3_DIT_PLAN_REUSE`、`H3_DIT_HOST_STAGE_TE`、`H3_DIT_HOST_STAGE_WAIT_BLOCKS`、`H3_DIT_TE_OVERLAP_BLOCKS`、`H3_DIT_W4A8_ATTN_WINDOW`、`H3_DIT_DIST_ATTN` | 各异 | `generate.c` | `TEXT_GAIN` ✓，其余 — | 全在 main 内直接 getenv |
| `H3_DIT_HOST_STAGE_THREADS` / `_VERBOSE` | int / =1 | `h3_w4a8_cache.c:1665/1625` | — | |
| `H3_DIT_INT8_ALLOW_SHRINK`、`H3_W4A8_CACHE_RESERVE_MIB`、`H3_W4A8_CACHE_REFIT_RESERVE_MIB` | =1 / int | `h3_w4a8_cache.c:1218/172/1148` | `REFIT` ✓ | |
| `H3_DIT_INT8_NO_CONVROT` | =1 | `h3_dit_int8.c:196` | — | |
| `H3_TOKEN_REDUCTION_BLOCKS/EARLY/SCALE` | int | `h3_dit.c`（Metal）与 `h3_dit_w4a8.c:713/725/745` | `BLOCKS/EARLY` ✓ | 跨后端共用语义 |
| `H3_NVFP4_*`（`VERBOSE`、`TE_LAYERS`、`HOST_ACT`、`DEV_ACT`、`HOST_GQA`、`HOST_GEMM`、`DUMP_DIR`、`DISABLE_DEV_CACHE`、`INT8_SIDECAR`） | 各异 | `h3_nvfp4.c`、`h3_nvfp4_cuda.cu`、`h3_dit_nvfp4.c` | `VERBOSE/HOST_GEMM` ✓ | `HOST_ACT/DEV_ACT/HOST_GQA` 双文件读 |
| `H3_TE_PREFAULT`、`H3_TE_PACKED_RESIDENT`、`H3_TE_MMAP_HOSTREG`、`H3_TE_MMAP_POPULATE`、`H3_TE_PROMPT_CACHE` | 各异 | `h3_nvfp4_cuda.cu:20/130/158`、`h3_nvfp4.c:189`、`generate.c:2211` | `PREFAULT/POPULATE/PROMPT_CACHE` ✓ | |
| `H3_VAE_CUDA_W_RESIDENT` | =1 | `h3_vae_int8_cuda.c:3473`、`generate.c:2686/2687/6068/6074` | ✓ | **5 处读** |
| `H3_VAE_WEIGHT_CACHE` | 默认开（`=0` 关） | `h3_vae_int8.c:460`、`generate.c:2500/6070/6075` | ✓ | 4 处读 |
| `H3_VAE_CUDA_GRAPH` | **未设时跟随 `H3_VAE_FLASH_SDPA`** | `h3_vae_int8_cuda.c:2219–2228` | ✓（09-30 已补） | 隐式耦合；cuDNN 路径 capture 会 abort |
| `H3_VAE_FLASH_SDPA` | =1 | `h3_vae_int8_cuda.c:1785/1791/1794/2228/2354`（`env_is_one` 5 次） | ✓ | 与上条为一对 |
| `H3_VAE_TILE_PIXELS` | int | `h3_vae_int8_cuda.c:1241`、`h3_video_vae.c:667`（Metal）、`generate.c:6124` | ✓ | |
| `H3_VAE_TILE_BATCH` | int（宏上限 4） | `h3_vae_int8.c:466` | ✓ | |
| `H3_VAE_NO_SPATIAL_TILE`、`H3_VAE_HOSTREG`、`H3_VAE_QKV_LAYOUT`、`H3_VAE_DEBLOCK`、`H3_VAE_NO_CONVROT`、`H3_VAE_SDPA_F32`、`H3_VAE_ENCODE_HOST` | 各异 | `h3_vae_int8_cuda.c`、`h3_vae_int8.c`、`generate.c:1440`、`h3_vae_encode_f32.c:39` | `NO_SPATIAL_TILE/HOSTREG/DEBLOCK` ✓ | `SDPA_F32` 需在 CUDA ctx 前 setenv TF32 |
| `H3_TAE_TIME_CHUNK` / `H3_TAE_NO_CHUNK` | int / =1 | `h3_tae_cuda.cu:388/393` | — | |
| `H3_TAOMATE_*`（`STAGE1_W/H`、`CLEAN_KV`、`LAYER_KV`、`CHUNK_OVERLAP`、`SINK_T`、`FORCE_CHUNKS`、`LORA`） | 各异 | `generate.c:2433–2533`、`h3_generate_preset.c:89` | — | 全部无文档；`CLEAN_KV` 与 `LAYER_KV` 互斥实现并存 |
| `H3_TURBO_LORA` | 路径 | `h3_generate_preset.c:146` | ✓ | |
| `H3_LATENT_UPSCALER`、`H3_LBH_CONV`、`H3_LBH_TIME_CHUNK`、`H3_LIFT_Z0_ONLY`、`H3_LIFT_ZLOW_ONLY`、`H3_LIFT_HIRES_WINDOW` | 各异 | `generate.c`、`h3_latent_upscaler.c/.cu`、`h3_selflift.c:49` | — | `LIFT_Z0_ONLY` 3 处读 |
| `H3_MAGCACHE_MAX_CONSEC`、`H3_REUSE_STEPS` | int | `generate.c:4316`、`h3_dit.c:2558`（Metal） | `MAGCACHE` ✓ | |
| `H3_ULYSSES_A2A_F32` | =1 | `generate.c:1398` | — | U>1 才生效 |
| `H3_AUDIO_VAE`、`H3_AUDIO_VAE_OVERLAP`、`H3_AUDIO_CONV_BACKEND` | 路径 / =1 / 字串 | `generate.c:220/6240`、`h3_gpu_cuda_audio.cu:253` | ✓ | |
| `H3_VISION_HOST`、`H3_VISION_SDPA` | =1 / 字串 | `h3_vision_cuda.c:83/92` | ✓ | |
| `H3_DUMP_DIR`、`H3_DUMP_ONLY`、`H3_SKIP_PPM` | 路径 / =1 | `generate.c:1529/1530/6273` | `DUMP_DIR` ✓ | |
| `H3_FFMPEG`、`H3_FFPROBE` | 路径 | `h3_ffmpeg.c:28/33` | ✓ | |
| `H3_T2AV_FORCE_PER_BLOCK` | =1 | `tools/h3_t2av_timing_probe.c` | — | 探针专用 |
| `H3_PROFILE` | =1 | `h3_dit.c`、`h3_gpu.m`、`h3_video_vae.c`（Metal） | ✓ | CUDA 侧不读 |

**重复/矛盾清单**

1. `H3_VAE_CUDA_GRAPH` 未设时跟随 `H3_VAE_FLASH_SDPA`（隐式耦合，且 cuDNN 路径捕获会 abort）。
2. `H3_DIT_ROPE_*` 被 VAE 读（前缀误导），且 4 文件各自解析。
3. `H3_CONVROT_SYLVESTER` host/device 双读，无单点。
4. `H3_CUDA_SDPA_BACKEND` 9 处读，registry 之外还有 4 个文件自行解析。
5. `H3_VAE_CUDA_W_RESIDENT` / `H3_VAE_WEIGHT_CACHE` 在 generate.c 与库内各读一次（generate 只为打日志，但仍是两套判断）。
6. `H3_TAOMATE_CLEAN_KV`（Packed-Q，OOM）与 `H3_TAOMATE_LAYER_KV`（真 hook）两条并存。
7. `H3_NVFP4_HOST_ACT` / `DEV_ACT` 同时存在（一正一反两开关）。
8. 三套 env 助手（`env_flag_is_one` / `env_is_one` / `env_int_default`）散在两文件。

---

## 7. 测试与 CMake

### 7.1 测试清单与归属

| 构建 | 目标 | 源 | 覆盖模块 |
| --- | --- | --- | --- |
| CMake 单测（12） | `h3_step_cache_test`、`h3_taomate_test`、`h3_generate_preset_test`、`h3_selflift_test`、`h3_lora_test`、`h3_ulysses_test`、`h3_te_cache_test`、`h3_tae_test`、`h3_vae_rope_test`、`h3_vae_resident_test`、`h3_vae_encode_f32_test`、`h3_multimodal_cuda_test` | `tests/test_h3_*.c` | 各对应模块（host 侧逻辑为主） |
| CMake 需要真权重 | `h3_dit_w4a8_cond_test`（657 行） | `tests/test_h3_dit_w4a8_cond.c` | DiT 条件行 |
| CMake 条件目标 | `h3_ulysses_test_a2a`（仅 NCCL 存在） | `tests/test_h3_ulysses_a2a.cu` | A2A |
| Makefile（Metal，20 个） | `h3_tests`、`h3_metal_tests`、`h3_bf16_tests`、`h3_tokenizer_tests`、`h3_text_tests`、`h3_audio_gpu_tests`、`h3_real_*_test`（12）、`h3_semantic_*_test`（2）、`h3_av_mux_test`、`h3_dit_bench(_864)` | `tests/test_*.c`、`bench_dit.c` | Metal 全路径 |

**没有 `enable_testing()` / `add_test()`**：CMake 侧 12 个单测无法 `ctest`，只能手工逐个跑。

### 7.2 无单测的 CUDA 模块

| 模块 | 现有验证手段 | 缺口 |
| --- | --- | --- |
| `h3_vae_int8_cuda.c`（3915） | 仅 `h3_cuda_generate` 端到端 + host `h3_vae_int8_probe` | 无 tile 拼接 / 常驻 / graph 的单测（`vae_resident_test` 只测 host 侧 want 判定，32 行） |
| `h3_gpu_cuda.cu` 46 个核 | W4A8 probes、`h3_sol_attn_bench`、`scripts/h3_sdpa_backend_smoke.sh` | 无 kernel 级数值对照测试 |
| `h3_nvfp4.c` / `h3_nvfp4_cuda.cu` | `h3_nvfp4_probe` | 无 |
| `h3_w4a8_cache.c` | `h3_w4a8_cache_probe`（8 月） | host stage 流水线（新增 580 行）无测试 |
| `h3_audio_vae_cuda.c` | 3 个 Au probes | 无 |
| `h3_latent_upscaler.*` | `tools/h3_lbh_oracle.c` + `.py`（是数值对照，但挂在 tools 不在 test） | 建议改名 `h3_latent_upscaler_test` 并进 add_test |
| `h3_token_ids.c`、`h3_sdpa_registry.c`、`h3_cuda_device_caps.c`、`h3_dit_nvfp4.c` | 无 / shell smoke / 无 / probe | |
| `tools/h3_cuda_generate.c` main | `tools/h3_768p_eval.sh`、`scripts/h3_accel_matrix_smoke.sh` | CLI 解析、shape 推导、preset 组合无单测 |

### 7.3 CMake 目标命名一致性

- 单测：`h3_<mod>_test`（12）✓；例外 `h3_ulysses_test_a2a`（后缀在中间）、`h3_dit_w4a8_cond_test`（模块名冗长但合规）。
- 探针：`h3_*_probe`（22）✓。
- 其它：`h3_cuda_info`、`h3_sol_attn_bench`、`h3_st_inventory`、`h3_nvfp4_int8_sidecar`、`h3_lbh_oracle`、`h3_cuda_generate`：无统一后缀，`lbh_oracle` 实为测试。
- Makefile 侧：单测 `h3_*_tests`（复数）vs `h3_real_*_test`（单数），与 CMake 不一致（Metal 上游，不建议动）。
- 源列表重复：`h3_safetensors.c h3_w4a8.c h3_w4a8_dequant.c h3_w4a8_cache.c h3_dit_int8.c h3_dit_nvfp4.c h3_nvfp4.c` 这组在 **~20 个目标**里逐个手写；本次 CMake diff 的 135 行中大半是给每个目标补 `h3_dit_int8.c h3_dit_nvfp4.c h3_nvfp4.c`。应抽成 `h3_host_core` / `h3_dit_cuda` STATIC。
- `install(TARGETS …)` 列表 22 行手工维护，含所有 probe。

---

## 8. 文档散落：`docs/superpowers/plans/` 46 份

### 8.1 当前有效（建议保留在顶层的 5 份 + 3 份基础文档）

| 文件 | 角色 |
| --- | --- |
| `2026-09-30-h3c-5090-fine-opt.md`（894 行） | **主施工册**（R1–R61、KPI-A/B） |
| `2026-09-16-h3c-community-speed-direction.md` | 方向档（废金线叙事、Task1–N 定义） |
| `2026-09-13-h3c-768p-eval-protocol.md` | 评测尺子（5s·1344×768） |
| `2026-09-22-h3c-accel-web-survey.md` | 外部口径复核（09-30 仍引用） |
| `2026-09-15-h3c-module-opt-plan.md` | 模块枢纽（09-16 注「口径作废，事实可引用」）——**边界**，可保留为枢纽或标 partially-superseded |
| `docs/nvidia_build.md`、`docs/nvidia_quant_contract.md`、`plans/h3-generation-regression-anchor.md` | 构建 / 量化契约 / 回归锚（长期有效） |

### 8.2 已被显式取代（文内自述）

| 文件 | 被谁取代 |
| --- | --- |
| `2026-08-21-h3-accel-survey.md` | 文内「2026-08-25 已 superseded」→ `08-25-h3c-4090-accel-plan` |
| `2026-09-13-h3c-5090-multigpu-extreme.md` | 文内「已被完整方案取代」→ `09-13-5090-extreme-full-plan` |
| `2026-09-16-h3c-scheme-selection.md`、`2026-09-15-h3c-module-opt-plan.md` | `09-16-community-speed-direction` 尾注「口径作废，事实可引用」 |

### 8.3 被 09-16「废金线叙事」隐式取代（建议加 superseded 横幅，指向 09-16 + 09-30）

所有以「金线 INT8 8-step + cuDNN 为默认、极速档 opt-in」为前提的计划：
`08-25-h3c-4090-accel-plan`、`08-25-h3c-accel-backends-cache-matrix`、`08-25-h3c-multi-gpu-arch-speed`、`08-26-h3c-next-tech-plan`、`09-11-h3c-next-opt-plan`、`09-11-h3c-5090-turbo-sku`、`09-11-h3c-5090-turbo-implementation`（已跟踪，入度 12——只加横幅不移动）、`09-13-h3c-5090-extreme-full-plan`、`09-13-h3c-turbo-explainer`（自述「状态以 out/verify_turbo_impl/PROGRESS.md 为准」）、`08-18-nvidia-accel-r1-r3`、`08-18-r1r3-closeout-fixes`、`docs/nvidia_optimization_plan.md`（1717 行，4090 质量优先状态跟踪；5090 由 09-30 接管）。

### 8.4 只读普查/分析（不是计划，保留作引用，建议归 `archive/surveys/` 或加「reference-only」标签）

`08-21-lightwan-h3-speed`、`08-21-remaining-quality-speed`、`08-25-h3c-vs-sglang-vllm-omni`、`08-25-h3c-vs-sglang-vllm-tech-deepdive`、`08-25-sla-vs-h3c`、`09-11-h3-few-step-quality`、`09-11-h3c-8x5090-768p-estimate`、`09-11-h3c-accel-principle-matrix`、`09-11-h3c-accel-survey-multigpu-5090`、`09-11-h3c-accel-survey-tech-deepdive`、`09-11-h3c-pcie-multigpu-lessons`、`09-11-h3c-two-stage-768p`、`09-13-h3c-5090-ulysses-driver-options`、`09-13-h3c-deepseek-opt-lessons`、`09-13-h3c-freetoken-analysis`、`09-13-h3c-minimax-lightning-survey`、`09-13-h3c-model-format-5090`、`09-13-h3c-plan-results-audit`、`09-13-h3c-selflift-comfy-analysis`、`09-13-h3c-taomate-h3-analysis`、`09-13-h3c-upstream-pr-review`（21 份）。

### 8.5 功能期计划（08-18 ×4，条件通道已落地）

`08-18-h3-generation-architecture`、`-roadmap`、`-implementation`（1150 行，入度 0）、`08-18-f2va-vision-condition-channel`——功能已在 `generate.c` 条件段落地，建议归 `archive/feature-plans/`。

### 8.6 链接入度（移动前必须处理）

相对链接入度最高：`09-11-5090-turbo-implementation` 12、`09-11-next-opt-plan` 11、`09-11-few-step-quality` 9、`08-25-sla-vs-h3c` 9、`08-25-4090-accel-plan` 8 …。**任何 `git mv` 到 `archive/` 都会打断 ≥100 条相对链接**，需要同步 `sed` 或改为只加横幅不移动（推荐先横幅）。另：`third_party/accel_survey/README.md` 也链接了 3 份 09-11 计划。

缺一份 `docs/superpowers/plans/README.md`（索引：当前有效 / superseded / reference-only 三栏）。`README.md` 只链接了 `nvidia_quant_contract.md` 与 `nvidia_build.md`。

---

## 9. 整理方案（三档）

### A 档：安全，建议立即（不碰热点源文件或只删已证死的局部符号）

| # | 动作 | 影响面 | 验证 |
| --- | --- | --- | --- |
| A1 | `.gitignore` 补 `build*/`、`h3_ulysses_test_a2a`、根目录可能产出的 `h3_cuda_generate`/`h3_cuda_info`/`h3_lbh_oracle`/`h3_nvfp4_int8_sidecar`；**不**忽略 `evals/`；确认 `third_party/accel_survey/*/` 六条已覆盖（可改为 `third_party/accel_survey/*/` + `!README.md` + `!sol-engine-excerpts/` 两行通配） | 仅 .gitignore | `git status --short` 无新噪声；`git check-ignore -v build-cuda/x` 命中 |
| A2 | 给 §8.2/8.3 共 ~15 份计划文首加统一横幅「> **Superseded（日期）**：现行方向见 09-16，施工见 09-30」；§8.4 21 份加「> reference-only」；新建 `docs/superpowers/plans/README.md` 三栏索引；`docs/nvidia_optimization_plan.md` 顶部加「4090 时代状态表；5090 见 09-30」 | 仅文档，不移动 | `rg -l '^> \*\*Superseded' docs | wc -l` = 计划数；链接零断裂（不移动） |
| A3 | 删 `vae_th`/`vae_th_ok` 4 行（generate.c）；删 `h3_cuda_sdpa_set_kvpad` 空桩 + 声明；删 `h3_tae_decode_f32`（`h3_tae.c/.h`）；删 `h3_w4a8_cache_max_blocks_fit`；删 `h3_dit_int8_find`、`h3_sdpa_backend_table`、`h3_dit_schedule_audio_condition_row`、`h3_segment_name`、`h3_weight_store_shards`（各文件 1 函数）。**暂不删** `h3_gpu_cuda.cu` 内 6 个死函数与 `h3_dit_w4a8.c` 内 6 个（热点文件，随下一次该文件的功能提交一起删） | 每处 ≤30 行，无调用者 | Linux `cmake --build` 全目标 0 warning 新增；Darwin `make test`（`h3_segment_name` 若属 Metal 共用需先 `rg -w` 复核）；12 单测通过 |
| A4 | 更新过期注释：`h3_vae_int8.h:16`「Full tiling still TODO」→ 指向 `h3_vae_decode_cuda_spatial` | 1 行 | 无 |
| A5 | 从 §6 表生成 `docs/env_reference.md`（CUDA 路径 ~97 个 env：名/默认/读取处/所属臂），Metal 族只列前缀 | 仅文档 | `rg -o 'getenv\("H3_[A-Z0-9_]+' | sort -u | wc -l` 与文档条数对得上 |
| A6 | 提交积压：先按 §3.1「粗分 3 批」提交（不需要 `add -p`，每批可独立编译） | 无代码变动 | 每批 `cmake --build` + 12 单测 |

### B 档：需要用户确认（改目录 / 改 CMake / 大 diff，但不改行为）

| # | 动作 | 影响面 | 验证 | 与施工冲突 |
| --- | --- | --- | --- | --- |
| B1 | **拆 `tools/h3_cuda_generate.c`**：第一步只抽 main 之前 56 个 static（io/jobs/taomate/lift/ulysses 胶水，~1,100 行）到 `tools/generate/h3_gen_*.c`，`main()` 不动；第二步引入 `h3_gen_ctx` + `h3_gen_cli.c`（CLI/`--help`/preset）；第三步切 cond / te / denoise / decode 四段并把 191 个 `goto cleanup` 收成 `gen_fail()` | CMake `h3_cuda_generate` 源列表；`tools/h3_768p_eval.sh` 不受影响 | 同 seed 同 preset：`phase_ms` 各字段 ±噪声、PPM/latent.raw **bit-exact**（纯重构）；`--help` 文本 diff 为空；12 单测 | **高**：09-30 R 项（日志、repeat_core、VAE 常驻接线）都在此文件。建议排在 S1 收口后、S2 开工前的窗口，或先只做第一步（main 之前，冲突面最小） |
| B2 | 实验臂归目录：`tools/probes/`（13 个 W4A8/DiT 里程碑 probe + 4 个音频 probe）、`tools/plugins/`（sol_cute/sage3 embed）、`experimental/`（`h3_dit_nvfp4*`、`h3_nvfp4_int8_sidecar`、`h3_step_cache*`？）。**Ulysses / TAE / token reduction** 是否算实验臂需用户定 | CMake 路径 + `target_include_directories`；文档路径引用 | `cmake` configure + 全目标 build；`rg 'tools/h3_w4a8_.*probe' docs` 更新链接 | 低（probe 无人改） |
| B3 | 归档旧计划：`git mv` §8.4 21 份 → `plans/archive/surveys/`，§8.5 4 份 → `plans/archive/feature-plans/`，§8.3 中已跟踪的 4 份留原位加横幅 | ≥100 条相对链接需 `sed` 重写；`third_party/accel_survey/README.md` 3 条 | 脚本检查所有 `](./…md)` 目标存在 | 低；但建议 A2 横幅先行，观察一个施工周期再移 |
| B4 | CMake 重构：`h3_host_core` STATIC（safetensors/w4a8/dequant/token_ids）+ `h3_dit_cuda` STATIC（dit_w4a8/int8/nvfp4/nvfp4/w4a8_cache）替换 ~20 处手写源列表；`enable_testing()` + 12 个 `add_test`；`h3_ulysses_test_a2a`→`h3_ulysses_a2a_test`；`h3_lbh_oracle`→`h3_latent_upscaler_test` 并 add_test | CMakeLists 大改；`install()` 列表 | 全目标 build；`ctest` 全绿；4090/5090 两台各 configure 一次 | 中（CMakeLists 本身有未提交 135 行，先提交再改） |
| B5 | 去重助手：`h3_env.h`（`h3_env_is_one/int/float`）替换 5 份；`h3_util.h`（`now_ms/elapsed_ms`、`fail`）；VAE host `rms_norm/silu_mul` 只留 `h3_vae_int8.c` 一份并导出；RoPE `HALF_OFFSET/SPLIT_HALF` 收成 `h3_dit_w4a8_rope_mode()` 单点 | 触 ~30 文件，每处几行 | build + 单测；`H3_DIT_ROPE_SPLIT_HALF=1` A/B 输出 bit-exact | 中（`h3_vae_int8_cuda.c`/`h3_dit_w4a8.c` 是热点） |
| B6 | 细分 10 批提交（替代 A6 粗分），需对 `generate.c`/`gpu_cuda.cu` 做 `git add -p` | 无代码变动 | 每批可编译（批 2 之前批 3–9 不能单独编译，需按序） | — |
| B7 | 本地清理：`make clean` 等价删除根目录 `*.o/*.d`、5 个 probe 二进制、`h3_real_video_encoder_test`、`tests/test_h3.o/.d`；8 个 `tmp_*` 脚本（VAE 对照期一次性）确认无用后删 | 均已 gitignore，纯本地 | `git status` 不变 | 无 |
| B8 | `h3_p2p_bw_probe.cu` 接进 CMake（可选目标）或移到 `tools/probes/` 并在 README 注明手工 nvcc | 1 个目标 | build | 无 |

### C 档：现在不做（与 5090 施工正面冲突或收益不抵风险）

| # | 动作 | 为什么现在不做 | 何时做 |
| --- | --- | --- | --- |
| C1 | 拆 `h3_gpu_cuda.cu`（§2.2 十段） | 09-30 R 项密集改 SDPA 选择、graph、GEMM epilogue、layer-KV；拆分 = 每个 R 项都要 rebase | KPI-A 收口后，按 §2.2 表一次切完 |
| C2 | 拆 `h3_vae_int8_cuda.c`（§2.3 四段）；`vae_run_layers_dev` 871 行拆函数 | Task1 VAE 常驻/36 层融合正在此处施工，未提交 diff 1811 行 | Task1 提交并验收后 |
| C3 | 拆 `h3_w4a8_cache.c` / 改名 `h3_dit_weight_cache`；拆 `h3_dit_w4a8.c`（Ulysses chain / token reduction / refiner） | host stage、clean-KV 未提交（971 + 529 行）；改名波及 20 个 CMake 目标与所有 probe | 同上 |
| C4 | env 语义整合（`H3_VAE_CUDA_GRAPH` 脱钩 `FLASH_SDPA`、`H3_DIT_ROPE_*` 改名、删 `TAOMATE_CLEAN_KV`、合并 `NVFP4_HOST/DEV_ACT`） | 会改行为；5090 评测脚本与 `out/verify_*` 复现命令依赖现名 | 先用 A5 文档钉住现状，施工册收口后再统一 |
| C5 | 删 Metal 族 `H3_BENCH_*` / `H3_DISABLE_*` / `H3_INT8_*` env 或改 Makefile 命名 | Metal 是 antirez 上游对齐面，本仓不改 | 不做 |
| C6 | 删 W4A8 原始臂（`h3_w4a8_dequant.c`、8 个 W4A8 probe） | `h3_w4a8_dequant.c` 承载 ConvRot Hadamard，INT8 主线仍依赖；probe 归目录即可（B2） | 不删，只归档 |
| C7 | 把 `h3_*` 源按模块建根目录子目录（`src/dit/`、`src/vae/`…） | 与 Metal Makefile `LIB_C` 平铺假设冲突，上游同步成本高；CMake/Makefile 双构建都要改 | 若未来与上游脱钩再议 |

---

## 10. 需要用户拍板的事

1. **拆 `generate.c` 吗？** 三种选择：(a) 不拆；(b) 只做 B1 第一步（抽 main 之前 1,100 行，冲突最小）；(c) 全拆到 `h3_gen_ctx`。以及排在 09-30 S1/S2/S3 的哪个窗口。
2. **旧计划处置**：只加横幅（A2）还是 `git mv archive/`（B3，需重写 ≥100 条链接）。§8.1 的 5 份「当前有效」名单是否认可，`09-15-module-opt-plan` 算有效还是 superseded。
3. **提交积压**：粗分 3 批（A6）还是细分 10 批（B6）；`evals/768p/` 与 `third_party/accel_survey/**/README.md` 是否入库；文档是否单独一批。
4. **哪些算实验臂**（决定 B2 移动范围）：NVFP4 DiT、sol/sage3 插件、`h3_step_cache`、Ulysses、TAE、token reduction、W4A8 里程碑 probe。
5. **死符号现在删还是随功能提交删**（A3 已排除热点文件内的 12 个；是否连这 12 个一起删）。
6. **本地清理**（B7）：根目录残留产物与 8 个 `tmp_*` 脚本是否可删。
