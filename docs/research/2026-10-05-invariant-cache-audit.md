# 精确不变量缓存覆盖审计（H3 源码级，只读）

> **日期：** 2026-10-05。**审计范围：** 本仓本地源码 `h3_dit.c`、`h3_dit_w4a8.c`、`h3_dit_schedule.c`、`h3_gpu_cuda.cu`、`h3_vae_int8_cuda.c`、`h3_te_cache.c`、`tools/h3_cuda_generate.c`（未访问 238 远端；本地副本即当前工作副本，与远端同源）。
> **框架依据：** `2026-10-05-acceleration-mechanisms.md` 机理族 4 之「精确不变量缓存（bit-identical）」——文本 embedding、RoPE 表、AdaLN conditioning 全程真常量，缓存 = 删重复计算，零误差，白捡；「精确不变量缓存与一切正交且永远该做」。
> **结论先行：** 三大类不变量中，**AdaLN conditioning 已收割干净**（两条 DiT 路径均一次性预算）；**文本 emb 运行时/跨 clip 已收割，跨进程为 opt-in**；**RoPE 表存在路径分裂**——BF16 路径（`h3_dit.c`）已缓存，INT8/W4A8 主生产路径（`h3_dit_w4a8.c`）与 VAE 路径（`h3_vae_int8_cuda.c`）仍每次 forward / 每 chunk 重算。

## 审计发现表

| 不变量 | 已缓存？ | 证据（源码定位） | 潜在零风险节省 |
| --- | --- | --- | --- |
| **文本 embedding（TE 前向，clip 级）** | **是**（运行时/跨 clip 全缓存；跨进程磁盘缓存 = opt-in，默认关） | `tools/h3_cuda_generate.c:4615-4677`（`h3_te_cache_load/save`，key=token_ids+te_layers+te_path；`te_prompt_cache: HIT/MISS` 日志；注释 "opt-in; gold default off"）；`h3_te_cache.c` 全文（key/load/save 三函数） | 同进程内已为零（每 run 只算一次 TE）。跨进程批量/重复 prompt 场景（T1 批量抽卡）开启 `--te-cache-dir` 可省整段 TE 前向——**已是现成开关，非新工程** |
| **文本 embedding 投影 + token refiner（`condition_proj` + 2× refiner block + final RMS）** | **是**（每 run 一次，产出 `text_h` 后全部 4-step/全 clip 复用） | `tools/h3_cuda_generate.c:4711`（`h3_dit_w4a8_refine_text_f32` 在采样循环外调用一次）；`h3_dit_w4a8.c:5950-6010`（refiner 主体，输出 `text_hidden` 被后续复用）；`text_h checksum` 日志仅打印一次 | 已为零（运行时无重复）。仅跨进程省 refiner 两 block 小 GEMM——量级 <TE 前向，价值低 |
| **AdaLN conditioning 参数（timestep emb + 每层 adaln_proj，共 50 block + final）** | **是**（双路径均一次性预算，每 step 内每层只查表） | BF16 路径：`h3_dit.c:1556`（"precompute AdaLN" 进度回调）、`h3_dit_schedule.c:237-331`（`h3_dit_schedule_precompute`：time_embeddings 一次 + 每 block 一次 `h3_gpu_linear_bf16` 预投影进 `schedule->blocks[block]` + final；采样循环只 `h3_dit_schedule_block()` 取行）。W4A8 路径：`h3_dit_w4a8.c:1748-1756, 3507-3516`（`adaln_t_table` 直接随权重加载成表）、`:3625-3691`（`t_row_from_sigma` 按 sigma 索引）、`:1908-1916`（`memcpy` 查行） | **≈0 —— 已收割干净**，无新增空间 |
| **DiT RoPE cos/sin 表（BF16 路径，`h3_dit.c`）** | **是**（per-run 一次预算，全 4-step 复用；含 token-reduction 的 reduced 表同样预算） | `h3_dit.c:883-947`（`prepare_rope` 一次性算出全 seq 的 cos/sin + reduced_cos/sin）；`:128-131`（`rope_cos/rope_sin/reduced_rope_cos/reduced_rope_sin` 常驻 `h3_dit` 结构体，per-step kernel 只读） | 已为零（此路径非主生产路径） |
| **DiT RoPE cos/sin 表（W4A8/INT8 主生产路径，`h3_dit_w4a8.c`）** | **否（待收益量化）**——每次 video forward 调 `h3_dit_w4a8_rope_tables_f32` 现算全表；仅 `inv_freq` 基频向量从权重常驻（`core->inv_freq`），未缓存展开后的 cos/sin 大表 | `h3_dit_w4a8.c:1921-1944`（`rope_tables_f32` 全表三重循环现算）；`:6628-6631`（`forward_video` 每调用即算，`:3484-3485` inv_freq 加载）；token-reduction 分支另算一份 `rcos/rsin`（`:6387-6389` `tr_build_reduced_rope`，同为每调用现算） | 表本身只依赖 (seq, 网格形状, spatial_scale)——clip 形状固定时**跨 4-step、跨 clip 均为真常量**。host F32 三重循环 seq×ROPE_HALF 量级，单次数十 ms 级；按 4-step×多 clip 计，**预估可回收 0.1–0.3s/run（量级估计，未 profile，待实测）**。bit-identical 可达（同一份浮点结果只算一次）。**优先级：P3——收益小但零风险、施工简单（per-shape memo）** |
| **VAE RoPE cos/sin 表（`h3_vae_int8_cuda.c`）** | **否**——每个 temporal chunk 调 `h3_vae_int8_prepare_rope_ex` host 现算，并逐 chunk H2D 拷贝进 `d_rope_cos/d_rope_sin` | `h3_vae_int8_cuda.c:2782-2800`（chunk 循环内 `prepare_rope_ex` + `cudaMemcpy H2D`）；`:2742-2752`（host buffer `h_rope_cos/h_rope_sin` 复用但内容每 chunk 重填）；`tests/test_h3_vae_rope.c`（存在 rope 专项测试） | 表只依赖 (latent_h, latent_w, chunk, origin)——clip 形状固定时全 clip 为真常量，**跨 clip 亦可复用**。现状 = 每 chunk 一次 host 三重循环 + 一次 H2D。量级估计：每 chunk ms~十 ms 级 × chunk 数（latent_t/chunk）×clip 数，**预估 0.05–0.2s/run（量级估计，未 profile）**。且把 H2D 移出 chunk 循环可顺带削 launch/同步气泡（族#6 叠加收益）。**优先级：P3，可与 VAE unpatch_d2h P0 施工同文件顺带做** |
| **文本 RoPE 位置（text token 的 pos_t=i, h=w=0）** | **部分**（并入上述整表计算，无独立缓存） | `h3_dit_w4a8.c:6610-6615`（每 forward 重写 `full_pt/full_ph/full_pw` 文本段）；`h3_nvfp4.c:761,1289`（`nvfp4_build_mrope_pos` per-call 构建） | 与整表缓存合并处理即可，无独立项 |
| **mrope_pos（多模态位置表，NVFP4 路径）** | **否（待确认使用频率）** | `h3_nvfp4.c:761,1289`（`nvfp4_build_mrope_pos` 每次构建） | 待源码确认调用频率（NVFP4 为探索线，非主生产）；量级小 |

## 汇总判定

1. **机制框架的「不变量缓存永远该做」在本仓已被执行了 2/3**：AdaLN（最大头，50 层 × per-step 投影）两条路径均已一次性预算——这部分**没有未收割空间**，此前的工程已做对了。
2. **真正未收割的是 RoPE 表，且呈路径分裂**：老的 BF16 路径已缓存，新的 W4A8/INT8 主生产路径反而每 forward 现算（DiT）+ 每 chunk 现算（VAE）。这是路径重写时丢掉的优化，不是新发现。
3. **量级诚实声明**：两项 RoPE 缺口的收益均为 ms~十 ms×次数 量级，**合计预估 0.15–0.5s/run（未 profile，纯结构推断）**——属于「零风险、小而确定」的尾料，**不改变 P0/P1 排序**；建议在 VAE P0 施工（同文件 `h3_vae_int8_cuda.c`）时顺带收割 VAE 侧，DiT W4A8 侧单独一个 per-shape memo 小拍。
4. **待源码确认项**：NVFP4 路径 `nvfp4_build_mrope_pos` 的调用频率与收益（探索线，暂不重要）；W4A8 路径 `rope_tables_f32` 是否在 per-step 循环内被调用（当前证据显示它在 `forward_video` 内、即 per-step；若实际调用点在外层则收益更低——**此项以 profile 为准**）。
