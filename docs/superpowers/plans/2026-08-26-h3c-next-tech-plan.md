# h3.c 后续开发技术方案（2026-08-26）

> **后续优化方案见** [2026-09-11-h3c-next-opt-plan.md](./2026-09-11-h3c-next-opt-plan.md)（本文 P0–P3 已大多落地；多卡 / 快档由该文接管）  
> **For agentic workers:** 执行本方案时用 superpowers:subagent-driven-development（推荐）或 executing-plans，按阶段收口；步骤用 checkbox 跟踪。  
> **冲突裁决：** 金线 KPI / 默认质量路径以本文 + 实测报告为准；机制细节以 [tech-deepdive](./2026-08-25-h3c-vs-sglang-vllm-tech-deepdive.md) 为准；矩阵协议以 [accel-matrix](./2026-08-25-h3c-accel-backends-cache-matrix.md) 为准。  
> **2026-09-11 五方案普查（多卡 5090，不改金线）：** [2026-09-11-h3c-accel-survey-multigpu-5090.md](./2026-09-11-h3c-accel-survey-multigpu-5090.md)

**Goal:** 在守住 INT8+cudnn 8-step 金线的前提下，按 ROI 把「可测矩阵 → 独立 TeaCache → 块级 cache 思想 → 条件图/注意力质量」分阶段落地，且默认永不被损质臂污染。

**Architecture:** Tier-0 质量默认只走 dense cuDNN + 全 8 步；Tier-1 注意力与步/块 cache 一律 opt-in CLI，挂在现有 `h3_sdpa` registry 与 `h3_cuda_generate` 采样环；验收只认 cafe 金线配方 + warm mean + PPM MAE∩有声。多卡 / HTTP serving 另开里程碑，不进 75s KPI。

**Tech Stack:** C/CUDA `h3_cuda_generate`，`h3_sdpa` / `h3_cuda_device_caps`，`libh3_sol_cute.so` / `libh3_sage3.so`，4090 `build-cuda` / 5090 `build-cuda-sm120`。

**Spec / 底稿（必读，勿另起炉灶）：**

| 文档 | 用途 |
| --- | --- |
| [tech-deepdive](./2026-08-25-h3c-vs-sglang-vllm-tech-deepdive.md) | 机制 ROI、误差预算、Amdahl、抄什么/不抄什么 |
| [accel-matrix](./2026-08-25-h3c-accel-backends-cache-matrix.md) | 统一对比协议、缺口清单、harness |
| [omni 产品对照](./2026-08-25-h3c-vs-sglang-vllm-omni.md) | 产品位、公开秒数禁横比 |
| [4090-accel-plan](./2026-08-25-h3c-4090-accel-plan.md) | 金线地板、KEEP/OPT-IN/DROP |
| [sla-vs-h3c](./2026-08-25-sla-vs-h3c.md) | SLA/Turbo 禁接金线 |
| [multi-gpu-arch](./2026-08-25-h3c-multi-gpu-arch-speed.md) | SDPA Tier、插件硬失败、多卡另册 |

## Global Constraints（金线，全文隐含）

- 质量默认：**INT8 ConvRot DiT + fp16 VAE tile+deblock，cfg=1，steps=8，864×480，有声**；SDPA=`cudnn`；reuse / magcache / tea / fb / easy / sage / sol **全关**
- 墙钟地板（勿承诺再无损砍半）：**4090 ~75s** / **5090 ~66s**（cafe warm）
- 真时长：`--seconds 5` → 120@24fps；**禁止** `--frames` 假时长
- sol / sage：**opt-in 不过门** → 不得改默认；MagCache：**FAIL**，默认永关
- Tea / Easy / FB：**未独立** → 本方案按阶段补；阈值禁止照抄 50-step `0.10` / `0.17`
- **密码不入库**；未要求不 commit / 不 push
- 禁止把框架 **50-step / 4×H200 / SSIM 0.93** 倍数写进 8-step 金线承诺

---

## 1. 目标与非目标（KPI 分列，勿混）

### 1.1 目标 A — 守住金线 KPI（产品承诺）

| 项 | 定义 |
| --- | --- |
| 配方 | 上表 Global Constraints |
| Prompt | cafe 574 ids，seed `0x11` |
| 指标 | warm e2e ≈4090 **75s** / 5090 **66s**；默认 log：`sdpa=cudnn`、`magcache=0`、`reuse off` |
| 门 | 相对同机 cudnn 成片「不可辨」；checksum 允许不同 |

**成功：** 任何 PR 合入后金线仍可复现；默认路径零污染。

### 1.2 目标 B — 对齐框架 cache **能力**（工程能力，非秒数承诺）

| 项 | 定义 |
| --- | --- |
| 能力 | 独立 TeaCache（+AV veto）；保守 FBCache / Cache-DiT **块级思想**；统一矩阵可填「已实现/FAIL/否默认」 |
| 指标 | 同 cafe 配方下 `dit_evals/skips` + PPM MAE + 有声门；**默认仍关** |
| 非指标 | 不承诺 3×/4×；不与 Sol-Engine 4.44× / SGLang 1.40× 横比 |

**成功：** CLI 可开关、矩阵有诚实数字（含 FAIL）、不过门不进默认。

### 1.3 目标 C — Serving / 多卡（单独里程碑）

| 项 | 定义 |
| --- | --- |
| 范围 | HTTP 网关、Ulysses/SP、请求级 quality、多卡延迟账本 |
| KPI | **不进** 75s/66s 金线；另立延迟与合同文档 |
| 建议 | 需要 API 时优先 **调用** SGLang/vLLM 做边缘网关，本仓继续守引擎 |

**成功：** 有独立里程碑文档与验收；绝不改写金线秒数叙事。

### 1.4 非目标（本方案窗口）

- 用框架 50-step / 多卡秒数改写 75s 承诺
- SLA / Turbo-SLA / 未蒸馏 4-step 插 INT8 金线
- nested-loop Sol、假 sage2、Python/Comfy 当质量栈
- EasyCache 无 AV 分账进有声默认；MagCache 回潮默认
- 再抠 8-step dense 间隙到 40–50s；AdaLN 融合当主线
- 一次把全部 cache + 注意力质量 + 多卡做完

---

## 2. 现状基线（一页表）

| 轴 | 状态 | 证据 / 备注 |
| --- | --- | --- |
| Tier-0 cudnn 金线 | ✅ 产品默认 | 4090 q1 ~75.4s；5090 warm ~66s |
| `sol_cute` | ✅ Tier-1；视觉 FAIL | DiT −15%；MAE ~35；e2e≈0 |
| `sage3` | ✅ Tier-1（5090）；视觉 FAIL | DiT −14%；MAE ~27；prep CUDA 已落地 |
| nested `sol` / 参考 `sage2` | 🚫 产品禁 | DROP-FAKE / 假核 |
| MagCache lite | ✅ 接线；质量 FAIL | thresh=0.08 → MAE 52–68；默认关 |
| `magcache-calibrated` | ◐ Tea **风格**，非独立模块 | 绑在 mag 旗下 |
| 独立 TeaCache | ❌ | P1 主交付 |
| FBCache / Cache-DiT 块级 | ❌ | P2；勿与 `h3_w4a8_cache` first_block 权窗混淆 |
| EasyCache | ❌ | 有声高风险；排在 FB 之后或研究档 |
| reuse | ✅ 默认关 | 与 mag 互斥 |
| Vision host SDPA→cuDNN | ❌ KEEP 未做 | F2VA/L2VA；T2VA 金线 0 收益 |
| 加速矩阵 harness | ◐ 计划有草稿 | `scripts/h3_accel_matrix_smoke.sh` / `out/verify_accel_matrix/` — **P0 须补齐可跑** |
| 多卡 / serving | 路线图有，产品未成 | **M-GPU** 另册，不进本表 KPI |

分解（4090 cafe，勿再空跑）：`te≈11.6 / dit≈47.5 / vae≈9.3 / 其它≈7 → ~75s`。跳 1–2 步理论上限 −5.3～−10.6s（**未证质量**）。

---

## 3. 技术原则

1. **误差预算单点：** 金线预算主要花在 INT8 权；默认用 dense SDPA + 全步换回视觉。任何近似（稀疏 attn / skip / 量化）叠上去必须 **单独过门**，禁止未测叠默认。
2. **AV veto：** 有声路径上，视频轨 skip 不得静默套音频；Tea/FB/Easy 必须能抬高 delta 或硬否决（对齐 Mag 弱 veto，目标更强）。
3. **MAE 门（本仓尺子）：** 相对同机 cudnn PPM RGB MAE + 口型/道具目视 + 有声；**禁止**用框架 SSIM 0.93「放宽」默认。
4. **禁假核：** nested-loop Sol、参考 sage2、`_prepare_torch` Sol ROOT、静默假后端 — 缺 `LIB`/`ROOT` **硬失败**。
5. **禁假时长：** 真 `--seconds`；禁止 `--frames` 冒充 5s。
6. **禁把 50-step 倍数写进 8-step 承诺：** 4.44× / Cache-DiT 1.40× / Tea 3.18× 只作机制启发，不作金线 KPI。
7. **阈值重标定：** 8-step INT8 cafe 上从严扫；官方 `0.10`/`0.17`/`RDT=0.12` **不可照抄**。
8. **失败即关：** 任一门不过 → 矩阵写 FAIL、默认否、不拧到 20/50 步假装赢。

---

## 4. 分阶段路线图（P0→P3 + 另册）

并行标注：`‖` = 可与同阶段其它任务并行。每阶段目标 **1–2 周收口**。

### P0 — 矩阵填实 + Mag harness（约 3–5 人天，‖ 文档与脚本）

**动机：** 先有诚实总表与可复现 FAIL 证据，避免后续实现「无对照」。对齐 deepdive「下一单位 ROI 前先把尺子立住」。

**任务清单**

- [ ] **P0.1** 确认/补齐 `scripts/h3_accel_matrix_smoke.sh`（若缺失则按 accel-matrix §3 新建）
- [ ] **P0.2** 建立/刷新 `out/verify_accel_matrix/REPORT.md`：引用已有数字，**禁止编造**
- [ ] **P0.3** 填 A 类四格：4090 cudnn / sol_cute；5090 cudnn / sage3（链 `verify_480p_quality_full`、`verify_sol_vis_gate`、`verify_5090_bench`）
- [ ] **P0.4** harness arm=`magcache` / `magcache_cal`：cafe 全长，报 evals/skips + MAE；文档一句澄清「本仓 Mag ≠ 论文完整；calibrated ≠ 独立 Tea」
- [ ] **P0.5** nvidia_build / help：矩阵入口一句；密码仍仅 env

**改哪些模块**

| 路径 | 动作 |
| --- | --- |
| `scripts/h3_accel_matrix_smoke.sh` | 新建或补齐 |
| `out/verify_accel_matrix/REPORT.md` | 填表 + 链报告 |
| `tools/h3_cuda_generate.c` | 仅注释澄清（默认逻辑不动） |
| `docs/nvidia_build.md` | 一小节指向矩阵 |

**验证命令 / 矩阵行**

```bash
# 4090 示例（路径按本机 build / ROOT）
./scripts/h3_accel_matrix_smoke.sh --gpu 4090 --arm cudnn
./scripts/h3_accel_matrix_smoke.sh --gpu 4090 --arm sol_cute   # 需 H3_SOL_*
./scripts/h3_accel_matrix_smoke.sh --gpu 4090 --arm magcache --thresh 0.08
# 5090
./scripts/h3_accel_matrix_smoke.sh --gpu 5090 --arm cudnn
./scripts/h3_accel_matrix_smoke.sh --gpu 5090 --arm sage3     # 需 H3_SAGE3_*
```

矩阵行至少：`4090_cudnn` / `4090_sol_cute` / `5090_cudnn` / `5090_sage3` / `*_magcache*`。

**通过 / 失败**

| 通过 | 失败 |
| --- | --- |
| REPORT 四格+Mag 均有 warm 数字或显式引用；默认仍 cudnn | 编造秒数；把 Mag/sol/sage 写进默认；密码入库 |

**回滚：** 删/还原脚本与 REPORT；不改 generate 默认。

**依赖硬件/插件：** 4090 `build-cuda` +（sol 行）`H3_SOL_CUTE_LIB`+`H3_SOL_ATTN_ROOT`；5090 `build-cuda-sm120` +（sage 行）`H3_SAGE3_LIB`+`H3_SAGE3_ROOT`。无 GPU 时只允许「引用已有 out/」填表，不得伪跑。

**优先级：** P0 阻塞 P1 验收叙事（可与 P1 设计文档 ‖）。

---

### P1 — 独立 TeaCache + AV veto（约 5–8 人天）

**动机：** 官方 4.44× / Sol-Engine 主力是 cache 臂；vLLM 也标 Tea。本仓需在 **8-step INT8 有声**上诚实测「还有没有 skip 空间」——不是抄 50-step 阈值。

**任务清单**

- [ ] **P1.1** 接口锁定（推荐）：独立 `--teacache-thresh` + `H3_TEACACHE_MAX_CONSEC`；**少动** mag 语义；默认 0=off。互斥：`reuse>1` 或 mag>0 时 Tea 忽略或硬错（实现前钉死一种，推荐硬错防叠）
- [ ] **P1.2** 决策：rel_L1 / poly 累积；首末步强制 EVAL；中间 SKIP → 现有 `extrapolate_velocity` 路径
- [ ] **P1.3** **AV veto：** 音频 residual/|ratio−1| 可抬高 delta 或否决 skip
- [ ] **P1.4** cafe 从严 thresh 扫（从显著严于 0.08 起）；填矩阵 `teacache_*`
- [ ] **P1.5** help + `docs/nvidia_build.md` 一小节；矩阵协议脚注：非 4.44×

**改哪些模块**

| 路径 | 动作 |
| --- | --- |
| `tools/h3_cuda_generate.c` | Tea 决策 + CLI + 日志 `dit_evals/skips` |
| （可选）小头文件 | 若决策函数抽出，保持单文件也可 |
| `scripts/h3_accel_matrix_smoke.sh` | arm=`teacache` |
| `out/verify_accel_matrix/REPORT.md` | 新行 |
| `docs/nvidia_build.md` | CLI 说明 |

**验证**

```bash
./scripts/h3_accel_matrix_smoke.sh --gpu 4090 --arm teacache --thresh <strict>
# 对照同机 cudnn warm：phase_ms + PPM MAE(f0/24/48/90/119) + 音频波形/口型
```

**通过 / 失败**

| 通过 | 失败 |
| --- | --- |
| CLI 可开关；默认 off；至少一档填表（可 FAIL）；AV veto 有测例或日志证明 | 默认打开；无 AV 分账；照抄 0.10/0.17 当「无损」；MAE 不过仍改默认 |

**回滚：** 默认 thresh=0；或 `#if 0` / 旗标拆除接线；金线路径无 Tea 符号依赖。

**依赖：** 与 P0 同卡；**不**依赖 sol/sage 插件。

**优先级：** 金线加速实验主线 #1。

---

### P2 — 保守 FBCache / Cache-DiT 块级思想（约 5–10 人天；可与 P3 ‖ 前期调研）

**动机：** 对齐 SGLang/vLLM `quality=high` 的块级 residual 能力；8-step 上可能「不能不算这一步但仍省中间块」。首块代理在 H3 50L packed AV 上 **未证**——当研究档。

**任务清单**

- [ ] **P2.1** 设计探测：步内跑 first block 得 residual 相似度 → full eval vs reuse；对齐 Fn/Bn + warmup + continuous cap **思想**（不必抄 Python hooks）
- [ ] **P2.2** CLI：`--fbcache-thresh`（或 `--cache-backend fb`，与 P1 命名统一）；默认 off；与 Tea/Mag **互斥**
- [ ] **P2.3** 若需暴露 first-block residual：最小改动 `h3_dit_w4a8_*`（勿碰 `h3_w4a8_cache` 权窗语义）
- [ ] **P2.4** cafe A/B；诚实填表（预期 skip 少或 FAIL）
- [ ] **P2.5**（可选）Cache-DiT 多块 residual 实验分支 — 仅独立目录；自建 MAE∩音频门，**禁止 SSIM 过门**

**改哪些模块**

| 路径 | 动作 |
| --- | --- |
| `tools/h3_cuda_generate.c` | FB 决策插点 |
| `h3_dit_w4a8.c` / `.h` | 仅当必须暴露 first-block residual |
| harness + REPORT | arm=`fbcache` |

**验证：** 同 P1 cafe 协议；矩阵行 `fbcache_*`。

**通过 / 失败**

| 通过 | 失败 |
| --- | --- |
| 可跑通；默认 off；矩阵有结论 | 默认开；与 Tea 静默叠乘；用 SSIM 宣称过门 |

**回滚：** 关旗标；DiT API 保持向后兼容（residual hook 可选编译或空实现）。

**依赖：** P1 接口约定（互斥表）先钉；GPU 同 P0。关注 SGLang #33720 in-place residual=0 类坑 — 本仓须保证探测信号非恒 0。

**优先级：** 研究档；不过门不进产品叙事。

---

### P3 — 双轨选一或 ‖：注意力质量过门 **或** Vision KEEP（约 1–2 周/轨）

**动机：** deepdive：注意力替换 e2e 有效天花板≈0（质量路径）；条件图 vision→cuDNN 是 **唯一 KEEP 大头**（T2VA 不受益）。P3 允许 **并行两条**，但验收 KPI 分列。

#### P3-A — sol_cute / sage3 质量过门（Tier-1 长期线）

- [ ] dense 前缀步/层、sink、布局等旋钮实验（独立 `out/verify_*`）
- [ ] 目标：MAE→「不可辨」**才**讨论质量候选；过不了 **不改默认**
- [ ] 矩阵更新；CI 冒烟保持硬失败语义

**模块：** `tools/h3_sol_cute_embed.*`、`tools/h3_sage3_embed.*`、`h3_gpu_cuda.cu`、`h3_sdpa_registry.c`（以旋钮为主，忌假核）。

**依赖插件：** 4090 Sol ROOT（无 `_prepare_torch`）；5090 Sage3 ROOT。

#### P3-B — Vision host SDPA → cuDNN（KEEP，条件图）

- [ ] **E3.1** 孤立 profile：`h3_vision_cuda.c` host `sdpa()` 在 layers 31.5s 中占比；**<5s 则停刀**
- [ ] **E3.2** host vs cuDNN 数值对拍（max abs / rel L2）
- [ ] **E3.3** L2VA/F2VA 同图：`layers_ms` 显著下降；主题/锁帧不回归
- [ ] **禁止**用 T2VA 75s 当这刀对照

**模块：** `h3_vision_cuda.c` / `.h`；复用现有 cuDNN SDPA 路径；`h3_sdpa` 若需 vision 侧 resolve。

**通过 / 失败（分列）**

| 轨 | 通过 | 失败 |
| --- | --- | --- |
| P3-A | 独立报告 MAE 达不可辨档，且产品显式批准候选 | 未过门改默认 |
| P3-B | profile≥5s 嫌疑 → 换核后 layers 显著降 + 质量门过 | 占比&lt;5s 仍硬换；弄坏 T2VA 默认 |

**回滚：** 后端旗标 / vision 编译开关回到 host SDPA。

**优先级：** P3-B 对 F/L 产品 ROI 更清晰；P3-A 为 Tier-1 商用打磨。可 ‖。

---

### M-GPU / Serving — 单独里程碑（不进 75s）

**不纳入 P0–P3 人天承诺。** 触发条件：产品明确要 HTTP / 多卡延迟故事。

| 项 | 约定 |
| --- | --- |
| KPI | p50/p99、拓扑、offload RAM — **另表** |
| 路径建议 | 网关调 SGLang/vLLM；或另开 multi-GPU 实现计划（见 multi-gpu-arch） |
| 禁止 | 把 4×H200 ~74s / 2×5090 560s 写成「相对本仓加速」 |

---

## 5. 模块落点

| 层 | 路径 | 职责 |
| --- | --- | --- |
| CLI / 采样环 | `tools/h3_cuda_generate.c` | `--sdpa-backend`、`--magcache-*`、未来 `--teacache-*` / `--fbcache-*`；`extrapolate_velocity`；phase_ms |
| SDPA registry | `h3_sdpa.h`、`h3_sdpa_registry.c` | resolve；auto→cudnn；缺插件硬失败 |
| 设备能力 | `h3_cuda_device_caps.h` / `.c` | SM/FP4/插件探测；generate 不散落 `sm==120` |
| DiT | `h3_dit_w4a8.c` / `.h` | 前向；FB 时可选 first-block residual |
| Vision | `h3_vision_cuda.c` / `.h` | P3-B host SDPA→cuDNN |
| 插件 | `tools/h3_sol_cute_embed.*`、`tools/h3_sage3_embed.*` | Tier-1；prep 留 device |
| 信息工具 | `tools/h3_cuda_info.c` | caps + resolve 打印 |
| 构建 | `CMakeLists.txt`；`build-cuda` / `build-cuda-sm120` | 见 `docs/nvidia_build.md` |
| 矩阵脚本 | `scripts/h3_accel_matrix_smoke.sh` | P0 入口 |
| 验收产物 | `out/verify_accel_matrix/REPORT.md` + 各 `out/verify_*` | 禁止编造 |
| 短冒烟（可复用） | `scripts/h3_sdpa_backend_smoke*.sh`（若存在） | 1s/2step；**不能**代替 cafe 金线门 |

**环境变量（正式，勿写死本机绝对路径进源码）：**

- `H3_CUDA_SDPA_BACKEND`
- `H3_SOL_CUTE_LIB` / `H3_SOL_ATTN_ROOT` / `H3_TRITON_OVERLAY`
- `H3_SAGE3_LIB` / `H3_SAGE3_ROOT` / `H3_SAGE3_TOOLS`
- 未来：`H3_TEACACHE_MAX_CONSEC` / `H3_FBCACHE_*`
- SSH 主机：`H3_SSH_4090`（默认 `haiyang@192.168.31.34`）/ `H3_SSH_5090`（默认 `yongbin@192.168.31.238`）；见 `docs/nvidia_build.md`
- SSH 密码：仅 `H3_SSH_PASS_*` 类 env，**永不入 repo**

---

## 6. 风险与决策门

| 风险 | 决策门 | 动作 |
| --- | --- | --- |
| 8-step 蒸馏后再 skip → 轨迹漂（Mag p4 / LightX2V #579） | MAE∩口型∩音频任一门 FAIL | 关旗标；矩阵 FAIL；不二次拧 thresh 捞默认 |
| Tea/FB 与 INT8 误差双花预算 | 仅 opt-in；默认 dense 全步 | 进默认需产品改口 + 不可辨门 |
| EasyCache 伤 H3 音频（#15326） | 无 AV 分账 | **拒绝合并** |
| Sol/Sage 核快片不过 | 已测 MAE FAIL | 保持 Tier-1；质量候选另批 |
| TE 抖动吃掉 DiT 收益 | 同时报 DiT + warm mean±std | 不以单次 cold e2e 决策 |
| Cache-DiT in-place residual=0 | 探测信号自检 | 恒 0 → 实现 FAIL，勿宣称加速 |
| 指标混用 4.44× / SSIM | 文档与 REPORT 脚注 | PR 审查否决混谈 |
| Vision 刀收益不足 | E3 孤立 SDPA &lt;5s | **停刀**，不硬换 |
| 产品要「大约 4×」 | 必须承认换 50-step BF16 基线 | 另开产品档，不改金线 |

**阶段门（建议）：**

```
P0 REPORT 齐 → 开 P1 实现
P1 矩阵有数字（可 FAIL）→ 开 P2
P2/P3 过「不可辨」→ 才上产品评审「是否候选默认」
默认变更 = 显式产品决策，本方案不自动改
```

---

## 7. 明确不做

1. 把 MagCache / Tea / FB / Easy / sol / sage **默认打开**
2. nested-loop Sol 再 bench；假 sage2 当产品；sparse `_prepare_torch` 当 Sol ROOT
3. SLA / Turbo-SLA / 未蒸馏 4-step / LightWan 80× **插 8-step INT8 金线**
4. 用框架 50-step、4×H200、SSIM、Sol-Engine 4.44× **改写 75s/66s**
5. EasyCache **无 AV 分账**进有声路径
6. Python/Comfy 替换 C 质量生成栈
7. 再抠 8-step dense 间隙到 40–50s；AdaLN/gate 融合排主线
8. Radial / SVG2 / NVFP4 DiT / FP8 DiT / Latent Upscaler 进本窗口金线
9. 多卡 Ulysses / HTTP serving **混进金线 KPI**
10. 密码、本机绝对路径默认值 **入库**

---

## 8. 阶段总表（执行摘要）

| 阶段 | 主题 | 约人天 | 关键验收 | 默认变？ |
| --- | --- | --- | ---: | --- |
| **P0** | 矩阵填实 + Mag harness | 3–5 | REPORT 有 cudnn/sol/sage/mag 行 | 否 |
| **P1** | 独立 TeaCache + AV | 5–8 | CLI off；cafe 至少一档数字 | 否 |
| **P2** | FB / Cache-DiT 思想 | 5–10 | 可跑；矩阵结论（可 FAIL） | 否 |
| **P3-A** ‖ | sol/sage 质量 | 5–10 | MAE→不可辨才候选 | 否* |
| **P3-B** ‖ | vision→cuDNN | 3–7 | E3 门 + F/L 降耗 | 否（T2VA） |
| **M-GPU** | serving/多卡 | 另册 | 独立 KPI | N/A |

\*仅当产品显式批准「质量候选」后另 PR 改默认。

---

## 9. 执行交接

方案已写入 `docs/superpowers/plans/2026-08-26-h3c-next-tech-plan.md`。

**建议执行顺序：** P0 → P1 →（P2 ‖ P3-B）→ P3-A；M-GPU 另触发。

**执行选项：**

1. **Subagent-Driven（推荐）** — 每阶段新 subagent + 两段 review  
2. **Inline Execution** — 本会话按阶段 checkpoint

本轮 **只产出方案与交叉链接**；不改 generate、不跑 GPU、不 commit。
