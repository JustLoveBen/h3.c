# 少步合同自检设计（文档先行）

> **日期：** 2026-10-26。  
> **上游：** followup §4「合同自检」、[`2026-10-09-pdmd-dmad-loras.md`](2026-10-09-pdmd-dmad-loras.md)。  
> **性质：** 设计说明 + Phase-1 已落地（`h3_fewstep_contract.*`）。不改脏 NVFP4 chain / 无关 WIP。  
> **目标：** 错 preset / 错 shift / 错 sampler / 错 LoRA 身份时 **FAIL 或硬警告**，避免 turbo4 sigma 冒充 PDMD/DMAD 通过。

---

## 1. 为什么需要

2026-10-07 诊断已证明 LoRA 能挂上，但用 `preset=turbo4` 的 Euler 表  
`sigma≈[1, 0.973, 0.9231, 0.8, 0]` 跑 DMAD/PDMD 会出网格噪或无效验收。  
合同绑的是 **steps + video/audio shift + sampler + 原件身份**，不是「4 步 + 任意 LoRA」。

量化臂（NVFP4）与少步臂正交；自检只覆盖少步合同，不拦 INT8/NVFP4 切换。

---

## 2. 合同表（验收身份）

| 身份 | steps | sampler | video shift | audio shift | cfg | LoRA 原件偏好 | 本仓现状 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PDMD-4NFE | 4 | Euler | 12 | 3 | 1 | `pdmd_4NFE_lora` 未截断 / 非 Kijai 降秩优先 | 可跑；**正式验收未做** |
| PDMD-2NFE | 2 | Euler | 12 | 3（观感档曾用 6） | 1 | `pdmd_2NFE_lora` | r2/r3/r4 复验过合同；T3 仍未验证 |
| DMAD-4 | 4 | **lcm + re-noise** | 12 | **2** | 1 | `full_critic` 非降秩 | **采样器未实现**；Euler 跑 DMAD **不算**通过 |

已有生成日志锚点（勿删）：

- `sigma[0..N] video: ... audio: ...`
- `adaln_dual_sigma: video_shift=... audio_shift=...`
- LoRA：`pairs=` / `refiner_pairs=`（若有）

---

## 3. 建议的 FAIL / WARN 矩阵

实现时用显式旋钮，例如 `--fewstep-contract pdmd4|pdmd2|dmad4` 或 env `H3_FEWSTEP_CONTRACT=`。  
未设合同则行为与今日相同（兼容）。

| 条件 | 级别 | 行为 |
| --- | --- | --- |
| `pdmd4` 但 `steps!=4` | FAIL | 退出非 0 |
| `pdmd4` 但 `--preset turbo4`（或其它改写 sigma 的 preset） | FAIL | 退出；提示用无 preset + serving shift |
| `pdmd4` 且 audio shift ≠ 3（`H3_AUDIO_SIGMA_SHIFT`） | FAIL | 默认 3；显式错误值拒绝 |
| `pdmd2` 且 audio shift ∉ {3,6} | WARN→可配置 FAIL | 论文指标 3；观感档 6 需 META 标明 |
| `dmad4` 且 sampler 仍是 `velocity_euler`（无 re-noise） | FAIL | 在实现 lcm 前直接拒绝「合同通过」路径 |
| `dmad4` 且 audio shift ≠ 2 | FAIL | |
| 任意合同 + `H3_DIT_NVFP4=1` 且未 `H3_FEWSTEP_ALLOW_NVFP4=1` | FAIL | 禁止首轮交叉验收 |
| LoRA 路径名含 `rank_39` / `rank_57` / `kijai` 且合同为正式金线 | WARN | 降秩不挡跑，标 `non_gold` |
| turbo4 sigma 表与 PDMD student_grid(shift) 不一致却挂 PDMD 权重 | FAIL | 应用合同表校验打印的 `sigma[0..]` |

PDMD-4 期望视频格点（serving `linspace` + shift 12）应与发布 `student_grid(4,12)` 一致；落地时用**黄金向量**写进测试，而不是口头「4 步」。

---

## 4. 最小实现切片（后续代码，非本提交）

1. **文档 / 日志（已有）**：继续保证打印 `sigma[]` 与 `adaln_dual_sigma`（已在 `h3_cuda_generate.c`）。  
2. **离线检：** `tools/p0_protocol_check_log.py` 可扩 `--expect-audio-shift 3` 等（少步用，不塞进 P0 视觉门默认）。  
3. **进程内：** 在 `h3_generate_preset_apply` 之后、建 sigma 之后增加 `h3_fewstep_contract_check(...)`；FAIL 走现有 `fprintf` + `return 1`。  
4. **不要**在本轮改脏的 `h3_lora.c` 大 diff；合同检查放新文件 `h3_fewstep_contract.c` 更干净。

---

## 5. 与 P0 的关系

P0 视觉门 **不启用** 少步合同（无 LoRA、8-step、shift 12/默认 audio）。  
本设计服务 followup §6 第 5 项与 P1 PDMD-4 验收，避免重蹈 turbo4 错合同。

---

## 6. 修订

| 日期 | 说明 |
| --- | --- |
| 2026-10-26 | 初稿：矩阵 + 实现切片；代码未落地。 |
| 2026-10-26 | Phase-1：`h3_fewstep_contract.c` + `--fewstep-contract` / `H3_FEWSTEP_CONTRACT`；early+sigma 金向量；dmad4 无 lcm 硬拒；Mac `make h3_fewstep_contract_test`。 |
