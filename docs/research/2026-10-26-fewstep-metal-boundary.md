# Fewstep / sampler：Metal vs CUDA 边界 + 剩余 TODO

> **日期：** 2026-10-26 Asia/Singapore。  
> **目的：** 一页钉死「少步合同只在哪条 CLI」以及开发侧还剩什么（不含 GPU 验收）。

---

## 1. 边界（硬）

| 路径 | fewstep / LoRA / `lcm_renoise` | 说明 |
| --- | --- | --- |
| **CUDA** `h3_cuda_generate`（CMake / 238） | ✅ `--fewstep-preset` / `--fewstep-contract` / `--sampler` / `--lora` | T3 与 DMAD 验收**唯一**入口 |
| **Metal** `./h3`（本 Makefile `all`） | ❌ 未接 | 无 LoRA 加载、无 fewstep 合同、无 `lcm_renoise`；勿在 Mac 上假验收少步 |

相关已有说明：[`2026-10-26-fewstep-preset.md`](2026-10-26-fewstep-preset.md)、[`2026-10-26-dmad-lcm-renoise-sampler.md`](2026-10-26-dmad-lcm-renoise-sampler.md)。

Host 单测（无 GPU）：

```bash
make fewstep_host_tests
```

打包：`h3_fewstep_contract_test`、`h3_sampler_test`、`dump_fewstep_student_grid`、`h3_waffle_test`、`waffle_same_config_gate.py --self-test`。

---

## 2. 已落地（开发，无需 238）

- 合同自检 + audio-env 冲突提前 FAIL  
- `--fewstep-preset` 一键绑定  
- `lcm_renoise` 采样器 + dmad4 自动选  
- `student_grid` σ 金表 + dump 工具  
- P0 日志协议检 / TE checksum / timing 行模板  
- 同配置 waffle 接缝门（本文同批）

---

## 3. 剩余 TODO（开发视角；GPU 项另授权）

| 项 | 状态 | 备注 |
| --- | --- | --- |
| Metal CLI 接 fewstep/LoRA/sampler | **不做**（明确非目标） | 验收永远走 CUDA generate |
| 与 Comfy 逐 bit 噪声对齐 | 开放 | 本仓 `fill_normal_f32`；bit-identical 非门 |
| PDMD 误用 `lcm_renoise` | 合同自检已挡 dmad 错配；PDMD 仍强制 Euler | 勿把 lcm 套 PDMD |
| PDMD-4NFE / DMAD `full_critic` GPU 目检+waffle | **待授权** | 开发骨架齐，238 未跑 |
| P0 INT8/NVFP4 视觉门 | **待授权** | runbook 已写 |
| NVFP4 链式 WIP 提交 | **待确认** | 工作区脏文件勿擅自动 |

---

## 4. 修订

| 日期 | 说明 |
| --- | --- |
| 2026-10-26 | 初稿：Metal/CUDA 边界 + TODO 表 + `make fewstep_host_tests`。 |
