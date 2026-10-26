# 同配置 waffle 接缝回归门（离线）

> **日期：** 2026-10-26 Asia/Singapore。  
> **工具：** [`tools/waffle_same_config_gate.py`](../../tools/waffle_same_config_gate.py)  
> **用途：** 在**同一协议**下，比较两次 `generate.log` 的 `waffle: frame=… b16=…` 行，做接缝 / 回归门。  
> **不需要 GPU：** 脚本只读日志；真机跑数仍由 P0 / T3 验收另开。

---

## 1. 什么时候用

| 场景 | 用不用 |
| --- | --- |
| 同臂同 seed / steps / graph / TE 路径，改了无关代码后对照旧 log | ✅ |
| P0 INT8 金线「今日 binary vs 昨日 binary」同 META | ✅ |
| INT8 vs NVFP4、PDMD vs DMAD、turbo4 vs dense 跨配置比 waffle | ❌ 禁止（见 [dmad-noise-diagnosis](2026-10-07-dmad-noise-diagnosis.md)） |
| 代替目检 / 当作投产质量结论 | ❌ 只是接缝辅助；目检仍是门 |

默认阈值（可改）：`--mean-tol 0.15`、`--max-tol 0.40`、`--require-frames 5`（与历史「均值超对照 >0.15 触发退化线」口径对齐，**仅**用于同配置）。

---

## 2. 用法

```bash
# 内建合成自检（CI / make fewstep_host_tests）
python3 tools/waffle_same_config_gate.py --self-test

# 同配置两份 generate.log
python3 tools/waffle_same_config_gate.py \
  out/run_a/generate.log out/run_b/generate.log

# 更严
python3 tools/waffle_same_config_gate.py --mean-tol 0.05 --max-tol 0.15 \
  out/run_a/generate.log out/run_b/generate.log
```

解析的行形如：

```
waffle: frame=0 b16=1.100
waffle: frame=24 b16=0.900 INVALID_LOWENERGY block_ratio=1.2 gate=10.00
```

`frame=` 集合必须一致；任一帧 `INVALID_LOWENERGY` 有无不一致 → FAIL；均值 / 最大 |Δb16| 超阈 → FAIL。

建议先用 [`tools/p0_protocol_check_log.py`](../../tools/p0_protocol_check_log.py) 确认两份 log 协议一致，再跑本门。

---

## 3. 与 P0 / fewstep 的关系

- P0 视觉门：目检 + **同配置** waffle；本脚本把「同配置 waffle」自动化成 exit code。  
- fewstep：PDMD/DMAD 验收时只在**同一合同**两次跑之间用；勿跨 preset 比 b16。  
- Metal `h3` CLI **不产出** CUDA `waffle:` 合同日志路径；少步 / P0 CUDA 验收见 [`2026-10-26-fewstep-metal-boundary.md`](2026-10-26-fewstep-metal-boundary.md)。

---

## 3b. P0 runbook

命令块与 Phase A/B 接缝见 [`2026-10-26-p0-visual-gate-runbook.md`](2026-10-26-p0-visual-gate-runbook.md) §4.3.1。

## 4. 修订

| 日期 | 说明 |
| --- | --- |
| 2026-10-26 | 初稿：离线同配置接缝门 + `--self-test`。 |
| 2026-10-26 | 链到 P0 runbook §4.3.1 示例。 |
