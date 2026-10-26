# turbo4 × fewstep 禁配日志检（离线）

> **工具：** [`tools/fewstep_forbidden_combo_check.py`](../../tools/fewstep_forbidden_combo_check.py)  
> **互补：** `p0_protocol_check_log.py` 已在 `fewstep_bound speed_preset=turbo4` 上 FAIL；本脚本覆盖**不完整 log / 命令回显**。

## FAIL

- `fewstep_bound` 的 `speed_preset` ∈ {turbo4, taomate, fast4}
- `fewstep_contract: ... preset=turbo4|…` 且无拒绝文案
- 同时出现 fewstep_* 行与 turbo4/taomate 预设横幅，且无拒绝
- 同一 transcript 里同时有 `--preset turbo4` 与 `--fewstep-*`，且无拒绝

## PASS

仅有正确拒绝（「互斥」「拒绝 --preset」「会改写 sigma」等），或干净 fewstep 合同 log。

```bash
python3 tools/fewstep_forbidden_combo_check.py --self-test
python3 tools/fewstep_forbidden_combo_check.py out/run/generate.log
```
