# fewstep 合同机读表（JSON）

> **日期：** 2026-10-26 Asia/Singapore。  
> **产物：** [`tests/fixtures/fewstep_contracts.json`](../../tests/fixtures/fewstep_contracts.json) + [`fewstep_contracts.schema.json`](../../tests/fixtures/fewstep_contracts.schema.json)  
> **源码：** `h3_fewstep_contracts_json_fprint` in `h3_fewstep_contract.c`；`./dump_fewstep_contracts`  
> **校验：** `python3 tools/fewstep_contracts_validate.py`；`make fewstep_host_tests` 含 `--check` fixture。

## 用途

给脚本 / CI / 文档生成器一份与 `check_early` / `--fewstep-preset` **同口径**的合同表（steps、sampler、shift、cfg、禁 turbo4 列表），无需解析 C。

```bash
./dump_fewstep_contracts                 # pretty JSON
./dump_fewstep_contracts --compact
./dump_fewstep_contracts --check tests/fixtures/fewstep_contracts.json
python3 tools/fewstep_contracts_validate.py --self-test
```

改合同逻辑后：更新 C 表 → 重跑 dump 覆盖 fixture → host tests。

## 与 help 文本

`h3_fewstep_help_fprint` 提供与 CLI 一致的 `--fewstep-contract` / `--fewstep-preset` / `--sampler` 说明；`./print_fewstep_help` 打印。`tools/h3_cuda_generate.c` 若有未提交 WIP，**不必**为改 help 去 stage 该文件——以本模块为准，generate 可后续改为调用同一函数。
