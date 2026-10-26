# C061 — generate 启动内联 env 合同（2026-10-27）

> **状态：** 离线开发落地（本提交）；**未**在 238 GPU 验收。  
> **动机：** DeepInfer T03/T12 — 配置即合同；测错比测慢更贵。离线 Python 打印机不够，必须在 `h3_cuda_generate` 启动日志里留下同一合同块。

---

## 1. 行为

启动解析 CLI / fewstep / sampler 之后、重活之前，打印：

```
# H3_ENV_CONTRACT  arm=int8|nvfp4  expect_p0=0|1
env H3_DIT_NVFP4=...  # role rule=...
...
cli seed=0x... skip_audio=N fewstep_preset=... fewstep_contract=... sampler=... sdpa=... sdpa_graph=N te_path=live|H3_TEXT_H_LOAD
```

若 `H3_EXPECT_P0_VISUAL=1`：按与 `tools/print_h3_env_contract.py --expect-p0-visual` 相同的 P0 规则检查；失败打印 `tag=FAIL_ENV` 并 `return 1`。  
**默认仍是只打印、不失败。**

共享实现：

| 文件 | 角色 |
|---|---|
| `h3_env_contract.h` / `.c` | `h3_env_contract_fprint` / `h3_env_contract_check_p0` |
| `tools/h3_cuda_generate.c` | 启动内联调用 |
| `tools/print_h3_env_contract_host.c` | host 对照打印（Makefile） |
| `tools/print_h3_env_contract.py` | **保留** Python 离线 twin |
| `tools/p0_protocol_check_log.py` | 识别 `# H3_ENV_CONTRACT`；`--expect-env-contract` |

CONTRACT 列表与 P0 规则与 Python 保持同步（改一处请改两处）。

---

## 2. 同批附带

| ID | 内容 |
|---|---|
| **C074** | 合同/环境/OOM 退出前 `tag=FAIL_CONTRACT|FAIL_ENV|FAIL_OOM_DIT|FAIL_OOM` |
| **C040** | `H3_VRAM_LEDGER=1` → `vram_ledger: point=before_dit\|peak_dit\|before_vae\|after_vae`（默认关；插桩 only） |
| **C070** | [`2026-10-27-cuda-sync-audit.md`](2026-10-27-cuda-sync-audit.md) 审计页 |

未做：`FAIL_OOM_VAE`（generate 内无独立「VAE OOM」fprintf 站点）；`H3_FORBID_HOT_SYNC`；GPU 墙钟。

---

## 3. Host 自检

```bash
make fewstep_host_tests
# 含：h3_env_contract_test、print_h3_env_contract_host、p0 --self-test、python print_h3_env_contract.py
```

---

## 4. 修订

| 日期 | 说明 |
|---|---|
| 2026-10-27 | 初版落地；无 GPU。 |
