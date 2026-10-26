# 少步一键 preset（`--fewstep-preset`）

> **日期：** 2026-10-26（Asia/Singapore）。  
> **上游：** [`2026-10-26-contract-selfcheck-design.md`](2026-10-26-contract-selfcheck-design.md)、[`2026-10-26-followup-dev-plan.md`](2026-10-26-followup-dev-plan.md) §4/§6 T3。  
> **性质：** 开发切片说明；**不替代** GPU 视觉验收。

## 动机

合同自检（`--fewstep-contract`）只**校验**，不负责填 steps / cfg / shift / sampler。  
一键验收时容易漏 `H3_AUDIO_SIGMA_SHIFT` 或误叠 `turbo4`。  
`--fewstep-preset`（或 `H3_FEWSTEP_PRESET`）把合同表绑成一次展开，仍走同一套 `h3_fewstep_contract_check_*`。

**不**并入 `--preset turbo4|taomate|…`：那些会改写 sigma；少步合同已硬拒。

## 展开表

| preset | steps | sampler | video / audio shift | cfg | 打开的合同 |
| --- | --- | --- | --- | --- | --- |
| `pdmd4` | 4 | `velocity_euler` | 12 / 3 | 1 | `pdmd4` |
| `pdmd2` | 2 | `velocity_euler` | 12 / 3 | 1 | `pdmd2` |
| `dmad4` | 4 | `lcm_renoise` | 12 / **2** | 1 | `dmad4` |

- audio：若环境尚无 `H3_AUDIO_SIGMA_SHIFT`，则 `setenv(..., 0)` 写入上表；已存在则不覆盖（错值由合同 FAIL）。  
- 显式 `--steps` / `--cfg` / `--sampler` **优先保留**（随后合同可能 FAIL）。  
- 与 `--fewstep-contract` 同名可；不同名 → FAIL。  
- 与 `--preset turbo4|taomate` → FAIL（CLI 早拒）。  
- **仍须** `--lora` 原件；preset 不装权重。

## 用法（CUDA generate，未在本机跑 GPU）

```bash
# PDMD-4NFE（Euler + shift 12/3）
./h3_cuda_generate ... --fewstep-preset pdmd4 --lora /path/pdmd_4NFE_lora.safetensors

# DMAD-4（lcm_renoise + shift 12/2）
./h3_cuda_generate ... --fewstep-preset dmad4 --lora /path/full_critic.safetensors

# 等价 env
H3_FEWSTEP_PRESET=pdmd2 ./h3_cuda_generate ... --lora /path/pdmd_2NFE_lora.safetensors
```

日志应依次出现：

1. `fewstep_preset: id=... steps=... cfg=... video_shift=... audio_shift=... sampler=... contract=...`
2. `fewstep_contract: id=...`（early 自检）
3. `fewstep_bound: contract=... steps=... cfg=... video_shift=... audio_shift=... sampler=...`
4. 去噪前 `sigma[...]` / `adaln_dual_sigma`；合同开启时还有 `fewstep_contract: sigma_ok ...`

## 单测

```bash
make h3_fewstep_contract_test && ./h3_fewstep_contract_test
```

## 非目标

- 不改 turbo4 语义；不提交 NVFP4 chain WIP；不在 238 跑生成。  
- 不把 LoRA 路径写进 preset（原件因人而异）。
