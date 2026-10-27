# DISARM: 无意义 NVFP4→INT8 转换/旁路（2026-10-27）

用户硬性：时间只花在有意义优化。Sep-13 host dequant 是过渡，不是目标。

## 已审核 · DISARM / 勿做

| 项 | 入口 | 处置 |
|----|------|------|
| Host dequant NVFP4→逐行 INT8 → cublasLt INT8 | `--preset nvfp4`；`h3_dit_w4a8_open_nvfp4` + `h3_dit_nvfp4_dequant_to_i8`；路径名含 `nvfp4` 自动 want | **DISARM**：preset 直接 fail；默认不再因 path hint 开 dequant；仅 `H3_DIT_NVFP4_HOST_DEQUANT=1` 考古 opt-in |
| INT8 sidecar 预计算 | `tools/h3_nvfp4_int8_sidecar`；`*.int8_sidecar/` | **DISARM**：工具默认 exit 2，需 `H3_FORCE_NVFP4_INT8_SIDECAR=1` |
| NVFP4 FC1→FC2 chain | `H3_DIT_NVFP4_CHAIN*` / `chain.cu` | **DISARM/WIP**：runtime unset；不测不启 |
| 为跑通而 naive requant / 重训 NVFP4 | 任意「先转 INT8 再声称 NVFP4」 | **勿做** |
| 重测 pruned→int8 convert e2e | `run_nvfp4_int8_convert.sh` 等 | **停止**；脚本勿作默认入口 |

## 保留 · 正路径

| 项 | 入口 | 说明 |
|----|------|------|
| 现成权重原生 FP4 GEMM | INT8 ConvRot `--dit` + `H3_DIT_NVFP4=1` + `H3_DIT_NVFP4_FILE=<…_nvfp4_w4a4.safetensors>` | 目标；C001 blank 需修/覆盖，不是改走 convert |
| KEEP_INT8 混合 | `H3_DIT_NVFP4_KEEP_INT8=qkv,out,…` | 敏感层**真** INT8 GEMM，其余 FP4；找最大可过 FP4 覆盖 |
| INT8 ConvRot 金线 | `--dit …_int8_convrot`，无 NVFP4 env | 对照/金线 |
| 无损墙钟 | `H3_DIT_EPI_FUSE_SL` tiled 等（EPI 默认 OFF，waffle 绿+正 WC 再议） | 与 quant 旁路无关 |

金线仍是 INT8 ConvRot；NVFP4 实验臂不得 default-on。
