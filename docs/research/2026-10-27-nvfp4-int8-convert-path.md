# NVFP4 → INT8 convert 路径状态（回答「int8 convert 测了吗」）

**Date:** 2026-10-27 Asia/Shanghai  
**结论先行：**

| 问题 | 答案 |
|------|------|
| 本轮 10-27 C001 测的是 convert 吗？ | **否**。C001 = `minimax_h3_fl2va_nvfp4_w4a4.safetensors` + `H3_DIT_NVFP4=1` **原生 FP4 GEMM**，waffle `INVALID_LOWENERGY` / 视觉 blank。 |
| 历史上 convert 测过吗？ | **是（2026-09-13）**：`--preset nvfp4` + `minimax_h3_fl2va_pruned_nvfp4.safetensors`（`nvfp4_awq_like`），host dequant NVFP4→逐行 INT8 → cublasLt INT8；cafe 能出片；dequant ~423–424s / 480p wall ~479s（见 `docs/superpowers/plans/2026-09-13-h3c-freetoken-analysis.md`）。 |
| 现成包还在 238 盘上吗？ | **2026-10-27 盘点：不在。** `models/diffusion_models/` 无 `minimax_h3_fl2va_pruned_nvfp4.safetensors`；仅有 `fastvideo_fasth3_8step_v2_pruned_nvfp4.safetensors`（同 awq_like 布局，但非 FL2VA 金线骨干）与 `minimax_h3_fl2va_nvfp4_w4a4.safetensors`（另一条）。`out/verify_nvfp4` 目录空。 |
| 本轮是否重跑 convert？ | **进行中**：从 HF（vonkaiser / lilcheaty）拉回 pruned_nvfp4，用当前 main 的 `--preset nvfp4` 重跑协议+waffle+OR。 |

## 两条路径勿混

1. **INT8 convert（`--preset nvfp4`）**  
   - 权：社区 `*_pruned_nvfp4.safetensors`（U8 nibble + F8_E4M3 scale + F32 scale_2）  
   - 运行：`h3_dit_nvfp4_dequant_to_i8` → INT8 GEMM（零原生 FP4 算力收益；热路径代价在 host dequant，可 sidecar 缓存）  
   - 历史：能出片，慢。

2. **原生 W4A4 GEMM（C001 / `H3_DIT_NVFP4=1`）**  
   - 权：`minimax_h3_fl2va_nvfp4_w4a4.safetensors`（`tools/quant_dit_nvfp4.py` 从 INT8 ConvRot offline requant）  
   - 运行：cuBLASLt FP4 GEMM  
   - 结果：单层能量对齐，多层 ~13% rel → blank；**不是** convert 路径。

## 入口（当前 main）

```bash
./build-cuda-sm120/h3_cuda_generate --preset nvfp4 \
  --dit /mnt/nvme1n1/workspace/models/diffusion_models/minimax_h3_fl2va_pruned_nvfp4.safetensors \
  ... # 同 cafe / turbo4 协议帧尺寸，勿叠 H3_DIT_NVFP4=1
```

日志应见：`dit_format: NVFP4 (非金线；dequant→INT8 GEMM)`。  
**禁止**把 `H3_DIT_NVFP4=1` + w4a4 sidecar 当成 convert 测试。

## 下一步（不重训）

1. 拉回 / 确认 pruned_nvfp4 → 本轮 convert 重测（本文件后续补 238 表）。  
2. 若 convert 视觉绿：优化 dequant（sidecar 常驻 / GPU dequant），勿默认换金线。  
3. 并行：`H3_DIT_NVFP4_KEEP_INT8` 混合精度（w4a4 路径，不重训）。


## 238 live (2026-10-27)

| item | result |
|------|--------|
| `minimax_h3_fl2va_pruned_nvfp4.safetensors` on disk | **NO** (models/diffusion_models empty of this name; only `.lock` + hf `.incomplete`) |
| other nvfp4 on disk | `minimax_h3_fl2va_nvfp4_w4a4.safetensors` (C001 path); `fastvideo_fasth3_8step_v2_pruned_nvfp4.safetensors` (awq_like, wrong backbone) |
| download | hf-mirror.com `vonkaiser/MiniMax-H3-NVFP4` → target Content-Length 12528636800 ≈12.53G |
| convert probe (substitute) | fastvideo pack → `dit_nvfp4_probe: PASS` dequant_qkv ok ~30.57s |
| convert e2e cafe | blocked until pack lands; runner `out/scripts/run_nvfp4_int8_convert.sh` (`--preset nvfp4`, H3_DIT_NVFP4 unset) |
| Sep-13 historical | same pack+`--preset nvfp4` → cafe OK; host dequant ~423–424s |

**「int8 convert测试了吗」:** historically **YES** (2026-09-13). This turn re-verify: weight missing → downloading → e2e pending. Oct-27 C001 ≠ convert (was w4a4+native FP4, blank).
