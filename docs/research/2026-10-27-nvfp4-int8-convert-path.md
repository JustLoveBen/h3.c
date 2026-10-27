# NVFP4 INT8 convert path — **DISARMED** (2026-10-27)

> **停止重测。** Sep-13 `--preset nvfp4` + host dequant→INT8 是过渡旁路（cafe 能出片，dequant ~423s，**零原生 FP4 收益**）。
>
> 正路径见 `2026-10-27-nvfp4-disarm-convert-bypasses.md` 与 `2026-10-27-nvfp4-keep-int8-hybrid.md`：
> 现成 w4a4 → 原生 FP4 GEMM + 可选 KEEP_INT8；勿再下载/重跑 pruned→int8 convert。

历史事实（勿当 TODO）：2026-09-13 cafe OK；2026-10-27 C001 是**另一条**（w4a4+原生 FP4，视觉 blank），≠ convert 未测。
