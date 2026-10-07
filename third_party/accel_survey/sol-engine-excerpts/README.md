# MiniMax-H3 Super Acceleration — 本地摘录（非整仓）

完整运行时在 [NVlabs/Sana `sol-engine`](https://github.com/NVlabs/Sana/tree/sol-engine/models/minimax_h3/super_acceleration)。本文件只记录本轮 WebFetch 到的合同，避免把整个 Sol-Engine 树拉进 h3.c。

公开页：https://nvlabs.github.io/Sana/Sol-Engine/H3-Super-Acceleration/

## 能拿到

- 项目页机制与 GB200 秒数（5s 768p **6.85s / 22.2× vs 已发表 SGLang**；页面明确 **非无损**）
- 仓内目录 `models/minimax_h3/super_acceleration/`：README、STAGE2_CONTRACT、BENCHMARK_REFERENCE.json、handoff_protocol.py、run_gb200.sh、stage1/stage2/vendor/tests
- 相邻：`models/minimax_h3/`（含 RTX5090 50-step Sol-Engine 4.52×，**不是** Super Accel）、`models/ltx2.5-refiner/`、`models/minimax_h3/Sol-H3/`（4-step FastH3 + Ulysses，B300）

## 拿不到 / 未下

- 整树 `sol-engine`（体积大、含多模型 runtime）
- 权重 / LightX2V LoRA / LTX-2.5 / TAEH3 / TAEHV（`BENCHMARK_REFERENCE.json`：`weights_or_outputs_bundled: false`）
- 感知质量门（仓内 `perceptual_quality_gate_run: false`）
- 与博客 22.2× **同配方** 的 matched baseline（仓内 `speedup_claim: null`）
- 5090 / 消费卡 Super Accel 配方（Stage-2 Sol 核合同钉 `cute_sm100`）

## v2 合同要点（job 6304303）

- Stage1：1×GB200，H3 FL2VA + LightX2V 4-step LoRA（`4step_v0.1_544p`，scale **0.0625**），896×512，124f，4 次 Transformer forward，**cache 关**，compile 开，**TAEH3** decode
- Handoff：loopback TCP，BF16 视频 `[1,3,121,384,672]` + FP32 PCM `[1,2,161333]`；**不是中间 MP4**
- Stage2：另 1×GB200，原版 LTX Video VAE encode + x2 latent upsample，3 updates，layer0 dense / 1–47 Sol-Attn `cute_sm100`，**TAEHV** 出片；**扔掉 Stage2 音频，mux 回 Stage1 的 H3 PCM**
- 拓扑：每请求 2 GPU；4 GPU 节点 = **两对独立流水**，不是 TP/SP/模型并行
- 热 E2E 中位 **6.761s**（S1 4.292 + S2 service 2.447）
