# accel_survey — 隔离下载（2026-09-11）

本目录只放外部加速方案源码/摘录，**不进产品默认路径**，巨型权重不入库。

| 方案 | 本地路径 | 来源 | 备注 |
| --- | --- | --- | --- |
| FastVideo / FastH3 | `FastVideo/` | [hao-ai-lab/FastVideo](https://github.com/hao-ai-lab/FastVideo) | `git clone` 对 `github.com:443` 失败时改 `codeload.github.com` tarball |
| VDN-H3 | `vdn-minimax-h3/` | [OpenVDN/vdn-minimax-h3](https://github.com/OpenVDN/vdn-minimax-h3) | depth/tarball；权重在 HF `OpenVDN/vdn-minimax-h3`，未下载 |
| RunningHub Lightning | `MiniMax-H3-MultiGPU-Lightning/` | [RH-RunningHub/MiniMax-H3-MultiGPU-Lightning](https://github.com/RH-RunningHub/MiniMax-H3-MultiGPU-Lightning) | 含 pinned SGLang 快照 `f8cbf000f4a5` |
| Sol Super Acceleration | `sol-engine-excerpts/` | [NVlabs/Sana `sol-engine`](https://github.com/NVlabs/Sana/tree/sol-engine) + [项目页](https://nvlabs.github.io/Sana/Sol-Engine/H3-Super-Acceleration/) | 完整仓未整树 clone；摘录 `models/minimax_h3/super_acceleration/` |
| Fal H3 Max | （无仓） | [fal.ai/minimax-h3-max](https://fal.ai/minimax-h3-max) · [blog](https://blog.fal.ai/introducing-h3-max-by-fal/) | 闭源 serving；见计划文档 |
| comfyui-SelfLift | `comfyui-SelfLift/` | [facok/comfyui-SelfLift](https://github.com/facok/comfyui-SelfLift) `@19ec540` | 只读浅克隆；**禁止嵌进生产**。分析见 `docs/superpowers/plans/2026-09-13-h3c-selflift-comfy-analysis.md` |
| FreeToken | `FreeToken/` | [FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken) `@9535656` | 只读浅克隆；**MoE LLM serving，不是视频 DiT**。分析见 `docs/superpowers/plans/2026-09-13-h3c-freetoken-analysis.md` |

对照分析：浅层 `docs/superpowers/plans/2026-09-11-h3c-accel-survey-multigpu-5090.md` · 机制深潜（含 RH 多卡 6000D）`docs/superpowers/plans/2026-09-11-h3c-accel-survey-tech-deepdive.md` · 少步质量 `docs/superpowers/plans/2026-09-11-h3-few-step-quality.md`
