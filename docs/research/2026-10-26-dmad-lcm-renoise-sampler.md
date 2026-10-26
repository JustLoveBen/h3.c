# DMAD LCM + simple re-noise sampler（2026-10-26）

状态：已在 CUDA `h3_cuda_generate` 落地 `lcm_renoise`；**尚未在 238 做 GPU 视觉验收**（本提交 develop-only）。

## 作者合同（对照）

见 `2026-10-09-pdmd-dmad-loras.md`：DMAD 官方图是 ComfyUI **`lcm` + scheduler `simple`**，audio shift **2**，video shift **12**，NFE **4**，CFG **1**。Euler 不是工作点。

## Comfy 语义 → 本仓公式

Comfy `ModelSamplingDiscreteFlow` + **CONST**（与 FLOW matching 一致）：

- 模型包装：`model_output = −v`（Comfy 对 raw velocity 取负）
- `calculate_denoised`: `denoised = X − σ · model_output = X + σ · v`（v = 本仓 DiT 原始输出）

`SamplerLCM.sample`（逐步）：

1. `denoised = model(X, σ)` → 即 x0 估计 = `X + σ·v`
2. 若本步为最后一步（下一 σ = 0）：`X ← denoised`
3. 否则：`noise ~ N(0,I)`，`X ← noise_scaling(σ′, noise, denoised)`  
   CONST：`noise_scaling = σ′·noise + (1−σ′)·denoised`

合并为本仓一步（`h3_sampler_lcm_renoise_step_f32`）：

```
x0 = X + σ · v
if σ′ ≈ 0:  X = x0
else:       X = (1 − σ′) · x0 + σ′ · ε
```

对比默认 `velocity_euler`：`X += (σ − σ′) · v`（无再注入噪声）。中段 σ′>0 时两者一般不相等。

## API

| 旋钮 | 值 |
|------|-----|
| `--sampler` / `H3_SAMPLER` | `velocity_euler`（默认）\| `lcm_renoise`（别名 `lcm`、`lcm+simple`） |
| `--fewstep-contract dmad4` | 要求 `has_lcm_renoise`；若未显式设 sampler，generate **自动**选 `lcm_renoise` |
| `H3_AUDIO_SIGMA_SHIFT` | dmad4 必须 **2** |

主路径（非 taomate / 非 lift transition）的每步 audio+video 都走同一 sampler。`lcm_renoise` + taomate/lift 会硬拒。

再噪声种子：`seed + 0x1C40000 + step`（video）、`seed + 0xA11D + 0x10000 + step`（audio），与初始 latent `seed` / audio `seed+0xA11D` 错开。

## 以后怎么验（GPU，需你点头）

```
unset H3_DIT_NVFP4 H3_DIT_NVFP4_CHAIN H3_SDPA_CUDNN_CUDA_GRAPH
export H3_AUDIO_SIGMA_SHIFT=2 H3_SKIP_PPM=1
./build-cuda-sm120/h3_cuda_generate \
  ... --lora <dmad_full_critic.h3repo> --lora-scale 1 \
  --steps 4 --cfg 1 --width 864 --height 480 --seconds 5 \
  --seed 0x76800001 --sdpa-backend cudnn \
  --fewstep-contract dmad4 --sampler lcm_renoise \
  --token-ids-file evals/768p/ids/p01_sprint.ids \
  --mux out/dmad4_lcm/.../out.mp4
```

期望日志：`fewstep_contract: dmad4 early_ok`、`sampler: lcm_renoise`、`sigma_ok ... audio_shift=2`。视觉门另开，不在本提交范围。

## 非目标 / TODO

- Metal CLI 路径未接（fewstep 合同在 CUDA generate）。
- 与 Comfy 逐 bit 对齐需同 seed 噪声实现；本仓用既有 `fill_normal_f32`。
- PDMD 仍用 `velocity_euler`；勿把 lcm 套到 PDMD 合同上。
