# MiniMax-H3 Super Acceleration

Source: https://raw.githubusercontent.com/NVlabs/Sana/sol-engine/models/minimax_h3/super_acceleration/README.md  
Upstream: https://github.com/NVlabs/Sana/tree/sol-engine/models/minimax_h3/super_acceleration

This directory owns a separate composite GB200 inference profile:

```text
MiniMax-H3 Stage 1 -> direct tensor/audio handoff -> LTX-2.5 Stage 2
```

It is not a MiniMax-H3 replacement model, a Diffusers-only arm, or a
Lightweight/YAML recipe. It combines two independently resident runtimes and
therefore stays outside `sol_engine`, the lightweight package API, and the YAML
configuration surface.

## Fixed v2 profile

- Stage 1 runs on one GB200: first-frame-conditioned MiniMax-H3 FL2VA at
  896x512, 124 frames, 24 FPS, with the LightX2V four-step LoRA merged at
  startup. It performs four Transformer forwards, keeps cache reuse disabled,
  compiles the fixed-shape DiT/VAE path, and decodes with TAEH3.
- The handoff is authenticated loopback TCP, not an intermediate MP4. It sends
  contiguous BF16 video `[1, 3, 121, 384, 672]` in the LTX VAE range and FP32
  stereo 32 kHz PCM `[1, 2, 161333]`. Stage 1 does not return until Stage 2 has
  acknowledged that both tensors are resident on its CUDA device.
- Stage 2 runs on a second GB200. It uses the original LTX-2.5 Video VAE input
  encoder, the learned x2 latent upsampler, original-VAE high-resolution
  first-frame conditioning, and the original Audio VAE. It performs three
  joint updates at sigma points `0.909375, 0.725, 0.421875, 0.0`.
- LTX Transformer layer 0 remains dense. Layers 1-47 use strict Sol-Attn with
  taus `1.0, 1.25, 1.5`; cross-attention and audio attention remain dense. Only
  the 48 Transformer blocks are compiled. TAEHV decodes the final video, and
  the original H3 PCM is muxed into the final H.264/AAC output.

One request therefore occupies two GPUs. The validated four-GPU launch runs two
independent one-Stage-1-GPU to one-Stage-2-GPU pairs concurrently; it is not
four-way tensor, context, or model parallelism.

## Validated latency, with claim limits

Formal Slurm job `6304303` completed 20 hot requests. Model load, compile prime,
and one complete warmup per stage were excluded. The cross-pair medians were:

| Boundary | Median |
| --- | ---: |
| Complete H3 Stage 1 -> final Stage 2 MP4 | 6.760544632 s |
| Stage 1 wall | 4.2919342775 s |
| Stage 2 resident service | 2.446938 s |

These values are absolute hot latency evidence for this v2 profile. There is no
matched same-profile end-to-end baseline, so this directory makes no speedup
claim. A perceptual quality gate was not run.

All model weights, LoRAs, checkpoints, and compile caches are reference-only
and are not distributed by this directory.
