# Stage-2 contract: H3-to-LTX-2.5 v2

Source: https://raw.githubusercontent.com/NVlabs/Sana/sol-engine/models/minimax_h3/super_acceleration/STAGE2_CONTRACT.md

## Input and transport

Each Stage-2 service process sees exactly one GB200 and accepts requests from
its paired Stage-1 process over authenticated loopback TCP.

| Tensor | Dtype | Shape | Meaning |
| --- | --- | --- | --- |
| Video | BF16 | `[1, 3, 121, 384, 672]` | NCTHW pixels, contiguous, finite, in `[-1, 1]` |
| Audio | FP32 | `[1, 2, 161333]` | Contiguous stereo PCM at 32000 Hz |

The default production handoff is `direct_tensor`. The MP4 path is a diagnostic
fallback and is not the profile represented by job 6304303.

## Frozen Stage-2 algorithm

1. Encode the direct 672x384/121-frame video with the original LTX-2.5 Video VAE input encoder.
2. Apply the official learned x2 latent spatial upsampler.
3. Encode the pinned 1344x768 opening frame with the original LTX-2.5 Video VAE as hard first-frame conditioning.
4. Encode the direct H3 PCM with the original LTX-2.5 Audio VAE and jointly denoise video and audio latents.
5. Run exactly three updates with sigma endpoints `[0.909375, 0.725, 0.421875, 0.0]`.
6. Decode the final video with the wide LTX TAEHV decoder.
7. Discard the jointly denoised Stage-2 audio output and mux the original H3 PCM into the final MP4.

Output: H.264 at 1344x768, 121 frames, 24 FPS, plus stereo AAC at 32 kHz.

## Attention and compile invariants

- Transformer has 48 layers; layer 0 video self-attention is dense.
- Layers 1-47 use strict Sol-Attn with taus `1.0, 1.25, 1.5`.
- A complete request records exactly 3 dense layer-0 calls, 141 Sol calls, and 141 actual `cute_sm100` kernel calls.
- Only the 48 Transformer blocks are compiled; the stateful Sol inner callable remains eager.
- Model offload, quantization, and cache-based denoising reuse are disabled.
