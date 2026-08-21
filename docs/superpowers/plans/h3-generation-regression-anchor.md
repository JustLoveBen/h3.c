# CUDA T2V 回归锚点（阶段 A）

构建见 [`docs/nvidia_build.md`](../../nvidia_build.md)。本文件冻结 **INT8 ConvRot DiT** 主题命令；**不是** W4A8。CUDA 路径是旁路，不替换 Metal `h3_generate`。

A 的 e2e 金标准是 `out/verify_r1r3_p3_vae/`（橙猫 / 水下），**不是** W4A8。无 condition 时 CUDA DiT 必须与今日 `forward_av` bit-for-bit；后续 pack / AdaLN 改动必须先过本门。

本阶段 **不** 把 `--first-frame` 做成产品默认；`--init-latent` lite **不得**冒充官方 F2VA / L2VA / FL2VA / Ref2VA。

## 4090 权路径（INT8 主题，非 W4A8）

主机源码常在 `/home/haiyang/h3_c/src_sync`。权重：

```sh
DIT_INT8=/home/haiyang/h3_c/models/comfy-org/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
VAE=/home/haiyang/h3_c/models/kijai/minimax_h3_video_vae_int8_convrot.safetensors
VAE_FP16=/home/haiyang/h3_c/models/comfy-org/vae/minimax_h3_video_vae_fp16.safetensors
TE=/home/haiyang/h3_c/models/comfy-org/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
```

P3 decode 用 INT8 VAE（速度档，与 `out/verify_r1r3_p3_vae/README.txt` 一致）。
**主题质量刀**用 fp16 VAE（`models/comfy-org/vae/minimax_h3_video_vae_fp16.safetensors`）
+ 默认 256px tiles + deblock16；不要 `H3_VAE_NO_SPATIAL_TILE=1`。
INT8 VAE 仍是速度档。产物对照：`out/verify_theme_ok/`、`out/verify_blockfix_cat/`、
`out/verify_stage_s_hq2/`（质量 PASS：`cat_cfg1/`、`uw_cfg1/`）。

## 主题档（864×480，5s，`--frames 17` → latent `7×30×54`）

DiT / TE / VAE：`--layers 50 --te-layers 50 --vae-blocks 36`  
采样：`--steps 8 --cfg 5`；reuse / magcache / Sage / token-reduction **关**  
SDPA：`H3_CUDA_SDPA_BACKEND=cudnn` / `--sdpa-backend cudnn`

### 橙猫（对照 `out/verify_r1r3_p3_vae/cat/`）

`--seed 17`，`out/verify2/tokens/v4_orange_cat.txt`（20 tok）  
latent 指纹：`mean=-0.1121 std=1.285`

```sh
H3_CUDA_SDPA_BACKEND=cudnn H3_AUDIO_CONV_BACKEND=gemm \
./build-cuda/h3_cuda_generate \
  --dit "$DIT_INT8" --vae "$VAE" --te "$TE" \
  --out /tmp/h3_reg_cat --width 864 --height 480 --seconds 5 \
  --frames 17 --steps 8 --layers 50 --te-layers 50 --vae-blocks 36 \
  --cfg 5 --seed 17 --sdpa-backend cudnn \
  --token-ids-file out/verify2/tokens/v4_orange_cat.txt
```

### 水下（对照 `out/verify_r1r3_p3_vae/uw/`）

同上，改 `--seed 91` 与 `--token-ids-file out/verify_official_prompt/b_underwater.ids`（55 tok）  
latent 指纹：`mean=-0.1058 std=1.679`

```sh
H3_CUDA_SDPA_BACKEND=cudnn H3_AUDIO_CONV_BACKEND=gemm \
./build-cuda/h3_cuda_generate \
  --dit "$DIT_INT8" --vae "$VAE" --te "$TE" \
  --out /tmp/h3_reg_uw --width 864 --height 480 --seconds 5 \
  --frames 17 --steps 8 --layers 50 --te-layers 50 --vae-blocks 36 \
  --cfg 5 --seed 91 --sdpa-backend cudnn \
  --token-ids-file out/verify_official_prompt/b_underwater.ids
```

### 主题质量刀（对齐 `theme_ok`；非速度档）

H3 是 **CFG-distilled**。金标 `a_orange_cat_fixed` / `theme_ok` 是 **`--cfg 1` + 官方 57 tok**。
`--cfg 5` + v4 20 tok 会过曝并把毛发推成描边「指纹」；同热 latent 换 fp16 decode 救不了。
质量刀：`--vae "$VAE_FP16"`、`--cfg 1`、官方 ids。不要 `H3_VAE_NO_SPATIAL_TILE` /
`H3_NVFP4_DEV_ACT` / magcache / reuse / Sage / W4A8。产物：`out/verify_stage_s_hq2/`。

T2VA **CLI 默认仍 cfg=5**（速度/主题可区分档，不当画质 PASS）。

```sh
H3_CUDA_SDPA_BACKEND=cudnn H3_AUDIO_CONV_BACKEND=gemm \
./build-cuda/h3_cuda_generate \
  --dit "$DIT_INT8" --vae "$VAE_FP16" --te "$TE" \
  --out /tmp/h3_hq_cat --width 864 --height 480 --seconds 5 \
  --frames 17 --steps 8 --layers 50 --te-layers 50 --vae-blocks 36 \
  --cfg 1 --seed 17 --sdpa-backend cudnn \
  --token-ids-file out/verify_int8_ab/a_orange_cat.ids
```

## 禁止（主题锚点）

- `--init-latent` / `--lock-first-frame` / `--init-first-frame`（非官方 condition）
- `--reuse`、`--magcache-*`、W4A8 DiT、`--cfg 1`
- Sage / token-reduction 默认开

日志必须出现：

- `dit_format: INT8 ConvRot`
- `cfg: scale=5`
- `reuse_schedule: disabled`
- `token_red=0`

## `forward_av` 自洽夹具（阶段 A 门）

不跑 8-step e2e 也能证明夹具入口可用。A 未改 `h3_cuda_generate` 时，e2e 对照 P3 日志即可，不必重跑 50L 出片。

```sh
cmake --build build-cuda -j --target h3_dit_w4a8_cond_test h3_cuda_generate
./build-cuda/h3_dit_w4a8_cond_test "$DIT_INT8"
```

Expected：`ok: forward_av self-consistent (stage A)`，exit 0。

## P3 金标准摘录（本轮未改 generate，不重跑）

`out/verify_r1r3_p3_vae/cat/generate.log`：

- `dit_format: INT8 ConvRot (theme-follow path)`
- `cfg: scale=5`
- `reuse_schedule: disabled`
- `token_red=0`
- `phase_ms: ... total=63753.0`（橙猫 ≈63.75s）

`out/verify_r1r3_p3_vae/uw/generate.log`：`phase_ms: ... total=68027.9`（≈68.03s）。

成片对照：`out/verify_theme_ok/` 与 `out/verify_r1r3_p3_vae/{cat,uw}/`。
