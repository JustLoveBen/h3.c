#!/usr/bin/env bash
# R-F: turbo4 + magcache A/B @ 238 (RTX 5090)
# 臂 A: turbo4 纯基线；臂 B/C: magcache 0.06 / 0.10（opt-in 解锁）
set -uo pipefail
ROOT=/mnt/nvme1n1/workspace/h3_c
GEN=$ROOT/build-cuda-sm120/h3_cuda_generate
M=/mnt/nvme1n1/workspace/models
DIT=$M/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
VAE=$M/vae/minimax_h3_video_vae_fp16.safetensors
TE=$M/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
AUDIO=$M/vae/minimax_h3_audio_vae_fp32.safetensors
LORA=$M/loras/minimax_h3_turbo_v4_step600_ema.safetensors
IDS=$ROOT/evals/768p/ids/p01_sprint.ids
SEED=0x48000001
REPEAT=3
OUT=$ROOT/out/review-2026-10-07/rf-magcache-turbo4
mkdir -p "$OUT"

run_arm() {
  local arm="$1"; shift
  local dir="$OUT/$arm"
  mkdir -p "$dir/gen"
  echo "=== arm $arm start $(date -u +%FT%TZ) ==="
  env "$@" "$GEN" \
    --dit "$DIT" --vae "$VAE" --te "$TE" --audio-vae "$AUDIO" \
    --lora "$LORA" \
    --out "$dir/gen" --mux "$dir/${arm}.mp4" \
    --width 864 --height 480 --seconds 5 \
    --preset turbo4 --steps 4 \
    --sdpa-backend cudnn --vae-decode official \
    --te-cache "$OUT/te_cache" \
    --token-ids-file "$IDS" \
    --seed "$SEED" \
    --repeat "$REPEAT" \
    --report "$dir/report.json" \
    ${EXTRA_ARGS:-} \
    > "$dir/generate.log" 2>&1
  local rc=$?
  echo "=== arm $arm exit=$rc $(date -u +%FT%TZ) ==="
  return $rc
}

EXTRA_ARGS=""
run_arm A_base
EXTRA_ARGS="--magcache-thresh 0.06"
export H3_TURBO4_ALLOW_CACHE=1
run_arm B_mag006
EXTRA_ARGS="--magcache-thresh 0.10"
run_arm C_mag010
echo ALL_DONE
