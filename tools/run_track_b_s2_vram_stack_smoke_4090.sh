#!/usr/bin/env bash
# Track-B S2 VRAM stack smoke: DMAD-4 + WINDOW=24 + C015 ON + cudnn (no sol).
# Confirms window+C015 stack still OK. Does NOT arm sol / does NOT flip defaults in code.
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source /home/haiyang/miniconda3/etc/profile.d/conda.sh
conda activate comfyui
unset LDFLAGS CFLAGS CXXFLAGS CPPFLAGS
unset H3_DIT_NVFP4 H3_DIT_NVFP4_FILE H3_DIT_NVFP4_CHAIN \
      H3_DIT_NVFP4_CHAIN_FC1_F32 H3_DIT_NVFP4_CHAIN_DIAG \
      H3_SDPA_CUDNN_CUDA_GRAPH H3_TEXT_H_LOAD H3_DIAG_W_CKSUM H3_DIAG_BLK0 \
      H3_KEEP_INT8 H3_GEN_MEM_TRACE H3_DIT_NVFP4_KEEP_INT8 \
      H3_DIT_NVFP4_HOST_DEQUANT H3_FEWSTEP_PRESET H3_FEWSTEP_CONTRACT \
      H3_DIT_EPI_FUSE H3_DIT_EPI_FUSE_INK H3_DIT_EPI_FUSE_SL \
      H3_SOL_FIRST_DENSE_STEPS H3_SOL_FIRST_DENSE_LAYERS H3_SDPA_STRICT \
      H3_CUDA_SDPA_BACKEND H3_SOL_CUTE_LIB H3_SOL_ATTN_ROOT H3_TRITON_OVERLAY \
      H3_SOL_CUTE_TOOLS H3_DIT_ACT_F16
export H3_SKIP_PPM=1 H3_VRAM_LEDGER=1 H3_TE_CACHE_AUTO=1
export H3_DIT_INT8_WINDOW=24
# Simulate product C015 default ON on pre-b99d56b binaries; on >=b99d56b unset also OK
export H3_DIT_ACT_FP16_STAGE="${H3_DIT_ACT_FP16_STAGE:-1}"
export LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}

BIN="$ROOT/build-cuda/h3_cuda_generate"
MODELS="${H3_MODELS:-/home/haiyang/workspace/models}"
DIT="$MODELS/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
VAE="$MODELS/minimax_h3_video_vae_fp16.safetensors"
TE="$MODELS/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
DMAD_LORA="$MODELS/loras/dmad_minimax_h3_4step_full_critic.h3repo.safetensors"
OUT="${OUT_ROOT:-$ROOT/out/track_b_s2_vram_stack_smoke_480p_20261028}"
IDS="$ROOT/evals/768p/ids/p10_street_wave_dolly.ids"
mkdir -p "$OUT"
SHA=$(cat "$ROOT/.synced_from_mac_sha" 2>/dev/null || echo unknown)
{
  echo "date_local: $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "claim: Track-B S2 VRAM stack smoke (window24+C015 ON+cudnn DMAD; no sol)"
  echo "git_sha: $SHA"
  echo "H3_DIT_INT8_WINDOW=$H3_DIT_INT8_WINDOW"
  echo "H3_DIT_ACT_FP16_STAGE=$H3_DIT_ACT_FP16_STAGE"
} | tee "$OUT/META.txt"

/usr/bin/time -f "WALL_SEC=%e" "$BIN" \
  --dit "$DIT" --vae "$VAE" --te "$TE" \
  --skip-audio \
  --width 864 --height 480 --seconds 5 --steps 4 --cfg 1 \
  --seed 0x76800002 --sdpa-backend cudnn \
  --token-ids-file "$IDS" \
  --fewstep-preset dmad4 --lora "$DMAD_LORA" --lora-scale 1 \
  --repeat 2 \
  --out "$OUT" --mux "$OUT/smoke.mp4" \
  >"$OUT/generate.log" 2>&1
RC=$?
echo "RUN_RC=$RC" | tee -a "$OUT/generate.log"
grep -E "H3_DIT_INT8_WINDOW|H3_DIT_ACT_FP16_STAGE|dit_act_fp16|vram_ledger: point=peak_dit|phase_ms:|WALL_SEC|fewstep_|error|Error|FAIL|sdpa" \
  "$OUT/generate.log" | tee "$OUT/KPI.txt" || true
exit "$RC"
