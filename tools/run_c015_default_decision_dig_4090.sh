#!/usr/bin/env bash
# C015 default-on decision dig @ Ada 4090.
# Multi-prompt waffle + warm WC (repeat 2). Euler-8 no-LoRA + DMAD-4 LoRA-on.
# Script measures only — NO silent default flip.
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
      H3_DIT_ACT_FP16_STAGE H3_DIT_ACT_F16
export H3_SKIP_PPM=1 H3_VRAM_LEDGER=1 H3_TE_CACHE_AUTO=1
export LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}

BIN="$ROOT/build-cuda/h3_cuda_generate"
MODELS="${H3_MODELS:-/home/haiyang/workspace/models}"
DIT="$MODELS/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
VAE="$MODELS/minimax_h3_video_vae_fp16.safetensors"
TE="$MODELS/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
DMAD_LORA="$MODELS/loras/dmad_minimax_h3_4step_full_critic.h3repo.safetensors"
OUT_ROOT="${OUT_ROOT:-$ROOT/out/c015_default_decision_dig_480p_20261028}"
SEED=0x76800002
W=864; H=480; SEC=5
IDS_DIR="$ROOT/evals/768p/ids"
EULER_PROMPTS=(p02_talking_head p06_orbit p10_street_wave_dolly)
DMAD_PROMPTS=(p02_talking_head p10_street_wave_dolly)

for f in "$BIN" "$DIT" "$VAE" "$TE" "$DMAD_LORA"; do
  [[ -s "$f" ]] || { echo "missing $f" >&2; exit 2; }
done
for p in "${EULER_PROMPTS[@]}" "${DMAD_PROMPTS[@]}"; do
  [[ -s "$IDS_DIR/${p}.ids" ]] || { echo "missing ids $p" >&2; exit 2; }
done
mkdir -p "$OUT_ROOT"
SHA=$(cat "$ROOT/.synced_from_mac_sha" 2>/dev/null || echo unknown)
{
  echo "date_local: $(date "+%Y-%m-%d %H:%M:%S %Z")"
  echo "host: $(hostname) / RTX 4090 sm_89"
  echo "repo: $ROOT"
  echo "git_sha: $SHA"
  echo "claim: C015 default-on decision dig"
  echo "euler_prompts: ${EULER_PROMPTS[*]}"
  echo "dmad_prompts: ${DMAD_PROMPTS[*]}"
} | tee "$OUT_ROOT/META.txt"

# Usage: run_arm ARM PROMPT STEPS STAGE_ON LORA_ON
# STAGE_ON=0|1  LORA_ON=0|1
run_arm() {
  local ARM=$1 PROMPT=$2 STEPS=$3 STAGE_ON=$4 LORA_ON=$5
  local OUT="$OUT_ROOT/$ARM"
  mkdir -p "$OUT"
  echo "==== $ARM prompt=$PROMPT steps=$STEPS stage=$STAGE_ON lora=$LORA_ON $(date "+%H:%M:%S") ===="
  local ENVARGS=()
  if [[ "$STAGE_ON" == "1" ]]; then ENVARGS+=(H3_DIT_ACT_FP16_STAGE=1); fi
  local LORAARGS=()
  if [[ "$LORA_ON" == "1" ]]; then
    LORAARGS=(--fewstep-preset dmad4 --lora "$DMAD_LORA" --lora-scale 1)
  fi
  set +e
  /usr/bin/time -f "WALL_SEC=%e" env "${ENVARGS[@]}" "$BIN" \
    --dit "$DIT" --vae "$VAE" --te "$TE" \
    --skip-audio \
    --width "$W" --height "$H" --seconds "$SEC" --steps "$STEPS" --cfg 1 \
    --seed "$SEED" --sdpa-backend cudnn \
    --token-ids-file "$IDS_DIR/${PROMPT}.ids" \
    --repeat 2 \
    "${LORAARGS[@]}" \
    --out "$OUT" --mux "$OUT/${ARM}.mp4" \
    >"$OUT/generate.log" 2>&1
  RC=$?
  set -e
  echo "RUN_RC=$RC" | tee -a "$OUT/generate.log"
  grep -E "dit_act_fp16_stage|dit_ws_ledger|vram_ledger: point=peak_dit|phase_ms:|WALL_SEC|pack qkv_f16|wide_f16|h3_lora:|fewstep_|error|Error|FAIL" \
    "$OUT/generate.log" | tee "$OUT/KPI.txt" || true
  return "$RC"
}

FAILS=0
for PROMPT in "${EULER_PROMPTS[@]}"; do
  run_arm "euler_${PROMPT}_off" "$PROMPT" 8 0 0 || FAILS=$((FAILS+1))
  run_arm "euler_${PROMPT}_on"  "$PROMPT" 8 1 0 || FAILS=$((FAILS+1))
done
for PROMPT in "${DMAD_PROMPTS[@]}"; do
  run_arm "dmad_${PROMPT}_off" "$PROMPT" 4 0 1 || FAILS=$((FAILS+1))
  run_arm "dmad_${PROMPT}_on"  "$PROMPT" 4 1 1 || FAILS=$((FAILS+1))
done

python3 "$ROOT/tools/c015_default_decision_aggregate.py" "$OUT_ROOT" "$ROOT" "$SHA"
echo "FAILS_run=$FAILS"
cat "$OUT_ROOT/RECOMMEND.txt"
