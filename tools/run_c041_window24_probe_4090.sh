#!/usr/bin/env bash
# C041: INT8 weight-window cap probe (30/41 → 24). One-shot; does NOT change default.
# 480p visual gate; Euler-8 + DMAD-4; PDMD skip; FaceRefine STOPPED; no NVFP4.
# Measurable: peak_dit / free@peak / WALL / dit_ms / host_hits / freeze window.
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source /home/haiyang/miniconda3/etc/profile.d/conda.sh
conda activate comfyui
unset LDFLAGS CFLAGS CXXFLAGS CPPFLAGS
unset H3_DIT_NVFP4 H3_DIT_NVFP4_FILE H3_DIT_NVFP4_CHAIN \
      H3_DIT_NVFP4_CHAIN_FC1_F32 H3_DIT_NVFP4_CHAIN_DIAG \
      H3_SDPA_CUDNN_CUDA_GRAPH H3_TEXT_H_LOAD H3_DIAG_W_CKSUM H3_DIAG_BLK0 \
      H3_KEEP_INT8 H3_GEN_MEM_TRACE H3_DIT_NVFP4_KEEP_INT8 \
      H3_DIT_NVFP4_HOST_DEQUANT H3_FEWSTEP_PRESET H3_FEWSTEP_CONTRACT \
      H3_DIT_INT8_WINDOW
export H3_SKIP_PPM=1
export H3_VRAM_LEDGER=1
export H3_TE_CACHE_AUTO=1
export LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}

BIN="$ROOT/build-cuda/h3_cuda_generate"
MODELS="${H3_MODELS:-/home/haiyang/workspace/models}"
DIT="$MODELS/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
VAE="$MODELS/minimax_h3_video_vae_fp16.safetensors"
TE="$MODELS/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
DMAD_LORA="$MODELS/loras/dmad_minimax_h3_4step_full_critic.h3repo.safetensors"
OUT_ROOT="${OUT_ROOT:-$ROOT/out/c041_window24_480p_20261028}"
SEED=0x76800002
IDS="$ROOT/evals/768p/ids/p10_street_wave_dolly.ids"
W=864; H=480; SEC=5

for f in "$BIN" "$DIT" "$VAE" "$TE" "$IDS" "$DMAD_LORA"; do
  [[ -s "$f" ]] || { echo "missing $f" >&2; exit 2; }
done
mkdir -p "$OUT_ROOT"
STAMP=$(date '+%Y-%m-%d %H:%M:%S %Z')
FREE0=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
{
  echo "date_local: $STAMP"
  echo "host: $(hostname) / RTX 4090 sm_89"
  echo "repo: $ROOT"
  echo "git_sha: $(cat $ROOT/.synced_from_mac_sha 2>/dev/null || echo unknown)"
  echo "arch: sm_89 total_vram_mib=24564 free_before_mib=$FREE0"
  echo "product_default: INT8"
  echo "KEEP=attn: off"
  echo "NVFP4: BLOCK / unset"
  echo "FaceRefine: STOPPED"
  echo "scheme_policy_t71u: Euler/DMAD primary; PDMD skipped"
  echo "seed: $SEED"
  echo "gate: ${W}x${H} seconds=${SEC} complex=p10_street_wave_dolly (480p visual)"
  echo "probe: C041 H3_DIT_INT8_WINDOW unset vs 24 (weight cache layers; NOT attn_window)"
  echo "env: H3_VRAM_LEDGER=1 H3_TE_CACHE_AUTO=1 H3_SKIP_PPM=1 --skip-audio --repeat 1"
  echo "note: C015/C042 act-FP16 staging deferred (not built; non bit-identical)"
} | tee "$OUT_ROOT/META.txt"

run_arm() {
  local ARM=$1 STEPS=$2; shift 2
  local EXTRA=("$@")
  local OUT="$OUT_ROOT/$ARM"
  mkdir -p "$OUT"
  echo "==== $ARM generate $(date '+%H:%M:%S') ===="
  # isolate env per arm
  unset H3_DIT_INT8_WINDOW
  if [[ "$ARM" == *"_w24"* ]]; then
    export H3_DIT_INT8_WINDOW=24
  fi
  set +e
  /usr/bin/time -f "WALL_SEC=%e" "$BIN" \
    --dit "$DIT" --vae "$VAE" --te "$TE" \
    --skip-audio \
    --width "$W" --height "$H" --seconds "$SEC" --steps "$STEPS" --cfg 1 \
    --seed "$SEED" --sdpa-backend cudnn \
    --token-ids-file "$IDS" \
    --repeat 1 \
    --out "$OUT" --mux "$OUT/${ARM}.mp4" \
    "${EXTRA[@]}" \
    >"$OUT/generate.log" 2>&1
  RC=$?
  set -e
  unset H3_DIT_INT8_WINDOW
  echo "RUN_RC=$RC" | tee -a "$OUT/generate.log"
  {
    echo "=== $ARM KPI extract ==="
    grep -E 'dit_int8_window_cap|dit_int8_cache:|dit_int8_cache_refit:|dit_int8_cache_freeze:|vram_ledger:|phase_ms:|WALL_SEC=|shape:|host_hits|sdpa_actual|OOM|out of memory|cudaError' \
      "$OUT/generate.log" || true
  } | tee "$OUT/KPI.txt"
  return 0
}

# baseline first (product window), then capped
run_arm euler8_base 8
run_arm euler8_w24 8
run_arm dmad4_base 4 --fewstep-preset dmad4 --lora "$DMAD_LORA"
run_arm dmad4_w24 4 --fewstep-preset dmad4 --lora "$DMAD_LORA"

# label clips
mkdir -p "$OUT_ROOT/labeled"
for arm in euler8_base euler8_w24 dmad4_base dmad4_w24; do
  src="$OUT_ROOT/$arm/${arm}.mp4"
  [[ -s "$src" ]] || continue
  ffmpeg -y -i "$src" -vf "drawtext=text='${arm}':fontsize=28:fontcolor=white:box=1:boxcolor=black@0.5:x=20:y=20" \
    -c:v libx264 -crf 18 -preset fast -an "$OUT_ROOT/labeled/${arm}_labeled.mp4" 2>/dev/null || \
    cp -f "$src" "$OUT_ROOT/labeled/${arm}_labeled.mp4"
done

echo "DONE $(date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a "$OUT_ROOT/META.txt"
