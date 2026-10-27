#!/usr/bin/env bash
# C064: gated TE∥DiT GPU prefetch unlock probe.
# Default: HOST_STAGE_TE on, latent_t>=32 skips GPU prefetch (unchanged).
# Opt-in: H3_DIT_HOST_STAGE_TE=0 + H3_DIT_TE_OVERLAP_BLOCKS=8 overrides skip.
# 480p visual; Euler-8 + DMAD-4; PDMD skip; FaceRefine STOPPED; no NVFP4.
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
      H3_DIT_INT8_WINDOW H3_DIT_TE_OVERLAP_BLOCKS H3_DIT_HOST_STAGE_TE \
      H3_DIT_EPI_FUSE H3_DIT_EPI_FUSE_SL H3_DIT_EPI_FUSE_INK
export H3_SKIP_PPM=1
export H3_VRAM_LEDGER=1
export H3_TE_CACHE_AUTO=1
export H3_LIFECYCLE_TRACE=1
export LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}

BIN="$ROOT/build-cuda/h3_cuda_generate"
MODELS="${H3_MODELS:-/home/haiyang/workspace/models}"
DIT="$MODELS/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
VAE="$MODELS/minimax_h3_video_vae_fp16.safetensors"
TE="$MODELS/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
DMAD_LORA="$MODELS/loras/dmad_minimax_h3_4step_full_critic.h3repo.safetensors"
OUT_ROOT="${OUT_ROOT:-$ROOT/out/c064_te_overlap_480p_20261028}"
SEED=0x76800003
IDS="$ROOT/evals/768p/ids/p10_street_wave_dolly.ids"
W=864; H=480; SEC=5
OV_N="${H3_C064_OV_BLOCKS:-8}"

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
  echo "C003: EPI_FUSE_SL keep OFF (do not reopen)"
  echo "C015/C042: deferred (not small this slice)"
  echo "scheme_policy_t71u: Euler/DMAD primary; PDMD skipped"
  echo "seed: $SEED"
  echo "gate: ${W}x${H} seconds=${SEC} complex=p10_street_wave_dolly (480p visual)"
  echo "probe: C064 HOST_STAGE default vs HOST_STAGE_TE=0 + TE_OVERLAP_BLOCKS=$OV_N"
  echo "env: H3_VRAM_LEDGER=1 H3_TE_CACHE_AUTO=1 H3_LIFECYCLE_TRACE=1 H3_SKIP_PPM=1 --skip-audio --repeat 1"
} | tee "$OUT_ROOT/META.txt"

run_arm() {
  local ARM=$1 STEPS=$2; shift 2
  local EXTRA=("$@")
  local OUT="$OUT_ROOT/$ARM"
  mkdir -p "$OUT"
  echo "==== $ARM generate $(date '+%H:%M:%S') ===="
  unset H3_DIT_TE_OVERLAP_BLOCKS H3_DIT_HOST_STAGE_TE
  if [[ "$ARM" == *"_ov"* ]]; then
    export H3_DIT_HOST_STAGE_TE=0
    export H3_DIT_TE_OVERLAP_BLOCKS="$OV_N"
  fi
  # hs arms: leave HOST_STAGE unset (=on) and TE_OVERLAP unset
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
  unset H3_DIT_TE_OVERLAP_BLOCKS H3_DIT_HOST_STAGE_TE
  echo "RUN_RC=$RC" | tee -a "$OUT/generate.log"
  {
    echo "=== $ARM KPI extract ==="
    grep -E 'dit_prefetch:|dit_host_stage:|lifecycle_|vram_ledger:|phase_ms:|WALL_SEC=|shape:|host_hits|sdpa_actual|OOM|out of memory|cudaError|PROTO' \
      "$OUT/generate.log" || true
  } | tee "$OUT/KPI.txt"
  return 0
}

run_arm euler8_hs 8
run_arm euler8_ov8 8
run_arm dmad4_hs 4 --fewstep-preset dmad4 --lora "$DMAD_LORA"
run_arm dmad4_ov8 4 --fewstep-preset dmad4 --lora "$DMAD_LORA"

mkdir -p "$OUT_ROOT/labeled"
for arm in euler8_hs euler8_ov8 dmad4_hs dmad4_ov8; do
  src="$OUT_ROOT/$arm/${arm}.mp4"
  [[ -s "$src" ]] || continue
  ffmpeg -y -i "$src" -vf "drawtext=text='${arm}':fontsize=28:fontcolor=white:box=1:boxcolor=black@0.5:x=20:y=20" \
    -c:v libx264 -crf 18 -preset fast -an "$OUT_ROOT/labeled/${arm}_labeled.mp4" 2>/dev/null || \
    cp -f "$src" "$OUT_ROOT/labeled/${arm}_labeled.mp4"
done

echo "DONE $(date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a "$OUT_ROOT/META.txt"
