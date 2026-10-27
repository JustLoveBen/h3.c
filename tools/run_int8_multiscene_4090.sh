#!/usr/bin/env bash
# INT8 visual gate on Ada 4090 (sm_89). Protocol matches keep_int8_multiscene_20261027.
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source /home/haiyang/miniconda3/etc/profile.d/conda.sh
conda activate comfyui
unset LDFLAGS CFLAGS CXXFLAGS CPPFLAGS
unset H3_DIT_NVFP4 H3_DIT_NVFP4_FILE H3_DIT_NVFP4_CHAIN \
      H3_DIT_NVFP4_CHAIN_FC1_F32 H3_DIT_NVFP4_CHAIN_DIAG \
      H3_SDPA_CUDNN_CUDA_GRAPH H3_TEXT_H_LOAD H3_DIAG_W_CKSUM H3_DIAG_BLK0 \
      H3_KEEP_INT8 H3_GEN_MEM_TRACE
export H3_SKIP_PPM=1
export LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}

BIN="$ROOT/build-cuda/h3_cuda_generate"
MODELS="${H3_MODELS:-/home/haiyang/workspace/models}"
DIT="$MODELS/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
VAE="$MODELS/minimax_h3_video_vae_fp16.safetensors"
TE="$MODELS/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
OUT_ROOT="${OUT_ROOT:-$ROOT/out/keep_int8_4090_20261027}"
SEED=0x76800002
PROMPTS=(p01_sprint p02_talking_head p03_corridor p05_neon_sign p07_market_track)

for f in "$BIN" "$DIT" "$VAE" "$TE"; do
  [[ -s "$f" ]] || { echo "missing $f" >&2; exit 2; }
done
mkdir -p "$OUT_ROOT"
{
  echo "date_local: $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "host: $(hostname) / RTX 4090 sm_89"
  echo "repo: $ROOT"
  echo "git_sha: $(cat $ROOT/.synced_from_mac_sha 2>/dev/null || echo unknown)"
  echo "arch: sm_89"
  echo "arm: INT8"
  echo "seed: $SEED"
  echo "WxH: 864x480 seconds=5 steps=8 cfg=1 sdpa=cudnn skip-audio"
} | tee "$OUT_ROOT/META.txt"

for P in "${PROMPTS[@]}"; do
  IDS="$ROOT/evals/768p/ids/${P}.ids"
  OUT="$OUT_ROOT/$P"
  mkdir -p "$OUT"
  [[ -s "$IDS" ]] || { echo "missing $IDS" >&2; exit 2; }
  echo "==== $P generate $(date '+%H:%M:%S') ===="
  set +e
  /usr/bin/time -f "WALL_SEC=%e" "$BIN" \
    --dit "$DIT" --vae "$VAE" --te "$TE" \
    --width 864 --height 480 --seconds 5 --steps 8 --cfg 1 \
    --seed "$SEED" --sdpa-backend cudnn --skip-audio \
    --token-ids-file "$IDS" \
    --out "$OUT" --mux "$OUT/${P}.mp4" \
    >"$OUT/generate.log" 2>&1
  RC=$?
  set -e
  echo "RUN_RC=$RC" | tee -a "$OUT/generate.log"
  grep -E "dit_format|phase_ms|sdpa_graph|WALL_SEC|error|Error|std|energy" "$OUT/generate.log" | tail -30 || true
  ls -lh "$OUT/${P}.mp4" 2>/dev/null || echo "NO_MP4 $P"
  if [[ "$RC" != 0 ]]; then
    echo "FAIL $P rc=$RC" >&2
    tail -80 "$OUT/generate.log" >&2
    exit "$RC"
  fi
done
echo DONE
ls -lh "$OUT_ROOT"/*/*.mp4
