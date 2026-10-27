#!/usr/bin/env bash
# C040 VRAM ledger buckets + C062 TE-cache AUTO HIT/MISS KPI on Ada 4090.
# Eval gate: 480p × 5s, complex prompt p10, INT8 product default.
# Arms:
#   A) Euler-8 INT8  --repeat 2  H3_TE_CACHE_AUTO=1  H3_VRAM_LEDGER=1
#   B) PDMD-4 INT8   --repeat 2  H3_TE_CACHE_AUTO=1  (beyond-DMAD preferred fewstep)
# No FaceRefine. No NVFP4. No KEEP=attn.
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
      H3_DIT_NVFP4_HOST_DEQUANT
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
# Visual/VRAM/TE KPI: skip-audio (same as INT8 multiscene gate). Override with H3_AUDIO_VAE=path.
AUDIO_VAE="${H3_AUDIO_VAE:-}"
OUT_ROOT="${OUT_ROOT:-$ROOT/out/c040_c062_kpi_480p_20261028}"
SEED=0x76800002
IDS="$ROOT/evals/768p/ids/p10_street_wave_dolly.ids"
W=864; H=480; SEC=5

for f in "$BIN" "$DIT" "$VAE" "$TE" "$IDS"; do
  [[ -s "$f" ]] || { echo "missing $f" >&2; exit 2; }
done
AUDIO_ARGS=(--skip-audio)
if [[ -n "$AUDIO_VAE" && -s "$AUDIO_VAE" ]]; then
  AUDIO_ARGS=(--audio-vae "$AUDIO_VAE")
fi
mkdir -p "$OUT_ROOT"
STAMP=$(date '+%Y-%m-%d %H:%M:%S %Z')
{
  echo "date_local: $STAMP"
  echo "host: $(hostname) / RTX 4090 sm_89"
  echo "repo: $ROOT"
  echo "git_sha: $(cat $ROOT/.synced_from_mac_sha 2>/dev/null || echo unknown)"
  echo "arch: sm_89"
  echo "product_default: INT8"
  echo "KEEP=attn: off (experimental only)"
  echo "NVFP4: BLOCK / unset"
  echo "FaceRefine: STOPPED (pt70u)"
  echo "seed: $SEED"
  echo "gate: ${W}x${H} seconds=${SEC} complex=p10_street_wave_dolly"
  echo "env: H3_VRAM_LEDGER=1 H3_TE_CACHE_AUTO=1 H3_SKIP_PPM=1"
} | tee "$OUT_ROOT/META.txt"

run_arm() {
  local ARM=$1 STEPS=$2 EXTRA=("${@:3}")
  local OUT="$OUT_ROOT/$ARM"
  mkdir -p "$OUT"
  echo "==== $ARM generate $(date '+%H:%M:%S') ===="
  set +e
  /usr/bin/time -f "WALL_SEC=%e" "$BIN" \
    --dit "$DIT" --vae "$VAE" --te "$TE" \
    "${AUDIO_ARGS[@]}" \
    --width "$W" --height "$H" --seconds "$SEC" --steps "$STEPS" --cfg 1 \
    --seed "$SEED" --sdpa-backend cudnn \
    --token-ids-file "$IDS" \
    --repeat 2 \
    --out "$OUT" --mux "$OUT/${ARM}.mp4" \
    "${EXTRA[@]}" \
    >"$OUT/generate.log" 2>&1
  RC=$?
  set -e
  echo "RUN_RC=$RC" | tee -a "$OUT/generate.log"
  {
    echo "=== $ARM KPI extract ==="
    grep -E 'dit_format|te_prompt_cache|te_encode:|vram_ledger|vram_ledger_delta|phase_ms|WALL_SEC|fewstep|H3_FEWSTEP|error|Error|FAIL' \
      "$OUT/generate.log" || true
  } | tee "$OUT/KPI.txt"
  ls -lh "$OUT"/*.mp4 2>/dev/null || echo "NO_MP4 $ARM"
  if [[ "$RC" != 0 ]]; then
    echo "FAIL $ARM rc=$RC" >&2
    tail -100 "$OUT/generate.log" >&2
    return "$RC"
  fi
  return 0
}

# Arm A: quality-preserving Euler-8 INT8 (lossless acceleration measurement)
run_arm euler8_int8_teauto 8

# Arm B: beyond-DMAD preferred fewstep PDMD-4 (not DMAD)
# Prefer preset if binary supports --fewstep-preset; else env.
set +e
"$BIN" --help 2>&1 | grep -q 'fewstep-preset' && HAS_PRESET=1 || HAS_PRESET=0
set -e
if [[ "$HAS_PRESET" == 1 ]]; then
  run_arm pdmd4_int8_teauto 4 --fewstep-preset pdmd4
else
  export H3_FEWSTEP_PRESET=pdmd4
  run_arm pdmd4_int8_teauto 4
  unset H3_FEWSTEP_PRESET
fi

# Burn labels (scheme + warm WALL) on clip1 mux if present
LABEL_DIR="$OUT_ROOT/labeled"
mkdir -p "$LABEL_DIR"
for ARM in euler8_int8_teauto pdmd4_int8_teauto; do
  IN="$OUT_ROOT/$ARM/${ARM}.mp4"
  # --repeat 2 may write clip0 / clip1 names; prefer primary mux
  if [[ ! -s "$IN" ]]; then
    IN=$(ls "$OUT_ROOT/$ARM"/*.mp4 2>/dev/null | head -1 || true)
  fi
  [[ -s "$IN" ]] || continue
  WALL=$(grep -E '^WALL_SEC=' "$OUT_ROOT/$ARM/generate.log" | tail -1 | cut -d= -f2)
  # warm ≈ second-clip share of WALL; report total WALL /2 as rough warm proxy + note
  TE_HIT=$(grep -c 'te_prompt_cache: HIT' "$OUT_ROOT/$ARM/generate.log" || true)
  SCHEME="864x480 ${ARM} INT8 | WALL ${WALL}s teHIT=${TE_HIT}"
  bash "$ROOT/tools/burn_eval_label.sh" "$IN" "$LABEL_DIR/${ARM}_labeled.mp4" "$SCHEME" \
    || echo "label failed $ARM"
done

# Aggregate summary
{
  echo "# C040+C062 KPI summary"
  echo "date: $STAMP"
  echo
  for ARM in euler8_int8_teauto pdmd4_int8_teauto; do
    echo "## $ARM"
    [[ -f "$OUT_ROOT/$ARM/KPI.txt" ]] && cat "$OUT_ROOT/$ARM/KPI.txt"
    echo
  done
} | tee "$OUT_ROOT/SUMMARY.md"

echo DONE
ls -lh "$OUT_ROOT"/*/*.mp4 "$LABEL_DIR"/*.mp4 2>/dev/null || true
