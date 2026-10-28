#!/usr/bin/env bash
# A1 GPU smoke + A2 RC1/RC2 complex gates + A3 C015+WINDOW24 ledger (4090).
# Uses tools/h3_product_generate.sh as the product entry.
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
      H3_DIT_INT8_WINDOW H3_CUDA_SDPA_BACKEND
# C015 product default ON — leave unset
export H3_SKIP_PPM=1
export H3_TE_CACHE_AUTO=1
export LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}

PG="$ROOT/tools/h3_product_generate.sh"
IDS="$ROOT/evals/768p/ids/p10_street_wave_dolly.ids"
OUT_ROOT="${OUT_ROOT:-$ROOT/out/a1a2a3_product_gates_20261028}"
SHARE="${SHARE:-$HOME/grokbot-share/a1a2a3_product_gates_20261028}"
SEED=0x76800002
mkdir -p "$OUT_ROOT/labeled" "$SHARE/labeled"
STAMP=$(date "+%Y-%m-%d %H:%M:%S %Z")
{
  echo "date_local: $STAMP"
  echo "host: $(hostname) / RTX 4090 sm_89"
  echo "synced_sha: $(cat $ROOT/.synced_from_mac_sha 2>/dev/null || echo unknown)"
  echo "entry: tools/h3_product_generate.sh"
  echo "C015: default ON (unset)"
  echo "FaceRefine: STOPPED; NVFP4: BLOCK; PDMD primary: NO"
  echo "gate_prompt: p10_street_wave_dolly complex 480p-style"
} | tee "$OUT_ROOT/META.txt"

extract_warm() {
  local log="$1"
  # Prefer last WALL_SEC if present; else last phase e2e-ish line
  local w
  w=$(grep -E "WALL_SEC=" "$log" 2>/dev/null | tail -1 | sed -n "s/.*WALL_SEC=\\([0-9.]*\\).*/\\1/p")
  if [[ -z "$w" ]]; then
    w=$(grep -E "e2e_ms=|wall_ms=" "$log" 2>/dev/null | tail -1 | head -c 80)
  fi
  echo "${w:-na}"
}

run_sku() {
  local NAME="$1" SKU="$2" SEC="$3" REPEAT="$4" WINDOW="$5"
  local OUT="$OUT_ROOT/$NAME"
  mkdir -p "$OUT"
  echo "==== $NAME sku=$SKU sec=$SEC repeat=$REPEAT window24=$WINDOW $(date "+%H:%M:%S") ====" | tee -a "$OUT_ROOT/META.txt"
  local EXTRA=(--sku "$SKU" --seconds "$SEC" --repeat "$REPEAT" --seed "$SEED" \
    --prompt-ids "$IDS" --out "$OUT" --mux "$OUT/${NAME}.mp4")
  if [[ "$WINDOW" == "1" ]]; then
    EXTRA+=(--window24)
  fi
  set +e
  if [[ "$WINDOW" == "1" ]]; then
    H3_VRAM_LEDGER=1 /usr/bin/time -f "WALL_SEC=%e" "$PG" "${EXTRA[@]}" >"$OUT/generate.log" 2>&1
  else
    /usr/bin/time -f "WALL_SEC=%e" "$PG" "${EXTRA[@]}" >"$OUT/generate.log" 2>&1
  fi
  RC=$?
  set -e
  echo "RUN_RC=$RC" | tee -a "$OUT/generate.log"
  {
    echo "=== $NAME extract ==="
    grep -E "dit_format|te_prompt_cache|fewstep_|sampler:|phase_ms|WALL_SEC|lora=|h3_lora|error|Error|FAIL|sdpa_actual|vram_|peak|H3_PRODUCT|window" \
      "$OUT/generate.log" || true
  } | tee "$OUT/extract.txt"

  WARM=$(extract_warm "$OUT/generate.log")
  LABEL=""
  case "$SKU" in
    rc0) LABEL="RC0 864x480 DMAD-4 INT8 C015 | warm ${WARM}s" ;;
    rc0_euler) LABEL="RC0 864x480 Euler-8 INT8 C015 | warm ${WARM}s" ;;
    rc1) LABEL="RC1 1152x640 DMAD-4 INT8 C015 w24 | warm ${WARM}s" ;;
    rc2) LABEL="RC2 1344x768 Euler-8 INT8 C015 w24 | warm ${WARM}s" ;;
    *) LABEL="$NAME | warm ${WARM}s" ;;
  esac
  # Prefer warm clip (_r2) if repeat>1
  SRC="$OUT/${NAME}.mp4"
  if [[ "$REPEAT" -gt 1 ]]; then
    # mux gets _rK suffix when N>1 per CLI help
    CAND=$(ls -1 "$OUT"/*_r${REPEAT}.mp4 2>/dev/null | head -1 || true)
    if [[ -z "$CAND" ]]; then
      CAND=$(ls -1t "$OUT"/*.mp4 2>/dev/null | head -1 || true)
    fi
    [[ -n "$CAND" ]] && SRC="$CAND"
  fi
  if [[ -s "$SRC" ]]; then
    "$ROOT/tools/burn_eval_label.sh" "$SRC" "$OUT/labeled/${NAME}_labeled.mp4" "$LABEL" || true
    cp -f "$OUT/labeled/${NAME}_labeled.mp4" "$SHARE/labeled/" 2>/dev/null || true
    cp -f "$SRC" "$SHARE/${NAME}.mp4" 2>/dev/null || true
  fi
  echo "$NAME RC=$RC WARM=$WARM SRC=$SRC" | tee -a "$OUT_ROOT/META.txt"
  return "$RC"
}

# A1: short 480p default path via product entry (do not break rc0)
echo "==== A1 smoke rc0 1s ====" 
set +e
run_sku a1_rc0_smoke rc0 1 1 0
A1_RC=$?
set -e

# A2+A3: RC1 1152 DMAD complex 5s warm + WINDOW24 ledger
set +e
run_sku a2_rc1_1152_dmad rc1 5 2 1
A2_RC1=$?
set -e

# A2+A3: RC2 1344 Euler complex 5s warm + WINDOW24 ledger
set +e
run_sku a2_rc2_1344_euler rc2 5 2 1
A2_RC2=$?
set -e

{
  echo "SUMMARY A1_RC=$A1_RC A2_RC1=$A2_RC1 A2_RC2=$A2_RC2"
  echo "labeled: $SHARE/labeled"
  echo "out: $OUT_ROOT"
} | tee -a "$OUT_ROOT/META.txt"

# Copy ledger-ish extracts
mkdir -p "$SHARE/logs"
cp -f "$OUT_ROOT"/*/extract.txt "$SHARE/logs/" 2>/dev/null || true
cp -f "$OUT_ROOT/META.txt" "$SHARE/"

exit 0
