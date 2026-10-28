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

# Parse process wall + cold/hot phase breakdown from generate.log.
# NEVER call process WALL_SEC "warm" — /usr/bin/time covers cold TE+VAE load too.
# For KPI labels: prefer last phase_ms with hot=1 (true warm clip); else mark COLD.
extract_timing() {
  local log="$1"
  local proc_wall hot_total_s cold_total_s hot_dit_s cold_te_s
  proc_wall=$(grep -E "^WALL_SEC=" "$log" 2>/dev/null | tail -1 | sed -n "s/.*WALL_SEC=\([0-9.]*\).*/\1/p")
  if [[ -z "$proc_wall" ]]; then
    proc_wall=$(grep -E "WALL_SEC=" "$log" 2>/dev/null | tail -1 | sed -n "s/.*WALL_SEC=\([0-9.]*\).*/\1/p")
  fi
  # phase_ms: ... total=MS ... hot=0|1 ...
  cold_total_s=$(grep -E "phase_ms:.*hot=0" "$log" 2>/dev/null | head -1 | sed -n "s/.*total=\([0-9.]*\).*/\1/p")
  hot_total_s=$(grep -E "phase_ms:.*hot=1" "$log" 2>/dev/null | tail -1 | sed -n "s/.*total=\([0-9.]*\).*/\1/p")
  cold_te_s=$(grep -E "phase_ms:.*hot=0" "$log" 2>/dev/null | head -1 | sed -n "s/.*te=\([0-9.]*\).*/\1/p")
  hot_dit_s=$(grep -E "phase_ms:.*hot=1" "$log" 2>/dev/null | tail -1 | sed -n "s/.*dit=\([0-9.]*\).*/\1/p")
  # ms → s
  if [[ -n "$cold_total_s" ]]; then cold_total_s=$(awk -v m="$cold_total_s" 'BEGIN{printf "%.2f", m/1000}'); fi
  if [[ -n "$hot_total_s" ]]; then hot_total_s=$(awk -v m="$hot_total_s" 'BEGIN{printf "%.2f", m/1000}'); fi
  if [[ -n "$cold_te_s" ]]; then cold_te_s=$(awk -v m="$cold_te_s" 'BEGIN{printf "%.1f", m/1000}'); fi
  if [[ -n "$hot_dit_s" ]]; then hot_dit_s=$(awk -v m="$hot_dit_s" 'BEGIN{printf "%.1f", m/1000}'); fi
  echo "proc_wall=${proc_wall:-na};hot_total=${hot_total_s:-na};cold_total=${cold_total_s:-na};cold_te=${cold_te_s:-na};hot_dit=${hot_dit_s:-na}"
}

flag_absurd_timing() {
  # Self-flag: 1s smoke must not be presented as multi-10s product warm KPI.
  local sec="$1" proc_wall="$2" has_hot="$3" name="$4"
  local alerts=0
  if awk -v s="$sec" -v w="$proc_wall" 'BEGIN{exit !(s+0<=2 && w+0>=30)}'; then
    echo "ALERT_KPI_TIMING: $name sec=${sec}s but proc_wall=${proc_wall}s (≥30). Likely COLD TE/VAE load — do NOT label as warm product KPI." | tee -a "$OUT_ROOT/META.txt"
    alerts=1
  fi
  if [[ "$has_hot" != "1" ]]; then
    echo "ALERT_KPI_TIMING: $name has no hot=1 phase (repeat=1 cold-only). Label must say COLD, not warm." | tee -a "$OUT_ROOT/META.txt"
    alerts=1
  fi
  return 0
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

  TIMING=$(extract_timing "$OUT/generate.log")
  # shellcheck disable=SC2086
  eval "${TIMING//;/ }"
  HAS_HOT=0
  [[ "$hot_total" != "na" && -n "$hot_total" ]] && HAS_HOT=1
  flag_absurd_timing "$SEC" "${proc_wall:-0}" "$HAS_HOT" "$NAME" || true

  # KPI seconds for burn-in: hot clip total if present; else cold with explicit COLD tag.
  if [[ "$HAS_HOT" == "1" ]]; then
    # Keep short for 480p burn-in width; always say hot (never mislabel WALL as warm).
    KPI_TAG="hot ${hot_total}s"
    [[ "$hot_dit" != "na" ]] && KPI_TAG="${KPI_TAG} dit${hot_dit}s"
    if awk -v p="${proc_wall:-0}" -v h="${hot_total:-0}" 'BEGIN{exit !(p+0 > h+0*1.25 && p+0>0 && h+0>0)}'; then
      KPI_TAG="${KPI_TAG} wall${proc_wall}s"
    fi
  else
    KPI_TAG="COLD ${proc_wall}s"
    [[ "$cold_te" != "na" ]] && KPI_TAG="${KPI_TAG} te${cold_te}s"
    KPI_TAG="${KPI_TAG} NOT-warm"
  fi

  LABEL=""
  case "$SKU" in
    rc0) LABEL="RC0 864x480 DMAD-4 INT8 | ${SEC}s ${KPI_TAG}" ;;
    rc0_euler) LABEL="RC0 864x480 Euler-8 INT8 | ${SEC}s ${KPI_TAG}" ;;
    rc1) LABEL="RC1 1152x640 DMAD-4 INT8 C015 w24 | ${SEC}s ${KPI_TAG}" ;;
    rc2) LABEL="RC2 1344x768 Euler-8 INT8 C015 w24 | ${SEC}s ${KPI_TAG}" ;;
    *) LABEL="$NAME | ${SEC}s ${KPI_TAG}" ;;
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
  echo "$NAME RC=$RC SEC=$SEC REPEAT=$REPEAT HAS_HOT=$HAS_HOT proc_wall=$proc_wall hot_total=$hot_total cold_total=$cold_total cold_te=$cold_te LABEL=$LABEL SRC=$SRC" | tee -a "$OUT_ROOT/META.txt"
  return "$RC"
}

# A1: short 480p smoke via product entry.
# repeat=2 so burn-in can quote hot=1 (NOT cold TE wall). 1s≠5s product KPI.
echo "==== A1 smoke rc0 1s repeat=2 (label hot, not cold TE) ===="
set +e
run_sku a1_rc0_smoke rc0 1 2 0
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
