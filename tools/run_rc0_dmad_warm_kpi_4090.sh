#!/usr/bin/env bash
# tools/run_rc0_dmad_warm_kpi_4090.sh — honest RC0 DMAD warm 1s + 5s (t89u).
# Never burn /usr/bin/time WALL as warm. Prefer last phase_ms hot=1.
# Baseline: DMAD-4 480p×5s warm e2e ~25.0s / dit ~17.0s @3cee9ff.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PG="$ROOT/tools/h3_product_generate.sh"
IDS="${IDS:-$ROOT/evals/768p/ids/p10_street_wave_dolly.ids}"
SEED="${SEED:-0x76800002}"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_ROOT="${OUT_ROOT:-$ROOT/out/rc0_dmad_warm_kpi_${STAMP}}"
SHARE="${SHARE:-$HOME/grokbot-share/rc0_dmad_warm_kpi_${STAMP}}"
mkdir -p "$OUT_ROOT/labeled" "$SHARE/labeled" "$SHARE/logs"

{
  echo "date_local: $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "host: $(hostname) / RTX 4090"
  echo "synced_sha: $(cat "$ROOT/.synced_from_mac_sha" 2>/dev/null || echo na)"
  echo "entry: $PG"
  echo "policy: warm = last phase_ms hot=1; WALL = cold+hot process — never label WALL as warm"
  echo "baseline_5s: warm_e2e~25.02s dit~16982ms @dmad4_vs_euler8_kpi"
} | tee "$OUT_ROOT/META.txt"

extract_hot() {
  local log="$1"
  local hot_total hot_dit cold_total cold_te proc_wall
  proc_wall=$(grep -E "^WALL_SEC=" "$log" 2>/dev/null | tail -1 | sed -n 's/.*WALL_SEC=\([0-9.]*\).*/\1/p')
  [[ -z "$proc_wall" ]] && proc_wall=$(grep -E "WALL_SEC=" "$log" 2>/dev/null | tail -1 | sed -n 's/.*WALL_SEC=\([0-9.]*\).*/\1/p')
  cold_total=$(grep -E "phase_ms:.*hot=0" "$log" 2>/dev/null | head -1 | sed -n 's/.*total=\([0-9.]*\).*/\1/p')
  hot_total=$(grep -E "phase_ms:.*hot=1" "$log" 2>/dev/null | tail -1 | sed -n 's/.*total=\([0-9.]*\).*/\1/p')
  cold_te=$(grep -E "phase_ms:.*hot=0" "$log" 2>/dev/null | head -1 | sed -n 's/.*te=\([0-9.]*\).*/\1/p')
  hot_dit=$(grep -E "phase_ms:.*hot=1" "$log" 2>/dev/null | tail -1 | sed -n 's/.*dit=\([0-9.]*\).*/\1/p')
  awk -v ct="$cold_total" -v ht="$hot_total" -v cte="$cold_te" -v hd="$hot_dit" -v pw="${proc_wall:-na}" 'BEGIN{
    if(ct!="") ct=sprintf("%.2f", ct/1000); else ct="na"
    if(ht!="") ht=sprintf("%.2f", ht/1000); else ht="na"
    if(cte!="") cte=sprintf("%.1f", cte/1000); else cte="na"
    if(hd!="") hd=sprintf("%.1f", hd/1000); else hd="na"
    printf "proc_wall=%s;hot_total=%s;cold_total=%s;cold_te=%s;hot_dit=%s\n", pw, ht, ct, cte, hd
  }'
}

run_one() {
  local NAME="$1" SEC="$2"
  local OUT="$OUT_ROOT/$NAME"
  mkdir -p "$OUT/labeled"
  echo "==== $NAME sec=$SEC repeat=2 $(date '+%H:%M:%S') ====" | tee -a "$OUT_ROOT/META.txt"
  set +e
  /usr/bin/time -f "WALL_SEC=%e" "$PG" --sku rc0 --seconds "$SEC" --repeat 2 \
    --seed "$SEED" --prompt-ids "$IDS" --out "$OUT" --mux "$OUT/${NAME}.mp4" \
    >"$OUT/generate.log" 2>&1
  RC=$?
  set -e
  echo "RUN_RC=$RC" | tee -a "$OUT/generate.log"
  grep -E "phase_ms|WALL_SEC|fewstep_|sampler:|H3_PRODUCT|error|Error|FAIL" "$OUT/generate.log" \
    | tee "$OUT/extract.txt" || true
  TIMING=$(extract_hot "$OUT/generate.log")
  eval "${TIMING//;/ }"
  if [[ "$hot_total" == "na" || -z "$hot_total" ]]; then
    echo "FAIL: $NAME no hot=1 — refuse warm KPI" | tee -a "$OUT_ROOT/META.txt"
    return 1
  fi
  # Absurd vs baseline: 5s expect ~25s; 1s expect roughly ≤8s warm e2e
  if awk -v s="$SEC" -v h="$hot_total" 'BEGIN{exit !(h+0 > s*12)}'; then
    echo "ALERT_KPI_TIMING: $NAME hot_total=$hot_total for ${SEC}s (>12×) REFUSE" | tee -a "$OUT_ROOT/META.txt"
    return 2
  fi
  LABEL="RC0 864x480 DMAD-4 INT8 | ${SEC}s hot ${hot_total}s dit${hot_dit}s"
  if awk -v p="${proc_wall:-0}" -v h="$hot_total" 'BEGIN{exit !(p+0 > h*1.25 && p+0>0)}'; then
    LABEL="${LABEL} wall${proc_wall}s"
  fi
  SRC=$(ls -1 "$OUT"/*_r2.mp4 2>/dev/null | head -1 || true)
  [[ -z "$SRC" ]] && SRC=$(ls -1t "$OUT"/*.mp4 2>/dev/null | head -1)
  "$ROOT/tools/burn_eval_label.sh" "$SRC" "$OUT/labeled/${NAME}_labeled.mp4" "$LABEL"
  cp -f "$OUT/labeled/${NAME}_labeled.mp4" "$SHARE/labeled/"
  cp -f "$SRC" "$SHARE/${NAME}.mp4"
  cp -f "$OUT/extract.txt" "$SHARE/logs/"
  echo "$NAME RC=$RC SEC=$SEC hot_total=$hot_total hot_dit=$hot_dit cold_total=$cold_total cold_te=$cold_te proc_wall=$proc_wall LABEL=$LABEL" \
    | tee -a "$OUT_ROOT/META.txt"
  # Also write machine-readable row
  echo "$NAME,$SEC,$hot_total,$hot_dit,$cold_total,$cold_te,$proc_wall" >> "$OUT_ROOT/kpi.csv"
  return "$RC"
}

echo "name,sec,hot_total_s,hot_dit_s,cold_total_s,cold_te_s,proc_wall_s" > "$OUT_ROOT/kpi.csv"
set +e
run_one rc0_dmad_1s 1; R1=$?
run_one rc0_dmad_5s 5; R5=$?
set -e

{
  echo "SUMMARY R1=$R1 R5=$R5"
  echo "kpi_csv: $OUT_ROOT/kpi.csv"
  echo "share: $SHARE"
  cat "$OUT_ROOT/kpi.csv"
} | tee -a "$OUT_ROOT/META.txt"
cp -f "$OUT_ROOT/META.txt" "$OUT_ROOT/kpi.csv" "$SHARE/"
# Mirror to box-visible share if present
if [[ -d /workspace ]]; then
  mkdir -p /workspace/rc0_dmad_warm_kpi_latest
  cp -f "$SHARE/labeled/"*.mp4 /workspace/rc0_dmad_warm_kpi_latest/ 2>/dev/null || true
  cp -f "$OUT_ROOT/META.txt" "$OUT_ROOT/kpi.csv" /workspace/rc0_dmad_warm_kpi_latest/ 2>/dev/null || true
fi
exit 0
