#!/usr/bin/env bash
# C073 dump-once → score-many GPU acceptance on Ada 4090.
# Gate: 480p × 5s, complex p10, INT8 product default.
# Scheme: Euler-8 primary (t71u); DMAD ok; PDMD contrast-only — not used here.
# Flow: DUMP (DiT→H3LAT1, no VAE) → REPLAY async=on → REPLAY async=off
#       → dump_once_score_many_check + waffle_same_config_gate (same-config).
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
      H3_DIT_NVFP4_HOST_DEQUANT H3_TE_CACHE_AUTO H3_VRAM_LEDGER \
      H3_DUMP_LATENT H3_REPLAY_LATENT H3_DUMP_ONLY H3_DUMP_DIR \
      H3_VAE_UNPATCH_ASYNC
export H3_SKIP_PPM=1
export LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}

BIN="$ROOT/build-cuda/h3_cuda_generate"
MODELS="${H3_MODELS:-/home/haiyang/workspace/models}"
DIT="$MODELS/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
VAE="$MODELS/minimax_h3_video_vae_fp16.safetensors"
TE="$MODELS/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
OUT_ROOT="${OUT_ROOT:-$ROOT/out/c073_dump_replay_480p_20261028}"
SEED=0x76800002
IDS="$ROOT/evals/768p/ids/p10_street_wave_dolly.ids"
W=864; H=480; SEC=5; STEPS=8
DUMP="$OUT_ROOT/latent_dump.h3lat"
SCORE="$OUT_ROOT/score"
CHECK="$ROOT/tools/dump_once_score_many_check.py"
WAFFLE="$ROOT/tools/waffle_same_config_gate.py"

for f in "$BIN" "$DIT" "$VAE" "$TE" "$IDS" "$CHECK" "$WAFFLE"; do
  [[ -s "$f" ]] || { echo "missing $f" >&2; exit 2; }
done
rm -rf "$OUT_ROOT"
mkdir -p "$OUT_ROOT" "$SCORE" "$OUT_ROOT/labeled"
STAMP=$(date '+%Y-%m-%d %H:%M:%S %Z')
{
  echo "date_local: $STAMP"
  echo "host: $(hostname) / RTX 4090 sm_89"
  echo "repo: $ROOT"
  echo "git_sha: $(cat $ROOT/.synced_from_mac_sha 2>/dev/null || echo unknown)"
  echo "catalog: C073 dump-once/score-many GPU acceptance"
  echo "product_default: INT8"
  echo "NVFP4: BLOCK/unset"
  echo "FaceRefine: STOPPED"
  echo "seed: $SEED"
  echo "gate: ${W}x${H} seconds=${SEC} steps=${STEPS} complex=p10 skip-audio"
  echo "scheme: Euler-8 INT8 (primary; DMAD ok; PDMD contrast-only per t71u)"
  echo "fewstep: unset (not PDMD)"
  echo "dump: H3_DUMP_LATENT → $DUMP"
  echo "score: two replays H3_VAE_UNPATCH_ASYNC=1 then =0"
} | tee "$OUT_ROOT/META.txt"
python3 "$ROOT/tools/print_h3_env_contract.py" --markdown >> "$OUT_ROOT/META.txt" 2>/dev/null || true

mkdir -p "$OUT_ROOT/dump"
echo "==== DUMP $(date '+%H:%M:%S') ===="
export H3_DUMP_LATENT="$DUMP"
set +e
/usr/bin/time -f "WALL_SEC=%e" "$BIN" \
  --dit "$DIT" --vae "$VAE" --te "$TE" \
  --width "$W" --height "$H" --seconds "$SEC" --steps "$STEPS" --cfg 1 \
  --seed "$SEED" --sdpa-backend cudnn --skip-audio \
  --token-ids-file "$IDS" \
  --out "$OUT_ROOT/dump" \
  >"$OUT_ROOT/dump/generate.log" 2>&1
RC=$?
set -e
unset H3_DUMP_LATENT
echo "DUMP_RC=$RC" | tee -a "$OUT_ROOT/dump/generate.log"
grep -E 'dump_latent|dump_once_score_many|H3_DUMP_ONCE|WALL_SEC|error|Error|FAIL' \
  "$OUT_ROOT/dump/generate.log" | tee "$OUT_ROOT/dump/KPI.txt" || true
[[ -s "$DUMP" ]] || { echo "FAIL: no dump $DUMP" >&2; tail -80 "$OUT_ROOT/dump/generate.log" >&2; exit 3; }

python3 "$CHECK" "$DUMP" | tee "$OUT_ROOT/dump/check_latent.txt"
python3 "$CHECK" --log "$OUT_ROOT/dump/generate.log" | tee "$OUT_ROOT/dump/check_log.txt"

run_replay() {
  local ARM=$1 ASYNC=$2
  local OUT="$SCORE/$ARM"
  mkdir -p "$OUT"
  echo "==== REPLAY $ARM async=$ASYNC $(date '+%H:%M:%S') ===="
  export H3_REPLAY_LATENT="$DUMP"
  export H3_VAE_UNPATCH_ASYNC="$ASYNC"
  set +e
  /usr/bin/time -f "WALL_SEC=%e" "$BIN" \
    --dit "$DIT" --vae "$VAE" --te "$TE" \
    --width "$W" --height "$H" --seconds "$SEC" --steps "$STEPS" --cfg 1 \
    --seed "$SEED" --sdpa-backend cudnn --skip-audio \
    --token-ids-file "$IDS" \
    --out "$OUT" --mux "$OUT/${ARM}.mp4" \
    >"$OUT/generate.log" 2>&1
  RC=$?
  set -e
  echo "RUN_RC=$RC" | tee -a "$OUT/generate.log"
  unset H3_REPLAY_LATENT H3_VAE_UNPATCH_ASYNC
  grep -E 'replay_latent|dump_once_score_many|H3_SCORE_MANY|unpatch_async|waffle|WALL_SEC|phase_ms|checksum|error|Error|FAIL' \
    "$OUT/generate.log" | tee "$OUT/KPI.txt" || true
  ls -lh "$OUT/${ARM}.mp4" 2>/dev/null || echo "NO_MP4 $ARM"
  python3 "$CHECK" --log "$OUT/generate.log" | tee "$OUT/check_log.txt"
  if [[ "$RC" != 0 ]]; then
    echo "FAIL replay $ARM rc=$RC" >&2
    tail -80 "$OUT/generate.log" >&2
    return "$RC"
  fi
  return 0
}

run_replay replay_async_on 1
run_replay replay_async_off 0

echo "==== waffle_same_config_gate async_on vs async_off ===="
set +e
python3 "$WAFFLE" \
  "$SCORE/replay_async_on/generate.log" \
  "$SCORE/replay_async_off/generate.log" \
  | tee "$SCORE/waffle_gate.txt"
WAFFLE_RC=$?
set -e
echo "WAFFLE_RC=$WAFFLE_RC" | tee -a "$SCORE/waffle_gate.txt"

# Labels
for ARM in replay_async_on replay_async_off; do
  IN="$SCORE/$ARM/${ARM}.mp4"
  [[ -s "$IN" ]] || continue
  WALL=$(grep -E '^WALL_SEC=' "$SCORE/$ARM/generate.log" | tail -1 | cut -d= -f2)
  LABEL="864x480 C073 ${ARM} INT8 | WALL ${WALL}s"
  bash "$ROOT/tools/burn_eval_label.sh" "$IN" "$OUT_ROOT/labeled/${ARM}_labeled.mp4" "$LABEL" \
    || echo "label failed $ARM"
done

{
  echo "# C073 dump→replay acceptance"
  echo "date: $STAMP"
  echo "dump: $DUMP"
  echo "dump_check:"; cat "$OUT_ROOT/dump/check_latent.txt"
  echo
  echo "## replay_async_on"; cat "$SCORE/replay_async_on/KPI.txt" 2>/dev/null || true
  echo
  echo "## replay_async_off"; cat "$SCORE/replay_async_off/KPI.txt" 2>/dev/null || true
  echo
  echo "## waffle_same_config_gate"; cat "$SCORE/waffle_gate.txt"
  echo "WAFFLE_RC=$WAFFLE_RC"
} | tee "$OUT_ROOT/SUMMARY.md"

if [[ "$WAFFLE_RC" != 0 ]]; then
  echo "C073 waffle gate FAIL (async on vs off)" >&2
  exit "$WAFFLE_RC"
fi
echo DONE
ls -lh "$DUMP" "$SCORE"/*/*.mp4 "$OUT_ROOT/labeled"/*.mp4 2>/dev/null || true
