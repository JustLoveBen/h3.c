#!/usr/bin/env bash
# DMAD-4 vs Euler-8 INT8 KPI @ 480p×5s complex p10 (Ada 4090).
# Policy (t71u): DMAD or Euler primary; PDMD contrast-only — skipped here.
# Arms (both --repeat 2, H3_TE_CACHE_AUTO=1 for warm + TE HIT):
#   A) Euler-8 INT8   (no fewstep)
#   B) DMAD-4 INT8    --fewstep-preset dmad4 + full_critic LoRA
# No FaceRefine. No NVFP4. No KEEP=attn. No PDMD.
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
      H3_DIT_NVFP4_HOST_DEQUANT H3_FEWSTEP_PRESET H3_FEWSTEP_CONTRACT
export H3_SKIP_PPM=1
export H3_TE_CACHE_AUTO=1
export LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}

BIN="$ROOT/build-cuda/h3_cuda_generate"
MODELS="${H3_MODELS:-/home/haiyang/workspace/models}"
DIT="$MODELS/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
VAE="$MODELS/minimax_h3_video_vae_fp16.safetensors"
TE="$MODELS/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
DMAD_LORA="$MODELS/loras/dmad_minimax_h3_4step_full_critic.h3repo.safetensors"
OUT_ROOT="${OUT_ROOT:-$ROOT/out/dmad4_vs_euler8_kpi_480p_20261028}"
SEED=0x76800002
IDS="$ROOT/evals/768p/ids/p10_street_wave_dolly.ids"
W=864; H=480; SEC=5

for f in "$BIN" "$DIT" "$VAE" "$TE" "$IDS" "$DMAD_LORA"; do
  [[ -s "$f" ]] || { echo "missing $f" >&2; exit 2; }
done
mkdir -p "$OUT_ROOT"
STAMP=$(date '+%Y-%m-%d %H:%M:%S %Z')
{
  echo "date_local: $STAMP"
  echo "host: $(hostname) / RTX 4090 sm_89"
  echo "repo: $ROOT"
  echo "git_sha: $(cat $ROOT/.synced_from_mac_sha 2>/dev/null || echo unknown)"
  echo "arch: sm_89"
  echo "product_default: INT8"
  echo "KEEP=attn: off"
  echo "NVFP4: BLOCK / unset"
  echo "FaceRefine: STOPPED"
  echo "scheme_policy_t71u: DMAD or Euler primary; PDMD contrast-only (skipped)"
  echo "seed: $SEED"
  echo "gate: ${W}x${H} seconds=${SEC} complex=p10_street_wave_dolly"
  echo "env: H3_TE_CACHE_AUTO=1 H3_SKIP_PPM=1 --repeat 2 --skip-audio"
  echo "C007_note: sol_cute_lib=0 want=0 on cudnn product path → N/A this host"
} | tee "$OUT_ROOT/META.txt"

run_arm() {
  local ARM=$1 STEPS=$2; shift 2
  local EXTRA=("$@")
  local OUT="$OUT_ROOT/$ARM"
  mkdir -p "$OUT"
  echo "==== $ARM generate $(date '+%H:%M:%S') ===="
  set +e
  /usr/bin/time -f "WALL_SEC=%e" "$BIN" \
    --dit "$DIT" --vae "$VAE" --te "$TE" \
    --skip-audio \
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
    grep -E 'dit_format|te_prompt_cache|te_encode:|fewstep_|sampler:|phase_ms|WALL_SEC|lora=|h3_lora: applied|error|Error|FAIL|sol_cute stats|sdpa_actual' \
      "$OUT/generate.log" || true
  } | tee "$OUT/KPI.txt"
  ls -lh "$OUT"/*.mp4 2>/dev/null || echo "NO_MP4 $ARM"
  if [[ "$RC" != 0 ]]; then
    echo "FAIL $ARM rc=$RC" >&2
    tail -80 "$OUT/generate.log" >&2
    return "$RC"
  fi
  return 0
}

# Arm A: Euler-8 INT8 primary
run_arm euler8_int8_teauto 8

# Arm B: DMAD-4 INT8 primary (lcm_renoise + full_critic LoRA)
run_arm dmad4_int8_teauto 4 \
  --fewstep-preset dmad4 \
  --lora "$DMAD_LORA" --lora-scale 1

# Labels: scheme + warm (clip2 / second WALL share)
LABEL_DIR="$OUT_ROOT/labeled"
mkdir -p "$LABEL_DIR"
for ARM in euler8_int8_teauto dmad4_int8_teauto; do
  IN="$OUT_ROOT/$ARM/${ARM}.mp4"
  if [[ ! -s "$IN" ]]; then
    IN=$(ls "$OUT_ROOT/$ARM"/*.mp4 2>/dev/null | head -1 || true)
  fi
  [[ -s "$IN" ]] || continue
  WALL=$(grep -E '^WALL_SEC=' "$OUT_ROOT/$ARM/generate.log" | tail -1 | cut -d= -f2)
  # Prefer last (warm) phase_ms total if two clips; else WALL/2 proxy
  WARM=$(grep -E '^phase_ms:' "$OUT_ROOT/$ARM/generate.log" | tail -1 \
    | sed -n 's/.*total=\([0-9.]*\).*/\1/p')
  if [[ -z "$WARM" ]]; then
    WARM=$(python3 -c "print(round(float('$WALL')/2, 2))" 2>/dev/null || echo "$WALL")
  else
    WARM=$(python3 -c "print(round(float('$WARM')/1000.0, 2))" 2>/dev/null || echo "$WARM")
  fi
  TE_HIT=$(grep -c 'te_prompt_cache: HIT' "$OUT_ROOT/$ARM/generate.log" || true)
  # AUTO path may omit HIT string; warm te=0.0 counts as reuse
  if [[ "$TE_HIT" == 0 ]]; then
    WTE=$(grep -E '^phase_ms:' "$OUT_ROOT/$ARM/generate.log" | tail -1 | sed -n 's/.*te=\([0-9.]*\).*/\1/p')
    python3 -c "import sys; sys.exit(0 if float('${WTE:-1}')==0 else 1)" 2>/dev/null && TE_HIT=1 || true
  fi
  # Prefer warm r2 mp4 for label visual
  if [[ -s "$OUT_ROOT/$ARM/${ARM}_r2.mp4" ]]; then
    IN="$OUT_ROOT/$ARM/${ARM}_r2.mp4"
  fi
  case "$ARM" in
    euler8*) SCHEME_NAME="Euler-8" ;;
    dmad4*)  SCHEME_NAME="DMAD-4" ;;
    *)       SCHEME_NAME="$ARM" ;;
  esac
  SCHEME="864x480 ${SCHEME_NAME} INT8 | warm ${WARM}s WALL ${WALL}s teReuse=${TE_HIT}"
  bash "$ROOT/tools/burn_eval_label.sh" "$IN" "$LABEL_DIR/${ARM}_labeled.mp4" "$SCHEME" \
    || echo "label failed $ARM"
done

{
  echo "# DMAD-4 vs Euler-8 KPI summary (t71u primary pair; PDMD skipped)"
  echo "date: $STAMP"
  echo "gate: 864x480 x 5s p10_street_wave_dolly INT8"
  echo
  printf '| arm | WALL_s | warm_total_s | dit_ms (last) | te_ms (last) | teHIT | steps | sampler |\n'
  printf '|-----|--------|--------------|---------------|--------------|-------|-------|----------|\n'
  for ARM in euler8_int8_teauto dmad4_int8_teauto; do
    WALL=$(grep -E '^WALL_SEC=' "$OUT_ROOT/$ARM/generate.log" 2>/dev/null | tail -1 | cut -d= -f2)
    LAST=$(grep -E '^phase_ms:' "$OUT_ROOT/$ARM/generate.log" 2>/dev/null | tail -1)
    DIT=$(echo "$LAST" | sed -n 's/.*dit=\([0-9.]*\).*/\1/p')
    TE=$(echo "$LAST" | sed -n 's/.*te=\([0-9.]*\).*/\1/p')
    TOT=$(echo "$LAST" | sed -n 's/.*total=\([0-9.]*\).*/\1/p')
    WARM_S=$(python3 -c "print(round(float('${TOT:-0}')/1000.0, 2))" 2>/dev/null)
    TE_HIT=$(grep -c 'te_prompt_cache: HIT' "$OUT_ROOT/$ARM/generate.log" 2>/dev/null || true)
    STEPS=$(grep -E 'fewstep_bound:|steps=' "$OUT_ROOT/$ARM/generate.log" 2>/dev/null | head -3)
    SAMP=$(grep -E '^sampler: selected=' "$OUT_ROOT/$ARM/generate.log" 2>/dev/null | head -1 \
      | sed -n 's/.*selected=\([^ ]*\).*/\1/p')
    NSTEPS=$(grep -E 'shape:.*steps=' "$OUT_ROOT/$ARM/generate.log" 2>/dev/null | head -1 \
      | sed -n 's/.*steps=\([0-9]*\).*/\1/p')
    printf '| %s | %s | %s | %s | %s | %s | %s | %s |\n' \
      "$ARM" "${WALL:-?}" "${WARM_S:-?}" "${DIT:-?}" "${TE:-?}" "${TE_HIT:-0}" "${NSTEPS:-?}" "${SAMP:-?}"
  done
  echo
  for ARM in euler8_int8_teauto dmad4_int8_teauto; do
    echo "## $ARM"
    [[ -f "$OUT_ROOT/$ARM/KPI.txt" ]] && cat "$OUT_ROOT/$ARM/KPI.txt"
    echo
  done
} | tee "$OUT_ROOT/SUMMARY.md"

echo DONE
ls -lh "$OUT_ROOT"/*/*.mp4 "$LABEL_DIR"/*.mp4 2>/dev/null || true
