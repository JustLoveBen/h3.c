#!/usr/bin/env bash
# tools/h3_dp_generate.sh — request-level data parallel (Phase 4B).
# One h3_cuda_generate per GPU, --ulysses 1. No MPS, no MIG, no P2P.
# Per-GPU TE cache ($H3_TE_CACHE/gpu$i). Never share the live cache dir.
# Never pkill. Never print passwords.
# Refuses >2 GPUs at --seconds>=5 unless --i-know (do not fill the box).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GPUS="0"
SECONDS=1
STEPS=""
ARM="turbo4_te"
PROMPT="p01_sprint"
DRY=0
I_KNOW=0
WIDTH=1344
HEIGHT=768

usage() {
  cat <<'EOF'
tools/h3_dp_generate.sh — request-level DP (not Ulysses)

  --gpus 0,1           2-way smoke default example
  --arm turbo4_te|turbo4_te_tae|gold|turbo4
  --prompt p01_sprint
  --seconds 1          smoke default; 5 = 768P ruler
  --steps N
  --width 1344 --height 768
  --dry-run
  --i-know             allow >2 GPUs when --seconds>=5
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --gpus) GPUS="$2"; shift 2 ;;
    --arm) ARM="$2"; shift 2 ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    --seconds) SECONDS="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --width) WIDTH="$2"; shift 2 ;;
    --height) HEIGHT="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --i-know) I_KNOW=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
N=${#GPU_ARR[@]}
if [ "$N" -lt 1 ]; then
  echo "need --gpus" >&2
  exit 2
fi
if [ "$I_KNOW" != 1 ] && [ "$N" -gt 2 ] && [ "$SECONDS" -ge 5 ]; then
  echo "refusing ${N}-way ${SECONDS}s. 2-way smoke, or --i-know." >&2
  exit 2
fi

H3_MODELS="${H3_MODELS:-/home/user/models}"
GEN="${H3_GENERATE:-$ROOT/build-cuda-sm120/h3_cuda_generate}"
TE_CACHE_ROOT="${H3_TE_CACHE:-$ROOT/out/verify_768p/te_cache}"
DIT="${DIT:-$H3_MODELS/minimax_h3_fl2va_pruned_int8_convrot.safetensors}"
VAE="${VAE:-$H3_MODELS/minimax_h3_video_vae_fp16.safetensors}"
TE="${TE:-$H3_MODELS/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors}"
AUDIO="${AUDIO:-$H3_MODELS/minimax_h3_audio_vae_fp32.safetensors}"
LORA="${H3_TURBO_LORA:-$H3_MODELS/minimax_h3_turbo_v4_step600_ema.safetensors}"
IDS="$ROOT/evals/768p/ids/${PROMPT}.ids"

if [ "$WIDTH" -eq 1344 ] && [ "$HEIGHT" -eq 768 ] && [ ! -s "$IDS" ]; then
  echo "missing $IDS (no cafe fallback)" >&2
  exit 2
fi

case "$PROMPT" in
  p01_sprint) SEED=0x76800001 ;;
  *) SEED=0x76800000 ;;
esac

steps_use="$STEPS"
extra=()
case "$ARM" in
  gold)
    steps_use="${steps_use:-8}"
    extra=(--steps "$steps_use" --cfg 1 --sdpa-backend cudnn)
    ;;
  turbo4)
    steps_use="${steps_use:-4}"
    extra=(--preset turbo4 --lora "$LORA" --steps "$steps_use")
    ;;
  turbo4_te)
    steps_use="${steps_use:-4}"
    extra=(--preset turbo4 --lora "$LORA" --steps "$steps_use")
    ;;
  turbo4_te_tae)
    steps_use="${steps_use:-4}"
    TAE="${H3_TAE:-$H3_MODELS/taeh3.safetensors}"
    extra=(--preset turbo4 --lora "$LORA" --steps "$steps_use"
           --vae-decode tae --tae "$TAE")
    ;;
  *)
    echo "dp arm $ARM not in gold|turbo4|turbo4_te|turbo4_te_tae" >&2
    exit 2
    ;;
esac

if [ -z "${H3_SAGE3_LIB:-}" ] && [ -n "${H3_SAGE3_ROOT:-}" ] && \
   [ -d "${H3_SAGE3_ROOT}/sageattn3" ] && \
   [ -f "$ROOT/build-cuda-sm120/libh3_sage3.so" ]; then
  export H3_SAGE3_LIB="$ROOT/build-cuda-sm120/libh3_sage3.so"
fi

echo "dp: n=$N gpus=${GPUS} arm=$ARM prompt=$PROMPT ${WIDTH}x${HEIGHT} ${SECONDS}s ulysses=1"

pids=()
gpu_ids=()
for i in "${GPU_ARR[@]}"; do
  gpu_ids+=("$i")
  out="$ROOT/out/verify_768p/dp/gpu${i}/${ARM}/${PROMPT}"
  mux="$ROOT/out/review_videos_768p/dp_gpu${i}_${ARM}_${PROMPT}_${SECONDS}s.mp4"
  cache="$TE_CACHE_ROOT/gpu${i}"
  mkdir -p "$out" "$cache" "$ROOT/out/review_videos_768p"
  if [ -d "$TE_CACHE_ROOT" ]; then
    find "$TE_CACHE_ROOT" -maxdepth 1 -type f -name 'h3te_*.bin' -exec cp -n {} "$cache"/ \;
  fi
  cmd=("$GEN"
    --dit "$DIT" --vae "$VAE" --te "$TE" --audio-vae "$AUDIO"
    --out "$out" --mux "$mux"
    --width "$WIDTH" --height "$HEIGHT" --seconds "$SECONDS"
    --seed "$SEED" --ulysses 1
    "${extra[@]}")
  if [ "$ARM" = "turbo4_te" ] || [ "$ARM" = "turbo4_te_tae" ]; then
    cmd+=(--te-cache "$cache")
  fi
  if [ -s "$IDS" ]; then
    cmd+=(--token-ids-file "$IDS")
  fi
  if [ "$DRY" = 1 ]; then
    echo "dry-run GPU$i mux=$mux"
    printf '  '
    printf '%q ' "${cmd[@]}"
    printf '\n'
    continue
  fi
  if [ ! -x "$GEN" ]; then
    echo "generate missing: $GEN" >&2
    exit 2
  fi
  (
    export CUDA_VISIBLE_DEVICES="$i"
    "${cmd[@]}"
  ) > "$out/generate.log" 2>&1 &
  pids+=("$!")
  echo "dp: launched gpu=$i pid=$! log=$out/generate.log mux=$mux"
done

if [ "$DRY" = 1 ]; then
  exit 0
fi

fail=0
for idx in "${!pids[@]}"; do
  pid="${pids[$idx]}"
  g="${gpu_ids[$idx]}"
  if ! wait "$pid"; then
    echo "dp: GPU$g pid=$pid FAIL" >&2
    tail -8 "$ROOT/out/verify_768p/dp/gpu${g}/${ARM}/${PROMPT}/generate.log" >&2 || true
    fail=1
  else
    echo "dp: GPU$g pid=$pid ok"
  fi
done
if [ "$fail" = 1 ]; then
  exit 1
fi
echo "dp: all $N ranks ok"
