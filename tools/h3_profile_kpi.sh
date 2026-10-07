#!/usr/bin/env bash
# KPI-A / KPI-B profile protocol (opt-dev R56).
# Warmup is clip 1 of in-process --repeat (not counted in P50/P95).
# P50/P95 only from `phase_ms total=` with hot=1.
# Never print passwords. Never pkill someone else's generate.
# Does not claim KPI-B ≤27.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KPI=""
REPEATS=2
NSYS=0
OUT=""
DRY=0
MEM_TRACE=0
IDS=""
SEED=""
PROMPT_ID=""
DMON=1

usage() {
  cat <<'EOF'
tools/h3_profile_kpi.sh — KPI-A/B profile (opt-dev R56)

  --kpi a|b          required
  --repeats N        in-process --repeat N (clip1=cold, clip2..=hot). default 2
  --nsys 0|1         wrap one nsys profile (default 0)
  --out DIR          required unless --dry-run
  --dry-run          print argv/env; no GPU
  --mem-trace        H3_GEN_MEM_TRACE=1
  --no-dmon          skip nvidia-smi dmon
  --ids FILE         token ids (default: input/cafe/prompt.ids)
  --seed N           noise seed
  --prompt-id ID     evals/768p id (sets ids + seed 0x76800000+index)

Env:
  H3_MODELS  DIT  VAE  TE  AUDIO  H3_TURBO_LORA  H3_LATENT_UPSCALER
  H3_GENERATE  H3_TE_CACHE  H3_SKIP_PPM (default 1)
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --kpi) KPI="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --nsys) NSYS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --mem-trace) MEM_TRACE=1; shift ;;
    --no-dmon) DMON=0; shift ;;
    --ids) IDS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --prompt-id) PROMPT_ID="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [ "$KPI" != "a" ] && [ "$KPI" != "b" ]; then
  echo "need --kpi a|b" >&2
  exit 2
fi
if ! [[ "$REPEATS" =~ ^[0-9]+$ ]] || [ "$REPEATS" -lt 1 ]; then
  echo "need --repeats N>=1" >&2
  exit 2
fi
if [ "$NSYS" != "0" ] && [ "$NSYS" != "1" ]; then
  echo "need --nsys 0|1" >&2
  exit 2
fi
if [ "$DRY" != 1 ] && [ -z "$OUT" ]; then
  echo "need --out DIR" >&2
  exit 2
fi

prompt_index() {
  case "$1" in
    p01_sprint) echo 1 ;;
    p02_talking_head) echo 2 ;;
    p03_corridor) echo 3 ;;
    p04_volumetric) echo 4 ;;
    p05_neon_sign) echo 5 ;;
    p06_orbit) echo 6 ;;
    p07_market_track) echo 7 ;;
    p08_ballroom_dolly) echo 8 ;;
    *) echo "" ;;
  esac
}

if [ -n "$PROMPT_ID" ]; then
  idx="$(prompt_index "$PROMPT_ID")"
  if [ -z "$idx" ]; then
    echo "unknown --prompt-id $PROMPT_ID" >&2
    exit 2
  fi
  IDS="${IDS:-$ROOT/evals/768p/ids/${PROMPT_ID}.ids}"
  if [ -z "$SEED" ]; then
    SEED=$((0x76800000 + idx))
  fi
fi

pick_file() {
  local p last=""
  for p in "$@"; do
    [ -n "$p" ] || continue
    last="$p"
    if [ -f "$p" ]; then
      printf '%s' "$p"
      return 0
    fi
  done
  printf '%s' "$last"
}

H3_MODELS="${H3_MODELS:-}"
if [ -z "$H3_MODELS" ]; then
  for cand in \
    /mnt/nvme1n1/workspace/models \
    /mnt/nvme1n1/workspace/h3_c/models \
    "$ROOT/models"; do
    if [ -d "$cand" ]; then
      H3_MODELS="$cand"
      break
    fi
  done
fi
H3_MODELS="${H3_MODELS:-$ROOT/models}"

DIT="$(pick_file "${DIT:-}" \
  "$H3_MODELS/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" \
  "$H3_MODELS/minimax_h3_fl2va_pruned_int8_convrot.safetensors")"
VAE="$(pick_file "${VAE:-}" \
  "$H3_MODELS/vae/minimax_h3_video_vae_fp16.safetensors" \
  "$H3_MODELS/minimax_h3_video_vae_fp16.safetensors")"
TE="$(pick_file "${TE:-}" \
  "$H3_MODELS/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" \
  "$H3_MODELS/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")"
AUDIO="$(pick_file "${AUDIO:-}" \
  "$H3_MODELS/vae/minimax_h3_audio_vae_fp32.safetensors" \
  "$H3_MODELS/minimax_h3_audio_vae_fp32.safetensors")"
LORA="$(pick_file "${H3_TURBO_LORA:-}" "${H3_TAOMATE_LORA:-}" \
  "$H3_MODELS/loras/minimax_h3_turbo_v4_step600_ema.safetensors" \
  "$H3_MODELS/minimax_h3_turbo_v4_step600_ema.safetensors")"
UPS="$(pick_file "${H3_LATENT_UPSCALER:-}" \
  "$H3_MODELS/latent_upscale_models/minimax_h3_latent_upscaler_3d_bf16.safetensors" \
  "$H3_MODELS/minimax_h3_latent_upscaler_3d_bf16.safetensors")"
GEN="${H3_GENERATE:-$ROOT/build-cuda-sm120/h3_cuda_generate}"
IDS="${IDS:-$ROOT/input/cafe/prompt.ids}"
TE_CACHE="${H3_TE_CACHE:-$ROOT/out/te_cache_kpi_a}"
SHA="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo nogit)"
export H3_SKIP_PPM="${H3_SKIP_PPM:-1}"

if [ "$KPI" = "a" ]; then
  WIDTH=864
  HEIGHT=480
  PRESET=turbo4
  STEPS=4
else
  WIDTH=1376
  HEIGHT=768
  PRESET=taomate-lift
  STEPS=5
fi

OUT="${OUT:-$ROOT/out/profile_kpi_${KPI}_dry}"
mkdir -p "$OUT" 2>/dev/null || true
LOG="$OUT/${KPI}_repeat${REPEATS}_${SHA}.log"
REPORT="$OUT/${KPI}_${SHA}.json"
MUX="$OUT/${KPI}.mp4"
NSYS_OUT="$OUT/${KPI}_hot_${SHA}_1"

cmd=("$GEN"
  --dit "$DIT" --vae "$VAE" --te "$TE" --audio-vae "$AUDIO"
  --lora "$LORA"
  --out "$OUT/gen" --mux "$MUX"
  --width "$WIDTH" --height "$HEIGHT" --seconds 5
  --preset "$PRESET" --steps "$STEPS"
  --sdpa-backend cudnn --vae-decode official
  --te-cache "$TE_CACHE"
  --token-ids-file "$IDS"
  --repeat "$REPEATS"
  --report "$REPORT")
if [ "$KPI" = "b" ]; then
  cmd+=(--latent-upscaler "$UPS")
fi
if [ -n "$SEED" ]; then
  cmd+=(--seed "$SEED")
fi

echo "profile_kpi: kpi=$KPI repeats=$REPEATS nsys=$NSYS sha=$SHA canvas=${WIDTH}x${HEIGHT} preset=$PRESET steps=$STEPS"
echo "profile_kpi: gen=$GEN"
echo "profile_kpi: ids=$IDS"
echo "profile_kpi: out=$OUT"
printf 'profile_kpi: argv='
printf ' %q' "${cmd[@]}"
printf '\n'
echo "profile_kpi: note KPI-B ≤27 is a hard wall (opt-dev §8.8); this script only records totals"

if [ "$DRY" = 1 ]; then
  echo "profile_kpi: dry-run (no GPU)"
  exit 0
fi

if [ ! -x "$GEN" ]; then
  echo "missing generate binary: $GEN" >&2
  exit 2
fi
if [ ! -f "$IDS" ]; then
  echo "missing token ids: $IDS" >&2
  exit 2
fi
if [ ! -f "$DIT" ] || [ ! -f "$VAE" ] || [ ! -f "$TE" ] || [ ! -f "$AUDIO" ] || [ ! -f "$LORA" ]; then
  echo "missing weights under H3_MODELS=$H3_MODELS" >&2
  exit 2
fi
if [ "$KPI" = "b" ] && [ ! -f "$UPS" ]; then
  echo "KPI-B needs LBH upscaler: $UPS" >&2
  exit 2
fi

mkdir -p "$OUT/gen" "$TE_CACHE"
if [ "$MEM_TRACE" = 1 ]; then
  export H3_GEN_MEM_TRACE=1
fi

dmon_pid=""
if [ "$DMON" = 1 ] && command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -q -d CLOCK,PERFORMANCE >"$OUT/${KPI}_clock_before.txt" 2>/dev/null || true
  nvidia-smi dmon -s pucvt -o DT >"$OUT/${KPI}_dmon.txt" 2>/dev/null &
  dmon_pid=$!
fi

run_rc=0
set +e
if [ "$NSYS" = 1 ] && command -v nsys >/dev/null 2>&1; then
  nsys profile --trace=cuda,nvtx --sample=none \
    --cuda-graph-trace=node \
    --output="$NSYS_OUT" --force-overwrite true \
    "${cmd[@]}" >"$LOG" 2>&1
  run_rc=$?
else
  "${cmd[@]}" >"$LOG" 2>&1
  run_rc=$?
fi
set -e

if [ -n "$dmon_pid" ]; then
  kill "$dmon_pid" 2>/dev/null || true
  wait "$dmon_pid" 2>/dev/null || true
  nvidia-smi -q -d CLOCK,PERFORMANCE >"$OUT/${KPI}_clock_after.txt" 2>/dev/null || true
fi

python3 - "$LOG" "$OUT/${KPI}_stats.txt" <<'PY'
import re, sys
log, outp = sys.argv[1], sys.argv[2]
colds, hots = [], []
n_phase = 0
with open(log, encoding="utf-8", errors="replace") as f:
    for line in f:
        if "phase_ms:" not in line or "total=" not in line:
            continue
        n_phase += 1
        tm = re.search(r"total=([0-9.]+)", line)
        hm = re.search(r"hot=([01])", line)
        if not tm:
            continue
        total = float(tm.group(1))
        hot = int(hm.group(1)) if hm else 0
        (hots if hot else colds).append(total)

def pct(xs, p):
    if not xs:
        return None
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * p / 100.0
    i = int(k)
    f = k - i
    if i + 1 >= len(ys):
        return ys[-1]
    return ys[i] * (1 - f) + ys[i + 1] * f

def fmt(v):
    return "na" if v is None else f"{v/1000.0:.3f}s"

lines = [
    f"phase_ms_lines={n_phase}",
    f"cold_n={len(colds)} cold_total={fmt(colds[0] if colds else None)}",
    f"hot_n={len(hots)} hot_p50={fmt(pct(hots, 50))} hot_p95={fmt(pct(hots, 95))}",
    f"hot_min={fmt(min(hots) if hots else None)} hot_max={fmt(max(hots) if hots else None)}",
    "kpi_source=phase_ms total= (hot=1 only)",
    "do_not_claim_kpi_b_le_27=1",
]
text = "\n".join(lines) + "\n"
open(outp, "w").write(text)
sys.stdout.write(text)
PY

# Copy naming from R56: DIR/<kpi>_<hot|cold>_<sha>_<n>.{log,json}
if [ -f "$REPORT.clip1.json" ]; then
  cp -f "$REPORT.clip1.json" "$OUT/${KPI}_cold_${SHA}_1.json"
fi
if [ -f "$REPORT.clip2.json" ]; then
  cp -f "$REPORT.clip2.json" "$OUT/${KPI}_hot_${SHA}_2.json"
fi
cp -f "$LOG" "$OUT/${KPI}_hot_${SHA}_1.log" 2>/dev/null || true

echo "profile_kpi: exit=$run_rc log=$LOG"
exit "$run_rc"
