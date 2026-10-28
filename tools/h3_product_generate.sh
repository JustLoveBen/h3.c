#!/usr/bin/env bash
# tools/h3_product_generate.sh — product generation entry (NFE × raise-canvas SKU).
#
# Operators pick a sellable SKU (or --nfe + --canvas) without hand-exporting
# env-soup. This wrapper resolves flags, refuses illegal combos, then execs
# build-cuda/h3_cuda_generate (override with H3_GENERATE).
#
# Evidence / policy:
#   NFE ladder @7cc2a0f — docs/research/2026-10-28-nfe-fewstep-contracts.md
#   Raise-canvas @cd3daf2 — docs/research/2026-10-28-raise-canvas-sku.md
#   t71u: primary = DMAD or Euler; PDMD = contrast-only
#   Do NOT sell: 1280×720, 1344+fewstep, FaceRefine, pdmd2, NVFP4, C035
#   Product SDPA default stays cudnn (S2 is opt-in via --s2).
#
# One-liners:
#   tools/h3_product_generate.sh --sku rc0 --prompt-ids evals/768p/ids/p10_street_wave_dolly.ids
#   tools/h3_product_generate.sh --sku rc1 --seconds 5 --out out/rc1
#   tools/h3_product_generate.sh --sku rc2 --seconds 5 --out out/rc2
#   tools/h3_product_generate.sh --nfe dmad4 --canvas 1152 --dry-run
#   tools/h3_product_generate.sh --sku rc0 --nfe euler8   # quality gold @480p
#
# shellcheck disable=SC1090,SC1091,SC2034,SC2086
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

SKU=""
NFE=""
CANVAS=""
SECONDS_N=5
SEED="0x76800002"
OUT=""
MUX=""
PROMPT_IDS=""
PROMPT_TEXT=""
DRY=0
S2=0
WINDOW24=0
SKIP_AUDIO=1
REPEAT=1
SDPA=""
EXTRA=()
SHOW_HELP=0
SKU_ID=""
WIDTH=0
HEIGHT=0

usage() {
  cat <<'USAGE'
tools/h3_product_generate.sh — product entry (NFE × raise-canvas)

SKU aliases (preferred):
  --sku rc0          864×480  + DMAD-4   (default speed @480p)
  --sku rc0_euler    864×480  + Euler-8  (quality / waffle gold)
  --sku rc0_dmad     alias of rc0
  --sku rc1          1152×640 + DMAD-4   (speed-face)
  --sku rc1c         1152×640 + PDMD-4   (contrast-only; not primary)
  --sku rc2          1344×768 + Euler-8  (max face)
  --sku rc960        960×544  + DMAD-4   (optional intermediate)

Or compose:
  --nfe euler8|dmad4|pdmd4
  --canvas 480|864x480|1152|1152x640|1344|1344x768|960|960x544

Common:
  --seconds N        default 5
  --seed N           default 0x76800002
  --prompt-ids FILE  --token-ids-file
  --prompt TEXT      free-text prompt (ids win if both set)
  --out DIR          --out
  --mux FILE.mp4     --mux (default: $OUT/<sku>.mp4 when --out set)
  --repeat N         in-process warm loop (default 1)
  --sdpa-backend B   default cudnn; with --s2 defaults to sol_cute
  --s2               source Track-B S2 env (WINDOW=24 + sol LAYERS=2)
  --window24         export H3_DIT_INT8_WINDOW=24 only (cudnn path)
  --with-audio       do not pass --skip-audio
  --dry-run          print resolved argv + env; do not exec
  -h|--help

REFUSED (exit 2):
  1280×720 / any W or H not %32==0
  1344 + dmad4|pdmd4 (4090 INT8 fewstep noise)
  pdmd2 / FaceRefine / inventing missing LoRA

Pass-through: args after -- go to h3_cuda_generate unchanged.

Docs:
  docs/research/2026-10-28-nfe-fewstep-contracts.md
  docs/research/2026-10-28-raise-canvas-sku.md
USAGE
}

die() { echo "h3_product_generate: $*" >&2; exit 2; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sku) SKU="${2:-}"; shift 2 ;;
    --nfe) NFE="${2:-}"; shift 2 ;;
    --canvas) CANVAS="${2:-}"; shift 2 ;;
    --seconds) SECONDS_N="${2:-}"; shift 2 ;;
    --seed) SEED="${2:-}"; shift 2 ;;
    --prompt-ids|--token-ids-file) PROMPT_IDS="${2:-}"; shift 2 ;;
    --prompt) PROMPT_TEXT="${2:-}"; shift 2 ;;
    --out) OUT="${2:-}"; shift 2 ;;
    --mux) MUX="${2:-}"; shift 2 ;;
    --repeat) REPEAT="${2:-}"; shift 2 ;;
    --sdpa-backend) SDPA="${2:-}"; shift 2 ;;
    --s2) S2=1; shift ;;
    --window24) WINDOW24=1; shift ;;
    --with-audio) SKIP_AUDIO=0; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) SHOW_HELP=1; shift ;;
    --) shift; EXTRA+=("$@"); break ;;
    *)
      die "unknown arg: $1 (use --help; pass-through after --)"
      ;;
  esac
done

if [[ "$SHOW_HELP" == 1 ]]; then
  usage
  exit 0
fi

normalize_sku() {
  local s
  s="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "$s" in
    rc0|rc0_dmad|480p_default|480p_dmad) printf '%s\n' "rc0" ;;
    rc0_euler|480p_euler|rc0_quality) printf '%s\n' "rc0_euler" ;;
    rc1|1152_dmad|rc1_dmad|1152_speed_face) printf '%s\n' "rc1" ;;
    rc1c|1152_pdmd|rc1_pdmd) printf '%s\n' "rc1c" ;;
    rc2|1344_euler|rc2_euler|1344_max_face) printf '%s\n' "rc2" ;;
    rc960|960|960_optional) printf '%s\n' "rc960" ;;
    *) die "unknown --sku '$1' (want rc0|rc0_euler|rc1|rc1c|rc2|rc960)" ;;
  esac
}

apply_sku() {
  local id="$1"
  case "$id" in
    rc0)
      NFE="${NFE:-dmad4}"
      CANVAS="${CANVAS:-480}"
      ;;
    rc0_euler)
      NFE="${NFE:-euler8}"
      CANVAS="${CANVAS:-480}"
      ;;
    rc1)
      NFE="${NFE:-dmad4}"
      CANVAS="${CANVAS:-1152}"
      ;;
    rc1c)
      NFE="${NFE:-pdmd4}"
      CANVAS="${CANVAS:-1152}"
      ;;
    rc2)
      NFE="${NFE:-euler8}"
      CANVAS="${CANVAS:-1344}"
      ;;
    rc960)
      NFE="${NFE:-dmad4}"
      CANVAS="${CANVAS:-960}"
      ;;
  esac
  SKU_ID="$id"
}

if [[ -n "$SKU" ]]; then
  apply_sku "$(normalize_sku "$SKU")"
fi

if [[ -z "$NFE" && -z "$CANVAS" ]]; then
  apply_sku rc0
fi
NFE="${NFE:-dmad4}"
CANVAS="${CANVAS:-480}"
SKU_ID="${SKU_ID:-custom}"

normalize_nfe() {
  local a
  a="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "$a" in
    euler8|euler-8|euler|e8) printf '%s\n' "euler8" ;;
    dmad4|dmad-4|dmad) printf '%s\n' "dmad4" ;;
    pdmd4|pdmd-4|pdmd) printf '%s\n' "pdmd4" ;;
    pdmd2|pdmd-2)
      die "REFUSED: pdmd2 not sellable (no 2NFE LoRA on 4090). See nfe-fewstep-contracts.md"
      ;;
    *) die "unknown --nfe '$1' (want euler8|dmad4|pdmd4)" ;;
  esac
}
NFE="$(normalize_nfe "$NFE")"

normalize_canvas() {
  local c w h
  c="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "$c" in
    480|480p|864x480|rc0) w=864; h=480 ;;
    960|960x544|rc960) w=960; h=544 ;;
    1152|1152x640|rc1) w=1152; h=640 ;;
    1344|1344x768|rc2|768p) w=1344; h=768 ;;
    1280|1280x720|720p|720)
      die "REFUSED: 1280×720 (W/H must %32==0). Use 1152×640 or 1344×768."
      ;;
    *)
      if [[ "$c" =~ ^([0-9]+)x([0-9]+)$ ]]; then
        w="${BASH_REMATCH[1]}"; h="${BASH_REMATCH[2]}"
      else
        die "unknown --canvas '$1' (want 480|1152|1344|960|WxH)"
      fi
      ;;
  esac
  if (( w % 32 != 0 || h % 32 != 0 )); then
    die "REFUSED geometry ${w}×${h}: W and H must each be multiples of 32"
  fi
  WIDTH="$w"
  HEIGHT="$h"
}
normalize_canvas "$CANVAS"

if [[ "$WIDTH" -eq 1344 && "$HEIGHT" -eq 768 ]]; then
  if [[ "$NFE" == "dmad4" || "$NFE" == "pdmd4" ]]; then
    die "REFUSED: 1344×768 + ${NFE} on 4090 INT8 (fewstep → window≈6 FAIL noise). Use --sku rc2 (Euler-8) or drop to --sku rc1 (1152 DMAD)."
  fi
fi

MODELS="${H3_MODELS:-/home/haiyang/workspace/models}"
GEN="${H3_GENERATE:-$ROOT/build-cuda/h3_cuda_generate}"
DIT="${H3_DIT:-$MODELS/minimax_h3_fl2va_pruned_int8_convrot.safetensors}"
VAE="${H3_VAE:-$MODELS/minimax_h3_video_vae_fp16.safetensors}"
TE="${H3_TE:-$MODELS/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors}"
DMAD_LORA="${H3_DMAD_LORA:-$MODELS/loras/dmad_minimax_h3_4step_full_critic.h3repo.safetensors}"
PDMD_LORA="${H3_PDMD_LORA:-$MODELS/loras/pdmd_4NFE_lora_model_0.h3repo.safetensors}"

export H3_SKIP_PPM="${H3_SKIP_PPM:-1}"

if [[ "$WINDOW24" == 1 ]]; then
  export H3_DIT_INT8_WINDOW=24
fi
if [[ "$S2" == 1 ]]; then
  # shellcheck source=/dev/null
  source "$ROOT/tools/h3_track_b_s2_env.sh"
  SDPA="${SDPA:-sol_cute}"
fi
SDPA="${SDPA:-cudnn}"

ARGV=(
  "$GEN"
  --dit "$DIT"
  --vae "$VAE"
  --te "$TE"
  --width "$WIDTH"
  --height "$HEIGHT"
  --seconds "$SECONDS_N"
  --seed "$SEED"
  --cfg 1
  --sdpa-backend "$SDPA"
  --repeat "$REPEAT"
)

case "$NFE" in
  euler8)
    ARGV+=(--steps 8)
    ;;
  dmad4)
    if [[ "$DRY" != 1 ]]; then
      [[ -s "$DMAD_LORA" ]] || die "missing DMAD LoRA: $DMAD_LORA"
    fi
    ARGV+=(--fewstep-preset dmad4 --lora "$DMAD_LORA" --lora-scale 1)
    ;;
  pdmd4)
    if [[ "$DRY" != 1 ]]; then
      [[ -s "$PDMD_LORA" ]] || die "missing PDMD LoRA: $PDMD_LORA"
    fi
    ARGV+=(--fewstep-preset pdmd4 --lora "$PDMD_LORA" --lora-scale 1)
    ;;
esac

if [[ "$SKIP_AUDIO" == 1 ]]; then
  ARGV+=(--skip-audio)
fi

if [[ -n "$PROMPT_IDS" ]]; then
  if [[ "$DRY" != 1 ]]; then
    [[ -s "$PROMPT_IDS" ]] || die "missing --prompt-ids $PROMPT_IDS"
  fi
  ARGV+=(--token-ids-file "$PROMPT_IDS")
elif [[ -n "$PROMPT_TEXT" ]]; then
  ARGV+=(--prompt "$PROMPT_TEXT")
fi

if [[ -n "$OUT" ]]; then
  if [[ "$DRY" != 1 ]]; then
    mkdir -p "$OUT"
  fi
  ARGV+=(--out "$OUT")
  if [[ -z "$MUX" ]]; then
    MUX="$OUT/${SKU_ID}_${NFE}_${WIDTH}x${HEIGHT}.mp4"
  fi
fi
if [[ -n "$MUX" ]]; then
  ARGV+=(--mux "$MUX")
fi

if [[ ${#EXTRA[@]} -gt 0 ]]; then
  ARGV+=("${EXTRA[@]}")
fi

export H3_PRODUCT_SKU="$SKU_ID"
export H3_PRODUCT_NFE="$NFE"
export H3_PRODUCT_WIDTH="$WIDTH"
export H3_PRODUCT_HEIGHT="$HEIGHT"
export H3_PRODUCT_SDPA="$SDPA"
export H3_WIDTH="$WIDTH"
export H3_HEIGHT="$HEIGHT"
if [[ "$NFE" == "dmad4" || "$NFE" == "pdmd4" ]]; then
  export H3_FEWSTEP_PRESET="$NFE"
else
  unset H3_FEWSTEP_PRESET H3_FEWSTEP_CONTRACT 2>/dev/null || true
fi

cat <<EOF >&2
h3_product_generate: sku=$SKU_ID nfe=$NFE canvas=${WIDTH}x${HEIGHT} sdpa=$SDPA seconds=$SECONDS_N
  C015=defaultON window24=${H3_DIT_INT8_WINDOW:-unset} s2=$S2 dry=$DRY
  gen=$GEN
EOF

if [[ "$DRY" == 1 ]]; then
  printf 'DRY_ARGV:'
  printf ' %q' "${ARGV[@]}"
  printf '\n'
  exit 0
fi

[[ -x "$GEN" || -f "$GEN" ]] || die "missing generate binary: $GEN (set H3_GENERATE)"
for f in "$DIT" "$VAE" "$TE"; do
  [[ -s "$f" ]] || die "missing weight: $f"
done

exec "${ARGV[@]}"
