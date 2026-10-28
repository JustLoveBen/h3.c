#!/usr/bin/env bash
# Burn a TOP-LEFT scheme label into an eval clip so A/B arms stay identifiable.
# Usage: tools/burn_eval_label.sh <in.mp4> <out.mp4> <label>
# Example label: "RC0 864x480 DMAD-4 | 1s hot 12.3s (dit 5.0s)"
# Do NOT burn process WALL_SEC from cold TE/VAE load as "warm". Prefer hot=1 phase_ms/1000.
# 1s smoke ≠ 5s product KPI — always include seconds in the label text.
set -euo pipefail
if [[ $# -lt 3 ]]; then
  echo "usage: $0 <in.mp4> <out.mp4> <label>" >&2
  exit 2
fi
IN=$1; OUT=$2; LABEL=$3
FONT="${H3_EVAL_LABEL_FONT:-/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf}"
if [[ ! -f "$FONT" ]]; then
  FONT="/System/Library/Fonts/Supplemental/Arial Bold.ttf"
fi
if [[ ! -f "$FONT" ]]; then
  FONT="/Library/Fonts/Arial Bold.ttf"
fi
# Adaptive fontsize so long KPI strings fit 480p (864w) without clipping.
W=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 "$IN" 2>/dev/null | head -1)
W=${W:-864}
# ~0.55*fontsize px per char (bold); leave 24px margin each side
FS=$(awk -v w="$W" -v n="${#LABEL}" 'BEGIN{
  fit=int((w-24)/(n*0.55+1));
  if(fit>28) fit=28;
  if(fit<14) fit=14;
  print fit
}')
mkdir -p "$(dirname "$OUT")"
# Escape drawtext specials: : \ '
ESC=${LABEL//\\/\\\\}
ESC=${ESC//:/\\:}
ESC=${ESC//\'/\\\'}
if ffmpeg -y -hide_banner -loglevel error -i "$IN" \
  -vf "drawtext=fontfile=${FONT}:text='${ESC}':x=12:y=12:fontsize=${FS}:fontcolor=white:borderw=3:bordercolor=black@0.8" \
  -c:v libx264 -preset veryfast -crf 14 -pix_fmt yuv420p -c:a copy -movflags +faststart \
  "$OUT"; then
  echo "burned label='$LABEL' fontsize=$FS → $OUT"
else
  echo "drawtext failed; writing unlabeled copy" >&2
  cp -f "$IN" "$OUT"
  exit 1
fi
