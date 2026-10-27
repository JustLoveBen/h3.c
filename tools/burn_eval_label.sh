#!/usr/bin/env bash
# Burn a TOP-LEFT scheme label into an eval clip so A/B arms stay identifiable.
# Usage: tools/burn_eval_label.sh <in.mp4> <out.mp4> <label>
# Example label: "864x480 PDMD-4 INT8 | warm 68.0s"  (scheme + warm end-to-end WALL_SEC, not dit-only)
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
# Prefer drawtext; fall back to plain copy with warning if filter missing.
mkdir -p "$(dirname "$OUT")"
if ffmpeg -y -hide_banner -loglevel error -i "$IN" \
  -vf "drawtext=fontfile=${FONT}:text='${LABEL//:/\\:}':x=12:y=12:fontsize=28:fontcolor=white:borderw=3:bordercolor=black@0.8" \
  -c:v libx264 -preset veryfast -crf 14 -pix_fmt yuv420p -c:a copy -movflags +faststart \
  "$OUT"; then
  echo "burned label='$LABEL' → $OUT"
else
  echo "drawtext failed; writing unlabeled copy" >&2
  cp -f "$IN" "$OUT"
  exit 1
fi
