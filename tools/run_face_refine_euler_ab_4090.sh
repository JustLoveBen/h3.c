#!/usr/bin/env bash
# FaceRefine steps A/B: reuse p10 detect/track + BEFORE base; Euler-8 (+ optional Euler-16)
# F2VA face-CU → stitch onto same PDMD-4 unlabeled base. Not pixel-locked img2img.
set -euo pipefail
ROOT="${H3_ROOT:-/home/haiyang/workspace/h3_c_main}"
MODELS="${H3_MODELS:-/home/haiyang/workspace/models}"
BIN="$ROOT/build-cuda/h3_cuda_generate"
PY="${H3_PY:-/home/haiyang/miniconda3/envs/comfyui/bin/python}"
BURN="$ROOT/tools/burn_eval_label.sh"
TOOL="$ROOT/tools/face_refine_crop_h3_stitch.py"
OUT="$ROOT/out/facerefine_p10_20261027"
SRC_UNLABELED="$ROOT/out/complex_480p_20261027/p10_street_wave_dolly_pdmd4/p10_street_wave_dolly_pdmd4.mp4"
IDS="$ROOT/evals/768p/ids/p10_face_cu_facerefine.ids"
REFINE_SIZE=544
SEED=0x76800002
# ARMS: space-separated step counts (gold Euler, no fewstep LoRA)
ARMS="${FACE_REFINE_EULER_ARMS:-8 16}"

export H3_SKIP_PPM=1
unset H3_NVFP4 H3_EPI_FUSE H3_EPI_FUSE_INK H3_EPI_FUSE_SL 2>/dev/null || true

test -f "$SRC_UNLABELED"
test -f "$BIN"
test -f "$IDS"
test -f "$OUT/work/track.json"
test -f "$OUT/work/first_frame_face.jpg"
FIRST="$OUT/work/first_frame_face.jpg"
mkdir -p "$OUT/labeled" "$OUT/work"

run_arm() {
  local STEPS="$1"
  local TAG="euler${STEPS}"
  local H3_OUT="$OUT/h3_face_cu_${TAG}"
  local MP4="$H3_OUT/face_cu_${TAG}.mp4"
  local STITCHED="$OUT/p10_FaceRefine_H3_Euler${STEPS}.mp4"
  mkdir -p "$H3_OUT"
  echo "==== FaceRefine F2VA Euler-${STEPS} $(date '+%H:%M:%S %Z') ===="
  set +e
  /usr/bin/time -f 'WALL_SEC=%e' -o "$OUT/work/h3_${TAG}_wall.txt" \
    "$BIN" \
      --dit "$MODELS/minimax_h3_fl2va_pruned_int8_convrot.safetensors" \
      --vae "$MODELS/minimax_h3_video_vae_fp16.safetensors" \
      --te "$MODELS/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" \
      --width "$REFINE_SIZE" --height "$REFINE_SIZE" --seconds 5 \
      --steps "$STEPS" --cfg 1 --sampler velocity_euler \
      --seed "$SEED" --sdpa-backend cudnn \
      --first-frame "$FIRST" \
      --token-ids-file "$IDS" \
      --out "$H3_OUT" \
      --mux "$MP4" \
      2>&1 | tee "$OUT/work/h3_${TAG}_generate.log"
  local RC=${PIPESTATUS[0]}
  set -e
  local WALL DIT_MS
  WALL=$(grep -oE '[0-9.]+' "$OUT/work/h3_${TAG}_wall.txt" | tail -1 || echo NA)
  DIT_MS=$(grep -oE 'dit_ms[=:][ ]*[0-9.]+' "$OUT/work/h3_${TAG}_generate.log" | tail -1 | grep -oE '[0-9.]+$' || true)
  if [[ -z "${DIT_MS:-}" ]]; then
    DIT_MS=$(grep -oE 'dit_schedule:.*dit_ms=[0-9.]+' "$OUT/work/h3_${TAG}_generate.log" | tail -1 | grep -oE '[0-9.]+$' || echo NA)
  fi
  echo "Euler${STEPS}_RC=$RC WALL=$WALL DIT_MS=$DIT_MS"
  if [[ $RC -ne 0 || ! -f "$MP4" ]]; then
    local ALT
    ALT=$(ls "$H3_OUT"/*.mp4 2>/dev/null | head -1 || true)
    if [[ -n "${ALT:-}" ]]; then
      cp -f "$ALT" "$MP4"
      echo "recovered mux $ALT"
    else
      echo "FAIL Euler-${STEPS} generate rc=$RC" >&2
      return "$RC"
    fi
  fi
  /usr/bin/time -f 'WALL_SEC=%e' -o "$OUT/work/stitch_${TAG}_wall.txt" \
    "$PY" "$TOOL" stitch \
      --video "$SRC_UNLABELED" \
      --refined "$MP4" \
      --track "$OUT/work/track.json" \
      --out "$STITCHED" | tee "$OUT/work/stitch_${TAG}.log"
  local SW
  SW=$(grep -oE '[0-9.]+' "$OUT/work/stitch_${TAG}_wall.txt" | tail -1 || echo NA)
  "$BURN" "$STITCHED" \
    "$OUT/labeled/p10_AFTER_FaceRefine_H3_Euler${STEPS}_labeled.mp4" \
    "864x480 FaceRefine H3-Euler${STEPS} | warm ${WALL}s dit ${DIT_MS}ms"
  "$BURN" "$MP4" \
    "$OUT/labeled/p10_face_cu_544_Euler${STEPS}_F2VA_labeled.mp4" \
    "544x544 F2VA-Euler${STEPS} face-CU | warm ${WALL}s"
  # evidence frames at early/late
  STEPS="$STEPS" STITCHED="$STITCHED" OUT="$OUT" "$PY" - <<'PY'
import cv2, os
from pathlib import Path
steps = os.environ["STEPS"]
out = Path(os.environ["OUT"]) / "labeled" / "frames"
out.mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(os.environ["STITCHED"])
fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
for t in (0.5, 1.5, 3.0, 4.5):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
    ok, fr = cap.read()
    if ok:
        cv2.imwrite(str(out / f"after_euler{steps}_t{t}.png"), fr)
cap.release()
print(f"frames ok Euler{steps}")
PY
  echo "Euler${STEPS}_WALL=$WALL" > "$OUT/work/euler${STEPS}_summary.txt"
  echo "Euler${STEPS}_DIT_MS=$DIT_MS" >> "$OUT/work/euler${STEPS}_summary.txt"
  echo "Euler${STEPS}_STITCH=$SW" >> "$OUT/work/euler${STEPS}_summary.txt"
  echo "Euler${STEPS}_RC=$RC" >> "$OUT/work/euler${STEPS}_summary.txt"
}

for S in $ARMS; do
  run_arm "$S"
done

# pull prior PDMD4 timings if present
PDMD_WALL=$(grep -oE '[0-9.]+' "$OUT/work/h3_wall.txt" 2>/dev/null | tail -1 || echo NA)
PDMD_DIT=$(grep -oE 'dit_ms[=:][ ]*[0-9.]+' "$OUT/work/h3_generate.log" 2>/dev/null | tail -1 | grep -oE '[0-9.]+$' || echo NA)
E8_WALL=$(grep Euler8_WALL "$OUT/work/euler8_summary.txt" 2>/dev/null | cut -d= -f2 || echo NA)
E8_DIT=$(grep Euler8_DIT "$OUT/work/euler8_summary.txt" 2>/dev/null | cut -d= -f2 || echo NA)
E16_WALL=$(grep Euler16_WALL "$OUT/work/euler16_summary.txt" 2>/dev/null | cut -d= -f2 || echo NA)
E16_DIT=$(grep Euler16_DIT "$OUT/work/euler16_summary.txt" 2>/dev/null | cut -d= -f2 || echo NA)

cat > "$OUT/labeled/META_steps_ab.txt" << META
date_local: $(date '+%Y-%m-%d %H:%M') Asia/Singapore (UTC+8)
host: lan-4090-31 / RTX 4090
scene: p10_street_wave_dolly — FaceRefine steps A/B on SAME BEFORE base
base: PDMD-4 INT8 864x480 unlabeled → stitch target (seed=$SEED)
FaceRefine path: YuNet track (reused) → H3 official F2VA face-CU ${REFINE_SIZE}x${REFINE_SIZE} → feather stitch
NOT Comfy-style pixel-locked img2img denoise (0.35–0.45 × N steps).

arms_face_cu:
  PDMD-4 F2VA (prior): warm ${PDMD_WALL}s dit_ms=${PDMD_DIT}
  Euler-8 F2VA (gold --steps 8, no fewstep LoRA): warm ${E8_WALL}s dit_ms=${E8_DIT}
  Euler-16 F2VA (--steps 16): warm ${E16_WALL}s dit_ms=${E16_DIT}

labeled:
  p10_BEFORE_PDMD-4_INT8_labeled.mp4
  p10_AFTER_FaceRefine_H3_PDMD4_labeled.mp4
  p10_AFTER_FaceRefine_H3_Euler8_labeled.mp4
  p10_AFTER_FaceRefine_H3_Euler16_labeled.mp4   # if arm ran
  p10_face_cu_544_*_F2VA_labeled.mp4

architecture_gap:
  Official --first-frame is F2VA COND (VAE encode → DiT condition rows) then full
  video regen from noise; it does NOT denoise the per-frame face crop latent.
  --init-latent + --init-first-frame is G3-lite debug (non-official), not a
  product FaceDetailer/img2img path. True crop-locked denoise needs a new
  pipeline (encode each crop frame → partial denoise → decode → stitch) that
  h3_cuda_generate does not expose as an official F2VA mode today.

hypothesis:
  If Euler8/16 ≫ PDMD4 on face detail → NFE was a real bottleneck.
  If Euler8/16 ≈ PDMD4 (still soft identity / stitch mush at wide) → architecture
  (F2VA regen vs pixel-locked img2img) dominates over step count.
META

# ensure BEFORE label still present
if [[ ! -f "$OUT/labeled/p10_BEFORE_PDMD-4_INT8_labeled.mp4" ]]; then
  "$BURN" "$SRC_UNLABELED" "$OUT/labeled/p10_BEFORE_PDMD-4_INT8_labeled.mp4" \
    "864x480 PDMD-4 INT8 | before FaceRefine"
fi

echo "DONE steps A/B → $OUT/labeled"
ls -la "$OUT/labeled"
