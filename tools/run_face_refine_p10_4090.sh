#!/usr/bin/env bash
# FaceRefine A/B on p10_street_wave_dolly (distant-face mush).
# crop → H3 F2VA face-CU (PDMD-4) → stitch; plus classical enhance baseline.
set -euo pipefail
ROOT="${H3_ROOT:-/home/haiyang/workspace/h3_c_main}"
MODELS="${H3_MODELS:-/home/haiyang/workspace/models}"
BIN="$ROOT/build-cuda/h3_cuda_generate"
PY="${H3_PY:-/home/haiyang/miniconda3/envs/comfyui/bin/python}"
YUNET="${H3_YUNET:-$MODELS/face_detect/face_detection_yunet_2023mar.onnx}"
BURN="$ROOT/tools/burn_eval_label.sh"
TOOL="$ROOT/tools/face_refine_crop_h3_stitch.py"
OUT="$ROOT/out/facerefine_p10_20261027"
SRC_UNLABELED="$ROOT/out/complex_480p_20261027/p10_street_wave_dolly_pdmd4/p10_street_wave_dolly_pdmd4.mp4"
LORA="$MODELS/loras/pdmd_4NFE_lora_model_0.h3repo.safetensors"
IDS="$ROOT/evals/768p/ids/p10_face_cu_facerefine.ids"
REFINE_SIZE=544
SEED=0x76800002

export H3_SKIP_PPM=1
unset H3_NVFP4 H3_EPI_FUSE H3_EPI_FUSE_INK H3_EPI_FUSE_SL 2>/dev/null || true

mkdir -p "$OUT/work" "$OUT/labeled"
echo "SRC=$SRC_UNLABELED"
test -f "$SRC_UNLABELED"
test -f "$YUNET"
test -f "$BIN"
test -f "$LORA"
test -f "$IDS"

# --- detect ---
/usr/bin/time -f 'WALL_SEC=%e' -o "$OUT/work/detect_wall.txt" \
  "$PY" "$TOOL" detect \
    --video "$SRC_UNLABELED" \
    --yunet "$YUNET" \
    --out-dir "$OUT/work" \
    --refine-size "$REFINE_SIZE" \
    --pad 0.65 | tee "$OUT/work/detect.log"
DETECT_WALL=$(grep -oE '[0-9.]+' "$OUT/work/detect_wall.txt" | tail -1)

# --- classical enhance baseline ---
/usr/bin/time -f 'WALL_SEC=%e' -o "$OUT/work/enhance_wall.txt" \
  "$PY" "$TOOL" enhance \
    --video "$SRC_UNLABELED" \
    --track "$OUT/work/track.json" \
    --out "$OUT/p10_FaceEnhance_classical.mp4" | tee "$OUT/work/enhance.log"
ENHANCE_WALL=$(grep -oE '[0-9.]+' "$OUT/work/enhance_wall.txt" | tail -1)

# --- H3 F2VA face-CU regen (PDMD-4) ---
H3_OUT="$OUT/h3_face_cu_pdmd4"
mkdir -p "$H3_OUT"
FIRST="$OUT/work/first_frame_face.jpg"
echo "H3 F2VA first-frame=$FIRST size=${REFINE_SIZE}x${REFINE_SIZE}"
set +e
/usr/bin/time -f 'WALL_SEC=%e' -o "$OUT/work/h3_wall.txt" \
  "$BIN" \
    --dit "$MODELS/minimax_h3_fl2va_pruned_int8_convrot.safetensors" \
    --vae "$MODELS/minimax_h3_video_vae_fp16.safetensors" \
    --te "$MODELS/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" \
    --width "$REFINE_SIZE" --height "$REFINE_SIZE" --seconds 5 \
    --fewstep-preset pdmd4 --lora "$LORA" --lora-scale 1 \
    --seed "$SEED" --sdpa-backend cudnn --cfg 1 \
    --first-frame "$FIRST" \
    --token-ids-file "$IDS" \
    --out "$H3_OUT" \
    --mux "$H3_OUT/face_cu_pdmd4.mp4" \
    2>&1 | tee "$OUT/work/h3_generate.log"
H3_RC=${PIPESTATUS[0]}
set -e
H3_WALL=$(grep -oE '[0-9.]+' "$OUT/work/h3_wall.txt" | tail -1 || echo NA)
DIT_MS=$(grep -oE 'dit_ms[=:][ ]*[0-9.]+' "$OUT/work/h3_generate.log" | tail -1 | grep -oE '[0-9.]+$' || \
  grep -oE 'dit[_ ]*ms[=: ]+[0-9.]+' "$OUT/work/h3_generate.log" | tail -1 | grep -oE '[0-9.]+$' || echo NA)
# fallback parse from common log lines
if [[ "$DIT_MS" == "NA" ]]; then
  DIT_MS=$(grep -oE 'warm_dit_ms[= ]*[0-9.]+|dit_ms[= ]*[0-9.]+|DiT .*[0-9]+\.[0-9]+ ms' "$OUT/work/h3_generate.log" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | tail -1 || echo NA)
fi
echo "H3_RC=$H3_RC H3_WALL=$H3_WALL DIT_MS=$DIT_MS"

if [[ $H3_RC -ne 0 || ! -f "$H3_OUT/face_cu_pdmd4.mp4" ]]; then
  # try alternate mux name
  ALT=$(ls "$H3_OUT"/*.mp4 2>/dev/null | head -1 || true)
  if [[ -n "${ALT:-}" ]]; then
    cp -f "$ALT" "$H3_OUT/face_cu_pdmd4.mp4"
    echo "recovered mux $ALT"
  else
    echo "H3 FaceRefine generate FAILED rc=$H3_RC" >&2
    # still label before + classical
  fi
fi

# --- stitch H3 refined face CU back ---
if [[ -f "$H3_OUT/face_cu_pdmd4.mp4" ]]; then
  /usr/bin/time -f 'WALL_SEC=%e' -o "$OUT/work/stitch_wall.txt" \
    "$PY" "$TOOL" stitch \
      --video "$SRC_UNLABELED" \
      --refined "$H3_OUT/face_cu_pdmd4.mp4" \
      --track "$OUT/work/track.json" \
      --out "$OUT/p10_FaceRefine_H3_PDMD4.mp4" | tee "$OUT/work/stitch.log"
  STITCH_WALL=$(grep -oE '[0-9.]+' "$OUT/work/stitch_wall.txt" | tail -1)
else
  STITCH_WALL=NA
fi

# --- labels ---
# before = original PDMD unlabeled, burn as BEFORE
"$BURN" "$SRC_UNLABELED" "$OUT/labeled/p10_BEFORE_PDMD-4_INT8_labeled.mp4" \
  "864x480 PDMD-4 INT8 | before FaceRefine"
# classical
if [[ -f "$OUT/p10_FaceEnhance_classical.mp4" ]]; then
  "$BURN" "$OUT/p10_FaceEnhance_classical.mp4" \
    "$OUT/labeled/p10_AFTER_FaceEnhance_classical_labeled.mp4" \
    "864x480 FaceEnhance classical | warm ${ENHANCE_WALL}s"
fi
# H3 FaceRefine
if [[ -f "$OUT/p10_FaceRefine_H3_PDMD4.mp4" ]]; then
  "$BURN" "$OUT/p10_FaceRefine_H3_PDMD4.mp4" \
    "$OUT/labeled/p10_AFTER_FaceRefine_H3_PDMD4_labeled.mp4" \
    "864x480 FaceRefine H3-PDMD4 | warm ${H3_WALL}s dit ${DIT_MS}ms"
fi
# also label the face CU alone for evidence
if [[ -f "$H3_OUT/face_cu_pdmd4.mp4" ]]; then
  "$BURN" "$H3_OUT/face_cu_pdmd4.mp4" \
    "$OUT/labeled/p10_face_cu_544_PDMD4_F2VA_labeled.mp4" \
    "544x544 F2VA-PDMD4 face-CU | warm ${H3_WALL}s"
fi

# META
cat > "$OUT/labeled/META.txt" << EOF
date_local: $(date '+%Y-%m-%d %H:%M') Asia/Singapore (UTC+8)
host: lan-4090-31 / RTX 4090
scene: p10_street_wave_dolly (distant-face mush / head-size pixel budget)
base_arm: PDMD-4 INT8 864x480 seed=$SEED
FaceRefine: crop(YuNet) → H3 F2VA face-CU ${REFINE_SIZE}x${REFINE_SIZE} PDMD-4 → feather stitch
also: classical Lanczos+unsharp enhance (no generative)

detect_WALL_s: $DETECT_WALL
enhance_WALL_s: $ENHANCE_WALL
h3_face_cu_WALL_s: $H3_WALL
h3_face_cu_dit_ms: $DIT_MS
h3_rc: $H3_RC
stitch_WALL_s: $STITCH_WALL

labeled:
  p10_BEFORE_PDMD-4_INT8_labeled.mp4
  p10_AFTER_FaceEnhance_classical_labeled.mp4
  p10_AFTER_FaceRefine_H3_PDMD4_labeled.mp4
  p10_face_cu_544_PDMD4_F2VA_labeled.mp4

work: $OUT/work (track.json, first_frame_face.jpg, crop_*.png)
4090_out: $OUT
gaps: no CodeFormer/GFPGAN weights this slice; H3 path is F2VA regen not pixel-locked img2img; single-subject YuNet track.
EOF

echo "DONE out=$OUT/labeled"
ls -la "$OUT/labeled"
