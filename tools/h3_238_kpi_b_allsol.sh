#!/usr/bin/env bash
# tools/h3_238_kpi_b_allsol.sh — KPI-B 2-stage all-sol on the 238 RTX 5090.
#
# Runs the HIRES stage through the official cute_sm120 sol kernel with the
# use-and-release hires QKVO path (H3_SOL_HIRES_QKVO=1) so the 32GB card can
# fit INT8 window + sol QKVO + adaln/GEMM activations. Compares against the
# mixed arm (low-res sol + hires cudnn) and the product cudnn arm.
#
# Constraints honored:
#   - INT8 window NOT reduced below 24 (default 34; no shrink).
#   - No overlap, no fake sage3, no TAE/LTX, no GEMM tile change.
#   - sage3 want>0 && hits==0 hard-fails (product guard).
#   - Official H3 VAE decode; taomate-lift 2-stage H3->H3 contract.
#
# Env required on 238:
#   H3_MODELS            model root (default /mnt/nvme1n1/workspace/models)
#   H3_SOL_ATTN_ROOT     official sparse_backends tree
#   H3_SOL_CUTE_LIB      libh3_sol_cute.so
#   H3_TAOMATE_LORA      TaoMate adapter safetensors
#   H3_LATENT_UPSCALER   LBH upscaler safetensors
#
# Usage:
#   bash tools/h3_238_kpi_b_allsol.sh [repeat] [out_tag]
#     repeat   in-process clips after weight load (hot = clip>=2). default 2
#     out_tag  subdir under out/review-2026-10-01/. default kpi-b-allsol-qkvo
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPEAT="${1:-2}"
TAG="${2:-kpi-b-allsol-qkvo}"

H3_MODELS="${H3_MODELS:-/mnt/nvme1n1/workspace/models}"
export H3_MODELS

# --- official sol cute plugin (fail loud if missing) ---
: "${H3_SOL_ATTN_ROOT:?set H3_SOL_ATTN_ROOT to official techniques/sparse_backends}"
: "${H3_SOL_CUTE_LIB:?set H3_SOL_CUTE_LIB to libh3_sol_cute.so}"
export H3_SOL_ATTN_ROOT H3_SOL_CUTE_LIB
export H3_SAGE3_TOOLS="${H3_SAGE3_TOOLS:-$ROOT/tools}"

# conda python3.11 sees torch/cutlass; distro python3.10 does not.
_PY311_SITE="${H3_PY311_SITE:-}"
if [ -z "$_PY311_SITE" ]; then
  for c in /opt/conda/lib/python3.11/site-packages \
           "$HOME/miniconda3/lib/python3.11/site-packages" \
           "$HOME/anaconda3/lib/python3.11/site-packages"; do
    [ -d "$c/torch" ] && _PY311_SITE="$c" && break
  done
fi
if [ -n "$_PY311_SITE" ]; then
  export PYTHONPATH="$_PY311_SITE:$H3_SAGE3_TOOLS${PYTHONPATH:+:$PYTHONPATH}"
  [ -d "$_PY311_SITE/torch/lib" ] && \
    export LD_LIBRARY_PATH="$_PY311_SITE/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# --- all-sol: hires also goes through sol (no max_seq skip) ---
export H3_SOL_MAX_SEQ=0
export H3_SOL_FIRST_DENSE_STEPS=0
export H3_SOL_FIRST_DENSE_LAYERS=0
# hires use-and-release QKVO: drop after last sol hit of each DiT step.
export H3_SOL_HIRES_QKVO="${H3_SOL_HIRES_QKVO:-1}"
# compile the hires sol kernel while INT8 is down, before first SDPA.
export H3_SOL_WARMUP_SEQ="${H3_SOL_WARMUP_SEQ:-38676}"

# --- INT8 window: do NOT reduce below 24 (default 34). No shrink. ---
export H3_LIFT_HIRES_WINDOW="${H3_LIFT_HIRES_WINDOW:-34}"
unset H3_DIT_INT8_ALLOW_SHRINK

# --- models ---
DIT="${DIT:-$H3_MODELS/minimax_h3_fl2va_pruned_int8_convrot.safetensors}"
VAE="${VAE:-$H3_MODELS/minimax_h3_video_vae_fp16.safetensors}"
TE="${TE:-$H3_MODELS/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors}"
AUDIO="${AUDIO:-$H3_MODELS/minimax_h3_audio_vae_fp32.safetensors}"
TM_LORA="${H3_TAOMATE_LORA:-$H3_MODELS/TaoMate-H3/adapter_model.safetensors}"
UPS="${H3_LATENT_UPSCALER:-$H3_MODELS/latent_upscale_models/minimax_h3_latent_upscaler_3d_bf16.safetensors}"
for f in "$DIT" "$VAE" "$TE" "$AUDIO" "$TM_LORA" "$UPS"; do
  [ -f "$f" ] || { echo "missing model: $f" >&2; exit 2; }
done

# --- 2-stage geometry: 864x480 draft -> LBH -> 1376x768 hires ---
export H3_TAOMATE_STAGE1_W="${H3_TAOMATE_STAGE1_W:-864}"
export H3_TAOMATE_STAGE1_H="${H3_TAOMATE_STAGE1_H:-480}"
export H3_TAOMATE_CLEAN_KV=0

WIDTH=1376
HEIGHT=768
SECONDS=5
STEPS="${STEPS:-5}"
SEED="${SEED:-0x76800001}"
GEN="${H3_GENERATE:-$ROOT/build-cuda-sm120/h3_cuda_generate}"
IDS="${H3_TOKEN_IDS:-$ROOT/evals/768p/ids/p01_sprint.ids}"

OUT="$ROOT/out/review-2026-10-01/${TAG}"
mkdir -p "$OUT"
MUX="$OUT/${TAG}_${SECONDS}s.mp4"
LOG="$OUT/generate.log"

echo "kpi-b all-sol qkvo: H3_SOL_HIRES_QKVO=$H3_SOL_HIRES_QKVO window=$H3_LIFT_HIRES_WINDOW"
echo "  draft ${H3_TAOMATE_STAGE1_W}x${H3_TAOMATE_STAGE1_H} -> LBH -> ${WIDTH}x${HEIGHT}"
echo "  warmup_seq=$H3_SOL_WARMUP_SEQ max_seq=$H3_SOL_MAX_SEQ repeat=$REPEAT"
echo "  out=$OUT"

cmd=("$GEN"
  --preset taomate-lift --dit "$DIT" --vae "$VAE" --te "$TE" --audio-vae "$AUDIO"
  --lora "$TM_LORA" --lora-scale 0.75 --latent-upscaler "$UPS"
  --steps "$STEPS" --cfg 1 --sdpa-backend sol_cute
  --width "$WIDTH" --height "$HEIGHT" --seconds "$SECONDS"
  --seed "$SEED" --repeat "$REPEAT"
  --out "$OUT" --mux "$MUX")
[ -f "$IDS" ] && cmd+=(--token-ids-file "$IDS")

echo "log=$LOG"
"${cmd[@]}" >"$LOG" 2>&1
rc=$?
echo "exit=$rc"
grep -E 'phase_ms:|repeat_clip:|sol_cute stats|sol_cute hits|hires_qkvo|adaln_dev|out of memory|av_sync|waffle|checksum|selflift:' "$LOG" | tail -n 60 || true
exit "$rc"
