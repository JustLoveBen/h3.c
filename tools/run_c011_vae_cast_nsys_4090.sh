#!/usr/bin/env bash
# C011 Track-A: nsys re-baseline of VAE cast/prescale AFTER product f16_direct/act_f16/fused_epi.
# Attribute residual cast kernels; gate >=0.3s warm fuseable meat before construction.
# No FaceRefine. No LoRA download. No C015 default flip. Product VAE M4 stays ON.
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
      H3_DIT_NVFP4_HOST_DEQUANT H3_FEWSTEP_PRESET H3_FEWSTEP_CONTRACT \
      H3_DIT_EPI_FUSE H3_DIT_EPI_FUSE_INK H3_DIT_EPI_FUSE_SL \
      H3_DIT_ACT_FP16_STAGE H3_REFINER_LORA H3_FACE_REFINE
export H3_SKIP_PPM=1
export H3_VRAM_LEDGER=1
export H3_TE_CACHE_AUTO=1
export LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}

BIN="$ROOT/build-cuda/h3_cuda_generate"
MODELS="${H3_MODELS:-/home/haiyang/workspace/models}"
DIT="$MODELS/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
VAE="$MODELS/minimax_h3_video_vae_fp16.safetensors"
TE="$MODELS/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
OUT_ROOT="${OUT_ROOT:-$ROOT/out/c011_vae_cast_nsys_480p_20261028}"
SEED=0x76800002
IDS="$ROOT/evals/768p/ids/p10_street_wave_dolly.ids"
W=864; H=480; SEC=5; STEPS=8
NSYS_REP="$OUT_ROOT/nsys_product"
for f in "$BIN" "$DIT" "$VAE" "$TE" "$IDS"; do
  [[ -s "$f" ]] || { echo "missing $f" >&2; exit 2; }
done
mkdir -p "$OUT_ROOT" "$OUT_ROOT/gen"
STAMP=$(date '+%Y-%m-%d %H:%M:%S %Z')
SHA=$(cat "$ROOT/.synced_from_mac_sha" 2>/dev/null || echo unknown)
{
  echo "date_local: $STAMP"
  echo "host: $(hostname) / RTX 4090 sm_89"
  echo "repo: $ROOT"
  echo "git_sha: $SHA"
  echo "claim: C011 VAE cast/prescale residual after product f16_direct"
  echo "scheme: Euler-8 INT8 no-LoRA TE_CACHE_AUTO --repeat 1 (nsys)"
} | tee "$OUT_ROOT/META.txt"

echo "==== nsys profile generate $(date '+%H:%M:%S') ===="
set +e
/usr/bin/time -f "WALL_SEC=%e" nsys profile \
  --force-overwrite=true \
  --sample=none \
  --cpuctxsw=none \
  --trace=cuda,nvtx \
  --cudabacktrace=none \
  --export=sqlite \
  -o "$NSYS_REP" \
  "$BIN" \
    --dit "$DIT" --vae "$VAE" --te "$TE" \
    --skip-audio \
    --width "$W" --height "$H" --seconds "$SEC" --steps "$STEPS" --cfg 1 \
    --seed "$SEED" --sdpa-backend cudnn \
    --token-ids-file "$IDS" \
    --repeat 1 \
    --out "$OUT_ROOT/gen" --mux "$OUT_ROOT/gen/c011_nsys.mp4" \
    >"$OUT_ROOT/generate.log" 2>&1
RC=$?
set -e
echo "NSYS_RC=$RC" | tee -a "$OUT_ROOT/generate.log"
grep -E 'vae_gemm|vae_gemm_path|fallback_cast|phase_ms|WALL_SEC|vae:|vae_repeat|error|Error|FAIL' \
  "$OUT_ROOT/generate.log" | tee "$OUT_ROOT/KPI_extract.txt" || true

echo "==== nsys stats cuda_gpu_kern_sum ===="
nsys stats --report cuda_gpu_kern_sum --format csv --force-export true \
  "$NSYS_REP.nsys-rep" >"$OUT_ROOT/cuda_gpu_kern_sum.csv" 2>"$OUT_ROOT/nsys_stats.err" || true
nsys stats --report cuda_gpu_kern_sum "$NSYS_REP.nsys-rep" \
  >"$OUT_ROOT/cuda_gpu_kern_sum.txt" 2>>"$OUT_ROOT/nsys_stats.err" || true

python3 "$ROOT/tools/c011_nsys_attribute.py" "$OUT_ROOT" "$SHA"
echo "==== DONE $(date '+%H:%M:%S') RC=$RC ===="
cat "$OUT_ROOT/VERDICT.txt"
exit "$RC"
