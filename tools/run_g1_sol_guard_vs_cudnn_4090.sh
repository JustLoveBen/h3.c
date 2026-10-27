#!/usr/bin/env bash
# Exp-G1 Track B: 4090 sol_cute+dense guardrails vs cudnn @ DMAD-4
# Gate: 480p x 5s complex p10 INT8. Do NOT change product default.
# Arms: (1) cudnn (2) sol FIRST_DENSE_*=0 (3) sol LAYERS=2 (4) sol STEPS=1
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
      H3_SOL_FIRST_DENSE_STEPS H3_SOL_FIRST_DENSE_LAYERS H3_SDPA_STRICT \
      H3_CUDA_SDPA_BACKEND H3_SOL_CUTE_LIB H3_SOL_ATTN_ROOT H3_TRITON_OVERLAY \
      H3_SOL_CUTE_TOOLS

export H3_SKIP_PPM=1
export H3_TE_CACHE_AUTO=1
export H3_VAE_PINNED_D2H=1
# Fairness: product INT8 window=41 OOMs sol pool @ seq≈4978 (smoke evidence).
# All arms match C041 window=24 + no VAE prewarm so SDPA is the only variable.
export H3_DIT_INT8_WINDOW="${H3_DIT_INT8_WINDOW:-24}"
export H3_VAE_PREWARM_DEV="${H3_VAE_PREWARM_DEV:-0}"
export PATH="$CONDA_PREFIX/bin:$PATH"
export LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:$ROOT/build-cuda:${LD_LIBRARY_PATH:-}

BIN="$ROOT/build-cuda/h3_cuda_generate"
SOL_LIB="$ROOT/build-cuda/libh3_sol_cute.so"
SOL_ATTN_ROOT="${H3_SOL_ATTN_ROOT:-/home/haiyang/h3_c/sol-engine-official/techniques/sparse_backends}"
TRITON_OVERLAY="${H3_TRITON_OVERLAY:-/home/haiyang/h3_c/triton34_overlay}"
MODELS="${H3_MODELS:-/home/haiyang/workspace/models}"
DIT="$MODELS/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
VAE="$MODELS/minimax_h3_video_vae_fp16.safetensors"
TE="$MODELS/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
DMAD_LORA="$MODELS/loras/dmad_minimax_h3_4step_full_critic.h3repo.safetensors"
OUT_ROOT="${OUT_ROOT:-$ROOT/out/g1_sol_guard_vs_cudnn_dmad4_480p_20261028}"
SEED=0x76800002
IDS="$ROOT/evals/768p/ids/p10_street_wave_dolly.ids"
W=864; H=480; SEC=5

for f in "$BIN" "$SOL_LIB" "$DIT" "$VAE" "$TE" "$IDS" "$DMAD_LORA"; do
  [[ -s "$f" ]] || { echo "missing $f" >&2; exit 2; }
done
[[ -d "$SOL_ATTN_ROOT/sol_attn" ]] || { echo "missing SOL_ATTN_ROOT=$SOL_ATTN_ROOT" >&2; exit 2; }
[[ -d "$TRITON_OVERLAY/triton" ]] || { echo "missing TRITON_OVERLAY=$TRITON_OVERLAY" >&2; exit 2; }

mkdir -p "$OUT_ROOT"
STAMP=$(date '+%Y-%m-%d %H:%M:%S %Z')
SHA=$(cat "$ROOT/.synced_from_mac_sha" 2>/dev/null || echo unknown)
{
  echo "track: B"
  echo "exp: G1"
  echo "date_local: $STAMP"
  echo "host: $(hostname) / RTX 4090 sm_89"
  echo "repo: $ROOT"
  echo "git_sha: $SHA"
  echo "product_default: INT8 cudnn (NOT changed by this opt-in probe)"
  echo "int8_window: 24 (matched all arms; product 41 cannot host sol_cute pool @480p)"
  echo "vae_prewarm_dev: 0 (matched; frees sol pool)"
  echo "gate: ${W}x${H} seconds=${SEC} complex=p10_street_wave_dolly DMAD-4"
  echo "seed: $SEED"
  echo "sol_lib: $SOL_LIB"
  echo "sol_attn_root: $SOL_ATTN_ROOT"
  echo "triton_overlay: $TRITON_OVERLAY"
  echo "PASS: arm3_waffle_mean <= arm1 + 0.10; warm_dit <= arm1*0.92; hits explained"
} | tee "$OUT_ROOT/META.txt"

arm_sol_env() {
  export H3_SOL_CUTE_LIB="$SOL_LIB"
  export H3_SOL_ATTN_ROOT="$SOL_ATTN_ROOT"
  export H3_TRITON_OVERLAY="$TRITON_OVERLAY"
  export H3_SOL_CUTE_TOOLS="$ROOT/tools"
  export PYTHONPATH="$TRITON_OVERLAY:$SOL_ATTN_ROOT:$ROOT/tools"
  export PYTHONHOME="$CONDA_PREFIX"
  export H3_SDPA_STRICT=0
}

disarm_sol_env() {
  unset H3_SOL_CUTE_LIB H3_SOL_ATTN_ROOT H3_TRITON_OVERLAY H3_SOL_CUTE_TOOLS
  unset H3_SOL_FIRST_DENSE_STEPS H3_SOL_FIRST_DENSE_LAYERS H3_SDPA_STRICT
  export PYTHONPATH=""
  unset PYTHONHOME
}

run_arm() {
  local ARM=$1 BACKEND=$2
  local OUT="$OUT_ROOT/$ARM"
  mkdir -p "$OUT"
  echo "==== $ARM backend=$BACKEND $(date '+%H:%M:%S') ===="
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || true
  set +e
  /usr/bin/time -f "WALL_SEC=%e" "$BIN" \
    --dit "$DIT" --vae "$VAE" --te "$TE" \
    --skip-audio \
    --width "$W" --height "$H" --seconds "$SEC" --steps 4 --cfg 1 \
    --seed "$SEED" --sdpa-backend "$BACKEND" \
    --token-ids-file "$IDS" \
    --repeat 2 \
    --fewstep-preset dmad4 \
    --lora "$DMAD_LORA" --lora-scale 1 \
    --out "$OUT" --mux "$OUT/${ARM}.mp4" \
    >"$OUT/generate.log" 2>&1
  RC=$?
  set -e
  echo "RUN_RC=$RC" | tee -a "$OUT/generate.log"
  {
    echo "=== $ARM KPI extract ==="
    grep -E 'device_caps:|sdpa_select:|sdpa_strict:|sol_cute|sage3|dit_format|te_prompt_cache|fewstep_|sampler:|phase_ms|WALL_SEC|lora=|h3_lora:|waffle:|waffle_cold|error|Error|FAIL|hits|want=|dense_skip|backend=' \
      "$OUT/generate.log" || true
  } | tee "$OUT/KPI.txt"
  ls -lh "$OUT"/*.mp4 2>/dev/null || echo "NO_MP4 $ARM"
  if [[ "$RC" != 0 ]]; then
    echo "FAIL $ARM rc=$RC" >&2
    tail -100 "$OUT/generate.log" >&2
    return "$RC"
  fi
  return 0
}

disarm_sol_env
run_arm arm1_cudnn_dmad4 cudnn

arm_sol_env
export H3_SOL_FIRST_DENSE_STEPS=0
export H3_SOL_FIRST_DENSE_LAYERS=0
run_arm arm2_sol_fullsparse_dmad4 sol_cute

arm_sol_env
export H3_SOL_FIRST_DENSE_STEPS=0
export H3_SOL_FIRST_DENSE_LAYERS=2
run_arm arm3_sol_layers2_dmad4 sol_cute

arm_sol_env
export H3_SOL_FIRST_DENSE_STEPS=1
export H3_SOL_FIRST_DENSE_LAYERS=0
run_arm arm4_sol_steps1_dmad4 sol_cute

disarm_sol_env

LABEL_DIR="$OUT_ROOT/labeled"
mkdir -p "$LABEL_DIR"
for ARM in arm1_cudnn_dmad4 arm2_sol_fullsparse_dmad4 arm3_sol_layers2_dmad4 arm4_sol_steps1_dmad4; do
  IN="$OUT_ROOT/$ARM/${ARM}_r2.mp4"
  [[ -s "$IN" ]] || IN="$OUT_ROOT/$ARM/${ARM}.mp4"
  [[ -s "$IN" ]] || IN=$(ls "$OUT_ROOT/$ARM"/*.mp4 2>/dev/null | head -1 || true)
  [[ -s "$IN" ]] || continue
  WALL=$(grep -E '^WALL_SEC=' "$OUT_ROOT/$ARM/generate.log" | tail -1 | cut -d= -f2)
  LAST=$(grep -E '^phase_ms:' "$OUT_ROOT/$ARM/generate.log" | tail -1)
  DIT_MS=$(echo "$LAST" | sed -n 's/.*dit=\([0-9.]*\).*/\1/p')
  TOT=$(echo "$LAST" | sed -n 's/.*total=\([0-9.]*\).*/\1/p')
  if [[ -n "$TOT" ]]; then
    WARM_S=$(awk -v t="$TOT" 'BEGIN{printf "%.2f", t/1000.0}')
  else
    WARM_S="?"
  fi
  WMEAN=$(OUT_ROOT="$OUT_ROOT" ARM="$ARM" python3 -c 'import re,pathlib,os;p=pathlib.Path(os.environ["OUT_ROOT"])/os.environ["ARM"]/"generate.log";t=p.read_text(errors="replace");rows=re.findall(r"^waffle:\s*frame=(\d+)\s+b16=([-+0-9.eE]+)",t,re.M);vals=[float(b) for _,b in rows[-5:]] if rows else [];print(round(sum(vals)/len(vals),3) if vals else "NA")')
  SCHEME="G1 TrackB ${ARM} | warm ${WARM_S}s dit ${DIT_MS}ms WALL ${WALL}s waffle ${WMEAN}"
  bash "$ROOT/tools/burn_eval_label.sh" "$IN" "$LABEL_DIR/${ARM}_labeled.mp4" "$SCHEME" \
    || echo "label failed $ARM"
done

export OUT_ROOT
python3 "$ROOT/tools/g1_aggregate_verdict.py" | tee "$OUT_ROOT/SUMMARY.md"

echo DONE
ls -lh "$OUT_ROOT"/*/*.mp4 "$LABEL_DIR"/*.mp4 2>/dev/null || true
cat "$OUT_ROOT/VERDICT.txt"
