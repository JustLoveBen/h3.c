#!/usr/bin/env bash
# C015 H3_DIT_ACT_FP16_STAGE waffle + peak_dit E2 gate on Ada 4090.
# Same-config A/B: baseline (env unset) vs stage=1.
# E2: waffle PASS + peak_dit drop ≥ 0.5 GiB. Default stays OFF regardless.
# No FaceRefine. No NVFP4. No KEEP=attn. No C018/C003-SL/FlashMLA.
# Scheme: Euler-8 INT8 (t71u primary; no LoRA so wide F16 can arm).
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
      H3_DIT_ACT_FP16_STAGE H3_DIT_ACT_F16
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
OUT_ROOT="${OUT_ROOT:-$ROOT/out/c015_fp16_stage_waffle_480p_20261028}"
SEED=0x76800002
IDS="$ROOT/evals/768p/ids/p10_street_wave_dolly.ids"
W=864; H=480; SEC=5; STEPS=8

for f in "$BIN" "$DIT" "$VAE" "$TE" "$IDS"; do
  [[ -s "$f" ]] || { echo "missing $f" >&2; exit 2; }
done
mkdir -p "$OUT_ROOT"
STAMP=$(date '+%Y-%m-%d %H:%M:%S %Z')
SHA=$(cat "$ROOT/.synced_from_mac_sha" 2>/dev/null || echo unknown)
{
  echo "date_local: $STAMP"
  echo "host: $(hostname) / RTX 4090 sm_89"
  echo "repo: $ROOT"
  echo "git_sha: $SHA"
  echo "claim: C015 H3_DIT_ACT_FP16_STAGE"
  echo "gate: waffle same-config + E2 peak_dit drop >= 0.5 GiB"
  echo "default: OFF (this probe only arms cand)"
  echo "scheme: Euler-8 INT8 no-LoRA (wide F16 arm)"
  echo "seed: $SEED"
  echo "geom: ${W}x${H} seconds=${SEC} complex=p10"
  echo "env_common: H3_VRAM_LEDGER=1 H3_TE_CACHE_AUTO=1 H3_SKIP_PPM=1 --skip-audio --repeat 2"
} | tee "$OUT_ROOT/META.txt"

run_arm() {
  local ARM=$1; shift
  local OUT="$OUT_ROOT/$ARM"
  mkdir -p "$OUT"
  echo "==== $ARM generate $(date '+%H:%M:%S') ===="
  set +e
  /usr/bin/time -f "WALL_SEC=%e" env "$@" "$BIN" \
    --dit "$DIT" --vae "$VAE" --te "$TE" \
    --skip-audio \
    --width "$W" --height "$H" --seconds "$SEC" --steps "$STEPS" --cfg 1 \
    --seed "$SEED" --sdpa-backend cudnn \
    --token-ids-file "$IDS" \
    --repeat 2 \
    --out "$OUT" --mux "$OUT/${ARM}.mp4" \
    >"$OUT/generate.log" 2>&1
  RC=$?
  set -e
  echo "RUN_RC=$RC" | tee -a "$OUT/generate.log"
  {
    echo "=== $ARM KPI extract ==="
    grep -E 'dit_act_fp16_stage|dit_ws_ledger|vram_ledger|vram_ledger_delta|waffle:|phase_ms|WALL_SEC|dit_fused_epi|error|Error|FAIL|C015' \
      "$OUT/generate.log" || true
  } | tee "$OUT/KPI.txt"
  ls -lh "$OUT"/*.mp4 2>/dev/null || echo "NO_MP4 $ARM"
  return "$RC"
}

extract_peak() {
  local log=$1
  # Ledger format: vram_ledger: point=peak_dit ... used_mib=22878.9
  python3 - "$log" <<'INNER'
import re, sys
text = open(sys.argv[1], encoding='utf-8', errors='replace').read()
vals = []
for m in re.finditer(r'vram_ledger:\s*point=peak_dit\b[^\n]*?used_mib=([0-9.]+)', text):
    vals.append(float(m.group(1)) / 1024.0)
print('NA' if not vals else f'{max(vals):.4f}')
INNER
}

# Arm A: baseline (stage unset)
unset H3_DIT_ACT_FP16_STAGE
run_arm base_euler8_off || { echo "FAIL base"; exit 1; }

# Arm B: stage ON
run_arm cand_euler8_stage H3_DIT_ACT_FP16_STAGE=1 || { echo "FAIL cand"; exit 1; }

BASE_LOG="$OUT_ROOT/base_euler8_off/generate.log"
CAND_LOG="$OUT_ROOT/cand_euler8_stage/generate.log"
PEAK_B=$(extract_peak "$BASE_LOG")
PEAK_C=$(extract_peak "$CAND_LOG")

echo "==== waffle gate ===="
set +e
# Near-lossless Track-A seam (not BIT_ID): historical defaults 0.15/0.40
python3 "$ROOT/tools/waffle_same_config_gate.py" \
  --mean-tol 0.15 --max-tol 0.40 --require-frames 5 \
  "$BASE_LOG" "$CAND_LOG" \
  >"$OUT_ROOT/waffle_gate.txt" 2>&1
WRC=$?
set -e
cat "$OUT_ROOT/waffle_gate.txt"

python3 - "$PEAK_B" "$PEAK_C" "$WRC" "$SHA" "$OUT_ROOT" <<'PY'
import sys
from pathlib import Path
pb, pc, wrc, sha, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], Path(sys.argv[5])
def f(x):
    try: return float(x)
    except: return None
b, c = f(pb), f(pc)
drop = (b - c) if (b is not None and c is not None) else None
waffle = "PASS" if wrc == 0 else "FAIL"
e2 = "PASS" if (waffle == "PASS" and drop is not None and drop >= 0.5) else "FAIL"
lines = [
  f"git_sha: {sha}",
  f"waffle: {waffle} (rc={wrc})",
  f"peak_dit_base_GiB: {pb}",
  f"peak_dit_cand_GiB: {pc}",
  f"peak_dit_drop_GiB: {drop if drop is not None else 'NA'}",
  f"E2_peak_drop_ge_0.5: {e2}",
  f"product_default: OFF (do not flip until waffle PASS and E2 PASS)",
  f"verdict: construction landed; gate={waffle}/{e2}",
]
(out / "VERDICT.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
exit "$WRC"
