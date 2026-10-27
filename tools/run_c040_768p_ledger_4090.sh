#!/usr/bin/env bash
# C040 768p VRAM ledger probe on Ada 4090 (24 GiB).
# Official 768P ruler: 1344×768 × 5s (NOT 1280×768).
# Scheme policy (t71u): Euler-8 primary; DMAD-4 second; PDMD skipped.
# Expect tight/OOM vs 480p peak_dit≈22.4 GiB — mid-run buckets still useful.
# No FaceRefine. No NVFP4. No KEEP=attn. skip-audio.
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
      H3_DIT_NVFP4_HOST_DEQUANT H3_FEWSTEP_PRESET H3_FEWSTEP_CONTRACT
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
DMAD_LORA="$MODELS/loras/dmad_minimax_h3_4step_full_critic.h3repo.safetensors"
OUT_ROOT="${OUT_ROOT:-$ROOT/out/c040_768p_ledger_20261028}"
SEED=0x76800002
IDS="$ROOT/evals/768p/ids/p10_street_wave_dolly.ids"
W=1344; H=768; SEC=5

for f in "$BIN" "$DIT" "$VAE" "$TE" "$IDS" "$DMAD_LORA"; do
  [[ -s "$f" ]] || { echo "missing $f" >&2; exit 2; }
done
mkdir -p "$OUT_ROOT"
STAMP=$(date '+%Y-%m-%d %H:%M:%S %Z')
FREE0=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
{
  echo "date_local: $STAMP"
  echo "host: $(hostname) / RTX 4090 sm_89"
  echo "repo: $ROOT"
  echo "git_sha: $(cat $ROOT/.synced_from_mac_sha 2>/dev/null || echo unknown)"
  echo "arch: sm_89 total_vram_mib=24564 free_before_mib=$FREE0"
  echo "product_default: INT8"
  echo "KEEP=attn: off"
  echo "NVFP4: BLOCK / unset"
  echo "FaceRefine: STOPPED"
  echo "scheme_policy_t71u: Euler/DMAD primary; PDMD skipped"
  echo "seed: $SEED"
  echo "gate: ${W}x${H} seconds=${SEC} complex=p10_street_wave_dolly (official 768P ruler)"
  echo "env: H3_VRAM_LEDGER=1 H3_TE_CACHE_AUTO=1 H3_SKIP_PPM=1 --skip-audio --repeat 1"
  echo "note: 480p peak_dit≈22.4GiB; 768p expected tight/OOM on 24GiB — capture mid-run buckets"
} | tee "$OUT_ROOT/META.txt"

run_arm() {
  local ARM=$1 STEPS=$2; shift 2
  local EXTRA=("$@")
  local OUT="$OUT_ROOT/$ARM"
  mkdir -p "$OUT"
  echo "==== $ARM generate $(date '+%H:%M:%S') ===="
  set +e
  /usr/bin/time -f "WALL_SEC=%e" "$BIN" \
    --dit "$DIT" --vae "$VAE" --te "$TE" \
    --skip-audio \
    --width "$W" --height "$H" --seconds "$SEC" --steps "$STEPS" --cfg 1 \
    --seed "$SEED" --sdpa-backend cudnn \
    --token-ids-file "$IDS" \
    --repeat 1 \
    --out "$OUT" --mux "$OUT/${ARM}.mp4" \
    "${EXTRA[@]}" \
    >"$OUT/generate.log" 2>&1
  RC=$?
  set -e
  echo "RUN_RC=$RC" | tee -a "$OUT/generate.log"
  {
    echo "=== $ARM ledger/KPI extract ==="
    grep -E 'vram_ledger|vram_ledger_delta|dit_format|shape:|phase_ms|WALL_SEC|fewstep_|sampler: selected|OOM|out of memory|cudaError|FAIL_|error:|Error' \
      "$OUT/generate.log" || true
  } | tee "$OUT/KPI.txt"
  ls -lh "$OUT"/*.mp4 2>/dev/null || echo "NO_MP4 $ARM (rc=$RC)"
  return 0  # always continue — OOM still yields buckets
}

# Arm A: Euler-8 INT8 primary
run_arm euler8_int8_ledger 8

# Arm B: DMAD-4 INT8 (same peak footprint interest; LoRA on)
run_arm dmad4_int8_ledger 4 \
  --fewstep-preset dmad4 \
  --lora "$DMAD_LORA" --lora-scale 1

# Labels only if mp4 exists
LABEL_DIR="$OUT_ROOT/labeled"
mkdir -p "$LABEL_DIR"
for ARM in euler8_int8_ledger dmad4_int8_ledger; do
  IN="$OUT_ROOT/$ARM/${ARM}.mp4"
  [[ -s "$IN" ]] || IN=$(ls "$OUT_ROOT/$ARM"/*.mp4 2>/dev/null | head -1 || true)
  [[ -s "$IN" ]] || continue
  WALL=$(grep -E '^WALL_SEC=' "$OUT_ROOT/$ARM/generate.log" | tail -1 | cut -d= -f2)
  PEAK=$(grep -E 'vram_ledger:.*peak_dit' "$OUT_ROOT/$ARM/generate.log" | tail -1 \
    | sed -n 's/.*used_mib=\([0-9.]*\).*/\1/p')
  case "$ARM" in
    euler8*) NAME=Euler-8 ;;
    dmad4*)  NAME=DMAD-4 ;;
  esac
  SCHEME="1344x768 ${NAME} INT8 | WALL ${WALL}s peak_dit ${PEAK:-?}MiB"
  bash "$ROOT/tools/burn_eval_label.sh" "$IN" "$LABEL_DIR/${ARM}_labeled.mp4" "$SCHEME" \
    || echo "label failed $ARM"
done

# Aggregate SUMMARY with ledger table
python3 << 'PY'
from pathlib import Path
import re
out = Path("/home/haiyang/workspace/h3_c_main/out/c040_768p_ledger_20261028")
meta = (out/"META.txt").read_text().strip()
arms = ["euler8_int8_ledger", "dmad4_int8_ledger"]
bucket_order = ["after_cuda","after_te","before_dit","peak_dit","after_dit_release","before_vae","after_vae"]
lines = [
  "# C040 768p VRAM ledger probe (t71u Euler/DMAD; PDMD skipped)",
  "", meta, "",
  "## Per-arm status",
  "",
  "| arm | RC | WALL_s | peak_dit_MiB | mp4 | OOM? |",
  "|-----|----|--------|--------------|-----|------|",
]
detail = []
for arm in arms:
    logp = out/arm/"generate.log"
    if not logp.exists():
        lines.append(f"| {arm} | missing | | | | |")
        continue
    log = logp.read_text(errors="replace")
    rc = re.findall(r"^RUN_RC=(\d+)", log, re.M)
    rc = rc[-1] if rc else "?"
    wall = re.findall(r"^WALL_SEC=([0-9.]+)", log, re.M)
    wall = wall[-1] if wall else "?"
    oom = bool(re.search(r"out of memory|cudaErrorMemoryAllocation|OOM", log, re.I))
    mp4s = list((out/arm).glob("*.mp4"))
    buckets = {}
    for m in re.finditer(r"vram_ledger:\s+point=(\S+)\s+free_mib=([0-9.]+)\s+used_mib=([0-9.]+)", log):
        buckets[m.group(1)] = (float(m.group(3)), float(m.group(2)))  # used, free
    # also try alternate format
    if not buckets:
        for m in re.finditer(r"vram_ledger(?:_delta)?:\s*(\S+).*?([0-9.]+)", log):
            pass
    peak = buckets.get("peak_dit", (None,None))[0]
    peak_s = f"{peak:.1f}" if peak is not None else "—"
    lines.append(f"| {arm} | {rc} | {wall} | {peak_s} | {'yes' if mp4s else 'no'} | {'yes' if oom else 'no'} |")
    detail.append((arm, buckets, log))

lines += ["", "## Buckets (used_mib / free_mib)", ""]
# compare table
hdr = "| bucket | " + " | ".join(arms) + " |"
sep = "|--------|" + "|".join(["------"]*len(arms)) + "|"
lines += [hdr, sep]
# gather all keys
keys = []
for b in bucket_order:
    keys.append(b)
for arm, buckets, _ in detail:
    for k in buckets:
        if k not in keys:
            keys.append(k)
for k in keys:
    cells = []
    for arm, buckets, _ in detail:
        if k in buckets:
            u, f = buckets[k]
            cells.append(f"{u:.1f}/{f:.1f}")
        else:
            cells.append("—")
    lines.append(f"| {k} | " + " | ".join(cells) + " |")

# deltas if present
lines += ["", "## Raw ledger lines", ""]
for arm, buckets, log in detail:
    lines.append(f"### {arm}")
    for ln in log.splitlines():
        if "vram_ledger" in ln:
            lines.append("    " + ln.strip())
    lines.append("")

lines += [
  "## Read-out",
  "- Compare peak_dit @1344×768 vs 480p≈22879 MiB to size C015/C041/C043 headroom on 24 GiB 4090.",
  "- OOM before peak_dit still useful: after_te / before_dit show TE+setup footprint.",
  "- C007 still N/A (cudnn path).",
]
(out/"SUMMARY.md").write_text("\n".join(lines)+"\n")
print("\n".join(lines))
PY

echo DONE
ls -lh "$OUT_ROOT"/*/*.mp4 "$LABEL_DIR"/*.mp4 2>/dev/null || true
