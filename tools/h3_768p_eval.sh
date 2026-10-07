#!/usr/bin/env bash
# 768P professional eval driver. Does not change generate defaults.
# Ruler: 5s · 1344×768 · wall-clock e2e. See
# docs/superpowers/plans/2026-09-13-h3c-768p-eval-protocol.md
# Never print passwords. Never pkill someone else's generate.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WIDTH=1344
HEIGHT=768
CLIP_SECONDS=5
STEPS=""
ARM="gold"
PROMPT="p01_sprint"
DRY=0
REPEAT=""
ALLOW_CANARY=0
REMOTE=0
ULYSSES=1

usage() {
  cat <<'EOF'
tools/h3_768p_eval.sh — 5s 1344×768 eval (new ruler)

  --arm gold|dit_nvfp4_w4a4|turbo4|turbo4_te|turbo4_tae|turbo4_te_tae|nvfp4|self-lift|ulysses2|taomate|taomate_te_tae|taomate_lift|taomate_lift_z0|taomate_p0|taomate_qfix|taomate_qfix2|taomate_qfix3|taomate_qfix4|taomate_comfy5s|taomate_comfy5s_lift|taomate_comfy5s2|taomate_comfy5s2_lift|taomate_comfy5s3|taomate_comfy5s3_lift|taomate_comfy5s4|taomate_comfy5s4_lift|taomate_comfy5s_qtile|taomate_comfy5s_qtile_lift|taomate_comfy5s_fullvae|taomate_comfy5s_fullvae_lift|taomate_comfy5s_vae_f32|taomate_comfy5s_vae_f32_lift|taomate_comfy5s_vae_align|taomate_comfy5s_vae_align_lift|taomate_comfy5s5|taomate_comfy5s5_lift|taomate_comfy5s6|taomate_comfy5s6_lift|taomate_comfy27|taomate_comfy27_lift
  --prompt p01_sprint|p02_talking_head|...|all
  --seconds 5
  --steps N          override (smoke only; gold quality is 8)
  --dry-run
  --repeat N         in-process clips after weight load (hot i>=2)
  --allow-canary     LAST RESORT geometry smoke; not professional eval
  --remote           ssh to 8×5090 and run there (needs H3_SSH_PASS_5090x8)
  --print-ruler

Env (remote defaults in out/verify_5090x8_gold/MODELS.md):
  H3_MODELS  DIT  VAE  TE  AUDIO  H3_TURBO_LORA  H3_TAE  H3_TE_CACHE
  H3_GENERATE  (default ./build-cuda-sm120/h3_cuda_generate)
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --arm) ARM="$2"; shift 2 ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    --seconds) CLIP_SECONDS="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --repeat) REPEAT="$2"; shift 2 ;;
    --allow-canary) ALLOW_CANARY=1; shift ;;
    --remote) REMOTE=1; shift ;;
    --print-ruler)
      echo "ruler=5s 768P canvas=${WIDTH}x${HEIGHT} fps=24"
      echo "protocol=docs/superpowers/plans/2026-09-13-h3c-768p-eval-protocol.md"
      echo "clips=out/review_videos_768p/"
      echo "progress=out/verify_768p/PROGRESS.md"
      exit 0
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

prompt_index() {
  case "$1" in
    p01_sprint) echo 1 ;;
    p02_talking_head) echo 2 ;;
    p03_corridor) echo 3 ;;
    p04_volumetric) echo 4 ;;
    p05_neon_sign) echo 5 ;;
    p06_orbit) echo 6 ;;
    p07_market_track) echo 7 ;;
    p08_ballroom_dolly) echo 8 ;;
    *) echo "" ;;
  esac
}

ALL_PROMPTS="p01_sprint p02_talking_head p03_corridor p04_volumetric p05_neon_sign p06_orbit p07_market_track p08_ballroom_dolly"

ssh_rsh() {
  local host="${H3_SSH_5090x8:-user@89.185.31.140}"
  local -a base=(ssh -p 22 -o StrictHostKeyChecking=accept-new)
  if [ -n "${H3_SSH_BIND:-}" ]; then
    base+=(-b "${H3_SSH_BIND}")
  else
    base+=(-b 192.168.0.92)
  fi
  if [ -n "${H3_SSH_PASS_5090x8:-}" ] && [ -z "${SSHPASS:-}" ]; then
    export SSHPASS="${H3_SSH_PASS_5090x8}"
  fi
  if [ -n "${SSHPASS:-}" ]; then
    sshpass -e "${base[@]}" "$host" "$@"
  else
    "${base[@]}" "$host" "$@"
  fi
}

if [ "$REMOTE" = 1 ]; then
  # Sync source then run the same script on the box. Password never echoed.
  if [ -x "$ROOT/tools/h3_5090x8_sync.sh" ]; then
    H3_SSH_BIND="${H3_SSH_BIND:-192.168.0.92}" "$ROOT/tools/h3_5090x8_sync.sh"
  fi
  extra=()
  [ -n "$STEPS" ] && extra+=(--steps "$STEPS")
  [ -n "$REPEAT" ] && extra+=(--repeat "$REPEAT")
  [ "$DRY" = 1 ] && extra+=(--dry-run)
  [ "$ALLOW_CANARY" = 1 ] && extra+=(--allow-canary)
  ssh_rsh "cd '${H3_REMOTE_ROOT:-/home/user/h3_c}' && \
    H3_MODELS='${H3_MODELS:-/home/user/models}' \
    CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-0}' \
    bash tools/h3_768p_eval.sh --arm '$ARM' --prompt '$PROMPT' --seconds '$CLIP_SECONDS' ${extra[*]-}"
  exit $?
fi

resolve_ids() {
  local id="$1"
  local ids="$ROOT/evals/768p/ids/${id}.ids"
  if [ -s "$ids" ]; then
    echo "$ids"
    return 0
  fi
  if python3 "$ROOT/tools/h3_tokenize_prompt.py" --prompt-id "$id"; then
    echo "$ids"
    return 0
  fi
  return 1
}

run_one() {
  local id="$1"
  local idx seed out_dir mux gen
  idx="$(prompt_index "$id")"
  if [ -z "$idx" ]; then
    echo "unknown prompt id: $id" >&2
    return 2
  fi
  seed=$((0x76800000 + idx))
  if [ "$ARM" = "taomate_comfy5s2" ] || [ "$ARM" = "taomate_comfy5s2_lift" ] || \
     [ "$ARM" = "taomate_comfy5s3" ] || [ "$ARM" = "taomate_comfy5s3_lift" ] || \
     [ "$ARM" = "taomate_comfy5s4" ] || [ "$ARM" = "taomate_comfy5s4_lift" ] || \
     [ "$ARM" = "taomate_comfy5s_qtile" ] || [ "$ARM" = "taomate_comfy5s_qtile_lift" ] || \
     [ "$ARM" = "taomate_comfy5s_fullvae" ] || [ "$ARM" = "taomate_comfy5s_fullvae_lift" ] || \
     [ "$ARM" = "taomate_comfy5s_vae_f32" ] || [ "$ARM" = "taomate_comfy5s_vae_f32_lift" ] || \
     [ "$ARM" = "taomate_comfy5s_vae_align" ] || [ "$ARM" = "taomate_comfy5s_vae_align_lift" ] || \
     [ "$ARM" = "taomate_comfy5s5" ] || [ "$ARM" = "taomate_comfy5s5_lift" ] || \
     [ "$ARM" = "taomate_comfy5s6" ] || [ "$ARM" = "taomate_comfy5s6_lift" ] || \
     [ "$ARM" = "taomate_comfy27" ] || [ "$ARM" = "taomate_comfy27_lift" ]; then
    WIDTH=1376
    HEIGHT=768
  fi

  local ids_file=""
  local canary=""
  if [ "$DRY" != 1 ]; then
    if ids_file="$(resolve_ids "$id")"; then
      true
    elif [ "$ALLOW_CANARY" = 1 ]; then
      echo "WARN: no ids for $id; --allow-canary cinematic (NOT professional eval)"
      canary="cinematic"
    else
      echo "missing token ids for $id. Tokenize with tools/h3_tokenize_prompt.py" >&2
      echo "or place evals/768p/ids/${id}.ids. Refusing cafe fallback." >&2
      return 2
    fi
  elif [ -s "$ROOT/evals/768p/ids/${id}.ids" ]; then
    ids_file="$ROOT/evals/768p/ids/${id}.ids"
  else
    ids_file="\$EVALS/768p/ids/${id}.ids"
  fi

  H3_MODELS="${H3_MODELS:-/home/user/models}"
  if [ -z "${H3_SAGE3_ROOT:-}" ] && [ -d /home/user/opt/sage3/sageattn3 ]; then
    export H3_SAGE3_ROOT=/home/user/opt/sage3
  fi
  # Only export the embed .so when ROOT actually has sageattn3. Pointing
  # H3_SAGE3_ROOT at build-cuda-sm120 is not a kernel tree.
  if [ -z "${H3_SAGE3_LIB:-}" ] && [ -n "${H3_SAGE3_ROOT:-}" ] && \
     [ -d "${H3_SAGE3_ROOT}/sageattn3" ] && \
     [ -f "$ROOT/build-cuda-sm120/libh3_sage3.so" ]; then
    export H3_SAGE3_LIB="$ROOT/build-cuda-sm120/libh3_sage3.so"
  fi
  export H3_SAGE3_TOOLS="${H3_SAGE3_TOOLS:-$ROOT/tools}"
  # Distro libpython (plugin) cannot see torch/fp4attn unless we put them
  # on PYTHONPATH. 768P sage3 hits=0 was this miss, not packed-seq alone.
  _h3_py310_site="${H3_PY310_SITE:-/home/user/opt/py310/lib/python3.10/site-packages}"
  _h3_pp="$H3_SAGE3_TOOLS"
  [ -d "$_h3_py310_site" ] && _h3_pp="$_h3_py310_site:$_h3_pp"
  [ -d "$H3_SAGE3_ROOT/egg/egg-tmp" ] && _h3_pp="$_h3_pp:$H3_SAGE3_ROOT/egg/egg-tmp"
  [ -f "$H3_SAGE3_ROOT/dist/sageattn3-1.0.0-py3.10-linux-x86_64.egg" ] && \
    _h3_pp="$_h3_pp:$H3_SAGE3_ROOT/dist/sageattn3-1.0.0-py3.10-linux-x86_64.egg"
  [ -d "$H3_SAGE3_ROOT" ] && _h3_pp="$_h3_pp:$H3_SAGE3_ROOT"
  export PYTHONPATH="$_h3_pp${PYTHONPATH:+:$PYTHONPATH}"
  if [ -d "$_h3_py310_site/torch/lib" ]; then
    export LD_LIBRARY_PATH="$_h3_py310_site/torch/lib:/home/user/opt/py310/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
  unset _h3_py310_site _h3_pp
  DIT="${DIT:-$H3_MODELS/minimax_h3_fl2va_pruned_int8_convrot.safetensors}"
  VAE="${VAE:-$H3_MODELS/minimax_h3_video_vae_fp16.safetensors}"
  TE="${TE:-$H3_MODELS/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors}"
  AUDIO="${AUDIO:-$H3_MODELS/minimax_h3_audio_vae_fp32.safetensors}"
  LORA="${H3_TURBO_LORA:-$H3_MODELS/minimax_h3_turbo_v4_step600_ema.safetensors}"
  TAE="${H3_TAE:-}"
  gen="${H3_GENERATE:-$ROOT/build-cuda-sm120/h3_cuda_generate}"

  out_dir="$ROOT/out/verify_768p/${ARM}/${id}"
  mkdir -p "$out_dir" "$ROOT/out/review_videos_768p"
  mux="$ROOT/out/review_videos_768p/${ARM}_${id}_${CLIP_SECONDS}s.mp4"

  local -a cmd=("$gen"
    --dit "$DIT" --vae "$VAE" --te "$TE" --audio-vae "$AUDIO"
    --out "$out_dir" --mux "$mux"
    --width "$WIDTH" --height "$HEIGHT" --seconds "$CLIP_SECONDS"
    --seed "$seed")
  if [ -n "$ids_file" ]; then
    cmd+=(--token-ids-file "$ids_file")
  else
    cmd+=(--token-canary "$canary")
  fi
  if [ -n "$REPEAT" ]; then
    cmd+=(--repeat "$REPEAT")
  fi

  local steps_use="$STEPS"
  ULYSSES=1
  case "$ARM" in
    gold)
      steps_use="${steps_use:-8}"
      cmd+=(--steps "$steps_use" --cfg 1 --sdpa-backend cudnn)
      ;;
    dit_nvfp4_w4a4)
      steps_use="${steps_use:-8}"
      NVFP4_W4A4="${H3_DIT_NVFP4_FILE:-$H3_MODELS/minimax_h3_fl2va_nvfp4_w4a4.safetensors}"
      if [ ! -f "$NVFP4_W4A4" ]; then
        echo "dit_nvfp4_w4a4 arm needs $NVFP4_W4A4" >&2
        return 2
      fi
      # DiT W4A4 FP4 GEMM: gold int8 DIT as base weights, runtime swaps the 4
      # linears/block to FP4 via H3_DIT_NVFP4. Same steps/cfg/sdpa as gold.
      export H3_DIT_NVFP4=1
      export H3_DIT_NVFP4_FILE="$NVFP4_W4A4"
      cmd+=(--steps "$steps_use" --cfg 1 --sdpa-backend cudnn)
      ;;
    turbo4)
      steps_use="${steps_use:-4}"
      cmd+=(--preset turbo4 --lora "$LORA" --steps "$steps_use")
      ;;
    turbo4_te)
      steps_use="${steps_use:-4}"
      cmd+=(--preset turbo4 --lora "$LORA" --steps "$steps_use"
            --te-cache "${H3_TE_CACHE:-$ROOT/out/verify_768p/te_cache}")
      ;;
    turbo4_tae)
      steps_use="${steps_use:-4}"
      if [ -z "$TAE" ]; then
        TAE="${H3_MODELS}/taeh3.safetensors"
      fi
      if [ ! -f "$TAE" ]; then
        echo "turbo4_tae needs H3_TAE (768p latent 84x48 is in contract; untested)." >&2
        return 2
      fi
      cmd+=(--preset turbo4 --lora "$LORA" --steps "$steps_use"
            --vae-decode tae --tae "$TAE")
      ;;
    nvfp4)
      steps_use="${steps_use:-8}"
      DIT_NV="${DIT_NVFP4:-$H3_MODELS/minimax_h3_fl2va_pruned_nvfp4.safetensors}"
      if [ ! -f "$DIT_NV" ]; then
        echo "nvfp4 arm needs $DIT_NV" >&2
        return 2
      fi
      cmd=("$gen"
        --preset nvfp4 --dit "$DIT_NV" --vae "$VAE" --te "$TE" --audio-vae "$AUDIO"
        --out "$out_dir" --mux "$mux"
        --width "$WIDTH" --height "$HEIGHT" --seconds "$CLIP_SECONDS"
        --seed "$seed" --steps "$steps_use" --cfg 1 --sdpa-backend cudnn)
      if [ -n "$ids_file" ]; then
        cmd+=(--token-ids-file "$ids_file")
      fi
      cmd+=(--te-cache "${H3_TE_CACHE:-$ROOT/out/verify_768p/te_cache}")
      ;;
    turbo4_te_tae)
      steps_use="${steps_use:-4}"
      if [ -z "$TAE" ]; then
        TAE="${H3_MODELS}/taeh3.safetensors"
      fi
      if [ ! -f "$TAE" ]; then
        echo "turbo4_te_tae needs H3_TAE" >&2
        return 2
      fi
      cmd+=(--preset turbo4 --lora "$LORA" --steps "$steps_use"
            --te-cache "${H3_TE_CACHE:-$ROOT/out/verify_768p/te_cache}"
            --vae-decode tae --tae "$TAE")
      ;;
    ulysses2)
      steps_use="${steps_use:-8}"
      ULYSSES=2
      cmd+=(--steps "$steps_use" --cfg 1 --sdpa-backend cudnn --ulysses 2)
      ;;
    self-lift)
      steps_use="${steps_use:-4}"
      UPS="${H3_LATENT_UPSCALER:-$H3_MODELS/latent_upscale_models/minimax_h3_latent_upscaler_3d_bf16.safetensors}"
      if [ ! -f "$UPS" ]; then
        echo "self-lift needs LBH upscaler at $UPS (or H3_LATENT_UPSCALER)" >&2
        return 2
      fi
      if [ -z "$TAE" ]; then
        TAE="${H3_MODELS}/taeh3.safetensors"
      fi
      cmd+=(--preset lift --lora "$LORA" --steps "$steps_use"
            --sdpa-backend sage3
            --te-cache "${H3_TE_CACHE:-$ROOT/out/verify_768p/te_cache}"
            --vae-decode tae --tae "$TAE"
            --latent-upscaler "$UPS")
      ;;
    taomate|taomate_te_tae|taomate_lift|taomate_lift_z0|taomate_p0|taomate_qfix|taomate_qfix2|taomate_qfix3|taomate_qfix4|taomate_comfy5s|taomate_comfy5s_lift|taomate_comfy5s2|taomate_comfy5s2_lift|taomate_comfy5s3|taomate_comfy5s3_lift|taomate_comfy5s4|taomate_comfy5s4_lift|taomate_comfy5s_qtile|taomate_comfy5s_qtile_lift|taomate_comfy5s_fullvae|taomate_comfy5s_fullvae_lift|taomate_comfy5s_vae_f32|taomate_comfy5s_vae_f32_lift|taomate_comfy5s_vae_align|taomate_comfy5s_vae_align_lift|taomate_comfy5s5|taomate_comfy5s5_lift|taomate_comfy5s6|taomate_comfy5s6_lift|taomate_comfy27|taomate_comfy27_lift)
      steps_use="${steps_use:-3}"
      if [ "$ARM" = "taomate_qfix4" ]; then
        steps_use="${STEPS:-6}"
      fi
      if [ "$ARM" = "taomate_comfy5s" ] || [ "$ARM" = "taomate_comfy5s_lift" ] || \
         [ "$ARM" = "taomate_comfy5s2" ] || [ "$ARM" = "taomate_comfy5s2_lift" ] || \
         [ "$ARM" = "taomate_comfy5s3" ] || [ "$ARM" = "taomate_comfy5s3_lift" ] || \
         [ "$ARM" = "taomate_comfy5s4" ] || [ "$ARM" = "taomate_comfy5s4_lift" ] || \
         [ "$ARM" = "taomate_comfy5s_qtile" ] || [ "$ARM" = "taomate_comfy5s_qtile_lift" ] || \
         [ "$ARM" = "taomate_comfy5s_fullvae" ] || [ "$ARM" = "taomate_comfy5s_fullvae_lift" ] || \
         [ "$ARM" = "taomate_comfy5s_vae_f32" ] || [ "$ARM" = "taomate_comfy5s_vae_f32_lift" ] || \
         [ "$ARM" = "taomate_comfy5s_vae_align" ] || [ "$ARM" = "taomate_comfy5s_vae_align_lift" ] || \
         [ "$ARM" = "taomate_comfy5s5" ] || [ "$ARM" = "taomate_comfy5s5_lift" ] || \
         [ "$ARM" = "taomate_comfy5s6" ] || [ "$ARM" = "taomate_comfy5s6_lift" ] || \
         [ "$ARM" = "taomate_comfy27" ] || [ "$ARM" = "taomate_comfy27_lift" ]; then
        steps_use="${STEPS:-5}"
      fi
      TM_LORA="${H3_TAOMATE_LORA:-$H3_MODELS/TaoMate-H3/adapter_model.safetensors}"
      if [ ! -f "$TM_LORA" ]; then
        echo "taomate needs H3_TAOMATE_LORA at $TM_LORA" >&2
        return 2
      fi
      if [ "$ARM" = "taomate_lift" ] || [ "$ARM" = "taomate_lift_z0" ] || \
         [ "$ARM" = "taomate_p0" ]; then
        UPS="${H3_LATENT_UPSCALER:-$H3_MODELS/latent_upscale_models/minimax_h3_latent_upscaler_3d_bf16.safetensors}"
        if [ ! -f "$UPS" ]; then
          echo "taomate_lift needs LBH upscaler at $UPS" >&2
          return 2
        fi
        if [ -z "$TAE" ]; then
          TAE="${H3_MODELS}/taeh3.safetensors"
        fi
        if [ ! -f "$TAE" ]; then
          echo "taomate_lift needs H3_TAE" >&2
          return 2
        fi
        if [ "$ARM" = "taomate_lift_z0" ] || [ "$ARM" = "taomate_p0" ]; then
          export H3_LIFT_Z0_ONLY=1
        fi
        if [ "$ARM" = "taomate_p0" ]; then
          export H3_SKIP_PPM=1
          export H3_TAE_TIME_CHUNK="${H3_TAE_TIME_CHUNK:-16}"
          mkdir -p "$ROOT/out/verify_taomate/cuda_cache"
          export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$ROOT/out/verify_taomate/cuda_cache}"
          export CUDA_CACHE_MAXSIZE="${CUDA_CACHE_MAXSIZE:-2147483647}"
        fi
        cmd+=(--preset taomate-lift --lora "$TM_LORA" --steps "$steps_use"
              --latent-upscaler "$UPS"
              --te-cache "${H3_TE_CACHE:-$ROOT/out/verify_768p/te_cache}"
              --vae-decode tae --tae "$TAE")
      elif [ "$ARM" = "taomate_comfy5s_lift" ] || [ "$ARM" = "taomate_comfy5s2_lift" ] || \
           [ "$ARM" = "taomate_comfy5s3_lift" ] || [ "$ARM" = "taomate_comfy5s4_lift" ] || \
           [ "$ARM" = "taomate_comfy5s_qtile_lift" ] || [ "$ARM" = "taomate_comfy5s_fullvae_lift" ] || \
           [ "$ARM" = "taomate_comfy5s_vae_f32_lift" ] || [ "$ARM" = "taomate_comfy5s_vae_align_lift" ] || \
           [ "$ARM" = "taomate_comfy5s5_lift" ] || \
           [ "$ARM" = "taomate_comfy5s6_lift" ] || \
           [ "$ARM" = "taomate_comfy27_lift" ]; then
        UPS="${H3_LATENT_UPSCALER:-$H3_MODELS/latent_upscale_models/minimax_h3_latent_upscaler_3d_bf16.safetensors}"
        if [ ! -f "$UPS" ]; then
          echo "$ARM needs LBH upscaler at $UPS" >&2
          return 2
        fi
        unset H3_LIFT_Z0_ONLY
        unset H3_TAOMATE_LAYER_KV
        unset H3_TAOMATE_FORCE_CHUNKS
        export H3_TAOMATE_CLEAN_KV=0
        export H3_TAOMATE_STAGE1_W="${H3_TAOMATE_STAGE1_W:-864}"
        export H3_TAOMATE_STAGE1_H="${H3_TAOMATE_STAGE1_H:-480}"
        unset H3_LIFT_NO_MAKE_ROOM
        export H3_LIFT_MAKE_ROOM_GIB="${H3_LIFT_MAKE_ROOM_GIB:-12}"
        export H3_DIT_INT8_ALLOW_SHRINK="${H3_DIT_INT8_ALLOW_SHRINK:-1}"
        if [ "$ARM" = "taomate_comfy5s3_lift" ]; then
          export H3_VAE_WEIGHT_CACHE="${H3_VAE_WEIGHT_CACHE:-1}"
          export H3_LIFT_HIRES_WINDOW="${H3_LIFT_HIRES_WINDOW:-21}"
        fi
        if [ "$ARM" = "taomate_comfy5s4_lift" ]; then
          export H3_VAE_WEIGHT_CACHE="${H3_VAE_WEIGHT_CACHE:-1}"
          export H3_LIFT_HIRES_WINDOW="${H3_LIFT_HIRES_WINDOW:-24}"
          export H3_INT8_GEMM_TILE_M="${H3_INT8_GEMM_TILE_M:-4096}"
        fi
        if [ "$ARM" = "taomate_comfy5s_qtile_lift" ]; then
          export H3_VAE_WEIGHT_CACHE="${H3_VAE_WEIGHT_CACHE:-1}"
          export H3_LIFT_HIRES_WINDOW="${H3_LIFT_HIRES_WINDOW:-24}"
          export H3_INT8_GEMM_TILE_M="${H3_INT8_GEMM_TILE_M:-4096}"
          export H3_VAE_TILE_OVERLAP="${H3_VAE_TILE_OVERLAP:-96}"
        fi
        if [ "$ARM" = "taomate_comfy5s_fullvae_lift" ] || \
           [ "$ARM" = "taomate_comfy5s_vae_f32_lift" ]; then
          export H3_VAE_WEIGHT_CACHE="${H3_VAE_WEIGHT_CACHE:-1}"
          export H3_LIFT_HIRES_WINDOW="${H3_LIFT_HIRES_WINDOW:-24}"
          export H3_INT8_GEMM_TILE_M="${H3_INT8_GEMM_TILE_M:-4096}"
          export H3_VAE_NO_SPATIAL_TILE=1
          export H3_VAE_SDPA_F32=1
          export NVIDIA_TF32_OVERRIDE=0
          unset H3_VAE_TILE_PIXELS
          unset H3_VAE_TILE_OVERLAP
        fi
        if [ "$ARM" = "taomate_comfy5s_vae_align_lift" ]; then
          export H3_VAE_WEIGHT_CACHE="${H3_VAE_WEIGHT_CACHE:-1}"
          export H3_LIFT_HIRES_WINDOW="${H3_LIFT_HIRES_WINDOW:-24}"
          export H3_INT8_GEMM_TILE_M="${H3_INT8_GEMM_TILE_M:-4096}"
          # Comfy MiniMaxH3VideoVAE: tiling=True tile_size=256 overlap=64.
          # Node 74 VAEDecode still tiled_decode; RoPE denom = tile H/W.
          unset H3_VAE_NO_SPATIAL_TILE
          unset H3_VAE_SDPA_F32
          unset NVIDIA_TF32_OVERRIDE
          export H3_VAE_TILE_PIXELS=256
          export H3_VAE_TILE_OVERLAP=64
          export H3_VAE_COMFY_BLEND=1
        fi
        SDPA_BE=cudnn
        if [ "$ARM" = "taomate_comfy5s5_lift" ] || \
           [ "$ARM" = "taomate_comfy5s6_lift" ] || \
           [ "$ARM" = "taomate_comfy27_lift" ]; then
          export H3_VAE_WEIGHT_CACHE="${H3_VAE_WEIGHT_CACHE:-1}"
          # WS-first then sage3 prebuild while INT8 empty, then refit INT8.
          # cudnn hires 2 NFE stays ~31s (compute, not LRU). sage3 ~11ms/layer.
          export H3_LIFT_HIRES_WINDOW="${H3_LIFT_HIRES_WINDOW:-50}"
          export H3_INT8_GEMM_TILE_M="${H3_INT8_GEMM_TILE_M:-4096}"
          export H3_DIT_NO_WS_MAKE_ROOM=1
          unset H3_DIT_INT8_ALLOW_SHRINK
          unset H3_LIFT_MAKE_ROOM_GIB
          export H3_LIFT_NO_MAKE_ROOM=1
          export H3_W4A8_CACHE_REFIT_RESERVE_MIB="${H3_W4A8_CACHE_REFIT_RESERVE_MIB:-2560}"
          unset H3_VAE_NO_SPATIAL_TILE
          unset H3_VAE_SDPA_F32
          unset NVIDIA_TF32_OVERRIDE
          export H3_VAE_TILE_PIXELS=256
          export H3_VAE_TILE_OVERLAP=64
          export H3_VAE_COMFY_BLEND=1
          SDPA_BE=sage3
            if [ "$ARM" = "taomate_comfy27_lift" ]; then
            # Node 74 kitchen: 4-tile GEMM + one batched cuDNN SDPA (D=64).
            # Naive force-flash was 305s; do not re-enable. Tile-local RoPE.
            unset H3_VAE_FLASH_SDPA
            export H3_VAE_TILE_BATCH=4
            export H3_VAE_CUDA_GRAPH=0
          fi
        fi
        cmd+=(--preset taomate-lift --lora "$TM_LORA" --steps "$steps_use"
              --lora-scale 0.75 --latent-upscaler "$UPS" --sdpa-backend "$SDPA_BE"
              --te-cache "${H3_TE_CACHE:-$ROOT/out/verify_768p/te_cache}")
        echo "$ARM: official VAE; 5s oneshot Comfy 2-stage: 3 NFE @ ${H3_TAOMATE_STAGE1_W}x${H3_TAOMATE_STAGE1_H} + LBH + 2 NFE @ ${WIDTH}x${HEIGHT}; "\
"simple-5 SplitSigmas; lora 0.75; ${SDPA_BE}; not TAE; KPI dual 27s"
      else
        cmd+=(--preset taomate --lora "$TM_LORA" --steps "$steps_use")
        if [ "$ARM" = "taomate_comfy5s" ] || [ "$ARM" = "taomate_comfy5s2" ] || \
           [ "$ARM" = "taomate_comfy5s3" ] || [ "$ARM" = "taomate_comfy5s4" ] || \
           [ "$ARM" = "taomate_comfy5s_qtile" ] || [ "$ARM" = "taomate_comfy5s_fullvae" ] || \
           [ "$ARM" = "taomate_comfy5s_vae_f32" ] || [ "$ARM" = "taomate_comfy5s_vae_align" ] || \
           [ "$ARM" = "taomate_comfy5s5" ] || [ "$ARM" = "taomate_comfy5s6" ] || \
           [ "$ARM" = "taomate_comfy27" ]; then
          unset H3_LIFT_Z0_ONLY
          unset H3_TAOMATE_LAYER_KV
          unset H3_TAOMATE_FORCE_CHUNKS
          export H3_TAOMATE_CLEAN_KV=0
          if [ "$ARM" = "taomate_comfy5s3" ] || [ "$ARM" = "taomate_comfy5s4" ] || \
             [ "$ARM" = "taomate_comfy5s_qtile" ] || [ "$ARM" = "taomate_comfy5s_fullvae" ] || \
             [ "$ARM" = "taomate_comfy5s_vae_f32" ] || [ "$ARM" = "taomate_comfy5s_vae_align" ] || \
             [ "$ARM" = "taomate_comfy5s5" ] || [ "$ARM" = "taomate_comfy5s6" ] || \
             [ "$ARM" = "taomate_comfy27" ]; then
            export H3_VAE_WEIGHT_CACHE="${H3_VAE_WEIGHT_CACHE:-1}"
          fi
          if [ "$ARM" = "taomate_comfy5s4" ] || [ "$ARM" = "taomate_comfy5s_qtile" ] || \
             [ "$ARM" = "taomate_comfy5s_fullvae" ] || \
             [ "$ARM" = "taomate_comfy5s_vae_f32" ] || \
             [ "$ARM" = "taomate_comfy5s_vae_align" ] || \
             [ "$ARM" = "taomate_comfy5s5" ] || [ "$ARM" = "taomate_comfy5s6" ] || \
             [ "$ARM" = "taomate_comfy27" ]; then
            export H3_INT8_GEMM_TILE_M="${H3_INT8_GEMM_TILE_M:-4096}"
          fi
          if [ "$ARM" = "taomate_comfy5s_qtile" ]; then
            export H3_VAE_TILE_OVERLAP="${H3_VAE_TILE_OVERLAP:-96}"
          fi
          if [ "$ARM" = "taomate_comfy5s_fullvae" ] || \
             [ "$ARM" = "taomate_comfy5s_vae_f32" ]; then
            export H3_VAE_NO_SPATIAL_TILE=1
            export H3_VAE_SDPA_F32=1
            export NVIDIA_TF32_OVERRIDE=0
            unset H3_VAE_TILE_PIXELS
            unset H3_VAE_TILE_OVERLAP
          fi
          if [ "$ARM" = "taomate_comfy5s_vae_align" ]; then
            unset H3_VAE_NO_SPATIAL_TILE
            unset H3_VAE_SDPA_F32
            unset NVIDIA_TF32_OVERRIDE
            export H3_VAE_TILE_PIXELS=256
            export H3_VAE_TILE_OVERLAP=64
            export H3_VAE_COMFY_BLEND=1
          fi
          SDPA_BE=cudnn
          if [ "$ARM" = "taomate_comfy5s5" ] || [ "$ARM" = "taomate_comfy5s6" ] || \
             [ "$ARM" = "taomate_comfy27" ]; then
            unset H3_VAE_NO_SPATIAL_TILE
            unset H3_VAE_SDPA_F32
            unset NVIDIA_TF32_OVERRIDE
            export H3_VAE_TILE_PIXELS=256
            export H3_VAE_TILE_OVERLAP=64
            export H3_VAE_COMFY_BLEND=1
            SDPA_BE=sage3
            if [ "$ARM" = "taomate_comfy27" ]; then
              unset H3_VAE_FLASH_SDPA
              export H3_VAE_TILE_BATCH=4
              export H3_VAE_CUDA_GRAPH=0
            fi
          fi
          cmd+=(--sdpa-backend "$SDPA_BE" --lora-scale 0.75
                --te-cache "${H3_TE_CACHE:-$ROOT/out/verify_768p/te_cache}")
          echo "$ARM: official VAE; 5s oneshot Comfy simple-5 dense (not 4 codec chunks); "\
"lora 0.75; ${SDPA_BE}; not TAE; KPI no-dual 45s"
        fi
        if [ "$ARM" = "taomate_qfix" ] || [ "$ARM" = "taomate_qfix2" ]; then
          export H3_TAOMATE_FORCE_CHUNKS=1
          export H3_TAOMATE_CHUNK_OVERLAP="${H3_TAOMATE_CHUNK_OVERLAP:-2}"
          export H3_TAOMATE_SINK_T=0
          export H3_TAOMATE_CLEAN_KV=0
          unset H3_LIFT_Z0_ONLY
          unset H3_TAOMATE_LAYER_KV
          cmd+=(--te-cache "${H3_TE_CACHE:-$ROOT/out/verify_768p/te_cache}")
          echo "$ARM: official VAE; shared-noise denoise_ov=${H3_TAOMATE_CHUNK_OVERLAP}; "\
"sink=0 (768P packed 14344 gemm-acc OOM); not Z0; not full-clip KV"
        fi
        if [ "$ARM" = "taomate_qfix3" ]; then
          export H3_TAOMATE_FORCE_CHUNKS=1
          export H3_TAOMATE_LAYER_KV=1
          export H3_TAOMATE_CHUNK_OVERLAP="${H3_TAOMATE_CHUNK_OVERLAP:-2}"
          export H3_TAOMATE_SINK_T="${H3_TAOMATE_SINK_T:-2}"
          export H3_TAOMATE_CLEAN_KV=0
          unset H3_LIFT_Z0_ONLY
          cmd+=(--te-cache "${H3_TE_CACHE:-$ROOT/out/verify_768p/te_cache}")
          echo "$ARM: official VAE; layer-wise KV hist=${H3_TAOMATE_CHUNK_OVERLAP} sink=${H3_TAOMATE_SINK_T}; "\
"Q=chunk rect flash (not sage3 pad, not packed 14344); not TAE; not Z0"
        fi
        if [ "$ARM" = "taomate_qfix4" ]; then
          export H3_TAOMATE_FORCE_CHUNKS=1
          export H3_TAOMATE_LAYER_KV=1
          # hist=6+sink=2 ≈11.0GiB KV mallocs, then chunk1 INT8 w_i8 OOM.
          # hist=4+sink=2 ≈8.25GiB; KEEP NFE=6. ALLOW_SHRINK so freeze
          # cannot lock a 17-block window before the rect expand.
          export H3_TAOMATE_CHUNK_OVERLAP="${H3_TAOMATE_CHUNK_OVERLAP:-4}"
          export H3_TAOMATE_SINK_T="${H3_TAOMATE_SINK_T:-2}"
          export H3_TAOMATE_CLEAN_KV=0
          export H3_DIT_INT8_ALLOW_SHRINK="${H3_DIT_INT8_ALLOW_SHRINK:-1}"
          unset H3_LIFT_Z0_ONLY
          cmd+=(--te-cache "${H3_TE_CACHE:-$ROOT/out/verify_768p/te_cache}")
          echo "$ARM: official VAE; layer-KV hist=${H3_TAOMATE_CHUNK_OVERLAP} sink=${H3_TAOMATE_SINK_T} nfe=${steps_use}; "\
"Q=chunk rect flash; 6-step midpoints; hist6 OOM so default hist=4; not TAE; not Z0"
        fi
        if [ "$ARM" = "taomate_te_tae" ]; then
          if [ -z "$TAE" ]; then
            TAE="${H3_MODELS}/taeh3.safetensors"
          fi
          if [ ! -f "$TAE" ]; then
            echo "taomate_te_tae needs H3_TAE" >&2
            return 2
          fi
          cmd+=(--te-cache "${H3_TE_CACHE:-$ROOT/out/verify_768p/te_cache}"
                --vae-decode tae --tae "$TAE")
        fi
      fi
      ;;
    *)
      echo "unknown --arm $ARM" >&2
      return 2
      ;;
  esac

  echo "ruler=${CLIP_SECONDS}s 768P ${WIDTH}x${HEIGHT} arm=$ARM prompt=$id seed=$(printf '0x%x' "$seed")"
  echo "mux=$mux"
  if [ "$DRY" = 1 ]; then
    printf 'dry-run:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    return 0
  fi
  if [ ! -x "$gen" ]; then
    echo "generate missing: $gen (sync/build on 5090x8 first)" >&2
    return 2
  fi
  # Do not pkill. If GPU is busy the process will queue or fail; caller decides.
  local log="$out_dir/generate.log"
  echo "log=$log"
  "${cmd[@]}" >"$log" 2>&1
  local rc=$?
  if grep -E 'phase_ms:|repeat_clip:|repeat_core:|taomate_lift:|selflift: time_chunk|dit_host_stage:' "$log" | tail -n 40; then
    true
  fi
  echo "exit=$rc"
  return "$rc"
}

if [ "$PROMPT" = "all" ]; then
  rc=0
  for p in $ALL_PROMPTS; do
    if ! run_one "$p"; then
      rc=1
    fi
  done
  exit "$rc"
fi
run_one "$PROMPT"
