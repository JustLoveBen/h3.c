#!/usr/bin/env bash
# Track-B S2 sellable recipe — OPT-IN env preset (source this file).
# Does NOT flip product SDPA default (stays cudnn). Does NOT write git config.
#
# Evidence:
#   Exp-G1 PASS @eb805b4 — sol LAYERS=2 vs cudnn DMAD-4 480p (dit −15%, waffle≈cudnn)
#   C041 — H3_DIT_INT8_WINDOW=24 required for sol headroom (window 41 OOMs sol pool)
#   C015 — H3_DIT_ACT_FP16_STAGE default ON @b99d56b (opt-out =0)
#
# Usage:
#   source tools/h3_track_b_s2_env.sh
#   # then pass --sdpa-backend sol_cute on the generate CLI (or export H3_CUDA_SDPA_BACKEND
#   # only for this shell — still not a product default flip)
#
# One-liner (after source + CLI backend):
#   H3_DIT_INT8_WINDOW=24 H3_SOL_FIRST_DENSE_LAYERS=2 H3_SOL_FIRST_DENSE_STEPS=0 \
#     --sdpa-backend sol_cute   # + sol lib/attn_root/triton overlay
#
# shellcheck disable=SC1090,SC2034
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "source this file; do not execute" >&2
  exit 2
fi

# --- VRAM stack (also valid under cudnn; required for sol) ---
export H3_DIT_INT8_WINDOW="${H3_DIT_INT8_WINDOW:-24}"
export H3_VAE_PREWARM_DEV="${H3_VAE_PREWARM_DEV:-0}"

# --- C015: product default ON since b99d56b; leave unset unless opting out ---
# unset H3_DIT_ACT_FP16_STAGE   # default ON
# export H3_DIT_ACT_FP16_STAGE=0  # opt-out

# --- Sol dense guardrail (G1 arm3) ---
export H3_SOL_FIRST_DENSE_LAYERS="${H3_SOL_FIRST_DENSE_LAYERS:-2}"
export H3_SOL_FIRST_DENSE_STEPS="${H3_SOL_FIRST_DENSE_STEPS:-0}"
# dense_skip conflicts with strict hits==want on sol arms
export H3_SDPA_STRICT="${H3_SDPA_STRICT:-0}"

# --- Paths (override per machine; no machine-local hard fail here) ---
: "${H3_SOL_CUTE_LIB:=${H3_SOL_CUTE_LIB:-}}"
: "${H3_SOL_ATTN_ROOT:=${H3_SOL_ATTN_ROOT:-}}"
: "${H3_TRITON_OVERLAY:=${H3_TRITON_OVERLAY:-}}"
: "${H3_SOL_CUTE_TOOLS:=${H3_SOL_CUTE_TOOLS:-}}"

# Do NOT export H3_CUDA_SDPA_BACKEND here — caller must opt in explicitly.
# Recommended CLI: --sdpa-backend sol_cute

cat <<EOF >&2
h3 Track-B S2 env preset active (opt-in; product default still cudnn):
  H3_DIT_INT8_WINDOW=$H3_DIT_INT8_WINDOW
  H3_SOL_FIRST_DENSE_LAYERS=$H3_SOL_FIRST_DENSE_LAYERS
  H3_SOL_FIRST_DENSE_STEPS=$H3_SOL_FIRST_DENSE_STEPS
  H3_SDPA_STRICT=$H3_SDPA_STRICT
  H3_VAE_PREWARM_DEV=$H3_VAE_PREWARM_DEV
  C015 act-FP16 stage: product default ON (unset); opt-out H3_DIT_ACT_FP16_STAGE=0
  Still need: --sdpa-backend sol_cute + H3_SOL_CUTE_LIB + H3_SOL_ATTN_ROOT + H3_TRITON_OVERLAY
  Doc: docs/research/2026-10-28-track-b-s2-sellable-recipe.md
EOF
