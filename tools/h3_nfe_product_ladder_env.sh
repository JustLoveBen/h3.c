#!/usr/bin/env bash
# NFE / few-step product ladder — OPT-IN env preset (source this file).
# Does NOT flip product SDPA default (stays cudnn). Does NOT download LoRA.
#
# Evidence / policy:
#   DMAD-4 vs Euler-8 KPI @3cee9ff — warm e2e 25.0s vs 35.8s (1.43×) @480p p10
#   Product primary = DMAD or Euler (t71u); PDMD contrast-only
#   pdmd2 NOT sellable until 2NFE LoRA is on disk (absent @4090 2026-10-28)
#   Doc: docs/research/2026-10-28-nfe-fewstep-contracts.md
#
# Usage:
#   source tools/h3_nfe_product_ladder_env.sh euler8   # Track-A quality gold
#   source tools/h3_nfe_product_ladder_env.sh dmad4    # Track-B speed fewstep
#   source tools/h3_nfe_product_ladder_env.sh pdmd4    # Track-B contrast
#   source tools/h3_nfe_product_ladder_env.sh          # print ladder help
#
# shellcheck disable=SC1090,SC2034
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "source this file; do not execute" >&2
  exit 2
fi

_H3_NFE_ARM="${1:-}"

# Clear stale fewstep env so a prior dmad4 shell cannot contaminate euler8
unset H3_FEWSTEP_PRESET H3_FEWSTEP_CONTRACT 2>/dev/null || true

# Product stack hygiene (orthogonal; safe defaults for KPI-style runs)
export H3_SKIP_PPM="${H3_SKIP_PPM:-1}"
# C015 act-FP16: product default ON since b99d56b — leave unset

_h3_nfe_help() {
  cat <<'HELP' >&2
h3 NFE product ladder (source with: euler8 | dmad4 | pdmd4):

  euler8  Track-A  NFE=8  velocity_euler  no LoRA
          CLI: --steps 8 --cfg 1 --sdpa-backend cudnn
          warm e2e ~35.8s @480p p10 (KPI)

  dmad4   Track-B  NFE=4  lcm_renoise 12/2  REQUIRES --lora full_critic
          sets: H3_FEWSTEP_PRESET=dmad4
          CLI: --fewstep-preset dmad4 --lora "$DMAD_LORA" --lora-scale 1
          warm e2e ~25.0s @480p p10 (1.43× vs Euler-8)

  pdmd4   Track-B contrast-only  NFE=4  velocity_euler 12/3  REQUIRES --lora pdmd4
          sets: H3_FEWSTEP_PRESET=pdmd4
          NOT product primary (t71u)

  NOT sold: pdmd2 (no LoRA on 4090), turbo4×fewstep, FaceRefine, NVFP4, C035 refiner
  Compose S2 WC: source tools/h3_track_b_s2_env.sh AFTER dmad4, then --sdpa-backend sol_cute
  Doc: docs/research/2026-10-28-nfe-fewstep-contracts.md
HELP
}

case "$_H3_NFE_ARM" in
  ""|help|-h|--help)
    _h3_nfe_help
    ;;
  euler8|Euler-8|EULER8)
    export H3_NFE_PRODUCT_ARM=euler8
    export H3_NFE_PRODUCT_STEPS=8
    export H3_NFE_PRODUCT_TRACK=A
    unset H3_FEWSTEP_PRESET H3_FEWSTEP_CONTRACT
    cat <<EOF >&2
h3 NFE ladder: euler8 (Track-A quality gold)
  steps=8 sampler=velocity_euler fewstep=OFF lora=none
  recommend: --steps 8 --cfg 1 --sdpa-backend cudnn
  KPI warm e2e ~35.8s @480p×5s p10 (3cee9ff)
EOF
    ;;
  dmad4|DMAD4|dmad-4)
    export H3_NFE_PRODUCT_ARM=dmad4
    export H3_NFE_PRODUCT_STEPS=4
    export H3_NFE_PRODUCT_TRACK=B
    export H3_FEWSTEP_PRESET=dmad4
    : "${H3_DMAD_LORA:=${H3_MODELS:-/home/haiyang/workspace/models}/loras/dmad_minimax_h3_4step_full_critic.h3repo.safetensors}"
    export H3_DMAD_LORA
    cat <<EOF >&2
h3 NFE ladder: dmad4 (Track-B speed fewstep)
  H3_FEWSTEP_PRESET=$H3_FEWSTEP_PRESET steps=4 sampler=lcm_renoise shifts=12/2
  MUST pass: --fewstep-preset dmad4 --lora "$H3_DMAD_LORA" --lora-scale 1
  H3_DMAD_LORA=$H3_DMAD_LORA
  KPI warm e2e ~25.0s @480p×5s p10 (1.43× vs Euler-8)
  Optional S2: source tools/h3_track_b_s2_env.sh && --sdpa-backend sol_cute
EOF
    ;;
  pdmd4|PDMD4|pdmd-4)
    export H3_NFE_PRODUCT_ARM=pdmd4
    export H3_NFE_PRODUCT_STEPS=4
    export H3_NFE_PRODUCT_TRACK=B_contrast
    export H3_FEWSTEP_PRESET=pdmd4
    : "${H3_PDMD_LORA:=${H3_MODELS:-/home/haiyang/workspace/models}/loras/pdmd_4NFE_lora_model_0.h3repo.safetensors}"
    export H3_PDMD_LORA
    cat <<EOF >&2
h3 NFE ladder: pdmd4 (Track-B CONTRAST-ONLY — not product primary)
  H3_FEWSTEP_PRESET=$H3_FEWSTEP_PRESET steps=4 sampler=velocity_euler shifts=12/3
  MUST pass: --fewstep-preset pdmd4 --lora "$H3_PDMD_LORA" --lora-scale 1
  H3_PDMD_LORA=$H3_PDMD_LORA
EOF
    ;;
  pdmd2|PDMD2)
    cat <<EOF >&2
h3 NFE ladder: pdmd2 REFUSED as product preset
  Contract id exists in h3_fewstep_contract, but 2NFE LoRA is NOT on disk @4090.
  Do not sell. Inventory + put weights, then re-open Track-B floor dig.
  Doc: docs/research/2026-10-28-nfe-fewstep-contracts.md
EOF
    return 2 2>/dev/null || true
    ;;
  *)
    echo "unknown arm: ${_H3_NFE_ARM} (want euler8|dmad4|pdmd4)" >&2
    _h3_nfe_help
    return 2 2>/dev/null || true
    ;;
esac
