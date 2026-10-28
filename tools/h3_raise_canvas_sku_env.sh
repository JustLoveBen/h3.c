#!/usr/bin/env bash
# Raise-canvas SKU — OPT-IN env preset (source this file).
# Does NOT flip product SDPA default (stays cudnn). Does NOT download LoRA.
# Does NOT invent FaceRefine / C035 weights.
#
# Evidence / policy:
#   ghost/face A/B @47b4cd8 — 1152 PDMD PASS; 1344 PDMD FAIL noise; 1344 Euler PASS
#   Product primary = DMAD or Euler (t71u); PDMD contrast-only
#   Geometry: W/H must %32==0 (1280×720 rejected)
#   Doc: docs/research/2026-10-28-raise-canvas-sku.md
#
# Usage:
#   source tools/h3_raise_canvas_sku_env.sh 480p         # RC0 default canvas
#   source tools/h3_raise_canvas_sku_env.sh 1152_dmad    # RC1 speed-face
#   source tools/h3_raise_canvas_sku_env.sh 1344_euler   # RC2 max-face
#   source tools/h3_raise_canvas_sku_env.sh              # print help
#
# Compose:
#   source tools/h3_nfe_product_ladder_env.sh dmad4
#   source tools/h3_raise_canvas_sku_env.sh 1152_dmad
# Optional S2 WC (not a face-px lever): source tools/h3_track_b_s2_env.sh
#
# shellcheck disable=SC1090,SC2034
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "source this file; do not execute" >&2
  exit 2
fi

_H3_RC_SKU="${1:-}"

_h3_rc_help() {
  cat <<'HELP' >&2
h3 raise-canvas SKU (source with: 480p | 1152_dmad | 1344_euler | 960 | 1152_pdmd):

  480p / 864x480     RC0  default canvas — pair with NFE euler8 or dmad4
  1152_dmad          RC1  1152×640 + recommend DMAD-4 (speed-face)
  1344_euler         RC2  1344×768 + recommend Euler-8 (max face)
  960 / 960x544      optional intermediate
  1152_pdmd          contrast-only at 1152 (NOT product primary)

  REFUSED:
    1280x720 / 720p     — W/H not %32==0
    1344_pdmd / 1344_dmad — 4090 INT8 fewstep @1344 → FAIL noise (window≈6)

  Exports: H3_RC_SKU H3_RC_WIDTH H3_RC_HEIGHT H3_RC_RECOMMEND_ARM
  Does NOT set H3_CUDA_SDPA_BACKEND. C015 default ON (leave unset).
  Doc: docs/research/2026-10-28-raise-canvas-sku.md
HELP
}

_h3_rc_check_geom() {
  local w="$1" h="$2"
  if (( w % 32 != 0 || h % 32 != 0 )); then
    cat <<EOF >&2
h3 raise-canvas: REFUSED geometry ${w}×${h}
  W and H must each be multiples of 32 (got W%32=$((w%32)) H%32=$((h%32))).
  Example reject: 1280×720.
  Doc: docs/research/2026-10-28-raise-canvas-sku.md
EOF
    return 2
  fi
  return 0
}

_h3_rc_export() {
  local sku="$1" w="$2" h="$3" arm="$4" note="$5"
  _h3_rc_check_geom "$w" "$h" || return 2
  export H3_RC_SKU="$sku"
  export H3_RC_WIDTH="$w"
  export H3_RC_HEIGHT="$h"
  export H3_RC_RECOMMEND_ARM="$arm"
  export H3_WIDTH="$w"
  export H3_HEIGHT="$h"
  cat <<EOF >&2
h3 raise-canvas SKU: $sku
  canvas: ${w}×${h}  recommend_arm=$arm
  CLI: --width $w --height $h
  $note
  Compose NFE: source tools/h3_nfe_product_ladder_env.sh $arm
  C015 default ON; optional C041: H3_DIT_INT8_WINDOW=24
  Doc: docs/research/2026-10-28-raise-canvas-sku.md
EOF
}

case "$_H3_RC_SKU" in
  ""|help|-h|--help)
    _h3_rc_help
    ;;
  480p|480|864x480|RC0|rc0)
    _h3_rc_export "480p" 864 480 "dmad4" \
      "RC0 default — far-face NOT fixed at this canvas; use 1152_dmad or 1344_euler for face budget." || return 2 2>/dev/null || true
    ;;
  960|960x544|RC960|rc960)
    _h3_rc_export "960" 960 544 "dmad4" \
      "Optional intermediate canvas between 480p and 1152." || return 2 2>/dev/null || true
    ;;
  1152_dmad|1152|1152x640|RC1|rc1)
    _h3_rc_export "1152_dmad" 1152 640 "dmad4" \
      "RC1 speed-face — MUST use fewstep≤DMAD/PDMD ceiling; pair with NFE dmad4 + LoRA." || return 2 2>/dev/null || true
    ;;
  1152_pdmd|1152_pdmd4)
    _h3_rc_export "1152_pdmd" 1152 640 "pdmd4" \
      "CONTRAST-ONLY at 1152 (t71u: not product primary). Prefer 1152_dmad for ship." || return 2 2>/dev/null || true
    ;;
  1344_euler|1344|1344x768|RC2|rc2)
    _h3_rc_export "1344_euler" 1344 768 "euler8" \
      "RC2 max-face — Euler-8 only on 4090 INT8. Quote longer WC. Optional WINDOW=24." || return 2 2>/dev/null || true
    ;;
  1280|1280x720|720p|720)
    cat <<EOF >&2
h3 raise-canvas: REFUSED 1280×720
  W/H must %32==0 (720%32=16). Use 1152×640 (speed-face) or 1344×768 (max-face).
  Doc: docs/research/2026-10-28-raise-canvas-sku.md
EOF
    return 2 2>/dev/null || true
    ;;
  1344_pdmd|1344_pdmd4|1344_dmad|1344_dmad4|1344_fewstep)
    cat <<EOF >&2
h3 raise-canvas: REFUSED 1344 + fewstep (PDMD/DMAD) on 4090 INT8
  Evidence @47b4cd8: 1344×768 PDMD-4 → INT8 window≈6 → FAIL noise mosaic.
  Use: source tools/h3_raise_canvas_sku_env.sh 1344_euler
    or drop canvas to 1152_dmad for speed-face fewstep.
  Doc: docs/research/2026-10-28-raise-canvas-sku.md
EOF
    return 2 2>/dev/null || true
    ;;
  *)
    if [[ "$_H3_RC_SKU" =~ ^([0-9]+)[xX]([0-9]+)$ ]]; then
      _w="${BASH_REMATCH[1]}"
      _hdim="${BASH_REMATCH[2]}"
      if ! _h3_rc_check_geom "$_w" "$_hdim"; then
        return 2 2>/dev/null || true
      fi
      if (( _w >= 1344 && _hdim >= 768 )); then
        cat <<EOF >&2
h3 raise-canvas: custom ${_w}×${_hdim} accepted as geometry, but fewstep@≥1344 is NOT sellable on 4090 INT8.
  Recommend arm: euler8 only. Refusing to set H3_FEWSTEP_*.
EOF
        export H3_RC_SKU="custom_${_w}x${_hdim}"
        export H3_RC_WIDTH="$_w"
        export H3_RC_HEIGHT="$_hdim"
        export H3_RC_RECOMMEND_ARM=euler8
        export H3_WIDTH="$_w"
        export H3_HEIGHT="$_hdim"
      else
        _h3_rc_export "custom_${_w}x${_hdim}" "$_w" "$_hdim" "dmad4" \
          "Custom canvas — pick NFE arm explicitly; fewstep ceiling remains ≤1152×640 on 4090." || return 2 2>/dev/null || true
      fi
    else
      echo "unknown sku: ${_H3_RC_SKU}" >&2
      _h3_rc_help
      return 2 2>/dev/null || true
    fi
    ;;
esac
