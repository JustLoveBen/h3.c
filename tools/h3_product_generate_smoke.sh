#!/usr/bin/env bash
# Shell unit smoke for tools/h3_product_generate.sh (no GPU required).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PG="$ROOT/tools/h3_product_generate.sh"
fail=0
pass=0

expect_ok() {
  local name="$1"; shift
  if out="$("$PG" "$@" --dry-run 2>&1)"; then
    echo "PASS $name"
    pass=$((pass + 1))
  else
    echo "FAIL $name (expected ok): $out" >&2
    fail=$((fail + 1))
  fi
}

expect_refuse() {
  local name="$1"; shift
  set +e
  out="$("$PG" "$@" --dry-run 2>&1)"
  rc=$?
  set -e
  if [[ "$rc" -eq 2 ]] && printf '%s' "$out" | grep -q 'REFUSED'; then
    echo "PASS $name (refused)"
    pass=$((pass + 1))
  else
    echo "FAIL $name (want refuse rc=2): rc=$rc out=$out" >&2
    fail=$((fail + 1))
  fi
}

expect_ok rc0 --sku rc0
expect_ok rc0_euler --sku rc0_euler
expect_ok rc1 --sku rc1
expect_ok rc1c --sku rc1c
expect_ok rc2 --sku rc2
expect_ok rc960 --sku rc960
expect_ok compose_dmad_1152 --nfe dmad4 --canvas 1152
expect_ok compose_euler_1344 --nfe euler8 --canvas 1344
expect_ok compose_euler_480 --nfe euler8 --canvas 480

expect_refuse refuse_1280 --canvas 1280x720 --nfe euler8
expect_refuse refuse_720p --nfe euler8 --canvas 720p
expect_refuse refuse_1344_dmad --nfe dmad4 --canvas 1344
expect_refuse refuse_1344_pdmd --nfe pdmd4 --canvas 1344
expect_refuse refuse_sku_1344_dmad --sku rc2 --nfe dmad4
expect_refuse refuse_pdmd2 --nfe pdmd2 --canvas 480

out="$("$PG" --sku rc1 --dry-run 2>&1)" || true
if printf '%s' "$out" | grep -q 'fewstep-preset dmad4' && printf '%s' "$out" | grep -q '1152'; then
  echo "PASS rc1 argv contains fewstep+1152"
  pass=$((pass + 1))
else
  echo "FAIL rc1 argv missing fewstep/1152: $out" >&2
  fail=$((fail + 1))
fi

out="$("$PG" --sku rc2 --dry-run 2>&1)" || true
if printf '%s' "$out" | grep -q -- '--steps 8' && ! printf '%s' "$out" | grep -q 'fewstep-preset'; then
  echo "PASS rc2 argv steps8 no-fewstep"
  pass=$((pass + 1))
else
  echo "FAIL rc2 argv: $out" >&2
  fail=$((fail + 1))
fi

echo "h3_product_generate_smoke: pass=$pass fail=$fail"
[[ "$fail" -eq 0 ]]
