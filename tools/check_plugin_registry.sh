#!/usr/bin/env bash
# check_plugin_registry.sh — ratchet lint: every H3_* env switch read via
# getenv()/h3_plugin_enabled() in the source tree must be registered in the
# tools/h3_plugin.h builtin catalog (H3_PLUGIN_BUILTIN_DEFS).
#
# A new experiment that forgets to register its switch gets flagged here.
# Exit 0 if all read switches are registered, 1 otherwise.
#
# Read-only: scans sources, never modifies anything.

set -u
cd "$(dirname "$0")/.."

PLUGIN_HDR="tools/h3_plugin.h"
if [ ! -f "$PLUGIN_HDR" ]; then
    echo "ERROR: $PLUGIN_HDR not found (run from repo root or tools/)" >&2
    exit 2
fi

TMPDIR_LINT="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_LINT"' EXIT

SRC_FILES=$(ls *.c *.cu *.h tools/*.c tools/*.h 2>/dev/null)

# 1. All H3_* vars read via getenv("H3_...") or h3_plugin_enabled("H3_...").
#    grep -h -o strips filenames; sort -u dedups. Doc-comment examples like
#    h3_plugin_enabled("H3_XXX") in h3_plugin.h are excluded (placeholder name).
grep -h -o -E '(getenv|h3_plugin_enabled)[[:space:]]*\([[:space:]]*"H3_[A-Z0-9_]+"' \
    $SRC_FILES 2>/dev/null \
    | sed -E 's/.*"(H3_[A-Z0-9_]+)"/\1/' | sort -u > "$TMPDIR_LINT/read.txt"

# 2. All vars registered in the builtin catalog: X("H3_...", ...) lines.
#    (Registration lines start with X( at line start; doc-comment examples don't.)
grep -o -E '^[[:space:]]*X\("H3_[A-Z0-9_]+"' "$PLUGIN_HDR" \
    | sed -E 's/.*X\("(H3_[A-Z0-9_]+)"/\1/' | sort -u > "$TMPDIR_LINT/registered.txt"

# 2b. Meta/infrastructure switches that drive the registry itself (not plugins).
cat > "$TMPDIR_LINT/exempt.txt" <<'EOF'
H3_PLUGIN_REPORT
H3_XXX
EOF

N_READ=$(wc -l < "$TMPDIR_LINT/read.txt" | tr -d ' ')
N_REG=$(wc -l < "$TMPDIR_LINT/registered.txt" | tr -d ' ')

comm -23 "$TMPDIR_LINT/read.txt" "$TMPDIR_LINT/registered.txt" \
    | comm -23 - "$TMPDIR_LINT/exempt.txt" > "$TMPDIR_LINT/unregistered.txt"
N_UNREG=$(wc -l < "$TMPDIR_LINT/unregistered.txt" | tr -d ' ')

echo "== plugin registry lint =="
echo "scanned sources : $(echo $SRC_FILES | wc -w | tr -d ' ') files (*.c *.cu *.h tools/*.c tools/*.h)"
echo "switches read   : $N_READ (via getenv/h3_plugin_enabled)"
echo "catalog entries : $N_REG (H3_PLUGIN_BUILTIN_DEFS in $PLUGIN_HDR)"
echo

if [ "$N_UNREG" -eq 0 ]; then
    echo "PASS: all $N_READ read switches are registered."
    exit 0
fi

echo "WARN: $N_UNREG switch(es) read via getenv but NOT in builtin catalog:"
while IFS= read -r name; do
    # Show first file:line where each unregistered switch is read.
    loc=$(grep -n -E "(getenv|h3_plugin_enabled)[[:space:]]*\([[:space:]]*\"$name\"" \
          $SRC_FILES 2>/dev/null | head -1 | cut -c1-120)
    echo "  - $name"
    [ -n "$loc" ] && echo "      first read: $loc"
done < "$TMPDIR_LINT/unregistered.txt"
echo
echo "Fix: append an X(...) entry to H3_PLUGIN_BUILTIN_DEFS in $PLUGIN_HDR,"
echo "     or migrate the read site to a registered switch."
exit 1
