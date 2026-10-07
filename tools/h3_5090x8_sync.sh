#!/usr/bin/env bash
# tools/h3_5090x8_sync.sh — source tree only. Never rsync --delete. Never print passwords.
set -euo pipefail
HOST="${H3_SSH_5090x8:-user@89.185.31.140}"
REMOTE_ROOT="${H3_REMOTE_ROOT:-/home/user/h3_c}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ -n "${H3_SSH_PASS_5090x8:-}" ] && [ -z "${SSHPASS:-}" ]; then
  export SSHPASS="${H3_SSH_PASS_5090x8}"
fi
SSH_BASE=(ssh -p 22 -o StrictHostKeyChecking=accept-new)
if [ -n "${H3_SSH_BIND:-}" ]; then
  SSH_BASE+=(-b "${H3_SSH_BIND}")
fi
if [ -n "${SSHPASS:-}" ]; then
  RSH=(sshpass -e "${SSH_BASE[@]}")
else
  RSH=("${SSH_BASE[@]}")
fi
# rsync -e wants one command string; caller may override with H3_RSYNC_RSH.
if [ -z "${H3_RSYNC_RSH:-}" ]; then
  H3_RSYNC_RSH="${RSH[*]}"
fi
"${RSH[@]}" "$HOST" "mkdir -p '$REMOTE_ROOT'"
rsync -az --exclude '.git/' --exclude 'out/' --exclude 'models/' \
  --exclude 'third_party/accel_survey/FastVideo/' \
  --exclude 'third_party/accel_survey/vdn-minimax-h3/' \
  --exclude 'third_party/accel_survey/MiniMax-H3-MultiGPU-Lightning/' \
  -e "$H3_RSYNC_RSH" "$ROOT/" "$HOST:$REMOTE_ROOT/"
echo "synced source -> $HOST:$REMOTE_ROOT (no --delete, no models)"
