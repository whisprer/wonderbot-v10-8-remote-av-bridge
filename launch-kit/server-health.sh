#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

wb_env

echo "=== WonderBot launch health ==="
echo "Repo:    $WB_REPO"
echo "Profile: $WB_PROFILE"
echo "Bridge:  $(wb_health_url)"
echo

echo "=== bridge PID ==="
PID_FILE="$WB_PID_DIR/bridge-server.pid"
if wb_is_pid_alive "$PID_FILE"; then
  echo "bridge-server: RUNNING pid=$(cat "$PID_FILE")"
else
  echo "bridge-server: NOT RUNNING"
fi
echo

echo "=== bridge health ==="
if command -v python >/dev/null 2>&1; then
  curl -fsS "$(wb_health_url)" | python -m json.tool || true
else
  curl -fsS "$(wb_health_url)" || true
fi
echo

echo "=== git status ==="
git status --short || true
