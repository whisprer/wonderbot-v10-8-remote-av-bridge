#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

wb_env

PID_FILE="$WB_PID_DIR/bridge-server.pid"
LOG_FILE="$WB_LOG_DIR/bridge-server.log"

if wb_is_pid_alive "$PID_FILE"; then
  echo "Bridge server already running: PID $(cat "$PID_FILE")"
  echo "Health: $(wb_health_url)"
  exit 0
fi

rm -f "$PID_FILE"

echo "Starting WonderBot bridge server..."
echo "Log: $LOG_FILE"

nohup python -m wonderbot.bridge server \
  --host "$WB_BRIDGE_HOST" \
  --port "$WB_BRIDGE_PORT" \
  --token "$WB_BRIDGE_TOKEN_EFFECTIVE" \
  >"$LOG_FILE" 2>&1 &

PID="$!"
echo "$PID" > "$PID_FILE"

echo "Bridge server PID: $PID"

for i in {1..20}; do
  if curl -fsS "$(wb_health_url)" >/dev/null 2>&1; then
    echo "Bridge server health OK: $(wb_health_url)"
    exit 0
  fi
  sleep 0.5
done

echo "WARNING: bridge server did not answer health check yet."
echo "Tail log with: $SCRIPT_DIR/server-tail-bridge-log.sh"
exit 1
