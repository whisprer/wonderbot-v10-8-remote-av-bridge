#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

PID_FILE="$WB_PID_DIR/bridge-server.pid"

if ! [[ -f "$PID_FILE" ]]; then
  echo "No bridge PID file found."
  exit 0
fi

PID="$(cat "$PID_FILE" 2>/dev/null || true)"

if [[ -z "$PID" ]]; then
  rm -f "$PID_FILE"
  echo "Empty bridge PID file removed."
  exit 0
fi

if kill -0 "$PID" 2>/dev/null; then
  echo "Stopping bridge server PID $PID..."
  kill "$PID" 2>/dev/null || true

  for i in {1..20}; do
    if ! kill -0 "$PID" 2>/dev/null; then
      rm -f "$PID_FILE"
      echo "Bridge server stopped."
      exit 0
    fi
    sleep 0.25
  done

  echo "Bridge server still alive; sending SIGKILL..."
  kill -9 "$PID" 2>/dev/null || true
else
  echo "Bridge PID $PID is not running."
fi

rm -f "$PID_FILE"
echo "Bridge server stopped/cleaned."
