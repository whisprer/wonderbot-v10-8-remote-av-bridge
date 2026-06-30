#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Stopping managed WonderBot server-side background processes..."
"$SCRIPT_DIR/server-stop-bridge.sh"

echo
echo "Note: if WonderBot CLI is open interactively, type /quit in that CLI terminal."
