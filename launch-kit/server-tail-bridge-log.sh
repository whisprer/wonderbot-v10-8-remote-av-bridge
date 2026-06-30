#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

LOG_FILE="$WB_LOG_DIR/bridge-server.log"
touch "$LOG_FILE"
tail -n 80 -f "$LOG_FILE"
