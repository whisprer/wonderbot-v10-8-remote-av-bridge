#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

wb_env

echo "Starting WonderBot CLI with TTS enabled."
echo "At the > prompt, use:"
echo "  /sensors"
echo "  /sense-watch forever 1 4"
echo
echo "Quit with:"
echo "  /quit"
echo

exec python -m wonderbot.cli \
  --config "$WB_PROFILE" \
  --backend "$WB_BACKEND" \
  --hf-device-map "$WB_HF_DEVICE_MAP" \
  --tts \
  --tts-device "$WB_TTS_DEVICE" \
  --diagnostics
