#!/usr/bin/env bash
set -euo pipefail

cd /srv/wonderbot-v10_8-remote-av-bridge

source /home/wofl/.venvs/wb-bridge/bin/activate
source /srv/wonderbot-env.sh

exec python -m wonderbot.cli \
  --config /srv/wonderbot-v10_8-remote-av-bridge/configs/profiles/dual-p40-server-qwen14b-remote-bridge-fp16-stt-base-cpu-sensitive.toml \
  --backend hf \
  --hf-device-map auto \
  --diagnostics \
  --live-lite \
  --live-lite-cycles "${WONDERBOT_LIVE_LITE_CYCLES:-forever}" \
  --live-lite-interval "${WONDERBOT_LIVE_LITE_INTERVAL:-2}" \
  --live-lite-cooldown "${WONDERBOT_LIVE_LITE_COOLDOWN:-8}"
