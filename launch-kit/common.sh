#!/usr/bin/env bash
set -Eeuo pipefail

WB_REPO="${WONDERBOT_REPO:-/srv/wonderbot-v10_8-remote-av-bridge}"
WB_VENV="${WONDERBOT_VENV:-/home/wofl/.venvs/wb-bridge}"
WB_ENV_FILE="${WONDERBOT_ENV_FILE:-/srv/wonderbot-env.sh}"
WB_PROFILE="${WONDERBOT_PROFILE:-$WB_REPO/configs/profiles/dual-p40-server-qwen14b-remote-bridge-fp16-stt-tiny-cpu-sensitive.toml}"

WB_BRIDGE_HOST="${WONDERBOT_BRIDGE_HOST:-0.0.0.0}"
WB_BRIDGE_PORT="${WONDERBOT_BRIDGE_PORT_NUM:-8765}"
WB_BACKEND="${WONDERBOT_BACKEND:-hf}"
WB_HF_DEVICE_MAP="${WONDERBOT_HF_DEVICE_MAP:-auto}"
WB_TTS_DEVICE="${WONDERBOT_TTS_DEVICE:-cpu}"

WB_STATE_DIR="$WB_REPO/.launch"
WB_PID_DIR="$WB_STATE_DIR/pids"
WB_LOG_DIR="$WB_STATE_DIR/logs"

mkdir -p "$WB_PID_DIR" "$WB_LOG_DIR"

wb_env() {
  cd "$WB_REPO"

  if [[ ! -d "$WB_VENV" ]]; then
    echo "ERROR: venv not found: $WB_VENV" >&2
    exit 1
  fi

  # shellcheck disable=SC1091
  source "$WB_VENV/bin/activate"

  if [[ -f "$WB_ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$WB_ENV_FILE"
  fi

  export PYTHONUNBUFFERED=1

  WB_BRIDGE_TOKEN_EFFECTIVE="${WONDERBOT_BRIDGE_TOKEN_VALUE:-${WONDERBOT_BRIDGE_TOKEN:-change-me}}"
  export WB_BRIDGE_TOKEN_EFFECTIVE
}

wb_is_pid_alive() {
  local pid_file="$1"
  [[ -f "$pid_file" ]] || return 1
  local pid
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

wb_health_url() {
  echo "http://127.0.0.1:${WB_BRIDGE_PORT}/health"
}
