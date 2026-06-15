#!/usr/bin/env bash
set -euo pipefail

WEIGHTS_ROOT="${WONDERBOT_WEIGHTS_ROOT:-/srv/weights/wonderbot}"
STATE_ROOT="${WONDERBOT_STATE_ROOT:-$WEIGHTS_ROOT/state}"
OFFLOAD_ROOT="${WONDERBOT_OFFLOAD_ROOT:-$WEIGHTS_ROOT/offload}"
HF_CACHE_ROOT="${WONDERBOT_HF_CACHE:-$WEIGHTS_ROOT/hf-cache}"
WORKSPACE_ROOT="${WONDERBOT_WORKSPACE:-$WEIGHTS_ROOT/workspace}"
PROFILE="${WONDERBOT_PROFILE:-/app/configs/profiles/dual-p40-server-qwen14b.toml}"

mkdir -p "$WEIGHTS_ROOT" "$STATE_ROOT" "$OFFLOAD_ROOT" "$HF_CACHE_ROOT" "$WORKSPACE_ROOT"

export HF_HOME="$HF_CACHE_ROOT"
export TRANSFORMERS_CACHE="$HF_CACHE_ROOT"
export BITSANDBYTES_NOWELCOME=1

# Seed empty state files if the mounted state root is brand new.
for path in   "$STATE_ROOT/memory.json"   "$STATE_ROOT/journal.json"   "$STATE_ROOT/long_term_memory.json"   "$STATE_ROOT/self_model.json"   "$STATE_ROOT/goals.json"   "$STATE_ROOT/plans.json"   "$STATE_ROOT/action_runs.json"; do
  if [ ! -f "$path" ]; then
    case "$(basename "$path")" in
      self_model.json)
        printf '{"entries": []}
' > "$path"
        ;;
      goals.json|plans.json|action_runs.json|memory.json|journal.json|long_term_memory.json)
        printf '[]
' > "$path"
        ;;
    esac
  fi
done

if [ ! -f "$STATE_ROOT/replay.jsonl" ]; then
  : > "$STATE_ROOT/replay.jsonl"
fi

if [ "$#" -eq 0 ]; then
  set -- wonderbot --config "$PROFILE"
  EXTRA_ARGS="${WONDERBOT_EXTRA_ARGS:---diagnostics}"
  if [ -n "$EXTRA_ARGS" ]; then
    # shellcheck disable=SC2206
    EXTRA_ARR=( $EXTRA_ARGS )
    set -- "$@" "${EXTRA_ARR[@]}"
  fi
fi

exec "$@"
