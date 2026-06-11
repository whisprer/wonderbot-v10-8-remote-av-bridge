#!/usr/bin/env bash
set -euo pipefail
if command -v python3.11 >/dev/null 2>&1; then
  python3.11 -m wonderbot.cli "$@"
else
  python3 -m wonderbot.cli "$@"
fi
