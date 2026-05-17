#!/bin/bash
set -euo pipefail

CONFIG_PATH=${1:-scripts/autolaunch/configs/main.json}

python3 scripts/autolaunch/stop_remote.py \
  --config "$CONFIG_PATH" \
  --kill-session
