#!/bin/bash
set -euo pipefail

CONFIG_PATH=${1:-scripts/autolaunch/configs/main.json}

python3 scripts/autolaunch/stop_calculations.py \
  --config "$CONFIG_PATH" \
  --service-timeout-sec 1
