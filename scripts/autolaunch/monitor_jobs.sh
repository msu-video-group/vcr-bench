#!/bin/bash
set -euo pipefail

CONFIG_PATH=${1:-scripts/autolaunch/configs/main.json}
shift || true

python3 scripts/autolaunch/monitor_jobs.py --config "$CONFIG_PATH" "$@"
