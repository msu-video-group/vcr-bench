#!/bin/bash
set -e
CONFIG_PATH="scripts/autolaunch/configs/main.json"
SERVICE_EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --test-mode|--skip-local-git)
      SERVICE_EXTRA_ARGS+=("$1")
      shift
      ;;
    --commit-message|--window-name)
      SERVICE_EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    *)
      if [[ "$1" == -* ]]; then
        SERVICE_EXTRA_ARGS+=("$1")
      else
        CONFIG_PATH="$1"
      fi
      shift
      ;;
  esac
done

python3 scripts/autolaunch/launch_remote.py --config "$CONFIG_PATH" "${SERVICE_EXTRA_ARGS[@]}"
