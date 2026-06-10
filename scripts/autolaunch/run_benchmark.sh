#!/bin/bash
# Launch the autolaunch service for a chosen benchmark preset.
#
# Usage:
#   ./scripts/autolaunch/run_benchmark.sh [preset]
#
#   preset = a name under scripts/autolaunch/configs/presets/ (without .json),
#            or a path to any tracks-config JSON. Default: full_benchmark.
#
# Examples:
#   ./scripts/autolaunch/run_benchmark.sh quick            # fast smoke slice
#   ./scripts/autolaunch/run_benchmark.sh full_white_box   # white-box only
#   ./scripts/autolaunch/run_benchmark.sh full_blackbox    # black-box only
#   ./scripts/autolaunch/run_benchmark.sh full_benchmark   # everything (default)
#
# The service skips cells whose results already exist, so re-running fills gaps.
# To reproduce a track from scratch, clear results/remote_attacks/<attack_name>
# (or point the preset's tracks at a fresh results_root) before launching.
set -euo pipefail
cd "$HOME/users/29m_pli/vcr-bench" || exit 1

PRESET="${1:-full_benchmark}"
CONFIG_DIR="scripts/autolaunch/configs"

if [[ -f "$PRESET" ]]; then
  TRACKS="$PRESET"
elif [[ -f "$CONFIG_DIR/presets/$PRESET.json" ]]; then
  TRACKS="$CONFIG_DIR/presets/$PRESET.json"
else
  echo "Unknown preset: $PRESET" >&2
  echo "Available presets:" >&2
  ls "$CONFIG_DIR/presets"/*.json 2>/dev/null | sed 's#.*/##; s#\.json$##' >&2
  exit 1
fi

# Container env so submitted sbatch jobs inherit the image/mounts.
export VCR_BENCH_CONTAINER_IMAGE="$HOME/users/29m_pli/python.sqsh"
export VCR_BENCH_CONTAINER_MOUNTS="$HOME/users/29m_pli/vcr-bench:/work,$HOME/.msu_vqmt:/root/.msu_vqmt"
# Cap per-task memory so 1-GPU array tasks pack onto a shared node.
export SBATCH_MEM_PER_CPU=15000M
mkdir -p scripts/autolaunch/logs scripts/autolaunch/runtime

# Derive an ephemeral main config from main.json that points at the chosen preset,
# so we keep all scheduler/slurm settings but swap the tracks list.
MAIN_OVERRIDE="scripts/autolaunch/runtime/main_${PRESET//\//_}.json"
python3 - "$CONFIG_DIR/main.json" "$TRACKS" "$MAIN_OVERRIDE" <<'PY'
import json, sys
base, tracks, out = sys.argv[1:4]
cfg = json.load(open(base))
cfg["tracks_config"] = tracks
json.dump(cfg, open(out, "w"), indent=2)
print(f"[run_benchmark] preset tracks_config={tracks} -> {out}")
PY

echo "[run_benchmark] launching service for preset '$PRESET' (tracks: $TRACKS)"
exec python3 scripts/autolaunch/service.py \
  --config "$MAIN_OVERRIDE" \
  >> scripts/autolaunch/logs/autolaunch.log 2>&1
