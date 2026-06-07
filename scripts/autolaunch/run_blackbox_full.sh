#!/bin/bash
# Launches the autolaunch service for the blackbox_full sweep with the correct
# container env so submitted sbatch jobs inherit VCR_BENCH_CONTAINER_IMAGE/MOUNTS.
cd "$HOME/users/29m_pli/vcr-bench" || exit 1
export VCR_BENCH_CONTAINER_IMAGE="$HOME/users/29m_pli/python.sqsh"
export VCR_BENCH_CONTAINER_MOUNTS="$HOME/users/29m_pli/vcr-bench:/work,$HOME/.msu_vqmt:/root/.msu_vqmt"
# Cap per-task memory so 1-GPU array tasks pack onto a shared node instead of
# each reserving the whole node's RAM (8 GPUs * 240G = 1920G < 2062G node mem).
# Honored by sbatch as a default since neither CLI nor batch_attack.sh set --mem.
export SBATCH_MEM_PER_CPU=15000M
mkdir -p scripts/autolaunch/logs
exec python3 scripts/autolaunch/service.py \
  --config scripts/autolaunch/configs/main_blackbox_full.json \
  >> scripts/autolaunch/logs/blackbox_full.log 2>&1
