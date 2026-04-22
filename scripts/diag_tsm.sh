#!/bin/bash
#SBATCH --job-name=diag_tsm
#SBATCH --output=logs/diag_tsm_%j.log
#SBATCH --error=logs/diag_tsm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"
CONTAINER_IMAGE="${VCR_BENCH_CONTAINER_IMAGE:-${SLURM_CONTAINER_IMAGE:-${HOME}/users/29m_pli/python.sqsh}}"
CONTAINER_MOUNTS="${VCR_BENCH_CONTAINER_MOUNTS:-${REPO_ROOT}:/work}"

mkdir -p logs
srun_args=()
[[ -n "${CONTAINER_IMAGE}" ]] && srun_args+=(--container-image "${CONTAINER_IMAGE}")
[[ -n "${CONTAINER_MOUNTS}" ]] && srun_args+=(--container-mounts "${CONTAINER_MOUNTS}")
srun "${srun_args[@]}" bash -lc "cd /work && python3 scripts/diag_tsm.py"
