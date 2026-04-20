#!/bin/bash
# vcr_bench Slurm offline preparation launcher.
# Runs model-independent attack preparation (style caching, BMTC mixer training, etc.)
# for one attack on one dataset. Writes a sentinel JSON to --output-json on success.

#SBATCH --job-name=prepare_vcr_bench
#SBATCH --output=logs/prepare_job_%j.log
#SBATCH --error=logs/prepare_job_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/scripts" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}/scripts"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${REPO_ROOT}"

CONTAINER_IMAGE="${VCR_BENCH_CONTAINER_IMAGE:-${SLURM_CONTAINER_IMAGE:-}}"
CONTAINER_MOUNTS="${VCR_BENCH_CONTAINER_MOUNTS:-${REPO_ROOT}:/work}"

echo "======================================="
echo "Prepare started on: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node name: ${SLURM_JOB_NODELIST:-local}"
echo "======================================="

attack=""
dataset="kinetics400"
num_videos=50
output_json=""
device="cuda"
eps=""
iter=""
bmtc_save_root=""
bmtc_num_epochs=""
style_steps=""
video_root=""
annotations=""
labels=""
dataset_subset=""
split="val"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --attack)        attack="$2";        shift ;;
        --dataset)       dataset="$2";       shift ;;
        --num-videos)    num_videos="$2";    shift ;;
        --output-json)   output_json="$2";   shift ;;
        --device)        device="$2";        shift ;;
        --eps)           eps="$2";           shift ;;
        --iter)          iter="$2";          shift ;;
        --bmtc-save-root)   bmtc_save_root="$2";   shift ;;
        --bmtc-num-epochs)  bmtc_num_epochs="$2";  shift ;;
        --style-steps)      style_steps="$2";      shift ;;
        --video-root)       video_root="$2";       shift ;;
        --annotations)      annotations="$2";      shift ;;
        --labels)           labels="$2";           shift ;;
        --dataset-subset)   dataset_subset="$2";   shift ;;
        --split)            split="$2";            shift ;;
    esac
    shift
done

if [[ -z "$attack" ]]; then
    echo "Error: --attack is required"
    exit 1
fi

cmd=(
    python3 -m vcr_bench.cli.prepare
    --attack "$attack"
    --dataset "$dataset"
    --num-videos "$num_videos"
    --device "$device"
)
[[ -n "$output_json" ]]      && cmd+=(--output-json "$output_json")
[[ -n "$eps" ]]              && cmd+=(--eps "$eps")
[[ -n "$iter" ]]             && cmd+=(--iter "$iter")
[[ -n "$bmtc_save_root" ]]   && cmd+=(--bmtc-save-root "$bmtc_save_root")
[[ -n "$bmtc_num_epochs" ]]  && cmd+=(--bmtc-num-epochs "$bmtc_num_epochs")
[[ -n "$style_steps" ]]      && cmd+=(--style-steps "$style_steps")
[[ -n "$video_root" ]]       && cmd+=(--video-root "$video_root")
[[ -n "$annotations" ]]      && cmd+=(--annotations "$annotations")
[[ -n "$labels" ]]           && cmd+=(--labels "$labels")
[[ -n "$dataset_subset" ]]   && cmd+=(--dataset-subset "$dataset_subset")
[[ -n "$split" ]]            && cmd+=(--split "$split")

mkdir -p logs
[[ -n "$output_json" ]] && mkdir -p "$(dirname "$output_json")"

printf -v cmd_str '%q ' "${cmd[@]}"

srun_args=(--exclusive --ntasks 1 -G 1)
[[ -n "${CONTAINER_IMAGE}" ]] && srun_args+=(--container-image "${CONTAINER_IMAGE}")
[[ -n "${CONTAINER_MOUNTS}" ]] && srun_args+=(--container-mounts "${CONTAINER_MOUNTS}")
srun "${srun_args[@]}" bash -lc "python3 -c \"import importlib.util, sys; mods=['transformers','diffusers','yacs','IQA_pytorch','pywt']; missing=[m for m in mods if importlib.util.find_spec(m) is None]; sys.exit(0 if not missing else 1)\" || pip install -q transformers diffusers yacs 'numpy<2' IQA_pytorch PyWavelets; cd /work && ${cmd_str}"

echo "Prepare finished on: $(date)"
