#!/bin/bash
# vcr_bench Slurm batch accuracy launcher.
# Interface intentionally mirrors video_classifiers/methods/test_models.sh.

#SBATCH --job-name=test_vcr_bench
#SBATCH --output=logs/test_job_%j.log
#SBATCH --error=logs/test_job_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:8
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

start_id=0
num_videos=100
dataset="kinetics400"
specific_model=""
specific_models=()
file_name="accuracy"
folder_name=""
device="cuda"
batch_size=1
num_workers=0
video_root=""
annotations=""
labels=""
checkpoint=""
backbone=""
weights_dataset=""
grad_forward_chunk_size=""
dataset_loading_flag=0
instant_preprocessing_flag=0
full_video_flag=0
verbose_flag=0
split="val"
pipeline_stage="test"
results_root="results/accuracy"
save_csv_flag=0
dataset_subset=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -s|--start-id) start_id="$2"; shift ;;
        -v|--num-videos) num_videos="$2"; shift ;;
        -n|--file-name) file_name="$2"; shift ;;
        --folder-name) folder_name="$2"; shift ;;
        -d|--dataset) dataset="$2"; shift ;;
        -m|--model)
            if [[ "$2" == *","* ]]; then
                IFS=',' read -r -a MODEL_PAIRS <<< "$2"
                for pair in "${MODEL_PAIRS[@]}"; do
                    specific_models+=("$pair")
                done
            else
                specific_model="$2"
            fi
            shift
            ;;
        --device) device="$2"; shift ;;
        --batch-size) batch_size="$2"; shift ;;
        --num-workers) num_workers="$2"; shift ;;
        --video-root) video_root="$2"; shift ;;
        --annotations) annotations="$2"; shift ;;
        --labels) labels="$2"; shift ;;
        --dataset-subset) dataset_subset="$2"; shift ;;
        --checkpoint) checkpoint="$2"; shift ;;
        --backbone) backbone="$2"; shift ;;
        --weights-dataset) weights_dataset="$2"; shift ;;
        --grad-forward-chunk-size) grad_forward_chunk_size="$2"; shift ;;
        --dataset-loading) dataset_loading_flag=1 ;;
        --instant-preprocessing) instant_preprocessing_flag=1 ;;
        --full-video|--full-videos) full_video_flag=1 ;;
        --verbose) verbose_flag=1 ;;
        --split) split="$2"; shift ;;
        --pipeline-stage) pipeline_stage="$2"; shift ;;
        --results-root) results_root="$2"; shift ;;
        --save-csv) save_csv_flag=1 ;;
    esac
    shift
done

if [[ "$dataset" != "kinetics400" ]]; then
    echo "Error: vcr_bench/test_models.sh currently supports only dataset=kinetics400"
    exit 1
fi

source "${SCRIPT_DIR}/models_kinetics400.sh"

if [[ -n "$specific_model" || ${#specific_models[@]} -gt 0 ]]; then
    if [[ -n "$specific_model" ]]; then
        method_names_local=("$specific_model")
    else
        method_names_local=("${specific_models[@]}")
    fi
else
    method_names_local=("${method_names[@]:$start_id}")
fi

mkdir -p logs

run_root="${results_root}/${file_name}"
[[ -n "$folder_name" ]] && run_root="${results_root}/${folder_name}"
mkdir -p "${run_root}/logs"

for current_model in "${method_names_local[@]}"; do
    echo "Testing model: ${current_model}"

    json_path="${run_root}/${current_model}.json"
    csv_path="${run_root}/${current_model}.csv"
    log_file="${run_root}/logs/${current_model}.log"

    cmd=(
        python3 -m vcr_bench.cli.test
        --model "$current_model"
        --dataset "$dataset"
        --num-videos "$num_videos"
        --device "$device"
        --batch-size "$batch_size"
        --num-workers "$num_workers"
        --split "$split"
        --pipeline-stage "$pipeline_stage"
    )
    if [[ "$save_csv_flag" -eq 1 ]]; then
        cmd+=(--output-csv "$csv_path")
    else
        cmd+=(--output-json "$json_path")
    fi
    [[ -n "$video_root" ]] && cmd+=(--video-root "$video_root")
    [[ -n "$annotations" ]] && cmd+=(--annotations "$annotations")
    [[ -n "$labels" ]] && cmd+=(--labels "$labels")
    [[ -n "$dataset_subset" ]] && cmd+=(--dataset-subset "$dataset_subset")
    [[ -n "$checkpoint" ]] && cmd+=(--checkpoint "$checkpoint")
    [[ -n "$backbone" ]] && cmd+=(--backbone "$backbone")
    [[ -n "$weights_dataset" ]] && cmd+=(--weights-dataset "$weights_dataset")
    [[ -n "$grad_forward_chunk_size" ]] && cmd+=(--grad-forward-chunk-size "$grad_forward_chunk_size")
    [[ "$dataset_loading_flag" -eq 1 ]] && cmd+=(--dataset-loading)
    [[ "$instant_preprocessing_flag" -eq 1 ]] && cmd+=(--instant-preprocessing)
    [[ "$full_video_flag" -eq 1 ]] && cmd+=(--full-videos)
    [[ "$verbose_flag" -eq 1 ]] && cmd+=(--verbose)

    printf -v cmd_str '%q ' "${cmd[@]}"

    srun_args=(--exclusive --ntasks 1 -G 1)
    [[ -n "${CONTAINER_IMAGE}" ]] && srun_args+=(--container-image "${CONTAINER_IMAGE}")
    [[ -n "${CONTAINER_MOUNTS}" ]] && srun_args+=(--container-mounts "${CONTAINER_MOUNTS}")
    srun "${srun_args[@]}" bash -lc "cd /work && ${cmd_str}" >> "${log_file}" 2>&1 &
done

wait
