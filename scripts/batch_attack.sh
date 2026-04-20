#!/bin/bash
# vcr_bench Slurm batch attack launcher.
# Interface intentionally mirrors video_classifiers/methods/batch_attack_flexible.sh.

#SBATCH --job-name=attack_vcr_bench
#SBATCH --output=logs/job_%j.log
#SBATCH --error=logs/job_%j.err
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

echo "======================================="
echo "Job started on: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node name: ${SLURM_JOB_NODELIST:-local}"
echo "Allocated nodes: ${SLURM_NODELIST:-local}"
echo "======================================="

attack_type=""
attack_name=""
comment=""
defence_name=""
dataset="kinetics400"
start_id=0
target_flag=0
vmaf_flag=0
num_videos=100
specific_model=""
specific_models=()
eps=""
iter=""
alpha=""
dump_freq=0
framewise_metrics_flag=0
adaptive_flag=0
full_video_flag=0
debug_flag=0
extended_debug_flag=0
visualize_defence_flag=0
save_defence_stages_flag=0
allow_misclassified_flag=0
separate_logs_flag=0
device="cuda"
batch_size=1
num_workers=0
verbose_flag=0
results_root="results"
video_root=""
annotations=""
labels=""
dataset_subset=""
checkpoint=""
backbone=""
weights_dataset=""
grad_forward_chunk_size=""
attack_sample_chunk_size=""
lite_attack_flag=0
pipeline_stage="test"
split="val"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -a|--attack-type) attack_type="$2"; shift ;;
        -n|--attack-name) attack_name="$2"; shift ;;
        -c|--comment) comment="$2"; shift ;;
        -def|--defence) defence_name="$2"; shift ;;
        -d|--dataset) dataset="$2"; shift ;;
        -s|--start-id) start_id="$2"; shift ;;
        -t|--target) target_flag=1 ;;
        --vmaf) vmaf_flag=1 ;;
        -f|--framewise-metrics) framewise_metrics_flag=1 ;;
        --adaptive) adaptive_flag=1 ;;
        --full-video|--full-videos) full_video_flag=1 ;;
        --visualize-defence) visualize_defence_flag=1 ;;
        --save-defence-stages) save_defence_stages_flag=1 ;;
        --allow-misclassified) allow_misclassified_flag=1 ;;
        --separate-logs) separate_logs_flag=1 ;;
        --verbose) verbose_flag=1 ;;
        --debug) debug_flag=1 ;;
        --extended-debug) extended_debug_flag=1 ;;
        -v|--num-videos) num_videos="$2"; shift ;;
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
        --eps) eps="$2"; shift ;;
        --alpha) alpha="$2"; shift ;;
        --dump-freq) dump_freq="$2"; shift ;;
        --iter) iter="$2"; shift ;;
        --device) device="$2"; shift ;;
        --batch-size) batch_size="$2"; shift ;;
        --num-workers) num_workers="$2"; shift ;;
        --results-root) results_root="$2"; shift ;;
        --video-root) video_root="$2"; shift ;;
        --annotations) annotations="$2"; shift ;;
        --labels) labels="$2"; shift ;;
        --dataset-subset) dataset_subset="$2"; shift ;;
        --checkpoint) checkpoint="$2"; shift ;;
        --backbone) backbone="$2"; shift ;;
        --weights-dataset) weights_dataset="$2"; shift ;;
        --grad-forward-chunk-size) grad_forward_chunk_size="$2"; shift ;;
        --attack-sample-chunk-size) attack_sample_chunk_size="$2"; shift ;;
        --lite-attack) lite_attack_flag=1 ;;
        --pipeline-stage) pipeline_stage="$2"; shift ;;
        --split) split="$2"; shift ;;
        --testing)
            echo "Error: testing mode belongs to vcr_bench/cli/test.py, not batch_attack.sh"
            exit 1
            ;;
    esac
    shift
done

if [[ -z "$attack_type" ]]; then
    echo "Error: --attack-type is required"
    exit 1
fi
if [[ -z "$attack_name" ]]; then
    echo "Error: --attack-name is required"
    exit 1
fi

models_file="${SCRIPT_DIR}/models_${dataset}.sh"
if [[ ! -f "$models_file" ]]; then
    echo "Error: unsupported dataset '${dataset}' for vcr_bench batch launcher: ${models_file} not found"
    exit 1
fi
source "$models_file"

method_names_local=()
attack_types_local=()
defence_names_local=()
adaptive_flags_local=()
target_flags_local=()

if [[ -n "$specific_model" || ${#specific_models[@]} -gt 0 ]]; then
    if [[ -n "$specific_model" ]]; then
        IFS=':' read -r -a MODEL_INFO <<< "$specific_model"
        _sm_name="${MODEL_INFO[0]}"
        _sm_attack="${MODEL_INFO[1]:-$attack_type}"
        _sm_target_raw="${MODEL_INFO[2]:-}"
        _sm_defence="${MODEL_INFO[3]:-$defence_name}"
        _sm_adaptive_raw="${MODEL_INFO[4]:-}"
        if [[ -z "$_sm_target_raw" ]]; then _sm_target="$target_flag"
        else case "${_sm_target_raw,,}" in 1|true|yes|target) _sm_target=1 ;; *) _sm_target=0 ;; esac; fi
        if [[ -z "$_sm_adaptive_raw" ]]; then _sm_adaptive="$adaptive_flag"
        else case "${_sm_adaptive_raw,,}" in 1|true|yes|adaptive) _sm_adaptive=1 ;; *) _sm_adaptive=0 ;; esac; fi
        method_names_local=("$_sm_name")
        attack_types_local=("$_sm_attack")
        target_flags_local=("$_sm_target")
        defence_names_local=("$_sm_defence")
        adaptive_flags_local=("$_sm_adaptive")
    else
        for model_pair in "${specific_models[@]}"; do
            IFS=':' read -r -a MODEL_INFO <<< "$model_pair"
            model_name="${MODEL_INFO[0]}"
            model_attack="${MODEL_INFO[1]:-$attack_type}"
            model_target="${MODEL_INFO[2]:-}"
            model_defence="${MODEL_INFO[3]:-$defence_name}"
            model_adaptive="${MODEL_INFO[4]:-}"
            if [[ -z "$model_target" ]]; then
                model_target="$target_flag"
            else
                case "${model_target,,}" in
                    1|true|yes|target) model_target=1 ;;
                    0|false|no|untarget) model_target=0 ;;
                    *) model_target="$target_flag" ;;
                esac
            fi
            if [[ -z "$model_adaptive" ]]; then
                model_adaptive="$adaptive_flag"
            else
                case "${model_adaptive,,}" in
                    1|true|yes|adaptive) model_adaptive=1 ;;
                    0|false|no|non-adaptive) model_adaptive=0 ;;
                    *) model_adaptive="$adaptive_flag" ;;
                esac
            fi
            method_names_local+=("$model_name")
            attack_types_local+=("$model_attack")
            target_flags_local+=("$model_target")
            defence_names_local+=("$model_defence")
            adaptive_flags_local+=("$model_adaptive")
        done
    fi
else
    method_names_local=("${method_names[@]:$start_id}")
    for ((i=0; i<${#method_names_local[@]}; i++)); do
        attack_types_local+=("$attack_type")
        target_flags_local+=("$target_flag")
        defence_names_local+=("$defence_name")
        adaptive_flags_local+=("$adaptive_flag")
    done
fi

if [[ "$visualize_defence_flag" -eq 1 ]]; then
    echo "Warning: --visualize-defence is accepted for compatibility but is not forwarded by vcr_bench CLI"
fi

mkdir -p logs

for ((i=0; i<${#method_names_local[@]}; i++)); do
    current_model="${method_names_local[$i]}"
    current_attack_type="${attack_types_local[$i]}"
    current_target_flag="${target_flags_local[$i]}"
    current_defence_name="${defence_names_local[$i]}"
    current_adaptive_flag="${adaptive_flags_local[$i]}"

    echo "Method: $current_model"
    echo "Attack: $current_attack_type"

    attack_type_label="untarget"
    [[ "$current_target_flag" -eq 1 ]] && attack_type_label="target"
    defence_label="${current_defence_name:-no_defence}"
    defence_type_label="non-adaptive"
    [[ "$current_adaptive_flag" -eq 1 ]] && defence_type_label="adaptive"
    comment_label="${comment// /_}"
    comment_label="${comment_label//\//_}"
    log_subdir="${current_attack_type}_${attack_type_label}_${defence_label}_${defence_type_label}"
    [[ -n "$comment_label" ]] && log_subdir="${log_subdir}_${comment_label}"
    log_dir="${results_root}/${attack_name}/${log_subdir}"
    mkdir -p "$log_dir"
    log_file="${log_dir}/${current_model}.log"
    : > "$log_file"

    cmd=(
        python3 -m vcr_bench.cli.attack
        --attack-name "$attack_name"
        --attack "$current_attack_type"
        --dataset "$dataset"
        --num-videos "$num_videos"
        --model "$current_model"
        --device "$device"
        --batch-size "$batch_size"
        --num-workers "$num_workers"
        --results-root "$results_root"
        --split "$split"
        --pipeline-stage "$pipeline_stage"
        --dump-freq "$dump_freq"
    )
    [[ -n "$eps" ]] && cmd+=(--eps "$eps")
    [[ -n "$alpha" ]] && cmd+=(--alpha "$alpha")
    [[ -n "$iter" ]] && cmd+=(--iter "$iter")
    [[ -n "$attack_sample_chunk_size" ]] && cmd+=(--attack-sample-chunk-size "$attack_sample_chunk_size")
    [[ "$current_target_flag" -eq 1 ]] && cmd+=(--target)
    if [[ "$vmaf_flag" -eq 1 ]]; then
        cmd+=(--vmaf)
    else
        cmd+=(--no-vmaf)
    fi
    [[ "$framewise_metrics_flag" -eq 1 ]] && cmd+=(--framewise-metrics)
    [[ "$full_video_flag" -eq 1 ]] && cmd+=(--full-videos)
    [[ "$allow_misclassified_flag" -eq 1 ]] && cmd+=(--allow-misclassified)
    [[ "$separate_logs_flag" -eq 1 ]] && cmd+=(--separate-logs)
    if [[ "$verbose_flag" -eq 1 || "$debug_flag" -eq 1 || "$extended_debug_flag" -eq 1 ]]; then
        cmd+=(--verbose)
    fi
    [[ -n "$current_defence_name" && "$current_defence_name" != "no_defence" ]] && cmd+=(--defence "$current_defence_name")
    [[ "$save_defence_stages_flag" -eq 1 ]] && cmd+=(--save-defence-stages)
    [[ -n "$comment" ]] && cmd+=(--comment "$comment")
    [[ "$current_adaptive_flag" -eq 1 ]] && cmd+=(--adaptive)
    [[ "$lite_attack_flag" -eq 1 ]] && cmd+=(--lite-attack)
    [[ -n "$video_root" ]] && cmd+=(--video-root "$video_root")
    [[ -n "$annotations" ]] && cmd+=(--annotations "$annotations")
    [[ -n "$labels" ]] && cmd+=(--labels "$labels")
    [[ -n "$dataset_subset" ]] && cmd+=(--dataset-subset "$dataset_subset")
    [[ -n "$checkpoint" ]] && cmd+=(--checkpoint "$checkpoint")
    [[ -n "$backbone" ]] && cmd+=(--backbone "$backbone")
    [[ -n "$weights_dataset" ]] && cmd+=(--weights-dataset "$weights_dataset")
    [[ -n "$grad_forward_chunk_size" ]] && cmd+=(--grad-forward-chunk-size "$grad_forward_chunk_size")

    printf -v cmd_str '%q ' "${cmd[@]}"

    srun_args=(--exclusive --ntasks 1 -G 1)
    [[ -n "${CONTAINER_IMAGE}" ]] && srun_args+=(--container-image "${CONTAINER_IMAGE}")
    [[ -n "${CONTAINER_MOUNTS}" ]] && srun_args+=(--container-mounts "${CONTAINER_MOUNTS}")
    srun "${srun_args[@]}" bash -lc "python3 -c \"import importlib.util, sys; mods=['transformers','diffusers','yacs','IQA_pytorch','pywt']; missing=[m for m in mods if importlib.util.find_spec(m) is None]; sys.exit(0 if not missing else 1)\" || pip install -q transformers diffusers yacs 'numpy<2' IQA_pytorch PyWavelets; cd /work && ${cmd_str}" >> "$log_file" 2>&1 &
done

wait
