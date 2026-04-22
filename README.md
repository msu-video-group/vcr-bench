# VCR Bench

`vcr-bench` is a video-classifier robustness benchmark with configurable model loading, dataset subset resolution, and remote Slurm execution.

## Project layout

- `vcr_bench/`: Python package
- `configs/`: committed defaults, checkpoint manifests, dataset subset manifests, benchmark suites
- `scripts/`: Slurm launch scripts
- `docs/`: focused operational docs

## Installation

```bash
pip install -e .
```

CLI entrypoints:

- `vcr-bench-attack`
- `vcr-bench-artifacts`
- `vcr-bench-test`
- `vcr-bench-prepare`
- `vcr-bench-remote`

## Configuration

Committed defaults live in:

- `configs/defaults.toml`
- `configs/checkpoints.toml`
- `configs/datasets.toml`
- `configs/benchmarks.toml`

Create `configs/local.toml` from `configs/local.toml.example` to override local paths, remote hosts, and Slurm settings.

## JSON presets

Configurable model, attack, defence, and run presets live under `configs/models/`, `configs/attacks/`, `configs/defences/`, and `configs/runs/`.

Examples:

```bash
vcr-bench-test --run-preset accuracy_amd_100 --print-resolved-preset
vcr-bench-attack --run-preset attack_x3d_ifgsm_debug --override attack.params.steps=3
vcr-bench-attack --attack-preset ifgsm --model-preset amd --dataset kinetics400 --lite-attack
vcr-bench-remote --remote main launch-preset --preset accuracy_amd_100 --dry-run
```

Preset overrides use dotted keys and JSON values:

```bash
--override attack.params.query_budget=2000
--override runtime.device=\"cpu\"
```

## Model checkpoints

Checkpoint resolution now prefers the Hugging Face manifest in `configs/checkpoints.toml`. Explicit `--checkpoint` still overrides manifest resolution.

Useful artifact commands:

```bash
vcr-bench-artifacts list-checkpoints --model x3d
vcr-bench-artifacts build-dataset-archive --dataset kinetics400 --subset kinetics400_mini_val --source-dir /path/to/subset
```

## Dataset subsets

Named dataset subsets are declared in `configs/datasets.toml`. Example:

```bash
vcr-bench-test --model x3d --dataset kinetics400 --dataset-subset kinetics400_mini_val --num-videos 8
```

Dataset subset archives are downloaded from Hugging Face, cached locally, then extracted into the configured data directory.

## VRAM profiling

Accuracy and attack CLIs can append a separate one-video CUDA memory profile CSV:

```bash
vcr-bench-attack \
  --attack-preset ifgsm \
  --model-preset amd \
  --dataset kinetics400 \
  --num-videos 1 \
  --lite-attack \
  --vram-profile-csv results/vram/model_vram.csv
```

The profiler writes one row for `no_grad_forward` and one row for `with_grad_forward_backward`, including peak allocated/reserved VRAM and deltas over the baseline before that pass.

## Remote execution

Remote execution targets `SSH + Slurm` and reads remote definitions from TOML config.

Examples:

```bash
vcr-bench-remote sync-remote --remote main --dry-run
vcr-bench-remote launch-accuracy-suite --remote main --suite default --dry-run
vcr-bench-remote launch-attack-suite --remote main --suite default --dry-run
```

More detail: [docs/REMOTE_DEBUGGING.md](docs/REMOTE_DEBUGGING.md)
