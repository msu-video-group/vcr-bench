# VCR-Bench

A benchmark for adversarial robustness of video classification models — run attacks, measure defences, profile VRAM, and orchestrate jobs on remote Slurm clusters.

---

## Quick start

```bash
# 1. Install
pip install -e .

# 2. Pull the model checkpoint and a mini dataset subset
vcr-bench-artifacts list-checkpoints --model x3d

# 3. Run your first attack (8 videos, IFGSM on X3D)
vcr-bench-attack --model x3d --attack ifgsm --dataset kinetics400 \
  --dataset-subset kinetics400_mini_val --num-videos 8
```

Results land in `results/` by default.

---

## CLI overview

| Command | Purpose |
|---|---|
| `vcr-bench-attack` | Run an adversarial attack against a model |
| `vcr-bench-test` | Measure clean accuracy (baseline) |
| `vcr-bench-artifacts` | Download / inspect checkpoints and dataset archives |
| `vcr-bench-prepare` | Prepare video data and model inputs |
| `vcr-bench-remote` | Launch and monitor jobs on a remote Slurm cluster |

### `vcr-bench-attack` key flags

```
--model <name>                 Model to attack (x3d, timesformer, videomae, …)
--attack <name>                Attack method (ifgsm, mifgsm, square, gradest, …)
--dataset kinetics400          Dataset name
--dataset-subset <name>        Named subset from configs/datasets.toml
--num-videos <n>               How many videos to process  [default: 25]
--eps <float>                  L∞ perturbation bound       [default: 8.0]
--alpha <float>                Step size                   [default: 1.0]
--iter <int>                   Gradient steps              [default: 10]
--defence <name>               Wrap model with a defence (gaussian_blur, temporal_median, …)
--adaptive                     Apply defence before the attack (adaptive evaluation)
--device cuda|cpu              Compute device              [default: cuda]
--results-root <dir>           Output directory            [default: results/]
--output-json <file>           Write a JSON summary
--vram-profile-csv <file>      Append VRAM profiling row
--lite-attack                  Optimised single-pass pipeline

# Preset shortcuts
--run-preset <name>            Load a complete run preset from configs/runs/
--model-preset <name>          Load model config from configs/models/
--attack-preset <name>         Load attack config from configs/attacks/
--defence-preset <name>        Load defence config from configs/defences/
--override key=value           Override any preset field (dotted key, JSON value)
```

### Preset examples

```bash
# Full preset
vcr-bench-attack --run-preset attack_x3d_ifgsm_debug --override attack.params.steps=3

# Mix presets
vcr-bench-attack --attack-preset ifgsm --model-preset amd --dataset kinetics400 --lite-attack

# Print the resolved config without running
vcr-bench-test --run-preset accuracy_amd_100 --print-resolved-preset
```

---

## System requirements

Python 3.10+, PyTorch with CUDA recommended.

The table below will be filled in once VRAM profiling data is collected for each model. Use `--vram-profile-csv` to contribute measurements.

| Model | Backbone | Inference VRAM | Attack VRAM (IFGSM, 10 steps) | Min GPU |
|---|---|---|---|---|
| X3D | — | — | — | — |
| TimeSformer | — | — | — | — |
| VideoMAE | — | — | — | — |
| SlowFast | — | — | — | — |
| SlowOnly | — | — | — | — |
| MViTv2 | — | — | — | — |
| VideoSwin | — | — | — | — |
| UniFormer | — | — | — | — |
| UniFormerV2 | — | — | — | — |
| InternVideo | — | — | — | — |
| InternVideo2 | — | — | — | — |
| ActionCLIP | — | — | — | — |
| C3D | — | — | — | — |
| I3D | — | — | — | — |
| TSM | — | — | — | — |
| TSN | — | — | — | — |

Use `--grad-forward-chunk-size <n>` to reduce VRAM on memory-constrained GPUs.

---

## Usage in code

```python
from vcr_bench.models import create_model
from vcr_bench.datasets import create_dataset
from vcr_bench.attacks import create_attack
from vcr_bench.utils.eval import run_attack

model = create_model("x3d", device="cuda")

dataset = create_dataset(
    "kinetics400",
    video_root="data/kinetics400/k400_val",
)

attack = create_attack("ifgsm", eps=8.0, alpha=1.0, steps=10)

run_attack(
    model=model,
    attack=attack,
    dataset=dataset,
    attack_name="ifgsm_untargeted",
    num_videos=25,
    seed=42,
    target=False,
    save_path="results/attack_results.csv",
    log_path="results/attack_log.csv",
)
```

**Adding a defence:**

```python
from vcr_bench.defences import create_defence

defence = create_defence("gaussian_blur", sigma=1.0)
defence.install(model)   # wraps model preprocessing

run_attack(model=model, attack=attack, dataset=dataset, ...)

defence.uninstall(model) # restore original model
```

---

## Configuration

```
configs/
  defaults.toml       # paths, remote hosts, Slurm settings
  checkpoints.toml    # HuggingFace checkpoint manifest
  datasets.toml       # named dataset subsets
  benchmarks.toml     # curated accuracy / attack suites
  local.toml.example  # copy to local.toml and edit
  models/             # per-model JSON presets
  attacks/            # per-attack JSON presets
  defences/           # per-defence JSON presets
  runs/               # complete run presets
```

Copy `configs/local.toml.example` to `configs/local.toml` and set your local data paths, remote SSH target, and Slurm partition.

---

## Docs

- [Data formats & pipeline conventions](docs/formats.md)
- [Adding a model](docs/adding_models.md)
- [Adding a dataset](docs/adding_datasets.md)
- [Adding an attack](docs/adding_attacks.md)
- [Adding a defence](docs/adding_defences.md)
- [Remote execution & Slurm guide](docs/REMOTE_DEBUGGING.md)

### Remote execution quick reference

```bash
# 1. Push local changes to the remote
vcr-bench-remote push-code --commit-message "my change"

# 2. Sync the remote checkout
vcr-bench-remote sync-remote --remote main

# 3. Launch a single focused job
vcr-bench-remote launch-job --remote main --mode attack \
  --model x3d --attack ifgsm --dataset kinetics400 \
  --dataset-subset kinetics400_mini_val --num-videos 8 \
  --attack-name debug_ifgsm

# 4. Launch the full benchmark suites
vcr-bench-remote launch-accuracy-suite --remote main --suite default
vcr-bench-remote launch-attack-suite   --remote main --suite default

# 5. Check status and fetch results
vcr-bench-remote job-status          --remote main --job-id 123456
vcr-bench-remote fetch-job-artifacts --remote main --job-id 123456 \
  --attack-name default_suite
```

Add `--dry-run` to any command to print what would be executed without contacting the cluster.

### VRAM profiling

```bash
vcr-bench-attack \
  --attack-preset ifgsm --model-preset amd \
  --dataset kinetics400 --num-videos 1 --lite-attack \
  --vram-profile-csv results/vram/x3d_ifgsm.csv
```

The CSV gets one row for `no_grad_forward` and one for `with_grad_forward_backward`, with peak allocated/reserved VRAM and deltas over baseline.

### Artifact management

```bash
# List available checkpoints for a model
vcr-bench-artifacts list-checkpoints --model x3d

# Build a dataset subset archive
vcr-bench-artifacts build-dataset-archive \
  --dataset kinetics400 --subset kinetics400_mini_val \
  --source-dir /path/to/subset
```

---

## Contributing

1. Fork the repo and create a feature branch off `master`.
2. Install in editable mode: `pip install -e ".[dev]"`.
3. Run the test suite: `pytest`.
4. Keep changes focused — one feature or fix per PR, no unrelated refactors.
5. If you add a model, attack, or defence:
   - Subclass the corresponding base in `vcr_bench/models/`, `vcr_bench/attacks/`, or `vcr_bench/defences/`.
   - Add a JSON preset under `configs/models/`, `configs/attacks/`, or `configs/defences/`.
   - If you have VRAM numbers, fill in the system requirements table above.
6. Open a PR with a short description of what changed and why.
