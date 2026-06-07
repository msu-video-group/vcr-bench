<div align="center">
  <img src="docs/assets/Logo1-cropped.svg" alt="VCR-Bench logo" width="320"/>
</div>

<br/>

<div align="center">

<h3>A modular open-source benchmark for video-classification robustness</h3>

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20recommended-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg?logo=ruff&logoColor=white)](https://github.com/astral-sh/ruff)

[![Models](https://img.shields.io/badge/models-30-1f6feb)](#models)
[![Attacks](https://img.shields.io/badge/attacks-14-8957e5)](#attacks)
[![Defences](https://img.shields.io/badge/defences-10-2da44e)](#defences)
[![Leaderboard](https://img.shields.io/badge/leaderboard-live-ff8800)](https://msu-video-group.github.io/vcr-bench/)
[![Venue](https://img.shields.io/badge/ACM%20MM-2026-b31b1b)](https://msu-video-group.github.io/vcr-bench/)

**[Leaderboard](https://msu-video-group.github.io/vcr-bench/)** ·
**[Quick start](#quick-start)** ·
**[Docs](#docs)** ·
**[Contributing](#contributing)**

</div>

---

**VCR-Bench** is a modular benchmark for evaluating the adversarial robustness of video classification models. It provides a unified pipeline for running white-box and black-box attacks, wrapping models with defences, measuring clean accuracy baselines, and collecting structured results — all through a single CLI or Python API.

The benchmark covers **30 video classification models**, **14 adversarial attacks** (gradient-based, query-based, and perceptual), and **10 defense wrappers**, making it straightforward to compare attack transferability, defence effectiveness, and model sensitivity under a common evaluation protocol.

<details>
<summary><b>Table of contents</b></summary>

- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [Supported components](#supported-components)
- [CLI overview](#cli-overview)
- [Usage in code](#usage-in-code)
- [Configuration](#configuration)
- [System requirements](#system-requirements)
- [Docs](#docs)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)
- [License](#license)

</details>

---

## How it works

<div align="center">
  <img src="docs/assets/scheme.svg" alt="VCR-Bench pipeline" width="700"/>
</div>

Every evaluation flows through a single standardized pipeline. A run starts from a configuration preset that fixes the dataset, the temporal sampling strategy, the perturbation budget, and the metric aggregation, so that any combination of components is measured under identical conditions. Videos are decoded once through a shared loader and handed to a classifier wrapper that exposes a common interface, which lets the same attack or defence operate unchanged across CNN, Transformer, and Vision-Language models.

For each clip the framework applies the selected adversarial attack against the wrapped model, optionally routing inputs through a defence wrapper. The resulting clean and adversarial predictions, together with perceptual-quality, runtime, and memory measurements, are logged in a uniform schema and aggregated into the clean accuracy, attack success rate, and robustness scores — making every reported number reproducible and directly comparable across models, attacks, and defences.

The `--adaptive` flag places the defence **before** the attack so that the adversary optimises through it, enabling adaptive evaluation.

---

## Quick start

> **Tip:** For an end-to-end walkthrough that installs all dependencies (including LPIPS/VMAF metric extras) and runs clean tests, attacks, and defences interactively, open [`getting_started.ipynb`](getting_started.ipynb).

```bash
# 1. Install
pip install -e .

# Optional: broader research dependencies (additional models, attacks, defences)
pip install -e ".[research]"

# 2. Pull a model checkpoint and a mini dataset subset
vcr-bench-artifacts list-checkpoints --model x3d

# 3. Run your first attack  (I-FGSM on X3D, CC BY demo videos)
vcr-bench-attack --model x3d --attack ifgsm --dataset kinetics400 \
  --dataset-subset demo --num-videos 8
```

Results land in `results/` by default. The `demo` subset auto-downloads from Hugging Face —
no manual dataset setup needed.

> **Datasets / videos.** VCR-Bench ships dataset *adapters* but **does not redistribute raw
> videos** for Kinetics-400, UCF-101 or SSv2. The `demo` subset above is a small,
> auto-downloaded set of **real Kinetics clips** (real labels) whose YouTube sources are
> **CC BY 3.0** — redistributable with attribution; for demos / smoke tests, not benchmark
> numbers (see [docs/demo_dataset.md](docs/demo_dataset.md)). To run on a real dataset,
> download it yourself and pass `--video-root` — see [docs/datasets.md](docs/datasets.md).

---

## Supported components

### Models

| Family | Models |
|---|---|
| CNN | C3D, C2D, CSN, R2+1D, I3D, I3D-NonLocal, SlowFast, SlowOnly |
| Transformer | TimeSformer, VideoMAE, VideoMAEv2, VideoSwin, MViTv2 |
| Unified | UniFormer, UniFormerV2, X3D, TAdaFormer, TANet, TIN, TPN, TRN, TSM, TSM-NonLocal, TSN |
| Vision-Language | ActionCLIP, InternVideo, InternVideo2, ONE-PEACE, AMD, UMT |

### Attacks

| Category | Methods |
|---|---|
| Gradient-based (L∞) | I-FGSM, MI-FGSM, AMI-FGSM, GradEst, GradEstV2 |
| Perceptual / spatial | StAdv, SSAH, StyleFool, Zhang-DISTS, Zhang-LPIPS, Zhang-SSIM, Flickering |
| Universal / transferable | UAP, BMTC, TENAD, Korhonen et al. |
| Query-based (black-box) | Square, Parsimonious |

### Defences

| Category | Methods |
|---|---|
| Spatial filtering | Gaussian Blur, Bilateral Filter, Domain Transform |
| Temporal | Temporal Median, Shuffle, Flip, Rotate |
| Compression / reconstruction | JPEG Compression, Crop+Resize, VideoPure, FreqPure |
| Stochastic | Randomized Smoothing |

---

## CLI overview

| Command | Purpose |
|---|---|
| `vcr-bench-attack` | Run an adversarial attack against a model |
| `vcr-bench-test` | Measure clean accuracy (baseline) |
| `vcr-bench-classify` | Classify an arbitrary folder of videos and save predictions |
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
--vram-profile-csv <file>      Append VRAM profiling rows, including attack peak
--lite-attack                  Optimised single-pass pipeline

# Preset shortcuts
--run-preset <name>            Load a complete run preset from configs/runs/
--model-preset <name>          Load model config from configs/models/
--attack-preset <name>         Load attack config from configs/attacks/
--defence-preset <name>        Load defence config from configs/defences/
--override key=value           Override any preset field (dotted key, JSON value)
```

### `vcr-bench-classify` example

```bash
vcr-bench-classify \
  --model x3d \
  --video-dir /path/to/videos \
  --recursive \
  --labels data/kinetics400/k400_val/annotations/k400_label_map_k400.txt \
  --output-csv results/classify/x3d_predictions.csv
```

### Preset examples

```bash
# Full preset
vcr-bench-attack --run-preset attack_x3d_ifgsm_debug --override attack.params.steps=3

# Mix presets
vcr-bench-attack --attack-preset ifgsm --model-preset videomae --dataset kinetics400 --lite-attack

# Print the resolved config without running
vcr-bench-test --run-preset accuracy_amd_100 --print-resolved-preset
```

---

## Usage in code

```python
from vcr_bench.models import create_model
from vcr_bench.datasets import create_dataset
from vcr_bench.attacks import create_attack
from vcr_bench.utils.eval import run_attack

model  = create_model("x3d", device="cuda")
dataset = create_dataset("kinetics400", video_root="data/kinetics400/k400_val")
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

**Applying a defence:**

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

Copy `configs/local.toml.example` to `configs/local.toml` and set your local data paths. The `[remote]` section is optional and only needed if you use `vcr-bench-remote`.

If a model, attack, or defence does not have a checked-in JSON preset yet, `vcr-bench` synthesizes a default preset from the component signature so it remains configurable through the same preset system.

---

## System requirements

Python 3.10+, PyTorch with CUDA recommended. Use `--grad-forward-chunk-size <n>` to reduce VRAM on memory-constrained GPUs.

### VMAF metric

VMAF scoring requires an FFmpeg build with `libvmaf` (most system packages omit it). To install one automatically:

**Linux:**
```bash
bash scripts/install_vmaf_ffmpeg.sh
# or, to export env vars into the current shell session:
source scripts/install_vmaf_ffmpeg.sh
```

**Windows / macOS:** run the *"Install FFmpeg With libvmaf"* cell in [`getting_started.ipynb`](getting_started.ipynb) — it downloads and extracts the right build automatically.

The script (and notebook cell) set three env vars that VCR-Bench reads:
```bash
export FFMPEG_BIN=/path/to/ffmpeg   # path to the libvmaf-enabled binary
export VMAF_BACKEND=ffmpeg
export VMAF_TIMEOUT_SEC=180
```

Pass `--vmaf` / `--no-vmaf` to `vcr-bench-attack` to explicitly enable or disable VMAF calculation per run.

### VRAM profiling

To collect per-model VRAM figures run:

```bash
vcr-bench-attack \
  --attack-preset ifgsm --model-preset x3d \
  --dataset kinetics400 --num-videos 1 --lite-attack \
  --vram-profile-csv results/vram/x3d_ifgsm.csv
```

The CSV records `no_grad_forward`, `with_grad_forward_backward`, and per-attack-call peaks (allocated / reserved MB). To profile all classifiers remotely:

```bash
vcr-bench-remote --config configs/local.toml \
  launch-attack-suite --suite vram_ifgsm10_all_classifiers
```

#### Measured peak VRAM per classifier

Peak **allocated** GPU memory for a single Kinetics-400 video on one NVIDIA A100-SXM4-80GB. **Inference** (`no_grad_forward`) and **White-box** (`with_grad_forward_backward` — one forward + one backward / differentiation) use each model's default multi-view test sampling; **Lite-attack** is a real `--lite-attack` I-FGSM run (single-view, single-pass pipeline). Models are sorted by white-box peak.

| Model | Backbone | Inference, GB | White-box, GB | Lite-attack, GB |
|---|---|---:|---:|---:|
| ONE-PEACE | vit_l40 | 7.39 | 67.97 | 37.05 |
| UMT | vit_large_p16 | 5.09 | 64.00 | 22.21 |
| InternVideo2 | vit_1b_p14 | 6.07 | 56.68 | 56.71 |
| VideoSwin | base_p244_w877 | 10.63 | 50.29 | 4.76 |
| TAdaFormer | large_14 | 3.03 | 41.54 | 14.98 |
| SlowOnly | r50 | 6.35 | 37.57 | 1.14 |
| VideoMAE | vit_base_p16 | 4.21 | 32.40 | 2.60 |
| VideoMAEv2 | vit_base_p16 | 4.21 | 32.40 | 2.60 |
| TANet | r50 | 3.60 | 32.32 | 8.66 |
| X3D | x3d_m | 4.31 | 31.36 | 1.15 |
| I3D-NonLocal | r50 | 5.45 | 28.37 | 1.00 |
| UniFormer | base | 1.89 | 27.03 | 6.99 |
| TPN | r50 | 3.84 | 26.52 | 1.13 |
| SlowFast | r101 | 3.55 | 25.67 | 1.03 |
| I3D | r50 | 5.42 | 23.42 | 0.85 |
| C2D | r50 | 3.49 | 20.17 | 0.66 |
| R2+1D | r34 | 4.76 | 19.59 | 0.80 |
| MViTv2 | base_p244 | 4.21 | 18.74 | 4.01 |
| AMD | vitb | 1.18 | 17.72 | 6.93 |
| TSM | r50 | 1.74 | 16.44 | 1.82 |
| TSM-NonLocal | r50 | 1.48 | 11.13 | 1.28 |
| UniFormerV2 | base | 1.48 | 11.05 | 1.36 |
| TSM-Surrogate | r50 | 0.92 | 8.30 | 0.97 |
| ILA | vit_b16 | 1.74 | 5.41 | 2.36 |
| ActionCLIP | vit_b16 | 0.98 | 4.06 | 4.12 |
| TSN | r101 | 0.45 | 3.27 | 0.56 |
| TimeSformer | vit_base_p16 | 0.52 | 2.63 | 1.13 |

> Figures are peak allocated memory (`torch.cuda.max_memory_allocated`); reserved peaks are higher. The Inference/White-box columns use the model's default multi-view test sampling, so for multi-view models the Lite-attack column is much lower — the lite pipeline attacks a single view in one pass. For models that are single-view by default (e.g. InternVideo2) Lite-attack ≈ White-box. ActionCLIP's inference figure is the steady-state forward; it excludes the one-time CLIP text-encoder pass (14 prompts × 400 classes ≈ 5.6k texts, ~14 GB) that is computed once and cached on the first forward.

---

## Docs

- [Data formats & pipeline conventions](docs/formats.md)
- [Component reference](docs/component_reference.md)
- [Adding a model](docs/adding_models.md)
- [Adding a dataset](docs/adding_datasets.md)
- [Adding an attack](docs/adding_attacks.md)
- [Adding a defence](docs/adding_defences.md)
- [Remote execution & Slurm guide](docs/REMOTE_DEBUGGING.md)

---

## Contributing

1. Fork the repo and create a feature branch off `master`.
2. Install in editable mode: `pip install -e ".[dev]"`.
3. Run the test suite: `pytest`.
4. Keep changes focused — one feature or fix per PR, no unrelated refactors.
5. If you add a model, attack, or defence:
   - Subclass the corresponding base in `vcr_bench/models/`, `vcr_bench/attacks/`, or `vcr_bench/defences/`.
   - Add a JSON preset under `configs/models/`, `configs/attacks/`, or `configs/defences/`.
6. Open a PR with a short description of what changed and why.

---

## Acknowledgements

- [MMAction2](https://github.com/open-mmlab/mmaction2)
- [VideoPure](https://github.com/deep-kaixun/VideoPure)
- [Background Mixup-induced Temporal Consistency (BMTC)](https://github.com/mlvccn/BMTC_TransferAttackVid)
- [StyleFool: Fooling Video Classification Systems via Style Transfer](https://arxiv.org/abs/2203.16000)
- [Temporal Shuffling for Defending Deep Action Recognition Models against Adversarial Attacks](https://arxiv.org/abs/2112.07921)
- [ActionCLIP](https://github.com/sallymmx/ActionCLIP)
- [Asymmetric Masked Distillation (AMD)](https://github.com/MCG-NJU/AMD)
- [X-CLIP](https://github.com/microsoft/VideoX/tree/master/X-CLIP)
- [InternVideo / InternVideo2](https://github.com/OpenGVLab/InternVideo)
- [ONE-PEACE](https://github.com/OFA-Sys/ONE-PEACE)
- [TAdaConv / TAdaFormer](https://github.com/alibaba-mmai-research/TAdaConv)
- [Unmasked Teacher (UMT)](https://github.com/OpenGVLab/unmasked_teacher)

We gratefully acknowledge all listed projects for their model weights, attack implementations, and defence code, which formed the basis of the corresponding components in VCR-Bench. Per-component licensing details are documented in [docs/licenses.md](docs/licenses.md).

---

## License

VCR-Bench's own code is released under the [MIT License](LICENSE).

Datasets, pretrained checkpoints, and bundled third-party components retain their original
licenses and download terms. Vendored third-party code ships its upstream license/notice
files — see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [docs/licenses.md](docs/licenses.md).
