# Third-Party Licenses

This document lists the licenses of components used in VCR-Bench.
It is an engineering reference, not legal advice.
Items marked **"not found"** require upstream confirmation before redistribution.

For the per-directory provenance of **vendored** third-party source code (each carries its
upstream `LICENSE` text or a `NOTICE.md`), see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

---

## Models

| Model(s) | Source | License |
|---|---|---|
| VideoMAEv2, UniformerV2, Uniformer, MViTv2, VideoMAE, VideoSwin, ActionCLIP, TimeSformer, TANet, X3D, SlowFast, SlowOnly, TSMNonLocal, TSM, R(2+1)D, I3DNonLocal, C2D, I3D, TPN, TSN, TIN, TRN | [MMAction2](https://github.com/open-mmlab/mmaction2) | Apache-2.0 |
| TAdaFormer | [TAdaConv](https://github.com/alibaba-mmai-research/TAdaConv) | Apache-2.0 |
| InternVideo2 | [InternVideo](https://github.com/OpenGVLab/InternVideo) | Apache-2.0 |
| ONE-PEACE | [ONE-PEACE](https://github.com/OFA-Sys/ONE-PEACE) | Apache-2.0 |
| ONE-PEACE (fairseq dependency) | [fairseq](https://github.com/facebookresearch/fairseq) | MIT |
| AMD | [MCG-NJU/AMD](https://github.com/MCG-NJU/AMD) | **code not distributed** with this repo. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). |
| ILA | [Francis-Rings/ILA](https://github.com/Francis-Rings/ILA) | **code not distributed** with this repo. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). |
| UMT | [OpenGVLab/unmasked_teacher](https://github.com/OpenGVLab/unmasked_teacher) | MIT — license included at `vcr_bench/models/umt/LICENSE` |
| FlashAttention (InternVideo dependency) | [flash-attention](https://github.com/Dao-AILab/flash-attention) | BSD-3-Clause |

---

## Attacks

| Attack | License |
|---|---|
| IFGSM, MIFGSM, AMIFGSM, UAP, Zhang-SSIM/LPIPS/DISTS, Korhonen et al., STAdv, SSAH, GradEstV2, Square | Project code |
| BMTC |  our own implementation of the method |
| StyleFool | our own implementation of the method |

---

## Defences

| Defence | License |
|---|---|
| Temporal median, shuffle, Gaussian blur, flip, crop/resize, rotate, bilateral, domain transform, randomized smoothing | Project code — no separate third-party license |
| VideoPure | **Code not distributed** with this repo |
| FreqPure |  **Code not distributed** with this repo |
| OpenAI guided-diffusion (FreqPure dependency) | MIT |
| Hugging Face diffusers (VideoPure dependency) | Apache-2.0 |
| **VideoPure diffusion weights** — auto-downloaded `damo-vilab/text-to-video-ms-1.7b` (default `VIDEOPURE_MODEL_ID`) | **CC BY-NC 4.0** ([model card](https://huggingface.co/damo-vilab/text-to-video-ms-1.7b); ali-vilab / ModelScope) — **non-commercial use only**. Any run that uses the VideoPure defence pulls these weights, so that evaluation path is research / non-commercial only, even though VCR-Bench's own code is MIT. |
| FreqPure diffusion weights — `256x256_diffusion_uncond.pt` from OpenAI (auto-downloaded from the original source) | MIT |

---

## Datasets

Model weights and benchmark results are produced on the following datasets.
Raw video files are **not** distributed with this repository.

| Dataset | License / Terms |
|---|---|
| Kinetics-400 | **Annotations**: CC BY 4.0 (Google/DeepMind; commercial use permitted with attribution — the original paper's "CC BY-NC" is superseded by the official CC BY 4.0). **Video clips**: we provide selected demo subset covered by CC BY 4.0.  |
| Something-Something V2 | We do not provide raw videos. Only adapter |
| UCF-101 |  We do not provide raw videos. Only adapter |

Download the videos yourself from the provider and point the tools at them with
`--video-root` (see [datasets.md](datasets.md)). For an out-of-the-box, redistributable
example, use the CC0/public-domain **demo** subset instead (see
[demo_dataset.md](demo_dataset.md)).

---

## Python Dependencies

### Core (installed by default)

| Package | License |
|---|---|
| `numpy` | BSD-3-Clause |
| `pandas` | BSD-3-Clause |
| `torch`, `torchvision` | BSD-3-Clause |
| `tqdm` | MIT / MPL-2.0 |
| `av` (PyAV) | BSD-3-Clause |
| `scikit-image` | BSD-3-Clause |
| `opencv-python-headless` | Apache-2.0 |
| `huggingface-hub` | Apache-2.0 |
| `tomli` | MIT |

### Research extras (`pip install "vcr-bench[research]"`)

| Package | License |
|---|---|
| `transformers` | Apache-2.0 |
| `diffusers` | Apache-2.0 |
| `timm` | Apache-2.0 |
| `einops` | MIT |
| `scipy` | BSD-3-Clause |
| `Pillow` | HPND (PIL-style) |
| `IQA-pytorch` | MIT |
| `pytorch-msssim` | MIT |
| `pyiqa` | **PolyForm Noncommercial 1.0.0 + NTU S-Lab License 1.0** |
| `PyWavelets` | MIT / BSD-3-Clause |
| `yacs` | Apache-2.0 |

### Dev tools (`pip install "vcr-bench[dev]"`)

| Package | License |
|---|---|
| `pytest` | MIT |
| `ruff` | MIT |
| `mypy` | MIT |

### External tools (runtime, not bundled in the repo)

These are **not** redistributed by VCR-Bench. They are downloaded or installed on the
user's machine at run time (e.g. by `getting_started.ipynb`) and invoked as separate
processes for VMAF metric computation.

| Tool | License | Notes |
|---|---|---|
| **FFmpeg** — GPL build from [BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds) (`ffmpeg-master-latest-*-gpl`) | **GPL-3.0** | The notebook downloads this build only because it bundles the `libvmaf` filter. It is run as a standalone executable via `subprocess` (mere aggregation — not linked into VCR-Bench), and is not committed to the repo. |
| **VMAF / libvmaf** — [Netflix/vmaf](https://github.com/Netflix/vmaf) | **BSD-2-Clause-Patent** (a.k.a. BSD+Patent) | Reached through the FFmpeg `libvmaf` filter above; the bundled VMAF model files carry the same Netflix license. |
