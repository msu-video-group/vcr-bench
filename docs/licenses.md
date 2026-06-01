# Third-Party Licenses

This document lists the licenses of components used in VCR-Bench.
It is an engineering reference, not legal advice.
Items marked **"not found"** require upstream confirmation before redistribution.

---

## Models

| Model(s) | Source | License |
|---|---|---|
| VideoMAEv2, UniformerV2, Uniformer, MViTv2, VideoMAE, VideoSwin, ActionCLIP, TimeSformer, TANet, X3D, SlowFast, SlowOnly, TSMNonLocal, TSM, R(2+1)D, I3DNonLocal, C2D, I3D, TPN, TSN, TIN, TRN | [MMAction2](https://github.com/open-mmlab/mmaction2) | Apache-2.0 |
| TAdaFormer | [TAdaConv](https://github.com/alibaba-mmai-research/TAdaConv) | Apache-2.0 |
| InternVideo2 | [InternVideo](https://github.com/OpenGVLab/InternVideo) | Apache-2.0 |
| ONE-PEACE | [ONE-PEACE](https://github.com/OFA-Sys/ONE-PEACE) | Apache-2.0 |
| ONE-PEACE (fairseq dependency) | [fairseq](https://github.com/facebookresearch/fairseq) | MIT |
| AMD | [MCG-NJU/AMD](https://github.com/MCG-NJU/AMD) | No LICENSE file — released as companion code to the paper (see repo README); research use only |
| ILA | [Francis-Rings/ILA](https://github.com/Francis-Rings/ILA) | No LICENSE file — released as companion code to the paper (see repo README); research use only |
| UMT | [OpenGVLab/unmasked_teacher](https://github.com/OpenGVLab/unmasked_teacher) | MIT (upstream); local copy is missing the license file |
| FlashAttention (InternVideo dependency) | [flash-attention](https://github.com/Dao-AILab/flash-attention) | BSD-3-Clause |

---

## Attacks

| Attack | License |
|---|---|
| IFGSM, MIFGSM, AMIFGSM, UAP, Zhang-SSIM/LPIPS/DISTS, Korhonen et al., STAdv, SSAH, GradEstV2, Square | Project code — no separate third-party license |
| BMTC | No LICENSE file — [mlvccn/BMTC_TransferAttackVid](https://github.com/mlvccn/BMTC_TransferAttackVid) released as companion code to the paper (see repo README); research use only |
| StyleFool | No LICENSE file — [yuxincao22/StyleFool](https://github.com/yuxincao22/StyleFool) released as companion code to [arXiv:2203.16000](https://arxiv.org/abs/2203.16000); research use only |

---

## Defences

| Defence | License |
|---|---|
| Temporal median, shuffle, Gaussian blur, flip, crop/resize, rotate, bilateral, domain transform, randomized smoothing | Project code — no separate third-party license |
| VideoPure | No LICENSE file — [deep-kaixun/VideoPure](https://github.com/deep-kaixun/VideoPure) released as companion code to [arXiv:2501.14999](https://arxiv.org/abs/2501.14999); copied files carry Apache-2.0 and MIT headers; research use only |
| FreqPure | No LICENSE file — [GaozhengPei/FreqPure](https://github.com/GaozhengPei/FreqPure) released as companion code to the paper (see repo README); research use only |
| OpenAI guided-diffusion (FreqPure dependency) | MIT |
| Hugging Face diffusers (VideoPure dependency) | Apache-2.0 |

---

## Datasets

Model weights and benchmark results are produced on the following datasets.
Raw video files are **not** distributed with this repository.

| Dataset | License / Terms |
|---|---|
| Kinetics-400 | CC BY 4.0 (video clips from YouTube — verify redistribution rights for raw videos) |
| Something-Something V2 | Provider terms apply — do not redistribute raw videos |
| UCF-101 | Research use; videos sourced from YouTube — verify current terms before redistribution |
| HMDB-51 | Research use; mixed sources — verify current terms before redistribution |

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
| `IQA-pytorch` | Verify from installed package — used for LPIPS/DISTS metrics |
| `pytorch-msssim` | MIT |
| `pyiqa` | Apache-2.0 (some model weights may carry CC BY-NC-SA terms — check the specific metric) |
| `PyWavelets` | MIT / BSD-3-Clause |
| `yacs` | Apache-2.0 |

### Dev tools (`pip install "vcr-bench[dev]"`)

| Package | License |
|---|---|
| `pytest` | MIT |
| `ruff` | MIT |
| `mypy` | MIT |

---

## Items Requiring Attention Before Redistribution

1. **AMD, ILA** — no LICENSE file in upstream repos; obtain written permission or find a replacement before distributing these classifiers.
2. **BMTC, StyleFool** — same as above for attacks.
3. **VideoPure, FreqPure** — same as above for defences. Add the MIT license text for the OpenAI guided-diffusion portions.
4. **UMT** — add the upstream MIT license file to `Classifiers/UMT/` if the local copy is redistributed.
5. **Dataset videos and model weights** — keep out of source archives unless the respective terms explicitly permit redistribution.
6. **pyiqa model weights** — check per-metric terms (some use CC BY-NC-SA which restricts commercial use).
