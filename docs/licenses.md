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
| AMD | [MCG-NJU/AMD](https://github.com/MCG-NJU/AMD) | No LICENSE file — research use only; weights not redistributed (bring-your-own). Provenance: [`vcr_bench/models/amd/vendor/NOTICE.md`](../vcr_bench/models/amd/vendor/NOTICE.md) |
| ILA | [Francis-Rings/ILA](https://github.com/Francis-Rings/ILA) | No LICENSE file — research use only; weights not redistributed (bring-your-own). Bundled `clip/` is MIT. Provenance: [`vcr_bench/models/ila/vendor/NOTICE.md`](../vcr_bench/models/ila/vendor/NOTICE.md) |
| UMT | [OpenGVLab/unmasked_teacher](https://github.com/OpenGVLab/unmasked_teacher) | MIT — license included at `vcr_bench/models/umt/LICENSE` |
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
| VideoPure | No upstream LICENSE — [deep-kaixun/VideoPure](https://github.com/deep-kaixun/VideoPure) released as companion code to [arXiv:2501.14999](https://arxiv.org/abs/2501.14999); research use only. Bundled sub-components carry their own licenses (diffusers Apache-2.0, RAFT BSD-3-Clause, flow_viz MIT) — see [`vcr_bench/defences/videopure/NOTICE.md`](../vcr_bench/defences/videopure/NOTICE.md) |
| FreqPure | No upstream LICENSE — [GaozhengPei/FreqPure](https://github.com/GaozhengPei/FreqPure) released as companion code to the paper; research use only. Bundled guided-diffusion is MIT — see [`vcr_bench/defences/freqpure/NOTICE.md`](../vcr_bench/defences/freqpure/NOTICE.md) |
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
| Kinetics-400 | **Annotations**: CC BY 4.0 (Google/DeepMind; commercial use permitted with attribution — the original paper's "CC BY-NC" is superseded by the official CC BY 4.0). **Video clips**: owned by the original YouTube uploaders, **not** covered by CC BY 4.0 — raw videos must **not** be redistributed. |
| Something-Something V2 | Provider terms apply — do not redistribute raw videos |
| UCF-101 | No formal license on the [CRCV page](https://www.crcv.ucf.edu/data/UCF101.php) — only a citation request. **Video clips** are sourced from YouTube and owned by the original uploaders; raw videos must **not** be redistributed. VCR-Bench ships an adapter only. |

VCR-Bench ships **adapters** for these datasets but **does not distribute any raw videos**.
Download the videos yourself from the provider and point the tools at them with
`--video-root` (see [datasets.md](datasets.md)). For an out-of-the-box, redistributable
example, use the CC0/public-domain **demo** subset instead (see
[demo_dataset.md](demo_dataset.md)).

| Demo subset (CC0) | Small pseudo-labelled set built from CC0 / public-domain Wikimedia Commons clips via [`scripts/build_demo_dataset.py`](../scripts/build_demo_dataset.py). Redistributable (incl. commercially), no attribution required; per-clip provenance recorded in `PROVENANCE.csv`. Labels are model-generated **pseudo-labels**, not ground truth — for demos only. |

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
| `IQA-pytorch` | MIT ([dingkeyan93/IQA-pytorch](https://github.com/dingkeyan93/IQA-pytorch); package imported as `IQA_pytorch`, used for LPIPS/DISTS) |
| `pytorch-msssim` | MIT |
| `pyiqa` | **PolyForm Noncommercial 1.0.0 + NTU S-Lab License 1.0** ([chaofengc/IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch)) — **non-commercial / research use only**; some metric weights additionally carry CC BY-NC-SA. Not the same package as `IQA-pytorch` above. |
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
| **FFmpeg** — GPL build from [BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds) (`ffmpeg-master-latest-*-gpl`) | **GPL-3.0** | The notebook downloads this build only because it bundles the `libvmaf` filter. It is run as a standalone executable via `subprocess` (mere aggregation — not linked into VCR-Bench), and is not committed to the repo. To avoid GPL entirely, supply an LGPL FFmpeg build with `libvmaf` and point `FFMPEG_BIN` at it. |
| **VMAF / libvmaf** — [Netflix/vmaf](https://github.com/Netflix/vmaf) | **BSD-2-Clause-Patent** (a.k.a. BSD+Patent) | Reached through the FFmpeg `libvmaf` filter above; the bundled VMAF model files carry the same Netflix license. |

---

## Items Requiring Attention Before Redistribution

1. **AMD, ILA** — no LICENSE file in upstream repos (research use only). VCR-Bench does **not** redistribute their weights: both are bring-your-own (no auto-download URL), so the user must supply the checkpoint via `--checkpoint-path`. Obtain written permission before redistributing these classifiers or their weights.
2. **BMTC, StyleFool** — same as above for attacks.
3. **VideoPure, FreqPure** — same as above for defences. Add the MIT license text for the OpenAI guided-diffusion portions.
4. **UMT** — upstream MIT license now included at `vcr_bench/models/umt/LICENSE`. (Vendored license/NOTICE files for all bundled third-party code are indexed in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).)
5. **Dataset videos and model weights** — keep out of source archives unless the respective terms explicitly permit redistribution. VCR-Bench does not bundle raw videos for Kinetics-400 / UCF-101 / SSv2; only the CC0 `demo` subset is auto-downloadable (`redistributable = true` in `configs/datasets.toml`).
6. **`pyiqa` is non-commercial** (PolyForm Noncommercial + NTU S-Lab) — any benchmark path that uses `pyiqa` metrics is **research / non-commercial only**, even though VCR-Bench's own code is MIT. Some `pyiqa` metric weights additionally carry CC BY-NC-SA. Do not confuse it with the MIT-licensed `IQA-pytorch` package.
7. **VideoPure diffusion weights are non-commercial.** The VideoPure defence auto-downloads `damo-vilab/text-to-video-ms-1.7b` (**CC BY-NC 4.0**). Any run using VideoPure is therefore **research / non-commercial only**. These weights are not bundled — they are fetched from the original Hugging Face repo at run time, so VCR-Bench does not redistribute them; do not vendor or re-host them.
8. **VMAF metric requires a GPL FFmpeg build.** `getting_started.ipynb` downloads a **GPL-3.0** FFmpeg build (BtbN) at run time purely to access `libvmaf` (**BSD+Patent**). It is invoked as a separate executable and is not bundled or linked, so VCR-Bench's MIT licensing is unaffected. Anyone wishing to avoid GPL can point `FFMPEG_BIN` at an LGPL FFmpeg build that includes `libvmaf`.
