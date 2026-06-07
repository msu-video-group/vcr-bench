# Third-Party Notices

VCR-Bench's own code is licensed under the [MIT License](LICENSE). It additionally
**vendors** (bundles copies of) source code from the third-party projects listed below.
Each vendored directory retains the upstream license text (`LICENSE`) and, where the
upstream ships no license, a `NOTICE.md` recording its provenance.

This file is the index; for the full engineering discussion (datasets, weights, runtime
downloads, non-commercial caveats) see [docs/licenses.md](docs/licenses.md).

## Vendored code with an upstream license

| Path | Upstream | License | File |
|---|---|---|---|
| `vcr_bench/models/vendor_mmaction/` | [open-mmlab/mmaction2](https://github.com/open-mmlab/mmaction2) | Apache-2.0 | [LICENSE](vcr_bench/models/vendor_mmaction/LICENSE) · [NOTICE](vcr_bench/models/vendor_mmaction/NOTICE.md) |
| `vcr_bench/models/tadaformer/vendor/` | [alibaba-mmai-research/TAdaConv](https://github.com/alibaba-mmai-research/TAdaConv) | Apache-2.0 | [LICENSE](vcr_bench/models/tadaformer/vendor/LICENSE) · [NOTICE](vcr_bench/models/tadaformer/vendor/NOTICE.md) |
| `vcr_bench/models/internvideo/` | [OpenGVLab/InternVideo](https://github.com/OpenGVLab/InternVideo) | Apache-2.0 | [LICENSE](vcr_bench/models/internvideo/LICENSE) |
| `vcr_bench/models/internvideo2/` | [OpenGVLab/InternVideo (InternVideo2)](https://github.com/OpenGVLab/InternVideo) | Apache-2.0 | [LICENSE](vcr_bench/models/internvideo2/LICENSE) |
| `vcr_bench/models/onepeace/` | [OFA-Sys/ONE-PEACE](https://github.com/OFA-Sys/ONE-PEACE) | Apache-2.0 | [LICENSE](vcr_bench/models/onepeace/LICENSE) |
| `vcr_bench/models/umt/` | [OpenGVLab/unmasked_teacher](https://github.com/OpenGVLab/unmasked_teacher) | MIT | [LICENSE](vcr_bench/models/umt/LICENSE) |

## Components NOT distributed (no upstream license — local-only)

The components below wrap upstream projects that ship **no LICENSE file** (all rights
reserved; released only as companion code to their papers, for research use). Because we
have no license to redistribute them, **their code is not included in this repository** —
it is git-ignored and lives only in a local checkout. VCR-Bench keeps the dynamic-plugin
hooks (registry keys, config presets) so that if you obtain the upstream code yourself and
place it at the path below, the component activates automatically; otherwise the component
is simply unavailable. Model weights for these are likewise bring-your-own via
`--checkpoint-path` (no auto-download). Obtain code/weights and clarify redistribution
rights directly with the upstream authors.

| Path (local-only) | Upstream | Kind |
|---|---|---|
| `vcr_bench/models/amd/` | [MCG-NJU/AMD](https://github.com/MCG-NJU/AMD) | model |
| `vcr_bench/models/ila/` | [Francis-Rings/ILA](https://github.com/Francis-Rings/ILA) (bundles openai/CLIP, MIT) | model |
| `vcr_bench/defences/videopure/` | [deep-kaixun/VideoPure](https://github.com/deep-kaixun/VideoPure) (bundles diffusers Apache-2.0, RAFT BSD-3-Clause) | defence |
| `vcr_bench/defences/freqpure/` | [GaozhengPei/FreqPure](https://github.com/GaozhengPei/FreqPure) (bundles openai/guided-diffusion, MIT) | defence |
| `vcr_bench/attacks/bmtc/` | [mlvccn/BMTC_TransferAttackVid](https://github.com/mlvccn/BMTC_TransferAttackVid) | attack |
| `vcr_bench/attacks/stylefool/` | [yuxincao22/StyleFool](https://github.com/yuxincao22/StyleFool) | attack |

## Runtime downloads & non-commercial caveats

Some components fetch weights or tools at run time rather than bundling them. The notable
licensing implications (VideoPure's **CC BY-NC 4.0** diffusion weights, the **GPL** FFmpeg
build used for VMAF, `pyiqa` non-commercial metrics) are documented in
[docs/licenses.md](docs/licenses.md) under *Items Requiring Attention Before Redistribution*.
