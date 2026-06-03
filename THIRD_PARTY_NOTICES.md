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
| `vcr_bench/models/ila/vendor/clip/` | [openai/CLIP](https://github.com/openai/CLIP) | MIT | [LICENSE](vcr_bench/models/ila/vendor/clip/LICENSE) |
| `vcr_bench/defences/videopure/videopure/schedule/` (diffusers-derived) | [huggingface/diffusers](https://github.com/huggingface/diffusers) | Apache-2.0 | [LICENSE_DIFFUSERS](vcr_bench/defences/videopure/videopure/schedule/LICENSE_DIFFUSERS) |
| `vcr_bench/defences/videopure/videopure/core/` (RAFT) | [princeton-vl/RAFT](https://github.com/princeton-vl/RAFT) | BSD-3-Clause | [LICENSE_RAFT](vcr_bench/defences/videopure/videopure/core/LICENSE_RAFT) |
| `vcr_bench/defences/freqpure/guided_diffusion/` | [openai/guided-diffusion](https://github.com/openai/guided-diffusion) | MIT | [LICENSE_GUIDED_DIFFUSION](vcr_bench/defences/freqpure/guided_diffusion/LICENSE_GUIDED_DIFFUSION) |

## Vendored code with NO upstream license (research use only)

These upstreams ship no LICENSE file and are released as companion code to their papers.
VCR-Bench does **not** redistribute their model weights (bring-your-own via
`--checkpoint-path`). Redistributing the code or weights requires written permission from
the upstream authors.

| Path | Upstream | Notice |
|---|---|---|
| `vcr_bench/models/amd/vendor/` | [MCG-NJU/AMD](https://github.com/MCG-NJU/AMD) | [NOTICE](vcr_bench/models/amd/vendor/NOTICE.md) |
| `vcr_bench/models/ila/vendor/models/` | [Francis-Rings/ILA](https://github.com/Francis-Rings/ILA) | [NOTICE](vcr_bench/models/ila/vendor/NOTICE.md) |
| `vcr_bench/defences/videopure/` (wrapper) | [deep-kaixun/VideoPure](https://github.com/deep-kaixun/VideoPure) | [NOTICE](vcr_bench/defences/videopure/NOTICE.md) |
| `vcr_bench/defences/freqpure/` (wrapper) | [GaozhengPei/FreqPure](https://github.com/GaozhengPei/FreqPure) | [NOTICE](vcr_bench/defences/freqpure/NOTICE.md) |

## Runtime downloads & non-commercial caveats

Some components fetch weights or tools at run time rather than bundling them. The notable
licensing implications (VideoPure's **CC BY-NC 4.0** diffusion weights, the **GPL** FFmpeg
build used for VMAF, `pyiqa` non-commercial metrics) are documented in
[docs/licenses.md](docs/licenses.md) under *Items Requiring Attention Before Redistribution*.
