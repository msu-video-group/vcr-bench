# Third-party notice — VideoPure (vendored)

Vendored from **deep-kaixun/VideoPure** — https://github.com/deep-kaixun/VideoPure

The upstream VideoPure repository ships **no LICENSE file**; it is released as
companion code to [arXiv:2501.14999](https://arxiv.org/abs/2501.14999) for
**research use only**.

This tree bundles several sub-components, each under its own license:

| Path | Origin | License |
|---|---|---|
| `videopure/schedule/scheduling_ddpm.py`, `scheduling_ddim.py`, and other diffusers-derived files | Hugging Face `diffusers` — https://github.com/huggingface/diffusers | Apache-2.0 — see [`videopure/schedule/LICENSE_DIFFUSERS`](videopure/schedule/LICENSE_DIFFUSERS) |
| `videopure/core/` (RAFT optical flow, incl. `raft-things.pth`) | princeton-vl/RAFT — https://github.com/princeton-vl/RAFT | BSD-3-Clause — see [`videopure/core/LICENSE_RAFT`](videopure/core/LICENSE_RAFT) |
| `videopure/core/utils/flow_viz.py` | Tom Runia | MIT (header retained in file) |

**Runtime download:** the VideoPure defence auto-downloads the diffusion model
`damo-vilab/text-to-video-ms-1.7b` (**CC BY-NC 4.0**, non-commercial) from its original
Hugging Face repo. Those weights are not bundled or redistributed here. Any run using
VideoPure is therefore research / non-commercial only.

Because the VideoPure wrapper itself carries no upstream license, redistribution of the
wrapper code requires written permission from the upstream authors. See
[docs/licenses.md](../../../docs/licenses.md).
