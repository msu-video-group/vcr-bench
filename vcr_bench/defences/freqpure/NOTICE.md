# Third-party notice — FreqPure (vendored)

Vendored from **GaozhengPei/FreqPure** — https://github.com/GaozhengPei/FreqPure

The upstream FreqPure repository ships **no LICENSE file**; it is released as
companion code to the paper for **research use only**.

This tree bundles one sub-component under its own license:

| Path | Origin | License |
|---|---|---|
| `guided_diffusion/` | OpenAI guided-diffusion — https://github.com/openai/guided-diffusion | MIT (OpenAI) — see [`guided_diffusion/LICENSE_GUIDED_DIFFUSION`](guided_diffusion/LICENSE_GUIDED_DIFFUSION) |

**Runtime download:** the FreqPure defence auto-downloads the OpenAI guided-diffusion
checkpoint `256x256_diffusion_uncond.pt` (MIT) from its original source; it is not
bundled or redistributed here.

Because the FreqPure wrapper itself carries no upstream license, redistribution of the
wrapper code requires written permission from the upstream authors. See
[docs/licenses.md](../../../docs/licenses.md).
