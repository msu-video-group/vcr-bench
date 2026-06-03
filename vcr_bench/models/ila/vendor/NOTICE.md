# Third-party notice — ILA (vendored)

Vendored from **Francis-Rings/ILA** — https://github.com/Francis-Rings/ILA

The upstream ILA repository ships **no LICENSE file**; it is released as companion
code to the paper for **research use only**. VCR-Bench does **not** redistribute the
ILA model weights (no auto-download URL; bring-your-own via `--checkpoint-path`).

This vendored tree bundles two sub-components:

| Path | Origin | License |
|---|---|---|
| `clip/` | OpenAI CLIP — https://github.com/openai/CLIP | MIT (OpenAI) — see [`clip/LICENSE`](clip/LICENSE) |
| `models/` | X-CLIP-style code from the ILA repo (derived from Microsoft VideoX X-CLIP, MIT) | no upstream license in ILA; research use only |

Because the ILA repository itself carries no license, redistribution of the `models/`
code or ILA weights requires written permission from the upstream authors. The `clip/`
sub-tree is MIT and may be redistributed under that license. See
[docs/licenses.md](../../../../docs/licenses.md).
