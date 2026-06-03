# Third-party notice — AMD (vendored)

Vendored from **MCG-NJU/AMD** — https://github.com/MCG-NJU/AMD

The upstream AMD repository ships **no LICENSE file**; it is released as companion
code to the paper for **research use only**. VCR-Bench does **not** redistribute the
AMD model weights (no auto-download URL; bring-your-own via `--checkpoint-path`).

`modeling_finetune.py` states in its own header that it is *"Based on BEiT, timm,
DINO and DeiT code bases"*. Those upstream projects and their licenses are:

| Upstream | Project | License |
|---|---|---|
| https://github.com/microsoft/unilm/tree/master/beit | BEiT | MIT (Microsoft) |
| https://github.com/rwightman/pytorch-image-models | timm | Apache-2.0 (Ross Wightman) |
| https://github.com/facebookresearch/dino | DINO | Apache-2.0 (Meta/Facebook) |
| https://github.com/facebookresearch/deit | DeiT | Apache-2.0 (Meta/Facebook) |

Because the AMD repository itself carries no license, redistribution of this code or
its weights requires written permission from the upstream authors. See
[docs/licenses.md](../../../../docs/licenses.md).
