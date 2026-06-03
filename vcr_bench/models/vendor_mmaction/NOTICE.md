# Third-party notice — vendored MMAction2 model definitions

This package is a curated subset of model definitions vendored from
**open-mmlab/mmaction2** — https://github.com/open-mmlab/mmaction2

Licensed under **Apache-2.0** — see [`LICENSE`](LICENSE). Copyright 2018-2019 OpenMMLab.

Most model dirs under `vcr_bench/models/` (e.g. `c3d`, `c2d`, `csn`, `i3d`,
`slowfast`, `slowonly`, `tanet`, `tpn`, `tsm`, `tsn`, `timesformer`, `videomae`,
`videomaev2`, `videoswin`, `mvitv2`, `uniformer`, `uniformerv2`, `x3d`, `actionclip`)
are thin wrappers that import the architectures defined here.

`onepeace.py` (`OnePeaceViT`) is adapted from **OFA-Sys/ONE-PEACE**
(https://github.com/OFA-Sys/ONE-PEACE), also Apache-2.0; see
[`../onepeace/LICENSE`](../onepeace/LICENSE).

See [docs/licenses.md](../../../docs/licenses.md).
