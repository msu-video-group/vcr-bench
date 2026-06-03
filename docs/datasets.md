# Datasets

VCR-Bench ships **adapters** for several action-recognition datasets but **does not
redistribute any raw video files**. Kinetics-400, UCF-101 and Something-Something
V2 contain third-party / copyrighted videos (see [licenses.md](licenses.md)); you must
obtain the videos yourself and point the tools at your local copy.

The only dataset that auto-downloads its videos is the CC0 **demo** subset
(see [demo_dataset.md](demo_dataset.md)).

## Supported adapters

| Dataset | `--dataset` | Raw videos |
|---|---|---|
| Kinetics-400 | `kinetics400` | bring your own (`--video-root`) |
| UCF-101 | `ucf101` | bring your own (`--video-root`) |
| Something-Something V2 | `sthv2` | bring your own (`--video-root`) |
| CC0 demo | `kinetics400 --dataset-subset demo` | auto-downloaded (redistributable) |

## Providing your own videos

Point the adapter at a local video directory plus an annotations file and a label map:

```bash
vcr-bench-test --model x3d --dataset kinetics400 \
  --video-root /data/kinetics400/val \
  --annotations /data/kinetics400/annotations/val.csv \
  --labels /path/to/label_map_k400.txt
```

When `--video-root` (and `--annotations`/`--labels`) are given, the adapter uses them
directly and never attempts a download. If you instead pass `--dataset-subset` for a
non-redistributable dataset and the videos are not present locally, the run fails with an
instructive error pointing you here.

Where to get the videos:

- **Kinetics-400** — e.g. the community mirror
  [cvdfoundation/kinetics-dataset](https://github.com/cvdfoundation/kinetics-dataset).
  Only the annotations are CC BY 4.0; the clips remain under their YouTube uploaders'
  copyright, so do not redistribute them.
- **UCF-101** — [crcv.ucf.edu/data/UCF101.php](https://www.crcv.ucf.edu/data/UCF101.php).
- **Something-Something V2** — [Qualcomm / 20BN provider terms](https://developer.qualcomm.com/software/ai-datasets/something-something).

## Annotation format

See [adding_datasets.md](adding_datasets.md) for the per-adapter CSV/JSON conventions and
how to register a new dataset.

## The `redistributable` flag

Each subset entry in `configs/datasets.toml` carries `redistributable = true|false`. Only
`true` subsets may auto-download a video archive; everything else requires local videos.
This is the single switch that keeps copyrighted videos out of the toolchain.
