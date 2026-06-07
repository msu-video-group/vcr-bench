# Demo Dataset (`--dataset-subset demo`)

A small, **redistributable** dataset for trying attacks/defences and running smoke tests
without downloading full Kinetics. It auto-downloads from Hugging Face and plugs straight
into the `kinetics400` adapter, label map, and K400-trained checkpoints.

```bash
vcr-bench-attack --model x3d --attack ifgsm --dataset kinetics400 \
  --dataset-subset demo --num-videos 8
```

## What it is

- **57 real Kinetics clips across 41 classes** — actual Kinetics-400 val/test videos, with
  their **real ground-truth labels** (not pseudo-labels).
- Every clip comes from a YouTube video published under **Creative Commons (CC BY 3.0)**, so
  the clips are redistributable (commercial use allowed) **with attribution**.
- Re-encoded to ~10 s / 30 fps / 256-px short side (Kinetics-like).
- Archive (`k400_demo_cc0.zip` on `maxv65/vcr-bench`) contains the clips,
  `annotations/k400_demo_cc0.csv`, the K400 label map, `ATTRIBUTION.md` (per-clip channel +
  URL, as CC BY requires), and `PROVENANCE.csv`.

> It is a **demo / smoke-test** set (≈1 clip per class over 41 classes), not a benchmark.
> For real numbers, run on full Kinetics with your own videos (see [datasets.md](datasets.md)).

## How it was built

`scripts/select_kinetics_cc.py` filters locally-available Kinetics clips by their YouTube
licence and keeps only the Creative Commons ones:

1. Join local Kinetics clips (val + test) to their labels via the annotation CSVs.
2. Look up each clip's YouTube licence — by default via **yt-dlp** (no API key needed; the
   `license` field is `Creative Commons Attribution…` for CC videos). A YouTube Data API key
   also works (`--api-key` / `$YT_API_KEY`, `--backend api`). Results are cached in
   `data/_meta/yt_license_cache.json`.
3. Keep `creativeCommon` clips, balanced round-robin across classes.
4. Re-encode and package the `kinetics400 / demo` archive with real labels + attribution.

```bash
# uses the val+test clips already under data/kinetics400/
python scripts/select_kinetics_cc.py \
  --video-dir data/kinetics400/k400_val/k400_val/val \
  --annotations data/kinetics400/k400_val/k400_val/annotations/k400_val.csv \
  --video-dir data/kinetics400/k400_test/extracted \
  --annotations data/kinetics400/k400_test/k400_test.csv \
  --per-class 5 --target-total 100

# inspect CC coverage first without building:
python scripts/select_kinetics_cc.py ... --dry-run
```

Publish it as the `demo` subset (flagged `redistributable = true` in
`configs/datasets.toml`):

```bash
huggingface-cli upload maxv65/vcr-bench data/k400_demo_cc0.zip k400_demo_cc0.zip
```

## Alternative: synthetic CC0 set

`scripts/build_demo_dataset.py` builds a different demo from **CC0 / public-domain Wikimedia
Commons** clips, pseudo-labelled by a classifier ensemble (no real labels, needs a GPU). It
is kept as an option if you want CC0/no-attribution content or more volume; see that script
and `scripts/demo_sources.toml`. The shipped `demo` subset is the Kinetics-CC one above.
