# Benchmark reproduction presets

Each file here is a **tracks config**: it lists the model × attack × defence matrix
the autolaunch service should run. Pick one to reproduce all or part of the
benchmark behind the explorer pages.

| Preset | What it runs | Writes to (`results/remote_attacks/…`) | Explorer page |
|--------|--------------|----------------------------------------|---------------|
| `full_white_box`  | 28 models × 8 targeted gradient/optimisation attacks × 8 defences, 100 videos | `full_white_box/` | White-Box |
| `full_blackbox`   | 28 models × 4 untargeted query attacks × 7 defences, 30 videos | `blackbox_full/` | Black-Box |
| `full_benchmark`  | both tracks above (== the live `../tracks.json`) | both folders | both |
| `quick`           | partial smoke: 3 models × 1 attack × (no_defence + flip), 8 videos | `quick_white_box/`, `quick_blackbox/` | — (scratch) |

The full presets reproduce the **exact** parameters that generated the current
results, so their outputs land in the same folders the explorer caches are built
from. `quick` uses distinct `attack_name`s and writes to scratch folders, so it
will never overwrite the canonical results.

## Run a preset (on the cluster, from the repo root)

```bash
./scripts/autolaunch/run_benchmark.sh quick            # fast end-to-end check
./scripts/autolaunch/run_benchmark.sh full_white_box   # white-box only
./scripts/autolaunch/run_benchmark.sh full_blackbox    # black-box only
./scripts/autolaunch/run_benchmark.sh full_benchmark   # everything (default)
```

`run_benchmark.sh` clones `../main.json` (keeping all scheduler/slurm settings),
swaps in the chosen preset as `tracks_config`, and starts the service. You can
also point it at any tracks JSON by path:

```bash
./scripts/autolaunch/run_benchmark.sh /abs/path/to/my_tracks.json
```

## Reproduction semantics

- The service **skips cells whose results already exist** (CSV present with enough
  rows, or recorded in a log). So re-running a preset only fills missing cells.
- To reproduce a track **from scratch**, clear its output folder first, e.g.
  `rm -rf results/remote_attacks/full_white_box`, then launch the preset.
- For a **partial** reproduction of real benchmark cells, copy a full preset and
  trim its `models` / `attacks` / `defences` lists while keeping `num_videos` and
  `attack_name` — the produced CSVs are then directly comparable to (and mergeable
  with) the full results.

## Rebuild the explorer data after a run

Pull the results locally, then rebuild the per-track caches:

```bash
python scripts/website/build_full_white_box_site.py --track full_white_box
python scripts/website/build_full_white_box_site.py --track blackbox_full
```

This regenerates `docs/data/website_cache.json` and
`docs/data/website_cache_blackbox.json` consumed by the explorer pages.
