# Data Formats and Pipeline Conventions

## Tensor formats

All video tensors carry an explicit format string. The canonical formats used across the codebase:

| String | Axes | Where used |
|--------|------|-----------|
| `NTHWC` | batch, time, height, width, channels | Raw decoded video, sampled clips, attack input/output, defence input/output |
| `NCTHW` | batch, channels, time, height, width | Preprocessed model input (most models) |
| `NTCHW` | batch, time, channels, height, width | Preprocessed model input (some models) |

> **Rule of thumb:** if the last axis is size 3 it's `NTHWC`; if the second axis is size 3 it's `NCTHW`.

Format conversion utility:

```python
from vcr_bench.utils.preprocessing import to_format

x_ncthw = to_format(x, src_format="NTHWC", dst_format="NCTHW")
```

## Value ranges

| Stage | Range | Dtype |
|-------|-------|-------|
| Decoded from disk | `[0, 255]` uint8 | `torch.uint8` |
| Sampled clips (passed to attacks/defences) | `[0, 255]` float | `torch.float32` |
| After model preprocessing | model-specific (typically mean/std normalised) | `torch.float32` |
| Attack perturbations (eps) | pixel units matching the sampled range | `torch.float32` |

Defences always receive and must return `[N, T, H, W, C]` float32 in `[0, 255]`.
Attacks operate on sampled tensors in `[0, 255]` before model preprocessing is applied.

## Pipeline stages

Models define behaviour per pipeline stage. Available stages: `"train"`, `"val"`, `"test"`, `"attack"`.

The stage controls:
- **Number of clips** — `test` typically uses multi-clip (e.g. 10 clips), `attack` uses 1 clip to save memory.
- **Spatial strategy** — `test` uses `three_crop`, `attack` uses `center_crop`.

The stage is set by the CLI via `--pipeline-stage` (default `"test"`).

## Data pipeline flow

```
disk
  │  decode_video_nthwc()
  ▼
[1, T_full, H, W, C] uint8   ← full decoded video
  │  _sample_video_public()
  ▼
[num_clips, clip_len, H, W, C] float32   ← sampled clips (NTHWC)
  │  (optional) defence.transform()      ← defences run HERE
  ▼
[num_clips, clip_len, H, W, C] float32   ← defended sampled clips
  │  _preprocess_sampled()
  ▼
[num_clips, C, clip_len, H', W'] float32  ← preprocessed (NCTHW or NTCHW)
  │  model(x)
  ▼
[num_clips, num_classes] float32          ← logits
  │  softmax + mean over clips
  ▼
[num_classes] float32                     ← final probabilities
```

## Sampling config fields

Defined in `vcr_bench/models/pipeline_config.py` as `VideoLoadingConfig`:

| Field | Default | Meaning |
|-------|---------|---------|
| `clip_len` | 16 | Frames per clip |
| `num_clips` | 1 | Number of temporal clips to sample |
| `frame_interval` | 1 | Stride between sampled frames |
| `temporal_strategy` | `"uniform"` | How to pick clip start positions |
| `full_videos` | `False` | Keep full decoded tensor alongside sampled |

## Preprocessing config fields

Defined as `VideoPreprocessingConfig`:

| Field | Default | Meaning |
|-------|---------|---------|
| `preprocessed_format` | `"NCTHW"` | Output tensor format |
| `resize_short` | 224 | Resize shorter spatial side to this |
| `crop_size` | 224 | Crop size after resize |
| `input_range` | 255.0 | Input scale before mean/std (use `1.0` if checkpoint expects [0,1]) |
| `mean` | `(0.485, 0.456, 0.406)` | Per-channel mean (applied after dividing by `input_range`) |
| `std` | `(0.229, 0.224, 0.225)` | Per-channel std |
| `spatial_strategy` | `"three_crop"` | `center_crop`, `three_crop`, `train_random_crop` |

## Core data types

**`VideoSampleRef`** (frozen dataclass, `vcr_bench/types.py`):
```python
id: str            # unique sample identifier
path: str          # path to video file
label: int | None  # ground-truth class index
label_name: str | None
metadata: dict
```

**`PredictionBundle`** (dataclass):
```python
logits: torch.Tensor     # raw model output [V, C]
probs: torch.Tensor      # softmax probabilities [C]
pred_label: int          # argmax class
metadata: dict
```

**`LoadedDatasetItem`** (dataclass):
```python
sample: VideoSampleRef
tensor: torch.Tensor        # sampled/preprocessed video
input_format: str
sampled: bool
preprocessed: bool
full_tensor: torch.Tensor | None
sampled_frame_ids: torch.Tensor | None
```
