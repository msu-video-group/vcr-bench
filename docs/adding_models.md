# Adding a Model

## Directory layout

```
vcr_bench/models/<name>/
    __init__.py        # empty or re-export
    model.py           # classifier implementation
    checkpoints/       # auto-created, stores downloaded .pth files
configs/models/<name>.json   # preset (backbone variants + params)
```

The registry finds your model by importing `vcr_bench.models.<name>.model` and looking for `create()` or `MODEL_CLASS`.

## 1. Implement the classifier

```python
# vcr_bench/models/mymodel/model.py
from __future__ import annotations
from typing import Any
import torch
from ..base import BaseVideoClassifier
from ..pipeline_config import PipelineStage


class MyModelClassifier(BaseVideoClassifier):
    model_name = "mymodel"
    raw_input_format = "NTHWC"
    preprocessed_format = "NCTHW"   # or "NTCHW" — must match self.model's expected input

    # Describes available backbones and weight sets. Used for checkpoint resolution
    # and by vcr-bench-artifacts list-checkpoints.
    MODEL_CAPABILITIES: dict[str, Any] = {
        "backbones": {
            "default": {
                "display_name": "MyModel",
                "datasets": {
                    "kinetics400": {
                        "num_classes": 400,
                        "checkpoint_url": "https://example.com/mymodel_k400.pth",
                        "checkpoint_filename": "mymodel_k400.pth",
                    }
                },
            }
        }
    }

    def __init__(
        self,
        checkpoint_path: str | None = None,
        backbone: str | None = None,
        weights_dataset: str | None = None,
        device: str = "cuda",
        grad_forward_chunk_size: int | None = None,
        load_weights: bool | None = None,
        num_classes: int | None = None,
        auto_download: bool = True,
    ) -> None:
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        self.backbone = backbone or "default"
        self.weights_dataset = weights_dataset or "kinetics400"
        if grad_forward_chunk_size is not None:
            self.grad_forward_chunk_size = int(grad_forward_chunk_size)

        weight_spec = self.get_weight_spec(self.backbone, self.weights_dataset)
        self.num_classes = int(num_classes if num_classes is not None else weight_spec.get("num_classes", 400))

        # Build per-stage loading and preprocessing configs (required by base class)
        self.loading_configs, self.preprocessing_configs = self._build_stage_config_dicts()
        self._active_stage: PipelineStage = "test"
        self._bind_stage_functions("test")

        # Build the actual PyTorch model
        self.model = _build_my_model(self.num_classes)

        # Load weights
        should_load = True if load_weights is None else bool(load_weights)
        if should_load:
            self.checkpoint_path = self.resolve_checkpoint_path(
                self.backbone,
                self.weights_dataset,
                checkpoint_path,
                auto_download=auto_download,
            )
            self._load_checkpoint(self.checkpoint_path)
        else:
            self.checkpoint_path = checkpoint_path

        self.model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False

    def _build_stage_config_dicts(self) -> tuple[dict, dict]:
        # Loading config controls temporal sampling per stage.
        common_loading = {
            "raw_input_format": "NTHWC",
            "sampled_format": "NTHWC",
            "clip_len": 16,
            "frame_interval": 4,
            "full_videos": False,
        }
        # Preprocessing config controls spatial transforms + normalisation.
        common_pre = {
            "preprocessed_format": "NCTHW",
            "resize_short": 256,
            "crop_size": 224,
            "input_range": 255.0,   # divide pixel values by this before mean/std
            "mean": (0.45, 0.45, 0.45),
            "std": (0.225, 0.225, 0.225),
        }
        loading = {
            "train": {**common_loading, "num_clips": 1},
            "val":   {**common_loading, "num_clips": 1},
            "test":  {**common_loading, "num_clips": 10},
            "attack":{**common_loading, "num_clips": 1},
        }
        preprocessing = {
            "train":  {**common_pre, "spatial_strategy": "train_random_crop"},
            "val":    {**common_pre, "spatial_strategy": "center_crop"},
            "test":   {**common_pre, "spatial_strategy": "three_crop"},
            "attack": {**common_pre, "spatial_strategy": "center_crop"},
        }
        return loading, preprocessing

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        # Use the base class helper for common checkpoint formats (handles module. prefix,
        # state_dict nesting, key remapping, etc.).
        self.load_converted_checkpoint(
            checkpoint_path,
            strict=False,
            strip_prefixes=("module.", "model."),
        )


MODEL_CLASS = MyModelClassifier


def create(**kwargs) -> MyModelClassifier:
    return MyModelClassifier(**kwargs)
```

### `_build_stage_config_dicts` notes

- `"test"` stage: use **multi-clip** (`num_clips > 1`) and `"three_crop"` for best accuracy.
- `"attack"` stage: use **single clip** and `"center_crop"` — multi-clip explodes memory during backprop.
- `input_range`: set to `255.0` if your mean/std are in `[0, 1]` range (ImageNet convention). Set to `1.0` if your checkpoint was trained with raw pixel mean/std.

### Custom checkpoint loading

If your checkpoint has unusual key structure, override `_load_checkpoint`:

```python
def _load_checkpoint(self, checkpoint_path: str) -> None:
    self.load_converted_checkpoint(
        checkpoint_path,
        strict=False,
        root_keys=("state_dict", "model"),       # unwrap these wrapper dicts
        strip_prefixes=("module.", "backbone."),  # strip these key prefixes
        key_replacements=(("encoder.", ""),),     # rename key patterns
        ignore_prefixes=("head.",),               # skip these keys entirely
    )
```

## 2. Create the preset config

```json
// configs/models/mymodel.json
{
  "kind": "model",
  "name": "mymodel",
  "default_variant": "k400",
  "variants": {
    "k400": {
      "factory_name": "mymodel",
      "params": {
        "backbone": "default",
        "weights_dataset": "kinetics400"
      }
    }
  }
}
```

## 3. Register the checkpoint (optional but recommended)

Add an entry to `configs/checkpoints.toml` so `vcr-bench-artifacts` can manage it:

```toml
[checkpoints.mymodel_default_kinetics400]
model = "mymodel"
backbone = "default"
weights_dataset = "kinetics400"
repo_id = "your-hf-org/checkpoints-mymodel"
filename = "mymodel_k400.pth"
revision = "main"
local_subdir = "checkpoints/mymodel"
```

## 4. Verify

```bash
# Instantiate without weights to check the class loads
python -c "from vcr_bench.models import create_model; m = create_model('mymodel', load_weights=False)"

# Run accuracy test on 8 videos
vcr-bench-test --model mymodel --dataset kinetics400 --num-videos 8 --device cuda
```

## Key base class methods

| Method | Must override? | Notes |
|--------|---------------|-------|
| `_build_stage_config_dicts()` | Yes | Returns `(loading_configs, preprocessing_configs)` dicts |
| `_load_checkpoint(path)` | If needed | Default handles `module.` prefix and `state_dict` nesting |
| `forward_preprocessed_with_grad(x)` | No | Default calls `self.model(x)` |
| `build_pipeline_config(stage)` | No | Built from `loading_configs`/`preprocessing_configs` |

The base class handles: temporal sampling, spatial cropping/resizing, normalisation, multi-clip aggregation, gradient computation with chunking, checkpoint download.
