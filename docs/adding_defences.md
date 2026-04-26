# Adding a Defence

## Directory layout

```
vcr_bench/defences/<name>/
    __init__.py    # empty or re-export
    defence.py     # defence implementation
configs/defences/<name>.json   # preset
```

The registry finds your defence by importing `vcr_bench.defences.<name>.defence` and looking for `create()` or `DEFENCE_CLASS`.

## 1. Implement the defence

```python
# vcr_bench/defences/mydefence/defence.py
from __future__ import annotations

import torch
from vcr_bench.defences.base import BaseVideoDefence


class MyDefence(BaseVideoDefence):

    # Set > 1 to enable voting: predict() is called this many times and
    # probabilities are averaged. Useful for randomised defences.
    voting_count: int = 1

    def __init__(
        self,
        param1: float = 1.0,
        param2: int = 3,
    ) -> None:
        self.param1 = param1
        self.param2 = param2

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, T, H, W, C] float32, values in [0, 255]
        # Must return a tensor of the same shape and range.

        # Example: per-frame spatial smoothing
        N, T, H, W, C = x.shape
        frames = x.reshape(N * T, H, W, C).permute(0, 3, 1, 2).float()  # [NT, C, H, W]

        # ... apply your defence to `frames` ...

        result = frames.permute(0, 2, 3, 1)              # back to [NT, H, W, C]
        return torch.clamp(result, 0, 255).reshape(N, T, H, W, C)


def create(**kwargs) -> MyDefence:
    return MyDefence(**kwargs)
```

### Tensor contract

- **Input `x`**: `[N, T, H, W, C]` float32, values in `[0, 255]`.
- **Output**: same shape, same value range `[0, 255]`.
- `transform` is called on the *sampled* clips, **before** model preprocessing. Think of it as a preprocessing wrapper, not a post-processing step.
- Preserve the shape exactly — the base class passes the result straight to `_preprocess_sampled`.

### How `install` / `uninstall` work

```python
defence.install(model)
# Now model._preprocess_sampled wraps transform() → original preprocess
# If voting_count > 1, model.predict() is also wrapped to average N calls

result = model.predict(x)   # transform runs transparently

defence.uninstall(model)    # original methods restored
```

You never need to call `install`/`uninstall` manually when using the CLI — the attack runner handles that.

### Lazy resource loading

For defences that load large models (like a diffusion checkpoint), load lazily in `transform` or a private `_load_model` method:

```python
def __init__(self, checkpoint_path: str = "...") -> None:
    self.checkpoint_path = checkpoint_path
    self._model = None   # load on first use

def _load_model(self, device: torch.device) -> None:
    if self._model is not None:
        return
    # ... load and cache model ...
    self._model = loaded_model

def transform(self, x: torch.Tensor) -> torch.Tensor:
    self._load_model(x.device)
    # use self._model
```

## 2. Create the preset config

```json
// configs/defences/mydefence.json
{
  "kind": "defence",
  "name": "mydefence",
  "default_variant": "default",
  "variants": {
    "default": {
      "factory_name": "mydefence",
      "params": {
        "param1": 1.0,
        "param2": 3
      }
    },
    "strong": {
      "factory_name": "mydefence",
      "params": {
        "param1": 2.0,
        "param2": 5
      }
    }
  }
}
```

## 3. Verify

```bash
# Non-adaptive: defence applied after attack
vcr-bench-attack \
  --model x3d --attack ifgsm \
  --defence mydefence \
  --dataset kinetics400 --num-videos 8

# Adaptive: defence applied before gradient computation (white-box adaptive attack)
vcr-bench-attack \
  --model x3d --attack ifgsm \
  --defence mydefence --adaptive \
  --dataset kinetics400 --num-videos 8

# With preset variant
vcr-bench-attack --model x3d --attack ifgsm \
  --defence-preset mydefence --defence-variant strong \
  --dataset kinetics400 --num-videos 8
```

### Non-adaptive vs adaptive

| Mode | What happens |
|------|-------------|
| Non-adaptive (default) | Attack runs on clean model; defence is applied only during evaluation |
| Adaptive (`--adaptive`) | Defence is installed on the model before the attack; gradients flow through `transform` |

For adaptive evaluation, make sure `transform` is differentiable (or at least approximately differentiable). Defences with hard randomness may need `voting_count > 1` to produce stable gradients.

## 4. Add to batch_attack dependency check (if needed)

`scripts/batch_attack.sh` line 299 auto-installs missing Python packages before running. If your defence needs a package that isn't in the container, add it there:

```bash
mods=['transformers','diffusers','yacs','IQA_pytorch','pywt', 'your_package']
```
