# Adding an Attack

## Directory layout

```
vcr_bench/attacks/<name>/
    __init__.py    # empty or re-export
    attack.py      # attack implementation
configs/attacks/<name>.json   # preset
```

The registry finds your attack by importing `vcr_bench.attacks.<name>.attack` and looking for `create()` or `ATTACK_CLASS`.

## 1. Implement the attack

```python
# vcr_bench/attacks/myattack/attack.py
from __future__ import annotations

import torch

from ..base import BaseVideoAttack
from vcr_bench.models.base import BaseVideoClassifier


class MyAttack(BaseVideoAttack):
    attack_name = "myattack"

    def __init__(
        self,
        eps: float = 8.0,
        alpha: float = 1.0,
        steps: int = 20,
        random_start: bool = False,
        # add any attack-specific parameters here
        momentum: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(eps=eps, alpha=alpha, steps=steps, random_start=random_start, **kwargs)
        self.momentum = float(momentum)

    def attack_sampled(
        self,
        model: BaseVideoClassifier,
        x: torch.Tensor,
        *,
        input_format: str = "NTHWC",
        y: torch.Tensor | int | None = None,
        targeted: bool = False,
    ) -> torch.Tensor:
        # x: [num_clips, T, H, W, C] float32 in [0, 255]  — already sampled, NOT preprocessed
        # Returns adversarial x of the same shape and value range.

        device = getattr(model, "device", x.device)
        x_ref = self._ensure_float_attack_tensor(x.detach().clone()).to(device)
        x_adv = self.clip_tensor(self._random_start(x_ref))  # initialise (random or clean)

        # Resolve target label (uses clean prediction if y is None)
        target = self._coerce_target(
            model, x_ref, input_format=input_format, y=y, targeted=targeted
        )

        grad_accum = torch.zeros_like(x_ref)

        for step in range(self.steps):
            x_adv = x_adv.detach().requires_grad_(True)

            # _compute_attack_grad handles: preprocess → forward → loss → autograd
            grad = self._compute_attack_grad(
                model, x_adv, input_format=input_format, target=target
            )

            # Example: MI-FGSM style momentum accumulation
            grad_accum = self.momentum * grad_accum + grad / (grad.abs().mean() + 1e-8)

            direction = self.gradient_direction(grad_accum, targeted=targeted)
            x_adv = x_adv.detach() + self.alpha * direction
            x_adv = self.project_linf(x_adv, x_ref)   # clip to eps ball
            x_adv = self.clip_tensor(x_adv)            # clip to valid range

        return x_adv.detach()


ATTACK_CLASS = MyAttack


def create(**kwargs) -> MyAttack:
    return MyAttack(**kwargs)
```

### Key base class helpers

| Helper | What it does |
|--------|-------------|
| `self._ensure_float_attack_tensor(x)` | Casts to float32 |
| `self._random_start(x_ref)` | Returns `x_ref + U(-eps, eps)` if `random_start` else clone |
| `self._coerce_target(model, x, ...)` | Returns a scalar long tensor with the target label |
| `self._compute_attack_grad(model, x_adv, ...)` | Preprocess → forward → NLL loss → autograd gradient |
| `self.gradient_direction(grad, targeted=...)` | Returns `sign(grad)` or `-sign(grad)` |
| `self.project_linf(x_adv, x_ref)` | Clips perturbation to `eps` L∞ ball around `x_ref` |
| `self.clip_tensor(x)` | Clips to `[clip_min, clip_max]` (default `[0, 255]`) |
| `self._preprocess_for_attack(model, x, ...)` | Preprocess + forward, returns softmax probs |
| `self._loss_from_probs(probs, target)` | NLL loss averaged over clips |

The base class `attack_sampled` is a working IFGSM implementation — subclass only if you need custom gradient logic. For most gradient-based attacks, only the loop body changes.

### Query-based attacks

For black-box attacks that don't use gradients, subclass `BaseVideoAttack` and override `attack_sampled` entirely. The `_preprocess_for_attack` helper still works as a model call:

```python
def attack_sampled(self, model, x, *, input_format="NTHWC", y=None, targeted=False):
    ...
    probs = self._preprocess_for_attack(model, x_adv, input_format=input_format, enable_grad=False)
    pred = probs.mean(0).argmax()
    ...
```

### Offline preparation

Some attacks (e.g. StyleFool) need a pre-computation step that runs once before the attack loop. Implement `prepare_offline`:

```python
def supports_offline_prepare(self) -> bool:
    return True

def prepare_offline(self, *, dataset, indices, seed, targeted, pipeline_stage, verbose, device="cpu", **kwargs) -> dict:
    # dataset[i] returns VideoSampleRef (bare, no loading pipeline)
    # Save results to disk; attack.py picks them up in attack_sampled via self.cache_path
    ...
    return {"prepared": len(indices)}
```

Then run `vcr-bench-prepare --attack myattack --dataset kinetics400` before attack runs.

## 2. Create the preset config

```json
// configs/attacks/myattack.json
{
  "kind": "attack",
  "name": "myattack",
  "default_variant": "default",
  "variants": {
    "default": {
      "factory_name": "myattack",
      "params": {
        "eps": 8.0,
        "alpha": 1.0,
        "steps": 20,
        "random_start": false,
        "momentum": 0.9
      }
    },
    "strong": {
      "inherits": "default",
      "params": {
        "steps": 50,
        "eps": 16.0
      }
    }
  }
}
```

`"inherits"` merges the named variant's params as a base, then the current `"params"` override on top.

## 3. Expose CLI flags (optional)

If your attack has extra hyperparameters you want configurable from the CLI without a preset, add them to `vcr_bench/cli/attack.py` in the parser section alongside the existing per-attack blocks.

## 4. Verify

```bash
# Quick test: 8 videos, lite pipeline
vcr-bench-attack \
  --model x3d --attack myattack \
  --dataset kinetics400 --num-videos 8 \
  --eps 8 --alpha 1 --iter 5 \
  --lite-attack

# With preset
vcr-bench-attack --attack-preset myattack --attack-variant strong \
  --model x3d --dataset kinetics400 --num-videos 8
```

## Tensor contract

- **Input `x`**: `[num_clips, T, H, W, C]` float32, values in `[0, 255]`. This is the *sampled* tensor before model preprocessing.
- **Output**: same shape and value range — the adversarial version.
- The attack must not call `model.preprocess_video()` directly; `_compute_attack_grad` handles that.
- The attack is responsible for keeping the adversarial example within `[clip_min, clip_max]` (default `[0, 255]`) and within the eps ball. Use `project_linf` and `clip_tensor`.
