from __future__ import annotations

import torch
from vcr_bench.attacks.base import BaseVideoAttack
from vcr_bench.models.base import BaseVideoClassifier


class IFGSMAttack(BaseVideoAttack):
    attack_name = "ifgsm"

    def __init__(
        self,
        eps: float = 8.0,
        alpha: float = 1.0,
        steps: int = 20,
        random_start: bool = False,
        clip_min: float | None = None,
        clip_max: float | None = None,
        sample_chunk_size: int | None = None,
        target_conf: float = 0.5,
    ) -> None:
        super().__init__(
            eps=eps,
            alpha=alpha,
            steps=steps,
            random_start=random_start,
            clip_min=clip_min,
            clip_max=clip_max,
            sample_chunk_size=sample_chunk_size,
        )
        self.target_conf = float(target_conf)

    def attack_sampled(
        self,
        model: BaseVideoClassifier,
        x: torch.Tensor,
        *,
        input_format: str = "NTHWC",
        y: torch.Tensor | int | None = None,
        targeted: bool = False,
    ) -> torch.Tensor:
        x_ref = self._ensure_float_attack_tensor(x.detach().clone())
        x_adv = self.clip_tensor(self._random_start(x_ref))
        target = self._coerce_target(model, x_ref, input_format=input_format, y=y, targeted=targeted)
        target_label = int(target.item())

        actual_iters = 0
        for step in range(self.steps):
            x_adv = x_adv.detach().requires_grad_(True)
            # Use _preprocess_for_attack so gradient flows through a single softmax
            # over raw model logits — avoids double-softmax for models like ActionCLIP
            # that already return probabilities from their forward().
            probs = self._preprocess_for_attack(model, x_adv, input_format=input_format, enable_grad=True)
            loss = self._loss_from_probs(probs, target)
            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv.detach() + self.alpha * self.gradient_direction(grad, targeted=targeted)
            x_adv = self.project_linf(x_adv, x_ref)
            x_adv = self.clip_tensor(x_adv)
            actual_iters = step + 1

            with torch.no_grad():
                cur_probs = self._preprocess_for_attack(model, x_adv, input_format=input_format, enable_grad=False)
                cur_probs_mean = cur_probs.detach().mean(dim=0)
                final_label = int(cur_probs_mean.argmax().item())
                if targeted:
                    if final_label == target_label and float(cur_probs_mean[target_label].item()) >= self.target_conf:
                        break
                else:
                    if final_label != target_label:
                        break

        self.last_result = {
            "iter_count": actual_iters,
            "target_label": target_label if targeted else -1,
        }
        return x_adv.detach()


ATTACK_CLASS = IFGSMAttack


def create(**kwargs) -> IFGSMAttack:
    return IFGSMAttack(**kwargs)
