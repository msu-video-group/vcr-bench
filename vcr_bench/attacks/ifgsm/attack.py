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

        with torch.no_grad():
            init_conf = model(x_ref, input_format=input_format, enable_grad=False)
            benign_label = int(torch.argmax(init_conf).item())
            if targeted:
                target_label = int(self._coerce_target(model, x_ref, input_format=input_format, y=y, targeted=True).item())
            else:
                target_label = benign_label
        ce = torch.nn.CrossEntropyLoss()
        adversarial_pred = torch.zeros_like(init_conf)
        adversarial_pred[target_label] = 1.0

        actual_iters = self.steps
        for step in range(self.steps):
            x_adv = x_adv.detach().requires_grad_(True)
            confidences = model(x_adv, input_format=input_format, enable_grad=True)
            confidences = confidences.unsqueeze(0)
            loss = ce(confidences, adversarial_pred.unsqueeze(0)) * (1 if targeted else -1)
            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv.detach() - self.alpha * torch.sign(grad)
            x_adv = self.project_linf(x_adv, x_ref)
            x_adv = self.clip_tensor(x_adv)

            with torch.no_grad():
                cur_conf = model(x_adv, input_format=input_format, enable_grad=False)
                final_label = int(torch.argmax(cur_conf).item())
                final_conf = float(cur_conf[final_label].item())
                if (targeted and final_label == target_label and final_conf >= self.target_conf) or (
                    (not targeted) and final_label != benign_label
                ):
                    actual_iters = step + 1
                    break

        self.last_result = {
            "iter_count": actual_iters,
            "target_label": int(target_label) if targeted else -1,
            "benign_label": int(benign_label),
        }
        return x_adv.detach()


ATTACK_CLASS = IFGSMAttack


def create(**kwargs) -> IFGSMAttack:
    return IFGSMAttack(**kwargs)
