from __future__ import annotations

import torch

from vcr_bench.attacks.base import BaseVideoAttack
from vcr_bench.attacks.utils import cross_entropy_from_probs, freeze_model_weights, predict_probs, safe_cuda_empty_cache, safe_cuda_reset_peak_memory_stats
from vcr_bench.models.base import BaseVideoClassifier


class MIFGSMAttack(BaseVideoAttack):
    attack_name = "mifgsm"

    def __init__(
        self,
        eps: float = 8.0,
        alpha: float = 1.0,
        steps: int = 20,
        target_conf: float = 0.5,
        sample_chunk_size: int | None = None,
    ) -> None:
        super().__init__(
            eps=eps,
            alpha=alpha,
            steps=steps,
            random_start=False,
            clip_min=0.0,
            clip_max=255.0,
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
        freeze_model_weights(model)
        safe_cuda_reset_peak_memory_stats(getattr(model, "device", x.device))
        init_tensor = self._ensure_float_attack_tensor(x.detach().clone()).to(getattr(model, "device", x.device))
        pert = torch.zeros_like(init_tensor, requires_grad=True)
        g_prev = torch.zeros_like(init_tensor)
        nu = 1.0

        init_probs = predict_probs(model, init_tensor, input_format=input_format, enable_grad=False)
        benign_label = int(torch.argmax(init_probs).item())
        benign_conf = float(init_probs[benign_label].item())
        target_label = int(self._coerce_target(model, init_tensor, input_format=input_format, y=y, targeted=targeted).item()) if targeted else -1

        final_label = benign_label
        final_conf = benign_conf
        iteration = 0
        for iteration in range(1, self.steps + 1):
            attacked = (init_tensor + pert).clamp(0.0, 255.0)
            probs = predict_probs(model, attacked, input_format=input_format, enable_grad=True)
            label = target_label if targeted else benign_label
            loss = cross_entropy_from_probs(probs, label) * (1 if targeted else -1)
            loss.backward()

            grad = pert.grad + g_prev * nu
            g_prev = grad.detach()
            pert.data -= self.alpha * torch.sign(grad)
            pert.data.clamp_(-self.eps, self.eps)
            pert.grad.zero_()

            attacked = (init_tensor + pert).clamp(0.0, 255.0)
            cur_probs = predict_probs(model, attacked, input_format=input_format, enable_grad=False)
            final_label = int(torch.argmax(cur_probs).item())
            final_conf = float(cur_probs[final_label].item())
            if (targeted and final_label == target_label and final_conf >= self.target_conf) or (
                (not targeted) and final_label != benign_label
            ):
                break
            safe_cuda_empty_cache()

        self.last_result = {
            "iter_count": int(iteration),
            "target_label": int(target_label),
            "benign_label": int(benign_label),
            "benign_conf": float(benign_conf),
            "final_label": int(final_label),
            "final_conf": float(final_conf),
        }
        return (init_tensor + pert).clamp(0.0, 255.0).detach().cpu()


ATTACK_CLASS = MIFGSMAttack


def create(**kwargs) -> MIFGSMAttack:
    return MIFGSMAttack(**kwargs)
