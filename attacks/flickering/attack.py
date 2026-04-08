from __future__ import annotations

import torch

from vg_videobench.attacks.base import BaseVideoAttack
from vg_videobench.attacks.utils import flatten_temporal_video, freeze_model_weights, predict_probs, restore_temporal_video


def _flickering_regularization(perturbation: torch.Tensor, beta_1: float = 0.5) -> torch.Tensor:
    norm_reg = torch.mean(perturbation ** 2) + 1e-12
    pert_roll_right = torch.roll(perturbation, shifts=1, dims=0)
    pert_roll_left = torch.roll(perturbation, shifts=-1, dims=0)
    diff_norm_reg = torch.mean((perturbation - pert_roll_right) ** 2) + 1e-12
    lap_norm_reg = torch.mean((-2 * perturbation + pert_roll_right + pert_roll_left) ** 2) + 1e-12
    return beta_1 * norm_reg + (1.0 - beta_1) * (diff_norm_reg + lap_norm_reg)


def _adversarial_loss(probs: torch.Tensor, label: int, targeted: bool) -> torch.Tensor:
    if targeted:
        return -torch.log(probs[label] + 1e-6)
    return -torch.log(1.0 - probs[label] + 1e-6)


class FlickeringAttack(BaseVideoAttack):
    attack_name = "flickering"

    def __init__(
        self,
        eps: float = 100.0,
        steps: int = 200,
        beta_1: float = 0.5,
        lambda_reg: float = 0.1,
        lr: float = 0.1,
        target_conf: float = 0.5,
        sample_chunk_size: int | None = None,
    ) -> None:
        super().__init__(eps=eps, alpha=1.0, steps=steps, clip_min=0.0, clip_max=255.0, sample_chunk_size=sample_chunk_size)
        self.beta_1 = float(beta_1)
        self.lambda_reg = float(lambda_reg)
        self.lr = float(lr)
        self.target_conf = float(target_conf)

    def attack_sampled(self, model, x, *, input_format="NTHWC", y=None, targeted=False):
        freeze_model_weights(model)
        source_tensor = x.detach().clone().float().to(getattr(model, "device", x.device))
        init_tensor, restore_shape = flatten_temporal_video(source_tensor)
        if init_tensor.ndim != 4 or init_tensor.shape[-1] not in (1, 3):
            raise RuntimeError(f"Flickering attack expects THWC input, got {tuple(init_tensor.shape)}")
        t, _, _, c = init_tensor.shape
        pert = torch.nn.Parameter(torch.empty((t, 1, 1, c), device=init_tensor.device).uniform_(-1e-6, 1e-6))
        optimizer = torch.optim.Adam([pert], lr=self.lr)

        init_probs = predict_probs(model, source_tensor, input_format=input_format, enable_grad=False)
        benign_label = int(torch.argmax(init_probs).item())
        benign_conf = float(init_probs[benign_label].item())
        target_label = int(self._coerce_target(model, init_tensor, input_format=input_format, y=y, targeted=targeted).item()) if targeted else -1

        final_label = benign_label
        final_conf = benign_conf
        iteration = 0
        for iteration in range(1, self.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            attacked = restore_temporal_video((init_tensor + pert).clamp(0.0, 255.0), restore_shape)
            probs = predict_probs(model, attacked, input_format=input_format, enable_grad=True)
            adv_loss = _adversarial_loss(probs, target_label if targeted else benign_label, targeted=targeted)
            reg_loss = _flickering_regularization(pert / 255.0, beta_1=self.beta_1)
            loss = adv_loss + self.lambda_reg * reg_loss
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                pert.clamp_(-self.eps, self.eps)

            attacked = restore_temporal_video((init_tensor + pert).clamp(0.0, 255.0), restore_shape)
            cur_probs = predict_probs(model, attacked, input_format=input_format, enable_grad=False)
            final_label = int(torch.argmax(cur_probs).item())
            final_conf = float(cur_probs[final_label].item())
            if (targeted and final_label == target_label and final_conf >= self.target_conf) or ((not targeted) and final_label != benign_label):
                break

        self.last_result = {"iter_count": int(iteration), "target_label": int(target_label)}
        return restore_temporal_video((init_tensor + pert).clamp(0.0, 255.0), restore_shape).detach().cpu()


ATTACK_CLASS = FlickeringAttack


def create(**kwargs) -> FlickeringAttack:
    return FlickeringAttack(**kwargs)
