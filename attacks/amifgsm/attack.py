from __future__ import annotations

import torch

from vg_videobench.attacks.mifgsm.attack import MIFGSMAttack


class AMIFGSMAttack(MIFGSMAttack):
    attack_name = "amifgsm"

    def __init__(
        self,
        eps: float = 8.0,
        alpha: float = 1.0,
        steps: int = 20,
        target_conf: float = 0.5,
        sample_chunk_size: int | None = None,
    ) -> None:
        super().__init__(eps=eps, alpha=alpha, steps=steps, target_conf=target_conf, sample_chunk_size=sample_chunk_size)

    def _adaptive_eps(self, x: torch.Tensor, device: torch.device) -> float:
        try:
            import pyiqa  # type: ignore

            niqe = pyiqa.create_metric("niqe", device=str(device))
            with torch.no_grad():
                vid = x.permute(0, 3, 1, 2) / 255.0
                score = float(niqe(vid).mean().item())
                if score > 0:
                    return 255.0 / (8.0 * score)
        except Exception:
            pass
        return float(self.eps)

    def attack_sampled(self, model, x, *, input_format="NTHWC", y=None, targeted=False):
        old_eps = self.eps
        device = torch.device(getattr(model, "device", x.device))
        self.eps = self._adaptive_eps(self._ensure_float_attack_tensor(x.detach().clone()).to(device), device)
        try:
            return super().attack_sampled(model, x, input_format=input_format, y=y, targeted=targeted)
        finally:
            self.eps = old_eps


ATTACK_CLASS = AMIFGSMAttack


def create(**kwargs) -> AMIFGSMAttack:
    return AMIFGSMAttack(**kwargs)
