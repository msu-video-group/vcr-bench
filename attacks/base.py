from __future__ import annotations

from abc import ABC

import torch
import torch.nn.functional as F

from vg_videobench.models.base import BaseVideoClassifier


class BaseVideoAttack(ABC):
    attack_name = "unknown"

    def __init__(
        self,
        *,
        eps: float = 0.0,
        alpha: float = 0.0,
        steps: int = 1,
        random_start: bool = False,
        clip_min: float | None = None,
        clip_max: float | None = None,
        sample_chunk_size: int | None = None,
    ) -> None:
        if eps < 0 or alpha <= 0 or steps <= 0:
            raise ValueError("Attack requires eps >= 0, alpha > 0, steps > 0")
        self.eps = float(eps)
        self.alpha = float(alpha)
        self.steps = int(steps)
        self.random_start = bool(random_start)
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.sample_chunk_size = None if sample_chunk_size in (None, 0) else int(sample_chunk_size)
        self.last_result: dict[str, object] = {}

    def clip_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if self.clip_min is None and self.clip_max is None:
            return x
        return x.clamp(
            min=self.clip_min if self.clip_min is not None else None,
            max=self.clip_max if self.clip_max is not None else None,
        )

    def project_linf(self, x_adv: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
        if self.eps <= 0:
            return x_ref
        delta = torch.clamp(x_adv - x_ref, min=-self.eps, max=self.eps)
        return x_ref + delta

    def _ensure_float_attack_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return x.float() if not torch.is_floating_point(x) else x

    def _preprocess_for_attack(
        self,
        model: BaseVideoClassifier,
        x_sampled: torch.Tensor,
        *,
        input_format: str,
        enable_grad: bool,
    ) -> torch.Tensor:
        pre = model.preprocess_video(x_sampled, input_format=input_format)
        logits = model.forward_preprocessed_with_grad(pre, enable_grad=enable_grad)
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        return torch.softmax(logits, dim=-1)

    def _coerce_target(
        self,
        model: BaseVideoClassifier,
        x_sampled: torch.Tensor,
        *,
        input_format: str,
        y: torch.Tensor | int | None,
        targeted: bool = False,
    ) -> torch.Tensor:
        device = x_sampled.device
        if y is None:
            with torch.no_grad():
                probs = self._preprocess_for_attack(
                    model,
                    x_sampled,
                    input_format=input_format,
                    enable_grad=False,
                )
                reduce_fn = torch.argmin if targeted else torch.argmax
                target = reduce_fn(probs.mean(dim=0), dim=0).view(1)
            return target.to(device=device, dtype=torch.long)
        if isinstance(y, int):
            return torch.tensor([y], device=device, dtype=torch.long)
        y = y.to(device=device, dtype=torch.long).reshape(-1)
        if y.numel() != 1:
            raise ValueError("Attack expects a single scalar label/target")
        return y

    def _loss_from_probs(self, probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs_mean = probs.mean(dim=0, keepdim=True)
        target = target.to(device=probs_mean.device, dtype=torch.long)
        return F.nll_loss(torch.log(probs_mean.clamp_min(1e-12)), target)

    def _compute_attack_grad(
        self,
        model: BaseVideoClassifier,
        x_adv: torch.Tensor,
        *,
        input_format: str,
        target: torch.Tensor,
    ) -> torch.Tensor:
        chunk = self.sample_chunk_size
        # Default path: process all clips/views together.
        if not isinstance(chunk, int) or chunk <= 0 or x_adv.ndim != 5 or x_adv.shape[0] <= chunk:
            probs = self._preprocess_for_attack(
                model,
                x_adv,
                input_format=input_format,
                enable_grad=True,
            )
            loss = self._loss_from_probs(probs, target)
            return torch.autograd.grad(loss, x_adv)[0]

        # Optional memory-saving mode: accumulate per-clip-chunk gradients.
        grad_accum = torch.zeros_like(x_adv)
        chunk_count = 0
        for start in range(0, int(x_adv.shape[0]), chunk):
            cur = x_adv[start : start + chunk].detach().requires_grad_(True)
            probs = self._preprocess_for_attack(
                model,
                cur,
                input_format=input_format,
                enable_grad=True,
            )
            loss = self._loss_from_probs(probs, target)
            grad_cur = torch.autograd.grad(loss, cur)[0]
            grad_accum[start : start + chunk] = grad_cur.detach()
            chunk_count += 1
        if chunk_count > 1:
            grad_accum = grad_accum / float(chunk_count)
        return grad_accum

    def gradient_direction(self, grad: torch.Tensor, *, targeted: bool) -> torch.Tensor:
        sign = torch.sign(grad)
        return -sign if targeted else sign

    def _random_start(self, x_ref: torch.Tensor) -> torch.Tensor:
        if not self.random_start or self.eps <= 0:
            return x_ref.clone()
        noise = torch.empty_like(x_ref).uniform_(-self.eps, self.eps)
        return x_ref + noise

    def attack_sampled(
        self,
        model: BaseVideoClassifier,
        x: torch.Tensor,
        *,
        input_format: str = "NTHWC",
        y: torch.Tensor | int | None = None,
        targeted: bool = False,
    ) -> torch.Tensor:
        device = getattr(model, "device", x.device)
        x_ref = self._ensure_float_attack_tensor(x.detach().clone()).to(device)
        x_adv = self.clip_tensor(self._random_start(x_ref))
        target = self._coerce_target(model, x_ref, input_format=input_format, y=y, targeted=targeted)
        target_label = int(target.item())

        actual_iters = 0
        for step in range(self.steps):
            x_adv = x_adv.detach().requires_grad_(True)
            probs = self._preprocess_for_attack(
                model,
                x_adv,
                input_format=input_format,
                enable_grad=True,
            )
            # Early stopping: reuse probs already computed for gradient (no extra forward pass).
            # For untargeted: target holds original label; stop when prediction differs.
            # For targeted: stop when prediction matches target label.
            if step > 0:
                pred = int(probs.detach().mean(dim=0).argmax().item())
                succeeded = (pred == target_label) if targeted else (pred != target_label)
                if succeeded:
                    actual_iters = step
                    break
            loss = self._loss_from_probs(probs, target)
            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv.detach() + self.alpha * self.gradient_direction(grad, targeted=targeted)
            x_adv = self.project_linf(x_adv, x_ref)
            x_adv = self.clip_tensor(x_adv)
            actual_iters = step + 1

        self.last_result = {
            "iter_count": actual_iters,
            "target_label": target_label if targeted else -1,
        }
        return x_adv.detach()

    def prepare(
        self,
        *,
        model: BaseVideoClassifier,
        dataset: object,
        indices: list[int],
        seed: int,
        targeted: bool,
        pipeline_stage: str,
        verbose: bool,
    ) -> None:
        return None

    def supports_offline_prepare(self) -> bool:
        """Return True if this attack supports model-independent offline preparation."""
        return False

    def prepare_offline(
        self,
        *,
        dataset: object,
        indices: list[int],
        seed: int,
        targeted: bool,
        pipeline_stage: str,
        verbose: bool,
        device: str = "cpu",
        **kwargs,
    ) -> dict:
        """Run model-independent preprocessing once and cache results to disk.

        Called by ``vg_videobench.cli.prepare`` before attack runs.
        ``dataset[idx]`` returns a ``VideoSampleRef`` (no loading pipeline configured),
        giving access to ``.path`` for each video.

        Returns a summary dict printed by the prepare CLI.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support offline preparation. "
            "Override supports_offline_prepare() → True and implement prepare_offline()."
        )

    def attack(
        self,
        model: BaseVideoClassifier,
        x: torch.Tensor,
        *,
        input_format: str = "NTHWC",
        y: torch.Tensor | int | None = None,
        targeted: bool = False,
    ) -> torch.Tensor:
        return self.attack_sampled(
            model,
            x,
            input_format=input_format,
            y=y,
            targeted=targeted,
        )

    def __call__(
        self,
        model: BaseVideoClassifier,
        x: torch.Tensor,
        *,
        input_format: str = "NTHWC",
        y: torch.Tensor | int | None = None,
        targeted: bool = False,
    ) -> torch.Tensor:
        return self.attack(
            model,
            x,
            input_format=input_format,
            y=y,
            targeted=targeted,
        )
