from __future__ import annotations

import torch

from vcr_bench.attacks.query_base import BaseQueryVideoAttack
from vcr_bench.attacks.utils import flatten_temporal_video, restore_temporal_video
from vcr_bench.models.base import BaseVideoClassifier
from vcr_bench.types import PredictionBundle


class GradEstAttack(BaseQueryVideoAttack):
    """NES-based black-box attack with tiled spatial perturbation."""
    attack_name = "gradest"

    def __init__(
        self,
        eps: float = 64.0,
        alpha: float | None = None,        # per-step size; auto = eps/steps
        steps: int = 200,
        sigma: float = 0.255,
        sigma_decay: float = 0.999,        # adaptive sigma: decay per step
        n_samples: int = 10,
        n_samples_late: int = 5,           # fewer samples in second half of steps
        spatial_size: int | None = 64,     # None = full frame (too noisy unless n_samples is large)
        query_budget: int = 100_000,
        momentum: float = 0.9,
        boost_second: float = 0.0,         # weight on -CE(second_label); 0 = disabled
        random_start: bool = False,
    ) -> None:
        super().__init__(eps=eps, alpha=alpha if alpha is not None else eps / steps, steps=steps, random_start=random_start)
        self.sigma = float(sigma)
        self.sigma_decay = float(sigma_decay)
        self.n_samples = int(n_samples)
        self.n_samples_late = int(n_samples_late)
        self.spatial_size = None if spatial_size is None else int(spatial_size)
        self.query_budget = int(query_budget)
        self.momentum = float(momentum)
        self.boost_second = float(boost_second)
        self.verbose = False

    def _dbg(self, msg: str) -> None:
        if self.verbose:
            print(f"[gradest] {msg}", flush=True)

    def prepare(self, *, model, dataset, indices, seed, targeted, pipeline_stage, verbose) -> None:
        del model, dataset, indices, seed, targeted, pipeline_stage
        self.verbose = bool(verbose)

    def _apply_tile(
        self,
        x_flat: torch.Tensor,
        pert2d: torch.Tensor,
        *,
        lo: float,
        hi: float,
    ) -> torch.Tensor:
        T, H, W, C = x_flat.shape
        N = pert2d.shape[0]
        rh = H // N + 1
        rw = W // N + 1
        tiled = pert2d.repeat(T, rh, rw, 1)[:T, :H, :W, :]
        return torch.clamp(x_flat + tiled, lo, hi)

    def attack_query(
        self,
        model: BaseVideoClassifier,
        x: torch.Tensor,
        *,
        query_label,
        input_format: str = "NTHWC",
        y=None,
        targeted: bool = False,
        video_path: str | None = None,
    ) -> torch.Tensor:
        del query_label

        x_ref = self._ensure_float_attack_tensor(x.detach().clone())
        x_flat, restore_shape = flatten_temporal_video(x_ref)
        _, H, W, C = x_flat.shape
        device = x_flat.device
        lo, hi = 0.0, 255.0
        eps = float(self.eps)
        N = min(H, W) if self.spatial_size is None else min(int(self.spatial_size), H, W)
        query_count = 0
        ce = torch.nn.CrossEntropyLoss()
        sigma = self.sigma  # mutable per-run copy for decay

        def query_bundle(flat: torch.Tensor) -> PredictionBundle:
            nonlocal query_count
            query_count += 1
            restored = restore_temporal_video(flat, restore_shape)
            with torch.no_grad():
                bundle = model(restored, input_format=input_format, enable_grad=False, return_full=True)
            if not isinstance(bundle, PredictionBundle):
                raise TypeError(f"Expected PredictionBundle, got {type(bundle)}")
            return bundle

        benign_bundle = query_bundle(x_flat)
        benign_probs = benign_bundle.probs.reshape(-1)
        benign_label = int(benign_probs.argmax().item())
        second_label_init = int(torch.topk(benign_probs, 2).indices[1].item())

        if targeted:
            if y is None:
                target_label = int(benign_probs.argmin().item())
            elif isinstance(y, int):
                target_label = y
            else:
                target_label = int(y.item())
            sign = -1.0
        else:
            target_label = benign_label
            sign = 1.0

        label_tensor = torch.tensor([target_label], device=device, dtype=torch.long)
        second_label_tensor = torch.tensor([second_label_init], device=device, dtype=torch.long)

        lower = torch.clamp(x_flat - eps, lo, hi)
        upper = torch.clamp(x_flat + eps, lo, hi)

        x_adv_flat = x_flat.clone()
        velocity = torch.zeros(N, N, C, device=device)
        best_adv_flat = x_adv_flat.clone()
        iters_done = 0
        final_label = benign_label
        found = False

        self._dbg(
            f"start shape={tuple(x_flat.shape)} eps={eps:.1f} alpha={self.alpha:.2f} N={N} steps={self.steps} "
            f"sigma={sigma:.4f} sigma_decay={self.sigma_decay} "
            f"n_samples={self.n_samples}/{self.n_samples_late} "
            f"momentum={self.momentum} boost_second={self.boost_second} "
            f"benign={benign_label} second_init={second_label_init}"
        )

        for step_i in range(self.steps):
            iters_done = step_i + 1
            if query_count >= self.query_budget:
                break

            # fewer samples in second half of steps
            n_samples = self.n_samples if step_i < self.steps // 2 else self.n_samples_late

            # NES gradient estimation (momentum on raw gradient, before normalization)
            grad = torch.zeros(N, N, C, device=device)
            for _ in range(n_samples):
                noise = torch.randn(N, N, C, device=device)
                plus_bundle = query_bundle(self._apply_tile(x_adv_flat, sigma * noise, lo=lo, hi=hi))
                minus_bundle = query_bundle(self._apply_tile(x_adv_flat, -sigma * noise, lo=lo, hi=hi))
                def _loss(logits: torch.Tensor) -> float:
                    l = ce(logits.reshape(1, -1), label_tensor)
                    if self.boost_second > 0.0:
                        l = l - self.boost_second * ce(logits.reshape(1, -1), second_label_tensor)
                    return float(l.item())
                loss_plus = _loss(plus_bundle.logits)
                loss_minus = _loss(minus_bundle.logits)
                grad += (loss_plus - loss_minus) * noise
            grad = grad / (2.0 * n_samples * sigma)

            velocity = self.momentum * velocity + (1.0 - self.momentum) * grad
            if velocity.abs().max() < 1e-12:
                sigma *= self.sigma_decay
                continue
            direction = torch.sign(velocity)

            x_adv_flat = self._apply_tile(x_adv_flat, sign * self.alpha * direction, lo=lo, hi=hi)
            x_adv_flat = torch.max(torch.min(x_adv_flat, upper), lower)

            # adaptive sigma decay
            sigma *= self.sigma_decay

            if query_count < self.query_budget:
                cur_bundle = query_bundle(x_adv_flat)
                cur_probs = cur_bundle.probs.reshape(-1)
                final_label = int(cur_probs.argmax().item())
                best_adv_flat = x_adv_flat.clone()
                delta_linf = float((x_adv_flat - x_flat).abs().max().item())
                benign_conf = float(cur_probs[benign_label].item())
                top2 = torch.topk(cur_probs, 2)
                second_label = int(top2.indices[1].item()) if final_label == benign_label else int(top2.indices[0].item())
                second_conf = float(cur_probs[second_label].item())
                if self.boost_second > 0.0:
                    second_label_tensor = torch.tensor([second_label], device=device, dtype=torch.long)
                self._dbg(
                    f"iter={step_i + 1}/{self.steps} queries={query_count}/{self.query_budget} "
                    f"label={final_label} benign_conf={benign_conf:.4f} second={second_label}:{second_conf:.4f} "
                    f"sigma={sigma:.4f} delta_linf={delta_linf:.4f}/{eps:.4f}"
                )
                if (not targeted and final_label != benign_label) or (targeted and final_label == target_label):
                    found = True
                    break

        self.last_result = {
            "iter_count": iters_done,
            "query_count": query_count,
            "target_label": target_label if targeted else -1,
            "benign_label": benign_label,
            "final_label": final_label,
            "effective_eps": eps,
        }
        self._dbg(f"done iters={iters_done} queries={query_count} benign={benign_label} final={final_label} found={found}")
        return restore_temporal_video(best_adv_flat, restore_shape).detach()


ATTACK_CLASS = GradEstAttack


def create(**kwargs) -> GradEstAttack:
    return GradEstAttack(**kwargs)
